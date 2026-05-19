"""
home_position.py — Manual PTZ home position management.

Why this exists
---------------
The original tracking script (track.py) calls ONVIF SetHomePosition / GotoHomePosition,
which only stores ONE home and only on the camera. For field deployment we need:

  1. A *manually set* home anchored to a physical landmark (runway threshold)
  2. Coordinates that survive camera reboots and can be version-controlled
  3. The ability to read the current PTZ position so we know what to put in the config

This module is intentionally free of network calls and OpenCV / torch imports
so it can be unit-tested in isolation. The track.py layer wires it to SOAP.

ONVIF range convention
----------------------
For most PTZ-capable cameras, AbsoluteMove uses normalized coordinates:
  pan:  -1.0 (full left)  → +1.0 (full right)
  tilt: -1.0 (full down)  → +1.0 (full up)
  zoom:  0.0 (wide)       → +1.0 (telephoto)

We allow zoom in [-1.0, 1.0] for compatibility with cameras whose zoom is
also normalized to that range — the validator can be tightened later if
your specific model rejects negative zoom.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Tuple


# ── Errors ───────────────────────────────────────────────────────────────────
class HomeConfigError(Exception):
    """Raised for any home-position related failure: bad file, bad XML,
    out-of-range value, SOAP fault. One exception type → one catch site in
    track.py keeps the calling code clean."""


# ── Validation ───────────────────────────────────────────────────────────────
_RANGE_MIN = -1.0
_RANGE_MAX = 1.0


def _validate_axis(name: str, value) -> float:
    """Coerce to float and check ONVIF range. Pure function — no side effects."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise HomeConfigError(
            f"{name} must be numeric, got {value!r} ({type(value).__name__})"
        )
    if not (_RANGE_MIN <= v <= _RANGE_MAX):
        raise HomeConfigError(
            f"{name} value {v} is out of range "
            f"[{_RANGE_MIN}, {_RANGE_MAX}]"
        )
    return v


# ── Value object ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HomePosition:
    """
    Immutable PTZ coordinate triple. Validated at construction time.

    `is_default=True` means we fell back to (0,0,0) because no config file
    was found — track.py uses this to log a warning so the user notices.
    """
    pan: float
    tilt: float
    zoom: float
    is_default: bool = field(default=False)

    def __post_init__(self):
        # frozen=True means we use object.__setattr__ to coerce
        object.__setattr__(self, "pan",  _validate_axis("pan",  self.pan))
        object.__setattr__(self, "tilt", _validate_axis("tilt", self.tilt))
        object.__setattr__(self, "zoom", _validate_axis("zoom", self.zoom))

    # ── Loading ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str) -> "HomePosition":
        """Strict load — raise on any problem (missing file, bad JSON, bad values)."""
        if not os.path.isfile(path):
            raise HomeConfigError(f"Home config file not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise HomeConfigError(f"Cannot parse home config {path}: invalid JSON — {e}")
        except OSError as e:
            raise HomeConfigError(f"Cannot read home config {path}: {e}")

        if not isinstance(data, dict):
            raise HomeConfigError(
                f"Home config {path} must be a JSON object, got {type(data).__name__}"
            )

        for key in ("pan", "tilt", "zoom"):
            if key not in data:
                raise HomeConfigError(f"Home config {path} is missing required key: {key}")

        return cls(pan=data["pan"], tilt=data["tilt"], zoom=data["zoom"], is_default=False)

    @classmethod
    def load_or_default(cls, path: str) -> "HomePosition":
        """
        Lenient load — fall back to (0,0,0) only if file is missing.
        Malformed / invalid files still raise: a typo in the config is a bug,
        not a reason to silently fly to the wrong position.
        """
        if not os.path.isfile(path):
            return cls(pan=0.0, tilt=0.0, zoom=0.0, is_default=True)
        return cls.load(path)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.pan, self.tilt, self.zoom)

    # ── Saving ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """
        Write this position to a JSON file atomically.

        We write to `path + ".tmp"` and then os.replace() it onto path,
        so a power loss or kill-signal in the middle of the write cannot
        leave a half-written file (which would brick the config).
        """
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {"pan": self.pan, "tilt": self.tilt, "zoom": self.zoom},
                f, indent=2,
            )
        os.replace(tmp, path)


# ── Module-level helpers ─────────────────────────────────────────────────────
def backup_then_save(home: HomePosition, path: str):
    """
    Save `home` to `path`. If `path` already exists, rename the old file to
    `path + ".bak"` first (overwriting any previous .bak). Returns the
    backup path if one was created, otherwise None.

    Why module-level rather than a method: it operates on a file system path,
    not on `self` — the prior file may have very different values from `home`.
    """
    backup_path = path + ".bak"
    created_backup = None
    if os.path.isfile(path):
        os.replace(path, backup_path)   # atomic rename; overwrites old .bak
        created_backup = backup_path
    home.save(path)
    return created_backup


# ── ONVIF XML builders ───────────────────────────────────────────────────────
# Kept here (not in track.py) so they can be tested with zero network deps.

_NS_PTZ20  = "http://www.onvif.org/ver20/ptz/wsdl"
_NS_SCHEMA = "http://www.onvif.org/ver10/schema"


def build_absolute_move_xml(token: str, pan: float, tilt: float, zoom: float) -> str:
    """
    Build the body XML for an ONVIF AbsoluteMove request.

    We return only the AbsoluteMove element (no SOAP envelope) — track.py's
    existing `soap()` helper wraps things in the envelope, and keeping the
    bodies envelope-free lets us reuse that helper.
    """
    if not token:
        raise ValueError("ProfileToken must be a non-empty string")
    pan  = _check_range("pan",  pan)
    tilt = _check_range("tilt", tilt)
    zoom = _check_range("zoom", zoom)

    return (
        f'<AbsoluteMove xmlns="{_NS_PTZ20}">'
        f'<ProfileToken>{token}</ProfileToken>'
        f'<Position>'
        f'<PanTilt xmlns="{_NS_SCHEMA}" x="{pan}" y="{tilt}"/>'
        f'<Zoom xmlns="{_NS_SCHEMA}" x="{zoom}"/>'
        f'</Position>'
        f'</AbsoluteMove>'
    )


def build_get_status_xml(token: str) -> str:
    """Body XML for an ONVIF GetStatus request — returns current PTZ position."""
    if not token:
        raise ValueError("ProfileToken must be a non-empty string")
    return (
        f'<GetStatus xmlns="{_NS_PTZ20}">'
        f'<ProfileToken>{token}</ProfileToken>'
        f'</GetStatus>'
    )


def _check_range(name: str, value: float) -> float:
    """Same as _validate_axis but raises ValueError — appropriate for XML builders
    (ValueError fits "bad argument", HomeConfigError fits "bad config file")."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    if not (_RANGE_MIN <= v <= _RANGE_MAX):
        raise ValueError(f"{name} value {v} out of range [{_RANGE_MIN}, {_RANGE_MAX}]")
    return v


# ── GetStatus response parser ────────────────────────────────────────────────
def parse_ptz_status(xml_text: str) -> Tuple[float, float, float]:
    """
    Extract (pan, tilt, zoom) from a SOAP GetStatusResponse.
    Raises HomeConfigError if the response is malformed or contains a fault.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise HomeConfigError(f"Cannot parse GetStatus response as XML: {e}")

    # Detect SOAP fault — search by local name to be namespace-tolerant
    for el in root.iter():
        local = el.tag.split("}")[-1]
        if local == "Fault":
            # Try to extract a human-readable reason
            reason = None
            for e2 in el.iter():
                if e2.tag.split("}")[-1] == "Text" and e2.text:
                    reason = e2.text.strip()
                    break
            raise HomeConfigError(f"SOAP Fault in GetStatus response: {reason or 'unknown'}")

    # Find PanTilt and Zoom — by local name (tolerant of namespace prefix variations)
    pan_tilt_el = None
    zoom_el     = None
    for el in root.iter():
        local = el.tag.split("}")[-1]
        if local == "PanTilt" and "x" in el.attrib and "y" in el.attrib and pan_tilt_el is None:
            # First PanTilt with x,y attrs — there's only one Position/PanTilt
            pan_tilt_el = el
        elif local == "Zoom" and "x" in el.attrib and zoom_el is None:
            zoom_el = el

    if pan_tilt_el is None:
        raise HomeConfigError("GetStatus response missing Position/PanTilt")
    if zoom_el is None:
        raise HomeConfigError("GetStatus response missing Position/Zoom")

    try:
        pan  = float(pan_tilt_el.attrib["x"])
        tilt = float(pan_tilt_el.attrib["y"])
        zoom = float(zoom_el.attrib["x"])
    except (KeyError, ValueError) as e:
        raise HomeConfigError(f"GetStatus response has malformed coordinates: {e}")

    return pan, tilt, zoom
