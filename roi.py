"""
roi.py — Region of Interest for detection filtering.

Why fractions instead of pixels
-------------------------------
The RTSP stream resolution can change (some cameras renegotiate after a
network blip, some return SD on first connect then HD a second later).
Storing the ROI as fractions of frame size (0.0 .. 1.0) means a saved
ROI keeps making sense regardless of the actual frame dimensions.

Boxes (YOLO output) are in pixels — we convert at the boundary.

Membership test: a box is "inside" the ROI if its CENTER point is inside.
Edge-based tests cause flicker when objects touch the boundary; center
tests are stable and match what humans expect ("the object is in the box").
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


class ROIError(Exception):
    """Any ROI-related failure: bad config, bad geometry, bad file."""


# A YOLO box: (x1, y1, x2, y2, conf) — all pixels except conf
Box = Tuple[int, int, int, int, float]


def _validate_fraction(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ROIError(f"{name} must be numeric, got {value!r}")
    if not (0.0 <= v <= 1.0):
        raise ROIError(f"{name}={v} out of range [0.0, 1.0]")
    return v


@dataclass(frozen=True)
class ROI:
    """Rectangle stored as fractions of frame width/height."""
    x1: float
    y1: float
    x2: float
    y2: float

    # Anything smaller than this fraction in either dimension is treated as
    # an accidental click rather than a real ROI selection.
    MIN_SIZE = 0.01

    def __post_init__(self):
        # Validate each axis is in [0,1]
        for name, val in (("x1", self.x1), ("y1", self.y1),
                          ("x2", self.x2), ("y2", self.y2)):
            object.__setattr__(self, name, _validate_fraction(name, val))

        # Ordering / size check
        if self.x2 <= self.x1:
            raise ROIError(f"invalid ROI: x2 ({self.x2}) must be > x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ROIError(f"invalid ROI: y2 ({self.y2}) must be > y1 ({self.y1})")

    # ── Geometry ─────────────────────────────────────────────────────────
    def area_fraction(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )

    @classmethod
    def from_pixels(cls, x1: int, y1: int, x2: int, y2: int,
                    width: int, height: int) -> "ROI":
        """
        Build an ROI from a pixel rectangle. Tolerates:
          - inverted corners (drag from bottom-right to top-left)
          - out-of-bounds coordinates (mouse left the frame during drag)
        """
        # Normalize corner order
        x_lo, x_hi = min(x1, x2), max(x1, x2)
        y_lo, y_hi = min(y1, y2), max(y1, y2)
        # Clamp to frame
        x_lo = max(0, min(width,  x_lo))
        x_hi = max(0, min(width,  x_hi))
        y_lo = max(0, min(height, y_lo))
        y_hi = max(0, min(height, y_hi))
        # Reject degenerate boxes (accidental clicks)
        if (x_hi - x_lo) < cls.MIN_SIZE * width or (y_hi - y_lo) < cls.MIN_SIZE * height:
            raise ROIError(
                f"ROI too small ({x_hi - x_lo}x{y_hi - y_lo}px); "
                f"need at least {cls.MIN_SIZE * 100:.0f}% of frame in each dimension"
            )
        return cls(x_lo / width, y_lo / height, x_hi / width, y_hi / height)

    # ── Membership ───────────────────────────────────────────────────────
    def contains_box(self, box: Box, width: int, height: int) -> bool:
        """True if the box's center is inside this ROI."""
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        px1, py1, px2, py2 = self.to_pixels(width, height)
        return (px1 <= cx <= px2) and (py1 <= cy <= py2)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        # Atomic write — never leave a half-written file if interrupted.
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"x1": self.x1, "y1": self.y1,
                       "x2": self.x2, "y2": self.y2}, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "ROI":
        if not os.path.isfile(path):
            raise ROIError(f"ROI file not found: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ROIError(f"cannot parse ROI file {path}: invalid JSON — {e}")

        if not isinstance(data, dict):
            raise ROIError(f"ROI file {path} must be a JSON object")

        for key in ("x1", "y1", "x2", "y2"):
            if key not in data:
                raise ROIError(f"ROI file {path} missing required key: {key}")

        return cls(x1=data["x1"], y1=data["y1"], x2=data["x2"], y2=data["y2"])

    @classmethod
    def load_or_none(cls, path: str) -> Optional["ROI"]:
        """Missing file → None (no ROI set yet). Malformed file still raises."""
        if not os.path.isfile(path):
            return None
        return cls.load(path)


# ── Top-level filter helper ─────────────────────────────────────────────────
def filter_boxes_in_roi(boxes: List[Box], roi: Optional[ROI],
                        width: int, height: int) -> List[Box]:
    """Apply ROI filter to a list of YOLO boxes.
    If `roi` is None, return the boxes unchanged (no filtering)."""
    if roi is None:
        return boxes
    return [b for b in boxes if roi.contains_box(b, width, height)]
