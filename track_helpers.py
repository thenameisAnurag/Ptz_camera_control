"""
track_helpers.py — Thin wrappers that turn home_position XML builders into
network calls. Kept separate from track.py so they can be tested with a
mocked `soap` callable.

Each function takes `soap_fn` as the first argument so tests can inject a
mock. In production, track.py passes its existing `soap` function.
"""

from typing import Callable, Tuple

from home_position import (
    HomePosition,
    HomeConfigError,
    build_absolute_move_xml,
    build_get_status_xml,
    parse_ptz_status,
)


SoapFn = Callable[[str], str]


def _check_fault(response: str) -> None:
    """Raise HomeConfigError if the SOAP response contains a Fault element."""
    # Cheap substring check first to avoid parsing every response
    if "Fault" not in response:
        return
    # Use the existing parser logic — it already detects faults
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(response)
    except ET.ParseError:
        return  # not XML, nothing to detect
    for el in root.iter():
        if el.tag.split("}")[-1] == "Fault":
            reason = None
            for e2 in el.iter():
                if e2.tag.split("}")[-1] == "Text" and e2.text:
                    reason = e2.text.strip()
                    break
            raise HomeConfigError(f"SOAP Fault: {reason or 'unknown'}")


def absolute_move(soap_fn: SoapFn, token: str, pan: float, tilt: float, zoom: float) -> None:
    """Move the camera to absolute PTZ coordinates."""
    body = build_absolute_move_xml(token, pan, tilt, zoom)
    response = soap_fn(body)
    _check_fault(response)


def get_current_position(soap_fn: SoapFn, token: str) -> Tuple[float, float, float]:
    """Query the camera for its current pan/tilt/zoom."""
    body = build_get_status_xml(token)
    response = soap_fn(body)
    return parse_ptz_status(response)


def goto_custom_home(soap_fn: SoapFn, token: str, home: HomePosition) -> None:
    """Move the camera to a user-configured home position."""
    absolute_move(soap_fn, token, home.pan, home.tilt, home.zoom)
