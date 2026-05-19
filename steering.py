"""
steering.py — Pan/tilt control law.

Why this module exists
----------------------
The original offset_to_ptz() in track.py used a LINEAR speed ramp from
spd_min (just outside dead zone) to spd_max (at frame edge). On a real
PTZ camera, this caused observable left-right oscillation:
t
  1. Target detected at, say, 40% right of center
  2. Linear ramp picks a fairly aggressive speed (~60% of spd_max)
  3. ONVIF ContinuousMove keeps the camera moving until "stop" is sent
  4. By the time the NEXT detection arrives (50-100ms later), the camera
     has overshot — the target is now LEFT of center
  5. Loop reverses, overshoots the other way → visible oscillation

The quadratic curve (curve_exponent=2.0) reshapes the speed-vs-offset
relationship so the camera creeps slowly when the target is mid-frame
and only ramps up when the target is near the edge. Combined with a
WIDER dead zone, the camera "lags but stays steady" — which was the
user's stated preference for airplane tracking.

This module has no torch / opencv / network dependencies, so it can be
unit-tested in isolation. track.py wires it into the YOLO output handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SteeringConfig:
    """
    Tunable parameters for the pan/tilt control law.

    dead_zone      : fraction of half-frame (0.0 - 1.0). No movement when
                     target offset is within this radius. Wider = camera
                     stops sooner, less jitter, more lag.
    spd_min        : pan/tilt velocity just outside the dead zone.
                     Should be small enough that the camera barely moves
                     when target is near-center.
    spd_max        : pan/tilt velocity at the frame edge. ONVIF speeds are
                     normalized [0, 1]. Most cameras: 0.18 ≈ 15-20°/sec.
                     Higher = faster but more overshoot.
    curve_exponent : speed curve shape.
                     1.0 = linear (the OLD behavior)
                     2.0 = quadratic (recommended — slow near center)
                     3.0 = cubic    (very gentle near center)
    pan_dir        : +1 or -1. -1 inverts pan velocity sign (verified by
                     pressing 'D' in manual mode and checking direction).
    tilt_dir       : +1 or -1. Same idea for tilt.
    """
    dead_zone: float = 0.15
    spd_min: float = 0.05
    spd_max: float = 0.18
    curve_exponent: float = 2.0
    pan_dir: int = -1
    tilt_dir: int = +1


def _axis_speed(dx: float, cfg: SteeringConfig) -> float:
    """
    Map a normalized offset (dx ∈ [-1, +1]) to a signed velocity using the
    configured speed curve. Returns 0 inside the dead zone.

    The curve:
      magnitude = ((|dx| - dead_zone) / (1 - dead_zone)) ^ curve_exponent
      speed     = spd_min + magnitude * (spd_max - spd_min)

    At |dx| = dead_zone:   magnitude = 0     → speed = spd_min
    At |dx| = 1.0:         magnitude = 1     → speed = spd_max
    With exponent=2.0, halfway between these gives magnitude=0.25, NOT 0.5.
    That's the slow-near-center behavior.
    """
    if abs(dx) <= cfg.dead_zone:
        return 0.0
    if cfg.dead_zone >= 1.0:
        # Degenerate config: dead zone covers everything
        return 0.0

    normalized = (abs(dx) - cfg.dead_zone) / (1.0 - cfg.dead_zone)
    magnitude = normalized ** cfg.curve_exponent
    speed = cfg.spd_min + magnitude * (cfg.spd_max - cfg.spd_min)

    # Sign comes from dx; caller flips with pan_dir/tilt_dir separately
    return speed if dx >= 0 else -speed


def compute_steering(cx_obj: int, cy_obj: int,
                     frame_w: int, frame_h: int,
                     cfg: SteeringConfig) -> Tuple[float, float]:
    """
    Decide what pan/tilt velocity to send to the PTZ given a target's
    pixel coordinates and the frame size.

    Parameters
    ----------
    cx_obj, cy_obj : target center in pixels
    frame_w, frame_h : current frame dimensions in pixels
    cfg : SteeringConfig

    Returns
    -------
    (pan, tilt) : signed velocities in ONVIF normalized range.
                  Either or both may be 0.0 when in the dead zone.
    """
    if frame_w <= 0 or frame_h <= 0:
        return 0.0, 0.0

    # Normalize: -1.0 (left/top) to +1.0 (right/bottom)
    dx = (cx_obj - frame_w / 2) / (frame_w / 2)
    dy = (cy_obj - frame_h / 2) / (frame_h / 2)

    pan_axis = _axis_speed(dx, cfg)
    tilt_axis = _axis_speed(dy, cfg)

    # Apply per-axis sign conventions (camera-specific orientation)
    pan = cfg.pan_dir * pan_axis
    tilt = cfg.tilt_dir * tilt_axis

    return float(pan), float(tilt)
