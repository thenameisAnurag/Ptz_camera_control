"""
tracker_state.py — ID-aware target selection on top of a tracking-by-detection
algorithm (BoT-SORT, ByteTrack, etc.).

This module is intentionally agnostic to *which* tracker produced the
results. We accept track tuples in the shape:

    (x1, y1, x2, y2, conf, track_id)

…and decide which track the PTZ should steer toward.

Policy (set by design decisions made before implementation):
  - First lock: closest-to-frame-center wins (user aims first, then enables
    tracking — so the thing they care about is already centered).
  - Once a track ID is committed: we follow that ID exclusively, even if
    a larger or more central object enters the frame.
  - If the committed ID disappears: coast for `coast_frames` frames (default
    45 ≈ 1.5s at 30 fps). During coast, return None so the camera doesn't
    chase a stand-in.
  - After coast expires: release the lock and pick a new target using the
    initial-lock policy (closest-to-center).
  - Tentative tracks (track_id is None) are never committed — they're what
    the underlying tracker emits before it has enough evidence to assign an
    ID. Picking one of these would defeat the whole point of using a tracker.

The miss counter only increments while a commitment is held. Frames with no
tracks at all are not "misses" of anything when we're unlocked.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

# Track tuple type alias for readability
Track = Tuple[int, int, int, int, float, Optional[int]]


# ── Pure helper ──────────────────────────────────────────────────────────────
def _center_distance_sq(x1: int, y1: int, x2: int, y2: int,
                        frame_w: int, frame_h: int) -> int:
    """
    Squared distance from a box's center to the frame center.
    Squared (not sqrt'd) because we only ever compare distances, and sqrt
    is monotonic — skipping it saves a few microseconds per call and keeps
    the result an int.
    """
    box_cx = (x1 + x2) // 2
    box_cy = (y1 + y2) // 2
    frame_cx = frame_w // 2
    frame_cy = frame_h // 2
    dx = box_cx - frame_cx
    dy = box_cy - frame_cy
    return dx * dx + dy * dy


# ── State container ──────────────────────────────────────────────────────────
@dataclass
class TrackerState:
    """
    Mutable state passed across pick_track() calls.

    committed_id : the track ID we're currently following. None = unlocked.
    miss_count   : consecutive frames where the committed ID was missing.
                   Reset to 0 whenever the committed track is found again,
                   or when the lock is released.
    coast_frames : how many consecutive missing frames are tolerated before
                   releasing the lock. Configurable so tests can use short
                   values and production can use 45.
    """
    committed_id: Optional[int] = None
    miss_count: int = 0
    coast_frames: int = 45

    def reset(self) -> None:
        """Wipe runtime state. coast_frames (config) is preserved."""
        self.committed_id = None
        self.miss_count = 0


# ── Helpers internal to pick_track ───────────────────────────────────────────
def _confirmed_tracks(tracks: List[Track]) -> List[Track]:
    """Filter out tentative (track_id=None) tracks."""
    return [t for t in tracks if t[5] is not None]


def _find_committed(tracks: List[Track], committed_id: int) -> Optional[Track]:
    """Return the track matching committed_id, or None."""
    for t in tracks:
        if t[5] == committed_id:
            return t
    return None


def _pick_closest_to_center(tracks: List[Track],
                            frame_w: int, frame_h: int) -> Optional[Track]:
    """
    Initial-lock policy: smallest center-distance wins.
    Tie-break by track_id (ascending) so results are deterministic.
    """
    if not tracks:
        return None
    return min(
        tracks,
        key=lambda t: (_center_distance_sq(t[0], t[1], t[2], t[3],
                                           frame_w, frame_h), t[5])
    )


# ── Main API ─────────────────────────────────────────────────────────────────
def pick_track(tracks: List[Track], state: TrackerState,
               frame_w: int, frame_h: int) -> Optional[Track]:
    """
    Decide which track (if any) the camera should steer toward this frame.

    Mutates `state` in place. Returns the chosen track tuple, or None if
    nothing should be steered toward this frame (either no targets, or we're
    coasting through a temporary occlusion).

    A returned None means "don't move the camera" — the caller's existing
    stale-frame handling can decide what to do next.
    """
    confirmed = _confirmed_tracks(tracks)

    # ── Case 1: we have a committed ID ───────────────────────────────────
    if state.committed_id is not None:
        match = _find_committed(confirmed, state.committed_id)
        if match is not None:
            # Lock holds — found our target
            state.miss_count = 0
            return match

        # Committed target missing this frame
        state.miss_count += 1
        if state.miss_count >= state.coast_frames:
            # Coast expired — release the lock and re-acquire
            state.committed_id = None
            state.miss_count = 0
            # Fall through to re-acquisition below
        else:
            # Still coasting — don't return a stand-in
            return None

    # ── Case 2: unlocked (either never locked, or just released) ─────────
    if not confirmed:
        # Nothing confirmed to commit to. Note: we do NOT increment
        # miss_count here because we have no commitment to miss.
        return None

    chosen = _pick_closest_to_center(confirmed, frame_w, frame_h)
    if chosen is not None:
        state.committed_id = chosen[5]
        state.miss_count = 0
    return chosen
