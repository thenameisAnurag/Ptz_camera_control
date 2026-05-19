"""
lost_home_timer.py — "no target for N seconds → go home" logic.

Design
------
Pure logic, no network or threading. Driven by an injected clock so tests
are deterministic and the same code runs in production with time.monotonic.

Two-state machine:
  - "locked"       : last update() said the camera has a confirmed target
  - "lost"         : last update() said no confirmed target
                     (started at lost_since_t)

When `now - lost_since_t >= threshold_seconds` AND the home signal has not
yet been consumed for this loss episode, should_go_home() returns True.

The "single-shot" behavior is critical: without it, every frame past the
threshold would re-fire the home command, drowning the PTZ command queue
and making the camera unable to actually leave home. consume_home_signal()
flips a flag that only resets when:
  - the camera regains a lock (recovery resets everything), or
  - reset() is called explicitly (e.g. on mode toggle)

The integration in track.py is:

    timer.update(locked=(tracker_state.committed_id is not None))
    if timer.consume_home_signal():
        ptz_thread.send({'action': 'home'})
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class LostHomeTimer:
    """
    Tracks consecutive time without a confirmed lock and emits a single
    "go home" signal when the threshold is crossed.

    Parameters
    ----------
    threshold_seconds : seconds of continuous "no lock" before we give up
                        and return home. Default 10.0 per the design spec.
    clock             : callable returning monotonically increasing seconds.
                        Defaults to time.monotonic. Tests inject a fake.
    """

    def __init__(self, threshold_seconds: float = 10.0,
                 clock: Optional[Callable[[], float]] = None):
        self.threshold_seconds: float = float(threshold_seconds)
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

        # State
        self._lost_since: Optional[float] = None   # wall time we entered "lost"
        self._signal_pending: bool = False         # threshold crossed, not yet consumed
        self._signal_consumed: bool = False        # consumed for current loss episode

    # ── Inputs ───────────────────────────────────────────────────────────
    def update(self, locked: bool) -> None:
        """
        Call once per frame with whether we currently have a confirmed lock.

        Locked → reset the lost timer entirely.
        Unlocked + first such frame → record lost_since.
        Unlocked + threshold crossed + signal not yet consumed → pend the signal.
        """
        if locked:
            # Any kind of lock recovery wipes the loss episode entirely:
            # the next loss starts a fresh timer AND a fresh single-shot.
            self._lost_since = None
            self._signal_pending = False
            self._signal_consumed = False
            return

        # Unlocked
        now = self._clock()
        if self._lost_since is None:
            self._lost_since = now
            self._signal_pending = False
            self._signal_consumed = False
            return

        # Already in a loss episode — check if we've crossed the threshold.
        elapsed = now - self._lost_since
        if elapsed >= self.threshold_seconds and not self._signal_consumed:
            self._signal_pending = True

    # ── Outputs ──────────────────────────────────────────────────────────
    def is_lost(self) -> bool:
        """True while we're in an active loss episode (timer is running)."""
        return self._lost_since is not None

    def should_go_home(self) -> bool:
        """
        True if the threshold has been crossed and the signal is still pending.
        This is a *peek* — it does not consume the signal. Use consume_home_signal()
        to actually act on it.
        """
        return self._signal_pending

    def consume_home_signal(self) -> bool:
        """
        Atomically read-and-clear the pending home signal. Returns True if the
        caller should now send the home command; subsequent calls return False
        until a new loss episode reaches the threshold.
        """
        if not self._signal_pending:
            return False
        self._signal_pending = False
        self._signal_consumed = True
        return True

    def reset(self) -> None:
        """
        Wipe all runtime state. threshold_seconds (config) is preserved.
        Call on mode toggles or anywhere the loss history shouldn't carry over.
        """
        self._lost_since = None
        self._signal_pending = False
        self._signal_consumed = False
