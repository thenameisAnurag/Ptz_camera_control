"""
Tests for steering.py — pan/tilt control law.

This is the function that decides "given a target at pixel (cx, cy), what
PTZ velocity should we send?" The old version oscillated because:
  1. Linear ramp from 0.08 to 0.35 — too aggressive when target is mid-frame
  2. Tight 10% dead zone — camera kept overshooting then correcting

The new version uses a QUADRATIC speed curve: speed grows slowly near the
dead zone and only ramps hard when the target is near the frame edge.
This matches our design decision "slow + smooth, camera lags but stays
steady."

Coordinate convention (same as before):
  dx = (cx - w/2) / (w/2)   →   -1.0 (full left) ... +1.0 (full right)
  dy = (cy - h/2) / (h/2)   →   -1.0 (top)       ... +1.0 (bottom)
"""

import pytest
from steering import compute_steering, SteeringConfig


# ── Convenience: standard 1280x720 frame ─────────────────────────────────────
W, H = 1280, 720
CX, CY = W // 2, H // 2  # 640, 360


def default_config():
    """Mirror what track.py will use in production."""
    return SteeringConfig(
        dead_zone=0.15,         # widened from 0.10
        spd_min=0.05,           # gentler
        spd_max=0.18,           # half the old 0.35
        curve_exponent=2.0,     # quadratic — slow near center, hard at edge
        pan_dir=-1,
        tilt_dir=+1,
    )


# ── Dead zone behavior ──────────────────────────────────────────────────────
class TestDeadZone:
    def test_exact_center_returns_zero_pan_zero_tilt(self):
        cfg = default_config()
        pan, tilt = compute_steering(CX, CY, W, H, cfg)
        assert pan == 0.0
        assert tilt == 0.0

    def test_inside_dead_zone_returns_zero(self):
        # Default dead_zone=0.15 → ±15% radius. At dx=0.10, we're well inside.
        cfg = default_config()
        offset_pixels = int(0.10 * W / 2)
        pan, tilt = compute_steering(CX + offset_pixels, CY, W, H, cfg)
        assert pan == 0.0

    def test_at_edge_of_dead_zone_still_zero(self):
        # Exactly at the dead zone boundary → still 0.
        # (compute_steering should only fire when |dx| > dead_zone, not >=.)
        cfg = default_config()
        offset_pixels = int(0.15 * W / 2)
        pan, tilt = compute_steering(CX + offset_pixels, CY, W, H, cfg)
        assert pan == 0.0
        assert tilt == 0.0

    def test_just_outside_dead_zone_returns_spd_min(self):
        # Slightly outside the dead zone should yield approximately spd_min,
        # not some larger value. This is the key fix vs the old linear ramp,
        # where speed grew linearly from min to max — too aggressive.
        cfg = default_config()
        # 1% past the dead zone edge
        offset_pixels = int((cfg.dead_zone + 0.01) * W / 2)
        pan, tilt = compute_steering(CX + offset_pixels, CY, W, H, cfg)
        # Should be very close to spd_min — within a few percent
        assert abs(pan) > 0.0
        assert abs(pan) < cfg.spd_min * 1.5


# ── Max speed at frame edge ──────────────────────────────────────────────────
class TestMaxSpeed:
    def test_far_left_edge_uses_spd_max(self):
        cfg = default_config()
        # Target at left edge of frame
        pan, tilt = compute_steering(0, CY, W, H, cfg)
        # PAN_DIR=-1, dx=-1 → pan = -1 * spd_max * -1 = +spd_max
        assert abs(pan) == pytest.approx(cfg.spd_max, abs=0.01)

    def test_far_right_edge_uses_spd_max(self):
        cfg = default_config()
        pan, tilt = compute_steering(W - 1, CY, W, H, cfg)
        assert abs(pan) == pytest.approx(cfg.spd_max, abs=0.01)

    def test_speed_never_exceeds_spd_max(self):
        cfg = default_config()
        # Iterate offsets across the full frame
        for cx in range(0, W, 32):
            pan, _ = compute_steering(cx, CY, W, H, cfg)
            assert abs(pan) <= cfg.spd_max + 1e-9


# ── Quadratic curve: slow near center, fast at edge ─────────────────────────
class TestQuadraticCurve:
    def test_halfway_is_much_closer_to_min_than_max(self):
        # This is THE behavioral test that distinguishes the new curve from
        # the old linear ramp.
        # At halfway between dead zone and frame edge, a linear ramp would
        # give (min+max)/2. A quadratic gives min + 0.25 * (max-min) — much
        # closer to min. This is what makes the camera move smoothly.
        cfg = default_config()
        # Halfway between dead zone (0.15) and edge (1.0) → dx ≈ 0.575
        offset_pixels = int(0.575 * W / 2)
        pan, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg)

        linear_midpoint = (cfg.spd_min + cfg.spd_max) / 2
        # Quadratic at halfway → magnitude = 0.5² = 0.25, so speed is at
        # 25% of the (min → max) span instead of 50%. The resulting speed
        # should be noticeably below the linear midpoint.
        assert abs(pan) < linear_midpoint
        # And specifically: quadratic's magnitude is 0.25, so the speed is
        # spd_min + 0.25 * (spd_max - spd_min)
        expected = cfg.spd_min + 0.25 * (cfg.spd_max - cfg.spd_min)
        assert abs(pan) == pytest.approx(expected, abs=0.005)

    def test_speed_grows_monotonically_with_offset(self):
        # As the target moves from dead zone edge toward frame edge,
        # the speed should never decrease.
        cfg = default_config()
        speeds = []
        for frac in [0.20, 0.30, 0.50, 0.70, 0.90, 0.99]:
            offset_pixels = int(frac * W / 2)
            pan, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg)
            speeds.append(abs(pan))
        # Each successive speed must be >= the previous
        for prev, cur in zip(speeds, speeds[1:]):
            assert cur >= prev


# ── Direction signs ─────────────────────────────────────────────────────────
class TestDirectionSigns:
    def test_target_right_of_center_pans_one_way(self):
        # With PAN_DIR=-1 and dx>0, pan = -1 * speed * +1 = NEGATIVE
        # (this matches the user's working config: 'D' moves right)
        cfg = default_config()
        pan, _ = compute_steering(W - 100, CY, W, H, cfg)
        assert pan < 0

    def test_target_left_of_center_pans_other_way(self):
        cfg = default_config()
        pan, _ = compute_steering(100, CY, W, H, cfg)
        assert pan > 0

    def test_target_below_center_tilts_one_way(self):
        # TILT_DIR=+1, dy>0 → tilt = +1 * speed * +1 = POSITIVE
        cfg = default_config()
        _, tilt = compute_steering(CX, H - 100, W, H, cfg)
        assert tilt > 0

    def test_target_above_center_tilts_other_way(self):
        cfg = default_config()
        _, tilt = compute_steering(CX, 100, W, H, cfg)
        assert tilt < 0


# ── Symmetry ─────────────────────────────────────────────────────────────────
class TestSymmetry:
    def test_equal_offset_left_and_right_gives_equal_speed(self):
        cfg = default_config()
        offset_pixels = 300
        pan_right, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg)
        pan_left, _ = compute_steering(CX - offset_pixels, CY, W, H, cfg)
        assert abs(pan_right) == pytest.approx(abs(pan_left), abs=1e-6)

    def test_equal_offset_up_and_down_gives_equal_speed(self):
        cfg = default_config()
        offset_pixels = 150
        _, tilt_down = compute_steering(CX, CY + offset_pixels, W, H, cfg)
        _, tilt_up = compute_steering(CX, CY - offset_pixels, W, H, cfg)
        assert abs(tilt_down) == pytest.approx(abs(tilt_up), abs=1e-6)


# ── Config knobs are respected ──────────────────────────────────────────────
class TestConfigRespected:
    def test_custom_dead_zone_widens_no_move_region(self):
        # With dead_zone=0.5, a target at dx=0.4 should still produce zero
        cfg = SteeringConfig(dead_zone=0.5, spd_min=0.05, spd_max=0.18,
                             curve_exponent=2.0, pan_dir=-1, tilt_dir=+1)
        offset_pixels = int(0.4 * W / 2)
        pan, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg)
        assert pan == 0.0

    def test_custom_spd_max_caps_speed(self):
        cfg = SteeringConfig(dead_zone=0.10, spd_min=0.02, spd_max=0.10,
                             curve_exponent=2.0, pan_dir=-1, tilt_dir=+1)
        pan, _ = compute_steering(0, CY, W, H, cfg)
        assert abs(pan) <= 0.10 + 1e-9

    def test_curve_exponent_1_gives_linear_ramp(self):
        # Linear ramp (the OLD behavior) — useful for regression testing
        cfg = SteeringConfig(dead_zone=0.10, spd_min=0.08, spd_max=0.35,
                             curve_exponent=1.0, pan_dir=-1, tilt_dir=+1)
        # Halfway from dead zone to edge → halfway between min and max
        offset_pixels = int(0.55 * W / 2)
        pan, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg)
        expected = (0.08 + 0.35) / 2
        assert abs(pan) == pytest.approx(expected, abs=0.02)

    def test_curve_exponent_3_is_even_gentler_than_2(self):
        # Higher exponent → even slower near center
        cfg_2 = SteeringConfig(dead_zone=0.10, spd_min=0.05, spd_max=0.20,
                               curve_exponent=2.0, pan_dir=-1, tilt_dir=+1)
        cfg_3 = SteeringConfig(dead_zone=0.10, spd_min=0.05, spd_max=0.20,
                               curve_exponent=3.0, pan_dir=-1, tilt_dir=+1)
        offset_pixels = int(0.40 * W / 2)
        pan_2, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg_2)
        pan_3, _ = compute_steering(CX + offset_pixels, CY, W, H, cfg_3)
        assert abs(pan_3) < abs(pan_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
