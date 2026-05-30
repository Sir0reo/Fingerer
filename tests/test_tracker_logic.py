import math
from types import SimpleNamespace

from tracker import distance, map_to_screen, MARGIN


def _pt(x, y):
    return SimpleNamespace(x=x, y=y)


def test_distance_zero_for_same_point():
    assert distance(_pt(0.5, 0.5), _pt(0.5, 0.5)) == 0.0


def test_distance_3_4_5_triangle():
    assert math.isclose(distance(_pt(0.0, 0.0), _pt(0.3, 0.4)), 0.5)


def test_map_center_maps_to_screen_center():
    x, y = map_to_screen(0.5, 0.5, 1920, 1080, margin=0.15)
    assert x == 960
    assert y == 540


def test_map_low_edge_clamps_to_min():
    x, y = map_to_screen(0.15, 0.15, 1920, 1080, margin=0.15)
    assert x == 1
    assert y == 1


def test_map_high_edge_clamps_to_max():
    x, y = map_to_screen(0.85, 0.85, 1920, 1080, margin=0.15)
    assert x == 1919
    assert y == 1079


def test_map_below_region_clamps_not_negative():
    x, y = map_to_screen(0.0, 0.0, 1920, 1080, margin=0.15)
    assert x == 1
    assert y == 1


def test_map_corner_never_hits_failsafe_origin():
    # Reaching the top-left of the active region must NOT map to (0, 0), which
    # would trip pyautogui's fail-safe and abort tracking.
    assert map_to_screen(0.15, 0.15, 1920, 1080) != (0, 0)
    assert map_to_screen(0.0, 0.0, 1920, 1080) != (0, 0)


from tracker import OneEuroFilter, CursorSmoother


def test_oneeuro_first_sample_passes_through():
    f = OneEuroFilter(min_cutoff=1.0)
    assert f.filter(5.0, t=0.0) == 5.0


def test_oneeuro_smooths_a_step_between_old_and_new():
    f = OneEuroFilter(min_cutoff=1.0)
    f.filter(0.0, t=0.0)
    out = f.filter(10.0, t=1 / 30)  # one frame later
    assert 0.0 < out < 10.0  # output lags toward the new value, not all the way


def test_oneeuro_converges_to_a_held_value():
    f = OneEuroFilter(min_cutoff=1.0)
    f.filter(0.0, t=0.0)
    t = 0.0
    for _ in range(120):  # hold 1.0 for ~4 seconds at 30 fps
        t += 1 / 30
        out = f.filter(1.0, t=t)
    assert abs(out - 1.0) < 0.05


def test_oneeuro_reset_restarts_passthrough():
    f = OneEuroFilter(min_cutoff=1.0)
    f.filter(3.0, t=0.0)
    f.filter(9.0, t=0.1)
    f.reset()
    assert f.filter(7.0, t=0.2) == 7.0


def test_cursor_smoother_returns_first_sample_rounded():
    s = CursorSmoother(min_cutoff=1.0)
    assert s.add(100.4, 200.6, t=0.0) == (100, 201)


from tracker import ClickLatch


def test_latch_fires_on_first_pinch():
    latch = ClickLatch(threshold=0.05, cooldown=0.3)
    assert latch.update(dist=0.02, now=0.0) is True


def test_latch_does_not_repeat_while_held():
    latch = ClickLatch(threshold=0.05, cooldown=0.3)
    assert latch.update(dist=0.02, now=0.0) is True
    # still pinched, no release yet -> no second fire
    assert latch.update(dist=0.02, now=0.1) is False
    assert latch.update(dist=0.02, now=0.5) is False


def test_latch_refires_after_release_and_cooldown():
    latch = ClickLatch(threshold=0.05, cooldown=0.3)
    assert latch.update(dist=0.02, now=0.0) is True
    latch.update(dist=0.20, now=0.1)   # released (above threshold)
    # released but cooldown (0.3s) not elapsed yet
    assert latch.update(dist=0.02, now=0.2) is False
    latch.update(dist=0.20, now=0.35)  # release again
    assert latch.update(dist=0.02, now=0.4) is True  # released + cooldown passed


def test_latch_no_fire_when_above_threshold():
    latch = ClickLatch(threshold=0.05, cooldown=0.3)
    assert latch.update(dist=0.10, now=0.0) is False


from tracker import (
    speed_to_margin,
    is_fist,
    SPEED_MIN,
    SPEED_MAX,
    DEFAULT_SPEED,
    MARGIN_AT_MIN_SPEED,
    MARGIN_AT_MAX_SPEED,
    FINGER_TIPS,
    FINGER_PIPS,
)


def test_speed_min_maps_to_smallest_margin():
    assert math.isclose(speed_to_margin(SPEED_MIN), MARGIN_AT_MIN_SPEED)


def test_speed_max_maps_to_largest_margin():
    assert math.isclose(speed_to_margin(SPEED_MAX), MARGIN_AT_MAX_SPEED)


def test_default_speed_matches_legacy_margin():
    # Default speed (4) should reproduce the original 0.15 active-region margin.
    assert math.isclose(speed_to_margin(DEFAULT_SPEED), MARGIN)


def test_speed_clamps_out_of_range():
    assert speed_to_margin(SPEED_MIN - 5) == speed_to_margin(SPEED_MIN)
    assert speed_to_margin(SPEED_MAX + 5) == speed_to_margin(SPEED_MAX)


def _hand(curled):
    """Build a fake 21-landmark hand. If curled, finger tips sit below (larger y)
    their PIPs; otherwise tips sit above (smaller y)."""
    pts = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        pts[pip] = SimpleNamespace(x=0.5, y=0.5)
        pts[tip] = SimpleNamespace(x=0.5, y=0.6 if curled else 0.4)
    return SimpleNamespace(landmark=pts)


def test_is_fist_true_for_curled_hand():
    assert is_fist(_hand(curled=True)) is True


def test_is_fist_false_for_open_hand():
    assert is_fist(_hand(curled=False)) is False


def test_is_fist_false_if_one_finger_extended():
    hand = _hand(curled=True)
    hand.landmark[FINGER_TIPS[1]] = SimpleNamespace(x=0.5, y=0.4)  # middle extended
    assert is_fist(hand) is False


from tracker import (
    smoothing_to_cutoff,
    SMOOTHING_MIN,
    SMOOTHING_MAX,
    MIN_CUTOFF_AT_MIN_SMOOTHING,
    MIN_CUTOFF_AT_MAX_SMOOTHING,
)


def test_min_smoothing_maps_to_most_responsive_cutoff():
    assert math.isclose(smoothing_to_cutoff(SMOOTHING_MIN), MIN_CUTOFF_AT_MIN_SMOOTHING)


def test_max_smoothing_maps_to_steadiest_cutoff():
    assert math.isclose(smoothing_to_cutoff(SMOOTHING_MAX), MIN_CUTOFF_AT_MAX_SMOOTHING)


def test_higher_smoothing_gives_lower_cutoff():
    # More smoothing must mean a lower min-cutoff (stronger low-pass).
    assert smoothing_to_cutoff(15) < smoothing_to_cutoff(5)


def test_smoothing_clamps_out_of_range():
    assert smoothing_to_cutoff(SMOOTHING_MIN - 9) == smoothing_to_cutoff(SMOOTHING_MIN)
    assert smoothing_to_cutoff(SMOOTHING_MAX + 9) == smoothing_to_cutoff(SMOOTHING_MAX)
