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


from tracker import Deadzone


def test_deadzone_first_move_always_passes():
    dz = Deadzone(radius=8)
    assert dz.filter(100, 100) == (100, 100)


def test_deadzone_suppresses_small_move():
    dz = Deadzone(radius=8)
    dz.filter(100, 100)
    assert dz.filter(103, 104) is None  # ~5 px < 8 -> ignored


def test_deadzone_allows_large_move():
    dz = Deadzone(radius=8)
    dz.filter(100, 100)
    assert dz.filter(100, 120) == (100, 120)  # 20 px >= 8 -> moves


def test_deadzone_reference_is_not_cumulative():
    # Small steps are measured from the last committed position, not summed.
    dz = Deadzone(radius=10)
    dz.filter(0, 0)
    assert dz.filter(6, 0) is None    # 6 < 10
    assert dz.filter(9, 0) is None    # still measured from (0,0): 9 < 10
    assert dz.filter(11, 0) == (11, 0)  # 11 >= 10 -> commits and moves


def test_deadzone_reset_clears_reference():
    dz = Deadzone(radius=8)
    dz.filter(50, 50)
    dz.reset()
    assert dz.filter(51, 51) == (51, 51)  # first move after reset always passes


from tracker import HoldClicker


def _recording_clicker(threshold=0.07, release_threshold=0.11):
    """A HoldClicker that records ('press'|'release', button) events."""
    events = []
    clicker = HoldClicker(
        press_cb=lambda b: events.append(("press", b)),
        release_cb=lambda b: events.append(("release", b)),
        threshold=threshold,
        release_threshold=release_threshold,
    )
    return clicker, events


def test_hold_presses_on_gesture_and_holds():
    clicker, events = _recording_clicker()
    assert clicker.update(left_dist=0.03, right_dist=0.5) == "left"
    # still held while below the release threshold -> no new event
    assert clicker.update(left_dist=0.03, right_dist=0.5) == "left"
    assert events == [("press", "left")]


def test_hold_releases_when_fingers_open_past_hysteresis():
    clicker, events = _recording_clicker()
    clicker.update(left_dist=0.03, right_dist=0.5)   # press left
    # opening just past threshold but within hysteresis keeps it held
    assert clicker.update(left_dist=0.09, right_dist=0.5) == "left"
    # opening past the release threshold releases it
    assert clicker.update(left_dist=0.20, right_dist=0.5) is None
    assert events == [("press", "left"), ("release", "left")]


def test_hold_only_one_button_closest_wins():
    clicker, events = _recording_clicker()
    # both gestures below threshold -> the closer one (right) wins
    assert clicker.update(left_dist=0.06, right_dist=0.02) == "right"
    assert events == [("press", "right")]


def test_hold_no_press_when_above_threshold():
    clicker, events = _recording_clicker()
    assert clicker.update(left_dist=0.20, right_dist=0.20) is None
    assert events == []


def test_release_all_releases_held_button():
    clicker, events = _recording_clicker()
    clicker.update(left_dist=0.03, right_dist=0.5)   # press left
    clicker.release_all()
    assert clicker.held is None
    assert events == [("press", "left"), ("release", "left")]


from tracker import (
    speed_to_margin,
    is_fist,
    SPEED_MIN,
    SPEED_MAX,
    MARGIN_AT_MIN_SPEED,
    MARGIN_AT_MAX_SPEED,
    FINGER_TIPS,
    FINGER_PIPS,
)


def test_speed_min_maps_to_smallest_margin():
    assert math.isclose(speed_to_margin(SPEED_MIN), MARGIN_AT_MIN_SPEED)


def test_speed_max_maps_to_largest_margin():
    assert math.isclose(speed_to_margin(SPEED_MAX), MARGIN_AT_MAX_SPEED)


def test_speed_interior_point_is_linear():
    # Speed 4 sits 1/3 of the way through the range: 0.05 + (1/3)*0.30 = 0.15.
    assert math.isclose(speed_to_margin(4), 0.15)


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
