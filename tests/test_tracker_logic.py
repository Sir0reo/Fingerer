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


def test_map_low_edge_clamps_to_zero():
    x, y = map_to_screen(0.15, 0.15, 1920, 1080, margin=0.15)
    assert x == 0
    assert y == 0


def test_map_high_edge_clamps_to_max():
    x, y = map_to_screen(0.85, 0.85, 1920, 1080, margin=0.15)
    assert x == 1920
    assert y == 1080


def test_map_below_region_clamps_not_negative():
    x, y = map_to_screen(0.0, 0.0, 1920, 1080, margin=0.15)
    assert x == 0
    assert y == 0


from tracker import Smoother


def test_smoother_single_value_returns_itself():
    s = Smoother(maxlen=5)
    assert s.add(100, 200) == (100, 200)


def test_smoother_averages_history():
    s = Smoother(maxlen=5)
    s.add(0, 0)
    assert s.add(10, 20) == (5, 10)  # mean of (0,0) and (10,20)


def test_smoother_respects_maxlen():
    s = Smoother(maxlen=2)
    s.add(0, 0)
    s.add(10, 10)
    # third value evicts the first; mean of (10,10) and (40,40)
    assert s.add(40, 40) == (25, 25)
