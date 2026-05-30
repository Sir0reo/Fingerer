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
