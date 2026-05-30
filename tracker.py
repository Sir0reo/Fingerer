"""Fingerer — finger mouse control tracking logic."""

import math
from collections import deque

import numpy as np

# --- Tunable constants ---------------------------------------------------
SMOOTHING = 5            # rolling-average window (frames)
MARGIN = 0.15            # inset fraction of frame edges for the active region
PINCH_THRESHOLD = 0.05   # normalized fingertip distance for a pinch
CLICK_COOLDOWN = 0.3     # seconds between clicks of the same button
CAM_INDEX = 0            # default webcam index


def distance(p1, p2):
    """Euclidean distance between two points with .x/.y attributes."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def map_to_screen(nx, ny, screen_w, screen_h, margin=MARGIN):
    """Map normalized (nx, ny) within the inset active region to screen pixels.

    The region [margin, 1-margin] maps onto [0, screen_dim]; out-of-region
    values clamp to the edges. Returns integer (x, y).
    """
    low, high = margin, 1.0 - margin
    x = np.interp(nx, [low, high], [0, screen_w])
    y = np.interp(ny, [low, high], [0, screen_h])
    return int(x), int(y)


class Smoother:
    """Rolling-average smoother for (x, y) coordinates."""

    def __init__(self, maxlen=SMOOTHING):
        self._xs = deque(maxlen=maxlen)
        self._ys = deque(maxlen=maxlen)

    def add(self, x, y):
        """Add a sample and return the integer (mean_x, mean_y)."""
        self._xs.append(x)
        self._ys.append(y)
        return (
            int(sum(self._xs) / len(self._xs)),
            int(sum(self._ys) / len(self._ys)),
        )

    def reset(self):
        self._xs.clear()
        self._ys.clear()
