"""Fingerer — finger mouse control tracking logic."""

import math

# --- Tunable constants ---------------------------------------------------
SMOOTHING = 5            # rolling-average window (frames)
MARGIN = 0.15            # inset fraction of frame edges for the active region
PINCH_THRESHOLD = 0.05   # normalized fingertip distance for a pinch
CLICK_COOLDOWN = 0.3     # seconds between clicks of the same button
CAM_INDEX = 0            # default webcam index


def distance(p1, p2):
    """Euclidean distance between two points with .x/.y attributes."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)
