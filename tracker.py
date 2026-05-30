"""Fingerer — finger mouse control tracking logic."""

import math
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # slam cursor to a corner to abort

# --- Tunable constants ---------------------------------------------------
SMOOTHING = 5            # rolling-average window (frames)
MARGIN = 0.15            # inset fraction of frame edges for the active region
PINCH_THRESHOLD = 0.05   # normalized fingertip distance for a pinch
CLICK_COOLDOWN = 0.3     # seconds between clicks of the same button
CAM_INDEX = 0            # default webcam index

# MediaPipe landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12


def distance(p1, p2):
    """Euclidean distance between two points with .x/.y attributes."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def map_to_screen(nx, ny, screen_w, screen_h, margin=MARGIN):
    """Map normalized (nx, ny) within the inset active region to screen pixels.

    The region [margin, 1-margin] maps onto the screen; out-of-region values
    clamp to the edges. The output is bounded to [1, screen_dim - 1] so it stays
    within valid pixel coordinates and never lands on pyautogui's (0, 0)
    fail-safe corner (which would abort tracking during normal use). Returns
    integer (x, y).
    """
    low, high = margin, 1.0 - margin
    x = np.interp(nx, [low, high], [1, screen_w - 1])
    y = np.interp(ny, [low, high], [1, screen_h - 1])
    return int(round(x)), int(round(y))


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


class ClickLatch:
    """Single-button pinch-click gate with release latch + cooldown."""

    def __init__(self, threshold=PINCH_THRESHOLD, cooldown=CLICK_COOLDOWN):
        self.threshold = threshold
        self.cooldown = cooldown
        self._pinched = False        # currently below threshold (held)
        self._last_click_time = -1e9

    def update(self, dist, now):
        """Return True exactly once per fresh pinch, honoring cooldown."""
        below = dist < self.threshold
        fire = False
        if below and not self._pinched:
            if now - self._last_click_time >= self.cooldown:
                fire = True
                self._last_click_time = now
        self._pinched = below
        return fire

    def reset(self):
        self._pinched = False
        self._last_click_time = -1e9


class FingerMouseTracker:
    """Runs webcam hand tracking on a background thread and drives the mouse."""

    def __init__(self, status_callback=None, cam_index=CAM_INDEX):
        self._status_callback = status_callback or (lambda s: None)
        self._cam_index = cam_index
        self._thread = None
        self._stop_event = threading.Event()
        self._screen_w, self._screen_h = pyautogui.size()
        self._smoother = Smoother(SMOOTHING)
        self._left = ClickLatch()
        self._right = ClickLatch()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def _status(self, msg):
        self._status_callback(msg)

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._smoother.reset()
        self._left.reset()
        self._right.reset()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self):
        cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._status("Camera error: could not open webcam")
            return
        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._status("Tracking — no hand")
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._status("Camera error: frame grab failed")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    self._process_hand(result.multi_hand_landmarks[0])
                else:
                    self._status("Tracking — no hand")
        except Exception as exc:  # keep the app alive; report and stop
            self._status(f"Camera error: {exc}")
        finally:
            cap.release()
            hands.close()
            self._status("Stopped")

    def _process_hand(self, landmarks):
        lm = landmarks.landmark
        index, thumb, middle = lm[INDEX_TIP], lm[THUMB_TIP], lm[MIDDLE_TIP]

        sx, sy = map_to_screen(index.x, index.y, self._screen_w, self._screen_h)
        smooth_x, smooth_y = self._smoother.add(sx, sy)
        pyautogui.moveTo(smooth_x, smooth_y)

        now = time.time()
        left_dist = distance(thumb, index)
        right_dist = distance(thumb, middle)

        # Always advance BOTH latches so their pinched/release state stays
        # coherent every frame; then apply left-click priority.
        left_fire = self._left.update(left_dist, now)
        right_fire = self._right.update(right_dist, now)
        if left_fire:
            pyautogui.click()
            self._status("Tracking — left click")
        elif right_fire:
            pyautogui.click(button="right")
            self._status("Tracking — right click")
        else:
            self._status("Tracking — hand detected")
