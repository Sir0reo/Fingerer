"""Fingerer — finger mouse control tracking logic."""

import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # slam cursor to a corner to abort

# --- Tunable constants ---------------------------------------------------
MARGIN = 0.15            # default inset fraction (overridden live by Speed slider)
PINCH_THRESHOLD = 0.05   # normalized distance for a click gesture
CLICK_COOLDOWN = 0.3     # seconds between clicks of the same button
CAM_INDEX = 0            # default webcam index
CAM_WIDTH, CAM_HEIGHT = 1280, 720   # request higher-res frames for better accuracy

# Speed slider -> active-region margin (higher speed = larger margin = faster cursor)
SPEED_MIN, SPEED_MAX = 1, 10
DEFAULT_SPEED = 4
MARGIN_AT_MIN_SPEED = 0.05   # slow: large active region
MARGIN_AT_MAX_SPEED = 0.35   # fast: small active region

# Smoothing slider -> One Euro Filter min-cutoff (higher slider = smoother).
# The One Euro Filter smooths hard when the hand is slow (kills jitter) but eases
# off when it moves fast (kills lag), unlike a fixed-window moving average.
SMOOTHING_MIN, SMOOTHING_MAX = 1, 20
DEFAULT_SMOOTHING = 10
MIN_CUTOFF_AT_MIN_SMOOTHING = 3.0   # low smoothing: very responsive
MIN_CUTOFF_AT_MAX_SMOOTHING = 0.2   # high smoothing: very steady
ONE_EURO_BETA = 0.01                # speed coefficient — keeps lag low on fast moves
ONE_EURO_DCUTOFF = 1.0              # derivative cutoff (filters the speed estimate)

# Two-fist stop gesture
TWO_FIST_HOLD = 0.4      # seconds both fists must be held to stop tracking

# MediaPipe landmark indices
THUMB_TIP = 4
INDEX_PIP = 6            # index finger middle joint (left-click target)
INDEX_TIP = 8
MIDDLE_TIP = 12
# Fingertip / PIP pairs for fist detection (index, middle, ring, pinky)
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)


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


def speed_to_margin(speed):
    """Map a Speed slider value (SPEED_MIN..SPEED_MAX) to an active-region margin.

    Higher speed -> larger margin -> smaller active region -> the cursor travels
    farther for the same hand movement. Returns a float margin.
    """
    speed = min(max(speed, SPEED_MIN), SPEED_MAX)
    frac = (speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)
    return MARGIN_AT_MIN_SPEED + frac * (MARGIN_AT_MAX_SPEED - MARGIN_AT_MIN_SPEED)


def is_fist(landmarks):
    """Return True if a hand is a closed fist (all four fingers curled).

    A finger is curled when its tip sits lower in the image (larger y) than its
    PIP joint. The thumb is ignored. `landmarks` is a MediaPipe hand-landmark
    object exposing `.landmark` (a sequence of points with .x/.y).
    """
    lm = landmarks.landmark
    return all(lm[tip].y > lm[pip].y for tip, pip in zip(FINGER_TIPS, FINGER_PIPS))


def smoothing_to_cutoff(value):
    """Map a Smoothing slider value to a One Euro Filter min-cutoff frequency.

    Higher slider -> lower cutoff -> stronger smoothing of slow movement.
    Returns a float cutoff frequency.
    """
    value = min(max(value, SMOOTHING_MIN), SMOOTHING_MAX)
    frac = (value - SMOOTHING_MIN) / (SMOOTHING_MAX - SMOOTHING_MIN)
    return MIN_CUTOFF_AT_MIN_SMOOTHING + frac * (
        MIN_CUTOFF_AT_MAX_SMOOTHING - MIN_CUTOFF_AT_MIN_SMOOTHING
    )


class OneEuroFilter:
    """One-dimensional One Euro Filter (Casiez, Roussel & Vogel, 2012).

    An adaptive low-pass filter: it smooths heavily at low speed (removing
    jitter) and lightly at high speed (removing lag). Operates in continuous
    time using per-sample timestamps.
    """

    def __init__(self, min_cutoff=1.0, beta=ONE_EURO_BETA, d_cutoff=ONE_EURO_DCUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x, t):
        """Return the smoothed value of sample x taken at time t (seconds)."""
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3
        self._t_prev = t
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class CursorSmoother:
    """Smooths a 2-D cursor position with one One Euro Filter per axis."""

    def __init__(self, min_cutoff):
        self._fx = OneEuroFilter(min_cutoff)
        self._fy = OneEuroFilter(min_cutoff)

    def add(self, x, y, t):
        """Add a sample at time t; return the smoothed integer (x, y)."""
        return (
            int(round(self._fx.filter(x, t))),
            int(round(self._fy.filter(y, t))),
        )

    def set_min_cutoff(self, cutoff):
        self._fx.min_cutoff = cutoff
        self._fy.min_cutoff = cutoff

    def reset(self):
        self._fx.reset()
        self._fy.reset()


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
        self._smoother = CursorSmoother(smoothing_to_cutoff(DEFAULT_SMOOTHING))
        self._left = ClickLatch()
        self._right = ClickLatch()
        self._margin = speed_to_margin(DEFAULT_SPEED)
        self._fist_since = None     # when both fists were first seen (stop hold)
        self._stop_reason = None    # status message to show on loop exit

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def set_speed(self, value):
        """Set cursor speed (SPEED_MIN..SPEED_MAX); higher = faster cursor."""
        self._margin = speed_to_margin(value)

    def set_smoothing(self, value):
        """Set smoothing strength (SMOOTHING_MIN..SMOOTHING_MAX); higher = steadier."""
        self._smoother.set_min_cutoff(smoothing_to_cutoff(value))

    def _status(self, msg):
        self._status_callback(msg)

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._smoother.reset()
        self._left.reset()
        self._right.reset()
        self._fist_since = None
        self._stop_reason = None
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=1,        # full-accuracy hand landmark model
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
                now = time.time()
                hand_lms = result.multi_hand_landmarks

                if not hand_lms:
                    self._fist_since = None
                    self._status("Tracking — no hand")
                elif len(hand_lms) >= 2 and is_fist(hand_lms[0]) and is_fist(hand_lms[1]):
                    # Two fists held -> stop tracking (deliberate gesture).
                    if self._fist_since is None:
                        self._fist_since = now
                    if now - self._fist_since >= TWO_FIST_HOLD:
                        self._stop_reason = "Stopped — two fists"
                        self._stop_event.set()
                        break
                    self._status("Two fists — hold to stop…")
                else:
                    self._fist_since = None
                    self._process_hand(hand_lms[0], now)
        except Exception as exc:  # keep the app alive; report and stop
            self._status(f"Camera error: {exc}")
        finally:
            cap.release()
            hands.close()
            self._status(self._stop_reason or "Stopped")

    def _process_hand(self, landmarks, now):
        lm = landmarks.landmark
        index_tip, thumb = lm[INDEX_TIP], lm[THUMB_TIP]
        index_pip, middle_tip = lm[INDEX_PIP], lm[MIDDLE_TIP]

        sx, sy = map_to_screen(
            index_tip.x, index_tip.y, self._screen_w, self._screen_h, self._margin
        )
        smooth_x, smooth_y = self._smoother.add(sx, sy, now)
        pyautogui.moveTo(smooth_x, smooth_y)

        # Left click: thumb tip touches the index finger's middle joint (PIP), so
        # the pointer (index tip) doesn't move when clicking.
        left_dist = distance(thumb, index_pip)
        right_dist = distance(thumb, middle_tip)

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
