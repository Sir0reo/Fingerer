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
# Per-gesture click thresholds (normalized distance). Hysteresis (release > press)
# avoids flicker.
# Left is intentionally generous so clicking takes little effort. It's gated on the
# thumb being bent *toward* the index knuckle (is_thumb_pointing_at), so the generous
# threshold doesn't misfire when the thumb just rests nearby — and straightening the
# thumb flips that gate, releasing instantly without needing a big movement.
# NOTE: release MUST stay above threshold (proper hysteresis); a release below the
# threshold makes it engage-then-release every frame and rapid-fire clicks.
LEFT_THRESHOLD = 0.30    # thumb bent toward the index knuckle (very easy)
LEFT_RELEASE = 0.35      # small band; the pointing gate also releases on straighten
# Right is intentionally strict (near-contact) and additionally gated on the middle
# fingertip being on top of the index fingertip — together these stop misfires.
RIGHT_THRESHOLD = 0.04   # middle fingertip on top of the index fingertip (strict)
RIGHT_RELEASE = 0.06

# A contact shorter than this is a single click; longer becomes a press-and-hold (drag).
HOLD_DELAY = 0.5         # seconds of sustained contact before a hold engages
# After a hold releases, suppress any click for this long so letting go of a hold
# never fires a stray click right after.
CLICK_COOLDOWN = 0.3     # seconds
CAM_INDEX = 0            # default webcam index
CAM_WIDTH, CAM_HEIGHT = 1280, 720   # request higher-res frames for better accuracy
CURSOR_DEADZONE = 8      # px; ignore cursor moves smaller than this (kills micro-jitter)

# Speed slider -> active-region margin (higher speed = larger margin = faster cursor)
SPEED_MIN, SPEED_MAX = 1, 10
DEFAULT_SPEED = 4        # default cursor sensitivity (gain)
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
THUMB_IP = 3             # thumb middle knuckle (the joint that bends to curl the tip)
THUMB_TIP = 4
INDEX_PIP = 6            # index finger middle joint (left-click target)
INDEX_TIP = 8
MIDDLE_TIP = 12
# Fingertip / PIP pairs for fist detection (index, middle, ring, pinky)
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)


def disable_background_throttling():
    """Best-effort: stop Windows from throttling this process when its window is
    not focused, so cursor tracking stays smooth after switching to another app.

    Windows 11 applies power throttling (EcoQoS) and lower scheduling priority to
    background processes; that starves the capture/tracking thread and makes the
    cursor lag once Fingerer loses focus. We opt out of execution-speed throttling
    and bump the priority class. No-op on non-Windows or if the calls fail.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetCurrentProcess()

        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS

        state = _PowerThrottlingState(
            Version=PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask=0,  # 0 = disable throttling (always run at full speed)
        )
        kernel32.SetProcessInformation(
            handle, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state)
        )
    except Exception:
        pass  # purely an optimization; never let it break startup


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


def is_thumb_pointing_at(thumb_tip, thumb_ip, target):
    """Return True if the thumb is bent so its tip points toward `target`.

    True when the thumb tip is closer to the target than the thumb's middle
    knuckle (IP joint) is — i.e. the last thumb segment angles toward the target.
    This lets the left click fire on a small, natural thumb bend pointing at the
    index knuckle, instead of requiring the whole thumb to reach over and touch.
    """
    return distance(thumb_tip, target) < distance(thumb_ip, target)


def is_middle_over_index(middle_tip, index_tip):
    """Return True if the middle fingertip sits on top of the index fingertip.

    "On top" means higher in the image — a smaller y. Requiring this makes the
    right-click gesture deliberate (the middle finger must be placed over the
    index fingertip) instead of firing whenever the two tips drift near each
    other, which caused false right-clicks.
    """
    return middle_tip.y < index_tip.y


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


class Deadzone:
    """Suppresses tiny cursor movement so micro-jitter is ignored.

    Returns the position to move to, or None when the new position is within
    `radius` pixels of the last committed position (meaning: don't move). The
    reference position only updates on a committed move, so the cursor stays
    perfectly still until a movement larger than the radius occurs.
    """

    def __init__(self, radius=CURSOR_DEADZONE):
        self.radius = radius
        self._last = None

    def filter(self, x, y):
        if self._last is None or math.hypot(x - self._last[0], y - self._last[1]) >= self.radius:
            self._last = (x, y)
            return self._last
        return None

    def reset(self):
        self._last = None


class GestureClicker:
    """Turns two gesture distances into clicks and delayed press-and-holds.

    A gesture going below its press threshold starts a contact. If it releases
    (opens past its release threshold) before `hold_delay` seconds, it fires a
    single click. If it stays in contact past `hold_delay`, the button is pressed
    and held down (for dragging) until release. At most one gesture is active at
    a time (the most-engaged one wins). Hysteresis (release > press) avoids
    flicker.

    Each callback takes a button name ("left"/"right"):
      click_cb   — a single click (a quick tap)
      press_cb   — mouse button down (a hold begins)
      release_cb — mouse button up (a hold ends)

    `update()` returns None or a (kind, button) tuple where kind is "contact",
    "click", or "hold", for status display.
    """

    def __init__(self, click_cb, press_cb, release_cb,
                 left_threshold=LEFT_THRESHOLD, left_release=LEFT_RELEASE,
                 right_threshold=RIGHT_THRESHOLD, right_release=RIGHT_RELEASE,
                 hold_delay=HOLD_DELAY, click_cooldown=CLICK_COOLDOWN):
        self._click = click_cb
        self._press = press_cb
        self._release = release_cb
        self._threshold = {"left": left_threshold, "right": right_threshold}
        self._release_threshold = {"left": left_release, "right": right_release}
        self._hold_delay = hold_delay
        self._click_cooldown = click_cooldown
        self._hold_enabled = True    # when False, gestures only ever single-click
        self._active = None         # gesture currently in contact: None|"left"|"right"
        self._contact_since = None
        self._holding = False       # button currently held down
        self._cooldown_until = 0.0  # clicks suppressed until this time (post-hold)

    def set_hold_enabled(self, enabled):
        """Enable/disable press-and-hold. When disabled, every contact is a click
        (no dragging), no matter how long it's held."""
        self._hold_enabled = bool(enabled)

    @property
    def active(self):
        return self._active

    @property
    def holding(self):
        return self._holding

    def update(self, left_dist, right_dist, now):
        """Advance one frame; return None or (kind, button) for status."""
        dists = {"left": left_dist, "right": right_dist}
        if self._active is None:
            candidates = []
            for button, dist in dists.items():
                if dist < self._threshold[button]:
                    candidates.append((dist / self._threshold[button], button))
            if candidates:
                candidates.sort()              # most-engaged gesture wins
                self._active = candidates[0][1]
                self._contact_since = now
                self._holding = False
                return ("contact", self._active)
            return None

        if dists[self._active] > self._release_threshold[self._active]:
            button, was_holding = self._active, self._holding
            self._active = None
            self._contact_since = None
            self._holding = False
            if was_holding:
                self._release(button)          # end of a hold
                self._cooldown_until = now + self._click_cooldown
                return None
            if now < self._cooldown_until:     # too soon after a hold -> swallow click
                return None
            self._click(button)                # short contact -> a click
            return ("click", button)

        if (self._hold_enabled and not self._holding
                and now - self._contact_since >= self._hold_delay):
            self._holding = True
            self._press(self._active)          # contact held long enough -> hold
        return ("hold" if self._holding else "contact", self._active)

    def release_all(self):
        """Release a held button on stop. A pending (un-held) contact is dropped
        without firing a click."""
        if self._active is not None and self._holding:
            self._release(self._active)
        self._active = None
        self._contact_since = None
        self._holding = False


class FingerMouseTracker:
    """Runs webcam hand tracking on a background thread and drives the mouse."""

    def __init__(self, status_callback=None, cam_index=CAM_INDEX):
        self._status_callback = status_callback or (lambda s: None)
        self._cam_index = cam_index
        self._thread = None
        self._stop_event = threading.Event()
        self._screen_w, self._screen_h = pyautogui.size()
        self._smoother = CursorSmoother(smoothing_to_cutoff(DEFAULT_SMOOTHING))
        self._deadzone = Deadzone(CURSOR_DEADZONE)
        self._clicker = GestureClicker(
            click_cb=lambda b: pyautogui.click(button=b),
            press_cb=lambda b: pyautogui.mouseDown(button=b),
            release_cb=lambda b: pyautogui.mouseUp(button=b),
        )
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

    def set_hold_enabled(self, enabled):
        """Enable/disable press-and-hold (drag). When off, gestures only click."""
        self._clicker.set_hold_enabled(enabled)

    def _status(self, msg):
        self._status_callback(msg)

    def start(self):
        if self.running:
            return
        disable_background_throttling()   # keep tracking smooth when unfocused
        self._stop_event.clear()
        self._smoother.reset()
        self._deadzone.reset()
        self._clicker.release_all()
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
            self._clicker.release_all()   # never leave a mouse button stuck down
            cap.release()
            hands.close()
            self._status(self._stop_reason or "Stopped")

    def _process_hand(self, landmarks, now):
        lm = landmarks.landmark
        index_tip, thumb = lm[INDEX_TIP], lm[THUMB_TIP]
        index_pip, middle_tip = lm[INDEX_PIP], lm[MIDDLE_TIP]
        thumb_ip = lm[THUMB_IP]

        sx, sy = map_to_screen(
            index_tip.x, index_tip.y, self._screen_w, self._screen_h, self._margin
        )
        smooth_x, smooth_y = self._smoother.add(sx, sy, now)
        pos = self._deadzone.filter(smooth_x, smooth_y)
        if pos is not None:   # skip sub-deadzone moves so tiny jitter is ignored
            pyautogui.moveTo(*pos)

        # Left = thumb bent so its tip points at the index knuckle (PIP); the pointer
        # (index tip) stays put. Gated on the thumb pointing at the knuckle so the
        # generous threshold fires on a small natural bend, not a big reach.
        left_dist = (
            distance(thumb, index_pip)
            if is_thumb_pointing_at(thumb, thumb_ip, index_pip)
            else float("inf")
        )
        # right = middle fingertip placed on top of the index fingertip.
        # Right click only counts when the middle tip is on top of the index tip;
        # otherwise force it out of range so it can never engage (kills misfires).
        right_dist = (
            distance(middle_tip, index_tip)
            if is_middle_over_index(middle_tip, index_tip)
            else float("inf")
        )
        event = self._clicker.update(left_dist, right_dist, now)
        if event is None:
            self._status("Tracking — hand detected")
        else:
            kind, button = event
            if kind == "click":
                self._status(f"Tracking — {button} click")
            elif kind == "hold":
                self._status(f"Tracking — {button} hold")
            else:  # contact, pre-hold
                self._status("Tracking — hand detected")
