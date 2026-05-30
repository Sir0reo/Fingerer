# Fingerer — Finger Mouse Control (Design)

Date: 2026-05-30
Status: Approved (pending spec review)

## Summary

A Windows Python desktop app that lets the user control the mouse cursor with
their index finger via webcam, using MediaPipe hand tracking. A small tkinter
GUI provides a Start/Stop toggle and a status label. The webcam/tracking loop
runs on a background thread so the GUI stays responsive.

## Goals

- Move the cursor with the index fingertip (MediaPipe landmark 8).
- Left click on a thumb (4) + index (8) pinch.
- Right click on a thumb (4) + middle (12) pinch.
- Smooth cursor motion (rolling average of the last 5 frames).
- Responsive GUI (tracking on a background thread).
- Ship with `requirements.txt` and PyInstaller `.exe` packaging instructions.

## Non-Goals (YAGNI)

- Multi-hand support, scrolling, drag, double-click, gesture configuration UI.
- Cross-platform support (Windows-only target).
- Persisting settings to disk.

## Project Layout

```
C:\Users\isaac\Fingerer\
├── app.py            # tkinter GUI + background-thread ownership
├── tracker.py        # FingerMouseTracker: capture, MediaPipe, mapping, clicks
├── requirements.txt
├── README.md         # run + PyInstaller packaging instructions
└── docs/superpowers/specs/2026-05-30-fingerer-design.md
```

## Architecture & Data Flow

```
[Webcam thread]  capture → MediaPipe Hands → landmarks
      │
      ├─ landmark 8 (index tip) → map active region → smooth (5-frame avg) → pyautogui.moveTo
      ├─ dist(4, 8) < threshold → LEFT click  (own cooldown + release latch)
      ├─ dist(4, 12) < threshold → RIGHT click (own cooldown + release latch)
      │
      └─ status string → queue.Queue → GUI label (polled via root.after)
```

## Component: `tracker.py`

Exposes a `FingerMouseTracker` class so the GUI never touches OpenCV/MediaPipe.

### Public interface

- `FingerMouseTracker(status_callback)` — `status_callback(str)` is invoked
  (from the worker thread) whenever the status changes.
- `start()` — opens the camera and starts the worker thread. Idempotent.
- `stop()` — signals the loop to stop, joins the thread, releases the camera.
  Idempotent and safe to call from the GUI/main thread.
- `running` (bool property) — current state.

### Internals

- **Worker thread** (`threading.Thread`) running the capture loop; a
  `threading.Event` (`_stop_event`) signals it to exit. On exit the loop
  releases the `cv2.VideoCapture` and calls `status_callback("Stopped")`.
- **MediaPipe Hands** configured for a single hand
  (`max_num_hands=1`, `min_detection_confidence=0.7`, `min_tracking_confidence=0.7`).
- **Mirroring** — frame is flipped horizontally (`cv2.flip(frame, 1)`) so moving
  the hand right moves the cursor right.
- **Active region with margin** — landmark 8's normalized (x, y) is taken within
  an inset rectangle of the frame (`MARGIN` fraction on each side) and mapped
  (via `numpy.interp`) to the full screen so the user can reach every edge.
- **Smoothing** — two `collections.deque(maxlen=SMOOTHING)` (SMOOTHING = 5)
  holding mapped x and y; the cursor target is their mean each frame.
- **Cursor move** — `pyautogui.moveTo(x, y)` with `pyautogui.PAUSE = 0`.
- **Clicks** — for each gesture, distance between the two fingertips is computed
  in normalized landmark space. Below `PINCH_THRESHOLD` it fires once
  (`pyautogui.click()` / `pyautogui.click(button="right")`), then is *latched*:
  it will not fire again until the fingers separate above the threshold, and a
  `CLICK_COOLDOWN` (~0.3s) minimum interval is enforced. If both gestures read
  below threshold on the same frame, **left wins** (right is suppressed that
  frame) to avoid ambiguity.
- **Status strings** — e.g. `"Tracking — hand detected"`, `"Tracking — no hand"`,
  `"Camera error: <detail>"`, `"Stopped"`. Pushed via `status_callback`.

### Tunable constants (module-level)

```python
SMOOTHING = 5            # rolling-average window (frames)
MARGIN = 0.15            # inset fraction of frame edges for the active region
PINCH_THRESHOLD = 0.05   # normalized fingertip distance for a pinch
CLICK_COOLDOWN = 0.3     # seconds between clicks of the same button
CAM_INDEX = 0            # default webcam
```

## Component: `app.py` (tkinter GUI)

- Single resizable-disabled window titled "Fingerer".
- A **Start/Stop** toggle `Button` and a **status** `Label`.
- Owns a `FingerMouseTracker`. Button toggles `start()`/`stop()` and updates its
  own text ("Start" ↔ "Stop").
- **Thread-safe status:** the tracker's `status_callback` puts strings on a
  `queue.Queue`; the GUI drains it on a `root.after(50, poll)` loop and updates
  the label. The worker thread never touches tkinter widgets directly.
- **Window close** (`WM_DELETE_WINDOW`) calls `tracker.stop()` then destroys the
  window so the camera is always released.

## Error Handling

- **Camera won't open** (`cap.isOpened()` false) → push `"Camera error: ..."`,
  the worker exits, and the GUI resets the button to "Start".
- **pyautogui fail-safe** left **enabled** (slam cursor to a screen corner to
  abort) as a manual kill switch.
- `stop()` is idempotent; window close always tears down cleanly even if the
  thread already exited.
- Exceptions inside the loop are caught, reported via status, and stop the loop
  rather than crashing the app.

## Dependencies (`requirements.txt`)

- `opencv-python`
- `mediapipe`
- `pyautogui`
- `numpy`
- `pyinstaller` (for packaging; dev-time only)

tkinter ships with the standard CPython Windows installer (no pip needed).
Targeting Python 3.11 (3.11.9 confirmed installed), which MediaPipe supports.

## Packaging (README)

PyInstaller one-file build, collecting MediaPipe's data files:

```
pyinstaller --onefile --windowed --name Fingerer ^
  --collect-all mediapipe app.py
```

Output: `dist\Fingerer.exe`. README documents this, the `--collect-all mediapipe`
necessity (model assets), and the `--windowed` flag (no console window).

## Testing / Verification

This is hardware-driven (webcam + live cursor), so verification is primarily
manual:

1. `pip install -r requirements.txt`, then `python app.py` launches the GUI.
2. Click **Start** → status shows tracking; index finger moves the cursor.
3. Thumb+index pinch → single left click; thumb+middle pinch → single right
   click; clicks do not machine-gun.
4. Click **Stop** → camera light turns off, cursor control ends.
5. Closing the window releases the camera.

Pure-logic helpers that don't need hardware (e.g. distance, region mapping,
smoothing average) are written as small free functions so they *could* be unit
tested, but automated tests are out of scope for this build.
