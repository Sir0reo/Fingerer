# Fingerer

Control your mouse with your finger via webcam (Windows).

- **Move cursor:** index fingertip (tiny movements are ignored, so the pointer
  stays steady while you hold still or click)
- **Left click / hold:** touch your **thumb** to the **middle joint of your index
  finger**. A quick tap clicks; holding keeps the left button down (so you can drag).
- **Right click / hold:** touch your **middle fingertip** to that same index knuckle.
  Quick tap = right click; hold to keep the right button down.
- **Stop tracking:** show **both hands as fists** for about half a second

Both click gestures touch the index *knuckle* (not the tip), so the pointer stays
put while you click. They trigger before your fingers fully meet, so only a small
movement is needed.

In the window, two sliders tune the feel live:

- **Speed** — how far the cursor moves per hand movement
- **Smoothing** — higher = steadier cursor. Uses an adaptive One Euro Filter, so it
  stays responsive (low lag) even at high smoothing — raise it if the cursor jitters.

## Requirements

- Windows, Python 3.11
- A webcam

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python app.py
```

Click **Start** to begin tracking, **Stop** to end. Slam your mouse into a
screen corner (pyautogui fail-safe) to abort instantly if needed.

## Run tests

```
pytest tests/ -v
```

## Package as a .exe (PyInstaller)

```
pyinstaller --onefile --windowed --name Fingerer --collect-all mediapipe app.py
```

- `--onefile` — single executable.
- `--windowed` — no console window.
- `--collect-all mediapipe` — bundles MediaPipe's model assets (required, or the
  built `.exe` will fail to find hand-tracking models at runtime).

The executable is created at `dist\Fingerer.exe` (~240 MB). First launch may be
slow while MediaPipe initializes. You can double-click it directly or pin it /
make a Desktop shortcut to launch Fingerer like any other app — no Python needed.
