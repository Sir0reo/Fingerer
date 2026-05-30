# Fingerer

Control your mouse with your finger via webcam (Windows).

- **Move cursor:** index fingertip
- **Left click:** touch your thumb tip to the **middle joint of your index finger**
  (keeps the pointer steady — touching the index *tip* would move the cursor)
- **Right click:** touch your thumb tip to your **middle fingertip**
- **Stop tracking:** show **both hands as fists** for about half a second

In the window, two sliders tune the feel live:

- **Speed** — how far the cursor moves per hand movement
- **Smoothing** — higher = steadier cursor, slightly more lag (raise this if it's jittery)

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
