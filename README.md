# Fingerer

Control your mouse with your finger via webcam (Windows).

- **Move cursor:** index fingertip
- **Left click:** pinch thumb + index finger
- **Right click:** pinch thumb + middle finger

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

The executable is created at `dist\Fingerer.exe`. First launch may be slow while
MediaPipe initializes.
