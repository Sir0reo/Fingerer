"""Fingerer — tkinter GUI for finger mouse control."""

import queue
import tkinter as tk

from tracker import (
    FingerMouseTracker,
    SPEED_MIN,
    SPEED_MAX,
    DEFAULT_SPEED,
    SMOOTHING_MIN,
    SMOOTHING_MAX,
    DEFAULT_SMOOTHING,
)


class FingererApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerer")
        self.root.resizable(False, False)

        self.status_queue = queue.Queue()
        self.tracker = FingerMouseTracker(
            status_callback=self.status_queue.put
        )

        tk.Label(root, text="Fingerer", font=("Segoe UI", 16, "bold")).pack(
            padx=40, pady=(20, 10)
        )
        self.toggle_btn = tk.Button(
            root, text="Start", width=14, command=self.toggle
        )
        self.toggle_btn.pack(pady=10)
        self.status_label = tk.Label(root, text="Idle", fg="gray")
        self.status_label.pack(padx=40, pady=(0, 10))

        # Speed slider — how far the cursor moves per hand movement.
        tk.Label(root, text="Speed").pack()
        self.speed_var = tk.IntVar(value=DEFAULT_SPEED)
        tk.Scale(
            root, from_=SPEED_MIN, to=SPEED_MAX, orient="horizontal", length=220,
            variable=self.speed_var, command=self.on_speed,
        ).pack(padx=20)

        # Smoothing slider — higher = steadier cursor, slightly more lag.
        tk.Label(root, text="Smoothing").pack()
        self.smooth_var = tk.IntVar(value=DEFAULT_SMOOTHING)
        tk.Scale(
            root, from_=SMOOTHING_MIN, to=SMOOTHING_MAX, orient="horizontal",
            length=220, variable=self.smooth_var, command=self.on_smooth,
        ).pack(padx=20, pady=(0, 10))

        # Hold toggle — when off, gestures only click (no press-and-hold / drag).
        self.hold_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            root, text="Enable hold (drag)", variable=self.hold_var,
            command=self.on_hold_toggle,
        ).pack(pady=(0, 6))

        tk.Label(
            root, text="Stop gesture: show both hands as fists",
            fg="gray", font=("Segoe UI", 8),
        ).pack(pady=(0, 12))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll_status()

    def on_speed(self, value):
        self.tracker.set_speed(int(float(value)))

    def on_smooth(self, value):
        self.tracker.set_smoothing(int(float(value)))

    def on_hold_toggle(self):
        self.tracker.set_hold_enabled(self.hold_var.get())

    def toggle(self):
        if self.tracker.running:
            self.tracker.stop()
            self.toggle_btn.config(text="Start")
        else:
            self.tracker.start()
            self.toggle_btn.config(text="Stop")

    def poll_status(self):
        try:
            while True:
                msg = self.status_queue.get_nowait()
                self.status_label.config(text=msg)
                if msg.startswith("Stopped") or msg.startswith("Camera error"):
                    self.toggle_btn.config(text="Start")
        except queue.Empty:
            pass
        self.root.after(50, self.poll_status)

    def on_close(self):
        self.tracker.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    FingererApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
