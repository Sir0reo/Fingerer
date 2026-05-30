"""Fingerer — tkinter GUI for finger mouse control."""

import queue
import tkinter as tk

from tracker import FingerMouseTracker


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
        self.status_label.pack(padx=40, pady=(0, 20))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.poll_status()

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
                if msg == "Stopped" or msg.startswith("Camera error"):
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
