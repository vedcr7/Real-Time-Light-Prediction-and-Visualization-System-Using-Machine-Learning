"""
realtime_gui.py
---------------
Real-time LDR state prediction and visualization GUI.

  • Reads live serial data from Arduino.
  • Classifies each reading as Light or Dark using the trained model.
  • Displays an animated scatter plot (Matplotlib embedded in Tkinter).
  • Shows a scrolling prediction log and current sensor stats.

Usage:
    python python/realtime_gui.py --port COM3

Press the "Stop" button or close the window to exit cleanly.
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import scrolledtext, ttk

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import BAUD_RATE, MODEL_PATH, build_realtime_features

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PLOT_POINTS = 100   # Keep only the last N points on the scatter plot
LOG_MAX_LINES   = 500   # Limit log widget line count
POLL_INTERVAL   = 50    # GUI refresh rate in milliseconds


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------
class SerialReader(threading.Thread):
    """Background thread that reads from Arduino and puts parsed values into a queue."""

    def __init__(self, port: str, baud: int, data_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.port        = port
        self.baud        = baud
        self.data_queue  = data_queue
        self.stop_event  = stop_event
        self.ser         = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=2)
            time.sleep(2)  # Let Arduino reset
        except serial.SerialException as exc:
            self.data_queue.put(("error", str(exc)))
            return

        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline().decode("utf-8", errors="replace").strip()
            except serial.SerialException as exc:
                self.data_queue.put(("error", str(exc)))
                break

            if not raw or raw.startswith("timestamp_ms"):
                continue

            parts = raw.split(",")
            if len(parts) != 2:
                continue
            try:
                ldr_value = int(parts[1])
                self.data_queue.put(("data", ldr_value))
            except ValueError:
                continue

        if self.ser and self.ser.is_open:
            self.ser.close()

    def stop(self):
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------
class LDRApp(tk.Tk):
    def __init__(self, port: str, baud: int, model_path: str):
        super().__init__()

        self.title("Real-Time LDR Light State Predictor")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Load model ----
        if not Path(model_path).exists():
            tk.messagebox.showerror(
                "Model Not Found",
                f"Model file '{model_path}' not found.\n"
                "Run train_model.py first to generate it.",
            )
            self.destroy()
            return

        self.model    = joblib.load(model_path)
        self.port     = port
        self.baud     = baud

        # ---- Internal state ----
        self.data_queue  : queue.Queue          = queue.Queue()
        self.stop_event  : threading.Event      = threading.Event()
        self.ldr_values  : list[int]            = []
        self.predictions : list[int]            = []
        self.timestamps  : list[datetime]       = []
        self.t0          : datetime | None      = None
        self.running     : bool                 = False

        self._build_ui()
        self._start_serial(port, baud)
        self._schedule_poll()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Top info bar ----
        info_frame = ttk.Frame(self, padding=8)
        info_frame.pack(fill="x", side="top")

        ttk.Label(info_frame, text="Port:").grid(row=0, column=0, sticky="w")
        self.lbl_port = ttk.Label(info_frame, text=self.port, foreground="steelblue")
        self.lbl_port.grid(row=0, column=1, padx=(4, 20), sticky="w")

        ttk.Label(info_frame, text="LDR Value:").grid(row=0, column=2, sticky="w")
        self.lbl_ldr = ttk.Label(info_frame, text="--", width=6, font=("Helvetica", 11, "bold"))
        self.lbl_ldr.grid(row=0, column=3, padx=(4, 20), sticky="w")

        ttk.Label(info_frame, text="State:").grid(row=0, column=4, sticky="w")
        self.lbl_state = ttk.Label(
            info_frame, text="--", width=6, font=("Helvetica", 11, "bold")
        )
        self.lbl_state.grid(row=0, column=5, padx=(4, 20), sticky="w")

        ttk.Label(info_frame, text="Readings:").grid(row=0, column=6, sticky="w")
        self.lbl_count = ttk.Label(info_frame, text="0")
        self.lbl_count.grid(row=0, column=7, padx=(4, 20), sticky="w")

        self.btn_stop = ttk.Button(info_frame, text="Stop", command=self._on_close)
        self.btn_stop.grid(row=0, column=8, padx=4)

        # ---- Main paned layout ----
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        # ---- Scatter plot ----
        plot_frame = ttk.LabelFrame(paned, text="Live Scatter Plot", padding=4)
        paned.add(plot_frame, weight=3)

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.fig.patch.set_facecolor("#1e1e2e")
        self.ax.set_facecolor("#2a2a3e")
        self.ax.set_title("LDR State Predictions", color="white")
        self.ax.set_xlabel("Reading #", color="white")
        self.ax.set_ylabel("LDR Value (0–1023)", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555")
        self.ax.axhline(512, color="#ffaa00", linewidth=1, linestyle="--", label="Threshold (512)")
        self.ax.legend(facecolor="#333", labelcolor="white", fontsize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ---- Log panel ----
        log_frame = ttk.LabelFrame(paned, text="Prediction Log", padding=4)
        paned.add(log_frame, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, state="disabled", wrap="none",
            font=("Courier", 9), bg="#1e1e2e", fg="#cdd6f4",
            insertbackground="white",
        )
        self.log_text.pack(fill="both", expand=True)

        # Tag colours for log entries
        self.log_text.tag_config("light", foreground="#a6e3a1")  # green
        self.log_text.tag_config("dark",  foreground="#f38ba8")  # red
        self.log_text.tag_config("info",  foreground="#89dceb")  # cyan
        self.log_text.tag_config("error", foreground="#fab387")  # orange

        self._log("GUI started. Waiting for serial data...", "info")

    # ------------------------------------------------------------------
    # Serial setup
    # ------------------------------------------------------------------
    def _start_serial(self, port: str, baud: int):
        self.reader = SerialReader(port, baud, self.data_queue, self.stop_event)
        self.reader.start()
        self.running = True
        self._log(f"Connecting to {port} at {baud} baud...", "info")

    # ------------------------------------------------------------------
    # Poll queue (runs on GUI thread via after())
    # ------------------------------------------------------------------
    def _schedule_poll(self):
        self._poll()

    def _poll(self):
        processed = 0
        while not self.data_queue.empty() and processed < 20:
            item = self.data_queue.get_nowait()
            kind, value = item

            if kind == "error":
                self._log(f"Serial error: {value}", "error")
                self.lbl_state.config(text="ERR", foreground="orange")
            elif kind == "data":
                self._handle_reading(value)

            processed += 1

        if self.running:
            self.after(POLL_INTERVAL, self._poll)

    # ------------------------------------------------------------------
    # Process a single LDR reading
    # ------------------------------------------------------------------
    def _handle_reading(self, ldr_value: int):
        now = datetime.now()

        if self.t0 is None:
            self.t0 = now
            self._log("First reading received. Prediction active.", "info")

        # Build feature vector and predict
        features = build_realtime_features(ldr_value, pd.Timestamp(now), pd.Timestamp(self.t0))
        prediction = int(self.model.predict(features)[0])
        label = "Light" if prediction == 1 else "Dark"
        colour = "#a6e3a1" if prediction == 1 else "#f38ba8"
        tag = "light" if prediction == 1 else "dark"

        # Update state
        self.ldr_values.append(ldr_value)
        self.predictions.append(prediction)
        self.timestamps.append(now)

        # Update info labels
        self.lbl_ldr.config(text=str(ldr_value))
        self.lbl_state.config(text=label, foreground=colour)
        self.lbl_count.config(text=str(len(self.ldr_values)))

        # Log entry
        ts_str = now.strftime("%H:%M:%S")
        self._log(f"[{ts_str}]  LDR={ldr_value:>4}  →  {label}", tag)

        # Refresh plot
        self._update_plot()

    # ------------------------------------------------------------------
    # Plot update
    # ------------------------------------------------------------------
    def _update_plot(self):
        # Trim to last MAX_PLOT_POINTS
        values  = self.ldr_values[-MAX_PLOT_POINTS:]
        preds   = self.predictions[-MAX_PLOT_POINTS:]
        indices = list(range(len(self.ldr_values) - len(values), len(self.ldr_values)))

        self.ax.cla()
        self.ax.set_facecolor("#2a2a3e")
        self.ax.set_title("LDR State Predictions (last 100)", color="white")
        self.ax.set_xlabel("Reading #", color="white")
        self.ax.set_ylabel("LDR Value (0–1023)", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555")
        self.ax.set_ylim(-20, 1050)

        # Threshold line
        self.ax.axhline(512, color="#ffaa00", linewidth=1, linestyle="--", label="Threshold (512)")

        # Scatter: green = Light, red = Dark
        preds_arr  = np.array(preds)
        values_arr = np.array(values)
        idx_arr    = np.array(indices)

        light_mask = preds_arr == 1
        dark_mask  = preds_arr == 0

        if light_mask.any():
            self.ax.scatter(
                idx_arr[light_mask], values_arr[light_mask],
                color="#a6e3a1", s=30, label="Light", zorder=3,
            )
        if dark_mask.any():
            self.ax.scatter(
                idx_arr[dark_mask], values_arr[dark_mask],
                color="#f38ba8", s=30, label="Dark", zorder=3,
            )

        self.ax.legend(facecolor="#333", labelcolor="white", fontsize=8)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Prediction log helper
    # ------------------------------------------------------------------
    def _log(self, message: str, tag: str = "info"):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n", tag)

        # Prune old lines
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > LOG_MAX_LINES:
            self.log_text.delete("1.0", f"{line_count - LOG_MAX_LINES}.0")

        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------
    def _on_close(self):
        self.running = False
        self.stop_event.set()
        self.after(300, self.destroy)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Real-time LDR light state predictor GUI."
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port of the Arduino (e.g. COM3, /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Baud rate (default: {BAUD_RATE})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Path to trained model pickle (default: {MODEL_PATH})",
    )
    args = parser.parse_args()

    app = LDRApp(port=args.port, baud=args.baud, model_path=args.model)
    app.mainloop()


if __name__ == "__main__":
    main()
