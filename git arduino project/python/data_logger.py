"""
data_logger.py
--------------
Reads LDR sensor values from an Arduino over serial and logs them to a
CSV file with wall-clock timestamps.

Usage:
    python python/data_logger.py --port COM3 --output data/sample_data.csv

On Linux/macOS the port is usually /dev/ttyUSB0 or /dev/ttyACM0.
Press Ctrl+C to stop logging.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

from utils import BAUD_RATE, DATA_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_serial_ports() -> list[str]:
    """Return a list of available serial port names."""
    return [p.device for p in serial.tools.list_ports.comports()]


def parse_line(raw: str) -> int | None:
    """
    Parse a single serial line from the Arduino.

    Expected format: "timestamp_ms,ldr_value"
    The header line ("timestamp_ms,ldr_value") is silently skipped.

    Returns the integer LDR value or None on parse failure.
    """
    line = raw.strip()
    if not line or line.startswith("timestamp_ms"):
        return None
    parts = line.split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------
def run_logger(port: str, output: str, baud: int = BAUD_RATE) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    print(f"[data_logger] Connecting to {port} at {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as exc:
        print(f"[ERROR] Could not open {port}: {exc}")
        print(f"Available ports: {list_serial_ports() or 'none found'}")
        sys.exit(1)

    # Allow Arduino time to reset after serial connection is established
    time.sleep(2)
    print(f"[data_logger] Logging to '{output}'. Press Ctrl+C to stop.\n")

    with open(output_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "ldr_value"])

        try:
            while True:
                try:
                    raw = ser.readline().decode("utf-8", errors="replace")
                except serial.SerialException as exc:
                    print(f"[WARN] Serial read error: {exc}. Retrying...")
                    time.sleep(0.5)
                    continue

                ldr = parse_line(raw)
                if ldr is None:
                    continue

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([ts, ldr])
                csv_file.flush()
                print(f"  {ts}  LDR={ldr:>4}")

        except KeyboardInterrupt:
            print("\n[data_logger] Stopped by user.")
        finally:
            ser.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log LDR sensor data from Arduino to CSV."
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port (e.g. COM3, /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--output",
        default=DATA_PATH,
        help=f"Output CSV file path (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Baud rate (default: {BAUD_RATE})",
    )
    args = parser.parse_args()
    run_logger(port=args.port, output=args.output, baud=args.baud)
