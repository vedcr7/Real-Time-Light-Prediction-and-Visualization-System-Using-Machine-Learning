"""
utils.py
--------
Shared utility functions for feature engineering, label encoding,
and serial port helpers used across the pipeline.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIGHT_THRESHOLD = 512   # LDR values above this are classified as "Light"
BAUD_RATE       = 9600
MODEL_PATH      = "python/model.pkl"
DATA_PATH       = "data/sample_data.csv"


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive timestamp-based features from a DataFrame that contains a
    'timestamp' column (datetime or string parseable by pandas).

    Added columns:
        hour         – hour of day (0-23)
        day          – day of month (1-31)
        month        – month of year (1-12)
        day_of_week  – Monday=0, Sunday=6
        elapsed_time – seconds since the first reading in the dataset
        ldr_value    – kept as-is (passed through)

    Returns a DataFrame with only the model-input columns.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"]        = df["timestamp"].dt.hour
    df["day"]         = df["timestamp"].dt.day
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    t0 = df["timestamp"].min()
    df["elapsed_time"] = (df["timestamp"] - t0).dt.total_seconds()

    feature_cols = ["ldr_value", "hour", "day", "month", "day_of_week", "elapsed_time"]
    return df[feature_cols]


def encode_labels(df: pd.DataFrame, threshold: int = LIGHT_THRESHOLD) -> pd.Series:
    """
    Encode LDR states as binary labels.
        Light (ldr_value >= threshold) -> 1
        Dark  (ldr_value <  threshold) -> 0
    """
    return (df["ldr_value"] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Feature vector for a single real-time reading
# ---------------------------------------------------------------------------
def build_realtime_features(ldr_value: int, timestamp: pd.Timestamp,
                             t0: pd.Timestamp) -> np.ndarray:
    """
    Build a single-row feature array for real-time inference.

    Parameters
    ----------
    ldr_value : raw ADC reading (0-1023)
    timestamp : current reading timestamp
    t0        : reference start time (first reading of the session)

    Returns
    -------
    np.ndarray of shape (1, 6)
    """
    hour        = timestamp.hour
    day         = timestamp.day
    month       = timestamp.month
    day_of_week = timestamp.dayofweek
    elapsed     = (timestamp - t0).total_seconds()

    return np.array([[ldr_value, hour, day, month, day_of_week, elapsed]])


FEATURE_NAMES = ["ldr_value", "hour", "day", "month", "day_of_week", "elapsed_time"]
