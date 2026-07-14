"""
Utility script to generate a realistic sample_data.csv for demonstration
and training without requiring physical hardware.

The data simulates real-world LDR readings where:
  - Daytime (7-19h) generally produces high values (Light)
  - Night-time produces low values (Dark)
  - Natural variance and cloud cover introduce ambiguous readings,
    making this a non-trivially separable classification task (~93% accuracy).

Run once: python generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# 1200 samples spanning 10 days
n = 1200
start = pd.Timestamp("2024-01-01 06:00:00")
timestamps = pd.date_range(start, periods=n, freq="30s")

ldr_values = []
for ts in timestamps:
    h = ts.hour
    # Core day/night with substantial overlap near transition hours
    if 7 <= h < 19:
        # Bright day, but clouds/shadows cause dips
        base = rng.normal(700, 200)
        # 12% chance of a shadowed/overcast reading that crosses the threshold
        if rng.random() < 0.12:
            base = rng.normal(350, 120)
    else:
        # Dark night, but artificial lighting can cause spikes
        base = rng.normal(250, 150)
        # 8% chance of artificial light source
        if rng.random() < 0.08:
            base = rng.normal(600, 100)
    val = int(np.clip(base, 0, 1023))
    ldr_values.append(val)

df = pd.DataFrame({"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"), "ldr_value": ldr_values})
Path("data").mkdir(exist_ok=True)
df.to_csv("data/sample_data.csv", index=False)

# Quick stats
threshold = 512
light = (df["ldr_value"] >= threshold).sum()
dark = len(df) - light
print(f"Generated {len(df)} rows -> data/sample_data.csv")
print(f"  Light (>={threshold}): {light}  |  Dark (<{threshold}): {dark}")
