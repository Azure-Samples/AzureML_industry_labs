"""
Generate a sample inference payload for the energy-forecast-batch endpoint.

This script creates a folder of .pt files representing 24 hours of hourly
demand forecasting inputs (a typical next-day forecast scenario). Features
are normalised using norm_stats.json from the training pipeline.

Usage:
    python generate_sample_payload.py --norm-stats <path-to-norm_stats.json>

    If --norm-stats is not provided, the script uses default normalisation
    values derived from the synthetic training data.
"""

import os
import json
import argparse
import torch
import numpy as np
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description="Generate sample inference payload")
parser.add_argument(
    "--norm-stats",
    type=str,
    default=None,
    help="Path to norm_stats.json from the training pipeline. "
         "If omitted, uses approximate defaults from the synthetic dataset.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="sample_payload",
    help="Output directory for .pt files (default: sample_payload/)",
)
parser.add_argument(
    "--start-date",
    type=str,
    default="2024-07-15",
    help="Start date for the sample forecast (YYYY-MM-DD, default: 2024-07-15)",
)
args = parser.parse_args()

# ── Load or define normalisation stats ───────────────────────────
if args.norm_stats and os.path.exists(args.norm_stats):
    with open(args.norm_stats) as f:
        stats = json.load(f)
    feat_mean = np.array(stats["feat_mean"], dtype=np.float32)
    feat_std = np.array(stats["feat_std"], dtype=np.float32)
    print(f"Loaded normalisation stats from {args.norm_stats}")
else:
    # Approximate stats from the 3-year synthetic dataset
    # These are close enough for a demo payload
    feat_mean = np.array([11.5, 3.0, 6.5, 0.286, 15.0, 600.0, 600.0, 600.0, 600.0, 600.0], dtype=np.float32)
    feat_std = np.array([6.92, 2.0, 3.45, 0.452, 10.0, 150.0, 150.0, 150.0, 150.0, 150.0], dtype=np.float32)
    print("Using approximate default normalisation stats (pass --norm-stats for exact values)")

os.makedirs(args.output_dir, exist_ok=True)

# ── Generate 24 hours of realistic sample data ──────────────────
start = datetime.strptime(args.start_date, "%Y-%m-%d")
rng = np.random.default_rng(123)

# Simulate a summer day demand profile (MWh)
hourly_demand_profile = [
    420, 390, 370, 360, 365, 400, 480, 580,  # 00:00–07:00
    650, 680, 670, 660, 650, 640, 630, 650,  # 08:00–15:00
    700, 750, 780, 740, 680, 600, 520, 460,  # 16:00–23:00
]

print(f"\nGenerating 24 sample .pt files for {args.start_date}...")
print(f"Scenario: Summer weekday with typical demand profile\n")

for hour in range(24):
    ts = start + timedelta(hours=hour)
    demand = hourly_demand_profile[hour] + rng.normal(0, 15)

    # Raw features (unnormalised)
    raw_features = np.array([
        float(ts.hour),                          # hour
        float(ts.weekday()),                     # day_of_week (0=Mon)
        float(ts.month),                         # month
        float(1 if ts.weekday() >= 5 else 0),    # is_weekend
        28.0 + 5 * np.sin(2 * np.pi * (ts.hour - 14) / 24) + rng.normal(0, 1),  # temperature (summer)
        demand + rng.normal(0, 10),              # demand_lag_1 (approx)
        demand * 0.98 + rng.normal(0, 15),       # demand_lag_24
        demand * 0.95 + rng.normal(0, 20),       # demand_lag_168
        demand * 1.01 + rng.normal(0, 8),        # demand_roll_mean_24
        demand * 0.99 + rng.normal(0, 10),       # demand_roll_mean_168
    ], dtype=np.float32)

    # Normalise
    normalised = (raw_features - feat_mean) / feat_std

    timestamp_str = ts.strftime("%Y-%m-%d_%H")
    payload = {
        "features": torch.tensor(normalised, dtype=torch.float32),
        "timestamp": timestamp_str,
    }

    filepath = os.path.join(args.output_dir, f"{timestamp_str}.pt")
    torch.save(payload, filepath)
    print(f"  {timestamp_str}  |  temp={raw_features[4]:.1f}°C  |  lag_1={raw_features[5]:.0f} MWh")

print(f"\n✅ Saved {24} files to {args.output_dir}/")
print(f"\nNext steps:")
print(f"  1. Upload:  az ml data create --name energy-inference-input --type uri_folder --path {args.output_dir}/")
print(f"  2. Invoke:  az ml batch-endpoint invoke --name energy-forecast-batch --input azureml:energy-inference-input:1")
print(f"     (Replace :1 with the version number returned by the data create command)")
