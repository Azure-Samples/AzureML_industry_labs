import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime, timedelta

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--processed_data",  type=str)
parser.add_argument("--seed",            type=int, default=42)
parser.add_argument("--n_years",         type=int, default=3)
args = parser.parse_args()

os.makedirs(args.processed_data, exist_ok=True)


# ── Synthetic data generation ────────────────────────────────────
def generate_synthetic_energy_data(n_years: int, seed: int):
    """
    Generate realistic hourly electricity demand data for a regional utility.

    Patterns modelled:
      - Daily seasonality: morning peak (7-9am), evening peak (5-8pm), overnight trough
      - Weekly seasonality: ~15% lower demand on weekends
      - Annual seasonality: higher in summer (AC) and winter (heating), lower spring/autumn
      - Temperature correlation: U-shaped (extreme temps drive HVAC demand)
      - Upward trend: simulating population growth / electrification
      - Heteroscedastic noise: higher variance during peak hours
    """
    rng = np.random.default_rng(seed)
    n_hours = n_years * 365 * 24
    t = np.arange(n_hours, dtype=np.float64)

    # Base demand (MWh)
    base = 500.0

    # Upward trend — ~2% per year
    trend = 10.0 * (t / (365 * 24))

    # Daily seasonality: two peaks
    hour_of_day = t % 24
    daily = (
        80.0  * np.exp(-0.5 * ((hour_of_day - 8)  / 1.5) ** 2) +   # morning peak
        120.0 * np.exp(-0.5 * ((hour_of_day - 18) / 2.0) ** 2) -   # evening peak
        60.0  * np.exp(-0.5 * ((hour_of_day - 3)  / 2.0) ** 2)     # overnight trough
    )

    # Weekly seasonality: lower on weekends
    day_of_week = (t // 24).astype(int) % 7
    weekly = np.where(day_of_week >= 5, -75.0, 0.0)

    # Annual seasonality: U-shape across months — high summer + winter
    day_of_year = (t // 24).astype(int) % 365
    annual = 60.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)  # peak ~Jan and ~Jul

    # Temperature (Celsius): annual cycle + daily noise
    temp_base = 15.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    temp_daily = 4.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    temperature = temp_base + temp_daily + rng.normal(0, 2.0, n_hours)

    # Temperature-driven demand: U-shape centred at ~18°C (comfort zone)
    temp_effect = 3.0 * (temperature - 18.0) ** 2

    # Heteroscedastic noise: higher during peak hours (6am–10pm)
    noise_scale = np.where((hour_of_day >= 6) & (hour_of_day <= 22), 30.0, 15.0)
    noise = rng.normal(0, 1, n_hours) * noise_scale

    demand = base + trend + daily + weekly + annual + temp_effect + noise
    demand = np.maximum(demand, 50.0)  # floor at 50 MWh

    return t, demand, temperature, hour_of_day, day_of_week, day_of_year


def engineer_features(demand, temperature, hour_of_day, day_of_week, day_of_year):
    """Create feature matrix from raw time series. Returns features and valid mask."""
    n = len(demand)
    month = ((day_of_year / 30.44) % 12).astype(int) + 1
    is_weekend = (day_of_week >= 5).astype(np.float32)

    # Lag features
    lag_1   = np.roll(demand, 1)
    lag_24  = np.roll(demand, 24)
    lag_168 = np.roll(demand, 168)

    # Rolling means
    roll_24  = np.convolve(demand, np.ones(24)  / 24,  mode="full")[:n]
    roll_168 = np.convolve(demand, np.ones(168) / 168, mode="full")[:n]

    # Stack features: [hour, dow, month, is_weekend, temp, lag1, lag24, lag168, roll24, roll168]
    features = np.column_stack([
        hour_of_day.astype(np.float32),
        day_of_week.astype(np.float32),
        month.astype(np.float32),
        is_weekend,
        temperature.astype(np.float32),
        lag_1.astype(np.float32),
        lag_24.astype(np.float32),
        lag_168.astype(np.float32),
        roll_24.astype(np.float32),
        roll_168.astype(np.float32),
    ])

    # Discard first 168 hours (1 week) — lags and rolling windows are invalid
    valid_from = 168
    return features[valid_from:], demand[valid_from:], valid_from


# ── Main ─────────────────────────────────────────────────────────
print(f"\n── Generating synthetic data ({args.n_years} years) ──────")
t, demand, temperature, hour_of_day, day_of_week, day_of_year = \
    generate_synthetic_energy_data(args.n_years, args.seed)

print(f"  Total hours generated: {len(demand)}")
print(f"  Demand range: {demand.min():.1f} – {demand.max():.1f} MWh")
print(f"  Temperature range: {temperature.min():.1f} – {temperature.max():.1f} °C")

print("\n── Engineering features ────────────────────────")
features, targets, valid_from = engineer_features(
    demand, temperature, hour_of_day, day_of_week, day_of_year
)
print(f"  Feature matrix: {features.shape}")
print(f"  Discarded first {valid_from} hours (warm-up)")

# ── Normalise features ───────────────────────────────────────────
feat_mean = features.mean(axis=0)
feat_std  = features.std(axis=0)
feat_std[feat_std == 0] = 1.0  # avoid division by zero
features_normed = (features - feat_mean) / feat_std

tgt_mean = targets.mean()
tgt_std  = targets.std()
targets_normed = (targets - tgt_mean) / tgt_std

# Save normalisation stats for inference
norm_stats = {
    "feat_mean": feat_mean.tolist(),
    "feat_std":  feat_std.tolist(),
    "tgt_mean":  float(tgt_mean),
    "tgt_std":   float(tgt_std),
}
with open(os.path.join(args.processed_data, "norm_stats.json"), "w") as f:
    json.dump(norm_stats, f)
print("  Saved normalisation statistics")

# ── Split: train (60%) / calibration (20%) / test (20%) ─────────
# Chronological split — no shuffling for time series
n = len(targets_normed)
train_end = int(0.6 * n)
cal_end   = int(0.8 * n)

splits = {
    "train":       (features_normed[:train_end],      targets_normed[:train_end]),
    "calibration": (features_normed[train_end:cal_end], targets_normed[train_end:cal_end]),
    "test":        (features_normed[cal_end:],          targets_normed[cal_end:]),
}

print(f"\n── Splits ─────────────────────────────────────")
for split_name, (feat, tgt) in splits.items():
    print(f"  {split_name}: {len(tgt)} samples")

# ── Save .pt tensors per sample ──────────────────────────────────
# Build timestamp strings for each valid sample
start_time = datetime(2022, 1, 1)  # synthetic start date

for split_name, (feat, tgt) in splits.items():
    split_dir = os.path.join(args.processed_data, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Compute the offset into the valid range for this split
    if split_name == "train":
        offset = 0
    elif split_name == "calibration":
        offset = train_end
    else:
        offset = cal_end

    for i in range(len(tgt)):
        global_idx = valid_from + offset + i
        ts = start_time + timedelta(hours=int(global_idx))
        timestamp_str = ts.strftime("%Y-%m-%d_%H")

        torch.save({
            "features":  torch.tensor(feat[i], dtype=torch.float32),
            "target":    torch.tensor(tgt[i], dtype=torch.float32),
            "timestamp": timestamp_str,
        }, os.path.join(split_dir, f"{timestamp_str}.pt"))

    print(f"  Saved {len(tgt)} .pt files to {split_name}/")

print(f"\n── Preprocessing complete ──────────────────────")
print(f"  Train samples:       {len(splits['train'][1])}")
print(f"  Calibration samples: {len(splits['calibration'][1])}")
print(f"  Test samples:        {len(splits['test'][1])}")
