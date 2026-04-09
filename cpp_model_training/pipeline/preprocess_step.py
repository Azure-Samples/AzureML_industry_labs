import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--processed_data", type=str)
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--n_samples",     type=int, default=1000)
args = parser.parse_args()

os.makedirs(args.processed_data, exist_ok=True)

# ── Generate synthetic data ──────────────────────────────────────
# True relationship: y = 3*x1 + 1.5*x2 - 2*x3 + 7 + noise
TRUE_WEIGHTS = [3.0, 1.5, -2.0]
TRUE_BIAS = 7.0
NOISE_STD = 0.5

print(f"\n── Generating synthetic data ───────────────────")
print(f"  Samples:     {args.n_samples}")
print(f"  True weights: {TRUE_WEIGHTS}")
print(f"  True bias:    {TRUE_BIAS}")
print(f"  Noise std:    {NOISE_STD}")

rng = np.random.default_rng(args.seed)
X = rng.standard_normal((args.n_samples, 3))
noise = rng.normal(0, NOISE_STD, args.n_samples)
y = X @ np.array(TRUE_WEIGHTS) + TRUE_BIAS + noise

# ── 80/20 train/test split ───────────────────────────────────────
split = int(0.8 * args.n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── Save as CSV ──────────────────────────────────────────────────
header = "x1,x2,x3,target"

def save_csv(path, features, targets):
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(len(targets)):
            row = ",".join(f"{v:.6f}" for v in features[i])
            f.write(f"{row},{targets[i]:.6f}\n")

save_csv(os.path.join(args.processed_data, "train.csv"), X_train, y_train)
save_csv(os.path.join(args.processed_data, "test.csv"), X_test, y_test)

print(f"  Train: {len(y_train)} samples -> train.csv")
print(f"  Test:  {len(y_test)} samples -> test.csv")
print(f"\n── Preprocessing complete ──────────────────────")
