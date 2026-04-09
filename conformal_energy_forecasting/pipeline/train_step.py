import os
import json
import argparse
import glob
import numpy as np
import torch
import torch.optim as optim
import mlflow
from torch.utils.data import Dataset, DataLoader, random_split
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.quantile_forecaster import QuantileForecaster

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--processed_data", type=str)
parser.add_argument("--model_output",   type=str)
parser.add_argument("--num_epochs",     type=int,   default=100)
parser.add_argument("--learning_rate",  type=float, default=1e-3)
parser.add_argument("--batch_size",     type=int,   default=256)
parser.add_argument("--val_split",      type=float, default=0.15)
parser.add_argument("--alpha",          type=float, default=0.1,
                    help="Miscoverage rate for conformal prediction (default: 0.1 = 90%% coverage)")
parser.add_argument("--quantile_lower", type=float, default=0.05)
parser.add_argument("--quantile_upper", type=float, default=0.95)
args = parser.parse_args()

os.makedirs(args.model_output, exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────
class TensorDataset(Dataset):
    def __init__(self, root_dir: str, split: str):
        split_dir = os.path.join(root_dir, split)
        self.samples = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        print(f"{split}: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=True)
        return data["features"], data["target"]


# ── Pinball (quantile) loss ──────────────────────────────────────
def pinball_loss(pred, target, quantiles):
    """
    Compute pinball loss for multiple quantiles.
    pred:      (batch, 3) — predicted quantiles [lower, median, upper]
    target:    (batch,)   — actual values
    quantiles: list of 3 quantile levels [0.05, 0.5, 0.95]
    """
    target = target.unsqueeze(1)  # (batch, 1)
    errors = target - pred        # (batch, 3)
    loss = torch.zeros_like(pred)
    for i, q in enumerate(quantiles):
        loss[:, i] = torch.where(
            errors[:, i] >= 0,
            q * errors[:, i],
            (q - 1) * errors[:, i]
        )
    return loss.mean()


# ── Training helpers ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, quantiles, device):
    model.train()
    total_loss, n = 0.0, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(features)
        loss = pinball_loss(pred, targets, quantiles)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        n += features.size(0)
    return total_loss / n


def val_epoch(model, loader, quantiles, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            pred = model(features)
            loss = pinball_loss(pred, targets, quantiles)
            total_loss += loss.item() * features.size(0)
            n += features.size(0)
            all_preds.append(pred.cpu())
            all_targets.append(targets.cpu())
    preds   = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    # MAE on median prediction (normalised space — used for model selection)
    median_pred = preds[:, 1]
    mae = (median_pred - targets).abs().mean().item()
    return total_loss / n, mae


# ── Setup ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

quantiles = [args.quantile_lower, 0.5, args.quantile_upper]

# Load normalisation stats
norm_stats_path = os.path.join(args.processed_data, "norm_stats.json")
with open(norm_stats_path) as f:
    norm_stats = json.load(f)

# Copy norm stats to model output for inference
with open(os.path.join(args.model_output, "norm_stats.json"), "w") as f:
    json.dump(norm_stats, f)

# Determine number of features from first sample
sample_path = glob.glob(os.path.join(args.processed_data, "train", "*.pt"))[0]
sample_data = torch.load(sample_path, weights_only=True)
n_features = sample_data["features"].shape[0]
print(f"Number of features: {n_features}")

# ── Load datasets ────────────────────────────────────────────────
full_train = TensorDataset(args.processed_data, "train")
cal_dataset = TensorDataset(args.processed_data, "calibration")
test_dataset = TensorDataset(args.processed_data, "test")

# Split training into train/val
val_size   = int(args.val_split * len(full_train))
train_size = len(full_train) - val_size
train_dataset, val_dataset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"  Train split: {train_size}  |  Val split: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)
cal_loader   = DataLoader(cal_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4)

# ── Model ────────────────────────────────────────────────────────
model = QuantileForecaster(n_features=n_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

# ── Train ────────────────────────────────────────────────────────
print("\n── Training ─────────────────────────────────────")
mlflow.start_run()
mlflow.log_params({
    "num_epochs":      args.num_epochs,
    "learning_rate":   args.learning_rate,
    "batch_size":      args.batch_size,
    "val_split":       args.val_split,
    "alpha":           args.alpha,
    "quantile_lower":  args.quantile_lower,
    "quantile_upper":  args.quantile_upper,
    "optimizer":       "Adam",
    "scheduler":       "ReduceLROnPlateau",
    "n_features":      n_features,
    "hidden1":         128,
    "hidden2":         64,
    "dropout":         0.2,
})

best_val_loss = float("inf")
patience_counter = 0
early_stop_patience = 15

for epoch in range(1, args.num_epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, quantiles, device)
    val_loss, val_mae = val_epoch(model, val_loader, quantiles, device)
    scheduler.step(val_loss)

    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "val_mae":    val_mae,
    }, step=epoch)

    print(f"Epoch {epoch:>3}  train_loss={train_loss:.6f}  "
          f"val_loss={val_loss:.6f}  val_mae={val_mae:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(args.model_output, "best_model.pt"))
        print(f"  ✅ New best saved (val_loss={best_val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"  ⏹️  Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load(
    os.path.join(args.model_output, "best_model.pt"), weights_only=True
))

# ── Conformal calibration (CQR) ─────────────────────────────────
print("\n── Conformal Calibration (CQR) ─────────────────")
model.eval()
cal_scores = []
with torch.no_grad():
    for features, targets in cal_loader:
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        q_lower = pred[:, 0]
        q_upper = pred[:, 2]
        # Nonconformity score: max(q_lower - y, y - q_upper)
        scores = torch.max(q_lower - targets, targets - q_upper)
        cal_scores.append(scores.cpu())

cal_scores = torch.cat(cal_scores, dim=0).numpy()
n_cal = len(cal_scores)

# Quantile at level (1 - alpha)(1 + 1/n_cal)
conformal_level = (1 - args.alpha) * (1 + 1 / n_cal)
conformal_quantile = float(np.quantile(cal_scores, min(conformal_level, 1.0)))
print(f"  Calibration samples:   {n_cal}")
print(f"  Conformal level:       {conformal_level:.6f}")
print(f"  Conformal quantile Q:  {conformal_quantile:.6f}")

# Verify calibration coverage on calibration set
cal_coverage = float(np.mean(cal_scores <= conformal_quantile))
print(f"  Calibration coverage:  {cal_coverage:.4f} (target: {1 - args.alpha:.2f})")

# Save conformal quantile alongside model
conformal_config = {
    "alpha":              args.alpha,
    "quantile_lower":     args.quantile_lower,
    "quantile_upper":     args.quantile_upper,
    "conformal_quantile": conformal_quantile,
    "n_calibration":      n_cal,
}
with open(os.path.join(args.model_output, "conformal_config.json"), "w") as f:
    json.dump(conformal_config, f)

# ── Test evaluation ──────────────────────────────────────────────
print("\n── Test set evaluation ──────────────────────────")
all_preds, all_targets = [], []
with torch.no_grad():
    for features, targets in test_loader:
        features, targets = features.to(device), targets.to(device)
        pred = model(features)
        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

preds   = torch.cat(all_preds, dim=0).numpy()
targets = torch.cat(all_targets, dim=0).numpy()

# Denormalise predictions and targets back to MWh for interpretable metrics
tgt_mean = norm_stats["tgt_mean"]
tgt_std  = norm_stats["tgt_std"]

preds_denorm   = preds * tgt_std + tgt_mean
targets_denorm = targets * tgt_std + tgt_mean

# Point forecast metrics (median) in real units (MWh)
median_pred = preds_denorm[:, 1]
test_mae  = float(np.mean(np.abs(median_pred - targets_denorm)))
test_rmse = float(np.sqrt(np.mean((median_pred - targets_denorm) ** 2)))
test_mape = float(np.mean(np.abs(median_pred - targets_denorm) / np.abs(targets_denorm).clip(min=1.0))) * 100

# Conformalized interval metrics (computed in normalised space, coverage is scale-invariant)
conf_lower = preds[:, 0] - conformal_quantile
conf_upper = preds[:, 2] + conformal_quantile
test_coverage = float(np.mean((targets >= conf_lower) & (targets <= conf_upper)))

# Interval width in real units (MWh)
conf_lower_denorm = preds_denorm[:, 0] - conformal_quantile * tgt_std
conf_upper_denorm = preds_denorm[:, 2] + conformal_quantile * tgt_std
mean_interval_width = float(np.mean(conf_upper_denorm - conf_lower_denorm))

# Raw quantile interval (before conformal adjustment)
raw_coverage = float(np.mean((targets >= preds[:, 0]) & (targets <= preds[:, 2])))
raw_interval_width = float(np.mean(preds_denorm[:, 2] - preds_denorm[:, 0]))

print(f"  Test MAE:                    {test_mae:.2f} MWh")
print(f"  Test RMSE:                   {test_rmse:.2f} MWh")
print(f"  Test MAPE:                   {test_mape:.2f}%")
print(f"  Raw interval coverage:       {raw_coverage:.4f}")
print(f"  Raw interval width:          {raw_interval_width:.2f} MWh")
print(f"  Conformal interval coverage: {test_coverage:.4f} (target: {1 - args.alpha:.2f})")
print(f"  Conformal interval width:    {mean_interval_width:.2f} MWh")

# ── Log final metrics ────────────────────────────────────────────
mlflow.log_metrics({
    "best_val_loss":           best_val_loss,
    "test_mae":                test_mae,
    "test_rmse":               test_rmse,
    "test_mape":               test_mape,
    "raw_coverage":            raw_coverage,
    "raw_interval_width":      raw_interval_width,
    "conformal_coverage":      test_coverage,
    "conformal_interval_width": mean_interval_width,
    "conformal_quantile":      conformal_quantile,
})
mlflow.end_run()

# ── Write outputs for downstream steps ───────────────────────────
with open(os.path.join(args.model_output, "metrics.txt"), "w") as f:
    f.write(str(round(best_val_loss, 6)))

with open(os.path.join(args.model_output, "test_mae.txt"), "w") as f:
    f.write(str(round(test_mae, 6)))

with open(os.path.join(args.model_output, "test_coverage.txt"), "w") as f:
    f.write(str(round(test_coverage, 6)))

print(f"\n── Complete ─────────────────────────────────────────")
print(f"  Test MAE:                  {test_mae:.2f} MWh")
print(f"  Conformal coverage:        {test_coverage:.4f}")
print(f"  Conformal interval width:  {mean_interval_width:.2f} MWh")
