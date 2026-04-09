import os
import json
import argparse
import subprocess
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--processed_data", type=str)
parser.add_argument("--model_output",   type=str)
parser.add_argument("--learning_rate",  type=float, default=0.01)
parser.add_argument("--epochs",         type=int,   default=1000)
args = parser.parse_args()

os.makedirs(args.model_output, exist_ok=True)

# ── Run the C++ training binary ──────────────────────────────────
cmd = [
    "train_cpp",
    "--data_dir",      args.processed_data,
    "--output_dir",    args.model_output,
    "--learning_rate", str(args.learning_rate),
    "--epochs",        str(args.epochs),
]

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

# ── Log metrics to MLflow ────────────────────────────────────────
mlflow.start_run()
mlflow.log_params({
    "learning_rate": args.learning_rate,
    "epochs":        args.epochs,
    "language":      "C++",
    "model_type":    "linear_regression",
})

test_mae_path = os.path.join(args.model_output, "test_mae.txt")
if os.path.exists(test_mae_path):
    with open(test_mae_path) as f:
        mlflow.log_metric("test_mae", float(f.read().strip()))

metrics_path = os.path.join(args.model_output, "metrics.txt")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        mlflow.log_metric("best_train_mse", float(f.read().strip()))

weights_path = os.path.join(args.model_output, "model_weights.json")
if os.path.exists(weights_path):
    with open(weights_path) as f:
        weights = json.load(f)
    for i, w in enumerate(weights["weights"]):
        mlflow.log_metric(f"weight_{i}", w)
    mlflow.log_metric("bias", weights["bias"])

mlflow.end_run()
print("\n── Metrics logged to MLflow ────────────────────")
