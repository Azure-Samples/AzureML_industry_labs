import os
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

parser = argparse.ArgumentParser()
parser.add_argument("--model_output",    type=str)
parser.add_argument("--register_output", type=str)
args = parser.parse_args()

os.makedirs(args.register_output, exist_ok=True)

# ── Check if training produced results ───────────────────────────
test_mae_file = os.path.join(args.model_output, "test_mae.txt")
if not os.path.exists(test_mae_file):
    print("No evaluation results found — skipping registration.")
    exit(0)

# ── Connect to Azure ML ──────────────────────────────────────────
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

MODEL_NAME = "cpp-linear-regression"

# ── Read metrics ─────────────────────────────────────────────────
with open(test_mae_file) as f:
    test_mae = float(f.read().strip())
print(f"Test MAE: {test_mae:.6f}")

train_mse = 0.0
metrics_path = os.path.join(args.model_output, "metrics.txt")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        train_mse = float(f.read().strip())
    print(f"Best train MSE: {train_mse:.6f}")

# ── Compare against existing registered model ────────────────────
should_register = False
try:
    existing = ml_client.models.get(MODEL_NAME, label="latest")
    existing_mae = float(existing.tags.get("test_mae", float("inf")))
    print(f"Existing model test MAE: {existing_mae:.6f}")

    if test_mae < existing_mae:
        print(f"New model is better ({test_mae:.6f} < {existing_mae:.6f}) — registering")
        should_register = True
    else:
        print(f"Existing model is better or equal — skipping")
        exit(0)
except Exception:
    print("No existing model found — registering first version")
    should_register = True

# ── Register ─────────────────────────────────────────────────────
tags = {
    "test_mae":       str(round(test_mae, 6)),
    "best_train_mse": str(round(train_mse, 6)),
    "language":        "C++",
}

ml_client.models.create_or_update(Model(
    path=args.model_output,
    name=MODEL_NAME,
    type=AssetTypes.CUSTOM_MODEL,
    description="Linear regression trained in pure C++ via gradient descent",
    tags=tags,
))

print(f"\nRegistered '{MODEL_NAME}' (test_mae={test_mae:.6f})")
