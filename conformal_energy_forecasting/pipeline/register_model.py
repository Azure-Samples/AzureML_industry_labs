import os
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import ManagedIdentityCredential

parser = argparse.ArgumentParser()
parser.add_argument("--model_output",    type=str)
parser.add_argument("--register_output", type=str)
args = parser.parse_args()

os.makedirs(args.register_output, exist_ok=True)

# ── Check if anything happened ───────────────────────────────────
test_mae_file = os.path.join(args.model_output, "test_mae.txt")
if not os.path.exists(test_mae_file):
    print("✅ No evaluation results found — skipping registration.")
    with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
        f.write("false")
    exit(0)

# ── Connect to Azure ML ──────────────────────────────────────────
ml_client = MLClient(
    credential=ManagedIdentityCredential(),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

MODEL_NAME = "conformal-energy-forecaster"

# ── Read metrics ─────────────────────────────────────────────────
with open(test_mae_file) as f:
    test_mae = float(f.read().strip())
print(f"Test MAE: {test_mae:.6f}")

val_loss = 0.0
metrics_path = os.path.join(args.model_output, "metrics.txt")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        val_loss = float(f.read().strip())
    print(f"Best val loss: {val_loss:.6f}")

test_coverage = 0.0
coverage_path = os.path.join(args.model_output, "test_coverage.txt")
if os.path.exists(coverage_path):
    with open(coverage_path) as f:
        test_coverage = float(f.read().strip())
    print(f"Test coverage: {test_coverage:.4f}")

# ── Compare against existing registered model ────────────────────
should_deploy = False
try:
    existing     = ml_client.models.get(MODEL_NAME, label="latest")
    existing_mae = float(existing.tags.get("test_mae", float("inf")))
    print(f"Existing model test MAE: {existing_mae:.6f}")

    if test_mae < existing_mae:
        print(f"✅ New model is better ({test_mae:.6f} < {existing_mae:.6f}) — registering and deploying")
        should_deploy = True
    else:
        print(f"⏭️  Existing model is better or equal — skipping")
        with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
            f.write("false")
        exit(0)
except Exception:
    print("No existing model found — registering first version")
    should_deploy = True

# ── Register ─────────────────────────────────────────────────────
model_path = args.model_output

tags = {
    "test_mae":      str(round(test_mae, 6)),
    "val_loss":      str(round(val_loss, 6)),
    "test_coverage": str(round(test_coverage, 4)),
}

ml_client.models.create_or_update(Model(
    path=model_path,
    name=MODEL_NAME,
    type=AssetTypes.CUSTOM_MODEL,
    description="Quantile regression MLP with conformal prediction for energy demand forecasting",
    tags=tags,
))

print(f"\n✅ Registered '{MODEL_NAME}' (test_mae={test_mae:.6f}, coverage={test_coverage:.4f})")

# ── Write deploy flag ────────────────────────────────────────────
with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
    f.write("true" if should_deploy else "false")
print(f"   Deploy flag: {should_deploy}")
