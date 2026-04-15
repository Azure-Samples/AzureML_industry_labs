import os
import argparse
import glob
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import ManagedIdentityCredential

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANAGED_IDENTITY_CLIENT_ID, MODEL_NAME, TRAINING_ASSET, TESTING_ASSET

parser = argparse.ArgumentParser()
parser.add_argument("--model_output",    type=str)
parser.add_argument("--register_output", type=str)
args = parser.parse_args()

os.makedirs(args.register_output, exist_ok=True)

# ── Check if anything happened ───────────────────────────────────
metrics_path = os.path.join(args.model_output, "metrics.txt")
model_trained_flag = os.path.join(args.model_output, "model_trained.flag")

def read_flag(flag_name):
    path = os.path.join(args.model_output, flag_name)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip() == "true"
    return False

training_changed = read_flag("training_changed.flag")
testing_changed  = read_flag("testing_changed.flag")
model_trained    = read_flag("model_trained.flag")

print(f"Training changed: {training_changed}")
print(f"Testing changed:  {testing_changed}")
print(f"Model trained:    {model_trained}")

# If no training happened and testing hasn't changed, nothing to do
if not model_trained and not testing_changed:
    print("✅ Nothing to register.")
    with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
        f.write("false")
    exit(0)

# ── Connect to Azure ML ──────────────────────────────────────────
ml_client = MLClient(
    credential=ManagedIdentityCredential(
        client_id=MANAGED_IDENTITY_CLIENT_ID
    ),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

# ── Read val acc ─────────────────────────────────────────────────
val_acc = 0.0
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        val_acc = float(f.read().strip())
    print(f"New model val acc: {val_acc:.2f}%")

# ── Compare against existing registered model ────────────────────
should_deploy = False
try:
    existing     = ml_client.models.get(MODEL_NAME, label="latest")
    existing_val = float(existing.tags.get("val_acc", 0))
    print(f"Existing model val acc: {existing_val:.2f}%")

    if model_trained and val_acc > existing_val:
        print(f"✅ New model is better ({val_acc:.2f}% > {existing_val:.2f}%) — registering and deploying")
        should_deploy = True
    elif not model_trained and testing_changed:
        print("ℹ️  Testing data changed, model unchanged — triggering re-inference on batch endpoint")
        should_deploy = True
    else:
        print(f"⏭️  Existing model is better or equal — skipping")
        with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
            f.write("false")
        exit(0)
except Exception:
    print("No existing model found — registering first version")
    should_deploy = True

# ── Determine model path ─────────────────────────────────────────
try:
    trained_on_version = str(int(ml_client.data.get(TRAINING_ASSET, label="latest").version) + (1 if training_changed else 0))
except Exception:
    trained_on_version = "unknown"

try:
    tested_on_version = str(int(ml_client.data.get(TESTING_ASSET, label="latest").version) + (1 if testing_changed else 0))
except Exception:
    tested_on_version = "unknown"

best_model_path = os.path.join(args.model_output, "best_model.pt")
if model_trained and os.path.exists(best_model_path):
    model_path = best_model_path
    print("Using newly trained model")
else:
    try:
        existing     = ml_client.models.get(MODEL_NAME, label="latest")
        download_dir = "/tmp/existing_model"
        ml_client.models.download(
            name=MODEL_NAME,
            version=existing.version,
            download_path=download_dir,
        )
        pt_files = glob.glob(os.path.join(download_dir, "**/*.pt"), recursive=True)
        if pt_files:
            model_path = pt_files[0]
            print(f"Using existing model v{existing.version}")
        else:
            print("❌ Could not find existing model weights")
            with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
                f.write("false")
            exit(1)
    except Exception as e:
        print(f"❌ Could not download existing model: {e}")
        with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
            f.write("false")
        exit(1)

# ── Register ─────────────────────────────────────────────────────
tags = {
    "trained_on_asset":   TRAINING_ASSET,
    "trained_on_version": trained_on_version,
    "tested_on_asset":    TESTING_ASSET,
    "tested_on_version":  tested_on_version,
    "training_changed":   str(training_changed),
    "testing_changed":    str(testing_changed),
    "model_retrained":    str(model_trained),
}
if val_acc > 0:
    tags["val_acc"] = str(round(val_acc, 4))

ml_client.models.create_or_update(Model(
    path=model_path,
    name=MODEL_NAME,
    type=AssetTypes.CUSTOM_MODEL,
    description="Brain tumour ResNet18 fine-tuned classifier",
    tags=tags,
))

print(f"\n✅ Registered '{MODEL_NAME}' (val_acc={val_acc:.2f}%)")

# ── Write deploy flag ─────────────────────────────────────────────
with open(os.path.join(args.register_output, "deploy.flag"), "w") as f:
    f.write("true" if should_deploy else "false")
print(f"   Deploy flag: {should_deploy}")
