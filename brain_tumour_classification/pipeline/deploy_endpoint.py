import os
import argparse
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    BatchRetrySettings,
    CodeConfiguration,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction, AssetTypes
from azure.identity import ManagedIdentityCredential

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MANAGED_IDENTITY_CLIENT_ID, ENDPOINT_NAME, DEPLOYMENT_NAME,
    MODEL_NAME, COMPUTE_CLUSTER, TESTING_ASSET,
)

parser = argparse.ArgumentParser()
parser.add_argument("--register_output", type=str)
args = parser.parse_args()

# ── Check if we should deploy ────────────────────────────────────
def read_flag(name):
    path = os.path.join(args.register_output, name)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip() == "true"
    return False

if not read_flag("deploy.flag"):
    print("⏭️  No deployment needed — existing model is still best.")
    exit(0)

# ── Connect ──────────────────────────────────────────────────────
ml_client = MLClient(
    credential=ManagedIdentityCredential(
        client_id=MANAGED_IDENTITY_CLIENT_ID
    ),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Get registered model ─────────────────────────────────────────
model = ml_client.models.get(MODEL_NAME, label="latest")
print(f"Deploying model: {model.name} v{model.version}")

# ── Create endpoint if it doesn't exist ──────────────────────────
try:
    ml_client.batch_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' already exists")
except Exception:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    ml_client.batch_endpoints.begin_create_or_update(BatchEndpoint(
        name=ENDPOINT_NAME,
        description="Brain tumour classification batch endpoint",
    )).result()
    print(f"✅ Endpoint created")

# ── Create or update deployment ───────────────────────────────────
env = ml_client.environments.get("brain-tumour-env", label="latest")

deployment = ModelBatchDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=model,
    code_configuration=CodeConfiguration(
        code=PROJECT_ROOT,
        scoring_script="pipeline/score.py"
    ),
    environment=env,
    compute=COMPUTE_CLUSTER,
    settings=ModelBatchDeploymentSettings(
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        instance_count=1,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=2, timeout=300),
    ),
)

print(f"Creating/updating deployment '{DEPLOYMENT_NAME}'...")
ml_client.batch_deployments.begin_create_or_update(deployment).result()

endpoint = ml_client.batch_endpoints.get(ENDPOINT_NAME)
endpoint.defaults.deployment_name = DEPLOYMENT_NAME
ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

print(f"✅ Deployment complete — '{DEPLOYMENT_NAME}' is now the default")

# ── Fetch testing data asset ──────────────────────────────────────
print("\nFetching testing data asset...")
try:
    testing_asset = ml_client.data.get(TESTING_ASSET, label="latest")
    print(f"Testing asset: {testing_asset.name} v{testing_asset.version}")
except Exception as e:
    print(f"❌ Could not find testing data asset: {e}")
    exit(1)

# ── Invoke batch endpoint ─────────────────────────────────────────
print("\nInvoking batch endpoint...")
job = ml_client.batch_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    input=Input(
        type=AssetTypes.URI_FOLDER,
        path=testing_asset.path,
    ),
    output=Output(
        type=AssetTypes.URI_FOLDER,
        path="azureml://datastores/workspaceblobstore/paths/batch_results/brain_tumour/",
    ),
)
print(f"✅ Batch inference job submitted: {job.name}")

# ── Wait for completion ───────────────────────────────────────────
print("\n⏳ Waiting for inference to complete...")
ml_client.jobs.stream(job.name)

completed = ml_client.jobs.get(job.name)
if completed.status == "Completed":
    print(f"✅ Inference complete — predictions written to 'predictions.csv'")
else:
    print(f"❌ Inference job finished with status: {completed.status}")
