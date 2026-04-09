"""
Deploy Batch Endpoint — Create or update the scoring endpoint

Creates (or updates) an Azure ML batch endpoint and deployment for the
R Gamma GLM model. Only deploys when the register step signals improvement
or when the endpoint does not yet exist.

Authentication uses the compute cluster's system-assigned managed identity
(SAMI) via environment variables injected by the Azure ML runtime.
No secrets or keys are stored in code.
"""
import os
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    BatchRetrySettings,
    CodeConfiguration,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.identity import ManagedIdentityCredential

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

deploy_flag = read_flag("deploy.flag")

# ── Connect ──────────────────────────────────────────────────────
ml_client = MLClient(
    credential=ManagedIdentityCredential(),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

ENDPOINT_NAME   = "r-claims-severity-batch"
DEPLOYMENT_NAME = "r-claims-severity-deployment"

# ── Check if endpoint exists ─────────────────────────────────────
endpoint_exists = True
try:
    ml_client.batch_endpoints.get(ENDPOINT_NAME)
except Exception:
    endpoint_exists = False

# Deploy if: new model is better OR endpoint is missing
if not deploy_flag and endpoint_exists:
    print("⏭️  No deployment needed — existing model is still best and endpoint exists.")
    exit(0)

if not deploy_flag and not endpoint_exists:
    print("⚠️  Model unchanged but endpoint is missing — redeploying.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Get registered model ─────────────────────────────────────────
model = ml_client.models.get("r-claims-severity-glm", label="latest")
print(f"Deploying model: {model.name} v{model.version}")

# ── Create endpoint if it doesn't exist ─────────────────────────
if not endpoint_exists:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    endpoint = BatchEndpoint(
        name=ENDPOINT_NAME,
        description="Insurance claims severity prediction using R Gamma GLM",
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print(f"✅ Endpoint created")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists")

# ── Create or update deployment ──────────────────────────────────
env = ml_client.environments.get("r-claims-forecast-env", label="latest")

deployment = ModelBatchDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=model,
    code_configuration=CodeConfiguration(
        code=PROJECT_ROOT,
        scoring_script="pipeline/score.py",
    ),
    environment=env,
    compute="cpu1",
    settings=ModelBatchDeploymentSettings(
        max_concurrency_per_instance=4,
        mini_batch_size=10,
        instance_count=1,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.csv",
        retry_settings=BatchRetrySettings(max_retries=2, timeout=300),
    ),
)

print(f"Creating/updating deployment '{DEPLOYMENT_NAME}'...")
ml_client.batch_deployments.begin_create_or_update(deployment).result()

# Set as default deployment
endpoint = ml_client.batch_endpoints.get(ENDPOINT_NAME)
if endpoint.defaults is None:
    from azure.ai.ml.entities import BatchEndpointDefaults
    endpoint.defaults = BatchEndpointDefaults(deployment_name=DEPLOYMENT_NAME)
else:
    endpoint.defaults.deployment_name = DEPLOYMENT_NAME
ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

print(f"✅ Deployment complete — '{DEPLOYMENT_NAME}' is now the default")
