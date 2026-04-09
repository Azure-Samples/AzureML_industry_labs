"""
R Claims Severity Forecasting — Pipeline Orchestration

Defines and submits a 4-step Azure ML pipeline:
  1. Preprocess  — generate synthetic insurance claims data (R)
  2. Train       — fit a Gamma GLM for claims severity (R)
  3. Register    — compare against existing model, register if better (Python)
  4. Deploy      — create/update a batch endpoint (Python)

Usage:
    python main.py

Requires:
    - Azure ML workspace config at .azureml/config.json (or parent directory)
    - Registered environment: r-claims-forecast-env
    - Compute cluster: cpu1
"""
import os
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Authenticate via local CLI credentials (az login)
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

CLUSTER = "cpu1"

print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# ── Components ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

preprocess_component = command(
    name="preprocess",
    display_name="Preprocess: generate synthetic insurance claims data",
    code=PROJECT_ROOT,
    command=(
        "Rscript pipeline/preprocess_step.R "
        "--processed_data ${{outputs.processed_data}} "
        "--seed 42 "
        "--n_policies 50000"
    ),
    environment="r-claims-forecast-env@latest",
    compute=CLUSTER,
    outputs={"processed_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
)

train_component = command(
    name="train",
    display_name="Train: Gamma GLM for claims severity",
    code=PROJECT_ROOT,
    command=(
        "Rscript pipeline/train_step.R "
        "--processed_data ${{inputs.processed_data}} "
        "--model_output ${{outputs.model_output}}"
    ),
    environment="r-claims-forecast-env@latest",
    compute=CLUSTER,
    inputs={"processed_data": Input(type=AssetTypes.URI_FOLDER)},
    outputs={"model_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
)

register_component = command(
    name="register",
    display_name="Register: model to registry",
    code=PROJECT_ROOT,
    command=(
        "python pipeline/register_model.py "
        "--model_output ${{inputs.model_output}} "
        "--register_output ${{outputs.register_output}}"
    ),
    environment="r-claims-forecast-env@latest",
    compute=CLUSTER,
    inputs={"model_output":    Input(type=AssetTypes.URI_FOLDER)},
    outputs={"register_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
)

deploy_component = command(
    name="deploy",
    display_name="Deploy: batch endpoint deployment",
    code=PROJECT_ROOT,
    command=(
        "python pipeline/deploy_endpoint.py "
        "--register_output ${{inputs.register_output}}"
    ),
    environment="r-claims-forecast-env@latest",
    compute=CLUSTER,
    inputs={"register_output": Input(type=AssetTypes.URI_FOLDER)},
)


# ── Pipeline ─────────────────────────────────────────────────────
@pipeline(name="r_claims_forecast_pipeline",
          description="Preprocess + Train (Gamma GLM in R) + Register + Deploy batch endpoint")
def r_claims_forecast_pipeline():
    preprocess_job = preprocess_component()
    train_job = train_component(
        processed_data=preprocess_job.outputs.processed_data
    )
    register_job = register_component(
        model_output=train_job.outputs.model_output
    )
    deploy_component(
        register_output=register_job.outputs.register_output
    )
    return {"model_output": train_job.outputs.model_output}


# ── Submit ────────────────────────────────────────────────────────
pipeline_job = r_claims_forecast_pipeline()

submitted = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="r-claims-forecasting"
)
print(f"✅ Pipeline submitted: {submitted.name}")
print(f"   Studio URL: {submitted.studio_url}")

# ── Wait for completion ───────────────────────────────────────────
print("\n⏳ Waiting for pipeline to complete...")
ml_client.jobs.stream(submitted.name)

completed = ml_client.jobs.get(submitted.name)

if completed.status == "Completed":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Pipeline completed successfully.                    ║")
    print("║  Check Azure ML Studio for metrics and artefacts.    ║")
    print("╚══════════════════════════════════════════════════════╝")
else:
    print(f"\n❌ Pipeline finished with status: {completed.status}")
