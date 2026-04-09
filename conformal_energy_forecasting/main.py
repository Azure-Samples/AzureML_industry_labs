import os
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

CLUSTER = "gpu1"

print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# ── Components ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

preprocess_component = command(
    name="preprocess",
    display_name="Preprocess: generate synthetic data + engineer features",
    code=PROJECT_ROOT,
    command=(
        "python pipeline/preprocess_step.py "
        "--processed_data ${{outputs.processed_data}} "
        "--seed 42 "
        "--n_years 3"
    ),
    environment="energy-forecast-env@latest",
    compute=CLUSTER,
    outputs={"processed_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
)

train_component = command(
    name="train",
    display_name="Train: quantile regression + conformal calibration",
    code=PROJECT_ROOT,
    command=(
        "python pipeline/train_step.py "
        "--processed_data ${{inputs.processed_data}} "
        "--model_output ${{outputs.model_output}} "
        "--num_epochs 100 "
        "--learning_rate 1e-3 "
        "--batch_size 256 "
        "--alpha 0.1"
    ),
    environment="energy-forecast-env@latest",
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
    environment="energy-forecast-env@latest",
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
    environment="energy-forecast-env@latest",
    compute=CLUSTER,
    inputs={"register_output": Input(type=AssetTypes.URI_FOLDER)},
)


# ── Pipeline ─────────────────────────────────────────────────────
@pipeline(name="energy_forecast_pipeline",
          description="Preprocess + Train (quantile regression + CQR) + Register + Deploy")
def energy_forecast_pipeline():
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
pipeline_job = energy_forecast_pipeline()

submitted = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="conformal-energy-forecasting"
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
