import os
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

ENV_NAME = "cpp-training-env@latest"
COMPUTE  = "cpu1"
PROJECT  = os.path.dirname(os.path.abspath(__file__))

# ── Components ───────────────────────────────────────────────────

preprocess_component = command(
    name="preprocess",
    display_name="Generate Synthetic Data",
    command=(
        "python pipeline/preprocess_step.py"
        " --processed_data ${{outputs.processed_data}}"
        " --seed 42"
        " --n_samples 1000"
    ),
    outputs={
        "processed_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
    },
    environment=ENV_NAME,
    compute=COMPUTE,
    code=PROJECT,
)

train_component = command(
    name="train_cpp",
    display_name="Train C++ Linear Regression",
    command=(
        "python pipeline/train_step.py"
        " --processed_data ${{inputs.processed_data}}"
        " --model_output ${{outputs.model_output}}"
        " --learning_rate 0.01"
        " --epochs 1000"
    ),
    inputs={
        "processed_data": Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={
        "model_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
    },
    environment=ENV_NAME,
    compute=COMPUTE,
    code=PROJECT,
)

register_component = command(
    name="register_model",
    display_name="Register Model",
    command=(
        "python pipeline/register_model.py"
        " --model_output ${{inputs.model_output}}"
        " --register_output ${{outputs.register_output}}"
    ),
    inputs={
        "model_output": Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={
        "register_output": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
    },
    environment=ENV_NAME,
    compute=COMPUTE,
    code=PROJECT,
)

# ── Pipeline ─────────────────────────────────────────────────────

@pipeline(
    name="cpp-model-training",
    description="Train a linear regression model in pure C++ on Azure ML",
)
def cpp_training_pipeline():
    preprocess_job = preprocess_component()
    train_job = train_component(
        processed_data=preprocess_job.outputs.processed_data,
    )
    register_component(
        model_output=train_job.outputs.model_output,
    )

# ── Submit ───────────────────────────────────────────────────────

pipeline_job = cpp_training_pipeline()
pipeline_job.settings.default_compute = COMPUTE

submitted = ml_client.jobs.create_or_update(pipeline_job)
print(f"Pipeline submitted: {submitted.studio_url}")
ml_client.jobs.stream(submitted.name)
