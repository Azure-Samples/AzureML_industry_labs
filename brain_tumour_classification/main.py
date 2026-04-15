import os
import json
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from config import (
    COMPUTE_CLUSTER, ENVIRONMENT_NAME, TRAINING_ASSET, TESTING_ASSET,
    RAW_DATA_PATH,
)

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
DATASTORE = ml_client.datastores.get_default()

print(f"✅ Connected to workspace: {ml_client.workspace_name}")

# ── Check previous data assets for manifests ─────────────────────
def get_previous_manifest(asset_name):
    try:
        latest = ml_client.data.get(asset_name, label="latest")
        tag = latest.tags.get("manifest", "")
        if tag:
            if tag.startswith("b'") or tag.startswith('b"'):
                tag = tag[2:-1]
            manifest_list = json.loads(tag)
            print(f"  {asset_name}: v{latest.version} ({len(manifest_list)} files)")
            return tag
        else:
            print(f"  {asset_name}: v{latest.version} (no manifest tag)")
            return "[]"
    except Exception:
        print(f"  {asset_name}: no previous asset found")
        return "[]"

print("Previous data assets:")
previous_manifest_training = get_previous_manifest(TRAINING_ASSET)
previous_manifest_testing  = get_previous_manifest(TESTING_ASSET)

# ── Write previous manifests to files ────────────────────────────
project_dir = os.path.dirname(os.path.abspath(__file__))

manifest_file_training = "/tmp/previous_manifest_training.json"
manifest_file_testing  = "/tmp/previous_manifest_testing.json"

with open(manifest_file_training, "w") as f:
    f.write(previous_manifest_training)

with open(manifest_file_testing, "w") as f:
    f.write(previous_manifest_testing)

print("Manifest files written to /tmp/")

# ── Components ───────────────────────────────────────────────────
PROJECT_ROOT = project_dir

preprocess_component = command(
    name="preprocess",
    display_name="Preprocess: raw images -> .pt tensors",
    code=PROJECT_ROOT,
    command=(
        "python pipeline/preprocess_step.py "
        "--raw_data ${{inputs.raw_data}} "
        "--processed_data ${{outputs.processed_data}} "
        "--manifest_file_training ${{inputs.manifest_file_training}} "
        "--manifest_file_testing ${{inputs.manifest_file_testing}} "
        "--n_augmentations 3 "
        "--debug_limit 7"
    ),
    environment=ENVIRONMENT_NAME,
    compute=COMPUTE_CLUSTER,
    inputs={
        "raw_data":               Input(type=AssetTypes.URI_FOLDER),
        "manifest_file_training": Input(type=AssetTypes.URI_FILE),
        "manifest_file_testing":  Input(type=AssetTypes.URI_FILE),
    },
    outputs={"processed_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
)

train_component = command(
    name="train",
    display_name="Train: ResNet18 fine-tuning + MLFlow logging",
    code=PROJECT_ROOT,
    command=(
        "python pipeline/train_step.py "
        "--processed_data ${{inputs.processed_data}} "
        "--model_output ${{outputs.model_output}} "
        "--num_epochs 25 "
        "--learning_rate 1e-5 "
        "--batch_size 32"
    ),
    environment=ENVIRONMENT_NAME,
    compute=COMPUTE_CLUSTER,
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
    environment=ENVIRONMENT_NAME,
    compute=COMPUTE_CLUSTER,
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
    environment=ENVIRONMENT_NAME,
    compute=COMPUTE_CLUSTER,
    inputs={"register_output": Input(type=AssetTypes.URI_FOLDER)},
)


# ── Pipeline ─────────────────────────────────────────────────────
@pipeline(name="brain_tumour_pipeline", description="Preprocess + Train + Register + Deploy")
def brain_tumour_pipeline(raw_data, manifest_file_training, manifest_file_testing):
    preprocess_job = preprocess_component(
        raw_data=raw_data,
        manifest_file_training=manifest_file_training,
        manifest_file_testing=manifest_file_testing,
    )
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
raw_data_uri = (
    f"azureml://subscriptions/{ml_client.subscription_id}"
    f"/resourcegroups/{ml_client.resource_group_name}"
    f"/workspaces/{ml_client.workspace_name}"
    f"/datastores/{DATASTORE.name}"
    f"/paths/{RAW_DATA_PATH}"
)

pipeline_job = brain_tumour_pipeline(
    raw_data=Input(type=AssetTypes.URI_FOLDER, path=raw_data_uri),
    manifest_file_training=Input(type=AssetTypes.URI_FILE, path=manifest_file_training),
    manifest_file_testing=Input(type=AssetTypes.URI_FILE, path=manifest_file_testing),
)

submitted = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="brain-tumour-classification"
)
print(f"✅ Pipeline submitted: {submitted.name}")
print(f"   Studio URL: {submitted.studio_url}")

# ── Wait for completion & register data assets ───────────────────
print("\n⏳ Waiting for pipeline to complete...")
ml_client.jobs.stream(submitted.name)

completed = ml_client.jobs.get(submitted.name)

if completed.status == "Completed":
    preprocess_run_id = None
    for child in ml_client.jobs.list(parent_job_name=submitted.name):
        if child.display_name == "preprocess_job":
            preprocess_run_id = child.name
            break

    if preprocess_run_id is None:
        print("❌ Could not find preprocess child job")
        exit(1)

    preprocess_output_path = (
        f"azureml://subscriptions/{ml_client.subscription_id}"
        f"/resourcegroups/{ml_client.resource_group_name}"
        f"/workspaces/{ml_client.workspace_name}"
        f"/datastores/workspaceblobstore"
        f"/paths/azureml/{preprocess_run_id}/processed_data/"
    )
    print(f"\n  Preprocess output: {preprocess_output_path}")

    from azureml.fsspec import AzureMachineLearningFileSystem
    fs = AzureMachineLearningFileSystem(preprocess_output_path)
    output_files = fs.ls("", detail=False)

    if any("NO_NEW_DATA" in f for f in output_files):
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║  No changes detected in raw data.                   ║")
        print("║  Using existing data assets.                        ║")
        print("║  Training, evaluation, and registration skipped.    ║")
        print("╚══════════════════════════════════════════════════════╝")
    else:
        training_changed = False
        testing_changed  = False
        flag_files = {f.split("/")[-1]: f for f in output_files}

        if "training_changed.flag" in flag_files:
            with fs.open(flag_files["training_changed.flag"], "rb") as f:
                training_changed = f.read().decode("utf-8").strip() == "true"

        if "testing_changed.flag" in flag_files:
            with fs.open(flag_files["testing_changed.flag"], "rb") as f:
                testing_changed = f.read().decode("utf-8").strip() == "true"

        print(f"\n  Training data changed: {training_changed}")
        print(f"  Testing data changed:  {testing_changed}")

        new_manifest_training = "[]"
        new_manifest_testing  = "[]"

        training_manifest_files = [f for f in output_files if "manifest_training.json" in f]
        testing_manifest_files  = [f for f in output_files if "manifest_testing.json" in f]

        if training_manifest_files:
            with fs.open(training_manifest_files[0], "rb") as f:
                new_manifest_training = f.read().decode("utf-8")

        if testing_manifest_files:
            with fs.open(testing_manifest_files[0], "rb") as f:
                new_manifest_testing = f.read().decode("utf-8")

        training_count = len(json.loads(new_manifest_training))
        testing_count  = len(json.loads(new_manifest_testing))

        if training_changed:
            try:
                v = int(ml_client.data.get(TRAINING_ASSET, label="latest").version)
            except Exception:
                v = 0
            ml_client.data.create_or_update(Data(
                path=preprocess_output_path + "Training/",
                type=AssetTypes.URI_FOLDER,
                name=TRAINING_ASSET,
                version=str(v + 1),
                description="Preprocessed brain tumour training tensors",
                tags={
                    "manifest":        new_manifest_training,
                    "n_augmentations": "3",
                    "total_raw_images": str(training_count),
                },
            ))
            print(f"  ✅ Registered '{TRAINING_ASSET}' v{v + 1} ({training_count} images)")
        else:
            print(f"  ⏭️  Training data unchanged — skipping training asset registration")

        if testing_changed:
            try:
                v = int(ml_client.data.get(TESTING_ASSET, label="latest").version)
            except Exception:
                v = 0
            ml_client.data.create_or_update(Data(
                path=preprocess_output_path + "Testing/",
                type=AssetTypes.URI_FOLDER,
                name=TESTING_ASSET,
                version=str(v + 1),
                description="Preprocessed brain tumour testing tensors",
                tags={
                    "manifest":        new_manifest_testing,
                    "total_raw_images": str(testing_count),
                },
            ))
            print(f"  ✅ Registered '{TESTING_ASSET}' v{v + 1} ({testing_count} images)")
        else:
            print(f"  ⏭️  Testing data unchanged — skipping testing asset registration")

        print("\n╔══════════════════════════════════════════════════════╗")
        if training_changed:
            print("║  Training data changed — model retrained.           ║")
        else:
            print("║  Training data unchanged — model NOT retrained.     ║")
        if testing_changed:
            print("║  Testing data changed — model re-evaluated.         ║")
        else:
            print("║  Testing data unchanged.                            ║")
        print(f"║  Training images: {training_count}")
        print(f"║  Testing images:  {testing_count}")
        print("╚══════════════════════════════════════════════════════╝")
else:
    print(f"\n❌ Pipeline finished with status: {completed.status}")
