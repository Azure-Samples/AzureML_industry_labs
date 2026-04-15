"""Pipeline orchestration — submit the readmission prediction pipeline.

Loads all components from the shared ML Registry and submits the pipeline
to the workspace configured in the local Azure ML config.json.

Usage:
    az ml workspace show --resource-group rg-readmit-dev --workspace-name readmit-dev-ws > .azureml/config.json
    python main.py
"""

from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

REGISTRY_NAME = "readmit-registry"


def main() -> None:
    credential = DefaultAzureCredential()

    # Workspace client (reads .azureml/config.json)
    ml_client = MLClient.from_config(credential=credential)

    # Registry client for loading shared components
    registry_client = MLClient(credential=credential, registry_name=REGISTRY_NAME)

    generate_data_component = registry_client.components.get("readmission_generate_data", label="latest")
    prep_component = registry_client.components.get("readmission_prep", label="latest")
    train_component = registry_client.components.get("readmission_train", label="latest")
    evaluate_component = registry_client.components.get("readmission_evaluate", label="latest")
    register_component = registry_client.components.get("readmission_register", label="latest")

    @pipeline(
        name="readmission-prediction",
        description="End-to-end pipeline for 30-day hospital readmission risk prediction.",
        default_compute="cpu-cluster",
    )
    def readmission_pipeline():
        generate_data = generate_data_component(num_samples=20000, seed=42)
        prep = prep_component(raw_data=generate_data.outputs.raw_data)
        train = train_component(
            train_data=prep.outputs.train_data,
            val_data=prep.outputs.val_data,
        )
        evaluate = evaluate_component(
            model_name="readmission-model",
            model_input=train.outputs.model_output,
            test_data=prep.outputs.test_data,
        )
        register = register_component(
            model_name="readmission-model",
            model_path=train.outputs.model_output,
            evaluation_output=evaluate.outputs.evaluation_output,
        )
        return {
            "train_data": prep.outputs.train_data,
            "val_data": prep.outputs.val_data,
            "test_data": prep.outputs.test_data,
            "trained_model": train.outputs.model_output,
            "evaluation_output": evaluate.outputs.evaluation_output,
            "model_info_output_path": register.outputs.model_info_output_path,
        }

    pipeline_job = readmission_pipeline()

    job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Pipeline submitted: {job.name}")
    print(f"Studio URL: {job.studio_url}")
    ml_client.jobs.stream(job.name)


if __name__ == "__main__":
    main()
