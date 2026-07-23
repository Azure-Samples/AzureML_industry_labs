"""Submit the document preprocessing pipeline and deploy batch embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute, IdentityConfiguration
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

from pipeline.deploy_endpoint import deploy_batch_endpoint


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy and invoke an Azure ML batch endpoint for document embeddings."
    )
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--azure-openai-endpoint", required=True)
    parser.add_argument("--azure-openai-deployment", required=True)
    parser.add_argument("--compute-name", default="batch-embeddings-cluster")
    parser.add_argument("--endpoint-name", default="document-embeddings-batch")
    parser.add_argument("--input-data", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--create-compute", action="store_true")
    return parser.parse_args()


def get_ml_client(args: argparse.Namespace) -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )


def ensure_compute(ml_client: MLClient, compute_name: str, create: bool) -> None:
    try:
        ml_client.compute.get(compute_name)
    except ResourceNotFoundError:
        if not create:
            raise RuntimeError(
                f"Compute '{compute_name}' does not exist. Re-run with --create-compute."
            ) from None
        ml_client.compute.begin_create_or_update(
            AmlCompute(
                name=compute_name,
                size="STANDARD_D4AS_V5",
                min_instances=0,
                max_instances=4,
                idle_time_before_scale_down=120,
                identity=IdentityConfiguration(type="system_assigned"),
            )
        ).result()


def create_preprocessing_pipeline(
    compute_name: str, input_data: str, processed_data: str
):
    preprocess_component = command(
        name="prepare_embedding_documents",
        display_name="Prepare documents for embedding",
        code=str(PROJECT_ROOT),
        command=(
            "python -m pipeline.preprocess_step "
            "--input-data ${{inputs.input_data}} "
            "--processed-data ${{outputs.processed_data}}"
        ),
        environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
        compute=compute_name,
        inputs={"input_data": Input(type=AssetTypes.URI_FOLDER)},
        outputs={
            "processed_data": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
                path=processed_data,
            )
        },
    )

    @pipeline(
        name="batch_document_embeddings",
        description="Validate and normalize documents for batch embedding generation.",
    )
    def embedding_pipeline():
        preprocess_job = preprocess_component(
            input_data=Input(type=AssetTypes.URI_FOLDER, path=input_data)
        )
        return {"processed_data": preprocess_job.outputs.processed_data}

    return embedding_pipeline()


def main() -> None:
    args = parse_args()
    ml_client = get_ml_client(args)
    ensure_compute(ml_client, args.compute_name, args.create_compute)

    processed_data = (
        "azureml://datastores/workspaceblobstore/paths/"
        f"batch-document-embeddings/{uuid4().hex}/"
    )
    pipeline_job = ml_client.jobs.create_or_update(
        create_preprocessing_pipeline(
            args.compute_name, args.input_data, processed_data
        ),
        experiment_name="batch-document-embeddings",
    )
    print(f"Pipeline submitted: {pipeline_job.name}")
    print(f"Studio URL: {pipeline_job.studio_url}")
    ml_client.jobs.stream(pipeline_job.name)
    completed = ml_client.jobs.get(pipeline_job.name)
    if completed.status != "Completed":
        raise RuntimeError(f"Pipeline finished with status '{completed.status}'.")

    deploy_batch_endpoint(
        ml_client=ml_client,
        project_root=PROJECT_ROOT,
        endpoint_name=args.endpoint_name,
        compute_name=args.compute_name,
        azure_openai_endpoint=args.azure_openai_endpoint,
        azure_openai_deployment=args.azure_openai_deployment,
    )
    scoring_job = ml_client.batch_endpoints.invoke(
        endpoint_name=args.endpoint_name,
        inputs={
            "input": Input(type=AssetTypes.URI_FOLDER, path=processed_data)
        },
    )
    print(f"Scoring job submitted: {scoring_job.name}")
    ml_client.jobs.stream(scoring_job.name)
    completed_scoring_job = ml_client.jobs.get(scoring_job.name)
    if completed_scoring_job.status != "Completed":
        raise RuntimeError(
            f"Scoring job finished with status '{completed_scoring_job.status}'."
        )

    output_dir = Path(args.output_dir) / scoring_job.name
    ml_client.jobs.download(
        name=scoring_job.name, output_name="score", download_path=output_dir
    )
    result_files = list(output_dir.rglob("embeddings.jsonl"))
    if not result_files:
        raise RuntimeError("The job completed without an embeddings.jsonl output.")
    with result_files[0].open(encoding="utf-8") as result_file:
        first_result = json.loads(next(result_file))
    print(f"Output: {result_files[0]}")
    print(f"Verified embedding dimensions: {len(first_result['embedding'])}")


if __name__ == "__main__":
    main()