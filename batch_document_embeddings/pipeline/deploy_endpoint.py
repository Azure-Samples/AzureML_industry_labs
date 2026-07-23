"""Create or update the Azure ML batch endpoint and its default deployment."""

from __future__ import annotations

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    BatchEndpoint,
    BuildContext,
    CodeConfiguration,
    Environment,
    Model,
    ModelBatchDeployment,
)


def deploy_batch_endpoint(
    ml_client: MLClient,
    project_root: Path,
    endpoint_name: str,
    compute_name: str,
    azure_openai_endpoint: str,
    azure_openai_deployment: str,
) -> ModelBatchDeployment:
    environment = ml_client.environments.create_or_update(
        Environment(
            name="batch-document-embeddings-env",
            description="Runtime for managed-identity Azure OpenAI batch scoring.",
            build=BuildContext(path=str(project_root), dockerfile_path="Dockerfile"),
        )
    )
    model = ml_client.models.create_or_update(
        Model(
            name="azure-openai-embedding-contract",
            path=str(project_root / "model"),
            type=AssetTypes.CUSTOM_MODEL,
            description="Metadata contract for the Azure OpenAI embedding deployment.",
            tags={"task": "embeddings", "provider": "Azure OpenAI"},
        )
    )
    endpoint = ml_client.batch_endpoints.begin_create_or_update(
        BatchEndpoint(
            name=endpoint_name,
            description="Generate Azure OpenAI embeddings for document collections.",
        )
    ).result()
    deployment = ModelBatchDeployment(
        name="default",
        endpoint_name=endpoint.name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=str(project_root), scoring_script="pipeline/score.py"
        ),
        compute=compute_name,
        instance_count=1,
        max_concurrency_per_instance=1,
        mini_batch_size=1,
        output_action="summary_only",
        error_threshold=0,
        environment_variables={
            "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
            "AZURE_OPENAI_DEPLOYMENT": azure_openai_deployment,
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
        },
    )
    deployment = ml_client.batch_deployments.begin_create_or_update(
        deployment
    ).result()
    endpoint.defaults.deployment_name = deployment.name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    return deployment