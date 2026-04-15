"""Register the trained model if the deploy flag is set."""

import argparse
import json
import os
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register readmission model")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--evaluation_output", type=str, required=True)
    parser.add_argument("--model_info_output_path", type=str, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    eval_dir = Path(args.evaluation_output)

    # Read deploy flag from evaluate step
    deploy_flag = int((eval_dir / "deploy_flag").read_text().strip())
    mlflow.log_metric("deploy_flag", deploy_flag)

    # Read metrics for tagging
    metrics: dict = {}
    metrics_file = eval_dir / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())

    out = Path(args.model_info_output_path)
    out.mkdir(parents=True, exist_ok=True)

    if deploy_flag == 1:
        print(f"Registering model: {args.model_name}")

        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential

        # Build MLClient from the run context environment variables
        subscription_id = os.environ["AZUREML_ARM_SUBSCRIPTION"]
        resource_group = os.environ["AZUREML_ARM_RESOURCEGROUP"]
        workspace_name = os.environ["AZUREML_ARM_WORKSPACE_NAME"]

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        model = Model(
            path=args.model_path,
            name=args.model_name,
            type=AssetTypes.MLFLOW_MODEL,
            description="30-day hospital readmission risk model",
            tags={k: str(v) for k, v in metrics.items()},
        )
        registered = ml_client.models.create_or_update(model)

        model_info = {
            "id": f"{registered.name}:{registered.version}",
            "name": registered.name,
            "version": registered.version,
            "metrics": metrics,
        }
        with open(out / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"Registered {registered.name} v{registered.version}")
    else:
        print("Model did not pass promotion gate — skipping registration.")
        model_info = {"registered": False, "reason": "deploy_flag=0"}
        with open(out / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
