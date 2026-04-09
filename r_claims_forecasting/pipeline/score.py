"""
Batch Scoring Script — Python-to-R Bridge

This script satisfies the Azure ML batch endpoint contract (init/run) while
delegating actual prediction to an R script (score_predict.R) via subprocess.

The R script loads the saved Gamma GLM (.rds), predicts on input CSVs, and
writes predictions with 90% prediction intervals.

No secrets are used — model artefacts are mounted by the Azure ML runtime
via the AZUREML_MODEL_DIR environment variable.
"""
import os
import glob
import json
import subprocess
import tempfile
import pandas as pd


def init():
    """Locate the R model artefacts and the R scoring script."""
    global model_dir, r_script_path

    model_dir = os.environ.get("AZUREML_MODEL_DIR")

    # Find model.rds within the model directory
    rds_files = glob.glob(os.path.join(model_dir, "**", "model.rds"), recursive=True)
    if not rds_files:
        raise FileNotFoundError(f"No model.rds found in {model_dir}")
    print(f"  R model: {rds_files[0]}")

    # Verify norm_stats.json exists
    stats_files = glob.glob(os.path.join(model_dir, "**", "norm_stats.json"), recursive=True)
    if not stats_files:
        raise FileNotFoundError(f"No norm_stats.json found in {model_dir}")
    print(f"  Norm stats: {stats_files[0]}")

    # Path to the R prediction helper script (bundled with the deployment code)
    r_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "score_predict.R"
    )
    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"R scoring script not found: {r_script_path}")
    print(f"  R script: {r_script_path}")

    print("✅ init() complete — R model and scoring script located")


def run(mini_batch):
    """
    Process a mini-batch of CSV files.
    Each CSV is expected to contain insurance policy records with the same
    columns used during training (minus claim_amount, which is predicted).
    """
    all_results = []

    for csv_path in mini_batch:
        csv_path = str(csv_path)
        if not csv_path.endswith(".csv"):
            continue

        try:
            # Create a temp file for predictions output
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w"
            ) as tmp:
                output_path = tmp.name

            # Find model.rds path
            rds_files = glob.glob(
                os.path.join(model_dir, "**", "model.rds"), recursive=True
            )

            # Call R to make predictions
            result = subprocess.run(
                [
                    "Rscript", r_script_path,
                    "--model_path", rds_files[0],
                    "--input_csv", csv_path,
                    "--output_csv", output_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"[WARN] R script failed for {csv_path}: {result.stderr}")
                continue

            # Read predictions back
            if os.path.exists(output_path):
                pred_df = pd.read_csv(output_path)
                all_results.append(pred_df)

        except Exception as e:
            print(f"[WARN] Failed on {csv_path}: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
