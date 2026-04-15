"""Evaluate trained model on the test set and decide whether to promote."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from matplotlib import pyplot as plt

import mlflow
import mlflow.sklearn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TARGET_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate readmission model")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--evaluation_output", type=str, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    test = pd.read_parquet(Path(args.test_data) / "test.parquet")
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    model = mlflow.sklearn.load_model(args.model_input)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_auc": test_auc,
    }
    mlflow.log_metrics(metrics)

    print(f"Test — AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"]))

    # --- Save evaluation artefacts ---
    out = Path(args.evaluation_output)
    out.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"], output_dict=True)
    with open(out / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Model promotion gate ---
    # In a real setting, compare against the currently registered model.
    # For this lab we gate on a minimum AUC threshold.
    auc_threshold = 0.60
    deploy_flag = 1 if test_auc >= auc_threshold else 0

    with open(out / "deploy_flag", "w") as f:
        f.write(str(deploy_flag))

    mlflow.log_metric("deploy_flag", deploy_flag)
    print(f"Deploy flag: {deploy_flag} (AUC {test_auc:.4f} vs threshold {auc_threshold})")

    # --- Confusion-style scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(y_proba[y_test == 0], bins=bins, alpha=0.6, label="No Readmit", color="steelblue")
    ax.hist(y_proba[y_test == 1], bins=bins, alpha=0.6, label="Readmit", color="coral")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Readmission Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out / "probability_distribution.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
