"""Train a readmission risk classifier with MLflow tracking."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TARGET_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train readmission model")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    train = pd.read_parquet(Path(args.train_data) / "train.parquet")
    val = pd.read_parquet(Path(args.val_data) / "val.parquet")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "subsample": args.subsample,
        "random_state": 42,
    }
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # --- Training metrics ---
    y_train_pred = model.predict(X_train)
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
    mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred))

    # --- Validation metrics ---
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    mlflow.log_metrics(
        {
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_auc": val_auc,
        }
    )
    print(f"Validation — AUC: {val_auc:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

    # --- Save model with signature ---
    signature = infer_signature(X_val, y_val_pred)
    input_example = X_val.head(5)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=args.model_output,
        signature=signature,
        input_example=input_example,
    )
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
