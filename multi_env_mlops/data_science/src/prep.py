"""Prepare raw patient data: clean, encode, and split into train/val/test."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TARGET_COL, NUMERIC_COLS, CATEGORICAL_COLS, ALL_FEATURE_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess patient data")
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    raw_path = Path(args.raw_data)
    # Try CSV first (Azure ML uri_file outputs have no extension)
    try:
        data = pd.read_csv(raw_path)
    except Exception:
        data = pd.read_parquet(raw_path)

    # Keep only expected columns
    data = data[ALL_FEATURE_COLS + [TARGET_COL]]

    # Drop rows with missing target
    data = data.dropna(subset=[TARGET_COL])

    # One-hot encode categoricals
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True)

    # Split: 70 / 15 / 15
    train, temp = train_test_split(data, test_size=0.30, random_state=42, stratify=data[TARGET_COL])
    val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp[TARGET_COL])

    mlflow.log_metric("train_size", train.shape[0])
    mlflow.log_metric("val_size", val.shape[0])
    mlflow.log_metric("test_size", test.shape[0])
    mlflow.log_metric("readmission_rate", float(data[TARGET_COL].mean()))

    # Save as parquet
    Path(args.train_data).mkdir(parents=True, exist_ok=True)
    Path(args.val_data).mkdir(parents=True, exist_ok=True)
    Path(args.test_data).mkdir(parents=True, exist_ok=True)

    train.to_parquet(Path(args.train_data) / "train.parquet", index=False)
    val.to_parquet(Path(args.val_data) / "val.parquet", index=False)
    test.to_parquet(Path(args.test_data) / "test.parquet", index=False)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")


if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
