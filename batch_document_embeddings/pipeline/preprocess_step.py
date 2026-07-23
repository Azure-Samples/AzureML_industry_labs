"""Normalize supported document files into line-delimited JSON records."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_processing.preprocess import normalize_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--processed-data", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_count, file_count = normalize_documents(
        Path(args.input_data), Path(args.processed_data) / "documents.jsonl"
    )
    print(f"Prepared {record_count} records from {file_count} files.")


if __name__ == "__main__":
    main()