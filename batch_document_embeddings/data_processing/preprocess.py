"""Read supported text formats and produce a normalized JSONL collection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".txt"}


def read_records(path: Path) -> Iterator[dict[str, str]]:
    if path.suffix == ".txt":
        for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if text.strip():
                yield {"id": f"{path.stem}-{line_number}", "text": text.strip()}
        return

    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as source:
            for row_number, row in enumerate(csv.DictReader(source)):
                text = row.get("text", "").strip()
                if text:
                    yield {
                        "id": row.get("id") or f"{path.stem}-{row_number}",
                        "text": text,
                    }
        return

    with path.open(encoding="utf-8") as source:
        records = json.load(source) if path.suffix == ".json" else map(json.loads, source)
        if isinstance(records, dict):
            records = [records]
        for row_number, record in enumerate(records):
            text = str(record.get("text", "")).strip()
            if text:
                yield {
                    "id": str(record.get("id") or f"{path.stem}-{row_number}"),
                    "text": text,
                }


def normalize_documents(input_dir: Path, output_file: Path) -> tuple[int, int]:
    source_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not source_files:
        raise ValueError(f"No supported documents found under {input_dir}.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    with output_file.open("w", encoding="utf-8") as target:
        for source_file in source_files:
            for record in read_records(source_file):
                target.write(json.dumps(record, ensure_ascii=True) + "\n")
                record_count += 1
    if record_count == 0:
        raise ValueError("Input files did not contain any non-empty 'text' values.")
    return record_count, len(source_files)