"""Azure ML batch scoring entry point for Azure OpenAI embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI


BATCH_SIZE = 16


def init() -> None:
    global client
    global deployment_name
    global output_file

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    output_file = Path(os.environ["AZUREML_BI_OUTPUT_PATH"]) / "embeddings.jsonl"


def batched(values: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def run(mini_batch: list[str]) -> list[str]:
    results: list[dict[str, object]] = []
    for file_name in mini_batch:
        path = Path(file_name)
        if path.suffix.lower() != ".jsonl":
            continue
        with path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        for record_batch in batched(records, BATCH_SIZE):
            response = client.embeddings.create(
                model=deployment_name,
                input=[record["text"] for record in record_batch],
            )
            embeddings = sorted(response.data, key=lambda item: item.index)
            results.extend(
                {
                    "id": record["id"],
                    "embedding": embedding.embedding,
                    "source_file": path.name,
                }
                for record, embedding in zip(record_batch, embeddings, strict=True)
            )

    if results:
        with output_file.open("a", encoding="utf-8") as target:
            for result in results:
                target.write(json.dumps(result) + "\n")
    return mini_batch