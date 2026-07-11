from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("data/processed/catapi_breed_documents.jsonl")
OUTPUT_PATH = Path("data/processed/catapi_chunks.jsonl")


def load_documents() -> list[dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"CatAPI documents not found: {INPUT_PATH}. "
            "Run scripts/build_catapi_documents.py first."
        )

    documents = []
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {INPUT_PATH}: {exc}"
                ) from exc
            documents.append(document)

    return documents


def build_chunk(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    breed_id = document.get("breed_id") or "unknown"
    breed_name = document.get("breed_name") or "Unknown breed"
    source = document.get("source") or "thecatapi"

    return {
        "chunk_id": f"catapi_{breed_id}_profile",
        "doc_id": document.get("doc_id") or f"catapi_{breed_id}",
        "breed_id": breed_id,
        "breed_name": breed_name,
        "source": source,
        "text": document.get("text") or "",
        "metadata": {
            "breed_id": breed_id,
            "breed_name": breed_name,
            "source": source,
            "origin": metadata.get("origin"),
            "reference_image_id": metadata.get("reference_image_id"),
            "wikipedia_url": metadata.get("wikipedia_url"),
            "image_url": metadata.get("image_url"),
            "hairless": metadata.get("hairless"),
            "shedding_level": metadata.get("shedding_level"),
            "social_needs": metadata.get("social_needs"),
            "vocalisation": metadata.get("vocalisation"),
        },
    }


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> int:
    try:
        documents = load_documents()
        chunks = [build_chunk(document) for document in documents]
        save_chunks(chunks)
    except (OSError, ValueError) as exc:
        print(f"Could not build CatAPI chunks: {exc}")
        return 1

    print(f"Saved {len(chunks)} CatAPI chunks to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
