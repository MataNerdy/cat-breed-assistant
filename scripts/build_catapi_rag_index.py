from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/catapi_chunks.jsonl")
CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "cat_breed_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"CatAPI chunks not found: {INPUT_PATH}. "
            "Run scripts/build_catapi_chunks.py first."
        )

    chunks = []
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {INPUT_PATH}: {exc}"
                ) from exc

    return chunks


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def rebuild_collection(chunks: list[dict[str, Any]]) -> int:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        ids.append(chunk["chunk_id"])
        documents.append(chunk.get("text") or "")
        metadatas.append(
            clean_metadata(
                {
                    "breed_id": chunk.get("breed_id"),
                    "breed_name": chunk.get("breed_name"),
                    "source": chunk.get("source"),
                    "origin": metadata.get("origin"),
                    "wikipedia_url": metadata.get("wikipedia_url"),
                    "reference_image_id": metadata.get("reference_image_id"),
                    "image_url": metadata.get("image_url"),
                    "hairless": metadata.get("hairless"),
                    "shedding_level": metadata.get("shedding_level"),
                    "social_needs": metadata.get("social_needs"),
                    "vocalisation": metadata.get("vocalisation"),
                }
            )
        )

    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection.count()


def main() -> int:
    try:
        chunks = load_chunks()
        count = rebuild_collection(chunks)
    except (OSError, ValueError) as exc:
        print(f"Could not build CatAPI RAG index: {exc}")
        return 1
    except Exception as exc:
        print(f"Could not build CatAPI RAG index. Check dependencies and model cache: {exc}")
        return 1

    print(f"Saved {count} documents to Chroma collection '{COLLECTION_NAME}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
