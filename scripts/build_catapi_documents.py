from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("data/raw/catapi_breeds.json")
OUTPUT_PATH = Path("data/processed/catapi_breed_documents.jsonl")
SOURCE_URL = "https://api.thecatapi.com/v1/breeds"


def load_raw_breeds() -> list[dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Raw CatAPI data not found: {INPUT_PATH}. "
            "Run scripts/fetch_catapi_breeds.py first."
        )

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of breeds in {INPUT_PATH}.")
    return data


def format_score(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def get_image_url(breed: dict[str, Any]) -> str | None:
    image = breed.get("image")
    if isinstance(image, dict):
        url = image.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def build_profile_text(breed: dict[str, Any]) -> str:
    name = breed.get("name") or "Unknown breed"
    description = breed.get("description") or "No description provided."
    temperament = breed.get("temperament") or "unknown"
    origin = breed.get("origin") or "unknown"
    life_span = breed.get("life_span") or "unknown"
    weight = breed.get("weight") if isinstance(breed.get("weight"), dict) else {}
    metric_weight = weight.get("metric") or "unknown"
    imperial_weight = weight.get("imperial") or "unknown"
    wikipedia_url = breed.get("wikipedia_url") or "not provided"

    lines = [
        f"Breed name: {name}",
        f"Description: {description}",
        f"Temperament: {temperament}",
        f"Origin: {origin}",
        f"Life span: {life_span} years",
        f"Weight: {metric_weight} kg ({imperial_weight} lb)",
        f"Grooming level: {format_score(breed.get('grooming'))}",
        f"Energy level: {format_score(breed.get('energy_level'))}",
        f"Health issues score: {format_score(breed.get('health_issues'))}",
        f"Child friendly score: {format_score(breed.get('child_friendly'))}",
        f"Dog friendly score: {format_score(breed.get('dog_friendly'))}",
        f"Stranger friendly score: {format_score(breed.get('stranger_friendly'))}",
        f"Intelligence score: {format_score(breed.get('intelligence'))}",
        f"Hypoallergenic: {format_score(breed.get('hypoallergenic'))}",
        f"Wikipedia URL: {wikipedia_url}",
    ]
    return "\n".join(lines)


def build_document(breed: dict[str, Any]) -> dict[str, Any]:
    breed_id = breed.get("id") or "unknown"
    breed_name = breed.get("name") or "Unknown breed"

    return {
        "doc_id": f"catapi_{breed_id}",
        "breed_id": breed_id,
        "breed_name": breed_name,
        "source": "thecatapi",
        "text": build_profile_text(breed),
        "metadata": {
            "origin": breed.get("origin"),
            "temperament": breed.get("temperament"),
            "life_span": breed.get("life_span"),
            "wikipedia_url": breed.get("wikipedia_url"),
            "reference_image_id": breed.get("reference_image_id"),
            "image_url": get_image_url(breed),
            "source_url": SOURCE_URL,
        },
    }


def save_documents(documents: list[dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for document in documents:
            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")


def main() -> int:
    try:
        breeds = load_raw_breeds()
        documents = [build_document(breed) for breed in breeds]
        save_documents(documents)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not build CatAPI documents: {exc}")
        return 1

    print(f"Saved {len(documents)} CatAPI documents to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
