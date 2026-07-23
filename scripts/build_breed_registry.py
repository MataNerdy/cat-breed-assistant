from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
DEFAULT_INPUT_PATH = Path("data/raw/catapi_breeds.json")
DEFAULT_OUTPUT_PATH = Path("data/curated/breed_registry.jsonl")


class DuplicateBreedIDError(ValueError):
    """Raised when the CatAPI snapshot contains duplicated breed ids."""


def load_catapi_records(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected root JSON object to be a list: {path}")
    return data


def normalize_aliases(alt_names: Any, primary_name: str) -> list[str]:
    if not isinstance(alt_names, str):
        return []

    aliases = []
    seen = set()
    primary_normalized = primary_name.strip().casefold()

    for raw_alias in alt_names.split(","):
        alias = raw_alias.strip()
        normalized = alias.casefold()
        if not alias or normalized == primary_normalized or normalized in seen:
            continue
        aliases.append(alias)
        seen.add(normalized)

    return aliases


def build_registry_record(
    record: dict[str, Any],
    breed_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    breed_id = breed_id or record["id"]
    name = name or record["name"]

    return {
        "schema_version": SCHEMA_VERSION,
        "breed_id": breed_id,
        "name_en": name,
        "name_ru": None,
        "aliases_en": normalize_aliases(record.get("alt_names"), name),
        "aliases_ru": [],
        "catapi": {
            "id": breed_id,
            "raw": record,
        },
        "wikidata": None,
        "sources": ["thecatapi"],
        "review": {
            "status": "not_enriched",
            "warnings": [],
        },
    }


def parse_breed_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None

    breed_ids = {item.strip() for item in value.split(",") if item.strip()}
    return breed_ids or None


def build_registry(
    records: list[Any],
    breed_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    registry = []
    seen_ids = set()
    skipped_records = 0
    duplicate_ids = 0

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            print(f"Warning: skipping record {index}: expected object.")
            skipped_records += 1
            continue

        breed_id = record.get("id")
        name = record.get("name")

        if not isinstance(breed_id, str) or not breed_id.strip():
            print(f"Warning: skipping record {index}: missing non-empty id.")
            skipped_records += 1
            continue

        if breed_id in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(breed_id)

        breed_id = breed_id.strip()
        if breed_ids is not None and breed_id not in breed_ids:
            continue

        if not isinstance(name, str) or not name.strip():
            print(f"Warning: skipping record {index}: missing non-empty name.")
            skipped_records += 1
            continue

        registry.append(build_registry_record(record, breed_id=breed_id, name=name.strip()))

    if duplicate_ids:
        raise DuplicateBreedIDError(f"Duplicate CatAPI breed ids found: {duplicate_ids}")

    registry.sort(key=lambda item: item["breed_id"])
    return registry, skipped_records, duplicate_ids


def write_jsonl_atomic(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic base breed registry from local CatAPI snapshot."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--breed-ids",
        help="Optional comma-separated CatAPI breed ids to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        records = load_catapi_records(args.input)
        registry, skipped_records, duplicate_ids = build_registry(
            records,
            breed_ids=parse_breed_ids(args.breed_ids),
        )
        write_jsonl_atomic(registry, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not build breed registry: {exc}")
        return 1

    print(f"Input records: {len(records)}")
    print(f"Written records: {len(registry)}")
    print(f"Skipped records: {skipped_records}")
    print(f"Duplicate IDs: {duplicate_ids}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
