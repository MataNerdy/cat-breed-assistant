from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wikidata_client import WikidataClient, WikidataClientError
from src.data.wikidata_resolver import (
    TARGET_BREED_IDS,
    ensure_overrides_file,
    load_overrides,
    load_registry,
    resolve_registry_records,
)


DEFAULT_REGISTRY_PATH = Path("data/curated/breed_registry.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/staging/wikidata_enrichment.jsonl")
DEFAULT_UNRESOLVED_PATH = Path("data/reports/wikidata_unresolved.jsonl")
DEFAULT_OVERRIDES_PATH = Path("data/curated/wikidata_overrides.json")
DEFAULT_CACHE_DIR = Path("data/cache/wikidata")


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
        description="Build Wikidata staging enrichment for selected CatAPI breeds."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--unresolved-output", type=Path, default=DEFAULT_UNRESOLVED_PATH)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        ensure_overrides_file(args.overrides)
        overrides = load_overrides(args.overrides)
        registry_records = load_registry(args.registry, set(TARGET_BREED_IDS))
        client = WikidataClient(
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        )
        enrichment = resolve_registry_records(registry_records, client, overrides)
        unresolved = [
            record for record in enrichment if record["match_method"] == "unresolved"
        ]
        write_jsonl_atomic(enrichment, args.output)
        write_jsonl_atomic(unresolved, args.unresolved_output)
    except (OSError, ValueError, WikidataClientError, json.JSONDecodeError) as exc:
        print(f"Could not build Wikidata enrichment: {exc}")
        return 1

    print(f"Input registry records: {len(registry_records)}")
    print(f"Written enrichment records: {len(enrichment)}")
    print(f"Unresolved records: {len(unresolved)}")
    print(f"Output: {args.output}")
    print(f"Unresolved output: {args.unresolved_output}")
    print(f"Cache dir: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
