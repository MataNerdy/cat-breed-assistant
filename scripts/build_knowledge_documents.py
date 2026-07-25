from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.knowledge_documents import (
    KnowledgeDocumentError,
    build_knowledge_documents,
    build_knowledge_documents_report,
    load_name_overrides,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from src.data.source_scope import load_scope_overrides


DEFAULT_REGISTRY_PATH = Path("data/curated/breed_registry.jsonl")
DEFAULT_WIKIDATA_PATH = Path("data/staging/wikidata_enrichment.jsonl")
DEFAULT_WIKIPEDIA_PATH = Path("data/staging/wikipedia_articles.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/knowledge_documents.jsonl")
DEFAULT_REPORT_PATH = Path("data/reports/knowledge_documents_report.json")
DEFAULT_SCOPE_OVERRIDES_PATH = Path("data/curated/broader_source_chunk_overrides.json")
DEFAULT_NAME_OVERRIDES_PATH = Path("data/curated/breed_name_overrides.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized knowledge documents from curated/staging sources."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--wikidata", type=Path, default=DEFAULT_WIKIDATA_PATH)
    parser.add_argument("--wikipedia", type=Path, default=DEFAULT_WIKIPEDIA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--scope-overrides",
        type=Path,
        default=DEFAULT_SCOPE_OVERRIDES_PATH,
    )
    parser.add_argument("--name-overrides", type=Path, default=DEFAULT_NAME_OVERRIDES_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry_records = read_jsonl(args.registry)
        wikidata_records = read_jsonl(args.wikidata)
        wikipedia_records = read_jsonl(args.wikipedia)
        scope_overrides = load_scope_overrides(args.scope_overrides)
        name_overrides = load_name_overrides(
            args.name_overrides,
            {record["breed_id"] for record in registry_records},
        )
        documents = build_knowledge_documents(
            registry_records=registry_records,
            wikidata_records=wikidata_records,
            wikipedia_records=wikipedia_records,
            scope_overrides=scope_overrides,
            name_overrides=name_overrides,
        )
        report = build_knowledge_documents_report(documents)
        write_jsonl_atomic(documents, args.output)
        write_json_atomic(report, args.report)
    except (OSError, ValueError, json.JSONDecodeError, KnowledgeDocumentError) as exc:
        print(f"Could not build knowledge documents: {exc}")
        return 1

    print(f"Registry records: {len(registry_records)}")
    print(f"Wikidata records: {len(wikidata_records)}")
    print(f"Wikipedia records: {len(wikipedia_records)}")
    print(f"Scope overrides: {args.scope_overrides}")
    print(f"Name overrides: {args.name_overrides}")
    print(f"Written knowledge documents: {len(documents)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
