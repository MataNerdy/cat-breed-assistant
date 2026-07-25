from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunking import (
    ChunkingError,
    build_chunks,
    build_chunks_report,
    build_corpus_integrity_report,
    load_broader_overrides,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_INPUT_PATH = Path("data/processed/knowledge_documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/knowledge_chunks.jsonl")
DEFAULT_REPORT_PATH = Path("data/reports/knowledge_chunks_report.json")
DEFAULT_SKIPPED_BROADER_PATH = Path("data/reports/skipped_broader_sources.jsonl")
DEFAULT_BROADER_OVERRIDES_PATH = Path("data/curated/broader_source_chunk_overrides.json")
DEFAULT_INTEGRITY_REPORT_PATH = Path("data/reports/corpus_integrity_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic retrieval chunks from knowledge documents."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--skipped-broader-output",
        type=Path,
        default=DEFAULT_SKIPPED_BROADER_PATH,
    )
    parser.add_argument(
        "--broader-overrides",
        type=Path,
        default=DEFAULT_BROADER_OVERRIDES_PATH,
    )
    parser.add_argument(
        "--integrity-report",
        type=Path,
        default=DEFAULT_INTEGRITY_REPORT_PATH,
    )
    parser.add_argument("--target-chars", type=int, default=2200)
    parser.add_argument("--max-chars", type=int, default=3000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = read_jsonl(args.input)
        broader_overrides = load_broader_overrides(args.broader_overrides)
        chunks, skipped_broader = build_chunks(
            documents,
            broader_overrides=broader_overrides,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
        )
        report = build_chunks_report(chunks, skipped_broader)
        integrity_report = build_corpus_integrity_report(chunks, skipped_broader)
        write_jsonl_atomic(chunks, args.output)
        write_jsonl_atomic(skipped_broader, args.skipped_broader_output)
        write_json_atomic(report, args.report)
        write_json_atomic(integrity_report, args.integrity_report)
    except (OSError, ValueError, json.JSONDecodeError, ChunkingError) as exc:
        print(f"Could not build knowledge chunks: {exc}")
        return 1

    print(f"Input documents: {len(documents)}")
    print(f"Written chunks: {len(chunks)}")
    print(f"Skipped broader documents: {len(skipped_broader)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    print(f"Skipped broader output: {args.skipped_broader_output}")
    print(f"Integrity report: {args.integrity_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
