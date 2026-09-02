from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cat_breed_assistant.evaluation.retrieval.io import (  # noqa: E402
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from src.cat_breed_assistant.evaluation.retrieval.migration import (  # noqa: E402
    is_canonical_question_type,
    normalize_legacy_question_type,
)
from src.cat_breed_assistant.evaluation.retrieval.review import (  # noqa: E402
    HumanReviewRecord,
    ReviewDecision,
    load_review_records,
)

DEFAULT_CANDIDATES_PATH = Path("data/evaluation/retrieval/v1/pilot_candidates.jsonl")
DEFAULT_REVIEWS_PATH = Path("data/evaluation/retrieval/v1/human_review_decisions.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/knowledge_chunks.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/evaluation/retrieval/benchmark_v1")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reviewed_value(
    candidate: dict[str, Any],
    review: HumanReviewRecord,
    field_name: str,
    edited_field_name: str,
) -> Any:
    value = getattr(review, edited_field_name)
    return value if review.decision == ReviewDecision.NEEDS_EDIT and review.edit_approved and value else candidate[field_name]


def _normalized_question_type(
    candidate: dict[str, Any],
    review: HumanReviewRecord,
) -> tuple[str, dict[str, Any]]:
    migration = normalize_legacy_question_type(str(candidate.get("question_type", "")))
    normalized = review.normalized_question_type or migration.normalized_question_type
    if normalized is None or not is_canonical_question_type(normalized):
        raise ValueError(
            f"{candidate.get('query_id')}: question_type requires human normalization "
            f"for raw value {migration.raw_question_type!r}"
        )
    return normalized, {
        "raw_question_type": migration.raw_question_type,
        "normalization_status": migration.status,
        "normalization_reason": migration.reason,
        "review_normalized_question_type": review.normalized_question_type,
    }


def build_benchmark_examples(
    candidates: list[dict[str, Any]],
    reviews: dict[str, HumanReviewRecord],
    chunk_ids: set[str],
) -> list[dict[str, Any]]:
    examples = []
    for candidate in sorted(candidates, key=lambda item: item["query_id"]):
        review = reviews.get(candidate["query_id"])
        if review is None:
            continue
        if review.decision == ReviewDecision.REJECTED or review.decision == ReviewDecision.SKIPPED:
            continue
        if review.decision == ReviewDecision.NEEDS_EDIT and not review.edit_approved:
            continue

        normalized_question_type, migration_metadata = _normalized_question_type(candidate, review)
        relevant_chunk_ids = _reviewed_value(
            candidate,
            review,
            "relevant_chunk_ids",
            "edited_relevant_chunk_ids",
        )
        missing_chunk_ids = [chunk_id for chunk_id in relevant_chunk_ids if chunk_id not in chunk_ids]
        if missing_chunk_ids:
            raise ValueError(
                f"{candidate['query_id']}: missing relevant chunk ids: {', '.join(missing_chunk_ids)}"
            )

        examples.append(
            {
                "schema_version": "1.0",
                "query_id": candidate["query_id"],
                "query": _reviewed_value(candidate, review, "query", "edited_query"),
                "answer": _reviewed_value(candidate, review, "answer", "edited_answer"),
                "relevant_chunk_ids": relevant_chunk_ids,
                "breed_id": candidate["breed_id"],
                "breed_name": candidate["breed_name"],
                "language": candidate["language"],
                "question_type": normalized_question_type,
                "difficulty": candidate["difficulty"],
                "source_candidate": {
                    "chunk_id": candidate["chunk_id"],
                    "document_id": candidate["document_id"],
                    "generator_provider": candidate["generator_provider"],
                    "generator_model": candidate["generator_model"],
                    "validator_provider": candidate["validator_provider"],
                    "validator_model": candidate["validator_model"],
                    "run_id": candidate["run_id"],
                    "source_chunk_hash": candidate["source_chunk_hash"],
                },
                "human_review": {
                    "decision": review.decision.value,
                    "reviewer_note": review.reviewer_note,
                    "edit_approved": review.edit_approved,
                    "reviewed_at": review.reviewed_at,
                },
                "migration": migration_metadata,
            }
        )
    return examples


def build_manifest(
    examples: list[dict[str, Any]],
    candidates_path: Path,
    reviews_path: Path,
    chunks_path: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "benchmark_version": "v1",
        "created_at": created_at,
        "number_of_examples": len(examples),
        "source_candidate_artifact": str(candidates_path),
        "source_candidate_artifact_sha256": sha256_file(candidates_path),
        "human_review_artifact": str(reviews_path),
        "human_review_artifact_sha256": sha256_file(reviews_path) if reviews_path.exists() else None,
        "chunks_artifact": str(chunks_path),
        "chunks_artifact_sha256": sha256_file(chunks_path),
        "counts_by_question_type": dict(Counter(example["question_type"] for example in examples)),
        "counts_by_breed": dict(Counter(example["breed_id"] for example in examples)),
        "counts_by_difficulty": dict(Counter(example["difficulty"] for example in examples)),
        "counts_by_language": dict(Counter(example["language"] for example in examples)),
    }


def export_benchmark(
    candidates_path: Path,
    reviews_path: Path,
    chunks_path: Path,
    output_dir: Path,
    created_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = read_jsonl(candidates_path)
    reviews = load_review_records(reviews_path)
    chunk_ids = {record["chunk_id"] for record in read_jsonl(chunks_path)}
    examples = build_benchmark_examples(candidates, reviews, chunk_ids)
    created_at = created_at or utc_now()
    manifest = build_manifest(examples, candidates_path, reviews_path, chunks_path, created_at)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(examples, output_dir / "benchmark.jsonl")
    write_json_atomic(manifest, output_dir / "manifest.json")
    return examples, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export human-reviewed retrieval benchmark v1.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples, _ = export_benchmark(
        candidates_path=args.candidates,
        reviews_path=args.reviews,
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        created_at=args.created_at,
    )
    print(f"Exported examples: {len(examples)}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
