from __future__ import annotations

import json
from pathlib import Path

from scripts.export_retrieval_benchmark import export_benchmark
from src.cat_breed_assistant.evaluation.retrieval.io import read_jsonl, write_jsonl_atomic
from src.cat_breed_assistant.evaluation.retrieval.review import (
    HumanReviewRecord,
    ReviewDecision,
    load_review_records,
    upsert_review_record,
    write_review_records_atomic,
)


def _candidate(query_id: str, status: str = "pending_review") -> dict:
    return {
        "query_id": query_id,
        "query": f"Question {query_id}?",
        "chunk_id": "bsho:wikipedia:en:0-lead:000",
        "document_id": "bsho:wikipedia:en",
        "breed_id": "bsho",
        "breed_name": "British Shorthair",
        "relevant_chunk_ids": ["bsho:wikipedia:en:0-lead:000"],
        "answer": f"Answer {query_id}",
        "evidence_quote": "British Shorthair evidence",
        "language": "ru",
        "question_type": "Physical Trait",
        "difficulty": "easy",
        "breed_name_present": True,
        "generator_provider": "gemini",
        "generator_model": "gemini-2.5-flash",
        "validator_provider": "mistral",
        "validator_model": "mistral-small-latest",
        "generator_prompt_version": "retrieval_eval_generator_v1",
        "validator_prompt_version": "retrieval_eval_validator_v1",
        "validation": {"approved": True},
        "status": status,
        "source_chunk_hash": "abc123",
        "created_at": "2026-08-10T00:00:00Z",
        "run_id": "test-run",
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    candidates_path = tmp_path / "pilot_candidates.jsonl"
    chunks_path = tmp_path / "knowledge_chunks.jsonl"
    write_jsonl_atomic([_candidate("q_2"), _candidate("q_1")], candidates_path)
    write_jsonl_atomic(
        [
            {
                "chunk_id": "bsho:wikipedia:en:0-lead:000",
                "text": "British Shorthair chunk",
            }
        ],
        chunks_path,
    )
    return candidates_path, chunks_path


def test_review_decision_persistence(tmp_path: Path) -> None:
    reviews_path = tmp_path / "human_review_decisions.jsonl"
    reviews = upsert_review_record(
        {},
        HumanReviewRecord(
            query_id="q_1",
            decision=ReviewDecision.APPROVED,
            reviewer_note="looks good",
            normalized_question_type="physical_characteristic",
        ),
    )

    write_review_records_atomic(reviews_path, reviews)
    loaded = load_review_records(reviews_path)

    assert loaded["q_1"].decision == ReviewDecision.APPROVED
    assert loaded["q_1"].reviewer_note == "looks good"


def test_benchmark_export_includes_only_human_approved_records(tmp_path: Path) -> None:
    candidates_path, chunks_path = _write_inputs(tmp_path)
    reviews_path = tmp_path / "human_review_decisions.jsonl"
    output_dir = tmp_path / "benchmark"
    write_review_records_atomic(
        reviews_path,
        {
            "q_1": HumanReviewRecord(
                query_id="q_1",
                decision=ReviewDecision.APPROVED,
                normalized_question_type="physical_characteristic",
                reviewed_at="2026-08-11T00:00:00Z",
            ),
            "q_2": HumanReviewRecord(
                query_id="q_2",
                decision=ReviewDecision.REJECTED,
                reviewed_at="2026-08-11T00:00:00Z",
            ),
        },
    )

    examples, manifest = export_benchmark(
        candidates_path=candidates_path,
        reviews_path=reviews_path,
        chunks_path=chunks_path,
        output_dir=output_dir,
        created_at="2026-08-12T00:00:00Z",
    )

    assert [example["query_id"] for example in examples] == ["q_1"]
    assert examples[0]["question_type"] == "physical_characteristic"
    assert manifest["number_of_examples"] == 1
    assert read_jsonl(output_dir / "benchmark.jsonl") == examples


def test_benchmark_export_uses_approved_edits(tmp_path: Path) -> None:
    candidates_path, chunks_path = _write_inputs(tmp_path)
    reviews_path = tmp_path / "human_review_decisions.jsonl"
    output_dir = tmp_path / "benchmark"
    write_review_records_atomic(
        reviews_path,
        {
            "q_1": HumanReviewRecord(
                query_id="q_1",
                decision=ReviewDecision.NEEDS_EDIT,
                edited_query="Edited question?",
                edited_answer="Edited answer",
                edited_relevant_chunk_ids=["bsho:wikipedia:en:0-lead:000"],
                normalized_question_type="physical_characteristic",
                edit_approved=True,
                reviewed_at="2026-08-11T00:00:00Z",
            )
        },
    )

    examples, _ = export_benchmark(
        candidates_path=candidates_path,
        reviews_path=reviews_path,
        chunks_path=chunks_path,
        output_dir=output_dir,
        created_at="2026-08-12T00:00:00Z",
    )

    assert examples[0]["query"] == "Edited question?"
    assert examples[0]["answer"] == "Edited answer"


def test_export_does_not_modify_candidate_artifact(tmp_path: Path) -> None:
    candidates_path, chunks_path = _write_inputs(tmp_path)
    reviews_path = tmp_path / "human_review_decisions.jsonl"
    before = candidates_path.read_bytes()
    write_review_records_atomic(
        reviews_path,
        {
            "q_1": HumanReviewRecord(
                query_id="q_1",
                decision=ReviewDecision.APPROVED,
                normalized_question_type="physical_characteristic",
            )
        },
    )

    export_benchmark(
        candidates_path=candidates_path,
        reviews_path=reviews_path,
        chunks_path=chunks_path,
        output_dir=tmp_path / "benchmark",
        created_at="2026-08-12T00:00:00Z",
    )

    assert candidates_path.read_bytes() == before


def test_benchmark_export_is_deterministic_with_fixed_timestamp(tmp_path: Path) -> None:
    candidates_path, chunks_path = _write_inputs(tmp_path)
    reviews_path = tmp_path / "human_review_decisions.jsonl"
    write_review_records_atomic(
        reviews_path,
        {
            "q_2": HumanReviewRecord(
                query_id="q_2",
                decision=ReviewDecision.APPROVED,
                normalized_question_type="physical_characteristic",
            ),
            "q_1": HumanReviewRecord(
                query_id="q_1",
                decision=ReviewDecision.APPROVED,
                normalized_question_type="physical_characteristic",
            ),
        },
    )

    export_benchmark(
        candidates_path=candidates_path,
        reviews_path=reviews_path,
        chunks_path=chunks_path,
        output_dir=tmp_path / "benchmark_a",
        created_at="2026-08-12T00:00:00Z",
    )
    export_benchmark(
        candidates_path=candidates_path,
        reviews_path=reviews_path,
        chunks_path=chunks_path,
        output_dir=tmp_path / "benchmark_b",
        created_at="2026-08-12T00:00:00Z",
    )

    assert (tmp_path / "benchmark_a" / "benchmark.jsonl").read_bytes() == (
        tmp_path / "benchmark_b" / "benchmark.jsonl"
    ).read_bytes()
    assert json.loads((tmp_path / "benchmark_a" / "manifest.json").read_text()) == json.loads(
        (tmp_path / "benchmark_b" / "manifest.json").read_text()
    )
