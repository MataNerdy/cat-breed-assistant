from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.cat_breed_assistant.evaluation.retrieval.io import read_jsonl
from src.cat_breed_assistant.evaluation.retrieval.question_types import QUESTION_TYPE_VALUES


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"
    SKIPPED = "skipped"


class HumanReviewRecord(BaseModel):
    query_id: str
    decision: ReviewDecision
    reviewer_note: str = ""
    edited_query: str | None = None
    edited_answer: str | None = None
    edited_relevant_chunk_ids: list[str] | None = None
    normalized_question_type: str | None = None
    edit_approved: bool = False
    reviewed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    @field_validator("query_id", "reviewer_note", mode="before")
    @classmethod
    def normalize_string(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("edited_query", "edited_answer", "normalized_question_type", mode="before")
    @classmethod
    def normalize_optional_string(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("normalized_question_type")
    @classmethod
    def validate_normalized_question_type(cls, value: str | None) -> str | None:
        if value is not None and value not in QUESTION_TYPE_VALUES:
            raise ValueError("normalized_question_type must be canonical")
        return value

    @field_validator("edited_relevant_chunk_ids")
    @classmethod
    def normalize_chunk_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        chunk_ids = [str(chunk_id).strip() for chunk_id in value if str(chunk_id).strip()]
        return chunk_ids or None


def load_review_records(path: Path) -> dict[str, HumanReviewRecord]:
    records = {}
    for record in read_jsonl(path):
        review_record = HumanReviewRecord.model_validate(record)
        records[review_record.query_id] = review_record
    return records


def write_review_records_atomic(
    path: Path,
    records: dict[str, HumanReviewRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            for query_id in sorted(records):
                output_file.write(records[query_id].model_dump_json() + "\n")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def upsert_review_record(
    records: dict[str, HumanReviewRecord],
    record: HumanReviewRecord,
) -> dict[str, HumanReviewRecord]:
    updated = dict(records)
    updated[record.query_id] = record
    return updated

