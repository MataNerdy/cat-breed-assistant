from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.cat_breed_assistant.evaluation.retrieval.question_types import (
    QUESTION_TYPE_VALUES,
    SAFE_QUESTION_TYPE_MAPPING,
    RetrievalQuestionType,
)


MigrationStatus = Literal["canonical", "normalized", "needs_review"]

AMBIGUOUS_QUESTION_TYPE_VALUES = {
    "boolean",
    "description",
    "factoid",
    "factual",
    "numeric_rating",
    "rating_retrieval",
}


@dataclass(frozen=True)
class QuestionTypeMigration:
    raw_question_type: str
    normalized_question_type: str | None
    status: MigrationStatus
    review_required: bool
    reason: str


def canonicalize_question_type_token(raw_question_type: str) -> str:
    normalized = raw_question_type.strip().casefold().replace("-", "_")
    return "_".join(normalized.split())


def normalize_legacy_question_type(raw_question_type: str) -> QuestionTypeMigration:
    raw = str(raw_question_type or "")
    normalized = canonicalize_question_type_token(raw)

    if not normalized:
        return QuestionTypeMigration(
            raw_question_type=raw,
            normalized_question_type=None,
            status="needs_review",
            review_required=True,
            reason="empty question_type",
        )

    try:
        canonical = RetrievalQuestionType(normalized)
    except ValueError:
        canonical = None

    if canonical is not None:
        return QuestionTypeMigration(
            raw_question_type=raw,
            normalized_question_type=canonical.value,
            status="canonical",
            review_required=False,
            reason="already canonical",
        )

    mapped = SAFE_QUESTION_TYPE_MAPPING.get(normalized)
    if mapped is not None:
        return QuestionTypeMigration(
            raw_question_type=raw,
            normalized_question_type=mapped.value,
            status="normalized",
            review_required=False,
            reason=f"mapped from legacy value {normalized!r}",
        )

    if normalized in AMBIGUOUS_QUESTION_TYPE_VALUES:
        reason = f"ambiguous legacy value {normalized!r}; human review must choose a canonical type"
    else:
        reason = f"unknown question_type {normalized!r}; human review must choose a canonical type"

    return QuestionTypeMigration(
        raw_question_type=raw,
        normalized_question_type=None,
        status="needs_review",
        review_required=True,
        reason=reason,
    )


def is_canonical_question_type(value: str) -> bool:
    return value in QUESTION_TYPE_VALUES

