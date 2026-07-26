from __future__ import annotations

import re
from typing import Any

from src.cat_breed_assistant.evaluation.retrieval.schemas import (
    GeneratedQuestion,
    SourceChunk,
    ValidationResult,
)


def normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query).strip().casefold()
    return normalized.rstrip("?？").strip()


def answer_leaked_in_query(query: str, answer: str) -> bool:
    query_norm = normalize_query(query)
    answer_norm = normalize_query(answer)
    if len(answer_norm) < 4:
        return False
    return answer_norm in query_norm


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", normalized_text(value)))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def question_field(question: Any, field_name: str) -> str:
    value = getattr(question, field_name, "")
    return str(value or "")


def locally_validate_candidate(
    candidate: GeneratedQuestion,
    chunk: SourceChunk,
    seen_queries: set[str],
    existing_for_breed: list[Any] | None = None,
) -> str | None:
    existing_for_breed = existing_for_breed or []
    if len(candidate.query.strip()) < 8:
        return "question_too_short"
    if candidate.evidence_quote not in chunk.text:
        return "evidence_quote_not_found"
    if answer_leaked_in_query(candidate.query, candidate.answer):
        return "answer_leaked_in_question"
    normalized = normalize_query(candidate.query)
    if normalized in seen_queries:
        return "duplicate_normalized_query"
    candidate_question_type = normalized_text(candidate.question_type)
    candidate_evidence = normalized_text(candidate.evidence_quote)
    for existing in existing_for_breed:
        if normalized == normalize_query(question_field(existing, "query")):
            return "duplicate_normalized_query_for_breed"
        if candidate_question_type and candidate_question_type == normalized_text(
            question_field(existing, "question_type")
        ):
            return "duplicate_question_type_for_breed"
        if candidate_evidence and candidate_evidence == normalized_text(
            question_field(existing, "evidence_quote")
        ):
            return "duplicate_evidence_quote_for_breed"
        if jaccard_similarity(candidate.query, question_field(existing, "query")) >= 0.8:
            return "near_duplicate_question_for_breed"
    return None


def validator_accepts(result: ValidationResult) -> bool:
    return bool(
        result.approved
        and result.is_answerable
        and result.answer_supported
        and result.evidence_supported
        and result.question_is_natural
        and not result.question_copies_source
        and not result.answer_leaked_in_question
        and not result.requires_external_knowledge
        and not result.is_ambiguous
        and result.fact_is_distinct_from_existing_questions
        and result.question_type_is_distinct
    )
