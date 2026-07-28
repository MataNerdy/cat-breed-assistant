from __future__ import annotations

import json

from typing import Any

from src.cat_breed_assistant.evaluation.retrieval.question_types import QUESTION_TYPE_VALUES
from src.cat_breed_assistant.evaluation.retrieval.schemas import GeneratedQuestion, SourceChunk


GENERATOR_PROMPT_VERSION = "retrieval_eval_generator_v1"
VALIDATOR_PROMPT_VERSION = "retrieval_eval_validator_v1"


GENERATOR_SYSTEM_PROMPT = """
You create retrieval evaluation questions for a cat breed assistant.
Return only valid JSON. Do not use markdown.
Generate questions that are answerable using only the provided source chunk.
The evidence_quote must be an exact substring of source_chunk.text.
Generate query and answer in the configured target languages even when the source chunk is in another language.
Do not translate evidence_quote.
Do not include the answer inside the question.
Do not require external knowledge.
Use only the allowed normalized question_type values.
""".strip()


VALIDATOR_SYSTEM_PROMPT = """
You validate retrieval evaluation candidates for a cat breed assistant.
Return only valid JSON. Do not use markdown.
Use only the provided source chunk, candidate answer and evidence quote.
Never assign a human approval status. The approved field means only model validation.
Validate target language, exact evidence support, normalized question type, and translation faithfulness.
""".strip()


def question_summary(question: Any) -> dict[str, str]:
    return {
        "query": str(getattr(question, "query", "")),
        "question_type": str(getattr(question, "question_type", "")),
        "answer": str(getattr(question, "answer", "")),
        "evidence_quote": str(getattr(question, "evidence_quote", "")),
        "chunk_id": str(getattr(question, "chunk_id", "")),
    }


def generator_user_prompt(
    chunk: SourceChunk,
    min_questions: int,
    max_questions: int,
    existing_questions: list[Any] | None = None,
    query_language: str = "ru",
    answer_language: str = "ru",
    global_question_type_counts: dict[str, int] | None = None,
) -> str:
    existing_questions = existing_questions or []
    global_question_type_counts = global_question_type_counts or {}
    used_types = sorted(
        {
            summary["question_type"]
            for question in existing_questions
            if (summary := question_summary(question))["question_type"]
        }
    )
    payload = {
        "task": "Generate retrieval evaluation question candidates.",
        "target_language": {
            "query_language": query_language,
            "answer_language": answer_language,
        },
        "constraints": {
            "min_questions": min_questions,
            "max_questions": max_questions,
            "allowed_question_types": list(QUESTION_TYPE_VALUES),
            "language_instruction": (
                "Write query in natural Russian when query_language is ru. "
                "Write answer in Russian when answer_language is ru. "
                "Do not use an English query. Do not mix Russian and English unless a proper noun, "
                "official breed name, organization name, DNA, TICA, WCF, or cattery name requires it. "
                "Keep official names of people, organizations and breeds as written in the evidence. "
                "Use the Russian canonical breed name from metadata when available. "
                "If no approved Russian name is available, use the project canonical name and do not invent a translation."
            ),
            "evidence_instruction": (
                "evidence_quote must be an exact unchanged substring of source_chunk.text. "
                "If the source is English, translate only the meaning of query and answer, not evidence_quote."
            ),
            "atomicity_instruction": (
                "Ask one question about one atomic fact. Do not add facts absent from source_chunk.text."
            ),
            "diversity_instruction": (
                "Choose facts that are distinct from existing questions for this breed. "
                "Avoid repeating question_type, atomic fact, evidence_quote, chunk_id, "
                "or obvious paraphrases when possible. Prefer underrepresented question_type values."
            ),
            "used_question_types_for_same_breed": used_types,
            "global_question_type_counts_in_current_run": global_question_type_counts,
            "schema": {
                "questions": [
                    {
                        "query": "string",
                        "answer": "string",
                        "evidence_quote": "exact substring of source text",
                        "question_type": "string",
                        "difficulty": "easy|medium|hard",
                        "breed_name_present": "boolean",
                    }
                ]
            },
        },
        "existing_questions_for_same_breed": [
            question_summary(question) for question in existing_questions
        ],
        "source_chunk": {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "breed_id": chunk.breed_id,
            "canonical_breed_name": chunk.breed_name,
            "canonical_breed_name_en": chunk.breed_name_en,
            "canonical_breed_name_ru": chunk.breed_name_ru,
            "aliases": chunk.aliases,
            "language": chunk.language,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "text": chunk.text,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validator_user_prompt(
    chunk: SourceChunk,
    candidate: GeneratedQuestion,
    existing_questions: list[Any] | None = None,
    query_language: str = "ru",
    answer_language: str = "ru",
) -> str:
    existing_questions = existing_questions or []
    payload = {
        "task": "Validate one retrieval evaluation candidate.",
        "target_language": {
            "query_language": query_language,
            "answer_language": answer_language,
        },
        "validation_schema": {
            "is_answerable": "boolean",
            "answer_supported": "boolean",
            "evidence_supported": "boolean",
            "question_is_natural": "boolean",
            "question_copies_source": "boolean",
            "answer_leaked_in_question": "boolean",
            "requires_external_knowledge": "boolean",
            "is_ambiguous": "boolean",
            "fact_is_distinct_from_existing_questions": "boolean",
            "question_type_is_distinct": "boolean",
            "query_language_is_correct": "boolean",
            "answer_language_is_correct": "boolean",
            "question_type_is_valid": "boolean",
            "question_type_matches_fact": "boolean",
            "translation_preserves_meaning": "boolean",
            "approved": "boolean",
            "rejection_reasons": ["string"],
        },
        "allowed_question_types": list(QUESTION_TYPE_VALUES),
        "validation_rules": [
            "query must be natural Russian when query_language is ru",
            "answer must be Russian when answer_language is ru",
            "query must not contain unnecessary English fragments",
            "answer must be fully supported by source_chunk.text",
            "evidence_quote must be an exact substring of source_chunk.text",
            "question_type must be one of allowed_question_types",
            "question_type must match the fact being tested",
            "question must test one atomic fact",
            "question must not duplicate existing questions for this breed",
            "question must not require external knowledge",
            "translation must preserve the meaning of source evidence",
            "the Russian breed name must not be invented by the model",
        ],
        "existing_questions_for_same_breed": [
            question_summary(question) for question in existing_questions
        ],
        "source_chunk": {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "breed_id": chunk.breed_id,
            "canonical_breed_name": chunk.breed_name,
            "canonical_breed_name_en": chunk.breed_name_en,
            "canonical_breed_name_ru": chunk.breed_name_ru,
            "language": chunk.language,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "text": chunk.text,
        },
        "candidate": candidate.model_dump(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
