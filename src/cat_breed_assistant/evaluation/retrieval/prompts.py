from __future__ import annotations

import json

from typing import Any

from src.cat_breed_assistant.evaluation.retrieval.schemas import GeneratedQuestion, SourceChunk


GENERATOR_PROMPT_VERSION = "retrieval_eval_generator_v1"
VALIDATOR_PROMPT_VERSION = "retrieval_eval_validator_v1"


GENERATOR_SYSTEM_PROMPT = """
You create retrieval evaluation questions for a cat breed assistant.
Return only valid JSON. Do not use markdown.
Generate questions that are answerable using only the provided source chunk.
The evidence_quote must be an exact substring of source_chunk.text.
Keep the question language the same as the source chunk language.
Do not include the answer inside the question.
Do not require external knowledge.
""".strip()


VALIDATOR_SYSTEM_PROMPT = """
You validate retrieval evaluation candidates for a cat breed assistant.
Return only valid JSON. Do not use markdown.
Use only the provided source chunk, candidate answer and evidence quote.
Never assign a human approval status. The approved field means only model validation.
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
) -> str:
    existing_questions = existing_questions or []
    payload = {
        "task": "Generate retrieval evaluation question candidates.",
        "constraints": {
            "min_questions": min_questions,
            "max_questions": max_questions,
            "diversity_instruction": (
                "Choose facts that are distinct from existing questions for this breed. "
                "Avoid repeating question_type, atomic fact, evidence_quote, chunk_id, "
                "or obvious paraphrases when possible."
            ),
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
) -> str:
    existing_questions = existing_questions or []
    payload = {
        "task": "Validate one retrieval evaluation candidate.",
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
            "approved": "boolean",
            "rejection_reasons": ["string"],
        },
        "existing_questions_for_same_breed": [
            question_summary(question) for question in existing_questions
        ],
        "source_chunk": {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "breed_id": chunk.breed_id,
            "canonical_breed_name": chunk.breed_name,
            "language": chunk.language,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "text": chunk.text,
        },
        "candidate": candidate.model_dump(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
