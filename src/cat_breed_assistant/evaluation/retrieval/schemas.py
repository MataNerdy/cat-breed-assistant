from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.cat_breed_assistant.evaluation.retrieval.question_types import (
    RetrievalQuestionType,
    all_question_type_counts,
    normalize_question_type,
)


Difficulty = Literal["easy", "medium", "hard"]
FinalStatus = Literal["pending_review", "approved", "rejected"]
ProviderName = Literal["gemini", "mistral"]


class ModelConfig(BaseModel):
    provider: ProviderName
    model: str
    temperature: float = 0.0
    timeout_seconds: int = 60
    max_retries: int = 2


class GenerationConfig(BaseModel):
    query_language: str = "ru"
    answer_language: str = "ru"


class PilotConfig(BaseModel):
    input_chunks_path: str = "data/processed/knowledge_chunks.jsonl"
    skipped_broader_sources_path: str = "data/reports/skipped_broader_sources.jsonl"
    output_dir: str = "data/evaluation/retrieval/v1"
    seed: int = 42
    breed_limit: int = 10
    target_count: int = 30
    questions_per_chunk_min: int = 1
    questions_per_chunk_max: int = 3
    generator_prompt_version: str = "retrieval_eval_generator_v1"
    validator_prompt_version: str = "retrieval_eval_validator_v1"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    generator_a: ModelConfig
    validator_a: ModelConfig
    generator_b: ModelConfig
    validator_b: ModelConfig

    @model_validator(mode="after")
    def validate_provider_pairs(self) -> "PilotConfig":
        if self.generator_a.provider == self.validator_a.provider:
            raise ValueError("generator_a and validator_a must use different providers")
        if self.generator_b.provider == self.validator_b.provider:
            raise ValueError("generator_b and validator_b must use different providers")
        return self


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    breed_id: str
    breed_name_en: str
    breed_name_ru: str | None = None
    aliases: list[str] = Field(default_factory=list)
    language: str
    source: str
    chunk_type: str
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    text_length: int | None = None

    @property
    def breed_name(self) -> str:
        return self.breed_name_ru if self.language == "ru" and self.breed_name_ru else self.breed_name_en


class SelectedChunk(BaseModel):
    chunk: SourceChunk
    selection_reason: str
    selection_index: int


class GeneratedQuestion(BaseModel):
    query: str
    answer: str
    evidence_quote: str
    question_type: RetrievalQuestionType
    difficulty: Difficulty
    breed_name_present: bool

    @field_validator("query", "answer", "evidence_quote")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @field_validator("question_type", mode="before")
    @classmethod
    def normalize_question_type_value(cls, value: object) -> RetrievalQuestionType:
        if isinstance(value, RetrievalQuestionType):
            return value
        if not isinstance(value, str):
            raise ValueError("invalid_question_type")
        normalized = normalize_question_type(value)
        if normalized is None:
            raise ValueError("invalid_question_type")
        return normalized


class GeneratedQuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]


class ValidationResult(BaseModel):
    is_answerable: bool
    answer_supported: bool
    evidence_supported: bool
    question_is_natural: bool
    question_copies_source: bool
    answer_leaked_in_question: bool
    requires_external_knowledge: bool
    is_ambiguous: bool
    fact_is_distinct_from_existing_questions: bool = True
    question_type_is_distinct: bool = True
    query_language_is_correct: bool = True
    answer_language_is_correct: bool = True
    question_type_is_valid: bool = True
    question_type_matches_fact: bool = True
    translation_preserves_meaning: bool = True
    approved: bool
    rejection_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_reasons(self) -> "ValidationResult":
        self.rejection_reasons = [reason.strip() for reason in self.rejection_reasons if reason.strip()]
        return self


class RetrievalEvaluationCandidate(BaseModel):
    query_id: str
    query: str
    chunk_id: str
    document_id: str
    breed_id: str
    breed_name: str
    relevant_chunk_ids: list[str]
    answer: str
    evidence_quote: str
    language: str
    question_type: str
    difficulty: Difficulty
    breed_name_present: bool
    generator_provider: str
    generator_model: str
    validator_provider: str
    validator_model: str
    generator_prompt_version: str
    validator_prompt_version: str
    validation: ValidationResult
    status: FinalStatus
    source_chunk_hash: str
    created_at: str
    run_id: str

    @model_validator(mode="after")
    def auto_candidates_are_pending_review(self) -> "RetrievalEvaluationCandidate":
        if self.status != "pending_review":
            raise ValueError("automatic evaluation candidates must be pending_review")
        if self.relevant_chunk_ids != [self.chunk_id]:
            raise ValueError("pilot candidates must reference exactly their source chunk")
        return self


class RejectionRecord(BaseModel):
    chunk_id: str
    stage: Literal["generator", "local_validation", "validator"]
    reason: str
    raw_output: str | None = None
    run_id: str


class ProviderErrorRecord(BaseModel):
    error_type: Literal["quota_exhausted", "timeout", "provider_error"]
    provider: str
    retryable: bool
    chunk_id: str
    run_id: str
    stage: Literal["generator", "validator"]
    reason: str
    created_at: str


class RunManifest(BaseModel):
    run_id: str
    seed: int
    started_at: str
    finished_at: str | None = None
    input_chunks_path: str
    input_chunks_sha256: str
    selected_breed_ids: list[str]
    selected_chunk_ids: list[str]
    generator_models: dict[str, str]
    validator_models: dict[str, str]
    prompt_versions: dict[str, str]
    generated_count: int = 0
    validator_approved_count: int = 0
    pending_review_count: int = 0
    rejected_count: int = 0
    total_available_breeds: int | None = None
    previously_covered_breeds: int | None = None
    target_unused_breeds: int | None = None
    newly_covered_breeds: int | None = None
    fully_covered_breeds: int | None = None
    partially_covered_breeds: int | None = None
    questions_per_breed: int | None = None
    coverage_by_breed: dict[str, dict[str, Any]] = Field(default_factory=dict)
    query_language: str = "ru"
    answer_language: str = "ru"
    generated_by_question_type: dict[str, int] = Field(default_factory=all_question_type_counts)
    rejected_invalid_question_type: int = 0
    rejected_wrong_query_language: int = 0
    rejected_wrong_answer_language: int = 0
    provider_error_count: int = 0
    stopped_by_circuit_breaker: bool = False


class DryRunResult(BaseModel):
    selected_breed_ids: list[str]
    selected_chunk_ids: list[str]
    api_key_available: dict[str, bool]
    input_chunks_sha256: str
    query_language: str = "ru"
    answer_language: str = "ru"
    allowed_question_types: list[str] = Field(default_factory=list)


class UnusedBreedDryRunResult(BaseModel):
    already_covered_breed_ids: list[str]
    unused_breed_ids: list[str]
    partially_covered_breed_ids: list[str]
    target_breed_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids_by_breed: dict[str, list[str]]
    chunk_counts_by_source: dict[str, int]
    shortfalls_by_breed: dict[str, str]
    requested_candidate_slots: int = 0
    expected_candidates_from_unused_breeds: int = 0
    expected_top_up_candidates_for_partially_covered_breeds: int = 0
    unavailable_candidate_slots: int = 0
    expected_max_candidates: int
    approximate_generator_calls: int
    approximate_validator_calls: int
    api_key_available: dict[str, bool]
    input_chunks_sha256: str
    query_language: str = "ru"
    answer_language: str = "ru"
    allowed_question_types: list[str] = Field(default_factory=list)
    used_question_types: dict[str, int] = Field(default_factory=dict)
    estimated_question_type_targets: dict[str, int] = Field(default_factory=dict)
    chunk_ids_by_available_question_type: dict[str, list[str]] = Field(default_factory=dict)
    sparse_question_types: list[str] = Field(default_factory=list)


JsonDict = dict[str, Any]
