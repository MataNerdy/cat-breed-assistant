from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cat_breed_assistant.evaluation.retrieval.generators import (
    EvaluationProviderError,
)
from src.cat_breed_assistant.evaluation.retrieval.pipeline import (
    candidates_by_breed,
    covered_breed_ids_from_candidates,
    dry_run,
    dry_run_unused_breeds,
    load_config,
    role_configs_for_index,
    run_pipeline,
    run_unused_breeds_pipeline,
)
from src.cat_breed_assistant.evaluation.retrieval.sampling import (
    deterministic_sample_chunks,
    is_informative_chunk,
    load_skipped_document_ids,
    load_source_chunks,
)
from src.cat_breed_assistant.evaluation.retrieval.schemas import (
    GeneratedQuestion,
    ModelConfig,
    PilotConfig,
    RetrievalEvaluationCandidate,
)
from src.cat_breed_assistant.evaluation.retrieval.validators import (
    locally_validate_candidate,
    normalize_query,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def chunk(
    breed_id: str,
    suffix: str,
    source: str = "wikipedia",
    document_id: str | None = None,
    text: str | None = None,
) -> dict:
    document_id = document_id or f"{breed_id}:{source}:en"
    return {
        "chunk_id": f"{document_id}:{suffix}:000",
        "document_id": document_id,
        "breed_id": breed_id,
        "breed_name_en": f"Breed {breed_id}",
        "breed_name_ru": None,
        "aliases": [f"Alias {breed_id}"],
        "language": "en",
        "source": source,
        "chunk_type": "structured_profile" if source == "thecatapi" else "section",
        "section_title": "Profile" if source == "thecatapi" else f"Section {suffix}",
        "section_path": ["Breed profile"] if source == "thecatapi" else [f"Section {suffix}"],
        "text": text
        or (
            f"Breed {breed_id} has a calm temperament and a plush coat. "
            "This source chunk includes enough concrete detail for retrieval "
            "evaluation and should be considered informative by the sampler."
        ),
        "text_length": 170,
        "provenance": {},
    }


def chunks_for_pipeline() -> list[dict]:
    records = []
    for breed_id in ["beng", "bsho", "mcoo", "sibe", "sphy", "pers"]:
        records.append(chunk(breed_id, "profile", source="thecatapi", document_id=f"{breed_id}:catapi:profile"))
        records.append(chunk(breed_id, "overview"))
        records.append(chunk(breed_id, "history"))
    return records


def chunks_for_unused_breeds() -> list[dict]:
    records = []
    for breed_id in ["abys", "bsho", "mcoo"]:
        for index, topic in enumerate(["origin", "temperament", "care"]):
            records.append(
                chunk(
                    breed_id,
                    topic,
                    source="wikipedia",
                    document_id=f"{breed_id}:wikipedia:en",
                    text=(
                        f"Fact {topic} for Breed {breed_id} gives a distinct "
                        f"{topic} detail for retrieval evaluation. This chunk "
                        "has enough length and concrete information to be used "
                        "as a standalone source for a question candidate."
                    ),
                )
            )
    return records


def candidate_record(
    breed_id: str,
    chunk_id: str,
    query: str | None = None,
    question_type: str = "origin",
    evidence_quote: str = "distinct origin detail",
) -> dict:
    query = query or f"What is one {question_type} fact for {breed_id}?"
    return {
        "query_id": f"q_{breed_id}_{question_type}_{chunk_id}",
        "query": query,
        "chunk_id": chunk_id,
        "document_id": f"{breed_id}:wikipedia:en",
        "breed_id": breed_id,
        "breed_name": f"Breed {breed_id}",
        "relevant_chunk_ids": [chunk_id],
        "answer": f"{question_type} answer",
        "evidence_quote": evidence_quote,
        "language": "en",
        "question_type": question_type,
        "difficulty": "easy",
        "breed_name_present": True,
        "generator_provider": "gemini",
        "generator_model": "gemini-test",
        "validator_provider": "mistral",
        "validator_model": "mistral-test",
        "generator_prompt_version": "g1",
        "validator_prompt_version": "v1",
        "validation": {
            "is_answerable": True,
            "answer_supported": True,
            "evidence_supported": True,
            "question_is_natural": True,
            "question_copies_source": False,
            "answer_leaked_in_question": False,
            "requires_external_knowledge": False,
            "is_ambiguous": False,
            "fact_is_distinct_from_existing_questions": True,
            "question_type_is_distinct": True,
            "approved": True,
            "rejection_reasons": [],
        },
        "status": "pending_review",
        "source_chunk_hash": "hash",
        "created_at": "2026-07-25T00:00:00Z",
        "run_id": "run",
    }


def config(tmp_path: Path, chunks_path: Path, skipped_path: Path, target_count: int = 4) -> PilotConfig:
    return PilotConfig(
        input_chunks_path=str(chunks_path),
        skipped_broader_sources_path=str(skipped_path),
        output_dir=str(tmp_path / "out"),
        seed=42,
        breed_limit=3,
        target_count=target_count,
        questions_per_chunk_min=1,
        questions_per_chunk_max=1,
        generator_a=ModelConfig(provider="gemini", model="gemini-test", temperature=0.3),
        validator_a=ModelConfig(provider="mistral", model="mistral-test", temperature=0.0),
        generator_b=ModelConfig(provider="mistral", model="mistral-test", temperature=0.3),
        validator_b=ModelConfig(provider="gemini", model="gemini-test", temperature=0.0),
    )


class FakeProvider:
    def __init__(self, model_config: ModelConfig, malformed: bool = False, reject: bool = False) -> None:
        self.provider = model_config.provider
        self.model = model_config.model
        self.malformed = malformed
        self.reject = reject

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        if self.malformed:
            return "{bad json"
        if "Validate one retrieval evaluation candidate" in user_prompt:
            return json.dumps(
                {
                    "is_answerable": not self.reject,
                    "answer_supported": not self.reject,
                    "evidence_supported": not self.reject,
                    "question_is_natural": True,
                    "question_copies_source": False,
                    "answer_leaked_in_question": False,
                    "requires_external_knowledge": False,
                    "is_ambiguous": False,
                    "approved": not self.reject,
                    "rejection_reasons": ["fake rejection"] if self.reject else [],
                }
            )
        payload = json.loads(user_prompt)
        text = payload["source_chunk"]["text"]
        quote = "calm temperament"
        if quote not in text:
            quote = text[: min(40, len(text))]
        return json.dumps(
            {
                "questions": [
                    {
                        "query": (
                            "What temperament is mentioned in "
                            f"{payload['source_chunk']['chunk_id']}?"
                        ),
                        "answer": "calm temperament",
                        "evidence_quote": quote,
                        "question_type": "fact_lookup",
                        "difficulty": "easy",
                        "breed_name_present": True,
                    }
                ]
            }
        )


class DistinctFakeProvider:
    def __init__(self, model_config: ModelConfig, reject_validation: bool = False) -> None:
        self.provider = model_config.provider
        self.model = model_config.model
        self.reject_validation = reject_validation

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        if payload["task"] == "Validate one retrieval evaluation candidate.":
            return json.dumps(
                {
                    "is_answerable": not self.reject_validation,
                    "answer_supported": not self.reject_validation,
                    "evidence_supported": not self.reject_validation,
                    "question_is_natural": True,
                    "question_copies_source": False,
                    "answer_leaked_in_question": False,
                    "requires_external_knowledge": False,
                    "is_ambiguous": False,
                    "fact_is_distinct_from_existing_questions": not self.reject_validation,
                    "question_type_is_distinct": not self.reject_validation,
                    "approved": not self.reject_validation,
                    "rejection_reasons": ["not distinct"] if self.reject_validation else [],
                }
            )
        source = payload["source_chunk"]
        chunk_id = source["chunk_id"]
        question_type = chunk_id.split(":")[-2]
        quote = source["text"].split(".")[0]
        return json.dumps(
            {
                "questions": [
                    {
                        "query": f"What {question_type} detail is stated for {source['breed_id']}?",
                        "answer": quote,
                        "evidence_quote": quote,
                        "question_type": question_type,
                        "difficulty": "easy",
                        "breed_name_present": True,
                    }
                ]
            }
        )


def fake_factory(model_config: ModelConfig):
    return FakeProvider(model_config)


def distinct_fake_factory(model_config: ModelConfig):
    return DistinctFakeProvider(model_config)


def test_deterministic_sampling(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    chunks = load_source_chunks(chunks_path)
    skipped = load_skipped_document_ids(skipped_path)

    first = deterministic_sample_chunks(chunks, skipped, seed=42, breed_limit=3, target_count=6)
    second = deterministic_sample_chunks(chunks, skipped, seed=42, breed_limit=3, target_count=6)

    assert [item.chunk.chunk_id for item in first] == [item.chunk.chunk_id for item in second]


def test_skipped_broader_documents_are_excluded(tmp_path: Path) -> None:
    skipped_doc = "beng:wikipedia:en"
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    rows = [chunk("beng", "profile", source="thecatapi"), chunk("beng", "overview", document_id=skipped_doc)]
    write_jsonl(chunks_path, rows)
    write_jsonl(skipped_path, [{"document_id": skipped_doc}])
    chunks = load_source_chunks(chunks_path)
    skipped = load_skipped_document_ids(skipped_path)

    assert not is_informative_chunk(chunks[1], skipped)


def test_schema_validation_rejects_non_pending_auto_candidate() -> None:
    with pytest.raises(ValueError, match="pending_review"):
        RetrievalEvaluationCandidate.model_validate(
            {
                "query_id": "q1",
                "query": "Question?",
                "chunk_id": "c1",
                "document_id": "d1",
                "breed_id": "b1",
                "breed_name": "Breed",
                "relevant_chunk_ids": ["c1"],
                "answer": "answer",
                "evidence_quote": "answer",
                "language": "en",
                "question_type": "fact",
                "difficulty": "easy",
                "breed_name_present": True,
                "generator_provider": "gemini",
                "generator_model": "g",
                "validator_provider": "mistral",
                "validator_model": "m",
                "generator_prompt_version": "g1",
                "validator_prompt_version": "v1",
                "validation": {
                    "is_answerable": True,
                    "answer_supported": True,
                    "evidence_supported": True,
                    "question_is_natural": True,
                    "question_copies_source": False,
                    "answer_leaked_in_question": False,
                    "requires_external_knowledge": False,
                    "is_ambiguous": False,
                    "approved": True,
                    "rejection_reasons": [],
                },
                "status": "approved",
                "source_chunk_hash": "hash",
                "created_at": "2026-07-25T00:00:00Z",
                "run_id": "run",
            }
        )


def test_evidence_quote_substring_validation() -> None:
    source = load_source_chunks_from_records([chunk("mcoo", "overview")])[0]
    generated = GeneratedQuestion(
        query="What is mentioned?",
        answer="unknown",
        evidence_quote="not in source",
        question_type="fact",
        difficulty="easy",
        breed_name_present=False,
    )

    assert locally_validate_candidate(generated, source, set()) == "evidence_quote_not_found"


def test_duplicate_query_rejection() -> None:
    source = load_source_chunks_from_records([chunk("mcoo", "overview")])[0]
    generated = GeneratedQuestion(
        query="What temperament is mentioned?",
        answer="calm temperament",
        evidence_quote="calm temperament",
        question_type="fact",
        difficulty="easy",
        breed_name_present=False,
    )
    seen = {normalize_query("What temperament is mentioned?")}

    assert locally_validate_candidate(generated, source, seen) == "duplicate_normalized_query"


def test_stable_provider_role_assignment(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    cfg = config(tmp_path, chunks_path, skipped_path)

    assert role_configs_for_index(cfg, 0)[0].provider == "gemini"
    assert role_configs_for_index(cfg, 1)[0].provider == "mistral"


def test_dry_run_does_not_create_outputs(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    cfg = config(tmp_path, chunks_path, skipped_path)

    result = dry_run(cfg)

    assert result.api_key_available == {"gemini": False, "mistral": False}
    assert not (tmp_path / "out" / "pilot_candidates.jsonl").exists()


def test_api_key_absence_blocks_real_run(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")

    with pytest.raises(EvaluationProviderError, match="Missing API keys"):
        run_pipeline(config(tmp_path, chunks_path, skipped_path), provider_factory=fake_factory)


def test_integration_with_fake_providers_creates_pending_candidates(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    cfg = config(tmp_path, chunks_path, skipped_path, target_count=3)

    manifest = run_pipeline(cfg, provider_factory=fake_factory)
    candidates = [json.loads(line) for line in (tmp_path / "out" / "pilot_candidates.jsonl").read_text().splitlines()]

    assert manifest.pending_review_count == 3
    assert all(candidate["status"] == "pending_review" for candidate in candidates)
    assert manifest.validator_approved_count == 3


def test_resume_skips_processed_chunk(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    cfg = config(tmp_path, chunks_path, skipped_path, target_count=2)

    run_pipeline(cfg, provider_factory=fake_factory)
    first = (tmp_path / "out" / "pilot_candidates.jsonl").read_text()
    run_pipeline(cfg, resume=True, provider_factory=fake_factory)
    second = (tmp_path / "out" / "pilot_candidates.jsonl").read_text()

    assert first == second


def test_malformed_llm_json_is_rejected(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    def factory(model_config: ModelConfig):
        return FakeProvider(model_config, malformed=True)

    manifest = run_pipeline(config(tmp_path, chunks_path, skipped_path, target_count=1), provider_factory=factory)
    rejections = [json.loads(line) for line in (tmp_path / "out" / "pilot_rejections.jsonl").read_text().splitlines()]

    assert manifest.pending_review_count == 0
    assert rejections
    assert rejections[0]["stage"] == "generator"


def test_validator_rejection_is_recorded(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    write_jsonl(chunks_path, chunks_for_pipeline())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    def factory(model_config: ModelConfig):
        return FakeProvider(model_config, reject=model_config.temperature == 0.0)

    manifest = run_pipeline(config(tmp_path, chunks_path, skipped_path, target_count=1), provider_factory=factory)
    rejections = [json.loads(line) for line in (tmp_path / "out" / "pilot_rejections.jsonl").read_text().splitlines()]

    assert manifest.pending_review_count == 0
    assert any(record["stage"] == "validator" for record in rejections)


def test_load_config_from_yaml() -> None:
    cfg = load_config(Path("configs/retrieval_eval_pilot.yaml"))

    assert cfg.seed == 42
    assert cfg.generator_a.provider == "gemini"
    assert cfg.validator_b.provider == "gemini"


def test_unused_breed_detection_uses_candidates_only(tmp_path: Path) -> None:
    candidates = [
        RetrievalEvaluationCandidate.model_validate(
            candidate_record("bsho", "bsho:wikipedia:en:origin:000")
        )
    ]

    assert covered_breed_ids_from_candidates(candidates) == {"bsho"}
    assert set(candidates_by_breed(candidates)) == {"bsho"}


def test_rejected_only_breed_does_not_count_as_covered(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, chunks_for_unused_breeds())
    write_jsonl(skipped_path, [])
    write_jsonl(tmp_path / "out" / "pilot_candidates.jsonl", [])
    write_jsonl(
        tmp_path / "out" / "pilot_rejections.jsonl",
        [
            {
                "chunk_id": "bsho:wikipedia:en:origin:000",
                "stage": "validator",
                "reason": "rejected",
                "raw_output": None,
                "run_id": "run",
            }
        ],
    )
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")

    result = dry_run_unused_breeds(cfg, questions_per_breed=3)

    assert "bsho" in result.unused_breed_ids
    assert "bsho" not in result.already_covered_breed_ids


def test_unused_breeds_resume_tops_up_partially_covered_breed(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, chunks_for_unused_breeds())
    write_jsonl(skipped_path, [])
    write_jsonl(
        tmp_path / "out" / "pilot_candidates.jsonl",
        [
            candidate_record("bsho", "bsho:wikipedia:en:origin:000", question_type="origin"),
            candidate_record(
                "bsho",
                "bsho:wikipedia:en:temperament:000",
                question_type="temperament",
                evidence_quote="distinct temperament detail",
            ),
        ],
    )
    write_jsonl(tmp_path / "out" / "pilot_rejections.jsonl", [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    manifest = run_unused_breeds_pipeline(
        cfg,
        questions_per_breed=3,
        resume=True,
        provider_factory=distinct_fake_factory,
    )
    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "pilot_candidates.jsonl").read_text().splitlines()
    ]

    bsho = [candidate for candidate in candidates if candidate["breed_id"] == "bsho"]
    assert len(bsho) == 3
    assert manifest.coverage_by_breed["bsho"]["existing_count"] == 2
    assert manifest.coverage_by_breed["bsho"]["generated_count"] == 1


def test_unused_breeds_round_robin_order(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, chunks_for_unused_breeds())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    run_unused_breeds_pipeline(
        cfg,
        questions_per_breed=2,
        provider_factory=distinct_fake_factory,
    )
    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "pilot_candidates.jsonl").read_text().splitlines()
    ]

    assert [candidate["breed_id"] for candidate in candidates] == [
        "abys",
        "bsho",
        "mcoo",
        "abys",
        "bsho",
        "mcoo",
    ]


def test_unused_breeds_caps_candidates_per_breed_at_three(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, chunks_for_unused_breeds())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    run_unused_breeds_pipeline(
        cfg,
        questions_per_breed=3,
        provider_factory=distinct_fake_factory,
    )
    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "pilot_candidates.jsonl").read_text().splitlines()
    ]

    assert max(
        sum(1 for candidate in candidates if candidate["breed_id"] == breed_id)
        for breed_id in {"abys", "bsho", "mcoo"}
    ) == 3


def test_local_validation_rejects_same_question_type_and_evidence_for_breed() -> None:
    source = load_source_chunks_from_records([chunk("mcoo", "origin", text="A distinct origin fact. More text for context.")])[0]
    generated = GeneratedQuestion(
        query="What origin detail is stated for mcoo?",
        answer="A distinct origin fact",
        evidence_quote="A distinct origin fact",
        question_type="origin",
        difficulty="easy",
        breed_name_present=True,
    )
    existing = [
        RetrievalEvaluationCandidate.model_validate(
            candidate_record(
                "mcoo",
                "mcoo:wikipedia:en:origin:000",
                query="What origin fact is stated for mcoo?",
                question_type="origin",
                evidence_quote="A distinct origin fact",
            )
        )
    ]

    assert (
        locally_validate_candidate(generated, source, set(), existing_for_breed=existing)
        == "duplicate_question_type_for_breed"
    )


def test_unused_breeds_shortfall_is_reported(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, [chunks_for_unused_breeds()[0]])
    write_jsonl(skipped_path, [])

    result = dry_run_unused_breeds(cfg, questions_per_breed=3)

    assert result.expected_max_candidates == 1
    assert "abys" in result.shortfalls_by_breed


def test_unused_breeds_dry_run_separates_unused_topups_and_shortfalls(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    records = [
        chunks_for_unused_breeds()[0],
        chunks_for_unused_breeds()[1],
        chunks_for_unused_breeds()[2],
        chunk(
            "bsho",
            "origin",
            source="wikipedia",
            document_id="bsho:wikipedia:en",
            text=(
                "Fact origin for Breed bsho gives a distinct origin detail. "
                "This chunk has enough length and concrete information to be "
                "used as a standalone source for a question candidate."
            ),
        ),
        chunk(
            "mcoo",
            "origin",
            source="wikipedia",
            document_id="mcoo:wikipedia:en",
            text=(
                "Fact origin for Breed mcoo gives a distinct origin detail. "
                "This chunk has enough length and concrete information to be "
                "used as a standalone source for a question candidate."
            ),
        ),
    ]
    write_jsonl(chunks_path, records)
    write_jsonl(skipped_path, [])
    write_jsonl(
        tmp_path / "out" / "pilot_candidates.jsonl",
        [
            candidate_record("bsho", "bsho:wikipedia:en:existing:000", question_type="existing"),
        ],
    )
    write_jsonl(tmp_path / "out" / "pilot_rejections.jsonl", [])

    result = dry_run_unused_breeds(cfg, questions_per_breed=3, resume=True)

    assert result.unused_breed_ids == ["abys", "mcoo"]
    assert result.partially_covered_breed_ids == ["bsho"]
    assert result.requested_candidate_slots == 8
    assert result.expected_candidates_from_unused_breeds == 4
    assert result.expected_top_up_candidates_for_partially_covered_breeds == 1
    assert result.unavailable_candidate_slots == 3
    assert result.expected_max_candidates == 5


def test_unused_breeds_deterministic_order(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, list(reversed(chunks_for_unused_breeds())))
    write_jsonl(skipped_path, [])

    first = dry_run_unused_breeds(cfg, questions_per_breed=2)
    second = dry_run_unused_breeds(cfg, questions_per_breed=2)

    assert first.unused_breed_ids == second.unused_breed_ids == ["abys", "bsho", "mcoo"]
    assert first.selected_chunk_ids_by_breed == second.selected_chunk_ids_by_breed


def test_unused_breeds_manifest_stats(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, chunks_for_unused_breeds())
    write_jsonl(skipped_path, [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    manifest = run_unused_breeds_pipeline(
        cfg,
        questions_per_breed=1,
        provider_factory=distinct_fake_factory,
    )

    assert manifest.total_available_breeds == 3
    assert manifest.previously_covered_breeds == 0
    assert manifest.target_unused_breeds == 3
    assert manifest.newly_covered_breeds == 3
    assert manifest.fully_covered_breeds == 3
    assert manifest.coverage_by_breed["abys"]["final_candidate_count"] == 1


def test_unused_breeds_skips_fully_covered_breed(tmp_path: Path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    cfg = config(tmp_path, chunks_path, skipped_path)
    write_jsonl(chunks_path, chunks_for_unused_breeds())
    write_jsonl(skipped_path, [])
    write_jsonl(
        tmp_path / "out" / "pilot_candidates.jsonl",
        [
            candidate_record("abys", "abys:wikipedia:en:origin:000", question_type="origin"),
            candidate_record(
                "abys",
                "abys:wikipedia:en:temperament:000",
                question_type="temperament",
                evidence_quote="distinct temperament detail",
            ),
            candidate_record(
                "abys",
                "abys:wikipedia:en:care:000",
                question_type="care",
                evidence_quote="distinct care detail",
            ),
        ],
    )
    write_jsonl(tmp_path / "out" / "pilot_rejections.jsonl", [])
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake")

    run_unused_breeds_pipeline(
        cfg,
        questions_per_breed=3,
        resume=True,
        provider_factory=distinct_fake_factory,
    )
    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "pilot_candidates.jsonl").read_text().splitlines()
    ]

    assert sum(1 for candidate in candidates if candidate["breed_id"] == "abys") == 3


def load_source_chunks_from_records(records: list[dict]):
    path = Path("/tmp/retrieval_eval_test_chunks.jsonl")
    write_jsonl(path, records)
    return load_source_chunks(path)
