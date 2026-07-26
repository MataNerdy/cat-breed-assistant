from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import yaml
from pydantic import ValidationError

from src.cat_breed_assistant.evaluation.retrieval.generators import (
    EvaluationProvider,
    EvaluationProviderError,
    api_key_availability,
    make_provider,
    provider_system_prompt,
    require_api_keys,
)
from src.cat_breed_assistant.evaluation.retrieval.io import (
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from src.cat_breed_assistant.evaluation.retrieval.prompts import (
    generator_user_prompt,
    validator_user_prompt,
)
from src.cat_breed_assistant.evaluation.retrieval.sampling import (
    choose_chunks_for_breed,
    deterministic_sample_chunks,
    is_informative_chunk,
    load_skipped_document_ids,
    load_source_chunks,
    selected_breed_ids,
    selected_chunk_ids,
)
from src.cat_breed_assistant.evaluation.retrieval.schemas import (
    DryRunResult,
    GeneratedQuestion,
    GeneratedQuestionBatch,
    ModelConfig,
    PilotConfig,
    RejectionRecord,
    RetrievalEvaluationCandidate,
    RunManifest,
    SelectedChunk,
    SourceChunk,
    UnusedBreedDryRunResult,
    ValidationResult,
)
from src.cat_breed_assistant.evaluation.retrieval.validators import (
    locally_validate_candidate,
    normalize_query,
    validator_accepts,
)


ProviderFactory = Callable[[ModelConfig], EvaluationProvider]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> PilotConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return PilotConfig.model_validate(data)


def apply_cli_overrides(
    config: PilotConfig,
    seed: int | None = None,
    target_count: int | None = None,
    breed_limit: int | None = None,
) -> PilotConfig:
    updates = {}
    if seed is not None:
        updates["seed"] = seed
    if target_count is not None:
        updates["target_count"] = target_count
    if breed_limit is not None:
        updates["breed_limit"] = breed_limit
    return config.model_copy(update=updates)


def role_configs_for_index(config: PilotConfig, index: int) -> tuple[ModelConfig, ModelConfig]:
    if index % 2 == 0:
        return config.generator_a, config.validator_a
    return config.generator_b, config.validator_b


def output_paths(config: PilotConfig) -> dict[str, Path]:
    output_dir = Path(config.output_dir)
    return {
        "candidates": output_dir / "pilot_candidates.jsonl",
        "rejections": output_dir / "pilot_rejections.jsonl",
        "manifest": output_dir / "pilot_run_manifest.json",
    }


def run_id_for(config: PilotConfig, input_sha: str) -> str:
    payload = {
        "seed": config.seed,
        "breed_limit": config.breed_limit,
        "target_count": config.target_count,
        "input_sha": input_sha,
    }
    return f"retrieval-eval-pilot-{stable_hash(payload)[:12]}"


def run_id_for_unused_breeds(
    config: PilotConfig,
    input_sha: str,
    questions_per_breed: int,
) -> str:
    payload = {
        "seed": config.seed,
        "questions_per_breed": questions_per_breed,
        "mode": "unused_breeds",
        "input_sha": input_sha,
    }
    return f"retrieval-eval-unused-breeds-{stable_hash(payload)[:12]}"


def load_existing_state(
    candidates_path: Path,
    rejections_path: Path,
) -> tuple[list[RetrievalEvaluationCandidate], list[RejectionRecord], set[str], set[str]]:
    candidates = [
        RetrievalEvaluationCandidate.model_validate(record)
        for record in read_jsonl(candidates_path)
    ]
    rejections = [RejectionRecord.model_validate(record) for record in read_jsonl(rejections_path)]
    processed_chunks = {candidate.chunk_id for candidate in candidates}
    processed_chunks.update(record.chunk_id for record in rejections)
    seen_queries = {normalize_query(candidate.query) for candidate in candidates}
    return candidates, rejections, processed_chunks, seen_queries


def candidates_by_breed(
    candidates: list[RetrievalEvaluationCandidate],
) -> dict[str, list[RetrievalEvaluationCandidate]]:
    grouped: dict[str, list[RetrievalEvaluationCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.breed_id].append(candidate)
    return dict(grouped)


def covered_breed_ids_from_candidates(
    candidates: list[RetrievalEvaluationCandidate],
) -> set[str]:
    return {candidate.breed_id for candidate in candidates}


def informative_chunks_by_breed(
    chunks: list[SourceChunk],
    skipped_document_ids: set[str],
) -> dict[str, list[SourceChunk]]:
    grouped: dict[str, list[SourceChunk]] = defaultdict(list)
    for chunk in chunks:
        if is_informative_chunk(chunk, skipped_document_ids):
            grouped[chunk.breed_id].append(chunk)
    return {
        breed_id: choose_chunks_for_breed(rows, max_chunks=max(3, len(rows)))
        for breed_id, rows in sorted(grouped.items())
        if rows
    }


def available_breed_ids(chunks_by_breed: dict[str, list[SourceChunk]]) -> list[str]:
    return sorted(chunks_by_breed)


def chunk_counts_by_source(chunks_by_breed: dict[str, list[SourceChunk]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for chunks in chunks_by_breed.values():
        for chunk in chunks:
            counts[chunk.source] += 1
    return dict(sorted(counts.items()))


def used_chunk_ids_for_breed(
    candidates: list[RetrievalEvaluationCandidate],
    breed_id: str,
) -> set[str]:
    return {candidate.chunk_id for candidate in candidates if candidate.breed_id == breed_id}


def create_manifest(
    config: PilotConfig,
    run_id: str,
    input_sha: str,
    selected: list[SelectedChunk],
    started_at: str,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        seed=config.seed,
        started_at=started_at,
        input_chunks_path=config.input_chunks_path,
        input_chunks_sha256=input_sha,
        selected_breed_ids=selected_breed_ids(selected),
        selected_chunk_ids=selected_chunk_ids(selected),
        generator_models={
            config.generator_a.provider: config.generator_a.model,
            config.generator_b.provider: config.generator_b.model,
        },
        validator_models={
            config.validator_a.provider: config.validator_a.model,
            config.validator_b.provider: config.validator_b.model,
        },
        prompt_versions={
            "generator": config.generator_prompt_version,
            "validator": config.validator_prompt_version,
        },
    )


def coverage_stats(
    all_breed_ids: list[str],
    target_breed_ids: list[str],
    existing_by_breed: dict[str, list[RetrievalEvaluationCandidate]],
    candidates: list[RetrievalEvaluationCandidate],
    new_candidates_by_breed: dict[str, int],
    rejected_by_breed: dict[str, int],
    questions_per_breed: int,
) -> dict[str, object]:
    current_by_breed = candidates_by_breed(candidates)
    previously_covered = {
        breed_id for breed_id, rows in existing_by_breed.items() if rows
    }
    newly_covered = {
        breed_id
        for breed_id in target_breed_ids
        if not existing_by_breed.get(breed_id) and current_by_breed.get(breed_id)
    }
    fully_covered = {
        breed_id
        for breed_id, rows in current_by_breed.items()
        if len(rows) >= questions_per_breed
    }
    partially_covered = {
        breed_id
        for breed_id, rows in current_by_breed.items()
        if 0 < len(rows) < questions_per_breed
    }
    coverage_by_breed = {}
    for breed_id in all_breed_ids:
        existing_count = len(existing_by_breed.get(breed_id, []))
        current_rows = current_by_breed.get(breed_id, [])
        coverage_by_breed[breed_id] = {
            "requested_count": questions_per_breed,
            "existing_count": existing_count,
            "generated_count": new_candidates_by_breed.get(breed_id, 0),
            "rejected_count": rejected_by_breed.get(breed_id, 0),
            "final_candidate_count": len(current_rows),
            "question_types": sorted({candidate.question_type for candidate in current_rows}),
            "chunk_ids": sorted({candidate.chunk_id for candidate in current_rows}),
        }
    return {
        "total_available_breeds": len(all_breed_ids),
        "previously_covered_breeds": len(previously_covered),
        "target_unused_breeds": len(
            [breed_id for breed_id in target_breed_ids if not existing_by_breed.get(breed_id)]
        ),
        "newly_covered_breeds": len(newly_covered),
        "fully_covered_breeds": len(fully_covered),
        "partially_covered_breeds": len(partially_covered),
        "questions_per_breed": questions_per_breed,
        "coverage_by_breed": coverage_by_breed,
    }


def update_manifest_counts(
    manifest: RunManifest,
    candidates: list[RetrievalEvaluationCandidate],
    rejections: list[RejectionRecord],
) -> RunManifest:
    return manifest.model_copy(
        update={
            "generated_count": len(candidates) + len(rejections),
            "validator_approved_count": len(candidates),
            "pending_review_count": len(candidates),
            "rejected_count": len(rejections),
            "finished_at": utc_now(),
        }
    )


def dry_run(config: PilotConfig) -> DryRunResult:
    input_path = Path(config.input_chunks_path)
    chunks = load_source_chunks(input_path)
    skipped = load_skipped_document_ids(Path(config.skipped_broader_sources_path))
    selected = deterministic_sample_chunks(
        chunks,
        skipped,
        seed=config.seed,
        breed_limit=config.breed_limit,
        target_count=config.target_count,
    )
    return DryRunResult(
        selected_breed_ids=selected_breed_ids(selected),
        selected_chunk_ids=selected_chunk_ids(selected),
        api_key_available=api_key_availability(),
        input_chunks_sha256=sha256_file(input_path),
    )


def dry_run_unused_breeds(
    config: PilotConfig,
    questions_per_breed: int,
    resume: bool = False,
) -> UnusedBreedDryRunResult:
    input_path = Path(config.input_chunks_path)
    chunks = load_source_chunks(input_path)
    skipped = load_skipped_document_ids(Path(config.skipped_broader_sources_path))
    chunks_by_breed = informative_chunks_by_breed(chunks, skipped)
    all_breeds = available_breed_ids(chunks_by_breed)
    paths = output_paths(config)
    existing_candidates, _, _, _ = load_existing_state(paths["candidates"], paths["rejections"])
    existing_by_breed = candidates_by_breed(existing_candidates)
    already_covered = sorted(covered_breed_ids_from_candidates(existing_candidates))
    partially_covered = sorted(
        breed_id
        for breed_id, rows in existing_by_breed.items()
        if 0 < len(rows) < questions_per_breed
    )
    target_breeds = [
        breed_id
        for breed_id in all_breeds
        if not existing_by_breed.get(breed_id)
        or (resume and len(existing_by_breed.get(breed_id, [])) < questions_per_breed)
    ]
    selected_by_breed: dict[str, list[str]] = {}
    shortfalls: dict[str, str] = {}
    requested_candidate_slots = 0
    expected_candidates_from_unused_breeds = 0
    expected_top_up_candidates = 0
    unavailable_candidate_slots = 0
    expected_max_candidates = 0
    for breed_id in target_breeds:
        used_chunks = used_chunk_ids_for_breed(existing_candidates, breed_id)
        available_chunks = [
            chunk for chunk in chunks_by_breed.get(breed_id, []) if chunk.chunk_id not in used_chunks
        ]
        missing = max(questions_per_breed - len(existing_by_breed.get(breed_id, [])), 0)
        available_for_breed = min(missing, len(available_chunks))
        shortfall_for_breed = max(missing - len(available_chunks), 0)
        selected_by_breed[breed_id] = [
            chunk.chunk_id for chunk in available_chunks[:missing]
        ]
        requested_candidate_slots += missing
        expected_max_candidates += available_for_breed
        unavailable_candidate_slots += shortfall_for_breed
        if existing_by_breed.get(breed_id):
            expected_top_up_candidates += available_for_breed
        else:
            expected_candidates_from_unused_breeds += available_for_breed
        if shortfall_for_breed:
            shortfalls[breed_id] = (
                f"Need {missing} more candidates, but only {len(available_chunks)} "
                "unused informative chunks are available."
            )
    return UnusedBreedDryRunResult(
        already_covered_breed_ids=already_covered,
        unused_breed_ids=[breed_id for breed_id in target_breeds if not existing_by_breed.get(breed_id)],
        partially_covered_breed_ids=partially_covered,
        target_breed_ids=target_breeds,
        selected_chunk_ids_by_breed=selected_by_breed,
        chunk_counts_by_source=chunk_counts_by_source(chunks_by_breed),
        shortfalls_by_breed=shortfalls,
        requested_candidate_slots=requested_candidate_slots,
        expected_candidates_from_unused_breeds=expected_candidates_from_unused_breeds,
        expected_top_up_candidates_for_partially_covered_breeds=expected_top_up_candidates,
        unavailable_candidate_slots=unavailable_candidate_slots,
        expected_max_candidates=expected_max_candidates,
        approximate_generator_calls=expected_max_candidates,
        approximate_validator_calls=expected_max_candidates,
        api_key_available=api_key_availability(),
        input_chunks_sha256=sha256_file(input_path),
    )


def parse_generated_batch(raw_output: str) -> GeneratedQuestionBatch:
    return GeneratedQuestionBatch.model_validate_json(raw_output)


def parse_validation_result(raw_output: str) -> ValidationResult:
    return ValidationResult.model_validate_json(raw_output)


def rejection(
    chunk_id: str,
    stage: str,
    reason: str,
    raw_output: str | None,
    run_id: str,
) -> RejectionRecord:
    return RejectionRecord.model_validate(
        {
            "chunk_id": chunk_id,
            "stage": stage,
            "reason": reason,
            "raw_output": raw_output,
            "run_id": run_id,
        }
    )


def query_id_for(run_id: str, chunk: SourceChunk, query: str) -> str:
    return f"q_{stable_hash({'run_id': run_id, 'chunk_id': chunk.chunk_id, 'query': normalize_query(query)})[:16]}"


def build_candidate(
    run_id: str,
    chunk: SourceChunk,
    generated,
    validation: ValidationResult,
    generator_config: ModelConfig,
    validator_config: ModelConfig,
    config: PilotConfig,
) -> RetrievalEvaluationCandidate:
    return RetrievalEvaluationCandidate(
        query_id=query_id_for(run_id, chunk, generated.query),
        query=generated.query,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        breed_id=chunk.breed_id,
        breed_name=chunk.breed_name,
        relevant_chunk_ids=[chunk.chunk_id],
        answer=generated.answer,
        evidence_quote=generated.evidence_quote,
        language=chunk.language,
        question_type=generated.question_type,
        difficulty=generated.difficulty,
        breed_name_present=generated.breed_name_present,
        generator_provider=generator_config.provider,
        generator_model=generator_config.model,
        validator_provider=validator_config.provider,
        validator_model=validator_config.model,
        generator_prompt_version=config.generator_prompt_version,
        validator_prompt_version=config.validator_prompt_version,
        validation=validation,
        status="pending_review",
        source_chunk_hash=stable_hash({"chunk_id": chunk.chunk_id, "text": chunk.text}),
        created_at=utc_now(),
        run_id=run_id,
    )


def make_manifest_with_unused_coverage(
    manifest: RunManifest,
    all_breed_ids: list[str],
    target_breed_ids: list[str],
    existing_by_breed: dict[str, list[RetrievalEvaluationCandidate]],
    candidates: list[RetrievalEvaluationCandidate],
    rejections: list[RejectionRecord],
    new_candidates_by_breed: dict[str, int],
    rejected_by_breed: dict[str, int],
    questions_per_breed: int,
) -> RunManifest:
    stats = coverage_stats(
        all_breed_ids,
        target_breed_ids,
        existing_by_breed,
        candidates,
        new_candidates_by_breed,
        rejected_by_breed,
        questions_per_breed,
    )
    return update_manifest_counts(manifest, candidates, rejections).model_copy(update=stats)


def append_rejection(
    rejections: list[RejectionRecord],
    rejected_by_breed: dict[str, int],
    chunk: SourceChunk,
    stage: str,
    reason: str,
    raw_output: str | None,
    run_id: str,
) -> None:
    rejections.append(rejection(chunk.chunk_id, stage, reason, raw_output, run_id))
    rejected_by_breed[chunk.breed_id] += 1


def run_unused_breeds_pipeline(
    config: PilotConfig,
    questions_per_breed: int = 3,
    resume: bool = False,
    provider_factory: ProviderFactory = make_provider,
) -> RunManifest:
    if questions_per_breed < 1:
        raise ValueError("questions_per_breed must be greater than zero")
    configs = [
        config.generator_a,
        config.validator_a,
        config.generator_b,
        config.validator_b,
    ]
    require_api_keys(configs)

    input_path = Path(config.input_chunks_path)
    chunks = load_source_chunks(input_path)
    skipped = load_skipped_document_ids(Path(config.skipped_broader_sources_path))
    chunks_by_breed = informative_chunks_by_breed(chunks, skipped)
    all_breeds = available_breed_ids(chunks_by_breed)
    input_sha = sha256_file(input_path)
    run_id = run_id_for_unused_breeds(config, input_sha, questions_per_breed)
    paths = output_paths(config)
    started_at = utc_now()

    candidates, rejections, _, seen_queries = load_existing_state(
        paths["candidates"],
        paths["rejections"],
    )
    existing_by_breed = candidates_by_breed(list(candidates))
    target_breeds = [
        breed_id
        for breed_id in all_breeds
        if not existing_by_breed.get(breed_id)
        or (resume and len(existing_by_breed.get(breed_id, [])) < questions_per_breed)
    ]
    selected = [
        SelectedChunk(chunk=chunk, selection_reason="unused breed round-robin source chunk", selection_index=index)
        for index, breed_id in enumerate(target_breeds)
        for chunk in chunks_by_breed.get(breed_id, [])[:questions_per_breed]
    ]
    manifest = create_manifest(config, run_id, input_sha, selected, started_at)
    providers: dict[tuple[str, str], EvaluationProvider] = {}
    new_candidates_by_breed: dict[str, int] = defaultdict(int)
    rejected_by_breed: dict[str, int] = defaultdict(int)
    next_chunk_index_by_breed: dict[str, int] = defaultdict(int)

    def provider_for(model_config: ModelConfig) -> EvaluationProvider:
        key = (model_config.provider, model_config.model)
        if key not in providers:
            providers[key] = provider_factory(model_config)
        return providers[key]

    def current_candidates_for_breed(breed_id: str) -> list[RetrievalEvaluationCandidate]:
        return [candidate for candidate in candidates if candidate.breed_id == breed_id]

    attempt_index = 0
    for round_index in range(questions_per_breed):
        for breed_id in target_breeds:
            if len(current_candidates_for_breed(breed_id)) >= questions_per_breed:
                continue
            chunks_for_breed = chunks_by_breed.get(breed_id, [])
            used_chunks = used_chunk_ids_for_breed(candidates, breed_id)
            chunk: SourceChunk | None = None
            while next_chunk_index_by_breed[breed_id] < len(chunks_for_breed):
                candidate_chunk = chunks_for_breed[next_chunk_index_by_breed[breed_id]]
                next_chunk_index_by_breed[breed_id] += 1
                if candidate_chunk.chunk_id not in used_chunks:
                    chunk = candidate_chunk
                    break
            if chunk is None:
                continue

            generator_config, validator_config = role_configs_for_index(config, attempt_index)
            attempt_index += 1
            existing_questions = current_candidates_for_breed(breed_id)
            raw_generated: str | None = None
            try:
                generator = provider_for(generator_config)
                raw_generated = generator.complete_json(
                    provider_system_prompt("generator"),
                    generator_user_prompt(
                        chunk,
                        min_questions=1,
                        max_questions=1,
                        existing_questions=existing_questions,
                    ),
                )
                generated_batch = parse_generated_batch(raw_generated)
            except (EvaluationProviderError, ValidationError, ValueError) as exc:
                append_rejection(
                    rejections,
                    rejected_by_breed,
                    chunk,
                    "generator",
                    str(exc),
                    raw_generated,
                    run_id,
                )
                persist(paths, candidates, rejections, manifest)
                continue

            accepted_for_breed = False
            for generated in generated_batch.questions:
                existing_questions = current_candidates_for_breed(breed_id)
                reason = locally_validate_candidate(
                    generated,
                    chunk,
                    seen_queries,
                    existing_for_breed=existing_questions,
                )
                if reason:
                    append_rejection(
                        rejections,
                        rejected_by_breed,
                        chunk,
                        "local_validation",
                        reason,
                        generated.model_dump_json(),
                        run_id,
                    )
                    continue
                raw_validation: str | None = None
                try:
                    validator = provider_for(validator_config)
                    raw_validation = validator.complete_json(
                        provider_system_prompt("validator"),
                        validator_user_prompt(
                            chunk,
                            generated,
                            existing_questions=existing_questions,
                        ),
                    )
                    validation = parse_validation_result(raw_validation)
                except (EvaluationProviderError, ValidationError, ValueError) as exc:
                    append_rejection(
                        rejections,
                        rejected_by_breed,
                        chunk,
                        "validator",
                        str(exc),
                        raw_validation,
                        run_id,
                    )
                    continue

                if not validator_accepts(validation):
                    append_rejection(
                        rejections,
                        rejected_by_breed,
                        chunk,
                        "validator",
                        "; ".join(validation.rejection_reasons) or "validator_rejected",
                        raw_validation,
                        run_id,
                    )
                    continue

                candidates.append(
                    build_candidate(
                        run_id,
                        chunk,
                        generated,
                        validation,
                        generator_config,
                        validator_config,
                        config,
                    )
                )
                seen_queries.add(normalize_query(generated.query))
                new_candidates_by_breed[breed_id] += 1
                accepted_for_breed = True
                break
            if accepted_for_breed:
                manifest = make_manifest_with_unused_coverage(
                    manifest,
                    all_breeds,
                    target_breeds,
                    existing_by_breed,
                    candidates,
                    rejections,
                    new_candidates_by_breed,
                    rejected_by_breed,
                    questions_per_breed,
                )
                persist(paths, candidates, rejections, manifest)
            else:
                manifest = make_manifest_with_unused_coverage(
                    manifest,
                    all_breeds,
                    target_breeds,
                    existing_by_breed,
                    candidates,
                    rejections,
                    new_candidates_by_breed,
                    rejected_by_breed,
                    questions_per_breed,
                )
                persist(paths, candidates, rejections, manifest)

    final_manifest = make_manifest_with_unused_coverage(
        manifest,
        all_breeds,
        target_breeds,
        existing_by_breed,
        candidates,
        rejections,
        new_candidates_by_breed,
        rejected_by_breed,
        questions_per_breed,
    )
    persist(paths, candidates, rejections, final_manifest)
    return final_manifest


def run_pipeline(
    config: PilotConfig,
    resume: bool = False,
    provider_factory: ProviderFactory = make_provider,
) -> RunManifest:
    configs = [
        config.generator_a,
        config.validator_a,
        config.generator_b,
        config.validator_b,
    ]
    require_api_keys(configs)

    input_path = Path(config.input_chunks_path)
    chunks = load_source_chunks(input_path)
    skipped = load_skipped_document_ids(Path(config.skipped_broader_sources_path))
    selected = deterministic_sample_chunks(
        chunks,
        skipped,
        seed=config.seed,
        breed_limit=config.breed_limit,
        target_count=config.target_count,
    )
    input_sha = sha256_file(input_path)
    run_id = run_id_for(config, input_sha)
    paths = output_paths(config)
    started_at = utc_now()

    if resume:
        candidates, rejections, processed_chunks, seen_queries = load_existing_state(
            paths["candidates"],
            paths["rejections"],
        )
    else:
        candidates, rejections, processed_chunks, seen_queries = [], [], set(), set()

    manifest = create_manifest(config, run_id, input_sha, selected, started_at)
    providers: dict[tuple[str, str], EvaluationProvider] = {}

    def provider_for(model_config: ModelConfig) -> EvaluationProvider:
        key = (model_config.provider, model_config.model)
        if key not in providers:
            providers[key] = provider_factory(model_config)
        return providers[key]

    for selected_item in selected:
        chunk = selected_item.chunk
        if chunk.chunk_id in processed_chunks:
            continue
        generator_config, validator_config = role_configs_for_index(
            config,
            selected_item.selection_index,
        )
        raw_generated: str | None = None
        try:
            generator = provider_for(generator_config)
            raw_generated = generator.complete_json(
                provider_system_prompt("generator"),
                generator_user_prompt(
                    chunk,
                    config.questions_per_chunk_min,
                    config.questions_per_chunk_max,
                ),
            )
            generated_batch = parse_generated_batch(raw_generated)
        except (EvaluationProviderError, ValidationError, ValueError) as exc:
            rejections.append(
                rejection(chunk.chunk_id, "generator", str(exc), raw_generated, run_id)
            )
            processed_chunks.add(chunk.chunk_id)
            persist(paths, candidates, rejections, manifest)
            continue

        accepted_for_chunk = 0
        for generated in generated_batch.questions:
            reason = locally_validate_candidate(generated, chunk, seen_queries)
            if reason:
                rejections.append(
                    rejection(
                        chunk.chunk_id,
                        "local_validation",
                        reason,
                        generated.model_dump_json(),
                        run_id,
                    )
                )
                continue
            raw_validation: str | None = None
            try:
                validator = provider_for(validator_config)
                raw_validation = validator.complete_json(
                    provider_system_prompt("validator"),
                    validator_user_prompt(chunk, generated),
                )
                validation = parse_validation_result(raw_validation)
            except (EvaluationProviderError, ValidationError, ValueError) as exc:
                rejections.append(
                    rejection(chunk.chunk_id, "validator", str(exc), raw_validation, run_id)
                )
                continue

            if not validator_accepts(validation):
                rejections.append(
                    rejection(
                        chunk.chunk_id,
                        "validator",
                        "; ".join(validation.rejection_reasons) or "validator_rejected",
                        raw_validation,
                        run_id,
                    )
                )
                continue

            candidates.append(
                build_candidate(
                    run_id,
                    chunk,
                    generated,
                    validation,
                    generator_config,
                    validator_config,
                    config,
                )
            )
            seen_queries.add(normalize_query(generated.query))
            accepted_for_chunk += 1
            if len(candidates) >= config.target_count:
                break
        processed_chunks.add(chunk.chunk_id)
        persist(paths, candidates, rejections, manifest)
        if len(candidates) >= config.target_count:
            break

    final_manifest = update_manifest_counts(manifest, candidates, rejections)
    persist(paths, candidates, rejections, final_manifest)
    return final_manifest


def persist(
    paths: dict[str, Path],
    candidates: list[RetrievalEvaluationCandidate],
    rejections: list[RejectionRecord],
    manifest: RunManifest,
) -> None:
    write_jsonl_atomic([item.model_dump(mode="json") for item in candidates], paths["candidates"])
    write_jsonl_atomic([item.model_dump(mode="json") for item in rejections], paths["rejections"])
    write_json_atomic(manifest.model_dump(mode="json"), paths["manifest"])
