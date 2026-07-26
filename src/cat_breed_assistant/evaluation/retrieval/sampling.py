from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.cat_breed_assistant.evaluation.retrieval.io import read_jsonl
from src.cat_breed_assistant.evaluation.retrieval.schemas import SelectedChunk, SourceChunk
from src.data.source_scope import is_service_section_path


MIN_INFORMATIVE_CHARS = 120
SKIPPED_SECTION_TITLES = {
    "gallery",
    "references",
    "reference",
    "external links",
    "see also",
    "images",
    "фотографии",
    "галерея",
    "примечания",
    "источники",
    "ссылки",
}


def load_source_chunks(path: Path) -> list[SourceChunk]:
    chunks = []
    for record in read_jsonl(path):
        chunks.append(SourceChunk.model_validate(record))
    return chunks


def load_skipped_document_ids(path: Path) -> set[str]:
    return {
        str(record["document_id"])
        for record in read_jsonl(path)
        if isinstance(record.get("document_id"), str)
    }


def is_informative_chunk(chunk: SourceChunk, skipped_document_ids: set[str]) -> bool:
    if chunk.document_id in skipped_document_ids:
        return False
    if not chunk.text.strip():
        return False
    if len(chunk.text.strip()) < MIN_INFORMATIVE_CHARS:
        return False
    if is_service_section_path(chunk.section_path):
        return False
    title = (chunk.section_title or "").strip().casefold()
    if title in SKIPPED_SECTION_TITLES:
        return False
    if chunk.chunk_type in {"gallery", "media"}:
        return False
    return True


def selection_reason(chunk: SourceChunk, selected_for_breed: list[SourceChunk]) -> str:
    if chunk.source == "thecatapi":
        return "structured CatAPI profile chunk"
    if not any(item.source == "wikipedia" for item in selected_for_breed):
        return "first informative Wikipedia chunk for breed"
    return "additional informative Wikipedia chunk from another section"


def chunk_sort_key(chunk: SourceChunk) -> tuple[int, str, str]:
    source_priority = 0 if chunk.source == "thecatapi" else 1
    return (source_priority, chunk.document_id, chunk.chunk_id)


def choose_chunks_for_breed(chunks: list[SourceChunk], max_chunks: int = 3) -> list[SourceChunk]:
    catapi = sorted([chunk for chunk in chunks if chunk.source == "thecatapi"], key=chunk_sort_key)
    wikipedia = sorted([chunk for chunk in chunks if chunk.source == "wikipedia"], key=chunk_sort_key)
    selected: list[SourceChunk] = []
    selected_paths: set[tuple[str, ...]] = set()
    if catapi:
        selected.append(catapi[0])

    for chunk in wikipedia:
        path_key = tuple(chunk.section_path or [chunk.section_title or ""])
        if path_key in selected_paths:
            continue
        selected.append(chunk)
        selected_paths.add(path_key)
        if len(selected) >= max_chunks:
            break
    return selected[:max_chunks]


def deterministic_sample_chunks(
    chunks: list[SourceChunk],
    skipped_document_ids: set[str],
    seed: int,
    breed_limit: int,
    target_count: int,
) -> list[SelectedChunk]:
    informative = [
        chunk for chunk in chunks if is_informative_chunk(chunk, skipped_document_ids)
    ]
    by_breed: dict[str, list[SourceChunk]] = defaultdict(list)
    for chunk in informative:
        by_breed[chunk.breed_id].append(chunk)

    eligible_breeds = sorted(breed_id for breed_id, rows in by_breed.items() if rows)
    rng = random.Random(seed)
    rng.shuffle(eligible_breeds)
    selected_breed_ids = sorted(eligible_breeds[:breed_limit])

    selected: list[SelectedChunk] = []
    for breed_id in selected_breed_ids:
        breed_selected: list[SourceChunk] = []
        for chunk in choose_chunks_for_breed(by_breed[breed_id], max_chunks=3):
            reason = selection_reason(chunk, breed_selected)
            breed_selected.append(chunk)
            selected.append(
                SelectedChunk(
                    chunk=chunk,
                    selection_reason=reason,
                    selection_index=len(selected),
                )
            )
            if len(selected) >= target_count:
                return selected
    return selected[:target_count]


def selected_breed_ids(selected: list[SelectedChunk]) -> list[str]:
    return sorted({item.chunk.breed_id for item in selected})


def selected_chunk_ids(selected: list[SelectedChunk]) -> list[str]:
    return [item.chunk.chunk_id for item in selected]


def chunk_hash_input(chunk: SourceChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "breed_id": chunk.breed_id,
        "text": chunk.text,
    }
