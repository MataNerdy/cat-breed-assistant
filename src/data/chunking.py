from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from src.data.source_scope import (
    SourceScopeError,
    is_service_section_path,
    load_scope_overrides,
    normalize_section_path,
    scope_key,
)


SCHEMA_VERSION = "1.0"
BROADER_SKIP_REASON = (
    "Broader source requires breed-specific section approval before chunking"
)
HARD_SPLIT_WARNING = "Text was hard-split because a single sentence exceeded max_chars"


class ChunkingError(ValueError):
    """Raised when knowledge documents cannot be chunked safely."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChunkingError(f"Invalid JSONL on line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise ChunkingError(f"Expected JSON object on line {line_number}: {path}")
            records.append(record)
    return records


def write_jsonl_atomic(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(record: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def load_broader_overrides(path: Path) -> dict[str, Any]:
    try:
        return load_scope_overrides(path, create_if_missing=True)
    except SourceScopeError as exc:
        raise ChunkingError(str(exc)) from exc


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return ascii_text or "section"


def normalize_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = re.sub(r"[ \t\r\f\v]+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def hard_split(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def split_long_paragraph(
    paragraph: str,
    target_chars: int,
    max_chars: int,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    sentences = split_sentences(paragraph)
    if not sentences:
        sentences = [paragraph]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(hard_split(sentence, max_chars))
            warnings.append(HARD_SPLIT_WARNING)
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars and (
            len(candidate) <= target_chars or not current
        ):
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks, sorted(set(warnings))


def split_text(
    text: str,
    target_chars: int = 2200,
    max_chars: int = 3000,
) -> tuple[list[str], list[str]]:
    text = text.strip()
    if not text:
        return [], []
    if len(text) <= max_chars:
        return [text], []

    warnings: list[str] = []
    parts: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in normalize_paragraphs(text):
        if len(paragraph) > max_chars:
            if current:
                parts.append("\n\n".join(current))
                current = []
                current_length = 0
            sentence_parts, sentence_warnings = split_long_paragraph(
                paragraph, target_chars, max_chars
            )
            parts.extend(sentence_parts)
            warnings.extend(sentence_warnings)
            continue

        separator = 2 if current else 0
        candidate_length = current_length + separator + len(paragraph)
        if current and candidate_length > max_chars:
            parts.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        elif current and current_length >= target_chars:
            parts.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length = candidate_length

    if current:
        parts.append("\n\n".join(current))
    return [part for part in parts if part.strip()], sorted(set(warnings))


def source_relation_for(document: dict[str, Any]) -> str:
    if document["source"] == "thecatapi":
        return "standalone_profile"
    resolution = document.get("provenance", {}).get("source_resolution") or {}
    return resolution.get("source_relation") or "unknown"


def embedding_header(document: dict[str, Any], section_path: list[str]) -> str:
    if document["language"] == "ru":
        breed = document.get("breed_name_ru") or document["breed_name_en"]
        return f"Порода: {breed}\nРаздел: {' > '.join(section_path)}"
    return f"Breed: {document['breed_name_en']}\nSection: {' > '.join(section_path)}"


def make_chunk(
    document: dict[str, Any],
    chunk_id: str,
    chunk_type: str,
    section_title: str,
    section_path: list[str],
    part_index: int,
    text: str,
    section_index: str | None = None,
    section_level: int | None = None,
    parent_index: str | None = None,
    warnings: list[str] | None = None,
    selection_method: str | None = None,
) -> dict[str, Any]:
    provenance = document.get("provenance") or {}
    if document["source"] == "thecatapi":
        chunk_provenance = {
            "source_name": provenance.get("source_name"),
            "source_url": provenance.get("source_url"),
        }
    else:
        chunk_provenance = {
            "source_url": provenance.get("source_url"),
            "page_id": provenance.get("page_id"),
            "revision_id": provenance.get("revision_id"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "document_id": document["document_id"],
        "breed_id": document["breed_id"],
        "breed_name_en": document["breed_name_en"],
        "breed_name_ru": document.get("breed_name_ru"),
        "aliases": document.get("aliases") or [],
        "language": document["language"],
        "source": document["source"],
        "source_relation": source_relation_for(document),
        "selection_method": selection_method,
        "chunk_type": chunk_type,
        "section_title": section_title,
        "section_path": section_path,
        "section_index": section_index,
        "section_level": section_level,
        "parent_index": parent_index,
        "part_index": part_index,
        "text": text,
        "embedding_text": f"{embedding_header(document, section_path)}\n\n{text}",
        "text_length": len(text),
        "provenance": chunk_provenance,
        "warnings": warnings or [],
    }


def chunk_catapi_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    text = (document.get("text") or "").strip()
    if not text:
        return []
    return [
        make_chunk(
            document=document,
            chunk_id=f"{document['breed_id']}:catapi:profile:000",
            chunk_type="structured_profile",
            section_title="Breed profile",
            section_path=["Breed profile"],
            part_index=0,
            text=text,
            warnings=document.get("warnings") or [],
        )
    ]


def chunk_wikipedia_document(
    document: dict[str, Any],
    target_chars: int,
    max_chars: int,
    scope_override: dict[str, Any] | None = None,
    excluded_section_paths: set[tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    include_lead = True
    approved_paths: set[tuple[str, ...]] | None = None
    selection_method = None
    if scope_override:
        include_lead = bool(scope_override["include_lead"])
        approved_paths = {
            normalize_section_path(path)
            for path in scope_override["included_section_paths"]
        }
        selection_method = (
            "manual_section_approval"
            if scope_override["source_relation"] == "section_of_another_article"
            else "manual_scope_override"
        )
    excluded_section_paths = excluded_section_paths or set()

    lead = (document.get("lead") or "").strip()
    if lead and include_lead:
        lead_parts, lead_split_warnings = split_text(lead, target_chars, max_chars)
        for part_index, lead_part in enumerate(lead_parts):
            chunks.append(
                make_chunk(
                    document=document,
                    chunk_id=f"{document['document_id']}:overview:{part_index:03d}",
                    chunk_type="overview",
                    section_title="Overview",
                    section_path=["Overview"],
                    part_index=part_index,
                    text=lead_part,
                    warnings=sorted(
                        set((document.get("warnings") or []) + lead_split_warnings)
                    ),
                    selection_method=selection_method,
                )
            )

    used_section_slugs: Counter[str] = Counter()
    matched_paths: set[tuple[str, ...]] = set()
    for section in document.get("sections") or []:
        text = (section.get("text") or "").strip()
        if not text:
            continue
        section_title = section.get("title") or "Untitled section"
        section_path = section.get("section_path") or [section_title]
        normalized_path = normalize_section_path(section_path)
        if is_service_section_path(section_path):
            continue
        if approved_paths is None and normalized_path in excluded_section_paths:
            continue
        if approved_paths is not None and normalized_path not in approved_paths:
            continue
        matched_paths.add(normalized_path)
        base_slug = f"{section.get('index') or 'section'}-{slugify(section_title)}"
        used_section_slugs[base_slug] += 1
        slug = (
            base_slug
            if used_section_slugs[base_slug] == 1
            else f"{base_slug}-{used_section_slugs[base_slug]}"
        )
        parts, split_warnings = split_text(text, target_chars, max_chars)
        for part_index, part in enumerate(parts):
            chunks.append(
                make_chunk(
                    document=document,
                    chunk_id=(
                        f"{document['document_id']}:{slug}:{part_index:03d}"
                    ),
                    chunk_type="section",
                    section_title=section_title,
                    section_path=section_path,
                    section_index=section.get("index"),
                    section_level=section.get("level"),
                    parent_index=section.get("parent_index"),
                    part_index=part_index,
                    text=part,
                    warnings=split_warnings,
                    selection_method=selection_method,
                )
            )
    if approved_paths is not None:
        missing_paths = sorted(approved_paths - matched_paths)
        if missing_paths:
            raise ChunkingError(
                f"Approved section paths were not found for {document['document_id']}: "
                f"{missing_paths}"
            )
    return chunks


def skipped_broader_record(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document["document_id"],
        "breed_id": document["breed_id"],
        "language": document["language"],
        "source_relation": source_relation_for(document),
        "reason": BROADER_SKIP_REASON,
    }


def build_chunks(
    documents: list[dict[str, Any]],
    broader_overrides: dict[str, Any] | None = None,
    target_chars: int = 2200,
    max_chars: int = 3000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    broader_overrides = broader_overrides or {}

    chunks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    effective_documents: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for document in documents:
        key = scope_key(document["breed_id"], document["language"])
        scope_override = broader_overrides.get(key)
        if scope_override and document["source"] == "wikipedia":
            document = {
                **document,
                "provenance": {
                    **(document.get("provenance") or {}),
                    "source_resolution": {
                        "method": "manual_section_approval"
                        if scope_override["source_relation"] == "section_of_another_article"
                        else "manual_scope_override",
                        "source_relation": scope_override["source_relation"],
                        "reason": scope_override["reason"],
                    },
                },
            }
        effective_documents.append((document, scope_override))

    approved_paths_by_source_url: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for document, scope_override in effective_documents:
        if document["source"] != "wikipedia" or not scope_override:
            continue
        source_url = (document.get("provenance") or {}).get("source_url")
        if not isinstance(source_url, str) or not source_url:
            continue
        for section_path in scope_override.get("included_section_paths") or []:
            approved_paths_by_source_url[source_url].add(normalize_section_path(section_path))

    for document, scope_override in effective_documents:
        relation = source_relation_for(document)
        if relation == "section_of_another_article" and not scope_override:
            raise ChunkingError(f"section_of_another_article requires override: {key}")
        if relation == "covered_by_broader_article" and not (
            scope_override and scope_override.get("included_section_paths")
        ):
            skipped.append(skipped_broader_record(document))
            continue
        if document["source"] == "thecatapi":
            chunks.extend(chunk_catapi_document(document))
        elif document["source"] == "wikipedia":
            source_url = (document.get("provenance") or {}).get("source_url")
            excluded_paths = (
                approved_paths_by_source_url.get(source_url, set())
                if isinstance(source_url, str)
                else set()
            )
            chunks.extend(
                chunk_wikipedia_document(
                    document,
                    target_chars,
                    max_chars,
                    scope_override=scope_override,
                    excluded_section_paths=excluded_paths,
                )
            )
        else:
            raise ChunkingError(f"Unsupported document source: {document['source']}")

    ensure_unique_chunk_ids(chunks)
    integrity = build_corpus_integrity_report(chunks, skipped)
    enforce_corpus_integrity(integrity)
    chunks.sort(
        key=lambda item: (
            item["breed_id"],
            item["source"],
            item["language"],
            item["document_id"],
            item["chunk_type"],
            item["section_index"] or "",
            item["part_index"],
            item["chunk_id"],
        )
    )
    skipped.sort(key=lambda item: (item["breed_id"], item["language"], item["document_id"]))
    return chunks, skipped


def build_corpus_integrity_report(
    chunks: list[dict[str, Any]],
    skipped_broader: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    skipped_broader = skipped_broader or []
    wikipedia_chunks = [chunk for chunk in chunks if chunk["source"] == "wikipedia"]

    def shared_by(field: str) -> list[dict[str, Any]]:
        values: dict[Any, set[str]] = defaultdict(set)
        for chunk in wikipedia_chunks:
            value = chunk.get("provenance", {}).get(field) if field != "source_url" else chunk.get("provenance", {}).get("source_url")
            if value:
                values[value].add(chunk["breed_id"])
        return [
            {field: value, "breed_ids": sorted(breed_ids)}
            for value, breed_ids in sorted(values.items(), key=lambda item: str(item[0]))
            if len(breed_ids) > 1
        ]

    text_groups = duplicate_groups_by_text(wikipedia_chunks, "text")
    embedding_groups = duplicate_groups_by_text(chunks, "embedding_text")
    service_paths = [
        {
            "chunk_id": chunk["chunk_id"],
            "breed_id": chunk["breed_id"],
            "section_path": chunk["section_path"],
        }
        for chunk in chunks
        if is_service_section_path(chunk.get("section_path") or [])
    ]
    shared_standalone = []
    for item in shared_by("source_url"):
        related = [
            chunk
            for chunk in wikipedia_chunks
            if chunk.get("provenance", {}).get("source_url") == item["source_url"]
        ]
        if any(chunk["source_relation"] == "standalone_article" for chunk in related):
            if len({chunk["breed_id"] for chunk in related if chunk["source_relation"] == "standalone_article"}) > 1:
                shared_standalone.append(item)

    return {
        "shared_source_url_between_breeds": shared_by("source_url"),
        "shared_page_id_between_breeds": shared_by("page_id"),
        "exact_duplicate_text_between_breeds": text_groups,
        "exact_duplicate_embedding_text": embedding_groups,
        "service_section_paths_in_chunks": service_paths,
        "standalone_article_sources_used_by_multiple_breeds": shared_standalone,
        "scoped_documents_without_override": [
            chunk["document_id"]
            for chunk in wikipedia_chunks
            if chunk["source_relation"] == "section_of_another_article"
            and chunk.get("selection_method") != "manual_section_approval"
        ],
        "overrides_with_missing_section_paths": [],
        "skipped_broader_sources": skipped_broader,
        "forbidden_cross_breed_duplicate_groups": text_groups,
    }


def duplicate_groups_by_text(chunks: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for chunk in chunks:
        value = chunk.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        grouped[value].append(
            {
                "chunk_id": chunk["chunk_id"],
                "breed_id": chunk["breed_id"],
                "document_id": chunk["document_id"],
            }
        )
    duplicates = []
    for value, rows in grouped.items():
        breed_ids = sorted({row["breed_id"] for row in rows})
        if len(breed_ids) > 1:
            duplicates.append(
                {
                    "text_length": len(value),
                    "breed_ids": breed_ids,
                    "chunks": sorted(rows, key=lambda item: item["chunk_id"]),
                }
            )
    return sorted(duplicates, key=lambda item: (item["breed_ids"], item["text_length"]))


def enforce_corpus_integrity(report: dict[str, Any]) -> None:
    errors = []
    if report["service_section_paths_in_chunks"]:
        errors.append("service section paths reached chunks")
    if report["standalone_article_sources_used_by_multiple_breeds"]:
        errors.append("standalone article source is shared by multiple breeds")
    if report["scoped_documents_without_override"]:
        errors.append("scoped documents without override")
    if report["overrides_with_missing_section_paths"]:
        errors.append("scope overrides with missing section paths")
    if report["forbidden_cross_breed_duplicate_groups"]:
        errors.append("cross-breed duplicate Wikipedia chunk text")
    if errors:
        raise ChunkingError("; ".join(errors))


def ensure_unique_chunk_ids(chunks: list[dict[str, Any]]) -> None:
    counts = Counter(chunk["chunk_id"] for chunk in chunks)
    duplicates = sorted(chunk_id for chunk_id, count in counts.items() if count > 1)
    if duplicates:
        raise ChunkingError(f"Duplicate chunk IDs: {duplicates}")


def size_bucket(length: int) -> str:
    if length <= 500:
        return "0-500"
    if length <= 1000:
        return "501-1000"
    if length <= 2000:
        return "1001-2000"
    if length <= 2500:
        return "2001-2500"
    if length <= 3000:
        return "2501-3000"
    return ">3000"


def duplicate_text_within_documents(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_document: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        by_document[chunk["document_id"]][chunk["text"]] += 1
    duplicates = []
    for document_id, counts in by_document.items():
        duplicate_count = sum(1 for count in counts.values() if count > 1)
        if duplicate_count:
            duplicates.append(
                {"document_id": document_id, "duplicate_text_values": duplicate_count}
            )
    return sorted(duplicates, key=lambda item: item["document_id"])


def build_chunks_report(
    chunks: list[dict[str, Any]],
    skipped_broader: list[dict[str, Any]],
) -> dict[str, Any]:
    lengths = [chunk["text_length"] for chunk in chunks]
    chunk_id_counts = Counter(chunk["chunk_id"] for chunk in chunks)
    split_sections = [
        {"document_id": document_id, "section_index": section_index, "parts": count}
        for (document_id, section_index), count in sorted(
            Counter(
                (chunk["document_id"], chunk["section_index"])
                for chunk in chunks
                if chunk["chunk_type"] == "section"
            ).items()
        )
        if count > 1
    ]
    chunks_per_breed = Counter(chunk["breed_id"] for chunk in chunks)
    return {
        "total_chunks": len(chunks),
        "catapi_chunks": sum(1 for chunk in chunks if chunk["source"] == "thecatapi"),
        "wikipedia_overview_chunks": sum(
            1 for chunk in chunks if chunk["chunk_type"] == "overview"
        ),
        "wikipedia_section_chunks": sum(
            1 for chunk in chunks if chunk["chunk_type"] == "section"
        ),
        "ru_chunks": sum(1 for chunk in chunks if chunk["language"] == "ru"),
        "en_chunks": sum(1 for chunk in chunks if chunk["language"] == "en"),
        "unique_breeds_represented": len({chunk["breed_id"] for chunk in chunks}),
        "chunks_by_source": dict(sorted(Counter(chunk["source"] for chunk in chunks).items())),
        "chunks_by_source_relation": dict(
            sorted(Counter(chunk["source_relation"] for chunk in chunks).items())
        ),
        "sections_split_into_multiple_chunks": split_sections,
        "maximum_chunk_length": max(lengths) if lengths else 0,
        "median_chunk_length": median(lengths) if lengths else 0,
        "minimum_non_empty_chunk_length": min(lengths) if lengths else 0,
        "chunks_longer_than_3000_characters": sum(1 for length in lengths if length > 3000),
        "empty_chunks": sum(1 for length in lengths if length == 0),
        "duplicate_chunk_ids": sorted(
            chunk_id for chunk_id, count in chunk_id_counts.items() if count > 1
        ),
        "duplicate_text_within_one_document": duplicate_text_within_documents(chunks),
        "broader_documents_skipped": len(skipped_broader),
        "warnings": [
            {"chunk_id": chunk["chunk_id"], "warnings": chunk["warnings"]}
            for chunk in chunks
            if chunk.get("warnings")
        ],
        "size_distribution": dict(
            sorted(Counter(size_bucket(length) for length in lengths).items())
        ),
        "longest_20_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "breed_id": chunk["breed_id"],
                "source": chunk["source"],
                "language": chunk["language"],
                "text_length": chunk["text_length"],
                "section_title": chunk["section_title"],
            }
            for chunk in sorted(chunks, key=lambda item: item["text_length"], reverse=True)[:20]
        ],
        "chunks_with_warnings": [
            {"chunk_id": chunk["chunk_id"], "warnings": chunk["warnings"]}
            for chunk in chunks
            if chunk.get("warnings")
        ],
        "skipped_broader_sources": skipped_broader,
        "corpus_integrity": build_corpus_integrity_report(chunks, skipped_broader),
        "chunks_per_breed": dict(sorted(chunks_per_breed.items())),
        "breeds_with_min_chunks": sorted(
            breed_id
            for breed_id, count in chunks_per_breed.items()
            if count == min(chunks_per_breed.values())
        )
        if chunks_per_breed
        else [],
        "breeds_with_max_chunks": sorted(
            breed_id
            for breed_id, count in chunks_per_breed.items()
            if count == max(chunks_per_breed.values())
        )
        if chunks_per_breed
        else [],
    }
