from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
CATAPI_SOURCE_URL = "https://api.thecatapi.com/v1/breeds"


class KnowledgeDocumentError(ValueError):
    """Raised when knowledge documents cannot be built safely."""


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
                raise KnowledgeDocumentError(
                    f"Invalid JSONL on line {line_number}: {path}"
                ) from exc
            if not isinstance(record, dict):
                raise KnowledgeDocumentError(
                    f"Expected JSON object on line {line_number}: {path}"
                )
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


def index_by_breed_id(records: list[dict[str, Any]], source_name: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        breed_id = record.get("breed_id")
        if not isinstance(breed_id, str) or not breed_id:
            raise KnowledgeDocumentError(f"{source_name} record has missing breed_id")
        if breed_id in indexed:
            raise KnowledgeDocumentError(f"Duplicate breed_id in {source_name}: {breed_id}")
        indexed[breed_id] = record
    return indexed


def unique_strings(values: list[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        folded = value.casefold()
        if value and folded not in seen:
            result.append(value)
            seen.add(folded)
    return result


def breed_names_and_aliases(
    registry_record: dict[str, Any],
    wikidata_record: dict[str, Any] | None,
) -> tuple[str, str | None, list[str]]:
    name_en = registry_record["name_en"]
    name_ru = registry_record.get("name_ru")
    if not name_ru and wikidata_record:
        name_ru = (wikidata_record.get("labels") or {}).get("ru")

    aliases = unique_strings(
        list(registry_record.get("aliases_en") or [])
        + list(registry_record.get("aliases_ru") or [])
        + list((wikidata_record or {}).get("aliases", {}).get("en") or [])
        + list((wikidata_record or {}).get("aliases", {}).get("ru") or [])
    )
    return name_en, name_ru, aliases


def format_rating(value: Any) -> str | None:
    if isinstance(value, int) and 0 <= value <= 5:
        return f"{value}/5"
    return None


def format_boolean(value: Any) -> str | None:
    if value == 1:
        return "yes"
    if value == 0:
        return "no"
    return None


def add_line(lines: list[str], label: str, value: Any, suffix: str = "") -> None:
    if value is None or value == "":
        return
    if isinstance(value, str) and not value.strip():
        return
    lines.append(f"{label}: {value}{suffix}")


def split_temperament(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return unique_strings(value.split(","))


def build_catapi_text(raw: dict[str, Any]) -> str:
    lines: list[str] = []
    add_line(lines, "Breed", raw.get("name"))
    add_line(lines, "Alternative names", raw.get("alt_names"))
    add_line(lines, "Temperament", raw.get("temperament"))
    add_line(lines, "Origin", raw.get("origin"))
    add_line(lines, "Country code", raw.get("country_code"))
    add_line(lines, "Description", raw.get("description"))
    add_line(lines, "Life span", raw.get("life_span"), " years")

    weight = raw.get("weight") if isinstance(raw.get("weight"), dict) else {}
    metric = weight.get("metric")
    imperial = weight.get("imperial")
    if metric or imperial:
        weight_parts = []
        if metric:
            weight_parts.append(f"{metric} kg")
        if imperial:
            weight_parts.append(f"{imperial} lb")
        lines.append(f"Weight: {' / '.join(weight_parts)}")

    rating_fields = [
        ("adaptability", "Adaptability"),
        ("affection_level", "Affection level"),
        ("child_friendly", "Child friendly"),
        ("dog_friendly", "Dog friendly"),
        ("energy_level", "Energy level"),
        ("grooming", "Grooming"),
        ("health_issues", "Health issues"),
        ("intelligence", "Intelligence"),
        ("shedding_level", "Shedding level"),
        ("social_needs", "Social needs"),
        ("stranger_friendly", "Stranger friendly"),
        ("vocalisation", "Vocalisation"),
    ]
    for key, label in rating_fields:
        rating = format_rating(raw.get(key))
        if rating:
            lines.append(f"{label}: {rating}")

    boolean_fields = [
        ("experimental", "Experimental"),
        ("hairless", "Hairless"),
        ("natural", "Natural"),
        ("rare", "Rare"),
        ("rex", "Rex"),
        ("suppressed_tail", "Suppressed tail"),
        ("short_legs", "Short legs"),
        ("hypoallergenic", "Hypoallergenic"),
    ]
    for key, label in boolean_fields:
        value = format_boolean(raw.get(key))
        if value:
            lines.append(f"{label}: {value}")

    add_line(lines, "Wikipedia URL", raw.get("wikipedia_url"))
    return "\n".join(lines)


def build_catapi_document(
    registry_record: dict[str, Any],
    wikidata_record: dict[str, Any] | None,
) -> dict[str, Any]:
    breed_id = registry_record["breed_id"]
    raw = registry_record.get("catapi", {}).get("raw")
    if not isinstance(raw, dict):
        raise KnowledgeDocumentError(f"Registry record has missing catapi.raw: {breed_id}")
    name_en, name_ru, aliases = breed_names_and_aliases(registry_record, wikidata_record)
    weight = raw.get("weight") if isinstance(raw.get("weight"), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": f"{breed_id}:catapi:profile",
        "breed_id": breed_id,
        "breed_name_en": name_en,
        "breed_name_ru": name_ru,
        "aliases": aliases,
        "language": "en",
        "source": "thecatapi",
        "document_type": "structured_profile",
        "title": f"{name_en} — structured breed profile",
        "text": build_catapi_text(raw),
        "structured_data": {
            "temperament": split_temperament(raw.get("temperament")),
            "origin": raw.get("origin"),
            "country_code": raw.get("country_code"),
            "weight": {
                "metric": weight.get("metric"),
                "imperial": weight.get("imperial"),
            },
            "life_span": raw.get("life_span"),
        },
        "provenance": {
            "source_name": "TheCatAPI",
            "source_url": CATAPI_SOURCE_URL,
        },
        "warnings": [],
    }


def build_wikipedia_document(
    article_record: dict[str, Any],
    registry_by_id: dict[str, dict[str, Any]],
    wikidata_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    breed_id = article_record.get("breed_id")
    if breed_id not in registry_by_id:
        raise KnowledgeDocumentError(f"Wikipedia article has unknown breed_id: {breed_id}")

    registry_record = registry_by_id[breed_id]
    wikidata_record = wikidata_by_id.get(breed_id)
    name_en, name_ru, aliases = breed_names_and_aliases(registry_record, wikidata_record)
    language = article_record["language"]
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": f"{breed_id}:wikipedia:{language}",
        "breed_id": breed_id,
        "breed_name_en": name_en,
        "breed_name_ru": name_ru,
        "aliases": aliases,
        "language": language,
        "source": "wikipedia",
        "document_type": "wikipedia_article",
        "title": article_record["title"],
        "lead": article_record.get("lead") or "",
        "sections": article_record.get("sections") or [],
        "provenance": {
            "source_url": article_record.get("source_url"),
            "page_id": article_record.get("page_id"),
            "revision_id": article_record.get("revision_id"),
            "retrieved_at": article_record.get("retrieved_at"),
            "source_resolution": article_record.get("source_resolution")
            or {
                "method": "wikidata_sitelink",
                "source_relation": "standalone_article",
                "reason": None,
            },
        },
        "warnings": article_record.get("warnings") or [],
    }


def build_knowledge_documents(
    registry_records: list[dict[str, Any]],
    wikidata_records: list[dict[str, Any]],
    wikipedia_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    registry_by_id = index_by_breed_id(registry_records, "registry")
    wikidata_by_id = index_by_breed_id(wikidata_records, "wikidata")
    missing_wikidata = set(registry_by_id) - set(wikidata_by_id)
    if missing_wikidata:
        raise KnowledgeDocumentError(
            f"Missing Wikidata enrichment for breed_ids: {sorted(missing_wikidata)}"
        )

    documents = [
        build_catapi_document(record, wikidata_by_id.get(record["breed_id"]))
        for record in registry_records
    ]
    documents.extend(
        build_wikipedia_document(record, registry_by_id, wikidata_by_id)
        for record in wikipedia_records
    )
    ensure_unique_document_ids(documents)
    documents.sort(
        key=lambda item: (
            item["breed_id"],
            item["source"],
            item["language"],
            item["document_type"],
            item["document_id"],
        )
    )
    return documents


def ensure_unique_document_ids(documents: list[dict[str, Any]]) -> None:
    counts = Counter(document["document_id"] for document in documents)
    duplicates = sorted(document_id for document_id, count in counts.items() if count > 1)
    if duplicates:
        raise KnowledgeDocumentError(f"Duplicate document IDs: {duplicates}")


def build_knowledge_documents_report(documents: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_ids = [
        document_id
        for document_id, count in Counter(
            document["document_id"] for document in documents
        ).items()
        if count > 1
    ]
    source_relations = Counter()
    for document in documents:
        if document["source"] == "thecatapi":
            source_relations["standalone_profile"] += 1
        else:
            resolution = document.get("provenance", {}).get("source_resolution") or {}
            source_relations[resolution.get("source_relation") or "unknown"] += 1

    return {
        "total_documents": len(documents),
        "catapi_documents": sum(1 for item in documents if item["source"] == "thecatapi"),
        "wikipedia_documents": sum(
            1 for item in documents if item["source"] == "wikipedia"
        ),
        "ru_wikipedia_documents": sum(
            1
            for item in documents
            if item["source"] == "wikipedia" and item["language"] == "ru"
        ),
        "en_wikipedia_documents": sum(
            1
            for item in documents
            if item["source"] == "wikipedia" and item["language"] == "en"
        ),
        "documents_by_source_relation": dict(sorted(source_relations.items())),
        "documents_by_language": dict(
            sorted(Counter(item["language"] for item in documents).items())
        ),
        "unique_breeds": len({item["breed_id"] for item in documents}),
        "duplicate_document_ids": sorted(duplicate_ids),
        "documents_with_warnings": sum(1 for item in documents if item.get("warnings")),
    }
