from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


SUPPORTED_SOURCE_RELATIONS = {
    "standalone_article",
    "covered_by_broader_article",
    "section_of_another_article",
}


SERVICE_SECTION_TITLES = {
    "references",
    "reference",
    "sources",
    "source",
    "notes",
    "bibliography",
    "literature",
    "further reading",
    "external links",
    "see also",
    "gallery",
    "galleries",
    "breed gallery",
    "photo gallery",
    "images",
    "примечания",
    "источники",
    "источник",
    "литература",
    "ссылки",
    "внешние ссылки",
    "см также",
    "галерея",
    "фотографии",
    "изображения",
}


class SourceScopeError(ValueError):
    """Raised when curated source-scope overrides are invalid."""


def normalize_heading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.casefold().strip()
    normalized = re.sub(r"[\s\u00a0]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(":：.。")
    normalized = normalized.replace("см.также", "см также")
    normalized = normalized.replace("см. также", "см также")
    normalized = normalized.replace(".", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_section_path(path: list[Any]) -> tuple[str, ...]:
    return tuple(normalize_heading(str(item)) for item in path if str(item).strip())


def is_service_section_title(title: str) -> bool:
    return normalize_heading(title) in SERVICE_SECTION_TITLES


def is_service_section_path(path: list[Any]) -> bool:
    return any(is_service_section_title(str(item)) for item in path)


def load_scope_overrides(path: Path, create_if_missing: bool = False) -> dict[str, dict[str, Any]]:
    if not path.exists():
        if create_if_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SourceScopeError(f"Expected source-scope overrides JSON object: {path}")
    return validate_scope_overrides(data)


def validate_scope_overrides(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        breed_id, language = parse_scope_key(key)
        if language not in {"en", "ru"}:
            raise SourceScopeError(f"Unsupported scope override language: {key}")
        if not isinstance(value, dict):
            raise SourceScopeError(f"Scope override value must be object: {key}")

        source_relation = value.get("source_relation")
        if source_relation not in SUPPORTED_SOURCE_RELATIONS:
            raise SourceScopeError(f"Unsupported source_relation for {key}: {source_relation!r}")

        include_lead = value.get("include_lead")
        if not isinstance(include_lead, bool):
            raise SourceScopeError(f"Scope override include_lead must be boolean: {key}")

        included_paths = value.get("included_section_paths")
        if not isinstance(included_paths, list):
            raise SourceScopeError(f"Scope override included_section_paths must be list: {key}")
        normalized_paths = []
        for section_path in included_paths:
            if not isinstance(section_path, list) or not all(
                isinstance(item, str) and item.strip() for item in section_path
            ):
                raise SourceScopeError(f"Invalid included section path for {key}: {section_path!r}")
            normalized_paths.append(section_path)

        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SourceScopeError(f"Scope override reason must be non-empty: {key}")

        validated[key] = {
            "breed_id": breed_id,
            "language": language,
            "source_relation": source_relation,
            "include_lead": include_lead,
            "included_section_paths": normalized_paths,
            "reason": reason.strip(),
        }
    return validated


def parse_scope_key(key: str) -> tuple[str, str]:
    if not isinstance(key, str) or key.count(":") != 1:
        raise SourceScopeError(f"Invalid scope override key: {key!r}")
    breed_id, language = key.split(":", maxsplit=1)
    if not breed_id or not language:
        raise SourceScopeError(f"Invalid scope override key: {key!r}")
    return breed_id, language


def scope_key(breed_id: str, language: str) -> str:
    return f"{breed_id}:{language}"
