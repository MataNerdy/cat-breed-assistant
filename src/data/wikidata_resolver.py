from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.data.wikidata_client import WikidataClient, WikidataClientError


def load_registry(path: Path, breed_ids: set[str] | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
            if breed_ids is None or record.get("breed_id") in breed_ids:
                records.append(record)
    records.sort(key=lambda item: item["breed_id"])
    return records


QID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")


def load_overrides(
    path: Path,
    known_breed_ids: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected overrides JSON object: {path}")
    overrides = {}
    for breed_id, value in data.items():
        if known_breed_ids is not None and breed_id not in known_breed_ids:
            raise ValueError(f"Unknown breed_id in Wikidata override: {breed_id}")
        if not isinstance(value, dict):
            raise ValueError(f"Expected Wikidata override object for {breed_id}")
        entity_id = value.get("entity_id") or value.get("wikidata_entity_id")
        if not isinstance(entity_id, str) or not QID_PATTERN.match(entity_id):
            raise ValueError(f"Invalid Wikidata Q-ID in override for {breed_id}")
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Wikidata override for {breed_id} requires a reason")
        overrides[breed_id] = {
            "entity_id": entity_id,
            "reason": reason.strip(),
        }
    return overrides


def ensure_overrides_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def resolve_breed_record(
    registry_record: dict[str, Any],
    client: WikidataClient,
    overrides: dict[str, dict[str, str]],
) -> dict[str, Any]:
    breed_id = registry_record["breed_id"]
    name_en = registry_record["name_en"]
    warnings = []

    override = overrides.get(breed_id)
    if override:
        entity_id = override.get("entity_id") or override.get("wikidata_entity_id")
        if entity_id:
            entity = client.get_entity(entity_id)
            return build_enrichment_record(
                registry_record,
                entity_id=entity_id,
                entity=entity,
                match_method="manual_override",
                match_confidence=1.0,
                warnings=[] if entity else ["Override entity could not be loaded"],
            )

    wikipedia_url = (
        registry_record.get("catapi", {}).get("raw", {}).get("wikipedia_url")
    )
    if isinstance(wikipedia_url, str) and wikipedia_url:
        try:
            entity_id = client.resolve_entity_id_from_wikipedia_url(wikipedia_url)
        except (ValueError, WikidataClientError) as exc:
            warnings.append(f"Wikipedia sitelink resolution failed: {exc}")
        else:
            if entity_id:
                entity = client.get_entity(entity_id)
                validation = validate_catapi_wikipedia_candidate(
                    registry_record,
                    entity,
                    wikipedia_url,
                )
                if validation["accepted"]:
                    return build_enrichment_record(
                        registry_record,
                        entity_id=entity_id,
                        entity=entity,
                        match_method="catapi_wikipedia_sitelink",
                        match_confidence=0.95,
                        warnings=[] if entity else ["Wikidata entity could not be loaded"],
                    )
                warnings.append(validation["warning"])

    label_match = resolve_single_exact_search_match(client.search_entities(name_en), name_en)
    if label_match["status"] == "matched":
        entity_id = label_match["entity_id"]
        entity = client.get_entity(entity_id)
        return build_enrichment_record(
            registry_record,
            entity_id=entity_id,
            entity=entity,
            match_method="exact_en_label",
            match_confidence=0.9,
            warnings=[] if entity else ["Wikidata entity could not be loaded"],
        )
    if label_match["status"] == "ambiguous":
        warnings.append("Multiple exact English label candidates found")

    for alias in registry_record.get("aliases_en", []):
        alias_match = resolve_single_exact_search_match(
            client.search_entities(alias),
            alias,
            use_aliases=True,
        )
        if alias_match["status"] == "matched":
            entity_id = alias_match["entity_id"]
            entity = client.get_entity(entity_id)
            return build_enrichment_record(
                registry_record,
                entity_id=entity_id,
                entity=entity,
                match_method="exact_en_alias",
                match_confidence=0.8,
                warnings=[] if entity else ["Wikidata entity could not be loaded"],
            )
        if alias_match["status"] == "ambiguous":
            warnings.append(f"Multiple exact English alias candidates found: {alias}")
            break

    warnings.append("Wikidata entity could not be resolved")
    return build_enrichment_record(
        registry_record,
        entity_id=None,
        entity=None,
        match_method="unresolved",
        match_confidence=0.0,
        warnings=warnings,
    )


def resolve_single_exact_search_match(
    results: list[dict[str, Any]],
    expected_text: str,
    use_aliases: bool = False,
) -> dict[str, str | None]:
    expected = expected_text.casefold()
    entity_ids = []

    for result in results:
        label = str(result.get("label", "")).casefold()
        aliases = [
            str(alias).casefold()
            for alias in result.get("aliases", [])
            if isinstance(alias, str)
        ]
        if label == expected or (use_aliases and expected in aliases):
            entity_id = result.get("id")
            if isinstance(entity_id, str) and entity_id not in entity_ids:
                entity_ids.append(entity_id)

    if len(entity_ids) == 1:
        return {"status": "matched", "entity_id": entity_ids[0]}
    if len(entity_ids) > 1:
        return {"status": "ambiguous", "entity_id": None}
    return {"status": "unmatched", "entity_id": None}


def normalize_match_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("_", " ")
    value = re.sub(r"[-–—]", " ", value)
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    value = re.sub(r"\bcat\b$", "", value).strip()
    return value


def source_names_for(registry_record: dict[str, Any]) -> set[str]:
    values = [registry_record.get("name_en")]
    values.extend(registry_record.get("aliases_en") or [])
    return {
        normalize_match_text(value)
        for value in values
        if isinstance(value, str) and value.strip()
    }


def page_title_from_wikipedia_url(wikipedia_url: str) -> str:
    from src.data.wikidata_client import parse_wikipedia_url

    _, _, title = parse_wikipedia_url(wikipedia_url)
    return title


def validate_catapi_wikipedia_candidate(
    registry_record: dict[str, Any],
    entity: dict[str, Any] | None,
    wikipedia_url: str,
) -> dict[str, Any]:
    if entity is None:
        return {"accepted": False, "warning": "catapi_wikipedia_entity_missing"}

    source_names = source_names_for(registry_record)
    entity_label = extract_label(entity, "en")
    normalized_label = normalize_match_text(entity_label) if entity_label else ""
    page_title = page_title_from_wikipedia_url(wikipedia_url)
    normalized_title = normalize_match_text(page_title)

    if (
        normalized_label in source_names
        or normalized_title in source_names
        or has_allowed_breed_modifier(normalized_label, source_names)
        or has_allowed_breed_modifier(normalized_title, source_names)
    ):
        return {"accepted": True, "warning": None}
    return {"accepted": False, "warning": "catapi_wikipedia_name_mismatch"}


def has_allowed_breed_modifier(candidate: str, source_names: set[str]) -> bool:
    allowed_suffixes = {"shorthair", "longhair", "bobtail"}
    for source_name in source_names:
        for suffix in allowed_suffixes:
            if candidate == f"{source_name} {suffix}":
                return True
    return False


def build_enrichment_record(
    registry_record: dict[str, Any],
    entity_id: str | None,
    entity: dict[str, Any] | None,
    match_method: str,
    match_confidence: float,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "breed_id": registry_record["breed_id"],
        "name_en": registry_record["name_en"],
        "entity_id": entity_id,
        "match_method": match_method,
        "match_confidence": match_confidence,
        "labels": {
            "en": extract_label(entity, "en"),
            "ru": extract_label(entity, "ru"),
        },
        "aliases": {
            "en": extract_aliases(entity, "en"),
            "ru": extract_aliases(entity, "ru"),
        },
        "sitelinks": {
            "enwiki": extract_sitelink(entity, "enwiki"),
            "ruwiki": extract_sitelink(entity, "ruwiki"),
        },
        "source": "wikidata",
        "warnings": warnings,
    }


def extract_label(entity: dict[str, Any] | None, language: str) -> str | None:
    if not entity:
        return None
    label = entity.get("labels", {}).get(language, {})
    value = label.get("value") if isinstance(label, dict) else None
    return value if isinstance(value, str) else None


def extract_aliases(entity: dict[str, Any] | None, language: str) -> list[str]:
    if not entity:
        return []
    aliases = entity.get("aliases", {}).get(language, [])
    if not isinstance(aliases, list):
        return []
    values = []
    seen = set()
    for alias in aliases:
        value = alias.get("value") if isinstance(alias, dict) else None
        if isinstance(value, str) and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def extract_sitelink(entity: dict[str, Any] | None, site_key: str) -> str | None:
    if not entity:
        return None
    sitelink = entity.get("sitelinks", {}).get(site_key, {})
    title = sitelink.get("title") if isinstance(sitelink, dict) else None
    return title if isinstance(title, str) else None


def resolve_registry_records(
    records: list[dict[str, Any]],
    client: WikidataClient,
    overrides: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    enrichment = [
        resolve_breed_record(record, client=client, overrides=overrides)
        for record in records
    ]
    enrichment.sort(key=lambda item: item["breed_id"])
    return enrichment
