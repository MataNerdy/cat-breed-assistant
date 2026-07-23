from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.wikidata_client import WikidataClient, WikidataClientError


TARGET_BREED_IDS = ("mcoo", "bsho", "sphy", "beng", "sibe")


def load_registry(path: Path, breed_ids: set[str]) -> list[dict[str, Any]]:
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
            if record.get("breed_id") in breed_ids:
                records.append(record)
    records.sort(key=lambda item: item["breed_id"])
    return records


def load_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected overrides JSON object: {path}")
    return data


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
        entity_id = override.get("wikidata_entity_id")
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
                return build_enrichment_record(
                    registry_record,
                    entity_id=entity_id,
                    entity=entity,
                    match_method="catapi_wikipedia_sitelink",
                    match_confidence=1.0,
                    warnings=[] if entity else ["Wikidata entity could not be loaded"],
                )

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
