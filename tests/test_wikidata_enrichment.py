from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

from scripts.build_wikidata_enrichment import write_jsonl_atomic
from src.data.wikidata_client import WikidataClient, WikidataClientError
from src.data.wikidata_resolver import (
    build_enrichment_record,
    load_registry,
    resolve_breed_record,
    resolve_registry_records,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.headers = {}
        self.calls = []

    def get(self, url: str, params: dict | None = None, timeout: int | None = None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self.responses:
            raise AssertionError("Unexpected HTTP call")
        return self.responses.pop(0)


def entity_payload(
    entity_id: str = "Q123",
    en_label: str = "Maine Coon",
    ru_label: str = "Мейн-кун",
    en_aliases: list[str] | None = None,
    ru_aliases: list[str] | None = None,
    enwiki: str = "Maine Coon",
    ruwiki: str = "Мейн-кун",
) -> dict:
    return {
        "entities": {
            entity_id: {
                "id": entity_id,
                "labels": {
                    "en": {"language": "en", "value": en_label},
                    "ru": {"language": "ru", "value": ru_label},
                },
                "aliases": {
                    "en": [
                        {"language": "en", "value": value}
                        for value in (en_aliases or [])
                    ],
                    "ru": [
                        {"language": "ru", "value": value}
                        for value in (ru_aliases or [])
                    ],
                },
                "sitelinks": {
                    "enwiki": {"site": "enwiki", "title": enwiki},
                    "ruwiki": {"site": "ruwiki", "title": ruwiki},
                },
            }
        }
    }


def registry_record(
    breed_id: str = "mcoo",
    name_en: str = "Maine Coon",
    wikipedia_url: str | None = "https://en.wikipedia.org/wiki/Maine_Coon",
    aliases_en: list[str] | None = None,
) -> dict:
    return {
        "breed_id": breed_id,
        "name_en": name_en,
        "aliases_en": aliases_en or [],
        "catapi": {
            "raw": {
                "wikipedia_url": wikipedia_url,
            }
        },
    }


def test_entity_id_is_resolved_from_wikipedia_url(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(
                {"query": {"pages": {"1": {"pageprops": {"wikibase_item": "Q123"}}}}}
            )
        ]
    )
    client = WikidataClient(cache_dir=tmp_path, session=session)

    entity_id = client.resolve_entity_id_from_wikipedia_url(
        "https://en.wikipedia.org/wiki/Maine_Coon"
    )

    assert entity_id == "Q123"
    assert session.calls[0]["params"]["titles"] == "Maine Coon"


def test_entity_is_loaded_and_saved_to_cache(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(entity_payload("Q123"))])
    client = WikidataClient(cache_dir=tmp_path, session=session)

    entity = client.get_entity("Q123")

    assert entity["id"] == "Q123"
    assert (tmp_path / "Q123.json").exists()


def test_cached_entity_avoids_http_call(tmp_path: Path) -> None:
    (tmp_path / "Q123.json").write_text(
        json.dumps(entity_payload("Q123")),
        encoding="utf-8",
    )
    session = FakeSession([])
    client = WikidataClient(cache_dir=tmp_path, session=session)

    entity = client.get_entity("Q123")

    assert entity["id"] == "Q123"
    assert session.calls == []


def test_cached_wikipedia_resolution_avoids_http_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "wikipedia_resolution_d8caa99243b5039c.json"
    cache_path.write_text(
        json.dumps(
            {
                "wikipedia_url": "https://en.wikipedia.org/wiki/Maine_Coon",
                "entity_id": "Q123",
            }
        ),
        encoding="utf-8",
    )
    session = FakeSession([])
    client = WikidataClient(cache_dir=tmp_path, session=session)

    entity_id = client.resolve_entity_id_from_wikipedia_url(
        "https://en.wikipedia.org/wiki/Maine_Coon"
    )

    assert entity_id == "Q123"
    assert session.calls == []


def test_override_has_priority(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(entity_payload("Q999", en_label="Override"))])
    client = WikidataClient(cache_dir=tmp_path, session=session)

    result = resolve_breed_record(
        registry_record(),
        client=client,
        overrides={"mcoo": {"wikidata_entity_id": "Q999", "reason": "verified"}},
    )

    assert result["entity_id"] == "Q999"
    assert result["match_method"] == "manual_override"
    assert len(session.calls) == 1
    assert "Special:EntityData/Q999.json" in session.calls[0]["url"]


def test_single_exact_label_is_confirmed(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse({"search": [{"id": "Q123", "label": "Maine Coon"}]}),
            FakeResponse(entity_payload("Q123")),
        ]
    )
    client = WikidataClient(cache_dir=tmp_path, session=session)

    result = resolve_breed_record(
        registry_record(wikipedia_url=None),
        client=client,
        overrides={},
    )

    assert result["entity_id"] == "Q123"
    assert result["match_method"] == "exact_en_label"


def test_multiple_exact_candidates_are_unresolved(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "search": [
                        {"id": "Q1", "label": "Maine Coon"},
                        {"id": "Q2", "label": "Maine Coon"},
                    ]
                }
            ),
        ]
    )
    client = WikidataClient(cache_dir=tmp_path, session=session)

    result = resolve_breed_record(
        registry_record(wikipedia_url=None, aliases_en=[]),
        client=client,
        overrides={},
    )

    assert result["entity_id"] is None
    assert result["match_method"] == "unresolved"
    assert "Multiple exact English label candidates found" in result["warnings"]


def test_entity_not_found_returns_unresolved(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse({"search": []})])
    client = WikidataClient(cache_dir=tmp_path, session=session)

    result = resolve_breed_record(
        registry_record(wikipedia_url=None, aliases_en=[]),
        client=client,
        overrides={},
    )

    assert result["entity_id"] is None
    assert result["match_method"] == "unresolved"
    assert "Wikidata entity could not be resolved" in result["warnings"]


def test_temporary_http_error_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.data.wikidata_client.time.sleep", lambda _: None)
    session = FakeSession(
        [
            FakeResponse({}, status_code=503, headers={"Retry-After": "0"}),
            FakeResponse(entity_payload("Q123")),
        ]
    )
    client = WikidataClient(cache_dir=tmp_path, session=session, retries=1)

    entity = client.get_entity("Q123")

    assert entity["id"] == "Q123"
    assert len(session.calls) == 2


def test_permanent_http_error_is_handled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.data.wikidata_client.time.sleep", lambda _: None)
    session = FakeSession([FakeResponse({}, status_code=404)])
    client = WikidataClient(cache_dir=tmp_path, session=session, retries=0)

    with pytest.raises(WikidataClientError):
        client.get_entity("Q404")


def test_registry_file_is_not_modified(tmp_path: Path) -> None:
    registry_path = tmp_path / "breed_registry.jsonl"
    registry_path.write_text(
        json.dumps(registry_record("mcoo"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = hashlib.sha256(registry_path.read_bytes()).hexdigest()

    records = load_registry(registry_path, {"mcoo"})

    after = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    assert before == after
    assert len(records) == 1


def test_enrichment_records_are_sorted_by_breed_id(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse({"search": [{"id": "Q2", "label": "Sphynx"}]}),
            FakeResponse(entity_payload("Q2", en_label="Sphynx")),
            FakeResponse({"search": [{"id": "Q1", "label": "Bengal"}]}),
            FakeResponse(entity_payload("Q1", en_label="Bengal")),
        ]
    )
    client = WikidataClient(cache_dir=tmp_path, session=session)
    records = [
        registry_record("sphy", "Sphynx", wikipedia_url=None),
        registry_record("beng", "Bengal", wikipedia_url=None),
    ]

    enrichment = resolve_registry_records(records, client=client, overrides={})

    assert [record["breed_id"] for record in enrichment] == ["beng", "sphy"]


def test_repeated_write_is_byte_identical(tmp_path: Path) -> None:
    records = [
        build_enrichment_record(
            registry_record("mcoo"),
            entity_id="Q123",
            entity=entity_payload("Q123")["entities"]["Q123"],
            match_method="catapi_wikipedia_sitelink",
            match_confidence=1.0,
            warnings=[],
        )
    ]
    output_path = tmp_path / "wikidata_enrichment.jsonl"

    write_jsonl_atomic(records, output_path)
    first_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    write_jsonl_atomic(records, output_path)
    second_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    assert first_hash == second_hash
