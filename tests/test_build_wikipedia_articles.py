from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_wikipedia_articles import (
    build_articles,
    read_jsonl,
    write_jsonl_atomic,
)
from src.data.wikipedia_client import WikipediaClientError


class FakeClient:
    def __init__(self, responses: dict[tuple[str, str], dict | Exception]) -> None:
        self.responses = responses
        self.calls = []

    def fetch_article(self, breed_id: str, language: str, title: str) -> dict:
        self.calls.append((breed_id, language, title))
        response = self.responses[(breed_id, language)]
        if isinstance(response, Exception):
            raise response
        return response


def enrichment_record(
    breed_id: str = "mcoo",
    enwiki: str | None = "Maine Coon",
    ruwiki: str | None = "Мейн-кун",
) -> dict:
    return {
        "breed_id": breed_id,
        "name_en": "Maine Coon",
        "sitelinks": {
            "enwiki": enwiki,
            "ruwiki": ruwiki,
        },
    }


def cached_response(title: str = "Maine Coon", page_id: int = 1) -> dict:
    return {
        "retrieved_at": "2026-07-23T00:00:00Z",
        "requested_language": "en",
        "requested_title": title,
        "api_response": {
            "parse": {
                "title": title,
                "pageid": page_id,
                "revid": page_id + 100,
                "text": "<p>Lead.</p><h2>History</h2><p>Text.</p>",
                "sections": [],
            }
        },
    }


def test_missing_sitelink_goes_to_unresolved() -> None:
    articles, unresolved = build_articles(
        [enrichment_record(ruwiki=None)],
        client=FakeClient({("mcoo", "en"): cached_response()}),
        breed_ids={"mcoo"},
        languages={"ru"},
    )

    assert articles == []
    assert unresolved[0]["reason"] == "missing_sitelink"


def test_missing_page_goes_to_unresolved() -> None:
    articles, unresolved = build_articles(
        [enrichment_record()],
        client=FakeClient({("mcoo", "en"): WikipediaClientError("HTTP 404")}),
        breed_ids={"mcoo"},
        languages={"en"},
    )

    assert articles == []
    assert unresolved[0]["reason"] == "http_error"


def test_one_page_error_does_not_break_pipeline() -> None:
    articles, unresolved = build_articles(
        [enrichment_record("mcoo"), enrichment_record("bsho", "British Shorthair")],
        client=FakeClient(
            {
                ("bsho", "en"): cached_response("British Shorthair", 2),
                ("mcoo", "en"): WikipediaClientError("HTTP 500"),
            }
        ),
        breed_ids={"mcoo", "bsho"},
        languages={"en"},
    )

    assert [article["breed_id"] for article in articles] == ["bsho"]
    assert [item["breed_id"] for item in unresolved] == ["mcoo"]


def test_records_are_sorted_by_breed_id_and_language() -> None:
    articles, _ = build_articles(
        [enrichment_record("mcoo"), enrichment_record("beng", "Bengal", "Бенгальская")],
        client=FakeClient(
            {
                ("mcoo", "en"): cached_response("Maine Coon", 3),
                ("mcoo", "ru"): cached_response("Мейн-кун", 4),
                ("beng", "en"): cached_response("Bengal cat", 1),
                ("beng", "ru"): cached_response("Бенгальская кошка", 2),
            }
        ),
        breed_ids={"mcoo", "beng"},
        languages={"ru", "en"},
    )

    assert [(item["breed_id"], item["language"]) for item in articles] == [
        ("beng", "en"),
        ("beng", "ru"),
        ("mcoo", "en"),
        ("mcoo", "ru"),
    ]


def test_two_writes_are_byte_identical(tmp_path: Path) -> None:
    articles, _ = build_articles(
        [enrichment_record()],
        client=FakeClient({("mcoo", "en"): cached_response()}),
        breed_ids={"mcoo"},
        languages={"en"},
    )
    output_path = tmp_path / "wikipedia_articles.jsonl"

    write_jsonl_atomic(articles, output_path)
    first_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    write_jsonl_atomic(articles, output_path)
    second_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    assert first_hash == second_hash


def test_input_files_are_not_modified(tmp_path: Path) -> None:
    registry_path = tmp_path / "breed_registry.jsonl"
    enrichment_path = tmp_path / "wikidata_enrichment.jsonl"
    registry_path.write_text("registry\n", encoding="utf-8")
    enrichment_path.write_text(json.dumps(enrichment_record()) + "\n", encoding="utf-8")
    before_registry = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    before_enrichment = hashlib.sha256(enrichment_path.read_bytes()).hexdigest()

    records = read_jsonl(enrichment_path)
    build_articles(
        records,
        client=FakeClient({("mcoo", "en"): cached_response()}),
        breed_ids={"mcoo"},
        languages={"en"},
    )

    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == before_registry
    assert hashlib.sha256(enrichment_path.read_bytes()).hexdigest() == before_enrichment
