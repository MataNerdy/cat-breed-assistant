from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_wikipedia_articles import (
    BROADER_SOURCE_WARNING,
    build_articles,
    load_source_overrides,
    read_jsonl,
    validate_source_overrides,
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


def missing_page_response(title: str = "Missing page") -> dict:
    return {
        "retrieved_at": "2026-07-23T00:00:00Z",
        "requested_language": "en",
        "requested_title": title,
        "api_response": {
            "error": {
                "code": "missingtitle",
                "info": "The page you specified doesn't exist.",
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


def test_missing_enwiki_goes_to_unresolved() -> None:
    articles, unresolved = build_articles(
        [enrichment_record(enwiki=None)],
        client=FakeClient({("mcoo", "ru"): cached_response("Мейн-кун")}),
        breed_ids={"mcoo"},
        languages={"en"},
    )

    assert articles == []
    assert unresolved[0]["language"] == "en"
    assert unresolved[0]["reason"] == "missing_sitelink"


def test_manual_override_has_priority_over_missing_sitelink() -> None:
    articles, unresolved = build_articles(
        [enrichment_record("ebur", enwiki=None, ruwiki=None)],
        client=FakeClient({("ebur", "en"): cached_response("Burmese cat")}),
        breed_ids={"ebur"},
        languages={"en"},
        source_overrides={
            "ebur:en": {
                "title": "Burmese cat",
                "source_relation": "covered_by_broader_article",
                "reason": "European Burmese is covered by Burmese cat.",
            }
        },
    )

    assert unresolved == []
    assert articles[0]["breed_id"] == "ebur"
    assert articles[0]["title"] == "Burmese cat"
    assert articles[0]["source_resolution"] == {
        "method": "manual_override",
        "source_relation": "covered_by_broader_article",
        "reason": "European Burmese is covered by Burmese cat.",
    }


def test_manual_override_has_priority_over_existing_sitelink() -> None:
    articles, unresolved = build_articles(
        [enrichment_record("bamb", enwiki="Wrong", ruwiki="Wrong")],
        client=FakeClient({("bamb", "ru"): cached_response("Бамбино (порода кошек)")}),
        breed_ids={"bamb"},
        languages={"ru"},
        source_overrides={
            "bamb:ru": {
                "title": "Бамбино (порода кошек)",
                "source_relation": "standalone_article",
                "reason": "Verified article.",
            }
        },
    )

    assert unresolved == []
    assert articles[0]["title"] == "Бамбино (порода кошек)"
    assert articles[0]["source_resolution"]["method"] == "manual_override"
    assert articles[0]["source_resolution"]["source_relation"] == "standalone_article"
    assert articles[0]["warnings"] == []
    assert articles[0]["source_resolution"]["reason"] == "Verified article."


def test_covered_by_broader_article_gets_warning() -> None:
    articles, unresolved = build_articles(
        [enrichment_record("ebur", enwiki=None)],
        client=FakeClient({("ebur", "en"): cached_response("Burmese cat")}),
        breed_ids={"ebur"},
        languages={"en"},
        source_overrides={
            "ebur:en": {
                "title": "Burmese cat",
                "source_relation": "covered_by_broader_article",
                "reason": "Covered by broader article.",
            }
        },
    )

    assert unresolved == []
    assert articles[0]["source_resolution"]["source_relation"] == (
        "covered_by_broader_article"
    )
    assert BROADER_SOURCE_WARNING in articles[0]["warnings"]


def test_wikidata_sitelink_source_resolution_is_recorded() -> None:
    articles, unresolved = build_articles(
        [enrichment_record()],
        client=FakeClient({("mcoo", "en"): cached_response()}),
        breed_ids={"mcoo"},
        languages={"en"},
    )

    assert unresolved == []
    assert articles[0]["source_resolution"] == {
        "method": "wikidata_sitelink",
        "source_relation": "standalone_article",
        "reason": None,
    }


def test_absent_source_overrides_file_is_allowed(tmp_path: Path) -> None:
    assert load_source_overrides(tmp_path / "missing.json") == {}


def test_load_source_overrides_rejects_missing_title(tmp_path: Path) -> None:
    path = tmp_path / "wikipedia_source_overrides.json"
    path.write_text(
        json.dumps({"ebur:en": {"source_relation": "covered_by_broader_article"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty title"):
        validate_source_overrides(
            load_source_overrides(path),
            [enrichment_record("ebur")],
        )


def test_invalid_override_key_raises_error() -> None:
    with pytest.raises(ValueError, match="Invalid source override key"):
        validate_source_overrides(
            {
                "ebur-en": {
                    "title": "Burmese cat",
                    "source_relation": "standalone_article",
                    "reason": "x",
                }
            },
            [enrichment_record("ebur")],
        )


def test_unknown_override_breed_id_raises_error() -> None:
    with pytest.raises(ValueError, match="Unknown breed_id"):
        validate_source_overrides(
            {
                "xxxx:en": {
                    "title": "Burmese cat",
                    "source_relation": "standalone_article",
                    "reason": "x",
                }
            },
            [enrichment_record("ebur")],
        )


def test_unsupported_override_language_raises_error() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        validate_source_overrides(
            {
                "ebur:de": {
                    "title": "Burmese cat",
                    "source_relation": "standalone_article",
                    "reason": "x",
                }
            },
            [enrichment_record("ebur")],
        )


def test_unknown_source_relation_raises_error() -> None:
    with pytest.raises(ValueError, match="unsupported source_relation"):
        validate_source_overrides(
            {
                "ebur:en": {
                    "title": "Burmese cat",
                    "source_relation": "maybe",
                    "reason": "x",
                }
            },
            [enrichment_record("ebur")],
        )


def test_empty_reason_raises_error() -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        validate_source_overrides(
            {
                "ebur:en": {
                    "title": "Burmese cat",
                    "source_relation": "standalone_article",
                    "reason": " ",
                }
            },
            [enrichment_record("ebur")],
        )


def test_wrong_verified_url_language_raises_error() -> None:
    with pytest.raises(ValueError, match="verified_url"):
        validate_source_overrides(
            {
                "ebur:ru": {
                    "title": "Бурма (порода кошек)",
                    "source_relation": "covered_by_broader_article",
                    "reason": "x",
                    "verified_url": "https://en.wikipedia.org/wiki/Burmese_cat",
                }
            },
            [enrichment_record("ebur")],
        )


def test_override_missing_page_gets_specific_unresolved_reason() -> None:
    articles, unresolved = build_articles(
        [enrichment_record("ebur", enwiki=None)],
        client=FakeClient({("ebur", "en"): missing_page_response("Missing")}),
        breed_ids={"ebur"},
        languages={"en"},
        source_overrides={
            "ebur:en": {
                "title": "Missing",
                "source_relation": "covered_by_broader_article",
                "reason": "Manual check.",
            }
        },
    )

    assert articles == []
    assert unresolved[0]["reason"] == "override_page_not_found"


def test_same_page_can_be_used_for_multiple_breed_ids() -> None:
    articles, unresolved = build_articles(
        [
            enrichment_record("bure", enwiki="Burmese cat"),
            enrichment_record("ebur", enwiki=None),
        ],
        client=FakeClient(
            {
                ("bure", "en"): cached_response("Burmese cat", page_id=10),
                ("ebur", "en"): cached_response("Burmese cat", page_id=10),
            }
        ),
        breed_ids={"bure", "ebur"},
        languages={"en"},
        source_overrides={
            "ebur:en": {
                "title": "Burmese cat",
                "source_relation": "covered_by_broader_article",
                "reason": "Covered by shared article.",
            }
        },
    )

    assert unresolved == []
    assert [(article["breed_id"], article["page_id"]) for article in articles] == [
        ("bure", 10),
        ("ebur", 10),
    ]
    assert articles[0]["source_resolution"]["method"] == "wikidata_sitelink"
    assert articles[1]["source_resolution"]["method"] == "manual_override"


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
