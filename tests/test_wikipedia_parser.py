from __future__ import annotations

import pytest

from src.data.wikipedia_parser import (
    clean_text,
    extract_page_metadata,
    parse_article_record,
)


def cached_response(html_text: str) -> dict:
    return {
        "retrieved_at": "2026-07-23T00:00:00Z",
        "requested_language": "en",
        "requested_title": "Maine Coon",
        "api_response": {
            "parse": {
                "title": "Maine Coon",
                "pageid": 123,
                "revid": 456,
                "text": html_text,
                "sections": [],
            }
        },
    }


def test_page_id_and_revision_id_are_extracted() -> None:
    metadata = extract_page_metadata(cached_response("<p>Lead.</p>"))

    assert metadata["page_id"] == 123
    assert metadata["revision_id"] == 456


def test_lead_is_extracted() -> None:
    article = parse_article_record(
        cached_response("<p>Maine Coon is a large cat.</p><h2>History</h2><p>Old.</p>"),
        breed_id="mcoo",
        language="en",
    )

    assert article["lead"] == "Maine Coon is a large cat."


def test_content_sections_are_preserved() -> None:
    article = parse_article_record(
        cached_response("<p>Lead.</p><h2>History</h2><p>Started.</p>"),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"] == [
        {"index": 1, "title": "History", "text": "Started."}
    ]


def test_service_sections_are_excluded() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>History</h2><p>Started.</p>"
            "<h2>References</h2><p>Ref.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert [section["title"] for section in article["sections"]] == ["History"]


def test_russian_service_sections_are_excluded() -> None:
    article = parse_article_record(
        {
            **cached_response(
                "<p>Лид.</p><h2>История</h2><p>Текст.</p>"
                "<h2>Примечания</h2><p>Сноска.</p>"
            ),
            "requested_language": "ru",
        },
        breed_id="mcoo",
        language="ru",
    )

    assert [section["title"] for section in article["sections"]] == ["История"]


def test_html_and_markup_are_cleaned() -> None:
    assert clean_text("  A&nbsp;cat <ignored> [1] \n with   spaces ") == (
        "A cat <ignored> with spaces"
    )


def test_reference_tail_markup_is_cleaned() -> None:
    assert clean_text("Вариант CFA 1 ]") == "Вариант CFA"
    assert clean_text("нормальный вес котят составляет 100 граммов ]") == (
        "нормальный вес котят составляет 100 граммов"
    )


def test_parser_requires_article_text() -> None:
    with pytest.raises(ValueError, match="text is missing"):
        parse_article_record(cached_response(""), breed_id="mcoo", language="en")
