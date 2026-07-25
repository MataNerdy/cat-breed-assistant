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


def test_navbox_inside_lead_is_removed_before_text_extraction() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Полезное описание породы.</p>"
            "<table class='navbox'><tr><td><p>Персидская (PER)</p></td></tr></table>"
            "<h2>История</h2><p>Текст.</p>"
        ),
        breed_id="hbro",
        language="ru",
    )

    assert article["lead"] == "Полезное описание породы."
    assert "Персидская" not in article["lead"]


def test_content_sections_are_preserved() -> None:
    article = parse_article_record(
        cached_response("<p>Lead.</p><h2>History</h2><p>Started.</p>"),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"] == [
        {
            "index": "1",
            "level": 2,
            "title": "History",
            "parent_index": None,
            "section_path": ["History"],
            "text": "Started.",
        }
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


def test_empty_parent_section_is_not_saved() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Health</h2>"
            "<h3>Heart disease</h3><p>Useful text.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert [section["title"] for section in article["sections"]] == ["Heart disease"]


def test_empty_parent_title_is_kept_in_child_section_path() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Health</h2>"
            "<h3>Heart disease</h3><p>Useful text.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"][0]["section_path"] == ["Health", "Heart disease"]


def test_nested_section_level_and_parent_index_are_saved() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Health</h2>"
            "<h3>Heart disease</h3><p>Useful text.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"][0]["index"] == "1.1"
    assert article["sections"][0]["level"] == 3
    assert article["sections"][0]["parent_index"] == "1"


def test_flat_section_has_section_path() -> None:
    article = parse_article_record(
        cached_response("<p>Lead.</p><h2>History</h2><p>Text.</p>"),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"][0]["section_path"] == ["History"]
    assert article["sections"][0]["parent_index"] is None


def test_gallery_section_is_excluded() -> None:
    article = parse_article_record(
        cached_response("<p>Lead.</p><h2>Gallery</h2><p>Caption.</p>"),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"] == []


def test_russian_photo_section_is_excluded() -> None:
    article = parse_article_record(
        {
            **cached_response("<p>Лид.</p><h2>Фотографии</h2><p>Подпись.</p>"),
            "requested_language": "ru",
        },
        breed_id="mcoo",
        language="ru",
    )

    assert article["sections"] == []


def test_content_section_with_image_is_not_removed() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Appearance</h2>"
            "<figure><figcaption>Cat photo</figcaption></figure>"
            "<p>Body is strong.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"][0]["title"] == "Appearance"
    assert article["sections"][0]["text"] == "Body is strong."


def test_child_of_references_is_excluded() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>References</h2>"
            "<h3>Literature</h3><p>Book.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"] == []


def test_navbox_is_excluded_but_regular_paragraph_remains() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>History</h2>"
            "<div class='navbox'><p>Navigation box breeds.</p></div>"
            "<p>Useful paragraph.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"][0]["text"] == "Useful paragraph."


def test_authority_control_is_excluded() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>History</h2>"
            "<div class='authority-control'><p>Authority data.</p></div>"
            "<p>Useful paragraph.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert "Authority data" not in article["sections"][0]["text"]


def test_nested_navigation_markup_does_not_leak_after_first_child_endtag() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Temperament</h2>"
            "<p>Russian Blue is calm.</p>"
            "<table class='metadata navbox'><tr><td><p>Persian (PER)</p></td></tr></table>"
            "<p>Useful ending.</p>"
        ),
        breed_id="rblu",
        language="en",
    )

    assert article["sections"][0]["text"] == "Russian Blue is calm.\n\nUseful ending."


def test_role_navigation_is_excluded_but_content_remains() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Temperament</h2>"
            "<p>Russian Blue is calm.</p>"
            "<div role='navigation'><p>Britannica J9U LCCN</p></div>"
        ),
        breed_id="rblu",
        language="en",
    )

    assert article["sections"][0]["text"] == "Russian Blue is calm."


def test_gallery_captions_are_excluded() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>History</h2>"
            "<ul class='gallery'><li>Caption only.</li></ul>"
            "<p>Useful paragraph.</p>"
        ),
        breed_id="mcoo",
        language="en",
    )

    assert article["sections"][0]["text"] == "Useful paragraph."


def test_gallery_caption_only_section_is_not_created() -> None:
    article = parse_article_record(
        cached_response(
            "<p>Lead.</p><h2>Coat colour overview</h2>"
            "<ul class='gallery mw-gallery-traditional'>"
            "<li>Ruddy female</li><li>Blue kitten</li><li>Sorrel</li>"
            "</ul>"
        ),
        breed_id="soma",
        language="en",
    )

    assert article["sections"] == []


def test_marker_only_text_is_not_saved_as_section() -> None:
    article = parse_article_record(
        cached_response("<p>Lead.</p><h2>Characteristics</h2><p>Source:</p>"),
        breed_id="dons",
        language="en",
    )

    assert article["sections"] == []


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
