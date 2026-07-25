from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.data.chunking import (
    BROADER_SKIP_REASON,
    HARD_SPLIT_WARNING,
    ChunkingError,
    build_chunks,
    build_chunks_report,
    load_broader_overrides,
    split_text,
    write_jsonl_atomic,
)


def catapi_document() -> dict:
    return {
        "document_id": "mcoo:catapi:profile",
        "breed_id": "mcoo",
        "breed_name_en": "Maine Coon",
        "breed_name_ru": "Мейн-кун",
        "aliases": ["Maine cat"],
        "language": "en",
        "source": "thecatapi",
        "document_type": "structured_profile",
        "text": "Breed: Maine Coon\nOrigin: United States",
        "provenance": {"source_name": "TheCatAPI"},
        "warnings": [],
    }


def wikipedia_document(relation: str = "standalone_article") -> dict:
    return {
        "document_id": "mcoo:wikipedia:en",
        "breed_id": "mcoo",
        "breed_name_en": "Maine Coon",
        "breed_name_ru": "Мейн-кун",
        "aliases": [],
        "language": "en",
        "source": "wikipedia",
        "document_type": "wikipedia_article",
        "lead": "Overview lead.",
        "sections": [
            {
                "index": "1",
                "level": 2,
                "parent_index": None,
                "section_path": ["Health"],
                "title": "Health",
                "text": "Short section.",
            },
            {
                "index": "2.1",
                "level": 3,
                "parent_index": "2",
                "section_path": ["Description", "Coat"],
                "title": "Coat",
                "text": "Life expectancy: 12-15 years.",
            },
            {
                "index": "3",
                "level": 2,
                "parent_index": None,
                "section_path": ["Empty"],
                "title": "Empty",
                "text": "   ",
            },
        ],
        "provenance": {
            "source_url": "https://en.wikipedia.org/wiki/Maine_Coon",
            "page_id": 1,
            "revision_id": 2,
            "source_resolution": {
                "method": "wikidata_sitelink",
                "source_relation": relation,
                "reason": None,
            },
        },
        "warnings": [],
    }


def test_catapi_document_creates_one_chunk() -> None:
    chunks, skipped = build_chunks([catapi_document()])

    assert skipped == []
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "structured_profile"
    assert chunks[0]["chunk_id"] == "mcoo:catapi:profile:000"


def test_wikipedia_lead_and_sections_create_chunks() -> None:
    chunks, _ = build_chunks([wikipedia_document()])

    assert [chunk["chunk_type"] for chunk in chunks] == ["overview", "section", "section"]
    assert chunks[0]["section_title"] == "Overview"


def test_empty_lead_does_not_create_overview_chunk() -> None:
    document = wikipedia_document()
    document["lead"] = ""
    chunks, _ = build_chunks([document])

    assert all(chunk["chunk_type"] != "overview" for chunk in chunks)


def test_short_section_is_not_removed_and_empty_section_is_removed() -> None:
    chunks, _ = build_chunks([wikipedia_document()])
    titles = [chunk["section_title"] for chunk in chunks]

    assert "Coat" in titles
    assert "Empty" not in titles


def test_section_path_and_embedding_header_are_preserved() -> None:
    chunks, _ = build_chunks([wikipedia_document()])
    coat = next(chunk for chunk in chunks if chunk["section_title"] == "Coat")

    assert coat["section_path"] == ["Description", "Coat"]
    assert "Section: Description > Coat" in coat["embedding_text"]
    assert not coat["text"].startswith("Breed:")


def test_long_section_is_split_by_paragraphs_without_overlap() -> None:
    first = "A" * 1500
    second = "B" * 1500
    parts, warnings = split_text(f"{first}\n\n{second}", target_chars=1400, max_chars=2000)

    assert warnings == []
    assert parts == [first, second]


def test_long_paragraph_is_split_by_sentences() -> None:
    text = "First sentence. " * 120 + "Second sentence. " * 120
    parts, warnings = split_text(text, target_chars=500, max_chars=1000)

    assert warnings == []
    assert len(parts) > 1
    assert all(len(part) <= 1000 for part in parts)


def test_one_sentence_over_hard_max_is_hard_split_with_warning() -> None:
    parts, warnings = split_text("x" * 3100, target_chars=2200, max_chars=3000)

    assert [len(part) for part in parts] == [3000, 100]
    assert warnings == [HARD_SPLIT_WARNING]


def test_long_section_parts_get_sequential_part_indexes() -> None:
    document = wikipedia_document()
    document["sections"][0]["text"] = "A" * 1500 + "\n\n" + "B" * 1500
    chunks, _ = build_chunks([document], target_chars=1400, max_chars=2000)
    health_chunks = [chunk for chunk in chunks if chunk["section_title"] == "Health"]

    assert [chunk["part_index"] for chunk in health_chunks] == [0, 1]
    assert len({chunk["chunk_id"] for chunk in health_chunks}) == 2


def test_chunk_ids_are_deterministic_and_unique() -> None:
    first, _ = build_chunks([wikipedia_document()])
    second, _ = build_chunks([wikipedia_document()])

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)


def test_no_normal_chunk_exceeds_max_chars() -> None:
    document = wikipedia_document()
    document["sections"][0]["text"] = "A" * 3001
    chunks, _ = build_chunks([document], target_chars=2200, max_chars=3000)

    assert all(chunk["text_length"] <= 3000 for chunk in chunks)


def test_broader_article_is_skipped() -> None:
    chunks, skipped = build_chunks([wikipedia_document("covered_by_broader_article")])

    assert chunks == []
    assert skipped[0]["reason"] == BROADER_SKIP_REASON


def test_section_of_another_article_requires_override() -> None:
    with pytest.raises(ChunkingError, match="requires override"):
        build_chunks([wikipedia_document("section_of_another_article")])


def test_scoped_article_creates_only_approved_section_and_no_lead() -> None:
    chunks, skipped = build_chunks(
        [wikipedia_document()],
        broader_overrides={
            "mcoo:en": {
                "breed_id": "mcoo",
                "language": "en",
                "source_relation": "section_of_another_article",
                "include_lead": False,
                "included_section_paths": [["Description", "Coat"]],
                "reason": "Only Coat is approved.",
            }
        },
    )

    assert skipped == []
    assert [chunk["section_path"] for chunk in chunks] == [["Description", "Coat"]]
    assert chunks[0]["source_relation"] == "section_of_another_article"
    assert chunks[0]["selection_method"] == "manual_section_approval"


def test_missing_approved_section_path_fails() -> None:
    with pytest.raises(ChunkingError, match="not found"):
        build_chunks(
            [wikipedia_document()],
            broader_overrides={
                "mcoo:en": {
                    "breed_id": "mcoo",
                    "language": "en",
                    "source_relation": "section_of_another_article",
                    "include_lead": False,
                    "included_section_paths": [["Missing"]],
                    "reason": "Missing path.",
                }
            },
        )


def test_service_section_paths_are_removed_in_chunking() -> None:
    document = wikipedia_document()
    document["sections"].append(
        {
            "index": "4.1",
            "level": 3,
            "parent_index": "4",
            "section_path": ["References", "Literature"],
            "title": "Literature",
            "text": "Reference text.",
        }
    )

    chunks, _ = build_chunks([document])

    assert ["References", "Literature"] not in [chunk["section_path"] for chunk in chunks]


def test_compact_russian_see_also_path_is_removed() -> None:
    document = wikipedia_document()
    document["sections"].append(
        {
            "index": "4",
            "level": 2,
            "parent_index": None,
            "section_path": ["См.также"],
            "title": "См.также",
            "text": "Other breeds.",
        }
    )

    chunks, _ = build_chunks([document])

    assert ["См.также"] not in [chunk["section_path"] for chunk in chunks]


def test_breed_gallery_and_source_sections_are_removed() -> None:
    document = wikipedia_document()
    document["sections"].extend(
        [
            {
                "index": "4",
                "level": 2,
                "parent_index": None,
                "section_path": ["Breed gallery"],
                "title": "Breed gallery",
                "text": "Caption.",
            },
            {
                "index": "5",
                "level": 2,
                "parent_index": None,
                "section_path": ["Источник"],
                "title": "Источник",
                "text": "Source text.",
            },
        ]
    )

    chunks, _ = build_chunks([document])
    paths = [chunk["section_path"] for chunk in chunks]

    assert ["Breed gallery"] not in paths
    assert ["Источник"] not in paths


def test_standalone_burmese_like_shared_page_is_chunked() -> None:
    doc = wikipedia_document("standalone_article")
    doc["breed_id"] = "bure"
    doc["document_id"] = "bure:wikipedia:en"
    chunks, skipped = build_chunks([doc])

    assert skipped == []
    assert chunks


def test_same_page_used_for_multiple_breeds_with_override_does_not_error() -> None:
    first = wikipedia_document()
    second = wikipedia_document()
    second["breed_id"] = "bure"
    second["document_id"] = "bure:wikipedia:en"
    second["provenance"]["page_id"] = first["provenance"]["page_id"]

    chunks, skipped = build_chunks(
        [first, second],
        broader_overrides={
            "bure:en": {
                "breed_id": "bure",
                "language": "en",
                "source_relation": "section_of_another_article",
                "include_lead": False,
                "included_section_paths": [["Health"]],
                "reason": "Only Health is approved for this test.",
            }
        },
    )

    assert skipped == []
    assert {chunk["breed_id"] for chunk in chunks} == {"mcoo", "bure"}


def test_empty_broader_overrides_object_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "broader_source_chunk_overrides.json"
    path.write_text("{}\n", encoding="utf-8")

    assert load_broader_overrides(path) == {}


def test_shared_standalone_source_without_override_fails() -> None:
    first = wikipedia_document()
    second = wikipedia_document()
    second["breed_id"] = "bure"
    second["document_id"] = "bure:wikipedia:en"
    second["provenance"]["page_id"] = first["provenance"]["page_id"]

    with pytest.raises(ChunkingError, match="standalone article source"):
        build_chunks([first, second])


def test_two_chunk_writes_have_same_sha(tmp_path: Path) -> None:
    chunks, _ = build_chunks([catapi_document(), wikipedia_document()])
    path = tmp_path / "knowledge_chunks.jsonl"

    write_jsonl_atomic(chunks, path)
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    write_jsonl_atomic(chunks, path)
    second = hashlib.sha256(path.read_bytes()).hexdigest()

    assert first == second


def test_report_contains_expected_diagnostics() -> None:
    chunks, skipped = build_chunks([catapi_document(), wikipedia_document()])
    report = build_chunks_report(chunks, skipped)

    assert report["empty_chunks"] == 0
    assert report["chunks_longer_than_3000_characters"] == 0
    assert report["duplicate_chunk_ids"] == []
