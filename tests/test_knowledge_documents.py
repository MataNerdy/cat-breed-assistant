from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.data.knowledge_documents import (
    KnowledgeDocumentError,
    build_knowledge_documents,
    build_knowledge_documents_report,
    write_jsonl_atomic,
)


def registry_record(breed_id: str = "mcoo", name: str = "Maine Coon") -> dict:
    return {
        "breed_id": breed_id,
        "name_en": name,
        "name_ru": None,
        "aliases_en": ["Maine cat"],
        "aliases_ru": [],
        "catapi": {
            "raw": {
                "id": breed_id,
                "name": name,
                "alt_names": "",
                "temperament": "Affectionate, Gentle",
                "origin": "United States",
                "description": "Large gentle cat.",
                "life_span": "12 - 15",
                "weight": {"metric": "5 - 11", "imperial": "11 - 24"},
                "affection_level": 5,
                "energy_level": 3,
                "grooming": None,
                "hairless": 0,
            }
        },
    }


def wikidata_record(breed_id: str = "mcoo") -> dict:
    return {
        "breed_id": breed_id,
        "labels": {"en": "Maine Coon", "ru": "Мейн-кун"},
        "aliases": {"en": ["Maine cat"], "ru": ["мейн кун"]},
    }


def wikipedia_record(
    breed_id: str = "mcoo",
    language: str = "en",
    page_id: int = 1,
    relation: str = "standalone_article",
) -> dict:
    return {
        "breed_id": breed_id,
        "language": language,
        "title": "Maine Coon" if language == "en" else "Мейн-кун",
        "lead": "Lead text.",
        "sections": [
            {
                "index": "1",
                "level": 2,
                "parent_index": None,
                "section_path": ["History"],
                "title": "History",
                "text": "Section text.",
            }
        ],
        "source_url": f"https://{language}.wikipedia.org/wiki/Maine_Coon",
        "page_id": page_id,
        "revision_id": page_id + 100,
        "retrieved_at": "2026-07-24T00:00:00Z",
        "source_resolution": {
            "method": "wikidata_sitelink",
            "source_relation": relation,
            "reason": None,
        },
        "wiki_project": f"{language}wiki",
        "warnings": [],
    }


def test_catapi_document_is_created_per_breed() -> None:
    documents = build_knowledge_documents(
        [registry_record("mcoo"), registry_record("bsho", "British Shorthair")],
        [wikidata_record("mcoo"), wikidata_record("bsho")],
        [],
    )

    assert [doc["document_id"] for doc in documents] == [
        "bsho:catapi:profile",
        "mcoo:catapi:profile",
    ]


def test_catapi_raw_json_is_not_copied_and_empty_fields_are_omitted() -> None:
    document = build_knowledge_documents(
        [registry_record()],
        [wikidata_record()],
        [],
    )[0]

    assert "catapi" not in document
    assert "raw" not in document
    assert "Grooming" not in document["text"]


def test_numeric_ratings_are_formatted() -> None:
    document = build_knowledge_documents(
        [registry_record()],
        [wikidata_record()],
        [],
    )[0]

    assert "Affection level: 5/5" in document["text"]
    assert "Energy level: 3/5" in document["text"]


def test_wikipedia_document_preserves_lead_sections_and_provenance() -> None:
    document = [
        doc
        for doc in build_knowledge_documents(
            [registry_record()],
            [wikidata_record()],
            [wikipedia_record()],
        )
        if doc["source"] == "wikipedia"
    ][0]

    assert document["lead"] == "Lead text."
    assert document["sections"][0]["section_path"] == ["History"]
    assert document["provenance"]["page_id"] == 1


def test_ru_and_en_wikipedia_documents_are_separate() -> None:
    documents = build_knowledge_documents(
        [registry_record()],
        [wikidata_record()],
        [wikipedia_record(language="ru"), wikipedia_record(language="en")],
    )

    assert {doc["document_id"] for doc in documents if doc["source"] == "wikipedia"} == {
        "mcoo:wikipedia:en",
        "mcoo:wikipedia:ru",
    }


def test_same_page_id_for_different_breeds_is_allowed() -> None:
    documents = build_knowledge_documents(
        [registry_record("bure", "Burmese"), registry_record("ebur", "European Burmese")],
        [wikidata_record("bure"), wikidata_record("ebur")],
        [wikipedia_record("bure", page_id=10), wikipedia_record("ebur", page_id=10)],
    )

    assert len([doc for doc in documents if doc["source"] == "wikipedia"]) == 2


def test_covered_by_broader_article_is_preserved() -> None:
    document = [
        doc
        for doc in build_knowledge_documents(
            [registry_record("ebur", "European Burmese")],
            [wikidata_record("ebur")],
            [wikipedia_record("ebur", relation="covered_by_broader_article")],
        )
        if doc["source"] == "wikipedia"
    ][0]

    assert document["provenance"]["source_resolution"]["source_relation"] == (
        "covered_by_broader_article"
    )


def test_scope_override_updates_wikipedia_source_relation() -> None:
    document = [
        doc
        for doc in build_knowledge_documents(
            [registry_record("bamb", "Bambino")],
            [wikidata_record("bamb")],
            [wikipedia_record("bamb")],
            scope_overrides={
                "bamb:en": {
                    "breed_id": "bamb",
                    "language": "en",
                    "source_relation": "section_of_another_article",
                    "include_lead": False,
                    "included_section_paths": [["Bambino"]],
                    "reason": "Only Bambino section is approved.",
                }
            },
        )
        if doc["source"] == "wikipedia"
    ][0]

    assert document["provenance"]["source_resolution"] == {
        "method": "manual_section_approval",
        "source_relation": "section_of_another_article",
        "reason": "Only Bambino section is approved.",
        "wiki_project": "enwiki",
    }


def test_name_override_has_priority_over_wikidata_label() -> None:
    document = build_knowledge_documents(
        [registry_record("bamb", "Bambino")],
        [wikidata_record("bamb")],
        [],
        name_overrides={"bamb": {"ru": "Бамбино"}},
    )[0]

    assert document["breed_name_ru"] == "Бамбино"


def test_wikipedia_document_preserves_wiki_project() -> None:
    record = wikipedia_record("chee", language="en")
    record["wiki_project"] = "simplewiki"
    record["source_url"] = "https://simple.wikipedia.org/wiki/Cheetoh"
    record["source_resolution"] = {
        "method": "manual_override",
        "source_relation": "standalone_article",
        "wiki_project": "simplewiki",
        "reason": "Verified Simple English article.",
    }
    document = [
        doc
        for doc in build_knowledge_documents(
            [registry_record("chee", "Cheetoh")],
            [wikidata_record("chee")],
            [record],
        )
        if doc["source"] == "wikipedia"
    ][0]

    assert document["language"] == "en"
    assert document["provenance"]["wiki_project"] == "simplewiki"
    assert document["provenance"]["source_resolution"]["wiki_project"] == "simplewiki"


def test_catapi_text_omits_mismatched_wikipedia_url() -> None:
    registry = registry_record("chee", "Cheetoh")
    registry["catapi"]["raw"]["wikipedia_url"] = (
        "https://en.wikipedia.org/wiki/Bengal_cat#Cheetoh"
    )
    wikidata = wikidata_record("chee")
    wikidata["labels"]["en"] = "Cheetoh"
    wikidata["match_method"] = "manual_override"

    document = build_knowledge_documents([registry], [wikidata], [])[0]

    assert "Bengal_cat" not in document["text"]
    assert "Wikipedia URL" not in document["text"]


def test_duplicate_document_id_raises_error() -> None:
    with pytest.raises(KnowledgeDocumentError, match="Duplicate document IDs"):
        build_knowledge_documents(
            [registry_record()],
            [wikidata_record()],
            [wikipedia_record(language="en"), wikipedia_record(language="en")],
        )


def test_unknown_wikipedia_breed_id_raises_error() -> None:
    with pytest.raises(KnowledgeDocumentError, match="unknown breed_id"):
        build_knowledge_documents(
            [registry_record("mcoo")],
            [wikidata_record("mcoo")],
            [wikipedia_record("xxxx")],
        )


def test_documents_are_sorted_deterministically() -> None:
    documents = build_knowledge_documents(
        [registry_record("mcoo"), registry_record("bsho", "British Shorthair")],
        [wikidata_record("mcoo"), wikidata_record("bsho")],
        [wikipedia_record("mcoo"), wikipedia_record("bsho")],
    )

    assert [doc["breed_id"] for doc in documents] == ["bsho", "bsho", "mcoo", "mcoo"]


def test_two_document_writes_have_same_sha(tmp_path: Path) -> None:
    documents = build_knowledge_documents(
        [registry_record()],
        [wikidata_record()],
        [wikipedia_record()],
    )
    path = tmp_path / "knowledge_documents.jsonl"

    write_jsonl_atomic(documents, path)
    first = hashlib.sha256(path.read_bytes()).hexdigest()
    write_jsonl_atomic(documents, path)
    second = hashlib.sha256(path.read_bytes()).hexdigest()

    assert first == second


def test_report_counts_documents() -> None:
    report = build_knowledge_documents_report(
        build_knowledge_documents(
            [registry_record()],
            [wikidata_record()],
            [wikipedia_record(language="ru"), wikipedia_record(language="en")],
        )
    )

    assert report["total_documents"] == 3
    assert report["catapi_documents"] == 1
    assert report["wikipedia_documents"] == 2
