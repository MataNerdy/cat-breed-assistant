from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.wikipedia_articles_viewer import (
    compute_article_stats,
    find_dataset_issues,
    load_jsonl,
    load_review_overrides,
    save_review_overrides_atomic,
    update_review_override,
)


def article(
    breed_id: str = "mcoo",
    language: str = "en",
    sections: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "breed_id": breed_id,
        "language": language,
        "title": "Maine Coon",
        "page_id": 1,
        "revision_id": 2,
        "source": "wikipedia",
        "source_url": "https://en.wikipedia.org/wiki/Maine_Coon",
        "retrieved_at": "2026-07-23T00:00:00Z",
        "lead": "Large friendly cat.",
        "sections": sections
        if sections is not None
        else [{"index": 1, "title": "History", "text": "Old breed."}],
        "warnings": [],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "articles.jsonl"
    write_jsonl(path, [article()])

    records = load_jsonl(path)

    assert records[0]["breed_id"] == "mcoo"


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "articles.jsonl"
    path.write_text('{"ok": true}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)


def test_compute_article_stats() -> None:
    stats = compute_article_stats(
        article(sections=[{"index": 1, "title": "History", "text": "Old breed."}])
    )

    assert stats["lead_length"] == len("Large friendly cat.")
    assert stats["section_count"] == 1
    assert stats["total_characters"] == len("Large friendly cat.") + len("Old breed.")


def test_find_empty_sections() -> None:
    issues = find_dataset_issues(
        [article(sections=[{"index": 1, "title": "Container", "text": ""}])]
    )

    assert issues["articles_with_empty_sections"][0]["empty_section_count"] == 1
    assert issues["zero_length_sections"][0]["title"] == "Container"


def test_find_duplicate_section_texts() -> None:
    issues = find_dataset_issues(
        [
            article(
                sections=[
                    {"index": 1, "title": "A", "text": "same text"},
                    {"index": 2, "title": "B", "text": "same text"},
                ]
            )
        ]
    )

    assert issues["duplicate_section_texts"][0]["count"] == 2


def test_save_review_overrides(tmp_path: Path) -> None:
    path = tmp_path / "wikipedia_review_overrides.json"

    save_review_overrides_atomic(
        {"mcoo:ru": {"status": "approved", "note": "ok", "excluded_sections": []}},
        path,
    )

    assert load_review_overrides(path)["mcoo:ru"]["status"] == "approved"


def test_update_existing_override(tmp_path: Path) -> None:
    path = tmp_path / "wikipedia_review_overrides.json"
    save_review_overrides_atomic(
        {"mcoo:ru": {"status": "not_reviewed", "note": "", "excluded_sections": []}},
        path,
    )

    updated = update_review_override(
        load_review_overrides(path),
        article_key="mcoo:ru",
        status="needs_cleanup",
        note="remove gallery",
        excluded_sections=["Gallery"],
    )
    save_review_overrides_atomic(updated, path)

    assert load_review_overrides(path)["mcoo:ru"] == {
        "status": "needs_cleanup",
        "note": "remove gallery",
        "excluded_sections": ["Gallery"],
    }


def test_atomic_write_keeps_original_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "wikipedia_review_overrides.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    def failing_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("tools.wikipedia_articles_viewer.os.replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_review_overrides_atomic({"new": True}, path)

    assert path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not (tmp_path / ".wikipedia_review_overrides.json.tmp").exists()


def test_source_jsonl_is_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "articles.jsonl"
    write_jsonl(path, [article()])
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    records = load_jsonl(path)
    compute_article_stats(records[0])
    find_dataset_issues(records)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
