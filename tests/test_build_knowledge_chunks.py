from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_knowledge_chunks import main


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def document() -> dict:
    return {
        "document_id": "mcoo:wikipedia:en",
        "breed_id": "mcoo",
        "breed_name_en": "Maine Coon",
        "breed_name_ru": "Мейн-кун",
        "aliases": [],
        "language": "en",
        "source": "wikipedia",
        "document_type": "wikipedia_article",
        "lead": "Lead.",
        "sections": [],
        "provenance": {
            "source_url": "https://en.wikipedia.org/wiki/Maine_Coon",
            "page_id": 1,
            "revision_id": 2,
            "source_resolution": {
                "method": "wikidata_sitelink",
                "source_relation": "standalone_article",
                "reason": None,
            },
        },
        "warnings": [],
    }


def test_build_knowledge_chunks_cli_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "knowledge_documents.jsonl"
    output = tmp_path / "knowledge_chunks.jsonl"
    report = tmp_path / "report.json"
    skipped = tmp_path / "skipped.jsonl"
    overrides = tmp_path / "overrides.json"
    write_jsonl(input_path, [document()])
    overrides.write_text("{}\n", encoding="utf-8")

    args = [
        "build_knowledge_chunks.py",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--report",
        str(report),
        "--skipped-broader-output",
        str(skipped),
        "--broader-overrides",
        str(overrides),
        "--target-chars",
        "2200",
        "--max-chars",
        "3000",
    ]
    monkeypatch.setattr("sys.argv", args)
    assert main() == 0
    first_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    monkeypatch.setattr("sys.argv", args)
    assert main() == 0
    second_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert json.loads(report.read_text(encoding="utf-8"))["total_chunks"] == 1
    assert skipped.read_text(encoding="utf-8") == ""
