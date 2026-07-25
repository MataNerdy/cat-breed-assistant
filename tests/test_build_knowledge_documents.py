from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_knowledge_documents import main


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_build_knowledge_documents_cli_is_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = tmp_path / "registry.jsonl"
    wikidata = tmp_path / "wikidata.jsonl"
    wikipedia = tmp_path / "wikipedia.jsonl"
    name_overrides = tmp_path / "name_overrides.json"
    output = tmp_path / "knowledge_documents.jsonl"
    report = tmp_path / "report.json"
    name_overrides.write_text("{}\n", encoding="utf-8")
    write_jsonl(
        registry,
        [
            {
                "breed_id": "mcoo",
                "name_en": "Maine Coon",
                "name_ru": None,
                "aliases_en": [],
                "aliases_ru": [],
                "catapi": {
                    "raw": {
                        "id": "mcoo",
                        "name": "Maine Coon",
                        "description": "Large cat.",
                    }
                },
            }
        ],
    )
    write_jsonl(
        wikidata,
        [{"breed_id": "mcoo", "labels": {"ru": "Мейн-кун"}, "aliases": {}}],
    )
    write_jsonl(
        wikipedia,
        [
            {
                "breed_id": "mcoo",
                "language": "en",
                "title": "Maine Coon",
                "lead": "Lead.",
                "sections": [],
                "source_url": "https://en.wikipedia.org/wiki/Maine_Coon",
                "page_id": 1,
                "revision_id": 2,
                "retrieved_at": "2026-07-24T00:00:00Z",
                "source_resolution": {
                    "method": "wikidata_sitelink",
                    "source_relation": "standalone_article",
                    "reason": None,
                },
            }
        ],
    )

    args = [
        "build_knowledge_documents.py",
        "--registry",
        str(registry),
        "--wikidata",
        str(wikidata),
        "--wikipedia",
        str(wikipedia),
        "--name-overrides",
        str(name_overrides),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    monkeypatch.setattr("sys.argv", args)
    assert main() == 0
    first_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    monkeypatch.setattr("sys.argv", args)
    assert main() == 0
    second_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert json.loads(report.read_text(encoding="utf-8"))["total_documents"] == 2
