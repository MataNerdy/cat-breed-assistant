from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_breed_registry import (
    DuplicateBreedIDError,
    build_registry,
    build_registry_record,
    load_catapi_records,
    normalize_aliases,
    parse_breed_ids,
    write_jsonl_atomic,
)


def make_breed(
    breed_id: str = "mcoo",
    name: str = "Maine Coon",
    alt_names: str = "",
    **extra: object,
) -> dict:
    record = {
        "id": breed_id,
        "name": name,
        "alt_names": alt_names,
        "origin": "United States",
        "temperament": "Gentle, Intelligent",
    }
    record.update(extra)
    return record


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_registry_record_schema() -> None:
    source = make_breed()
    record = build_registry_record(source)

    assert record == {
        "schema_version": "1.0",
        "breed_id": "mcoo",
        "name_en": "Maine Coon",
        "name_ru": None,
        "aliases_en": [],
        "aliases_ru": [],
        "catapi": {
            "id": "mcoo",
            "raw": source,
        },
        "wikidata": None,
        "sources": ["thecatapi"],
        "review": {
            "status": "not_enriched",
            "warnings": [],
        },
    }


def test_alt_names_become_aliases() -> None:
    aliases = normalize_aliases("Coon Cat, Maine Cat", "Maine Coon")

    assert aliases == ["Coon Cat", "Maine Cat"]


def test_full_source_record_is_preserved_in_catapi_raw() -> None:
    source = make_breed(weight={"metric": "5 - 9"}, nested={"a": [1, 2]})

    record = build_registry_record(source)

    assert record["catapi"]["raw"] == source
    assert record["catapi"]["raw"]["nested"] == {"a": [1, 2]}


def test_build_registry_preserves_raw_id_and_name_values() -> None:
    source = make_breed(" mcoo ", " Maine Coon ")

    registry, _, _ = build_registry([source])

    assert registry[0]["breed_id"] == "mcoo"
    assert registry[0]["name_en"] == "Maine Coon"
    assert registry[0]["catapi"]["raw"] == source


def test_duplicate_aliases_are_removed() -> None:
    aliases = normalize_aliases(
        "Coon Cat, Maine Cat, Coon Cat,  , maine cat, Maine Coon",
        "Maine Coon",
    )

    assert aliases == ["Coon Cat", "Maine Cat"]


def test_duplicate_id_raises_error() -> None:
    records = [make_breed("mcoo"), make_breed("mcoo", "Maine Coon Copy")]

    with pytest.raises(DuplicateBreedIDError):
        build_registry(records)


def test_root_json_must_be_list(tmp_path: Path) -> None:
    input_path = tmp_path / "catapi_breeds.json"
    input_path.write_text(json.dumps({"id": "mcoo"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected root JSON object to be a list"):
        load_catapi_records(input_path)


def test_record_without_id_is_skipped() -> None:
    registry, skipped_records, duplicate_ids = build_registry(
        [{"name": "No ID"}, make_breed("mcoo")]
    )

    assert [item["breed_id"] for item in registry] == ["mcoo"]
    assert skipped_records == 1
    assert duplicate_ids == 0


def test_records_are_sorted_by_breed_id() -> None:
    registry, _, _ = build_registry(
        [make_breed("sphy", "Sphynx"), make_breed("bsho", "British Shorthair")]
    )

    assert [item["breed_id"] for item in registry] == ["bsho", "sphy"]


def test_breed_id_filter_does_not_depend_on_input_order() -> None:
    records = [
        make_breed("sphy", "Sphynx"),
        make_breed("bsho", "British Shorthair"),
        make_breed("mcoo", "Maine Coon"),
    ]

    registry, _, _ = build_registry(records, breed_ids=parse_breed_ids("mcoo,bsho"))

    assert [item["breed_id"] for item in registry] == ["bsho", "mcoo"]


def test_two_runs_create_byte_identical_jsonl(tmp_path: Path) -> None:
    records = [make_breed("mcoo"), make_breed("bsho", "British Shorthair")]
    registry, _, _ = build_registry(records)
    output_path = tmp_path / "breed_registry.jsonl"

    write_jsonl_atomic(registry, output_path)
    first_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    first_content = output_path.read_bytes()

    write_jsonl_atomic(registry, output_path)
    second_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert first_content == output_path.read_bytes()


def test_atomic_write_does_not_replace_target_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "breed_registry.jsonl"
    output_path.write_text("original\n", encoding="utf-8")

    def failing_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("scripts.build_breed_registry.os.replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_jsonl_atomic([build_registry_record(make_breed())], output_path)

    assert output_path.read_text(encoding="utf-8") == "original\n"
    assert not (tmp_path / ".breed_registry.jsonl.tmp").exists()


def test_written_jsonl_has_expected_content(tmp_path: Path) -> None:
    registry, _, _ = build_registry([make_breed("mcoo")])
    output_path = tmp_path / "breed_registry.jsonl"

    write_jsonl_atomic(registry, output_path)

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert read_jsonl(output_path)[0]["breed_id"] == "mcoo"

def test_filter_does_not_hide_duplicate_ids():
    records = [
        make_breed("mcoo"),
        make_breed("mcoo", "Duplicate"),
        make_breed("bsho", "British Shorthair"),
    ]

    with pytest.raises(DuplicateBreedIDError):
        build_registry(records, breed_ids={"bsho"})