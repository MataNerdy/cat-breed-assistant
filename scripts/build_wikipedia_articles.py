from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wikipedia_client import WikipediaClient, WikipediaClientError
from src.data.wikipedia_parser import parse_article_record


DEFAULT_INPUT_PATH = Path("data/staging/wikidata_enrichment.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/staging/wikipedia_articles.jsonl")
DEFAULT_UNRESOLVED_PATH = Path("data/reports/wikipedia_unresolved.jsonl")
DEFAULT_CACHE_DIR = Path("data/cache/wikipedia")
DEFAULT_BREED_IDS = ("beng", "bsho", "mcoo", "sibe", "sphy")
DEFAULT_LANGUAGES = ("ru", "en")


def parse_csv(value: str | None, default: tuple[str, ...]) -> set[str]:
    if value is None:
        return set(default)
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or set(default)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
    return records


def write_jsonl_atomic(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def unresolved_record(
    breed_id: str,
    language: str,
    title: str | None,
    reason: str,
    warning: str,
) -> dict[str, Any]:
    return {
        "breed_id": breed_id,
        "language": language,
        "title": title,
        "reason": reason,
        "warnings": [warning],
    }


def build_articles(
    enrichment_records: list[dict[str, Any]],
    client: WikipediaClient,
    breed_ids: set[str],
    languages: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    articles = []
    unresolved = []

    selected_records = [
        record
        for record in enrichment_records
        if record.get("breed_id") in breed_ids
    ]
    selected_records.sort(key=lambda item: item["breed_id"])

    for record in selected_records:
        breed_id = record["breed_id"]
        sitelinks = record.get("sitelinks") or {}

        for language in sorted(languages):
            site_key = f"{language}wiki"
            title = sitelinks.get(site_key)
            if not isinstance(title, str) or not title:
                unresolved.append(
                    unresolved_record(
                        breed_id,
                        language,
                        None,
                        "missing_sitelink",
                        f"{language} Wikipedia sitelink is missing",
                    )
                )
                continue

            try:
                cached_response = client.fetch_article(breed_id, language, title)
                article = parse_article_record(cached_response, breed_id, language)
            except WikipediaClientError as exc:
                unresolved.append(
                    unresolved_record(
                        breed_id,
                        language,
                        title,
                        "http_error",
                        str(exc),
                    )
                )
            except (ValueError, KeyError) as exc:
                unresolved.append(
                    unresolved_record(
                        breed_id,
                        language,
                        title,
                        "parse_error",
                        str(exc),
                    )
                )
            else:
                articles.append(article)

    articles.sort(key=lambda item: (item["breed_id"], item["language"]))
    unresolved.sort(key=lambda item: (item["breed_id"], item["language"]))
    return articles, unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build staged Wikipedia article texts for matched cat breeds."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--unresolved-output", type=Path, default=DEFAULT_UNRESOLVED_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--breed-ids", help="Comma-separated breed ids to process.")
    parser.add_argument("--languages", help="Comma-separated Wikipedia languages.")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        enrichment_records = read_jsonl(args.input)
        client = WikipediaClient(
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        )
        articles, unresolved = build_articles(
            enrichment_records,
            client=client,
            breed_ids=parse_csv(args.breed_ids, DEFAULT_BREED_IDS),
            languages=parse_csv(args.languages, DEFAULT_LANGUAGES),
        )
        write_jsonl_atomic(articles, args.output)
        write_jsonl_atomic(unresolved, args.unresolved_output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not build Wikipedia articles: {exc}")
        return 1

    print(f"Input enrichment records: {len(enrichment_records)}")
    print(f"Written article records: {len(articles)}")
    print(f"Unresolved records: {len(unresolved)}")
    print(f"Output: {args.output}")
    print(f"Unresolved output: {args.unresolved_output}")
    print(f"Cache dir: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
