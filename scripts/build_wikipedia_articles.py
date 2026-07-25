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
DEFAULT_SOURCE_OVERRIDES_PATH = Path("data/curated/wikipedia_source_overrides.json")
DEFAULT_CACHE_DIR = Path("data/cache/wikipedia")
DEFAULT_LANGUAGES = ("ru", "en")
BROADER_SOURCE_WARNING = (
    "Source article covers a broader topic; sections require breed-specific review "
    "before chunking"
)
SUPPORTED_LANGUAGES = {"en", "ru"}
SUPPORTED_WIKI_PROJECTS = {"enwiki", "ruwiki", "simplewiki"}
WIKI_PROJECT_DOMAINS = {
    "enwiki": "https://en.wikipedia.org/wiki/",
    "ruwiki": "https://ru.wikipedia.org/wiki/",
    "simplewiki": "https://simple.wikipedia.org/wiki/",
}
SUPPORTED_SOURCE_RELATIONS = {
    "standalone_article",
    "covered_by_broader_article",
    "section_of_another_article",
    "redirect",
}


def parse_csv(value: str | None, default: tuple[str, ...] | None = None) -> set[str] | None:
    if value is None:
        return set(default) if default is not None else None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    if parsed:
        return parsed
    return set(default) if default is not None else None


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


def load_source_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected source overrides JSON object: {path}")

    overrides = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid source override value for {key!r}")
        overrides[key] = value
    return overrides


def validate_source_overrides(
    source_overrides: dict[str, dict[str, str]],
    enrichment_records: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    known_breed_ids = {
        record["breed_id"]
        for record in enrichment_records
        if isinstance(record.get("breed_id"), str)
    }
    validated = {}

    for key, value in source_overrides.items():
        breed_id, language = parse_override_key(key)
        if breed_id not in known_breed_ids:
            raise ValueError(f"Unknown breed_id in source override {key!r}")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language in source override {key!r}")

        title = value.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Source override {key!r} must contain a non-empty title")

        source_relation = value.get("source_relation")
        if source_relation not in SUPPORTED_SOURCE_RELATIONS:
            raise ValueError(
                f"Source override {key!r} has unsupported source_relation: "
                f"{source_relation!r}"
            )

        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Source override {key!r} must contain a non-empty reason")

        wiki_project = value.get("wiki_project") or f"{language}wiki"
        if wiki_project not in SUPPORTED_WIKI_PROJECTS:
            raise ValueError(f"Unsupported wiki_project in source override {key!r}")

        content_language = value.get("content_language") or language
        if content_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported content_language in source override {key!r}")
        if content_language != language:
            raise ValueError(
                f"Source override {key!r} content_language must match key language"
            )

        verified_url = value.get("verified_url")
        if verified_url is not None:
            expected_prefix = WIKI_PROJECT_DOMAINS[wiki_project]
            if not isinstance(verified_url, str) or not verified_url.startswith(
                expected_prefix
            ):
                raise ValueError(
                    f"Source override {key!r} verified_url must start with "
                    f"{expected_prefix}"
                )

        validated[key] = {
            "title": title.strip(),
            "source_relation": source_relation,
            "reason": reason.strip(),
            "wiki_project": wiki_project,
            "content_language": content_language,
            **(
                {"verified_url": verified_url}
                if isinstance(verified_url, str) and verified_url
                else {}
            ),
        }

    return validated


def parse_override_key(key: str) -> tuple[str, str]:
    if not isinstance(key, str) or key.count(":") != 1:
        raise ValueError(f"Invalid source override key: {key!r}")
    breed_id, language = key.split(":", maxsplit=1)
    if not breed_id or not language:
        raise ValueError(f"Invalid source override key: {key!r}")
    return breed_id, language


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


def source_resolution(
    method: str,
    source_relation: str,
    reason: str | None = None,
    wiki_project: str | None = None,
) -> dict[str, str | None]:
    resolution = {
        "method": method,
        "source_relation": source_relation,
        "reason": reason,
    }
    if wiki_project:
        resolution["wiki_project"] = wiki_project
    return resolution


def is_missing_page_response(cached_response: dict[str, Any]) -> bool:
    error = cached_response.get("api_response", {}).get("error")
    if not isinstance(error, dict):
        return False
    return error.get("code") in {"missingtitle", "invalidtitle"}


def build_articles(
    enrichment_records: list[dict[str, Any]],
    client: WikipediaClient,
    breed_ids: set[str] | None,
    languages: set[str],
    source_overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    articles = []
    unresolved = []
    source_overrides = source_overrides or {}

    selected_records = [
        record
        for record in enrichment_records
        if breed_ids is None or record.get("breed_id") in breed_ids
    ]
    selected_records.sort(key=lambda item: item["breed_id"])

    for record in selected_records:
        breed_id = record["breed_id"]
        sitelinks = record.get("sitelinks") or {}

        for language in sorted(languages):
            site_key = f"{language}wiki"
            override = source_overrides.get(f"{breed_id}:{language}")
            title = override["title"] if override else sitelinks.get(site_key)
            resolution = (
                source_resolution(
                    "manual_override",
                    override["source_relation"],
                    override["reason"],
                    override.get("wiki_project") or f"{language}wiki",
                )
                if override
                else source_resolution("wikidata_sitelink", "standalone_article")
            )
            if not isinstance(title, str) or not title:
                reason = (
                    "missing_verified_source"
                    if any(key.startswith(f"{breed_id}:") for key in source_overrides)
                    else "missing_sitelink"
                )
                unresolved.append(
                    unresolved_record(
                        breed_id,
                        language,
                        None,
                        reason,
                        f"{language} Wikipedia source is missing",
                    )
                )
                continue

            try:
                wiki_project = (
                    override.get("wiki_project") or f"{language}wiki"
                    if override
                    else f"{language}wiki"
                )
                cached_response = client.fetch_article(
                    breed_id,
                    language,
                    title,
                    wiki_project=wiki_project,
                )
                if override and is_missing_page_response(cached_response):
                    unresolved.append(
                        unresolved_record(
                            breed_id,
                            language,
                            title,
                            "override_page_not_found",
                            f"Manual override page was not found: {title}",
                        )
                    )
                    continue
                article = parse_article_record(cached_response, breed_id, language)
                article["source_resolution"] = resolution
                article["wiki_project"] = wiki_project
                if resolution["source_relation"] != "standalone_article":
                    warnings = article.setdefault("warnings", [])
                    if BROADER_SOURCE_WARNING not in warnings:
                        warnings.append(BROADER_SOURCE_WARNING)
                    article["warnings"] = sorted(warnings)
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
    parser.add_argument("--source-overrides", type=Path, default=DEFAULT_SOURCE_OVERRIDES_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--breed-ids", help="Comma-separated breed ids to process.")
    parser.add_argument("--languages", help="Comma-separated Wikipedia languages.")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        enrichment_records = read_jsonl(args.input)
        source_overrides = validate_source_overrides(
            load_source_overrides(args.source_overrides),
            enrichment_records,
        )
        client = WikipediaClient(
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        )
        articles, unresolved = build_articles(
            enrichment_records,
            client=client,
            breed_ids=parse_csv(args.breed_ids),
            languages=parse_csv(args.languages, DEFAULT_LANGUAGES) or set(DEFAULT_LANGUAGES),
            source_overrides=source_overrides,
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
    print(f"Source overrides: {args.source_overrides}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Cache hits: {client.cache_hits}")
    print(f"HTTP requests: {client.http_requests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
