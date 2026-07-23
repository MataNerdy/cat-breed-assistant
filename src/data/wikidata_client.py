from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests


DEFAULT_CACHE_DIR = Path("data/cache/wikidata")
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 2
USER_AGENT = (
    "cat-breed-assistant/0.1 "
    "(educational data pipeline; https://github.com/MataNerdy/cat-breed-assistant)"
)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class WikidataClientError(RuntimeError):
    """Raised when a Wikidata or Wikipedia HTTP request cannot be completed."""


def parse_wikipedia_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    host_parts = parsed.netloc.split(".")
    if len(host_parts) < 3 or host_parts[1] != "wikipedia":
        raise ValueError(f"Unsupported Wikipedia URL: {url}")

    language = host_parts[0]
    prefix = "/wiki/"
    if not parsed.path.startswith(prefix):
        raise ValueError(f"Unsupported Wikipedia URL path: {url}")

    title = unquote(parsed.path[len(prefix) :]).replace("_", " ")
    return language, f"{language}wiki", title


class WikidataClient:
    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        session: requests.Session | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
        refresh_cache: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.retries = retries
        self.refresh_cache = refresh_cache
        self.session.headers.update({"User-Agent": USER_AGENT})

    def resolve_entity_id_from_wikipedia_url(self, wikipedia_url: str) -> str | None:
        resolution_cache_path = self._wikipedia_resolution_cache_path(wikipedia_url)
        if resolution_cache_path.exists() and not self.refresh_cache:
            data = json.loads(resolution_cache_path.read_text(encoding="utf-8"))
            entity_id = data.get("entity_id")
            return entity_id if isinstance(entity_id, str) and entity_id else None

        language, site_key, title = parse_wikipedia_url(wikipedia_url)

        if not self.refresh_cache:
            cached_entity_id = self._find_cached_entity_by_sitelink(site_key, title)
            if cached_entity_id:
                self._write_wikipedia_resolution_cache(
                    resolution_cache_path,
                    wikipedia_url,
                    cached_entity_id,
                )
                return cached_entity_id

        data = self._request_json(
            f"https://{language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "pageprops",
                "titles": title,
                "redirects": "1",
                "format": "json",
            },
        )
        pages = data.get("query", {}).get("pages", {})
        if not isinstance(pages, dict):
            return None

        for page in pages.values():
            pageprops = page.get("pageprops") if isinstance(page, dict) else None
            if isinstance(pageprops, dict):
                entity_id = pageprops.get("wikibase_item")
                if isinstance(entity_id, str) and entity_id:
                    self._write_wikipedia_resolution_cache(
                        resolution_cache_path,
                        wikipedia_url,
                        entity_id,
                    )
                    return entity_id
        return None

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        cache_path = self._entity_cache_path(entity_id)
        if cache_path.exists() and not self.refresh_cache:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return data.get("entities", {}).get(entity_id)

        data = self._request_json(
            f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json",
            params=None,
        )
        entities = data.get("entities", {})
        entity = entities.get(entity_id) if isinstance(entities, dict) else None
        if not isinstance(entity, dict) or entity.get("missing"):
            return None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return entity

    def search_entities(self, query: str, language: str = "en") -> list[dict[str, Any]]:
        cache_path = self._search_cache_path(query, language)
        if cache_path.exists() and not self.refresh_cache:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            search = data.get("search", [])
            return search if isinstance(search, list) else []

        data = self._request_json(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": language,
                "uselang": language,
                "type": "item",
                "limit": "10",
                "format": "json",
            },
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        search = data.get("search", [])
        return search if isinstance(search, list) else []

    def _request_json(
        self,
        url: str,
        params: dict[str, str] | None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in TRANSIENT_STATUS_CODES:
                    last_error = WikidataClientError(
                        f"Temporary HTTP {response.status_code} for {url}"
                    )
                    if attempt < self.retries:
                        self._sleep_before_retry(response, attempt)
                        continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise WikidataClientError(f"Expected JSON object from {url}")
                return data
            except (requests.RequestException, ValueError, WikidataClientError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break

        raise WikidataClientError(str(last_error))

    def _sleep_before_retry(self, response: requests.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                time.sleep(float(retry_after))
                return
            except ValueError:
                pass
        time.sleep(0.25 * (attempt + 1))

    def _entity_cache_path(self, entity_id: str) -> Path:
        return self.cache_dir / f"{entity_id}.json"

    def _search_cache_path(self, query: str, language: str) -> Path:
        digest = hashlib.sha256(f"{language}:{query}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"search_{language}_{digest}.json"

    def _wikipedia_resolution_cache_path(self, wikipedia_url: str) -> Path:
        digest = hashlib.sha256(wikipedia_url.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"wikipedia_resolution_{digest}.json"

    def _write_wikipedia_resolution_cache(
        self,
        cache_path: Path,
        wikipedia_url: str,
        entity_id: str,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "wikipedia_url": wikipedia_url,
                    "entity_id": entity_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _find_cached_entity_by_sitelink(self, site_key: str, title: str) -> str | None:
        normalized_title = title.replace("_", " ")
        for cache_path in sorted(self.cache_dir.glob("Q*.json")):
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entities = data.get("entities", {})
            if not isinstance(entities, dict):
                continue
            for entity_id, entity in entities.items():
                sitelink = entity.get("sitelinks", {}).get(site_key)
                if not isinstance(sitelink, dict):
                    continue
                cached_title = str(sitelink.get("title", "")).replace("_", " ")
                if cached_title == normalized_title:
                    return str(entity_id)
        return None
