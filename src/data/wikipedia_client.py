from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_CACHE_DIR = Path("data/cache/wikipedia")
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 2
USER_AGENT = (
    "cat-breed-assistant/0.1 "
    "(educational data pipeline; https://github.com/MataNerdy/cat-breed-assistant)"
)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class WikipediaClientError(RuntimeError):
    """Raised when a Wikipedia HTTP request cannot be completed."""


class WikipediaClient:
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

    def fetch_article(self, breed_id: str, language: str, title: str) -> dict[str, Any]:
        cache_path = self._cache_path(breed_id, language)
        if cache_path.exists() and not self.refresh_cache:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        api_response = self._request_json(
            f"https://{language}.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "text|sections|revid|displaytitle|properties",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
        )
        cached = {
            "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
                "+00:00", "Z"
            ),
            "requested_language": language,
            "requested_title": title,
            "api_response": api_response,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cached, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return cached

    def _request_json(
        self,
        url: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in TRANSIENT_STATUS_CODES:
                    last_error = WikipediaClientError(
                        f"Temporary HTTP {response.status_code} for {url}"
                    )
                    if attempt < self.retries:
                        self._sleep_before_retry(response, attempt)
                        continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise WikipediaClientError(f"Expected JSON object from {url}")
                return data
            except (requests.RequestException, ValueError, WikipediaClientError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break

        raise WikipediaClientError(str(last_error))

    def _sleep_before_retry(self, response: requests.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                time.sleep(float(retry_after))
                return
            except ValueError:
                pass
        time.sleep(0.25 * (attempt + 1))

    def _cache_path(self, breed_id: str, language: str) -> Path:
        return self.cache_dir / f"{breed_id}_{language}.json"
