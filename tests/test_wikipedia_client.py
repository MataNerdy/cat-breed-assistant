from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src.data.wikipedia_client import WikipediaClient, WikipediaClientError


class FakeResponse:
    def __init__(
        self,
        payload: dict,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.headers = {}
        self.calls = []

    def get(self, url: str, params: dict | None = None, timeout: int | None = None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self.responses:
            raise AssertionError("Unexpected HTTP call")
        return self.responses.pop(0)


def parse_payload(title: str = "Maine Coon") -> dict:
    return {
        "parse": {
            "title": title,
            "pageid": 1,
            "revid": 2,
            "text": "<p>Maine Coon text.</p>",
            "sections": [],
        }
    }


def test_article_successfully_loads(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(parse_payload())])
    client = WikipediaClient(cache_dir=tmp_path, session=session)

    cached = client.fetch_article("mcoo", "en", "Maine Coon")

    assert cached["api_response"]["parse"]["title"] == "Maine Coon"
    assert session.calls[0]["params"]["page"] == "Maine Coon"


def test_response_is_saved_to_cache(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(parse_payload())])
    client = WikipediaClient(cache_dir=tmp_path, session=session)

    client.fetch_article("mcoo", "en", "Maine Coon")

    assert (tmp_path / "mcoo_en.json").exists()


def test_cached_response_avoids_http_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "mcoo_en.json"
    cache_path.write_text(
        json.dumps(
            {
                "retrieved_at": "2026-07-23T00:00:00Z",
                "requested_language": "en",
                "requested_title": "Maine Coon",
                "api_response": parse_payload(),
            }
        ),
        encoding="utf-8",
    )
    session = FakeSession([])
    client = WikipediaClient(cache_dir=tmp_path, session=session)

    cached = client.fetch_article("mcoo", "en", "Maine Coon")

    assert cached["retrieved_at"] == "2026-07-23T00:00:00Z"
    assert session.calls == []


def test_refresh_cache_updates_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "mcoo_en.json"
    cache_path.write_text(
        json.dumps(
            {
                "retrieved_at": "old",
                "requested_language": "en",
                "requested_title": "Maine Coon",
                "api_response": parse_payload("Old Title"),
            }
        ),
        encoding="utf-8",
    )
    session = FakeSession([FakeResponse(parse_payload("New Title"))])
    client = WikipediaClient(cache_dir=tmp_path, session=session, refresh_cache=True)

    cached = client.fetch_article("mcoo", "en", "Maine Coon")

    assert cached["api_response"]["parse"]["title"] == "New Title"
    assert json.loads(cache_path.read_text(encoding="utf-8"))["retrieved_at"] != "old"


def test_temporary_http_error_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.data.wikipedia_client.time.sleep", lambda _: None)
    session = FakeSession(
        [
            FakeResponse({}, status_code=503, headers={"Retry-After": "0"}),
            FakeResponse(parse_payload()),
        ]
    )
    client = WikipediaClient(cache_dir=tmp_path, session=session, retries=1)

    cached = client.fetch_article("mcoo", "en", "Maine Coon")

    assert cached["api_response"]["parse"]["title"] == "Maine Coon"
    assert len(session.calls) == 2


def test_permanent_http_error_is_handled(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse({}, status_code=404)])
    client = WikipediaClient(cache_dir=tmp_path, session=session, retries=0)

    with pytest.raises(WikipediaClientError):
        client.fetch_article("mcoo", "en", "Maine Coon")
