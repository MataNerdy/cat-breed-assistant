from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote


SCHEMA_VERSION = "1.0"
EXCLUDED_SECTION_TITLES = {
    "ru": {
        "примечания",
        "ссылки",
        "литература",
        "источники",
        "см. также",
    },
    "en": {
        "references",
        "external links",
        "further reading",
        "bibliography",
        "see also",
        "notes",
    },
}


class ArticleHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, str]] = []
        self._skip_depth = 0
        self._current_tag: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if (
            tag in {"style", "script", "table", "figure", "sup"}
            or "mw-editsection" in classes
            or "reference" in classes
            or "navbox" in classes
        ):
            self._skip_depth += 1
            return

        if self._skip_depth:
            return

        if tag in {"p", "li", "h2", "h3"}:
            self._flush()
            self._current_tag = tag
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if tag == self._current_tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._current_tag:
            return
        self._buffer.append(data)

    def _flush(self) -> None:
        if not self._current_tag:
            return
        text = clean_text(" ".join(self._buffer))
        if text:
            block_type = "heading" if self._current_tag in {"h2", "h3"} else "text"
            self.blocks.append({"type": block_type, "text": text})
        self._current_tag = None
        self._buffer = []


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\[\s*\d+\s*\]", "", value)
    value = re.sub(r"\b\d+\s*\]", "", value)
    value = re.sub(r"\s+\]", "", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_excluded_section(title: str, language: str) -> bool:
    normalized = clean_text(title).casefold()
    return normalized in EXCLUDED_SECTION_TITLES.get(language, set())


def extract_page_metadata(cached_response: dict[str, Any]) -> dict[str, Any]:
    parse = cached_response.get("api_response", {}).get("parse")
    if not isinstance(parse, dict):
        raise ValueError("Wikipedia API response does not contain parse data.")

    title = parse.get("title")
    page_id = parse.get("pageid")
    revision_id = parse.get("revid")
    if not isinstance(title, str) or not title:
        raise ValueError("Wikipedia page title is missing.")
    if not isinstance(page_id, int):
        raise ValueError("Wikipedia page_id is missing.")
    if not isinstance(revision_id, int):
        raise ValueError("Wikipedia revision_id is missing.")

    return {
        "title": title,
        "page_id": page_id,
        "revision_id": revision_id,
    }


def extract_blocks(cached_response: dict[str, Any]) -> list[dict[str, str]]:
    parse = cached_response.get("api_response", {}).get("parse")
    if not isinstance(parse, dict):
        raise ValueError("Wikipedia API response does not contain parse data.")
    text = parse.get("text")
    if isinstance(text, dict):
        html_text = text.get("*")
    else:
        html_text = text
    if not isinstance(html_text, str) or not html_text.strip():
        raise ValueError("Wikipedia article text is missing.")

    extractor = ArticleHTMLExtractor()
    extractor.feed(html_text)
    extractor.close()
    return extractor.blocks


def extract_lead(blocks: list[dict[str, str]]) -> str:
    lead_parts = []
    for block in blocks:
        if block["type"] == "heading":
            break
        if block["type"] == "text":
            lead_parts.append(block["text"])
    return "\n\n".join(lead_parts).strip()


def extract_sections(blocks: list[dict[str, str]], language: str) -> list[dict[str, Any]]:
    sections = []
    current_title: str | None = None
    current_text: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_text
        if current_title and not is_excluded_section(current_title, language):
            sections.append(
                {
                    "index": len(sections) + 1,
                    "title": current_title,
                    "text": "\n\n".join(current_text).strip(),
                }
            )
        current_title = None
        current_text = []

    for block in blocks:
        if block["type"] == "heading":
            flush()
            current_title = block["text"]
        elif current_title and block["type"] == "text":
            current_text.append(block["text"])
    flush()
    return sections


def build_source_url(language: str, title: str) -> str:
    quoted_title = quote(title.replace(" ", "_"), safe="()_-")
    return f"https://{language}.wikipedia.org/wiki/{quoted_title}"


def parse_article_record(
    cached_response: dict[str, Any],
    breed_id: str,
    language: str,
) -> dict[str, Any]:
    metadata = extract_page_metadata(cached_response)
    blocks = extract_blocks(cached_response)
    lead = extract_lead(blocks)
    if not lead:
        raise ValueError("Wikipedia article lead is empty.")

    return {
        "schema_version": SCHEMA_VERSION,
        "breed_id": breed_id,
        "language": language,
        "title": metadata["title"],
        "page_id": metadata["page_id"],
        "revision_id": metadata["revision_id"],
        "source": "wikipedia",
        "source_url": build_source_url(language, metadata["title"]),
        "retrieved_at": cached_response["retrieved_at"],
        "lead": lead,
        "sections": extract_sections(blocks, language),
        "warnings": [],
    }
