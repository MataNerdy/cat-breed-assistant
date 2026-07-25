from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

from src.data.source_scope import is_service_section_path, is_service_section_title


SCHEMA_VERSION = "1.0"
SKIPPED_CONTAINER_CLASSES = {
    "authority-control",
    "catlinks",
    "gallery",
    "metadata",
    "mw-gallery",
    "mw-references-wrap",
    "navbox",
    "noprint",
    "portal",
    "reflist",
    "thumb",
    "vertical-navbox",
}


class ArticleHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, str]] = []
        self.warnings: list[str] = []
        self._skip_depth = 0
        self._current_tag: str | None = None
        self._current_level: int | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if "gallery" in classes or "mw-gallery" in classes:
            self.warnings.append("Gallery/media markup was skipped during parsing.")
        if (
            tag in {"style", "script", "table", "figure", "sup"}
            or "mw-editsection" in classes
            or "reference" in classes
            or classes.intersection(SKIPPED_CONTAINER_CLASSES)
        ):
            self._skip_depth += 1
            return

        if self._skip_depth:
            return

        if tag in {"p", "li", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self._current_tag = tag
            self._current_level = int(tag[1]) if tag.startswith("h") else None
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
            block_type = (
                "heading"
                if self._current_tag in {"h2", "h3", "h4", "h5", "h6"}
                else "text"
            )
            block: dict[str, Any] = {"type": block_type, "text": text}
            if block_type == "heading":
                block["level"] = self._current_level
            self.blocks.append(block)
        self._current_tag = None
        self._current_level = None
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
    return is_service_section_title(clean_text(title))


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


def extract_blocks(cached_response: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
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
    return extractor.blocks, sorted(set(extractor.warnings))


def extract_lead(blocks: list[dict[str, Any]]) -> str:
    lead_parts = []
    for block in blocks:
        if block["type"] == "heading":
            break
        if block["type"] == "text":
            lead_parts.append(block["text"])
    return "\n\n".join(lead_parts).strip()


def _section_index_for(level: int, counters: dict[int, int]) -> str:
    counters[level] = counters.get(level, 0) + 1
    for stale_level in list(counters):
        if stale_level > level:
            del counters[stale_level]
    ordered_levels = sorted(item for item in counters if item >= 2 and item <= level)
    return ".".join(str(counters[item]) for item in ordered_levels)


def extract_sections(blocks: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    sections = []
    current_heading: dict[str, Any] | None = None
    current_text: list[str] = []
    heading_stack: list[dict[str, Any]] = []
    counters: dict[int, int] = {}

    def flush() -> None:
        nonlocal current_heading, current_text
        if (
            current_heading
            and current_text
            and not is_service_section_path([item["title"] for item in heading_stack])
        ):
            path = [item["title"] for item in heading_stack]
            sections.append(
                {
                    "index": current_heading["index"],
                    "level": current_heading["level"],
                    "title": current_heading["title"],
                    "parent_index": current_heading["parent_index"],
                    "section_path": path,
                    "text": "\n\n".join(current_text).strip(),
                }
            )
        current_heading = None
        current_text = []

    for block in blocks:
        if block["type"] == "heading":
            flush()
            level = int(block.get("level") or 2)
            title = block["text"]
            index = _section_index_for(level, counters)
            heading_stack[:] = [
                item for item in heading_stack if int(item["level"]) < level
            ]
            parent_index = heading_stack[-1]["index"] if heading_stack else None
            current_heading = {
                "index": index,
                "level": level,
                "title": title,
                "parent_index": parent_index,
            }
            heading_stack.append(current_heading)
        elif current_heading and block["type"] == "text":
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
    blocks, warnings = extract_blocks(cached_response)
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
        "warnings": warnings,
    }
