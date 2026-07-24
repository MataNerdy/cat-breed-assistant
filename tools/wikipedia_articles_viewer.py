from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEFAULT_ARTICLES_PATH = Path("data/staging/wikipedia_articles.jsonl")
DEFAULT_OVERRIDES_PATH = Path("data/curated/wikipedia_review_overrides.json")
REVIEW_STATUSES = ("not_reviewed", "approved", "needs_cleanup", "rejected")
REQUIRED_FIELDS = {
    "schema_version",
    "breed_id",
    "language",
    "title",
    "page_id",
    "revision_id",
    "source",
    "source_url",
    "retrieved_at",
    "lead",
    "sections",
    "warnings",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {jsonl_path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {jsonl_path}: "
                    "expected object."
                )
            records.append(record)
    return records


def compute_article_stats(article: dict[str, Any]) -> dict[str, Any]:
    sections = article.get("sections") or []
    lead = article.get("lead") or ""
    section_texts = [section.get("text") or "" for section in sections]
    empty_sections = [
        section for section in sections if not (section.get("text") or "").strip()
    ]
    total_characters = len(lead) + sum(len(text) for text in section_texts)
    duplicate_texts = [
        text
        for text, count in Counter(section_texts).items()
        if text.strip() and count > 1
    ]
    duplicate_titles = [
        title
        for title, count in Counter(
            section.get("title") for section in sections
        ).items()
        if title and count > 1
    ]

    return {
        "lead_length": len(lead),
        "section_count": len(sections),
        "total_characters": total_characters,
        "empty_section_count": len(empty_sections),
        "empty_section_titles": [section.get("title") for section in empty_sections],
        "warnings_count": len(article.get("warnings") or []),
        "duplicate_section_text_count": len(duplicate_texts),
        "duplicate_section_title_count": len(duplicate_titles),
        "duplicate_section_titles": duplicate_titles,
    }


def compute_dataset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [compute_article_stats(record) for record in records]
    article_count = len(records)
    total_sections = sum(item["section_count"] for item in stats)
    total_empty_sections = sum(item["empty_section_count"] for item in stats)
    total_characters = sum(item["total_characters"] for item in stats)
    lead_lengths = [item["lead_length"] for item in stats]
    section_counts = [item["section_count"] for item in stats]

    return {
        "record_count": article_count,
        "breed_count": len({record.get("breed_id") for record in records}),
        "language_count": len({record.get("language") for record in records}),
        "total_sections": total_sections,
        "empty_sections": total_empty_sections,
        "articles_with_warnings": sum(
            1 for record in records if record.get("warnings")
        ),
        "total_characters": total_characters,
        "average_lead_length": (
            sum(lead_lengths) / article_count if article_count else 0
        ),
        "average_sections_per_article": (
            sum(section_counts) / article_count if article_count else 0
        ),
    }


def find_dataset_issues(records: list[dict[str, Any]]) -> dict[str, Any]:
    articles_with_empty_sections = []
    zero_length_sections = []
    short_sections = []
    duplicate_titles = []
    duplicate_texts = []
    missing_required_fields = []
    article_lengths = []
    section_distribution = []

    for article in records:
        article_key = article_key_for(article)
        stats = compute_article_stats(article)
        article_lengths.append(
            {
                "article": article_key,
                "breed_id": article.get("breed_id"),
                "language": article.get("language"),
                "title": article.get("title"),
                "total_characters": stats["total_characters"],
            }
        )
        section_distribution.append(
            {
                "article": article_key,
                "section_count": stats["section_count"],
            }
        )
        missing = sorted(REQUIRED_FIELDS - set(article.keys()))
        if missing:
            missing_required_fields.append({"article": article_key, "missing": missing})

        sections = article.get("sections") or []
        if stats["empty_section_count"]:
            articles_with_empty_sections.append(
                {
                    "article": article_key,
                    "empty_section_count": stats["empty_section_count"],
                    "empty_section_titles": stats["empty_section_titles"],
                }
            )

        section_texts = [section.get("text") or "" for section in sections]
        for text, count in Counter(section_texts).items():
            if text.strip() and count > 1:
                duplicate_texts.append(
                    {
                        "article": article_key,
                        "count": count,
                        "preview": text[:150],
                    }
                )

        section_titles = [section.get("title") for section in sections]
        for title, count in Counter(section_titles).items():
            if title and count > 1:
                duplicate_titles.append(
                    {"article": article_key, "title": title, "count": count}
                )

        for section in sections:
            text = section.get("text") or ""
            row = {
                "article": article_key,
                "index": section.get("index"),
                "title": section.get("title"),
                "text_length": len(text),
            }
            if not text.strip():
                zero_length_sections.append(row)
            elif len(text) < 50:
                short_sections.append(row)

    return {
        "articles_with_empty_sections": articles_with_empty_sections,
        "longest_articles": sorted(
            article_lengths,
            key=lambda item: item["total_characters"],
            reverse=True,
        )[:10],
        "shortest_articles": sorted(
            article_lengths,
            key=lambda item: item["total_characters"],
        )[:10],
        "zero_length_sections": zero_length_sections,
        "short_sections": short_sections,
        "duplicate_section_titles": duplicate_titles,
        "duplicate_section_texts": duplicate_texts,
        "missing_required_fields": missing_required_fields,
        "section_distribution": section_distribution,
    }


def load_review_overrides(path: str | Path) -> dict[str, Any]:
    overrides_path = Path(path)
    if not overrides_path.exists():
        return {}
    data = json.loads(overrides_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected review overrides object: {overrides_path}")
    return data


def save_review_overrides_atomic(
    overrides: dict[str, Any],
    path: str | Path,
) -> None:
    overrides_path = Path(path)
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = overrides_path.with_name(f".{overrides_path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(overrides, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, overrides_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def update_review_override(
    overrides: dict[str, Any],
    article_key: str,
    status: str,
    note: str,
    excluded_sections: list[str],
) -> dict[str, Any]:
    updated = dict(overrides)
    updated[article_key] = {
        "status": status,
        "note": note,
        "excluded_sections": excluded_sections,
    }
    return updated


def article_key_for(article: dict[str, Any]) -> str:
    return f"{article.get('breed_id')}:{article.get('language')}"


def article_label(article: dict[str, Any]) -> str:
    return (
        f"{article.get('breed_id')} | {article.get('language')} | "
        f"{article.get('title')}"
    )


def article_matches_query(article: dict[str, Any], query: str) -> bool:
    if not query.strip():
        return True
    needle = query.casefold()
    haystacks = [
        article.get("lead") or "",
        article.get("title") or "",
        article.get("breed_id") or "",
        article.get("language") or "",
    ]
    for section in article.get("sections") or []:
        haystacks.append(section.get("title") or "")
        haystacks.append(section.get("text") or "")
    return any(needle in value.casefold() for value in haystacks)


def section_matches_query(section: dict[str, Any], query: str) -> bool:
    if not query.strip():
        return True
    needle = query.casefold()
    return needle in str(section.get("title") or "").casefold() or needle in str(
        section.get("text") or ""
    ).casefold()


def filter_articles(
    records: list[dict[str, Any]],
    breed_id: str,
    language: str,
    only_warnings: bool,
    only_empty_sections: bool,
    query: str,
) -> list[dict[str, Any]]:
    filtered = records
    if breed_id != "all":
        filtered = [record for record in filtered if record.get("breed_id") == breed_id]
    if language != "all":
        filtered = [record for record in filtered if record.get("language") == language]
    if only_warnings:
        filtered = [record for record in filtered if record.get("warnings")]
    if only_empty_sections:
        filtered = [
            record
            for record in filtered
            if compute_article_stats(record)["empty_section_count"] > 0
        ]
    if query.strip():
        filtered = [record for record in filtered if article_matches_query(record, query)]
    return sorted(filtered, key=lambda item: (item.get("breed_id"), item.get("language")))


@st.cache_data
def load_jsonl_cached(path: str) -> list[dict[str, Any]]:
    return load_jsonl(path)


def render_summary(records: list[dict[str, Any]]) -> None:
    summary = compute_dataset_summary(records)
    cols = st.columns(4)
    cols[0].metric("Records", summary["record_count"])
    cols[1].metric("Breeds", summary["breed_count"])
    cols[2].metric("Languages", summary["language_count"])
    cols[3].metric("Sections", summary["total_sections"])

    cols = st.columns(4)
    cols[0].metric("Empty sections", summary["empty_sections"])
    cols[1].metric("Articles with warnings", summary["articles_with_warnings"])
    cols[2].metric("Total characters", summary["total_characters"])
    cols[3].metric("Avg lead length", f"{summary['average_lead_length']:.1f}")

    st.metric("Average sections per article", f"{summary['average_sections_per_article']:.1f}")


def render_article(article: dict[str, Any], query: str) -> None:
    stats = compute_article_stats(article)
    st.subheader(article.get("title", "Untitled article"))

    metadata = {
        "breed_id": article.get("breed_id"),
        "language": article.get("language"),
        "page_id": article.get("page_id"),
        "revision_id": article.get("revision_id"),
        "retrieved_at": article.get("retrieved_at"),
        "lead length": stats["lead_length"],
        "section count": stats["section_count"],
        "total characters": stats["total_characters"],
    }
    st.dataframe(pd.DataFrame([metadata]), hide_index=True, use_container_width=True)

    source_url = article.get("source_url")
    if source_url:
        st.markdown(f"Source: [{source_url}]({source_url})")

    warnings = article.get("warnings") or []
    if warnings:
        st.warning("\n".join(warnings))

    st.markdown("### Lead")
    if query.strip() and query.casefold() in (article.get("lead") or "").casefold():
        st.info("Search query matches the lead.")
    st.write(article.get("lead") or "")

    hide_empty_sections = st.checkbox("Hide empty sections", value=False)
    sections = article.get("sections") or []
    visible_sections = [
        section
        for section in sections
        if (not hide_empty_sections or (section.get("text") or "").strip())
        and section_matches_query(section, query)
    ]

    st.markdown("### Sections")
    for section in visible_sections:
        text = section.get("text") or ""
        prefix = ""
        if "level" in section:
            prefix = "  " * max(int(section.get("level") or 1) - 1, 0)
        title = section.get("title") or "Untitled"
        expander_title = (
            f"{prefix}[{section.get('index')}] {title} — {len(text)} characters"
        )
        with st.expander(expander_title, expanded=False):
            if "level" in section or "parent_index" in section:
                st.caption(
                    f"level={section.get('level')} | "
                    f"parent_index={section.get('parent_index')}"
                )
            if not text.strip():
                st.info("Empty container section")
            else:
                if query.strip() and section_matches_query(section, query):
                    st.info("Search query matches this section.")
                st.write(text)

    st.markdown("### Section diagnostics")
    rows = []
    has_level = any("level" in section for section in sections)
    has_parent_index = any("parent_index" in section for section in sections)
    for section in sections:
        text = section.get("text") or ""
        row = {
            "index": section.get("index"),
            "title": section.get("title"),
            "text length": len(text),
            "is empty": not bool(text.strip()),
            "preview": text[:150],
        }
        if has_level:
            row["level"] = section.get("level")
        if has_parent_index:
            row["parent index"] = section.get("parent_index")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("Raw article JSON"):
        st.json(article)


def render_compare(records: list[dict[str, Any]], breed_id: str) -> None:
    breed_articles = {
        article.get("language"): article
        for article in records
        if article.get("breed_id") == breed_id
    }
    cols = st.columns(2)
    for column, language in zip(cols, ("ru", "en"), strict=False):
        article = breed_articles.get(language)
        with column:
            st.markdown(f"### {language.upper()}")
            if not article:
                st.warning("Article is missing.")
                continue
            stats = compute_article_stats(article)
            st.write(f"**Title:** {article.get('title')}")
            st.write(f"**Lead length:** {stats['lead_length']}")
            st.write(f"**Sections:** {stats['section_count']}")
            st.write(f"**Total characters:** {stats['total_characters']}")
            st.markdown("**Section titles**")
            for section in article.get("sections") or []:
                st.write(f"- {section.get('title')}")


def render_diagnostics(records: list[dict[str, Any]]) -> None:
    issues = find_dataset_issues(records)
    st.subheader("Dataset diagnostics")
    for title, key in (
        ("Articles with empty sections", "articles_with_empty_sections"),
        ("Longest articles", "longest_articles"),
        ("Shortest articles", "shortest_articles"),
        ("Zero-length sections", "zero_length_sections"),
        ("Sections shorter than 50 characters", "short_sections"),
        ("Duplicate section titles", "duplicate_section_titles"),
        ("Duplicate section texts", "duplicate_section_texts"),
        ("Missing required fields", "missing_required_fields"),
        ("Section count distribution", "section_distribution"),
    ):
        with st.expander(title, expanded=key in {"articles_with_empty_sections"}):
            rows = issues[key]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.success("No issues found.")


def render_review_form(article: dict[str, Any], overrides_path: Path) -> None:
    st.subheader("Manual review override")
    article_key = article_key_for(article)
    overrides = load_review_overrides(overrides_path)
    current = overrides.get(article_key, {})
    sections = article.get("sections") or []
    section_titles = [section.get("title") for section in sections if section.get("title")]

    status = st.selectbox(
        "Review status",
        REVIEW_STATUSES,
        index=REVIEW_STATUSES.index(current.get("status", "not_reviewed"))
        if current.get("status", "not_reviewed") in REVIEW_STATUSES
        else 0,
    )
    note = st.text_area("Review note", value=current.get("note", ""))
    excluded_sections = st.multiselect(
        "Sections to exclude later",
        section_titles,
        default=[
            title for title in current.get("excluded_sections", []) if title in section_titles
        ],
    )

    if st.button("Save review override"):
        updated = update_review_override(
            overrides,
            article_key=article_key,
            status=status,
            note=note,
            excluded_sections=excluded_sections,
        )
        save_review_overrides_atomic(updated, overrides_path)
        st.success(f"Saved override for {article_key}")


def main() -> None:
    st.set_page_config(
        page_title="Wikipedia Articles Viewer",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Wikipedia Articles Viewer")
    st.caption("Developer-only viewer for staged Wikipedia article inspection.")

    articles_path = Path(
        st.sidebar.text_input("Articles JSONL path", str(DEFAULT_ARTICLES_PATH))
    )
    overrides_path = Path(
        st.sidebar.text_input("Review overrides path", str(DEFAULT_OVERRIDES_PATH))
    )

    try:
        records = load_jsonl_cached(str(articles_path))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    if not records:
        st.warning("No article records found.")
        return

    breed_options = ["all"] + sorted({record.get("breed_id") for record in records})
    language_options = ["all"] + sorted({record.get("language") for record in records})

    breed_id = st.sidebar.selectbox("breed_id", breed_options)
    language = st.sidebar.selectbox("language", language_options)
    only_warnings = st.sidebar.checkbox("Only articles with warnings")
    only_empty_sections = st.sidebar.checkbox("Only articles with empty sections")
    query = st.sidebar.text_input("Search in title and article text")

    filtered = filter_articles(
        records,
        breed_id=breed_id,
        language=language,
        only_warnings=only_warnings,
        only_empty_sections=only_empty_sections,
        query=query,
    )

    if not filtered:
        st.warning("No articles match the current filters.")
        render_summary(records)
        render_diagnostics(records)
        return

    article_options = {article_label(article): article for article in filtered}
    selected_label = st.sidebar.selectbox("Article", tuple(article_options.keys()))
    selected_article = article_options[selected_label]
    compare_mode = st.sidebar.checkbox("Compare RU / EN")

    tab_article, tab_diagnostics = st.tabs(["Article", "Dataset diagnostics"])

    with tab_article:
        render_summary(records)
        if compare_mode:
            compare_breed = selected_article.get("breed_id")
            render_compare(records, compare_breed)
        else:
            render_article(selected_article, query)
            render_review_form(selected_article, overrides_path)

    with tab_diagnostics:
        render_diagnostics(records)


if __name__ == "__main__":
    main()
