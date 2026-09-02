from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cat_breed_assistant.evaluation.retrieval.io import read_jsonl  # noqa: E402
from src.cat_breed_assistant.evaluation.retrieval.migration import (  # noqa: E402
    normalize_legacy_question_type,
)
from src.cat_breed_assistant.evaluation.retrieval.question_types import (  # noqa: E402
    QUESTION_TYPE_VALUES,
)
from src.cat_breed_assistant.evaluation.retrieval.review import (  # noqa: E402
    HumanReviewRecord,
    ReviewDecision,
    load_review_records,
    upsert_review_record,
    write_review_records_atomic,
)

CANDIDATES_PATH = Path("data/evaluation/retrieval/v1/pilot_candidates.jsonl")
REVIEWS_PATH = Path("data/evaluation/retrieval/v1/human_review_decisions.jsonl")
CHUNKS_PATH = Path("data/processed/knowledge_chunks.jsonl")


def _chunk_text(chunks: dict[str, dict], chunk_id: str) -> str:
    return str(chunks.get(chunk_id, {}).get("text", "Chunk not found."))


def _filter_options(records: list[dict], field_name: str) -> list[str]:
    return sorted({str(record.get(field_name, "")) for record in records if record.get(field_name)})


def _review_status(query_id: str, reviews: dict[str, HumanReviewRecord]) -> str:
    record = reviews.get(query_id)
    return record.decision.value if record else "unreviewed"


def _apply_filters(
    candidates: list[dict],
    reviews: dict[str, HumanReviewRecord],
    breed: str,
    question_type: str,
    difficulty: str,
    generator: str,
    validator: str,
    status: str,
) -> list[dict]:
    filtered = []
    for candidate in candidates:
        if breed != "all" and candidate.get("breed_id") != breed:
            continue
        if question_type != "all" and candidate.get("question_type") != question_type:
            continue
        if difficulty != "all" and candidate.get("difficulty") != difficulty:
            continue
        if generator != "all" and candidate.get("generator_provider") != generator:
            continue
        if validator != "all" and candidate.get("validator_provider") != validator:
            continue
        if status != "all" and _review_status(candidate["query_id"], reviews) != status:
            continue
        filtered.append(candidate)
    return filtered


def _save_review(candidate: dict, decision: ReviewDecision, reviews: dict[str, HumanReviewRecord]) -> None:
    query_id = candidate["query_id"]
    record = HumanReviewRecord(
        query_id=query_id,
        decision=decision,
        reviewer_note=st.session_state.get(f"reviewer_note_{query_id}", ""),
        edited_query=st.session_state.get(f"edited_query_{query_id}"),
        edited_answer=st.session_state.get(f"edited_answer_{query_id}"),
        edited_relevant_chunk_ids=[
            chunk_id.strip()
            for chunk_id in st.session_state.get(f"edited_relevant_chunk_ids_{query_id}", "").split(",")
            if chunk_id.strip()
        ],
        normalized_question_type=st.session_state.get(f"normalized_question_type_{query_id}"),
        edit_approved=bool(st.session_state.get(f"edit_approved_{query_id}", False)),
    )
    updated = upsert_review_record(reviews, record)
    write_review_records_atomic(REVIEWS_PATH, updated)
    st.session_state["saved"] = True
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Retrieval Candidate Review", layout="wide")
    st.title("Retrieval Evaluation Review")

    candidates = read_jsonl(CANDIDATES_PATH)
    chunks = {record["chunk_id"]: record for record in read_jsonl(CHUNKS_PATH)}
    reviews = load_review_records(REVIEWS_PATH)

    if not candidates:
        st.warning(f"No candidates found: `{CANDIDATES_PATH}`")
        return

    status_counts = Counter(_review_status(candidate["query_id"], reviews) for candidate in candidates)
    st.caption(
        " | ".join(
            f"{name}: {status_counts.get(name, 0)}"
            for name in ("unreviewed", "approved", "needs_edit", "rejected", "skipped")
        )
    )

    with st.sidebar:
        st.header("Filters")
        breed = st.selectbox("Breed", ["all", *_filter_options(candidates, "breed_id")])
        question_type = st.selectbox(
            "Question type",
            ["all", *_filter_options(candidates, "question_type")],
        )
        difficulty = st.selectbox("Difficulty", ["all", *_filter_options(candidates, "difficulty")])
        generator = st.selectbox("Generator", ["all", *_filter_options(candidates, "generator_provider")])
        validator = st.selectbox("Validator", ["all", *_filter_options(candidates, "validator_provider")])
        status = st.selectbox(
            "Review status",
            ["all", "unreviewed", "approved", "needs_edit", "rejected", "skipped"],
        )

    filtered = _apply_filters(candidates, reviews, breed, question_type, difficulty, generator, validator, status)
    if not filtered:
        st.info("No candidates match the selected filters.")
        return

    index = st.number_input("Candidate", min_value=1, max_value=len(filtered), value=1) - 1
    candidate = filtered[index]
    review = reviews.get(candidate["query_id"])
    migration = normalize_legacy_question_type(str(candidate.get("question_type", "")))

    st.subheader(f"{candidate['breed_name']} · {candidate['query_id']}")
    st.caption(
        f"{index + 1}/{len(filtered)} filtered · current review: {_review_status(candidate['query_id'], reviews)}"
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Question**")
        st.write(candidate["query"])
        st.markdown("**Answer**")
        st.write(candidate["answer"])
        st.markdown("**Evidence quote**")
        st.write(candidate["evidence_quote"])
        st.markdown("**Metadata**")
        st.json(
            {
                "breed_id": candidate["breed_id"],
                "chunk_id": candidate["chunk_id"],
                "question_type": candidate["question_type"],
                "difficulty": candidate["difficulty"],
                "generator": candidate["generator_provider"],
                "validator": candidate["validator_provider"],
                "migration": migration.__dict__,
            }
        )

    with right:
        st.markdown("**Relevant chunk IDs**")
        st.write(candidate["relevant_chunk_ids"])
        for chunk_id in candidate["relevant_chunk_ids"]:
            with st.expander(chunk_id, expanded=True):
                st.write(_chunk_text(chunks, chunk_id))

    st.divider()
    st.subheader("Review decision")
    default_type = (
        review.normalized_question_type
        if review and review.normalized_question_type
        else migration.normalized_question_type
        if migration.normalized_question_type
        else QUESTION_TYPE_VALUES[0]
    )
    st.selectbox(
        "Canonical question type",
        QUESTION_TYPE_VALUES,
        index=QUESTION_TYPE_VALUES.index(default_type),
        key=f"normalized_question_type_{candidate['query_id']}",
    )
    st.text_area(
        "Reviewer note",
        value=review.reviewer_note if review else "",
        key=f"reviewer_note_{candidate['query_id']}",
    )
    st.text_area(
        "Edited question",
        value=review.edited_query if review and review.edited_query else candidate["query"],
        key=f"edited_query_{candidate['query_id']}",
    )
    st.text_area(
        "Edited answer",
        value=review.edited_answer if review and review.edited_answer else candidate["answer"],
        key=f"edited_answer_{candidate['query_id']}",
    )
    st.text_input(
        "Edited relevant chunk IDs, comma-separated",
        value=", ".join(
            review.edited_relevant_chunk_ids
            if review and review.edited_relevant_chunk_ids
            else candidate["relevant_chunk_ids"]
        ),
        key=f"edited_relevant_chunk_ids_{candidate['query_id']}",
    )
    st.checkbox(
        "Use edited version in benchmark",
        value=review.edit_approved if review else False,
        key=f"edit_approved_{candidate['query_id']}",
    )

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Approve", use_container_width=True):
        _save_review(candidate, ReviewDecision.APPROVED, reviews)
    if col2.button("Needs edit", use_container_width=True):
        _save_review(candidate, ReviewDecision.NEEDS_EDIT, reviews)
    if col3.button("Reject", use_container_width=True):
        _save_review(candidate, ReviewDecision.REJECTED, reviews)
    if col4.button("Skip", use_container_width=True):
        _save_review(candidate, ReviewDecision.SKIPPED, reviews)


if __name__ == "__main__":
    main()
