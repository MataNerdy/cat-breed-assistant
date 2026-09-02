from __future__ import annotations

from typing import Any

from src.rag.catapi_retriever import retrieve_catapi_context


def retrieve_structured_context(question: str, top_k: int = 3) -> dict[str, Any]:
    """Retrieve CatAPI context with the current deterministic structured retriever."""
    return retrieve_catapi_context(question, top_k=top_k)

