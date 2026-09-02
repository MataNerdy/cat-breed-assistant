from __future__ import annotations

from typing import Any, Protocol


class Retriever(Protocol):
    """Small protocol for future retrieval implementations."""

    def retrieve(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Return retrieval metadata and ranked context records."""

