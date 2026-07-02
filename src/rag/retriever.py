import logging
from typing import Any

from src.rag.vector_store import CHROMA_PATH, get_chroma_client, get_existing_collection


LOGGER = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL: Any | None = None


def _embedding_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _MODEL


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 5,
    breed: str | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from local ChromaDB, returning [] if unavailable."""
    if not CHROMA_PATH.exists():
        LOGGER.warning(
            "Chroma index path '%s' was not found. Build it with "
            "`python scripts/build_rag_index.py`.",
            CHROMA_PATH,
        )
        return []

    try:
        client = get_chroma_client(CHROMA_PATH)
        collection = get_existing_collection(client)
        if collection is None:
            return []

        query_embedding = _embedding_model().encode([question], convert_to_numpy=True)[
            0
        ].tolist()
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if breed and breed != "Unknown breed":
            query_kwargs["where"] = {"breed": breed}

        result = collection.query(
            **query_kwargs,
        )
    except Exception as error:
        LOGGER.warning("RAG retrieval failed: %s", error)
        return []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "text": document,
            "metadata": metadata or {},
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
