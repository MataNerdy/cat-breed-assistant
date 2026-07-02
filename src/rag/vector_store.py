import logging
from pathlib import Path

import chromadb


LOGGER = logging.getLogger(__name__)
CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "cat_breed_chunks"


def get_chroma_client(path: Path = CHROMA_PATH) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(path))


def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(name=COLLECTION_NAME)


def get_existing_collection(client: chromadb.PersistentClient):
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        LOGGER.warning(
            "Chroma collection '%s' was not found. Build it with "
            "`python scripts/build_rag_index.py`.",
            COLLECTION_NAME,
        )
        return None
