from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from src.rag.preprocess import read_jsonl
from src.rag.vector_store import CHROMA_PATH, get_chroma_client, get_or_create_collection


CHUNKS_PATH = Path("data/processed/rag_chunks.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run `python scripts/download_hf_dataset.py` first."
        )

    chunks = read_jsonl(CHUNKS_PATH)
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")

    model = SentenceTransformer(MODEL_NAME)
    client = get_chroma_client(CHROMA_PATH)
    collection = get_or_create_collection(client)

    # Rebuild collection content in a transparent way.
    existing = collection.get()
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        documents = [chunk["text"] for chunk in batch]
        embeddings = model.encode(documents, convert_to_numpy=True).tolist()

        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[chunk["metadata"] for chunk in batch],
        )

        print(f"Indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    print(f"Saved Chroma index to {CHROMA_PATH}")


if __name__ == "__main__":
    main()
