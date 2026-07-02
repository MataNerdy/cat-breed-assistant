from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from src.rag.preprocess import (
    SOURCE_DATASET,
    extract_assistant_chunks,
    extract_image_path,
    write_jsonl,
)


SPLIT = "train"
RAW_PATH = Path("data/raw/train_examples.jsonl")
CHUNKS_PATH = Path("data/processed/rag_chunks.jsonl")


def main() -> None:
    dataset = load_dataset(SOURCE_DATASET, split=SPLIT)
    raw_rows = []
    chunks = []

    for row_index, example in enumerate(dataset):
        source_id = str(example.get("id") or row_index)
        raw_rows.append(
            {
                "id": source_id,
                "image": extract_image_path(example.get("image")),
                "conversations": example.get("conversations"),
            }
        )
        chunks.extend(extract_assistant_chunks(example, SPLIT, row_index))

    raw_count = write_jsonl(RAW_PATH, raw_rows)
    chunk_count = write_jsonl(CHUNKS_PATH, chunks)

    print(f"Saved {raw_count} raw examples to {RAW_PATH}")
    print(f"Saved {chunk_count} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
