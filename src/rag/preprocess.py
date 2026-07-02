import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_DATASET = "YZhao09/cat_breed_meowticulous"


def extract_image_path(image: Any) -> str:
    """Extract a stable image path from common Hugging Face image objects."""
    if image is None:
        return ""

    if isinstance(image, str):
        return image

    if isinstance(image, dict):
        for key in ("path", "filename", "file_name"):
            value = image.get(key)
            if value:
                return str(value)
        return ""

    for attr in ("filename", "path"):
        value = getattr(image, attr, None)
        if value:
            return str(value)

    return ""


def extract_breed_from_image_path(image_path: str) -> str:
    """Infer breed from paths like .../British_Shorthair/British_Shorthair_147.jpg."""
    parts = Path(image_path).parts
    if len(parts) >= 2:
        candidate = parts[-2]
    elif image_path:
        candidate = Path(image_path).stem.rsplit("_", 1)[0]
    else:
        candidate = "Unknown breed"

    return candidate.replace("_", " ").strip() or "Unknown breed"


def _conversation_text(turn: Any) -> str:
    if not isinstance(turn, dict):
        return ""

    for key in ("value", "content", "text", "message"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _is_assistant_turn(turn: Any, turn_index: int) -> bool:
    if not isinstance(turn, dict):
        return False

    role = str(
        turn.get("from")
        or turn.get("role")
        or turn.get("speaker")
        or turn.get("author")
        or ""
    ).casefold()

    if role:
        return role in {"assistant", "gpt", "model", "bot"}

    # Some conversation datasets alternate user/assistant without explicit roles.
    return turn_index % 2 == 1


def extract_assistant_chunks(example: dict, split: str, row_index: int) -> list[dict]:
    source_id = str(example.get("id") or row_index)
    image_path = extract_image_path(example.get("image"))
    breed = extract_breed_from_image_path(image_path)
    conversations = example.get("conversations") or []
    chunks = []

    if not isinstance(conversations, list):
        return chunks

    for turn_index, turn in enumerate(conversations):
        if not _is_assistant_turn(turn, turn_index):
            continue

        text = _conversation_text(turn)
        if not text:
            continue

        chunks.append(
            {
                "chunk_id": f"{split}_{source_id}_turn_{turn_index}",
                "text": text,
                "metadata": {
                    "source_id": source_id,
                    "breed": breed,
                    "image_path": image_path,
                    "turn_index": turn_index,
                    "source_dataset": SOURCE_DATASET,
                    "split": split,
                },
            }
        )

    return chunks


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
