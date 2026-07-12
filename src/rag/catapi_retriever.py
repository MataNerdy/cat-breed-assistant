from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS_PATH = Path("data/processed/catapi_chunks.jsonl")

BREED_ALIASES = {
    "британ": "British Shorthair",
    "британск": "British Shorthair",
    "british": "British Shorthair",
    "british shorthair": "British Shorthair",
    "мейн": "Maine Coon",
    "мейн кун": "Maine Coon",
    "мейн-кун": "Maine Coon",
    "maine": "Maine Coon",
    "maine coon": "Maine Coon",
    "сфинкс": "Sphynx",
    "sphynx": "Sphynx",
    "сиам": "Siamese",
    "сиамск": "Siamese",
    "siamese": "Siamese",
    "перс": "Persian",
    "персид": "Persian",
    "persian": "Persian",
    "бенгал": "Bengal",
    "bengal": "Bengal",
    "рэгдолл": "Ragdoll",
    "ragdoll": "Ragdoll",
    "абиссин": "Abyssinian",
    "abyssinian": "Abyssinian",
}

STOPWORDS = {
    "the",
    "and",
    "cat",
    "cats",
    "кошка",
    "кошки",
    "кот",
    "коты",
    "котик",
    "котики",
    "какая",
    "какой",
    "какие",
    "как",
    "про",
    "для",
    "что",
    "чем",
    "или",
    "это",
}


def load_catapi_chunks(path: str | Path = DEFAULT_CHUNKS_PATH) -> list[dict]:
    """Load prepared CatAPI chunks from JSONL."""
    chunks_path = Path(path)
    if not chunks_path.exists():
        return []

    chunks = []
    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {chunks_path}: {exc}"
                ) from exc
    return chunks


@lru_cache(maxsize=1)
def _cached_chunks() -> tuple[dict, ...]:
    return tuple(load_catapi_chunks())


def detect_breed_alias(question: str) -> str | None:
    """Detect an explicitly mentioned breed from Russian or English aliases."""
    question_lower = question.lower().replace("ё", "е")
    for alias, breed_name in sorted(BREED_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in question_lower:
            return breed_name
    return None


def _find_chunk_by_breed(breed_name: str, chunks: list[dict]) -> dict | None:
    target = breed_name.lower()
    for chunk in chunks:
        if str(chunk.get("breed_name", "")).lower() == target:
            return chunk
    return None


def _extract_field(text: str, field_name: str) -> str | None:
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+?:|\Z)"
    match = re.search(pattern, text, flags=re.S)
    return match.group(1).strip() if match else None


def _extract_int(text: str, field_name: str) -> int | None:
    value = _extract_field(text, field_name)
    if value is None:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _parse_weight_kg(text: str) -> tuple[int, int] | None:
    value = _extract_field(text, "Weight")
    if not value:
        return None
    match = re.search(r"(\d+)\s*-\s*(\d+)\s*kg", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace("ё", "е")
    tokens = re.findall(r"[a-zа-я0-9-]+", normalized)
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _profile_from_chunk(chunk: dict) -> dict[str, Any]:
    text = chunk.get("text", "")
    metadata = chunk.get("metadata") or {}
    return {
        "breed_id": chunk.get("breed_id"),
        "breed_name": chunk.get("breed_name"),
        "source": chunk.get("source", "thecatapi"),
        "text": text,
        "metadata": metadata,
        "description": _extract_field(text, "Description") or "",
        "temperament": _extract_field(text, "Temperament") or "",
        "weight_kg": _parse_weight_kg(text),
        "grooming_level": _extract_int(text, "Grooming level"),
        "energy_level": _extract_int(text, "Energy level"),
        "health_issues_score": _extract_int(text, "Health issues score"),
        "child_friendly_score": _extract_int(text, "Child friendly score"),
        "dog_friendly_score": _extract_int(text, "Dog friendly score"),
        "stranger_friendly_score": _extract_int(text, "Stranger friendly score"),
        "intelligence_score": _extract_int(text, "Intelligence score"),
        "hairless": (_extract_field(text, "Hairless") or "").lower(),
        "shedding_level": _extract_int(text, "Shedding level"),
        "social_needs_score": _extract_int(text, "Social needs score"),
        "vocalisation_score": _extract_int(text, "Vocalisation score"),
        "hypoallergenic": _extract_int(text, "Hypoallergenic"),
    }


def _structured_score(question: str, profile: dict[str, Any]) -> int:
    q = question.lower().replace("ё", "е")
    score = 0

    text_blob = " ".join(
        (
            str(profile.get("breed_name") or ""),
            str(profile.get("description") or ""),
            str(profile.get("temperament") or ""),
        )
    )
    score += len(_tokenize(question) & _tokenize(text_blob))

    weight = profile.get("weight_kg")
    if weight:
        _, max_weight = weight
        if any(word in q for word in ("large", "big", "гигант", "больш", "крупн")):
            score += 5 if max_weight >= 8 else 2 if max_weight >= 6 else 0
        if any(word in q for word in ("small", "tiny", "маленьк", "небольш")):
            score += 4 if max_weight <= 5 else 0

    temperament = str(profile.get("temperament") or "").lower()
    if any(word in q for word in ("gentle", "calm", "quiet", "спокойн", "тих", "ласков")):
        for trait in ("gentle", "calm", "quiet", "easy going", "patient", "affectionate"):
            if trait in temperament:
                score += 3

    if any(word in q for word in ("active", "energetic", "playful", "активн", "энергич", "игрив")):
        if (profile.get("energy_level") or 0) >= 4:
            score += 4
        for trait in ("active", "energetic", "playful", "lively"):
            if trait in temperament:
                score += 2

    if any(word in q for word in ("friendly", "social", "общительн", "дружелюб", "любит людей")):
        if (profile.get("social_needs_score") or 0) >= 4:
            score += 3
        for trait in ("friendly", "social", "sociable", "loving"):
            if trait in temperament:
                score += 2

    if any(word in q for word in ("vocal", "talkative", "разговорч", "болтлив", "мяука")):
        if (profile.get("vocalisation_score") or 0) >= 4:
            score += 6

    if any(word in q for word in ("intelligent", "clever", "умн", "сообразительн")):
        if (profile.get("intelligence_score") or 0) >= 4:
            score += 4

    if any(word in q for word in ("grooming", "care", "уход", "шерсть", "вычес")):
        grooming = profile.get("grooming_level") or 0
        score += 5 if grooming >= 4 else 2 if grooming >= 2 else 0

    if any(word in q for word in ("shedding", "линяет", "линька")):
        if (profile.get("shedding_level") or 0) >= 4:
            score += 4

    if any(word in q for word in ("hairless", "без шерсти", "лыс", "гол")):
        if profile.get("hairless") in {"yes", "1", "true"}:
            score += 8

    if any(word in q for word in ("hypoallergenic", "аллерг")):
        if profile.get("hypoallergenic") == 1:
            score += 5

    return score


def _result_from_chunk(chunk: dict, rank: int, score: int | float | None) -> dict:
    metadata = chunk.get("metadata") or {}
    return {
        "rank": rank,
        "score": score,
        "breed_id": chunk.get("breed_id", ""),
        "breed_name": chunk.get("breed_name", ""),
        "source": chunk.get("source", "thecatapi"),
        "text": chunk.get("text", ""),
        "metadata": metadata,
    }


def retrieve_catapi_context(question: str, top_k: int = 3) -> dict:
    """Retrieve CatAPI context using alias detection and structured field scoring."""
    chunks = list(_cached_chunks())
    if not chunks:
        return {
            "strategy": "no_match",
            "detected_breed": None,
            "warning": "No relevant CatAPI context found.",
            "results": [],
        }

    detected_breed = detect_breed_alias(question)
    if detected_breed:
        chunk = _find_chunk_by_breed(detected_breed, chunks)
        if chunk:
            return {
                "strategy": "alias_exact_breed",
                "detected_breed": detected_breed,
                "warning": None,
                "results": [_result_from_chunk(chunk, rank=1, score=None)],
            }

    scored = []
    for chunk in chunks:
        score = _structured_score(question, _profile_from_chunk(chunk))
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = [
        _result_from_chunk(chunk, rank=index, score=score)
        for index, (score, chunk) in enumerate(scored[:top_k], start=1)
    ]

    if not results:
        return {
            "strategy": "no_match",
            "detected_breed": None,
            "warning": "No relevant CatAPI context found.",
            "results": [],
        }

    return {
        "strategy": "structured_fields",
        "detected_breed": results[0]["breed_name"] if results else None,
        "warning": None,
        "results": results,
    }
