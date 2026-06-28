import json
from functools import lru_cache
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "breed_profiles.json"
UNKNOWN_BREED = "Unknown breed"


@lru_cache(maxsize=1)
def _load_breed_profiles_cached() -> tuple[dict, ...]:
    with DATA_PATH.open(encoding="utf-8") as file:
        return tuple(json.load(file))


def load_breed_profiles() -> list[dict]:
    """Load local cat breed profiles from JSON."""
    return [dict(profile) for profile in _load_breed_profiles_cached()]


def _search_terms(profile: dict) -> list[str]:
    terms = [profile["breed"], *profile.get("aliases", [])]
    return sorted({term.strip().casefold() for term in terms if term.strip()})


def _match_position(question: str, profile: dict) -> int | None:
    positions = [
        question.find(term)
        for term in _search_terms(profile)
        if term and question.find(term) != -1
    ]

    if not positions:
        return None

    return min(positions)


def detect_breed(question: str, profiles: list[dict]) -> dict | None:
    """Detect the first mentioned breed by canonical name or aliases."""
    normalized_question = question.strip().casefold()
    matches = []

    for profile in profiles:
        position = _match_position(normalized_question, profile)
        if position is not None:
            matches.append((position, profile["breed"], profile))

    if not matches:
        return None

    return sorted(matches, key=lambda item: (item[0], item[1]))[0][2]


def _detect_all_breeds(question: str, profiles: list[dict]) -> list[dict]:
    normalized_question = question.strip().casefold()
    matches = []

    for profile in profiles:
        position = _match_position(normalized_question, profile)
        if position is not None:
            matches.append((position, profile["breed"], profile))

    return [match[2] for match in sorted(matches, key=lambda item: (item[0], item[1]))]


def build_breed_context(question: str) -> dict:
    """Build a simple local RAG-lite context for the user question."""
    profiles = load_breed_profiles()
    detected_profile = detect_breed(question, profiles)
    mentioned_profiles = _detect_all_breeds(question, profiles)
    is_fallback = detected_profile is None

    if is_fallback:
        return {
            "breed": UNKNOWN_BREED,
            "aliases": [],
            "origin": "",
            "appearance": [],
            "temperament": [],
            "care": [],
            "health_notes": [
                "This is not medical advice.",
                "Owners should consult a veterinarian for care or health questions.",
            ],
            "fun_facts": [],
            "differs_from_other_breeds": [],
            "is_fallback": True,
            "fallback_note": (
                "Порода не найдена в локальной базе. Сейчас доступны профили: "
                f"{', '.join(profile['breed'] for profile in profiles)}."
            ),
            "available_breeds": [profile["breed"] for profile in profiles],
            "mentioned_breeds": [],
        }

    primary_profile = detected_profile

    context = dict(primary_profile)
    context["is_fallback"] = is_fallback
    context["fallback_note"] = ""
    context["available_breeds"] = [profile["breed"] for profile in profiles]
    context["mentioned_breeds"] = [
        {
            "breed": profile["breed"],
            "appearance": profile.get("appearance", []),
            "temperament": profile.get("temperament", []),
            "care": profile.get("care", []),
            "differs_from_other_breeds": profile.get(
                "differs_from_other_breeds", []
            ),
        }
        for profile in mentioned_profiles
    ]

    return context
