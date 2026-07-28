from __future__ import annotations

from enum import Enum


class RetrievalQuestionType(str, Enum):
    ORIGIN = "origin"
    LIFESPAN = "lifespan"
    TEMPERAMENT = "temperament"
    PHYSICAL_CHARACTERISTIC = "physical_characteristic"
    HISTORY = "history"
    ALIAS = "alias"
    BEHAVIOR = "behavior"
    HEALTH = "health"
    BREED_DEVELOPMENT = "breed_development"


QUESTION_TYPE_VALUES = tuple(item.value for item in RetrievalQuestionType)


SAFE_QUESTION_TYPE_MAPPING = {
    "origin_country": RetrievalQuestionType.ORIGIN,
    "country_of_origin": RetrievalQuestionType.ORIGIN,
    "place_of_origin": RetrievalQuestionType.ORIGIN,
    "location": RetrievalQuestionType.ORIGIN,
    "life_expectancy": RetrievalQuestionType.LIFESPAN,
    "life_span": RetrievalQuestionType.LIFESPAN,
    "lifespan_range": RetrievalQuestionType.LIFESPAN,
    "appearance": RetrievalQuestionType.PHYSICAL_CHARACTERISTIC,
    "physical_trait": RetrievalQuestionType.PHYSICAL_CHARACTERISTIC,
    "physical trait": RetrievalQuestionType.PHYSICAL_CHARACTERISTIC,
    "coat": RetrievalQuestionType.PHYSICAL_CHARACTERISTIC,
    "historical_fact": RetrievalQuestionType.HISTORY,
    "development_year": RetrievalQuestionType.BREED_DEVELOPMENT,
    "breed_creation": RetrievalQuestionType.BREED_DEVELOPMENT,
    "breed_recognition": RetrievalQuestionType.BREED_DEVELOPMENT,
    "alternative_name": RetrievalQuestionType.ALIAS,
    "alternative_names": RetrievalQuestionType.ALIAS,
    "alt_name": RetrievalQuestionType.ALIAS,
}


def normalize_question_type(raw_question_type: str) -> RetrievalQuestionType | None:
    normalized = raw_question_type.strip().casefold().replace("-", "_")
    normalized = "_".join(normalized.split())
    try:
        return RetrievalQuestionType(normalized)
    except ValueError:
        return SAFE_QUESTION_TYPE_MAPPING.get(normalized)


def all_question_type_counts(default: int = 0) -> dict[str, int]:
    return {question_type: default for question_type in QUESTION_TYPE_VALUES}
