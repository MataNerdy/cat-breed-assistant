from src.cat_breed_assistant.evaluation.retrieval.migration import (
    normalize_legacy_question_type,
)


def test_canonical_question_type_is_preserved() -> None:
    result = normalize_legacy_question_type("temperament")

    assert result.normalized_question_type == "temperament"
    assert result.status == "canonical"
    assert result.review_required is False


def test_legacy_question_type_is_normalized() -> None:
    result = normalize_legacy_question_type("Physical Trait")

    assert result.normalized_question_type == "physical_characteristic"
    assert result.status == "normalized"
    assert result.review_required is False
    assert result.raw_question_type == "Physical Trait"


def test_ambiguous_question_type_requires_review() -> None:
    result = normalize_legacy_question_type("numeric_rating")

    assert result.normalized_question_type is None
    assert result.status == "needs_review"
    assert result.review_required is True

