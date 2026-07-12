from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.catapi_retriever import retrieve_catapi_context


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_british_alias() -> None:
    result = retrieve_catapi_context("Чем британские котики отличаются от обычных?")

    assert_true(result["strategy"] == "alias_exact_breed", result)
    assert_true(result["detected_breed"] == "British Shorthair", result)
    assert_true(bool(result["results"]), result)


def check_sphynx_alias() -> None:
    result = retrieve_catapi_context("Как ухаживать за сфинксом?")

    assert_true(result["strategy"] == "alias_exact_breed", result)
    assert_true(result["detected_breed"] == "Sphynx", result)
    assert_true(bool(result["results"]), result)


def check_structured_fields() -> None:
    result = retrieve_catapi_context("Какая кошка большая, спокойная и ласковая?")
    retrieved_breeds = {item["breed_name"] for item in result["results"]}

    assert_true(result["strategy"] == "structured_fields", result)
    assert_true(bool(result["results"]), result)
    assert_true(
        bool({"British Shorthair", "Ragamuffin", "Ragdoll"} & retrieved_breeds),
        result,
    )


def check_no_match_guard() -> None:
    result = retrieve_catapi_context("Какая кошка умеет чинить ноутбук?")

    assert_true(result["strategy"] == "no_match", result)
    assert_true(result["results"] == [], result)
    assert_true(bool(result["warning"]), result)
    assert_true(result["detected_breed"] is None, result)


def main() -> None:
    checks = (
        check_british_alias,
        check_sphynx_alias,
        check_structured_fields,
        check_no_match_guard,
    )

    for check in checks:
        check()
        print(f"OK: {check.__name__}")

    print("All CatAPI retriever checks passed.")


if __name__ == "__main__":
    main()
