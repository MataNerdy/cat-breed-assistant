from src.cat_knowledge import generate_mock_answer
from src.gemini_client import generate_gemini_answer
from src.llm_client import generate_llm_answer

from .schemas import AskResponse, AnswerMode


def _api_breed_name(breed_context: dict) -> str:
    breed = breed_context["breed"]
    if "британ" in breed.casefold():
        return "British Shorthair"

    return breed


def generate_answer(question: str, mode: AnswerMode) -> AskResponse:
    breed_context = generate_mock_answer(question)

    if mode == "mock":
        answer = breed_context["answer"]
    elif mode == "openai":
        answer = generate_llm_answer(question, breed_context)
    elif mode == "gemini":
        answer = generate_gemini_answer(question, breed_context)
    else:
        raise ValueError(f"Неподдерживаемый режим ответа: {mode}")

    return AskResponse(
        answer=answer,
        breed=_api_breed_name(breed_context),
        mode=mode,
    )
