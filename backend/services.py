from src.cat_knowledge import generate_mock_answer
from src.breed_retriever import build_breed_context
from src.gemini_client import generate_gemini_answer
from src.llm_client import generate_llm_answer

from .schemas import AskResponse, AnswerMode


def generate_answer(question: str, mode: AnswerMode) -> AskResponse:
    breed_context = build_breed_context(question)

    if mode == "mock":
        answer = generate_mock_answer(question, breed_context)
    elif mode == "openai":
        answer = generate_llm_answer(question, breed_context)
    elif mode == "gemini":
        answer = generate_gemini_answer(question, breed_context)
    else:
        raise ValueError(f"Неподдерживаемый режим ответа: {mode}")

    return AskResponse(
        answer=answer,
        breed=breed_context["breed"],
        mode=mode,
        breed_context=breed_context,
    )
