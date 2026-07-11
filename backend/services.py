from src.cat_knowledge import generate_mock_answer
from src.breed_retriever import build_breed_context
from src.gemini_client import generate_gemini_answer
from src.llm_client import generate_llm_answer
from src.mistral_client import generate_mistral_answer

from .schemas import AskResponse, AnswerMode


def generate_answer(
    question: str,
    mode: AnswerMode,
    use_rag: bool = False,
) -> AskResponse:
    breed_context = build_breed_context(question)
    # Vector retrieval is temporarily disabled while the project moves from the
    # old dataset prototype to a future CatAPI/Wikipedia data layer.
    retrieved_context = []

    if mode == "mock":
        answer = generate_mock_answer(question, breed_context, retrieved_context)
    elif mode == "openai":
        answer = generate_llm_answer(question, breed_context, retrieved_context)
    elif mode == "gemini":
        answer = generate_gemini_answer(question, breed_context, retrieved_context)
    elif mode == "mistral":
        answer = generate_mistral_answer(question, breed_context, retrieved_context)
    else:
        raise ValueError(f"Неподдерживаемый режим ответа: {mode}")

    return AskResponse(
        answer=answer,
        breed=breed_context["breed"],
        mode=mode,
        breed_context=breed_context,
        retrieved_context=retrieved_context,
        rag_enabled=use_rag,
    )
