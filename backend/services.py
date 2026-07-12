from src.cat_knowledge import generate_mock_answer
from src.breed_retriever import build_breed_context
from src.gemini_client import generate_gemini_answer
from src.llm_client import generate_llm_answer
from src.mistral_client import generate_mistral_answer
from src.rag.catapi_retriever import retrieve_catapi_context

from .schemas import AskResponse, AnswerMode


def generate_answer(
    question: str,
    mode: AnswerMode,
    use_rag: bool = False,
) -> AskResponse:
    breed_context = build_breed_context(question)
    retrieval = (
        retrieve_catapi_context(question, top_k=3)
        if use_rag
        else {
            "strategy": None,
            "detected_breed": None,
            "warning": None,
            "results": [],
        }
    )
    retrieved_context = retrieval["results"]

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
        breed=retrieval.get("detected_breed") or breed_context["breed"],
        mode=mode,
        breed_context=breed_context,
        retrieved_context=retrieved_context,
        rag_enabled=use_rag,
        retrieval_strategy=retrieval.get("strategy"),
        detected_breed=retrieval.get("detected_breed"),
        warning=retrieval.get("warning"),
        image_url=_top_metadata(retrieved_context).get("image_url"),
        reference_image_id=_top_metadata(retrieved_context).get("reference_image_id"),
    )


def _top_metadata(retrieved_context: list[dict]) -> dict:
    if not retrieved_context:
        return {}
    metadata = retrieved_context[0].get("metadata") or {}
    return metadata
