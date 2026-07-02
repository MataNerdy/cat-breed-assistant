import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MODE_LABELS = {
    "Mock mode": "mock",
    "OpenAI mode": "openai",
    "Gemini mode": "gemini",
    "Mistral mode": "mistral",
}


def ask_backend(question: str, mode: str, use_rag: bool) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question, "mode": mode, "use_rag": use_rag},
        timeout=30,
    )

    if response.ok:
        return response.json()

    try:
        detail = response.json().get("detail", "Backend вернул ошибку.")
    except ValueError:
        detail = response.text or "Backend вернул ошибку."

    raise RuntimeError(detail)


st.set_page_config(
    page_title="Cat Breed Assistant",
    page_icon="🐱",
    layout="centered",
)


st.title("Cat Breed Assistant")
st.write(
    "Учебный Streamlit frontend для сервиса про породы кошек. "
    "Ответы запрашиваются у FastAPI backend."
)

question = st.text_input(
    "Ваш вопрос",
    placeholder="Например: Расскажи про британских короткошёрстных кошек",
)

mode_label = st.radio(
    "Режим ответа",
    tuple(MODE_LABELS.keys()),
    horizontal=True,
)
use_rag = st.checkbox("Use RAG retrieval")

if st.button("Спросить", type="primary"):
    if not question.strip():
        st.warning("Введите вопрос, чтобы я мог подготовить ответ.")
    else:
        mode = MODE_LABELS[mode_label]

        with st.spinner("Спрашиваю backend..."):
            try:
                result = ask_backend(question.strip(), mode, use_rag)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Backend не запущен. Запустите его командой: "
                    "`uvicorn backend.main:app --reload --port 8000`."
                )
            except requests.exceptions.Timeout:
                st.error("Backend слишком долго отвечает. Попробуйте ещё раз.")
            except RuntimeError as error:
                st.error(str(error))
            except requests.exceptions.RequestException as error:
                st.error(f"Не удалось обратиться к backend: {error}")
            else:
                breed = result["breed"]
                st.subheader(
                    "Порода не найдена" if breed == "Unknown breed" else breed
                )
                st.caption(
                    f"Detected breed: {breed} | Mode: {result['mode']} | "
                    f"RAG: {'on' if result.get('rag_enabled') else 'off'}"
                )
                st.markdown(result["answer"])

                retrieved_context = result.get("retrieved_context") or []
                if retrieved_context:
                    with st.expander("Retrieved context"):
                        for index, chunk in enumerate(retrieved_context, start=1):
                            metadata = chunk.get("metadata") or {}
                            score = chunk.get("score", chunk.get("distance"))
                            st.markdown(f"**Chunk {index}**")
                            st.caption(
                                " | ".join(
                                    part
                                    for part in (
                                        f"Breed: {metadata.get('breed', 'unknown')}",
                                        f"Source: {metadata.get('source_id', 'unknown')}",
                                        (
                                            f"Distance: {score:.4f}"
                                            if isinstance(score, (int, float))
                                            else None
                                        ),
                                    )
                                    if part
                                )
                            )
                            st.write(chunk.get("text", ""))

                breed_context = result.get("breed_context") or {}
                if breed_context:
                    with st.expander("Использованные факты"):
                        if breed_context.get("fallback_note"):
                            st.info(breed_context["fallback_note"])

                        st.markdown("**Внешний вид**")
                        for item in breed_context.get("appearance", []):
                            st.write(f"- {item}")

                        st.markdown("**Характер**")
                        for item in breed_context.get("temperament", []):
                            st.write(f"- {item}")

                        st.markdown("**Уход**")
                        for item in breed_context.get("care", []):
                            st.write(f"- {item}")
