import os
import logging

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MODE_LABELS = {
    "Mock mode": "mock",
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

if st.button("Спросить", type="primary"):
    if not question.strip():
        st.warning("Введите вопрос, чтобы я мог подготовить ответ.")
    else:
        mode = MODE_LABELS[mode_label]

        with st.spinner("Спрашиваю backend..."):
            try:
                result = ask_backend(question.strip(), mode, use_rag=True)
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
                LOGGER.info(
                    "Cat assistant response: mode=%s breed=%s detected_breed=%s "
                    "retrieval_strategy=%s context_count=%s warning=%s "
                    "reference_image_id=%s",
                    result.get("mode"),
                    result.get("breed"),
                    result.get("detected_breed"),
                    result.get("retrieval_strategy"),
                    len(result.get("retrieved_context") or []),
                    result.get("warning"),
                    result.get("reference_image_id"),
                )

                for chunk in result.get("retrieved_context") or []:
                    metadata = chunk.get("metadata") or {}
                    LOGGER.info(
                        "Retrieved CatAPI context: rank=%s score=%s breed_name=%s "
                        "breed_id=%s origin=%s wikipedia_url=%s reference_image_id=%s",
                        chunk.get("rank"),
                        chunk.get("score"),
                        chunk.get("breed_name"),
                        chunk.get("breed_id"),
                        metadata.get("origin"),
                        metadata.get("wikipedia_url"),
                        metadata.get("reference_image_id"),
                    )

                if result.get("warning"):
                    st.warning(result["warning"])

                if result.get("image_url"):
                    st.image(result["image_url"])

                st.markdown(result["answer"])
