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
}


def ask_backend(question: str, mode: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question, "mode": mode},
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
                result = ask_backend(question.strip(), mode)
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
                st.subheader(result["breed"])
                st.caption(f"Mode: {result['mode']}")
                st.markdown(result["answer"])
