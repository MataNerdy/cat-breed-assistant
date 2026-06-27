import os

import streamlit as st
from dotenv import load_dotenv

from src.cat_knowledge import generate_mock_answer
from src.gemini_client import generate_gemini_answer
from src.llm_client import generate_llm_answer


load_dotenv()


st.set_page_config(
    page_title="Cat Breed Assistant",
    page_icon="🐱",
    layout="centered",
)


st.title("Cat Breed Assistant")
st.write(
    "Учебный Streamlit-сервис, который отвечает на вопросы про породы кошек. "
    "Можно использовать локальную заглушку, OpenAI API или Gemini API."
)

question = st.text_input(
    "Ваш вопрос",
    placeholder="Например: Расскажи про британских короткошёрстных кошек",
)

mode = st.radio(
    "Режим ответа",
    ("Mock mode", "OpenAI mode", "Gemini mode"),
    horizontal=True,
)

if st.button("Спросить", type="primary"):
    if not question.strip():
        st.warning("Введите вопрос, чтобы я мог подготовить ответ.")
    else:
        result = generate_mock_answer(question)

        st.subheader(result["breed"])

        if mode == "Mock mode":
            st.markdown(result["answer"])

            st.markdown("### Ключевые факты")
            for fact in result["facts"]:
                st.write(f"- {fact}")

            st.markdown("### Особенности ухода")
            for note in result["care_notes"]:
                st.write(f"- {note}")
        elif mode == "OpenAI mode" and not os.getenv("OPENAI_API_KEY"):
            st.error(
                "API-ключ не найден. Создайте файл `.env` и добавьте туда "
                "`OPENAI_API_KEY`."
            )
        elif mode == "OpenAI mode":
            with st.spinner("Готовлю OpenAI-ответ..."):
                try:
                    answer = generate_llm_answer(question, result)
                except Exception as error:
                    st.error(f"Не удалось получить ответ от OpenAI: {error}")
                else:
                    st.markdown(answer)
        elif not os.getenv("GEMINI_API_KEY"):
            st.error(
                "Gemini API key не найден. Создайте файл `.env` и добавьте "
                "туда `GEMINI_API_KEY`."
            )
        else:
            with st.spinner("Готовлю Gemini-ответ..."):
                try:
                    answer = generate_gemini_answer(question, result)
                except Exception as error:
                    st.error(f"Не удалось получить ответ от Gemini: {error}")
                else:
                    st.markdown(answer)
