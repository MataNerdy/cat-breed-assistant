import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
Ты дружелюбный ассистент для учебного приложения про породы кошек.
Отвечай на русском языке понятно для обычного пользователя.
Стиль: доброжелательный, спокойный, с лёгким юмором, без слишком научного тона.

Не выдумывай медицинские рекомендации. Если вопрос касается здоровья,
советуй обратиться к ветеринару и не ставь диагнозы.

Ответ должен объяснять:
- кто такие британские короткошёрстные кошки;
- как они выглядят;
- какой у них характер;
- чем они отличаются от других пород;
- базовые советы по уходу.
""".strip()


def _format_breed_context(breed_context: dict) -> str:
    facts = "\n".join(f"- {fact}" for fact in breed_context.get("facts", []))
    care_notes = "\n".join(
        f"- {note}" for note in breed_context.get("care_notes", [])
    )

    return (
        f"Порода: {breed_context.get('breed', 'Неизвестно')}\n\n"
        f"Базовый ответ:\n{breed_context.get('answer', '')}\n\n"
        f"Факты:\n{facts}\n\n"
        f"Уход:\n{care_notes}"
    )


def generate_gemini_answer(question: str, breed_context: dict) -> str:
    """Generate a friendly cat breed answer using the Gemini API."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key не найден. Создайте файл `.env` и добавьте туда "
            "`GEMINI_API_KEY`."
        )

    client = genai.Client(api_key=api_key)
    prompt = (
        "Вопрос пользователя:\n"
        f"{question.strip()}\n\n"
        "Контекст о породе:\n"
        f"{_format_breed_context(breed_context)}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )
    except Exception as error:
        raise RuntimeError(
            "Gemini API временно не ответил или вернул ошибку. "
            "Проверьте ключ, лимиты и подключение к интернету."
        ) from error

    if not response.text:
        raise RuntimeError("Gemini вернул пустой ответ. Попробуйте ещё раз.")

    return response.text
