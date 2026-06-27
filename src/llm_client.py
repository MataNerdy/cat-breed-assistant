import os

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """
Ты дружелюбный ассистент для учебного приложения про породы кошек.
Отвечай на русском языке понятно для обычного пользователя.
Стиль: доброжелательный, спокойный, с лёгким юмором, без слишком научного тона.

Не выдумывай медицинские рекомендации. Если вопрос касается здоровья,
советуй обратиться к ветеринару и не ставь диагнозы.

Если пользователь спрашивает про британских короткошёрстных кошек,
объясни:
- кто такие британские короткошёрстные кошки;
- как они выглядят;
- какой у них характер;
- чем они отличаются от других пород;
- что важно знать про уход.
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


def generate_llm_answer(question: str, breed_context: dict) -> str:
    """Generate a friendly cat breed answer using the OpenAI API."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API-ключ не найден. Создайте файл `.env` и добавьте туда "
            "`OPENAI_API_KEY`."
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Вопрос пользователя:\n"
                    f"{question.strip()}\n\n"
                    "Контекст о породе:\n"
                    f"{_format_breed_context(breed_context)}"
                ),
            },
        ],
    )

    return response.output_text
