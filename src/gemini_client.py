import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
Ты дружелюбный ассистент для учебного приложения про породы кошек.
Отвечай на русском языке понятно для обычного пользователя.
Стиль: доброжелательный, спокойный, с лёгким юмором, без слишком научного тона.

Используй только факты из переданного контекста о породе.
Если передан retrieved context из RAG, считай его основным источником фактов.
Не выдумывай факты, которых нет в контексте.
Если данных мало, честно скажи, что информации в локальной базе мало.
Если breed указан как Unknown breed, не отвечай про British Shorthair по умолчанию.
Объясни, что нужной породы или животного нет в локальной базе, и предложи
доступные породы из контекста.

Не выдумывай медицинские рекомендации. Если вопрос касается здоровья,
советуй обратиться к ветеринару и не ставь диагнозы.
""".strip()


def _format_breed_context(breed_context: dict) -> str:
    def bullets(key: str) -> str:
        return "\n".join(f"- {item}" for item in breed_context.get(key, []))

    mentioned_breeds = "\n".join(
        f"- {profile['breed']}: "
        f"{'; '.join(profile.get('differs_from_other_breeds', [])[:2])}"
        for profile in breed_context.get("mentioned_breeds", [])
    )
    available_breeds = ", ".join(breed_context.get("available_breeds", []))

    return (
        f"Порода: {breed_context.get('breed', 'Неизвестно')}\n\n"
        f"Происхождение: {breed_context.get('origin', 'Неизвестно')}\n\n"
        f"Внешний вид:\n{bullets('appearance')}\n\n"
        f"Характер:\n{bullets('temperament')}\n\n"
        f"Уход:\n{bullets('care')}\n\n"
        f"Осторожные заметки о здоровье:\n{bullets('health_notes')}\n\n"
        f"Интересные факты:\n{bullets('fun_facts')}\n\n"
        f"Отличия от других пород:\n{bullets('differs_from_other_breeds')}\n\n"
        f"Упомянутые породы в вопросе:\n{mentioned_breeds}\n\n"
        f"Доступные породы в локальной базе: {available_breeds}\n\n"
        f"Fallback note: {breed_context.get('fallback_note', '')}"
    )


def _format_retrieved_context(retrieved_context: list[dict] | None) -> str:
    if not retrieved_context:
        return "RAG-контекст не передан."

    parts = []
    for index, chunk in enumerate(retrieved_context, start=1):
        metadata = chunk.get("metadata") or {}
        distance = chunk.get("distance", chunk.get("score", "unknown"))
        parts.append(
            "\n".join(
                (
                    f"Chunk {index}",
                    f"Breed: {metadata.get('breed', 'unknown')}",
                    f"Source id: {metadata.get('source_id', 'unknown')}",
                    f"Distance: {distance}",
                    f"Text: {chunk.get('text', '')}",
                )
            )
        )

    return "\n\n".join(parts)


def generate_gemini_answer(
    question: str,
    breed_context: dict,
    retrieved_context: list[dict] | None = None,
) -> str:
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
        f"{_format_breed_context(breed_context)}\n\n"
        "Retrieved context из RAG:\n"
        f"{_format_retrieved_context(retrieved_context)}"
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
