import os

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """
Ты дружелюбный ассистент для учебного приложения про породы кошек.
Отвечай на русском языке понятно для обычного пользователя.
Стиль: доброжелательный, спокойный, с лёгким юмором, без слишком научного тона.

Используй только факты из переданного контекста о породе.
Если передан CatAPI retrieved context, считай его основным источником фактов.
Не выдумывай факты, которых нет в контексте.
Если данных мало, честно скажи, что информации в CatAPI базе мало.
Если breed указан как Unknown breed, не отвечай про British Shorthair по умолчанию.
Объясни, что нужной породы или животного нет в CatAPI базе, и предложи
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
        return "CatAPI retrieved context не передан."

    parts = []
    for index, chunk in enumerate(retrieved_context, start=1):
        metadata = chunk.get("metadata") or {}
        score = chunk.get("score", "unknown")
        parts.append(
            "\n".join(
                (
                    f"Chunk {index}",
                    f"Rank: {chunk.get('rank', index)}",
                    f"Score: {score}",
                    f"Breed: {chunk.get('breed_name') or metadata.get('breed_name', 'unknown')}",
                    f"Breed id: {chunk.get('breed_id') or metadata.get('breed_id', 'unknown')}",
                    f"Source: {chunk.get('source') or metadata.get('source', 'thecatapi')}",
                    f"Origin: {metadata.get('origin', 'unknown')}",
                    f"Wikipedia URL: {metadata.get('wikipedia_url') or ''}",
                    f"Reference image id: {metadata.get('reference_image_id') or ''}",
                    f"Text: {chunk.get('text', '')}",
                )
            )
        )

    return "\n\n".join(parts)


def generate_llm_answer(
    question: str,
    breed_context: dict,
    retrieved_context: list[dict] | None = None,
) -> str:
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
                    f"{_format_breed_context(breed_context)}\n\n"
                    "CatAPI retrieved context:\n"
                    f"{_format_retrieved_context(retrieved_context)}"
                ),
            },
        ],
    )

    return response.output_text
