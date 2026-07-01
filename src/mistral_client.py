import os

from dotenv import load_dotenv
from mistralai import Mistral


MODEL = "mistral-small-latest"

SYSTEM_PROMPT = """
Ты дружелюбный ассистент для учебного приложения про породы кошек.
Отвечай на русском языке понятно для обычного пользователя.
Стиль: доброжелательный, спокойный, с лёгким юмором, без слишком научного тона.

Используй только факты из переданного контекста о породе.
Не выдумывай факты, которых нет в контексте.
Если данных мало, честно скажи, что информации в локальной базе мало.
Если breed указан как Unknown breed, не отвечай про British Shorthair по умолчанию.
Объясни, что нужной породы или животного нет в локальной базе, и предложи
доступные породы из контекста.

Не выдумывай медицинские рекомендации. Если вопрос касается здоровья,
мягко советуй обратиться к ветеринару и не ставь диагнозы.
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


def generate_mistral_answer(question: str, breed_context: dict) -> str:
    """Generate a friendly cat breed answer using the Mistral API."""
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "Mistral API key не найден. Создайте файл `.env` и добавьте туда "
            "`MISTRAL_API_KEY`."
        )

    client = Mistral(api_key=api_key)

    try:
        response = client.chat.complete(
            model=MODEL,
            messages=[
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
    except Exception as error:
        raise RuntimeError(
            "Mistral API временно не ответил или вернул ошибку. "
            "Проверьте ключ, лимиты и подключение к интернету."
        ) from error

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Mistral вернул пустой ответ. Попробуйте ещё раз.")

    return answer
