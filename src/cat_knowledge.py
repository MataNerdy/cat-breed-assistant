INTRO_TEMPLATES = (
    "{breed} — порода с характерным внешним видом и довольно узнаваемым темпераментом.",
    "Если коротко, {breed} — не просто красивое название, а целый набор породных особенностей.",
    "{breed} — хороший пример того, как внешность, характер и уход складываются в породный профиль.",
)


def _pick_template(question: str) -> str:
    index = len(question.strip()) % len(INTRO_TEMPLATES)
    return INTRO_TEMPLATES[index]


def _first(items: list[str], fallback: str) -> str:
    return items[0] if items else fallback


def _mock_rag_answer(breed_context: dict, retrieved_context: list[dict]) -> str:
    breeds = []
    facts = []

    for chunk in retrieved_context:
        metadata = chunk.get("metadata") or {}
        breed = metadata.get("breed")
        if breed and breed not in breeds:
            breeds.append(breed)

        text = " ".join(str(chunk.get("text", "")).split())
        if text:
            facts.append(text[:320])

    breed_label = ", ".join(breeds[:3]) or breed_context["breed"]
    fact_lines = "\n".join(
        f"{index}. {fact}" for index, fact in enumerate(facts[:3], start=1)
    )

    return (
        f"Нашёл похожие фрагменты в RAG-индексе для: **{breed_label}**.\n\n"
        f"{fact_lines}\n\n"
        "Это mock-ответ: он показывает, какие факты достал retrieval layer. "
        "Для более связного текста включите один из LLM modes."
    )


def generate_mock_answer(
    question: str,
    breed_context: dict,
    retrieved_context: list[dict] | None = None,
) -> str:
    """Build a deterministic mock answer from local breed context."""
    breed = breed_context["breed"]
    fallback_note = breed_context.get("fallback_note")
    retrieved_context = retrieved_context or []

    if retrieved_context:
        return _mock_rag_answer(breed_context, retrieved_context)

    if breed_context.get("is_fallback"):
        available_breeds = ", ".join(breed_context.get("available_breeds", []))
        return (
            f"_{fallback_note}_\n\n"
            "Я не буду притворяться, что знаю всё на свете. В локальной базе пока "
            "нет профиля для породы или животного из вашего вопроса.\n\n"
            f"Сейчас можно спросить про: {available_breeds}.\n\n"
            "Например: «Расскажи про мейн-куна» или «Как ухаживать за сфинксом?»"
        )

    intro = _pick_template(question).format(breed=breed)

    appearance = _first(
        breed_context.get("appearance", []),
        "Внешний вид в локальной базе пока описан кратко.",
    )
    temperament = _first(
        breed_context.get("temperament", []),
        "О характере пока мало данных в локальной базе.",
    )
    difference = _first(
        breed_context.get("differs_from_other_breeds", []),
        "Отличия от других пород пока описаны кратко.",
    )
    care = _first(
        breed_context.get("care", []),
        "Для ухода лучше ориентироваться на рекомендации заводчика и ветеринара.",
    )
    fun_fact = _first(
        breed_context.get("fun_facts", []),
        "Интересный факт пока не добавлен в локальную базу.",
    )

    parts = [
        intro,
        f"**Внешний вид:** {appearance}",
        f"**Характер:** {temperament}",
        f"**Чем отличается:** {difference}",
        f"**Уход:** {care}",
        f"**Интересный факт:** {fun_fact}",
    ]

    return "\n\n".join(parts)
