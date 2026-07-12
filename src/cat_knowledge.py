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
    if not retrieved_context:
        return (
            "В CatAPI базе не нашлось подходящей информации по этому вопросу. "
            "Попробуйте явно назвать породу, например: «сфинкс», «мейн-кун» "
            "или «британская короткошёрстная»."
        )

    top = retrieved_context[0]
    metadata = top.get("metadata") or {}
    breed = top.get("breed_name") or metadata.get("breed_name") or breed_context["breed"]
    origin = metadata.get("origin", "не указано")
    wikipedia_url = metadata.get("wikipedia_url")
    text = " ".join(str(top.get("text", "")).split())

    description = _extract_line(text, "Description")
    temperament = _extract_line(text, "Temperament")

    parts = [
        f"Нашёл в CatAPI базе породу **{breed}**.",
        f"**Происхождение:** {origin}.",
    ]

    if description:
        parts.append(f"**Описание:** {description}")
    if temperament:
        parts.append(f"**Характер:** {temperament}")

    parts.append(
        "Это mock-ответ: он показывает, какие факты достал CatAPI retrieval. "
        "Для более связного ответа можно включить один из LLM modes."
    )

    if wikipedia_url:
        parts.append(f"Источник: {wikipedia_url}")

    return "\n\n".join(parts)


def _extract_line(text: str, field_name: str) -> str:
    marker = f"{field_name}:"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    known_markers = (
        "Breed name:",
        "Description:",
        "Temperament:",
        "Origin:",
        "Life span:",
        "Weight:",
        "Grooming level:",
        "Energy level:",
        "Health issues score:",
        "Child friendly score:",
        "Dog friendly score:",
        "Stranger friendly score:",
        "Intelligence score:",
        "Hairless:",
        "Shedding level:",
        "Social needs score:",
        "Vocalisation score:",
        "Hypoallergenic:",
        "Wikipedia URL:",
    )
    end_positions = [tail.find(marker) for marker in known_markers if tail.find(marker) > 0]
    if end_positions:
        tail = tail[: min(end_positions)]
    return tail.strip()


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
