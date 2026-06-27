from fastapi import FastAPI, HTTPException

from .schemas import AskRequest, AskResponse
from .services import generate_answer


app = FastAPI(title="Cat Breed Assistant API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Введите вопрос.")

    try:
        return generate_answer(question, request.mode)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Не удалось получить ответ от выбранного provider. "
                "Проверьте API-ключ, квоты и подключение."
            ),
        ) from error
