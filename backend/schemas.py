from typing import Any, Literal

from pydantic import BaseModel, Field


AnswerMode = Literal["mock", "openai", "gemini", "mistral"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: AnswerMode = "mock"


class AskResponse(BaseModel):
    answer: str
    breed: str
    mode: AnswerMode
    breed_context: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str
