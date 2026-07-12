from typing import Any, Literal

from pydantic import BaseModel, Field


AnswerMode = Literal["mock", "openai", "gemini", "mistral"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: AnswerMode = "mock"
    use_rag: bool = True


class AskResponse(BaseModel):
    answer: str
    breed: str
    mode: AnswerMode
    breed_context: dict[str, Any] | None = None
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    rag_enabled: bool = False
    retrieval_strategy: str | None = None
    detected_breed: str | None = None
    warning: str | None = None
    image_url: str | None = None
    reference_image_id: str | None = None


class ErrorResponse(BaseModel):
    detail: str
