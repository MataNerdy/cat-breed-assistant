from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Protocol

from dotenv import load_dotenv

from src.cat_breed_assistant.evaluation.retrieval.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
)
from src.cat_breed_assistant.evaluation.retrieval.schemas import ModelConfig


class EvaluationProviderError(RuntimeError):
    """Raised when an evaluation LLM provider cannot produce a usable response."""


class EvaluationProvider(Protocol):
    provider: str
    model: str

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """Return a raw JSON string from the provider."""


def extract_json_text(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = min((idx for idx in [text.find("{"), text.find("[")] if idx >= 0), default=-1)
    if start > 0:
        text = text[start:]
    return text.strip()


def ensure_json(raw_output: str) -> str:
    text = extract_json_text(raw_output)
    json.loads(text)
    return text


class BaseRetryingProvider:
    provider: str

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = config.model

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(self._complete, system_prompt, user_prompt)
                    raw_output = future.result(timeout=self.config.timeout_seconds)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
                return ensure_json(raw_output)
            except TimeoutError as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        raise EvaluationProviderError(
            f"{self.provider} provider failed after retries: {last_error}"
        ) from last_error

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiEvaluationProvider(BaseRetryingProvider):
    provider = "gemini"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EvaluationProviderError("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
            ),
        )
        if not response.text:
            raise EvaluationProviderError("Gemini returned an empty response")
        return response.text


class MistralEvaluationProvider(BaseRetryingProvider):
    provider = "mistral"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EvaluationProviderError("MISTRAL_API_KEY is not configured")
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.complete(
            model=self.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise EvaluationProviderError("Mistral returned an empty response")
        return str(content)


def make_provider(config: ModelConfig) -> EvaluationProvider:
    if config.provider == "gemini":
        return GeminiEvaluationProvider(config)
    if config.provider == "mistral":
        return MistralEvaluationProvider(config)
    raise EvaluationProviderError(f"Unsupported provider: {config.provider}")


def api_key_availability() -> dict[str, bool]:
    load_dotenv()
    return {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "mistral": bool(os.getenv("MISTRAL_API_KEY")),
    }


def require_api_keys(configs: list[ModelConfig]) -> None:
    available = api_key_availability()
    missing = sorted({config.provider for config in configs if not available[config.provider]})
    if missing:
        raise EvaluationProviderError(
            f"Missing API keys for providers: {', '.join(missing)}"
        )


def provider_system_prompt(kind: str) -> str:
    if kind == "generator":
        return GENERATOR_SYSTEM_PROMPT
    if kind == "validator":
        return VALIDATOR_SYSTEM_PROMPT
    raise ValueError(f"Unknown provider prompt kind: {kind}")
