"""Provider factory — resolves provider key → instance using settings."""
from __future__ import annotations

from typing import Any

from ...config import get_settings
from ...core.errors import ProviderError
from .base import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider


def get_provider(name: str | None = None) -> BaseLLMProvider:
    s = get_settings()
    key = (name or s.llm_default_provider).lower()
    if key == "ollama":
        return OllamaProvider(base_url=s.ollama_base_url)
    if key == "openai":
        return OpenAIProvider(api_key=s.openai_api_key, base_url=s.openai_base_url)
    raise ProviderError(f"Unknown LLM provider: {name}")


def available_providers() -> list[dict[str, Any]]:
    s = get_settings()
    ollama = OllamaProvider(base_url=s.ollama_base_url)
    openai = OpenAIProvider(api_key=s.openai_api_key, base_url=s.openai_base_url)
    return [
        {"name": "ollama", "configured": ollama.is_configured(), "default_model": s.llm_default_model},
        {"name": "openai", "configured": openai.is_configured(), "default_model": "gpt-4o-mini"},
    ]
