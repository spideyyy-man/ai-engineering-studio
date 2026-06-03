"""Ollama provider — local LLM + embeddings via HTTP."""
from __future__ import annotations

from typing import Any

import httpx

from ...core.errors import ProviderError
from .base import BaseLLMProvider, ChatMessage, LLMResponse


class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def is_configured(self) -> bool:
        try:
            with self._client() as c:
                r = c.get("/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def chat(self, messages: list[ChatMessage], *, model: str, temperature: float = 0.2,
             max_tokens: int | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            with self._client() as c:
                r = c.post("/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama chat failed: {e}") from e

        msg = data.get("message", {})
        return LLMResponse(
            text=msg.get("content", ""),
            model=model,
            provider=self.name,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
            raw=data,
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        try:
            out: list[list[float]] = []
            with self._client() as c:
                # Ollama /api/embeddings is per-text
                for t in texts:
                    r = c.post("/api/embeddings", json={"model": model, "prompt": t})
                    r.raise_for_status()
                    out.append(r.json().get("embedding", []))
            return out
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama embed failed: {e}") from e
