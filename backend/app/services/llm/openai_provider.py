"""OpenAI provider — optional, requires OPENAI_API_KEY."""
from __future__ import annotations


from ...core.errors import ProviderError
from .base import BaseLLMProvider, ChatMessage, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or ""
        self.base_url = base_url

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        if not self.api_key:
            raise ProviderError("OpenAI not configured — OPENAI_API_KEY missing.")
        # Lazy import so the dep is optional at startup
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list[ChatMessage], *, model: str, temperature: float = 0.2,
             max_tokens: int | None = None) -> LLMResponse:
        client = self._client()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise ProviderError(f"OpenAI chat failed: {e}") from e

        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        client = self._client()
        try:
            resp = client.embeddings.create(model=model, input=texts)
        except Exception as e:
            raise ProviderError(f"OpenAI embed failed: {e}") from e
        return [d.embedding for d in resp.data]
