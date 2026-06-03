"""Settings — read-only view + provider connectivity check."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..config import get_settings
from ..schemas import SettingsRead
from ..services.llm import available_providers, get_provider

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsRead)
def read_settings():
    s = get_settings()
    return SettingsRead(
        llm_default_provider=s.llm_default_provider,
        llm_default_model=s.llm_default_model,
        embedding_default_provider=s.embedding_default_provider,
        embedding_default_model=s.embedding_default_model,
        vector_store_default=s.vector_store_default,
        ollama_base_url=s.ollama_base_url,
        openai_configured=bool(s.openai_api_key),
        anthropic_configured=bool(s.anthropic_api_key),
    )


@router.get("/providers")
def get_providers():
    return available_providers()


@router.post("/test-llm")
def test_llm(provider: str | None = Query(None)):
    p = get_provider(provider)
    ok = p.ping()
    return {"provider": p.name, "ok": ok}
