from pydantic import BaseModel


class SettingsRead(BaseModel):
    llm_default_provider: str
    llm_default_model: str
    embedding_default_provider: str
    embedding_default_model: str
    vector_store_default: str
    ollama_base_url: str
    openai_configured: bool
    anthropic_configured: bool


class SettingsUpdate(BaseModel):
    llm_default_provider: str | None = None
    llm_default_model: str | None = None
    embedding_default_provider: str | None = None
    embedding_default_model: str | None = None
    vector_store_default: str | None = None
    ollama_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
