"""Application settings — loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "AI Engineering Studio"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_cors_origins: str = "http://localhost:5180,http://127.0.0.1:5180"

    # DB
    database_url: str = f"sqlite:///{(BACKEND_ROOT / 'storage' / 'studio.db').as_posix()}"

    # Storage
    storage_root: Path = BACKEND_ROOT / "storage"
    datasets_dir: Path = BACKEND_ROOT / "storage" / "datasets"
    models_dir: Path = BACKEND_ROOT / "storage" / "models"
    vectorstores_dir: Path = BACKEND_ROOT / "storage" / "vectorstores"
    artifacts_dir: Path = BACKEND_ROOT / "storage" / "artifacts"

    # LLM / embeddings
    llm_default_provider: str = "ollama"
    llm_default_model: str = "llama3"
    embedding_default_provider: str = "ollama"
    embedding_default_model: str = "nomic-embed-text"

    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    # hosted embedding / rerank providers
    cohere_api_key: str = ""
    voyage_api_key: str = ""
    google_api_key: str = ""

    # Vector
    vector_store_default: str = "faiss"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    ann_index_default: str = "hnsw"

    # RAG defaults (keys mirror services/rag_studio/catalog.py)
    rag_default_strategy: str = "advanced"        # see RAG_STRATEGIES
    rag_chunking_strategy: str = "recursive"
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_retriever: str = "hybrid"                 # dense | sparse | hybrid | mmr
    rag_top_k: int = 50
    rag_fusion: str = "rrf"                       # rrf | weighted
    rag_hybrid_alpha: float = 0.5                 # dense weight in hybrid fusion
    rag_query_transform: str = "none"             # none | rewrite | hyde | rag_fusion | ...
    rag_reranker: str = "none"                    # none | cross_encoder | cohere_rerank | ...
    rag_rerank_top_n: int = 8
    rag_embedding_dims: int = 0                   # 0 = model default; >0 = MRL truncation
    rag_compress_target_tokens: int = 2000
    rag_eval_metrics: str = "faithfulness,context_precision,answer_relevance"

    # Limits
    max_upload_mb: int = 200
    max_training_minutes: int = 30

    @field_validator("storage_root", "datasets_dir", "models_dir", "vectorstores_dir", "artifacts_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = (BACKEND_ROOT / p).resolve()
        return p

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.app_cors_origins.split(",") if o.strip()]

    @property
    def rag_eval_metrics_list(self) -> List[str]:
        return [m.strip() for m in self.rag_eval_metrics.split(",") if m.strip()]

    def ensure_dirs(self) -> None:
        for p in (
            self.storage_root,
            self.datasets_dir,
            self.models_dir,
            self.vectorstores_dir,
            self.artifacts_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
