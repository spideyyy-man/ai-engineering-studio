from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RAGSource(BaseModel):
    kind: str  # pdf, docx, txt, csv, url, youtube
    location: str  # filepath or URL


class RAGIngestRequest(BaseModel):
    project_id: str
    name: str
    sources: list[RAGSource]
    # index-time pipeline properties
    loader: str | None = None                 # see catalog LOADERS (auto-detect if None)
    chunking_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    enrich: str | None = None                 # e.g. "contextual_retrieval"
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dims: int = 0                   # 0 = model default; >0 = Matryoshka (MRL) truncation
    quantization: str = "none"                # none | int8 | binary
    vector_store: str | None = None
    ann_index: str | None = None              # flat | hnsw | ivf | ivfpq ...
    ann_params: dict[str, Any] = {}           # e.g. {"m": 32, "ef_construction": 200}


class RAGIndexRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    name: str
    source_type: str
    chunking_strategy: str
    embedding_provider: str
    embedding_model: str
    vector_store: str
    n_chunks: int
    metrics_json: dict[str, Any] = {}
    created_at: datetime


class RAGQueryRequest(BaseModel):
    index_id: str
    question: str
    # strategy template (a saved wiring) — overrides individual knobs when set
    strategy: str | None = None               # see catalog RAG_STRATEGIES
    # query-time pipeline properties
    query_transform: str = "none"             # none | rewrite | hyde | multi_query | rag_fusion | ...
    transform_n: int = 4                      # variants for multi_query / rag_fusion
    retriever_type: str = "hybrid"            # dense | sparse | hybrid | mmr | similarity
    k: int = 10
    fusion: str = "rrf"                       # rrf | weighted
    hybrid_alpha: float = 0.5                 # dense weight in hybrid fusion
    reranker: str = "none"                    # none | cross_encoder | cohere_rerank | ...
    rerank_top_n: int = 8
    context_processor: str | None = None      # parent_document | sentence_window | compression | ...
    compress_target_tokens: int = 2000
    generate_answer: bool = True
    citations: bool = True
    llm_provider: str | None = None
    llm_model: str | None = None


class RAGIngestResponse(BaseModel):
    index: RAGIndexRead
    n_chunks: int
    vocab_size: int
    build_ms: int
    embed_kind: str = "tfidf"          # tfidf | neural
    dim: int = 0
    sample_chunks: list[str] = []
    notes: list[str] = []              # honest degradation messages


class RAGRetrievedDoc(BaseModel):
    content: str
    score: float | None = None
    metadata: dict[str, Any] = {}


class RAGQueryResponse(BaseModel):
    docs: list[RAGRetrievedDoc]
    answer: str | None = None
    answer_kind: str = "none"          # llm | extractive | none
    retriever: str = "dense"
    k: int = 10
    latency_ms: int = 0
    prompt: str | None = None
    tokens: int | None = None
    trace: dict[str, Any] = {}         # per-stage debug trace
    notes: list[str] = []              # honest degradation messages
