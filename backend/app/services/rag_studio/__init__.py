"""RAG Studio — Tab 4.

The component & strategy catalog (the "matplotlib primitives") lives in
`catalog.py`. Ingest/query execution is Phase 3; the catalog is what powers
the visual builder, the strategy templates, and the dev-SDK parameter forms.
"""
from __future__ import annotations

from .catalog import (
    catalog,
    PIPELINE_STAGES,
    LOADERS,
    CONNECTORS,
    CHUNKING_STRATEGIES,
    EMBEDDING_MODELS,
    EMBEDDING_TECHNIQUES,
    VECTOR_STORES,
    ANN_INDEXES,
    RETRIEVERS,
    QUERY_TRANSFORMS,
    RERANKERS,
    CONTEXT_PROCESSORS,
    GENERATION,
    EVAL_METRICS,
    RAG_STRATEGIES,
    DEFAULT_STACK,
)

__all__ = [
    "catalog",
    "PIPELINE_STAGES",
    "LOADERS",
    "CONNECTORS",
    "CHUNKING_STRATEGIES",
    "EMBEDDING_MODELS",
    "EMBEDDING_TECHNIQUES",
    "VECTOR_STORES",
    "ANN_INDEXES",
    "RETRIEVERS",
    "QUERY_TRANSFORMS",
    "RERANKERS",
    "CONTEXT_PROCESSORS",
    "GENERATION",
    "EVAL_METRICS",
    "RAG_STRATEGIES",
    "DEFAULT_STACK",
]
