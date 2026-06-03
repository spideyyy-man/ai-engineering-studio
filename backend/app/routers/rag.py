"""Tab 4 — RAG Studio. Catalog + build index (ingest) + query playground.

v1 engine = TF-IDF + cosine (offline, no model downloads). See
services/rag_studio/pipeline.py. The catalog advertises the full component set;
ingest/query run the naive→advanced path end-to-end so the page works like
ML/DL Studio.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.errors import NotFoundError, ValidationError
from ..database import get_db
from ..models import Project, RAGIndex
from ..schemas import (
    RAGIndexRead,
    RAGIngestRequest,
    RAGIngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from ..services.rag_studio import catalog, pipeline

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/catalog")
def get_catalog():
    return catalog()


# ───────────────────────────── ingest (build index) ────────────────────────
@router.post("/ingest", response_model=RAGIngestResponse)
def ingest(payload: RAGIngestRequest, db: Session = Depends(get_db)):
    project = db.get(Project, payload.project_id)
    if not project:
        raise NotFoundError("project")
    if not payload.sources:
        raise ValidationError("provide at least one source (text / url / file / csv)")

    # resolve every source to text
    docs = []
    for src in payload.sources:
        try:
            text = pipeline.load_source(src.kind, src.location)
        except ValueError as e:
            raise ValidationError(str(e))
        if text.strip():
            docs.append({"text": text, "source": f"{src.kind}:{src.location[:60]}"})
    if not docs:
        raise ValidationError("no readable text found in the provided sources")

    source_type = payload.sources[0].kind if len({s.kind for s in payload.sources}) == 1 else "mixed"

    embed_provider = (payload.embedding_provider or "tfidf").lower()
    embed_model = payload.embedding_model or "tfidf"

    # create the row first so we have a stable index_id for the on-disk store
    row = RAGIndex(
        project_id=payload.project_id,
        name=payload.name,
        source_type=source_type,
        chunking_strategy=payload.chunking_strategy,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        embedding_provider=embed_provider,
        embedding_model=embed_model,
        vector_store=payload.vector_store or pipeline.VECTOR_STORE,
        n_chunks=0,
        storage_path="",
        metrics_json={},
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    try:
        result = pipeline.build_index(
            project_id=payload.project_id,
            index_id=row.id,
            docs=docs,
            chunk_impl=payload.chunking_strategy,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            embed_provider=embed_provider,
            embed_model=embed_model,
        )
    except Exception as e:  # noqa: BLE001
        db.delete(row)
        db.commit()
        raise ValidationError(f"index build failed: {e}")

    row.n_chunks = result["n_chunks"]
    row.storage_path = result["storage_path"]
    row.embedding_model = result["embed_model"]
    row.metrics_json = {
        "vocab_size": result["vocab_size"], "build_ms": result["build_ms"],
        "n_sources": len(docs), "embed_kind": result["embed_kind"], "dim": result["dim"],
    }
    db.commit()
    db.refresh(row)

    return RAGIngestResponse(
        index=RAGIndexRead.model_validate(row),
        n_chunks=result["n_chunks"],
        vocab_size=result["vocab_size"],
        build_ms=result["build_ms"],
        embed_kind=result["embed_kind"],
        dim=result["dim"],
        sample_chunks=result["sample_chunks"],
        notes=result["notes"],
    )


# ───────────────────────────── query ───────────────────────────────────────
@router.post("/query", response_model=RAGQueryResponse)
def query(payload: RAGQueryRequest, db: Session = Depends(get_db)):
    row = db.get(RAGIndex, payload.index_id)
    if not row:
        raise NotFoundError("index")
    if not row.storage_path or not Path(row.storage_path).exists():
        raise ValidationError("index store not found on disk — rebuild the index")

    t0 = time.perf_counter()
    res = pipeline.run_query(
        storage_path=row.storage_path,
        question=payload.question,
        transform=payload.query_transform,
        transform_n=payload.transform_n,
        retriever=payload.retriever_type,
        k=payload.k,
        fusion=payload.fusion,
        alpha=payload.hybrid_alpha,
        rerank=payload.reranker,
        rerank_top_n=payload.rerank_top_n,
        compress=payload.context_processor or "none",
        target_tokens=payload.compress_target_tokens,
        generate=payload.generate_answer,
        provider=payload.llm_provider,
        model=payload.llm_model,
        citations=payload.citations,
    )

    return RAGQueryResponse(
        docs=res["docs"],
        answer=res["answer"],
        answer_kind=res["answer_kind"],
        retriever=payload.retriever_type,
        k=payload.k,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt=res["prompt"],
        trace=res["trace"],
        notes=res["notes"],
    )


# ───────────────────────────── registry ────────────────────────────────────
@router.get("/indexes", response_model=list[RAGIndexRead])
def list_indexes(project_id: str | None = Query(None), db: Session = Depends(get_db)):
    q = db.query(RAGIndex)
    if project_id:
        q = q.filter(RAGIndex.project_id == project_id)
    return q.order_by(RAGIndex.created_at.desc()).all()


@router.get("/indexes/{index_id}", response_model=RAGIndexRead)
def get_index(index_id: str, db: Session = Depends(get_db)):
    row = db.get(RAGIndex, index_id)
    if not row:
        raise NotFoundError("index")
    return row


@router.delete("/indexes/{index_id}")
def delete_index(index_id: str, db: Session = Depends(get_db)):
    row = db.get(RAGIndex, index_id)
    if not row:
        raise NotFoundError("index")
    if row.storage_path and Path(row.storage_path).exists():
        shutil.rmtree(row.storage_path, ignore_errors=True)
    db.delete(row)
    db.commit()
    return {"message": "deleted", "id": index_id}
