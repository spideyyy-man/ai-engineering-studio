"""RAG Studio — a real, config-driven execution engine.

Every stage from the architecture (chunk → embed → store · transform → retrieve →
rerank → compress → generate) is honored from the pipeline config the builder
sends. What can run with the installed libs (numpy + scikit-learn + httpx) runs
for real:

  • Chunk:     fixed / recursive / sentence / semantic
  • Embed:     tfidf (sklearn, offline) · ollama / openai embeddings (if reachable)
  • Retrieve:  dense (cosine) · sparse (BM25, implemented here) · hybrid (RRF) · MMR
  • Transform: none · rewrite · hyde · multi_query · rag_fusion   (LLM-backed)
  • Rerank:    none · llm (RankGPT-style)
  • Compress:  none · extractive token budget
  • Generate:  ollama / openai (if reachable) · extractive fallback

Selections that need an uninstalled library (e.g. a neural reranker, FAISS/Qdrant,
a self-hosted dense embedder) are accepted but **degraded with an explicit note**
returned to the UI — the choice stays visible, the engine stays honest.
"""
from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ...core.paths import vectorstore_path
from ...config import get_settings

VECTOR_STORE = "numpy"
RRF_K = 60

# ───────────────────────────── loaders ─────────────────────────────────────
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_TOK_RE = re.compile(r"[a-z0-9]+")


def _strip_html(html: str) -> str:
    html = _TAG_RE.sub(" ", html)
    text = _HTML_RE.sub(" ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# Local files the RAG loader may read: text-like types only, and only inside
# backend/storage or a folder listed in RAG_ALLOWED_DIRS.
_TEXT_FILE_EXT = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".csv", ".tsv"}
_CSV_FILE_EXT = {".csv", ".tsv"}
_MAX_REDIRECTS = 5


def _allowed_dirs() -> list[Path]:
    s = get_settings()
    return [Path(d).expanduser().resolve() for d in [s.storage_root, *s.rag_allowed_dirs_list]]


def _resolve_local_file(location: str, allowed_ext: set[str]) -> Path:
    """Resolve a user-typed path and refuse anything outside the allowed folders."""
    raw = (location or "").strip().strip('"')
    if not raw:
        raise ValueError("no file path given")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = get_settings().storage_root / p
    p = p.resolve()
    if p.suffix.lower() not in allowed_ext:
        raise ValueError(f"'{p.name}' is not a supported text file ({', '.join(sorted(allowed_ext))})")
    if not any(p.is_relative_to(d) for d in _allowed_dirs()):
        raise ValueError(
            f"'{raw}' is outside the folders RAG Studio may read. Put the file in backend/storage, "
            "or add its folder to RAG_ALLOWED_DIRS in .env and restart the backend."
        )
    if not p.is_file():
        raise ValueError(f"file not found: {raw}")
    return p


def _check_public_url(url: str) -> None:
    """Allow only http(s) URLs whose host resolves to public IP addresses.

    Stops the URL loader from being used to reach localhost, the LAN or cloud
    metadata endpoints. Set RAG_ALLOW_PRIVATE_URLS=true to turn this off.
    """
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http:// and https:// URLs are supported")
    if not parsed.host:
        raise ValueError("URL has no host")
    if get_settings().rag_allow_private_urls:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host '{parsed.host}'") from e
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise ValueError(
                f"'{parsed.host}' points to a private or local address; "
                "set RAG_ALLOW_PRIVATE_URLS=true in .env to allow this"
            )


def _fetch_url(url: str) -> httpx.Response:
    """GET a URL, re-checking the host on every redirect hop."""
    for _ in range(_MAX_REDIRECTS + 1):
        _check_public_url(url)
        r = httpx.get(url, timeout=20, follow_redirects=False,
                      headers={"User-Agent": "ai-studio-rag/1.0"})
        if r.is_redirect:
            url = urljoin(str(r.url), r.headers["location"])
            continue
        r.raise_for_status()
        return r
    raise ValueError(f"too many redirects (more than {_MAX_REDIRECTS})")


def load_source(kind: str, location: str) -> str:
    kind = (kind or "").lower()
    if kind in ("text", "raw", "paste"):
        return location or ""
    if kind == "url":
        try:
            r = _fetch_url(location.strip())
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"could not fetch URL '{location}': {e}")
        return _strip_html(r.text)
    if kind in ("txt", "md", "file"):
        p = _resolve_local_file(location, _TEXT_FILE_EXT)
        return p.read_text(encoding="utf-8", errors="ignore")
    if kind == "csv":
        import pandas as pd
        p = _resolve_local_file(location, _CSV_FILE_EXT)
        df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        return "\n".join(" · ".join(f"{c}: {v}" for c, v in row.items()) for _, row in df.iterrows())
    if kind in ("pdf", "docx", "pptx"):
        raise ValueError(f"'{kind}' parsing needs a parser library not installed here — use text / url / txt / csv.")
    raise ValueError(f"unknown source kind '{kind}'")


# ───────────────────────────── chunking ────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _budget(tokens: int) -> int:
    return max(120, int(tokens) * 4)  # ~4 chars/token


def chunk_text(text: str, strategy: str, size: int, overlap: int, vectorizer=None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    b, ov = _budget(size), min(_budget(overlap), _budget(size) // 2)
    if strategy == "fixed":
        step = max(1, b - ov)
        return [text[i:i + b].strip() for i in range(0, len(text), step) if text[i:i + b].strip()]
    if strategy == "sentence":
        return _pack(_SENT_RE.split(text), b)
    if strategy == "semantic":
        return _semantic(text, b)
    # recursive (default)
    return _pack(_recursive_pieces(text, ["\n\n", "\n", ". ", " "], b), b, ov)


def _pack(pieces: list[str], budget: int, ov: int = 0) -> list[str]:
    out, cur = [], ""
    for p in (s.strip() for s in pieces if s and s.strip()):
        if cur and len(cur) + len(p) + 1 > budget:
            out.append(cur.strip())
            cur = (cur[-ov:] + " " + p).strip() if ov else p
        else:
            cur = f"{cur} {p}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def _recursive_pieces(text: str, seps: list[str], budget: int) -> list[str]:
    if len(text) <= budget or not seps:
        return [text]
    out: list[str] = []
    for part in text.split(seps[0]):
        out.extend([part] if len(part) <= budget else _recursive_pieces(part, seps[1:], budget))
    return [p for p in out if p.strip()]


def _semantic(text: str, budget: int) -> list[str]:
    """Sentence embeddings (TF-IDF) → cut where adjacent similarity drops."""
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if len(sents) < 3:
        return _pack(sents, budget)
    try:
        v = TfidfVectorizer().fit_transform(sents)
        sims = [float(cosine_similarity(v[i], v[i + 1])[0][0]) for i in range(len(sents) - 1)]
        thr = float(np.percentile(sims, 25))
        out, cur = [], sents[0]
        for i in range(1, len(sents)):
            if sims[i - 1] < thr or len(cur) + len(sents[i]) > budget:
                out.append(cur)
                cur = sents[i]
            else:
                cur = f"{cur} {sents[i]}"
        out.append(cur)
        return out
    except Exception:  # noqa: BLE001
        return _pack(sents, budget)


# ───────────────────────────── embedding ───────────────────────────────────
def _embed_remote(texts: list[str], provider: str, model: str, s) -> np.ndarray | None:
    """Embed via Ollama/OpenAI over HTTP. Returns L2-normalized matrix or None."""
    to = httpx.Timeout(120.0, connect=1.5)
    try:
        vecs: list[list[float]] = []
        if provider == "ollama":
            for t in texts:
                r = httpx.post(f"{s.ollama_base_url}/api/embeddings",
                               json={"model": model, "prompt": t}, timeout=to)
                r.raise_for_status()
                vecs.append(r.json()["embedding"])
        elif provider == "openai" and s.openai_api_key:
            r = httpx.post(f"{s.openai_base_url}/embeddings",
                           headers={"Authorization": f"Bearer {s.openai_api_key}"},
                           json={"model": model, "input": texts}, timeout=to)
            r.raise_for_status()
            vecs = [d["embedding"] for d in r.json()["data"]]
        else:
            return None
        m = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.clip(norms, 1e-9, None)
    except Exception:  # noqa: BLE001
        return None


# ───────────────────────────── build (index-time) ──────────────────────────
def build_index(
    project_id: str,
    index_id: str,
    docs: list[dict[str, str]],
    *,
    chunk_impl: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed_provider: str = "tfidf",
    embed_model: str = "tfidf",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    notes: list[str] = []

    chunks: list[dict[str, Any]] = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"], chunk_impl, chunk_size, chunk_overlap)):
            chunks.append({"content": c, "metadata": {"source": d.get("source", "text"), "chunk": i}})
    if not chunks:
        raise ValueError("no text could be extracted/chunked from the provided sources")
    texts = [c["content"] for c in chunks]

    store = vectorstore_path(project_id, index_id)
    (store / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    # always build a TF-IDF space (powers offline dense + is the fallback)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50_000)
    matrix = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, store / "tfidf.joblib")
    save_npz(store / "matrix.npz", matrix)

    embed_kind, dim = "tfidf", len(vectorizer.vocabulary_)
    if embed_provider in ("ollama", "openai"):
        s = get_settings()
        dense = _embed_remote(texts, embed_provider, embed_model, s)
        if dense is not None:
            np.save(store / "dense.npy", dense)
            embed_kind, dim = "neural", int(dense.shape[1])
        else:
            notes.append(
                f"embedder '{embed_provider}:{embed_model}' was unreachable — fell back to TF-IDF "
                f"(start Ollama or set an OpenAI key to use it)."
            )
    elif embed_provider not in ("tfidf", "sklearn", ""):
        notes.append(
            f"embedder '{embed_model}' needs a library not installed here (sentence-transformers / API) — "
            f"using TF-IDF. Selection saved in the pipeline config."
        )

    meta = {
        "embed_kind": embed_kind, "embed_provider": embed_provider, "embed_model": embed_model,
        "dim": dim, "n_chunks": len(chunks),
        "chunk_impl": chunk_impl, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
    }
    (store / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {
        **meta, "build_ms": int((time.perf_counter() - t0) * 1000),
        "storage_path": str(store), "vocab_size": len(vectorizer.vocabulary_),
        "sample_chunks": texts[:3], "notes": notes,
    }


# ───────────────────────────── retrieval primitives ────────────────────────
def _load(store: Path):
    meta = json.loads((store / "meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((store / "chunks.json").read_text(encoding="utf-8"))
    return meta, chunks


def _dense_scores(store: Path, meta: dict, queries: list[str], s) -> np.ndarray:
    """Mean cosine score per chunk across all (transformed) queries."""
    if meta["embed_kind"] == "neural":
        dense = np.load(store / "dense.npy")
        qm = _embed_remote(queries, meta["embed_provider"], meta["embed_model"], s)
        if qm is None:  # provider went away since build — fall back to tfidf
            return _tfidf_scores(store, queries)
        return (qm @ dense.T).mean(axis=0)
    return _tfidf_scores(store, queries)


def _tfidf_scores(store: Path, queries: list[str]) -> np.ndarray:
    vectorizer = joblib.load(store / "tfidf.joblib")
    matrix = load_npz(store / "matrix.npz")
    qv = vectorizer.transform(queries)
    return cosine_similarity(qv, matrix).mean(axis=0)


def _bm25_scores(chunks: list[dict], queries: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    docs = [_TOK_RE.findall(c["content"].lower()) for c in chunks]
    N = len(docs)
    dl = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = float(dl.mean()) or 1.0
    df: Counter = Counter()
    tfs = []
    for d in docs:
        tf = Counter(d)
        tfs.append(tf)
        for term in tf:
            df[term] += 1
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    qterms = _TOK_RE.findall(" ".join(queries).lower())
    scores = np.zeros(N, dtype=np.float32)
    for i, tf in enumerate(tfs):
        sc = 0.0
        for t in set(qterms):
            if t in tf:
                f = tf[t]
                sc += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl[i] / avgdl))
        scores[i] = sc
    return scores


def _rrf(*orderings: list[int], n: int) -> list[int]:
    score: dict[int, float] = {}
    for order in orderings:
        for rank, idx in enumerate(order):
            score[idx] = score.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])][:n]


def _mmr(scores: np.ndarray, sim_matrix, k: int, lam: float) -> list[int]:
    cand = np.argsort(-scores)[: min(len(scores), max(k * 4, 20))].tolist()
    sub = cosine_similarity(sim_matrix[cand])
    pos = {c: i for i, c in enumerate(cand)}
    selected: list[int] = []
    while cand and len(selected) < k:
        best, bs = cand[0], -1e9
        for c in cand:
            red = max((sub[pos[c]][pos[x]] for x in selected), default=0.0)
            v = lam * scores[c] - (1 - lam) * red
            if v > bs:
                best, bs = c, v
        selected.append(best)
        cand.remove(best)
    return selected


# ───────────────────────────── query (query-time) ──────────────────────────
def run_query(
    storage_path: str,
    question: str,
    *,
    transform: str = "none",
    transform_n: int = 4,
    retriever: str = "hybrid",
    k: int = 50,
    fusion: str = "rrf",
    alpha: float = 0.5,
    rerank: str = "none",
    rerank_top_n: int = 8,
    compress: str = "none",
    target_tokens: int = 2000,
    generate: bool = True,
    provider: str | None = None,
    model: str | None = None,
    citations: bool = True,
) -> dict[str, Any]:
    store = Path(storage_path)
    s = get_settings()
    meta, chunks = _load(store)
    notes: list[str] = []
    trace: dict[str, Any] = {}
    N = len(chunks)
    k = max(1, min(k, N))

    # ── 1. query transform
    queries, tinfo = _transform_query(question, transform, transform_n, s, notes)
    trace["transform"] = tinfo

    # ── 2. retrieve
    dense = _dense_scores(store, meta, queries, s)
    sparse = _bm25_scores(chunks, queries)
    pool = max(k, rerank_top_n if rerank != "none" else k)

    if retriever in ("dense", "similarity"):
        order = np.argsort(-dense)[:pool].tolist(); primary = dense
    elif retriever == "sparse":
        order = np.argsort(-sparse)[:pool].tolist(); primary = sparse
    elif retriever == "mmr":
        sim = np.load(store / "dense.npy") if meta["embed_kind"] == "neural" else load_npz(store / "matrix.npz")
        order = _mmr(dense, sim, pool, alpha); primary = dense
    else:  # hybrid
        d_order = np.argsort(-dense).tolist()
        s_order = np.argsort(-sparse).tolist()
        if fusion == "weighted":
            dn = dense / (dense.max() or 1); sn = sparse / (sparse.max() or 1)
            combined = alpha * dn + (1 - alpha) * sn
            order = np.argsort(-combined)[:pool].tolist(); primary = combined
        else:  # rrf
            order = _rrf(d_order, s_order, n=pool)
            primary = dense
    trace["retrieved"] = len(order)

    # ── 3. rerank
    if rerank == "llm" and generate is not False:
        order, rinfo = _llm_rerank(question, order, chunks, rerank_top_n, provider, model, s, notes)
        trace["reranked"] = rinfo
    elif rerank not in ("none", ""):
        notes.append(f"reranker '{rerank}' needs a model not installed here — kept retrieval order.")
        order = order[:rerank_top_n]
    else:
        order = order[:rerank_top_n] if rerank_top_n and len(order) > rerank_top_n else order

    docs = [{"content": chunks[i]["content"], "score": round(float(primary[i]), 4),
             "metadata": chunks[i]["metadata"]} for i in order]

    # ── 4. compress
    if compress == "compression" or (compress not in ("none", "") and compress):
        before = sum(len(d["content"]) for d in docs)
        docs = _compress(docs, target_tokens)
        trace["compressed"] = {"chars_before": before, "chars_after": sum(len(d["content"]) for d in docs)}

    # ── 5. generate
    answer = answer_kind = prompt = None
    if generate:
        g = _generate(question, docs, provider, model, citations, s)
        answer, answer_kind, prompt = g["answer"], g["answer_kind"], g["prompt"]
        if g.get("note"):
            notes.append(g["note"])

    return {"docs": docs, "answer": answer, "answer_kind": answer_kind or "none",
            "prompt": prompt, "trace": trace, "notes": notes,
            "embed_kind": meta["embed_kind"]}


def _transform_query(q: str, transform: str, n: int, s, notes: list[str]):
    if transform in ("none", "", None):
        return [q], {"impl": "none", "queries": [q]}
    if transform == "rewrite":
        out = _llm(f"Rewrite this search query to be clearer and more complete. Return only the query:\n{q}", s)
        if out:
            return [out.strip()], {"impl": "rewrite", "queries": [out.strip()]}
    if transform == "hyde":
        out = _llm(f"Write a short hypothetical passage that would answer: {q}", s)
        if out:
            return [q, out.strip()], {"impl": "hyde", "queries": [q, "(hypothetical doc)"]}
    if transform in ("multi_query", "rag_fusion"):
        out = _llm(f"Generate {n} alternative phrasings of this question, one per line:\n{q}", s)
        if out:
            qs = [q] + [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:n]
            return qs, {"impl": transform, "queries": qs}
    notes.append(f"query transform '{transform}' needs an LLM — used the raw query "
                 f"(start Ollama or set an OpenAI key).")
    return [q], {"impl": transform + " (degraded)", "queries": [q]}


def _llm_rerank(q, order, chunks, top_n, provider, model, s, notes):
    listing = "\n".join(f"[{i}] {chunks[idx]['content'][:300]}" for i, idx in enumerate(order[:20]))
    out = _llm(f"Rank these passages by relevance to the question '{q}'. "
               f"Return the bracket numbers in best-first order, comma-separated.\n{listing}", s,
               provider, model)
    if not out:
        notes.append("LLM reranker unreachable — kept retrieval order.")
        return order[:top_n], {"impl": "llm (degraded)"}
    picked = [int(x) for x in re.findall(r"\d+", out)]
    new = [order[i] for i in picked if i < len(order)]
    new += [i for i in order if i not in new]
    return new[:top_n], {"impl": "llm", "kept": top_n}


def _compress(docs: list[dict], target_tokens: int) -> list[dict]:
    budget = _budget(target_tokens)
    out, used = [], 0
    for d in docs:
        if used >= budget:
            break
        room = budget - used
        c = d["content"][:room]
        out.append({**d, "content": c})
        used += len(c)
    return out


def _generate(question, docs, provider, model, citations, s) -> dict[str, Any]:
    context = "\n\n".join(f"[{i + 1}] {d['content']}" for i, d in enumerate(docs))
    cite = "Cite sources inline as [n]. " if citations else ""
    prompt = (f"Answer using ONLY the context. {cite}If it isn't in the context, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    text = _llm(prompt, s, provider, model)
    if text is not None:
        return {"answer": text.strip(), "answer_kind": "llm", "prompt": prompt}
    top = docs[: min(3, len(docs))]
    extractive = "\n\n".join(f"[{i + 1}] {d['content'][:500].strip()}" for i, d in enumerate(top))
    return {"answer": extractive, "answer_kind": "extractive", "prompt": prompt,
            "note": "no LLM reachable — answer is the top passages verbatim (extractive). "
                    "Start Ollama or set an OpenAI key in Settings for synthesis."}


def _llm(prompt: str, s, provider: str | None = None, model: str | None = None) -> str | None:
    provider = provider or s.llm_default_provider
    model = model or s.llm_default_model
    to = httpx.Timeout(60.0, connect=1.5)
    try:
        if provider == "ollama":
            r = httpx.post(f"{s.ollama_base_url}/api/generate",
                           json={"model": model, "prompt": prompt, "stream": False}, timeout=to)
            r.raise_for_status()
            return r.json().get("response", "")
        if provider == "openai" and s.openai_api_key:
            r = httpx.post(f"{s.openai_base_url}/chat/completions",
                           headers={"Authorization": f"Bearer {s.openai_api_key}"},
                           json={"model": model, "messages": [{"role": "user", "content": prompt}]}, timeout=to)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return None
    return None
