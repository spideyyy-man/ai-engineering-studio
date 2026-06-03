"""RAG Studio — the component & strategy catalog (the "matplotlib primitives").

Pure data. Every RAG variant (naive → agentic) is a different wiring of the same
canonical pipeline stages; this module is the single registry of:

  • pipeline stages (index-time + query-time)
  • loaders / connectors / modalities
  • chunking strategies (full taxonomy, with tuning knobs)
  • embedding models (2026) + embedding techniques (MRL, quantization, asymmetry)
  • vector stores + ANN index algorithms (with params)
  • retrievers, query transforms, rerankers, context processors
  • generation options + evaluation metrics
  • RAG strategy templates (the taxonomy) + the recommended default stack

Param specs use the same shape as ML/DL Studio's HyperparamSpec
({type, default, min, max, step, label, options}) so the existing frontend
param-form component renders these unchanged. No heavy imports here — the
registry must import instantly whether or not langchain/faiss/etc. are present.
"""
from __future__ import annotations

from typing import Any

# ───────────────────────────── pipeline stages ─────────────────────────────
# Each RAG pipeline is a DAG of these typed stages. `phase` = index | query.
PIPELINE_STAGES: list[dict[str, Any]] = [
    # index-time
    {"n": 1, "key": "connect", "label": "Connect / Ingest", "phase": "index",
     "io": "source → raw bytes", "purpose": "Pull from files, web, APIs, DBs, SaaS"},
    {"n": 2, "key": "parse", "label": "Parse / Load", "phase": "index",
     "io": "bytes → structured doc", "purpose": "Extract text + layout + tables + images"},
    {"n": 3, "key": "clean", "label": "Clean / Normalize", "phase": "index",
     "io": "doc → clean doc", "purpose": "Dedup, boilerplate strip, encoding fixes"},
    {"n": 4, "key": "chunk", "label": "Chunk / Split", "phase": "index",
     "io": "doc → chunks", "purpose": "Break into retrievable units"},
    {"n": 5, "key": "enrich", "label": "Enrich / Metadata", "phase": "index",
     "io": "chunks → annotated chunks", "purpose": "Titles, summaries, entities, contextual prefix"},
    {"n": 6, "key": "embed", "label": "Embed", "phase": "index",
     "io": "chunks → vectors", "purpose": "Dense / sparse / multi-vector"},
    {"n": 7, "key": "store", "label": "Index / Store", "phase": "index",
     "io": "vectors → index", "purpose": "ANN index + keyword index + doc store"},
    # query-time
    {"n": 8, "key": "query_transform", "label": "Query transform", "phase": "query",
     "io": "query → query/queries", "purpose": "Rewrite, expand, decompose, route"},
    {"n": 9, "key": "retrieve", "label": "Retrieve", "phase": "query",
     "io": "query → candidates", "purpose": "Dense + sparse + filtered search"},
    {"n": 10, "key": "rerank", "label": "Rerank", "phase": "query",
     "io": "candidates → ranked", "purpose": "Precision boost on top-N"},
    {"n": 11, "key": "compress", "label": "Compress / Filter", "phase": "query",
     "io": "ranked → context", "purpose": "Fit window, drop noise, dedup"},
    {"n": 12, "key": "assemble", "label": "Assemble", "phase": "query",
     "io": "context + query → prompt", "purpose": "Build grounded prompt"},
    {"n": 13, "key": "generate", "label": "Generate", "phase": "query",
     "io": "prompt → answer", "purpose": "LLM call, structured output"},
    {"n": 14, "key": "post_process", "label": "Post-process", "phase": "query",
     "io": "answer → answer+citations", "purpose": "Cite, verify, guardrail"},
    {"n": 15, "key": "evaluate", "label": "Evaluate", "phase": "query",
     "io": "trace → scores", "purpose": "Faithfulness, relevance, etc. (offline/online)"},
]

# ───────────────────────────── loaders / connectors ─────────────────────────
LOADERS: list[dict[str, Any]] = [
    {"key": "pymupdf", "label": "PDF — PyMuPDF", "kinds": ["pdf"], "lib": "pymupdf",
     "description": "Fast text extraction from PDFs."},
    {"key": "pdfplumber", "label": "PDF — pdfplumber", "kinds": ["pdf"], "lib": "pdfplumber",
     "description": "Table-aware PDF extraction."},
    {"key": "docling", "label": "Layout — Docling", "kinds": ["pdf", "docx", "pptx", "html"], "lib": "docling",
     "description": "Structure-aware parsing across many types (single API)."},
    {"key": "vision_llm", "label": "Vision-LLM parse", "kinds": ["pdf", "image"], "lib": "openai|gemini",
     "description": "GPT-4o / Gemini parsing for scanned / complex docs."},
    {"key": "ocr_tesseract", "label": "OCR — Tesseract", "kinds": ["image", "pdf"], "lib": "pytesseract",
     "description": "OCR fallback for scanned pages."},
    {"key": "docx", "label": "Word — python-docx", "kinds": ["docx"], "lib": "python-docx",
     "description": "DOCX text + structure."},
    {"key": "pptx", "label": "PowerPoint — python-pptx", "kinds": ["pptx"], "lib": "python-pptx",
     "description": "Slide text extraction."},
    {"key": "xlsx", "label": "Excel — openpyxl/pandas", "kinds": ["xlsx", "csv"], "lib": "openpyxl",
     "description": "Spreadsheet rows as structured records."},
    {"key": "trafilatura", "label": "Web — trafilatura", "kinds": ["url", "html"], "lib": "trafilatura",
     "description": "Clean main-content extraction from web pages."},
    {"key": "beautifulsoup", "label": "HTML — BeautifulSoup", "kinds": ["html"], "lib": "beautifulsoup4",
     "description": "Raw HTML parsing."},
    {"key": "firecrawl", "label": "Crawler — Firecrawl", "kinds": ["url"], "lib": "firecrawl",
     "description": "JS-heavy site crawling to markdown."},
    {"key": "treesitter", "label": "Code — tree-sitter", "kinds": ["code"], "lib": "tree-sitter",
     "description": "AST-aware code chunking by function/class."},
    {"key": "unstructured", "label": "Universal — unstructured", "kinds": ["any"], "lib": "unstructured",
     "description": "Single API across many file types."},
    {"key": "youtube", "label": "YouTube transcript", "kinds": ["youtube"], "lib": "youtube-transcript-api",
     "description": "Pull a video transcript as text."},
    {"key": "whisper", "label": "Audio — Whisper", "kinds": ["audio", "video"], "lib": "openai-whisper",
     "description": "Speech-to-text for audio/video."},
]

CONNECTORS: list[dict[str, Any]] = [
    {"key": "s3", "label": "S3 / GCS / Azure Blob", "incremental": True},
    {"key": "gdrive", "label": "Google Drive", "incremental": True},
    {"key": "sharepoint", "label": "SharePoint", "incremental": True},
    {"key": "notion", "label": "Notion", "incremental": True},
    {"key": "confluence", "label": "Confluence", "incremental": True},
    {"key": "slack", "label": "Slack", "incremental": True},
    {"key": "gmail", "label": "Gmail", "incremental": True},
    {"key": "postgres", "label": "Postgres / MySQL", "incremental": True},
    {"key": "mongodb", "label": "MongoDB", "incremental": True},
    {"key": "github", "label": "GitHub", "incremental": True},
    {"key": "salesforce", "label": "Salesforce", "incremental": True},
    {"key": "zendesk", "label": "Zendesk", "incremental": True},
]

MODALITIES = ["text", "image", "audio", "video", "table", "code"]

# ───────────────────────────── chunking taxonomy ───────────────────────────
_CHUNK_SIZE = {"type": "int", "default": 512, "min": 64, "max": 4096, "step": 32, "label": "Chunk size (tokens)"}
_OVERLAP = {"type": "int", "default": 64, "min": 0, "max": 512, "step": 8, "label": "Overlap (tokens)"}

CHUNKING_STRATEGIES: list[dict[str, Any]] = [
    {"key": "fixed", "label": "Fixed-size", "category": "basic",
     "description": "Every N chars/tokens with fixed overlap. Trivial; baseline only.",
     "params": {"size": _CHUNK_SIZE, "overlap": _OVERLAP}},
    {"key": "sliding", "label": "Sliding window", "category": "basic",
     "description": "Fixed size with overlap stride; preserves boundary context.",
     "params": {"size": _CHUNK_SIZE, "overlap": {**_OVERLAP, "default": 128}}},
    {"key": "recursive", "label": "Recursive character", "category": "structural", "recommended": True,
     "description": "Split on a hierarchy of separators (¶ → sentence → word). Pragmatic default.",
     "params": {"size": _CHUNK_SIZE, "overlap": _OVERLAP,
                "separators": {"type": "text", "default": "\\n\\n,\\n, ,", "label": "Separators (csv)"}}},
    {"key": "sentence", "label": "Sentence-based", "category": "structural",
     "description": "Split on sentence boundaries (spaCy/NLTK). Clean units.",
     "params": {"max_sentences": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1, "label": "Max sentences"}}},
    {"key": "sentence_window", "label": "Sentence-window", "category": "structural",
     "description": "Index one sentence, return ±N neighbours. Precision + context.",
     "params": {"window": {"type": "int", "default": 3, "min": 1, "max": 10, "step": 1, "label": "± window"}}},
    {"key": "semantic", "label": "Semantic", "category": "semantic",
     "description": "Embed sentences, cut where embedding distance spikes. Topic-coherent.",
     "params": {"threshold": {"type": "float", "default": 0.75, "min": 0.5, "max": 0.99, "step": 0.01, "label": "Breakpoint threshold"}}},
    {"key": "structure", "label": "Structure / layout-aware", "category": "structural",
     "description": "Split on markdown headers / HTML tags / PDF layout / code AST.",
     "params": {}},
    {"key": "hierarchical", "label": "Hierarchical (parent-child)", "category": "advanced",
     "description": "Multi-level small children + larger parents. Best precision+context.",
     "params": {"child_size": {**_CHUNK_SIZE, "default": 256, "label": "Child size"},
                "parent_size": {**_CHUNK_SIZE, "default": 2048, "label": "Parent size"}}},
    {"key": "proposition", "label": "Proposition / agentic", "category": "advanced",
     "description": "LLM extracts atomic standalone facts as chunks. High quality, expensive.",
     "params": {"model": {"type": "text", "default": "haiku", "label": "Extractor model"}}},
    {"key": "late", "label": "Late chunking", "category": "advanced",
     "description": "Embed whole doc first, then pool token-embeddings into chunks (needs long-context embedder).",
     "params": {"size": _CHUNK_SIZE}},
    {"key": "contextual", "label": "Contextual chunking", "category": "advanced", "recommended": True,
     "description": "Prepend an LLM-generated context blurb per chunk before embedding. Big quality gain.",
     "params": {"size": _CHUNK_SIZE, "overlap": _OVERLAP,
                "model": {"type": "text", "default": "haiku", "label": "Context model"},
                "cache": {"type": "bool", "default": True, "label": "Prompt cache"}}},
    {"key": "modality", "label": "Modality-specific", "category": "advanced",
     "description": "Tables kept whole + linearized; code by function; images captioned.",
     "params": {}},
]

# ───────────────────────────── embedding models ────────────────────────────
EMBEDDING_MODELS: list[dict[str, Any]] = [
    # dense (API)
    {"key": "gemini-embedding-2", "provider": "google", "type": "dense-api", "dims": 3072,
     "label": "Gemini Embedding 2 (multimodal)", "multimodal": True},
    {"key": "cohere-embed-v4", "provider": "cohere", "type": "dense-api", "dims": 1536, "label": "Cohere embed-v4"},
    {"key": "voyage-3-large", "provider": "voyage", "type": "dense-api", "dims": 1024, "label": "Voyage-3-large"},
    {"key": "text-embedding-3-large", "provider": "openai", "type": "dense-api", "dims": 3072,
     "label": "OpenAI text-embedding-3-large"},
    {"key": "text-embedding-3-small", "provider": "openai", "type": "dense-api", "dims": 1536,
     "label": "OpenAI text-embedding-3-small"},
    # dense (open / self-host)
    {"key": "qwen3-embedding-4b", "provider": "sentence-transformers", "type": "dense-open", "dims": 2560,
     "label": "Qwen3-Embedding 4B"},
    {"key": "bge-m3", "provider": "sentence-transformers", "type": "dense-open", "dims": 1024, "label": "BGE-M3"},
    {"key": "jina-v3", "provider": "sentence-transformers", "type": "dense-open", "dims": 1024, "label": "Jina v3"},
    {"key": "nomic-embed-text", "provider": "ollama", "type": "dense-open", "dims": 768, "label": "Nomic Embed (Ollama)"},
    {"key": "bge-small-en", "provider": "sentence-transformers", "type": "dense-open", "dims": 384, "label": "BGE small"},
    {"key": "all-MiniLM-L6-v2", "provider": "sentence-transformers", "type": "dense-open", "dims": 384, "label": "MiniLM-L6"},
    # sparse / learned-sparse
    {"key": "splade", "provider": "sentence-transformers", "type": "sparse", "label": "SPLADE (learned sparse)"},
    {"key": "bm25", "provider": "rank-bm25", "type": "sparse", "label": "BM25 (lexical baseline)"},
    # multi-vector / late interaction
    {"key": "colbert-v2", "provider": "ragatouille", "type": "multi-vector", "label": "ColBERTv2 (late interaction)"},
    {"key": "colpali", "provider": "ragatouille", "type": "multi-vector", "label": "ColPali (visual docs)"},
    # multimodal
    {"key": "clip", "provider": "sentence-transformers", "type": "multimodal", "label": "CLIP (text+image)"},
    {"key": "jina-clip", "provider": "sentence-transformers", "type": "multimodal", "label": "Jina-CLIP"},
]

EMBEDDING_TECHNIQUES: list[dict[str, Any]] = [
    {"key": "mrl", "label": "Matryoshka (MRL)",
     "description": "Truncate dims (e.g. 3072→512) to trade accuracy for storage/speed without re-embedding.",
     "params": {"dims": {"type": "int", "default": 512, "min": 64, "max": 4096, "step": 64, "label": "Truncated dims"}}},
    {"key": "asymmetry", "label": "Query/doc asymmetry",
     "description": "Instruction-tuned models embed queries and docs differently; expose the prompt prefix.",
     "params": {"query_prefix": {"type": "text", "default": "", "label": "Query prefix"},
                "doc_prefix": {"type": "text", "default": "", "label": "Doc prefix"}}},
    {"key": "quantization", "label": "Quantization",
     "description": "float32 → int8 / binary vectors. Binary + rescoring cuts storage ~32× with small recall loss.",
     "params": {"mode": {"type": "select", "default": "none", "options": ["none", "int8", "binary"], "label": "Mode"}}},
]

# ───────────────────────────── vector stores / ANN ─────────────────────────
VECTOR_STORES: list[dict[str, Any]] = [
    {"key": "pgvector", "label": "pgvector (Postgres)", "indexes": ["hnsw", "ivfflat"],
     "description": "Best default ≤~50M vectors; no extra service; SQL filters."},
    {"key": "qdrant", "label": "Qdrant", "indexes": ["hnsw"], "recommended": True,
     "description": "Dedicated prod, fast filtered queries, single binary."},
    {"key": "milvus", "label": "Milvus / Zilliz", "indexes": ["hnsw", "ivf", "ivfpq", "diskann"],
     "description": "Billion-scale, distributed, K8s."},
    {"key": "weaviate", "label": "Weaviate", "indexes": ["hnsw"], "description": "Built-in hybrid + modules."},
    {"key": "pinecone", "label": "Pinecone", "indexes": ["proprietary"], "description": "Zero-ops managed, serverless."},
    {"key": "chroma", "label": "Chroma", "indexes": ["hnsw"], "description": "Easiest local dev; small prod."},
    {"key": "lancedb", "label": "LanceDB", "indexes": ["ivfpq"], "description": "Embedded / edge / data-lake."},
    {"key": "faiss", "label": "FAISS (local)", "indexes": ["flat", "hnsw", "ivf", "ivfpq"], "recommended": True,
     "description": "In-process, fast, file-backed. Default local."},
    {"key": "elasticsearch", "label": "Elasticsearch / OpenSearch", "indexes": ["hnsw"],
     "description": "Keyword-first legacy stacks (HNSW + BM25)."},
]

ANN_INDEXES: list[dict[str, Any]] = [
    {"key": "flat", "label": "Flat / brute-force",
     "description": "Exact; small data / ground truth.", "params": {}},
    {"key": "hnsw", "label": "HNSW", "recommended": True,
     "description": "Graph-based; best recall/latency default.",
     "params": {"m": {"type": "int", "default": 16, "min": 4, "max": 64, "step": 2, "label": "M"},
                "ef_construction": {"type": "int", "default": 200, "min": 16, "max": 1000, "step": 16, "label": "ef_construction"},
                "ef_search": {"type": "int", "default": 64, "min": 8, "max": 1000, "step": 8, "label": "ef_search"}}},
    {"key": "ivf", "label": "IVF", "description": "Cluster + search nearest lists; memory-efficient.",
     "params": {"nlist": {"type": "int", "default": 100, "min": 1, "max": 65536, "step": 1, "label": "nlist"},
                "nprobe": {"type": "int", "default": 8, "min": 1, "max": 1024, "step": 1, "label": "nprobe"}}},
    {"key": "ivfpq", "label": "IVF-PQ", "description": "IVF + product quantization; memory-efficient at scale.",
     "params": {"nlist": {"type": "int", "default": 1024, "min": 1, "max": 65536, "step": 1, "label": "nlist"},
                "m": {"type": "int", "default": 16, "min": 1, "max": 96, "step": 1, "label": "PQ subquantizers"}}},
    {"key": "diskann", "label": "DiskANN / Vamana", "description": "Billion-scale on SSD.", "params": {}},
    {"key": "scann", "label": "ScaNN", "description": "Google anisotropic quantization; fast.", "params": {}},
]

# ───────────────────────────── retrievers ──────────────────────────────────
RETRIEVERS: list[dict[str, Any]] = [
    {"key": "dense", "label": "Dense (vector)", "description": "Cosine/dot-product ANN.",
     "params": {"k": {"type": "int", "default": 10, "min": 1, "max": 200, "step": 1, "label": "Top-k"}}},
    {"key": "sparse", "label": "Sparse / lexical", "description": "BM25 / SPLADE.",
     "params": {"k": {"type": "int", "default": 10, "min": 1, "max": 200, "step": 1, "label": "Top-k"}}},
    {"key": "hybrid", "label": "Hybrid (dense + sparse)", "recommended": True,
     "description": "Combine dense + BM25/SPLADE via RRF or weighted fusion. Usually the biggest quality win.",
     "params": {"dense_k": {"type": "int", "default": 50, "min": 1, "max": 200, "step": 1, "label": "Dense k"},
                "sparse_k": {"type": "int", "default": 50, "min": 1, "max": 200, "step": 1, "label": "Sparse k"},
                "fusion": {"type": "select", "default": "rrf", "options": ["rrf", "weighted"], "label": "Fusion"},
                "alpha": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Dense weight (α)"}}},
    {"key": "metadata", "label": "Metadata-filtered", "description": "Pre/post filter by tenant, date, source, ACL.",
     "params": {}},
    {"key": "mmr", "label": "MMR (diverse)", "description": "Diversity-aware to reduce redundancy.",
     "params": {"k": {"type": "int", "default": 10, "min": 1, "max": 100, "step": 1, "label": "Top-k"},
                "lambda": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "λ (relevance↔diversity)"}}},
    {"key": "auto", "label": "Auto-retrieval", "description": "LLM infers metadata filters from the query.", "params": {}},
]

# ───────────────────────────── query transforms ────────────────────────────
QUERY_TRANSFORMS: list[dict[str, Any]] = [
    {"key": "none", "label": "None", "cost": "0", "description": "Use the raw query."},
    {"key": "rewrite", "label": "Query rewrite", "cost": "1 call", "description": "Clean/expand the raw query."},
    {"key": "hyde", "label": "HyDE", "cost": "1 call", "description": "Generate a hypothetical answer, embed that to retrieve."},
    {"key": "multi_query", "label": "Multi-query", "cost": "N retrievals",
     "description": "Generate N paraphrases, retrieve each, union.",
     "params": {"n": {"type": "int", "default": 4, "min": 2, "max": 10, "step": 1, "label": "Variants"}}},
    {"key": "rag_fusion", "label": "RAG-Fusion", "cost": "N + fusion", "recommended": True,
     "description": "Multi-query + RRF over merged results.",
     "params": {"n": {"type": "int", "default": 4, "min": 2, "max": 10, "step": 1, "label": "Variants"}}},
    {"key": "step_back", "label": "Step-back prompting", "cost": "1 call", "description": "Ask a broader question first for context."},
    {"key": "decomposition", "label": "Decomposition", "cost": "N sub-queries", "description": "Split a complex query into sub-questions."},
    {"key": "routing", "label": "Routing", "cost": "1 classifier", "description": "Classify query → pick index / strategy / datasource."},
]

# ───────────────────────────── rerankers ───────────────────────────────────
RERANKERS: list[dict[str, Any]] = [
    {"key": "none", "label": "None", "params": {}},
    {"key": "cross_encoder", "label": "Cross-encoder (BGE/MiniLM/Jina)", "selfhost": True, "recommended": True,
     "description": "Query+doc scored jointly. Self-hostable.",
     "params": {"top_n": {"type": "int", "default": 8, "min": 1, "max": 50, "step": 1, "label": "Keep top-N"}}},
    {"key": "cohere_rerank", "label": "Cohere Rerank", "selfhost": False,
     "description": "Hosted reranker; strong, easy.",
     "params": {"top_n": {"type": "int", "default": 8, "min": 1, "max": 50, "step": 1, "label": "Keep top-N"}}},
    {"key": "voyage_rerank", "label": "Voyage Rerank", "selfhost": False, "description": "Hosted reranker.",
     "params": {"top_n": {"type": "int", "default": 8, "min": 1, "max": 50, "step": 1, "label": "Keep top-N"}}},
    {"key": "colbert", "label": "ColBERT late interaction", "selfhost": True,
     "description": "Token-level MaxSim; doubles as retriever + reranker.", "params": {}},
    {"key": "llm_rerank", "label": "LLM rerank (RankGPT / listwise)", "selfhost": False,
     "description": "LLM orders candidates; highest quality, highest cost.", "params": {}},
]

# ───────────────────────────── context processing ──────────────────────────
CONTEXT_PROCESSORS: list[dict[str, Any]] = [
    {"key": "contextual_retrieval", "label": "Contextual Retrieval",
     "description": "Prepend an LLM context blurb before embedding; contextual embeddings + BM25 → big failure drop."},
    {"key": "parent_document", "label": "Parent-document (small-to-big)",
     "description": "Retrieve small chunks, return their larger parents."},
    {"key": "sentence_window", "label": "Sentence-window",
     "description": "Retrieve a sentence, expand to ±N neighbours."},
    {"key": "compression", "label": "Compression (LLMLingua)",
     "description": "Extractive filtering to drop irrelevant tokens, fit the window, cut cost.",
     "params": {"target_tokens": {"type": "int", "default": 2000, "min": 256, "max": 16000, "step": 128, "label": "Target tokens"}}},
    {"key": "dedup", "label": "Dedup & dedensify", "description": "Remove near-duplicate chunks."},
]

# ───────────────────────────── generation ──────────────────────────────────
GENERATION: list[dict[str, Any]] = [
    {"key": "model_router", "label": "Model router",
     "description": "Pick model by cost/latency/quality; fallback chains."},
    {"key": "grounded_prompt", "label": "Grounded prompt template",
     "description": "Answer only from context; refuse if unsupported."},
    {"key": "citations", "label": "Citations",
     "description": "Map sentences back to source chunks (span-level if possible)."},
    {"key": "structured_output", "label": "Structured output",
     "description": "JSON schema / function-calling for typed answers."},
    {"key": "streaming", "label": "Streaming + token budgeting",
     "description": "Stream tokens; cap context/output budget."},
]

# ───────────────────────────── evaluation ──────────────────────────────────
EVAL_METRICS: dict[str, list[dict[str, str]]] = {
    "retrieval": [
        {"key": "recall_at_k", "label": "Recall@k"},
        {"key": "precision_at_k", "label": "Precision@k"},
        {"key": "mrr", "label": "MRR"},
        {"key": "ndcg", "label": "NDCG"},
        {"key": "hit_rate", "label": "Hit rate"},
    ],
    "generation": [
        {"key": "faithfulness", "label": "Faithfulness"},
        {"key": "answer_relevance", "label": "Answer Relevance"},
        {"key": "context_precision", "label": "Context Precision"},
        {"key": "context_recall", "label": "Context Recall"},
        {"key": "context_relevance", "label": "Context Relevance"},
    ],
    "online": [
        {"key": "thumbs", "label": "Thumbs up/down"},
        {"key": "ctr", "label": "Click-through"},
        {"key": "escalation", "label": "Escalation rate"},
        {"key": "groundedness", "label": "Groundedness check"},
    ],
}

# ───────────────────────────── RAG strategy taxonomy (templates) ───────────
RAG_STRATEGIES: list[dict[str, Any]] = [
    {"key": "naive", "label": "Naive / Standard", "wiring": "chunk → embed → retrieve top-k → stuff → generate",
     "when": "Baseline, simple Q&A"},
    {"key": "advanced", "label": "Advanced", "wiring": "+ query transform (pre) + rerank/compress (post)",
     "when": "Default production quality", "recommended": True},
    {"key": "modular", "label": "Modular", "wiring": "Stages swappable/reorderable as a graph",
     "when": "The platform's native model"},
    {"key": "hybrid", "label": "Hybrid", "wiring": "dense + BM25/SPLADE → RRF", "when": "Almost always better than dense-only"},
    {"key": "rag_fusion", "label": "Multi-query / RAG-Fusion", "wiring": "N query variants → union/RRF",
     "when": "Ambiguous or broad queries"},
    {"key": "hyde", "label": "HyDE", "wiring": "embed a hypothetical answer", "when": "Sparse/jargon corpora, weak queries"},
    {"key": "self_rag", "label": "Self-RAG", "wiring": "model emits reflection tokens; decides to retrieve + critiques",
     "when": "Reduce hallucination, adaptive retrieval"},
    {"key": "crag", "label": "Corrective RAG (CRAG)", "wiring": "grade docs; if weak → web/fallback search",
     "when": "Robustness when KB is incomplete"},
    {"key": "adaptive", "label": "Adaptive RAG", "wiring": "router classifies complexity → none / single / multi-step",
     "when": "Cost + latency optimization"},
    {"key": "agentic", "label": "Agentic RAG", "wiring": "LLM agent with tools, multi-hop, ReAct loop",
     "when": "Complex, multi-source, reasoning tasks"},
    {"key": "graph", "label": "Graph RAG", "wiring": "build knowledge graph + community summaries; query graph",
     "when": "'Connect the dots' / global questions"},
    {"key": "raptor", "label": "Hierarchical / RAPTOR", "wiring": "recursive cluster+summarize tree; retrieve multi-level",
     "when": "Long docs, multi-granularity questions"},
    {"key": "parent_document", "label": "Parent-document", "wiring": "retrieve small, return big",
     "when": "Precision retrieval + rich context"},
    {"key": "contextual", "label": "Contextual Retrieval", "wiring": "contextual chunk prefixes + hybrid + rerank",
     "when": "Strong general-purpose upgrade"},
    {"key": "multimodal", "label": "Multimodal RAG", "wiring": "image/table/chart embedding + retrieval",
     "when": "PDFs with figures, scanned docs, slides"},
    {"key": "cag", "label": "Long-context / CAG", "wiring": "skip retrieval, load corpus into context + prompt cache",
     "when": "Small, stable corpora"},
    {"key": "conversational", "label": "Conversational / memory RAG", "wiring": "history-aware rewrite + session memory",
     "when": "Chat assistants"},
]

# ───────────────────────────── recommended default stack (v1) ──────────────
DEFAULT_STACK: dict[str, str] = {
    "parsing": "docling",
    "chunking": "recursive (512/64) + contextual prefix",
    "embedding": "voyage-3-large (or qwen3-embedding self-host)",
    "vector_db": "qdrant (or pgvector if already on Postgres)",
    "retrieval": "hybrid (dense + BM25) → RRF",
    "rerank": "cohere_rerank (or BGE-reranker)",
    "generation": "frontier model w/ citation prompt + router",
    "eval": "RAGAS-style metrics + auto golden set",
    "tracing": "Langfuse / Phoenix + OpenTelemetry",
}


def catalog() -> dict[str, Any]:
    """Everything the frontend needs to render the RAG builder + strategy picker."""
    return {
        "pipeline_stages": PIPELINE_STAGES,
        "loaders": LOADERS,
        "connectors": CONNECTORS,
        "modalities": MODALITIES,
        "chunking": CHUNKING_STRATEGIES,
        "embeddings": EMBEDDING_MODELS,
        "embedding_techniques": EMBEDDING_TECHNIQUES,
        "vector_stores": VECTOR_STORES,
        "ann_indexes": ANN_INDEXES,
        "retrievers": RETRIEVERS,
        "query_transforms": QUERY_TRANSFORMS,
        "rerankers": RERANKERS,
        "context_processors": CONTEXT_PROCESSORS,
        "generation": GENERATION,
        "eval_metrics": EVAL_METRICS,
        "strategies": RAG_STRATEGIES,
        "default_stack": DEFAULT_STACK,
    }
