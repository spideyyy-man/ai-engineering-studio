import { api } from "./client";

export interface ParamSpec {
  type: string; default?: unknown; min?: number; max?: number; step?: number;
  options?: string[]; label?: string;
}
interface CatItem {
  key: string; label: string; description?: string; recommended?: boolean;
  params?: Record<string, ParamSpec>;
  [k: string]: unknown;
}

export interface RagCatalog {
  pipeline_stages: { n: number; key: string; label: string; phase: string; io: string; purpose: string }[];
  loaders: CatItem[];
  chunking: CatItem[];
  embeddings: (CatItem & { provider: string; dims?: number; type?: string })[];
  embedding_techniques: CatItem[];
  vector_stores: (CatItem & { indexes?: string[] })[];
  ann_indexes: CatItem[];
  retrievers: CatItem[];
  query_transforms: CatItem[];
  rerankers: CatItem[];
  context_processors: CatItem[];
  generation: CatItem[];
  eval_metrics: Record<string, { key: string; label: string }[]>;
  strategies: { key: string; label: string; wiring: string; when: string; recommended?: boolean }[];
  default_stack: Record<string, string>;
}

export interface RagSource { kind: string; location: string }

export interface RagIngestRequest {
  project_id: string; name: string; sources: RagSource[];
  loader?: string;
  chunking_strategy?: string; chunk_size?: number; chunk_overlap?: number;
  enrich?: string;
  embedding_provider?: string; embedding_model?: string; embedding_dims?: number;
  vector_store?: string; ann_index?: string;
}

export interface RagIndex {
  id: string; project_id: string; name: string; source_type: string;
  chunking_strategy: string; embedding_provider: string; embedding_model: string;
  vector_store: string; n_chunks: number; metrics_json: Record<string, unknown>; created_at: string;
}

export interface RagIngestResponse {
  index: RagIndex; n_chunks: number; vocab_size: number; build_ms: number;
  embed_kind: string; dim: number; sample_chunks: string[]; notes: string[];
}

export interface RagRetrievedDoc { content: string; score: number | null; metadata: Record<string, unknown> }

export interface RagQueryRequest {
  index_id: string; question: string;
  strategy?: string;
  query_transform?: string; transform_n?: number;
  retriever_type?: string; k?: number; fusion?: string; hybrid_alpha?: number;
  reranker?: string; rerank_top_n?: number;
  context_processor?: string; compress_target_tokens?: number;
  generate_answer?: boolean; citations?: boolean;
  llm_provider?: string; llm_model?: string;
}

export interface RagQueryResponse {
  docs: RagRetrievedDoc[]; answer: string | null; answer_kind: string;
  retriever: string; k: number; latency_ms: number; prompt: string | null;
  trace: Record<string, unknown>; notes: string[];
}

export const ragApi = {
  catalog: () => api.get<RagCatalog>("/rag/catalog").then((r) => r.data),
  indexes: (project_id?: string) =>
    api.get<RagIndex[]>("/rag/indexes", { params: project_id ? { project_id } : {} }).then((r) => r.data),
  ingest: (p: RagIngestRequest) => api.post<RagIngestResponse>("/rag/ingest", p).then((r) => r.data),
  query: (p: RagQueryRequest) => api.post<RagQueryResponse>("/rag/query", p).then((r) => r.data),
  deleteIndex: (id: string) => api.delete(`/rag/indexes/${id}`).then((r) => r.data),
};
