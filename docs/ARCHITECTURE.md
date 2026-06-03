# Architecture

## 1. High-level

```
                ┌─────────────────────────────────────────────────────────┐
                │                       Frontend                          │
                │   React + Vite + TS + Tailwind + Plotly.js + Zustand     │
                │   8 Studios (Data, ML, DL, RAG, Agents, Deploy, Hub,    │
                │   Settings) — all consume a typed FastAPI client.       │
                └────────────────────────────┬────────────────────────────┘
                                             │ HTTP/JSON (axios)
                                             ▼
                ┌─────────────────────────────────────────────────────────┐
                │                       Backend                           │
                │                       FastAPI                           │
                │  ┌─────────────────────────────────────────────────┐    │
                │  │  routers/  (projects, data, ml, dl, rag,        │    │
                │  │              agents, deployment, settings)      │    │
                │  └────────────────────────┬────────────────────────┘    │
                │                           │                             │
                │  ┌────────────────────────▼────────────────────────┐    │
                │  │   services/  (data_engineering, ml_studio,      │    │
                │  │   dl_studio, rag_studio, agent_studio,          │    │
                │  │   deployment_studio, llm, codegen)              │    │
                │  └──┬────────────┬────────────┬──────────────┬─────┘    │
                │     │            │            │              │          │
                │     ▼            ▼            ▼              ▼          │
                │  ┌─────┐    ┌────────┐    ┌────────┐    ┌────────────┐  │
                │  │ DB  │    │Storage │    │  LLM   │    │ Vectorstore│  │
                │  │SQLi/│    │ Files  │    │ Ollama │    │ FAISS/     │  │
                │  │ PG  │    │.pkl .h5│    │ /OpenAI│    │ Chroma/Qdr │  │
                │  └─────┘    └────────┘    └────────┘    └────────────┘  │
                └─────────────────────────────────────────────────────────┘
```

## 2. Module responsibilities

### Backend services (single source of truth — routers are thin)

| Module | Owns |
|---|---|
| `services/data_engineering/` | Profiling, quality checks, recommendations, code-export for preprocessing pipelines |
| `services/ml_studio/` | Problem detection, model registry, training, tuning, evaluation, pickle/ONNX export |
| `services/dl_studio/` | TF/Keras builders for ANN/CNN/RNN/Transformer + training callbacks |
| `services/rag_studio/` | Loaders, chunkers, embedders, vectorstore adapters, retrievers, evaluation |
| `services/agent_studio/` | Agent runtimes (ReAct, tool-calling), tool registry, traces |
| `services/deployment_studio/` | FastAPI/Streamlit/Docker generators, registry, drift |
| `services/llm/` | Provider abstraction — `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider` share `BaseLLMProvider` |
| `services/codegen/` | Jinja templates for every studio's exported code |

Each service exposes a small Python API. Routers only translate HTTP ↔ Pydantic ↔ service.

## 3. Frontend architecture

- **Routing**: `react-router-dom` with one route per studio.
- **State**: Zustand store per studio (`useDataStore`, `useMLStore`, …). Keeps current project, active dataset, in-flight task IDs.
- **API client**: `src/api/*.ts` — one typed module per backend router. Generated types in `src/api/types.ts` mirror Pydantic schemas.
- **Shared components**: `FileUpload`, `DataTable`, `Card`, `MetricBadge`, `CodeBlock`, `RecommendationsList`.
- **Theme**: Tailwind with custom `studio-{name}` accent colors so each tab has identity.
- **Charts**: `react-plotly.js`. Backend returns Plotly figure JSON; frontend renders.

## 4. Project lifecycle

```
   Create Project
       │
       ▼
   Upload Dataset ──► Profile (Tab 1) ──► Recommend preprocessing
                                                │
                                                ▼
                                Choose path: ML / DL / RAG / Agents
                                                │
                  ┌─────────────┬───────────────┼────────────┬──────────────┐
                  ▼             ▼               ▼            ▼              ▼
              ML Studio     DL Studio      RAG Studio    Agent Studio   Deploy Studio
              (train,      (build NN,      (index,       (compose,      (FastAPI,
              tune, eval)   train, eval)    retrieve,     monitor)       Docker,
                                            evaluate)                    drift)
                  │             │               │            │              │
                  └─────────────┴───────────────┴────────────┴──────────────┘
                                                │
                                                ▼
                                          Artifacts + Code Export
```

Every step (a) records to DB, (b) writes a typed artifact to `storage/`, (c) emits generated code accessible via API.

## 5. Async / long-running work

For training jobs and RAG indexing that exceed a few seconds:

- v1: FastAPI `BackgroundTasks` + in-DB job table (`runs`) with status polling.
- v2: swap in Celery + Redis (or RQ) — service interfaces unchanged.

## 6. Code-generation engine

`services/codegen/` uses Jinja2 templates keyed by `{studio, framework}`:

- `data_engineering/sklearn_pipeline.py.j2`
- `ml_studio/sklearn_train.py.j2`
- `ml_studio/xgboost_train.py.j2`
- `dl_studio/keras_ann.py.j2`
- `rag_studio/langchain.py.j2`
- `rag_studio/llamaindex.py.j2`
- `agent_studio/react_agent.py.j2`
- `deployment_studio/fastapi_app.py.j2`
- `deployment_studio/dockerfile.j2`

The exported `.py` is *runnable* without the studio — that's the whole point.

## 7. Production deployment

```
Cloud Run / ECS / Kubernetes
├── backend (uvicorn + gunicorn workers)
├── frontend (static build behind nginx or Cloudflare Pages)
├── PostgreSQL (managed: RDS / Supabase / Cloud SQL)
├── Object storage (S3 / R2 / GCS) — replaces local storage/
├── Qdrant Cloud (or managed Pinecone) — replaces local FAISS
└── Optional: GPU pool for DL training (Modal, RunPod, GKE)
```

CI: GitHub Actions → build images → push to registry → deploy via Terraform.

## 8. Extensibility (future roadmap)

The service-per-studio + provider abstraction makes it cheap to add:

- GANs / VAEs / Diffusion → new architectures in `dl_studio/architectures/`
- Whisper / multimodal RAG → new loaders in `rag_studio/loaders/`
- Kubernetes orchestration → new target in `deployment_studio/targets/`
- New LLM providers → implement `BaseLLMProvider`
- Distributed / GPU training → new executor in `services/training_runner/`

See [ROADMAP.md](ROADMAP.md) for staged plans.
