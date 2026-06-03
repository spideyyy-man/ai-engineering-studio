# API Surface

All endpoints under `/api/v1`. Full OpenAPI at `/docs` once the server is running.

## Projects (`/projects`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/projects` | List projects |
| POST | `/projects` | Create project `{name, description}` |
| GET | `/projects/{id}` | Project detail |
| DELETE | `/projects/{id}` | Delete project + cascade artifacts |

## Data Engineering (`/data`) — Tab 1

| Method | Path | Purpose |
|---|---|---|
| POST | `/data/upload` | Upload CSV/Excel (multipart). Returns `dataset_id`. |
| GET | `/data/datasets` | List datasets (filterable by `project_id`) |
| GET | `/data/datasets/{id}` | Dataset metadata + sample rows |
| GET | `/data/datasets/{id}/profile` | Full EDA profile JSON (shape, dtypes, missing, describe, correlations, plots) |
| GET | `/data/datasets/{id}/quality` | Quality report: outliers, skew, imbalance, leakage flags |
| GET | `/data/datasets/{id}/recommendations` | Suggested preprocessing steps |
| POST | `/data/datasets/{id}/codegen` | Generate sklearn preprocessing pipeline `.py` from chosen steps |
| GET | `/data/datasets/{id}/report` | Download HTML profile report |
| DELETE | `/data/datasets/{id}` | Delete dataset |

## ML Studio (`/ml`) — Tab 2

| Method | Path | Purpose |
|---|---|---|
| POST | `/ml/detect-task` | `{dataset_id, target_col}` → `classification | regression | clustering` |
| GET | `/ml/algorithms?task=classification` | List supported algorithms for a task |
| POST | `/ml/train` | `{dataset_id, target_col, algorithm, hyperparams, test_size, scaler, ...}` → `run_id` |
| GET | `/ml/runs/{run_id}` | Run status + metrics + artifact paths |
| POST | `/ml/tune` | Grid/random/bayes search |
| GET | `/ml/models/{id}/download?fmt=pkl|onnx` | Artifact download |
| POST | `/ml/models/{id}/codegen` | Generate training `.py` |

## DL Studio (`/dl`) — Tab 3

| Method | Path | Purpose |
|---|---|---|
| POST | `/dl/build` | `{architecture, input_shape, layers, ...}` → preview model summary |
| POST | `/dl/train` | Start training job |
| GET | `/dl/runs/{run_id}` | Status + loss/acc history |
| GET | `/dl/models/{id}/download?fmt=h5|savedmodel|onnx` | |
| POST | `/dl/models/{id}/codegen` | |

## RAG Studio (`/rag`) — Tab 4

| Method | Path | Purpose |
|---|---|---|
| POST | `/rag/ingest` | `{sources, chunking, embedding, vector_store, ...}` → `index_id` |
| GET | `/rag/indexes` | List indexes |
| POST | `/rag/query` | `{index_id, question, k, retriever_type}` → docs + answer |
| POST | `/rag/evaluate` | Recall, context relevance, faithfulness |
| POST | `/rag/codegen` | Generate LangChain / LlamaIndex `.py` |

## Agents (`/agents`) — Tab 5

| Method | Path | Purpose |
|---|---|---|
| POST | `/agents` | Create agent `{name, type, tools, system_prompt, ...}` |
| GET | `/agents` | List agents |
| POST | `/agents/{id}/run` | `{input}` → `run_id` |
| GET | `/agents/runs/{run_id}` | Trace + tokens + cost |

## Deployment (`/deploy`) — Tab 6

| Method | Path | Purpose |
|---|---|---|
| POST | `/deploy/generate` | `{model_id, target}` → FastAPI/Docker artifact dir |
| GET | `/deploy/{id}/download` | Zip download |
| GET | `/deploy/registry` | Model registry view |
| POST | `/deploy/drift-check` | Compare prod data vs training distribution |

## Settings (`/settings`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/settings` | Current LLM/embedding/vectorstore defaults |
| PATCH | `/settings` | Update |
| POST | `/settings/test-llm` | Ping configured LLM provider |
