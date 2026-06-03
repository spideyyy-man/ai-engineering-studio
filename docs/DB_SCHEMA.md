# Database Schema

Single-user local — no `users` table in v1. All FKs that would point to `users` are pre-included as nullable columns so the migration to multi-user is a config flip, not a schema rewrite.

## Tables

### `projects`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| name | str |  |
| description | text |  |
| created_at | datetime |  |
| updated_at | datetime |  |
| user_id | UUID nullable | reserved for multi-user |

### `datasets`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| project_id | FK projects.id | cascade |
| name | str | |
| source_type | enum | csv, excel, sql, mongo, api, sample |
| storage_path | str | path in storage/datasets |
| n_rows | int | |
| n_cols | int | |
| size_bytes | int | |
| dtypes_json | json | column → dtype |
| profile_json | json | EDA report payload |
| created_at | datetime | |

### `ml_models`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| project_id | FK | |
| dataset_id | FK datasets.id | |
| task_type | enum | classification, regression, clustering |
| algorithm | str | LogisticRegression, RandomForest, ... |
| hyperparams_json | json | |
| metrics_json | json | accuracy/f1/mae/... |
| artifact_path | str | path to .pkl / .onnx |
| code_path | str | path to generated .py |
| status | enum | pending, running, succeeded, failed |
| created_at | datetime | |

### `dl_models`
Same shape as `ml_models` but with extra cols: `architecture`, `epochs`, `batch_size`, `optimizer`, `history_json` (loss/acc per epoch).

### `rag_indexes`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| project_id | FK | |
| name | str | |
| source_type | enum | pdf, docx, txt, csv, url, youtube |
| chunking_strategy | enum | fixed, recursive, semantic |
| embedding_provider | str | ollama, openai, ... |
| embedding_model | str | |
| vector_store | enum | faiss, chroma, qdrant |
| n_chunks | int | |
| storage_path | str | path to vectorstore dir |
| metrics_json | json | retrieval eval |
| created_at | datetime | |

### `agents`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| project_id | FK | |
| name | str | |
| agent_type | enum | react, tool_calling, multi_agent |
| llm_provider | str | |
| llm_model | str | |
| system_prompt | text | |
| tools_json | json | list of tool configs |
| config_json | json | memory, max_iters, etc. |
| created_at | datetime | |

### `agent_runs`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| agent_id | FK | |
| input | text | |
| output | text | |
| trace_json | json | tool calls, tokens |
| total_tokens | int | |
| cost_usd | float | |
| duration_ms | int | |
| status | enum | succeeded, failed |
| created_at | datetime | |

### `deployments`
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| project_id | FK | |
| target_type | enum | fastapi, docker, streamlit, dash |
| artifact_path | str | dir with generated code |
| status | enum | generated, deployed, archived |
| created_at | datetime | |

### `runs` (generic background-job table)
| Col | Type | Notes |
|---|---|---|
| id | UUID PK | |
| project_id | FK | |
| kind | enum | train_ml, train_dl, index_rag, agent_run |
| target_id | UUID | FK to whichever table `kind` implies |
| status | enum | queued, running, succeeded, failed |
| progress_pct | int | 0-100 |
| logs_path | str | path to streamed log file |
| error | text | |
| started_at | datetime | |
| finished_at | datetime | |

## Relationships

```
projects 1──* datasets
projects 1──* ml_models      datasets 1──* ml_models
projects 1──* dl_models      datasets 1──* dl_models
projects 1──* rag_indexes
projects 1──* agents         agents 1──* agent_runs
projects 1──* deployments
projects 1──* runs
```
