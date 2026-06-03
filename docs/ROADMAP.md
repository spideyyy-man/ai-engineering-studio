# Development Roadmap

## Phase 0 — Scaffold (this session)
- [x] Repo structure, docs, configs
- [x] FastAPI app with routers stubbed
- [x] DB models + migrations
- [x] React shell with 8-tab navigation
- [x] **Tab 1: Data Engineering & EDA — fully working end-to-end**
- [x] LLM provider abstraction (Ollama + OpenAI stubs)
- [x] Code generation engine (Jinja)

## Phase 1 — ML Studio (Tab 2)
- Classification, regression, clustering training
- Hyperparameter tuning (grid + random)
- Pickle + ONNX export
- Generated sklearn training script

## Phase 2 — DL Studio (Tab 3)
- ANN tabular
- CNN image (transfer learning: ResNet, EfficientNet)
- LSTM/GRU text + time series
- Transformer text classifier
- H5 + SavedModel + ONNX export

## Phase 3 — RAG Studio (Tab 4)
- PDF/DOCX/TXT/URL loaders
- Chunking strategies (fixed, recursive, semantic)
- FAISS + Chroma + Qdrant
- MMR + hybrid + reranking
- RAGAS-style evaluation
- LangChain + LlamaIndex code export

## Phase 4 — Agents (Tab 5)
- ReAct loop with tool calling
- Tool registry: search, SQL, RAG, calculator, custom HTTP
- Multi-agent orchestration (supervisor pattern)
- Token & cost accounting per run

## Phase 5 — Deployment & MLOps (Tab 6)
- FastAPI inference server generation
- Dockerfile + compose + requirements
- Lightweight experiment tracker (rows in `runs`)
- Drift detection (KS test, PSI)

## Phase 6 — Productionize
- PostgreSQL migration
- Background workers (Celery + Redis)
- Object storage (S3) adapter
- Auth (JWT + multi-user)
- Cloud deploy templates (Cloud Run, ECS, K8s)

## Phase 7 — Advanced
- GANs / VAEs / Diffusion (DL Studio)
- Whisper + multimodal RAG
- Distributed training (Modal/RunPod runner)
- GPU scheduling
- Kubernetes operator
- Commercial SaaS layer (billing, orgs, projects-per-org)

## Quality bar at each phase
Every studio must ship with:
1. A working end-to-end happy path
2. Generated, runnable `.py` artifact
3. Explanations + recommendations visible in UI
4. At least one downloadable artifact (model/index/code)
5. Errors surfaced in UI, not swallowed
