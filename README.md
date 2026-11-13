# AI Engineering Studio

A local, developer-friendly web app for prototyping **machine learning**, **deep learning** and **RAG** systems through guided screens, with **readable, exportable Python code at every step**.

Most AutoML tools hide what they do. This one shows you the data issues it found, *why* it recommends a preprocessing step, the model and metrics it produced, and the code to reproduce all of it, so you learn while you build and you own the result.

**Who it's for:** ML and AI engineers who want a quick baseline or prototype, and students or small teams learning how real pipelines fit together.

> **Status: early alpha.** Data, ML, DL and RAG work end to end. Agents and Deployment are placeholders. Expect rough edges, and please report them.

---

## What works today

| Studio | Status | What you can do |
|---|---|---|
| **Data Engineering & EDA** | ✅ Working | Upload CSV / TSV / Excel / Parquet / JSON. Profile columns, detect quality issues, get preprocessing recommendations, apply fixes with history and undo, detect the task type, compute a baseline, and export a scikit-learn preprocessing script. |
| **Machine Learning** | ✅ Working | Train classification, regression and clustering models (scikit-learn, XGBoost, LightGBM) with automatic preprocessing, cross-validation and grid-based tuning. Download the pickled pipeline and a usage script. |
| **Deep Learning** | 🟡 Partial | Pick a preset (ANN, Wide & Deep, CNN, ResNet50 transfer, LSTM, GRU, Transformer, time-series LSTM), edit layers, check shapes, and export runnable Keras code. In-app training works for **tabular** data only and needs TensorFlow installed. |
| **RAG Engineering** | 🟡 Partial | Ingest pasted text, web pages, or local `.txt` / `.md` / `.csv` files. Choose fixed, recursive, sentence or semantic chunking, then retrieve with dense (TF-IDF), sparse (BM25), hybrid (RRF) or MMR. Optional query rewriting, LLM reranking and answer generation through Ollama or OpenAI. FAISS, Chroma, Qdrant and PDF/DOCX parsing are **not wired up yet**. |
| **AI Agents** | ⏳ Planned | Shows the planned agent types and tools only. The API returns "not implemented". |
| **Deployment & MLOps** | ⏳ Planned | Shows the planned deployment targets only. The API returns "not implemented". |
| **Projects Hub** | ✅ Working | Create, rename and delete projects. Every dataset, model and index belongs to a project. |
| **Settings** | ✅ Working | Read-only view of the configured providers, plus an LLM connectivity check. |

See [docs/ROADMAP.md](docs/ROADMAP.md) for what's planned next.

---

## Tech stack

- **Frontend:** React 18, Vite, TypeScript (strict), Tailwind CSS, Zustand, Plotly.js, axios
- **Backend:** FastAPI, SQLAlchemy 2, Pydantic 2, pandas, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras (optional at runtime), Jinja2 for code generation
- **LLMs and embeddings:** Ollama by default (`llama3`, `nomic-embed-text`), OpenAI optional
- **Database:** SQLite, created automatically in `backend/storage/`
- **Storage:** local files under `backend/storage/projects/<project-id>/`

Architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · API: [docs/API.md](docs/API.md) · Database: [docs/DB_SCHEMA.md](docs/DB_SCHEMA.md)

> Some of those docs describe the *target* design (for example the FAISS/Chroma/Qdrant adapters and an Anthropic provider). Where they disagree with the table above, the table is the current truth.

---

## Quick start

Requirements: Python 3.11, Node.js 20+, and optionally [Ollama](https://ollama.com) for the LLM features.

### 1. Configure

```bash
cp .env.example .env
```

The defaults work for local use. Fill in `OPENAI_API_KEY` only if you want OpenAI.

### 2. Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate         # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API runs at <http://localhost:8000>, with interactive docs at <http://localhost:8000/docs>.

> `requirements.txt` is currently large (it includes TensorFlow and several RAG libraries that aren't used yet), so the first install can take a while. Slimming it down is on the list.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI runs at <http://localhost:5180>.

### 4. (Optional) Ollama for RAG answers

```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

### Docker

```bash
docker compose up --build
```

Add `--profile ollama` to also start an Ollama container.

---

## Security notes

This is a **local, single-user tool**. It has **no login**, so:

- Keep it on your own machine. `docker-compose.yml` binds every port to `127.0.0.1` for that reason, and `uvicorn` binds to localhost by default. Don't expose it to a network or the internet.
- The RAG file loader only reads text files inside `backend/storage/`. To let it read another folder, add it to `RAG_ALLOWED_DIRS` in `.env`.
- The RAG URL loader refuses localhost and private-network addresses unless you set `RAG_ALLOW_PRIVATE_URLS=true`.
- Uploads are capped by `MAX_UPLOAD_MB` (default 200 MB).
- Models are saved with pickle. Only load model files that this app created.

---

## Repository layout

```
backend/
  app/
    routers/      HTTP endpoints, one file per studio (thin)
    services/     the real logic: data_engineering, ml_studio, dl_studio, rag_studio, llm, codegen
    models/       SQLAlchemy tables
    schemas/      Pydantic request/response models
    core/         errors, logging, storage paths
  storage/        created at runtime (datasets, models, indexes, SQLite DB)
frontend/
  src/
    pages/        one page per studio
    components/   shared UI
    api/          typed API client, one module per router
    store/        Zustand stores
docs/             architecture, API, DB schema, roadmap
```

---

## Contributing

Contributions are welcome. Good places to start:

- Wire up a real vector store (FAISS or Chroma) in `services/rag_studio/`
- Add PDF/DOCX loading to RAG Studio
- Build the first Deployment generator: a FastAPI `/predict` service for a trained ML model
- Add tests (there are none yet) and a CI workflow

Open an issue first for anything large, so we can agree on the approach.

---

## License

[MIT](LICENSE) © 2026 spideyyy-man and AI Engineering Studio contributors
