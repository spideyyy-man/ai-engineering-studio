# AI Engineering Studio

A production-grade, transparent, developer-friendly web platform for building **Machine Learning**, **Deep Learning**, **RAG**, and **AI Agent** systems through guided workflows — with **code generation** at every step.

> **Status:** scaffold + Tab 1 (Data Engineering & EDA Studio) fully implemented. Tabs 2–6 are stubbed with working contracts.

---

## Why this exists

Most AutoML tools hide everything. This one shows you the code, the model, the metrics, and *why* it picked a given preprocessing step — so you learn while you ship.

Target users: **developers, students, startups, and solution-based companies** who want to go from a CSV to a deployable model (or a corpus to a working RAG) without losing visibility into what's happening.

---

## Tabs

| # | Studio | What it does |
|---|---|---|
| 1 | **Data Engineering & EDA** | Upload, profile, detect quality issues, get preprocessing recommendations, export code |
| 2 | **Machine Learning** | Auto-detect problem type, train classical models, tune hyperparams, export pickle/ONNX |
| 3 | **Deep Learning** | ANN / CNN / LSTM / GRU / Transformer with training controls and export |
| 4 | **RAG Engineering** | Ingest → chunk → embed → index → retrieve → evaluate, with Ollama-first defaults |
| 5 | **AI Agents** | ReAct, tool-calling, multi-agent — with prompt templates and monitoring |
| 6 | **Deployment & MLOps** | FastAPI/Docker/Streamlit generation, experiment tracking, drift detection |
| 7 | **Projects Hub** | Project list, artifacts browser |
| 8 | **Settings** | LLM providers, paths, theme |

---

## Tech stack

**Frontend:** React 18 + Vite + TypeScript + TailwindCSS + Plotly.js + Zustand + React Router + axios

**Backend:** FastAPI + SQLAlchemy + scikit-learn + TensorFlow/Keras + pandas + NumPy

**Vector / RAG:** FAISS (default), Chroma, Qdrant (pluggable)

**LLM / Embeddings:** Ollama (default — `llama3`, `nomic-embed-text`); OpenAI optional

**DB:** SQLite (dev), PostgreSQL (prod)

**Storage:** local filesystem; S3-compatible interface for later

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate         # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend at <http://localhost:8000>, OpenAPI docs at <http://localhost:8000/docs>.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

UI at <http://localhost:5180>.

### 3. (Optional) Ollama for RAG / Agents

```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

---

## Repository layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full breakdown, including:

- folder structure
- database schema
- API surface
- data flow diagrams
- development roadmap
- production deployment strategy

---

## License

TBD.
