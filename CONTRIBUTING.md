# Contributing to AI Engineering Studio

Thanks for wanting to help! This guide explains how to get a change from your machine into the project.

## Ways to contribute

- **Pick up a good first issue.** Look for the [`good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) label.
- **Build a planned studio.** AI Agents and Deployment are placeholders today. Open an issue saying which part you'd like to take, so two people don't build the same thing.
- **Wire up a real backend for RAG Studio:** FAISS or Chroma vector stores, PDF/DOCX loading, a neural reranker.
- **Fix a bug** or improve tests, accessibility or performance.
- **Improve docs:** README, code comments, setup notes.
- **Propose an idea** by opening a feature-request issue.

## Development setup

Requirements: Python 3.11+ and Node.js 20.19+ or 22.13+.

1. Fork the repo and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/ai-engineering-studio.git
   cd ai-engineering-studio
   cp .env.example .env
   ```
2. Backend (in one terminal):
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
   pip install -r requirements-dev.txt
   uvicorn app.main:app --reload --port 8000
   ```
3. Frontend (in another terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open http://localhost:5180.
4. Try it with the sample data in [`samples/`](samples): upload `customer_churn.csv` in Data & EDA, or paste `ml_glossary.md` into RAG Studio.
5. Create a branch from `main`:
   ```bash
   git checkout -b feat/faiss-vector-store
   ```

## Where things live

- `backend/app/routers/`: HTTP endpoints, one file per studio. Keep them thin.
- `backend/app/services/<studio>/`: the real logic. Most changes go here.
- `backend/tests/`: pytest suite. Add a test with every fix or feature.
- `frontend/src/pages/`: one page per studio. `frontend/src/api/` has a typed client per router.

## Guidelines

- **Typed code.** Pydantic schemas for every request and response on the backend, and TypeScript with no `any` on the frontend unless there's a good reason.
- **Show your work.** The point of this project is transparency. New features should explain *why* they did something and, where possible, export runnable code.
- **Fail loudly and helpfully.** Raise `ValidationError` / `NotFoundError` from `app.core.errors` with a message a user can act on, rather than letting a raw exception through.
- **No secrets in code.** Use `.env` (git-ignored) and document any new setting in `.env.example`.
- **Mind the safety rules.** Don't let new endpoints read arbitrary file paths or fetch arbitrary URLs; reuse the helpers in `app/core/paths.py` and `app/services/rag_studio/pipeline.py`.
- **Keep pull requests focused:** one feature or fix per PR.

## Before opening a pull request

```bash
# backend
cd backend && ruff check . && pytest

# frontend
cd frontend && npm run lint && npm run build
```

All of these must pass; CI runs the same checks. Then:

- Write a clear PR title, e.g. `feat(rag): add FAISS vector store`.
- Describe what changed and why, and add screenshots for UI changes.
- Link the issue it closes (`Closes #12`).

### Commit messages

We loosely follow [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

## Code of Conduct

By taking part you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
