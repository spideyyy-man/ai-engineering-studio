"""Shared test setup.

Settings and the database engine are created when `app` is first imported, so
the environment must point at a throwaway storage folder *before* that import.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="aies-tests-"))
STORAGE = _TMP / "storage"
ALLOWED = _TMP / "allowed"      # an extra folder RAG may read
OUTSIDE = _TMP / "outside"      # a folder RAG must not read
for _d in (STORAGE, ALLOWED, OUTSIDE):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.update(
    APP_ENV="test",
    DATABASE_URL=f"sqlite:///{(STORAGE / 'test.db').as_posix()}",
    STORAGE_ROOT=str(STORAGE),
    DATASETS_DIR=str(STORAGE / "datasets"),
    MODELS_DIR=str(STORAGE / "models"),
    VECTORSTORES_DIR=str(STORAGE / "vectorstores"),
    ARTIFACTS_DIR=str(STORAGE / "artifacts"),
    MAX_UPLOAD_MB="1",
    RAG_ALLOWED_DIRS=str(ALLOWED),
    RAG_ALLOW_PRIVATE_URLS="false",
    # Point the LLM at a closed port so nothing ever reaches a real Ollama.
    OLLAMA_BASE_URL="http://127.0.0.1:9",
    OPENAI_API_KEY="",
)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def paths():
    return {"storage": STORAGE, "allowed": ALLOWED, "outside": OUTSIDE}


@pytest.fixture()
def project_id(client):
    r = client.post("/api/v1/projects", json={"name": "test project"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _classification_csv() -> bytes:
    rows = ["sepal_len,sepal_wid,petal_len,species"]
    for i in range(60):
        cls = ["setosa", "versicolor", "virginica"][i % 3]
        base = (i % 3) * 2.0
        rows.append(f"{5 + base + (i % 5) * 0.1:.2f},{3 + (i % 4) * 0.1:.2f},{1 + base + (i % 7) * 0.1:.2f},{cls}")
    return ("\n".join(rows) + "\n").encode()


@pytest.fixture(scope="session")
def classification_csv() -> bytes:
    return _classification_csv()


@pytest.fixture()
def dataset_id(client, project_id, classification_csv):
    r = client.post(
        "/api/v1/data/upload",
        data={"project_id": project_id},
        files={"file": ("flowers.csv", classification_csv, "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
