"""RAG studio: build an index, query it, and the source-safety rules."""
import pytest

DOC = (
    "Gradient boosting builds an ensemble of shallow trees, each one correcting the last. "
    "Random forests average many deep trees trained on bootstrap samples. "
    "Logistic regression is a linear model for classification that outputs probabilities. "
    "K-means clustering groups points around a fixed number of centroids."
)


def _ingest(client, project_id, kind, location):
    return client.post(
        "/api/v1/rag/ingest",
        json={"project_id": project_id, "name": "idx", "sources": [{"kind": kind, "location": location}],
              "chunking_strategy": "sentence", "chunk_size": 32, "chunk_overlap": 0},
    )


def test_ingest_and_query_text(client, project_id):
    r = _ingest(client, project_id, "text", DOC)
    assert r.status_code == 200, r.text
    index_id = r.json()["index"]["id"]
    assert r.json()["n_chunks"] >= 1

    for retriever in ("dense", "sparse", "hybrid", "mmr"):
        q = client.post(
            "/api/v1/rag/query",
            json={"index_id": index_id, "question": "what averages bootstrap trees?",
                  "retriever_type": retriever, "k": 3, "generate_answer": False},
        )
        assert q.status_code == 200, f"{retriever}: {q.text}"
        assert q.json()["docs"], retriever


def test_file_inside_allowed_dir_is_read(client, project_id, paths):
    f = paths["allowed"] / "notes.md"
    f.write_text(DOC, encoding="utf-8")
    assert _ingest(client, project_id, "file", str(f)).status_code == 200


def test_file_inside_storage_is_read(client, project_id, paths):
    f = paths["storage"] / "inside.txt"
    f.write_text(DOC, encoding="utf-8")
    assert _ingest(client, project_id, "txt", "inside.txt").status_code == 200


@pytest.mark.parametrize("name", ["secret.txt", "../outside/secret.txt"])
def test_file_outside_allowed_dirs_is_refused(client, project_id, paths, name):
    (paths["outside"] / "secret.txt").write_text("do not read", encoding="utf-8")
    target = paths["outside"] / "secret.txt" if name == "secret.txt" else paths["allowed"] / name
    r = _ingest(client, project_id, "file", str(target))
    assert r.status_code == 422
    assert "outside the folders" in r.json()["message"]


def test_non_text_file_is_refused(client, project_id, paths):
    key = paths["allowed"] / "id_rsa"
    key.write_text("-----BEGIN KEY-----", encoding="utf-8")
    r = _ingest(client, project_id, "file", str(key))
    assert r.status_code == 422


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/healthz",
    "http://169.254.169.254/latest/meta-data",
    "http://10.1.2.3/",
    "http://[::1]/",
    "file:///etc/passwd",
])
def test_private_and_non_http_urls_are_refused(client, project_id, url):
    r = _ingest(client, project_id, "url", url)
    assert r.status_code == 422
