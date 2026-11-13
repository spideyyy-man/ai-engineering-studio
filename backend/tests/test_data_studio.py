"""Data Engineering & EDA studio: upload, analysis endpoints, upload safety."""
from pathlib import Path


def _upload(client, project_id, name, content=b"a,b\n1,2\n3,4\n"):
    return client.post(
        "/api/v1/data/upload",
        data={"project_id": project_id},
        files={"file": (name, content, "text/csv")},
    )


def test_upload_reads_shape(client, project_id, classification_csv):
    r = _upload(client, project_id, "flowers.csv", classification_csv)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["n_rows"] == 60
    assert body["n_cols"] == 4


def test_analysis_endpoints(client, dataset_id):
    base = f"/api/v1/data/datasets/{dataset_id}"
    for suffix in ("", "/profile", "/quality", "/recommendations", "/detect-task", "/pipeline-export"):
        r = client.get(base + suffix)
        assert r.status_code == 200, f"{suffix}: {r.text}"


def test_codegen_returns_python(client, dataset_id):
    r = client.post(f"/api/v1/data/datasets/{dataset_id}/codegen", json={"target": "species", "steps": []})
    assert r.status_code == 200, r.text
    assert "import" in r.json()["code"]


def test_upload_strips_directory_from_filename(client, project_id, paths):
    r = _upload(client, project_id, "../../../evil.csv")
    assert r.status_code == 201, r.text
    stored = Path(r.json()["storage_path"]).resolve()
    assert stored.name == "evil.csv"
    assert stored.is_relative_to((paths["storage"] / "projects" / project_id / "datasets").resolve())


def test_upload_does_not_overwrite_same_name(client, project_id):
    first = _upload(client, project_id, "same.csv").json()["storage_path"]
    second = _upload(client, project_id, "same.csv").json()["storage_path"]
    assert first != second
    assert Path(first).exists() and Path(second).exists()


def test_upload_size_limit(client, project_id):
    big = b"a\n" + b"1\n" * 600_000  # ~1.2 MB, limit is 1 MB in tests
    r = _upload(client, project_id, "big.csv", big)
    assert r.status_code == 422
    assert "upload limit" in r.json()["message"]


def test_upload_rejects_unsupported_type(client, project_id):
    r = _upload(client, project_id, "script.py", b"print('hi')")
    assert r.status_code == 422
