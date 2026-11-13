"""DL studio: scaffold, shape validation and code export (no TensorFlow needed)."""


def test_scaffold_validate_codegen(client):
    r = client.post(
        "/api/v1/dl/scaffold",
        json={"architecture": "ann", "data_type": "tabular", "input_shape": [4], "n_classes": 3},
    )
    assert r.status_code == 200, r.text
    spec = r.json()

    v = client.post("/api/v1/dl/validate", json={"spec": spec})
    assert v.status_code == 200, v.text
    report = v.json()
    assert report["ok"], report["errors"]
    assert report["total_params"] > 0

    c = client.post("/api/v1/dl/codegen", json={"spec": spec})
    assert c.status_code == 200, c.text
    assert "keras" in c.json()["code"]


def test_architecture_catalog(client):
    r = client.get("/api/v1/dl/architectures")
    assert r.status_code == 200
    assert len(r.json()) >= 5
