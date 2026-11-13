"""ML studio: algorithm catalog, training and artifact download."""


def test_algorithm_catalog(client):
    r = client.get("/api/v1/ml/algorithms")
    assert r.status_code == 200
    tasks = {a["task"] for a in r.json()}
    assert {"classification", "regression", "clustering"} <= tasks


def test_train_logistic_regression_and_download(client, project_id, dataset_id):
    r = client.post(
        "/api/v1/ml/train",
        json={
            "project_id": project_id,
            "dataset_id": dataset_id,
            "target_col": "species",
            "algorithm": "logistic_regression",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "classification"
    assert body["metrics"]

    model_id = body["model_id"]
    pkl = client.get(f"/api/v1/ml/models/{model_id}/download", params={"fmt": "pkl"})
    assert pkl.status_code == 200
    assert len(pkl.content) > 0


def test_train_rejects_dataset_from_another_project(client, dataset_id):
    other = client.post("/api/v1/projects", json={"name": "other"}).json()["id"]
    r = client.post(
        "/api/v1/ml/train",
        json={"project_id": other, "dataset_id": dataset_id, "target_col": "species",
              "algorithm": "logistic_regression"},
    )
    assert r.status_code == 422
