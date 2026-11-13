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


def test_train_binary_text_labels_with_cv(client, project_id):
    """Binary targets like yes/no used to crash: sklearn expected pos_label=1."""
    from pathlib import Path

    sample = Path(__file__).resolve().parents[2] / "samples" / "customer_churn.csv"
    up = client.post(
        "/api/v1/data/upload",
        data={"project_id": project_id},
        files={"file": ("customer_churn.csv", sample.read_bytes(), "text/csv")},
    )
    assert up.status_code == 201, up.text

    base = client.get(f"/api/v1/data/datasets/{up.json()['id']}/baseline", params={"target_col": "churned"})
    assert base.status_code == 200, base.text

    r = client.post(
        "/api/v1/ml/train",
        json={
            "project_id": project_id,
            "dataset_id": up.json()["id"],
            "target_col": "churned",
            "algorithm": "random_forest_clf",
            "cv_strategy": "stratified_kfold",
            "cv_folds": 3,
        },
    )
    assert r.status_code == 200, r.text
    metrics = r.json()["metrics"]
    assert metrics["average"] == "binary"
    assert metrics["positive_label"] == "yes"
    assert 0.0 <= metrics["f1"] <= 1.0


def test_train_rejects_dataset_from_another_project(client, dataset_id):
    other = client.post("/api/v1/projects", json={"name": "other"}).json()["id"]
    r = client.post(
        "/api/v1/ml/train",
        json={"project_id": other, "dataset_id": dataset_id, "target_col": "species",
              "algorithm": "logistic_regression"},
    )
    assert r.status_code == 422
