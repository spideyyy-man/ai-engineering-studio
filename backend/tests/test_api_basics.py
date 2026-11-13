"""Health checks and the Projects API."""


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_lists_docs(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"


def test_project_crud(client):
    r = client.post("/api/v1/projects", json={"name": "crud", "description": "first"})
    assert r.status_code == 201
    pid = r.json()["id"]

    assert any(p["id"] == pid for p in client.get("/api/v1/projects").json())

    r = client.patch(f"/api/v1/projects/{pid}", json={"name": "renamed"})
    assert r.status_code == 200
    assert r.json()["name"] == "renamed"

    assert client.delete(f"/api/v1/projects/{pid}").status_code == 200
    r = client.get(f"/api/v1/projects/{pid}")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_unhandled_errors_hide_details_outside_development(client):
    """APP_ENV is 'test' here, so a 500 must not leak the exception text."""
    from app.main import app

    @app.get("/__boom")
    def boom():
        raise RuntimeError("secret internal detail")

    r = client.get("/__boom")
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
