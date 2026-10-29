from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _load_app(monkeypatch, tmp_path: Path):
    frontend = tmp_path / "dist-labeling"
    frontend.mkdir(parents=True, exist_ok=True)
    (frontend / "assets").mkdir(exist_ok=True)
    (frontend / "index.html").write_text("<html>labeler</html>", encoding="utf-8")
    monkeypatch.setenv("BEND_LABELING_USERNAME", "test-labeler")
    monkeypatch.setenv("BEND_LABELING_PASSWORD", "correct-horse-battery")
    monkeypatch.setenv("BEND_LABELING_FRONTEND_DIR", str(frontend))
    sys.modules.pop("apps.labeling.main", None)
    return importlib.import_module("apps.labeling.main").app


def test_standalone_auth_and_surface_restriction(monkeypatch, tmp_path: Path) -> None:
    app = _load_app(monkeypatch, tmp_path)
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/training/bend-labeling/assets").status_code == 401
    login = client.post("/api/auth/login", json={
        "username": "test-labeler",
        "password": "correct-horse-battery",
    })
    assert login.status_code == 200
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200
    assert client.get("/api/training/bend-labeling/assets", headers=headers).status_code == 200
    assert client.get("/api/jobs", headers=headers).status_code == 404


def test_standalone_rejects_bad_password(monkeypatch, tmp_path: Path) -> None:
    app = _load_app(monkeypatch, tmp_path)
    client = TestClient(app)
    response = client.post("/api/auth/login", json={
        "username": "test-labeler",
        "password": "wrong-password",
    })
    assert response.status_code == 401
