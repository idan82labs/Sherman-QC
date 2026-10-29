"""Minimal authenticated server for sharing the Bend Label Studio."""

from __future__ import annotations

import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.api.services import bend_labeling_service


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(os.environ.get(
    "BEND_LABELING_FRONTEND_DIR",
    str(PROJECT_ROOT / "frontend" / "dist-labeling"),
)).resolve()
USERNAME = os.environ.get("BEND_LABELING_USERNAME", "sherman-labeler").strip()
PASSWORD = os.environ.get("BEND_LABELING_PASSWORD", "")
SESSION_TOKEN = secrets.token_urlsafe(48)
MAX_ATTEMPTS = 10
ATTEMPT_WINDOW_SECONDS = 10 * 60
_attempts: Dict[str, Deque[float]] = defaultdict(deque)


if not PASSWORD or len(PASSWORD) < 14:
    raise RuntimeError("BEND_LABELING_PASSWORD must contain at least 14 characters")
if not FRONTEND_DIR.joinpath("index.html").is_file():
    raise RuntimeError("Build the labeling frontend into frontend/dist-labeling before starting")


class LoginRequest(BaseModel):
    username: str
    password: str


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip", "").strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def _enforce_login_rate_limit(request: Request) -> None:
    now = time.monotonic()
    attempts = _attempts[_client_key(request)]
    while attempts and now - attempts[0] > ATTEMPT_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many sign-in attempts. Try again later.")
    attempts.append(now)


def require_labeler(authorization: Optional[str] = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, SESSION_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or expired token")


app = FastAPI(
    title="Sherman Bend Label Studio",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; worker-src 'self' blob:; frame-ancestors 'none'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "bend-labeling"}


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request):
    _enforce_login_rate_limit(request)
    valid_username = hmac.compare_digest(payload.username, USERNAME)
    valid_password = hmac.compare_digest(payload.password, PASSWORD)
    if not valid_username or not valid_password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    _attempts.pop(_client_key(request), None)
    now = "2026-07-26T00:00:00+00:00"
    return {
        "access_token": SESSION_TOKEN,
        "token_type": "bearer",
        "user": {
            "id": 1,
            "username": USERNAME,
            "email": "labeling@sherman.local",
            "role": "operator",
            "is_active": True,
            "created_at": now,
            "last_login": now,
        },
    }


@app.get("/api/auth/me", dependencies=[Depends(require_labeler)])
def me():
    return {
        "id": 1,
        "username": USERNAME,
        "email": "labeling@sherman.local",
        "role": "operator",
        "is_active": True,
        "created_at": "2026-07-26T00:00:00+00:00",
        "last_login": None,
    }


@app.get("/api/training/bend-labeling/assets", dependencies=[Depends(require_labeler)])
def list_assets(
    q: Optional[str] = None,
    source: Optional[str] = None,
    asset_kind: Optional[str] = None,
    label_status: Optional[str] = None,
):
    return bend_labeling_service.list_assets(q, source, asset_kind, label_status)


@app.get("/api/training/bend-labeling/assets/{asset_id}/file", dependencies=[Depends(require_labeler)])
def get_asset_file(asset_id: str):
    return bend_labeling_service.get_asset_file(asset_id)


@app.get("/api/training/bend-labeling/assets/{asset_id}/labels", dependencies=[Depends(require_labeler)])
def get_asset_labels(asset_id: str):
    return bend_labeling_service.get_asset_labels(asset_id)


@app.put("/api/training/bend-labeling/assets/{asset_id}/labels", dependencies=[Depends(require_labeler)])
def put_asset_labels(asset_id: str, payload: Dict[str, Any]):
    return bend_labeling_service.put_asset_labels(asset_id, payload)


app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="labeling-assets")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/bend-labeling", status_code=307)


@app.get("/{full_path:path}", include_in_schema=False)
def frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    requested = (FRONTEND_DIR / full_path).resolve()
    try:
        requested.relative_to(FRONTEND_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    if requested.is_file():
        return FileResponse(requested)
    return FileResponse(FRONTEND_DIR / "index.html")
