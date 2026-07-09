from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.routes.manual_assistant import router as manual_assistant_router
from apps.api.services import manual_assistant_service
from infrastructure.rag.manual_store import DATA_DIR, INDEX_PATH


BASE_DIR = Path(__file__).resolve().parents[2]


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


def _frontend_dir() -> Path:
    return Path(os.environ.get("SHERMAN_CHAT_FRONTEND_DIR", BASE_DIR / "frontend"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="ShermanChat",
        description="Manual-grounded AI chat for Sherman QC operational and software docs.",
        version="1.0.0",
    )

    origins = _allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With", "X-Sherman-Chat-Token"],
    )

    app.include_router(manual_assistant_router)

    @app.get("/api/health", include_in_schema=False)
    def health():
        return {
            "status": "healthy",
            "service": "sherman-chat",
            "provider": manual_assistant_service.active_provider(),
            "configured_provider": os.environ.get("SHERMAN_CHAT_PROVIDER", "mock"),
            "retrieval_backend": os.environ.get("SHERMAN_RETRIEVAL_BACKEND", "local"),
            "manual_data_dir": str(DATA_DIR),
            "index_ready": INDEX_PATH.exists(),
        }

    frontend_dir = _frontend_dir()
    if frontend_dir.exists():
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend_app(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")

            requested = (frontend_dir / full_path).resolve()
            try:
                requested.relative_to(frontend_dir.resolve())
            except ValueError:
                raise HTTPException(status_code=404, detail="Not found") from None

            if requested.is_file():
                return FileResponse(requested)

            index_path = frontend_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            raise HTTPException(status_code=404, detail="Frontend not found")

    return app


app = create_app()
