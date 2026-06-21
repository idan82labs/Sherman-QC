from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.api.legacy_context import legacy
from apps.api.routes import ALL_ROUTERS


def create_app() -> FastAPI:
    app = FastAPI(
        title=legacy.app.title,
        description=legacy.app.description,
        version=legacy.app.version,
        contact=legacy.app.contact,
        license_info=legacy.app.license_info,
        openapi_tags=legacy.app.openapi_tags,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=legacy.ALLOWED_ORIGINS,
        allow_credentials="*" not in legacy.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With", "X-Sherman-Chat-Token"],
    )

    app.add_exception_handler(HTTPException, legacy.http_exception_handler)
    app.add_exception_handler(ValidationError, legacy.validation_exception_handler)

    for router in ALL_ROUTERS:
        app.include_router(router)

    raw_frontend_dir = legacy.BASE_DIR / "frontend"
    frontend_dist_dir = raw_frontend_dir / "dist"
    frontend_dir = frontend_dist_dir if (frontend_dist_dir / "index.html").exists() else raw_frontend_dir
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
