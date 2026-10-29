"""FastAPI bootstrap entrypoint.

The API surface is now assembled in `apps.api.main`.
Legacy endpoint implementations remain available in `backend/_server_legacy_impl.py`
during the strangler migration so behavior stays stable while modules move.
"""

from __future__ import annotations

from apps.api.main import app, create_app

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
