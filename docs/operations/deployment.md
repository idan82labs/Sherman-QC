# Deployment

Deployment is still tied to the current product runtime. During the reordering phase, the main requirement is that `backend/server.py` remains a stable bootstrap entrypoint and the frontend production build remains reproducible from source.

Generated runtime artifacts are intentionally not tracked in git.
