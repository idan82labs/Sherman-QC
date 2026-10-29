# Troubleshooting

## Frontend build fails after module moves
Check feature-local shims under `frontend/react/src/features/*` before editing migrated pages directly.

## FastAPI import issues during migration
Use `apps/api/main.py` as the assembled app entrypoint and `backend/_server_legacy_impl.py` for legacy implementations still being strangled.

## Open3D/macOS runtime issues
See `/Users/idant/82Labs/Sherman QC/Full Project/docs/open3d_macos_runtime.md`.
