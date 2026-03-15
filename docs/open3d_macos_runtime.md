# Open3D Runtime Notes for macOS (Accuracy-First)

This project runs high-accuracy Open3D ICP in production paths. On this MacBook class,
`Python 3.9 + Open3D 0.18` can segfault during registration, while `Python 3.11 + Open3D 0.19`
is stable for the same datasets.

## Recommended Runtime

- Backend process: Python 3.11
- Analysis worker: Python 3.11
- Open3D: 0.19.x
- Keep Open3D ICP enabled (no algorithm downgrade)

## What is implemented in code

- `backend/server.py` picks analysis worker interpreter with priority:
  1. `ANALYSIS_PYTHON` env
  2. `/opt/homebrew/bin/python3.11`
  3. `/usr/local/bin/python3.11`
  4. current interpreter
- Worker thread env is set for numerical libs (override with `ANALYSIS_WORKER_THREADS`):
  - `OMP_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `NUMEXPR_NUM_THREADS`
  - `VECLIB_MAXIMUM_THREADS`
- `run.py` re-execs itself with Python 3.11 on macOS if launched under older Python.

## Operational notes

- Progress may appear at ~58% for a while during flip-hypothesis ICP (this is expected on dense scans).
- This hold is compute-bound, not a crash, when worker process remains alive.
- For max throughput on high-core systems, set `ANALYSIS_WORKER_THREADS` explicitly (example: `16`).

## References

- Open3D issue: ICP segfault in 0.18 (`registration_icp` crash)
  - https://github.com/isl-org/Open3D/issues/6935
- Open3D issue: known crash behavior around old bindings/runtime interactions
  - https://github.com/isl-org/Open3D/issues/6936
- Open3D docs (current release docs)
  - https://www.open3d.org/docs/release/
