# Sherman QC

Sherman QC is a sheet-metal quality-control system that compares CAD/MBD intent against 3D scan evidence and produces operator-facing inspection decisions, bend-level reporting, and regression metrics.

Published `main` now contains the correspondence-gated conformity baseline:
- bend-specific claims are gated by observability, correspondence readiness, and datum trust
- part release uses `AUTO_PASS | HOLD | AUTO_FAIL`
- ambiguous or unmeasurable cases stay explicit instead of being silently coerced into pass/fail
- corpus evaluation is split into correspondence, completion, metrology, position, and abstention scoreboards

## What The Repo Contains

### Product runtime
- FastAPI backend for CAD/scan analysis and bend inspection
- React operator UI for bend review, live scan monitoring, and job detail workflows
- PDF/JSON reporting and structured release summaries
- CAD-local and global bend matching, alignment, and per-bend evidence surfaces

### Regression and research
- real bend corpus under `data/bend_corpus`
- corpus and single-case evaluators
- docs covering the conformity model, bend-state semantics, and regression methodology
- test coverage for runtime semantics, evaluator behavior, and frontend contract changes

## Current Mainline Direction

The current `main` branch is centered on Bayesian-style conformity assessment with explicit abstention in product form:
- separate physical bend formation from measurement observability
- treat nominal-feature correspondence as first-class state
- only issue bend-specific metrology/position claims when the claim is decision-ready
- keep unconditional geometry metrics diagnostic-only rather than release-driving

In practical repo terms, the important areas are:
- `backend/feature_detection/bend_detector.py`
- `backend/bend_inspection_pipeline.py`
- `domains/bend/services/runtime_semantics.py`
- `scripts/evaluate_bend_corpus.py`
- `frontend/react/src/pages/BendInspection.tsx`

Recommended architecture reading:
- `docs/BEND_STATE_FACTOR_MODEL_ARCHITECTURE.md`
- `docs/BEND_STATE_MODEL_BACKLOG.md`
- `docs/BEND_REGRESSION_TESTING.md`
- `docs/PROFESSOR_FOLLOW_UP_MEMO_20260417.md`

## Repository Layout

```text
.
├── backend/                         FastAPI server, QC pipeline, PDF/report generation
├── domains/                         Shared domain semantics and service helpers
├── frontend/react/                  React + TypeScript operator UI
├── data/bend_corpus/                Corpus metadata, labels, ready regression definitions
├── docs/                            Architecture notes, research memos, implementation notes
├── scripts/                         Evaluation, corpus maintenance, training, utilities
├── tests/                           Portable regression and evaluator tests
├── run.py                           Backend startup helper
└── requirements.txt                 Python dependency set
```

Key entry points:
- `backend/server.py`
- `backend/bend_inspection_pipeline.py`
- `backend/feature_detection/bend_detector.py`
- `frontend/react/src/pages/BendInspection.tsx`
- `frontend/react/src/pages/JobDetail.tsx`
- `scripts/evaluate_bend_case.py`
- `scripts/evaluate_bend_corpus.py`

## Quick Start

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run.py
```

Default backend URL:
- `http://localhost:8080`

Runtime notes:
- macOS should use Python `3.11+` for Open3D stability
- additional runtime notes are in `docs/open3d_macos_runtime.md`

### Frontend

```bash
cd frontend/react
npm ci
npm run dev
```

Default frontend URL:
- `http://localhost:5173`

Canonical frontend baseline:
- Node `20.x`
- `./scripts/check_frontend_toolchain.sh` is the supported environment preflight

### Core evaluators

```bash
python3 scripts/evaluate_bend_case.py --help
python3 scripts/evaluate_bend_corpus.py --help
```

## Core Workflows

### 1. General QC analysis
Use the product flow to compare scan geometry against a CAD reference and inspect:
- alignment quality
- deviation summaries
- dimensions and report outputs
- AI-assisted analysis summaries

### 2. Bend Inspection
Use Bend Inspection when you need:
- bend completion progress
- bend-specific claim gating
- in-tolerance / out-of-tolerance metrology
- on-position / unknown-position state
- release blockers and hold reasons
- 3D CAD overlay and operator evidence panels

### 3. Corpus regression
Use the corpus tooling to test runtime changes on real parts before shipping:

```bash
python3 scripts/evaluate_bend_corpus.py
```

Related scripts:
- `scripts/evaluate_bend_case.py`
- `scripts/build_bend_corpus.py`
- `scripts/enrich_bend_corpus_from_specs.py`
- `scripts/annotate_bend_unary_predictions.py`
- `scripts/train_bend_unary_models.py`

## Recommended Reading Order

If you are entering for product/runtime work:
1. `README.md`
2. `backend/server.py`
3. `backend/bend_inspection_pipeline.py`
4. `frontend/react/src/pages/BendInspection.tsx`
5. `frontend/react/src/pages/JobDetail.tsx`

If you are entering for the bend-model/evaluator track:
1. `docs/BEND_REGRESSION_TESTING.md`
2. `docs/BEND_STATE_FACTOR_MODEL_ARCHITECTURE.md`
3. `docs/BEND_STATE_MODEL_BACKLOG.md`
4. `docs/PROFESSOR_FOLLOW_UP_MEMO_20260417.md`
5. `scripts/evaluate_bend_corpus.py`

Additional useful docs:
- `docs/BEND_CORPUS_DATA_CARD.md`
- `docs/RESEARCH_POSITIONING_BEND_WORK.md`
- `docs/MATH_PRECISION_RND_PLAN.md`

## Validation

### Mainline backend and evaluator suite

This is the concise mainline suite used for the current published baseline:

```bash
PYTHONPATH=\"$(pwd):$(pwd)/backend\" pytest -q \
  backend/tests/test_bend_inspection_pipeline.py \
  backend/tests/test_bend_detector.py \
  tests/test_evaluate_bend_corpus.py \
  tests/test_evaluate_bend_case.py
```

### Frontend validation

```bash
./scripts/check_frontend_toolchain.sh
```

Recommended frontend build checks:

```bash
cd frontend/react
npm run typecheck
npm run build
```

## Data And Artifact Policy

Tracked in git:
- runtime/backend code
- frontend source
- tests
- deterministic scripts
- corpus metadata and labels
- docs

Not tracked in git:
- `frontend/dist/`
- `output/`
- `data/bend_corpus/evaluation/`
- local FAISS indexes
- ad hoc logs, caches, and scratch output
- downloaded/generated CAD or scan binaries that are not intended source assets

The repo tracks reproducible source and stable fixtures, not local runtime byproducts.

## Current Benchmark Posture

The current published baseline establishes:
- correspondence-gated bend claims
- explicit abstention instead of silent overclaiming
- success-aware local assignment conflict resolution
- STL sanity handling, including CAD-duplicate scan detection

For current methodology and benchmark framing:
- `docs/BEND_REGRESSION_TESTING.md`
- `data/bend_corpus/ready_regression.json`

## API Surface

Representative endpoints:
- `POST /api/analyze`
- `GET /api/jobs/{id}`
- `GET /api/result/{id}`
- `POST /api/bend-inspection/analyze`
- `GET /api/bend-inspection/{id}`
- `GET /api/bend-inspection/{id}/table`
- `GET /api/parts`

See `backend/server.py` for the current full surface.

## Notes

- `main` is now the authoritative unified baseline; do not resurrect the old pre-merge feature branches as working truth.
- The dirty enterprise reorder worktree was intentionally kept separate and should not be merged wholesale.
- The next meaningful work is ambiguity/risk calibration on top of the published correspondence-gated baseline, not merge mechanics.

## License

MIT License. See `LICENSE`.
