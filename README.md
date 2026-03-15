# Sherman QC

AI-assisted quality control for sheet-metal parts using CAD references, 3D scans, bend inspection, and operator-facing reporting.

Sherman QC now includes two connected tracks in one repo:
- a production-oriented inspection product: upload, analyze, live scan, Bend Inspection UI, PDF/reporting, structured bend counts
- a research/regression track: labeled bend corpus, staged partial-scan evaluation, observability-aware bend-state modeling, and benchmark docs

## What This Repo Does

### Production capabilities
- General scan-vs-CAD analysis with alignment, deviation measurement, and reporting
- Bend Inspection workflow for partial and full scans
- 3D CAD bend overlay with per-bend status and deltas
- Structured completed-bend count reporting alongside direct evidence counts
- Live Scan monitoring and job detail workflows
- PDF and JSON report generation

### Research and regression capabilities
- Organized bend corpus under `data/bend_corpus`
- Corpus evaluator and case evaluator for real industrial parts
- Bend observability fields, candidate/null assignment plumbing, and unary-model infrastructure
- Research docs for the current factor-model direction

## Current Direction

The main technical problem being worked in this repo is:
- reliable bend progression inference from partial industrial scans

The current architectural direction is:
- separate physical bend completion, observability, and evidence assignment
- keep operator-facing labels derived from those internals
- use structured count decoding rather than relying only on raw heuristic matches

This is documented in:
- `docs/BEND_STATE_FACTOR_MODEL_ARCHITECTURE.md`
- `docs/BEND_STATE_MODEL_BACKLOG.md`
- `docs/BEND_REGRESSION_TESTING.md`

## Repository Layout

```text
Full Project/
├── backend/                         FastAPI server, QC pipeline, bend runtime
├── frontend/react/                  React + TypeScript product UI
├── data/bend_corpus/                Corpus metadata, labels, specs, drawings references
├── docs/                            Research notes, architecture, regression docs
├── scripts/                         Corpus builders, evaluators, training/annotation scripts
├── tests/                           Portable regression and math tests
├── run.py                           Backend startup helper
└── requirements.txt                 Root Python dependency set
```

Key files and entry points:
- `backend/server.py`: API surface
- `backend/bend_inspection_pipeline.py`: bend runtime and staged inspection logic
- `backend/bend_analysis_worker.py`: bend-job processing and structured count reporting
- `frontend/react/src/pages/BendInspection.tsx`: bend operator workflow
- `frontend/react/src/pages/LiveScan.tsx`: live monitoring flow
- `scripts/evaluate_bend_corpus.py`: corpus benchmark runner
- `scripts/train_bend_unary_models.py`: unary-model training bootstrap

## Quick Start

### Backend

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run.py
```

Backend default URL:
- `http://localhost:8080`

Notes:
- On macOS, `run.py` prefers Python `3.11+` for Open3D stability.
- Additional macOS runtime notes are in `docs/open3d_macos_runtime.md`.

### Frontend

```bash
cd frontend/react
npm ci
npm run dev
```

Frontend default URL:
- `http://localhost:5173`

Production build check:

```bash
cd frontend/react
npm run build
```

## Core Workflows

### 1. General QC analysis
Use the upload/product flow to compare a scan against a CAD reference and inspect:
- alignment
- regional deviation
- dimensions
- AI-assisted summaries
- downloadable reports

### 2. Bend Inspection
Use Bend Inspection for staged or final sheet-metal scans when you need:
- completed bend count
- remaining bend count
- in-spec / warning / out-of-spec bend status
- per-bend deltas
- 3D bend overlay on the original CAD
- structured count vs direct evidence count

### 3. Corpus and regression evaluation
Use the corpus tooling to evaluate runtime changes on real part families:

```bash
python3 scripts/evaluate_bend_corpus.py
```

Useful related scripts:
- `scripts/evaluate_bend_case.py`
- `scripts/build_bend_corpus.py`
- `scripts/enrich_bend_corpus_from_specs.py`
- `scripts/annotate_bend_unary_predictions.py`
- `scripts/train_bend_unary_models.py`

## Recommended Reading Order

If you are entering the repo for product work:
1. `README.md`
2. `backend/server.py`
3. `frontend/react/src/pages/BendInspection.tsx`
4. `frontend/react/src/pages/LiveScan.tsx`

If you are entering the repo for the bend-model/research track:
1. `docs/BEND_REGRESSION_TESTING.md`
2. `docs/BEND_STATE_FACTOR_MODEL_ARCHITECTURE.md`
3. `docs/BEND_STATE_MODEL_BACKLOG.md`
4. `docs/ONE_PAGER_PARTIAL_BEND_STATE_INFERENCE.md`
5. `docs/PROFESSOR_RESPONSE_IMPLEMENTATION_NOTE_20260315.md`

Additional useful docs:
- `docs/BEND_CORPUS_DATA_CARD.md`
- `docs/RESEARCH_POSITIONING_BEND_WORK.md`
- `docs/MATH_PRECISION_RND_PLAN.md`
- `docs/OPEN_RESEARCH_QUESTION_FOR_PROFESSOR.md`
- `docs/PROFESSOR_MEETING_QUESTIONS_20260315.md`

## Testing

### Backend validation
Portable backend/regression suite used in the clean integration merge:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
pytest -q \
  backend/tests/test_bend_report_generator.py \
  backend/tests/test_bend_analysis_worker.py \
  backend/tests/test_bend_unary_model.py \
  backend/tests/test_bend_detector.py \
  backend/tests/test_bend_inspection_pipeline.py \
  tests/test_evaluate_bend_corpus.py \
  tests/test_bend_improvement_loop_math.py \
  tests/test_feature_measurement.py
```

### Frontend validation

```bash
cd frontend/react
npm run build
```

## Data and Artifact Policy

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
- local status logs and scratchpads
- downloaded/generated scan and CAD binaries that are not intended source assets

This is deliberate. The repo tracks reproducible source, not local runtime output.

## Current Benchmark Signal

The current regression work established that structured bend-count decoding materially improves staged partial-scan accuracy over the earlier heuristic-only path.

The exact benchmark history and methodology live in:
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

Read `backend/server.py` for the full current surface.

## Notes

- The repo now reflects the unified merge of product work, bend runtime work, and research/docs.
- The old dirty feature worktree was intentionally cleaned after merge; a backup snapshot of its dirty state was preserved outside the repo before cleanup.
- If you want to continue the bend-model work, start from `main` and the docs listed above rather than reviving the pre-merge feature branch.

## License

MIT License. See `LICENSE`.
