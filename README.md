# Sherman QC

AI-assisted quality control for sheet-metal parts using CAD references, 3D scans, bend inspection, live scanning, and operator-facing reporting.

The repo is now being reordered toward an enterprise architecture without rewriting product behavior. The active migration pattern is a strangler refactor: preserve current flows, freeze contracts, introduce new module boundaries, then retire old entrypoints only after the new seams are validated.

For agent work, start with [AGENTS.md](AGENTS.md), the dated
[current checkpoint](docs/CURRENT_STATUS.md), and the
[actual runtime map](docs/REPO_MAP.md). The
[production generalization roadmap](docs/research/2026-09-07_production_generalization_roadmap.md)
separates research evidence from production requirements. Research remains
unqualified for autonomous part acceptance/rejection.

## Current Capabilities

### Product/runtime
- general scan-vs-CAD analysis
- Bend Inspection for partial and full scans
- 3D CAD/scan overlay with bend annotations and callouts
- structured bend count reporting alongside direct evidence counts
- live scan/session workflows
- JSON/PDF reporting

### Research/regression
- organized bend corpus under `data/bend_corpus`
- corpus and case evaluators
- unary-model and structured count experimentation
- benchmark and methodology docs

## Enterprise Reordering Direction

The codebase is being reordered around these top-level concerns:

```text
apps/
  api/          FastAPI assembly, route groups, dependency wiring
  worker/       background job entrypoints

domains/
  bend/         bend inspection and structured count logic
  reporting/    report assembly/export
  qc/           general QC analysis
  dimensions/   specs and dimensions
  catalog/      part/live-session catalog behavior
  auth/         auth domain

infrastructure/
  geometry/     CAD/scan loading, transforms, clipping, alignment helpers
  persistence/  artifact/file persistence
  ai/           AI provider adapters
  queue/        worker transport
  telemetry/    logging/metrics/tracing

contracts/
  api/
  jobs/
  ui/

research/
  corpus/
  evaluation/
  models/
  docs/
```

See:
- `docs/ENTERPRISE_REARCHITECTURE_PLAN.md`
- `docs/architecture/system-overview.md`
- `docs/architecture/runtime-flows.md`
- `docs/architecture/contracts.md`
- `docs/architecture/module-classification.md`

## Repository Layout

```text
FullProject/
├── apps/                          New application entrypoints and route groups
├── backend/                       Legacy runtime modules retained during strangler migration
├── contracts/                     Stable API/job/UI contracts
├── domains/                       Extracted domain logic
├── frontend/react/                React product UI
├── infrastructure/                Shared technical adapters and geometry helpers
├── research/                      Research-model, corpus, and evaluator homes
├── scripts/                       Deterministic corpus/build/eval/train scripts
├── data/bend_corpus/              Corpus metadata and labels
├── docs/                          Architecture, product, research, operations docs
├── tests/                         Portable regression and math tests
└── requirements.txt
```

## Primary Entry Points

### Backend
- bootstrap: `backend/server.py`
- assembled app: `apps/api/main.py`
- legacy implementation host during migration: `backend/_server_legacy_impl.py`
- active bend subprocess: `backend/bend_analysis_worker.py`
- bend application wrapper (migration seam, not yet the active worker path): `domains/bend/application/analyze_bend_job.py`

### Frontend
- Bend Inspection feature page: `frontend/react/src/features/bend-inspection/pages/BendInspectionPage.tsx`
- bend annotation mapping seam: `frontend/react/src/features/bend-inspection/model/annotationMapper.ts`
- shared 3D host: `frontend/react/src/shared/3d/ThreeViewer.tsx`
- legacy page routes under `frontend/react/src/pages/*` remain compatibility wrappers where needed

### Research
- corpus evaluator: `scripts/evaluate_bend_corpus.py`
- unary model training: `scripts/train_bend_unary_models.py`
- annotation pass: `scripts/annotate_bend_unary_predictions.py`

## Quick Start

Run from this Git root (`FullProject/`). These are runtime/setup commands, not
permission to open research inputs. See [local development](docs/operations/local-dev.md)
for source-only checks and environment isolation; reuse an existing environment.

### Backend

```bash
# Only when creating a new application environment:
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 .venv/bin/python run.py --port 9000
```

### Frontend

```bash
npm --prefix frontend/react ci
npm --prefix frontend/react run dev
```

### Validation

Choose the tier relevant to the change. Do not run broad tests or benchmarks under
a source-only gate; inspect imports/fixtures first. Frontend commands preserve cwd:

```bash
npm --prefix frontend/react run lint
npm --prefix frontend/react run build
```

The isolated synthetic Python smoke command is in
[local development](docs/operations/local-dev.md#source-only-validation).

## Documentation Map

### Architecture
- `docs/architecture/system-overview.md`
- `docs/architecture/runtime-flows.md`
- `docs/architecture/backend-modules.md`
- `docs/architecture/frontend-modules.md`
- `docs/architecture/contracts.md`

### Product
- `docs/product/bend-inspection.md`
- `docs/product/live-scan.md`
- `docs/product/reports.md`

### Research
- `docs/research/corpus-data-card.md`
- `docs/research/regression-framework.md`
- `docs/research/bend-state-model.md`
- `docs/research/benchmark-history.md`

### Operations
- `docs/operations/local-dev.md`
- `docs/operations/deployment.md`
- `docs/operations/troubleshooting.md`

### ADRs
- `docs/adr/ADR-001-domain-boundaries.md`
- `docs/adr/ADR-002-canonical-bend-detector.md`
- `docs/adr/ADR-003-report-contract.md`
- `docs/adr/ADR-004-research-runtime-separation.md`

## Current Technical Direction

The main hard problem is still reliable bend progression inference from partial industrial scans. The current runtime/research direction is:
- separate physical bend completion, observability, and assignment
- keep operator-facing labels derived from those internal states
- use structured count decoding instead of raw heuristic count alone

Primary references:
- `docs/BEND_STATE_FACTOR_MODEL_ARCHITECTURE.md`
- `docs/BEND_STATE_MODEL_BACKLOG.md`
- `docs/PROFESSOR_RESPONSE_IMPLEMENTATION_NOTE_20260315.md`
- `docs/BEND_REGRESSION_TESTING.md`

## Artifact Policy

Tracked in git:
- source code
- tests
- deterministic scripts
- corpus metadata and labels
- docs

Not tracked in git:
- `frontend/dist/`
- `output/`
- `data/bend_corpus/evaluation/`
- local FAISS indexes
- runtime logs/scratchpads
- downloaded/generated binaries that are not intended source assets

## Notes

- This repo is in an active enterprise reordering phase; maintain behavior first, simplify structure second, and delete old seams last.
- If you change Bend Inspection or the 3D annotation path, keep the bend-domain assembly in the feature layer rather than the generic viewer.
