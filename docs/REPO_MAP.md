# Repository map: actual execution paths

Source review: 2026-09-07. This is a navigation aid, not a runtime qualification.
Read [current status](CURRENT_STATUS.md) before using research commands or data.

## Bend Inspection today

```text
frontend/react/src/features/bend-inspection/pages/BendInspectionPage.tsx
  -> backend/server.py -> apps/api/main.py -> apps/api/routes/bend_inspection.py
  -> apps/api/services/bend_inspection_service.py
  -> legacy in-memory jobs / BackgroundTasks / subprocess launcher
  -> backend/bend_analysis_worker.py
  -> domains/bend/services runtime shim -> backend/bend_inspection_pipeline.py
  -> report/count enrichment + heatmap/artifacts -> API polling -> feature UI
```

`backend/_server_legacy_impl.py` still hosts shared state and launch behavior.
`domains/bend/application/analyze_bend_job.py` and the typed UI mapper exist, but
the reviewed active worker calls the runtime directly. `apps/worker/` is not yet
a durable queue implementation. Treat the enterprise diagrams as a migration
target until a call-path test proves the new boundary is in use.

## Where to start by task

| Task | Source seam |
| --- | --- |
| API/job lifecycle | `apps/api/routes/bend_inspection.py`, `apps/api/services/bend_inspection_service.py`, legacy launcher |
| Actual bend geometry | `backend/bend_analysis_worker.py`, `backend/bend_inspection_pipeline.py`, `backend/bend_detector.py` |
| General scan/CAD QC | `backend/qc_engine.py` and `infrastructure/geometry/` |
| Typed outputs and semantics | `contracts/ui/bend_results.py`, `domains/bend/mappers/ui_contracts.py`, `domains/bend/services/runtime_semantics.py` |
| Counts versus observations | `domains/bend/services/structured_count_reporting_service.py`, BendInspection feature page |
| Bend annotations/viewer | feature `model/annotationMapper.ts`, `frontend/react/src/shared/3d/ThreeViewer.tsx` |
| Qualification policy | `docs/release/ENTERPRISE_BEND_PROMOTION_GATES.md`; implementation `apps/api/services/bend_readiness_service.py` |
| Research mechanisms | Named `backend/f1_*.py`, `scripts/`, and their explicit test files; start with the dated status, not latest filename |
| Research evidence | Released narrative `docs/research/` reports linked from the roadmap; do not infer access rights from location |

## Known seams that matter in production

- Reviewed bend routes lack server-side auth dependencies; UI login is insufficient.
- Jobs are process-local; admission, cancellation, restart recovery and atomic
  artifact publication are not an end-to-end durable execution contract.
- Readiness checks validate declaration fields/hash syntax, not the referenced
  evidence against the exact deployed candidate. A `PASS` field is not proof.
- Unknown/unobserved semantics and structured-count UI totals can overstate
  completion. Acquisition scope is not carried end to end by the upload contract.
- A mutable latest-model pointer can influence reported counts. A release must pin
  the whole candidate/config/model bundle and propagate failures explicitly.

These are source-review findings, not changes made by this documentation update.
Implementation priorities and acceptance checks are in the
[production roadmap](research/2026-09-07_production_generalization_roadmap.md).

## Search and test narrowly

Pass explicit paths to `rg`, e.g.:

```bash
rg -n 'UNKNOWN|UNOBSERVED|FORMED' domains/bend/services/runtime_semantics.py
rg -n 'map_bend_report_to_contract|analyze_bend_job' apps domains backend/bend_analysis_worker.py
```

Do not search raw datasets, artifact directories or authorization records for
context. A server import can create files and load research status; importing it
is not a harmless source inspection. Command tiers and an isolated synthetic
smoke test are documented in [local development](operations/local-dev.md).
