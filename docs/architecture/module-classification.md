# Module Classification

This document classifies overlapping modules during the enterprise reordering so deletions are explicit and reversible.

## Backend

### Canonical (target)
- `apps/api/main.py`
- `apps/api/routes/*`
- `apps/api/services/*`
- `domains/bend/*`
- `domains/reporting/*`
- `contracts/ui/*`
- `infrastructure/*`

### Compatibility shims
- `backend/server.py`
  - bootstrap only
- `backend/_server_legacy_impl.py`
  - legacy endpoint implementation host during strangler migration
- service wrappers that still delegate into legacy runtime while extraction is incomplete

### Pending classification
- legacy bend detection/report assembly helpers that still live under `backend/*`
- geometry helper overlap between old backend modules and `infrastructure/geometry/*`

## Frontend

### Canonical (target)
- `frontend/react/src/features/*`
- `frontend/react/src/shared/api/*`
- `frontend/react/src/shared/3d/*`
- `frontend/react/src/shared/types/*`

### Compatibility shims
- `frontend/react/src/pages/*`
  - route-preserving wrappers while feature pages stabilize
- `frontend/react/src/components/ThreeViewer/index.tsx`
  - compatibility wrapper to shared 3D host
- feature-local service/type/component re-export shims used to preserve moved page behavior during migration

### Pending classification
- large shared viewer internals still inside `frontend/react/src/shared/3d/ThreeViewer.tsx`
- any duplicate page-side data shaping that still exists outside feature mappers

## Research/runtime boundary

### Canonical research entrypoints
- `research/*`
- `scripts/evaluate_bend_corpus.py`
- `scripts/train_bend_unary_models.py`
- `scripts/annotate_bend_unary_predictions.py`

### Compatibility shims
- runtime code that still imports research-model behavior through old backend modules rather than explicit research/domain boundaries
