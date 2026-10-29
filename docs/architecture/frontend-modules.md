# Frontend Modules

## Current direction
The frontend is being reordered around feature modules and shared 3D/runtime infrastructure, while preserving the current pages and behavior.

## Feature layout
- `src/features/bend-inspection/*`
- `src/features/live-scan/*`
- `src/features/qc-jobs/*`
- `src/features/part-catalog/*`
- `src/shared/api/*`
- `src/shared/3d/*`
- `src/shared/types/*`

## Compatibility policy
Old page paths under `src/pages/*` remain temporary wrappers to preserve routing and keep the migration low-risk.

## Bend annotation rule
The generic 3D viewer must not invent bend-domain semantics. Bend Inspection is responsible for mapping backend bend matches into viewer annotations and callout intent.
