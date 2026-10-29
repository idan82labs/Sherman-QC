# Backend Modules

## Entry layer
- `backend/server.py`: bootstrap only
- `backend/_server_legacy_impl.py`: temporary legacy implementation host during migration
- `apps/api/main.py`: FastAPI assembly
- `apps/api/routes/*`: route groups
- `apps/api/services/*`: application-facing service shims

## Domain layer
- `domains/bend/*`: bend runtime, matching, contracts, report mapping
- `domains/reporting/*`: report assembly/export services
- `domains/qc/*`, `domains/dimensions/*`, `domains/catalog/*`, `domains/auth/*`: target homes for extracted logic

## Infrastructure layer
- `infrastructure/geometry/*`: CAD/scan loading and geometry helpers
- `infrastructure/persistence/*`: artifact/file persistence adapters
- `infrastructure/queue/*`: worker transport
- `infrastructure/ai/*`: model/provider adapters
- `infrastructure/telemetry/*`: logging/metrics/tracing adapters

## Contracts
- `contracts/api/*`: public request/response schemas
- `contracts/jobs/*`: worker payload/result schemas
- `contracts/ui/*`: frontend-facing contracts

## Research
- `research/corpus/*`, `research/evaluation/*`, `research/models/*`
- runtime code may use research models, but the boundary must stay explicit
