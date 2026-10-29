# Bend Inspection

Bend Inspection compares a partial or full scan against finished CAD and reports:
- completed and remaining bends
- structured count vs direct evidence count
- per-bend angle/radius/line-center deltas
- 3D CAD/scan overlay with bend annotations and callouts
- exportable report artifacts

## Frontend ownership
- feature page: `frontend/react/src/features/bend-inspection/pages/BendInspectionPage.tsx`
- bend annotation mapping: `frontend/react/src/features/bend-inspection/model/annotationMapper.ts`
- shared viewer host: `frontend/react/src/shared/3d/ThreeViewer.tsx`

## Backend ownership
- application wrapper: `domains/bend/application/analyze_bend_job.py`
- runtime service: `domains/bend/services/runtime_service.py`
- report service: `domains/reporting/services/bend_report_service.py`
