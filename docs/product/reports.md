# Reports

Sherman QC produces JSON and PDF/operator-facing summaries for both general QC and bend inspection.

## Migration direction
Reporting is being unified so domain-specific payloads feed one report model instead of ad hoc per-endpoint shape assembly.

Current related modules:
- `domains/reporting/services/bend_report_service.py`
- `backend/bend_report_generator.py`
- `apps/api/routes/reports.py`
