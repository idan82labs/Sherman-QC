# Contracts

The migration freezes these contracts first and moves internal code behind them:
- `AnalysisJobResult`
- `BendInspectionResult`
- `BendMatchResult`
- `StructuredCountResult`
- `ReportSummaryResult`

## Why
- frontend pages should consume stable result shapes
- worker/report/runtime changes should be isolated behind mappers
- domain extraction should not break product behavior during migration

## Current implementation
Initial UI contracts live in `contracts/ui/` and are mapped from runtime dictionaries in `domains/bend/mappers/ui_contracts.py`.
