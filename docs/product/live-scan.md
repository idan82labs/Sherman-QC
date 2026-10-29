# Live Scan

Live Scan handles in-process scanning and operator monitoring for recognition, coverage, and downstream QC transitions.

## Current migration shape
- legacy page behavior is preserved
- feature page path: `frontend/react/src/features/live-scan/pages/LiveScanPage.tsx`
- API route group: `apps/api/routes/live_scan.py`
- service shim: `apps/api/services/live_scan_service.py`
