from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import reports_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Reports"])

bind(router, "/api/download/{job_id}/pdf", reports_service.download_pdf, ['GET'])
bind(router, "/api/download/{job_id}/json", reports_service.download_json, ['GET'])
bind(router, "/api/heatmap/{job_id}/{filename}", reports_service.get_heatmap, ['GET'])
bind(router, "/api/heatmaps/{job_id}", reports_service.list_heatmaps, ['GET'])
bind(router, "/api/files/{job_id}/{filename:path}", reports_service.get_uploaded_file, ['GET'])
