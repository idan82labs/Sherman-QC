from __future__ import annotations

from fastapi import APIRouter, Depends
from apps.api.services import reports_service
from apps.api.services.bend_inspection_service import require_bend_job_owner_if_applicable

router = APIRouter(tags=["Reports"], dependencies=[Depends(require_bend_job_owner_if_applicable)])

router.add_api_route("/api/download/{job_id}/pdf", reports_service.download_pdf, methods=['GET'])
router.add_api_route("/api/download/{job_id}/json", reports_service.download_json, methods=['GET'])
router.add_api_route("/api/heatmap/{job_id}/{filename}", reports_service.get_heatmap, methods=['GET'])
router.add_api_route("/api/heatmaps/{job_id}", reports_service.list_heatmaps, methods=['GET'])
router.add_api_route("/api/files/{job_id}/{filename:path}", reports_service.get_uploaded_file, methods=['GET'])
