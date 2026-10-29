from __future__ import annotations

from fastapi import APIRouter, Depends
from apps.api.services import jobs_service
from apps.api.services.bend_inspection_service import require_bend_job_owner_if_applicable
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Jobs"])


def bind_job(path: str, endpoint, methods: list[str]):
    router.add_api_route(path, endpoint, methods=methods,
                         dependencies=[Depends(require_bend_job_owner_if_applicable)])


bind_job("/api/progress/{job_id}", jobs_service.get_progress, ['GET'])
bind_job("/api/progress/{job_id}/stream", jobs_service.stream_progress, ['GET'])
bind_job("/api/result/{job_id}", jobs_service.get_result, ['GET'])
bind_job("/api/deviations/{job_id}", jobs_service.get_deviations, ['GET'])
bind_job("/api/aligned-scan/{job_id}.ply", jobs_service.get_aligned_scan, ['GET'])
bind_job("/api/reference-mesh/{job_id}.ply", jobs_service.get_reference_mesh, ['GET'])
bind(router, "/api/jobs", jobs_service.list_jobs, ['GET'])
bind_job("/api/jobs/{job_id}", jobs_service.get_job_details, ['GET'])
bind_job("/api/jobs/{job_id}", jobs_service.delete_job, ['DELETE'])
bind_job("/api/jobs/{job_id}/dimensions", jobs_service.get_job_dimensions, ['GET'])
bind_job("/api/jobs/{job_id}/bends", jobs_service.get_job_bends, ['GET'])
bind_job("/api/jobs/{job_id}/correlations", jobs_service.get_job_correlations, ['GET'])
bind_job("/api/jobs/{job_id}/enhanced-analysis", jobs_service.get_enhanced_analysis, ['GET'])
bind(router, "/api/stats", jobs_service.get_stats, ['GET'])
