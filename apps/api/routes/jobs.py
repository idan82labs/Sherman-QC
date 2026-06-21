from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import jobs_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Jobs"])

bind(router, "/api/progress/{job_id}", jobs_service.get_progress, ['GET'])
bind(router, "/api/progress/{job_id}/stream", jobs_service.stream_progress, ['GET'])
bind(router, "/api/result/{job_id}", jobs_service.get_result, ['GET'])
bind(router, "/api/deviations/{job_id}", jobs_service.get_deviations, ['GET'])
bind(router, "/api/aligned-scan/{job_id}.ply", jobs_service.get_aligned_scan, ['GET'])
bind(router, "/api/reference-mesh/{job_id}.ply", jobs_service.get_reference_mesh, ['GET'])
bind(router, "/api/jobs", jobs_service.list_jobs, ['GET'])
bind(router, "/api/jobs/{job_id}", jobs_service.get_job_details, ['GET'])
bind(router, "/api/jobs/{job_id}", jobs_service.delete_job, ['DELETE'])
bind(router, "/api/jobs/{job_id}/dimensions", jobs_service.get_job_dimensions, ['GET'])
bind(router, "/api/jobs/{job_id}/bends", jobs_service.get_job_bends, ['GET'])
bind(router, "/api/jobs/{job_id}/correlations", jobs_service.get_job_correlations, ['GET'])
bind(router, "/api/jobs/{job_id}/enhanced-analysis", jobs_service.get_enhanced_analysis, ['GET'])
bind(router, "/api/stats", jobs_service.get_stats, ['GET'])
