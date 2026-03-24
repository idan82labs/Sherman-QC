from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import bend_inspection_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Bend Inspection"])

bind(router, "/api/bend-inspection/analyze", bend_inspection_service.run_bend_inspection, ['POST'])
bind(router, "/api/bend-inspection/jobs", bend_inspection_service.list_bend_inspection_jobs, ['GET'])
bind(router, "/api/bend-inspection/{job_id}", bend_inspection_service.get_bend_inspection_result, ['GET'])
bind(router, "/api/bend-inspection/{job_id}/pdf", bend_inspection_service.download_bend_inspection_pdf, ['GET'])
bind(router, "/api/bend-inspection/{job_id}/overlay/manifest.json", bend_inspection_service.get_bend_overlay_manifest, ['GET'])
bind(router, "/api/bend-inspection/{job_id}/overlay/overview.png", bend_inspection_service.get_bend_overlay_overview, ['GET'])
bind(router, "/api/bend-inspection/{job_id}/overlay/issues/{filename}", bend_inspection_service.get_bend_overlay_issue, ['GET'])
bind(router, "/api/bend-inspection/{job_id}/overlay/{filename}", bend_inspection_service.get_bend_overlay_file, ['GET'])
bind(router, "/api/bend-inspection/{job_id}/table", bend_inspection_service.get_bend_inspection_table, ['GET'])
bind(router, "/api/bend-inspection/extract-cad", bend_inspection_service.extract_cad_bends, ['POST'])
bind(router, "/api/bend-inspection/analyze-from-catalog", bend_inspection_service.analyze_from_catalog, ['POST'])
bind(router, "/api/parts/with-cad", bend_inspection_service.get_parts_with_cad, ['GET'])
bind(router, "/api/bend-inspection/{job_id}", bend_inspection_service.delete_bend_inspection_job, ['DELETE'])

bind(router, "/api/aligned-scan/{job_id}.ply", bend_inspection_service.get_aligned_scan, ['GET', 'HEAD'])
bind(router, "/api/reference-mesh/{job_id}.ply", bend_inspection_service.get_reference_mesh, ['GET', 'HEAD'])
