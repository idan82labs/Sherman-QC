from __future__ import annotations

from fastapi import APIRouter, Depends
from apps.api.services import bend_inspection_service
from apps.api.services import bend_readiness_service

router = APIRouter(tags=["Bend Inspection"])


def bind_operator(path: str, endpoint, methods: list[str]):
    router.add_api_route(path, endpoint, methods=methods,
                         dependencies=[Depends(bend_inspection_service.require_bend_operator)])


def bind_owned(path: str, endpoint, methods: list[str]):
    router.add_api_route(
        path, endpoint, methods=methods,
        dependencies=[Depends(bend_inspection_service.require_bend_operator),
                      Depends(bend_inspection_service.require_bend_job_owner)],
    )

bind_operator("/api/bend-inspection/analyze", bend_inspection_service.run_bend_inspection, ['POST'])
bind_operator("/api/bend-inspection/jobs", bend_inspection_service.list_bend_inspection_jobs, ['GET'])
bind_operator("/api/bend-inspection/readiness", bend_readiness_service.get_bend_inspection_readiness, ['GET'])
bind_owned("/api/bend-inspection/{job_id}", bend_inspection_service.get_bend_inspection_result, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/labels", bend_inspection_service.get_bend_inspection_labels, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/labels", bend_inspection_service.put_bend_inspection_labels, ['PUT'])
bind_owned("/api/bend-inspection/{job_id}/pdf", bend_inspection_service.download_bend_inspection_pdf, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/overlay/manifest.json", bend_inspection_service.get_bend_overlay_manifest, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/overlay/overview.png", bend_inspection_service.get_bend_overlay_overview, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/overlay/issues/{filename}", bend_inspection_service.get_bend_overlay_issue, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/overlay/{filename}", bend_inspection_service.get_bend_overlay_file, ['GET'])
bind_owned("/api/bend-inspection/{job_id}/table", bend_inspection_service.get_bend_inspection_table, ['GET'])
bind_operator("/api/bend-inspection/extract-cad", bend_inspection_service.extract_cad_bends, ['POST'])
bind_operator("/api/bend-inspection/analyze-from-catalog", bend_inspection_service.analyze_from_catalog, ['POST'])
bind_operator("/api/parts/with-cad", bend_inspection_service.get_parts_with_cad, ['GET'])
bind_owned("/api/bend-inspection/{job_id}", bend_inspection_service.delete_bend_inspection_job, ['DELETE'])

# These two URLs are shared with the general QC workflow; conditionally guard
# the bend namespace so either router order preserves ordinary QC behavior.
router.add_api_route("/api/aligned-scan/{job_id}.ply", bend_inspection_service.get_aligned_scan,
                     methods=['GET', 'HEAD'], dependencies=[Depends(bend_inspection_service.require_bend_job_owner_if_applicable)])
router.add_api_route("/api/reference-mesh/{job_id}.ply", bend_inspection_service.get_reference_mesh,
                     methods=['GET', 'HEAD'], dependencies=[Depends(bend_inspection_service.require_bend_job_owner_if_applicable)])
