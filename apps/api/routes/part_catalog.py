from __future__ import annotations

from fastapi import APIRouter, Depends
from apps.api.services.bend_inspection_service import require_bend_operator
from apps.api.services import part_catalog_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Part Catalog"])

bind(router, "/api/parts", part_catalog_service.list_parts, ['GET'])
bind(router, "/api/parts", part_catalog_service.create_part, ['POST'])
bind(router, "/api/parts/stats", part_catalog_service.get_parts_stats, ['GET'])
bind(router, "/api/parts/{part_id}", part_catalog_service.get_part, ['GET'])
bind(router, "/api/parts/{part_id}", part_catalog_service.update_part, ['PUT'])
bind(router, "/api/parts/{part_id}", part_catalog_service.delete_part, ['DELETE'])
router.add_api_route("/api/parts/{part_id}/cad", part_catalog_service.upload_part_cad,
                     methods=['POST'], dependencies=[Depends(require_bend_operator)])
router.add_api_route("/api/parts/{part_id}/drawing", part_catalog_service.upload_part_drawing,
                     methods=['POST'], dependencies=[Depends(require_bend_operator)])
router.add_api_route("/api/parts/{part_id}/preview.ply", part_catalog_service.download_part_preview,
                     methods=['GET'], dependencies=[Depends(require_bend_operator)])
bind(router, "/api/parts/{part_id}/cad", part_catalog_service.download_part_cad, ['GET'])
bind(router, "/api/parts/{part_id}/bend-specs", part_catalog_service.add_bend_spec, ['POST'])
bind(router, "/api/parts/{part_id}/bend-specs", part_catalog_service.get_bend_specs, ['GET'])
bind(router, "/api/parts/import-csv", part_catalog_service.import_parts_csv, ['POST'])
bind(router, "/api/parts/by-number/{part_number}", part_catalog_service.get_part_by_number, ['GET'])
