from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import part_catalog_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Part Catalog"])

bind(router, "/api/parts", part_catalog_service.list_parts, ['GET'])
bind(router, "/api/parts", part_catalog_service.create_part, ['POST'])
bind(router, "/api/parts/stats", part_catalog_service.get_parts_stats, ['GET'])
bind(router, "/api/parts/{part_id}", part_catalog_service.get_part, ['GET'])
bind(router, "/api/parts/{part_id}", part_catalog_service.update_part, ['PUT'])
bind(router, "/api/parts/{part_id}", part_catalog_service.delete_part, ['DELETE'])
bind(router, "/api/parts/{part_id}/cad", part_catalog_service.upload_part_cad, ['POST'])
bind(router, "/api/parts/{part_id}/cad", part_catalog_service.download_part_cad, ['GET'])
bind(router, "/api/parts/{part_id}/bend-specs", part_catalog_service.add_bend_spec, ['POST'])
bind(router, "/api/parts/{part_id}/bend-specs", part_catalog_service.get_bend_specs, ['GET'])
bind(router, "/api/parts/import-csv", part_catalog_service.import_parts_csv, ['POST'])
bind(router, "/api/parts/by-number/{part_number}", part_catalog_service.get_part_by_number, ['GET'])
