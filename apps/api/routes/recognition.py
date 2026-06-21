from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import recognition_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Part Recognition"])

bind(router, "/api/recognize", recognition_service.recognize_part, ['POST'])
bind(router, "/api/recognize/status", recognition_service.get_recognition_status, ['GET'])
bind(router, "/api/recognize/compare", recognition_service.compare_embeddings, ['POST'])
bind(router, "/api/recognize/reindex", recognition_service.reindex_all_parts, ['POST'])
