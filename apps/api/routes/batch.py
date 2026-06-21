from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import batch_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Batch"])

bind(router, "/api/batch/analyze", batch_service.start_batch_analysis, ['POST'])
bind(router, "/api/batch/{batch_id}", batch_service.get_batch_status, ['GET'])
bind(router, "/api/batch/{batch_id}/summary", batch_service.get_batch_summary, ['GET'])
bind(router, "/api/batch/{batch_id}/stream", batch_service.stream_batch_progress, ['GET'])
bind(router, "/api/batch", batch_service.list_batches, ['GET'])
bind(router, "/api/batch/{batch_id}", batch_service.delete_batch, ['DELETE'])
