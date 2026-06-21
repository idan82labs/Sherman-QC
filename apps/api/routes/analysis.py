from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import analysis_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["Analysis"])

bind(router, "/api/analyze", analysis_service.start_analysis, ['POST'])
