from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import spc_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["SPC"])

bind(router, "/api/spc/capability", spc_service.calculate_capability, ['POST'])
bind(router, "/api/spc/control-charts", spc_service.generate_control_charts, ['POST'])
bind(router, "/api/spc/histogram", spc_service.generate_histogram, ['POST'])
bind(router, "/api/spc/analyze", spc_service.full_spc_analysis, ['POST'])
