from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import system_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["System"])

bind(router, "/", system_service.root, ['GET'])
bind(router, "/api/health", system_service.health, ['GET'])
