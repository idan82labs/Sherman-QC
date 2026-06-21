from __future__ import annotations

from fastapi import APIRouter
from apps.api.services import gdt_service
from apps.api.routes._helpers import bind

router = APIRouter(tags=["GD&T"])

bind(router, "/api/gdt/flatness", gdt_service.calculate_flatness, ['POST'])
bind(router, "/api/gdt/cylindricity", gdt_service.calculate_cylindricity, ['POST'])
bind(router, "/api/gdt/circularity", gdt_service.calculate_circularity, ['POST'])
bind(router, "/api/gdt/position", gdt_service.calculate_position, ['POST'])
bind(router, "/api/gdt/parallelism", gdt_service.calculate_parallelism, ['POST'])
bind(router, "/api/gdt/perpendicularity", gdt_service.calculate_perpendicularity, ['POST'])
bind(router, "/api/gdt/types", gdt_service.get_gdt_types, ['GET'])
