from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter


def _load_all_routers() -> list[APIRouter]:
    from apps.api.routes.analysis import router as analysis_router
    from apps.api.routes.auth import router as auth_router
    from apps.api.routes.batch import router as batch_router
    from apps.api.routes.bend_inspection import router as bend_inspection_router
    from apps.api.routes.gdt import router as gdt_router
    from apps.api.routes.jobs import router as jobs_router
    from apps.api.routes.live_scan import router as live_scan_router
    from apps.api.routes.manual_assistant import router as manual_assistant_router
    from apps.api.routes.part_catalog import router as part_catalog_router
    from apps.api.routes.recognition import router as recognition_router
    from apps.api.routes.reports import router as reports_router
    from apps.api.routes.spc import router as spc_router
    from apps.api.routes.system import router as system_router

    return [
        system_router,
        auth_router,
        analysis_router,
        jobs_router,
        reports_router,
        batch_router,
        gdt_router,
        spc_router,
        bend_inspection_router,
        part_catalog_router,
        recognition_router,
        live_scan_router,
        manual_assistant_router,
    ]


class _LazyRouters:
    def __iter__(self) -> Iterator[APIRouter]:
        return iter(_load_all_routers())


ALL_ROUTERS = _LazyRouters()
