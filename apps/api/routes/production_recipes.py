"""``/api/production`` routes (C05): parts, onboarding, release drafts, release gate, recipes, flats.

Endpoints (capabilities from C04 ``require_capability``)::

    GET    /parts                                       line.read
    POST   /parts                                       parts.write     201 {part, sku}
    GET    /parts/{part_id}/onboarding                  line.read
    GET    /parts/{part_id}/release-draft               line.read
    PUT    /parts/{part_id}/release-draft               parts.write     body ReleaseRequestV1
    POST   /parts/{part_id}/recipe/release              recipe.release  201 released / 200 unchanged
    GET    /parts/{part_id}/flat                        line.read
    POST   /parts/{part_id}/flat                        parts.write     multipart "file" (DXF / flat STEP)
    POST   /parts/{part_id}/flat/proposal               parts.write     ?expected_revision=N
    PUT    /parts/{part_id}/flat/binding                parts.write     {expected_revision, state, bend_line_map?, mirrored?}
    DELETE /parts/{part_id}/flat                        parts.write     ?expected_revision=N (required)
    GET    /parts/{part_id}/chirality                   line.read       left/right map status (schedules a run)
    POST   /parts/{part_id}/chirality/retry             parts.write     re-run a FAILED map
    GET    /recipes/{recipe_id}/chirality               line.read       read-only (station: ?station_id=)
    GET    /recipes/by-sku/{sku}                        line.read
    GET    /recipes/{recipe_id}                         line.read
    GET    /recipes/{recipe_id}/expectations            line.read
    POST   /recipes/{recipe_id}/expectations/recompute  recipe.release  202 {job_id, created}
    POST   /recipes/{recipe_id}/retire                  admin

Errors: ``{"detail": {"codes": [...], "messages": {code: text}, ...}}`` with
404 / 409 / 413 / 422 / 502 / 503 / 504 (``production_recipe_service``).
A release or release draft with any ``confirmation`` but OPERATOR_CONFIRMED
is refused 422 CONFIRMATION_NOT_ALLOWED; DELETE /flat without
``expected_revision`` is 422 FLAT_REVISION_REQUIRED.
A release refused by the compile also carries ``detail.compile`` (E1 issues,
unmarked graph bends) for the wizard. Not configured -> 503 (C04).

Dependencies that tests (and C11) override: :func:`get_line_settings`
(re-exported from C04, so the capability checks see the same settings),
:func:`get_recipe_store`, :func:`get_line_store`, :func:`get_catalog_port`,
:func:`get_clock`, :func:`get_compiler`, :func:`get_flat_tool` and
:func:`get_chirality_service`.

The chirality routes are display only (``apps.production_line.recipes.chirality``):
the left/right map of the active recipe's installed compile, never an inspection input.

This module never imports the legacy server: the catalog port resolves
``legacy.get_catalog()`` lazily on first use.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from apps.api.services.production_recipe_service import (
    RecipeApiError,
    RecipeContext,
    RecipeService,
)
from apps.production_line.recipes.store import RecipeStore, utc_now
from apps.production_line.security.dependencies import (
    get_line_settings,
    require_capability,
    require_line_settings,
)
from apps.production_line.security.principals import Principal
from apps.production_line.settings import LineSettings

__all__ = [
    "router",
    "get_line_settings",
    "get_recipe_store",
    "get_line_store",
    "get_catalog_port",
    "get_clock",
    "get_compiler",
    "get_flat_tool",
    "get_recipe_context",
    "get_chirality_service",
]

router = APIRouter(prefix="/api/production", tags=["production"])

_stores: dict[Path, Any] = {}
_stores_lock = threading.Lock()
_catalog_singleton: dict[str, Any] = {}


def _cached(kind: str, path: Path, factory: Callable[[Path], Any]) -> Any:
    key = Path(f"{kind}:{path}")
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = _stores[key] = factory(path)
        return store


def get_recipe_store(settings: LineSettings = Depends(require_line_settings)) -> RecipeStore:
    return _cached("recipes", settings.paths.db_recipes, RecipeStore)


def get_line_store(settings: LineSettings = Depends(require_line_settings)) -> Any:
    from apps.production_line.store.line_store import LineStore

    return _cached("line", settings.paths.db_line, LineStore)


def get_catalog_port() -> Any:
    """The legacy catalog (``legacy.get_catalog()``, resolved lazily on first use)."""
    from apps.production_line.recipes.catalog_port import LegacyCatalogPort

    port = _catalog_singleton.get("port")
    if port is None:
        port = _catalog_singleton["port"] = LegacyCatalogPort()
    return port


def get_clock() -> Callable[[], datetime]:
    return utc_now


def get_compiler(settings: LineSettings = Depends(require_line_settings)) -> Any:
    from apps.production_line.recipes.release_service import CompileLauncher

    return CompileLauncher.from_settings(settings)


def get_flat_tool(settings: LineSettings = Depends(require_line_settings)) -> Any:
    from apps.production_line.recipes.flat_intake import FlatTool

    return FlatTool.from_settings(settings)


def get_chirality_service(settings: LineSettings = Depends(require_line_settings)) -> Any:
    from apps.production_line.recipes.chirality import ChiralityLauncher, ChiralityService

    def factory(root: Path) -> Any:
        return ChiralityService(data_root=root, runner=ChiralityLauncher.from_settings(settings).run)

    return _cached("chirality", Path(settings.data_root), factory)


def get_recipe_context(
    settings: LineSettings = Depends(require_line_settings),
    store: RecipeStore = Depends(get_recipe_store),
    line_store: Any = Depends(get_line_store),
    catalog: Any = Depends(get_catalog_port),
    clock: Callable[[], datetime] = Depends(get_clock),
    compiler: Any = Depends(get_compiler),
    flat_tool: Any = Depends(get_flat_tool),
) -> RecipeContext:
    return RecipeContext(catalog=catalog, store=store, data_root=Path(settings.data_root), line_store=line_store,
                         compiler=compiler, flat_tool=flat_tool, clock=clock)


def _service(ctx: RecipeContext = Depends(get_recipe_context)) -> RecipeService:
    return RecipeService(ctx)


def _http(exc: RecipeApiError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.detail())


def _call(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except RecipeApiError as exc:
        raise _http(exc) from None


def _user(principal: Principal) -> Principal:
    # Write routes are user-only (devices hold line.read at most; this is belt and braces).
    if principal.kind != "user":
        raise HTTPException(status_code=403, detail="capability requires a user",
                            headers={"X-Sherman-Auth-Code": "CAPABILITY_REQUIRED"})
    return principal


# ---------------------------------------------------------------------------- parts


@router.get("/parts")
def list_parts(
    search: Optional[str] = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.list_parts, search, limit, offset)


@router.post("/parts", status_code=201)
def create_part(
    body: dict[str, Any] = Body(...),
    principal: Principal = Depends(require_capability("parts.write")),
    service: RecipeService = Depends(_service),
) -> Any:
    _user(principal)
    return _call(service.create_part, body)


@router.get("/parts/{part_id}/onboarding")
def onboarding(
    part_id: str,
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.onboarding, part_id)


@router.get("/parts/{part_id}/release-draft")
def get_release_draft(
    part_id: str,
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.get_release_draft, part_id)


@router.put("/parts/{part_id}/release-draft")
def put_release_draft(
    part_id: str,
    body: dict[str, Any] = Body(...),
    principal: Principal = Depends(require_capability("parts.write")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.put_release_draft, part_id, body, _user(principal))


@router.post("/parts/{part_id}/recipe/release")
def release_recipe(
    part_id: str,
    body: dict[str, Any] = Body(...),
    principal: Principal = Depends(require_capability("recipe.release")),
    service: RecipeService = Depends(_service),
) -> Any:
    status, doc = _call(service.release, part_id, body, _user(principal))
    return JSONResponse(status_code=status, content=doc)


# ---------------------------------------------------------------------------- chirality (display only)


def _active_recipe_of_part(part_id: str, store: RecipeStore, catalog: Any) -> Any:
    from domains.production.recipe.canonical_sku import SkuError, canonical_sku

    part = catalog.get_part(part_id)
    if part is None:
        raise HTTPException(status_code=404, detail={"codes": ["PART_NOT_FOUND"],
                                                     "messages": {"PART_NOT_FOUND": f"part {part_id} not found"}})
    try:
        sku: Optional[str] = canonical_sku(str(part.get("part_number") or ""))
    except SkuError:
        sku = None
    active = store.get_active(sku) if sku is not None else None
    return active if active is not None else store.get_active_for_part(part_id)


@router.get("/parts/{part_id}/chirality")
def part_chirality(
    part_id: str,
    _p: Principal = Depends(require_capability("line.read")),
    store: RecipeStore = Depends(get_recipe_store),
    catalog: Any = Depends(get_catalog_port),
    chirality: Any = Depends(get_chirality_service),
) -> Any:
    doc = chirality.ensure(store, _active_recipe_of_part(part_id, store, catalog))
    return {**doc, "part_id": part_id}


@router.post("/parts/{part_id}/chirality/retry")
def retry_part_chirality(
    part_id: str,
    principal: Principal = Depends(require_capability("parts.write")),
    store: RecipeStore = Depends(get_recipe_store),
    catalog: Any = Depends(get_catalog_port),
    chirality: Any = Depends(get_chirality_service),
) -> Any:
    _user(principal)
    doc = chirality.ensure(store, _active_recipe_of_part(part_id, store, catalog), retry_failed=True)
    return {**doc, "part_id": part_id}


# ---------------------------------------------------------------------------- flats


@router.get("/parts/{part_id}/flat")
def get_flat(
    part_id: str,
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.get_flat, part_id)


@router.post("/parts/{part_id}/flat", status_code=201)
def upload_flat(
    part_id: str,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_capability("parts.write")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.upload_flat, part_id, file.file, file.filename or "", _user(principal))


@router.post("/parts/{part_id}/flat/proposal")
def propose_flat(
    part_id: str,
    expected_revision: Optional[int] = Query(default=None, ge=1),
    principal: Principal = Depends(require_capability("parts.write")),
    service: RecipeService = Depends(_service),
) -> Any:
    _user(principal)
    return _call(service.propose_flat, part_id, expected_revision)


@router.put("/parts/{part_id}/flat/binding")
def set_flat_binding(
    part_id: str,
    body: dict[str, Any] = Body(...),
    principal: Principal = Depends(require_capability("parts.write")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.set_flat_binding, part_id, body, _user(principal))


@router.delete("/parts/{part_id}/flat")
def delete_flat(
    part_id: str,
    expected_revision: Optional[int] = Query(default=None, ge=1),
    principal: Principal = Depends(require_capability("parts.write")),
    service: RecipeService = Depends(_service),
) -> Any:
    _user(principal)
    return _call(service.delete_flat, part_id, expected_revision)


# ---------------------------------------------------------------------------- recipes


@router.get("/recipes/by-sku/{sku}")
def recipe_by_sku(
    sku: str,
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.recipe_by_sku, sku)


@router.get("/recipes/{recipe_id}")
def get_recipe(
    recipe_id: str,
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.recipe, recipe_id)


@router.get("/recipes/{recipe_id}/expectations")
def get_expectations(
    recipe_id: str,
    _p: Principal = Depends(require_capability("line.read")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.expectations, recipe_id)


@router.post("/recipes/{recipe_id}/expectations/recompute", status_code=202)
def recompute_expectations(
    recipe_id: str,
    principal: Principal = Depends(require_capability("recipe.release")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.recompute, recipe_id, _user(principal))


@router.post("/recipes/{recipe_id}/retire")
def retire_recipe(
    recipe_id: str,
    principal: Principal = Depends(require_capability("admin")),
    service: RecipeService = Depends(_service),
) -> Any:
    return _call(service.retire, recipe_id, _user(principal))


@router.get("/recipes/{recipe_id}/chirality")
def recipe_chirality(
    recipe_id: str,
    _p: Principal = Depends(require_capability("line.read", "station_id")),
    store: RecipeStore = Depends(get_recipe_store),
    chirality: Any = Depends(get_chirality_service),
) -> Any:
    stored = store.get(recipe_id)
    if stored is None:
        raise HTTPException(status_code=404, detail={"codes": ["RECIPE_NOT_FOUND"],
                                                     "messages": {"RECIPE_NOT_FOUND": f"recipe {recipe_id} not found"}})
    return chirality.read(store, stored)


def reset_caches() -> None:
    """Drop cached stores and the catalog port (tests)."""
    with _stores_lock:
        _stores.clear()
    _catalog_singleton.clear()
