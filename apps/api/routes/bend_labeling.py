from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends

from apps.api.legacy_context import legacy
from apps.api.services import bend_labeling_service


router = APIRouter(
    prefix="/api/training/bend-labeling",
    tags=["Training Bend Labeling"],
    dependencies=[Depends(legacy.require_auth)],
)


@router.get("/assets")
def list_assets(
    q: Optional[str] = None,
    source: Optional[str] = None,
    asset_kind: Optional[str] = None,
    label_status: Optional[str] = None,
):
    return bend_labeling_service.list_assets(q, source, asset_kind, label_status)


@router.get("/assets/{asset_id}/file")
def get_asset_file(asset_id: str):
    return bend_labeling_service.get_asset_file(asset_id)


@router.get("/assets/{asset_id}/labels")
def get_asset_labels(asset_id: str):
    return bend_labeling_service.get_asset_labels(asset_id)


@router.put("/assets/{asset_id}/labels")
def put_asset_labels(asset_id: str, payload: dict[str, Any]):
    return bend_labeling_service.put_asset_labels(asset_id, payload)
