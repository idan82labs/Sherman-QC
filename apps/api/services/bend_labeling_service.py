"""Application service for the standalone human bend-labeling studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse

from backend.bend_training_labels import (
    BendTrainingLabelError,
    list_ply_assets,
    load_labels,
    resolve_asset,
    save_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS_ROOT = PROJECT_ROOT / "data" / "bend_corpus" / "annotations" / "human_bend_lines_v1"


def _resolve_or_404(asset_id: str):
    try:
        return resolve_asset(PROJECT_ROOT, LABELS_ROOT, asset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="PLY asset not found") from exc


def list_assets(
    q: Optional[str] = Query(default=None, max_length=200),
    source: Optional[str] = Query(default=None),
    asset_kind: Optional[str] = Query(default=None),
    label_status: Optional[str] = Query(default=None),
):
    inventory = list_ply_assets(PROJECT_ROOT, LABELS_ROOT)
    assets = inventory
    if q:
        needle = q.casefold()
        assets = [
            asset for asset in assets
            if needle in asset["name"].casefold() or needle in asset["relative_path"].casefold()
        ]
    if source:
        assets = [asset for asset in assets if asset["source"] == source]
    if asset_kind:
        assets = [asset for asset in assets if asset["asset_kind"] == asset_kind]
    if label_status:
        assets = [asset for asset in assets if asset["label_status"] == label_status]
    assets.sort(key=lambda asset: (
        asset["label_status"] == "complete",
        asset["source"] not in {"source-assets", "data", "uploads", "live_scans"},
        asset["name"].casefold(),
        asset["relative_path"].casefold(),
    ))
    return {
        "assets": assets,
        "total": len(assets),
        "inventory_total": len(inventory),
        "labels_root": "data/bend_corpus/annotations/human_bend_lines_v1",
    }


def get_asset_file(asset_id: str):
    asset, path = _resolve_or_404(asset_id)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=asset["name"],
        headers={"Cache-Control": "private, max-age=3600"},
    )


def get_asset_labels(asset_id: str):
    asset, _path = _resolve_or_404(asset_id)
    try:
        return load_labels(LABELS_ROOT, asset)
    except BendTrainingLabelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def put_asset_labels(asset_id: str, payload: dict[str, Any]):
    asset, _path = _resolve_or_404(asset_id)
    try:
        return save_labels(LABELS_ROOT, payload, asset)
    except BendTrainingLabelError as exc:
        status = 409 if str(exc).startswith("revision conflict:") else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
