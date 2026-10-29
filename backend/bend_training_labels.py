"""Inventory and durable storage for browser-authored bend training labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
LABEL_KIND = "sherman_qc_human_bend_labels"
COORDINATE_FRAME = "source_ply_raw"
REVIEW_STATUSES = frozenset({"in_progress", "complete", "needs_review"})
ASSET_SOURCES = ("source-assets", "data", "uploads", "live_scans", "output")


class BendTrainingLabelError(ValueError):
    """Raised when an inventory or label contract is invalid."""


def default_asset_roots(project_root: Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    return {
        "source-assets": (root.parent / "source-assets").resolve(),
        "data": (root / "data").resolve(),
        "uploads": (root / "uploads").resolve(),
        "live_scans": (root / "live_scans").resolve(),
        "output": (root / "output").resolve(),
    }


def asset_id_for(source: str, relative_path: str) -> str:
    payload = f"{source}:{relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _classify_asset(source: str, relative_path: str) -> str:
    name = Path(relative_path).name.lower()
    if "reference_mesh" in name or "reference-mesh" in name:
        return "reference_mesh"
    if "aligned_scan" in name or "aligned-scan" in name:
        return "aligned_scan"
    if source == "live_scans":
        return "live_scan"
    return "scan"


def _read_ply_header(path: Path) -> tuple[str | None, int | None, int | None]:
    """Read only the bounded PLY header, never the point payload."""
    try:
        with path.open("rb") as handle:
            header = handle.read(256 * 1024)
    except OSError:
        return None, None, None
    marker = b"end_header"
    end = header.find(marker)
    if end < 0:
        return None, None, None
    text = header[: end + len(marker)].decode("ascii", errors="replace")
    ply_format: str | None = None
    vertices: int | None = None
    faces: int | None = None
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 2 and parts[0] == "format":
            ply_format = parts[1]
        elif len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            try:
                vertices = int(parts[2])
            except ValueError:
                vertices = None
        elif len(parts) == 3 and parts[:2] == ["element", "face"]:
            try:
                faces = int(parts[2])
            except ValueError:
                faces = None
    return ply_format, vertices, faces


def _label_path(labels_root: Path, asset_id: str) -> Path:
    return Path(labels_root) / f"{asset_id}.json"


def _load_unchecked_label_summary(labels_root: Path, asset_id: str) -> dict[str, Any]:
    path = _label_path(labels_root, asset_id)
    if not path.is_file():
        return {
            "label_status": "not_started",
            "bend_count": 0,
            "annotator": None,
            "revision": 0,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "label_status": "needs_review",
            "bend_count": 0,
            "annotator": None,
            "revision": 0,
        }
    return {
        "label_status": payload.get("review_status", "needs_review"),
        "bend_count": len(payload.get("bends", [])) if isinstance(payload.get("bends"), list) else 0,
        "annotator": payload.get("annotator"),
        "revision": payload.get("revision", 0),
    }


def iter_ply_assets(
    project_root: Path,
    labels_root: Path,
    *,
    roots: dict[str, Path] | None = None,
) -> Iterable[dict[str, Any]]:
    configured = roots or default_asset_roots(project_root)
    for source in ASSET_SOURCES:
        source_root = configured.get(source)
        if source_root is None or not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.ply"), key=lambda item: str(item).lower()):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(source_root.resolve()).as_posix()
            except ValueError:
                continue
            asset_id = asset_id_for(source, relative)
            stat = resolved.stat()
            ply_format, vertex_count, face_count = _read_ply_header(resolved)
            yield {
                "asset_id": asset_id,
                "name": resolved.name,
                "source": source,
                "relative_path": relative,
                "asset_kind": _classify_asset(source, relative),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "ply_format": ply_format,
                "vertex_count": vertex_count,
                "face_count": face_count,
                **_load_unchecked_label_summary(labels_root, asset_id),
            }


def list_ply_assets(project_root: Path, labels_root: Path) -> list[dict[str, Any]]:
    return list(iter_ply_assets(project_root, labels_root))


def resolve_asset(
    project_root: Path,
    labels_root: Path,
    asset_id: str,
) -> tuple[dict[str, Any], Path]:
    for asset in iter_ply_assets(project_root, labels_root):
        if asset["asset_id"] != asset_id:
            continue
        roots = default_asset_roots(project_root)
        root = roots[asset["source"]]
        path = (root / asset["relative_path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise BendTrainingLabelError("asset path escaped its configured root") from exc
        if not path.is_file() or path.suffix.lower() != ".ply":
            raise FileNotFoundError(asset_id)
        return asset, path
    raise FileNotFoundError(asset_id)


def empty_labels(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LABEL_KIND,
        "asset_id": asset["asset_id"],
        "source": asset["source"],
        "source_relative_path": asset["relative_path"],
        "coordinate_frame": COORDINATE_FRAME,
        "review_status": "in_progress",
        "annotator": "",
        "revision": 0,
        "no_bends_present": False,
        "bends": [],
        "notes": "",
    }


def _point(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise BendTrainingLabelError(f"{path} must be a three-number array")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BendTrainingLabelError(f"{path}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise BendTrainingLabelError(f"{path}[{index}] must be a finite number")
        result.append(number)
    return result


def validate_labels(payload: Any, asset: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BendTrainingLabelError("payload must be a JSON object")
    allowed = {
        "schema_version", "kind", "asset_id", "source", "source_relative_path",
        "coordinate_frame", "review_status", "annotator", "revision",
        "no_bends_present", "bends", "notes", "created_at", "updated_at",
    }
    extra = sorted(set(payload) - allowed)
    if extra:
        raise BendTrainingLabelError(f"payload contains unexpected keys: {', '.join(extra)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BendTrainingLabelError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("kind") != LABEL_KIND:
        raise BendTrainingLabelError(f"kind must be {LABEL_KIND}")
    if payload.get("asset_id") != asset["asset_id"]:
        raise BendTrainingLabelError("asset_id does not match the requested PLY")
    if payload.get("source") != asset["source"]:
        raise BendTrainingLabelError("source does not match the requested PLY")
    if payload.get("source_relative_path") != asset["relative_path"]:
        raise BendTrainingLabelError("source_relative_path does not match the requested PLY")
    if payload.get("coordinate_frame") != COORDINATE_FRAME:
        raise BendTrainingLabelError(f"coordinate_frame must be {COORDINATE_FRAME}")
    status = payload.get("review_status")
    if status not in REVIEW_STATUSES:
        raise BendTrainingLabelError("review_status is invalid")
    annotator = payload.get("annotator")
    if not isinstance(annotator, str) or not annotator.strip():
        raise BendTrainingLabelError("annotator must be a non-empty string")
    annotator = annotator.strip()
    if len(annotator) > 120:
        raise BendTrainingLabelError("annotator must be 120 characters or fewer")
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise BendTrainingLabelError("revision must be a non-negative integer")
    no_bends = payload.get("no_bends_present")
    if not isinstance(no_bends, bool):
        raise BendTrainingLabelError("no_bends_present must be a boolean")
    notes = payload.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise BendTrainingLabelError("notes must be a string of 4000 characters or fewer")
    bends = payload.get("bends")
    if not isinstance(bends, list):
        raise BendTrainingLabelError("bends must be an array")
    if no_bends and bends:
        raise BendTrainingLabelError("no_bends_present cannot be true when bend lines exist")
    if status == "complete" and not bends and not no_bends:
        raise BendTrainingLabelError("complete labels require bend lines or no_bends_present=true")
    normalized_bends: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, bend in enumerate(bends):
        path = f"bends[{index}]"
        if not isinstance(bend, dict):
            raise BendTrainingLabelError(f"{path} must be an object")
        extra_bend = sorted(set(bend) - {"bend_id", "line_start", "line_end", "confidence", "notes"})
        if extra_bend:
            raise BendTrainingLabelError(f"{path} contains unexpected keys: {', '.join(extra_bend)}")
        bend_id = bend.get("bend_id")
        if not isinstance(bend_id, str) or not bend_id.strip():
            raise BendTrainingLabelError(f"{path}.bend_id must be a non-empty string")
        bend_id = bend_id.strip()
        if bend_id in ids:
            raise BendTrainingLabelError(f"duplicate bend_id: {bend_id}")
        ids.add(bend_id)
        start = _point(bend.get("line_start"), f"{path}.line_start")
        end = _point(bend.get("line_end"), f"{path}.line_end")
        if start == end:
            raise BendTrainingLabelError(f"{path} must define a nonzero line segment")
        confidence = bend.get("confidence", "certain")
        if confidence not in {"certain", "uncertain"}:
            raise BendTrainingLabelError(f"{path}.confidence must be certain or uncertain")
        bend_notes = bend.get("notes", "")
        if not isinstance(bend_notes, str) or len(bend_notes) > 1000:
            raise BendTrainingLabelError(f"{path}.notes must be 1000 characters or fewer")
        normalized_bends.append({
            "bend_id": bend_id,
            "line_start": start,
            "line_end": end,
            "confidence": confidence,
            "notes": bend_notes,
        })
    normalized = {
        **payload,
        "annotator": annotator,
        "notes": notes,
        "bends": normalized_bends,
    }
    try:
        return json.loads(json.dumps(normalized, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise BendTrainingLabelError(f"payload is not finite JSON: {exc}") from exc


def load_labels(labels_root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    path = _label_path(labels_root, asset["asset_id"])
    if not path.is_file():
        return empty_labels(asset)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BendTrainingLabelError(f"could not read saved labels: {exc}") from exc
    return validate_labels(payload, asset)


def save_labels(labels_root: Path, payload: Any, asset: dict[str, Any]) -> dict[str, Any]:
    validated = validate_labels(payload, asset)
    labels_root = Path(labels_root)
    labels_root.mkdir(parents=True, exist_ok=True)
    path = _label_path(labels_root, asset["asset_id"])
    current_revision = 0
    created_at = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        current = load_labels(labels_root, asset)
        current_revision = current["revision"]
        created_at = current.get("created_at", created_at)
    if validated["revision"] != current_revision:
        raise BendTrainingLabelError(
            f"revision conflict: expected {current_revision}, received {validated['revision']}"
        )
    now = datetime.now(timezone.utc).isoformat()
    saved = {
        **validated,
        "revision": current_revision + 1,
        "created_at": created_at,
        "updated_at": now,
    }
    serialized = json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{asset['asset_id']}.", suffix=".tmp", dir=labels_root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return saved
