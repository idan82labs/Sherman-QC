#!/usr/bin/env python3
"""Deterministic native-resolution renderer for EXP_CAD_0030 CF-PDG V3.

This module is deliberately data-only.  It accepts an already-computed,
source-only render payload and never opens a scan, CAD source, evaluator
manifest, label, gold record, expected count, or quality token.  The caller is
responsible for scientific computation; this file only validates and renders
the frozen visual contract.

Every asset is 3000 x 2400 pixels.  The global paired diagnostic is emitted
even when no component survives.  Local packs are emitted for the deterministic
union of:

* every retained component;
* every component changed by any robustness row; and
* the top twelve central nodes by descending amplitude, then ascending node ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


EXPERIMENT_ID = "EXP_CAD_0030"
ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
SCHEMA = "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1"
MANIFEST_SCHEMA = "EXP_CAD_0030_CFPDG_NATIVE_VISUAL_MANIFEST_V1"
LEGACY_EXP29_SCHEMA = "EXP_CAD_0029_CFPDG_VISUAL_PAYLOAD_V1"
WIDTH_PX = 3000
HEIGHT_PX = 2400
DPI = 100
FOLD_HALO_MM = 6.408002808988148
TOP_NODE_COUNT = 12

BACKGROUND = "#081018"
PANEL = "#101a24"
TEXT = "#e8f0f5"
MUTED = "#91a1ae"
GRID = "#314352"
FIRST = "#43d5ff"
SECOND = "#ff8b43"
SELECTED = "#66e36f"
REJECTED = "#ef476f"
HALO = "#ffd166"
ACTIVE = "#f72585"


class RenderPayloadError(ValueError):
    """Raised before rendering when the source-only payload is malformed."""


def _array(value: Any, *, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 + len(shape_tail) or tuple(result.shape[1:]) != shape_tail:
        raise RenderPayloadError(f"{name} must have shape (N,{','.join(map(str, shape_tail))})")
    if not np.all(np.isfinite(result.astype(np.float64, copy=False))):
        raise RenderPayloadError(f"{name} contains non-finite values")
    return result


def _point_rows(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise RenderPayloadError(f"{name} must have finite shape (N,3)")
    return result


def _vector(value: Any, *, length: int, name: str, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (length,):
        raise RenderPayloadError(f"{name} must have length {length}")
    if np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result)):
        raise RenderPayloadError(f"{name} contains non-finite values")
    return result


def _mask(value: Any, *, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.shape != (length,):
        raise RenderPayloadError(f"{name} must have length {length}")
    return result


def _safe_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RenderPayloadError(f"{name} must be finite")
    return result


def _nullable_float(value: Any, *, name: str) -> float:
    if value is None:
        return math.nan
    result = float(value)
    if math.isinf(result):
        raise RenderPayloadError(f"{name} must not contain infinity; encode unavailable as null")
    return result


def _format_optional(value: Any, *, precision: int = 4) -> str:
    """Human-readable nullable scalar; never display a scientific null as NaN."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.{precision}f}" if math.isfinite(numeric) else "N/A"


def _nullable_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    if value is None:
        return np.full(length, math.nan, dtype=np.float64)
    result = np.asarray([math.nan if row is None else float(row) for row in value], dtype=np.float64)
    if result.shape != (length,) or np.any(np.isinf(result)):
        raise RenderPayloadError(f"{name} must have length {length} with finite-or-null values")
    return result


def _nullable_sheet_vector(
    value: Any,
    *,
    length: int,
    name: str,
    valid_mask: np.ndarray | None = None,
    selected_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Validate nullable two-sided ownership without a numeric fallback.

    Sheet ``0`` is not a side: valid ownership is exactly ``-1`` or ``+1``.
    When nearest-CAD validity is supplied, nullness is coupled to it in both
    directions. A local diagnostic without that mask may leave rejected
    support unowned, but every selected sample must have real ownership.
    """

    rows = [] if value is None else list(value)
    if len(rows) != length:
        raise RenderPayloadError(f"{name} must have length {length}")
    result = np.full(length, math.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if row is None:
            continue
        try:
            numeric = float(row)
        except (TypeError, ValueError) as error:
            raise RenderPayloadError(
                f"{name}[{index}] must be -1, +1, or null"
            ) from error
        if not math.isfinite(numeric) or numeric not in (-1.0, 1.0):
            raise RenderPayloadError(f"{name}[{index}] must be -1, +1, or null")
        result[index] = numeric
    owned = np.isfinite(result)
    if valid_mask is not None:
        validity = np.asarray(valid_mask, dtype=bool)
        if validity.shape != (length,):
            raise RenderPayloadError(f"{name} validity coupling length mismatch")
        if not np.array_equal(owned, validity):
            raise RenderPayloadError(
                f"{name} must be null iff nearest-CAD-valid is false"
            )
    if selected_mask is not None:
        selected = np.asarray(selected_mask, dtype=bool)
        if selected.shape != (length,):
            raise RenderPayloadError(f"{name} selection coupling length mismatch")
        if np.any(selected & ~owned):
            raise RenderPayloadError(f"{name} selected support requires ±1 ownership")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(row) for row in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    for value in result:
        if abs(float(value)) > 1e-15:
            if value < 0.0:
                result *= -1.0
            break
    result[result == 0.0] = 0.0
    return result


def _camera_basis(cad_xyz: np.ndarray, supplied: Any | None) -> np.ndarray:
    if supplied is not None:
        basis = np.asarray(supplied, dtype=np.float64)
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise RenderPayloadError("camera_basis_rows must be finite 3x3")
        gram = basis @ basis.T
        if not np.allclose(gram, np.eye(3), atol=1e-10, rtol=0.0):
            raise RenderPayloadError("camera_basis_rows must be orthonormal")
        if np.linalg.det(basis) < 0.0:
            raise RenderPayloadError("camera_basis_rows must be proper")
        return basis
    if len(cad_xyz) < 3:
        return np.eye(3)
    centered = cad_xyz - np.mean(cad_xyz, axis=0)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    first = _canonical_sign(vectors[:, order[0]])
    second = _canonical_sign(vectors[:, order[1]])
    third = np.cross(first, second)
    third /= max(float(np.linalg.norm(third)), 1e-15)
    second = np.cross(third, first)
    second /= max(float(np.linalg.norm(second)), 1e-15)
    return np.stack([first, second, third], axis=0)


def _project(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - center) @ basis.T


def _configure_axis(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, color=TEXT, fontsize=10, pad=6)
    axis.tick_params(colors=MUTED, labelsize=7)
    axis.grid(color=GRID, alpha=0.22, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def _empty(axis: plt.Axes, text: str) -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=12, transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _downsample_indices(length: int, maximum: int = 60000) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    step = int(math.ceil(length / maximum))
    return np.arange(0, length, step, dtype=np.int64)


def _discrete_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    palette = np.asarray(
        [
            [0.27, 0.84, 1.00, 0.86],
            [1.00, 0.55, 0.26, 0.86],
            [0.40, 0.89, 0.44, 0.86],
            [0.95, 0.15, 0.52, 0.86],
            [0.68, 0.46, 1.00, 0.86],
            [1.00, 0.82, 0.40, 0.86],
            [0.40, 0.78, 0.70, 0.86],
            [0.72, 0.78, 0.83, 0.86],
        ]
    )
    return palette[np.mod(values, len(palette))]


def _validate_member(member: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    expected_slot = "FIRST" if ordinal == 0 else "SECOND"
    if member.get("slot") != expected_slot:
        raise RenderPayloadError(f"member {ordinal} slot must be {expected_slot}")
    raw = _array(member.get("raw_xyz_mm", []), shape_tail=(3,), name=f"{expected_slot}.raw_xyz_mm")
    raw_valid = _mask(
        member.get("raw_nearest_cad_valid_mask", []),
        length=len(raw),
        name=f"{expected_slot}.raw_nearest_cad_valid_mask",
    )
    voxel = _array(member.get("voxel_xyz_mm", []), shape_tail=(3,), name=f"{expected_slot}.voxel_xyz_mm")
    voxel_valid = _mask(
        member.get("voxel_nearest_cad_valid_mask", []), length=len(voxel),
        name=f"{expected_slot}.voxel_nearest_cad_valid_mask",
    )
    selected = _mask(
        member.get("voxel_selected_by_any_winning_cluster_mask", []),
        length=len(voxel), name=f"{expected_slot}.voxel_selected_by_any_winning_cluster_mask",
    )
    if np.any(selected & ~voxel_valid):
        raise RenderPayloadError(
            f"{expected_slot} winning-cluster selection is not a subset of nearest-CAD validity"
        )
    regions = _vector(member.get("voxel_region_ids", []), length=len(voxel), name=f"{expected_slot}.voxel_region_ids", dtype=np.int64)
    sheets = _nullable_sheet_vector(
        member.get("voxel_sheet_ids", []),
        length=len(voxel),
        name=f"{expected_slot}.voxel_sheet_ids",
        valid_mask=voxel_valid,
        selected_mask=selected,
    )
    return {
        "slot": expected_slot,
        "opaque_token": str(member.get("opaque_token", expected_slot)),
        "raw": raw.astype(np.float64),
        "raw_valid": raw_valid,
        "voxel": voxel.astype(np.float64),
        "voxel_valid": voxel_valid,
        "selected": selected,
        "regions": regions,
        "sheets": sheets,
    }


def _validate_cad(cad: Mapping[str, Any]) -> dict[str, Any]:
    xyz = _array(cad.get("nodes_xyz_mm", []), shape_tail=(3,), name="cad.nodes_xyz_mm").astype(np.float64)
    count = len(xyz)
    normals = _array(cad.get("normals", []), shape_tail=(3,), name="cad.normals").astype(np.float64)
    if len(normals) != count:
        raise RenderPayloadError("cad.normals length mismatch")
    tangent_e1 = _array(
        cad.get("tangent_e1", []), shape_tail=(3,), name="cad.tangent_e1"
    ).astype(np.float64)
    tangent_e2 = _array(
        cad.get("tangent_e2", []), shape_tail=(3,), name="cad.tangent_e2"
    ).astype(np.float64)
    if len(tangent_e1) != count or len(tangent_e2) != count:
        raise RenderPayloadError("cad tangent basis length mismatch")
    if count:
        basis_norms = np.column_stack(
            (
                np.linalg.norm(tangent_e1, axis=1),
                np.linalg.norm(tangent_e2, axis=1),
                np.linalg.norm(normals, axis=1),
            )
        )
        orthogonality = np.column_stack(
            (
                np.einsum("ij,ij->i", tangent_e1, tangent_e2),
                np.einsum("ij,ij->i", tangent_e1, normals),
                np.einsum("ij,ij->i", tangent_e2, normals),
            )
        )
        handedness = np.cross(tangent_e1, tangent_e2)
        if not np.allclose(basis_norms, 1.0, atol=1e-8, rtol=0.0):
            raise RenderPayloadError("cad tangent_e1/tangent_e2/normals must be unit length")
        if not np.allclose(orthogonality, 0.0, atol=1e-8, rtol=0.0):
            raise RenderPayloadError("cad tangent_e1/tangent_e2/normals must be orthogonal")
        if not np.allclose(handedness, normals, atol=1e-8, rtol=0.0):
            raise RenderPayloadError(
                "cad tangent basis must be right-handed: cross(tangent_e1,tangent_e2)==normal"
            )
    edges = np.asarray(cad.get("edges", []), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise RenderPayloadError("cad.edges must have shape (E,2)")
    if np.any(edges < 0) or np.any(edges >= count):
        raise RenderPayloadError("cad.edges contains out-of-range node index")
    result: dict[str, Any] = {
        "xyz": xyz,
        "normals": normals,
        "tangent_e1": tangent_e1,
        "tangent_e2": tangent_e2,
        "edges": edges,
    }
    result["regions"] = _vector(
        cad.get("smooth_region_ids", []), length=count,
        name="cad.smooth_region_ids", dtype=np.int64,
    )
    result["folds"] = _vector(
        cad.get("fold_ids", []), length=count, name="cad.fold_ids", dtype=np.int64,
    )
    for key, output in (
        ("interior_mask", "interior"),
        ("common_support_mask", "common"),
        ("excluded_mask", "excluded"),
        ("active_mask", "active"),
    ):
        result[output] = _mask(cad.get(key, []), length=count, name=f"cad.{key}")
    halo_value = np.asarray(cad.get("fold_halo_mask", []), dtype=bool)
    if halo_value.shape == (count,):
        result["halo"] = halo_value
        result["fold_training_masks"] = None
    elif halo_value.ndim == 2 and halo_value.shape[1] == count:
        # Some producers preserve the four post-halo training masks.  Recover
        # the actual halo exclusion by removing the held fold itself from each
        # row's complement, then taking the union across held folds.
        held = np.arange(halo_value.shape[0], dtype=np.int64)[:, None] == result["folds"][None, :]
        result["halo"] = np.any(~halo_value & ~held, axis=0)
        result["fold_training_masks"] = halo_value
    else:
        raise RenderPayloadError("cad.fold_halo_mask must be N or F-by-N")
    result["area"] = _vector(cad.get("area_mm2", []), length=count, name="cad.area_mm2")
    if np.any(result["area"] < 0.0):
        raise RenderPayloadError("cad.area_mm2 must be nonnegative")
    return result


def _validate_node(node: Mapping[str, Any], cad_count: int) -> dict[str, Any]:
    node_id = int(node["node_id"])
    cad_index = int(node.get("cad_index", node_id))
    if cad_index < 0 or cad_index >= cad_count:
        raise RenderPayloadError(f"node {node_id} cad_index out of range")
    if "valid" not in node or "invalid_reason" not in node:
        raise RenderPayloadError(f"node {node_id} requires explicit valid and invalid_reason")
    node_valid = bool(node["valid"])
    node_invalid_reason = node["invalid_reason"]
    if node_valid and node_invalid_reason is not None:
        raise RenderPayloadError(f"node {node_id} valid row has invalid_reason")
    if not node_valid and not isinstance(node_invalid_reason, str):
        raise RenderPayloadError(f"node {node_id} invalid row needs a reason")
    diagnostics = node.get("member_diagnostics", [])
    if len(diagnostics) != 2:
        raise RenderPayloadError(f"node {node_id} requires two member_diagnostics")
    parsed = []
    for ordinal, diagnostic in enumerate(diagnostics):
        slot = "FIRST" if ordinal == 0 else "SECOND"
        if diagnostic.get("slot") != slot:
            raise RenderPayloadError(f"node {node_id} diagnostic {ordinal} slot mismatch")
        if "valid" not in diagnostic or "invalid_reason" not in diagnostic:
            raise RenderPayloadError(
                f"node {node_id}.{slot} requires explicit valid and invalid_reason"
            )
        diagnostic_valid = bool(diagnostic["valid"])
        invalid_reason = diagnostic["invalid_reason"]
        if diagnostic_valid and invalid_reason is not None:
            raise RenderPayloadError(f"node {node_id}.{slot} valid diagnostic has invalid_reason")
        if not diagnostic_valid and not isinstance(invalid_reason, str):
            raise RenderPayloadError(f"node {node_id}.{slot} invalid diagnostic needs a reason")
        local_xyz = _point_rows(
            diagnostic.get("local_support_xyz_mm", []),
            name=f"node {node_id}.{slot}.local_support_xyz_mm",
        )
        local_selected = _mask(
            diagnostic.get("local_support_selected_mask", []),
            length=len(local_xyz),
            name=f"node {node_id}.{slot}.local_support_selected_mask",
        )
        local_valid_value = diagnostic.get("local_support_nearest_cad_valid_mask")
        local_valid = (
            None
            if local_valid_value is None
            else _mask(
                local_valid_value,
                length=len(local_xyz),
                name=f"node {node_id}.{slot}.local_support_nearest_cad_valid_mask",
            )
        )
        if local_valid is not None and np.any(local_selected & ~local_valid):
            raise RenderPayloadError(
                f"node {node_id}.{slot} selected local support is nearest-CAD invalid"
            )
        ownership = _nullable_sheet_vector(
            diagnostic.get("local_sheet_ids", []),
            length=len(local_xyz),
            name=f"node {node_id}.{slot}.local_sheet_ids",
            valid_mask=local_valid,
            selected_mask=local_selected,
        )
        if local_valid is None:
            local_valid = np.isfinite(ownership)
        clusters = []
        for cluster_index, cluster in enumerate(diagnostic.get("height_clusters_mm", [])):
            values = np.asarray(cluster, dtype=np.float64)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise RenderPayloadError(f"node {node_id}.{slot} cluster {cluster_index} invalid")
            clusters.append(values)
        winner = diagnostic.get("winner_cluster_index")
        runner = diagnostic.get("runner_cluster_index")
        for label, value in (("winner", winner), ("runner", runner)):
            if value is not None and (int(value) < 0 or int(value) >= len(clusters)):
                raise RenderPayloadError(f"node {node_id}.{slot} {label} cluster index invalid")
        footprint = np.asarray(diagnostic.get("footprint_occupancy", []), dtype=bool)
        if footprint.shape != (2, 16):
            raise RenderPayloadError(f"node {node_id}.{slot} footprint must be 2x16")
        plane_coefficients = _nullable_vector(
            diagnostic.get("plane_coefficients"), length=3,
            name=f"node {node_id}.{slot}.plane_coefficients",
        )
        plane_p90 = _nullable_float(
            diagnostic.get("plane_p90_mm"),
            name=f"node {node_id}.{slot}.plane_p90_mm",
        )
        feature_pre = _nullable_vector(
            diagnostic.get("feature_pre_nuisance"), length=3,
            name=f"node {node_id}.{slot}.feature_pre_nuisance",
        )
        feature_post = _nullable_vector(
            diagnostic.get("feature_post_nuisance"), length=3,
            name=f"node {node_id}.{slot}.feature_post_nuisance",
        )
        required_finite = bool(
            np.all(np.isfinite(plane_coefficients))
            and math.isfinite(plane_p90)
            and np.all(np.isfinite(feature_pre))
            and np.all(np.isfinite(feature_post))
        )
        if diagnostic_valid and not required_finite:
            raise RenderPayloadError(
                f"node {node_id}.{slot} valid diagnostic requires finite plane, p90, and feature fields"
            )
        if diagnostic_valid and winner is None:
            raise RenderPayloadError(
                f"node {node_id}.{slot} valid diagnostic requires a winner cluster"
            )
        if not diagnostic_valid and required_finite:
            raise RenderPayloadError(
                f"node {node_id}.{slot} has all required fields and must be marked valid"
            )
        if not diagnostic_valid and winner is None:
            if np.any(local_selected):
                raise RenderPayloadError(
                    f"node {node_id}.{slot} invalid no-winner diagnostic cannot select local support"
                )
            if np.any(np.isfinite(plane_coefficients)) or math.isfinite(plane_p90):
                raise RenderPayloadError(
                    f"node {node_id}.{slot} invalid no-winner diagnostic requires null plane and p90"
                )
        parsed.append(
            {
                "slot": slot,
                "local_xyz": local_xyz,
                "local_valid": local_valid,
                "local_selected": local_selected,
                "local_sheet_ids": ownership,
                "clusters": clusters,
                "winner": None if winner is None else int(winner),
                "runner": None if runner is None else int(runner),
                "footprint": footprint,
                "valid": diagnostic_valid,
                "invalid_reason": invalid_reason,
                "plane_coefficients": plane_coefficients,
                "plane_p90_mm": plane_p90,
                "feature_pre": feature_pre,
                "feature_post": feature_post,
            }
        )
    amplitude = _nullable_float(
        node.get("amplitude_mm"), name=f"node {node_id}.amplitude_mm"
    )
    split_noise = _nullable_float(
        node.get("split_noise_mm"), name=f"node {node_id}.split_noise_mm"
    )
    threshold = _nullable_float(
        node.get("threshold_mm"), name=f"node {node_id}.threshold_mm"
    )
    node_fields_finite = all(math.isfinite(value) for value in (amplitude, split_noise, threshold))
    if node_valid and not node_fields_finite:
        raise RenderPayloadError(
            f"node {node_id} valid row requires finite amplitude, split noise, and threshold"
        )
    if not node_valid and node_fields_finite:
        raise RenderPayloadError(f"node {node_id} has all required fields and must be marked valid")
    if node_valid and not all(row["valid"] for row in parsed):
        raise RenderPayloadError(
            f"node {node_id} valid row requires two valid member diagnostics"
        )
    active = bool(node.get("active", False))
    if active and not node_valid:
        raise RenderPayloadError(f"node {node_id} active row must be valid")
    component_id = None if node.get("component_id") is None else str(node.get("component_id"))
    if not node_valid and component_id is not None:
        raise RenderPayloadError(f"node {node_id} invalid row cannot belong to a component")
    return {
        "node_id": node_id,
        "cad_index": cad_index,
        "valid": node_valid,
        "invalid_reason": node_invalid_reason,
        "amplitude": amplitude,
        "split_noise": split_noise,
        "threshold": threshold,
        "active": active,
        "component_id": component_id,
        "diagnostics": parsed,
    }


def _validate_component(component: Mapping[str, Any], node_ids: set[int]) -> dict[str, Any]:
    component_id = str(component["component_id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", component_id):
        raise RenderPayloadError(f"component ID is not filename-safe: {component_id!r}")
    ids = [int(value) for value in component.get("node_ids", [])]
    if len(ids) != len(set(ids)) or any(value not in node_ids for value in ids):
        raise RenderPayloadError(f"component {component_id} has invalid node_ids")
    energies = _nullable_vector(
        component.get("member_energy_mm2"), length=2,
        name=f"component {component_id}.member_energy_mm2",
    )
    variants = []
    for row in component.get("variant_rows", []):
        variants.append(
            {
                "row_id": str(row["row_id"]),
                "matched_component_id": None if row.get("matched_component_id") is None else str(row.get("matched_component_id")),
                "jaccard": _safe_float(row.get("jaccard"), name=f"component {component_id} variant jaccard"),
                "medoid_displacement_mm": _nullable_float(row.get("medoid_displacement_mm"), name=f"component {component_id} variant displacement"),
                "energy_ratio": _nullable_float(row.get("energy_ratio"), name=f"component {component_id} variant energy ratio"),
                "attributed_slot": None if row.get("attributed_slot") is None else str(row.get("attributed_slot")),
                "passed": bool(row.get("passed", False)),
            }
        )
    attributed_slot = component.get("attributed_slot")
    if attributed_slot not in (None, "FIRST", "SECOND"):
        raise RenderPayloadError(f"component {component_id} has invalid attributed_slot")
    return {
        "component_id": component_id,
        "node_ids": ids,
        "retained": bool(component.get("retained", False)),
        "robustness_changed": bool(component.get("robustness_changed", False)),
        "area_mm2": _safe_float(component.get("area_mm2", 0.0), name=f"component {component_id}.area_mm2"),
        "geodesic_span_mm": _safe_float(component.get("geodesic_span_mm", 0.0), name=f"component {component_id}.geodesic_span_mm"),
        "energies": energies,
        "energy_ratio": _nullable_float(component.get("energy_ratio"), name=f"component {component_id}.energy_ratio"),
        "attributed_slot": attributed_slot,
        "pairwise_energy_fraction": _nullable_float(component.get("pairwise_energy_fraction"), name=f"component {component_id}.pairwise_energy_fraction"),
        "variants": variants,
    }


def _validate_variant_member_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical_members: Sequence[Mapping[str, Any]],
    row_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(rows) != 2:
        raise RenderPayloadError(f"{row_id} member_support requires two rows")
    tokens_in_row_order = [str(row.get("opaque_token")) for row in rows]
    by_token = {str(row.get("opaque_token")): row for row in rows}
    canonical_tokens = [str(row["opaque_token"]) for row in canonical_members]
    if set(by_token) != set(canonical_tokens) or len(by_token) != 2:
        raise RenderPayloadError(f"{row_id} member_support opaque-token mismatch")
    output = []
    for ordinal, base in enumerate(canonical_members):
        slot = "FIRST" if ordinal == 0 else "SECOND"
        source = by_token[str(base["opaque_token"])]
        voxel = _point_rows(source.get("voxel_xyz_mm", []), name=f"{row_id}.{slot}.voxel_xyz_mm")
        voxel_valid = _mask(
            source.get("voxel_nearest_cad_valid_mask", []), length=len(voxel),
            name=f"{row_id}.{slot}.voxel_nearest_cad_valid_mask",
        )
        selected = _mask(
            source.get("voxel_selected_by_any_winning_cluster_mask", []),
            length=len(voxel),
            name=f"{row_id}.{slot}.voxel_selected_by_any_winning_cluster_mask",
        )
        if np.any(selected & ~voxel_valid):
            raise RenderPayloadError(f"{row_id}.{slot} selected voxel is nearest-CAD invalid")
        output.append(
            {
                **base,
                "slot": slot,
                "voxel": voxel,
                "voxel_valid": voxel_valid,
                "selected": selected,
                "regions": _vector(
                    source.get("voxel_region_ids", []), length=len(voxel),
                    name=f"{row_id}.{slot}.voxel_region_ids", dtype=np.int64,
                ),
                "sheets": _nullable_sheet_vector(
                    source.get("voxel_sheet_ids", []),
                    length=len(voxel),
                    name=f"{row_id}.{slot}.voxel_sheet_ids",
                    valid_mask=voxel_valid,
                    selected_mask=selected,
                ),
                "support_mode": "CENTRAL_RAW_CONTEXT_PLUS_ROW_SPECIFIC_VOXELS",
            }
        )
    return output, tokens_in_row_order


def _canonical_changed_diagnostics(
    row: Mapping[str, Any], *, canonical_tokens: Sequence[str]
) -> list[dict[str, Any]]:
    diagnostics = row.get("member_diagnostics", [])
    if len(diagnostics) != 2:
        raise RenderPayloadError("changed diagnostic requires two member rows")
    tokens = [str(row.get("first_token")), str(row.get("second_token"))]
    by_token = {token: dict(diagnostic) for token, diagnostic in zip(tokens, diagnostics, strict=True)}
    if set(by_token) != set(canonical_tokens) or len(by_token) != 2:
        raise RenderPayloadError("changed diagnostic opaque-token mismatch")
    output = []
    for ordinal, token in enumerate(canonical_tokens):
        diagnostic = by_token[token]
        diagnostic["slot"] = "FIRST" if ordinal == 0 else "SECOND"
        output.append(diagnostic)
    return output


def _feature_amplitude_from_diagnostics(diagnostics: Sequence[Mapping[str, Any]]) -> float | None:
    if len(diagnostics) != 2:
        return None
    first = diagnostics[0].get("feature_post_nuisance")
    second = diagnostics[1].get("feature_post_nuisance")
    if first is None or second is None:
        return None
    first_array = np.asarray([math.nan if value is None else float(value) for value in first])
    second_array = np.asarray([math.nan if value is None else float(value) for value in second])
    if first_array.shape != (3,) or second_array.shape != (3,):
        return None
    delta = first_array - second_array
    return float(np.linalg.norm(delta)) if np.all(np.isfinite(delta)) else None


def _derive_energy(
    node_ids: Sequence[int],
    diagnostics_by_node: Mapping[int, Sequence[Mapping[str, Any]]],
    cad_area: np.ndarray,
) -> tuple[list[float | None], float | None]:
    if not node_ids:
        return [None, None], None
    energies: list[float | None] = []
    denominator = math.fsum(float(cad_area[int(node_id)]) for node_id in node_ids)
    for member in range(2):
        terms = []
        for node_id in node_ids:
            diagnostic = diagnostics_by_node.get(int(node_id), [None, None])[member]
            if not isinstance(diagnostic, Mapping):
                return [None, None], None
            vector = diagnostic.get("feature_post_nuisance")
            if vector is None or any(value is None for value in vector):
                return [None, None], None
            values = np.asarray(vector, dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                return [None, None], None
            terms.append(float(cad_area[int(node_id)] * np.dot(values, values)))
        energies.append(math.fsum(terms) / denominator if denominator > 0.0 else None)
    if any(value is None for value in energies):
        return energies, None
    larger, smaller = max(energies), min(energies)
    return energies, float(larger / max(smaller, 1e-12))


def _changed_visual_components(
    robustness: Mapping[str, Any],
    *,
    canonical_members: Sequence[Mapping[str, Any]],
    central_nodes: Sequence[Mapping[str, Any]],
    central_components: Sequence[Mapping[str, Any]],
    cad: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if int(robustness.get("semantic_row_count", -1)) != 32:
        raise RenderPayloadError("robustness.semantic_row_count must be 32")
    rows = robustness.get("rows", [])
    if not isinstance(rows, list) or [row.get("row_id") for row in rows] != [f"R{index:02d}" for index in range(32)]:
        raise RenderPayloadError("robustness rows must be exact R00-R31 order")
    node_by_id = {int(row["node_id"]): row for row in central_nodes}
    central_by_id = {str(row["component_id"]): row for row in central_components}
    canonical_tokens = [str(row["opaque_token"]) for row in canonical_members]
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_row_change_reasons = {
        "ROW_STATUS_CHANGED",
        "COMPONENT_CARDINALITY_CHANGED",
        "COMPONENT_MATCH_INCOMPLETE",
        "COMPONENT_MATCH_DUPLICATE",
        "COMPONENT_UNMATCHED",
        "COMPONENT_MATCH_OUT_OF_RANGE",
        "COMPONENT_MATCH_ID_CHANGED",
        "COMPONENT_NODE_IDS_CHANGED",
        "COMPONENT_NODE_HASH_CHANGED",
        "COMPONENT_JACCARD_CHANGED",
        "COMPONENT_MEDOID_CHANGED",
        "COMPONENT_MEDOID_NODE_CHANGED",
        "COMPONENT_ATTRIBUTED_TOKEN_CHANGED",
        "ROW_ATTRIBUTED_TOKEN_CHANGED",
        "CENTRAL_EMPTY_VARIANT_COMPONENT_BIRTH",
    }
    for row in rows:
        row_id = str(row["row_id"])
        passed = row.get("passed")
        row_changed = row.get("row_changed")
        reasons = row.get("row_change_reasons")
        if not isinstance(passed, bool) or not isinstance(row_changed, bool):
            raise RenderPayloadError(f"{row_id} passed and row_changed must be booleans")
        if not isinstance(reasons, list) or any(not isinstance(value, str) for value in reasons):
            raise RenderPayloadError(f"{row_id} row_change_reasons must be a string list")
        if reasons != sorted(set(reasons)):
            raise RenderPayloadError(f"{row_id} row_change_reasons must be sorted and unique")
        unknown_reasons = sorted(set(reasons) - allowed_row_change_reasons)
        if unknown_reasons:
            raise RenderPayloadError(f"{row_id} unknown row_change_reasons: {unknown_reasons}")
        if row_changed != bool(reasons):
            raise RenderPayloadError(
                f"{row_id} row_changed must equal bool(row_change_reasons)"
            )
        changed_sources = row.get("changed_visual_components", [])
        local_sources = row.get("changed_component_local_diagnostics", [])
        member_support_sources = row.get("member_support", [])
        evidence_required = (not passed) or row_changed
        if not evidence_required:
            if changed_sources or local_sources or member_support_sources:
                raise RenderPayloadError(
                    f"{row_id} passed unchanged row carries changed visual evidence"
                )
            continue
        # A failed no-component row still has scientific value globally, but
        # it creates no local component obligation.
        matches = list(row.get("matches", []))
        if not changed_sources and not central_components:
            if local_sources or member_support_sources:
                raise RenderPayloadError(
                    f"{row_id} zero-component row must not carry orphan local evidence"
                )
            continue
        if not member_support_sources or not local_sources:
            raise RenderPayloadError(
                f"{row_id} failed-or-changed component row requires full local evidence"
            )
        render_members, tokens_in_row_order = _validate_variant_member_support(
            member_support_sources, canonical_members=canonical_members, row_id=row_id,
        )
        diagnostic_sources = {
            int(item["node_id"]): _canonical_changed_diagnostics(
                item, canonical_tokens=canonical_tokens
            )
            for item in local_sources
        }
        if len(diagnostic_sources) != len(local_sources):
            raise RenderPayloadError(f"{row_id} duplicate changed diagnostic node")

        matched_central_ids = {
            f"C{int(match['central_component_id']):04d}" for match in matches
        }
        for source in changed_sources:
            component_id = str(source["component_id"])
            if component_id in seen:
                raise RenderPayloadError(f"duplicate changed component {component_id}")
            seen.add(component_id)
            node_ids = [int(value) for value in source.get("node_ids", [])]
            if any(value not in diagnostic_sources for value in node_ids):
                raise RenderPayloadError(f"{component_id} lacks variant-local node diagnostics")
            raw_energies = source.get("member_energy_mm2")
            energy_by_token = {
                token: value
                for token, value in zip(tokens_in_row_order, raw_energies or [None, None], strict=True)
            }
            energies = [energy_by_token.get(token) for token in canonical_tokens]
            attributed_slot = _slot_for_token(
                source.get("attributed_opaque_member"), canonical_tokens
            )
            variant_numeric = int(component_id.rsplit("C", 1)[1])
            match = next(
                (
                    item for item in matches
                    if int(item.get("variant_component_id", -1)) == variant_numeric
                ),
                None,
            )
            variant_row = {
                "row_id": row_id,
                "matched_component_id": None if match is None else f"C{int(match['central_component_id']):04d}",
                "jaccard": 0.0 if match is None else match.get("area_weighted_jaccard"),
                "medoid_displacement_mm": None if match is None else match.get("medoid_graph_geodesic_displacement_mm"),
                "energy_ratio": source.get("energy_ratio"),
                "attributed_slot": attributed_slot,
                "passed": False if match is None else bool(match.get("passed", False)),
            }
            canonical_source = {
                "component_id": component_id,
                "node_ids": node_ids,
                "retained": False,
                "robustness_changed": True,
                "area_mm2": source.get("area_mm2"),
                "geodesic_span_mm": source.get("geodesic_span_mm"),
                "member_energy_mm2": energies,
                "energy_ratio": source.get("energy_ratio"),
                "attributed_slot": attributed_slot,
                "pairwise_energy_fraction": source.get("pairwise_energy_fraction"),
                "variant_rows": [variant_row],
            }
            component = _validate_component(canonical_source, set(node_by_id))
            component["source_row_id"] = row_id
            component["vanished_in_variant"] = False
            component["member_support"] = render_members
            overrides = []
            for node_id in node_ids:
                central = node_by_id[node_id]
                diagnostics = diagnostic_sources[node_id]
                amplitude = _feature_amplitude_from_diagnostics(diagnostics)
                override = _validate_node(
                    {
                        "node_id": node_id,
                        "cad_index": central["cad_index"],
                        "valid": amplitude is not None,
                        "invalid_reason": None if amplitude is not None else "VARIANT_LOCAL_FEATURE_UNAVAILABLE",
                        "amplitude_mm": amplitude,
                        "split_noise_mm": central["split_noise"] if math.isfinite(central["split_noise"]) else None,
                        "threshold_mm": central["threshold"] if math.isfinite(central["threshold"]) else None,
                        "active": True,
                        "component_id": component_id,
                        "member_diagnostics": diagnostics,
                    },
                    len(cad["xyz"]),
                )
                if not override["valid"]:
                    raise RenderPayloadError(f"active changed component {component_id} has invalid node")
                overrides.append(override)
            component["node_overrides"] = overrides
            output.append(component)

        for central_id, central_component in central_by_id.items():
            if central_id in matched_central_ids:
                continue
            component_id = f"V_{row_id}_VANISHED_{central_id}"
            if component_id in seen:
                raise RenderPayloadError(f"duplicate vanished component {component_id}")
            seen.add(component_id)
            node_ids = list(central_component["node_ids"])
            if any(value not in diagnostic_sources for value in node_ids):
                raise RenderPayloadError(f"{component_id} lacks vanished-node diagnostics")
            energies, ratio = _derive_energy(node_ids, diagnostic_sources, cad["area"])
            canonical_source = {
                "component_id": component_id,
                "node_ids": node_ids,
                "retained": False,
                "robustness_changed": True,
                "area_mm2": central_component["area_mm2"],
                "geodesic_span_mm": central_component["geodesic_span_mm"],
                "member_energy_mm2": energies,
                "energy_ratio": ratio,
                "attributed_slot": None,
                "pairwise_energy_fraction": None,
                "variant_rows": [
                    {
                        "row_id": row_id,
                        "matched_component_id": None,
                        "jaccard": 0.0,
                        "medoid_displacement_mm": None,
                        "energy_ratio": ratio,
                        "attributed_slot": None,
                        "passed": False,
                    }
                ],
            }
            component = _validate_component(canonical_source, set(node_by_id))
            component["source_row_id"] = row_id
            component["vanished_in_variant"] = True
            component["member_support"] = render_members
            overrides = []
            for node_id in node_ids:
                central = node_by_id[node_id]
                diagnostics = diagnostic_sources[node_id]
                amplitude = _feature_amplitude_from_diagnostics(diagnostics)
                overrides.append(
                    _validate_node(
                        {
                            "node_id": node_id,
                            "cad_index": central["cad_index"],
                            "valid": amplitude is not None,
                            "invalid_reason": None if amplitude is not None else "VANISHED_VARIANT_LOCAL_FEATURE_UNAVAILABLE",
                            "amplitude_mm": amplitude,
                            "split_noise_mm": central["split_noise"] if math.isfinite(central["split_noise"]) else None,
                            "threshold_mm": central["threshold"] if math.isfinite(central["threshold"]) else None,
                            "active": False,
                            "component_id": component_id,
                            "member_diagnostics": diagnostics,
                        },
                        len(cad["xyz"]),
                    )
                )
            component["node_overrides"] = overrides
            output.append(component)
    return sorted(output, key=lambda item: (str(item["source_row_id"]), str(item["component_id"])))


def _slot_for_token(token: Any, opaque_tokens: Sequence[str]) -> str | None:
    if token is None:
        return None
    text = str(token)
    if text in ("FIRST", "SECOND"):
        return text
    if text == opaque_tokens[0]:
        return "FIRST"
    if text == opaque_tokens[1]:
        return "SECOND"
    return None


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the render-only schema without external reads."""

    if payload.get("schema") == LEGACY_EXP29_SCHEMA:
        raise RenderPayloadError(
            "EXP_CAD_0029 visual payload is source-free superseded and is not accepted by EXP_CAD_0030"
        )
    if payload.get("schema") != SCHEMA:
        raise RenderPayloadError(f"schema must be {SCHEMA}")
    expected_top_level = {
        "schema", "pair_id", "members", "cad", "global", "nodes", "components", "robustness"
    }
    if set(payload) != expected_top_level:
        raise RenderPayloadError(
            "payload top-level keys must be exactly " + ",".join(sorted(expected_top_level))
        )
    if not isinstance(payload.get("global"), Mapping) or not isinstance(payload.get("robustness"), Mapping):
        raise RenderPayloadError("global and robustness must be objects")
    # If the producer includes provenance flags in the global diagnostic they
    # must remain false.  The renderer itself is render-only and performs no
    # source reads, so their absence is not treated as evidence either way.
    for key in (
        "evaluator_manifest_accessed", "evaluator_roles_opened", "labels_opened",
        "gold_opened", "expected_counts_opened", "quality_identity_opened",
    ):
        if key in payload["global"] and payload["global"][key] is not False:
            raise RenderPayloadError(f"global.{key} must be false when present")
    if len(payload.get("members", [])) != 2:
        raise RenderPayloadError("payload requires exactly two opaque members")
    members = [_validate_member(row, ordinal) for ordinal, row in enumerate(payload["members"])]
    cad = _validate_cad(payload.get("cad", {}))
    nodes = [_validate_node(row, len(cad["xyz"])) for row in payload.get("nodes", [])]
    ids = [row["node_id"] for row in nodes]
    expected_ids = list(range(len(cad["xyz"])))
    if ids != expected_ids:
        raise RenderPayloadError(
            "nodes must cover every CAD node exactly once in node_id range(N) order"
        )
    if [row["cad_index"] for row in nodes] != expected_ids:
        raise RenderPayloadError("every node must satisfy cad_index == node_id")
    if not np.array_equal(
        np.asarray([row["active"] for row in nodes], dtype=bool), cad["active"]
    ):
        raise RenderPayloadError("node active flags must exactly equal cad.active_mask")
    components = [_validate_component(row, set(ids)) for row in payload.get("components", [])]
    component_ids = [row["component_id"] for row in components]
    if component_ids != sorted(component_ids) or len(component_ids) != len(set(component_ids)):
        raise RenderPayloadError("components must have unique lexicographically ascending IDs")
    known = set(component_ids)
    for node in nodes:
        if node["component_id"] is not None and node["component_id"] not in known:
            raise RenderPayloadError(f"node {node['node_id']} references unknown component")
    changed_components = _changed_visual_components(
        payload["robustness"],
        canonical_members=members,
        central_nodes=nodes,
        central_components=components,
        cad=cad,
    )
    return {
        "pair_id": str(payload.get("pair_id", "OPAQUE_PAIR")),
        "members": members,
        "cad": cad,
        "nodes": nodes,
        "components": components,
        "changed_components": changed_components,
        "global": dict(payload["global"]),
        "robustness": dict(payload["robustness"]),
        "camera_basis": _camera_basis(cad["xyz"], payload.get("camera_basis_rows")),
    }


def select_visual_inventory(data: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Return the frozen local-pack inventory in deterministic order."""

    central_components = [
        row
        for row in data["components"]
        if bool(row["retained"]) or bool(row["robustness_changed"])
    ]
    components = central_components + list(data.get("changed_components", []))
    nodes = sorted(
        (row for row in data["nodes"] if math.isfinite(float(row["amplitude"]))),
        key=lambda row: (-float(row["amplitude"]), int(row["node_id"]))
    )[:TOP_NODE_COUNT]
    return components, nodes


def _figure(rows: int = 4, columns: int = 4) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(rows, columns, figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, facecolor=BACKGROUND)
    axes_array = np.asarray(axes, dtype=object).reshape(rows, columns)
    return figure, axes_array


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=DPI,
        facecolor=BACKGROUND,
        edgecolor=BACKGROUND,
        metadata={"Software": "ShermanQC EXP30 deterministic renderer"},
    )
    plt.close(figure)
    with Image.open(path) as image:
        if image.size != (WIDTH_PX, HEIGHT_PX):
            raise RuntimeError(f"native visual has wrong dimensions: {image.size}")


def _same_camera_limits(projected: Sequence[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    nonempty = [row for row in projected if len(row)]
    if not nonempty:
        return (-1.0, 1.0), (-1.0, 1.0)
    combined = np.concatenate(nonempty, axis=0)
    low = np.min(combined[:, :2], axis=0)
    high = np.max(combined[:, :2], axis=0)
    span = np.maximum(high - low, 1e-6)
    padding = 0.04 * span
    return (float(low[0] - padding[0]), float(high[0] + padding[0])), (float(low[1] - padding[1]), float(high[1] + padding[1]))


def _scatter_support(axis: plt.Axes, member: Mapping[str, Any], raw_projected: np.ndarray, voxel_projected: np.ndarray, limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    raw_ids = _downsample_indices(len(raw_projected))
    raw_valid = member["raw_valid"][raw_ids]
    if len(raw_ids):
        axis.scatter(raw_projected[raw_ids[~raw_valid], 0], raw_projected[raw_ids[~raw_valid], 1], s=0.25, color=REJECTED, alpha=0.12, linewidths=0)
        axis.scatter(raw_projected[raw_ids[raw_valid], 0], raw_projected[raw_ids[raw_valid], 1], s=0.3, color=FIRST, alpha=0.14, linewidths=0)
    invalid = ~member["voxel_valid"]
    valid_not_selected = member["voxel_valid"] & ~member["selected"]
    axis.scatter(voxel_projected[invalid, 0], voxel_projected[invalid, 1], s=3.0, color=REJECTED, marker="x", alpha=0.38)
    axis.scatter(voxel_projected[valid_not_selected, 0], voxel_projected[valid_not_selected, 1], s=2.8, color=MUTED, alpha=0.34, linewidths=0)
    axis.scatter(voxel_projected[member["selected"], 0], voxel_projected[member["selected"], 1], s=3.2, color=SELECTED, alpha=0.78, linewidths=0)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")


def _signed_feature_heatmap(
    axis: plt.Axes, nodes: Sequence[Mapping[str, Any]], *, title: str
) -> None:
    _configure_axis(axis, title)
    if not nodes:
        _empty(axis, "NO FINITE SIGNED FEATURES")
        return
    matrix = np.vstack(
        [
            np.concatenate([row["diagnostics"][member][stage] for row in nodes])
            for member, stage in (
                (0, "feature_pre"), (0, "feature_post"),
                (1, "feature_pre"), (1, "feature_post"),
            )
        ]
    )
    finite = np.abs(matrix[np.isfinite(matrix)])
    scale = max(float(np.max(finite)) if len(finite) else 1.0, 1e-9)
    axis.imshow(
        matrix, cmap="coolwarm", vmin=-scale, vmax=scale,
        aspect="auto", interpolation="nearest",
    )
    axis.set_yticks(
        np.arange(4), ["FIRST pre", "FIRST post", "SECOND pre", "SECOND post"],
        fontsize=6,
    )
    centers = np.arange(len(nodes)) * 3 + 1
    axis.set_xticks(centers, [str(row["node_id"]) for row in nodes], fontsize=5)
    axis.set_xlabel("node blocks; each block is signed [h, 3g1, 3g2]", color=TEXT, fontsize=6)
    axis.text(
        0.99, 0.02, f"shared signed scale ±{scale:.4g} mm",
        transform=axis.transAxes, ha="right", va="bottom", color=TEXT, fontsize=6,
        bbox={"facecolor": PANEL, "edgecolor": GRID, "alpha": 0.75},
    )


def _render_global(path: Path, data: Mapping[str, Any]) -> None:
    figure, axes = _figure()
    cad = data["cad"]
    basis = data["camera_basis"]
    center = np.average(cad["xyz"], axis=0, weights=np.maximum(cad["area"], 1e-15)) if len(cad["xyz"]) else np.zeros(3)
    cad_p = _project(cad["xyz"], center, basis)
    raw_p = [_project(row["raw"], center, basis) for row in data["members"]]
    voxel_p = [_project(row["voxel"], center, basis) for row in data["members"]]
    primary = [cad_p, *raw_p, *voxel_p]
    depth = [row[:, [0, 2, 1]] for row in primary]
    primary_limits = _same_camera_limits(primary)
    depth_limits = _same_camera_limits(depth)

    for view_ordinal, (reorder, limits, view_label) in enumerate(
        (([0, 1, 2], primary_limits, "PCA1-PCA2"), ([0, 2, 1], depth_limits, "PCA1-PCA3 depth"))
    ):
        for member_ordinal, member in enumerate(data["members"]):
            axis = axes[0, 2 * view_ordinal + member_ordinal]
            _configure_axis(
                axis,
                f"{member['opaque_token']} raw + voxel — {view_label} (fixed paired camera)",
            )
            _scatter_support(
                axis,
                member,
                raw_p[member_ordinal][:, reorder],
                voxel_p[member_ordinal][:, reorder],
                limits,
            )
            if view_ordinal == 1:
                axis.text(
                    0.01, 0.02, "equal mm axes; no depth exaggeration",
                    transform=axis.transAxes, color=TEXT, fontsize=5,
                    bbox={"facecolor": PANEL, "edgecolor": GRID, "alpha": 0.72},
                )
            axis.legend(
                handles=[
                    Patch(color=FIRST, label="raw nearest-CAD valid"),
                    Patch(color=REJECTED, label="nearest-CAD invalid"),
                    Patch(color=MUTED, label="voxel valid, not a winning cluster"),
                    Patch(color=SELECTED, label="voxel in ≥1 winning cluster"),
                ],
                loc="lower right", fontsize=6, facecolor=PANEL, labelcolor=TEXT,
            )

    limits = primary_limits
    axis = axes[1, 0]
    _configure_axis(axis, "CAD interior / common support / exclusions")
    colors = np.tile(np.asarray([0.35, 0.40, 0.45, 0.35]), (len(cad_p), 1))
    colors[cad["interior"]] = np.asarray([0.30, 0.65, 0.86, 0.55])
    colors[cad["common"]] = np.asarray([0.40, 0.90, 0.44, 0.88])
    colors[cad["excluded"]] = np.asarray([0.94, 0.28, 0.42, 0.88])
    axis.scatter(cad_p[:, 0], cad_p[:, 1], c=colors, s=4.0, linewidths=0)
    axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_aspect("equal", adjustable="box")
    axis.legend(
        handles=[
            Patch(color="#4da6db", label="interior"),
            Patch(color=SELECTED, label="common support"),
            Patch(color=REJECTED, label="excluded"),
        ], fontsize=6, loc="lower right", facecolor=PANEL, labelcolor=TEXT,
    )

    axis = axes[1, 1]
    _configure_axis(axis, "Smooth-region and selected/rejected sheet ownership")
    axis.scatter(cad_p[:, 0], cad_p[:, 1], c=_discrete_colors(cad["regions"]), s=5.0, linewidths=0)
    for ordinal, member in enumerate(data["members"]):
        chosen = member["selected"]
        invalid = ~member["voxel_valid"]
        valid_not_selected = member["voxel_valid"] & ~chosen
        axis.scatter(voxel_p[ordinal][invalid, 0], voxel_p[ordinal][invalid, 1], color=REJECTED, marker="x", s=4.0, alpha=0.28)
        axis.scatter(voxel_p[ordinal][valid_not_selected, 0], voxel_p[ordinal][valid_not_selected, 1], color=MUTED, s=2.0, alpha=0.22, linewidths=0)
        if np.any(chosen):
            sheet_colors = _discrete_colors(member["sheets"][chosen] + 2 * ordinal)
            axis.scatter(voxel_p[ordinal][chosen, 0], voxel_p[ordinal][chosen, 1], c=sheet_colors, s=2.2, alpha=0.35, linewidths=0)
    axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_aspect("equal", adjustable="box")
    axis.legend(
        handles=[
            Patch(color="#45d7ff", label="smooth/sheet owner color"),
            Patch(color=MUTED, label="valid voxel, not winning"),
            Patch(color=REJECTED, label="nearest-CAD invalid voxel ×"),
        ], fontsize=6, loc="lower right", facecolor=PANEL, labelcolor=TEXT,
    )

    axis = axes[1, 2]
    _configure_axis(axis, f"Frozen folds + strict > {FOLD_HALO_MM:.12f} mm halo")
    axis.scatter(cad_p[:, 0], cad_p[:, 1], c=_discrete_colors(cad["folds"]), s=4.0, linewidths=0)
    axis.scatter(cad_p[cad["halo"], 0], cad_p[cad["halo"], 1], facecolors="none", edgecolors=HALO, s=11.0, linewidths=0.45)
    axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_aspect("equal", adjustable="box")

    nodes = data["nodes"]
    node_p = cad_p[[row["cad_index"] for row in nodes]] if nodes else np.empty((0, 3))
    amplitudes = np.asarray([row["amplitude"] for row in nodes])
    thresholds = np.asarray([row["threshold"] for row in nodes])
    noises = np.asarray([row["split_noise"] for row in nodes])
    active = np.asarray([row["active"] for row in nodes], dtype=bool)

    axis = axes[1, 3]
    _configure_axis(axis, "Central amplitude and frozen threshold T")
    if nodes:
        finite_scale = np.r_[amplitudes[np.isfinite(amplitudes)], thresholds[np.isfinite(thresholds)]]
        scale = max(float(np.max(finite_scale)) if len(finite_scale) else 1.0, 1e-9)
        finite_amplitude = np.isfinite(amplitudes)
        axis.scatter(node_p[finite_amplitude, 0], node_p[finite_amplitude, 1], c=amplitudes[finite_amplitude], cmap="magma", norm=Normalize(0.0, scale), s=6.0, linewidths=0)
        axis.scatter(node_p[~finite_amplitude, 0], node_p[~finite_amplitude, 1], color=MUTED, marker="x", s=5.0, alpha=0.45)
        axis.scatter(node_p[active, 0], node_p[active, 1], facecolors="none", edgecolors=ACTIVE, s=20.0, linewidths=0.7)
        axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_aspect("equal", adjustable="box")
        axis.legend(
            handles=[
                Patch(color="#d04a8d", label="finite amplitude (shared scale)"),
                Patch(color=MUTED, label="invalid/null amplitude ×"),
                Patch(facecolor="none", edgecolor=ACTIVE, label="active node ring"),
            ], fontsize=5, loc="lower right", facecolor=PANEL, labelcolor=TEXT,
        )
    else:
        _empty(axis, "NO CENTRAL NODES")

    axis = axes[2, 0]
    _configure_axis(axis, "Split noise vs frozen T (strict amplitude > T)")
    if nodes:
        node_ids = np.arange(len(nodes))
        axis.plot(node_ids, noises, color=HALO, linewidth=0.8, label="split noise")
        axis.plot(node_ids, thresholds, color=REJECTED, linewidth=0.8, label="T")
        axis.scatter(node_ids, amplitudes, c=np.where(active, ACTIVE, MUTED), s=5.0, label="amplitude")
        axis.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT)
    else:
        _empty(axis, "NO CENTRAL NODES")

    axis = axes[2, 1]
    _configure_axis(axis, "Active induced graph + frozen components")
    if nodes:
        cad_to_local = {row["cad_index"]: index for index, row in enumerate(nodes)}
        for left, right in cad["edges"]:
            if int(left) in cad_to_local and int(right) in cad_to_local:
                li, ri = cad_to_local[int(left)], cad_to_local[int(right)]
                if active[li] and active[ri]:
                    axis.plot(node_p[[li, ri], 0], node_p[[li, ri], 1], color="#c8d6e5", alpha=0.35, linewidth=0.45)
        component_index = {row["component_id"]: i for i, row in enumerate(data["components"])}
        node_components = np.asarray([component_index.get(row["component_id"], -1) for row in nodes])
        component_colors = _discrete_colors(np.maximum(node_components, 0))
        component_colors[node_components < 0] = np.asarray([0.45, 0.48, 0.52, 0.35])
        axis.scatter(node_p[:, 0], node_p[:, 1], c=component_colors, s=np.where(active, 12.0, 3.0), linewidths=0)
        axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.set_aspect("equal", adjustable="box")
    else:
        _empty(axis, "NO ACTIVE GRAPH\nterminal visual abstention is still auditable")

    components = data["components"]
    axis = axes[2, 2]
    _configure_axis(axis, "Corrected member energies / ratio / attribution")
    if components:
        x = np.arange(len(components))
        width = 0.34
        axis.bar(x - width / 2, [row["energies"][0] for row in components], width, color=FIRST, label=data["members"][0]["opaque_token"])
        axis.bar(x + width / 2, [row["energies"][1] for row in components], width, color=SECOND, label=data["members"][1]["opaque_token"])
        axis.set_xticks(
            x,
            [
                f"{row['component_id']}\nR={_format_optional(row['energy_ratio'], precision=2)}\n"
                f"{row['attributed_slot'] or 'ABSTAIN'}"
                for row in components
            ],
            fontsize=6,
        )
        axis.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT)
    else:
        _empty(axis, "ZERO RETAINED OR CHANGED COMPONENTS\nNO ATTRIBUTION")

    axis = axes[2, 3]
    _configure_axis(axis, "Variant persistence and component matching")
    variant_rows = [(row["component_id"], variant) for row in components for variant in row["variants"]]
    if variant_rows:
        x = np.arange(len(variant_rows))
        values = [variant["jaccard"] for _, variant in variant_rows]
        colors = [SELECTED if variant["passed"] else REJECTED for _, variant in variant_rows]
        axis.bar(x, values, color=colors, width=0.8)
        axis.axhline(0.6, color=HALO, linestyle="--", linewidth=0.8)
        axis.set_xticks(x, [f"{cid}:{variant['row_id']}" for cid, variant in variant_rows], rotation=90, fontsize=4)
        axis.set_ylim(0.0, 1.05)
    else:
        _empty(axis, "NO VARIANT COMPONENT MATCHES")

    top = sorted(
        (row for row in nodes if math.isfinite(float(row["amplitude"]))),
        key=lambda row: (-row["amplitude"], row["node_id"]),
    )[:TOP_NODE_COUNT]
    axis = axes[3, 0]
    _signed_feature_heatmap(
        axis, top,
        title="Signed f=[h,3g1,3g2] pre/post nuisance — fixed top nodes",
    )

    axis = axes[3, 1]
    _configure_axis(axis, "16 x 2 footprint occupancy — fixed top nodes")
    if top:
        matrix = np.concatenate(
            [np.concatenate([row["diagnostics"][0]["footprint"], row["diagnostics"][1]["footprint"]], axis=0) for row in top],
            axis=0,
        )
        axis.imshow(matrix.astype(float), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        axis.set_xlabel("16 angular sectors", color=TEXT, fontsize=7)
        axis.set_ylabel("node × member × radial band", color=TEXT, fontsize=7)
    else:
        _empty(axis, "NO FOOTPRINTS")

    axis = axes[3, 2]
    _configure_axis(axis, "All height clusters with winner / runner scores")
    if top:
        y = 0
        ticks = []
        labels = []
        for row in top:
            for diagnostic in row["diagnostics"]:
                for cluster_index, cluster in enumerate(diagnostic["clusters"]):
                    marker = "o" if cluster_index == diagnostic["winner"] else "^" if cluster_index == diagnostic["runner"] else "."
                    color = SELECTED if cluster_index == diagnostic["winner"] else HALO if cluster_index == diagnostic["runner"] else MUTED
                    axis.scatter(cluster, np.full(len(cluster), y), marker=marker, color=color, s=8.0, alpha=0.75)
                ticks.append(y); labels.append(f"{row['node_id']}:{diagnostic['slot'][0]}")
                y += 1
        axis.set_yticks(ticks, labels, fontsize=5)
        axis.axvline(0.0, color=MUTED, linewidth=0.6)
        axis.set_xlabel("signed height (mm)", color=TEXT, fontsize=7)
    else:
        _empty(axis, "NO HEIGHT CLUSTERS")

    axis = axes[3, 3]
    _configure_axis(axis, "Plane fit p90 (limit 0.3 mm)")
    if top:
        x = np.arange(len(top))
        width = 0.34
        axis.bar(x - width / 2, [row["diagnostics"][0]["plane_p90_mm"] for row in top], width, color=FIRST)
        axis.bar(x + width / 2, [row["diagnostics"][1]["plane_p90_mm"] for row in top], width, color=SECOND)
        axis.axhline(0.3, color=REJECTED, linestyle="--", linewidth=0.8)
        axis.set_xticks(x, [str(row["node_id"]) for row in top], fontsize=6)
    else:
        _empty(axis, "NO PLANE FITS")

    status = str(data["global"].get("status", "SOURCE_ONLY_DIAGNOSTIC"))
    figure.suptitle(
        f"EXP_CAD_0030 CF-PDG V3 — {data['pair_id']} — global paired native audit — {status}",
        color=TEXT, fontsize=18, y=0.995,
    )
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.04, top=0.965, wspace=0.24, hspace=0.34)
    _save(figure, path)


def _node_axes(node: Mapping[str, Any], cad: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = node["cad_index"]
    # Plane coefficients and signed gradient features are defined in this
    # frozen producer basis. Reconstructing a convenient tangent basis here
    # would rotate g1/g2 and make an otherwise attractive visual false.
    return (
        np.asarray(cad["tangent_e1"][index], dtype=np.float64),
        np.asarray(cad["tangent_e2"][index], dtype=np.float64),
        np.asarray(cad["normals"][index], dtype=np.float64),
    )


def _render_node(path: Path, data: Mapping[str, Any], node: Mapping[str, Any]) -> None:
    figure, axes = _figure(3, 4)
    cad = data["cad"]
    origin = cad["xyz"][node["cad_index"]]
    tangent1, tangent2, normal = _node_axes(node, cad)
    local_basis = np.stack([tangent1, tangent2, normal], axis=0)

    projected_local = [
        _project(diagnostic["local_xyz"], origin, local_basis)
        for diagnostic in node["diagnostics"]
    ]
    local_xy_limits = _same_camera_limits(projected_local)
    plane_pairs: list[np.ndarray] = []
    for local, diagnostic in zip(projected_local, node["diagnostics"], strict=True):
        selected = diagnostic["local_selected"]
        coefficients = diagnostic["plane_coefficients"]
        if np.any(selected) and np.all(np.isfinite(coefficients)):
            rows = local[selected]
            fitted = coefficients[0] + coefficients[1] * rows[:, 0] + coefficients[2] * rows[:, 1]
            plane_pairs.append(np.column_stack((fitted, rows[:, 2])))
        else:
            plane_pairs.append(np.empty((0, 2), dtype=np.float64))
    nonempty_plane_pairs = [row for row in plane_pairs if len(row)]
    if nonempty_plane_pairs:
        values = np.concatenate(nonempty_plane_pairs, axis=0).reshape(-1)
        low = float(np.min(values)); high = float(np.max(values))
        padding = max((high - low) * 0.06, 1e-3)
        plane_limits = (low - padding, high + padding)
    else:
        plane_limits = (-1.0, 1.0)

    for ordinal, diagnostic in enumerate(node["diagnostics"]):
        token = data["members"][ordinal]["opaque_token"]
        local = projected_local[ordinal]
        axis = axes[0, ordinal]
        _configure_axis(axis, f"{token} local raw/voxel selected-rejected + sheet ownership")
        selected = diagnostic["local_selected"]
        local_valid = diagnostic["local_valid"]
        invalid = ~local_valid
        valid_rejected = local_valid & ~selected
        axis.scatter(
            local[invalid, 0], local[invalid, 1], color=REJECTED,
            marker="x", s=7.0, alpha=0.45,
        )
        axis.scatter(
            local[valid_rejected, 0], local[valid_rejected, 1], color=MUTED,
            s=5.0, alpha=0.45, linewidths=0,
        )
        if np.any(selected):
            axis.scatter(local[selected, 0], local[selected, 1], c=_discrete_colors(diagnostic["local_sheet_ids"][selected] + ordinal * 2), s=6.0, alpha=0.75, linewidths=0)
        axis.axhline(0.0, color=MUTED, linewidth=0.6); axis.axvline(0.0, color=MUTED, linewidth=0.6)
        axis.set_xlim(*local_xy_limits[0]); axis.set_ylim(*local_xy_limits[1])
        axis.set_aspect("equal", adjustable="box")
        axis.legend(
            handles=[
                Patch(color=REJECTED, label="nearest-CAD invalid; no sheet owner"),
                Patch(color=MUTED, label="valid, not winning"),
                Patch(color=SELECTED, label="winning support, ±1 sheet owner"),
            ],
            loc="lower right", fontsize=5, facecolor=PANEL, labelcolor=TEXT,
        )

        axis = axes[0, ordinal + 2]
        _configure_axis(axis, f"{token} all height clusters — winner ○ runner △")
        if diagnostic["clusters"]:
            for cluster_index, cluster in enumerate(diagnostic["clusters"]):
                marker = "o" if cluster_index == diagnostic["winner"] else "^" if cluster_index == diagnostic["runner"] else "."
                color = SELECTED if cluster_index == diagnostic["winner"] else HALO if cluster_index == diagnostic["runner"] else MUTED
                axis.scatter(cluster, np.full(len(cluster), cluster_index), marker=marker, color=color, s=18.0, alpha=0.8)
                axis.text(float(np.max(cluster)) if len(cluster) else 0.0, cluster_index, f" med|h|={abs(float(np.median(cluster))):.3f}", color=TEXT, fontsize=6, va="center")
            axis.axvline(0.0, color=MUTED, linewidth=0.6)
        else:
            _empty(axis, "NO ELIGIBLE HEIGHT CLUSTERS")

    for ordinal, diagnostic in enumerate(node["diagnostics"]):
        token = data["members"][ordinal]["opaque_token"]
        axis = axes[1, ordinal]
        _configure_axis(axis, f"{token} 16 x 2 footprint occupancy")
        axis.imshow(diagnostic["footprint"].astype(float), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        axis.set_yticks([0, 1], ["inner", "outer"], fontsize=7)
        axis.set_xticks(np.arange(16), [str(i) for i in range(16)], fontsize=5)

        axis = axes[1, ordinal + 2]
        _configure_axis(axis, f"{token} observed h vs fitted a+b·u+c·v + p90")
        pair = plane_pairs[ordinal]
        if len(pair):
            axis.scatter(
                pair[:, 0], pair[:, 1], color=FIRST if ordinal == 0 else SECOND,
                s=5.0, alpha=0.45,
            )
            axis.plot(plane_limits, plane_limits, color=SELECTED, linewidth=1.1)
        else:
            _empty(axis, "FULL PLANE FIT UNAVAILABLE")
        axis.set_xlim(*plane_limits); axis.set_ylim(*plane_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("fitted h (mm)", color=TEXT, fontsize=7)
        axis.set_ylabel("observed h (mm)", color=TEXT, fontsize=7)
        axis.text(
            0.02, 0.96,
            f"p90={_format_optional(diagnostic['plane_p90_mm'])} mm\nlimit=0.3000 mm",
            transform=axis.transAxes, va="top", color=TEXT, fontsize=8,
        )

    axis = axes[2, 0]
    _configure_axis(axis, "Signed f=[h,3g1,3g2] pre/post nuisance")
    positions = np.arange(3)
    width = 0.18
    for ordinal, diagnostic in enumerate(node["diagnostics"]):
        color = FIRST if ordinal == 0 else SECOND
        offset = (-1.5 + 2 * ordinal) * width
        axis.bar(positions + offset, diagnostic["feature_pre"], width, color=color, alpha=0.38, hatch="//", label=f"{diagnostic['slot']} pre")
        axis.bar(positions + offset + width, diagnostic["feature_post"], width, color=color, alpha=0.9, label=f"{diagnostic['slot']} post")
    axis.axhline(0.0, color=MUTED, linewidth=0.7)
    axis.set_xticks(positions, ["h", "3g1", "3g2"])
    axis.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT)

    axis = axes[2, 1]
    _configure_axis(axis, "Split noise / frozen T / central amplitude")
    values = [node["split_noise"], node["threshold"], node["amplitude"]]
    axis.bar(np.arange(3), values, color=[HALO, REJECTED, ACTIVE])
    axis.set_xticks(np.arange(3), ["split noise", "T", "amplitude"], fontsize=7)
    axis.text(0.03, 0.96, f"active={node['active']}\nstrict amp>T = {node['amplitude'] > node['threshold']}", transform=axis.transAxes, va="top", color=TEXT, fontsize=8)

    component = next((row for row in data["components"] if row["component_id"] == node["component_id"]), None)
    axis = axes[2, 2]
    _configure_axis(axis, "Active induced graph / fold / 6.408 mm halo context")
    center = np.average(cad["xyz"], axis=0, weights=np.maximum(cad["area"], 1e-15))
    projected = _project(cad["xyz"], center, data["camera_basis"])
    axis.scatter(projected[:, 0], projected[:, 1], c=_discrete_colors(cad["folds"]), s=3.0, alpha=0.45, linewidths=0)
    axis.scatter(projected[cad["halo"], 0], projected[cad["halo"], 1], facecolors="none", edgecolors=HALO, s=8.0, linewidths=0.35)
    axis.scatter(projected[node["cad_index"], 0], projected[node["cad_index"], 1], color=ACTIVE, s=60.0, marker="*", edgecolors="white", linewidths=0.5)
    if component:
        node_map = {row["node_id"]: row for row in data["nodes"]}
        indices = [node_map[value]["cad_index"] for value in component["node_ids"]]
        axis.scatter(projected[indices, 0], projected[indices, 1], facecolors="none", edgecolors=SELECTED, s=20.0, linewidths=0.8)
    axis.set_aspect("equal", adjustable="box")

    axis = axes[2, 3]
    _configure_axis(axis, "Corrected component energies / ratio / attribution")
    if component:
        axis.bar([0, 1], component["energies"], color=[FIRST, SECOND])
        axis.set_xticks([0, 1], [data["members"][0]["opaque_token"], data["members"][1]["opaque_token"]])
        axis.text(
            0.03, 0.96,
            f"component={component['component_id']}\nratio={_format_optional(component['energy_ratio'])}\n"
            f"attribution={component['attributed_slot'] or 'ABSTAIN'}\n"
            f"pair energy fraction={_format_optional(component['pairwise_energy_fraction'])}",
            transform=axis.transAxes, va="top", color=TEXT, fontsize=8,
        )
    else:
        _empty(axis, "NODE IS NOT IN A FROZEN COMPONENT\nNO LATE ATTRIBUTION")

    figure.suptitle(
        f"EXP_CAD_0030 local node pack — node {node['node_id']} — component {node['component_id'] or 'none'}",
        color=TEXT, fontsize=18, y=0.995,
    )
    figure.subplots_adjust(left=0.045, right=0.985, bottom=0.055, top=0.955, wspace=0.26, hspace=0.34)
    _save(figure, path)


def _render_component(path: Path, data: Mapping[str, Any], component: Mapping[str, Any]) -> None:
    figure, axes = _figure(4, 4)
    cad = data["cad"]
    nodes_by_id = {row["node_id"]: row for row in data["nodes"]}
    override_by_id = {
        row["node_id"]: row for row in component.get("node_overrides", [])
    }
    members = [override_by_id.get(node_id, nodes_by_id[node_id]) for node_id in component["node_ids"]]
    support_members = component.get("member_support", data["members"])
    indices = [row["cad_index"] for row in members]
    center = np.average(cad["xyz"], axis=0, weights=np.maximum(cad["area"], 1e-15))
    projected = _project(cad["xyz"], center, data["camera_basis"])

    raw_projected = [
        _project(member["raw"], center, data["camera_basis"])
        for member in support_members
    ]
    voxel_projected = [
        _project(member["voxel"], center, data["camera_basis"])
        for member in support_members
    ]
    if not indices:
        raise RenderPayloadError(f"component {component['component_id']} has no nodes")
    low = np.min(projected[indices, :2], axis=0)
    high = np.max(projected[indices, :2], axis=0)
    padding = np.maximum((high - low) * 0.75, 5.0)
    primary_limits = (
        (float(low[0] - padding[0]), float(high[0] + padding[0])),
        (float(low[1] - padding[1]), float(high[1] + padding[1])),
    )
    raw_masks = [
        np.all((row[:, :2] >= low - padding) & (row[:, :2] <= high + padding), axis=1)
        for row in raw_projected
    ]
    voxel_masks = [
        np.all((row[:, :2] >= low - padding) & (row[:, :2] <= high + padding), axis=1)
        for row in voxel_projected
    ]
    depth_scope = [projected[indices][:, [0, 2, 1]]]
    depth_scope.extend(row[mask][:, [0, 2, 1]] for row, mask in zip(raw_projected, raw_masks))
    depth_scope.extend(row[mask][:, [0, 2, 1]] for row, mask in zip(voxel_projected, voxel_masks))
    depth_limits = _same_camera_limits(depth_scope)

    for view_ordinal, (reorder, limits, view_label) in enumerate(
        (([0, 1, 2], primary_limits, "PCA1-PCA2"), ([0, 2, 1], depth_limits, "PCA1-PCA3 depth"))
    ):
        for member_ordinal, member in enumerate(support_members):
            axis = axes[0, 2 * view_ordinal + member_ordinal]
            support_label = (
                "central raw context + row voxel"
                if member.get("support_mode") == "CENTRAL_RAW_CONTEXT_PLUS_ROW_SPECIFIC_VOXELS"
                else "central raw + voxel"
            )
            _configure_axis(
                axis,
                f"{member['opaque_token']} {support_label} — {view_label} (fixed paired camera)",
            )
            raw = raw_projected[member_ordinal][:, reorder]
            voxel = voxel_projected[member_ordinal][:, reorder]
            cad_view = projected[:, reorder]
            raw_ids = np.flatnonzero(raw_masks[member_ordinal])
            raw_ids = raw_ids[_downsample_indices(len(raw_ids), 30000)]
            raw_valid = member["raw_valid"][raw_ids]
            axis.scatter(
                raw[raw_ids[~raw_valid], 0], raw[raw_ids[~raw_valid], 1],
                color=REJECTED, s=0.35, alpha=0.13, linewidths=0,
            )
            axis.scatter(
                raw[raw_ids[raw_valid], 0], raw[raw_ids[raw_valid], 1],
                color=FIRST, s=0.35, alpha=0.14, linewidths=0,
            )
            invalid = voxel_masks[member_ordinal] & ~member["voxel_valid"]
            valid_not_selected = (
                voxel_masks[member_ordinal] & member["voxel_valid"] & ~member["selected"]
            )
            chosen = voxel_masks[member_ordinal] & member["selected"]
            axis.scatter(
                voxel[invalid, 0], voxel[invalid, 1], color=REJECTED,
                marker="x", s=5.0, alpha=0.42,
            )
            axis.scatter(
                voxel[valid_not_selected, 0], voxel[valid_not_selected, 1],
                color=MUTED, s=3.2, alpha=0.34, linewidths=0,
            )
            if np.any(chosen):
                axis.scatter(
                    voxel[chosen, 0], voxel[chosen, 1],
                    c=_discrete_colors(member["sheets"][chosen] + member_ordinal * 2),
                    s=4.0, alpha=0.72, linewidths=0,
                )
            axis.scatter(
                cad_view[indices, 0], cad_view[indices, 1], facecolors="none",
                edgecolors=HALO, s=28.0, linewidths=0.8,
            )
            axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1])
            axis.set_aspect("equal", adjustable="box")
            if view_ordinal == 1:
                axis.text(
                    0.01, 0.02, "equal mm axes; no depth exaggeration",
                    transform=axis.transAxes, color=TEXT, fontsize=5,
                    bbox={"facecolor": PANEL, "edgecolor": GRID, "alpha": 0.72},
                )
            sheet_handles = [
                Patch(
                    color=_discrete_colors(np.asarray([sheet + member_ordinal * 2]))[0],
                    label=f"voxel selected sheet {sheet}",
                )
                for sheet in sorted(set(int(value) for value in member["sheets"][chosen]))
            ]
            axis.legend(
                handles=[
                    Patch(color=FIRST, label="raw nearest-CAD valid"),
                    Patch(color=REJECTED, label="nearest-CAD invalid"),
                    Patch(color=MUTED, label="voxel valid, not winning"),
                    *sheet_handles,
                    Patch(facecolor="none", edgecolor=HALO, label="component CAD node"),
                ],
                loc="lower right", fontsize=5, facecolor=PANEL, labelcolor=TEXT,
            )

    axis = axes[1, 0]
    _configure_axis(axis, "Smooth regions / folds / strict halo")
    axis.scatter(projected[:, 0], projected[:, 1], c=_discrete_colors(cad["regions"]), s=3.0, alpha=0.38, linewidths=0)
    axis.scatter(projected[cad["halo"], 0], projected[cad["halo"], 1], facecolors="none", edgecolors=HALO, s=7.0, linewidths=0.35)
    if indices:
        axis.scatter(projected[indices, 0], projected[indices, 1], c=_discrete_colors(cad["folds"][indices]), s=28.0, edgecolors="white", linewidths=0.35)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(
        handles=[
            Patch(color="#45d7ff", label="smooth region color"),
            Patch(facecolor="none", edgecolor=HALO, label=f">{FOLD_HALO_MM:.3f} mm halo boundary"),
            Patch(color=ACTIVE, label="component fold color"),
        ], fontsize=5, loc="lower right", facecolor=PANEL, labelcolor=TEXT,
    )

    axis = axes[1, 1]
    _configure_axis(axis, "Active induced component graph")
    index_set = set(indices)
    for left, right in cad["edges"]:
        if int(left) in index_set and int(right) in index_set:
            axis.plot(projected[[left, right], 0], projected[[left, right], 1], color=TEXT, alpha=0.55, linewidth=0.8)
    if indices:
        node_active = [row["active"] for row in members]
        axis.scatter(projected[indices, 0], projected[indices, 1], c=[ACTIVE if value else MUTED for value in node_active], s=32.0, linewidths=0)
        low = np.min(projected[indices, :2], axis=0); high = np.max(projected[indices, :2], axis=0)
        padding = np.maximum((high - low) * 0.12, 0.5)
        axis.set_xlim(float(low[0] - padding[0]), float(high[0] + padding[0]))
        axis.set_ylim(float(low[1] - padding[1]), float(high[1] + padding[1]))
        axis.set_aspect("equal", adjustable="box")
    else:
        _empty(axis, "EMPTY COMPONENT")

    amplitudes = [row["amplitude"] for row in members]
    thresholds = [row["threshold"] for row in members]
    noises = [row["split_noise"] for row in members]
    axis = axes[1, 2]
    _configure_axis(axis, "Amplitude / split noise / frozen T")
    if members:
        x = np.arange(len(members))
        axis.plot(x, thresholds, color=REJECTED, label="T", linewidth=1.0)
        axis.plot(x, noises, color=HALO, label="split noise", linewidth=1.0)
        axis.scatter(x, amplitudes, color=ACTIVE, s=13.0, label="amplitude")
        axis.set_xticks(x, [str(row["node_id"]) for row in members], rotation=90, fontsize=5)
        axis.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT)
    else:
        _empty(axis, "NO COMPONENT NODES")

    axis = axes[1, 3]
    _signed_feature_heatmap(
        axis, members,
        title="Signed f=[h,3g1,3g2] pre/post nuisance by component node",
    )

    axis = axes[2, 0]
    _configure_axis(axis, "16 x 2 footprint completeness")
    if members:
        matrix = np.concatenate([np.concatenate([row["diagnostics"][0]["footprint"], row["diagnostics"][1]["footprint"]], axis=0) for row in members], axis=0)
        axis.imshow(matrix.astype(float), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    else:
        _empty(axis, "NO FOOTPRINTS")

    axis = axes[2, 1]
    _configure_axis(axis, "Plane fit p90 by node/member")
    if members:
        x = np.arange(len(members)); width = 0.34
        axis.bar(x - width / 2, [row["diagnostics"][0]["plane_p90_mm"] for row in members], width, color=FIRST)
        axis.bar(x + width / 2, [row["diagnostics"][1]["plane_p90_mm"] for row in members], width, color=SECOND)
        axis.axhline(0.3, color=REJECTED, linestyle="--", linewidth=0.8)
    else:
        _empty(axis, "NO PLANE FITS")

    axis = axes[2, 2]
    _configure_axis(axis, "All height-cluster rows / winner / runner")
    if members:
        for node_position, row in enumerate(members):
            for diagnostic_position, diagnostic in enumerate(row["diagnostics"]):
                for cluster_index, cluster in enumerate(diagnostic["clusters"]):
                    if len(cluster) == 0:
                        continue
                    color = SELECTED if cluster_index == diagnostic["winner"] else HALO if cluster_index == diagnostic["runner"] else MUTED
                    offsets = np.linspace(-0.055, 0.055, len(cluster))
                    x = node_position + 0.22 * diagnostic_position + offsets
                    marker = "o" if cluster_index == diagnostic["winner"] else "^" if cluster_index == diagnostic["runner"] else "."
                    axis.scatter(x, cluster, color=color, marker=marker, s=10.0, alpha=0.72)
        axis.axhline(0.0, color=MUTED, linewidth=0.6)
        axis.set_xticks(
            np.arange(len(members)) + 0.11,
            [str(row["node_id"]) for row in members], rotation=90, fontsize=5,
        )
    else:
        _empty(axis, "NO HEIGHT CLUSTERS")

    axis = axes[2, 3]
    _configure_axis(axis, "Corrected member energies / ratio / attribution")
    if np.any(np.isfinite(component["energies"])):
        axis.bar([0, 1], component["energies"], color=[FIRST, SECOND])
        axis.set_xticks([0, 1], [data["members"][0]["opaque_token"], data["members"][1]["opaque_token"]])
        axis.text(
            0.03, 0.96,
            f"ratio={_format_optional(component['energy_ratio'])}\n"
            f"attribution={component['attributed_slot'] or 'ABSTAIN'}\n"
            f"pair energy fraction={_format_optional(component['pairwise_energy_fraction'])}",
            transform=axis.transAxes, va="top", color=TEXT, fontsize=8,
        )
    else:
        _empty(axis, "CORRECTED ENERGIES UNAVAILABLE\nNO ATTRIBUTION")

    axis = axes[3, 0]
    _configure_axis(axis, "Variant component area-Jaccard (minimum 0.6)")
    if component["variants"]:
        x = np.arange(len(component["variants"]))
        axis.bar(x, [row["jaccard"] for row in component["variants"]], color=[SELECTED if row["passed"] else REJECTED for row in component["variants"]])
        axis.axhline(0.6, color=HALO, linestyle="--", linewidth=0.8)
        axis.set_xticks(x, [row["row_id"] for row in component["variants"]], rotation=90, fontsize=5)
    else:
        _empty(axis, "NO VARIANT MATCH RECORDS")

    axis = axes[3, 1]
    _configure_axis(axis, "Variant medoid displacement (maximum 3 mm)")
    if component["variants"] and any(math.isfinite(row["medoid_displacement_mm"]) for row in component["variants"]):
        x = np.arange(len(component["variants"]))
        axis.bar(x, [row["medoid_displacement_mm"] for row in component["variants"]], color=[SELECTED if row["passed"] else REJECTED for row in component["variants"]])
        axis.axhline(3.0, color=HALO, linestyle="--", linewidth=0.8)
        axis.set_xticks(x, [row["row_id"] for row in component["variants"]], rotation=90, fontsize=5)
    else:
        _empty(axis, "UNMATCHED / VANISHED COMPONENT\nMEDOID DISPLACEMENT NOT DEFINED")

    axis = axes[3, 2]
    _configure_axis(axis, "Variant corrected-energy ratio (minimum 2.0)")
    finite_ratio_variants = [
        row for row in component["variants"] if math.isfinite(row["energy_ratio"])
    ]
    if finite_ratio_variants:
        x = np.arange(len(finite_ratio_variants))
        axis.bar(x, [row["energy_ratio"] for row in finite_ratio_variants], color=[SELECTED if row["passed"] else REJECTED for row in finite_ratio_variants])
        axis.axhline(2.0, color=HALO, linestyle="--", linewidth=0.8)
        axis.set_xticks(x, [f"{row['row_id']}\n{row['attributed_slot'] or 'ABSTAIN'}" for row in finite_ratio_variants], rotation=90, fontsize=4)
    else:
        _empty(axis, "VARIANT ENERGY RATIO UNAVAILABLE")

    axis = axes[3, 3]
    _configure_axis(axis, "Frozen component accounting")
    axis.set_axis_off()
    table = axis.table(
        cellText=[
            ["component ID", component["component_id"]],
            ["node count", str(len(component["node_ids"]))],
            ["area mm²", f"{component['area_mm2']:.6f}"],
            ["geodesic span mm", f"{component['geodesic_span_mm']:.6f}"],
            ["retained", str(component["retained"])],
            ["robustness changed", str(component["robustness_changed"])],
            ["source row", str(component.get("source_row_id") or "CENTRAL")],
            ["vanished in variant", str(bool(component.get("vanished_in_variant", False)))],
            ["energy ratio", _format_optional(component["energy_ratio"], precision=6)],
            ["attribution", component["attributed_slot"] or "ABSTAIN"],
        ],
        colWidths=[0.52, 0.48], loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1.0, 1.35)
    for cell in table.get_celld().values():
        cell.set_facecolor(PANEL); cell.set_edgecolor(GRID); cell.get_text().set_color(TEXT)

    suffix = (
        f" — row {component['source_row_id']}"
        f" — {'VANISHED CENTRAL' if component.get('vanished_in_variant') else 'VARIANT LOCAL'}"
        if component.get("source_row_id") is not None else " — CENTRAL"
    )
    figure.suptitle(
        f"EXP_CAD_0030 component pack — {component['component_id']}{suffix}",
        color=TEXT, fontsize=18, y=0.995,
    )
    figure.subplots_adjust(left=0.04, right=0.985, bottom=0.045, top=0.96, wspace=0.25, hspace=0.34)
    _save(figure, path)


def render_native_visuals(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Render the complete frozen native visual pack and return its manifest."""

    data = validate_payload(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = {"global_paired_diagnostic.png", "native_visual_manifest.json"}

    global_path = output_dir / "global_paired_diagnostic.png"
    _render_global(global_path, data)

    selected_components, selected_nodes = select_visual_inventory(data)
    component_assets = []
    for component in selected_components:
        filename = f"component_{component['component_id']}.png"
        expected_names.add(filename)
        path = output_dir / filename
        _render_component(path, data, component)
        component_assets.append(
            {
                "component_id": component["component_id"],
                "source_row_id": component.get("source_row_id"),
                "vanished_in_variant": bool(component.get("vanished_in_variant", False)),
                "path": filename,
                "selection_reasons": [
                    reason
                    for reason, selected in (
                        ("RETAINED_COMPONENT", component["retained"]),
                        ("ROBUSTNESS_CHANGED_COMPONENT", component["robustness_changed"]),
                    )
                    if selected
                ],
                "sha256": _sha256(path),
                "width_px": WIDTH_PX,
                "height_px": HEIGHT_PX,
                "fixed_same_scale_views": ["PCA1_PCA2", "PCA1_PCA3_DEPTH"],
                "support_mode": (
                    "CENTRAL_RAW_CONTEXT_PLUS_ROW_SPECIFIC_VOXELS"
                    if component.get("member_support") is not None
                    else "CENTRAL_RAW_AND_VOXELS"
                ),
            }
        )

    node_assets = []
    for rank, node in enumerate(selected_nodes, start=1):
        filename = f"node_{node['node_id']:06d}.png"
        expected_names.add(filename)
        path = output_dir / filename
        _render_node(path, data, node)
        node_assets.append(
            {
                "node_id": node["node_id"],
                "central_amplitude_rank": rank,
                "amplitude_mm": node["amplitude"],
                "path": filename,
                "selection_reasons": ["FIXED_TOP_12_CENTRAL_AMPLITUDE_DESC_THEN_NODE_ID"],
                "sha256": _sha256(path),
                "width_px": WIDTH_PX,
                "height_px": HEIGHT_PX,
            }
        )

    # Refuse stale PNGs because a prior component set could otherwise be
    # mistaken for the current experiment's required visual inventory.
    for existing in output_dir.glob("*.png"):
        if existing.name not in expected_names:
            existing.unlink()

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "render_payload_schema": SCHEMA,
        "render_payload_semantic_sha256": _semantic_sha256(payload),
        "pair_id": data["pair_id"],
        "native_resolution": {"width_px": WIDTH_PX, "height_px": HEIGHT_PX},
        "same_camera_and_color_scale_for_both_members": True,
        "global_fixed_same_scale_views": ["PCA1_PCA2", "PCA1_PCA3_DEPTH"],
        "global_diagnostic_emitted_even_zero_components": True,
        "fold_exclusion_halo_mm_strictly_greater_than": FOLD_HALO_MM,
        "global_asset": {
            "path": global_path.name,
            "sha256": _sha256(global_path),
            "width_px": WIDTH_PX,
            "height_px": HEIGHT_PX,
        },
        "component_assets": component_assets,
        "node_assets": node_assets,
        "retained_component_count": sum(row["retained"] for row in data["components"]),
        "robustness_changed_component_count": sum(row["robustness_changed"] for row in data["components"]),
        "row_specific_changed_component_asset_count": sum(
            row.get("source_row_id") is not None for row in selected_components
        ),
        "vanished_central_component_asset_count": sum(
            bool(row.get("vanished_in_variant", False)) for row in selected_components
        ),
        "fixed_top_amplitude_node_count": len(selected_nodes),
        "manual_original_resolution_review_required": True,
        "manual_review_status": "AWAITING_MANUAL_NATIVE_RESOLUTION_REVIEW",
        "provenance": {
            "render_only": True,
            "external_source_opened_by_renderer": False,
            "evaluator_data_opened": False,
            "gold_labels_expected_counts_or_quality_identity_opened": False,
            "canonical_exp30_schema_only": True,
        },
    }
    manifest_path = output_dir / "native_visual_manifest.json"
    _canonical_json(manifest_path, manifest)
    return manifest


__all__ = [
    "FOLD_HALO_MM",
    "HEIGHT_PX",
    "EXPERIMENT_ID",
    "ARCHITECTURE",
    "MANIFEST_SCHEMA",
    "RenderPayloadError",
    "SCHEMA",
    "TOP_NODE_COUNT",
    "WIDTH_PX",
    "render_native_visuals",
    "select_visual_inventory",
    "validate_payload",
]
