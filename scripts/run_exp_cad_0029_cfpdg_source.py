#!/usr/bin/env python3
"""Hash-gated physical source runner for EXP_CAD_0030 CF-PDG V3.

The process fails before opening either scan unless two independent approvals
bind the exact CAD preflight and exact source implementation/test hashes.  It
never accepts an evaluator manifest, role, label, gold, expected count, part
quality, or filename-derived decision input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import exp_cad_0029_cfpdg_core as CAD
from scripts import exp_cad_0029_cfpdg_source as SRC


EXPECTED_SCHEMA = "EXP_CAD_0030_CFPDG_V3_PREREGISTRATION_V1"
EXPECTED_EXPERIMENT = "EXP_CAD_0030"
EXPECTED_ARCHITECTURE = "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED"
RUNNER_IDENTITY_PATH = Path(__file__).resolve()
VISUAL_ONLY_ROBUSTNESS_FIELDS = frozenset(
    {
        "variant_components",
        "member_support",
        "changed_component_local_diagnostics",
        "changed_visual_components",
    }
)
CAD_APPROVAL_BOUND_FILE_KEYS = frozenset(
    {
        "preregistration", "cad_preflight", "cad_only_audit_packet",
        "nodes", "edges", "edge_geodesic_mm", "mesh", "core",
        "cad_only_runner", "cad_only_wrapper", "cad_only_tests",
        "cad_identity_tests",
    }
)
SOURCE_APPROVAL_BOUND_FILE_KEYS = frozenset(
    set(CAD_APPROVAL_BOUND_FILE_KEYS)
    | {
        "cad_approval", "source_module", "source_runner",
        "source_runner_base", "source_tests", "renderer",
        "renderer_tests", "orchestration", "orchestration_tests",
    }
)
CAD_APPROVAL_ARTIFACT_KEYS = frozenset(
    {
        "CAD_ONLY_PREFLIGHT.json", "CAD_GRAPH_NODES.npy",
        "CAD_GRAPH_EDGES.npy", "CAD_GRAPH_EDGE_GEODESIC_MM.npy",
    }
)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = CAD.sha256_path(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch")


def _bound_file_sha(payload: Mapping[str, Any], key: str) -> str:
    bound = payload.get("bound_files")
    if not isinstance(bound, Mapping) or not isinstance(bound.get(key), Mapping):
        raise ValueError(f"approval lacks exact bound_files.{key}")
    value = bound[key].get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"approval has invalid bound_files.{key}.sha256")
    return value


def _bound_artifact_sha(payload: Mapping[str, Any], key: str) -> str:
    bound = payload.get("bound_cad_artifacts")
    if not isinstance(bound, Mapping):
        raise ValueError("approval lacks bound_cad_artifacts")
    value = bound.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"approval lacks exact bound_cad_artifacts.{key}")
    return value


def _require_exact_approval_key_sets(
    payload: Mapping[str, Any],
    *,
    bound_file_keys: frozenset[str],
    require_cad_artifacts: bool,
) -> None:
    bound_files = payload.get("bound_files")
    if not isinstance(bound_files, Mapping) or set(bound_files) != set(bound_file_keys):
        raise ValueError("approval bound_files keys are not the exact frozen set")
    artifacts = payload.get("bound_cad_artifacts")
    if require_cad_artifacts:
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(CAD_APPROVAL_ARTIFACT_KEYS):
            raise ValueError("CAD approval artifact keys are not the exact frozen set")
    elif artifacts is not None:
        raise ValueError("source approval must not carry an unbound CAD-artifact namespace")


def validate_cad_approval_header(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP_CAD_0030_V3_INDEPENDENT_CAD_ONLY_APPROVAL_V1":
        raise ValueError("unexpected independent CAD approval schema")
    if payload.get("decision") != "PASS_CAD_ONLY_FREEZE_FOR_SOURCE_RUNNER_IMPLEMENTATION":
        raise ValueError("independent CAD approval decision unavailable")
    if payload.get("physical_source_execution_authorized") is not False:
        raise ValueError("CAD approval must not authorize physical execution")
    if (
        payload.get("experiment_id") != EXPECTED_EXPERIMENT
        or payload.get("architecture") != EXPECTED_ARCHITECTURE
    ):
        raise ValueError("independent CAD approval identity mismatch")


def validate_source_approval_header(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "EXP_CAD_0030_V3_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1":
        raise ValueError("unexpected independent source approval schema")
    if payload.get("decision") != "APPROVE_ONE_PHYSICAL_PRIMARY_REPLAY_EXECUTION":
        raise ValueError("independent source approval decision unavailable")
    if payload.get("physical_source_execution_authorized") is not True:
        raise ValueError("source approval does not authorize physical execution")
    if (
        payload.get("experiment_id") != EXPECTED_EXPERIMENT
        or payload.get("architecture") != EXPECTED_ARCHITECTURE
    ):
        raise ValueError("independent source approval identity mismatch")


def validate_cad_preflight_identity(
    payload: Mapping[str, Any], preregistration_sha256: str
) -> None:
    if payload.get("schema") != "EXP_CAD_0030_CFPDG_V3_CAD_ONLY_PREFLIGHT_V1":
        raise ValueError("unexpected EXP30 CAD preflight schema")
    if (
        payload.get("experiment_id") != EXPECTED_EXPERIMENT
        or payload.get("architecture") != EXPECTED_ARCHITECTURE
    ):
        raise ValueError("CAD preflight experiment identity mismatch")
    if payload.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("CAD preflight preregistration hash mismatch")
    if payload.get("status") != "PASS_CAD_ONLY_ACTIONABLE" or payload.get("cad_actionable") is not True:
        raise ValueError("CAD is not actionable")
    if payload.get("source_execution_authorized") is not False:
        raise ValueError("CAD preflight improperly self-authorizes source execution")


def verify_pre_scan_approvals(
    *,
    preregistration: Path,
    cad_preflight: Path,
    cad_approval: Path,
    source_runner_approval: Path,
    nodes_path: Path,
    edges_path: Path,
    edge_weights_path: Path,
    mesh_path: Path,
    source_module_path: Path,
    runner_path: Path,
    source_tests_path: Path,
    cad_audit_packet_path: Path,
    renderer_path: Path,
    renderer_tests_path: Path,
    orchestration_path: Path,
    orchestration_tests_path: Path,
    cad_only_runner_path: Path,
    cad_only_wrapper_path: Path,
    cad_only_tests_path: Path,
    cad_identity_tests_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Verify every approval before any scan hash or point-cloud open call."""

    prereg = _load_json(preregistration)
    if prereg.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected preregistration schema")
    if prereg.get("experiment_id") != EXPECTED_EXPERIMENT or prereg.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("unexpected experiment identity")
    preregistration_sha256 = CAD.sha256_path(preregistration)
    cad_payload = _load_json(cad_preflight)
    validate_cad_preflight_identity(cad_payload, preregistration_sha256)
    cad_approval_payload = _load_json(cad_approval)
    validate_cad_approval_header(cad_approval_payload)
    _require_exact_approval_key_sets(
        cad_approval_payload,
        bound_file_keys=CAD_APPROVAL_BOUND_FILE_KEYS,
        require_cad_artifacts=True,
    )
    if _bound_file_sha(cad_approval_payload, "preregistration") != preregistration_sha256:
        raise ValueError("CAD approval preregistration mismatch")
    if _bound_file_sha(cad_approval_payload, "cad_only_audit_packet") != CAD.sha256_path(cad_audit_packet_path):
        raise ValueError("CAD approval packet mismatch")
    for key, path in (
        ("cad_preflight", cad_preflight),
        ("nodes", nodes_path),
        ("edges", edges_path),
        ("edge_geodesic_mm", edge_weights_path),
        ("mesh", mesh_path),
        ("core", Path(CAD.__file__)),
        ("cad_only_runner", cad_only_runner_path),
        ("cad_only_wrapper", cad_only_wrapper_path),
        ("cad_only_tests", cad_only_tests_path),
        ("cad_identity_tests", cad_identity_tests_path),
    ):
        if _bound_file_sha(cad_approval_payload, key) != CAD.sha256_path(path):
            raise ValueError(f"CAD approval file mismatch: {key}")
    for key, path in (
        ("CAD_ONLY_PREFLIGHT.json", cad_preflight),
        ("CAD_GRAPH_NODES.npy", nodes_path),
        ("CAD_GRAPH_EDGES.npy", edges_path),
        ("CAD_GRAPH_EDGE_GEODESIC_MM.npy", edge_weights_path),
    ):
        if _bound_artifact_sha(cad_approval_payload, key) != CAD.sha256_path(path):
            raise ValueError(f"CAD approval artifact mismatch: {key}")

    source_approval = _load_json(source_runner_approval)
    validate_source_approval_header(source_approval)
    _require_exact_approval_key_sets(
        source_approval,
        bound_file_keys=SOURCE_APPROVAL_BOUND_FILE_KEYS,
        require_cad_artifacts=False,
    )
    source_named = {
        "preregistration": CAD.sha256_path(preregistration),
        "cad_preflight": CAD.sha256_path(cad_preflight),
        "cad_approval": CAD.sha256_path(cad_approval),
        "nodes": CAD.sha256_path(nodes_path),
        "edges": CAD.sha256_path(edges_path),
        "edge_geodesic_mm": CAD.sha256_path(edge_weights_path),
        "mesh": CAD.sha256_path(mesh_path),
        "source_module": CAD.sha256_path(source_module_path),
        "source_runner": CAD.sha256_path(runner_path),
        "source_runner_base": CAD.sha256_path(Path(__file__)),
        "source_tests": CAD.sha256_path(source_tests_path),
        "cad_only_audit_packet": CAD.sha256_path(cad_audit_packet_path),
        "renderer": CAD.sha256_path(renderer_path),
        "renderer_tests": CAD.sha256_path(renderer_tests_path),
        "orchestration": CAD.sha256_path(orchestration_path),
        "orchestration_tests": CAD.sha256_path(orchestration_tests_path),
        "core": CAD.sha256_path(Path(CAD.__file__)),
        "cad_only_runner": CAD.sha256_path(cad_only_runner_path),
        "cad_only_wrapper": CAD.sha256_path(cad_only_wrapper_path),
        "cad_only_tests": CAD.sha256_path(cad_only_tests_path),
        "cad_identity_tests": CAD.sha256_path(cad_identity_tests_path),
    }
    for name, actual in source_named.items():
        if _bound_file_sha(source_approval, name) != actual:
            raise ValueError(f"source approval does not bind {name}")
    return prereg, cad_payload


def load_transform(path: Path, token: str, scan_sha256: str, expected_semantic_sha: str) -> np.ndarray:
    _require_hash(path, expected_semantic_sha, f"{token} transform semantic")
    payload = _load_json(path)
    if payload.get("case_token") != token or payload.get("scan_sha256") != scan_sha256:
        raise ValueError(f"{token} transform binding mismatch")
    for field in (
        "selected_absolute_transform",
        "raw_scan_to_reference_transform",
        "transform",
    ):
        if field in payload:
            transform = np.asarray(payload[field], dtype=np.float64)
            break
    else:
        raise ValueError(f"{token} transform matrix unavailable")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{token} transform invalid")
    if not np.array_equal(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{token} transform bottom row is not exact rigid homogeneous form")
    rotation = transform[:3, :3]
    orthogonality = rotation.T @ rotation
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(
        orthogonality, np.eye(3), rtol=0.0, atol=1e-8
    ):
        raise ValueError(f"{token} transform rotation is not orthonormal")
    if determinant <= 0.0 or not math.isclose(
        determinant, 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(f"{token} transform is not a proper rotation")
    return transform


def load_cad_source_artifacts_before_scan(
    *,
    nodes_path: Path,
    edges_path: Path,
    edge_weights_path: Path,
    mesh_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, o3d.geometry.TriangleMesh]:
    """Load and validate all CAD artifacts before first physical scan byte."""

    nodes = np.load(nodes_path, allow_pickle=False)
    edges = np.load(edges_path, allow_pickle=False)
    edge_weights = np.load(edge_weights_path, allow_pickle=False)
    required_fields = (
        "x", "y", "z", "nx", "ny", "nz",
        "e1x", "e1y", "e1z", "e2x", "e2y", "e2z",
        "parent_face", "candidate_ordinal", "smooth_region", "area_mm2",
        "fold_id", "graph_component",
    )
    if nodes.ndim != 1 or tuple(nodes.dtype.names or ()) != required_fields:
        raise ValueError("CAD node artifact dtype/schema mismatch")
    for field in required_fields[:12] + ("area_mm2",):
        if not np.all(np.isfinite(nodes[field])):
            raise ValueError(f"CAD node artifact nonfinite field: {field}")
    if len(nodes) == 0 or np.any(nodes["area_mm2"] <= 0.0):
        raise ValueError("CAD node artifact empty or nonpositive area")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("CAD edge artifact must be integer E-by-2")
    if np.any(edges < 0) or np.any(edges >= len(nodes)):
        raise ValueError("CAD edge artifact contains out-of-range node")
    if (
        edge_weights.ndim != 1
        or len(edge_weights) != len(edges)
        or not np.all(np.isfinite(edge_weights))
        or np.any(edge_weights <= 0.0)
    ):
        raise ValueError("CAD edge-geodesic artifact invalid")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path.resolve()))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError("accepted CAD mesh is empty or unreadable")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not np.all(np.isfinite(vertices))
        or np.any(triangles < 0)
        or np.any(triangles >= len(vertices))
    ):
        raise ValueError("accepted CAD mesh geometry is malformed")
    return nodes, edges, edge_weights, mesh


def raycast_regions_and_sheets(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
    smooth_region_by_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    answer = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, dtype=np.float32)))
    primitive = answer["primitive_ids"].numpy().astype(np.int64)
    closest = answer["points"].numpy().astype(np.float64)
    normals = answer["primitive_normals"].numpy().astype(np.float64)
    valid = primitive < len(smooth_region_by_face)
    regions = np.full(len(points), -1, dtype=np.int64)
    regions[valid] = np.asarray(smooth_region_by_face, dtype=np.int64)[primitive[valid]]
    signed = np.sum((np.asarray(points) - closest) * normals, axis=1)
    sheets = np.where(signed >= 0.0, 1, -1).astype(np.int8)
    sheets[~valid] = 0
    return regions, sheets, valid


def nullable_sheet_ids(sheets: np.ndarray, valid: np.ndarray) -> list[int | None]:
    """Serialize sheet ownership without inventing a numeric invalid sheet."""

    sheet_array = np.asarray(sheets, dtype=np.int64)
    valid_array = np.asarray(valid, dtype=bool)
    if sheet_array.shape != valid_array.shape or sheet_array.ndim != 1:
        raise ValueError("sheet ownership/validity shape mismatch")
    if np.any(valid_array & ~np.isin(sheet_array, (-1, 1))):
        raise ValueError("nearest-CAD-valid sheet ownership must be -1 or +1")
    return [int(sheet) if is_valid else None for sheet, is_valid in zip(sheet_array, valid_array, strict=True)]


def member_fit_variant(
    *,
    raw_points_cad: np.ndarray,
    raw_normals_cad: np.ndarray,
    nodes: np.ndarray,
    scene: o3d.t.geometry.RaycastingScene,
    smooth_region_by_face: np.ndarray,
    local_config: Mapping[str, Any],
    voxel_mm: float,
    partition_mask: np.ndarray | None,
    pose_translation: np.ndarray | None,
    pose_rotation: tuple[int, float] | None,
    fixed_centroid: np.ndarray,
    keep_diagnostics: bool,
) -> dict[str, Any]:
    points = np.asarray(raw_points_cad, dtype=np.float64)
    normals = np.asarray(raw_normals_cad, dtype=np.float64)
    if pose_translation is not None:
        points = points + np.asarray(pose_translation, dtype=np.float64)
    if pose_rotation is not None:
        points, normals = SRC.rotate_about_center(
            points, normals, fixed_centroid, pose_rotation[0], pose_rotation[1]
        )
    voxel_points, voxel_normals, voxel_keys = SRC.voxelize_partition(
        points, normals, float(voxel_mm), partition_mask
    )
    if len(voxel_points):
        voxel_regions, voxel_sheets, voxel_selected = raycast_regions_and_sheets(
            scene, voxel_points, smooth_region_by_face
        )
    else:
        voxel_regions = np.empty(0, dtype=np.int64)
        voxel_sheets = np.empty(0, dtype=np.int8)
        voxel_selected = np.empty(0, dtype=bool)
    features, valid, selected_counts, diagnostics, winning_selected = SRC.fit_scan_to_nodes(
        nodes=nodes,
        voxel_points=voxel_points,
        voxel_normals=voxel_normals,
        voxel_keys=voxel_keys,
        voxel_regions=voxel_regions,
        voxel_sheets=voxel_sheets,
        config=local_config,
        keep_diagnostics=keep_diagnostics,
    )
    return {
        "features": features,
        "valid": valid,
        "selected_counts": selected_counts,
        "diagnostics": diagnostics,
        "voxel_points": voxel_points,
        "voxel_normals": voxel_normals,
        "voxel_keys": voxel_keys,
        "voxel_regions": voxel_regions,
        "voxel_sheets": voxel_sheets,
        "voxel_nearest_cad_valid": voxel_selected,
        "voxel_selected": winning_selected,
    }


def count_ratio_gate(
    first: Mapping[str, Any], second: Mapping[str, Any], minimum: float, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    first_valid = np.asarray(first["valid"], dtype=bool).copy()
    second_valid = np.asarray(second["valid"], dtype=bool).copy()
    common = first_valid & second_valid
    ratio = np.full(len(first_valid), np.nan, dtype=np.float64)
    ratio[common] = (
        np.asarray(first["selected_counts"], dtype=np.float64)[common]
        / np.asarray(second["selected_counts"], dtype=np.float64)[common]
    )
    rejected = common & ((ratio < float(minimum)) | (ratio > float(maximum)))
    first_valid[rejected] = False
    second_valid[rejected] = False
    return first_valid, second_valid


def evaluate_pair_row(
    *,
    row_id: str,
    first_fit: Mapping[str, Any],
    second_fit: Mapping[str, Any],
    nodes: np.ndarray,
    fold_training_masks: Sequence[np.ndarray],
    fixed_centroid: np.ndarray,
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    local = prereg["local_surface_fit"]
    first_valid, second_valid = count_ratio_gate(
        first_fit,
        second_fit,
        float(local["minimum_member_point_count_ratio"]),
        float(local["maximum_member_point_count_ratio"]),
    )
    pair = SRC.crossfit_pair_field(
        nodes=nodes,
        feature_first=np.asarray(first_fit["features"]),
        valid_first=first_valid,
        feature_second=np.asarray(second_fit["features"]),
        valid_second=second_valid,
        fold_training_masks=fold_training_masks,
        fixed_centroid=fixed_centroid,
        config=prereg["global_crossfit_nuisance"],
        pair_config=prereg["pair_actionability"],
    )
    return {
        "row_id": row_id,
        "first_fit": first_fit,
        "second_fit": second_fit,
        "first_valid": first_valid,
        "second_valid": second_valid,
        "pair": pair,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def semantic_robustness_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep SOURCE_RESULT decision-scale while declaring every omitted field."""

    omitted = sorted(VISUAL_ONLY_ROBUSTNESS_FIELDS)
    return [
        {
            key: value
            for key, value in row.items()
            if key not in VISUAL_ONLY_ROBUSTNESS_FIELDS
        }
        | {"visual_only_fields_omitted_from_semantic_result": omitted}
        for row in rows
    ]


def _support_gate_for_split_complete(
    *,
    first_support: np.ndarray,
    second_support: np.ndarray,
    nodes: np.ndarray,
    fixed_centroid: np.ndarray,
    pair_config: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    area = np.asarray(nodes["area_mm2"], dtype=np.float64)
    interior = math.fsum(float(value) for value in area)
    common = np.asarray(first_support, dtype=bool) & np.asarray(second_support, dtype=bool)
    union = np.asarray(first_support, dtype=bool) | np.asarray(second_support, dtype=bool)
    support_area = math.fsum(float(value) for value in area[common])
    union_area = math.fsum(float(value) for value in area[union])
    one_sided_area = math.fsum(float(value) for value in area[union & ~common])
    design = SRC.nuisance_design(nodes, fixed_centroid)
    rank, condition = SRC.normalized_design_rank_condition(
        design[common], area[common]
    )
    metrics = {
        "node_count": int(np.count_nonzero(common)),
        "area_mm2": support_area,
        "area_fraction_of_cad_interior": support_area / interior,
        "rank": rank,
        "condition": condition,
        "union_area_mm2": union_area,
        "one_sided_support_area_fraction": one_sided_area / union_area if union_area else 0.0,
    }
    passed = bool(
        metrics["node_count"] >= int(pair_config["minimum_common_support_node_count"])
        and metrics["area_fraction_of_cad_interior"]
        >= float(pair_config["minimum_common_support_area_fraction_of_cad_interior"])
        and rank == int(pair_config["required_common_support_rigid_rank"])
        and condition <= float(pair_config["maximum_common_support_rigid_condition_number"])
        and metrics["one_sided_support_area_fraction"]
        <= float(pair_config["maximum_one_sided_support_area_fraction_of_union"])
    )
    return passed, metrics


def _row_component_result(
    *,
    row: Mapping[str, Any],
    threshold: np.ndarray,
    split_complete: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    prereg: Mapping[str, Any],
    first_token: str,
    second_token: str,
    central: bool,
) -> dict[str, Any]:
    pair = row["pair"]
    if pair.get("actionable") is not True:
        return {
            "row_id": row["row_id"],
            "status": "ABSTAIN",
            "reason": pair.get("reason", "PAIR_NOT_ACTIONABLE"),
            "components": [],
            "active": np.zeros(len(nodes), dtype=bool),
            "amplitude": np.full(len(nodes), np.nan),
        }
    support = np.asarray(pair["common"], dtype=bool) & np.isfinite(threshold)
    if central:
        support &= np.asarray(split_complete, dtype=bool)
    residual = np.asarray(pair["residual"], dtype=np.float64)
    amplitude = np.full(len(nodes), np.nan, dtype=np.float64)
    amplitude[support] = np.linalg.norm(residual[support], axis=1)
    active = support & (amplitude > threshold)
    node_config = prereg["node_and_component"]
    components, labels = SRC.induced_components(
        active=active,
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        minimum_node_count=int(node_config["minimum_component_node_count"]),
        minimum_area_mm2=float(node_config["minimum_component_area_mm2"]),
        minimum_span_mm=float(node_config["minimum_component_geodesic_span_mm"]),
    )
    attribution = SRC.attribute_components(
        components=components,
        area=np.asarray(nodes["area_mm2"], dtype=np.float64),
        feature_first=np.asarray(row["first_fit"]["features"], dtype=np.float64),
        feature_second=np.asarray(row["second_fit"]["features"], dtype=np.float64),
        prediction=np.asarray(pair["prediction"], dtype=np.float64),
        first_token=first_token,
        second_token=second_token,
        minimum_ratio=float(prereg["late_cad_attribution"]["minimum_larger_to_smaller_energy_ratio"]),
        major_fraction=float(prereg["late_cad_attribution"]["major_component_pairwise_energy_fraction"]),
        denominator_floor=float(prereg["late_cad_attribution"]["energy_ratio_denominator_floor_mm2"]),
    )
    return {
        "row_id": row["row_id"],
        "status": "ACTIONABLE" if attribution.get("actionable") else "ABSTAIN",
        "reason": None if attribution.get("actionable") else attribution.get("reason"),
        "components": attribution["components"],
        "attributed_opaque_member": attribution.get("attributed_opaque_member"),
        "total_pair_energy_mm4": attribution.get("total_pair_energy_mm4"),
        "active": active,
        "amplitude": amplitude,
        "component_labels": labels,
        "support": support,
    }


def _robustness_compare(
    *,
    central: Mapping[str, Any],
    variant: Mapping[str, Any],
    nodes: np.ndarray,
    full_graph: Any,
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    area = np.asarray(nodes["area_mm2"], dtype=np.float64)
    bijection = SRC.canonical_component_bijection(
        central["components"], variant["components"], area
    )
    if bijection is None:
        reason = (
            "EXACT_BIJECTION_OPTIMUM_RECONSTRUCTION_FAILED"
            if len(central["components"]) == len(variant["components"])
            else "COMPONENT_CARDINALITY_MISMATCH"
        )
        return {
            "passed": False,
            "reason": reason,
            "matches": [],
        }
    assignment, matrix = bijection
    config = prereg["robustness_inventory"]
    matches = []
    passed = True
    for central_id, variant_id in enumerate(assignment):
        first = central["components"][central_id]
        second = variant["components"][variant_id]
        jaccard = float(matrix[central_id, variant_id])
        first_medoid = int(first["area_medoid_node_id"])
        second_medoid = int(second["area_medoid_node_id"])
        displacement = float(
            dijkstra(full_graph, directed=False, indices=first_medoid)[second_medoid]
        )
        variant_energy_ratio = second.get("energy_ratio")
        component_pass = bool(
            jaccard >= float(config["minimum_area_weighted_component_jaccard"])
            and np.isfinite(displacement)
            and displacement <= float(config["maximum_geodesic_component_centroid_displacement_mm"])
            and variant_energy_ratio is not None
            and np.isfinite(float(variant_energy_ratio))
            and float(variant_energy_ratio)
                >= float(config["minimum_energy_ratio_every_variant"])
        )
        passed &= component_pass
        matches.append(
            {
                "central_component_id": central_id,
                "variant_component_id": variant_id,
                "area_weighted_jaccard": jaccard,
                "medoid_graph_geodesic_displacement_mm": displacement,
                "passed": component_pass,
            }
        )
    if central.get("attributed_opaque_member") != variant.get("attributed_opaque_member"):
        passed = False
    return {
        "passed": bool(passed),
        "reason": None if passed else "COMPONENT_OR_ATTRIBUTION_ROBUSTNESS_GATE",
        "matches": matches,
    }


def _robustness_row_change(
    *,
    central: Mapping[str, Any],
    variant: Mapping[str, Any],
    comparison: Mapping[str, Any],
    central_row: Mapping[str, Any],
    variant_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a local geometry/matching/opaque-decision visual obligation.

    Ordinary within-gate energy and ratio drift remains visible in the compact
    global persistence panel, but must not materialize a second full local
    support pack.  In particular, a valid R31 member swap changes the ordinal
    FIRST/SECOND slot by construction; when its physical opaque attribution is
    unchanged that representational swap is not a ``row_changed`` event.
    """

    reasons: set[str] = set()
    central_components = list(central.get("components", []))
    variant_components = list(variant.get("components", []))
    matches = list(comparison.get("matches", []))
    if central.get("status") != variant.get("status"):
        reasons.add("ROW_STATUS_CHANGED")
    if len(central_components) != len(variant_components):
        reasons.add("COMPONENT_CARDINALITY_CHANGED")
    if len(matches) != len(central_components):
        reasons.add("COMPONENT_MATCH_INCOMPLETE")
    match_by_central = {
        int(row["central_component_id"]): row for row in matches
    }
    if len(match_by_central) != len(matches):
        reasons.add("COMPONENT_MATCH_DUPLICATE")

    for central_id, first in enumerate(central_components):
        match = match_by_central.get(central_id)
        if match is None:
            reasons.add("COMPONENT_UNMATCHED")
            continue
        variant_id = int(match["variant_component_id"])
        if variant_id < 0 or variant_id >= len(variant_components):
            reasons.add("COMPONENT_MATCH_OUT_OF_RANGE")
            continue
        second = variant_components[variant_id]
        if variant_id != central_id:
            reasons.add("COMPONENT_MATCH_ID_CHANGED")
        if first.get("node_ids") != second.get("node_ids"):
            reasons.add("COMPONENT_NODE_IDS_CHANGED")
        if first.get("node_ids_sha256") != second.get("node_ids_sha256"):
            reasons.add("COMPONENT_NODE_HASH_CHANGED")
        if float(match.get("area_weighted_jaccard", math.nan)) != 1.0:
            reasons.add("COMPONENT_JACCARD_CHANGED")
        if float(match.get("medoid_graph_geodesic_displacement_mm", math.nan)) != 0.0:
            reasons.add("COMPONENT_MEDOID_CHANGED")
        if first.get("area_medoid_node_id") != second.get("area_medoid_node_id"):
            reasons.add("COMPONENT_MEDOID_NODE_CHANGED")
        if first.get("higher_energy_opaque_member") != second.get(
            "higher_energy_opaque_member"
        ):
            reasons.add("COMPONENT_ATTRIBUTED_TOKEN_CHANGED")

    central_token = central.get("attributed_opaque_member")
    variant_token = variant.get("attributed_opaque_member")

    if central_token != variant_token:
        reasons.add("ROW_ATTRIBUTED_TOKEN_CHANGED")
    return {
        "row_changed": bool(reasons),
        "row_change_reasons": sorted(reasons),
    }


def _member_swap_contract(
    *,
    central_row: Mapping[str, Any],
    swap_row: Mapping[str, Any],
    central_components: Mapping[str, Any],
    swap_components: Mapping[str, Any],
    atol: float = 1e-12,
) -> dict[str, Any]:
    first_pair = central_row["pair"]
    second_pair = swap_row["pair"]
    checks: dict[str, bool] = {}
    checks["common_support_equal"] = bool(
        np.array_equal(first_pair.get("common"), second_pair.get("common"))
    )
    if first_pair.get("actionable") and second_pair.get("actionable"):
        checks["nuisance_parameters_negate"] = bool(
            len(first_pair["fold_rows"]) == len(second_pair["fold_rows"])
            and all(
                np.allclose(
                    np.asarray(first["parameters"]),
                    -np.asarray(second["parameters"]),
                    rtol=0.0,
                    atol=float(atol),
                )
                for first, second in zip(
                    first_pair["fold_rows"], second_pair["fold_rows"], strict=True
                )
            )
        )
        checks["prediction_negates"] = bool(
            np.allclose(
                first_pair["prediction"], -second_pair["prediction"],
                rtol=0.0, atol=float(atol), equal_nan=True,
            )
        )
        checks["residual_negates"] = bool(
            np.allclose(
                first_pair["residual"], -second_pair["residual"],
                rtol=0.0, atol=float(atol), equal_nan=True,
            )
        )
    else:
        checks["nuisance_parameters_negate"] = first_pair.get("reason") == second_pair.get("reason")
        checks["prediction_negates"] = checks["nuisance_parameters_negate"]
        checks["residual_negates"] = checks["nuisance_parameters_negate"]
    checks["component_node_ids_equal"] = [
        component["node_ids"] for component in central_components["components"]
    ] == [component["node_ids"] for component in swap_components["components"]]
    checks["amplitudes_equal"] = bool(
        np.allclose(
            central_components["amplitude"], swap_components["amplitude"],
            rtol=0.0, atol=float(atol), equal_nan=True,
        )
    )
    checks["active_masks_equal"] = bool(
        np.array_equal(central_components["active"], swap_components["active"])
    )
    checks["areas_equal"] = bool(
        len(central_components["components"]) == len(swap_components["components"])
        and all(
            abs(float(first["area_mm2"]) - float(second["area_mm2"])) <= float(atol)
            for first, second in zip(
                central_components["components"], swap_components["components"], strict=True
            )
        )
    )
    checks["per_component_member_energies_exchange"] = bool(
        len(central_components["components"]) == len(swap_components["components"])
        and all(
            first.get("first_member_energy_mm2") is not None
            and first.get("second_member_energy_mm2") is not None
            and second.get("first_member_energy_mm2") is not None
            and second.get("second_member_energy_mm2") is not None
            and math.isclose(
                float(first["first_member_energy_mm2"]),
                float(second["second_member_energy_mm2"]),
                rel_tol=0.0,
                abs_tol=float(atol),
            )
            and math.isclose(
                float(first["second_member_energy_mm2"]),
                float(second["first_member_energy_mm2"]),
                rel_tol=0.0,
                abs_tol=float(atol),
            )
            for first, second in zip(
                central_components["components"],
                swap_components["components"],
                strict=True,
            )
        )
    )
    central_winner = central_components.get("attributed_opaque_member")
    swap_winner = swap_components.get("attributed_opaque_member")

    def winning_slot(row: Mapping[str, Any], winner: Any) -> str | None:
        if winner is None:
            return None
        if winner == row.get("first_token"):
            return "FIRST"
        if winner == row.get("second_token"):
            return "SECOND"
        return "INVALID"

    central_slot = winning_slot(central_row, central_winner)
    swap_slot = winning_slot(swap_row, swap_winner)
    checks["winning_slot_flips_before_token_remap"] = bool(
        central_winner is not None
        and swap_winner is not None
        and central_slot in ("FIRST", "SECOND")
        and swap_slot in ("FIRST", "SECOND")
        and central_slot != swap_slot
    )
    checks["opaque_token_equal_after_remap"] = bool(
        central_winner is not None
        and swap_winner is not None
        and central_winner == swap_winner
    )
    return {"passed": all(checks.values()), "checks": checks, "atol": float(atol), "rtol": 0.0}


def _normalise_local_diagnostic(diagnostic: Any) -> Mapping[str, Any]:
    if not isinstance(diagnostic, Mapping):
        return {"valid": False, "reason": "NOT_RETAINED"}
    return _jsonable(diagnostic)


def _renderer_member_diagnostic(
    *,
    slot: str,
    diagnostic: Any,
    feature_pre: np.ndarray,
    feature_post: np.ndarray,
) -> Mapping[str, Any]:
    diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
    clusters = list(diagnostic.get("clusters", []))
    local_xyz = list(diagnostic.get("local_support_xyz_mm", []))
    local_sheets = list(diagnostic.get("local_support_sheet_ids", []))
    if not local_xyz and clusters:
        local_xyz = [point for cluster in clusters for point in cluster.get("xyz_mm", [])]
        local_sheets = [sheet for cluster in clusters for sheet in cluster.get("sheet_ids", [])]
    winner = diagnostic.get("winner_cluster_id")
    winner_points = set()
    if winner is not None and 0 <= int(winner) < len(clusters):
        winner_points = {
            tuple(float(value) for value in point)
            for point in clusters[int(winner)].get("xyz_mm", [])
        }
    selected = [
        tuple(float(value) for value in point) in winner_points for point in local_xyz
    ]
    winner_row = (
        clusters[int(winner)]
        if winner is not None and 0 <= int(winner) < len(clusters)
        else None
    )
    occupied_source = diagnostic.get("occupied_cells")
    if occupied_source is None and winner_row is not None:
        occupied_source = winner_row.get("occupied_cells", [])
    occupied = np.zeros((2, 16), dtype=bool)
    for cell in occupied_source or []:
        cell = int(cell)
        if 0 <= cell < 32:
            occupied[cell // 16, cell % 16] = True
    if winner_row is not None:
        plane = winner_row.get("plane_parameters")
        p90 = winner_row.get("plane_fit_p90_mm")
    else:
        plane, p90 = None, None
    plane_array = None if plane is None else np.asarray(plane, dtype=np.float64)
    p90_value = None if p90 is None else float(p90)
    scientific_valid = bool(
        diagnostic.get("valid") is True
        and np.all(np.isfinite(feature_pre))
        and np.all(np.isfinite(feature_post))
        and plane_array is not None
        and plane_array.shape == (3,)
        and np.all(np.isfinite(plane_array))
        and p90_value is not None
        and math.isfinite(p90_value)
    )
    if len(local_sheets) != len(local_xyz):
        raise ValueError("local sheet ownership length mismatch")
    if any(int(sheet) not in (-1, 1) for sheet in local_sheets):
        raise ValueError("local nearest-CAD-valid sheet ownership must be -1 or +1")
    return {
        "slot": slot,
        "valid": scientific_valid,
        "invalid_reason": None if scientific_valid else diagnostic.get("reason", "NONFINITE_OR_INVALID_LOCAL_FIT"),
        "local_support_xyz_mm": local_xyz,
        "local_support_nearest_cad_valid_mask": [True] * len(local_xyz),
        "local_support_selected_mask": selected,
        "local_sheet_ids": local_sheets,
        "height_clusters_mm": [row.get("height_values_mm", []) for row in clusters],
        "winner_cluster_index": winner,
        "runner_cluster_index": diagnostic.get("runner_cluster_id"),
        "footprint_occupancy": occupied.tolist(),
        "plane_coefficients": None if plane_array is None else plane_array.tolist(),
        "plane_p90_mm": p90_value,
        "feature_pre_nuisance": _jsonable(np.asarray(feature_pre, dtype=np.float64)),
        "feature_post_nuisance": _jsonable(np.asarray(feature_post, dtype=np.float64)),
    }


def _selected_node_diagnostics(
    *, fit: Mapping[str, Any], nodes: np.ndarray, node_ids: Sequence[int], local_config: Mapping[str, Any]
) -> Mapping[int, Any]:
    ids = np.asarray(sorted(set(int(value) for value in node_ids)), dtype=np.int64)
    if not len(ids):
        return {}
    _features, _valid, _counts, diagnostics, _selected = SRC.fit_scan_to_nodes(
        nodes=nodes[ids],
        voxel_points=np.asarray(fit["voxel_points"]),
        voxel_normals=np.asarray(fit["voxel_normals"]),
        voxel_keys=np.asarray(fit["voxel_keys"]),
        voxel_regions=np.asarray(fit["voxel_regions"]),
        voxel_sheets=np.asarray(fit["voxel_sheets"]),
        config=local_config,
        keep_diagnostics=True,
    )
    return {int(node_id): diagnostics[index] for index, node_id in enumerate(ids)}


def execute_source(
    *,
    prereg: Mapping[str, Any],
    cad_preflight: Mapping[str, Any],
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    raw_first: np.ndarray,
    normals_first: np.ndarray,
    raw_second: np.ndarray,
    normals_second: np.ndarray,
    transform_first: np.ndarray,
    transform_second: np.ndarray,
    first_token: str,
    second_token: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    points_first, normal_first = SRC.apply_rigid_transform(
        raw_first, normals_first, transform_first
    )
    points_second, normal_second = SRC.apply_rigid_transform(
        raw_second, normals_second, transform_second
    )
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    face_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    topology = CAD.welded_mesh_topology(
        vertices,
        triangles,
        face_normals,
        weld_tolerance_mm=float(prereg["cad_graph"]["coordinate_weld_tolerance_mm"]),
        smooth_angle_deg=float(prereg["cad_graph"]["smooth_region_max_dihedral_deg_inclusive"]),
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.int64)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    smooth_regions = np.asarray(topology["smooth_region_labels"], dtype=np.int64)
    fixed_centroid = np.asarray(
        cad_preflight["cad_metrics"]["cad_interior_area_weighted_centroid_mm"],
        dtype=np.float64,
    )
    graph = CAD.weighted_node_graph(len(nodes), edges, edge_weights)
    fold_masks = CAD.fold_training_masks_with_halo(
        np.column_stack((nodes["x"], nodes["y"], nodes["z"])),
        nodes["smooth_region"],
        nodes["fold_id"],
        graph,
        fold_count=int(prereg["cad_only_actionability"]["fold_count"]),
        halo_mm=float(prereg["cad_only_actionability"]["fold_exclusion_halo_mm_strictly_greater_than"]),
    )
    local = prereg["local_surface_fit"]
    identity_first = SRC.canonical_identity_bits(points_first, normal_first)
    identity_second = SRC.canonical_identity_bits(points_second, normal_second)
    checker_first = SRC.spatial_checker_bits(points_first, 0.75 * float(prereg["immutable_inputs"]["sheet_thickness_mm"]))
    checker_second = SRC.spatial_checker_bits(points_second, 0.75 * float(prereg["immutable_inputs"]["sheet_thickness_mm"]))
    members = {
        "A": (points_first, normal_first, identity_first, checker_first),
        "B": (points_second, normal_second, identity_second, checker_second),
    }
    fit_cache: dict[tuple[Any, ...], Mapping[str, Any]] = {}

    def get_fit(
        member: str,
        *,
        voxel: float,
        partition: tuple[str, int] | None = None,
        translation: tuple[int, float] | None = None,
        rotation: tuple[int, float] | None = None,
        diagnostics: bool = False,
    ) -> Mapping[str, Any]:
        key = (member, float(voxel), partition, translation, rotation, bool(diagnostics))
        if key in fit_cache:
            return fit_cache[key]
        point, normal, identity, checker = members[member]
        mask = None
        if partition is not None:
            bits = identity if partition[0] == "IDENTITY" else checker
            mask = bits == int(partition[1])
        vector = None
        if translation is not None:
            vector = np.zeros(3, dtype=np.float64)
            vector[int(translation[0])] = float(translation[1])
        fit_cache[key] = member_fit_variant(
            raw_points_cad=point,
            raw_normals_cad=normal,
            nodes=nodes,
            scene=scene,
            smooth_region_by_face=smooth_regions,
            local_config=local,
            voxel_mm=float(voxel),
            partition_mask=mask,
            pose_translation=vector,
            pose_rotation=rotation,
            fixed_centroid=fixed_centroid,
            keep_diagnostics=diagnostics,
        )
        return fit_cache[key]

    central_voxel = float(local["central_voxel_mm"])
    central_a = get_fit("A", voxel=central_voxel, diagnostics=True)
    central_b = get_fit("B", voxel=central_voxel, diagnostics=True)
    row_specs: list[tuple[str, Mapping[str, Any], Mapping[str, Any], str, str]] = []
    row_specs.append(("R00", central_a, central_b, first_token, second_token))
    row_specs.append(("R01", get_fit("A", voxel=0.15), get_fit("B", voxel=0.15), first_token, second_token))
    row_specs.append(("R02", get_fit("A", voxel=0.225), get_fit("B", voxel=0.225), first_token, second_token))
    for row_id, family, bit in (
        ("R03", "IDENTITY", 0), ("R04", "IDENTITY", 1),
        ("R05", "SPATIAL", 0), ("R06", "SPATIAL", 1),
    ):
        row_specs.append(
            (
                row_id,
                get_fit("A", voxel=central_voxel, partition=(family, bit)),
                get_fit("B", voxel=central_voxel, partition=(family, bit)),
                first_token,
                second_token,
            )
        )
    translation = float(prereg["robustness_inventory"]["pose_translation_mm"])
    rotation = float(prereg["robustness_inventory"]["pose_rotation_deg"])
    next_row = 7
    for member in ("A", "B"):
        for kind in ("T", "R"):
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    kwargs = (
                        {"translation": (axis, sign * translation)}
                        if kind == "T"
                        else {"rotation": (axis, sign * rotation)}
                    )
                    changed = get_fit(member, voxel=central_voxel, **kwargs)
                    first_fit = changed if member == "A" else central_a
                    second_fit = changed if member == "B" else central_b
                    row_specs.append((f"R{next_row:02d}", first_fit, second_fit, first_token, second_token))
                    next_row += 1
    row_specs.append(("R31", central_b, central_a, second_token, first_token))
    if [row[0] for row in row_specs] != SRC.robustness_row_ids():
        raise RuntimeError("R00-R31 inventory mismatch")
    rows = [
        evaluate_pair_row(
            row_id=row_id,
            first_fit=first_fit,
            second_fit=second_fit,
            nodes=nodes,
            fold_training_masks=fold_masks,
            fixed_centroid=fixed_centroid,
            prereg=prereg,
        )
        | {"first_token": token_a, "second_token": token_b}
        for row_id, first_fit, second_fit, token_a, token_b in row_specs
    ]
    split_rows = rows[3:7]
    if not all(row["pair"].get("actionable") for row in split_rows):
        split_noise = np.full(len(nodes), math.inf)
        split_complete = np.zeros(len(nodes), dtype=bool)
        split_first_support = np.zeros(len(nodes), dtype=bool)
        split_second_support = np.zeros(len(nodes), dtype=bool)
    else:
        split_first_support = np.logical_and.reduce(
            [np.asarray(row["first_valid"], dtype=bool) for row in split_rows]
        )
        split_second_support = np.logical_and.reduce(
            [np.asarray(row["second_valid"], dtype=bool) for row in split_rows]
        )
        split_noise, split_complete = SRC.split_noise_from_rows(
            split_rows[0]["pair"]["residual"],
            split_rows[1]["pair"]["residual"],
            split_rows[2]["pair"]["residual"],
            split_rows[3]["pair"]["residual"],
            [row["pair"]["common"] for row in split_rows],
        )
    split_pass, split_metrics = _support_gate_for_split_complete(
        first_support=split_first_support,
        second_support=split_second_support,
        nodes=nodes,
        fixed_centroid=fixed_centroid,
        pair_config=prereg["pair_actionability"],
    )
    threshold = np.maximum(
        max(0.15, 0.1 * float(prereg["immutable_inputs"]["sheet_thickness_mm"])),
        4.0 * split_noise,
    )
    component_results = []
    for index, row in enumerate(rows):
        component_results.append(
            _row_component_result(
                row=row,
                threshold=threshold,
                split_complete=split_complete,
                nodes=nodes,
                edges=edges,
                edge_weights=edge_weights,
                prereg=prereg,
                first_token=row["first_token"],
                second_token=row["second_token"],
                central=index == 0,
            )
        )
    central = component_results[0]
    robustness = []
    if central["components"]:
        for variant in component_results:
            comparison = _robustness_compare(
                central=central,
                variant=variant,
                nodes=nodes,
                full_graph=graph,
                prereg=prereg,
            )
            variant_index = int(variant["row_id"][1:])
            change = _robustness_row_change(
                central=central,
                variant=variant,
                comparison=comparison,
                central_row=rows[0],
                variant_row=rows[variant_index],
            )
            robustness.append(
                {
                    "row_id": variant["row_id"],
                    "status": variant["status"],
                    "component_count": len(variant["components"]),
                    "component_hashes": [row["node_ids_sha256"] for row in variant["components"]],
                    "attributed_opaque_member": variant.get("attributed_opaque_member"),
                    **comparison,
                    **change,
                }
            )
        swap_contract = _member_swap_contract(
            central_row=rows[0],
            swap_row=rows[31],
            central_components=component_results[0],
            swap_components=component_results[31],
            atol=float(prereg["member_swap_contract"]["nuisance_parameters_negate_atol"]),
        )
        robustness[31]["member_swap_contract"] = swap_contract
        robustness[31]["passed"] = bool(
            robustness[31]["passed"] and swap_contract["passed"]
        )
        if not robustness[31]["passed"]:
            robustness[31]["reason"] = "MEMBER_SWAP_CONTRACT_GATE"
    else:
        every_empty = all(not row["components"] for row in component_results)
        robustness = [
            {
                "row_id": row["row_id"],
                "status": row["status"],
                "component_count": len(row["components"]),
                "component_hashes": [],
                "attributed_opaque_member": None,
                "passed": False,
                "reason": "CENTRAL_EMPTY_TERMINAL_VISUAL_ABSTAIN",
                "matches": [],
                "all_variants_empty_diagnostic": bool(every_empty),
                "row_changed": bool(row["components"]),
                "row_change_reasons": (
                    ["CENTRAL_EMPTY_VARIANT_COMPONENT_BIRTH"]
                    if row["components"] else []
                ),
            }
            for row in component_results
        ]
    robust_pass = bool(robustness and all(row["passed"] for row in robustness))
    if not rows[0]["pair"].get("actionable"):
        status, reason = "ABSTAIN", rows[0]["pair"].get("reason")
    elif not split_pass:
        status, reason = "ABSTAIN", "INCOMPLETE_SPLIT_SUPPORT"
    elif not central["components"]:
        status, reason = "ABSTAIN", "NO_CENTRAL_COMPONENTS"
    elif central["status"] != "ACTIONABLE":
        status, reason = "ABSTAIN", central.get("reason")
    elif not robust_pass:
        status, reason = "ABSTAIN", "ROBUSTNESS_GATE"
    else:
        status, reason = "PAIRWISE_DEVIATION_ATTRIBUTED", None

    # Materialize variant-local evidence only for rows/components that changed
    # or failed.  This second pass is deterministic and uses already-voxelized
    # source arrays; it does not alter any scientific result.
    for row_index, robustness_row in enumerate(robustness):
        robustness_row["row_descriptor"] = prereg["robustness_inventory"]["exact_row_ids"][
            robustness_row["row_id"]
        ]
        robustness_row["variant_components"] = _jsonable(
            component_results[row_index]["components"]
        )
        changed_ids = sorted(
            {
                int(node_id)
                for component in (
                    list(component_results[0]["components"])
                    + list(component_results[row_index]["components"])
                )
                for node_id in component["node_ids"]
            }
        )
        if (
            (not robustness_row["passed"] or robustness_row["row_changed"])
            and changed_ids
        ):
            first_diagnostics = _selected_node_diagnostics(
                fit=rows[row_index]["first_fit"],
                nodes=nodes,
                node_ids=changed_ids,
                local_config=local,
            )
            second_diagnostics = _selected_node_diagnostics(
                fit=rows[row_index]["second_fit"],
                nodes=nodes,
                node_ids=changed_ids,
                local_config=local,
            )
            row_pair = rows[row_index]["pair"]
            row_prediction = np.asarray(
                row_pair.get("prediction", np.full((len(nodes), 3), np.nan)),
                dtype=np.float64,
            )
            robustness_row["changed_component_local_diagnostics"] = [
                {
                    "node_id": node_id,
                    "first_token": rows[row_index]["first_token"],
                    "second_token": rows[row_index]["second_token"],
                    "member_diagnostics": [
                        _renderer_member_diagnostic(
                            slot="FIRST",
                            diagnostic=first_diagnostics.get(node_id),
                            feature_pre=rows[row_index]["first_fit"]["features"][node_id],
                            feature_post=(
                                rows[row_index]["first_fit"]["features"][node_id]
                                - 0.5 * row_prediction[node_id]
                            ),
                        ),
                        _renderer_member_diagnostic(
                            slot="SECOND",
                            diagnostic=second_diagnostics.get(node_id),
                            feature_pre=rows[row_index]["second_fit"]["features"][node_id],
                            feature_post=(
                                rows[row_index]["second_fit"]["features"][node_id]
                                + 0.5 * row_prediction[node_id]
                            ),
                        ),
                    ],
                }
                for node_id in changed_ids
            ]
            robustness_row["member_support"] = [
                {
                    "slot": "FIRST",
                    "opaque_token": rows[row_index]["first_token"],
                    "voxel_xyz_mm": rows[row_index]["first_fit"]["voxel_points"].tolist(),
                    "voxel_selected_mask": rows[row_index]["first_fit"]["voxel_selected"].tolist(),
                    "voxel_nearest_cad_valid_mask": rows[row_index]["first_fit"]["voxel_nearest_cad_valid"].tolist(),
                    "voxel_selected_by_any_winning_cluster_mask": rows[row_index]["first_fit"]["voxel_selected"].tolist(),
                    "voxel_region_ids": rows[row_index]["first_fit"]["voxel_regions"].tolist(),
                    "voxel_sheet_ids": nullable_sheet_ids(
                        rows[row_index]["first_fit"]["voxel_sheets"],
                        rows[row_index]["first_fit"]["voxel_nearest_cad_valid"],
                    ),
                },
                {
                    "slot": "SECOND",
                    "opaque_token": rows[row_index]["second_token"],
                    "voxel_xyz_mm": rows[row_index]["second_fit"]["voxel_points"].tolist(),
                    "voxel_selected_mask": rows[row_index]["second_fit"]["voxel_selected"].tolist(),
                    "voxel_nearest_cad_valid_mask": rows[row_index]["second_fit"]["voxel_nearest_cad_valid"].tolist(),
                    "voxel_selected_by_any_winning_cluster_mask": rows[row_index]["second_fit"]["voxel_selected"].tolist(),
                    "voxel_region_ids": rows[row_index]["second_fit"]["voxel_regions"].tolist(),
                    "voxel_sheet_ids": nullable_sheet_ids(
                        rows[row_index]["second_fit"]["voxel_sheets"],
                        rows[row_index]["second_fit"]["voxel_nearest_cad_valid"],
                    ),
                },
            ]
            robustness_row["changed_visual_components"] = [
                {
                    "component_id": f"V_{robustness_row['row_id']}_C{int(component['component_id']):04d}",
                    "node_ids": component["node_ids"],
                    "area_mm2": component["area_mm2"],
                    "geodesic_span_mm": component["two_sweep_span_mm"],
                    "member_energy_mm2": [
                        component.get("first_member_energy_mm2"),
                        component.get("second_member_energy_mm2"),
                    ],
                    "energy_ratio": component.get("energy_ratio"),
                    "attributed_opaque_member": component.get("higher_energy_opaque_member"),
                    "pairwise_energy_fraction": component.get("pair_energy_fraction"),
                }
                for component in component_results[row_index]["components"]
            ]
        else:
            robustness_row["changed_component_local_diagnostics"] = []
            robustness_row["member_support"] = []
            robustness_row["changed_visual_components"] = []
    central_pair = rows[0]["pair"]
    node_points = np.column_stack((nodes["x"], nodes["y"], nodes["z"]))
    node_normals = np.column_stack((nodes["nx"], nodes["ny"], nodes["nz"]))
    node_first_tangent = np.column_stack((nodes["e1x"], nodes["e1y"], nodes["e1z"]))
    node_second_tangent = np.column_stack((nodes["e2x"], nodes["e2y"], nodes["e2z"]))
    central_diagnostics_a = central_a["diagnostics"]
    central_diagnostics_b = central_b["diagnostics"]
    node_rows = []
    component_by_node = np.asarray(central.get("component_labels", np.full(len(nodes), -1)))
    central_prediction = np.asarray(
        central_pair.get("prediction", np.full((len(nodes), 3), np.nan)),
        dtype=np.float64,
    )
    for node_id in range(len(nodes)):
        feature_a = np.asarray(central_a["features"][node_id], dtype=np.float64)
        feature_b = np.asarray(central_b["features"][node_id], dtype=np.float64)
        post_a = feature_a - 0.5 * central_prediction[node_id]
        post_b = feature_b + 0.5 * central_prediction[node_id]
        component_id = (
            None
            if int(component_by_node[node_id]) < 0
            else f"C{int(component_by_node[node_id]):04d}"
        )
        amplitude_display = float(central["amplitude"][node_id])
        split_display = float(split_noise[node_id])
        threshold_display = float(threshold[node_id])
        node_valid = bool(
            math.isfinite(amplitude_display)
            and math.isfinite(split_display)
            and math.isfinite(threshold_display)
        )
        node_rows.append(
            {
                "node_id": node_id,
                "cad_index": node_id,
                "valid": node_valid,
                "invalid_reason": None if node_valid else "MISSING_SPLIT_OR_CENTRAL_SUPPORT",
                "amplitude_mm": amplitude_display if math.isfinite(amplitude_display) else None,
                "split_noise_mm": split_display if math.isfinite(split_display) else None,
                "threshold_mm": threshold_display if math.isfinite(threshold_display) else None,
                "active": bool(central["active"][node_id]),
                "component_id": component_id,
                "member_diagnostics": [
                    _renderer_member_diagnostic(
                        slot="FIRST", diagnostic=central_diagnostics_a[node_id],
                        feature_pre=feature_a, feature_post=post_a,
                    ),
                    _renderer_member_diagnostic(
                        slot="SECOND", diagnostic=central_diagnostics_b[node_id],
                        feature_pre=feature_b, feature_post=post_b,
                    ),
                ],
            }
        )
    raw_region_a, raw_sheet_a, raw_selected_a = raycast_regions_and_sheets(scene, points_first, smooth_regions)
    raw_region_b, raw_sheet_b, raw_selected_b = raycast_regions_and_sheets(scene, points_second, smooth_regions)
    common_support = np.asarray(central_pair.get("common", np.zeros(len(nodes), dtype=bool))) & split_complete
    halo_display = np.zeros(len(nodes), dtype=bool)
    for fold, mask in enumerate(fold_masks):
        halo_display |= (np.asarray(nodes["fold_id"]) != fold) & ~np.asarray(mask)

    renderer_components = []
    for central_component_id, component in enumerate(central["components"]):
        variant_rows = []
        changed = False
        for robustness_row in robustness:
            match = next(
                (
                    row for row in robustness_row.get("matches", [])
                    if int(row["central_component_id"]) == central_component_id
                ),
                None,
            )
            if match is None:
                changed = True
                variant_rows.append(
                    {
                        "row_id": robustness_row["row_id"],
                        "matched_component_id": None,
                        "jaccard": 0.0,
                        "medoid_displacement_mm": None,
                        "energy_ratio": None,
                        "attributed_slot": None,
                        "passed": False,
                    }
                )
                continue
            variant_component = component_results[
                int(robustness_row["row_id"][1:])
            ]["components"][int(match["variant_component_id"])]
            attributed_token = variant_component.get("higher_energy_opaque_member")
            attributed_slot = (
                "FIRST" if attributed_token == rows[int(robustness_row["row_id"][1:])]["first_token"]
                else "SECOND" if attributed_token == rows[int(robustness_row["row_id"][1:])]["second_token"]
                else None
            )
            row_changed = not bool(match["passed"]) or float(match["area_weighted_jaccard"]) < 1.0
            changed |= (
                row_changed
                or not bool(robustness_row["passed"])
                or bool(robustness_row["row_changed"])
            )
            variant_rows.append(
                {
                    "row_id": robustness_row["row_id"],
                    "matched_component_id": f"C{int(match['variant_component_id']):04d}",
                    "jaccard": float(match["area_weighted_jaccard"]),
                    "medoid_displacement_mm": float(match["medoid_graph_geodesic_displacement_mm"]),
                    "energy_ratio": (
                        float(variant_component["energy_ratio"])
                        if variant_component.get("energy_ratio") is not None
                        else None
                    ),
                    "attributed_slot": attributed_slot,
                    "passed": bool(match["passed"]),
                }
            )
        token = component.get("higher_energy_opaque_member")
        renderer_components.append(
            {
                "component_id": f"C{central_component_id:04d}",
                "node_ids": component["node_ids"],
                "retained": True,
                "robustness_changed": bool(changed),
                "area_mm2": float(component["area_mm2"]),
                "geodesic_span_mm": float(component["two_sweep_span_mm"]),
                "member_energy_mm2": [
                    component.get("first_member_energy_mm2"),
                    component.get("second_member_energy_mm2"),
                ],
                "energy_ratio": component.get("energy_ratio"),
                "attributed_slot": (
                    "FIRST" if token == first_token else "SECOND" if token == second_token else None
                ),
                "pairwise_energy_fraction": component.get("pair_energy_fraction"),
                "variant_rows": variant_rows,
            }
        )
    renderer_components.sort(key=lambda row: row["component_id"])

    visual_payload = {
        "schema": "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1",
        "pair_id": f"PAIR_{first_token}_{second_token}",
        "members": [
            {
                "slot": "FIRST", "opaque_token": first_token,
                "raw_xyz_mm": points_first.tolist(),
                "raw_selected_mask": raw_selected_a.tolist(),
                "raw_nearest_cad_valid_mask": raw_selected_a.tolist(),
                "raw_region_ids": raw_region_a.tolist(),
                "raw_sheet_ids": nullable_sheet_ids(raw_sheet_a, raw_selected_a),
                "voxel_xyz_mm": central_a["voxel_points"].tolist(),
                "voxel_selected_mask": central_a["voxel_selected"].tolist(),
                "voxel_nearest_cad_valid_mask": central_a["voxel_nearest_cad_valid"].tolist(),
                "voxel_selected_by_any_winning_cluster_mask": central_a["voxel_selected"].tolist(),
                "voxel_region_ids": central_a["voxel_regions"].tolist(),
                "voxel_sheet_ids": nullable_sheet_ids(
                    central_a["voxel_sheets"], central_a["voxel_nearest_cad_valid"]
                ),
            },
            {
                "slot": "SECOND", "opaque_token": second_token,
                "raw_xyz_mm": points_second.tolist(),
                "raw_selected_mask": raw_selected_b.tolist(),
                "raw_nearest_cad_valid_mask": raw_selected_b.tolist(),
                "raw_region_ids": raw_region_b.tolist(),
                "raw_sheet_ids": nullable_sheet_ids(raw_sheet_b, raw_selected_b),
                "voxel_xyz_mm": central_b["voxel_points"].tolist(),
                "voxel_selected_mask": central_b["voxel_selected"].tolist(),
                "voxel_nearest_cad_valid_mask": central_b["voxel_nearest_cad_valid"].tolist(),
                "voxel_selected_by_any_winning_cluster_mask": central_b["voxel_selected"].tolist(),
                "voxel_region_ids": central_b["voxel_regions"].tolist(),
                "voxel_sheet_ids": nullable_sheet_ids(
                    central_b["voxel_sheets"], central_b["voxel_nearest_cad_valid"]
                ),
            },
        ],
        "cad": {
            "nodes_xyz_mm": node_points.tolist(),
            "normals": node_normals.tolist(),
            "tangent_e1": node_first_tangent.tolist(),
            "tangent_e2": node_second_tangent.tolist(),
            "smooth_region_ids": nodes["smooth_region"].tolist(),
            "interior_mask": [True] * len(nodes),
            "common_support_mask": common_support.tolist(),
            "excluded_mask": (~common_support).tolist(),
            "active_mask": central["active"].tolist(),
            "edges": edges.tolist(),
            "fold_ids": nodes["fold_id"].tolist(),
            "fold_halo_mask": halo_display.tolist(),
            "area_mm2": nodes["area_mm2"].tolist(),
        },
        "global": {
            "status": status,
            "reason": reason,
            "split_complete_metrics": split_metrics,
            "central_pair": _jsonable(
                {key: value for key, value in central_pair.items() if key not in {"prediction", "residual", "common", "union"}}
            ),
            "attributed_opaque_member": central.get("attributed_opaque_member"),
            "robustness_pass": robust_pass,
            "evaluator_manifest_accessed": False,
            "evaluator_roles_opened": False,
            "labels_opened": False,
            "gold_opened": False,
            "expected_counts_opened": False,
            "quality_identity_opened": False,
        },
        "nodes": node_rows,
        "components": renderer_components,
        "robustness": {"semantic_row_count": 32, "rows": _jsonable(robustness)},
    }
    semantic_robustness = semantic_robustness_rows(robustness)
    semantic = {
        "schema": "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1",
        "experiment_id": EXPECTED_EXPERIMENT,
        "architecture": EXPECTED_ARCHITECTURE,
        "pair_id": f"PAIR_{first_token}_{second_token}",
        "opaque_members": [first_token, second_token],
        "status": status,
        "reason": reason,
        "attributed_opaque_member": central.get("attributed_opaque_member") if status == "PAIRWISE_DEVIATION_ATTRIBUTED" else None,
        "central_component_count": len(central["components"]),
        "central_components": _jsonable(central["components"]),
        "split_complete_metrics": split_metrics,
        "robustness": semantic_robustness,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_tolerance_decision": False,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_OPAQUE_PAIR_DIAGNOSTIC",
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
    }
    return semantic, visual_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--cad-preflight", required=True, type=Path)
    parser.add_argument("--cad-approval", required=True, type=Path)
    parser.add_argument("--source-runner-approval", required=True, type=Path)
    parser.add_argument("--nodes", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--edge-geodesic-mm", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--scan-first", required=True, type=Path)
    parser.add_argument("--scan-second", required=True, type=Path)
    parser.add_argument("--transform-first", required=True, type=Path)
    parser.add_argument("--transform-second", required=True, type=Path)
    parser.add_argument("--source-tests", required=True, type=Path)
    parser.add_argument("--cad-audit-packet", required=True, type=Path)
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument("--renderer-tests", required=True, type=Path)
    parser.add_argument("--orchestration", required=True, type=Path)
    parser.add_argument("--orchestration-tests", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    # Critical ordering: no scan path is hashed or opened before this returns.
    prereg, cad_preflight = verify_pre_scan_approvals(
        preregistration=args.preregistration.resolve(),
        cad_preflight=args.cad_preflight.resolve(),
        cad_approval=args.cad_approval.resolve(),
        source_runner_approval=args.source_runner_approval.resolve(),
        nodes_path=args.nodes.resolve(),
        edges_path=args.edges.resolve(),
        edge_weights_path=args.edge_geodesic_mm.resolve(),
        mesh_path=args.mesh.resolve(),
        source_module_path=Path(SRC.__file__).resolve(),
        runner_path=RUNNER_IDENTITY_PATH,
        source_tests_path=args.source_tests.resolve(),
        cad_audit_packet_path=args.cad_audit_packet.resolve(),
        renderer_path=args.renderer.resolve(),
        renderer_tests_path=args.renderer_tests.resolve(),
        orchestration_path=args.orchestration.resolve(),
        orchestration_tests_path=args.orchestration_tests.resolve(),
        cad_only_runner_path=(ROOT / "scripts" / "run_exp_cad_0029_cad_only_preflight.py").resolve(),
        cad_only_wrapper_path=(ROOT / "scripts" / "run_exp_cad_0030_cad_only_preflight.py").resolve(),
        cad_only_tests_path=(ROOT / "tests" / "test_exp_cad_0029_cfpdg_preflight.py").resolve(),
        cad_identity_tests_path=(ROOT / "tests" / "test_exp_cad_0030_cfpdg_cad_identity.py").resolve(),
    )
    immutable = prereg["immutable_inputs"]
    _require_hash(args.mesh, immutable["accepted_mesh_sha256"], "accepted mesh")
    artifacts = cad_preflight["artifacts"]
    _require_hash(args.nodes, artifacts["nodes"]["sha256"], "CAD nodes")
    _require_hash(args.edges, artifacts["edges"]["sha256"], "CAD edges")
    _require_hash(
        args.edge_geodesic_mm,
        artifacts["edge_geodesic_mm"]["sha256"],
        "CAD edge geodesic weights",
    )
    tokens = list(immutable["opaque_case_tokens_in_canonical_order"])
    scan_hashes = immutable["scan_sha256_without_loading_bytes"]
    transform_hashes = immutable["frozen_transform_semantic_sha256"]
    transform_first = load_transform(
        args.transform_first,
        tokens[0],
        scan_hashes[tokens[0]],
        transform_hashes[tokens[0]],
    )
    transform_second = load_transform(
        args.transform_second,
        tokens[1],
        scan_hashes[tokens[1]],
        transform_hashes[tokens[1]],
    )
    nodes, edges, edge_weights, mesh = load_cad_source_artifacts_before_scan(
        nodes_path=args.nodes,
        edges_path=args.edges,
        edge_weights_path=args.edge_geodesic_mm,
        mesh_path=args.mesh,
    )
    # Approvals, both transforms, and all CAD artifacts are now fully valid.
    # Hashing these two opaque paths is the first physical-scan byte access.
    _require_hash(args.scan_first, scan_hashes[tokens[0]], f"{tokens[0]} scan")
    _require_hash(args.scan_second, scan_hashes[tokens[1]], f"{tokens[1]} scan")
    first_cloud = o3d.io.read_point_cloud(str(args.scan_first.resolve()))
    second_cloud = o3d.io.read_point_cloud(str(args.scan_second.resolve()))
    if not first_cloud.has_points() or not second_cloud.has_points():
        raise ValueError("physical scan is empty")
    if not first_cloud.has_normals() or not second_cloud.has_normals():
        raise ValueError("frozen CF-PDG requires source normals; no estimation fallback")
    raw_first = np.asarray(first_cloud.points, dtype=np.float64)
    normal_first = np.asarray(first_cloud.normals, dtype=np.float64)
    raw_second = np.asarray(second_cloud.points, dtype=np.float64)
    normal_second = np.asarray(second_cloud.normals, dtype=np.float64)
    for label, points, normals in (
        (tokens[0], raw_first, normal_first),
        (tokens[1], raw_second, normal_second),
    ):
        if (
            points.ndim != 2 or points.shape[1] != 3
            or normals.shape != points.shape
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(normals))
        ):
            raise ValueError(f"{label} scan point/normal shape or finiteness gate")
    semantic, visual = execute_source(
        prereg=prereg,
        cad_preflight=cad_preflight,
        nodes=nodes,
        edges=edges,
        edge_weights=edge_weights,
        mesh=mesh,
        raw_first=raw_first,
        normals_first=normal_first,
        raw_second=raw_second,
        normals_second=normal_second,
        transform_first=transform_first,
        transform_second=transform_second,
        first_token=tokens[0],
        second_token=tokens[1],
    )
    bindings = {
        "preregistration_sha256": CAD.sha256_path(args.preregistration),
        "cad_preflight_sha256": CAD.sha256_path(args.cad_preflight),
        "cad_approval_sha256": CAD.sha256_path(args.cad_approval),
        "source_runner_approval_sha256": CAD.sha256_path(args.source_runner_approval),
        "source_module_sha256": CAD.sha256_path(Path(SRC.__file__)),
        "source_runner_sha256": CAD.sha256_path(RUNNER_IDENTITY_PATH),
        "source_runner_base_sha256": CAD.sha256_path(Path(__file__)),
        "source_tests_sha256": CAD.sha256_path(args.source_tests),
        "nodes_sha256": CAD.sha256_path(args.nodes),
        "edges_sha256": CAD.sha256_path(args.edges),
        "edge_geodesic_mm_sha256": CAD.sha256_path(args.edge_geodesic_mm),
        "mesh_sha256": CAD.sha256_path(args.mesh),
        "scan_sha256": {tokens[0]: scan_hashes[tokens[0]], tokens[1]: scan_hashes[tokens[1]]},
        "transform_semantic_sha256": {
            tokens[0]: transform_hashes[tokens[0]],
            tokens[1]: transform_hashes[tokens[1]],
        },
    }
    semantic = dict(semantic) | {
        "bindings": bindings,
        "physical_scan_load_performed": True,
        "physical_scan_hash_call_count": 2,
        "physical_scan_point_cloud_load_call_count": 2,
        "physical_scan_byte_open_operation_count": 4,
    }
    visual = dict(visual)
    visual["global"] = dict(visual["global"]) | {"bindings": bindings}
    args.output_root.mkdir(parents=True, exist_ok=True)
    CAD.atomic_json(args.output_root / "SOURCE_RESULT.json", _jsonable(semantic))
    CAD.atomic_json(
        args.output_root / "SOURCE_VISUAL_PAYLOAD.json", _jsonable(visual)
    )
    print(
        json.dumps(
            {
                "status": semantic["status"],
                "reason": semantic["reason"],
                "central_component_count": semantic["central_component_count"],
                "attributed_opaque_member": semantic["attributed_opaque_member"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
