"""Inference-safe LOCAL_PLANE_FRAGMENT_PROBE_V2 core.

This research-only module consumes frozen compact TRANSITION_ATOM_V2 tables.
It never reads point samples, ownership, solver state, truth, CAD, filenames,
or part/family identity, and it emits no candidate or birth column.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


LOCAL_PLANE_FRAGMENT_PROFILE = "LOCAL_PLANE_FRAGMENT_PROBE_V2"
PROBE_SCHEMA_VERSION = 2
PROBE_KIND = "f1_local_plane_fragment_probe_v2"
ORPHAN_RULE_ID = "INHERITED_HIGH_FIT_SINGLETON_ORPHAN_V1"
ORPHAN_KIND = "INHERITED_HIGH_FIT_SINGLETON"
INPUT_PROJECTION_PROFILE = "LPF_INPUT_PROJECTION_V2"
PLANE_NORMAL_DIAMETER_DEG = 15.0
PLANE_PAIR_MIN_ANGLE_DEG = 20.0
PLANE_PAIR_MAX_PROJECTIVE_ANGLE_DEG = 90.0
MINIMUM_PARENT_COUNT = 3
MINIMUM_AXIAL_SPAN_BINS = 2
MINIMUM_OCCUPIED_BINS = 3
MINIMUM_SIDE_BINS = 3
MINIMUM_TRANSITION_RUNG_BINS = 2
MINIMUM_CONSECUTIVE_RUN = 3
MAXIMUM_EMPTY_BIN_RUN = 1
MAXIMUM_INTERNAL_BRANCH_COUNT = 0
MAXIMUM_TLS_AXIS_ANGLE_DEG = 4.0
GLOBAL_LINE_AXIS_ANGLE_DEG = 5.0
GLOBAL_LINE_TRANSVERSE_OFFSET_MM = 4.0

MAX_INPUT_WORKING_POINTS = 200_000
MAX_TRANSITION_PARENTS = 4_000
MAX_TRANSITION_MODES = 16_000
MAX_TRANSITION_MODE_EDGES = 200_000
MAX_MODES_PER_PARENT = 4
MAX_LOCAL_PLANE_FRAGMENTS = 16_000
MAX_LOCAL_PLANE_PAIRS = 32_768
MAX_LOCAL_TRANSITION_COMPONENTS = 65_536
MAX_COMPONENTS_PER_LOCAL_PAIR = 4_096
MAX_FRAGMENT_WITNESS_MODE_IDS = 8
MAX_FRAGMENT_WITNESS_PARENT_IDS = 8
MAX_COMPONENT_WITNESS_MODE_IDS = 8
MAX_COMPONENT_WITNESS_PARENT_IDS = 8
MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT = 4
MAX_LOCAL_PLANE_ORPHANS = 16_000
MAX_FRAGMENT_PLUS_ORPHAN_RECORDS = 16_000
MAX_ORPHAN_WITNESS_MODE_EDGE_IDS = 8
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3

REJECTION_ORDER = (
    ("minimum_parent_count", "COMPONENT_TOO_SMALL"),
    ("minimum_axial_span", "INSUFFICIENT_AXIAL_SPAN"),
    ("minimum_occupied_bins", "INSUFFICIENT_OCCUPIED_BINS"),
    ("minimum_left_side_bins", "INSUFFICIENT_LEFT_SIDE_BINS"),
    ("minimum_right_side_bins", "INSUFFICIENT_RIGHT_SIDE_BINS"),
    ("minimum_transition_rung_bins", "INSUFFICIENT_TRANSITION_RUNG_BINS"),
    ("minimum_consecutive_occupied_run", "INSUFFICIENT_CONSECUTIVE_BIN_RUN"),
    ("maximum_empty_bin_run", "EXCESSIVE_EMPTY_BIN_RUN"),
    ("no_internal_branching", "INTERNAL_BRANCHING"),
    ("tls_axis", "TLS_AXIS_MISMATCH"),
    ("line_residual", "LINE_RESIDUAL_EXCEEDED"),
    ("transition_certificate_complete", "MODE_OVERFLOW_CERTIFICATE_DISABLED"),
    ("certified_three_bin_window", "NO_CERTIFIED_THREE_BIN_WINDOW"),
)

FORBIDDEN_KEYS = frozenset(
    {
        "point_indices", "source_indices", "source_point_indices", "vertex_indices",
        "points", "source_points", "point_coordinates", "coordinates", "bbox_points",
        "gold", "manual_truth", "expected_count", "requested_count", "cad", "drawing",
        "filename", "scan_path", "part_id", "part_key", "family_id", "split_group_id",
        "atom_owners", "parent_owners", "mode_side_assignments", "solver", "objective",
        "reduced_cost", "birth_columns",
    }
)

HARD_CAPS = {
    "input_working_point_cap": MAX_INPUT_WORKING_POINTS,
    "transition_parent_cap": MAX_TRANSITION_PARENTS,
    "transition_mode_cap": MAX_TRANSITION_MODES,
    "transition_mode_edge_cap": MAX_TRANSITION_MODE_EDGES,
    "modes_per_parent_cap": MAX_MODES_PER_PARENT,
    "local_plane_fragment_cap": MAX_LOCAL_PLANE_FRAGMENTS,
    "local_plane_pair_cap": MAX_LOCAL_PLANE_PAIRS,
    "local_transition_component_cap": MAX_LOCAL_TRANSITION_COMPONENTS,
    "components_per_local_pair_cap": MAX_COMPONENTS_PER_LOCAL_PAIR,
    "fragment_witness_mode_ids_cap": MAX_FRAGMENT_WITNESS_MODE_IDS,
    "fragment_witness_parent_ids_cap": MAX_FRAGMENT_WITNESS_PARENT_IDS,
    "component_witness_mode_ids_cap": MAX_COMPONENT_WITNESS_MODE_IDS,
    "component_witness_parent_ids_cap": MAX_COMPONENT_WITNESS_PARENT_IDS,
    "global_plane_match_ids_per_fragment_cap": MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT,
    "local_plane_orphan_cap": MAX_LOCAL_PLANE_ORPHANS,
    "fragment_plus_orphan_record_cap": MAX_FRAGMENT_PLUS_ORPHAN_RECORDS,
    "orphan_witness_mode_edge_ids_cap": MAX_ORPHAN_WITNESS_MODE_EDGE_IDS,
}

TRANSITION_ATOM_CAPS = {
    "input_point_cap": MAX_INPUT_WORKING_POINTS,
    "parent_cap": MAX_TRANSITION_PARENTS,
    "mode_cap": MAX_TRANSITION_MODES,
    "mode_children_per_parent_cap": MAX_MODES_PER_PARENT,
    "parent_plus_mode_cap": 20_000,
    "mode_edge_cap": MAX_TRANSITION_MODE_EDGES,
}

ALLOWED_POOL_PROJECTION_FIELDS = (
    "support_representation_profile",
    "source_point_serialization",
    "support_parent_table",
    "support_mode_table",
    "support_mode_edges",
    "plane_table",
    "transition_parent_table_sha256",
    "transition_mode_table_sha256",
    "transition_mode_edge_table_sha256",
    "transition_local_spacing_mm",
    "transition_atom_caps",
    "working_point_budget",
    "transient_sampled_point_count",
)

ORPHAN_ROLE_CONTRACT = {
    "local_pair_seed_allowed": False,
    "component_membership_allowed": False,
    "parent_mass_contribution_allowed": False,
    "global_plane_coverage_allowed": False,
    "transition_certificate_allowed": False,
}

GLOBAL_PLANE_COVERAGE_FIELDS = {
    "coverage_state", "best_global_plane_id", "best_mode_match_count",
    "best_mode_coverage_fraction", "best_parent_match_count",
    "best_parent_coverage_fraction", "best_sum_dimensionless_residual",
    "full_coverage_plane_ids", "partial_coverage_plane_ids", "match_ids_truncated",
}
FRAGMENT_RECORD_FIELDS = {
    "local_plane_fragment_id", "normal_projective", "offset", "centroid",
    "tangent_basis", "centroid_extent_uv", "mode_count", "parent_count",
    "point_count_sum", "normalized_parent_mass_sum", "same_surface_edge_count",
    "normal_diameter_deg", "max_mode_to_fragment_normal_deg", "surface_fit_p90_mm",
    "mode_overflow_parent_count", "transition_certificate_allowed",
    "member_mode_ids_sha256", "member_parent_ids_sha256", "witness_mode_ids",
    "witness_parent_ids", "witness_ids_truncated", "global_plane_coverage",
    "fragment_sha256",
}
ORPHAN_RECORD_FIELDS = {
    "local_plane_orphan_id", "orphan_kind", "orphan_rule_id", "mode_atom_id",
    "parent_atom_id", "mode_point_count", "inherited_mode_surface_fit_p90_mm",
    "computed_singleton_surface_fit_p90_mm", "plane_fit_tolerance_mm",
    "excess_over_tolerance_mm", "parent_mode_overflow",
    "parent_transition_certificate_allowed", "incident_same_surface_edge_count",
    "incident_cross_mode_edge_count", "incident_mode_edge_ids_sha256",
    "witness_mode_edge_ids", "witness_ids_truncated", "member_mode_ids_sha256",
    "member_parent_ids_sha256", "role_contract", "orphan_sha256",
}
GLOBAL_PAIR_COMPARISON_FIELDS = {
    "state", "global_plane_ids", "total_dimensionless_residual",
}
COMPONENT_RECORD_FIELDS = {
    "local_transition_component_id", "support_parent_count", "support_mode_count",
    "normalized_parent_mass", "support_parent_ids_sha256", "support_mode_ids_sha256",
    "witness_parent_ids", "witness_mode_ids", "witness_ids_truncated",
    "intersection_span_endpoints", "occupied_bin_count", "left_side_bin_count",
    "right_side_bin_count", "transition_rung_bin_count",
    "longest_consecutive_occupied_run", "maximum_empty_bin_run",
    "internal_branch_count", "tls_axis_angle_deg", "line_p90_mm",
    "certified_three_bin_window_count", "mode_overflow_parent_count",
    "geometry_gate_depth", "geometry_gates", "first_rejection_code",
    "geometry_certificate_valid", "global_pair_line_state",
    "global_pair_axis_angle_deg", "global_pair_transverse_offset_mm", "component_sha256",
}
PAIR_RECORD_FIELDS = {
    "local_plane_pair_id", "fragment_ids", "plane_angle_deg", "intersection_line_hash",
    "intersection_anchor", "intersection_axis", "transverse_parent_count",
    "left_mode_count", "right_mode_count", "cross_mode_rung_count", "component_count",
    "global_pair_comparison", "pair_rejection_counts", "components", "pair_sha256",
}
INPUT_CONTRACT_ATTESTATION_FIELDS = {
    "projection_profile", "support_representation_profile", "source_point_serialization",
    "transition_atom_caps", "working_point_budget", "transient_sampled_point_count",
    "transition_parent_count", "transition_mode_count", "transition_mode_edge_count",
    "transition_parent_plus_mode_count", "maximum_observed_modes_per_parent",
    "table_hashes", "projected_input_sha256", "all_inherited_caps_satisfied",
    "input_contract_attestation_sha256",
}
TOP_LEVEL_FIELDS = {
    "schema_version", "kind", "profile", "research_only", "inference_safe",
    "prediction_mutated", "birth_columns_enabled", "ownership_consumed",
    "solver_consumed", "truth_inputs_consumed", "source_points_consumed",
    "source_points_serialized", "scan_sha256", "candidate_pool_sha256",
    "a0_prediction_sha256", "input_table_hashes", "output_table_hashes",
    "engineering_profile_sha256", "inference_semantic_sha256",
    "input_contract_attestation", "thresholds", "hard_caps", "case_counters",
    "local_plane_fragment_table", "local_plane_orphan_table", "local_plane_pair_records",
}


@dataclass(frozen=True)
class _Fragment:
    record: dict[str, Any]
    mode_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    full_global_matches: tuple[tuple[str, float], ...]


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _table_sha256(records: Sequence[Mapping[str, Any]], id_field: str | None = None) -> str:
    ordered = list(records)
    if id_field is not None:
        ordered = sorted(ordered, key=lambda item: str(item[id_field]))
    return _sha256_json(ordered)


def _validate_sha256(label: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or text.lower() != text or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{label} must be a lowercase 64-character sha256")
    return text


def reject_forbidden_fields(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"LPF probe forbids {path}.{key}")
            reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, child in enumerate(payload):
            reject_forbidden_fields(child, f"{path}[{index}]")


def project_lpf_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in ALLOWED_POOL_PROJECTION_FIELDS if field not in raw_pool]
    if missing:
        raise ValueError(f"LPF input projection missing fields: {missing}")
    projection = {
        field: deepcopy(raw_pool[field]) for field in ALLOWED_POOL_PROJECTION_FIELDS
    }
    if set(projection) != set(ALLOWED_POOL_PROJECTION_FIELDS):
        raise AssertionError("LPF input projection schema mismatch")
    return projection


def _canonical_projected_input_for_hash(
    projected_input: Mapping[str, Any]
) -> dict[str, Any]:
    canonical = deepcopy(dict(projected_input))
    for field, id_field in (
        ("support_parent_table", "parent_atom_id"),
        ("support_mode_table", "mode_atom_id"),
        ("support_mode_edges", "mode_edge_id"),
    ):
        canonical[field] = sorted(
            (dict(item) for item in canonical[field]),
            key=lambda item: str(item[id_field]),
        )
    canonical["plane_table"] = sorted(
        (dict(item) for item in canonical["plane_table"]),
        key=lambda item: str(item.get("canonical_plane_id") or item.get("plane_id") or ""),
    )
    return canonical


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    result = int(value)
    if float(value) != float(result) or result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _vector(value: Any, label: str, *, allow_zero: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite three-vector")
    if not allow_zero and float(np.linalg.norm(result)) <= 1e-12:
        raise ValueError(f"{label} must be nonzero")
    return result


def _unit(value: Any) -> np.ndarray:
    vector = _vector(value, "direction")
    return vector / float(np.linalg.norm(vector))


def canonical_projective_normal(value: Any) -> np.ndarray:
    normal = _unit(value)
    for component in normal:
        if abs(float(component)) > 1e-12:
            if component < 0.0:
                normal = -normal
            break
    normal[np.abs(normal) <= 1e-12] = 0.0
    return normal


def projective_angle_deg(left: Any, right: Any) -> float:
    cosine = abs(float(np.dot(_unit(left), _unit(right))))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _canonical_plane(record: Mapping[str, Any]) -> tuple[np.ndarray, float, str]:
    raw = _unit(record.get("normal"))
    normal = canonical_projective_normal(raw)
    offset = _finite(record.get("offset", 0.0), "plane offset")
    if float(np.dot(raw, normal)) < 0.0:
        offset = -offset
    plane_id = str(record.get("canonical_plane_id") or record.get("plane_id") or "")
    if not plane_id:
        raise ValueError("global planes require an ID")
    return normal, offset, plane_id


def _weighted_quantile(
    values: Sequence[tuple[float, str, int]], fraction: float
) -> float:
    ordered = sorted(values, key=lambda item: (item[0], item[1]))
    total = sum(weight for _value, _id, weight in ordered)
    if total <= 0:
        raise ValueError("weighted quantile requires positive integer weight")
    target = int(math.ceil(float(fraction) * total))
    cumulative = 0
    for value, _record_id, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(ordered[-1][0])


def _fit_fragment(
    mode_ids: Sequence[str],
    modes: Mapping[str, Mapping[str, Any]],
    *,
    known_normal_diameter_deg: float | None = None,
) -> dict[str, Any]:
    records = [modes[mode_id] for mode_id in sorted(mode_ids)]
    moment = np.zeros((3, 3), dtype=np.float64)
    for mode in records:
        normal = _unit(mode["normal_projective"])
        weight = max(1, _nonnegative_int(mode.get("point_count", 0), "mode point_count"))
        moment += float(weight) * np.outer(normal, normal)
    _values, vectors = np.linalg.eigh(moment)
    normal = canonical_projective_normal(vectors[:, -1])
    offsets = []
    for mode in records:
        mode_id = str(mode["mode_atom_id"])
        weight = max(1, int(mode["point_count"]))
        value = -float(np.dot(normal, _vector(mode["centroid"], "mode centroid", allow_zero=True)))
        offsets.append((value, mode_id, weight))
    offset = _weighted_quantile(offsets, 0.5)
    residuals = []
    for mode in records:
        mode_id = str(mode["mode_atom_id"])
        centroid = _vector(mode["centroid"], "mode centroid", allow_zero=True)
        residual = abs(float(np.dot(normal, centroid)) + offset) + _finite(
            mode.get("surface_fit_p90_mm", 0.0), "mode surface_fit_p90_mm"
        )
        residuals.append((residual, mode_id, max(1, int(mode["point_count"]))))
    diameter = (
        float(known_normal_diameter_deg)
        if known_normal_diameter_deg is not None
        else max(
            (
                projective_angle_deg(left["normal_projective"], right["normal_projective"])
                for index, left in enumerate(records)
                for right in records[index + 1 :]
            ),
            default=0.0,
        )
    )
    weights = np.asarray([max(1, int(mode["point_count"])) for mode in records], dtype=np.float64)
    centroids = np.asarray([mode["centroid"] for mode in records], dtype=np.float64)
    centroid = np.average(centroids, axis=0, weights=weights)
    return {
        "normal": normal,
        "offset": offset,
        "centroid": centroid,
        "normal_diameter_deg": diameter,
        "max_mode_to_fragment_normal_deg": max(
            projective_angle_deg(mode["normal_projective"], normal) for mode in records
        ),
        "surface_fit_p90_mm": _weighted_quantile(residuals, 0.9),
    }


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ normal)))]
    first = canonical_projective_normal(np.cross(normal, reference))
    second = _unit(np.cross(normal, first))
    return first, second


def _round_vector(value: Any, places: int = 12) -> list[float]:
    return [round(float(component), places) for component in value]


def _record_hash(record: Mapping[str, Any], hash_field: str) -> str:
    return _sha256_json({key: value for key, value in record.items() if key != hash_field})


def _global_coverage(
    mode_ids: Sequence[str],
    modes: Mapping[str, Mapping[str, Any]],
    planes: Sequence[Mapping[str, Any]],
    plane_fit_tolerance_mm: float,
) -> tuple[dict[str, Any], tuple[tuple[str, float], ...]]:
    matches = []
    mode_count = len(mode_ids)
    for plane in sorted(planes, key=lambda item: _canonical_plane(item)[2]):
        normal, offset, plane_id = _canonical_plane(plane)
        matched = []
        residual_sum = 0.0
        parents = set()
        for mode_id in sorted(mode_ids):
            mode = modes[mode_id]
            normal_residual = projective_angle_deg(mode["normal_projective"], normal) / 15.0
            distance_residual = abs(
                float(np.dot(normal, _vector(mode["centroid"], "mode centroid", allow_zero=True)))
                + offset
            ) / plane_fit_tolerance_mm
            if normal_residual <= 1.0 + 1e-12 and distance_residual <= 1.0 + 1e-12:
                matched.append(mode_id)
                parents.add(str(mode["parent_atom_id"]))
                residual_sum += normal_residual + distance_residual
        matches.append((plane_id, len(matched), len(parents), residual_sum))
    matches.sort(key=lambda item: (-item[1], item[3], item[0]))
    best = matches[0] if matches else (None, 0, 0, 0.0)
    best_id, best_modes, best_parents, best_residual = best
    fragment_parent_count = len({str(modes[mode_id]["parent_atom_id"]) for mode_id in mode_ids})
    state = "NONE" if best_modes == 0 else ("FULL" if best_modes == mode_count else "PARTIAL")
    full = tuple(sorted((plane_id, residual) for plane_id, count, _parents, residual in matches if count == mode_count))
    partial = tuple(sorted((plane_id, residual) for plane_id, count, _parents, residual in matches if 0 < count < mode_count))
    record = {
        "coverage_state": state,
        "best_global_plane_id": best_id if best_modes else None,
        "best_mode_match_count": int(best_modes),
        "best_mode_coverage_fraction": round(best_modes / max(mode_count, 1), 8),
        "best_parent_match_count": int(best_parents),
        "best_parent_coverage_fraction": round(best_parents / max(fragment_parent_count, 1), 8),
        "best_sum_dimensionless_residual": round(best_residual, 8) if best_modes else None,
        "full_coverage_plane_ids": [item[0] for item in full[:MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT]],
        "partial_coverage_plane_ids": [item[0] for item in partial[:MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT]],
        "match_ids_truncated": bool(
            len(full) > MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT
            or len(partial) > MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT
        ),
    }
    return record, full


def _validate_inputs(candidate_pool: Mapping[str, Any]) -> dict[str, Any]:
    if set(candidate_pool) != set(ALLOWED_POOL_PROJECTION_FIELDS):
        raise ValueError("LPF core requires the exact allowlisted input projection")
    if candidate_pool.get("support_representation_profile") != "TRANSITION_ATOM_V2":
        raise ValueError("LPF input must use TRANSITION_ATOM_V2")
    if candidate_pool.get("source_point_serialization") is not False:
        raise ValueError("LPF input must attest source_point_serialization=false")
    parents = [dict(item) for item in candidate_pool.get("support_parent_table") or ()]
    modes = [dict(item) for item in candidate_pool.get("support_mode_table") or ()]
    edges = [dict(item) for item in candidate_pool.get("support_mode_edges") or ()]
    planes = [dict(item) for item in candidate_pool.get("plane_table") or ()]
    reject_forbidden_fields(
        {
            "support_parent_table": parents,
            "support_mode_table": modes,
            "support_mode_edges": edges,
            "plane_table": planes,
        }
    )
    if not parents or not modes:
        raise ValueError("LPF probe requires nonempty frozen V2 parent and mode tables")
    if len(parents) > MAX_TRANSITION_PARENTS:
        raise ValueError("transition parent cap breached")
    if len(modes) > MAX_TRANSITION_MODES:
        raise ValueError("transition mode cap breached")
    if len(edges) > MAX_TRANSITION_MODE_EDGES:
        raise ValueError("transition mode edge cap breached")
    if len(parents) + len(modes) > 20_000:
        raise ValueError("transition parent-plus-mode cap breached")
    if dict(candidate_pool.get("transition_atom_caps") or {}) != TRANSITION_ATOM_CAPS:
        raise ValueError("frozen TRANSITION_ATOM_V2 cap profile mismatch")
    working_count = _nonnegative_int(
        candidate_pool.get("transient_sampled_point_count", 0), "transient sampled point count"
    )
    working_budget = _nonnegative_int(
        candidate_pool.get("working_point_budget", 0), "working point budget"
    )
    if working_budget != MAX_INPUT_WORKING_POINTS:
        raise ValueError("LPF working_point_budget must equal 200000")
    if working_count > working_budget:
        raise ValueError("inherited working-point count exceeds budget")
    spacing = _finite(candidate_pool.get("transition_local_spacing_mm", 0.0), "transition local spacing")
    if spacing <= 0.0:
        raise ValueError("transition local spacing must be positive")

    claimed = {
        "parent_table_sha256": _validate_sha256(
            "transition parent table hash", candidate_pool.get("transition_parent_table_sha256")
        ),
        "mode_table_sha256": _validate_sha256(
            "transition mode table hash", candidate_pool.get("transition_mode_table_sha256")
        ),
        "mode_edge_table_sha256": _validate_sha256(
            "transition mode edge table hash", candidate_pool.get("transition_mode_edge_table_sha256")
        ),
    }
    actual = {
        "parent_table_sha256": _table_sha256(parents, "parent_atom_id"),
        "mode_table_sha256": _table_sha256(modes, "mode_atom_id"),
        "mode_edge_table_sha256": _table_sha256(edges, "mode_edge_id"),
    }
    if claimed != actual:
        raise ValueError("frozen V2 table hash mismatch")

    parent_by_id = {}
    for parent in parents:
        parent_id = str(parent.get("parent_atom_id") or "")
        if not parent_id or parent_id in parent_by_id:
            raise ValueError("transition parent IDs must be unique and nonempty")
        _vector(parent.get("centroid"), "parent centroid", allow_zero=True)
        point_count = _nonnegative_int(parent.get("point_count", 0), "parent point_count")
        mass = _nonnegative_int(parent.get("normalized_support_mass", 0), "parent mass")
        if point_count <= 0 or mass < 1 or mass > 4:
            raise ValueError("transition parent point count/mass is invalid")
        children = [str(value) for value in parent.get("mode_child_ids") or ()]
        if len(children) > MAX_MODES_PER_PARENT or len(set(children)) != len(children):
            raise ValueError("transition parent child-mode cap/uniqueness violated")
        parent_by_id[parent_id] = parent
    mode_by_id = {}
    modes_by_parent: dict[str, list[str]] = defaultdict(list)
    for mode in modes:
        mode_id = str(mode.get("mode_atom_id") or "")
        parent_id = str(mode.get("parent_atom_id") or "")
        if not mode_id or mode_id in mode_by_id or parent_id not in parent_by_id:
            raise ValueError("transition mode IDs/parent references are invalid")
        _vector(mode.get("centroid"), "mode centroid", allow_zero=True)
        _vector(mode.get("normal_projective"), "mode normal")
        if _nonnegative_int(mode.get("point_count", 0), "mode point_count") <= 0:
            raise ValueError("mode point_count must be positive")
        if _finite(mode.get("surface_fit_p90_mm", 0.0), "mode surface fit") < 0.0:
            raise ValueError("mode surface fit must be nonnegative")
        mode_by_id[mode_id] = mode
        modes_by_parent[parent_id].append(mode_id)
    for parent_id, mode_ids in modes_by_parent.items():
        if len(mode_ids) > MAX_MODES_PER_PARENT:
            raise ValueError("modes per physical parent exceed cap")
        declared = {str(value) for value in parent_by_id[parent_id].get("mode_child_ids") or ()}
        if declared != set(mode_ids):
            raise ValueError("parent mode_child_ids do not match mode table")
    edge_by_id = {}
    for edge in edges:
        edge_id = str(edge.get("mode_edge_id") or "")
        pair = tuple(str(value) for value in edge.get("mode_ids") or ())
        kind = str(edge.get("edge_kind") or "")
        if not edge_id or edge_id in edge_by_id or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("transition edge IDs/endpoints are invalid")
        if not set(pair) <= set(mode_by_id):
            raise ValueError("transition edge references absent mode")
        if kind not in {"SAME_SURFACE", "CROSS_MODE_LOCAL"}:
            raise ValueError("transition edge kind is invalid")
        for field in ("normal_residual", "plane_residual", "spatial_residual"):
            if _finite(edge.get(field, 0.0), f"edge {field}") < 0.0:
                raise ValueError("transition edge residuals must be nonnegative")
        edge_by_id[edge_id] = edge
    plane_ids = set()
    for plane in planes:
        _normal, _offset, plane_id = _canonical_plane(plane)
        if plane_id in plane_ids:
            raise ValueError("global plane IDs must be unique")
        plane_ids.add(plane_id)
    return {
        "parents": parents,
        "modes": modes,
        "edges": edges,
        "planes": planes,
        "parent_by_id": parent_by_id,
        "mode_by_id": mode_by_id,
        "edge_by_id": edge_by_id,
        "spacing": spacing,
        "working_count": working_count,
        "input_hashes": claimed,
    }


def _input_contract_attestation(
    projected_input: Mapping[str, Any], validated: Mapping[str, Any]
) -> dict[str, Any]:
    parents = validated["parents"]
    modes = validated["modes"]
    edges = validated["edges"]
    planes = validated["planes"]
    modes_per_parent: Counter[str] = Counter(str(mode["parent_atom_id"]) for mode in modes)
    table_hashes = {
        **dict(validated["input_hashes"]),
        "global_plane_table_sha256": _table_sha256(
            sorted(planes, key=lambda plane: _canonical_plane(plane)[2])
        ),
    }
    record = {
        "projection_profile": INPUT_PROJECTION_PROFILE,
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "transition_atom_caps": dict(TRANSITION_ATOM_CAPS),
        "working_point_budget": int(projected_input["working_point_budget"]),
        "transient_sampled_point_count": int(projected_input["transient_sampled_point_count"]),
        "transition_parent_count": len(parents),
        "transition_mode_count": len(modes),
        "transition_mode_edge_count": len(edges),
        "transition_parent_plus_mode_count": len(parents) + len(modes),
        "maximum_observed_modes_per_parent": max(modes_per_parent.values(), default=0),
        "table_hashes": table_hashes,
        "projected_input_sha256": _sha256_json(
            _canonical_projected_input_for_hash(projected_input)
        ),
        "all_inherited_caps_satisfied": bool(
            len(parents) <= MAX_TRANSITION_PARENTS
            and len(modes) <= MAX_TRANSITION_MODES
            and len(edges) <= MAX_TRANSITION_MODE_EDGES
            and len(parents) + len(modes) <= 20_000
            and max(modes_per_parent.values(), default=0) <= MAX_MODES_PER_PARENT
            and 0 <= int(projected_input["transient_sampled_point_count"])
            <= int(projected_input["working_point_budget"])
            == MAX_INPUT_WORKING_POINTS
        ),
    }
    record["input_contract_attestation_sha256"] = _record_hash(
        record, "input_contract_attestation_sha256"
    )
    return record


def build_input_contract_attestation(projected_input: Mapping[str, Any]) -> dict[str, Any]:
    validated = _validate_inputs(projected_input)
    return _input_contract_attestation(projected_input, validated)


def _build_fragments(
    *,
    parents: Mapping[str, Mapping[str, Any]],
    modes: Mapping[str, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    planes: Sequence[Mapping[str, Any]],
    plane_fit_tolerance_mm: float,
    counters: Counter[str],
) -> tuple[list[_Fragment], list[dict[str, Any]], dict[str, str], set[str]]:
    roots = {mode_id: mode_id for mode_id in modes}
    members = {mode_id: {mode_id} for mode_id in modes}
    diameters = {mode_id: 0.0 for mode_id in modes}
    versions = {mode_id: 0 for mode_id in modes}
    rejected_pairs: dict[tuple[str, int, str, int], str] = {}

    def find(mode_id: str) -> str:
        while roots[mode_id] != mode_id:
            roots[mode_id] = roots[roots[mode_id]]
            mode_id = roots[mode_id]
        return mode_id

    same_edges = sorted(
        (edge for edge in edges if str(edge["edge_kind"]) == "SAME_SURFACE"),
        key=lambda edge: (
            round(float(edge.get("normal_residual", 0.0)), 12),
            round(float(edge.get("plane_residual", 0.0)), 12),
            round(float(edge.get("spatial_residual", 0.0)), 12),
            str(edge["mode_edge_id"]),
        ),
    )
    for edge in same_edges:
        left, right = (str(value) for value in edge["mode_ids"])
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        counters["plane_union_attempt_count"] += 1
        cache_key = (left_root, versions[left_root], right_root, versions[right_root])
        cached_rejection = rejected_pairs.get(cache_key)
        if cached_rejection is not None:
            counters[cached_rejection] += 1
            continue
        proposed = sorted(members[left_root] | members[right_root])
        cross_diameter = max(
            projective_angle_deg(
                modes[left_mode]["normal_projective"], modes[right_mode]["normal_projective"]
            )
            for left_mode in members[left_root]
            for right_mode in members[right_root]
        )
        proposed_diameter = max(
            diameters[left_root], diameters[right_root], cross_diameter
        )
        if proposed_diameter > PLANE_NORMAL_DIAMETER_DEG + 1e-12:
            counters["plane_union_reject_normal_diameter_count"] += 1
            rejected_pairs[cache_key] = "plane_union_reject_normal_diameter_count"
            continue
        fit = _fit_fragment(
            proposed, modes, known_normal_diameter_deg=proposed_diameter
        )
        if fit["surface_fit_p90_mm"] > plane_fit_tolerance_mm + 1e-12:
            counters["plane_union_reject_fit_count"] += 1
            rejected_pairs[cache_key] = "plane_union_reject_fit_count"
            continue
        canonical, absorbed = sorted((left_root, right_root))
        roots[absorbed] = canonical
        roots[canonical] = canonical
        members[canonical] = set(proposed)
        diameters[canonical] = proposed_diameter
        versions[canonical] = max(versions[left_root], versions[right_root]) + 1
        del members[absorbed]
        diameters.pop(absorbed, None)
        versions.pop(absorbed, None)
        counters["plane_union_accept_count"] += 1

    final_sets: dict[str, set[str]] = defaultdict(set)
    for mode_id in sorted(modes):
        final_sets[find(mode_id)].add(mode_id)
    same_edge_count_by_root: Counter[str] = Counter()
    for edge in same_edges:
        left, right = (str(value) for value in edge["mode_ids"])
        root = find(left)
        if root == find(right):
            same_edge_count_by_root[root] += 1
    fragments = []
    orphans: list[dict[str, Any]] = []
    orphan_mode_ids: set[str] = set()
    fragment_of_mode = {}
    incident_edges: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        for mode_id in edge["mode_ids"]:
            incident_edges[str(mode_id)].append(edge)
    ordered_components = sorted(
        ((root, tuple(sorted(values))) for root, values in final_sets.items()),
        key=lambda item: item[1],
    )
    for root, mode_ids in ordered_components:
        if len(fragments) >= MAX_LOCAL_PLANE_FRAGMENTS:
            raise ValueError("local plane fragment cap breached")
        fit = _fit_fragment(
            mode_ids, modes, known_normal_diameter_deg=diameters[root]
        )
        if fit["normal_diameter_deg"] > PLANE_NORMAL_DIAMETER_DEG + 1e-12:
            raise AssertionError("final fragment exceeds normal diameter")
        fit_mm = float(fit["surface_fit_p90_mm"])
        tolerance_mm = float(plane_fit_tolerance_mm)
        if not math.isfinite(fit_mm):
            raise ValueError("non-finite final fragment fit")
        if len(mode_ids) > 1 and fit_mm > tolerance_mm:
            raise ValueError("multi-mode final component exceeds plane-fit tolerance")
        if len(mode_ids) == 1 and fit_mm > tolerance_mm:
            mode_id = mode_ids[0]
            mode = modes[mode_id]
            parent_id = str(mode["parent_atom_id"])
            parent = parents[parent_id]
            full_edge_ids = tuple(
                sorted(str(edge["mode_edge_id"]) for edge in incident_edges.get(mode_id, ()))
            )
            orphan_id = "LPO_" + _sha256_json(
                [LOCAL_PLANE_FRAGMENT_PROFILE, ORPHAN_RULE_ID, mode_id, parent_id]
            )[:16]
            orphan_record = {
                "local_plane_orphan_id": orphan_id,
                "orphan_kind": ORPHAN_KIND,
                "orphan_rule_id": ORPHAN_RULE_ID,
                "mode_atom_id": mode_id,
                "parent_atom_id": parent_id,
                "mode_point_count": int(mode["point_count"]),
                "inherited_mode_surface_fit_p90_mm": float(
                    mode["surface_fit_p90_mm"]
                ),
                "computed_singleton_surface_fit_p90_mm": fit_mm,
                "plane_fit_tolerance_mm": tolerance_mm,
                "excess_over_tolerance_mm": fit_mm - tolerance_mm,
                "parent_mode_overflow": bool(parent.get("mode_overflow", False)),
                "parent_transition_certificate_allowed": bool(
                    parent.get("transition_certificate_allowed", True)
                ),
                "incident_same_surface_edge_count": sum(
                    str(edge["edge_kind"]) == "SAME_SURFACE"
                    for edge in incident_edges.get(mode_id, ())
                ),
                "incident_cross_mode_edge_count": sum(
                    str(edge["edge_kind"]) == "CROSS_MODE_LOCAL"
                    for edge in incident_edges.get(mode_id, ())
                ),
                "incident_mode_edge_ids_sha256": _sha256_json(list(full_edge_ids)),
                "witness_mode_edge_ids": list(
                    full_edge_ids[:MAX_ORPHAN_WITNESS_MODE_EDGE_IDS]
                ),
                "witness_ids_truncated": len(full_edge_ids)
                > MAX_ORPHAN_WITNESS_MODE_EDGE_IDS,
                "member_mode_ids_sha256": _sha256_json([mode_id]),
                "member_parent_ids_sha256": _sha256_json([parent_id]),
                "role_contract": dict(ORPHAN_ROLE_CONTRACT),
            }
            orphan_record["orphan_sha256"] = _record_hash(
                orphan_record, "orphan_sha256"
            )
            orphans.append(orphan_record)
            orphan_mode_ids.add(mode_id)
            if len(orphans) > MAX_LOCAL_PLANE_ORPHANS:
                raise ValueError("local plane orphan cap breached")
            continue
        parent_ids = tuple(sorted({str(modes[mode_id]["parent_atom_id"]) for mode_id in mode_ids}))
        fragment_id = "LPF_" + _sha256_json([LOCAL_PLANE_FRAGMENT_PROFILE, list(mode_ids)])[:16]
        normal = fit["normal"]
        centroid = fit["centroid"]
        tangent_u, tangent_v = _tangent_basis(normal)
        projections = np.asarray(
            [
                [
                    float(np.dot(_vector(modes[mode_id]["centroid"], "mode centroid", allow_zero=True) - centroid, tangent_u)),
                    float(np.dot(_vector(modes[mode_id]["centroid"], "mode centroid", allow_zero=True) - centroid, tangent_v)),
                ]
                for mode_id in mode_ids
            ],
            dtype=np.float64,
        )
        extent = (
            (float(np.min(projections[:, 0])), float(np.max(projections[:, 0]))),
            (float(np.min(projections[:, 1])), float(np.max(projections[:, 1]))),
        )
        overflow_count = sum(
            bool(parents[parent_id].get("mode_overflow", False))
            or not bool(parents[parent_id].get("transition_certificate_allowed", True))
            for parent_id in parent_ids
        )
        coverage, full_matches = _global_coverage(
            mode_ids, modes, planes, plane_fit_tolerance_mm
        )
        record = {
            "local_plane_fragment_id": fragment_id,
            "normal_projective": _round_vector(normal, 12),
            "offset": round(float(fit["offset"]), 6),
            "centroid": _round_vector(centroid, 6),
            "tangent_basis": [_round_vector(tangent_u, 12), _round_vector(tangent_v, 12)],
            "centroid_extent_uv": [
                [round(value, 6) for value in extent[0]],
                [round(value, 6) for value in extent[1]],
            ],
            "mode_count": len(mode_ids),
            "parent_count": len(parent_ids),
            "point_count_sum": sum(int(modes[mode_id]["point_count"]) for mode_id in mode_ids),
            "normalized_parent_mass_sum": sum(int(parents[parent_id]["normalized_support_mass"]) for parent_id in parent_ids),
            "same_surface_edge_count": int(same_edge_count_by_root[root]),
            "normal_diameter_deg": round(float(fit["normal_diameter_deg"]), 8),
            "max_mode_to_fragment_normal_deg": round(float(fit["max_mode_to_fragment_normal_deg"]), 8),
            "surface_fit_p90_mm": round(float(fit["surface_fit_p90_mm"]), 6),
            "mode_overflow_parent_count": int(overflow_count),
            "transition_certificate_allowed": overflow_count == 0,
            "member_mode_ids_sha256": _sha256_json(list(mode_ids)),
            "member_parent_ids_sha256": _sha256_json(list(parent_ids)),
            "witness_mode_ids": list(mode_ids[:MAX_FRAGMENT_WITNESS_MODE_IDS]),
            "witness_parent_ids": list(parent_ids[:MAX_FRAGMENT_WITNESS_PARENT_IDS]),
            "witness_ids_truncated": bool(
                len(mode_ids) > MAX_FRAGMENT_WITNESS_MODE_IDS
                or len(parent_ids) > MAX_FRAGMENT_WITNESS_PARENT_IDS
            ),
            "global_plane_coverage": coverage,
        }
        record["fragment_sha256"] = _record_hash(record, "fragment_sha256")
        fragment = _Fragment(
            record=record,
            mode_ids=mode_ids,
            parent_ids=parent_ids,
            normal=normal,
            offset=float(fit["offset"]),
            centroid=centroid,
            full_global_matches=full_matches,
        )
        fragments.append(fragment)
        for mode_id in mode_ids:
            if mode_id in fragment_of_mode:
                raise AssertionError("mode belongs to multiple local fragments")
            fragment_of_mode[mode_id] = fragment_id
    fragments.sort(key=lambda item: item.record["local_plane_fragment_id"])
    orphans.sort(key=lambda item: item["local_plane_orphan_id"])
    if len(fragments) + len(orphans) > MAX_FRAGMENT_PLUS_ORPHAN_RECORDS:
        raise ValueError("fragment-plus-orphan cap breached")
    return fragments, orphans, fragment_of_mode, orphan_mode_ids


def _plane_intersection(
    left_normal: np.ndarray,
    left_offset: float,
    right_normal: np.ndarray,
    right_offset: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    angle = projective_angle_deg(left_normal, right_normal)
    if angle < PLANE_PAIR_MIN_ANGLE_DEG - 1e-12 or angle > PLANE_PAIR_MAX_PROJECTIVE_ANGLE_DEG + 1e-12:
        raise ValueError("local plane pair angle is ineligible")
    axis = canonical_projective_normal(np.cross(left_normal, right_normal))
    matrix = np.vstack((left_normal, right_normal, axis))
    anchor = np.linalg.solve(
        matrix, np.asarray((-left_offset, -right_offset, 0.0), dtype=np.float64)
    )
    return anchor, axis, angle


def _line_distance(point: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> float:
    delta = point - anchor
    return float(np.linalg.norm(delta - float(np.dot(delta, axis)) * axis))


def _connected_components(
    mode_ids: set[str], edges: Sequence[Mapping[str, Any]]
) -> list[tuple[str, ...]]:
    adjacency = {mode_id: set() for mode_id in mode_ids}
    for edge in edges:
        left, right = (str(value) for value in edge["mode_ids"])
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    unseen = set(mode_ids)
    result = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        unseen -= component
        result.append(tuple(sorted(component)))
    return result


def _longest_consecutive_run(occupied: set[int]) -> int:
    longest = current = 0
    previous = None
    for bin_id in sorted(occupied):
        current = current + 1 if previous is not None and bin_id == previous + 1 else 1
        longest = max(longest, current)
        previous = bin_id
    return longest


def _maximum_empty_run(occupied: set[int]) -> int:
    if not occupied:
        return 0
    longest = current = 0
    for bin_id in range(min(occupied), max(occupied) + 1):
        current = 0 if bin_id in occupied else current + 1
        longest = max(longest, current)
    return longest


def _weighted_tls(
    parent_ids: Sequence[str], parents: Mapping[str, Mapping[str, Any]], fallback_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray([parents[parent_id]["centroid"] for parent_id in parent_ids], dtype=np.float64)
    weights = np.asarray(
        [int(parents[parent_id]["normalized_support_mass"]) for parent_id in parent_ids],
        dtype=np.float64,
    )
    anchor = np.average(points, axis=0, weights=weights)
    if len(points) >= 2:
        weighted = (points - anchor) * np.sqrt(weights[:, None])
        _singular, _values, basis = np.linalg.svd(weighted, full_matrices=False)
        axis = canonical_projective_normal(basis[0])
    else:
        axis = fallback_axis
    residuals = [
        (
            _line_distance(point, anchor, axis),
            parent_id,
            int(parents[parent_id]["normalized_support_mass"]),
        )
        for point, parent_id in zip(points, parent_ids)
    ]
    return anchor, axis, _weighted_quantile(residuals, 0.9)


def _global_pair_comparison(
    left: _Fragment, right: _Fragment
) -> tuple[dict[str, Any], tuple[str, str] | None]:
    candidates = []
    for left_id, left_residual in left.full_global_matches:
        for right_id, right_residual in right.full_global_matches:
            if left_id == right_id:
                continue
            ids = tuple(sorted((left_id, right_id)))
            candidates.append((left_residual + right_residual, ids))
    if not candidates:
        return {
            "state": "NO_DISTINCT_FULL_GLOBAL_PAIR",
            "global_plane_ids": [],
            "total_dimensionless_residual": None,
        }, None
    residual, ids = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "state": "FULL_DISTINCT_GLOBAL_PAIR",
        "global_plane_ids": list(ids),
        "total_dimensionless_residual": round(float(residual), 8),
    }, ids


def _line_comparison(
    *,
    local_anchor: np.ndarray,
    local_axis: np.ndarray,
    global_plane_ids: tuple[str, str] | None,
    planes_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, float | None, float | None]:
    if global_plane_ids is None:
        return "UNAVAILABLE", None, None
    left_normal, left_offset, _left_id = _canonical_plane(planes_by_id[global_plane_ids[0]])
    right_normal, right_offset, _right_id = _canonical_plane(planes_by_id[global_plane_ids[1]])
    try:
        global_anchor, global_axis, _angle = _plane_intersection(
            left_normal, left_offset, right_normal, right_offset
        )
    except (ValueError, np.linalg.LinAlgError):
        return "NON_EQUIVALENT", None, None
    axis_angle = projective_angle_deg(local_axis, global_axis)
    symmetric_offset = max(
        _line_distance(local_anchor, global_anchor, global_axis),
        _line_distance(global_anchor, local_anchor, local_axis),
    )
    state = (
        "EQUIVALENT"
        if axis_angle <= GLOBAL_LINE_AXIS_ANGLE_DEG + 1e-12
        and symmetric_offset <= GLOBAL_LINE_TRANSVERSE_OFFSET_MM + 1e-12
        else "NON_EQUIVALENT"
    )
    return state, axis_angle, symmetric_offset


def _build_component_record(
    *,
    pair_id: str,
    mode_ids: tuple[str, ...],
    left: _Fragment,
    right: _Fragment,
    anchor: np.ndarray,
    axis: np.ndarray,
    restricted_edges: Sequence[Mapping[str, Any]],
    parents: Mapping[str, Mapping[str, Any]],
    modes: Mapping[str, Mapping[str, Any]],
    fragment_of_mode: Mapping[str, str],
    bin_width: float,
    tube_tolerance: float,
    global_pair_ids: tuple[str, str] | None,
    planes_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parent_ids = tuple(sorted({str(modes[mode_id]["parent_atom_id"]) for mode_id in mode_ids}))
    projections = {
        parent_id: float(np.dot(_vector(parents[parent_id]["centroid"], "parent centroid", allow_zero=True), axis))
        for parent_id in parent_ids
    }
    lower, upper = min(projections.values()), max(projections.values())
    bin_by_parent = {
        parent_id: int(math.floor((projection - lower) / bin_width + 1e-9))
        for parent_id, projection in projections.items()
    }
    occupied_bins = set(bin_by_parent.values())
    left_id = left.record["local_plane_fragment_id"]
    right_id = right.record["local_plane_fragment_id"]
    left_bins = {
        bin_by_parent[str(modes[mode_id]["parent_atom_id"])]
        for mode_id in mode_ids
        if fragment_of_mode[mode_id] == left_id
    }
    right_bins = {
        bin_by_parent[str(modes[mode_id]["parent_atom_id"])]
        for mode_id in mode_ids
        if fragment_of_mode[mode_id] == right_id
    }
    cross_edges = [edge for edge in restricted_edges if str(edge["edge_kind"]) == "CROSS_MODE_LOCAL"]
    rung_parent_ids = {
        str(modes[str(mode_id)]["parent_atom_id"])
        for edge in cross_edges
        for mode_id in edge["mode_ids"]
    }
    rung_bins = {bin_by_parent[parent_id] for parent_id in rung_parent_ids}
    bin_neighbors: dict[int, set[int]] = defaultdict(set)
    for edge in restricted_edges:
        left_mode, right_mode = (str(value) for value in edge["mode_ids"])
        if left_mode not in mode_ids or right_mode not in mode_ids:
            continue
        left_bin = bin_by_parent[str(modes[left_mode]["parent_atom_id"])]
        right_bin = bin_by_parent[str(modes[right_mode]["parent_atom_id"])]
        if left_bin != right_bin:
            bin_neighbors[left_bin].add(right_bin)
            bin_neighbors[right_bin].add(left_bin)
    minimum_bin, maximum_bin = min(occupied_bins), max(occupied_bins)
    branch_count = sum(
        len(neighbors) > 2 and bin_id not in {minimum_bin, maximum_bin}
        for bin_id, neighbors in bin_neighbors.items()
    )
    _tls_anchor, tls_axis, line_p90 = _weighted_tls(parent_ids, parents, axis)
    tls_angle = projective_angle_deg(tls_axis, axis)
    certified_windows = 0
    for start in range(minimum_bin, maximum_bin - 1):
        required = set(range(start, start + 3))
        if (
            required <= occupied_bins
            and required <= left_bins
            and required <= right_bins
            and len(required & rung_bins) >= 2
        ):
            certified_windows += 1
    overflow_count = sum(
        bool(parents[parent_id].get("mode_overflow", False))
        or not bool(parents[parent_id].get("transition_certificate_allowed", True))
        for parent_id in parent_ids
    )
    gates = {
        "minimum_parent_count": len(parent_ids) >= MINIMUM_PARENT_COUNT,
        "minimum_axial_span": upper - lower >= MINIMUM_AXIAL_SPAN_BINS * bin_width,
        "minimum_occupied_bins": len(occupied_bins) >= MINIMUM_OCCUPIED_BINS,
        "minimum_left_side_bins": len(left_bins) >= MINIMUM_SIDE_BINS,
        "minimum_right_side_bins": len(right_bins) >= MINIMUM_SIDE_BINS,
        "minimum_transition_rung_bins": len(rung_bins) >= MINIMUM_TRANSITION_RUNG_BINS,
        "minimum_consecutive_occupied_run": _longest_consecutive_run(occupied_bins) >= MINIMUM_CONSECUTIVE_RUN,
        "maximum_empty_bin_run": _maximum_empty_run(occupied_bins) <= MAXIMUM_EMPTY_BIN_RUN,
        "no_internal_branching": branch_count <= MAXIMUM_INTERNAL_BRANCH_COUNT,
        "tls_axis": tls_angle <= MAXIMUM_TLS_AXIS_ANGLE_DEG,
        "line_residual": line_p90 <= tube_tolerance,
        "transition_certificate_complete": bool(
            overflow_count == 0
            and left.record["transition_certificate_allowed"]
            and right.record["transition_certificate_allowed"]
        ),
        "certified_three_bin_window": certified_windows > 0,
    }
    first_rejection = next((code for gate, code in REJECTION_ORDER if not gates[gate]), None)
    gate_depth = next((index for index, (gate, _code) in enumerate(REJECTION_ORDER) if not gates[gate]), len(REJECTION_ORDER))
    local_endpoints = (anchor + lower * axis, anchor + upper * axis)
    line_state, global_angle, global_offset = _line_comparison(
        local_anchor=anchor,
        local_axis=axis,
        global_plane_ids=global_pair_ids,
        planes_by_id=planes_by_id,
    )
    component_id = "LTC_" + _sha256_json([pair_id, list(parent_ids), list(mode_ids)])[:16]
    record = {
        "local_transition_component_id": component_id,
        "support_parent_count": len(parent_ids),
        "support_mode_count": len(mode_ids),
        "normalized_parent_mass": sum(int(parents[parent_id]["normalized_support_mass"]) for parent_id in parent_ids),
        "support_parent_ids_sha256": _sha256_json(list(parent_ids)),
        "support_mode_ids_sha256": _sha256_json(list(mode_ids)),
        "witness_parent_ids": list(parent_ids[:MAX_COMPONENT_WITNESS_PARENT_IDS]),
        "witness_mode_ids": list(mode_ids[:MAX_COMPONENT_WITNESS_MODE_IDS]),
        "witness_ids_truncated": bool(
            len(parent_ids) > MAX_COMPONENT_WITNESS_PARENT_IDS
            or len(mode_ids) > MAX_COMPONENT_WITNESS_MODE_IDS
        ),
        "intersection_span_endpoints": [_round_vector(endpoint, 6) for endpoint in local_endpoints],
        "occupied_bin_count": len(occupied_bins),
        "left_side_bin_count": len(left_bins),
        "right_side_bin_count": len(right_bins),
        "transition_rung_bin_count": len(rung_bins),
        "longest_consecutive_occupied_run": _longest_consecutive_run(occupied_bins),
        "maximum_empty_bin_run": _maximum_empty_run(occupied_bins),
        "internal_branch_count": int(branch_count),
        "tls_axis_angle_deg": round(float(tls_angle), 8),
        "line_p90_mm": round(float(line_p90), 6),
        "certified_three_bin_window_count": certified_windows,
        "mode_overflow_parent_count": int(overflow_count),
        "geometry_gate_depth": gate_depth,
        "geometry_gates": gates,
        "first_rejection_code": first_rejection,
        "geometry_certificate_valid": all(gates.values()),
        "global_pair_line_state": line_state,
        "global_pair_axis_angle_deg": None if global_angle is None else round(float(global_angle), 8),
        "global_pair_transverse_offset_mm": None if global_offset is None else round(float(global_offset), 6),
    }
    record["component_sha256"] = _record_hash(record, "component_sha256")
    return record


def engineering_profile_descriptor_v2() -> dict[str, Any]:
    return {
        "profile": LOCAL_PLANE_FRAGMENT_PROFILE,
        "schema_version": PROBE_SCHEMA_VERSION,
        "formulas": {
            "fragment_normal": "point_count_weighted_projective_moment",
            "fragment_offset": "point_count_weighted_lower_median",
            "fragment_fit": "weighted_p90(abs(n_dot_c_plus_d)+mode_surface_fit_p90_mm)",
            "fragment_partition": "ordered_same_surface_union_find_complete_diameter_fit",
            "pair_seed": "cross_mode_local_edges_only",
            "component_support": "restricted_existing_edges_only",
            "physical_support": "unique_parent",
            "global_membership": "angle_over_15_plus_distance_over_plane_fit_tolerance",
        },
        "final_component_classification": {
            "rule_id": ORPHAN_RULE_ID,
            "fit_source": "unrounded_binary64_fit_fragment_result",
            "comparison": "STRICT_GT_NO_EPSILON",
            "equality_result": "VALID_LOCAL_PLANE_FRAGMENT",
            "singleton_over_tolerance_result": ORPHAN_KIND,
            "multimode_over_tolerance_result": "FAIL_CLOSED",
            "nonfinite_result": "FAIL_CLOSED",
        },
        "orphan_role_contract": dict(ORPHAN_ROLE_CONTRACT),
        "mode_partition_invariant": "EXACT_DISJOINT_UNION_OF_VALID_FRAGMENT_MODES_AND_ORPHAN_MODES",
        "input_projection_profile": INPUT_PROJECTION_PROFILE,
        "exact_schema_field_sets": {
            "top_level": sorted(TOP_LEVEL_FIELDS),
            "fragment": sorted(FRAGMENT_RECORD_FIELDS),
            "global_plane_coverage": sorted(GLOBAL_PLANE_COVERAGE_FIELDS),
            "orphan": sorted(ORPHAN_RECORD_FIELDS),
            "orphan_role_contract": sorted(ORPHAN_ROLE_CONTRACT),
            "pair": sorted(PAIR_RECORD_FIELDS),
            "global_pair_comparison": sorted(GLOBAL_PAIR_COMPARISON_FIELDS),
            "component": sorted(COMPONENT_RECORD_FIELDS),
            "input_contract_attestation": sorted(INPUT_CONTRACT_ATTESTATION_FIELDS),
        },
        "canonical_ids": {
            "fragment": "LPF_sha256([profile,sorted_full_member_mode_ids])_16",
            "orphan": "LPO_sha256([profile,orphan_rule_id,mode_atom_id,parent_atom_id])_16",
            "pair": "LPP_sha256([sorted_fragment_ids])_16",
            "component": "LTC_sha256([pair_id,sorted_parent_ids,sorted_mode_ids])_16",
        },
        "hard_caps": dict(HARD_CAPS),
        "first_rejection_order": [list(item) for item in REJECTION_ORDER],
        "rounding": {
            "normal_axis_tangent": 12,
            "centroid_offset_endpoint_extent": 6,
            "dimensionless_residual": 8,
        },
    }


def engineering_profile_sha256(
    thresholds: Mapping[str, Any],
    *,
    descriptor: Mapping[str, Any] | None = None,
) -> str:
    return _sha256_json(
        {
            "descriptor": dict(descriptor or engineering_profile_descriptor_v2()),
            "case_thresholds": dict(thresholds),
        }
    )


def build_local_plane_fragment_probe(
    candidate_pool: Mapping[str, Any],
    *,
    scan_sha256: str,
    candidate_pool_sha256: str,
    a0_prediction_sha256: str,
) -> dict[str, Any]:
    """Build one canonical LPF-M1 inference-safe payload from frozen tables."""

    scan_hash = _validate_sha256("scan_sha256", scan_sha256)
    pool_hash = _validate_sha256("candidate_pool_sha256", candidate_pool_sha256)
    prediction_hash = _validate_sha256("a0_prediction_sha256", a0_prediction_sha256)
    validated = _validate_inputs(candidate_pool)
    parents = validated["parent_by_id"]
    modes = validated["mode_by_id"]
    edges = validated["edges"]
    planes = validated["planes"]
    spacing = float(validated["spacing"])
    plane_fit_tolerance = min(3.0, max(1.25, 1.5 * spacing))
    line_tube_tolerance = min(3.0, max(1.5, 1.25 * spacing))
    bin_width = max(3.0, 2.0 * spacing)
    thresholds = {
        "plane_normal_diameter_deg": PLANE_NORMAL_DIAMETER_DEG,
        "plane_fit_tolerance_mm": plane_fit_tolerance,
        "plane_pair_min_angle_deg": PLANE_PAIR_MIN_ANGLE_DEG,
        "plane_pair_max_projective_angle_deg": PLANE_PAIR_MAX_PROJECTIVE_ANGLE_DEG,
        "line_tube_tolerance_mm": line_tube_tolerance,
        "axial_bin_width_mm": bin_width,
        "minimum_parent_count": MINIMUM_PARENT_COUNT,
        "minimum_axial_span_bins": MINIMUM_AXIAL_SPAN_BINS,
        "minimum_occupied_bins": MINIMUM_OCCUPIED_BINS,
        "minimum_side_bins": MINIMUM_SIDE_BINS,
        "minimum_transition_rung_bins": MINIMUM_TRANSITION_RUNG_BINS,
        "minimum_consecutive_run": MINIMUM_CONSECUTIVE_RUN,
        "maximum_empty_bin_run": MAXIMUM_EMPTY_BIN_RUN,
        "maximum_internal_branch_count": MAXIMUM_INTERNAL_BRANCH_COUNT,
        "maximum_tls_axis_angle_deg": MAXIMUM_TLS_AXIS_ANGLE_DEG,
        "global_line_axis_angle_deg": GLOBAL_LINE_AXIS_ANGLE_DEG,
        "global_line_transverse_offset_mm": GLOBAL_LINE_TRANSVERSE_OFFSET_MM,
    }
    counters: Counter[str] = Counter(
        {
            "input_parent_count": len(parents),
            "input_mode_count": len(modes),
            "input_same_surface_edge_count": sum(str(edge["edge_kind"]) == "SAME_SURFACE" for edge in edges),
            "input_cross_mode_edge_count": sum(str(edge["edge_kind"]) == "CROSS_MODE_LOCAL" for edge in edges),
            "input_global_plane_count": len(planes),
            "inherited_working_point_count": int(validated["working_count"]),
        }
    )
    fragments, orphans, fragment_of_mode, orphan_mode_ids = _build_fragments(
        parents=parents,
        modes=modes,
        edges=edges,
        planes=planes,
        plane_fit_tolerance_mm=plane_fit_tolerance,
        counters=counters,
    )
    counters["local_plane_fragment_count"] = len(fragments)
    counters["singleton_fragment_count"] = sum(fragment.record["mode_count"] == 1 for fragment in fragments)
    counters["valid_singleton_fragment_count"] = counters["singleton_fragment_count"]
    counters["overflow_ineligible_fragment_count"] = sum(
        int(fragment.record["mode_overflow_parent_count"]) > 0 for fragment in fragments
    )
    counters["valid_fragment_mode_count"] = sum(
        int(fragment.record["mode_count"]) for fragment in fragments
    )
    counters["orphan_mode_count"] = len(orphan_mode_ids)
    counters["local_plane_orphan_count"] = len(orphans)
    counters["high_fit_singleton_orphan_count"] = len(orphans)
    counters["orphan_incident_same_surface_edge_count"] = sum(
        int(orphan["incident_same_surface_edge_count"]) for orphan in orphans
    )
    counters["orphan_incident_cross_mode_edge_count"] = sum(
        int(orphan["incident_cross_mode_edge_count"]) for orphan in orphans
    )
    counters["mode_partition_record_count"] = len(fragments) + len(orphans)
    counters["fragment_no_global_match_count"] = sum(
        fragment.record["global_plane_coverage"]["coverage_state"] == "NONE" for fragment in fragments
    )
    counters["fragment_partial_global_match_count"] = sum(
        fragment.record["global_plane_coverage"]["coverage_state"] == "PARTIAL" for fragment in fragments
    )
    counters["fragment_full_global_match_count"] = sum(
        fragment.record["global_plane_coverage"]["coverage_state"] == "FULL" for fragment in fragments
    )
    fragment_by_id = {fragment.record["local_plane_fragment_id"]: fragment for fragment in fragments}
    cross_edges = sorted(
        (edge for edge in edges if str(edge["edge_kind"]) == "CROSS_MODE_LOCAL"),
        key=lambda item: str(item["mode_edge_id"]),
    )
    counters["cross_mode_seed_edge_count"] = len(cross_edges)
    pair_to_seed_edges: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in cross_edges:
        left_mode, right_mode = (str(value) for value in edge["mode_ids"])
        if left_mode not in fragment_of_mode or right_mode not in fragment_of_mode:
            counters["excluded_cross_mode_seed_edge_count_due_to_orphan"] += 1
            continue
        left_fragment = fragment_of_mode[left_mode]
        right_fragment = fragment_of_mode[right_mode]
        if left_fragment == right_fragment:
            continue
        pair_to_seed_edges[tuple(sorted((left_fragment, right_fragment)))].append(edge)
    if len(pair_to_seed_edges) > MAX_LOCAL_PLANE_PAIRS:
        raise ValueError("local plane pair cap breached")
    counters["local_plane_pair_count"] = len(pair_to_seed_edges)

    planes_by_id = {_canonical_plane(plane)[2]: plane for plane in planes}
    pair_records = []
    total_components = 0
    first_rejections: Counter[str] = Counter()
    funnel_names = [gate for gate, _code in REJECTION_ORDER]
    for pair_ids in sorted(pair_to_seed_edges):
        left, right = (fragment_by_id[fragment_id] for fragment_id in pair_ids)
        pair_id = "LPP_" + _sha256_json([list(pair_ids)])[:16]
        pair_rejections: Counter[str] = Counter()
        global_comparison, global_pair_ids = _global_pair_comparison(left, right)
        component_records = []
        angle = projective_angle_deg(left.normal, right.normal)
        transverse_parent_ids: set[str] = set()
        left_modes: set[str] = set()
        right_modes: set[str] = set()
        retained_cross_edges: list[Mapping[str, Any]] = []
        anchor = np.zeros(3, dtype=np.float64)
        axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        try:
            anchor, axis, angle = _plane_intersection(
                left.normal, left.offset, right.normal, right.offset
            )
        except (ValueError, np.linalg.LinAlgError):
            pair_rejections["PLANE_PAIR_ANGLE_INELIGIBLE"] += 1
        else:
            counters["angle_eligible_local_pair_count"] += 1
            transverse_parent_ids = {
                parent_id
                for parent_id, parent in parents.items()
                if _line_distance(
                    _vector(parent["centroid"], "parent centroid", allow_zero=True), anchor, axis
                )
                <= line_tube_tolerance + 1e-12
            }
            left_modes = {
                mode_id for mode_id in left.mode_ids
                if str(modes[mode_id]["parent_atom_id"]) in transverse_parent_ids
            }
            right_modes = {
                mode_id for mode_id in right.mode_ids
                if str(modes[mode_id]["parent_atom_id"]) in transverse_parent_ids
            }
            if not transverse_parent_ids:
                pair_rejections["NO_TRANSVERSE_PARENT"] += 1
                counters["pair_no_transverse_parent_count"] += 1
            if not left_modes:
                pair_rejections["MISSING_LEFT_MODE"] += 1
                counters["pair_missing_left_mode_count"] += 1
            if not right_modes:
                pair_rejections["MISSING_RIGHT_MODE"] += 1
                counters["pair_missing_right_mode_count"] += 1
            supported_modes = left_modes | right_modes
            restricted_edges = []
            for edge in edges:
                edge_modes = tuple(str(value) for value in edge["mode_ids"])
                if not set(edge_modes) <= supported_modes:
                    continue
                edge_kind = str(edge["edge_kind"])
                if edge_kind == "SAME_SURFACE":
                    if fragment_of_mode[edge_modes[0]] != fragment_of_mode[edge_modes[1]]:
                        continue
                    if fragment_of_mode[edge_modes[0]] not in pair_ids:
                        continue
                elif edge_kind == "CROSS_MODE_LOCAL":
                    if {fragment_of_mode[edge_modes[0]], fragment_of_mode[edge_modes[1]]} != set(pair_ids):
                        continue
                    retained_cross_edges.append(edge)
                else:
                    continue
                restricted_edges.append(edge)
            if not retained_cross_edges:
                pair_rejections["NO_CROSS_MODE_RUNG"] += 1
                counters["pair_no_cross_mode_rung_count"] += 1
            components = _connected_components(supported_modes, restricted_edges)
            if len(components) > MAX_COMPONENTS_PER_LOCAL_PAIR:
                raise ValueError("components per local pair cap breached")
            total_components += len(components)
            if total_components > MAX_LOCAL_TRANSITION_COMPONENTS:
                raise ValueError("local transition component cap breached")
            for component_mode_ids in components:
                component_edges = [
                    edge for edge in restricted_edges
                    if set(str(value) for value in edge["mode_ids"]) <= set(component_mode_ids)
                ]
                record = _build_component_record(
                    pair_id=pair_id,
                    mode_ids=component_mode_ids,
                    left=left,
                    right=right,
                    anchor=anchor,
                    axis=axis,
                    restricted_edges=component_edges,
                    parents=parents,
                    modes=modes,
                    fragment_of_mode=fragment_of_mode,
                    bin_width=bin_width,
                    tube_tolerance=line_tube_tolerance,
                    global_pair_ids=global_pair_ids,
                    planes_by_id=planes_by_id,
                )
                component_records.append(record)
                counters["local_transition_component_count"] += 1
                if record["geometry_certificate_valid"]:
                    counters["geometry_certificate_count"] += 1
                    if global_pair_ids is None:
                        counters["certificate_no_distinct_full_global_pair_count"] += 1
                    elif record["global_pair_line_state"] == "EQUIVALENT":
                        counters["certificate_global_pair_line_equivalent_count"] += 1
                    else:
                        counters["certificate_global_pair_line_non_equivalent_count"] += 1
                if record["first_rejection_code"] is not None:
                    first_rejections[str(record["first_rejection_code"])] += 1
                for index, gate_name in enumerate(funnel_names):
                    if all(record["geometry_gates"][name] for name in funnel_names[: index + 1]):
                        counters[f"components_after_{gate_name}"] += 1
        pair_record = {
            "local_plane_pair_id": pair_id,
            "fragment_ids": list(pair_ids),
            "plane_angle_deg": round(float(angle), 8),
            "intersection_line_hash": _sha256_json(
                [_round_vector(anchor, 6), _round_vector(axis, 12)]
            ),
            "intersection_anchor": _round_vector(anchor, 6),
            "intersection_axis": _round_vector(axis, 12),
            "transverse_parent_count": len(transverse_parent_ids),
            "left_mode_count": len(left_modes),
            "right_mode_count": len(right_modes),
            "cross_mode_rung_count": len(retained_cross_edges),
            "component_count": len(component_records),
            "global_pair_comparison": global_comparison,
            "pair_rejection_counts": dict(sorted(pair_rejections.items())),
            "components": sorted(
                component_records, key=lambda item: item["local_transition_component_id"]
            ),
        }
        pair_record["pair_sha256"] = _record_hash(pair_record, "pair_sha256")
        pair_records.append(pair_record)
    pair_records.sort(key=lambda item: item["local_plane_pair_id"])
    counters["first_rejection_counts"] = dict(sorted(first_rejections.items()))
    required_counter_names = (
        "plane_union_attempt_count", "plane_union_accept_count",
        "plane_union_reject_normal_diameter_count", "plane_union_reject_fit_count",
        "angle_eligible_local_pair_count", "local_transition_component_count",
        "geometry_certificate_count", "pair_no_transverse_parent_count",
        "pair_missing_left_mode_count", "pair_missing_right_mode_count",
        "pair_no_cross_mode_rung_count", "certificate_no_distinct_full_global_pair_count",
        "certificate_global_pair_line_non_equivalent_count",
        "certificate_global_pair_line_equivalent_count",
        "excluded_cross_mode_seed_edge_count_due_to_orphan",
    )
    for name in required_counter_names:
        counters.setdefault(name, 0)
    for gate_name in funnel_names:
        counters.setdefault(f"components_after_{gate_name}", 0)
    fragment_records = [fragment.record for fragment in fragments]
    global_plane_hash = _table_sha256(
        sorted(planes, key=lambda plane: _canonical_plane(plane)[2])
    )
    input_attestation = _input_contract_attestation(candidate_pool, validated)
    engineering_hash = engineering_profile_sha256(thresholds)
    output_table_hashes = {
        "local_plane_fragment_table_sha256": _table_sha256(
            fragment_records, "local_plane_fragment_id"
        ),
        "local_plane_orphan_table_sha256": _table_sha256(
            orphans, "local_plane_orphan_id"
        ),
        "local_plane_pair_records_sha256": _table_sha256(
            pair_records, "local_plane_pair_id"
        ),
    }
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "kind": PROBE_KIND,
        "profile": LOCAL_PLANE_FRAGMENT_PROFILE,
        "research_only": True,
        "inference_safe": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "scan_sha256": scan_hash,
        "candidate_pool_sha256": pool_hash,
        "a0_prediction_sha256": prediction_hash,
        "input_table_hashes": {
            **validated["input_hashes"],
            "global_plane_table_sha256": global_plane_hash,
        },
        "output_table_hashes": output_table_hashes,
        "engineering_profile_sha256": engineering_hash,
        "inference_semantic_sha256": _sha256_json(
            {
                "profile": LOCAL_PLANE_FRAGMENT_PROFILE,
                "engineering_profile_sha256": engineering_hash,
                "projected_input_sha256": input_attestation["projected_input_sha256"],
                "output_table_hashes": output_table_hashes,
            }
        ),
        "input_contract_attestation": input_attestation,
        "thresholds": thresholds,
        "hard_caps": dict(HARD_CAPS),
        "case_counters": dict(sorted(counters.items())),
        "local_plane_fragment_table": fragment_records,
        "local_plane_orphan_table": orphans,
        "local_plane_pair_records": pair_records,
    }
    validate_local_plane_fragment_payload(payload, projected_input=candidate_pool)
    return payload


def validate_local_plane_fragment_payload(
    payload: Mapping[str, Any],
    *,
    projected_input: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on the exact V2 schemas, hashes, partition, and semantics."""

    reject_forbidden_fields(payload)
    if set(payload) != TOP_LEVEL_FIELDS:
        raise ValueError(
            f"LPF payload top-level schema mismatch: {sorted(set(payload) ^ TOP_LEVEL_FIELDS)}"
        )
    if (
        int(payload["schema_version"]) != PROBE_SCHEMA_VERSION
        or payload["kind"] != PROBE_KIND
        or payload["profile"] != LOCAL_PLANE_FRAGMENT_PROFILE
    ):
        raise ValueError("LPF V2 payload identity is invalid")
    expected_flags = {
        "research_only": True,
        "inference_safe": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
    }
    if any(payload[key] is not value for key, value in expected_flags.items()):
        raise ValueError("LPF inference-safety flags are invalid")
    for field in (
        "scan_sha256", "candidate_pool_sha256", "a0_prediction_sha256",
        "engineering_profile_sha256", "inference_semantic_sha256",
    ):
        _validate_sha256(field, payload[field])
    for table_group in ("input_table_hashes", "output_table_hashes"):
        if not isinstance(payload[table_group], Mapping):
            raise ValueError("LPF table hashes must be mappings")
        for key, value in payload[table_group].items():
            _validate_sha256(str(key), value)
    if dict(payload["hard_caps"]) != HARD_CAPS:
        raise ValueError("LPF hard-cap schema differs from frozen profile")
    thresholds = dict(payload["thresholds"])
    expected_engineering_hash = engineering_profile_sha256(thresholds)
    if payload["engineering_profile_sha256"] != expected_engineering_hash:
        raise ValueError("LPF engineering profile hash does not recompute")
    attestation = dict(payload["input_contract_attestation"])
    if set(attestation) != INPUT_CONTRACT_ATTESTATION_FIELDS:
        raise ValueError("LPF input contract attestation schema mismatch")
    if attestation.get("projection_profile") != INPUT_PROJECTION_PROFILE:
        raise ValueError("LPF input projection profile is invalid")
    if attestation.get("support_representation_profile") != "TRANSITION_ATOM_V2":
        raise ValueError("LPF attested support representation is invalid")
    if attestation.get("source_point_serialization") is not False:
        raise ValueError("LPF attestation permits source-point serialization")
    if dict(attestation.get("transition_atom_caps") or {}) != TRANSITION_ATOM_CAPS:
        raise ValueError("LPF attested inherited caps are invalid")
    if attestation.get("input_contract_attestation_sha256") != _record_hash(
        attestation, "input_contract_attestation_sha256"
    ):
        raise ValueError("LPF input contract attestation hash mismatch")
    if attestation.get("all_inherited_caps_satisfied") is not True:
        raise ValueError("LPF inherited input caps are not satisfied")
    for value in dict(attestation.get("table_hashes") or {}).values():
        _validate_sha256("attested table hash", value)
    if dict(payload["input_table_hashes"]) != dict(attestation["table_hashes"]):
        raise ValueError("LPF top-level and attested input hashes differ")
    if projected_input is not None:
        validated_input = _validate_inputs(projected_input)
        recomputed_attestation = _input_contract_attestation(
            projected_input, validated_input
        )
        if attestation != recomputed_attestation:
            raise ValueError("LPF input contract attestation does not recompute")
    fragments = [dict(item) for item in payload["local_plane_fragment_table"]]
    orphans = [dict(item) for item in payload["local_plane_orphan_table"]]
    pairs = [dict(item) for item in payload["local_plane_pair_records"]]
    if (
        len(fragments) > MAX_LOCAL_PLANE_FRAGMENTS
        or len(orphans) > MAX_LOCAL_PLANE_ORPHANS
        or len(fragments) + len(orphans) > MAX_FRAGMENT_PLUS_ORPHAN_RECORDS
        or len(pairs) > MAX_LOCAL_PLANE_PAIRS
    ):
        raise ValueError("LPF output structural cap breached")
    fragment_ids = [str(item.get("local_plane_fragment_id") or "") for item in fragments]
    if not all(fragment_ids) or len(set(fragment_ids)) != len(fragment_ids):
        raise ValueError("LPF fragment IDs must be unique and nonempty")
    if fragment_ids != sorted(fragment_ids):
        raise ValueError("LPF fragment table is not canonically ordered")
    component_ids = set()
    total_components = 0
    for fragment in fragments:
        if set(fragment) != FRAGMENT_RECORD_FIELDS:
            raise ValueError("LPF fragment nested schema mismatch")
        if fragment.get("fragment_sha256") != _record_hash(fragment, "fragment_sha256"):
            raise ValueError("LPF fragment hash mismatch")
        if int(fragment.get("mode_count", -1)) < 1 or int(fragment.get("parent_count", -1)) < 1:
            raise ValueError("LPF fragment counts must be positive")
        if len(fragment.get("witness_mode_ids") or ()) > MAX_FRAGMENT_WITNESS_MODE_IDS:
            raise ValueError("LPF fragment mode witness cap breached")
        if len(fragment.get("witness_parent_ids") or ()) > MAX_FRAGMENT_WITNESS_PARENT_IDS:
            raise ValueError("LPF fragment parent witness cap breached")
        truncated_expected = (
            int(fragment["mode_count"]) > len(fragment.get("witness_mode_ids") or ())
            or int(fragment["parent_count"]) > len(fragment.get("witness_parent_ids") or ())
        )
        if bool(fragment.get("witness_ids_truncated")) != truncated_expected:
            raise ValueError("LPF fragment witness truncation flag is inconsistent")
        coverage = dict(fragment.get("global_plane_coverage") or {})
        if set(coverage) != GLOBAL_PLANE_COVERAGE_FIELDS:
            raise ValueError("LPF global-plane coverage nested schema mismatch")
        if coverage.get("coverage_state") not in {"NONE", "PARTIAL", "FULL"}:
            raise ValueError("LPF global plane coverage enum is invalid")
        for key in ("full_coverage_plane_ids", "partial_coverage_plane_ids"):
            if len(coverage.get(key) or ()) > MAX_GLOBAL_PLANE_MATCH_IDS_PER_FRAGMENT:
                raise ValueError("LPF global plane witness cap breached")
        if float(fragment["surface_fit_p90_mm"]) > float(
            thresholds["plane_fit_tolerance_mm"]
        ):
            raise ValueError("LPF valid fragment exceeds exact fit tolerance")
    orphan_ids = []
    orphan_mode_ids = set()
    for orphan in orphans:
        if set(orphan) != ORPHAN_RECORD_FIELDS:
            raise ValueError("LPF orphan nested schema mismatch")
        orphan_id = str(orphan.get("local_plane_orphan_id") or "")
        mode_id = str(orphan.get("mode_atom_id") or "")
        if not orphan_id or not mode_id or mode_id in orphan_mode_ids:
            raise ValueError("LPF orphan IDs/mode partition are invalid")
        orphan_ids.append(orphan_id)
        orphan_mode_ids.add(mode_id)
        if orphan.get("orphan_kind") != ORPHAN_KIND or orphan.get("orphan_rule_id") != ORPHAN_RULE_ID:
            raise ValueError("LPF orphan identity is invalid")
        if set(dict(orphan.get("role_contract") or {})) != set(ORPHAN_ROLE_CONTRACT):
            raise ValueError("LPF orphan role nested schema mismatch")
        if dict(orphan["role_contract"]) != ORPHAN_ROLE_CONTRACT:
            raise ValueError("LPF orphan role contract permits participation")
        if float(orphan["computed_singleton_surface_fit_p90_mm"]) <= float(
            orphan["plane_fit_tolerance_mm"]
        ):
            raise ValueError("LPF orphan is not strictly above tolerance")
        if len(orphan.get("witness_mode_edge_ids") or ()) > MAX_ORPHAN_WITNESS_MODE_EDGE_IDS:
            raise ValueError("LPF orphan witness edge cap breached")
        if bool(orphan.get("witness_ids_truncated")) != (
            int(orphan["incident_same_surface_edge_count"])
            + int(orphan["incident_cross_mode_edge_count"])
            > MAX_ORPHAN_WITNESS_MODE_EDGE_IDS
        ):
            raise ValueError("LPF orphan witness truncation flag is inconsistent")
        if orphan.get("orphan_sha256") != _record_hash(orphan, "orphan_sha256"):
            raise ValueError("LPF orphan record hash mismatch")
    if orphan_ids != sorted(orphan_ids) or len(orphan_ids) != len(set(orphan_ids)):
        raise ValueError("LPF orphan table is not uniquely/canonically ordered")
    pair_ids = [str(item.get("local_plane_pair_id") or "") for item in pairs]
    if not all(pair_ids) or len(set(pair_ids)) != len(pair_ids) or pair_ids != sorted(pair_ids):
        raise ValueError("LPF pair IDs must be unique, nonempty, and sorted")
    for pair in pairs:
        if set(pair) != PAIR_RECORD_FIELDS:
            raise ValueError("LPF pair nested schema mismatch")
        fragment_pair = tuple(str(value) for value in pair.get("fragment_ids") or ())
        if len(fragment_pair) != 2 or fragment_pair[0] == fragment_pair[1] or not set(fragment_pair) <= set(fragment_ids):
            raise ValueError("LPF pair fragment references are invalid")
        if pair.get("pair_sha256") != _record_hash(pair, "pair_sha256"):
            raise ValueError("LPF pair hash mismatch")
        comparison = dict(pair.get("global_pair_comparison") or {})
        if set(comparison) != GLOBAL_PAIR_COMPARISON_FIELDS:
            raise ValueError("LPF global-pair comparison nested schema mismatch")
        if comparison.get("state") not in {
            "NO_DISTINCT_FULL_GLOBAL_PAIR", "FULL_DISTINCT_GLOBAL_PAIR"
        }:
            raise ValueError("LPF global pair comparison enum is invalid")
        components = list(pair.get("components") or ())
        if len(components) > MAX_COMPONENTS_PER_LOCAL_PAIR:
            raise ValueError("LPF components-per-pair cap breached")
        if int(pair.get("component_count", -1)) != len(components):
            raise ValueError("LPF pair component count mismatch")
        total_components += len(components)
        for component in components:
            if set(component) != COMPONENT_RECORD_FIELDS:
                raise ValueError("LPF component nested schema mismatch")
            component_id = str(component.get("local_transition_component_id") or "")
            if not component_id or component_id in component_ids:
                raise ValueError("LPF component IDs must be globally unique")
            component_ids.add(component_id)
            if component.get("component_sha256") != _record_hash(component, "component_sha256"):
                raise ValueError("LPF component hash mismatch")
            if len(component.get("witness_mode_ids") or ()) > MAX_COMPONENT_WITNESS_MODE_IDS:
                raise ValueError("LPF component mode witness cap breached")
            if len(component.get("witness_parent_ids") or ()) > MAX_COMPONENT_WITNESS_PARENT_IDS:
                raise ValueError("LPF component parent witness cap breached")
            truncated_expected = (
                int(component.get("support_mode_count", 0)) > len(component.get("witness_mode_ids") or ())
                or int(component.get("support_parent_count", 0)) > len(component.get("witness_parent_ids") or ())
            )
            if bool(component.get("witness_ids_truncated")) != truncated_expected:
                raise ValueError("LPF component witness truncation flag is inconsistent")
            gates = dict(component.get("geometry_gates") or {})
            if set(gates) != {gate for gate, _code in REJECTION_ORDER}:
                raise ValueError("LPF geometry gate schema/order is invalid")
            first = next((code for gate, code in REJECTION_ORDER if not gates[gate]), None)
            if component.get("first_rejection_code") != first:
                raise ValueError("LPF first rejection is inconsistent")
            if bool(component.get("geometry_certificate_valid")) != all(gates.values()):
                raise ValueError("LPF certificate validity is inconsistent")
            expected_depth = next(
                (
                    index
                    for index, (gate_name, _code) in enumerate(REJECTION_ORDER)
                    if gates[gate_name] is False
                ),
                len(REJECTION_ORDER),
            )
            if int(component.get("geometry_gate_depth", -1)) != expected_depth:
                raise ValueError("LPF geometry gate depth is inconsistent")
            if orphan_mode_ids & set(component.get("witness_mode_ids") or ()):
                raise ValueError("LPF component references an orphan mode")
            if component.get("global_pair_line_state") not in {
                "UNAVAILABLE", "EQUIVALENT", "NON_EQUIVALENT"
            }:
                raise ValueError("LPF global line state enum is invalid")
    if total_components > MAX_LOCAL_TRANSITION_COMPONENTS:
        raise ValueError("LPF component cap breached")
    expected_output_hashes = {
        "local_plane_fragment_table_sha256": _table_sha256(
        fragments, "local_plane_fragment_id"
        ),
        "local_plane_orphan_table_sha256": _table_sha256(
            orphans, "local_plane_orphan_id"
        ),
        "local_plane_pair_records_sha256": _table_sha256(pairs, "local_plane_pair_id"),
    }
    if dict(payload["output_table_hashes"]) != expected_output_hashes:
        raise ValueError("LPF output table hash mismatch")
    expected_semantic_hash = _sha256_json(
        {
            "profile": LOCAL_PLANE_FRAGMENT_PROFILE,
            "engineering_profile_sha256": expected_engineering_hash,
            "projected_input_sha256": attestation["projected_input_sha256"],
            "output_table_hashes": expected_output_hashes,
        }
    )
    if payload["inference_semantic_sha256"] != expected_semantic_hash:
        raise ValueError("LPF inference semantic hash does not recompute")
    counters = dict(payload["case_counters"])
    if int(counters.get("orphan_mode_count", -1)) != len(orphans):
        raise ValueError("LPF orphan mode counter mismatch")
    if int(counters.get("local_plane_orphan_count", -1)) != len(orphans):
        raise ValueError("LPF orphan table counter mismatch")
    if int(counters.get("high_fit_singleton_orphan_count", -1)) != len(orphans):
        raise ValueError("LPF high-fit singleton counter mismatch")
    if int(counters.get("valid_fragment_mode_count", -1)) + len(orphans) != int(
        counters.get("input_mode_count", -2)
    ):
        raise ValueError("LPF mode role partition count is incomplete")
    if int(counters.get("mode_partition_record_count", -1)) != len(fragments) + len(orphans):
        raise ValueError("LPF mode partition record counter mismatch")
    if projected_input is not None:
        recompute_counters: Counter[str] = Counter()
        expected_fragments, expected_orphans, _fragment_map, expected_orphan_modes = _build_fragments(
            parents=validated_input["parent_by_id"],
            modes=validated_input["mode_by_id"],
            edges=validated_input["edges"],
            planes=validated_input["planes"],
            plane_fit_tolerance_mm=float(thresholds["plane_fit_tolerance_mm"]),
            counters=recompute_counters,
        )
        if [item.record for item in expected_fragments] != fragments or expected_orphans != orphans:
            raise ValueError("LPF exact mode partition does not recompute")
        if expected_orphan_modes != orphan_mode_ids:
            raise ValueError("LPF orphan mode set does not recompute")
    canonical_json_bytes(payload)


def validate_lpf_memory_profile_row(row: Mapping[str, Any]) -> None:
    """Validate the flat isolated-process resource row used by LPF batching."""

    required = {
        "wall_seconds",
        "peak_rss_bytes",
        "input_json_bytes",
        "serialized_output_bytes",
        "case_counters",
        "replay_probe_sha256s",
        "replay_count",
    }
    if set(row) != required:
        raise ValueError(f"LPF memory row schema mismatch: {sorted(set(row) ^ required)}")
    wall_seconds = _finite(row["wall_seconds"], "memory wall_seconds")
    if wall_seconds < 0.0:
        raise ValueError("memory wall_seconds must be nonnegative")
    peak = _nonnegative_int(row["peak_rss_bytes"], "memory peak_rss_bytes")
    if peak >= PEAK_RSS_LIMIT_BYTES:
        raise ValueError("LPF process peak RSS reaches or exceeds 16 GiB")
    _nonnegative_int(row["input_json_bytes"], "memory input_json_bytes")
    _nonnegative_int(row["serialized_output_bytes"], "memory serialized_output_bytes")
    counters = dict(row["case_counters"])
    for field in (
        "local_plane_fragment_count",
        "local_plane_pair_count",
        "local_transition_component_count",
        "geometry_certificate_count",
    ):
        _nonnegative_int(counters.get(field), f"memory case_counters.{field}")
    replay_count = _nonnegative_int(row["replay_count"], "memory replay_count")
    replay_hashes = list(row["replay_probe_sha256s"])
    if replay_count != len(replay_hashes):
        raise ValueError("memory replay_count does not match replay hashes")
    for digest in replay_hashes:
        _validate_sha256("memory replay hash", digest)
    if replay_hashes and len(set(replay_hashes)) != 1:
        raise ValueError("LPF replay outputs are not byte-identical")


def verify_orphan_deletion_counterfactual(
    projected_input: Mapping[str, Any],
    *,
    scan_sha256: str,
    candidate_pool_sha256: str,
    a0_prediction_sha256: str,
) -> dict[str, Any]:
    """Prove that classified orphans are equivalent to pre-graph deletion."""

    normal = build_local_plane_fragment_probe(
        projected_input,
        scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
    )
    orphan_mode_ids = {
        str(orphan["mode_atom_id"]) for orphan in normal["local_plane_orphan_table"]
    }
    deletion = deepcopy(dict(projected_input))
    deletion["support_mode_table"] = [
        mode for mode in deletion["support_mode_table"]
        if str(mode["mode_atom_id"]) not in orphan_mode_ids
    ]
    deletion["support_mode_edges"] = [
        edge for edge in deletion["support_mode_edges"]
        if not orphan_mode_ids & {str(value) for value in edge["mode_ids"]}
    ]
    for parent in deletion["support_parent_table"]:
        parent["mode_child_ids"] = [
            str(mode_id) for mode_id in parent.get("mode_child_ids") or ()
            if str(mode_id) not in orphan_mode_ids
        ]
    deletion["transition_parent_table_sha256"] = _table_sha256(
        deletion["support_parent_table"], "parent_atom_id"
    )
    deletion["transition_mode_table_sha256"] = _table_sha256(
        deletion["support_mode_table"], "mode_atom_id"
    )
    deletion["transition_mode_edge_table_sha256"] = _table_sha256(
        deletion["support_mode_edges"], "mode_edge_id"
    )
    reference = build_local_plane_fragment_probe(
        deletion,
        scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
    )
    if normal["local_plane_fragment_table"] != reference["local_plane_fragment_table"]:
        raise AssertionError("orphan changes valid local fragment table")
    if normal["local_plane_pair_records"] != reference["local_plane_pair_records"]:
        raise AssertionError("orphan changes local pair/component/certificate table")
    normal_certificate_ids = sorted(
        component["local_transition_component_id"]
        for pair in normal["local_plane_pair_records"]
        for component in pair["components"]
        if component["geometry_certificate_valid"]
    )
    reference_certificate_ids = sorted(
        component["local_transition_component_id"]
        for pair in reference["local_plane_pair_records"]
        for component in pair["components"]
        if component["geometry_certificate_valid"]
    )
    if normal_certificate_ids != reference_certificate_ids:
        raise AssertionError("orphan changes geometry certificate set")
    downstream_counter_names = {
        "local_plane_fragment_count",
        "valid_singleton_fragment_count",
        "overflow_ineligible_fragment_count",
        "valid_fragment_mode_count",
        "fragment_no_global_match_count",
        "fragment_partial_global_match_count",
        "fragment_full_global_match_count",
        "local_plane_pair_count",
        "angle_eligible_local_pair_count",
        "local_transition_component_count",
        "geometry_certificate_count",
        "pair_no_transverse_parent_count",
        "pair_missing_left_mode_count",
        "pair_missing_right_mode_count",
        "pair_no_cross_mode_rung_count",
        "certificate_no_distinct_full_global_pair_count",
        "certificate_global_pair_line_non_equivalent_count",
        "certificate_global_pair_line_equivalent_count",
        "first_rejection_counts",
        *(f"components_after_{gate}" for gate, _code in REJECTION_ORDER),
    }
    normal_downstream = {
        key: normal["case_counters"].get(key) for key in sorted(downstream_counter_names)
    }
    reference_downstream = {
        key: reference["case_counters"].get(key) for key in sorted(downstream_counter_names)
    }
    normal_downstream["effective_cross_mode_seed_edge_count"] = int(
        normal["case_counters"]["cross_mode_seed_edge_count"]
    ) - int(
        normal["case_counters"]["excluded_cross_mode_seed_edge_count_due_to_orphan"]
    )
    reference_downstream["effective_cross_mode_seed_edge_count"] = int(
        reference["case_counters"]["cross_mode_seed_edge_count"]
    )
    if normal_downstream != reference_downstream:
        raise AssertionError("orphan changes non-orphan downstream counters")
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "kind": "f1_local_plane_orphan_deletion_counterfactual_v2",
        "profile": LOCAL_PLANE_FRAGMENT_PROFILE,
        "orphan_mode_count": len(orphan_mode_ids),
        "orphan_mode_ids_sha256": _sha256_json(sorted(orphan_mode_ids)),
        "fragment_table_identical": True,
        "pair_component_table_identical": True,
        "certificate_set_identical": True,
        "non_orphan_downstream_counters_identical": True,
        "non_orphan_downstream_counters_sha256": _sha256_json(normal_downstream),
        "normal_inference_semantic_sha256": normal["inference_semantic_sha256"],
        "deletion_reference_inference_semantic_sha256": reference[
            "inference_semantic_sha256"
        ],
    }
