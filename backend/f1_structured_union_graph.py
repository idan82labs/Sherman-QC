"""Deterministic, geometry-only IR for research F1 structured inference.

This module is deliberately not connected to the runtime bend detector.  It
sanitizes a scan-only candidate pool, retains alternative line geometries,
completes sparse path/atom incidence from the bounded compact atom table, and
emits proof-carrying continuation and non-binding topology proposals.

No filename, part identifier, expected count, CAD, drawing, or manual label is
accepted as an inference feature.  Candidate source names are evidence only;
they are never part of latent path identity.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from statistics import median
from typing import Any

import numpy as np


BEND_STRAIGHT_COUNTED = "BEND_STRAIGHT_COUNTED"
CORNER_NONCOUNTED = "CORNER_NONCOUNTED"
PERIMETER_NONCOUNTED = "PERIMETER_NONCOUNTED"
HOLE_NONCOUNTED = "HOLE_NONCOUNTED"
JUNCTION_NONCOUNTED = "JUNCTION_NONCOUNTED"
CURVE_NONCOUNTED = "CURVE_NONCOUNTED"
RESIDUAL_NONCOUNTED = "RESIDUAL_NONCOUNTED"

SEMANTIC_LABELS = (
    BEND_STRAIGHT_COUNTED,
    CORNER_NONCOUNTED,
    PERIMETER_NONCOUNTED,
    HOLE_NONCOUNTED,
    JUNCTION_NONCOUNTED,
    CURVE_NONCOUNTED,
    RESIDUAL_NONCOUNTED,
)

INCIDENCE_PROFILE = "INCIDENCE_V1"
CONTINUATION_PROFILE = "CONTINUATION_V1"
GENERIC_MIN_BINS = 2
SUPPORT_ATOM_TABLE_CAP = 20_000
LATENT_PATH_CAP = 32_768

ALLOWED_POOL_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm",
        "inference_contract",
        "source_point_serialization",
        "support_atom_table",
        "support_atom_edges",
        "plane_table",
        "candidates",
        "line_objects",
    }
)

_GEOMETRY_FIELDS = {
    "span_endpoints",
    "line_start",
    "line_end",
    "bend_line_start",
    "bend_line_end",
    "anchor",
    "axis_direction",
    "span_mm",
    "geometry_kind",
}
_JOIN_FIELDS = {
    "candidate_id",
    "bend_id",
    "source_candidate_id",
    "candidate_source",
    "representative_source",
    "cluster_member_ids",
    "cluster_sources",
    "cluster_member_count",
}
_PANEL_FIELDS = {
    "incident_plane_ids",
    "merged_incident_plane_pairs",
    "incident_flange_ids",
    "canonical_flange_pair",
    "candidate_incident_flange_pairs",
    "seed_incident_flange_ids",
    "surface1_id",
    "surface2_id",
    "surface1_normal",
    "surface2_normal",
}
_SUPPORT_FIELDS = {
    "support_atom_ids",
    "support_atom_ids_truncated",
    "support_count",
    "weight",
    "area_proxy_mm2",
    "finite_support_fraction",
    "finite_support_longest_run_fraction",
    "endpoint_support_fraction",
    "median_line_distance_mm",
    "p90_line_distance_mm",
    "two_sided_transition_score",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "cluster_max_finite_support_fraction",
    "cluster_max_finite_support_longest_run_fraction",
    "cluster_max_endpoint_support_fraction",
    "cluster_max_two_sided_transition_score",
    "cluster_max_two_sided_support_balance",
    "cluster_max_two_sided_min_support_fraction",
    "cluster_max_two_sided_normal_coherence",
    "cluster_mean_median_line_distance_mm",
}
ALLOWED_RECORD_FIELDS = frozenset(
    _GEOMETRY_FIELDS | _JOIN_FIELDS | _PANEL_FIELDS | _SUPPORT_FIELDS
)

# Correlated parameterizations count once.  Unknown generators remain visible
# as their own group; source groups are evidence, never identity.
_SOURCE_GROUP = {
    "cylindrical_standard": "cylindrical_family",
    "cylindrical_wide": "cylindrical_family",
    "bend_hypothesis": "patch_transition_family",
    "control_support_strip": "patch_transition_family",
    "strict_blind_owned": "patch_transition_family",
    "evidence_trace": "patch_transition_family",
    "fragment_merge": "patch_transition_family",
    "endpoint_connector": "patch_transition_family",
    "plane_lattice": "plane_lattice",
    "scan_mesh_adjacency": "mesh_adjacency",
    "zero_shot_support_strip": "zero_shot_strip",
}

_SUPPORT_FEATURES = tuple(sorted(_SUPPORT_FIELDS - {"support_atom_ids"}))
_CONTINUATION_RESIDUAL_KEYS = frozenset(
    {"axis", "perpendicular", "gap", "chord", "refit", "overlap"}
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _vector(value: Any, *, allow_zero: bool = True) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("expected a three-dimensional vector")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError("geometry vectors must be finite")
    if not allow_zero and math.sqrt(sum(component * component for component in result)) <= 1e-12:
        raise ValueError("geometry vector must be nonzero")
    return result  # type: ignore[return-value]


def _unit(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("expected a finite three-vector")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError("expected a nonzero vector")
    return array / norm


def _canonical_axis(value: Sequence[float]) -> np.ndarray:
    axis = _unit(value)
    nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if len(nonzero) and axis[int(nonzero[0])] < 0.0:
        axis = -axis
    return axis


def _canonical_line(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("span_endpoints")
    if raw is None and record.get("line_start") is not None:
        raw = (record.get("line_start"), record.get("line_end"))
    if raw is None and record.get("bend_line_start") is not None:
        raw = (record.get("bend_line_start"), record.get("bend_line_end"))
    if raw is not None:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
            raise ValueError("line endpoints must be a 2x3 sequence")
        endpoints = np.asarray((_vector(raw[0]), _vector(raw[1])), dtype=np.float64)
    else:
        anchor = np.asarray(_vector(record.get("anchor")), dtype=np.float64)
        axis = _unit(_vector(record.get("axis_direction"), allow_zero=False))
        span = _finite_float(record.get("span_mm"), default=-1.0)
        if span <= 1e-9:
            raise ValueError("line span must be positive")
        endpoints = np.asarray((anchor - 0.5 * span * axis, anchor + 0.5 * span * axis))
    delta = endpoints[1] - endpoints[0]
    span = float(np.linalg.norm(delta))
    if span <= 1e-9:
        raise ValueError("line span must be nonzero")
    axis = delta / span
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        endpoints = endpoints[::-1]
        axis = -axis
    midpoint = np.mean(endpoints, axis=0)
    # Endpoint reversal must not manufacture distinct hashes through ``-0.0``.
    endpoints[np.abs(endpoints) < 1e-15] = 0.0
    axis[np.abs(axis) < 1e-15] = 0.0
    midpoint[np.abs(midpoint) < 1e-15] = 0.0
    endpoints = np.round(endpoints, 12)
    axis = np.round(axis, 12)
    midpoint = np.round(midpoint, 12)
    span = round(span, 12)
    return {
        "geometry_kind": "LINE",
        "span_endpoints": endpoints.astype(float).tolist(),
        "axis_direction": axis.astype(float).tolist(),
        "midpoint": midpoint.astype(float).tolist(),
        "span_mm": span,
    }


def _geometry_key(geometry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(geometry.get("geometry_kind") or "LINE"),
        *(round(float(value), 6) for value in geometry["axis_direction"]),
        *(round(float(value), 6) for value in geometry["midpoint"]),
        round(float(geometry["span_mm"]), 6),
    )


def _sanitize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(record[key]) for key in ALLOWED_RECORD_FIELDS if key in record}


def strict_geometry_view(candidate_pool: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only pool fields permitted to enter graph construction."""

    view = {
        key: deepcopy(candidate_pool[key])
        for key in ALLOWED_POOL_FIELDS
        if key in candidate_pool
    }
    view["candidates"] = [
        _sanitize_record(record)
        for record in candidate_pool.get("candidates") or ()
        if isinstance(record, Mapping)
    ]
    view["line_objects"] = [
        _sanitize_record(record)
        for record in candidate_pool.get("line_objects") or ()
        if isinstance(record, Mapping)
    ]
    view["support_atom_table"] = [
        {
            key: deepcopy(record[key])
            for key in (
                "atom_id",
                "centroid",
                "normal",
                "local_spacing_mm",
                "curvature_proxy",
                "axis_direction",
                "weight",
                "area_proxy_mm2",
            )
            if key in record
        }
        for record in candidate_pool.get("support_atom_table") or ()
        if isinstance(record, Mapping)
    ]
    view["support_atom_edges"] = [
        {key: deepcopy(record[key]) for key in ("atom_ids", "weight") if key in record}
        for record in candidate_pool.get("support_atom_edges") or ()
        if isinstance(record, Mapping)
    ]
    view["plane_table"] = [
        {
            key: deepcopy(record[key])
            for key in ("plane_id", "normal", "offset", "support_count", "merged_plane_ids")
            if key in record
        }
        for record in candidate_pool.get("plane_table") or ()
        if isinstance(record, Mapping)
    ]
    return view


def _candidate_aliases(candidate: Mapping[str, Any]) -> set[str]:
    source = str(candidate.get("candidate_source") or "")
    aliases: set[str] = set()
    for field in ("candidate_id", "bend_id", "source_candidate_id"):
        value = str(candidate.get(field) or "")
        if value:
            aliases.add(value)
            if source and not value.startswith(f"{source}::"):
                aliases.add(f"{source}::{value}")
    return aliases


def _plain_pair(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    left, right = str(value[0]), str(value[1])
    if not left or not right or left == right:
        return None
    return tuple(sorted((left, right)))


def _namespaced_pair(value: Any, namespace: str) -> tuple[str, str] | None:
    pair = _plain_pair(value)
    if pair is None:
        return None
    prefix = f"{namespace}:"
    return tuple(sorted(value if value.startswith(prefix) else f"{prefix}{value}" for value in pair))


def _panel_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, list[list[str]]]:
    direct_plane: set[tuple[str, str]] = set()
    merged_plane: set[tuple[str, str]] = set()
    flange: set[tuple[str, str]] = set()
    mesh: set[tuple[str, str]] = set()
    for record in records:
        pair = _namespaced_pair(record.get("incident_plane_ids"), "PLANE")
        if pair:
            direct_plane.add(pair)
        for raw in record.get("merged_incident_plane_pairs") or ():
            pair = _namespaced_pair(raw, "PLANE")
            if pair:
                merged_plane.add(pair)
        for field in ("incident_flange_ids", "canonical_flange_pair", "seed_incident_flange_ids"):
            pair = _namespaced_pair(record.get(field), "FLANGE")
            if pair:
                flange.add(pair)
        for raw in record.get("candidate_incident_flange_pairs") or ():
            pair = _namespaced_pair(raw, "FLANGE")
            if pair:
                flange.add(pair)
        pair = _namespaced_pair((record.get("surface1_id"), record.get("surface2_id")), "MESH_SURFACE")
        if pair:
            mesh.add(pair)
    return {
        # Structured direct incidence is authoritative.  Merged provenance is
        # a fallback only; mixing both makes a valid direct pair ambiguous.
        "incident_plane_pairs": [
            list(pair) for pair in sorted(direct_plane or merged_plane)
        ],
        "incident_flange_pairs": [list(pair) for pair in sorted(flange)],
        "incident_mesh_surface_pairs": [list(pair) for pair in sorted(mesh)],
    }


def _primary_panel_pair(evidence: Mapping[str, Sequence[Sequence[str]]]) -> tuple[list[str] | None, str | None]:
    for field, namespace in (
        ("incident_plane_pairs", "PLANE"),
        ("incident_flange_pairs", "FLANGE"),
        ("incident_mesh_surface_pairs", "MESH_SURFACE"),
    ):
        values = list(evidence.get(field) or ())
        if len(values) == 1:
            return list(values[0]), namespace
    return None, None


def _source_group(source: str) -> str:
    return _SOURCE_GROUP.get(source, source or "unknown")


def _source_evidence(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, dict[str, float]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        raw_sources = [str(value) for value in record.get("cluster_sources") or () if str(value)]
        source = str(record.get("candidate_source") or record.get("representative_source") or "unknown")
        if not raw_sources and source != "cross_source_line_object":
            raw_sources = [source]
        if not raw_sources:
            representative = str(record.get("representative_source") or "")
            raw_sources = [representative] if representative else ["unknown"]
        for raw_source in sorted(set(raw_sources)):
            grouped[_source_group(raw_source)].append(record)
    evidence: dict[str, dict[str, float]] = {}
    for group, members in sorted(grouped.items()):
        features: dict[str, float] = {}
        for field in _SUPPORT_FEATURES:
            values = [_finite_float(member.get(field), default=math.nan) for member in members]
            values = [value for value in values if math.isfinite(value)]
            if values:
                features[field] = min(values) if "distance" in field else max(values)
        evidence[group] = features
    return sorted(grouped), evidence


def _normalize_atom_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(records) > SUPPORT_ATOM_TABLE_CAP:
        raise ValueError("support atom table exceeds the exact graph cap")
    atoms: list[dict[str, Any]] = []
    seen: set[str] = set()
    positive_areas = [
        _finite_float(record.get("area_proxy_mm2"))
        for record in records
        if _finite_float(record.get("area_proxy_mm2")) > 0.0
    ]
    median_area = float(median(positive_areas)) if positive_areas else 1.0
    for record in records:
        atom_id = str(record.get("atom_id") or "")
        if not atom_id or atom_id in seen:
            raise ValueError("support atom IDs must be nonempty and unique")
        seen.add(atom_id)
        area = max(_finite_float(record.get("area_proxy_mm2")), 0.0)
        mass = int(np.clip(round(area / max(median_area, 1e-12)), 1, 4))
        atoms.append(
            {
                "atom_id": atom_id,
                "centroid": list(_vector(record.get("centroid"))),
                "normal": list(_vector(record.get("normal"))),
                "local_spacing_mm": max(_finite_float(record.get("local_spacing_mm")), 0.0),
                "curvature_proxy": max(_finite_float(record.get("curvature_proxy")), 0.0),
                "axis_direction": list(_vector(record.get("axis_direction") or (0.0, 0.0, 0.0))),
                "weight": max(_finite_float(record.get("weight")), 0.0),
                "area_proxy_mm2": area,
                "normalized_support_mass": mass,
            }
        )
    return sorted(atoms, key=lambda item: item["atom_id"])


def _normalize_edges(records: Sequence[Mapping[str, Any]], atom_ids: set[str]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], float] = {}
    for record in records:
        pair = _plain_pair(record.get("atom_ids"))
        if pair is None or not set(pair) <= atom_ids:
            raise ValueError("support atom edge references invalid atoms")
        unique[pair] = _finite_float(record.get("weight"))
    return [{"atom_ids": list(pair), "weight": unique[pair]} for pair in sorted(unique)]


def _normalize_planes(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    planes = []
    for record in records:
        raw_id = str(record.get("plane_id") or "")
        if not raw_id:
            raise ValueError("plane IDs must be nonempty")
        normal = _unit(_vector(record.get("normal"), allow_zero=False))
        offset = _finite_float(record.get("offset"))
        nonzero = np.flatnonzero(np.abs(normal) > 1e-12)
        if len(nonzero) and normal[int(nonzero[0])] < 0.0:
            normal = -normal
            offset = -offset
        planes.append(
            {
                "plane_id": raw_id if raw_id.startswith("PLANE:") else f"PLANE:{raw_id}",
                "normal": normal.tolist(),
                "offset": offset,
                "support_count": int(record.get("support_count") or 0),
                "merged_plane_ids": sorted(
                    value if str(value).startswith("PLANE:") else f"PLANE:{value}"
                    for value in record.get("merged_plane_ids") or ()
                ),
            }
        )
    planes.sort(key=lambda item: item["plane_id"])
    if len({item["plane_id"] for item in planes}) != len(planes):
        raise ValueError("plane IDs must be unique")
    return planes


def _raw_paths(
    candidates: Sequence[Mapping[str, Any]], line_objects: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    aliases: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        for alias in _candidate_aliases(candidate):
            aliases[alias].append(candidate)
    seeds = list(line_objects) if line_objects else list(candidates)
    provisional: list[dict[str, Any]] = []
    invalid = 0
    for seed_index, seed in enumerate(seeds):
        members: list[Mapping[str, Any]] = []
        seen_members: set[int] = set()
        for member_id in seed.get("cluster_member_ids") or ():
            for member in aliases.get(str(member_id), ()):
                if id(member) not in seen_members:
                    seen_members.add(id(member))
                    members.append(member)
        if not members:
            for alias in _candidate_aliases(seed):
                for member in aliases.get(alias, ()):
                    if id(member) not in seen_members:
                        seen_members.add(id(member))
                        members.append(member)
        records = [seed, *members]
        geometries: dict[tuple[Any, ...], tuple[dict[str, Any], list[Mapping[str, Any]]]] = {}
        for record in records:
            try:
                geometry = _canonical_line(record)
            except (TypeError, ValueError):
                continue
            key = _geometry_key(geometry)
            if key not in geometries:
                geometries[key] = (geometry, [])
            geometries[key][1].append(record)
        if not geometries:
            invalid += 1
            continue
        for option_index, (geometry, exact_records) in enumerate(
            geometries[key] for key in sorted(geometries)
        ):
            panel = _panel_evidence(records)
            primary_pair, primary_namespace = _primary_panel_pair(panel)
            groups, evidence = _source_evidence(exact_records)
            raw_sources = sorted(
                {
                    str(source)
                    for source in seed.get("cluster_sources") or ()
                    if str(source)
                }
                | {
                    str(record.get("candidate_source") or "")
                    for record in exact_records
                    if str(record.get("candidate_source") or "")
                    and str(record.get("candidate_source")) != "cross_source_line_object"
                }
            )
            geometry_sources = sorted(
                {
                    str(record.get("candidate_source") or "")
                    for record in exact_records
                    if str(record.get("candidate_source") or "")
                    and str(record.get("candidate_source"))
                    != "cross_source_line_object"
                }
            )
            upstream_ids = sorted(
                {
                    str(atom_id)
                    for record in exact_records
                    for atom_id in record.get("support_atom_ids") or ()
                    if str(atom_id)
                }
            )
            provisional.append(
                {
                    **geometry,
                    **panel,
                    "primary_panel_pair": primary_pair,
                    "primary_panel_pair_namespace": primary_namespace,
                    "independent_source_groups": groups,
                    "source_group_evidence": evidence,
                    "raw_member_sources": raw_sources,
                    "geometry_member_sources": geometry_sources,
                    "raw_member_count": len(exact_records),
                    "_upstream_support_atom_ids": upstream_ids,
                    "_upstream_truncated": any(
                        bool(record.get("support_atom_ids_truncated")) for record in exact_records
                    ),
                    "_provisional_order": (seed_index, option_index),
                }
            )
    if len(provisional) > LATENT_PATH_CAP:
        raise ValueError("latent path alternatives exceed the exact graph cap")
    return provisional, invalid


def _unoriented_angle_deg(left: Sequence[float], right: Sequence[float]) -> float:
    cosine = float(np.clip(abs(float(np.dot(_unit(left), _unit(right)))), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _line_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    lp = np.asarray(left["span_endpoints"], dtype=np.float64)
    rp = np.asarray(right["span_endpoints"], dtype=np.float64)
    lu, ru = _unit(left["axis_direction"]), _unit(right["axis_direction"])
    lm, rm = np.mean(lp, axis=0), np.mean(rp, axis=0)
    delta = rm - lm
    offset = max(
        float(np.linalg.norm(delta - lu * float(np.dot(delta, lu)))),
        float(np.linalg.norm(delta - ru * float(np.dot(delta, ru)))),
    )
    li = sorted(float(np.dot(point - lm, lu)) for point in lp)
    ri = sorted(float(np.dot(point - lm, lu)) for point in rp)
    overlap = max(0.0, min(li[1], ri[1]) - max(li[0], ri[0]))
    return {
        "axis_angle_deg": _unoriented_angle_deg(lu, ru),
        "symmetric_line_distance_mm": offset,
        "shorter_span_overlap": overlap / max(min(float(left["span_mm"]), float(right["span_mm"])), 1e-9),
    }


def _median_spacing(atoms: Sequence[Mapping[str, Any]]) -> float:
    values = [float(atom["local_spacing_mm"]) for atom in atoms if float(atom["local_spacing_mm"]) > 0.0]
    return float(median(values)) if values else 1.0


def _incidence_residuals(path: Mapping[str, Any], atom: Mapping[str, Any], case_spacing: float) -> dict[str, float]:
    endpoints = np.asarray(path["span_endpoints"], dtype=np.float64)
    p0, p1 = endpoints
    axis = _unit(path["axis_direction"])
    length = float(np.linalg.norm(p1 - p0))
    centroid = np.asarray(atom["centroid"], dtype=np.float64)
    t = float(np.dot(centroid - p0, axis))
    clipped = p0 + np.clip(t, 0.0, length) * axis
    perpendicular = float(np.linalg.norm(centroid - clipped))
    endpoint = max(0.0, -t, t - length)
    spacing = float(atom.get("local_spacing_mm") or case_spacing)
    tau_perp = min(3.0, max(1.5, 1.25 * spacing))
    tau_end = min(6.0, max(3.0, 2.0 * spacing))
    normal = _unit(atom["normal"])
    normal_angle = math.degrees(math.asin(float(np.clip(abs(np.dot(normal, axis)), 0.0, 1.0))))
    atom_axis = np.asarray(atom.get("axis_direction") or (0.0, 0.0, 0.0), dtype=np.float64)
    axis_available = float(np.linalg.norm(atom_axis)) > 1e-12
    axis_residual = _unoriented_angle_deg(atom_axis, axis) / 15.0 if axis_available else 0.0
    return {
        "perpendicular": perpendicular / tau_perp,
        "endpoint": endpoint / tau_end,
        "normal": normal_angle / 20.0,
        "axis": axis_residual,
        "axis_available": 1.0 if axis_available else 0.0,
        "axial_coordinate": t,
    }


def _incident(residuals: Mapping[str, float]) -> bool:
    return (
        residuals["perpendicular"] <= 1.0
        and residuals["endpoint"] <= 1.0
        and residuals["normal"] <= 1.0
        and (not residuals["axis_available"] or residuals["axis"] <= 1.0)
    )


def _adjacency(edges: Sequence[Mapping[str, Any]], atom_ids: set[str]) -> dict[str, set[str]]:
    result = {atom_id: set() for atom_id in atom_ids}
    for edge in edges:
        left, right = edge["atom_ids"]
        if left in result and right in result:
            result[left].add(right)
            result[right].add(left)
    return result


def _components(atom_ids: set[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    unseen = set(atom_ids)
    components: list[list[str]] = []
    while unseen:
        start = min(unseen)
        queue = [start]
        unseen.remove(start)
        component = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current) or (), reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _longest_consecutive_run(values: Iterable[int]) -> int:
    ordered = sorted(set(values))
    best = current = 0
    previous: int | None = None
    for value in ordered:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        best = max(best, current)
        previous = value
    return best


def _plane_lookup(planes: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for plane in planes:
        lookup[str(plane["plane_id"])] = plane
        for alias in plane.get("merged_plane_ids") or ():
            lookup[str(alias)] = plane
    return lookup


def _atom_side(path: Mapping[str, Any], atom: Mapping[str, Any], planes: Mapping[str, Mapping[str, Any]]) -> str:
    pair = list(path.get("primary_panel_pair") or ())
    if path.get("primary_panel_pair_namespace") != "PLANE" or len(pair) != 2:
        return "UNKNOWN"
    centroid = np.asarray(atom["centroid"], dtype=np.float64)
    normal = _unit(atom["normal"])
    choices = []
    for panel_id in pair:
        plane = planes.get(str(panel_id))
        if plane is None:
            continue
        plane_normal = _unit(plane["normal"])
        angle = _unoriented_angle_deg(normal, plane_normal) / 15.0
        distance = abs(float(np.dot(centroid, plane_normal)) + float(plane["offset"])) / 3.0
        choices.append((angle + distance, str(panel_id)))
    return min(choices)[1] if choices else "UNKNOWN"


def complete_all_path_incidence(
    paths: Sequence[Mapping[str, Any]],
    atoms: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    planes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Complete exact path incidence from the complete compact atom table."""

    result = [dict(path) for path in paths]
    atom_by_id = {str(atom["atom_id"]): atom for atom in atoms}
    atom_table_hash = _json_hash(atoms)
    if not atoms:
        for path in result:
            path.update(
                {
                    "support_atom_ids": [],
                    "support_bin_ids": [],
                    "support_atom_bins": {},
                    "support_atom_sides": {},
                    "support_atom_ids_truncated": bool(path.pop("_upstream_truncated", False)),
                    "incidence_complete": False,
                    "incidence_origin": {
                        "upstream_verified": [],
                        "equivalent_path_propagation": [],
                        "compact_atom_completion": [],
                        "recomputed_from_complete_atom_table": False,
                    },
                    "incidence_certificate": {
                        "tolerance_profile": INCIDENCE_PROFILE,
                        "atom_table_sha256": atom_table_hash,
                        "max_dimensionless_residuals": {
                            "perpendicular": 0.0,
                            "endpoint": 0.0,
                            "normal": 0.0,
                            "axis": 0.0,
                        },
                    },
                }
            )
            path.pop("_upstream_support_atom_ids", None)
        return result

    spacing = _median_spacing(atoms)
    tau_equivalent = max(3.0, 2.0 * spacing)
    equivalent: dict[int, set[int]] = {index: {index} for index in range(len(result))}
    for left in range(len(result)):
        for right in range(left):
            metrics = _line_metrics(result[left], result[right])
            if (
                metrics["axis_angle_deg"] <= 4.0
                and metrics["symmetric_line_distance_mm"] <= tau_equivalent
                and metrics["shorter_span_overlap"] >= 0.75
            ):
                equivalent[left].add(right)
                equivalent[right].add(left)

    full_adjacency = _adjacency(edges, set(atom_by_id))
    planes_by_id = _plane_lookup(planes)
    bin_width = max(3.0, 2.0 * spacing)
    upstream_by_path = [
        {
            atom_id
            for atom_id in path.get("_upstream_support_atom_ids") or ()
            if atom_id in atom_by_id
        }
        for path in result
    ]
    for index, path in enumerate(result):
        own = upstream_by_path[index]
        propagated = {
            atom_id
            for other in equivalent[index] - {index}
            for atom_id in upstream_by_path[other]
        }
        residuals = {
            atom_id: _incidence_residuals(path, atom, spacing)
            for atom_id, atom in atom_by_id.items()
        }
        eligible = {atom_id for atom_id, values in residuals.items() if _incident(values)}
        bins = {
            atom_id: int(math.floor(float(residuals[atom_id]["axial_coordinate"]) / bin_width + 1e-9))
            for atom_id in eligible
        }
        retained: set[str] = set()
        for component in _components(eligible, full_adjacency):
            occupied = {bins[atom_id] for atom_id in component}
            if len(occupied) >= GENERIC_MIN_BINS and _longest_consecutive_run(occupied) >= GENERIC_MIN_BINS:
                retained.update(component)
        ordered = sorted(retained)
        upstream_verified = sorted(own & retained)
        propagation_verified = sorted((propagated & retained) - set(upstream_verified))
        direct = sorted(retained - set(upstream_verified) - set(propagation_verified))
        support_bins = {atom_id: f"BIN_{bins[atom_id]:+07d}" for atom_id in ordered}
        maxima = {
            key: max((float(residuals[atom_id][key]) for atom_id in ordered), default=0.0)
            for key in ("perpendicular", "endpoint", "normal", "axis")
        }
        path.update(
            {
                "support_atom_ids": ordered,
                "support_bin_ids": sorted(set(support_bins.values())),
                "support_atom_bins": support_bins,
                "support_atom_sides": {
                    atom_id: _atom_side(path, atom_by_id[atom_id], planes_by_id)
                    for atom_id in ordered
                },
                "support_atom_ids_truncated": False,
                "incidence_complete": bool(ordered),
                "incidence_origin": {
                    "upstream_verified": upstream_verified,
                    "equivalent_path_propagation": propagation_verified,
                    "compact_atom_completion": direct,
                    "recomputed_from_complete_atom_table": True,
                },
                "incidence_certificate": {
                    "tolerance_profile": INCIDENCE_PROFILE,
                    "atom_table_sha256": atom_table_hash,
                    "max_dimensionless_residuals": maxima,
                    "axial_bin_mm": bin_width,
                    "recomputed_from_complete_atom_table": True,
                },
            }
        )
        path.pop("_upstream_support_atom_ids", None)
        path.pop("_upstream_truncated", None)
    return result


def _completed_support_signature(path: Mapping[str, Any]) -> str:
    return _json_hash(sorted(str(value) for value in path.get("support_atom_ids") or ()))


def _stable_path_id(path: Mapping[str, Any]) -> str:
    canonical = {
        "geometry_kind": str(path.get("geometry_kind") or "LINE"),
        "axis_direction": [round(float(value), 6) for value in path["axis_direction"]],
        "midpoint": [round(float(value), 6) for value in path["midpoint"]],
        "span_mm": round(float(path["span_mm"]), 6),
        "incident_plane_pairs": path.get("incident_plane_pairs") or [],
        "incident_flange_pairs": path.get("incident_flange_pairs") or [],
        "incident_mesh_surface_pairs": path.get("incident_mesh_surface_pairs") or [],
        "completed_support_signature": _completed_support_signature(path),
    }
    return f"LP_{_json_hash(canonical)[:16]}"


def _merge_same_identity(paths: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        grouped[str(path["latent_path_id"])].append(path)
    merged = []
    for path_id, members in sorted(grouped.items()):
        base = deepcopy(members[0])
        base["independent_source_groups"] = sorted(
            {value for member in members for value in member.get("independent_source_groups") or ()}
        )
        combined_evidence: dict[str, dict[str, float]] = defaultdict(dict)
        for member in members:
            for group, features in (member.get("source_group_evidence") or {}).items():
                for field, value in features.items():
                    previous = combined_evidence[group].get(field)
                    combined_evidence[group][field] = (
                        min(previous, value) if previous is not None and "distance" in field
                        else max(previous or 0.0, value)
                    )
        base["source_group_evidence"] = {
            group: dict(sorted(features.items())) for group, features in sorted(combined_evidence.items())
        }
        base["raw_member_sources"] = sorted(
            {value for member in members for value in member.get("raw_member_sources") or ()}
        )
        base["geometry_member_sources"] = sorted(
            {
                value
                for member in members
                for value in member.get("geometry_member_sources") or ()
            }
        )
        base["raw_member_count"] = sum(int(member.get("raw_member_count") or 0) for member in members)
        base.pop("_provisional_order", None)
        merged.append(base)
    return merged


def _panel_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    lp, rp = left.get("primary_panel_pair"), right.get("primary_panel_pair")
    if lp is None or rp is None:
        return True
    return (
        left.get("primary_panel_pair_namespace") == right.get("primary_panel_pair_namespace")
        and tuple(lp) == tuple(rp)
    )


def build_duplicate_relations(paths: Sequence[Mapping[str, Any]], spacing: float) -> list[dict[str, Any]]:
    tolerance = max(3.0, 2.0 * spacing)
    relations = []
    for left in range(len(paths)):
        for right in range(left):
            if not _panel_compatible(paths[left], paths[right]):
                continue
            metrics = _line_metrics(paths[left], paths[right])
            if (
                metrics["axis_angle_deg"] <= 4.0
                and metrics["symmetric_line_distance_mm"] <= tolerance
                and metrics["shorter_span_overlap"] >= 0.75
            ):
                ids = sorted((str(paths[left]["latent_path_id"]), str(paths[right]["latent_path_id"])))
                relations.append(
                    {
                        "duplicate_id": f"DUP_{_json_hash(ids)[:16]}",
                        "path_ids": ids,
                        "relation_kind": "GEOMETRIC_EQUIVALENT",
                        "hard": True,
                        "dimensionless_residuals": {
                            "axis": metrics["axis_angle_deg"] / 4.0,
                            "perpendicular": metrics["symmetric_line_distance_mm"] / tolerance,
                            "overlap_deficit": max(0.0, 0.75 - metrics["shorter_span_overlap"]) / 0.75,
                        },
                        "engineering_tolerance_profile": INCIDENCE_PROFILE,
                    }
                )
    return sorted(relations, key=lambda item: item["duplicate_id"])


def _assign_duplicate_groups(paths: list[dict[str, Any]], relations: Sequence[Mapping[str, Any]]) -> None:
    parent = {str(path["latent_path_id"]): str(path["latent_path_id"]) for path in paths}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for relation in relations:
        left, right = relation["path_ids"]
        lr, rr = find(left), find(right)
        if lr != rr:
            parent[max(lr, rr)] = min(lr, rr)
    members: dict[str, list[str]] = defaultdict(list)
    for path_id in parent:
        members[find(path_id)].append(path_id)
    group_by_id = {
        path_id: f"DUP_GROUP_{_json_hash(sorted(group))[:16]}"
        for group in members.values()
        if len(group) > 1
        for path_id in group
    }
    for path in paths:
        group = group_by_id.get(str(path["latent_path_id"]))
        path["hard_duplicate_group_id"] = group
        path["has_hard_duplicate"] = group is not None


def _label_eligibility(geometry_kind: str) -> dict[str, bool]:
    if geometry_kind == "LINE":
        enabled = {
            BEND_STRAIGHT_COUNTED,
            CORNER_NONCOUNTED,
            PERIMETER_NONCOUNTED,
            RESIDUAL_NONCOUNTED,
        }
    elif geometry_kind == "LOOP":
        enabled = {HOLE_NONCOUNTED, RESIDUAL_NONCOUNTED}
    elif geometry_kind == "POLYLINE":
        enabled = {CURVE_NONCOUNTED, RESIDUAL_NONCOUNTED}
    elif geometry_kind == "NODE":
        enabled = {JUNCTION_NONCOUNTED, RESIDUAL_NONCOUNTED}
    else:
        enabled = {RESIDUAL_NONCOUNTED}
    return {label: label in enabled for label in SEMANTIC_LABELS}


def build_engineering_label_eligibility(
    paths: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, bool] | dict[str, dict[str, bool]]:
    """Expose the frozen pre-learning eligibility table used by the graph.

    Unsupported paths are residual-only even when their geometry kind could
    represent a semantic object; compact scan incidence is a hard prerequisite.
    """

    def for_path(path: Mapping[str, Any]) -> dict[str, bool]:
        eligibility = _label_eligibility(str(path.get("geometry_kind") or ""))
        if not path.get("incidence_complete"):
            return {
                label: label == RESIDUAL_NONCOUNTED for label in SEMANTIC_LABELS
            }
        return eligibility

    if isinstance(paths, Mapping):
        return for_path(paths)
    return {
        str(path.get("latent_path_id") or _json_hash(_geometry_key(path))): for_path(path)
        for path in paths
    }


def _endpoint_nodes(paths: list[dict[str, Any]], spacing: float) -> list[dict[str, Any]]:
    tolerance = min(6.0, max(3.0, 2.0 * spacing))
    endpoints = [
        (str(path["latent_path_id"]), endpoint_id, np.asarray(point, dtype=np.float64))
        for path in paths
        for endpoint_id, point in enumerate(path["span_endpoints"])
    ]
    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(endpoints)):
        for right in range(left):
            if float(np.linalg.norm(endpoints[left][2] - endpoints[right][2])) <= tolerance:
                lr, rr = find(left), find(right)
                parent[max(lr, rr)] = min(lr, rr)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(endpoints)):
        grouped[find(index)].append(index)
    nodes = []
    endpoint_to_node: dict[tuple[str, int], str] = {}
    for group in sorted(grouped.values(), key=lambda values: min(values)):
        members = sorted((endpoints[index][0], endpoints[index][1]) for index in group)
        centroid = np.mean([endpoints[index][2] for index in group], axis=0)
        node_id = f"NODE_{_json_hash({'members': members, 'centroid': np.round(centroid, 6).tolist()})[:16]}"
        for member in members:
            endpoint_to_node[member] = node_id
        nodes.append(
            {
                "node_id": node_id,
                "centroid": centroid.astype(float).tolist(),
                "path_endpoints": [list(member) for member in members],
                "degree": len(members),
                "node_type_eligibility": [
                    "ORDINARY_END",
                    "CONTINUATION",
                    "MULTIWAY_JUNCTION",
                    "PERIMETER_END",
                    "HOLE_NODE",
                ],
            }
        )
    for path in paths:
        path_id = str(path["latent_path_id"])
        path["endpoint_node_ids"] = [endpoint_to_node[(path_id, 0)], endpoint_to_node[(path_id, 1)]]
        path["endpoint_types"] = ["UNRESOLVED", "UNRESOLVED"]
        path["endpoint_node_degrees"] = [
            next(node["degree"] for node in nodes if node["node_id"] == node_id)
            for node_id in path["endpoint_node_ids"]
        ]
    return sorted(nodes, key=lambda item: item["node_id"])


def build_panel_boundary_arcs(
    paths: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize exclusive boundary ownership domains without preselecting an owner.

    Geometric duplicate alternatives share one arc.  Other alternatives on the
    same panel pair remain separate arcs because a panel pair can legitimately
    contain more than one physical bend.
    """

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        pair = tuple(str(value) for value in path.get("primary_panel_pair") or ())
        if len(pair) != 2:
            continue
        duplicate_group = str(path.get("hard_duplicate_group_id") or "")
        key = (
            str(path.get("primary_panel_pair_namespace") or ""),
            *pair,
            duplicate_group or str(path["latent_path_id"]),
        )
        grouped[key].append(path)
    arcs = []
    for key, members in sorted(grouped.items()):
        compatible = sorted(str(path["latent_path_id"]) for path in members)
        geometry = {
            "panel_pair_namespace": key[0],
            "panel_pair": list(key[1:3]),
            "compatible_object_ids": compatible,
        }
        arcs.append(
            {
                "arc_id": f"ARC_{_json_hash(geometry)[:16]}",
                **geometry,
                "residual_owner_allowed": True,
                "ownership_semantics": "EXACTLY_ONE_COMPATIBLE_OBJECT_OR_RESIDUAL",
            }
        )
    return arcs


def _line_distance(point: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> float:
    delta = point - anchor
    return float(np.linalg.norm(delta - axis * float(np.dot(delta, axis))))


def _weighted_tls(atom_ids: Sequence[str], atom_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray([atom_by_id[atom_id]["centroid"] for atom_id in atom_ids], dtype=np.float64)
    weights = np.asarray(
        [max(float(atom_by_id[atom_id].get("normalized_support_mass") or 1.0), 1.0) for atom_id in atom_ids],
        dtype=np.float64,
    )
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / max(float(np.sum(weights)), 1e-12)
    values, vectors = np.linalg.eigh(covariance)
    axis = _canonical_axis(vectors[:, int(np.argmax(values))])
    distances = np.linalg.norm(centered - np.outer(centered @ axis, axis), axis=1)
    return center, axis, float(np.quantile(distances, 0.90))


def _orientation_families(paths: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    families: list[list[Mapping[str, Any]]] = []
    for path in sorted(paths, key=lambda item: str(item["latent_path_id"])):
        for family in families:
            if all(_unoriented_angle_deg(path["axis_direction"], member["axis_direction"]) <= 4.0 for member in family):
                family.append(path)
                break
        else:
            families.append([path])
    return families


def _bridge_path(
    allowed: set[str], starts: set[str], targets: set[str], adjacency: Mapping[str, set[str]]
) -> list[str] | None:
    queue: deque[str] = deque(sorted(starts & allowed))
    predecessor: dict[str, str | None] = {atom_id: None for atom_id in queue}
    target = None
    while queue:
        current = queue.popleft()
        if current in targets:
            target = current
            break
        for neighbor in sorted(adjacency.get(current) or ()):
            if neighbor in allowed and neighbor not in predecessor:
                predecessor[neighbor] = current
                queue.append(neighbor)
    if target is None:
        return None
    path = []
    cursor: str | None = target
    while cursor is not None:
        path.append(cursor)
        cursor = predecessor[cursor]
    return list(reversed(path))


def generate_proof_friendly_continuations(
    paths: Sequence[Mapping[str, Any]],
    atoms: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spacing = _median_spacing(atoms)
    tau_perp = min(6.0, max(3.0, 2.0 * spacing))
    tau_gap = min(12.0, max(6.0, 4.0 * spacing))
    axial_bin = max(3.0, 2.0 * spacing)
    atom_by_id = {str(atom["atom_id"]): atom for atom in atoms}
    adjacency = _adjacency(edges, set(atom_by_id))
    eligible_paths = [
        path
        for path in paths
        if path.get("incidence_complete")
        and path.get("support_atom_ids")
        and path.get("primary_panel_pair") is not None
        and path.get("straight_line_eligible")
        and not path.get("has_hard_duplicate")
    ]
    by_pair: dict[tuple[str, tuple[str, str]], list[Mapping[str, Any]]] = defaultdict(list)
    for path in eligible_paths:
        by_pair[
            (
                str(path["primary_panel_pair_namespace"]),
                tuple(str(value) for value in path["primary_panel_pair"]),
            )
        ].append(path)
    proposals: list[dict[str, Any]] = []
    for pair_key, pair_paths in sorted(by_pair.items()):
        for family in _orientation_families(pair_paths):
            weighted = np.zeros(3, dtype=np.float64)
            for path in family:
                mass = sum(
                    float(atom_by_id[atom_id].get("normalized_support_mass") or 1.0)
                    for atom_id in path["support_atom_ids"]
                )
                weighted += mass * _canonical_axis(path["axis_direction"])
            family_axis = _canonical_axis(weighted)
            rows = []
            for path in family:
                endpoints = np.asarray(path["span_endpoints"], dtype=np.float64)
                projections = endpoints @ family_axis
                left_endpoint = int(np.argmin(projections))
                right_endpoint = int(np.argmax(projections))
                rows.append(
                    {
                        "path": path,
                        "lo": float(np.min(projections)),
                        "hi": float(np.max(projections)),
                        "left_endpoint": left_endpoint,
                        "right_endpoint": right_endpoint,
                    }
                )
            rows.sort(key=lambda row: (row["lo"], row["hi"], str(row["path"]["latent_path_id"])))
            candidates: dict[tuple[str, str], dict[str, Any]] = {}
            for left_index, left in enumerate(rows):
                for right in rows[left_index + 1 :]:
                    first, second = left["path"], right["path"]
                    if right["lo"] < left["lo"]:
                        continue
                    angle = _unoriented_angle_deg(first["axis_direction"], second["axis_direction"])
                    first_mid = np.asarray(first["midpoint"], dtype=np.float64)
                    second_mid = np.asarray(second["midpoint"], dtype=np.float64)
                    perpendicular = max(
                        _line_distance(first_mid, second_mid, _unit(second["axis_direction"])),
                        _line_distance(second_mid, first_mid, _unit(first["axis_direction"])),
                    )
                    gap = max(0.0, right["lo"] - left["hi"])
                    overlap = max(0.0, min(left["hi"], right["hi"]) - max(left["lo"], right["lo"]))
                    maximum_overlap = max(2.0 * axial_bin, 0.10 * min(float(first["span_mm"]), float(second["span_mm"])))
                    first_point = np.asarray(first["span_endpoints"][left["right_endpoint"]], dtype=np.float64)
                    second_point = np.asarray(second["span_endpoints"][right["left_endpoint"]], dtype=np.float64)
                    chord = second_point - first_point
                    if float(np.linalg.norm(chord)) <= 1e-12:
                        continue
                    chord_angle = _unoriented_angle_deg(chord, family_axis)
                    support_union = sorted(set(first["support_atom_ids"]) | set(second["support_atom_ids"]))
                    if len(support_union) < 2:
                        continue
                    tls_center, tls_axis, p90 = _weighted_tls(support_union, atom_by_id)
                    residuals = {
                        "axis": angle / 4.0,
                        "perpendicular": perpendicular / tau_perp,
                        "gap": gap / tau_gap,
                        "chord": chord_angle / 6.0,
                        "refit": p90 / tau_perp,
                        "overlap": overlap / maximum_overlap,
                    }
                    if any(value > 1.0 for value in residuals.values()):
                        continue
                    low = min(left["hi"] - axial_bin, right["lo"] - axial_bin)
                    high = max(left["hi"] + axial_bin, right["lo"] + axial_bin)
                    allowed = set()
                    for atom_id, atom in atom_by_id.items():
                        atom_residuals = _incidence_residuals(
                            {
                                "span_endpoints": [
                                    (tls_center + low * family_axis - float(np.dot(tls_center, family_axis)) * family_axis).tolist(),
                                    (tls_center + high * family_axis - float(np.dot(tls_center, family_axis)) * family_axis).tolist(),
                                ],
                                "axis_direction": tls_axis.tolist(),
                            },
                            atom,
                            spacing,
                        )
                        projection = float(np.dot(np.asarray(atom["centroid"], dtype=np.float64), family_axis))
                        if low <= projection <= high and _incident(atom_residuals):
                            allowed.add(atom_id)
                    first_bins = {
                        atom_id: int(math.floor(float(np.dot(np.asarray(atom_by_id[atom_id]["centroid"]), family_axis)) / axial_bin))
                        for atom_id in first["support_atom_ids"]
                    }
                    second_bins = {
                        atom_id: int(math.floor(float(np.dot(np.asarray(atom_by_id[atom_id]["centroid"]), family_axis)) / axial_bin))
                        for atom_id in second["support_atom_ids"]
                    }
                    first_cut = max(first_bins.values()) - 1
                    second_cut = min(second_bins.values()) + 1
                    starts = {atom_id for atom_id, bin_id in first_bins.items() if bin_id >= first_cut}
                    targets = {atom_id for atom_id, bin_id in second_bins.items() if bin_id <= second_cut}
                    bridge = _bridge_path(allowed, starts, targets, adjacency)
                    if bridge is None:
                        continue
                    bridge_bins = {
                        int(math.floor(float(np.dot(np.asarray(atom_by_id[atom_id]["centroid"]), family_axis)) / axial_bin))
                        for atom_id in bridge
                    }
                    if bridge_bins and max(bridge_bins) - min(bridge_bins) + 1 - len(bridge_bins) > 1:
                        continue
                    path_ids = [str(first["latent_path_id"]), str(second["latent_path_id"])]
                    certificate = {
                        "path_ids": path_ids,
                        "bridge_atom_ids": bridge,
                        "residuals": {key: round(value, 12) for key, value in residuals.items()},
                    }
                    candidates[tuple(path_ids)] = {
                        "continuation_id": f"CONT_{_json_hash(certificate)[:16]}",
                        "path_ids": path_ids,
                        "endpoint_ids": [left["right_endpoint"], right["left_endpoint"]],
                        "dimensionless_residuals": residuals,
                        "engineering_tolerance_profile": CONTINUATION_PROFILE,
                        "bridge_atom_ids": bridge,
                        "bridge_certificate_sha256": _json_hash(certificate),
                    }
            successor: dict[str, tuple[tuple[Any, ...], tuple[str, str]]] = {}
            predecessor: dict[str, tuple[tuple[Any, ...], tuple[str, str]]] = {}
            for key, proposal in candidates.items():
                residuals = proposal["dimensionless_residuals"]
                rank = (
                    residuals["gap"],
                    residuals["perpendicular"],
                    residuals["axis"],
                    residuals["refit"],
                    residuals["chord"],
                    key[1],
                )
                if key[0] not in successor or rank < successor[key[0]][0]:
                    successor[key[0]] = (rank, key)
                pred_rank = (*rank[:-1], key[0])
                if key[1] not in predecessor or pred_rank < predecessor[key[1]][0]:
                    predecessor[key[1]] = (pred_rank, key)
            proposals.extend(
                candidates[key]
                for key in sorted(candidates)
                if successor.get(key[0], (None, None))[1] == key
                and predecessor.get(key[1], (None, None))[1] == key
            )
    return validate_continuations(paths, proposals)


def validate_continuations(
    paths: Sequence[Mapping[str, Any]], continuations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Validate endpoint-aware, proof-carrying continuation forest records."""

    path_by_id = {str(path["latent_path_id"]): path for path in paths}
    result = []
    endpoint_degree: dict[tuple[str, int], int] = defaultdict(int)
    parent = {path_id: path_id for path_id in path_by_id}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    seen_ids: set[str] = set()
    for raw in sorted(continuations, key=lambda item: str(item.get("continuation_id") or "")):
        continuation_id = str(raw.get("continuation_id") or "")
        if not continuation_id or continuation_id in seen_ids:
            raise ValueError("continuation IDs must be nonempty and unique")
        seen_ids.add(continuation_id)
        path_ids = list(raw.get("path_ids") or ())
        endpoint_ids = list(raw.get("endpoint_ids") or ())
        if len(path_ids) != 2 or len(endpoint_ids) != 2:
            raise ValueError("continuation requires two path and endpoint IDs")
        if any(str(path_id) not in path_by_id for path_id in path_ids):
            raise ValueError("continuation references an absent path")
        endpoint_ids = [int(value) for value in endpoint_ids]
        if any(value not in (0, 1) for value in endpoint_ids):
            raise ValueError("continuation endpoints must be 0 or 1")
        left, right = (path_by_id[str(path_ids[0])], path_by_id[str(path_ids[1])])
        if (
            left.get("primary_panel_pair_namespace") != right.get("primary_panel_pair_namespace")
            or tuple(left.get("primary_panel_pair") or ()) != tuple(right.get("primary_panel_pair") or ())
        ):
            raise ValueError("continuations require the same namespaced panel pair")
        residuals = dict(raw.get("dimensionless_residuals") or {})
        if set(residuals) != _CONTINUATION_RESIDUAL_KEYS:
            raise ValueError("incomplete continuation residual certificate")
        if any(not math.isfinite(float(residuals[key])) for key in _CONTINUATION_RESIDUAL_KEYS):
            raise ValueError("non-finite continuation residual")
        if any(float(residuals[key]) > 1.0 or float(residuals[key]) < 0.0 for key in _CONTINUATION_RESIDUAL_KEYS):
            raise ValueError("continuation outside frozen engineering tolerance")
        if raw.get("engineering_tolerance_profile") != CONTINUATION_PROFILE:
            raise ValueError("continuation tolerance profile mismatch")
        if not raw.get("bridge_certificate_sha256"):
            raise ValueError("continuation lacks scan-support certificate")
        for path_id, endpoint_id in zip(path_ids, endpoint_ids):
            key = (str(path_id), endpoint_id)
            endpoint_degree[key] += 1
            if endpoint_degree[key] > 1:
                raise ValueError("continuation endpoint degree exceeds one")
        left_root, right_root = find(str(path_ids[0])), find(str(path_ids[1]))
        if left_root == right_root:
            raise ValueError("continuation proposal graph must be acyclic")
        parent[right_root] = left_root
        result.append(deepcopy(dict(raw)))
    return result


def _panel_pair_groups(paths: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in paths:
        if path.get("primary_panel_pair_namespace") != "PLANE":
            continue
        pair = _plain_pair(path.get("primary_panel_pair"))
        if pair:
            grouped[pair].append(str(path["latent_path_id"]))
    return {pair: sorted(ids) for pair, ids in sorted(grouped.items())}


def generate_relational_factor_proposals(
    paths: Sequence[Mapping[str, Any]], endpoint_nodes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Emit generic, disabled topology hypotheses without assigning labels."""

    factors: list[dict[str, Any]] = []
    pair_paths = _panel_pair_groups(paths)
    edges = set(pair_paths)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    panels = sorted(adjacency)
    for left_index, left in enumerate(panels):
        for right in panels[left_index + 1 :]:
            if tuple(sorted((left, right))) in edges:
                continue
            common = sorted(adjacency[left] & adjacency[right])
            if len(common) < 2:
                continue
            cross_pairs = sorted(tuple(sorted((host, wall))) for wall in common for host in (left, right))
            if any(pair not in edges for pair in cross_pairs):
                continue
            chord_pairs = sorted(pair for pair in edges if pair[0] in common and pair[1] in common)
            if not chord_pairs:
                continue
            possible = len(common) * (len(common) - 1) / 2.0
            cross_spans = [
                median(
                    float(next(path["span_mm"] for path in paths if path["latent_path_id"] == path_id))
                    for path_id in pair_paths[pair]
                )
                for pair in cross_pairs
            ]
            chord_spans = [
                median(
                    float(next(path["span_mm"] for path in paths if path["latent_path_id"] == path_id))
                    for path_id in pair_paths[pair]
                )
                for pair in chord_pairs
            ]
            members = {
                "host_panels": [left, right],
                "common_neighbor_panels": common,
                "cross_relations": [
                    {"panel_pair": list(pair), "path_ids": pair_paths[pair]} for pair in cross_pairs
                ],
                "chord_relations": [
                    {"panel_pair": list(pair), "path_ids": pair_paths[pair]} for pair in chord_pairs
                ],
            }
            factors.append(
                {
                    "factor_id": f"FACTOR_BICLIQUE_{_json_hash(members)[:16]}",
                    "factor_kind": "BICLIQUE_CHORD_CANDIDATE",
                    "relation_members": members,
                    "label_implications": [],
                    "node_type_implications": [],
                    "feature_vector": {
                        "common_neighbor_count": len(common),
                        "topology_coverage": len(chord_pairs) / max(possible, 1.0),
                        "median_cross_to_chord_span_ratio": float(median(cross_spans)) / max(float(median(chord_spans)), 1e-9),
                    },
                    "weight_status": "DISABLED_PENDING_FAMILY_EVIDENCE",
                    "conflicting_factor_ids": [],
                }
            )
    for node in endpoint_nodes:
        if int(node.get("degree") or 0) < 3:
            continue
        members = list(node.get("path_endpoints") or ())
        factors.append(
            {
                "factor_id": f"FACTOR_JUNCTION_{_json_hash(members)[:16]}",
                "factor_kind": "ENDPOINT_MULTIWAY_CANDIDATE",
                "relation_members": {"node_id": node["node_id"], "path_endpoints": members},
                "label_implications": [],
                "node_type_implications": [],
                "feature_vector": {"endpoint_degree": int(node["degree"])},
                "weight_status": "DISABLED_PENDING_FAMILY_EVIDENCE",
                "conflicting_factor_ids": [],
            }
        )
    return sorted(factors, key=lambda item: item["factor_id"])


def build_factor_conflicts(factors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only pairwise semantic/node contradictions; disjoint factors coexist."""

    assignments = []
    for factor in factors:
        labels = {
            (str(item.get("path_id")), str(item.get("label")))
            for item in factor.get("label_implications") or ()
        }
        nodes = {
            (
                str(item.get("path_id")),
                int(item.get("endpoint_id") or 0),
                str(item.get("node_type")),
            )
            for item in factor.get("node_type_implications") or ()
        }
        required_atoms = {
            str(value) for value in factor.get("required_support_atom_ids") or () if str(value)
        }
        assignments.append((factor, labels, nodes, required_atoms))
    conflicts = []
    for left in range(len(assignments)):
        for right in range(left):
            lf, ll, ln, la = assignments[left]
            rf, rl, rn, ra = assignments[right]
            label_conflict = any(
                left_path == right_path and left_label != right_label
                for left_path, left_label in ll
                for right_path, right_label in rl
            )
            node_conflict = any(
                lp == rp and le == re and lt != rt
                for lp, le, lt in ln
                for rp, re, rt in rn
            )
            support_conflict = bool(la & ra)
            if label_conflict or node_conflict or support_conflict:
                ids = sorted((str(lf["factor_id"]), str(rf["factor_id"])))
                reason = (
                    "CONTRADICTORY_LABELS"
                    if label_conflict
                    else "CONTRADICTORY_NODE_TYPES"
                    if node_conflict
                    else "EXCLUSIVE_SUPPORT_CONFLICT"
                )
                conflicts.append(
                    {
                        "factor_ids": ids,
                        "reason": reason,
                    }
                )
    return sorted(conflicts, key=lambda item: tuple(item["factor_ids"]))


def _canonical_graph_hash(payload: Mapping[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key not in {"graph_sha256", "diagnostics"}}
    return _json_hash(material)


def validate_structured_graph(graph: Mapping[str, Any]) -> None:
    paths = list(graph.get("latent_paths") or ())
    path_ids = [str(path.get("latent_path_id") or "") for path in paths]
    if any(not value for value in path_ids) or len(path_ids) != len(set(path_ids)):
        raise ValueError("latent path IDs must be nonempty and unique")
    atom_ids = {str(atom["atom_id"]) for atom in graph.get("support_atom_table") or ()}
    for path in paths:
        if set(path.get("label_eligibility") or {}) != set(SEMANTIC_LABELS):
            raise ValueError("every path needs complete seven-label eligibility")
        if not set(path.get("support_atom_ids") or ()) <= atom_ids:
            raise ValueError("path incidence references an absent atom")
        if path.get("support_atom_ids_truncated") and path.get("incidence_complete"):
            raise ValueError("completed exact incidence cannot remain truncated")
        pair = list(path.get("primary_panel_pair") or ())
        namespace = path.get("primary_panel_pair_namespace")
        if pair and (len(pair) != 2 or any(not str(value).startswith(f"{namespace}:") for value in pair)):
            raise ValueError("primary panel pair violates namespace discipline")
    validate_continuations(paths, list(graph.get("continuations") or ()))
    factor_ids = {str(factor["factor_id"]) for factor in graph.get("topology_factors") or ()}
    for conflict in graph.get("factor_conflicts") or ():
        if not set(conflict.get("factor_ids") or ()) <= factor_ids:
            raise ValueError("factor conflict references an absent factor")


def build_structured_union_graph(candidate_pool: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete production-neutral structured research graph."""

    pool = strict_geometry_view(candidate_pool)
    contract = dict(pool.get("inference_contract") or {})
    if any(
        bool(contract.get(key))
        for key in (
            "part_identifier_exposed",
            "expected_count_exposed",
            "manual_truth_exposed",
            "cad_exposed",
        )
    ):
        raise ValueError("candidate pool violates the strict-blind inference contract")
    if pool.get("source_point_serialization") is not False:
        raise ValueError("candidate pool must not serialize source points")

    atoms = _normalize_atom_table(pool.get("support_atom_table") or ())
    atom_ids = {str(atom["atom_id"]) for atom in atoms}
    edges = _normalize_edges(pool.get("support_atom_edges") or (), atom_ids)
    planes = _normalize_planes(pool.get("plane_table") or ())
    provisional, invalid_count = _raw_paths(
        pool.get("candidates") or (), pool.get("line_objects") or ()
    )
    completed = complete_all_path_incidence(provisional, atoms, edges, planes)
    for path in completed:
        path["latent_path_id"] = _stable_path_id(path)
    paths = _merge_same_identity(completed)
    spacing = _median_spacing(atoms)
    duplicate_relations = build_duplicate_relations(paths, spacing)
    _assign_duplicate_groups(paths, duplicate_relations)
    for path in paths:
        path["straight_line_eligible"] = path.get("geometry_kind") == "LINE"
        path["label_eligibility"] = build_engineering_label_eligibility(path)
        path["semantic_label"] = RESIDUAL_NONCOUNTED
        path["counted"] = False
        path["relations"] = []
    paths.sort(key=lambda item: str(item["latent_path_id"]))
    endpoint_nodes = _endpoint_nodes(paths, spacing)
    panel_boundary_arcs = build_panel_boundary_arcs(paths)
    continuations = generate_proof_friendly_continuations(paths, atoms, edges)
    factors = generate_relational_factor_proposals(paths, endpoint_nodes)
    factor_conflicts = build_factor_conflicts(factors)
    factor_conflict_lookup: dict[str, list[str]] = defaultdict(list)
    for conflict in factor_conflicts:
        left, right = conflict["factor_ids"]
        factor_conflict_lookup[left].append(right)
        factor_conflict_lookup[right].append(left)
    for factor in factors:
        factor["conflicting_factor_ids"] = sorted(factor_conflict_lookup[factor["factor_id"]])

    complete_count = sum(bool(path["incidence_complete"]) for path in paths)
    supported_count = sum(bool(path["support_atom_ids"]) for path in paths)
    propagated_count = sum(
        bool(path["incidence_origin"]["equivalent_path_propagation"]) for path in paths
    )
    hard_pairs = [list(relation["path_ids"]) for relation in duplicate_relations if relation["hard"]]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "sherman_qc_f1_structured_union_graph",
        "algorithm": "f1_structured_union_graph_ir_v2",
        "inference_contract": {
            "uses_part_identifier": False,
            "uses_filename_features": False,
            "uses_expected_count": False,
            "uses_manual_truth": False,
            "uses_cad": False,
            "source_point_serialization": False,
            "sanitized_pool_fields": sorted(ALLOWED_POOL_FIELDS),
        },
        "engineering_profiles": {
            "incidence": INCIDENCE_PROFILE,
            "continuation": CONTINUATION_PROFILE,
            "generic_minimum_axial_bins": GENERIC_MIN_BINS,
        },
        "semantic_labels": list(SEMANTIC_LABELS),
        "support_atom_table_sha256": _json_hash(atoms),
        "plane_table": planes,
        "support_atom_table": atoms,
        "support_atom_edges": edges,
        "latent_paths": paths,
        "endpoint_nodes": endpoint_nodes,
        "continuations": continuations,
        "duplicate_pairs": duplicate_relations,
        "duplicate_relations": duplicate_relations,
        "hard_duplicate_pairs": hard_pairs,
        "panel_boundary_arcs": panel_boundary_arcs,
        "topology_factors": factors,
        "factor_conflicts": factor_conflicts,
        "invariants": {
            "strict_geometry_whitelist_enforced": True,
            "source_independent_path_identity": True,
            "panel_namespaces_separate": True,
            "incidence_scope": "complete_serialized_compact_atom_table",
            "continuations_require_scan_bridge_certificate": True,
            "continuation_proposal_graph_is_forest": True,
            "topology_factors_are_nonbinding": True,
            "topology_factors_mutate_labels": False,
            "panel_boundary_arc_model_implemented": True,
        },
        "diagnostics": {
            "ignored_top_level_metadata_fields": sorted(set(candidate_pool) - ALLOWED_POOL_FIELDS),
            "input_candidate_count": len(pool.get("candidates") or ()),
            "input_line_object_count": len(pool.get("line_objects") or ()),
            "invalid_line_geometry_count": invalid_count,
            "latent_path_count": len(paths),
            "serialized_plane_count": len(planes),
            "serialized_support_atom_count": len(atoms),
            "serialized_support_atom_edge_count": len(edges),
            "incidence_complete_path_count": complete_count,
            "supported_path_count": supported_count,
            "zero_support_path_count": len(paths) - supported_count,
            "equivalent_path_propagation_count": propagated_count,
            "hard_duplicate_pair_count": len(hard_pairs),
            "continuation_proposal_count": len(continuations),
            "endpoint_node_count": len(endpoint_nodes),
            "topology_factor_count": len(factors),
            "factor_conflict_count": len(factor_conflicts),
            "panel_boundary_arc_count": len(panel_boundary_arcs),
        },
    }
    validate_structured_graph(payload)
    payload["graph_sha256"] = _canonical_graph_hash(payload)
    return payload


def build_biclique_chord_regression_factors(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Nonbinding historical alias: regenerate generic disabled factors only."""

    result = deepcopy(dict(graph))
    factors = generate_relational_factor_proposals(
        result.get("latent_paths") or (), result.get("endpoint_nodes") or ()
    )
    result["topology_factors"] = factors
    result["factor_conflicts"] = build_factor_conflicts(factors)
    result["bend_count"] = count_bend_components(
        result.get("latent_paths") or (), result.get("continuations") or ()
    )
    result["graph_sha256"] = _canonical_graph_hash(result)
    return result


def apply_relational_chord_corner_factor(
    graph: Mapping[str, Any], **_ignored: Any
) -> dict[str, Any]:
    """Deprecated nonbinding alias; it never assigns bend/corner labels."""

    result = build_biclique_chord_regression_factors(graph)
    result["bend_count"] = count_bend_components(result["latent_paths"], ())
    result["graph_sha256"] = _canonical_graph_hash(result)
    return result


def apply_457_biclique_chord_regression_labels(
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the historical 457 label mutation in an explicit regression API.

    This function is intentionally excluded from generic graph construction.
    It preserves the designed 457 post-lock regression while the production-
    neutral factor builder emits disabled proposals with no label mutation.
    """

    result = build_biclique_chord_regression_factors(graph)
    paths = [deepcopy(path) for path in result.get("latent_paths") or ()]
    by_id = {str(path["latent_path_id"]): path for path in paths}

    def strength(path_id: str) -> tuple[Any, ...]:
        path = by_id[path_id]
        transition = max(
            (
                _finite_float(features.get("two_sided_transition_score"))
                for features in (path.get("source_group_evidence") or {}).values()
            ),
            default=0.0,
        )
        return (
            "plane_lattice" in (path.get("geometry_member_sources") or ()),
            len(path.get("raw_member_sources") or ()),
            transition,
            float(path.get("span_mm") or 0.0),
            path_id,
        )

    candidates = [
        factor
        for factor in result.get("topology_factors") or ()
        if factor.get("factor_kind") == "BICLIQUE_CHORD_CANDIDATE"
        and int((factor.get("feature_vector") or {}).get("common_neighbor_count") or 0) >= 4
        and float(
            (factor.get("feature_vector") or {}).get(
                "median_cross_to_chord_span_ratio"
            )
            or 0.0
        )
        >= 2.5
    ]
    candidates.sort(
        key=lambda factor: (
            -float(
                (factor.get("feature_vector") or {}).get(
                    "median_cross_to_chord_span_ratio"
                )
                or 0.0
            ),
            -float((factor.get("feature_vector") or {}).get("topology_coverage") or 0.0),
            str(factor["factor_id"]),
        )
    )
    selected = candidates[0] if candidates else None
    compatibility_factor: dict[str, Any] | None = None
    if selected is not None:
        members = selected["relation_members"]
        counted_ids = sorted(
            max(map(str, relation["path_ids"]), key=strength)
            for relation in members["cross_relations"]
        )
        corner_ids = sorted(
            max(map(str, relation["path_ids"]), key=strength)
            for relation in members["chord_relations"]
        )
        counted_set, corner_set = set(counted_ids), set(corner_ids)
        corner_pairs = {
            tuple(map(str, relation["panel_pair"]))
            for relation in members["chord_relations"]
        }
        for path in paths:
            path_id = str(path["latent_path_id"])
            if path_id in counted_set:
                path["semantic_label"] = BEND_STRAIGHT_COUNTED
                path["counted"] = True
                path["relations"].append("457_regression_host_wall_boundary")
            elif path_id in corner_set:
                path["semantic_label"] = CORNER_NONCOUNTED
                path["counted"] = False
                path["endpoint_types"] = ["MULTIWAY_JUNCTION", "MULTIWAY_JUNCTION"]
                path["relations"].append("457_regression_two_ended_wall_chord")
            if tuple(path.get("primary_panel_pair") or ()) in corner_pairs:
                path["endpoint_types"] = ["MULTIWAY_JUNCTION", "MULTIWAY_JUNCTION"]
            if path.get("primary_panel_pair_namespace") == "PLANE":
                path["primary_panel_pair"] = [
                    str(value).removeprefix("PLANE:")
                    for value in path.get("primary_panel_pair") or ()
                ]
        compatibility_factor = {
            "factor_id": "RELATIONAL_CHORD_CORNER_457_REGRESSION",
            "factor_kind": "BICLIQUE_CHORD_457_REGRESSION_ONLY",
            "host_panels": [
                str(value).removeprefix("PLANE:")
                for value in members["host_panels"]
            ],
            "wall_panels": [
                str(value).removeprefix("PLANE:")
                for value in members["common_neighbor_panels"]
            ],
            "counted_path_ids": counted_ids,
            "noncounted_path_ids": corner_ids,
            "feature_vector": deepcopy(selected["feature_vector"]),
            "weight_status": "REGRESSION_COMPATIBILITY_ONLY",
        }
    result["latent_paths"] = sorted(paths, key=lambda path: str(path["latent_path_id"]))
    result["topology_factors"] = (
        [compatibility_factor] if compatibility_factor is not None else []
    )
    result["factor_conflicts"] = []
    result["bend_count"] = count_bend_components(result["latent_paths"], ())
    result["graph_sha256"] = _canonical_graph_hash(result)
    return result


def count_bend_components(
    latent_paths: Sequence[Mapping[str, Any]],
    continuations: Sequence[Mapping[str, Any] | Sequence[str]],
) -> int:
    """Count counted continuation components with endpoint-aware validation."""

    counted = {
        str(path["latent_path_id"]): path
        for path in latent_paths
        if path.get("semantic_label") == BEND_STRAIGHT_COUNTED
    }
    if not counted:
        return 0
    normalized = []
    for index, raw in enumerate(continuations):
        if isinstance(raw, Mapping):
            if not bool(raw.get("selected", True)):
                continue
            path_ids = list(raw.get("path_ids") or ())
            endpoint_ids = list(raw.get("endpoint_ids") or (1, 0))
        else:
            path_ids = list(raw)
            endpoint_ids = [1, 0]
        if len(path_ids) != 2 or any(str(value) not in counted for value in path_ids):
            raise ValueError("continuations may join counted paths only")
        left, right = counted[str(path_ids[0])], counted[str(path_ids[1])]
        if (
            left.get("primary_panel_pair_namespace") != right.get("primary_panel_pair_namespace")
            or tuple(left.get("primary_panel_pair") or ()) != tuple(right.get("primary_panel_pair") or ())
        ):
            raise ValueError("continuations require the same incident panel pair")
        normalized.append((f"COUNT_{index}", [str(value) for value in path_ids], [int(value) for value in endpoint_ids]))
    parent = {path_id: path_id for path_id in counted}
    endpoint_degree: dict[tuple[str, int], int] = defaultdict(int)

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for _continuation_id, path_ids, endpoint_ids in normalized:
        for path_id, endpoint_id in zip(path_ids, endpoint_ids):
            endpoint_degree[(path_id, endpoint_id)] += 1
            if endpoint_degree[(path_id, endpoint_id)] > 1:
                raise ValueError("continuation endpoint degree exceeds one")
        left_root, right_root = find(path_ids[0]), find(path_ids[1])
        if left_root == right_root:
            raise ValueError("counted continuation graph must be acyclic")
        parent[right_root] = left_root
    return len(counted) - len(normalized)


def solve_structured_union_baseline(candidate_pool: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility entry point for the historical 457 regression baseline."""

    return apply_457_biclique_chord_regression_labels(
        build_structured_union_graph(candidate_pool)
    )
