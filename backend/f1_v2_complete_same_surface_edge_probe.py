"""Inference-safe V2 complete-SAME_SURFACE edge-semantics probe.

This is a profile-separated research lane.  It consumes only the frozen
TRANSITION_ATOM_V2 compact projection, retains every inherited local rung,
replaces mutual-top-eight SAME_SURFACE pruning with the complete eligible
26-neighbour graph, and then runs the frozen LPF V2 geometry.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from typing import Any

import numpy as np

from backend import f1_local_plane_fragment_probe as lpf


PRIMITIVE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "f1_v2_complete_same_surface_edge_probe_v1"
INPUT_PROJECTION_PROFILE = "V2_COMPLETE_EDGE_INPUT_PROJECTION_V1"
EDGE_RULE_ID = "COMPLETE_ELIGIBLE_SAME_SURFACE_NO_MUTUAL_TOPK_V1"
CONTROL_EDGE_RULE_ID = "TRANSITION_ATOM_V2_MUTUAL_TOP8_SAME_SURFACE"
COUNTERFACTUAL_PROFILE = "DELETE_ADDED_SAME_SURFACE_EDGES_RESTORES_LPF_V2_V1"

ALLOWED_EDGE_PROBE_INPUT_FIELDS = tuple(lpf.ALLOWED_POOL_PROJECTION_FIELDS)
COMPLETE_EDGE_FIELDS = {
    "mode_edge_id", "mode_ids", "edge_kind", "spatial_residual",
    "normal_residual", "plane_residual",
}
EDGE_DELTA_FIELDS = {
    "mode_edge_id", "mode_ids", "parent_atom_ids", "parent_voxel_keys",
    "spatial_residual", "normal_residual", "plane_residual", "delta_kind",
    "edge_delta_sha256",
}
EDGE_SEMANTICS_ATTESTATION_FIELDS = {
    "edge_rule_id", "control_edge_rule_id", "same_surface_angle_deg",
    "surface_gap_tolerance_mm", "parent_voxel_level", "parent_voxel_size_mm",
    "adjacent_parent_pair_count", "eligible_mode_pair_evaluation_count",
    "current_same_surface_edge_count", "current_cross_mode_edge_count",
    "complete_same_surface_edge_count", "added_same_surface_edge_count",
    "retained_current_same_surface_edge_count", "missing_current_same_surface_edge_count",
    "current_same_surface_edge_ids_sha256", "current_cross_mode_edge_table_sha256",
    "complete_same_surface_edge_table_sha256", "added_same_surface_edge_ids_sha256",
    "effective_mode_edge_table_sha256", "all_current_same_surface_edges_reproduced",
    "current_same_surface_records_byte_identical", "cross_mode_edges_byte_identical",
    "all_added_edges_connect_distinct_parents", "all_added_edges_connect_26_neighbor_voxels",
    "no_neighbor_degree_truncation", "strict_binary64_comparisons",
    "edge_semantics_attestation_sha256",
}
FRAGMENT_EDGE_ATTRIBUTION_FIELDS = {
    "local_plane_fragment_id", "member_mode_ids_sha256",
    "accepted_same_surface_union_edge_count",
    "accepted_current_same_surface_union_edge_count",
    "accepted_added_same_surface_union_edge_count",
    "accepted_added_same_surface_union_edge_ids_sha256",
    "witness_accepted_added_same_surface_union_edge_ids", "witness_ids_truncated",
    "edge_semantics_dependent", "fragment_edge_attribution_sha256",
}
COMPONENT_EDGE_ATTRIBUTION_FIELDS = {
    "local_transition_component_id", "local_plane_pair_id", "fragment_ids",
    "component_added_same_surface_edge_count", "component_added_same_surface_edge_ids_sha256",
    "incident_fragment_added_union_edge_count", "incident_fragment_added_union_edge_ids_sha256",
    "causal_added_same_surface_edge_count", "causal_added_same_surface_edge_ids_sha256",
    "witness_causal_added_same_surface_edge_ids", "witness_ids_truncated",
    "edge_semantics_dependent", "geometry_certificate_valid",
    "component_edge_attribution_sha256",
}
TOP_LEVEL_FIELDS = {
    "schema_version", "kind", "profile", "research_only", "inference_safe",
    "prediction_mutated", "birth_columns_enabled", "ownership_consumed",
    "solver_consumed", "truth_inputs_consumed", "source_points_consumed",
    "source_points_serialized", "scan_sha256", "candidate_pool_sha256",
    "a0_prediction_sha256", "frozen_control_probe_sha256", "input_table_hashes",
    "replacement_edge_table_hashes", "output_table_hashes",
    "engineering_profile_sha256", "inference_semantic_sha256",
    "input_contract_attestation", "edge_semantics_attestation", "thresholds",
    "hard_caps", "case_counters", "complete_same_surface_edge_table",
    "same_surface_edge_delta_table", "fragment_edge_attribution_table",
    "component_edge_attribution_table", "local_plane_fragment_table",
    "local_plane_orphan_table", "local_plane_pair_records",
}

MAX_ADJACENT_PARENT_PAIRS = 52_000
MAX_COMPLETE_EDGE_EVALUATIONS = 832_000
MAX_COMPLETE_SAME_SURFACE_EDGES = 832_000
MAX_INHERITED_CROSS_MODE_EDGES = 24_000
MAX_EFFECTIVE_MODE_EDGES = 856_000
MAX_EDGE_ID_WITNESSES = 8
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
HARD_CAPS = {
    **dict(lpf.HARD_CAPS),
    "parent_plus_mode_record_cap": 20_000,
    "adjacent_parent_pair_cap": MAX_ADJACENT_PARENT_PAIRS,
    "complete_edge_evaluation_cap": MAX_COMPLETE_EDGE_EVALUATIONS,
    "complete_same_surface_edge_cap": MAX_COMPLETE_SAME_SURFACE_EDGES,
    "inherited_cross_mode_edge_cap": MAX_INHERITED_CROSS_MODE_EDGES,
    "effective_mode_edge_cap": MAX_EFFECTIVE_MODE_EDGES,
    "edge_id_witness_cap": MAX_EDGE_ID_WITNESSES,
}

FORBIDDEN_KEYS = frozenset(set(lpf.FORBIDDEN_KEYS) | {
    "cad_metadata", "ownership", "potential", "fitted_potential", "learned_score",
    "model_score", "emitted_births", "transition_strip_table", "support_strip",
    "corridor", "evidence_trace", "ridge_candidate", "junction_path",
})


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _table_sha(rows: Sequence[Mapping[str, Any]], id_field: str | None = None) -> str:
    values = [dict(row) for row in rows]
    if id_field:
        values.sort(key=lambda row: str(row[id_field]))
    return _sha(values)


def _record_hash(row: Mapping[str, Any], hash_field: str) -> str:
    return _sha({key: value for key, value in row.items() if key != hash_field})


def _require_sha(label: str, value: Any) -> str:
    return lpf._validate_sha256(label, value)


def _reject_forbidden(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"edge probe forbids {path}.{key}")
            _reject_forbidden(value, f"{path}.{key}")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            _reject_forbidden(value, f"{path}[{index}]")


def project_edge_probe_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS if field not in raw_pool]
    if missing:
        raise ValueError(f"edge-probe input projection missing fields: {missing}")
    projected = {field: deepcopy(raw_pool[field]) for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS}
    if tuple(projected) != ALLOWED_EDGE_PROBE_INPUT_FIELDS:
        raise AssertionError("edge-probe projection order mismatch")
    return projected


def _validate_projected_input(projected_input: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(projected_input) != ALLOWED_EDGE_PROBE_INPUT_FIELDS:
        if set(projected_input) != set(ALLOWED_EDGE_PROBE_INPUT_FIELDS):
            raise ValueError("edge core requires exact allowlisted input projection")
    _reject_forbidden(projected_input)
    validated = lpf._validate_inputs(projected_input)
    parents = validated["parents"]
    keys: set[tuple[int, int, int]] = set()
    levels: set[int] = set()
    sizes: set[float] = set()
    for parent in parents:
        raw_key = parent.get("voxel_key")
        if not isinstance(raw_key, Sequence) or len(raw_key) != 3:
            raise ValueError("parent voxel_key must be an integer three-vector")
        key = tuple(int(value) for value in raw_key)
        if any(float(raw) != float(value) for raw, value in zip(raw_key, key)):
            raise ValueError("parent voxel_key must be integral")
        if key in keys:
            raise ValueError("parent voxel keys must be unique")
        keys.add(key)
        levels.add(int(parent.get("voxel_level")))
        size = float(parent.get("voxel_size_mm"))
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("parent voxel size must be positive binary64")
        sizes.add(size)
    if len(levels) != 1:
        raise ValueError("parent voxel level is not case-global")
    if len(sizes) != 1:
        raise ValueError("parent voxel size is not case-global")
    validated = dict(validated)
    validated.update(parent_voxel_level=next(iter(levels)),
                     parent_voxel_size_mm=next(iter(sizes)), parent_voxel_keys=keys)
    return validated


def _neighbor_26(key: tuple[int, int, int]):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx or dy or dz:
                    yield (key[0] + dx, key[1] + dy, key[2] + dz)


def _build_complete(projected_input: Mapping[str, Any], validated: Mapping[str, Any] | None = None):
    data = dict(validated or _validate_projected_input(projected_input))
    parents = data["parent_by_id"]
    modes = data["mode_by_id"]
    key_to_parent = {
        tuple(int(v) for v in parent["voxel_key"]): parent_id
        for parent_id, parent in parents.items()
    }
    modes_by_parent: dict[str, list[str]] = defaultdict(list)
    for mode_id, mode in modes.items():
        modes_by_parent[str(mode["parent_atom_id"])].append(mode_id)
    spacing = float(data["spacing"])
    tau = min(3.0, max(1.25, 1.5 * spacing))
    voxel_size = float(data["parent_voxel_size_mm"])
    complete: dict[str, dict[str, Any]] = {}
    adjacent = evaluations = 0
    for left_parent_id in sorted(parents):
        left_key = tuple(int(v) for v in parents[left_parent_id]["voxel_key"])
        for right_key in sorted(_neighbor_26(left_key)):
            right_parent_id = key_to_parent.get(right_key)
            if right_parent_id is None or right_parent_id <= left_parent_id:
                continue
            adjacent += 1
            if adjacent > MAX_ADJACENT_PARENT_PAIRS:
                raise ValueError("adjacent parent pair cap breached")
            for left_mode_id in sorted(modes_by_parent[left_parent_id]):
                for right_mode_id in sorted(modes_by_parent[right_parent_id]):
                    evaluations += 1
                    if evaluations > MAX_COMPLETE_EDGE_EVALUATIONS:
                        raise ValueError("complete edge evaluation cap breached")
                    left_mode, right_mode = modes[left_mode_id], modes[right_mode_id]
                    angle = lpf.projective_angle_deg(
                        left_mode["normal_projective"], right_mode["normal_projective"]
                    )
                    if angle > 15.0:
                        continue
                    delta = lpf._vector(right_mode["centroid"], "mode centroid", allow_zero=True) - lpf._vector(
                        left_mode["centroid"], "mode centroid", allow_zero=True
                    )
                    left_normal = lpf._unit(left_mode["normal_projective"])
                    right_normal = lpf._unit(right_mode["normal_projective"])
                    gap = max(abs(float(np.dot(left_normal, delta))),
                              abs(float(np.dot(right_normal, delta))))
                    if gap > tau:
                        continue
                    pair = tuple(sorted((left_mode_id, right_mode_id)))
                    edge_id = "E_" + _sha([list(pair), "SAME_SURFACE"])[:16]
                    row = {
                        "mode_edge_id": edge_id, "mode_ids": list(pair),
                        "edge_kind": "SAME_SURFACE",
                        "spatial_residual": float(np.linalg.norm(delta)) / voxel_size,
                        "normal_residual": angle / 15.0,
                        "plane_residual": gap / tau,
                    }
                    previous = complete.get(edge_id)
                    if previous is not None and previous != row:
                        raise ValueError("same-surface edge-ID collision")
                    complete[edge_id] = row
                    if len(complete) > MAX_COMPLETE_SAME_SURFACE_EDGES:
                        raise ValueError("complete same-surface edge cap breached")
    return sorted(complete.values(), key=lambda row: row["mode_edge_id"]), adjacent, evaluations


def build_complete_same_surface_edges(projected_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _build_complete(projected_input)[0]


def _edge_tables(projected_input: Mapping[str, Any], validated: Mapping[str, Any]):
    complete, adjacent, evaluations = _build_complete(projected_input, validated)
    inherited_same = sorted((dict(e) for e in validated["edges"] if e["edge_kind"] == "SAME_SURFACE"),
                            key=lambda row: row["mode_edge_id"])
    inherited_cross = sorted((dict(e) for e in validated["edges"] if e["edge_kind"] == "CROSS_MODE_LOCAL"),
                             key=lambda row: row["mode_edge_id"])
    if len(inherited_cross) > MAX_INHERITED_CROSS_MODE_EDGES:
        raise ValueError("inherited CROSS_MODE_LOCAL edge cap breached")
    complete_by_id = {row["mode_edge_id"]: row for row in complete}
    missing = [row["mode_edge_id"] for row in inherited_same if row["mode_edge_id"] not in complete_by_id]
    differing = [row["mode_edge_id"] for row in inherited_same
                 if row["mode_edge_id"] in complete_by_id
                 and canonical_json_bytes(row) != canonical_json_bytes(complete_by_id[row["mode_edge_id"]])]
    if missing or differing:
        raise ValueError(f"inherited SAME_SURFACE subset proof failed: missing={missing[:8]} differing={differing[:8]}")
    current_ids = {row["mode_edge_id"] for row in inherited_same}
    added = [row for row in complete if row["mode_edge_id"] not in current_ids]
    effective = sorted([*complete, *inherited_cross], key=lambda row: row["mode_edge_id"])
    if len(effective) > MAX_EFFECTIVE_MODE_EDGES:
        raise ValueError("effective mode edge cap breached")
    signatures = [(tuple(row["mode_ids"]), row["edge_kind"]) for row in effective]
    ids = [row["mode_edge_id"] for row in effective]
    if len(ids) != len(set(ids)) or len(signatures) != len(set(signatures)):
        raise ValueError("effective edge table contains duplicate identity")
    return complete, inherited_same, inherited_cross, added, effective, adjacent, evaluations


def _input_attestation(projected_input: Mapping[str, Any], validated: Mapping[str, Any]):
    row = lpf._input_contract_attestation(projected_input, validated)
    row["projection_profile"] = INPUT_PROJECTION_PROFILE
    row["input_contract_attestation_sha256"] = _record_hash(row, "input_contract_attestation_sha256")
    return row


def _accepted_union_edges(modes: Mapping[str, Mapping[str, Any]], edges: Sequence[Mapping[str, Any]], tolerance: float):
    roots = {mode_id: mode_id for mode_id in modes}
    members = {mode_id: {mode_id} for mode_id in modes}
    diameters = {mode_id: 0.0 for mode_id in modes}
    versions = {mode_id: 0 for mode_id in modes}
    rejected: dict[tuple[str, int, str, int], str] = {}
    accepted: list[dict[str, Any]] = []
    def find(mode_id: str) -> str:
        while roots[mode_id] != mode_id:
            roots[mode_id] = roots[roots[mode_id]]
            mode_id = roots[mode_id]
        return mode_id
    ordered = sorted((e for e in edges if e["edge_kind"] == "SAME_SURFACE"), key=lambda e: (
        round(float(e["normal_residual"]), 12), round(float(e["plane_residual"]), 12),
        round(float(e["spatial_residual"]), 12), str(e["mode_edge_id"])))
    for edge in ordered:
        left, right = map(str, edge["mode_ids"])
        a, b = find(left), find(right)
        if a == b:
            continue
        if a > b: a, b = b, a
        key = (a, versions[a], b, versions[b])
        if key in rejected:
            continue
        proposed = sorted(members[a] | members[b])
        cross_diameter = max(lpf.projective_angle_deg(modes[x]["normal_projective"], modes[y]["normal_projective"])
                             for x in members[a] for y in members[b])
        diameter = max(diameters[a], diameters[b], cross_diameter)
        if diameter > lpf.PLANE_NORMAL_DIAMETER_DEG + 1e-12:
            rejected[key] = "normal"; continue
        fit = lpf._fit_fragment(proposed, modes, known_normal_diameter_deg=diameter)
        if fit["surface_fit_p90_mm"] > tolerance + 1e-12:
            rejected[key] = "fit"; continue
        canonical, absorbed = sorted((a, b))
        roots[absorbed] = canonical
        members[canonical] = set(proposed)
        diameters[canonical] = diameter
        versions[canonical] = max(versions[a], versions[b]) + 1
        del members[absorbed]; diameters.pop(absorbed); versions.pop(absorbed)
        accepted.append(dict(edge))
    return accepted


def _profile_fragments(parents, modes, edges, planes, tolerance, counters):
    fragments, orphans, fragment_of_mode, orphan_modes = lpf._build_fragments(
        parents=parents, modes=modes, edges=edges, planes=planes,
        plane_fit_tolerance_mm=tolerance, counters=counters)
    remap: dict[str, str] = {}
    result = []
    for fragment in fragments:
        new_id = "VEF_" + _sha([PRIMITIVE_PROFILE, list(fragment.mode_ids)])[:16]
        old_id = fragment.record["local_plane_fragment_id"]
        remap[old_id] = new_id
        record = dict(fragment.record)
        record["local_plane_fragment_id"] = new_id
        record["fragment_sha256"] = _record_hash(record, "fragment_sha256")
        result.append(replace(fragment, record=record))
    for mode_id, old_id in list(fragment_of_mode.items()):
        fragment_of_mode[mode_id] = remap[old_id]
    profile_orphans = []
    for orphan in orphans:
        row = dict(orphan)
        row["local_plane_orphan_id"] = "VEO_" + _sha([
            PRIMITIVE_PROFILE, lpf.ORPHAN_RULE_ID, row["mode_atom_id"], row["parent_atom_id"]])[:16]
        row["orphan_sha256"] = _record_hash(row, "orphan_sha256")
        profile_orphans.append(row)
    result.sort(key=lambda f: f.record["local_plane_fragment_id"])
    profile_orphans.sort(key=lambda r: r["local_plane_orphan_id"])
    return result, profile_orphans, fragment_of_mode, orphan_modes


def _build_pairs(*, parents, modes, edges, planes, fragments, fragment_of_mode,
                 line_tolerance, bin_width, counters, added_ids, fragment_added_ids,
                 frozen_two_sided_one_parent_mode_sets=()):
    fragment_by_id = {f.record["local_plane_fragment_id"]: f for f in fragments}
    cross = sorted((e for e in edges if e["edge_kind"] == "CROSS_MODE_LOCAL"), key=lambda e: e["mode_edge_id"])
    pair_seeds: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in cross:
        a, b = map(str, edge["mode_ids"])
        if a not in fragment_of_mode or b not in fragment_of_mode:
            counters["excluded_cross_mode_seed_edge_count_due_to_orphan"] += 1; continue
        fa, fb = fragment_of_mode[a], fragment_of_mode[b]
        if fa != fb: pair_seeds[tuple(sorted((fa, fb)))].append(edge)
    if len(pair_seeds) > lpf.MAX_LOCAL_PLANE_PAIRS: raise ValueError("local plane pair cap breached")
    planes_by_id = {lpf._canonical_plane(p)[2]: p for p in planes}
    pair_records = []
    attributions = []
    total_components = 0
    first_rejections: Counter[str] = Counter()
    funnel = [name for name, _ in lpf.REJECTION_ORDER]
    for pair_ids in sorted(pair_seeds):
        left, right = (fragment_by_id[i] for i in pair_ids)
        pair_id = "VEP_" + _sha([list(pair_ids)])[:16]
        pair_rejections: Counter[str] = Counter()
        global_comparison, global_pair_ids = lpf._global_pair_comparison(left, right)
        component_records = []
        angle = lpf.projective_angle_deg(left.normal, right.normal)
        transverse: set[str] = set(); left_modes: set[str] = set(); right_modes: set[str] = set()
        retained_cross = []; anchor = np.zeros(3); axis = np.asarray((1., 0., 0.))
        try:
            anchor, axis, angle = lpf._plane_intersection(left.normal, left.offset, right.normal, right.offset)
        except (ValueError, np.linalg.LinAlgError):
            pair_rejections["PLANE_PAIR_ANGLE_INELIGIBLE"] += 1
        else:
            counters["angle_eligible_local_pair_count"] += 1
            transverse = {pid for pid,p in parents.items() if lpf._line_distance(
                lpf._vector(p["centroid"], "parent centroid", allow_zero=True), anchor, axis) <= line_tolerance + 1e-12}
            left_modes = {m for m in left.mode_ids if str(modes[m]["parent_atom_id"]) in transverse}
            right_modes = {m for m in right.mode_ids if str(modes[m]["parent_atom_id"]) in transverse}
            if not transverse: pair_rejections["NO_TRANSVERSE_PARENT"] += 1; counters["pair_no_transverse_parent_count"] += 1
            if not left_modes: pair_rejections["MISSING_LEFT_MODE"] += 1; counters["pair_missing_left_mode_count"] += 1
            if not right_modes: pair_rejections["MISSING_RIGHT_MODE"] += 1; counters["pair_missing_right_mode_count"] += 1
            supported = left_modes | right_modes
            restricted = []
            for edge in edges:
                em = tuple(map(str, edge["mode_ids"]))
                if not set(em) <= supported: continue
                if edge["edge_kind"] == "SAME_SURFACE":
                    if fragment_of_mode[em[0]] != fragment_of_mode[em[1]] or fragment_of_mode[em[0]] not in pair_ids: continue
                elif edge["edge_kind"] == "CROSS_MODE_LOCAL":
                    if {fragment_of_mode[em[0]], fragment_of_mode[em[1]]} != set(pair_ids): continue
                    retained_cross.append(edge)
                else: continue
                restricted.append(edge)
            if not retained_cross: pair_rejections["NO_CROSS_MODE_RUNG"] += 1; counters["pair_no_cross_mode_rung_count"] += 1
            components = lpf._connected_components(supported, restricted)
            if len(components) > lpf.MAX_COMPONENTS_PER_LOCAL_PAIR: raise ValueError("components per local pair cap breached")
            total_components += len(components)
            if total_components > lpf.MAX_LOCAL_TRANSITION_COMPONENTS: raise ValueError("component cap breached")
            for component_modes in components:
                mode_set = set(component_modes)
                component_edges = [e for e in restricted if set(map(str,e["mode_ids"])) <= mode_set]
                record = lpf._build_component_record(
                    pair_id=pair_id, mode_ids=component_modes, left=left, right=right,
                    anchor=anchor, axis=axis, restricted_edges=component_edges,
                    parents=parents, modes=modes, fragment_of_mode=fragment_of_mode,
                    bin_width=bin_width, tube_tolerance=line_tolerance,
                    global_pair_ids=global_pair_ids, planes_by_id=planes_by_id)
                parent_ids = tuple(sorted({str(modes[m]["parent_atom_id"]) for m in component_modes}))
                component_id = "VEC_" + _sha([pair_id, list(parent_ids), list(component_modes)])[:16]
                record["local_transition_component_id"] = component_id
                record["component_sha256"] = _record_hash(record, "component_sha256")
                component_records.append(record)
                counters["local_transition_component_count"] += 1
                if len(parent_ids) == 1:
                    endpoints = record["intersection_span_endpoints"]
                    if endpoints[0] != endpoints[1]: counters["one_parent_finite_span_count"] += 1
                    else: counters["one_parent_zero_span_count"] += 1
                if record["geometry_certificate_valid"]: counters["geometry_certificate_count"] += 1
                if record["first_rejection_code"]: first_rejections[str(record["first_rejection_code"])] += 1
                for index, gate in enumerate(funnel):
                    if all(record["geometry_gates"][x] for x in funnel[:index+1]): counters[f"components_after_{gate}"] += 1
                contained = sorted(str(e["mode_edge_id"]) for e in component_edges if str(e["mode_edge_id"]) in added_ids)
                incident = sorted(fragment_added_ids.get(pair_ids[0], set()) | fragment_added_ids.get(pair_ids[1], set()))
                causal = sorted(set(contained) | set(incident))
                if (
                    len(parent_ids) >= 2
                    and causal
                    and any(set(target) <= mode_set for target in frozen_two_sided_one_parent_mode_sets)
                ):
                    counters["two_sided_one_parent_support_extended_count"] += 1
                attribution = {
                    "local_transition_component_id": component_id, "local_plane_pair_id": pair_id,
                    "fragment_ids": list(pair_ids), "component_added_same_surface_edge_count": len(contained),
                    "component_added_same_surface_edge_ids_sha256": _sha(contained),
                    "incident_fragment_added_union_edge_count": len(incident),
                    "incident_fragment_added_union_edge_ids_sha256": _sha(incident),
                    "causal_added_same_surface_edge_count": len(causal),
                    "causal_added_same_surface_edge_ids_sha256": _sha(causal),
                    "witness_causal_added_same_surface_edge_ids": causal[:MAX_EDGE_ID_WITNESSES],
                    "witness_ids_truncated": len(causal) > MAX_EDGE_ID_WITNESSES,
                    "edge_semantics_dependent": bool(causal),
                    "geometry_certificate_valid": bool(record["geometry_certificate_valid"]),
                }
                attribution["component_edge_attribution_sha256"] = _record_hash(attribution,"component_edge_attribution_sha256")
                attributions.append(attribution)
        pair = {
            "local_plane_pair_id": pair_id, "fragment_ids": list(pair_ids),
            "plane_angle_deg": round(float(angle),8),
            "intersection_line_hash": _sha([lpf._round_vector(anchor,6),lpf._round_vector(axis,12)]),
            "intersection_anchor": lpf._round_vector(anchor,6), "intersection_axis": lpf._round_vector(axis,12),
            "transverse_parent_count": len(transverse), "left_mode_count": len(left_modes),
            "right_mode_count": len(right_modes), "cross_mode_rung_count": len(retained_cross),
            "component_count": len(component_records), "global_pair_comparison": global_comparison,
            "pair_rejection_counts": dict(sorted(pair_rejections.items())),
            "components": sorted(component_records,key=lambda r:r["local_transition_component_id"]),
        }
        pair["pair_sha256"] = _record_hash(pair,"pair_sha256")
        pair_records.append(pair)
    counters["first_rejection_counts"] = dict(sorted(first_rejections.items()))
    pair_records.sort(key=lambda r:r["local_plane_pair_id"])
    attributions.sort(key=lambda r:r["local_transition_component_id"])
    return pair_records, attributions


def build_v2_complete_same_surface_edge_probe(
    projected_input: Mapping[str, Any], *, scan_sha256: str,
    candidate_pool_sha256: str, a0_prediction_sha256: str,
    frozen_control_probe_sha256: str,
) -> dict[str, Any]:
    scan_hash = _require_sha("scan_sha256", scan_sha256)
    pool_hash = _require_sha("candidate_pool_sha256", candidate_pool_sha256)
    a0_hash = _require_sha("a0_prediction_sha256", a0_prediction_sha256)
    control_hash = _require_sha("frozen_control_probe_sha256", frozen_control_probe_sha256)
    validated = _validate_projected_input(projected_input)
    # Regenerate, never consume, the frozen control.  Besides proving the exact
    # control hash, this supplies the preregistered 4,788-style intervention
    # population without opening gold or adding any inference input.
    frozen_control = regenerate_frozen_control_probe(
        projected_input, scan_sha256=scan_hash,
        candidate_pool_sha256=pool_hash, a0_prediction_sha256=a0_hash,
    )
    if _sha(frozen_control) != control_hash:
        raise ValueError("regenerated frozen LPF control hash mismatch")
    frozen_two_sided_one_parent_mode_sets = []
    for pair in frozen_control["local_plane_pair_records"]:
        for component in pair["components"]:
            if (
                int(component["support_parent_count"]) == 1
                and int(component["support_mode_count"]) > 1
                and int(component["left_side_bin_count"]) > 0
                and int(component["right_side_bin_count"]) > 0
                and int(component["transition_rung_bin_count"]) > 0
                and component["intersection_span_endpoints"][0]
                    == component["intersection_span_endpoints"][1]
            ):
                # At one parent the inherited cap guarantees at most four modes,
                # so the frozen witness is the complete, untruncated mode set.
                if component["witness_ids_truncated"]:
                    raise ValueError("one-parent frozen component witness truncated")
                frozen_two_sided_one_parent_mode_sets.append(
                    tuple(sorted(map(str, component["witness_mode_ids"])))
                )
    complete, current_same, current_cross, added, effective, adjacent, evaluations = _edge_tables(projected_input, validated)
    parents, modes, planes = validated["parent_by_id"], validated["mode_by_id"], validated["planes"]
    spacing = float(validated["spacing"])
    plane_tol = min(3.0,max(1.25,1.5*spacing)); line_tol=min(3.0,max(1.5,1.25*spacing)); bin_width=max(3.0,2.0*spacing)
    thresholds = {
        "same_surface_angle_deg":15.0, "surface_gap_tolerance_mm":plane_tol,
        "plane_normal_diameter_deg":lpf.PLANE_NORMAL_DIAMETER_DEG,
        "plane_fit_tolerance_mm":plane_tol, "plane_pair_min_angle_deg":lpf.PLANE_PAIR_MIN_ANGLE_DEG,
        "plane_pair_max_projective_angle_deg":lpf.PLANE_PAIR_MAX_PROJECTIVE_ANGLE_DEG,
        "line_tube_tolerance_mm":line_tol, "axial_bin_width_mm":bin_width,
        "minimum_parent_count":lpf.MINIMUM_PARENT_COUNT, "minimum_axial_span_bins":lpf.MINIMUM_AXIAL_SPAN_BINS,
        "minimum_occupied_bins":lpf.MINIMUM_OCCUPIED_BINS, "minimum_side_bins":lpf.MINIMUM_SIDE_BINS,
        "minimum_transition_rung_bins":lpf.MINIMUM_TRANSITION_RUNG_BINS,
        "minimum_consecutive_run":lpf.MINIMUM_CONSECUTIVE_RUN,
        "maximum_empty_bin_run":lpf.MAXIMUM_EMPTY_BIN_RUN,
        "maximum_internal_branch_count":lpf.MAXIMUM_INTERNAL_BRANCH_COUNT,
        "maximum_tls_axis_angle_deg":lpf.MAXIMUM_TLS_AXIS_ANGLE_DEG,
        "global_line_axis_angle_deg":lpf.GLOBAL_LINE_AXIS_ANGLE_DEG,
        "global_line_transverse_offset_mm":lpf.GLOBAL_LINE_TRANSVERSE_OFFSET_MM,
    }
    counters=Counter({"input_parent_count":len(parents),"input_mode_count":len(modes),
        "input_same_surface_edge_count":len(current_same),"input_cross_mode_edge_count":len(current_cross),
        "input_global_plane_count":len(planes),"inherited_working_point_count":int(validated["working_count"]),
        "complete_same_surface_edge_count":len(complete),"added_same_surface_edge_count":len(added),
        "effective_mode_edge_count":len(effective),"adjacent_parent_pair_count":adjacent,
        "eligible_mode_pair_evaluation_count":evaluations,"one_parent_finite_span_count":0,
        "one_parent_zero_span_count":0,"two_sided_one_parent_support_extended_count":0})
    fragments, orphans, fragment_of_mode, orphan_modes = _profile_fragments(parents,modes,effective,planes,plane_tol,counters)
    counters["local_plane_fragment_count"] = len(fragments)
    counters["local_plane_orphan_count"] = len(orphans)
    counters["local_plane_pair_count"] = 0
    counters["valid_fragment_mode_count"] = sum(int(f.record["mode_count"]) for f in fragments)
    counters["orphan_mode_count"] = len(orphan_modes)
    accepted = _accepted_union_edges(modes,effective,plane_tol)
    added_ids={r["mode_edge_id"] for r in added}; current_ids={r["mode_edge_id"] for r in current_same}
    accepted_by_fragment: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for edge in accepted:
        fid=fragment_of_mode.get(str(edge["mode_ids"][0]))
        if fid: accepted_by_fragment[fid].append(edge)
    fragment_attribution=[]; fragment_added_ids:dict[str,set[str]]={}
    for fragment in fragments:
        fid=fragment.record["local_plane_fragment_id"]; rows=accepted_by_fragment.get(fid,[])
        added_union=sorted(r["mode_edge_id"] for r in rows if r["mode_edge_id"] in added_ids)
        current_union=sum(r["mode_edge_id"] in current_ids for r in rows)
        fragment_added_ids[fid]=set(added_union)
        row={"local_plane_fragment_id":fid,"member_mode_ids_sha256":fragment.record["member_mode_ids_sha256"],
             "accepted_same_surface_union_edge_count":len(rows),
             "accepted_current_same_surface_union_edge_count":current_union,
             "accepted_added_same_surface_union_edge_count":len(added_union),
             "accepted_added_same_surface_union_edge_ids_sha256":_sha(added_union),
             "witness_accepted_added_same_surface_union_edge_ids":added_union[:MAX_EDGE_ID_WITNESSES],
             "witness_ids_truncated":len(added_union)>MAX_EDGE_ID_WITNESSES,
             "edge_semantics_dependent":bool(added_union)}
        row["fragment_edge_attribution_sha256"]=_record_hash(row,"fragment_edge_attribution_sha256")
        fragment_attribution.append(row)
    pair_records, component_attribution = _build_pairs(parents=parents,modes=modes,edges=effective,planes=planes,
        fragments=fragments,fragment_of_mode=fragment_of_mode,line_tolerance=line_tol,bin_width=bin_width,
        counters=counters,added_ids=added_ids,fragment_added_ids=fragment_added_ids,
        frozen_two_sided_one_parent_mode_sets=frozen_two_sided_one_parent_mode_sets)
    counters["local_plane_pair_count"] = len(pair_records)
    for counter_name in (
        "local_transition_component_count", "geometry_certificate_count",
        "angle_eligible_local_pair_count", "pair_no_transverse_parent_count",
        "pair_missing_left_mode_count", "pair_missing_right_mode_count",
        "pair_no_cross_mode_rung_count",
        "excluded_cross_mode_seed_edge_count_due_to_orphan",
    ):
        counters.setdefault(counter_name, 0)
    fragment_records=[f.record for f in fragments]
    delta=[]
    for edge in added:
        mode_ids=list(edge["mode_ids"]); parent_ids=[str(modes[m]["parent_atom_id"]) for m in mode_ids]
        row={"mode_edge_id":edge["mode_edge_id"],"mode_ids":mode_ids,"parent_atom_ids":parent_ids,
             "parent_voxel_keys":[list(parents[p]["voxel_key"]) for p in parent_ids],
             "spatial_residual":edge["spatial_residual"],"normal_residual":edge["normal_residual"],
             "plane_residual":edge["plane_residual"],"delta_kind":"ADDED_BY_COMPLETE_ELIGIBILITY"}
        row["edge_delta_sha256"]=_record_hash(row,"edge_delta_sha256"); delta.append(row)
    attestation={"edge_rule_id":EDGE_RULE_ID,"control_edge_rule_id":CONTROL_EDGE_RULE_ID,
        "same_surface_angle_deg":15.0,"surface_gap_tolerance_mm":plane_tol,
        "parent_voxel_level":validated["parent_voxel_level"],"parent_voxel_size_mm":validated["parent_voxel_size_mm"],
        "adjacent_parent_pair_count":adjacent,"eligible_mode_pair_evaluation_count":evaluations,
        "current_same_surface_edge_count":len(current_same),"current_cross_mode_edge_count":len(current_cross),
        "complete_same_surface_edge_count":len(complete),"added_same_surface_edge_count":len(added),
        "retained_current_same_surface_edge_count":len(current_same),"missing_current_same_surface_edge_count":0,
        "current_same_surface_edge_ids_sha256":_sha(sorted(current_ids)),
        "current_cross_mode_edge_table_sha256":_table_sha(current_cross,"mode_edge_id"),
        "complete_same_surface_edge_table_sha256":_table_sha(complete,"mode_edge_id"),
        "added_same_surface_edge_ids_sha256":_sha(sorted(added_ids)),
        "effective_mode_edge_table_sha256":_table_sha(effective,"mode_edge_id"),
        "all_current_same_surface_edges_reproduced":True,"current_same_surface_records_byte_identical":True,
        "cross_mode_edges_byte_identical":True,"all_added_edges_connect_distinct_parents":True,
        "all_added_edges_connect_26_neighbor_voxels":True,"no_neighbor_degree_truncation":True,
        "strict_binary64_comparisons":True}
    attestation["edge_semantics_attestation_sha256"]=_record_hash(attestation,"edge_semantics_attestation_sha256")
    input_attestation=_input_attestation(projected_input,validated)
    edge_hashes={"current_same_surface_edge_table_sha256":_table_sha(current_same,"mode_edge_id"),
        "current_cross_mode_edge_table_sha256":_table_sha(current_cross,"mode_edge_id"),
        "complete_same_surface_edge_table_sha256":_table_sha(complete,"mode_edge_id"),
        "added_same_surface_edge_table_sha256":_table_sha(added,"mode_edge_id"),
        "effective_mode_edge_table_sha256":_table_sha(effective,"mode_edge_id")}
    output_hashes={"complete_same_surface_edge_table_sha256":_table_sha(complete,"mode_edge_id"),
        "same_surface_edge_delta_table_sha256":_table_sha(delta,"mode_edge_id"),
        "fragment_edge_attribution_table_sha256":_table_sha(fragment_attribution,"local_plane_fragment_id"),
        "component_edge_attribution_table_sha256":_table_sha(component_attribution,"local_transition_component_id"),
        "local_plane_fragment_table_sha256":_table_sha(fragment_records,"local_plane_fragment_id"),
        "local_plane_orphan_table_sha256":_table_sha(orphans,"local_plane_orphan_id"),
        "local_plane_pair_records_sha256":_table_sha(pair_records,"local_plane_pair_id")}
    descriptor={"profile":PRIMITIVE_PROFILE,"schema_version":PROBE_SCHEMA_VERSION,"edge_rule_id":EDGE_RULE_ID,
        "control_edge_rule_id":CONTROL_EDGE_RULE_ID,"input_projection_profile":INPUT_PROJECTION_PROFILE,
        "downstream_geometry_profile":lpf.LOCAL_PLANE_FRAGMENT_PROFILE,"hard_caps":HARD_CAPS}
    engineering_hash=_sha({"descriptor":descriptor,"case_thresholds":thresholds})
    semantic_hash=_sha({"profile":PRIMITIVE_PROFILE,"engineering_profile_sha256":engineering_hash,
        "projected_input_sha256":input_attestation["projected_input_sha256"],
        "frozen_control_probe_sha256":control_hash,
        "edge_semantics_attestation_sha256":attestation["edge_semantics_attestation_sha256"],
        "output_table_hashes":output_hashes})
    payload={"schema_version":PROBE_SCHEMA_VERSION,"kind":PROBE_KIND,"profile":PRIMITIVE_PROFILE,
        "research_only":True,"inference_safe":True,"prediction_mutated":False,"birth_columns_enabled":False,
        "ownership_consumed":False,"solver_consumed":False,"truth_inputs_consumed":False,
        "source_points_consumed":False,"source_points_serialized":False,"scan_sha256":scan_hash,
        "candidate_pool_sha256":pool_hash,"a0_prediction_sha256":a0_hash,"frozen_control_probe_sha256":control_hash,
        "input_table_hashes":dict(input_attestation["table_hashes"]),"replacement_edge_table_hashes":edge_hashes,
        "output_table_hashes":output_hashes,"engineering_profile_sha256":engineering_hash,
        "inference_semantic_sha256":semantic_hash,"input_contract_attestation":input_attestation,
        "edge_semantics_attestation":attestation,"thresholds":thresholds,"hard_caps":dict(HARD_CAPS),
        "case_counters":dict(sorted(counters.items())),"complete_same_surface_edge_table":complete,
        "same_surface_edge_delta_table":delta,"fragment_edge_attribution_table":fragment_attribution,
        "component_edge_attribution_table":component_attribution,"local_plane_fragment_table":fragment_records,
        "local_plane_orphan_table":orphans,"local_plane_pair_records":pair_records}
    validate_v2_complete_same_surface_edge_probe_payload(payload, projected_input=projected_input)
    return payload


def validate_v2_complete_same_surface_edge_probe_payload(payload: Mapping[str, Any], *, projected_input: Mapping[str, Any] | None = None) -> None:
    _reject_forbidden(payload)
    if set(payload)!=TOP_LEVEL_FIELDS: raise ValueError(f"edge probe top-level schema mismatch: {sorted(set(payload)^TOP_LEVEL_FIELDS)}")
    if payload["schema_version"]!=PROBE_SCHEMA_VERSION or payload["kind"]!=PROBE_KIND or payload["profile"]!=PRIMITIVE_PROFILE:
        raise ValueError("edge probe identity invalid")
    for field,expected in {"research_only":True,"inference_safe":True,"prediction_mutated":False,
        "birth_columns_enabled":False,"ownership_consumed":False,"solver_consumed":False,
        "truth_inputs_consumed":False,"source_points_consumed":False,"source_points_serialized":False}.items():
        if payload[field] is not expected: raise ValueError(f"edge probe flag invalid: {field}")
    for field in ("scan_sha256","candidate_pool_sha256","a0_prediction_sha256","frozen_control_probe_sha256","engineering_profile_sha256","inference_semantic_sha256"):
        _require_sha(field,payload[field])
    if dict(payload["hard_caps"])!=HARD_CAPS: raise ValueError("edge probe hard caps mismatch")
    tables=(("complete_same_surface_edge_table",COMPLETE_EDGE_FIELDS,"mode_edge_id"),
            ("same_surface_edge_delta_table",EDGE_DELTA_FIELDS,"mode_edge_id"),
            ("fragment_edge_attribution_table",FRAGMENT_EDGE_ATTRIBUTION_FIELDS,"local_plane_fragment_id"),
            ("component_edge_attribution_table",COMPONENT_EDGE_ATTRIBUTION_FIELDS,"local_transition_component_id"),
            ("local_plane_fragment_table",lpf.FRAGMENT_RECORD_FIELDS,"local_plane_fragment_id"),
            ("local_plane_orphan_table",lpf.ORPHAN_RECORD_FIELDS,"local_plane_orphan_id"),
            ("local_plane_pair_records",lpf.PAIR_RECORD_FIELDS,"local_plane_pair_id"))
    hash_names={name:("complete_same_surface_edge_table_sha256" if name=="complete_same_surface_edge_table" else name+"_sha256") for name,_,_ in tables}
    for name,fields,id_field in tables:
        rows=list(payload[name]); ids=[]
        for row in rows:
            if set(row)!=fields: raise ValueError(f"{name} record schema mismatch")
            ids.append(str(row[id_field]))
        if ids!=sorted(ids) or len(ids)!=len(set(ids)): raise ValueError(f"{name} ordering/identity invalid")
        if payload["output_table_hashes"][hash_names[name]]!=_table_sha(rows,id_field): raise ValueError(f"{name} hash mismatch")
    for row in payload["same_surface_edge_delta_table"]:
        if row["edge_delta_sha256"]!=_record_hash(row,"edge_delta_sha256"): raise ValueError("edge delta record hash mismatch")
    for name,hfield in (("fragment_edge_attribution_table","fragment_edge_attribution_sha256"),("component_edge_attribution_table","component_edge_attribution_sha256")):
        for row in payload[name]:
            if row[hfield]!=_record_hash(row,hfield): raise ValueError(f"{name} record hash mismatch")
    att=dict(payload["edge_semantics_attestation"])
    if set(att)!=EDGE_SEMANTICS_ATTESTATION_FIELDS or att["edge_semantics_attestation_sha256"]!=_record_hash(att,"edge_semantics_attestation_sha256"):
        raise ValueError("edge semantics attestation invalid")
    if int(payload["case_counters"].get("one_parent_finite_span_count",-1))!=0: raise ValueError("one-parent finite span invariant violated")
    for pair in payload["local_plane_pair_records"]:
        for component in pair["components"]:
            if int(component["support_parent_count"])==1 and component["intersection_span_endpoints"][0]!=component["intersection_span_endpoints"][1]:
                raise ValueError("one-parent component has finite serialized span")
    expected_semantic=_sha({"profile":PRIMITIVE_PROFILE,"engineering_profile_sha256":payload["engineering_profile_sha256"],
        "projected_input_sha256":payload["input_contract_attestation"]["projected_input_sha256"],
        "frozen_control_probe_sha256":payload["frozen_control_probe_sha256"],
        "edge_semantics_attestation_sha256":att["edge_semantics_attestation_sha256"],
        "output_table_hashes":payload["output_table_hashes"]})
    if payload["inference_semantic_sha256"]!=expected_semantic: raise ValueError("semantic hash mismatch")
    if projected_input is not None:
        validated=_validate_projected_input(projected_input)
        if payload["input_contract_attestation"]!=_input_attestation(projected_input,validated): raise ValueError("input attestation mismatch")
        complete,current_same,current_cross,added,effective,_,_=_edge_tables(projected_input,validated)
        if payload["complete_same_surface_edge_table"]!=complete: raise ValueError("complete edge rebuild mismatch")
        if payload["replacement_edge_table_hashes"]["effective_mode_edge_table_sha256"]!=_table_sha(effective,"mode_edge_id"):
            raise ValueError("effective edge hash mismatch")


def regenerate_frozen_control_probe(projected_input: Mapping[str, Any], *, scan_sha256: str, candidate_pool_sha256: str, a0_prediction_sha256: str) -> dict[str, Any]:
    return lpf.build_local_plane_fragment_probe(projected_input,scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,a0_prediction_sha256=a0_prediction_sha256)


def verify_edge_deletion_counterfactual(projected_input: Mapping[str, Any], replacement_payload: Mapping[str, Any], *,
        scan_sha256: str, candidate_pool_sha256: str, a0_prediction_sha256: str,
        expected_control_probe_sha256: str) -> dict[str, Any]:
    validate_v2_complete_same_surface_edge_probe_payload(replacement_payload,projected_input=projected_input)
    validated=_validate_projected_input(projected_input)
    complete,current_same,current_cross,added,effective,_,_=_edge_tables(projected_input,validated)
    added_ids={row["mode_edge_id"] for row in added}
    deleted=[row for row in effective if row["mode_edge_id"] not in added_ids]
    inherited=sorted((dict(row) for row in projected_input["support_mode_edges"]),key=lambda row:row["mode_edge_id"])
    if canonical_json_bytes(deleted)!=canonical_json_bytes(inherited): raise ValueError("edge deletion did not restore inherited table")
    control=regenerate_frozen_control_probe(projected_input,scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,a0_prediction_sha256=a0_prediction_sha256)
    actual=_sha(control); expected=_require_sha("expected_control_probe_sha256",expected_control_probe_sha256)
    if actual!=expected or replacement_payload["frozen_control_probe_sha256"]!=expected:
        raise ValueError("frozen control reproduction mismatch")
    return {"counterfactual_profile":COUNTERFACTUAL_PROFILE,"restored_inherited_edge_table_sha256":_table_sha(deleted,"mode_edge_id"),
            "frozen_control_probe_sha256":actual,"added_edge_count_deleted":len(added_ids),"valid":True}


__all__ = [
    "PRIMITIVE_PROFILE", "PROBE_SCHEMA_VERSION", "PROBE_KIND", "INPUT_PROJECTION_PROFILE",
    "EDGE_RULE_ID", "CONTROL_EDGE_RULE_ID", "COUNTERFACTUAL_PROFILE",
    "ALLOWED_EDGE_PROBE_INPUT_FIELDS", "COMPLETE_EDGE_FIELDS", "EDGE_DELTA_FIELDS",
    "EDGE_SEMANTICS_ATTESTATION_FIELDS", "FRAGMENT_EDGE_ATTRIBUTION_FIELDS",
    "COMPONENT_EDGE_ATTRIBUTION_FIELDS", "TOP_LEVEL_FIELDS", "HARD_CAPS",
    "canonical_json_bytes", "project_edge_probe_input", "build_complete_same_surface_edges",
    "build_v2_complete_same_surface_edge_probe", "validate_v2_complete_same_surface_edge_probe_payload",
    "regenerate_frozen_control_probe", "verify_edge_deletion_counterfactual",
]
