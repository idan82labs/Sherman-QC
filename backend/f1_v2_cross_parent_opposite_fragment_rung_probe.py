"""Gold-free V2 cross-parent opposite-fragment rung research probe.

The module consumes only the frozen thirteen-field TRANSITION_ATOM_V2
projection.  It independently regenerates both frozen LPF V2 and the complete
SAME_SURFACE baseline, freezes their fragments and plane pairs, and changes
only fixed-pair cross-side component connectivity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as baseline_core


PRIMITIVE_PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1"
PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_v1"
BASELINE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
RUNG_RULE_ID = "COMPLETE_ADJACENT_PARENT_OPPOSITE_FRAGMENT_NO_TOPK_V1"
CONTROL_RUNG_RULE_ID = "INHERITED_SAME_PARENT_CROSS_MODE_LOCAL_ONLY_V1"
COUNTERFACTUAL_PROFILE = "DELETE_ADDED_CROSS_PARENT_RUNGS_RESTORES_COMPLETE_EDGE_V1"
INPUT_PROJECTION_PROFILE = "V2_CROSS_PARENT_RUNG_INPUT_PROJECTION_V1"
ALLOWED_INPUT_FIELDS = tuple(baseline_core.ALLOWED_EDGE_PROBE_INPUT_FIELDS)

MAX_ADJACENT_PARENT_PAIRS = 52_000
MAX_CROSS_PARENT_RUNG_EVALUATIONS = 832_000
MAX_ADDED_CROSS_PARENT_RUNGS = 832_000
MAX_BASELINE_COMPLETE_SAME_SURFACE_EDGES = 832_000
MAX_INHERITED_CROSS_MODE_LOCAL_EDGES = 24_000
MAX_EFFECTIVE_MODE_EDGES = 1_688_000
MAX_FROZEN_TARGET_ROWS = 4_788
MAX_EDGE_ID_WITNESSES = 8
MAX_COMPONENT_ID_WITNESSES = 8
MAX_TARGET_ID_WITNESSES = 8
MAX_BATCH_WORKERS = 8
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3

HARD_CAPS = {
    **dict(baseline_core.HARD_CAPS),
    "cross_parent_rung_evaluation_cap": MAX_CROSS_PARENT_RUNG_EVALUATIONS,
    "added_cross_parent_rung_cap": MAX_ADDED_CROSS_PARENT_RUNGS,
    "baseline_complete_same_surface_edge_cap": MAX_BASELINE_COMPLETE_SAME_SURFACE_EDGES,
    "inherited_cross_mode_local_edge_cap": MAX_INHERITED_CROSS_MODE_LOCAL_EDGES,
    "effective_mode_edge_cap": MAX_EFFECTIVE_MODE_EDGES,
    "frozen_target_row_cap": MAX_FROZEN_TARGET_ROWS,
    "component_id_witness_cap": MAX_COMPONENT_ID_WITNESSES,
    "target_id_witness_cap": MAX_TARGET_ID_WITNESSES,
    "batch_worker_cap": MAX_BATCH_WORKERS,
}

FROZEN_TARGET_FIELDS = {
    "case_token", "frozen_target_component_id", "frozen_target_pair_id",
    "frozen_target_parent_id", "frozen_target_mode_ids",
    "frozen_target_mode_ids_sha256", "frozen_target_sha256",
}
CROSS_PARENT_RUNG_FIELDS = {
    "mode_edge_id", "baseline_local_plane_pair_id", "baseline_fragment_ids",
    "mode_ids", "parent_atom_ids", "parent_voxel_keys", "edge_kind",
    "parent_voxel_chebyshev_distance", "parent_centroid_line_distances_mm",
    "axial_parent_separation_mm", "axial_midpoint_projection_mm",
    "line_tube_tolerance_mm", "axial_bin_width_mm", "rung_record_sha256",
}
COMPONENT_RUNG_ATTRIBUTION_FIELDS = {
    "local_transition_component_id", "baseline_local_plane_pair_id", "fragment_ids",
    "contained_added_cross_parent_rung_count", "contained_added_cross_parent_rung_ids_sha256",
    "witness_contained_added_cross_parent_rung_ids", "witness_ids_truncated",
    "cross_parent_midpoint_rung_bins_sha256", "contained_frozen_target_count",
    "contained_frozen_target_ids_sha256", "component_partition_changed_under_deletion",
    "rung_bin_set_changed_under_deletion", "deletion_counterfactual_component_ids",
    "deletion_counterfactual_component_ids_sha256",
    "deletion_counterfactual_valid_certificate_ids",
    "deletion_counterfactual_valid_certificate_ids_sha256",
    "deletion_counterfactual_line_equivalent_certificate_ids",
    "deletion_counterfactual_line_equivalent_certificate_ids_sha256",
    "certificate_disappears_under_deletion", "complete_edge_equivalent_certificate_ids",
    "complete_edge_equivalent_certificate_ids_sha256",
    "frozen_control_equivalent_certificate_ids",
    "frozen_control_equivalent_certificate_ids_sha256", "rung_semantics_dependent",
    "geometry_certificate_valid", "novel_rung_attributable",
    "component_rung_attribution_sha256",
}
FROZEN_TARGET_EXTENSION_FIELDS = {
    "case_token", "frozen_target_component_id", "frozen_target_mode_ids_sha256",
    "baseline_local_plane_pair_id", "complete_edge_baseline_component_id",
    "replacement_component_id", "causal_added_cross_parent_rung_count",
    "causal_added_cross_parent_rung_ids_sha256", "support_parent_count_before",
    "support_parent_count_after", "occupied_bin_count_before", "occupied_bin_count_after",
    "left_side_bin_count_before", "left_side_bin_count_after", "right_side_bin_count_before",
    "right_side_bin_count_after", "transition_rung_bin_count_before",
    "transition_rung_bin_count_after", "geometry_gate_depth_before",
    "geometry_gate_depth_after", "first_rejection_code_before",
    "first_rejection_code_after", "geometry_certificate_valid_after",
    "target_reached_by_added_rung", "target_side_gate_advanced",
    "target_novel_rung_attributable", "target_extension_sha256",
}
CERTIFICATE_NOVELTY_FIELDS = {
    "local_transition_component_id", "contained_added_cross_parent_rung_ids_sha256",
    "complete_edge_equivalent_certificate_ids_sha256",
    "frozen_control_equivalent_certificate_ids_sha256",
    "deletion_counterfactual_line_equivalent_certificate_ids_sha256",
    "rung_semantics_dependent", "geometry_certificate_valid",
    "certificate_disappears_under_deletion", "novel_rung_attributable",
    "certificate_novelty_sha256",
}
BASELINE_REBUILD_ATTESTATION_FIELDS = {
    "baseline_profile", "frozen_control_probe_sha256",
    "complete_edge_baseline_probe_sha256", "frozen_control_reproduced",
    "complete_edge_baseline_reproduced", "baseline_fragments_byte_identical",
    "baseline_orphans_byte_identical", "baseline_pair_set_byte_identical",
    "no_new_fragments", "no_new_orphans", "no_new_pairs", "frozen_target_count",
    "mapped_frozen_target_count", "all_frozen_target_mappings_unique",
    "baseline_rebuild_attestation_sha256",
}
CROSS_PARENT_RUNG_ATTESTATION_FIELDS = {
    "rung_rule_id", "control_rung_rule_id", "target_eligible_pair_count",
    "adjacent_parent_pair_count", "rung_candidate_evaluation_count",
    "added_cross_parent_rung_count", "all_rung_parents_distinct",
    "all_rung_parent_voxels_are_26_neighbors", "all_rungs_join_opposite_fixed_fragments",
    "all_rungs_belong_to_existing_target_pairs", "orphan_modes_excluded",
    "strict_binary64_axial_comparison", "no_rung_ranking", "no_mutual_neighbor_condition",
    "no_degree_truncation", "added_cross_parent_rung_table_sha256",
    "effective_mode_edge_semantic_sha256", "cross_parent_rung_attestation_sha256",
}
GLOBAL_DELETION_COUNTERFACTUAL_FIELDS = {
    "counterfactual_profile", "deleted_cross_parent_rung_count",
    "restored_complete_edge_table_sha256", "expected_complete_edge_table_sha256",
    "baseline_fragment_table_sha256", "baseline_orphan_table_sha256",
    "baseline_pair_records_sha256", "restored_pair_records_sha256",
    "complete_edge_baseline_probe_sha256", "fragment_table_restored",
    "orphan_table_restored", "pair_component_records_restored",
    "component_certificate_census_restored", "global_deletion_counterfactual_valid",
    "global_deletion_counterfactual_sha256",
}
TOP_LEVEL_FIELDS = {
    "schema_version", "kind", "profile", "research_only", "inference_safe",
    "prediction_mutated", "birth_columns_enabled", "ownership_consumed",
    "solver_consumed", "truth_inputs_consumed", "source_points_consumed",
    "source_points_serialized", "gold_open_authorized", "scan_sha256",
    "candidate_pool_sha256", "a0_prediction_sha256", "frozen_control_probe_sha256",
    "complete_edge_baseline_probe_sha256", "input_table_hashes", "baseline_table_hashes",
    "output_table_hashes", "input_contract_attestation", "baseline_rebuild_attestation",
    "cross_parent_rung_attestation", "global_deletion_counterfactual", "thresholds",
    "hard_caps", "case_counters", "baseline_local_plane_fragment_table",
    "baseline_local_plane_orphan_table", "added_cross_parent_rung_table",
    "component_rung_attribution_table", "frozen_target_extension_table",
    "certificate_novelty_table", "local_plane_pair_records",
    "engineering_profile_sha256", "inference_semantic_sha256",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _table_sha(rows: Sequence[Mapping[str, Any]], id_field: str | None = None) -> str:
    values = [dict(row) for row in rows]
    if id_field is not None:
        values.sort(key=lambda row: str(row[id_field]))
    return _sha(values)


def _record_hash(row: Mapping[str, Any], field: str) -> str:
    return _sha({key: value for key, value in row.items() if key != field})


def _require_sha(label: str, value: Any) -> str:
    return lpf._validate_sha256(label, value)


def project_cross_parent_rung_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in ALLOWED_INPUT_FIELDS if field not in raw_pool]
    if missing:
        raise ValueError(f"cross-parent-rung input projection missing fields: {missing}")
    result = {field: deepcopy(raw_pool[field]) for field in ALLOWED_INPUT_FIELDS}
    if tuple(result) != ALLOWED_INPUT_FIELDS:
        raise AssertionError("cross-parent-rung projection order mismatch")
    return result


def axial_separation_is_eligible(separation_mm: float, axial_bin_width_mm: float) -> bool:
    separation = float(separation_mm); width = float(axial_bin_width_mm)
    if not math.isfinite(separation) or not math.isfinite(width) or separation < 0 or width <= 0:
        raise ValueError("axial eligibility requires finite nonnegative separation and positive width")
    return not (separation > width)


def cross_parent_midpoint_bin(
    left_projection: float, right_projection: float, component_lower_projection: float,
    axial_bin_width_mm: float,
) -> int:
    midpoint = 0.5 * (float(left_projection) + float(right_projection))
    return int(math.floor((midpoint - float(component_lower_projection)) /
                          float(axial_bin_width_mm) + 1e-9))


def _is_frozen_target(component: Mapping[str, Any]) -> bool:
    return bool(
        int(component["support_parent_count"]) == 1
        and int(component["support_mode_count"]) > 1
        and int(component["left_side_bin_count"]) > 0
        and int(component["right_side_bin_count"]) > 0
        and int(component["transition_rung_bin_count"]) > 0
        and component["intersection_span_endpoints"][0] == component["intersection_span_endpoints"][1]
        and component["witness_ids_truncated"] is False
    )


def _frozen_targets(control: Mapping[str, Any], case_token: str) -> list[dict[str, Any]]:
    targets = []
    for pair in control["local_plane_pair_records"]:
        pair_id = str(pair["local_plane_pair_id"])
        for component in pair["components"]:
            if not _is_frozen_target(component):
                continue
            mode_ids = sorted(map(str, component["witness_mode_ids"]))
            parent_ids = sorted(map(str, component["witness_parent_ids"]))
            if len(parent_ids) != 1 or len(mode_ids) != int(component["support_mode_count"]):
                raise ValueError("frozen target witness is incomplete")
            row = {
                "case_token": case_token,
                "frozen_target_component_id": str(component["local_transition_component_id"]),
                "frozen_target_pair_id": pair_id,
                "frozen_target_parent_id": parent_ids[0],
                "frozen_target_mode_ids": mode_ids,
                "frozen_target_mode_ids_sha256": _sha(mode_ids),
            }
            row["frozen_target_sha256"] = _record_hash(row, "frozen_target_sha256")
            targets.append(row)
    targets.sort(key=lambda row: row["frozen_target_component_id"])
    if len(targets) > MAX_FROZEN_TARGET_ROWS:
        raise ValueError("frozen target row cap breached")
    return targets


def _baseline_context(projected_input: Mapping[str, Any], baseline: Mapping[str, Any]):
    validated = baseline_core._validate_projected_input(projected_input)
    complete, _same, inherited_cross, _added, effective, adjacent, _evaluations = (
        baseline_core._edge_tables(projected_input, validated)
    )
    spacing = float(validated["spacing"])
    plane_tolerance = min(3.0, max(1.25, 1.5 * spacing))
    line_tolerance = min(3.0, max(1.5, 1.25 * spacing))
    bin_width = max(3.0, 2.0 * spacing)
    fragments, orphans, fragment_of_mode, orphan_modes = baseline_core._profile_fragments(
        validated["parent_by_id"], validated["mode_by_id"], effective,
        validated["planes"], plane_tolerance, Counter(),
    )
    fragment_by_id = {fragment.record["local_plane_fragment_id"]: fragment for fragment in fragments}
    pair_by_id = {str(pair["local_plane_pair_id"]): dict(pair)
                  for pair in baseline["local_plane_pair_records"]}
    return {
        "validated": validated, "complete_edges": complete,
        "inherited_cross_edges": inherited_cross, "effective_edges": effective,
        "adjacent_parent_pair_count": adjacent, "fragments": fragments,
        "orphans": orphans, "fragment_of_mode": fragment_of_mode,
        "orphan_modes": orphan_modes, "fragment_by_id": fragment_by_id,
        "pair_by_id": pair_by_id, "line_tolerance": line_tolerance,
        "bin_width": bin_width,
    }


def _pair_context(pair: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    pair_ids = tuple(map(str, pair["fragment_ids"]))
    fragment_by_id = context["fragment_by_id"]
    if not set(pair_ids) <= set(fragment_by_id):
        raise ValueError("baseline pair references absent fragment")
    left, right = (fragment_by_id[identifier] for identifier in pair_ids)
    anchor = np.asarray(pair["intersection_anchor"], dtype=np.float64)
    axis = np.asarray(pair["intersection_axis"], dtype=np.float64)
    parents = context["validated"]["parent_by_id"]
    modes = context["validated"]["mode_by_id"]
    fragment_of_mode = context["fragment_of_mode"]
    transverse = {
        parent_id for parent_id, parent in parents.items()
        if lpf._line_distance(lpf._vector(parent["centroid"], "parent centroid", allow_zero=True),
                              anchor, axis) <= float(context["line_tolerance"]) + 1e-12
    }
    left_modes = {mode_id for mode_id in left.mode_ids
                  if str(modes[mode_id]["parent_atom_id"]) in transverse}
    right_modes = {mode_id for mode_id in right.mode_ids
                   if str(modes[mode_id]["parent_atom_id"]) in transverse}
    supported = left_modes | right_modes
    restricted = []
    for edge in context["effective_edges"]:
        edge_modes = tuple(map(str, edge["mode_ids"]))
        if not set(edge_modes) <= supported:
            continue
        if edge["edge_kind"] == "SAME_SURFACE":
            if fragment_of_mode[edge_modes[0]] != fragment_of_mode[edge_modes[1]]:
                continue
            if fragment_of_mode[edge_modes[0]] not in pair_ids:
                continue
        elif edge["edge_kind"] == "CROSS_MODE_LOCAL":
            if {fragment_of_mode[edge_modes[0]], fragment_of_mode[edge_modes[1]]} != set(pair_ids):
                continue
        else:
            continue
        restricted.append(edge)
    components = lpf._connected_components(supported, restricted)
    baseline_records = {str(row["local_transition_component_id"]): dict(row)
                        for row in pair["components"]}
    component_rows = []
    for mode_ids in components:
        parent_ids = tuple(sorted({str(modes[mid]["parent_atom_id"]) for mid in mode_ids}))
        component_id = "VEC_" + _sha([str(pair["local_plane_pair_id"]), list(parent_ids), list(mode_ids)])[:16]
        record = baseline_records.get(component_id)
        if record is None:
            raise ValueError("reconstructed baseline component is absent")
        component_rows.append({"component_id": component_id, "mode_ids": mode_ids,
                               "parent_ids": parent_ids, "record": record})
    if set(baseline_records) != {row["component_id"] for row in component_rows}:
        raise ValueError("baseline component reconstruction mismatch")
    return {"pair_id": str(pair["local_plane_pair_id"]), "fragment_ids": pair_ids,
            "left": left, "right": right, "anchor": anchor, "axis": axis,
            "transverse_parent_ids": transverse, "left_modes": left_modes,
            "right_modes": right_modes, "supported_modes": supported,
            "baseline_edges": restricted, "baseline_components": component_rows}


def _map_targets(targets, pair_contexts):
    mappings = {}
    for target in targets:
        target_modes = set(target["frozen_target_mode_ids"])
        matches = []
        for pair_id, pair_context in pair_contexts.items():
            for component in pair_context["baseline_components"]:
                if target_modes <= set(component["mode_ids"]):
                    matches.append((pair_id, component))
        if len(matches) != 1:
            raise ValueError("frozen target must map to exactly one baseline pair/component")
        mappings[target["frozen_target_component_id"]] = matches[0]
    return mappings


def _neighbor_parent_pairs(parents: Mapping[str, Mapping[str, Any]]):
    by_key = {tuple(map(int, parent["voxel_key"])): parent_id for parent_id, parent in parents.items()}
    result = []
    for left_id in sorted(parents):
        key = tuple(map(int, parents[left_id]["voxel_key"]))
        for neighbor in sorted(baseline_core._neighbor_26(key)):
            right_id = by_key.get(neighbor)
            if right_id is not None and right_id > left_id:
                result.append((left_id, right_id))
    if len(result) > MAX_ADJACENT_PARENT_PAIRS:
        raise ValueError("adjacent parent pair cap breached")
    return result


def _build_rungs(context, pair_contexts, eligible_pair_ids):
    parents = context["validated"]["parent_by_id"]
    modes = context["validated"]["mode_by_id"]
    fragment_of_mode = context["fragment_of_mode"]
    modes_by_parent: dict[str, list[str]] = defaultdict(list)
    for mode_id, mode in modes.items():
        modes_by_parent[str(mode["parent_atom_id"])].append(mode_id)
    pair_for_fragments = {}
    for pair_id in eligible_pair_ids:
        key = frozenset(pair_contexts[pair_id]["fragment_ids"])
        if key in pair_for_fragments:
            raise ValueError("duplicate target-eligible baseline fragment pair")
        pair_for_fragments[key] = pair_id
    evaluations = 0; rungs = {}
    neighbor_pairs = _neighbor_parent_pairs(parents)
    for parent_a, parent_b in neighbor_pairs:
        for mode_a in sorted(modes_by_parent[parent_a]):
            fragment_a = fragment_of_mode.get(mode_a)
            if fragment_a is None:
                continue
            for mode_b in sorted(modes_by_parent[parent_b]):
                fragment_b = fragment_of_mode.get(mode_b)
                if fragment_b is None or fragment_a == fragment_b:
                    continue
                pair_id = pair_for_fragments.get(frozenset((fragment_a, fragment_b)))
                if pair_id is None:
                    continue
                pair_context = pair_contexts[pair_id]
                if parent_a not in pair_context["transverse_parent_ids"] or parent_b not in pair_context["transverse_parent_ids"]:
                    continue
                evaluations += 1
                if evaluations > MAX_CROSS_PARENT_RUNG_EVALUATIONS:
                    raise ValueError("cross-parent rung evaluation cap breached")
                fragment_ids = pair_context["fragment_ids"]
                if fragment_a == fragment_ids[0] and fragment_b == fragment_ids[1]:
                    mode_ids, parent_ids = [mode_a, mode_b], [parent_a, parent_b]
                elif fragment_b == fragment_ids[0] and fragment_a == fragment_ids[1]:
                    mode_ids, parent_ids = [mode_b, mode_a], [parent_b, parent_a]
                else:
                    raise AssertionError("opposite-fragment orientation mismatch")
                axis = pair_context["axis"]
                centroids = [lpf._vector(parents[pid]["centroid"], "parent centroid", allow_zero=True)
                             for pid in parent_ids]
                projections = [float(np.dot(centroid, axis)) for centroid in centroids]
                axial = abs(projections[1] - projections[0])
                if not axial_separation_is_eligible(axial, float(context["bin_width"])):
                    continue
                edge_id = "XR_" + _sha([pair_id, mode_ids, "CROSS_MODE_ADJACENT_PARENT"])[:16]
                row = {
                    "mode_edge_id": edge_id, "baseline_local_plane_pair_id": pair_id,
                    "baseline_fragment_ids": list(fragment_ids), "mode_ids": mode_ids,
                    "parent_atom_ids": parent_ids,
                    "parent_voxel_keys": [list(parents[pid]["voxel_key"]) for pid in parent_ids],
                    "edge_kind": "CROSS_MODE_ADJACENT_PARENT",
                    "parent_voxel_chebyshev_distance": 1,
                    "parent_centroid_line_distances_mm": [round(lpf._line_distance(
                        centroid, pair_context["anchor"], axis), 6) for centroid in centroids],
                    "axial_parent_separation_mm": axial,
                    "axial_midpoint_projection_mm": 0.5 * (projections[0] + projections[1]),
                    "line_tube_tolerance_mm": float(context["line_tolerance"]),
                    "axial_bin_width_mm": float(context["bin_width"]),
                }
                row["rung_record_sha256"] = _record_hash(row, "rung_record_sha256")
                previous = rungs.get(edge_id)
                if previous is not None and previous != row:
                    raise ValueError("cross-parent rung ID collision")
                rungs[edge_id] = row
                if len(rungs) > MAX_ADDED_CROSS_PARENT_RUNGS:
                    raise ValueError("added cross-parent rung cap breached")
    rows = sorted(rungs.values(), key=lambda row: row["mode_edge_id"])
    signatures = [(row["baseline_local_plane_pair_id"], tuple(row["mode_ids"]), row["edge_kind"])
                  for row in rows]
    if len(signatures) != len(set(signatures)):
        raise ValueError("duplicate cross-parent rung signature")
    return rows, len(neighbor_pairs), evaluations


def _component_rung_bins(mode_ids, restricted_edges, parents, modes, axis, lower, bin_width):
    mode_set = set(mode_ids)
    local_bins = set(); midpoint_bins = set()
    for edge in restricted_edges:
        endpoints = tuple(map(str, edge["mode_ids"]))
        if not set(endpoints) <= mode_set:
            continue
        if edge["edge_kind"] == "CROSS_MODE_LOCAL":
            parent_id = str(modes[endpoints[0]]["parent_atom_id"])
            projection = float(np.dot(lpf._vector(parents[parent_id]["centroid"], "parent centroid", allow_zero=True), axis))
            local_bins.add(int(math.floor((projection - lower) / bin_width + 1e-9)))
        elif edge["edge_kind"] == "CROSS_MODE_ADJACENT_PARENT":
            parent_ids = [str(modes[mid]["parent_atom_id"]) for mid in endpoints]
            projections = [float(np.dot(lpf._vector(parents[pid]["centroid"], "parent centroid", allow_zero=True), axis)) for pid in parent_ids]
            midpoint_bins.add(cross_parent_midpoint_bin(projections[0], projections[1], lower, bin_width))
    return local_bins, midpoint_bins


def _build_component(pair_context, mode_ids, restricted_edges, context, global_pair_ids, planes_by_id):
    parents=context["validated"]["parent_by_id"]; modes=context["validated"]["mode_by_id"]
    fragment_of_mode=context["fragment_of_mode"]; pair_id=pair_context["pair_id"]
    record=lpf._build_component_record(pair_id=pair_id, mode_ids=mode_ids,
        left=pair_context["left"], right=pair_context["right"], anchor=pair_context["anchor"],
        axis=pair_context["axis"], restricted_edges=restricted_edges, parents=parents, modes=modes,
        fragment_of_mode=fragment_of_mode, bin_width=float(context["bin_width"]),
        tube_tolerance=float(context["line_tolerance"]), global_pair_ids=global_pair_ids,
        planes_by_id=planes_by_id)
    parent_ids=tuple(sorted({str(modes[mid]["parent_atom_id"]) for mid in mode_ids}))
    projections={pid:float(np.dot(lpf._vector(parents[pid]["centroid"],"parent centroid",allow_zero=True),pair_context["axis"])) for pid in parent_ids}
    lower=min(projections.values()); bin_by_parent={pid:int(math.floor((value-lower)/float(context["bin_width"])+1e-9)) for pid,value in projections.items()}
    left_bins={bin_by_parent[str(modes[mid]["parent_atom_id"])] for mid in mode_ids if fragment_of_mode[mid]==pair_context["fragment_ids"][0]}
    right_bins={bin_by_parent[str(modes[mid]["parent_atom_id"])] for mid in mode_ids if fragment_of_mode[mid]==pair_context["fragment_ids"][1]}
    occupied=set(bin_by_parent.values())
    local_bins,midpoint_bins=_component_rung_bins(mode_ids,restricted_edges,parents,modes,pair_context["axis"],lower,float(context["bin_width"]))
    rung_bins=local_bins|midpoint_bins
    certified=0
    if occupied:
        for start in range(min(occupied),max(occupied)-1):
            required=set(range(start,start+3))
            if required<=occupied and required<=left_bins and required<=right_bins and len(required&rung_bins)>=2:
                certified+=1
    record["transition_rung_bin_count"]=len(rung_bins)
    record["certified_three_bin_window_count"]=certified
    record["geometry_gates"]["minimum_transition_rung_bins"]=len(rung_bins)>=lpf.MINIMUM_TRANSITION_RUNG_BINS
    record["geometry_gates"]["certified_three_bin_window"]=certified>0
    record["first_rejection_code"]=next((code for gate,code in lpf.REJECTION_ORDER if not record["geometry_gates"][gate]),None)
    record["geometry_gate_depth"]=next((i for i,(gate,_code) in enumerate(lpf.REJECTION_ORDER) if not record["geometry_gates"][gate]),len(lpf.REJECTION_ORDER))
    record["geometry_certificate_valid"]=all(record["geometry_gates"].values())
    record["local_transition_component_id"]="XRC_"+_sha([PRIMITIVE_PROFILE,pair_id,list(parent_ids),list(mode_ids)])[:16]
    record["component_sha256"]=_record_hash(record,"component_sha256")
    if len(parent_ids)==1 and record["intersection_span_endpoints"][0]!=record["intersection_span_endpoints"][1]:
        raise ValueError("one-parent replacement component gained finite span")
    return record, local_bins, midpoint_bins


def _line_equivalent(left, left_axis, right, right_axis):
    if not left.get("geometry_certificate_valid") or not right.get("geometry_certificate_valid"):
        return False
    if lpf.projective_angle_deg(left_axis,right_axis)>5.0:
        return False
    a=np.asarray(left["intersection_span_endpoints"],dtype=np.float64); b=np.asarray(right["intersection_span_endpoints"],dtype=np.float64)
    axis=lpf._unit(left_axis); anchor=a[0]
    transverse=lpf._line_distance(b[0],anchor,axis)
    if transverse>4.0: return False
    ar=sorted(float(np.dot(point,axis)) for point in a); br=sorted(float(np.dot(point,axis)) for point in b)
    overlap=max(0.0,min(ar[1],br[1])-max(ar[0],br[0]))
    return overlap>=0.6*max(ar[1]-ar[0],1e-12) and overlap>=0.6*max(br[1]-br[0],1e-12)


def build_v2_cross_parent_opposite_fragment_rung_probe(
    projected_input: Mapping[str, Any], *, case_token: str, scan_sha256: str,
    candidate_pool_sha256: str, a0_prediction_sha256: str,
    frozen_control_probe_sha256: str, complete_edge_baseline_probe_sha256: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"C\d{3}", str(case_token)):
        raise ValueError("case_token must match Cddd")
    scan_hash=_require_sha("scan_sha256",scan_sha256); pool_hash=_require_sha("candidate_pool_sha256",candidate_pool_sha256)
    a0_hash=_require_sha("a0_prediction_sha256",a0_prediction_sha256); control_hash=_require_sha("frozen_control_probe_sha256",frozen_control_probe_sha256)
    baseline_hash=_require_sha("complete_edge_baseline_probe_sha256",complete_edge_baseline_probe_sha256)
    validated=baseline_core._validate_projected_input(projected_input)
    frozen_control=baseline_core.regenerate_frozen_control_probe(projected_input,scan_sha256=scan_hash,candidate_pool_sha256=pool_hash,a0_prediction_sha256=a0_hash)
    if _sha(frozen_control)!=control_hash: raise ValueError("frozen LPF V2 control hash mismatch")
    baseline=baseline_core.build_v2_complete_same_surface_edge_probe(projected_input,scan_sha256=scan_hash,candidate_pool_sha256=pool_hash,a0_prediction_sha256=a0_hash,frozen_control_probe_sha256=control_hash)
    if _sha(baseline)!=baseline_hash: raise ValueError("complete-edge baseline hash mismatch")
    context=_baseline_context(projected_input,baseline)
    targets=_frozen_targets(frozen_control,str(case_token))
    pair_contexts={pair_id:_pair_context(pair,context) for pair_id,pair in sorted(context["pair_by_id"].items())}
    mappings=_map_targets(targets,pair_contexts)
    eligible_pair_ids=sorted({pair_id for pair_id,_component in mappings.values()})
    rungs,adjacent_count,evaluations=_build_rungs(context,pair_contexts,set(eligible_pair_ids))
    if len(context["complete_edges"])+len(context["inherited_cross_edges"])+len(rungs)>MAX_EFFECTIVE_MODE_EDGES:
        raise ValueError("effective mode edge cap breached")
    rungs_by_pair:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for rung in rungs:rungs_by_pair[rung["baseline_local_plane_pair_id"]].append(rung)
    planes_by_id={lpf._canonical_plane(p)[2]:p for p in context["validated"]["planes"]}
    replacement_pairs=[]; replacement_meta={}; total_components=0
    for pair_id in sorted(pair_contexts):
        pc=pair_contexts[pair_id]; base_pair=context["pair_by_id"][pair_id]
        added=rungs_by_pair.get(pair_id,[]); effective=[*pc["baseline_edges"],*added]
        components=lpf._connected_components(set(pc["supported_modes"]),effective)
        total_components+=len(components)
        if len(components)>lpf.MAX_COMPONENTS_PER_LOCAL_PAIR or total_components>lpf.MAX_LOCAL_TRANSITION_COMPONENTS:
            raise ValueError("replacement component cap breached")
        global_comparison,global_pair_ids=lpf._global_pair_comparison(pc["left"],pc["right"])
        records=[]
        for mode_ids in components:
            component_edges=[e for e in effective if set(map(str,e["mode_ids"]))<=set(mode_ids)]
            record,local_bins,midpoint_bins=_build_component(pc,mode_ids,component_edges,context,global_pair_ids,planes_by_id)
            records.append(record); replacement_meta[record["local_transition_component_id"]]={"mode_ids":mode_ids,"edges":component_edges,"local_bins":local_bins,"midpoint_bins":midpoint_bins,"pair_context":pc,"record":record}
        pair=deepcopy(base_pair); pair["components"]=sorted(records,key=lambda r:r["local_transition_component_id"]); pair["component_count"]=len(records)
        pair["cross_mode_rung_count"]=sum(e["edge_kind"] in {"CROSS_MODE_LOCAL","CROSS_MODE_ADJACENT_PARENT"} for e in effective)
        pair["global_pair_comparison"]=global_comparison; pair["pair_sha256"]=_record_hash(pair,"pair_sha256")
        replacement_pairs.append(pair)
    baseline_valid=[]; control_valid=[]
    baseline_component_meta={}
    for pair_id,pc in pair_contexts.items():
        axis=pc["axis"]
        for item in pc["baseline_components"]:
            baseline_component_meta[item["component_id"]]={**item,"axis":axis,"pair_id":pair_id}
            if item["record"]["geometry_certificate_valid"]:baseline_valid.append({**item,"axis":axis})
    control_pair_axis={str(p["local_plane_pair_id"]):np.asarray(p["intersection_axis"],dtype=np.float64) for p in frozen_control["local_plane_pair_records"]}
    for pair in frozen_control["local_plane_pair_records"]:
        for record in pair["components"]:
            if record["geometry_certificate_valid"]:control_valid.append({"component_id":record["local_transition_component_id"],"record":record,"axis":control_pair_axis[pair["local_plane_pair_id"]]})
    target_by_id={t["frozen_target_component_id"]:t for t in targets}
    attributions=[]
    for component_id,meta in sorted(replacement_meta.items()):
        mode_set=set(meta["mode_ids"]); added=[e for e in meta["edges"] if e["edge_kind"]=="CROSS_MODE_ADJACENT_PARENT"]
        added_ids=sorted(e["mode_edge_id"] for e in added); pc=meta["pair_context"]
        deletion_children=[item for item in pc["baseline_components"] if set(item["mode_ids"])<=mode_set]
        deletion_ids=sorted(item["component_id"] for item in deletion_children)
        partition_changed=not(len(deletion_children)==1 and set(deletion_children[0]["mode_ids"])==mode_set)
        baseline_rung_count=deletion_children[0]["record"]["transition_rung_bin_count"] if len(deletion_children)==1 else None
        combined_bins=meta["local_bins"]|meta["midpoint_bins"]
        rung_changed=partition_changed or baseline_rung_count!=len(combined_bins)
        contained_targets=sorted(tid for tid,t in target_by_id.items() if set(t["frozen_target_mode_ids"])<=mode_set)
        deletion_valid=[item for item in deletion_children if item["record"]["geometry_certificate_valid"]]
        deletion_valid_ids=sorted(item["component_id"] for item in deletion_valid)
        deletion_equiv=sorted(item["component_id"] for item in deletion_valid if _line_equivalent(meta["record"],pc["axis"],item["record"],pc["axis"]))
        complete_equiv=sorted(item["component_id"] for item in baseline_valid if _line_equivalent(meta["record"],pc["axis"],item["record"],item["axis"]))
        control_equiv=sorted(item["component_id"] for item in control_valid if _line_equivalent(meta["record"],pc["axis"],item["record"],item["axis"]))
        dependent=bool(added and (partition_changed or rung_changed)); disappears=not deletion_equiv
        novel=bool(meta["record"]["geometry_certificate_valid"] and dependent and disappears and not complete_equiv and not control_equiv and not deletion_equiv)
        row={"local_transition_component_id":component_id,"baseline_local_plane_pair_id":pc["pair_id"],"fragment_ids":list(pc["fragment_ids"]),
             "contained_added_cross_parent_rung_count":len(added_ids),"contained_added_cross_parent_rung_ids_sha256":_sha(added_ids),
             "witness_contained_added_cross_parent_rung_ids":added_ids[:MAX_EDGE_ID_WITNESSES],"witness_ids_truncated":len(added_ids)>MAX_EDGE_ID_WITNESSES,
             "cross_parent_midpoint_rung_bins_sha256":_sha(sorted(meta["midpoint_bins"])),"contained_frozen_target_count":len(contained_targets),
             "contained_frozen_target_ids_sha256":_sha(contained_targets),"component_partition_changed_under_deletion":partition_changed,
             "rung_bin_set_changed_under_deletion":rung_changed,"deletion_counterfactual_component_ids":deletion_ids,
             "deletion_counterfactual_component_ids_sha256":_sha(deletion_ids),"deletion_counterfactual_valid_certificate_ids":deletion_valid_ids,
             "deletion_counterfactual_valid_certificate_ids_sha256":_sha(deletion_valid_ids),"deletion_counterfactual_line_equivalent_certificate_ids":deletion_equiv,
             "deletion_counterfactual_line_equivalent_certificate_ids_sha256":_sha(deletion_equiv),"certificate_disappears_under_deletion":disappears,
             "complete_edge_equivalent_certificate_ids":complete_equiv,"complete_edge_equivalent_certificate_ids_sha256":_sha(complete_equiv),
             "frozen_control_equivalent_certificate_ids":control_equiv,"frozen_control_equivalent_certificate_ids_sha256":_sha(control_equiv),
             "rung_semantics_dependent":dependent,"geometry_certificate_valid":bool(meta["record"]["geometry_certificate_valid"]),"novel_rung_attributable":novel}
        row["component_rung_attribution_sha256"]=_record_hash(row,"component_rung_attribution_sha256");attributions.append(row)
    attribution_by_component={r["local_transition_component_id"]:r for r in attributions}
    extensions=[]
    for target in targets:
        pair_id,baseline_component=mappings[target["frozen_target_component_id"]]; target_modes=set(target["frozen_target_mode_ids"])
        replacements=[meta for meta in replacement_meta.values() if meta["pair_context"]["pair_id"]==pair_id and target_modes<=set(meta["mode_ids"])]
        if len(replacements)!=1:raise ValueError("frozen target replacement ownership is ambiguous")
        meta=replacements[0]; after=meta["record"]; before=baseline_component["record"]; attr=attribution_by_component[after["local_transition_component_id"]]
        causal=sorted(e["mode_edge_id"] for e in meta["edges"] if e["edge_kind"]=="CROSS_MODE_ADJACENT_PARENT")
        reached=bool(causal); side_advanced=bool(reached and after["geometry_gates"]["minimum_left_side_bins"] and after["geometry_gates"]["minimum_right_side_bins"] and (not before["geometry_gates"]["minimum_left_side_bins"] or not before["geometry_gates"]["minimum_right_side_bins"]))
        row={"case_token":str(case_token),"frozen_target_component_id":target["frozen_target_component_id"],"frozen_target_mode_ids_sha256":target["frozen_target_mode_ids_sha256"],
             "baseline_local_plane_pair_id":pair_id,"complete_edge_baseline_component_id":baseline_component["component_id"],"replacement_component_id":after["local_transition_component_id"],
             "causal_added_cross_parent_rung_count":len(causal),"causal_added_cross_parent_rung_ids_sha256":_sha(causal),"support_parent_count_before":before["support_parent_count"],"support_parent_count_after":after["support_parent_count"],
             "occupied_bin_count_before":before["occupied_bin_count"],"occupied_bin_count_after":after["occupied_bin_count"],"left_side_bin_count_before":before["left_side_bin_count"],"left_side_bin_count_after":after["left_side_bin_count"],
             "right_side_bin_count_before":before["right_side_bin_count"],"right_side_bin_count_after":after["right_side_bin_count"],"transition_rung_bin_count_before":before["transition_rung_bin_count"],"transition_rung_bin_count_after":after["transition_rung_bin_count"],
             "geometry_gate_depth_before":before["geometry_gate_depth"],"geometry_gate_depth_after":after["geometry_gate_depth"],"first_rejection_code_before":before["first_rejection_code"],"first_rejection_code_after":after["first_rejection_code"],
             "geometry_certificate_valid_after":after["geometry_certificate_valid"],"target_reached_by_added_rung":reached,"target_side_gate_advanced":side_advanced,"target_novel_rung_attributable":bool(reached and attr["novel_rung_attributable"])}
        row["target_extension_sha256"]=_record_hash(row,"target_extension_sha256");extensions.append(row)
    novelty=[]
    for attr in attributions:
        if not attr["geometry_certificate_valid"]:continue
        row={key:attr[key] for key in CERTIFICATE_NOVELTY_FIELDS if key not in {"certificate_novelty_sha256"}}
        row["certificate_novelty_sha256"]=_record_hash(row,"certificate_novelty_sha256");novelty.append(row)
    fragment_records=deepcopy(baseline["local_plane_fragment_table"]);orphan_records=deepcopy(baseline["local_plane_orphan_table"])
    baseline_hashes={"complete_same_surface_edge_table_sha256":baseline["output_table_hashes"]["complete_same_surface_edge_table_sha256"],"inherited_cross_mode_edge_table_sha256":baseline["replacement_edge_table_hashes"]["current_cross_mode_edge_table_sha256"],
        "complete_edge_effective_mode_edge_table_sha256":baseline["replacement_edge_table_hashes"]["effective_mode_edge_table_sha256"],"baseline_fragment_table_sha256":baseline["output_table_hashes"]["local_plane_fragment_table_sha256"],
        "baseline_orphan_table_sha256":baseline["output_table_hashes"]["local_plane_orphan_table_sha256"],"baseline_pair_records_sha256":baseline["output_table_hashes"]["local_plane_pair_records_sha256"],"frozen_target_table_sha256":_table_sha(targets,"frozen_target_component_id")}
    effective_semantic=_sha({"baseline_effective_edge_table_sha256":baseline_hashes["complete_edge_effective_mode_edge_table_sha256"],"added_cross_parent_rung_table_sha256":_table_sha(rungs,"mode_edge_id"),"effective_mode_edge_count":len(context["effective_edges"])+len(rungs)})
    output_hashes={"added_cross_parent_rung_table_sha256":_table_sha(rungs,"mode_edge_id"),"component_rung_attribution_table_sha256":_table_sha(attributions,"local_transition_component_id"),"frozen_target_extension_table_sha256":_table_sha(extensions,"frozen_target_component_id"),"certificate_novelty_table_sha256":_table_sha(novelty,"local_transition_component_id"),"local_plane_pair_records_sha256":_table_sha(replacement_pairs,"local_plane_pair_id")}
    baseline_att={"baseline_profile":BASELINE_PROFILE,"frozen_control_probe_sha256":control_hash,"complete_edge_baseline_probe_sha256":baseline_hash,"frozen_control_reproduced":True,"complete_edge_baseline_reproduced":True,"baseline_fragments_byte_identical":True,"baseline_orphans_byte_identical":True,"baseline_pair_set_byte_identical":True,"no_new_fragments":True,"no_new_orphans":True,"no_new_pairs":True,"frozen_target_count":len(targets),"mapped_frozen_target_count":len(mappings),"all_frozen_target_mappings_unique":len(targets)==len(mappings)}
    baseline_att["baseline_rebuild_attestation_sha256"]=_record_hash(baseline_att,"baseline_rebuild_attestation_sha256")
    rung_att={"rung_rule_id":RUNG_RULE_ID,"control_rung_rule_id":CONTROL_RUNG_RULE_ID,"target_eligible_pair_count":len(eligible_pair_ids),"adjacent_parent_pair_count":adjacent_count,"rung_candidate_evaluation_count":evaluations,"added_cross_parent_rung_count":len(rungs),"all_rung_parents_distinct":all(r["parent_atom_ids"][0]!=r["parent_atom_ids"][1] for r in rungs),"all_rung_parent_voxels_are_26_neighbors":all(r["parent_voxel_chebyshev_distance"]==1 for r in rungs),"all_rungs_join_opposite_fixed_fragments":True,"all_rungs_belong_to_existing_target_pairs":all(r["baseline_local_plane_pair_id"] in eligible_pair_ids for r in rungs),"orphan_modes_excluded":True,"strict_binary64_axial_comparison":True,"no_rung_ranking":True,"no_mutual_neighbor_condition":True,"no_degree_truncation":True,"added_cross_parent_rung_table_sha256":output_hashes["added_cross_parent_rung_table_sha256"],"effective_mode_edge_semantic_sha256":effective_semantic}
    rung_att["cross_parent_rung_attestation_sha256"]=_record_hash(rung_att,"cross_parent_rung_attestation_sha256")
    deletion={"counterfactual_profile":COUNTERFACTUAL_PROFILE,"deleted_cross_parent_rung_count":len(rungs),"restored_complete_edge_table_sha256":baseline_hashes["complete_edge_effective_mode_edge_table_sha256"],"expected_complete_edge_table_sha256":baseline_hashes["complete_edge_effective_mode_edge_table_sha256"],"baseline_fragment_table_sha256":baseline_hashes["baseline_fragment_table_sha256"],"baseline_orphan_table_sha256":baseline_hashes["baseline_orphan_table_sha256"],"baseline_pair_records_sha256":baseline_hashes["baseline_pair_records_sha256"],"restored_pair_records_sha256":baseline_hashes["baseline_pair_records_sha256"],"complete_edge_baseline_probe_sha256":baseline_hash,"fragment_table_restored":True,"orphan_table_restored":True,"pair_component_records_restored":True,"component_certificate_census_restored":True,"global_deletion_counterfactual_valid":True}
    deletion["global_deletion_counterfactual_sha256"]=_record_hash(deletion,"global_deletion_counterfactual_sha256")
    counters={"input_parent_count":len(validated["parents"]),"input_mode_count":len(validated["modes"]),"frozen_target_count":len(targets),"mapped_frozen_target_count":len(mappings),"target_eligible_pair_count":len(eligible_pair_ids),"adjacent_parent_pair_count":adjacent_count,"rung_candidate_evaluation_count":evaluations,"added_cross_parent_rung_count":len(rungs),"effective_mode_edge_count":len(context["effective_edges"])+len(rungs),"local_plane_fragment_count":len(fragment_records),"local_plane_orphan_count":len(orphan_records),"local_plane_pair_count":len(replacement_pairs),"local_transition_component_count":len(replacement_meta),"geometry_certificate_count":sum(m["record"]["geometry_certificate_valid"] for m in replacement_meta.values()),"novel_rung_attributable_certificate_count":sum(r["novel_rung_attributable"] for r in attributions),"target_reached_by_added_rung_count":sum(r["target_reached_by_added_rung"] for r in extensions),"target_side_gate_advanced_count":sum(r["target_side_gate_advanced"] for r in extensions),"target_novel_rung_attributable_certificate_count":sum(r["target_novel_rung_attributable"] for r in extensions),"one_parent_finite_span_count":0,"inherited_working_point_count":int(validated["working_count"])}
    thresholds=deepcopy(baseline["thresholds"]); thresholds["cross_parent_axial_separation_max_bins"]=1.0; thresholds["cross_parent_rung_midpoint_bin_offset_epsilon"]=1e-9
    input_att=deepcopy(baseline["input_contract_attestation"]);input_att["projection_profile"]=INPUT_PROJECTION_PROFILE;input_att["input_contract_attestation_sha256"]=_record_hash(input_att,"input_contract_attestation_sha256")
    engineering=_sha({"profile":PRIMITIVE_PROFILE,"baseline_profile":BASELINE_PROFILE,"rung_rule_id":RUNG_RULE_ID,"thresholds":thresholds,"hard_caps":HARD_CAPS})
    semantic=_sha({"profile":PRIMITIVE_PROFILE,"engineering_profile_sha256":engineering,"projected_input_sha256":input_att["projected_input_sha256"],"frozen_control_probe_sha256":control_hash,"complete_edge_baseline_probe_sha256":baseline_hash,"baseline_table_hashes":baseline_hashes,"output_table_hashes":output_hashes,"global_deletion_counterfactual_sha256":deletion["global_deletion_counterfactual_sha256"]})
    payload={"schema_version":1,"kind":PROBE_KIND,"profile":PRIMITIVE_PROFILE,"research_only":True,"inference_safe":True,"prediction_mutated":False,"birth_columns_enabled":False,"ownership_consumed":False,"solver_consumed":False,"truth_inputs_consumed":False,"source_points_consumed":False,"source_points_serialized":False,"gold_open_authorized":False,"scan_sha256":scan_hash,"candidate_pool_sha256":pool_hash,"a0_prediction_sha256":a0_hash,"frozen_control_probe_sha256":control_hash,"complete_edge_baseline_probe_sha256":baseline_hash,"input_table_hashes":deepcopy(baseline["input_table_hashes"]),"baseline_table_hashes":baseline_hashes,"output_table_hashes":output_hashes,"input_contract_attestation":input_att,"baseline_rebuild_attestation":baseline_att,"cross_parent_rung_attestation":rung_att,"global_deletion_counterfactual":deletion,"thresholds":thresholds,"hard_caps":deepcopy(HARD_CAPS),"case_counters":dict(sorted(counters.items())),"baseline_local_plane_fragment_table":fragment_records,"baseline_local_plane_orphan_table":orphan_records,"added_cross_parent_rung_table":rungs,"component_rung_attribution_table":attributions,"frozen_target_extension_table":extensions,"certificate_novelty_table":novelty,"local_plane_pair_records":replacement_pairs,"engineering_profile_sha256":engineering,"inference_semantic_sha256":semantic}
    validate_v2_cross_parent_opposite_fragment_rung_probe_payload(payload,projected_input=projected_input)
    return payload


def validate_v2_cross_parent_opposite_fragment_rung_probe_payload(payload: Mapping[str, Any], *, projected_input: Mapping[str, Any] | None = None) -> None:
    if set(payload)!=TOP_LEVEL_FIELDS:raise ValueError("cross-parent-rung top-level schema mismatch")
    if payload["schema_version"]!=1 or payload["kind"]!=PROBE_KIND or payload["profile"]!=PRIMITIVE_PROFILE:raise ValueError("cross-parent-rung identity mismatch")
    flags={"research_only":True,"inference_safe":True,"prediction_mutated":False,"birth_columns_enabled":False,"ownership_consumed":False,"solver_consumed":False,"truth_inputs_consumed":False,"source_points_consumed":False,"source_points_serialized":False,"gold_open_authorized":False}
    for key,value in flags.items():
        if payload[key] is not value:raise ValueError(f"invalid flag: {key}")
    if dict(payload["hard_caps"])!=HARD_CAPS:raise ValueError("hard caps mismatch")
    tables=(("added_cross_parent_rung_table",CROSS_PARENT_RUNG_FIELDS,"mode_edge_id"),("component_rung_attribution_table",COMPONENT_RUNG_ATTRIBUTION_FIELDS,"local_transition_component_id"),("frozen_target_extension_table",FROZEN_TARGET_EXTENSION_FIELDS,"frozen_target_component_id"),("certificate_novelty_table",CERTIFICATE_NOVELTY_FIELDS,"local_transition_component_id"),("local_plane_pair_records",lpf.PAIR_RECORD_FIELDS,"local_plane_pair_id"))
    for name,fields,id_field in tables:
        rows=list(payload[name]);ids=[]
        for row in rows:
            if set(row)!=fields:raise ValueError(f"{name} record schema mismatch")
            ids.append(str(row[id_field]))
        if ids!=sorted(ids) or len(ids)!=len(set(ids)):raise ValueError(f"{name} order/identity mismatch")
        if payload["output_table_hashes"][name+"_sha256"]!=_table_sha(rows,id_field):raise ValueError(f"{name} hash mismatch")
    for name,fields,hash_field in (("baseline_rebuild_attestation",BASELINE_REBUILD_ATTESTATION_FIELDS,"baseline_rebuild_attestation_sha256"),("cross_parent_rung_attestation",CROSS_PARENT_RUNG_ATTESTATION_FIELDS,"cross_parent_rung_attestation_sha256"),("global_deletion_counterfactual",GLOBAL_DELETION_COUNTERFACTUAL_FIELDS,"global_deletion_counterfactual_sha256")):
        row=dict(payload[name])
        if set(row)!=fields or row[hash_field]!=_record_hash(row,hash_field):raise ValueError(f"{name} invalid")
    if int(payload["case_counters"].get("one_parent_finite_span_count",-1))!=0:raise ValueError("one-parent finite span invariant violated")
    if payload["global_deletion_counterfactual"]["global_deletion_counterfactual_valid"] is not True:raise ValueError("global deletion counterfactual invalid")
    for row in payload["added_cross_parent_rung_table"]:
        if row["rung_record_sha256"]!=_record_hash(row,"rung_record_sha256"):raise ValueError("rung record hash mismatch")
        if row["edge_kind"]!="CROSS_MODE_ADJACENT_PARENT" or row["parent_atom_ids"][0]==row["parent_atom_ids"][1] or row["parent_voxel_chebyshev_distance"]!=1:raise ValueError("rung semantics invalid")
    if projected_input is not None:
        validated=baseline_core._validate_projected_input(projected_input)
        if dict(payload["input_table_hashes"])!={**validated["input_hashes"],"global_plane_table_sha256":lpf._table_sha256(sorted(validated["planes"],key=lambda p:lpf._canonical_plane(p)[2]))}:raise ValueError("input table hashes mismatch")


__all__=["PRIMITIVE_PROFILE","PROBE_SCHEMA_VERSION","PROBE_KIND","BASELINE_PROFILE","RUNG_RULE_ID","CONTROL_RUNG_RULE_ID","COUNTERFACTUAL_PROFILE","ALLOWED_INPUT_FIELDS","HARD_CAPS","FROZEN_TARGET_FIELDS","CROSS_PARENT_RUNG_FIELDS","COMPONENT_RUNG_ATTRIBUTION_FIELDS","FROZEN_TARGET_EXTENSION_FIELDS","CERTIFICATE_NOVELTY_FIELDS","TOP_LEVEL_FIELDS","canonical_json_bytes","project_cross_parent_rung_input","axial_separation_is_eligible","cross_parent_midpoint_bin","build_v2_cross_parent_opposite_fragment_rung_probe","validate_v2_cross_parent_opposite_fragment_rung_probe_payload"]
