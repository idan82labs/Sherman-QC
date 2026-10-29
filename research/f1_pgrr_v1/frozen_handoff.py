"""Post-commit PGRR handoff to the exact frozen C006 target interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as complete_core
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as v1_core
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as v2_core
from .config import CONFIG, canonical_json_bytes, sha_json
from . import geometry


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "raw_source": Path("data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply"),
    "transition_pool_c006": Path("output/research/f1-gold27-20260803/candidate_pool_transition_v2_probe/C006.json"),
    "frozen_control_c006": Path("output/research/f1-gold27-20260803/local_plane_fragment_probe_v2/C006.json"),
    "complete_baseline_c006": Path("output/research/f1-gold27-20260803/v2_complete_same_surface_edge_probe_v1/C006.json"),
    "mapping_preflight": Path("output/research/f1-gold27-20260803/v2_cross_parent_rung_mapping_preflight_v1.json"),
}

EXPECTED_HASHES = {
    "raw_source": geometry.SOURCE_SHA256,
    "transition_pool_c006": "d4fb403ace02ff604160979852aec4205eb444f6bbb78f809713297c219c5940",
    "frozen_control_c006": "a97005d3cc2f8af1ffc2234b6bc769357e33aef4bf222abc79962fb20ced9ee3",
    "complete_baseline_c006": "23f661fe23bce2671cdefa3a29ad9b5b9938c023180d415357c87061029376d2",
    "mapping_preflight": "9d314e7ebe5900f65c79039d8f729add3b8f1ad5cc4e190cc330103c9eec60cd",
}

_POPCOUNT = np.asarray([int(index).bit_count() for index in range(256)], dtype=np.uint8)


@dataclass(slots=True)
class FrozenContext:
    projected: dict[str, Any]
    control: dict[str, Any]
    baseline: dict[str, Any]
    mapping: dict[str, Any]
    context: dict[str, Any]
    pair_contexts: dict[str, dict[str, Any]]
    relations_c006: list[dict[str, Any]]
    all_relations: list[dict[str, Any]]
    baseline_outcomes: list[dict[str, Any]]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_context() -> FrozenContext:
    paths = {name: (ROOT / relative).resolve() for name, relative in INPUTS.items()}
    for name, path in paths.items():
        if _file_sha(path) != EXPECTED_HASHES[name]:
            raise geometry.PGRRStop(
                "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", f"{name} hash mismatch"
            )
    raw_pool = _read(paths["transition_pool_c006"])
    projected = v2_core.project_mapping_preflight_input(raw_pool)
    control = _read(paths["frozen_control_c006"])
    baseline = _read(paths["complete_baseline_c006"])
    mapping = _read(paths["mapping_preflight"])
    cases = {str(row["case_token"]): row for row in mapping["cases"]}
    summary = cases["C006"]
    regenerated_control = complete_core.regenerate_frozen_control_probe(
        deepcopy(projected),
        scan_sha256=str(summary["scan_sha256"]),
        candidate_pool_sha256=str(summary["candidate_pool_sha256"]),
        a0_prediction_sha256=str(summary["a0_prediction_sha256"]),
    )
    regenerated_baseline = complete_core.build_v2_complete_same_surface_edge_probe(
        deepcopy(projected),
        scan_sha256=str(summary["scan_sha256"]),
        candidate_pool_sha256=str(summary["candidate_pool_sha256"]),
        a0_prediction_sha256=str(summary["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(summary["frozen_control_probe_sha256"]),
    )
    if canonical_json_bytes(regenerated_control) != canonical_json_bytes(control):
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "frozen control reproduction failed"
        )
    if canonical_json_bytes(regenerated_baseline) != canonical_json_bytes(baseline):
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "frozen baseline reproduction failed"
        )
    c006 = v2_core._build_case_from_payloads(
        deepcopy(projected),
        case_token="C006",
        scan_sha256=str(summary["scan_sha256"]),
        candidate_pool_sha256=str(summary["candidate_pool_sha256"]),
        a0_prediction_sha256=str(summary["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(summary["frozen_control_probe_sha256"]),
        complete_edge_baseline_probe_sha256=str(summary["complete_edge_baseline_probe_sha256"]),
        frozen_control=regenerated_control,
        complete_edge_baseline=regenerated_baseline,
    )
    all_relations = [dict(row) for row in mapping["frozen_target_baseline_relation_table"]]
    if len(all_relations) != CONFIG.frozen_target_count:
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "frozen target ledger changed"
        )
    relations_c006 = [row for row in all_relations if row["case_token"] == "C006"]
    if len(relations_c006) != CONFIG.c006_target_count:
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "C006 projection changed"
        )
    if c006["frozen_target_baseline_relation_table"] != relations_c006:
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "C006 target rows changed"
        )
    context = v1_core._baseline_context(projected, regenerated_baseline)
    pair_contexts = {
        pair_id: v1_core._pair_context(pair, context)
        for pair_id, pair in sorted(context["pair_by_id"].items())
    }
    baseline_outcomes = [dict(row) for row in mapping["frozen_target_outcome_table"]]
    if len(baseline_outcomes) != CONFIG.frozen_target_count:
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "baseline outcome table changed"
        )
    return FrozenContext(
        projected=projected,
        control=control,
        baseline=baseline,
        mapping=mapping,
        context=context,
        pair_contexts=pair_contexts,
        relations_c006=relations_c006,
        all_relations=all_relations,
        baseline_outcomes=baseline_outcomes,
    )


def _count_bits(bits: np.ndarray) -> int:
    return int(np.sum(_POPCOUNT[np.asarray(bits, dtype=np.uint8)], dtype=np.uint64))


def _bitset_boolean(bits: np.ndarray, point_count: int) -> np.ndarray:
    return np.unpackbits(np.asarray(bits, dtype=np.uint8), bitorder="little")[:point_count].astype(bool)


def _line_distances(points: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> np.ndarray:
    delta = points - anchor
    projection = delta @ axis
    transverse = delta - projection[:, None] * axis
    return np.linalg.norm(transverse, axis=1)


def _distance_to_parent_boxes(
    points: np.ndarray, parent_ids: Sequence[str], parents: Mapping[str, Mapping[str, Any]], voxel_size: float
) -> float:
    if len(points) == 0 or not parent_ids:
        return math.inf
    minimum = math.inf
    for parent_id in parent_ids:
        key = np.asarray(parents[parent_id]["voxel_key"], dtype=np.float64)
        low = key * voxel_size
        high = (key + 1.0) * voxel_size
        displacement = np.maximum(np.maximum(low - points, points - high), 0.0)
        minimum = min(minimum, float(np.min(np.linalg.norm(displacement, axis=1))))
    return minimum


def _candidate_rebuild_matches(commitment: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> None:
    expected = dict(commitment)
    expected.pop("role", None)
    expected.pop("commitment_sha256", None)
    observed = geometry.public_commitment(rebuilt)
    if canonical_json_bytes(expected) != canonical_json_bytes(observed):
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP",
            "post-commit candidate rebuild differs from frozen commitment",
        )


def _target_parent_sets(
    relation: Mapping[str, Any], frozen: FrozenContext
) -> tuple[dict[str, Any], list[str], list[str]]:
    pair_id = str(relation["baseline_local_plane_pair_id"])
    pair = frozen.pair_contexts[pair_id]
    modes = frozen.context["validated"]["mode_by_id"]
    left = sorted({str(modes[mid]["parent_atom_id"]) for mid in pair["left_modes"]})
    right = sorted({str(modes[mid]["parent_atom_id"]) for mid in pair["right_modes"]})
    if not left or not right:
        raise geometry.PGRRStop(
            "PGRR_V1_INVALID_BINDING_OR_EXECUTION_STOP", "frozen pair has empty endpoint support"
        )
    return pair, left, right


def evaluate_handoff(
    *,
    raw_source: Path,
    producer_commitment: Mapping[str, Any],
    rebuilt_result: Mapping[str, Any],
    frozen: FrozenContext,
) -> dict[str, Any]:
    _candidate_rebuild_matches(producer_commitment, rebuilt_result)
    internal = rebuilt_result["_internal"]
    sample: geometry.Sample = internal["sample"]
    field: geometry.NormalField = internal["field"]
    ribbons: Sequence[geometry.Ribbon] = internal["deduplicated"]
    left_edges = internal["left"]
    right_edges = internal["right"]
    validated = frozen.context["validated"]
    parents = validated["parent_by_id"]
    voxel_size = float(validated["parent_voxel_size_mm"])
    parent_by_key = {
        tuple(map(int, row["voxel_key"])): str(parent_id)
        for parent_id, row in parents.items()
    }
    point_parent = np.empty(len(sample.xyz), dtype=object)
    keys = np.floor(sample.xyz / voxel_size).astype(np.int64)
    for index, key in enumerate(keys):
        point_parent[index] = parent_by_key.get(tuple(map(int, key)), "")
    handoff_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    formal_near_misses: list[dict[str, Any]] = []
    complete_subjects: list[dict[str, Any]] = []
    for relation in frozen.relations_c006:
        target_id = str(relation["frozen_target_component_id"])
        pair, left_parents, right_parents = _target_parent_sets(relation, frozen)
        anchor = np.asarray(pair["anchor"], dtype=np.float64)
        axis = np.asarray(pair["axis"], dtype=np.float64)
        tolerance = float(frozen.context["line_tolerance"])
        tube_distances = _line_distances(sample.xyz, anchor, axis)
        tube = tube_distances <= tolerance
        for ribbon_index, ribbon in enumerate(ribbons):
            ribbon_id = producer_commitment["deduplicated_ribbons"][ribbon_index]["ribbon_id"]
            bin_by_vertex = np.full(len(sample.xyz), -1, dtype=np.int16)
            bin_by_vertex[ribbon.member_indices] = ribbon.bin_indices
            first_points = ribbon.member_indices[ribbon.bin_indices == 0]
            last_points = ribbon.member_indices[ribbon.bin_indices == ribbon.bin_count - 1]

            def orientation(
                first_allowed: Sequence[str], last_allowed: Sequence[str], name: str
            ) -> dict[str, Any]:
                first_counts = sorted(
                    (
                        int(np.count_nonzero(point_parent[first_points] == parent_id)),
                        parent_id,
                    )
                    for parent_id in first_allowed
                )
                last_counts = sorted(
                    (
                        int(np.count_nonzero(point_parent[last_points] == parent_id)),
                        parent_id,
                    )
                    for parent_id in last_allowed
                )
                first_pass = [(count, pid) for count, pid in first_counts if count >= CONFIG.minimum_points_per_angular_bin]
                last_pass = [(count, pid) for count, pid in last_counts if count >= CONFIG.minimum_points_per_angular_bin]
                assignments = sorted(
                    (first_pid, last_pid, first_count, last_count)
                    for first_count, first_pid in first_pass
                    for last_count, last_pid in last_pass
                    if first_pid != last_pid
                )
                chosen = assignments[0] if assignments else None
                endpoint_ok = chosen is not None
                selected_first = chosen[0] if chosen else None
                selected_last = chosen[1] if chosen else None
                interior_counts = []
                interior_margins = []
                for angular_bin in range(1, ribbon.bin_count - 1):
                    points = ribbon.member_indices[ribbon.bin_indices == angular_bin]
                    distances = tube_distances[points]
                    interior_counts.append(int(np.count_nonzero(distances <= tolerance)))
                    interior_margins.append(float(tolerance - np.min(distances)))
                interior_ok = all(value > 0 for value in interior_counts)
                edge_member = (bin_by_vertex[left_edges] >= 0) & (bin_by_vertex[right_edges] >= 0)
                edge_delta = np.abs(bin_by_vertex[left_edges] - bin_by_vertex[right_edges])
                adjacent = edge_member & (edge_delta == 1)
                boundary_counts = []
                unconstrained_counts = []
                for boundary in range(ribbon.bin_count - 1):
                    boundary_mask = adjacent & (
                        np.minimum(bin_by_vertex[left_edges], bin_by_vertex[right_edges]) == boundary
                    )
                    unconstrained_counts.append(int(np.count_nonzero(boundary_mask)))
                    if boundary == 0 and selected_first is not None:
                        permitted_left = tube | (point_parent == selected_first)
                        permitted_right = tube | (point_parent == selected_first)
                    elif boundary == ribbon.bin_count - 2 and selected_last is not None:
                        permitted_left = tube | (point_parent == selected_last)
                        permitted_right = tube | (point_parent == selected_last)
                    else:
                        permitted_left = tube
                        permitted_right = tube
                    boundary_counts.append(int(np.count_nonzero(
                        boundary_mask & permitted_left[left_edges] & permitted_right[right_edges]
                    )))
                boundary_ok = all(value > 0 for value in boundary_counts)
                pair_ok = bool(
                    endpoint_ok
                    and selected_first in pair["transverse_parent_ids"]
                    and selected_last in pair["transverse_parent_ids"]
                    and selected_first != selected_last
                )
                return {
                    "orientation": name,
                    "first_parent_id": selected_first,
                    "last_parent_id": selected_last,
                    "first_parent_max_support_count": max((count for count, _pid in first_counts), default=0),
                    "last_parent_max_support_count": max((count for count, _pid in last_counts), default=0),
                    "endpoint_support_ok": endpoint_ok,
                    "interior_tube_ok": interior_ok,
                    "boundary_edge_ok": boundary_ok,
                    "frozen_pair_support_ok": pair_ok,
                    "interior_tube_counts": interior_counts,
                    "interior_tube_margins": interior_margins,
                    "adjacent_boundary_edge_counts": boundary_counts,
                    "unconstrained_boundary_edge_counts": unconstrained_counts,
                    "first_parent_distance_margin": CONFIG.formal_near_miss_density_fraction * field.density_scale - _distance_to_parent_boxes(
                        sample.xyz[first_points], first_allowed, parents, voxel_size
                    ),
                    "last_parent_distance_margin": CONFIG.formal_near_miss_density_fraction * field.density_scale - _distance_to_parent_boxes(
                        sample.xyz[last_points], last_allowed, parents, voxel_size
                    ),
                }

            orientations = [
                orientation(left_parents, right_parents, "FORWARD"),
                orientation(right_parents, left_parents, "SWAPPED"),
            ]
            passing = [
                row for row in orientations
                if row["endpoint_support_ok"] and row["interior_tube_ok"]
                and row["boundary_edge_ok"] and row["frozen_pair_support_ok"]
            ]
            selected = min(
                passing,
                key=lambda row: (
                    str(row["first_parent_id"]), str(row["last_parent_id"]), row["orientation"]
                ),
                default=orientations[0],
            )
            complete = bool(passing)
            failures = [
                name for name in (
                    "endpoint_support_ok", "interior_tube_ok", "boundary_edge_ok", "frozen_pair_support_ok"
                ) if not selected[name]
            ]
            near_code = None
            if not complete and len(failures) == 1:
                failure = failures[0]
                if failure == "endpoint_support_ok" and (
                    selected["first_parent_distance_margin"] >= 0.0
                    or selected["last_parent_distance_margin"] >= 0.0
                ):
                    near_code = "ENDPOINT_DISTANCE_NEAR_MISS"
                elif failure == "interior_tube_ok":
                    zero_bins = [index for index, count in enumerate(selected["interior_tube_counts"]) if count == 0]
                    if len(zero_bins) == 1 and selected["interior_tube_margins"][zero_bins[0]] >= -CONFIG.formal_near_miss_density_fraction * field.density_scale:
                        near_code = "INTERIOR_TUBE_BIN_NEAR_MISS"
                elif failure == "boundary_edge_ok":
                    missing = [index for index, count in enumerate(selected["adjacent_boundary_edge_counts"]) if count == 0]
                    if len(missing) == 1 and selected["unconstrained_boundary_edge_counts"][missing[0]] > 0:
                        near_code = "ADJACENT_BOUNDARY_EDGE_NEAR_MISS"
            row = {
                "target_id": target_id,
                "ribbon_id": ribbon_id,
                "forward": orientations[0],
                "swapped": orientations[1],
                "selected_orientation": selected["orientation"],
                "complete_target_handoff": complete,
                "formal_near_miss_code": near_code,
                "failure_codes": failures,
                "tube_tolerance": tolerance,
                "density_scale": field.density_scale,
            }
            row["handoff_row_sha256"] = sha_json(row)
            handoff_rows.append(row)
            if complete:
                relation_id = "PGRRC_" + sha_json([target_id, ribbon_id, selected["orientation"]])[:20]
                cut_index = (ribbon.bin_count - 1) // 2
                chain = {
                    "relation_id": relation_id,
                    "target_id": target_id,
                    "ribbon_id": ribbon_id,
                    "relation_type": "PGRR_SUPPORT_CHAIN",
                    "node_count": ribbon.bin_count,
                    "edge_count": ribbon.bin_count - 1,
                    "first_parent_id": selected["first_parent_id"],
                    "last_parent_id": selected["last_parent_id"],
                    "central_cut_index": cut_index,
                    "removed_edge_count": 1,
                    "full_connected": True,
                    "cut_connected": False,
                }
                chain["chain_sha256"] = sha_json(chain)
                chain_rows.append(chain)
                complete_subjects.append({"target_id": target_id, "ribbon_id": ribbon_id, "relation_id": relation_id})
            elif near_code is not None:
                formal_near_misses.append({
                    "target_id": target_id,
                    "ribbon_id": ribbon_id,
                    "near_miss_code": near_code,
                    "handoff_row_sha256": row["handoff_row_sha256"],
                })
            if len(complete_subjects) + len(formal_near_misses) > CONFIG.visual_subject_maximum:
                raise geometry.PGRRStop(
                    "PGRR_V1_NONSELECTIVE_REPRESENTATION_ARCHITECTURE_STOP",
                    "129th decisive or formal-near-miss subject encountered",
                )
    if chain_rows:
        decision = "PGRR_V1_FROZEN_INTERFACE_INCOMPATIBLE_STOP"
        interface = {
            "supports_intermediate_nodes_without_semantic_change": False,
            "reason": "FROZEN_COMPONENT_BUILDER_ACCEPTS_ONLY_MODE_IDS_AND_MODE_EDGES",
            "direct_parent_to_parent_collapse_used": False,
        }
    else:
        decision = "PGRR_V1_NO_COMPLETE_FROZEN_HANDOFF_ARCHITECTURE_STOP"
        interface = {
            "supports_intermediate_nodes_without_semantic_change": False,
            "reason": "NOT_REACHED_NO_COMPLETE_CHAIN",
            "direct_parent_to_parent_collapse_used": False,
        }
    return {
        "handoff_rows": sorted(handoff_rows, key=lambda row: (row["target_id"], row["ribbon_id"])),
        "chain_rows": sorted(chain_rows, key=lambda row: row["relation_id"]),
        "complete_subjects": complete_subjects,
        "formal_near_misses": formal_near_misses,
        "interface": interface,
        "decision": decision,
        "baseline_outcomes": frozen.baseline_outcomes,
        "baseline_target_outcome_sha256": sha_json(frozen.baseline_outcomes),
        "aggregate_metrics": {
            "complete_target_handoff_count": len(chain_rows),
            "aggregate_added_pgrr_support_chain_count": 0,
            "target_reached_by_pgrr_support_chain_count": 0,
            "target_side_gate_advanced_count": 0,
            "target_pgrr_attributable_certificate_count": 0,
            "target_novel_pgrr_attributable_certificate_count": 0,
            "mid_arc_cut_connected_chain_count": 0,
        },
    }
