"""Gold-free CIIP-V1 frozen-target attribution probe.

The module consumes the immutable V4 evidence container and reconstructs the
finite H0 merge witnesses without recomputing CIIP.  It then maps those
witnesses through the already-frozen V2 target, tube, graph, side-gate, and
certificate-equivalence machinery.  No persistence cutoff, ranking, tuning,
labels, CAD, or expected-count information is accepted.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as complete_core
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as v1_core
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as v2_core


SCHEMA_VERSION = 1
KIND = "f1_ciip_v1_frozen_target_attribution_probe_v1"
CONSTRUCTION_VERSION = "CIIP_V1_CANONICAL_FINITE_H0_MERGE_WITNESS_V1"
ORIGIN_FAMILY = "CIIP_V1"
EXPECTED_TARGET_COUNT = 4_788


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def record_hash(row: Mapping[str, Any], hash_field: str = "row_sha256") -> str:
    return sha_json({key: value for key, value in row.items() if key != hash_field})


def table_hash(rows: Sequence[Mapping[str, Any]], id_field: str) -> str:
    return sha_json(sorted((dict(row) for row in rows), key=lambda row: str(row[id_field])))


def morton_xyz(value: int) -> tuple[int, int, int]:
    coordinates = [0, 0, 0]
    for bit in range(5):
        for axis in range(3):
            coordinates[axis] |= ((int(value) >> (3 * bit + axis)) & 1) << bit
    return tuple(coordinates)


def locked_grid_transform(source_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Reconstruct only the locked depth-five physical grid transform."""
    xyz = np.asarray(source_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("physical-coordinate reconstruction requires finite Nx3 source")
    minimum = np.min(xyz, axis=0)
    ranges = np.max(xyz, axis=0) - minimum
    diameter = float(np.max(ranges))
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("locked grid diameter is invalid")
    root_side = float(float(diameter * 4.0) / 3.0)
    origin = minimum + ranges / 2.0 - root_side / 2.0
    return origin, float(root_side / 32.0)


def _cell_position(morton: int, origin: np.ndarray, cell_side: float) -> list[float]:
    xyz = np.asarray(morton_xyz(morton), dtype=np.float64)
    return [float(value) for value in origin + (xyz + 0.5) * cell_side]


def _hex(value: Any) -> str:
    if hasattr(value, "tobytes"):
        value = value.tobytes()
    return bytes(value).hex()


def build_merge_witnesses(
    *, field_rows: np.ndarray, adjacency_rows: np.ndarray,
    persistence_rows: np.ndarray, source_xyz: np.ndarray, source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay the locked elder-rule merges and emit one witness per finite row."""
    origin, cell_side = locked_grid_transform(source_xyz)
    witnesses: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    finite_expected = {
        (_hex(row["birth_row_id_digest_128"]), _hex(row["death_row_id_digest_128"])):
        row for row in persistence_rows if int(row["status_u8"]) == 0
    }
    emitted: set[tuple[str, str]] = set()
    for arm in (0, 1):
        fields = np.asarray(field_rows[field_rows["arm_u8"] == arm])
        vertex_count = len(fields)
        if vertex_count == 0:
            raise ValueError("CIIP arm has no field rows")
        edge_f = np.maximum(
            fields["filtration_u16"][adjacency_rows["lower_vertex_u32"]],
            fields["filtration_u16"][adjacency_rows["upper_vertex_u32"]],
        ).astype(np.uint16)
        event_dtype = np.dtype([
            ("f", "<u2"), ("kind", "u1"), ("a", "<u4"), ("b", "<u4"),
            ("c", "u1"), ("d", "u1"), ("payload", "<u4"),
        ])
        events = np.empty(vertex_count + len(adjacency_rows), dtype=event_dtype)
        events[:vertex_count]["f"] = fields["filtration_u16"]
        events[:vertex_count]["kind"] = 0
        events[:vertex_count]["a"] = np.arange(vertex_count, dtype=np.uint32)
        events[:vertex_count]["b"] = 0
        events[:vertex_count]["c"] = 0
        events[:vertex_count]["d"] = 0
        events[:vertex_count]["payload"] = np.arange(vertex_count, dtype=np.uint32)
        edge_events = events[vertex_count:]
        edge_events["f"] = edge_f
        edge_events["kind"] = 1
        edge_events["a"] = adjacency_rows["lower_vertex_u32"]
        edge_events["b"] = adjacency_rows["upper_vertex_u32"]
        edge_events["c"] = adjacency_rows["edge_kind_u8"]
        edge_events["d"] = adjacency_rows["axis_u8"]
        edge_events["payload"] = np.arange(len(adjacency_rows), dtype=np.uint32)
        order = np.argsort(events, order=("f", "kind", "c", "d", "a", "b"), kind="stable")
        parent = np.arange(vertex_count, dtype=np.int32)
        rank = np.zeros(vertex_count, dtype=np.uint8)
        active = np.zeros(vertex_count, dtype=bool)
        owner = np.arange(vertex_count, dtype=np.uint32)

        def find(value: int) -> int:
            root = value
            while int(parent[root]) != root:
                root = int(parent[root])
            while int(parent[value]) != value:
                nxt = int(parent[value]); parent[value] = root; value = nxt
            return root

        for event_index in order:
            event = events[int(event_index)]
            if int(event["kind"]) == 0:
                active[int(event["payload"])] = True
                continue
            edge_index = int(event["payload"])
            edge = adjacency_rows[edge_index]
            left = int(edge["lower_vertex_u32"]); right = int(edge["upper_vertex_u32"])
            if not active[left] or not active[right]:
                raise ValueError("edge appeared before endpoint activation")
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                continue
            left_owner, right_owner = int(owner[left_root]), int(owner[right_root])
            left_key = (int(fields["filtration_u16"][left_owner]), left_owner)
            right_key = (int(fields["filtration_u16"][right_owner]), right_owner)
            if left_key <= right_key:
                survivor, loser, loser_root = left_owner, right_owner, right_root
                dying_vertex, surviving_vertex = right, left
            else:
                survivor, loser, loser_root = right_owner, left_owner, left_root
                dying_vertex, surviving_vertex = left, right
            birth_id = _hex(fields["row_id_digest_128"][loser])
            death_id = _hex(edge["edge_id_digest_128"])
            key = (birth_id, death_id)
            persistence = finite_expected.get(key)
            if persistence is None:
                raise ValueError("replayed finite merge is absent from persistence table")
            dying = fields[dying_vertex]; surviving_row = fields[surviving_vertex]
            dying_position = _cell_position(int(dying["morton_u64"]), origin, cell_side)
            surviving_position = _cell_position(
                int(surviving_row["morton_u64"]), origin, cell_side)
            death_position = [
                0.5 * (left_value + right_value)
                for left_value, right_value in zip(dying_position, surviving_position)
            ]
            witness_id = "CIIPW_" + sha_json({
                "source_sha256": source_sha256,
                "interval_id": _hex(persistence["interval_id_digest_128"]),
                "construction_version": CONSTRUCTION_VERSION,
            })[:24]
            row = {
                "witness_id": witness_id,
                "source_sha256": source_sha256,
                "ciip_interval_or_merge_id": _hex(persistence["interval_id_digest_128"]),
                "birth_filtration_value": int(persistence["birth_f_u16"]),
                "death_filtration_value": int(persistence["death_f_or_essential_u16"]),
                "persistence_value": int(persistence["lifetime_u16"]),
                "dying_component_id": birth_id,
                "surviving_component_id": _hex(fields["row_id_digest_128"][survivor]),
                "dying_representative_cell": {
                    "row_id": _hex(dying["row_id_digest_128"]),
                    "arm": arm, "scale": int(dying["scale_u8"]),
                    "morton": int(dying["morton_u64"]),
                },
                "surviving_representative_cell": {
                    "row_id": _hex(surviving_row["row_id_digest_128"]),
                    "arm": arm, "scale": int(surviving_row["scale_u8"]),
                    "morton": int(surviving_row["morton_u64"]),
                },
                "death_cell": {
                    "edge_id": death_id,
                    "edge_kind": int(edge["edge_kind_u8"]),
                    "axis": int(edge["axis_u8"]),
                    "lower_vertex": left, "upper_vertex": right,
                },
                "dying_representative_position": dying_position,
                "surviving_representative_position": surviving_position,
                "death_position": death_position,
                "construction_version": CONSTRUCTION_VERSION,
            }
            row["row_sha256"] = record_hash(row)
            witnesses.append(row); emitted.add(key)
            if rank[left_root] < rank[right_root] or (
                rank[left_root] == rank[right_root] and right_root < left_root
            ):
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1
            owner[left_root] = survivor
    if emitted != set(finite_expected):
        missing = sorted(set(finite_expected) - emitted)
        for birth_id, death_id in missing:
            row = {
                "ciip_interval_or_merge_id": _hex(finite_expected[(birth_id, death_id)]["interval_id_digest_128"]),
                "birth_row_id": birth_id, "death_row_id": death_id,
                "reason": "TWO_DISTINCT_PREDECESSOR_COMPONENTS_NOT_RECONSTRUCTED",
            }
            row["row_sha256"] = record_hash(row)
            exceptions.append(row)
    witnesses.sort(key=lambda row: row["witness_id"])
    exceptions.sort(key=lambda row: row["ciip_interval_or_merge_id"])
    if len(witnesses) + len(exceptions) != len(finite_expected):
        raise ValueError("finite merge accounting mismatch")
    return witnesses, exceptions


def _position_parent_id(
    position: Sequence[float], *, voxel_size: float,
    parent_by_key: Mapping[tuple[int, int, int], str],
) -> str | None:
    key = tuple(int(math.floor(float(value) / voxel_size)) for value in position)
    return parent_by_key.get(key)


def map_witnesses_to_targets(
    *, witnesses: Sequence[Mapping[str, Any]], context: Mapping[str, Any],
    pair_contexts: Mapping[str, Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]], source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = context["validated"]["parent_by_id"]
    modes = context["validated"]["mode_by_id"]
    voxel_size = float(context["validated"]["parent_voxel_size_mm"])
    parent_by_key = {
        tuple(map(int, row["voxel_key"])): str(parent_id)
        for parent_id, row in parents.items()
    }
    modes_by_parent: dict[str, list[str]] = defaultdict(list)
    for mode_id, mode in modes.items():
        modes_by_parent[str(mode["parent_atom_id"])].append(str(mode_id))
    baseline_signatures = {
        (str(pc["pair_id"]), tuple(sorted(map(str, edge["mode_ids"]))))
        for pc in pair_contexts.values() for edge in pc["baseline_edges"]
    }
    relations_by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for relation in relations:
        relations_by_pair[str(relation["baseline_local_plane_pair_id"])].append(relation)
    relation_rows: list[dict[str, Any]] = []
    rungs: list[dict[str, Any]] = []
    for witness in witnesses:
        dying_parent = _position_parent_id(
            witness["dying_representative_position"], voxel_size=voxel_size,
            parent_by_key=parent_by_key)
        surviving_parent = _position_parent_id(
            witness["surviving_representative_position"], voxel_size=voxel_size,
            parent_by_key=parent_by_key)
        if dying_parent is None or surviving_parent is None or dying_parent == surviving_parent:
            continue
        death = np.asarray(witness["death_position"], dtype=np.float64)
        for pair_id, pair_relations in sorted(relations_by_pair.items()):
            pair_context = pair_contexts.get(pair_id)
            if pair_context is None:
                continue
            if lpf._line_distance(death, pair_context["anchor"], pair_context["axis"]) > float(
                context["line_tolerance"]
            ) + 1e-12:
                continue
            fragment_ids = tuple(map(str, pair_context["fragment_ids"]))
            dying_by_fragment = defaultdict(list)
            surviving_by_fragment = defaultdict(list)
            for mode_id in modes_by_parent.get(dying_parent, ()): 
                fragment_id = context["fragment_of_mode"].get(mode_id)
                if fragment_id is not None:
                    dying_by_fragment[str(fragment_id)].append(mode_id)
            for mode_id in modes_by_parent.get(surviving_parent, ()): 
                fragment_id = context["fragment_of_mode"].get(mode_id)
                if fragment_id is not None:
                    surviving_by_fragment[str(fragment_id)].append(mode_id)
            orientations = []
            if dying_by_fragment[fragment_ids[0]] and surviving_by_fragment[fragment_ids[1]]:
                orientations.append((
                    min(dying_by_fragment[fragment_ids[0]]),
                    min(surviving_by_fragment[fragment_ids[1]]),
                    dying_parent, surviving_parent,
                ))
            if dying_by_fragment[fragment_ids[1]] and surviving_by_fragment[fragment_ids[0]]:
                orientations.append((
                    min(surviving_by_fragment[fragment_ids[0]]),
                    min(dying_by_fragment[fragment_ids[1]]),
                    surviving_parent, dying_parent,
                ))
            for left_mode, right_mode, left_parent, right_parent in orientations:
                if (pair_id, tuple(sorted((left_mode, right_mode)))) in baseline_signatures:
                    continue
                if left_mode not in pair_context["supported_modes"] or right_mode not in pair_context["supported_modes"]:
                    continue
                for relation in pair_relations:
                    target_id = str(relation["frozen_target_component_id"])
                    if not bool(relation["target_parent_in_frozen_transverse_set"]):
                        continue
                    relation_id = "CIIPREL_" + sha_json({
                        "source_sha256": source_sha256, "target_id": target_id,
                        "witness_id": witness["witness_id"],
                    })[:24]
                    relation_row = {
                        "relation_id": relation_id,
                        "source_sha256": source_sha256,
                        "target_id": target_id,
                        "witness_id": str(witness["witness_id"]),
                        "baseline_local_plane_pair_id": pair_id,
                        "baseline_fragment_ids": list(fragment_ids),
                        "mode_ids": [left_mode, right_mode],
                        "parent_atom_ids": [left_parent, right_parent],
                        "death_cell": deepcopy(witness["death_cell"]),
                        "source_cross_bound": True,
                        "representatives_in_opposite_frozen_side_regions": True,
                        "death_cell_in_frozen_target_tube": True,
                        "preexisting_structural_validity_checks_accept": True,
                    }
                    relation_row["row_sha256"] = record_hash(relation_row)
                    rung_id = "CIIPR_" + sha_json({
                        "source_sha256": source_sha256, "target_id": target_id,
                        "witness_id": witness["witness_id"],
                        "endpoint_cells": [witness["dying_representative_cell"], witness["surviving_representative_cell"]],
                        "death_cell": witness["death_cell"],
                        "construction_version": CONSTRUCTION_VERSION,
                    })[:24]
                    parent_positions = [
                        lpf._vector(parents[parent_id]["centroid"], "parent centroid", allow_zero=True)
                        for parent_id in (left_parent, right_parent)
                    ]
                    projections = [float(np.dot(value, pair_context["axis"])) for value in parent_positions]
                    rung = {
                        "mode_edge_id": rung_id,
                        "baseline_local_plane_pair_id": pair_id,
                        "baseline_fragment_ids": list(fragment_ids),
                        "mode_ids": [left_mode, right_mode],
                        "parent_atom_ids": [left_parent, right_parent],
                        "parent_voxel_keys": [list(parents[value]["voxel_key"]) for value in (left_parent, right_parent)],
                        "edge_kind": "CROSS_MODE_ADJACENT_PARENT",
                        "parent_voxel_chebyshev_distance": max(abs(int(a) - int(b)) for a, b in zip(
                            parents[left_parent]["voxel_key"], parents[right_parent]["voxel_key"])),
                        "parent_centroid_line_distances_mm": [round(lpf._line_distance(
                            value, pair_context["anchor"], pair_context["axis"]), 6) for value in parent_positions],
                        "axial_parent_separation_mm": abs(projections[1] - projections[0]),
                        "axial_midpoint_projection_mm": 0.5 * (projections[0] + projections[1]),
                        "line_tube_tolerance_mm": float(context["line_tolerance"]),
                        "axial_bin_width_mm": float(context["bin_width"]),
                        "origin_family": ORIGIN_FAMILY,
                        "origin_witness_id": str(witness["witness_id"]),
                        "origin_target_id": target_id,
                        "origin_relation_id": relation_id,
                    }
                    rung["rung_record_sha256"] = record_hash(rung, "rung_record_sha256")
                    relation_rows.append(relation_row); rungs.append(rung)
    relation_rows.sort(key=lambda row: row["relation_id"])
    rungs.sort(key=lambda row: row["mode_edge_id"])
    if len({row["relation_id"] for row in relation_rows}) != len(relation_rows):
        raise ValueError("duplicate CIIP witness-target relation")
    if len({row["mode_edge_id"] for row in rungs}) != len(rungs):
        raise ValueError("duplicate CIIP added rung")
    return relation_rows, rungs


def evaluate_target_population(
    *, relations: Sequence[Mapping[str, Any]], baseline_case_payloads: Mapping[str, Mapping[str, Any]],
    c006_mapping_payload: Mapping[str, Any], c006_context: Mapping[str, Any],
    c006_pair_contexts: Mapping[str, Mapping[str, Any]], c006_control: Mapping[str, Any],
    c006_rungs: Sequence[Mapping[str, Any]], witness_relations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate the exact augmented and all-CIIP-edge ablation states."""
    replacement_pairs, replacement_meta, _before = v2_core._build_replacement_lane(
        context=c006_context, pair_contexts=c006_pair_contexts,
        added_rungs=c006_rungs, frozen_control=c006_control,
        mapping_payload=c006_mapping_payload,
    )
    attributions = v2_core._ordinary_attribution(
        replacement_meta=replacement_meta, pair_contexts=c006_pair_contexts,
        frozen_control=c006_control, mapping_payload=c006_mapping_payload,
    )
    attribution_by_component = {
        str(row["local_transition_component_id"]): row for row in attributions
    }
    witness_ids_by_target: dict[str, list[str]] = defaultdict(list)
    rung_ids_by_target: dict[str, list[str]] = defaultdict(list)
    for row in witness_relations:
        witness_ids_by_target[str(row["target_id"])].append(str(row["witness_id"]))
    for row in c006_rungs:
        rung_ids_by_target[str(row["origin_target_id"])].append(str(row["mode_edge_id"]))
    baseline_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for case_token, payload in baseline_case_payloads.items():
        for item in payload["complete_edge_before_component_table"]:
            baseline_records[(case_token, str(item["component_record"]["local_transition_component_id"]))] = item["component_record"]
    c006_replacement_by_target: dict[str, Mapping[str, Any]] = {}
    for relation in c006_mapping_payload["frozen_target_baseline_relation_table"]:
        if relation["target_baseline_relation_status"] != v2_core.RELATION_COMPONENT_UNIQUE:
            continue
        pair_id = str(relation["baseline_local_plane_pair_id"])
        target_modes = set(map(str, relation["frozen_target_mode_ids"]))
        candidates = [meta for meta in replacement_meta.values() if meta["pair_context"]["pair_id"] == pair_id and target_modes <= set(meta["mode_ids"])]
        if len(candidates) != 1:
            raise ValueError("target replacement image is not unique")
        c006_replacement_by_target[str(relation["frozen_target_component_id"])] = candidates[0]
    rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    for relation in sorted(relations, key=lambda row: (str(row["case_token"]), str(row["frozen_target_component_id"]))):
        case_token = str(relation["case_token"]); target_id = str(relation["frozen_target_component_id"])
        baseline_component_id = relation.get("complete_edge_baseline_component_id")
        baseline = baseline_records.get((case_token, str(baseline_component_id))) if baseline_component_id else None
        baseline_valid = bool(baseline and baseline["geometry_certificate_valid"])
        reached = bool(rung_ids_by_target.get(target_id))
        if case_token == "C006" and relation["target_baseline_relation_status"] == v2_core.RELATION_COMPONENT_UNIQUE:
            meta = c006_replacement_by_target[target_id]
            augmented = meta["record"]
            augmented_id = str(augmented["local_transition_component_id"])
            augmented_valid = bool(augmented["geometry_certificate_valid"])
            side_ablation = bool(baseline and baseline["geometry_gates"]["minimum_left_side_bins"] and baseline["geometry_gates"]["minimum_right_side_bins"])
            side_augmented = bool(augmented["geometry_gates"]["minimum_left_side_bins"] and augmented["geometry_gates"]["minimum_right_side_bins"])
            side_advanced = bool(reached and not side_ablation and side_augmented)
            attribution = attribution_by_component[augmented_id]
            attributable = bool(reached and augmented_valid and not baseline_valid and side_advanced)
            novel = bool(attributable and attribution["novel_rung_attributable"])
            equivalent = list(attribution["complete_edge_equivalent_certificate_ids"])
        else:
            augmented_id = str(baseline_component_id) if baseline_component_id else None
            augmented_valid = baseline_valid
            side_ablation = bool(baseline and baseline["geometry_gates"]["minimum_left_side_bins"] and baseline["geometry_gates"]["minimum_right_side_bins"])
            side_augmented = side_ablation
            side_advanced = attributable = novel = False
            equivalent = [str(baseline_component_id)] if baseline_valid and baseline_component_id else []
        ablation_certificates = [str(baseline_component_id)] if baseline_valid and baseline_component_id else []
        augmented_certificates = [augmented_id] if augmented_valid and augmented_id else []
        attributable_ids = augmented_certificates if attributable else []
        novel_ids = augmented_certificates if novel else []
        row = {
            "target_id": target_id,
            "reached_by_ciip": reached,
            "ciip_witness_ids": sorted(set(witness_ids_by_target.get(target_id, ()))),
            "ciip_rung_ids": sorted(set(rung_ids_by_target.get(target_id, ()))),
            "side_gate_ablation_state": side_ablation,
            "side_gate_augmented_state": side_augmented,
            "certificate_ids_ablation": ablation_certificates,
            "certificate_ids_augmented": augmented_certificates,
            "ciip_attributable_certificate_ids": attributable_ids,
            "novel_ciip_attributable_certificate_ids": novel_ids,
            "equivalent_baseline_certificate_ids": sorted(set(equivalent)),
        }
        row["row_sha256"] = record_hash(row)
        rows.append(row)
        if attributable_ids:
            cert = {
                "target_id": target_id,
                "augmented_certificate_ids": attributable_ids,
                "ablation_certificate_ids": ablation_certificates,
                "derivation_ciip_rung_ids": row["ciip_rung_ids"],
                "side_gate_advanced": side_advanced,
                "novel": novel,
                "equivalent_baseline_certificate_ids": row["equivalent_baseline_certificate_ids"],
            }
            cert["row_sha256"] = record_hash(cert)
            certificate_rows.append(cert)
    if len(rows) != EXPECTED_TARGET_COUNT or len({row["target_id"] for row in rows}) != EXPECTED_TARGET_COUNT:
        raise ValueError("exact frozen target population was not preserved")
    return rows, certificate_rows, replacement_pairs


def aggregate_metrics(
    *, persistence_rows: np.ndarray, witnesses: Sequence[Mapping[str, Any]],
    exceptions: Sequence[Mapping[str, Any]], witness_relations: Sequence[Mapping[str, Any]],
    rungs: Sequence[Mapping[str, Any]], target_rows: Sequence[Mapping[str, Any]],
    baseline_hash_reproduction_ok: bool, producer_validator_consensus_ok: bool,
) -> dict[str, Any]:
    return {
        "finite_h0_merge_count": int(np.count_nonzero(persistence_rows["status_u8"] == 0)),
        "localizable_ciip_witness_count": len(witnesses),
        "nonlocalizable_ciip_merge_count": len(exceptions),
        "ciip_witness_target_relation_count": len(witness_relations),
        "aggregate_added_ciip_rung_count": len(rungs),
        "target_reached_by_ciip_rung_count": sum(bool(row["reached_by_ciip"]) for row in target_rows),
        "target_side_gate_advanced_count": sum(
            not bool(row["side_gate_ablation_state"]) and bool(row["side_gate_augmented_state"])
            for row in target_rows),
        "target_ciip_attributable_certificate_count": sum(bool(row["ciip_attributable_certificate_ids"]) for row in target_rows),
        "target_novel_ciip_attributable_certificate_count": sum(bool(row["novel_ciip_attributable_certificate_ids"]) for row in target_rows),
        "target_certificate_equivalent_to_frozen_baseline_count": sum(bool(row["equivalent_baseline_certificate_ids"]) for row in target_rows),
        "untouched_target_count": sum(not bool(row["reached_by_ciip"]) for row in target_rows),
        "baseline_hash_reproduction_ok": bool(baseline_hash_reproduction_ok),
        "producer_validator_consensus_ok": bool(producer_validator_consensus_ok),
    }


def decision_from_metrics(metrics: Mapping[str, Any]) -> str:
    scientific = (
        "aggregate_added_ciip_rung_count", "target_reached_by_ciip_rung_count",
        "target_side_gate_advanced_count", "target_novel_ciip_attributable_certificate_count",
    )
    if not metrics["baseline_hash_reproduction_ok"] or not metrics["producer_validator_consensus_ok"]:
        return "INVALID_CIIP_V1_ATTRIBUTION_TRANCHE_STOP"
    if any(int(metrics[key]) <= 0 for key in scientific):
        return "CIIP_V1_NO_NOVEL_TARGET_ATTRIBUTION_ARCHITECTURE_STOP"
    return "CIIP_V1_FROZEN_TARGET_ATTRIBUTION_ARCHITECTURE_SURVIVES"


__all__ = [
    "CONSTRUCTION_VERSION", "EXPECTED_TARGET_COUNT", "KIND", "ORIGIN_FAMILY",
    "SCHEMA_VERSION", "aggregate_metrics", "build_merge_witnesses",
    "canonical_json_bytes", "decision_from_metrics", "evaluate_target_population",
    "locked_grid_transform", "map_witnesses_to_targets", "record_hash", "sha_json",
    "table_hash",
]
