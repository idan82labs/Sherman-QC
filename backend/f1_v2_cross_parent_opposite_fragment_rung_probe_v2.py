"""Schema-2 mapping-only preflight for the cross-parent rung experiment.

No rung is constructed here.  The core regenerates and hash-checks frozen LPF
V2 and the complete-SAME_SURFACE baseline, preserves every frozen target, and
records total fragment-pair lineage plus the deliberately partial fixed-tube
component image authorized by the mapping amendment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from backend import f1_local_plane_fragment_probe as lpf
from backend import f1_v2_complete_same_surface_edge_probe as complete_core
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as v1_core


PRIMITIVE_PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V2"
PROBE_SCHEMA_VERSION = 2
PROBE_KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_v2"
MAPPING_PREFLIGHT_KIND = "f1_v2_cross_parent_rung_target_baseline_relation_preflight_v1"
TARGET_BASELINE_RELATION_RULE_ID = "EXACT_FRAGMENT_PAIR_LINEAGE_PARTIAL_COMPONENT_IMAGE_V1"
BASELINE_PROFILE = complete_core.PRIMITIVE_PROFILE
RUNG_RULE_ID = "COMPLETE_ADJACENT_PARENT_OPPOSITE_FRAGMENT_NO_TOPK_V1"
CONTROL_RUNG_RULE_ID = "INHERITED_SAME_PARENT_CROSS_MODE_LOCAL_ONLY_V1"
INPUT_PROJECTION_PROFILE = "V2_CROSS_PARENT_RUNG_MAPPING_INPUT_PROJECTION_V2"
ALLOWED_INPUT_FIELDS = tuple(complete_core.ALLOWED_EDGE_PROBE_INPUT_FIELDS)

RELATION_COMPONENT_UNIQUE = "PAIR_UNIQUE_COMPONENT_UNIQUE"
RELATION_COMPONENT_NONE_OUTSIDE = (
    "PAIR_UNIQUE_COMPONENT_NONE_PARENT_OUTSIDE_FROZEN_TUBE"
)
TARGET_BASELINE_RELATION_STATUS = frozenset(
    {RELATION_COMPONENT_UNIQUE, RELATION_COMPONENT_NONE_OUTSIDE}
)
OUTCOME_COMPONENT_PENDING = "TARGET_COMPONENT_IMAGE_PRESENT_RUNG_EXECUTION_NOT_AUTHORIZED"
OUTCOME_STRUCTURALLY_UNREACHABLE = (
    "TARGET_STRUCTURALLY_UNREACHABLE_PARENT_OUTSIDE_FROZEN_PAIR_TUBE"
)
MAX_TARGET_BASELINE_RELATION_ROWS = 4_788
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3

FROZEN_TARGET_BASELINE_RELATION_FIELDS = {
    "case_token", "frozen_target_component_id", "frozen_target_pair_id",
    "frozen_target_parent_id", "frozen_target_mode_ids",
    "frozen_target_mode_ids_sha256", "frozen_target_sha256",
    "baseline_mode_fragment_rows", "baseline_mode_fragment_rows_sha256",
    "baseline_fragment_ids", "baseline_fragment_ids_sha256",
    "baseline_pair_mapping_count", "baseline_local_plane_pair_id",
    "baseline_pair_sha256", "baseline_transverse_parent_count",
    "baseline_transverse_parent_ids_sha256", "target_parent_line_distance_mm",
    "line_tube_tolerance_mm", "target_parent_in_frozen_transverse_set",
    "baseline_component_mapping_count", "complete_edge_baseline_component_id",
    "target_baseline_relation_status", "target_baseline_relation_sha256",
}

FROZEN_TARGET_OUTCOME_FIELDS = {
    "case_token", "frozen_target_component_id", "frozen_target_mode_ids_sha256",
    "baseline_local_plane_pair_id", "complete_edge_baseline_component_id",
    "replacement_component_id", "target_baseline_relation_status",
    "causal_added_cross_parent_rung_count",
    "causal_added_cross_parent_rung_ids_sha256", "support_parent_count_before",
    "support_parent_count_after", "occupied_bin_count_before",
    "occupied_bin_count_after", "left_side_bin_count_before",
    "left_side_bin_count_after", "right_side_bin_count_before",
    "right_side_bin_count_after", "transition_rung_bin_count_before",
    "transition_rung_bin_count_after", "geometry_gate_depth_before",
    "geometry_gate_depth_after", "first_rejection_code_before",
    "first_rejection_code_after", "geometry_certificate_valid_after",
    "target_reached_by_added_rung", "target_side_gate_advanced",
    "target_novel_rung_attributable", "target_outcome_code",
    "target_outcome_sha256",
}

CASE_ATTESTATION_FIELDS = {
    "mapping_preflight_kind", "target_baseline_relation_rule_id",
    "frozen_control_reproduced", "complete_edge_baseline_reproduced",
    "frozen_target_rows_regenerated_from_control", "frozen_target_rows_byte_identical",
    "frozen_target_id_sets_exact", "frozen_target_mode_hash_sets_exact",
    "every_target_mode_has_one_baseline_fragment",
    "every_target_maps_to_exactly_two_fragments", "every_target_has_exactly_one_pair_lineage",
    "every_in_tube_target_has_one_component_image",
    "every_out_of_tube_target_has_zero_component_images",
    "no_unrecognised_relation_status_exists", "relation_table_untruncated",
    "outcome_table_complete", "threshold_hash_unchanged",
    "all_pair_membership_row_count_matches_baseline",
    "pair_line_tube_membership_unchanged", "no_rungs_constructed",
    "ordinary_batch_authorized", "gold_open_authorized",
    "case_mapping_attestation_sha256",
}

TOP_LEVEL_FIELDS = {
    "schema_version", "kind", "profile", "mapping_preflight_kind",
    "target_baseline_relation_rule_id", "research_only", "inference_safe",
    "prediction_mutated", "birth_columns_enabled", "ownership_consumed",
    "solver_consumed", "truth_inputs_consumed", "source_points_consumed",
    "source_points_serialized", "ordinary_batch_authorized",
    "designated_replays_authorized", "full_synthetic_resource_run_authorized",
    "compact_bundle_authorized", "gold_open_authorized", "case_token",
    "scan_sha256", "candidate_pool_sha256", "a0_prediction_sha256",
    "frozen_control_probe_sha256", "complete_edge_baseline_probe_sha256",
    "input_table_hashes", "baseline_table_hashes", "thresholds",
    "hard_caps", "case_counters", "case_mapping_attestation",
    "frozen_target_table_sha256", "frozen_target_projection",
    "all_pair_membership_table", "all_pair_membership_table_sha256",
    "thresholds_sha256", "frozen_target_baseline_relation_table_sha256",
    "frozen_target_outcome_table_sha256", "frozen_target_baseline_relation_table",
    "frozen_target_outcome_table", "engineering_profile_sha256",
    "inference_semantic_sha256",
}

HARD_CAPS = {
    "input_working_point_cap": 200_000,
    "transition_parent_cap": 4_000,
    "transition_mode_cap": 16_000,
    "modes_per_parent_cap": 4,
    "parent_plus_mode_record_cap": 20_000,
    "target_baseline_relation_row_cap": MAX_TARGET_BASELINE_RELATION_ROWS,
}

ALL_PAIR_MEMBERSHIP_FIELDS = {
    "baseline_local_plane_pair_id", "transverse_parent_ids",
    "transverse_parent_ids_sha256",
}

TARGET_ELIGIBLE_PAIR_FIELDS = {
    "baseline_local_plane_pair_id", "frozen_target_component_ids",
    "frozen_target_component_ids_sha256",
}
COMPONENT_TABLE_FIELDS = {
    "baseline_local_plane_pair_id", "component_record",
}
ORDINARY_V2_CRITERION_KEYS = (
    "profile_identity_valid", "input_projection_valid",
    "frozen_control_reproduced", "complete_edge_baseline_reproduced",
    "mapping_case_reproduced", "target_pair_population_exact",
    "rung_parents_distinct", "rung_parent_voxels_adjacent",
    "rungs_join_opposite_fixed_fragments", "rungs_use_target_eligible_pairs_only",
    "rung_parents_inside_frozen_tube", "orphan_modes_excluded",
    "axial_rule_exact", "no_pair_births", "no_fragment_or_orphan_births",
    "present_target_replacement_mapping_unique",
    "structural_negative_outcomes_forced", "one_parent_components_zero_span",
    "global_deletion_counterfactual_executed_and_valid",
    "all_structural_caps_satisfied", "safety_flags_valid",
)
DELETION_EVIDENCE_FIELDS = {
    "counterfactual_profile", "executed_independent_rebuild",
    "deleted_added_cross_parent_rung_count", "actual_effective_edge_table_sha256",
    "deleted_added_cross_parent_rung_table",
    "deleted_added_cross_parent_rung_table_sha256",
    "expected_effective_edge_table_sha256", "actual_fragment_table_sha256",
    "expected_fragment_table_sha256", "actual_orphan_table_sha256",
    "expected_orphan_table_sha256", "actual_pair_records_sha256",
    "expected_pair_records_sha256", "actual_all_pair_membership_table_sha256",
    "expected_all_pair_membership_table_sha256", "actual_component_count",
    "expected_component_count", "actual_certificate_count",
    "expected_certificate_count", "all_required_equalities",
    "rebuilt_effective_edge_table", "frozen_effective_edge_table",
    "rebuilt_fragment_table", "frozen_fragment_table",
    "rebuilt_orphan_table", "frozen_orphan_table",
    "rebuilt_pair_records", "frozen_pair_records",
    "rebuilt_all_pair_membership_table", "frozen_all_pair_membership_table",
    "rebuilt_component_table", "frozen_component_table",
    "rebuilt_geometry_gate_table", "frozen_geometry_gate_table",
    "deletion_evidence_sha256",
}
ORDINARY_TOP_LEVEL_FIELDS = {
    "schema_version", "kind", "profile", "artifact_mode", "research_only",
    "inference_safe", "prediction_mutated", "birth_columns_enabled",
    "ownership_consumed", "solver_consumed", "truth_inputs_consumed",
    "source_points_consumed", "source_points_serialized",
    "designated_replays_authorized", "full_synthetic_resource_run_authorized",
    "gold_open_authorized", "case_token", "scan_sha256",
    "candidate_pool_sha256", "a0_prediction_sha256",
    "frozen_control_probe_sha256", "complete_edge_baseline_probe_sha256",
    "mapping_case_sha256", "input_table_hashes", "baseline_table_hashes",
    "thresholds", "hard_caps", "target_eligible_pair_table",
    "frozen_target_projection", "frozen_target_baseline_relation_table",
    "all_pair_membership_table", "baseline_local_plane_fragment_table",
    "baseline_local_plane_orphan_table", "complete_edge_before_component_table",
    "added_cross_parent_rung_table", "replacement_component_table",
    "component_rung_attribution_table", "frozen_target_outcome_table",
    "certificate_novelty_table", "local_plane_pair_records",
    "global_deletion_counterfactual", "output_table_hashes", "case_counters",
    "ordinary_v2_criteria", "all_ordinary_v2_criteria_valid", "packet_status",
    "engineering_profile_sha256", "inference_semantic_sha256",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _table_sha(rows: Sequence[Mapping[str, Any]], id_field: str) -> str:
    return _sha(sorted((dict(row) for row in rows), key=lambda row: str(row[id_field])))


def _record_hash(row: Mapping[str, Any], field: str) -> str:
    return _sha({key: value for key, value in row.items() if key != field})


def _require_sha(label: str, value: Any) -> str:
    return lpf._validate_sha256(label, value)


def project_mapping_preflight_input(raw_pool: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in ALLOWED_INPUT_FIELDS if field not in raw_pool]
    if missing:
        raise ValueError(f"V2 mapping input projection missing fields: {missing}")
    projected = {field: deepcopy(raw_pool[field]) for field in ALLOWED_INPUT_FIELDS}
    if tuple(projected) != ALLOWED_INPUT_FIELDS:
        raise AssertionError("V2 mapping projection order mismatch")
    return projected


def build_all_pair_membership_rows(
    pair_contexts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Canonical complete pair-to-frozen-transverse-parent relation."""
    rows = []
    for pair_id, pair_context in sorted(pair_contexts.items()):
        parent_ids = sorted(map(str, pair_context["transverse_parent_ids"]))
        rows.append({
            "baseline_local_plane_pair_id": str(pair_id),
            "transverse_parent_ids": parent_ids,
            "transverse_parent_ids_sha256": _sha(parent_ids),
        })
    return rows


def independently_regenerate_all_pair_membership_rows(
    projected_input: Mapping[str, Any],
    complete_edge_baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct every pair context in a separate mapping-only call."""
    independent_context = v1_core._baseline_context(
        deepcopy(dict(projected_input)), deepcopy(dict(complete_edge_baseline)))
    independent_pair_contexts = {
        pair_id: v1_core._pair_context(pair, independent_context)
        for pair_id, pair in sorted(independent_context["pair_by_id"].items())
    }
    return build_all_pair_membership_rows(independent_pair_contexts)


def frozen_target_projection(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "case_token", "frozen_target_component_id", "frozen_target_pair_id",
        "frozen_target_parent_id", "frozen_target_mode_ids",
        "frozen_target_mode_ids_sha256", "frozen_target_sha256",
    )
    result = [{field: deepcopy(row[field]) for field in fields} for row in rows]
    result.sort(key=lambda row: (row["case_token"], row["frozen_target_component_id"]))
    return result


def derive_target_preservation_predicates(
    relation_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    relation_projection = frozen_target_projection(relation_rows)
    frozen_projection = frozen_target_projection(frozen_rows)
    return {
        "frozen_target_rows_byte_identical": relation_projection == frozen_projection,
        "frozen_target_id_sets_exact": {
            str(row["frozen_target_component_id"]) for row in relation_projection
        } == {
            str(row["frozen_target_component_id"]) for row in frozen_projection
        },
        "frozen_target_mode_hash_sets_exact": {
            str(row["frozen_target_mode_ids_sha256"]) for row in relation_projection
        } == {
            str(row["frozen_target_mode_ids_sha256"]) for row in frozen_projection
        },
    }


def _exact_pair_lineage(
    target: Mapping[str, Any], fragment_of_mode: Mapping[str, str],
    baseline_pairs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[list[str]], list[str]]:
    mode_fragment_rows = [
        [str(mode_id), fragment_of_mode.get(str(mode_id))]
        for mode_id in target["frozen_target_mode_ids"]
    ]
    if any(fragment_id is None for _mode_id, fragment_id in mode_fragment_rows):
        raise ValueError("TARGET_MODE_HAS_NO_BASELINE_FRAGMENT")
    fragment_ids = sorted({str(fragment_id) for _mode_id, fragment_id in mode_fragment_rows})
    if len(fragment_ids) != 2:
        raise ValueError("TARGET_DOES_NOT_MAP_TO_EXACTLY_TWO_FRAGMENTS")
    matches = [dict(pair) for pair in baseline_pairs
               if sorted(map(str, pair["fragment_ids"])) == fragment_ids]
    if len(matches) != 1:
        raise ValueError("TARGET_PAIR_LINEAGE_NOT_UNIQUE")
    return matches[0], mode_fragment_rows, fragment_ids


def _relation_row(
    *, target: Mapping[str, Any], pair: Mapping[str, Any],
    mode_fragment_rows: list[list[str]], fragment_ids: list[str],
    pair_context: Mapping[str, Any], context: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    target_modes = set(map(str, target["frozen_target_mode_ids"]))
    target_parent = str(target["frozen_target_parent_id"])
    parent_in_tube = target_parent in pair_context["transverse_parent_ids"]
    component_matches = [component for component in pair_context["baseline_components"]
                         if target_modes <= set(component["mode_ids"])]
    if parent_in_tube:
        if len(component_matches) != 1:
            raise ValueError("IN_TUBE_TARGET_COMPONENT_IMAGE_NOT_UNIQUE")
        status = RELATION_COMPONENT_UNIQUE
        baseline_component = component_matches[0]
        baseline_component_id: str | None = str(baseline_component["component_id"])
    else:
        if component_matches:
            raise ValueError("OUT_OF_TUBE_TARGET_HAS_COMPONENT_IMAGE")
        status = RELATION_COMPONENT_NONE_OUTSIDE
        baseline_component = None
        baseline_component_id = None
    parents = context["validated"]["parent_by_id"]
    parent_centroid = lpf._vector(
        parents[target_parent]["centroid"], "target parent centroid", allow_zero=True)
    distance = lpf._line_distance(
        parent_centroid, pair_context["anchor"], pair_context["axis"])
    transverse_ids = sorted(map(str, pair_context["transverse_parent_ids"]))
    row = {
        "case_token": str(target["case_token"]),
        "frozen_target_component_id": str(target["frozen_target_component_id"]),
        "frozen_target_pair_id": str(target["frozen_target_pair_id"]),
        "frozen_target_parent_id": target_parent,
        "frozen_target_mode_ids": list(map(str, target["frozen_target_mode_ids"])),
        "frozen_target_mode_ids_sha256": str(target["frozen_target_mode_ids_sha256"]),
        "frozen_target_sha256": str(target["frozen_target_sha256"]),
        "baseline_mode_fragment_rows": mode_fragment_rows,
        "baseline_mode_fragment_rows_sha256": _sha(mode_fragment_rows),
        "baseline_fragment_ids": fragment_ids,
        "baseline_fragment_ids_sha256": _sha(fragment_ids),
        "baseline_pair_mapping_count": 1,
        "baseline_local_plane_pair_id": str(pair["local_plane_pair_id"]),
        "baseline_pair_sha256": str(pair["pair_sha256"]),
        "baseline_transverse_parent_count": len(transverse_ids),
        "baseline_transverse_parent_ids_sha256": _sha(transverse_ids),
        "target_parent_line_distance_mm": float(distance),
        "line_tube_tolerance_mm": float(context["line_tolerance"]),
        "target_parent_in_frozen_transverse_set": parent_in_tube,
        "baseline_component_mapping_count": len(component_matches),
        "complete_edge_baseline_component_id": baseline_component_id,
        "target_baseline_relation_status": status,
    }
    row["target_baseline_relation_sha256"] = _record_hash(
        row, "target_baseline_relation_sha256")
    return row, baseline_component


def _outcome_row(
    relation: Mapping[str, Any], baseline_component: Mapping[str, Any] | None,
) -> dict[str, Any]:
    present = relation["target_baseline_relation_status"] == RELATION_COMPONENT_UNIQUE
    before = None if baseline_component is None else baseline_component["record"]
    row = {
        "case_token": str(relation["case_token"]),
        "frozen_target_component_id": str(relation["frozen_target_component_id"]),
        "frozen_target_mode_ids_sha256": str(relation["frozen_target_mode_ids_sha256"]),
        "baseline_local_plane_pair_id": str(relation["baseline_local_plane_pair_id"]),
        "complete_edge_baseline_component_id": relation["complete_edge_baseline_component_id"],
        "replacement_component_id": None,
        "target_baseline_relation_status": str(relation["target_baseline_relation_status"]),
        "causal_added_cross_parent_rung_count": 0,
        "causal_added_cross_parent_rung_ids_sha256": _sha([]),
        "support_parent_count_before": None if before is None else int(before["support_parent_count"]),
        "support_parent_count_after": None,
        "occupied_bin_count_before": None if before is None else int(before["occupied_bin_count"]),
        "occupied_bin_count_after": None,
        "left_side_bin_count_before": None if before is None else int(before["left_side_bin_count"]),
        "left_side_bin_count_after": None,
        "right_side_bin_count_before": None if before is None else int(before["right_side_bin_count"]),
        "right_side_bin_count_after": None,
        "transition_rung_bin_count_before": None if before is None else int(before["transition_rung_bin_count"]),
        "transition_rung_bin_count_after": None,
        "geometry_gate_depth_before": None if before is None else int(before["geometry_gate_depth"]),
        "geometry_gate_depth_after": None,
        "first_rejection_code_before": None if before is None else before["first_rejection_code"],
        "first_rejection_code_after": None,
        "geometry_certificate_valid_after": False,
        "target_reached_by_added_rung": False,
        "target_side_gate_advanced": False,
        "target_novel_rung_attributable": False,
        "target_outcome_code": (
            OUTCOME_COMPONENT_PENDING if present else OUTCOME_STRUCTURALLY_UNREACHABLE
        ),
    }
    row["target_outcome_sha256"] = _record_hash(row, "target_outcome_sha256")
    return row


def _build_case_from_payloads(
    projected_input: Mapping[str, Any], *, case_token: str,
    scan_sha256: str, candidate_pool_sha256: str, a0_prediction_sha256: str,
    frozen_control_probe_sha256: str, complete_edge_baseline_probe_sha256: str,
    frozen_control: Mapping[str, Any], complete_edge_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not re.fullmatch(r"C\d{3}", str(case_token)):
        raise ValueError("case_token must match Cddd")
    scan_hash = _require_sha("scan_sha256", scan_sha256)
    pool_hash = _require_sha("candidate_pool_sha256", candidate_pool_sha256)
    a0_hash = _require_sha("a0_prediction_sha256", a0_prediction_sha256)
    control_hash = _require_sha("frozen_control_probe_sha256", frozen_control_probe_sha256)
    baseline_hash = _require_sha(
        "complete_edge_baseline_probe_sha256", complete_edge_baseline_probe_sha256)
    if _sha(frozen_control) != control_hash:
        raise ValueError("frozen LPF control hash mismatch")
    if _sha(complete_edge_baseline) != baseline_hash:
        raise ValueError("complete-edge baseline hash mismatch")
    if complete_edge_baseline.get("schema_version") != 1 or complete_edge_baseline.get("profile") != BASELINE_PROFILE:
        raise ValueError("complete-edge baseline profile mismatch")
    context = v1_core._baseline_context(projected_input, complete_edge_baseline)
    targets = v1_core._frozen_targets(frozen_control, str(case_token))
    pair_contexts = {
        pair_id: v1_core._pair_context(pair, context)
        for pair_id, pair in sorted(context["pair_by_id"].items())
    }
    all_pair_rows = build_all_pair_membership_rows(pair_contexts)
    independently_regenerated_pair_rows = independently_regenerate_all_pair_membership_rows(
        projected_input, complete_edge_baseline)
    relations = []
    outcomes = []
    for target in targets:
        pair, mode_fragment_rows, fragment_ids = _exact_pair_lineage(
            target, context["fragment_of_mode"], list(context["pair_by_id"].values()))
        pair_id = str(pair["local_plane_pair_id"])
        relation, baseline_component = _relation_row(
            target=target, pair=pair, mode_fragment_rows=mode_fragment_rows,
            fragment_ids=fragment_ids, pair_context=pair_contexts[pair_id], context=context)
        relations.append(relation)
        outcomes.append(_outcome_row(relation, baseline_component))
    relations.sort(key=lambda row: (row["case_token"], row["frozen_target_component_id"]))
    outcomes.sort(key=lambda row: (row["case_token"], row["frozen_target_component_id"]))
    if len(relations) > MAX_TARGET_BASELINE_RELATION_ROWS:
        raise ValueError("target relation row cap breached")
    present_count = sum(
        row["target_baseline_relation_status"] == RELATION_COMPONENT_UNIQUE
        for row in relations)
    absent_count = len(relations) - present_count
    target_ids = [row["frozen_target_component_id"] for row in relations]
    outcome_ids = [row["frozen_target_component_id"] for row in outcomes]
    if target_ids != outcome_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("target relation/outcome identity mismatch")
    baseline_hashes = {
        "baseline_fragment_table_sha256": complete_edge_baseline["output_table_hashes"]["local_plane_fragment_table_sha256"],
        "baseline_orphan_table_sha256": complete_edge_baseline["output_table_hashes"]["local_plane_orphan_table_sha256"],
        "baseline_pair_records_sha256": complete_edge_baseline["output_table_hashes"]["local_plane_pair_records_sha256"],
        "baseline_effective_mode_edge_table_sha256": complete_edge_baseline["replacement_edge_table_hashes"]["effective_mode_edge_table_sha256"],
        "baseline_thresholds_sha256": _sha(complete_edge_baseline["thresholds"]),
    }
    thresholds = deepcopy(complete_edge_baseline["thresholds"])
    target_table_sha = _table_sha(targets, "frozen_target_component_id")
    target_projection = frozen_target_projection(targets)
    target_predicates = derive_target_preservation_predicates(relations, targets)
    all_pair_membership_sha = _table_sha(
        all_pair_rows, "baseline_local_plane_pair_id")
    thresholds_sha = _sha(thresholds)
    relation_sha = _table_sha(relations, "frozen_target_component_id")
    outcome_sha = _table_sha(outcomes, "frozen_target_component_id")
    attestation = {
        "mapping_preflight_kind": MAPPING_PREFLIGHT_KIND,
        "target_baseline_relation_rule_id": TARGET_BASELINE_RELATION_RULE_ID,
        "frozen_control_reproduced": True,
        "complete_edge_baseline_reproduced": True,
        "frozen_target_rows_regenerated_from_control": True,
        **target_predicates,
        "every_target_mode_has_one_baseline_fragment": all(
            all(fragment_id is not None for _mode_id, fragment_id in row["baseline_mode_fragment_rows"])
            for row in relations),
        "every_target_maps_to_exactly_two_fragments": all(
            len(row["baseline_fragment_ids"]) == 2 for row in relations),
        "every_target_has_exactly_one_pair_lineage": all(
            row["baseline_pair_mapping_count"] == 1 for row in relations),
        "every_in_tube_target_has_one_component_image": all(
            (not row["target_parent_in_frozen_transverse_set"])
            or row["baseline_component_mapping_count"] == 1 for row in relations),
        "every_out_of_tube_target_has_zero_component_images": all(
            row["target_parent_in_frozen_transverse_set"]
            or row["baseline_component_mapping_count"] == 0 for row in relations),
        "no_unrecognised_relation_status_exists": all(
            row["target_baseline_relation_status"] in TARGET_BASELINE_RELATION_STATUS
            for row in relations),
        "relation_table_untruncated": len(relations) == len(targets),
        "outcome_table_complete": len(outcomes) == len(targets),
        "threshold_hash_unchanged": (
            thresholds_sha == _sha(complete_edge_baseline["thresholds"])),
        "all_pair_membership_row_count_matches_baseline": (
            len(all_pair_rows)
            == int(complete_edge_baseline["case_counters"]["local_plane_pair_count"])
        ),
        "pair_line_tube_membership_unchanged": (
            all_pair_rows == independently_regenerated_pair_rows),
        "no_rungs_constructed": True,
        "ordinary_batch_authorized": False,
        "gold_open_authorized": False,
    }
    attestation["case_mapping_attestation_sha256"] = _record_hash(
        attestation, "case_mapping_attestation_sha256")
    counters = {
        "frozen_target_count": len(targets),
        "pair_lineage_mapped_target_count": len(relations),
        "target_component_image_present_count": present_count,
        "target_component_image_absent_outside_tube_count": absent_count,
        "relation_row_count": len(relations),
        "outcome_row_count": len(outcomes),
        "target_eligible_pair_count": len({row["baseline_local_plane_pair_id"] for row in relations}),
        "all_pair_membership_row_count": len(all_pair_rows),
        "added_cross_parent_rung_count": 0,
        "target_reached_by_added_rung_count": 0,
        "target_side_gate_advanced_count": 0,
        "target_novel_rung_attributable_count": 0,
    }
    engineering_hash = _sha({
        "profile": PRIMITIVE_PROFILE, "schema_version": PROBE_SCHEMA_VERSION,
        "mapping_preflight_kind": MAPPING_PREFLIGHT_KIND,
        "relation_rule_id": TARGET_BASELINE_RELATION_RULE_ID,
        "baseline_profile": BASELINE_PROFILE, "thresholds": thresholds,
        "hard_caps": HARD_CAPS,
    })
    semantic_hash = _sha({
        "profile": PRIMITIVE_PROFILE,
        "engineering_profile_sha256": engineering_hash,
        "frozen_control_probe_sha256": control_hash,
        "complete_edge_baseline_probe_sha256": baseline_hash,
        "frozen_target_table_sha256": target_table_sha,
        "all_pair_membership_table_sha256": all_pair_membership_sha,
        "thresholds_sha256": thresholds_sha,
        "frozen_target_baseline_relation_table_sha256": relation_sha,
        "frozen_target_outcome_table_sha256": outcome_sha,
        "case_mapping_attestation_sha256": attestation["case_mapping_attestation_sha256"],
    })
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION, "kind": PROBE_KIND,
        "profile": PRIMITIVE_PROFILE, "mapping_preflight_kind": MAPPING_PREFLIGHT_KIND,
        "target_baseline_relation_rule_id": TARGET_BASELINE_RELATION_RULE_ID,
        "research_only": True, "inference_safe": True, "prediction_mutated": False,
        "birth_columns_enabled": False, "ownership_consumed": False,
        "solver_consumed": False, "truth_inputs_consumed": False,
        "source_points_consumed": False, "source_points_serialized": False,
        "ordinary_batch_authorized": False, "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "compact_bundle_authorized": False, "gold_open_authorized": False,
        "case_token": str(case_token), "scan_sha256": scan_hash,
        "candidate_pool_sha256": pool_hash, "a0_prediction_sha256": a0_hash,
        "frozen_control_probe_sha256": control_hash,
        "complete_edge_baseline_probe_sha256": baseline_hash,
        "input_table_hashes": deepcopy(complete_edge_baseline["input_table_hashes"]),
        "baseline_table_hashes": baseline_hashes, "thresholds": thresholds,
        "hard_caps": deepcopy(HARD_CAPS), "case_counters": counters,
        "case_mapping_attestation": attestation,
        "frozen_target_table_sha256": target_table_sha,
        "frozen_target_projection": target_projection,
        "all_pair_membership_table": all_pair_rows,
        "all_pair_membership_table_sha256": all_pair_membership_sha,
        "thresholds_sha256": thresholds_sha,
        "frozen_target_baseline_relation_table_sha256": relation_sha,
        "frozen_target_outcome_table_sha256": outcome_sha,
        "frozen_target_baseline_relation_table": relations,
        "frozen_target_outcome_table": outcomes,
        "engineering_profile_sha256": engineering_hash,
        "inference_semantic_sha256": semantic_hash,
    }
    validate_v2_cross_parent_mapping_case_payload(payload)
    return payload


def build_v2_cross_parent_mapping_case(
    projected_input: Mapping[str, Any], *, case_token: str, scan_sha256: str,
    candidate_pool_sha256: str, a0_prediction_sha256: str,
    frozen_control_probe_sha256: str, complete_edge_baseline_probe_sha256: str,
) -> dict[str, Any]:
    """Regenerate both baselines and emit mapping/accounting only—never rungs."""
    control = complete_core.regenerate_frozen_control_probe(
        projected_input, scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256)
    baseline = complete_core.build_v2_complete_same_surface_edge_probe(
        projected_input, scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
        frozen_control_probe_sha256=frozen_control_probe_sha256)
    return _build_case_from_payloads(
        projected_input, case_token=case_token, scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
        frozen_control_probe_sha256=frozen_control_probe_sha256,
        complete_edge_baseline_probe_sha256=complete_edge_baseline_probe_sha256,
        frozen_control=control, complete_edge_baseline=baseline)


def execute_global_deletion_counterfactual(
    projected_input: Mapping[str, Any],
    complete_edge_baseline: Mapping[str, Any],
    *, deleted_added_cross_parent_rung_count: int,
    added_cross_parent_rung_table: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct the complete-edge fixed-pair lane after deletion."""
    deleted_rungs = sorted(
        (deepcopy(dict(row)) for row in (added_cross_parent_rung_table or ())),
        key=lambda row: str(row["mode_edge_id"]),)
    if int(deleted_added_cross_parent_rung_count) != len(deleted_rungs):
        raise ValueError("deletion count does not bind the added-rung table")
    independent_context = v1_core._baseline_context(
        deepcopy(dict(projected_input)), deepcopy(dict(complete_edge_baseline)))
    independent_pair_contexts = {
        pair_id: v1_core._pair_context(pair, independent_context)
        for pair_id, pair in sorted(independent_context["pair_by_id"].items())
    }
    actual_fragments = [fragment.record for fragment in independent_context["fragments"]]
    actual_orphans = list(independent_context["orphans"])
    rebuilt_pairs, _unused_attribution = complete_core._build_pairs(
        parents=independent_context["validated"]["parent_by_id"],
        modes=independent_context["validated"]["mode_by_id"],
        edges=independent_context["effective_edges"],
        planes=independent_context["validated"]["planes"],
        fragments=independent_context["fragments"],
        fragment_of_mode=independent_context["fragment_of_mode"],
        line_tolerance=float(independent_context["line_tolerance"]),
        bin_width=float(independent_context["bin_width"]),
        counters=Counter(), added_ids=set(), fragment_added_ids={})
    actual_pairs = sorted(
        (deepcopy(dict(pair)) for pair in rebuilt_pairs),
        key=lambda row: str(row["local_plane_pair_id"]),)
    frozen_pairs = sorted(
        (deepcopy(dict(pair)) for pair in complete_edge_baseline["local_plane_pair_records"]),
        key=lambda row: str(row["local_plane_pair_id"]),)
    # _pair_context reconstructs every baseline component membership and fails
    # if even one component record/ID is absent or extra.
    actual_component_count = sum(len(pair["components"]) for pair in actual_pairs)
    expected_component_count = int(
        complete_edge_baseline["case_counters"]["local_transition_component_count"])
    actual_certificate_count = sum(
        bool(component["geometry_certificate_valid"])
        for pair in actual_pairs for component in pair["components"])
    expected_certificate_count = int(
        complete_edge_baseline["case_counters"]["geometry_certificate_count"])
    actual_membership = build_all_pair_membership_rows(independent_pair_contexts)
    expected_membership = independently_regenerate_all_pair_membership_rows(
        projected_input, complete_edge_baseline)
    rebuilt_effective_edges = sorted(
        (deepcopy(dict(edge)) for edge in independent_context["effective_edges"]),
        key=lambda row: str(row["mode_edge_id"]),)
    frozen_effective_edges = sorted([
        *deepcopy(list(complete_edge_baseline["complete_same_surface_edge_table"])),
        *[
            deepcopy(dict(edge)) for edge in projected_input["support_mode_edges"]
            if edge["edge_kind"] == "CROSS_MODE_LOCAL"
        ],
    ], key=lambda row: str(row["mode_edge_id"]))
    actual_edge_hash = complete_core._table_sha(
        rebuilt_effective_edges, "mode_edge_id")
    expected_edge_hash = str(
        complete_edge_baseline["replacement_edge_table_hashes"][
            "effective_mode_edge_table_sha256"])
    actual_fragment_hash = complete_core._table_sha(
        actual_fragments, "local_plane_fragment_id")
    expected_fragment_hash = str(
        complete_edge_baseline["output_table_hashes"]["local_plane_fragment_table_sha256"])
    actual_orphan_hash = complete_core._table_sha(
        actual_orphans, "local_plane_orphan_id")
    expected_orphan_hash = str(
        complete_edge_baseline["output_table_hashes"]["local_plane_orphan_table_sha256"])
    actual_pair_hash = complete_core._table_sha(actual_pairs, "local_plane_pair_id")
    expected_pair_hash = str(
        complete_edge_baseline["output_table_hashes"]["local_plane_pair_records_sha256"])
    actual_membership_hash = _table_sha(
        actual_membership, "baseline_local_plane_pair_id")
    expected_membership_hash = _table_sha(
        expected_membership, "baseline_local_plane_pair_id")
    rebuilt_components = [
        {"baseline_local_plane_pair_id": pair["local_plane_pair_id"],
         "component_record": deepcopy(component)}
        for pair in actual_pairs for component in pair["components"]]
    frozen_components = [
        {"baseline_local_plane_pair_id": pair["local_plane_pair_id"],
         "component_record": deepcopy(component)}
        for pair in frozen_pairs for component in pair["components"]]
    rebuilt_geometry = [
        {
            "local_transition_component_id": row["component_record"]["local_transition_component_id"],
            "geometry_gate_depth": row["component_record"]["geometry_gate_depth"],
            "geometry_gates": deepcopy(row["component_record"]["geometry_gates"]),
            "first_rejection_code": row["component_record"]["first_rejection_code"],
            "geometry_certificate_valid": row["component_record"]["geometry_certificate_valid"],
        }
        for row in rebuilt_components]
    frozen_geometry = [
        {
            "local_transition_component_id": row["component_record"]["local_transition_component_id"],
            "geometry_gate_depth": row["component_record"]["geometry_gate_depth"],
            "geometry_gates": deepcopy(row["component_record"]["geometry_gates"]),
            "first_rejection_code": row["component_record"]["first_rejection_code"],
            "geometry_certificate_valid": row["component_record"]["geometry_certificate_valid"],
        }
        for row in frozen_components]
    equalities = (
        rebuilt_effective_edges == frozen_effective_edges,
        actual_fragments == list(complete_edge_baseline["local_plane_fragment_table"]),
        actual_orphans == list(complete_edge_baseline["local_plane_orphan_table"]),
        actual_pairs == frozen_pairs,
        actual_membership == expected_membership,
        rebuilt_components == frozen_components,
        rebuilt_geometry == frozen_geometry,
        actual_edge_hash == expected_edge_hash,
        actual_fragment_hash == expected_fragment_hash,
        actual_orphan_hash == expected_orphan_hash,
        actual_pair_hash == expected_pair_hash,
        actual_component_count == expected_component_count,
        actual_certificate_count == expected_certificate_count,
    )
    row = {
        "counterfactual_profile": v1_core.COUNTERFACTUAL_PROFILE,
        "executed_independent_rebuild": bool(independent_pair_contexts or not actual_pairs),
        "deleted_added_cross_parent_rung_count": int(deleted_added_cross_parent_rung_count),
        "deleted_added_cross_parent_rung_table": deleted_rungs,
        "deleted_added_cross_parent_rung_table_sha256": complete_core._table_sha(
            deleted_rungs, "mode_edge_id"),
        "actual_effective_edge_table_sha256": actual_edge_hash,
        "expected_effective_edge_table_sha256": expected_edge_hash,
        "actual_fragment_table_sha256": actual_fragment_hash,
        "expected_fragment_table_sha256": expected_fragment_hash,
        "actual_orphan_table_sha256": actual_orphan_hash,
        "expected_orphan_table_sha256": expected_orphan_hash,
        "actual_pair_records_sha256": actual_pair_hash,
        "expected_pair_records_sha256": expected_pair_hash,
        "actual_all_pair_membership_table_sha256": actual_membership_hash,
        "expected_all_pair_membership_table_sha256": expected_membership_hash,
        "actual_component_count": actual_component_count,
        "expected_component_count": expected_component_count,
        "actual_certificate_count": actual_certificate_count,
        "expected_certificate_count": expected_certificate_count,
        "all_required_equalities": all(equalities),
        "rebuilt_effective_edge_table": rebuilt_effective_edges,
        "frozen_effective_edge_table": frozen_effective_edges,
        "rebuilt_fragment_table": actual_fragments,
        "frozen_fragment_table": deepcopy(list(complete_edge_baseline["local_plane_fragment_table"])),
        "rebuilt_orphan_table": actual_orphans,
        "frozen_orphan_table": deepcopy(list(complete_edge_baseline["local_plane_orphan_table"])),
        "rebuilt_pair_records": actual_pairs,
        "frozen_pair_records": frozen_pairs,
        "rebuilt_all_pair_membership_table": actual_membership,
        "frozen_all_pair_membership_table": expected_membership,
        "rebuilt_component_table": rebuilt_components,
        "frozen_component_table": frozen_components,
        "rebuilt_geometry_gate_table": rebuilt_geometry,
        "frozen_geometry_gate_table": frozen_geometry,
    }
    row["deletion_evidence_sha256"] = _record_hash(row, "deletion_evidence_sha256")
    return row


def _build_replacement_lane(
    *, context: Mapping[str, Any], pair_contexts: Mapping[str, Mapping[str, Any]],
    added_rungs: Sequence[Mapping[str, Any]], frozen_control: Mapping[str, Any],
    mapping_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rungs_by_pair: dict[str, list[dict[str, Any]]] = {}
    for rung in added_rungs:
        rungs_by_pair.setdefault(str(rung["baseline_local_plane_pair_id"]), []).append(dict(rung))
    planes_by_id = {
        lpf._canonical_plane(plane)[2]: plane for plane in context["validated"]["planes"]
    }
    replacement_pairs: list[dict[str, Any]] = []
    replacement_meta: dict[str, dict[str, Any]] = {}
    total_components = 0
    for pair_id, pair_context in sorted(pair_contexts.items()):
        baseline_pair = context["pair_by_id"][pair_id]
        added = rungs_by_pair.get(pair_id, [])
        effective = [*pair_context["baseline_edges"], *added]
        components = lpf._connected_components(
            set(pair_context["supported_modes"]), effective)
        total_components += len(components)
        if (
            len(components) > lpf.MAX_COMPONENTS_PER_LOCAL_PAIR
            or total_components > lpf.MAX_LOCAL_TRANSITION_COMPONENTS
        ):
            raise ValueError("ordinary V2 replacement component cap breached")
        global_comparison, global_pair_ids = lpf._global_pair_comparison(
            pair_context["left"], pair_context["right"])
        records = []
        for mode_ids in components:
            component_edges = [
                edge for edge in effective
                if set(map(str, edge["mode_ids"])) <= set(mode_ids)
            ]
            record, local_bins, midpoint_bins = v1_core._build_component(
                pair_context, mode_ids, component_edges, context,
                global_pair_ids, planes_by_id)
            records.append(record)
            replacement_meta[str(record["local_transition_component_id"])] = {
                "mode_ids": mode_ids, "edges": component_edges,
                "local_bins": local_bins, "midpoint_bins": midpoint_bins,
                "pair_context": pair_context, "record": record,
            }
        pair = deepcopy(dict(baseline_pair))
        pair["components"] = sorted(
            records, key=lambda row: str(row["local_transition_component_id"]))
        pair["component_count"] = len(records)
        pair["cross_mode_rung_count"] = sum(
            edge["edge_kind"] in {
                "CROSS_MODE_LOCAL", "CROSS_MODE_ADJACENT_PARENT"}
            for edge in effective)
        pair["global_pair_comparison"] = global_comparison
        pair["pair_sha256"] = _record_hash(pair, "pair_sha256")
        replacement_pairs.append(pair)
    replacement_pairs.sort(key=lambda row: str(row["local_plane_pair_id"]))
    before_components = [
        {
            "baseline_local_plane_pair_id": pair_id,
            "component_record": deepcopy(component["record"]),
        }
        for pair_id, pair_context in sorted(pair_contexts.items())
        for component in pair_context["baseline_components"]
    ]
    before_components.sort(
        key=lambda row: (
            row["baseline_local_plane_pair_id"],
            row["component_record"]["local_transition_component_id"]))
    return replacement_pairs, replacement_meta, before_components


def _ordinary_attribution(
    *, replacement_meta: Mapping[str, Mapping[str, Any]],
    pair_contexts: Mapping[str, Mapping[str, Any]],
    frozen_control: Mapping[str, Any],
    mapping_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_valid = []
    for pair_id, pair_context in pair_contexts.items():
        for item in pair_context["baseline_components"]:
            if item["record"]["geometry_certificate_valid"]:
                baseline_valid.append({**item, "axis": pair_context["axis"]})
    control_valid = []
    for pair in frozen_control["local_plane_pair_records"]:
        axis = np.asarray(pair["intersection_axis"], dtype=np.float64)
        for record in pair["components"]:
            if record["geometry_certificate_valid"]:
                control_valid.append({
                    "component_id": str(record["local_transition_component_id"]),
                    "record": record, "axis": axis,
                })
    rows = []
    target_modes = {
        str(row["frozen_target_component_id"]): set(map(str, row["frozen_target_mode_ids"]))
        for row in mapping_payload["frozen_target_baseline_relation_table"]
    }
    for component_id, meta in sorted(replacement_meta.items()):
        mode_set = set(meta["mode_ids"])
        pair_context = meta["pair_context"]
        added = [
            edge for edge in meta["edges"]
            if edge["edge_kind"] == "CROSS_MODE_ADJACENT_PARENT"]
        added_ids = sorted(str(edge["mode_edge_id"]) for edge in added)
        deletion_children = [
            item for item in pair_context["baseline_components"]
            if set(item["mode_ids"]) <= mode_set]
        deletion_ids = sorted(item["component_id"] for item in deletion_children)
        partition_changed = not (
            len(deletion_children) == 1
            and set(deletion_children[0]["mode_ids"]) == mode_set)
        baseline_rung_count = (
            deletion_children[0]["record"]["transition_rung_bin_count"]
            if len(deletion_children) == 1 else None)
        combined_bins = set(meta["local_bins"]) | set(meta["midpoint_bins"])
        rung_changed = partition_changed or baseline_rung_count != len(combined_bins)
        deletion_valid = [
            item for item in deletion_children
            if item["record"]["geometry_certificate_valid"]]
        deletion_valid_ids = sorted(item["component_id"] for item in deletion_valid)
        deletion_equivalent = sorted(
            item["component_id"] for item in deletion_valid
            if v1_core._line_equivalent(
                meta["record"], pair_context["axis"],
                item["record"], pair_context["axis"]))
        complete_equivalent = sorted(
            item["component_id"] for item in baseline_valid
            if v1_core._line_equivalent(
                meta["record"], pair_context["axis"], item["record"], item["axis"]))
        control_equivalent = sorted(
            item["component_id"] for item in control_valid
            if v1_core._line_equivalent(
                meta["record"], pair_context["axis"], item["record"], item["axis"]))
        dependent = bool(added_ids and (partition_changed or rung_changed))
        disappears = not deletion_equivalent
        novel = bool(
            meta["record"]["geometry_certificate_valid"] and dependent
            and disappears and not complete_equivalent and not control_equivalent)
        contained_targets = sorted(
            target_id for target_id, modes in target_modes.items()
            if modes <= mode_set)
        row = {
            "local_transition_component_id": component_id,
            "baseline_local_plane_pair_id": pair_context["pair_id"],
            "fragment_ids": list(pair_context["fragment_ids"]),
            "contained_added_cross_parent_rung_count": len(added_ids),
            "contained_added_cross_parent_rung_ids_sha256": _sha(added_ids),
            "witness_contained_added_cross_parent_rung_ids": added_ids[:8],
            "witness_ids_truncated": len(added_ids) > 8,
            "cross_parent_midpoint_rung_bins_sha256": _sha(sorted(meta["midpoint_bins"])),
            "contained_frozen_target_count": len(contained_targets),
            "contained_frozen_target_ids_sha256": _sha(contained_targets),
            "component_partition_changed_under_deletion": partition_changed,
            "rung_bin_set_changed_under_deletion": rung_changed,
            "deletion_counterfactual_component_ids": deletion_ids,
            "deletion_counterfactual_component_ids_sha256": _sha(deletion_ids),
            "deletion_counterfactual_valid_certificate_ids": deletion_valid_ids,
            "deletion_counterfactual_valid_certificate_ids_sha256": _sha(deletion_valid_ids),
            "deletion_counterfactual_line_equivalent_certificate_ids": deletion_equivalent,
            "deletion_counterfactual_line_equivalent_certificate_ids_sha256": _sha(deletion_equivalent),
            "certificate_disappears_under_deletion": disappears,
            "complete_edge_equivalent_certificate_ids": complete_equivalent,
            "complete_edge_equivalent_certificate_ids_sha256": _sha(complete_equivalent),
            "frozen_control_equivalent_certificate_ids": control_equivalent,
            "frozen_control_equivalent_certificate_ids_sha256": _sha(control_equivalent),
            "rung_semantics_dependent": dependent,
            "geometry_certificate_valid": bool(meta["record"]["geometry_certificate_valid"]),
            "novel_rung_attributable": novel,
        }
        row["component_rung_attribution_sha256"] = _record_hash(
            row, "component_rung_attribution_sha256")
        rows.append(row)
    return rows


def _ordinary_target_outcomes(
    *, mapping_payload: Mapping[str, Any],
    replacement_meta: Mapping[str, Mapping[str, Any]],
    attribution_by_component: Mapping[str, Mapping[str, Any]],
    pair_contexts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    outcomes = []
    relations = {
        str(row["frozen_target_component_id"]): row
        for row in mapping_payload["frozen_target_baseline_relation_table"]}
    mapping_outcomes = {
        str(row["frozen_target_component_id"]): row
        for row in mapping_payload["frozen_target_outcome_table"]}
    for target_id, relation in sorted(relations.items()):
        if relation["target_baseline_relation_status"] == RELATION_COMPONENT_NONE_OUTSIDE:
            outcomes.append(deepcopy(dict(mapping_outcomes[target_id])))
            continue
        pair_id = str(relation["baseline_local_plane_pair_id"])
        target_modes = set(map(str, relation["frozen_target_mode_ids"]))
        baseline_component_id = str(relation["complete_edge_baseline_component_id"])
        baseline_component = next(
            item for item in pair_contexts[pair_id]["baseline_components"]
            if item["component_id"] == baseline_component_id)
        replacements = [
            meta for meta in replacement_meta.values()
            if meta["pair_context"]["pair_id"] == pair_id
            and target_modes <= set(meta["mode_ids"])]
        if len(replacements) != 1:
            raise ValueError("present target replacement component image is not unique")
        meta = replacements[0]
        after = meta["record"]
        before = baseline_component["record"]
        attribution = attribution_by_component[
            str(after["local_transition_component_id"])]
        causal = sorted(
            str(edge["mode_edge_id"]) for edge in meta["edges"]
            if edge["edge_kind"] == "CROSS_MODE_ADJACENT_PARENT")
        reached = bool(causal)
        side_advanced = bool(
            reached
            and after["geometry_gates"]["minimum_left_side_bins"]
            and after["geometry_gates"]["minimum_right_side_bins"]
            and (
                not before["geometry_gates"]["minimum_left_side_bins"]
                or not before["geometry_gates"]["minimum_right_side_bins"]))
        row = {
            "case_token": str(relation["case_token"]),
            "frozen_target_component_id": target_id,
            "frozen_target_mode_ids_sha256": str(relation["frozen_target_mode_ids_sha256"]),
            "baseline_local_plane_pair_id": pair_id,
            "complete_edge_baseline_component_id": baseline_component_id,
            "replacement_component_id": str(after["local_transition_component_id"]),
            "target_baseline_relation_status": RELATION_COMPONENT_UNIQUE,
            "causal_added_cross_parent_rung_count": len(causal),
            "causal_added_cross_parent_rung_ids_sha256": _sha(causal),
            "support_parent_count_before": int(before["support_parent_count"]),
            "support_parent_count_after": int(after["support_parent_count"]),
            "occupied_bin_count_before": int(before["occupied_bin_count"]),
            "occupied_bin_count_after": int(after["occupied_bin_count"]),
            "left_side_bin_count_before": int(before["left_side_bin_count"]),
            "left_side_bin_count_after": int(after["left_side_bin_count"]),
            "right_side_bin_count_before": int(before["right_side_bin_count"]),
            "right_side_bin_count_after": int(after["right_side_bin_count"]),
            "transition_rung_bin_count_before": int(before["transition_rung_bin_count"]),
            "transition_rung_bin_count_after": int(after["transition_rung_bin_count"]),
            "geometry_gate_depth_before": int(before["geometry_gate_depth"]),
            "geometry_gate_depth_after": int(after["geometry_gate_depth"]),
            "first_rejection_code_before": before["first_rejection_code"],
            "first_rejection_code_after": after["first_rejection_code"],
            "geometry_certificate_valid_after": bool(after["geometry_certificate_valid"]),
            "target_reached_by_added_rung": reached,
            "target_side_gate_advanced": side_advanced,
            "target_novel_rung_attributable": bool(
                reached and attribution["novel_rung_attributable"]),
            "target_outcome_code": "TARGET_COMPONENT_IMAGE_PRESENT_ORDINARY_V2_EVALUATED",
        }
        row["target_outcome_sha256"] = _record_hash(row, "target_outcome_sha256")
        outcomes.append(row)
    outcomes.sort(key=lambda row: (row["case_token"], row["frozen_target_component_id"]))
    return outcomes


def new_ordinary_v2_criteria() -> dict[str, bool]:
    """Fail-closed ordinary criterion initialization."""
    return {key: False for key in ORDINARY_V2_CRITERION_KEYS}


def build_v2_cross_parent_ordinary_case(
    projected_input: Mapping[str, Any], *, case_token: str, scan_sha256: str,
    candidate_pool_sha256: str, a0_prediction_sha256: str,
    frozen_control_probe_sha256: str, complete_edge_baseline_probe_sha256: str,
    expected_mapping_case_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one authorized ordinary V2 case; never opens gold or runs replays."""
    control = complete_core.regenerate_frozen_control_probe(
        projected_input, scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256)
    baseline = complete_core.build_v2_complete_same_surface_edge_probe(
        projected_input, scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
        frozen_control_probe_sha256=frozen_control_probe_sha256)
    mapping_payload = _build_case_from_payloads(
        projected_input, case_token=case_token, scan_sha256=scan_sha256,
        candidate_pool_sha256=candidate_pool_sha256,
        a0_prediction_sha256=a0_prediction_sha256,
        frozen_control_probe_sha256=frozen_control_probe_sha256,
        complete_edge_baseline_probe_sha256=complete_edge_baseline_probe_sha256,
        frozen_control=control, complete_edge_baseline=baseline)
    mapping_case_sha = _sha(mapping_payload)
    if expected_mapping_case_sha256 is not None:
        if _require_sha("expected_mapping_case_sha256", expected_mapping_case_sha256) != mapping_case_sha:
            raise ValueError("ordinary V2 mapping case hash mismatch")
    context = v1_core._baseline_context(projected_input, baseline)
    pair_contexts = {
        pair_id: v1_core._pair_context(pair, context)
        for pair_id, pair in sorted(context["pair_by_id"].items())
    }
    relations = list(mapping_payload["frozen_target_baseline_relation_table"])
    eligible_pair_ids = sorted({
        str(row["baseline_local_plane_pair_id"]) for row in relations})
    target_ids_by_pair: dict[str, list[str]] = {}
    for row in relations:
        target_ids_by_pair.setdefault(
            str(row["baseline_local_plane_pair_id"]), []).append(
                str(row["frozen_target_component_id"]))
    target_eligible_pair_table = []
    for pair_id in eligible_pair_ids:
        target_ids = sorted(target_ids_by_pair[pair_id])
        target_eligible_pair_table.append({
            "baseline_local_plane_pair_id": pair_id,
            "frozen_target_component_ids": target_ids,
            "frozen_target_component_ids_sha256": _sha(target_ids),
        })
    added_rungs, adjacent_parent_count, evaluation_count = v1_core._build_rungs(
        context, pair_contexts, set(eligible_pair_ids))
    replacement_pairs, replacement_meta, before_components = _build_replacement_lane(
        context=context, pair_contexts=pair_contexts, added_rungs=added_rungs,
        frozen_control=control, mapping_payload=mapping_payload)
    attributions = _ordinary_attribution(
        replacement_meta=replacement_meta, pair_contexts=pair_contexts,
        frozen_control=control, mapping_payload=mapping_payload)
    attribution_by_component = {
        str(row["local_transition_component_id"]): row for row in attributions}
    outcomes = _ordinary_target_outcomes(
        mapping_payload=mapping_payload, replacement_meta=replacement_meta,
        attribution_by_component=attribution_by_component,
        pair_contexts=pair_contexts)
    replacement_components = [
        {
            "baseline_local_plane_pair_id": meta["pair_context"]["pair_id"],
            "component_record": deepcopy(meta["record"]),
        }
        for _component_id, meta in sorted(replacement_meta.items())
    ]
    replacement_components.sort(key=lambda row: (
        row["baseline_local_plane_pair_id"],
        row["component_record"]["local_transition_component_id"]))
    novelty = []
    for attribution in attributions:
        if not attribution["geometry_certificate_valid"]:
            continue
        row = {
            key: deepcopy(attribution[key])
            for key in v1_core.CERTIFICATE_NOVELTY_FIELDS
            if key != "certificate_novelty_sha256"
        }
        row["certificate_novelty_sha256"] = _record_hash(
            row, "certificate_novelty_sha256")
        novelty.append(row)
    deletion = execute_global_deletion_counterfactual(
        projected_input, baseline,
        deleted_added_cross_parent_rung_count=len(added_rungs),
        added_cross_parent_rung_table=added_rungs)
    criteria = new_ordinary_v2_criteria()
    criteria["profile_identity_valid"] = bool(
        mapping_payload["schema_version"] == PROBE_SCHEMA_VERSION
        and mapping_payload["kind"] == PROBE_KIND
        and mapping_payload["profile"] == PRIMITIVE_PROFILE)
    criteria["input_projection_valid"] = bool(
        set(projected_input) == set(ALLOWED_INPUT_FIELDS))
    criteria["frozen_control_reproduced"] = bool(
        _sha(control) == frozen_control_probe_sha256)
    criteria["complete_edge_baseline_reproduced"] = bool(
        _sha(baseline) == complete_edge_baseline_probe_sha256)
    criteria["mapping_case_reproduced"] = bool(
        expected_mapping_case_sha256 is None
        or mapping_case_sha == expected_mapping_case_sha256)
    criteria["target_pair_population_exact"] = bool(
        set(eligible_pair_ids) == {
            row["baseline_local_plane_pair_id"]
            for row in target_eligible_pair_table})
    criteria["rung_parents_distinct"] = all(
        row["parent_atom_ids"][0] != row["parent_atom_ids"][1]
        for row in added_rungs)
    criteria["rung_parent_voxels_adjacent"] = all(
        max(abs(int(left) - int(right)) for left, right in zip(
            row["parent_voxel_keys"][0], row["parent_voxel_keys"][1])) == 1
        for row in added_rungs)
    criteria["rungs_join_opposite_fixed_fragments"] = all(
        [context["fragment_of_mode"].get(mode_id) for mode_id in row["mode_ids"]]
        == list(row["baseline_fragment_ids"])
        for row in added_rungs)
    criteria["rungs_use_target_eligible_pairs_only"] = all(
        row["baseline_local_plane_pair_id"] in set(eligible_pair_ids)
        for row in added_rungs)
    criteria["rung_parents_inside_frozen_tube"] = all(
        set(row["parent_atom_ids"]) <= pair_contexts[
            row["baseline_local_plane_pair_id"]]["transverse_parent_ids"]
        for row in added_rungs)
    criteria["orphan_modes_excluded"] = all(
        not (set(row["mode_ids"]) & set(context["orphan_modes"]))
        for row in added_rungs)
    criteria["axial_rule_exact"] = all(
        float(row["axial_parent_separation_mm"])
        <= float(row["axial_bin_width_mm"])
        for row in added_rungs)
    criteria["no_pair_births"] = bool(
        {row["local_plane_pair_id"] for row in replacement_pairs}
        == set(context["pair_by_id"]))
    criteria["no_fragment_or_orphan_births"] = bool(
        mapping_payload["baseline_table_hashes"]["baseline_fragment_table_sha256"]
        == baseline["output_table_hashes"]["local_plane_fragment_table_sha256"]
        and mapping_payload["baseline_table_hashes"]["baseline_orphan_table_sha256"]
        == baseline["output_table_hashes"]["local_plane_orphan_table_sha256"])
    present_outcomes = [
        row for row in outcomes
        if row["target_baseline_relation_status"] == RELATION_COMPONENT_UNIQUE]
    structural_outcomes = [
        row for row in outcomes
        if row["target_baseline_relation_status"] == RELATION_COMPONENT_NONE_OUTSIDE]
    criteria["present_target_replacement_mapping_unique"] = bool(
        len(present_outcomes)
        == int(mapping_payload["case_counters"]["target_component_image_present_count"])
        and all(isinstance(row["replacement_component_id"], str) for row in present_outcomes))
    mapping_structural = [
        row for row in mapping_payload["frozen_target_outcome_table"]
        if row["target_baseline_relation_status"] == RELATION_COMPONENT_NONE_OUTSIDE]
    criteria["structural_negative_outcomes_forced"] = bool(
        structural_outcomes == mapping_structural)
    criteria["one_parent_components_zero_span"] = all(
        int(row["component_record"]["support_parent_count"]) != 1
        or row["component_record"]["intersection_span_endpoints"][0]
        == row["component_record"]["intersection_span_endpoints"][1]
        for row in replacement_components)
    criteria["global_deletion_counterfactual_executed_and_valid"] = bool(
        deletion["executed_independent_rebuild"]
        and deletion["all_required_equalities"])
    criteria["all_structural_caps_satisfied"] = bool(
        len(added_rungs) <= v1_core.MAX_ADDED_CROSS_PARENT_RUNGS
        and len(context["effective_edges"]) + len(added_rungs)
        <= v1_core.MAX_EFFECTIVE_MODE_EDGES
        and len(replacement_components) <= lpf.MAX_LOCAL_TRANSITION_COMPONENTS)
    criteria["safety_flags_valid"] = bool(
        mapping_payload["gold_open_authorized"] is False
        and mapping_payload["truth_inputs_consumed"] is False
        and mapping_payload["ownership_consumed"] is False
        and mapping_payload["solver_consumed"] is False
        and mapping_payload["birth_columns_enabled"] is False)
    all_criteria_valid = all(criteria.values())
    output_hashes = {
        "target_eligible_pair_table_sha256": _table_sha(
            target_eligible_pair_table, "baseline_local_plane_pair_id"),
        "added_cross_parent_rung_table_sha256": complete_core._table_sha(
            added_rungs, "mode_edge_id"),
        "complete_edge_before_component_table_sha256": _sha(before_components),
        "replacement_component_table_sha256": _sha(replacement_components),
        "component_rung_attribution_table_sha256": complete_core._table_sha(
            attributions, "local_transition_component_id"),
        "frozen_target_outcome_table_sha256": _table_sha(
            outcomes, "frozen_target_component_id"),
        "certificate_novelty_table_sha256": complete_core._table_sha(
            novelty, "local_transition_component_id"),
        "local_plane_pair_records_sha256": complete_core._table_sha(
            replacement_pairs, "local_plane_pair_id"),
        "global_deletion_counterfactual_sha256": deletion["deletion_evidence_sha256"],
    }
    counters = {
        "frozen_target_count": len(relations),
        "target_component_image_present_count": len(present_outcomes),
        "target_component_image_absent_outside_tube_count": len(structural_outcomes),
        "target_eligible_pair_count": len(eligible_pair_ids),
        "adjacent_parent_pair_count": adjacent_parent_count,
        "rung_candidate_evaluation_count": evaluation_count,
        "added_cross_parent_rung_count": len(added_rungs),
        "replacement_component_count": len(replacement_components),
        "replacement_geometry_certificate_count": sum(
            bool(row["component_record"]["geometry_certificate_valid"])
            for row in replacement_components),
        "target_reached_by_added_rung_count": sum(
            bool(row["target_reached_by_added_rung"]) for row in outcomes),
        "target_side_gate_advanced_count": sum(
            bool(row["target_side_gate_advanced"]) for row in outcomes),
        "target_novel_rung_attributable_certificate_count": sum(
            bool(row["target_novel_rung_attributable"]) for row in outcomes),
        "one_parent_finite_span_count": sum(
            int(row["component_record"]["support_parent_count"]) == 1
            and row["component_record"]["intersection_span_endpoints"][0]
            != row["component_record"]["intersection_span_endpoints"][1]
            for row in replacement_components),
    }
    ordinary_hard_caps = {**HARD_CAPS, **v1_core.HARD_CAPS}
    engineering_hash = _sha({
        "profile": PRIMITIVE_PROFILE, "artifact_mode": "ORDINARY_V2_CASE",
        "rung_rule_id": RUNG_RULE_ID, "criteria": list(ORDINARY_V2_CRITERION_KEYS),
        "thresholds": mapping_payload["thresholds"], "hard_caps": ordinary_hard_caps})
    semantic_hash = _sha({
        "profile": PRIMITIVE_PROFILE,
        "engineering_profile_sha256": engineering_hash,
        "mapping_case_sha256": mapping_case_sha,
        "output_table_hashes": output_hashes,
        "ordinary_v2_criteria": criteria,
    })
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION, "kind": PROBE_KIND,
        "profile": PRIMITIVE_PROFILE, "artifact_mode": "ORDINARY_V2_CASE",
        "research_only": True, "inference_safe": True,
        "prediction_mutated": False, "birth_columns_enabled": False,
        "ownership_consumed": False, "solver_consumed": False,
        "truth_inputs_consumed": False, "source_points_consumed": False,
        "source_points_serialized": False, "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "gold_open_authorized": False, "case_token": str(case_token),
        "scan_sha256": str(scan_sha256),
        "candidate_pool_sha256": str(candidate_pool_sha256),
        "a0_prediction_sha256": str(a0_prediction_sha256),
        "frozen_control_probe_sha256": str(frozen_control_probe_sha256),
        "complete_edge_baseline_probe_sha256": str(complete_edge_baseline_probe_sha256),
        "mapping_case_sha256": mapping_case_sha,
        "input_table_hashes": deepcopy(mapping_payload["input_table_hashes"]),
        "baseline_table_hashes": deepcopy(mapping_payload["baseline_table_hashes"]),
        "thresholds": deepcopy(mapping_payload["thresholds"]),
        "hard_caps": ordinary_hard_caps,
        "target_eligible_pair_table": target_eligible_pair_table,
        "frozen_target_projection": deepcopy(mapping_payload["frozen_target_projection"]),
        "frozen_target_baseline_relation_table": deepcopy(relations),
        "all_pair_membership_table": deepcopy(mapping_payload["all_pair_membership_table"]),
        "baseline_local_plane_fragment_table": deepcopy(baseline["local_plane_fragment_table"]),
        "baseline_local_plane_orphan_table": deepcopy(baseline["local_plane_orphan_table"]),
        "complete_edge_before_component_table": before_components,
        "added_cross_parent_rung_table": added_rungs,
        "replacement_component_table": replacement_components,
        "component_rung_attribution_table": attributions,
        "frozen_target_outcome_table": outcomes,
        "certificate_novelty_table": novelty,
        "local_plane_pair_records": replacement_pairs,
        "global_deletion_counterfactual": deletion,
        "output_table_hashes": output_hashes, "case_counters": counters,
        "ordinary_v2_criteria": criteria,
        "all_ordinary_v2_criteria_valid": all_criteria_valid,
        "packet_status": (
            "ORDINARY_V2_CASE_COMPLETE" if all_criteria_valid
            else "INVALID_EXPERIMENT_STOP"),
        "engineering_profile_sha256": engineering_hash,
        "inference_semantic_sha256": semantic_hash,
    }
    validate_v2_cross_parent_ordinary_case_payload(payload)
    return payload


def validate_v2_cross_parent_ordinary_case_payload(
    payload: Mapping[str, Any], *, projected_input: Mapping[str, Any] | None = None,
) -> None:
    if set(payload) != ORDINARY_TOP_LEVEL_FIELDS:
        raise ValueError("ordinary V2 payload top-level schema mismatch")
    if (
        payload.get("schema_version") != PROBE_SCHEMA_VERSION
        or payload.get("kind") != PROBE_KIND
        or payload.get("profile") != PRIMITIVE_PROFILE
        or payload.get("artifact_mode") != "ORDINARY_V2_CASE"
    ):
        raise ValueError("ordinary V2 identity rejects V1/foreign artifact")
    criteria = dict(payload["ordinary_v2_criteria"])
    if set(criteria) != set(ORDINARY_V2_CRITERION_KEYS):
        raise ValueError("ordinary V2 criterion schema mismatch")
    if any(value is not True and value is not False for value in criteria.values()):
        raise ValueError("ordinary V2 criteria must be Boolean")
    if payload["all_ordinary_v2_criteria_valid"] is not all(criteria.values()):
        raise ValueError("ordinary V2 aggregate criterion mismatch")
    if set(payload["global_deletion_counterfactual"]) != DELETION_EVIDENCE_FIELDS:
        raise ValueError("ordinary V2 deletion evidence schema mismatch")
    deletion = payload["global_deletion_counterfactual"]
    if deletion["deletion_evidence_sha256"] != _record_hash(
        deletion, "deletion_evidence_sha256"):
        raise ValueError("ordinary V2 deletion evidence hash mismatch")
    if deletion["deleted_added_cross_parent_rung_table"] != payload["added_cross_parent_rung_table"]:
        raise ValueError("ordinary V2 deletion input does not match added-rung table")
    if deletion["deleted_added_cross_parent_rung_table_sha256"] != payload[
        "output_table_hashes"]["added_cross_parent_rung_table_sha256"]:
        raise ValueError("ordinary V2 deletion added-rung hash mismatch")
    raw_deletion_equalities = (
        deletion["rebuilt_effective_edge_table"] == deletion["frozen_effective_edge_table"],
        deletion["rebuilt_fragment_table"] == deletion["frozen_fragment_table"],
        deletion["rebuilt_orphan_table"] == deletion["frozen_orphan_table"],
        deletion["rebuilt_pair_records"] == deletion["frozen_pair_records"],
        deletion["rebuilt_all_pair_membership_table"]
        == deletion["frozen_all_pair_membership_table"],
        deletion["rebuilt_component_table"] == deletion["frozen_component_table"],
        deletion["rebuilt_geometry_gate_table"] == deletion["frozen_geometry_gate_table"],
    )
    if not (
        deletion["executed_independent_rebuild"]
        and deletion["all_required_equalities"]
        and all(raw_deletion_equalities)
    ):
        raise ValueError("ordinary V2 deletion evidence is not valid")
    relation_ids = [
        row["frozen_target_component_id"]
        for row in payload["frozen_target_baseline_relation_table"]]
    outcome_ids = [
        row["frozen_target_component_id"]
        for row in payload["frozen_target_outcome_table"]]
    if relation_ids != outcome_ids:
        raise ValueError("ordinary V2 target outcome identity mismatch")
    for row in payload["target_eligible_pair_table"]:
        if set(row) != TARGET_ELIGIBLE_PAIR_FIELDS:
            raise ValueError("ordinary V2 eligible-pair schema mismatch")
        ids = list(row["frozen_target_component_ids"])
        if ids != sorted(ids) or row["frozen_target_component_ids_sha256"] != _sha(ids):
            raise ValueError("ordinary V2 eligible-pair target hash mismatch")
    for row in payload["added_cross_parent_rung_table"]:
        if set(row) != v1_core.CROSS_PARENT_RUNG_FIELDS:
            raise ValueError("ordinary V2 rung schema mismatch")
        if row["rung_record_sha256"] != _record_hash(row, "rung_record_sha256"):
            raise ValueError("ordinary V2 rung hash mismatch")
    structural = [
        row for row in payload["frozen_target_outcome_table"]
        if row["target_baseline_relation_status"] == RELATION_COMPONENT_NONE_OUTSIDE]
    for row in structural:
        if (
            row["replacement_component_id"] is not None
            or row["causal_added_cross_parent_rung_count"] != 0
            or row["target_reached_by_added_rung"]
            or row["target_side_gate_advanced"]
            or row["target_novel_rung_attributable"]
            or row["target_outcome_code"] != OUTCOME_STRUCTURALLY_UNREACHABLE
        ):
            raise ValueError("ordinary V2 structural-negative outcome changed")
    if projected_input is not None:
        rebuilt = build_v2_cross_parent_ordinary_case(
            projected_input, case_token=str(payload["case_token"]),
            scan_sha256=str(payload["scan_sha256"]),
            candidate_pool_sha256=str(payload["candidate_pool_sha256"]),
            a0_prediction_sha256=str(payload["a0_prediction_sha256"]),
            frozen_control_probe_sha256=str(payload["frozen_control_probe_sha256"]),
            complete_edge_baseline_probe_sha256=str(
                payload["complete_edge_baseline_probe_sha256"]),
            expected_mapping_case_sha256=str(payload["mapping_case_sha256"]))
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(payload):
            raise ValueError("ordinary V2 independent canonical rebuild mismatch")


def validate_v2_cross_parent_mapping_case_payload(
    payload: Mapping[str, Any], *, projected_input: Mapping[str, Any] | None = None,
) -> None:
    if set(payload) != TOP_LEVEL_FIELDS:
        raise ValueError("V2 mapping payload top-level schema mismatch")
    if (
        payload.get("schema_version") != PROBE_SCHEMA_VERSION
        or payload.get("kind") != PROBE_KIND
        or payload.get("profile") != PRIMITIVE_PROFILE
    ):
        raise ValueError("V2 mapping payload rejects V1/foreign identity")
    expected_flags = {
        "research_only": True, "inference_safe": True,
        "prediction_mutated": False, "birth_columns_enabled": False,
        "ownership_consumed": False, "solver_consumed": False,
        "truth_inputs_consumed": False, "source_points_consumed": False,
        "source_points_serialized": False, "ordinary_batch_authorized": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "compact_bundle_authorized": False, "gold_open_authorized": False,
    }
    for field, expected in expected_flags.items():
        if payload.get(field) is not expected:
            raise ValueError(f"V2 mapping flag invalid: {field}")
    if payload.get("mapping_preflight_kind") != MAPPING_PREFLIGHT_KIND:
        raise ValueError("V2 mapping preflight kind mismatch")
    if payload.get("target_baseline_relation_rule_id") != TARGET_BASELINE_RELATION_RULE_ID:
        raise ValueError("V2 mapping relation rule mismatch")
    if dict(payload["hard_caps"]) != HARD_CAPS:
        raise ValueError("V2 mapping hard caps mismatch")
    relations = list(payload["frozen_target_baseline_relation_table"])
    outcomes = list(payload["frozen_target_outcome_table"])
    target_projection = list(payload["frozen_target_projection"])
    all_pair_rows = list(payload["all_pair_membership_table"])
    if target_projection != frozen_target_projection(target_projection):
        raise ValueError("V2 frozen target projection order/schema mismatch")
    if any(set(row) != v1_core.FROZEN_TARGET_FIELDS for row in target_projection):
        raise ValueError("V2 frozen target projection schema mismatch")
    if _table_sha(target_projection, "frozen_target_component_id") != payload["frozen_target_table_sha256"]:
        raise ValueError("V2 frozen target projection hash mismatch")
    pair_ids = []
    for row in all_pair_rows:
        if set(row) != ALL_PAIR_MEMBERSHIP_FIELDS:
            raise ValueError("V2 all-pair membership row schema mismatch")
        parent_ids = list(map(str, row["transverse_parent_ids"]))
        if parent_ids != sorted(parent_ids) or len(parent_ids) != len(set(parent_ids)):
            raise ValueError("V2 all-pair transverse parent order/identity mismatch")
        if row["transverse_parent_ids_sha256"] != _sha(parent_ids):
            raise ValueError("V2 all-pair transverse parent hash mismatch")
        pair_ids.append(str(row["baseline_local_plane_pair_id"]))
    if pair_ids != sorted(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("V2 all-pair membership order/identity mismatch")
    if _table_sha(all_pair_rows, "baseline_local_plane_pair_id") != payload["all_pair_membership_table_sha256"]:
        raise ValueError("V2 all-pair membership table hash mismatch")
    if payload["thresholds_sha256"] != _sha(payload["thresholds"]):
        raise ValueError("V2 threshold hash mismatch")
    for table, fields, hash_field in (
        (relations, FROZEN_TARGET_BASELINE_RELATION_FIELDS, "target_baseline_relation_sha256"),
        (outcomes, FROZEN_TARGET_OUTCOME_FIELDS, "target_outcome_sha256"),
    ):
        ids = []
        for row in table:
            if set(row) != fields or row[hash_field] != _record_hash(row, hash_field):
                raise ValueError("V2 mapping row schema/hash mismatch")
            ids.append((str(row["case_token"]), str(row["frozen_target_component_id"])))
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("V2 mapping table order/identity mismatch")
    if _table_sha(relations, "frozen_target_component_id") != payload["frozen_target_baseline_relation_table_sha256"]:
        raise ValueError("V2 relation table hash mismatch")
    if _table_sha(outcomes, "frozen_target_component_id") != payload["frozen_target_outcome_table_sha256"]:
        raise ValueError("V2 outcome table hash mismatch")
    if [row["frozen_target_component_id"] for row in relations] != [row["frozen_target_component_id"] for row in outcomes]:
        raise ValueError("V2 relation/outcome target set mismatch")
    for relation, outcome in zip(relations, outcomes):
        status = relation["target_baseline_relation_status"]
        if status not in TARGET_BASELINE_RELATION_STATUS:
            raise ValueError("unrecognised target-baseline relation status")
        in_tube = relation["target_parent_in_frozen_transverse_set"]
        mapping_count = int(relation["baseline_component_mapping_count"])
        component_id = relation["complete_edge_baseline_component_id"]
        if status == RELATION_COMPONENT_UNIQUE:
            if not in_tube or mapping_count != 1 or not isinstance(component_id, str):
                raise ValueError("in-tube target relation is inconsistent")
        else:
            if in_tube or mapping_count != 0 or component_id is not None:
                raise ValueError("out-of-tube target relation is inconsistent")
            null_fields = (
                "replacement_component_id", "support_parent_count_before",
                "support_parent_count_after", "occupied_bin_count_before",
                "occupied_bin_count_after", "left_side_bin_count_before",
                "left_side_bin_count_after", "right_side_bin_count_before",
                "right_side_bin_count_after", "transition_rung_bin_count_before",
                "transition_rung_bin_count_after", "geometry_gate_depth_before",
                "geometry_gate_depth_after", "first_rejection_code_before",
                "first_rejection_code_after",
            )
            if any(outcome[field] is not None for field in null_fields):
                raise ValueError("out-of-tube outcome geometry must remain null")
            if outcome["target_outcome_code"] != OUTCOME_STRUCTURALLY_UNREACHABLE:
                raise ValueError("out-of-tube outcome code mismatch")
        if (
            int(outcome["causal_added_cross_parent_rung_count"]) != 0
            or outcome["causal_added_cross_parent_rung_ids_sha256"] != _sha([])
            or outcome["geometry_certificate_valid_after"] is not False
            or outcome["target_reached_by_added_rung"] is not False
            or outcome["target_side_gate_advanced"] is not False
            or outcome["target_novel_rung_attributable"] is not False
        ):
            raise ValueError("mapping-only outcome illegally claims rung effect")
    attestation = dict(payload["case_mapping_attestation"])
    if set(attestation) != CASE_ATTESTATION_FIELDS or attestation["case_mapping_attestation_sha256"] != _record_hash(attestation, "case_mapping_attestation_sha256"):
        raise ValueError("V2 case mapping attestation invalid")
    derived = {
        **derive_target_preservation_predicates(relations, target_projection),
        "every_target_mode_has_one_baseline_fragment": all(
            all(fragment_id is not None for _mode, fragment_id in row["baseline_mode_fragment_rows"])
            for row in relations),
        "every_target_maps_to_exactly_two_fragments": all(len(row["baseline_fragment_ids"]) == 2 for row in relations),
        "every_target_has_exactly_one_pair_lineage": all(row["baseline_pair_mapping_count"] == 1 for row in relations),
        "every_in_tube_target_has_one_component_image": all((not row["target_parent_in_frozen_transverse_set"]) or row["baseline_component_mapping_count"] == 1 for row in relations),
        "every_out_of_tube_target_has_zero_component_images": all(row["target_parent_in_frozen_transverse_set"] or row["baseline_component_mapping_count"] == 0 for row in relations),
        "no_unrecognised_relation_status_exists": all(row["target_baseline_relation_status"] in TARGET_BASELINE_RELATION_STATUS for row in relations),
        "relation_table_untruncated": len(relations) == int(payload["case_counters"]["frozen_target_count"]),
        "outcome_table_complete": len(outcomes) == len(relations),
        "all_pair_membership_row_count_matches_baseline": (
            len(all_pair_rows)
            == int(payload["case_counters"]["all_pair_membership_row_count"])
        ),
    }
    if any(attestation[field] is not value for field, value in derived.items()):
        raise ValueError("V2 mapping attestation predicate mismatch")
    if projected_input is not None:
        validated = complete_core._validate_projected_input(projected_input)
        expected_input_hashes = {
            **validated["input_hashes"],
            "global_plane_table_sha256": lpf._table_sha256(
                sorted(validated["planes"], key=lambda plane: lpf._canonical_plane(plane)[2])),
        }
        if dict(payload["input_table_hashes"]) != expected_input_hashes:
            raise ValueError("V2 mapping input hashes mismatch")
        independently_regenerated_control = complete_core.regenerate_frozen_control_probe(
            projected_input, scan_sha256=str(payload["scan_sha256"]),
            candidate_pool_sha256=str(payload["candidate_pool_sha256"]),
            a0_prediction_sha256=str(payload["a0_prediction_sha256"]))
        if _sha(independently_regenerated_control) != payload["frozen_control_probe_sha256"]:
            raise ValueError("V2 independent frozen control reproduction mismatch")
        independently_regenerated_baseline = complete_core.build_v2_complete_same_surface_edge_probe(
            projected_input, scan_sha256=str(payload["scan_sha256"]),
            candidate_pool_sha256=str(payload["candidate_pool_sha256"]),
            a0_prediction_sha256=str(payload["a0_prediction_sha256"]),
            frozen_control_probe_sha256=str(payload["frozen_control_probe_sha256"]))
        if _sha(independently_regenerated_baseline) != payload["complete_edge_baseline_probe_sha256"]:
            raise ValueError("V2 independent complete-edge baseline reproduction mismatch")
        expected_targets = v1_core._frozen_targets(
            independently_regenerated_control, str(payload["case_token"]))
        expected_target_projection = frozen_target_projection(expected_targets)
        if target_projection != expected_target_projection:
            raise ValueError("V2 independent frozen target projection mismatch")
        expected_pair_rows = independently_regenerate_all_pair_membership_rows(
            projected_input, independently_regenerated_baseline)
        if all_pair_rows != expected_pair_rows:
            raise ValueError("V2 independent all-pair membership comparison mismatch")
        if payload["thresholds"] != independently_regenerated_baseline["thresholds"]:
            raise ValueError("V2 independent threshold comparison mismatch")
        if (
            payload["gold_open_authorized"] is not False
            or payload["truth_inputs_consumed"] is not False
        ):
            raise ValueError("V2 independent safety predicate mismatch")
        independent_predicates = {
            "frozen_control_reproduced": True,
            "complete_edge_baseline_reproduced": True,
            "frozen_target_rows_regenerated_from_control": (
                target_projection == expected_target_projection),
            "frozen_target_rows_byte_identical": (
                frozen_target_projection(relations) == expected_target_projection),
            "frozen_target_id_sets_exact": (
                {row["frozen_target_component_id"] for row in relations}
                == {row["frozen_target_component_id"] for row in expected_targets}),
            "frozen_target_mode_hash_sets_exact": (
                {row["frozen_target_mode_ids_sha256"] for row in relations}
                == {row["frozen_target_mode_ids_sha256"] for row in expected_targets}),
            "threshold_hash_unchanged": (
                payload["thresholds_sha256"]
                == _sha(independently_regenerated_baseline["thresholds"])),
            "all_pair_membership_row_count_matches_baseline": (
                len(all_pair_rows)
                == int(independently_regenerated_baseline["case_counters"]["local_plane_pair_count"])),
            "pair_line_tube_membership_unchanged": all_pair_rows == expected_pair_rows,
            "no_rungs_constructed": (
                int(payload["case_counters"]["added_cross_parent_rung_count"]) == 0),
            "ordinary_batch_authorized": False,
            "gold_open_authorized": False,
        }
        if any(attestation[field] is not value for field, value in independent_predicates.items()):
            raise ValueError("V2 independent attestation predicate mismatch")


__all__ = [
    "PRIMITIVE_PROFILE", "PROBE_SCHEMA_VERSION", "PROBE_KIND",
    "MAPPING_PREFLIGHT_KIND", "TARGET_BASELINE_RELATION_RULE_ID",
    "BASELINE_PROFILE", "RUNG_RULE_ID", "CONTROL_RUNG_RULE_ID",
    "TARGET_BASELINE_RELATION_STATUS", "RELATION_COMPONENT_UNIQUE",
    "RELATION_COMPONENT_NONE_OUTSIDE", "FROZEN_TARGET_BASELINE_RELATION_FIELDS",
    "FROZEN_TARGET_OUTCOME_FIELDS", "ALL_PAIR_MEMBERSHIP_FIELDS",
    "TARGET_ELIGIBLE_PAIR_FIELDS", "COMPONENT_TABLE_FIELDS",
    "ORDINARY_V2_CRITERION_KEYS", "DELETION_EVIDENCE_FIELDS",
    "ORDINARY_TOP_LEVEL_FIELDS",
    "CASE_ATTESTATION_FIELDS", "TOP_LEVEL_FIELDS",
    "HARD_CAPS", "canonical_json_bytes", "project_mapping_preflight_input",
    "build_all_pair_membership_rows",
    "independently_regenerate_all_pair_membership_rows",
    "frozen_target_projection", "derive_target_preservation_predicates",
    "build_v2_cross_parent_mapping_case",
    "validate_v2_cross_parent_mapping_case_payload",
    "execute_global_deletion_counterfactual",
    "new_ordinary_v2_criteria",
    "build_v2_cross_parent_ordinary_case",
    "validate_v2_cross_parent_ordinary_case_payload",
]
