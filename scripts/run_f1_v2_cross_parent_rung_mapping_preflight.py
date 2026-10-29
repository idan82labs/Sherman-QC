#!/usr/bin/env python3
"""Run the amended 37-case target-to-baseline mapping-only preflight."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import f1_local_plane_fragment_probe as lpf  # noqa: E402
from backend import f1_v2_complete_same_surface_edge_probe as baseline_core  # noqa: E402
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as rung_core  # noqa: E402
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as mapping_core  # noqa: E402


SCHEMA_VERSION = 1
KIND = "f1_v2_cross_parent_rung_target_baseline_relation_preflight_v1"
PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V2"
RELATION_RULE_ID = "EXACT_FRAGMENT_PAIR_LINEAGE_PARTIAL_COMPONENT_IMAGE_V1"
EXPECTED_CASE_TOKENS = frozenset(f"C{index:03d}" for index in range(1, 38))
PREREGISTERED_INFERENCE_MANIFEST_SHA256 = "bb06613a96a063463bac951cd9a0aa3a31c7c0ce2b894ddab07261abfcae5f66"
PREREGISTERED_V2_POOL_MANIFEST_SHA256 = "c3ed8f6356c49c28ca56670cc9d061fe25dc4f1dea71625886e989330f8df3c7"
FROZEN_CONTROL_MANIFEST_SHA256 = "3b9dfa156746a01cb38195f7d75c92e23bff839228de96e425b8902edf262510"
COMPLETE_EDGE_BASELINE_MANIFEST_SHA256 = "4e26a8638b347a018678be51df1f6bd51b4bee865ed8c4b80155e5c867a758f5"
MAX_WORKERS = 8
MAX_RELATION_ROWS = 4_788
EXPECTED_ALL_PAIR_MEMBERSHIP_ROWS = 16_988
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
RELATION_STATUS_PRESENT = mapping_core.RELATION_COMPONENT_UNIQUE
RELATION_STATUS_ABSENT = mapping_core.RELATION_COMPONENT_NONE_OUTSIDE
OUTCOME_PRESENT = mapping_core.OUTCOME_COMPONENT_PENDING
OUTCOME_ABSENT = mapping_core.OUTCOME_STRUCTURALLY_UNREACHABLE
DECISION_GO = "GO_NEXT_PROFESSOR_PACKET"
DECISION_INVALID = "INVALID_AMENDED_EXPERIMENT_STOP"

RELATION_FIELDS = frozenset(mapping_core.FROZEN_TARGET_BASELINE_RELATION_FIELDS)
OUTCOME_FIELDS = frozenset(mapping_core.FROZEN_TARGET_OUTCOME_FIELDS)
CASE_FIELDS = frozenset({
    "case_token", "status", "scan_sha256", "candidate_pool_sha256",
    "a0_prediction_sha256", "frozen_control_probe_sha256",
    "complete_edge_baseline_probe_sha256", "frozen_target_count",
    "relation_row_count", "outcome_row_count", "component_image_present_count",
    "component_image_absent_outside_tube_count", "relation_table_sha256",
    "outcome_table_sha256", "thresholds_sha256", "pair_tube_membership_sha256",
    "frozen_target_projection_sha256", "regenerated_target_projection_sha256",
    "frozen_thresholds_sha256", "regenerated_thresholds_sha256",
    "all_pair_membership_row_count", "frozen_all_pair_membership_table_sha256",
    "regenerated_all_pair_membership_table_sha256", "frozen_control_reproduced",
    "complete_edge_baseline_reproduced", "frozen_target_projection_equal",
    "thresholds_unchanged", "all_pair_line_tube_memberships_unchanged", "wall_seconds",
    "peak_rss_bytes", "input_json_bytes", "serialized_evidence_bytes",
})
ALL_PAIR_MEMBERSHIP_FIELDS = frozenset(mapping_core.ALL_PAIR_MEMBERSHIP_FIELDS)
AGGREGATE_PAIR_MEMBERSHIP_FIELDS = frozenset({
    "case_token", *ALL_PAIR_MEMBERSHIP_FIELDS,
})
FORBIDDEN_MAPPING_PAYLOAD_KEYS = frozenset({
    "gold", "manual_truth", "sealed_gold", "truth_table", "expected_count",
    "ownership", "birth_columns", "added_cross_parent_rung_table",
    "replay_attestation", "synthetic_resource_attestation", "compact_bundle",
    "solver",
})
MAPPING_PREFLIGHT_CRITERIA = (
    "all_declared_source_hashes_match",
    "all_frozen_control_probe_hashes_reproduce",
    "all_complete_edge_baseline_hashes_reproduce",
    "frozen_target_count_is_4788",
    "frozen_target_rows_are_byte_identical",
    "frozen_target_id_sets_are_exactly_equal",
    "frozen_target_mode_hash_sets_are_exactly_equal",
    "every_target_mode_has_one_baseline_fragment",
    "every_target_maps_to_exactly_two_fragments",
    "every_target_has_exactly_one_pair_lineage",
    "every_in_tube_target_has_one_component_image",
    "every_out_of_tube_target_has_zero_component_images",
    "no_unrecognised_relation_status_exists",
    "relation_row_count_is_4788",
    "relation_table_is_untruncated",
    "outcome_row_count_is_4788",
    "c001_locked_regression_matches",
    "all_threshold_hashes_are_unchanged",
    "all_pair_line_tube_memberships_are_unchanged",
    "gold_open_authorized_false",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _record_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return _sha_json({key: item for key, item in value.items() if key != hash_field})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _rows(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw = payload.get("cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} rows must be objects")
        token = str(row.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} tokens must be unique")
        result[token] = dict(row)
    if set(result) != EXPECTED_CASE_TOKENS:
        raise ValueError(f"{label} must contain exact C001-C037")
    return result


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _frozen_target_projection(
    control: Mapping[str, Any], case_token: str,
) -> list[dict[str, Any]]:
    return mapping_core.frozen_target_projection(
        rung_core._frozen_targets(control, case_token)
    )


def _relation_target_projection(
    relations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return mapping_core.frozen_target_projection(relations)


def build_all_pair_membership_table(
    projected: Mapping[str, Any], baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the complete fixed-tube membership relation for every baseline pair."""

    rows = mapping_core.independently_regenerate_all_pair_membership_rows(
        projected, baseline
    )
    if any(set(row) != ALL_PAIR_MEMBERSHIP_FIELDS for row in rows):
        raise AssertionError("all-pair membership schema drift")
    expected_count = int(baseline["case_counters"]["local_plane_pair_count"])
    expected_pair_ids = sorted(
        str(pair["local_plane_pair_id"])
        for pair in baseline["local_plane_pair_records"]
    )
    actual_pair_ids = [row["baseline_local_plane_pair_id"] for row in rows]
    if (
        len(rows) != expected_count
        or len(set(actual_pair_ids)) != len(rows)
        or actual_pair_ids != expected_pair_ids
    ):
        raise ValueError("complete all-pair membership population mismatch")
    return rows


def derive_target_preservation_predicates(
    *, frozen_targets: Sequence[Mapping[str, Any]],
    regenerated_targets: Sequence[Mapping[str, Any]],
    relation_targets: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    frozen_ids = {str(row["frozen_target_component_id"]) for row in frozen_targets}
    relation_ids = {
        str(row["frozen_target_component_id"]) for row in relation_targets
    }
    frozen_mode_hashes = {
        str(row["frozen_target_mode_ids_sha256"]) for row in frozen_targets
    }
    relation_mode_hashes = {
        str(row["frozen_target_mode_ids_sha256"]) for row in relation_targets
    }
    return {
        "target_rows_byte_identical": list(relation_targets)
        == list(frozen_targets)
        == list(regenerated_targets),
        "target_ids_equal": (
            relation_ids == frozen_ids
            and len(relation_targets) == len(frozen_targets)
            and len(relation_ids) == len(relation_targets)
        ),
        "target_mode_hashes_equal": (
            relation_mode_hashes == frozen_mode_hashes
            and len(relation_targets) == len(frozen_targets)
        ),
    }


def derive_pair_membership_predicate(
    frozen_memberships: Sequence[Mapping[str, Any]],
    regenerated_memberships: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        len(frozen_memberships) == len(regenerated_memberships)
        and len({row["baseline_local_plane_pair_id"] for row in frozen_memberships})
        == len(frozen_memberships)
        and list(frozen_memberships) == list(regenerated_memberships)
    )


def derive_threshold_predicate(
    frozen_thresholds: Mapping[str, Any],
    regenerated_thresholds: Mapping[str, Any],
) -> bool:
    return _canonical_bytes(frozen_thresholds) == _canonical_bytes(
        regenerated_thresholds
    )


def _derive_case_audit(
    *, case_token: str, projected: Mapping[str, Any],
    frozen_control: Mapping[str, Any], regenerated_control: Mapping[str, Any],
    frozen_baseline: Mapping[str, Any], regenerated_baseline: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    relations = [dict(row) for row in payload["frozen_target_baseline_relation_table"]]
    outcomes = [dict(row) for row in payload["frozen_target_outcome_table"]]
    frozen_targets = _frozen_target_projection(frozen_control, case_token)
    regenerated_targets = [
        dict(row) for row in payload["frozen_target_projection"]
    ]
    relation_targets = _relation_target_projection(relations)

    # These are deliberately separate calls over separately sourced baseline
    # payloads.  Equality, not an attestation Boolean, establishes invariance.
    frozen_memberships = build_all_pair_membership_table(projected, frozen_baseline)
    regenerated_memberships = [
        dict(row) for row in payload["all_pair_membership_table"]
    ]

    target_predicates = derive_target_preservation_predicates(
        frozen_targets=frozen_targets,
        regenerated_targets=regenerated_targets,
        relation_targets=relation_targets,
    )
    frozen_thresholds = deepcopy(frozen_baseline["thresholds"])
    regenerated_thresholds = deepcopy(payload["thresholds"])
    counters = dict(payload["case_counters"])
    audit = {
        "frozen_target_count": len(frozen_targets),
        "frozen_control_reproduced": _canonical_bytes(regenerated_control)
        == _canonical_bytes(frozen_control),
        "complete_edge_baseline_reproduced": _canonical_bytes(regenerated_baseline)
        == _canonical_bytes(frozen_baseline),
        **target_predicates,
        "outcome_ids_equal": [
            row["frozen_target_component_id"] for row in outcomes
        ] == [row["frozen_target_component_id"] for row in relations],
        "component_image_present_count": int(
            counters["target_component_image_present_count"]
        ),
        "component_image_absent_outside_tube_count": int(
            counters["target_component_image_absent_outside_tube_count"]
        ),
        "frozen_target_projection_sha256": _sha_json(frozen_targets),
        "regenerated_target_projection_sha256": _sha_json(regenerated_targets),
        "frozen_thresholds_sha256": _sha_json(frozen_thresholds),
        "regenerated_thresholds_sha256": _sha_json(regenerated_thresholds),
        "thresholds_unchanged": derive_threshold_predicate(
            frozen_thresholds, regenerated_thresholds
        ),
        "all_pair_membership_row_count": len(frozen_memberships),
        "frozen_all_pair_membership_table_sha256": _sha_json(frozen_memberships),
        "regenerated_all_pair_membership_table_sha256": _sha_json(
            regenerated_memberships
        ),
        "all_pair_line_tube_memberships_unchanged": derive_pair_membership_predicate(
            frozen_memberships, regenerated_memberships
        ),
    }
    return audit, frozen_memberships, regenerated_memberships


def build_relation_and_outcome_tables(
    *, case_token: str, projected: Mapping[str, Any], control: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the amended mapping rule and derive every audit from source rows."""

    payload = mapping_core._build_case_from_payloads(
        projected,
        case_token=case_token,
        scan_sha256=str(control["scan_sha256"]),
        candidate_pool_sha256=str(control["candidate_pool_sha256"]),
        a0_prediction_sha256=str(control["a0_prediction_sha256"]),
        frozen_control_probe_sha256=_sha_json(control),
        complete_edge_baseline_probe_sha256=_sha_json(baseline),
        frozen_control=control,
        complete_edge_baseline=baseline,
    )
    mapping_core.validate_v2_cross_parent_mapping_case_payload(
        payload, projected_input=projected
    )
    relations = [dict(row) for row in payload["frozen_target_baseline_relation_table"]]
    outcomes = [dict(row) for row in payload["frozen_target_outcome_table"]]
    audit, _memberships, _regenerated_memberships = _derive_case_audit(
        case_token=case_token,
        projected=projected,
        frozen_control=control,
        regenerated_control=deepcopy(control),
        frozen_baseline=baseline,
        regenerated_baseline=deepcopy(baseline),
        payload=payload,
    )
    return relations, outcomes, audit


def _run_case(task: tuple[str, str, str, str, str, str, str, str]) -> dict[str, Any]:
    (
        token, scan_raw, pool_raw, a0_raw, frozen_control_raw,
        frozen_baseline_raw, expected_control_sha, expected_baseline_sha,
    ) = task
    scan, pool, a0 = Path(scan_raw), Path(pool_raw), Path(a0_raw)
    started = perf_counter()
    scan_sha, pool_sha, a0_sha = sha256_file(scan), sha256_file(pool), sha256_file(a0)
    raw_pool = _read(pool)
    projected = mapping_core.project_mapping_preflight_input(raw_pool)
    frozen_control = _read(Path(frozen_control_raw))
    frozen_baseline = _read(Path(frozen_baseline_raw))
    if _sha_json(frozen_control) != expected_control_sha:
        raise ValueError(f"{token}: frozen-control file hash mismatch")
    if _sha_json(frozen_baseline) != expected_baseline_sha:
        raise ValueError(f"{token}: complete-edge baseline file hash mismatch")

    regenerated_control = baseline_core.regenerate_frozen_control_probe(
        deepcopy(projected), scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha, a0_prediction_sha256=a0_sha,
    )
    regenerated_baseline = baseline_core.build_v2_complete_same_surface_edge_probe(
        deepcopy(projected), scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha, a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=expected_control_sha,
    )
    payload = mapping_core._build_case_from_payloads(
        deepcopy(projected), case_token=token,
        scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=expected_control_sha,
        complete_edge_baseline_probe_sha256=expected_baseline_sha,
        frozen_control=regenerated_control,
        complete_edge_baseline=regenerated_baseline,
    )
    mapping_core.validate_v2_cross_parent_mapping_case_payload(
        payload, projected_input=projected
    )
    relations = [dict(row) for row in payload["frozen_target_baseline_relation_table"]]
    outcomes = [dict(row) for row in payload["frozen_target_outcome_table"]]
    audit, memberships, regenerated_memberships = _derive_case_audit(
        case_token=token,
        projected=projected,
        frozen_control=frozen_control,
        regenerated_control=regenerated_control,
        frozen_baseline=frozen_baseline,
        regenerated_baseline=regenerated_baseline,
        payload=payload,
    )
    aggregate_memberships = [
        {"case_token": token, **deepcopy(row)} for row in memberships
    ]
    aggregate_regenerated_memberships = [
        {"case_token": token, **deepcopy(row)}
        for row in regenerated_memberships
    ]
    if any(
        set(row) != AGGREGATE_PAIR_MEMBERSHIP_FIELDS
        for row in aggregate_memberships + aggregate_regenerated_memberships
    ):
        raise AssertionError("aggregate all-pair membership schema drift")
    frozen_target_projection = _frozen_target_projection(frozen_control, token)
    control_sha = str(payload["frozen_control_probe_sha256"])
    baseline_sha = str(payload["complete_edge_baseline_probe_sha256"])
    deterministic_evidence = {
        "case_token": token,
        "scan_sha256": scan_sha,
        "candidate_pool_sha256": pool_sha,
        "a0_prediction_sha256": a0_sha,
        "frozen_control_probe_sha256": control_sha,
        "complete_edge_baseline_probe_sha256": baseline_sha,
        "relations": relations,
        "outcomes": outcomes,
        "all_pair_memberships": aggregate_memberships,
        "regenerated_all_pair_memberships": aggregate_regenerated_memberships,
        "frozen_target_projection": frozen_target_projection,
        "audit": audit,
    }
    peak = _peak_rss_bytes()
    if peak >= PEAK_RSS_LIMIT_BYTES:
        raise MemoryError(f"{token}: mapping preflight RSS cap breached")
    case = {
        "case_token": token,
        "status": "SUCCESS",
        "scan_sha256": scan_sha,
        "candidate_pool_sha256": pool_sha,
        "a0_prediction_sha256": a0_sha,
        "frozen_control_probe_sha256": control_sha,
        "complete_edge_baseline_probe_sha256": baseline_sha,
        "frozen_target_count": audit["frozen_target_count"],
        "relation_row_count": len(relations),
        "outcome_row_count": len(outcomes),
        "component_image_present_count": audit["component_image_present_count"],
        "component_image_absent_outside_tube_count": audit["component_image_absent_outside_tube_count"],
        "relation_table_sha256": _sha_json(relations),
        "outcome_table_sha256": _sha_json(outcomes),
        "thresholds_sha256": audit["frozen_thresholds_sha256"],
        "pair_tube_membership_sha256": audit[
            "frozen_all_pair_membership_table_sha256"
        ],
        "frozen_target_projection_sha256": audit[
            "frozen_target_projection_sha256"
        ],
        "regenerated_target_projection_sha256": audit[
            "regenerated_target_projection_sha256"
        ],
        "frozen_thresholds_sha256": audit["frozen_thresholds_sha256"],
        "regenerated_thresholds_sha256": audit["regenerated_thresholds_sha256"],
        "all_pair_membership_row_count": audit["all_pair_membership_row_count"],
        "frozen_all_pair_membership_table_sha256": audit[
            "frozen_all_pair_membership_table_sha256"
        ],
        "regenerated_all_pair_membership_table_sha256": audit[
            "regenerated_all_pair_membership_table_sha256"
        ],
        "frozen_control_reproduced": audit["frozen_control_reproduced"],
        "complete_edge_baseline_reproduced": audit[
            "complete_edge_baseline_reproduced"
        ],
        "frozen_target_projection_equal": audit["target_rows_byte_identical"],
        "thresholds_unchanged": audit["thresholds_unchanged"],
        "all_pair_line_tube_memberships_unchanged": audit["all_pair_line_tube_memberships_unchanged"],
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": peak,
        "input_json_bytes": pool.stat().st_size,
        "serialized_evidence_bytes": len(_canonical_bytes(deterministic_evidence)),
    }
    if set(case) != CASE_FIELDS:
        raise AssertionError("mapping-preflight case schema drift")
    return {
        "case": case,
        "relations": relations,
        "outcomes": outcomes,
        "memberships": aggregate_memberships,
        "regenerated_memberships": aggregate_regenerated_memberships,
        "frozen_target_projection": frozen_target_projection,
        "audit": audit,
    }


def _c001_locked_regression(rows: Sequence[Mapping[str, Any]]) -> bool:
    matches = [row for row in rows if row["case_token"] == "C001" and row["frozen_target_component_id"] == "LTC_32d3609a79a9c314"]
    if len(matches) != 1:
        return False
    row = matches[0]
    return bool(
        row["frozen_target_pair_id"] == "LPP_daf852ef51c06429"
        and row["frozen_target_parent_id"] == "P_47b0d4bd56e0f8ce"
        and row["frozen_target_mode_ids"] == ["M_73e7f7dd53e69d0f", "M_d5d9b9a524d05ccc"]
        and row["baseline_fragment_ids"] == ["VEF_6fa481eb4e1f5bc8", "VEF_ca3d2d1302651a00"]
        and row["baseline_pair_mapping_count"] == 1
        and row["baseline_local_plane_pair_id"] == "VEP_7610cbf5af1e3558"
        and row["baseline_pair_sha256"] == "e48e7f2d53e2885a6bbc5e5af9256c81e1761b3b0e0bfe9efd104f9b65f805b2"
        and row["target_parent_line_distance_mm"] == 7.421545400596292
        and row["line_tube_tolerance_mm"] == 3.0
        and row["target_parent_in_frozen_transverse_set"] is False
        and row["baseline_component_mapping_count"] == 0
        and row["complete_edge_baseline_component_id"] is None
        and row["target_baseline_relation_status"] == RELATION_STATUS_ABSENT
    )


def _contains_forbidden_mapping_payload_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_MAPPING_PAYLOAD_KEYS:
                return True
            if _contains_forbidden_mapping_payload_key(nested):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_forbidden_mapping_payload_key(item) for item in value)
    return False


def derive_mapping_only_safety_predicate(artifact: Mapping[str, Any]) -> bool:
    outcomes = artifact.get("frozen_target_outcome_table")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        return False
    return bool(
        artifact.get("gold_open_authorized") is False
        and artifact.get("truth_inputs_consumed") is False
        and artifact.get("ordinary_batch_authorized") is False
        and not _contains_forbidden_mapping_payload_key(artifact)
        and all(
            isinstance(row, Mapping)
            and row.get("causal_added_cross_parent_rung_count") == 0
            and row.get("target_reached_by_added_rung") is False
            and row.get("target_side_gate_advanced") is False
            and row.get("target_novel_rung_attributable") is False
            for row in outcomes
        )
    )


def _build_criteria(
    *, cases, relations, outcomes, audits, source_hashes_match: bool,
    safety_predicate: bool,
) -> dict[str, bool]:
    criteria = {key: False for key in MAPPING_PREFLIGHT_CRITERIA}
    all_success = len(cases) == 37 and all(case["status"] == "SUCCESS" for case in cases)
    statuses = {row["target_baseline_relation_status"] for row in relations}
    criteria.update({
        "all_declared_source_hashes_match": source_hashes_match and all_success,
        "all_frozen_control_probe_hashes_reproduce": all_success
        and all(audit["frozen_control_reproduced"] for audit in audits),
        "all_complete_edge_baseline_hashes_reproduce": all_success
        and all(audit["complete_edge_baseline_reproduced"] for audit in audits),
        "frozen_target_count_is_4788": len(relations) == MAX_RELATION_ROWS and sum(case["frozen_target_count"] for case in cases) == MAX_RELATION_ROWS,
        "frozen_target_rows_are_byte_identical": all(audit["target_rows_byte_identical"] for audit in audits),
        "frozen_target_id_sets_are_exactly_equal": all(audit["target_ids_equal"] for audit in audits),
        "frozen_target_mode_hash_sets_are_exactly_equal": all(audit["target_mode_hashes_equal"] for audit in audits),
        "every_target_mode_has_one_baseline_fragment": all(all(fragment_id is not None for _mode_id, fragment_id in row["baseline_mode_fragment_rows"]) for row in relations),
        "every_target_maps_to_exactly_two_fragments": all(len(row["baseline_fragment_ids"]) == 2 for row in relations),
        "every_target_has_exactly_one_pair_lineage": all(row["baseline_pair_mapping_count"] == 1 for row in relations),
        "every_in_tube_target_has_one_component_image": all((not row["target_parent_in_frozen_transverse_set"]) or row["baseline_component_mapping_count"] == 1 for row in relations),
        "every_out_of_tube_target_has_zero_component_images": all(row["target_parent_in_frozen_transverse_set"] or row["baseline_component_mapping_count"] == 0 for row in relations),
        "no_unrecognised_relation_status_exists": statuses <= {RELATION_STATUS_PRESENT, RELATION_STATUS_ABSENT},
        "relation_row_count_is_4788": len(relations) == MAX_RELATION_ROWS,
        "relation_table_is_untruncated": len(relations) == MAX_RELATION_ROWS and len({row["frozen_target_component_id"] for row in relations}) == MAX_RELATION_ROWS,
        "outcome_row_count_is_4788": len(outcomes) == MAX_RELATION_ROWS and all(audit["outcome_ids_equal"] for audit in audits),
        "c001_locked_regression_matches": _c001_locked_regression(relations),
        "all_threshold_hashes_are_unchanged": all_success
        and all(
            audit["thresholds_unchanged"]
            and audit["frozen_thresholds_sha256"]
            == audit["regenerated_thresholds_sha256"]
            for audit in audits
        ),
        "all_pair_line_tube_memberships_are_unchanged": all_success
        and sum(audit["all_pair_membership_row_count"] for audit in audits)
        == EXPECTED_ALL_PAIR_MEMBERSHIP_ROWS
        and all(
            audit["all_pair_line_tube_memberships_unchanged"]
            and audit["frozen_all_pair_membership_table_sha256"]
            == audit["regenerated_all_pair_membership_table_sha256"]
            for audit in audits
        ),
        "gold_open_authorized_false": safety_predicate,
    })
    return criteria


def mapping_preflight_decision(criteria: Mapping[str, Any]) -> tuple[str, bool]:
    valid = (
        set(criteria) == set(MAPPING_PREFLIGHT_CRITERIA)
        and all(criteria[key] is True for key in MAPPING_PREFLIGHT_CRITERIA)
    )
    return (DECISION_GO, True) if valid else (DECISION_INVALID, False)


def run_mapping_preflight(
    *, inference_manifest_path: Path, candidate_pool_manifest_path: Path,
    a0_prediction_root: Path, frozen_control_manifest_path: Path,
    complete_edge_baseline_manifest_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    inference_manifest_path = inference_manifest_path.resolve()
    candidate_pool_manifest_path = candidate_pool_manifest_path.resolve()
    a0_prediction_root = a0_prediction_root.resolve()
    frozen_control_manifest_path = frozen_control_manifest_path.resolve()
    complete_edge_baseline_manifest_path = complete_edge_baseline_manifest_path.resolve()
    output_path = output_path.resolve()
    source_hashes = {
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "candidate_pool_manifest_sha256": sha256_file(candidate_pool_manifest_path),
        "frozen_control_manifest_sha256": sha256_file(frozen_control_manifest_path),
        "complete_edge_baseline_manifest_sha256": sha256_file(complete_edge_baseline_manifest_path),
    }
    expected_source_hashes = {
        "inference_manifest_sha256": PREREGISTERED_INFERENCE_MANIFEST_SHA256,
        "candidate_pool_manifest_sha256": PREREGISTERED_V2_POOL_MANIFEST_SHA256,
        "frozen_control_manifest_sha256": FROZEN_CONTROL_MANIFEST_SHA256,
        "complete_edge_baseline_manifest_sha256": COMPLETE_EDGE_BASELINE_MANIFEST_SHA256,
    }
    if source_hashes != expected_source_hashes:
        raise ValueError("declared mapping-preflight source manifest hash mismatch")
    inference_rows = _rows(_read(inference_manifest_path), "inference manifest")
    pool_rows = _rows(_read(candidate_pool_manifest_path), "candidate-pool manifest")
    control_rows = _rows(_read(frozen_control_manifest_path), "frozen-control manifest")
    baseline_rows = _rows(_read(complete_edge_baseline_manifest_path), "complete-edge baseline manifest")
    tasks = []
    for token in sorted(EXPECTED_CASE_TOKENS):
        scan = Path(str(inference_rows[token]["source_scan_path"])).resolve()
        pool = Path(str(pool_rows[token]["path"])).resolve()
        a0 = a0_prediction_root / "cases" / token / "prediction_owned_regions.json"
        frozen_control = Path(str(control_rows[token]["probe_path"])).resolve()
        frozen_baseline = complete_edge_baseline_manifest_path.parent / f"{token}.json"
        tasks.append((
            token, str(scan), str(pool), str(a0), str(frozen_control),
            str(frozen_baseline), str(control_rows[token]["probe_sha256"]),
            str(baseline_rows[token]["replacement_probe_sha256"]),
        ))
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        results = list(executor.map(_run_case, tasks))
    results.sort(key=lambda item: item["case"]["case_token"])
    cases = [item["case"] for item in results]
    relations = sorted((row for item in results for row in item["relations"]), key=lambda row: (row["case_token"], row["frozen_target_component_id"]))
    outcomes = sorted((row for item in results for row in item["outcomes"]), key=lambda row: (row["case_token"], row["frozen_target_component_id"]))
    memberships = sorted(
        (row for item in results for row in item["memberships"]),
        key=lambda row: (row["case_token"], row["baseline_local_plane_pair_id"]),
    )
    regenerated_memberships = sorted(
        (row for item in results for row in item["regenerated_memberships"]),
        key=lambda row: (row["case_token"], row["baseline_local_plane_pair_id"]),
    )
    frozen_target_projection = sorted(
        (row for item in results for row in item["frozen_target_projection"]),
        key=lambda row: (row["case_token"], row["frozen_target_component_id"]),
    )
    audits = [item["audit"] for item in results]
    threshold_hash_table = [
        {
            "case_token": case["case_token"],
            "thresholds_sha256": case["frozen_thresholds_sha256"],
        }
        for case in cases
    ]
    regenerated_threshold_hash_table = [
        {
            "case_token": case["case_token"],
            "thresholds_sha256": case["regenerated_thresholds_sha256"],
        }
        for case in cases
    ]
    artifact_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "profile": PROFILE,
        "target_baseline_relation_rule_id": RELATION_RULE_ID,
        "research_only": True,
        "inference_safe": True,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "truth_inputs_consumed": False,
        "ordinary_batch_authorized": False,
        "gold_open_authorized": False,
        "case_count": len(cases),
        "successful_case_count": sum(case["status"] == "SUCCESS" for case in cases),
        "workers": workers,
        "isolated_process_per_case": True,
        **source_hashes,
        "target_row_count": len(relations),
        "target_component_image_present_count": sum(row["target_baseline_relation_status"] == RELATION_STATUS_PRESENT for row in relations),
        "target_component_image_absent_outside_tube_count": sum(row["target_baseline_relation_status"] == RELATION_STATUS_ABSENT for row in relations),
        "relation_table_sha256": _sha_json(relations),
        "outcome_table_sha256": _sha_json(outcomes),
        "frozen_target_projection_sha256": _sha_json(frozen_target_projection),
        "regenerated_frozen_target_projection_sha256": _sha_json(
            _relation_target_projection(relations)
        ),
        "all_pair_membership_row_count": len(memberships),
        "all_pair_membership_table_sha256": _sha_json(memberships),
        "regenerated_all_pair_membership_table_sha256": _sha_json(
            regenerated_memberships
        ),
        "threshold_hash_table_sha256": _sha_json(threshold_hash_table),
        "regenerated_threshold_hash_table_sha256": _sha_json(
            regenerated_threshold_hash_table
        ),
        "maximum_peak_rss_bytes": max(case["peak_rss_bytes"] for case in cases),
        "cases": cases,
        "frozen_target_projection": frozen_target_projection,
        "all_pair_membership_table": memberships,
        "threshold_hash_table": threshold_hash_table,
        "frozen_target_baseline_relation_table": relations,
        "frozen_target_outcome_table": outcomes,
    }
    safety_predicate = derive_mapping_only_safety_predicate(artifact_base)
    criteria = _build_criteria(
        cases=cases, relations=relations, outcomes=outcomes, audits=audits,
        source_hashes_match=source_hashes == expected_source_hashes,
        safety_predicate=safety_predicate,
    )
    decision, next_packet_authorized = mapping_preflight_decision(criteria)
    artifact = {
        **artifact_base,
        "mapping_preflight_criteria": criteria,
        "next_professor_packet_authorized": next_packet_authorized,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(artifact) + b"\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_mapping_preflight(
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
        output_path=args.output,
        workers=args.workers,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "decision": result["decision"],
        "target_row_count": result["target_row_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
