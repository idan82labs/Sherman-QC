#!/usr/bin/env python3
"""Validate and compact the amended 4,788-row mapping-only preflight."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (  # noqa: E402
    CASE_FIELDS,
    DECISION_GO,
    KIND as PREFLIGHT_KIND,
    MAPPING_PREFLIGHT_CRITERIA,
    OUTCOME_ABSENT,
    OUTCOME_FIELDS,
    OUTCOME_PRESENT,
    PROFILE,
    RELATION_FIELDS,
    RELATION_RULE_ID,
    RELATION_STATUS_ABSENT,
    RELATION_STATUS_PRESENT,
    _c001_locked_regression,
)
from backend import (  # noqa: E402
    f1_v2_cross_parent_opposite_fragment_rung_probe as v1_rung_core,
)
from backend import (  # noqa: E402
    f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as v2_mapping_core,
)


SCHEMA_VERSION = 2
KIND = "f1_v2_cross_parent_rung_mapping_preflight_compact_evidence_v2"
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
EXPECTED_TARGET_ROWS = 4_788
EXPECTED_ALL_PAIR_ROWS = 16_988
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

PREFLIGHT_TOP_FIELDS = {
    "schema_version", "kind", "profile", "target_baseline_relation_rule_id",
    "research_only", "inference_safe", "source_points_consumed",
    "source_points_serialized", "truth_inputs_consumed",
    "ordinary_batch_authorized", "gold_open_authorized",
    "next_professor_packet_authorized", "decision", "case_count",
    "successful_case_count", "workers", "isolated_process_per_case",
    "inference_manifest_sha256", "candidate_pool_manifest_sha256",
    "frozen_control_manifest_sha256",
    "complete_edge_baseline_manifest_sha256", "target_row_count",
    "target_component_image_present_count",
    "target_component_image_absent_outside_tube_count",
    "relation_table_sha256", "outcome_table_sha256",
    "frozen_target_projection_sha256",
    "regenerated_frozen_target_projection_sha256",
    "all_pair_membership_row_count", "all_pair_membership_table_sha256",
    "regenerated_all_pair_membership_table_sha256",
    "threshold_hash_table_sha256", "regenerated_threshold_hash_table_sha256",
    "mapping_preflight_criteria", "maximum_peak_rss_bytes", "cases",
    "frozen_target_projection", "all_pair_membership_table",
    "threshold_hash_table",
    "frozen_target_baseline_relation_table", "frozen_target_outcome_table",
}

FROZEN_TARGET_FIELDS = (
    "case_token", "frozen_target_component_id", "frozen_target_pair_id",
    "frozen_target_parent_id", "frozen_target_mode_ids",
    "frozen_target_mode_ids_sha256", "frozen_target_sha256",
)

ALL_PAIR_MEMBERSHIP_FIELDS = (
    "baseline_local_plane_pair_id", "transverse_parent_ids",
    "transverse_parent_ids_sha256",
)

FORBIDDEN_ARTIFACT_KEYS = {
    "gold", "manual_truth", "sealed_gold", "expected_count", "ownership",
    "birth_columns", "added_cross_parent_rung_table", "replay_attestation",
    "synthetic_resource_attestation", "compact_bundle", "solver",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _record_hash(row: Mapping[str, Any], field: str) -> str:
    return _sha_json({key: value for key, value in row.items() if key != field})


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _walk_forbidden(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_ARTIFACT_KEYS:
                raise ValueError(f"forbidden mapping-preflight key at {path}.{key}")
            _walk_forbidden(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _walk_forbidden(nested, f"{path}[{index}]")


def _manifest_rows(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw = payload.get("cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    rows: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} rows must be objects")
        token = str(item.get("case_token") or "")
        if not token or token in rows:
            raise ValueError(f"{label} tokens must be nonempty and unique")
        rows[token] = dict(item)
    if set(rows) != set(EXPECTED_CASE_TOKENS):
        raise ValueError(f"{label} must contain exact C001-C037")
    return rows


def _exact_frozen_target_projection(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the immutable target fields without trusting packet attestations."""
    projected = [
        {field: deepcopy(row[field]) for field in FROZEN_TARGET_FIELDS}
        for row in rows
    ]
    projected.sort(
        key=lambda row: (
            str(row["case_token"]), str(row["frozen_target_component_id"])
        )
    )
    return projected


def _derive_source_case_evidence(
    *, case_token: str, candidate_pool_path: Path,
    frozen_control_path: Path, complete_edge_baseline_path: Path,
) -> dict[str, Any]:
    """Independently reconstruct targets, all pairs, thresholds, and safety."""
    raw_pool = _read(candidate_pool_path)
    frozen_control = _read(frozen_control_path)
    baseline = _read(complete_edge_baseline_path)
    projected_input = v2_mapping_core.project_mapping_preflight_input(raw_pool)
    frozen_targets = v1_rung_core._frozen_targets(frozen_control, case_token)
    target_projection = _exact_frozen_target_projection(frozen_targets)
    context = v1_rung_core._baseline_context(projected_input, baseline)
    pair_rows = []
    for pair_id, pair in sorted(context["pair_by_id"].items()):
        pair_context = v1_rung_core._pair_context(pair, context)
        parent_ids = sorted(map(str, pair_context["transverse_parent_ids"]))
        pair_rows.append({
            "baseline_local_plane_pair_id": str(pair_id),
            "transverse_parent_ids": parent_ids,
            "transverse_parent_ids_sha256": _sha_json(parent_ids),
        })
    if len(pair_rows) != int(baseline["case_counters"]["local_plane_pair_count"]):
        raise ValueError(f"{case_token}: regenerated all-pair census mismatch")
    if len({row["baseline_local_plane_pair_id"] for row in pair_rows}) != len(pair_rows):
        raise ValueError(f"{case_token}: regenerated pair IDs are not unique")
    for field, expected in {
        "research_only": True, "inference_safe": True,
        "prediction_mutated": False, "birth_columns_enabled": False,
        "ownership_consumed": False, "solver_consumed": False,
        "truth_inputs_consumed": False, "source_points_consumed": False,
        "source_points_serialized": False,
    }.items():
        if baseline.get(field) is not expected or frozen_control.get(field) is not expected:
            raise ValueError(f"{case_token}: frozen source safety mismatch: {field}")
    return {
        "frozen_target_projection": target_projection,
        "frozen_target_ids": {
            str(row["frozen_target_component_id"]) for row in target_projection
        },
        "frozen_target_mode_hashes": {
            str(row["frozen_target_mode_ids_sha256"]) for row in target_projection
        },
        "all_pair_membership_table": pair_rows,
        "all_pair_membership_table_sha256": _sha_json(pair_rows),
        "thresholds_sha256": _sha_json(baseline["thresholds"]),
        "gold_open_authorized_false": True,
    }


def _assert_exact_target_preservation(
    relation_rows: Sequence[Mapping[str, Any]],
    expected_projection: Sequence[Mapping[str, Any]],
) -> None:
    actual_projection = _exact_frozen_target_projection(relation_rows)
    expected = [dict(row) for row in expected_projection]
    if {
        str(row["frozen_target_component_id"]) for row in actual_projection
    } != {
        str(row["frozen_target_component_id"]) for row in expected
    }:
        raise ValueError("frozen target ID sets are not exactly equal")
    if {
        str(row["frozen_target_mode_ids_sha256"]) for row in actual_projection
    } != {
        str(row["frozen_target_mode_ids_sha256"]) for row in expected
    }:
        raise ValueError("frozen target mode-hash sets are not exactly equal")
    if actual_projection != expected:
        raise ValueError("frozen target projection is not byte-identical")


def _assert_exact_all_pair_memberships(
    actual_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    declared_sha256: str,
) -> None:
    actual = [dict(row) for row in actual_rows]
    expected = [dict(row) for row in expected_rows]
    for row in actual:
        if set(row) != set(ALL_PAIR_MEMBERSHIP_FIELDS):
            raise ValueError("all-pair membership row schema mismatch")
        parent_ids = list(row["transverse_parent_ids"])
        if parent_ids != sorted(set(map(str, parent_ids))):
            raise ValueError("all-pair transverse parent IDs are not canonical")
        if row["transverse_parent_ids_sha256"] != _sha_json(parent_ids):
            raise ValueError("all-pair transverse parent hash mismatch")
    if actual != expected:
        raise ValueError("complete all-pair membership table mismatch")
    if _require_sha(declared_sha256, "all-pair membership table SHA") != _sha_json(actual):
        raise ValueError("complete all-pair membership table SHA mismatch")


def _assert_exact_threshold_hash(actual_sha256: str, expected_sha256: str) -> None:
    if _require_sha(actual_sha256, "threshold SHA") != _require_sha(
        expected_sha256, "expected threshold SHA"
    ):
        raise ValueError("threshold hash changed")


def _assert_criteria_match_derived(
    asserted: Mapping[str, Any], derived: Mapping[str, bool],
) -> None:
    for key, value in derived.items():
        if value is not True or asserted.get(key) is not value:
            raise ValueError(f"asserted criterion is not source-derived: {key}")


def _validate_relation(row: Mapping[str, Any]) -> None:
    if set(row) != set(RELATION_FIELDS):
        raise ValueError("mapping relation row schema mismatch")
    if _require_sha(
        row.get("target_baseline_relation_sha256"), "relation row SHA"
    ) != _record_hash(row, "target_baseline_relation_sha256"):
        raise ValueError("mapping relation row SHA mismatch")
    modes = list(row.get("frozen_target_mode_ids") or ())
    if not modes or modes != sorted(set(str(value) for value in modes)):
        raise ValueError("frozen target mode IDs must be sorted and unique")
    if _require_sha(row.get("frozen_target_mode_ids_sha256"), "target mode SHA") != _sha_json(modes):
        raise ValueError("frozen target mode-set SHA mismatch")
    frozen_target = {field: deepcopy(row[field]) for field in FROZEN_TARGET_FIELDS[:-1]}
    if _require_sha(row.get("frozen_target_sha256"), "frozen target SHA") != _sha_json(frozen_target):
        raise ValueError("frozen target record SHA mismatch")
    mode_fragments = list(row.get("baseline_mode_fragment_rows") or ())
    if (
        len(mode_fragments) != len(modes)
        or [str(item[0]) for item in mode_fragments] != modes
        or any(len(item) != 2 or item[1] is None for item in mode_fragments)
    ):
        raise ValueError("baseline mode-fragment mapping is incomplete")
    if _require_sha(row.get("baseline_mode_fragment_rows_sha256"), "mode-fragment SHA") != _sha_json(mode_fragments):
        raise ValueError("baseline mode-fragment table SHA mismatch")
    fragments = list(row.get("baseline_fragment_ids") or ())
    if fragments != sorted(set(str(item[1]) for item in mode_fragments)) or len(fragments) != 2:
        raise ValueError("target does not map to exactly two fragments")
    if _require_sha(row.get("baseline_fragment_ids_sha256"), "fragment-set SHA") != _sha_json(fragments):
        raise ValueError("baseline fragment-set SHA mismatch")
    if int(row.get("baseline_pair_mapping_count") or -1) != 1:
        raise ValueError("target pair lineage is not exactly one")
    _require_sha(row.get("baseline_pair_sha256"), "baseline pair SHA")
    _require_sha(row.get("baseline_transverse_parent_ids_sha256"), "transverse-parent SHA")
    distance = float(row.get("target_parent_line_distance_mm"))
    tube = float(row.get("line_tube_tolerance_mm"))
    if not math.isfinite(distance) or not math.isfinite(tube) or tube <= 0.0:
        raise ValueError("target line-distance diagnostic is invalid")
    status = row.get("target_baseline_relation_status")
    in_tube = row.get("target_parent_in_frozen_transverse_set")
    mapping_count = int(row.get("baseline_component_mapping_count") or 0)
    component_id = row.get("complete_edge_baseline_component_id")
    if status == RELATION_STATUS_PRESENT:
        if in_tube is not True or mapping_count != 1 or not isinstance(component_id, str) or not component_id:
            raise ValueError("in-tube target component-image predicate mismatch")
    elif status == RELATION_STATUS_ABSENT:
        if in_tube is not False or mapping_count != 0 or component_id is not None:
            raise ValueError("out-of-tube target component-image predicate mismatch")
    else:
        raise ValueError("unrecognised target-baseline relation status")


def _validate_outcome(
    row: Mapping[str, Any], relation: Mapping[str, Any]
) -> None:
    if set(row) != set(OUTCOME_FIELDS):
        raise ValueError("mapping outcome row schema mismatch")
    if _require_sha(row.get("target_outcome_sha256"), "outcome row SHA") != _record_hash(
        row, "target_outcome_sha256"
    ):
        raise ValueError("mapping outcome row SHA mismatch")
    for field in (
        "case_token", "frozen_target_component_id",
        "frozen_target_mode_ids_sha256", "baseline_local_plane_pair_id",
        "target_baseline_relation_status", "complete_edge_baseline_component_id",
    ):
        if row.get(field) != relation.get(field):
            raise ValueError("mapping outcome/relation identity mismatch")
    expected_code = (
        OUTCOME_PRESENT
        if relation["target_baseline_relation_status"] == RELATION_STATUS_PRESENT
        else OUTCOME_ABSENT
    )
    if row.get("target_outcome_code") != expected_code:
        raise ValueError("mapping outcome code mismatch")
    if row.get("replacement_component_id") is not None:
        raise ValueError("mapping-only outcome cannot claim a replacement component")
    if int(row.get("causal_added_cross_parent_rung_count") or 0) != 0:
        raise ValueError("mapping-only outcome cannot claim a causal rung")
    if _require_sha(
        row.get("causal_added_cross_parent_rung_ids_sha256"),
        "causal-rung ID-set SHA",
    ) != _sha_json([]):
        raise ValueError("mapping-only outcome causal-rung set must be empty")
    if row.get("geometry_certificate_valid_after") is not False:
        raise ValueError("mapping-only outcome cannot claim an after certificate")
    for field in (
        "target_reached_by_added_rung", "target_side_gate_advanced",
        "target_novel_rung_attributable",
    ):
        if row.get(field) is not False:
            raise ValueError("mapping-only outcome cannot receive rung credit")
    before_integer_fields = (
        "support_parent_count_before", "occupied_bin_count_before",
        "left_side_bin_count_before", "right_side_bin_count_before",
        "transition_rung_bin_count_before", "geometry_gate_depth_before",
    )
    after_fields = (
        "support_parent_count_after", "occupied_bin_count_after",
        "left_side_bin_count_after", "right_side_bin_count_after",
        "transition_rung_bin_count_after", "geometry_gate_depth_after",
        "first_rejection_code_after",
    )
    if any(row.get(field) is not None for field in after_fields):
        raise ValueError("mapping-only outcome after geometry must remain null")
    if relation["target_baseline_relation_status"] == RELATION_STATUS_ABSENT:
        absent_null_fields = before_integer_fields + ("first_rejection_code_before",)
        if any(row.get(field) is not None for field in absent_null_fields):
            raise ValueError(
                "structurally unreachable outcome geometry must be null, not zero"
            )
    else:
        if any(
            not isinstance(row.get(field), int) or isinstance(row.get(field), bool)
            for field in before_integer_fields
        ):
            raise ValueError("present component outcome requires baseline integer geometry")
        rejection = row.get("first_rejection_code_before")
        if rejection is not None and not isinstance(rejection, str):
            raise ValueError("baseline first-rejection code must be a string or null")


def validate_and_compact(
    *,
    preflight_path: Path,
    expected_preflight_sha256: str,
    inference_manifest_path: Path,
    candidate_pool_manifest_path: Path,
    a0_prediction_root: Path,
    frozen_control_manifest_path: Path,
    complete_edge_baseline_manifest_path: Path,
) -> dict[str, Any]:
    preflight_path = preflight_path.resolve()
    actual_preflight_sha = _sha_file(preflight_path)
    if actual_preflight_sha != _require_sha(
        expected_preflight_sha256, "expected mapping-preflight SHA"
    ):
        raise ValueError("mapping-preflight external SHA mismatch")
    artifact = _read(preflight_path)
    if set(artifact) != PREFLIGHT_TOP_FIELDS:
        raise ValueError("mapping-preflight top-level schema mismatch")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("kind") != PREFLIGHT_KIND
        or artifact.get("profile") != PROFILE
        or artifact.get("target_baseline_relation_rule_id") != RELATION_RULE_ID
    ):
        raise ValueError("mapping-preflight identity mismatch; V1 is rejected")
    _walk_forbidden(artifact)
    for field, expected in {
        "research_only": True,
        "inference_safe": True,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "truth_inputs_consumed": False,
        "ordinary_batch_authorized": False,
        "gold_open_authorized": False,
        "next_professor_packet_authorized": True,
    }.items():
        if artifact.get(field) is not expected:
            raise ValueError(f"mapping-preflight safety/authorization mismatch: {field}")
    if artifact.get("decision") != DECISION_GO:
        raise ValueError("only a successful mapping preflight can be compacted")
    if (
        int(artifact.get("case_count") or -1) != 37
        or int(artifact.get("successful_case_count") or -1) != 37
        or int(artifact.get("workers") or -1) not in range(1, 9)
        or artifact.get("isolated_process_per_case") is not True
    ):
        raise ValueError("mapping-preflight case/isolation contract mismatch")

    source_paths = {
        "inference_manifest_sha256": inference_manifest_path.resolve(),
        "candidate_pool_manifest_sha256": candidate_pool_manifest_path.resolve(),
        "frozen_control_manifest_sha256": frozen_control_manifest_path.resolve(),
        "complete_edge_baseline_manifest_sha256": complete_edge_baseline_manifest_path.resolve(),
    }
    for field, path in source_paths.items():
        if _sha_file(path) != _require_sha(artifact.get(field), f"preflight {field}"):
            raise ValueError(f"mapping-preflight source hash mismatch: {field}")
    inference_rows = _manifest_rows(_read(inference_manifest_path), "inference manifest")
    pool_rows = _manifest_rows(_read(candidate_pool_manifest_path), "candidate-pool manifest")
    control_rows = _manifest_rows(_read(frozen_control_manifest_path), "frozen-control manifest")
    baseline_rows = _manifest_rows(
        _read(complete_edge_baseline_manifest_path), "complete-edge baseline manifest"
    )
    case_rows = _manifest_rows(artifact, "mapping-preflight cases")
    maximum_rss = 0
    source_evidence: dict[str, dict[str, Any]] = {}
    for token in EXPECTED_CASE_TOKENS:
        case = case_rows[token]
        if set(case) != set(CASE_FIELDS) or case.get("status") != "SUCCESS":
            raise ValueError(f"{token}: mapping-preflight case schema/status mismatch")
        expected_files = {
            "scan_sha256": Path(str(inference_rows[token].get("source_scan_path") or "")).resolve(),
            "candidate_pool_sha256": Path(str(pool_rows[token].get("path") or "")).resolve(),
            "a0_prediction_sha256": a0_prediction_root.resolve() / "cases" / token / "prediction_owned_regions.json",
        }
        for field, path in expected_files.items():
            if not path.is_file() or _sha_file(path) != _require_sha(case.get(field), f"{token}.{field}"):
                raise ValueError(f"{token}: source byte hash changed: {field}")
        if _require_sha(case.get("frozen_control_probe_sha256"), f"{token}.control SHA") != _require_sha(
            control_rows[token].get("probe_sha256"), f"{token}.manifest control SHA"
        ):
            raise ValueError(f"{token}: frozen-control hash mismatch")
        if _require_sha(case.get("complete_edge_baseline_probe_sha256"), f"{token}.baseline SHA") != _require_sha(
            baseline_rows[token].get("replacement_probe_sha256"), f"{token}.manifest baseline SHA"
        ):
            raise ValueError(f"{token}: complete-edge baseline hash mismatch")
        frozen_control_path = Path(
            str(control_rows[token].get("probe_path") or "")
        ).resolve()
        frozen_baseline_path = (
            complete_edge_baseline_manifest_path.resolve().parent / f"{token}.json"
        )
        if (
            not frozen_control_path.is_file()
            or _sha_file(frozen_control_path)
            != _require_sha(control_rows[token].get("probe_sha256"), f"{token}.control file SHA")
        ):
            raise ValueError(f"{token}: frozen-control source bytes changed")
        if (
            not frozen_baseline_path.is_file()
            or _sha_file(frozen_baseline_path)
            != _require_sha(
                baseline_rows[token].get("replacement_probe_sha256"),
                f"{token}.baseline file SHA",
            )
        ):
            raise ValueError(f"{token}: complete-edge baseline source bytes changed")
        source = _derive_source_case_evidence(
            case_token=token,
            candidate_pool_path=Path(str(pool_rows[token]["path"])).resolve(),
            frozen_control_path=frozen_control_path,
            complete_edge_baseline_path=frozen_baseline_path,
        )
        source_evidence[token] = source
        expected_case_values = {
            "frozen_target_count": len(source["frozen_target_projection"]),
            "frozen_target_projection_sha256": _sha_json(
                source["frozen_target_projection"]
            ),
            "regenerated_target_projection_sha256": _sha_json(
                source["frozen_target_projection"]
            ),
            "thresholds_sha256": source["thresholds_sha256"],
            "frozen_thresholds_sha256": source["thresholds_sha256"],
            "regenerated_thresholds_sha256": source["thresholds_sha256"],
            "all_pair_membership_row_count": len(
                source["all_pair_membership_table"]
            ),
            "pair_tube_membership_sha256": source[
                "all_pair_membership_table_sha256"
            ],
            "frozen_all_pair_membership_table_sha256": source[
                "all_pair_membership_table_sha256"
            ],
            "regenerated_all_pair_membership_table_sha256": source[
                "all_pair_membership_table_sha256"
            ],
        }
        for field, expected in expected_case_values.items():
            if case.get(field) != expected:
                raise ValueError(f"{token}: case evidence is not source-derived: {field}")
        for field in (
            "frozen_control_reproduced", "complete_edge_baseline_reproduced",
            "frozen_target_projection_equal", "thresholds_unchanged",
            "all_pair_line_tube_memberships_unchanged",
        ):
            if case.get(field) is not True:
                raise ValueError(f"{token}: source-derived case predicate is false: {field}")
        maximum_rss = max(maximum_rss, int(case.get("peak_rss_bytes") or -1))
    if maximum_rss >= PEAK_RSS_LIMIT_BYTES or maximum_rss != int(
        artifact.get("maximum_peak_rss_bytes") or -1
    ):
        raise ValueError("mapping-preflight RSS evidence mismatch")

    raw_relations = artifact.get("frozen_target_baseline_relation_table")
    raw_outcomes = artifact.get("frozen_target_outcome_table")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        raise ValueError("mapping relation table must be a sequence")
    if not isinstance(raw_outcomes, Sequence) or isinstance(raw_outcomes, (str, bytes)):
        raise ValueError("mapping outcome table must be a sequence")
    relations = [dict(row) for row in raw_relations if isinstance(row, Mapping)]
    outcomes = [dict(row) for row in raw_outcomes if isinstance(row, Mapping)]
    if len(relations) != EXPECTED_TARGET_ROWS or len(outcomes) != EXPECTED_TARGET_ROWS:
        raise ValueError("mapping preflight requires exact untruncated 4,788-row tables")
    relation_keys = [
        (str(row.get("case_token") or ""), str(row.get("frozen_target_component_id") or ""))
        for row in relations
    ]
    outcome_keys = [
        (str(row.get("case_token") or ""), str(row.get("frozen_target_component_id") or ""))
        for row in outcomes
    ]
    if relation_keys != sorted(relation_keys) or len(set(relation_keys)) != EXPECTED_TARGET_ROWS:
        raise ValueError("mapping relation table order/identity mismatch")
    if outcome_keys != relation_keys:
        raise ValueError("mapping outcome target set/order mismatch")
    for relation, outcome in zip(relations, outcomes):
        _validate_relation(relation)
        _validate_outcome(outcome, relation)
    if _require_sha(artifact.get("relation_table_sha256"), "relation table SHA") != _sha_json(relations):
        raise ValueError("mapping relation table SHA mismatch")
    if _require_sha(artifact.get("outcome_table_sha256"), "outcome table SHA") != _sha_json(outcomes):
        raise ValueError("mapping outcome table SHA mismatch")

    raw_projection = artifact.get("frozen_target_projection")
    if not isinstance(raw_projection, Sequence) or isinstance(raw_projection, (str, bytes)):
        raise ValueError("frozen target projection must be a sequence")
    artifact_projection = [dict(row) for row in raw_projection if isinstance(row, Mapping)]
    expected_projection = [
        deepcopy(row)
        for token in EXPECTED_CASE_TOKENS
        for row in source_evidence[token]["frozen_target_projection"]
    ]
    if len(artifact_projection) != EXPECTED_TARGET_ROWS:
        raise ValueError("frozen target projection requires exact 4,788 rows")
    _assert_exact_target_preservation(relations, expected_projection)
    if artifact_projection != expected_projection:
        raise ValueError("artifact frozen target projection differs from frozen source")
    projection_sha = _sha_json(expected_projection)
    for field in (
        "frozen_target_projection_sha256",
        "regenerated_frozen_target_projection_sha256",
    ):
        if _require_sha(artifact.get(field), field) != projection_sha:
            raise ValueError(f"source-derived target projection hash mismatch: {field}")

    raw_memberships = artifact.get("all_pair_membership_table")
    if not isinstance(raw_memberships, Sequence) or isinstance(raw_memberships, (str, bytes)):
        raise ValueError("all-pair membership table must be a sequence")
    memberships = [dict(row) for row in raw_memberships if isinstance(row, Mapping)]
    membership_keys = [
        (str(row.get("case_token") or ""), str(row.get("baseline_local_plane_pair_id") or ""))
        for row in memberships
    ]
    if (
        len(memberships) != EXPECTED_ALL_PAIR_ROWS
        or int(artifact.get("all_pair_membership_row_count") or -1)
        != EXPECTED_ALL_PAIR_ROWS
        or membership_keys != sorted(membership_keys)
        or len(set(membership_keys)) != EXPECTED_ALL_PAIR_ROWS
    ):
        raise ValueError("complete all-pair membership population is not exact 16,988")
    expected_aggregate_memberships = []
    for token in EXPECTED_CASE_TOKENS:
        actual_case_rows = []
        for row in memberships:
            if row.get("case_token") == token:
                actual_case_rows.append({
                    field: deepcopy(row[field]) for field in ALL_PAIR_MEMBERSHIP_FIELDS
                })
        expected_case_rows = source_evidence[token]["all_pair_membership_table"]
        _assert_exact_all_pair_memberships(
            actual_case_rows,
            expected_case_rows,
            _sha_json(actual_case_rows),
        )
        expected_aggregate_memberships.extend(
            {"case_token": token, **deepcopy(row)} for row in expected_case_rows
        )
    if memberships != expected_aggregate_memberships:
        raise ValueError("aggregate all-pair membership table differs from frozen source")
    membership_sha = _sha_json(expected_aggregate_memberships)
    for field in (
        "all_pair_membership_table_sha256",
        "regenerated_all_pair_membership_table_sha256",
    ):
        if _require_sha(artifact.get(field), field) != membership_sha:
            raise ValueError(f"source-derived all-pair table hash mismatch: {field}")

    raw_thresholds = artifact.get("threshold_hash_table")
    if not isinstance(raw_thresholds, Sequence) or isinstance(raw_thresholds, (str, bytes)):
        raise ValueError("threshold hash table must be a sequence")
    threshold_rows = [dict(row) for row in raw_thresholds if isinstance(row, Mapping)]
    expected_threshold_rows = [
        {
            "case_token": token,
            "thresholds_sha256": source_evidence[token]["thresholds_sha256"],
        }
        for token in EXPECTED_CASE_TOKENS
    ]
    if threshold_rows != expected_threshold_rows:
        raise ValueError("threshold hash table changed from frozen baseline")
    for row in threshold_rows:
        _assert_exact_threshold_hash(
            row["thresholds_sha256"],
            source_evidence[str(row["case_token"])]["thresholds_sha256"],
        )
    threshold_table_sha = _sha_json(expected_threshold_rows)
    for field in (
        "threshold_hash_table_sha256",
        "regenerated_threshold_hash_table_sha256",
    ):
        if _require_sha(artifact.get(field), field) != threshold_table_sha:
            raise ValueError(f"source-derived threshold table hash mismatch: {field}")
    criteria = artifact.get("mapping_preflight_criteria")
    if not isinstance(criteria, Mapping):
        raise ValueError("mapping-preflight criteria must be an object")
    if set(criteria) != set(MAPPING_PREFLIGHT_CRITERIA) or any(
        criteria[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA
    ):
        raise ValueError("mapping-preflight criteria are not exact/all true")
    if not _c001_locked_regression(relations):
        raise ValueError("C001 amended locked regression mismatch")
    status_counts = Counter(row["target_baseline_relation_status"] for row in relations)
    if (
        sum(status_counts.values()) != EXPECTED_TARGET_ROWS
        or int(artifact.get("target_row_count") or -1) != EXPECTED_TARGET_ROWS
        or int(artifact.get("target_component_image_present_count") or -1)
        != status_counts[RELATION_STATUS_PRESENT]
        or int(artifact.get("target_component_image_absent_outside_tube_count") or -1)
        != status_counts[RELATION_STATUS_ABSENT]
    ):
        raise ValueError("mapping-preflight relation census mismatch")
    relation_projection = _exact_frozen_target_projection(relations)
    derived_criteria = {
        "all_declared_source_hashes_match": True,
        "all_frozen_control_probe_hashes_reproduce": all(
            case_rows[token]["frozen_control_reproduced"] is True
            for token in EXPECTED_CASE_TOKENS
        ),
        "all_complete_edge_baseline_hashes_reproduce": all(
            case_rows[token]["complete_edge_baseline_reproduced"] is True
            for token in EXPECTED_CASE_TOKENS
        ),
        "frozen_target_count_is_4788": len(relations) == EXPECTED_TARGET_ROWS,
        "frozen_target_rows_are_byte_identical": relation_projection == expected_projection,
        "frozen_target_id_sets_are_exactly_equal": {
            row["frozen_target_component_id"] for row in relation_projection
        } == {
            row["frozen_target_component_id"] for row in expected_projection
        },
        "frozen_target_mode_hash_sets_are_exactly_equal": {
            row["frozen_target_mode_ids_sha256"] for row in relation_projection
        } == {
            row["frozen_target_mode_ids_sha256"] for row in expected_projection
        },
        "every_target_mode_has_one_baseline_fragment": all(
            len(row["baseline_mode_fragment_rows"]) == len(row["frozen_target_mode_ids"])
            and all(item[1] is not None for item in row["baseline_mode_fragment_rows"])
            for row in relations
        ),
        "every_target_maps_to_exactly_two_fragments": all(
            len(row["baseline_fragment_ids"]) == 2 for row in relations
        ),
        "every_target_has_exactly_one_pair_lineage": all(
            row["baseline_pair_mapping_count"] == 1 for row in relations
        ),
        "every_in_tube_target_has_one_component_image": all(
            not row["target_parent_in_frozen_transverse_set"]
            or row["baseline_component_mapping_count"] == 1
            for row in relations
        ),
        "every_out_of_tube_target_has_zero_component_images": all(
            row["target_parent_in_frozen_transverse_set"]
            or row["baseline_component_mapping_count"] == 0
            for row in relations
        ),
        "no_unrecognised_relation_status_exists": set(status_counts)
        <= {RELATION_STATUS_PRESENT, RELATION_STATUS_ABSENT},
        "relation_row_count_is_4788": len(relations) == EXPECTED_TARGET_ROWS,
        "relation_table_is_untruncated": len(set(relation_keys)) == EXPECTED_TARGET_ROWS,
        "outcome_row_count_is_4788": len(outcomes) == EXPECTED_TARGET_ROWS
        and outcome_keys == relation_keys,
        "c001_locked_regression_matches": _c001_locked_regression(relations),
        "all_threshold_hashes_are_unchanged": threshold_rows == expected_threshold_rows,
        "all_pair_line_tube_memberships_are_unchanged": memberships
        == expected_aggregate_memberships,
        "gold_open_authorized_false": (
            artifact.get("gold_open_authorized") is False
            and artifact.get("truth_inputs_consumed") is False
        ),
    }
    _assert_criteria_match_derived(criteria, derived_criteria)
    compact_cases = [
        {
            "case_token": token,
            "frozen_target_count": int(case_rows[token]["frozen_target_count"]),
            "relation_row_count": int(case_rows[token]["relation_row_count"]),
            "outcome_row_count": int(case_rows[token]["outcome_row_count"]),
            "component_image_present_count": int(case_rows[token]["component_image_present_count"]),
            "component_image_absent_outside_tube_count": int(
                case_rows[token]["component_image_absent_outside_tube_count"]
            ),
            "relation_table_sha256": case_rows[token]["relation_table_sha256"],
            "outcome_table_sha256": case_rows[token]["outcome_table_sha256"],
            "thresholds_sha256": case_rows[token]["thresholds_sha256"],
            "pair_tube_membership_sha256": case_rows[token]["pair_tube_membership_sha256"],
            "frozen_target_projection_sha256": case_rows[token][
                "frozen_target_projection_sha256"
            ],
            "all_pair_membership_row_count": int(
                case_rows[token]["all_pair_membership_row_count"]
            ),
            "peak_rss_bytes": int(case_rows[token]["peak_rss_bytes"]),
        }
        for token in EXPECTED_CASE_TOKENS
    ]
    c001 = next(
        row for row in relations
        if row["case_token"] == "C001"
        and row["frozen_target_component_id"] == "LTC_32d3609a79a9c314"
    )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "profile": PROFILE,
        "research_only": True,
        "inference_safe": True,
        "truth_inputs_consumed": False,
        "ordinary_batch_authorized": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "compact_bundle_authorized": False,
        "gold_open_authorized": False,
        "next_professor_packet_authorized": True,
        "decision": DECISION_GO,
        "mapping_preflight_sha256": actual_preflight_sha,
        **{field: _require_sha(artifact[field], field) for field in source_paths},
        "target_row_count": EXPECTED_TARGET_ROWS,
        "target_component_image_present_count": status_counts[RELATION_STATUS_PRESENT],
        "target_component_image_absent_outside_tube_count": status_counts[RELATION_STATUS_ABSENT],
        "relation_table_sha256": artifact["relation_table_sha256"],
        "outcome_table_sha256": artifact["outcome_table_sha256"],
        "frozen_target_projection_sha256": projection_sha,
        "all_pair_membership_row_count": EXPECTED_ALL_PAIR_ROWS,
        "all_pair_membership_table_sha256": membership_sha,
        "threshold_hash_table_sha256": threshold_table_sha,
        "mapping_preflight_criteria": deepcopy(criteria),
        "maximum_peak_rss_bytes": maximum_rss,
        "c001_locked_regression": deepcopy(c001),
        "cases_sha256": _sha_json(compact_cases),
        "cases": compact_cases,
    }
    evidence["compact_evidence_sha256"] = _record_hash(
        evidence, "compact_evidence_sha256"
    )
    return evidence


def canonical_evidence_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = validate_and_compact(
        preflight_path=args.preflight,
        expected_preflight_sha256=args.expected_preflight_sha256,
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_evidence_bytes(evidence))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": _sha_file(args.output.resolve()),
                "mapping_preflight_sha256": evidence["mapping_preflight_sha256"],
                "decision": evidence["decision"],
                "ordinary_batch_authorized": False,
                "gold_open_authorized": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
