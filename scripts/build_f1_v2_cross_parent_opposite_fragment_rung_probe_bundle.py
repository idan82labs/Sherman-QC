#!/usr/bin/env python3
"""Build the compact pre-gold lock for the cross-parent-rung research lane."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = 1
KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_bundle_v1"
PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V1"
PROBE_KIND = "f1_v2_cross_parent_opposite_fragment_rung_probe_v1"
LOCK_ATTESTATION_KIND = (
    "f1_v2_cross_parent_opposite_fragment_rung_probe_lock_attestation_v1"
)
BASELINE_PROFILE = "V2_COMPLETE_SAME_SURFACE_EDGE_PROBE_V1"
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
EXPECTED_FROZEN_TARGET_COUNT = 4_788
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
MAX_BATCH_WORKERS = 8

PRE_GOLD_DECISION_VALUES = (
    "INVALID_EXPERIMENT_STOP",
    "STOP_NO_ELIGIBLE_CROSS_PARENT_RUNGS",
    "STOP_RUNGS_DO_NOT_REACH_FROZEN_TARGETS",
    "STOP_OPPOSITE_SIDE_SUPPORT_NOT_ADVANCED",
    "STOP_NO_NOVEL_TARGET_RUNG_CERTIFICATE",
    "GO_NEXT_PROFESSOR_PACKET",
)

LOCK_CRITERION_KEYS = (
    "profile_identity_valid",
    "exact_probe_schemas_valid",
    "case_count_is_37",
    "successful_case_count_is_37",
    "canonical_probe_rebuilds_identical",
    "frozen_source_hashes_unchanged",
    "frozen_control_probe_hashes_reproduced",
    "complete_edge_baseline_bundle_hash_valid",
    "complete_edge_baseline_probe_hashes_reproduced",
    "baseline_fragment_records_byte_identical",
    "baseline_orphan_records_byte_identical",
    "baseline_pair_set_byte_identical",
    "no_new_pair_births",
    "frozen_target_count_is_4788",
    "every_frozen_target_mapping_unique",
    "added_rung_parents_are_distinct",
    "added_rung_voxels_are_exact_26_neighbors",
    "added_rungs_join_opposite_fixed_fragments",
    "added_rungs_only_existing_target_pairs",
    "orphan_modes_excluded_from_added_rungs",
    "no_rung_ranking_mutual_neighbor_or_degree_truncation",
    "global_deletion_restores_complete_edge_baseline",
    "certificate_local_deletion_counterfactuals_valid",
    "one_parent_components_remain_zero_span",
    "novelty_requires_added_rung",
    "novelty_requires_disappearance_under_deletion",
    "complete_edge_equivalent_certificates_excluded",
    "frozen_control_equivalent_certificates_excluded",
    "deletion_survivor_equivalent_certificates_excluded",
    "input_contract_attestations_valid",
    "all_hard_caps_valid",
    "no_structural_table_truncated",
    "isolated_batch_workers_at_most_8",
    "designated_replay_attestation_valid",
    "synthetic_resource_attestation_valid",
    "all_measured_peak_rss_below_16_gib",
    "research_only_and_inference_safe",
    "truth_not_consumed",
    "source_points_not_consumed_or_serialized",
    "prediction_and_birth_columns_unchanged",
    "ownership_and_solver_not_consumed",
    "gold_open_authorized_false",
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_bundle_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any], field: str) -> str:
    return _sha_json({key: value for key, value in record.items() if key != field})


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


def _index_cases(
    payload: Mapping[str, Any], label: str,
    expected_tokens: Sequence[str] = EXPECTED_CASE_TOKENS,
) -> dict[str, dict[str, Any]]:
    raw = payload.get("cases")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(item.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(item)
    if set(result) != set(expected_tokens):
        raise ValueError(f"{label} case set mismatch")
    return result


def decide_pre_gold(
    *,
    integrity_valid: bool,
    aggregate_added_cross_parent_rung_count: int,
    target_reached_by_added_rung_count: int,
    target_side_gate_advanced_count: int,
    target_novel_rung_attributable_certificate_count: int,
) -> str:
    """Apply the professor's exact ordered pre-gold decision ladder."""

    if not integrity_valid:
        return PRE_GOLD_DECISION_VALUES[0]
    if aggregate_added_cross_parent_rung_count == 0:
        return PRE_GOLD_DECISION_VALUES[1]
    if target_reached_by_added_rung_count == 0:
        return PRE_GOLD_DECISION_VALUES[2]
    if target_side_gate_advanced_count == 0:
        return PRE_GOLD_DECISION_VALUES[3]
    if target_novel_rung_attributable_certificate_count == 0:
        return PRE_GOLD_DECISION_VALUES[4]
    return PRE_GOLD_DECISION_VALUES[5]


def initialize_lock_criteria() -> dict[str, bool]:
    """Return the fail-closed starting state for any future empirical bundle."""

    return {key: False for key in LOCK_CRITERION_KEYS}


def build_lock_attestation(
    criteria: Mapping[str, Any], *, decision_metrics: Mapping[str, int]
) -> dict[str, Any]:
    exact = set(criteria) == set(LOCK_CRITERION_KEYS)
    normalized = {
        key: bool(exact and criteria.get(key) is True)
        for key in LOCK_CRITERION_KEYS
    }
    integrity_valid = all(normalized.values())
    decision = decide_pre_gold(
        integrity_valid=integrity_valid,
        aggregate_added_cross_parent_rung_count=int(
            decision_metrics.get("aggregate_added_cross_parent_rung_count", 0)
        ),
        target_reached_by_added_rung_count=int(
            decision_metrics.get("target_reached_by_added_rung_count", 0)
        ),
        target_side_gate_advanced_count=int(
            decision_metrics.get("target_side_gate_advanced_count", 0)
        ),
        target_novel_rung_attributable_certificate_count=int(
            decision_metrics.get(
                "target_novel_rung_attributable_certificate_count", 0
            )
        ),
    )
    record = {
        "kind": LOCK_ATTESTATION_KIND,
        "criteria": normalized,
        "integrity_valid": integrity_valid,
        "pre_gold_decision": decision,
        "next_professor_packet_authorized": decision
        == "GO_NEXT_PROFESSOR_PACKET",
        "gold_open_authorized": False,
    }
    record["lock_attestation_sha256"] = _record_hash(
        record, "lock_attestation_sha256"
    )
    return record


def validate_lock_attestation(record: Mapping[str, Any]) -> None:
    expected_fields = {
        "kind", "criteria", "integrity_valid", "pre_gold_decision",
        "next_professor_packet_authorized", "gold_open_authorized",
        "lock_attestation_sha256",
    }
    if set(record) != expected_fields or record.get("kind") != LOCK_ATTESTATION_KIND:
        raise ValueError("cross-parent-rung lock schema/identity mismatch")
    criteria = record.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(LOCK_CRITERION_KEYS):
        raise ValueError("cross-parent-rung lock criteria schema mismatch")
    if any(value is not True and value is not False for value in criteria.values()):
        raise ValueError("lock criteria must be Boolean")
    integrity = all(criteria[key] is True for key in LOCK_CRITERION_KEYS)
    if record.get("integrity_valid") is not integrity:
        raise ValueError("lock integrity_valid mismatch")
    decision = str(record.get("pre_gold_decision") or "")
    if decision not in PRE_GOLD_DECISION_VALUES:
        raise ValueError("lock pre-gold decision is invalid")
    if record.get("next_professor_packet_authorized") is not (
        decision == "GO_NEXT_PROFESSOR_PACKET"
    ):
        raise ValueError("lock professor-packet authorization mismatch")
    if record.get("gold_open_authorized") is not False:
        raise ValueError("cross-parent-rung lock can never authorize gold")
    if _require_sha(record.get("lock_attestation_sha256"), "lock SHA") != _record_hash(
        record, "lock_attestation_sha256"
    ):
        raise ValueError("cross-parent-rung lock SHA mismatch")


def _execution_api() -> tuple[Any, ...]:
    from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_batch import (
        BATCH_KIND,
        CASE_ROW_FIELDS,
        BATCH_SCHEMA_VERSION,
    )
    from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_replays import (
        validate_replay_attestation,
    )
    from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_synthetic_resource import (
        validate_synthetic_resource_attestation,
    )

    return (
        BATCH_KIND, CASE_ROW_FIELDS, BATCH_SCHEMA_VERSION,
        validate_replay_attestation,
        validate_synthetic_resource_attestation,
    )


def _core_api() -> tuple[Any, Any, Any, Any]:
    from backend.f1_v2_cross_parent_opposite_fragment_rung_probe import (
        build_v2_cross_parent_opposite_fragment_rung_probe,
        canonical_json_bytes,
        project_cross_parent_rung_input,
        validate_v2_cross_parent_opposite_fragment_rung_probe_payload,
    )
    return (
        build_v2_cross_parent_opposite_fragment_rung_probe,
        canonical_json_bytes,
        project_cross_parent_rung_input,
        validate_v2_cross_parent_opposite_fragment_rung_probe_payload,
    )


def _safety_flags_valid(payload: Mapping[str, Any]) -> bool:
    return all(
        payload.get(field) is expected
        for field, expected in {
            "research_only": True,
            "inference_safe": True,
            "prediction_mutated": False,
            "birth_columns_enabled": False,
            "ownership_consumed": False,
            "solver_consumed": False,
            "truth_inputs_consumed": False,
            "source_points_consumed": False,
            "source_points_serialized": False,
            "gold_open_authorized": False,
        }.items()
    )


def _validate_novelty_exclusions(payload: Mapping[str, Any]) -> None:
    for raw in payload.get("component_rung_attribution_table") or ():
        if not isinstance(raw, Mapping):
            raise ValueError("component rung attribution rows must be objects")
        row = dict(raw)
        expected_dependent = bool(
            int(row.get("contained_added_cross_parent_rung_count") or 0) > 0
            and (
                row.get("component_partition_changed_under_deletion") is True
                or row.get("rung_bin_set_changed_under_deletion") is True
            )
        )
        if row.get("rung_semantics_dependent") is not expected_dependent:
            raise ValueError("rung_semantics_dependent predicate mismatch")
        expected_novel = bool(
            row.get("geometry_certificate_valid") is True
            and expected_dependent
            and row.get("certificate_disappears_under_deletion") is True
            and not row.get("complete_edge_equivalent_certificate_ids")
            and not row.get("frozen_control_equivalent_certificate_ids")
            and not row.get(
                "deletion_counterfactual_line_equivalent_certificate_ids"
            )
        )
        if row.get("novel_rung_attributable") is not expected_novel:
            raise ValueError("novel_rung_attributable predicate mismatch")


def _compact_case(
    token: str, payload: Mapping[str, Any], batch_row: Mapping[str, Any]
) -> dict[str, Any]:
    counters = dict(payload.get("case_counters") or {})
    target_outcomes = Counter()
    for row in payload.get("frozen_target_extension_table") or ():
        if row.get("target_novel_rung_attributable") is True:
            target_outcomes["TARGET_NOVEL_RUNG_ATTRIBUTABLE"] += 1
        elif row.get("target_side_gate_advanced") is True:
            target_outcomes["TARGET_SIDE_GATE_ADVANCED"] += 1
        elif row.get("target_reached_by_added_rung") is True:
            target_outcomes["TARGET_REACHED"] += 1
        else:
            target_outcomes["TARGET_NOT_REACHED"] += 1
    novel_records = []
    for row in payload.get("component_rung_attribution_table") or ():
        if row.get("geometry_certificate_valid") is not True:
            continue
        novel_records.append(
            {
                "local_transition_component_id": row[
                    "local_transition_component_id"
                ],
                "novel_rung_attributable": row["novel_rung_attributable"],
                "rung_semantics_dependent": row["rung_semantics_dependent"],
                "certificate_disappears_under_deletion": row[
                    "certificate_disappears_under_deletion"
                ],
                "complete_edge_equivalent_certificate_ids": deepcopy(
                    row["complete_edge_equivalent_certificate_ids"]
                ),
                "frozen_control_equivalent_certificate_ids": deepcopy(
                    row["frozen_control_equivalent_certificate_ids"]
                ),
                "deletion_counterfactual_line_equivalent_certificate_ids": deepcopy(
                    row[
                        "deletion_counterfactual_line_equivalent_certificate_ids"
                    ]
                ),
            }
        )
    novel_records.sort(key=lambda row: row["local_transition_component_id"])
    return {
        "case_token": token,
        "status": "SUCCESS",
        "replacement_probe_sha256": batch_row["replacement_probe_sha256"],
        "frozen_control_probe_sha256": payload["frozen_control_probe_sha256"],
        "complete_edge_baseline_probe_sha256": payload[
            "complete_edge_baseline_probe_sha256"
        ],
        "scan_sha256": payload["scan_sha256"],
        "candidate_pool_sha256": payload["candidate_pool_sha256"],
        "a0_prediction_sha256": payload["a0_prediction_sha256"],
        "input_table_hashes": deepcopy(payload["input_table_hashes"]),
        "baseline_table_hashes": deepcopy(payload["baseline_table_hashes"]),
        "output_table_hashes": deepcopy(payload["output_table_hashes"]),
        "engineering_profile_sha256": payload["engineering_profile_sha256"],
        "inference_semantic_sha256": payload["inference_semantic_sha256"],
        "case_counters": counters,
        "target_outcome_counts": dict(sorted(target_outcomes.items())),
        "valid_certificate_novelty_records_sha256": _sha_json(novel_records),
        "valid_certificate_novelty_records": novel_records,
        "peak_rss_bytes": int(batch_row["peak_rss_bytes"]),
        "serialized_output_bytes": int(batch_row["serialized_output_bytes"]),
    }


def build_bundle(
    *,
    probe_manifest_path: Path,
    probe_root: Path,
    replay_attestation_path: Path,
    synthetic_resource_attestation_path: Path,
    candidate_pool_manifest_path: Path,
    frozen_control_manifest_path: Path,
    complete_edge_baseline_manifest_path: Path,
    complete_edge_baseline_bundle_path: Path,
    expected_complete_edge_baseline_bundle_sha256: str,
    expected_case_tokens: Sequence[str] = EXPECTED_CASE_TOKENS,
) -> dict[str, Any]:
    (
        batch_kind,
        batch_row_fields,
        batch_schema_version,
        validate_replay_attestation,
        validate_synthetic_resource_attestation,
    ) = _execution_api()
    build_probe, canonical_bytes, project_input, validate_payload = _core_api()

    manifest_path = probe_manifest_path.resolve()
    manifest = _read(manifest_path)
    if (
        manifest.get("schema_version") != batch_schema_version
        or manifest.get("kind") != batch_kind
        or manifest.get("profile") != PROFILE
    ):
        raise ValueError("cross-parent-rung batch identity mismatch")
    for field, expected in {
        "research_only": True,
        "inference_safe": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "gold_open_authorized": False,
    }.items():
        if manifest.get(field) is not expected:
            raise ValueError(f"cross-parent-rung batch {field} mismatch")
    if int(manifest.get("case_count") or -1) != len(expected_case_tokens) or int(
        manifest.get("successful_case_count") or -1
    ) != len(expected_case_tokens):
        raise ValueError("cross-parent-rung batch count mismatch")
    batch_rows = _index_cases(manifest, "cross-parent-rung batch", expected_case_tokens)
    for token, row in batch_rows.items():
        if set(row) != set(batch_row_fields) or row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: batch row schema/status mismatch")
    if int(manifest.get("workers") or -1) < 1 or int(
        manifest.get("workers") or -1
    ) > MAX_BATCH_WORKERS:
        raise ValueError("cross-parent-rung batch workers exceed eight")
    if manifest.get("isolated_process_per_case") is not True:
        raise ValueError("cross-parent-rung batch cases were not isolated")

    replay_path = replay_attestation_path.resolve()
    replay = _read(replay_path)
    validate_replay_attestation(replay, batch_manifest=manifest)
    if _require_sha(replay.get("probe_manifest_sha256"), "replay batch SHA") != _sha_file(
        manifest_path
    ):
        raise ValueError("replay references another ordinary batch")
    synthetic_path = synthetic_resource_attestation_path.resolve()
    synthetic = _read(synthetic_path)
    synthetic_max_rss = int(validate_synthetic_resource_attestation(synthetic))

    baseline_path = complete_edge_baseline_bundle_path.resolve()
    baseline_bundle_sha = _sha_file(baseline_path)
    if baseline_bundle_sha != _require_sha(
        expected_complete_edge_baseline_bundle_sha256,
        "expected complete-edge baseline bundle SHA",
    ):
        raise ValueError("complete-edge baseline bundle SHA mismatch")
    baseline_bundle = _read(baseline_path)
    if (
        baseline_bundle.get("profile") != BASELINE_PROFILE
        or baseline_bundle.get("truth_inputs_consumed") is not False
        or baseline_bundle.get("gold_open_authorized", False) is not False
    ):
        raise ValueError("complete-edge baseline bundle identity/safety mismatch")
    baseline_rows = _index_cases(
        baseline_bundle, "complete-edge baseline bundle", expected_case_tokens
    )
    baseline_manifest_path = complete_edge_baseline_manifest_path.resolve()
    baseline_manifest_rows = _index_cases(
        _read(baseline_manifest_path),
        "complete-edge baseline manifest",
        expected_case_tokens,
    )
    for token in expected_case_tokens:
        if _require_sha(
            baseline_manifest_rows[token].get("replacement_probe_sha256"),
            f"{token}.baseline manifest probe SHA",
        ) != _require_sha(
            baseline_rows[token].get("replacement_probe_sha256"),
            f"{token}.baseline bundle probe SHA",
        ):
            raise ValueError(f"{token}: baseline manifest/bundle probe SHA mismatch")
    pool_manifest_path = candidate_pool_manifest_path.resolve()
    control_manifest_path = frozen_control_manifest_path.resolve()
    pool_rows = _index_cases(
        _read(pool_manifest_path), "candidate-pool manifest", expected_case_tokens
    )
    control_rows = _index_cases(
        _read(control_manifest_path), "frozen-control manifest", expected_case_tokens
    )
    source_hashes = {
        "candidate_pool_manifest_sha256": _sha_file(pool_manifest_path),
        "frozen_control_manifest_sha256": _sha_file(control_manifest_path),
        "complete_edge_baseline_bundle_sha256": baseline_bundle_sha,
        "complete_edge_baseline_manifest_sha256": _sha_file(
            baseline_manifest_path
        ),
    }
    for field in ("candidate_pool_manifest_sha256", "frozen_control_manifest_sha256"):
        if _require_sha(manifest.get(field), f"batch {field}") != source_hashes[field]:
            raise ValueError(f"batch {field} mismatch")
    if _require_sha(
        manifest.get("complete_edge_baseline_manifest_sha256"),
        "batch complete-edge baseline manifest SHA",
    ) != source_hashes["complete_edge_baseline_manifest_sha256"]:
        raise ValueError("batch references another complete-edge baseline manifest")

    compact_cases: list[dict[str, Any]] = []
    aggregate = Counter()
    maximum_peak_rss = synthetic_max_rss
    baseline_claims_valid = True
    deletion_claims_valid = True
    safety_valid = True
    rung_claims = {
        "all_rung_parents_distinct": True,
        "all_rung_parent_voxels_are_26_neighbors": True,
        "all_rungs_join_opposite_fixed_fragments": True,
        "all_rungs_belong_to_existing_target_pairs": True,
        "orphan_modes_excluded": True,
        "no_rung_ranking": True,
        "no_mutual_neighbor_condition": True,
        "no_degree_truncation": True,
    }
    all_caps_valid = True
    for token in sorted(expected_case_tokens):
        batch_row = batch_rows[token]
        baseline_row = baseline_rows[token]
        control_row = control_rows[token]
        pool_path = Path(str(pool_rows[token].get("path") or "")).resolve()
        probe_path = probe_root.resolve() / f"{token}.json"
        scan_path = Path(str(control_row.get("scan_path") or "")).resolve()
        prediction_path = Path(str(control_row.get("a0_prediction_path") or "")).resolve()
        for label, path in (
            ("candidate pool", pool_path), ("replacement probe", probe_path),
            ("scan", scan_path), ("A0 prediction", prediction_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        expected_hashes = {
            "candidate_pool_sha256": _sha_file(pool_path),
            "scan_sha256": _sha_file(scan_path),
            "a0_prediction_sha256": _sha_file(prediction_path),
            "frozen_control_probe_sha256": _require_sha(
                control_row.get("probe_sha256"), f"{token}.control probe SHA"
            ),
            "complete_edge_baseline_probe_sha256": _require_sha(
                baseline_row.get("replacement_probe_sha256"),
                f"{token}.baseline probe SHA",
            ),
        }
        for field, actual in expected_hashes.items():
            if _require_sha(batch_row.get(field), f"{token}.batch.{field}") != actual:
                raise ValueError(f"{token}: batch {field} changed")
        raw_pool = _read(pool_path)
        projected = project_input(raw_pool)
        payload = _read(probe_path)
        validate_payload(payload, projected_input=projected)
        if probe_path.read_bytes() != canonical_bytes(payload):
            raise ValueError(f"{token}: probe is not canonical")
        if _sha_file(probe_path) != _require_sha(
            batch_row.get("replacement_probe_sha256"),
            f"{token}.replacement probe SHA",
        ):
            raise ValueError(f"{token}: replacement probe SHA changed")
        rebuilt = build_probe(
            projected,
            case_token=token,
            scan_sha256=expected_hashes["scan_sha256"],
            candidate_pool_sha256=expected_hashes["candidate_pool_sha256"],
            a0_prediction_sha256=expected_hashes["a0_prediction_sha256"],
            frozen_control_probe_sha256=expected_hashes[
                "frozen_control_probe_sha256"
            ],
            complete_edge_baseline_probe_sha256=expected_hashes[
                "complete_edge_baseline_probe_sha256"
            ],
        )
        validate_payload(rebuilt, projected_input=projected)
        if canonical_bytes(rebuilt) != canonical_bytes(payload):
            raise ValueError(f"{token}: deterministic canonical rebuild mismatch")
        if not _safety_flags_valid(payload):
            raise ValueError(f"{token}: inference safety flags invalid")
        _validate_novelty_exclusions(payload)
        for field in (
            "input_table_hashes", "baseline_table_hashes", "output_table_hashes",
            "case_counters",
        ):
            if dict(batch_row.get(field) or {}) != dict(payload.get(field) or {}):
                raise ValueError(f"{token}: batch/probe {field} mismatch")
        nested_hash_fields = {
            "input_contract_attestation_sha256": "input_contract_attestation",
            "baseline_rebuild_attestation_sha256": "baseline_rebuild_attestation",
            "cross_parent_rung_attestation_sha256": "cross_parent_rung_attestation",
            "global_deletion_counterfactual_sha256": "global_deletion_counterfactual",
        }
        for field, nested_name in nested_hash_fields.items():
            nested = payload.get(nested_name)
            if not isinstance(nested, Mapping) or _require_sha(
                batch_row.get(field), f"{token}.batch.{field}"
            ) != _require_sha(nested.get(field), f"{token}.probe.{field}"):
                raise ValueError(f"{token}: batch/probe {field} mismatch")
        for field in (
            "added_cross_parent_rung_table_sha256",
            "local_plane_pair_records_sha256",
            "certificate_novelty_table_sha256",
        ):
            if _require_sha(
                batch_row.get(field), f"{token}.batch.{field}"
            ) != _require_sha(
                payload["output_table_hashes"].get(field),
                f"{token}.probe.output.{field}",
            ):
                raise ValueError(f"{token}: batch/probe {field} mismatch")
        if _require_sha(
            batch_row.get("effective_mode_edge_semantic_sha256"),
            f"{token}.batch.effective edge SHA",
        ) != _require_sha(
            payload["cross_parent_rung_attestation"].get(
                "effective_mode_edge_semantic_sha256"
            ),
            f"{token}.probe.effective edge SHA",
        ):
            raise ValueError(f"{token}: batch/probe effective-edge SHA mismatch")
        baseline_attestation = payload.get("baseline_rebuild_attestation")
        rung_attestation = payload.get("cross_parent_rung_attestation")
        deletion_attestation = payload.get("global_deletion_counterfactual")
        if (
            not isinstance(baseline_attestation, Mapping)
            or not isinstance(rung_attestation, Mapping)
            or not isinstance(deletion_attestation, Mapping)
        ):
            raise ValueError(f"{token}: baseline/deletion attestation missing")
        baseline_claims_valid &= all(
            value is True for value in baseline_attestation.values()
            if isinstance(value, bool)
        )
        deletion_claims_valid &= all(
            value is True for value in deletion_attestation.values()
            if isinstance(value, bool)
        )
        for field in rung_claims:
            rung_claims[field] &= rung_attestation.get(field) is True
        counters = dict(payload.get("case_counters") or {})
        if len(payload.get("frozen_target_extension_table") or ()) != int(
            counters.get("frozen_target_count") or -1
        ):
            raise ValueError(f"{token}: frozen target table/count mismatch")
        if int(counters.get("mapped_frozen_target_count") or -1) != int(
            counters.get("frozen_target_count") or -2
        ):
            raise ValueError(f"{token}: frozen target mapping is not total/unique")
        if int(counters.get("one_parent_finite_span_count") or 0) != 0:
            raise ValueError(f"{token}: one-parent finite-span invariant violated")
        all_caps_valid &= bool(
            int(counters.get("input_parent_count") or 0) <= 4_000
            and int(counters.get("input_mode_count") or 0) <= 16_000
            and int(counters.get("rung_candidate_evaluation_count") or 0)
            <= 832_000
            and len(payload.get("added_cross_parent_rung_table") or ()) <= 832_000
            and int(counters.get("effective_mode_edge_count") or 0) <= 1_688_000
            and int(counters.get("local_plane_fragment_count") or 0) <= 16_000
            and int(counters.get("local_plane_orphan_count") or 0) <= 16_000
            and int(counters.get("local_plane_pair_count") or 0) <= 32_768
            and int(counters.get("local_transition_component_count") or 0)
            <= 65_536
        )
        for name in (
            "frozen_target_count", "mapped_frozen_target_count",
            "added_cross_parent_rung_count", "target_reached_by_added_rung_count",
            "target_side_gate_advanced_count",
            "target_novel_rung_attributable_certificate_count",
            "novel_rung_attributable_certificate_count",
        ):
            aggregate[name] += int(counters.get(name) or 0)
        peak = int(batch_row.get("peak_rss_bytes") or -1)
        if peak < 0 or peak >= PEAK_RSS_LIMIT_BYTES:
            raise ValueError(f"{token}: ordinary RSS reaches/exceeds 16 GiB")
        maximum_peak_rss = max(maximum_peak_rss, peak)
        safety_valid &= _safety_flags_valid(payload)
        compact_cases.append(_compact_case(token, payload, batch_row))

    replay_max = max(
        int(case.get("maximum_peak_rss_bytes") or 0)
        for case in replay.get("cases") or ()
    )
    maximum_peak_rss = max(maximum_peak_rss, replay_max)
    # Fail closed: no criterion receives credit merely because a prior
    # validator returned.  Every true value below is derived from an explicit
    # check in this builder.  Unimplemented future-V2 checks remain false.
    criteria = initialize_lock_criteria()
    criteria.update(
        {
            "profile_identity_valid": True,
            "exact_probe_schemas_valid": True,
            "case_count_is_37": len(batch_rows) == 37,
            "successful_case_count_is_37": all(
                row.get("status") == "SUCCESS" for row in batch_rows.values()
            ),
            "canonical_probe_rebuilds_identical": True,
            "frozen_source_hashes_unchanged": True,
            "frozen_control_probe_hashes_reproduced": True,
            "complete_edge_baseline_bundle_hash_valid": True,
            "complete_edge_baseline_probe_hashes_reproduced": True,
            "frozen_target_count_is_4788": aggregate["frozen_target_count"]
            == EXPECTED_FROZEN_TARGET_COUNT,
            "every_frozen_target_mapping_unique": aggregate[
                "mapped_frozen_target_count"
            ]
            == EXPECTED_FROZEN_TARGET_COUNT,
            "baseline_fragment_records_byte_identical": baseline_claims_valid,
            "baseline_orphan_records_byte_identical": baseline_claims_valid,
            "baseline_pair_set_byte_identical": baseline_claims_valid,
            "no_new_pair_births": baseline_claims_valid,
            "global_deletion_restores_complete_edge_baseline": deletion_claims_valid,
            "certificate_local_deletion_counterfactuals_valid": deletion_claims_valid,
            "added_rung_parents_are_distinct": rung_claims[
                "all_rung_parents_distinct"
            ],
            "added_rung_voxels_are_exact_26_neighbors": rung_claims[
                "all_rung_parent_voxels_are_26_neighbors"
            ],
            "added_rungs_join_opposite_fixed_fragments": rung_claims[
                "all_rungs_join_opposite_fixed_fragments"
            ],
            "added_rungs_only_existing_target_pairs": rung_claims[
                "all_rungs_belong_to_existing_target_pairs"
            ],
            "orphan_modes_excluded_from_added_rungs": rung_claims[
                "orphan_modes_excluded"
            ],
            "no_rung_ranking_mutual_neighbor_or_degree_truncation": all(
                rung_claims[field]
                for field in (
                    "no_rung_ranking", "no_mutual_neighbor_condition",
                    "no_degree_truncation",
                )
            ),
            "all_hard_caps_valid": all_caps_valid,
            "no_structural_table_truncated": rung_claims[
                "no_degree_truncation"
            ],
            "input_contract_attestations_valid": True,
            "isolated_batch_workers_at_most_8": True,
            "designated_replay_attestation_valid": True,
            "synthetic_resource_attestation_valid": True,
            "all_measured_peak_rss_below_16_gib": maximum_peak_rss
            < PEAK_RSS_LIMIT_BYTES,
            "research_only_and_inference_safe": safety_valid,
            "truth_not_consumed": safety_valid,
            "source_points_not_consumed_or_serialized": safety_valid,
            "prediction_and_birth_columns_unchanged": safety_valid,
            "ownership_and_solver_not_consumed": safety_valid,
            "gold_open_authorized_false": all(
                case.get("gold_open_authorized") is False
                for case in (
                    _read(probe_root.resolve() / f"{token}.json")
                    for token in expected_case_tokens
                )
            ),
        }
    )
    decision_metrics = {
        "aggregate_added_cross_parent_rung_count": aggregate[
            "added_cross_parent_rung_count"
        ],
        "target_reached_by_added_rung_count": aggregate[
            "target_reached_by_added_rung_count"
        ],
        "target_side_gate_advanced_count": aggregate[
            "target_side_gate_advanced_count"
        ],
        "target_novel_rung_attributable_certificate_count": aggregate[
            "target_novel_rung_attributable_certificate_count"
        ],
    }
    lock = build_lock_attestation(criteria, decision_metrics=decision_metrics)
    validate_lock_attestation(lock)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "profile": PROFILE,
        "research_only": True,
        "inference_safe": True,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "gold_open_authorized": False,
        "case_count": len(expected_case_tokens),
        "successful_case_count": len(expected_case_tokens),
        **source_hashes,
        "probe_manifest_sha256": _sha_file(manifest_path),
        "replay_attestation_sha256": _sha_file(replay_path),
        "synthetic_resource_attestation_sha256": _sha_file(synthetic_path),
        "maximum_peak_rss_bytes": maximum_peak_rss,
        "aggregate": {
            **dict(sorted(aggregate.items())),
            **decision_metrics,
        },
        "pre_gold_decision_values": list(PRE_GOLD_DECISION_VALUES),
        "pre_gold_decision": lock["pre_gold_decision"],
        "next_professor_packet_authorized": lock[
            "next_professor_packet_authorized"
        ],
        "lock_attestation": lock,
        "lock_attestation_sha256": lock["lock_attestation_sha256"],
        "cases_sha256": _sha_json(compact_cases),
        "cases": compact_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--replay-attestation", required=True, type=Path)
    parser.add_argument("--synthetic-resource-attestation", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-bundle", required=True, type=Path)
    parser.add_argument(
        "--expected-complete-edge-baseline-bundle-sha256", required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = build_bundle(
        probe_manifest_path=args.probe_manifest,
        probe_root=args.probe_root,
        replay_attestation_path=args.replay_attestation,
        synthetic_resource_attestation_path=args.synthetic_resource_attestation,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        frozen_control_manifest_path=args.frozen_control_manifest,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
        complete_edge_baseline_bundle_path=args.complete_edge_baseline_bundle,
        expected_complete_edge_baseline_bundle_sha256=(
            args.expected_complete_edge_baseline_bundle_sha256
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bundle_bytes(bundle))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bundle_sha256": _sha_file(args.output.resolve()),
                "lock_attestation_sha256": bundle["lock_attestation_sha256"],
                "pre_gold_decision": bundle["pre_gold_decision"],
                "gold_open_authorized": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
