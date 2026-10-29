#!/usr/bin/env python3
"""Independently validate the authorized ordinary V2 batch and compact evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as core  # noqa: E402
from backend import f1_v2_cross_parent_opposite_fragment_rung_probe as v1_core  # noqa: E402
from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (  # noqa: E402
    MAPPING_PREFLIGHT_CRITERIA,
    _c001_locked_regression,
)


SCHEMA_VERSION = 1
KIND = "f1_v2_cross_parent_ordinary_batch_compact_evidence_v1"
PROFILE = "V2_CROSS_PARENT_OPPOSITE_FRAGMENT_RUNG_PROBE_V2"
ORDINARY_ARTIFACT_KIND = "f1_v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2"
PACKET_STATUS = "ORDINARY_V2_BATCH_COMPLETE_NEXT_PROFESSOR_PACKET"
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
EXPECTED_TARGET_ROWS = 4_788
EXPECTED_PRESENT_ROWS = 4_772
EXPECTED_ABSENT_ROWS = 16
EXPECTED_ELIGIBLE_PAIR_ROWS = 4_783
EXPECTED_ALL_PAIR_ROWS = 16_988
PEAK_RSS_LIMIT_BYTES = 16 * 1024**3
MAX_VALIDATOR_WORKERS = 8
EXPECTED_MAPPING_PREFLIGHT_SHA256 = (
    "9d314e7ebe5900f65c79039d8f729add3b8f1ad5cc4e190cc330103c9eec60cd"
)
EXPECTED_MAPPING_COMPACT_SHA256 = (
    "869f733f48344d3a8cb3a1bb8aae7a28c6e6e10a60e16c43819affc9bccd3954"
)
EXPECTED_TARGET_PROJECTION_SHA256 = (
    "04d964eb0d855948d5522fc0338dfbaa83c7e5b0bd51c87ebf303aeedc7d634b"
)
EXPECTED_RELATION_SHA256 = (
    "dd57cfe2787fc04eaf49ed28c878514e44988fa852935f127df8b2397f876042"
)
EXPECTED_MAPPING_OUTCOME_SHA256 = (
    "d7d9c422ecd861cb2684e9e4e749418308c57cc3eb96b32796f6a410c4a9d3e8"
)
EXPECTED_ALL_PAIR_SHA256 = (
    "cd957ef0c6b83f302fb6a5e19e42426d018e65b8b178229b7460d4806ecbbcb1"
)
EXPECTED_THRESHOLD_SHA256 = (
    "7abd75a9a89efdd66eb71b9828e6bcf3145b9b5c3d3c55c0d64f258f2ca0800f"
)
EXPECTED_ELIGIBLE_PAIR_IDS_SHA256 = (
    "de7af6f03457a2a5f45b6121fc2dd4ece74fe00a1a83459aac1d72c9b83ac08d"
)
EXPECTED_INFERENCE_MANIFEST_SHA256 = (
    "bb06613a96a063463bac951cd9a0aa3a31c7c0ce2b894ddab07261abfcae5f66"
)
EXPECTED_POOL_MANIFEST_SHA256 = (
    "c3ed8f6356c49c28ca56670cc9d061fe25dc4f1dea71625886e989330f8df3c7"
)
EXPECTED_CONTROL_MANIFEST_SHA256 = (
    "3b9dfa156746a01cb38195f7d75c92e23bff839228de96e425b8902edf262510"
)
EXPECTED_BASELINE_MANIFEST_SHA256 = (
    "4e26a8638b347a018678be51df1f6bd51b4bee865ed8c4b80155e5c867a758f5"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_KEYS = frozenset({
    "gold", "manual_truth", "sealed_gold", "truth_table", "expected_count",
    "ownership", "birth_columns", "learning", "solver", "replay_attestation",
    "synthetic_resource_attestation", "runtime_integration",
    "production_inspection", "main_branch_promotion",
})
DELETION_TABLE_KEYS = (
    "effective_mode_edge_table", "local_plane_fragment_table",
    "local_plane_orphan_table", "local_plane_pair_records",
    "all_pair_membership_table", "component_table", "geometry_gate_table",
)
CASE_FIELDS = frozenset({
    "case_token", "status", "case_artifact_path", "case_artifact_sha256",
    "scan_sha256", "candidate_pool_sha256", "a0_prediction_sha256",
    "frozen_control_probe_sha256", "complete_edge_baseline_probe_sha256",
    "mapping_case_sha256", "pre_rung_gate_criteria", "pre_rung_gate_sha256",
    "output_table_hashes", "case_counters", "ordinary_v2_criteria",
    "deletion_evidence_sha256", "wall_seconds", "peak_rss_bytes",
    "input_json_bytes", "serialized_output_bytes",
})
AGGREGATE_TABLE_SOURCE_FIELDS = {
    "target_eligible_pair_table": "target_eligible_pair_table",
    "frozen_target_projection": "frozen_target_projection",
    "frozen_target_baseline_relation_table": "frozen_target_baseline_relation_table",
    "all_pair_membership_table": "all_pair_membership_table",
    "complete_edge_before_component_table": "complete_edge_before_component_table",
    "added_cross_parent_rung_table": "added_cross_parent_rung_table",
    "replacement_component_table": "replacement_component_table",
    "component_rung_attribution_table": "component_rung_attribution_table",
    "frozen_target_outcome_table": "frozen_target_outcome_table",
    "certificate_novelty_table": "certificate_novelty_table",
    "global_deletion_counterfactual_table": "global_deletion_counterfactual",
}
TOP_FIELDS = frozenset({
    "schema_version", "kind", "profile", "artifact_mode", "research_only",
    "inference_safe", "prediction_mutated", "birth_columns_enabled",
    "ownership_consumed", "solver_consumed", "truth_inputs_consumed",
    "source_points_consumed", "source_points_serialized",
    "designated_replays_authorized", "full_synthetic_resource_run_authorized",
    "gold_open_authorized", "case_count", "successful_case_count", "workers",
    "isolated_process_per_case", "peak_rss_limit_bytes",
    "maximum_peak_rss_bytes", "inference_manifest_sha256",
    "candidate_pool_manifest_sha256", "frozen_control_manifest_sha256",
    "complete_edge_baseline_manifest_sha256",
    "mapping_preflight_external_sha256",
    "mapping_compact_evidence_external_sha256",
    "frozen_target_projection_sha256", "mapping_relation_table_sha256",
    "mapping_outcome_table_sha256", "all_pair_membership_table_sha256",
    "threshold_hash_table_sha256", "target_eligible_pair_ids_sha256",
    "target_eligible_pair_count", "cases", "case_payloads",
    "aggregate_tables", "aggregate_table_hashes", "aggregate_counters",
    "ordinary_v2_criteria", "all_ordinary_v2_criteria_valid",
    "packet_status", "next_professor_packet_authorized",
})


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(row: Mapping[str, Any], hash_field: str) -> str:
    return _sha_json({key: value for key, value in row.items() if key != hash_field})


def _require_sha(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _rows_by_token(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} rows must be objects")
        token = str(row.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be unique")
        result[token] = dict(row)
    if set(result) != set(EXPECTED_CASE_TOKENS):
        raise ValueError(f"{label} must contain exact C001-C037")
    return result


def _mapping_case_expectations(
    mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    def group(rows: Sequence[Mapping[str, Any]], *, strip_case: bool = False):
        result = {token: [] for token in EXPECTED_CASE_TOKENS}
        for raw in rows:
            row = deepcopy(dict(raw))
            token = str(row.get("case_token") or "")
            if token not in result:
                raise ValueError("mapping evidence contains foreign case token")
            if strip_case:
                del row["case_token"]
            result[token].append(row)
        return result

    projections = group(mapping["frozen_target_projection"])
    relations = group(mapping["frozen_target_baseline_relation_table"])
    outcomes = group(mapping["frozen_target_outcome_table"])
    memberships = group(mapping["all_pair_membership_table"], strip_case=True)
    thresholds = {
        str(row["case_token"]): str(row["thresholds_sha256"])
        for row in mapping["threshold_hash_table"]
    }
    if set(thresholds) != set(EXPECTED_CASE_TOKENS):
        raise ValueError("mapping threshold evidence is incomplete")
    return {
        token: {
            "frozen_target_projection": projections[token],
            "frozen_target_baseline_relation_table": relations[token],
            "frozen_target_outcome_table": outcomes[token],
            "all_pair_membership_table": memberships[token],
            "thresholds_sha256": thresholds[token],
        }
        for token in EXPECTED_CASE_TOKENS
    }


def _expected_aggregate_tables(
    case_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        key: [] for key in AGGREGATE_TABLE_SOURCE_FIELDS
    }
    for token in EXPECTED_CASE_TOKENS:
        payload = case_payloads[token]
        for aggregate_name, source_name in AGGREGATE_TABLE_SOURCE_FIELDS.items():
            raw = payload[source_name]
            rows = [raw] if isinstance(raw, Mapping) else list(raw)
            for row in rows:
                value = deepcopy(dict(row))
                if "case_token" in value:
                    if str(value["case_token"]) != token:
                        raise ValueError("case-token mismatch while aggregating tables")
                else:
                    value = {"case_token": token, **value}
                result[aggregate_name].append(value)
    return result


def _derive_independent_pre_rung_gate(
    *, token: str, source_mapping: Mapping[str, Any],
    expected_projection: Sequence[Mapping[str, Any]],
    expected_relations: Sequence[Mapping[str, Any]],
    expected_outcomes: Sequence[Mapping[str, Any]],
    expected_memberships: Sequence[Mapping[str, Any]],
    expected_threshold_sha256: str, source_hashes_match: bool,
) -> dict[str, bool]:
    criteria = {key: False for key in MAPPING_PREFLIGHT_CRITERIA}
    projection = list(source_mapping["frozen_target_projection"])
    relations = list(source_mapping["frozen_target_baseline_relation_table"])
    outcomes = list(source_mapping["frozen_target_outcome_table"])
    memberships = list(source_mapping["all_pair_membership_table"])
    relation_projection = core.frozen_target_projection(relations)
    expected_projection = [dict(row) for row in expected_projection]
    expected_relations = [dict(row) for row in expected_relations]
    expected_outcomes = [dict(row) for row in expected_outcomes]
    expected_memberships = [dict(row) for row in expected_memberships]
    expected_count = len(expected_projection)
    ids = [row["frozen_target_component_id"] for row in relations]
    expected_ids = [row["frozen_target_component_id"] for row in expected_projection]
    mode_hashes = sorted(row["frozen_target_mode_ids_sha256"] for row in relations)
    expected_mode_hashes = sorted(
        row["frozen_target_mode_ids_sha256"] for row in expected_projection
    )
    criteria.update({
        "all_declared_source_hashes_match": bool(source_hashes_match),
        "all_frozen_control_probe_hashes_reproduce": True,
        "all_complete_edge_baseline_hashes_reproduce": True,
        "frozen_target_count_is_4788": len(projection) == expected_count,
        "frozen_target_rows_are_byte_identical": (
            projection == relation_projection == expected_projection
        ),
        "frozen_target_id_sets_are_exactly_equal": (
            ids == expected_ids and len(ids) == len(set(ids))
        ),
        "frozen_target_mode_hash_sets_are_exactly_equal": (
            mode_hashes == expected_mode_hashes
        ),
        "every_target_mode_has_one_baseline_fragment": all(
            all(fragment_id is not None for _mode_id, fragment_id in row[
                "baseline_mode_fragment_rows"
            ]) for row in relations
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
        "no_unrecognised_relation_status_exists": all(
            row["target_baseline_relation_status"] in {
                core.RELATION_COMPONENT_UNIQUE,
                core.RELATION_COMPONENT_NONE_OUTSIDE,
            } for row in relations
        ),
        "relation_row_count_is_4788": (
            relations == expected_relations and len(relations) == expected_count
        ),
        "relation_table_is_untruncated": (
            relations == expected_relations
            and len(ids) == len(set(ids)) == expected_count
        ),
        "outcome_row_count_is_4788": (
            outcomes == expected_outcomes and len(outcomes) == expected_count
        ),
        "c001_locked_regression_matches": (
            token != "C001" or _c001_locked_regression(relations)
        ),
        "all_threshold_hashes_are_unchanged": (
            _sha_json(source_mapping["thresholds"]) == expected_threshold_sha256
        ),
        "all_pair_line_tube_memberships_are_unchanged": (
            memberships == expected_memberships
        ),
        "gold_open_authorized_false": (
            source_mapping.get("gold_open_authorized") is False
            and source_mapping.get("truth_inputs_consumed") is False
            and source_mapping.get("ordinary_batch_authorized") is False
        ),
    })
    return criteria


def _assert_pre_rung_gate_derived(
    asserted: Mapping[str, Any], derived: Mapping[str, bool], declared_sha256: Any,
) -> None:
    if (
        set(asserted) != set(MAPPING_PREFLIGHT_CRITERIA)
        or set(derived) != set(MAPPING_PREFLIGHT_CRITERIA)
        or any(derived[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA)
        or dict(asserted) != dict(derived)
        or _require_sha(declared_sha256, "pre-rung gate SHA")
        != _sha_json(asserted)
    ):
        raise ValueError("per-case pre-rung gate is copied, forged, or underived")


def _walk_forbidden(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden ordinary-V2 key at {path}.{key}")
            _walk_forbidden(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _walk_forbidden(nested, f"{path}[{index}]")


def _assert_external_sha(path: Path, expected_sha256: str) -> str:
    expected = _require_sha(expected_sha256, "expected authoritative SHA")
    actual = _sha_file(path)
    if actual != expected:
        raise ValueError("ordinary authoritative external SHA mismatch")
    return actual


def _assert_exact_table(
    rows: Any, *, expected_count: int, id_fields: tuple[str, ...],
    declared_sha256: str, label: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = [dict(row) for row in rows if isinstance(row, Mapping)]
    keys = [tuple(str(row.get(field) or "") for field in id_fields) for row in result]
    if (
        len(result) != expected_count
        or keys != sorted(keys)
        or len(set(keys)) != expected_count
    ):
        raise ValueError(f"{label} exact count/order/identity mismatch")
    if _require_sha(declared_sha256, f"{label} SHA") != _sha_json(result):
        raise ValueError(f"{label} SHA mismatch")
    return result


def _derive_target_eligible_pair_ids(
    relation_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    pair_ids = sorted({str(row["baseline_local_plane_pair_id"]) for row in relation_rows})
    if len(pair_ids) != EXPECTED_ELIGIBLE_PAIR_ROWS:
        raise ValueError("target-eligible pair population is not exact 4,783")
    if _sha_json(pair_ids) != EXPECTED_ELIGIBLE_PAIR_IDS_SHA256:
        raise ValueError("target-eligible pair ID-set hash mismatch")
    return pair_ids


def _assert_structural_negative_outcome(row: Mapping[str, Any]) -> None:
    null_fields = (
        "complete_edge_baseline_component_id", "replacement_component_id",
        "support_parent_count_before", "support_parent_count_after",
        "occupied_bin_count_before", "occupied_bin_count_after",
        "left_side_bin_count_before", "left_side_bin_count_after",
        "right_side_bin_count_before", "right_side_bin_count_after",
        "transition_rung_bin_count_before", "transition_rung_bin_count_after",
        "geometry_gate_depth_before", "geometry_gate_depth_after",
        "first_rejection_code_before", "first_rejection_code_after",
    )
    if any(row.get(field) is not None for field in null_fields):
        raise ValueError("structural-negative geometry must remain null, not zero")
    if (
        int(row.get("causal_added_cross_parent_rung_count") or 0) != 0
        or row.get("causal_added_cross_parent_rung_ids_sha256") != _sha_json([])
        or row.get("geometry_certificate_valid_after") is not False
        or row.get("target_reached_by_added_rung") is not False
        or row.get("target_side_gate_advanced") is not False
        or row.get("target_novel_rung_attributable") is not False
        or row.get("target_outcome_code")
        != core.OUTCOME_STRUCTURALLY_UNREACHABLE
    ):
        raise ValueError("structural-negative outcome illegally claims rung effect")


def _assert_executed_deletion_restoration(
    *, rebuilt_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    frozen_source_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    asserted_valid: Any,
) -> dict[str, str]:
    """Require raw canonical row equality; asserted hashes/True never suffice."""
    if set(rebuilt_tables) != set(DELETION_TABLE_KEYS):
        raise ValueError("executed deletion rebuilt-table schema mismatch")
    if set(frozen_source_tables) != set(DELETION_TABLE_KEYS):
        raise ValueError("frozen deletion source-table schema mismatch")
    derived_hashes: dict[str, str] = {}
    all_equal = True
    for name in DELETION_TABLE_KEYS:
        rebuilt = [dict(row) for row in rebuilt_tables[name]]
        frozen = [dict(row) for row in frozen_source_tables[name]]
        equal = rebuilt == frozen
        all_equal = all_equal and equal
        derived_hashes[f"{name}_sha256"] = _sha_json(rebuilt)
    if not all_equal:
        raise ValueError("executed deletion raw canonical tables do not restore baseline")
    if asserted_valid is not True:
        raise ValueError("executed deletion validity predicate is false")
    return derived_hashes


def _derive_frozen_deletion_source_tables(
    *, projected_input: Mapping[str, Any],
    complete_edge_baseline: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Rebuild the six deletion baselines directly from frozen source rows."""
    context = v1_core._baseline_context(
        deepcopy(dict(projected_input)), deepcopy(dict(complete_edge_baseline))
    )
    pair_contexts = {
        pair_id: v1_core._pair_context(pair, context)
        for pair_id, pair in sorted(context["pair_by_id"].items())
    }
    components = []
    for pair_id, pair_context in sorted(pair_contexts.items()):
        for component in sorted(
            pair_context["baseline_components"],
            key=lambda item: str(item["component_id"]),
        ):
            components.append({
                "baseline_local_plane_pair_id": str(pair_id),
                "component_record": deepcopy(component["record"]),
            })
    geometry = [
        {
            "local_transition_component_id": row["component_record"][
                "local_transition_component_id"
            ],
            "geometry_gate_depth": row["component_record"]["geometry_gate_depth"],
            "geometry_gates": deepcopy(row["component_record"]["geometry_gates"]),
            "first_rejection_code": row["component_record"]["first_rejection_code"],
            "geometry_certificate_valid": row["component_record"][
                "geometry_certificate_valid"
            ],
        }
        for row in components
    ]
    return {
        "effective_mode_edge_table": sorted(
            (deepcopy(dict(row)) for row in context["effective_edges"]),
            key=lambda row: str(row["mode_edge_id"]),
        ),
        "local_plane_fragment_table": sorted(
            (deepcopy(fragment.record) for fragment in context["fragments"]),
            key=lambda row: str(row["local_plane_fragment_id"]),
        ),
        "local_plane_orphan_table": sorted(
            (deepcopy(dict(row)) for row in context["orphans"]),
            key=lambda row: str(row["local_plane_orphan_id"]),
        ),
        "local_plane_pair_records": sorted(
            (
                deepcopy(dict(row))
                for row in complete_edge_baseline["local_plane_pair_records"]
            ),
            key=lambda row: str(row["local_plane_pair_id"]),
        ),
        "all_pair_membership_table": core.build_all_pair_membership_rows(
            pair_contexts
        ),
        "component_table": components,
        "geometry_gate_table": geometry,
    }


def _assert_deletion_input_binding(
    *, added_rung_table: Sequence[Mapping[str, Any]],
    deleted_rung_count: Any, deleted_rung_table_sha256: Any,
) -> None:
    """Bind deletion to the exact ordinary intervention, never only its count."""
    added = [dict(row) for row in added_rung_table]
    if int(deleted_rung_count) != len(added):
        raise ValueError("deletion rung count does not bind the ordinary rung table")
    if _require_sha(
        deleted_rung_table_sha256, "deleted added-rung table SHA"
    ) != _sha_json(added):
        raise ValueError("deletion added-rung table SHA does not bind ordinary input")


def _assert_criteria_derived(
    asserted: Mapping[str, Any], derived: Mapping[str, bool],
) -> None:
    if set(asserted) != set(core.ORDINARY_V2_CRITERION_KEYS):
        raise ValueError("ordinary-V2 criterion schema mismatch")
    if set(derived) != set(core.ORDINARY_V2_CRITERION_KEYS):
        raise ValueError("ordinary-V2 derived criterion schema mismatch")
    for key in core.ORDINARY_V2_CRITERION_KEYS:
        if derived[key] is not True or asserted.get(key) is not derived[key]:
            raise ValueError(f"ordinary-V2 asserted criterion is not derived: {key}")


def _expected_eligible_pair_table(
    relation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    targets_by_pair: dict[str, list[str]] = {}
    for row in relation_rows:
        targets_by_pair.setdefault(
            str(row["baseline_local_plane_pair_id"]), []
        ).append(str(row["frozen_target_component_id"]))
    result = []
    for pair_id, target_ids in sorted(targets_by_pair.items()):
        ids = sorted(target_ids)
        result.append({
            "baseline_local_plane_pair_id": pair_id,
            "frozen_target_component_ids": ids,
            "frozen_target_component_ids_sha256": _sha_json(ids),
        })
    return result


def _ordinary_deletion_tables(
    deletion: Mapping[str, Any], *, prefix: str,
) -> dict[str, Sequence[Mapping[str, Any]]]:
    return {
        "effective_mode_edge_table": deletion[f"{prefix}_effective_edge_table"],
        "local_plane_fragment_table": deletion[f"{prefix}_fragment_table"],
        "local_plane_orphan_table": deletion[f"{prefix}_orphan_table"],
        "local_plane_pair_records": deletion[f"{prefix}_pair_records"],
        "all_pair_membership_table": deletion[
            f"{prefix}_all_pair_membership_table"
        ],
        "component_table": deletion[f"{prefix}_component_table"],
        "geometry_gate_table": deletion[f"{prefix}_geometry_gate_table"],
    }


def _derive_case_criteria(
    *, payload: Mapping[str, Any], projected: Mapping[str, Any],
    source_mapping: Mapping[str, Any], source_baseline: Mapping[str, Any],
    source_deletion_tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, bool]:
    relations = list(payload["frozen_target_baseline_relation_table"])
    eligible = list(payload["target_eligible_pair_table"])
    rungs = list(payload["added_cross_parent_rung_table"])
    outcomes = list(payload["frozen_target_outcome_table"])
    replacements = list(payload["replacement_component_table"])
    replacement_pairs = list(payload["local_plane_pair_records"])
    deletion = dict(payload["global_deletion_counterfactual"])
    eligible_ids = {row["baseline_local_plane_pair_id"] for row in eligible}
    all_pair_by_id = {
        row["baseline_local_plane_pair_id"]: set(row["transverse_parent_ids"])
        for row in source_mapping["all_pair_membership_table"]
    }
    context = v1_core._baseline_context(projected, source_baseline)
    fragment_of_mode = context["fragment_of_mode"]
    orphan_modes = set(context["orphan_modes"])
    present = [
        row for row in outcomes
        if row["target_baseline_relation_status"] == core.RELATION_COMPONENT_UNIQUE
    ]
    structural = [
        row for row in outcomes
        if row["target_baseline_relation_status"] == core.RELATION_COMPONENT_NONE_OUTSIDE
    ]
    replacement_ids = {
        row["component_record"]["local_transition_component_id"]
        for row in replacements
    }
    source_pair_ids = {
        row["local_plane_pair_id"]
        for row in source_baseline["local_plane_pair_records"]
    }
    derived = {key: False for key in core.ORDINARY_V2_CRITERION_KEYS}
    derived.update({
        "profile_identity_valid": (
            payload.get("schema_version") == core.PROBE_SCHEMA_VERSION
            and payload.get("kind") == core.PROBE_KIND
            and payload.get("profile") == core.PRIMITIVE_PROFILE
            and payload.get("artifact_mode") == "ORDINARY_V2_CASE"
        ),
        "input_projection_valid": tuple(projected) == tuple(core.ALLOWED_INPUT_FIELDS),
        "frozen_control_reproduced": True,
        "complete_edge_baseline_reproduced": True,
        "mapping_case_reproduced": (
            payload["mapping_case_sha256"] == _sha_json(source_mapping)
        ),
        "target_pair_population_exact": eligible
        == _expected_eligible_pair_table(relations),
        "rung_parents_distinct": all(
            len(set(row["parent_atom_ids"])) == 2 for row in rungs
        ),
        "rung_parent_voxels_adjacent": all(
            max(
                abs(int(left) - int(right))
                for left, right in zip(
                    row["parent_voxel_keys"][0], row["parent_voxel_keys"][1]
                )
            ) == 1
            for row in rungs
        ),
        "rungs_join_opposite_fixed_fragments": all(
            [fragment_of_mode.get(mode_id) for mode_id in row["mode_ids"]]
            == list(row["baseline_fragment_ids"])
            for row in rungs
        ),
        "rungs_use_target_eligible_pairs_only": all(
            row["baseline_local_plane_pair_id"] in eligible_ids for row in rungs
        ),
        "rung_parents_inside_frozen_tube": all(
            set(row["parent_atom_ids"])
            <= all_pair_by_id[row["baseline_local_plane_pair_id"]]
            for row in rungs
        ),
        "orphan_modes_excluded": all(
            not (set(row["mode_ids"]) & orphan_modes) for row in rungs
        ),
        "axial_rule_exact": all(
            float(row["axial_parent_separation_mm"])
            <= float(row["axial_bin_width_mm"])
            for row in rungs
        ),
        "no_pair_births": {
            row["local_plane_pair_id"] for row in replacement_pairs
        } == source_pair_ids,
        "no_fragment_or_orphan_births": (
            payload["baseline_local_plane_fragment_table"]
            == source_baseline["local_plane_fragment_table"]
            and payload["baseline_local_plane_orphan_table"]
            == source_baseline["local_plane_orphan_table"]
        ),
        "present_target_replacement_mapping_unique": all(
            isinstance(row["replacement_component_id"], str)
            and row["replacement_component_id"] in replacement_ids
            for row in present
        ),
        "structural_negative_outcomes_forced": all(
            _structural_negative_is_valid(row) for row in structural
        ),
        "one_parent_components_zero_span": all(
            int(row["component_record"]["support_parent_count"]) != 1
            or row["component_record"]["intersection_span_endpoints"][0]
            == row["component_record"]["intersection_span_endpoints"][1]
            for row in replacements
        ),
        "global_deletion_counterfactual_executed_and_valid": (
            deletion["executed_independent_rebuild"] is True
            and deletion["all_required_equalities"] is True
            and _ordinary_deletion_tables(deletion, prefix="rebuilt")
            == source_deletion_tables
            and _ordinary_deletion_tables(deletion, prefix="frozen")
            == source_deletion_tables
        ),
        "all_structural_caps_satisfied": (
            len(rungs) <= v1_core.MAX_ADDED_CROSS_PARENT_RUNGS
            and len(replacements) <= core.lpf.MAX_LOCAL_TRANSITION_COMPONENTS
        ),
        "safety_flags_valid": all(
            payload.get(field) is expected
            for field, expected in {
                "research_only": True, "inference_safe": True,
                "prediction_mutated": False, "birth_columns_enabled": False,
                "ownership_consumed": False, "solver_consumed": False,
                "truth_inputs_consumed": False, "source_points_consumed": False,
                "source_points_serialized": False,
                "designated_replays_authorized": False,
                "full_synthetic_resource_run_authorized": False,
                "gold_open_authorized": False,
            }.items()
        ),
    })
    return derived


def _structural_negative_is_valid(row: Mapping[str, Any]) -> bool:
    try:
        _assert_structural_negative_outcome(row)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def validate_ordinary_case_independently(
    *, payload: Mapping[str, Any], raw_candidate_pool: Mapping[str, Any],
    frozen_control: Mapping[str, Any], complete_edge_baseline: Mapping[str, Any],
    expected_pre_rung_evidence: Mapping[str, Any] | None = None,
    asserted_pre_rung_gate: Mapping[str, Any] | None = None,
    asserted_pre_rung_gate_sha256: Any = None,
) -> dict[str, Any]:
    """Validate one result against source rows without trusting attestations."""
    if set(payload) != set(core.ORDINARY_TOP_LEVEL_FIELDS):
        raise ValueError("ordinary V2 case top-level schema mismatch")
    _walk_forbidden(payload)
    core.validate_v2_cross_parent_ordinary_case_payload(payload)
    projected = core.project_mapping_preflight_input(raw_candidate_pool)
    if _sha_json(frozen_control) != payload["frozen_control_probe_sha256"]:
        raise ValueError("ordinary case frozen-control source hash mismatch")
    if _sha_json(complete_edge_baseline) != payload[
        "complete_edge_baseline_probe_sha256"
    ]:
        raise ValueError("ordinary case complete-edge source hash mismatch")
    source_mapping = core._build_case_from_payloads(
        projected,
        case_token=str(payload["case_token"]),
        scan_sha256=str(payload["scan_sha256"]),
        candidate_pool_sha256=str(payload["candidate_pool_sha256"]),
        a0_prediction_sha256=str(payload["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(payload["frozen_control_probe_sha256"]),
        complete_edge_baseline_probe_sha256=str(
            payload["complete_edge_baseline_probe_sha256"]
        ),
        frozen_control=frozen_control,
        complete_edge_baseline=complete_edge_baseline,
    )
    if expected_pre_rung_evidence is not None:
        if asserted_pre_rung_gate is None:
            raise ValueError("ordinary case lacks asserted per-case pre-rung gate")
        derived_pre_gate = _derive_independent_pre_rung_gate(
            token=str(payload["case_token"]), source_mapping=source_mapping,
            expected_projection=expected_pre_rung_evidence[
                "frozen_target_projection"
            ],
            expected_relations=expected_pre_rung_evidence[
                "frozen_target_baseline_relation_table"
            ],
            expected_outcomes=expected_pre_rung_evidence[
                "frozen_target_outcome_table"
            ],
            expected_memberships=expected_pre_rung_evidence[
                "all_pair_membership_table"
            ],
            expected_threshold_sha256=str(
                expected_pre_rung_evidence["thresholds_sha256"]
            ),
            source_hashes_match=True,
        )
        _assert_pre_rung_gate_derived(
            asserted_pre_rung_gate, derived_pre_gate,
            asserted_pre_rung_gate_sha256,
        )
    for field in (
        "frozen_target_projection", "frozen_target_baseline_relation_table",
        "all_pair_membership_table",
    ):
        if payload[field] != source_mapping[field]:
            raise ValueError(f"ordinary case source-bound mapping mismatch: {field}")
    if payload["thresholds"] != complete_edge_baseline["thresholds"]:
        raise ValueError("ordinary case source-bound threshold mismatch")
    expected_eligible = _expected_eligible_pair_table(
        source_mapping["frozen_target_baseline_relation_table"]
    )
    if payload["target_eligible_pair_table"] != expected_eligible:
        raise ValueError("ordinary case target-eligible pair table mismatch")
    source_deletion = _derive_frozen_deletion_source_tables(
        projected_input=projected,
        complete_edge_baseline=complete_edge_baseline,
    )
    deletion = dict(payload["global_deletion_counterfactual"])
    _assert_deletion_input_binding(
        added_rung_table=payload["added_cross_parent_rung_table"],
        deleted_rung_count=deletion["deleted_added_cross_parent_rung_count"],
        deleted_rung_table_sha256=deletion[
            "deleted_added_cross_parent_rung_table_sha256"
        ],
    )
    if deletion["deleted_added_cross_parent_rung_table"] != payload[
        "added_cross_parent_rung_table"
    ]:
        raise ValueError("deletion raw input table differs from ordinary rungs")
    _assert_executed_deletion_restoration(
        rebuilt_tables=_ordinary_deletion_tables(deletion, prefix="rebuilt"),
        frozen_source_tables=source_deletion,
        asserted_valid=deletion["all_required_equalities"],
    )
    _assert_executed_deletion_restoration(
        rebuilt_tables=_ordinary_deletion_tables(deletion, prefix="frozen"),
        frozen_source_tables=source_deletion,
        asserted_valid=deletion["all_required_equalities"],
    )
    relations = list(payload["frozen_target_baseline_relation_table"])
    outcomes = list(payload["frozen_target_outcome_table"])
    if len(relations) != len(outcomes):
        raise ValueError("ordinary case target outcome table is incomplete")
    relation_by_id = {
        row["frozen_target_component_id"]: row for row in relations
    }
    rung_ids = {row["mode_edge_id"] for row in payload["added_cross_parent_rung_table"]}
    present_count = 0
    absent_count = 0
    for outcome in outcomes:
        target_id = outcome["frozen_target_component_id"]
        relation = relation_by_id.get(target_id)
        if relation is None:
            raise ValueError("ordinary outcome target absent from frozen relation")
        if outcome["target_outcome_sha256"] != _record_hash(
            outcome, "target_outcome_sha256"
        ):
            raise ValueError("ordinary target outcome row SHA mismatch")
        if relation["target_baseline_relation_status"] == core.RELATION_COMPONENT_NONE_OUTSIDE:
            _assert_structural_negative_outcome(outcome)
            absent_count += 1
        else:
            present_count += 1
            if not isinstance(outcome["replacement_component_id"], str):
                raise ValueError("present target lacks unique replacement component")
        causal_count = int(outcome["causal_added_cross_parent_rung_count"])
        if causal_count < 0:
            raise ValueError("ordinary target causal-rung count is negative")
        # Exact source rebuild below binds the full attribution table; the
        # outcome hash must at minimum bind the sorted causal ID set digest.
        if not _SHA_RE.fullmatch(str(outcome["causal_added_cross_parent_rung_ids_sha256"])):
            raise ValueError("ordinary target causal-rung hash invalid")
    derived = _derive_case_criteria(
        payload=payload, projected=projected, source_mapping=source_mapping,
        source_baseline=complete_edge_baseline,
        source_deletion_tables=source_deletion,
    )
    _assert_criteria_derived(payload["ordinary_v2_criteria"], derived)
    if payload["all_ordinary_v2_criteria_valid"] is not True:
        raise ValueError("ordinary case aggregate criteria are not valid")
    if payload["packet_status"] != "ORDINARY_V2_CASE_COMPLETE":
        raise ValueError("ordinary case packet status is not complete")
    # The core call is an independent executable rebuild from projected frozen
    # inputs.  Canonical equality detects forged complete/attribution tables.
    rebuilt = core.build_v2_cross_parent_ordinary_case(
        projected,
        case_token=str(payload["case_token"]),
        scan_sha256=str(payload["scan_sha256"]),
        candidate_pool_sha256=str(payload["candidate_pool_sha256"]),
        a0_prediction_sha256=str(payload["a0_prediction_sha256"]),
        frozen_control_probe_sha256=str(payload["frozen_control_probe_sha256"]),
        complete_edge_baseline_probe_sha256=str(
            payload["complete_edge_baseline_probe_sha256"]
        ),
        expected_mapping_case_sha256=str(payload["mapping_case_sha256"]),
    )
    if _canonical_bytes(rebuilt) != _canonical_bytes(payload):
        raise ValueError("ordinary case independent canonical rebuild mismatch")
    return {
        "case_token": str(payload["case_token"]),
        "case_sha256": _sha_json(payload),
        "target_count": len(outcomes),
        "present_count": present_count,
        "absent_count": absent_count,
        "eligible_pair_count": len(expected_eligible),
        "all_pair_count": len(payload["all_pair_membership_table"]),
        "added_rung_count": len(payload["added_cross_parent_rung_table"]),
        "target_reached_count": sum(
            row["target_reached_by_added_rung"] is True for row in outcomes
        ),
        "target_side_gate_advanced_count": sum(
            row["target_side_gate_advanced"] is True for row in outcomes
        ),
        "target_novel_rung_attributable_count": sum(
            row["target_novel_rung_attributable"] is True for row in outcomes
        ),
        "deletion_evidence_sha256": deletion["deletion_evidence_sha256"],
    }


def _validate_case_task(task: tuple[Any, ...]) -> dict[str, Any]:
    """Fresh-process ordinary case validation over paths and small evidence."""
    (
        token, case_path_raw, pool_path_raw, control_path_raw,
        baseline_path_raw, scan_path_raw, a0_path_raw, manifest_row_raw,
        expected_pre_rung_evidence,
    ) = task
    case_path = Path(case_path_raw)
    pool_path = Path(pool_path_raw)
    control_path = Path(control_path_raw)
    baseline_path = Path(baseline_path_raw)
    scan_path = Path(scan_path_raw)
    a0_path = Path(a0_path_raw)
    manifest_row = dict(manifest_row_raw)
    case_payload = _read(case_path)
    if _sha_file(case_path) != _require_sha(
        manifest_row["case_artifact_sha256"], f"{token}.ordinary case SHA"
    ):
        raise ValueError(f"{token}: persisted ordinary case SHA mismatch")
    for field in (
        "scan_sha256", "candidate_pool_sha256", "a0_prediction_sha256",
        "frozen_control_probe_sha256", "complete_edge_baseline_probe_sha256",
        "mapping_case_sha256",
    ):
        if manifest_row[field] != case_payload[field]:
            raise ValueError(f"{token}: manifest/case binding mismatch: {field}")
    if manifest_row["ordinary_v2_criteria"] != case_payload[
        "ordinary_v2_criteria"
    ]:
        raise ValueError(f"{token}: manifest criteria differ from case payload")
    if manifest_row["output_table_hashes"] != case_payload["output_table_hashes"]:
        raise ValueError(f"{token}: manifest table hashes differ from case payload")
    if manifest_row["case_counters"] != case_payload["case_counters"]:
        raise ValueError(f"{token}: manifest counters differ from case payload")
    if manifest_row["deletion_evidence_sha256"] != case_payload[
        "global_deletion_counterfactual"
    ]["deletion_evidence_sha256"]:
        raise ValueError(f"{token}: manifest deletion hash differs from case payload")
    pre_gate = manifest_row["pre_rung_gate_criteria"]
    if (
        not isinstance(pre_gate, Mapping)
        or set(pre_gate) != set(MAPPING_PREFLIGHT_CRITERIA)
        or any(pre_gate[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA)
        or manifest_row["pre_rung_gate_sha256"] != _sha_json(pre_gate)
    ):
        raise ValueError(f"{token}: pre-rung gate is not exact/source-derived")
    for field, path, expected in (
        ("scan_sha256", scan_path, case_payload["scan_sha256"]),
        ("candidate_pool_sha256", pool_path, case_payload["candidate_pool_sha256"]),
        ("a0_prediction_sha256", a0_path, case_payload["a0_prediction_sha256"]),
        ("frozen_control_probe_sha256", control_path, case_payload[
            "frozen_control_probe_sha256"
        ]),
        ("complete_edge_baseline_probe_sha256", baseline_path, case_payload[
            "complete_edge_baseline_probe_sha256"
        ]),
    ):
        if not path.is_file() or _sha_file(path) != _require_sha(
            expected, f"{token}.{field}"
        ):
            raise ValueError(f"{token}: source byte hash mismatch: {field}")
    summary = validate_ordinary_case_independently(
        payload=case_payload,
        raw_candidate_pool=_read(pool_path),
        frozen_control=_read(control_path),
        complete_edge_baseline=_read(baseline_path),
        expected_pre_rung_evidence=expected_pre_rung_evidence,
        asserted_pre_rung_gate=pre_gate,
        asserted_pre_rung_gate_sha256=manifest_row["pre_rung_gate_sha256"],
    )
    rss = int(manifest_row.get("peak_rss_bytes") or -1)
    if rss < 0 or rss >= PEAK_RSS_LIMIT_BYTES:
        raise ValueError(f"{token}: ordinary case RSS cap breached")
    summary["peak_rss_bytes"] = rss
    return summary


def _ordered_case_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["case_token"]))
    if [row["case_token"] for row in ordered] != list(EXPECTED_CASE_TOKENS):
        raise ValueError("independent validation result order/census mismatch")
    return ordered


def validate_and_compact(
    *, authoritative_artifact_path: Path,
    expected_authoritative_sha256: str,
    mapping_preflight_path: Path, mapping_compact_evidence_path: Path,
    inference_manifest_path: Path, candidate_pool_manifest_path: Path,
    a0_prediction_root: Path, frozen_control_manifest_path: Path,
    complete_edge_baseline_manifest_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Validate all 37 persisted ordinary case artifacts and compact evidence."""
    if workers < 1 or workers > MAX_VALIDATOR_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_VALIDATOR_WORKERS}]")
    authoritative_artifact_path = authoritative_artifact_path.resolve()
    authoritative_sha = _assert_external_sha(
        authoritative_artifact_path, expected_authoritative_sha256
    )
    manifest = _read(authoritative_artifact_path)
    _walk_forbidden(manifest)
    if set(manifest) != set(TOP_FIELDS):
        raise ValueError("ordinary V2 aggregate top-level schema mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != ORDINARY_ARTIFACT_KIND
        or manifest.get("profile") != PROFILE
        or manifest.get("artifact_mode") != "ORDINARY_V2_BATCH"
    ):
        raise ValueError("ordinary V2 aggregate identity rejects V1/foreign artifact")
    expected_flags = {
        "research_only": True, "inference_safe": True,
        "prediction_mutated": False, "birth_columns_enabled": False,
        "ownership_consumed": False, "solver_consumed": False,
        "truth_inputs_consumed": False, "source_points_consumed": False,
        "source_points_serialized": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "gold_open_authorized": False,
    }
    for field, expected in expected_flags.items():
        if manifest.get(field) is not expected:
            raise ValueError(f"ordinary aggregate safety mismatch: {field}")
    if (
        int(manifest.get("case_count") or -1) != 37
        or int(manifest.get("successful_case_count") or -1) != 37
        or int(manifest.get("workers") or -1) not in range(1, 9)
        or manifest.get("isolated_process_per_case") is not True
        or int(manifest.get("peak_rss_limit_bytes") or -1) != PEAK_RSS_LIMIT_BYTES
    ):
        raise ValueError("ordinary aggregate case/isolation/RSS envelope mismatch")
    source_paths = {
        "inference_manifest_sha256": (
            inference_manifest_path.resolve(), EXPECTED_INFERENCE_MANIFEST_SHA256
        ),
        "candidate_pool_manifest_sha256": (
            candidate_pool_manifest_path.resolve(), EXPECTED_POOL_MANIFEST_SHA256
        ),
        "frozen_control_manifest_sha256": (
            frozen_control_manifest_path.resolve(), EXPECTED_CONTROL_MANIFEST_SHA256
        ),
        "complete_edge_baseline_manifest_sha256": (
            complete_edge_baseline_manifest_path.resolve(),
            EXPECTED_BASELINE_MANIFEST_SHA256,
        ),
        "mapping_preflight_external_sha256": (
            mapping_preflight_path.resolve(), EXPECTED_MAPPING_PREFLIGHT_SHA256
        ),
        "mapping_compact_evidence_external_sha256": (
            mapping_compact_evidence_path.resolve(), EXPECTED_MAPPING_COMPACT_SHA256
        ),
    }
    for field, (path, frozen_sha) in source_paths.items():
        actual = _sha_file(path)
        if actual != frozen_sha or manifest.get(field) != frozen_sha:
            raise ValueError(f"ordinary aggregate frozen binding mismatch: {field}")
    mapping = _read(mapping_preflight_path)
    mapping_compact = _read(mapping_compact_evidence_path)
    if (
        mapping.get("frozen_target_projection_sha256")
        != EXPECTED_TARGET_PROJECTION_SHA256
        or mapping.get("relation_table_sha256") != EXPECTED_RELATION_SHA256
        or mapping.get("outcome_table_sha256") != EXPECTED_MAPPING_OUTCOME_SHA256
        or mapping.get("all_pair_membership_table_sha256")
        != EXPECTED_ALL_PAIR_SHA256
        or mapping.get("threshold_hash_table_sha256")
        != EXPECTED_THRESHOLD_SHA256
        or mapping_compact.get("mapping_preflight_sha256")
        != EXPECTED_MAPPING_PREFLIGHT_SHA256
    ):
        raise ValueError("ordinary aggregate mapping evidence binding mismatch")
    mapping_expectations = _mapping_case_expectations(mapping)
    for field, expected in {
        "frozen_target_projection_sha256": EXPECTED_TARGET_PROJECTION_SHA256,
        "mapping_relation_table_sha256": EXPECTED_RELATION_SHA256,
        "mapping_outcome_table_sha256": EXPECTED_MAPPING_OUTCOME_SHA256,
        "all_pair_membership_table_sha256": EXPECTED_ALL_PAIR_SHA256,
        "threshold_hash_table_sha256": EXPECTED_THRESHOLD_SHA256,
        "target_eligible_pair_ids_sha256": EXPECTED_ELIGIBLE_PAIR_IDS_SHA256,
        "target_eligible_pair_count": EXPECTED_ELIGIBLE_PAIR_ROWS,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"ordinary aggregate frozen mapping binding changed: {field}")
    inference_rows = _rows_by_token(_read(inference_manifest_path), "inference")
    pool_rows = _rows_by_token(_read(candidate_pool_manifest_path), "candidate pool")
    control_rows = _rows_by_token(_read(frozen_control_manifest_path), "control")
    baseline_rows = _rows_by_token(
        _read(complete_edge_baseline_manifest_path), "complete-edge baseline"
    )
    manifest_rows = _rows_by_token(manifest, "ordinary aggregate")
    embedded_case_payloads = _rows_by_token(
        {"cases": manifest.get("case_payloads")}, "embedded ordinary payloads"
    )
    aggregate_counters = manifest.get("aggregate_counters")
    if not isinstance(aggregate_counters, Mapping):
        raise ValueError("ordinary aggregate counters must be an object")
    tasks = []
    for token in EXPECTED_CASE_TOKENS:
        manifest_row = manifest_rows[token]
        if set(manifest_row) != set(CASE_FIELDS) or manifest_row.get("status") != "SUCCESS":
            raise ValueError(f"{token}: ordinary case did not succeed")
        case_path = Path(str(manifest_row["case_artifact_path"])).resolve()
        expected_case_dir = authoritative_artifact_path.with_name(
            f"{authoritative_artifact_path.stem}_cases"
        )
        if case_path != expected_case_dir / f"{token}.json":
            raise ValueError(f"{token}: ordinary case path escapes authoritative tree")
        if not case_path.is_file():
            raise FileNotFoundError(case_path)
        # Parent-only byte/embedding check.  The 1.1 GB embedded manifest is
        # never serialized into a child-process task.
        case_payload = _read(case_path)
        if _sha_file(case_path) != _require_sha(
            manifest_row["case_artifact_sha256"], f"{token}.ordinary case SHA"
        ):
            raise ValueError(f"{token}: persisted ordinary case SHA mismatch")
        if embedded_case_payloads[token] != case_payload:
            raise ValueError(f"{token}: embedded case payload differs from persisted bytes")
        pool_path = Path(str(pool_rows[token]["path"])).resolve()
        control_path = Path(str(control_rows[token]["probe_path"])).resolve()
        baseline_path = complete_edge_baseline_manifest_path.resolve().parent / f"{token}.json"
        scan_path = Path(str(inference_rows[token]["source_scan_path"])).resolve()
        a0_path = a0_prediction_root.resolve() / "cases" / token / "prediction_owned_regions.json"
        tasks.append((
            token, str(case_path), str(pool_path), str(control_path),
            str(baseline_path), str(scan_path), str(a0_path),
            deepcopy(manifest_row), deepcopy(mapping_expectations[token]),
        ))
    expected_aggregate_tables = _expected_aggregate_tables(embedded_case_payloads)
    aggregate_tables = manifest.get("aggregate_tables")
    aggregate_table_hashes = manifest.get("aggregate_table_hashes")
    if (
        not isinstance(aggregate_tables, Mapping)
        or set(aggregate_tables) != set(AGGREGATE_TABLE_SOURCE_FIELDS)
        or not isinstance(aggregate_table_hashes, Mapping)
        or set(aggregate_table_hashes)
        != {f"{name}_sha256" for name in AGGREGATE_TABLE_SOURCE_FIELDS}
    ):
        raise ValueError("ordinary aggregate full-table schema mismatch")
    for name, expected_rows in expected_aggregate_tables.items():
        actual_rows = [dict(row) for row in aggregate_tables[name]]
        if actual_rows != expected_rows:
            raise ValueError(f"ordinary aggregate table differs from case artifacts: {name}")
        if _require_sha(
            aggregate_table_hashes[f"{name}_sha256"], f"aggregate {name} SHA"
        ) != _sha_json(actual_rows):
            raise ValueError(f"ordinary aggregate table SHA mismatch: {name}")
    with ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=1
    ) as executor:
        raw_case_summaries = list(executor.map(_validate_case_task, tasks))
    case_summaries = _ordered_case_summaries(raw_case_summaries)
    all_relation_rows = [
        row for token in EXPECTED_CASE_TOKENS
        for row in embedded_case_payloads[token][
            "frozen_target_baseline_relation_table"
        ]
    ]
    all_outcomes = [
        row for token in EXPECTED_CASE_TOKENS
        for row in embedded_case_payloads[token]["frozen_target_outcome_table"]
    ]
    all_eligible_pair_ids = {
        row["baseline_local_plane_pair_id"]
        for token in EXPECTED_CASE_TOKENS
        for row in embedded_case_payloads[token]["target_eligible_pair_table"]
    }
    all_pair_count = sum(row["all_pair_count"] for row in case_summaries)
    max_rss = max(row["peak_rss_bytes"] for row in case_summaries)
    added_rung_count = sum(row["added_rung_count"] for row in case_summaries)
    reached_count = sum(row["target_reached_count"] for row in case_summaries)
    advanced_count = sum(
        row["target_side_gate_advanced_count"] for row in case_summaries
    )
    novel_count = sum(
        row["target_novel_rung_attributable_count"] for row in case_summaries
    )
    case_criteria_rows = [
        dict(embedded_case_payloads[token]["ordinary_v2_criteria"])
        for token in EXPECTED_CASE_TOKENS
    ]
    if (
        len(all_relation_rows) != EXPECTED_TARGET_ROWS
        or len(all_outcomes) != EXPECTED_TARGET_ROWS
        or all_pair_count != EXPECTED_ALL_PAIR_ROWS
        or len(all_eligible_pair_ids) != EXPECTED_ELIGIBLE_PAIR_ROWS
        or _sha_json(sorted(all_eligible_pair_ids))
        != EXPECTED_ELIGIBLE_PAIR_IDS_SHA256
    ):
        raise ValueError("ordinary aggregate exact target/pair census mismatch")
    present_count = sum(
        row["target_baseline_relation_status"] == core.RELATION_COMPONENT_UNIQUE
        for row in all_outcomes
    )
    absent_count = len(all_outcomes) - present_count
    if present_count != EXPECTED_PRESENT_ROWS or absent_count != EXPECTED_ABSENT_ROWS:
        raise ValueError("ordinary aggregate 4,772/16 relation census mismatch")
    aggregate_fields = {
        "aggregate_added_cross_parent_rung_count": added_rung_count,
        "target_reached_by_added_rung_count": reached_count,
        "target_side_gate_advanced_count": advanced_count,
        "target_novel_rung_attributable_certificate_count": novel_count,
        "target_component_image_present_count": present_count,
        "target_component_image_absent_outside_tube_count": absent_count,
        "frozen_target_count": EXPECTED_TARGET_ROWS,
        "target_eligible_pair_count": EXPECTED_ELIGIBLE_PAIR_ROWS,
        "all_pair_membership_row_count": EXPECTED_ALL_PAIR_ROWS,
    }
    for field, expected in aggregate_fields.items():
        if field not in aggregate_counters or int(aggregate_counters[field]) != expected:
            raise ValueError(f"ordinary aggregate source-derived counter mismatch: {field}")
    if int(manifest.get("maximum_peak_rss_bytes") or -1) != max_rss:
        raise ValueError("ordinary aggregate maximum RSS mismatch")
    criteria = manifest.get("ordinary_v2_criteria")
    derived_aggregate_criteria = {
        key: all(row[key] is True for row in case_criteria_rows)
        for key in core.ORDINARY_V2_CRITERION_KEYS
    }
    _assert_criteria_derived(criteria, derived_aggregate_criteria)
    if (
        manifest.get("all_ordinary_v2_criteria_valid") is not True
        or manifest.get("packet_status") != PACKET_STATUS
        or manifest.get("next_professor_packet_authorized") is not True
    ):
        raise ValueError("ordinary aggregate packet decision mismatch")
    evidence = {
        "schema_version": SCHEMA_VERSION, "kind": KIND, "profile": PROFILE,
        "research_only": True, "inference_safe": True,
        "truth_inputs_consumed": False, "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "gold_open_authorized": False, "ordinary_batch_authorized": False,
        "next_professor_packet_authorized": True, "packet_status": PACKET_STATUS,
        "ordinary_authoritative_sha256": authoritative_sha,
        **{field: frozen_sha for field, (_path, frozen_sha) in source_paths.items()},
        **aggregate_fields,
        "maximum_peak_rss_bytes": max_rss,
        "ordinary_v2_criteria": deepcopy(criteria),
        "case_summaries_sha256": _sha_json(case_summaries),
        "case_summaries": case_summaries,
    }
    evidence["compact_evidence_sha256"] = _record_hash(
        evidence, "compact_evidence_sha256"
    )
    return evidence


# Full artifact validation is added against the finalized authoritative runner
# schema.  These helpers are intentionally usable by tamper tests without ever
# executing the ordinary batch.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-artifact", required=True, type=Path)
    parser.add_argument("--expected-authoritative-sha256", required=True)
    parser.add_argument("--mapping-preflight", required=True, type=Path)
    parser.add_argument("--mapping-compact-evidence", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def canonical_evidence_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def main() -> int:
    args = build_parser().parse_args()
    evidence = validate_and_compact(
        authoritative_artifact_path=args.authoritative_artifact,
        expected_authoritative_sha256=args.expected_authoritative_sha256,
        mapping_preflight_path=args.mapping_preflight,
        mapping_compact_evidence_path=args.mapping_compact_evidence,
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_evidence_bytes(evidence))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": _sha_file(args.output.resolve()),
        "ordinary_authoritative_sha256": evidence["ordinary_authoritative_sha256"],
        "packet_status": evidence["packet_status"],
        "gold_open_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
