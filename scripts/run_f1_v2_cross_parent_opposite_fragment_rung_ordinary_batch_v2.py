#!/usr/bin/env python3
"""Run the professor-authorized 37-case ordinary V2 rung batch.

The runner is deliberately fail-closed.  It accepts only the frozen Gold37
sources and corrected mapping preflight, executes each case in a fresh child
process, persists every complete case payload, and emits the untruncated tables
consumed by the independent validator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as core  # noqa: E402
from backend import f1_v2_complete_same_surface_edge_probe as complete_core  # noqa: E402
from scripts.run_f1_v2_cross_parent_rung_mapping_preflight import (  # noqa: E402
    MAPPING_PREFLIGHT_CRITERIA,
    _c001_locked_regression,
)
from scripts.validate_f1_v2_cross_parent_ordinary_batch import (  # noqa: E402
    AGGREGATE_TABLE_SOURCE_FIELDS,
    CASE_FIELDS,
    EXPECTED_ALL_PAIR_SHA256,
    EXPECTED_ALL_PAIR_ROWS,
    EXPECTED_BASELINE_MANIFEST_SHA256,
    EXPECTED_CASE_TOKENS,
    EXPECTED_CONTROL_MANIFEST_SHA256,
    EXPECTED_ELIGIBLE_PAIR_IDS_SHA256,
    EXPECTED_ELIGIBLE_PAIR_ROWS,
    EXPECTED_INFERENCE_MANIFEST_SHA256,
    EXPECTED_MAPPING_COMPACT_SHA256,
    EXPECTED_MAPPING_OUTCOME_SHA256,
    EXPECTED_MAPPING_PREFLIGHT_SHA256,
    EXPECTED_POOL_MANIFEST_SHA256,
    EXPECTED_RELATION_SHA256,
    EXPECTED_TARGET_PROJECTION_SHA256,
    EXPECTED_TARGET_ROWS,
    EXPECTED_THRESHOLD_SHA256,
    ORDINARY_ARTIFACT_KIND,
    PACKET_STATUS,
    PEAK_RSS_LIMIT_BYTES,
    PROFILE,
    TOP_FIELDS,
)


SCHEMA_VERSION = 2
KIND = ORDINARY_ARTIFACT_KIND
MAX_WORKERS = 8
INVALID_PACKET_STATUS = "INVALID_EXPERIMENT_STOP"


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used for case files."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def _rows_by_token(
    payload: Mapping[str, Any], *, label: str,
    expected_tokens: Sequence[str] = EXPECTED_CASE_TOKENS,
) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{label} cases must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} case rows must be objects")
        token = str(raw.get("case_token") or "")
        if not token or token in result:
            raise ValueError(f"{label} case tokens must be nonempty and unique")
        result[token] = dict(raw)
    if set(result) != set(expected_tokens):
        raise ValueError(f"{label} must contain the exact authorized case tokens")
    return result


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _group_rows(
    rows: Sequence[Mapping[str, Any]], *, strip_case_token: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        token: [] for token in EXPECTED_CASE_TOKENS
    }
    for raw in rows:
        row = deepcopy(dict(raw))
        token = str(row.get("case_token") or "")
        if token not in result:
            raise ValueError("mapping table contains a foreign case token")
        if strip_case_token:
            del row["case_token"]
        result[token].append(row)
    return result


def _validate_mapping_sources(
    *, mapping_preflight_path: Path, mapping_compact_evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if sha256_file(mapping_preflight_path) != EXPECTED_MAPPING_PREFLIGHT_SHA256:
        raise ValueError("corrected mapping preflight external SHA mismatch")
    if sha256_file(mapping_compact_evidence_path) != EXPECTED_MAPPING_COMPACT_SHA256:
        raise ValueError("mapping compact evidence external SHA mismatch")
    mapping = _read(mapping_preflight_path)
    compact = _read(mapping_compact_evidence_path)
    criteria = mapping.get("mapping_preflight_criteria")
    if (
        not isinstance(criteria, Mapping)
        or set(criteria) != set(MAPPING_PREFLIGHT_CRITERIA)
        or any(criteria[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA)
        or mapping.get("decision") != "GO_NEXT_PROFESSOR_PACKET"
        or mapping.get("next_professor_packet_authorized") is not True
    ):
        raise ValueError("corrected mapping pre-rung gate is not exact/all-true")
    if (
        mapping.get("frozen_target_projection_sha256")
        != EXPECTED_TARGET_PROJECTION_SHA256
        or mapping.get("relation_table_sha256") != EXPECTED_RELATION_SHA256
        or mapping.get("outcome_table_sha256") != EXPECTED_MAPPING_OUTCOME_SHA256
        or mapping.get("all_pair_membership_table_sha256")
        != EXPECTED_ALL_PAIR_SHA256
    ):
        raise ValueError("corrected mapping table binding mismatch")
    if (
        int(mapping.get("target_row_count") or -1) != EXPECTED_TARGET_ROWS
        or int(mapping.get("all_pair_membership_row_count") or -1)
        != EXPECTED_ALL_PAIR_ROWS
        or mapping.get("threshold_hash_table_sha256") != EXPECTED_THRESHOLD_SHA256
        or compact.get("mapping_preflight_sha256")
        != EXPECTED_MAPPING_PREFLIGHT_SHA256
    ):
        raise ValueError("corrected mapping census/compact binding mismatch")
    cases = _rows_by_token(mapping, label="corrected mapping preflight")
    return mapping, cases


def _mapping_expectations(mapping: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    projections = _group_rows(mapping["frozen_target_projection"])
    relations = _group_rows(mapping["frozen_target_baseline_relation_table"])
    outcomes = _group_rows(mapping["frozen_target_outcome_table"])
    memberships = _group_rows(
        mapping["all_pair_membership_table"], strip_case_token=True
    )
    cases = _rows_by_token(mapping, label="corrected mapping preflight")
    thresholds: dict[str, str] = {}
    for raw in mapping["threshold_hash_table"]:
        token = str(raw["case_token"])
        if token in thresholds or token not in EXPECTED_CASE_TOKENS:
            raise ValueError("mapping threshold table token mismatch")
        thresholds[token] = str(raw["thresholds_sha256"])
    if set(thresholds) != set(EXPECTED_CASE_TOKENS):
        raise ValueError("mapping threshold table is incomplete")
    return {
        token: {
            "frozen_target_projection": projections[token],
            "frozen_target_baseline_relation_table": relations[token],
            "frozen_target_outcome_table": outcomes[token],
            "all_pair_membership_table": memberships[token],
            "thresholds_sha256": thresholds[token],
            "mapping_case_summary": cases[token],
        }
        for token in EXPECTED_CASE_TOKENS
    }


def new_pre_rung_gate_criteria() -> dict[str, bool]:
    """Return the fail-closed per-case mapping gate."""
    return {key: False for key in MAPPING_PREFLIGHT_CRITERIA}


def _derive_pre_rung_gate(
    *, token: str, source_hashes_match: bool, projected: Mapping[str, Any],
    frozen_control: Mapping[str, Any], regenerated_control: Mapping[str, Any],
    frozen_baseline: Mapping[str, Any], regenerated_baseline: Mapping[str, Any],
    mapping_payload: Mapping[str, Any], expected_mapping: Mapping[str, Any],
) -> dict[str, bool]:
    """Derive the exact 20-key gate from this worker's regenerated evidence."""
    criteria = new_pre_rung_gate_criteria()
    relations = list(mapping_payload["frozen_target_baseline_relation_table"])
    outcomes = list(mapping_payload["frozen_target_outcome_table"])
    target_projection = list(mapping_payload["frozen_target_projection"])
    expected_projection = list(expected_mapping["frozen_target_projection"])
    expected_relations = list(expected_mapping[
        "frozen_target_baseline_relation_table"
    ])
    expected_outcomes = list(expected_mapping["frozen_target_outcome_table"])
    expected_memberships = list(expected_mapping["all_pair_membership_table"])
    summary = dict(expected_mapping["mapping_case_summary"])
    frozen_memberships = core.independently_regenerate_all_pair_membership_rows(
        projected, frozen_baseline
    )
    regenerated_memberships = core.independently_regenerate_all_pair_membership_rows(
        projected, regenerated_baseline
    )
    relation_projection = core.frozen_target_projection(relations)
    target_ids = [str(row["frozen_target_component_id"]) for row in relations]
    expected_target_ids = [
        str(row["frozen_target_component_id"]) for row in expected_projection
    ]
    target_mode_hashes = [
        str(row["frozen_target_mode_ids_sha256"]) for row in relations
    ]
    expected_mode_hashes = [
        str(row["frozen_target_mode_ids_sha256"]) for row in expected_projection
    ]
    expected_case_count = int(summary["frozen_target_count"])
    recognized_statuses = {
        core.RELATION_COMPONENT_UNIQUE, core.RELATION_COMPONENT_NONE_OUTSIDE,
    }
    criteria.update({
        "all_declared_source_hashes_match": bool(source_hashes_match),
        "all_frozen_control_probe_hashes_reproduce": (
            canonical_json_bytes(regenerated_control)
            == canonical_json_bytes(frozen_control)
        ),
        "all_complete_edge_baseline_hashes_reproduce": (
            canonical_json_bytes(regenerated_baseline)
            == canonical_json_bytes(frozen_baseline)
        ),
        # These aggregate-named criteria are evaluated as the exact contribution
        # of this case to the frozen 4,788-row mapping census.
        "frozen_target_count_is_4788": (
            len(target_projection) == expected_case_count
            and len(expected_projection) == expected_case_count
        ),
        "frozen_target_rows_are_byte_identical": (
            target_projection == relation_projection == expected_projection
        ),
        "frozen_target_id_sets_are_exactly_equal": (
            target_ids == expected_target_ids
            and len(target_ids) == len(set(target_ids))
        ),
        "frozen_target_mode_hash_sets_are_exactly_equal": (
            sorted(target_mode_hashes) == sorted(expected_mode_hashes)
            and len(target_mode_hashes) == len(expected_mode_hashes)
        ),
        "every_target_mode_has_one_baseline_fragment": all(
            all(fragment_id is not None for _mode_id, fragment_id in row[
                "baseline_mode_fragment_rows"
            ])
            for row in relations
        ),
        "every_target_maps_to_exactly_two_fragments": all(
            len(row["baseline_fragment_ids"]) == 2 for row in relations
        ),
        "every_target_has_exactly_one_pair_lineage": all(
            int(row["baseline_pair_mapping_count"]) == 1 for row in relations
        ),
        "every_in_tube_target_has_one_component_image": all(
            not row["target_parent_in_frozen_transverse_set"]
            or int(row["baseline_component_mapping_count"]) == 1
            for row in relations
        ),
        "every_out_of_tube_target_has_zero_component_images": all(
            row["target_parent_in_frozen_transverse_set"]
            or int(row["baseline_component_mapping_count"]) == 0
            for row in relations
        ),
        "no_unrecognised_relation_status_exists": all(
            row["target_baseline_relation_status"] in recognized_statuses
            for row in relations
        ),
        "relation_row_count_is_4788": (
            len(relations) == expected_case_count
            and len(expected_relations) == expected_case_count
        ),
        "relation_table_is_untruncated": (
            relations == expected_relations
            and len(target_ids) == len(set(target_ids)) == expected_case_count
        ),
        "outcome_row_count_is_4788": (
            outcomes == expected_outcomes
            and len(outcomes) == expected_case_count
            and [row["frozen_target_component_id"] for row in outcomes]
            == target_ids
        ),
        "c001_locked_regression_matches": (
            token != "C001" or _c001_locked_regression(relations)
        ),
        "all_threshold_hashes_are_unchanged": (
            canonical_json_bytes(mapping_payload["thresholds"])
            == canonical_json_bytes(frozen_baseline["thresholds"])
            == canonical_json_bytes(regenerated_baseline["thresholds"])
            and _sha_json(mapping_payload["thresholds"])
            == expected_mapping["thresholds_sha256"]
        ),
        "all_pair_line_tube_memberships_are_unchanged": (
            list(mapping_payload["all_pair_membership_table"])
            == frozen_memberships == regenerated_memberships
            == expected_memberships
        ),
        "gold_open_authorized_false": (
            mapping_payload.get("gold_open_authorized") is False
            and mapping_payload.get("truth_inputs_consumed") is False
            and mapping_payload.get("ordinary_batch_authorized") is False
            and mapping_payload.get("source_points_consumed") is False
            and mapping_payload.get("source_points_serialized") is False
        ),
    })
    if set(criteria) != set(MAPPING_PREFLIGHT_CRITERIA):
        raise AssertionError("per-case pre-rung gate schema drift")
    return criteria


def _run_case(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        token, scan_raw, pool_raw, a0_raw, control_raw, baseline_raw,
        scan_sha, pool_sha, a0_sha, control_sha, baseline_sha, expected_mapping,
    ) = task
    scan_path = Path(scan_raw)
    pool_path = Path(pool_raw)
    a0_path = Path(a0_raw)
    control_path = Path(control_raw)
    baseline_path = Path(baseline_raw)
    started = perf_counter()
    actual_hashes = {
        "scan": sha256_file(scan_path),
        "candidate pool": sha256_file(pool_path),
        "A0 prediction": sha256_file(a0_path),
        "frozen control": sha256_file(control_path),
        "complete-edge baseline": sha256_file(baseline_path),
    }
    expected_hashes = {
        "scan": scan_sha,
        "candidate pool": pool_sha,
        "A0 prediction": a0_sha,
        "frozen control": control_sha,
        "complete-edge baseline": baseline_sha,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"{token}: source byte hash mismatch")
    projected = core.project_mapping_preflight_input(_read(pool_path))
    frozen_control = _read(control_path)
    frozen_baseline = _read(baseline_path)
    if (
        _sha_json(frozen_control) != control_sha
        or _sha_json(frozen_baseline) != baseline_sha
    ):
        raise ValueError(f"{token}: frozen source is not canonical/hash-bound")
    regenerated_control = complete_core.regenerate_frozen_control_probe(
        deepcopy(projected), scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha, a0_prediction_sha256=a0_sha,
    )
    regenerated_baseline = complete_core.build_v2_complete_same_surface_edge_probe(
        deepcopy(projected), scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha, a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=control_sha,
    )
    mapping_payload = core._build_case_from_payloads(
        deepcopy(projected), case_token=token,
        scan_sha256=scan_sha, candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=control_sha,
        complete_edge_baseline_probe_sha256=baseline_sha,
        frozen_control=regenerated_control,
        complete_edge_baseline=regenerated_baseline,
    )
    core.validate_v2_cross_parent_mapping_case_payload(mapping_payload)
    pre_rung_gate = _derive_pre_rung_gate(
        token=token, source_hashes_match=actual_hashes == expected_hashes,
        projected=projected, frozen_control=frozen_control,
        regenerated_control=regenerated_control,
        frozen_baseline=frozen_baseline,
        regenerated_baseline=regenerated_baseline,
        mapping_payload=mapping_payload, expected_mapping=expected_mapping,
    )
    if any(pre_rung_gate[key] is not True for key in MAPPING_PREFLIGHT_CRITERIA):
        failed = [key for key, value in pre_rung_gate.items() if value is not True]
        raise ValueError(f"{token}: per-case pre-rung gate failed: {failed}")
    payload = core.build_v2_cross_parent_ordinary_case(
        projected,
        case_token=token,
        scan_sha256=scan_sha,
        candidate_pool_sha256=pool_sha,
        a0_prediction_sha256=a0_sha,
        frozen_control_probe_sha256=control_sha,
        complete_edge_baseline_probe_sha256=baseline_sha,
        expected_mapping_case_sha256=_sha_json(mapping_payload),
    )
    core.validate_v2_cross_parent_ordinary_case_payload(payload)
    for field in (
        "frozen_target_projection",
        "frozen_target_baseline_relation_table",
        "all_pair_membership_table",
    ):
        if payload[field] != expected_mapping[field]:
            raise ValueError(f"{token}: pre-rung mapping table changed: {field}")
    if _sha_json(payload["thresholds"]) != expected_mapping["thresholds_sha256"]:
        raise ValueError(f"{token}: pre-rung threshold hash changed")
    peak_rss = _peak_rss_bytes()
    if peak_rss >= PEAK_RSS_LIMIT_BYTES:
        raise MemoryError(
            f"{token}: peak RSS cap breached: {peak_rss} >= {PEAK_RSS_LIMIT_BYTES}"
        )
    encoded = canonical_json_bytes(payload)
    return {
        "case_token": token,
        "payload": payload,
        "pre_rung_gate_criteria": pre_rung_gate,
        "wall_seconds": round(perf_counter() - started, 6),
        "peak_rss_bytes": peak_rss,
        "input_json_bytes": pool_path.stat().st_size,
        "serialized_output_bytes": len(encoded),
    }


def _aggregate_tables(
    case_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {
        name: [] for name in AGGREGATE_TABLE_SOURCE_FIELDS
    }
    for token in EXPECTED_CASE_TOKENS:
        payload = case_payloads[token]
        for aggregate_name, source_name in AGGREGATE_TABLE_SOURCE_FIELDS.items():
            raw = payload[source_name]
            rows = [raw] if isinstance(raw, Mapping) else list(raw)
            for source_row in rows:
                row = deepcopy(dict(source_row))
                if "case_token" in row:
                    if str(row["case_token"]) != token:
                        raise ValueError("case token changed while aggregating")
                else:
                    row = {"case_token": token, **row}
                tables[aggregate_name].append(row)
    return tables


def _aggregate_counters(
    case_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    outcomes = [
        row
        for token in EXPECTED_CASE_TOKENS
        for row in case_payloads[token]["frozen_target_outcome_table"]
    ]
    eligible_ids = sorted({
        str(row["baseline_local_plane_pair_id"])
        for token in EXPECTED_CASE_TOKENS
        for row in case_payloads[token]["target_eligible_pair_table"]
    })
    if (
        len(eligible_ids) != EXPECTED_ELIGIBLE_PAIR_ROWS
        or _sha_json(eligible_ids) != EXPECTED_ELIGIBLE_PAIR_IDS_SHA256
    ):
        raise ValueError("ordinary target-eligible pair census/hash mismatch")
    present = sum(
        row["target_baseline_relation_status"]
        == core.RELATION_COMPONENT_UNIQUE
        for row in outcomes
    )
    return {
        "aggregate_added_cross_parent_rung_count": sum(
            len(case_payloads[token]["added_cross_parent_rung_table"])
            for token in EXPECTED_CASE_TOKENS
        ),
        "target_reached_by_added_rung_count": sum(
            row["target_reached_by_added_rung"] is True for row in outcomes
        ),
        "target_side_gate_advanced_count": sum(
            row["target_side_gate_advanced"] is True for row in outcomes
        ),
        "target_novel_rung_attributable_certificate_count": sum(
            row["target_novel_rung_attributable"] is True for row in outcomes
        ),
        "target_component_image_present_count": present,
        "target_component_image_absent_outside_tube_count": len(outcomes) - present,
        "frozen_target_count": len(outcomes),
        "target_eligible_pair_count": len(eligible_ids),
        "all_pair_membership_row_count": sum(
            len(case_payloads[token]["all_pair_membership_table"])
            for token in EXPECTED_CASE_TOKENS
        ),
    }


def _build_case_row(
    *, result: Mapping[str, Any], case_path: Path,
) -> dict[str, Any]:
    payload = result["payload"]
    pre_rung_criteria = result["pre_rung_gate_criteria"]
    if (
        not isinstance(pre_rung_criteria, Mapping)
        or set(pre_rung_criteria) != set(MAPPING_PREFLIGHT_CRITERIA)
        or any(
            pre_rung_criteria[key] is not True
            for key in MAPPING_PREFLIGHT_CRITERIA
        )
    ):
        raise ValueError("case row refuses a non-derived/non-passing pre-rung gate")
    encoded = canonical_json_bytes(payload)
    row = {
        "case_token": str(result["case_token"]),
        "status": "SUCCESS",
        "case_artifact_path": str(case_path.resolve()),
        "case_artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "scan_sha256": payload["scan_sha256"],
        "candidate_pool_sha256": payload["candidate_pool_sha256"],
        "a0_prediction_sha256": payload["a0_prediction_sha256"],
        "frozen_control_probe_sha256": payload["frozen_control_probe_sha256"],
        "complete_edge_baseline_probe_sha256": payload[
            "complete_edge_baseline_probe_sha256"
        ],
        "mapping_case_sha256": payload["mapping_case_sha256"],
        "pre_rung_gate_criteria": deepcopy(dict(pre_rung_criteria)),
        "pre_rung_gate_sha256": _sha_json(pre_rung_criteria),
        "output_table_hashes": deepcopy(payload["output_table_hashes"]),
        "case_counters": deepcopy(payload["case_counters"]),
        "ordinary_v2_criteria": deepcopy(payload["ordinary_v2_criteria"]),
        "deletion_evidence_sha256": payload[
            "global_deletion_counterfactual"
        ]["deletion_evidence_sha256"],
        "wall_seconds": float(result["wall_seconds"]),
        "peak_rss_bytes": int(result["peak_rss_bytes"]),
        "input_json_bytes": int(result["input_json_bytes"]),
        "serialized_output_bytes": len(encoded),
    }
    if set(row) != set(CASE_FIELDS):
        raise AssertionError("ordinary V2 case row schema drift")
    return row


def _persist_authoritative_artifact(
    *, output_path: Path, artifact: Mapping[str, Any],
    case_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist canonical sibling cases, then the authoritative JSON envelope."""
    output_path = output_path.resolve()
    case_dir = output_path.with_name(f"{output_path.stem}_cases")
    for token in EXPECTED_CASE_TOKENS:
        _atomic_write(
            case_dir / f"{token}.json",
            canonical_json_bytes(case_payloads[token]),
        )
    _atomic_write(output_path, canonical_json_bytes(artifact) + b"\n")


def run_batch(
    *, inference_manifest_path: Path, candidate_pool_manifest_path: Path,
    a0_prediction_root: Path, frozen_control_manifest_path: Path,
    complete_edge_baseline_manifest_path: Path, mapping_preflight_path: Path,
    mapping_compact_evidence_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{MAX_WORKERS}]")
    inference_manifest_path = inference_manifest_path.resolve()
    candidate_pool_manifest_path = candidate_pool_manifest_path.resolve()
    a0_prediction_root = a0_prediction_root.resolve()
    frozen_control_manifest_path = frozen_control_manifest_path.resolve()
    complete_edge_baseline_manifest_path = complete_edge_baseline_manifest_path.resolve()
    mapping_preflight_path = mapping_preflight_path.resolve()
    mapping_compact_evidence_path = mapping_compact_evidence_path.resolve()
    output_path = output_path.resolve()
    source_hashes = {
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "candidate_pool_manifest_sha256": sha256_file(candidate_pool_manifest_path),
        "frozen_control_manifest_sha256": sha256_file(frozen_control_manifest_path),
        "complete_edge_baseline_manifest_sha256": sha256_file(
            complete_edge_baseline_manifest_path
        ),
    }
    expected_source_hashes = {
        "inference_manifest_sha256": EXPECTED_INFERENCE_MANIFEST_SHA256,
        "candidate_pool_manifest_sha256": EXPECTED_POOL_MANIFEST_SHA256,
        "frozen_control_manifest_sha256": EXPECTED_CONTROL_MANIFEST_SHA256,
        "complete_edge_baseline_manifest_sha256": EXPECTED_BASELINE_MANIFEST_SHA256,
    }
    if source_hashes != expected_source_hashes:
        raise ValueError("ordinary V2 frozen source manifest SHA mismatch")
    mapping, mapping_cases = _validate_mapping_sources(
        mapping_preflight_path=mapping_preflight_path,
        mapping_compact_evidence_path=mapping_compact_evidence_path,
    )
    expectations = _mapping_expectations(mapping)
    inference_rows = _rows_by_token(_read(inference_manifest_path), label="inference")
    pool_rows = _rows_by_token(_read(candidate_pool_manifest_path), label="candidate pool")
    control_rows = _rows_by_token(_read(frozen_control_manifest_path), label="control")
    baseline_rows = _rows_by_token(
        _read(complete_edge_baseline_manifest_path), label="complete-edge baseline"
    )
    tasks = []
    for token in EXPECTED_CASE_TOKENS:
        scan = Path(str(inference_rows[token]["source_scan_path"])).resolve()
        pool = Path(str(pool_rows[token]["path"])).resolve()
        a0 = a0_prediction_root / "cases" / token / "prediction_owned_regions.json"
        control = Path(str(control_rows[token]["probe_path"])).resolve()
        baseline = complete_edge_baseline_manifest_path.parent / f"{token}.json"
        for label, path in (
            ("scan", scan), ("candidate pool", pool), ("A0 prediction", a0),
            ("control", control), ("complete-edge baseline", baseline),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{token}: missing {label}: {path}")
        bindings = {
            "scan_sha256": str(mapping_cases[token]["scan_sha256"]),
            "candidate_pool_sha256": str(mapping_cases[token]["candidate_pool_sha256"]),
            "a0_prediction_sha256": str(mapping_cases[token]["a0_prediction_sha256"]),
            "frozen_control_probe_sha256": str(
                mapping_cases[token]["frozen_control_probe_sha256"]
            ),
            "complete_edge_baseline_probe_sha256": str(
                mapping_cases[token]["complete_edge_baseline_probe_sha256"]
            ),
        }
        if (
            bindings["frozen_control_probe_sha256"]
            != str(control_rows[token]["probe_sha256"])
            or bindings["complete_edge_baseline_probe_sha256"]
            != str(baseline_rows[token]["replacement_probe_sha256"])
        ):
            raise ValueError(f"{token}: mapping/source per-case binding mismatch")
        tasks.append((
            token, str(scan), str(pool), str(a0), str(control), str(baseline),
            bindings["scan_sha256"], bindings["candidate_pool_sha256"],
            bindings["a0_prediction_sha256"],
            bindings["frozen_control_probe_sha256"],
            bindings["complete_edge_baseline_probe_sha256"], expectations[token],
        ))
    with ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=1
    ) as executor:
        results = list(executor.map(_run_case, tasks))
    results.sort(key=lambda item: str(item["case_token"]))
    if [item["case_token"] for item in results] != list(EXPECTED_CASE_TOKENS):
        raise RuntimeError("ordinary V2 batch requires exact C001-C037 success")
    case_payloads = {
        str(result["case_token"]): deepcopy(result["payload"])
        for result in results
    }
    case_dir = output_path.with_name(f"{output_path.stem}_cases")
    case_rows = []
    for result in results:
        token = str(result["case_token"])
        case_path = case_dir / f"{token}.json"
        case_rows.append(_build_case_row(result=result, case_path=case_path))
    aggregate_tables = _aggregate_tables(case_payloads)
    aggregate_hashes = {
        f"{name}_sha256": _sha_json(rows)
        for name, rows in aggregate_tables.items()
    }
    aggregate_counters = _aggregate_counters(case_payloads)
    aggregate_criteria = {
        key: all(
            case_payloads[token]["ordinary_v2_criteria"][key] is True
            for token in EXPECTED_CASE_TOKENS
        )
        for key in core.ORDINARY_V2_CRITERION_KEYS
    }
    all_valid = all(aggregate_criteria.values())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "profile": PROFILE,
        "artifact_mode": "ORDINARY_V2_BATCH",
        "research_only": True,
        "inference_safe": True,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "designated_replays_authorized": False,
        "full_synthetic_resource_run_authorized": False,
        "gold_open_authorized": False,
        "case_count": len(case_rows),
        "successful_case_count": len(case_rows),
        "workers": workers,
        "isolated_process_per_case": True,
        "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
        **source_hashes,
        "mapping_preflight_external_sha256": EXPECTED_MAPPING_PREFLIGHT_SHA256,
        "mapping_compact_evidence_external_sha256": EXPECTED_MAPPING_COMPACT_SHA256,
        "frozen_target_projection_sha256": EXPECTED_TARGET_PROJECTION_SHA256,
        "mapping_relation_table_sha256": EXPECTED_RELATION_SHA256,
        "mapping_outcome_table_sha256": EXPECTED_MAPPING_OUTCOME_SHA256,
        "all_pair_membership_table_sha256": EXPECTED_ALL_PAIR_SHA256,
        "threshold_hash_table_sha256": EXPECTED_THRESHOLD_SHA256,
        "target_eligible_pair_ids_sha256": EXPECTED_ELIGIBLE_PAIR_IDS_SHA256,
        "target_eligible_pair_count": EXPECTED_ELIGIBLE_PAIR_ROWS,
        "maximum_peak_rss_bytes": max(
            int(row["peak_rss_bytes"]) for row in case_rows
        ),
        "aggregate_counters": aggregate_counters,
        "aggregate_tables": aggregate_tables,
        "aggregate_table_hashes": aggregate_hashes,
        "ordinary_v2_criteria": aggregate_criteria,
        "all_ordinary_v2_criteria_valid": all_valid,
        "packet_status": PACKET_STATUS if all_valid else INVALID_PACKET_STATUS,
        "next_professor_packet_authorized": all_valid,
        "cases": case_rows,
        "case_payloads": [case_payloads[token] for token in EXPECTED_CASE_TOKENS],
    }
    if set(artifact) != set(TOP_FIELDS):
        raise AssertionError("ordinary V2 authoritative top-level schema drift")
    # Do not expose a partially authoritative batch: all payloads and aggregate
    # material are complete before any final path is replaced.
    _persist_authoritative_artifact(
        output_path=output_path, artifact=artifact, case_payloads=case_payloads
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--candidate-pool-manifest", required=True, type=Path)
    parser.add_argument("--a0-prediction-root", required=True, type=Path)
    parser.add_argument("--frozen-control-manifest", required=True, type=Path)
    parser.add_argument("--complete-edge-baseline-manifest", required=True, type=Path)
    parser.add_argument("--mapping-preflight", required=True, type=Path)
    parser.add_argument("--mapping-compact-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_batch(
        inference_manifest_path=args.inference_manifest,
        candidate_pool_manifest_path=args.candidate_pool_manifest,
        a0_prediction_root=args.a0_prediction_root,
        frozen_control_manifest_path=args.frozen_control_manifest,
        complete_edge_baseline_manifest_path=args.complete_edge_baseline_manifest,
        mapping_preflight_path=args.mapping_preflight,
        mapping_compact_evidence_path=args.mapping_compact_evidence,
        output_path=args.output,
        workers=args.workers,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "case_count": artifact["case_count"],
        "packet_status": artifact["packet_status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
