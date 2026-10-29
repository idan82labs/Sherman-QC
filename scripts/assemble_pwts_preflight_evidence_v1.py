#!/usr/bin/env python3
"""Independently assemble compact PWTS preflight and integrity evidence.

This assembler consumes complete validator-authored tables/evidence and derives
the sole terminal status.  It never executes geometry, opens gold, or trusts a
runner assertion that a criterion passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


PRELIGHT_ID = "PWTS_REPRESENTATION_PREFLIGHT_V1"
VALID_STATUS = "PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET"
INVALID_STATUS = "INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP"
EXPECTED_CONFIGURATION_SHA256 = "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
PROCESS_ROLES = ("BASE_A", "BASE_B", "ORDER_PERMUTED", "INDEPENDENT_VALIDATOR")

PER_PROCESS_CPU_SECONDS = 900
PER_PROCESS_WALL_SECONDS = 1200
PER_PROCESS_PEAK_RSS_BYTES = 12_884_901_888
PER_PROCESS_SCRATCH_BYTES = 4_294_967_296
PER_CASE_SERIALIZED_BYTES = 100_663_296
AGGREGATE_CPU_SECONDS = 133_200
AGGREGATE_WALL_SECONDS = 177_600
AGGREGATE_SERIALIZED_BYTES = 2_147_483_648

TAMPER_IDS = (
    "T01_RAW_BYTE_CHANGED",
    "T02_CASE_SOURCE_BINDING_EXCHANGE",
    "T03_SAMPLED_POINT_ID_SUBSTITUTION",
    "T04_UNAUTHORIZED_CONFIGURATION",
    "T05_SEED_ROW_REMOVAL",
    "T06_TRACE_NODE_ID_MUTATION",
    "T07_MEMBERSHIP_SIDE_EXCHANGE",
    "T08_FORGED_CERTIFICATE_VALIDITY",
    "T09_DELETION_EVIDENCE_FORGERY",
    "T10_ROW_ORDER_DUPLICATE_DELETE",
    "T11_SOURCE_POINT_SERIALIZATION",
    "T12_RESOURCE_ACCOUNTING_FORGERY",
    "T13_FORGED_EQUIVALENCE",
    "T14_SEMANTIC_IMPORT_BOUNDARY",
)
SAFETY_FALSE_FIELDS = (
    "gold_truth", "expected_counts", "manual_labels", "cad_reference",
    "prior_f1_candidates", "lpf_semantics", "mesh_topology", "solver",
    "tuned_tracers", "heatmap_acceptance", "learning", "ownership_births",
    "replays", "synthetic_rescue", "runtime_main", "production",
    "benchmark_reporting",
)
VALIDATOR_CRITERIA = (
    "ALL_SOURCE_BINDINGS_VALID",
    "COMPLETE_TABLES_PRESENT_AND_CANONICAL",
    "TABLE_DENOMINATORS_EXACT",
    "STRUCTURAL_CAPS_RESPECTED",
    "IDENTIFIERS_AND_SOURCE_BINDINGS_RECOMPUTED",
    "STATIC_IMPORT_ALLOWLIST_VALID",
    "NO_SOURCE_POINT_SERIALIZATION",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected object: {path}")
    if payload.get("preflight_id") != PRELIGHT_ID:
        raise ValueError(f"foreign preflight identifier: {path}")
    return dict(payload)


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _source_manifest_valid(payload: Mapping[str, Any]) -> bool:
    content = dict(payload)
    asserted = content.pop("input_manifest_sha256", None)
    rows = content.get("cases")
    return bool(
        asserted == hashlib.sha256(canonical_json_bytes(content)).hexdigest()
        and content.get("kind") == "pwts_source_lock_v1"
        and content.get("case_tokens") == list(EXPECTED_CASE_TOKENS)
        and isinstance(rows, list)
        and [row.get("case_token") for row in rows] == list(EXPECTED_CASE_TOKENS)
        and content.get("semantic_work_performed") is False
        and content.get("source_points_serialized") is False
    )


def _resources_valid(orchestration: Mapping[str, Any]) -> bool:
    rows = orchestration.get("resource_rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CASE_TOKENS) * len(PROCESS_ROLES):
        return False
    expected = [(token, role) for token in EXPECTED_CASE_TOKENS for role in PROCESS_ROLES]
    observed = [(row.get("case_token"), row.get("process_role")) for row in rows]
    if observed != expected:
        return False
    for row in rows:
        if not (
            row.get("success") is True
            and row.get("monitor_available") is True
            and row.get("full_lifetime_accounting") is True
            and row.get("reported_passing") is True
            and row.get("rss_cap_exceeded") is False
            and row.get("scratch_cap_exceeded") is False
            and row.get("wall_cap_exceeded") is False
            and 0 <= float(row.get("cpu_seconds", -1)) <= PER_PROCESS_CPU_SECONDS
            and 0 <= float(row.get("wall_seconds", -1)) <= PER_PROCESS_WALL_SECONDS
            and 0 <= int(row.get("process_tree_peak_rss_bytes", -1)) <= PER_PROCESS_PEAK_RSS_BYTES
            and 0 <= int(row.get("scratch_bytes", -1)) <= PER_PROCESS_SCRATCH_BYTES
            and 0 < int(row.get("serialized_bytes", 0)) <= PER_CASE_SERIALIZED_BYTES
            and isinstance(row.get("payload_sha256"), str)
        ):
            return False
    return bool(
        sum(float(row["cpu_seconds"]) for row in rows) <= AGGREGATE_CPU_SECONDS
        and sum(float(row["wall_seconds"]) for row in rows) <= AGGREGATE_WALL_SECONDS
        and sum(int(row["serialized_bytes"]) for row in rows) <= AGGREGATE_SERIALIZED_BYTES
    )


def _validator_criteria_valid(compact: Mapping[str, Any]) -> bool:
    rows = compact.get("criteria")
    if not isinstance(rows, list):
        return False
    by_id = {str(row.get("criterion_id")): row for row in rows if isinstance(row, Mapping)}
    return bool(
        set(by_id) == set(VALIDATOR_CRITERIA)
        and all(
            by_id[key].get("result") is True
            and isinstance(by_id[key].get("source_table_digest"), str)
            and len(by_id[key]["source_table_digest"]) == 64
            and int(by_id[key].get("denominator", -1)) >= 0
            for key in VALIDATOR_CRITERIA
        )
    )


def _determinism_valid(payload: Mapping[str, Any]) -> bool:
    rows = payload.get("cases")
    if not isinstance(rows, list) or [row.get("case_token") for row in rows] != list(EXPECTED_CASE_TOKENS):
        return False
    return all(
        row.get("base_a_base_b_equivalent") is True
        and row.get("order_permutation_equivalent") is True
        and row.get("independent_validator_equivalent") is True
        and row.get("first_mismatch") is None
        and isinstance(row.get("comparison_table_digests"), Mapping)
        for row in rows
    )


def _deletions_valid(payload: Mapping[str, Any], valid_ids: Sequence[str]) -> bool:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not valid_ids:
        return False
    expected = [(certificate_id, arm) for certificate_id in sorted(valid_ids) for arm in ("DELETE_LEFT", "DELETE_RIGHT")]
    observed = [(row.get("certificate_id"), row.get("deletion_arm")) for row in rows]
    if observed != expected:
        return False
    for row in rows:
        arm = row["deletion_arm"]
        rejection = "LEFT_SUPPORT_INSUFFICIENT" if arm == "DELETE_LEFT" else "RIGHT_SUPPORT_INSUFFICIENT"
        deleted_side_count = row.get("rebuilt_deleted_side_support_count")
        if deleted_side_count is None:
            deleted_side_count = row.get(
                "rebuilt_left_support_count" if arm == "DELETE_LEFT"
                else "rebuilt_right_support_count"
            )
        if not (
            row.get("rebuilt_valid") is False
            and row.get("first_rejection_code") == rejection
            and int(deleted_side_count if deleted_side_count is not None else -1) == 0
            and int(row.get("deleted_id_count", 0)) > 0
            and isinstance(row.get("deleted_id_digest"), str)
            and len(row["deleted_id_digest"]) == 64
            and (
                row.get("trace_identity_unchanged") is True
                or (
                    isinstance(row.get("trace_id"), str)
                    and row.get("raw_id_deletion_applied") is True
                )
            )
        ):
            return False
    return True


def _tamper_valid(payload: Mapping[str, Any]) -> bool:
    rows = payload.get("tests")
    return bool(
        isinstance(rows, list)
        and [row.get("regression_id") for row in rows] == list(TAMPER_IDS)
        and all(
            row.get("rejected") is True
            and isinstance(row.get("expected_rejection"), str)
            and isinstance(row.get("observed_errors"), list)
            and any(
                str(row["expected_rejection"]).lower() in str(error).lower()
                for error in row["observed_errors"]
            )
            for row in rows
        )
    )


def _safety_valid(payload: Mapping[str, Any]) -> bool:
    return bool(
        set(SAFETY_FALSE_FIELDS).issubset(payload)
        and all(payload[field] is False for field in SAFETY_FALSE_FIELDS)
        and payload.get("source_points_serialized") is False
    )


def assemble_evidence(
    *, input_manifest_path: Path, execution_lock_path: Path,
    orchestration_result_path: Path, authoritative_artifact_path: Path,
    validator_compact_path: Path, determinism_path: Path, deletion_path: Path,
    tamper_results_path: Path, safety_attestation_path: Path, output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "input_manifest": input_manifest_path.resolve(),
        "execution_lock": execution_lock_path.resolve(),
        "orchestration_result": orchestration_result_path.resolve(),
        "authoritative_artifact": authoritative_artifact_path.resolve(),
        "validator_compact": validator_compact_path.resolve(),
        "determinism": determinism_path.resolve(),
        "bilateral_deletion": deletion_path.resolve(),
        "tamper_results": tamper_results_path.resolve(),
        "safety_attestation": safety_attestation_path.resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [label for label, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"missing PWTS evidence: {missing}")
    source = _read(paths["input_manifest"])
    lock = _read(paths["execution_lock"])
    orchestration = _read(paths["orchestration_result"])
    compact = _read(paths["validator_compact"])
    determinism = _read(paths["determinism"])
    deletion = _read(paths["bilateral_deletion"])
    tamper = _read(paths["tamper_results"])
    safety = _read(paths["safety_attestation"])
    authoritative_sha = sha256_file(paths["authoritative_artifact"])
    valid_ids = compact.get("valid_certificate_ids")
    if not isinstance(valid_ids, list) or valid_ids != sorted(set(valid_ids)):
        valid_ids = []
    criteria = {
        "SOURCE_LOCK_VALID": _source_manifest_valid(source),
        "CONFIGURATION_AND_CODE_LOCK_VALID": bool(
            lock.get("expected_configuration_sha256") == EXPECTED_CONFIGURATION_SHA256
            and lock.get("observed_configuration_sha256") == EXPECTED_CONFIGURATION_SHA256
            and lock.get("configuration_hash_match") is True
            and isinstance(lock.get("source_sha256"), Mapping)
        ),
        "AUTHORITATIVE_ARTIFACT_BOUND": bool(
            compact.get("authoritative_artifact_sha256") == authoritative_sha
            and paths["authoritative_artifact"].stat().st_size <= AGGREGATE_SERIALIZED_BYTES
        ),
        "VALIDATOR_CRITERIA_VALID": _validator_criteria_valid(compact),
        "AT_LEAST_ONE_VALID_CERTIFICATE": bool(valid_ids),
        "DETERMINISM_AND_PERMUTATION_VALID": _determinism_valid(determinism),
        "BILATERAL_DELETION_VALID": _deletions_valid(deletion, valid_ids),
        "RESOURCE_EVIDENCE_VALID": _resources_valid(orchestration),
        "ALL_TAMPER_REGRESSIONS_REJECTED": _tamper_valid(tamper),
        "SAFETY_ATTESTATION_CLOSED": _safety_valid(safety),
    }
    all_valid = all(criteria.values())
    compact_output = {
        "schema_version": 1,
        "preflight_id": PRELIGHT_ID,
        "kind": "pwts_independently_assembled_compact_evidence_v1",
        "terminal_status": VALID_STATUS if all_valid else INVALID_STATUS,
        "next_professor_packet_authorized": all_valid,
        "gold_open_authorized": False,
        "runtime_main_production_authorized": False,
        "expected_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "case_count": len(EXPECTED_CASE_TOKENS),
        "valid_certificate_count": len(valid_ids),
        "destructive_negative_arm_count": len(deletion.get("rows") or []),
        "criteria": [
            {"criterion_id": key, "result": value} for key, value in sorted(criteria.items())
        ],
        "authoritative_artifact_sha256": authoritative_sha,
        "authoritative_artifact_bytes": paths["authoritative_artifact"].stat().st_size,
        "validator_table_digests": compact.get("table_digests"),
        "source_points_serialized": False,
        "accuracy_or_gold_statement_present": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = output_dir / "compact_evidence.json"
    _atomic_write(compact_path, compact_output)
    integrity_files = {**paths, "compact_evidence": compact_path}
    integrity = {
        "schema_version": 1,
        "preflight_id": PRELIGHT_ID,
        "kind": "pwts_evidence_integrity_index_v1",
        "terminal_status": compact_output["terminal_status"],
        "files": [
            {
                "label": label, "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(integrity_files.items())
        ],
    }
    integrity["integrity_index_content_sha256"] = hashlib.sha256(
        canonical_json_bytes(integrity)
    ).hexdigest()
    _atomic_write(output_dir / "integrity_index.json", integrity)
    return compact_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--orchestration-result", required=True, type=Path)
    parser.add_argument("--authoritative-artifact", required=True, type=Path)
    parser.add_argument("--validator-compact", required=True, type=Path)
    parser.add_argument("--determinism", required=True, type=Path)
    parser.add_argument("--bilateral-deletion", required=True, type=Path)
    parser.add_argument("--tamper-results", required=True, type=Path)
    parser.add_argument("--safety-attestation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = assemble_evidence(
        input_manifest_path=args.input_manifest,
        execution_lock_path=args.execution_lock,
        orchestration_result_path=args.orchestration_result,
        authoritative_artifact_path=args.authoritative_artifact,
        validator_compact_path=args.validator_compact,
        determinism_path=args.determinism,
        deletion_path=args.bilateral_deletion,
        tamper_results_path=args.tamper_results,
        safety_attestation_path=args.safety_attestation,
        output_dir=args.output_dir,
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0 if result["terminal_status"] == VALID_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
