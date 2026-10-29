#!/usr/bin/env python3
"""Validate and aggregate the complete PWTS preflight process matrix.

This is the post-run trust boundary.  It reads only locked manifests, compact
worker payloads, process-accounting evidence, and locked source text.  It does
not import or execute the PWTS semantic implementation and it has no gold,
CAD, legacy-F1, runtime, benchmark, or accuracy lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from research.pwts_v1.independent_validator import (
    INVALID_STATUS,
    PREFLIGHT_ID,
    REQUIRED_FIELDS,
    RULING_CONFIGURATION_SHA256,
    TABLES,
    VALID_STATUS,
    ValidationPolicy,
    canonical_json_bytes,
    derive_expected_criteria,
    forbidden_imports,
    imported_modules,
    row_key,
    sha256_json,
    sorted_id_digest,
    table_digest,
    validate_evidence,
)
from research.pwts_v1.tamper_suite import (
    require_all_tampers_rejected,
    run_tamper_suite,
)


EXPECTED_CASE_TOKENS = tuple(f"C{number:03d}" for number in range(1, 38))
PROCESS_ROLES = ("BASE_A", "BASE_B", "ORDER_PERMUTED", "INDEPENDENT_VALIDATOR")
SEMANTIC_TABLES = TABLES[:10]
COMPARISONS = (
    ("BASE_A_VS_BASE_B", "BASE_B"),
    ("BASE_A_VS_ORDER_PERMUTED", "ORDER_PERMUTED"),
    ("BASE_A_VS_INDEPENDENT_VALIDATOR", "INDEPENDENT_VALIDATOR"),
)
LOCKED_SOURCE_KEYS = (
    "runner_source",
    "semantic_core_source",
    "independent_validator_source",
    "tamper_test_source",
    "schema",
    "configuration",
    "input_manifest",
)
IMPORT_REPORT_KEYS = {
    "runner": "runner_source",
    "semantic_core": "semantic_core_source",
    "independent_validator": "independent_validator_source",
}
SAFETY_FALSE_FIELDS = (
    "gold_truth",
    "expected_counts",
    "manual_labels",
    "cad_reference",
    "prior_f1_candidates",
    "legacy_f1",
    "lpf_semantics",
    "mesh_topology",
    "solver",
    "poisson_solver",
    "tuned_tracers",
    "heatmap_acceptance",
    "learning",
    "ownership_births",
    "prediction_births",
    "replays",
    "synthetic_rescue",
    "runtime_main",
    "production",
    "benchmark_reporting",
    "accuracy_reporting",
)


class PreflightOutputError(ValueError):
    """A fail-closed violation in the completed process matrix."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_ascii_bytes(value: Any) -> bytes:
    """Canonical form used by the nonsemantic source-lock producer."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightOutputError(f"{label}: unreadable JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PreflightOutputError(f"{label}: JSON root must be an object")
    return dict(value)


def _atomic_write(path: Path, value: Any) -> None:
    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightOutputError(message)


def _validate_source_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _read_object(path, "source manifest")
    asserted = payload.get("input_manifest_sha256")
    content = dict(payload)
    content.pop("input_manifest_sha256", None)
    observed = hashlib.sha256(_canonical_ascii_bytes(content)).hexdigest()
    rows = payload.get("cases")
    _require(payload.get("preflight_id") == PREFLIGHT_ID, "source manifest: wrong preflight_id")
    _require(payload.get("kind") == "pwts_source_lock_v1", "source manifest: wrong kind")
    _require(asserted == observed, "source manifest: internal SHA mismatch")
    _require(payload.get("case_tokens") == list(EXPECTED_CASE_TOKENS), "source manifest: case envelope mismatch")
    _require(payload.get("case_count") == len(EXPECTED_CASE_TOKENS), "source manifest: case count mismatch")
    _require(payload.get("semantic_work_performed") is False, "source manifest: semantic work asserted")
    _require(payload.get("sampling_performed") is False, "source manifest: sampling asserted")
    _require(payload.get("source_points_serialized") is False, "source manifest: source points serialized")
    _require(isinstance(rows, list), "source manifest: cases must be an array")
    _require(
        [row.get("case_token") for row in rows if isinstance(row, Mapping)]
        == list(EXPECTED_CASE_TOKENS),
        "source manifest: cases are missing, duplicated, or noncanonical",
    )
    by_token: dict[str, dict[str, Any]] = {}
    required = {
        "case_token", "raw_path", "raw_byte_count", "raw_ply_sha256",
        "ply_header_sha256", "parsed_vertex_count", "finite_coordinate_count",
        "exact_duplicate_count", "ignored_property_name_digest",
    }
    for raw_row in rows:
        _require(isinstance(raw_row, Mapping), "source manifest: case row is not an object")
        row = dict(raw_row)
        token = str(row.get("case_token"))
        _require(required.issubset(row), f"{token}: incomplete source binding")
        coordinate_digest = row.get("canonical_coordinate_multiset_digest")
        coordinate_sha = row.get("canonical_coordinate_multiset_sha256")
        _require(
            isinstance(coordinate_digest or coordinate_sha, str)
            and (coordinate_digest is None or coordinate_sha is None or coordinate_digest == coordinate_sha),
            f"{token}: coordinate multiset digest mismatch",
        )
        raw_path = Path(str(row["raw_path"])).resolve()
        _require(raw_path.is_file(), f"{token}: locked raw source is missing")
        _require(raw_path.stat().st_size == int(row["raw_byte_count"]), f"{token}: raw byte count mismatch")
        _require(sha256_file(raw_path) == row["raw_ply_sha256"], f"{token}: raw source hash mismatch")
        by_token[token] = row
    return payload, by_token


def _validate_execution_lock(
    path: Path, *, input_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], str]:
    payload = _read_object(path, "execution lock")
    _require(payload.get("preflight_id") == PREFLIGHT_ID, "execution lock: wrong preflight_id")
    _require(payload.get("kind") == "pwts_execution_lock_v1", "execution lock: wrong kind")
    _require(
        payload.get("expected_configuration_sha256") == RULING_CONFIGURATION_SHA256
        and payload.get("observed_configuration_sha256") == RULING_CONFIGURATION_SHA256
        and payload.get("configuration_hash_match") is True,
        "execution lock: unauthorized configuration",
    )
    source_paths = payload.get("source_path")
    source_hashes = payload.get("source_sha256")
    _require(isinstance(source_paths, Mapping) and isinstance(source_hashes, Mapping), "execution lock: source maps missing")
    _require(set(source_paths) == set(LOCKED_SOURCE_KEYS), "execution lock: source path set mismatch")
    _require(set(source_hashes) == set(LOCKED_SOURCE_KEYS), "execution lock: source hash set mismatch")
    resolved: dict[str, Path] = {}
    for key in LOCKED_SOURCE_KEYS:
        source_path = Path(str(source_paths[key])).resolve()
        _require(source_path.is_file(), f"execution lock: missing locked source {key}")
        _require(sha256_file(source_path) == source_hashes[key], f"execution lock: changed source {key}")
        resolved[key] = source_path
    _require(
        resolved["input_manifest"] == input_manifest_path.resolve(),
        "execution lock: source manifest path mismatch",
    )
    content = dict(payload)
    asserted_lock_sha = content.pop("execution_lock_sha256", None)
    _require(
        asserted_lock_sha == hashlib.sha256(_canonical_ascii_bytes(content)).hexdigest(),
        "execution lock: internal SHA mismatch",
    )
    configuration = _read_object(resolved["configuration"], "locked configuration")
    _require(
        sha256_json(configuration) == RULING_CONFIGURATION_SHA256,
        "execution lock: configuration content digest mismatch",
    )
    return payload, resolved, str(source_hashes["schema"])


def _validate_static_imports(source_paths: Mapping[str, Path]) -> dict[str, list[str]]:
    reports: dict[str, list[str]] = {}
    for report_name, lock_name in IMPORT_REPORT_KEYS.items():
        source = source_paths[lock_name].read_text(encoding="utf-8")
        bad = forbidden_imports(source)
        _require(not bad, f"{report_name}: forbidden imports: {list(bad)}")
        modules = list(imported_modules(source))
        prohibited_names = [
            module for module in modules
            if any(marker in module.lower() for marker in ("gold", "cad", "legacy", "accuracy"))
        ]
        _require(not prohibited_names, f"{report_name}: prohibited-lane imports: {prohibited_names}")
        reports[report_name] = modules
    _require(not forbidden_imports(Path(__file__).read_text(encoding="utf-8")), "aggregator imports a semantic implementation")
    return reports


def _validate_orchestration(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    payload = _read_object(path, "orchestration result")
    rows = payload.get("resource_rows")
    expected_matrix = [(token, role) for token in EXPECTED_CASE_TOKENS for role in PROCESS_ROLES]
    _require(payload.get("preflight_id") == PREFLIGHT_ID, "orchestration: wrong preflight_id")
    _require(payload.get("kind") == "pwts_orchestration_result_v1", "orchestration: wrong kind")
    _require(payload.get("schema_version") == 1, "orchestration: wrong schema version")
    _require(payload.get("terminal_status") == INVALID_STATUS, "orchestration illegally granted valid status")
    _require(payload.get("failure_code") == "AWAITING_INDEPENDENT_VALIDATION_AND_TAMPER_SUITE", "orchestration did not complete cleanly")
    _require(payload.get("configuration_hash_match") is True, "orchestration: configuration hash mismatch")
    _require(
        payload.get("expected_configuration_sha256") == RULING_CONFIGURATION_SHA256
        and payload.get("observed_configuration_sha256") == RULING_CONFIGURATION_SHA256,
        "orchestration: unauthorized configuration",
    )
    _require(payload.get("semantic_process_started") is True, "orchestration: semantic matrix was not started")
    _require(payload.get("completed_process_count") == len(expected_matrix), "orchestration: process count incomplete")
    _require(payload.get("expected_process_count") == len(expected_matrix), "orchestration: expected process count mismatch")
    _require(payload.get("workers") == 1 and payload.get("cpu_only") is True, "orchestration: execution mode mismatch")
    _require(payload.get("fresh_process_per_case_variant") is True, "orchestration: processes were not isolated")
    _require(payload.get("process_role_order") == list(PROCESS_ROLES), "orchestration: role order mismatch")
    _require(payload.get("source_points_serialized") is False, "orchestration: source points serialized")
    _require(payload.get("gold_or_truth_consumed") is False, "orchestration: gold/truth consumed")
    _require(payload.get("runtime_or_production_authorized") is False, "orchestration: runtime/production authorized")
    _require(payload.get("next_professor_packet_authorized") is False, "orchestration self-authorized next packet")
    _require(isinstance(rows, list), "orchestration: resource rows missing")
    observed_matrix = [
        (row.get("case_token"), row.get("process_role"))
        for row in rows if isinstance(row, Mapping)
    ]
    _require(observed_matrix == expected_matrix, "orchestration: resource matrix missing, duplicated, or noncanonical")
    by_key = {(str(row["case_token"]), str(row["process_role"])): dict(row) for row in rows}
    for key, row in by_key.items():
        _require(row.get("success") is True, f"{key}: process unsuccessful")
        _require(row.get("return_code") == 0, f"{key}: nonzero return code")
        _require(row.get("failure_code") is None, f"{key}: process failure asserted")
        _require(row.get("monitor_available") is True, f"{key}: monitor unavailable")
        _require(row.get("full_lifetime_accounting") is True, f"{key}: shortened resource accounting")
        _require(row.get("reported_passing") is True, f"{key}: resource row did not pass")
        _require(row.get("rss_cap_exceeded") is False, f"{key}: RSS cap exceeded")
        _require(row.get("scratch_cap_exceeded") is False, f"{key}: scratch cap exceeded")
        _require(row.get("wall_cap_exceeded") is False, f"{key}: wall cap exceeded")
    _require(
        float(payload.get("aggregate_cpu_seconds", -1))
        == sum(float(row["cpu_seconds"]) for row in rows),
        "orchestration: aggregate CPU accounting mismatch",
    )
    _require(
        float(payload.get("aggregate_wall_seconds", -1))
        == sum(float(row["wall_seconds"]) for row in rows),
        "orchestration: aggregate wall accounting mismatch",
    )
    _require(
        int(payload.get("aggregate_serialized_bytes", -1))
        == sum(int(row["serialized_bytes"]) for row in rows),
        "orchestration: aggregate serialization accounting mismatch",
    )
    return payload, by_key


def _payload_path(process_root: Path, token: str, role: str) -> Path:
    return process_root / token / role / "output" / "case_payload.json"


def _validate_payloads(
    *, process_root: Path, resources: Mapping[tuple[str, str], Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]], schema_sha256: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    expected_paths = {
        _payload_path(process_root, token, role).resolve()
        for token in EXPECTED_CASE_TOKENS for role in PROCESS_ROLES
    }
    observed_paths = {path.resolve() for path in process_root.rglob("case_payload.json")}
    _require(observed_paths == expected_paths, "worker payload set is not the exact 148-file matrix")
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    payload_hashes: dict[tuple[str, str], str] = {}
    for token in EXPECTED_CASE_TOKENS:
        for role in PROCESS_ROLES:
            key = (token, role)
            path = _payload_path(process_root, token, role)
            resource = resources[key]
            observed_sha = sha256_file(path)
            _require(path.stat().st_size == int(resource.get("serialized_bytes", -1)), f"{token}/{role}: payload byte count mismatch")
            _require(observed_sha == resource.get("payload_sha256"), f"{token}/{role}: payload hash mismatch")
            stdout = process_root / token / role / "stdout.log"
            stderr = process_root / token / role / "stderr.log"
            _require(stdout.is_file() and stderr.is_file(), f"{token}/{role}: process logs missing")
            _require(sha256_file(stdout) == resource.get("stdout_sha256"), f"{token}/{role}: stdout hash mismatch")
            _require(sha256_file(stderr) == resource.get("stderr_sha256"), f"{token}/{role}: stderr hash mismatch")
            payload = _read_object(path, f"{token}/{role} payload")
            _require(path.read_bytes() == canonical_json_bytes(payload), f"{token}/{role}: payload is not canonical JSON")
            _require(
                set(payload) == {"preflight_id", "case_token", "process_role", "source_points_serialized", "tables"},
                f"{token}/{role}: payload envelope schema mismatch",
            )
            _require(payload["preflight_id"] == PREFLIGHT_ID, f"{token}/{role}: wrong preflight_id")
            _require(payload["case_token"] == token and payload["process_role"] == role, f"{token}/{role}: payload binding mismatch")
            _require(payload["source_points_serialized"] is False, f"{token}/{role}: source points serialized")
            tables = payload.get("tables")
            _require(isinstance(tables, Mapping) and set(tables) == set(SEMANTIC_TABLES), f"{token}/{role}: semantic table set mismatch")
            for table in SEMANTIC_TABLES:
                rows = tables[table]
                _require(isinstance(rows, list), f"{token}/{role}/{table}: table is not an array")
                keys = []
                for index, row in enumerate(rows):
                    _require(isinstance(row, Mapping), f"{token}/{role}/{table}[{index}]: row is not an object")
                    _require(REQUIRED_FIELDS[table].issubset(row), f"{token}/{role}/{table}[{index}]: schema fields missing")
                    _require(row.get("case_token") == token, f"{token}/{role}/{table}[{index}]: foreign case row")
                    keys.append(row_key(table, row))
                _require(keys == sorted(keys) and len(keys) == len(set(keys)), f"{token}/{role}/{table}: noncanonical or duplicate rows")
            manifest_row = tables["case_manifest"]
            _require(len(manifest_row) == 1, f"{token}/{role}: case_manifest must have exactly one row")
            case = manifest_row[0]
            source = source_rows[token]
            coordinate_digest = source.get("canonical_coordinate_multiset_digest") or source.get("canonical_coordinate_multiset_sha256")
            bindings = {
                "raw_path": str(source["raw_path"]),
                "raw_byte_count": int(source["raw_byte_count"]),
                "raw_ply_sha256": source["raw_ply_sha256"],
                "observed_raw_ply_sha256": source["raw_ply_sha256"],
                "ply_header_sha256": source["ply_header_sha256"],
                "parsed_vertex_count": int(source["parsed_vertex_count"]),
                "finite_coordinate_count": int(source["finite_coordinate_count"]),
                "exact_duplicate_count": int(source["exact_duplicate_count"]),
                "ignored_property_name_digest": source["ignored_property_name_digest"],
                "canonical_coordinate_multiset_digest": coordinate_digest,
                "config_sha256": RULING_CONFIGURATION_SHA256,
                "schema_sha256": schema_sha256,
                "source_points_serialized": False,
            }
            for field, expected in bindings.items():
                _require(case.get(field) == expected, f"{token}/{role}: source/config/schema binding mismatch at {field}")
            payloads[key] = payload
            payload_hashes[key] = observed_sha
    return payloads, payload_hashes


def _compare_semantic_tables(
    payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    equivalence: list[dict[str, Any]] = []
    determinism_cases: list[dict[str, Any]] = []
    for token in EXPECTED_CASE_TOKENS:
        base_tables = payloads[(token, "BASE_A")]["tables"]
        role_table_digests: dict[str, dict[str, str]] = {
            table: {} for table in SEMANTIC_TABLES
        }
        role_table_counts: dict[str, dict[str, int]] = {
            table: {} for table in SEMANTIC_TABLES
        }
        for role in PROCESS_ROLES:
            tables = payloads[(token, role)]["tables"]
            for table in SEMANTIC_TABLES:
                rows = tables[table]
                role_table_digests[table][role] = table_digest(rows)
                role_table_counts[table][role] = len(rows)
                _require(
                    canonical_json_bytes(rows) == canonical_json_bytes(base_tables[table]),
                    f"{token}/{role}/{table}: semantic byte mismatch against BASE_A",
                )
        for comparison, other_role in COMPARISONS:
            equivalence.append({
                "case_token": token,
                "comparison": comparison,
                "per_table_row_counts": {
                    table: {"BASE_A": role_table_counts[table]["BASE_A"], other_role: role_table_counts[table][other_role]}
                    for table in SEMANTIC_TABLES
                },
                "per_table_digests": {
                    table: {"BASE_A": role_table_digests[table]["BASE_A"], other_role: role_table_digests[table][other_role]}
                    for table in SEMANTIC_TABLES
                },
                "semantic_equivalence": True,
                "first_mismatch": None,
            })
        determinism_cases.append({
            "case_token": token,
            "base_a_base_b_equivalent": True,
            "order_permutation_equivalent": True,
            "independent_validator_equivalent": True,
            "first_mismatch": None,
            "comparison_table_digests": role_table_digests,
            "comparison_table_row_counts": role_table_counts,
        })
    equivalence.sort(key=lambda row: row_key("equivalence", row))
    return equivalence, determinism_cases


def _resource_rows(orchestration_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {field: row[field] for field in REQUIRED_FIELDS["resource"]}
        for row in orchestration_rows
    ]
    rows.sort(key=lambda row: row_key("resource", row))
    return rows


def _safety(import_reports: Mapping[str, list[str]]) -> dict[str, Any]:
    return {
        **{field: False for field in SAFETY_FALSE_FIELDS},
        "import_reports": {name: list(modules) for name, modules in sorted(import_reports.items())},
    }


def validate_and_aggregate(
    *, source_manifest_path: Path, execution_lock_path: Path,
    orchestration_result_path: Path, process_root: Path, output_dir: Path,
) -> dict[str, Any]:
    """Validate the exact matrix and atomically write the six derived artifacts."""
    source_manifest_path = source_manifest_path.resolve()
    execution_lock_path = execution_lock_path.resolve()
    orchestration_result_path = orchestration_result_path.resolve()
    process_root = process_root.resolve()
    output_dir = output_dir.resolve()

    source_manifest, source_rows = _validate_source_manifest(source_manifest_path)
    execution_lock, source_paths, schema_sha256 = _validate_execution_lock(
        execution_lock_path, input_manifest_path=source_manifest_path,
    )
    import_reports = _validate_static_imports(source_paths)
    orchestration, resources = _validate_orchestration(orchestration_result_path)
    payloads, payload_hashes = _validate_payloads(
        process_root=process_root, resources=resources,
        source_rows=source_rows, schema_sha256=schema_sha256,
    )
    equivalence_rows, determinism_cases = _compare_semantic_tables(payloads)

    evidence: dict[str, Any] = {
        "preflight_id": PREFLIGHT_ID,
        "schema_version": 1,
        "terminal_status": VALID_STATUS,
        "config_sha256": RULING_CONFIGURATION_SHA256,
        "schema_sha256": schema_sha256,
        "source_manifest_sha256": source_manifest["input_manifest_sha256"],
        **{table: [] for table in TABLES},
        "safety": _safety(import_reports),
    }
    for table in SEMANTIC_TABLES:
        for token in EXPECTED_CASE_TOKENS:
            evidence[table].extend(payloads[(token, "BASE_A")]["tables"][table])
        evidence[table].sort(key=lambda row: row_key(table, row))
    evidence["equivalence"] = equivalence_rows
    evidence["resource"] = _resource_rows(orchestration["resource_rows"])
    policy = ValidationPolicy(
        expected_configuration_sha256=RULING_CONFIGURATION_SHA256,
        expected_schema_sha256=schema_sha256,
        expected_source_manifest_sha256=str(source_manifest["input_manifest_sha256"]),
        expected_case_tokens=EXPECTED_CASE_TOKENS,
    )
    evidence["criteria"] = derive_expected_criteria(evidence, policy)
    validation = validate_evidence(evidence, policy=policy)
    _require(validation.valid, "authoritative evidence invalid: " + "; ".join(validation.errors))

    tamper_rows = run_tamper_suite(evidence, policy=policy)
    require_all_tampers_rejected(tamper_rows)
    for row in tamper_rows:
        _require(
            any(row.expected_rejection.lower() in error.lower() for error in row.observed_errors),
            f"{row.regression_id}: expected rejection class was not observed",
        )

    authoritative_bytes = canonical_json_bytes(evidence)
    _require(
        len(authoritative_bytes) <= policy.aggregate_serialized_bytes,
        "aggregate authoritative serialization cap exceeded",
    )
    authoritative_sha = hashlib.sha256(authoritative_bytes).hexdigest()
    valid_certificate_ids = sorted(
        str(row["certificate_id"])
        for row in evidence["certificate"] if row.get("valid") is True
    )
    _require(valid_certificate_ids and len(valid_certificate_ids) == len(set(valid_certificate_ids)), "valid certificate IDs are empty or nonunique")
    memberships = {
        (str(row["case_token"]), str(row["trace_id"]), str(row["side"]).lower()): []
        for row in evidence["strip_membership"]
    }
    for row in evidence["strip_membership"]:
        memberships[(str(row["case_token"]), str(row["trace_id"]), str(row["side"]).lower())].append(str(row["point_id"]))
    certificate_index = {
        (str(row["case_token"]), str(row["certificate_id"])): row
        for row in evidence["certificate"] if row.get("valid") is True
    }
    bilateral_rows = []
    for row in evidence["destructive_negative"]:
        key = (str(row["case_token"]), str(row["certificate_id"]))
        cert = certificate_index.get(key)
        _require(cert is not None, f"{key}: deletion row does not bind a valid certificate")
        side = "left" if row["deletion_arm"] == "DELETE_LEFT" else "right"
        independently_recomputed = sorted_id_digest(memberships.get((key[0], str(cert["trace_id"]), side), []))
        _require(independently_recomputed == row["deleted_id_digest"], f"{key}/{side}: deletion support digest mismatch")
        bilateral_rows.append({
            **row,
            "independently_recomputed_side_support_digest": independently_recomputed,
            "rebuilt_deleted_side_support_count": row[f"rebuilt_{side}_support_count"],
            "trace_identity_unchanged": row["trace_id"] == cert["trace_id"],
        })
    bilateral_rows.sort(key=lambda row: (
        str(row["certificate_id"]),
        0 if row["deletion_arm"] == "DELETE_LEFT" else 1,
    ))
    expected_deletions = [(certificate_id, arm) for certificate_id in valid_certificate_ids for arm in ("DELETE_LEFT", "DELETE_RIGHT")]
    _require(
        [(row["certificate_id"], row["deletion_arm"]) for row in bilateral_rows] == expected_deletions,
        "bilateral deletion matrix is incomplete or noncanonical",
    )

    determinism = {
        "schema_version": 1,
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_determinism_and_permutation_evidence_v1",
        "terminal_status": VALID_STATUS,
        "semantic_tables": list(SEMANTIC_TABLES),
        "cases": determinism_cases,
    }
    safety_attestation = {
        "schema_version": 1,
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_static_safety_attestation_v1",
        "terminal_status": VALID_STATUS,
        "source_points_serialized": False,
        **evidence["safety"],
        "locked_source_sha256": execution_lock["source_sha256"],
        "derivation": "static_ast_and_locked_execution_evidence",
    }
    bilateral = {
        "schema_version": 1,
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_complete_bilateral_deletion_evidence_v1",
        "terminal_status": VALID_STATUS,
        "valid_certificate_ids": valid_certificate_ids,
        "rows": bilateral_rows,
    }
    tamper_results = {
        "schema_version": 1,
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_real_evidence_tamper_results_v1",
        "terminal_status": VALID_STATUS,
        "baseline_authoritative_artifact_sha256": authoritative_sha,
        "tests": [
            {
                "regression_id": row.regression_id,
                "expected_rejection": row.expected_rejection,
                "rejected": row.rejected,
                "observed_errors": list(row.observed_errors),
            }
            for row in tamper_rows
        ],
    }
    per_case_counts = {
        token: {
            table: sum(1 for row in evidence[table] if row.get("case_token") == token)
            for table in SEMANTIC_TABLES
        }
        for token in EXPECTED_CASE_TOKENS
    }
    rejection_denominators = [
        {
            "case_token": token,
            "stage": stage,
            "denominator": next(iter({int(row["denominator"]) for row in group})),
        }
        for token in EXPECTED_CASE_TOKENS
        for stage in sorted({row["stage"] for row in evidence["rejection_census"] if row["case_token"] == token})
        for group in [[row for row in evidence["rejection_census"] if row["case_token"] == token and row["stage"] == stage]]
    ]
    validator_compact = {
        "schema_version": 1,
        "preflight_id": PREFLIGHT_ID,
        "kind": "pwts_independent_validator_compact_v1",
        "terminal_status": VALID_STATUS,
        "authoritative_artifact_sha256": authoritative_sha,
        "authoritative_artifact_bytes": len(authoritative_bytes),
        "case_count": len(EXPECTED_CASE_TOKENS),
        "process_payload_count": len(payloads),
        "valid_certificate_count": len(valid_certificate_ids),
        "valid_certificate_ids": valid_certificate_ids,
        "destructive_negative_arm_count": len(evidence["destructive_negative"]),
        "per_case_semantic_row_counts": per_case_counts,
        "global_table_row_counts": validation.derived["table_row_counts"],
        "rejection_denominators": rejection_denominators,
        "table_digests": validation.derived["table_digests"],
        "criteria": evidence["criteria"],
        "case_payload_sha256": [
            {"case_token": token, "process_role": role, "sha256": payload_hashes[(token, role)]}
            for token in EXPECTED_CASE_TOKENS for role in PROCESS_ROLES
        ],
        "source_points_serialized": False,
        "accuracy_or_gold_statement_present": False,
    }

    outputs = {
        "authoritative_artifact.json": evidence,
        "validator_compact.json": validator_compact,
        "determinism.json": determinism,
        "bilateral_deletion.json": bilateral,
        "tamper_results.json": tamper_results,
        "safety_attestation.json": safety_attestation,
    }
    for name, value in outputs.items():
        _atomic_write(output_dir / name, value)
    return validator_compact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--orchestration-result", required=True, type=Path)
    parser.add_argument("--process-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_and_aggregate(
            source_manifest_path=args.source_manifest,
            execution_lock_path=args.execution_lock,
            orchestration_result_path=args.orchestration_result,
            process_root=args.process_root,
            output_dir=args.output_dir,
        )
    except (OSError, PreflightOutputError, ValueError, TypeError, KeyError) as exc:
        failure = {
            "preflight_id": PREFLIGHT_ID,
            "terminal_status": INVALID_STATUS,
            "error": f"{type(exc).__name__}: {exc}",
            "next_professor_packet_authorized": False,
            "gold_runtime_production_authorized": False,
        }
        print(canonical_json_bytes(failure).decode("utf-8"))
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
