from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from research.pwts_v1.config import get_config_copy
from research.pwts_v1.independent_validator import (
    RULING_CONFIGURATION_SHA256,
    VALID_STATUS,
    canonical_json_bytes,
)
from research.pwts_v1.schema import SCHEMA
from scripts import validate_pwts_preflight_outputs_v1 as aggregator
from tests.test_pwts_independent_validator_v1 import make_real_like_compact_evidence


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inactive_clone_rows(evidence: dict, token: str) -> None:
    manifest = copy.deepcopy(next(row for row in evidence["case_manifest"] if row["case_token"] == "C002"))
    manifest["case_token"] = token
    manifest["sampled_point_id_digest"] = hashlib.sha256(f"sample:{token}".encode()).hexdigest()
    manifest["validator_sampled_point_id_digest"] = manifest["sampled_point_id_digest"]
    evidence["case_manifest"].append(manifest)
    field = copy.deepcopy(next(row for row in evidence["field_summary"] if row["case_token"] == "C002"))
    field["case_token"] = token
    evidence["field_summary"].append(field)
    for row in [item for item in evidence["rejection_census"] if item["case_token"] == "C002"]:
        cloned = copy.deepcopy(row)
        cloned["case_token"] = token
        evidence["rejection_census"].append(cloned)


def _matrix_fixture(tmp_path: Path) -> dict[str, Path]:
    evidence = make_real_like_compact_evidence()
    for token in aggregator.EXPECTED_CASE_TOKENS[2:]:
        _inactive_clone_rows(evidence, token)
    for table in aggregator.SEMANTIC_TABLES:
        evidence[table].sort(key=lambda row: aggregator.row_key(table, row))

    locks = tmp_path / "locks"
    config_path = locks / "configuration.json"
    schema_path = locks / "schema.json"
    _write(config_path, get_config_copy())
    _write(schema_path, SCHEMA)
    schema_sha = _sha(schema_path)

    source_rows = []
    for token in aggregator.EXPECTED_CASE_TOKENS:
        raw_path = tmp_path / "raw" / f"{token}.ply"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes((f"locked-{token}-bytes\n" * 8).encode("ascii"))
        raw_sha = _sha(raw_path)
        coordinate_digest = hashlib.sha256(f"coordinates:{token}".encode()).hexdigest()
        ignored_digest = hashlib.sha256(f"ignored:{token}".encode()).hexdigest()
        header_sha = hashlib.sha256(f"header:{token}".encode()).hexdigest()
        source_rows.append({
            "case_token": token,
            "raw_path": str(raw_path.resolve()),
            "raw_byte_count": raw_path.stat().st_size,
            "raw_ply_sha256": raw_sha,
            "ply_header_sha256": header_sha,
            "parsed_vertex_count": 130,
            "finite_coordinate_count": 130,
            "exact_duplicate_count": 2,
            "ignored_property_name_digest": ignored_digest,
            "canonical_coordinate_multiset_sha256": coordinate_digest,
            "canonical_coordinate_multiset_digest": coordinate_digest,
        })
        manifest_row = next(row for row in evidence["case_manifest"] if row["case_token"] == token)
        manifest_row.update({
            "raw_path": str(raw_path.resolve()),
            "raw_byte_count": raw_path.stat().st_size,
            "raw_ply_sha256": raw_sha,
            "observed_raw_ply_sha256": raw_sha,
            "ply_header_sha256": header_sha,
            "parsed_vertex_count": 130,
            "finite_coordinate_count": 130,
            "exact_duplicate_count": 2,
            "ignored_property_name_digest": ignored_digest,
            "canonical_coordinate_multiset_digest": coordinate_digest,
            "config_sha256": RULING_CONFIGURATION_SHA256,
            "schema_sha256": schema_sha,
            "source_points_serialized": False,
        })

    source_manifest = {
        "schema_version": 1,
        "preflight_id": aggregator.PREFLIGHT_ID,
        "kind": "pwts_source_lock_v1",
        "case_count": 37,
        "case_tokens": list(aggregator.EXPECTED_CASE_TOKENS),
        "semantic_work_performed": False,
        "sampling_performed": False,
        "source_points_serialized": False,
        "cases": source_rows,
    }
    source_manifest["input_manifest_sha256"] = hashlib.sha256(
        aggregator._canonical_ascii_bytes(source_manifest)
    ).hexdigest()
    manifest_path = locks / "input_manifest.json"
    manifest_path.write_bytes(aggregator._canonical_ascii_bytes(source_manifest))

    safe_sources = {}
    for name in ("runner_source", "semantic_core_source", "independent_validator_source", "tamper_test_source"):
        path = locks / f"{name}.py"
        path.write_text("import hashlib\nimport json\n", encoding="utf-8")
        safe_sources[name] = path
    source_paths = {
        **safe_sources,
        "schema": schema_path,
        "configuration": config_path,
        "input_manifest": manifest_path,
    }
    execution_lock = {
        "preflight_id": aggregator.PREFLIGHT_ID,
        "kind": "pwts_execution_lock_v1",
        "expected_configuration_sha256": RULING_CONFIGURATION_SHA256,
        "source_sha256": {name: _sha(path) for name, path in source_paths.items()},
        "source_path": {name: str(path.resolve()) for name, path in source_paths.items()},
        "observed_configuration_sha256": RULING_CONFIGURATION_SHA256,
        "configuration_hash_match": True,
    }
    execution_lock["execution_lock_sha256"] = hashlib.sha256(
        aggregator._canonical_ascii_bytes(execution_lock)
    ).hexdigest()
    execution_lock_path = tmp_path / "execution_lock.json"
    execution_lock_path.write_bytes(aggregator._canonical_ascii_bytes(execution_lock))

    process_root = tmp_path / "processes"
    resource_rows = []
    for token in aggregator.EXPECTED_CASE_TOKENS:
        semantic = {
            table: [copy.deepcopy(row) for row in evidence[table] if row.get("case_token") == token]
            for table in aggregator.SEMANTIC_TABLES
        }
        for role in aggregator.PROCESS_ROLES:
            payload = {
                "preflight_id": aggregator.PREFLIGHT_ID,
                "case_token": token,
                "process_role": role,
                "source_points_serialized": False,
                "tables": semantic,
            }
            role_root = process_root / token / role
            payload_path = role_root / "output" / "case_payload.json"
            _write(payload_path, payload)
            stdout = role_root / "stdout.log"
            stderr = role_root / "stderr.log"
            stdout.write_bytes(b"")
            stderr.write_bytes(b"")
            resource_rows.append({
                "case_token": token,
                "process_role": role,
                "return_code": 0,
                "cpu_seconds": 1.0,
                "wall_seconds": 2.0,
                "process_tree_peak_rss_bytes": 50_000_000,
                "scratch_bytes": 1_000,
                "serialized_bytes": payload_path.stat().st_size,
                "monitor_available": True,
                "full_lifetime_accounting": True,
                "reported_passing": True,
                "rss_cap_exceeded": False,
                "scratch_cap_exceeded": False,
                "wall_cap_exceeded": False,
                "payload_sha256": _sha(payload_path),
                "stdout_sha256": _sha(stdout),
                "stderr_sha256": _sha(stderr),
                "success": True,
                "failure_code": None,
            })
    orchestration = {
        "schema_version": 1,
        "preflight_id": aggregator.PREFLIGHT_ID,
        "kind": "pwts_orchestration_result_v1",
        "terminal_status": aggregator.INVALID_STATUS,
        "failure_code": "AWAITING_INDEPENDENT_VALIDATION_AND_TAMPER_SUITE",
        "expected_configuration_sha256": RULING_CONFIGURATION_SHA256,
        "observed_configuration_sha256": RULING_CONFIGURATION_SHA256,
        "configuration_hash_match": True,
        "semantic_process_started": True,
        "completed_process_count": 148,
        "expected_process_count": 148,
        "workers": 1,
        "cpu_only": True,
        "fresh_process_per_case_variant": True,
        "process_role_order": list(aggregator.PROCESS_ROLES),
        "source_points_serialized": False,
        "gold_or_truth_consumed": False,
        "runtime_or_production_authorized": False,
        "next_professor_packet_authorized": False,
        "aggregate_cpu_seconds": sum(row["cpu_seconds"] for row in resource_rows),
        "aggregate_wall_seconds": sum(row["wall_seconds"] for row in resource_rows),
        "aggregate_serialized_bytes": sum(row["serialized_bytes"] for row in resource_rows),
        "resource_rows": resource_rows,
    }
    orchestration_path = tmp_path / "orchestration_result.json"
    _write(orchestration_path, orchestration)
    return {
        "source_manifest": manifest_path,
        "execution_lock": execution_lock_path,
        "orchestration": orchestration_path,
        "process_root": process_root,
    }


def _run(paths: dict[str, Path], output: Path) -> dict:
    return aggregator.validate_and_aggregate(
        source_manifest_path=paths["source_manifest"],
        execution_lock_path=paths["execution_lock"],
        orchestration_result_path=paths["orchestration"],
        process_root=paths["process_root"],
        output_dir=output,
    )


def _refresh_resource_binding(paths: dict[str, Path], token: str, role: str) -> None:
    payload_path = aggregator._payload_path(paths["process_root"], token, role)
    orchestration = aggregator._read_object(paths["orchestration"], "fixture orchestration")
    row = next(
        item for item in orchestration["resource_rows"]
        if item["case_token"] == token and item["process_role"] == role
    )
    row["serialized_bytes"] = payload_path.stat().st_size
    row["payload_sha256"] = _sha(payload_path)
    orchestration["aggregate_serialized_bytes"] = sum(
        item["serialized_bytes"] for item in orchestration["resource_rows"]
    )
    _write(paths["orchestration"], orchestration)


def test_complete_37_by_4_matrix_builds_all_six_valid_artifacts(tmp_path: Path) -> None:
    paths = _matrix_fixture(tmp_path)
    output = tmp_path / "derived"
    result = _run(paths, output)
    assert result["terminal_status"] == VALID_STATUS
    assert result["case_count"] == 37
    assert result["process_payload_count"] == 148
    assert result["valid_certificate_count"] == 1
    assert result["destructive_negative_arm_count"] == 2
    assert {path.name for path in output.iterdir()} == {
        "authoritative_artifact.json", "validator_compact.json", "determinism.json",
        "bilateral_deletion.json", "tamper_results.json", "safety_attestation.json",
    }
    tamper = aggregator._read_object(output / "tamper_results.json", "tamper")
    assert len(tamper["tests"]) == 14
    assert all(row["rejected"] for row in tamper["tests"])


def test_missing_one_of_148_payloads_fails_closed_before_writes(tmp_path: Path) -> None:
    paths = _matrix_fixture(tmp_path)
    aggregator._payload_path(paths["process_root"], "C037", "INDEPENDENT_VALIDATOR").unlink()
    output = tmp_path / "derived"
    with pytest.raises(aggregator.PreflightOutputError, match="exact 148-file matrix"):
        _run(paths, output)
    assert not output.exists()


def test_role_semantic_mismatch_is_rejected_even_with_updated_file_binding(tmp_path: Path) -> None:
    paths = _matrix_fixture(tmp_path)
    payload_path = aggregator._payload_path(paths["process_root"], "C002", "BASE_B")
    payload = aggregator._read_object(payload_path, "payload")
    payload["tables"]["field_summary"][0]["seed_eligible_count"] = 7
    _write(payload_path, payload)
    _refresh_resource_binding(paths, "C002", "BASE_B")
    with pytest.raises(aggregator.PreflightOutputError, match="semantic byte mismatch"):
        _run(paths, tmp_path / "derived")


def test_payload_hash_tamper_fails_before_semantic_acceptance(tmp_path: Path) -> None:
    paths = _matrix_fixture(tmp_path)
    payload_path = aggregator._payload_path(paths["process_root"], "C003", "ORDER_PERMUTED")
    payload_path.write_bytes(payload_path.read_bytes() + b" ")
    with pytest.raises(aggregator.PreflightOutputError, match="payload byte count mismatch"):
        _run(paths, tmp_path / "derived")
