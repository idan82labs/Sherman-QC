from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import assemble_pwts_preflight_evidence_v1 as assembler


def _write(path: Path, value: object) -> None:
    path.write_bytes(assembler.canonical_json_bytes(value))


def _evidence_fixture(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    tokens = ("C001",)
    monkeypatch.setattr(assembler, "EXPECTED_CASE_TOKENS", tokens)
    source = {
        "preflight_id": assembler.PRELIGHT_ID, "kind": "pwts_source_lock_v1",
        "case_tokens": list(tokens), "semantic_work_performed": False,
        "source_points_serialized": False, "cases": [{"case_token": "C001"}],
    }
    source["input_manifest_sha256"] = hashlib.sha256(
        assembler.canonical_json_bytes(source)
    ).hexdigest()
    artifact = tmp_path / "authoritative.json"
    artifact.write_bytes(b"authoritative-complete-table-stream")
    artifact_sha = assembler.sha256_file(artifact)
    resource_rows = []
    for role in assembler.PROCESS_ROLES:
        resource_rows.append({
            "case_token": "C001", "process_role": role, "success": True,
            "monitor_available": True, "rss_cap_exceeded": False,
            "full_lifetime_accounting": True, "reported_passing": True,
            "scratch_cap_exceeded": False, "wall_cap_exceeded": False,
            "cpu_seconds": 1.0, "wall_seconds": 2.0,
            "process_tree_peak_rss_bytes": 100, "scratch_bytes": 10,
            "serialized_bytes": 20, "payload_sha256": "a" * 64,
        })
    payloads = {
        "input_manifest": source,
        "execution_lock": {
            "preflight_id": assembler.PRELIGHT_ID,
            "expected_configuration_sha256": assembler.EXPECTED_CONFIGURATION_SHA256,
            "observed_configuration_sha256": assembler.EXPECTED_CONFIGURATION_SHA256,
            "configuration_hash_match": True, "source_sha256": {"runner": "b" * 64},
        },
        "orchestration_result": {
            "preflight_id": assembler.PRELIGHT_ID, "resource_rows": resource_rows,
        },
        "validator_compact": {
            "preflight_id": assembler.PRELIGHT_ID,
            "authoritative_artifact_sha256": artifact_sha,
            "valid_certificate_ids": ["CERT-1"], "table_digests": {"certificate": "c" * 64},
            "criteria": [
                {
                    "criterion_id": criterion, "result": True, "numerator": 1,
                    "denominator": 1, "source_table_digest": "d" * 64,
                }
                for criterion in assembler.VALIDATOR_CRITERIA
            ],
        },
        "determinism": {
            "preflight_id": assembler.PRELIGHT_ID,
            "cases": [{
                "case_token": "C001", "base_a_base_b_equivalent": True,
                "order_permutation_equivalent": True,
                "independent_validator_equivalent": True, "first_mismatch": None,
                "comparison_table_digests": {"certificate": "e" * 64},
            }],
        },
        "bilateral_deletion": {
            "preflight_id": assembler.PRELIGHT_ID,
            "rows": [
                {
                    "certificate_id": "CERT-1", "deletion_arm": arm,
                    "rebuilt_valid": False,
                    "first_rejection_code": (
                        "LEFT_SUPPORT_INSUFFICIENT" if arm == "DELETE_LEFT"
                        else "RIGHT_SUPPORT_INSUFFICIENT"
                    ),
                    "rebuilt_deleted_side_support_count": 0, "deleted_id_count": 32,
                    "deleted_id_digest": "f" * 64, "trace_identity_unchanged": True,
                }
                for arm in ("DELETE_LEFT", "DELETE_RIGHT")
            ],
        },
        "tamper_results": {
            "preflight_id": assembler.PRELIGHT_ID,
            "tests": [
                {
                    "regression_id": tamper_id, "rejected": True,
                    "expected_rejection": "rejected",
                    "observed_errors": ["artifact rejected"],
                }
                for tamper_id in assembler.TAMPER_IDS
            ],
        },
        "safety_attestation": {
            "preflight_id": assembler.PRELIGHT_ID, "source_points_serialized": False,
            **{field: False for field in assembler.SAFETY_FALSE_FIELDS},
        },
    }
    paths = {"authoritative_artifact": artifact}
    for label, payload in payloads.items():
        path = tmp_path / f"{label}.json"
        _write(path, payload)
        paths[label] = path
    return paths


def _assemble(paths: dict[str, Path], output: Path) -> dict:
    return assembler.assemble_evidence(
        input_manifest_path=paths["input_manifest"],
        execution_lock_path=paths["execution_lock"],
        orchestration_result_path=paths["orchestration_result"],
        authoritative_artifact_path=paths["authoritative_artifact"],
        validator_compact_path=paths["validator_compact"],
        determinism_path=paths["determinism"], deletion_path=paths["bilateral_deletion"],
        tamper_results_path=paths["tamper_results"],
        safety_attestation_path=paths["safety_attestation"], output_dir=output,
    )


def test_assembler_derives_valid_status_only_from_complete_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = _evidence_fixture(tmp_path, monkeypatch)
    result = _assemble(paths, tmp_path / "assembled")
    assert result["terminal_status"] == assembler.VALID_STATUS
    assert result["next_professor_packet_authorized"] is True
    assert result["gold_open_authorized"] is False
    assert (tmp_path / "assembled/integrity_index.json").is_file()


def test_assembler_fails_closed_when_one_tamper_regression_is_accepted(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = _evidence_fixture(tmp_path, monkeypatch)
    tamper = assembler._read(paths["tamper_results"])
    tamper["tests"][0]["rejected"] = False
    _write(paths["tamper_results"], tamper)
    result = _assemble(paths, tmp_path / "assembled")
    assert result["terminal_status"] == assembler.INVALID_STATUS
    criteria = {row["criterion_id"]: row["result"] for row in result["criteria"]}
    assert criteria["ALL_TAMPER_REGRESSIONS_REJECTED"] is False
