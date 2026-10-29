from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from backend import f1_v2_cross_parent_opposite_fragment_rung_probe_v2 as core
from scripts import run_f1_v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2 as runner
from scripts import validate_f1_v2_cross_parent_ordinary_batch as validator


def _minimal_case_payload(token: str) -> dict:
    return {
        "case_token": token,
        "scan_sha256": "1" * 64,
        "candidate_pool_sha256": "2" * 64,
        "a0_prediction_sha256": "3" * 64,
        "frozen_control_probe_sha256": "4" * 64,
        "complete_edge_baseline_probe_sha256": "5" * 64,
        "mapping_case_sha256": "6" * 64,
        "output_table_hashes": {"table_sha256": "7" * 64},
        "case_counters": {"counter": 1},
        "ordinary_v2_criteria": {
            key: True for key in core.ORDINARY_V2_CRITERION_KEYS
        },
        "global_deletion_counterfactual": {
            "deletion_evidence_sha256": "8" * 64,
        },
    }


def _result(token: str) -> dict:
    payload = _minimal_case_payload(token)
    return {
        "case_token": token,
        "payload": payload,
        "pre_rung_gate_criteria": {
            key: True for key in runner.MAPPING_PREFLIGHT_CRITERIA
        },
        "wall_seconds": 1.25,
        "peak_rss_bytes": 1234,
        "input_json_bytes": 5678,
        "serialized_output_bytes": len(runner.canonical_json_bytes(payload)),
    }


def test_runner_schema_is_the_independent_validator_schema() -> None:
    assert runner.SCHEMA_VERSION == 2
    assert runner.KIND == validator.ORDINARY_ARTIFACT_KIND
    assert runner.KIND == "f1_v2_cross_parent_opposite_fragment_rung_ordinary_batch_v2"
    assert runner.CASE_FIELDS is validator.CASE_FIELDS
    assert runner.TOP_FIELDS is validator.TOP_FIELDS
    assert (
        runner.AGGREGATE_TABLE_SOURCE_FIELDS
        is validator.AGGREGATE_TABLE_SOURCE_FIELDS
    )


def test_case_row_matches_exact_schema_and_canonical_case_digest(tmp_path: Path) -> None:
    criteria = {key: True for key in runner.MAPPING_PREFLIGHT_CRITERIA}
    result = _result("C001")
    case_path = tmp_path / "batch_cases" / "C001.json"
    row = runner._build_case_row(result=result, case_path=case_path)
    encoded = runner.canonical_json_bytes(result["payload"])
    assert set(row) == set(validator.CASE_FIELDS)
    assert row["case_artifact_path"] == str(case_path.resolve())
    assert row["case_artifact_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert row["serialized_output_bytes"] == len(encoded)
    assert row["pre_rung_gate_criteria"] == criteria
    assert row["pre_rung_gate_sha256"] == runner._sha_json(criteria)


def test_pre_rung_gate_is_false_first_and_source_derived() -> None:
    empty = runner.new_pre_rung_gate_criteria()
    assert tuple(empty) == tuple(runner.MAPPING_PREFLIGHT_CRITERIA)
    assert all(value is False for value in empty.values())

    fixture_path = Path(__file__).with_name(
        "test_f1_v2_cross_parent_opposite_fragment_rung_probe.py"
    )
    spec = importlib.util.spec_from_file_location("runner_v1_fixture", fixture_path)
    fixture_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixture_module)
    projected, control, baseline, kwargs = fixture_module._fixture()
    mapping = core._build_case_from_payloads(
        projected, frozen_control=control, complete_edge_baseline=baseline, **kwargs
    )
    expected = {
        "frozen_target_projection": mapping["frozen_target_projection"],
        "frozen_target_baseline_relation_table": mapping[
            "frozen_target_baseline_relation_table"
        ],
        "frozen_target_outcome_table": mapping["frozen_target_outcome_table"],
        "all_pair_membership_table": mapping["all_pair_membership_table"],
        "thresholds_sha256": runner._sha_json(mapping["thresholds"]),
        "mapping_case_summary": {
            "frozen_target_count": len(mapping["frozen_target_projection"]),
        },
    }
    derived = runner._derive_pre_rung_gate(
        token=kwargs["case_token"], source_hashes_match=True,
        projected=projected, frozen_control=control,
        regenerated_control=control, frozen_baseline=baseline,
        regenerated_baseline=baseline, mapping_payload=mapping,
        expected_mapping=expected,
    )
    assert set(derived) == set(runner.MAPPING_PREFLIGHT_CRITERIA)
    assert all(derived.values())
    tampered_control = {**control, "scan_sha256": "f" * 64}
    rejected = runner._derive_pre_rung_gate(
        token=kwargs["case_token"], source_hashes_match=True,
        projected=projected, frozen_control=tampered_control,
        regenerated_control=control, frozen_baseline=baseline,
        regenerated_baseline=baseline, mapping_payload=mapping,
        expected_mapping=expected,
    )
    assert rejected["all_frozen_control_probe_hashes_reproduce"] is False


def test_full_aggregate_tables_are_literal_untruncated_case_concatenations() -> None:
    payloads = {}
    for index, token in enumerate(runner.EXPECTED_CASE_TOKENS):
        payload = {}
        for aggregate_name, source_name in runner.AGGREGATE_TABLE_SOURCE_FIELDS.items():
            if aggregate_name == "global_deletion_counterfactual_table":
                payload[source_name] = {"ordinal": index, "raw": [index, index + 1]}
            else:
                payload[source_name] = [{"ordinal": index, "raw": [index]}]
        payloads[token] = payload
    tables = runner._aggregate_tables(payloads)
    assert set(tables) == set(validator.AGGREGATE_TABLE_SOURCE_FIELDS)
    assert all(len(rows) == 37 for rows in tables.values())
    for rows in tables.values():
        assert [row["case_token"] for row in rows] == list(
            runner.EXPECTED_CASE_TOKENS
        )
        assert [row["ordinal"] for row in rows] == list(range(37))


def test_persistence_uses_canonical_sibling_cases_and_exact_embedded_payloads(
    tmp_path: Path,
) -> None:
    payloads = {
        token: {"case_token": token, "nested": {"b": 2, "a": 1}}
        for token in runner.EXPECTED_CASE_TOKENS
    }
    output = tmp_path / "ordinary.json"
    artifact = {
        "schema_version": 2,
        "kind": runner.KIND,
        "case_payloads": [payloads[token] for token in runner.EXPECTED_CASE_TOKENS],
    }
    runner._persist_authoritative_artifact(
        output_path=output, artifact=artifact, case_payloads=payloads
    )
    sibling = tmp_path / "ordinary_cases"
    for token in runner.EXPECTED_CASE_TOKENS:
        case_path = sibling / f"{token}.json"
        assert case_path.read_bytes() == runner.canonical_json_bytes(payloads[token])
        assert json.loads(case_path.read_bytes()) == payloads[token]
    assert output.read_bytes() == runner.canonical_json_bytes(artifact) + b"\n"
    assert json.loads(output.read_bytes())["case_payloads"] == artifact["case_payloads"]


def test_case_worker_checks_sources_and_mapping_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"source-{index}.json" for index in range(5)]
    for index, path in enumerate(paths):
        path.write_bytes(runner.canonical_json_bytes({"source": index}))
    hashes = [runner.sha256_file(path) for path in paths]
    expected_mapping = {
        "frozen_target_projection": [{"case_token": "C001", "id": "T"}],
        "frozen_target_baseline_relation_table": [
            {"case_token": "C001", "id": "T"}
        ],
        "all_pair_membership_table": [{"pair": "P"}],
        "thresholds_sha256": runner._sha_json({"threshold": 1}),
    }
    payload = {
        **_minimal_case_payload("C001"),
        "thresholds": {"threshold": 1},
        "frozen_target_projection": expected_mapping["frozen_target_projection"],
        "frozen_target_baseline_relation_table": expected_mapping[
            "frozen_target_baseline_relation_table"
        ],
        "all_pair_membership_table": expected_mapping["all_pair_membership_table"],
    }
    monkeypatch.setattr(runner.core, "project_mapping_preflight_input", lambda _raw: {})
    monkeypatch.setattr(
        runner.complete_core, "regenerate_frozen_control_probe",
        lambda *_args, **_kwargs: {"source": 3},
    )
    monkeypatch.setattr(
        runner.complete_core, "build_v2_complete_same_surface_edge_probe",
        lambda *_args, **_kwargs: {"source": 4},
    )
    mapping_payload = {"mapping": True}
    monkeypatch.setattr(
        runner.core, "_build_case_from_payloads",
        lambda *_args, **_kwargs: mapping_payload,
    )
    monkeypatch.setattr(
        runner.core, "validate_v2_cross_parent_mapping_case_payload",
        lambda _payload, **_kwargs: None,
    )
    derived_gate = {
        key: True for key in runner.MAPPING_PREFLIGHT_CRITERIA
    }
    monkeypatch.setattr(runner, "_derive_pre_rung_gate", lambda **_kwargs: derived_gate)
    monkeypatch.setattr(
        runner.core, "build_v2_cross_parent_ordinary_case",
        lambda _projected, **_kwargs: payload,
    )
    monkeypatch.setattr(
        runner.core, "validate_v2_cross_parent_ordinary_case_payload",
        lambda _payload, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_peak_rss_bytes", lambda: 4096)
    task = (
        "C001", *(str(path) for path in paths), *hashes, expected_mapping,
    )
    result = runner._run_case(task)
    assert result["payload"] == payload
    assert result["pre_rung_gate_criteria"] == derived_gate
    assert result["peak_rss_bytes"] == 4096
    assert result["serialized_output_bytes"] == len(
        runner.canonical_json_bytes(payload)
    )
    tampered = list(task)
    tampered[6] = "f" * 64
    with pytest.raises(ValueError, match="source byte hash mismatch"):
        runner._run_case(tuple(tampered))


def test_worker_bound_and_fresh_process_contract_are_explicit(tmp_path: Path) -> None:
    kwargs = {
        "inference_manifest_path": tmp_path / "missing-inference.json",
        "candidate_pool_manifest_path": tmp_path / "missing-pool.json",
        "a0_prediction_root": tmp_path,
        "frozen_control_manifest_path": tmp_path / "missing-control.json",
        "complete_edge_baseline_manifest_path": tmp_path / "missing-baseline.json",
        "mapping_preflight_path": tmp_path / "missing-mapping.json",
        "mapping_compact_evidence_path": tmp_path / "missing-compact.json",
        "output_path": tmp_path / "out.json",
    }
    with pytest.raises(ValueError, match=r"workers must be in \[1,8\]"):
        runner.run_batch(**kwargs, workers=0)
    with pytest.raises(ValueError, match=r"workers must be in \[1,8\]"):
        runner.run_batch(**kwargs, workers=9)
    source = inspect.getsource(runner.run_batch)
    assert "max_tasks_per_child=1" in source
    assert "max_workers=workers" in source


def test_cli_dispatches_all_authoritative_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}

    def fake_run_batch(**kwargs):
        captured.update(kwargs)
        return {"case_count": 37, "packet_status": runner.PACKET_STATUS}

    monkeypatch.setattr(runner, "run_batch", fake_run_batch)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "a" * 64)
    output = tmp_path / "ordinary.json"
    code = runner.main([
        "--inference-manifest", str(tmp_path / "i.json"),
        "--candidate-pool-manifest", str(tmp_path / "p.json"),
        "--a0-prediction-root", str(tmp_path / "a0"),
        "--frozen-control-manifest", str(tmp_path / "c.json"),
        "--complete-edge-baseline-manifest", str(tmp_path / "b.json"),
        "--mapping-preflight", str(tmp_path / "m.json"),
        "--mapping-compact-evidence", str(tmp_path / "mc.json"),
        "--output", str(output),
        "--workers", "8",
    ])
    assert code == 0
    assert captured["workers"] == 8
    assert captured["output_path"] == output
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["case_count"] == 37
    assert stdout["packet_status"] == runner.PACKET_STATUS
