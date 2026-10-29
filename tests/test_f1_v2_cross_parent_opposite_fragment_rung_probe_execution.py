from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import diagnose_f1_v2_cross_parent_opposite_fragment_rung_probe as diagnose
from scripts import run_f1_v2_cross_parent_opposite_fragment_rung_probe_batch as batch_module
from scripts.diagnose_f1_v2_cross_parent_opposite_fragment_rung_probe import (
    PROBE_KIND,
    PRIMITIVE_PROFILE,
    build_cross_parent_rung_probe_artifact,
)
from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_batch import (
    BATCH_KIND,
    CASE_ROW_FIELDS,
    MAX_BATCH_WORKERS,
    PEAK_RSS_LIMIT_BYTES,
)
from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_replays import (
    DESIGNATED_REPLAY_CASES,
    REPLAY_KIND,
    SELECTION_REASONS,
    _resource_record,
    canonical_replay_attestation_bytes,
    validate_replay_attestation,
)
from scripts.run_f1_v2_cross_parent_opposite_fragment_rung_probe_synthetic_resource import (
    ADDED_CROSS_PARENT_RUNG_COUNT,
    BASELINE_COMPLETE_EDGE_COUNT,
    CANDIDATE_EVALUATION_COUNT,
    COMPONENT_INDEX_BUCKET_COUNT,
    EFFECTIVE_EDGE_COUNT,
    INHERITED_LOCAL_RUNG_COUNT,
    KIND as RESOURCE_KIND,
    MODE_COUNT,
    MODES_PER_PARENT,
    PARENT_COUNT,
    PROFILE as RESOURCE_PROFILE_ID,
    RESOURCE_PROFILE,
    RUN_FIELDS as RESOURCE_RUN_FIELDS,
    _record_hash,
    validate_synthetic_resource_attestation,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _ordinary_row(token: str) -> dict:
    output_hashes = {
        "added_cross_parent_rung_table_sha256": _sha(token + "-rungs"),
        "component_rung_attribution_table_sha256": _sha(token + "-component-attribution"),
        "frozen_target_extension_table_sha256": _sha(token + "-targets"),
        "certificate_novelty_table_sha256": _sha(token + "-novelty"),
        "local_plane_pair_records_sha256": _sha(token + "-pairs"),
    }
    row = {
        "case_token": token,
        "status": "SUCCESS",
        "replacement_probe_sha256": _sha(token + "-replacement"),
        "frozen_control_probe_sha256": _sha(token + "-control"),
        "complete_edge_baseline_probe_sha256": _sha(token + "-baseline"),
        "scan_sha256": _sha(token + "-scan"),
        "candidate_pool_sha256": _sha(token + "-pool"),
        "a0_prediction_sha256": _sha(token + "-a0"),
        "input_table_hashes": {"parent_table_sha256": _sha(token + "-parents")},
        "baseline_table_hashes": {"baseline_pair_records_sha256": _sha(token + "-baseline-pairs")},
        "output_table_hashes": output_hashes,
        "added_cross_parent_rung_table_sha256": output_hashes["added_cross_parent_rung_table_sha256"],
        "effective_mode_edge_semantic_sha256": _sha(token + "-effective"),
        "local_plane_pair_records_sha256": output_hashes["local_plane_pair_records_sha256"],
        "certificate_novelty_table_sha256": output_hashes["certificate_novelty_table_sha256"],
        "input_contract_attestation_sha256": _sha(token + "-input-attestation"),
        "baseline_rebuild_attestation_sha256": _sha(token + "-baseline-attestation"),
        "cross_parent_rung_attestation_sha256": _sha(token + "-rung-attestation"),
        "global_deletion_counterfactual_sha256": _sha(token + "-deletion"),
        "engineering_profile_sha256": _sha(token + "-engineering"),
        "inference_semantic_sha256": _sha(token + "-semantic"),
        "case_counters": {"added_cross_parent_rung_count": 10},
        "wall_seconds": 1.0,
        "peak_rss_bytes": 200_000_000,
        "input_json_bytes": 10_000,
        "serialized_output_bytes": 20_000,
    }
    assert set(row) == CASE_ROW_FIELDS
    return row


def _batch() -> dict:
    return {
        "schema_version": 1,
        "kind": BATCH_KIND,
        "profile": PRIMITIVE_PROFILE,
        "case_count": 37,
        "successful_case_count": 37,
        "workers": 8,
        "isolated_process_per_case": True,
        "gold_open_authorized": False,
        "cases": [_ordinary_row(f"C{index:03d}") for index in range(1, 38)],
    }


def _replay(batch: dict) -> dict:
    rows = {row["case_token"]: row for row in batch["cases"]}
    cases = []
    for token in sorted(DESIGNATED_REPLAY_CASES):
        ordinary = rows[token]
        runs = [_resource_record(ordinary, run_index=index) for index in (1, 2, 3)]
        cases.append(
            {
                "case_token": token,
                "selection_reason": SELECTION_REASONS[token],
                "batch_replacement_probe_sha256": ordinary["replacement_probe_sha256"],
                "batch_frozen_control_probe_sha256": ordinary["frozen_control_probe_sha256"],
                "batch_complete_edge_baseline_probe_sha256": ordinary["complete_edge_baseline_probe_sha256"],
                "batch_added_cross_parent_rung_table_sha256": ordinary["added_cross_parent_rung_table_sha256"],
                "batch_effective_mode_edge_semantic_sha256": ordinary["effective_mode_edge_semantic_sha256"],
                "batch_local_plane_pair_records_sha256": ordinary["local_plane_pair_records_sha256"],
                "batch_certificate_novelty_table_sha256": ordinary["certificate_novelty_table_sha256"],
                "batch_inference_semantic_sha256": ordinary["inference_semantic_sha256"],
                "replay_count": 3,
                "replay_replacement_probe_sha256s": [ordinary["replacement_probe_sha256"]] * 3,
                "maximum_peak_rss_bytes": 200_000_000,
                "runs": runs,
            }
        )
    return {
        "schema_version": 1,
        "kind": REPLAY_KIND,
        "profile": PRIMITIVE_PROFILE,
        "research_only": True,
        "inference_safe": True,
        "gold_open_authorized": False,
        "case_count": 4,
        "total_isolated_run_count": 12,
        "probe_manifest_sha256": _sha("batch"),
        "inference_manifest_sha256": _sha("inference"),
        "candidate_pool_manifest_sha256": _sha("pool-manifest"),
        "frozen_control_manifest_sha256": _sha("control-manifest"),
        "complete_edge_baseline_manifest_sha256": _sha("baseline-manifest"),
        "cases": cases,
    }


def _resource_attestation() -> dict:
    runs = []
    for index in (1, 2, 3):
        row = {
            "run_index": index,
            "status": "SUCCESS",
            "parent_count": PARENT_COUNT,
            "mode_count": MODE_COUNT,
            "modes_per_parent": MODES_PER_PARENT,
            "adjacent_parent_pair_count": 52_000,
            "cross_parent_rung_candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
            "baseline_complete_same_surface_edge_count": BASELINE_COMPLETE_EDGE_COUNT,
            "inherited_cross_mode_local_edge_count": INHERITED_LOCAL_RUNG_COUNT,
            "added_cross_parent_rung_count": ADDED_CROSS_PARENT_RUNG_COUNT,
            "effective_mode_edge_count": EFFECTIVE_EDGE_COUNT,
            "candidate_index_entry_count": CANDIDATE_EVALUATION_COUNT,
            "component_edge_index_bucket_count": COMPONENT_INDEX_BUCKET_COUNT,
            "component_edge_index_entry_count": EFFECTIVE_EDGE_COUNT,
            "baseline_complete_edge_table_sha256": _sha("baseline-table"),
            "inherited_local_rung_table_sha256": _sha("local-rungs"),
            "added_cross_parent_rung_table_sha256": _sha("added-rungs"),
            "effective_mode_edge_semantic_sha256": _sha("effective"),
            "candidate_index_sha256": _sha("candidates"),
            "component_edge_index_sha256": _sha("component-index"),
            "canonical_resource_semantic_sha256": _sha("semantic"),
            "wall_seconds": float(index),
            "peak_rss_bytes": 100_000_000 + index,
        }
        assert set(row) == RESOURCE_RUN_FIELDS
        runs.append(row)
    record = {
        "schema_version": 1,
        "kind": RESOURCE_KIND,
        "profile": RESOURCE_PROFILE_ID,
        "resource_profile": RESOURCE_PROFILE,
        "resource_envelope_only": True,
        "physical_voxel_realizable": False,
        "resource_limits": {
            "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
            "parent_count": PARENT_COUNT,
            "mode_count": MODE_COUNT,
            "modes_per_parent": MODES_PER_PARENT,
            "adjacent_parent_pair_count": 52_000,
            "candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
            "baseline_complete_edge_count": BASELINE_COMPLETE_EDGE_COUNT,
            "inherited_local_rung_count": INHERITED_LOCAL_RUNG_COUNT,
            "added_cross_parent_rung_count": ADDED_CROSS_PARENT_RUNG_COUNT,
            "effective_edge_count": EFFECTIVE_EDGE_COUNT,
            "component_index_bucket_count": COMPONENT_INDEX_BUCKET_COUNT,
        },
        "runs": runs,
        "semantic_hashes_identical": True,
        "all_runs_below_16_gib": True,
    }
    record["attestation_sha256"] = _record_hash(record, "attestation_sha256")
    return record


def test_diagnostic_exact_cli_opaque_scan_a0_and_two_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = tmp_path / "scan.ply"
    pool = tmp_path / "C001.json"
    a0 = tmp_path / "a0.json"
    scan.write_bytes(b"opaque scan, not json")
    a0.write_bytes(b"opaque prediction, not json")
    pool.write_text(json.dumps({"allowed": True}), encoding="utf-8")
    control_sha = hashlib.sha256(b"control").hexdigest()
    baseline_sha = hashlib.sha256(b"baseline").hexdigest()
    observed = {}

    def replacement_build(projected, **kwargs):
        observed.update(kwargs)
        return {
            "schema_version": 1,
            "kind": PROBE_KIND,
            "profile": PRIMITIVE_PROFILE,
            "gold_open_authorized": False,
            "frozen_control_probe_sha256": kwargs["frozen_control_probe_sha256"],
            "complete_edge_baseline_probe_sha256": kwargs["complete_edge_baseline_probe_sha256"],
        }

    monkeypatch.setattr(diagnose, "_control_api", lambda: (lambda *_a, **_k: {}, lambda _p: b"control", lambda *_a, **_k: None))
    monkeypatch.setattr(diagnose, "_baseline_api", lambda: (lambda *_a, **_k: {}, lambda _p: b"baseline", lambda *_a, **_k: None))
    monkeypatch.setattr(
        diagnose,
        "_replacement_api",
        lambda: (
            replacement_build,
            lambda payload: json.dumps(payload, sort_keys=True).encode("ascii"),
            lambda raw: {"allowed": raw["allowed"]},
            lambda *_a, **_k: None,
        ),
    )
    payload, encoded = build_cross_parent_rung_probe_artifact(
        scan_path=scan,
        candidate_pool_path=pool,
        a0_prediction_path=a0,
        expected_frozen_control_probe_sha256=control_sha,
        expected_complete_edge_baseline_probe_sha256=baseline_sha,
    )
    assert encoded and payload["gold_open_authorized"] is False
    assert observed["case_token"] == "C001"
    assert observed["scan_sha256"] == hashlib.sha256(scan.read_bytes()).hexdigest()
    assert observed["a0_prediction_sha256"] == hashlib.sha256(a0.read_bytes()).hexdigest()
    options = {option for action in diagnose.build_parser()._actions for option in action.option_strings if option not in {"-h", "--help"}}
    assert options == {
        "--scan", "--candidate-pool", "--a0-prediction",
        "--expected-frozen-control-probe-sha256",
        "--expected-complete-edge-baseline-probe-sha256", "--output",
    }


def test_batch_contract_is_compact_gold_closed_and_worker_bounded() -> None:
    required = {
        "replacement_probe_sha256", "frozen_control_probe_sha256",
        "complete_edge_baseline_probe_sha256", "baseline_rebuild_attestation_sha256",
        "cross_parent_rung_attestation_sha256", "global_deletion_counterfactual_sha256",
        "peak_rss_bytes", "serialized_output_bytes",
    }
    assert required <= CASE_ROW_FIELDS
    assert not any("replay" in field or "source_points" in field for field in CASE_ROW_FIELDS)
    assert MAX_BATCH_WORKERS == 8


def test_ordinary_worker_extracts_exact_compact_core_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = tmp_path / "scan.ply"
    pool = tmp_path / "C001.json"
    a0 = tmp_path / "a0.json"
    output = tmp_path / "C001-output.json"
    scan.write_bytes(b"scan")
    pool.write_bytes(b"pool")
    a0.write_bytes(b"a0")
    output_hashes = {
        "added_cross_parent_rung_table_sha256": _sha("rungs"),
        "component_rung_attribution_table_sha256": _sha("attribution"),
        "frozen_target_extension_table_sha256": _sha("targets"),
        "certificate_novelty_table_sha256": _sha("novelty"),
        "local_plane_pair_records_sha256": _sha("pairs"),
    }
    payload = {
        "frozen_control_probe_sha256": _sha("control"),
        "complete_edge_baseline_probe_sha256": _sha("baseline"),
        "scan_sha256": _sha("scan"),
        "candidate_pool_sha256": _sha("pool"),
        "a0_prediction_sha256": _sha("a0"),
        "input_table_hashes": {"parent_table_sha256": _sha("parent-table")},
        "baseline_table_hashes": {"baseline_pair_records_sha256": _sha("baseline-pairs")},
        "output_table_hashes": output_hashes,
        "input_contract_attestation": {"input_contract_attestation_sha256": _sha("input-att")},
        "baseline_rebuild_attestation": {"baseline_rebuild_attestation_sha256": _sha("baseline-att")},
        "cross_parent_rung_attestation": {
            "cross_parent_rung_attestation_sha256": _sha("rung-att"),
            "effective_mode_edge_semantic_sha256": _sha("effective"),
        },
        "global_deletion_counterfactual": {"global_deletion_counterfactual_sha256": _sha("deletion")},
        "engineering_profile_sha256": _sha("engineering"),
        "inference_semantic_sha256": _sha("semantic"),
        "case_counters": {"added_cross_parent_rung_count": 7},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("ascii")
    monkeypatch.setattr(
        batch_module,
        "build_cross_parent_rung_probe_artifact",
        lambda **_kwargs: (payload, encoded),
    )
    row = batch_module._run_case(
        (
            "C001", str(scan), str(pool), str(a0), _sha("control"),
            _sha("baseline"), str(output),
        )
    )
    assert set(row) == CASE_ROW_FIELDS
    assert row["effective_mode_edge_semantic_sha256"] == _sha("effective")
    assert row["added_cross_parent_rung_table_sha256"] == _sha("rungs")
    assert row["local_plane_pair_records_sha256"] == _sha("pairs")
    assert row["certificate_novelty_table_sha256"] == _sha("novelty")
    assert output.read_bytes() == encoded


def test_exact_designated_replay_attestation_validates() -> None:
    batch = _batch()
    attestation = _replay(batch)
    validate_replay_attestation(attestation, batch_manifest=batch)
    assert {case["case_token"] for case in attestation["cases"]} == DESIGNATED_REPLAY_CASES
    assert sum(len(case["runs"]) for case in attestation["cases"]) == 12
    assert canonical_replay_attestation_bytes(attestation) == canonical_replay_attestation_bytes(deepcopy(attestation))


@pytest.mark.parametrize(
    "run_field",
    [
        "replacement_probe_sha256", "frozen_control_probe_sha256",
        "complete_edge_baseline_probe_sha256", "added_cross_parent_rung_table_sha256",
        "effective_mode_edge_semantic_sha256", "local_plane_pair_records_sha256",
        "certificate_novelty_table_sha256", "inference_semantic_sha256",
    ],
)
def test_replay_every_identity_must_equal_ordinary(run_field: str) -> None:
    batch = _batch()
    attestation = _replay(batch)
    attestation["cases"][0]["runs"][0][run_field] = _sha("mutated")
    with pytest.raises(ValueError, match="differs from batch|replacement hash"):
        validate_replay_attestation(attestation, batch_manifest=batch)


def test_replay_rss_boundary_and_gold_flag_fail_closed() -> None:
    batch = _batch()
    attestation = _replay(batch)
    attestation["gold_open_authorized"] = True
    with pytest.raises(ValueError, match="safety"):
        validate_replay_attestation(attestation, batch_manifest=batch)
    attestation = _replay(batch)
    run = attestation["cases"][0]["runs"][0]
    run["peak_rss_bytes"] = PEAK_RSS_LIMIT_BYTES
    with pytest.raises(ValueError, match="16 GiB|resource profile"):
        validate_replay_attestation(attestation, batch_manifest=batch)


def test_synthetic_exact_1688000_edge_envelope_validates() -> None:
    attestation = _resource_attestation()
    assert validate_synthetic_resource_attestation(attestation) == 100_000_003
    assert all(run["effective_mode_edge_count"] == 1_688_000 for run in attestation["runs"])


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda value: value.update({"physical_voxel_realizable": True}), "abstract envelope"),
        (lambda value: value["runs"][0].update({"effective_mode_edge_count": 1_687_999}), "effective_mode_edge_count"),
        (lambda value: value["runs"][1].update({"canonical_resource_semantic_sha256": _sha("different")}), "semantic hashes"),
        (lambda value: value["runs"][2].update({"peak_rss_bytes": PEAK_RSS_LIMIT_BYTES}), "RSS"),
    ],
)
def test_synthetic_resource_attestation_fails_closed(mutation, message: str) -> None:
    attestation = _resource_attestation()
    mutation(attestation)
    attestation["attestation_sha256"] = _record_hash(attestation, "attestation_sha256")
    with pytest.raises(ValueError, match=message):
        validate_synthetic_resource_attestation(attestation)
