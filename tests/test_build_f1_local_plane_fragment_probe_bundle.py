from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_f1_local_plane_fragment_probe_bundle as bundle_module
from scripts.build_f1_local_plane_fragment_probe_bundle import (
    BUNDLE_KIND,
    LOCK_CRITERION_KEYS,
    build_local_plane_fragment_probe_bundle,
    build_lock_attestation,
    canonical_bundle_bytes,
    validate_lock_attestation,
)
from scripts.diagnose_f1_local_plane_fragment_probe import (
    ALLOWED_POOL_PROJECTION_FIELDS,
    build_parser as diagnose_parser,
    project_lpf_input,
)
from scripts.run_f1_local_plane_fragment_probe_replays import (
    KIND as REPLAY_KIND,
    _resource_record,
    canonical_replay_attestation_bytes,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@pytest.fixture(autouse=True)
def fake_core(monkeypatch):
    def validate(payload, *, projected_input=None):
        assert payload["schema_version"] == 2
        assert payload["kind"] == "f1_local_plane_fragment_probe_v2"
        assert projected_input is not None

    monkeypatch.setattr(
        bundle_module,
        "_core_api",
        lambda: (lambda *_args, **_kwargs: None, _canonical, validate),
    )
    monkeypatch.setattr(
        bundle_module,
        "_projected_rebuild",
        lambda pool_path, **_hashes: json.loads(
            (pool_path.parent / "probe.json").read_text(encoding="utf-8")
        ),
    )
    monkeypatch.setattr(
        bundle_module, "PREREGISTERED_INFERENCE_MANIFEST_SHA256", "a" * 64
    )
    monkeypatch.setattr(
        bundle_module, "PREREGISTERED_V2_POOL_MANIFEST_SHA256", "b" * 64
    )


def _pool(token: str) -> dict:
    return {
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "support_parent_table": [],
        "support_mode_table": [],
        "support_mode_edges": [],
        "plane_table": [],
        "transition_parent_table_sha256": "1" * 64,
        "transition_mode_table_sha256": "2" * 64,
        "transition_mode_edge_table_sha256": "3" * 64,
        "transition_local_spacing_mm": 1.0,
        "transition_atom_caps": {
            "input_point_cap": 200_000,
            "parent_cap": 4_000,
            "mode_cap": 16_000,
            "mode_children_per_parent_cap": 4,
            "parent_plus_mode_cap": 20_000,
            "mode_edge_cap": 200_000,
        },
        "working_point_budget": 200_000,
        "transient_sampled_point_count": 200_000,
        "excluded_case_token": token,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    rows = []
    artifacts = {}
    for index in range(1, 38):
        token = f"C{index:03d}"
        root = tmp_path / token
        root.mkdir(parents=True)
        scan, pool_path, prediction, probe = (
            root / "scan.ply",
            root / "pool.json",
            root / "prediction.json",
            root / "probe.json",
        )
        scan.write_bytes(f"scan-{token}".encode())
        pool_path.write_text(json.dumps(_pool(token), sort_keys=True), encoding="utf-8")
        prediction.write_bytes(f"prediction-{token}".encode())
        input_attestation = {
            "projection_profile": "LPF_INPUT_PROJECTION_V2",
            "input_contract_attestation_sha256": f"{index + 500:064x}",
        }
        payload = {
            "schema_version": 2,
            "kind": "f1_local_plane_fragment_probe_v2",
            "profile": "LOCAL_PLANE_FRAGMENT_PROBE_V2",
            "research_only": True,
            "inference_safe": True,
            "prediction_mutated": False,
            "birth_columns_enabled": False,
            "ownership_consumed": False,
            "solver_consumed": False,
            "truth_inputs_consumed": False,
            "source_points_consumed": False,
            "source_points_serialized": False,
            "scan_sha256": _sha(scan),
            "candidate_pool_sha256": _sha(pool_path),
            "a0_prediction_sha256": _sha(prediction),
            "input_table_hashes": {"parent_table_sha256": "1" * 64},
            "output_table_hashes": {"local_plane_orphan_table_sha256": "4" * 64},
            "input_contract_attestation": input_attestation,
            "inference_semantic_sha256": f"{index + 600:064x}",
            "engineering_profile_sha256": f"{index + 700:064x}",
            "local_plane_fragment_table": [],
            "local_plane_orphan_table": [],
            "local_plane_pair_records": [],
        }
        probe.write_bytes(_canonical(payload))
        mode_count = 300 if token == "C030" else (250 if token == "C003" else 100)
        edge_count = 400 if token == "C003" else (250 if token == "C030" else 200)
        counters = {
            "inherited_working_point_count": 200_000,
            "input_mode_count": mode_count,
            "input_same_surface_edge_count": edge_count // 2,
            "input_cross_mode_edge_count": edge_count - edge_count // 2,
        }
        row = {
            "case_token": token,
            "status": "SUCCESS",
            "probe_path": str(probe),
            "probe_sha256": _sha(probe),
            "scan_path": str(scan),
            "scan_sha256": _sha(scan),
            "candidate_pool_path": str(pool_path),
            "candidate_pool_sha256": _sha(pool_path),
            "a0_prediction_path": str(prediction),
            "a0_prediction_sha256": _sha(prediction),
            "input_table_hashes": payload["input_table_hashes"],
            "output_table_hashes": payload["output_table_hashes"],
            "input_contract_attestation_sha256": input_attestation[
                "input_contract_attestation_sha256"
            ],
            "inference_semantic_sha256": payload["inference_semantic_sha256"],
            "engineering_profile_sha256": payload["engineering_profile_sha256"],
            "case_counters": counters,
            "peak_rss_bytes": 100_000_000,
            "wall_seconds": 1.0,
            "input_json_bytes": pool_path.stat().st_size,
            "serialized_output_bytes": probe.stat().st_size,
        }
        rows.append(row)
        artifacts[token] = {"row": row, "payload": payload, "probe": probe}
    manifest = {
        "schema_version": 2,
        "kind": "f1_local_plane_fragment_probe_batch_v2",
        "profile": "LOCAL_PLANE_FRAGMENT_PROBE_V2",
        "case_count": 37,
        "successful_case_count": 37,
        "research_only": True,
        "inference_safe": True,
        "prediction_changed": False,
        "prediction_mutated": False,
        "birth_columns_enabled": False,
        "ownership_consumed": False,
        "solver_consumed": False,
        "truth_inputs_consumed": False,
        "source_points_consumed": False,
        "source_points_serialized": False,
        "isolated_process_per_case": True,
        "inference_manifest_sha256": "a" * 64,
        "candidate_pool_manifest_sha256": "b" * 64,
        "cases": rows,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    replay_cases = []
    for token, reason in (
        ("C001", "INHERITED_200000_POINT_REFERENCE"),
        ("C003", "MAXIMUM_TRANSITION_MODE_EDGE_COUNT"),
        ("C030", "MAXIMUM_TRANSITION_MODE_COUNT"),
    ):
        batch_hash = artifacts[token]["row"]["probe_sha256"]
        runs = [
            _resource_record(
                {
                    "probe_sha256": batch_hash,
                    "peak_rss_bytes": 100_000_000 + run_index,
                    "wall_seconds": 1.0,
                    "input_json_bytes": artifacts[token]["row"]["input_json_bytes"],
                    "serialized_output_bytes": artifacts[token]["row"]["serialized_output_bytes"],
                },
                run_index,
            )
            for run_index in (1, 2, 3)
        ]
        replay_cases.append(
            {
                "case_token": token,
                "selection_reason": reason,
                "working_point_count": 200_000,
                "transition_mode_count": artifacts[token]["row"]["case_counters"]["input_mode_count"],
                "transition_mode_edge_count": sum(
                    artifacts[token]["row"]["case_counters"][key]
                    for key in ("input_same_surface_edge_count", "input_cross_mode_edge_count")
                ),
                "batch_probe_sha256": batch_hash,
                "replay_count": 3,
                "replay_probe_sha256s": [batch_hash] * 3,
                "maximum_peak_rss_bytes": 100_000_003,
                "runs": runs,
            }
        )
    replay = {
        "schema_version": 2,
        "kind": REPLAY_KIND,
        "profile": "LOCAL_PLANE_FRAGMENT_PROBE_V2",
        "research_only": True,
        "inference_safe": True,
        "case_count": 3,
        "total_isolated_run_count": 9,
        "probe_manifest_sha256": _sha(manifest_path),
        "inference_manifest_sha256": "a" * 64,
        "candidate_pool_manifest_sha256": "b" * 64,
        "cases": replay_cases,
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_bytes(canonical_replay_attestation_bytes(replay))
    return manifest_path, replay_path, artifacts


def test_bundle_requires_replays_and_emits_complete_ready_lock(tmp_path: Path) -> None:
    manifest, replay, _ = _fixture(tmp_path)
    first = build_local_plane_fragment_probe_bundle(
        probe_manifest_path=manifest, replay_attestation_path=replay
    )
    second = build_local_plane_fragment_probe_bundle(
        probe_manifest_path=manifest, replay_attestation_path=replay
    )
    assert canonical_bundle_bytes(first) == canonical_bundle_bytes(second)
    assert first["schema_version"] == 2 and first["kind"] == BUNDLE_KIND
    assert first["locked"] is first["ready_for_gold"] is True
    assert first["replay_attestation_sha256"] == _sha(replay)
    assert set(first["lock_attestation"]["criteria"]) == set(LOCK_CRITERION_KEYS)
    assert all(first["lock_attestation"]["criteria"].values())
    validate_lock_attestation(first["lock_attestation"])


def test_bundle_rejects_one_run_fallback(tmp_path: Path) -> None:
    manifest, replay_path, _ = _fixture(tmp_path)
    replay = json.loads(replay_path.read_text())
    replay["cases"][0]["runs"] = replay["cases"][0]["runs"][:1]
    replay["cases"][0]["replay_probe_sha256s"] = replay["cases"][0]["replay_probe_sha256s"][:1]
    replay["cases"][0]["replay_count"] = 1
    replay_path.write_bytes(canonical_replay_attestation_bytes(replay))
    with pytest.raises(ValueError, match="exactly three|equal three"):
        build_local_plane_fragment_probe_bundle(
            probe_manifest_path=manifest, replay_attestation_path=replay_path
        )


@pytest.mark.parametrize("criterion", LOCK_CRITERION_KEYS)
def test_ready_for_gold_requires_every_lock_criterion(criterion: str) -> None:
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    criteria[criterion] = False
    record = build_lock_attestation(criteria)
    assert record["ready_for_gold"] is False
    validate_lock_attestation(record)


def test_lock_attestation_rejects_mutation() -> None:
    record = build_lock_attestation({key: True for key in LOCK_CRITERION_KEYS})
    record["criteria"][LOCK_CRITERION_KEYS[0]] = False
    with pytest.raises(ValueError, match="ready_for_gold|hash mismatch"):
        validate_lock_attestation(record)


def test_projection_is_exact_deep_copy_and_cli_remains_four_inputs() -> None:
    raw = _pool("C001")
    projected = project_lpf_input(raw)
    assert tuple(projected) == ALLOWED_POOL_PROJECTION_FIELDS
    assert "excluded_case_token" not in projected
    projected["support_parent_table"].append({"mutated": True})
    assert raw["support_parent_table"] == []
    options = {
        option
        for action in diagnose_parser()._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {"--scan", "--candidate-pool", "--a0-prediction", "--output"}


def test_projection_missing_field_fails_closed() -> None:
    raw = _pool("C001")
    raw.pop("transition_atom_caps")
    with pytest.raises(ValueError, match="missing fields"):
        project_lpf_input(raw)
