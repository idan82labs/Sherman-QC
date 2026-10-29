from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import diagnose_f1_v2_complete_same_surface_edge_probe as diagnose_module
from scripts.diagnose_f1_v2_complete_same_surface_edge_probe import (
    ALLOWED_EDGE_PROBE_INPUT_FIELDS,
    PROBE_KIND,
    PROBE_SCHEMA_VERSION,
    PRIMITIVE_PROFILE,
    build_v2_complete_same_surface_edge_probe_artifact,
    build_parser as diagnose_parser,
    project_edge_probe_input,
)
from scripts.run_f1_v2_complete_same_surface_edge_probe_batch import (
    BATCH_KIND,
    BATCH_ROW_FIELDS,
    PEAK_RSS_LIMIT_BYTES,
)
from scripts.run_f1_v2_complete_same_surface_edge_probe_replays import (
    OUTSIDE_MAX_REASON,
    REPLAY_KIND,
    REPLAY_SCHEMA_VERSION,
    SELECTION_REASONS,
    _resource_record,
    canonical_replay_attestation_bytes,
    select_designated_cases,
    validate_replay_attestation,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _row(token: str, *, complete_count: int) -> dict:
    edge_hashes = {
        "complete_same_surface_edge_table_sha256": _sha(token + "-complete"),
        "effective_mode_edge_table_sha256": _sha(token + "-effective"),
    }
    result = {
        "case_token": token,
        "status": "SUCCESS",
        "replacement_probe_sha256": _sha(token + "-replacement"),
        "frozen_control_probe_sha256": _sha(token + "-control"),
        "scan_sha256": _sha(token + "-scan"),
        "candidate_pool_sha256": _sha(token + "-pool"),
        "a0_prediction_sha256": _sha(token + "-a0"),
        "input_table_hashes": {"parent_table_sha256": _sha(token + "-parent")},
        "replacement_edge_table_hashes": edge_hashes,
        "output_table_hashes": {"local_plane_pair_records_sha256": _sha(token + "-pair")},
        "input_contract_attestation_sha256": _sha(token + "-input-attestation"),
        "edge_semantics_attestation_sha256": _sha(token + "-edge-attestation"),
        "engineering_profile_sha256": _sha(token + "-engineering"),
        "inference_semantic_sha256": _sha(token + "-semantic"),
        "case_counters": {
            "inherited_working_point_count": 200_000,
            "input_mode_count": 2_465 if token == "C030" else 2_000,
            "input_same_surface_edge_count": 5_000,
            "input_cross_mode_edge_count": 1_301 if token == "C003" else 1_000,
            "complete_same_surface_edge_count": complete_count,
        },
        "wall_seconds": 1.0,
        "peak_rss_bytes": 120_000_000,
        "input_json_bytes": 10_000,
        "serialized_output_bytes": 20_000,
    }
    assert set(result) == BATCH_ROW_FIELDS
    return result


def _batch() -> dict:
    rows = []
    for index in range(1, 38):
        token = f"C{index:03d}"
        complete = 900 if token in {"C004", "C005"} else 100 + index
        rows.append(_row(token, complete_count=complete))
    return {
        "schema_version": 1,
        "kind": BATCH_KIND,
        "profile": PRIMITIVE_PROFILE,
        "case_count": 37,
        "successful_case_count": 37,
        "cases": rows,
    }


def _attestation(batch: dict) -> dict:
    rows = {row["case_token"]: row for row in batch["cases"]}
    selection = select_designated_cases(batch)
    cases = []
    for token in sorted(selection):
        ordinary = rows[token]
        runs = [
            _resource_record(ordinary, run_index=run_index)
            for run_index in (1, 2, 3)
        ]
        counters = ordinary["case_counters"]
        edge_hashes = ordinary["replacement_edge_table_hashes"]
        cases.append(
            {
                "case_token": token,
                "selection_reason": selection[token],
                "working_point_count": counters["inherited_working_point_count"],
                "transition_mode_count": counters["input_mode_count"],
                "transition_mode_edge_count": counters["input_same_surface_edge_count"]
                + counters["input_cross_mode_edge_count"],
                "complete_same_surface_edge_count": counters[
                    "complete_same_surface_edge_count"
                ],
                "batch_replacement_probe_sha256": ordinary[
                    "replacement_probe_sha256"
                ],
                "batch_frozen_control_probe_sha256": ordinary[
                    "frozen_control_probe_sha256"
                ],
                "batch_complete_same_surface_edge_table_sha256": edge_hashes[
                    "complete_same_surface_edge_table_sha256"
                ],
                "batch_effective_mode_edge_table_sha256": edge_hashes[
                    "effective_mode_edge_table_sha256"
                ],
                "batch_inference_semantic_sha256": ordinary[
                    "inference_semantic_sha256"
                ],
                "replay_count": 3,
                "replay_replacement_probe_sha256s": [
                    ordinary["replacement_probe_sha256"]
                ]
                * 3,
                "maximum_peak_rss_bytes": 120_000_000,
                "runs": runs,
            }
        )
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "kind": REPLAY_KIND,
        "profile": PRIMITIVE_PROFILE,
        "research_only": True,
        "inference_safe": True,
        "case_count": 4,
        "total_isolated_run_count": 12,
        "probe_manifest_sha256": _sha("batch"),
        "inference_manifest_sha256": _sha("inference"),
        "candidate_pool_manifest_sha256": _sha("pool-manifest"),
        "frozen_control_manifest_sha256": _sha("control-manifest"),
        "cases": cases,
    }


def test_selection_uses_lexicographically_smallest_outside_maximum() -> None:
    selected = select_designated_cases(_batch())
    assert selected == {**SELECTION_REASONS, "C004": OUTSIDE_MAX_REASON}


@pytest.mark.parametrize(
    "counter,value,error",
    [
        ("input_mode_count", 9_999, "C030"),
        ("input_same_surface_edge_count", 9_999, "C003"),
    ],
)
def test_frozen_designated_extrema_are_recomputed(
    counter: str, value: int, error: str
) -> None:
    batch = _batch()
    batch["cases"][1]["case_counters"][counter] = value
    with pytest.raises(ValueError, match=error):
        select_designated_cases(batch)


def test_exact_twelve_run_attestation_validates_and_is_canonical() -> None:
    batch = _batch()
    attestation = _attestation(batch)
    validate_replay_attestation(attestation, batch_manifest=batch)
    assert [case["case_token"] for case in attestation["cases"]] == [
        "C001",
        "C003",
        "C004",
        "C030",
    ]
    assert canonical_replay_attestation_bytes(attestation) == canonical_replay_attestation_bytes(
        deepcopy(attestation)
    )


def test_wrong_fourth_case_is_rejected() -> None:
    batch = _batch()
    attestation = _attestation(batch)
    fourth = next(case for case in attestation["cases"] if case["case_token"] == "C004")
    fourth["case_token"] = "C005"
    with pytest.raises(ValueError, match="exact four"):
        validate_replay_attestation(attestation, batch_manifest=batch)


def test_exact_three_runs_per_case_is_required() -> None:
    batch = _batch()
    attestation = _attestation(batch)
    case = attestation["cases"][0]
    case["runs"] = case["runs"][:2]
    case["replay_count"] = 2
    case["replay_replacement_probe_sha256s"] = case[
        "replay_replacement_probe_sha256s"
    ][:2]
    with pytest.raises(ValueError, match="exactly three"):
        validate_replay_attestation(attestation, batch_manifest=batch)


@pytest.mark.parametrize(
    "field,batch_field",
    [
        ("replacement_probe_sha256", "replacement_probe_sha256"),
        ("frozen_control_probe_sha256", "frozen_control_probe_sha256"),
        ("complete_same_surface_edge_table_sha256", None),
        ("effective_mode_edge_table_sha256", None),
        ("inference_semantic_sha256", "inference_semantic_sha256"),
    ],
)
def test_every_replay_identity_must_equal_ordinary_batch(
    field: str, batch_field: str | None
) -> None:
    batch = _batch()
    attestation = _attestation(batch)
    run = attestation["cases"][0]["runs"][0]
    run[field] = _sha("mutated-" + field)
    # Keep resource hashing internally consistent: the failure must be the
    # ordinary-batch identity check, not merely the resource digest.
    record = {key: value for key, value in run.items() if key != "resource_profile_sha256"}
    run["resource_profile_sha256"] = hashlib.sha256(
        __import__("json").dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="differs from ordinary batch|run replacement"):
        validate_replay_attestation(attestation, batch_manifest=batch)


def test_replay_rss_at_limit_is_rejected() -> None:
    batch = _batch()
    attestation = _attestation(batch)
    run = attestation["cases"][0]["runs"][0]
    run["peak_rss_bytes"] = PEAK_RSS_LIMIT_BYTES
    record = {key: value for key, value in run.items() if key != "resource_profile_sha256"}
    run["resource_profile_sha256"] = hashlib.sha256(
        __import__("json").dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    attestation["cases"][0]["maximum_peak_rss_bytes"] = PEAK_RSS_LIMIT_BYTES
    with pytest.raises(ValueError, match="16 GiB"):
        validate_replay_attestation(attestation, batch_manifest=batch)


@pytest.mark.parametrize("field", ["input_json_bytes", "serialized_output_bytes"])
def test_replay_resource_sizes_must_be_constant(field: str) -> None:
    batch = _batch()
    attestation = _attestation(batch)
    run = attestation["cases"][0]["runs"][0]
    run[field] += 1
    record = {key: value for key, value in run.items() if key != "resource_profile_sha256"}
    run["resource_profile_sha256"] = hashlib.sha256(
        __import__("json").dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match=field):
        validate_replay_attestation(attestation, batch_manifest=batch)


def test_v1_control_identity_cannot_satisfy_replacement_replay() -> None:
    batch = _batch()
    attestation = _attestation(batch)
    attestation["kind"] = "f1_local_plane_fragment_probe_replay_attestation_v2"
    with pytest.raises(ValueError, match="identity"):
        validate_replay_attestation(attestation, batch_manifest=batch)


def test_diagnostic_cli_and_projection_are_exact() -> None:
    raw = {field: [] for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS}
    raw["support_representation_profile"] = "TRANSITION_ATOM_V2"
    raw["source_point_serialization"] = False
    raw["excluded_truth"] = {"gold": True}
    projected = project_edge_probe_input(raw)
    assert tuple(projected) == ALLOWED_EDGE_PROBE_INPUT_FIELDS
    assert "excluded_truth" not in projected
    projected["support_parent_table"].append({"changed": True})
    assert raw["support_parent_table"] == []
    options = {
        option
        for action in diagnose_parser()._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {
        "--scan",
        "--candidate-pool",
        "--a0-prediction",
        "--expected-control-probe-sha256",
        "--output",
    }


def test_missing_projection_field_fails_closed_and_batch_has_no_replay_claim() -> None:
    raw = {field: [] for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS}
    raw.pop("plane_table")
    with pytest.raises(ValueError, match="missing fields"):
        project_edge_probe_input(raw)
    assert not any("replay" in field for field in BATCH_ROW_FIELDS)


def test_diagnostic_hashes_scan_and_a0_as_opaque_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = tmp_path / "scan.ply"
    pool = tmp_path / "pool.json"
    a0 = tmp_path / "a0.json"
    scan.write_bytes(b"not JSON scan bytes")
    a0.write_bytes(b"not JSON prediction bytes")
    raw = {field: [] for field in ALLOWED_EDGE_PROBE_INPUT_FIELDS}
    raw["support_representation_profile"] = "TRANSITION_ATOM_V2"
    raw["source_point_serialization"] = False
    pool.write_text(json.dumps(raw), encoding="utf-8")
    expected_control = hashlib.sha256(b"control-bytes").hexdigest()
    observed: dict[str, object] = {}
    replacement_call_count = 0

    def control_build(projected, **hashes):
        observed["control_projected"] = projected
        observed["control_hashes"] = hashes
        return {"control": True}

    def control_validate(_payload, *, projected_input=None):
        assert projected_input is not None

    def replacement_build(projected, **hashes):
        nonlocal replacement_call_count
        replacement_call_count += 1
        observed["replacement_projected"] = projected
        observed["replacement_hashes"] = hashes
        return {
            "schema_version": PROBE_SCHEMA_VERSION,
            "kind": PROBE_KIND,
            "profile": PRIMITIVE_PROFILE,
            "frozen_control_probe_sha256": hashes["frozen_control_probe_sha256"],
        }

    def replacement_validate(_payload, *, projected_input=None):
        assert projected_input is not None

    monkeypatch.setattr(
        diagnose_module,
        "_control_api",
        lambda: (control_build, lambda _payload: b"control-bytes", control_validate),
    )
    monkeypatch.setattr(
        diagnose_module,
        "_replacement_api",
        lambda: (
            replacement_build,
            lambda payload: json.dumps(payload, sort_keys=True).encode("ascii"),
            replacement_validate,
        ),
    )
    payload, encoded = build_v2_complete_same_surface_edge_probe_artifact(
        scan_path=scan,
        candidate_pool_path=pool,
        a0_prediction_path=a0,
        expected_control_probe_sha256=expected_control,
    )
    assert encoded
    assert payload["frozen_control_probe_sha256"] == expected_control
    assert observed["control_hashes"] == {
        "scan_sha256": hashlib.sha256(scan.read_bytes()).hexdigest(),
        "candidate_pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "a0_prediction_sha256": hashlib.sha256(a0.read_bytes()).hexdigest(),
    }
    assert observed["replacement_hashes"] == {
        **observed["control_hashes"],
        "frozen_control_probe_sha256": expected_control,
    }
    with pytest.raises(ValueError, match="control hash differs"):
        build_v2_complete_same_surface_edge_probe_artifact(
            scan_path=scan,
            candidate_pool_path=pool,
            a0_prediction_path=a0,
            expected_control_probe_sha256="0" * 64,
        )
    assert replacement_call_count == 1
