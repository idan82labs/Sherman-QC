from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
INTERFACE_PATH = ROOT / "scripts" / "exp_cad_0031_source_interface.py"
INTERFACE_SPEC = importlib.util.spec_from_file_location("exp31_interface", INTERFACE_PATH)
INTERFACE = importlib.util.module_from_spec(INTERFACE_SPEC)
assert INTERFACE_SPEC and INTERFACE_SPEC.loader
INTERFACE_SPEC.loader.exec_module(INTERFACE)

SCIENCE_PATH = ROOT / "scripts" / "exp_cad_0029_cfpdg_source.py"
SCIENCE_SPEC = importlib.util.spec_from_file_location("exp31_parent_science", SCIENCE_PATH)
SCIENCE = importlib.util.module_from_spec(SCIENCE_SPEC)
assert SCIENCE_SPEC and SCIENCE_SPEC.loader
SCIENCE_SPEC.loader.exec_module(SCIENCE)

RUNNER_PATH = ROOT / "scripts" / "run_exp_cad_0031_cfpdg_source.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("exp31_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(RUNNER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_rows(count: int = 128) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    points = np.column_stack((0.17 * index, np.sin(index), np.cos(index)))
    normals = np.column_stack(
        (
            0.2 + 0.001 * index,
            0.4 - 0.0005 * index,
            np.ones(count, dtype=np.float64),
        )
    )
    return points, normals


def _filter(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    ids: np.ndarray | None = None,
) -> dict:
    return INTERFACE.filter_source_rows(
        points,
        normals,
        member_token="OPAQUE_MEMBER",
        original_row_ids=ids,
    )


def test_zero_normal_tail_is_removed_without_changing_clean_prefix() -> None:
    points, normals = _clean_rows()
    extended_points = np.vstack((points, [[999.0, 0.0, 0.0]] * 3))
    extended_normals = np.vstack((normals, np.zeros((3, 3))))
    result = _filter(extended_points, extended_normals)
    assert np.array_equal(result["points"], points)
    assert np.array_equal(result["normals"], normals)
    assert np.array_equal(result["original_row_ids"], np.arange(len(points)))
    report = result["report"]
    assert report["exclusive_rejection_counts"] == {
        "NONFINITE_POINT": 0,
        "NONFINITE_NORMAL": 0,
        "NORMAL_NORM_LE_1E_12": 3,
    }
    assert report["rejected_original_row_ids_sorted"] == [128, 129, 130]
    assert report["actionable"] is False
    assert report["replacement_interpolation_or_normal_estimation_performed"] is False


def test_clean_data_is_identity_and_byte_unchanged() -> None:
    points, normals = _clean_rows(1024)
    result = _filter(points, normals)
    assert result["points"].tobytes() == np.ascontiguousarray(points).tobytes()
    assert result["normals"].tobytes() == np.ascontiguousarray(normals).tobytes()
    assert np.array_equal(result["original_row_ids"], np.arange(len(points)))
    assert result["report"]["rejected_row_count"] == 0
    assert result["report"]["rejected_original_row_ids_sorted"] == []


def test_nonfinite_and_zero_reason_precedence_is_exact_and_exhaustive() -> None:
    points = np.asarray(
        [[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    normals = np.asarray(
        [[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    result = _filter(points, normals)
    report = result["report"]
    assert report["raw_predicate_counts"] == {
        "NONFINITE_POINT": 1,
        "NONFINITE_NORMAL": 2,
        "NORMAL_NORM_LE_1E_12": 1,
    }
    assert report["exclusive_rejection_counts"] == {
        "NONFINITE_POINT": 1,
        "NONFINITE_NORMAL": 1,
        "NORMAL_NORM_LE_1E_12": 1,
    }
    assert report["rejected_original_row_ids_sorted"] == [0, 1, 2]
    assert np.array_equal(result["original_row_ids"], [3])


def test_parent_norm_overflow_is_rejected_and_every_retained_normal_canonicalizes() -> None:
    points = np.zeros((4, 3), dtype=np.float64)
    normals = np.asarray(
        [
            [1e200, 1e200, 1e200],
            [0.0, 0.0, 1.0],
            [1e-6, -2e-6, 3e-6],
            [-0.2, 0.7, 0.5],
        ],
        dtype=np.float64,
    )
    result = _filter(points, normals)
    assert result["report"]["exclusive_rejection_counts"] == {
        "NONFINITE_POINT": 0,
        "NONFINITE_NORMAL": 1,
        "NORMAL_NORM_LE_1E_12": 0,
    }
    assert result["report"]["rejected_original_row_ids_sorted"] == [0]
    canonical = SCIENCE.canonical_unoriented_normals(result["normals"])
    assert canonical.shape == (3, 3)
    assert np.all(np.isfinite(canonical))
    assert np.allclose(np.linalg.norm(canonical, axis=1), 1.0)


def test_normal_threshold_and_actionability_boundaries_are_inclusive() -> None:
    just_above = np.nextafter(1e-12, np.inf)
    result = _filter(
        np.zeros((3, 3)),
        np.asarray([[1e-12, 0.0, 0.0], [just_above, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert result["report"]["rejected_original_row_ids_sorted"] == [0]

    exact_one_percent = INTERFACE.evaluate_actionability(
        original_count=10_000_000, retained_count=9_900_000,
    )
    assert exact_one_percent["actionable"] is True
    over_one_percent = INTERFACE.evaluate_actionability(
        original_count=10_000_000, retained_count=9_899_999,
    )
    assert over_one_percent["actionable"] is False
    assert over_one_percent["actionability_reasons"] == [
        "INVALID_FRACTION_EXCEEDS_0.01"
    ]
    count_boundary = INTERFACE.evaluate_actionability(
        original_count=100_000, retained_count=100_000,
    )
    assert count_boundary["actionable"] is True
    below_count = INTERFACE.evaluate_actionability(
        original_count=99_999, retained_count=99_999,
    )
    assert below_count["actionable"] is False
    assert below_count["actionability_reasons"] == [
        "VALID_ROW_COUNT_BELOW_100000"
    ]


def test_filter_is_permutation_equivariant_when_original_identities_travel() -> None:
    points, normals = _clean_rows(64)
    normals[[7, 31]] = 0.0
    ids = np.arange(len(points), dtype=np.uint64)
    first = _filter(points, normals, ids=ids)
    permutation = np.random.default_rng(20260830).permutation(len(points))
    second = _filter(
        points[permutation], normals[permutation], ids=ids[permutation]
    )
    first_order = np.argsort(first["original_row_ids"], kind="mergesort")
    second_order = np.argsort(second["original_row_ids"], kind="mergesort")
    assert np.array_equal(
        first["original_row_ids"][first_order],
        second["original_row_ids"][second_order],
    )
    assert np.array_equal(first["points"][first_order], second["points"][second_order])
    assert np.array_equal(first["normals"][first_order], second["normals"][second_order])
    for key in (
        "original_identity_set_sha256", "retained_identity_set_sha256",
        "rejected_identity_set_sha256",
        "rejected_identity_sha256_by_exclusive_reason",
    ):
        assert first["report"][key] == second["report"][key]


def test_identity_and_spatial_partitions_match_clean_rows_by_original_identity() -> None:
    points, normals = _clean_rows(256)
    invalid = np.asarray([5, 19, 101, 211])
    corrupted = normals.copy()
    corrupted[invalid] = 0.0
    result = _filter(points, corrupted)
    keep = np.ones(len(points), dtype=bool)
    keep[invalid] = False
    expected_points = points[keep]
    expected_normals = normals[keep]
    assert np.array_equal(result["points"], expected_points)
    assert np.array_equal(result["normals"], expected_normals)
    assert np.array_equal(result["original_row_ids"], np.flatnonzero(keep))
    assert np.array_equal(
        SCIENCE.canonical_identity_bits(result["points"], result["normals"]),
        SCIENCE.canonical_identity_bits(expected_points, expected_normals),
    )
    assert np.array_equal(
        SCIENCE.spatial_checker_bits(result["points"], 1.5),
        SCIENCE.spatial_checker_bits(expected_points, 1.5),
    )


def test_filter_replay_is_byte_exact_and_does_not_mutate_inputs() -> None:
    points, normals = _clean_rows(512)
    normals[-4:] = 0.0
    original_points = points.copy()
    original_normals = normals.copy()
    first = _filter(points, normals)
    second = _filter(points, normals)
    assert np.array_equal(points, original_points)
    assert np.array_equal(normals, original_normals)
    for key in ("points", "normals", "original_row_ids"):
        assert first[key].tobytes() == second[key].tobytes()
    assert json.dumps(first["report"], sort_keys=True, separators=(",", ":")) == json.dumps(
        second["report"], sort_keys=True, separators=(",", ":")
    )


def test_pair_gate_requires_both_members_without_using_identity_as_feature() -> None:
    first = {"report": {"member_token": "OPAQUE_MEMBER", "actionable": True}}
    second = {"report": {"member_token": "SECOND", "actionable": False}}
    pair = INTERFACE.pair_actionability(first, second)
    assert pair == {
        "actionable": False,
        "failed_member_tokens": ["SECOND"],
        "status": "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN",
    }


@pytest.mark.parametrize(
    "points,normals,ids",
    [
        (np.zeros(3), np.zeros((1, 3)), None),
        (np.zeros((2, 3)), np.zeros((1, 3)), None),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.asarray([0])),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.asarray([1, 1])),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.asarray([-1, 1])),
    ],
)
def test_malformed_shape_or_identity_fails_closed(points, normals, ids) -> None:
    with pytest.raises(ValueError):
        _filter(points, normals, ids=ids)


def test_exp30_terminal_parent_is_unchanged_and_hash_bound() -> None:
    prereg = json.loads(
        (ROOT / "docs/research/2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json").read_text()
    )
    lineage = prereg["lineage"]
    assert _sha(ROOT / lineage["terminal_parent_result_path"]) == lineage[
        "terminal_parent_result_sha256"
    ]
    assert _sha(ROOT / lineage["terminal_parent_report_path"]) == lineage[
        "terminal_parent_report_sha256"
    ]
    assert _sha(ROOT / lineage["terminal_parent_orchestration_path"]) == lineage[
        "terminal_parent_orchestration_sha256"
    ]
    assert _sha(ROOT / lineage["scientific_parent_preregistration_path"]) == lineage[
        "scientific_parent_preregistration_sha256"
    ]
    assert _sha(ROOT / lineage["scientific_semantics_module_path"]) == lineage[
        "scientific_semantics_module_sha256"
    ]
    assert _sha(ROOT / lineage["scientific_execute_source_parent_path"]) == lineage[
        "scientific_execute_source_parent_sha256_at_preregistration"
    ]
    terminal = json.loads((ROOT / lineage["terminal_parent_result_path"]).read_text())
    assert terminal["status"] == "COMPLETE_TERMINAL_SOURCE_INTERFACE_FAILURE_NO_RERUN"
    assert terminal["artifact_reachability"]["native_visual_status"] == "NOT_REACHED"
    assert terminal["execution_accounting"]["physical_scan_byte_open_operation_count"] == 4


def _preregs() -> tuple[dict, dict]:
    v4 = json.loads(
        (ROOT / "docs/research/2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json").read_text()
    )
    parent = json.loads(
        (ROOT / v4["lineage"]["scientific_parent_preregistration_path"]).read_text()
    )
    return v4, parent


def _runner_arguments(points: np.ndarray, normals: np.ndarray) -> dict:
    v4, parent = _preregs()
    return {
        "v4_prereg": v4,
        "parent_prereg": parent,
        "cad_preflight": {},
        "nodes": np.empty(0),
        "edges": np.empty((0, 2), dtype=np.int64),
        "edge_weights": np.empty(0),
        "mesh": object(),
        "raw_first": points,
        "normals_first": normals,
        "raw_second": points.copy(),
        "normals_second": normals.copy(),
        "transform_first": np.eye(4),
        "transform_second": np.eye(4),
        "first_token": "N009",
        "second_token": "N010",
    }


def test_runner_filters_before_unchanged_science_and_matches_manual_prefilter(monkeypatch) -> None:
    count = 100_001
    index = np.arange(count, dtype=np.float64)
    points = np.column_stack((index * 0.001, np.zeros(count), np.ones(count)))
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    normals[-1] = 0.0
    manually_filtered = INTERFACE.filter_source_rows(
        points, normals, member_token="N009",
        original_row_ids=np.arange(count, dtype=np.uint64),
    )
    captured = {}
    parent_semantic = {
        "schema": "EXP_CAD_0030_CFPDG_SOURCE_RESULT_V1",
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "status": "ABSTAIN",
        "reason": "SYNTHETIC_PARENT",
        "numeric": {"unchanged": [1.0, 2.0, 3.0]},
    }
    parent_visual = {
        "schema": "EXP_CAD_0030_CFPDG_VISUAL_PAYLOAD_V1",
        "global": {"status": "SYNTHETIC_PARENT"},
        "numeric": {"unchanged": [4.0, 5.0]},
    }

    def fake_parent_execute_source(**kwargs):
        captured.update(kwargs)
        return parent_semantic, parent_visual

    monkeypatch.setattr(RUNNER.PARENT, "execute_source", fake_parent_execute_source)
    semantic, visual, filtered = RUNNER.execute_source_v4(
        **_runner_arguments(points, normals)
    )
    assert np.array_equal(captured["raw_first"], manually_filtered["points"])
    assert np.array_equal(captured["normals_first"], manually_filtered["normals"])
    assert np.array_equal(captured["raw_second"], manually_filtered["points"])
    assert np.array_equal(captured["normals_second"], manually_filtered["normals"])
    assert np.array_equal(
        filtered["first_original_row_ids"], np.arange(100_000, dtype=np.uint64)
    )
    assert np.array_equal(
        filtered["second_original_row_ids"], np.arange(100_000, dtype=np.uint64)
    )
    assert semantic["schema"] == "EXP_CAD_0031_SOURCE_INTERFACE_EXECUTION_ENVELOPE_V1"
    assert semantic["scientific_execute_source_reached"] is True
    assert semantic["scientific_core"]["semantic_payload"] == parent_semantic
    assert semantic["scientific_core"]["semantic_payload_sha256"] == RUNNER._canonical_payload_sha256(
        parent_semantic
    )
    assert semantic["scientific_core"]["visual_payload_sha256"] == RUNNER._canonical_payload_sha256(
        parent_visual
    )
    assert visual is parent_visual
    assert visual == parent_visual
    assert semantic["visual_payload_identity"]["silent_relabeling_performed"] is False


def test_runner_persists_both_reports_and_stops_before_science_when_nonactionable(
    monkeypatch,
) -> None:
    points, normals = _clean_rows(32)

    def forbidden_parent(**kwargs):
        raise AssertionError("scientific execute_source must not run before pair gate")

    monkeypatch.setattr(RUNNER.PARENT, "execute_source", forbidden_parent)
    semantic, visual, filtered = RUNNER.execute_source_v4(
        **_runner_arguments(points, normals)
    )
    assert visual is None
    assert semantic["status"] == "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
    assert semantic["scientific_execute_source_reached"] is False
    assert set(semantic["source_interface"]) == {"first", "second", "pair"}
    assert semantic["source_interface"]["first"]["original_row_count"] == 32
    assert semantic["source_interface"]["second"]["original_row_count"] == 32
    assert np.array_equal(filtered["first_original_row_ids"], np.arange(32))
    assert np.array_equal(filtered["second_original_row_ids"], np.arange(32))


def test_runner_thresholds_are_frozen_not_caller_tunable() -> None:
    v4, parent = _preregs()
    RUNNER._validate_preregistration_contracts(v4, parent)
    for key, value in (
        ("maximum_invalid_fraction_per_member", 0.0100001),
        ("minimum_valid_row_count_per_member", 99_999),
        ("maximum_invalid_fraction_inclusive", False),
    ):
        mutated = json.loads(json.dumps(v4))
        mutated["source_interface_actionability"][key] = value
        with pytest.raises(ValueError):
            RUNNER._validate_preregistration_contracts(mutated, parent)
    assert "maximum_invalid_fraction" not in RUNNER.execute_source_v4.__annotations__
    assert "minimum_valid_row_count" not in RUNNER.execute_source_v4.__annotations__
    summary = json.loads(RUNNER.source_free_contract_summary())
    assert summary["thresholds_caller_tunable"] is False


@pytest.mark.parametrize(
    "first_token,second_token",
    [("N010", "N009"), ("WRONG", "N010"), ("N009", "WRONG")],
)
def test_runner_rejects_swapped_or_wrong_opaque_token_order(
    first_token: str, second_token: str
) -> None:
    points, normals = _clean_rows(32)
    arguments = _runner_arguments(points, normals)
    arguments["first_token"] = first_token
    arguments["second_token"] = second_token
    with pytest.raises(ValueError, match="token/order"):
        RUNNER.execute_source_v4(**arguments)


def test_exp31_approval_headers_and_namespaces_are_exact() -> None:
    digest = "a" * 64
    cad = {
        "schema": RUNNER.CAD_REUSE_APPROVAL_SCHEMA,
        "experiment_id": RUNNER.EXPECTED_EXPERIMENT,
        "architecture": RUNNER.EXPECTED_ARCHITECTURE,
        "decision": "PASS_REUSED_EXP30_CAD_FREEZE_FOR_EXP31_SOURCE_INTERFACE",
        "physical_source_execution_authorized": False,
        "bound_files": {
            key: {"sha256": digest} for key in RUNNER.CAD_REUSE_BOUND_KEYS
        },
        "bound_cad_artifacts": {
            key: digest for key in RUNNER.CAD_ARTIFACT_KEYS
        },
    }
    RUNNER._validate_approval_header(cad, source=False)
    source = {
        "schema": RUNNER.SOURCE_APPROVAL_SCHEMA,
        "experiment_id": RUNNER.EXPECTED_EXPERIMENT,
        "architecture": RUNNER.EXPECTED_ARCHITECTURE,
        "decision": "APPROVE_ONE_EXP31_PHYSICAL_PRIMARY_REPLAY_EXECUTION",
        "physical_source_execution_authorized": True,
        "bound_files": {
            key: {"sha256": digest} for key in RUNNER.SOURCE_BOUND_KEYS
        },
    }
    RUNNER._validate_approval_header(source, source=True)
    for mutation in (
        source | {"schema": "EXP_CAD_0030_V3_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1"},
        source | {"physical_source_execution_authorized": False},
        source | {
            "bound_files": dict(source["bound_files"])
            | {"unexpected": {"sha256": digest}}
        },
        source | {"bound_cad_artifacts": {}},
    ):
        with pytest.raises(ValueError):
            RUNNER._validate_approval_header(mutation, source=True)


def test_source_cli_orders_full_approval_before_first_scan_byte_and_has_no_tunable_filter_cli() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    main_start = source.index("def main(")
    approval = source.index("verify_pre_scan_approvals(", main_start)
    transforms = source.index("transform_first = PARENT.load_transform(", main_start)
    cad_load = source.index(
        "nodes, edges, weights, mesh = PARENT.load_cad_source_artifacts_before_scan(",
        main_start,
    )
    first_scan_hash = source.index("PARENT._require_hash(", cad_load)
    first_cloud_load = source.index("o3d.io.read_point_cloud", first_scan_hash)
    interface = source.index("execute_source_v4(", first_cloud_load)
    assert approval < transforms < cad_load < first_scan_hash < first_cloud_load < interface
    for forbidden in (
        "--maximum-invalid-fraction", "--minimum-valid-row-count",
        "--evaluator", "--role", "--label", "--gold", "--expected", "--quality",
    ):
        assert forbidden not in source
    assert "np.arange(len(first_points), dtype=np.uint64)" in source
    assert "np.arange(len(second_points), dtype=np.uint64)" in source


def _main_argv(tmp_path: Path) -> list[str]:
    names = (
        "preregistration", "scientific_preregistration", "terminal_parent_result",
        "terminal_parent_report", "terminal_parent_orchestration",
        "cad_reuse_audit_packet", "cad_preflight", "cad_reuse_approval",
        "source_runner_audit_packet", "source_runner_approval", "nodes", "edges", "edge_geodesic_mm", "mesh",
        "scan_first", "scan_second", "transform_first", "transform_second",
        "source_tests", "parent_renderer", "parent_renderer_tests", "orchestrator",
        "orchestrator_tests",
    )
    argv = []
    for name in names:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        argv.extend(("--" + name.replace("_", "-"), str(path)))
    argv.extend(("--output-root", str(tmp_path / "output")))
    return argv


class _Cloud:
    def __init__(self, *, points=None, normals=None, has_normals=True):
        self.points = np.asarray(
            [[0.0, 0.0, 0.0]] if points is None else points, dtype=np.float64
        )
        self.normals = np.asarray(
            [[0.0, 0.0, 1.0]] if normals is None else normals, dtype=np.float64
        )
        self._has_normals = has_normals

    def has_points(self):
        return len(self.points) > 0

    def has_normals(self):
        return self._has_normals


@pytest.mark.parametrize(
    "failure_kind,expected_stage,expected_loads,expected_opens",
    [
        ("FIRST_LOAD", "FIRST_POINT_CLOUD_LOAD", 1, 3),
        ("SECOND_LOAD", "SECOND_POINT_CLOUD_LOAD", 2, 4),
        ("MISSING_NORMALS", "SOURCE_NORMAL_PRESENCE_GATE", 2, 4),
        ("SHAPE", "SOURCE_ARRAY_SHAPE_GATE", 2, 4),
        ("INTERFACE", "SOURCE_INTERFACE_AND_PARENT_SCIENCE", 2, 4),
    ],
)
def test_every_post_byte_source_failure_persists_exact_accounting(
    monkeypatch, tmp_path: Path, failure_kind: str, expected_stage: str,
    expected_loads: int, expected_opens: int,
) -> None:
    v4, parent = _preregs()
    monkeypatch.setattr(
        RUNNER, "verify_pre_scan_approvals", lambda **kwargs: (v4, parent, {})
    )
    monkeypatch.setattr(RUNNER.PARENT, "load_transform", lambda *args, **kwargs: np.eye(4))
    monkeypatch.setattr(
        RUNNER.PARENT,
        "load_cad_source_artifacts_before_scan",
        lambda **kwargs: (np.empty(0), np.empty((0, 2)), np.empty(0), object()),
    )
    monkeypatch.setattr(RUNNER.PARENT, "_require_hash", lambda *args, **kwargs: None)
    load_count = 0

    def fake_read(path):
        nonlocal load_count
        load_count += 1
        if failure_kind == "FIRST_LOAD" and load_count == 1:
            raise RuntimeError("first load failure")
        if failure_kind == "SECOND_LOAD" and load_count == 2:
            raise RuntimeError("second load failure")
        if failure_kind == "MISSING_NORMALS":
            return _Cloud(has_normals=False)
        if failure_kind == "SHAPE":
            return _Cloud(points=np.zeros((2, 3)), normals=np.zeros((1, 3)))
        return _Cloud()

    monkeypatch.setattr(RUNNER.o3d.io, "read_point_cloud", fake_read)
    if failure_kind == "INTERFACE":
        monkeypatch.setattr(
            RUNNER, "execute_source_v4",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("interface failure")),
        )
    with pytest.raises((RuntimeError, ValueError)):
        RUNNER.main(_main_argv(tmp_path))
    accounting = json.loads(
        (tmp_path / "output/SOURCE_CHILD_FAILURE_ACCOUNTING.json").read_text()
    )
    assert accounting["failure_stage"] == expected_stage
    assert accounting["physical_scan_hash_call_count"] == 2
    assert accounting["physical_scan_point_cloud_load_call_count"] == expected_loads
    assert accounting["physical_scan_byte_open_operation_count"] == expected_opens


def test_source_approval_rejection_precedes_scan_hash_and_creates_no_output(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        RUNNER,
        "verify_pre_scan_approvals",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("approval rejected")),
    )
    monkeypatch.setattr(
        RUNNER.PARENT,
        "_require_hash",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scan hash must not be reached")
        ),
    )
    with pytest.raises(ValueError, match="approval rejected"):
        RUNNER.main(_main_argv(tmp_path))
    assert not (tmp_path / "output").exists()


def test_existing_regular_file_output_root_is_rejected_before_approval_or_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    argv = _main_argv(tmp_path)
    output = tmp_path / "output"
    output.write_bytes(b"not-a-directory")
    monkeypatch.setattr(
        RUNNER,
        "verify_pre_scan_approvals",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("output topology must fail before approval")
        ),
    )
    with pytest.raises(ValueError, match="empty directory"):
        RUNNER.main(argv)


def test_persistence_failure_retains_durable_full_scan_accounting_journal(
    monkeypatch, tmp_path: Path
) -> None:
    v4, parent = _preregs()
    monkeypatch.setattr(
        RUNNER, "verify_pre_scan_approvals", lambda **kwargs: (v4, parent, {})
    )
    monkeypatch.setattr(RUNNER.PARENT, "load_transform", lambda *args, **kwargs: np.eye(4))
    monkeypatch.setattr(
        RUNNER.PARENT,
        "load_cad_source_artifacts_before_scan",
        lambda **kwargs: (np.empty(0), np.empty((0, 2)), np.empty(0), object()),
    )
    monkeypatch.setattr(RUNNER.PARENT, "_require_hash", lambda *args, **kwargs: None)
    monkeypatch.setattr(RUNNER.o3d.io, "read_point_cloud", lambda path: _Cloud())
    monkeypatch.setattr(
        RUNNER,
        "execute_source_v4",
        lambda **kwargs: (
            {
                "schema": "EXP_CAD_0031_SOURCE_INTERFACE_RESULT_V1",
                "status": "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN",
                "source_interface": {"first": {}, "second": {}, "pair": {}},
                "scientific_execute_source_reached": False,
            },
            None,
            {
                "first_original_row_ids": np.asarray([0], dtype=np.uint64),
                "second_original_row_ids": np.asarray([0], dtype=np.uint64),
            },
        ),
    )
    monkeypatch.setattr(
        RUNNER, "_atomic_npy",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        RUNNER.main(_main_argv(tmp_path))
    journal = json.loads(
        (tmp_path / "output/SOURCE_SCAN_ACCOUNTING_JOURNAL.json").read_text()
    )
    assert journal["status"] == "FAIL_TERMINAL_NO_RERUN"
    assert journal["stage"] == "PERSIST_SOURCE_ARTIFACTS"
    assert journal["physical_scan_hash_call_count"] == 2
    assert journal["physical_scan_point_cloud_load_call_count"] == 2
    assert journal["physical_scan_byte_open_operation_count"] == 4


def test_full_exp31_pre_scan_verifier_binds_parent_cad_and_fresh_authority(
    tmp_path: Path,
) -> None:
    paths = {
        "v4_preregistration": ROOT / "docs/research/2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json",
        "parent_preregistration": ROOT / "docs/research/2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json",
        "terminal_parent_result": ROOT / "output/research/autoresearch-loop/EXP_CAD_0030_RESULT.json",
        "terminal_parent_report": ROOT / "docs/research/2026-08-30_exp_cad_0030_terminal_source_interface_failure.md",
        "terminal_parent_orchestration": ROOT / "output/research/autoresearch-loop/EXP_CAD_0030_ORCHESTRATION_RESULT.json",
        "cad_reuse_audit_packet": ROOT / "docs/research/2026-08-30_exp_cad_0031_v4_cad_reuse_audit_packet.json",
        "cad_preflight": ROOT / "output/research/autoresearch-loop/exp-cad-0030-cfpdg-v3-cad-preflight-lock3/CAD_ONLY_PREFLIGHT.json",
        "nodes": ROOT / "output/research/autoresearch-loop/exp-cad-0030-cfpdg-v3-cad-preflight-lock3/CAD_GRAPH_NODES.npy",
        "edges": ROOT / "output/research/autoresearch-loop/exp-cad-0030-cfpdg-v3-cad-preflight-lock3/CAD_GRAPH_EDGES.npy",
        "edge_geodesic_mm": ROOT / "output/research/autoresearch-loop/exp-cad-0030-cfpdg-v3-cad-preflight-lock3/CAD_GRAPH_EDGE_GEODESIC_MM.npy",
        "mesh": ROOT / "output/research/autoresearch-loop/exp-cad-0015-new-physical-intake-v1/cad_meshes/42637000_B.ply",
        "cad_core": ROOT / "scripts/exp_cad_0029_cfpdg_core.py",
        "cad_only_runner": ROOT / "scripts/run_exp_cad_0029_cad_only_preflight.py",
        "cad_only_wrapper": ROOT / "scripts/run_exp_cad_0030_cad_only_preflight.py",
        "cad_only_tests": ROOT / "tests/test_exp_cad_0029_cfpdg_preflight.py",
        "cad_identity_tests": ROOT / "tests/test_exp_cad_0030_cfpdg_cad_identity.py",
    }
    cad_approval_path = tmp_path / "cad_reuse_approval.json"
    cad_approval = {
        "schema": RUNNER.CAD_REUSE_APPROVAL_SCHEMA,
        "experiment_id": RUNNER.EXPECTED_EXPERIMENT,
        "architecture": RUNNER.EXPECTED_ARCHITECTURE,
        "decision": "PASS_REUSED_EXP30_CAD_FREEZE_FOR_EXP31_SOURCE_INTERFACE",
        "physical_source_execution_authorized": False,
        "bound_files": {
            key: {"sha256": _sha(paths[key])}
            for key in RUNNER.CAD_REUSE_BOUND_KEYS
        },
        "bound_cad_artifacts": {
            "CAD_ONLY_PREFLIGHT.json": _sha(paths["cad_preflight"]),
            "CAD_GRAPH_NODES.npy": _sha(paths["nodes"]),
            "CAD_GRAPH_EDGES.npy": _sha(paths["edges"]),
            "CAD_GRAPH_EDGE_GEODESIC_MM.npy": _sha(paths["edge_geodesic_mm"]),
        },
    }
    cad_approval_path.write_text(json.dumps(cad_approval), encoding="utf-8")
    source_audit_path = tmp_path / "source_runner_audit_packet.json"
    source_paths = {
        **paths,
        "cad_reuse_approval": cad_approval_path,
        "source_runner_audit_packet": source_audit_path,
        "interface_module": ROOT / "scripts/exp_cad_0031_source_interface.py",
        "parent_science_module": ROOT / "scripts/exp_cad_0029_cfpdg_source.py",
        "parent_execute_source": ROOT / "scripts/run_exp_cad_0029_cfpdg_source.py",
        "source_runner": ROOT / "scripts/run_exp_cad_0031_cfpdg_source.py",
        "source_tests": Path(__file__),
        "parent_renderer": ROOT / "scripts/render_exp_cad_0030_cfpdg_native.py",
        "parent_renderer_tests": ROOT / "tests/test_exp_cad_0030_cfpdg_native_renderer.py",
        "orchestrator": ROOT / "scripts/orchestrate_exp_cad_0031_cfpdg_primary_replay.py",
        "orchestrator_tests": ROOT / "tests/test_exp_cad_0031_cfpdg_orchestration.py",
    }
    source_audit_path.write_text(
        json.dumps(
            {
                "schema": "EXP_CAD_0031_V4_SOURCE_RUNNER_AUDIT_PACKET_V1",
                "experiment_id": RUNNER.EXPECTED_EXPERIMENT,
                "architecture": RUNNER.EXPECTED_ARCHITECTURE,
                "status": "PASS_SOURCE_FREE_AUDIT_EXECUTION_NOT_AUTHORIZED",
                "physical_source_execution_authorized": False,
                "bound_files": {
                    key: {"sha256": _sha(path)}
                    for key, path in source_paths.items()
                    if key != "source_runner_audit_packet"
                },
            }
        ),
        encoding="utf-8",
    )
    source_approval_path = tmp_path / "source_approval.json"
    source_approval = {
        "schema": RUNNER.SOURCE_APPROVAL_SCHEMA,
        "experiment_id": RUNNER.EXPECTED_EXPERIMENT,
        "architecture": RUNNER.EXPECTED_ARCHITECTURE,
        "decision": "APPROVE_ONE_EXP31_PHYSICAL_PRIMARY_REPLAY_EXECUTION",
        "physical_source_execution_authorized": True,
        "bound_files": {
            key: {"sha256": _sha(source_paths[key])}
            for key in RUNNER.SOURCE_BOUND_KEYS
        },
    }
    source_approval_path.write_text(json.dumps(source_approval), encoding="utf-8")
    v4, parent, cad = RUNNER.verify_pre_scan_approvals(
        v4_preregistration=paths["v4_preregistration"],
        parent_preregistration=paths["parent_preregistration"],
        terminal_parent_result=paths["terminal_parent_result"],
        terminal_parent_report=paths["terminal_parent_report"],
        terminal_parent_orchestration=paths["terminal_parent_orchestration"],
        cad_reuse_audit_packet=paths["cad_reuse_audit_packet"],
        cad_preflight=paths["cad_preflight"],
        cad_reuse_approval=cad_approval_path,
        source_runner_audit_packet=source_audit_path,
        source_approval=source_approval_path,
        nodes=paths["nodes"], edges=paths["edges"],
        edge_geodesic_mm=paths["edge_geodesic_mm"], mesh=paths["mesh"],
        source_tests=source_paths["source_tests"],
        parent_renderer=source_paths["parent_renderer"],
        parent_renderer_tests=source_paths["parent_renderer_tests"],
        orchestrator=source_paths["orchestrator"],
        orchestrator_tests=source_paths["orchestrator_tests"],
    )
    assert v4["experiment_id"] == "EXP_CAD_0031"
    assert parent["experiment_id"] == "EXP_CAD_0030"
    assert cad["preregistration_sha256"] == RUNNER.EXPECTED_PARENT_SHA256

    wrong = dict(source_approval) | {
        "schema": "EXP_CAD_0030_V3_INDEPENDENT_SOURCE_RUNNER_APPROVAL_V1"
    }
    source_approval_path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="approval schema"):
        RUNNER.verify_pre_scan_approvals(
            v4_preregistration=paths["v4_preregistration"],
            parent_preregistration=paths["parent_preregistration"],
            terminal_parent_result=paths["terminal_parent_result"],
            terminal_parent_report=paths["terminal_parent_report"],
            terminal_parent_orchestration=paths["terminal_parent_orchestration"],
            cad_reuse_audit_packet=paths["cad_reuse_audit_packet"],
            cad_preflight=paths["cad_preflight"],
            cad_reuse_approval=cad_approval_path,
            source_runner_audit_packet=source_audit_path,
            source_approval=source_approval_path,
            nodes=paths["nodes"], edges=paths["edges"],
            edge_geodesic_mm=paths["edge_geodesic_mm"], mesh=paths["mesh"],
            source_tests=source_paths["source_tests"],
            parent_renderer=source_paths["parent_renderer"],
            parent_renderer_tests=source_paths["parent_renderer_tests"],
            orchestrator=source_paths["orchestrator"],
            orchestrator_tests=source_paths["orchestrator_tests"],
        )
