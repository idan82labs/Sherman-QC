from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping
import weakref

import pytest


# The candidate runner must cross its stdlib barrier before native imports.
R = importlib.import_module("scripts.run_exp_cad_0035_cfpdg_source")

import numpy as np
import open3d as o3d

from scripts import exp_cad_0031_source_interface as INTAKE
from scripts import exp_cad_0032_cfpdg_v5_engine as V5
from scripts import exp_cad_0034_execution_contract as G34
from scripts import exp_cad_0035_execution_contract as G35
from scripts import run_exp_cad_0034_cfpdg_source as LEGACY


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _source_free_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for module, governance in ((R, G35), (LEGACY, G34)):
        monkeypatch.setattr(module, "np", np)
        monkeypatch.setattr(module, "o3d", o3d)
        monkeypatch.setattr(module, "INTAKE", INTAKE)
        monkeypatch.setattr(module, "V5", V5)
        monkeypatch.setattr(module, "GOVERNANCE", governance)


@pytest.fixture
def preregistrations() -> tuple[Mapping[str, Any], ...]:
    names = (
        "2026-08-30_exp_cad_0035_sequential_member_intake_preregistration.json",
        "2026-08-30_exp_cad_0034_cfpdg_v5_physical_successor_preregistration.json",
        "2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json",
        "2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json",
        "2026-08-30_exp_cad_0032_cfpdg_v5_execution_equivalent_preregistration.json",
    )
    return tuple(
        json.loads((ROOT / "docs/research" / name).read_text(encoding="utf-8"))
        for name in names
    )


def _cloud(rows: int, *, boundary: bool = False) -> Any:
    points = np.empty((rows, 3), dtype=np.float64)
    points[:, 0] = np.arange(rows, dtype=np.float64)
    points[:, 1:] = 0.0
    normals = np.zeros((rows, 3), dtype=np.float64)
    normals[:, 2] = 1.0
    if boundary:
        points[0, 0] = np.nan
        normals[0, 0] = np.nan
        normals[1, 0] = np.inf
        normals[2] = 0.0
        normals[3] = [float.fromhex("0x1.19799812dea11p-40"), 0.0, 0.0]
        normals[4] = [float.fromhex("0x1.19799812dea12p-40"), 0.0, 0.0]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def _run_sequential(
    tmp_path: Path,
    *,
    rows: int = 16,
    boundary: bool = False,
    reader: Any | None = None,
    collector: Any | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {"N009": "9" * 64, "N010": "a" * 64}
    if reader is None:
        reader = lambda *_a, **_k: _cloud(rows, boundary=boundary)
    accounting = R._new_scan_accounting()
    lifecycle = R._new_lifecycle_accounting(mode="PRIMARY")
    first, second = R.perform_sequential_member_intake(
        mode="PRIMARY",
        scan_first=tmp_path / "N009.ply",
        scan_second=tmp_path / "N010.ply",
        journal_path=tmp_path / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json",
        lifecycle_path=tmp_path / "SOURCE_LIFECYCLE_MEASUREMENT.json",
        accounting=accounting,
        lifecycle=lifecycle,
        expected_hashes=expected,
        expected_row_counts={"N009": rows, "N010": rows},
        sha256_func=lambda path: expected[Path(path).stem],
        point_cloud_reader=reader,
        collector=collector,
    )
    return first, second, accounting, lifecycle


def _metrics() -> dict[str, Any]:
    counters = {
        "unique_member_fit_count": 38,
        "kd_node_query_count": 306_926,
        "local_fit_call_count": 306_926,
        "robustness_row_count": 32,
    }
    return {
        "schema": R.EXP32_METRICS_SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "architecture": V5.ARCHITECTURE,
        **counters,
        "operation_counters": counters,
        "full_fit_inventory_gate_passed": True,
        "full_fit_batched_kd_gate_passed": True,
        "full_fit_kd_query_gate_passed": True,
        "fit_or_robustness_row_skipped": False,
        "scientific_threshold_or_gate_changed": False,
    }


def _semantic() -> dict[str, Any]:
    return {
        "schema": R.PARENT_SEMANTIC_SCHEMA,
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "status": "ABSTAIN",
        "reason": "PAIR_SUPPORT_GATE",
    }


def _visual() -> dict[str, Any]:
    return {"schema": R.PARENT_VISUAL_SCHEMA, "experiment_id": "EXP_CAD_0030"}


def test_import_barrier_loads_no_candidate_numeric_native_or_contract_modules() -> None:
    code = """
import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(strict=True)
s=importlib.util.spec_from_file_location('exp35_barrier_probe',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
names=['numpy','open3d','scripts.exp_cad_0031_source_interface','scripts.exp_cad_0032_cfpdg_v5_engine','scripts.exp_cad_0035_execution_contract','scripts.run_exp_cad_0034_cfpdg_source']
print(json.dumps({name:(name in sys.modules) for name in names},sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(R.__file__).resolve())],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(json.loads(result.stdout).values()) == {False}


def test_fresh_identity_and_parser_include_lifecycle_contract() -> None:
    assert R.EXPERIMENT_ID == "EXP_CAD_0035"
    assert R.PREREGISTRATION_SHA256 == (
        "462274a567cd0ed84b041b8e0576f93ceb1ce6b4df9abe75e63c15ec1f268cb2"
    )
    assert R.ENGINE_ACCOUNTING_SCHEMA == LEGACY.ENGINE_ACCOUNTING_SCHEMA
    help_text = R.build_parser().format_help()
    for flag in (
        "--mode",
        "--scan-accounting-journal",
        "--lifecycle-measurement",
        "--source-launch-capability",
        "--source-launch-child-claim",
    ):
        assert flag in help_text


def test_preregistration_matches_new_shape_and_frozen_science(
    preregistrations: tuple[Mapping[str, Any], ...]
) -> None:
    exp35, _exp34, parent, exp31, exp32 = preregistrations
    R._validate_preregistrations(exp35, parent, exp31, exp32)


def test_both_hashes_then_one_cloud_at_a_time_with_exact_release_proof(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    cloud_refs: list[weakref.ReferenceType[Any]] = []
    journal = tmp_path / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    expected = {"N009": "9" * 64, "N010": "a" * 64}
    accounting = R._new_scan_accounting()
    lifecycle = R._new_lifecycle_accounting(mode="PRIMARY")

    def hasher(path: Path) -> str:
        payload = json.loads(journal.read_text(encoding="utf-8"))
        events.append((Path(path).stem, payload["stage"]))
        return expected[Path(path).stem]

    def reader(path: str, **kwargs: Any) -> Any:
        token = Path(path).stem
        payload = json.loads(journal.read_text(encoding="utf-8"))
        events.append((token, payload["stage"]))
        if token == "N010":
            assert cloud_refs[0]() is None
            assert payload["stage"] == "BEFORE_N010_POINT_CLOUD_LOAD"
        assert kwargs == {
            "format": "auto",
            "remove_nan_points": False,
            "remove_infinite_points": False,
            "print_progress": False,
        }
        value = _cloud(16, boundary=True)
        cloud_refs.append(weakref.ref(value))
        return value

    first, second = R.perform_sequential_member_intake(
        mode="PRIMARY",
        scan_first=tmp_path / "N009.ply",
        scan_second=tmp_path / "N010.ply",
        journal_path=journal,
        lifecycle_path=tmp_path / "SOURCE_LIFECYCLE_MEASUREMENT.json",
        accounting=accounting,
        lifecycle=lifecycle,
        expected_hashes=expected,
        expected_row_counts={"N009": 16, "N010": 16},
        sha256_func=hasher,
        point_cloud_reader=reader,
    )
    assert events == [
        ("N009", "BEFORE_N009_SHA256"),
        ("N010", "BEFORE_N010_SHA256"),
        ("N009", "BEFORE_N009_POINT_CLOUD_LOAD"),
        ("N010", "BEFORE_N010_POINT_CLOUD_LOAD"),
    ]
    assert [row["operation"] for row in accounting["attempted_operations"]] == [
        "SHA256",
        "SHA256",
        "POINT_CLOUD_LOAD",
        "POINT_CLOUD_LOAD",
    ]
    assert (
        accounting["physical_scan_hash_call_count"],
        accounting["physical_scan_point_cloud_load_call_count"],
        accounting["physical_scan_byte_open_operation_count"],
    ) == (2, 2, 4)
    assert lifecycle["_filter_source_rows_call_count_by_token"] == {
        "N009": 1,
        "N010": 1,
    }
    assert R._lifecycle_payload(lifecycle) == {
        "schema": R.LIFECYCLE_MEASUREMENT_SCHEMA,
        "experiment_id": "EXP_CAD_0035",
        "mode": "PRIMARY",
        "status": "IN_PROGRESS",
        "explicit_gc_collect_call_count": 2,
        "maximum_live_source_cloud_count": 1,
        "maximum_live_unfiltered_member_bundle_count": 1,
        "N009_release_verified": True,
        "N010_release_verified": True,
        "ordered_events": [
            "N009_RELEASE_VERIFIED_BEFORE_N010_LOAD",
            "N010_RELEASE_VERIFIED_BEFORE_PAIR_ACTIONABILITY",
        ],
    }
    for retained in (first, second):
        assert retained["points"].flags.owndata
        assert retained["normals"].flags.owndata
        assert retained["original_row_ids"].flags.owndata
        assert retained["points"].flags.c_contiguous


def test_explicit_collector_called_exactly_once_per_successful_member(
    tmp_path: Path,
) -> None:
    calls = 0

    def collector() -> int:
        nonlocal calls
        calls += 1
        return gc.collect()

    _first, _second, _accounting, lifecycle = _run_sequential(
        tmp_path, collector=collector
    )
    assert calls == 2
    assert lifecycle["explicit_gc_collect_call_count"] == 2


def test_filter_source_rows_is_instrumented_exactly_once_per_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    original = INTAKE.filter_source_rows

    def filter_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls.append(str(kwargs["member_token"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(INTAKE, "filter_source_rows", filter_spy)
    _first, _second, _accounting, lifecycle = _run_sequential(tmp_path)
    assert calls == ["N009", "N010"]
    assert lifecycle["_filter_source_rows_call_count_by_token"] == {
        "N009": 1,
        "N010": 1,
    }


@pytest.mark.parametrize(("failed", "hashes"), (("N009", 1), ("N010", 2)))
def test_hash_mismatch_is_terminal_before_any_cloud_load(
    tmp_path: Path, failed: str, hashes: int
) -> None:
    loads = 0
    accounting = R._new_scan_accounting()
    lifecycle = R._new_lifecycle_accounting(mode="PRIMARY")

    def reader(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal loads
        loads += 1
        raise AssertionError("load reached after hash failure")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        R.perform_sequential_member_intake(
            mode="PRIMARY",
            scan_first=tmp_path / "N009.ply",
            scan_second=tmp_path / "N010.ply",
            journal_path=tmp_path / "journal.json",
            lifecycle_path=tmp_path / "lifecycle.json",
            accounting=accounting,
            lifecycle=lifecycle,
            expected_hashes={"N009": "9" * 64, "N010": "a" * 64},
            expected_row_counts={"N009": 16, "N010": 16},
            sha256_func=lambda path: (
                "0" * 64
                if Path(path).stem == failed
                else {"N009": "9" * 64, "N010": "a" * 64}[Path(path).stem]
            ),
            point_cloud_reader=reader,
        )
    assert loads == 0
    assert accounting["physical_scan_hash_call_count"] == hashes


def test_failed_n009_release_proof_forbids_n010_load(tmp_path: Path) -> None:
    held: list[Any] = []
    loads: list[str] = []

    def reader(path: str, **_kwargs: Any) -> Any:
        token = Path(path).stem
        loads.append(token)
        value = _cloud(16)
        if token == "N009":
            held.append(value)
        return value

    with pytest.raises(RuntimeError, match="weak-reference release proof"):
        _run_sequential(tmp_path, reader=reader)
    assert loads == ["N009"]


@pytest.mark.parametrize("held_kind", ("raw_points", "raw_normals", "raw_ids"))
def test_held_n009_raw_array_forbids_release_and_n010_load(
    held_kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    held: list[Any] = []
    loads: list[str] = []
    original_materialize = R.materialize_loaded_cloud
    original_filter = INTAKE.filter_source_rows

    def reader(path: str, **_kwargs: Any) -> Any:
        loads.append(Path(path).stem)
        return _cloud(16)

    if held_kind in {"raw_points", "raw_normals"}:
        index = 0 if held_kind == "raw_points" else 1

        def materialize(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            arrays = original_materialize(*args, **kwargs)
            if kwargs["token"] == "N009":
                held.append(arrays[index])
            return arrays

        monkeypatch.setattr(R, "materialize_loaded_cloud", materialize)
    else:

        def filter_rows(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs["member_token"] == "N009":
                held.append(kwargs["original_row_ids"])
            return original_filter(*args, **kwargs)

        monkeypatch.setattr(INTAKE, "filter_source_rows", filter_rows)

    with pytest.raises(RuntimeError, match="weak-reference release proof"):
        _run_sequential(tmp_path, reader=reader)
    assert held
    assert loads == ["N009"]


def test_sequential_filtered_outputs_equal_exact_legacy_batch_oracle(
    preregistrations: tuple[Mapping[str, Any], ...], tmp_path: Path
) -> None:
    exp35, exp34, parent, exp31, exp32 = preregistrations
    sequential_first, sequential_second, _accounting, _lifecycle = _run_sequential(
        tmp_path / "sequential", rows=32, boundary=True
    )
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_accounting = LEGACY._new_scan_accounting()
    expected = {"N009": "9" * 64, "N010": "a" * 64}
    clouds = LEGACY.perform_ordered_scan_access(
        mode="PRIMARY",
        scan_first=legacy_root / "N009.ply",
        scan_second=legacy_root / "N010.ply",
        journal_path=legacy_root / "journal.json",
        accounting=legacy_accounting,
        expected_hashes=expected,
        sha256_func=lambda path: expected[Path(path).stem],
        point_cloud_reader=lambda *_a, **_k: _cloud(32, boundary=True),
    )
    raw_first, normals_first = LEGACY.materialize_loaded_cloud(
        clouds[0], token="N009", expected_row_count=32
    )
    raw_second, normals_second = LEGACY.materialize_loaded_cloud(
        clouds[1], token="N010", expected_row_count=32
    )
    legacy = LEGACY.execute_source_exp34(
        exp34_prereg=exp34,
        parent_prereg=parent,
        exp31_prereg=exp31,
        v5_prereg=exp32,
        cad_preflight={},
        nodes=np.zeros(LEGACY.EXPECTED_NODE_COUNT),
        edges=np.zeros((LEGACY.EXPECTED_EDGE_COUNT, 2), dtype=np.int64),
        edge_weights=np.ones(LEGACY.EXPECTED_EDGE_COUNT),
        mesh=object(),
        raw_first=raw_first,
        normals_first=normals_first,
        raw_second=raw_second,
        normals_second=normals_second,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="N009",
        second_token="N010",
    )
    sequential = R.execute_filtered_source_exp35(
        exp35_prereg=exp35,
        parent_prereg=parent,
        exp31_prereg=exp31,
        v5_prereg=exp32,
        cad_preflight={},
        nodes=np.zeros(R.EXPECTED_NODE_COUNT),
        edges=np.zeros((R.EXPECTED_EDGE_COUNT, 2), dtype=np.int64),
        edge_weights=np.ones(R.EXPECTED_EDGE_COUNT),
        mesh=object(),
        first=sequential_first,
        second=sequential_second,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="N009",
        second_token="N010",
    )
    assert sequential["source_interface"] == legacy["source_interface"]
    for report in (
        sequential["source_interface"]["first"],
        sequential["source_interface"]["second"],
    ):
        assert report["raw_predicate_counts"]["NONFINITE_POINT"] == 1
        assert report["raw_predicate_counts"]["NONFINITE_NORMAL"] == 2
        assert report["exclusive_rejection_counts"] == {
            "NONFINITE_POINT": 1,
            "NONFINITE_NORMAL": 1,
            "NORMAL_NORM_LE_1E_12": 2,
        }
        assert report["rejected_original_row_ids_sorted"] == [0, 1, 2, 3]
    assert sequential["retained_identity_hashes"] == legacy["retained_identity_hashes"]
    assert sequential["engine_accounting"] == legacy["engine_accounting"]
    for new, old in (
        (sequential_first, INTAKE.filter_source_rows(raw_first, normals_first, member_token="N009", original_row_ids=np.arange(32, dtype=np.uint64))),
        (sequential_second, INTAKE.filter_source_rows(raw_second, normals_second, member_token="N010", original_row_ids=np.arange(32, dtype=np.uint64))),
    ):
        for field in ("points", "normals", "original_row_ids"):
            assert new[field].dtype == old[field].dtype
            assert new[field].shape == old[field].shape
            assert new[field].tobytes() == old[field].tobytes()


def _array_differential(
    preregistrations: tuple[Mapping[str, Any], ...],
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    exp35, exp34, parent, exp31, exp32 = preregistrations
    common = {
        "parent_prereg": parent,
        "exp31_prereg": exp31,
        "v5_prereg": exp32,
        "cad_preflight": {},
        "nodes": np.zeros(R.EXPECTED_NODE_COUNT),
        "edges": np.zeros((R.EXPECTED_EDGE_COUNT, 2), dtype=np.int64),
        "edge_weights": np.ones(R.EXPECTED_EDGE_COUNT),
        "mesh": object(),
        "raw_first": points.copy(),
        "normals_first": normals.copy(),
        "raw_second": points.copy(),
        "normals_second": normals.copy(),
        "transform_first": np.eye(4),
        "transform_second": np.eye(4),
        "first_token": "N009",
        "second_token": "N010",
    }
    sequential = R.execute_source_exp35(exp35_prereg=exp35, **common)
    legacy = LEGACY.execute_source_exp34(exp34_prereg=exp34, **common)
    return sequential, legacy


def test_baseline_insertion_and_permutation_equal_legacy_on_same_input_with_remap(
    preregistrations: tuple[Mapping[str, Any], ...]
) -> None:
    points = np.column_stack(
        (np.arange(20, dtype=np.float64), np.zeros(20), np.ones(20))
    )
    normals = np.tile([0.0, 0.0, 1.0], (20, 1))
    points[3, 0] = np.nan
    normals[7] = 0.0
    baseline_sequential, baseline_legacy = _array_differential(
        preregistrations, points, normals
    )
    assert baseline_sequential == baseline_legacy
    baseline_filtered = INTAKE.filter_source_rows(
        points,
        normals,
        member_token="N009",
        original_row_ids=np.arange(len(points), dtype=np.uint64),
    )
    baseline_ids = baseline_filtered["original_row_ids"]

    inserted_points = np.insert(points, 5, [[123.0, 0.0, 1.0]], axis=0)
    inserted_normals = np.insert(normals, 5, [[0.0, 0.0, 1.0]], axis=0)
    inserted_sequential, inserted_legacy = _array_differential(
        preregistrations, inserted_points, inserted_normals
    )
    assert inserted_sequential == inserted_legacy
    inserted = INTAKE.filter_source_rows(
        inserted_points,
        inserted_normals,
        member_token="N009",
        original_row_ids=np.arange(len(inserted_points), dtype=np.uint64),
    )["original_row_ids"]
    remapped_insertion = np.asarray(
        [int(value) if value < 5 else int(value) - 1 for value in inserted if value != 5],
        dtype=np.uint64,
    )
    assert np.array_equal(remapped_insertion, baseline_ids)

    permutation = np.asarray([7, 1, 19, 3, 12, 0, 9, 4, 16, 2, 18, 5, 14, 6, 10, 8, 11, 13, 15, 17])
    permuted_sequential, permuted_legacy = _array_differential(
        preregistrations, points[permutation], normals[permutation]
    )
    assert permuted_sequential == permuted_legacy
    permuted_ids = INTAKE.filter_source_rows(
        points[permutation],
        normals[permutation],
        member_token="N009",
        original_row_ids=np.arange(len(permutation), dtype=np.uint64),
    )["original_row_ids"]
    remapped_permutation = permutation[permuted_ids.astype(np.int64)]
    assert np.array_equal(np.sort(remapped_permutation), np.sort(baseline_ids))


def test_pair_called_once_and_engine_zero_on_interface_abstention(
    preregistrations: tuple[Mapping[str, Any], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp35, _exp34, parent, exp31, exp32 = preregistrations
    calls = {"pair": 0, "engine": 0}
    pair = INTAKE.pair_actionability

    def pair_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls["pair"] += 1
        return pair(*args, **kwargs)

    def engine(**_kwargs: Any) -> Any:
        calls["engine"] += 1
        raise AssertionError("engine reached on interface abstention")

    monkeypatch.setattr(INTAKE, "pair_actionability", pair_spy)
    monkeypatch.setattr(V5, "execute_parent_science_v5", engine)
    filtered = INTAKE.filter_source_rows(
        np.zeros((8, 3)),
        np.tile([0.0, 0.0, 1.0], (8, 1)),
        member_token="N009",
        original_row_ids=np.arange(8, dtype=np.uint64),
    )
    second = dict(filtered)
    second["report"] = dict(filtered["report"], member_token="N010")
    result = R.execute_filtered_source_exp35(
        exp35_prereg=exp35,
        parent_prereg=parent,
        exp31_prereg=exp31,
        v5_prereg=exp32,
        cad_preflight={},
        nodes=np.zeros(R.EXPECTED_NODE_COUNT),
        edges=np.zeros((R.EXPECTED_EDGE_COUNT, 2), dtype=np.int64),
        edge_weights=np.ones(R.EXPECTED_EDGE_COUNT),
        mesh=object(),
        first=filtered,
        second=second,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="N009",
        second_token="N010",
    )
    assert calls == {"pair": 1, "engine": 0}
    assert result["scientific_execute_source_reached"] is False


def test_actionable_filtered_pair_calls_direct_unchanged_v5_once(
    preregistrations: tuple[Mapping[str, Any], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp35, _exp34, parent, exp31, exp32 = preregistrations
    calls = {"pair": 0, "engine": 0}
    points = np.zeros((100_000, 3), dtype=np.float64)
    normals = np.tile([0.0, 0.0, 1.0], (100_000, 1))
    first = INTAKE.filter_source_rows(
        points, normals, member_token="N009", original_row_ids=np.arange(100_000, dtype=np.uint64)
    )
    second = INTAKE.filter_source_rows(
        points, normals, member_token="N010", original_row_ids=np.arange(100_000, dtype=np.uint64)
    )
    original_pair = INTAKE.pair_actionability

    def pair_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls["pair"] += 1
        return original_pair(*args, **kwargs)

    def engine(**kwargs: Any) -> tuple[Mapping[str, Any], ...]:
        calls["engine"] += 1
        assert len(kwargs["raw_first"]) == len(kwargs["raw_second"]) == 100_000
        return _semantic(), _visual(), _metrics()

    monkeypatch.setattr(INTAKE, "pair_actionability", pair_spy)
    monkeypatch.setattr(V5, "execute_parent_science_v5", engine)
    monkeypatch.setattr(
        V5,
        "filter_source_rows_v5",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("forbidden V5 filter")),
    )
    result = R.execute_filtered_source_exp35(
        exp35_prereg=exp35,
        parent_prereg=parent,
        exp31_prereg=exp31,
        v5_prereg=exp32,
        cad_preflight={},
        nodes=np.zeros(R.EXPECTED_NODE_COUNT),
        edges=np.zeros((R.EXPECTED_EDGE_COUNT, 2), dtype=np.int64),
        edge_weights=np.ones(R.EXPECTED_EDGE_COUNT),
        mesh=object(),
        first=first,
        second=second,
        transform_first=np.eye(4),
        transform_second=np.eye(4),
        first_token="N009",
        second_token="N010",
    )
    assert calls == {"pair": 1, "engine": 1}
    assert result["scientific_execute_source_reached"] is True
    assert result["engine_accounting"]["v5_engine_invocation_count"] == 1


def test_true_actionable_science_payload_is_byte_identical_to_exact_exp34_entry(
    preregistrations: tuple[Mapping[str, Any], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp35, exp34, parent, exp31, exp32 = preregistrations
    assert hashlib.sha256(Path(LEGACY.__file__).read_bytes()).hexdigest() == (
        "098e907278dc9398ddfff8fafa468d6b675ab221c8729d8449c8d2807d1324b1"
    )
    points = np.column_stack(
        (
            np.arange(100_000, dtype=np.float64),
            np.zeros(100_000, dtype=np.float64),
            np.ones(100_000, dtype=np.float64),
        )
    )
    normals = np.tile([0.0, 0.0, 1.0], (100_000, 1))
    first = INTAKE.filter_source_rows(
        points,
        normals,
        member_token="N009",
        original_row_ids=np.arange(100_000, dtype=np.uint64),
    )
    second = INTAKE.filter_source_rows(
        points,
        normals,
        member_token="N010",
        original_row_ids=np.arange(100_000, dtype=np.uint64),
    )
    engine_calls: list[tuple[str, str, int, int]] = []

    def engine(**kwargs: Any) -> tuple[Mapping[str, Any], ...]:
        engine_calls.append(
            (
                str(kwargs["first_token"]),
                str(kwargs["second_token"]),
                len(kwargs["raw_first"]),
                len(kwargs["raw_second"]),
            )
        )
        semantic = {
            **_semantic(),
            "nested": {"actionable_science_fixture": True, "rows": 100_000},
        }
        visual = {
            **_visual(),
            "nested": {"segments": [[0.0, 1.0], [2.0, 3.0]]},
        }
        metrics = {**_metrics(), "nested": {"oracle": "EXACT_SHARED_V5"}}
        return semantic, visual, metrics

    monkeypatch.setattr(V5, "execute_parent_science_v5", engine)
    common = {
        "parent_prereg": parent,
        "exp31_prereg": exp31,
        "v5_prereg": exp32,
        "cad_preflight": {},
        "nodes": np.zeros(R.EXPECTED_NODE_COUNT),
        "edges": np.zeros((R.EXPECTED_EDGE_COUNT, 2), dtype=np.int64),
        "edge_weights": np.ones(R.EXPECTED_EDGE_COUNT),
        "mesh": object(),
        "transform_first": np.eye(4),
        "transform_second": np.eye(4),
        "first_token": "N009",
        "second_token": "N010",
    }
    sequential = R.execute_filtered_source_exp35(
        exp35_prereg=exp35,
        first=first,
        second=second,
        **common,
    )
    legacy = LEGACY.execute_source_exp34(
        exp34_prereg=exp34,
        raw_first=points,
        normals_first=normals,
        raw_second=points,
        normals_second=normals,
        **common,
    )
    assert engine_calls == [
        ("N009", "N010", 100_000, 100_000),
        ("N009", "N010", 100_000, 100_000),
    ]
    assert sequential["source_interface"]["pair"]["actionable"] is True
    assert sequential["scientific_execute_source_reached"] is True
    for field in (
        "source_interface",
        "retained_identity_hashes",
        "semantic_payload",
        "visual_payload",
        "engine_metrics",
        "engine_accounting",
    ):
        assert R._strict_canonical_bytes(sequential[field]) == R._strict_canonical_bytes(
            legacy[field]
        )


def test_lifecycle_publication_has_exact_field_inventory(tmp_path: Path) -> None:
    lifecycle = R._new_lifecycle_accounting(mode="PRIMARY")
    lifecycle.update(
        {
            "status": "COMPLETE_RELEASE_VERIFIED",
            "explicit_gc_collect_call_count": 2,
            "maximum_live_source_cloud_count": 1,
            "maximum_live_unfiltered_member_bundle_count": 1,
            "N009_release_verified": True,
            "N010_release_verified": True,
            "ordered_events": [
                "N009_RELEASE_VERIFIED_BEFORE_N010_LOAD",
                "N010_RELEASE_VERIFIED_BEFORE_PAIR_ACTIONABILITY",
            ],
        }
    )
    path = tmp_path / "SOURCE_LIFECYCLE_MEASUREMENT.json"
    R.write_lifecycle_measurement(path, lifecycle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "experiment_id",
        "mode",
        "status",
        "explicit_gc_collect_call_count",
        "maximum_live_source_cloud_count",
        "maximum_live_unfiltered_member_bundle_count",
        "N009_release_verified",
        "N010_release_verified",
        "ordered_events",
    }
    assert not any(key.startswith("_") for key in payload)


def _strict_write(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _capability_case(
    tmp_path: Path, *, mode: str = "PRIMARY"
) -> tuple[argparse.Namespace, dict[str, Any], list[str]]:
    attempt = tmp_path / "attempt"
    primary_measurement = attempt / "primary/measurement"
    replay_measurement = attempt / "replay/measurement"
    authorized = {
        "result_namespace": str(attempt),
        "primary_source": str(attempt / "primary/source"),
        "replay_source": str(attempt / "replay/source"),
        "primary_measurement": str(primary_measurement),
        "replay_measurement": str(replay_measurement),
        "primary_renderer": str(attempt / "primary/renderer"),
        "replay_renderer": str(attempt / "replay/renderer"),
        "primary_scan_accounting_journal": str(primary_measurement / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"),
        "replay_scan_accounting_journal": str(replay_measurement / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"),
        "primary_lifecycle_measurement": str(primary_measurement / "SOURCE_LIFECYCLE_MEASUREMENT.json"),
        "replay_lifecycle_measurement": str(replay_measurement / "SOURCE_LIFECYCLE_MEASUREMENT.json"),
        "primary_source_launch_capability": str(primary_measurement / "SOURCE_LAUNCH_CAPABILITY.json"),
        "replay_source_launch_capability": str(replay_measurement / "SOURCE_LAUNCH_CAPABILITY.json"),
        "primary_source_child_claim": str(primary_measurement / "SOURCE_CHILD.claim.json"),
        "replay_source_child_claim": str(replay_measurement / "SOURCE_CHILD.claim.json"),
        "manual_native_review": str(attempt / "EXP_CAD_0035_MANUAL_NATIVE_REVIEW.json"),
        "orchestration_result": str(attempt / "EXP_CAD_0035_ORCHESTRATION_RESULT.json"),
        "global_result": str(tmp_path / "EXP_CAD_0035_RESULT.json"),
        "launch_claim": str(attempt / "EXP_CAD_0035_LAUNCH.claim.json"),
    }
    prefix = "primary" if mode == "PRIMARY" else "replay"
    measurement = Path(authorized[f"{prefix}_measurement"])
    static = {
        "parent_preregistration": tmp_path / "parent.json",
        "exp31_preregistration": tmp_path / "exp31.json",
        "exp32_preregistration": tmp_path / "exp32.json",
        "cad_preflight": tmp_path / "cad.json",
        "cad_nodes": tmp_path / "nodes.npy",
        "cad_edges": tmp_path / "edges.npy",
        "cad_edge_geodesic_mm": tmp_path / "weights.npy",
        "accepted_mesh": tmp_path / "mesh.ply",
        "scan_N009": tmp_path / "N009.ply",
        "scan_N010": tmp_path / "N010.ply",
        "transform_N009": tmp_path / "T9.json",
        "transform_N010": tmp_path / "T10.json",
    }
    args = argparse.Namespace(
        mode=mode,
        preregistration=tmp_path / "prereg.json",
        lock=tmp_path / "lock.json",
        lock_sha256="a" * 64,
        detached_approval=tmp_path / "approval.json",
        detached_approval_sha256="b" * 64,
        launch_claim=Path(authorized["launch_claim"]),
        launch_claim_sha256="",
        source_launch_capability=measurement / "SOURCE_LAUNCH_CAPABILITY.json",
        source_launch_child_claim=measurement / "SOURCE_CHILD.claim.json",
        parent_preregistration=static["parent_preregistration"],
        exp31_preregistration=static["exp31_preregistration"],
        v5_preregistration=static["exp32_preregistration"],
        cad_preflight=static["cad_preflight"],
        nodes=static["cad_nodes"],
        edges=static["cad_edges"],
        edge_geodesic_mm=static["cad_edge_geodesic_mm"],
        mesh=static["accepted_mesh"],
        scan_first=static["scan_N009"],
        scan_second=static["scan_N010"],
        transform_first=static["transform_N009"],
        transform_second=static["transform_N010"],
        output_root=Path(authorized[f"{prefix}_source"]),
        measurement_root=measurement,
        scan_accounting_journal=Path(authorized[f"{prefix}_scan_accounting_journal"]),
        lifecycle_measurement=Path(authorized[f"{prefix}_lifecycle_measurement"]),
    )
    bound_paths = {
        "exp35_source_runner": Path(R.__file__).resolve(),
        "exp35_monitor": tmp_path / "monitor.py",
        "exp35_orchestrator": tmp_path / "orchestrator.py",
        "renderer": tmp_path / "renderer.py",
        **{key: value for key, value in static.items() if not key.startswith("scan_")},
    }
    bound = {key: {"path": str(path)} for key, path in bound_paths.items()}
    scans = {
        key: {"path": str(path)}
        for key, path in static.items()
        if key.startswith("scan_")
    }
    resolved_inputs = {
        key: str(path)
        for key, path in {
            **bound_paths,
            **{key: Path(row["path"]) for key, row in scans.items()},
        }.items()
    }
    launch = {
        "schema": R.LAUNCH_CLAIM_SCHEMA,
        "experiment_id": R.EXPERIMENT_ID,
        "architecture": R.ARCHITECTURE,
        "status": "IN_PROGRESS_CONSUMED_BEFORE_FIRST_CHILD",
        "invocation_ordinal": 1,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": R.PREREGISTRATION_SHA256,
        "lock_path": str(args.lock),
        "lock_sha256": args.lock_sha256,
        "detached_approval_path": str(args.detached_approval),
        "detached_approval_sha256": args.detached_approval_sha256,
        "resolved_inputs": dict(sorted(resolved_inputs.items())),
        "resolved_output_paths": {
            key: str(R.ROOT / value) for key, value in sorted(authorized.items())
        },
        "maximum_primary_source_children": 1,
        "maximum_replay_source_children": 1,
        "physical_source_execution_authorized": True,
        "evaluator_input_authorized": False,
    }
    args.launch_claim_sha256 = _strict_write(args.launch_claim, launch)
    lock = {"bound_files": bound, "physical_scans": scans, "authorized_outputs": authorized}
    raw = ["--unit-capability", mode]
    command = R._canonical_full_command(raw)
    capability = {
        "schema": R.SOURCE_CAPABILITY_SCHEMA,
        "experiment_id": R.EXPERIMENT_ID,
        "status": "AUTHORIZED_NOT_CLAIMED",
        "source_child_ordinal": 1 if mode == "PRIMARY" else 2,
        "mode": mode,
        "command_argv": command,
        "command_sha256": hashlib.sha256(R._strict_canonical_bytes(command)).hexdigest(),
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": R.PREREGISTRATION_SHA256,
        "lock_path": str(args.lock),
        "lock_sha256": args.lock_sha256,
        "detached_approval_path": str(args.detached_approval),
        "detached_approval_sha256": args.detached_approval_sha256,
        "launch_claim_path": str(args.launch_claim),
        "launch_claim_sha256": args.launch_claim_sha256,
        "output_root": str(args.output_root),
        "measurement_root": str(args.measurement_root),
        "scan_accounting_journal": str(args.scan_accounting_journal),
        "lifecycle_measurement": str(args.lifecycle_measurement),
        "source_launch_capability_path": str(args.source_launch_capability),
        "source_launch_child_claim_path": str(args.source_launch_child_claim),
        "evaluator_input_authorized": False,
    }
    _strict_write(args.source_launch_capability, capability)
    return args, lock, raw


def test_capability_claim_is_o_excl_durable_and_binds_actual_source_pid(
    tmp_path: Path,
) -> None:
    args, lock, raw = _capability_case(tmp_path)
    claim = R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert claim["source_process_pid"] == os.getpid()
    assert claim["source_child_ordinal"] == 1
    persisted = json.loads(args.source_launch_child_claim.read_text(encoding="utf-8"))
    assert persisted == claim
    assert stat.S_IMODE(args.source_launch_child_claim.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)


def test_missing_capability_rejects_before_child_claim(tmp_path: Path) -> None:
    args, lock, raw = _capability_case(tmp_path)
    args.source_launch_capability.unlink()
    with pytest.raises(FileNotFoundError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


@pytest.mark.parametrize("tamper", ("command", "command_hash", "ordinal", "raw_order", "mode"))
def test_wrong_capability_command_hash_order_ordinal_or_mode_rejects(
    tmp_path: Path, tamper: str
) -> None:
    args, lock, raw = _capability_case(tmp_path)
    payload = json.loads(args.source_launch_capability.read_text(encoding="utf-8"))
    if tamper == "command":
        payload["command_argv"] = payload["command_argv"] + ["--extra"]
    elif tamper == "command_hash":
        payload["command_sha256"] = "0" * 64
    elif tamper == "ordinal":
        payload["source_child_ordinal"] = 2
    elif tamper == "raw_order":
        raw = list(reversed(raw))
    else:
        args.mode = "REPLAY"
    if tamper not in {"raw_order", "mode"}:
        _strict_write(args.source_launch_capability, payload)
    with pytest.raises(ValueError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


def test_replay_claim_uses_ordinal_two_and_remains_one_shot(tmp_path: Path) -> None:
    args, lock, raw = _capability_case(tmp_path, mode="REPLAY")
    claim = R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert claim["mode"] == "REPLAY"
    assert claim["source_child_ordinal"] == 2
    assert claim["source_process_pid"] == os.getpid()
    with pytest.raises(FileExistsError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)


def test_child_claim_writer_fails_closed_on_zero_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claim = tmp_path / "SOURCE_CHILD.claim.json"
    monkeypatch.setattr(R.os, "write", lambda descriptor, data: 0)
    with pytest.raises(OSError, match="made no progress"):
        R._o_excl_canonical_write(claim, {"schema": "TEST"})
    assert claim.exists()
