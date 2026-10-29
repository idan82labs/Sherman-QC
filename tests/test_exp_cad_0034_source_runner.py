from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Mapping

import pytest


# Import the runner before importing any candidate/native dependency.  A
# separate subprocess test below proves that this import is a true barrier.
R = importlib.import_module("scripts.run_exp_cad_0034_cfpdg_source")

import numpy as np
import open3d as o3d

from scripts import exp_cad_0031_source_interface as INTAKE
from scripts import exp_cad_0032_cfpdg_v5_engine as V5
from scripts import exp_cad_0034_execution_contract as GOVERNANCE


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _source_free_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "np", np)
    monkeypatch.setattr(R, "o3d", o3d)
    monkeypatch.setattr(R, "INTAKE", INTAKE)
    monkeypatch.setattr(R, "V5", V5)
    monkeypatch.setattr(R, "GOVERNANCE", GOVERNANCE)


@pytest.fixture
def preregistrations() -> tuple[Mapping[str, Any], ...]:
    names = (
        "2026-08-30_exp_cad_0034_cfpdg_v5_physical_successor_preregistration.json",
        "2026-08-30_exp_cad_0030_cfpdg_v3_preregistration.json",
        "2026-08-30_exp_cad_0031_cfpdg_v4_preregistration.json",
        "2026-08-30_exp_cad_0032_cfpdg_v5_execution_equivalent_preregistration.json",
    )
    return tuple(
        json.loads((ROOT / "docs/research" / name).read_text(encoding="utf-8"))
        for name in names
    )


def _strict_write(path: Path, value: Any) -> str:
    block = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(block)
    return hashlib.sha256(block).hexdigest()


def _metrics() -> dict[str, Any]:
    counters = {
        "unique_member_fit_count": 38,
        "batched_kd_fit_count": 38,
        "kd_node_query_count": 306_926,
        "supplemental_diagnostic_batched_kd_fit_count": 0,
        "supplemental_diagnostic_kd_node_query_count": 0,
        "local_fit_call_count": 306_926,
        "compact_local_fit_call_count": 290_772,
        "diagnostic_local_fit_call_count": 16_154,
        "nuisance_design_build_count": 1,
        "nuisance_design_cache_hit_count": 0,
        "nuisance_design_hit_count": 0,
        "induced_components_call_count": 0,
        "induced_components_cache_hit_count": 0,
        "component_streamed_dijkstra_row_count": 0,
        "component_dijkstra_batch_call_count": 0,
        "induced_components_streamed_dijkstra_count": 0,
        "robustness_row_count": 32,
    }
    return {
        "schema": R.EXP32_METRICS_SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "architecture": V5.ARCHITECTURE,
        **counters,
        "operation_counters": dict(counters),
        "full_fit_inventory_gate_passed": True,
        "full_fit_batched_kd_gate_passed": True,
        "full_fit_kd_query_gate_passed": True,
        "fit_or_robustness_row_skipped": False,
        "scientific_threshold_or_gate_changed": False,
        "elapsed_wall_seconds_non_scientific": 12.5,
    }


def _semantic() -> dict[str, Any]:
    return {
        "schema": R.PARENT_SEMANTIC_SCHEMA,
        "experiment_id": "EXP_CAD_0030",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V3_HARDENED",
        "status": "ABSTAIN",
        "reason": "PAIR_SUPPORT_GATE",
        "robustness": [f"R{index:02d}" for index in range(32)],
    }


def _visual() -> dict[str, Any]:
    return {
        "schema": R.PARENT_VISUAL_SCHEMA,
        "experiment_id": "EXP_CAD_0030",
        "values": [0.0, 1.0],
    }


def _execute_kwargs(
    preregistrations: tuple[Mapping[str, Any], ...], rows: int
) -> dict[str, Any]:
    exp34, parent, exp31, exp32 = preregistrations
    points = np.zeros((rows, 3), dtype=np.float64)
    normals = np.zeros((rows, 3), dtype=np.float64)
    normals[:, 0] = 1.0
    return {
        "exp34_prereg": exp34,
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


def test_import_barrier_loads_no_candidate_numeric_or_native_modules() -> None:
    code = """
import importlib.util,json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(strict=True)
s=importlib.util.spec_from_file_location('exp34_barrier_probe',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
names=['numpy','open3d','scripts.exp_cad_0031_source_interface','scripts.exp_cad_0032_cfpdg_v5_engine','scripts.exp_cad_0034_execution_contract']
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


def test_parser_exposes_exact_capability_and_source_contract() -> None:
    help_text = R.build_parser().format_help()
    for flag in (
        "--mode",
        "--launch-claim",
        "--launch-claim-sha256",
        "--source-launch-capability",
        "--source-launch-child-claim",
        "--scan-accounting-journal",
    ):
        assert flag in help_text


def test_canonical_command_preserves_attested_python_invocation_spelling() -> None:
    attestation = json.loads(
        (
            ROOT
            / "docs/research/2026-08-30_exp_cad_0033_source_free_environment_attestation.json"
        ).read_text(encoding="utf-8")
    )
    command = R._canonical_full_command(["--probe"])
    assert command[0] == str(Path(sys.executable))
    assert command[0] == attestation["environment"]["python_executable_invocation_path"]
    assert str(Path(command[0]).resolve()) == attestation["environment"][
        "python_executable_realpath"
    ]


def test_interface_abstention_invokes_exact_filter_pair_and_zero_engine(
    preregistrations: tuple[Mapping[str, Any], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"filter": 0, "pair": 0, "engine": 0}
    original_filter = INTAKE.filter_source_rows
    original_pair = INTAKE.pair_actionability

    def filter_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls["filter"] += 1
        return original_filter(*args, **kwargs)

    def pair_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls["pair"] += 1
        return original_pair(*args, **kwargs)

    def forbidden_engine(**_kwargs: Any) -> Any:
        calls["engine"] += 1
        raise AssertionError("V5 engine reached on interface abstention")

    monkeypatch.setattr(INTAKE, "filter_source_rows", filter_spy)
    monkeypatch.setattr(INTAKE, "pair_actionability", pair_spy)
    monkeypatch.setattr(V5, "execute_parent_science_v5", forbidden_engine)
    monkeypatch.setattr(
        V5,
        "filter_source_rows_v5",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("forbidden V5 filter invoked")
        ),
    )
    result = R.execute_source_exp34(**_execute_kwargs(preregistrations, rows=8))
    assert calls == {"filter": 2, "pair": 1, "engine": 0}
    assert result["status"] == "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
    assert result["scientific_execute_source_reached"] is False
    assert result["engine_accounting"] == R._zero_engine_accounting()
    assert result["visual_payload"] is None
    assert result["engine_metrics"] is None


def test_actionable_path_filters_once_then_calls_direct_v5_once(
    preregistrations: tuple[Mapping[str, Any], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"filter": 0, "pair": 0, "engine": 0}
    original_filter = INTAKE.filter_source_rows
    original_pair = INTAKE.pair_actionability

    def filter_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls["filter"] += 1
        return original_filter(*args, **kwargs)

    def pair_spy(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        calls["pair"] += 1
        return original_pair(*args, **kwargs)

    def engine_spy(**kwargs: Any) -> tuple[Mapping[str, Any], ...]:
        calls["engine"] += 1
        assert len(kwargs["raw_first"]) == 100_000
        assert len(kwargs["raw_second"]) == 100_000
        return _semantic(), _visual(), _metrics()

    monkeypatch.setattr(INTAKE, "filter_source_rows", filter_spy)
    monkeypatch.setattr(INTAKE, "pair_actionability", pair_spy)
    monkeypatch.setattr(V5, "execute_parent_science_v5", engine_spy)
    monkeypatch.setattr(
        V5,
        "filter_source_rows_v5",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("forbidden V5 filter invoked")
        ),
    )
    result = R.execute_source_exp34(
        **_execute_kwargs(preregistrations, rows=100_000)
    )
    assert calls == {"filter": 2, "pair": 1, "engine": 1}
    assert result["scientific_execute_source_reached"] is True
    assert result["semantic_payload"] == _semantic()
    assert result["visual_payload"] == _visual()
    accounting = result["engine_accounting"]
    assert accounting["v5_engine_invocation_count"] == 1
    assert accounting["kd_node_query_count"] == 306_926
    assert accounting["robustness_row_count"] == 32


def _transform_payload() -> dict[str, Any]:
    return {
        "schema": R.TRANSFORM_SCHEMA,
        "experiment_id": "EXP_CAD_0015",
        "case_token": "N009",
        "scan_sha256": R.EXPECTED_SCAN_SHA256["N009"],
        "raw_scan_point_count": R.EXPECTED_ROW_COUNTS["N009"],
        "alignment_abstained": False,
        "evaluator_data_opened": False,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
        "raw_scan_to_reference_transform": np.eye(4).tolist(),
    }


def test_transform_accepts_only_exact_raw_scan_proper_rigid_field() -> None:
    transform = R.validate_transform_payload(
        _transform_payload(),
        token="N009",
        scan_sha256=R.EXPECTED_SCAN_SHA256["N009"],
        expected_row_count=R.EXPECTED_ROW_COUNTS["N009"],
    )
    assert transform.dtype == np.float64
    assert transform.flags.c_contiguous
    assert np.array_equal(transform, np.eye(4))


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("schema", "WRONG"),
        ("experiment_id", "EXP_CAD_9999"),
        ("case_token", "N010"),
        ("scan_sha256", "0" * 64),
        ("raw_scan_point_count", 1),
        ("alignment_abstained", True),
        ("evaluator_data_opened", True),
        ("gold_or_expected_counts_used", True),
        ("generalization_evidence", True),
    ),
)
def test_transform_rejects_linkage_or_sealed_flag_drift(
    mutation: str, value: Any
) -> None:
    payload = _transform_payload()
    payload[mutation] = value
    with pytest.raises(ValueError):
        R.validate_transform_payload(
            payload,
            token="N009",
            scan_sha256=R.EXPECTED_SCAN_SHA256["N009"],
            expected_row_count=R.EXPECTED_ROW_COUNTS["N009"],
        )


@pytest.mark.parametrize("alternate", ("transform", "selected_absolute_transform"))
def test_transform_rejects_any_alternate_field_even_with_raw_field(
    alternate: str,
) -> None:
    payload = _transform_payload()
    payload[alternate] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="forbidden alternate"):
        R.validate_transform_payload(
            payload,
            token="N009",
            scan_sha256=R.EXPECTED_SCAN_SHA256["N009"],
            expected_row_count=R.EXPECTED_ROW_COUNTS["N009"],
        )


@pytest.mark.parametrize("kind", ("missing", "nonfinite", "bottom", "reflection"))
def test_transform_rejects_missing_or_nonproper_geometry(kind: str) -> None:
    payload = _transform_payload()
    if kind == "missing":
        del payload["raw_scan_to_reference_transform"]
    elif kind == "nonfinite":
        payload["raw_scan_to_reference_transform"][0][0] = float("nan")
    elif kind == "bottom":
        payload["raw_scan_to_reference_transform"][3][0] = 1.0
    else:
        payload["raw_scan_to_reference_transform"][0][0] = -1.0
    with pytest.raises(ValueError):
        R.validate_transform_payload(
            payload,
            token="N009",
            scan_sha256=R.EXPECTED_SCAN_SHA256["N009"],
            expected_row_count=R.EXPECTED_ROW_COUNTS["N009"],
        )


def test_scan_access_checkpoints_exact_canonical_order_before_each_call(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"
    paths = {
        "N009": tmp_path / "N009.ply",
        "N010": tmp_path / "N010.ply",
    }
    events: list[tuple[str, str, str]] = []

    def hasher(path: Path) -> str:
        payload = json.loads(journal.read_text(encoding="utf-8"))
        token = Path(path).stem
        events.append((token, "SHA256", payload["stage"]))
        return R.EXPECTED_SCAN_SHA256[token]

    def reader(path: str, **kwargs: Any) -> object:
        payload = json.loads(journal.read_text(encoding="utf-8"))
        token = Path(path).stem
        events.append((token, "POINT_CLOUD_LOAD", payload["stage"]))
        assert kwargs == {
            "format": "auto",
            "remove_nan_points": False,
            "remove_infinite_points": False,
            "print_progress": False,
        }
        return object()

    accounting = R._new_scan_accounting()
    first, second = R.perform_ordered_scan_access(
        mode="PRIMARY",
        scan_first=paths["N009"],
        scan_second=paths["N010"],
        journal_path=journal,
        accounting=accounting,
        sha256_func=hasher,
        point_cloud_reader=reader,
    )
    assert first is not second
    assert events == [
        ("N009", "SHA256", "BEFORE_N009_SHA256"),
        ("N010", "SHA256", "BEFORE_N010_SHA256"),
        ("N009", "POINT_CLOUD_LOAD", "BEFORE_N009_POINT_CLOUD_LOAD"),
        ("N010", "POINT_CLOUD_LOAD", "BEFORE_N010_POINT_CLOUD_LOAD"),
    ]
    assert accounting["physical_scan_hash_call_count"] == 2
    assert accounting["physical_scan_point_cloud_load_call_count"] == 2
    assert accounting["physical_scan_byte_open_operation_count"] == 4


@pytest.mark.parametrize(("failed_token", "hash_count"), (("N009", 1), ("N010", 2)))
def test_scan_hash_mismatch_is_terminal_before_any_load(
    tmp_path: Path, failed_token: str, hash_count: int
) -> None:
    loads = 0

    def hasher(path: Path) -> str:
        token = Path(path).stem
        return "0" * 64 if token == failed_token else R.EXPECTED_SCAN_SHA256[token]

    def reader(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal loads
        loads += 1
        raise AssertionError("loader reached after hash mismatch")

    accounting = R._new_scan_accounting()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        R.perform_ordered_scan_access(
            mode="PRIMARY",
            scan_first=tmp_path / "N009.ply",
            scan_second=tmp_path / "N010.ply",
            journal_path=tmp_path / "journal.json",
            accounting=accounting,
            sha256_func=hasher,
            point_cloud_reader=reader,
        )
    assert loads == 0
    assert accounting["physical_scan_hash_call_count"] == hash_count
    assert accounting["physical_scan_point_cloud_load_call_count"] == 0


def _binary_ply(rows: list[list[float]]) -> bytes:
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(rows)}",
        "property double x",
        "property double y",
        "property double z",
        "property double nx",
        "property double ny",
        "property double nz",
        "end_header",
    ]
    return ("\n".join(header) + "\n").encode("ascii") + b"".join(
        struct.pack("<6d", *row) for row in rows
    )


def test_synthetic_ply_loader_preserves_order_nonfinite_rows_and_normals(
    tmp_path: Path,
) -> None:
    rows = [
        [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [float("nan"), 1.0, 0.0, 0.0, 1.0, 0.0],
        [4.0, float("inf"), 0.0, 0.0, 0.0, 1.0],
        [6.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    ]
    paths = [tmp_path / "N009.ply", tmp_path / "N010.ply"]
    for path in paths:
        path.write_bytes(_binary_ply(rows))
    accounting = R._new_scan_accounting()
    clouds = R.perform_ordered_scan_access(
        mode="PRIMARY",
        scan_first=paths[0],
        scan_second=paths[1],
        journal_path=tmp_path / "journal.json",
        accounting=accounting,
        expected_hashes={
            "N009": R._scan_sha256(paths[0]),
            "N010": R._scan_sha256(paths[1]),
        },
    )
    points, normals = R.materialize_loaded_cloud(
        clouds[0], token="N009", expected_row_count=4
    )
    assert points.shape == normals.shape == (4, 3)
    assert points.flags.c_contiguous and normals.flags.c_contiguous
    assert points[0, 0] == 2.0 and points[3, 0] == 6.0
    assert np.isnan(points[1, 0])
    assert np.isposinf(points[2, 1])
    assert np.array_equal(normals[0], [1.0, 0.0, 0.0])
    assert np.array_equal(normals[3], [0.0, 0.0, 0.0])


def test_loaded_cloud_requires_exact_rows_and_supplied_matching_normals() -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="normal"):
        R.materialize_loaded_cloud(cloud, token="N009", expected_row_count=3)
    cloud.normals = o3d.utility.Vector3dVector(np.ones((3, 3)))
    with pytest.raises(ValueError, match="point row"):
        R.materialize_loaded_cloud(cloud, token="N009", expected_row_count=4)


def test_interface_publication_has_no_visual_or_exp32_metrics(
    preregistrations: tuple[Mapping[str, Any], ...], tmp_path: Path
) -> None:
    computation = R.execute_source_exp34(
        **_execute_kwargs(preregistrations, rows=4)
    )
    source = tmp_path / "source"
    measurement = tmp_path / "measurement"
    outer = R.publish_source_artifacts(
        output_root=source,
        measurement_root=measurement,
        computation=computation,
    )
    assert outer["status"] == "ABSTAIN_SOURCE_INTERFACE_NONACTIONABLE_TERMINAL_NO_RERUN"
    assert outer["scientific_execute_source_reached"] is False
    assert not (source / "SOURCE_VISUAL_PAYLOAD.json").exists()
    assert not (source / "SOURCE_SEMANTIC_PAYLOAD.json").exists()
    assert not (measurement / "EXP32_ENGINE_METRICS.json").exists()
    assert json.loads((source / "SOURCE_RESULT.json").read_text()) == outer


def test_science_publication_preserves_nested_identity_and_separates_metrics(
    tmp_path: Path,
) -> None:
    metrics = _metrics()
    metrics["elapsed_wall_seconds_non_scientific"] = float("inf")
    computation: dict[str, Any] = {
        "status": "ABSTAIN",
        "reason": "PAIR_SUPPORT_GATE",
        "scientific_execute_source_reached": True,
        "source_interface": {
            "first": {"schema": INTAKE.SCHEMA},
            "second": {"schema": INTAKE.SCHEMA},
            "pair": {"actionable": True, "status": "PASS_SOURCE_INTERFACE_ACTIONABLE"},
        },
        "retained_identity_hashes": {"schema": "X"},
        "semantic_payload": _semantic(),
        "visual_payload": _visual() | {"nonfinite": float("nan")},
        "engine_metrics": metrics,
        "engine_accounting": R._science_engine_accounting(metrics),
    }
    source = tmp_path / "source"
    measurement = tmp_path / "measurement"
    outer = R.publish_source_artifacts(
        output_root=source,
        measurement_root=measurement,
        computation=computation,
    )
    assert outer["scientific_execute_source_reached"] is True
    assert json.loads((source / "SOURCE_SEMANTIC_PAYLOAD.json").read_text())["experiment_id"] == "EXP_CAD_0030"
    assert json.loads((source / "SOURCE_VISUAL_PAYLOAD.json").read_text())["nonfinite"] is None
    persisted_metrics = json.loads(
        (measurement / "EXP32_ENGINE_METRICS.json").read_text()
    )
    assert persisted_metrics["schema"] == R.EXP32_METRICS_SCHEMA
    assert persisted_metrics["elapsed_wall_seconds_non_scientific"] is None
    assert not (source / "EXP32_ENGINE_METRICS.json").exists()


def _capability_case(tmp_path: Path, *, mode: str = "PRIMARY") -> tuple[argparse.Namespace, dict[str, Any], list[str]]:
    attempt = tmp_path / "attempt"
    primary_measurement = attempt / "primary" / "measurement"
    replay_measurement = attempt / "replay" / "measurement"
    authorized = {
        "result_namespace": str(attempt),
        "primary_source": str(attempt / "primary" / "source"),
        "replay_source": str(attempt / "replay" / "source"),
        "primary_measurement": str(primary_measurement),
        "replay_measurement": str(replay_measurement),
        "primary_renderer": str(attempt / "primary" / "renderer"),
        "replay_renderer": str(attempt / "replay" / "renderer"),
        "primary_scan_accounting_journal": str(primary_measurement / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"),
        "replay_scan_accounting_journal": str(replay_measurement / "SOURCE_SCAN_ACCOUNTING_JOURNAL.json"),
        "primary_source_launch_capability": str(primary_measurement / "SOURCE_LAUNCH_CAPABILITY.json"),
        "replay_source_launch_capability": str(replay_measurement / "SOURCE_LAUNCH_CAPABILITY.json"),
        "primary_source_child_claim": str(primary_measurement / "SOURCE_CHILD.claim.json"),
        "replay_source_child_claim": str(replay_measurement / "SOURCE_CHILD.claim.json"),
        "manual_native_review": str(attempt / "EXP_CAD_0034_MANUAL_NATIVE_VISUAL_REVIEW.json"),
        "orchestration_result": str(attempt / "EXP_CAD_0034_ORCHESTRATION_RESULT.json"),
        "global_result": str(tmp_path / "EXP_CAD_0034_RESULT.json"),
        "launch_claim": str(attempt / "EXP_CAD_0034_LAUNCH.claim.json"),
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
    )
    bound_paths = {
        "exp34_source_runner": Path(R.__file__).resolve(),
        "exp34_monitor": tmp_path / "monitor.py",
        "exp34_orchestrator": tmp_path / "orchestrator.py",
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
        key: str(path) for key, path in {**bound_paths, **{k: Path(v["path"]) for k, v in scans.items()}}.items()
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
    lock = {
        "bound_files": bound,
        "physical_scans": scans,
        "authorized_outputs": authorized,
    }
    raw = ["--unit-capability", mode]
    command = R._canonical_full_command(raw)
    command_sha = hashlib.sha256(R._strict_canonical_bytes(command)).hexdigest()
    capability = {
        "schema": R.SOURCE_CAPABILITY_SCHEMA,
        "experiment_id": R.EXPERIMENT_ID,
        "status": "AUTHORIZED_NOT_CLAIMED",
        "source_child_ordinal": 1 if mode == "PRIMARY" else 2,
        "mode": mode,
        "command_argv": command,
        "command_sha256": command_sha,
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
        "source_launch_capability_path": str(args.source_launch_capability),
        "source_launch_child_claim_path": str(args.source_launch_child_claim),
        "evaluator_input_authorized": False,
    }
    _strict_write(args.source_launch_capability, capability)
    return args, lock, raw


def test_exact_capability_creates_one_durable_child_claim_and_second_attempt_fails(
    tmp_path: Path,
) -> None:
    args, lock, raw = _capability_case(tmp_path)
    result = R.validate_and_claim_source_capability(
        args=args, lock=lock, raw_argv=raw
    )
    assert result["status"] == "CLAIMED_BEFORE_STATIC_OR_SCAN_ACCESS"
    assert result["source_child_ordinal"] == 1
    assert args.source_launch_child_claim.is_file()
    with pytest.raises(FileExistsError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)


def test_direct_invocation_without_capability_fails_before_child_claim(
    tmp_path: Path,
) -> None:
    args, lock, raw = _capability_case(tmp_path)
    args.source_launch_capability.unlink()
    with pytest.raises(FileNotFoundError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


def test_missing_global_launch_claim_rejects_capability(tmp_path: Path) -> None:
    args, lock, raw = _capability_case(tmp_path)
    args.launch_claim.unlink()
    with pytest.raises(FileNotFoundError):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


@pytest.mark.parametrize("tamper", ("command", "command_hash"))
def test_wrong_canonical_command_or_hash_rejects_capability(
    tmp_path: Path, tamper: str
) -> None:
    args, lock, raw = _capability_case(tmp_path)
    payload = json.loads(args.source_launch_capability.read_text())
    if tamper == "command":
        payload["command_argv"] = payload["command_argv"] + ["--extra"]
    else:
        payload["command_sha256"] = "0" * 64
    _strict_write(args.source_launch_capability, payload)
    with pytest.raises(ValueError, match="capability"):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


def test_primary_capability_cannot_be_used_as_replay(tmp_path: Path) -> None:
    args, lock, raw = _capability_case(tmp_path, mode="PRIMARY")
    args.mode = "REPLAY"
    with pytest.raises(ValueError, match="locked REPLAY schedule"):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


def test_replay_capability_requires_replay_path_and_ordinal(tmp_path: Path) -> None:
    args, lock, raw = _capability_case(tmp_path, mode="REPLAY")
    payload = json.loads(args.source_launch_capability.read_text())
    payload["source_child_ordinal"] = 1
    _strict_write(args.source_launch_capability, payload)
    with pytest.raises(ValueError, match="capability"):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


def test_capability_symlink_is_rejected_before_read_or_claim(tmp_path: Path) -> None:
    args, lock, raw = _capability_case(tmp_path)
    payload = args.source_launch_capability.read_bytes()
    target = tmp_path / "capability-target.json"
    target.write_bytes(payload)
    args.source_launch_capability.unlink()
    args.source_launch_capability.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        R.validate_and_claim_source_capability(args=args, lock=lock, raw_argv=raw)
    assert not args.source_launch_child_claim.exists()


def test_strict_publication_maps_all_nonfinite_scalars_to_null(tmp_path: Path) -> None:
    path = tmp_path / "canonical.json"
    R.canonical_json_write(
        path,
        {"nan": float("nan"), "pos": float("inf"), "neg": -float("inf")},
    )
    assert path.read_bytes() == b'{"nan":null,"neg":null,"pos":null}'
