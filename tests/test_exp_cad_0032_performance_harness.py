from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load(ROOT / "scripts/exp_cad_0032_performance_harness.py", "exp32_harness_test")
W = _load(ROOT / "scripts/exp_cad_0032_performance_worker.py", "exp32_worker_test")


def _nodes(count: int = 11) -> np.ndarray:
    dtype = [
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
        ("e1x", "<f8"), ("e1y", "<f8"), ("e1z", "<f8"),
        ("e2x", "<f8"), ("e2y", "<f8"), ("e2z", "<f8"),
    ]
    value = np.zeros(count, dtype=dtype)
    value["x"] = np.arange(count)
    value["ny"] = 1.0
    value["e1x"] = 1.0
    value["e2z"] = 1.0
    return value


def test_synthetic_envelope_is_deterministic_finite_and_prefix_nested() -> None:
    nodes = _nodes()
    full_points, full_normals = H.synthetic_block(nodes, 0, 100, member_index=0)
    prefix_points, prefix_normals = H.synthetic_block(nodes, 0, 37, member_index=0)
    again_points, again_normals = H.synthetic_block(nodes, 0, 100, member_index=0)
    assert np.array_equal(full_points[:37], prefix_points)
    assert np.array_equal(full_normals[:37], prefix_normals)
    assert np.array_equal(full_points, again_points)
    assert np.array_equal(full_normals, again_normals)
    assert np.isfinite(full_points).all()
    assert np.isfinite(full_normals).all()
    assert np.allclose(np.linalg.norm(full_normals, axis=1), 1.0)


def test_members_are_independent_acquisitions_without_planted_normal_offset() -> None:
    nodes = _nodes()
    first, first_normals = H.synthetic_block(nodes, 0, 100, member_index=0)
    second, second_normals = H.synthetic_block(nodes, 0, 100, member_index=1)
    assert not np.array_equal(first, second)
    assert set(np.unique(np.abs(first_normals))) <= {0.0, 1.0}
    assert set(np.unique(np.abs(second_normals))) <= {0.0, 1.0}


def test_canonical_worker_write_is_order_invariant_and_byte_exact(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left_hash = W.canonical_json_write(left, {"b": [2, 3], "a": np.int64(1)})
    right_hash = W.canonical_json_write(right, {"a": 1, "b": [2, 3]})
    assert left_hash == right_hash
    assert left.read_bytes() == right.read_bytes()
    assert hashlib.sha256(left.read_bytes()).hexdigest() == left_hash


def test_canonical_worker_write_matches_frozen_nonfinite_to_null_policy(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    W.canonical_json_write(path, {"nan": float("nan"), "inf": float("inf")})
    assert path.read_text(encoding="utf-8") == '{"inf":null,"nan":null}'


def test_exact_operation_counters_exclude_nondeterministic_metrics() -> None:
    metrics = {
        "operation_counters": {
            "unique_member_fit_count": 38,
            "robustness_row_count": 32,
            "kd_node_query_count": 99,
        },
        "elapsed_seconds": 9.1,
        "peak_rss_bytes": 123,
    }
    assert W._exact_operation_counters(metrics) == {
        "kd_node_query_count": 99,
        "robustness_row_count": 32,
        "unique_member_fit_count": 38,
    }


def test_time_l_rss_parser_supports_macos_and_gnu() -> None:
    assert H.parse_time_l_max_rss("  123456 maximum resident set size\n") == 123456
    assert H.parse_time_l_max_rss("Maximum resident set size (kbytes): 2048\n") == 2 * 1024**2
    assert H.parse_time_l_max_rss("nothing") is None


def _full_run(index: int, *, source_wall: float = 10.0, render_wall: float = 2.0):
    source = {
        "semantic_canonical_sha256": "a" * 64,
        "visual_canonical_sha256": "b" * 64,
        "scientific_tree_canonical_sha256": "c" * 64,
        "engine_operation_counters": {
            "unique_member_fit_count": 38,
            "robustness_row_count": 32,
        },
        "worker_operation_counters": {"scientific_execute_call_count": 1},
        "equivalence_checks": {
            "raycast_tie_differential_passed": True,
            "kd_tie_differential_passed": True,
        },
    }
    measurement = {
        "root_pid": 100 + index,
        "returncode": 0,
        "timed_out": False,
        "wall_seconds": source_wall,
        "peak_process_tree_rss_bytes": 1000,
    }
    renderer = {"render_manifest_canonical_sha256": "d" * 64}
    render_measurement = {
        "root_pid": 200 + index,
        "returncode": 0,
        "timed_out": False,
        "wall_seconds": render_wall,
        "peak_process_tree_rss_bytes": 1000,
    }
    return {
        "source": source,
        "source_measurement": measurement,
        "source_full_tree_hash_map": {"WORKER_RESULT.json": {"sha256": "f" * 64}},
        "renderer": renderer,
        "renderer_measurement": render_measurement,
        "renderer_full_tree_hash_map": {"WORKER_RESULT.json": {"sha256": "0" * 64}},
    }


def test_gate_requires_exact_replay_full_workload_scaling_and_bundle() -> None:
    runs = [_full_run(index) for index in range(3)]
    half = {"source_measurement": {"wall_seconds": 5.0}}
    gates = H.evaluate_gates(
        full_runs=runs,
        half_run=half,
        differential={"status": "PASS"},
        bundle_logical_bytes=1000,
        bundle_allocated_bytes=1000,
    )
    assert gates["status"] == "PASS"
    assert gates["scaling"]["ratio"] == 2.0
    runs[2]["source"]["visual_canonical_sha256"] = "e" * 64
    failed = H.evaluate_gates(
        full_runs=runs,
        half_run=half,
        differential={"status": "PASS"},
        bundle_logical_bytes=1000,
        bundle_allocated_bytes=1000,
    )
    assert failed["status"] == "FAIL"
    assert failed["checks"]["three_run_visual_canonical_hash_exact"] is False


def test_scientific_tree_identity_ignores_monitoring_time() -> None:
    source = _full_run(0)["source"] | {
        "semantic_schema": "SEMANTIC",
        "visual_schema": "VISUAL",
        "retained_row_identity": {"first": {"count": 1}, "second": {"count": 1}},
    }
    first = H._scientific_tree_identity(source | {"engine_metrics": {"elapsed_seconds": 1.0}})
    second = H._scientific_tree_identity(source | {"engine_metrics": {"elapsed_seconds": 9.0}})
    assert first == second


def test_full_tree_hash_map_covers_every_relative_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "nested/b.bin").write_bytes(b"b")
    tree = H.full_tree_hash_map(tmp_path)
    assert set(tree) == {"a.bin", "nested/b.bin"}
    assert tree["a.bin"]["sha256"] == hashlib.sha256(b"a").hexdigest()


def test_same_cardinality_cad_substitution_is_rejected_by_hash() -> None:
    bindings = {
        key: {"sha256": digest} for key, digest in H.EXPECTED_CAD_SHA256.items()
        if key != "cad_preflight"
    }
    H.validate_frozen_cad_hashes(bindings)
    bindings["cad_nodes"] = {"sha256": "0" * 64}
    with pytest.raises(ValueError, match="cad_nodes"):
        H.validate_frozen_cad_hashes(bindings)
