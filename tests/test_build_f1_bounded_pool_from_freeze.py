from pathlib import Path

import numpy as np
import pytest

from scripts import build_f1_bounded_pool_from_freeze as builder
from scripts.build_f1_bounded_pool_from_freeze import (
    _declared_ply_vertex_count,
    _decomposition_path,
    _memory_safe_worker_cap,
)


def test_decomposition_discovery_is_opaque_and_exact(tmp_path: Path):
    expected = tmp_path / "C001" / "result" / "opaque" / "decomposition_result.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}")

    assert _decomposition_path([tmp_path], "C001") == expected.resolve()


def test_decomposition_discovery_fails_on_ambiguity(tmp_path: Path):
    for name in ("a", "b"):
        path = tmp_path / "C001" / "result" / name / "decomposition_result.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")

    with pytest.raises(ValueError, match="expected one"):
        _decomposition_path([tmp_path], "C001")


def test_memory_safe_parallelism_uses_measured_per_job_rss():
    gib = 1024**3
    assert _memory_safe_worker_cap(
        measured_peak_rss_bytes=12 * gib,
        memory_budget_bytes=96 * gib,
    ) == 8
    assert _memory_safe_worker_cap(
        measured_peak_rss_bytes=1 * gib,
        memory_budget_bytes=96 * gib,
    ) == 18


def test_memory_safe_parallelism_rejects_one_job_over_budget():
    gib = 1024**3
    with pytest.raises(ValueError, match="exceeds"):
        _memory_safe_worker_cap(
            measured_peak_rss_bytes=97 * gib,
            memory_budget_bytes=96 * gib,
        )


def test_ply_vertex_budget_is_read_from_header_without_loading_geometry(tmp_path: Path):
    path = tmp_path / "scan.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 200001\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    assert _declared_ply_vertex_count(path) == 200_001


def _frozen_case(source: Path, digest: str = "expected-hash") -> dict:
    return {
        "case_token": "C001",
        "source_scan_path": str(source),
        "source_scan_sha256": digest,
    }


def _decomposition(path: Path, scan: Path) -> None:
    path.write_text(
        '{"scan_path": "'
        + str(scan)
        + '", "part_id": "must-be-stripped", '
        '"provenance": {"coordinate_frame": "scan_raw"}}',
        encoding="utf-8",
    )


def test_build_one_rejects_decomposition_scan_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(b"scan")
    decomposition = tmp_path / "decomposition_result.json"
    _decomposition(decomposition, scan)
    monkeypatch.setattr(builder, "_decomposition_path", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(builder, "_sha256", lambda path: "actual-hash")

    with pytest.raises(ValueError, match="does not match frozen source"):
        builder.build_one(
            case=_frozen_case(scan),
            decomposition_roots=[tmp_path],
            source_case_id=None,
            output_dir=tmp_path / "out",
            voxel_mm=1.8,
        )


def test_build_one_fails_closed_when_mesh_lane_exceeds_global_vertex_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(b"scan")
    decomposition = tmp_path / "decomposition_result.json"
    _decomposition(decomposition, scan)
    monkeypatch.setattr(builder, "_decomposition_path", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(builder, "_sha256", lambda path: "expected-hash")
    empty = np.empty((0, 3), dtype=np.float64)
    monkeypatch.setattr(
        builder,
        "_load_geometry_views",
        lambda *args, **kwargs: (empty, empty, empty, empty),
    )
    monkeypatch.setattr(
        builder,
        "_scan_mesh_adjacency_candidates",
        lambda path: ([], {"vertex_count": builder.GLOBAL_POINT_BUDGET + 1}),
    )

    with pytest.raises(ValueError, match="scan-mesh adjacency consumed"):
        builder.build_one(
            case=_frozen_case(scan),
            decomposition_roots=[tmp_path],
            source_case_id=None,
            output_dir=tmp_path / "out",
            voxel_mm=1.8,
        )


def test_build_one_disables_oversized_mesh_lane_before_second_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scan = tmp_path / "scan.ply"
    scan.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 200001\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    decomposition = tmp_path / "decomposition_result.json"
    _decomposition(decomposition, scan)
    monkeypatch.setattr(builder, "_decomposition_path", lambda *args, **kwargs: decomposition)
    monkeypatch.setattr(builder, "_sha256", lambda path: "expected-hash")
    empty = np.empty((0, 3), dtype=np.float64)
    monkeypatch.setattr(
        builder,
        "_load_geometry_views",
        lambda *args, **kwargs: (empty, empty, empty, empty),
    )
    monkeypatch.setattr(
        builder,
        "_scan_mesh_adjacency_candidates",
        lambda path: (_ for _ in ()).throw(AssertionError("mesh reader must stay disabled")),
    )
    monkeypatch.setattr(
        builder,
        "build_case_pool",
        lambda **kwargs: {
            "source_point_serialization": False,
            "working_point_contract_version": "GLOBAL_POINT_BUDGET_V1",
            "working_point_budget": builder.GLOBAL_POINT_BUDGET,
            "transient_sampled_point_count": 0,
            "transient_plane_point_count": 0,
            "candidate_count": 0,
            "line_object_count": 0,
            "generator_diagnostics": {},
        },
    )
    output = tmp_path / "out"
    output.mkdir()

    builder.build_one(
        case=_frozen_case(scan),
        decomposition_roots=[tmp_path],
        source_case_id=None,
        output_dir=output,
        voxel_mm=1.8,
    )

    payload = builder._read(output / "C001.json")
    assert payload["generator_diagnostics"]["scan_mesh_adjacency"] == {
        "available": False,
        "reason": "disabled_by_global_vertex_budget",
        "declared_vertex_count": 200_001,
        "consumed_vertex_count": 0,
        "candidate_count": 0,
    }
