"""C06 viewer assets: strict PLY I/O, deterministic voxel decimation and the
index-aligned viewer copies built from a (fake) E3 output directory.

Everything runs on synthetic data in tmp_path (the SYN-HAT-1 contract example
via ``domains.production.inspection.fake_engine_output``); no real scan or CAD
is read.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from contracts.production.inspection_result import BendGeometryFileV1
from domains.production.inspection.engine_output import load_result
from domains.production.inspection.fake_engine_output import build_fake_engine_output
from domains.production.inspection.ply_io import PlyFormatError, read_ply, write_mesh_ply, write_points_ply
from domains.production.inspection.viewer_assets import (
    VOXEL_GROWTH,
    VOXEL_START_MM,
    build_viewer_assets,
    read_deviations,
    voxel_decimate_indices,
)


# ---------------------------------------------------------------- PLY I/O
def test_points_ply_round_trip_and_fixed_bytes(tmp_path: Path) -> None:
    pts = np.array([[0.0, 1.0, 2.0], [-3.5, 4.25, 1e3]], dtype=np.float64)
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    write_points_ply(a, pts)
    write_points_ply(b, pts)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
    data = read_ply(a)
    assert data.faces is None and data.format == "binary_little_endian"
    np.testing.assert_allclose(data.vertices, pts.astype(np.float32))


def test_mesh_round_trip(tmp_path: Path) -> None:
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    path = tmp_path / "m.ply"
    write_mesh_ply(path, v, f)
    data = read_ply(path, require_faces=True)
    np.testing.assert_array_equal(data.faces, f)
    np.testing.assert_allclose(data.vertices, v)


def test_ascii_and_big_endian_quads_are_read(tmp_path: Path) -> None:
    ascii_ply = tmp_path / "ascii.ply"
    ascii_ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0 1\n1 0 0 2\n1 1 0 3\n0 1 0 4\n4 0 1 2 3\n"
    )
    data = read_ply(ascii_ply)
    np.testing.assert_array_equal(data.faces, [[0, 1, 2], [0, 2, 3]])  # quad fan-triangulated
    be = tmp_path / "be.ply"
    header = b"ply\nformat binary_big_endian 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
    be.write_bytes(header + struct.pack(">6d", 1, 2, 3, 4, 5, 6))
    np.testing.assert_allclose(read_ply(be).vertices, [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda raw: raw[:-5], "truncated"),
        (lambda raw: raw.replace(b"format binary_little_endian 1.0", b"format binary_little_endian 2.0"), "format"),
        (lambda raw: raw.replace(b"ply\n", b"plx\n", 1), "not a PLY"),
    ],
)
def test_malformed_ply_is_rejected(tmp_path: Path, mutate, needle: str) -> None:
    path = tmp_path / "p.ply"
    write_points_ply(path, np.zeros((3, 3)))
    path.write_bytes(mutate(path.read_bytes()))
    with pytest.raises(PlyFormatError, match=needle):
        read_ply(path)


def test_non_finite_and_out_of_range_faces_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nan.ply"
    write_points_ply(path, np.array([[0, 0, np.nan]]))
    with pytest.raises(PlyFormatError, match="non-finite"):
        read_ply(path)
    mesh = tmp_path / "bad.ply"
    write_mesh_ply(mesh, np.zeros((3, 3)), np.array([[0, 1, 3]]))
    with pytest.raises(PlyFormatError, match="out of range"):
        read_ply(mesh)
    pts = tmp_path / "pts.ply"
    write_points_ply(pts, np.zeros((3, 3)))
    with pytest.raises(PlyFormatError, match="no faces"):
        read_ply(pts, require_faces=True)


# ---------------------------------------------------------------- decimation
def test_no_decimation_below_cap() -> None:
    pts = np.random.default_rng(0).normal(size=(100, 3))
    idx, voxel = voxel_decimate_indices(pts, 100)
    assert voxel is None
    np.testing.assert_array_equal(idx, np.arange(100))


def test_decimation_is_deterministic_sorted_and_capped() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 50, size=(20_000, 3))
    idx, voxel = voxel_decimate_indices(pts, 3_000)
    idx2, voxel2 = voxel_decimate_indices(pts.copy(), 3_000)
    np.testing.assert_array_equal(idx, idx2)
    assert voxel == voxel2
    assert 0 < len(idx) <= 3_000
    assert np.all(np.diff(idx) > 0)  # ascending original order, no duplicates
    k = round(np.log(voxel / VOXEL_START_MM) / np.log(VOXEL_GROWTH))
    assert voxel == pytest.approx(VOXEL_START_MM * VOXEL_GROWTH**k)
    # one point per occupied voxel
    keys = np.floor(pts[idx] / voxel).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == len(idx)
    # and every occupied voxel is represented
    assert len(np.unique(np.floor(pts / voxel).astype(np.int64), axis=0)) == len(idx)


def test_decimation_keeps_point_nearest_the_voxel_centroid() -> None:
    # All points inside one 0.5 mm voxel; the centroid is (0.2, 0.2, 0.2).
    pts = np.array([[0.01, 0.01, 0.01], [0.21, 0.19, 0.2], [0.38, 0.4, 0.39]])
    idx, voxel = voxel_decimate_indices(pts, 1)
    assert voxel == VOXEL_START_MM
    np.testing.assert_array_equal(idx, [1])
    # a tie keeps the lowest index
    tie = np.array([[0.125, 0.125, 0.125], [0.375, 0.375, 0.375]])
    np.testing.assert_array_equal(voxel_decimate_indices(tie, 1)[0], [0])


def test_read_deviations_rejects_misalignment_and_infinity(tmp_path: Path) -> None:
    path = tmp_path / "d.f32"
    path.write_bytes(np.array([0.1, np.nan, -0.2], dtype="<f4").tobytes())
    assert len(read_deviations(path, 3)) == 3
    with pytest.raises(ValueError, match="not index-aligned"):
        read_deviations(path, 4)
    path.write_bytes(np.array([0.1, np.inf, -0.2], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="infinities"):
        read_deviations(path, 3)
    path.write_bytes(b"\x00" * 7)
    with pytest.raises(ValueError, match="multiple of 4"):
        read_deviations(path, 1)


# ---------------------------------------------------------------- build_viewer_assets
def _build(tmp_path: Path, scenario: str = "final", cap: int = 400_000, mutate=None):
    fake = build_fake_engine_output(tmp_path / "ws", scenario=scenario)
    if mutate is not None:
        mutate(fake)
    result = load_result(fake.result_path)
    out = tmp_path / "ws" / "result"
    return fake, result, out, build_viewer_assets(result, fake.engine_dir, out, cap_points=cap)


def test_viewer_copies_are_index_aligned_after_decimation(tmp_path: Path) -> None:
    fake, result, out, build = _build(tmp_path, cap=5_000)
    va = build.assets
    assert build.source_points == fake.point_count and build.kept_points == va.point_count <= 5_000
    assert build.voxel_mm is not None
    scan = read_ply(out / va.registered_scan).vertices
    dev = np.fromfile(out / va.deviations, dtype="<f4")
    assert len(dev) == len(scan) == va.point_count
    # every viewer point carries the deviation of the very same engine point
    engine_pts = read_ply(fake.engine_dir / "registered_scan.ply").vertices.astype(np.float32)
    engine_dev = np.fromfile(fake.engine_dir / "deviations.f32", dtype="<f4")
    lookup = {tuple(p): d for p, d in zip(map(tuple, engine_pts), engine_dev)}
    for p, d in zip(map(tuple, scan.astype(np.float32)), dev):
        expected = lookup[p]
        assert (np.isnan(expected) and np.isnan(d)) or expected == d
    assert va.deviation_reference_stage == result.stage_decision.stage_model_id
    assert va.heatmap_unavailable_reason is None and va.segment_labels is None
    assert va.stage_mesh == "stage_mesh.ply" and va.final_mesh == "final_mesh.ply"
    assert read_ply(out / "stage_mesh.ply", require_faces=True).faces is not None
    assert BendGeometryFileV1.model_validate_json((out / va.bend_geometry).read_bytes()) == BendGeometryFileV1.from_result(result)
    assert build.warnings == ()


def test_viewer_build_is_byte_deterministic(tmp_path: Path) -> None:
    _, result, out, _ = _build(tmp_path, cap=4_000)
    fake2 = tmp_path / "ws"
    out2 = tmp_path / "out2"
    build_viewer_assets(result, fake2 / "engine", out2, cap_points=4_000)
    for name in ("viewer_scan.ply", "viewer_deviations.f32", "stage_mesh.ply", "final_mesh.ply", "bend_geometry.json"):
        assert (out / name).read_bytes() == (out2 / name).read_bytes(), name


def test_misaligned_deviations_drop_the_heatmap_not_the_scan(tmp_path: Path) -> None:
    def truncate(fake):
        path = fake.engine_dir / "deviations.f32"
        path.write_bytes(path.read_bytes()[:-4])

    _, _, out, build = _build(tmp_path, mutate=truncate)
    va = build.assets
    assert va.deviations is None and va.deviation_reference_stage is None
    assert va.heatmap_unavailable_reason == "VIEWER_DEVIATIONS_INVALID"
    assert va.registered_scan == "viewer_scan.ply"
    assert not (out / "viewer_deviations.f32").exists()
    assert any(w.startswith("VIEWER_DEVIATIONS_INVALID") for w in build.warnings)


def test_infinite_deviation_is_never_published(tmp_path: Path) -> None:
    def poison(fake):
        dev = np.fromfile(fake.engine_dir / "deviations.f32", dtype="<f4")
        dev[3] = np.inf
        (fake.engine_dir / "deviations.f32").write_bytes(dev.tobytes())

    _, _, out, build = _build(tmp_path, mutate=poison)
    assert build.assets.deviations is None and not (out / "viewer_deviations.f32").exists()


def test_missing_scan_drops_scan_and_heatmap_with_reason(tmp_path: Path) -> None:
    _, _, out, build = _build(tmp_path, mutate=lambda f: (f.engine_dir / "registered_scan.ply").unlink())
    va = build.assets
    assert va.registered_scan is None and va.point_count is None and va.deviations is None
    assert va.heatmap_unavailable_reason == "VIEWER_SCAN_UNAVAILABLE"
    assert any(w.startswith("VIEWER_SCAN_MISSING") for w in build.warnings)
    assert not (out / "viewer_scan.ply").exists()


def test_point_count_mismatch_rejects_the_scan(tmp_path: Path) -> None:
    def shrink(fake):
        pts = read_ply(fake.engine_dir / "registered_scan.ply").vertices
        write_points_ply(fake.engine_dir / "registered_scan.ply", pts[:-1])

    _, _, _, build = _build(tmp_path, mutate=shrink)
    assert build.assets.registered_scan is None
    assert any("result says" in w for w in build.warnings)


def test_corrupt_stage_mesh_is_dropped_with_warning(tmp_path: Path) -> None:
    def corrupt(fake):
        path = fake.engine_dir / "display" / "stage_mesh.ply"
        path.write_bytes(path.read_bytes()[:-10])

    _, _, out, build = _build(tmp_path, mutate=corrupt)
    assert build.assets.stage_mesh is None and build.assets.final_mesh == "final_mesh.ply"
    assert not (out / "stage_mesh.ply").exists()
    assert any(w.startswith("VIEWER_MESH_INVALID") for w in build.warnings)


def test_engine_bend_geometry_disagreeing_with_rows_is_regenerated(tmp_path: Path) -> None:
    def tamper(fake):
        path = fake.engine_dir / "bend_geometry.json"
        data = json.loads(path.read_text())
        data["bends"][0]["label_anchor_mm"] = [999.0, 999.0, 999.0]
        path.write_text(json.dumps(data))

    _, result, out, build = _build(tmp_path, mutate=tamper)
    assert any(w.startswith("VIEWER_BEND_GEOMETRY_REGENERATED") for w in build.warnings)
    geo = BendGeometryFileV1.model_validate_json((out / "bend_geometry.json").read_bytes())
    assert geo == BendGeometryFileV1.from_result(result)


def test_stale_viewer_files_are_removed(tmp_path: Path) -> None:
    out = tmp_path / "ws" / "result"
    out.mkdir(parents=True)
    (out / "viewer_deviations.f32").write_bytes(b"stale")

    def truncate(fake):
        path = fake.engine_dir / "deviations.f32"
        path.write_bytes(path.read_bytes()[:-4])

    _, _, out, build = _build(tmp_path, mutate=truncate)
    assert build.assets.deviations is None
    assert not (out / "viewer_deviations.f32").exists()


def test_invalid_inspection_publishes_no_scan(tmp_path: Path) -> None:
    _, result, out, build = _build(tmp_path, scenario="invalid")
    va = build.assets
    assert va.registered_scan is None and va.deviations is None
    assert va.heatmap_unavailable_reason == result.viewer_assets.heatmap_unavailable_reason
    assert va.final_mesh == "final_mesh.ply" and va.stage_mesh is None
    assert build.warnings == ()
