"""E1 compile: stage models S_k (Method A), sample clouds, display meshes, determinism."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from contracts.production.canonical import canonical_json_bytes
from contracts.production.compiled_part import SAMPLE_CLOUD_ARRAYS, CompiledPartV1
from domains.production.engine.compile import compile_part, stage_transforms_from_compiled
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.geometry import apply4

K = 0.33


@pytest.fixture(scope="module")
def hat(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hat_stages")
    shape, info = tp.hat_part(width=30.0)
    step = tp.write_step(shape, tmp / "hat.step")
    recipe = dev_recipe(step, tp.candidates_from_info(info), sku="SYN-HATS", inspection_stage="IN_PROCESS")
    out = tmp / "out"
    compiled = compile_part(recipe, step, out).compiled
    assert compiled.status == "COMPILED"
    return info, recipe, compiled, out, step


def _m(t) -> np.ndarray:
    return np.array(t, dtype=np.float64)


def test_one_prefix_stage_per_k_with_the_first_k_bends_formed(hat):
    _, _, c, _, _ = hat
    uids = [b.bend_uid for b in c.bends]
    assert [s.stage_model_id for s in c.stages] == [f"S{k}" for k in range(len(uids) + 1)]
    for k, s in enumerate(c.stages):
        assert s.kind == "SEQUENCE_PREFIX" and list(s.formed_bend_uids) == uids[:k]
        assert s.method == "METHOD_A_ROTATE_BACK" and s.k_factor_used == K
    final = c.stages[-1]
    for t in final.panel_transforms.values():
        assert _m(t) == pytest.approx(np.eye(4), abs=1e-12)


def test_flat_stage_is_planar_and_has_the_developed_length(hat):
    info, _, c, out, _ = hat
    s0 = c.stages[0]
    data = np.load(out / s0.sample_cloud.rel_path, allow_pickle=False)
    pts, nrm, skin = data["points"].astype(np.float64), data["normals"].astype(np.float64), data["skin"]
    datum = next(p for p in c.panels if p.is_datum)
    n0 = np.array(datum.skin0_plane.normal)
    p0 = np.array(datum.skin0_plane.point_mm)
    on_skin = skin != 2
    height = (pts[on_skin] - p0) @ n0
    # in the flat blank every skin point lies on one of the two skin planes (0 and -t)
    assert np.all((np.abs(height) < 1e-3) | (np.abs(height + info.thickness) < 1e-3))
    assert np.all(np.abs(np.abs(nrm[on_skin] @ n0) - 1.0) < 1e-4)
    # developed length = sum of OML lengths - bend deductions (analytic flat)
    _, _, flat_length = tp.profile_flat(info, K)
    across = np.cross(n0, [0.0, 1.0, 0.0])
    extent = float(np.ptp(pts @ across))
    assert extent == pytest.approx(flat_length, abs=0.6)  # sampling grid: within one spacing
    # the compiled stage bend lines of unformed bends are the strip centre lines of that flat
    xs = sorted(float(0.5 * (_m(ln.start_mm) + _m(ln.end_mm)) @ across) for ln in s0.bend_lines.values())
    _, lines, _ = tp.profile_flat(info, K)
    expected = sorted(float(a[0]) for a, _ in lines)
    assert np.allclose(np.diff(xs), np.diff(expected), atol=1e-6)


def test_formed_bend_lines_are_the_oml_sharp_and_never_the_cylinder_axis(hat):
    _, _, c, _, _ = hat
    final = c.stages[-1]
    for b in c.bends:
        line = final.bend_lines[b.bend_uid]
        assert line.start_mm == b.loci.oml_virtual_sharp.start_mm
        axis = _m(b.loci.cylinder_axis.start_mm)
        assert np.linalg.norm(_m(line.start_mm) - axis) > b.inner_radius_mm + b.thickness_mm


def test_unformed_strip_has_the_neutral_length_and_moves_the_child(hat):
    _, _, c, _, _ = hat
    s0, final = c.stages[0], c.stages[-1]
    b = c.bends[0]
    t_parent = _m(s0.panel_transforms[b.parent_panel_id])
    t_child = _m(s0.panel_transforms[b.child_panel_id])
    child_tangent = _m(b.loci.outer_tangent_child.start_mm)
    parent_tangent = _m(b.loci.outer_tangent_parent.start_mm)
    # the child's tangent line lands one neutral strip beyond the parent's, in the parent's plane
    gap = np.linalg.norm(apply4(t_child, child_tangent) - apply4(t_parent, parent_tangent))
    assert gap == pytest.approx(b.neutral_length_mm, abs=1e-6)
    assert b.neutral_length_mm == pytest.approx(math.radians(b.sweep_deg) * (b.inner_radius_mm + K * b.thickness_mm),
                                                abs=1e-6)
    assert final.bend_lines[b.bend_uid] != s0.bend_lines[b.bend_uid]


def test_transforms_are_reproducible_from_the_compiled_part_alone(hat):
    _, _, c, _, _ = hat
    for s in c.stages:
        recomputed = stage_transforms_from_compiled(c, s.formed_bend_uids)
        for pid, t in s.panel_transforms.items():
            assert recomputed[pid] == pytest.approx(_m(t), abs=1e-5)
    # an out-of-order formed set (only the last bend) is also derivable
    last = c.bends[-1]
    t = stage_transforms_from_compiled(c, [last.bend_uid])
    assert t[last.child_panel_id] == pytest.approx(t[last.parent_panel_id], abs=1e-12)


def test_sample_clouds_have_the_contract_arrays_and_labels(hat):
    _, _, c, out, _ = hat
    for s in c.stages:
        path = out / s.sample_cloud.rel_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == s.sample_cloud.sha256
        data = np.load(path, allow_pickle=False)
        assert tuple(sorted(data.files)) == tuple(sorted(SAMPLE_CLOUD_ARRAYS))
        assert data["points"].dtype == np.float32 and data["normals"].dtype == np.float32
        assert data["segment_kind"].dtype == np.uint8 and data["skin"].dtype == np.uint8
        assert data["segment_index"].dtype == np.int32
        m = len(data["points"])
        assert m == s.sample_cloud.point_count and all(len(data[k]) == m for k in data.files)
        assert np.allclose(np.linalg.norm(data["normals"], axis=1), 1.0, atol=1e-5)
        kind, idx, skin = data["segment_kind"], data["segment_index"], data["skin"]
        assert set(np.unique(kind)) <= {0, 1, 2} and set(np.unique(skin)) <= {0, 1, 2}
        assert np.all(idx[kind == 1] < len(c.bends)) and np.all(idx[kind != 1] < len(c.panels))
        assert np.all(idx >= 0) and np.all((kind == 2) <= (skin == 2))


def test_final_stage_points_lie_on_the_cad_surfaces(hat):
    _, _, c, out, _ = hat
    data = np.load(out / c.stages[-1].sample_cloud.rel_path, allow_pickle=False)
    pts, kind, idx, skin = data["points"].astype(np.float64), data["segment_kind"], data["segment_index"], data["skin"]
    for i, panel in enumerate(c.panels):
        for s, plane in ((0, panel.skin0_plane), (1, panel.skin1_plane)):
            sel = (kind == 0) & (idx == i) & (skin == s)
            assert sel.sum() > 100
            d = (pts[sel] - np.array(plane.point_mm)) @ np.array(plane.normal)
            assert np.max(np.abs(d)) < 1e-3
    for j, b in enumerate(c.bends):
        axis_p = np.array(b.axis.point_mm)
        axis_d = np.array(b.axis.direction)
        sel = (kind == 1) & (idx == j) & (skin != 2)
        rel = pts[sel] - axis_p
        radial = np.linalg.norm(rel - np.outer(rel @ axis_d, axis_d), axis=1)
        r_in, r_out = b.inner_radius_mm, b.inner_radius_mm + b.thickness_mm
        assert np.all(np.minimum(np.abs(radial - r_in), np.abs(radial - r_out)) < 1e-3)
    # spacing on a planar panel is ~0.5 mm
    from scipy.spatial import cKDTree

    sel = (kind == 0) & (idx == 0) & (skin == 0)
    d, _ = cKDTree(pts[sel]).query(pts[sel], k=2)
    assert np.median(d[:, 1]) == pytest.approx(0.5, abs=0.01)


def test_display_meshes_are_binary_ply_with_the_declared_counts(hat):
    _, _, c, out, _ = hat
    for s in c.stages:
        raw = (out / s.display_mesh.rel_path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == s.display_mesh.sha256
        head, _, body = raw.partition(b"end_header\n")
        text = head.decode("ascii")
        assert "format binary_little_endian 1.0" in text
        assert f"element vertex {s.display_mesh.vertex_count}" in text
        assert f"element face {s.display_mesh.face_count}" in text
        assert "property list uchar int vertex_indices" in text
        assert len(body) == s.display_mesh.vertex_count * 12 + s.display_mesh.face_count * 13
        verts = np.frombuffer(body[: s.display_mesh.vertex_count * 12], dtype="<f4").reshape(-1, 3)
        assert np.isfinite(verts).all()


def test_compile_is_deterministic_for_identical_inputs(hat, tmp_path):
    _, recipe, c, out, step = hat
    again = compile_part(recipe, step, tmp_path / "again").compiled
    assert again.compiled_part_sha256 == c.compiled_part_sha256
    for s in c.stages:
        assert (tmp_path / "again" / s.sample_cloud.rel_path).read_bytes() == (out / s.sample_cloud.rel_path).read_bytes()


def test_compiled_part_json_round_trips_and_validates(hat):
    _, _, c, out, _ = hat
    raw = (out / "compiled_part.json").read_bytes()
    assert raw == canonical_json_bytes(c)
    reloaded = CompiledPartV1.model_validate_json(raw)
    assert reloaded.compiled_part_sha256 == c.compiled_part_sha256 == reloaded.compute_hash()


def test_a_wrong_unfold_is_caught_by_the_flat_blank_self_check(hat, tmp_path, monkeypatch):
    from domains.production.engine.compile import stages

    _, recipe, _, _, step = hat
    real = stages.bend_unfold_matrix

    def wrong_sign(axis_point, axis_dir, sweep_rad, p_out, neutral_length):
        return real(axis_point, axis_dir, -sweep_rad, p_out, neutral_length)  # folds further instead of back

    monkeypatch.setattr(stages, "bend_unfold_matrix", wrong_sign)
    c = compile_part(recipe, step, tmp_path / "wrong").compiled
    assert c.status == "BLOCKED" and c.blocker_codes == ("STAGE_MODEL_FAILED",)
    assert "S0 is not flat" in next(i.detail for i in c.issues if i.code == "STAGE_MODEL_FAILED")
    assert c.diagnostics["s0_flatness_max_mm"] > 1.0


def test_stage_clouds_are_reproducible_from_the_final_cloud_and_the_compiled_part(hat):
    from domains.production.engine.compile import stage_points_from_compiled

    _, _, c, out, _ = hat
    final = np.load(out / c.stages[-1].sample_cloud.rel_path, allow_pickle=False)
    for s in c.stages:
        ref = np.load(out / s.sample_cloud.rel_path, allow_pickle=False)
        pts, nrm = stage_points_from_compiled(c, final["points"], final["normals"], final["segment_kind"],
                                              final["segment_index"], s.formed_bend_uids)
        assert np.max(np.abs(pts - ref["points"].astype(np.float64))) < 2e-3  # float32 + 6-decimal transforms
        assert np.max(np.abs(nrm - ref["normals"].astype(np.float64))) < 1e-3
    # an out-of-order hypothesis (only the last bend formed) keeps that bend's panels rigidly together
    last = c.bends[-1]
    pts, _ = stage_points_from_compiled(c, final["points"], final["normals"], final["segment_kind"],
                                        final["segment_index"], [last.bend_uid])
    assert np.isfinite(pts).all()
