"""E2 virtual ZEISS-like scanner: occlusion, noise and artefact statistics, I/O."""
from __future__ import annotations

import json

import numpy as np
import pytest

from domains.production.engine.synthetic import brep
from domains.production.engine.synthetic.families import FeatureRequest, build_family
from domains.production.engine.synthetic.geometry import FoldedPart, FormState
from domains.production.engine.synthetic.ply import read_ply, write_ply
from domains.production.engine.synthetic.scanner import (
    LABEL_OUTLIER,
    SCAN_PLANS,
    ScanPlan,
    ScanSetup,
    ScannerModel,
    Viewpoint,
    default_ray_threads,
    median_nn_spacing,
    random_pose,
    scan_surface,
)

QUIET = dict(
    noise_sigma_mm=0.0, noise_incidence_gain=0.0, edge_flare_probability=0.0, edge_erosion_probability=0.0,
    specular_dropout_probability=0.0, random_dropout_probability=0.0, outlier_fraction=0.0,
    duplicate_skin_offset_mm=0.0, duplicate_skin_rotation_deg=0.0, view_jitter_mm=0.0,
)


def _surface(family="L_BRACKET", formed=None, seed=4):
    d = build_family(family, np.random.default_rng(seed), FeatureRequest())
    fp = FoldedPart(d, FormState.nominal(d, formed))
    return fp, brep.tessellate(brep.build_pieces(fp), 0.005, 0.08)


def _single(elev=80.0, az=10.0):
    return ScanPlan("T", (ScanSetup("TOP", (Viewpoint(az, elev),)),))


def test_hidden_bottom_skin_is_never_captured_from_above():
    fp, surf = _surface(formed=())  # flat blank, skin 0 = +Z (identity placement)
    assert fp.contract_skin0_is_flat_skin0()
    res = scan_surface(surf, ScannerModel(), SCAN_PLANS["TOP_ONLY"], np.random.default_rng(1), spacing_mm=1.0)
    real = ~res.outlier
    assert real.sum() > 1000
    assert int(np.sum(res.skin[real] == 1)) == 0  # back faces: none
    assert int(np.sum(res.skin[real] == 0)) > 0.9 * real.sum()


def test_every_point_is_the_first_hit_from_its_camera():
    import open3d as o3d

    _, surf = _surface("U_CHANNEL")
    pose = random_pose(np.random.default_rng(3))
    raw = surf.transformed(pose)
    res = scan_surface(raw, ScannerModel(**QUIET), SCAN_PLANS["FULL"], np.random.default_rng(2), spacing_mm=1.2)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(raw.vertices.astype(np.float32)), o3d.core.Tensor(raw.triangles.astype(np.uint32)))
    cams = np.asarray(res.stats["cameras_raw"])[res.view]
    d = res.ideal_raw - cams
    dist = np.linalg.norm(d, axis=1)
    rays = np.hstack([cams, d / dist[:, None]]).astype(np.float32)
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    assert np.all(np.abs(t_hit - dist) < 2e-3), "a captured point is occluded from its own camera"
    # idealised capture: points lie on the surface exactly
    assert np.allclose(res.points, res.ideal_raw, atol=1e-9)


def test_range_noise_matches_the_model():
    fp, surf = _surface(formed=())
    model = ScannerModel(**{**QUIET, "noise_sigma_mm": 0.05, "triangulation_angle_deg": 0.0})
    res = scan_surface(surf, model, _single(), np.random.default_rng(5), spacing_mm=0.6)
    sel = (res.skin == 0) & (res.seg_kind == 0)
    pts = res.points[sel]
    cam = np.asarray(res.stats["cameras_raw"][0])
    n = np.array([0.0, 0.0, 1.0])
    cos_t = np.abs((res.ideal_raw[sel] - cam) @ n) / np.linalg.norm(res.ideal_raw[sel] - cam, axis=1)
    resid = (pts[:, 2] - fp.design.thickness) / (0.05 * cos_t)
    assert len(resid) > 20000
    assert abs(float(resid.mean())) < 0.05
    assert float(resid.std()) == pytest.approx(1.0, abs=0.03)


def test_grazing_incidence_drops_points():
    _, surf = _surface(formed=())
    model = ScannerModel(**{**QUIET, "grazing_start_deg": 30.0, "grazing_cutoff_deg": 40.0, "triangulation_angle_deg": 0.0})
    steep = scan_surface(surf, model, _single(elev=85.0), np.random.default_rng(0), spacing_mm=1.0)
    with pytest.raises(ValueError, match="no points"):
        scan_surface(surf, model, _single(elev=42.0), np.random.default_rng(0), spacing_mm=1.0)
    assert steep.stats["points"] > 1000


def test_edge_flare_pushes_boundary_points_away_from_the_camera():
    _, surf = _surface(formed=())
    model = ScannerModel(**{**QUIET, "edge_flare_probability": 1.0, "edge_flare_mm": 0.5})
    res = scan_surface(surf, model, _single(), np.random.default_rng(7), spacing_mm=0.8)
    assert res.flare.sum() > 50
    cam = np.asarray(res.stats["cameras_raw"][0])
    ray = res.ideal_raw - cam
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    push = np.einsum("ij,ij->i", res.points - res.ideal_raw, ray)
    assert np.all(push[res.flare] >= -1e-9) and np.all(push[res.flare] <= 0.5 + 1e-9)
    assert np.allclose(push[~res.flare], 0.0, atol=1e-9)
    off = scan_surface(surf, ScannerModel(**QUIET), _single(), np.random.default_rng(7), spacing_mm=0.8)
    assert not off.flare.any()


def test_duplicate_skin_offset_is_a_rigid_setup_misregistration():
    _, surf = _surface("Z_STEP")
    raw = surf.transformed(random_pose(np.random.default_rng(11)))
    model = ScannerModel(**{**QUIET, "duplicate_skin_offset_mm": 0.3, "duplicate_skin_fraction": 0.5})
    res = scan_surface(raw, model, SCAN_PLANS["FULL"], np.random.default_rng(4), spacing_mm=1.5)
    delta = res.points - res.ideal_raw
    s0, s1 = res.setup == 0, res.setup == 1
    assert s0.sum() > 0 and s1.sum() > 0
    assert np.allclose(delta[s0], 0.0, atol=1e-9)
    assert np.allclose(delta[s1], delta[s1][0], atol=1e-9)  # one translation for the whole setup
    assert 0.0 < float(np.linalg.norm(delta[s1][0])) < 2.0


def test_outliers_are_labelled_and_counted():
    _, surf = _surface()
    model = ScannerModel(**{**QUIET, "outlier_fraction": 0.01})
    res = scan_surface(surf, model, SCAN_PLANS["TOP_ONLY"], np.random.default_rng(9), spacing_mm=1.0)
    n_out = int(res.outlier.sum())
    assert n_out == round(0.01 * (len(res.points) - n_out))
    assert np.all(res.seg_kind[res.outlier] == LABEL_OUTLIER) and np.all(np.isnan(res.ideal_raw[res.outlier]))
    assert np.all(res.view[res.outlier] == -1)


def test_scanner_is_deterministic_by_seed():
    _, surf = _surface("HAT")
    a = scan_surface(surf, ScannerModel(), SCAN_PLANS["FULL"], np.random.default_rng(21), spacing_mm=1.5)
    b = scan_surface(surf, ScannerModel(), SCAN_PLANS["FULL"], np.random.default_rng(21), spacing_mm=1.5)
    c = scan_surface(surf, ScannerModel(), SCAN_PLANS["FULL"], np.random.default_rng(22), spacing_mm=1.5)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.seg_index, b.seg_index)
    assert not (a.points.shape == c.points.shape and np.array_equal(a.points, c.points))


def test_more_occlusion_means_fewer_points():
    _, surf = _surface("CHANNEL_WITH_RETURNS")
    raw = surf.transformed(random_pose(np.random.default_rng(2)))
    counts = [
        scan_surface(raw, ScannerModel(), SCAN_PLANS[p], np.random.default_rng(1), spacing_mm=1.5).stats["points"]
        for p in ("FULL", "TOP_ONLY", "SPARSE", "SINGLE")
    ]
    assert counts == sorted(counts, reverse=True) and counts[-1] < 0.5 * counts[0]


def test_mesh_faces_face_their_camera():
    _, surf = _surface()
    res = scan_surface(surf, ScannerModel(**QUIET), _single(), np.random.default_rng(3), spacing_mm=1.0, output="mesh")
    f = res.faces
    assert f is not None and len(f) > 1000 and f.min() >= 0 and f.max() < len(res.points)
    v = res.points
    normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    cam = np.asarray(res.stats["cameras_raw"][0])
    to_cam = cam - v[f].mean(axis=1)
    assert np.mean(np.einsum("ij,ij->i", normal, to_cam) > 0) > 0.99


def test_scanner_model_json_and_validation(tmp_path):
    m = ScannerModel(noise_sigma_mm=0.031, provenance="MODEL_CARD", name="card")
    p = tmp_path / "model.json"
    m.to_json(p)
    assert ScannerModel.from_json(p) == m
    doc = json.loads(p.read_text())
    for bad in ({"noise_sigma_mm": -0.1}, {"grazing_start_deg": 85.0}, {"outlier_fraction": 1.5}, {"provenance": "GUESS"},
                {"point_spacing_mm": 0.0}):
        with pytest.raises(ValueError):
            ScannerModel.from_dict({**doc, **bad})
    with pytest.raises(ValueError, match="unknown"):
        ScannerModel.from_dict({**doc, "colour": "red"})
    # a measured card states every field: nothing silently falls back to an assumed default
    partial = {k: v for k, v in doc.items() if k != "duplicate_skin_fraction"}
    with pytest.raises(ValueError, match="must state every field"):
        ScannerModel.from_dict(partial)
    assert ScannerModel.from_dict({**partial, "provenance": "ASSUMED"}).duplicate_skin_fraction == ScannerModel().duplicate_skin_fraction
    assert m.sha256() == ScannerModel.from_json(p).sha256() != ScannerModel().sha256()
    for bad in ({"duplicate_skin_fraction": 1.2}, {"duplicate_skin_patch_mm": 0.0}):
        with pytest.raises(ValueError):
            ScannerModel.from_dict({**doc, **bad})
    with pytest.raises(ValueError, match="schema"):
        ScannerModel.from_dict({**doc, "schema": "other"})
    plan = SCAN_PLANS["FULL"]
    assert ScanPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    with pytest.raises(ValueError):
        ScanPlan.from_dict({"name": "x", "setups": []})


def test_random_pose_is_a_proper_rigid_transform():
    for s in range(20):
        m = random_pose(np.random.default_rng(s))
        r = m[:3, :3]
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-12) and np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_ply_roundtrip_and_rejections(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 3)) * 50
    faces = rng.integers(0, 100, size=(40, 3))
    h1 = write_ply(tmp_path / "a.ply", pts, faces, comments=("units mm",))
    h2 = write_ply(tmp_path / "b.ply", pts, faces, comments=("units mm",))
    assert h1 == h2
    v, f, comments = read_ply(tmp_path / "a.ply")
    assert np.allclose(v, pts.astype(np.float32), atol=0) and np.array_equal(f, faces) and comments == ["units mm"]
    v2, f2, _ = read_ply(tmp_path / "pts.ply") if write_ply(tmp_path / "pts.ply", pts) else (None, None, None)
    assert f2 is None and len(v2) == 100
    with pytest.raises(ValueError):
        write_ply(tmp_path / "bad.ply", np.array([[np.nan, 0, 0]]))
    with pytest.raises(ValueError):
        write_ply(tmp_path / "bad.ply", pts, np.array([[0, 1, 100]]))
    header = (tmp_path / "a.ply").read_bytes().split(b"end_header")[0].decode()
    assert "binary_little_endian 1.0" in header and "property list uchar int vertex_indices" in header


def test_flare_only_at_true_depth_discontinuities():
    """Dropout holes are not edges: with heavy random dropout and no real edge in a
    view's interior, flare must stay at the plate's silhouette."""
    fp, surf = _surface(formed=())
    model = ScannerModel(**{**QUIET, "random_dropout_probability": 0.3, "edge_flare_probability": 1.0, "edge_flare_mm": 0.3,
                            "triangulation_angle_deg": 0.0})
    res = scan_surface(surf, model, _single(), np.random.default_rng(8), spacing_mm=0.8)
    assert res.flare.any()
    # distance of flare points (ideal) to the blank outline in the flat frame (identity placement)
    from domains.production.engine.synthetic.flat import flat_pattern

    outline = flat_pattern(fp.design).outline
    p = res.ideal_raw[res.flare][:, :2]
    seg_a, seg_b = outline, np.roll(outline, -1, axis=0)
    ab = seg_b - seg_a
    tt = np.clip(np.einsum("nij,ij->ni", p[:, None, :] - seg_a[None], ab) / np.einsum("ij,ij->i", ab, ab)[None], 0, 1)
    dist = np.linalg.norm(p[:, None, :] - (seg_a[None] + tt[..., None] * ab[None]), axis=2).min(axis=1)
    holes = [(np.array(c), r) for c, r in ((h[:2], h[2]) for h in flat_pattern(fp.design).holes)]
    for c, r in holes:  # hole rims are real edges too
        dist = np.minimum(dist, np.abs(np.linalg.norm(p - c, axis=1) - r))
    spacing = 0.8 * 0.85 * 1.2  # ray spacing incl. perspective margin
    assert float(np.max(dist)) < 2.5 * spacing


def _voxel_setups(res, voxel):
    keys = np.floor(res.ideal_raw[~res.outlier] / voxel).astype(np.int64)
    setups = res.setup[~res.outlier]
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    both = np.zeros(inv.max() + 1, dtype=np.int64)
    for s_id in (0, 1):
        present = np.zeros(inv.max() + 1, dtype=bool)
        present[inv[setups == s_id]] = True
        both += present
    return int(np.sum(both > 1))


def test_setups_merge_into_one_surface_except_duplicate_skin_patches():
    """Overlapping setups do not double the surface (one owner per voxel), except in
    the model's duplicate-skin patches, where both setups' (offset) points survive."""
    _, surf = _surface("U_CHANNEL")
    raw = surf.transformed(random_pose(np.random.default_rng(3)))
    voxel = 1.0
    merged = scan_surface(raw, ScannerModel(**{**QUIET, "duplicate_skin_fraction": 0.0}), SCAN_PLANS["FULL"],
                          np.random.default_rng(2), spacing_mm=voxel)
    assert merged.stats["cross_setup_dropped"] > 1000 and merged.stats["duplicate_skin_points"] == 0
    assert _voxel_setups(merged, voxel) == 0
    doubled = scan_surface(raw, ScannerModel(**{**QUIET, "duplicate_skin_fraction": 1.0}), SCAN_PLANS["FULL"],
                           np.random.default_rng(2), spacing_mm=voxel)
    assert doubled.stats["cross_setup_dropped"] == 0 and doubled.stats["duplicate_skin_points"] > 1000
    assert _voxel_setups(doubled, voxel) > 1000
    some = scan_surface(raw, ScannerModel(**{**QUIET, "duplicate_skin_fraction": 0.3}), SCAN_PLANS["FULL"],
                        np.random.default_rng(2), spacing_mm=voxel)
    frac = some.stats["duplicate_skin_points"] / doubled.stats["duplicate_skin_points"]
    assert 0.1 < frac < 0.55  # coherent patches: about the requested share of the overlap
    assert merged.stats["points"] < some.stats["points"] < doubled.stats["points"]


def test_point_order_carries_no_label():
    """Outliers are not a contiguous tail and setups/views interleave in file order;
    mesh faces are re-indexed consistently."""
    _, surf = _surface("HAT")
    raw = surf.transformed(random_pose(np.random.default_rng(1)))
    res = scan_surface(raw, ScannerModel(**{**QUIET, "outlier_fraction": 0.01}), SCAN_PLANS["FULL"], np.random.default_rng(4),
                       spacing_mm=1.5, output="mesh")
    out_idx = np.flatnonzero(res.outlier)
    assert len(out_idx) > 20 and out_idx[-1] - out_idx[0] > 10 * len(out_idx)
    assert not np.all(out_idx == np.arange(len(res.points) - len(out_idx), len(res.points)))
    real = ~res.outlier
    assert not np.all(np.diff(res.setup[real].astype(int)) >= 0) and not np.all(np.diff(res.view[real].astype(int)) >= 0)
    # faces still connect points of one view (the range grid of that view) and face their camera
    f = res.faces
    assert len(f) > 1000 and np.all(res.view[f[:, 0]] == res.view[f[:, 1]]) and np.all(res.view[f[:, 1]] == res.view[f[:, 2]])
    assert not np.any(res.outlier[f.ravel()])
    v = res.points
    normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    cams = np.asarray(res.stats["cameras_raw"])[res.view[f[:, 0]]]
    assert np.mean(np.einsum("ij,ij->i", normal, cams - v[f].mean(axis=1)) > 0) > 0.99


def test_ray_thread_count_does_not_change_the_scan(monkeypatch):
    _, surf = _surface("TWO_DIRECTION_TAB")
    raw = surf.transformed(random_pose(np.random.default_rng(6)))
    runs = [scan_surface(raw, ScannerModel(), SCAN_PLANS["FULL"], np.random.default_rng(9), spacing_mm=1.2, ray_threads=t)
            for t in (1, 4, 0)]
    for r in runs[1:]:
        assert np.array_equal(r.points, runs[0].points) and np.array_equal(r.seg_index, runs[0].seg_index)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert default_ray_threads() == 3
    monkeypatch.setenv("OMP_NUM_THREADS", "not-a-number")
    assert default_ray_threads() == 0
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert default_ray_threads() == 0
    with pytest.raises(ValueError):
        scan_surface(raw, ScannerModel(), SCAN_PLANS["SINGLE"], np.random.default_rng(9), spacing_mm=1.2, ray_threads=-1)


def test_median_nn_spacing_statistic():
    g = np.stack(np.meshgrid(np.arange(200) * 0.5, np.arange(300) * 0.5, indexing="ij"), axis=-1).reshape(-1, 2)
    pts = np.column_stack([g, np.zeros(len(g))])
    perm = np.random.default_rng(0).permutation(len(pts))
    assert median_nn_spacing(pts[perm]) == pytest.approx(0.5, abs=1e-12)
    # doubling every point (a full duplicate skin 0.05 mm away) halves nothing but the NN of the doubled points
    doubled = np.vstack([pts, pts + np.array([0.0, 0.0, 0.05])])[np.random.default_rng(1).permutation(2 * len(pts))]
    assert median_nn_spacing(doubled) == pytest.approx(0.05, abs=1e-9)
    with pytest.raises(ValueError):
        median_nn_spacing(pts[:1])
