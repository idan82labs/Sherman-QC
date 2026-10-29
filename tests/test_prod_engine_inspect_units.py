"""Unit tests of the per-scan engine (E3) building blocks: no OCP, no E1/E2 geometry.

Run: PYT tests/test_prod_engine_inspect_units.py -q
"""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from domains.production.engine.inspect import cloud, geom
from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config
from domains.production.engine.inspect.decode import _raw_state
from domains.production.engine.inspect.metrology import Measurement, robust_location, u_budget
from domains.production.engine.inspect.registration import mean_normal
from domains.production.engine.inspect.scan_io import ScanReadError, read_scan


# --------------------------------------------------------------------------- scan ingest
def _ply_binary(points: np.ndarray, endian: str = "<", faces: int = 0, extra: bool = False) -> bytes:
    fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
    props = ["property float x", "property float y", "property float z"]
    dt = [("x", endian + "f4"), ("y", endian + "f4"), ("z", endian + "f4")]
    if extra:
        props.append("property uchar red")
        dt.append(("red", "u1"))
    header = ["ply", f"format {fmt} 1.0", f"element vertex {len(points)}", *props]
    if faces:
        header += [f"element face {faces}", "property list uchar int vertex_indices"]
    header.append("end_header")
    arr = np.zeros(len(points), dtype=dt)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    body = arr.tobytes()
    if faces:
        rec = np.zeros(faces, dtype=[("n", "u1"), ("i", endian + "i4", (3,))])
        rec["n"] = 3
        body += rec.tobytes()
    return ("\n".join(header) + "\n").encode() + body


def test_ply_binary_le_be_and_ascii_read_identically(tmp_path):
    pts = np.array([[0.0, 1.0, 2.0], [3.5, -4.25, 5.0], [6.0, 7.0, -8.0]])
    (tmp_path / "le.ply").write_bytes(_ply_binary(pts, "<", faces=2, extra=True))
    (tmp_path / "be.ply").write_bytes(_ply_binary(pts, ">"))
    ascii_body = "\n".join(" ".join(f"{v:g}" for v in p) for p in pts)
    (tmp_path / "a.ply").write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                                    f"property float z\nend_header\n{ascii_body}\n")
    le, be, asc = (read_scan(tmp_path / n) for n in ("le.ply", "be.ply", "a.ply"))
    assert (le.format, be.format, asc.format) == ("PLY_BINARY_LE", "PLY_BINARY_BE", "PLY_ASCII")
    for s in (le, be, asc):
        np.testing.assert_allclose(s.points, pts)
    assert le.face_count == 2 and le.vertex_count == 3


def test_stl_binary_dedupes_and_counts_like_the_probe(tmp_path):
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]], dtype="<f4")
    data = b"\0" * 80 + struct.pack("<I", 2)
    for t in tri:
        data += np.zeros(3, "<f4").tobytes() + t.tobytes() + b"\0\0"
    (tmp_path / "s.stl").write_bytes(data)
    s = read_scan(tmp_path / "s.stl")
    assert s.format == "STL_BINARY" and s.vertex_count == 6 and s.face_count == 2
    assert len(s.points) == 4  # shared vertices merged


def test_nonfinite_vertices_are_dropped_and_counted(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]])
    (tmp_path / "n.ply").write_bytes(_ply_binary(pts))
    s = read_scan(tmp_path / "n.ply")
    assert len(s.points) == 2 and s.dropped_nonfinite == 1


@pytest.mark.parametrize("blob", [b"", b"hello world", b"ply\nformat binary_little_endian 1.0\nelement vertex 5\n"
                                  b"property float x\nproperty float y\nproperty float z\nend_header\n\x00\x00"])
def test_unreadable_scans_raise_scan_read_error(tmp_path, blob):
    p = tmp_path / "bad.ply"
    p.write_bytes(blob)
    with pytest.raises(ScanReadError):
        read_scan(p)


# --------------------------------------------------------------------------- config
def test_config_hash_follows_every_value_and_unknown_keys_are_refused(tmp_path):
    base = InspectConfig()
    assert base.sha256() == InspectConfig().sha256()
    assert load_config(None, time_budget_s=30).sha256() != base.sha256()
    bad = tmp_path / "c.json"
    bad.write_text('{"no_such_key": 1}')
    with pytest.raises(ConfigError):
        load_config(bad)
    bad.write_text('{"stations_min": "five"}')
    with pytest.raises(ConfigError):
        load_config(bad)


# --------------------------------------------------------------------------- geometry
def test_robust_plane_ignores_outliers_and_a_second_skin():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-20, 20, size=(4000, 2))
    z = 0.01 * xy[:, 0] + rng.normal(0, 0.02, len(xy))
    pts = np.column_stack([xy, z])
    pts[:200, 2] += 3.0  # gross outliers
    pl = geom.robust_plane(pts)
    true_n = geom.unit([-0.01, 0.0, 1.0])
    assert geom.angle_between_deg(pl.n, true_n) < 0.05


def test_plane_intersection_and_rotation_helpers():
    a = geom.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    b = geom.Plane(geom.unit([1.0, 0.0, 1.0]), 0.0)
    p, d = geom.plane_intersection(a, b)
    assert abs(abs(d[1]) - 1.0) < 1e-12 and abs(p[0]) < 1e-12 and abs(p[2]) < 1e-12
    r = geom.rotation_between(np.array([1.0, 0, 0]), np.array([0, 0, -1.0]))
    np.testing.assert_allclose(r @ np.array([1.0, 0, 0]), [0, 0, -1.0], atol=1e-12)
    assert abs(np.linalg.det(r) - 1.0) < 1e-12


def test_mean_normal_is_immune_to_sign_flips():
    rng = np.random.default_rng(3)
    n = geom.unit([0.2, 0.1, 1.0])
    noisy = np.array([geom.unit(n + rng.normal(0, 0.01, 3)) for _ in range(500)])
    noisy[::2] *= -1.0
    m = mean_normal(noisy, n)
    assert geom.angle_between_deg(m, n) < 0.2


def test_skin_peaks_finds_both_skins_one_thickness_apart():
    rng = np.random.default_rng(5)
    h = np.concatenate([rng.normal(0.0, 0.02, 3000), rng.normal(-1.5, 0.02, 1500)])
    top, bottom = geom.skin_peaks(h, 1.5)
    assert abs(top) < 0.05 and abs(bottom + 1.5) < 0.05
    top1, bottom1 = geom.skin_peaks(rng.normal(0.3, 0.02, 3000), 1.5)
    assert bottom1 is None and abs(top1 - 0.3) < 0.05


def test_voxel_cloud_maps_every_point_and_normals_are_unit():
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 50, size=(20000, 2))
    pts = np.column_stack([xy, np.zeros(len(xy))])
    vox = cloud.voxelize(pts, 1.0)
    assert vox.inverse.shape == (len(pts),) and vox.counts.sum() == len(pts)
    cloud.estimate_normals(vox, 14)
    assert np.allclose(np.linalg.norm(vox.normals, axis=1), 1.0)
    assert np.median(np.abs(vox.normals[:, 2])) > 0.999
    # Poisson process at 8 points/mm^2: expected nearest-neighbour distance 0.5 / sqrt(8)
    assert 0.14 < cloud.median_spacing(pts) < 0.2


# --------------------------------------------------------------------------- decode / U / location
@pytest.mark.parametrize("sweep_meas, expected", [(0.5, "NOT_YET_FORMED"), (3.9, "NOT_YET_FORMED"), (30.0, "PARTIALLY_FORMED"),
                                                  (88.0, "FORMED"), (110.0, "FORMED")])
def test_raw_state_thresholds(sweep_meas, expected):
    assert _raw_state(sweep_meas, 90.0, InspectConfig()) == expected


def test_raw_state_without_evidence_is_undecided():
    assert _raw_state(None, 90.0, InspectConfig()) is None


def test_robust_location_takes_the_dominant_level_over_a_minority_notch():
    vals = np.array([20.0, 20.01, 19.99, 20.02, 20.0, 15.0, 15.02])
    value, sigma, how = robust_location(vals)
    assert how == "DOMINANT_LEVEL" and abs(value - 20.0) < 0.02 and sigma < 0.05


def test_u_budget_grows_with_every_component_and_never_drops_below_the_floor():
    cfg = InspectConfig()
    floor = u_budget("length", {}, cfg)
    assert floor == pytest.approx(cfg.coverage_factor * cfg.u_floor_length_mm)
    prev = floor
    comps: dict[str, float] = {}
    for key in ("station", "sharp_mm", "far_sharp_mm", "edge_mm", "thickness_mm", "pose_mm", "k_sigma_mm"):
        comps[key] = 0.1
        cur = u_budget("length", comps, cfg)
        assert cur > prev
        prev = cur
    assert u_budget("angle", {"tilt_deg": 0.2}, cfg) > u_budget("angle", {}, cfg)


def test_unknown_measurement_never_carries_a_value():
    m = Measurement.unknown("OCCLUDED", "X")
    assert m.value is None and m.u is None and m.reason == "OCCLUDED"


# --------------------------------------------------------------------------- fold kinematics
def _bend_ns(ba: float, sweep_deg: float, r_n: float):
    """A bend whose parent's neutral surface is z = 0 for x < 0, folding up (+z) about a y-parallel axis."""
    from types import SimpleNamespace

    return SimpleNamespace(model=SimpleNamespace(neutral_length_mm=ba), sweep=math.radians(sweep_deg),
                           p_out=np.array([-1.0, 0.0, 0.0]), axis_point=np.array([0.0, 0.0, r_n]),
                           axis_dir=np.array([0.0, -1.0, 0.0]))


def test_fold_transform_conserves_the_developed_length_with_the_arc_centred_on_the_bend_line():
    from domains.production.engine.inspect.articulated import fold_transform

    r_n, sweep = 3.0, 90.0
    ba = math.radians(sweep) * r_n
    bend = _bend_ns(ba, sweep, r_n)
    # the child's attachment edge on the neutral surface at the final pose (the strip fully consumed)
    s = math.radians(sweep)
    p_final = np.array([r_n * math.sin(s), 0.0, r_n - r_n * math.cos(s)])
    np.testing.assert_allclose(fold_transform(bend, bend.sweep), np.eye(4), atol=1e-12)
    # flat (theta = 0) is E1's Method A: the attachment edge lies BA beyond the nominal tangent
    np.testing.assert_allclose(geom.apply(fold_transform(bend, 0.0), p_final[None])[0], [ba, 0.0, 0.0], atol=1e-9)
    for frac in (0.3, 0.5, 0.7):
        th = frac * s
        used = r_n * th
        rest = 0.5 * (ba - used)  # the unconsumed strip, half on each side of the arc
        # neutral path: flat parent gain, the arc of radius r_n about the moved axis, flat child gain
        expected = np.array([rest + r_n * math.sin(th) + rest * math.cos(th), 0.0,
                             r_n - r_n * math.cos(th) + rest * math.sin(th)])
        got = geom.apply(fold_transform(bend, th), p_final[None])[0]
        np.testing.assert_allclose(got, expected, atol=1e-9)
        # mutate-and-check: the rotation-only model this replaces misplaces the child by mm
        old = geom.apply(geom.rotation_about_line(bend.axis_point, bend.axis_dir, th - bend.sweep), p_final[None])[0]
        assert np.linalg.norm(old - expected) > 0.3


# --------------------------------------------------------------------------- skin levels
def test_station_side_fit_finds_a_skin_the_panel_plane_misses_by_tenths():
    from domains.production.engine.inspect.metrology import _fit_side, _skin_level

    rng = np.random.default_rng(11)
    t = 1.5
    xy = rng.uniform(0, 12, size=(400, 2))
    pts = np.column_stack([xy, -0.27 + rng.normal(0, 0.02, len(xy))])  # the OML skin, 0.27 mm below the plane
    plane = geom.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    cfg = InspectConfig()
    level = _skin_level(plane.distance(pts), 0.0, 0.6, 0.15, cfg.min_station_points)
    assert level is not None and abs(level + 0.27) < 0.02
    fit, inner, support, _ = _fit_side(pts, plane, t, cfg)
    assert fit is not None and not inner and support >= 380
    assert abs(fit.d + 0.27) < 0.02  # the fitted skin, not the panel plane
    # the plane on the data: no shift, the target level itself
    assert _skin_level(plane.distance(pts + [0, 0, 0.27]), 0.0, 0.6, 0.15, cfg.min_station_points) == 0.0


def test_child_slab_levels_ignore_cut_wall_clutter_between_the_skins():
    from types import SimpleNamespace

    from domains.production.engine.inspect.articulated import _refine_skin_levels

    rng = np.random.default_rng(12)
    t, d_mid = 1.5, 0.27  # true mid-plane at 0; the split-at-mid-plane estimate is 0.27 mm off
    skin = rng.normal(t / 2, 0.02, 3000)  # the one visible skin (upper)
    clutter = rng.uniform(-t / 2, t / 2, 600)  # walls / flare inside the slab
    h = np.concatenate([skin, clutter])
    pts = np.column_stack([rng.uniform(0, 50, len(h)), rng.uniform(0, 50, len(h)), h])
    bend = SimpleNamespace(oml_skin=0)
    new_mid, seen = _refine_skin_levels(pts, np.array([0.0, 0.0, 1.0]), d_mid, t, 1.0, bend, (0, 1), InspectConfig())
    assert abs(new_mid) < 0.02 and seen == (0,)
    # both skins seen: the mid-plane between the two levels
    both = np.concatenate([rng.normal(0.75, 0.02, 2000), rng.normal(-0.75, 0.02, 2000), clutter])
    pts2 = np.column_stack([rng.uniform(0, 50, len(both)), rng.uniform(0, 50, len(both)), both])
    mid2, seen2 = _refine_skin_levels(pts2, np.array([0.0, 0.0, 1.0]), 0.0, t, 1.0, bend, (0,), InspectConfig())
    assert abs(mid2) < 0.02 and seen2 == (0, 1)


# --------------------------------------------------------------------------- datum edges, erosion, pose bias
def test_inplane_rows_use_the_cut_wall_and_downweight_bare_skin_edges():
    from types import SimpleNamespace

    from domains.production.engine.inspect.registration import _inplane_constraints

    cfg = InspectConfig()
    g = np.arange(2.5, 27.5, 0.25)
    # edge 1 (y = 0, material at y > 0): skin seen to 0.2 mm short of the edge, no wall points
    e1 = np.array([[x, y] for x in g for y in np.arange(0.2, 3.0, 0.25)])
    # edge 2 (x = 40, material at x < 40): skin to 0.05 mm short, plus its cut wall at the edge
    e2_skin = np.array([[40.0 - xo, y] for y in g for xo in np.arange(0.05, 3.0, 0.25)])
    e2_wall = np.array([[40.0 + dx, y] for y in g for dx in (-0.02, 0.0, 0.02)])
    uv = np.vstack([e1, e2_skin, e2_wall])
    interior = np.r_[np.zeros(len(e1) + len(e2_skin), bool), np.ones(len(e2_wall), bool)]
    segs = [(np.array([0.0, 0.0]), np.array([30.0, 0.0]), np.array([0.0, -1.0])),
            (np.array([40.0, 0.0]), np.array([40.0, 30.0]), np.array([1.0, 0.0]))]
    model = SimpleNamespace(holes=[])
    con = _inplane_constraints(uv, model, segs, cfg, 0.0, 0.25, np.array([20.0, 15.0]), 20.0, wide=False,
                               interior=interior)
    walled = np.asarray(con["walled"])
    res = np.asarray(con["edge_res"])
    w = np.asarray(con["weights"])
    assert walled.any() and (~walled).any()
    assert np.all(np.abs(res[walled]) < 0.03)  # the wall is the edge
    assert np.all(np.abs(res[~walled] + 0.2) < 0.03)  # the eroded skin extreme
    assert np.all(w[walled] == 1.0) and np.all(w[~walled] == cfg.edge_unwalled_weight)
    # without the wall flags every bin is a bare skin edge
    con0 = _inplane_constraints(uv, model, segs, cfg, 0.0, 0.25, np.array([20.0, 15.0]), 20.0, wide=False)
    assert not any(con0["walled"])


def test_pose_bias_bound_and_erosion_scale():
    from domains.production.engine.inspect.registration import InPlaneFit, _erosion_scale
    from types import SimpleNamespace

    cfg = InspectConfig()
    assert _erosion_scale(np.zeros(10), np.full(10, -0.2), 0.5, cfg) == pytest.approx(0.2)
    assert _erosion_scale(np.zeros(10), np.full(10, -5.0), 0.5, cfg) == pytest.approx(cfg.erosion_max_spacing * 0.5)
    assert _erosion_scale(np.zeros(2), np.zeros(2), 0.5, cfg) == pytest.approx(cfg.erosion_default_spacing * 0.5)
    model = SimpleNamespace(p0=np.zeros(3), f1=np.array([1.0, 0, 0]), f2=np.array([0, 1.0, 0]),
                            polygon=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))
    model.to2d = lambda pts: np.asarray(pts, dtype=np.float64)[:, :2]
    base = dict(T=np.eye(4), information=np.eye(3), length_scale=10.0, edge_bins=0, edge_bins_expected=0, holes_seen=0,
                holes_total=0, edge_rms=0.0, hole_rms=0.0, degenerate=False, translation_cov=np.eye(2),
                rotation_sigma_rad=0.0, skin_heights={}, datum_points=0, datum_coverage=1.0, hole_mismatch=0)
    # two unwalled rows that shift tx by +0.1 and -0.05 mm when eroded: worst case is the larger side
    fit = InPlaneFit(**base, bias_gain=np.array([[0.0, 0.0], [0.1, -0.05], [0.0, 0.0]]))
    assert fit.position_bias_mm(model, np.array([5.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.1)
    assert fit.position_bias_mm(model, np.array([5.0, 5.0, 0.0]), np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert InPlaneFit(**base).position_bias_mm(model, np.zeros(3), np.array([1.0, 0, 0])) == 0.0


def test_u_budget_includes_the_bias_terms():
    cfg = InspectConfig()
    assert u_budget("position", {"pose_bias_mm": 0.2}, cfg) > u_budget("position", {}, cfg)
    assert u_budget("length", {"erosion_mm": 0.2}, cfg) > u_budget("length", {}, cfg)


# --------------------------------------------------------------------------- unplanned creases
def test_crease_crossings_end_the_flange_only_at_creases_along_the_bend():
    from domains.production.engine.inspect.metrology import BendMeasure, _crease_crossings

    n = 5
    bm = BendMeasure(bend=0, ok=True, z=np.linspace(0, 40, n), valid=np.ones(n, bool),
                     sharp=np.column_stack([np.zeros(n), np.linspace(0, 40, n), np.zeros(n)]),
                     line_dir=np.array([0.0, 1.0, 0.0]))
    w = np.array([1.0, 0.0, 0.0])
    along = (np.array([20.0, 5.0, 0.3]), np.array([0.0, 1.0, 0.0]), 0.05, "C1")
    across = (np.array([10.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.05, "C2")
    unresolved = (None, np.array([0.0, 1.0, 0.0]), 0.0, "C3")
    out = _crease_crossings(bm, w, 1.0, [along, across, unresolved])
    assert len(out) == n and all(abs(v - 20.0) < 1e-9 and u == 0.05 for v, u in out)
    assert _crease_crossings(bm, w, 1.0, [across, unresolved]) == []


# --------------------------------------------------------------------------- config validation (fix stage)
@pytest.mark.parametrize("bad", [{"stations_min": 0}, {"u_floor_length_mm": float("nan")}, {"stations_min": 12},
                                 {"flat_fraction_max": 0.9}, {"time_budget_s": -1.0}, {"u_floor_angle_deg": 0.0},
                                 {"min_station_points": 2.5}, {"theta_min_deg": 5.0}, {"pose_probe_deg": 0.05},
                                 {"coarse_peaks_per_family": 8}])
def test_config_refuses_non_finite_out_of_range_inconsistent_and_removed_values(bad):
    with pytest.raises(ConfigError):
        load_config(None, **bad)


def test_config_is_validated_on_construction_and_the_dead_probes_are_gone():
    cfg = InspectConfig()
    assert not hasattr(cfg, "pose_probe_deg") and not hasattr(cfg, "coarse_peaks_per_family")
    with pytest.raises(ConfigError):
        InspectConfig(stations_min=0)  # would divide by zero in the station layout
    with pytest.raises(ConfigError):
        InspectConfig(u_skin_level_mm=float("inf"))  # would break the canonical hash


# --------------------------------------------------------------------------- ingest: gross outliers
def test_gross_outlier_vertices_are_dropped_and_the_part_is_kept():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.0, 100.0, (5000, 3))
    extra = np.array([[3e38, 0.0, 0.0], [0.0, 0.0, 5e4], [600.0, 50.0, 50.0]])  # sentinel, 50 m, a fixture 6 m-ish away
    mask = cloud.gross_inliers(np.vstack([pts, extra]), 0.5, 10.0)
    assert mask[:5000].all() and not mask[5000] and not mask[5001] and mask[5002]
    # mutate-and-check: without the filter the sentinel breaks the integer voxel keys
    with pytest.warns(RuntimeWarning):
        cloud.voxelize(np.vstack([pts, extra[:1]]), 1.0)


# --------------------------------------------------------------------------- measured sheet thickness
def test_tile_thickness_measures_a_sheet_off_its_cad_thickness():
    from domains.production.engine.inspect.registration import tile_thickness

    rng = np.random.default_rng(4)
    cfg = InspectConfig()
    uv = rng.uniform(0.0, 120.0, (40000, 2))
    t_real = 2.3  # 15 % thicker than the CAD's 2.0 mm
    upper = rng.random(len(uv)) < 0.6
    h = np.where(upper, 0.0, -t_real) + rng.normal(0.0, 0.02, len(uv))
    res = tile_thickness(h, uv, 2.0, cfg)
    assert res is not None
    t, sigma, tiles = res
    assert abs(t - t_real) < 0.01 and tiles >= 50 and sigma >= cfg.thickness_floor_mm
    # mutate-and-check: the registration's CAD-window skin search misses or mislocates the second skin
    bottom = geom.skin_peaks(h[:4000], 2.0)[1]
    assert bottom is None or abs(bottom + t_real) > 0.08
    # one skin only: nothing is measured, the compiled t stays in force
    assert tile_thickness(rng.normal(0.0, 0.02, 6000), uv[:6000], 2.0, cfg) is None


def test_single_skin_offsets_enter_u_through_the_other_flange_and_the_frame_height():
    from domains.production.engine.inspect.metrology import BendMeasure, Thickness, position_measurement, thickness_along

    cfg = InspectConfig()
    n = 5
    zs = np.linspace(5.0, 45.0, n)
    bm = BendMeasure(bend=0, ok=True, z=zs, valid=np.ones(n, bool),
                     sharp=np.column_stack([np.full(n, 0.1), zs, np.zeros(n)]), line_point=np.zeros(3),
                     line_dir=np.array([0.0, 1.0, 0.0]), w_parent=np.array([1.0, 0.0, 0.0]),
                     w_child=np.array([0.0, 0.0, 1.0]), sin_sweep=1.0, inner_child=1.0)
    thick = Thickness(t_mm=2.0, u_mm=0.05, measured=False, frame_mm=0.04)
    # the child's OML came from its inner skin: the sharp moves along the parent by u_t / sin(sweep)
    assert thickness_along(bm, True, thick) == pytest.approx(0.05)
    assert thickness_along(bm, False, thick) == 0.0
    line = (np.zeros(3), np.array([0.0, 50.0, 0.0]))
    m = position_measurement(bm, line, np.array([1.0, 0.0, 0.0]), 0.01, cfg, thick=thick,
                             datum_normal=np.array([1.0, 0.0, 0.0]))
    assert m.value == pytest.approx(0.1) and m.components["thickness_mm"] == pytest.approx(0.05)
    assert m.components["frame_height_mm"] == pytest.approx(0.04)
    # mutate-and-check: two skins seen everywhere, no thickness term
    bm.inner_child = 0.0
    m2 = position_measurement(bm, line, np.array([1.0, 0.0, 0.0]), 0.01, cfg, thick=thick)
    assert "thickness_mm" not in m2.components and m2.u < m.u


# --------------------------------------------------------------------------- backgauge across a kinked strip
def _kinked_island(theta_deg: float = 4.0, t: float = 2.0, ri: float = 3.0, k: float = 0.33):
    """A flat side panel (skin 0 at z = 0, x 0..50) and a panel beyond a strip kinked down by theta, 80 mm long."""
    from types import SimpleNamespace

    from domains.production.engine.inspect.articulated import PanelFit

    th = math.radians(theta_deg)
    ro = ri + t
    centre_z = -t - ri  # the kink bends toward -z: skin 0 is its convex side
    ys = np.arange(0.0, 60.0, 0.4)
    d_q = np.array([math.cos(th), 0.0, -math.sin(th)])
    n_q = np.array([math.sin(th), 0.0, math.cos(th)])
    p_pts, q_pts, arc = [], [], []
    for x in np.arange(1.0, 50.0, 0.4):
        for z in (0.0, -t):
            p_pts += [(x, y, z) for y in ys]
    for phi in np.linspace(0.0, th, 6):
        for r in (ro, ri):
            arc += [(50.0 + r * math.sin(phi), y, centre_z + r * math.cos(phi)) for y in ys]
    end0 = np.array([50.0 + ro * math.sin(th), 0.0, centre_z + ro * math.cos(th)])
    end1 = np.array([50.0 + ri * math.sin(th), 0.0, centre_z + ri * math.cos(th)])
    for s in np.arange(0.2, 80.0, 0.4):
        for e in (end0, end1):
            q_pts += [tuple(e + d_q * s + np.array([0.0, y, 0.0])) for y in ys]
    for f in np.linspace(0.1, 0.9, 6):  # the far cut wall, perpendicular to the panel
        q_pts += [tuple(end0 + d_q * 80.0 - n_q * (f * t) + np.array([0.0, y, 0.0])) for y in ys]
    pts = np.array(p_pts + arc + q_pts, dtype=np.float64)
    ip = np.arange(len(p_pts))
    iq = np.arange(len(p_pts) + len(arc), len(pts))
    vox = cloud.VoxelCloud(points=pts, inverse=np.arange(len(pts)), counts=np.ones(len(pts), dtype=np.int64), voxel=0.4)
    p_fit = PanelFit(0, "DATUM", np.eye(4), n0=np.array([0.0, 0.0, 1.0]), d0=0.0, t=t, region=ip, voxels=ip)
    q_fit = PanelFit(1, "FITTED", np.eye(4), n0=n_q, d0=float(n_q @ end0), t=t, region=iq, voxels=iq)
    frame = SimpleNamespace(o=np.array([50.0, 30.0, centre_z]), a=np.array([0.0, 1.0, 0.0]), u=np.array([1.0, 0.0, 0.0]))
    art = SimpleNamespace(panels=[p_fit, q_fit], bends=[SimpleNamespace(frame=frame)])
    part = SimpleNamespace(bends=[SimpleNamespace(parent=0, child=1, ri=ri, label="B9")], k_factor=k)
    developed = 50.0 + th * (ri + k * t) + 80.0  # neutral fibre: flat + arc + flat
    return pts, vox, art, part, developed


def test_backgauge_island_is_unfolded_about_a_kinked_strip_to_its_developed_length():
    from domains.production.engine.inspect.metrology import PointIndex, _far_free, unfold_island

    cfg = InspectConfig()
    pts, vox, art, part, developed = _kinked_island()
    idx = PointIndex(pts, vox)
    unfolded, notes = unfold_island(idx, part, art, 0, [0, 1], [0], 0, cfg)
    assert notes == ("unfolded_B9_deg=4.000",)
    mask = np.zeros(len(pts), bool)
    mask[art.panels[0].region] = True
    mask[art.panels[1].region] = True
    plane = art.panels[0].skin_plane(0)
    args = (mask, plane, 2.0, np.array([0.0, 30.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 10.0,
            1.0, 0.4, 0.4, cfg)
    fe = _far_free(unfolded, *args)
    assert fe.walled and not fe.truncated
    assert abs(fe.value - developed) < 0.03, (fe.value, developed)
    # mutate-and-check: on the raw points the run leaves the side panel's slab ~32 mm past the kink and is
    # flagged truncated (the old engine reported that end as a backgauge 50 mm short)
    raw = _far_free(pts, *args)
    assert raw.truncated and raw.value < developed - 20.0


def test_bare_far_edges_prove_too_long_never_too_short():
    from domains.production.engine.inspect.metrology import Measurement, bare_edge_untrusted

    cfg = InspectConfig()

    def m(value, seen):
        return Measurement(value=value, u=0.2, method="OML_SHARP_TO_FREE_EDGE_MEDIAN", walls_seen=seen, wall_stations=5)

    short_bare = bare_edge_untrusted(m(98.0, 0), 100.0, 0.5, 0.5, cfg)
    assert short_bare is not None and short_bare.reason == "EDGE_NOT_SEEN" and short_bare.value is None
    assert bare_edge_untrusted(m(98.0, 5), 100.0, 0.5, 0.5, cfg) is None  # the wall is seen: a real short side
    assert bare_edge_untrusted(m(102.0, 0), 100.0, 0.5, 0.5, cfg) is None  # material is there: too long stands
    assert bare_edge_untrusted(m(99.9, 0), 100.0, 0.5, 0.5, cfg) is None  # a pass (erosion is in U) stands


def test_backgauge_k_sigma_comes_from_hashed_stage_geometry_never_from_diagnostics():
    from types import SimpleNamespace

    from domains.production.engine.inspect.engine import _k_sigma

    def ln(a, b):
        return SimpleNamespace(start_mm=a, end_mm=b)

    stage = SimpleNamespace(bend_lines={"bj": ln((0, 0, 0), (0, 100, 0)), "bc": ln((50, 0, 0), (50, 100, 0)),
                                        "bd": ln((10, -5, 0), (90, -5, 0))})
    bends = [SimpleNamespace(uid="bj", side_panel={"A": 0}, side_out={"A": np.array([1.0, 0.0, 0.0])}),
             SimpleNamespace(uid="bc", sweep=math.radians(90.0), t=2.0),
             SimpleNamespace(uid="bd", sweep=math.radians(45.0), t=2.0)]
    diag = {"backgauge_nominals": {"S1": {"k_sigma_mm": 99.0}}}
    part = SimpleNamespace(stages={"S1": stage}, bends=bends, k_sigma=0.05,
                           stage_transform=lambda sid, p: np.eye(4), compiled=SimpleNamespace(diagnostics=diag))
    notes: list[str] = []
    value = _k_sigma(part, "S1", 0, "A", [1, 2], 120.0, notes.append, 1)
    # only the strip the stations cross counts: sweep * t of bc, times sigma_K
    assert value == pytest.approx(0.05 * math.radians(90.0) * 2.0)
    assert notes and "not used" in notes[0]
    # a far boundary short of the strip: no K sensitivity
    assert _k_sigma(part, "S1", 0, "A", [1, 2], 40.0) == 0.0
