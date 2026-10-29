"""A1 accuracy-loop unit tests of the per-scan engine (E3): non-round datum holes, per-direction
in-plane trust of bend-line positions, and the nominal station profile of flange lengths.

Pure numpy (no OCP, no E1/E2 geometry, no physical data).
Run: PYT tests/test_prod_engine_inspect_a1.py -q
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.registration import (
    DatumModel,
    InPlaneFit,
    _inplane_constraints,
    _nonround_hole_filled,
    _round_hole,
    _straight_runs,
)


def _circle(c, r, n=72):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], axis=1)


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _rounded_rect(x0, y0, x1, y1, r, n=8):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.asarray(pts)


def _obround(c, half_len, r, n=16):
    pts = []
    for a in np.radians(np.linspace(-90, 90, n)):
        pts.append([c[0] + half_len + r * math.cos(a), c[1] + r * math.sin(a)])
    for a in np.radians(np.linspace(90, 270, n)):
        pts.append([c[0] - half_len + r * math.cos(a), c[1] + r * math.sin(a)])
    return np.asarray(pts)


def _model(holes, blocked=None, outline=(-100.0, -60.0, 100.0, 60.0)):
    x0, y0, x1, y1 = outline
    poly = _rect(x0, y0, x1, y1)
    return DatumModel(p0=np.zeros(3), n0=np.array([0.0, 0.0, 1.0]), f1=np.array([1.0, 0.0, 0.0]),
                      f2=np.array([0.0, 1.0, 0.0]), t=2.0, polygon=poly, free=np.ones(4, dtype=bool),
                      holes=[np.asarray(h, dtype=np.float64) for h in holes], area=(x1 - x0) * (y1 - y0),
                      bbox=(x0, y0, x1, y1), hole_blocked=blocked or [])


def _grid(x0, y0, x1, y1, step):
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


# --------------------------------------------------------------------------- hole shapes
def test_round_hole_needs_the_area_of_a_circle_not_just_equidistant_vertices():
    assert _round_hole(_circle((0, 0), 2.5))
    assert _round_hole(_circle((3, 4), 40.0, n=12))  # a coarse 12-gon still encloses ~95 % of pi r^2
    assert not _round_hole(_rect(0, 0, 50, 80))  # corners equidistant from the centre
    assert not _round_hole(_rect(0, 0, 10, 10))
    assert not _round_hole(_rounded_rect(-35, -30, 45, 35, 8.0))  # a window: arcs crowd the vertex list
    assert not _round_hole(_obround((0, 0), 1.2, 1.5))  # a slot


def test_straight_runs_of_a_rounded_window_are_its_four_sides():
    runs = _straight_runs(_rounded_rect(0, 0, 60, 20, 3.0))
    lengths = sorted(float(np.linalg.norm(b - a)) for a, b in runs if np.linalg.norm(b - a) > 5)
    assert lengths == pytest.approx([14.0, 14.0, 54.0, 54.0], abs=1e-6)
    assert all(np.linalg.norm(b - a) < 1.0 for a, b in _straight_runs(_circle((0, 0), 10.0)))  # arcs: short pieces
    rect = _straight_runs(_rect(0, 0, 30, 10))
    assert len(rect) == 4 and sorted(round(float(np.linalg.norm(b - a)), 6) for a, b in rect) == [10, 10, 30, 30]


def test_hole_segments_point_into_the_hole_and_stop_short_of_the_opposite_side():
    model = _model([_rect(-30, -10, 30, 10)])
    segs = model.hole_segments(min_length=7.0, reach=6.0)
    assert len(segs) == 4
    for a, b, n_in, lim in segs:
        mid = 0.5 * (a + b)
        assert abs(mid[0]) < 30.5 and abs(mid[1]) < 10.5
        inside = mid + n_in * 1.0
        assert abs(inside[0]) < 30 and abs(inside[1]) < 10  # the normal points into the hole
        clear = 20.0 if abs(n_in[1]) > 0.5 else 60.0
        assert lim == pytest.approx(0.45 * clear, abs=0.01)
    # a narrow slot (2 mm wide): its opposite side is inside any useful band -> no rows
    assert _model([_rect(-30, -1, 30, 1)]).hole_segments(7.0, 6.0) == []


def test_hole_segments_skip_the_part_of_a_window_a_flat_tab_fills_at_stage_0():
    # window x in [-30, 30], y in [-10, 10]; a tab hinged on its right side lies flat in it at S_0
    # (x 6 .. 30, y -8 .. 8): the right side is a bend line, the top/bottom near the tab are blocked
    tab = _grid(6.0, -8.0, 29.7, 8.0, 0.5)
    model = _model([_rect(-30, -10, 30, 10)], blocked=[tab])
    segs = model.hole_segments(min_length=7.0, reach=6.0)
    for a, b, n_in, lim in segs:
        if n_in[0] < -0.5:  # the right side (normal -x into the hole) touches the tab: never used
            pytest.fail("the hinge side of a tab window was used")
        if abs(n_in[1]) > 0.5:  # top / bottom runs: only the part clear of the tab, band short of it
            assert max(a[0], b[0]) <= 7.0 or lim <= 2.0 - 0.6 + 1e-9
    assert any(n_in[0] > 0.5 for _, _, n_in, _ in segs)  # the open left side anchors the pose


def test_slot_and_window_fill_is_judged_on_the_open_core_not_a_mean_radius_disc():
    window = _rounded_rect(-35, -30, 45, 35, 8.0)
    model = _model([window])
    mat = _grid(-99, -59, 99, 59, 0.5)
    from matplotlib.path import Path as MplPath

    open_scan = mat[~MplPath(window).contains_points(mat)]
    assert not _nonround_hole_filled(open_scan, model, 0, 0.5)  # an open window (the old disc rule fired here)
    assert _nonround_hole_filled(mat, model, 0, 0.5)  # material where the CAD has the window: filled
    # the surroundings not seen: not judged
    far = mat[np.abs(mat[:, 0]) > 80]
    assert not _nonround_hole_filled(far, model, 0, 0.5)
    # the old rule's failure: a slot's own cut walls inside the mean radius are not material in the hole
    slot = _obround((0.0, 0.0), 1.2, 1.5)
    smodel = _model([slot])
    walls = np.vstack([open_scan, _circle((1.2, 0.0), 1.45, 40), _circle((-1.2, 0.0), 1.45, 40)])
    assert not _nonround_hole_filled(walls, smodel, 0, 0.25)


def test_inplane_rows_of_a_window_have_zero_residual_when_the_scan_fits():
    window = _rect(-30, -10, 30, 10)
    model = _model([window])
    cfg = InspectConfig()
    from matplotlib.path import Path as MplPath

    mat = _grid(-60, -40, 60, 40, 0.25)
    mat = mat[~MplPath(window).contains_points(mat)]
    segs = model.hole_segments(2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm, 2.0 * cfg.edge_band_mm)
    con = _inplane_constraints(mat, model, segs, cfg, 0.0, 0.25, np.zeros(2), 50.0, wide=True)
    res = np.asarray(con["edge_res"])
    assert len(res) >= 20 and np.all(np.abs(res) < 0.3)
    assert con["hole_mismatch"] == 0 and con["hole_res"] == []  # a window is not binned radially
    rows = np.asarray(con["rows"])
    assert np.linalg.eigvalsh(rows.T @ rows).min() > 3.0  # the window alone fixes rotation and both shifts


# --------------------------------------------------------------------------- per-direction pose trust
def _fit(undetermined, cov=None):
    base = dict(T=np.eye(4), information=np.eye(3), length_scale=50.0, edge_bins=0, edge_bins_expected=0,
                holes_seen=0, holes_total=0, edge_rms=0.0, hole_rms=0.0, degenerate=True,
                translation_cov=np.eye(2), rotation_sigma_rad=0.0, skin_heights={}, datum_points=0,
                datum_coverage=1.0, hole_mismatch=0)
    return InPlaneFit(**base, pose_cov=np.eye(3) * 0.01 if cov is None else cov,
                      undetermined=np.asarray(undetermined, dtype=np.float64).reshape(-1, 3))


def test_a_free_in_plane_direction_withholds_only_the_positions_measured_along_it():
    model = _model([])
    fit = _fit([[0.0, 0.0, 1.0]])  # ty free: the datum is bounded by bends on its +-y sides
    p = np.array([10.0, 50.0, 0.0])
    assert fit.position_determined(model, p, np.array([1.0, 0.0, 0.0]))  # measured along x
    assert not fit.position_determined(model, p, np.array([0.0, 1.0, 0.0]))  # measured along y
    assert fit.position_determined(model, p, np.array([0.0, 0.0, 1.0]))  # along the datum normal
    # the sigma along x ignores the free direction (and its regularised variance): rotation lever
    # -50 / L = -1 at y = 50, plus tx
    cov = np.diag([0.0001, 0.0004, 1e6])
    assert _fit([[0, 0, 1]], cov).position_sigma_mm(model, p, np.array([1.0, 0.0, 0.0])) == \
        pytest.approx(math.sqrt(0.0005))
    # a free rotation moves every in-plane position away from the centre
    rot = _fit([[1.0, 0.0, 0.0]])
    assert not rot.position_determined(model, np.array([90.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert rot.position_determined(model, np.array([90.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert _fit([]).position_determined(model, p, np.array([0.0, 1.0, 0.0]))


# --------------------------------------------------------------------------- nominal station profile
def _bend_part(panel_outline, holes=(), adjacent=None):
    """A minimal PartView stand-in: bend 0 along +x at y = 0, side B panel in the plane z = 0 toward +y."""
    from domains.production.engine.inspect import metrology as M  # noqa: F401

    axis_dir = np.array([1.0, 0.0, 0.0])
    loci = {"oml_virtual_sharp": (np.array([0.0, 0.0, 0.0]), np.array([40.0, 0.0, 0.0]))}
    bend = SimpleNamespace(uid="bu_test", axis_dir=axis_dir, axis_point=np.array([0.0, 0.0, -1.0]), z0=0.0,
                           z1=40.0, side_out={"B": np.array([0.0, 1.0, 0.0])}, side_panel={"B": 1},
                           locus=lambda name: loci[name])
    panel = SimpleNamespace(model=SimpleNamespace(outer_boundary_mm=[tuple(p) + (0.0,) for p in panel_outline],
                                                  holes_mm=[[tuple(p) + (0.0,) for p in h] for h in holes]))
    bends = [bend]
    index = {"bu_test": 0}
    if adjacent is not None:
        a0, a1 = adjacent
        bends.append(SimpleNamespace(locus=lambda name: (np.asarray(a0, float), np.asarray(a1, float))))
        index["bu_adj"] = 1
    return SimpleNamespace(bends=bends, panels=[None, panel], bend_index=index)


def test_side_profile_follows_a_stepped_far_edge_and_a_tapered_adjacent_bend():
    from domains.production.engine.inspect.metrology import SideProfile

    # far edge at y = 30 with a 1.5 mm tab (y = 31.5) over x 15 .. 25
    outline = [(0, 0), (40, 0), (40, 30), (25, 30), (25, 31.5), (15, 31.5), (15, 30), (0, 30)]
    sp = SideProfile.build(_bend_part(outline), 0, "B", "FREE_EDGE", None)
    assert sp.at(5.0) == pytest.approx(30.0) and sp.at(20.0) == pytest.approx(31.5)
    assert sp.covers(30.0) and not sp.covers(33.0)
    zs, vals = sp.across(14.0, 3.0)
    assert vals.min() == pytest.approx(30.0) and vals.max() == pytest.approx(31.5)
    # a window between the bend and the far edge does not end the flange: the farthest crossing
    win = [(18, 10), (22, 10), (22, 20), (18, 20)]
    sp2 = SideProfile.build(_bend_part([(0, 0), (40, 0), (40, 30), (0, 30)], holes=[win]), 0, "B", "FREE_EDGE", None)
    assert sp2.at(20.0) == pytest.approx(30.0)
    # an adjacent bend running from (0, 50) to (40, 60): the flange length grows along the bend
    part = _bend_part([(0, 0), (40, 0), (40, 60), (0, 50)], adjacent=((0, 50, 0), (40, 60, 0)))
    adj = SideProfile.build(part, 0, "B", "ADJACENT_BEND", "bu_adj")
    assert adj.at(0.0) == pytest.approx(50.0) and adj.at(40.0) == pytest.approx(60.0)
    assert adj.at(20.0) == pytest.approx(55.0) and adj.covers(55.0)


def test_flattening_takes_a_slanted_far_edge_at_the_station_centre():
    from domains.production.engine.inspect.geom import Plane
    from domains.production.engine.inspect.metrology import _far_free

    cfg = InspectConfig()
    # far edge s = 30 + 1.5 z (a taper), material grid on both skins of a 2 mm sheet in z = 0 .. -2
    g = _grid(-6.0, 0.0, 6.0, 45.0, 0.25)
    keep = g[:, 1] <= 30.0 + 1.5 * g[:, 0]
    pts = np.column_stack([g[keep, 0], g[keep, 1], np.zeros(int(keep.sum()))])
    pts = np.vstack([pts, pts - np.array([0.0, 0.0, 2.0])])
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    args = (pts, np.ones(len(pts), dtype=bool), plane, 2.0, np.zeros(3), np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), 8.0, 1.0, 0.25, 0.8, cfg)
    plain = _far_free(*args)
    assert plain.value > 34.0  # the strip's far corner (z = +4) dominates without the profile
    zs = np.arange(-4.0, 4.0001, 0.1)
    flat = _far_free(*args, flatten=(zs, 1.5 * zs))
    assert flat.value == pytest.approx(30.0, abs=0.3)  # the edge at the station centre


# --------------------------------------------------------------------------- displaced formed flange
def test_a_formed_flange_displaced_beyond_the_slack_is_found_by_the_wide_search():
    from domains.production.engine.inspect.articulated import HingeFrame, _displaced_formed, _hough

    cfg = InspectConfig()
    # hinge frame: u along the parent's continuation, v toward the child; nominal inner child line at
    # rho = -1 (u = 1 at theta = 90 deg), outer at rho = -4 (t = 3)
    frame = HingeFrame(o=np.zeros(3), a=np.array([0.0, 0.0, 1.0]), u=np.array([1.0, 0.0, 0.0]),
                       v=np.array([0.0, 1.0, 0.0]), z0=0.0, z1=40.0, rho_iml=-1.0, delta=-3.0, band=12.0)
    zz = np.arange(0.0, 40.0, 0.5)
    # the real bend sits 8 mm farther along u (a backgauge error): the parent's two skins (v = 0, -3)
    # run on to u = 9, which near the nominal hinge reads as a flat continuation ...
    pu, pz = np.meshgrid(np.arange(-30.0, 9.0, 0.4), zz)
    parent = np.vstack([np.column_stack([pu.ravel(), np.full(pu.size, v), pz.ravel()]) for v in (0.0, -3.0)])
    # ... and the formed child (90 deg) stands at u = 9 / 12 (inner / outer skin), 40 mm tall
    cv, cz = np.meshgrid(np.arange(0.5, 40.0, 0.4), zz)
    child = np.vstack([np.column_stack([np.full(cv.size, u), cv.ravel(), cz.ravel()]) for u in (9.0, 12.0)])
    pts = np.vstack([parent, child])
    z, uc, vc = pts[:, 2], pts[:, 0], pts[:, 1]
    reach = frame.band * cfg.hough_fine_reach
    keep = np.hypot(uc, vc) < reach
    z, uc, vc = z[keep], uc[keep], vc[keep]
    thetas = np.radians(np.arange(cfg.theta_min_deg, 90.0 + cfg.theta_overbend_deg + 1e-9, 1.0))
    sc, _, _, _ = _hough(z, uc, vc, frame, thetas, cfg, reach)
    narrow_theta = math.degrees(thetas[int(np.argmax(sc))])
    assert narrow_theta < cfg.formed_fraction_min * 90.0  # the narrow window settles on the parent's clutter
    alt = _displaced_formed(z, uc, vc, frame, 90.0, cfg, reach)
    assert alt is not None
    theta, rho, score = math.degrees(alt[0]), alt[1], alt[2]
    assert abs(theta - 90.0) < 1.0 and abs(rho - (-9.0)) < 0.3
    assert score >= cfg.displaced_support_ratio * float(sc.max())
    # a child at its nominal place is left to the narrow search
    child0 = np.vstack([np.column_stack([np.full(cv.size, u), cv.ravel(), cz.ravel()]) for u in (1.0, 4.0)])
    pts0 = np.vstack([parent[parent[:, 0] < 1.0], child0])
    assert _displaced_formed(pts0[:, 2], pts0[:, 0], pts0[:, 1], frame, 90.0, cfg, reach) is None
