"""A1 iteration 4 unit tests of the per-scan engine (E3): datum registration from holes, weakly determined
in-plane directions, child poses on their fitted planes, and the station spread in an angle's U.

Every rule here is geometric; no part, family or file is named. Pure numpy (no OCP, no E1/E2 geometry,
no physical data).
Run: PYT tests/test_prod_engine_inspect_a1_iter4.py -q
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config


# --------------------------------------------------------------------------- shared synthetic datum
def _strip_model(free_ends=(True, True), holes=((-30.0, 0.0, 3.1), (1.5, 0.0, 3.1)), blocked=None,
                 x0=-40.0, x1=40.0, half_w=6.0, t=1.5):
    """A strip datum (z = 0 skin 0, skin 1 at -t) whose long sides are bend edges; its short ends are free."""
    from domains.production.engine.inspect.registration import DatumModel

    poly = np.array([[x0, -half_w], [x1, -half_w], [x1, half_w], [x0, half_w]], dtype=np.float64)
    free = np.array([False, free_ends[1], False, free_ends[0]])
    hl = []
    for cx, cy, r in holes:
        a = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
        hl.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1))
    area = (x1 - x0) * 2 * half_w - sum(math.pi * r * r for _, _, r in holes)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    return DatumModel(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                      t, poly, free, hl, area, (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                      hole_blocked=blocked if blocked is not None else [np.zeros((0, 2)) for _ in hl])


def _sheet_points(model, step=0.42, extra=None, seed=0, walls=False):
    """Both skins of the datum's material (holes empty), plus ``extra`` coplanar rectangles (x0, x1, y0, y1)."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = model.bbox
    rects = [(x0, x1, y0, y1)] + list(extra or [])
    pts = []
    for (a0, a1, b0, b1) in rects:
        xs, ys = np.meshgrid(np.arange(a0, a1, step), np.arange(b0, b1, step))
        uv = np.stack([xs.ravel(), ys.ravel()], axis=1)
        uv = uv + rng.uniform(-0.15, 0.15, uv.shape)
        ok = np.ones(len(uv), dtype=bool)
        for hp in model.holes:
            c = hp.mean(axis=0)
            r = float(np.mean(np.linalg.norm(hp - c, axis=1)))
            ok &= np.linalg.norm(uv - c, axis=1) > r
        uv = uv[ok]
        for z in (0.0, -model.t):
            pts.append(np.column_stack([uv, np.full(len(uv), z) + rng.normal(0.0, 0.01, len(uv))]))
    return np.vstack(pts)


def _pose(rot_deg=(23.0, -41.0, 67.0), shift=(12.0, -35.0, 80.0)):
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rot_deg, degrees=True).as_matrix()
    T[:3, 3] = shift
    return T


def _apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def _err(T_est, pose, probe=np.zeros(3)):
    """(rotation error deg, distance at ``probe`` mm) of an estimated raw -> model transform vs the true pose."""
    D = T_est @ pose
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(D[:3, :3])) - 1.0) / 2.0))))
    return ang, float(np.linalg.norm(_apply(D, probe[None])[0] - probe))


def _inplane(T, deg, centre):
    """``T`` followed by an in-plane rotation (about the datum normal z) by ``deg`` about ``centre``."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    C = np.eye(4)
    C[:3, 3] = centre
    Ci = np.eye(4)
    Ci[:3, 3] = -np.asarray(centre)
    return C @ R @ Ci @ T


# --------------------------------------------------------------------------- 1. holes found in a scan
def test_scan_holes_finds_round_holes_and_drops_slots_and_the_outside():
    from domains.production.engine.inspect.registration import scan_holes

    rng = np.random.default_rng(3)
    xs, ys = np.meshgrid(np.arange(0.0, 90.0, 0.45), np.arange(0.0, 30.0, 0.45))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1) + rng.normal(0.0, 0.03, (xs.size, 2))
    holes = [((20.0, 15.0), 2.7), ((55.0, 12.0), 1.8)]
    keep = np.ones(len(pts), dtype=bool)
    for c, r in holes:
        keep &= np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) > r
    keep &= ~((np.abs(pts[:, 0] - 75.0) < 1.5) & (np.abs(pts[:, 1] - 15.0) < 7.0))  # a 3 x 14 mm slot
    found = sorted(scan_holes(pts[keep], 0.45, 1.0, 3.6), key=lambda h: h[0][0])
    assert len(found) == 2
    for (c, r), (fc, fr) in zip(holes, found):
        assert np.linalg.norm(fc - np.asarray(c)) < 0.2
        assert abs(fr - r) < 0.3


def test_datum_hole_targets_use_the_open_bore_of_a_stepped_hole():
    from domains.production.engine.inspect.registration import datum_hole_targets

    a = np.linspace(0.0, 2 * math.pi, 120, endpoint=False)
    ring = np.stack([1.8 * np.cos(a), 1.8 * np.sin(a)], axis=1) + np.array([1.5, 0.0])  # countersink material
    model = _strip_model(holes=((-30.0, 0.0, 3.1), (1.5, 0.0, 3.1)), blocked=[np.zeros((0, 2)), ring])
    window = np.array([[10.0, -3.0], [20.0, -3.0], [20.0, 3.0], [10.0, 3.0]])  # a window: not round
    model.holes.append(window)
    model.hole_blocked.append(np.zeros((0, 2)))
    targets = datum_hole_targets(model)
    assert len(targets) == 2
    assert targets[0][1] == pytest.approx(3.1, abs=0.01)
    assert targets[1][1] == pytest.approx(1.8, abs=0.02)


# --------------------------------------------------------------------------- 2. datum poses from hole pairs
def _segment_of(raw_pts, pose):
    from domains.production.engine.inspect.registration import PlaneSegment

    n = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    top = raw_pts[np.abs((raw_pts - pose[:3, 3]) @ n) < 0.2]
    return PlaneSegment(n=n, d=float(np.median(top @ n)), indices=np.zeros(0, dtype=np.int64), area=1000.0,
                        centroid=top.mean(axis=0), rms=0.01)


def test_hole_pairs_give_the_datum_pose_where_its_outline_cannot():
    """A strip datum whose long sides are bends lies inside a larger flat sheet (its neighbours not formed yet):
    the outline cannot fix it across the strip, its two holes can."""
    from domains.production.engine.inspect.registration import hole_pair_hypotheses

    cfg = InspectConfig()
    model = _strip_model()
    cad = _sheet_points(model, extra=[(-40.0, 40.0, -20.0, -6.0)])
    pose = _pose()
    raw = _apply(pose, cad)
    hyps = hole_pair_hypotheses(raw, [_segment_of(raw, pose)], SimpleNamespace(t=model.t), model, cfg, 0.42,
                                mirrored=False)
    assert hyps
    best = min(_err(h.T, pose) for h in hyps)
    assert best[0] < 0.3 and best[1] < 0.5
    assert all(h.info.get("hole_pair_hits") == 2 for h in hyps)
    # a datum with a single round hole gets none (one hole fixes no rotation)
    one = _strip_model(holes=((-30.0, 0.0, 3.1),))
    assert hole_pair_hypotheses(_apply(pose, _sheet_points(one)), [_segment_of(raw, pose)], SimpleNamespace(t=one.t),
                                one, cfg, 0.42, mirrored=False) == []
    # and nothing is proposed when off: the configured count is 0
    assert hole_pair_hypotheses(raw, [_segment_of(raw, pose)], SimpleNamespace(t=model.t), model,
                                replace(cfg, hole_pair_hypotheses=0), 0.42, mirrored=False) == []


# --------------------------------------------------------------------------- 3. the fine search follows its window's edge
def test_fine_search_walks_past_its_window_edge():
    from domains.production.engine.inspect.registration import fine_search

    cfg = InspectConfig()
    model = _strip_model()
    pts = _sheet_points(model)
    centre = np.array([*model.polygon.mean(axis=0), 0.0])
    start = _inplane(np.eye(4), 4.5, centre)  # raw == model frame: the truth is the identity
    walked = fine_search(pts, start, model, cfg)
    stuck = fine_search(pts, start, model, replace(cfg, fine_search_walks=0))
    assert _err(walked, np.eye(4), centre)[0] < 0.6
    assert _err(stuck, np.eye(4), centre)[0] > 2.0


# --------------------------------------------------------------------------- 4. a weakly determined in-plane direction
def _one_free_end_strip():
    """A 155 x 28 mm strip bounded by bends on three sides; its one free end and two holes 12 mm apart near the
    middle fix the in-plane rotation only weakly (between 1 and 3 effective observations)."""
    return _strip_model(free_ends=(False, True), holes=((-0.3, 0.0, 2.7), (11.7, 0.0, 2.7)), x0=-77.5, x1=77.5,
                        half_w=14.2, t=2.0)


def test_refine_moves_a_weakly_determined_direction_to_its_optimum():
    from domains.production.engine.inspect.registration import refine_datum

    cfg = InspectConfig()
    model = _one_free_end_strip()
    pts = _sheet_points(model, step=0.45)
    centre = np.array([*model.polygon.mean(axis=0), 0.0])
    start = _inplane(np.eye(4), 1.5, centre)
    fit = refine_datum(pts, start, model, cfg, 0.45)
    # weak: still reported undetermined (positions across it stay withheld), but moved to its optimum
    assert cfg.refine_min_information <= fit.detail["min_information"] < cfg.degenerate_min_information
    assert fit.undetermined.size > 0
    assert _err(fit.T, np.eye(4), centre)[0] < 0.3
    frozen = refine_datum(pts, start, model, replace(cfg, refine_min_information=cfg.degenerate_min_information),
                          0.45)
    assert _err(frozen.T, np.eye(4), centre)[0] > 1.0


# --------------------------------------------------------------------------- 5. a child's pose on its fitted plane
def test_pose_on_plane_turns_the_kinematic_pose_about_the_hinge():
    from domains.production.engine.inspect.articulated import pose_on_plane

    c, s = math.cos(math.radians(1.2)), math.sin(math.radians(1.2))
    A = np.eye(4)
    A[:3, :3] = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]  # kinematic pose 1.2 deg off about the hinge (x axis)
    n0, d0 = np.array([0.0, 0.0, 1.0]), 0.0  # the fitted plane
    p0, n_cad = np.array([10.0, 40.0, 0.0]), np.array([0.0, 0.0, 1.0])
    hinge = np.zeros(3)
    out = pose_on_plane(A, p0, n_cad, n0, d0, hinge, 10.0)
    assert np.allclose(out[:3, :3] @ n_cad, n0, atol=1e-9)
    assert abs(float(_apply(out, p0[None])[0] @ n0) - d0) < 1e-9
    assert np.allclose(_apply(out, hinge[None])[0], hinge, atol=1e-9)  # turned about the hinge
    far = np.array([10.0, 80.0, 0.0])  # a grandchild hinge across the panel stays on the sheet
    assert abs(float(_apply(out, far[None])[0] @ n0)) < 1e-9
    assert abs(float(_apply(A, far[None])[0] @ n0)) > 1.0
    # a riser (no turn allowed): only moved along the normal, its orientation is the kinematics'
    kept = pose_on_plane(A, p0, n_cad, n0, d0, hinge, 0.0)
    assert np.allclose(kept[:3, :3], A[:3, :3])
    assert abs(float(_apply(kept, p0[None])[0] @ n0) - d0) < 1e-9
    # a kinematic normal far off the fitted one is left alone
    B = np.eye(4)
    B[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    assert np.allclose(pose_on_plane(B, p0, n_cad, n0, d0, hinge, 10.0), B)


# --------------------------------------------------------------------------- 6. the station spread joins an angle's U
def test_station_spread_covers_a_systematic_trend_not_a_lone_outlier():
    from domains.production.engine.inspect.metrology import robust_levels, station_spread, u_budget

    cfg = InspectConfig()
    trend = np.linspace(88.95, 89.36, 11)
    value, sig, how, used = robust_levels(trend)
    spread = station_spread(used)
    q10, q90 = np.quantile(trend, [0.1, 0.9])
    assert spread == pytest.approx((q90 - q10) / (2 * math.sqrt(3)))
    base = {"station": 0.04, "tilt_deg": 0.12}
    assert u_budget("angle", {**base, "spread": spread}, cfg) > u_budget("angle", base, cfg) + 0.04
    flat = np.full(11, 89.5) + np.array([0.0, 0.01, -0.01, 0.0, 0.02, -0.02, 0.0, 0.01, -0.01, 0.0, 3.0])
    v2, _, how2, used2 = robust_levels(flat)
    assert how2 == "DOMINANT_LEVEL" and len(used2) == 10
    assert station_spread(used2) < 0.02


# --------------------------------------------------------------------------- 7. config
def test_iteration4_config_fields_and_ranges():
    cfg = InspectConfig()
    assert cfg.refine_min_information == 1.0
    assert cfg.fine_search_walks == 2
    assert cfg.hole_core_clearance_mm == 0.75
    assert cfg.hole_pair_hypotheses == 4 and cfg.hole_pair_max_holes == 8
    assert cfg.child_orient_max_deg == 10.0 and cfg.child_orient_min_depth_mm == 8.0
    for key, bad in (("refine_min_information", 0.0), ("fine_search_walks", 11), ("hole_core_clearance_mm", -1.0),
                     ("hole_pair_max_holes", 1), ("child_orient_max_deg", 30.0), ("hole_pair_hypotheses", 2.5)):
        with pytest.raises(ConfigError):
            load_config(None, **{key: bad})


# --------------------------------------------------------------------------- 8. refinement slots go to distinct poses
def test_candidates_that_refine_onto_one_pose_share_one_slot():
    from domains.production.engine.inspect.engine import _same_cluster

    model = _strip_model()
    b = np.linalg.inv(_pose())  # a raw -> model candidate

    def model_side(M):  # a motion in the model frame applied after the candidate
        return M @ b

    def turn_about_normal(deg):
        c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        R = np.eye(4)
        R[:2, :2] = [[c, -s_], [s_, c]]
        return R  # about the datum normal through the datum's reference point (the model origin here)

    shift = np.eye(4)
    shift[:3, 3] = model.f1 * 1.2
    lift = np.eye(4)
    lift[:3, 3] = model.n0 * model.t
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # face down: 180 deg about the datum's x axis
    assert _same_cluster(model_side(turn_about_normal(1.5)), b, model, model.t)  # converges onto one pose
    assert _same_cluster(model_side(shift), b, model, model.t)
    assert not _same_cluster(model_side(lift), b, model, model.t)  # the other skin height: its own hypothesis
    assert not _same_cluster(model_side(turn_about_normal(6.0)), b, model, model.t)
    assert not _same_cluster(model_side(flip), b, model, model.t)


# --------------------------------------------------------------------------- 9. a joggle's riser from its step
def _hinge_frame(rho_iml=-0.1, delta=-1.5, band=8.1):
    from domains.production.engine.inspect.articulated import HingeFrame

    return HingeFrame(o=np.zeros(3), a=np.array([0.0, 0.0, 1.0]), u=np.array([1.0, 0.0, 0.0]),
                      v=np.array([0.0, 1.0, 0.0]), z0=0.0, z1=60.0, rho_iml=rho_iml, delta=delta, band=band)


def _joggle_band(step=1.7, t=1.5, run=2.6, tilt_deg=0.0, skins=("top", "bottom"), z_step=1.0, du=0.05):
    """A joggle in a hinge frame (parent skins at v = -0.1 and -1.6 for u < 0): a curved S riser with no flat
    material from u = 0 to ``run`` (a smooth step), then the far flange ``step`` higher, tilted by ``tilt_deg``."""
    prof = []
    for u in np.arange(-15.0, 0.0, du):
        prof += [(u, -0.1, "top"), (u, -0.1 - t, "bottom")]
    for u in np.arange(0.0, run, du):
        h = step * 0.5 * (1.0 - math.cos(math.pi * u / run))  # an S: zero slope at both ends, steepest mid-way
        prof += [(u, -0.1 + h, "top"), (u, -0.1 - t + h, "bottom")]
    tl = math.tan(math.radians(tilt_deg))
    for u in np.arange(run, 20.0, du):
        dv = (u - run) * tl
        prof += [(u, -0.1 + step + dv, "top"), (u, -0.1 - t + step + dv, "bottom")]
    prof = np.asarray([(u, v) for u, v, s in prof if s in skins])
    zs = np.arange(0.0, 60.0, z_step)
    return (np.repeat(prof[:, 0], len(zs)), np.repeat(prof[:, 1], len(zs)), np.tile(zs, len(prof)))


def _joggle_stubs(monkeypatch, riser_len=3.1, far_start=3.0, t=1.5):
    """Parent / bend stubs for ``_joggle_step``: the kinematics put the far flange ``riser_len sin(theta)`` higher."""
    from domains.production.engine.inspect import articulated as AR

    monkeypatch.setattr(AR, "_joggle_offset", lambda part, bend, grand, parent, frame, theta: riser_len * math.sin(theta))
    tangent = np.array([far_start, 0.0, 0.0])
    gb = SimpleNamespace(locus=lambda name: (tangent, tangent + np.array([0.0, 0.0, 60.0])))
    part = SimpleNamespace(t=t, bends={1: gb})
    bend = SimpleNamespace(sweep=math.radians(35.0))
    parent = SimpleNamespace(t=t, n0=np.array([0.0, 1.0, 0.0]), d0=-0.1, A=np.eye(4))
    return part, bend, parent


def test_a_joggles_far_flange_step_gives_its_riser_angle(monkeypatch):
    from domains.production.engine.inspect.articulated import _joggle_step

    cfg = InspectConfig()
    part, bend, parent = _joggle_stubs(monkeypatch)
    fr = _hinge_frame()
    u, v, z = _joggle_band(step=1.7)
    jog = _joggle_step(part, bend, 1, parent, fr, z, u, v, cfg)
    assert jog is not None and jog.two_skins
    assert jog.step == pytest.approx(1.7, abs=0.02)
    assert math.degrees(jog.theta) == pytest.approx(math.degrees(math.asin(1.7 / 3.1)), abs=0.3)
    # the riser's mid-plane passes through the S's middle: both skins are half way up the step at u = 1.3, where
    # the mid-surface is at v = -0.85 + 0.85
    assert jog.rho_mid == pytest.approx(-1.3 * math.sin(jog.theta), abs=0.05)
    # one visible skin (the other occluded) gives the same step: the skin is moved to the mid-plane by t/2
    u1, v1, z1 = _joggle_band(step=1.7, skins=("top",))
    one = _joggle_step(part, bend, 1, parent, fr, z1, u1, v1, cfg)
    assert one is not None and not one.two_skins and one.step == pytest.approx(1.7, abs=0.03)


def test_no_joggle_step_without_a_parallel_far_flange_at_a_formed_step(monkeypatch):
    from domains.production.engine.inspect.articulated import _joggle_step

    cfg = InspectConfig()
    part, bend, parent = _joggle_stubs(monkeypatch)
    fr = _hinge_frame()
    for kw in ({"step": 0.0}, {"step": 0.5}, {"step": 1.7, "tilt_deg": 15.0}):
        u, v, z = _joggle_band(**kw)
        assert _joggle_step(part, bend, 1, parent, fr, z, u, v, cfg) is None, kw


def test_joggle_pairs_are_parallel_flanges_not_u_turns():
    from domains.production.engine.inspect.articulated import _joggle_pair

    up = np.array([0.0, 0.0, 1.0])
    panels = [SimpleNamespace(n0=up), SimpleNamespace(n0=np.array([1.0, 0.0, 0.0])), SimpleNamespace(n0=up),
              SimpleNamespace(n0=-up)]
    bend = SimpleNamespace(parent=0)
    joggle = SimpleNamespace(panels=panels, bends={5: SimpleNamespace(child=2)})
    hook = SimpleNamespace(panels=panels, bends={5: SimpleNamespace(child=3)})
    assert _joggle_pair(joggle, bend, 5)
    assert not _joggle_pair(hook, bend, 5)


def test_a_model_riser_lies_on_its_line_pair():
    from domains.production.engine.inspect.articulated import PanelFit, _model_riser

    fr = _hinge_frame()
    th, rho_i = math.radians(33.0), 0.12
    bend = SimpleNamespace(model=SimpleNamespace(neutral_length_mm=0.36), sweep=math.radians(35.0),
                           p_out=np.array([-1.0, 0.0, 0.0]), axis_point=np.zeros(3), axis_dir=np.array([0.0, 0.0, 1.0]),
                           oml_skin=1, child=1)
    part = SimpleNamespace(t=1.5, panels=[None, SimpleNamespace(p0=np.array([1.0, 0.5, 0.0]),
                                                                n0=np.array([-math.sin(math.radians(35.0)),
                                                                             math.cos(math.radians(35.0)), 0.0]))])
    parent = PanelFit(0, "FITTED", np.eye(4), n0=np.array([0.0, 1.0, 0.0]), d0=-0.1, t=1.5)
    fit = _model_riser(part, bend, parent, fr, th, rho_i, np.arange(4), (0, 1), 3.2)
    assert fit.state == "FITTED" and fit.model_plane and fit.model_u_deg == pytest.approx(3.2)
    d = np.array([math.cos(th), math.sin(th), 0.0])
    n = fr.line_normal(th)
    for s in (0.5, 1.5, 2.5):
        p_in = s * d + rho_i * n                 # on the inner line
        p_out = s * d + (rho_i + fr.delta) * n   # on the outer line, one thickness off
        planes = [fit.skin_plane(0), fit.skin_plane(1)]
        dist = sorted(abs(float(pl.distance(p[None])[0])) for pl in planes for p in (p_in, p_out))
        assert dist[0] < 1e-9 and dist[1] < 1e-9  # each line lies on one of the riser's skins
    assert abs(float(abs(fit.n0 @ n)) - 1.0) < 1e-9  # the plane is the line's, at the step's angle


# --------------------------------------------------------------------------- 10. twins told apart by their holes
def _holed_sheet(fill=False, t=1.5, step=0.4, seed=3):
    """A 60 x 40 mm sheet (skin 0 at z = 0, normal +z) with a 6 mm hole at (10, 5): open, or filled with material."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(-30.0, 30.0, step), np.arange(-20.0, 20.0, step))
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1) + rng.uniform(-0.1, 0.1, (xs.size, 2))
    if not fill:
        uv = uv[np.linalg.norm(uv - np.array([10.0, 5.0]), axis=1) > 3.0]
    return np.vstack([np.column_stack([uv, np.full(len(uv), z)]) for z in (0.0, -t)])


def _holed_part_and_hyp(pts, T=np.eye(4), owner_value=-1):
    from domains.production.engine.inspect.articulated import PanelFit

    a = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
    hole = np.stack([10.0 + 3.0 * np.cos(a), 5.0 + 3.0 * np.sin(a), np.zeros_like(a)], axis=1)
    part = SimpleNamespace(t=1.5, panels=[SimpleNamespace(model=SimpleNamespace(holes_mm=[hole.tolist()]))])
    pf = PanelFit(0, "DATUM", np.eye(4), n0=np.array([0.0, 0.0, 1.0]), d0=0.0, t=1.5)
    art = SimpleNamespace(panels=[pf], owner=np.full(len(pts), owner_value, dtype=np.int64))
    vox = SimpleNamespace(inverse=np.arange(len(pts)))
    return part, SimpleNamespace(fit=SimpleNamespace(T=T), art=art), vox


def test_hole_evidence_confirms_an_open_hole_and_contradicts_a_filled_one():
    from domains.production.engine.inspect.engine import _hole_evidence

    cfg = InspectConfig()
    open_pts = _holed_sheet(fill=False)
    part, h, vox = _holed_part_and_hyp(open_pts)
    assert _hole_evidence(h, open_pts, vox, part, cfg, 0.4) == (1, 0)
    filled = _holed_sheet(fill=True)
    part, h, vox = _holed_part_and_hyp(filled)
    assert _hole_evidence(h, filled, vox, part, cfg, 0.4) == (0, 1)
    # material in the hole owned by another fitted panel (a tab still flat in its window) is not a contradiction
    part, h, vox = _holed_part_and_hyp(filled, owner_value=7)
    assert _hole_evidence(h, filled, vox, part, cfg, 0.4) == (0, 0)
    # an unobserved area is not judged
    part, h, vox = _holed_part_and_hyp(open_pts[np.abs(open_pts[:, 0] - 10.0) > 12.0])
    assert _hole_evidence(h, open_pts[np.abs(open_pts[:, 0] - 10.0) > 12.0], vox, part, cfg, 0.4) == (0, 0)


def test_a_twin_whose_holes_land_on_sheet_loses_to_one_whose_holes_are_open():
    from domains.production.engine.inspect.engine import _ambiguity

    cfg = InspectConfig()
    rot = np.diag([-1.0, -1.0, 1.0, 1.0])
    # the best total decodes in the planned order (b2, rank 1, formed first), its twin out of order
    best = _hyp_twin(np.eye(4), 2.040, ["NOT_YET_FORMED", "FORMED"])
    rival = _hyp_twin(rot, 2.033, ["FORMED", "NOT_YET_FORMED"])
    ev = {id(best): (1, 7), id(rival): (10, 0)}
    chosen = _ambiguity([best, rival], _twin_part(0.32), cfg, evidence=lambda h: ev[id(h)])[3]
    assert chosen is rival  # the holes decide before the order prior
    # one hole's difference is not enough to overrule the totals and the order
    ev = {id(best): (9, 1), id(rival): (10, 1)}
    assert _ambiguity([best, rival], _twin_part(0.32), cfg, evidence=lambda h: ev[id(h)])[3] is best
    # a clearly worse rival (outside the ambiguity margin) is never chosen, holes or not
    far = _hyp_twin(rot, 2.040 - 2 * cfg.ambiguity_margin, ["NOT_YET_FORMED", "FORMED"])
    ev = {id(best): (0, 9), id(far): (10, 0)}
    assert _ambiguity([best, far], _twin_part(0.32), cfg, evidence=lambda h: ev[id(h)])[3] is best


def _twin_part(rms):
    rot = np.diag([-1.0, -1.0, 1.0, 1.0])
    cand = SimpleNamespace(kind="ROTATION", rms_mm=rms, transform=rot.tolist(), bend_permutation={"b1": "b2", "b2": "b1"})
    sym = SimpleNamespace(candidates=[cand], threshold_rms_mm=0.3)
    stages = [SimpleNamespace(symmetry=sym, stage_model_id=f"S{k}") for k in range(3)]
    pts = np.array([[x, y, 0.0] for x in np.linspace(-50, 50, 21) for y in np.linspace(-30, 30, 13)])
    bends = [SimpleNamespace(uid="b1", index=0, rank=2), SimpleNamespace(uid="b2", index=1, rank=1)]
    return SimpleNamespace(n=2, bends=bends, compiled=SimpleNamespace(stages=stages),
                           cloud=lambda sid: SimpleNamespace(points=pts), stage_for_formed=lambda formed: None)


def _hyp_twin(T, total, states):
    return SimpleNamespace(fit=SimpleNamespace(T=T), total=total, explained=0.99, explained_tight=0.97,
                           art=object(), decodes=[SimpleNamespace(state=s) for s in states])


def test_iteration4_joggle_and_hole_evidence_config():
    cfg = InspectConfig()
    assert cfg.joggle_theta_tol_deg == 8.0 and cfg.joggle_parallel_deg == 6.0
    assert cfg.joggle_riser_length_u_fraction == 0.1 and cfg.hole_evidence_min_diff == 2
    for key, bad in (("joggle_window_mm", 0.0), ("joggle_parallel_deg", 50.0), ("joggle_min_points", 0),
                     ("joggle_riser_length_u_fraction", 1.5), ("hole_evidence_open_fraction", 1.5),
                     ("hole_evidence_min_diff", 0), ("hole_evidence_margin_mm", 0.0)):
        with pytest.raises(ConfigError):
            load_config(None, **{key: bad})


# --------------------------------------------------------------------------- 11. a bowed panel re-seated near its bend
def _strip_two_skins(offset, t=0.92, tilt_deg=0.0, step=0.35, seed=5):
    """Station-strip material of a panel whose OML (z = 0 nominally, outward +z) sits ``offset`` mm out, tilted."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(2.0, 40.0, step), np.arange(0.0, 30.0, step))
    x, y = xs.ravel(), ys.ravel()
    tl = math.tan(math.radians(tilt_deg))
    pts = [np.column_stack([x, y, offset + x * tl + dz + rng.normal(0, 0.01, len(x))]) for dz in (0.0, -t)]
    return np.vstack(pts)


def test_a_bowed_panels_plane_is_reseated_on_its_own_material_near_the_bend():
    from domains.production.engine.inspect.geom import Plane
    from domains.production.engine.inspect.metrology import _reseat_plane

    cfg = InspectConfig()
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)  # the region-wide OML plane
    q = _reseat_plane(_strip_two_skins(1.3), plane, 0.92, cfg)
    assert q is not None and float(q.n @ plane.n) > 0.9999
    assert float(-q.d) == pytest.approx(-1.3, abs=0.02)  # the OML now sits on the strip's outer skin
    assert _reseat_plane(_strip_two_skins(0.15), plane, 0.92, cfg) is None  # already seated: the stations find it
    assert _reseat_plane(_strip_two_skins(1.3, tilt_deg=12.0), plane, 0.92, cfg) is None  # too far turned: refused
    assert _reseat_plane(_strip_two_skins(5.0), plane, 0.92, cfg) is None  # beyond the search: not this panel


# --------------------------------------------------------------------------- 12. a window keeps its wider core clearance
def test_a_round_bore_gets_the_tight_core_clearance_and_a_window_keeps_the_wide_one():
    from domains.production.engine.inspect.registration import DatumModel

    a = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
    bore = np.stack([-20.0 + 3.1 * np.cos(a), 3.1 * np.sin(a)], axis=1)  # a countersunk hole: open bore r = 1.8
    ring = np.array([[-20.0 + r * math.cos(t_), r * math.sin(t_)] for r in np.arange(1.8, 3.1, 0.1)
                     for t_ in np.linspace(0.0, 2 * math.pi, 90, endpoint=False)])
    window = np.array([[5.0, -8.0], [38.0, -8.0], [38.0, 8.0], [5.0, 8.0]])  # a window with a tab flat in it
    tab = np.array([[x, y] for x in np.arange(7.0, 30.0, 0.1) for y in np.arange(-6.0, 6.0, 0.1)])
    poly = np.array([[-40.0, -15.0], [40.0, -15.0], [40.0, 15.0], [-40.0, 15.0]])
    m = DatumModel(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                   1.5, poly, np.array([True, True, True, True]), [bore, window], 80 * 30 - 30 * 16,
                   (-40.0, -15.0, 40.0, 15.0), hole_blocked=[ring, tab])
    pix = 0.25
    k, origin, kin, kf = m.kernel(pix, 8.0, 2.0, 4.0, hole_weight=1.0, clearance=0.75, clearance_window=1.5)

    def forbidden(x, y):
        j, i = int(round((x - origin[0]) / pix)), int(round((y - origin[1]) / pix))
        return bool(kf[i, j] > 0)

    assert forbidden(-20.0, 0.0) and forbidden(-20.0 + 0.9, 0.0)  # the open bore: its core reaches 0.75 from the ring
    assert not forbidden(29.9 + 1.0, 0.0)  # 1 mm beside the tab (kerf, flare): not penalised in a window
    assert forbidden(29.9 + 1.75, 0.0)  # but the window's open gap beyond 1.5 mm is


def test_iteration4_reseat_and_window_clearance_config():
    cfg = InspectConfig()
    assert cfg.window_core_clearance_mm == 1.5 and cfg.hole_core_clearance_mm == 0.75
    assert cfg.local_reseat_min_mm == 0.3 and cfg.local_reseat_max_mm == 3.0
    assert cfg.local_reseat_max_tilt_deg == 6.0 and cfg.local_reseat_min_share == 0.25
    for key, bad in (("window_core_clearance_mm", 0.0), ("local_reseat_max_mm", -1.0),
                     ("local_reseat_max_tilt_deg", 0.0), ("local_reseat_min_share", 2.0)):
        with pytest.raises(ConfigError):
            load_config(None, **{key: bad})
