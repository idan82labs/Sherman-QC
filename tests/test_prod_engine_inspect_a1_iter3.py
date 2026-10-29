"""A1 iteration 3 unit tests of the per-scan engine (E3): half-made, one-sided and windowed sheet-metal geometry.

Every rule here is geometric; no part, family or file is named. Pure numpy (no OCP, no E1/E2 geometry,
no physical data).
Run: PYT tests/test_prod_engine_inspect_a1_iter3.py -q
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.inspect.config import InspectConfig, load_config


# --------------------------------------------------------------------------- 1. a tab lying flat in its window
def _window_panel():
    """A 100 x 80 mm panel (z = 0, skin 0 outward +z) with a 40 x 30 mm window at x 30..70, y 20..50."""
    outline = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 80.0, 0.0), (0.0, 80.0, 0.0)]
    window = [(30.0, 20.0, 0.0), (70.0, 20.0, 0.0), (70.0, 50.0, 0.0), (30.0, 50.0, 0.0)]
    model = SimpleNamespace(outer_boundary_mm=outline, holes_mm=[window])
    return SimpleNamespace(panels=[SimpleNamespace(model=model)])


def test_material_inside_a_window_is_not_the_panel_around_it():
    from domains.production.engine.inspect.articulated import in_footprint

    part = _window_panel()
    q = np.array([
        [50.0, 35.0, 0.0],    # window core: a tab lying flat in it (not yet formed)
        [30.3, 35.0, 0.0],    # within the margin of the window's edge: still the panel's cut edge
        [10.0, 10.0, 0.0],    # panel
        [100.4, 40.0, 0.0],   # just past the outline (a cut a few tenths long): kept
        [103.0, 40.0, 0.0],   # beyond the grown outline
    ])
    keep = in_footprint(q, part, 0, np.eye(4), np.array([0.0, 0.0, 1.0]), 0.6)
    assert keep.tolist() == [False, True, True, True, False]
    # a small round hole shrunk by the margin has no core left: nothing changes there
    hole = [(10.0 + 2.0 * math.cos(a), 60.0 + 2.0 * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    part.panels[0].model.holes_mm = [hole]
    assert in_footprint(np.array([[10.0, 60.0, 0.0]]), part, 0, np.eye(4), np.array([0.0, 0.0, 1.0]), 3.0).all()


# --------------------------------------------------------------------------- 2. the free edge of a lip on a box's rim
def test_a_free_edge_does_not_run_into_a_coplanar_panel_beyond_its_outline():
    """A 12 mm lip whose grown region ran (through touching coplanar tabs) into the lip 300 mm away."""
    from domains.production.engine.inspect.metrology import PointIndex, _far_free, _panel_footprint_region
    from domains.production.engine.inspect.geom import Plane

    cfg = InspectConfig()
    t = 1.5
    ys = np.concatenate([np.arange(0.0, 12.0, 0.2), np.arange(300.0, 312.0, 0.2)])
    zs = np.arange(-5.0, 5.0, 0.25)
    yy, zz = np.meshgrid(ys, zs)
    skins = [np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()]) for x in (0.0, -t)]
    pts = np.vstack(skins)
    vox = SimpleNamespace(inverse=np.arange(len(pts)), points=pts)
    idx = PointIndex(pts, vox)
    outline = [(0.0, 0.0, -10.0), (0.0, 12.0, -10.0), (0.0, 12.0, 10.0), (0.0, 0.0, 10.0)]
    part = SimpleNamespace(panels=[SimpleNamespace(model=SimpleNamespace(outer_boundary_mm=outline, holes_mm=[]))])
    art = SimpleNamespace(panels=[SimpleNamespace(region=np.arange(len(pts)), voxels=np.arange(len(pts)), A=np.eye(4),
                                                  n0=np.array([1.0, 0.0, 0.0]))])
    plane = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    origin, w, a = np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    whole = idx.region_mask(("panel", 0), art.panels[0].region)
    far_old = _far_free(pts, whole, plane, t, origin, w, a, 6.0, 0.5, 0.2, 0.5, cfg).value
    mask = _panel_footprint_region(idx, part, art, 0, cfg.free_edge_footprint_margin_mm)
    far_new = _far_free(pts, mask, plane, t, origin, w, a, 6.0, 0.5, 0.2, 0.5, cfg).value
    assert far_old > 300.0  # the region alone takes the other lip's edge
    assert far_new == pytest.approx(11.8, abs=0.3)
    # a flange cut long is still followed past the margin over contiguous material
    long_ys = np.arange(0.0, 20.0, 0.2)
    yy, zz = np.meshgrid(long_ys, zs)
    lp = np.vstack([np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()]) for x in (0.0, -t)])
    lvox = SimpleNamespace(inverse=np.arange(len(lp)), points=lp)
    lidx = PointIndex(lp, lvox)
    lart = SimpleNamespace(panels=[SimpleNamespace(region=np.arange(len(lp)), voxels=np.arange(len(lp)), A=np.eye(4),
                                                   n0=np.array([1.0, 0.0, 0.0]))])
    lmask = _panel_footprint_region(lidx, part, lart, 0, cfg.free_edge_footprint_margin_mm)
    assert _far_free(lp, lmask, plane, t, origin, w, a, 6.0, 0.5, 0.2, 0.5, cfg).value == pytest.approx(19.8, abs=0.3)


# --------------------------------------------------------------------------- 3. a joggle half made
def _frame(rho_iml=-0.1, delta=-1.5, band=8.1):
    from domains.production.engine.inspect.articulated import HingeFrame

    return HingeFrame(o=np.zeros(3), a=np.array([0.0, 0.0, 1.0]), u=np.array([1.0, 0.0, 0.0]),
                      v=np.array([0.0, 1.0, 0.0]), z0=0.0, z1=60.0, rho_iml=rho_iml, delta=delta, band=band)


def _sweep(prof, width=60.0, z_step=1.0):
    prof = np.asarray(prof)
    zs = np.arange(0.0, width, z_step)
    return np.column_stack([np.repeat(prof[:, 0], len(zs)), np.repeat(prof[:, 1], len(zs)), np.tile(zs, len(prof))])


def _riser_then_tab(riser=3.75, t=1.5, step=0.05, tab_deg=35.0, tab=13.0):
    """Parent (u < 0) and a flat riser (u 0 .. riser), then a tab rising at ``tab_deg`` from u = riser: the
    first bend of a joggle still flat, the second formed (both skins, inner v = -0.1, outer v = -1.6)."""
    th = math.radians(tab_deg)
    prof = []
    for u in np.arange(-15.0, riser, step):
        prof += [(u, -0.1), (u, -0.1 - t)]
    for s in np.arange(0.0, tab, step):
        prof += [(riser + s * math.cos(th), -0.1 + s * math.sin(th)),
                 (riser + s * math.cos(th) + t * math.sin(th), -0.1 - t * math.cos(th) + s * math.sin(th))]
    return _sweep(prof)


def _section(prof):
    fr = _frame()
    z, uc, vc = fr.coords(_sweep(prof))
    keep = np.hypot(uc, vc) < fr.band
    return fr, z[keep], uc[keep], vc[keep]


def test_a_formed_tab_beyond_a_flat_riser_belongs_to_the_bend_beyond_the_riser():
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    fr = _frame()
    z, uc, vc = fr.coords(_riser_then_tab())
    keep = np.hypot(uc, vc) < fr.band
    z, uc, vc = z[keep], uc[keep], vc[keep]
    thetas = np.radians(np.arange(-10.0, 60.0 + 1e-9, 1.0))
    plain = _hough(z, uc, vc, fr, thetas, cfg, fr.band, continuity=True)[0]
    assert math.degrees(thetas[int(np.argmax(plain))]) > 20.0  # the tab, read as this bend formed off its line
    lim = _hough(z, uc, vc, fr, thetas, cfg, fr.band, continuity=True, grand_u=3.9)[0]
    # this bend is flat (the decode's flat limit); the tab is the next bend's
    assert abs(math.degrees(thetas[int(np.argmax(lim))])) <= cfg.flat_sweep_max_deg


def test_a_whole_joggle_still_reads_its_riser_formed_with_the_grandchild_limit():
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    # a 35 deg riser 2.6 mm long, then a tail parallel to the parent (both joggle bends formed)
    th, t, prof = math.radians(35.0), 1.5, []
    for u in np.arange(-15.0, 0.0, 0.05):
        prof += [(u, -0.1), (u, -0.1 - t)]
    for s in np.arange(0.0, 2.6, 0.05):
        prof += [(s * math.cos(th), -0.1 + s * math.sin(th)),
                 (s * math.cos(th) + t * math.sin(th), -0.1 - t * math.cos(th) + s * math.sin(th))]
    for u in np.arange(2.6 * math.cos(th) + 0.3, 2.6 * math.cos(th) + 14.0, 0.05):
        prof += [(u, -0.1 + 1.7), (u, -0.1 - t + 1.7)]
    fr, z, uc, vc = _section(prof)
    thetas = np.radians(np.arange(-10.0, 60.0 + 1e-9, 1.0))
    sc = _hough(z, uc, vc, fr, thetas, cfg, fr.band, continuity=True, grand_u=3.9)[0]
    assert math.degrees(thetas[int(np.argmax(sc))]) >= 15.0


def test_the_grandchild_limit_only_bounds_the_side_toward_it():
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    # a formed 35 deg flange whose bend line sits 1 mm toward the parent (the side away from the riser's end)
    th, t, d, prof = math.radians(35.0), 1.5, -1.0, []
    for u in np.arange(-15.0, d, 0.05):
        prof += [(u, -0.1), (u, -0.1 - t)]
    for s in np.arange(0.0, 12.0, 0.05):
        prof += [(d + s * math.cos(th), -0.1 + s * math.sin(th)),
                 (d + s * math.cos(th) + t * math.sin(th), -0.1 - t * math.cos(th) + s * math.sin(th))]
    fr, z, uc, vc = _section(prof)
    thetas = np.radians(np.arange(-10.0, 60.0 + 1e-9, 1.0))
    sc, rh, _, _ = _hough(z, uc, vc, fr, thetas, cfg, fr.band, grand_u=3.9)
    k = int(np.argmax(sc))
    assert math.degrees(thetas[k]) == pytest.approx(35.0, abs=1.5)
    assert rh[k] == pytest.approx(-0.1 - d * math.sin(th), abs=0.3)  # measured where it is, not pulled to nominal


def test_a_short_childs_line_support_stops_where_the_child_ends():
    """A 35 deg riser 2.4 mm long and, beyond its end, a longer two-skin run that a 12 deg line crosses (the
    sheet beyond the next bend): only the cut at the riser's end keeps that run from outscoring the riser."""
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    fr = _frame()

    def run(th_deg, s0, s1):
        th = math.radians(th_deg)
        d, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        return [tuple(s * d + rho * n) for s in np.arange(s0, s1, 0.05) for rho in (-0.1, -1.6)]

    z, uc, vc = fr.coords(_sweep(run(35.0, 1.0, 3.4) + run(12.0, 4.0, 8.0)))
    thetas = np.radians(np.arange(-10.0, 60.0 + 1e-9, 1.0))

    def best(cut):
        sc = _hough(z, uc, vc, fr, thetas, cfg, fr.band, continuity=True, s_cut=cut)[0]
        return math.degrees(thetas[int(np.argmax(sc))])

    assert best(None) < 20.0  # a shallow line through the run beyond the riser outscores it
    end = np.full(len(thetas), 3.6)
    # support stops where the riser ends: its own line wins (a 2.4 mm run fixes the angle to a few degrees)
    assert best((end, -1.0, 61.0)) == pytest.approx(35.0, abs=3.0)
    assert best((end, 100.0, 200.0)) < 20.0  # beside the next bend's axial span: no cut


# --------------------------------------------------------------------------- 4. one visible skin
def _formed_90(skins=("oml",), inner_density=1.0):
    """Parent (u < 0, both skins) and a child formed 90 deg at nominal: inner line u = 0.8, outer u = 2.3."""
    prof = []
    for u in np.arange(-15.0, -1.0, 0.05):
        prof += [(u, -0.8), (u, -2.3)]
    vs = np.arange(1.0, 12.0, 0.05)
    if "oml" in skins:
        prof += [(2.3, v) for v in vs]
    if "iml" in skins:
        prof += [(0.8, v) for v in vs[:: max(1, int(round(1.0 / inner_density)))]]
    return _sweep(prof)


def test_one_visible_skin_is_read_nearest_the_nominal_lines():
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    fr = _frame(rho_iml=-0.8, delta=-1.5, band=8.8)
    th = np.radians([90.0])
    for skins in (("oml",), ("iml",)):
        z, uc, vc = fr.coords(_formed_90(skins))
        keep = np.hypot(uc, vc) < 3 * fr.band
        _, rh, ni, no = _hough(z[keep], uc[keep], vc[keep], fr, th, cfg, 3 * fr.band)
        assert rh[0] == pytest.approx(-0.8, abs=0.15), skins  # the inner line at nominal, not one thickness off
    # both skins seen (a weak inner one): the reading that uses both stands
    z, uc, vc = fr.coords(_formed_90(("oml", "iml"), inner_density=0.25))
    keep = np.hypot(uc, vc) < 3 * fr.band
    _, rh, ni, no = _hough(z[keep], uc[keep], vc[keep], fr, th, cfg, 3 * fr.band)
    assert rh[0] == pytest.approx(-0.8, abs=0.15) and min(ni[0], no[0]) > 0


def test_a_slab_seen_on_one_skin_is_named_by_its_hinge_lines():
    from domains.production.engine.inspect.articulated import _one_skin_mid

    fr = _frame(rho_iml=-0.8, delta=-1.5, band=8.8)
    th = math.radians(90.0)
    nrm = fr.line_normal(th)  # +rho
    bend = SimpleNamespace(oml_skin=1)
    t, sig = 1.5, -1.0
    o_off = float(nrm @ fr.o)
    # the visible plane on the outer line (rho -2.3): it is the OML; the mid-plane lies t/2 toward the inner line
    mid, seen = _one_skin_mid(o_off - 2.3, nrm, fr, -0.8, t, sig, bend)
    assert seen == (1,) and mid == pytest.approx(o_off - 1.55)
    # on the inner line (rho -0.8): the IML
    mid, seen = _one_skin_mid(o_off - 0.8, nrm, fr, -0.8, t, sig, bend)
    assert seen == (0,) and mid == pytest.approx(o_off - 1.55)
    # a slab a few tenths off still takes the nearer line
    mid, seen = _one_skin_mid(o_off - 2.0, nrm, fr, -0.8, t, sig, bend)
    assert seen == (1,)


# --------------------------------------------------------------------------- 5. a near-symmetric part half made
class _Bends(list):
    pass


def _near_symmetric_part(rms: float):
    """Two bends (B1 rank 1, B2 rank 2) that a 180 deg rotation about z maps onto each other within ``rms``."""
    rot = np.diag([-1.0, -1.0, 1.0, 1.0])
    cand = SimpleNamespace(kind="ROTATION", rms_mm=rms, transform=rot.tolist(), bend_permutation={"b1": "b2", "b2": "b1"})
    sym = SimpleNamespace(candidates=[cand], threshold_rms_mm=0.3)
    stages = [SimpleNamespace(symmetry=sym, stage_model_id=f"S{k}") for k in range(3)]
    pts = np.array([[x, y, 0.0] for x in np.linspace(-50, 50, 21) for y in np.linspace(-30, 30, 13)])
    bends = [SimpleNamespace(uid="b1", index=0, rank=1), SimpleNamespace(uid="b2", index=1, rank=2)]
    return SimpleNamespace(n=2, bends=bends, compiled=SimpleNamespace(stages=stages),
                           cloud=lambda sid: SimpleNamespace(points=pts),
                           stage_for_formed=lambda formed: None)


def _hyp(T, total, states):
    return SimpleNamespace(fit=SimpleNamespace(T=T), total=total, explained=0.99, explained_tight=0.97,
                           art=object(), decodes=[SimpleNamespace(state=s) for s in states])


def test_the_order_prior_picks_the_in_order_image_of_a_near_symmetric_half_made_part():
    from domains.production.engine.inspect.engine import _ambiguity

    cfg = InspectConfig()
    rot = np.diag([-1.0, -1.0, 1.0, 1.0])
    wrong = _hyp(np.eye(4), 1.7570, ["NOT_YET_FORMED", "UNEXPECTED_FORMED"])  # out of order: B2 before B1
    right = _hyp(rot, 1.7564, ["FORMED", "NOT_YET_FORMED"])
    # E1 finds the rotation 0.32 mm RMS: above its 0.3 mm symmetry threshold, a near twin
    ambiguous, _, _, chosen = _ambiguity([wrong, right], _near_symmetric_part(0.32), cfg)
    assert chosen is right and not ambiguous  # a near twin never makes the pose ambiguous
    # a clearly worse near twin is not chosen, and an unrelated rotation (no symmetry) is never a twin
    worse = _hyp(rot, 1.7570 - 2 * cfg.ambiguity_margin, ["FORMED", "NOT_YET_FORMED"])
    assert _ambiguity([wrong, worse], _near_symmetric_part(0.32), cfg)[3] is wrong
    assert _ambiguity([wrong, right], _near_symmetric_part(5.0), cfg)[3] is wrong


def test_config_carries_the_iteration_3_settings_and_checks_them():
    cfg = InspectConfig()
    assert cfg.free_edge_footprint_margin_mm == 5.0
    assert cfg.one_skin_fraction == 0.05 and cfg.one_skin_tie_fraction == 0.1
    assert cfg.near_twin_rms_mm == 1.0
    for bad in ({"free_edge_footprint_margin_mm": 0.0}, {"one_skin_fraction": 1.5}, {"near_twin_rms_mm": -1.0}):
        with pytest.raises(Exception):
            load_config(None, **bad)


# --------------------------------------------------------------------------- 6. a stepped datum hole
def _hole_model(blocked):
    from domains.production.engine.inspect.registration import DatumModel

    poly = np.array([[-20.0, -15.0], [20.0, -15.0], [20.0, 15.0], [-20.0, 15.0]])
    ang = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
    hole = np.column_stack([3.1 * np.cos(ang), 3.1 * np.sin(ang)])
    return DatumModel(p0=np.zeros(3), n0=np.array([0.0, 0.0, 1.0]), f1=np.array([1.0, 0.0, 0.0]),
                      f2=np.array([0.0, 1.0, 0.0]), t=1.5, polygon=poly, free=np.zeros(4, dtype=bool), holes=[hole],
                      area=1200.0 - math.pi * 3.1 ** 2, bbox=(-20.0, -15.0, 20.0, 15.0), hole_blocked=[blocked])


def _disc(r0, r1, step=0.2):
    g = np.arange(-r1, r1 + 1e-9, step)
    xx, yy = np.meshgrid(g, g)
    p = np.column_stack([xx.ravel(), yy.ravel()])
    r = np.hypot(p[:, 0], p[:, 1])
    return p[(r >= r0) & (r < r1)]


def test_a_stepped_holes_own_bore_is_not_a_filled_hole_but_a_filled_through_hole_is():
    from domains.production.engine.inspect.registration import _inplane_constraints

    cfg = InspectConfig()
    around = _disc(3.1, 12.0)  # datum material around the hole (skin 0)
    bore = _disc(1.7, 3.1)  # the stepped hole's smaller bore: material on skin 1 inside the skin-0 loop

    def mismatch(model, uv):
        return _inplane_constraints(uv, model, [], cfg, 0.0, 0.2, np.zeros(2), 10.0, wide=False)["hole_mismatch"]

    # the CAD says the bore is there (compiled stage material in the hole): the scan showing it is the part
    assert mismatch(_hole_model(_disc(1.7, 2.8)), np.vstack([around, bore])) == 0
    # a through-hole (nothing compiled inside) with material in it: filled (wrong part)
    assert mismatch(_hole_model(np.zeros((0, 2))), np.vstack([around, bore, _disc(0.0, 1.7)])) == 1
    # stage material over a small sector only: the rest of the disc is judged, and it is filled
    sector = _disc(0.0, 2.8)
    sector = sector[(sector[:, 0] > 1.2) & (np.abs(sector[:, 1]) < 0.8)]
    assert mismatch(_hole_model(sector), np.vstack([around, _disc(0.0, 3.1)])) == 1


# --------------------------------------------------------------------------- 7. a short crease ends a flange only where it runs
def test_a_short_crease_ends_the_free_edge_only_at_the_stations_it_spans():
    from domains.production.engine.inspect.metrology import BendMeasure, _crease_covers, _crease_crossings

    n = 11
    zs = np.linspace(5.0, 295.0, n)  # a 300 mm flange's stations (line point at y = 0, axis along y)
    bm = BendMeasure(bend=0, ok=True, z=zs, valid=np.ones(n, bool),
                     sharp=np.column_stack([np.zeros(n), zs, np.zeros(n)]), line_dir=np.array([0.0, 1.0, 0.0]))
    bm.line_point = np.zeros(3)
    bm.width = 29.0
    w = np.array([1.0, 0.0, 0.0])
    poly = np.array([[6.9, 56.7, 0.3], [6.9, 83.2, 0.3]])  # a 26.5 mm crease
    short = (np.array([6.9, 60.0, 0.3]), np.array([0.0, 1.0, 0.0]), 0.2, "C1", poly)
    out = _crease_crossings(bm, w, 1.0, [short])
    covered = [i for i, z in enumerate(zs) if 56.7 - 14.5 <= z <= 83.2 + 14.5]
    assert len(out) == len(covered) and 0 < len(out) < 0.6 * n  # too few stations to end the side
    assert _crease_covers(short, bm) is True and _crease_covers(short, bm, 250.0) is False
    # a crease without a recorded extent keeps the old reading (it spans the flange)
    assert len(_crease_crossings(bm, w, 1.0, [short[:4]])) == n
    # a crease outside the stations' span is no crease of this side at all
    far = (np.array([6.9, 400.0, 0.3]), np.array([0.0, 1.0, 0.0]), 0.2, "C2", poly + np.array([0.0, 350.0, 0.0]))
    assert _crease_covers(far, bm) is False
