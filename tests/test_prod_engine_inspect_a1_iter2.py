"""A1 iteration 2 unit tests of the per-scan engine (E3): a datum that leaves an in-plane direction free
gets it from the cut ends of fitted flanges that no bend moves along it.

Pure numpy (no OCP, no E1/E2 geometry, no physical data).
Run: PYT tests/test_prod_engine_inspect_a1_iter2.py -q
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.inspect.config import InspectConfig, load_config
from domains.production.engine.inspect.registration import (
    ChildEdgeRows,
    DatumModel,
    InPlaneFit,
    child_edge_rows,
    resolve_free_directions,
)

T_SHEET = 2.0


def _datum_model() -> DatumModel:
    """Datum x in [-50, 50], y in [-30, 30] (z = 0 skin 0, z = -2 skin 1): bends on +-y, free ends at +-x."""
    poly = np.array([[-50.0, -30.0], [50.0, -30.0], [50.0, 30.0], [-50.0, 30.0]])
    free = np.array([False, True, False, True])
    return DatumModel(p0=np.zeros(3), n0=np.array([0.0, 0.0, 1.0]), f1=np.array([1.0, 0.0, 0.0]),
                      f2=np.array([0.0, 1.0, 0.0]), t=T_SHEET, polygon=poly, free=free, holes=[], area=6000.0,
                      bbox=(-50.0, -30.0, 50.0, 30.0), hole_blocked=[])


def _edge(edge_id: str, pts, usable: bool = True):
    return SimpleNamespace(edge_id=edge_id, polyline_mm=[tuple(map(float, p)) for p in pts], registration_usable=usable)


def _tab_panel(y0: float, y1: float, x_skin0: float, z0: float, z1: float, panel_id: str, *, edges=("lo", "hi", "top")):
    """A flange standing in the plane x = x_skin0 (skin 0, outward +x), y in [y0, y1], z in [z0, z1]."""
    outline = [(x_skin0, y0, z0), (x_skin0, y1, z0), (x_skin0, y1, z1), (x_skin0, y0, z1)]
    fe = []
    if "lo" in edges:
        fe.append(_edge(f"{panel_id}.E0", [(x_skin0, y0, z0), (x_skin0, y0, z1)]))
    if "hi" in edges:
        fe.append(_edge(f"{panel_id}.E1", [(x_skin0, y1, z0), (x_skin0, y1, z1)]))
    if "top" in edges:
        fe.append(_edge(f"{panel_id}.E2", [(x_skin0, y0, z1), (x_skin0, y1, z1)]))
    model = SimpleNamespace(outer_boundary_mm=outline, holes_mm=[], free_edges=fe)
    return SimpleNamespace(panel_id=panel_id, model=model, n0=np.array([1.0, 0.0, 0.0]))


class _Part(SimpleNamespace):
    def path_bends(self, panel_index: int) -> list[int]:
        return list(self.paths.get(panel_index, []))


def _part(panels, bends, paths) -> _Part:
    return _Part(datum=0, rigid_followers=set(), panels=panels, bends=bends, paths=paths, t=T_SHEET)


def _slab(y0, y1, z0, z1, x_skin0, step=0.25, shift_y=0.0, walls=True):
    """Scan points of a standing flange: both skins and its two cut ends (walls), shifted along y."""
    ys = np.arange(y0, y1 + 1e-9, step)
    zs = np.arange(z0, z1 + 1e-9, step)
    yy, zz = np.meshgrid(ys, zs)
    skins = [np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()]) for x in (x_skin0, x_skin0 - T_SHEET)]
    out = skins
    if walls:
        xs = np.arange(x_skin0 - T_SHEET + 0.25, x_skin0 - 0.2, 0.25)
        for y in (y0, y1):
            xx, zw = np.meshgrid(xs, zs)
            out.append(np.column_stack([xx.ravel(), np.full(xx.size, y), zw.ravel()]))
    pts = np.vstack(out)
    pts[:, 1] += shift_y
    return pts


def _fitted(x_skin0: float):
    return SimpleNamespace(state="FITTED", A=np.eye(4), n0=np.array([1.0, 0.0, 0.0]), d0=x_skin0, t=T_SHEET)


def _free_ty_fit(info_ty: float = 0.0) -> InPlaneFit:
    info = np.diag([60.0, 40.0, info_ty])
    cov = np.diag([1e-4, 4e-4, 0.0])
    return InPlaneFit(T=np.eye(4), information=info, length_scale=max(10.0, 0.5 * math.sqrt(6000.0)), edge_bins=10,
                      edge_bins_expected=10, holes_seen=0, holes_total=0, edge_rms=0.02, hole_rms=float("nan"),
                      degenerate=True, translation_cov=cov[1:, 1:], rotation_sigma_rad=0.0002, skin_heights={0: 0.0},
                      datum_points=1000, datum_coverage=1.0, hole_mismatch=0, pose_cov=cov,
                      undetermined=np.array([[0.0, 0.0, 1.0]]), erosion_mm=0.1)


def _scene(shift_y: float, *, second_tab=True, tab_edges=("lo", "hi", "top"), walls=True):
    """Datum + a tab bent up about an axis along y on the +x end (y -10 .. 10) and, optionally, a second
    one on the -x end (y -15 .. 5): their cut ends at y = const observe the datum's free ty."""
    cfg = InspectConfig()
    model = _datum_model()
    panels = [SimpleNamespace(panel_id="P0", model=SimpleNamespace(outer_boundary_mm=[], holes_mm=[], free_edges=[]),
                              n0=np.array([0.0, 0.0, 1.0])),
              _tab_panel(-10.0, 10.0, 52.0, 3.0, 23.0, "P1", edges=tab_edges)]
    bends = [SimpleNamespace(axis_dir=np.array([0.0, 1.0, 0.0]))]
    paths = {1: [0]}
    pts = [_slab(-10.0, 10.0, 3.0, 23.0, 52.0, shift_y=shift_y, walls=walls)]
    fits = [SimpleNamespace(state="DATUM", A=np.eye(4), n0=np.array([0.0, 0.0, 1.0]), d0=0.0, t=T_SHEET), _fitted(52.0)]
    if second_tab:
        # the -x tab stands in x = -52 (skin 0 outward -x is modelled as a flange whose skin 0 faces +x at -50)
        panels.append(_tab_panel(-15.0, 5.0, -50.0, 3.0, 18.0, "P2"))
        bends.append(SimpleNamespace(axis_dir=np.array([0.0, -1.0, 0.0])))
        paths[2] = [1]
        pts.append(_slab(-15.0, 5.0, 3.0, 18.0, -50.0, shift_y=shift_y, walls=walls))
        fits.append(_fitted(-50.0))
    part = _part(panels, bends, paths)
    points = np.vstack(pts)
    owner = np.full(len(points), -1, dtype=np.int64)
    return cfg, model, part, fits, points, owner


def test_flange_ends_observe_the_free_datum_direction_and_fix_it():
    cfg, model, part, fits, points, owner = _scene(0.4)
    rows = child_edge_rows(points, owner, part, model, fits, cfg, 0.25)
    # four cut ends (y = const, normals along the bend axes); the tops (normal along the datum normal) are not used
    assert sorted(rows.labels) == ["P1/P1.E0#0", "P1/P1.E1#0", "P2/P2.E0#0", "P2/P2.E1#0"]
    assert np.all(rows.walled) and np.all(np.abs(rows.rows[:, 1]) < 1e-9)  # observe ty (and rotation), not tx
    obs = -rows.rhs * np.sign(rows.rows[:, 2])  # offset along +y
    assert np.allclose(obs, 0.4, atol=0.13)  # every end sits 0.4 mm along +y of where the frame puts it
    fit, info = resolve_free_directions(_free_ty_fit(), np.eye(4), rows, model, cfg)
    assert fit is not None, info
    assert fit.T[1, 3] == pytest.approx(-0.4, abs=0.03)  # the frame moves the scan back onto the CAD
    assert fit.T[0, 3] == pytest.approx(0.0, abs=1e-9) and np.allclose(fit.T[:3, :3], np.eye(3), atol=1e-6)
    assert not fit.degenerate and fit.undetermined.size == 0
    assert fit.position_determined(model, np.array([0.0, 30.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert 0.0 < fit.position_sigma_mm(model, np.array([0.0, 30.0, 0.0]), np.array([0.0, 1.0, 0.0])) < 0.06
    assert info["edges_used"] and len(info["edges_used"]) == 4 and info["remaining"] == 0


def test_only_the_free_direction_moves_whatever_the_ends_say_about_the_others():
    cfg, model, part, fits, points, owner = _scene(0.4)
    rows = child_edge_rows(points, owner, part, model, fits, cfg, 0.25)
    # pretend the ends also disagree with the datum along tx: the datum's own tx must stay
    tampered = ChildEdgeRows(np.column_stack([rows.rows[:, 0], np.full(len(rows.rhs), 0.3), rows.rows[:, 2]]),
                             rows.rhs + 0.2, rows.weights, rows.walled, rows.edge, rows.labels, rows.spread)
    fit, info = resolve_free_directions(_free_ty_fit(), np.eye(4), tampered, model, cfg)
    assert fit is not None, info
    assert fit.T[0, 3] == pytest.approx(0.0, abs=1e-9) and abs(fit.T[1, 3]) > 0.1
    assert np.allclose(fit.T[:3, :3], np.eye(3), atol=1e-9)


def test_one_cut_end_is_not_enough_and_a_determined_pose_is_left_alone():
    cfg, model, part, fits, points, owner = _scene(0.4, second_tab=False, tab_edges=("lo", "top"))
    rows = child_edge_rows(points, owner, part, model, fits, cfg, 0.25)
    assert rows.labels == ["P1/P1.E0#0"]
    fit, info = resolve_free_directions(_free_ty_fit(), np.eye(4), rows, model, cfg)
    assert fit is None and info["reason"].startswith("TOO_FEW_CONSISTENT_EDGES")
    determined = _free_ty_fit(info_ty=40.0)
    determined.undetermined = np.zeros((0, 3))
    determined.degenerate = False
    cfg2, model2, part2, fits2, points2, owner2 = _scene(0.4)
    rows2 = child_edge_rows(points2, owner2, part2, model2, fits2, cfg2, 0.25)
    fit2, info2 = resolve_free_directions(determined, np.eye(4), rows2, model2, cfg2)
    assert fit2 is None and info2["reason"] == "NO_FREE_DIRECTION"


def test_a_cut_end_that_disagrees_is_dropped_and_two_that_disagree_are_refused():
    cfg, model, part, fits, points, owner = _scene(0.4)
    # a neighbour's material 1.5 mm beyond P2's upper end (y = 5): that end reads 1.5 mm long
    extra = _slab(5.0, 6.5, 3.0, 18.0, -50.0, walls=False)
    extra[:, 1] += 0.4
    pts = np.vstack([points, extra])
    own = np.full(len(pts), -1, dtype=np.int64)
    rows = child_edge_rows(pts, own, part, model, fits, cfg, 0.25)
    fit, info = resolve_free_directions(_free_ty_fit(), np.eye(4), rows, model, cfg)
    assert fit is not None, info
    assert any(r.startswith("P2/P2.E1#0") for r in info["rejected"])
    assert fit.T[1, 3] == pytest.approx(-0.4, abs=0.05)
    # the same material owned by another articulated panel is never read as this flange's end
    own2 = np.r_[np.full(len(points), -1), np.full(len(extra), 7)].astype(np.int64)
    rows2 = child_edge_rows(pts, own2, part, model, fits, cfg, 0.25)
    fit2, info2 = resolve_free_directions(_free_ty_fit(), np.eye(4), rows2, model, cfg)
    assert fit2 is not None and not info2["rejected"]
    # only two ends, and they disagree by 2 mm: nothing is trusted
    cfg3, model3, part3, fits3, points3, owner3 = _scene(0.4, second_tab=False)
    bad = _slab(10.0, 12.0, 3.0, 23.0, 52.0, walls=False)
    bad[:, 1] += 0.4
    p3 = np.vstack([points3, bad])
    rows3 = child_edge_rows(p3, np.full(len(p3), -1, dtype=np.int64), part3, model3, fits3, cfg3, 0.25)
    fit3, info3 = resolve_free_directions(_free_ty_fit(), np.eye(4), rows3, model3, cfg3)
    assert fit3 is None and "TOO_FEW" in info3["reason"]


def test_model_material_just_beyond_a_cut_end_blocks_it():
    cfg, model, part, fits, points, owner = _scene(0.0)
    # a flat tab sits in its window: the window's side (model material) 0.5 mm beyond P1's ends
    yy, zz = np.meshgrid(np.arange(10.5, 14.0, 0.5), np.arange(3.0, 23.0, 0.5))
    side = np.column_stack([np.full(yy.size, 51.0), yy.ravel(), zz.ravel()])
    blockers = np.vstack([side, side * np.array([1.0, -1.0, 1.0])])
    rows = child_edge_rows(points, owner, part, model, fits, cfg, 0.25, blockers=blockers)
    assert not any(lbl.startswith("P1/") for lbl in rows.labels)
    assert any(lbl.startswith("P2/") for lbl in rows.labels)
    # material 2.5 mm beyond only shortens the band: the end is still read, and correctly
    far = blockers.copy()
    far[:, 1] += np.sign(far[:, 1]) * 2.0
    rows2 = child_edge_rows(points, owner, part, model, fits, cfg, 0.25, blockers=far)
    assert sum(lbl.startswith("P1/") for lbl in rows2.labels) == 2
    assert np.all(np.abs(rows2.rhs) < 0.15)


def test_ends_of_a_flange_behind_a_crossing_bend_or_off_the_axis_are_not_used():
    cfg, model, part, fits, points, owner = _scene(0.4)
    # P1 hangs on a path whose second bend crosses the first: it can move along y
    part.bends.append(SimpleNamespace(axis_dir=np.array([1.0, 0.0, 0.0])))
    part.paths[1] = [0, 2]
    rows = child_edge_rows(points, owner, part, model, fits, cfg, 0.25)
    assert not any(lbl.startswith("P1/") for lbl in rows.labels)
    # an edge 10 deg off the axis normal is not a flange end for this purpose (only within 5 deg)
    cfg2, model2, part2, fits2, points2, owner2 = _scene(0.4, second_tab=False)
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    tip = (52.0, -10.0 + 20.0 * s, 3.0 + 20.0 * c)
    pm = part2.panels[1].model
    pm.outer_boundary_mm = [(52.0, -10.0, 3.0), (52.0, 10.0, 3.0), (52.0, 10.0, 23.0), tip]
    pm.free_edges[0].polyline_mm = [(52.0, -10.0, 3.0), tip]
    rows2 = child_edge_rows(points2, owner2, part2, model2, fits2, cfg2, 0.25)
    assert "P1/P1.E0#0" not in rows2.labels
    wide = load_config(None, lateral_axis_tol_deg=15.0)  # the same edge is read once the tolerance admits it
    assert "P1/P1.E0#0" in child_edge_rows(points2, owner2, part2, model2, fits2, wide, 0.25).labels


def test_config_carries_the_flange_end_settings_and_checks_them():
    cfg = InspectConfig()
    assert cfg.lateral_min_edges >= 2 and 0 < cfg.lateral_axis_tol_deg <= 45
    with pytest.raises(Exception):
        load_config(None, lateral_min_edges=0)
    with pytest.raises(Exception):
        load_config(None, lateral_axis_tol_deg=50.0)
    assert load_config(None, lateral_outlier_mm=0.4).sha256() != cfg.sha256()


# --------------------------------------------------------------------------- hinge search: joggles
def _frame(rho_iml=-0.1, delta=-1.5, fold=None):
    from domains.production.engine.inspect.articulated import HingeFrame

    return HingeFrame(o=np.zeros(3), a=np.array([0.0, 0.0, 1.0]), u=np.array([1.0, 0.0, 0.0]),
                      v=np.array([0.0, 1.0, 0.0]), z0=0.0, z1=100.0, rho_iml=rho_iml, delta=delta, band=8.1, fold=fold)


def test_the_line_window_is_the_parents_plane_when_flat_and_widens_with_the_angle():
    from domains.production.engine.inspect.articulated import rho_window

    cfg = InspectConfig()
    fr = _frame()
    c0, h0 = rho_window(fr, 0.0, cfg.hinge_rho_slack_mm, cfg)
    assert c0 == pytest.approx(-0.1) and h0 == pytest.approx(cfg.hinge_rho_slack_flat_mm)
    _, h35 = rho_window(fr, math.radians(35.0), cfg.hinge_rho_slack_mm, cfg)
    assert h35 == pytest.approx(min(cfg.hinge_rho_slack_mm, cfg.hinge_rho_slack_flat_mm
                                    + cfg.hinge_rho_slack_mm * math.sin(math.radians(35.0))))
    assert rho_window(fr, math.radians(90.0), cfg.hinge_rho_slack_mm, cfg)[1] == pytest.approx(cfg.hinge_rho_slack_mm)
    # a bend line placed d off moves the child's line by d sin(theta): at flat by nothing
    assert cfg.hinge_rho_slack_flat_mm < 0.5 * 1.5  # a sheet one thickness off is outside the flat window


def _joggle_section(offset=1.7, riser=2.6, t=1.5, step=0.05, width=60.0, z_step=1.0):
    """Cross-section points (u, v) of a parent (u < 0), a 35 deg riser and a tail parallel to the parent
    ``offset`` higher, both skins, swept along z."""
    th = math.radians(35.0)
    prof = []
    for u in np.arange(-15.0, 0.0, step):  # parent: inner skin v = -0.1, outer v = -1.6
        prof += [(u, -0.1), (u, -0.1 - t)]
    for s in np.arange(0.0, riser, step):  # riser
        prof += [(s * math.cos(th), -0.1 + s * math.sin(th)), (s * math.cos(th) + t * math.sin(th), -0.1 - t * math.cos(th) + s * math.sin(th))]
    u_end = riser * math.cos(th)
    for u in np.arange(u_end + 0.3, u_end + 14.0, step):  # tail
        prof += [(u, -0.1 + offset), (u, -0.1 - t + offset)]
    prof = np.asarray(prof)
    zs = np.arange(0.0, width, z_step)
    pts = np.column_stack([np.repeat(prof[:, 0], len(zs)), np.repeat(prof[:, 1], len(zs)), np.tile(zs, len(prof))])
    return pts


def test_hinge_continuity_rejects_the_far_flange_of_a_joggle_as_a_flat_continuation():
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig(hinge_rho_slack_flat_mm=0.8)  # wide enough that the tail's lower skin is in the window
    pts = _joggle_section(offset=0.3)
    fr = _frame()
    z, uc, vc = fr.coords(pts)
    keep = np.hypot(uc, vc) < fr.band
    z, uc, vc = z[keep], uc[keep], vc[keep]
    thetas = np.radians(np.arange(-10.0, 60.0 + 1e-9, 1.0))
    plain = _hough(z, uc, vc, fr, thetas, cfg, fr.band)[0]
    cont = _hough(z, uc, vc, fr, thetas, cfg, fr.band, continuity=True)[0]
    near_flat = np.abs(np.degrees(thetas)) <= cfg.flat_continuity_max_deg
    assert abs(math.degrees(thetas[int(np.argmax(plain))])) < 10.0  # the tail wins without continuity
    assert np.all(cont[near_flat] < 0.5 * plain[near_flat].max())  # it starts millimetres from the hinge
    assert math.degrees(thetas[int(np.argmax(cont))]) >= 15.0  # the riser's (formed) angle; the station
    # planes measure it precisely afterwards (the line support of a 2.6 mm riser is coarse)
    # a genuinely flat continuation (unformed bend) is continuous from the hinge and still found
    flat = np.column_stack([np.repeat(np.arange(-15.0, 15.0, 0.05), 2), np.tile([-0.1, -1.6], 600)])
    fp = np.column_stack([np.repeat(flat[:, 0], 60), np.repeat(flat[:, 1], 60), np.tile(np.arange(0.0, 60.0, 1.0), len(flat))])
    z, uc, vc = fr.coords(fp)
    keep = np.hypot(uc, vc) < fr.band
    sc = _hough(z[keep], uc[keep], vc[keep], fr, thetas, cfg, fr.band, continuity=True)[0]
    assert abs(math.degrees(thetas[int(np.argmax(sc))])) <= 1.0


# --------------------------------------------------------------------------- end to end: a joggled flange (E2 -> E1 -> E3)
_JOGGLE = {"t": 1.5, "width": 150.0, "segments": [("PBASE", 80.0), ("PA", 21.4), ("PJ", 3.6), ("PT", 13.0)],
           "sweeps": [90.0, 35.0, 35.0], "radii": [0.78, 0.1, 0.1], "seq": [0, 1, 2]}


def _profile_case(root, name: str, scenario: str, plan: str, dirs: list[str]):
    """An E2 DEV case of a custom profile (a 90 deg bend, then a 3.6 mm riser between two 35 deg bends)."""
    import dataclasses

    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic import cases as C, families as F

    def build(family, rng, req=F.FeatureRequest()):
        segs = [F.ProfileSegment(k, ln, "flange", F._chiral_holes(rng, 3, 4.0) if i == 0 else ())
                for i, (k, ln) in enumerate(_JOGGLE["segments"])]
        return F.build_profile(family="Z_STEP", thickness=_JOGGLE["t"], width=_JOGGLE["width"], segments=segs,
                               sweeps=_JOGGLE["sweeps"], directions=dirs, radii=_JOGGLE["radii"],
                               sequence=_JOGGLE["seq"], groups=[], params={"test": name})

    orig_build, orig_min = C.build_family, F._min_straight
    C.build_family, F._min_straight = build, (lambda t: 2.0)  # a 2.6 mm straight riser, as on real joggles
    try:
        base = next(s for s in C.plan_cases("smoke", "DEV") if s.family == "Z_STEP")
        spec = dataclasses.replace(base, case_id=f"A1J-{name}", scenario=scenario, exact=scenario == "COMPLETE_NOMINAL",
                                   plan=plan, point_target=None)
        C.generate_case(spec, root)
    finally:
        C.build_family, F._min_straight = orig_build, orig_min
    d = root / spec.case_id
    recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
    assert compile_part(recipe, d / "part.step", d / "compiled", flat_path=d / "flat.dxf").status == "COMPILED"
    return d


def _inspect(d):
    import hashlib

    from domains.production.engine.inspect.engine import InspectOptions, inspect_scan
    from domains.production.engine.inspect.part import load_inputs

    scan = next(d.glob("scan.*"))
    inputs = load_inputs(d / "compiled", d / "recipe.json", scan, hashlib.sha256(scan.read_bytes()).hexdigest())
    return inspect_scan(inputs, d / "e3", InspectConfig(), InspectOptions(write_outputs=False)).result


@pytest.fixture(scope="module")
def joggles(tmp_path_factory):
    root = tmp_path_factory.mktemp("a1_joggles")
    return {"full": _profile_case(root, "full", "COMPLETE_NOMINAL", "FULL", ["UP", "UP", "DOWN"]),
            "top": _profile_case(root, "top", "COMPLETE_NOMINAL", "TOP_ONLY", ["UP", "UP", "DOWN"]),
            "bump": _profile_case(root, "bump", "COMPLETE_NOMINAL", "FULL", ["UP", "UP", "UP"])}


def test_a_joggle_scanned_all_round_decodes_formed_and_measures_close_to_truth(joggles):
    import json

    d = joggles["full"]
    res = _inspect(d)
    truth = json.loads((d / "truth.json").read_text())
    assert res.inspection_validity.state == "VALID"
    assert [b.state for b in res.bends] == ["FORMED", "FORMED", "FORMED"]  # the riser's two bends were read flat
    for b in res.bends:
        for ch in b.characteristics:
            assert ch.decision != "FAIL"
            tdev = (truth["characteristics"].get(ch.characteristic_id) or {}).get("true_deviation")
            if ch.state == "MEASURED" and tdev is not None and ch.kind in ("BEND_LINE_POSITION", "FLANGE_LENGTH"):
                assert abs(float(ch.deviation) - float(tdev)) <= max(0.15, float(ch.uncertainty)), ch.characteristic_id


def test_a_joggle_seen_from_one_side_is_never_failed_on_a_station_that_left_its_riser(joggles):
    res = _inspect(joggles["top"])
    assert res.inspection_validity.state == "VALID"
    decided = [(b.label, ch.kind, ch.decision) for b in res.bends for ch in b.characteristics]
    assert not any(dec == "FAIL" for _, _, dec in decided), decided  # a nominal part: nothing may FAIL
    assert all(b.state in ("FORMED", "UNKNOWN") for b in res.bends)


def test_two_short_steps_the_same_way_are_two_formed_bends_not_one_bend_onto_the_far_flange(joggles):
    """35 + 35 deg the same way, 3.6 mm apart: the riser's region must not grow onto the flange at 70 deg."""
    res = _inspect(joggles["bump"])
    assert [b.state for b in res.bends] == ["FORMED", "FORMED", "FORMED"]
    for b in res.bends:
        for ch in b.characteristics:
            assert ch.decision != "FAIL", (b.label, ch.kind, ch.deviation)
            if ch.kind == "INCLUDED_ANGLE":
                assert ch.state == "MEASURED" and abs(float(ch.deviation)) <= float(ch.uncertainty)


# --------------------------------------------------------------------------- end to end: flange ends fix a free datum direction
@pytest.fixture(scope="module")
def hat_free_datum(tmp_path_factory):
    """The E2 smoke HAT with its datum's free edges unusable for registration: only two holes remain."""
    import dataclasses
    import json
    import shutil

    from contracts.production.canonical import canonical_json_bytes
    from contracts.production.compiled_part import CompiledPartV1
    from contracts.production.part_recipe import PartRecipeV1
    from domains.production.engine.compile import compile_part
    from domains.production.engine.synthetic.cases import generate_case, plan_cases

    root = tmp_path_factory.mktemp("a1_hat")
    spec = next(s for s in plan_cases("smoke", "DEV") if s.family == "HAT")
    spec = dataclasses.replace(spec, case_id="A1J-hat", scenario="COMPLETE_NOMINAL", exact=True, plan="FULL")
    generate_case(spec, root)
    d = root / spec.case_id
    recipe = PartRecipeV1.model_validate_json((d / "recipe.json").read_bytes())
    assert compile_part(recipe, d / "part.step", d / "compiled", flat_path=d / "flat.dxf").status == "COMPILED"
    comp = d / "compiled_datum_edges_unusable"
    shutil.copytree(d / "compiled", comp)
    doc = json.loads((comp / "compiled_part.json").read_text())
    for fe in next(p for p in doc["panels"] if p["is_datum"])["free_edges"]:
        fe["registration_usable"] = False
    doc.pop("compiled_part_sha256")
    (comp / "compiled_part.json").write_bytes(canonical_json_bytes(CompiledPartV1.finalize(**doc)))
    return d, comp


def test_flange_ends_fix_what_two_datum_holes_leave_free_and_the_positions_stay_true(hat_free_datum):
    import hashlib
    import json

    from domains.production.engine.inspect.engine import InspectOptions, inspect_scan
    from domains.production.engine.inspect.part import load_inputs

    d, comp = hat_free_datum
    scan = next(d.glob("scan.*"))
    inputs = load_inputs(comp, d / "recipe.json", scan, hashlib.sha256(scan.read_bytes()).hexdigest())
    res = inspect_scan(inputs, d / "e3", InspectConfig(), InspectOptions(write_outputs=False)).result
    truth = json.loads((d / "truth.json").read_text())["characteristics"]
    assert res.inspection_validity.state == "VALID"
    assert not res.inspection_validity.registration.in_plane_degenerate
    assert any(w.startswith("IN_PLANE_POSE_FROM_FLANGE_ENDS") for w in res.warnings)
    for b in res.bends:
        pos = next(c for c in b.characteristics if c.kind == "BEND_LINE_POSITION")
        assert pos.state == "MEASURED" and b.display.measured_polyline_mm is not None
        tdev = truth[pos.characteristic_id]["true_deviation"]
        assert abs(float(pos.deviation) - float(tdev)) <= float(pos.uncertainty), (b.label, pos.deviation, tdev)
