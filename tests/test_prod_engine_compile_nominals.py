"""E1 compile: nominal conventions (OML flange length, included angle, backgauge) and recipe policy.

Expected flange lengths are analytic (testparts.ProfileInfo.segment_oml_lengths: OML sharp
to OML sharp, or OML sharp to the free end), independent of the compile's B-rep analysis.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from contracts.production.compiled_part import check_compiled_against_recipe
from domains.production.engine.compile import compile_part
from domains.production.engine.compile import testparts as tp
from domains.production.engine.compile.dev_recipe import dev_recipe
from domains.production.engine.compile.nominals import _sharp_crossing

K = 0.33


def _compile(tmp, shape, cands, sku, **kw):
    step = tp.write_step(shape, tmp / f"{sku}.step")
    recipe = dev_recipe(step, cands, sku=sku, **kw)
    return recipe, compile_part(recipe, step, tmp / f"out_{sku}").compiled


@pytest.fixture(scope="module")
def hat(tmp_path_factory):
    shape, info = tp.hat_part(width=40.0)
    tmp = tmp_path_factory.mktemp("hat")
    recipe, compiled = _compile(tmp, shape, tp.candidates_from_info(info), "SYN-HAT", inspection_stage="IN_PROCESS")
    return info, recipe, compiled


def _flanges(compiled, uid):
    return {c.side: c for c in compiled.characteristics if c.bend_uid == uid and c.kind == "FLANGE_LENGTH"}


def _profile_index(info, compiled_bend):
    sharp = np.array(compiled_bend.loci.oml_virtual_sharp.start_mm)
    return min(range(len(info.bends)), key=lambda i: float(np.linalg.norm(
        info.bends[i].oml_sharp[[0, 2]] - sharp[[0, 2]])))


def test_compiled_hat_is_consistent_with_its_recipe(hat):
    info, recipe, c = hat
    assert c.status == "COMPILED", c.blocker_codes
    assert check_compiled_against_recipe(c, recipe) == []
    assert {i.code for i in c.issues} == {"K_FACTOR_DEFAULTED", "FLAT_NOT_BOUND"}


def test_flange_lengths_equal_the_analytic_oml_lengths(hat):
    info, _, c = hat
    seg = info.segment_oml_lengths
    n = len(info.bends)
    for b in c.bends:
        i = _profile_index(info, b)  # profile bend i joins segments i (incoming) and i + 1 (outgoing)
        flanges = _flanges(c, b.bend_uid)
        for side in b.sides:
            outward = np.array(side.outward_dir)
            k = i + 1 if float(outward @ info.bends[i].flange_out_dir) > 0.99 else i
            assert float(outward @ (info.bends[i].flange_out_dir if k == i + 1 else -info.bends[i].flange_in_dir)) > 0.99
            f = flanges[side.side]
            assert f.nominal == pytest.approx(seg[k], abs=1e-6)
            free = k in (0, n)
            assert f.boundary_kind == ("FREE_EDGE" if free else "ADJACENT_BEND")
            assert side.boundary_kind == f.boundary_kind
            assert f.convention == ("OML_SHARP_TO_FREE_EDGE" if free else "OML_SHARP_TO_OML_SHARP")
            assert (f.adjacent_bend_uid is not None) == (f.boundary_kind == "ADJACENT_BEND")


def test_side_a_is_the_flange_with_the_larger_plus_x_outward(hat):
    _, _, c = hat
    for b in c.bends:
        a, bb = b.sides
        assert a.side == "A" and bb.side == "B"
        da, db = np.array(a.outward_dir), np.array(bb.outward_dir)
        for axis in range(3):
            if abs(da[axis] - db[axis]) >= 1e-6:
                assert da[axis] > db[axis]
                break


def test_angle_and_position_nominals(hat):
    _, _, c = hat
    for b in c.bends:
        angle = c.nominal(f"{b.bend_uid}/INCLUDED_ANGLE")
        assert angle.nominal == pytest.approx(90.0, abs=1e-6) and angle.convention == "INCLUDED_DEG"
        assert angle.nominal == pytest.approx(180.0 - b.sweep_deg, abs=1e-9)
        pos = c.nominal(f"{b.bend_uid}/BEND_LINE_POSITION")
        assert pos.nominal == 0.0 and pos.convention == "OML_SHARP_OFFSET_ALONG_SIDE_A"


def test_backgauge_nominals_follow_the_bend_deduction(hat):
    info, _, c = hat
    seg = info.segment_oml_lengths
    ro, ri, t = info.bends[0].ro, info.bends[0].ri, info.thickness
    bd = 2 * ro * math.tan(math.radians(45.0)) - math.radians(90.0) * (ri + K * t)
    bgs = sorted((x.stage_index, x) for x in c.characteristics if x.kind == "BACKGAUGE_LENGTH")
    assert [k for k, _ in bgs] == [1, 2, 3]
    # the recipe sequence is the profile order here: after bend k, the flat run goes to the far free edge
    order = [_profile_index(info, b) for b in c.bends]
    assert order == [0, 1, 2, 3]
    for k, x in bgs:
        expected = sum(seg[k:]) - (len(info.bends) - k) * bd
        assert x.nominal == pytest.approx(expected, abs=1e-4), k
        assert x.boundary_kind == "FREE_EDGE" and x.convention == "OML_SHARP_TO_FAR_BOUNDARY_AT_STAGE"
        assert x.bend_uid == c.bends[k - 1].bend_uid


def test_notch_does_not_move_a_free_edge_nominal(tmp_path):
    shape, info = tp.l_part(width=40.0)
    notched = tp.cut_box(shape, (-1.0, 15.0, -1.0), (4.0, 22.0, 5.0))  # 4 mm deep notch over 7 of 40 mm
    _, c = _compile(tmp_path, notched, tp.candidates_from_info(info), "SYN-LN")
    base = next(f for f in _flanges(c, c.bends[0].bend_uid).values() if f.boundary_kind == "FREE_EDGE"
                and abs(f.nominal - info.segment_oml_lengths[0]) < 1e-6)
    assert base.nominal == pytest.approx(info.segment_oml_lengths[0], abs=1e-6)
    detail = c.diagnostics["flange_nominals"]
    spread = [v["spread_mm"] for v in detail.values() if v["kind"] == "FREE_EDGE" and v["spread_mm"]]
    assert max(spread) == pytest.approx(4.0, abs=1e-6)  # the notch is seen, then ignored by the median


@pytest.mark.parametrize("sweep, ri", [(120.0, 2.0), (45.0, 2.0), (90.0, 8.0)])
def test_acute_shallow_and_large_radius_bends(tmp_path, sweep, ri):
    shape, info = tp.l_part(width=30.0, sweep=sweep, ri=ri)
    _, c = _compile(tmp_path, shape, tp.candidates_from_info(info), f"SYN-L{int(sweep)}-{int(ri)}")
    assert c.status == "COMPILED", c.blocker_codes
    (b,) = c.bends
    assert b.supported and b.inner_radius_mm == pytest.approx(ri, abs=1e-9)
    assert b.nominal_included_deg == pytest.approx(180.0 - sweep, abs=1e-6)
    values = sorted(f.nominal for f in _flanges(c, b.bend_uid).values())
    assert values == pytest.approx(sorted(info.segment_oml_lengths), abs=1e-5)
    assert b.neutral_length_mm == pytest.approx(math.radians(sweep) * (ri + K * info.thickness), abs=1e-6)


def test_tolerances_and_requirements_come_from_the_recipe(tmp_path):
    shape, info = tp.z_part(width=30.0)
    overrides = {1: {"angle_deg": {"plus": 0.25, "minus": 0.75},
                     "flange_inspection": {"side_a": "REQUIRED", "side_b": "NOT_INSPECTED"}}}
    tol = {"angle_deg": {"plus": 1.0, "minus": 1.0}, "flange_mm": {"plus": 0.4, "minus": 0.6},
           "surface_profile_mm": 0.5, "source": "DRAWING"}  # no position / backgauge band
    recipe, c = _compile(tmp_path, shape, tp.candidates_from_info(info), "SYN-ZT", bend_overrides=overrides,
                         tolerances=tol)
    assert check_compiled_against_recipe(c, recipe) == []
    b1, b2 = c.bends
    side_b = c.nominal(f"{b1.bend_uid}/FLANGE_LENGTH/B")
    assert side_b.recipe_requirement == "NOT_INSPECTED"
    assert (side_b.tolerance.plus, side_b.tolerance.minus) == (0.4, 0.6)  # carried although not inspected
    angle = c.nominal(f"{b1.bend_uid}/INCLUDED_ANGLE")
    assert (angle.tolerance.plus, angle.tolerance.minus, angle.tolerance.origin) == (0.25, 0.75, "RECIPE_BEND_OVERRIDE")
    assert c.nominal(f"{b2.bend_uid}/INCLUDED_ANGLE").tolerance.origin == "RECIPE_DEFAULT"
    pos = c.nominal(f"{b1.bend_uid}/BEND_LINE_POSITION")
    assert pos.tolerance is None and pos.recipe_requirement == "OPTIONAL"
    bg = next(x for x in c.characteristics if x.kind == "BACKGAUGE_LENGTH")
    assert bg.recipe_requirement == "REQUIRED" and (bg.tolerance.plus, bg.tolerance.minus) == (0.4, 0.6)
    assert all(x.tolerance is None or x.tolerance.standard == "DRAWING" for x in c.characteristics)


def test_angle_convention_disagreement_is_a_named_warning(tmp_path):
    shape, info = tp.l_part(width=30.0)
    cands = tp.candidates_from_info(info)
    bad = [dict(cands[0], target_angle=88.0)]  # recipe says 92 deg included, CAD is 90
    _, c = _compile(tmp_path, shape, bad, "SYN-LA")
    warn = [i for i in c.issues if i.code == "ANGLE_CONVENTION_DISAGREES_WITH_RECIPE"]
    assert c.status == "COMPILED" and len(warn) == 1 and warn[0].bend_uid == c.bends[0].bend_uid
    assert c.nominal(f"{c.bends[0].bend_uid}/INCLUDED_ANGLE").nominal == pytest.approx(90.0, abs=1e-6)
    ok = [dict(cands[0], target_angle=90.3)]  # within 0.5 deg: no warning
    _, c2 = _compile(tmp_path, shape, ok, "SYN-LB")
    assert not [i for i in c2.issues if i.code == "ANGLE_CONVENTION_DISAGREES_WITH_RECIPE"]


def test_general_measurement_is_a_toleranced_characteristic(tmp_path):
    shape, info = tp.l_part(width=30.0)
    gms = [{"id": "gm-width", "label": "Width", "point_start": [5.0, 0.0, 0.0], "point_end": [5.0, 30.0, 0.0]}]
    recipe, c = _compile(tmp_path, shape, tp.candidates_from_info(info), "SYN-LG", general_measurements=gms)
    gm = c.nominal("gm/gm-width")
    assert gm.nominal == pytest.approx(30.0) and gm.convention == "CAD_STRAIGHT_LINE"
    assert gm.recipe_requirement == "REQUIRED" and gm.tolerance.plus == 0.8
    assert check_compiled_against_recipe(c, recipe) == []


def test_sharp_crossing_handles_parallel_and_angled_far_bends():
    o, a, w = np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    parallel = (np.array([30.0, 0.0, 0.0]), np.array([30.0, 50.0, 0.0]))
    assert _sharp_crossing(o, a, w, 10.0, parallel) == pytest.approx(30.0)
    assert _sharp_crossing(o, a, w, 40.0, parallel) == pytest.approx(30.0)
    angled = (np.array([30.0, 0.0, 0.0]), np.array([40.0, 50.0, 0.0]))  # far bend at ~11 deg
    assert _sharp_crossing(o, a, w, 25.0, angled) == pytest.approx(35.0)
    along = (np.array([30.0, 0.0, 0.0]), np.array([80.0, 0.0, 0.0]))  # runs along the flange direction
    assert _sharp_crossing(o, a, w, 25.0, along) is None


def test_machine_moves_are_carried_and_the_part_stays_consistent(tmp_path):
    shape, info = tp.hat_part(width=30.0)
    recipe, c = _compile(tmp_path, shape, tp.candidates_from_info(info), "SYN-HMOVE", machine_moves=[[3, 4]],
                         inspection_stage="IN_PROCESS")
    assert [b.machine_move_index for b in c.bends] == [None, None, 1, 1]
    assert check_compiled_against_recipe(c, recipe) == []


def test_two_direction_part_nominals(tmp_path):
    # base 60 x 40 (t 2, Ri 2) with up flanges along +X (15 high) and +Y (12 high)
    shape = tp.two_direction_part(a=60.0, b=40.0, h1=15.0, h2=12.0)
    step = tp.write_step(shape, tmp_path / "two.step")
    from domains.production.engine.compile.api import load_part_model

    m = load_part_model(step)
    cands = []
    for b in sorted(m.bends.values(), key=lambda b: b.graph_bend_id):
        mid = (b.r_parent + b.r_child) / np.linalg.norm(b.r_parent + b.r_child)
        p = b.axis_point + (b.ri + b.thickness / 2) * mid
        cands.append({"line_start": (p + b.z0 * b.axis_dir).tolist(), "line_end": (p + b.z1 * b.axis_dir).tolist(),
                      "target_angle": b.sweep_deg, "target_radius": b.ri})
    recipe = dev_recipe(step, cands, sku="SYN-TWO")
    c = compile_part(recipe, step, tmp_path / "out_two").compiled
    assert c.status == "COMPILED" and check_compiled_against_recipe(c, recipe) == []
    lengths = {}
    for b in c.bends:
        fl = sorted(f.nominal for f in _flanges(c, b.bend_uid).values())
        lengths[round(abs(b.axis.direction[0]))] = fl
    # bend along Y (+X edge): base OML 60 + Ri + t = 64, flange t + 15 = 17; along X: 44 and 14
    assert lengths[0] == pytest.approx([17.0, 64.0], abs=1e-6)
    assert lengths[1] == pytest.approx([14.0, 44.0], abs=1e-6)
    bg = next(x for x in c.characteristics if x.kind == "BACKGAUGE_LENGTH")
    assert bg.boundary_kind == "FREE_EDGE" and bg.nominal in (pytest.approx(64.0, abs=1e-6), pytest.approx(44.0, abs=1e-6))


def test_windowed_base_panel_is_one_keyhole_polygon(tmp_path):
    shape, info = tp.window_u_part()
    recipe, c = _compile(tmp_path, shape, tp.candidates_from_info(info), "SYN-WIN")
    assert c.status == "COMPILED" and c.diagnostics["s0_flatness_max_mm"] < 1e-6
    base = next(p for p in c.panels if p.is_datum)
    pts = np.array(base.outer_boundary_mm)
    y = pts[:, 1]
    assert y.min() == pytest.approx(0.0, abs=1e-6) and y.max() == pytest.approx(80.0, abs=1e-6)
    assert len(base.free_edges) >= 2 and not base.holes_mm  # the window is not a hole: it splits the panel
    for fe in base.free_edges:
        assert fe.registration_usable


def test_free_edge_levels_dominant_continuous_and_ambiguous():
    from domains.production.engine.compile.nominals import _dominant_level

    stepped_70 = np.array([168.0] * 15 + [158.03] * 6)
    value, detail = _dominant_level(stepped_70)
    assert value == pytest.approx(168.0) and "71%" in detail
    stepped_50 = np.array([168.0] * 11 + [158.03] * 10)
    assert _dominant_level(stepped_50)[0] is None  # flips with station placement: no nominal
    taper = np.linspace(40.0, 50.0, 21)  # a slanted free edge: every station differs
    assert _dominant_level(taper)[0] == pytest.approx(45.0)


def test_half_corridor_notch_gives_no_flange_nominal(tmp_path):
    shape, info = tp.l_part(width=40.0)
    notched = tp.cut_box(shape, (-1.0, -1.0, -1.0), (6.0, 20.5, 5.0))  # 6 mm deep over ~half the 40 mm corridor
    _, c = _compile(tmp_path, notched, tp.candidates_from_info(info), "SYN-LHALF")
    assert c.status == "COMPILED"
    base = next(f for f in _flanges(c, c.bends[0].bend_uid).values()
                if f.nominal is None or abs(f.nominal - info.segment_oml_lengths[1]) > 1e-6)
    assert base.nominal is None and base.unavailable_reason == "NOMINAL_UNAVAILABLE"
    assert base.tolerance is not None and base.recipe_requirement == "REQUIRED"
    warn = [i for i in c.issues if i.code == "NOMINAL_UNAVAILABLE"]
    assert warn and "stepped" in warn[0].detail


def test_corner_rounds_at_the_corridor_ends_do_not_move_the_free_edge_nominal(tmp_path):
    shape, info = tp.l_part(width=30.0)
    rounded = tp.round_vertical_edges(shape, [(0.0, 0.0), (0.0, 30.0)], 9.0)  # rounds cover 60 % of the corridor
    _, c = _compile(tmp_path, rounded, tp.candidates_from_info(info), "SYN-LROUND")
    assert c.status == "COMPILED" and not [f for f in c.unsupported_features]
    base_len = info.segment_oml_lengths[0]
    values = sorted(f.nominal for f in _flanges(c, c.bends[0].bend_uid).values())
    assert values == pytest.approx(sorted([base_len, info.segment_oml_lengths[1]]), abs=1e-6)
    detail = next(v for v in c.diagnostics["flange_nominals"].values() if v["value"] == pytest.approx(base_len))
    assert detail["spread_mm"] > 1.0 and "parallel to the bend" in detail["detail"]


def test_backgauge_nominals_record_their_k_sensitivity(hat):
    # every unformed strip crossed adds sweep * t per unit K: pi * 2 mm for these 90 deg, t = 2 bends
    info, _, c = hat
    diag = c.diagnostics["backgauge_nominals"]
    uids = [b.bend_uid for b in c.bends]
    n = len(uids)
    for k in range(1, n):
        d = diag[f"S{k}"]
        assert d["crossed_bend_uids"] == uids[k:]
        assert d["d_nominal_d_k_mm"] == pytest.approx((n - k) * math.pi / 2 * info.thickness, abs=1e-4)
        assert d["k_sigma_mm"] == pytest.approx(abs(d["d_nominal_d_k_mm"]) * c.k_factor.uncertainty, abs=1e-9)


def test_recipe_thickness_that_disagrees_with_the_cad_blocks(tmp_path):
    shape, info = tp.l_part(width=30.0)
    _, wrong = _compile(tmp_path, shape, tp.candidates_from_info(info), "SYN-LT3", thickness_mm=3.0)
    assert wrong.status == "BLOCKED" and wrong.blocker_codes == ("THICKNESS_INCONSISTENT",)
    issue = next(i for i in wrong.issues if i.code == "THICKNESS_INCONSISTENT")
    assert "recipe thickness 3 mm" in issue.detail and "2.0000" in issue.detail
    _, close = _compile(tmp_path, shape, tp.candidates_from_info(info), "SYN-LT2", thickness_mm=2.03)
    assert close.status == "COMPILED", close.blocker_codes  # within 0.05 mm + 2 %
