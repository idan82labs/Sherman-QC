"""A1 iteration 6 unit tests of the per-scan engine (E3): a fold formed the wrong way is read about the mirrored
hinge (the CAD axis reflected across the parent's mid-plane), and its bend-line position is measured by mirror
symmetry.

Every rule here is geometric; no part, family or file is named. Pure numpy (no OCP, no E1/E2 geometry,
no physical data).
Run: PYT tests/test_prod_engine_inspect_a1_iter6.py -q
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config

# A 90 deg bend about the z axis through the origin: the parent lies at x < 0 between its inner skin y = -RI and
# its outer skin y = -(RI + T); the CAD fold turns the child toward +y (v = a x u with u = +x, a = +z). Skin 0 is
# the bend's outer (OML) skin: the parent's skin 0 is y = -(RI + T), outward normal -y. Its mid-plane is y = -1.5.
RI, T, K = 1.0, 1.0, 0.5


def _bend(sweep_deg=90.0):
    sweep = math.radians(sweep_deg)
    ba = sweep * (RI + K * T)
    loci = {"outer_tangent_child": (np.array([RI + T, 0.0, 0.0]), np.array([RI + T, 0.0, 50.0])),
            "inner_tangent_child": (np.array([RI, 0.0, 0.0]), np.array([RI, 0.0, 50.0]))}
    b = SimpleNamespace(model=SimpleNamespace(neutral_length_mm=ba), sweep=sweep, p_out=np.array([-1.0, 0.0, 0.0]),
                        axis_point=np.zeros(3), axis_dir=np.array([0.0, 0.0, 1.0]), parent=0, child=1, oml_skin=0,
                        ri=RI, t=T, z0=0.0, z1=50.0)
    b.locus = lambda name: loci[name]
    return b


def _part():
    return SimpleNamespace(t=T, panels=[SimpleNamespace(n0=np.array([0.0, -1.0, 0.0]),
                                                        p0=np.array([-10.0, -(RI + T), 0.0]))])


def _child_pts(x, ys):
    return np.array([[x, y, 10.0] for y in ys])


# --------------------------------------------------------------------------- 1. mirrored fold kinematics
def test_mirrored_fold_starts_from_the_same_flat_blank():
    from domains.production.engine.inspect.articulated import fold_transform, mirrored_bend

    b = _bend()
    mb = mirrored_bend(_part(), b)
    assert mb.axis_point == pytest.approx([0.0, -(2 * RI + T), 0.0])  # the CAD axis reflected across y = -1.5
    assert fold_transform(mb, 0.0) == pytest.approx(fold_transform(b, 0.0))
    assert mb.oml_skin == 1 - b.oml_skin and mb.child == b.child and mb.parent == b.parent


def test_mirrored_fold_at_the_sweep_is_the_cad_fold_reflected_with_its_skins_kept():
    """At the nominal sweep the wrong-way child is the CAD child reflected across the parent's mid-plane, but the
    material keeps its skins: the CAD inner skin (x = RI) becomes the fold's outer skin (x = RI + T), going down."""
    from domains.production.engine.inspect.articulated import fold_transform, mirrored_bend
    from domains.production.engine.inspect.geom import apply

    b = _bend()
    G = fold_transform(mirrored_bend(_part(), b), b.sweep)
    ys = np.arange(0.5, 12.0, 0.5)
    inner, outer, mid = (apply(G, _child_pts(x, ys)) for x in (RI, RI + T, RI + 0.5 * T))
    assert inner[:, 0] == pytest.approx(RI + T) and inner[:, 1] == pytest.approx(-(2 * RI + T) - ys)
    assert outer[:, 0] == pytest.approx(RI) and outer[:, 1] == pytest.approx(-(2 * RI + T) - ys)
    # the mid-plane as a set: the CAD child's mid-plane reflected across y = -1.5
    assert mid[:, 0] == pytest.approx(RI + 0.5 * T) and mid[:, 1] == pytest.approx(-3.0 - ys)
    # a proper motion (no mirror image of the outline) that keeps the axial coordinate
    assert np.linalg.det(G[:3, :3]) == pytest.approx(1.0)
    assert inner[:, 2] == pytest.approx(10.0)


def test_mirrored_hinge_frame_reads_the_cad_cross_section():
    """In the mirrored frame the wrong-way fold's inner line has the CAD fold's offset at every angle."""
    from domains.production.engine.inspect.articulated import PanelFit, hinge_frame, mirrored_hinge_frame

    b = _bend()
    parent = PanelFit(0, "DATUM", np.eye(4), n0=np.array([0.0, -1.0, 0.0]), d0=RI + T, t=T)
    cfg = InspectConfig()
    fr = hinge_frame(b, parent, cfg)
    fm, _ = mirrored_hinge_frame(_part(), b, parent, fr)
    assert fm.o == pytest.approx([0.0, -(2 * RI + T), 0.0])
    assert fm.a == pytest.approx(-fr.a) and fm.v == pytest.approx(-fr.v) and fm.u == pytest.approx(fr.u)
    assert (fm.z0, fm.z1) == pytest.approx((-fr.z1, -fr.z0))
    for th in (b.sweep, math.radians(30.0), math.radians(60.0)):
        assert fm.rho_iml_at(th) == pytest.approx(fr.rho_iml_at(th), abs=1e-9)


# --------------------------------------------------------------------------- 2. the two-skin search reads it
def _wrong_way_band(skins=("inner", "outer"), clutter=False):
    """Parent (x < 0, skins y = -1 / -2) and a child formed 90 deg the wrong way: its inner skin x = RI and outer
    x = RI + T, going down from y = -3 (the mirrored axis at (0, -3))."""
    prof = []
    for x in np.arange(-15.0, -0.5, 0.05):
        prof += [(x, -RI), (x, -RI - T)]
    ys = -(2 * RI + T) - np.arange(0.5, 14.0, 0.05)
    if "inner" in skins:
        prof += [(RI, y) for y in ys]
    if "outer" in skins:
        prof += [(RI + T, y) for y in ys]
    if clutter:
        prof += [(x, 0.3 * x) for x in np.arange(0.5, 6.0, 0.05)]
    prof = np.asarray(prof)
    zs = np.arange(0.0, 50.0, 1.0)
    return np.column_stack([np.repeat(prof[:, 0], len(zs)), np.repeat(prof[:, 1], len(zs)), np.tile(zs, len(prof))])


def _search(pts, frame, thetas_deg):
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    z, uc, vc = frame.coords(pts)
    keep = np.hypot(uc, vc) < 3 * frame.band
    return _hough(z[keep], uc[keep], vc[keep], frame, np.radians(np.asarray(thetas_deg, dtype=np.float64)), cfg,
                  3 * frame.band)


def test_a_wrong_way_flange_is_found_about_the_mirrored_hinge_and_not_about_the_cad_one():
    from domains.production.engine.inspect.articulated import PanelFit, hinge_frame, mirrored_hinge_frame

    b = _bend()
    parent = PanelFit(0, "DATUM", np.eye(4), n0=np.array([0.0, -1.0, 0.0]), d0=RI + T, t=T)
    cfg = InspectConfig()
    fr = hinge_frame(b, parent, cfg)
    fm, _ = mirrored_hinge_frame(_part(), b, parent, fr)
    pts = _wrong_way_band()
    thetas = np.arange(11.0, 150.0, 1.0)
    sc_m, rh_m, ni_m, no_m = _search(pts, fm, thetas)
    k = int(np.argmax(sc_m))
    assert thetas[k] == pytest.approx(90.0, abs=1.5)
    assert rh_m[k] == pytest.approx(fr.rho_iml_at(b.sweep), abs=0.15)  # the CAD fold's inner-line offset
    assert min(ni_m[k], no_m[k]) > 0.5 * max(ni_m[k], no_m[k])  # both skins
    # the CAD-direction search sees no flange of that support anywhere in its range
    sc_c, _, _, _ = _search(pts, fr, np.arange(-10.0, 116.0, 1.0))
    assert float(sc_m[k]) >= 1.5 * float(sc_c.max())


def test_a_sheet_is_two_parallel_skins_even_when_they_read_closer_than_its_thickness():
    """A thin lip's two skins often read closer than its thickness; one surface alone is no sheet."""
    from domains.production.engine.inspect.articulated import _sheet_skins

    cfg = InspectConfig()
    ys = np.arange(1.5, 12.0, 0.02)
    for sep in (0.6, 1.0, 1.4):  # in thicknesses (T = 1)
        uc = np.concatenate([np.full(len(ys), RI), np.full(len(ys), RI + sep * T)])
        vc = np.concatenate([ys, ys])
        vis, partner = _sheet_skins(uc, vc, math.radians(90.0), -RI, T, 20.0, cfg)
        assert partner > 0.8 * vis, sep
    vis, partner = _sheet_skins(np.full(len(ys), RI), ys, math.radians(90.0), -RI, T, 20.0, cfg)
    assert vis > 100 and partner == 0.0


# --------------------------------------------------------------------------- 3. position by mirror symmetry
def _reversed_measure(shift_x=0.0):
    """A BendMeasure of the wrong-way fold: its own outer sharp (x = RI + T + shift, y = -RI) at 5 stations."""
    from domains.production.engine.inspect.metrology import BendMeasure

    zs = np.linspace(5.0, 45.0, 5)
    rev = np.column_stack([np.full(5, RI + T + shift_x), np.full(5, -RI), zs])
    cad = np.column_stack([np.full(5, RI + shift_x), np.full(5, -(RI + T)), zs])  # where the CAD OML planes meet
    mid = -(RI + 0.5 * T)
    return BendMeasure(bend=0, ok=True, z=zs, width=8.0, sharp=cad, valid=np.ones(5, dtype=bool),
                       angles=np.full(5, 270.0), line_point=cad[0] * [1, 1, 0], line_dir=np.array([0.0, 0.0, 1.0]),
                       sweep_deg=90.0, w_parent=np.array([-1.0, 0.0, 0.0]), w_child=np.array([0.0, -1.0, 0.0]),
                       reversed=True, rev_sharp=rev, rev_line_point=rev[0] * [1, 1, 0],
                       rev_mirror=(np.array([0.0, 1.0, 0.0]), mid), visible=(5.0, 45.0))


def test_a_wrong_way_fold_on_its_nominal_bend_line_reads_its_nominal_position():
    from domains.production.engine.inspect.metrology import reversed_position_measurement

    cfg = InspectConfig()
    nominal = (np.array([RI + T, -(RI + T), 0.0]), np.array([RI + T, -(RI + T), 50.0]))  # the CAD OML sharp
    out_a = np.array([-1.0, 0.0, 0.0])  # side A: the parent, outward away from the bend
    m = reversed_position_measurement(_reversed_measure(), nominal, out_a, 0.05, cfg)
    assert m is not None and m.value == pytest.approx(0.0, abs=1e-9)
    assert m.method.startswith("STATION_REVERSED_FOLD_SHARP_MIRRORED")
    assert any("BEND_FORMED_WRONG_WAY" in n for n in m.notes)
    # a bend line placed 0.7 mm toward the child reads 0.7 mm off (side A's outward direction: -0.7)
    m2 = reversed_position_measurement(_reversed_measure(0.7), nominal, out_a, 0.05, cfg)
    assert m2.value == pytest.approx(-0.7, abs=1e-9)


def test_reversed_position_needs_the_fold_s_own_sharps():
    from dataclasses import replace

    from domains.production.engine.inspect.metrology import reversed_position_measurement

    cfg = InspectConfig()
    nominal = (np.array([RI + T, -(RI + T), 0.0]), np.array([RI + T, -(RI + T), 50.0]))
    bm = _reversed_measure()
    assert reversed_position_measurement(replace(bm, reversed=False), nominal, np.array([-1.0, 0, 0]), 0.05,
                                         cfg) is None
    few = bm.rev_sharp.copy()
    few[1:] = np.nan  # one resolved station
    assert reversed_position_measurement(replace(bm, rev_sharp=few), nominal, np.array([-1.0, 0, 0]), 0.05,
                                         cfg) is None


# --------------------------------------------------------------------------- 4. config
def test_reversed_search_config_is_range_checked(tmp_path):
    cfg = InspectConfig()
    assert cfg.reversed_support_ratio == pytest.approx(1.5) and cfg.reversed_two_skin_share == pytest.approx(0.2)
    for key, bad in (("reversed_support_ratio", 0.9), ("reversed_two_skin_share", 0.0)):
        p = tmp_path / f"{key}.json"
        p.write_text(f'{{"{key}": {bad}}}')
        with pytest.raises(ConfigError):
            load_config(p)
