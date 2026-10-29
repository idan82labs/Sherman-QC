"""A1 iteration 5 unit tests of the per-scan engine (E3): a formed flange whose hinge frame sits a few
millimetres off, read on both of its skins; a bend formed the wrong way never reads its nominal angle.

Every rule here is geometric; no part, family or file is named. Pure numpy (no OCP, no E1/E2 geometry,
no physical data).
Run: PYT tests/test_prod_engine_inspect_a1_iter5.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config


# --------------------------------------------------------------------------- 1. a flange beyond its window
def _frame(rho_iml=-0.8, delta=-1.5, band=8.8):
    """Hinge frame at the origin: u the parent's flat continuation, v the side the child rotates to.

    At 90 deg the child's line normal is -u, so the nominal inner line ``rho_iml`` = -0.8 sits at u = 0.8
    and the outer line (``rho_iml + delta``) at u = 2.3.
    """
    from domains.production.engine.inspect.articulated import HingeFrame

    return HingeFrame(o=np.zeros(3), a=np.array([0.0, 0.0, 1.0]), u=np.array([1.0, 0.0, 0.0]),
                      v=np.array([0.0, 1.0, 0.0]), z0=0.0, z1=60.0, rho_iml=rho_iml, delta=delta, band=band)


def _sweep(prof, width=60.0, z_step=1.0):
    prof = np.asarray(prof)
    zs = np.arange(0.0, width, z_step)
    return np.column_stack([np.repeat(prof[:, 0], len(zs)), np.repeat(prof[:, 1], len(zs)), np.tile(zs, len(prof))])


def _flange_90(shift_u=0.0, skins=("iml", "oml"), t=1.5):
    """Parent (u < 0 side, skins v = -0.8 / -2.3) and a child formed 90 deg, its bend line moved ``shift_u``
    along u (negative: toward the parent): inner skin at u = 0.8 + shift_u, outer at 2.3 + shift_u."""
    prof = []
    for u in np.arange(-15.0, -1.0 + shift_u, 0.05):
        prof += [(u, -0.8), (u, -0.8 - t)]
    vs = np.arange(1.0, 14.0, 0.05)
    if "iml" in skins:
        prof += [(0.8 + shift_u, v) for v in vs]
    if "oml" in skins:
        prof += [(0.8 + t + shift_u, v) for v in vs]
    return _sweep(prof)


def _hough_at(pts, thetas_deg, cfg=None):
    from domains.production.engine.inspect.articulated import _hough

    cfg = cfg or InspectConfig()
    fr = _frame()
    z, uc, vc = fr.coords(pts)
    keep = np.hypot(uc, vc) < 3 * fr.band
    th = np.radians(np.asarray(thetas_deg, dtype=np.float64))
    return _hough(z[keep], uc[keep], vc[keep], fr, th, cfg, 3 * fr.band)


def test_a_flange_whose_far_skin_lies_past_the_window_is_read_on_both_skins():
    """The bend line sits 3.5 mm toward the parent (a frame placed off by an ancestor's moved line): the flange's
    outer skin falls inside the inner-line window, its inner skin outside. Read on both skins, not one."""
    sc, rh, ni, no = _hough_at(_flange_90(-3.5), [90.0])
    assert rh[0] == pytest.approx(-(0.8 - 3.5), abs=0.15)  # the real inner skin (u = -2.7), outside the window
    assert min(ni[0], no[0]) > 0.5 * max(ni[0], no[0])  # both skins carry it
    # the two-skin reading also wins the angle search against the diagonals through the sheet
    thetas = np.arange(30.0, 116.0, 1.0)
    sc, _, _, _ = _hough_at(_flange_90(-3.5), thetas)
    assert thetas[int(np.argmax(sc))] == pytest.approx(90.0, abs=1.5)


def test_one_visible_skin_past_the_window_keeps_the_nominal_reading():
    """Only the outer skin seen: nothing says which skin it is, so the reading inside the window stands."""
    _, rh, ni, no = _hough_at(_flange_90(-3.5, skins=("oml",)), [90.0])
    assert rh[0] == pytest.approx(-(2.3 - 3.5), abs=0.15)  # the visible skin read as the inner one (in the window)
    assert min(ni[0], no[0]) <= 0.05 * max(ni[0], no[0])


def test_a_flange_at_its_nominal_place_reads_as_before():
    for skins in (("iml", "oml"), ("oml",), ("iml",)):
        _, rh, _, _ = _hough_at(_flange_90(0.0, skins=skins), [90.0])
        assert rh[0] == pytest.approx(-0.8, abs=0.15), skins


def test_the_rereading_never_reaches_past_the_grandchild_limit_or_near_flat():
    from domains.production.engine.inspect.articulated import _hough

    cfg = InspectConfig()
    fr = _frame()
    th = np.radians([90.0])

    def run(shift_u, c=cfg, grand_u=None, thetas=th):
        z, uc, vc = fr.coords(_flange_90(shift_u))
        keep = np.hypot(uc, vc) < 3 * fr.band
        return _hough(z[keep], uc[keep], vc[keep], fr, thetas, c, 3 * fr.band, grand_u=grand_u)

    # moved 3.5 mm away from the parent: its inner skin is the window's outer-line reading, read on both skins
    _, rh, ni, no = run(3.5)
    assert rh[0] == pytest.approx(-(0.8 + 3.5), abs=0.15) and min(ni[0], no[0]) > 0
    # a parallel bend of the child 2 mm along u (that side's window stops at 1 mm): never read past it
    _, rh, _, _ = run(3.5, grand_u=2.0)
    assert abs(rh[0] + 4.3) > 1.0
    # on the side away from such a bend the limit does not apply
    _, rh, _, _ = run(-3.5, grand_u=2.0)
    assert rh[0] == pytest.approx(2.7, abs=0.15)
    # below the angle floor the one-skin window reading stands
    off = load_config(None, one_skin_reread_min_deg=90.0)
    _, rh2, ni2, no2 = run(-3.5, c=off, thetas=np.radians([89.0]))
    assert min(ni2[0], no2[0]) <= 0.05 * max(ni2[0], no2[0])


# --------------------------------------------------------------------------- 2. a formed reading astray
def test_the_wide_search_runs_for_flat_partial_and_astray_formed_readings_only():
    from domains.production.engine.inspect.articulated import _wide_search_needed

    cfg = InspectConfig()
    assert _wide_search_needed(2.0, 90.0, cfg) == (True, False)  # flat: the existing rule
    assert _wide_search_needed(60.0, 90.0, cfg) == (True, False)  # partial
    assert _wide_search_needed(93.0, 90.0, cfg) == (False, False)  # formed near nominal
    assert _wide_search_needed(80.0, 90.0, cfg) == (True, True)  # formed, 10 deg open
    assert _wide_search_needed(107.0, 90.0, cfg) == (True, True)  # formed, 17 deg closed
    assert _wide_search_needed(50.0, 35.0, cfg) == (False, False)  # a shallow bend never (its own rule)


def test_a_sheet_beyond_the_window_outweighs_a_crossing_reading_astray():
    """The window holds only a short two-skin run at 110 deg (clutter) and lines crossing the flange's skins
    obliquely; the formed 90 deg flange lies 6 mm toward the parent, past the window and one thickness more.
    The narrow reading is formed but astray, and the wide search finds the flange."""
    from domains.production.engine.inspect.articulated import _displaced_formed, _hough, _wide_search_needed

    cfg = InspectConfig()
    fr = _frame()
    th = math.radians(110.0)
    d, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    clutter = [tuple(s * d + rho * n) for s in np.arange(1.0, 4.0, 0.05) for rho in (-0.8, -2.3)]
    pts = np.vstack([_flange_90(-6.0), _sweep(clutter)])
    z, uc, vc = fr.coords(pts)
    keep = np.hypot(uc, vc) < 3 * fr.band
    z, uc, vc = z[keep], uc[keep], vc[keep]
    thetas = np.arange(60.0, 115.01, 1.0)
    sc, rh, _, _ = _hough(z, uc, vc, fr, np.radians(thetas), cfg, 3 * fr.band)
    k = int(np.argmax(sc))
    assert thetas[k] >= 100.0  # the narrow window's best reads formed, far from 90 deg
    assert _wide_search_needed(float(thetas[k]), 90.0, cfg) == (True, True)
    alt = _displaced_formed(z, uc, vc, fr, 90.0, cfg, 3 * fr.band)
    assert alt is not None
    assert math.degrees(alt[0]) == pytest.approx(90.0, abs=1.0)
    assert alt[1] == pytest.approx(-(0.8 - 6.0), abs=0.15)  # the flange's inner skin, 6 mm off nominal
    assert alt[2] >= cfg.displaced_support_ratio * sc[k]


def test_config_carries_the_iteration_5_settings_and_checks_them():
    cfg = InspectConfig()
    assert cfg.one_skin_reread_min_deg == 20.0 and cfg.one_skin_reread_min_share == 0.3
    assert cfg.displaced_check_dev_deg == 8.0
    with pytest.raises(ConfigError):
        load_config(None, one_skin_reread_min_share=0.0)
    with pytest.raises(ConfigError):
        load_config(None, one_skin_reread_min_deg=95.0)
    with pytest.raises(ConfigError):
        load_config(None, displaced_check_dev_deg=0.0)


# --------------------------------------------------------------------------- 3. a bend formed the wrong way
def _rot(axis, deg):
    from domains.production.engine.inspect.geom import rotation_matrix

    return rotation_matrix(np.asarray(axis, dtype=np.float64), math.radians(deg))


def test_the_sweep_is_signed_by_the_side_the_child_lies_on():
    """Hinge axis a = u x v: the CAD fold turns the child from u toward v. The parent's OML normal (outside of
    the fold, -v) turned by +theta about a is the child's OML normal at sweep theta; turned by -theta it is a child
    formed the wrong way (the other side of the parent's plane). The magnitude is the unsigned dihedral."""
    from domains.production.engine.inspect.geom import signed_sweep_deg

    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    a = np.cross(u, v)
    n_p = -v
    for deg in (5.0, 25.0, 90.0, 120.0, 149.0):
        n_c = _rot(a, deg) @ n_p
        assert signed_sweep_deg(n_p, n_c, a) == pytest.approx(math.degrees(math.acos(float(n_p @ n_c))), abs=1e-9)
        assert signed_sweep_deg(n_p, n_c, a) == pytest.approx(deg, abs=1e-9)
        n_r = _rot(a, -deg) @ n_p
        assert signed_sweep_deg(n_p, n_r, a) == pytest.approx(-deg, abs=1e-9)
    # within the wrap margin of 180 a reading keeps its positive sign (a hem is no wrong-way fold)
    assert signed_sweep_deg(n_p, _rot(a, -175.0) @ n_p, a) == pytest.approx(175.0, abs=1e-9)
    assert signed_sweep_deg(n_p, _rot(a, -175.0) @ n_p, a, wrap_margin_deg=2.0) == pytest.approx(-175.0, abs=1e-9)
    # a twisted child plane (normal tilted 3 deg out of the section plane) keeps the unsigned magnitude and its sign
    tilt = _rot(u, 3.0)
    n_t = tilt @ (_rot(a, 90.0) @ n_p)
    assert signed_sweep_deg(n_p, n_t, a) == pytest.approx(math.degrees(math.acos(float(n_p @ n_t))), abs=1e-9)
    assert signed_sweep_deg(n_p, tilt @ (_rot(a, -90.0) @ n_p), a) < 0


def test_a_negative_sweep_decodes_as_formed_the_wrong_way_beyond_the_flat_limits():
    from domains.production.engine.inspect.decode import _raw_state

    cfg = InspectConfig()
    # the CAD side: unchanged
    assert _raw_state(88.0, 90.0, cfg) == "FORMED"
    assert _raw_state(40.0, 90.0, cfg) == "PARTIALLY_FORMED"
    assert _raw_state(3.0, 90.0, cfg) == "NOT_YET_FORMED"
    # the other side: a kink within the flat limits is a flat sheet, beyond them the bend was formed the wrong way
    assert _raw_state(-3.0, 90.0, cfg) == "NOT_YET_FORMED"
    assert _raw_state(-9.0, 90.0, cfg) == "NOT_YET_FORMED"  # 10 % of the sweep: within flat_fraction_max
    assert _raw_state(-40.0, 90.0, cfg) == "PARTIALLY_FORMED"
    assert _raw_state(-88.0, 90.0, cfg) == "FORMED"
    assert _raw_state(-10.0, 10.0, cfg) == "FORMED"  # a shallow bend formed fully the wrong way


def test_decode_reports_the_signed_fraction():
    from types import SimpleNamespace

    from domains.production.engine.inspect.decode import decode_states

    cfg = InspectConfig()
    bends = [SimpleNamespace(sweep=math.radians(25.0), rank=1, uid="b1"),
             SimpleNamespace(sweep=math.radians(90.0), rank=2, uid="b2")]
    part = SimpleNamespace(bends=bends, n=2)
    out = decode_states(part, [(-11.4, None), (89.0, None)], cfg)
    assert out[0].state == "PARTIALLY_FORMED" and out[0].formed_fraction == pytest.approx(-11.4 / 25.0)
    assert out[1].state == "FORMED"


def test_config_carries_the_wrap_margin_and_checks_it():
    assert InspectConfig().reversed_wrap_margin_deg == 30.0
    with pytest.raises(ConfigError):
        load_config(None, reversed_wrap_margin_deg=90.0)
    with pytest.raises(ConfigError):
        load_config(None, reversed_wrap_margin_deg=-1.0)
