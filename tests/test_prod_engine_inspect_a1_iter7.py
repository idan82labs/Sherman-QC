"""A1 iteration 7 unit tests of the per-scan engine (E3): each side's station window starts on that side's own
flat material, past the formed arc's tangent line (``metrology._flat_start``), so a real inner radius larger than
the CAD's no longer tilts the station planes toward an underbent reading.

Every rule here is geometric; no part, family or file is named. Pure numpy (no OCP, no E1/E2 geometry,
no physical data).
Run: PYT tests/test_prod_engine_inspect_a1_iter7.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config
from domains.production.engine.inspect.geom import Plane
from domains.production.engine.inspect.metrology import _fit_side, _flat_start

CFG = InspectConfig()
LENGTH = 100.0  # axial length of the synthetic bend (mm)


def _side(ri: float, t: float, depth: float, s_from: float, *, sweep_deg: float = 90.0, skins=(0, 1),
          noise: float = 0.01, ds: float = 0.05, dz: float = 1.0, seed: int = 0, transition_mm: float | None = None):
    """``(s, d, z)`` of one side's skins next to a bend of real inner radius ``ri``.

    ``s`` is the distance from the OML virtual sharp along the side, ``d`` the signed distance to the side's OML
    plane (outward positive; the outer skin's flat is 0, the inner skin's ``-t``). The formed arc ends at
    ``s_T = (ri + t) tan(sweep / 2)``; before it each skin is its arc (radius ``ri + t`` or ``ri``, same centre),
    which bends toward the centre (negative ``d``). ``transition_mm`` replaces the arc's tail by a gentle
    springback transition (as physical scans show): each skin sits ``x^2 / (2 transition_mm)`` toward the centre at
    ``x`` before the tangent line.
    """
    rng = np.random.default_rng(seed)
    ro = ri + t
    half = math.radians(sweep_deg) / 2.0
    s_t = ro * math.tan(half)
    s = np.arange(s_from, depth, ds)
    z = np.arange(0.0, LENGTH, dz)
    S, Z = np.meshgrid(s, z, indexing="ij")
    S, Z = S.ravel(), Z.ravel()
    out_s, out_d, out_z = [], [], []
    for k in skins:
        r = ro if k == 0 else ri
        base = 0.0 if k == 0 else -t
        x = np.clip(s_t - S, 0.0, None)  # distance before the tangent line, along the side
        # on the arc (centre at distance ro from the OML plane, on the inside): the skin sits r - sqrt(r^2 - x^2) in
        if transition_mm is not None:
            d = base - x * x / (2.0 * transition_mm)
            ok = np.ones(len(x), dtype=bool)
        else:
            d = np.where(x > 0.0, base - (r - np.sqrt(np.clip(r * r - x * x, 0.0, None))), base)
            ok = x < r  # past the arc's quarter the skin turns away (not part of this side)
        out_s.append(S[ok])
        out_d.append(d[ok] + rng.normal(0.0, noise, int(ok.sum())))
        out_z.append(Z[ok])
    return np.concatenate(out_s), np.concatenate(out_d), np.concatenate(out_z), s_t


def _start0(ri_cad: float, t: float, sweep_deg: float = 90.0) -> float:
    return (ri_cad + t) * math.tan(math.radians(sweep_deg) / 2.0) + 0.5


# --------------------------------------------------------------------------- 1. the arc's tail moves the window
def test_a_radius_larger_than_the_cad_moves_the_window_past_the_real_tangent_line():
    t, depth = 2.0, 20.0
    start0 = _start0(0.8, t)  # 3.3 mm: the CAD radius 0.8 mm
    s, d, z, s_t = _side(2.6, t, depth, start0)  # the real radius 2.6 mm: the arc ends at 4.6 mm
    got = _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG)
    # the window may keep a tail whose depth stays below the test's departure (sqrt(2 R dev) before s_T)
    assert s_t - math.sqrt(2 * (2.6 + t) * CFG.flat_start_dev_mm) - CFG.flat_start_step_mm <= got <= s_t + 0.3


def test_a_large_sweep_amplifies_the_shift_and_it_is_still_found():
    """A 120 deg sweep moves the tangent line by (ro_real - ro_cad) tan(60 deg)."""
    t, depth = 1.0, 25.0
    start0 = _start0(0.5, t, 120.0)
    s, d, z, s_t = _side(1.5, t, depth, start0, sweep_deg=120.0)
    got = _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG)
    assert s_t - 0.7 <= got <= s_t + 0.3
    assert got > start0 + 0.5


def test_the_cad_radius_keeps_the_cad_window():
    t, depth = 1.5, 20.0
    start0 = _start0(1.0, t)
    s, d, z, _ = _side(1.0, t, depth, start0)
    assert _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG) == pytest.approx(start0)


def test_the_inner_skin_alone_finds_the_same_line():
    t, depth = 2.0, 20.0
    start0 = _start0(0.8, t)
    s, d, z, s_t = _side(2.6, t, depth, start0, skins=(1,))
    got = _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG)
    assert s_t - math.sqrt(2 * 2.6 * CFG.flat_start_dev_mm) - CFG.flat_start_step_mm - 0.25 <= got <= s_t + 0.3


def test_a_deep_arc_whose_skin_leaves_the_band_still_moves_the_window():
    """Far past the CAD tangent line the arc sits deeper than the skin band; the strip losing its skin is a
    departure too."""
    t, depth = 1.0, 25.0
    start0 = _start0(0.2, t)  # 1.7 mm
    s, d, z, s_t = _side(4.0, t, depth, start0, skins=(0,))  # the arc ends at 5.0 mm
    got = _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG)
    assert got >= s_t - 0.8


# --------------------------------------------------------------------------- 2. what must not move it
@pytest.mark.parametrize("slope_deg", [-2.0, 1.5])
def test_a_tilted_flat_side_is_a_line_not_a_tail(slope_deg):
    t, depth = 1.5, 30.0
    start0 = _start0(1.0, t)
    s, d, z, _ = _side(1.0, t, depth, start0)
    d = d + math.tan(math.radians(slope_deg)) * (s - start0)
    assert _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG) == pytest.approx(start0)


@pytest.mark.parametrize("radius_mm", [-400.0, 400.0])
def test_a_bowed_panel_does_not_move_the_window(radius_mm):
    """A panel bowed with a 400 mm radius either way (its skin 0.5 mm off a 40 mm chord) is no arc tail."""
    t, depth = 1.5, 40.0
    start0 = _start0(1.0, t)
    s, d, z, _ = _side(1.0, t, depth, start0)
    d = d + (s - 20.0) ** 2 / (2.0 * radius_mm)
    assert _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG) == pytest.approx(start0)


@pytest.mark.parametrize("radius_mm, peak_mm", [(25.0, 3.0), (15.0, 2.0), (40.0, 4.0), (25.0, 0.5)])
def test_a_short_tab_curved_throughout_keeps_the_cad_window(radius_mm, peak_mm):
    """A tab with no flat material (its skin a 15-40 mm radius all along) has nothing to move to: the line beyond
    any strip is not straight, so no strip is judged flat and the CAD start stays."""
    rng = np.random.default_rng(1)
    t, depth = 1.5, 7.5
    start0 = _start0(0.8, t, 30.0)
    s = np.arange(start0, depth, 0.05)
    z = np.arange(0.0, 12.0, 0.5)
    S, Z = np.meshgrid(s, z, indexing="ij")
    S, Z = S.ravel(), Z.ravel()
    d = -(S - peak_mm) ** 2 / (2.0 * radius_mm)
    ss, dd, zz = np.r_[S, S], np.r_[d, d - t] + rng.normal(0.0, 0.008, 2 * len(S)), np.r_[Z, Z]
    assert _flat_start(ss, dd, zz, start0, depth, t, 0.0, 12.0, CFG) == pytest.approx(start0)


def test_a_gap_in_the_scan_at_the_window_start_moves_no_data_out():
    """A flat side whose first millimetre was not scanned: the strip there is not judged, and the window starts at
    the first flat strip, past the gap, so no scanned point is dropped."""
    t, depth = 1.5, 20.0
    start0 = _start0(1.0, t)
    s, d, z, _ = _side(1.0, t, depth, start0)
    seen = s >= start0 + 1.0
    got = _flat_start(s[seen], d[seen], z[seen], start0, depth, t, 0.0, LENGTH, CFG)
    assert start0 <= got <= start0 + 1.0 + 1e-9


def test_the_search_is_capped_and_can_be_turned_off():
    t, depth = 2.0, 20.0
    start0 = _start0(0.8, t)
    s, d, z, _ = _side(2.6, t, depth, start0)
    off = load_config(None, flat_start_max_shift_mm=0.0)
    assert _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, off) == pytest.approx(start0)
    capped = load_config(None, flat_start_max_shift_mm=0.5)
    assert _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, capped) <= start0 + 0.5 + 1e-9
    # a short side keeps a reference: never past depth - test strip - gap - half the reference
    short = 7.0
    s2, d2, z2, _ = _side(2.6, t, short, start0)
    got = _flat_start(s2, d2, z2, start0, short, t, 0.0, LENGTH, CFG)
    assert got <= short - CFG.flat_start_test_mm - CFG.flat_start_gap_mm - 0.5 * CFG.flat_start_ref_mm + 1e-9


def test_too_few_points_keep_the_cad_window():
    t, depth = 2.0, 20.0
    start0 = _start0(0.8, t)
    s, d, z, _ = _side(2.6, t, depth, start0, ds=0.5, dz=50.0)
    assert _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG) == pytest.approx(start0)


# --------------------------------------------------------------------------- 3. the angle the stations read
def _station_plane(s, d, z, lo: float, hi: float, t: float) -> Plane:
    """The side's station plane (``_fit_side``) on the window ``[lo, hi)``, points embedded as x = s, y = z,
    height = d (the OML plane is height 0, outward normal +height)."""
    m = (s >= lo) & (s < hi)
    pts = np.c_[s[m], z[m], d[m]]
    pl, _, _, _ = _fit_side(pts, Plane(np.array([0.0, 0.0, 1.0]), 0.0), t, CFG)
    assert pl is not None
    return pl


@pytest.mark.parametrize("transition_mm", [None, 14.0])
def test_the_station_plane_on_the_moved_window_reads_the_flat_side(transition_mm):
    """With the CAD window the arc's tail (a circular one, or the gentle springback transition physical scans show)
    tilts a short side's station plane toward the bend's centre near the bend: an underbent reading. On the moved
    window most of that tilt is gone."""
    t, depth = 2.0, 9.0
    start0 = _start0(0.8, t)
    s, d, z, _ = _side(3.0 if transition_mm else 2.6, t, depth, start0, noise=0.005, transition_mm=transition_mm)
    tilt = lambda pl: math.degrees(math.atan2(float(pl.n[0]), float(pl.n[2])))  # noqa: E731
    before = tilt(_station_plane(s, d, z, start0, depth, t))
    start = _flat_start(s, d, z, start0, depth, t, 0.0, LENGTH, CFG)
    after = tilt(_station_plane(s, d, z, start, depth, t))
    assert start > start0
    assert before < -0.08  # the plane rises away from the bend: its side leans away, underbent
    assert abs(after) < 0.6 * abs(before)


# --------------------------------------------------------------------------- 4. config
def test_flat_start_config_is_range_checked(tmp_path):
    for key, bad in (("flat_start_step_mm", 0.0), ("flat_start_test_mm", -1.0), ("flat_start_ref_mm", 0.0),
                     ("flat_start_dev_mm", 0.0), ("flat_start_max_shift_mm", -0.5), ("flat_start_gap_mm", -0.1),
                     ("flat_start_chunk_mm", 0.0)):
        with pytest.raises(ConfigError):
            load_config(None, **{key: bad})
    assert load_config(None, flat_start_max_shift_mm=0.0).flat_start_max_shift_mm == 0.0
    assert load_config(None, flat_start_gap_mm=0.0).flat_start_gap_mm == 0.0
