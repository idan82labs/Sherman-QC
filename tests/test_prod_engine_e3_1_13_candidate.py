"""e3-1.13.0 (promoted from the rc candidate): one-skin swap in the station re-seat, near-tie withholding helpers, safety-net U rules."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from contracts.production.characteristics import guard_band_decision
from domains.production.engine.inspect import metrology as M
from domains.production.engine.inspect.config import ENGINE_VERSION, InspectConfig, load_config
from domains.production.engine.inspect.engine import _gb, review_u, withhold_u
from domains.production.engine.inspect.geom import Plane


def _strip(t=0.92, clutter=6000, seed=0):
    """A flat strip whose two skins lie at z = 0 and z = t (outer), plus clutter far from both (the arc, other sheets)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 20, size=(3000, 2))
    inner = np.c_[xy[:1500], rng.normal(0.0, 0.01, 1500)]
    outer = np.c_[xy[1500:], rng.normal(t, 0.01, 1500)]
    junk = np.c_[rng.uniform(0, 20, size=(clutter, 2)), rng.uniform(-30, -10, clutter)]
    return np.r_[inner, outer, junk]


def test_version_is_e3_1_13_0():
    assert ENGINE_VERSION == "e3-1.14.0"  # the e3-1.13 parts are carried unchanged


def test_one_skin_side_on_its_inner_skin_moves_one_thickness_out():
    t, cfg = 0.92, InspectConfig()
    pts = _strip(t)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)  # the region took the inner skin for the OML
    assert M._reseat_plane(pts, plane, t, cfg) is None  # e3-1.12 behaviour: the share guard counts the clutter
    q = M._reseat_plane(pts, plane, t, cfg, one_skin=True)
    assert q is not None and abs(float(q.d) - t) < 0.05 and float(q.n @ plane.n) > 0.999


def test_swap_needs_a_one_skin_panel_and_can_be_switched_off():
    t = 0.92
    pts = _strip(t)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    assert M._reseat_plane(pts, plane, t, InspectConfig(), one_skin=False) is None
    assert M._reseat_plane(pts, plane, t, InspectConfig(one_skin_swap=0), one_skin=True) is None


def test_plane_one_thickness_outside_the_outer_skin_moves_in():
    t, cfg = 0.92, InspectConfig()
    pts = _strip(t)
    plane = Plane(np.array([0.0, 0.0, 1.0]), 2 * t)  # one thickness above the outer skin
    q = M._reseat_plane(pts, plane, t, cfg, one_skin=True)
    assert q is not None and abs(float(q.d) - t) < 0.05


def test_a_true_two_skin_side_is_left_alone():
    t, cfg = 0.92, InspectConfig()
    pts = _strip(t)
    plane = Plane(np.array([0.0, 0.0, 1.0]), t)  # already on the outer skin
    assert M._reseat_plane(pts, plane, t, cfg, one_skin=True) is None


def _nom(nominal=10.0, plus=0.5, minus=0.5):
    return SimpleNamespace(nominal=nominal, tolerance=SimpleNamespace(plus=plus, minus=minus))


@pytest.mark.parametrize("value,u", [(10.0, 0.1), (10.3, 0.1), (11.0, 0.1), (8.0, 0.2), (9.45, 0.04), (10.6, 0.05)])
def test_withheld_u_is_always_review_and_never_smaller(value, u):
    nom = _nom()
    for delta in (0.0, 0.3, 2.0):
        u2 = withhold_u(nom, value, u, delta)
        assert u2 >= u
        assert _gb(nom, value, u2) == "REVIEW"


def test_widening_u_never_turns_fail_into_pass():
    nom = _nom()
    for value in np.linspace(8.0, 12.0, 81):
        for u in (0.05, 0.2):
            d0 = _gb(nom, value, u)
            for s in (0.0, 0.1, 1.0):
                d1 = _gb(nom, value, math.hypot(u, 2 * s))
                assert not (d0 == "FAIL" and d1 == "PASS")
                assert not (d0 != "PASS" and d1 == "PASS")


def test_safety_net_config_flags():
    cfg = InspectConfig()  # e3-1.13.0 defaults: the promoted arm (a), K = 2, policy 1, serial, no record
    assert (cfg.resample_check_k, cfg.resample_check_policy) == (2, 1)
    assert (cfg.resample_check_parallel, cfg.resample_check_record) == (0, 0)
    c = load_config(None, resample_check_k=2, resample_check_policy=1)
    assert c.resample_check_k == 2 and c.resample_check_policy == 1
    assert guard_band_decision(0.0, 0.1, 0.5, 0.5) == "PASS"
