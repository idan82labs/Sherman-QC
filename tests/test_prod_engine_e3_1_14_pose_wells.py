"""e3-1.14.0 (promoted from the rc candidate): dense cell thinning, tilt pose wells and alternative-pose withholding helpers (synthetic)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config


def test_dense_thin_keeps_one_actual_point_per_cell_nearest_the_mean():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 5, size=(20000, 3))
    out = E._dense_thin(pts, 0.63)
    keys = E._cell_keys(pts, 0.63)
    assert len(out) == len(np.unique(keys))
    # every kept point is an input point
    s = {tuple(p) for p in pts}
    assert all(tuple(p) in s for p in out)
    # and it is the one nearest its cell's mean
    for p in out[:200]:
        k = E._cell_keys(np.vstack([pts, p]), 0.63)[-1]
        cell = pts[keys == k]
        d = np.square(cell - cell.mean(axis=0)).sum(axis=1)
        assert np.isclose(np.square(p - cell.mean(axis=0)).sum(), d.min())


def test_dense_thin_is_order_independent_as_a_set():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 3, size=(5000, 3))
    a = E._dense_thin(pts, 0.63)
    b = E._dense_thin(pts[rng.permutation(len(pts))], 0.63)
    assert sorted(map(tuple, a)) == sorted(map(tuple, b))


def test_cell_density():
    pts = np.repeat(np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]]), 10, axis=0)
    assert E._cell_density(pts, 0.63) == 10.0


def _model(half_u=48.0, half_v=230.0):
    poly = np.array([[-half_u, -half_v], [half_u, -half_v], [half_u, half_v], [-half_u, half_v]])
    f1, f2, n0 = np.eye(3)
    return SimpleNamespace(polygon=poly, p0=np.zeros(3), f1=f1, f2=f2, n0=n0,
                           to2d=lambda q: np.asarray(q)[:, :2])


@pytest.mark.parametrize("deg", [0.25, 0.63, 1.0])
def test_tilt_and_gap(deg):
    m = _model()
    T = E._tilt_about(np.eye(4), np.zeros(3), m.f1, deg)
    assert math.isclose(E._pose_gap_mm(T, np.eye(4), m), 230.0 * math.sin(math.radians(deg)), rel_tol=1e-3)
    assert E._pose_gap_mm(np.eye(4), np.eye(4), m) == 0.0


def test_pose_well_config():
    c = InspectConfig()
    assert c.pose_wells == 1 and c.pose_well_switch_margin < c.pose_well_alt_margin
    assert load_config(None, pose_wells=0).pose_wells == 0
    with pytest.raises(ConfigError):
        load_config(None, pose_well_alt_margin=-0.1)


def test_alternative_withhold_never_creates_a_pass():
    nom = SimpleNamespace(nominal=72.6, tolerance=SimpleNamespace(plus=0.5, minus=0.5))
    for v in np.linspace(71.0, 74.5, 71):
        for u in (0.05, 0.2):
            d0 = E._gb(nom, v, u)
            if d0 not in ("PASS", "FAIL"):
                continue
            for delta in (0.0, 0.9, 1.3):
                u2 = E.withhold_u(nom, v, u, delta)
                assert E._gb(nom, v, u2) == "REVIEW"


def test_thinned_cloud_passes_the_guard_and_thinning_can_be_switched_off():
    from domains.production.engine.inspect.engine import _density_guard
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 6, size=(60000, 3)) * np.array([1.0, 1.0, 0.05])
    cfg = InspectConfig()
    assert _density_guard(pts, cfg) is not None            # above the native envelope
    thin = E._dense_thin(pts, cfg.density_guard_cell_mm)
    assert _density_guard(thin, cfg) is None                # one point per cell
    assert load_config(None, dense_thin=0).dense_thin == 0  # e3-1.13 behaviour: refuse
