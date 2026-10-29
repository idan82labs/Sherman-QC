"""e3-1.13.0 density guard (``engine._density_guard``): synthetic only, no scan is read.

A cleaned scan above the validated native envelope (more than 15 points per occupied 0.63 mm cell -- the densest
native scan surveyed reads 12.94 -- or more than 3,000,000 points -- the largest native holds 2.40 M) is not
measured: E3 writes its result at once with validity UNSUPPORTED_INPUT and the reason DENSITY_ABOVE_VALIDATED_RANGE,
so every characteristic is UNKNOWN and the summary is INVALID_INSPECTION. It can never PASS.
"""
import numpy as np
import pytest

from contracts.production.inspection_result import derive_summary
from domains.production.engine.inspect.config import ConfigError, InspectConfig, load_config
from domains.production.engine.inspect.engine import _density_guard


def _sheet(spacing, size=40.0, t=1.0, seed=0, jitter=0.25):
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, size, spacing)
    xx, yy = np.meshgrid(g, g)
    xy = np.c_[xx.ravel(), yy.ravel()]
    xy = xy + rng.uniform(-jitter * spacing, jitter * spacing, xy.shape)
    return np.r_[np.c_[xy, np.zeros(len(xy))], np.c_[xy, np.full(len(xy), t)]]


def test_defaults_and_validation():
    cfg = InspectConfig()
    assert (cfg.density_guard_cell_mm, cfg.density_guard_ppc, cfg.density_guard_max_points) == (0.63, 15.0, 3_000_000)
    assert load_config(None, density_guard_ppc=0.0, density_guard_max_points=0).density_guard_ppc == 0.0
    with pytest.raises(ConfigError):
        load_config(None, density_guard_cell_mm=0.0)
    with pytest.raises(ConfigError):
        load_config(None, density_guard_ppc=-1.0)


def test_native_density_passes_the_guard_unchanged():
    pts = _sheet(0.40)
    before = pts.copy()
    assert _density_guard(pts, InspectConfig()) is None
    assert np.array_equal(pts, before)


def test_dense_input_is_refused_with_the_reason():
    reason = _density_guard(_sheet(0.10), InspectConfig())
    assert reason is not None and reason.startswith("DENSITY_ABOVE_VALIDATED_RANGE:points_per_0.63mm_cell=")
    assert _density_guard(_sheet(0.10), load_config(None, density_guard_ppc=0.0)) is None   # switched off


def test_point_count_criterion():
    pts = _sheet(0.40, size=20.0)
    cfg = load_config(None, density_guard_max_points=len(pts) - 1)
    assert _density_guard(pts, cfg) == f"DENSITY_ABOVE_VALIDATED_RANGE:points={len(pts)}>{len(pts) - 1}"


def test_a_refused_scan_can_never_pass():
    s = derive_summary(validity_state="UNSUPPORTED_INPUT", bends=())
    assert s.state == "INVALID_INSPECTION" and "VALIDITY:UNSUPPORTED_INPUT" in s.reasons
