"""Isolated synthetic checks pinning the V9 `max_angular_gap_deg` 5 -> 30 support-policy delta.

Constructed supports through the unchanged `build_finite_support_instance` adapter, varying only the
angular-gap cap (the angular audit's checks 1-4): threshold/monotonicity, lattice phase and numeric
uniqueness, missing interior samples, and local-versus-global support with axial-only 0.3 mm
neighbours. These are clean-model facts about the adapter, not physical predictions.
"""
import importlib
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inst():
    saved = {k: v for k, v in sys.modules.items() if k == "backend" or k.startswith("backend.")}
    for k in saved:
        del sys.modules[k]
    package = types.ModuleType("backend"); package.__path__ = [str(ROOT/"backend")]; sys.modules["backend"] = package
    try:
        yield importlib.import_module("backend.cad_guided_instances")
    finally:
        for k in tuple(sys.modules):
            if k == "backend" or k.startswith("backend."):
                del sys.modules[k]
        sys.modules.update(saved)


def _support(angles_deg, *, radius=.8, axial=(0., 1., 2., 3., 4.), phase_deg=0., drop=(), jitter=0., seed=0,
             flange_axial=None):
    """Arc rows at the given cross-section angles (about +z, centre at the origin) repeated at each axial
    station, plus two flange strips (tangent planes at 0 and 90 deg) that always satisfy the flange rules."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in angles_deg:
        if a in drop:
            continue
        for z in axial:
            th = math.radians(a+phase_deg+(rng.normal(0, jitter) if jitter else 0.))
            rows.append((radius*math.cos(th), radius*math.sin(th), z))
    arc = np.array(rows, float)
    fz = axial if flange_axial is None else flange_axial
    f0 = np.array([(radius, -d, z) for d in (1., 2., 3., 4.) for z in fz], float)      # tangent at 0 deg
    f1 = np.array([(-d, radius, z) for d in (1., 2., 3., 4.) for z in fz], float)      # tangent at 90 deg
    points = np.vstack((arc, f0, f1))
    ids = (list(range(len(arc))), [list(range(len(arc), len(arc)+len(f0))), list(range(len(arc)+len(f0), len(points)))])
    measurement = dict(measured_axis_point=[0., 0., 0.], measured_axis_direction=[0., 0., 1.],
                       measured_radius_mm=radius, circle_max_residual_mm=1e-6)
    return points, measurement, ids


def _run(inst, gap, angles, **kwargs):
    points, measurement, (arc, flanges) = _support(angles, **kwargs)
    policy = inst.SupportPolicy(max_axial_gap_mm=4., max_angular_gap_deg=gap, min_arc_sweep_deg=20., min_local_points=3)
    out = inst.build_finite_support_instance(points, measurement, arc, flanges, feature_id="f", policy=policy)
    if out["status"] == "SUPPORTED":
        assert out["reasons"] == [] and out["instance"] is not None
        return "SUPPORTED", out["instance"]
    assert out["instance"] is None and len(out["reasons"]) == 1
    return out["reasons"][0], None


def test_threshold_and_monotonicity_on_uniform_angle_grids(inst):
    dense = (0., 5., 10., 15., 20.)
    fixture = (0., 22.5, 45.)
    for gap in (5., 10., 15., 20., 25., 30., 45., 90.):
        assert _run(inst, gap, dense)[0] == "SUPPORTED", gap          # every 5-supported grid stays supported at 30
    for gap, expected in ((5., "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"), (15., "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"),
                          (22.4, "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"), (22.5, "SUPPORTED"), (25., "SUPPORTED"),
                          (30., "SUPPORTED"), (45., "SUPPORTED")):
        assert _run(inst, gap, fixture)[0] == expected, gap
    # Just above the new cap: a 30.5 deg step refuses at 30 and is admitted at 31 (numerical guard is far smaller).
    assert _run(inst, 30., (0., 30.5, 61.))[0] == "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"
    assert _run(inst, 31., (0., 30.5, 61.))[0] == "SUPPORTED"
    status, out = _run(inst, 30., fixture)
    assert out["observation_status"] == "SAMPLE_SUPPORTED_CONDITIONAL" and out["continuous_surface_coverage"] == "NOT_ESTABLISHED"
    assert all(abs(slab["max_angular_gap_deg"]-22.5) < 1e-9 and abs(slab["arc_sweep_deg"]-45.) < 1e-9 for slab in out["local_coverage"])


def test_lattice_phase_and_numeric_uniqueness_are_not_cured_by_the_gap_cap(inst):
    # Zero phase: three columns (22.5 / 45 / 67.5) as in the connected fixture after the ideal exclusion.
    assert _run(inst, 30., (22.5, 45., 67.5))[0] == "SUPPORTED"
    # Half-step phase inside the same ideal exclusion interval leaves two columns: refused for any cap
    # (too few distinct angular stations), so raising the cap cannot make the support phase-robust.
    for gap in (5., 30., 90.):
        assert _run(inst, gap, (33.75, 56.25))[0] == "INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT", gap
    # Per-row angular jitter (0.05 deg) on the same two nominal columns creates many distinct
    # floating-point angles, so the exact-uniqueness station count no longer refuses: the 22.5 deg
    # inter-column gap is admitted at 30 and refused at 5. This is the documented limitation (exact
    # uniqueness is not a noise-resolved station count), not a claim of three resolved stations.
    assert _run(inst, 30., (33.75, 56.25), jitter=.05, seed=7)[0] == "SUPPORTED"
    assert _run(inst, 5., (33.75, 56.25), jitter=.05, seed=7)[0] == "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"


def test_missing_interior_samples_bridge_up_to_the_cap_and_refuse_beyond_it(inst):
    grid = (0., 15., 30., 45., 60.)
    assert _run(inst, 30., grid)[0] == "SUPPORTED"
    assert _run(inst, 30., grid, drop=(30.,))[0] == "SUPPORTED"                       # one missing sample: 30 deg gap bridged
    assert _run(inst, 30., grid, drop=(15., 30.))[0] == "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"   # 45 deg gap refused
    assert _run(inst, 5., grid, drop=(30.,))[0] == "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"
    assert _run(inst, 30., (0., 22.5, 45., 67.5), drop=(45.,))[0] == "DISCONNECTED_LOCAL_ANGULAR_SUPPORT"  # 45 deg gap from a lost column


def test_local_slab_support_is_required_and_axial_only_neighbours_are_not_angular_stations(inst):
    # Global angles are rich, but one slab (z in [4, 8]) only carries a single angular column: local
    # refusal persists whatever the angular cap is.
    def build(gap):
        # Rich angular columns only at z in [0, 3]; the second slab [4, 8] carries one angular column.
        pts_a, m, (arc_a, fl_a) = _support((0., 5., 10., 15., 20.), axial=(0., 1., 2., 3.), flange_axial=(0., 2., 4., 6., 8.))
        pts_b, _, (arc_b, _) = _support((30.,), axial=(4.5, 5.5, 6.5, 7.5, 8.))
        arc_b = [i+len(pts_a) for i in arc_b]
        points = np.vstack((pts_a, pts_b[:len(arc_b)]))
        policy = inst.SupportPolicy(max_axial_gap_mm=4., max_angular_gap_deg=gap, min_arc_sweep_deg=20., min_local_points=3)
        out = inst.build_finite_support_instance(points, m, arc_a+arc_b, fl_a, feature_id="f", policy=policy)
        return out["status"] if out["status"] == "SUPPORTED" else out["reasons"][0]
    for gap in (5., 30., 90.):
        assert build(gap) == "INSUFFICIENT_LOCAL_ANGULAR_STATIONS", gap
    # Axial-only 0.3 mm neighbours on ONE angular column are many rows 0.3 mm apart but zero angular spacing:
    # generic row spacing is not an angular-spacing measurement.
    points, m, (arc, flanges) = _support((0.,), axial=tuple(np.arange(0., 4.01, .3)))
    policy = inst.SupportPolicy(max_axial_gap_mm=4., max_angular_gap_deg=30., min_arc_sweep_deg=20., min_local_points=3)
    out = inst.build_finite_support_instance(points, m, arc, flanges, feature_id="f", policy=policy)
    assert out["status"] == "UNAVAILABLE" and out["reasons"] == ["INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT"]


def test_chord_and_arc_length_arithmetic_behind_the_declared_reason():
    r = .8
    assert abs(math.degrees(2*math.asin(.3/(2*r)))-21.61384575) < 1e-6
    assert abs(math.degrees(.3/r)-21.48591732) < 1e-6
    assert abs(2*r*math.sin(math.radians(15.))-.41411047) < 1e-6
    n = max(3, int(round(r*(math.pi/2)/.3))+1)
    assert n == 5 and [round(v, 3) for v in np.linspace(0., 90., n)] == [0., 22.5, 45., 67.5, 90.]
    alpha = math.degrees(math.acos(1-.04/r))
    assert abs(alpha-18.19487234) < 1e-6 and abs((90-2*alpha)-53.61025532) < 1e-6
    assert abs(r*(1-math.cos(math.radians(22.5)))/2-.030448187) < 1e-8
