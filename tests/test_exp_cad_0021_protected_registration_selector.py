import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp_cad_0021_protected_registration_selector.py"
SPEC = importlib.util.spec_from_file_location("exp21", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _event(event_id="E1", bins=8, supported=8, points=100, residual=0.2):
    return {
        "event_id": event_id,
        "bin_count": bins,
        "supported_bin_count": supported,
        "carrier_point_count": points,
        "median_carrier_radial_residual_mm": residual,
        "curvature_qualified_bin_bitset": [False] * bins,
        "curvature_qualified_bin_count": 0,
        "curvature_qualified_longest_run": 0,
    }


def _global(fraction=0.8, median=0.5, p90=1.2):
    return {
        "within_2mm_fraction": fraction,
        "median_distance_mm": median,
        "p90_distance_mm": p90,
    }


def _row(candidate_id, *, protected=True, global_pass=True, fraction=0.8, median=0.5, p90=1.2, bounded=True):
    return {
        "candidate_id": candidate_id,
        "bounded_rigid_delta": bounded,
        "protected_neighborhood_pass": protected,
        "material_global_improvement_pass": global_pass,
        "global_visible_reference_metrics": _global(fraction, median, p90),
        "delta_stats": {"translation_mm": 0.1, "rotation_deg": 0.1},
    }


def test_apply_transform_and_rigid_delta_stats():
    transform = np.eye(4)
    transform[:3, 3] = [1.0, -2.0, 2.0]
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert np.allclose(MODULE.apply_transform(points, transform), points + [1.0, -2.0, 2.0])
    stats = MODULE.rigid_delta_stats(transform)
    assert stats["translation_mm"] == 3.0
    assert stats["rotation_deg"] == 0.0
    assert abs(stats["rotation_determinant"] - 1.0) < 1e-12


def test_global_gain_rejects_hidden_tail_regression():
    passed, reasons = MODULE.global_improvement_decision(
        _global(0.80, 0.50, 1.20),
        _global(0.81, 0.49, 1.25),
    )
    assert not passed
    assert "VISIBLE_REFERENCE_P90_REGRESSION" in reasons


def test_protected_neighborhood_rejects_supported_bin_loss_even_if_global_improves():
    passed, reasons = MODULE.protected_neighborhood_decision(
        [_event(supported=8)],
        [_event(supported=7, points=100, residual=0.1)],
    )
    assert not passed
    assert "E1:SUPPORTED_BIN_LOSS" in reasons
    selected = MODULE.select_candidate(
        [
            _row("BASELINE_IDENTITY"),
            _row("P2PL_T2P5", protected=False, global_pass=True, fraction=0.85, median=0.4, p90=1.0),
        ]
    )
    assert selected == "BASELINE_IDENTITY"


def test_selector_accepts_bounded_pareto_improvement_with_protected_support():
    protected, reasons = MODULE.protected_neighborhood_decision(
        [_event()],
        [_event(points=95, residual=0.25)],
    )
    assert protected and not reasons
    global_pass, reasons = MODULE.global_improvement_decision(
        _global(0.80, 0.50, 1.20),
        _global(0.81, 0.49, 1.15),
    )
    assert global_pass and not reasons
    selected = MODULE.select_candidate(
        [
            _row("BASELINE_IDENTITY"),
            _row("P2PL_T2P5", fraction=0.81, median=0.49, p90=1.15),
            _row("P2P_T2P5", fraction=0.805, median=0.48, p90=1.10),
        ]
    )
    assert selected == "P2PL_T2P5"


def test_selector_rejects_unbounded_candidate():
    selected = MODULE.select_candidate(
        [
            _row("BASELINE_IDENTITY"),
            _row("P2PL_T2P5", fraction=0.90, median=0.2, p90=0.5, bounded=False),
        ]
    )
    assert selected == "BASELINE_IDENTITY"


def test_absolute_similarity_rejects_reflection():
    transform = np.eye(4)
    transform[0, 0] = -1.0
    stats = MODULE.absolute_similarity_stats(transform)
    assert stats["reflection_detected"]
    assert not stats["proper_similarity_transform"]


def test_curvature_guard_rejects_loss_even_when_annulus_bins_remain():
    baseline = _event()
    baseline["curvature_qualified_bin_bitset"][3] = True
    baseline["curvature_qualified_bin_count"] = 1
    baseline["curvature_qualified_longest_run"] = 1
    trial = _event()
    passed, reasons = MODULE.protected_neighborhood_decision([baseline], [trial])
    assert not passed
    assert "E1:CURVATURE_QUALIFIED_BIN_LOSS" in reasons


def test_planar_strip_is_not_curvature_qualified():
    event = {"event_id": "E1", "line_start_mm": [0, 0, 0], "line_end_mm": [0, 0, 16], "radii_mm": [2, 3]}
    points = []
    normals = []
    # A planar strip tangent to radius 3 can fill every annulus bin, but its
    # plane competitor is better and its radial-normal aperture is narrow.
    for z in np.linspace(0.1, 15.9, 160):
        for y in np.linspace(-0.2, 0.2, 3):
            points.append([3.0, y, z])
            normals.append([1.0, 0.0, 0.0])
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    neighborhoods = MODULE.build_frozen_event_neighborhoods(points, [event], thickness_mm=1.0)
    metric, _ = MODULE.evaluate_frozen_event_support(
        points, normals, event, neighborhoods[0], thickness_mm=1.0
    )
    assert metric["supported_bin_count"] == metric["bin_count"]
    assert metric["curvature_qualified_bin_count"] == 0
