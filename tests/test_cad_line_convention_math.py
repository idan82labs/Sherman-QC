"""Invented circular sections; carrier semantics are not physical Gold truth."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def evaluator():
    path = Path(__file__).resolve().parents[1] / "backend/cad_guided_evaluation.py"
    name = "invented_carrier_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("angle", [20., 45., 90., 135., 160.])
@pytest.mark.parametrize("radius", [1., 5., 50.])
def test_ideal_tangent_intersection_formula_against_independent_linear_solve(angle, radius):
    h = math.radians(angle / 2)
    normals = np.array([[math.cos(h), math.sin(h)], [math.cos(h), -math.sin(h)]])
    intersection = np.linalg.solve(normals, [radius, radius])
    expected = np.array([radius / math.cos(h), 0.])
    np.testing.assert_allclose(intersection, expected, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(normals @ intersection, [radius, radius], rtol=2e-15, atol=2e-13)
    # End-point radius vectors are the tangent-plane normals scaled by R.
    for normal in normals:
        assert math.isclose(np.dot(normal, radius * normal), radius, rel_tol=2e-15)


def test_perfect_but_different_carrier_fails_unchanged_four_mm_matcher(evaluator):
    def line(name, x):
        return {"bend_id": name, "line_start": [x, 0., 0.], "line_end": [x, 0., 30.]}

    center = line("center", 0.)
    midpoint = line("half_thickness_mid_arc", 5.5)
    intersection = line("ideal_tangent_intersection", 5. * math.sqrt(2.))
    limits = evaluator.Thresholds()
    mismatch = evaluator._match_lines([midpoint], [center], limits)
    assert (mismatch["true_positive"], mismatch["false_positive"], mismatch["false_negative"]) == (0, 1, 1)
    identity = evaluator._match_lines([midpoint], [midpoint], limits)
    assert (identity["true_positive"], identity["false_positive"], identity["false_negative"]) == (1, 0, 0)
    metrics = evaluator.pair_metrics(np.array([midpoint["line_start"], midpoint["line_end"]]),
                                     np.array([intersection["line_start"], intersection["line_end"]]))
    assert metrics["axis_angle_deg"] == 0.
    assert math.isclose(metrics["transverse_offset_mm"], 1.5710678118654755, abs_tol=1e-14)
    # Matching tolerance cannot prove two carriers denote the same measurand.
    assert evaluator._match_lines([midpoint], [intersection], limits)["true_positive"] == 1
