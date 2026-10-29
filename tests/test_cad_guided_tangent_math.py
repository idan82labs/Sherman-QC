"""Independent pre-run math checks for the measured-flange tangent hypothesis.

The earlier free-circle stress failures remain failures. These tests define a
new conditional model, not additional unseen evidence or a replacement score.
No physical inputs, application packages, native CAD or automatic registration.
Only the reviewed geometry-test helper and its named pure kernel are loaded.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


_HELPER_PATH = Path(__file__).resolve().with_name("test_cad_guided_measurement_geometry.py")
_SPEC = importlib.util.spec_from_file_location("_cad_tangent_constructed_helpers", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
geometry = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = geometry
_SPEC.loader.exec_module(geometry)
measurement = geometry.measurement


SIGNED_SWEEPS = (-135.0, -90.0, -45.0, 45.0, 90.0, 135.0)
RADII = (0.3, 1.0, 30.0)


def independent_tangent_family(radius, sweep_deg, *, rotated=False):
    """Solve the analytic inner-face plane equations without runtime helpers.

    Inner outward normals are(-1,0) and(-cos(phi),-sin(phi)). Their plane
    equations require n_i.c=d_i+R. Thus g=(-1,-tan(phi/2)), h=c_true-R_true*g.
    The two *measured planes* fix h/g; the candidate radius remains free.
    """
    phi = math.radians(sweep_deg)
    center = np.array([0.21, -0.37]) * radius
    g = np.array([-1.0, -math.tan(phi / 2.0)])
    h = center - radius * g
    theta = np.linspace(0.11 * phi, 0.89 * phi, 41)
    points = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    normals = np.array([[-1.0, 0.0], [-math.cos(phi), -math.sin(phi)]])
    offsets = normals @ center - radius
    if rotated:
        angle = 0.63
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        translation = radius * np.array([13.0, -17.0])
        points = points @ rotation.T + translation
        h = h @ rotation.T + translation
        center = center @ rotation.T + translation
        g = g @ rotation.T
        normals = normals @ rotation.T
        offsets = offsets + normals @ translation
    return points, h, g, center, normals, offsets, theta


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("sweep", SIGNED_SWEEPS)
@pytest.mark.parametrize("rotated", (False, True))
def test_tangent_residual_and_jacobian_against_independent_central_differences(radius, sweep, rotated):
    points, h, g, _, normals, offsets, _ = independent_tangent_family(radius, sweep, rotated=rotated)
    # Trial radius differs from true radius, so this is not only a zero-residual
    # derivative check. All quantities except radius remain fixed in the fit.
    trial = 0.93 * radius
    predicted_center = h + trial * g
    residual, jacobian = measurement.tangent_residual_jacobian(trial, points, h, g)
    expected_residual = np.sqrt(np.sum((points - predicted_center)**2, axis=1)) - trial
    assert residual.shape == (len(points),)
    assert jacobian.shape == (len(points), 1)
    assert np.allclose(residual, expected_residual, atol=2e-12 * radius, rtol=2e-12)
    # Signed distance, not absolute plane distance: wrong-side tangency is not
    # another solution of these equations.
    assert np.allclose(normals @ predicted_center - offsets, trial, atol=2e-12 * radius, rtol=0)
    step = 1e-6 * radius
    plus = np.linalg.norm(points - (h + (trial + step) * g), axis=1) - (trial + step)
    minus = np.linalg.norm(points - (h + (trial - step) * g), axis=1) - (trial - step)
    numerical_jacobian = (plus - minus) / (2.0 * step)
    assert np.allclose(jacobian[:, 0], numerical_jacobian, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("sweep", SIGNED_SWEEPS)
def test_contact_derivative_vanishes_and_interior_sensitivity_has_closed_form(sweep):
    phi = math.radians(sweep)
    radius = 7.0
    center = np.array([0.31, -0.27])
    g = np.array([-1.0, -math.tan(phi / 2.0)])
    h = center - radius * g
    # Sample both contact limits and interior points without any flange labels
    # entering the function under test.
    fractions = np.array([0.0, 1e-6, 1e-4, 0.25, 0.5, 0.75, 1.0 - 1e-4,
                          1.0 - 1e-6, 1.0])
    theta = fractions * phi
    points = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    residual, jacobian = measurement.tangent_residual_jacobian(radius, points, h, g)
    expected = np.cos(theta) + math.tan(phi / 2.0) * np.sin(theta) - 1.0
    assert np.allclose(residual, 0.0, atol=3e-14, rtol=0)
    assert np.allclose(jacobian[:, 0], expected, atol=4e-14, rtol=0)
    assert abs(jacobian[0, 0]) < 4e-14
    assert abs(jacobian[-1, 0]) < 4e-14
    assert jacobian[4, 0] == pytest.approx(1.0 / math.cos(phi / 2.0) - 1.0, abs=4e-14)
    assert 0.0 < jacobian[1, 0] < jacobian[2, 0] < jacobian[4, 0]
    # Repeated observations change raw singular magnitude, not normalized
    # sensitivity. This is a mathematical identifiability diagnostic, not a
    # calibrated radius uncertainty or independent sample count.
    repeated = np.repeat(points, 13, axis=0)
    _, duplicate_jacobian = measurement.tangent_residual_jacobian(radius, repeated, h, g)
    original_sensitivity = np.linalg.norm(jacobian) / math.sqrt(len(points))
    duplicate_sensitivity = np.linalg.norm(duplicate_jacobian) / math.sqrt(len(repeated))
    assert duplicate_sensitivity == pytest.approx(original_sensitivity, rel=1e-13)


@pytest.mark.parametrize("radius", (1.0, 5.0, 30.0))
@pytest.mark.parametrize("sweep", SIGNED_SWEEPS)
@pytest.mark.parametrize("factor", (0.9, 1.1))
def test_raw_positions_with_wrong_nominal_radius_signed_sweep_and_shifted_axis(radius, sweep, factor):
    sign = math.copysign(1.0, sweep)
    shift = (0.04, -sign * 0.03, 0.0)
    points, truth = geometry.constructed_scan(
        radius, sweep, radius_factor=factor, sweep_magnitude_change=3.0,
        pose="oblique", center_shift=shift,
    )
    feature = replace(geometry.declared_feature(radius, sweep, "oblique"),
                      fit_model="measured_flange_tangent")
    result, row = geometry.inspect_one(points, feature)
    assert row["status"] == "MEASURED_CONDITIONAL", row
    fitted = row["measurements"]
    assert fitted["measured_radius_mm"] == pytest.approx(truth["radius_mm"], abs=2e-8 * radius)
    assert fitted["radius_deviation_mm"] == pytest.approx((factor - 1.0) * radius, abs=2e-8 * radius)
    assert fitted["measured_sweep_deg"] == pytest.approx(truth["sweep_deg"], abs=1e-8)
    assert fitted["sweep_deviation_deg"] == pytest.approx(sign * 3.0, abs=1e-8)
    center_error = np.asarray(fitted["measured_axis_point"]) - truth["axis_point"]
    assert np.linalg.norm(np.cross(center_error, truth["axis"])) < 5e-8 * radius
    assert abs(fitted["measured_radius_mm"] - feature.nominal_radius_mm) > 0.09 * radius
    assert result["observed_bend_count"] is None
    assert result["conformance"] == "INDETERMINATE"
    assert row["uncertainty_status"] == "UNCALIBRATED"


@pytest.mark.parametrize("sweep", (-90.0, 90.0))
def test_a_circle_not_tangent_to_both_observed_flanges_is_rejected(sweep):
    # Curved observations are a translated circle; finite planar observations
    # retain their original equations. There is deliberately no common tangent
    # circular extrusion at the unchanged declared residual limit.
    arc, _ = geometry.constructed_scan(5.0, sweep, include_flanges=False,
                                       center_shift=(0.08, 0.0, 0.0))
    flanges, _ = geometry.constructed_scan(5.0, sweep, include_arc=False)
    feature = replace(geometry.declared_feature(5.0, sweep), fit_model="measured_flange_tangent")
    _, row = geometry.inspect_one(np.vstack((arc, flanges)), feature)
    assert row["status"] == "UNAVAILABLE", row
    assert row["measurements"] is None


@pytest.mark.parametrize("kind", ("no_arc", "flat_compatible", "near_contact_only"))
def test_tangent_hypothesis_does_not_invent_curvature_from_flanges(kind):
    options = {
        "no_arc": {"include_arc": False},
        "flat_compatible": {"arc_angles_deg": (34.5, 45.0, 55.5)},
        "near_contact_only": {"arc_angles_deg": np.linspace(0.0, 0.1, 41)},
    }[kind]
    points, _ = geometry.constructed_scan(1.0, 90.0, **options)
    feature = replace(geometry.declared_feature(1.0), fit_model="measured_flange_tangent")
    result, row = geometry.inspect_one(points, feature)
    assert row["status"] == "UNAVAILABLE", row
    assert row["measurements"] is None
    assert result["measured_conditional_feature_count"] == 0
    assert result["observed_bend_count"] is None
    if kind == "flat_compatible":
        assert row["reasons"] == ["ARC_COMPATIBLE_WITH_FLAT"]


def test_free_circle_baseline_remains_the_default_and_unknown_fit_model_is_rejected():
    feature = geometry.declared_feature()
    assert feature.fit_model == "free_circle"
    with pytest.raises(ValueError):
        replace(feature, fit_model="copy_nominal_radius")


def test_tangent_entry_preserves_readonly_positions_and_feature_declaration():
    points, _ = geometry.constructed_scan(5.0, -90.0, radius_factor=1.1)
    original = points.copy()
    points.setflags(write=False)
    feature = replace(geometry.declared_feature(5.0, -90.0), fit_model="measured_flange_tangent")
    original_feature = repr(feature)
    _, row = geometry.inspect_one(points, feature)
    assert row["status"] == "MEASURED_CONDITIONAL", row
    assert np.array_equal(points, original)
    assert points.flags.writeable is False
    assert repr(feature) == original_feature
