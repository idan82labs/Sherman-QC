"""Constructed position-only checks for the explicit offline measurement entry.

Frozen before execution: 324 exact cases, 24 deterministic bounded-noise cases,
and the named negatives below. This is not native feature extraction, automatic
registration, manufacturing identity, physical accuracy, bend count, or F1.
The paired baseline is specifically a *nominal-radius-copy ablation*, not a run
of the incumbent whole engine. It is scored on the same declared eligible cases.

Run only after source review, with one worker, BLAS/OpenMP one thread, a 300 s
wall/240 s CPU/2 GiB process-tree RSS budget. No test runs a subprocess or reads
input files. The only project source loaded is the named pure offline module.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "backend" / "cad_guided_measurement.py"
_SPEC = importlib.util.spec_from_file_location("_cad_guided_measurement_geometry_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
measurement = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves the defining module; never import the backend package.
sys.modules[_SPEC.name] = measurement
_SPEC.loader.exec_module(measurement)


RADII_MM = (1.0, 5.0, 30.0)
SIGNED_SWEEPS_DEG = (-135.0, -90.0, -45.0, 45.0, 90.0, 135.0)
RADIUS_FACTORS = (0.9, 1.0, 1.1)
SWEEP_MAGNITUDE_CHANGES_DEG = (-3.0, 0.0, 3.0)
POSES = ("identity", "oblique")
EXACT_CASES = tuple(itertools.product(
    RADII_MM, SIGNED_SWEEPS_DEG, RADIUS_FACTORS,
    SWEEP_MAGNITUDE_CHANGES_DEG, POSES,
))
NOISE_CASES = tuple(itertools.product(
    RADII_MM, (-135.0, -90.0, 90.0, 135.0), (0.0005, 0.001),
))


def rigid_pose(name, scale):
    """Independent Rodrigues construction; no fitted orientation enters truth."""
    if name == "identity":
        return np.eye(3), np.zeros(3)
    assert name == "oblique"
    direction = np.array([2.0, -1.0, 3.0]) / math.sqrt(14.0)
    x, y, z = direction
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = 0.73
    rotation = (math.cos(angle) * np.eye(3)
                + (1.0 - math.cos(angle)) * np.outer(direction, direction)
                + math.sin(angle) * cross)
    return rotation, scale * np.array([31.0, -47.0, 19.0])


def declared_feature(radius=5.0, sweep=90.0, pose="identity", feature_id="invented:0"):
    """All thresholds fixed here before the first constructed evaluation."""
    rotation, translation = rigid_pose(pose, radius)
    return measurement.CADGuidedFeature(
        feature_id=feature_id,
        nominal_radius_mm=radius,
        nominal_sweep_deg=sweep,
        origin=tuple(translation),
        axis=tuple(rotation[:, 2]),
        start_radial=tuple(rotation[:, 0]),
        span_mm=(0.0, 6.0 * radius),
        flange_window_mm=(0.5 * radius, 2.5 * radius),
        search_margin_mm=0.6 * radius,
        max_fit_residual_mm=0.01 * radius,
        plane_exclusion_mm=0.015 * radius,
        max_tangency_error_mm=0.02 * radius,
        nominal_source="constructed CAD declaration; not verified native ownership",
    )


def constructed_scan(nominal_radius=5.0, nominal_sweep=90.0, *, radius_factor=1.0,
                     sweep_magnitude_change=0.0, pose="identity", noise_fraction=0.0,
                     arc_angles_deg=None, include_arc=True, include_flanges=True,
                     arc_radius_factors=None, axial_fraction=(0.0, 1.0),
                     center_shift=(0.0, 0.0, 0.0)):
    """Sample analytic circular extrusion and its two finite tangent flanges.

    Only returned N-by-3 positions and the nominal declaration enter inference.
    Angles, memberships, true radius and true axis stay in this test generator.
    Both flanges use the *actual* radius/sweep, so deviated parts remain tangent.
    """
    sign = math.copysign(1.0, nominal_sweep)
    actual_sweep = sign * (abs(nominal_sweep) + sweep_magnitude_change)
    actual_radius = nominal_radius * radius_factor
    phi = math.radians(actual_sweep)
    axial = np.linspace(axial_fraction[0], axial_fraction[1], 9) * 6.0 * nominal_radius
    shift = np.asarray(center_shift, dtype=float) * nominal_radius
    blocks = []
    if include_arc:
        angles = (np.linspace(0.0, actual_sweep, 81) if arc_angles_deg is None
                  else np.asarray(arc_angles_deg, dtype=float))
        radii = ((actual_radius,) if arc_radius_factors is None else
                 tuple(nominal_radius * value for value in arc_radius_factors))
        for radius in radii:
            # Columns ordered by angle and independently replicated along z.
            blocks.append(np.array([
                (radius * math.cos(math.radians(theta)),
                 radius * math.sin(math.radians(theta)), z)
                for theta in angles for z in axial
            ], dtype=float))
    if include_flanges:
        first_contact = np.array([actual_radius, 0.0, 0.0])
        second_contact = actual_radius * np.array([math.cos(phi), math.sin(phi), 0.0])
        first_ray = np.array([0.0, -sign, 0.0])
        second_ray = sign * np.array([-math.sin(phi), math.cos(phi), 0.0])
        for contact, ray in ((first_contact, first_ray), (second_contact, second_ray)):
            blocks.append(np.array([
                contact + distance * ray + np.array([0.0, 0.0, z])
                for distance in np.linspace(0.6, 2.4, 17) * nominal_radius
                for z in axial
            ], dtype=float))
    local = np.vstack(blocks) if blocks else np.empty((0, 3), dtype=float)
    local = local + shift
    if noise_fraction:
        # Bounded deterministic coordinates, not a claim of scanner noise law.
        index = np.arange(len(local), dtype=float)[:, None] + 1.0
        noise = np.sin(index * np.array([[math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0)]]))
        local = local + noise_fraction * nominal_radius * noise
    rotation, translation = rigid_pose(pose, nominal_radius)
    points = local @ rotation.T + translation
    truth = {
        "radius_mm": actual_radius,
        "sweep_deg": actual_sweep,
        "axis": rotation[:, 2],
        "axis_point": shift @ rotation.T + translation,
        "sampled_axial_interval_mm": np.asarray(axial_fraction) * 6.0 * nominal_radius,
    }
    return points, truth


def inspect_one(points, feature, registration_status="DECLARED_ALIGNED"):
    result = measurement.inspect_cad_guided_scan(
        points, [feature], registration_status=registration_status,
    )
    assert result["observed_bend_count"] is None
    assert result["conformance"] == "INDETERMINATE"
    assert result["production_use_allowed"] is False
    assert result["feature_identity_status"] == "DECLARED_NOT_AUTOMATICALLY_VERIFIED"
    row = result["features"][0]
    assert row["conformance"] == "INDETERMINATE"
    assert row["uncertainty_status"] == "UNCALIBRATED"
    assert row["physical_completion_state"] == "UNKNOWN"
    if row["measurements"] is not None:
        assert row["measurements"]["physical_trim_endpoints"] is None
        assert row["measurements"]["physical_extent_deviation_mm"] is None
        assert row["measurements"]["continuous_surface_coverage"] == "NOT_ESTABLISHED"
        assert "support_row_indices" not in row["measurements"]
    return result, row


def analytic_finite_arc_eligibility(radius, sweep, factor, magnitude_change):
    """Independent predeclared planar alternative for the discrete exact arc.

    All sampled circle points are known convex-polygon vertices in angular order.
    Enumerate this polygon's edges directly (no Qhull, fit, runtime selector or
    runtime strip helper). Width/2 is the best free-offset enclosing line error.
    In particular, some short45° strata must abstain at the declared .01R limit.
    """
    actual_magnitude = abs(sweep) + magnitude_change
    angles = np.linspace(0.0, actual_magnitude, 81)
    actual_radius = radius * factor
    radians = np.radians(angles)
    eligible = ((angles >= 8.0) & (angles <= abs(sweep) - 8.0)
                & (actual_radius * (1.0 - np.cos(radians)) > 0.015 * radius)
                & (actual_radius * (1.0 - np.cos(np.radians(actual_magnitude) - radians))
                   > 0.015 * radius))
    selected = radians[eligible]
    if len(selected) < 2:
        return False, {"flat_halfwidth_mm": 0.0, "arc_span_deg": 0.0}
    polygon = actual_radius * np.column_stack((np.cos(selected), np.sin(selected)))
    widths = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        normal = np.array([-edge[1], edge[0]]) / math.hypot(*edge)
        widths.append(float(np.ptp(polygon @ normal)))
    halfwidth = min(widths) / 2.0
    span = math.degrees(float(np.ptp(selected)))
    # Matrix cases are separated from both boundaries; this is not a tuned guard.
    assert abs(halfwidth - 0.01 * radius) > 1e-6 * radius
    assert abs(span - 20.0) > 1e-4
    return halfwidth > 0.01 * radius and span >= 20.0, {
        "flat_halfwidth_mm": halfwidth, "arc_span_deg": span,
    }


def run_frozen_paired_radius_ablation():
    """Run exactly324 cases once, retaining every unexpected case in the report."""
    failures = []
    measured = expected_measured = expected_abstained = 0
    baseline_squared = challenger_squared = 0.0
    baseline_relative_squared = challenger_relative_squared = 0.0
    max_radius_error = max_sweep_error = max_axis_line_error = 0.0
    scored = 0
    reasons = {}
    for radius, sweep, factor, delta, pose in EXACT_CASES:
        case = (radius, sweep, factor, delta, pose)
        feature = declared_feature(radius, sweep, pose)
        points, truth = constructed_scan(
            radius, sweep, radius_factor=factor, sweep_magnitude_change=delta, pose=pose,
        )
        should_measure, eligibility = analytic_finite_arc_eligibility(radius, sweep, factor, delta)
        expected_measured += int(should_measure)
        expected_abstained += int(not should_measure)
        result, row = inspect_one(points, feature)
        got_measurement = row["status"] == "MEASURED_CONDITIONAL"
        measured += int(got_measurement)
        if got_measurement != should_measure:
            failures.append({"case": case, "expected_measured": should_measure,
                             "status": row["status"], "reasons": row["reasons"],
                             "analytic_eligibility": eligibility})
        if not got_measurement:
            for reason in row["reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
            if row["measurements"] is not None or result["measured_conditional_feature_count"] != 0:
                failures.append({"case": case, "failure": "unavailable retained measurements"})
            continue
        fitted = row["measurements"]
        radius_error = fitted["measured_radius_mm"] - truth["radius_mm"]
        sweep_error = fitted["measured_sweep_deg"] - truth["sweep_deg"]
        axis_line_error = float(np.linalg.norm(np.cross(
            np.asarray(fitted["measured_axis_point"]) - truth["axis_point"], truth["axis"],
        )))
        max_radius_error = max(max_radius_error, abs(radius_error))
        max_sweep_error = max(max_sweep_error, abs(sweep_error))
        max_axis_line_error = max(max_axis_line_error, axis_line_error)
        checks = {
            "radius": abs(radius_error) <= 2e-8 * radius,
            "sweep": abs(sweep_error) <= 1e-8,
            "signed_radius_deviation": abs(fitted["radius_deviation_mm"]
                                           - (truth["radius_mm"] - radius)) <= 2e-8 * radius,
            "signed_sweep_deviation": abs(fitted["sweep_deviation_deg"]
                                          - (truth["sweep_deg"] - sweep)) <= 1e-8,
            "opening_convention": abs(fitted["measured_opening_deg"]
                                      - (180.0 - abs(truth["sweep_deg"]))) <= 1e-8,
            "axis_direction": np.allclose(fitted["measured_axis_direction"], truth["axis"],
                                          atol=2e-10, rtol=0),
            "axis_line": axis_line_error <= 5e-8 * radius,
            "sampled_extent_only": np.allclose(
                fitted["sampled_axial_interval_mm"], truth["sampled_axial_interval_mm"],
                atol=5e-8 * radius, rtol=0),
            "conditional_count": result["measured_conditional_feature_count"] == 1,
        }
        for name, passed in checks.items():
            if not passed:
                failures.append({"case": case, "failure": name,
                                 "radius_error_mm": radius_error,
                                 "sweep_error_deg": sweep_error,
                                 "axis_line_error_mm": axis_line_error})
        if should_measure:
            # Baseline only copies the nominal; neither side sees the true radius.
            baseline_error = radius - truth["radius_mm"]
            baseline_squared += baseline_error**2
            challenger_squared += radius_error**2
            baseline_relative_squared += (baseline_error / radius)**2
            challenger_relative_squared += (radius_error / radius)**2
            scored += 1
    report = {
        "scope": "constructed single-inner-face offline measurement; no count/F1",
        "baseline": "nominal-radius-copy ablation, NOT incumbent whole engine",
        "frozen_cases": len(EXACT_CASES), "expected_measured_cases": expected_measured,
        "expected_abstention_cases": expected_abstained, "measured_cases": measured,
        "same_case_scored_pairs": scored, "abstention_reasons": reasons,
        "baseline_radius_rmse_mm": math.sqrt(baseline_squared / scored) if scored else None,
        "challenger_radius_rmse_mm": math.sqrt(challenger_squared / scored) if scored else None,
        "baseline_radius_normalized_rmse": math.sqrt(baseline_relative_squared / scored) if scored else None,
        "challenger_radius_normalized_rmse": math.sqrt(challenger_relative_squared / scored) if scored else None,
        "max_radius_error_mm": max_radius_error, "max_sweep_error_deg": max_sweep_error,
        "max_axis_line_error_mm": max_axis_line_error, "failures": failures,
        "registration_and_feature_identity": "stipulated; not measured by this test",
        "physical_evaluation": False, "production_use_allowed": False,
    }
    return report


def test_frozen_324_case_matrix_and_paired_nominal_copy_radius_ablation():
    assert len(EXACT_CASES) == 324
    report = run_frozen_paired_radius_ablation()
    print("CAD_GUIDED_CONSTRUCTED_ABLATION " + json.dumps(report, sort_keys=True))
    assert not report["failures"], report
    assert report["same_case_scored_pairs"] == report["expected_measured_cases"]
    assert report["expected_abstention_cases"] > 0
    assert report["baseline_radius_normalized_rmse"] > 0.05
    assert report["challenger_radius_normalized_rmse"] < 2e-8


@pytest.mark.parametrize("radius,sweep,noise_fraction", NOISE_CASES)
def test_bounded_deterministic_noise_positions(radius, sweep, noise_fraction):
    feature = declared_feature(radius, sweep, "oblique")
    points, truth = constructed_scan(
        radius, sweep, radius_factor=1.07, sweep_magnitude_change=2.0,
        pose="oblique", noise_fraction=noise_fraction,
    )
    _, row = inspect_one(points, feature)
    assert row["status"] == "MEASURED_CONDITIONAL", row
    fitted = row["measurements"]
    assert abs(fitted["measured_radius_mm"] - truth["radius_mm"]) < 0.005 * radius
    assert abs(fitted["measured_sweep_deg"] - truth["sweep_deg"]) < 0.1
    # These are construction error limits, not measurement uncertainty intervals.
    assert fitted["circle_max_residual_mm"] < 0.01 * radius


@pytest.mark.parametrize("radius", RADII_MM)
def test_fitted_radius_changes_without_copying_nominal_or_clipping_at_25(radius):
    feature = declared_feature(radius)
    fitted_radii = []
    for factor in (0.9, 1.1):
        points, truth = constructed_scan(radius, radius_factor=factor)
        _, row = inspect_one(points, feature)
        assert row["status"] == "MEASURED_CONDITIONAL", row
        fitted_radii.append(row["measurements"]["measured_radius_mm"])
        assert fitted_radii[-1] == pytest.approx(truth["radius_mm"], abs=2e-8 * radius)
        assert abs(fitted_radii[-1] - feature.nominal_radius_mm) > 0.09 * radius
    assert fitted_radii[1] - fitted_radii[0] == pytest.approx(0.2 * radius, abs=4e-8 * radius)
    if radius == 30.0:
        assert min(fitted_radii) > 25.0


@pytest.mark.parametrize("kind", ("no_arc", "short_arc", "no_flanges", "short_axial"))
def test_partial_or_missing_support_never_copies_cad_as_a_measurement(kind):
    kwargs = {
        "no_arc": {"include_arc": False},
        "short_arc": {"arc_angles_deg": np.linspace(43.0, 47.0, 21)},
        "no_flanges": {"include_flanges": False},
        "short_axial": {"axial_fraction": (0.4, 0.6)},
    }[kind]
    points, _ = constructed_scan(**kwargs)
    result, row = inspect_one(points, declared_feature())
    assert row["status"] == "UNAVAILABLE", row
    assert row["measurements"] is None
    assert result["measured_conditional_feature_count"] == 0


def test_three_angular_columns_admit_a_free_offset_flat_witness():
    points, _ = constructed_scan(1.0, 90.0, arc_angles_deg=(34.5, 45.0, 55.5))
    best_flat_error = (1.0 - math.cos(math.radians(10.5))) / 2.0
    assert 0.008 < best_flat_error < 0.009 < 0.01
    _, row = inspect_one(points, declared_feature(1.0))
    assert row["status"] == "UNAVAILABLE"
    assert row["reasons"] == ["ARC_COMPATIBLE_WITH_FLAT"]


def test_mixed_radii_are_not_reported_as_one_bend_radius():
    points, _ = constructed_scan(arc_radius_factors=(0.9, 1.1))
    _, row = inspect_one(points, declared_feature())
    assert row["status"] == "UNAVAILABLE", row
    assert row["measurements"] is None
    assert row["reasons"] == ["ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL"]


@pytest.mark.parametrize("registration", ("UNTRUSTED", "AMBIGUOUS", None, ["DECLARED_ALIGNED"],
                                         np.array(["DECLARED_ALIGNED", "AMBIGUOUS"])))
def test_ambiguous_or_undeclared_registration_abstains(registration):
    points, _ = constructed_scan()
    result, row = inspect_one(points, declared_feature(), registration_status=registration)
    assert result["status"] == "UNAVAILABLE"
    assert row["measurements"] is None
    assert row["reasons"] == ["REGISTRATION_NOT_DECLARED_ALIGNED"]


def test_a_far_declared_feature_cannot_acquire_support():
    points, _ = constructed_scan()
    feature = replace(declared_feature(), origin=(1000.0, -1000.0, 500.0))
    _, row = inspect_one(points, feature)
    assert row["status"] == "UNAVAILABLE"
    assert row["reasons"] == ["INSUFFICIENT_FLANGE_SUPPORT"]


def test_duplicate_id_is_invalid_and_shared_support_clears_both_rows():
    points, _ = constructed_scan()
    first = declared_feature()
    with pytest.raises(ValueError, match="duplicate feature identity"):
        measurement.inspect_cad_guided_scan(points, [first, first], registration_status="DECLARED_ALIGNED")
    second = replace(first, feature_id="invented:1")
    result = measurement.inspect_cad_guided_scan(
        points, [first, second], registration_status="DECLARED_ALIGNED",
    )
    assert result["measured_conditional_feature_count"] == 0
    assert result["observed_bend_count"] is None
    for row in result["features"]:
        assert row["status"] == "UNAVAILABLE"
        assert row["measurements"] is None
        assert row["reasons"] == ["AMBIGUOUS_SHARED_FEATURE_SUPPORT"]


def test_nonoverlapping_features_keep_independent_measurements_and_unknown_count():
    first = declared_feature()
    shift = np.array([100.0, 200.0, 300.0])
    second = replace(first, feature_id="invented:1", origin=tuple(shift))
    points, _ = constructed_scan()
    result = measurement.inspect_cad_guided_scan(
        np.vstack((points, points + shift)), [first, second], registration_status="DECLARED_ALIGNED",
    )
    assert result["measured_conditional_feature_count"] == 2
    assert result["observed_bend_count"] is None
    assert all(row["status"] == "MEASURED_CONDITIONAL" for row in result["features"])
    assert all(row["conformance"] == "INDETERMINATE" for row in result["features"])


def test_late_nonfinite_input_clears_the_whole_feature_set():
    points, _ = constructed_scan()
    points[-1, -1] = np.nan
    first = declared_feature()
    second = replace(first, feature_id="invented:1")
    result = measurement.inspect_cad_guided_scan(
        points, [first, second], registration_status="DECLARED_ALIGNED",
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["measured_conditional_feature_count"] == 0
    for row in result["features"]:
        assert row["measurements"] is None
        assert row["reasons"] == ["UNSUPPORTED_POINT_NUMERICS"]


def test_signed_integer_minimum_cannot_wrap_past_numerical_guard():
    points, _ = constructed_scan()
    points = points.astype(np.int64)
    points[-1, -1] = np.iinfo(np.int64).min
    result, row = inspect_one(points, declared_feature())
    assert result["status"] == "UNAVAILABLE"
    assert row["measurements"] is None
    assert row["reasons"] == ["UNSUPPORTED_POINT_NUMERICS"]


def test_nonempty_200000_raw_rows_and_200001_whole_input_rejection():
    points, _ = constructed_scan()
    # Exactly one full-size fixture, outside both parameter matrices. Replication
    # exercises the admission/all-row path, not independent geometric evidence.
    points = np.tile(points, (math.ceil(200_000 / len(points)), 1))[:200_000].copy()
    assert len(points) == 200_000
    result, row = inspect_one(points, declared_feature())
    assert row["status"] == "MEASURED_CONDITIONAL", row
    assert result["measured_conditional_feature_count"] == 1
    too_many = np.vstack((points, points[-1]))
    result, row = inspect_one(too_many, declared_feature())
    assert result["status"] == "UNAVAILABLE"
    assert row["measurements"] is None
    assert row["reasons"] == ["POINT_BUDGET_OR_EMPTY_INPUT"]


def test_short_arc_jacobian_condition_abstains_even_with_exact_circle_data():
    feature = replace(declared_feature(1.0), max_fit_residual_mm=1e-6)
    angles = np.linspace(-1.0, 1.0, 81) * math.pi / 180.0
    cross_section = np.column_stack((np.cos(angles), np.sin(angles)))
    with pytest.raises(measurement._Unavailable, match="ARC_ILL_CONDITIONED"):
        measurement._circle(cross_section, feature)


@pytest.mark.parametrize("radius", (0.3, 1.0, 7.0))
def test_circle_geometric_jacobian_matches_central_differences(radius):
    angles = np.linspace(-0.9, 1.7, 37)
    points = np.column_stack((np.cos(angles), np.sin(angles))) * (radius + 0.4)
    points += np.array([0.21, -0.73])
    parameters = np.array([0.13, -0.62, radius])
    residual, jacobian = measurement.circle_residual_jacobian(parameters, points)
    assert np.allclose(residual, np.linalg.norm(points - parameters[:2], axis=1) - radius)
    numeric = np.empty_like(jacobian)
    for column in range(3):
        step = 1e-6 * max(1.0, abs(parameters[column]))
        perturbation = np.eye(3)[column] * step
        plus = measurement.circle_residual_jacobian(parameters + perturbation, points)[0]
        minus = measurement.circle_residual_jacobian(parameters - perturbation, points)[0]
        numeric[:, column] = (plus - minus) / (2.0 * step)
    assert np.allclose(jacobian, numeric, atol=2e-9, rtol=2e-8)
    assert np.array_equal(jacobian[:, 2], -np.ones(len(points)))


def test_plane_intersection_is_not_substituted_for_the_measured_circle_axis():
    # For this90° inner face, tangent planes intersect at(R,R,z); the cylinder
    # axis is(0,0,z), separated by sqrt(2)R. Both statements derive from fixture
    # equations, not the fitter or the CAD declaration's expected radius.
    radius = 5.0
    shift = (0.05, -0.04, 0.0)
    points, truth = constructed_scan(radius, center_shift=shift)
    _, row = inspect_one(points, declared_feature(radius))
    assert row["status"] == "MEASURED_CONDITIONAL", row
    fitted_center = np.asarray(row["measurements"]["measured_axis_point"])
    sharp_line_point = truth["axis_point"] + np.array([radius, radius, 0.0])
    assert np.linalg.norm(fitted_center[:2] - truth["axis_point"][:2]) < 1e-7
    assert np.linalg.norm(fitted_center[:2] - sharp_line_point[:2]) == pytest.approx(
        math.sqrt(2.0) * radius, abs=1e-7,
    )


def test_readonly_caller_positions_and_frozen_feature_are_unchanged():
    points, _ = constructed_scan(5.0, -90.0, radius_factor=1.1, pose="oblique")
    before = points.copy()
    points.setflags(write=False)
    feature = declared_feature(5.0, -90.0, "oblique")
    feature_before = repr(feature)
    _, row = inspect_one(points, feature)
    assert row["status"] == "MEASURED_CONDITIONAL", row
    assert np.array_equal(points, before)
    assert points.flags.writeable is False
    assert repr(feature) == feature_before


def test_single_axial_station_cannot_underflow_span_policy_into_a_measurement():
    points, _ = constructed_scan(.01)
    points[:81 * 9, 2] = .03  # Arc alone is one station; both flange fits stay 3-D.
    feature = replace(declared_feature(.01), min_axial_span_fraction=np.nextafter(0., 1.))
    assert feature.min_axial_span_fraction * (feature.span_mm[1] - feature.span_mm[0]) == 0.
    _, row = inspect_one(points, feature)
    assert row["status"] == "UNAVAILABLE"
    assert row["measurements"] is None
    assert row["reasons"] == ["INSUFFICIENT_DISTINCT_AXIAL_STATIONS"]
