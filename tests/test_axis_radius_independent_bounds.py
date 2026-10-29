"""Independent, bounded mathematical challenges for plane-to-radius inference.

Only explicitly constructed positions and caller policies enter the analyzers.
Construction normals, true radius, and true axis stay in the assertion channel.
There are no file readers, runtime/native-geometry imports, or physical fixtures.
"""

import math

import numpy as np
from numpy.testing import assert_allclose

from research.geometry.axis_conditioned_radius import analyze_axis_conditioned_radius
from research.geometry.plane_conditioned_radius import analyze_plane_conditioned_radius
from research.geometry.position_axis_hypothesis import analyze_position_axis_hypothesis


def _unoriented_angle(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return math.atan2(math.hypot(*np.cross(first, second)), abs(float(first @ second)))


def _tilted_axis_double_radius_example():
    """Fifteen observations admit true R=2, but the fitted direction gives R=4."""
    first = 0.001 * np.array([
        [x, y, 0.0] for x in (-1.0, 1.0) for y in (-1.0, 1.0)
    ])
    second = 0.001 * np.array([
        [0.0, y, z] for z in (-1.0, 1.0) for y in (-1.0, 1.0)
    ])
    theta = 0.01
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta), math.cos(theta)],
    ])
    observed_planes = (first @ rotation.T, second @ rotation.T)
    epsilon = max(
        float(np.max(np.linalg.norm(observed - true, axis=1)))
        for observed, true in zip(observed_planes, (first, second))
    )

    # The true cylinder has radius 2 about the line (-2, y, -2). Both true
    # planes x=0 and z=0 are tangent to it. Choose each axial sample so that
    # projection perpendicular to the coherently tilted plane-pair direction
    # lies on a radius-4 circle. The radius samples themselves have zero error.
    phi = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    radial_x, radial_z = 2 * np.cos(phi), 2 * np.sin(phi)
    x, z = radial_x - 2, radial_z - 2
    y = (math.cos(theta) * z + 2 - np.sqrt(16 - radial_x * radial_x)) / math.sin(theta)
    points = np.column_stack((x, y, z))
    assert_allclose(np.hypot(points[:, 0] + 2, points[:, 2] + 2), 2, rtol=0, atol=1e-14)
    assert sum(map(len, observed_planes)) + len(points) == 15
    return observed_planes, points, epsilon, theta, rotation


def test_correlated_projection_errors_nearly_attain_normal_and_axis_bounds():
    # The true normal is not estimated by a second fit. Orthogonal projection
    # explicitly constructs true coplanar counterparts, all exactly epsilon
    # from the observed rectangle in exact arithmetic. This reaches the normal
    # bound asin(sqrt(N)*epsilon/s2), unlike a typically loose random-noise case.
    theta = 0.001
    epsilon = math.sin(theta)
    normal = np.array([0.0, math.sin(theta), math.cos(theta)])
    first = np.array([[x, y, 0.0] for x in (-2.0, 2.0) for y in (-1.0, 1.0)])
    second = np.array([[0.0, y, z] for y in (-100.0, 100.0) for z in (-100.0, 100.0)])
    true_first = first - (first @ normal)[:, None] * normal
    assert_allclose(true_first @ normal, 0, rtol=0, atol=1e-15)
    assert_allclose(np.linalg.norm(true_first - first, axis=1), epsilon, rtol=0, atol=1e-18)

    result = analyze_position_axis_hypothesis(
        first, second, unit="mm", position_error_bound=epsilon, max_axis_angle_bound=0.01,
    )
    assert result["input_count"] == 8
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    first_plane = result["plane_neighborhoods"][0]
    assert_allclose(first_plane["singular_values"], [4, 2, 0], rtol=0, atol=1e-14)
    normal_error = _unoriented_angle(first_plane["plane_witness"]["normal"], normal)
    normal_bound = first_plane["normal_angle_bound_radians"]
    true_axis = np.cross(normal, [1.0, 0.0, 0.0])
    axis_error = _unoriented_angle(result["hypothesis"]["axis"], true_axis)
    axis_bound = result["hypothesis"]["axis_angle_bound_radians"]
    assert_allclose([normal_error, axis_error], theta, rtol=0, atol=1e-15)
    assert normal_error <= normal_bound
    assert axis_error <= axis_bound
    assert normal_error / normal_bound > 0.999999
    assert axis_error / axis_bound > 0.98
    assert result["hypothesis"]["axis_is_verified"] is False


def test_ignoring_fitted_axis_uncertainty_can_exclude_the_true_radius_by_factor_two():
    planes, points, epsilon, theta, _ = _tilted_axis_double_radius_example()
    direction = analyze_position_axis_hypothesis(
        *planes, unit="mm", position_error_bound=epsilon, max_axis_angle_bound=0.1,
    )
    hypothesis = direction["hypothesis"]
    assert direction["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    actual_axis_error = _unoriented_angle(hypothesis["axis"], [0.0, 1.0, 0.0])
    assert_allclose(actual_axis_error, theta, rtol=0, atol=1e-14)
    assert actual_axis_error <= hypothesis["axis_angle_bound_radians"]

    # This deliberate misuse is the negative control: the exact-axis routine
    # cannot cover unknown direction error without an explicit bridge.
    naive = analyze_axis_conditioned_radius(
        points, axis=hypothesis["axis"], unit="mm", position_error_bound=epsilon,
        max_axial_gap=1000, max_radius_width=0.01,
    )
    assert naive["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert naive["circle_witness"]["compatible_with_error_bound"] is True
    assert_allclose(naive["circle_witness"]["radius"], 4, rtol=0, atol=1e-10)
    lower, upper = naive["radius_interval"]
    assert lower > 3.9  # The true radius 2 has been falsely excluded by misuse.
    assert upper - lower < 0.01
    assert naive["hypothesis"]["axis_is_verified"] is False
    assert naive["production_disposition"] == "NOT_AUTHORIZED"


def test_common_rigid_rotation_relaxation_contains_the_true_radius():
    planes, points, epsilon, _, rotation = _tilted_axis_double_radius_example()
    direction = analyze_position_axis_hypothesis(
        *planes, unit="mm", position_error_bound=epsilon, max_axis_angle_bound=0.1,
    )
    hypothesis = direction["hypothesis"]
    pivot = points[0]
    footprint = max(math.hypot(*(point - pivot)) for point in points)
    delta = hypothesis["axis_angle_bound_radians"]
    inflated = epsilon + 2 * math.sin(delta / 2) * (footprint + epsilon)

    # One common rotation about one observed pivot aligns the true axis while
    # preserving its radius. Independent per-point rotations are not used.
    rotated = pivot + (points - pivot) @ rotation.T
    rotated_axis = rotation @ np.array([0.0, 1.0, 0.0])
    rotated_axis_point = pivot + rotation @ (np.array([-2.0, 0.0, -2.0]) - pivot)
    assert_allclose(
        np.linalg.norm(np.cross(rotated - rotated_axis_point, rotated_axis), axis=1),
        2, rtol=0, atol=1e-12,
    )
    assert np.max(np.linalg.norm(rotated - points, axis=1)) <= inflated
    assert inflated > 1000 * epsilon
    relaxed = analyze_axis_conditioned_radius(
        points, axis=hypothesis["axis"], unit="mm", position_error_bound=inflated,
        max_axial_gap=1000, max_radius_width=0.01,
    )
    lower, upper = relaxed["radius_interval"]
    assert lower <= 2
    assert upper is None or upper >= 2
    assert relaxed["status"] == "ABSTAINED"
    assert "RADIUS_UPPER_BOUND_UNRESOLVED" in relaxed["reasons"]
    # The relaxed witness is not a witness of the original plane/cylinder model.
    assert relaxed["production_disposition"] == "NOT_AUTHORIZED"


def test_plane_radius_bridge_retains_true_radius_and_exports_only_relaxed_evidence():
    planes, points, epsilon, _, rotation = _tilted_axis_double_radius_example()
    result = analyze_plane_conditioned_radius(
        *planes, points, unit="mm", position_error_bound=epsilon,
        max_axis_angle_bound=0.1, max_axial_gap=1000, max_radius_width=0.01,
    )
    assert result["input_count"] == 15
    assert result["status"] == "ABSTAINED"
    assert result["radius_interval"] == [0.0, None]
    assert result["original_joint_compatibility"] == "UNESTABLISHED"
    assert result["radius_interval_scope"] == "NECESSARY_OUTER_CONSTRAINT_UNDER_ORIGINAL_JOINT_ASSUMPTIONS"
    assert result["hypothesis"]["axis_location"] is None
    assert result["production_disposition"] == "NOT_AUTHORIZED"

    propagation = result["uncertainty_propagation"]
    pivot = np.asarray(propagation["pivot"])
    footprint = max(math.hypot(*(point - pivot)) for point in points)
    chord = 2 * math.sin(result["hypothesis"]["axis_angle_bound_radians"] / 2)
    expected_budget = epsilon + chord * (footprint + epsilon)
    assert_allclose(propagation["observed_footprint_radius"], footprint, rtol=0, atol=1e-12)
    assert_allclose(propagation["propagated_position_error_bound"], expected_budget, rtol=0, atol=1e-12)
    assert propagation["radius_input_error_bound"] >= expected_budget
    assert propagation["radius_effective_error_bound"] >= propagation["radius_input_error_bound"]
    rotated = pivot + (points - pivot) @ rotation.T
    assert np.max(np.linalg.norm(rotated - points, axis=1)) <= expected_budget
    true_axial = (points - pivot) @ np.array([0.0, 1.0, 0.0])
    observed_axial = (points - pivot) @ np.asarray(result["hypothesis"]["axis"])
    assert np.max(np.abs(true_axial - observed_axial)) <= propagation["axial_sample_coordinate_error_bound"]

    diagnostic = result["relaxed_radius_diagnostic"]
    assert diagnostic["scope"] == "ROTATED_ERROR_RELAXATION_ONLY_NOT_ORIGINAL_JOINT_MODEL_WITNESS"
    assert diagnostic["circle_witness"]["compatible_with_error_bound"] is True
    assert "RELAXED_RADIUS_PARALLEL_PLANE_COMPATIBLE" in result["reasons"]
    assert "RELAXED_RADIUS_RADIUS_UPPER_BOUND_UNRESOLVED" in result["reasons"]

    # Even an excellent circle in the relaxed model must not export physical
    # cylinder-axis geometry or turn any model/support verification flag true.
    def check_payload(value):
        if isinstance(value, dict):
            assert not {"axis_point", "axis_endpoints"}.intersection(value)
            for key, item in value.items():
                if key.endswith("is_verified"):
                    assert item is False
                check_payload(item)
        elif isinstance(value, list):
            for item in value:
                check_payload(item)

    check_payload(result)


def test_tilted_observed_axial_span_does_not_establish_true_finite_support():
    planes, _, epsilon, _, _ = _tilted_axis_double_radius_example()
    phi = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, math.pi / 2])
    # All true positions lie on ONE cross-section y=0. The fitted tilted axis
    # manufactures a positive observed axial span even though true span is zero.
    points = np.column_stack((5 * np.sin(phi), np.zeros(len(phi)), 5 * np.cos(phi)))
    result = analyze_plane_conditioned_radius(
        *planes, points, unit="mm", position_error_bound=epsilon,
        max_axis_angle_bound=0.1, max_axial_gap=1000, max_radius_width=1,
    )
    assert len(points) + sum(map(len, planes)) == 15
    assert np.ptp(points[:, 1]) == 0
    groups = result["relaxed_radius_diagnostic"]["axial_sample_groups"]
    assert len(groups) == 1
    assert groups[0]["sample_max"] > groups[0]["sample_min"]
    assert result["status"] == "ABSTAINED"
    assert "RELAXED_RADIUS_AXIAL_EXTENT_UNRESOLVED" in result["reasons"]
    assert groups[0]["scope"] == "OBSERVED_PROJECTIONS_ONLY"
    assert groups[0]["physical_support_is_verified"] is False
    assert groups[0]["group_membership_is_verified_for_true_axis"] is False
    assert result["hypothesis"]["physical_support_is_verified"] is False
    assert result["original_joint_compatibility"] == "UNESTABLISHED"
