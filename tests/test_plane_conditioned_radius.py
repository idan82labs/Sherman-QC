"""Independent synthetic bridge oracles; no runtime/native/physical inputs."""
import inspect
import json
import math

import numpy as np
from numpy.testing import assert_allclose as numpy_assert_allclose, assert_array_equal
import pytest

from research.geometry import plane_conditioned_radius as bridge_module
from research.geometry.axis_conditioned_radius import analyze_axis_conditioned_radius
from research.geometry.plane_conditioned_radius import analyze_plane_conditioned_radius


def assert_allclose(actual, expected, *, atol=1e-11):
    numpy_assert_allclose(actual, expected, rtol=0, atol=atol)


def fixture():
    first = np.array([[x, y, 5] for x in (-2, 2) for y in (-2, 2)], float)
    second = np.array([[5, y, z] for z in (-2, 2) for y in (-2, 2)], float)
    # Four integer-coordinate section points satisfy x*x+z*z=25.
    region = np.array([[x, y, z] for x, z in ((0, 5), (3, 4), (4, 3), (5, 0))
                       for y in (-1, 0, 1)], float)
    return first, second, region


def analyze(*regions, **changes):
    arguments = dict(unit="mm", position_error_bound=1e-6, max_axis_angle_bound=0.1,
                     max_axial_gap=1.1, max_radius_width=0.1)
    arguments.update(changes)
    return analyze_plane_conditioned_radius(*(regions or fixture()), **arguments)


def projector(axis):
    return np.outer(axis, axis)


def angle_between_lines(first, second):
    return math.atan2(np.linalg.norm(np.cross(first, second)), abs(float(np.dot(first, second))))


def rotation_from_quaternion(q):
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def rotation_between_lines(true_axis, estimated_axis):
    true_axis = np.asarray(true_axis)
    estimated_axis = np.asarray(estimated_axis)
    if true_axis @ estimated_axis < 0:
        true_axis = -true_axis
    vector = np.cross(true_axis, estimated_axis)
    x, y, z = vector
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + skew + skew @ skew/(1+float(true_axis @ estimated_axis))
    return rotation, true_axis


def assert_encloses(result, radius, *, atol=1e-10):
    lower, upper = result["radius_interval"]
    assert lower <= radius + atol
    if upper is not None:
        assert upper >= radius - atol


def assert_scope(result):
    assert result["schema_version"] == 1
    assert result["kind"] == "PLANE_CONDITIONED_RADIUS_V1"
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["status"] in {"ABSTAINED", "CONDITIONAL_RADIUS_HYPOTHESIS"}
    assert result["original_joint_compatibility"] == "UNESTABLISHED"
    assert result["radius_interval_scope"] == "NECESSARY_OUTER_CONSTRAINT_UNDER_ORIGINAL_JOINT_ASSUMPTIONS"
    assert result["hypothesis"]["axis_location"] is None
    assert {
        "SUPPLIED_REGIONS_NOT_REGION_DISCOVERY", "COMMON_AXIS_RELATION_STIPULATED_NOT_VERIFIED",
        "SAME_POSITION_FITS_NOT_INDEPENDENT_EVIDENCE", "POSITION_ERROR_BOUND_UNCALIBRATED",
        "RADIUS_OUTER_CONSTRAINTS_NOT_EXACT_FEASIBLE_SET", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        "RELAXED_WITNESS_NOT_ORIGINAL_JOINT_MODEL_WITNESS",
        "SAMPLE_COMPATIBILITY_NOT_SURFACE_IDENTITY", "NO_CONTINUITY_OR_PHYSICAL_EXTENT_CLAIM",
        "NO_PHYSICAL_AXIS_LOCATION_OR_MARKER", "NO_FORMATION_THICKNESS_OR_CONFORMANCE_CLAIM",
        "RESOLUTION_POLICIES_NOT_PHYSICAL_QUALIFICATION",
    } <= set(result["limitations"])
    def check_nested(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"axis_point", "axis_endpoints"}
                if key.endswith("is_verified"):
                    assert item is False
                check_nested(item)
        elif isinstance(value, list):
            for item in value:
                check_nested(item)
    check_nested(result)
    if result["status"] == "ABSTAINED":
        assert result["reasons"]
    else:
        assert result["reasons"] == []
        assert result["axis_analysis"]["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
        assert result["relaxed_radius_diagnostic"]["status"] == "SAMPLE_CIRCLE_WITNESS"
    diagnostic = result["relaxed_radius_diagnostic"]
    if diagnostic is not None:
        assert diagnostic["scope"] == "ROTATED_ERROR_RELAXATION_ONLY_NOT_ORIGINAL_JOINT_MODEL_WITNESS"
        assert result["radius_interval"] == diagnostic["radius_interval"]
        for group in diagnostic["axial_sample_groups"]:
            assert group["scope"] == "OBSERVED_PROJECTIONS_ONLY"
            assert group["group_membership_is_verified_for_true_axis"] is False
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def assert_rotated_counterparts_bounded(result, observed, truth, true_axis, *, atol=1e-10):
    propagation = result["uncertainty_propagation"]
    if propagation is None:
        return
    pivot = np.asarray(propagation["pivot"])
    axis = np.asarray(result["hypothesis"]["axis"])
    rotation, aligned_true_axis = rotation_between_lines(true_axis, axis)
    assert_allclose(rotation.T @ rotation, np.eye(3))
    assert_allclose(np.linalg.det(rotation), 1)
    assert_allclose(rotation @ aligned_true_axis, axis)
    rotated = (truth-pivot) @ rotation.T + pivot
    assert np.max(np.linalg.norm(rotated-observed, axis=1)) <= (
        propagation["radius_input_error_bound"] + atol
    )
    actual_axial = (truth-pivot) @ aligned_true_axis
    observed_axial = (observed-pivot) @ axis
    assert np.max(np.abs(actual_axial-observed_axial)) <= (
        propagation["axial_sample_coordinate_error_bound"] + atol
    )


def test_observation_channel_needs_positions_not_axis_normals_radius_or_truth():
    parameters = inspect.signature(analyze_plane_conditioned_radius).parameters
    assert set(parameters) == {"first_neighborhood", "second_neighborhood", "cylinder_region",
                              "unit", "position_error_bound", "max_axis_angle_bound",
                              "max_axial_gap", "max_radius_width"}


@pytest.mark.parametrize("epsilon", [0, 1e-6, 0.001])
def test_hand_computed_planes_and_radius_five_remain_conditional(epsilon):
    result = analyze(position_error_bound=epsilon, max_radius_width=0.5)
    assert_scope(result)
    assert result["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    assert result["input_count"] == 20
    assert result["input_counts"] == {"first_neighborhood": 4, "second_neighborhood": 4, "cylinder_region": 12}
    assert_allclose(projector(result["hypothesis"]["axis"]), np.diag([0, 1, 0]))
    assert_encloses(result, 5)
    assert_allclose(result["relaxed_radius_diagnostic"]["circle_witness"]["radius"], 5)
    assert_rotated_counterparts_bounded(result, fixture()[2], fixture()[2], [0, 1, 0])


def test_rotation_propagation_budget_and_numerical_allowances_are_separate():
    result = analyze(position_error_bound=0.001, max_radius_width=0.5)
    propagation = result["uncertainty_propagation"]
    region = fixture()[2]
    pivot = region.mean(axis=0)
    footprint = float(np.max(np.linalg.norm(region-pivot, axis=1)))
    assert_allclose(propagation["pivot"], pivot)
    assert_allclose(propagation["observed_footprint_radius"], footprint)
    assert_allclose(propagation["true_counterpart_footprint_bound"], footprint+0.001)
    chord = 2*math.sin(result["hypothesis"]["axis_angle_bound_radians"]/2)
    assert_allclose(propagation["rotation_chord_bound"], chord)
    assert_allclose(propagation["rotation_displacement_bound"], chord*(footprint+0.001))
    assert_allclose(propagation["propagated_position_error_bound"], 0.001+chord*(footprint+0.001))
    assert propagation["radius_input_error_bound"] == (
        propagation["propagated_position_error_bound"] + propagation["bridge_numerical_allowance"]
    )
    assert propagation["radius_effective_error_bound"] == (
        propagation["radius_input_error_bound"] + propagation["radius_numerical_allowance"]
    )
    assert propagation["axial_sample_coordinate_error_bound"] == propagation["radius_effective_error_bound"]
    assert_scope(result)


def tilted_radius_two_counterexample():
    # Independently supplied during mathematical review: the wrong-axis
    # projection is EXACTLY a radius-four circle, while x*x+z*z is exactly four.
    theta = 0.01
    rotation = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)],
                         [0, math.sin(theta), math.cos(theta)]])
    first = np.array([[x, y, 0] for x in (-0.001, 0.001) for y in (-0.001, 0.001)])
    second = np.array([[0, y, z] for z in (-0.001, 0.001) for y in (-0.001, 0.001)])
    observed_first, observed_second = first @ rotation.T, second @ rotation.T
    epsilon = max(np.max(np.linalg.norm(observed_first-first, axis=1)),
                  np.max(np.linalg.norm(observed_second-second, axis=1)))
    phi = np.arange(1, 8)*0.2
    x, z = 2*np.cos(phi), 2*np.sin(phi)
    y = (math.cos(theta)*z-np.sqrt(16-x*x))/math.sin(theta)
    region = np.column_stack((x, y, z))
    return observed_first, observed_second, region, float(epsilon), rotation[:, 1]


def test_exact_radius_two_can_look_like_four_when_axis_error_is_ignored():
    first, second, region, epsilon, expected_axis = tilted_radius_two_counterexample()
    result = analyze(first, second, region, position_error_bound=epsilon,
                     max_axis_angle_bound=0.05, max_axial_gap=1000, max_radius_width=0.01)
    assert_allclose(region[:, 0]**2+region[:, 2]**2, np.full(7, 4))
    # Exact independent projected-circle equation, no normals/labels passed to bridge.
    transverse = expected_axis[1]*region[:, 2]-expected_axis[2]*region[:, 1]
    assert_allclose(region[:, 0]**2+transverse**2, np.full(7, 16))
    assert_allclose(projector(result["hypothesis"]["axis"]), projector(expected_axis))
    naive = analyze_axis_conditioned_radius(
        region, axis=result["hypothesis"]["axis"], unit="mm", position_error_bound=epsilon,
        max_axial_gap=1000, max_radius_width=0.01,
    )
    assert naive["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert_allclose(naive["circle_witness"]["radius"], 4)
    assert naive["circle_witness"]["max_radial_residual"] < 1e-11
    assert naive["radius_interval"][0] > 3.99
    assert naive["radius_interval"][1] < 4.01
    assert naive["radius_interval"][0] > 2  # Its narrow interval excludes TRUE R.
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert_encloses(result, 2)
    assert result["radius_interval"] == [0.0, None]
    assert "RELAXED_RADIUS_PARALLEL_PLANE_COMPATIBLE" in result["reasons"]
    assert_rotated_counterparts_bounded(result, region, region, [0, 1, 0])


def test_broad_axis_cone_can_force_abstention_even_when_fitted_axis_is_exact():
    first, second, region = fixture()
    first = np.array([[x, y, 5] for x in (-0.1, 0.1) for y in (-0.1, 0.1)])
    second = np.array([[5, y, z] for z in (-0.1, 0.1) for y in (-0.1, 0.1)])
    result = analyze(first, second, region, position_error_bound=0.01,
                     max_axis_angle_bound=0.3, max_radius_width=0.1)
    assert_scope(result)
    assert_allclose(projector(result["hypothesis"]["axis"]), np.diag([0, 1, 0]))
    assert result["axis_analysis"]["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert result["status"] == "ABSTAINED"
    assert result["uncertainty_propagation"]["radius_input_error_bound"] > 0.5
    assert_encloses(result, 5)


def test_near_tight_plane_normal_cone_still_propagates_true_tilt_to_radius():
    # Independent normal-bound boundary: s2=2 and sqrt(N)=2, while projection
    # onto the true tilted plane moves every first-neighborhood point by sinθ.
    theta = 0.1
    epsilon = math.sin(theta)
    first = np.array([[x, y, 0] for x in (-2, 2) for y in (-1, 1)], float)
    true_normal = np.array([0, math.sin(theta), math.cos(theta)])
    true_first = first - (first @ true_normal)[:, None]*true_normal
    assert_allclose(np.linalg.norm(first-true_first, axis=1), np.full(4, epsilon))
    assert_allclose(true_first @ true_normal, np.zeros(4))
    second = np.array([[0, y, z] for y in (-100, 100) for z in (-100, 100)], float)
    rotation = np.array([[1, 0, 0], [0, math.cos(theta), math.sin(theta)],
                         [0, -math.sin(theta), math.cos(theta)]])
    region = fixture()[2] @ rotation.T  # True R=5 cylinder; cylinder positional noise is zero.
    result = analyze(first, second, region, position_error_bound=epsilon,
                     max_axis_angle_bound=0.2, max_axial_gap=5, max_radius_width=0.1)
    first_cone = result["axis_analysis"]["plane_neighborhoods"][0]["normal_angle_bound_radians"]
    assert theta <= first_cone <= theta+1e-11
    assert result["hypothesis"]["axis_angle_bound_radians"] >= theta
    assert_encloses(result, 5)
    assert_rotated_counterparts_bounded(result, region, region, rotation[:, 1])
    assert_scope(result)


@pytest.mark.parametrize("seed", [9, 105, 720])
def test_se3_invariance_for_radius_and_unoriented_axis(seed):
    rng = np.random.default_rng(seed)
    rotation = rotation_from_quaternion(rng.normal(size=4))
    translation = rng.uniform(-10, 10, size=3)
    original = analyze()
    moved_regions = tuple(region @ rotation.T+translation for region in fixture())
    moved = analyze(*moved_regions)
    assert_scope(moved)
    assert moved["status"] == original["status"]
    assert_allclose(projector(moved["hypothesis"]["axis"]),
                    rotation @ projector(original["hypothesis"]["axis"]) @ rotation.T)
    assert_allclose(moved["uncertainty_propagation"]["pivot"],
                    rotation @ original["uncertainty_propagation"]["pivot"]+translation)
    assert_allclose(moved["uncertainty_propagation"]["observed_footprint_radius"],
                    original["uncertainty_propagation"]["observed_footprint_radius"])
    assert_allclose(moved["radius_interval"], original["radius_interval"], atol=1e-9)
    assert_encloses(moved, 5)
    assert_rotated_counterparts_bounded(moved, moved_regions[2], moved_regions[2], rotation[:, 1])


def test_unit_scaling_preserves_dimensionless_geometry_and_scales_radius_bounds():
    original = analyze()
    result = analyze(*(region*0.001 for region in fixture()), unit="m", position_error_bound=1e-9,
                     max_axial_gap=0.0011, max_radius_width=0.0001)
    assert_scope(result)
    assert result["status"] == original["status"]
    assert_allclose(result["radius_interval"], np.asarray(original["radius_interval"])*0.001, atol=1e-12)
    lengths = {"pivot", "observed_footprint_radius", "true_counterpart_footprint_bound",
               "rotation_displacement_bound", "propagated_position_error_bound", "bridge_numerical_allowance",
               "radius_input_error_bound", "radius_numerical_allowance", "radius_effective_error_bound",
               "axial_sample_coordinate_error_bound"}
    for key in lengths:
        assert_allclose(result["uncertainty_propagation"][key],
                        np.asarray(original["uncertainty_propagation"][key])*0.001, atol=1e-12)
    for key in ("axis_angle_bound_radians", "rotation_chord_bound"):
        assert_allclose(result["uncertainty_propagation"][key], original["uncertainty_propagation"][key])


def test_neighborhood_reversal_permutation_and_duplicate_rows_preserve_bounds():
    first, second, region = fixture()
    original = analyze(first, second, region)
    result = analyze(second[::-1], first[::-1], np.repeat(region[::-1], 2, axis=0))
    assert_scope(result)
    assert result["input_count"] == 32
    assert_allclose(result["radius_interval"], original["radius_interval"])
    assert_allclose(result["uncertainty_propagation"]["pivot"], original["uncertainty_propagation"]["pivot"])
    assert_allclose(projector(result["hypothesis"]["axis"]), projector(original["hypothesis"]["axis"]))


def test_short_arc_and_flat_alternative_force_abstention_after_propagation():
    first, second, _ = fixture()
    x = np.linspace(-1, 1, 5)
    radius = 1000
    z = x*x/(radius+np.sqrt(radius*radius-x*x))
    region = np.array([[px, y, pz] for px, pz in zip(x, z) for y in (-1, 0, 1)])
    result = analyze(first, second, region, position_error_bound=0.0003, max_radius_width=1e9)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["relaxed_radius_diagnostic"]["parallel_plane_witness"]["compatible_with_error_bound"] is True
    assert_encloses(result, radius)


def test_explicit_observed_axial_gap_is_never_filled_or_called_physical_topology():
    first, second, _ = fixture()
    region = np.array([[x, y, z] for x, z in ((0, 5), (3, 4), (4, 3), (5, 0))
                       for y in (-8, -3, 3, 8)], float)
    result = analyze(first, second, region, max_axial_gap=5)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert "RELAXED_RADIUS_AXIAL_SAMPLE_GAP" in result["reasons"]
    groups = result["relaxed_radius_diagnostic"]["axial_sample_groups"]
    assert len(groups) == 2
    assert_allclose([[group["sample_min"], group["sample_max"]] for group in groups], [[-8, -3], [3, 8]])
    assert all(group["physical_support_is_verified"] is False for group in groups)


def test_radius_and_gap_policies_remain_explicit_resolution_not_qualification():
    result = analyze(max_radius_width=1e-10, max_axial_gap=1.25)
    assert_scope(result)
    assert result["policy"] == {"max_axis_angle_bound_radians": 0.1, "max_radius_width": 1e-10, "max_axial_gap": 1.25}
    assert result["status"] == "ABSTAINED"
    assert "RELAXED_RADIUS_RADIUS_PRECISION_NOT_ESTABLISHED" in result["reasons"]


@pytest.mark.parametrize("region_index", [0, 1, 2])
@pytest.mark.parametrize("missing", [None, [], np.empty((0, 3))])
def test_missing_evidence_abstains_without_inventing_a_radius(region_index, missing):
    regions = list(fixture())
    regions[region_index] = missing
    result = analyze(*regions)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["radius_interval"] == [0, None]
    assert result["relaxed_radius_diagnostic"] is None


def test_unresolved_parallel_axis_never_calls_radius_analysis(monkeypatch):
    first, _, region = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("unresolved axis was passed to radius fitting")
    monkeypatch.setattr(bridge_module, "analyze_axis_conditioned_radius", forbidden)
    result = analyze(first, first+[0, 0, 3], region)
    assert_scope(result)
    assert "AXIS_PARALLEL_ALTERNATIVE_NOT_EXCLUDED" in result["reasons"]


def test_shared_cap_covers_all_three_regions_before_conversion_or_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input cap was checked too late")
    monkeypatch.setattr(bridge_module.np, "asarray", forbidden)
    monkeypatch.setattr(bridge_module, "analyze_position_axis_hypothesis", forbidden)
    with pytest.raises(ValueError, match="256 total"):
        analyze([[0, 0, 0]]*85, [[0, 0, 0]]*86, [[0, 0, 0]]*86)


def test_exact_shared_cap_is_supported():
    first, second, _ = fixture()
    region = np.array([[5*np.sin(phi), y, 5*np.cos(phi)] for phi in np.linspace(0, math.pi/2, 31)
                       for y in np.linspace(-1, 1, 8)])
    result = analyze(first, second, region)
    assert_scope(result)
    assert result["input_count"] == 256
    assert result["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    assert_encloses(result, 5)


def test_input_arrays_are_unchanged():
    regions = fixture()
    before = tuple(region.copy() for region in regions)
    analyze(*regions)
    for expected, actual in zip(before, regions):
        assert_array_equal(expected, actual)


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("unit", None), ("position_error_bound", -1),
    ("position_error_bound", True), ("position_error_bound", float("nan")),
    ("position_error_bound", float("inf")), ("position_error_bound", "0.1"),
    ("max_axis_angle_bound", 0), ("max_axis_angle_bound", math.pi/2),
    ("max_axis_angle_bound", float("inf")), ("max_axial_gap", 0), ("max_axial_gap", True),
    ("max_axial_gap", float("nan")), ("max_radius_width", -1), ("max_radius_width", "0.1"),
])
def test_invalid_policy_is_rejected(field, value):
    with pytest.raises(ValueError):
        analyze(**{field: value})


@pytest.mark.parametrize("positions", [1, "positions", iter([[0, 0, 0]]), [[0, 0]],
    [[True, 0, 0]], [["1", 0, 0]], [[float("nan"), 0, 0]], [[float("inf"), 0, 0]],
    np.zeros((3, 4)), np.zeros((3, 3), dtype=complex), np.zeros((3, 3), dtype=bool), np.array(1)])
def test_invalid_regions_are_rejected(positions):
    first, second, _ = fixture()
    with pytest.raises(ValueError):
        analyze(first, second, positions)


def test_unrepresentable_footprint_abstains_with_finite_json():
    first, second, _ = fixture()
    region = [[-1.7e308, 0, 0], [1.7e308, 0, 0], [0, 1, 0]]
    result = analyze(first, second, region)
    assert_scope(result)
    assert result["reasons"] == ["BRIDGE_NUMERICAL_RANGE_UNSUPPORTED"]


def run_synthetic_bridge_sweep(*, expanded=False):
    """Fixed seed; 80 default cases or explicit bounded 1,000-case stress sweep."""
    rng = np.random.default_rng(20260908)
    count = 1000 if expanded else 80
    radii = 10**rng.uniform(-3, 3, size=count)
    epsilons = radii*10**rng.uniform(-8, -2, size=count)
    turns = 10**rng.uniform(-3, math.log10(math.pi/2), size=count)
    rotations = np.stack([rotation_from_quaternion(q) for q in rng.normal(size=(count, 4))])
    shifts = rng.uniform(-5, 5, size=(count, 3))*radii[:, None]
    first = np.array([[x, y, 1] for x in (-1, 1) for y in (-1, 1)], float)[None, :, :]
    first = np.broadcast_to(first, (count, 4, 3)).copy()
    second = np.zeros_like(first)
    corners = np.array([[u, v] for u in (-1, 1) for v in (-1, 1)])
    second[:, :, 0] = np.sin(turns)[:, None]+np.cos(turns)[:, None]*corners[:, 0]
    second[:, :, 1] = corners[:, 1]
    second[:, :, 2] = np.cos(turns)[:, None]-np.sin(turns)[:, None]*corners[:, 0]
    fractions = np.array([0, 0.5, 1])
    region = np.zeros((count, 9, 3))
    region[:, :, 0] = np.repeat(np.sin(turns[:, None]*fractions), 3, axis=1)
    region[:, :, 2] = np.repeat(np.cos(turns[:, None]*fractions), 3, axis=1)
    region[:, :, 1] = np.tile([-1, 0, 1], 3)
    truths = [np.einsum("cij,cnj->cni", rotations, positions)*radii[:, None, None]+shifts[:, None, :]
              for positions in (first, second, region)]
    observations = []
    for truth in truths:
        noise = rng.normal(size=truth.shape)
        noise /= np.linalg.norm(noise, axis=-1, keepdims=True)
        noise *= rng.uniform(0, 0.9, size=truth.shape[:-1]+(1,))*epsilons[:, None, None]
        observed = truth+noise
        assert np.all(np.linalg.norm(observed-truth, axis=-1) <= epsilons[:, None])
        observations.append(observed)
    counts = {"cases": count, "positions_per_case": 17, "CONDITIONAL_RADIUS_HYPOTHESIS": 0,
              "ABSTAINED": 0, "propagation_available": 0, "finite_radius_upper_bounds": 0}
    for index in range(count):
        result = analyze(*(positions[index] for positions in observations),
                         position_error_bound=float(epsilons[index]), max_axis_angle_bound=0.35,
                         max_axial_gap=float(5*radii[index]), max_radius_width=float(0.1*radii[index]))
        assert_scope(result)
        assert_encloses(result, radii[index], atol=1e-9*radii[index])
        if result["uncertainty_propagation"] is not None:
            counts["propagation_available"] += 1
            assert angle_between_lines(result["hypothesis"]["axis"], rotations[index, :, 1]) <= (
                result["hypothesis"]["axis_angle_bound_radians"]+1e-11
            )
            assert_rotated_counterparts_bounded(result, observations[2][index], truths[2][index],
                                                rotations[index, :, 1], atol=1e-9*radii[index])
        counts[result["status"]] += 1
        counts["finite_radius_upper_bounds"] += result["radius_interval"][1] is not None
    return counts


def test_fixed_seed_80_case_joint_error_radius_and_rotation_containment():
    counts = run_synthetic_bridge_sweep()
    assert counts["cases"] == 80
    assert counts["CONDITIONAL_RADIUS_HYPOTHESIS"] > 0
    assert counts["ABSTAINED"] > 0
    assert counts["propagation_available"] > 40
