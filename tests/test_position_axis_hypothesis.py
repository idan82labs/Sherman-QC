"""Synthetic position-only plane/axis oracles; no product or data imports."""
import inspect
import json
import math

import numpy as np
from numpy.testing import assert_allclose as numpy_assert_allclose, assert_array_equal
import pytest

from research.geometry import position_axis_hypothesis as axis_module
from research.geometry.position_axis_hypothesis import analyze_position_axis_hypothesis


def assert_allclose(actual, expected, *, atol=1e-11):
    numpy_assert_allclose(actual, expected, rtol=0, atol=atol)


def planes():
    # Four rectangle corners on z=2 and x=3; both long directions have span 4.
    first = np.array([[x, y, 2] for x in (-2, 2) for y in (-1, 1)], float)
    second = np.array([[3, y, z] for z in (-2, 2) for y in (-1, 1)], float)
    return first, second


def analyze(first=None, second=None, **changes):
    defaults = dict(unit="mm", position_error_bound=0.001, max_axis_angle_bound=0.1)
    defaults.update(changes)
    base_first, base_second = planes()
    return analyze_position_axis_hypothesis(
        base_first if first is None else first,
        base_second if second is None else second, **defaults,
    )


def unoriented_angle(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return math.atan2(np.linalg.norm(np.cross(first, second)), abs(float(first @ second)))


def projector(direction):
    vector = np.asarray(direction)
    return np.outer(vector, vector)


def assert_scope(result):
    assert result["schema_version"] == 1
    assert result["kind"] == "POSITION_AXIS_HYPOTHESIS_V1"
    assert result["status"] in {"BOUNDED_DIRECTION_HYPOTHESIS", "ABSTAINED"}
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    hypothesis = result["hypothesis"]
    assert hypothesis["axis_is_verified"] is False
    assert hypothesis["regions_are_verified"] is False
    assert hypothesis["axis_location"] is None
    assert hypothesis["model"] == "PAIR_OF_IDEAL_PLANES"
    assert {
        "SUPPLIED_NEIGHBORHOODS_NOT_REGION_DISCOVERY", "AXIS_AND_REGIONS_UNVERIFIED",
        "POSITION_ERROR_BOUND_UNCALIBRATED", "BOUND_NOT_NUMERICALLY_CERTIFIED",
        "NORMALS_ESTIMATED_FROM_SAME_POSITIONS_NOT_INDEPENDENT_EVIDENCE",
        "DIRECTION_ONLY_NO_CYLINDER_AXIS_LOCATION", "UNORIENTED_AXIS_NO_SIGNED_BEND_ANGLE",
        "SAMPLE_COMPATIBILITY_NOT_SURFACE_IDENTITY", "NO_CONTINUITY_OR_FINITE_SUPPORT_CLAIM",
        "NO_RADIUS_FORMATION_THICKNESS_OR_CONFORMANCE_CLAIM",
    } <= set(result["limitations"])
    if result["status"] == "ABSTAINED":
        assert result["reasons"]
    else:
        assert result["reasons"] == []
        assert hypothesis["axis"] is not None
        assert hypothesis["axis_angle_bound_radians"] <= result["policy"]["max_axis_angle_bound_radians"]
    for plane in result["plane_neighborhoods"]:
        assert plane["effective_error_bound"] == (
            result["position_error_bound"] + plane["numerical_allowance"]
        )
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def assert_plane_distances(plane, points, *, atol=1e-11):
    witness = plane["plane_witness"]
    if witness is None:
        return
    normal = np.asarray(witness["normal"])
    assert_allclose(np.linalg.norm(normal), 1)
    signed = ((points - np.asarray(witness["reference_point"])) @ normal
              - witness["offset_from_reference"])
    assert_allclose(witness["max_residual"], np.max(np.abs(signed)), atol=atol)
    assert witness["compatible_with_error_bound"] == (
        witness["max_residual"] <= plane["effective_error_bound"]
    )


def rotation_from_quaternion(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def test_observation_channel_has_no_supplied_axis_normals_labels_or_radius():
    parameters = inspect.signature(analyze_position_axis_hypothesis).parameters
    assert set(parameters) == {
        "first_neighborhood", "second_neighborhood", "unit", "position_error_bound",
        "max_axis_angle_bound",
    }
    for name in ("unit", "position_error_bound", "max_axis_angle_bound"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("epsilon", [0, 0.001])
def test_hand_computed_orthogonal_planes_and_normal_cone(epsilon):
    first, second = planes()
    result = analyze(first, second, position_error_bound=epsilon)
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert result["input_count"] == result["unique_count"] == 8
    assert_allclose(projector(result["hypothesis"]["axis"]), np.diag([0, 1, 0]))
    assert_allclose(result["axis_diagnostics"]["normal_separation_radians"], math.pi / 2)
    for plane, points, normal in zip(result["plane_neighborhoods"], (first, second),
                                     ([0, 0, 1], [1, 0, 0])):
        assert_plane_distances(plane, points)
        assert_allclose(plane["singular_values"], [4, 2, 0])
        assert_allclose(projector(plane["plane_witness"]["normal"]), projector(normal))
        # sqrt(N)=2, s2=2: sin(alpha) <= effective epsilon in this rectangle.
        expected = math.asin(plane["effective_error_bound"]) + 128 * np.finfo(float).eps
        assert_allclose(plane["normal_angle_bound_radians"], expected, atol=1e-15)
    assert result["axis_diagnostics"]["parallel_alternative_not_excluded"] is False


def test_second_hand_computed_pair_has_oblique_unoriented_axis():
    # x+2y+2z=6, 2x+y-2z=3. Their cross normal is proportional to (-2,2,-1).
    first = np.array([[6, 0, 0] + u*np.array([-2, 1, 0]) + v*np.array([-2, 0, 1])
                      for u in (-1, 1) for v in (-1, 1)], float)
    second = np.array([[0, 3, 0] + u*np.array([1, -2, 0]) + v*np.array([0, 2, 1])
                       for u in (-1, 1) for v in (-1, 1)], float)
    assert_allclose(first @ [1, 2, 2], [6]*4)
    assert_allclose(second @ [2, 1, -2], [3]*4)
    result = analyze(first, second)
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert_allclose(projector(result["hypothesis"]["axis"]), projector(np.array([-2, 2, -1])/3))
    for plane, points in zip(result["plane_neighborhoods"], (first, second)):
        assert_plane_distances(plane, points)


@pytest.mark.parametrize("separation", [0, 3, -100])
def test_parallel_planes_never_supply_axis_direction(separation):
    first, _ = planes()
    result = analyze(first, first + [0, 0, separation])
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["hypothesis"]["axis"] is None
    assert result["hypothesis"]["axis_angle_bound_radians"] is None
    assert "PARALLEL_ALTERNATIVE_NOT_EXCLUDED" in result["reasons"]
    assert "AXIS_DIRECTION_BOUND_UNRESOLVED" in result["reasons"]


def test_nearly_parallel_planes_are_ambiguous_under_position_budget():
    first, _ = planes()
    second = first.copy()
    second[:, 2] += 3 + 0.0001*second[:, 0]
    result = analyze(first, second, position_error_bound=0.001)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert "PARALLEL_ALTERNATIVE_NOT_EXCLUDED" in result["reasons"]
    # A concrete parallel alternative: remove the small slope, well within epsilon.
    assert np.max(np.abs(second[:, 2] - 5)) <= 0.001


def test_small_plane_separation_can_resolve_under_explicit_near_exact_budget():
    first, _ = planes()
    second = first.copy()
    second[:, 2] += 0.00001*second[:, 0]
    result = analyze(first, second, position_error_bound=0, max_axis_angle_bound=1e-5)
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert_allclose(projector(result["hypothesis"]["axis"]), np.diag([0, 1, 0]))


@pytest.mark.parametrize("first", [
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    [[2, 3, 4], [4, 7, 10], [6, 11, 16], [8, 15, 22]],
])
def test_rank_deficient_position_neighborhood_abstains(first):
    result = analyze(first)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert "FIRST_RANK_DEFICIENT_NEIGHBORHOOD" in result["reasons"]
    assert result["hypothesis"]["axis"] is None


def test_thin_planar_neighborhood_cannot_gain_orientation_from_tiny_residual():
    first = np.array([[x, y, 0] for x in (-2, 2) for y in (-1e-5, 1e-5)])
    result = analyze(first, position_error_bound=0.001)
    assert_scope(result)
    assert result["plane_neighborhoods"][0]["plane_witness"]["max_residual"] == 0
    assert result["status"] == "ABSTAINED"
    assert "FIRST_PLANE_NORMAL_UNRESOLVED" in result["reasons"]


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("missing", [None, [], np.empty((0, 3))])
def test_missing_neighborhood_preserves_unknown_not_missing_bend(side, missing):
    pair = list(planes())
    pair[side] = missing
    result = analyze_position_axis_hypothesis(*pair, unit="mm", position_error_bound=0.001,
                                             max_axis_angle_bound=0.1)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert f"{('FIRST', 'SECOND')[side]}_MISSING_NEIGHBORHOOD" in result["reasons"]
    assert result["hypothesis"]["axis"] is None


@pytest.mark.parametrize("first", [[[1, 2, 3]], [[1, 2, 3]]*5,
                                   [[0, 0, 0], [1, 2, 3], [0, 0, 0]]])
def test_repeated_or_insufficient_positions_do_not_create_plane_evidence(first):
    result = analyze(first)
    assert_scope(result)
    assert "FIRST_INSUFFICIENT_DISTINCT_POSITIONS" in result["reasons"]


def test_nonplanar_neighborhood_abstains_without_outlier_trimming():
    # A regular tetrahedron, not a planar patch.
    first = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], float)
    result = analyze(first)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert "FIRST_NO_ALL_POINT_PLANE_WITNESS" in result["reasons"]
    assert result["plane_neighborhoods"][0]["unique_count"] == 4
    assert_plane_distances(result["plane_neighborhoods"][0], first)


def test_all_point_plane_check_does_not_replace_maximum_with_rms():
    first = np.array([[x, y, 0] for x in (-2, -1, 0, 1, 2) for y in (-2, -1, 0, 1, 2)]
                     + [[0, 0, 0.2]], float)
    result = analyze(first, position_error_bound=0.05)
    plane = result["plane_neighborhoods"][0]
    assert_allclose(plane["plane_witness"]["max_residual"], 0.1)
    assert plane["plane_witness"]["compatible_with_error_bound"] is False
    assert "FIRST_NO_ALL_POINT_PLANE_WITNESS" in result["reasons"]
    assert_scope(result)


def test_precision_policy_remains_attached_and_can_require_abstention():
    result = analyze(max_axis_angle_bound=0.0001)
    assert result["policy"] == {"max_axis_angle_bound_radians": 0.0001}
    assert result["status"] == "ABSTAINED"
    assert result["hypothesis"]["axis_angle_bound_radians"] > 0.0001
    assert "AXIS_ANGULAR_PRECISION_NOT_ESTABLISHED" in result["reasons"]
    assert_scope(result)


@pytest.mark.parametrize("seed", [3, 41, 901])
def test_se3_equivariance_is_for_unoriented_direction_not_display_sign(seed):
    rng = np.random.default_rng(seed)
    rotation = rotation_from_quaternion(rng.normal(size=4))
    translation = rng.uniform(-5, 5, size=3)
    assert_allclose(rotation.T @ rotation, np.eye(3))
    assert_allclose(np.linalg.det(rotation), 1)
    first, second = planes()
    original = analyze(first, second)
    result = analyze(first @ rotation.T + translation, second @ rotation.T + translation)
    assert_scope(result)
    assert result["status"] == original["status"]
    assert_allclose(projector(result["hypothesis"]["axis"]),
                    rotation @ projector(original["hypothesis"]["axis"]) @ rotation.T)
    for plane, moved in zip(result["plane_neighborhoods"],
                            (first @ rotation.T + translation, second @ rotation.T + translation)):
        assert_plane_distances(plane, moved)
    # Numerical allowances depend on coordinate magnitude; they are not physical information.
    assert_allclose(result["hypothesis"]["axis_angle_bound_radians"],
                    original["hypothesis"]["axis_angle_bound_radians"], atol=1e-11)


def test_units_scale_all_dimensional_inputs_and_keep_angular_geometry():
    first, second = planes()
    original = analyze(first, second)
    result = analyze(first*0.001, second*0.001, unit="m", position_error_bound=1e-6)
    assert_scope(result)
    assert result["unit"] == "m"
    assert result["status"] == original["status"]
    assert_allclose(result["hypothesis"]["axis_angle_bound_radians"],
                    original["hypothesis"]["axis_angle_bound_radians"], atol=1e-15)
    assert_allclose(projector(result["hypothesis"]["axis"]), projector(original["hypothesis"]["axis"]))
    for before, after in zip(original["plane_neighborhoods"], result["plane_neighborhoods"]):
        for field in ("numerical_allowance", "effective_error_bound", "singular_values"):
            assert_allclose(np.asarray(after[field]), np.asarray(before[field])*0.001, atol=1e-14)
        for field in ("reference_point", "offset_from_reference", "max_residual"):
            assert_allclose(np.asarray(after["plane_witness"][field]),
                            np.asarray(before["plane_witness"][field])*0.001)


def test_neighborhood_reversal_row_permutation_and_repeats_preserve_geometry():
    first, second = planes()
    result = analyze(first, second)
    changed = analyze(np.repeat(second[[3, 1, 0, 2]], 3, axis=0), first[::-1])
    assert_scope(changed)
    assert changed["input_count"] == 16
    assert changed["unique_count"] == 8
    assert changed["status"] == result["status"]
    assert_allclose(projector(changed["hypothesis"]["axis"]), projector(result["hypothesis"]["axis"]))
    assert_allclose(changed["hypothesis"]["axis_angle_bound_radians"],
                    result["hypothesis"]["axis_angle_bound_radians"], atol=1e-15)
    assert changed["plane_neighborhoods"][0]["input_count"] == 12
    assert changed["plane_neighborhoods"][0]["unique_count"] == 4


def test_correlated_rotation_error_can_give_perfect_plane_fits_and_wrong_axis():
    first, second = planes()
    # Rotate both true planes together; no independent/noise-cancellation assumption.
    angle = 0.01
    rotation = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)],
                         [0, math.sin(angle), math.cos(angle)]])
    observed = [first @ rotation.T, second @ rotation.T]
    error = max(np.max(np.linalg.norm(measured-true, axis=1))
                for measured, true in zip(observed, (first, second)))
    result = analyze(*observed, position_error_bound=float(error))
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    actual_error = unoriented_angle(result["hypothesis"]["axis"], [0, 1, 0])
    assert_allclose(actual_error, angle, atol=1e-14)
    assert actual_error <= result["hypothesis"]["axis_angle_bound_radians"]
    for plane in result["plane_neighborhoods"]:
        assert plane["plane_witness"]["max_residual"] < 1e-12


def test_same_flange_positions_leave_cylinder_axis_location_and_radius_unknown():
    # Analytic counterexample: these patches lie on z=0 and x=0 for cylinders
    # centred at (-2,0,-2) with R=2 OR (-4,0,-4) with R=4. Only direction agrees.
    first = np.array([[x, y, 0] for x in (-10, -8) for y in (-2, 2)], float)
    second = np.array([[0, y, z] for z in (-10, -8) for y in (-2, 2)], float)
    result = analyze(first, second)
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert result["hypothesis"]["axis_location"] is None
    assert "radius" not in result
    assert_allclose(projector(result["hypothesis"]["axis"]), np.diag([0, 1, 0]))


def test_separate_objects_with_arbitrary_gap_still_only_make_a_direction_hypothesis():
    first, second = planes()
    result = analyze(first, second + [10, 100, 20])
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert result["hypothesis"]["regions_are_verified"] is False
    assert "axial_sample_groups" not in result
    assert "axis_endpoints" not in result


def run_synthetic_containment_sweep(*, expanded=False):
    """Explicit bounded optional 10,000-case sweep; pytest uses 360 by default."""
    # Truth and noise are created independently of fitted normals. The optional
    # sweep is 10,000 x 2 x 25 x 3 = 1,500,000 truth floats, not physical data.
    rng = np.random.default_rng(20260907)
    if expanded:
        count = 10000
        scales = 10**rng.uniform(-9, 9, size=count)
        budgets = scales * 10**rng.uniform(-10, -1, size=count)
        betas = 10**rng.uniform(-8, math.log10(math.pi/2), size=count)
        # Include exactly parallel, exact orthogonal and near-orthogonal cases.
        betas[::97] = 0
        betas[1::97] = math.pi/2
        betas[2::97] = math.pi/2 - 1e-12
        selected_rotations = np.stack([
            rotation_from_quaternion(q) for q in rng.normal(size=(count, 4))
        ])
    else:
        # 5 scales x 4 error fractions x 6 separations x 3 rigid orientations.
        configurations = [(scale, fraction, beta, orientation)
                          for scale in (1e-6, 1e-3, 1, 1e3, 1e6)
                          for fraction in (1e-8, 1e-5, 1e-3, 0.02)
                          for beta in (1e-5, 0.001, 0.03, 0.2, 0.7, math.pi/2)
                          for orientation in range(3)]
        count = len(configurations)
        scales = np.asarray([case[0] for case in configurations])
        budgets = scales * np.asarray([case[1] for case in configurations])
        betas = np.asarray([case[2] for case in configurations])
        rotations = np.stack([rotation_from_quaternion(q) for q in rng.normal(size=(3, 4))])
        selected_rotations = rotations[[case[3] for case in configurations]]
    grid = np.array([[u, v] for u in (-3, -1.5, 0, 1.5, 3)
                     for v in (-2, -1, 0, 1, 2)], float)
    truth = np.zeros((count, 2, len(grid), 3))
    truth[:, 0, :, 0] = grid[:, 0]
    truth[:, 0, :, 1] = grid[:, 1]
    truth[:, 1, :, 0] = np.cos(betas)[:, None]*grid[:, 0]
    truth[:, 1, :, 1] = grid[:, 1]
    truth[:, 1, :, 2] = -np.sin(betas)[:, None]*grid[:, 0]
    truth = np.einsum("cij,cpnj->cpni", selected_rotations, truth)
    truth = (truth + np.array([5, -2, 3])) * scales[:, None, None, None]
    normals = np.zeros((count, 2, 3))
    normals[:, 0, 2] = 1
    normals[:, 1, 0] = np.sin(betas)
    normals[:, 1, 2] = np.cos(betas)
    normals = np.einsum("cij,cpj->cpi", selected_rotations, normals)
    true_axes = selected_rotations[:, :, 1]
    perturbations = rng.normal(size=truth.shape)
    perturbations /= np.linalg.norm(perturbations, axis=-1, keepdims=True)
    perturbations *= rng.uniform(0, 0.95, size=truth.shape[:-1] + (1,))
    perturbations *= budgets[:, None, None, None]
    observed = truth + perturbations
    # The safety margin keeps roundoff from making the constructed physical
    # deviation exceed its declared budget at the tested scales.
    assert np.all(np.linalg.norm(observed-truth, axis=-1) <= budgets[:, None, None])
    counts = {"cases": count, "positions_per_case": 50,
              "BOUNDED_DIRECTION_HYPOTHESIS": 0, "ABSTAINED": 0,
              "axis_bounds_available": 0,
              "max_normal_error_over_bound": 0.0, "max_axis_error_over_bound": 0.0}
    for index, pair in enumerate(observed):
        result = analyze(*pair, position_error_bound=float(budgets[index]), max_axis_angle_bound=0.25)
        assert_scope(result)
        for plane, normal in zip(result["plane_neighborhoods"], normals[index]):
            assert plane["plane_witness"] is not None
            actual = unoriented_angle(plane["plane_witness"]["normal"], normal)
            assert actual <= plane["normal_angle_bound_radians"] + 1e-12
            counts["max_normal_error_over_bound"] = max(
                counts["max_normal_error_over_bound"], actual/plane["normal_angle_bound_radians"]
            )
        hypothesis = result["hypothesis"]
        if hypothesis["axis_angle_bound_radians"] is not None:
            assert betas[index] > 0  # Exactly parallel truth cannot identify an axis.
            actual = unoriented_angle(hypothesis["axis"], true_axes[index])
            assert actual <= hypothesis["axis_angle_bound_radians"] + 1e-11
            counts["axis_bounds_available"] += 1
            counts["max_axis_error_over_bound"] = max(
                counts["max_axis_error_over_bound"], actual/hypothesis["axis_angle_bound_radians"]
            )
        counts[result["status"]] += 1
    return counts


def test_vectorized_fixed_seed_360_case_bounded_error_containment_sweep():
    counts = run_synthetic_containment_sweep()
    assert counts["cases"] == 360
    assert counts["BOUNDED_DIRECTION_HYPOTHESIS"] >= 150
    assert counts["ABSTAINED"] > 0


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("unit", None), ("position_error_bound", -1),
    ("position_error_bound", True), ("position_error_bound", float("nan")),
    ("position_error_bound", float("inf")), ("position_error_bound", "0.01"),
    ("max_axis_angle_bound", 0), ("max_axis_angle_bound", -0.1),
    ("max_axis_angle_bound", math.pi/2), ("max_axis_angle_bound", float("inf")),
    ("max_axis_angle_bound", True), ("max_axis_angle_bound", "0.1"),
])
def test_invalid_policy_rejected(field, value):
    with pytest.raises(ValueError):
        analyze(**{field: value})


@pytest.mark.parametrize("positions", [
    1, "planes", iter([[0, 0, 0]]), [[0, 0]], [[0, 0, 0, 0]],
    [[False, 0, 0]], [["1", 2, 3]], [[complex(1), 2, 3]],
    [[float("nan"), 1, 2]], [[float("inf"), 1, 2]],
    np.ones((3, 3, 1)), np.ones((3, 4)), np.array(3), np.zeros((2, 3), dtype=bool),
    np.zeros((2, 3), dtype=complex), np.zeros((2, 3), dtype=object),
])
def test_invalid_positions_rejected(positions):
    with pytest.raises(ValueError):
        analyze(positions)


def test_total_sample_cap_is_shared_and_checked_before_array_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized neighborhood converted before cap rejection")
    monkeypatch.setattr(axis_module.np, "asarray", forbidden)
    with pytest.raises(ValueError, match="256 total"):
        analyze_position_axis_hypothesis([[0, 0, 0]]*129, [[0, 0, 0]]*128,
                                         unit="mm", position_error_bound=0.1, max_axis_angle_bound=0.1)


def test_oversized_row_rejected_before_array_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized row converted")
    monkeypatch.setattr(axis_module.np, "asarray", forbidden)
    with pytest.raises(ValueError, match="exactly three"):
        analyze_position_axis_hypothesis([[0]*1000], [], unit="mm", position_error_bound=0.1,
                                         max_axis_angle_bound=0.1)


def test_exact_256_row_cap_works_without_duplicate_evidence():
    first = np.array([[x, y, 0] for x in np.linspace(-2, 2, 16) for y in np.linspace(-1, 1, 8)])
    second = first[:, [2, 1, 0]]
    result = analyze(first, second)
    assert_scope(result)
    assert result["input_count"] == result["unique_count"] == 256
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"


def test_inputs_are_not_modified():
    first, second = planes()
    preserved = [first.copy(), second.copy()]
    analyze(first, second)
    assert_array_equal(first, preserved[0])
    assert_array_equal(second, preserved[1])


def test_svd_failure_abstains_without_fabricating_direction(monkeypatch):
    def unavailable(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic failure")
    monkeypatch.setattr(axis_module.np.linalg, "svd", unavailable)
    result = analyze()
    assert_scope(result)
    assert result["hypothesis"]["axis"] is None
    assert "FIRST_PLANE_FIT_UNAVAILABLE" in result["reasons"]


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_extreme_but_supported_finite_scales_have_finite_outputs(scale):
    first, second = planes()
    result = analyze(first*scale, second*scale, position_error_bound=0.001*scale)
    assert_scope(result)
    assert result["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
    assert_allclose(projector(result["hypothesis"]["axis"]), np.diag([0, 1, 0]))


def test_unrepresentable_coordinate_difference_abstains_with_finite_json():
    first = [[-1.7e308, 0, 0], [1.7e308, 0, 0], [0, 1, 0]]
    result = analyze(first)
    assert_scope(result)
    assert "FIRST_NUMERICAL_RANGE_UNSUPPORTED" in result["reasons"]


def test_large_translation_guard_cannot_masquerade_as_added_physical_precision():
    first, second = planes()
    original = analyze(first, second, position_error_bound=0)
    shifted = analyze(first + 1e14, second + 1e14, position_error_bound=0)
    assert_scope(shifted)
    assert shifted["plane_neighborhoods"][0]["numerical_allowance"] > 1
    assert shifted["status"] == "ABSTAINED"
    assert original["status"] == "BOUNDED_DIRECTION_HYPOTHESIS"
