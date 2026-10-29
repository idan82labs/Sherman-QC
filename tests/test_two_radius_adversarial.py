"""Independent position-only mixed-radius oracles; synthetic memory only.

The hidden centers, radii and assignments below are test constructions, never
inference inputs. A two-band sample witness is not unique physical segmentation.
"""
import inspect
import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pytest

from research.geometry.plane_conditioned_two_radius import analyze_plane_conditioned_two_radius


def planes():
    """Two broad exact planes with common z direction, not tangent labels."""
    first = np.array([[x, 0.0, z] for x in (-100.0, 100.0) for z in (-100.0, 100.0)])
    second = np.array([[0.0, y, z] for y in (-100.0, 100.0) for z in (-100.0, 100.0)])
    return first, second


def extrude(section, levels=(-1.0, 0.0, 1.0)):
    return np.array([[x, y, z] for x, y in section for z in levels], float)


def two_arcs(radii=(2.0, 3.0), angles=None, center=(0.0, 0.0)):
    angles = np.linspace(0.2, 1.4, 7) if angles is None else np.asarray(angles)
    section = np.array([[center[0] + r * math.cos(t), center[1] + r * math.sin(t)]
                        for r in radii for t in angles])
    return extrude(section)


def ambiguous_section(*, perturb=False):
    """Eight samples on both {2,4}@0 and {2.5,4.5}@(3,0).

    A tiny angular perturbation keeps the first explanation exact, makes the
    quartic design full rank, and preserves the second within 0.001 distance.
    """
    points, labels = [], []
    for i, radius in enumerate((2.0, 4.0)):
        for j, alternative in enumerate((2.5, 4.5)):
            x = (radius**2 - alternative**2 + 9) / 6
            y = math.sqrt(radius**2 - x**2)
            for sign in (-1, 1):
                points.append((x, sign * y))
                labels.append((i, j))
    points = np.array(points)
    if perturb:
        for k, (point, (i, _)) in enumerate(zip(points, labels)):
            angle = math.atan2(point[1], point[0]) + 1e-5 * (k + 1)
            radius = (2.0, 4.0)[i]
            points[k] = radius * np.array([math.cos(angle), math.sin(angle)])
    return points, np.array(labels)


def parallel_flat_alternative_section():
    """Two circles and two parallel lines explain the same noisy samples."""
    points = []
    for radius in (2.0, 4.0):
        for y in (-1.0, 1.0):
            for sign in (-1, 1):
                x = sign * math.sqrt(radius**2 - y**2)
                angle = math.atan2(y, x) + 1e-5 * (len(points) + 1)
                points.append((radius * math.cos(angle), radius * math.sin(angle)))
    return np.array(points)


def quartic_design(points):
    x, y = np.asarray(points).T
    square = x * x + y * y
    return np.column_stack((square * x, square * y, x * x, x * y, y * y,
                            x, y, np.ones(len(points)))), -(square * square)


def analyze(region=None, *, neighborhoods=None, **changes):
    policy = dict(unit="mm", position_error_bound=1e-6, max_axis_angle_bound=0.1,
                  max_axial_gap=1.1, max_radius_width=0.1, min_band_points=6,
                  max_split_evaluations=255)
    policy.update(changes)
    first, second = planes() if neighborhoods is None else neighborhoods
    return analyze_plane_conditioned_two_radius(
        first, second, two_arcs() if region is None else region, **policy,
    )


def assert_scope(result):
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["original_joint_compatibility"] == "UNESTABLISHED"
    assert result["physical_face_count"] is None
    assert result["thickness"] is None
    assert result["partition_is_unique"] is False
    assert result["partition_uniqueness"] == "UNESTABLISHED"
    assert result["physical_geometry_is_verified"] is False
    assert "radius_interval" not in result
    assert "SINGLE_PLANE_ALTERNATIVE_NOT_EXHAUSTIVE_SURFACE_MODEL_SELECTION" in result["limitations"]

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"axis_point", "axis_endpoints"}
                if key.endswith(("is_verified", "are_verified")):
                    assert item is False, key
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(result)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    if result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS":
        assert result["search_complete_within_declared_policy"] is True
        assert result["models"] and result["reasons"] == []
    else:
        assert result["status"] == "ABSTAINED"
        assert result["models"] == [] and result["reasons"]


def assert_all_point_witness(result):
    points = np.asarray(result["mixed_positions"])
    for model in result["models"]:
        assert model["all_points_accounted_for"] is True
        assert model["point_count"] == len(points)
        assert model["scope"] == "TWO_CONCENTRIC_CIRCLE_WITNESS_IN_ROTATED_ERROR_RELAXATION_ONLY"
        indices = [index for band in model["bands"] for index in band["point_indices"]]
        assert sorted(indices) == list(range(len(points)))
        projected = ((points - model["projection_pivot"]) / model["projection_scale"]
                     @ np.asarray(model["projection_basis"]))
        distances = np.linalg.norm(projected - model["normalized_projection_center"], axis=1)
        distances *= model["projection_scale"]
        assert model["witness_radii"][1] - model["witness_radii"][0] > (
            2 * result["global_band_input_error_bound"]
        )
        for band, radius in zip(model["bands"], model["witness_radii"]):
            assert band["position_error_bound"] == result["global_band_input_error_bound"]
            residual = np.max(np.abs(distances[band["point_indices"]] - radius))
            assert residual <= result["global_band_input_error_bound"] + 1e-12
            assert band["radius_interval_scope"] == (
                "CONDITIONAL_ON_PROPOSED_BAND_MEMBERSHIP_AND_AXIS_ROTATION_RELAXATION"
            )
            lower, upper = band["radius_interval"]
            assert upper is not None and lower <= radius <= upper


def test_product_polynomial_coefficients_recover_common_center_without_assignments():
    center = np.array([0.4, -0.7])
    region = two_arcs(center=center)
    section = np.unique(region[:, :2], axis=0)
    matrix, target = quartic_design(section)
    coefficient, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    assert rank == 8
    assert_allclose(-coefficient[:2] / 4, center, atol=2e-11, rtol=0)
    d1, d2 = [float(center @ center) - radius**2 for radius in (2.0, 3.0)]
    cx, cy = center
    expected = [-4 * cx, -4 * cy, 4 * cx * cx + d1 + d2,
                8 * cx * cy, 4 * cy * cy + d1 + d2,
                -2 * (d1 + d2) * cx, -2 * (d1 + d2) * cy, d1 * d2]
    assert_allclose(coefficient, expected, atol=2e-9, rtol=0)
    assert_allclose(matrix @ expected, target, atol=5e-13, rtol=0)


@pytest.mark.parametrize("perturb", [False, True])
def test_distinct_concentric_explanations_can_share_every_observed_sample(perturb):
    section, labels = ambiguous_section(perturb=perturb)
    distances_a = np.linalg.norm(section, axis=1)
    distances_b = np.linalg.norm(section - [3, 0], axis=1)
    targets_a = np.array([2.0, 4.0])[labels[:, 0]]
    targets_b = np.array([2.5, 4.5])[labels[:, 1]]
    assert_allclose(distances_a, targets_a, atol=1e-14, rtol=0)
    assert np.max(np.abs(distances_b - targets_b)) < (0.001 if perturb else 1e-14)
    for model in range(2):
        for band in range(2):
            support = section[labels[:, model] == band]
            assert len(support) == 4
            assert np.linalg.matrix_rank(support - support.mean(axis=0)) == 2
    matrix, target = quartic_design(section)
    coefficient, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    assert rank == (8 if perturb else 7)
    if perturb:
        assert_allclose(-coefficient[:2] / 4, [0, 0], atol=5e-10, rtol=0)
        assert np.max(np.linalg.norm(section - ambiguous_section()[0], axis=1)) < 0.001


def test_two_cylinder_sample_witness_need_not_exclude_two_flat_faces():
    section = parallel_flat_alternative_section()
    assert_allclose(np.linalg.norm(section, axis=1), [2.0] * 4 + [4.0] * 4,
                    atol=1e-14, rtol=0)
    # Changing y to its sign gives counterparts on two parallel planes after
    # z extrusion, without moving any point farther than the common error.
    assert np.max(np.abs(np.abs(section[:, 1]) - 1.0)) < 0.001
    assert np.linalg.matrix_rank(quartic_design(section)[0]) == 8


def test_public_input_channel_does_not_accept_center_axis_radius_or_membership():
    parameters = inspect.signature(analyze_plane_conditioned_two_radius).parameters
    assert set(parameters) == {
        "first_neighborhood", "second_neighborhood", "mixed_region", "unit",
        "position_error_bound", "max_axis_angle_bound", "max_axial_gap", "max_radius_width",
        "min_band_points", "max_split_evaluations",
    }


def test_independent_arcs_produce_a_constructive_all_point_common_center_witness():
    result = analyze()
    assert_scope(result)
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert len(result["models"]) == 1
    assert_all_point_witness(result)
    assert_allclose(result["models"][0]["witness_radii"], [2, 3], atol=1e-11, rtol=0)
    assert result["input_count"] == 50


def test_exact_ambiguous_centers_abstain_at_rank_deficiency():
    result = analyze(extrude(ambiguous_section()[0]), position_error_bound=0)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_diagnostics"]["algebraic_rank"] == 7
    assert "ALGEBRAIC_CENTER_RANK_UNRESOLVED" in result["reasons"]


def test_full_rank_sample_witness_intervals_are_not_global_across_valid_partitions():
    section, labels = ambiguous_section(perturb=True)
    result = analyze(extrude(section), position_error_bound=0.001)
    assert_scope(result)
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert result["search_diagnostics"]["algebraic_rank"] == 8
    assert_all_point_witness(result)
    assert len(result["models"]) == 1
    model = result["models"][0]
    assert_allclose(model["witness_radii"], [2, 4], atol=1e-8, rtol=0)
    # The second legitimate physical model uses a DIFFERENT sample partition.
    # Its exact two-circle counterparts lie within the ORIGINAL .001 budget,
    # yet its radii fall outside the fitted-assignment intervals. This is not
    # an enclosure bug because the output never claims an all-partition bound.
    alternatives = np.array([2.5, 4.5])[labels[:, 1]]
    assert np.max(np.abs(np.linalg.norm(section - [3, 0], axis=1) - alternatives)) < 0.001
    assert model["bands"][0]["radius_interval"][1] < 2.5
    assert model["bands"][1]["radius_interval"][1] < 4.5
    assert "BAND_INTERVALS_CONDITIONAL_ON_PROPOSED_MEMBERSHIP_NOT_GLOBAL_RADIUS_BOUNDS" in (
        result["limitations"]
    )


def test_positive_two_circle_witness_can_still_be_two_flat_physical_faces():
    result = analyze(extrude(parallel_flat_alternative_section()), position_error_bound=0.001)
    assert_scope(result)
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert_all_point_witness(result)
    assert result["flat_alternative"]["compatible_with_error_bound"] is False
    assert result["physical_geometry_is_verified"] is False


@pytest.mark.parametrize("outlier", [[0, 0, 0], [7, 7, 0], [10, -10, 3]])
def test_a_third_incompatible_sample_is_not_dropped_from_the_whole_component(outlier):
    points = np.concatenate((two_arcs(), np.array([outlier], float)))
    result = analyze(points)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert len(result["mixed_positions"]) == len(np.unique(points, axis=0))
    assert list(map(float, outlier)) in result["mixed_positions"]


def test_many_axial_samples_do_not_replace_projected_spread():
    section = np.array([[r * math.cos(t), r * math.sin(t)]
                        for r in (2.0, 3.0) for t in (0.2, 1.4)])
    region = extrude(section, levels=np.linspace(-1, 1, 11))
    result = analyze(region)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert np.linalg.matrix_rank(quartic_design(section)[0]) < 8
    # The four section points can also lie on a single circle; that valid
    # early ambiguity stop need not reach the rank check at all.
    rank = result["search_diagnostics"]["algebraic_rank"]
    assert rank is None or rank < 8


def test_unresolved_nearby_bands_preserve_a_single_radius_alternative():
    result = analyze(two_arcs(radii=(2.0, 2.00001)), position_error_bound=0.0001)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert "SINGLE_RADIUS_ALTERNATIVE_COMPATIBLE" in result["reasons"]


def test_point_permutation_cannot_hide_an_incompatible_sample_or_change_a_clean_witness():
    original = two_arcs()
    permutation = np.random.default_rng(705).permutation(len(original))
    assert analyze(original) == analyze(original[permutation])
    contaminated = np.concatenate((original, [[0.0, 0.0, 0.0]]))
    assert analyze(contaminated) == analyze(contaminated[::-1])


@pytest.mark.parametrize("seed", [9, 31, 89])
@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_rigid_motion_and_unit_rescaling_preserve_conditional_radius_witness(seed, scale):
    rng = np.random.default_rng(seed)
    quaternion = rng.normal(size=4)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    translation = rng.uniform(-4, 4, size=3) * scale
    transformed = lambda p: np.asarray(p) @ rotation.T * scale + translation
    result = analyze(transformed(two_arcs()), neighborhoods=tuple(transformed(p) for p in planes()),
                     unit="m" if scale == 0.001 else "mm", position_error_bound=1e-6 * scale,
                     max_axial_gap=1.1 * scale, max_radius_width=0.1 * scale)
    assert_scope(result)
    assert result["status"] == "TWO_RADIUS_SAMPLE_HYPOTHESIS"
    assert_all_point_witness(result)
    assert_allclose(result["models"][0]["witness_radii"], np.array([2, 3]) * scale,
                    atol=1e-9 * scale, rtol=0)
