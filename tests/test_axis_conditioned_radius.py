"""Independent, synthetic radius/ambiguity oracles; no detector or data imports.

Only the stipulated axis and positions enter the proposed observation channel.
The tests retain construction truth separately, and a sample-circle witness is
never a physical formation, continuity, identity, or production claim.
"""

import inspect
import json
import math

import numpy as np
from numpy.testing import assert_allclose as numpy_assert_allclose, assert_array_equal
import pytest

from research.geometry import axis_conditioned_radius as radius_module
from research.geometry.axis_conditioned_radius import analyze_axis_conditioned_radius


def assert_allclose(actual, expected, *, atol=1e-11):
    numpy_assert_allclose(actual, expected, atol=atol, rtol=0)


def cloud(section, axial_levels=(-1.0, 0.0, 1.0)):
    """Extrude hand-supplied x-z coordinates, without fixture truth metadata."""
    section = tuple(section)
    return np.asarray([[x, y, z] for y in axial_levels for x, z in section], dtype=float)


def quarter_circle(axial_levels=(-1.0, 0.0, 1.0)):
    # Four integer points on x^2 + z^2 = 25, not a fit/generator inverse.
    return cloud(((0, 5), (3, 4), (4, 3), (5, 0)), axial_levels)


def analyze(points=None, **changes):
    arguments = {
        "axis": (0.0, 1.0, 0.0),
        "unit": "mm",
        "position_error_bound": 1e-7,
        "max_axial_gap": 1.01,
        "max_radius_width": 0.1,
    }
    arguments.update(changes)
    return analyze_axis_conditioned_radius(
        quarter_circle() if points is None else points, **arguments
    )


def assert_scope_and_json(result):
    """Check explicit limits even when geometry happens to fit perfectly."""
    assert result["kind"] == "AXIS_CONDITIONED_RADIUS_V1"
    assert result["schema_version"] == 1
    assert result["status"] in {"SAMPLE_CIRCLE_WITNESS", "ABSTAINED"}
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["hypothesis"]["axis_is_verified"] is False
    assert result["hypothesis"]["region_is_verified"] is False
    assert result["hypothesis"]["model"] == "SINGLE_CIRCULAR_CYLINDER_FACE"
    assert {
        "AXIS_AND_REGION_CONDITIONAL", "POSITION_ERROR_BOUND_UNCALIBRATED",
        "OUTER_BOUND_NOT_NUMERICALLY_CERTIFIED", "SAMPLE_COMPATIBILITY_NOT_SURFACE_IDENTITY",
        "NO_FORMATION_ANGLE_THICKNESS_OR_CONFORMANCE_CLAIM", "NO_CONTINUITY_FROM_SAMPLES",
    } <= set(result["limitations"])
    assert isinstance(result["reasons"], list)
    if result["status"] == "ABSTAINED":
        assert result["reasons"]
    assert result["numerical_allowance"] >= 0
    assert result["effective_error_bound"] >= result["position_error_bound"]
    assert result["effective_error_bound"] == pytest.approx(
        result["position_error_bound"] + result["numerical_allowance"],
        rel=0, abs=1e-15 * max(1, result["effective_error_bound"]),
    )
    # Python's default JSON encoder permits NaN/Infinity; this deliberately does not.
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded) == result


def assert_radius_encloses(result, true_radius, *, atol=1e-10):
    lower, upper = result["radius_interval"]
    assert math.isfinite(lower)
    assert 0 <= lower <= true_radius + atol
    if upper is not None:
        assert math.isfinite(upper)
        assert upper >= true_radius - atol


def assert_witness_residuals_are_all_point_distances(result, points):
    axis = np.asarray(result["hypothesis"]["axis"])
    circle = result["circle_witness"]
    if circle is not None:
        radial_distances = np.linalg.norm(
            np.cross(points - np.asarray(circle["axis_point"]), axis), axis=1
        )
        residuals = np.abs(radial_distances - circle["radius"])
        assert_allclose(circle["max_radial_residual"], residuals.max())
        assert_allclose(circle["rms_radial_residual"], np.sqrt(np.mean(residuals ** 2)))
        assert circle["compatible_with_error_bound"] == bool(
            circle["max_radial_residual"] <= result["effective_error_bound"]
        )
    plane = result["parallel_plane_witness"]
    if plane is not None:
        normal = np.asarray(plane["normal"])
        assert_allclose(np.linalg.norm(normal), 1)
        assert_allclose(normal @ axis, 0)
        distances = np.abs((points - np.asarray(plane["point"])) @ normal)
        assert_allclose(plane["max_residual"], distances.max())
        assert_allclose(plane["width"], 2 * plane["max_residual"])
        assert plane["compatible_with_error_bound"] == bool(
            plane["max_residual"] <= result["effective_error_bound"]
        )


def test_public_observation_channel_accepts_no_normals_labels_or_radius_prior():
    parameters = inspect.signature(analyze_axis_conditioned_radius).parameters
    assert set(parameters) == {
        "points", "axis", "unit", "position_error_bound", "max_axial_gap",
        "max_radius_width",
    }
    for name, parameter in parameters.items():
        if name != "points":
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("error_bound", [0.0, 1e-7])
def test_hand_computed_quarter_circle_has_radius_five_and_no_plane_witness(error_bound):
    points = quarter_circle()
    result = analyze(points, position_error_bound=error_bound)
    assert_scope_and_json(result)
    assert result["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert result["input_count"] == result["unique_count"] == 12
    circle = result["circle_witness"]
    assert_allclose(circle["radius"], 5)
    assert_allclose(circle["axis_point"], [0, 0, 0])
    assert circle["compatible_with_error_bound"] is True
    assert_radius_encloses(result, 5)
    # The narrowest strip is 5 <= x+z <= 7, so its width is sqrt(2).
    assert_allclose(result["parallel_plane_witness"]["width"], math.sqrt(2))
    assert_allclose(result["parallel_plane_witness"]["max_residual"], 1 / math.sqrt(2))
    assert result["parallel_plane_witness"]["compatible_with_error_bound"] is False
    groups = result["axial_sample_groups"]
    assert len(groups) == 1
    assert_allclose([groups[0]["sample_min"], groups[0]["sample_max"]], [-1, 1])
    assert_allclose(groups[0]["axis_endpoints"], [[0, -1, 0], [0, 1, 0]])
    assert_witness_residuals_are_all_point_distances(result, points)


def test_one_triangle_radius_enclosure_matches_independent_side_area_bounds():
    # Distinct section vertices: (0,5), (3,4), (5,0). Sides sqrt(10),
    # sqrt(20), sqrt(50); oriented double-area magnitude 10; R = abc/(2D)=5.
    points = cloud(((0, 5), (3, 4), (5, 0)))
    result = analyze(points, position_error_bound=0.02, max_radius_width=1)
    epsilon = result["effective_error_bound"]
    edges = (math.sqrt(10), math.sqrt(20), math.sqrt(50))
    area_error = epsilon * sum(edges) + 3 * epsilon ** 2
    expected_lower = math.prod(edge - 2 * epsilon for edge in edges) / (
        2 * (10 + area_error)
    )
    expected_upper = math.prod(edge + 2 * epsilon for edge in edges) / (
        2 * (10 - area_error)
    )
    assert_allclose(result["radius_interval"], [expected_lower, expected_upper], atol=1e-9)
    assert_radius_encloses(result, 5)
    assert result["status"] == "SAMPLE_CIRCLE_WITNESS"


def test_required_radius_precision_is_not_replaced_by_a_small_fit_residual():
    result = analyze(position_error_bound=0.02, max_radius_width=0.001)
    lower, upper = result["radius_interval"]
    assert result["circle_witness"]["compatible_with_error_bound"] is True
    assert upper is not None and upper - lower > 0.001
    assert result["status"] == "ABSTAINED"
    assert "RADIUS_PRECISION_NOT_ESTABLISHED" in result["reasons"]
    assert_scope_and_json(result)


def test_stipulated_gap_and_precision_policy_are_carried_with_the_result():
    result = analyze(max_axial_gap=1.25, max_radius_width=0.037)
    assert result["policy"] == {"max_axial_gap": 1.25, "max_radius_width": 0.037}


def test_minimum_width_strip_is_not_a_least_squares_plane():
    # Area is 6, longest side sqrt(18), hence min strip width = 12/sqrt(18).
    # The triangle is asymmetric; a centroid/PCA line need not minimize max error.
    points = cloud(((0, 0), (4, 0), (1, 3)))
    result = analyze(points, position_error_bound=1.42)
    plane = result["parallel_plane_witness"]
    assert_allclose(plane["width"], 2 * math.sqrt(2))
    assert_allclose(plane["max_residual"], math.sqrt(2))
    assert plane["compatible_with_error_bound"] is True
    assert result["status"] == "ABSTAINED"
    assert_witness_residuals_are_all_point_distances(result, points)


def test_large_radius_over_short_fixed_footprint_remains_plane_compatible():
    radius = 1000.0
    x = np.linspace(-1, 1, 5)
    # Stable exact-circle branch z = R - sqrt(R^2-x^2); visible sagitta ~0.0005.
    z = x ** 2 / (radius + np.sqrt(radius ** 2 - x ** 2))
    points = cloud(zip(x, z))
    result = analyze(points, position_error_bound=0.0003, max_radius_width=1e6)
    assert result["status"] == "ABSTAINED"
    assert result["parallel_plane_witness"]["compatible_with_error_bound"] is True
    assert "PARALLEL_PLANE_COMPATIBLE" in result["reasons"]
    assert_radius_encloses(result, radius)
    assert_scope_and_json(result)


def test_midpoint_observation_fits_wrong_radius_exactly_but_keeps_true_radius_possible():
    epsilon = 0.001
    h = 4 * epsilon
    true_section = np.array([[-1, 0], [0, h], [1, 0]])
    observed_section = np.array([[-1, h / 4], [0, 3 * h / 4], [1, h / 4]])
    plane_section = np.array([[-1, h / 2], [0, h / 2], [1, h / 2]])
    assert_allclose(np.linalg.norm(observed_section - true_section, axis=1), epsilon)
    assert_allclose(np.linalg.norm(observed_section - plane_section, axis=1), epsilon)
    true_radius = (1 + h ** 2) / (2 * h)
    observed_circle_radius = 1 / h + h / 4
    assert true_radius == pytest.approx(125.002, abs=1e-12)
    assert observed_circle_radius == pytest.approx(250.001, abs=1e-12)
    result = analyze(cloud(observed_section), position_error_bound=epsilon,
                     max_radius_width=1e6)
    assert result["status"] == "ABSTAINED"
    assert_allclose(result["circle_witness"]["radius"], observed_circle_radius, atol=1e-7)
    assert result["circle_witness"]["compatible_with_error_bound"] is True
    assert result["parallel_plane_witness"]["compatible_with_error_bound"] is True
    assert_allclose(result["parallel_plane_witness"]["width"], 2 * epsilon)
    assert_radius_encloses(result, true_radius)
    assert result["radius_interval"][1] is None
    assert "RADIUS_UPPER_BOUND_UNRESOLVED" in result["reasons"]
    assert_scope_and_json(result)


def test_sparse_multiplane_positions_can_be_concyclic_without_observing_a_transition():
    # These points come from two separate planar flange patches, z=0 and x=0.
    # Their four section endpoints happen to lie on the circle centred (-9,-9).
    points = cloud(((-10, 0), (-8, 0), (0, -10), (0, -8)))
    result = analyze(points)
    assert result["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert_allclose(result["circle_witness"]["radius"], math.sqrt(82))
    assert_allclose(result["circle_witness"]["axis_point"], [-9, 0, -9])
    assert_radius_encloses(result, math.sqrt(82))
    # The witness must retain the unverified region/axis and production exclusion.
    assert_scope_and_json(result)


def test_dense_hidden_flange_patches_do_not_supply_a_measured_bend_radius():
    coordinates = np.linspace(-10, -8, 9)
    section = [(x, 0) for x in coordinates] + [(0, z) for z in coordinates]
    result = analyze(cloud(section))
    assert result["status"] == "ABSTAINED"
    circle = result["circle_witness"]
    assert circle is None or circle["compatible_with_error_bound"] is False
    # Analytic models with inner R=2 and R=4 can both hide behind these same
    # flange observations. They do not satisfy this single circular-face model;
    # its failed witness is therefore not an infeasible/absent physical bend.
    assert_scope_and_json(result)


@pytest.mark.parametrize("radius", [0.5, 5.0, 1000.0])
@pytest.mark.parametrize("amplitude", [0.2, 0.9, 1.0])
def test_known_bounded_3d_perturbations_cannot_exclude_the_true_radius(radius, amplitude):
    angles = np.linspace(0, math.pi / 2, 9)
    truth = cloud(zip(radius * np.sin(angles), radius * np.cos(angles)),
                  axial_levels=(-0.5 * radius, 0.0, 0.5 * radius))
    epsilon = 0.003 * radius
    index = np.arange(1, len(truth) + 1, dtype=float)
    directions = np.column_stack((np.sin(index), np.cos(1.3 * index), np.sin(2.1 * index)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    observed = truth + amplitude * epsilon * directions
    assert np.max(np.linalg.norm(observed - truth, axis=1)) <= epsilon + 1e-12 * radius
    result = analyze(observed, position_error_bound=epsilon,
                     max_axial_gap=0.6 * radius, max_radius_width=radius)
    assert_radius_encloses(result, radius, atol=1e-9 * radius)
    assert result["radius_interval"][1] is not None
    assert_scope_and_json(result)
    # A least-squares optimizer may miss an existing bounded-error circle. Either
    # status is allowed here: enclosure, not fit success, is the independent oracle.


def test_disconnected_axial_samples_are_not_joined_into_one_supported_segment():
    points = quarter_circle((-8, -6, -3, 3, 6, 8))
    result = analyze(points, max_axial_gap=3.01)
    assert result["circle_witness"]["compatible_with_error_bound"] is True
    assert result["status"] == "ABSTAINED"
    assert "AXIAL_SAMPLE_GAP" in result["reasons"]
    groups = result["axial_sample_groups"]
    assert len(groups) == 2
    assert_allclose([[g["sample_min"], g["sample_max"]] for g in groups],
                    [[-8, -3], [3, 8]])
    assert_allclose([g["axis_endpoints"] for g in groups],
                    [[[0, -8, 0], [0, -3, 0]], [[0, 3, 0], [0, 8, 0]]])
    assert_scope_and_json(result)


def test_exact_gap_threshold_may_group_samples_but_larger_gap_must_not():
    points = quarter_circle((0, 1, 2))
    joined = analyze(points, max_axial_gap=1.0)
    split = analyze(points, max_axial_gap=0.99)
    assert len(joined["axial_sample_groups"]) == 1
    assert joined["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert len(split["axial_sample_groups"]) == 3
    assert split["status"] == "ABSTAINED"


def test_one_ulp_smaller_gap_limit_cannot_bridge_exact_cartesian_sample_gaps():
    # The Cartesian y coordinates and directed y axis have exactly observed unit
    # gaps; normalization must not round the stricter caller limit up to a bridge.
    result = analyze(quarter_circle((0, 1, 2)), max_axial_gap=np.nextafter(1.0, 0.0))
    assert len(result["axial_sample_groups"]) == 3
    assert result["status"] == "ABSTAINED"


def test_one_cross_section_has_no_positive_sampled_axial_extent():
    result = analyze(quarter_circle((7,)))
    assert result["circle_witness"]["compatible_with_error_bound"] is True
    assert len(result["axial_sample_groups"]) == 1
    group = result["axial_sample_groups"][0]
    assert group["sample_min"] == group["sample_max"]
    assert result["status"] == "ABSTAINED"
    assert "AXIAL_EXTENT_UNRESOLVED" in result["reasons"]
    assert_scope_and_json(result)


def test_mixed_inner_and_outer_faces_are_not_one_radius_estimate():
    angles = np.linspace(0, math.pi / 2, 7)
    sections = [(r * math.sin(phi), r * math.cos(phi))
                for r in (2.0, 3.0) for phi in angles]
    points = cloud(sections)
    result = analyze(points, position_error_bound=0.001)
    assert result["status"] == "ABSTAINED"
    circle = result["circle_witness"]
    assert circle is None or circle["compatible_with_error_bound"] is False
    assert_scope_and_json(result)


def test_a_single_outlier_is_not_hidden_by_rms_or_dense_circle_support():
    angles = np.linspace(0, math.pi / 2, 25)
    points = cloud(zip(5 * np.sin(angles), 5 * np.cos(angles)))
    points[-1] += [0.5, 0, 0]
    result = analyze(points, position_error_bound=0.01)
    assert result["status"] == "ABSTAINED"
    circle = result["circle_witness"]
    assert circle is None or circle["max_radial_residual"] > result["effective_error_bound"]
    assert_witness_residuals_are_all_point_distances(result, points)
    assert_scope_and_json(result)


def test_proper_rigid_motion_transforms_witnesses_and_preserves_scalar_geometry():
    points = quarter_circle()
    rotation = np.array([[-2 / 3, 2 / 15, 11 / 15],
                         [2 / 3, -1 / 3, 2 / 3],
                         [1 / 3, 14 / 15, 2 / 15]])
    translation = np.array([11.0, -7.0, 3.0])
    assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert_allclose(np.linalg.det(rotation), 1, atol=1e-15)
    original = analyze(points, position_error_bound=0.002)
    moved_points = points @ rotation.T + translation
    moved = analyze(moved_points, axis=rotation @ np.array([0, 1, 0]),
                    position_error_bound=0.002)
    assert moved["status"] == original["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert_allclose(moved["radius_interval"], original["radius_interval"], atol=1e-9)
    assert_allclose(moved["circle_witness"]["radius"], original["circle_witness"]["radius"])
    assert_allclose(moved["circle_witness"]["axis_point"],
                    rotation @ original["circle_witness"]["axis_point"] + translation)
    assert_allclose(moved["parallel_plane_witness"]["width"],
                    original["parallel_plane_witness"]["width"])
    for before, after in zip(original["axial_sample_groups"], moved["axial_sample_groups"]):
        assert_allclose([after["sample_min"], after["sample_max"]],
                        [before["sample_min"], before["sample_max"]])
        assert_allclose(after["axis_endpoints"],
                        np.asarray(before["axis_endpoints"]) @ rotation.T + translation)
    assert_witness_residuals_are_all_point_distances(moved, moved_points)
    assert_scope_and_json(moved)


def test_axis_reversal_changes_only_directed_axial_coordinate_convention():
    points = quarter_circle((-8, -5, 2, 4))
    original = analyze(points, max_axial_gap=3.01)
    reversed_axis = analyze(points, axis=(0, -1, 0), max_axial_gap=3.01)
    assert reversed_axis["status"] == original["status"] == "ABSTAINED"
    assert_allclose(reversed_axis["radius_interval"], original["radius_interval"])
    assert_allclose(reversed_axis["circle_witness"]["radius"],
                    original["circle_witness"]["radius"])
    assert_allclose(reversed_axis["circle_witness"]["axis_point"],
                    original["circle_witness"]["axis_point"])
    groups = original["axial_sample_groups"]
    reverse_groups = reversed_axis["axial_sample_groups"]
    assert len(groups) == len(reverse_groups) == 2
    for before, after in zip(groups, reverse_groups[::-1]):
        assert_allclose([after["sample_min"], after["sample_max"]],
                        [-before["sample_max"], -before["sample_min"]])
        assert_allclose(after["axis_endpoints"], before["axis_endpoints"][::-1])


def test_explicit_unit_conversion_scales_all_length_outputs():
    points = quarter_circle() + np.array([11, -7, 3])
    mm = analyze(points, position_error_bound=0.02, max_radius_width=1)
    metres = analyze(points / 1000, unit="m", position_error_bound=0.00002,
                     max_axial_gap=0.00101, max_radius_width=0.001)
    assert mm["status"] == metres["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert mm["unit"] == "mm" and metres["unit"] == "m"
    for key in ("position_error_bound", "numerical_allowance", "effective_error_bound"):
        assert_allclose(metres[key], mm[key] / 1000, atol=1e-15)
    assert_allclose(metres["radius_interval"], np.asarray(mm["radius_interval"]) / 1000,
                    atol=1e-12)
    for key in ("radius", "max_radial_residual", "rms_radial_residual", "axis_point"):
        assert_allclose(metres["circle_witness"][key],
                        np.asarray(mm["circle_witness"][key]) / 1000, atol=1e-12)
    for key in ("width", "max_residual", "point"):
        assert_allclose(metres["parallel_plane_witness"][key],
                        np.asarray(mm["parallel_plane_witness"][key]) / 1000, atol=1e-12)
    assert_allclose(metres["parallel_plane_witness"]["normal"],
                    mm["parallel_plane_witness"]["normal"])
    for before, after in zip(mm["axial_sample_groups"], metres["axial_sample_groups"]):
        assert_allclose([after["sample_min"], after["sample_max"]],
                        np.array([before["sample_min"], before["sample_max"]]) / 1000)
        assert_allclose(after["axis_endpoints"], np.asarray(before["axis_endpoints"]) / 1000)
        assert after["point_count"] == before["point_count"]
    assert_scope_and_json(metres)


def test_permutation_and_repetition_do_not_create_new_sample_support():
    points = quarter_circle()
    original = analyze(points)
    repeated_points = np.repeat(points[::-1], 3, axis=0)
    repeated = analyze(repeated_points)
    assert repeated["input_count"] == 36
    assert original["unique_count"] == repeated["unique_count"] == 12
    assert repeated["status"] == original["status"]
    assert_allclose(repeated["radius_interval"], original["radius_interval"])
    assert_allclose(repeated["circle_witness"]["radius"], original["circle_witness"]["radius"])
    assert repeated["axial_sample_groups"] == original["axial_sample_groups"]


def test_input_array_is_not_modified_or_retained_as_mutable_report_evidence():
    points = quarter_circle()
    before = points.copy()
    result = analyze(points)
    assert_array_equal(points, before)
    serialized = json.dumps(result, allow_nan=False, sort_keys=True)
    points[:] = 100
    assert json.dumps(result, allow_nan=False, sort_keys=True) == serialized


@pytest.mark.parametrize("points", [
    np.empty((0, 3)),
    np.array([[1.0, 2.0, 3.0]]),
    np.array([[0.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
    np.repeat([[1.0, 2.0, 3.0]], 8, axis=0),
    cloud(((0, 0),)),
    cloud(((-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0))),
])
def test_valid_insufficient_or_collinear_samples_abstain_without_invalid_json(points):
    result = analyze(points)
    assert result["status"] == "ABSTAINED"
    assert_scope_and_json(result)


def test_hard_input_cap_counts_original_rows_before_deduplication():
    points = np.vstack([np.tile(quarter_circle(), (21, 1)), quarter_circle()[:4]])
    assert len(points) == 256
    result = analyze(points)
    assert result["input_count"] == 256
    assert result["unique_count"] == 12
    assert result["status"] == "SAMPLE_CIRCLE_WITNESS"
    with pytest.raises(ValueError):
        analyze(np.vstack([points, points[0]]))


@pytest.mark.parametrize("points", [
    None, 3, np.array(3), "points", [], [1, 2, 3], [[1, 2], [3, 4]], [[1, 2, 3, 4]],
    [[[1, 2, 3]]], [[1, 2, 3], [4, 5]], [[1, 2, None]],
    [["0", "1", "2"]], [[0, True, 2]], np.ones((3, 3), dtype=bool),
    np.ones((3, 3), dtype=complex), [[math.nan, 0, 0]], [[math.inf, 0, 0]],
    [[-math.inf, 0, 0]],
])
def test_malformed_positions_raise_value_error(points):
    # Call directly because the local convenience helper uses None for defaults.
    with pytest.raises(ValueError):
        analyze_axis_conditioned_radius(
            points, axis=(0, 1, 0), unit="mm", position_error_bound=0,
            max_axial_gap=1, max_radius_width=1,
        )


@pytest.mark.parametrize("axis", [
    None, 1, "axis", (0, 1), (0, 1, 0, 0), [[0, 1, 0]],
    (0, 0, 0), (0, 2, 0), (1, 1, 0), (0, math.nan, 0), (0, math.inf, 0),
    ("0", "1", "0"), (False, True, False), (0, True, 0), (0j, 1 + 0j, 0j),
])
def test_malformed_or_unstipulated_nonunit_axis_raises_value_error(axis):
    with pytest.raises(ValueError):
        analyze(axis=axis)


@pytest.mark.parametrize("unit", [None, "", "cm", "MM", "meters", 1, True, ["mm"]])
def test_units_are_explicit_and_not_inferred_or_coerced(unit):
    with pytest.raises(ValueError):
        analyze(unit=unit)


@pytest.mark.parametrize("name", ["position_error_bound", "max_axial_gap", "max_radius_width"])
@pytest.mark.parametrize("value", [None, True, "0.1", complex(1), math.nan, math.inf,
                                   -math.inf, -0.01, [0.1], np.array([0.1]),
                                   pytest.param(10 ** 1000, id="integer_overflow")])
def test_malformed_uncertainty_gap_and_precision_scalars_raise_value_error(name, value):
    with pytest.raises(ValueError):
        analyze(**{name: value})


@pytest.mark.parametrize("name", ["max_axial_gap", "max_radius_width"])
def test_gap_and_radius_width_must_be_strictly_positive(name):
    with pytest.raises(ValueError):
        analyze(**{name: 0.0})


@pytest.mark.parametrize("points", [
    pytest.param([[0.0] * 4096], id="too_wide_row"),
    pytest.param([[0.0, 0.0, 0.0]] * 257, id="too_many_rows"),
])
def test_invalid_list_shape_is_rejected_before_numpy_allocation(points, monkeypatch):
    def reject_allocation(*_args, **_kwargs):
        pytest.fail("invalid/over-cap positions were coerced before shape validation")

    monkeypatch.setattr(radius_module.np, "asarray", reject_allocation)
    with pytest.raises(ValueError):
        analyze(points)


def test_numeric_numpy_scalars_and_read_only_real_arrays_are_accepted():
    points = quarter_circle().astype(np.float64)
    points.flags.writeable = False
    result = analyze(points, axis=np.array([0, 1, 0], dtype=np.int64),
                     position_error_bound=np.float64(1e-7),
                     max_axial_gap=np.float64(1.01), max_radius_width=np.float64(0.1))
    assert result["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert_scope_and_json(result)


def test_injected_numerical_enclosure_disagreement_cannot_authorize_witness(monkeypatch):
    # Deliberately corrupt a necessary bound. This is targeted failure injection,
    # not an oracle for the actual radius-bound implementation tested above.
    monkeypatch.setattr(radius_module, "_radius_outer_interval",
                        lambda _points, _epsilon: (1e6, 1e6 + 1e-4, 3))
    result = analyze()
    assert result["circle_witness"]["compatible_with_error_bound"] is True
    assert result["status"] == "ABSTAINED"
    assert "NUMERICAL_ENCLOSURE_INCONSISTENT" in result["reasons"]
    assert_scope_and_json(result)


def test_optimizer_failure_preserves_possible_radius_and_does_not_invent_an_axis(monkeypatch):
    monkeypatch.setattr(radius_module, "_circle_witness", lambda _points: None)
    result = analyze()
    assert result["status"] == "ABSTAINED"
    assert "NO_VERIFIED_CIRCLE_WITNESS" in result["reasons"]
    assert result["circle_witness"] is None
    assert_radius_encloses(result, 5)
    assert len(result["axial_sample_groups"]) == 1
    for group in result["axial_sample_groups"]:
        assert group["axis_geometry_source"] == "UNAVAILABLE"
        assert group["axis_endpoints"] is None
    assert_scope_and_json(result)


@pytest.mark.parametrize("factor", [1e-300, 1e-150, 1e150, 1e300])
def test_large_and_small_finite_scales_preserve_the_dimensionless_circle(factor):
    result = analyze(quarter_circle() * factor, position_error_bound=1e-7 * factor,
                     max_axial_gap=1.01 * factor, max_radius_width=0.1 * factor)
    assert result["status"] == "SAMPLE_CIRCLE_WITNESS"
    assert_allclose(result["circle_witness"]["radius"] / factor, 5, atol=1e-9)
    normalized_lower, normalized_upper = np.asarray(result["radius_interval"]) / factor
    assert normalized_lower <= 5 + 1e-9
    assert normalized_upper >= 5 - 1e-9
    assert_scope_and_json(result)


@pytest.mark.parametrize("points,error_bound", [
    (np.array([[1e308, 0, 0], [-1e308, 0, 0], [0, 1, 1e308], [0, -1, -1e308]]), 0.0),
    (quarter_circle(), 1e308),
    (quarter_circle() * 1e-300, 0.001),
])
def test_unsupported_finite_numeric_range_abstains_with_valid_json(points, error_bound):
    result = analyze(points, position_error_bound=error_bound)
    assert result["status"] == "ABSTAINED"
    assert "NUMERICAL_RANGE_UNSUPPORTED" in result["reasons"]
    assert result["circle_witness"] is None
    assert_scope_and_json(result)
