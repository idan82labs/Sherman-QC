"""Independent single-face tangency necessities on invented strips and arcs.

No engine construction labels, physical inputs, files, solver patches, or runtime
imports are used. Public checks never treat the algebraic center as ground truth.
Private center-ball checks compare its complete enclosure with a known circle.
"""
import json
import math

import numpy as np
import pytest

from research.geometry.plane_conditioned_radius import analyze_plane_conditioned_radius
from research.geometry.same_face_tangency import _center_ball, analyze_same_face_tangency
from research.geometry.tangent_plane_side_envelope import analyze_tangent_plane_sides


def plane_x(offset, axial_shift=0.0, tangent_shift=0.0):
    return np.array([[offset, y+axial_shift, z+tangent_shift]
                     for y in (-1., 0., 1.) for z in (-1., 0., 1.)])


def plane_z(offset, axial_shift=0.0, tangent_shift=0.0):
    return np.array([[x+tangent_shift, y+axial_shift, offset]
                     for y in (-1., 0., 1.) for x in (-1., 0., 1.)])


def partial_arc(radius=5.0):
    # Four exact 3-4-5 cross-section positions, repeated at two axial stations.
    # None is a contact point with the positive-x or positive-z tangent plane.
    section = np.array([[-5., 0.], [-4., -3.], [-3., -4.], [0., -5.]])*(radius/5)
    return np.array([[x, y, z] for x, z in section for y in (-.5, .5)])


def arguments(**changes):
    policy = dict(unit="mm", position_error_bound=1e-6, max_axis_angle_bound=.1,
                  max_axial_gap=2., max_radius_width=.25)
    policy.update(changes)
    return policy


def analyze(first, second, cylinder, **changes):
    return analyze_same_face_tangency(first, second, cylinder, **arguments(**changes))


def baseline(first, second, cylinder, **changes):
    return analyze_plane_conditioned_radius(first, second, cylinder, **arguments(**changes))


def scope(result):
    json.dumps(result, allow_nan=False)
    assert result["kind"] == "SAME_FACE_TANGENCY_NECESSARY_CHECK_V1"
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["two_radius_or_multiple_surface_model_is_excluded"] is False
    assert result["original_joint_compatibility"] == "UNESTABLISHED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    assert result["physical_axis_location_is_verified"] is False
    assert result["tangency_is_verified"] is False
    assert result["contact_is_verified"] is False
    assert result["policy"]["minimum_contact_sample_count"] == 0
    assert result["policy"]["candidate_residual_selection_or_trimming"] is False
    assert result["policy"]["centroid_error_averages_away"] is False
    assert result["policy"]["source_radius_fit_is_replaced"] is False
    assert result["policy"]["center_solver_limit"] == 1
    assert result["policy"]["flange_centroid_check_limit"] == 2
    assert len(result["flange_checks"]) <= 2
    if result["analysis_available"]:
        assert result["center_enclosure"] is not None
        assert len(result["flange_checks"]) == 2
        assert result["single_same_face_tangent_model_excluded"] == any(
            check["centroid_is_robustly_inside_every_admissible_disk"]
            for check in result["flange_checks"]
        )
        for check in result["flange_checks"]:
            assert check["centroid_original_position_error_bound"] == result["position_error_bound"]
            assert check["rotated_centroid_error_bound"] >= result["position_error_bound"]
            assert check["necessary_minimum_distance_for_tangency"] == result["radius_interval"][0]
    else:
        assert result["status"] == "ABSTAINED"
        assert result["single_same_face_tangent_model_excluded"] is False


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("angular_span", [.6, 1.7, 5.0])
def test_all_point_center_ball_contains_true_center_under_correlated_bounded_noise(seed, angular_span):
    rng = np.random.default_rng(seed)
    center = np.array([-.15, .2])
    radius, error, upper = .7, .002, .72
    angles = np.linspace(.2, .2+angular_span, 17)
    truth = center+radius*np.column_stack((np.cos(angles), np.sin(angles)))
    directions = rng.normal(size=truth.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Shared displacement is not reduced by averaging more samples.
    noise = error*(.3*directions*rng.uniform(0., 1., size=(len(truth), 1)) + [.45, .3])
    assert np.max(np.linalg.norm(noise, axis=1)) <= error
    observed = truth+noise
    discrepancy = np.abs(np.sum((observed-center)**2, axis=1)-radius**2)
    assert np.max(discrepancy) <= 2*upper*error+error**2
    result = _center_ball(observed, error, upper)
    assert result is not None
    json.dumps(result, allow_nan=False)
    assert result["all_point_count"] == len(observed)
    assert math.isclose(result["square_discrepancy_bound"], 2*upper*error+error**2)
    assert result["minimum_singular_value"] > result["singular_value_numerical_allowance"]
    assert np.linalg.norm(center-np.asarray(result["center"])) <= result["error_radius"]


def test_exact_partial_circle_has_a_padded_center_enclosure_not_an_asserted_exact_fit():
    center = np.array([.2, -.1])
    points = center+.7*np.array([[-1., 0.], [-.8, -.6], [-.6, -.8], [0., -1.]])
    result = _center_ball(points, 0., .7)
    assert result is not None
    assert result["square_discrepancy_bound"] == 0.
    assert result["error_radius"] > 0.
    assert result["residual_numerical_allowance"] > 0.
    assert np.linalg.norm(center-np.asarray(result["center"])) <= result["error_radius"]


@pytest.mark.parametrize("height", [0., 1e-14])
def test_rank_or_conditioning_unavailable_center_does_not_supply_false_precision(height):
    points = np.array([[-1., 0.], [0., height], [1., 0.]])
    assert _center_ball(points, 1e-6, 10.) is None


def test_fewer_than_three_projected_samples_have_no_center_enclosure():
    assert _center_ball(np.array([[0., 0.], [1., 1.]]), 1e-6, 5.) is None


def test_secant_plane_aliases_excluded_even_though_partial_circle_and_side_checks_pass():
    first, second, cylinder = plane_x(3.), plane_z(3.), partial_arc()
    radius = baseline(first, second, cylinder)
    assert radius["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    assert radius["radius_interval"][0] <= 5. <= radius["radius_interval"][1]
    for plane in (first, second):
        side = analyze_tangent_plane_sides(plane, cylinder, unit="mm", position_error_bound=1e-6,
                                          max_normal_angle_bound=.1)
        assert side["analysis_available"] is True
        assert side["single_same_face_tangent_cylinder_excluded"] is False
        assert len(side["side_envelope"]["robust_negative_indices"]) == len(cylinder)
    result = analyze(first, second, cylinder)
    scope(result)
    assert result["analysis_available"] is True
    assert result["status"] == "SINGLE_SAME_FACE_TANGENT_MODEL_EXCLUDED"
    assert result["single_same_face_tangent_model_excluded"] is True
    assert result["radius_interval"] == radius["radius_interval"]
    assert result["center_enclosure"]["all_point_count"] == 8
    assert all(check["all_point_count"] == 9 for check in result["flange_checks"])
    assert all(check["centroid_is_robustly_inside_every_admissible_disk"] for check in result["flange_checks"])
    assert all(check["true_projected_centroid_center_distance_upper_bound"] < 4.
               for check in result["flange_checks"])


def test_one_incompatible_flange_suffices_without_invalidating_the_cylinder_radius():
    first, second, cylinder = plane_x(3.), plane_z(5.), partial_arc()
    result = analyze(first, second, cylinder)
    scope(result)
    assert result["analysis_available"] is True
    assert result["single_same_face_tangent_model_excluded"] is True
    assert [check["centroid_is_robustly_inside_every_admissible_disk"]
            for check in result["flange_checks"]] == [True, False]
    assert result["radius_interval"][0] <= 5. <= result["radius_interval"][1]


def test_actual_tangent_planes_do_not_require_sampled_arc_endpoints_or_contact():
    first, second, cylinder = plane_x(5.), plane_z(5.), partial_arc()
    # All circle samples are far from both positive-side tangent contacts.
    assert np.min(np.abs(cylinder[:, 0]-5.)) >= 5.
    assert np.min(np.abs(cylinder[:, 2]-5.)) >= 5.
    result = analyze(first, second, cylinder)
    scope(result)
    assert result["analysis_available"] is True
    assert result["status"] == "TANGENCY_NOT_EXCLUDED"
    assert result["single_same_face_tangent_model_excluded"] is False
    assert all(check["true_projected_centroid_center_distance_upper_bound"] >= 5.
               for check in result["flange_checks"])


def test_long_axial_flange_lever_arm_receives_its_own_rotation_error_bound():
    cylinder = partial_arc()
    near = analyze(plane_x(5.), plane_z(5.), cylinder)
    first, second = plane_x(5., axial_shift=1e6), plane_z(5., axial_shift=-1e6)
    far = analyze(first, second, cylinder)
    source_radius = baseline(first, second, cylinder)
    scope(near)
    scope(far)
    assert source_radius["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    assert far["analysis_available"] is True
    assert far["single_same_face_tangent_model_excluded"] is False
    chord = source_radius["uncertainty_propagation"]["rotation_chord_bound"]
    for local, distant in zip(near["flange_checks"], far["flange_checks"]):
        lever = math.hypot(*distant["observed_centroid_offset_from_pivot"])
        expected = far["position_error_bound"] + chord*(lever+far["position_error_bound"])
        assert lever >= 1e6
        assert math.isclose(distant["rotated_centroid_error_bound"], expected, rel_tol=1e-12)
        assert distant["rotated_centroid_error_bound"] > 1000*local["rotated_centroid_error_bound"]
        assert distant["rotated_centroid_error_bound"] > .1


def test_outer_circle_and_inner_flange_exclude_only_the_same_face_relation():
    # x=5 and z=5 are tangent to a concentric inner radius-5 surface. Observing
    # the radius-6 outer surface does not make that multiface arrangement invalid.
    first, second, outer = plane_x(5.), plane_z(5.), partial_arc(6.)
    source_radius = baseline(first, second, outer)
    assert source_radius["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    result = analyze(first, second, outer)
    scope(result)
    assert result["analysis_available"] is True
    assert result["single_same_face_tangent_model_excluded"] is True
    assert result["two_radius_or_multiple_surface_model_is_excluded"] is False
    assert result["radius_interval"] == source_radius["radius_interval"]
    assert result["radius_interval"][0] <= 6. <= result["radius_interval"][1]
    assert "INNER_OUTER_MIXTURES_AND_MULTIPLE_FACES_NOT_EXCLUDED_BY_THIS_CHECK" in result["limitations"]


@pytest.mark.parametrize("first,second", [
    (plane_x(8.), plane_z(9.)),
    (plane_x(3., tangent_shift=10.), plane_z(3., tangent_shift=10.)),
])
def test_detached_planes_or_outside_centroids_can_remain_not_excluded(first, second):
    # The first pair misses the circle; the second consists of secant infinite
    # planes whose sampled patch centroids are outside the disk. Neither is a
    # same-face tangency witness, even though this necessary test cannot reject.
    result = analyze(first, second, partial_arc())
    scope(result)
    assert result["analysis_available"] is True
    assert result["status"] == "TANGENCY_NOT_EXCLUDED"
    assert result["single_same_face_tangent_model_excluded"] is False
    assert "OUTSIDE_CENTROIDS_CAN_HIDE_SECANT_PLANES_AND_DETACHED_CIRCLES" in result["limitations"]


def test_unbounded_collinear_radius_is_unavailable_not_an_excluded_tangency():
    cylinder = np.array([[-1., -.5, 0.], [0., 0., 0.], [1., .5, 0.]])
    result = analyze(plane_x(5.), plane_z(5.), cylinder)
    scope(result)
    assert result["radius_interval"][1] is None
    assert result["analysis_available"] is False
    assert result["center_enclosure"] is None
    assert result["flange_checks"] == []
    assert result["reasons"] == ["CONDITIONAL_SINGLE_RADIUS_REQUIRED"]


def test_nonconditional_finite_radius_policy_does_not_acquire_a_tangency_verdict():
    result = analyze(plane_x(3.), plane_z(3.), partial_arc(), max_radius_width=1e-12)
    scope(result)
    assert result["radius_interval"][1] is not None
    assert result["analysis_available"] is False
    assert result["reasons"] == ["CONDITIONAL_SINGLE_RADIUS_REQUIRED"]


def test_parallel_plane_axis_hypothesis_is_unavailable_not_a_model_exclusion():
    result = analyze(plane_x(5.), plane_x(6.), partial_arc())
    scope(result)
    assert result["analysis_available"] is False
    assert result["reasons"] == ["CONDITIONAL_SINGLE_RADIUS_REQUIRED"]


def test_permutation_and_exact_repeats_preserve_the_geometric_assessment():
    first, second, cylinder = plane_x(3.), plane_z(3.), partial_arc()
    original = analyze(first, second, cylinder)
    repeated = analyze(np.repeat(first[::-1], 2, axis=0), np.repeat(second[::-1], 2, axis=0),
                       np.repeat(cylinder[::-1], 2, axis=0))
    scope(original)
    scope(repeated)
    assert repeated["status"] == original["status"]
    assert repeated["radius_interval"] == original["radius_interval"]
    assert repeated["center_enclosure"] == original["center_enclosure"]
    assert repeated["flange_checks"] == original["flange_checks"]


@pytest.mark.parametrize("scale", [1e-3, 1., 1e3])
@pytest.mark.parametrize("offset,excluded", [(3., True), (5., False)])
@pytest.mark.parametrize("angle", [0., .37])
def test_rigid_transform_and_unit_scaling_preserve_exclusion_scope(scale, offset, excluded, angle):
    axis = np.array([1., 2., 3.])/math.sqrt(14)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                      [-axis[1], axis[0], 0.]])
    rotation = np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross @ cross)
    translation = np.array([2., -3., 1.])
    regions = [(region @ rotation.T+translation)*scale
               for region in (plane_x(offset), plane_z(offset), partial_arc())]
    result = analyze(*regions, unit="m" if scale == 1e-3 else "mm",
                     position_error_bound=1e-6*scale,
                     max_axial_gap=2.*scale, max_radius_width=.25*scale)
    scope(result)
    assert result["analysis_available"] is True
    assert result["single_same_face_tangent_model_excluded"] is excluded
    assert result["radius_interval"][0] <= 5.*scale <= result["radius_interval"][1]


@pytest.mark.parametrize("changes", [
    {"unit": "cm"}, {"unit": True}, {"unit": None},
    {"position_error_bound": -1.}, {"position_error_bound": float("nan")},
    {"position_error_bound": float("inf")}, {"position_error_bound": True},
    {"max_axis_angle_bound": 0.}, {"max_axis_angle_bound": math.pi/2},
    {"max_axis_angle_bound": True}, {"max_axial_gap": 0.},
    {"max_radius_width": 0.}, {"max_radius_width": None},
])
def test_invalid_explicit_units_and_policies_are_rejected(changes):
    with pytest.raises(ValueError):
        analyze(plane_x(5.), plane_z(5.), partial_arc(), **changes)


def test_shared_raw_cap_is_checked_before_exact_repeat_deduplication():
    with pytest.raises(ValueError, match="256"):
        analyze([[0., 0., 0.]]*85, [[0., 0., 0.]]*86, [[0., 0., 0.]]*86)


@pytest.mark.parametrize("bad", [
    [[0., 0.]], [[True, 0., 0.]], [["zero", 0., 0.]], [[float("inf"), 0., 0.]],
])
def test_malformed_or_nonfinite_positions_are_rejected(bad):
    with pytest.raises(ValueError):
        analyze(plane_x(5.), plane_z(5.), bad)


def test_missing_region_is_unavailable_not_a_missing_bend_or_defect():
    result = analyze(None, plane_z(5.), partial_arc())
    scope(result)
    assert result["analysis_available"] is False
    assert result["center_enclosure"] is None
    assert result["flange_checks"] == []
