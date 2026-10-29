"""Invented side-bound mathematics and scope controls, no physical inputs."""
import inspect
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from research.geometry.position_axis_hypothesis import _fit_plane
from research.geometry.tangent_plane_side_envelope import (
    _signed_cone_intervals, analyze_tangent_plane_sides,
)


def plane_grid(z=0.0):
    return np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1)], dtype=float)


def analyze(candidates, *, plane=None, **changes):
    policy = dict(unit="mm", position_error_bound=0.001, max_normal_angle_bound=0.1)
    policy.update(changes)
    return analyze_tangent_plane_sides(plane_grid() if plane is None else plane, candidates, **policy)


def internal_cone(alpha, normal=(0, 0, 1)):
    # Isolated analytic test of the PRIVATE cone formula, not a public verified-
    # plane input. z=0 observations are compatible with every stipulated cone
    # below because the exact true normal (0,0,1) is at its center.
    return dict(reasons=[], plane_witness=dict(normal=list(normal)),
                normal_angle_bound_radians=alpha, numerical_allowance=0.0)


def assert_scope(result):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    assert result["two_radius_or_multiple_surface_model_is_excluded"] is False
    assert result["tangency_is_verified"] is result["contact_is_verified"] is False
    assert result["policy"]["plane_fit_uses_original_error_bound"] is True
    assert result["policy"]["candidate_residual_selection_or_trimming"] is False
    envelope = result["side_envelope"]
    if envelope is not None:
        for key in ("two_radius_or_multiple_surface_model_is_excluded", "joint_halfspace_membership_is_verified",
                    "tangency_is_verified", "contact_is_verified", "physical_membership_is_verified",
                    "enclosure_is_numerically_certified"):
            assert envelope[key] is False
        assert len(envelope["signed_intervals"]) == len(result["candidate_positions"])
        positive, negative, unresolved = [set(envelope[key]) for key in (
            "robust_positive_indices", "robust_negative_indices", "unresolved_indices",
        )]
        assert not positive & negative and not positive & unresolved and not negative & unresolved
        assert positive | negative | unresolved == set(range(len(result["candidate_positions"])))
        possible_positive = set(envelope["positive_side_not_excluded_indices"])
        possible_negative = set(envelope["negative_side_not_excluded_indices"])
        assert possible_positive == positive | unresolved
        assert possible_negative == negative | unresolved
        assert result["single_same_face_tangent_cylinder_excluded"] == bool(positive and negative)


def test_public_signature_accepts_no_verified_plane_normals_or_face_labels():
    signature = inspect.signature(analyze_tangent_plane_sides)
    assert set(signature.parameters) == {
        "plane_region", "candidate_points", "unit", "position_error_bound", "max_normal_angle_bound",
    }
    assert all(p.default == inspect.Parameter.empty for p in signature.parameters.values())
    assert all(signature.parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
               for key in ("unit", "position_error_bound", "max_normal_angle_bound"))


def test_exact_cone_extrema_include_interior_and_boundary_attainers():
    alpha, beta = math.pi / 6, math.pi / 12
    p = np.array([[0, 0, 2], [0, 0, -2], [2, 0, 0],
                  [2 * math.sin(beta), 0, 2 * math.cos(beta)],
                  [-2 * math.sin(beta), 0, -2 * math.cos(beta)], [0, 0, 0]])
    result = _signed_cone_intervals(plane_grid(), p, internal_cone(alpha), 0)
    assert result is not None
    expected = [[math.sqrt(3), 2], [-2, -math.sqrt(3)], [-1, 1],
                [math.sqrt(2), 2], [-2, -math.sqrt(2)], [0, 0]]
    assert_allclose(result["normal_cone_signed_intervals"], expected, rtol=0, atol=2e-15)
    # The scaled/reference-anchored centroid need not reproduce mathematical
    # zero bit-for-bit. Use an independent machine-epsilon/input-scale tolerance,
    # not the implementation's own uncertainty guard; retain the extrema checks.
    centroid_roundoff = 8 * np.finfo(float).eps * np.max(np.abs(np.vstack((plane_grid(), p))))
    assert_allclose(result["observed_plane_centroid"], [0, 0, 0], rtol=0, atol=centroid_roundoff)
    assert result["candidate_order_is_preserved"] is True


@pytest.mark.parametrize("alpha", [0, 0.01, math.pi / 6, 1.4])
def test_signed_cone_ranges_contain_independent_normal_samples(alpha):
    p = np.array([[0, 0, 2], [0, 0, -2], [2, 0, 0], [0.2, -1.7, 0.4], [0, 0, 0]])
    envelope = _signed_cone_intervals(plane_grid(), p, internal_cone(alpha), 0.01)
    bounds = np.asarray(envelope["signed_intervals"])
    for polar in np.linspace(0, alpha, 8):
        for azimuth in np.linspace(0, 2 * math.pi, 17):
            normal = np.array([math.sin(polar) * math.cos(azimuth),
                               math.sin(polar) * math.sin(azimuth), math.cos(polar)])
            signed = p @ normal
            assert np.all(bounds[:, 0] <= signed)
            assert np.all(signed <= bounds[:, 1])


def test_zero_width_cone_and_sign_flip_preserve_intervals_with_reversed_sides():
    p = np.array([[2, -1, 0.5], [-2, 4, -0.3], [0, 0, 0]])
    one = _signed_cone_intervals(plane_grid(), p, internal_cone(0), 0.01)
    two = _signed_cone_intervals(plane_grid(), p, internal_cone(0, normal=(0, 0, -1)), 0.01)
    assert_allclose(one["normal_cone_signed_intervals"], np.column_stack((p[:, 2], p[:, 2])), atol=1e-15)
    assert_allclose(two["signed_intervals"], -np.asarray(one["signed_intervals"])[:, ::-1], atol=1e-15)
    assert one["robust_positive_indices"] == two["robust_negative_indices"]
    assert one["robust_negative_indices"] == two["robust_positive_indices"]


def test_boundary_nodes_are_retained_in_both_closed_halfspaces():
    result = analyze([[1, 0, 0], [0, 0, 1], [0, 0, -1]], position_error_bound=0)
    assert_scope(result)
    boundary = result["candidate_positions"].index([1.0, 0.0, 0.0])
    envelope = result["side_envelope"]
    assert boundary in envelope["unresolved_indices"]
    assert boundary in envelope["positive_side_not_excluded_indices"]
    assert boundary in envelope["negative_side_not_excluded_indices"]
    assert result["single_same_face_tangent_cylinder_excluded"] is True


def test_correlated_centroid_and_candidate_errors_require_the_full_two_epsilon():
    epsilon = 0.01
    result = analyze([[0, 0, -epsilon]], plane=plane_grid(z=epsilon), position_error_bound=epsilon)
    assert_scope(result)
    assert result["analysis_available"] is True
    envelope = result["side_envelope"]
    assert envelope["plane_centroid_position_error_bound"] == epsilon
    assert envelope["candidate_position_error_bound"] == epsilon
    assert envelope["combined_position_error_bound"] == 2 * epsilon
    assert_allclose(envelope["observed_plane_centroid"], [0, 0, epsilon], atol=0)
    assert envelope["signed_intervals"][0][0] <= 0 <= envelope["signed_intervals"][0][1]
    assert envelope["unresolved_indices"] == [0]
    # The fitted observed plane is epsilon above the true z=0 plane, while the
    # candidate is epsilon below it. Neither is an outlier under the premise.
    assert abs(result["candidate_positions"][0][2] - envelope["observed_plane_centroid"][2]) == 2 * epsilon


def test_plane_normal_uncertainty_covers_distant_same_plane_candidate():
    epsilon = 0.01
    observed_plane = plane_grid()
    observed_plane[:, 2] = epsilon / 2 + epsilon / 4 * observed_plane[:, 0]
    result = analyze([[10, 0, -epsilon]], plane=observed_plane, position_error_bound=epsilon)
    assert_scope(result)
    assert result["analysis_available"] is True
    lower, upper = result["side_envelope"]["signed_intervals"][0]
    assert lower <= 0 <= upper
    assert result["side_envelope"]["unresolved_indices"] == [0]


@pytest.mark.parametrize("seed", [0, 17, 81, 20260907])
@pytest.mark.parametrize("epsilon", [0, 0.001, 0.05])
def test_bounds_contain_known_signed_truth_under_bounded_plane_and_candidate_errors(seed, epsilon):
    rng = np.random.default_rng(seed)
    true_plane = plane_grid()
    observed_plane = true_plane.copy()
    observed_plane[:, 2] += rng.uniform(-0.2, 0.2, size=len(true_plane)) * epsilon
    true_candidates = rng.uniform(-2, 2, size=(20, 3))
    direction = rng.normal(size=true_candidates.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    observed_candidates = true_candidates + direction * 0.8 * epsilon
    result = analyze(observed_candidates, plane=observed_plane, position_error_bound=epsilon)
    assert_scope(result)
    assert result["analysis_available"] is True
    correspondence = {tuple(row): i for i, row in enumerate(observed_candidates)}
    sign = math.copysign(1, result["side_envelope"]["plane_normal"][2])
    for row, bounds in zip(result["candidate_positions"], result["side_envelope"]["signed_intervals"]):
        truth = sign * true_candidates[correspondence[tuple(row)], 2]
        assert bounds[0] <= truth <= bounds[1]


@pytest.mark.parametrize("seed", [0, 17, 20260907])
def test_true_same_face_tangent_circle_points_are_never_excluded_from_correct_side(seed):
    epsilon, radius = 0.001, 2.0
    rng = np.random.default_rng(seed)
    plane = plane_grid(z=radius)
    plane[:, 2] += rng.uniform(-0.2, 0.2, size=len(plane)) * epsilon
    true_circle = np.array([[radius * math.sin(theta), y, radius * math.cos(theta)]
                            for theta in np.linspace(0, math.pi, 7) for y in (-0.5, 0, 0.5)])
    direction = rng.normal(size=true_circle.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    observed = true_circle + 0.8 * epsilon * direction
    result = analyze(observed, plane=plane, position_error_bound=epsilon)
    assert_scope(result)
    assert result["analysis_available"] is True
    assert result["single_same_face_tangent_cylinder_excluded"] is False
    assert set(result["side_envelope"]["negative_side_not_excluded_indices"]) == set(range(len(observed)))


def test_robust_straddle_excludes_only_single_same_face_tangent_cylinder():
    result = analyze([[0, 0, -1], [0, 0, 1]])
    assert_scope(result)
    assert result["status"] == "SINGLE_SAME_FACE_TANGENT_CYLINDER_EXCLUDED"
    assert result["single_same_face_tangent_cylinder_excluded"] is True
    assert result["two_radius_or_multiple_surface_model_is_excluded"] is False


def test_valid_inner_outer_layers_can_straddle_inner_flange_without_mixture_rejection():
    candidates = np.array([[radius * math.sin(theta), y, radius * math.cos(theta)]
                           for radius in (2, 2.2) for theta in (0.1, 0.6, 1.1) for y in (-0.5, 0, 0.5)])
    result = analyze(candidates, plane=plane_grid(z=2), position_error_bound=1e-6)
    assert_scope(result)
    assert result["single_same_face_tangent_cylinder_excluded"] is True
    assert result["two_radius_or_multiple_surface_model_is_excluded"] is False
    assert_allclose(np.hypot(candidates[:9, 0], candidates[:9, 2]), 2, atol=1e-15)
    assert_allclose(np.hypot(candidates[9:, 0], candidates[9:, 2]), 2.2, atol=1e-15)
    assert "VALID_INNER_OUTER_MIXTURE_CAN_STRADDLE_AN_INNER_FLANGE_PLANE" in result["limitations"]


@pytest.mark.parametrize("center_height", [0, 3])
def test_one_sided_subset_does_not_prove_tangency_or_contact(center_height):
    # At center_height=0 the plane z=0 is SECANT to the full cylinder. At 3 it is
    # detached from a radius-1 cylinder. The sampled upper-quadrant arcs are all
    # on one side in either case, so this necessary test cannot establish contact.
    p = [[math.sin(theta), y, center_height + math.cos(theta)]
         for theta in (0.2, 0.7, 1.2) for y in (-0.5, 0, 0.5)]
    result = analyze(p, position_error_bound=1e-6)
    assert_scope(result)
    assert result["single_same_face_tangent_cylinder_excluded"] is False
    assert result["tangency_is_verified"] is result["contact_is_verified"] is False
    assert center_height != 1


def test_joint_incompatibility_of_chosen_normal_does_not_become_verified_membership():
    result = analyze([[4, 0, 0.001], [-4, 0, -0.001]], position_error_bound=0.01)
    assert_scope(result)
    envelope = result["side_envelope"]
    assert envelope["unresolved_indices"] == [0, 1]
    assert envelope["joint_halfspace_membership_is_verified"] is False


@pytest.mark.parametrize("seed", [0, 17, 20260907])
def test_public_permutation_and_exact_repeat_canonicalization(seed):
    plane = plane_grid()
    candidates = np.array([[1, 2, 0], [0, 0, 1], [-1, 0, -1]], dtype=float)
    reference = analyze(candidates, plane=plane)
    rng = np.random.default_rng(seed)
    assert analyze(candidates[rng.permutation(len(candidates))], plane=plane[rng.permutation(len(plane))]) == reference
    repeated = analyze(np.concatenate([candidates, candidates]), plane=np.concatenate([plane, plane]))
    assert repeated["input_count"] == 2 * reference["input_count"]
    for key in ("plane_positions", "candidate_positions", "plane_analysis", "side_envelope", "status"):
        assert repeated[key] == reference[key]


def test_private_helper_preserves_candidate_order_and_duplicate_identity():
    candidates = [[1, 0, 1], [-1, 0, -1], [1, 0, 1]]
    envelope = _signed_cone_intervals(plane_grid(), candidates, _fit_plane(plane_grid(), 0.001), 0.001)
    assert envelope["candidate_count"] == 3
    assert envelope["robust_positive_indices"] == [0, 2]
    assert envelope["robust_negative_indices"] == [1]
    assert envelope["signed_intervals"][0] == envelope["signed_intervals"][2]


def test_rigid_motion_and_units_preserve_signed_intervals_up_to_normal_display_sign():
    plane, p = plane_grid(), np.array([[1, 2, 0], [0, 0, 1], [-1, 0, -1]], dtype=float)
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, -1], [0.4, 3, 0.7], [0.5, -1, 4]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    moved_plane = (plane @ rotation.T + [4, -5, 7]) / 1000
    moved = (p @ rotation.T + [4, -5, 7]) / 1000
    original = analyze(p, plane=plane, position_error_bound=0.0001)
    transformed = analyze(moved, plane=moved_plane, unit="m", position_error_bound=1e-7)
    assert_scope(original)
    assert_scope(transformed)
    one = original["side_envelope"]
    two = transformed["side_envelope"]
    alignment = float(np.asarray(two["plane_normal"]) @ (rotation @ np.asarray(one["plane_normal"])))
    source_ids = {tuple(row): index for index, row in enumerate(p)}
    target_ids = {tuple(row): index for index, row in enumerate(moved)}
    original_bounds = {source_ids[tuple(row)]: interval for row, interval in
                       zip(original["candidate_positions"], one["signed_intervals"])}
    for row, interval in zip(transformed["candidate_positions"], two["signed_intervals"]):
        restored = np.asarray(interval) * 1000
        if alignment < 0:
            restored = -restored[::-1]
        assert_allclose(restored, original_bounds[target_ids[tuple(row)]], atol=2e-11, rtol=0)
    assert original["single_same_face_tangent_cylinder_excluded"] == transformed["single_same_face_tangent_cylinder_excluded"]


@pytest.mark.parametrize("field,value", [
    ("unit", "cm"), ("unit", None), ("position_error_bound", -1), ("position_error_bound", True),
    ("position_error_bound", math.inf), ("position_error_bound", math.nan),
    ("max_normal_angle_bound", 0), ("max_normal_angle_bound", math.pi / 2),
    ("max_normal_angle_bound", math.inf), ("max_normal_angle_bound", True),
])
def test_policies_validate_even_when_both_inputs_are_empty(field, value):
    with pytest.raises(ValueError):
        analyze([], plane=[], **{field: value})


def test_shared_raw_cap_is_checked_before_either_input_is_converted_or_deduplicated():
    class Oversize(list):
        def __len__(self):
            return 248

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversized shared input converted")

    with pytest.raises(ValueError, match="256"):
        analyze(Oversize(), plane=plane_grid())
    with pytest.raises(ValueError, match="256"):
        analyze([[0, 0, 1]] * 248, plane=plane_grid())
    permitted = analyze([[0, 0, 1]] * 247, plane=plane_grid())
    assert_scope(permitted)
    assert permitted["input_count"] == 256
    assert permitted["analysis_available"] is True


@pytest.mark.parametrize("plane", [None, [], [[0, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]])
def test_missing_or_rank_deficient_plane_yields_no_side_exclusion(plane):
    # Call public API directly because the convenience helper uses None as its
    # default invented plane, while the public API treats None as missing input.
    result = analyze_tangent_plane_sides(plane, [[0, 0, 1], [0, 0, -1]], unit="mm",
        position_error_bound=0.001, max_normal_angle_bound=0.1)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["analysis_available"] is False
    assert result["side_envelope"] is None


def test_noisy_nonplanar_union_is_not_trimmed_into_a_plane_before_side_analysis():
    plane = np.concatenate([plane_grid(z=-0.1), plane_grid(z=0.1)])
    result = analyze([[0, 0, 1]], plane=plane, position_error_bound=0.001)
    assert_scope(result)
    assert result["analysis_available"] is False
    assert "PLANE_NO_ALL_POINT_PLANE_WITNESS" in result["reasons"]
    assert result["side_envelope"] is None


def test_explicit_normal_precision_policy_can_abstain_without_changing_epsilon():
    result = analyze([[0, 0, 1]], position_error_bound=0.01, max_normal_angle_bound=0.0001)
    assert_scope(result)
    assert result["reasons"] == ["PLANE_NORMAL_PRECISION_NOT_ESTABLISHED"]
    assert result["side_envelope"] is None


@pytest.mark.parametrize("candidates", [None, [], np.empty((0, 3))])
def test_empty_candidate_set_is_not_an_empty_physical_part(candidates):
    result = analyze(candidates)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["analysis_available"] is True
    assert result["reasons"] == ["NO_CANDIDATE_POSITIONS"]
    assert result["side_envelope"]["signed_intervals"] == []


@pytest.mark.parametrize("points", [
    [[0, 0]], [[0, 0, "1"]], [[0, 0, True]], [[0, 0, math.nan]], [[0, 0, math.inf]],
])
def test_candidate_shape_and_numbers_are_strict(points):
    with pytest.raises(ValueError):
        analyze(points)


def test_true_overflowing_coordinate_range_abstains_without_nonfinite_payload():
    plane = [[-1.7e308, 0, 0], [1.7e308, 0, 0], [0, 1.7e308, 0]]
    result = analyze([[0, 0, 1]], plane=plane)
    assert_scope(result)
    assert result["analysis_available"] is False
    assert result["side_envelope"] is None


@pytest.mark.parametrize("analysis,epsilon", [
    (internal_cone(0, normal=(0, 0, 0)), 0),
    (internal_cone(math.pi / 2), 0),
    (internal_cone(0), 1e308),
    ({}, 0),
    (dict(internal_cone(0), reasons=["NO_ALL_POINT_PLANE_WITNESS"]), 0),
])
def test_private_unavailable_inputs_return_none_without_claimed_bounds(analysis, epsilon):
    assert _signed_cone_intervals(plane_grid(), [[0, 0, 1]], analysis, epsilon) is None
