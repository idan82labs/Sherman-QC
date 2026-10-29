"""Bounded invented-position checks; no physical fixtures or runtime imports."""
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from research.geometry import position_bend_candidates as module


def invented_cloud():
    # Independently specified integer/rational points on x²+z²=25 and its
    # two tangent flanges. Construction labels never enter the analyzer.
    return np.array(
        [(x, y, 5) for x in (-4, -2, 0) for y in (-1, 0, 1)]
        + [(5, y, z) for z in (-4, -2, 0) for y in (-1, 0, 1)]
        + [(x, y, z) for x, z in ((1.4, 4.8), (3, 4), (4, 3), (4.8, 1.4))
           for y in (-1, 0, 1)], dtype=float,
    )


def analyze(points=None, **changes):
    arguments = dict(
        unit="mm", position_error_bound=1e-6, neighborhood_radii=[2.5],
        max_normal_angle_bound=0.1, min_region_points=6, max_regions=64,
        sample_gap=2.1, max_axis_angle_bound=0.1, max_axial_gap=1.1,
        max_radius_width=0.1, max_pair_evaluations=2016, max_bridge_evaluations=256,
    )
    arguments.update(changes)
    return module.analyze_position_bend_candidates(
        invented_cloud() if points is None else points, **arguments,
    )


def check_scope(result):
    json.dumps(result, allow_nan=False)
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    for candidate in result["candidates"]:
        association = candidate["association"]
        assert association["physical_connectivity_is_verified"] is False
        assert association["joint_graph_realizability_is_verified"] is False
        assert association["same_object_is_verified"] is False
        assert association["tangency_is_verified"] is False
        assert association["entire_remainder_component_retained"] is True
        assert candidate["bridge_analysis"]["original_joint_compatibility"] == "UNESTABLISHED"


def test_unlabelled_rational_cloud_removes_supplied_plane_and_cylinder_region_oracles():
    points = invented_cloud()
    result = analyze(points)
    check_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["extended_regions"]) == 2
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert len(candidate["transition_point_indices"]) == 12
    assert candidate["bridge_analysis"]["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    lo, hi = candidate["bridge_analysis"]["radius_interval"]
    assert lo <= 5 <= hi
    assert hi-lo < 0.001
    canonical = np.asarray(result["positions"])
    transition = canonical[candidate["transition_point_indices"]]
    assert_allclose(transition[:, 0]**2+transition[:, 2]**2, 25)
    supports = [set(region["point_indices"]) for region in result["extended_regions"]]
    supports.append(set(candidate["transition_point_indices"]))
    assert sum(map(len, supports)) == len(set.union(*supports)) == len(points)
    assert_allclose(points, invented_cloud())


def test_exact_repeats_and_input_permutations_do_not_multiply_hypotheses():
    expected = analyze()
    points = invented_cloud()
    result = analyze(np.repeat(points[::-1], 3, axis=0))
    check_scope(result)
    assert result["input_count"] == 90
    for key in ("positions", "extended_regions", "candidates", "graph", "search_diagnostics"):
        assert result[key] == expected[key]


def test_components_use_induced_graph_not_an_incompatible_shortcut():
    adjacency = np.zeros((5, 5), bool)
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 4)):
        adjacency[first, second] = adjacency[second, first] = True
    assert module._components(adjacency, np.array([True, True, False, True, True])) == [(0, 1), (3, 4)]


def test_pairwise_possible_links_do_not_claim_joint_graph_realizability():
    g, epsilon, side = 1.0, 0.1, 1.2
    points = np.array([[0, 0, 0], [side, 0, 0], [side/2, math.sqrt(3)*side/2, 0]])
    result = analyze(points, neighborhood_radii=[2], min_region_points=3,
                     position_error_bound=epsilon, max_normal_angle_bound=0.5, sample_gap=g)
    check_scope(result)
    assert result["graph"]["edge_count"] == 3
    assert result["graph"]["joint_realizability_is_verified"] is False
    assert 3*side-3*math.sqrt(3)*epsilon > 3*g
    # Convex perimeter supporting planes give this lower bound for ANY allowed
    # perturbations, excluding simultaneous shortness despite pairwise edges.
    assert result["status"] == "ABSTAINED"


def test_disconnected_seed_ball_support_is_explicitly_split_not_merged():
    one = np.array([[x, y, 0] for x in (-0.5, 0.5) for y in (-0.5, 0.5)])
    points = np.concatenate((one, one+[5, 0, 0]))
    result = analyze(points, neighborhood_radii=[7], min_region_points=4, sample_gap=1.1)
    check_scope(result)
    assert result["search_diagnostics"]["split_seed_region_count"] == 1
    assert len(result["extended_regions"]) == 2
    assert all(len(region["point_indices"]) == 4 for region in result["extended_regions"])
    assert result["candidates"] == []


def test_subminimum_split_support_is_counted_and_not_a_plane_claim():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    result = analyze(points, neighborhood_radii=[2], min_region_points=3, sample_gap=0.1)
    check_scope(result)
    assert result["search_diagnostics"]["split_seed_region_count"] == 1
    assert result["search_diagnostics"]["subminimum_component_count"] == 3
    assert result["extended_regions"] == []


def test_fixed_seed_normal_extension_can_miss_other_compatible_true_plane_samples():
    # This explicitly records a recall limitation, not a feature to tune away.
    theta = 0.001
    normal = np.array([0, math.sin(theta), math.cos(theta)])
    true_points = np.array([[0, 10, 0], [0, -10, 0]], float)
    plane = {"plane_witness": {"reference_point": [0, 0, 0],
             "normal": normal.tolist(), "offset_from_reference": 0}, "numerical_allowance": 0}
    accepted = module._fixed_plane_membership(true_points, plane, epsilon=0.001, numerical=0)
    assert not accepted.any()
    assert "FIXED_SEED_PLANE_EXTENSION_CAN_MISS_COMPATIBLE_TRUE_PLANE_SAMPLES" in analyze()["limitations"]


@pytest.mark.parametrize("failure", ["NUMERICAL_RANGE_UNSUPPORTED", "PLANE_FIT_UNAVAILABLE"])
def test_unavailable_extension_analysis_is_incomplete_not_silent_missing_region(monkeypatch, failure):
    monkeypatch.setattr(module, "_fit_plane", lambda *args: {"reasons": [failure]})
    result = analyze()
    check_scope(result)
    assert result["reasons"] == ["EXTENSION_PLANE_ANALYSIS_UNAVAILABLE"]
    assert result["search_complete_within_declared_policy"] is False
    assert result["extended_regions"] == result["candidates"] == []


@pytest.mark.parametrize("reason", ["AXIS_FIRST_PLANE_FIT_UNAVAILABLE",
    "BRIDGE_NUMERICAL_RANGE_UNSUPPORTED", "RELAXED_RADIUS_NUMERICAL_ENCLOSURE_INCONSISTENT"])
def test_unavailable_bridge_analysis_clears_all_candidates(monkeypatch, reason):
    monkeypatch.setattr(module, "analyze_plane_conditioned_radius", lambda *args, **kwargs: {"reasons": [reason]})
    result = analyze()
    check_scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["candidates"] == result["extended_regions"] == []
    assert result["search_diagnostics"]["bridge_evaluation_count"] == 1
    assert result["reasons"] == ["BRIDGE_ANALYSIS_UNAVAILABLE"]


def test_distance_failure_abstains_without_nonfinite_evidence(monkeypatch):
    monkeypatch.setattr(module, "_distances", lambda *args: None)
    result = analyze()
    check_scope(result)
    assert result["reasons"] == ["ASSOCIATION_NUMERICAL_RANGE_UNSUPPORTED"]
    assert result["search_complete_within_declared_policy"] is False


def test_finite_adjacency_threshold_does_not_excuse_overflowing_link_interval():
    gap, uncertainty = 1.7e308, 5e306
    distance = gap+uncertainty
    assert math.isfinite(distance)
    assert module._sample_link_bound(distance, uncertainty, gap) is None
    assert module._sample_link_bound(1.0, 0.25, 1.5) == {
        "observed_min_distance": 1.0, "minimum_distance_interval": [0.75, 1.25],
        "all_error_models_have_a_sample_link_within_gap": True,
    }


def test_overflowing_link_bound_stops_before_bridge_evaluation(monkeypatch):
    monkeypatch.setattr(module, "_sample_link_bound", lambda *args: None)
    result = analyze()
    check_scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["search_diagnostics"]["bridge_evaluation_count"] == 0
    assert result["reasons"] == ["SAMPLE_LINK_BOUND_NUMERICAL_RANGE_UNSUPPORTED"]


def test_final_payload_guard_clears_an_unexpected_nonfinite_nested_field(monkeypatch):
    original = module.analyze_plane_conditioned_radius
    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        result["unexpected_nonfinite_diagnostic"] = float("nan")
        return result
    monkeypatch.setattr(module, "analyze_plane_conditioned_radius", inject)
    result = analyze()
    check_scope(result)
    assert result["search_complete_within_declared_policy"] is False
    assert result["candidates"] == []
    assert result["reasons"] == ["CANDIDATE_PAYLOAD_NUMERICAL_RANGE_UNSUPPORTED"]


@pytest.mark.parametrize("field,value", [
    ("sample_gap", 0), ("sample_gap", True), ("sample_gap", math.inf),
    ("max_axis_angle_bound", math.pi/2), ("max_axis_angle_bound", 0),
    ("max_axial_gap", -1), ("max_radius_width", "1"),
    ("max_pair_evaluations", 0), ("max_pair_evaluations", 2017),
    ("max_pair_evaluations", 1.5), ("max_pair_evaluations", True),
    ("max_bridge_evaluations", 0), ("max_bridge_evaluations", 257),
    ("max_bridge_evaluations", math.inf), ("max_bridge_evaluations", "256"),
])
def test_invalid_explicit_policies_rejected(field, value):
    with pytest.raises(ValueError):
        analyze(**{field: value})


def test_oversize_raw_cloud_rejected_before_numpy_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversize payload converted")
    monkeypatch.setattr(np, "asarray", forbidden)
    with pytest.raises(ValueError, match="256"):
        analyze([[0, 0, 0]]*257)


@pytest.mark.parametrize("points", [[], np.empty((0, 3)), [[1, 2, 3]]])
def test_missing_cloud_remains_unknown(points):
    result = analyze(points)
    check_scope(result)
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is True


def run_discovery_sweep(*, expanded=False):
    """48 default or 1,000 explicit invented 30-position trials, fixed policies.

    Noise and transformations are constructed independently, with truth retained
    only here. These are conditional tests, not a scanner simulator or accuracy
    estimate. Sparse support and fixed neighborhood scale remain stipulated.
    """
    rng = np.random.default_rng(20260907)
    count = 1000 if expanded else 48
    counts = {"cases": count, "positions_per_case": 30, "positive": 0,
              "abstained": 0, "incomplete": 0, "radius_intervals_checked": 0}
    for index in range(count):
        scale = 10**rng.uniform(-3, 3)
        epsilon = scale*10**rng.uniform(-7, -2)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        translation = rng.uniform(-5, 5, size=3)*scale
        truth = invented_cloud() @ rotation.T*scale+translation
        noise = rng.normal(size=truth.shape)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        noise *= rng.uniform(0, 0.5, size=(len(truth), 1))*epsilon
        observed = truth+noise
        assert np.max(np.linalg.norm(observed-truth, axis=1)) <= epsilon
        result = analyze(observed, position_error_bound=epsilon,
                         neighborhood_radii=[2.5*scale], sample_gap=2.1*scale,
                         max_axial_gap=1.1*scale, max_radius_width=0.1*scale)
        check_scope(result)
        counts["positive" if result["status"] == "POSITION_GEOMETRY_CANDIDATES" else "abstained"] += 1
        counts["incomplete"] += not result["search_complete_within_declared_policy"]
        canonical = np.asarray(result["positions"])
        for candidate in result["candidates"]:
            # Verify correspondence to the constructed transition BEFORE making
            # a true-radius containment assertion about this candidate model.
            support = canonical[candidate["transition_point_indices"]]
            indices = np.argmin(np.linalg.norm(support[:, None, :]-observed[None, :, :], axis=2), axis=1)
            if len(indices) == 12 and set(indices.tolist()) == set(range(18, 30)):
                bounds = candidate["bridge_analysis"]["radius_interval"]
                assert bounds[0] <= 5*scale+1e-9*scale
                assert bounds[1] is None or bounds[1] >= 5*scale-1e-9*scale
                counts["radius_intervals_checked"] += 1
    return counts


def test_fixed_seed_position_discovery_sweep():
    result = run_discovery_sweep()
    assert result["cases"] == 48
    assert result["positive"] > 0
    assert result["abstained"] > 0
    assert result["radius_intervals_checked"] > 40
