"""Independent unlabeled synthetic oracles; no native/runtime/physical inputs.

The section equation and surface supports below are constructed directly here.
Their truth labels are used only by assertions, never passed to inference.
Synthetic policy choices are not physical threshold calibration.
"""
import inspect
import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pytest

from research.geometry import position_bend_candidates as candidate_module
from research.geometry.position_bend_candidates import analyze_position_bend_candidates


def quarter_bend():
    """R=2 quarter-cylinder, axis y, with two tangent planar flanges."""
    axial = np.linspace(-1.0, 1.0, 5)
    first = np.array([[x, y, 2.0] for x in np.linspace(-2.0, 0.0, 5) for y in axial])
    second = np.array([[2.0, y, z] for z in np.linspace(-2.0, 0.0, 5) for y in axial])
    transition = np.array([
        [2 * math.sin(j * math.pi / 16), y, 2 * math.cos(j * math.pi / 16)]
        for j in range(1, 8) for y in axial
    ])
    return first, second, transition


def cloud():
    return np.concatenate(quarter_bend())


def analyze(points=None, **changes):
    policy = dict(
        unit="mm", position_error_bound=1e-6, neighborhood_radii=[0.8],
        max_normal_angle_bound=0.1, min_region_points=9, max_regions=64,
        sample_gap=0.81, max_axis_angle_bound=0.1, max_axial_gap=0.51,
        max_radius_width=0.1, max_pair_evaluations=2016, max_bridge_evaluations=256,
    )
    policy.update(changes)
    return analyze_position_bend_candidates(cloud() if points is None else points, **policy)


def point_set(points):
    return frozenset(tuple(row) for row in np.round(np.asarray(points), 9))


def support_points(result, indices):
    return np.asarray(result["positions"])[indices]


def assert_scope(result):
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    assert "observed_count" not in result
    assert "formed" not in result
    assert "formation_state" not in result

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith(("is_verified", "are_verified", "is_physically_verified")):
                    assert item is False, key
                assert key not in {"axis_point", "axis_endpoints"}
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(result)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    if result["status"] == "POSITION_GEOMETRY_CANDIDATES":
        assert result["search_complete_within_declared_policy"] is True
        assert any(candidate["bridge_analysis"]["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
                   for candidate in result["candidates"])
    for candidate in result["candidates"]:
        association = candidate["association"]
        assert association["scope"] == "POSSIBLE_SAMPLED_GRAPH_ONLY"
        assert association["entire_remainder_component_retained"] is True
        assert candidate["bridge_analysis"]["original_joint_compatibility"] == "UNESTABLISHED"


def assert_radius_contains(candidate, radius):
    lower, upper = candidate["bridge_analysis"]["radius_interval"]
    assert lower <= radius
    assert upper is None or upper >= radius


def test_observation_channel_is_only_unlabelled_positions():
    signature = inspect.signature(analyze_position_bend_candidates)
    assert set(signature.parameters) == {
        "points", "unit", "position_error_bound", "neighborhood_radii",
        "max_normal_angle_bound", "min_region_points", "max_regions", "sample_gap",
        "max_axis_angle_bound", "max_axial_gap", "max_radius_width",
        "max_pair_evaluations", "max_bridge_evaluations",
    }
    assert signature.parameters["points"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY
               for name, parameter in signature.parameters.items() if name != "points")


def test_independent_quarter_section_discovers_flange_supports_and_complete_arc():
    first, second, transition = quarter_bend()
    # These analytic facts are test truth, absent from inference arguments.
    assert_allclose(transition[:, 0] ** 2 + transition[:, 2] ** 2, 4, rtol=0, atol=2e-15)
    assert np.all(first[:, 2] == 2)
    assert np.all(second[:, 0] == 2)
    result = analyze()
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert result["input_count"] == result["unique_count"] == 85
    assert len(result["extended_regions"]) == 2
    discovered = {point_set(support_points(result, region["point_indices"]))
                  for region in result["extended_regions"]}
    assert discovered == {point_set(first), point_set(second)}
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert point_set(support_points(result, candidate["transition_point_indices"])) == point_set(transition)
    assert_radius_contains(candidate, 2)
    assert candidate["bridge_analysis"]["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS"
    assert_allclose(candidate["bridge_analysis"]["relaxed_radius_diagnostic"]
                    ["circle_witness"]["radius"], 2, rtol=0, atol=1e-12)


def test_missing_transition_is_no_candidate_not_missing_or_unformed_bend():
    first, second, _ = quarter_bend()
    result = analyze(np.concatenate([first, second]))
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["extended_regions"]) == 2
    assert result["status"] == "ABSTAINED"
    assert result["candidates"] == []
    assert result["search_diagnostics"]["association_count"] == 0


def test_connected_off_cylinder_contaminant_is_retained_and_forces_abstention():
    _, _, transition = quarter_bend()
    contaminant = np.array([[2.4 / math.sqrt(2), 0, 2.4 / math.sqrt(2)]])
    result = analyze(np.concatenate([cloud(), contaminant]))
    assert_scope(result)
    assert result["search_complete_within_declared_policy"] is True
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    support = support_points(result, candidate["transition_point_indices"])
    assert point_set(support) == point_set(np.concatenate([transition, contaminant]))
    assert len(support) == 36
    assert candidate["bridge_analysis"]["status"] == "ABSTAINED"
    assert result["status"] == "ABSTAINED"
    assert "RELAXED_RADIUS_NO_VERIFIED_CIRCLE_WITNESS" in candidate["bridge_analysis"]["reasons"]


def test_disconnected_arc_cannot_enter_original_transition_component():
    _, _, transition = quarter_bend()
    disconnected = transition + [0, 3.5, 0]
    result = analyze(np.concatenate([cloud(), disconnected]))
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    selected = point_set(support_points(result, candidate["transition_point_indices"]))
    assert selected == point_set(transition)
    assert selected.isdisjoint(point_set(disconnected))
    assert_radius_contains(candidate, 2)


def test_sample_link_across_an_angular_hole_is_not_material_continuity():
    first, second, transition = quarter_bend()
    # Deliberately remove one angular column. The remaining two surface pieces
    # still have sample distances below the declared graph policy.
    remaining = transition[~np.isclose(transition[:, 0], math.sqrt(2))]
    result = analyze(np.concatenate([first, second, remaining]))
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert len(candidate["transition_point_indices"]) == 30
    assert candidate["association"]["physical_connectivity_is_verified"] is False
    assert candidate["bridge_analysis"]["hypothesis"]["physical_support_is_verified"] is False
    assert_radius_contains(candidate, 2)


def test_two_axial_sample_components_are_not_counted_as_two_physical_bends():
    first, second, transition = quarter_bend()
    remaining = transition[transition[:, 1] != 0]
    result = analyze(np.concatenate([first, second, remaining]))
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(result["candidates"]) == 2
    supports = [point_set(support_points(result, candidate["transition_point_indices"]))
                for candidate in result["candidates"]]
    assert supports[0].isdisjoint(supports[1])
    assert supports[0] | supports[1] == point_set(remaining)
    for candidate in result["candidates"]:
        assert len(candidate["transition_point_indices"]) == 14
        assert_radius_contains(candidate, 2)
    assert result["physical_bend_count"] is None


def test_two_cylindrical_generator_strips_are_also_exactly_planar():
    positions = np.array([
        [2 * math.cos(angle), y, 2 * math.sin(angle)]
        for angle in (-0.2, 0.2) for y in np.linspace(-1, 1, 5)
    ])
    assert_allclose(positions[:, 0] ** 2 + positions[:, 2] ** 2, 4, rtol=0, atol=2e-15)
    assert_allclose(positions[:, 0], 2 * math.cos(0.2), rtol=0, atol=0)
    result = analyze(positions, neighborhood_radii=[1.4], min_region_points=6)
    assert_scope(result)
    assert result["plane_discovery"]["regions"]
    assert len(result["extended_regions"]) == 1
    assert len(result["extended_regions"][0]["point_indices"]) == 10
    assert result["status"] == "ABSTAINED"
    assert result["candidates"] == []


@pytest.mark.parametrize("seed", [0, 1, 17, 4921, 20260907])
def test_row_permutation_cannot_choose_a_different_hypothesis_prefix(seed):
    points = cloud()
    permuted = points[np.random.default_rng(seed).permutation(len(points))]
    assert analyze(permuted) == analyze(points)


def test_exact_duplicates_do_not_add_geometric_support():
    points = cloud()
    result = analyze(np.concatenate([points, points[::-1]]))
    reference = analyze(points)
    assert_scope(result)
    assert result["input_count"] == 170
    assert result["unique_count"] == 85
    for key in ("positions", "extended_regions", "candidates", "search_diagnostics", "status"):
        assert result[key] == reference[key]


def rotation(quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@pytest.mark.parametrize("quaternion,translation", [
    ([1.0, 2.0, 3.0, 4.0], [4.0, -3.0, 7.0]),
    ([0.2, -0.3, 0.1, 0.8], [-8.0, 11.0, 6.0]),
    ([0.0, 1.0, 0.0, 0.0], [100.0, -120.0, 80.0]),
])
def test_rigid_motion_preserves_supports_and_true_radius(quaternion, translation):
    transform = rotation(np.asarray(quaternion))
    translation = np.asarray(translation)
    first, second, transition = quarter_bend()
    result = analyze(cloud() @ transform.T + translation)
    assert_scope(result)
    assert result["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(result["extended_regions"]) == 2
    restored = (np.asarray(result["positions"]) - translation) @ transform
    regions = {point_set(restored[region["point_indices"]]) for region in result["extended_regions"]}
    assert regions == {point_set(first), point_set(second)}
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert point_set(restored[candidate["transition_point_indices"]]) == point_set(transition)
    assert_radius_contains(candidate, 2)
    estimated = np.asarray(candidate["bridge_analysis"]["hypothesis"]["axis"])
    assert_allclose(np.outer(estimated, estimated), np.outer(transform[:, 1], transform[:, 1]),
                    rtol=0, atol=1e-12)


def test_unit_conversion_scales_every_length_policy_not_angular_policy():
    millimeters = analyze()
    meters = analyze(
        cloud() / 1000, unit="m", position_error_bound=1e-9,
        neighborhood_radii=[0.0008], sample_gap=0.00081,
        max_axial_gap=0.00051, max_radius_width=0.0001,
    )
    assert_scope(meters)
    assert meters["status"] == millimeters["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(meters["candidates"]) == len(millimeters["candidates"]) == 1
    for key in ("plane_region_ids", "transition_point_indices"):
        assert meters["candidates"][0][key] == millimeters["candidates"][0][key]
    assert_radius_contains(meters["candidates"][0], 0.002)
    assert_allclose(np.array(meters["candidates"][0]["bridge_analysis"]["radius_interval"]) * 1000,
                    millimeters["candidates"][0]["bridge_analysis"]["radius_interval"],
                    rtol=0, atol=1e-10)


@pytest.mark.parametrize("changes,reason", [
    ({"max_regions": 35}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_pair_evaluations": 5}, "PAIR_EVALUATION_BUDGET_EXCEEDED"),
    ({"max_bridge_evaluations": 1}, "BRIDGE_EVALUATION_BUDGET_EXCEEDED"),
])
def test_exhaustion_clears_candidates_and_preflights_before_any_bridge(monkeypatch, changes, reason):
    points = np.concatenate([cloud(), cloud() + [10, 0, 0]])
    reference = analyze(points)
    assert reference["status"] == "POSITION_GEOMETRY_CANDIDATES"
    assert len(reference["candidates"]) == 2
    assert reference["search_diagnostics"]["association_count"] == 2
    calls = []

    def forbidden_bridge(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("No bridge should run before the entire search fits its declared caps")

    monkeypatch.setattr(candidate_module, "analyze_plane_conditioned_radius", forbidden_bridge)
    result = analyze(points, **changes)
    assert_scope(result)
    assert calls == []
    assert result["status"] == "ABSTAINED"
    assert result["search_complete_within_declared_policy"] is False
    assert result["candidates"] == []
    assert result["extended_regions"] == []
    assert result["search_diagnostics"]["bridge_evaluation_count"] == 0
    assert result["reasons"] == [reason]


def test_oversize_duplicate_payload_is_rejected_before_deduplication():
    with pytest.raises(ValueError, match="at most 256 input positions"):
        analyze(np.tile([[0.0, 0.0, 0.0]], (257, 1)))


def signed_tangent_bend(angle_degrees, radius, *, flange_extent=1.0):
    """Direct signed-angle construction; labels remain assertion-only truth.

    Travel starts at (0, R), tangent to z=R. The start flange extends
    backwards along the signed start tangent; the end flange continues along
    the signed end tangent. Seven interior stations never include the seams.
    """
    angle = math.radians(angle_degrees)
    direction = math.copysign(1.0, angle)
    axial = radius * np.linspace(-0.5, 0.5, 5)
    stations = radius * np.linspace(0, flange_extent, 5)
    first = np.array([[-direction * distance, y, radius]
                      for distance in stations for y in axial])
    second = np.array([
        [radius * math.sin(angle) + direction * distance * math.cos(angle), y,
         radius * math.cos(angle) - direction * distance * math.sin(angle)]
        for distance in stations for y in axial
    ])
    transition = np.array([
        [radius * math.sin(j * angle / 8), y, radius * math.cos(j * angle / 8)]
        for j in range(1, 8) for y in axial
    ])
    return first, second, transition


def _assert_entire_graph_component(result, candidate):
    """Independently recover the induced remainder component, with no trimming."""
    points = np.asarray(result["positions"])
    regions = {region["region_id"]: set(region["point_indices"])
               for region in result["extended_regions"]}
    first, second = [regions[identifier] for identifier in candidate["plane_region_ids"]]
    support = set(candidate["transition_point_indices"])
    assert first.isdisjoint(second)
    assert support.isdisjoint(first | second)
    assert support <= set(range(len(points)))
    remaining = set(range(len(points))) - first - second
    threshold = result["graph"]["edge_distance_threshold"]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    reached, frontier = set(), {min(support)}
    while frontier:
        reached.update(frontier)
        adjacent = set()
        for index in frontier:
            adjacent.update(np.flatnonzero(distances[index] <= threshold).tolist())
        frontier = (adjacent & remaining) - reached
    assert reached == support
    for plane in (first, second):
        assert float(np.min(distances[np.ix_(sorted(support), sorted(plane))])) <= threshold


def run_structural_discovery_matrix():
    """Run 120 fixed-policy synthetic cases and return stratified aggregate counts.

    These are hypothesis-search outcomes, never physical detection accuracy.
    The clean 30 cases isolate actual signed bend angle and radius scale. Three
    separately reported stress strata use precisely the same policies: remove
    the middle angular column, mix in a +10% concentric radius, or spread the
    five flange stations over 2R instead of R. No per-angle tuning occurs.

    Containment is checked only for candidates whose two recovered supports are
    the constructed true flanges and whose transition consists exclusively of
    true radius-R positions. Mixtures do not satisfy that single-cylinder model.
    """
    strata = ("clean", "missing_middle_arc", "concentric_mixture", "sparse_flange_stations")
    counts = {
        name: dict(cases=0, candidate_cases=0, abstained_cases=0,
                   incomplete_search_cases=0, budget_failure_cases=0,
                   no_plane_region_cases=0, candidates=0, conditional_candidates=0,
                   abstained_bridges=0, matched_joint_model_candidates=0,
                   containment_checks=0, containment_failures=0)
        for name in strata
    }
    outcomes_by_absolute_angle = {
        name: {angle: dict(candidate_cases=0, abstained_cases=0)
               for angle in (30, 60, 90, 120, 150)}
        for name in strata
    }
    for name in strata:
        for angle in (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150):
            for radius in (0.5, 2.0, 20.0):
                first, second, inner = signed_tangent_bend(
                    angle, radius, flange_extent=2.0 if name == "sparse_flange_stations" else 1.0,
                )
                assert_allclose(inner[:, 0] ** 2 + inner[:, 2] ** 2, radius ** 2,
                                rtol=2e-15, atol=0)
                transition = inner
                if name == "missing_middle_arc":
                    # Delete station j=4, not a data-dependent residual selection.
                    transition = np.concatenate([inner[:15], inner[20:]])
                elif name == "concentric_mixture":
                    outer = inner * [1.1, 1.0, 1.1]
                    transition = np.concatenate([inner, outer])
                points = np.concatenate([first, second, transition])
                assert len(points) <= 256
                result = analyze(
                    points, position_error_bound=5e-7 * radius,
                    neighborhood_radii=[0.4 * radius], sample_gap=0.405 * radius,
                    max_axial_gap=0.255 * radius, max_radius_width=0.05 * radius,
                )
                assert_scope(result)
                assert point_set(result["positions"]) == point_set(points)
                current = counts[name]
                current["cases"] += 1
                positive = result["status"] == "POSITION_GEOMETRY_CANDIDATES"
                outcome = "candidate_cases" if positive else "abstained_cases"
                current[outcome] += 1
                outcomes_by_absolute_angle[name][abs(angle)][outcome] += 1
                current["incomplete_search_cases"] += int(
                    not result["search_complete_within_declared_policy"]
                )
                current["budget_failure_cases"] += int(any(
                    "BUDGET" in reason for reason in
                    result["reasons"] + result["plane_discovery"]["reasons"]
                ))
                current["no_plane_region_cases"] += int(not result["plane_discovery"]["regions"])
                current["candidates"] += len(result["candidates"])
                regions = {region["region_id"]: point_set(support_points(result, region["point_indices"]))
                           for region in result["extended_regions"]}
                true_flanges = {point_set(first), point_set(second)}
                true_cylinder_positions = point_set(inner)
                for candidate in result["candidates"]:
                    _assert_entire_graph_component(result, candidate)
                    bridge = candidate["bridge_analysis"]
                    if bridge["status"] == "CONDITIONAL_RADIUS_HYPOTHESIS":
                        current["conditional_candidates"] += 1
                    else:
                        current["abstained_bridges"] += 1
                    planes_match = {regions[index] for index in candidate["plane_region_ids"]} == true_flanges
                    selected = point_set(support_points(result, candidate["transition_point_indices"]))
                    if planes_match and selected <= true_cylinder_positions:
                        current["matched_joint_model_candidates"] += 1
                        current["containment_checks"] += 1
                        lower, upper = bridge["radius_interval"]
                        if not (lower <= radius and (upper is None or upper >= radius)):
                            current["containment_failures"] += 1
    return {"cases": sum(values["cases"] for values in counts.values()),
            "strata": counts, "outcomes_by_absolute_angle": outcomes_by_absolute_angle}


def test_fixed_structural_matrix_preserves_coverage_and_failure_receipt():
    receipt = run_structural_discovery_matrix()
    assert receipt["cases"] == 120
    common = dict(cases=30, incomplete_search_cases=0, budget_failure_cases=0,
                  containment_failures=0)
    assert receipt["strata"] == {
        "clean": dict(common, candidate_cases=30, abstained_cases=0,
                      no_plane_region_cases=0, candidates=30, conditional_candidates=30,
                      abstained_bridges=0, matched_joint_model_candidates=30, containment_checks=30),
        "missing_middle_arc": dict(
            common, candidate_cases=18, abstained_cases=12, no_plane_region_cases=0,
            candidates=18, conditional_candidates=18, abstained_bridges=0,
            matched_joint_model_candidates=18, containment_checks=18,
        ),
        "concentric_mixture": dict(
            common, candidate_cases=0, abstained_cases=30, no_plane_region_cases=0,
            candidates=30, conditional_candidates=0, abstained_bridges=30,
            matched_joint_model_candidates=0, containment_checks=0,
        ),
        "sparse_flange_stations": dict(
            common, candidate_cases=0, abstained_cases=30, no_plane_region_cases=30,
            candidates=0, conditional_candidates=0, abstained_bridges=0,
            matched_joint_model_candidates=0, containment_checks=0,
        ),
    }
    for angle in (30, 60, 90, 120, 150):
        assert receipt["outcomes_by_absolute_angle"]["clean"][angle] == {
            "candidate_cases": 6, "abstained_cases": 0,
        }
        # The fixed gap bridges the removed angular column only at <=90 degrees.
        # These positives remain conditional: graph links do not fill material.
        positive = 6 if angle <= 90 else 0
        assert receipt["outcomes_by_absolute_angle"]["missing_middle_arc"][angle] == {
            "candidate_cases": positive, "abstained_cases": 6 - positive,
        }
        for name in ("concentric_mixture", "sparse_flange_stations"):
            assert receipt["outcomes_by_absolute_angle"][name][angle] == {
                "candidate_cases": 0, "abstained_cases": 6,
            }
    json.dumps(receipt, allow_nan=False)
