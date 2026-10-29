"""Two predeclared graph hypotheses on the unchanged invented 300-case matrix.

No physical data or tuning: same rows, noise seed, k, cap, epsilon and work limits.
Full constructed support matching is an oracle only, never an inference input.
"""
from collections import Counter
import inspect
import json
import math

import numpy as np
import pytest

from research.geometry.graph_envelope_bend_candidates import analyze_graph_envelope_bend_candidates
from tests.test_adaptive_bend_candidates import synthetic_bend
from tests.test_cone_closed_bend_candidates import analyze as baseline_analyze


STRATA = ("clean", "missing_middle_arc", "concentric_mixture",
          "sparse_flange_stations", "sparse_concentric_mixture")
CONDITIONAL_STATUSES = {"CONDITIONAL_SINGLE_RADIUS_HYPOTHESIS", "CONDITIONAL_TWO_RADIUS_HYPOTHESIS"}


def analyze(points=None, *, scale=1.0, symmetry="mutual", **changes):
    policy = dict(
        unit="mm", position_error_bound=1e-6*scale, neighbor_counts=[8],
        max_neighborhood_radius=1.5*scale, max_normal_angle_bound=0.1,
        min_region_points=9, max_regions=64, graph_neighbor_count=10,
        max_link_distance=1.5*scale, max_axis_angle_bound=0.1,
        max_axial_gap=0.51*scale, max_radius_width=0.1*scale,
        max_pair_evaluations=2016, max_bridge_evaluations=256,
        min_band_points=9, max_split_evaluations=255, max_plane_evaluations=2048,
        graph_symmetry=symmetry,
    )
    policy.update(changes)
    points = np.concatenate(synthetic_bend())*scale if points is None else points
    return analyze_graph_envelope_bend_candidates(points, **policy)


def assert_scope(result):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["production_disposition"] == "NOT_AUTHORIZED"
    assert result["physical_bend_count"] is None
    assert result["physical_geometry_is_verified"] is False
    assert result["graph"]["edge_field_selects"] == "UPPER_PROPOSAL_GRAPH"
    assert result["graph"]["measurement_error_bound_is_relaxed"] is False
    assert result["plane_discovery"]["policy"]["final_error_bound_is_relaxed"] is False
    for candidate in result["candidates"]:
        association = candidate["association"]
        assert association["scope"] == "PAIRWISE_INTERVAL_RANK_RELAXATION_ONLY"
        assert association["entire_remainder_component_retained"] is True
        for key in ("physical_connectivity_is_verified", "joint_graph_realizability_is_verified",
                    "tangency_is_verified", "same_object_is_verified", "relaxed_links_establish_true_gap"):
            assert association[key] is False
        for analysis in (candidate["single_radius_analysis"], candidate["two_radius_analysis"]):
            if analysis is not None:
                assert analysis["original_joint_compatibility"] == "UNESTABLISHED"
        if candidate["two_radius_analysis"] is not None:
            assert candidate["two_radius_analysis"]["thickness"] is None
            assert candidate["two_radius_analysis"]["physical_face_count"] is None


def assert_whole_component(result, candidate):
    """Independent BFS on serialized upper edges; no discovery helper reuse."""
    regions = {r["region_id"]: set(r["point_indices"]) for r in result["extended_regions"]}
    first, second = [regions[index] for index in candidate["plane_region_ids"]]
    support = set(candidate["transition_point_indices"])
    assert first.isdisjoint(second) and support.isdisjoint(first | second)
    remaining = set(range(result["unique_count"])) - first - second
    adjacency = {index: set() for index in range(result["unique_count"])}
    for one, two in result["graph"]["edges"]:
        adjacency[one].add(two)
        adjacency[two].add(one)
    reached, frontier = set(), {min(support)}
    while frontier:
        reached.update(frontier)
        frontier = set.union(*(adjacency[index] for index in frontier)) & remaining - reached
    assert reached == support
    for region, count in zip((first, second), candidate["association"]["plane_relaxed_graph_edge_counts"]):
        assert sum(len(adjacency[index] & region) for index in support) == count > 0


def matched_candidates(observed, result, flange_sizes):
    original = {tuple(row): index for index, row in enumerate(observed)}
    assert len(original) == len(observed)
    ids = [original[tuple(row)] for row in result["positions"]]
    regions = {r["region_id"]: frozenset(ids[i] for i in r["point_indices"])
               for r in result["extended_regions"]}
    first_size, second_size = flange_sizes
    start = first_size + second_size
    flanges = {frozenset(range(first_size)), frozenset(range(first_size, start))}
    transition = frozenset(range(start, len(observed)))
    matches = [c for c in result["candidates"]
               if {regions[i] for i in c["plane_region_ids"]} == flanges
               and frozenset(ids[i] for i in c["transition_point_indices"]) == transition]
    return matches, ids


def run_matrix():
    rng = np.random.default_rng(2026090717)
    receipts, summaries = [], {}
    for name in STRATA:
        mixture = name.endswith("concentric_mixture")
        for angle in (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150):
            for radius in (0.5, 2, 20):
                first, second, transition = synthetic_bend(
                    angle, radius, flange_extent=2 if name.startswith("sparse_") else 1,
                    mixture=mixture,
                )
                if name == "missing_middle_arc":
                    transition = np.concatenate([transition[:15], transition[20:]])
                exact = np.concatenate([first, second, transition])
                epsilon, scale = 1e-6*radius/2, radius/2
                direction = rng.normal(size=exact.shape)
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                noisy = exact + direction*rng.uniform(0.25, 0.95, size=(len(exact), 1))*epsilon
                error = np.linalg.norm(noisy-exact, axis=1)
                assert np.all((error > 0) & (error <= epsilon))
                for noise, observed in (("exact", exact), ("bounded_noise", noisy)):
                    baseline = baseline_analyze(observed, scale=scale)
                    old_matches, _ = matched_candidates(observed, baseline, (len(first), len(second)))
                    old_conditional = any(c["status"] in CONDITIONAL_STATUSES for c in old_matches)
                    for symmetry in ("mutual", "union"):
                        result = analyze(observed, scale=scale, symmetry=symmetry)
                        assert_scope(result)
                        for candidate in result["candidates"]:
                            assert_whole_component(result, candidate)
                        matches, ids = matched_candidates(observed, result, (len(first), len(second)))
                        count = Counter(
                            cases=1, associations=len(result["candidates"]),
                            matched_associations=len(matches), unmatched_associations=len(result["candidates"])-len(matches),
                            matched_cases=int(bool(matches)),
                            conditional_matched_cases=int(any(c["status"] in CONDITIONAL_STATUSES for c in matches)),
                            conditional_candidates=sum(c["status"] in CONDITIONAL_STATUSES for c in result["candidates"]),
                            incomplete_cases=int(not result["search_complete_within_declared_policy"]),
                            baseline_associations=len(baseline["candidates"]),
                            baseline_matched_cases=int(bool(old_matches)),
                            baseline_conditional_matched_cases=int(old_conditional),
                            lost_baseline_matched_cases=int(bool(old_matches) and not matches),
                            lost_baseline_conditional_cases=int(old_conditional and not any(c["status"] in CONDITIONAL_STATUSES for c in matches)),
                            checked_single_intervals=0, checked_band_intervals=0,
                            unmatched_band_intervals=0, containment_failures=0,
                            all_proposed_single_intervals=sum(c["single_radius_analysis"]["radius_interval"] is not None for c in result["candidates"]),
                            all_proposed_band_intervals=sum(len(m["bands"]) for c in result["candidates"]
                                if c["two_radius_analysis"] is not None for m in c["two_radius_analysis"]["models"]),
                        )
                        for candidate in matches:
                            single, two = candidate["single_radius_analysis"], candidate["two_radius_analysis"]
                            if not mixture and single["radius_interval"] is not None:
                                lower, upper = single["radius_interval"]
                                count["checked_single_intervals"] += 1
                                count["containment_failures"] += int(not (lower <= radius and (upper is None or radius <= upper)))
                            if two is not None:
                                original = {tuple(row): i for i, row in enumerate(observed)}
                                band_ids = [original[tuple(row)] for row in two["mixed_positions"]]
                                start = len(first) + len(second)
                                middle = start + len(transition)//2
                                for model in two["models"]:
                                    for band in model["bands"]:
                                        layer = frozenset(band_ids[i] for i in band["point_indices"])
                                        truth = {frozenset(range(start, middle)): radius,
                                                 frozenset(range(middle, len(observed))): 1.1*radius} if mixture else {}
                                        if layer not in truth:
                                            continue
                                        lower, upper = band["radius_interval"]
                                        count["checked_band_intervals"] += 1
                                        count["containment_failures"] += int(not (lower <= truth[layer] and (upper is None or truth[layer] <= upper)))
                        count["unmatched_band_intervals"] = count["all_proposed_band_intervals"]-count["checked_band_intervals"]
                        key = (noise, symmetry, name)
                        summaries.setdefault(key, Counter()).update(count)
                        receipts.append(dict(stratum=name, noise=noise, symmetry=symmetry,
                                             angle=angle, radius=radius, **dict(count),
                                             final_regions=len(result["extended_regions"]),
                                             plane_evaluations=result["plane_discovery"]["diagnostics"]["plane_evaluation_count"],
                                             reasons=result["reasons"]))
    return dict(receipts=receipts, summaries={"/".join(k): dict(v) for k, v in summaries.items()})


@pytest.fixture(scope="module")
def matrix():
    return run_matrix()


def test_fixed_matrix_denominators_and_paired_baseline_retention(matrix):
    assert len(matrix["receipts"]) == 600
    for counts in matrix["summaries"].values():
        assert counts["cases"] == 30
        assert counts["associations"] == counts["matched_associations"] + counts["unmatched_associations"]
        assert counts["lost_baseline_matched_cases"] == counts["lost_baseline_conditional_cases"] == 0
        assert counts["incomplete_cases"] == counts["containment_failures"] == 0
    print("GRAPH_ENVELOPE_MATRIX=" + json.dumps(matrix["summaries"], sort_keys=True))
    print("GRAPH_ENVELOPE_WORK=" + json.dumps({
        "max_plane_evaluations": max(c["plane_evaluations"] for c in matrix["receipts"]),
        "max_final_regions": max(c["final_regions"] for c in matrix["receipts"]),
    }, sort_keys=True))


def test_fixed_matrix_uses_same_bounded_work_and_error(matrix):
    assert max(c["plane_evaluations"] for c in matrix["receipts"]) <= 2048
    assert max(c["final_regions"] for c in matrix["receipts"]) <= 64
    assert all(c["unmatched_band_intervals"] == 0 for c in matrix["receipts"])


@pytest.mark.parametrize("symmetry,expected", [("mutual", [30, 30, 30, 24, 18]),
                                               ("union", [30, 30, 30, 30, 30])])
def test_recorded_first_graph_comparison_is_development_regression_only(matrix, symmetry, expected):
    for noise in ("exact", "bounded_noise"):
        counts = [matrix["summaries"][f"{noise}/{symmetry}/{name}"] for name in STRATA]
        assert [c["conditional_matched_cases"] for c in counts] == expected
        assert sum(c["checked_single_intervals"] for c in counts) == (84 if symmetry == "mutual" else 90)
        assert sum(c["checked_band_intervals"] for c in counts) == (96 if symmetry == "mutual" else 120)
        assert sum(c["all_proposed_band_intervals"] for c in counts) == sum(c["checked_band_intervals"] for c in counts)


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
@pytest.mark.parametrize("mixture", [False, True])
def test_rigid_motion_and_unit_scale_preserve_full_support_enclosures(symmetry, mixture):
    points = np.concatenate(synthetic_bend(mixture=mixture))
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, -1], [0.4, 3, 0.7], [0.5, -1, 4]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    moved = (points @ rotation.T + [4, -5, 7])/1000
    result = analyze(moved, symmetry=symmetry, scale=0.001, unit="m")
    assert_scope(result)
    matches, _ = matched_candidates(moved, result, (25, 25))
    assert len(matches) == 1
    candidate = matches[0]
    assert_whole_component(result, candidate)
    if mixture:
        for band, radius in zip(candidate["two_radius_analysis"]["models"][0]["bands"], (0.002, 0.0022)):
            assert band["radius_interval"][0] <= radius <= band["radius_interval"][1]
    else:
        lower, upper = candidate["single_radius_analysis"]["radius_interval"]
        assert lower <= 0.002 <= upper


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
@pytest.mark.parametrize("mixture", [False, True])
def test_full_support_model_and_permutation(symmetry, mixture):
    points = np.concatenate(synthetic_bend(mixture=mixture))
    result = analyze(points, symmetry=symmetry)
    assert_scope(result)
    matches, _ = matched_candidates(points, result, (25, 25))
    assert len(matches) == 1 and matches[0]["status"] != "ABSTAINED"
    permuted = points[np.random.default_rng(19).permutation(len(points))]
    assert analyze(permuted, symmetry=symmetry) == result
    repeated = analyze(np.concatenate([permuted, permuted]), symmetry=symmetry)
    for key in ("positions", "extended_regions", "candidates", "search_diagnostics", "status"):
        assert repeated[key] == result[key]


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_connected_contaminant_is_retained_and_models_abstain(symmetry):
    extra = [[2.4/math.sqrt(2), 0, 2.4/math.sqrt(2)]]
    result = analyze(np.concatenate([np.concatenate(synthetic_bend()), extra]), symmetry=symmetry)
    assert_scope(result)
    assert result["status"] == "ABSTAINED"
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert_whole_component(result, candidate)
    assert len(candidate["transition_point_indices"]) == 36
    assert candidate["single_radius_analysis"]["status"] == candidate["two_radius_analysis"]["status"] == "ABSTAINED"


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
@pytest.mark.parametrize("changes,reason", [
    ({"max_regions": 3}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_plane_evaluations": 1}, "PLANE_DISCOVERY_INCOMPLETE"),
    ({"max_pair_evaluations": 5}, "PAIR_EVALUATION_BUDGET_EXCEEDED"),
    ({"max_bridge_evaluations": 1}, "BRIDGE_EVALUATION_BUDGET_EXCEEDED"),
])
def test_early_budget_failure_clears_all_prefixes(symmetry, changes, reason):
    points = np.concatenate(synthetic_bend())
    result = analyze(np.concatenate([points, points+[15, 0, 0]]), symmetry=symmetry, **changes)
    assert_scope(result)
    assert not result["search_complete_within_declared_policy"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["reasons"] == [reason]
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 0


@pytest.mark.parametrize("symmetry", ["mutual", "union"])
def test_late_split_exhaustion_clears_earlier_success(symmetry):
    points = np.concatenate([np.concatenate(synthetic_bend()),
                             np.concatenate(synthetic_bend(mixture=True))+[15, 0, 0]])
    result = analyze(points, symmetry=symmetry, max_split_evaluations=68)
    assert_scope(result)
    assert not result["search_complete_within_declared_policy"]
    assert result["candidates"] == result["extended_regions"] == []
    assert result["reasons"] == ["TWO_RADIUS_SEARCH_INCOMPLETE"]
    assert result["search_diagnostics"]["single_radius_evaluation_count"] == 2


@pytest.mark.parametrize("symmetry", [None, "", "adaptive", "Mutual", True, []])
def test_graph_symmetry_must_be_explicit_and_valid_even_on_empty_input(symmetry):
    with pytest.raises(ValueError):
        analyze([], symmetry=symmetry)


def test_public_policies_are_required_keywords():
    signature = inspect.signature(analyze_graph_envelope_bend_candidates)
    assert "graph_symmetry" in signature.parameters
    assert all(p.default == inspect.Parameter.empty for p in signature.parameters.values())
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for name, p in signature.parameters.items() if name != "points")


def test_raw_cap_precedes_array_conversion():
    class Oversize(list):
        def __len__(self):
            return 257

        def __array__(self, *args, **kwargs):
            raise AssertionError("oversize conversion")

    with pytest.raises(ValueError, match="256"):
        analyze(Oversize())
