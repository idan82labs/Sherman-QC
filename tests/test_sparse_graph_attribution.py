"""Fixed sparse-mixture graph attribution, with no graph-policy experiments.

Sixty analyzer calls cover the 30 exact and identical bounded-noise sparse
mixtures from the 150-case baseline. The earlier 120 geometries only advance the
original seed-2026090717 RNG stream. No k, epsilon, radius, graph or fit cap changes.
Construction IDs remain diagnostic oracles, never analyzer inputs.
"""
from collections import Counter
import json

import numpy as np
import pytest

from tests.test_cone_closed_bend_candidates import (
    analyze, assert_scope, assert_whole_component, synthetic_bend,
)


SEED = 2026090717
ANGLES = (-150, -120, -90, -60, -30, 30, 60, 90, 120, 150)
RADII = (0.5, 2, 20)
STRATA = ("clean", "missing_middle_arc", "concentric_mixture",
          "sparse_flange_stations", "sparse_concentric_mixture")
FIRST = frozenset(range(25))
SECOND = frozenset(range(25, 50))
TRANSITION = frozenset(range(50, 120))


def _components(adjacency, support):
    remaining, groups = set(support), []
    while remaining:
        component, frontier = set(), {min(remaining)}
        while frontier:
            component.update(frontier)
            frontier = set.union(*(adjacency[index] for index in frontier)) & remaining - component
        remaining -= component
        groups.append(frozenset(component))
    return groups


def _graph_state(observed, result):
    graph = result["graph"]
    original = {tuple(row): index for index, row in enumerate(observed)}
    ids = [original[tuple(row)] for row in result["positions"]]
    edges = {tuple(sorted((ids[first], ids[second]))) for first, second in graph["edges"]}
    adjacency = {index: set() for index in range(len(observed))}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    # Reconstruct the disclosed observed-distance calculation from public rows.
    # This produces diagnostics only; neither edges nor analyzer inputs change.
    magnitude = float(np.max(np.abs(observed)))
    normalized = observed / magnitude
    delta = normalized[:, None, :] - normalized[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2)) * magnitude
    radii = np.empty(len(observed))
    for canonical_index, original_index in enumerate(ids):
        radii[original_index] = graph["local_radii"][canonical_index]
    return dict(edges=edges, adjacency=adjacency, distances=distances,
                radii=radii, guard=graph["distance_numerical_allowance"],
                rank_allowance=graph["rank_membership_uncertainty_allowance"],
                cap=graph["max_link_distance"], cap_allowance=graph["link_cap_uncertainty_allowance"],
                canonical_ids=ids)


def _edge_detail(state, first, second):
    distance = float(state["distances"][first, second])
    rank_gaps = [distance - float(state["radii"][index]) for index in (first, second)]
    included = [gap <= state["guard"] for gap in rank_gaps]
    in_cap = distance - state["cap"] <= state["guard"]
    observed_edge = tuple(sorted((first, second))) in state["edges"]
    assert observed_edge == (all(included) and in_cap)
    return dict(original_point_ids=[first, second], distance=distance,
                observed_edge=observed_edge, directed_rank_membership=included,
                distance_minus_kth_radius=rank_gaps,
                rank_boundary_sensitive=[abs(gap) <= state["rank_allowance"] for gap in rank_gaps],
                distance_minus_link_cap=distance-state["cap"],
                link_cap_sensitive=abs(distance-state["cap"]) <= state["cap_allowance"])


def _cut_receipt(state, left, right):
    pairs = [_edge_detail(state, first, second) for first in sorted(left) for second in sorted(right)]
    reasons = Counter()
    for pair in pairs:
        if pair["observed_edge"]:
            reasons["OBSERVED_EDGE"] += 1
        elif pair["distance_minus_link_cap"] > state["guard"]:
            reasons["OUTSIDE_OBSERVED_LINK_CAP"] += 1
        elif not any(pair["directed_rank_membership"]):
            reasons["NEITHER_DIRECTED_RANK_MEMBERSHIP"] += 1
        else:
            reasons["ONLY_ONE_DIRECTED_RANK_MEMBERSHIP"] += 1
    return dict(
        pair_count=len(pairs), edge_count=reasons["OBSERVED_EDGE"],
        exclusion_counts=dict(sorted(reasons.items())),
        rank_boundary_sensitive_pairs=sum(any(pair["rank_boundary_sensitive"]) for pair in pairs),
        rank_boundary_sensitive_missing_pairs=sum(not pair["observed_edge"] and any(pair["rank_boundary_sensitive"]) for pair in pairs),
        link_cap_sensitive_pairs=sum(pair["link_cap_sensitive"] for pair in pairs),
        # Fixed reporting bound, not geometry trimming or a search restriction.
        nearest_pair_examples=sorted(pairs, key=lambda item: (item["distance"], item["original_point_ids"]))[:4],
    )


def _case_receipt(observed, result, state):
    ids = state["canonical_ids"]
    supports = {region["region_id"]: frozenset(ids[index] for index in region["point_indices"])
                for region in result["extended_regions"]}
    full_flange_present = [flange in supports.values() for flange in (FIRST, SECOND)]
    groups = _components(state["adjacency"], TRANSITION)
    flange_components = []
    for flange in (FIRST, SECOND):
        flange_components.append([
            dict(original_point_ids=sorted(component), sample_count=len(component),
                 cut_to_remaining_flange=_cut_receipt(state, component, flange-component))
            for component in _components(state["adjacency"], flange)
        ])
    first_cut, second_cut = _cut_receipt(state, TRANSITION, FIRST), _cut_receipt(state, TRANSITION, SECOND)
    components = []
    for group in groups:
        components.append(dict(
            original_point_ids=sorted(group), sample_count=len(group),
            inner_layer_count=len(group & frozenset(range(50, 85))),
            outer_layer_count=len(group & frozenset(range(85, 120))),
            seam_edge_counts=[sum(len(state["adjacency"][index] & flange) for index in group)
                              for flange in (FIRST, SECOND)],
            cut_to_remaining_transition=_cut_receipt(state, group, TRANSITION-group),
        ))
    matched = []
    for candidate in result["candidates"]:
        assert_whole_component(result, candidate)
        if ({supports[index] for index in candidate["plane_region_ids"]} == {FIRST, SECOND}
                and frozenset(ids[index] for index in candidate["transition_point_indices"]) == TRANSITION):
            two = candidate["two_radius_analysis"]
            matched.append(dict(status=candidate["status"],
                                single_reasons=candidate["single_radius_analysis"]["reasons"],
                                two_reasons=None if two is None else two["reasons"]))
    conditional = sum(candidate["status"] != "ABSTAINED" for candidate in matched)
    if not result["search_complete_within_declared_policy"]:
        classification = "SEARCH_INCOMPLETE"
    elif not all(full_flange_present):
        classification = "ABSENT_FULL_FLANGE_SUPPORT"
    elif len(groups) != 1:
        classification = "DISCONNECTED_TRANSITION"
    elif not first_cut["edge_count"] and not second_cut["edge_count"]:
        classification = "LACKS_BOTH_SEAM_LINKS"
    elif not first_cut["edge_count"]:
        classification = "LACKS_FIRST_SEAM_LINK"
    elif not second_cut["edge_count"]:
        classification = "LACKS_SECOND_SEAM_LINK"
    elif not matched:
        classification = "JOINT_ASSOCIATION_NOT_EMITTED"
    elif not conditional:
        classification = "RADIUS_ABSTAINED"
    else:
        classification = "COMPLETE_JOINT_CONDITIONAL"
    return dict(
        classification=classification, status=result["status"], reasons=result["reasons"],
        search_complete=result["search_complete_within_declared_policy"],
        final_region_count=len(supports), full_constructed_flange_present=full_flange_present,
        constructed_flange_graph_components=flange_components,
        final_region_construction_memberships=[dict(
            region_id=region_id, sample_count=len(support),
            first_flange_members=len(support & FIRST), second_flange_members=len(support & SECOND),
            transition_members=len(support & TRANSITION),
        ) for region_id, support in supports.items()],
        transition_component_count=len(groups), transition_components=components,
        transition_to_first_flange=first_cut, transition_to_second_flange=second_cut,
        matched_complete_joint_associations=len(matched), matched_conditional_joint_associations=conditional,
        matched_joint_model_reasons=matched, total_associations=len(result["candidates"]),
        unmatched_associations=len(result["candidates"])-len(matched),
        graph_edge_count=len(state["edges"]),
        graph_rank_sensitive_directed_count=result["graph"]["diagnostics"]["rank_boundary_sensitive_directed_count"],
    )


def _edge_class(edge):
    first, second = edge
    if first in TRANSITION and second in TRANSITION:
        return "transition_internal"
    if (first in TRANSITION) != (second in TRANSITION):
        return "transition_seam"
    return "flange_or_cross_flange"


def run_sparse_graph_attribution():
    """Fixed baseline RNG, policies and rows; graph attribution only."""
    rng = np.random.default_rng(SEED)
    receipts, comparisons = [], []
    advanced_cases = advanced_positions = 0
    for name in STRATA:
        for angle in ANGLES:
            for radius in RADII:
                first, second, transition = synthetic_bend(
                    angle, radius, flange_extent=2 if name.startswith("sparse_") else 1,
                    mixture=name.endswith("concentric_mixture"),
                )
                if name == "missing_middle_arc":
                    transition = np.concatenate([transition[:15], transition[20:]])
                exact = np.concatenate([first, second, transition])
                direction = rng.normal(size=exact.shape)
                direction /= np.linalg.norm(direction, axis=1, keepdims=True)
                epsilon = 1e-6*radius/2
                noisy = exact + direction*rng.uniform(0.25, 0.95, size=(len(exact), 1))*epsilon
                if name != "sparse_concentric_mixture":
                    advanced_cases += 1
                    advanced_positions += len(exact)
                    continue
                states = {}
                for noise, observed in (("exact", exact), ("bounded_noise", noisy)):
                    error = np.linalg.norm(observed-exact, axis=1)
                    assert np.all(error <= epsilon)
                    assert np.all(error > 0) if noise == "bounded_noise" else np.all(error == 0)
                    result = analyze(observed, scale=radius/2)
                    assert_scope(result)
                    state = _graph_state(observed, result)
                    states[noise] = state
                    receipt = _case_receipt(observed, result, state)
                    receipt.update(angle_degrees=angle, radius=radius, noise=noise,
                                   maximum_error_over_epsilon=float(np.max(error)/epsilon))
                    receipts.append(receipt)
                removed = states["exact"]["edges"]-states["bounded_noise"]["edges"]
                added = states["bounded_noise"]["edges"]-states["exact"]["edges"]
                relevant_removed = [edge for edge in sorted(removed) if _edge_class(edge) != "flange_or_cross_flange"]
                comparisons.append(dict(
                    angle_degrees=angle, radius=radius,
                    removed_edges_by_class=dict(sorted(Counter(map(_edge_class, removed)).items())),
                    added_edges_by_class=dict(sorted(Counter(map(_edge_class, added)).items())),
                    # Full removed transition/seam edge list is bounded by this
                    # 120-row graph and explains the exact-to-noisy cut change.
                    removed_transition_or_seam_edges=[_edge_detail(states["bounded_noise"], *edge) for edge in relevant_removed],
                    removed_flange_or_cross_flange_edges=[_edge_detail(states["bounded_noise"], *edge)
                        for edge in sorted(removed) if _edge_class(edge) == "flange_or_cross_flange"],
                ))
    strata = {}
    for noise in ("exact", "bounded_noise"):
        selected = [receipt for receipt in receipts if receipt["noise"] == noise]
        strata[noise] = dict(cases=len(selected),
            classification_counts=dict(sorted(Counter(receipt["classification"] for receipt in selected).items())),
            matched_complete_joint_associations=sum(receipt["matched_complete_joint_associations"] for receipt in selected),
            matched_conditional_joint_associations=sum(receipt["matched_conditional_joint_associations"] for receipt in selected),
            total_associations=sum(receipt["total_associations"] for receipt in selected),
            unmatched_associations=sum(receipt["unmatched_associations"] for receipt in selected),
            incomplete_cases=sum(not receipt["search_complete"] for receipt in selected))
    return dict(cases=len(receipts), paired_geometries=len(comparisons), seed=SEED,
                earlier_cases_rng_advanced_only=advanced_cases, earlier_positions_rng_advanced_only=advanced_positions,
                interpretation="OBSERVED_GRAPH_STAGE_ATTRIBUTION_NOT_JOINT_TOPOLOGY_OR_ACCURACY",
                strata=strata, case_receipts=receipts, edge_change_receipts=comparisons)


@pytest.fixture(scope="module")
def attribution():
    return run_sparse_graph_attribution()


def test_fixed_sparse_matrix_uses_original_rng_order_and_exact_denominators(attribution):
    assert attribution["cases"] == 60 and attribution["paired_geometries"] == 30
    assert attribution["seed"] == SEED
    assert attribution["earlier_cases_rng_advanced_only"] == 120
    assert attribution["earlier_positions_rng_advanced_only"] == 11100
    for noise, stratum in attribution["strata"].items():
        assert stratum["cases"] == sum(stratum["classification_counts"].values()) == 30
        assert stratum["total_associations"] == stratum["matched_complete_joint_associations"] + stratum["unmatched_associations"]
        assert stratum["matched_conditional_joint_associations"] <= stratum["matched_complete_joint_associations"]
        assert noise in ("exact", "bounded_noise")
    json.dumps(attribution, allow_nan=False)


def test_component_cuts_and_seam_pair_counts_are_complete(attribution):
    for case in attribution["case_receipts"]:
        components = case["transition_components"]
        assert len(components) == case["transition_component_count"]
        assert sum(component["sample_count"] for component in components) == 70
        assert set().union(*(set(component["original_point_ids"]) for component in components)) == TRANSITION
        for component in components:
            assert component["inner_layer_count"] + component["outer_layer_count"] == component["sample_count"]
            cut = component["cut_to_remaining_transition"]
            assert cut["pair_count"] == component["sample_count"]*(70-component["sample_count"])
            assert cut["edge_count"] == 0
            assert sum(cut["exclusion_counts"].values()) == cut["pair_count"]
        for key in ("transition_to_first_flange", "transition_to_second_flange"):
            cut = case[key]
            assert cut["pair_count"] == sum(cut["exclusion_counts"].values()) == 1750
        for flange, components in zip((FIRST, SECOND), case["constructed_flange_graph_components"]):
            assert sum(component["sample_count"] for component in components) == 25
            assert set().union(*(set(component["original_point_ids"]) for component in components)) == flange
            for component in components:
                cut = component["cut_to_remaining_flange"]
                assert cut["pair_count"] == component["sample_count"]*(25-component["sample_count"])
                assert cut["edge_count"] == 0
                assert sum(cut["exclusion_counts"].values()) == cut["pair_count"]


def test_sparse_attribution_reports_all_failures_without_changing_policy(attribution):
    print(json.dumps(attribution["strata"], sort_keys=True))
    print(json.dumps({"nonconditional_cases": [
        dict({key: case[key] for key in ("angle_degrees", "radius", "noise", "classification",
                                        "full_constructed_flange_present", "transition_component_count")},
             flange_component_sizes=[[component["sample_count"] for component in components]
                                     for components in case["constructed_flange_graph_components"]])
        for case in attribution["case_receipts"] if case["classification"] != "COMPLETE_JOINT_CONDITIONAL"
    ]}, sort_keys=True))
    assert all(case["classification"] != "JOINT_ASSOCIATION_NOT_EMITTED" for case in attribution["case_receipts"])


def test_recorded_first_sparse_attribution_is_a_regression_not_an_accuracy_gate(attribution):
    assert attribution["strata"] == {
        "exact": dict(cases=30,
                      classification_counts={"ABSENT_FULL_FLANGE_SUPPORT": 12, "COMPLETE_JOINT_CONDITIONAL": 18},
                      matched_complete_joint_associations=18, matched_conditional_joint_associations=18,
                      total_associations=54, unmatched_associations=36, incomplete_cases=0),
        "bounded_noise": dict(cases=30,
                              classification_counts={"ABSENT_FULL_FLANGE_SUPPORT": 14, "COMPLETE_JOINT_CONDITIONAL": 16},
                              matched_complete_joint_associations=16, matched_conditional_joint_associations=16,
                              total_associations=52, unmatched_associations=36, incomplete_cases=0),
    }
