"""Fixed source-only attribution of missing adjacent central-flange seeds.

Exactly 12 union-policy calls: U(90,90)/S(90,-90), radii (2,3), full arcs,
central lengths .25/2/6, exact and the EXISTING challenge RNG row stream. No
alternate k/cap/epsilon, label-assisted engine call, or candidate modification.
Within-cap oracle subsets below are diagnostic SVD summaries, never proposals
fed to inference or evidence that a different policy would succeed.
"""
from collections import Counter
import json

import numpy as np
import pytest

from tests.test_adjacent_bend_challenge import (
    challenge_inputs, classify_result, connected_components,
)
from tests.test_graph_envelope_bend_candidates import analyze, assert_scope


def representative_inputs():
    # Iterate the original generator, rather than reseeding selected cases.
    # Unselected geometry is constructed only to preserve its original RNG use.
    for case, fixture, observed in challenge_inputs():
        if (case["angles"] in ((90, 90), (90, -90)) and case["radii"] == (2, 3)
                and case["coverage"] == "full"):
            yield case, fixture, observed


def _spread(points, flange):
    if len(points) == 0:
        return dict(sample_count=0, singular_values=[0.0, 0.0, 0.0],
                    second_singular_value=0.0, along_flange_span=0.0,
                    axial_span=0.0, maximum_constructed_plane_residual=0.0)
    centered = points-np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    singular = np.pad(singular, (0, 3-len(singular)))
    tangent, normal = np.asarray(flange["tangent"]), np.asarray(flange["normal"])
    # The fixed fixture's extrusion axis is +Y (no pose in this challenge).
    along = points@tangent
    return dict(sample_count=len(points), singular_values=singular.tolist(),
                second_singular_value=float(singular[1]),
                along_flange_span=float(np.ptp(along)), axial_span=float(np.ptp(points[:, 1])),
                maximum_constructed_plane_residual=float(np.max(np.abs((points-flange["start"])@normal))))


def _case_attribution(case, fixture, observed, result):
    discovery = result["plane_discovery"]
    points = np.asarray(result["positions"])
    original = {tuple(row): index for index, row in enumerate(observed)}
    canonical_ids = [original[tuple(row)] for row in points]
    central = frozenset(map(int, fixture["support_indices"]["flange_1"]))
    names = {name: frozenset(map(int, indices)) for name, indices in fixture["support_indices"].items()}
    canonical_central = np.asarray([index in central for index in canonical_ids])
    source_names = {index: name for name, ids in names.items() for index in ids}
    final_supports = {
        region["region_id"]: frozenset(canonical_ids[index] for index in region["point_indices"])
        for region in result["extended_regions"]
    }
    adjacency = {index: set() for index in range(len(observed))}
    for one, two in result["graph"]["edges"]:
        first, second = canonical_ids[one], canonical_ids[two]
        adjacency[first].add(second)
        adjacency[second].add(first)
    components = connected_components(adjacency, central)
    # Same bounded observed-distance arithmetic as the disclosed query, computed
    # only for attribution on these already constructed rows. No model fits.
    magnitude = float(np.max(np.abs(points)))
    normalized = points/magnitude
    difference = normalized[:, None, :]-normalized[None, :, :]
    distances = np.sqrt(np.sum(difference*difference, axis=2))*magnitude
    cap = discovery["policy"]["max_neighborhood_radius"]
    guard = discovery["distance_numerical_allowance"]
    query_receipts, central_queries = [], []
    for query in discovery["queries"]:
        seed = query["seed_index"]
        original_seed = canonical_ids[seed]
        indices = query.get("point_indices", [])
        support = frozenset(canonical_ids[index] for index in indices)
        intersects = bool(support & central)
        centered_on_central = original_seed in central
        if not intersects and not centered_on_central:
            continue
        within_cap = (distances[seed] <= cap) | (distances[seed]-cap <= guard)
        central_inside = within_cap & canonical_central
        radius = query["radius"]
        receipt = dict(
            query_id=query["query_id"], original_seed_index=original_seed,
            seed_construction_support=source_names[original_seed],
            centered_on_central_flange=centered_on_central,
            query_status=query["status"], kth_distance=radius,
            kth_distance_minus_cap=None if radius is None else radius-cap,
            queried_support_count=len(support), queried_central_count=len(support & central),
            queried_support_is_pure_central=bool(support) and support <= central,
            queried_support_memberships={name: len(support & ids) for name, ids in names.items()},
            accepted_seed_has_cone_proposal="cone_proposal" in query,
            cone_candidate_point_count=query.get("cone_proposal", {}).get("candidate_point_count"),
            within_cap_all_support_count=int(np.count_nonzero(within_cap)),
            within_cap_central_original_ids=[canonical_ids[index] for index in np.flatnonzero(central_inside)],
            within_cap_oracle_central_spread=_spread(points[central_inside], fixture["truth"]["flanges"][1]),
            queried_full_support_spread=_spread(points[indices], fixture["truth"]["flanges"][1]),
            oracle_subset_is_an_inference_input=False,
        )
        query_receipts.append(receipt)
        if centered_on_central:
            central_queries.append(receipt)
    pure = [query for query in query_receipts if query["queried_support_is_pure_central"]]
    within_cap_sizes = Counter(query["within_cap_oracle_central_spread"]["sample_count"] for query in central_queries)
    base_receipt = classify_result(fixture, observed, result)
    return dict(
        case, search_complete=result["search_complete_within_declared_policy"], reasons=result["reasons"],
        central_flange_sample_count=len(central),
        complete_central_flange_present=central in final_supports.values(),
        central_induced_graph_component_sizes=sorted(map(len, components)),
        central_induced_graph_components=[sorted(component) for component in components],
        central_seed_query_count=len(central_queries),
        central_seed_query_status_counts=dict(sorted(Counter(query["query_status"] for query in central_queries).items())),
        queried_supports_intersecting_central_count=sum(query["queried_central_count"] > 0 for query in query_receipts),
        pure_central_queried_neighborhood_count=len(pure),
        accepted_pure_central_seed_query_count=sum(query["accepted_seed_has_cone_proposal"] for query in pure),
        central_seed_nominal_cap_miss_count=sum(query["query_status"] == "NOMINAL_RADIUS_CAP_EXCEEDED" for query in central_queries),
        within_cap_central_support_size_counts=dict(sorted(within_cap_sizes.items())),
        minimum_within_cap_central_second_singular_value=min(query["within_cap_oracle_central_spread"]["second_singular_value"] for query in central_queries),
        maximum_within_cap_central_second_singular_value=max(query["within_cap_oracle_central_spread"]["second_singular_value"] for query in central_queries),
        final_region_memberships=[dict(region_id=region_id,
            memberships={name: len(support & ids) for name, ids in names.items()})
            for region_id, support in final_supports.items()],
        discovery_diagnostics=discovery["diagnostics"], events=base_receipt["events"],
        query_receipts=query_receipts,
    )


def run_adjacent_seed_attribution():
    receipts = []
    for case, fixture, observed in representative_inputs():
        result = analyze(observed, symmetry="union")
        assert_scope(result)
        assert result["plane_discovery"]["policy"]["neighbor_counts"] == [8]
        assert result["plane_discovery"]["policy"]["max_neighborhood_radius"] == 1.5
        assert result["graph"]["neighbor_count"] == 10
        assert result["graph"]["max_link_distance"] == 1.5
        assert result["graph"]["position_error_bound"] == 1e-6
        receipts.append(_case_attribution(case, fixture, observed, result))
    return dict(analyzer_calls=len(receipts), symmetry="union", noise_seed=2026090702,
                interpretation="FIXED_QUERY_STAGE_ATTRIBUTION_NOT_ALTERNATIVE_SEED_POLICY_EVALUATION",
                receipts=receipts)


@pytest.fixture(scope="module")
def attribution():
    return run_adjacent_seed_attribution()


def test_declared_representative_subset_preserves_original_case_stream_and_counts(attribution):
    assert attribution["analyzer_calls"] == 12
    cases = {tuple((tuple(item["angles"]), tuple(item["radii"]), item["middle_length"], item["noise"]))
             for item in attribution["receipts"]}
    assert cases == {(angles, (2, 3), length, noise)
                     for angles in ((90, 90), (90, -90)) for length in (0.25, 2, 6)
                     for noise in ("exact", "bounded_noise")}
    assert all(item["coverage"] == "full" for item in attribution["receipts"])
    json.dumps(attribution, allow_nan=False)


def test_all_central_queries_and_graph_components_have_explicit_denominators(attribution):
    for case in attribution["receipts"]:
        assert case["central_flange_sample_count"] == case["central_seed_query_count"] == 25
        assert sum(case["central_seed_query_status_counts"].values()) == 25
        assert sum(case["central_induced_graph_component_sizes"]) == 25
        assert sum(case["within_cap_central_support_size_counts"].values()) == 25
        assert case["accepted_pure_central_seed_query_count"] <= case["pure_central_queried_neighborhood_count"]
        for query in case["query_receipts"]:
            assert sum(query["queried_support_memberships"].values()) == query["queried_support_count"]
            assert len(query["within_cap_central_original_ids"]) == query["within_cap_oracle_central_spread"]["sample_count"]
            assert query["oracle_subset_is_an_inference_input"] is False
            if query["query_status"] == "NOMINAL_RADIUS_CAP_EXCEEDED":
                assert query["queried_support_count"] == 0
                assert query["kth_distance_minus_cap"] > 0


def test_initial_stage_receipts_are_reported_without_a_success_or_seed_count_gate(attribution):
    keys = ("angles", "middle_length", "noise", "complete_central_flange_present",
            "central_induced_graph_component_sizes", "central_seed_query_status_counts",
            "pure_central_queried_neighborhood_count", "accepted_pure_central_seed_query_count",
            "central_seed_nominal_cap_miss_count", "within_cap_central_support_size_counts",
            "minimum_within_cap_central_second_singular_value", "maximum_within_cap_central_second_singular_value")
    print("ADJACENT_SEED_ATTRIBUTION="+json.dumps([
        {key: case[key] for key in keys} for case in attribution["receipts"]
    ], sort_keys=True))
