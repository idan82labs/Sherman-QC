"""Bounded interval-rank graph envelopes on fixed observed node identities.

The upper graph is NOT a realizability witness, even for an individual edge.
The lower graph is a conservative implication under the stipulated simultaneous
position bound and chosen sampled-connection policy. Neither is physical truth.
"""
from __future__ import annotations

import math

import numpy as np

from research.geometry.adaptive_position_regions import _input, _rank_radius, _within
from research.geometry.position_axis_hypothesis import _number
from research.geometry.position_bend_candidates import _finite_payload
from research.geometry.position_plane_regions import MAX_POINTS, _distances, _integer


def _stop(result, reason, *, complete=False):
    result.update(status="ABSTAINED", edges=[], lower_edges=[], observed_edges=[],
                  edge_count=0, lower_edge_count=0, observed_edge_count=0,
                  search_complete_within_declared_policy=complete, reasons=[reason])
    return result


def _directed_interval_ranks(lower, upper, k):
    """Conservative complete-tie kNN ranks, excluding query AND target.

    j belongs iff fewer than k OTHER nodes l have distance(i,l)<distance(i,j).
    At most k-1 potentially closer competitors implies guaranteed inclusion.
    At most k-1 definitely closer competitors only means not excluded by these
    separate distance intervals; their joint Euclidean realization is unchecked.
    """
    n = len(lower)
    guaranteed = np.zeros((n, n), dtype=bool)
    outer = np.zeros((n, n), dtype=bool)
    for query in range(n):
        others = np.concatenate((np.arange(query), np.arange(query + 1, n)))
        lows, highs = lower[query, others], upper[query, others]
        # Rows are competitors, columns are targets. Diagonal entries are the
        # target itself and MUST NOT be counted, including at positive epsilon.
        potentially_closer = lows[:, None] < highs[None, :]
        definitely_closer = highs[:, None] < lows[None, :]
        np.fill_diagonal(potentially_closer, False)
        np.fill_diagonal(definitely_closer, False)
        guaranteed[query, others] = np.count_nonzero(potentially_closer, axis=0) < k
        outer[query, others] = np.count_nonzero(definitely_closer, axis=0) < k
    return guaranteed, outer


def _symmetrize(directed, symmetry):
    return directed & directed.T if symmetry == "mutual" else directed | directed.T


def _edges(adjacency):
    return np.transpose(np.nonzero(np.triu(adjacency, 1))).tolist()


def build_position_graph_envelope(
    points, *, unit, position_error_bound, neighbor_count, max_link_distance, symmetry,
):
    """Return lower, guarded-observed and outer graphs for one declared policy.

    ``symmetry='mutual'`` requires both directed memberships; ``'union'`` requires
    either. Union is a DIFFERENT sampled-connection policy, not a repaired mutual
    graph. Distance and rank uncertainty do not relax any downstream measurement
    bound. The ``edges`` field is the upper proposal graph, not verified topology.

    Exact repeats in the OBSERVED input are canonicalized once, after the raw
    256-row cap. These node identities then remain fixed: hypothetical corrected
    positions that coincide are NOT deduplicated or assigned a smaller rank k.
    """
    k = _integer(neighbor_count, "neighbor_count", 2, MAX_POINTS - 1)
    cap = _number(max_link_distance, "max_link_distance")
    if not isinstance(symmetry, str) or symmetry not in ("mutual", "union"):
        raise ValueError("symmetry must explicitly be 'mutual' or 'union'")
    count, p, epsilon = _input(points, unit, position_error_bound)
    n = len(p)
    result = {
        "schema_version": 1, "kind": "POSITION_INTERVAL_RANK_GRAPH_ENVELOPE_V1",
        "status": "ABSTAINED", "production_disposition": "NOT_AUTHORIZED", "unit": unit,
        "input_count": count, "unique_count": n, "positions": p.tolist(),
        "position_error_bound": epsilon, "neighbor_count": k, "max_link_distance": cap,
        "symmetry": symmetry, "neighbor_count_excludes_seed": True,
        "search_complete_within_declared_policy": False,
        "scope": "FIXED_OBSERVED_NODE_INTERVAL_RANK_SAMPLED_GRAPH_ENVELOPE_ONLY",
        "upper_edge_scope": "NOT_EXCLUDED_BY_PAIRWISE_INTERVAL_RANK_RELAXATION",
        "lower_edge_scope": "CONSERVATIVE_GUARANTEE_CONDITIONAL_ON_FIXED_NODES_ERROR_AND_CHOSEN_GRAPH_POLICY",
        "observed_edge_scope": "COMPLETE_OBSERVED_KTH_TIES_AND_CAP_WITH_NUMERICAL_GUARD",
        "edge_field_selects": "UPPER_PROPOSAL_GRAPH",
        "distance_numerical_allowance": 0.0, "distance_interval_radius": 0.0,
        "epsilon_expands_proposal_edges": True,
        "measurement_error_bound_is_relaxed": False,
        "joint_realizability_is_verified": False,
        "individual_upper_edge_realizability_is_verified": False,
        "physical_connectivity_is_verified": False,
        "enclosure_is_numerically_certified": False,
        "neighborhood_membership_is_physically_verified": False,
        "policy": {
            "node_identity": "CANONICAL_DISTINCT_OBSERVED_ROWS_FIXED_UNDER_ERROR",
            "latent_coincidences_remove_nodes": False,
            "symmetry": symmetry,
            "union_is_not_mutual_graph": symmetry == "union",
            "lower_rank_test": "COUNT_OTHER_COMPETITOR_LOWER_LT_TARGET_UPPER_LT_K",
            "upper_rank_test": "COUNT_OTHER_COMPETITOR_UPPER_LT_TARGET_LOWER_LT_K",
            "rank_competitors_exclude_query_and_target": True,
            "complete_distance_ties_are_included": True,
            "lower_cap_test": "PAIR_DISTANCE_UPPER_LE_DECLARED_CAP",
            "upper_cap_test": "PAIR_DISTANCE_LOWER_LE_DECLARED_CAP",
            "max_raw_positions": MAX_POINTS,
            "max_undirected_edges": MAX_POINTS * (MAX_POINTS - 1) // 2,
            "k_or_distance_cap_adaptation": False,
        },
        "local_observed_radii": [], "edges": [], "lower_edges": [], "observed_edges": [],
        "edge_count": 0, "lower_edge_count": 0, "observed_edge_count": 0,
        "diagnostics": {
            "lower_directed_memberships": 0, "outer_directed_memberships": 0,
            "observed_directed_memberships": 0, "upper_only_edge_count": 0,
            "observed_edges_not_lower_count": 0,
            "logical_rank_comparison_count": 0,
        },
        "reasons": [],
        "limitations": [
            "RESEARCH_ONLY_NOT_RUNTIME_INTEGRATION",
            "SIMULTANEOUS_POSITION_ERROR_BOUND_UNCALIBRATED",
            "ENGINEERING_NUMERICAL_GUARD_NOT_DIRECTED_ROUNDING_CERTIFICATE",
            "UPPER_RANK_MEMBERSHIP_NOT_EVEN_INDIVIDUAL_EDGE_REALIZABILITY",
            "PAIRWISE_DISTANCE_INTERVALS_SHARE_UNMODELED_COORDINATE_DEPENDENCIES",
            "UPPER_EDGES_NEED_NOT_BE_JOINTLY_REALIZABLE",
            "LOWER_ENVELOPE_IS_SUFFICIENT_NOT_EXHAUSTIVE_GUARANTEED_GRAPH",
            "FIXED_NODE_POLICY_DIFFERS_FROM_REDEDUPLICATING_LATENT_POSITIONS",
            "UNION_AND_MUTUAL_ARE_DIFFERENT_DECLARED_SAMPLED_CONNECTION_POLICIES",
            "DENSITY_ANISOTROPY_AND_CAP_CAN_FRAGMENT_ONE_SURFACE_EVEN_WITH_ZERO_ERROR",
            "ADDED_EDGES_CAN_MERGE_DISTINCT_SURFACES_OR_UNRELATED_OBJECTS",
            "WHOLE_COMPONENT_GEOMETRIC_REFIT_STILL_REQUIRED_NO_TRIMMING",
            "SAMPLED_CONNECTIVITY_NOT_PHYSICAL_CONTACT_CONTINUITY_OR_BEND_COUNT",
        ],
    }
    if n <= k:
        return _stop(result, "INSUFFICIENT_OTHER_POSITIONS", complete=True)
    distances, guard = _distances(p)
    uncertainty = 2 * epsilon + guard
    if distances is None or not math.isfinite(uncertainty):
        return _stop(result, "NUMERICAL_RANGE_UNSUPPORTED")
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.maximum(0.0, distances - uncertainty)
        upper = distances + uncertainty
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return _stop(result, "NUMERICAL_RANGE_UNSUPPORTED")
    guaranteed, outer = _directed_interval_ranks(lower, upper, k)
    radii = np.asarray([_rank_radius(row, seed, k) for seed, row in enumerate(distances)])
    observed = _within(distances, radii[:, None], guard)
    np.fill_diagonal(observed, False)
    lower_graph = _symmetrize(guaranteed, symmetry) & (upper <= cap)
    outer_graph = _symmetrize(outer, symmetry) & (lower <= cap)
    observed_graph = _symmetrize(observed, symmetry) & _within(distances, cap, guard)
    for adjacency in (lower_graph, outer_graph, observed_graph):
        np.fill_diagonal(adjacency, False)
    if (np.any(lower_graph & ~observed_graph) or np.any(observed_graph & ~outer_graph)):
        return _stop(result, "NUMERICAL_ENVELOPE_INCONSISTENT")
    result.update(
        status="SAMPLED_GRAPH_ENVELOPE", search_complete_within_declared_policy=True,
        distance_numerical_allowance=guard, distance_interval_radius=uncertainty,
        local_observed_radii=radii.tolist(),
        edges=_edges(outer_graph), lower_edges=_edges(lower_graph), observed_edges=_edges(observed_graph),
        edge_count=int(np.count_nonzero(np.triu(outer_graph, 1))),
        lower_edge_count=int(np.count_nonzero(np.triu(lower_graph, 1))),
        observed_edge_count=int(np.count_nonzero(np.triu(observed_graph, 1))),
    )
    result["diagnostics"] = {
        "lower_directed_memberships": int(np.count_nonzero(guaranteed)),
        "outer_directed_memberships": int(np.count_nonzero(outer)),
        "observed_directed_memberships": int(np.count_nonzero(observed)),
        "upper_only_edge_count": int(np.count_nonzero(np.triu(outer_graph & ~observed_graph, 1))),
        "observed_edges_not_lower_count": int(np.count_nonzero(np.triu(observed_graph & ~lower_graph, 1))),
        "logical_rank_comparison_count": 2 * n * (n - 1) * (n - 2),
    }
    if not result["edges"]:
        result["reasons"] = ["NO_UPPER_GRAPH_EDGES_WITHIN_DECLARED_POLICY"]
    if not _finite_payload(result):
        return _stop(result, "NUMERICAL_RANGE_UNSUPPORTED")
    return result
