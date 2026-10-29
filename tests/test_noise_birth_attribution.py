"""Fixed synthetic ablations of adaptive region birth; no physical inputs.

Declared before execution: 48 cases = six signed angles (+/-30, 90, 150
degrees), radius 2 mm, dense/sparse concentric mixtures, and four noise
locations (none, flanges only, transition only, all positions). Seed 2026090718
generates one shared noise array per geometry with lengths 25--95% of the
unchanged 1e-6 mm error bound. The public development policy is unchanged:
discovery k=8, radius cap 1.5 mm, region cap 64; observed graph k=10/cap 1.5.

This diagnostic does not recover an accepted prefix or lift a cap. Discovery
overflow implies at least 65 accepted supports; it does NOT disclose the full
accepted count. Construction membership is an assertion/report-only oracle.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from research.geometry.position_bend_candidates import _fixed_plane_membership


_FIXTURE_PATH = Path(__file__).with_name("test_adaptive_bend_candidates.py")
_SPEC = importlib.util.spec_from_file_location("_noise_attribution_fixture", _FIXTURE_PATH)
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)

SEED = 2026090718
EPSILON = 1e-6
ANGLES = (-150, -90, -30, 30, 90, 150)
NOISE_LOCATIONS = ("exact", "flanges_only", "transition_only", "all")


def _classify_support(support, first, second, transition):
    if support <= first:
        return "first_flange"
    if support <= second:
        return "second_flange"
    if support <= transition:
        return "transition_only"
    return "cross_construction_support"


def _case_receipt(observed, result):
    discovery, graph = result["plane_discovery"], result["graph"]
    original_index = {tuple(row): index for index, row in enumerate(observed)}
    assert len(original_index) == len(observed) == 120
    ids = [original_index[tuple(row)] for row in result["positions"]]
    first, second, transition = frozenset(range(25)), frozenset(range(25, 50)), frozenset(range(50, 120))
    ids_from_support = lambda support: frozenset(ids[index] for index in support)
    canonical = np.asarray(result["positions"])
    seed_classes, extension_classes = Counter(), Counter()
    seed_missed, seed_total = 0, 0
    pure_flange_seeds, complete_flange_seed_extensions = 0, 0
    for seed in discovery["regions"]:
        support = ids_from_support(seed["point_indices"])
        seed_classes[_classify_support(support, first, second, transition)] += 1
        truth = first if support <= first else second if support <= second else None
        if truth is None:
            continue
        pure_flange_seeds += 1
        compatible = _fixed_plane_membership(
            canonical, seed["plane_analysis"], EPSILON,
            max(discovery["distance_numerical_allowance"], graph["distance_numerical_allowance"]),
        )
        assert compatible is not None
        membership = frozenset(ids[index] for index in np.flatnonzero(compatible))
        seed_missed += len(truth - membership)
        seed_total += len(truth)
        complete_flange_seed_extensions += int(truth <= membership)

    regions = {}
    full_flange_extensions = 0
    for region in result["extended_regions"]:
        support = ids_from_support(region["point_indices"])
        regions[region["region_id"]] = support
        extension_classes[_classify_support(support, first, second, transition)] += 1
        full_flange_extensions += int(support in (first, second))
    matched_associations = 0
    for candidate in result["candidates"]:
        _FIXTURE.assert_whole_component(result, candidate)
        matched_associations += int(
            {regions[index] for index in candidate["plane_region_ids"]} == {first, second}
            and ids_from_support(candidate["transition_point_indices"]) == transition
        )
    complete_discovery = discovery["search_complete_within_declared_policy"]
    overflow = discovery["reasons"] == ["REGION_CANDIDATE_BUDGET_EXCEEDED"]
    retained = len(discovery["regions"])
    return {
        "status": result["status"],
        "search_complete": result["search_complete_within_declared_policy"],
        "reasons": result["reasons"],
        "discovery_complete": complete_discovery,
        "discovery_reasons": discovery["reasons"],
        "seed_queries": discovery["seed_query_count"],
        "unique_support_queries": discovery["unique_support_query_count"],
        "query_status_counts": dict(sorted(Counter(q["status"] for q in discovery["queries"]).items())),
        "query_support_size_counts": dict(sorted(Counter(q["support_count"] for q in discovery["queries"]).items())),
        "rank_sensitive_seed_queries": sum(q["rank_boundary_sensitive_count"] > 0 for q in discovery["queries"]),
        "retained_seed_count": retained,
        "accepted_seed_count_if_complete": retained if complete_discovery else None,
        "accepted_seed_count_lower_bound": 65 if overflow else retained,
        "seed_support_classes": dict(sorted(seed_classes.items())) if complete_discovery else None,
        "pure_flange_seed_count": pure_flange_seeds if complete_discovery else None,
        "fixed_plane_full_flange_membership_seed_count": complete_flange_seed_extensions if complete_discovery else None,
        "fixed_plane_missed_true_flange_memberships": seed_missed if complete_discovery else None,
        "fixed_plane_true_flange_membership_denominator": seed_total if complete_discovery else None,
        "graph_edges": graph["edge_count"],
        "graph_rank_sensitive_directed_count": graph["diagnostics"]["rank_boundary_sensitive_directed_count"],
        "retained_extended_region_count": len(result["extended_regions"]),
        "extension_support_classes": dict(sorted(extension_classes.items())),
        "full_constructed_flange_extension_count": full_flange_extensions,
        "retained_association_count": len(result["candidates"]),
        "matched_whole_joint_association_count": matched_associations,
        "unmatched_association_count": len(result["candidates"]) - matched_associations,
        "search_diagnostics": result["search_diagnostics"],
    }


def _geometry_cases(angle, density, rng):
    exact = np.concatenate(_FIXTURE.synthetic_bend(
        angle, 2.0, flange_extent=1 if density == "dense" else 2, mixture=True,
    ))
    direction = rng.normal(size=exact.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    noise = direction * rng.uniform(0.25, 0.95, size=(len(exact), 1)) * EPSILON
    cases = []
    for location in NOISE_LOCATIONS:
        selected = np.zeros(len(exact), dtype=bool)
        if location in ("flanges_only", "all"):
            selected[:50] = True
        if location in ("transition_only", "all"):
            selected[50:] = True
        observed = exact.copy()
        observed[selected] += noise[selected]
        errors = np.linalg.norm(observed - exact, axis=1)
        assert np.all(errors[~selected] == 0)
        assert np.all((errors[selected] > 0) & (errors[selected] <= EPSILON))
        result = _FIXTURE.analyze(observed)
        _FIXTURE.assert_scope(result)
        receipt = _case_receipt(observed, result)
        receipt.update(angle_degrees=angle, radius=2.0, density=density, noise_location=location,
                       positions_perturbed=int(np.count_nonzero(selected)),
                       maximum_error_over_epsilon=float(np.max(errors) / EPSILON))
        cases.append(receipt)
    return cases


def run_noise_birth_attribution():
    """Return bounded public-output diagnostics; no uncapped model reanalysis."""
    rng = np.random.default_rng(SEED)
    cases = [case for density in ("dense", "sparse") for angle in ANGLES
             for case in _geometry_cases(angle, density, rng)]
    strata = {}
    for density in ("dense", "sparse"):
        for location in NOISE_LOCATIONS:
            selected = [case for case in cases if case["density"] == density and case["noise_location"] == location]
            completed = [case for case in selected if case["discovery_complete"]]
            strata[density + "/" + location] = {
                "cases": len(selected),
                "incomplete_cases": sum(not case["search_complete"] for case in selected),
                "discovery_cap_cases": sum(case["discovery_reasons"] == ["REGION_CANDIDATE_BUDGET_EXCEEDED"] for case in selected),
                "complete_discovery_cases": len(completed),
                "unique_support_queries": sum(case["unique_support_queries"] for case in selected),
                "retained_seed_count": sum(case["retained_seed_count"] for case in selected),
                "pure_flange_seed_count_in_complete_discoveries": sum(case["pure_flange_seed_count"] for case in completed),
                "fixed_plane_full_flange_membership_seed_count": sum(case["fixed_plane_full_flange_membership_seed_count"] for case in completed),
                "fixed_plane_missed_true_flange_memberships": sum(case["fixed_plane_missed_true_flange_memberships"] for case in completed),
                "fixed_plane_true_flange_membership_denominator": sum(case["fixed_plane_true_flange_membership_denominator"] for case in completed),
                "retained_extended_region_count": sum(case["retained_extended_region_count"] for case in selected),
                "full_constructed_flange_extension_count": sum(case["full_constructed_flange_extension_count"] for case in selected),
                "retained_association_count": sum(case["retained_association_count"] for case in selected),
                "matched_whole_joint_association_count": sum(case["matched_whole_joint_association_count"] for case in selected),
                "unmatched_association_count": sum(case["unmatched_association_count"] for case in selected),
                "reasons": dict(sorted(Counter(reason for case in selected for reason in case["reasons"]).items())),
            }
    return {"cases": len(cases), "seed": SEED, "epsilon": EPSILON,
            "interpretation": "SYNTHETIC_STAGE_ATTRIBUTION_NOT_ACCURACY_OR_PHYSICAL_GENERALIZATION",
            "strata": strata, "case_receipts": cases}


@pytest.fixture(scope="module")
def attribution():
    return run_noise_birth_attribution()


def test_declared_matrix_and_noise_locations_have_exact_denominators(attribution):
    assert attribution["cases"] == 48
    assert attribution["seed"] == SEED and attribution["epsilon"] == EPSILON
    assert len(attribution["strata"]) == 8
    for case in attribution["case_receipts"]:
        assert case["positions_perturbed"] == {"exact": 0, "flanges_only": 50,
                                               "transition_only": 70, "all": 120}[case["noise_location"]]
        assert 0 <= case["maximum_error_over_epsilon"] <= 1
        assert case["seed_queries"] == 120
        assert sum(case["query_status_counts"].values()) == 120
        assert sum(case["query_support_size_counts"].values()) == 120
        assert case["retained_association_count"] == case["matched_whole_joint_association_count"] + case["unmatched_association_count"]
    for stratum in attribution["strata"].values():
        assert stratum["cases"] == 6
        assert stratum["discovery_cap_cases"] <= stratum["incomplete_cases"]
    json.dumps(attribution, allow_nan=False)


def test_overflow_keeps_lower_bound_distinct_from_complete_count(attribution):
    for case in attribution["case_receipts"]:
        if not case["discovery_complete"]:
            assert case["discovery_reasons"] == ["REGION_CANDIDATE_BUDGET_EXCEEDED"]
            assert case["accepted_seed_count_lower_bound"] == 65
            assert case["accepted_seed_count_if_complete"] is None
            assert case["retained_seed_count"] == 0
            assert case["pure_flange_seed_count"] is None
            assert case["fixed_plane_true_flange_membership_denominator"] is None
        else:
            assert case["accepted_seed_count_if_complete"] == case["retained_seed_count"] <= 64
            assert case["fixed_plane_missed_true_flange_memberships"] <= case["fixed_plane_true_flange_membership_denominator"]
            assert case["fixed_plane_true_flange_membership_denominator"] == 25 * case["pure_flange_seed_count"]


def test_attribution_receipt_records_all_strata_without_a_positive_rate_gate(attribution):
    print(json.dumps(attribution["strata"], sort_keys=True))
    print(json.dumps({"capped_cases": [
        {key: case[key] for key in ("density", "angle_degrees", "noise_location",
                                    "unique_support_queries", "accepted_seed_count_lower_bound",
                                    "query_support_size_counts")}
        for case in attribution["case_receipts"] if not case["discovery_complete"]
    ]}, sort_keys=True))
    assert {case["angle_degrees"] for case in attribution["case_receipts"]} == set(ANGLES)
    assert {case["noise_location"] for case in attribution["case_receipts"]} == set(NOISE_LOCATIONS)


def test_fixed_stage_attribution_regression_receipt(attribution):
    # Recorded after the predeclared first run; this freezes diagnostics without
    # making them accuracy targets or changing any inference policy.
    columns = (
        "discovery_cap_cases", "unique_support_queries", "retained_seed_count",
        "pure_flange_seed_count_in_complete_discoveries",
        "fixed_plane_full_flange_membership_seed_count",
        "fixed_plane_missed_true_flange_memberships",
        "fixed_plane_true_flange_membership_denominator",
        "retained_extended_region_count", "full_constructed_flange_extension_count",
        "retained_association_count", "matched_whole_joint_association_count",
        "unmatched_association_count",
    )
    observed = {name: tuple(values[column] for column in columns)
                for name, values in attribution["strata"].items()}
    assert observed == {
        "dense/exact": (0, 494, 238, 188, 188, 0, 4700, 22, 12, 30, 6, 24),
        "dense/flanges_only": (0, 494, 238, 188, 74, 306, 4700, 110, 12, 408, 6, 402),
        "dense/transition_only": (2, 573, 136, 136, 136, 0, 3400, 8, 8, 4, 4, 0),
        "dense/all": (2, 573, 136, 136, 54, 215, 3400, 63, 8, 227, 4, 223),
        "sparse/exact": (0, 486, 250, 200, 200, 0, 5000, 22, 8, 16, 4, 12),
        "sparse/flanges_only": (0, 468, 232, 182, 67, 331, 4550, 101, 6, 369, 2, 367),
        "sparse/transition_only": (2, 563, 136, 136, 136, 0, 3400, 8, 8, 4, 4, 0),
        "sparse/all": (2, 545, 124, 124, 40, 256, 3100, 72, 6, 357, 2, 355),
    }
