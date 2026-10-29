import copy
import pytest
from backend.cad_finite_trimmed_cylinder_domain_v1 import certify_graph_strip


def span(points, edge, weights=None):
    return dict(poles=points, weights=weights or [1.] * len(points), native_edge_id=edge, actual_interval_proven=True)


def boundary():
    return [[span([[0, 0], [0, 10]], 1),
             span([[0, 10], [1, 12], [2, 10]], 2, [1, .7, 1]),
             span([[2, 10], [2, 0]], 3),
             span([[2, 0], [1, -2], [0, 0]], 4, [1, 2, 1])]]


def certify(wires):
    return certify_graph_strip(wires, native_brep_valid=True, exact_active_domains=True)


def test_rational_curved_strip_retains_true_boundary_not_rectangle():
    b = boundary(); result = certify(b)
    assert result["status"] == "PROVED"
    assert result["positive_strip_width_lower_bound_mm"] > 9.9
    assert result["unfold_certificate"] is False
    assert result["ordered_original_boundary"][1]["poles"] == b[0][1]["poles"]
    assert result["angular_support_metric_mm"] == [0, 2]
    core = result["safe_rectangular_interior_core_mm"]
    assert core[:2] == [0, 2] and 0 < core[2] < core[3] < 10
    assert result["core_semantics"] == "SUBSET_OF_CURVED_MATERIAL_NOT_COMPLETE_FOOTPRINT"


def test_reversed_loop_and_cycle_shift_invariant():
    b = boundary()[0]
    for shift in range(4):
        reordered = b[shift:] + b[:shift]
        assert certify([reordered])["status"] == "PROVED"
        reverse = [{**s, "poles": list(reversed(s["poles"])), "weights": list(reversed(s["weights"]))} for s in reversed(reordered)]
        assert certify([reverse])["status"] == "PROVED"


@pytest.mark.parametrize("mutation,reason", [
    (lambda b: b.append(copy.deepcopy(b[0])), "HOLES_OR_MULTIPLE_WIRES_UNSUPPORTED"),
    (lambda b: b[0][1]["poles"].__setitem__(1, [3, 12]), "END_CHAIN_SINGLE_VALUEDNESS_UNPROVEN"),
    (lambda b: b[0][1]["weights"].__setitem__(1, -1), "NONPOSITIVE_RATIONAL_WEIGHT"),
    (lambda b: b[0][1]["poles"].__setitem__(0, [.01, 10]), "ORDERED_WIRE_NOT_EXACTLY_CLOSED"),
    (lambda b: b[0][3]["poles"].__setitem__(1, [1, 11]), "END_CHAINS_NOT_CERTIFIABLY_SEPARATED"),
])
def test_negative_controls(mutation, reason):
    b = boundary(); mutation(b)
    result = certify(b)
    assert result["status"] == "REFUSED" and result["reason"] == reason


def test_native_validity_and_exact_interval_receipts_required():
    assert certify_graph_strip(boundary(), native_brep_valid=False, exact_active_domains=True)["status"] == "REFUSED"
    assert certify_graph_strip(boundary(), native_brep_valid=True, exact_active_domains=False)["status"] == "REFUSED"


@pytest.mark.parametrize("mutation", [
    lambda b: b[0][0]["poles"].insert(1, [0, -1]),
    lambda b: b[0][1]["poles"][1].__setitem__(1, float("nan")),
    lambda b: b[0][1]["weights"].__setitem__(1, float("inf")),
])
def test_malformed_or_backtracking_coefficients_refuse(mutation):
    b = boundary(); mutation(b)
    assert certify(b)["status"] == "REFUSED"


def test_budgets_refuse_without_partial_success():
    assert certify_graph_strip(boundary(), native_brep_valid=True, exact_active_domains=True, maximum_spans=3)["status"] == "REFUSED"
    assert certify_graph_strip(boundary(), native_brep_valid=True, exact_active_domains=True, maximum_poles=4)["status"] == "REFUSED"


def test_multispan_end_graph_with_different_rational_weights():
    b = boundary()
    b[0][1:2] = [span([[0, 10], [.5, 12], [1, 11]], 2, [1, .01, 2]),
                  span([[1, 11], [1.5, 12], [2, 10]], 2, [2, 100, 1])]
    assert certify(b)["status"] == "PROVED"
