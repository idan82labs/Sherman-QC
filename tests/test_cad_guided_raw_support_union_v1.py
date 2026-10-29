"""Independent invented-only tests for the conditional raw-support union planner.

Authored source-only: no import, collection, execution or geometry experiment
was used to construct these expectations. The XYZ rows are merely distinct
coordinate labels, deliberately not valid circle/flange fitting fixtures.

These tests establish set/parity/provenance proposal contracts only. They do
not establish a common source context, finite support, a physical bend identity,
an independently successful old fit, or production/count authority. Real role-
filter exclusions and byte lineage remain a future measurement-wrapper gate.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
import importlib
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(fit_model="free_circle", max_fit_residual_mm=.05,
    max_tangency_error_mm=.1, min_arc_sweep_deg=20., max_circle_condition=1e6,
    min_plane_spread_ratio=1e-8, max_normal_deviation_deg=10.)
BUDGET_MAXIMA = dict(max_records=512, max_raw_rows=1_000_000,
    max_input_role_rows=4_000_000, max_ownership_links=4_000_000,
    max_parity_links=4_000_000, max_set_row_work=32_000_000,
    max_member_comparisons=262_144, max_ordering_work=128_000_000,
    max_input_binding_bytes=64*1024**2, max_plan_bytes=8*1024**2,
    max_working_reservation_bytes=4*1024**3)
BUDGET_FIELDS = ("max_records", "max_raw_rows", "max_input_role_rows",
    "max_ownership_links", "max_parity_links", "max_set_row_work",
    "max_member_comparisons", "max_ordering_work", "max_input_binding_bytes",
    "max_plan_bytes", "max_working_reservation_bytes")
REQUIRED_DOWNSTREAM = ["COMPLETE_ORIGINAL_CONTEXT_PUBLICATION",
    "ALL_CONTRIBUTING_ASSOCIATION_ROLE_FILTER_EXCLUSIONS",
    "ORIGINAL_BYTE_AND_RAW_ROW_LINEAGE",
    "ONE_FRESH_WHOLE_SUPPORT_FIT_AND_FINITE_SUPPORT_CHECKS"]


@pytest.fixture(scope="module")
def planner():
    # Do not import backend/__init__.py or a server/application package.
    prefix = "_invented_raw_support_union_v1_backend"
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
    package = ModuleType(prefix)
    package.__package__, package.__path__ = prefix, [str(ROOT / "backend")]
    sys.modules[prefix] = package
    try:
        yield importlib.import_module(prefix + ".cad_guided_raw_support_union_v1")
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def _points(count=32):
    # Collinear labels are sufficient for set/alias tests, not fitting evidence.
    return np.column_stack((np.arange(count, dtype="<f8"), np.zeros(count), np.zeros(count)))


def _record(number, arc, first, second, **metadata):
    return dict(record_id=f"{number:064x}",
        raw_support=dict(arc_rows=sorted(arc), flange_rows=[sorted(first), sorted(second)]),
        fit_policy=deepcopy(POLICY), **metadata)


def _non_nested():
    return [
        _record(1, [0, 3], [10, 13], [20, 23]),
        _record(2, [0, 4], [10, 14], [20, 24]),
        _record(3, [0, 5], [10, 15], [20, 25]),
    ]


def _roles(raw):
    """Comparison modulo the physically unassigned global flange-label flip."""
    return tuple(raw["arc_rows"]), tuple(sorted(tuple(r) for r in raw["flange_rows"]))


def _flip(record):
    answer = deepcopy(record)
    answer["raw_support"]["flange_rows"].reverse()
    return answer


def _labelings(records):
    """Independent tiny brute-force oracle, not the planner's graph algorithm.

    Fix x_0=0 to remove only the global label gauge. A labeling is consistent
    iff no shared raw flange ID is assigned both global labels. At most four
    invented records are admitted here; this is not a large search fixture.
    """
    assert 1 <= len(records) <= 4
    accepted = []
    for tail in product((0, 1), repeat=len(records)-1):
        assignment = (0, *tail)
        union = [set(), set()]
        for record, flip in zip(records, assignment):
            for side, rows in enumerate(record["raw_support"]["flange_rows"]):
                union[side ^ flip].update(rows)
        if not union[0].intersection(union[1]):
            accepted.append(assignment)
    return accepted


def _call(planner, points=None, records=None, **kwargs):
    points = _points() if points is None else points
    records = _non_nested() if records is None else records
    old_points, old_records = points.copy(), deepcopy(records)
    answer = planner.plan_raw_support_unions_v1(points, records, **kwargs)
    np.testing.assert_array_equal(points, old_points)
    assert records == old_records
    assert answer["schema"] == "RAW_SUPPORT_UNION_PLAN_V1"
    assert answer["measurement_admissible"] is False
    assert answer["production_use_allowed"] is False
    assert answer["observed_bend_count"] is None
    assert answer["physical_event_identity"] == "NOT_ESTABLISHED"
    assert answer["required_downstream_validations"] == REQUIRED_DOWNSTREAM
    assert answer["downstream_validation_status"] == "NOT_PERFORMED_MANDATORY_BEFORE_MEASUREMENT"
    assert answer["source_binding_scope"] == "CANONICAL_POINTS_AND_STRUCTURAL_RAW_ROLES_POLICY_ONLY"
    assert answer["memory_semantics"] == "PROSPECTIVE_ACCOUNTED_RESERVATION_NOT_MEASURED_RSS"
    assert all(group["measurement_admissible"] is False for group in answer["groups"])
    return answer


def _complete(answer):
    assert answer["status"] == "COMPLETE"
    assert answer.get("discarded_plan_prefix") is not True
    return answer["groups"]


def _only(answer):
    groups = _complete(answer)
    assert len(groups) == 1
    return groups[0]


def _refused_group(answer, reason):
    group = _only(answer)
    assert group["status"] == "UNAVAILABLE"
    assert reason in group["reasons"]
    assert group["raw_support"] is None and group["common_anchor_raw_support"] is None
    assert group["cover_record_id"] is None and group["proposal_lane"] is None
    return group


def _fatal(answer):
    assert answer["status"] == "UNAVAILABLE"
    assert answer["groups"] == []
    assert answer["discarded_plan_prefix"] is True
    assert answer["reasons"]


def _full_alignment(group, records):
    actual = group["raw_support"]
    members = {r["record_id"]: r for r in records}
    assert group["member_record_ids"] == sorted(members)
    permutations = group["member_flange_permutations"]
    assert set(permutations) == set(members)
    arc, flanges = set(), [set(), set()]
    for identity, record in members.items():
        permutation = permutations[identity]
        assert type(permutation) is list and len(permutation) == 2
        assert all(type(value) is int for value in permutation)
        assert sorted(permutation) == [0, 1]
        arc.update(record["raw_support"]["arc_rows"])
        for side, rows in enumerate(record["raw_support"]["flange_rows"]):
            flanges[permutation[side]].update(rows)
    assert actual == dict(arc_rows=sorted(arc), flange_rows=[sorted(r) for r in flanges])
    assert not arc.intersection(flanges[0] | flanges[1])
    assert not flanges[0].intersection(flanges[1])


def test_complete_non_nested_union_has_all_rows_and_nary_anchors(planner):
    records = _non_nested()
    assert _labelings(records) == [(0, 0, 0)]
    group = _only(_call(planner, records=records))
    assert group["status"] == "UNION_PROPOSED"
    assert group["proposal_lane"] == "WHOLE_COMPONENT_RAW_ROLE_UNION"
    assert group["cover_record_id"] is None
    assert _roles(group["raw_support"]) == ((0, 3, 4, 5), ((10, 13, 14, 15), (20, 23, 24, 25)))
    assert _roles(group["common_anchor_raw_support"]) == ((0,), ((10,), (20,)))
    assert group["fit_policy"] == POLICY
    evidence = group["parity_evidence"]
    assert evidence["relative_component_count"] == 1
    assert evidence["constraint_consistent"] is True
    assert evidence["checked_star_links"] == 4 and evidence["common_anchor_counts"] == [1, 1, 1]
    assert evidence["fallback_reason"] == "NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER"
    _full_alignment(group, records)


@pytest.mark.parametrize("order", [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
def test_input_record_order_is_not_a_union_choice(planner, order):
    records = _non_nested()
    expected = _only(_call(planner, records=records))
    actual = _only(_call(planner, records=[records[i] for i in order]))
    assert actual == expected


@pytest.mark.parametrize("flips", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                  (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
def test_local_and_global_role_flips_preserve_unordered_union_and_anchors(planner, flips):
    records = [_flip(r) if flip else r for r, flip in zip(_non_nested(), flips)]
    # The oracle fixes record0's gauge; the implementation may choose its own
    # deterministic global gauge, but must preserve full roles and permutations.
    assert _labelings(records) == [tuple(flip ^ flips[0] for flip in flips)]
    group = _only(_call(planner, records=records))
    assert group["status"] == "UNION_PROPOSED"
    assert group["raw_support"] == dict(arc_rows=[0, 3, 4, 5],
        flange_rows=[[10, 13, 14, 15], [20, 23, 24, 25]])
    assert group["common_anchor_raw_support"] == dict(arc_rows=[0], flange_rows=[[10], [20]])
    assert group["parity_evidence"]["global_flip_rule"] == "LEXICOGRAPHIC_SORTED_RAW_FLANGE_UNIONS"
    for record, flip in zip(records, flips):
        assert group["member_flange_permutations"][record["record_id"]] == [flip, 1 ^ flip]
    _full_alignment(group, records)


@pytest.mark.parametrize("shape", ["chain", "cycle"])
def test_consistent_parity_without_global_anchors_refuses_for_anchors(planner, shape):
    records = [
        _record(1, [0, 3], [10, 11], [20, 21]),
        _record(2, [0, 4], [11, 12], [21, 22]),
        _record(3, [0, 5], [12, 13] if shape == "chain" else [10, 12],
                [22, 23] if shape == "chain" else [20, 22]),
    ]
    assert _labelings(records) == [(0, 0, 0)]
    group = _refused_group(_call(planner, records=records), "NO_COMMON_RAW_ROLE_ANCHORS")
    assert "INCONSISTENT_GLOBAL_FLANGE_PARITY" not in group["reasons"]
    assert "DISCONNECTED_GLOBAL_FLANGE_PARITY" not in group["reasons"]
    assert group["parity_evidence"]["constraint_consistent"] is True
    assert group["parity_evidence"]["relative_component_count"] == 1
    assert group["parity_evidence"]["common_anchor_counts"] == [1, 0, 0]


@pytest.mark.parametrize("kind", ["odd_cycle", "parallel_contradiction"])
def test_inconsistent_parity_never_drops_an_edge_or_member(planner, kind):
    if kind == "odd_cycle":
        records = [
            _record(1, [0, 3], [10, 11], [20, 21]),
            _record(2, [0, 4], [11, 12], [21, 22]),
            _record(3, [0, 5], [12, 20], [10, 22]),
        ]
    else:
        records = [_record(1, [0, 3], [10], [20]),
                   _record(2, [0, 4], [10, 20], [30])]
    assert _labelings(records) == []
    group = _refused_group(_call(planner, records=records), "INCONSISTENT_GLOBAL_FLANGE_PARITY")
    assert group["member_record_ids"] == sorted(r["record_id"] for r in records)
    assert group["parity_evidence"]["constraint_consistent"] is False


@pytest.mark.parametrize("partly_connected", [False, True])
def test_disconnected_relative_parity_does_not_choose_arbitrary_component_flips(planner, partly_connected):
    records = [_record(1, [0, 3], [10, 13], [20, 23]),
               _record(2, [0, 4], [11, 14], [21, 24])]
    if partly_connected:
        records.append(_record(3, [0, 5], [10, 15], [20, 25]))
    assert len(_labelings(records)) > 1
    group = _refused_group(_call(planner, records=records), "DISCONNECTED_GLOBAL_FLANGE_PARITY")
    assert group["parity_evidence"]["constraint_consistent"] is True
    assert group["parity_evidence"]["relative_component_count"] == 2


@pytest.mark.parametrize("role", ["arc", "first_flange", "second_flange"])
def test_pairwise_role_overlap_is_not_a_common_nary_anchor(planner, role):
    records = _non_nested()
    if role == "arc":
        replacement = [[0, 1], [1, 2], [0, 2]]
        for r, rows in zip(records, replacement): r["raw_support"]["arc_rows"] = rows
    else:
        index, base = (0, 10) if role == "first_flange" else (1, 20)
        replacement = [[base, base+1], [base+1, base+2], [base, base+2]]
        for r, rows in zip(records, replacement): r["raw_support"]["flange_rows"][index] = rows
    assert all(set(replacement[i]) & set(replacement[j]) for i, j in ((0, 1), (1, 2), (0, 2)))
    assert not set.intersection(*(set(rows) for rows in replacement))
    assert len(_labelings(records)) == 1
    _refused_group(_call(planner, records=records), "NO_COMMON_RAW_ROLE_ANCHORS")


@pytest.mark.parametrize("coordinate_alias", [False, True])
def test_arc_flange_conflict_is_a_whole_component_veto(planner, coordinate_alias):
    points = _points()
    records = [_record(1, [0, 3], [10, 13], [20, 23]),
               _record(2, [4, 5], [11, 14], [21, 24])]
    if coordinate_alias:
        points[11] = points[0]  # distinct IDs, exact same XYZ, opposite roles
    else:
        records[1]["raw_support"]["flange_rows"][0] = [0, 14]
    group = _refused_group(_call(planner, points, records), "CROSS_ROLE_POSITION_CONFLICT")
    assert group["member_record_ids"] == sorted(r["record_id"] for r in records)


def test_policy_mismatch_cannot_be_hidden_by_a_larger_union(planner):
    records = _non_nested()
    records[1]["fit_policy"]["max_fit_residual_mm"] = .1
    _refused_group(_call(planner, records=records), "INCOMPATIBLE_MEMBER_FIT_POLICIES")


def test_previous_fit_failure_metadata_does_not_select_members(planner):
    records = _non_nested()
    before = _call(planner, records=records)
    for index, record in enumerate(records):
        record.update(status="UNAVAILABLE" if index != 1 else "MEASURED_CONDITIONAL",
            reasons=["INVENTED_PRIOR_FIT_FAILURE"] if index != 1 else [],
            measurements=None, finite_support=None)
    after = _call(planner, records=records)
    group = _only(after)
    assert group["status"] == "UNION_PROPOSED"
    assert group["member_record_ids"] == sorted(r["record_id"] for r in records)
    assert _roles(group["raw_support"]) == ((0, 3, 4, 5), ((10, 13, 14, 15), (20, 23, 24, 25)))
    _full_alignment(group, records)
    assert after["structural_records_sha256"] == before["structural_records_sha256"]
    # The binding intentionally does not attest ignored old answers/statuses.
    # Complete original context and byte lineage remain mandatory downstream.


@pytest.mark.parametrize("role", ["arc", "flange"])
def test_duplicate_xyz_with_distinct_union_raw_ids_is_not_coordinate_deduplication(planner, role):
    points = _points()
    first, second = (3, 4) if role == "arc" else (13, 14)
    points[second] = points[first]
    group = _refused_group(_call(planner, points, _non_nested()), "DUPLICATE_POSITION_IN_UNION_SUPPORT")
    assert len(group["member_record_ids"]) == 3


def test_shared_flange_context_alone_does_not_merge_arc_ownership(planner):
    records = [_record(1, [0, 3], [10, 13], [20, 23]),
               _record(2, [1, 4], [10, 14], [20, 24])]
    groups = _complete(_call(planner, records=records))
    assert len(groups) == 2
    assert all(g["status"] == "EXISTING_COVER" for g in groups)
    assert {tuple(g["member_record_ids"]) for g in groups} == {(r["record_id"],) for r in records}
    assert {g["cover_record_id"] for g in groups} == {r["record_id"] for r in records}


@pytest.mark.parametrize("old_fit_failed", [False, True])
def test_existing_cover_lane_is_not_replaced_by_union_based_on_prior_fit_status(planner, old_fit_failed):
    records = _non_nested()
    cover = _record(9, [0, 3, 4, 5], [10, 13, 14, 15], [20, 23, 24, 25],
        status="UNAVAILABLE" if old_fit_failed else "MEASURED_CONDITIONAL",
        reasons=["INVENTED_OLD_FIT_FAILURE"] if old_fit_failed else [],
        measurements=None, finite_support=None)
    records.append(cover)
    group = _only(_call(planner, records=records))
    assert group["status"] == "EXISTING_COVER"
    assert group["proposal_lane"] == "EXISTING_RAW_SUPPORT_COVER"
    assert group["cover_record_id"] == cover["record_id"]
    assert _roles(group["raw_support"]) == _roles(cover["raw_support"])
    _full_alignment(group, records)
    assert group["parity_evidence"]["fallback_evaluated"] is False
    # This proves status-independent planner branching, not behavior after a
    # future measurement wrapper's fresh existing-cover fit actually fails.


def test_existing_cover_keeps_duplicate_position_geometry_for_downstream_refusal(planner):
    points, records = _points(), _non_nested()
    points[4] = points[3]
    cover = _record(9, [0, 3, 4, 5], [10, 13, 14, 15], [20, 23, 24, 25])
    records.append(cover)
    group = _only(_call(planner, points, records))
    assert group["status"] == "EXISTING_COVER"
    assert group["cover_record_id"] == cover["record_id"]
    assert group["raw_support"]["arc_rows"] == [0, 3, 4, 5]
    assert "DUPLICATE_POSITION_IN_UNION_SUPPORT" not in group["reasons"]
    # The fallback-only duplicate-position veto must not silently change the
    # old existing-cover set contract. No geometry admission follows here.


def test_default_budget_values_are_the_declared_hard_maxima(planner):
    budgets = planner.RawSupportUnionBudgetsV1()
    assert {f.name for f in fields(budgets)} == set(BUDGET_FIELDS)
    assert asdict(budgets) == BUDGET_MAXIMA
    assert all(type(value) is int for value in asdict(budgets).values())
    assert _only(_call(planner, budgets=budgets))["status"] == "UNION_PROPOSED"


@pytest.mark.parametrize("field", BUDGET_FIELDS)
@pytest.mark.parametrize("kind", ["boolean", "float", "above_hard_max"])
def test_noninteger_and_over_hard_max_budgets_cannot_admit_a_plan(planner, field, kind):
    value = True if kind == "boolean" else float(BUDGET_MAXIMA[field]) if kind == "float" else BUDGET_MAXIMA[field]+1
    try:
        budgets = replace(planner.RawSupportUnionBudgetsV1(), **{field: value})
    except (TypeError, ValueError):
        return  # exact typed admission may be enforced by construction itself
    _fatal(_call(planner, budgets=budgets))


@pytest.mark.parametrize("field,exact", [("max_records", 3), ("max_raw_rows", 32)])
def test_exact_input_budget_and_one_row_or_record_over_is_fail_closed(planner, field, exact):
    defaults = planner.RawSupportUnionBudgetsV1()
    exact_budget = replace(defaults, **{field: exact})
    assert _only(_call(planner, budgets=exact_budget))["status"] == "UNION_PROPOSED"
    _fatal(_call(planner, budgets=replace(defaults, **{field: exact-1})))


@pytest.mark.parametrize("field,counter,exact", [
    ("max_input_role_rows", "input_role_rows", 18),
    ("max_ownership_links", "ownership_links", 2),
    ("max_parity_links", "parity_links", 4),
    ("max_set_row_work", "set_row_work", 213),
    ("max_ordering_work", "ordering_work", 110),
    ("max_plan_bytes", "plan_bytes_reserved", 11128),
    ("max_working_reservation_bytes", "working_reservation_bytes", 26259712),
])
def test_exact_tiny_union_work_reservation_and_one_under(planner, field, counter, exact):
    # Independent arithmetic for three records with six memberships apiece:
    # ownership has two shared-arc star links; two shared flanges give four
    # parity links. Set-work is 54+6+30+30+6+18+18+15+24+12 = 213;
    # ordering is 12+18+72+6+2 = 110. The plan reservation is
    # 8192+2048+3*256+8*(12+3) = 11128. Working reservation includes
    # 3*768 point bytes, three record envelopes, 18 memberships and 3*8MiB.
    defaults = planner.RawSupportUnionBudgetsV1()
    answer = _call(planner, budgets=replace(defaults, **{field: exact}))
    assert _only(answer)["status"] == "UNION_PROPOSED"
    assert answer["work"][counter] == exact
    refused = _call(planner, budgets=replace(defaults, **{field: exact-1}))
    _fatal(refused)
    assert refused["work"][counter] == exact


def test_structural_input_binding_exact_encoded_byte_boundary(planner):
    records = _non_nested()
    # Construct the independent minimal structural view: the invented records
    # already contain only the three admitted fields, in identity order.
    exact = len(json.dumps(records, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8"))
    defaults = planner.RawSupportUnionBudgetsV1()
    answer = _call(planner, records=records,
        budgets=replace(defaults, max_input_binding_bytes=exact))
    assert _only(answer)["status"] == "UNION_PROPOSED"
    assert answer["work"]["input_binding_bytes"] == exact
    refused = _call(planner, records=records,
        budgets=replace(defaults, max_input_binding_bytes=exact-1))
    _fatal(refused)
    assert refused["work"]["input_binding_bytes"] == exact


def test_existing_cover_member_comparison_exact_and_one_under(planner):
    records = [_record(1, [0], [10], [20])]
    defaults = planner.RawSupportUnionBudgetsV1()
    answer = _call(planner, records=records,
        budgets=replace(defaults, max_member_comparisons=1))
    assert _only(answer)["status"] == "EXISTING_COVER"
    assert answer["work"]["member_comparisons"] == 1
    refused = _call(planner, records=records,
        budgets=replace(defaults, max_member_comparisons=0))
    _fatal(refused)
    assert refused["work"]["member_comparisons"] == 1


def test_zero_count_caps_and_minimum_report_envelope_admit_empty_plan(planner):
    zero_fields = ("max_records", "max_input_role_rows", "max_ownership_links",
        "max_parity_links", "max_set_row_work", "max_member_comparisons", "max_ordering_work")
    budgets = replace(planner.RawSupportUnionBudgetsV1(), max_raw_rows=1,
        max_plan_bytes=8192, **{field: 0 for field in zero_fields})
    answer = _call(planner, points=_points(1), records=[], budgets=budgets)
    assert _complete(answer) == []
    assert answer["input_record_count"] == 0 and answer["input_role_row_count"] == 0
    assert answer["work"]["plan_bytes_reserved"] == 8192
    assert all(answer["work"][field.removeprefix("max_")] == 0 for field in zero_fields[1:])


def test_report_envelope_below_minimum_is_invalid_configuration(planner):
    budgets = replace(planner.RawSupportUnionBudgetsV1(), max_plan_bytes=8191)
    answer = _call(planner, points=_points(1), records=[], budgets=budgets)
    _fatal(answer)
    assert answer["work"] == {} and answer["configuration"] is None
    assert any("INVALID_UNION_BUDGET_VALUE" in reason for reason in answer["reasons"])
    # This is a configured failure-envelope minimum, not an actual JSON-size
    # observation or a claim that every nonempty plan fits into 8192 bytes.


def test_late_membership_budget_refusal_keeps_attempted_count_not_a_prefix(planner):
    budgets = replace(planner.RawSupportUnionBudgetsV1(), max_input_role_rows=17)
    answer = _call(planner, budgets=budgets)
    _fatal(answer)
    assert answer["work"]["input_role_rows"] == 18
    assert answer["work"]["set_row_work"] == 36  # two prior six-row records owned
    assert "structural_records_sha256" not in answer and "input_record_count" not in answer


@pytest.mark.parametrize("source_itemsize,expected", [(8, 4198408704), (16, 4270408704)])
def test_full_declared_envelope_memory_arithmetic_without_arrays(planner, source_itemsize, expected):
    # Only scalar accounting is exercised: no million-row array, role lists,
    # geometry or native buffers are allocated, and no RSS claim follows.
    raw_rows, record_count, memberships = 1_000_000, 512, 4_000_000
    point_reservation = 3*max(raw_rows*3*source_itemsize, raw_rows*24)
    independent = (1024**2 + point_reservation + 8192*record_count
        + 1024*memberships + 3*8*1024**2)
    assert independent == expected <= 4*1024**3
    work = planner._Work(asdict(planner.RawSupportUnionBudgetsV1()))
    work.point_reservation, work.record_count = point_reservation, record_count
    work.values["input_role_rows"] = memberships
    work.config["max_working_reservation_bytes"] = expected
    work.memory()
    assert work.values["working_reservation_bytes"] == expected
    work.config["max_working_reservation_bytes"] = expected-1
    with pytest.raises(ValueError, match="UNION_BUDGET:working_reservation_bytes"):
        work.memory()
    assert work.values["working_reservation_bytes"] == expected


@pytest.mark.parametrize("value", [None, {}, SimpleNamespace()])
def test_budget_object_requires_exact_typed_declaration(planner, value):
    _fatal(_call(planner, budgets=value))


@pytest.mark.parametrize("fault", ["bool_row", "out_of_range", "duplicate_record_id", "unsorted_rows"])
def test_late_invalid_record_clears_all_groups_without_a_valid_prefix(planner, fault):
    records = _non_nested()
    if fault == "bool_row": records[-1]["raw_support"]["arc_rows"] = [False, 5]
    elif fault == "out_of_range": records[-1]["raw_support"]["arc_rows"] = [0, 32]
    elif fault == "duplicate_record_id": records[-1]["record_id"] = records[0]["record_id"]
    else: records[-1]["raw_support"]["arc_rows"] = [5, 0]
    _fatal(_call(planner, records=records))


def test_late_final_report_failure_discards_an_already_completed_union_group(planner, monkeypatch):
    original = json.JSONEncoder.iterencode
    observed = []

    def fail_completed_report(self, value, *args, **kwargs):
        if (type(value) is dict and value.get("schema") == "RAW_SUPPORT_UNION_PLAN_V1"
                and value.get("status") == "COMPLETE" and value.get("groups")):
            observed.append(deepcopy(value["groups"]))
            raise ValueError("INVENTED_LATE_COMPLETE_REPORT_FAILURE")
        return original(self, value, *args, **kwargs)

    # The explicit failure is after a complete group exists, not merely an
    # invalid final input record before any output could be constructed.
    with monkeypatch.context() as patched:
        patched.setattr(json.JSONEncoder, "iterencode", fail_completed_report)
        answer = _call(planner)
    assert observed and any(g["status"] == "UNION_PROPOSED" for g in observed[0])
    _fatal(answer)


def test_late_final_ordering_budget_discards_completed_group(planner, monkeypatch):
    original = planner._Work.ordered
    observed = []

    def observe_final_group_order(self, values, *, key=None):
        if (type(values) is list and values and type(values[0]) is dict
                and values[0].get("status") == "UNION_PROPOSED"):
            observed.append(deepcopy(values))
        return original(self, values, key=key)

    budgets = replace(planner.RawSupportUnionBudgetsV1(), max_ordering_work=109)
    with monkeypatch.context() as patched:
        patched.setattr(planner._Work, "ordered", observe_final_group_order)
        answer = _call(planner, budgets=budgets)
    assert observed and observed[0][0]["status"] == "UNION_PROPOSED"
    _fatal(answer)
    assert answer["work"]["ordering_work"] == 110
