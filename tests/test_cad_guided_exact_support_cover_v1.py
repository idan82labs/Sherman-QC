"""Independent invented-data contract tests; no fit/native/physical inputs.

Run only through a separately reviewed bounded source gate with an inert backend
namespace. This file authors no geometry thresholds and makes no count/fit claim.
Original raw IDs determine exact covers; represented positions only connect
components and expose cross-role conflicts. All fixtures are stipulated here.
"""

import copy
import hashlib
import itertools
import json

import numpy as np
import pytest

from backend import cad_guided_exact_support_cover_v1 as subject


POLICY = {"fit_model": "free_circle", "max_fit_residual_mm": .05,
          "max_tangency_error_mm": .1, "min_arc_sweep_deg": 20.,
          "max_circle_condition": 1000., "min_plane_spread_ratio": .01,
          "max_normal_deviation_deg": 15.}


def _points(count=32, dtype=np.float64):
    # Distinct exact integer coordinates; no shape/fitting answer is supplied.
    rows = np.arange(count, dtype=np.int64)
    return np.column_stack((rows, 2*rows+1, 3*rows-2)).astype(dtype)


def _id(number):
    return f"{number:064x}"


def _record(number, arc, first, second, policy=None):
    return {"record_id": _id(number), "raw_support": {
        "arc_rows": list(arc), "flange_rows": [list(first), list(second)]},
        "fit_policy": copy.deepcopy(POLICY if policy is None else policy)}


def _nested_records():
    return [_record(1, [0, 1], [4, 5], [7]),
            _record(2, [1, 2], [8, 9], [5, 6]),
            _record(3, [0, 1, 2, 3], [4, 5, 6], [7, 8, 9])]


def _run(records, points=None):
    return subject.plan_exact_support_covers_v1(_points() if points is None else points, records)


def _one_group(answer, expected_ids, status, cover=None):
    assert answer["status"] == "COMPLETE", answer
    assert len(answer["groups"]) == 1, answer
    group = answer["groups"][0]
    assert group["member_record_ids"] == sorted(_id(i) for i in expected_ids)
    assert group["status"] == status, group
    assert group["cover_record_id"] == (None if cover is None else _id(cover))
    assert type(group["reasons"]) is list
    assert all(type(reason) is str for reason in group["reasons"])
    if status == "COVERED":
        assert group["reasons"] == []
        assert set(group["member_flange_permutations"]) == set(group["member_record_ids"])
    else:
        assert group["reasons"]
        assert not group["member_flange_permutations"]
    return group


def _fatal(answer):
    assert answer["status"] == "UNAVAILABLE", answer
    assert answer["groups"] == [], "A fatal failure must erase even completed component prefixes"
    assert answer["reasons"]
    assert answer["discarded_plan_prefix"] is True
    for name in ("raw_points_sha256", "input_record_count", "input_role_row_count",
                 "input_record_ids", "component_count"):
        assert name not in answer
    assert type(answer["work"]) is dict
    assert all(type(value) is int and value >= 0 for value in answer["work"].values())


def test_nested_partials_have_one_existing_full_raw_cover_and_role_permutations():
    records = _nested_records()
    group = _one_group(_run(records), [1, 2, 3], "COVERED", 3)
    assert group["member_flange_permutations"] == {
        _id(1): [0, 1], _id(2): [1, 0], _id(3): [0, 1]}
    # The covering set is literally a supplied record, not a manufactured union.
    cover = next(record for record in records if record["record_id"] == group["cover_record_id"])
    assert set(cover["raw_support"]["arc_rows"]) == set().union(
        *(set(record["raw_support"]["arc_rows"]) for record in records))


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_component_result_is_invariant_to_all_six_record_orders(order):
    records = _nested_records()
    baseline = _run(records)
    answer = _run([records[index] for index in order])
    assert answer["status"] == "COMPLETE"
    assert answer["groups"] == baseline["groups"]


@pytest.mark.parametrize("reverse_roles", [False, True])
def test_identical_cover_partitions_choose_lexicographic_id_not_input_order(reverse_roles):
    left = _record(9, [0, 1, 2], [3, 4], [5, 6])
    right = _record(2, [0, 1, 2], [5, 6], [3, 4])
    if reverse_roles:
        left["raw_support"]["flange_rows"].reverse()
        right["raw_support"]["flange_rows"].reverse()
    group = _one_group(_run([left, right]), [2, 9], "COVERED", 2)
    assert group["member_flange_permutations"] == {_id(2): [0, 1], _id(9): [1, 0]}


def test_nonnested_arc_overlap_has_no_cover_and_cannot_split_favorably():
    records = [_record(1, [0, 1], [4], [5]), _record(2, [1, 2], [4], [5])]
    _one_group(_run(records), [1, 2], "UNAVAILABLE")


def test_bridge_joins_incomparable_maxima_into_one_unavailable_component():
    records = [_record(1, [0, 1], [6], [7]),
               _record(2, [2, 3], [6], [7]),
               _record(3, [1, 2], [6], [7])]
    _one_group(_run(records), [1, 2, 3], "UNAVAILABLE")


def test_complete_arc_without_complete_flange_union_is_not_a_cover():
    records = [_record(1, [0, 1, 2], [4], [7]),
               _record(2, [0, 1], [4, 5], [7])]
    _one_group(_run(records), [1, 2], "UNAVAILABLE")


def test_total_flange_union_without_direct_role_mapping_is_not_a_cover():
    # Both full raw unions exist, but the child's first role straddles the two
    # covering roles. A total-flange subset test alone would falsely admit it.
    records = [_record(1, [0, 1], [4, 5], [6, 7]),
               _record(2, [0], [4, 6], [5, 7])]
    _one_group(_run(records), [1, 2], "UNAVAILABLE")


def test_identical_total_support_but_incompatible_partitions_cannot_choose_a_winner():
    records = [_record(1, [0, 1], [4, 5], [6, 7]),
               _record(2, [0, 1], [4, 6], [5, 7])]
    _one_group(_run(records), [1, 2], "UNAVAILABLE")


@pytest.mark.parametrize("duplicate_position", [False, True])
def test_arc_to_flange_conflict_links_records_even_without_shared_arc(duplicate_position):
    points = _points()
    second_arc = 4
    if duplicate_position:
        points[10] = points[4]
        second_arc = 10
    records = [_record(1, [0, 1], [4], [5]),
               _record(2, [second_arc], [6], [7])]
    group = _one_group(_run(records, points), [1, 2], "UNAVAILABLE")
    assert "CROSS_ROLE_POSITION_CONFLICT" in group["reasons"]


def test_cross_role_claim_in_an_already_uncoverable_record_is_not_discarded():
    records = [_record(1, [0, 1], [6], [7]),
               _record(2, [1, 2], [6], [7]),
               _record(3, [6], [8], [9]),
               _record(4, [10], [11], [12])]
    records[2].update(status="UNAVAILABLE", reasons=["INVENTED_PRIOR_FIT_FAILURE"])
    answer = _run(records)
    assert answer["status"] == "COMPLETE" and len(answer["groups"]) == 2
    bad = next(group for group in answer["groups"] if _id(1) in group["member_record_ids"])
    good = next(group for group in answer["groups"] if _id(4) in group["member_record_ids"])
    assert bad["member_record_ids"] == [_id(1), _id(2), _id(3)]
    assert bad["status"] == "UNAVAILABLE" and bad["cover_record_id"] is None
    assert "CROSS_ROLE_POSITION_CONFLICT" in bad["reasons"]
    assert good["status"] == "COVERED" and good["cover_record_id"] == _id(4)


@pytest.mark.parametrize("duplicate_position", [False, True])
def test_shared_flange_only_does_not_merge_separate_arc_components(duplicate_position):
    points = _points()
    flange = 4
    if duplicate_position:
        points[10] = points[4]
        flange = 10
    records = [_record(1, [0, 1], [4], [5]),
               _record(2, [2, 3], [flange], [6])]
    answer = _run(records, points)
    assert answer["status"] == "COMPLETE" and len(answer["groups"]) == 2
    assert {tuple(group["member_record_ids"]) for group in answer["groups"]} == {(_id(1),), (_id(2),)}
    assert all(group["status"] == "COVERED" for group in answer["groups"])


def test_position_alias_connectivity_does_not_coalesce_original_raw_ids_for_cover():
    points = _points()
    points[10] = points[0]
    records = [_record(1, [0, 1], [4], [5]), _record(2, [10], [4], [5])]
    _one_group(_run(records, points), [1, 2], "UNAVAILABLE")


def test_existing_member_must_actually_retain_both_duplicate_coordinate_rows():
    points = _points()
    points[10] = points[0]
    records = [_record(1, [0, 1, 10], [4], [5]), _record(2, [10], [4], [5])]
    _one_group(_run(records, points), [1, 2], "COVERED", 1)
    # This is a set-plan result only. A later canonical fit may refuse duplicate
    # positions; the helper cannot infer geometric validity from this cover.


def test_own_record_cross_role_coordinate_alias_is_component_veto_not_fatal():
    points = _points()
    points[4] = points[0]
    group = _one_group(_run([_record(1, [0], [4], [5])], points), [1], "UNAVAILABLE")
    assert "CROSS_ROLE_POSITION_CONFLICT" in group["reasons"]


@pytest.mark.parametrize("changed", [
    {"max_fit_residual_mm": .051}, {"max_tangency_error_mm": .2},
    {"min_arc_sweep_deg": 21.},
])
def test_incompatible_policies_veto_the_entire_connected_component(changed):
    records = _nested_records()
    records[0]["fit_policy"].update(changed)
    _one_group(_run(records), [1, 2, 3], "UNAVAILABLE")


def test_policy_dictionary_order_does_not_change_compatibility():
    records = _nested_records()
    records[0]["fit_policy"] = dict(reversed(list(records[0]["fit_policy"].items())))
    _one_group(_run(records), [1, 2, 3], "COVERED", 3)


def test_different_policies_in_unconnected_components_remain_separate():
    other = dict(POLICY, max_fit_residual_mm=.06)
    records = [_record(1, [0], [4], [5]),
               _record(2, [1], [6], [7], other)]
    answer = _run(records)
    assert answer["status"] == "COMPLETE" and len(answer["groups"]) == 2
    assert all(group["status"] == "COVERED" for group in answer["groups"])


def test_points_and_record_trees_are_not_mutated_or_reused_as_output_containers():
    points, records = _points(), _nested_records()
    before_points, before_records = points.tobytes(), copy.deepcopy(records)
    answer = _run(records, points)
    assert points.tobytes() == before_points and records == before_records
    original_groups = copy.deepcopy(answer["groups"])
    records[0]["raw_support"]["arc_rows"].append(20)
    records[0]["fit_policy"]["max_fit_residual_mm"] = .09
    points[:] = 0
    assert answer["groups"] == original_groups
    json.dumps(answer, allow_nan=False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint32])
def test_supported_numeric_point_dtypes_preserve_exact_set_plan(dtype):
    # Shift before unsigned conversion to avoid manufacturing a large value.
    points = (_points()+4).astype(dtype)
    _one_group(_run(_nested_records(), points), [1, 2, 3], "COVERED", 3)


def test_empty_records_are_complete_only_for_valid_nonempty_points():
    answer = _run([])
    assert answer["status"] == "COMPLETE" and answer["groups"] == []
    _fatal(_run([], np.empty((0, 3), dtype=float)))


@pytest.mark.parametrize("bad", [None, {}, (), "records", [None], [True]])
def test_malformed_record_container_whole_refuses(bad):
    _fatal(_run(bad))


@pytest.mark.parametrize("fault", [
    "id_missing", "id_bool", "id_short", "id_nonhex", "id_upper", "duplicate_record_id",
    "raw_missing", "raw_list", "arc_bool", "arc_negative", "arc_outside", "arc_float",
    "arc_duplicate", "flange_bool", "flange_duplicate", "flange_count", "flange_nonlist",
    "arc_flange_same_id", "flange_flange_same_id", "policy_missing", "policy_list",
    "policy_nan", "policy_infinity", "policy_cycle", "policy_unserializable",
    "arc_empty", "first_empty", "second_empty", "arc_unsorted", "flange_unsorted",
    "policy_wrong_model", "policy_bool", "policy_numpy_scalar", "policy_array",
    "policy_negative", "policy_above_range", "policy_extra", "policy_missing_numeric",
])
def test_malformed_record_fields_whole_clear_even_after_valid_record(fault):
    records = [_record(1, [0], [1], [2]), _record(2, [3], [4], [5])]
    record, raw = records[1], records[1]["raw_support"]
    if fault == "id_missing": record.pop("record_id")
    elif fault == "id_bool": record["record_id"] = True
    elif fault == "id_short": record["record_id"] = "1"
    elif fault == "id_nonhex": record["record_id"] = "z"*64
    elif fault == "id_upper": record["record_id"] = "A"*64
    elif fault == "duplicate_record_id": record["record_id"] = records[0]["record_id"]
    elif fault == "raw_missing": record.pop("raw_support")
    elif fault == "raw_list": record["raw_support"] = []
    elif fault == "arc_bool": raw["arc_rows"] = [True]
    elif fault == "arc_negative": raw["arc_rows"] = [-1]
    elif fault == "arc_outside": raw["arc_rows"] = [32]
    elif fault == "arc_float": raw["arc_rows"] = [3.]
    elif fault == "arc_duplicate": raw["arc_rows"] = [3, 3]
    elif fault == "flange_bool": raw["flange_rows"][0] = [False]
    elif fault == "flange_duplicate": raw["flange_rows"][0] = [4, 4]
    elif fault == "flange_count": raw["flange_rows"] = [[4]]
    elif fault == "flange_nonlist": raw["flange_rows"][0] = (4,)
    elif fault == "arc_flange_same_id": raw["flange_rows"][0] = [3]
    elif fault == "flange_flange_same_id": raw["flange_rows"][1] = [4]
    elif fault == "policy_missing": record.pop("fit_policy")
    elif fault == "policy_list": record["fit_policy"] = []
    elif fault == "policy_nan": record["fit_policy"]["max_fit_residual_mm"] = float("nan")
    elif fault == "policy_infinity": record["fit_policy"]["max_fit_residual_mm"] = float("inf")
    elif fault == "policy_cycle": record["fit_policy"]["self"] = record["fit_policy"]
    elif fault == "policy_unserializable": record["fit_policy"]["max_fit_residual_mm"] = object()
    elif fault == "arc_empty": raw["arc_rows"] = []
    elif fault == "first_empty": raw["flange_rows"][0] = []
    elif fault == "second_empty": raw["flange_rows"][1] = []
    elif fault == "arc_unsorted": raw["arc_rows"] = [7, 3]
    elif fault == "flange_unsorted": raw["flange_rows"][0] = [8, 4]
    elif fault == "policy_wrong_model": record["fit_policy"]["fit_model"] = "wrong"
    elif fault == "policy_bool": record["fit_policy"]["max_fit_residual_mm"] = True
    elif fault == "policy_numpy_scalar": record["fit_policy"]["max_fit_residual_mm"] = np.float64(.05)
    elif fault == "policy_array": record["fit_policy"]["max_fit_residual_mm"] = np.array([.05])
    elif fault == "policy_negative": record["fit_policy"]["max_fit_residual_mm"] = -.05
    elif fault == "policy_above_range": record["fit_policy"]["max_fit_residual_mm"] = 1e12+1.
    elif fault == "policy_extra": record["fit_policy"]["extra"] = 1.
    elif fault == "policy_missing_numeric": record["fit_policy"].pop("max_fit_residual_mm")
    else: raise AssertionError("UNDECLARED_FAULT")
    _fatal(_run(records))


@pytest.mark.parametrize("fault", ["list", "shape", "flat", "complex", "object", "bool",
    "nan", "infinity", "above_bound", "below_bound"])
def test_malformed_points_whole_refuse(fault):
    points = _points()
    if fault == "list": points = points.tolist()
    elif fault == "shape": points = np.zeros((32, 2))
    elif fault == "flat": points = points.ravel()
    elif fault == "complex": points = points.astype(complex)
    elif fault == "object": points = points.astype(object)
    elif fault == "bool": points = points.astype(bool)
    elif fault == "nan": points[0, 0] = float("nan")
    elif fault == "infinity": points[0, 0] = float("inf")
    elif fault == "above_bound": points[0, 0] = np.nextafter(1e6, float("inf"))
    elif fault == "below_bound": points[0, 0] = np.nextafter(-1e6, -float("inf"))
    _fatal(_run(_nested_records(), points))


@pytest.mark.parametrize("edge", [-1e6, 1e6])
def test_exact_coordinate_admission_boundary_is_inclusive(edge):
    points = _points()
    points[31, 2] = edge  # Unused rows are still in whole-array admission scope.
    _one_group(_run(_nested_records(), points), [1, 2, 3], "COVERED", 3)


def test_ndarray_subclass_refuses_before_any_coercion(monkeypatch):
    class Subclass(np.ndarray):
        pass
    points = _points().view(Subclass)
    def forbidden(*args, **kwargs):
        raise AssertionError("COERCION_MUST_NOT_RUN_FOR_SUBCLASS")
    monkeypatch.setattr(subject.np, "asarray", forbidden)
    answer = _run(_nested_records(), points)
    _fatal(answer)
    assert "COVER_POINTS_REQUIRE_NDARRAY" in answer["reasons"][0]


def test_complete_plan_binds_all_original_records_rows_and_points():
    points, records = _points(), _nested_records()
    answer = _run(records, points)
    group = _one_group(answer, [1, 2, 3], "COVERED", 3)
    assert answer["input_record_ids"] == [_id(1), _id(2), _id(3)]
    assert answer["input_record_count"] == 3 and answer["input_role_row_count"] == 21
    assert answer["component_count"] == 1
    assert answer["raw_points_sha256"] == hashlib.sha256(points.astype("<f8").tobytes()).hexdigest()
    encoded_ids = json.dumps([_id(1), _id(2), _id(3)], sort_keys=True, separators=(",", ":")).encode()
    assert group["group_id"] == hashlib.sha256(encoded_ids).hexdigest()
    assert answer["production_use_allowed"] is False and answer["observed_bend_count"] is None
    assert answer["physical_event_identity"] == "NOT_ESTABLISHED"


def test_declared_capacity_literals_are_not_implicitly_expanded():
    assert subject.MAX_RECORDS == 512 and subject.MAX_RAW_ROWS == 1_000_000
    assert subject.MAX_GRAPH_ROW_VISITS == 4_000_000
    assert subject.MAX_GRAPH_LINKS == 4_000_000
    assert subject.MAX_COVER_ROW_CHECKS == 32_000_000
    assert subject.MAX_COMPONENT_MEMBER_COMPARISONS == 262_144
    assert subject.MAX_PLAN_BYTES == 8*1024**2


def test_independent_small_fixture_work_accounting():
    answer = _run(_nested_records())
    # Three supports have5+6+10 rows. Shared arc coordinates induce1+2+1
    # star links. Union/candidate scans each charge8+10+16=34 rows; the sole
    # full candidate checks34 more, giving102; it compares all three members.
    assert answer["status"] == "COMPLETE"
    assert answer["work"] == dict(graph_row_visits=21, graph_links=4,
                                  cover_row_checks=102, member_comparisons=3)
    assert answer["cover_row_checks_semantics"] == "CONSERVATIVE_SET_ROW_WORK_NOT_LITERAL_CPU_OR_ALL_ALLOCATION_WORK"


@pytest.mark.parametrize("constant,exact", [
    ("MAX_RECORDS", 3), ("MAX_RAW_ROWS", 32),
    ("MAX_GRAPH_ROW_VISITS", 21), ("MAX_GRAPH_LINKS", 4),
    ("MAX_COVER_ROW_CHECKS", 102), ("MAX_COMPONENT_MEMBER_COMPARISONS", 3),
])
def test_each_work_capacity_exact_boundary_and_one_over_whole_refusal(monkeypatch, constant, exact):
    monkeypatch.setattr(subject, constant, exact)
    _one_group(_run(_nested_records()), [1, 2, 3], "COVERED", 3)
    monkeypatch.setattr(subject, constant, exact-1)
    failed = _run(_nested_records())
    _fatal(failed)
    if constant in ("MAX_RECORDS", "MAX_RAW_ROWS", "MAX_GRAPH_ROW_VISITS"):
        assert failed["work"] == dict(graph_row_visits=0, graph_links=0,
                                      cover_row_checks=0, member_comparisons=0)
    elif constant == "MAX_GRAPH_LINKS":
        assert failed["work"]["graph_links"] == 4
    elif constant == "MAX_COVER_ROW_CHECKS":
        assert failed["work"]["cover_row_checks"] == 102
    elif constant == "MAX_COMPONENT_MEMBER_COMPARISONS":
        assert failed["work"]["member_comparisons"] == 3


def test_complete_encoded_plan_exact_byte_boundary_and_one_over(monkeypatch):
    baseline = _run(_nested_records())
    assert baseline["status"] == "COMPLETE"
    encoded = json.dumps(baseline, allow_nan=False, separators=(",", ":")).encode()
    monkeypatch.setattr(subject, "MAX_PLAN_BYTES", len(encoded))
    assert _run(_nested_records()) == baseline
    monkeypatch.setattr(subject, "MAX_PLAN_BYTES", len(encoded)-1)
    failed = _run(_nested_records())
    _fatal(failed)
    assert failed["work"] == baseline["work"], "Final publication failure must retain attempted work"


def test_late_second_component_work_failure_clears_already_covered_prefix(monkeypatch):
    records = [_record(1, [0], [1], [2]), _record(2, [3], [4], [5])]
    # First singleton consumes5union+5candidate+5direct-role checks=15.
    # The second union attempts5 more: failure cannot publish singleton1.
    monkeypatch.setattr(subject, "MAX_COVER_ROW_CHECKS", 15)
    answer = _run(records)
    _fatal(answer)
    assert answer["work"] == dict(graph_row_visits=6, graph_links=0,
                                  cover_row_checks=20, member_comparisons=1)


def test_unavailable_component_still_contributes_all_graph_rows():
    points = _points()
    points[4] = points[0]
    answer = _run([_record(1, [0], [4], [5])], points)
    _one_group(answer, [1], "UNAVAILABLE")
    assert answer["work"] == dict(graph_row_visits=3, graph_links=0,
                                  cover_row_checks=0, member_comparisons=0)


def test_late_ordinary_exception_whole_clears_and_preserves_work(monkeypatch):
    original = subject._encoded
    def encoded(value):
        if value == [_id(2)]:
            raise RuntimeError("INVENTED_SECOND_COMPONENT_ERROR")
        return original(value)
    monkeypatch.setattr(subject, "_encoded", encoded)
    answer = _run([_record(1, [0], [1], [2]), _record(2, [3], [4], [5])])
    _fatal(answer)
    assert answer["work"]["cover_row_checks"] == 15
    assert answer["work"]["member_comparisons"] == 1


def test_base_exception_is_not_misreported_as_completed_or_refused_plan(monkeypatch):
    def interrupted(value):
        raise KeyboardInterrupt("INVENTED_INTERRUPT")
    monkeypatch.setattr(subject, "_encoded", interrupted)
    with pytest.raises(KeyboardInterrupt, match="INVENTED_INTERRUPT"):
        _run(_nested_records())
