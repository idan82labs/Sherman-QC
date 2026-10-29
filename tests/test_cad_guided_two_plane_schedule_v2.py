"""Independent invented V2 allocation oracles; no geometry or native execution.

Only three named, pure modules are lazily loaded under an inert package. The
equation API has no support-size/curvature information: angular distractors here
are declarations, not a qualification of broad physical flange recovery. V1 is
an executable prefix oracle only inside a separately supervised future test run.
No test, target import, experiment or source collection runs at module import.
"""
from dataclasses import replace
import importlib.util
import inspect
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
AXES = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


@pytest.fixture
def kernels(monkeypatch):
    name = "_independent_pair_schedule_v2"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "backend")]
    monkeypatch.setitem(sys.modules, name, package)
    loaded = []
    for stem in ("cad_guided_plane_search_v6", "cad_guided_two_plane_schedule_v1",
                 "cad_guided_two_plane_schedule_v2"):
        spec = importlib.util.spec_from_file_location(name+"."+stem, ROOT/"backend"/(stem+".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return types.SimpleNamespace(search=loaded[0], v1=loaded[1], v2=loaded[2])


def _planes(k, normals, base=1009, alpha=0.):
    cones = [alpha]*len(normals) if isinstance(alpha, (int, float)) else alpha
    return tuple(k.search.PlaneEquationV6(base+37*i, tuple(float(x) for x in normal),
        (0., 0., 0.), 0., float(cone)) for i, (normal, cone) in enumerate(zip(normals, cones)))


def _args(k, scan=None, cad=None, alpha=0.):
    # 3*3 + 3*2 + 3*2 = 21 orthogonal unordered target pairs, independently
    # determined from three axis groups. IDs are unrelated to retained fixtures.
    scan = AXES[:2] if scan is None else scan
    cad = (AXES[0],)*3+(AXES[1],)*3+(AXES[2],)*2 if cad is None else cad
    return dict(scan_equations=_planes(k, scan, alpha=alpha),
        cad_equations=_planes(k, cad, base=7001), retained_triplet_hypothesis_count=0,
        retained_triplet_unsigned_count=0)


def _v2(k, args):
    return k.v2.schedule_two_plane_pairs_v2(**args)


def _v1(k, args):
    copied = dict(args)
    if "budgets" in copied:
        values = copied["budgets"]
        copied["budgets"] = k.v1.TwoPlaneScheduleBudgetsV1(**{
            key: getattr(values, key) for key in k.v1.TwoPlaneScheduleBudgetsV1.__dataclass_fields__})
    return k.v1.schedule_two_plane_pairs_v1(**copied)


def _safety(answer):
    assert answer["accepted_registration"] is False
    assert answer["production_use_allowed"] is False
    assert answer["observed_bend_count"] is None
    assert answer["search_exhaustive"] is False
    assert answer["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert "UNVERIFIED" in answer["support_provenance"]
    assert not any(key in answer for key in ("rotation", "translation_mm", "predictions", "candidates"))
    json.dumps(answer, allow_nan=False)


def _complete(answer, previous=None):
    _safety(answer)
    assert answer["schema_version"] == "two_plane_schedule_v2"
    assert answer["status"] == "COMPLETE" and answer["selected_policy_evaluation_complete"] is True
    assert not any(key in answer for key in ("scan_planes", "cad_planes", "scan_pairs", "cad_windows"))
    breadth, rows = answer["breadth_ledger"], answer["scheduled_matches"]
    if previous is not None:
        assert previous["status"] == "COMPLETE"
        # Whole nested report: windows, layout clocks, counts, inputs and byte
        # accounting. This is stronger than comparing only the selected IDs.
        assert breadth == previous
    assert answer["breadth_statistics"] == breadth["statistics"]
    assert rows[:len(breadth["scheduled_matches"])] == breadth["scheduled_matches"]
    assert answer["input_digests"] == breadth["input_digests"]
    assert answer["selected_scan_pair_ids"] == breadth["selected_scan_pair_ids"]
    assert len(rows) <= 256 and len(rows) % 2 == 0
    assert answer["usable_schedule"] is bool(rows)
    seen = set()
    for index, (left, right) in enumerate(zip(rows[::2], rows[1::2])):
        assert left["scan_pair_id"] == right["scan_pair_id"]
        assert left["scan_plane_ids"] == right["scan_plane_ids"]
        assert left["cad_plane_ids"] == right["cad_plane_ids"][::-1]
        assert left["canonical_cad_pair_index"] == right["canonical_cad_pair_index"]
        key = (left["scan_pair_id"], left["canonical_cad_pair_index"])
        assert key not in seen
        seen.add(key)
        for ordinal, row in enumerate((left, right), start=2*index):
            assert row["ordered_match_index"] == ordinal
            assert row["reserved_hypothesis_records"] == 32
            assert row["reserved_unsigned_assignments"] == 1
    stats, joint = answer["statistics"], answer["joint_budget"]
    assert stats["scheduled_ordered_matches"] == len(rows)
    assert stats["reserved_two_plane_slots"] == 32*len(rows)
    assert joint["joint_hypothesis_records_reserved"] == joint["retained_triplet_hypothesis_count"]+32*len(rows)
    assert joint["joint_unsigned_assignments_reserved"] == joint["retained_triplet_unsigned_count"]+len(rows)
    assert joint["joint_hypothesis_records_reserved"] <= joint["max_joint_hypothesis_records"]
    assert joint["joint_unsigned_assignments_reserved"] <= joint["max_joint_unsigned_assignments"]
    assert joint["remaining_ordered_match_capacity"] == joint["ordered_match_capacity_before_both_order_rounding"]-len(rows)
    assert stats["compatible_occurrences"] == stats["scheduled_unordered_occurrences"]+stats["omitted_compatible_occurrences"]
    original = {row["scan_pair_id"]: row for row in breadth["cad_windows"]}
    depth = {row["scan_pair_id"]: row for row in answer["residual_windows"]}
    assert len(depth) == len(answer["residual_windows"])
    for pair_id, row in depth.items():
        before = original[pair_id]
        assert row["complete"] is True and row["both_target_orders_required"] is True
        assert row["compatible_layout_reused_without_geometry"] is True
        assert row["full_cached_index_visits"] == 0
        assert row["layout_index"] == before["layout_index"]
        assert row["compatible_pair_ids_sha256"] == before["compatible_pair_ids_sha256"]
        assert set(row["scheduled_canonical_pair_indices"]).isdisjoint(before["scheduled_canonical_pair_indices"])
        assert set(row["scheduled_compatible_positions"]).isdisjoint(before["scheduled_compatible_positions"])
        count = len(row["scheduled_canonical_pair_indices"])
        assert count == len(set(row["scheduled_canonical_pair_indices"]))
        assert row["reserved_hypothesis_records"] == 64*count
        assert row["reserved_unsigned_assignments"] == 2*count
        assert row["direct_cached_index_lookups"] == count
    for row in answer["final_scan_pair_ledger"]:
        pair_id = row["scan_pair_id"]
        count = sum(match["scan_pair_id"] == pair_id for match in rows)
        assert row["scheduled_ordered_matches"] == count
        assert row["breadth_scheduled_ordered_matches"]+row["residual_scheduled_ordered_matches"] == count
        if pair_id in original:
            assert row["compatible_window_was_evaluated"] is True
            assert row["final_omitted_compatible_unordered_count"] == original[pair_id]["compatible_unordered_count"]-count//2
        else:
            assert row["compatible_window_was_evaluated"] is False
            assert row["compatible_unordered_count"] is None
            assert row["final_omitted_compatible_unordered_count"] is None
    assert {row["scan_pair_id"] for row in answer["final_scan_pair_ledger"]} == set(breadth["selected_scan_pair_ids"])
    assert answer["residual_statistics"]["residual_cad_index_builds"] == 0
    assert answer["residual_statistics"]["residual_geometry_window_tests"] == 0


def _refused(answer):
    _safety(answer)
    assert answer["status"] == "UNAVAILABLE"
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["usable_schedule"] is False
    assert answer["breadth_ledger"] is None
    for key in ("scheduled_matches", "residual_windows", "final_scan_pair_ledger",
                "scan_pairs", "cad_windows", "scan_planes", "cad_planes", "selected_scan_pair_ids"):
        assert answer.get(key, []) == []
    assert answer["reasons"]
    assert len(json.dumps(answer, separators=(",", ":")).encode()) <= 16384


@pytest.mark.parametrize("h3,u3", ((8192, 0), (8160, 0), (8096, 0), (8064, 0),
    (0, 1024), (0, 1023), (0, 1022), (0, 0)))
def test_exact_complete_v1_breadth_survives_zero_odd_exhausted_and_ample_capacity(kernels, h3, u3):
    args = _args(kernels, scan=(AXES[0],)*2+(AXES[1],)*2)
    args.update(retained_triplet_hypothesis_count=h3, retained_triplet_unsigned_count=u3)
    old = _v1(kernels, args)
    answer = _v2(kernels, args); _complete(answer, old)
    if old["joint_budget"]["remaining_ordered_match_capacity"] < 2:
        assert answer["scheduled_matches"] == old["scheduled_matches"]


def test_all_21_target_pairs_reached_without_second_geometry_or_false_pose(kernels):
    args = _args(kernels); answer = _v2(kernels, args)
    _complete(answer, _v1(kernels, args))
    cad = args["cad_equations"]
    oracle = {index for index, (a, b) in enumerate(itertools.combinations(range(8), 2))
              if cad[a].normal != cad[b].normal}
    assert len(oracle) == 21
    assert {row["canonical_cad_pair_index"] for row in answer["scheduled_matches"]} == oracle
    assert len(answer["breadth_ledger"]["scheduled_matches"]) == 4
    assert len(answer["scheduled_matches"]) == 42
    assert answer["joint_budget"]["reserved_two_plane_hypothesis_records"] == 1344
    assert answer["statistics"]["omitted_compatible_occurrences"] == 0
    # Literal independently counted operations: one window, two breadth ranks,
    # nineteen remaining indices, two mapping plus two membership comparisons
    # conservatively charged per selected rank.
    expected = dict(residual_priority_work=0, residual_pair_visits=1,
        residual_breadth_position_tests=2, residual_rank_selections=19,
        residual_skip_position_tests=76, residual_index_lookups=19,
        residual_cache_reference_records=1, residual_cache_reference_bytes=512,
        residual_retained_bytes=3776, residual_report_records=42,
        residual_report_reserved_bytes=46272, final_ledger_tests=43)
    for name, count in expected.items():
        assert answer["residual_statistics"][name] == count


@pytest.mark.parametrize("maximum_ordered", (4, 6, 8, 20, 40))
def test_partial_last_window_uses_only_unselected_ranks_and_reconciles_omissions(kernels, maximum_ordered):
    args = _args(kernels); args["max_joint_hypothesis_records"] = maximum_ordered*32
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    old = answer["breadth_ledger"]["cad_windows"][0]
    depth, = answer["residual_windows"]
    remaining = [i for i in range(21) if i not in old["scheduled_compatible_positions"]]
    take = maximum_ordered//2-2
    # Direct finite list partition, independent of the optimized rank-skip code.
    expected = [] if not take else [remaining[j*len(remaining)//take] for j in range(take)]
    assert depth["scheduled_compatible_positions"] == expected
    assert len(answer["scheduled_matches"]) == maximum_ordered
    assert depth["final_omitted_compatible_unordered_count"] == 21-maximum_ordered//2


def test_repeated_layout_does_not_confuse_source_pair_specific_depth(kernels):
    args = _args(kernels, scan=(AXES[0],)*2+(AXES[1],)*2)
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    windows = answer["breadth_ledger"]["cad_windows"]
    assert len(windows) == 4 and len({x["layout_index"] for x in windows}) == 1
    assert [x["occurrence_index"] for x in windows] == list(range(4))
    assert len(answer["scheduled_matches"]) == 4*42
    assert all(row["final_omitted_compatible_unordered_count"] == 0 for row in answer["final_scan_pair_ledger"])
    assert answer["residual_statistics"]["residual_cache_reference_records"] == 1


def test_full_joint_cap_keeps_all_original_breadth_and_exact_unscheduled_denominator(kernels):
    args = _args(kernels, scan=(AXES[0],)*3+(AXES[1],)*3)
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    assert len(answer["breadth_ledger"]["cad_windows"]) == 9
    assert len(answer["breadth_ledger"]["scheduled_matches"]) == 36
    assert len(answer["scheduled_matches"]) == 256
    assert answer["joint_budget"]["joint_hypothesis_records_reserved"] == 8192
    assert answer["joint_budget"]["remaining_ordered_match_capacity"] == 0
    assert answer["statistics"]["compatible_occurrences"] == 9*21
    assert answer["statistics"]["omitted_compatible_occurrences"] == 61
    # Five complete omitted windows (19 each), one partial (15), three unfilled.
    assert [row["residual_scheduled_unordered_count"] for row in answer["residual_windows"]] == [19]*5+[15, 0, 0, 0]


@pytest.mark.parametrize("which", ("empty_scan", "one_scan", "empty_cad", "one_cad", "zero_pair_quota"))
def test_empty_or_unscheduled_is_complete_but_never_absence_or_pose(kernels, which):
    args = _args(kernels)
    if which == "zero_pair_quota":
        args["budgets"] = replace(kernels.v2.TwoPlaneScheduleBudgetsV2(), max_selected_scan_pairs=0)
    else:
        role = "scan_equations" if "scan" in which else "cad_equations"
        args[role] = args[role][:0 if which.startswith("empty") else 1]
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    assert answer["scheduled_matches"] == []


def test_incompatible_angle_window_is_not_filled_from_a_different_layout(kernels):
    args = _args(kernels, scan=(AXES[0], (.5, math.sqrt(.75), 0.)))
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    window, = answer["breadth_ledger"]["cad_windows"]
    assert window["compatible_unordered_count"] == 0
    assert answer["scheduled_matches"] == []
    assert answer["residual_statistics"]["residual_index_lookups"] == 0


def test_residual_priority_uses_declared_margin_not_only_nominal_orthogonality(kernels):
    args = _args(kernels, scan=AXES, alpha=(.01, .12, .02))
    args["max_joint_hypothesis_records"] = 14*32  # twelve breadth orders + one bundle
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    added = [row for row in answer["residual_windows"] if row["residual_scheduled_unordered_count"]]
    assert len(added) == 1
    source_ids = [args["scan_equations"][i].plane_id for i in (0, 2)]
    chosen = answer["breadth_ledger"]["scan_pairs"][added[0]["scan_pair_id"]]
    assert chosen["scan_plane_ids"] == source_ids
    oracle = 1.-math.hypot(2*math.sin(.01/2), 2*math.sin(.02/2))
    assert abs(added[0]["normal_independence_margin"]-oracle) < 2e-14
    assert answer["residual_statistics"]["residual_priority_work"] == 6


def test_confident_duplicate_declarations_can_still_starve_a_later_useful_pair(kernels):
    # The last pair is designated useful by this toy observer ONLY; that label
    # is not passed to the scheduler. Equal confidence cannot reveal usefulness.
    args = _args(kernels, scan=(AXES[0],)*2+(AXES[1],)*2)
    args["max_joint_hypothesis_records"] = 18*32
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    order = answer["breadth_ledger"]["selected_scan_pair_ids"]
    used = [r["scan_pair_id"] for r in answer["residual_windows"] if r["residual_scheduled_unordered_count"]]
    assert used == order[:1]
    last = next(r for r in answer["final_scan_pair_ledger"] if r["scan_pair_id"] == order[-1])
    assert last["residual_scheduled_ordered_matches"] == 0
    assert last["final_omitted_compatible_unordered_count"] == 19
    assert "NOT_GEOMETRIC_DISPROOF" in last["omission_scope"]


@pytest.mark.parametrize("theta", (.015, .08, math.pi/2))
@pytest.mark.parametrize("alpha", (0., .04))
def test_two_row_margin_matches_independent_svd_not_a_pose_certificate(kernels, theta, alpha):
    normals = (AXES[0], (math.cos(theta), math.sin(theta), 0.))
    args = _args(kernels, scan=normals, cad=normals, alpha=alpha)
    answer = _v2(kernels, args); _complete(answer, _v1(kernels, args))
    n = np.asarray(normals); n /= np.linalg.norm(n, axis=1)[:, None]
    singular = np.linalg.svd(n, compute_uv=False)
    row, = answer["breadth_ledger"]["scan_pairs"]
    margin = singular[-1]-math.sqrt(2)*2*math.sin(alpha/2)
    assert abs(row["normal_independence_margin"]-margin) < 2e-14
    assert abs(row["normal_condition_ratio"]-singular[-1]/singular[0]) < 2e-14
    if margin <= 0. or singular[-1]/singular[0] < .02:
        assert not answer["scheduled_matches"]


@pytest.mark.parametrize("mode", ("signs", "cyclic_proper", "half_turn_proper"))
def test_exact_axis_sign_and_proper_rotation_preserve_this_tied_constructed_schedule(kernels, mode):
    args = _args(kernels, scan=AXES); before = _v2(kernels, args); _complete(before)
    if mode == "signs":
        transform = lambda n: tuple(-x for x in n)
    elif mode == "cyclic_proper":
        transform = lambda n: (n[2], n[0], n[1])
    else:
        transform = lambda n: (-n[0], -n[1], n[2])
    changed = dict(args)
    for role in ("scan_equations", "cad_equations"):
        changed[role] = tuple(replace(p, normal=transform(p.normal),
            anchor_mm=(31., -19., 7.), offset_from_anchor_mm=2.) for p in args[role])
    after = _v2(kernels, changed); _complete(after)
    assert after["scheduled_matches"] == before["scheduled_matches"]
    assert after["input_digests"] != before["input_digests"]


def test_cad_order_permutation_keeps_exhaustive_constructed_pair_set_not_universal_order(kernels):
    args = _args(kernels); before = _v2(kernels, args); _complete(before)
    permuted = dict(args, cad_equations=tuple(reversed(args["cad_equations"])))
    after = _v2(kernels, permuted); _complete(after)
    projection = lambda r: {(tuple(x["scan_plane_ids"]), tuple(x["cad_plane_ids"])) for x in r["scheduled_matches"]}
    assert projection(before) == projection(after)
    assert "NOT_RENUMBERING_OR_ORDER_INVARIANT" in after["identity_ties"]


@pytest.mark.parametrize("factor,admitted", ((1.+5e-9, True), (1.+2e-8, False)))
def test_near_unit_admission_has_declared_boundary_not_silent_arbitrary_rescaling(kernels, factor, admitted):
    args = _args(kernels)
    args["scan_equations"] = tuple(replace(p, normal=tuple(factor*x for x in p.normal)) for p in args["scan_equations"])
    answer = _v2(kernels, args)
    if admitted:
        _complete(answer, _v1(kernels, args))
        assert len(answer["scheduled_matches"]) == 42
    else:
        _refused(answer)


@pytest.mark.parametrize("role", ("scan_equations", "cad_equations"))
@pytest.mark.parametrize("variant", ("list", "duplicate_id", "untyped", "bool_id", "nan", "cone"))
def test_malformed_declared_inputs_whole_refuse(kernels, role, variant):
    args = _args(kernels); values = args[role]
    if variant == "list": args[role] = list(values)
    elif variant == "duplicate_id": args[role] = (values[0], values[0])
    elif variant == "untyped": args[role] = ({"normal": AXES[0]},)
    else:
        fields = {"bool_id": dict(plane_id=True), "nan": dict(offset_from_anchor_mm=float("nan")),
                  "cone": dict(normal_angle_bound_radians=.175)}[variant]
        args[role] = (replace(values[0], **fields),)+values[1:]
    _refused(_v2(kernels, args))


@pytest.mark.parametrize("variant", ("v1_type", "dictionary", "bool_new_cap", "over_new_cap", "zero_chunk"))
def test_exact_new_budget_type_and_hard_maxima(kernels, variant):
    args = _args(kernels)
    if variant == "v1_type": budget = kernels.v1.TwoPlaneScheduleBudgetsV1()
    elif variant == "dictionary": budget = {}
    else:
        fields = {"bool_new_cap": dict(max_residual_pair_visits=True),
                  "over_new_cap": dict(max_residual_index_lookups=129),
                  "zero_chunk": dict(chunk_pair_rows=0)}[variant]
        budget = replace(kernels.v2.TwoPlaneScheduleBudgetsV2(), **fields)
    args["budgets"] = budget
    answer = _v2(kernels, args); _refused(answer)
    assert answer["breadth_statistics"] == {}


@pytest.mark.parametrize("field,value", (("retained_triplet_hypothesis_count", True),
    ("retained_triplet_hypothesis_count", -1), ("max_joint_hypothesis_records", 8193),
    ("retained_triplet_unsigned_count", 1025)))
def test_invalid_joint_declarations_refuse_before_pair_arithmetic(kernels, field, value):
    args = _args(kernels); args[field] = value
    answer = _v2(kernels, args); _refused(answer)
    assert answer["breadth_statistics"]["scan_pair_tests"] == 0


@pytest.mark.parametrize("field,needed", (
    ("max_residual_pair_visits", 1), ("max_residual_breadth_position_tests", 2),
    ("max_residual_rank_selections", 19), ("max_residual_skip_position_tests", 76),
    ("max_residual_index_lookups", 19), ("max_residual_cache_reference_records", 1),
    ("max_residual_cache_reference_bytes", 512), ("max_residual_retained_bytes", 3776),
    ("max_residual_report_records", 42), ("max_residual_report_reserved_bytes", 46272),
    ("max_final_ledger_tests", 43)))
def test_literal_exact_new_cap_passes_one_less_whole_clears(kernels, field, needed):
    args = _args(kernels)
    args["budgets"] = replace(kernels.v2.TwoPlaneScheduleBudgetsV2(), **{field: needed})
    _complete(_v2(kernels, args))
    args["budgets"] = replace(args["budgets"], **{field: needed-1})
    refused = _v2(kernels, args); _refused(refused)
    assert refused["discarded_breadth_scheduled_matches"] == 4
    assert refused["residual_statistics"][field.removeprefix("max_")] == needed


def test_priority_sort_work_exact_cap_and_one_less(kernels):
    args = _args(kernels, scan=AXES)
    args["budgets"] = replace(kernels.v2.TwoPlaneScheduleBudgetsV2(), max_residual_priority_work=6)
    _complete(_v2(kernels, args))
    args["budgets"] = replace(args["budgets"], max_residual_priority_work=5)
    refused = _v2(kernels, args); _refused(refused)
    assert refused["residual_statistics"]["residual_priority_work"] == 6


def test_only_one_cad_build_and_original_complete_windows_are_called(kernels, monkeypatch):
    calls = dict(index=0, window=0); build, window = kernels.v1._cad_index, kernels.v1._window
    def counted_build(*args, **kwargs):
        calls["index"] += 1
        return build(*args, **kwargs)
    def counted_window(*args, **kwargs):
        calls["window"] += 1
        return window(*args, **kwargs)
    monkeypatch.setattr(kernels.v1, "_cad_index", counted_build)
    monkeypatch.setattr(kernels.v1, "_window", counted_window)
    answer = _v2(kernels, _args(kernels, scan=(AXES[0],)*2+(AXES[1],)*2)); _complete(answer)
    assert calls == dict(index=1, window=4)
    assert answer["residual_statistics"]["residual_index_lookups"] == 4*19


def test_budget_snapshot_not_live_caller_instance_controls_depth(kernels, monkeypatch):
    budget = kernels.v2.TwoPlaneScheduleBudgetsV2()
    args = dict(_args(kernels), budgets=budget)
    original = kernels.v1._cad_index
    def mutate_after_admission(*positional, **keywords):
        object.__setattr__(budget, "max_residual_index_lookups", 0)
        return original(*positional, **keywords)
    monkeypatch.setattr(kernels.v1, "_cad_index", mutate_after_admission)
    answer = _v2(kernels, args); _complete(answer)
    assert budget.max_residual_index_lookups == 0
    assert answer["budgets"]["max_residual_index_lookups"] == 128
    assert len(answer["scheduled_matches"]) == 42


def test_distinct_exact_layouts_with_equal_digests_keep_separate_byte_identity(kernels):
    # Deliberately mocked cache identity, not a claim to find a SHA collision.
    class Work:
        def __init__(self): self.charges = []
        def charge(self, *args): self.charges.append(args)
        def memory(self): pass
        def retain(self, size): self.charges.append(("retained", size))
    digest = "a"*64
    keys = (struct.pack("<4i", 2, 5, 8, 13), struct.pack("<4i", 3, 7, 11, 17))
    layouts = {key: dict(layout_index=index, sha256=digest) for index, key in enumerate(keys)}
    work = Work(); refs = kernels.v2._cache_references(layouts, work)
    window = dict(scheduled_compatible_positions=[0, 2], compatible_unordered_count=4,
                  compatible_pair_ids_sha256=digest)
    assert refs[0][0] is keys[0] and refs[1][0] is keys[1]
    assert kernels.v2._residual_positions(window, refs[0], 4, work)[2] == [5, 13]
    assert kernels.v2._residual_positions(window, refs[1], 4, work)[2] == [7, 17]


def test_late_residual_exception_discards_successful_breadth_and_depth_prefix(kernels, monkeypatch):
    original = kernels.v2._residual_positions; calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("private late diagnostic")
        return original(*args, **kwargs)
    monkeypatch.setattr(kernels.v2, "_residual_positions", fail_second)
    args = _args(kernels, scan=(AXES[0],)*2+(AXES[1],)*2)
    answer = _v2(kernels, args); _refused(answer)
    assert answer["discarded_breadth_scheduled_matches"] == 16
    assert answer["discarded_residual_completed_windows"] == 1
    assert answer["discarded_scheduled_matches"] == 54
    assert answer["residual_statistics"]["residual_scheduled_ordered_matches"] == 38
    assert "private late" not in json.dumps(answer)


def test_late_original_breadth_exception_never_starts_residual_lane(kernels, monkeypatch):
    original = kernels.v1._window; calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("private breadth failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(kernels.v1, "_window", fail_second)
    args = _args(kernels, scan=(AXES[0],)*2+(AXES[1],)*2)
    answer = _v2(kernels, args); _refused(answer)
    assert answer["discarded_breadth_scheduled_matches"] == 4
    assert answer["breadth_statistics"]["completed_windows"] == 1
    assert answer["residual_statistics"] == {}
    assert "private breadth" not in json.dumps(answer)


def test_working_memory_failure_after_complete_breadth_cannot_return_prefix(kernels, monkeypatch):
    original = kernels.v2._ResidualWork.memory
    def fail_after_depth_allocation(self, extra=0):
        if self.stats["residual_retained_bytes"] >= 3000:
            raise kernels.v1._Unavailable("SCHEDULE_V2_WORKING_PAYLOAD_BUDGET")
        return original(self, extra)
    monkeypatch.setattr(kernels.v2._ResidualWork, "memory", fail_after_depth_allocation)
    answer = _v2(kernels, _args(kernels)); _refused(answer)
    assert answer["discarded_breadth_scheduled_matches"] == 4
    assert answer["residual_statistics"]["residual_retained_bytes"] == 3520


def test_initial_residual_scratch_refusal_retains_attempted_peak_not_empty_statistics(kernels):
    args = _args(kernels)
    args["budgets"] = replace(kernels.v2.TwoPlaneScheduleBudgetsV2(), max_working_bytes=131072)
    previous = _v1(kernels, args)
    assert previous["status"] == "COMPLETE" and len(previous["scheduled_matches"]) == 4
    answer = _v2(kernels, args); _refused(answer)
    # One-pair V1 retained payload: 65,536 fixed +16,384 reports +924 numeric
    # arrays +84 exact-layout bytes. V2's initial fixed scratch adds65,536.
    assert answer["discarded_breadth_scheduled_matches"] == 4
    assert answer["residual_statistics"]["working_peak_bytes"] == 148464
    assert answer["residual_statistics"]["residual_fixed_scratch_bytes"] == 65536
    assert answer["residual_statistics"]["residual_report_records"] == 0
    assert answer["residual_statistics"]["residual_index_lookups"] == 0


def test_final_report_overflow_clears_even_fully_scheduled_depth(kernels, monkeypatch):
    original = kernels.v1._size
    def fail_final(value, maximum):
        if type(value) is dict and value.get("schema_version") == "two_plane_schedule_v2" and value["status"] == "COMPLETE":
            raise kernels.v1._Unavailable("SCHEDULE_REPORT_BYTE_BUDGET")
        return original(value, maximum)
    monkeypatch.setattr(kernels.v1, "_size", fail_final)
    answer = _v2(kernels, _args(kernels)); _refused(answer)
    assert answer["discarded_scheduled_matches"] == 42
    assert answer["discarded_residual_completed_windows"] == 1


def test_bad_cached_layout_length_is_fatal_not_a_partial_schedule(kernels, monkeypatch):
    original = kernels.v2._cache_references
    def corrupt(*args, **kwargs):
        values = original(*args, **kwargs)
        return [(key[:-4], digest) for key, digest in values]
    monkeypatch.setattr(kernels.v2, "_cache_references", corrupt)
    answer = _v2(kernels, _args(kernels)); _refused(answer)
    assert answer["discarded_breadth_scheduled_matches"] == 4
    assert answer["residual_statistics"]["residual_index_lookups"] == 0


def test_no_geometric_outcome_input_or_refund_and_baseexception_not_relabelled(kernels, monkeypatch):
    assert tuple(inspect.signature(kernels.v2.schedule_two_plane_pairs_v2).parameters) == (
        "scan_equations", "cad_equations", "retained_triplet_hypothesis_count",
        "retained_triplet_unsigned_count", "max_joint_hypothesis_records",
        "max_joint_unsigned_assignments", "budgets")
    def interrupt(*args, **kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(kernels.v2, "_residual_positions", interrupt)
    with pytest.raises(KeyboardInterrupt): _v2(kernels, _args(kernels))


def test_toy_48_group_selection_is_not_monotone_under_added_passing_proposals(kernels):
    # The same already-loaded pure strata function is used by initialization.
    # This is NOT an adapter, registration or physical recovery experiment.
    old = ["old-"+str(i) for i in range(48)]
    extended = old+["new-passing-group"]
    before = {old[i] for i in kernels.search._strata(len(old), 48)}
    after = {extended[i] for i in kernels.search._strata(len(extended), 48)}
    assert len(before) == len(after) == 48
    assert "new-passing-group" in after
    assert before-after and not before.issubset(after)
