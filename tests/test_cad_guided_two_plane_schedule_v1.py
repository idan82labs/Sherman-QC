"""Independent invented scheduling/math oracles; no raw geometry or native work.

The root test author did not implement the scheduler. Plane declarations remain
unverified until the engine adapter authenticates them. This suite qualifies
bounded schedule behavior, not pose coverage, full-cap runtime or registration.
No tests execute at import; only two named pure source modules are loaded lazily.
"""
from dataclasses import replace
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kernel(monkeypatch):
    package = types.ModuleType("_independent_pair_schedule_v1")
    package.__path__ = [str(ROOT / "backend")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded = []
    for name in ("cad_guided_plane_search_v6", "cad_guided_two_plane_schedule_v1"):
        spec = importlib.util.spec_from_file_location(package.__name__+"."+name, ROOT/"backend"/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded[-1]


def _planes(kernel, normals, base=10, alpha=0.):
    return tuple(kernel._search.PlaneEquationV6(base+i, tuple(float(x) for x in normal),
        (0., 0., 0.), 0., alpha) for i, normal in enumerate(normals))


def _args(kernel, copies=1):
    normals = [(1., 0., 0.)]*copies+[(0., 1., 0.)]*copies
    return dict(scan_equations=_planes(kernel, normals), cad_equations=_planes(kernel, normals, 100),
        retained_triplet_hypothesis_count=0, retained_triplet_unsigned_count=0)


def _run(kernel, args):
    return kernel.schedule_two_plane_pairs_v1(**args)


def _safety(result):
    assert result["accepted_registration"] is False and result["production_use_allowed"] is False
    assert result["observed_bend_count"] is None and result["search_exhaustive"] is False
    assert result["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    assert "UNVERIFIED" in result["support_provenance"]
    assert not any(key in result for key in ("rotation", "translation_mm", "candidates", "predictions"))
    json.dumps(result, allow_nan=False)


def _complete(result):
    _safety(result)
    assert result["status"] == "COMPLETE" and result["selected_policy_evaluation_complete"] is True
    rows, stats, joint = result["scheduled_matches"], result["statistics"], result["joint_budget"]
    assert stats["scheduled_ordered_matches"] == len(rows)
    assert stats["reserved_two_plane_slots"] == 32*len(rows)
    assert joint["joint_hypothesis_records_reserved"] == joint["retained_triplet_hypothesis_count"]+32*len(rows)
    assert joint["joint_unsigned_assignments_reserved"] == joint["retained_triplet_unsigned_count"]+len(rows)
    assert joint["joint_hypothesis_records_reserved"] <= joint["max_joint_hypothesis_records"]
    assert joint["joint_unsigned_assignments_reserved"] <= joint["max_joint_unsigned_assignments"]
    assert len(rows) <= 256 and len(rows) % 2 == 0
    assert stats["completed_windows"] == len(result["cad_windows"])
    assert stats["compatible_occurrences"] == stats["scheduled_unordered_occurrences"]+stats["omitted_compatible_occurrences"]
    for first, second in zip(rows[::2], rows[1::2]):
        assert first["scan_plane_ids"] == second["scan_plane_ids"]
        assert first["cad_plane_ids"] == second["cad_plane_ids"][::-1]
        assert first["canonical_cad_pair_index"] == second["canonical_cad_pair_index"]
        assert first["scan_pair_id"] == second["scan_pair_id"]
    for index, row in enumerate(rows):
        assert row["ordered_match_index"] == index
        assert row["reserved_hypothesis_records"] == 32 and row["reserved_unsigned_assignments"] == 1
        assert len(set(row["scan_plane_ids"])) == len(set(row["cad_plane_ids"])) == 2
    for row in result["scan_pairs"]:
        assert row["scheduled_ordered_matches"] == sum(x["scan_pair_id"] == row["scan_pair_id"] for x in rows)


def _refused(result):
    _safety(result)
    assert result["status"] == "UNAVAILABLE" and result["selected_policy_evaluation_complete"] is False
    assert result["usable_schedule"] is False and result["reasons"]
    for key in ("scheduled_matches", "scan_pairs", "cad_windows", "scan_planes", "cad_planes"):
        assert result[key] == []
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= 8192


def test_two_independent_planes_schedule_both_orders_and_all_64_future_slots(kernel):
    result = _run(kernel, _args(kernel)); _complete(result)
    assert [(x["scan_plane_ids"], x["cad_plane_ids"]) for x in result["scheduled_matches"]] == [
        ([10, 11], [100, 101]), ([10, 11], [101, 100])]
    assert result["statistics"]["reserved_two_plane_slots"] == 64
    assert result["statistics"]["scan_pair_tests"] == result["statistics"]["cad_pair_tests"] == 1
    assert result["cad_windows"][0]["compatible_unordered_count"] == 1


@pytest.mark.parametrize("theta", (0., .001, .02, .039, .041, .08, .5, math.pi/2))
@pytest.mark.parametrize("alpha", (0., .04))
def test_public_two_row_rank_matches_independent_svd(kernel, theta, alpha):
    first, second = (1., 0., 0.), (math.cos(theta), math.sin(theta), 0.)
    args = _args(kernel)
    args["scan_equations"] = _planes(kernel, (first, second), alpha=alpha)
    args["cad_equations"] = _planes(kernel, (first, second), base=100, alpha=alpha)
    result = _run(kernel, args); _complete(result)
    normals = np.asarray([first, second]); normals /= np.linalg.norm(normals, axis=1)[:, None]
    singular = np.linalg.svd(normals, compute_uv=False)
    ratio = singular[-1]/singular[0]
    margin = singular[-1]-math.sqrt(2)*(2*math.sin(alpha/2))
    row, = result["scan_pairs"]
    assert abs(row["normal_condition_ratio"]-ratio) < 2e-14
    assert abs(row["normal_independence_margin"]-margin) < 2e-14
    admitted = bool(ratio >= .02 and margin > 0.)
    assert bool(result["scheduled_matches"]) is admitted


def test_target_rank_and_cones_do_not_borrow_source_admission(kernel):
    theta = .08
    normals = ((1., 0., 0.), (math.cos(theta), math.sin(theta), 0.))
    args = _args(kernel)
    args["scan_equations"] = _planes(kernel, normals)
    args["cad_equations"] = _planes(kernel, normals, 100, alpha=.04)
    result = _run(kernel, args); _complete(result)
    assert result["scan_pairs"][0]["normal_independence_margin"] > 0
    assert result["cad_index"]["rank_qualified_pair_count"] == 0
    assert result["scheduled_matches"] == []


def test_incompatible_angles_are_not_scheduled(kernel):
    args = _args(kernel)
    args["cad_equations"] = _planes(kernel, ((1., 0., 0.), (.6, .8, 0.)), 100)
    result = _run(kernel, args); _complete(result)
    assert result["scheduled_matches"] == []
    assert result["cad_windows"][0]["outside_index_window_pair_count"] == 1


@pytest.mark.parametrize("h3,u3,expected", ((8192, 0, 0), (8160, 0, 0), (8128, 0, 2),
    (0, 1024, 0), (0, 1023, 0), (0, 1022, 2), (8191, 1024, 0)))
def test_joint_quota_never_splits_both_order_bundle_or_refunds_triplet_records(kernel, h3, u3, expected):
    args = _args(kernel)
    args.update(retained_triplet_hypothesis_count=h3, retained_triplet_unsigned_count=u3)
    result = _run(kernel, args); _complete(result)
    assert len(result["scheduled_matches"]) == expected
    assert result["joint_budget"]["pair_lane_starved_by_joint_quota"] is (expected == 0)
    if expected == 0:
        assert "cad_index" not in result and result["cad_windows"] == []
        assert result["scan_pairs"][0]["status"] == "UNSCHEDULED_TRIPLET_FIRST_JOINT_QUOTA"


def test_repeated_exact_layout_fair_bins_cover_all_positions_without_case_ids(kernel):
    result = _run(kernel, _args(kernel, copies=4)); _complete(result)
    windows = result["cad_windows"]
    assert len(windows) == 16 and all(w["compatible_unordered_count"] == 16 for w in windows)
    assert len({w["layout_index"] for w in windows}) == 1
    assert {p for w in windows[:8] for p in w["scheduled_compatible_positions"]} == set(range(16))
    assert [w["occurrence_index"] for w in windows] == list(range(16))
    for occurrence, window in enumerate(windows):
        assert window["scheduled_compatible_positions"] == [occurrence % 8, 8+occurrence % 8]


def test_one_bin_final_quota_has_independent_repeated_layout_clock(kernel):
    args = _args(kernel, copies=3)
    args["max_joint_hypothesis_records"] = 6*32  # two pairs then one, always both orders
    result = _run(kernel, args); _complete(result)
    first, last = result["cad_windows"]
    assert (first["actual_bin_count"], last["actual_bin_count"]) == (2, 1)
    assert first["layout_index"] == last["layout_index"]
    assert last["layout_occurrence_index"] == 1 and last["occurrence_index"] == 0
    assert last["scheduled_compatible_positions"] == [0]
    assert len(result["scheduled_matches"]) == 6


def test_participation_priority_selects_uncovered_pair_before_reusing_planes(kernel):
    args = _args(kernel, copies=2)
    args["budgets"] = replace(kernel.TwoPlaneScheduleBudgetsV1(), max_selected_scan_pairs=2)
    result = _run(kernel, args); _complete(result)
    pairs = [result["scan_pairs"][i]["scan_plane_ids"] for i in result["selected_scan_pair_ids"]]
    assert len(pairs) == 2 and len(set(pairs[0]+pairs[1])) == 4
    assert any(row["status"] == "UNSCHEDULED_SCAN_PAIR_QUOTA" for row in result["scan_pairs"])


@pytest.mark.parametrize("chunk", (1, 3, 65536))
def test_chunk_size_keeps_complete_window_oracle_and_identical_match_order(kernel, chunk):
    args = _args(kernel, copies=3)
    args["budgets"] = replace(kernel.TwoPlaneScheduleBudgetsV1(), chunk_pair_rows=chunk)
    result = _run(kernel, args); _complete(result)
    pairs = list(itertools.combinations(range(6), 2))
    expected = [i for i, (a, b) in enumerate(pairs) if (a < 3) != (b < 3)]
    for window in result["cad_windows"]:
        assert window["compatible_unordered_count"] == 9
        assert window["examined_pair_count"] == 9
        assert window["scheduled_canonical_pair_indices"] == [expected[i] for i in window["scheduled_compatible_positions"]]
    assert result["statistics"]["window_pair_tests"] == 81


def test_anchors_offsets_and_normal_signs_do_not_select_correspondence(kernel):
    args = _args(kernel, copies=2)
    plain = _run(kernel, args); _complete(plain)
    for name in ("scan_equations", "cad_equations"):
        args[name] = tuple(replace(p, normal=tuple(-x for x in p.normal),
            anchor_mm=(123., -41., 72.), offset_from_anchor_mm=-3.) for p in args[name])
    changed = _run(kernel, args); _complete(changed)
    assert changed["scheduled_matches"] == plain["scheduled_matches"]
    assert changed["input_digests"] != plain["input_digests"]


@pytest.mark.parametrize("role", ("scan_equations", "cad_equations"))
@pytest.mark.parametrize("variant", ("list", "duplicate", "untyped", "nonunit", "nan", "cone", "bool_id"))
def test_invalid_equations_clear_complete_schedule(kernel, role, variant):
    args = _args(kernel); values = args[role]
    if variant == "list": args[role] = list(values)
    elif variant == "duplicate": args[role] = (values[0], values[0])
    elif variant == "untyped": args[role] = (dict(plane_id=1),)
    else:
        fields = {"nonunit": dict(normal=(2., 0., 0.)), "nan": dict(offset_from_anchor_mm=float("nan")),
            "cone": dict(normal_angle_bound_radians=.175), "bool_id": dict(plane_id=True)}[variant]
        args[role] = (replace(values[0], **fields), values[1])
    _refused(_run(kernel, args))


@pytest.mark.parametrize("field,value", (("retained_triplet_hypothesis_count", True),
    ("retained_triplet_hypothesis_count", 8193), ("retained_triplet_unsigned_count", -1),
    ("max_joint_hypothesis_records", 8193), ("max_joint_unsigned_assignments", 1025)))
def test_invalid_joint_counts_refuse_before_pair_math(kernel, field, value):
    args = _args(kernel); args[field] = value
    result = _run(kernel, args); _refused(result)
    assert result["statistics"]["scan_pair_tests"] == 0


@pytest.mark.parametrize("field", ("max_scan_pair_records", "max_scan_selection_tests", "max_cad_pair_records",
    "max_window_pair_tests", "max_compatible_layout_bytes", "max_compatible_layouts", "max_working_bytes", "max_report_bytes"))
def test_hard_work_or_payload_cap_is_whole_refusal(kernel, field):
    args = _args(kernel)
    args["budgets"] = replace(kernel.TwoPlaneScheduleBudgetsV1(), **{field: 0})
    _refused(_run(kernel, args))


def test_sort_work_cap_is_explicit_and_refuses_before_sort(kernel):
    args = _args(kernel, copies=2)
    args["budgets"] = replace(kernel.TwoPlaneScheduleBudgetsV1(), max_cad_sort_work=17)
    result = _run(kernel, args); _refused(result)
    assert result["statistics"]["cad_sort_work"] == 18  # six pairs times ceil(log2 six)
    assert result["statistics"]["completed_windows"] == 0


@pytest.mark.parametrize("which", ("empty_scan", "one_scan", "empty_cad", "one_cad", "zero_pair_quota"))
def test_empty_or_omitted_schedule_is_complete_not_absence_or_pose(kernel, which):
    args = _args(kernel)
    if which == "zero_pair_quota":
        args["budgets"] = replace(kernel.TwoPlaneScheduleBudgetsV1(), max_selected_scan_pairs=0)
    else:
        role = "scan_equations" if "scan" in which else "cad_equations"
        args[role] = args[role][:0 if which.startswith("empty") else 1]
    result = _run(kernel, args); _complete(result)
    assert result["scheduled_matches"] == [] and result["usable_schedule"] is False


def test_late_window_failure_keeps_attempted_counts_but_no_matches(kernel, monkeypatch):
    args = _args(kernel, copies=2); original = kernel._window; calls = []
    def fail_second(*a, **kw):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("private diagnostic should not leak")
        return original(*a, **kw)
    monkeypatch.setattr(kernel, "_window", fail_second)
    result = _run(kernel, args); _refused(result)
    assert result["discarded_scheduled_matches"] == 4 and result["discarded_completed_windows"] == 1
    assert result["statistics"]["scheduled_ordered_matches"] == 4
    assert "private diagnostic" not in json.dumps(result)


def test_final_report_failure_cannot_release_favorable_schedule(kernel, monkeypatch):
    args = _args(kernel); original = kernel._size
    def fail_complete(value, maximum):
        if type(value) is dict and value.get("schema_version") == "two_plane_schedule_v1" and value["status"] == "COMPLETE":
            raise kernel._Unavailable("SCHEDULE_REPORT_BYTE_BUDGET")
        return original(value, maximum)
    monkeypatch.setattr(kernel, "_size", fail_complete)
    result = _run(kernel, args); _refused(result)
    assert result["discarded_scheduled_matches"] == 2
