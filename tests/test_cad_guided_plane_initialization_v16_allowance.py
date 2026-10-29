"""Initializer V16 allowance, receipts, checkpoints, reuse and the explicit finite-subset publication
contract on the REAL two-sheet registration fixture and the completion fixture. Counted synthetic costs
and fake clocks only; no 1860 s burn, no physical data, no native CAD."""
import copy
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UNIT = "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"


@pytest.fixture(scope="module")
def fixture():
    spec = importlib.util.spec_from_file_location("_integration_fixture_v4_for_v16",
        ROOT/"tests/test_cad_guided_context_v4_engine_integration_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m():
    prefix = "backend"
    saved = {name: value for name, value in sys.modules.items() if name == prefix or name.startswith(prefix+".")}
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType(prefix); package.__path__ = [str(ROOT/"backend")]; sys.modules[prefix] = package
    try:
        yield types.SimpleNamespace(init=importlib.import_module("backend.cad_guided_plane_initialization_v16"),
                                    v5=importlib.import_module("backend.cad_guided_global_registration_v5"),
                                    context2=importlib.import_module("backend.cad_guided_context_measurement_v2"),
                                    context3=importlib.import_module("backend.cad_guided_context_measurement_v3"),
                                    context4=importlib.import_module("backend.cad_guided_two_plane_context_v4"),
                                    workflow17=importlib.import_module("backend.cad_guided_mesh_offline_v17"))
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]
        sys.modules.update(saved)


def _fake_clock(step=0., start=0.):
    state = dict(cpu=start, wall=start)
    def clock():
        state["cpu"] += step; state["wall"] += step
        return state["cpu"], state["wall"]
    clock.state = state
    return clock


def _register(m, fixture, **overrides):
    arrays, nominal = fixture._arrays()
    observed, scan_triangles, cad, faces, local = arrays
    cad_points = np.unique(cad.reshape(-1, 3), axis=0)
    policy = fixture._policy(types.SimpleNamespace(init=m.init))
    options = dict(position_error_bounds_mm=.025, cad_initialization_points=cad_points, registration_policy=policy,
                   unit_assumption=UNIT, search_budgets=m.init._search.PlaneSearchBudgetsV6(max_bank_planes=2),
                   two_plane_context_mode="v4")
    options.update(overrides)
    answer, roles = m.init.register_measurement_contexts_v16(observed, scan_triangles, cad, faces, local, **options)
    return answer, roles, (observed, nominal)


def test_active_allowance_reproduces_the_v15_supported_context_with_full_ledger_receipts_reuse_and_checkpoints(m, fixture):
    seen = []
    answer, roles, _ = _register(m, fixture, refinement_allowance=m.v5.RefinementAllowanceV1(),
                                 stage_checkpoint=lambda record: seen.append(record))
    assert answer["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS", answer["reasons"]
    evidence = answer["evidence"]
    assert evidence["registration_implementation"] == m.v5.REGISTRATION_IMPLEMENTATION
    assert evidence["initialization_implementation"] == m.init.INITIALIZATION_IMPLEMENTATION
    assert evidence["candidate_evaluation_complete"] is True and evidence["evaluated_subset_complete"] is True
    ledger = evidence["candidate_disposition_ledger"]
    assert ledger["schema"] == "candidate_disposition_ledger_v1" and ledger["selected_count"] == evidence["candidate_count"]
    assert ledger["completed"] == [row["index"] for row in evidence["candidates"]] and not (ledger["expired"] or ledger["omitted"] or ledger["failed"] or ledger["not_attempted"])
    assert evidence["candidate_omission_status"] == "NONE" and evidence["publication_contract"] == m.init.PUBLICATION_CONTRACT
    assert evidence["candidate_set_scope"] == "WHOLE_DECLARED_FINITE_CANDIDATE_SET" and evidence["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    for row in evidence["candidates"]:
        assert row["execution_status"] == "ATTEMPTED" and row["allowance_lane"] == "main" and row["cost"]["cpu_seconds"] >= 0.
        assert row["conditional_context_support"]["query_reuse"] == "REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING"
    resources = evidence["query_resources"]
    assert resources["reused_query_calls"] == len(evidence["candidates"]) and resources["reused_query_rows"] > 0
    assert evidence["cooperative_chunk_rows"] == 32768 and resources["chunked_query_calls"] == 0    # fixture below one chunk
    receipts = evidence["cost_receipts"]
    assert receipts["allowance"]["main_refined_groups"] == 12 and receipts["cpu_guard_tripped"] is False
    assert [r["label"] for r in receipts["receipts"] if r["lane"] == "main"] == ["REFINE_CANDIDATE_%d" % row["index"] for row in evidence["candidates"]]
    assert evidence["stage_checkpoints"]["count"] == len(seen) > 0 and evidence["stage_checkpoints"]["truncated"] is False
    stages = [record["stage"] for record in seen]
    assert stages[:1] == ["COMPLETION_PREFILTER_BEGIN"] and "REFINEMENT_BEGIN" in stages and stages[-1] == "CONTEXT_PUBLICATION_END"
    assert all(record["process_cpu_seconds"] >= 0. for record in seen)
    assert evidence["initialization"]["stage_checkpoints"]["count"] >= 2 and evidence["initialization"]["allowance_implementation"] == m.init.ALLOWANCE_IMPLEMENTATION
    assert roles is not None


def test_no_allowance_keeps_v15_semantics_without_receipts(m, fixture):
    answer, _, _ = _register(m, fixture)
    evidence = answer["evidence"]
    assert answer["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS" and evidence["refinement_allowance"] is None
    assert evidence["cooperative_chunk_rows"] is None and evidence["candidate_evaluation_complete"] is True
    assert evidence["cost_receipts"]["allowance"] is None and evidence["query_resources"]["reused_query_calls"] >= 1


def test_tiny_main_row_quota_expires_the_first_candidate_deterministically_and_omits_the_rest(m, fixture):
    allowance = m.v5.RefinementAllowanceV1(main_query_rows=60)
    answer, roles, _ = _register(m, fixture, refinement_allowance=allowance)
    evidence = answer["evidence"]
    assert answer["status"] == "UNAVAILABLE" and answer["reasons"] == ["NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS"]
    rows = evidence["candidates"]
    assert rows[0]["execution_status"] == "EXPIRED" and rows[0]["reasons"] == ["EXPIRED_BY_LANE_WORK_QUOTA"]
    assert rows[0]["allowance_reason"] == "LANE_WORK_QUOTA:main" and "conditional_context_support" not in rows[0]
    assert all(row["execution_status"] == "OMITTED" and row["reasons"] == ["OMITTED_BY_LANE_WORK_QUOTA"] for row in rows[1:])
    ledger = evidence["candidate_disposition_ledger"]
    assert ledger["expired"] == [rows[0]["index"]] and ledger["omitted"] == [row["index"] for row in rows[1:]] and ledger["completed"] == []
    assert evidence["candidate_evaluation_complete"] is False and evidence["evaluated_subset_complete"] is True
    assert evidence["candidate_omission_status"] == "ALLOWANCE_OMISSIONS_RETAINED_UNKNOWN"
    assert evidence["measurement_context_indices"] == [] and roles is None
    again, _, _ = _register(m, fixture, refinement_allowance=allowance)
    assert again["evidence"]["candidate_disposition_ledger"] == ledger          # deterministic: same inputs, same trace


def test_main_group_quota_bounds_the_refinement_schedule_deterministically(m, fixture):
    answer, _, _ = _register(m, fixture, refinement_allowance=m.v5.RefinementAllowanceV1(main_refined_groups=1))
    init = answer["evidence"]["initialization"]
    schedule = init["completion_lane"]["refinement_schedule"]
    assert schedule["main_quota"] == 1 and schedule["main_scheduled"] == 1 and len(init["start_groups"]) <= 1+schedule["completion_scheduled"]
    assert schedule["rule"].startswith("MAIN_LANE_STRATA_FIRST_WITHIN_DECLARED_LANE_QUOTAS")
    omitted = [g for g in init["all_passing_start_groups"] if g["refinement_disposition"] == "NOT_SCHEDULED_BY_DECLARED_GROUP_QUOTA"]
    assert len(omitted) == len(init["all_passing_start_groups"])-len(init["start_groups"])


def test_checkpoint_ledger_is_bounded_and_declares_truncation(m, fixture, monkeypatch):
    monkeypatch.setattr(m.init, "MAX_CHECKPOINT_BYTES", 400)
    seen = []
    answer, _, _ = _register(m, fixture, refinement_allowance=m.v5.RefinementAllowanceV1(),
                             stage_checkpoint=lambda record: seen.append(record))
    ledger = answer["evidence"]["stage_checkpoints"]
    assert ledger["truncated"] is True and ledger["bytes"] <= 400 and ledger["count"] == len(seen) >= 1
    assert ledger["retention"] == "EMITTED_TO_CALLER_CALLBACK_AND_RESULT_NOT_A_DURABILITY_ATTESTATION"


def test_invalid_allowance_clock_or_checkpoint_types_refuse_before_any_work(m, fixture):
    for bad in (dict(refinement_allowance=dict(enabled=True)), dict(clock="now"), dict(stage_checkpoint=5)):
        answer, roles, _ = _register(m, fixture, **bad)
        assert answer["status"] == "UNAVAILABLE" and roles is None
        assert answer["reasons"] and answer["reasons"][0] in ("INVALID_REFINEMENT_ALLOWANCE", "INVALID_ALLOWANCE_CLOCK", "INVALID_STAGE_CHECKPOINT")


def _completion_scene(m):
    spec = importlib.util.spec_from_file_location("_completion_fixture_v16", ROOT/"tests/test_cad_guided_plane_initialization_v16_completion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completion_lane_skips_without_allocation_and_rolls_back_only_on_typed_expiry(m):
    scene = _completion_scene(m)
    arrays, nominal, pose = scene._arrays()
    observed, scan_triangles, cad, faces, local = arrays
    keep = faces != 37
    small = scene._grid((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 2.5, 4.), (2., 3.5, 5.), True)
    cad = np.concatenate((cad[keep], small[0][small[1]]))
    faces = np.concatenate((faces[keep], np.full(len(small[1]), 37, dtype=np.int64)))
    local = np.concatenate((local[keep], np.arange(len(small[1]), dtype=np.int64)))
    configuration = scene._configuration(m.init)
    # 1. Allocation zero: the lane is skipped before any of its prefilter work.
    m.init._attach_allowance(configuration, m.v5.RefinementAllowanceV1(completion_refined_groups=0), None, None)
    result, context = m.init._initialize(observed, scan_triangles, cad, faces, local, .025, configuration, UNIT)
    assert result["status"] == "COMPLETE", result["reasons"]
    lane = result["completion_lane"]
    assert lane["status"] == "SKIPPED_NO_REFINEMENT_ALLOCATION" and lane["allocation_remaining"] == 0 and lane["hypotheses"] == []
    assert result["statistics"]["completion_passing_start_group_count"] == 0 and result["statistics"]["main_passing_start_group_count"] >= 1
    # 2. Typed expiry inside the lane (completion row allocation below the first child's 125 attempted rows,
    #    above its first 25-row screen): the in-flight child is EXPIRED, later children OMITTED, rollback of
    #    completion groups only. A 5-row allocation expires BEFORE any face index is built (all OMITTED).
    configuration = scene._configuration(m.init)
    m.init._attach_allowance(configuration, m.v5.RefinementAllowanceV1(completion_query_rows=100), None, None)
    result, context = m.init._initialize(observed, scan_triangles, cad, faces, local, .025, configuration, UNIT)
    assert result["status"] == "COMPLETE", result["reasons"]
    lane = result["completion_lane"]
    assert lane["status"] == "EXPIRED_BY_ALLOWANCE:LANE_WORK_QUOTA" and lane["usable"] is False
    assert lane["transaction"]["status"] == "ROLLED_BACK_BY_TYPED_ALLOWANCE_EXPIRY" and lane["transaction"]["main_groups_retained"] == result["statistics"]["main_passing_start_group_count"]
    assert lane["transaction"]["expired_phase"] == "SUPPORT_EVALUATION" and lane["transaction"]["restored_duplicate_main_groups"] == 0
    assert result["statistics"]["completion_passing_start_group_count"] == 0 and result["start_groups"]
    assert all(row["exact_start_group_index"] is None for row in lane["hypotheses"])
    statuses = [row["prefilter_status"] for row in lane["hypotheses"]]
    assert statuses == ["EXPIRED_BY_ALLOWANCE_UNKNOWN", "OMITTED_BY_ALLOWANCE_UNKNOWN", "OMITTED_BY_ALLOWANCE_UNKNOWN"]
    assert lane["refinement_schedule"]["lane_transaction"]["reason"] == "LANE_WORK_QUOTA:completion"
    assert configuration["meter"].lane_rows["completion"] == 100 == configuration["meter"].lane_rows_completed["completion"]
    configuration = scene._configuration(m.init)
    m.init._attach_allowance(configuration, m.v5.RefinementAllowanceV1(completion_query_rows=5), None, None)
    result, context = m.init._initialize(observed, scan_triangles, cad, faces, local, .025, configuration, UNIT)
    lane = result["completion_lane"]
    assert lane["status"] == "EXPIRED_BY_ALLOWANCE:LANE_WORK_QUOTA" and lane["transaction"]["expired_phase"] == "FACE_INDEX_BUILD"
    assert lane["face_index_evidence"] == [] and [row["prefilter_status"] for row in lane["hypotheses"]] == ["OMITTED_BY_ALLOWANCE_UNKNOWN"]*3
    assert configuration["meter"].lane_rows["completion"] == 0
    # 3. A lane budget refusal (declared budget, not an allowance) stays fatal for the whole initialization.
    configuration = scene._configuration(m.init)
    tiny = m.init.PlaneInitializationBudgetsV12(**dict(m.init.asdict(configuration["completion_budgets"]), max_prefilter_query_rows=10))
    configuration["completion_budgets"] = tiny
    m.init._attach_allowance(configuration, m.v5.RefinementAllowanceV1(), None, None)
    result, context = m.init._initialize(observed, scan_triangles, cad, faces, local, .025, configuration, UNIT)
    assert result["status"] != "COMPLETE" and context is None and result["completion_lane"]["usable"] is False


# --- Codex B4/B2 closure matrix: explicit candidate fixtures, controlled clocks, real V17 binding ----------

def _force_schedule(m, monkeypatch, groups):
    """Explicit candidate fixture: on the six-group context-V4 scene the MAIN refinement schedule is forced to
    exactly `groups` (exact group indices, in that order) when the lane quota equals their number; every
    other strata call is the real one. Group 4 is the scene's supported pose; the others are rejected contexts."""
    real = m.init._search._strata
    def strata(count, quota):
        return list(groups) if (count, quota) == (6, len(groups)) else list(real(count, quota))
    monkeypatch.setattr(m.init._search, "_strata", strata)


def _v17_arrays(observed, scan_triangles, cad, faces, local):
    cad_points = np.unique(cad.reshape(-1, 3), axis=0)
    return (dict(raw_points=observed, points=observed), dict(triangle_raw_rows=scan_triangles),
            dict(triangles=cad, source_face_indices=faces, source_triangle_indices=local, points=cad_points))


def _bind_v17(m, registration, arrays, *, context_index):
    """The REAL successor binder on an ACTUAL initializer-V16 registration (no handcrafted evidence)."""
    scan, topology, cad = _v17_arrays(*arrays)
    expected = m.init._context_configuration("v4", m.context4.TwoPlaneContextBudgetsV4())[1]
    return m.workflow17._registration_binding(registration, scan, topology, cad, None, .025, context_index=context_index,
                                              expected_two_plane_context_configuration=expected)


def test_first_candidate_supported_then_cpu_guard_omits_the_second_with_exact_ledger_and_real_v17_binding(m, fixture, monkeypatch):
    _force_schedule(m, monkeypatch, [4, 5])
    guard, clock, seen = 150., _fake_clock(step=0.), []
    def checkpoint(record):
        seen.append(record)
        if record["stage"] == "REFINE_CANDIDATE_END" and record["candidate"] == 0:
            clock.state["cpu"] = guard+1.          # the guard trips exactly after the first candidate's END checkpoint
    allowance = m.v5.RefinementAllowanceV1(main_refined_groups=2, cpu_guard_seconds=guard, cpu_budget_seconds=1860.)
    answer, roles, _ = _register(m, fixture, refinement_allowance=allowance, clock=clock, stage_checkpoint=checkpoint)
    evidence = answer["evidence"]
    assert answer["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS", answer["reasons"]
    rows = evidence["candidates"]
    assert [row["exact_group_index"] for row in rows] == [4, 5] and len(rows) == 2
    assert rows[0]["execution_status"] == "ATTEMPTED" and rows[0]["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"
    assert rows[1]["execution_status"] == "OMITTED" and rows[1]["reasons"] == ["OMITTED_BY_CPU_GUARD"]
    assert "conditional_context_support" not in rows[1] and "rotation" not in rows[1] and rows[1]["allowance_reason"] == "CPU_GUARD"
    ledger = evidence["candidate_disposition_ledger"]
    assert {k: ledger[k] for k in ("completed", "failed", "expired", "omitted", "not_attempted", "selected_count")} == \
        dict(completed=[0], failed=[], expired=[], omitted=[1], not_attempted=[], selected_count=2)
    assert evidence["candidate_evaluation_complete"] is False and evidence["evaluated_subset_complete"] is True
    assert evidence["candidate_omission_status"] == "ALLOWANCE_OMISSIONS_RETAINED_UNKNOWN"
    assert evidence["candidate_set_scope"] == "EVALUATED_ALLOWANCE_SUBSET_NOT_WHOLE_FINITE_SET"
    assert evidence["measurement_context_search_scope"].startswith("EVALUATED_ALLOWANCE_SUBSET")
    assert evidence["measurement_context_indices"] == [0] and roles is not None
    assert evidence["cost_receipts"]["cpu_guard_tripped"] is True and evidence["cost_receipts"]["lane_stopped"] == {}   # guard omission at candidate start is not a lane stop
    assert [r["stage"] for r in seen if r["stage"].startswith("REFINE_CANDIDATE")] == \
        ["REFINE_CANDIDATE_BEGIN", "REFINE_CANDIDATE_END", "REFINE_CANDIDATE_OMITTED"]
    assert m.context3._registration_evaluation_acceptable(evidence) is True
    # A forged strict whole-set flag contradicted by the published disposition ledger is refused by the helper
    # and by the real binder (never labelled STRICT_WHOLE_SET); legacy evidence without the new fields keeps
    # the unchanged V2 meaning.
    forged = dict(evidence, candidate_evaluation_complete=True)
    assert m.context3._registration_evaluation_acceptable(forged) is False
    legacy = {k: v for k, v in forged.items() if k not in ("candidate_disposition_ledger", "evaluated_subset_complete",
                                                           "publication_contract", "candidate_omission_status")}
    assert m.context3._registration_evaluation_acceptable(legacy) is True
    arrays, _ = fixture._arrays()
    forged_answer = copy.deepcopy(answer); forged_answer["evidence"]["candidate_evaluation_complete"] = True
    with pytest.raises(ValueError, match="INCOMPLETE_INITIALIZATION_CANNOT_REGISTER"):
        _bind_v17(m, forged_answer, arrays, context_index=0)
    # The ACTUAL initializer subset binds through the REAL V17 registration binder; the omitted candidate cannot.
    binding = _bind_v17(m, answer, arrays, context_index=0)
    assert binding["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING" and binding["start_origin_lanes"] == ["MAIN"]
    assert binding["registration_evaluation"] == "EVALUATED_ALLOWANCE_SUBSET_EXPIRED_AND_OMITTED_UNKNOWN"
    with pytest.raises(ValueError, match="INVALID_MEASUREMENT_CONTEXT_PUBLICATION"):
        _bind_v17(m, answer, arrays, context_index=1)
    # ...and a bare subset claim without its ledger is refused by the same real binder.
    bare = copy.deepcopy(answer); bare["evidence"]["candidate_disposition_ledger"] = None
    with pytest.raises(ValueError, match="INCOMPLETE_INITIALIZATION_CANNOT_REGISTER"):
        _bind_v17(m, bare, arrays, context_index=0)


@pytest.mark.parametrize("kind", ["surface", "unclassified"])
def test_fatal_failure_after_a_completed_candidate_is_failed_not_completed_and_later_candidates_stay_not_attempted(m, fixture, monkeypatch, kind):
    _force_schedule(m, monkeypatch, [4, 5, 3])       # supported first, the fatal one second, one never reached
    real, calls = m.v5._diagnose_with_reusable_query, []
    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise (m.v5._SurfaceFailure("INJECTED_SURFACE_FAILURE") if kind == "surface" else TypeError("INJECTED_UNCLASSIFIED"))
        return real(*args, **kwargs)
    reference, _, _ = _register(m, fixture, refinement_allowance=m.v5.RefinementAllowanceV1(main_refined_groups=3))
    assert [r["execution_status"] for r in reference["evidence"]["candidates"]] == ["ATTEMPTED"]*3
    monkeypatch.setattr(m.v5, "_diagnose_with_reusable_query", failing)
    answer, roles, _ = _register(m, fixture, refinement_allowance=m.v5.RefinementAllowanceV1(main_refined_groups=3))
    evidence = answer["evidence"]
    assert answer["status"] == "UNAVAILABLE" and roles is None and answer["transform_scan_to_cad"] is None
    rows = evidence["candidates"]
    assert [r["execution_status"] for r in rows] == ["ATTEMPTED", "FAILED", "NOT_ATTEMPTED"]
    completed = reference["evidence"]["candidates"][0]
    scheduled_only = ("supported_before_schedule",)        # added by context scheduling, which the fatal run never reaches
    assert rows[0]["conditional_context_support"] == {k: v for k, v in completed["conditional_context_support"].items() if k not in scheduled_only}
    assert rows[0]["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT" and rows[0]["reasons"] == completed["reasons"]
    assert rows[1]["reasons"] == (["INJECTED_SURFACE_FAILURE"] if kind == "surface" else ["UNCLASSIFIED_FATAL_FAILURE:TypeError"])
    assert rows[1]["numerical_consumer_status"].startswith("NOT_COMPLETED_FATAL") and "conditional_context_support" not in rows[1]
    assert rows[2]["reasons"] == ["NOT_ATTEMPTED_AFTER_FATAL_FAILURE"]
    ledger = evidence["candidate_disposition_ledger"]
    assert {k: ledger[k] for k in ("completed", "failed", "expired", "omitted", "not_attempted", "selected_count")} == \
        dict(completed=[0], failed=[1], expired=[], omitted=[], not_attempted=[2], selected_count=3)
    assert rows[1]["cost"]["disposition"] == "FAILED"
    assert evidence["candidate_evaluation_complete"] is False and evidence["evaluated_subset_complete"] is False
    assert evidence["numerical_consumer_complete"] is False and evidence["candidate_search_complete"] is False
    assert evidence["measurement_context_indices"] == [] and evidence["attempted_measurement_context_indices"] == []
    assert m.context3._registration_evaluation_acceptable(evidence) is False
    assert evidence["cost_receipts"]["lane_stopped"] == {}       # fatal is not a typed lane stop


def _modified_completion_arrays(scene):
    """The completion scene with the big z=0 face replaced by a small patch: one main triplet becomes a
    two-of-three parent whose passing completion child duplicates an existing MAIN group's pose."""
    arrays, nominal, pose = scene._arrays()
    observed, scan_triangles, cad, faces, local = arrays
    keep = faces != 37
    small = scene._grid((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 2.5, 4.), (2., 3.5, 5.), True)
    cad = np.concatenate((cad[keep], small[0][small[1]]))
    faces = np.concatenate((faces[keep], np.full(len(small[1]), 37, dtype=np.int64)))
    local = np.concatenate((local[keep], np.arange(len(small[1]), dtype=np.int64)))
    return observed, scan_triangles, cad, faces, local


def _register_completion_scene(m, scene, arrays, **overrides):
    observed, scan_triangles, cad, faces, local = arrays
    # A CAD-coverage floor the tiny scene can meet, so that main poses carry supported contexts here.
    policy = replace(scene._policy(m.init), min_cad_coverage=.01, min_scan_coverage=.6)
    options = dict(position_error_bounds_mm=.025, cad_initialization_points=np.unique(cad.reshape(-1, 3), axis=0),
                   registration_policy=policy, unit_assumption=UNIT,
                   search_budgets=m.init._search.PlaneSearchBudgetsV6(max_bank_planes=3), two_plane_context_mode="v4")
    options.update(overrides)
    return m.init.register_measurement_contexts_v16(observed, scan_triangles, cad, faces, local, **options)


def test_completion_rollback_restores_duplicate_main_provenance_and_the_real_v17_binder_accepts_the_retained_main(m):
    scene = _completion_scene(m)
    arrays = _modified_completion_arrays(scene)
    wire = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    full, _ = _register_completion_scene(m, scene, arrays, refinement_allowance=m.v5.RefinementAllowanceV1(main_refined_groups=48))
    init_full = full["evidence"]["initialization"]
    lane = init_full["completion_lane"]
    assert lane["status"] == "COMPLETE" and lane["duplicate_pose_count"] >= 1 and lane["passing_completion_count"] >= 1
    duplicated = [g for g in init_full["all_passing_start_groups"] if g.get("completion_indices")]
    assert len(duplicated) == 1 and duplicated[0].get("lane") is None and duplicated[0]["hypothesis_indices"]   # a MAIN group
    gi = duplicated[0]["exact_group_index"]
    child = [row for row in lane["hypotheses"] if row["prefilter_status"] == "PASSED" and row["exact_start_group_index"] == gi][0]
    later = [row for row in lane["hypotheses"] if row["completion_index"] > child["completion_index"]]
    assert later, "the fixture must schedule a child after the duplicate-main child"
    # Row allocation = the attempted rows through the duplicate child; the next child's first screen then expires.
    quota = sum(s["attempted_box_screen_rows"]+s["attempted_query_rows"] for row in lane["hypotheses"]
                if row["completion_index"] <= child["completion_index"] for s in row["assigned_support"])
    rolled, roles = _register_completion_scene(m, scene, arrays,
        refinement_allowance=m.v5.RefinementAllowanceV1(main_refined_groups=48, completion_query_rows=quota))
    init_rolled = rolled["evidence"]["initialization"]
    lane = init_rolled["completion_lane"]
    assert lane["status"] == "EXPIRED_BY_ALLOWANCE:LANE_WORK_QUOTA" and lane["usable"] is False
    transaction = lane["transaction"]
    assert transaction["restored_duplicate_main_groups"] == 1 and transaction["main_groups_restored_from_snapshot"] is True
    assert transaction["rolled_back_start_groups"] == 0 and transaction["expired_phase"] == "SUPPORT_EVALUATION"
    rows = {row["completion_index"]: row for row in lane["hypotheses"]}
    kept = rows[child["completion_index"]]
    assert kept["prefilter_status"] == "PASSED" and kept["assigned_support"] == child["assigned_support"]   # attempted evidence retained
    assert kept["exact_start_group_index"] is None and kept["refinement_disposition"] == "DISCARDED_BY_LANE_TRANSACTION_UNKNOWN"
    assert rows[later[0]["completion_index"]]["prefilter_status"] == "EXPIRED_BY_ALLOWANCE_UNKNOWN"
    assert not any(g.get("completion_indices") for g in init_rolled["all_passing_start_groups"])   # no dangling child index
    assert rolled["evidence"]["cost_receipts"]["lane_query_rows_attempted"]["completion"] == quota
    # Byte-equivalent main-group provenance: identical to the run whose completion lane is disabled.
    disabled, _ = _register_completion_scene(m, scene, arrays, refinement_allowance=m.v5.RefinementAllowanceV1(main_refined_groups=48),
                                             completion_policy=m.init.CompletionLanePolicyV1(enabled=False))
    init_disabled = disabled["evidence"]["initialization"]
    assert wire(init_rolled["all_passing_start_groups"]) == wire(init_disabled["all_passing_start_groups"])
    assert wire(init_rolled["start_groups"]) == wire(init_disabled["start_groups"])
    assert init_full["all_passing_start_groups"][gi]["completion_indices"] == [child["completion_index"]]
    assert {k: v for k, v in init_full["all_passing_start_groups"][gi].items() if k != "completion_indices"} == init_rolled["all_passing_start_groups"][gi]
    # The retained main context of the duplicated group is accepted by the REAL V17 binder...
    candidate = [row for row in rolled["evidence"]["candidates"] if row["exact_group_index"] == gi][0]
    assert candidate["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"
    assert candidate["index"] in rolled["evidence"]["measurement_context_indices"]
    binding = _bind_v17(m, rolled, arrays, context_index=candidate["index"])
    assert binding["start_origin_lanes"] == ["MAIN"] and binding["registration_evaluation"] == "STRICT_WHOLE_SET"
    # ...whereas the dangling pre-fix provenance (origin into the rolled-back, unusable lane) is refused by it.
    dangling = copy.deepcopy(rolled)
    for group in (dangling["evidence"]["initialization"]["start_groups"][candidate["index"]],
                  dangling["evidence"]["initialization"]["all_passing_start_groups"][gi],
                  dangling["evidence"]["candidates"][candidate["index"]]):
        group["completion_indices"] = [child["completion_index"]]
    with pytest.raises(ValueError, match="SELECTED_GROUP_COMPLETION_ORIGIN_LEDGER_INCOMPLETE|INCOMPLETE_COMPLETION_LANE_PUBLICATION"):
        _bind_v17(m, dangling, arrays, context_index=candidate["index"])


def test_completion_face_queries_charge_attempted_rows_before_work_and_complete_only_after_success(m, monkeypatch):
    scene = _completion_scene(m)
    arrays, nominal, pose = scene._arrays()
    configuration = scene._configuration(m.init)
    result, context = m.init._initialize(*arrays, .025, configuration, UNIT)
    faces = scene._faces(context)
    m.init._old._bound_support(context, cad_patch=faces[17])
    points = np.random.default_rng(0).uniform(0., 10., (40, 3))
    def queries(allowance):
        meter = m.v5._CpuMeter(allowance, clock=_fake_clock(step=0.))
        shim = dict(statistics=m.init._completion_statistics(), face_index_evidence=[])
        query = m.init._FaceQueriesV16(context, {17: faces[17]}, configuration["completion_budgets"],
                                       configuration["surface_budgets"], shim, meter=meter)
        meter.begin("completion", "PROBE")
        return query, meter
    query, meter = queries(m.v5.RefinementAllowanceV1())
    query.screen(17, points, .5)
    assert meter.lane_rows["completion"] == 40 == meter.lane_rows_completed["completion"]
    query(17, points)
    assert meter.lane_rows["completion"] == 80 == meter.lane_rows_completed["completion"] and meter.lane_calls_completed["completion"] == 2
    # A primitive failure AFTER admission: attempted rows stay charged, completed rows do not; fatal, not typed.
    monkeypatch.setattr(m.init._box, "screen_face_box_rows", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("INJECTED_PRIMITIVE_FAILURE")))
    with pytest.raises(RuntimeError, match="INJECTED_PRIMITIVE_FAILURE"):
        query.screen(17, points, .5)
    assert meter.lane_rows["completion"] == 120 and meter.lane_rows_completed["completion"] == 80
    assert meter.lane_calls["completion"] == 3 and meter.lane_calls_completed["completion"] == 2
    monkeypatch.undo()
    # A typed stop BEFORE admission charges nothing.
    query, meter = queries(m.v5.RefinementAllowanceV1(completion_query_rows=50))
    query.screen(17, points, .5)
    with pytest.raises(m.v5._AllowanceExhausted, match="LANE_WORK_QUOTA:completion"):
        query.screen(17, points, .5)
    assert meter.lane_rows["completion"] == 40 == meter.lane_rows_completed["completion"]
    with pytest.raises(m.v5._AllowanceExhausted):
        query(17, points)
    assert meter.lane_rows["completion"] == 40


def test_completion_lane_typed_stops_before_preparation_at_zero_row_allocation_and_between_face_builds(m, monkeypatch):
    scene = _completion_scene(m)
    arrays = _modified_completion_arrays(scene)
    def run(allowance=None, clock=None):
        configuration = scene._configuration(m.init)
        if allowance is not None:
            m.init._attach_allowance(configuration, allowance, clock, None)
        result, context = m.init._initialize(*arrays, .025, configuration, UNIT)
        assert result["status"] == "COMPLETE", result["reasons"]
        return result["completion_lane"], configuration
    # (i) the absolute guard is already exceeded before the lane: nothing is enumerated, bound or built.
    lane, configuration = run(m.v5.RefinementAllowanceV1(cpu_guard_seconds=150.), _fake_clock(step=0., start=200.))
    assert lane["status"] == "EXPIRED_BY_ALLOWANCE:CPU_GUARD" and lane["transaction"]["expired_phase"] == "ENUMERATION"
    assert lane["parents"] == [] and lane["hypotheses"] == [] and lane["face_index_evidence"] == []
    assert configuration["meter"].lane_rows["completion"] == 0 and configuration["meter"].guard_tripped is True
    # (ii) a zero completion-row allocation expires BEFORE any face index is built: the first query is inadmissible.
    lane, configuration = run(m.v5.RefinementAllowanceV1(completion_query_rows=0))
    assert lane["status"] == "EXPIRED_BY_ALLOWANCE:LANE_WORK_QUOTA" and lane["transaction"]["expired_phase"] == "FACE_INDEX_BUILD"
    assert lane["face_index_evidence"] == [] and lane["hypotheses"] and configuration["meter"].lane_rows["completion"] == 0
    assert all(row["prefilter_status"] == "OMITTED_BY_ALLOWANCE_UNKNOWN" for row in lane["hypotheses"])
    # (iii) expiry between two eager face-index builds: exactly one face is built, no query is attempted.
    clock, built = _fake_clock(step=0.), []
    original = m.init._FaceQueriesV16._stop
    def hook(self, rows):
        if rows == 0:                          # the between-build stop (screens and queries carry rows > 0)
            if built:
                clock.state["cpu"] = 1e6      # past the guard before the SECOND build
            built.append(rows)
        original(self, rows)
    monkeypatch.setattr(m.init._FaceQueriesV16, "_stop", hook)
    lane, configuration = run(m.v5.RefinementAllowanceV1(), clock)
    assert lane["status"] == "EXPIRED_BY_ALLOWANCE:CPU_GUARD" and lane["transaction"]["expired_phase"] == "FACE_INDEX_BUILD"
    assert len(built) == 2 and len(lane["face_index_evidence"]) == 1 and configuration["meter"].lane_rows["completion"] == 0
    assert all(row["prefilter_status"] == "OMITTED_BY_ALLOWANCE_UNKNOWN" for row in lane["hypotheses"])
    monkeypatch.undo()
    # (iv) the completion query chunk is capped by the cooperative chunk when an allowance is active.
    lane, configuration = run(m.v5.RefinementAllowanceV1(cooperative_chunk_rows=1024))
    assert lane["query_chunk_rows"] == 1024 and lane["status"] == "COMPLETE"
    lane, configuration = run(m.v5.RefinementAllowanceV1())
    assert lane["query_chunk_rows"] == 32768
    lane, configuration = run()
    assert lane["query_chunk_rows"] == min(configuration["completion_budgets"].chunk_rows, configuration["surface_budgets"].max_query_rows) == 65536
    assert lane["phase"] == "COMPLETE" and "expired_phase" not in lane


def test_completion_lane_plans_and_issues_every_query_within_the_cooperative_chunk(m, monkeypatch):
    """Full lane on the modified scene (25-row supports): every screen/query stays within the 1024-row cap and
    the planned call denominator is computed with the capped chunk (declared, not inferred)."""
    scene = _completion_scene(m)
    arrays = _modified_completion_arrays(scene)
    calls, screens = [], []
    original, original_screen = m.init._FaceQueriesV16.__call__, m.init._FaceQueriesV16.screen
    monkeypatch.setattr(m.init._FaceQueriesV16, "__call__", lambda self, face, points: (calls.append(len(points)), original(self, face, points))[1])
    monkeypatch.setattr(m.init._FaceQueriesV16, "screen", lambda self, face, points, gate: (screens.append(len(points)), original_screen(self, face, points, gate))[1])
    configuration = scene._configuration(m.init)
    m.init._attach_allowance(configuration, m.v5.RefinementAllowanceV1(cooperative_chunk_rows=1024), None, None)
    result, context = m.init._initialize(*arrays, .025, configuration, UNIT)
    lane = result["completion_lane"]
    assert lane["status"] == "COMPLETE" and lane["query_chunk_rows"] == 1024 and calls and screens
    assert max(calls+screens) <= 1024
    stats = lane["statistics"]
    assert stats["planned_query_calls"] == sum((s["support_count"]+1024-1)//1024 for row in lane["hypotheses"] for s in row["assigned_support"])


def test_a_completion_support_longer_than_the_cooperative_chunk_is_screened_and_queried_in_capped_chunks(m, monkeypatch):
    """A 1500-row support through the REAL per-support evaluator and the REAL metered V16 face queries: box
    screens and finite queries are issued as 1024 + 476 rows, the whole support is accounted attempted and
    completed, and no primitive call exceeds the declared cooperative chunk."""
    scene = _completion_scene(m)
    arrays, nominal, pose = scene._arrays()
    configuration = scene._configuration(m.init)
    result, context = m.init._initialize(*arrays, .025, configuration, UNIT)
    faces = scene._faces(context)
    passing = scene._passing_triplet(result)
    support = scene._support_on_face(passing, 17)
    patch = dict(context["scan_supports"][support["scan_plane_id"]])
    patch["raw_vertex_rows"] = np.tile(np.asarray(patch["raw_vertex_rows"], dtype=np.int64), 60)     # 25 rows -> 1500 rows
    assert len(patch["raw_vertex_rows"]) == 1500
    scans = dict(context["scan_supports"]); scans[support["scan_plane_id"]] = patch
    match = dict(scan_plane_id=support["scan_plane_id"], cad_plane_id=17, normal_sign=support["normal_sign"])
    row = dict(rotation=passing["rotation"], translation_mm=passing["translation_mm"], prefilter_status="NOT_ATTEMPTED",
               prefilter_reasons=[], assigned_support=[m.init._fresh_support(match, patch, faces[17])])
    m.init._old._bound_support(context, cad_patch=faces[17])
    meter = m.v5._CpuMeter(m.v5.RefinementAllowanceV1(cooperative_chunk_rows=1024), clock=_fake_clock(step=0.))
    stats = m.init._completion_statistics()
    query = m.init._FaceQueriesV16(context, {17: faces[17]}, configuration["completion_budgets"], configuration["surface_budgets"],
                                   dict(statistics=stats, face_index_evidence=[]), meter=meter)
    calls, screens = [], []
    original, original_screen = m.init._FaceQueriesV16.__call__, m.init._FaceQueriesV16.screen
    monkeypatch.setattr(m.init._FaceQueriesV16, "__call__", lambda self, face, points: (calls.append(len(points)), original(self, face, points))[1])
    monkeypatch.setattr(m.init._FaceQueriesV16, "screen", lambda self, face, points, gate: (screens.append(len(points)), original_screen(self, face, points, gate))[1])
    meter.begin("completion", "PROBE")
    chunk = min(configuration["completion_budgets"].chunk_rows, configuration["surface_budgets"].max_query_rows, 1024)
    passed, rotation, translation = m.init._evaluate_completion_supports(row, context, scans, query, chunk, configuration["policy"], stats)
    assert passed is True and row["prefilter_status"] == "PASSED"
    assert screens == [1024, 476] and calls == [1024, 476]
    evaluated = row["assigned_support"][0]
    assert evaluated["support_count"] == evaluated["completed_box_screen_rows"] == evaluated["query_rows"] == 1500
    assert meter.lane_rows["completion"] == 3000 == meter.lane_rows_completed["completion"] and meter.lane_calls_completed["completion"] == 4
    assert stats["completed_query_rows"] == 1500 and stats["completed_box_screen_rows"] == 1500


def _long_support_probe(m, chunk_rows=1024):
    """The 1500-row support fixture through the REAL per-support evaluator and the REAL metered V16 face
    queries, with a controllable clock (returns everything the guard-stop tests need)."""
    scene = _completion_scene(m)
    arrays, nominal, pose = scene._arrays()
    configuration = scene._configuration(m.init)
    result, context = m.init._initialize(*arrays, .025, configuration, UNIT)
    faces = scene._faces(context)
    passing = scene._passing_triplet(result)
    support = scene._support_on_face(passing, 17)
    patch = dict(context["scan_supports"][support["scan_plane_id"]])
    patch["raw_vertex_rows"] = np.tile(np.asarray(patch["raw_vertex_rows"], dtype=np.int64), 60)
    scans = dict(context["scan_supports"]); scans[support["scan_plane_id"]] = patch
    match = dict(scan_plane_id=support["scan_plane_id"], cad_plane_id=17, normal_sign=support["normal_sign"])
    row = dict(rotation=passing["rotation"], translation_mm=passing["translation_mm"], prefilter_status="NOT_ATTEMPTED",
               prefilter_reasons=[], assigned_support=[m.init._fresh_support(match, patch, faces[17])])
    m.init._old._bound_support(context, cad_patch=faces[17])
    clock = _fake_clock(step=0.)
    meter = m.v5._CpuMeter(m.v5.RefinementAllowanceV1(cooperative_chunk_rows=chunk_rows, cpu_guard_seconds=150.), clock=clock)
    stats = m.init._completion_statistics()
    query = m.init._FaceQueriesV16(context, {17: faces[17]}, configuration["completion_budgets"], configuration["surface_budgets"],
                                   dict(statistics=stats, face_index_evidence=[]), meter=meter)
    meter.begin("completion", "PROBE")
    chunk = min(configuration["completion_budgets"].chunk_rows, configuration["surface_budgets"].max_query_rows, chunk_rows)
    return types.SimpleNamespace(row=row, context=context, scans=scans, query=query, chunk=chunk, policy=configuration["policy"],
                                 stats=stats, meter=meter, clock=clock)


@pytest.mark.parametrize("stop_after", ["first_screen_chunk", "first_query_chunk"])
def test_guard_expiry_between_chunks_of_a_long_support_stops_before_the_next_primitive_with_exact_counts(m, monkeypatch, stop_after):
    """The clock jumps past the guard right after the first successful screen chunk (or the first successful
    finite-query chunk): the typed stop fires BEFORE the next primitive, attempted equals completed exactly,
    and the partially evaluated child is never PASSED (INCOMPLETE until the lane rollback marks it unknown)."""
    probe = _long_support_probe(m)
    calls, screens = [], []
    original, original_screen = m.init._FaceQueriesV16.__call__, m.init._FaceQueriesV16.screen
    def screen(self, face, points, gate):
        answer = original_screen(self, face, points, gate)
        screens.append(len(points))
        if stop_after == "first_screen_chunk" and len(screens) == 1:
            probe.clock.state["cpu"] = 1e6
        return answer
    def call(self, face, points):
        answer = original(self, face, points)
        calls.append(len(points))
        if stop_after == "first_query_chunk" and len(calls) == 1:
            probe.clock.state["cpu"] = 1e6
        return answer
    monkeypatch.setattr(m.init._FaceQueriesV16, "screen", screen)
    monkeypatch.setattr(m.init._FaceQueriesV16, "__call__", call)
    with pytest.raises(m.v5._AllowanceExhausted, match="CPU_GUARD"):
        m.init._evaluate_completion_supports(probe.row, probe.context, probe.scans, probe.query, probe.chunk, probe.policy, probe.stats)
    meter, support = probe.meter, probe.row["assigned_support"][0]
    if stop_after == "first_screen_chunk":
        assert screens == [1024] and calls == []                       # no next primitive
        assert meter.lane_rows["completion"] == 1024 == meter.lane_rows_completed["completion"]
        assert meter.lane_calls["completion"] == 1 == meter.lane_calls_completed["completion"]
        assert support["completed_box_screen_rows"] == 1024 and support["box_screening_status"] == "INCOMPLETE"
        assert support["query_rows"] == 0 and support["finite_query_disposition"] == "NOT_ATTEMPTED"
    else:
        assert screens == [1024, 476] and calls == [1024]              # the second query chunk never starts
        assert meter.lane_rows["completion"] == 1500+1024 == meter.lane_rows_completed["completion"]
        assert meter.lane_calls["completion"] == 3 == meter.lane_calls_completed["completion"]
        assert support["box_screening_status"] == "COMPLETE" and support["query_rows"] == 1024
        assert support["finite_query_disposition"] == "INCOMPLETE"
    assert support["status"] == "INCOMPLETE" and probe.row["prefilter_status"] == "INCOMPLETE"   # never PASSED
    assert probe.stats["passing_hypothesis_count"] == 0 and probe.stats["hypotheses_evaluated"] == 0
    assert meter.guard_tripped is True
    # In the lane, that typed expiry is what _rollback_completion_lane turns into EXPIRED_BY_ALLOWANCE_UNKNOWN.
    assert m.init._rollback_completion_lane(dict(completion_lane=dict(hypotheses=[probe.row])), [], {}, 0,
                                            m.v5._AllowanceExhausted("CPU_GUARD"))["status"] == "ROLLED_BACK_BY_TYPED_ALLOWANCE_EXPIRY"
    assert probe.row["prefilter_status"] == "EXPIRED_BY_ALLOWANCE_UNKNOWN"


def test_checkpoint_snapshots_are_owned_and_reconciled_at_each_publication_depth(m, fixture):
    seen = []
    answer, _, _ = _register(m, fixture, refinement_allowance=m.v5.RefinementAllowanceV1(), stage_checkpoint=lambda r: seen.append(r))
    evidence = answer["evidence"]
    inner, outer = evidence["initialization"]["stage_checkpoints"], evidence["stage_checkpoints"]
    encoded = lambda records: sum(len(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False)) for r in records)
    assert inner["count"] == len(inner["records"]) and inner["bytes"] == encoded(inner["records"])
    assert outer["count"] == len(outer["records"]) == len(seen) and outer["bytes"] == encoded(outer["records"])
    assert 0 < inner["count"] < outer["count"] and outer["records"][:inner["count"]] == inner["records"] == seen[:inner["count"]]
    assert inner["reconciliation"].startswith("COUNT_EQUALS_OWNED_RECORDS_AND_BYTES_EQUAL_THEIR_ENCODED_SUM")
    assert inner["records"] is not outer["records"] and all(a is not b for a, b in zip(inner["records"], outer["records"]))
    outer["records"].append(dict(stage="INJECTED_LATER"))          # a later mutation cannot reach the inner snapshot
    assert inner["count"] == len(inner["records"]) and inner["records"][-1]["stage"] != "INJECTED_LATER"
