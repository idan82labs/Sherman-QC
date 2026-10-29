"""Independent source-authored context-capacity integration regression.

The actual initializer cases use only 50 invented raw scan rows, 64 raw scan
triangles and either four or eight CAD triangles. The four-patch case forces
eight ordered CAD assignments onto ONE scan pair, beyond the old four-target
consumer quota. It runs real context generation and finite prefiltering, not
registration refinement, native loading, measurement or physical input.

The byte cases reuse qualified ordinary constructors and pytest fixtures. Their
native/search/measurement seams and context metadata are explicitly mocked;
the small nonempty schedule, request/profile checks and owned-array binders are
real. Consistent reported metadata is not recomputed context geometry or proof.
No fixture is called as an ordinary function. No backend import or computation
occurs at module import. This source does not assert that an unrun check passed.
"""
from dataclasses import asdict
import copy
import importlib

import numpy as np
import pytest

from .test_cad_guided_two_plane_schedule_integration_v3 import (
    modules, adapter, _tiny_rectangles, _prefilter, _owned, _sha,
)


def _context_module(m, mode):
    return m.initialize._context if mode == "v1" else importlib.import_module(
        "backend.cad_guided_two_plane_context_v2")


def _profile(m, mode):
    context = _context_module(m, mode)
    budget_type = getattr(context, "TwoPlaneContextBudgets"+mode.upper())
    result = dict(schema_version="two_plane_context_"+mode, budgets=asdict(budget_type()))
    if mode == "v2":
        result["scheduling_capacity"] = copy.deepcopy(context._base()["scheduling_capacity"])
    return result


def _four_cad_patches():
    """Same scan, plus one displaced CAD copy with separate source identities."""
    inputs, nominal = _tiny_rectangles()
    cad = _owned(np.concatenate((inputs[2], inputs[2]+np.array([32., 24., 40.]))))
    faces = _owned([101, 101, 307, 307, 1301, 1301, 1607, 1607], "<i8")
    local = _owned([0, 1, 0, 1, 0, 1, 0, 1], "<i8")
    return (inputs[0], inputs[1], cad, faces, local), nominal


def _options(m, mode):
    return {} if mode == "v1" else dict(two_plane_schedule_mode="v2",
        two_plane_schedule_budgets=m.schedule_v2.TwoPlaneScheduleBudgetsV2())


@pytest.mark.parametrize("mode,ordered", (("v1", 4), ("v2", 8)))
def test_actual_one_scan_pair_exceeds_old_quota_without_losing_finite_truth(modules, monkeypatch, mode, ordered):
    inputs, nominal = _four_cad_patches()
    before = tuple(value.tobytes() for value in inputs)
    calls = []
    for version in ("v1", "v2"):
        context = _context_module(modules, version)
        name = "propose_two_plane_context_"+version
        original = getattr(context, name)
        def spy(*args, _version=version, _original=original, **kwargs):
            calls.append((_version, args, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(context, name, spy)
    answer = _prefilter(modules, inputs, **_options(modules, mode))
    assert answer["status"] == "COMPLETE" and answer["start_groups"], answer["reasons"]
    assert [row[0] for row in calls] == [mode]
    assert len(calls[0][1][7]) == ordered
    if mode == "v1":
        assert "budgets" not in calls[0][2]
    else:
        assert type(calls[0][2]["budgets"]) is _context_module(modules, "v2").TwoPlaneContextBudgetsV2
    assert len(inputs[0]) == 50 and len(inputs[1]) == 64 and len(inputs[2]) == 8
    assert len(answer["hypothesis_evidence"]["scan_bank"]) == 2
    assert len(answer["patch_evidence"]["cad"]["patches"]) == 4
    schedule = answer["two_plane_schedule"]
    matches = schedule["scheduled_matches"]
    assert len(matches) == ordered
    assert len({tuple(row["scan_plane_ids"]) for row in matches}) == 1
    cad_orders = {tuple(row["cad_plane_ids"]) for row in matches}
    assert len(cad_orders) == ordered
    assert all(pair[::-1] in cad_orders for pair in cad_orders)
    assert len({tuple(sorted(pair)) for pair in cad_orders}) == ordered//2
    if mode == "v2":
        assert len(schedule["breadth_ledger"]["scheduled_matches"]) == 4
        assert matches[:4] == schedule["breadth_ledger"]["scheduled_matches"]
    joint = answer["joint_proposal_accounting"]
    assert joint["triplet_records"] == joint["triplet_unsigned_reservation"] == 0
    assert joint["ordered_pair_matches"] == ordered
    assert joint["pair_slot_reservation"] == joint["total_hypothesis_records"] == 32*ordered
    assert len(answer["hypotheses"]) == 32*ordered
    helper = answer["two_plane_context"]
    modules.initialize._validate_two_plane_context_configuration(helper, mode)
    stats = helper["statistics"]
    assert stats["signed_families_planned"] == stats["signed_families_attempted"] == stats["signed_families_completed"] == 4*ordered
    assert stats["candidate_slots_planned"] == stats["slot_records_completed"] == 32*ordered
    assert len(helper["families"]) == 4*ordered
    assert all(len(family["slots"]) == 8 for family in helper["families"])
    assert helper["accepted_registration"] is helper["finite_support_match_verified"] is False
    assert answer["statistics"]["completed_query_calls"] > 0
    # Test the complete finite-passing pool, not a monotonicity claim about the
    # subsequent <=48-group refinement subsampling policy.
    truths = [group for group in answer["all_passing_start_groups"] if
        np.max(np.linalg.norm(inputs[0]@np.asarray(group["initialized_rotation"]).T
            +group["initialized_translation_mm"]-nominal, axis=1)) < 1e-8]
    assert truths
    assert all(abs(np.linalg.det(np.asarray(group["initialized_rotation"]))-1.) < 1e-12 for group in truths)
    assert before == tuple(value.tobytes() for value in inputs)
    assert all(not value.flags.writeable for value in inputs)
    expected_context = {name+"_sha256": _sha(value) for name, value in zip(
        ("scan_points", "scan_triangles", "cad_triangles", "cad_face_ids", "cad_source_triangle_ids"), inputs)}
    assert helper["input_digests"] == expected_context
    assert answer["input_digests"]["scan_points_sha256"] == _sha(inputs[0])
    assert answer["input_digests"]["scan_triangle_raw_rows_sha256"] == _sha(inputs[1])
    assert answer["accepted_registration"] is False and answer["observed_bend_count"] is None


def test_actual_omitted_and_explicit_v1_still_have_identical_small_results(modules):
    inputs, _ = _tiny_rectangles()
    default = _prefilter(modules, inputs)
    explicit = _prefilter(modules, inputs, two_plane_schedule_mode="v1", two_plane_schedule_budgets=None)
    assert default["status"] == explicit["status"] == "COMPLETE"
    assert default == explicit
    assert default["two_plane_context"]["schema_version"] == "two_plane_context_v1"
    assert "scheduling_capacity" not in default["two_plane_context"]
    assert "two_plane_schedule_configuration" not in default


@pytest.mark.parametrize("case", ("empty_v1", "empty_v2", "disabled_v1"))
def test_actual_empty_or_disabled_pair_lane_never_calls_a_context_consumer(modules, monkeypatch, case):
    inputs, _ = _tiny_rectangles()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("context")
        raise AssertionError("EMPTY_OR_DISABLED_LANE_MUST_NOT_CONSUME_A_CONTEXT_PROFILE")
    for mode in ("v1", "v2"):
        monkeypatch.setattr(_context_module(modules, mode), "propose_two_plane_context_"+mode, forbidden)
    monkeypatch.setattr(modules.initialize, "_validate_two_plane_context_configuration", forbidden)
    if case == "empty_v1":
        options = dict(budgets=modules.initialize.PlaneInitializationBudgetsV12(max_hypothesis_records=0))
    elif case == "empty_v2":
        options = dict(two_plane_schedule_mode="v2",
            two_plane_schedule_budgets=modules.schedule_v2.TwoPlaneScheduleBudgetsV2(max_selected_scan_pairs=0))
    else: options = dict(two_plane_enabled=False)
    answer = _prefilter(modules, inputs, **options)
    assert answer["status"] == "COMPLETE" and answer["start_groups"] == [], answer["reasons"]
    assert calls == [] and answer["statistics"]["completed_query_calls"] == 0
    assert answer["hypotheses"] == []
    joint = answer["joint_proposal_accounting"]
    assert joint["ordered_pair_matches"] == joint["pair_slot_reservation"] == 0
    assert joint["pair_lane_enabled"] is (case != "disabled_v1")
    if case == "disabled_v1":
        assert answer["two_plane_context"] is None
        assert answer["two_plane_schedule"]["status"] == "DISABLED_BY_EXPLICIT_POLICY"
    else:
        assert answer["two_plane_context"] == dict(status="NOT_CALLED_EMPTY_DECLARED_SCHEDULE",
            usable_proposals=False, accepted_registration=False)


@pytest.mark.parametrize("mode", ("v1", "v2"))
@pytest.mark.parametrize("fault", ("missing", "extra", "changed", "float", "bool"))
def test_configuration_checks_every_budget_field_with_exact_types(modules, mode, fault):
    baseline = _profile(modules, mode)
    modules.initialize._validate_two_plane_context_configuration(baseline, mode)
    # All 18 fields, not only the newly changed per-pair quota. Each damaged
    # mapping starts fresh so the first rejected field cannot mask later ones.
    assert len(baseline["budgets"]) == 18
    for key, value in baseline["budgets"].items():
        damaged = copy.deepcopy(baseline)
        if fault == "missing": damaged["budgets"].pop(key)
        elif fault == "extra": damaged["budgets"]["unexpected_"+key] = value
        elif fault == "changed": damaged["budgets"][key] = value-1
        elif fault == "float": damaged["budgets"][key] = float(value)
        else: damaged["budgets"][key] = True
        with pytest.raises(ValueError, match="TWO_PLANE_CONTEXT_BUDGET_PROFILE_MISMATCH"):
            modules.initialize._validate_two_plane_context_configuration(damaged, mode)


@pytest.mark.parametrize("fault", ("missing", "none", "extra", "policy", "float", "bool"))
def test_v2_configuration_requires_the_complete_exact_capacity_profile(modules, fault):
    helper = _profile(modules, "v2")
    if fault == "missing": helper.pop("scheduling_capacity")
    elif fault == "none": helper["scheduling_capacity"] = None
    elif fault == "extra": helper["scheduling_capacity"]["unrequested"] = 0
    elif fault == "policy": helper["scheduling_capacity"]["policy"] = "UNREQUESTED_POLICY"
    elif fault == "float": helper["scheduling_capacity"]["maximum_ordered_matches"] = 256.
    else: helper["scheduling_capacity"]["slots_per_sign"] = True
    with pytest.raises(ValueError, match="TWO_PLANE_CONTEXT_CAPACITY_MISMATCH"):
        modules.initialize._validate_two_plane_context_configuration(helper, "v2")


@pytest.mark.parametrize("mode,fault,reason", (
    ("v2", "schema", "TWO_PLANE_CONTEXT_SCHEMA_MISMATCH"),
    ("v2", "missing_budget", "TWO_PLANE_CONTEXT_BUDGET_PROFILE_MISMATCH"),
    ("v2", "nonquota_budget", "TWO_PLANE_CONTEXT_BUDGET_PROFILE_MISMATCH"),
    ("v2", "boolean_budget", "TWO_PLANE_CONTEXT_BUDGET_PROFILE_MISMATCH"),
    ("v2", "missing_capacity", "TWO_PLANE_CONTEXT_CAPACITY_MISMATCH"),
    ("v2", "changed_capacity", "TWO_PLANE_CONTEXT_CAPACITY_MISMATCH"),
    ("v1", "schema", "TWO_PLANE_CONTEXT_SCHEMA_MISMATCH"),
    ("v1", "unrequested_capacity", "UNREQUESTED_V2_CONTEXT_CAPACITY"),
))
def test_actual_invalid_context_return_never_constructs_finite_queries(modules, monkeypatch, mode, fault, reason):
    inputs, _ = _tiny_rectangles()
    context = _context_module(modules, mode)
    name = "propose_two_plane_context_"+mode
    original = getattr(context, name)
    calls = []
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result["status"] == "COMPLETE" and result["statistics"]["slot_records_completed"] == 64
        calls.append("context_complete")
        if fault == "schema": result["schema_version"] = "two_plane_context_"+("v1" if mode == "v2" else "v2")
        elif fault == "missing_budget": result["budgets"].pop("max_hash_bytes")
        elif fault == "nonquota_budget": result["budgets"]["max_scan_row_visits"] -= 1
        elif fault == "boolean_budget": result["budgets"]["max_projection_cache_entries"] = True
        elif fault == "missing_capacity": result.pop("scheduling_capacity")
        elif fault == "changed_capacity": result["scheduling_capacity"]["maximum_ordered_matches"] -= 1
        else: result["scheduling_capacity"] = None
        return result
    def forbidden(*args, **kwargs):
        calls.append("finite_queries")
        raise AssertionError("PROFILE_VALIDATION_MUST_PRECEDE_FINITE_QUERY_CONSTRUCTION")
    monkeypatch.setattr(context, name, damaged)
    monkeypatch.setattr(modules.initialize, "_FaceQueries", forbidden)
    answer = _prefilter(modules, inputs, **_options(modules, mode))
    assert calls == ["context_complete"]
    assert answer["status"] == "UNAVAILABLE" and answer["start_groups"] == []
    assert answer["selected_policy_evaluation_complete"] is False
    assert answer["numerical_consumer_complete"] is False
    assert answer["statistics"]["completed_query_calls"] == 0
    assert reason in answer["reasons"]


def _byte_pairs(state, m, mode):
    """Nonempty real schedule + explicitly mocked context publication only.

    Keep the existing two mocked triplet-origin measurement contexts. Additional
    pair records are deliberately rejected placeholders, not generated poses.
    This fixture proves binding reaches a nonzero-pair profile check, not that
    the helper actually produced these byte-workflow context records.
    """
    state.install("v2")
    initialization = state.registration["evidence"]["initialization"]
    evidence = state.registration["evidence"]
    equation = m.initialize._search.PlaneEquationV6
    scans = tuple(equation(row["equation"]["plane_id"], tuple(row["equation"]["normal"]),
        tuple(row["equation"]["anchor_mm"]), row["equation"]["offset_from_anchor_mm"],
        row["equation"]["normal_angle_bound_radians"]) for row in initialization["hypothesis_evidence"]["scan_planes"])
    cads = tuple(equation(row["plane_id"], tuple(row["normal"]), tuple(row["area_centroid_mm"]), 0., 0.)
        for row in initialization["patch_evidence"]["cad"]["patches"])
    scheduler = getattr(m, "schedule_"+mode)
    kwargs = dict(retained_triplet_hypothesis_count=2, retained_triplet_unsigned_count=2)
    if mode == "v2":
        state.schedule_budgets = m.schedule_v2.TwoPlaneScheduleBudgetsV2()
        kwargs["budgets"] = state.schedule_budgets
        state.expected = dict(mode="v2", budgets=asdict(state.schedule_budgets))
        for publication in (evidence, initialization):
            publication["two_plane_schedule_configuration"] = copy.deepcopy(state.expected)
    else:
        state.expected = None
        for publication in (evidence, initialization): publication.pop("two_plane_schedule_configuration")
    schedule = getattr(scheduler, "schedule_two_plane_pairs_"+mode)(scans, cads, **kwargs)
    assert schedule["status"] == "COMPLETE" and len(schedule["scheduled_matches"]) == 8
    initialization["two_plane_schedule"] = schedule
    count = len(schedule["scheduled_matches"])
    initialization["joint_proposal_accounting"].update(ordered_pair_matches=count,
        pair_slot_reservation=32*count, total_hypothesis_records=2+32*count)
    initialization["hypotheses"].extend(dict(hypothesis_index=index,
        proposal_lane="TWO_PLANE_CONTEXT", status="REJECTED", prefilter_status="NOT_APPLICABLE",
        exact_start_group_index=None, fixture="MOCK_REJECTED_RECORD_NOT_CONTEXT_GEOMETRY")
        for index in range(2, 2+32*count))
    helper = _profile(m, mode)
    helper.update(status="COMPLETE", selected_policy_evaluation_complete=True,
        accepted_registration=False, finite_support_match_verified=False,
        statistics=dict(slot_records_completed=32*count),
        fixture="MOCK_CONTEXT_METADATA_NOT_RECOMPUTED_GEOMETRY",
        input_digests={key+"_sha256": initialization["input_digests"][name+"_sha256"]
            for key, name in (("scan_points", "scan_points"), ("scan_triangles", "scan_triangle_raw_rows"),
                ("cad_triangles", "cad_triangles"), ("cad_face_ids", "cad_face_ids"),
                ("cad_source_triangle_ids", "cad_source_triangle_ids"))})
    initialization["two_plane_context"] = helper
    return helper


def _bind(state, m, index=0):
    return m.workflow_v12._registration_binding(state.registration, state.scan, state.topology,
        state.cad, state.scan["engine_to_raw_row_indices"], .05, context_index=index,
        expected_two_plane_schedule_configuration=state.expected)


@pytest.mark.parametrize("mode", ("v1", "v2"))
@pytest.mark.parametrize("version", ("v12", "v13"))
def test_byte_nonempty_pair_publication_binds_its_requested_context_profile(adapter, modules, mode, version):
    helper = _byte_pairs(adapter, modules, mode)
    for index in (0, 1):
        assert _bind(adapter, modules, index)["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING"
    result = adapter.run(version, mode)
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE", result
    assert len(result["registration_array_binding"]) == 2
    assert helper["budgets"]["max_ordered_cad_matches_per_scan_pair"] == (4 if mode == "v1" else 256)
    assert len(adapter.measurement_calls) == len(adapter.validation_calls) == 1
    assert result["observed_bend_count"] is None and result["registration"]["transform_scan_to_cad"] is None


@pytest.mark.parametrize("mode", ("v1", "v2"))
@pytest.mark.parametrize("fault", ("schema", "missing_budget", "nonquota_budget", "float_budget", "boolean_budget", "other_mode_budget"))
def test_direct_byte_binding_refuses_profile_mismatch_for_nonempty_pairs(adapter, modules, mode, fault):
    helper = _byte_pairs(adapter, modules, mode)
    _bind(adapter, modules)
    if fault == "schema": helper["schema_version"] = "two_plane_context_v0"
    elif fault == "missing_budget": helper["budgets"].pop("max_report_bytes")
    elif fault == "nonquota_budget": helper["budgets"]["max_index_sort_work"] -= 1
    elif fault == "float_budget": helper["budgets"]["max_candidate_slots"] = 8192.
    elif fault == "boolean_budget": helper["budgets"]["chunk_rows"] = True
    else: helper["budgets"]["max_ordered_cad_matches_per_scan_pair"] = 256 if mode == "v1" else 4
    reason = "SCHEMA_MISMATCH" if fault == "schema" else "BUDGET_PROFILE_MISMATCH"
    with pytest.raises(ValueError, match="TWO_PLANE_CONTEXT_"+reason):
        _bind(adapter, modules)
    assert adapter.measurement_calls == adapter.validation_calls == []


@pytest.mark.parametrize("case", ("empty_v1", "empty_v2", "disabled_v1"))
@pytest.mark.parametrize("version", ("v12", "v13"))
def test_byte_empty_and_disabled_lanes_do_not_require_a_complete_helper_profile(adapter, modules, monkeypatch, case, version):
    mode = "v2" if case == "empty_v2" else "v1"
    adapter.install(mode)
    initialization = adapter.registration["evidence"]["initialization"]
    initialization["two_plane_context"] = dict(status="NOT_CALLED_EMPTY_DECLARED_SCHEDULE",
        usable_proposals=False, accepted_registration=False)
    options = {}
    if case == "disabled_v1":
        initialization["joint_proposal_accounting"]["pair_lane_enabled"] = False
        initialization["two_plane_schedule"]["status"] = "DISABLED_BY_EXPLICIT_POLICY"
        initialization["two_plane_context"] = None
        options["two_plane_enabled"] = False
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("profile")
        raise AssertionError("NO_CONTEXT_CONSUMER_FOR_EMPTY_OR_DISABLED_LANE")
    monkeypatch.setattr(modules.initialize, "_validate_two_plane_context_configuration", forbidden)
    result = adapter.run(version, mode, **options)
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE", result
    assert calls == [] and len(result["registration_array_binding"]) == 2
    assert len(adapter.measurement_calls) == len(adapter.validation_calls) == 1


def _assert_whole_byte_refusal(state, result):
    assert result["status"] == "UNAVAILABLE" and result["phase"] == "WHOLE_PART_REGISTRATION", result
    assert result["measurements"] is None and result["predictions"] == []
    assert "registration_array_binding" not in result
    assert result["registration"]["status"] == "UNAVAILABLE"
    evidence = result["registration"]["evidence"]
    assert evidence["measurement_context_indices"] == []
    assert evidence["attempted_measurement_context_indices"] == [0, 1]
    assert state.measurement_calls == state.validation_calls == []


@pytest.mark.parametrize("version", ("v12", "v13"))
def test_second_context_capacity_mismatch_discards_first_bound_context(adapter, modules, version):
    helper = _byte_pairs(adapter, modules, "v2")
    def damage():
        helper["scheduling_capacity"]["maximum_distinct_scan_pairs"] = 64.
    adapter.damage_second = damage
    result = adapter.run(version)
    assert [kwargs["context_index"] for _, kwargs in adapter.binding_calls] == [0, 1]
    _assert_whole_byte_refusal(adapter, result)


@pytest.mark.parametrize("version", ("v12", "v13"))
@pytest.mark.parametrize("publication", ("evidence", "initialization"))
def test_nonempty_context_cannot_override_mismatched_requested_schedule_publication(adapter, modules, version, publication):
    _byte_pairs(adapter, modules, "v2")
    evidence = adapter.registration["evidence"]
    target = evidence if publication == "evidence" else evidence["initialization"]
    target["two_plane_schedule_configuration"]["budgets"]["max_selected_scan_pairs"] -= 1
    result = adapter.run(version)
    _assert_whole_byte_refusal(adapter, result)


@pytest.mark.parametrize("version", ("v12", "v13"))
@pytest.mark.parametrize("marker", ("schema", "capacity_none"))
def test_unrequested_context_v2_marker_alone_refuses_and_clears_all_contexts(adapter, modules, version, marker):
    adapter.install("v1")
    helper = _profile(modules, "v1")
    if marker == "schema": helper["schema_version"] = "two_plane_context_v2"
    else: helper["scheduling_capacity"] = None
    initialization = adapter.registration["evidence"]["initialization"]
    initialization["two_plane_context"] = helper
    assert initialization["joint_proposal_accounting"]["ordered_pair_matches"] == 0
    assert modules.workflow_v12._has_two_plane_schedule_v2_publication(adapter.registration["evidence"]) is True
    with pytest.raises(ValueError, match="UNREQUESTED_V2_TWO_PLANE_SCHEDULE_PUBLICATION"):
        modules.workflow_v12._two_plane_schedule_binding(adapter.registration, None)
    result = adapter.run(version, "v1")
    _assert_whole_byte_refusal(adapter, result)
