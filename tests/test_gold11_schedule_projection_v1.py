"""Invented-only wrapper tests; prospective, not executed by source review.

Exact repository import closure: this file, tests/__init__.py, the existing
test_gold11_contact_scoring_v2.py (ordinary constructors only; no fixture calls),
scripts/gold11_schedule_projection_v1.py, scripts/gold11_contact_scoring_v2.py,
scripts/gold11_mesh_scoring_v1.py, scripts/evaluate_f1_against_manual_gold.py,
scripts/f1_physical_evaluation_contract_v2.py. The scripts package is inert.
Existing NumPy/SciPy enter through the qualified matcher, never backend/native
geometry. Every payload and digest below is invented; no result/file is read.
"""

import copy
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

from .test_gold11_contact_scoring_v2 import _configuration, _refusal, _wire, _workflow


ROOT = Path(__file__).resolve().parents[1]
BREADTH = dict(max_scan_planes=128, max_cad_planes=2048, max_scan_pair_records=8128,
    max_selected_scan_pairs=64, max_scan_selection_tests=520192, max_cad_pair_records=2096128,
    max_cad_sort_work=44018688, max_window_pair_tests=134152192,
    max_compatible_layout_bytes=134217728, max_compatible_layouts=64,
    max_working_bytes=268435456, max_report_bytes=33554432, chunk_pair_rows=65536)
FULL = dict(BREADTH, max_residual_priority_work=384, max_residual_pair_visits=64,
    max_residual_breadth_position_tests=128, max_residual_rank_selections=128,
    max_residual_skip_position_tests=512, max_residual_index_lookups=128,
    max_residual_cache_reference_records=64, max_residual_cache_reference_bytes=32768,
    max_residual_retained_bytes=1048576, max_residual_report_records=512,
    max_residual_report_reserved_bytes=1048576, max_final_ledger_tests=320)
INITIALIZATION = dict(max_hypothesis_records=8192, max_prefilter_unsigned_assignments=1024,
    max_cached_cad_faces=2048, max_indexed_face_triangles=500_000,
    max_prefilter_query_calls=32768, max_prefilter_query_rows=64_000_000,
    chunk_rows=65536, max_node_visits=512_000_000, max_triangle_pairs=256_000_000,
    max_certificate_pairs=1_024_000_000, max_normal_pairs=32_000_000,
    max_work_units=4_000_000_000, max_face_build_nodes=131071,
    max_face_build_triangle_visits=10_000_000, max_refined_start_groups=48,
    max_report_bytes=100_663_296)


@pytest.fixture(scope="module")
def wrapper():
    name = "_invented_schedule_projection_v1"
    package = ModuleType(name)
    package.__path__ = [str(ROOT / "scripts")]
    prior = {k: v for k, v in sys.modules.items() if k == name or k.startswith(name + ".")}
    sys.modules[name] = package
    try:
        yield importlib.import_module(name + ".gold11_schedule_projection_v1")
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        sys.modules.update(prior)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _request(mode="v2"):
    return dict(mode=mode, budgets=dict(FULL if mode == "v2" else BREADTH),
        initialization_budgets=dict(INITIALIZATION))


def _args(blob, *, schedule="v2", scan="v7"):
    return dict(expected_result_sha256=_sha(blob), expected_scan_plane_mode=scan,
        expected_scan_plane_configuration=_configuration() if scan == "composite_v1" else None,
        expected_schedule_request=_request(schedule))


def _schedule(mode="v2"):
    row = dict(schema_version="two_plane_schedule_" + mode, status="COMPLETE",
        selected_policy_evaluation_complete=True, accepted_registration=False,
        production_use_allowed=False, finite_support_match_verified=False,
        budgets=dict(FULL if mode == "v2" else BREADTH),
        input_digests=dict(scan_bank_equations_sha256=_sha(b"invented-scan-equations"),
            cad_bank_equations_sha256=_sha(b"invented-cad-equations")),
        scheduled_matches=[], usable_schedule=False)
    if mode == "v2":
        row.update(breadth_ledger=_schedule("v1"), residual_windows=[], residual_statistics={})
    return row


def _payload(*, schedule="v2", scan="v7", count=1, finite=True):
    value = _workflow(count=count, mode=scan, finite=finite)
    policies = value["policies"]
    evidence = value["registration"]["evidence"]
    initialization = evidence["initialization"]
    policies.update(two_plane_enabled=True, initialization_budgets=dict(INITIALIZATION))
    initialization.update(budgets=dict(INITIALIZATION), two_plane_schedule=_schedule(schedule))
    evidence["measurement_context_indices"] = [0, 1]
    value["registration_array_binding"] = [{}, {}]
    if schedule == "v2":
        config = dict(mode="v2", budgets=dict(FULL))
        for parent in (policies, evidence, initialization):
            parent["two_plane_schedule_configuration"] = copy.deepcopy(config)
        for binding in value["registration_array_binding"]:
            binding.update(two_plane_schedule_mode="v2",
                two_plane_schedule_configuration_sha256=_sha(_wire(config)),
                two_plane_schedule_input_digests=copy.deepcopy(initialization["two_plane_schedule"]["input_digests"]))
    return value


def _project(wrapper, payload, **kwargs):
    blob = _wire(payload)
    args = _args(blob, **kwargs)
    return blob, args, wrapper.project_gold11_schedule_workflow_v1(blob, **args)


def _revalidate(wrapper, blob, args, projection):
    stored = _wire(projection)
    return wrapper.revalidate_gold11_schedule_projection_v1(stored,
        expected_projection_sha256=_sha(stored), result_bytes=blob, **args)


def _reject(wrapper, payload, **kwargs):
    blob, args, projection = _project(wrapper, payload, **kwargs)
    assert projection["disposition"] == "REJECTED"
    assert projection["reason"] and projection["contact_projection"] is None
    assert projection["schedule_evidence"] is None
    verified = _revalidate(wrapper, blob, args, projection)
    assert set(verified["scorer_input"]) == {"status", "phase", "reasons", "predictions",
        "observed_bend_count", "production_use_allowed"}
    assert verified["scorer_input"]["status"] == "UNAVAILABLE"
    assert verified["scorer_input"]["phase"] == "SUPERVISOR"
    assert verified["scorer_input"]["predictions"] == []
    assert verified["scorer_input"]["reasons"][0].startswith("RUNNER_SCHEDULE_BINDING_REJECTED:")
    return verified


def test_independent_full_budget_literals_and_inert_import_closure(wrapper):
    assert wrapper.BREADTH_BUDGET_MAXIMA == BREADTH
    assert wrapper.SCHEDULE_BUDGET_MAXIMA == FULL
    assert wrapper.INITIALIZATION_BUDGET_MAXIMA == INITIALIZATION
    names = {key.rsplit(".", 1)[-1] for key in sys.modules
        if key.startswith("_invented_schedule_projection_v1.")}
    assert names == {"gold11_schedule_projection_v1", "gold11_contact_scoring_v2",
        "gold11_mesh_scoring_v1", "evaluate_f1_against_manual_gold", "f1_physical_evaluation_contract_v2"}


@pytest.mark.parametrize("schedule,scan", [("v1", "v7"), ("v2", "v7"),
    ("v1", "composite_v1"), ("v2", "composite_v1")])
def test_complete_binding_preserves_exact_inner_projection(wrapper, schedule, scan):
    blob, args, result = _project(wrapper, _payload(schedule=schedule, scan=scan), schedule=schedule, scan=scan)
    assert result["disposition"] == "ADMITTED"
    assert result["schedule_evidence"]["status"] == "COMPLETED_CONFIGURATION_BOUND"
    assert result["schedule_evidence"]["verified_array_binding_count"] == 2
    inner = wrapper._contact.project_gold11_contact_workflow_v2(blob,
        **{key: value for key, value in args.items() if key != "expected_schedule_request"})
    assert _wire(result["contact_projection"]) == _wire(inner)
    assert _wire(_revalidate(wrapper, blob, args, result)["scorer_input"]) == _wire(inner)


def test_inner_receives_same_original_bytes_object(wrapper, monkeypatch):
    blob = b" \n" + _wire(_payload()) + b"\n"
    original = wrapper._contact.project_gold11_contact_workflow_v2
    seen = []
    def spy(actual, **kwargs):
        seen.append(actual)
        return original(actual, **kwargs)
    monkeypatch.setattr(wrapper._contact, "project_gold11_contact_workflow_v2", spy)
    result = wrapper.project_gold11_schedule_workflow_v1(blob, **_args(blob))
    assert result["disposition"] == "ADMITTED" and len(seen) == 1 and seen[0] is blob


@pytest.mark.parametrize("change", ["mode", "extra", "missing_schedule", "schedule_bool", "schedule_float",
    "schedule_negative", "schedule_over", "zero_chunk", "missing_initialization", "initialization_bool",
    "initialization_float", "initialization_over", "zero_initialization_chunk", "custom_v1"])
def test_invalid_independent_requests_raise_before_inference_attribution(wrapper, change):
    request = _request("v1" if change == "custom_v1" else "v2")
    if change == "mode": request["mode"] = True
    elif change == "extra": request["extra"] = 0
    elif change == "missing_schedule": del request["budgets"]["max_scan_planes"]
    elif change == "schedule_bool": request["budgets"]["max_scan_planes"] = True
    elif change == "schedule_float": request["budgets"]["max_scan_planes"] = 128.
    elif change == "schedule_negative": request["budgets"]["max_scan_planes"] = -1
    elif change == "schedule_over": request["budgets"]["max_scan_planes"] = 129
    elif change == "zero_chunk": request["budgets"]["chunk_pair_rows"] = 0
    elif change == "missing_initialization": del request["initialization_budgets"]["max_hypothesis_records"]
    elif change == "initialization_bool": request["initialization_budgets"]["max_hypothesis_records"] = True
    elif change == "initialization_float": request["initialization_budgets"]["max_hypothesis_records"] = 8192.
    elif change == "initialization_over": request["initialization_budgets"]["max_hypothesis_records"] = 8193
    elif change == "zero_initialization_chunk": request["initialization_budgets"]["chunk_rows"] = 0
    else: request["budgets"]["max_scan_planes"] -= 1
    with pytest.raises(ValueError):
        wrapper.project_gold11_schedule_workflow_v1(b"{}", expected_result_sha256=_sha(b"{}"),
            expected_scan_plane_mode="v7", expected_schedule_request=request)


@pytest.mark.parametrize("location", [0, 1, 2])
@pytest.mark.parametrize("change", ["missing", "null", "bool", "extra"])
def test_each_completed_v2_configuration_location_is_mandatory_and_typed(wrapper, location, change):
    value = _payload()
    parent = [value["policies"], value["registration"]["evidence"], value["registration"]["evidence"]["initialization"]][location]
    if change == "missing": del parent["two_plane_schedule_configuration"]
    elif change == "null": parent["two_plane_schedule_configuration"] = None
    elif change == "bool": parent["two_plane_schedule_configuration"]["budgets"]["max_scan_planes"] = True
    else: parent["two_plane_schedule_configuration"]["extra"] = 0
    _reject(wrapper, value)


def test_all_returned_configuration_copies_cannot_change_the_independent_request(wrapper):
    value = _payload()
    for parent in (value["policies"], value["registration"]["evidence"], value["registration"]["evidence"]["initialization"]):
        parent["two_plane_schedule_configuration"]["budgets"]["max_residual_pair_visits"] -= 1
    _reject(wrapper, value)


@pytest.mark.parametrize("location", ["policies", "initialization"])
@pytest.mark.parametrize("change", ["missing", "bool", "float", "extra", "coherent_lower"])
def test_full_initialization_request_is_bound_at_both_locations(wrapper, location, change):
    value = _payload()
    parent, key = (value["policies"], "initialization_budgets") if location == "policies" else (value["registration"]["evidence"]["initialization"], "budgets")
    if change == "missing": del parent[key]
    elif change == "bool": parent[key]["max_hypothesis_records"] = True
    elif change == "float": parent[key]["max_hypothesis_records"] = 8192.
    elif change == "extra": parent[key]["extra"] = 0
    else:
        value["policies"]["initialization_budgets"]["max_hypothesis_records"] = 4096
        value["registration"]["evidence"]["initialization"]["budgets"]["max_hypothesis_records"] = 4096
    _reject(wrapper, value)


@pytest.mark.parametrize("change", ["missing", "schema", "status", "complete_flag", "accepted", "production",
    "support", "budget", "digest", "breadth_missing", "breadth_budget", "breadth_digest"])
def test_schedule_and_breadth_metadata_cannot_contradict_bound_request(wrapper, change):
    value = _payload()
    init = value["registration"]["evidence"]["initialization"]
    row = init["two_plane_schedule"]
    if change == "missing": del init["two_plane_schedule"]
    elif change == "schema": row["schema_version"] = "two_plane_schedule_v1"
    elif change == "status": row["status"] = "UNAVAILABLE"
    elif change == "complete_flag": row["selected_policy_evaluation_complete"] = 1
    elif change == "accepted": row["accepted_registration"] = True
    elif change == "production": row["production_use_allowed"] = True
    elif change == "support": row["finite_support_match_verified"] = True
    elif change == "budget": row["budgets"]["max_scan_planes"] = 127
    elif change == "digest": row["input_digests"]["scan_bank_equations_sha256"] = "no"
    elif change == "breadth_missing": row["breadth_ledger"] = None
    elif change == "breadth_budget": row["breadth_ledger"]["budgets"]["max_scan_planes"] = 127
    else: row["breadth_ledger"]["input_digests"]["scan_bank_equations_sha256"] = _sha(b"different")
    _reject(wrapper, value)


@pytest.mark.parametrize("change", ["missing", "short", "long", "nonlist", "late_null", "late_mode",
    "late_sha", "late_digest", "late_extra", "duplicate_context", "bool_context", "late_marker"])
def test_every_context_binding_is_checked_without_successful_prefix(wrapper, change):
    value = _payload(count=2)
    rows = value["registration_array_binding"]
    if change == "missing": del value["registration_array_binding"]
    elif change == "short": rows.pop()
    elif change == "long": rows.append(copy.deepcopy(rows[0]))
    elif change == "nonlist": value["registration_array_binding"] = {}
    elif change == "late_null": rows[-1] = None
    elif change == "late_mode": rows[-1]["two_plane_schedule_mode"] = "v1"
    elif change == "late_sha": rows[-1]["two_plane_schedule_configuration_sha256"] = _sha(b"wrong")
    elif change == "late_digest": rows[-1]["two_plane_schedule_input_digests"]["cad_bank_equations_sha256"] = _sha(b"wrong")
    elif change == "late_extra": rows[-1]["two_plane_schedule_extra"] = None
    elif change == "duplicate_context": value["registration"]["evidence"]["measurement_context_indices"] = [0, 0]
    elif change == "bool_context": value["registration"]["evidence"]["measurement_context_indices"] = [0, True]
    else: rows[-1]["residual_windows"] = []
    _reject(wrapper, value)


@pytest.mark.parametrize("location", ["policy_config", "evidence_config", "initialization_config", "schema",
    "breadth", "residual", "statistics", "array_mode", "array_sha", "array_digest", "policy_marker"])
def test_v1_rejects_unrequested_v2_markers_including_nulls(wrapper, location):
    value = _payload(schedule="v1")
    evidence = value["registration"]["evidence"]
    init = evidence["initialization"]
    if location.endswith("_config"):
        parent = {"policy_config": value["policies"], "evidence_config": evidence, "initialization_config": init}[location]
        parent["two_plane_schedule_configuration"] = None
    elif location == "schema": init["two_plane_schedule"]["schema_version"] = "two_plane_schedule_v2"
    elif location == "breadth": init["two_plane_schedule"]["breadth_ledger"] = None
    elif location == "residual": init["two_plane_schedule"]["residual_windows"] = []
    elif location == "statistics": init["two_plane_schedule"]["residual_statistics"] = {}
    elif location == "policy_marker": value["policies"]["residual_windows"] = []
    else:
        key = {"array_mode": "two_plane_schedule_mode", "array_sha": "two_plane_schedule_configuration_sha256",
            "array_digest": "two_plane_schedule_input_digests"}[location]
        value["registration_array_binding"][-1][key] = None
    _reject(wrapper, value, schedule="v1")


@pytest.mark.parametrize("mode", ["v1", "v2"])
def test_early_refusal_without_metadata_is_not_fabricated_complete_binding(wrapper, mode):
    blob, args, result = _project(wrapper, _refusal(), schedule=mode)
    assert result["disposition"] == "ADMITTED"
    assert result["schedule_evidence"]["status"] == "PRESENT_METADATA_CONSISTENT_NOT_COMPLETE_WORKFLOW"
    assert result["schedule_evidence"]["schedule"] is None
    assert result["schedule_evidence"]["verified_array_binding_count"] is None
    assert _revalidate(wrapper, blob, args, result)["scorer_input"]["disposition"] == "REFUSED"


@pytest.mark.parametrize("bad", [False, True])
def test_early_refusal_checks_every_present_configuration(wrapper, bad):
    value = _refusal("INPUT_PREPARATION")
    value["policies"] = dict(two_plane_enabled=True, initialization_budgets=dict(INITIALIZATION),
        two_plane_schedule_configuration=dict(mode="v2", budgets=dict(FULL)))
    if bad:
        value["policies"]["two_plane_schedule_configuration"]["budgets"]["max_scan_planes"] = 127
        _reject(wrapper, value)
    else:
        assert _project(wrapper, value)[2]["disposition"] == "ADMITTED"


@pytest.mark.parametrize("prefix", [False, True])
def test_late_schedule_refusal_may_retain_breadth_evidence_but_no_usable_matches(wrapper, prefix):
    value = _refusal("WHOLE_PART_REGISTRATION")
    schedule = _schedule()
    schedule.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False,
        scheduled_matches=[{}] if prefix else [], usable_schedule=prefix)
    value["registration"] = dict(evidence=dict(measurement_context_indices=[],
        initialization=dict(two_plane_schedule=schedule)))
    if prefix: _reject(wrapper, value)
    else:
        result = _project(wrapper, value)[2]
        assert result["disposition"] == "ADMITTED"
        assert result["schedule_evidence"]["schedule"]["breadth_status"] == "COMPLETE"


@pytest.mark.parametrize("change", ["empty", "scan_only", "cad_only", "unknown_key", "bad_hex", "conflict"])
def test_partial_refusal_digest_metadata_is_not_promoted_to_complete_or_silently_corrected(wrapper, change):
    value = _refusal("WHOLE_PART_REGISTRATION")
    row = _schedule()
    row.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False)
    if change == "empty": row["input_digests"] = {}
    elif change == "scan_only": del row["input_digests"]["cad_bank_equations_sha256"]
    elif change == "cad_only": del row["input_digests"]["scan_bank_equations_sha256"]
    elif change == "unknown_key": row["input_digests"]["other"] = _sha(b"other")
    elif change == "bad_hex": row["input_digests"]["scan_bank_equations_sha256"] = "bad"
    else: row["input_digests"]["scan_bank_equations_sha256"] = _sha(b"different")
    value["registration"] = dict(evidence=dict(initialization=dict(two_plane_schedule=row)))
    if change in ("unknown_key", "bad_hex", "conflict"):
        _reject(wrapper, value)
    else:
        result = _project(wrapper, value)[2]
        assert result["disposition"] == "ADMITTED"
        assert result["schedule_evidence"]["status"] == "PRESENT_METADATA_CONSISTENT_NOT_COMPLETE_WORKFLOW"


@pytest.mark.parametrize("count,finite,expected", [(1, False, "COMPLETE_NO_LOCALIZED_CARRIERS"), (0, True, "REFUSED")])
def test_completed_no_carrier_and_zero_measurements_remain_distinct(wrapper, count, finite, expected):
    blob, args, result = _project(wrapper, _payload(count=count, finite=finite))
    assert result["disposition"] == "ADMITTED"
    inner = _revalidate(wrapper, blob, args, result)["scorer_input"]
    assert inner["disposition"] == expected and inner["predictions"] == []


@pytest.mark.parametrize("blob", [b"{}", b'{"schema":1,"schema":2}', b'{"unused":NaN}',
    b'{"unused":1e999}', b"\xff", b"[", b""])
def test_malformed_original_bytes_never_leave_an_inner_projection(wrapper, blob):
    args = _args(blob)
    result = wrapper.project_gold11_schedule_workflow_v1(blob, **args)
    assert result["disposition"] == "REJECTED" and result["contact_projection"] is None
    assert result["schedule_evidence"] is None
    assert _revalidate(wrapper, blob, args, result)["scorer_input"]["predictions"] == []


@pytest.mark.parametrize("change", ["wrong_hash", "one_byte_budget"])
def test_result_hash_and_byte_cap_fail_closed(wrapper, change):
    blob = _wire(_payload())
    args = _args(blob)
    if change == "wrong_hash": args["expected_result_sha256"] = _sha(b"not-the-result")
    else: args["max_result_bytes"] = len(blob) - 1
    result = wrapper.project_gold11_schedule_workflow_v1(blob, **args)
    assert result["disposition"] == "REJECTED" and result["contact_projection"] is None
    assert _revalidate(wrapper, blob, args, result)["scorer_input"]["predictions"] == []


@pytest.mark.parametrize("change", ["configuration", "context_count", "inner_endpoint", "result_binding",
    "numeric_type", "extra", "reason"])
def test_canonical_parent_reconstruction_rejects_self_rehashed_wrapper_tampering(wrapper, change):
    blob, args, result = _project(wrapper, _payload())
    if change == "configuration": result["requested_schedule"]["budgets"]["max_scan_planes"] = 127
    elif change == "context_count": result["schedule_evidence"]["verified_array_binding_count"] = 1
    elif change == "inner_endpoint": result["contact_projection"]["predictions"][0]["line_start"][0] = 1
    elif change == "result_binding": result["result_binding"]["supplied_sha256"] = _sha(b"other")
    elif change == "numeric_type": result["result_binding"]["supplied_bytes"] = float(result["result_binding"]["supplied_bytes"])
    elif change == "extra": result["extra"] = None
    else: result["reason"] = "invented"
    with pytest.raises(ValueError, match="RECONSTRUCTION_MISMATCH"):
        _revalidate(wrapper, blob, args, result)


def test_same_semantic_result_reserialization_cannot_reuse_old_projection_binding(wrapper):
    blob, args, result = _project(wrapper, _payload())
    changed = b" " + blob
    args["expected_result_sha256"] = _sha(changed)
    with pytest.raises(ValueError, match="RECONSTRUCTION_MISMATCH"):
        _revalidate(wrapper, changed, args, result)


def test_projection_whitespace_is_canonical_but_its_exact_receipt_sha_is_still_required(wrapper):
    blob, args, result = _project(wrapper, _payload())
    plain = _wire(result)
    spaced = b" \n" + plain + b"\n"
    with pytest.raises(ValueError, match="PROJECTION_SHA256_MISMATCH"):
        wrapper.revalidate_gold11_schedule_projection_v1(spaced,
            expected_projection_sha256=_sha(plain), result_bytes=blob, **args)
    verified = wrapper.revalidate_gold11_schedule_projection_v1(spaced,
        expected_projection_sha256=_sha(spaced), result_bytes=blob, **args)
    assert _wire(verified["projection"]) == plain


@pytest.mark.parametrize("extra", [0, 1])
def test_exact_retained_wrapper_cap_precedes_whitespace_decoding(wrapper, monkeypatch, extra):
    blob = _wire(_payload())
    args = _args(blob)
    args["max_wrapper_bytes"] = 32768
    projection = wrapper.project_gold11_schedule_workflow_v1(blob, **args)
    assert projection["disposition"] == "ADMITTED"
    plain = _wire(projection)
    assert len(plain) < 32768
    padded = plain + b" " * (32768 + extra - len(plain))
    if extra:
        calls = []
        def forbidden(*unused, **kwargs):
            calls.append(True)
            raise AssertionError("OVER_CAP_WRAPPER_MUST_NOT_DECODE")
        monkeypatch.setattr(wrapper._contact, "_decode", forbidden)
        with pytest.raises(ValueError, match="RETAINED_PROJECTION_BYTE_BUDGET"):
            wrapper.revalidate_gold11_schedule_projection_v1(padded,
                expected_projection_sha256=_sha(padded), result_bytes=blob, **args)
        assert calls == []
    else:
        result = wrapper.revalidate_gold11_schedule_projection_v1(padded,
            expected_projection_sha256=_sha(padded), result_bytes=blob, **args)
        assert result["projection"] == projection


def test_wrapper_byte_exhaustion_discards_the_entire_successful_inner_result(wrapper):
    blob = _wire(_payload(count=128))
    args = _args(blob)
    args["max_wrapper_bytes"] = 32768
    result = wrapper.project_gold11_schedule_workflow_v1(blob, **args)
    assert result["disposition"] == "REJECTED" and result["reason"] == "SCHEDULE_WRAPPER_BYTE_BUDGET"
    assert result["contact_projection"] is None and result["schedule_evidence"] is None
    assert len(_wire(result)) < 32768


def test_runner_rejections_preserve_all_references_in_the_unchanged_paired_scorer(wrapper):
    cases = ["invented-case-" + str(index) for index in range(11)]
    families = {case: "invented-family-" + str(index % 6) for index, case in enumerate(cases)}
    references = {case: [dict(bend_id="reference-" + str(j), line_start=[20*j, 0, 0],
        line_end=[20*j, 0, 10]) for j in range(7 if index < 5 else 6)]
        for index, case in enumerate(cases)}
    by_arm = {}
    requests = {}
    for arm, mode in (("control", "v7"), ("challenger", "composite_v1")):
        value = _payload(scan=mode)
        value["registration_array_binding"][-1]["two_plane_schedule_configuration_sha256"] = _sha(b"wrong")
        refused = _reject(wrapper, value, scan=mode)["scorer_input"]
        by_arm[arm] = {case: copy.deepcopy(refused) for case in cases}
        requests[arm] = dict(mode=mode, configuration=_configuration() if mode == "composite_v1" else None)
    scored = wrapper._contact.score_gold11_contact_workflows_v2(reference_records_by_case=references,
        control_results_by_case=by_arm["control"], challenger_results_by_case=by_arm["challenger"],
        case_family_by_id=families, scan_plane_requests_by_arm=requests)
    assert (scored["case_count"], scored["family_count"], scored["reference_line_count"]) == (11, 6, 71)
    for arm in ("control", "challenger"):
        row = scored[arm]
        assert row["geometry"]["true_positive"] == row["geometry"]["false_positive"] == 0
        assert row["geometry"]["false_negative"] == 71
        assert row["geometry"]["precision"] is None and row["geometry"]["f1"] is None
        assert row["geometry"]["recall"] == 0
        assert row["refused_case_count"] == 11 and row["malformed_case_count"] == 0


@pytest.mark.parametrize("key,value", [("max_result_bytes", 0), ("max_result_bytes", True),
    ("max_wrapper_bytes", 32767), ("max_wrapper_bytes", 16*1024**2+1)])
def test_invalid_caller_limits_raise(wrapper, key, value):
    blob = _wire(_payload())
    args = _args(blob)
    args[key] = value
    with pytest.raises(ValueError, match="CALLER_WRAPPER_LIMITS"):
        wrapper.project_gold11_schedule_workflow_v1(blob, **args)
