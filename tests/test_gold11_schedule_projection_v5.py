"""Invented-byte context-V4 binding tests for schedule projection V3; no backend imports or data.

Derived from the V2 selection with V15/V14/V4 identities, the nine context-V4
declarations, the 512M coordinate maximum, and new registration-implementation
binding controls at the workflow, evidence and array-binding levels.

The V1 test module supplies ordinary structural constructors, never fixtures.
All RESULT/projection fixtures are capped below 100 KiB. Static backend reads
are restricted to three named context source files; no backend is executed.
Actual projection and exact-byte reconstruction run only in a separately
authorized bounded gate. No physical accuracy or execution proof is asserted.
"""

import ast
import copy
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

from .test_gold11_schedule_projection_v1 import _args as _old_args, _schedule as _v1_schedule, INITIALIZATION as _V1_INITIALIZATION, FULL as _V1_FULL
from .test_gold11_contact_scoring_v4 import _workflow as _workflow_v3


def _old_payload(*, schedule="v2", scan="v7", count=1, finite=True):
    """The V1 projection test's payload builder on the V3 contact grammar (cover V3 / context V2)."""
    value = _workflow_v3(count=count, mode=scan, finite=finite)
    policies = value["policies"]
    evidence = value["registration"]["evidence"]
    initialization = evidence["initialization"]
    policies.update(two_plane_enabled=True, initialization_budgets=dict(_V1_INITIALIZATION))
    initialization.update(budgets=dict(_V1_INITIALIZATION), two_plane_schedule=_v1_schedule(schedule))
    evidence["measurement_context_indices"] = [0, 1]
    value["registration_array_binding"] = [{}, {}]
    if schedule == "v2":
        config = dict(mode="v2", budgets=dict(_V1_FULL))
        for parent in (policies, evidence, initialization):
            parent["two_plane_schedule_configuration"] = copy.deepcopy(config)
        for binding in value["registration_array_binding"]:
            binding.update(two_plane_schedule_mode="v2",
                two_plane_schedule_configuration_sha256=hashlib.sha256(_wire(config)).hexdigest(),
                two_plane_schedule_input_digests=copy.deepcopy(initialization["two_plane_schedule"]["input_digests"]))
    return value
from .test_gold11_contact_scoring_v2 import _refusal, _wire


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CAP = 100 * 1024
INIT = "CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1"
FLOW = "CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1"
CONTEXT = "RAW_EXTREMA_KD_PARTITION_CONTEXT_V4_STAGED_SCAN_PROJECTION"
REGISTRATION = "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1"
SCOPE = "BOTH_SUCCESSOR_ARMS_CONTEXT_V4_NOT_HISTORICAL_PAIRED_CONTROL"
BUDGETS = dict(max_scan_rows=1000000, max_scan_triangles=2000000,
    max_cad_triangles=500000, max_scan_planes=128, max_cad_planes=2048,
    max_scan_pairs=64, max_ordered_cad_matches_per_scan_pair=256,
    max_signed_families=1024, max_candidate_slots=8192,
    max_scan_row_visits=64000000, max_cad_vertex_visits=128000000,
    max_index_visits=64000000, max_index_sort_work=256000000,
    max_hash_bytes=1073741824, max_projection_cache_entries=6144,
    max_working_bytes=268435456, max_report_bytes=33554432, chunk_rows=65536,
    max_extrema_indexes=129, max_extrema_indexed_rows=4000000,
    max_extrema_node_visits=16000000, max_extrema_coordinate_visits=512000000,
    max_extrema_queries=3072)
DECLARATIONS = dict(
    scan_projection_arithmetic="STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1",
    scan_extrema_partition="RANK_PARTITION_KD_TREE_GLOBAL_HALVED_ROOT_SPAN_AXIS_SCHEDULE_STABLE_SORT_LEAF64_V3",
    scan_projection_bound_scope="ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
    cad_projection_arithmetic="UNCHANGED_V1_NUMPY_MATMUL_PROJECTION",
    scan_projection_chunk_scope="FIXED_EXTREMA_LEAF64_NOT_CHUNK_ROWS",
    chunk_rows_scope="INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY",
    extrema_guard_status="NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_V3_INVARIANTS",
    extrema_membership_scope="SUM_OF_DISTINCT_INDEX_SUPPORT_SIZES_INCLUDING_WHOLE_SCAN",
    extrema_memory_scope="SHARED_RAW_OWNER_PLUS_ALL_LIVE_INDEX_PAYLOAD_UNDER_PARENT_CAP_NOT_RSS")
PARENT_KEYS = ("initialization_implementation", "two_plane_context_configuration",
    "two_plane_context_configuration_sha256")
ARRAY_KEYS = ("initialization_implementation", "two_plane_context_mode",
    "two_plane_context_implementation", "two_plane_context_configuration_sha256",
    "registration_implementation")
MODES = [("v1", "v7"), ("v2", "v7"), ("v1", "composite_v1"), ("v2", "composite_v1")]


@pytest.fixture(scope="module")
def wrapper():
    name = "_invented_schedule_context_projection_v5"
    package = ModuleType(name)
    package.__path__ = [str(ROOT / "scripts")]
    prior = {key: value for key, value in sys.modules.items()
        if key == name or key.startswith(name + ".")}
    sys.modules[name] = package
    try:
        yield importlib.import_module(name + ".gold11_schedule_projection_v5")
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        sys.modules.update(prior)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _small(value):
    blob = _wire(value)
    assert 0 < len(blob) < FIXTURE_CAP
    return blob


def _request():
    return dict(mode="v4", implementation=CONTEXT, schema_version="two_plane_context_v4",
        budgets=dict(BUDGETS), **DECLARATIONS)


def _parents(value):
    evidence = value["registration"]["evidence"]
    return [value, value["policies"], evidence, evidence["initialization"]]


def _publish(value, request):
    digest = _sha(_small(request))
    for parent in _parents(value):
        parent.update(initialization_implementation=INIT,
            two_plane_context_configuration=copy.deepcopy(request),
            two_plane_context_configuration_sha256=digest)
    for binding in value["registration_array_binding"]:
        binding.update(initialization_implementation=INIT, two_plane_context_mode="v4",
            two_plane_context_implementation=CONTEXT, registration_implementation=REGISTRATION,
            two_plane_context_configuration_sha256=digest)
    value["registration_implementation"] = REGISTRATION
    value["registration"]["evidence"]["registration_implementation"] = REGISTRATION


def _payload(*, schedule="v2", scan="v7", empty=False, finite=True):
    value = _old_payload(schedule=schedule, scan=scan, count=2, finite=finite)
    value.update(implementation=FLOW, context_request_scope=SCOPE)
    evidence, initialization = _parents(value)[2:]
    evidence["measurement_context_indices"] = [0, 1, 2]
    value["registration_array_binding"].append(copy.deepcopy(value["registration_array_binding"][-1]))
    _publish(value, _request())
    ledger = initialization["two_plane_schedule"]
    ledger["scheduled_matches"] = [] if empty else [dict(schedule_index=index,
        scan_pair_id=0, cad_pair_index=0, unsigned_assignment_reservation=1,
        reserved_hypothesis_slots=32, scan_plane_ids=[0, 1],
        cad_plane_ids=[0, 1] if index == 0 else [1, 0]) for index in range(2)]
    ledger["usable_schedule"] = not empty
    initialization["joint_proposal_accounting"] = dict(pair_lane_enabled=True,
        ordered_pair_matches=0 if empty else 2, pair_slot_reservation=0 if empty else 64)
    if empty:
        helper = dict(status="NOT_CALLED_EMPTY_DECLARED_SCHEDULE",
            usable_proposals=False, accepted_registration=False)
    else:
        helper = dict(schema_version="two_plane_context_v4", status="COMPLETE",
            selected_policy_evaluation_complete=True, accepted_registration=False,
            production_use_allowed=False, finite_support_match_verified=False,
            usable_proposals=True, budgets=dict(BUDGETS), **DECLARATIONS)
    initialization["two_plane_context"] = helper
    return value


def _args(blob, *, schedule="v2", scan="v7"):
    return dict(_old_args(blob, schedule=schedule, scan=scan), expected_context_request=_request())


def _project(wrapper, payload, **modes):
    blob = _small(payload)
    args = _args(blob, **modes)
    projected = wrapper.project_gold11_schedule_workflow_v5(blob, **args)
    _small(projected)
    return blob, args, projected


def _reconstruct(wrapper, blob, args, projection):
    retained = _small(projection)
    return wrapper.revalidate_gold11_schedule_projection_v5(retained,
        expected_projection_sha256=_sha(retained), result_bytes=blob, **args)


def _cleared(wrapper, blob, args, projection):
    assert projection["disposition"] == "REJECTED" and projection["reason"]
    for key in ("contact_projection", "context_evidence", "schedule_evidence"):
        assert projection[key] is None
    scorer = _reconstruct(wrapper, blob, args, projection)["scorer_input"]
    assert set(scorer) == {"status", "phase", "reasons", "predictions",
        "observed_bend_count", "production_use_allowed"}
    assert scorer["status"] == "UNAVAILABLE" and scorer["phase"] == "SUPERVISOR"
    assert scorer["predictions"] == [] and scorer["observed_bend_count"] is None
    assert scorer["production_use_allowed"] is False
    assert scorer["reasons"][0].startswith("RUNNER_CONTEXT_BINDING_REJECTED:")


def _reject(wrapper, value, **modes):
    _cleared(wrapper, *_project(wrapper, value, **modes))


def test_independent_literals_static_backend_defaults_and_import_closure(wrapper):
    assert wrapper.CONTEXT_BUDGET_MAXIMA == BUDGETS
    assert wrapper.CONTEXT_DECLARATIONS == DECLARATIONS
    assert wrapper._context_request(_request()) == _request()
    defaults, maxima = {}, {}
    for version in (1, 2, 4):
        # Source-only AST literals, not import/eval of backend policy code.
        path = ROOT / "backend" / ("cad_guided_two_plane_context_v%d.py" % version)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
            and node.name == "TwoPlaneContextBudgetsV%d" % version)
        defaults.update({node.target.id: ast.literal_eval(node.value) for node in cls.body
            if isinstance(node, ast.AnnAssign)})
        if version in (1, 4):
            name = "_MAXIMA" if version == 1 else "_EXTRA_MAXIMA"
            node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
            maxima.update(dict(ast.literal_eval(node.value)))
        else:
            node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "_MAXIMA" for target in node.targets))
            expected = ast.parse("tuple((name, 256 if name == 'max_ordered_cad_matches_per_scan_pair' else maximum) for name, maximum in _v1._MAXIMA)", mode="eval").body
            assert ast.dump(node.value) == ast.dump(expected)
            maxima["max_ordered_cad_matches_per_scan_pair"] = 256
    assert defaults == maxima == BUDGETS
    names = {key.rsplit(".", 1)[-1] for key in sys.modules
        if key.startswith("_invented_schedule_context_projection_v5.")}
    assert names == {"gold11_schedule_projection_v5", "gold11_schedule_projection_v1",
        "gold11_contact_scoring_v4", "gold11_contact_scoring_v2", "gold11_mesh_scoring_v1",
        "evaluate_f1_against_manual_gold", "f1_physical_evaluation_contract_v2"}


@pytest.mark.parametrize("schedule,scan", MODES)
@pytest.mark.parametrize("empty", [False, True])
def test_complete_both_successor_arms_and_schedule_cross_controls(wrapper, schedule, scan, empty):
    blob, args, projected = _project(wrapper, _payload(schedule=schedule, scan=scan, empty=empty),
        schedule=schedule, scan=scan)
    assert projected["disposition"] == "ADMITTED"
    context = projected["context_evidence"]
    assert context["configuration_presence"] == dict(workflow=True, policies=True,
        registration=True, initialization=True)
    assert context["verified_array_binding_count"] == 3
    assert context["configuration_sha256"] == _sha(_small(_request()))
    inner = wrapper._contact3.project_gold11_contact_workflow_v4(blob,
        **{key: value for key, value in args.items()
            if key not in ("expected_schedule_request", "expected_context_request")})
    assert _small(projected["contact_projection"]) == _small(inner)
    assert projected["contact_projection"]["schema"] == "GOLD11_CONTACT_WORKFLOW_PROJECTION_V4"
    assert projected["contact_projection_implementation"] == wrapper._contact3.PROJECTION_SCHEMA
    stale = wrapper._contact.project_gold11_contact_workflow_v2(blob,
        **{key: value for key, value in args.items()
            if key not in ("expected_schedule_request", "expected_context_request")})
    assert stale["disposition"] == "MALFORMED_RESULT"  # frozen V2 grammar refuses the V9 payload; V4 no longer delegates to it
    assert _small(_reconstruct(wrapper, blob, args, projected)["scorer_input"]) == _small(inner)
    assert len(inner["predictions"]) == 2


def test_original_bytes_object_passes_through_both_layers(wrapper, monkeypatch):
    blob = b" \n" + _small(_payload()) + b"\n"
    assert len(blob) < FIXTURE_CAP
    original = wrapper._contact3.project_gold11_contact_workflow_v4
    seen = []
    def spy(actual, **kwargs):
        seen.append(actual)
        return original(actual, **kwargs)
    monkeypatch.setattr(wrapper._contact3, "project_gold11_contact_workflow_v4", spy)
    monkeypatch.setattr(wrapper._contact, "project_gold11_contact_workflow_v2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("frozen contact V2 projector must not be called")))
    projected = wrapper.project_gold11_schedule_workflow_v5(blob, **_args(blob))
    assert projected["disposition"] == "ADMITTED" and len(seen) == 1 and seen[0] is blob


@pytest.mark.parametrize("schedule,scan", MODES)
def test_early_refusal_does_not_fabricate_returned_configuration(wrapper, schedule, scan):
    value = dict(_refusal(), implementation=FLOW)
    blob, args, projected = _project(wrapper, value, schedule=schedule, scan=scan)
    assert projected["disposition"] == "ADMITTED"
    assert projected["context_evidence"]["configuration_presence"] == dict.fromkeys(
        ("workflow", "policies", "registration", "initialization"), False)
    assert projected["context_evidence"]["verified_array_binding_count"] is None
    assert projected["requested_context"] == _request()
    assert _reconstruct(wrapper, blob, args, projected)["scorer_input"]["disposition"] == "REFUSED"


@pytest.mark.parametrize("schedule,scan", MODES)
def test_real_unavailable_helper_may_omit_budgets_only_before_completion(wrapper, schedule, scan):
    value = _payload(schedule=schedule, scan=scan)
    value.update(_refusal("REGISTRATION"))
    init = _parents(value)[-1]
    helper = init["two_plane_context"]
    helper.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False,
        usable_proposals=False)
    del helper["budgets"]
    value["registration"]["evidence"]["measurement_context_indices"] = []
    value["registration_array_binding"] = []
    blob, args, projected = _project(wrapper, value, schedule=schedule, scan=scan)
    assert projected["disposition"] == "ADMITTED"
    assert _reconstruct(wrapper, blob, args, projected)["scorer_input"]["disposition"] == "REFUSED"


@pytest.mark.parametrize("schedule,scan", MODES)
def test_later_workflow_refusal_may_retain_completed_helper(wrapper, schedule, scan):
    value = _payload(schedule=schedule, scan=scan)
    value.update(_refusal("MEASUREMENT"))
    blob, args, projected = _project(wrapper, value, schedule=schedule, scan=scan)
    assert projected["disposition"] == "ADMITTED"
    assert projected["context_evidence"]["verified_array_binding_count"] == 3
    assert _reconstruct(wrapper, blob, args, projected)["scorer_input"]["disposition"] == "REFUSED"


@pytest.mark.parametrize("change", ["unknown_status", "missing_status", "selected_int",
    "selected_true", "usable_int", "usable_true", "accepted", "finite", "production", "authority_missing"])
def test_early_helper_cannot_claim_contradictory_state_or_authority(wrapper, change):
    value = _payload()
    value.update(_refusal("REGISTRATION"))
    helper = _parents(value)[-1]["two_plane_context"]
    helper.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False,
        usable_proposals=False)
    del helper["budgets"]
    value["registration"]["evidence"]["measurement_context_indices"] = []
    value["registration_array_binding"] = []
    if change == "unknown_status": helper["status"] = "INVENTED_DONE"
    elif change == "missing_status": del helper["status"]
    elif change == "selected_int": helper["selected_policy_evaluation_complete"] = 0
    elif change == "selected_true": helper["selected_policy_evaluation_complete"] = True
    elif change == "usable_int": helper["usable_proposals"] = 0
    elif change == "usable_true": helper["usable_proposals"] = True
    elif change == "accepted": helper["accepted_registration"] = True
    elif change == "finite": helper["finite_support_match_verified"] = True
    elif change == "production": helper["production_use_allowed"] = True
    else: del helper["accepted_registration"]
    _reject(wrapper, value)


@pytest.mark.parametrize("location", [0, 1, 2, 3])
@pytest.mark.parametrize("change", ["missing", "partial", "digest", "identity", "mode", "budget", "extra"])
def test_all_four_context_publications_clear_whole_result(wrapper, location, change):
    value = _payload()
    parent = _parents(value)[location]
    if change == "missing":
        for key in PARENT_KEYS: del parent[key]
    elif change == "partial": del parent["two_plane_context_configuration_sha256"]
    elif change == "digest": parent["two_plane_context_configuration_sha256"] = "0" * 64
    elif change == "identity": parent["initialization_implementation"] = "V13"
    elif change == "mode": parent["two_plane_context_configuration"]["mode"] = "v3"
    elif change == "budget": parent["two_plane_context_configuration"]["budgets"]["max_scan_rows"] -= 1
    else: parent["two_plane_context_unrequested"] = None
    _reject(wrapper, value)


@pytest.mark.parametrize("location", [0, 1, 2])
@pytest.mark.parametrize("key", ARRAY_KEYS)
@pytest.mark.parametrize("change", ["missing", "different"])
def test_first_middle_last_context_binding_every_key(wrapper, location, key, change):
    value = _payload(empty=True)
    binding = value["registration_array_binding"][location]
    if change == "missing": del binding[key]
    else: binding[key] = "wrong"
    _reject(wrapper, value)


@pytest.mark.parametrize("change", ["absent", "null", "zero", "short", "extra", "typed", "unknown"])
def test_binding_denominator_and_unknown_context_fields(wrapper, change):
    value = _payload()
    if change == "absent": del value["registration_array_binding"]
    elif change == "null": value["registration_array_binding"] = None
    elif change == "zero":
        value["registration_array_binding"] = []
        value["registration"]["evidence"]["measurement_context_indices"] = []
    elif change == "short": value["registration_array_binding"].pop()
    elif change == "extra": value["registration_array_binding"].append(copy.deepcopy(value["registration_array_binding"][-1]))
    elif change == "typed": value["registration_array_binding"][-1] = []
    else: value["registration_array_binding"][-1]["two_plane_context_unknown"] = None
    _reject(wrapper, value)


@pytest.mark.parametrize("key", list(BUDGETS))
@pytest.mark.parametrize("change", ["bool", "float", "over"])
def test_each_caller_budget_is_exact_integer_and_bounded(wrapper, key, change):
    request = _request()
    request["budgets"][key] = True if change == "bool" else float(BUDGETS[key]) if change == "float" else BUDGETS[key] + 1
    args = _args(b"{}")
    args["expected_context_request"] = request
    with pytest.raises(ValueError):
        wrapper.project_gold11_schedule_workflow_v5(b"{}", **args)


@pytest.mark.parametrize("change", ["mode", "identity", "schema", "missing", "extra", "budget_missing",
    "budget_extra", "negative", "chunk", "query_relation"])
def test_invalid_caller_context_has_no_case_refusal_attribution(wrapper, change):
    request = _request()
    if change == "mode": request["mode"] = "v3"
    elif change == "identity": request["implementation"] = "V3"
    elif change == "schema": request["schema_version"] = "two_plane_context_v3"
    elif change == "missing": del request["mode"]
    elif change == "extra": request["extra"] = 0
    elif change == "budget_missing": del request["budgets"]["max_scan_rows"]
    elif change == "budget_extra": request["budgets"]["extra"] = 0
    elif change == "negative": request["budgets"]["max_scan_rows"] = -1
    elif change == "chunk": request["budgets"]["chunk_rows"] = 2
    else: request["budgets"]["max_signed_families"] = 1023
    args = _args(b"{}")
    args["expected_context_request"] = request
    with pytest.raises(ValueError): wrapper.project_gold11_schedule_workflow_v5(b"{}", **args)


@pytest.mark.parametrize("key", list(DECLARATIONS))
@pytest.mark.parametrize("where", ["caller", "helper"])
def test_every_arithmetic_scope_declaration_is_closed(wrapper, key, where):
    if where == "caller":
        args = _args(b"{}")
        args["expected_context_request"][key] = "UNDECLARED_ARITHMETIC"
        with pytest.raises(ValueError): wrapper.project_gold11_schedule_workflow_v5(b"{}", **args)
    else:
        value = _payload()
        _parents(value)[-1]["two_plane_context"][key] = "UNDECLARED_ARITHMETIC"
        _reject(wrapper, value)


@pytest.mark.parametrize("change", ["missing", "unavailable", "budget_missing", "budget_float",
    "wrong_schema", "selected", "accepted", "finite", "production", "empty_schedule"])
def test_completed_nonempty_context_requires_real_complete_report(wrapper, change):
    value = _payload()
    init = _parents(value)[-1]
    helper = init["two_plane_context"]
    if change == "missing": del init["two_plane_context"]
    elif change == "unavailable": helper.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False)
    elif change == "budget_missing": del helper["budgets"]
    elif change == "budget_float": helper["budgets"]["max_scan_rows"] = 1000000.
    elif change == "wrong_schema": helper["schema_version"] = "two_plane_context_v3"
    elif change == "selected": helper["selected_policy_evaluation_complete"] = 1
    elif change == "accepted": helper["accepted_registration"] = True
    elif change == "finite": helper["finite_support_match_verified"] = True
    elif change == "production": helper["production_use_allowed"] = True
    else: init["two_plane_schedule"]["scheduled_matches"] = []
    _reject(wrapper, value)


@pytest.mark.parametrize("change", ["extra", "usable_int", "accepted_int", "schedule_nonempty",
    "schedule_unavailable", "selected_int", "usable_int_schedule", "joint_missing",
    "disabled", "enabled_int", "ordered_nonzero", "ordered_bool", "ordered_float",
    "slots_nonzero", "slots_bool", "slots_float"])
def test_empty_schedule_sentinel_is_exact_and_jointly_bound(wrapper, change):
    value = _payload(empty=True)
    init = _parents(value)[-1]
    helper, schedule, joint = [init[key] for key in
        ("two_plane_context", "two_plane_schedule", "joint_proposal_accounting")]
    if change == "extra": helper["schema_version"] = "two_plane_context_v4"
    elif change == "usable_int": helper["usable_proposals"] = 0
    elif change == "accepted_int": helper["accepted_registration"] = 0
    elif change == "schedule_nonempty": schedule["scheduled_matches"] = [dict(invented=True)]
    elif change == "schedule_unavailable": schedule.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False)
    elif change == "selected_int": schedule["selected_policy_evaluation_complete"] = 1
    elif change == "usable_int_schedule": schedule["usable_schedule"] = 0
    elif change == "joint_missing": del init["joint_proposal_accounting"]
    elif change == "disabled": joint["pair_lane_enabled"] = False
    elif change == "enabled_int": joint["pair_lane_enabled"] = 1
    elif change.startswith("ordered"):
        joint["ordered_pair_matches"] = {"ordered_nonzero": 1, "ordered_bool": False, "ordered_float": 0.}[change]
    else: joint["pair_slot_reservation"] = {"slots_nonzero": 32, "slots_bool": False, "slots_float": 0.}[change]
    _reject(wrapper, value)


@pytest.mark.parametrize("change", ["workflow_identity", "scope", "missing_scope", "units", "carrier_units"])
def test_no_identity_scope_or_unit_relabeling(wrapper, change):
    value = _payload()
    if change == "workflow_identity": value["implementation"] = "CAD_GUIDED_MESH_OFFLINE_V13"
    elif change == "scope": value["context_request_scope"] = "HISTORICAL_PAIRED_CONTROL"
    elif change == "missing_scope": del value["context_request_scope"]
    elif change == "units": value["units_status"] = "CALIBRATED_MM"
    else: value["predictions"][-1]["unit"] = "inch"
    _reject(wrapper, value)


def test_requested_submaximal_policy_is_bound_not_replaced_by_defaults(wrapper):
    request = _request()
    request["budgets"].update(max_scan_rows=500000, max_signed_families=1000,
        max_extrema_queries=3000, chunk_rows=3)
    value = _payload()
    _publish(value, request)
    _parents(value)[-1]["two_plane_context"]["budgets"] = dict(request["budgets"])
    blob = _small(value)
    args = _args(blob)
    args["expected_context_request"] = request
    projected = wrapper.project_gold11_schedule_workflow_v5(blob, **args)
    assert projected["disposition"] == "ADMITTED" and projected["requested_context"] == request
    assert projected["context_evidence"]["configuration_sha256"] == _sha(_small(request))
    assert _reconstruct(wrapper, blob, args, projected)["scorer_input"]["disposition"] == "SCORED_CONDITIONAL"


@pytest.mark.parametrize("change", ["hash", "duplicate", "nan", "overflow", "utf8", "truncated", "array", "depth", "byte_type", "cap"])
def test_exact_input_bytes_and_strict_json_clear_whole_result(wrapper, change):
    blob = _small(_payload())
    if change == "duplicate": blob = blob[:-1] + b',"later":{"x":1,"x":2}}'
    elif change == "nan": blob = blob[:-1] + b',"later":NaN}'
    elif change == "overflow": blob = blob[:-1] + b',"later":1e999}'
    elif change == "utf8": blob = b"\xff"
    elif change == "truncated": blob = blob[:-1]
    elif change == "array": blob = b"[]"
    elif change == "depth": blob = blob[:-1] + b',"later":' + b"[" * 66 + b"0" + b"]" * 66 + b"}"
    assert len(blob) < FIXTURE_CAP
    args = _args(blob)
    if change == "hash": args["expected_result_sha256"] = "0" * 64
    elif change == "byte_type": blob = bytearray(blob)
    elif change == "cap": args["max_result_bytes"] = len(blob) - 1
    projected = wrapper.project_gold11_schedule_workflow_v5(blob, **args)
    _cleared(wrapper, blob, args, projected)


@pytest.mark.parametrize("change", ["projection_hash", "result_whitespace", "context_digest",
    "configuration", "contact", "schedule", "extra", "schema"])
def test_exact_retained_projection_reconstruction_rejects_tampering(wrapper, change):
    blob, args, projected = _project(wrapper, _payload())
    assert projected["disposition"] == "ADMITTED"
    if change == "context_digest": projected["context_evidence"]["configuration_sha256"] = "0" * 64
    elif change == "configuration": projected["requested_context"]["budgets"]["max_scan_rows"] -= 1
    elif change == "contact": projected["contact_projection"]["predictions"].pop()
    elif change == "schedule": projected["schedule_evidence"]["verified_array_binding_count"] = 2
    elif change == "extra": projected["unrequested"] = True
    elif change == "schema": projected["schema"] = "GOLD11_SCHEDULE_BOUND_CONTACT_PROJECTION_V1"
    retained = _small(projected)
    expected = "0" * 64 if change == "projection_hash" else _sha(retained)
    if change == "result_whitespace": blob = b" " + blob
    with pytest.raises(ValueError):
        wrapper.revalidate_gold11_schedule_projection_v5(retained,
            expected_projection_sha256=expected, result_bytes=blob, **args)


def test_complete_finite_refusal_keeps_no_localized_carriers_without_relabeling(wrapper):
    blob, args, projected = _project(wrapper, _payload(finite=False))
    assert projected["disposition"] == "ADMITTED"
    scorer = _reconstruct(wrapper, blob, args, projected)["scorer_input"]
    assert scorer["disposition"] == "COMPLETE_NO_LOCALIZED_CARRIERS"
    assert scorer["predictions"] == [] and scorer["observed_bend_count"] is None
    assert scorer["production_use_allowed"] is False


@pytest.mark.parametrize("location", ["workflow", "evidence"])
@pytest.mark.parametrize("change", ["missing", "stale_v3", "non_string"])
def test_registration_implementation_is_bound_at_workflow_and_evidence_levels(wrapper, location, change):
    value = _payload()
    parent = value if location == "workflow" else value["registration"]["evidence"]
    if change == "missing":
        del parent["registration_implementation"]
    elif change == "stale_v3":
        parent["registration_implementation"] = "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
    else:
        parent["registration_implementation"] = True
    _reject(wrapper, value)


@pytest.mark.parametrize("location", ["workflow", "evidence"])
def test_missing_registration_implementation_is_tolerated_only_before_completion(wrapper, location):
    value = _payload()
    value.update(_refusal("REGISTRATION"))
    init = _parents(value)[-1]
    helper = init["two_plane_context"]
    helper.update(status="UNAVAILABLE", selected_policy_evaluation_complete=False, usable_proposals=False)
    del helper["budgets"]
    value["registration"]["evidence"]["measurement_context_indices"] = []
    value["registration_array_binding"] = []
    parent = value if location == "workflow" else value["registration"]["evidence"]
    del parent["registration_implementation"]
    blob, args, projected = _project(wrapper, value)
    assert projected["disposition"] == "ADMITTED"
    assert _reconstruct(wrapper, blob, args, projected)["scorer_input"]["disposition"] == "REFUSED"
    parent["registration_implementation"] = "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
    _reject(wrapper, value)


def test_v3_identities_are_the_published_v15_v14_v4_strings(wrapper):
    assert wrapper.WORKFLOW_IMPLEMENTATION == FLOW and wrapper.INITIALIZATION_IMPLEMENTATION == INIT
    assert wrapper.CONTEXT_IMPLEMENTATION == CONTEXT and wrapper.REGISTRATION_IMPLEMENTATION == REGISTRATION
    assert wrapper.SCHEMA == "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5"
    assert set(wrapper._ARRAY_KEYS) == set(ARRAY_KEYS)


def test_additive_v14_v15_schedule_keys_are_tolerated_by_the_projection(wrapper):
    value = _payload()
    evidence = value["registration"]["evidence"]
    evidence["measurement_context_schedule_policy"] = "POSE_CLASS_DEDUP_THEN_RANKED_BUDGET_CAP_V1"
    evidence["measurement_context_schedule"] = dict(policy="POSE_CLASS_DEDUP_THEN_RANKED_BUDGET_CAP_V1",
        max_measurement_contexts=8, supported_context_count=3, scheduled_context_count=3,
        dispositions=[dict(index=i, disposition="SCHEDULED") for i in range(3)])
    value["measurement_context_cap"] = dict(value=8, policy_maximum=8, feature_count=2, raw_rows=48,
        window_budget=4096, crop_row_budget=64000000, schedule="POSE_CLASS_DEDUP_THEN_RANKED_BUDGET_CAP_V1")
    blob, args, projected = _project(wrapper, value)
    assert projected["disposition"] == "ADMITTED"
    assert projected["context_evidence"]["verified_array_binding_count"] == 3
