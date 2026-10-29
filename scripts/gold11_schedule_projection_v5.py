"""V5 of the pure context binding: the V4 text with the V17 workflow / V16 initializer (allowance V1) /
registration V5 identities, calling contact scoring V4 directly. V4 stays frozen.

Original V4 header: V4 of the pure context binding around the unchanged schedule/contact projector.

V3 text with the V16 workflow / V15 initializer (completion lane V1) / kernel V2
identities; V3 stays frozen. V3 text with the V15 workflow / V14 initializer / context-V4 identities, the
nine V4 context declarations (including the kd-partition label), the 512M
extrema coordinate maximum, and additionally the V4 registration implementation
label on the parent evidence and every array binding.

No backend imports, geometry, raw support proof or execution attestation.
Original RESULT bytes pass unchanged through V1; independently declared context
configuration/implementation metadata is then bound at every available level.
Only exact-byte parent reconstruction exposes the untouched contact projection.
"""
import hashlib

from . import gold11_schedule_projection_v1 as _schedule
from . import gold11_contact_scoring_v4 as _contact3

_contact = _schedule._contact  # V2 helpers (hex/decode/same/request) are the V3 text
CONTACT_PROJECTION_IMPLEMENTATION = _contact3.PROJECTION_SCHEMA
_require = _schedule._require
_canonical = _schedule._canonical
SCHEMA = "GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5"
AUTHORITY = _schedule.AUTHORITY
MAX_RESULT_BYTES = _schedule.MAX_RESULT_BYTES
MAX_WRAPPER_BYTES = _schedule.MAX_WRAPPER_BYTES
MIN_WRAPPER_BYTES = _schedule.MIN_WRAPPER_BYTES
INITIALIZATION_IMPLEMENTATION = "CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1"
WORKFLOW_IMPLEMENTATION = "CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1"
CONTEXT_IMPLEMENTATION = "RAW_EXTREMA_KD_PARTITION_CONTEXT_V4_STAGED_SCAN_PROJECTION"
REGISTRATION_IMPLEMENTATION = "SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1"
CONTEXT_SCOPE = "BOTH_SUCCESSOR_ARMS_CONTEXT_V4_NOT_HISTORICAL_PAIRED_CONTROL"
CONTEXT_BUDGET_MAXIMA = dict(max_scan_rows=1000000, max_scan_triangles=2000000,
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
CONTEXT_DECLARATIONS = dict(
    scan_projection_arithmetic="STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1",
    scan_extrema_partition="RANK_PARTITION_KD_TREE_GLOBAL_HALVED_ROOT_SPAN_AXIS_SCHEDULE_STABLE_SORT_LEAF64_V3",
    scan_projection_bound_scope="ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL",
    cad_projection_arithmetic="UNCHANGED_V1_NUMPY_MATMUL_PROJECTION",
    scan_projection_chunk_scope="FIXED_EXTREMA_LEAF64_NOT_CHUNK_ROWS",
    chunk_rows_scope="INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY",
    extrema_guard_status="NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_V3_INVARIANTS",
    extrema_membership_scope="SUM_OF_DISTINCT_INDEX_SUPPORT_SIZES_INCLUDING_WHOLE_SCAN",
    extrema_memory_scope="SHARED_RAW_OWNER_PLUS_ALL_LIVE_INDEX_PAYLOAD_UNDER_PARENT_CAP_NOT_RSS")
_PARENT_KEYS = {"initialization_implementation", "two_plane_context_configuration",
    "two_plane_context_configuration_sha256"}
_ARRAY_KEYS = {"initialization_implementation", "two_plane_context_mode",
    "two_plane_context_implementation", "two_plane_context_configuration_sha256",
    "registration_implementation"}


def _context_request(value):
    expected = dict(mode="v4", implementation=CONTEXT_IMPLEMENTATION,
        schema_version="two_plane_context_v4", budgets=None, **CONTEXT_DECLARATIONS)
    _require(type(value) is dict and set(value) == set(expected), "INVALID_CALLER_CONTEXT_REQUEST")
    _require(all(type(value[key]) is str and value[key] == item
        for key, item in expected.items() if key != "budgets"), "INVALID_CALLER_CONTEXT_IMPLEMENTATION")
    budgets = value["budgets"]
    _require(type(budgets) is dict and set(budgets) == set(CONTEXT_BUDGET_MAXIMA)
        and all(type(budgets[key]) is int and 0 <= budgets[key] <= maximum
            for key, maximum in CONTEXT_BUDGET_MAXIMA.items())
        and budgets["chunk_rows"] >= 3
        and budgets["max_extrema_queries"] <= 3*budgets["max_signed_families"],
        "INVALID_CALLER_CONTEXT_BUDGETS")
    expected["budgets"] = dict(budgets)
    return expected


def _context_evidence(payload, request, *, completed):
    _require(payload.get("implementation") == WORKFLOW_IMPLEMENTATION,
        "WORKFLOW_V17_IMPLEMENTATION_REQUIRED")
    if "registration_implementation" in payload:
        _require(payload["registration_implementation"] == REGISTRATION_IMPLEMENTATION,
            "REGISTRATION_V4_IMPLEMENTATION_REQUIRED:workflow")
    else:
        _require(not completed, "MISSING_COMPLETE_REGISTRATION_IMPLEMENTATION:workflow")
    if "context_request_scope" in payload:
        _require(payload["context_request_scope"] == CONTEXT_SCOPE, "CONTEXT_COMPARATOR_SCOPE_MISMATCH")
    else:
        _require(not completed, "MISSING_COMPLETE_CONTEXT_COMPARATOR_SCOPE")
    policies = _schedule._parent(payload.get("policies"), "INVALID_CONTEXT_POLICIES_PARENT")
    registration = _schedule._parent(payload.get("registration"), "INVALID_CONTEXT_REGISTRATION_PARENT")
    evidence = _schedule._parent(registration.get("evidence") if registration else None,
        "INVALID_CONTEXT_EVIDENCE_PARENT")
    initialization = _schedule._parent(evidence.get("initialization") if evidence else None,
        "INVALID_CONTEXT_INITIALIZATION_PARENT")
    configuration_sha = hashlib.sha256(_canonical(request, MIN_WRAPPER_BYTES)).hexdigest()
    presence = {}
    for name, parent in (("workflow", payload), ("policies", policies),
            ("registration", evidence), ("initialization", initialization)):
        keys = set() if parent is None else {key for key in parent
            if (key.startswith("two_plane_context") or key == "initialization_implementation")
            and not (name == "initialization" and key == "two_plane_context")}
        _require(keys <= _PARENT_KEYS, "UNKNOWN_CONTEXT_PUBLICATION_KEY:"+name)
        if parent is not None and "initialization_implementation" in parent:
            _require(parent["initialization_implementation"] == INITIALIZATION_IMPLEMENTATION,
                "INITIALIZATION_V16_IMPLEMENTATION_REQUIRED:"+name)
        if name == "registration" and parent is not None:
            if "registration_implementation" in parent:
                _require(parent["registration_implementation"] == REGISTRATION_IMPLEMENTATION,
                    "REGISTRATION_V4_IMPLEMENTATION_REQUIRED:"+name)
            else:
                _require(not completed, "MISSING_COMPLETE_REGISTRATION_IMPLEMENTATION:"+name)
        pair = {"two_plane_context_configuration", "two_plane_context_configuration_sha256"}
        present = bool(pair & keys)
        presence[name] = present
        if present:
            _require(keys == _PARENT_KEYS, "PARTIAL_CONTEXT_CONFIGURATION_PUBLICATION:"+name)
            actual = _context_request(parent["two_plane_context_configuration"])
            _require(_contact._same(actual, request)
                and parent["two_plane_context_configuration_sha256"] == configuration_sha,
                "RETURNED_CONTEXT_CONFIGURATION_MISMATCH:"+name)
        else:
            _require(not completed, "MISSING_COMPLETE_CONTEXT_CONFIGURATION:"+name)
    helper = initialization.get("two_plane_context") if initialization else None
    schedule = initialization.get("two_plane_schedule") if initialization else None
    joint = initialization.get("joint_proposal_accounting") if initialization else None
    sentinel = dict(status="NOT_CALLED_EMPTY_DECLARED_SCHEDULE", usable_proposals=False,
        accepted_registration=False)
    if type(helper) is dict and helper.get("status") == sentinel["status"]:
        _require(_contact._same(helper, sentinel)
            and type(schedule) is dict and schedule.get("status") == "COMPLETE"
            and schedule.get("selected_policy_evaluation_complete") is True
            and type(schedule.get("scheduled_matches")) is list and schedule["scheduled_matches"] == []
            and schedule.get("usable_schedule") is False
            and type(joint) is dict and joint.get("pair_lane_enabled") is True
            and type(joint.get("ordered_pair_matches")) is int and joint["ordered_pair_matches"] == 0
            and type(joint.get("pair_slot_reservation")) is int and joint["pair_slot_reservation"] == 0,
            "CONTEXT_NOT_CALLED_SENTINEL_REQUIRES_EMPTY_DECLARED_SCHEDULE")
    elif helper is not None:
        _require(type(helper) is dict and helper.get("schema_version") == "two_plane_context_v4",
            "UNREQUESTED_CONTEXT_IMPLEMENTATION")
        _require(all(type(helper.get(key)) is str and helper[key] == value
            for key, value in CONTEXT_DECLARATIONS.items()), "CONTEXT_HELPER_ARITHMETIC_MISMATCH")
        _require(helper.get("status") in ("COMPLETE", "UNAVAILABLE")
            and helper.get("selected_policy_evaluation_complete") is (helper["status"] == "COMPLETE")
            and type(helper.get("usable_proposals")) is bool
            and (helper["status"] == "COMPLETE" or helper["usable_proposals"] is False)
            and all(helper.get(key) is False for key in
                ("accepted_registration", "finite_support_match_verified", "production_use_allowed")),
            "CONTEXT_HELPER_STATE_OR_AUTHORITY_MISMATCH")
        if "budgets" in helper:
            _require(_contact._same(helper["budgets"], request["budgets"]), "CONTEXT_HELPER_BUDGET_MISMATCH")
        else:
            _require(helper.get("status") != "COMPLETE", "COMPLETE_CONTEXT_HELPER_BUDGET_REQUIRED")
        if completed:
            _require(helper.get("status") == "COMPLETE"
                and helper.get("selected_policy_evaluation_complete") is True
                and all(helper.get(key) is False for key in
                    ("accepted_registration", "finite_support_match_verified", "production_use_allowed"))
                and type(schedule) is dict and type(schedule.get("scheduled_matches")) is list
                and bool(schedule["scheduled_matches"]), "COMPLETED_WORKFLOW_REQUIRES_COMPLETE_NONEMPTY_CONTEXT")
    else:
        _require(not completed, "MISSING_COMPLETE_CONTEXT_HELPER")
    bindings = payload.get("registration_array_binding")
    if bindings is not None:
        _require(type(bindings) is list and len(bindings) <= _schedule.MAX_CONTEXTS
            and all(type(item) is dict for item in bindings), "INVALID_CONTEXT_ARRAY_BINDING_LEDGER")
    else:
        _require(not completed, "MISSING_COMPLETE_CONTEXT_ARRAY_BINDINGS")
    for item in bindings or []:
        keys = {key for key in item if key.startswith("two_plane_context")
            or key in ("initialization_implementation", "registration_implementation")}
        _require(keys == _ARRAY_KEYS and item["initialization_implementation"] == INITIALIZATION_IMPLEMENTATION
            and item["two_plane_context_mode"] == "v4"
            and item["two_plane_context_implementation"] == CONTEXT_IMPLEMENTATION
            and item["registration_implementation"] == REGISTRATION_IMPLEMENTATION
            and item["two_plane_context_configuration_sha256"] == configuration_sha,
            "CONTEXT_ARRAY_BINDING_MISMATCH")
    return dict(status="COMPLETED_CONFIGURATION_BOUND" if completed else
        "PRESENT_METADATA_CONSISTENT_NOT_COMPLETE_WORKFLOW", configuration_presence=presence,
        configuration_sha256=configuration_sha,
        verified_array_binding_count=len(bindings) if bindings is not None else None,
        scope="REPORTED_METADATA_CONSISTENCY_NOT_RAW_SUPPORT_OR_EXECUTION_PROOF")


def _project_schedule_with_contact_v4(result_bytes, *, expected_result_sha256, expected_scan_plane_mode,
        expected_schedule_request, expected_scan_plane_configuration=None, max_result_bytes=MAX_RESULT_BYTES,
        max_wrapper_bytes=MAX_WRAPPER_BYTES):
    """The V1 schedule projection body, byte-for-byte in logic, with the V3 contact projection.

    V1 delegates to the frozen contact projection V2, whose measurement grammar (cover V2,
    context V1, seven-key policy, one material-side value) rejects every V9 measurement. V4
    therefore runs V1's own request/evidence helpers around contact V3 instead.
    """
    _require(_contact._hex(expected_result_sha256), "INVALID_CALLER_RESULT_SHA256")
    request = _schedule._request(expected_schedule_request)
    scan_request = _contact._request(expected_scan_plane_mode, expected_scan_plane_configuration)
    _require(type(max_result_bytes) is int and 1 <= max_result_bytes <= MAX_RESULT_BYTES
        and type(max_wrapper_bytes) is int and MIN_WRAPPER_BYTES <= max_wrapper_bytes <= MAX_WRAPPER_BYTES,
        "INVALID_CALLER_WRAPPER_LIMITS")
    limits = dict(result_bytes=max_result_bytes, wrapper_bytes=max_wrapper_bytes,
        wrapper_nodes=_schedule.MAX_WRAPPER_NODES, contexts=_schedule.MAX_CONTEXTS)
    binding = dict(expected_sha256=expected_result_sha256, supplied_sha256=None,
        supplied_bytes=len(result_bytes) if type(result_bytes) is bytes else None, matched=False)
    try:
        _require(type(result_bytes) is bytes and 1 <= len(result_bytes) <= max_result_bytes,
            "RESULT_BYTE_TYPE_OR_BUDGET")
        binding["supplied_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        binding["matched"] = binding["supplied_sha256"] == expected_result_sha256
        _require(binding["matched"], "CALLER_RESULT_SHA256_MISMATCH")
        contact = _contact3.project_gold11_contact_workflow_v4(result_bytes,
            expected_result_sha256=expected_result_sha256, expected_scan_plane_mode=expected_scan_plane_mode,
            expected_scan_plane_configuration=expected_scan_plane_configuration,
            max_result_bytes=max_result_bytes)
        _require(contact["disposition"] != "MALFORMED_RESULT", "CONTACT_PROJECTION_REJECTED:" + str(contact["reason"]))
        payload = _contact._decode(result_bytes)
        evidence = _schedule._schedule_evidence(payload, request, completed=payload["phase"] == "COMPLETE")
        output = _schedule._base(request, scan_request, binding, limits)
        output.update(disposition="ADMITTED", schedule_evidence=evidence, contact_projection=contact)
        _schedule._canonical(output, max_wrapper_bytes)
        return output
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as error:
        output = _schedule._base(request, scan_request, binding, limits)
        output["reason"] = str(error)[:256] or type(error).__name__
        _schedule._canonical(output, max_wrapper_bytes)
        return output


def _base(request, context, scan_request, binding, limits):
    result = _schedule._base(request, scan_request, binding, limits)
    result.update(schema=SCHEMA, requested_context=context, context_evidence=None,
                  contact_projection_implementation=CONTACT_PROJECTION_IMPLEMENTATION)
    return result


def project_gold11_schedule_workflow_v5(result_bytes, *, expected_result_sha256,
        expected_scan_plane_mode, expected_schedule_request, expected_context_request,
        expected_scan_plane_configuration=None, max_result_bytes=MAX_RESULT_BYTES,
        max_wrapper_bytes=MAX_WRAPPER_BYTES):
    _require(_contact._hex(expected_result_sha256), "INVALID_CALLER_RESULT_SHA256")
    request = _schedule._request(expected_schedule_request)
    context = _context_request(expected_context_request)
    scan = _contact._request(expected_scan_plane_mode, expected_scan_plane_configuration)
    _require(type(max_result_bytes) is int and 1 <= max_result_bytes <= MAX_RESULT_BYTES
        and type(max_wrapper_bytes) is int and MIN_WRAPPER_BYTES <= max_wrapper_bytes <= MAX_WRAPPER_BYTES,
        "INVALID_CALLER_WRAPPER_LIMITS")
    limits = dict(result_bytes=max_result_bytes, wrapper_bytes=max_wrapper_bytes,
        wrapper_nodes=_schedule.MAX_WRAPPER_NODES, contexts=_schedule.MAX_CONTEXTS)
    binding = dict(expected_sha256=expected_result_sha256, supplied_sha256=None,
        supplied_bytes=len(result_bytes) if type(result_bytes) is bytes else None, matched=False)
    try:
        _require(type(result_bytes) is bytes and 1 <= len(result_bytes) <= max_result_bytes,
            "RESULT_BYTE_TYPE_OR_BUDGET")
        binding["supplied_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        binding["matched"] = binding["supplied_sha256"] == expected_result_sha256
        _require(binding["matched"], "CALLER_RESULT_SHA256_MISMATCH")
        scheduled = _project_schedule_with_contact_v4(result_bytes,
            expected_result_sha256=expected_result_sha256, expected_scan_plane_mode=expected_scan_plane_mode,
            expected_scan_plane_configuration=expected_scan_plane_configuration,
            expected_schedule_request=request, max_result_bytes=max_result_bytes,
            max_wrapper_bytes=max_wrapper_bytes)
        _require(scheduled["disposition"] == "ADMITTED", "SCHEDULE_PROJECTION_REJECTED:"+str(scheduled["reason"]))
        payload = _contact._decode(result_bytes)
        evidence = _context_evidence(payload, context, completed=payload["phase"] == "COMPLETE")
        output = _base(request, context, scan, binding, limits)
        output.update(disposition="ADMITTED", schedule_evidence=scheduled["schedule_evidence"],
            context_evidence=evidence, contact_projection=scheduled["contact_projection"])
        _canonical(output, max_wrapper_bytes)
        return output
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as error:
        output = _base(request, context, scan, binding, limits)
        output["reason"] = str(error)[:256] or type(error).__name__
        _canonical(output, max_wrapper_bytes)
        return output


def revalidate_gold11_schedule_projection_v5(projection_bytes, *, expected_projection_sha256,
        result_bytes, **request):
    _require(_contact._hex(expected_projection_sha256), "INVALID_CALLER_PROJECTION_SHA256")
    cap = request.get("max_wrapper_bytes", MAX_WRAPPER_BYTES)
    _require(type(cap) is int and MIN_WRAPPER_BYTES <= cap <= MAX_WRAPPER_BYTES,
        "INVALID_CALLER_WRAPPER_LIMITS")
    _require(type(projection_bytes) is bytes and 1 <= len(projection_bytes) <= cap,
        "RETAINED_PROJECTION_BYTE_BUDGET")
    _require(hashlib.sha256(projection_bytes).hexdigest() == expected_projection_sha256,
        "RETAINED_PROJECTION_SHA256_MISMATCH")
    stored = _contact._decode(projection_bytes)
    rebuilt = project_gold11_schedule_workflow_v5(result_bytes, **request)
    _require(_canonical(stored, cap) == _canonical(rebuilt, cap), "RETAINED_PROJECTION_RECONSTRUCTION_MISMATCH")
    if rebuilt["disposition"] == "ADMITTED":
        scorer_input = rebuilt["contact_projection"]
    else:
        scorer_input = dict(status="UNAVAILABLE", phase="SUPERVISOR",
            reasons=["RUNNER_CONTEXT_BINDING_REJECTED:"+rebuilt["reason"][:160]], predictions=[],
            observed_bend_count=None, production_use_allowed=False)
    return dict(projection=rebuilt, scorer_input=scorer_input)
