"""Prospective pure schedule/initialization binding around contact scorer V2.

No backend import, geometry, filesystem or execution attestation. A request is
an independently frozen effective schedule budget map plus the complete V12
initialization map. V1's effective budgets must be the fixed native defaults;
the corresponding engine call still passes schedule mode v1 and budgets None.

Two bounded parses of the original result are possible (inner projection, then
schedule inspection). JSON node/depth guards follow decoding: external process
CPU/RSS/wall budgets remain mandatory. No public arbitrary-wrapper unwrap API
exists. Scoring input is returned only by exact-byte parent reconstruction.
"""

import hashlib
import json

from . import gold11_contact_scoring_v2 as _contact


SCHEMA = "GOLD11_SCHEDULE_BOUND_CONTACT_PROJECTION_V1"
AUTHORITY = "SUPPLIED_BYTES_AND_FROZEN_REQUEST_ONLY_NOT_RAW_SUPPORT_OR_EXECUTION_ATTESTATION"
MAX_RESULT_BYTES = _contact.MAX_RESULT_BYTES
MAX_WRAPPER_BYTES = 16 * 1024**2
MIN_WRAPPER_BYTES = 32768
MAX_WRAPPER_NODES = 300_000
MAX_CONTEXTS = 48
BREADTH_BUDGET_MAXIMA = dict(max_scan_planes=128, max_cad_planes=2048,
    max_scan_pair_records=8128, max_selected_scan_pairs=64, max_scan_selection_tests=520192,
    max_cad_pair_records=2096128, max_cad_sort_work=44018688, max_window_pair_tests=134152192,
    max_compatible_layout_bytes=134217728, max_compatible_layouts=64,
    max_working_bytes=268435456, max_report_bytes=33554432, chunk_pair_rows=65536)
SCHEDULE_BUDGET_MAXIMA = dict(BREADTH_BUDGET_MAXIMA,
    max_residual_priority_work=384, max_residual_pair_visits=64,
    max_residual_breadth_position_tests=128, max_residual_rank_selections=128,
    max_residual_skip_position_tests=512, max_residual_index_lookups=128,
    max_residual_cache_reference_records=64, max_residual_cache_reference_bytes=32768,
    max_residual_retained_bytes=1048576, max_residual_report_records=512,
    max_residual_report_reserved_bytes=1048576, max_final_ledger_tests=320)
INITIALIZATION_BUDGET_MAXIMA = dict(max_hypothesis_records=8192,
    max_prefilter_unsigned_assignments=1024, max_cached_cad_faces=2048,
    max_indexed_face_triangles=500_000, max_prefilter_query_calls=32768,
    max_prefilter_query_rows=64_000_000, chunk_rows=65536, max_node_visits=512_000_000,
    max_triangle_pairs=256_000_000, max_certificate_pairs=1_024_000_000,
    max_normal_pairs=32_000_000, max_work_units=4_000_000_000,
    max_face_build_nodes=131071, max_face_build_triangle_visits=10_000_000,
    max_refined_start_groups=48, max_report_bytes=100_663_296)
_DIGEST_KEYS = {"scan_bank_equations_sha256", "cad_bank_equations_sha256"}
_BINDING_KEYS = {"two_plane_schedule_mode", "two_plane_schedule_configuration_sha256",
    "two_plane_schedule_input_digests"}
_V2_MARKERS = {"breadth_ledger", "residual_windows", "residual_statistics",
    "breadth_statistics", "final_scan_pair_ledger"}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _budgets(value, maxima, positive, reason):
    _require(type(value) is dict and set(value) == set(maxima)
        and all(type(v) is int and 0 <= v <= maxima[k] for k, v in value.items())
        and value[positive] > 0, reason)
    return {key: value[key] for key in maxima}


def _request(value):
    _require(type(value) is dict and set(value) == {"mode", "budgets", "initialization_budgets"}
        and type(value["mode"]) is str and value["mode"] in ("v1", "v2"),
        "INVALID_CALLER_SCHEDULE_REQUEST")
    mode = value["mode"]
    caps = BREADTH_BUDGET_MAXIMA if mode == "v1" else SCHEDULE_BUDGET_MAXIMA
    budgets = _budgets(value["budgets"], caps, "chunk_pair_rows", "INVALID_CALLER_SCHEDULE_BUDGETS")
    if mode == "v1":
        _require(_contact._same(budgets, BREADTH_BUDGET_MAXIMA), "V1_REQUIRES_FROZEN_DEFAULT_EFFECTIVE_BUDGETS")
    initialization = _budgets(value["initialization_budgets"], INITIALIZATION_BUDGET_MAXIMA,
        "chunk_rows", "INVALID_CALLER_INITIALIZATION_BUDGETS")
    return dict(mode=mode, budgets=budgets, initialization_budgets=initialization)


def _canonical(value, cap):
    _contact._json_tree(value, maximum_nodes=MAX_WRAPPER_NODES)
    chunks, size = [], 0
    for piece in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        blob = piece.encode("utf-8")
        size += len(blob)
        _require(size <= cap, "SCHEDULE_WRAPPER_BYTE_BUDGET")
        chunks.append(blob)
    return b"".join(chunks)


def _parent(value, reason):
    _require(value is None or type(value) is dict, reason)
    return value


def _digests(value, *, complete):
    if not complete and (value is None or value == {}):
        return None
    _require(type(value) is dict and set(value) <= _DIGEST_KEYS
        and (not complete or set(value) == _DIGEST_KEYS)
        and all(_contact._hex(v) for v in value.values()), "INVALID_SCHEDULE_EQUATION_DIGESTS")
    return {key: value[key] for key in sorted(value)}


def _schedule(row, request, *, completed, breadth=False):
    """Metadata consistency only; never replay scheduling or raw support tests."""
    if row is None:
        _require(not completed, "MISSING_COMPLETE_SCHEDULE_PUBLICATION")
        return None
    _require(type(row) is dict, "INVALID_SCHEDULE_PUBLICATION")
    mode = "v1" if breadth else request["mode"]
    _require(row.get("schema_version") == "two_plane_schedule_" + mode,
        "SCHEDULE_SCHEMA_REQUEST_MISMATCH")
    if mode == "v1":
        _require(not (_V2_MARKERS & set(row)), "UNREQUESTED_V2_SCHEDULE_MARKER")
    status = row.get("status")
    _require(type(status) is str and status in ("COMPLETE", "UNAVAILABLE"), "INVALID_SCHEDULE_STATUS")
    done = status == "COMPLETE"
    _require(not completed or done, "INCOMPLETE_COMPLETED_SCHEDULE")
    _require(type(row.get("selected_policy_evaluation_complete")) is bool
        and row["selected_policy_evaluation_complete"] is done, "INVALID_SCHEDULE_COMPLETION_FLAG")
    for key in ("accepted_registration", "production_use_allowed", "finite_support_match_verified"):
        _require(row.get(key) is False, "INVALID_SCHEDULE_AUTHORITY")
    expected = {k: request["budgets"][k] for k in BREADTH_BUDGET_MAXIMA} if breadth else request["budgets"]
    if "budgets" in row:
        _require(_contact._same(row["budgets"], expected), "SCHEDULE_BUDGET_REQUEST_MISMATCH")
    else:
        _require(not done, "MISSING_COMPLETE_SCHEDULE_BUDGETS")
    digests = _digests(row.get("input_digests"), complete=done)
    if not done:
        _require(row.get("scheduled_matches") == [] and type(row.get("scheduled_matches")) is list
            and row.get("usable_schedule") is False, "INCOMPLETE_SCHEDULE_HAS_USABLE_PREFIX")
    result = dict(status=status, input_digests=digests)
    if mode == "v2":
        nested = _schedule(row.get("breadth_ledger"), request, completed=done, breadth=True)
        if nested is not None and nested["input_digests"] is not None and digests is not None:
            _require(all(nested["input_digests"][key] == digests[key]
                for key in nested["input_digests"].keys() & digests.keys()), "SCHEDULE_BREADTH_DIGEST_MISMATCH")
        result["breadth_status"] = nested["status"] if nested else None
    return result


def _schedule_evidence(payload, request, *, completed):
    policies = _parent(payload.get("policies"), "INVALID_SCHEDULE_POLICIES_PARENT")
    registration = _parent(payload.get("registration"), "INVALID_SCHEDULE_REGISTRATION_PARENT")
    evidence = _parent(registration.get("evidence") if registration else None, "INVALID_SCHEDULE_EVIDENCE_PARENT")
    initialization = _parent(evidence.get("initialization") if evidence else None, "INVALID_SCHEDULE_INITIALIZATION_PARENT")
    parents = {"policies": policies, "registration": evidence, "initialization": initialization}
    expected = dict(mode="v2", budgets=dict(request["budgets"])) if request["mode"] == "v2" else None
    configuration_sha = hashlib.sha256(_canonical(expected, MIN_WRAPPER_BYTES)).hexdigest() if expected else None
    presence = {}
    for name, parent in parents.items():
        present = parent is not None and "two_plane_schedule_configuration" in parent
        presence[name] = present
        if parent is not None:
            _require(not ((_BINDING_KEYS | _V2_MARKERS) & set(parent)), "MISPLACED_SCHEDULE_BINDING_METADATA")
        if request["mode"] == "v1":
            _require(not present, "UNREQUESTED_V2_SCHEDULE_CONFIGURATION")
        elif present:
            _require(_contact._same(parent["two_plane_schedule_configuration"], expected),
                "RETURNED_SCHEDULE_CONFIGURATION_MISMATCH")
        else:
            _require(not completed, "MISSING_COMPLETE_SCHEDULE_CONFIGURATION:" + name)
    if policies is not None and "two_plane_enabled" in policies:
        _require(policies["two_plane_enabled"] is True, "PAIR_LANE_NOT_EXPLICITLY_ENABLED")
    elif completed:
        raise ValueError("MISSING_COMPLETED_PAIR_LANE_POLICY")
    initialization_presence = {}
    for name, parent, key in (("policies", policies, "initialization_budgets"),
            ("initialization", initialization, "budgets")):
        present = parent is not None and key in parent
        initialization_presence[name] = present
        if present:
            _require(_contact._same(parent[key], request["initialization_budgets"]),
                "RETURNED_INITIALIZATION_BUDGET_MISMATCH")
        else:
            _require(not completed, "MISSING_COMPLETE_INITIALIZATION_BUDGET:" + name)
    row = initialization.get("two_plane_schedule") if initialization else None
    schedule = _schedule(row, request, completed=completed)
    context_ids = evidence.get("measurement_context_indices") if evidence else None
    if context_ids is not None:
        _require(type(context_ids) is list and len(context_ids) <= MAX_CONTEXTS
            and all(type(v) is int and 0 <= v < MAX_CONTEXTS for v in context_ids)
            and len(set(context_ids)) == len(context_ids), "INVALID_SCHEDULE_CONTEXT_LEDGER")
    if completed:
        _require(bool(context_ids), "MISSING_COMPLETE_SCHEDULE_CONTEXT_LEDGER")
    bindings = payload.get("registration_array_binding")
    if bindings is not None:
        _require(type(bindings) is list and len(bindings) <= MAX_CONTEXTS
            and all(type(row) is dict for row in bindings), "INVALID_SCHEDULE_ARRAY_BINDING_LEDGER")
        _require(context_ids is not None and len(bindings) == len(context_ids),
            "SCHEDULE_CONTEXT_BINDING_COUNT_MISMATCH")
    elif completed:
        raise ValueError("MISSING_COMPLETE_SCHEDULE_ARRAY_BINDINGS")
    for binding in bindings or []:
        _require(not (_V2_MARKERS & set(binding)), "MISPLACED_ARRAY_SCHEDULE_MARKER")
        keys = {k for k in binding if k.startswith("two_plane_schedule")}
        if request["mode"] == "v1":
            _require(not keys, "UNREQUESTED_V2_ARRAY_BINDING")
        else:
            _require(keys == _BINDING_KEYS and binding["two_plane_schedule_mode"] == "v2"
                and binding["two_plane_schedule_configuration_sha256"] == configuration_sha,
                "INVALID_V2_ARRAY_CONFIGURATION_BINDING")
            _require(schedule is not None and schedule["status"] == "COMPLETE"
                and _digests(binding["two_plane_schedule_input_digests"], complete=True) == schedule["input_digests"],
                "INVALID_V2_ARRAY_EQUATION_BINDING")
    return dict(status="COMPLETED_CONFIGURATION_BOUND" if completed else
            "PRESENT_METADATA_CONSISTENT_NOT_COMPLETE_WORKFLOW",
        configuration_presence=presence, initialization_budget_presence=initialization_presence,
        configuration_sha256=configuration_sha, schedule=schedule,
        measurement_context_indices=list(context_ids) if context_ids is not None else None,
        verified_array_binding_count=len(bindings) if bindings is not None else None,
        scope="REPORTED_METADATA_CONSISTENCY_NOT_SCHEDULE_REPLAY_OR_RAW_SUPPORT_PROOF")


def _base(request, scan_request, binding, limits):
    return dict(schema=SCHEMA, disposition="REJECTED", reason=None, authority=AUTHORITY,
        requested_schedule=request, requested_scan_planes=scan_request, result_binding=binding,
        schedule_evidence=None, contact_projection=None, limits=limits,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE")


def project_gold11_schedule_workflow_v1(result_bytes, *, expected_result_sha256,
        expected_scan_plane_mode, expected_schedule_request,
        expected_scan_plane_configuration=None, max_result_bytes=MAX_RESULT_BYTES,
        max_wrapper_bytes=MAX_WRAPPER_BYTES):
    """Return a fresh whole-arm envelope; invalid caller requests raise first.

    expected_schedule_request has exactly mode, full effective budgets and full
    initialization_budgets. V1 effective budgets describe fixed defaults, not a
    non-None budget argument to the native engine. Malformed returned metadata
    never retains a successful inner projection or diagnostic prefix.
    """
    _require(_contact._hex(expected_result_sha256), "INVALID_CALLER_RESULT_SHA256")
    request = _request(expected_schedule_request)
    scan_request = _contact._request(expected_scan_plane_mode, expected_scan_plane_configuration)
    _require(type(max_result_bytes) is int and 1 <= max_result_bytes <= MAX_RESULT_BYTES
        and type(max_wrapper_bytes) is int and MIN_WRAPPER_BYTES <= max_wrapper_bytes <= MAX_WRAPPER_BYTES,
        "INVALID_CALLER_WRAPPER_LIMITS")
    limits = dict(result_bytes=max_result_bytes, wrapper_bytes=max_wrapper_bytes,
        wrapper_nodes=MAX_WRAPPER_NODES, contexts=MAX_CONTEXTS)
    binding = dict(expected_sha256=expected_result_sha256, supplied_sha256=None,
        supplied_bytes=len(result_bytes) if type(result_bytes) is bytes else None, matched=False)
    try:
        _require(type(result_bytes) is bytes and 1 <= len(result_bytes) <= max_result_bytes,
            "RESULT_BYTE_TYPE_OR_BUDGET")
        binding["supplied_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        binding["matched"] = binding["supplied_sha256"] == expected_result_sha256
        _require(binding["matched"], "CALLER_RESULT_SHA256_MISMATCH")
        contact = _contact.project_gold11_contact_workflow_v2(result_bytes,
            expected_result_sha256=expected_result_sha256, expected_scan_plane_mode=expected_scan_plane_mode,
            expected_scan_plane_configuration=expected_scan_plane_configuration,
            max_result_bytes=max_result_bytes)
        _require(contact["disposition"] != "MALFORMED_RESULT", "CONTACT_PROJECTION_REJECTED:" + str(contact["reason"]))
        payload = _contact._decode(result_bytes)
        evidence = _schedule_evidence(payload, request, completed=payload["phase"] == "COMPLETE")
        output = _base(request, scan_request, binding, limits)
        output.update(disposition="ADMITTED", schedule_evidence=evidence, contact_projection=contact)
        _canonical(output, max_wrapper_bytes)
        return output
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError) as error:
        output = _base(request, scan_request, binding, limits)
        output["reason"] = str(error)[:256] or type(error).__name__
        _canonical(output, max_wrapper_bytes)
        return output


def revalidate_gold11_schedule_projection_v1(projection_bytes, *, expected_projection_sha256,
        result_bytes, **request):
    """Reconstruct from retained bytes before returning any scorer input.

    Caller pins both exact byte streams independently. Canonical comparison
    preserves JSON value types but permits harmless projection whitespace.
    Returned scorer_input is an untouched admitted inner projection or the exact
    pre-existing six-key supervisor refusal. No arbitrary-wrapper unwrap API is
    provided; future transport/dispatch/finalization must be qualified separately.
    """
    _require(_contact._hex(expected_projection_sha256), "INVALID_CALLER_PROJECTION_SHA256")
    exact_cap = request.get("max_wrapper_bytes", MAX_WRAPPER_BYTES)
    _require(type(exact_cap) is int and MIN_WRAPPER_BYTES <= exact_cap <= MAX_WRAPPER_BYTES,
        "INVALID_CALLER_WRAPPER_LIMITS")
    _require(type(projection_bytes) is bytes and 1 <= len(projection_bytes) <= exact_cap,
        "RETAINED_PROJECTION_BYTE_BUDGET")
    _require(hashlib.sha256(projection_bytes).hexdigest() == expected_projection_sha256,
        "RETAINED_PROJECTION_SHA256_MISMATCH")
    stored = _contact._decode(projection_bytes)
    rebuilt = project_gold11_schedule_workflow_v1(result_bytes, **request)
    cap = rebuilt["limits"]["wrapper_bytes"]
    _require(_canonical(stored, cap) == _canonical(rebuilt, cap), "RETAINED_PROJECTION_RECONSTRUCTION_MISMATCH")
    if rebuilt["disposition"] == "ADMITTED":
        scorer_input = rebuilt["contact_projection"]
    else:
        scorer_input = dict(status="UNAVAILABLE", phase="SUPERVISOR",
            reasons=["RUNNER_SCHEDULE_BINDING_REJECTED:" + rebuilt["reason"][:160]], predictions=[],
            observed_bend_count=None, production_use_allowed=False)
    return dict(projection=rebuilt, scorer_input=scorer_input)
