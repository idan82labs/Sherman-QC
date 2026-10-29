"""Idempotent, pose-free initialization report refusal; pure stdlib only.

Only the exact bounded V6 refusal schema is exempt from a detailed report cap.
Its counters describe retained lists/prior canonical evidence, never unseen work.
No exception text, geometry decision, native import or file IO belongs here.
"""

import json
import string


MAX_DETAILED_BYTES = 96*1024**2
MAX_ENVELOPE_BYTES = 4096
MAX_RECORD_COUNT = 2**63-1
_SCHEMA = "initialization_report_refusal_envelope_v6"
_REASONS = frozenset(("INITIALIZATION_REPORT_BUDGET_EXCEEDED",
    "INITIALIZATION_REPORT_SERIALIZATION_FAILURE", "INVALID_INITIALIZATION_REPORT_CAP",
    "INVALID_INITIALIZATION_REPORT_CONTAINER", "INVALID_INITIALIZATION_REFUSAL_ENVELOPE"))
_COUNT_STATUSES = frozenset(("RETAINED_RECORD_COUNT", "NO_RETAINED_LIST_PRESENT",
    "UNKNOWN_MALFORMED_CONTAINER", "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER"))
_CODE_CHARACTERS = frozenset(string.ascii_uppercase+string.digits+"_ .:-/")
_SCHEMA_CHARACTERS = frozenset(string.ascii_letters+string.digits+"_.-")
_FIXED = dict(schema_version=_SCHEMA, status="UNAVAILABLE", transform_scan_to_cad=None,
    search_complete=False, selected_policy_evaluation_complete=False,
    accepted_registration=False, production_use_allowed=False, observed_bend_count=None,
    conformance="INDETERMINATE", detailed_report_discarded_whole=True,
    evidence_scope="NO_PARTIAL_REPORT_OR_POSE_PUBLISHED",
    count_basis="RETAINED_LIST_LENGTHS_OR_PRIOR_CANONICAL_COUNTS_NOT_UNSEEN_WORK",
    byte_evidence_kind="ENCODED_PREFIX_COUNT_CAPPED_AT_LIMIT_PLUS_ONE")
_EVENT_KEYS = frozenset(("boundary_reason", "detailed_report_limit_bytes",
    "detailed_report_limit_valid", "encoded_bytes_examined", "source_schema_version",
    "source_reasons", "source_reasons_complete", "discarded_hypothesis_records",
    "discarded_hypothesis_count_status", "discarded_candidate_records",
    "discarded_candidate_count_status"))
_ENVELOPE_KEYS = frozenset(_FIXED) | (_EVENT_KEYS-{"boundary_reason"}) | frozenset(
    ("reasons", "hypotheses", "start_groups", "first_refusal"))


def _integer(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def _code(value):
    return type(value) is str and 1 <= len(value) <= 180 and all(c in _CODE_CHARACTERS for c in value)


def _schema(value):
    return value is None or (type(value) is str and 1 <= len(value) <= 96
        and all(c in _SCHEMA_CHARACTERS for c in value))


def _count_valid(value, status):
    if type(status) is not str or status not in _COUNT_STATUSES:
        return False
    if status == "UNKNOWN_MALFORMED_CONTAINER":
        return value is None
    return _integer(value, MAX_RECORD_COUNT) and (status != "NO_RETAINED_LIST_PRESENT" or value == 0)


def _event_valid(value):
    if (type(value) is not dict or len(value) != len(_EVENT_KEYS)
            or not all(type(key) is str for key in value) or value.keys() != _EVENT_KEYS):
        return False
    reason, limit = value["boundary_reason"], value["detailed_report_limit_bytes"]
    if type(reason) is not str or reason not in _REASONS or not _integer(limit, MAX_DETAILED_BYTES):
        return False
    if type(value["detailed_report_limit_valid"]) is not bool:
        return False
    if not value["detailed_report_limit_valid"] and (limit != 0 or reason != "INVALID_INITIALIZATION_REPORT_CAP"):
        return False
    if not _integer(value["encoded_bytes_examined"], limit+1) or not _schema(value["source_schema_version"]):
        return False
    if reason == "INITIALIZATION_REPORT_BUDGET_EXCEEDED" and (
            not value["detailed_report_limit_valid"] or value["encoded_bytes_examined"] != limit+1):
        return False
    if reason == "INVALID_INITIALIZATION_REPORT_CAP" and (
            value["detailed_report_limit_valid"] or value["encoded_bytes_examined"] != 0):
        return False
    reasons = value["source_reasons"]
    if type(reasons) is not list or len(reasons) > 3 or not all(_code(reason) for reason in reasons):
        return False
    if type(value["source_reasons_complete"]) is not bool:
        return False
    return all(_count_valid(value["discarded_"+name+"_records"], value["discarded_"+name+"_count_status"])
               for name in ("hypothesis", "candidate"))


def _event_copy(value):
    event = {key: value[key] for key in _EVENT_KEYS}
    event["source_reasons"] = list(event["source_reasons"])
    return event


def _canonical_bytes(value):
    # Called ONLY after fixed-schema validation: all strings/keys are safe ASCII
    # without JSON escapes, all integers are small and depth is fixed. This exact
    # compact-JSON length avoids re-entering a failing general-purpose encoder.
    if value is None:
        return 4
    if type(value) is bool:
        return 4 if value else 5
    if type(value) is int:
        return len(str(value))
    if type(value) is str:
        return len(value)+2
    if type(value) is list:
        return 2+max(0, len(value)-1)+sum(_canonical_bytes(item) for item in value)
    return 2+max(0, len(value)-1)+sum(len(key)+3+_canonical_bytes(item) for key, item in value.items())


def _canonical(value):
    """Bounded validation only; neither a flag nor a schema name grants trust."""
    if (type(value) is not dict or len(value) != len(_ENVELOPE_KEYS)
            or not all(type(key) is str for key in value) or value.keys() != _ENVELOPE_KEYS):
        return False
    if any(type(value[key]) is not type(expected) or value[key] != expected for key, expected in _FIXED.items()):
        return False
    for key in ("hypotheses", "start_groups"):
        if type(value[key]) is not list or value[key]:
            return False
    reasons = value["reasons"]
    if type(reasons) is not list or len(reasons) != 1:
        return False
    event = {key: value[key] for key in _EVENT_KEYS if key != "boundary_reason"}
    event["boundary_reason"] = reasons[0]
    if not _event_valid(event) or not _event_valid(value["first_refusal"]):
        return False
    return _canonical_bytes(value) <= MAX_ENVELOPE_BYTES


def _copy_envelope(value):
    result = dict(value)
    for key in ("reasons", "source_reasons", "hypotheses", "start_groups"):
        result[key] = list(value[key])
    result["first_refusal"] = _event_copy(value["first_refusal"])
    return result


def _container_count(value, key):
    if type(value) is not dict:
        return None, "UNKNOWN_MALFORMED_CONTAINER"
    if key not in value:
        return 0, "NO_RETAINED_LIST_PRESENT"
    rows = value[key]
    if type(rows) is not list:
        return None, "UNKNOWN_MALFORMED_CONTAINER"
    return len(rows), "RETAINED_RECORD_COUNT"


def _merge_counts(entries):
    known = [count for count, status in entries if status not in
             ("UNKNOWN_MALFORMED_CONTAINER", "NO_RETAINED_LIST_PRESENT")]
    unknown = any(status in ("UNKNOWN_MALFORMED_CONTAINER", "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER")
                  for _, status in entries)
    if known:
        return max(known), "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER" if unknown else "RETAINED_RECORD_COUNT"
    return (None, "UNKNOWN_MALFORMED_CONTAINER") if unknown else (0, "NO_RETAINED_LIST_PRESENT")


def _retained_evidence(result):
    # Only fixed depth is inspected. Pools at these levels can alias: MAX, not
    # addition, preserves a known lower bound without claiming distinct work.
    if _canonical(result):
        return ((result["discarded_hypothesis_records"], result["discarded_hypothesis_count_status"]),
            (result["discarded_candidate_records"], result["discarded_candidate_count_status"]),
            _event_copy(result["first_refusal"]))
    hypothesis = [_container_count(result, "hypotheses")]
    candidate = [] if type(result) is dict else [(None, "UNKNOWN_MALFORMED_CONTAINER")]
    first = None
    if type(result) is dict and "evidence" in result:
        evidence = result["evidence"]
        candidate.append(_container_count(evidence, "candidates"))
        if type(evidence) is not dict:
            hypothesis.append((None, "UNKNOWN_MALFORMED_CONTAINER"))
        elif "initialization" in evidence:
            inner = evidence["initialization"]
            if _canonical(inner):
                hypothesis.append((inner["discarded_hypothesis_records"], inner["discarded_hypothesis_count_status"]))
                candidate.append((inner["discarded_candidate_records"], inner["discarded_candidate_count_status"]))
                first = _event_copy(inner["first_refusal"])
            else:
                hypothesis.append(_container_count(inner, "hypotheses"))
                if type(inner) is not dict:
                    candidate.append((None, "UNKNOWN_MALFORMED_CONTAINER"))
    return _merge_counts(hypothesis), _merge_counts(candidate), first


def _source_metadata(result):
    if type(result) is not dict:
        return None, [], False
    schema = result.get("schema_version")
    schema = schema if _schema(schema) else None
    reasons = result.get("reasons", [])
    if type(reasons) is not list:
        return schema, [], False
    # Inspect at most three entries; large/malformed diagnostics are explicitly
    # incomplete, never converted via arbitrary __str__ or silently authenticated.
    selected = [reason for reason in reasons[:3] if _code(reason)]
    return schema, selected, len(reasons) <= 3 and len(selected) == len(reasons)


def _refusal(result, reason, maximum, examined, valid_limit):
    # A refusal must also survive malformed ordinary dictionaries whose key
    # equality/hash hooks raise. Unknown is safer than a fabricated zero; keep
    # independently recovered count evidence if only source diagnostics fail.
    try:
        hypotheses, candidates, first = _retained_evidence(result)
    except Exception:
        hypotheses = candidates = (None, "UNKNOWN_MALFORMED_CONTAINER")
        first = None
    try:
        source_schema, source_reasons, complete = _source_metadata(result)
    except Exception:
        source_schema, source_reasons, complete = None, [], False
    event = dict(boundary_reason=reason, detailed_report_limit_bytes=maximum,
        detailed_report_limit_valid=valid_limit, encoded_bytes_examined=examined,
        source_schema_version=source_schema, source_reasons=source_reasons, source_reasons_complete=complete,
        discarded_hypothesis_records=hypotheses[0], discarded_hypothesis_count_status=hypotheses[1],
        discarded_candidate_records=candidates[0], discarded_candidate_count_status=candidates[1])
    envelope = dict(_FIXED, **{key: value for key, value in event.items() if key != "boundary_reason"},
        reasons=[reason], hypotheses=[], start_groups=[], first_refusal=_event_copy(event) if first is None else first)
    return envelope


def bound_initialization_report_v6(result, maximum):
    """Keep an encodable detailed report or replace it with a canonical refusal.

Canonical refusals retain their FIRST evidence and are detached/idempotent even
under a zero detailed cap. This is schema validation, not source authentication.
Ordinary report failures are caught; process-control BaseExceptions propagate.
"""
    valid_limit = _integer(maximum, MAX_DETAILED_BYTES)
    if not valid_limit:
        return _refusal(result, "INVALID_INITIALIZATION_REPORT_CAP", 0, 0, False)
    examined = 0
    try:
        if _canonical(result):
            return _copy_envelope(result)
        if type(result) is not dict:
            return _refusal(result, "INVALID_INITIALIZATION_REPORT_CONTAINER", maximum, 0, True)
        if result.get("detailed_report_discarded_whole") is True or result.get("schema_version") == _SCHEMA:
            return _refusal(result, "INVALID_INITIALIZATION_REFUSAL_ENVELOPE", maximum, 0, True)
        for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False,
                sort_keys=True, separators=(",", ":")).iterencode(result):
            # ensure_ascii means len(chunk) is exactly its UTF-8 byte count.
            # Record only the proven prefix through limit+1, not an invented
            # full-report size after early termination or an encoding exception.
            examined = min(maximum+1, examined+len(chunk))
            if examined > maximum:
                return _refusal(result, "INITIALIZATION_REPORT_BUDGET_EXCEEDED", maximum, examined, True)
        return result
    except Exception:
        return _refusal(result, "INITIALIZATION_REPORT_SERIALIZATION_FAILURE", maximum, examined, True)
