"""Pure, bounded native-status coverage accounting; never an admission gate.

Accepts caller-owned structural arrays from a diagnostic, not native handles or
files. Coverage concerns the supplied typed occurrence tree only. Neither that
tree nor an optional caller-supplied digest is authenticated here. No geometry,
native context calculation, healing, observed bend count or production decision.
"""
from collections import namedtuple
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NativeContextCoverageLimitsV3:
    max_entities: int = 2048
    max_occurrences: int = 4096
    max_depth: int = 32
    max_details: int = 1024
    max_expected_context_pairs: int = 4096
    max_observed_context_pairs: int = 4096
    max_statuses_per_list: int = 32
    max_total_status_values: int = 32768
    max_ancestor_steps: int = 131072


# Hard caps must not share the caller-visible (frozen but bypassable) dataclass
# instance. Captured integer limits are validated against these literal values.
_LIMIT_MAXIMA = (
    ("max_entities", 2048), ("max_occurrences", 4096), ("max_depth", 32),
    ("max_details", 1024), ("max_expected_context_pairs", 4096),
    ("max_observed_context_pairs", 4096), ("max_statuses_per_list", 32),
    ("max_total_status_values", 32768), ("max_ancestor_steps", 131072),
)
_LimitSnapshot = namedtuple("_LimitSnapshot", tuple(name for name, _ in _LIMIT_MAXIMA))
_TYPE_CAPS = dict(COMPOUND=32, COMPSOLID=16, SOLID=8, SHELL=64,
                  FACE=256, WIRE=512, EDGE=1024, VERTEX=1024)
_CHILD_TYPES = {
    "COMPOUND": frozenset(_TYPE_CAPS), "COMPSOLID": frozenset(("SOLID",)),
    "SOLID": frozenset(("SHELL",)), "SHELL": frozenset(("FACE",)),
    "FACE": frozenset(("WIRE",)), "WIRE": frozenset(("EDGE",)),
    "EDGE": frozenset(("VERTEX",)), "VERTEX": frozenset(),
}
# These are the context relations required by the declared analyzer contract,
# NOT every possible ancestor pair (e.g. FACE->SHELL is intentionally absent).
_CONTEXT_TYPES = frozenset((
    ("VERTEX", "EDGE"), ("VERTEX", "FACE"), ("EDGE", "FACE"),
    ("WIRE", "FACE"), ("SHELL", "SOLID"),
))
_ENTITY_FIELDS = frozenset(("native_id", "type", "type_index",
                            "occurrence_count", "location_is_identity"))
_DETAIL_REQUIRED = frozenset(("native_id", "self_status_codes", "contexts",
                              "detail_complete", "errors"))
_DETAIL_OPTIONAL = frozenset(("subtree_valid", "native_tolerance_mm",
                              "native_tolerance_numeric_valid", "geometry_type",
                              "degenerated", "same_parameter", "same_range"))


class _CoverageUnavailable(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise _CoverageUnavailable(reason)


def _integer(value, low, high, reason):
    _require(type(value) is int and low <= value <= high, reason)
    return value


def _rows(value, cap, reason):
    _require(type(value) is list and len(value) <= cap, reason)
    return tuple(value)


def _record(value, required, optional, reason):
    _require(type(value) is dict and len(value) <= len(required | optional), reason)
    _require(all(type(key) is str for key in value), reason)
    result = dict(value)
    _require(required <= result.keys() <= required | optional, reason)
    return result


def _validated_limits(limits):
    _require(type(limits) is NativeContextCoverageLimitsV3, "LIMITS_SCHEMA")
    values = tuple(getattr(limits, name) for name, _ in _LIMIT_MAXIMA)
    for value, (_, maximum) in zip(values, _LIMIT_MAXIMA):
        _integer(value, 1, maximum, "LIMITS_OUTSIDE_FIXED_CAPS")
    return _LimitSnapshot(*values)  # No caller-object reads after validation.


def _entities(values, limits):
    rows = _rows(values, limits.max_entities, "ENTITY_BUDGET_OR_SCHEMA")
    _require(bool(rows), "EMPTY_ENTITY_SET")
    result, indices = {}, {kind: set() for kind in _TYPE_CAPS}
    for value in rows:
        row = _record(value, _ENTITY_FIELDS, frozenset(), "ENTITY_SCHEMA")
        identifier = _integer(row["native_id"], 0, len(rows)-1, "ENTITY_ID")
        _require(identifier not in result, "DUPLICATE_ENTITY_ID")
        kind = row["type"]
        _require(type(kind) is str and len(kind) <= 16 and kind in _TYPE_CAPS,
                 "UNSUPPORTED_ENTITY_TYPE")
        index = _integer(row["type_index"], 0, _TYPE_CAPS[kind]-1, "ENTITY_TYPE_INDEX")
        _require(index not in indices[kind], "DUPLICATE_ENTITY_TYPE_INDEX")
        indices[kind].add(index)
        _require(len(indices[kind]) <= _TYPE_CAPS[kind], "ENTITY_TYPE_BUDGET")
        count = _integer(row["occurrence_count"], 1, limits.max_occurrences,
                         "ENTITY_OCCURRENCE_COUNT")
        _require(type(row["location_is_identity"]) is bool, "ENTITY_LOCATION_SCHEMA")
        result[identifier] = (kind, index, count, row["location_is_identity"])
    _require(set(result) == set(range(len(rows))), "ENTITY_ID_DENOMINATOR")
    _require(all(values == set(range(len(values))) for values in indices.values()),
             "ENTITY_TYPE_INDEX_DENOMINATOR")
    return result


def _topology(values, entities, limits):
    supplied = _rows(values, limits.max_occurrences, "OCCURRENCE_BUDGET_OR_SCHEMA")
    _require(bool(supplied), "EMPTY_OCCURRENCE_TREE")
    rows, depths = [], []
    counts = {identifier: 0 for identifier in entities}
    children = [[] for _ in supplied]
    for index, value in enumerate(supplied):
        row = _rows(value, 3, "OCCURRENCE_SCHEMA")
        _require(len(row) == 3, "OCCURRENCE_SCHEMA")
        identifier = _integer(row[0], 0, len(entities)-1, "OCCURRENCE_ENTITY_ID")
        parent = _integer(row[1], -1, index-1, "OCCURRENCE_PARENT_ORDER")
        orientation = _integer(row[2], 0, 3, "OCCURRENCE_ORIENTATION_ENUM")
        _require((index == 0) == (parent == -1), "EXACT_ONE_OCCURRENCE_ROOT")
        depth = 0 if parent == -1 else depths[parent]+1
        _require(depth <= limits.max_depth, "OCCURRENCE_DEPTH_BUDGET")
        if parent != -1:
            parent_kind = entities[rows[parent][0]][0]
            _require(entities[identifier][0] in _CHILD_TYPES[parent_kind],
                     "UNSUPPORTED_TYPE_TRANSITION")
            children[parent].append(identifier)
        rows.append((identifier, parent, orientation))
        depths.append(depth)
        counts[identifier] += 1
    _require(all(counts[key] == row[2] for key, row in entities.items()),
             "ENTITY_OCCURRENCE_COUNT_MISMATCH")
    # A native identity has one immediate-child multiset. Cumulative orientation
    # may reverse its uses/vertices; it must NOT change this identity signature.
    signatures = {}
    for index, row in enumerate(rows):
        signature = tuple(sorted(children[index]))  # multiplicity is retained
        _require(row[0] not in signatures or signatures[row[0]] == signature,
                 "INCONSISTENT_NATIVE_CHILD_IDENTITIES")
        signatures[row[0]] = signature
    expected, steps = set(), 0
    for identifier, parent, _ in rows:
        seen = {identifier}
        while parent != -1:
            steps += 1
            _require(steps <= limits.max_ancestor_steps, "ANCESTOR_WORK_BUDGET")
            owner = rows[parent][0]
            _require(owner not in seen, "RECURSIVE_NATIVE_IDENTITY")
            seen.add(owner)
            if (entities[identifier][0], entities[owner][0]) in _CONTEXT_TYPES:
                expected.add((identifier, owner))
                _require(len(expected) <= limits.max_expected_context_pairs,
                         "EXPECTED_CONTEXT_PAIR_BUDGET")
            parent = rows[parent][1]
    return expected, len(rows), steps


def _statuses(value, limits, counters):
    supplied = _rows(value, limits.max_statuses_per_list, "STATUS_LIST_BUDGET_OR_SCHEMA")
    counters["statuses"] += len(supplied)
    _require(counters["statuses"] <= limits.max_total_status_values, "TOTAL_STATUS_BUDGET")
    values = tuple(_integer(code, 0, 255, "STATUS_CODE") for code in supplied)
    _require(len(set(values)) == len(values), "DUPLICATE_STATUS_CODE")
    _require(not (0 in values and len(values) != 1), "MIXED_CLEAR_ERROR_STATUS")
    return tuple(sorted(values))


def _detail_metadata(row):
    """Bound ignored diagnostic metadata; do not interpret it as geometry proof."""
    for name in ("detail_complete", "degenerated", "same_parameter", "same_range",
                 "native_tolerance_numeric_valid", "subtree_valid"):
        if name in row:
            _require(type(row[name]) is bool, "DETAIL_BOOLEAN_SCHEMA")
    if "geometry_type" in row:
        _require(type(row["geometry_type"]) is str and 0 < len(row["geometry_type"]) <= 128,
                 "DETAIL_GEOMETRY_NAME_SCHEMA")
    if "native_tolerance_mm" in row:
        value = row["native_tolerance_mm"]
        _require(value is None or (type(value) in (int, float)
                 and math.isfinite(value) and value >= 0), "DETAIL_TOLERANCE_SCHEMA")
        if "native_tolerance_numeric_valid" in row:
            _require(row["native_tolerance_numeric_valid"] == (value is not None),
                     "DETAIL_TOLERANCE_FLAG_CONTRADICTION")
    errors = _rows(row["errors"], 4, "DETAIL_ERRORS_SCHEMA")
    _require(all(type(error) is str and 0 < len(error) <= 160 for error in errors),
             "DETAIL_ERRORS_SCHEMA")
    return (row["detail_complete"] and not errors and "subtree_valid" in row
            and row.get("native_tolerance_numeric_valid", True))


def _details(values, entities, expected, limits):
    supplied = _rows(values, limits.max_details, "DETAIL_BUDGET_OR_SCHEMA")
    details, contexts, counters = {}, {}, dict(statuses=0, contexts=0)
    for value in supplied:
        row = _record(value, _DETAIL_REQUIRED, _DETAIL_OPTIONAL, "DETAIL_SCHEMA")
        identifier = _integer(row["native_id"], 0, len(entities)-1, "DETAIL_ENTITY_ID")
        _require(entities[identifier][0] not in ("COMPOUND", "COMPSOLID"),
                 "CONTAINER_DETAIL_UNSUPPORTED")
        _require(identifier not in details, "DUPLICATE_DETAIL_ID")
        usable = _detail_metadata(row)
        codes = _statuses(row["self_status_codes"], limits, counters)
        _require(not (codes and codes != (0,) and row.get("subtree_valid") is True),
                 "SELF_SUBTREE_CONTRADICTION")
        details[identifier] = (codes, usable, row.get("subtree_valid"))
        for value in _rows(row["contexts"], limits.max_observed_context_pairs,
                           "CONTEXT_LIST_BUDGET_OR_SCHEMA"):
            counters["contexts"] += 1
            _require(counters["contexts"] <= limits.max_observed_context_pairs,
                     "OBSERVED_CONTEXT_PAIR_BUDGET")
            pair_row = _rows(value, 2, "CONTEXT_RECORD_SCHEMA")
            _require(len(pair_row) == 2, "CONTEXT_RECORD_SCHEMA")
            owner = _integer(pair_row[0], 0, len(entities)-1, "CONTEXT_OWNER_ID")
            pair = (identifier, owner)
            _require(pair not in contexts, "DUPLICATE_OBSERVED_CONTEXT_PAIR")
            _require(pair in expected, "UNEXPECTED_CONTEXT_PAIR")
            contexts[pair] = (_statuses(pair_row[1], limits, counters), usable)
    return details, contexts, counters


def _disposition(codes, usable):
    if codes and codes != (0,):
        return "ERROR"  # Retain a known negative even when other evidence is missing.
    return "CLEAR" if codes == (0,) and usable else "MISSING"


def validate_native_context_coverage_v3(entities, occurrences, entity_details, *,
                                      native_valid, source_report_sha256=None,
                                      limits=NativeContextCoverageLimitsV3()):
    """Account for every required self/context witness in supplied arrays.

    Return COVERAGE_COMPLETE (possibly with errors), COVERAGE_INCOMPLETE, or
    UNAVAILABLE. Dispositions are CLEAR/ERROR/MISSING; missing cached evidence
    is not a native error. Malformed/over-budget input clears all result rows
    and counts. Every admission flag is false, even for fully clear coverage.
    Input arrays must be stable during the call; no input object is returned.
    """
    hash_valid = (type(source_report_sha256) is str and len(source_report_sha256) == 64
                  and all(char in "0123456789abcdef" for char in source_report_sha256))
    base = dict(schema="NATIVE_CONTEXT_COVERAGE_V3", status="UNAVAILABLE", reasons=[],
                source_report_sha256=source_report_sha256 if hash_valid else None,
                source_binding="CALLER_DECLARED_DIGEST_NOT_AUTHENTICATED_BYTES_OR_NATIVE_IDENTITY",
                native_valid=native_valid if type(native_valid) is bool else None,
                native_valid_scope="CALLER_REPORTED_OBSERVATION_NOT_RECOMPUTED",
                coverage_complete=False, counts=None, self_dispositions=[],
                context_dispositions=[], local_admission_allowed=False,
                global_admission_allowed=False, production_use_allowed=False,
                observed_bend_count=None, feature_claims_produced=False)
    try:
        limits = _validated_limits(limits)
        _require(native_valid is None or type(native_valid) is bool, "NATIVE_VALID_SCHEMA")
        _require(source_report_sha256 is None or hash_valid, "SOURCE_REPORT_DIGEST_SCHEMA")
        identities = _entities(entities, limits)
        expected, occurrence_count, steps = _topology(occurrences, identities, limits)
        details, observed, counters = _details(entity_details, identities, expected, limits)
        self_rows, context_rows = [], []
        self_counts = dict(CLEAR=0, ERROR=0, MISSING=0)
        context_counts = dict(CLEAR=0, ERROR=0, MISSING=0)
        incomplete_details = sum(not row[1] for row in details.values())
        subtree_invalid = sum(row[2] is False for row in details.values())
        for identifier in sorted(identities):
            kind = identities[identifier][0]
            if kind in ("COMPOUND", "COMPSOLID"):
                continue
            codes, usable, subtree = details.get(identifier, ((), False, None))
            disposition = _disposition(codes, usable)
            self_counts[disposition] += 1
            self_rows.append(dict(native_id=identifier, type=kind, disposition=disposition,
                                  status_codes=list(codes), subtree_valid=subtree))
        for identifier, owner in sorted(expected):
            codes, usable = observed.get((identifier, owner), ((), False))
            disposition = _disposition(codes, usable)
            context_counts[disposition] += 1
            context_rows.append(dict(native_id=identifier, context_id=owner,
                relation=identities[identifier][0]+"->"+identities[owner][0],
                disposition=disposition, status_codes=list(codes)))
        _require(not (native_valid is True and
                 (self_counts["ERROR"] or context_counts["ERROR"] or subtree_invalid)),
                 "GLOBAL_NATIVE_OBSERVATION_CONTRADICTION")
        complete = not (self_counts["MISSING"] or context_counts["MISSING"] or incomplete_details)
        reasons = []
        if self_counts["MISSING"]:
            reasons.append("MISSING_SELF_EVIDENCE")
        if context_counts["MISSING"]:
            reasons.append("MISSING_CONTEXT_EVIDENCE")
        if incomplete_details:
            reasons.append("SOURCE_DETAIL_INCOMPLETE")
        return {**base, "status": "COVERAGE_COMPLETE" if complete else "COVERAGE_INCOMPLETE",
                "reasons": reasons, "coverage_complete": complete,
                "counts": dict(entities=len(identities), occurrences=occurrence_count,
                    required_self_records=len(self_rows), observed_detail_records=len(details),
                    expected_context_pairs=len(expected), observed_context_pairs=len(observed),
                    self=self_counts, contexts=context_counts,
                    incomplete_details=incomplete_details, subtree_invalid=subtree_invalid,
                    status_values=counters["statuses"], ancestor_steps=steps),
                "self_dispositions": self_rows, "context_dispositions": context_rows}
    except _CoverageUnavailable as error:
        return {**base, "reasons": [str(error)]}
    except Exception:
        return {**base, "reasons": ["VALIDATION_EXCEPTION"]}
