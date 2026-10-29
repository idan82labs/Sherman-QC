"""Shape-only, opt-in cached native-context observation; never admission.

No CLI, physical reader, cache population or product activation. Native imports
are lazy. External audited supervision must enforce time/RSS/output custody.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType


_READ_ORDERS = ("KEYED_THEN_ITERATOR", "ITERATOR_THEN_KEYED")
_SOURCE_BINDINGS = (
    ("scripts/v2_native_validity_diagnostic.py",
     "e2767721c5b6c1bdf02b77cb42d4c6c2cbc68402b6e63407afecfb92d8cfca91"),
    ("backend/cad_guided_native_context_coverage_v3.py",
     "e9a366cf36106495dfd4d8563f639af8ba185f9113fea0ac40ad030d3c0b4eeb"),
)
_MAX_SOURCE_BYTES = 64 * 1024
_MAX_CONTEXTS = 4096
_MAX_CONTEXTS_PER_ENTITY = 64
_MAX_STATUS_VALUES = 32768  # Self + keyed + iterator, including both copies.
_MAX_REPORT_BYTES = 768 * 1024


class _CollectionStop(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise _CollectionStop(reason)


def _load_bound_source(relative_path, expected_digest):
    """Execute only the verified bytes of a named inert source dependency."""
    path = Path(__file__).resolve().parents[1] / relative_path
    with path.open("rb") as stream:
        source = stream.read(_MAX_SOURCE_BYTES + 1)
    _require(len(source) <= _MAX_SOURCE_BYTES, "DEPENDENCY_SOURCE_BYTE_BUDGET")
    _require(hashlib.sha256(source).hexdigest() == expected_digest,
             "DEPENDENCY_SOURCE_BINDING_MISMATCH")
    name = "_context_v3_bound_" + expected_digest
    module = ModuleType(name)
    module.__file__ = str(path)
    previous = sys.modules.get(name)
    sys.modules[name] = module  # Dataclass decoration needs its defining module.
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def _dependencies():
    legacy, coverage = (_load_bound_source(*binding) for binding in _SOURCE_BINDINGS)
    return legacy, coverage.validate_native_context_coverage_v3


def _make_analyzer(solid):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return BRepCheck_Analyzer(solid, True, False, True)


def _native_status_type():
    """Capture the concrete binding enum, never a scalar-coercion contract."""
    from OCP.BRepCheck import BRepCheck_NoError
    enum_type = type(BRepCheck_NoError)
    _require(enum_type not in (bool, int, float, complex, str, bytes),
             "NATIVE_STATUS_ENUM_TYPE")
    _require(int(BRepCheck_NoError) == 0 and
             str(BRepCheck_NoError).split(".")[-1] == "BRepCheck_NoError",
             "NATIVE_NO_ERROR_ENUM_CONTRACT")
    return enum_type


def _base(read_order):
    return dict(schema="NATIVE_CONTEXT_COLLECTION_V3", status="UNAVAILABLE",
        scope="SHAPE_ONLY_CACHED_OBSERVATION_NOT_NATIVE_OR_LOCAL_ADMISSION",
        reasons=[], read_order=read_order if read_order in _READ_ORDERS else None,
        source_bindings=[dict(path=p, sha256=h) for p, h in _SOURCE_BINDINGS],
        source_binding_scope="REVIEWED_SOURCE_BYTES_NOT_RUNTIME_OR_SHAPE_AUTHENTICITY",
        native_identity_scope="THIS_SUPPLIED_SHAPE_INVOCATION_ONLY",
        native_coordinate_scope="SUPPLIED_NATIVE_FRAME_UNITS_NOT_ESTABLISHED_HERE",
        inherited_tolerance_field_scope="native_tolerance_mm_IS_LEGACY_NAME_VALUES_ARE_UNVERIFIED_NATIVE_UNITS",
        native_analyzer_count=0, native_valid=None,
        analyzer_options=dict(geometric_controls=True, parallel=False, exact_method=True),
        topology_complete=False, read_sequence_complete=False,
        keyed_reads_complete=False, iterator_traversal_complete=False,
        channels_consistent=False, entities=[], occurrences=[], owner_path=[],
        self_details=[], keyed_observations=[], iterator_observations=[],
        cache_queries=[], channel_details={}, coverage={}, comparison=None,
        counters=None, local_admission_allowed=False, global_admission_allowed=False,
        production_use_allowed=False, observed_bend_count=None,
        feature_claims_produced=False, physical_generalization_claim=False,
        external_supervision_required=dict(wall_seconds=120, cpu_seconds=90,
            sampled_process_tree_rss_bytes=1024**3, combined_output_bytes=1024**2))


def _clear(report, reason):
    base = _base(report.get("read_order"))
    base.update(native_analyzer_count=report.get("native_analyzer_count", 0),
                native_valid=report.get("native_valid"), reasons=[reason])
    return base


def _bounded(report):
    try:
        encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) <= _MAX_REPORT_BYTES:
            return report
    except Exception:
        return _clear(report, "REPORT_ENCODING_FAILURE")
    return _clear(report, "REPORT_BYTE_BUDGET")


def _codes(values, legacy, names, counters, status_type):
    def counted_values():
        for value in values:
            counters["status_values"] += 1
            _require(counters["status_values"] <= _MAX_STATUS_VALUES,
                     "TOTAL_STATUS_VALUE_BUDGET")
            _require(type(value) is status_type, "NATIVE_STATUS_ENUM_TYPE")
            try:
                int(value), str(value)
            except Exception:
                raise _CollectionStop("NATIVE_STATUS_ENUM_CONVERSION") from None
            yield value

    try:
        codes, complete = legacy._status_codes(counted_values(), names)
    except legacy.DiagnosticStop:
        raise _CollectionStop("NATIVE_STATUS_ENUM_CONTRACT") from None
    _require(complete or not codes, "NATIVE_STATUS_LIST_TRUNCATED")
    _require(type(codes) is list and len(codes) <= 32, "NATIVE_STATUS_SCHEMA")
    _require(all(type(code) is int and 0 <= code <= 255 for code in codes),
             "NATIVE_STATUS_CODE")
    _require(len(set(codes)) == len(codes), "DUPLICATE_NATIVE_STATUS")
    _require(not (0 in codes and len(codes) != 1), "MIXED_NATIVE_CLEAR_ERROR")
    return sorted(codes)


def _self_details(shapes, entities, analyzer, legacy, names, counters, status_type):
    rows, results = [], {}
    for shape, entity in zip(shapes, entities):
        if entity["type"] in ("COMPOUND", "COMPSOLID"):
            continue
        row = dict(native_id=entity["native_id"], self_status_codes=[],
                   detail_complete=False, errors=[])
        rows.append(row)
        try:
            row.update(legacy._properties(shape, entity["type"]))
        except legacy.DiagnosticStop:
            raise _CollectionStop("NATIVE_PROPERTY_CONTRACT") from None
        except Exception:
            row["errors"].append("PROPERTY_READ_ERROR")
        try:
            value = analyzer.IsValid(shape)
            _require(type(value) is bool, "NATIVE_SUBTREE_BOOLEAN_SCHEMA")
            row["subtree_valid"] = value
        except _CollectionStop:
            raise
        except Exception:
            row["errors"].append("SUBTREE_READ_ERROR")
        try:
            result = analyzer.Result(shape)
            _require(result is not None, "NATIVE_RESULT_UNAVAILABLE")
            results[entity["native_id"]] = result
            row["self_status_codes"] = _codes(result.Status(), legacy, names, counters, status_type)
        except _CollectionStop as error:
            if str(error) != "NATIVE_RESULT_UNAVAILABLE":
                raise
            row["errors"].append("SELF_STATUS_READ_ERROR")
        except Exception:
            row["errors"].append("SELF_STATUS_READ_ERROR")
        row["detail_complete"] = bool(row["self_status_codes"]) and not row["errors"]
    return rows, results


def _keyed(expected, rows, results, shapes, legacy, names, counters, status_type):
    observations, queries = [], []
    outcomes = {row["native_id"]: [True, None] for row in rows}
    for identifier, owner in expected:
        result = results.get(identifier)
        query = [identifier, owner, "RESULT_UNAVAILABLE", "NOT_CALLED"]
        queries.append(query)
        if result is None:
            outcomes[identifier] = [False, "RESULT_UNAVAILABLE"]
            continue
        try:
            counters["cache_membership_calls"] += 1
            present = result.IsStatusOnShape(shapes[owner])
            _require(type(present) is bool, "NATIVE_CACHE_MEMBERSHIP_SCHEMA")
            query[2] = "PRESENT" if present else "ABSENT"
        except _CollectionStop:
            raise
        except Exception:
            query[2] = "ERROR"
            outcomes[identifier] = [False, "CHANNEL_READ_ERROR"]
            continue
        if not present:
            continue  # Missing cache is not calculated or replaced with NoError.
        try:
            counters["keyed_status_calls"] += 1
            codes = _codes(result.StatusOnShape(shapes[owner]), legacy, names, counters, status_type)
            observations.append([identifier, owner, codes])
            query[3] = "READ" if codes else "EMPTY"
        except _CollectionStop:
            raise
        except Exception:
            query[3] = "ERROR"
            outcomes[identifier] = [False, "CHANNEL_READ_ERROR"]
    return observations, queries, outcomes


def _iterator(rows, results, mapping, legacy, names, counters, status_type):
    observations, outcomes, seen = [], {}, set()
    for row in rows:
        identifier = row["native_id"]
        result = results.get(identifier)
        outcomes[identifier] = [result is not None, None if result is not None else "RESULT_UNAVAILABLE"]
        if result is None:
            continue
        count = 0
        try:
            result.InitContextIterator()
            while True:
                more = result.MoreShapeInContext()
                _require(type(more) is bool, "NATIVE_ITERATOR_BOOLEAN_SCHEMA")
                if not more:
                    break
                _require(count < _MAX_CONTEXTS_PER_ENTITY and len(observations) < _MAX_CONTEXTS,
                         "ITERATOR_CONTEXT_BUDGET")
                context = result.ContextualShape()
                owner = mapping.FindIndex(context)
                _require(type(owner) is int and owner > 0, "ITERATOR_CONTEXT_OUTSIDE_TOPOLOGY")
                pair = (identifier, owner - 1)
                _require(pair not in seen, "DUPLICATE_ITERATOR_CONTEXT")
                seen.add(pair)
                _require(counters["iterator_status_calls"] < _MAX_CONTEXTS,
                         "ITERATOR_STATUS_CALL_BUDGET")
                counters["iterator_status_calls"] += 1
                codes = _codes(result.StatusOnShape(), legacy, names, counters, status_type)
                observations.append([*pair, codes])
                count += 1
                result.NextShapeInContext()
        except _CollectionStop:
            raise
        except Exception:
            outcomes[identifier] = [False, "CHANNEL_READ_ERROR"]
    return observations, outcomes


def _channel_coverage(entities, occurrences, self_rows, observations, outcomes, native_valid, validate):
    grouped = {row["native_id"]: [] for row in self_rows}
    for identifier, owner, codes in observations:
        grouped[identifier].append([owner, codes])
    details = []
    for row in self_rows:
        complete, error = outcomes[row["native_id"]]
        details.append({**row, "contexts": grouped[row["native_id"]],
            "detail_complete": row["detail_complete"] and complete,
            "errors": [*row["errors"], *([error] if error else [])]})
    result = validate(entities, occurrences, details, native_valid=native_valid)
    return {key: result[key] for key in ("status", "reasons", "coverage_complete", "counts",
            "local_admission_allowed", "global_admission_allowed", "production_use_allowed")}


def _comparison(expected, keyed, iterator):
    left = {(a, b): tuple(codes) for a, b, codes in keyed}
    right = {(a, b): tuple(codes) for a, b, codes in iterator}
    expected = set(expected)
    return dict(expected_pairs=len(expected),
        matching_pairs=sum(left[pair] == right[pair] for pair in left.keys() & right.keys()),
        keyed_only=[list(pair) for pair in sorted(left.keys() - right.keys())],
        iterator_only=[list(pair) for pair in sorted(right.keys() - left.keys())],
        differing_status_pairs=[list(pair) for pair in sorted(left.keys() & right.keys())
                                if left[pair] != right[pair]],
        missing_both=[list(pair) for pair in sorted(expected - left.keys() - right.keys())],
        unexpected_iterator_pairs=[list(pair) for pair in sorted(right.keys() - expected)])


def collect_cached_native_contexts_v3(shape, *, allow_ocp=False,
                                     read_order="KEYED_THEN_ITERATOR"):
    """Observe one caller-supplied native shape using exactly one analyzer.

    Shape custody, native frame/units and external supervision are the caller's
    responsibility. Both read channels are diagnostics; neither grants admission.
    """
    report = _base(read_order if type(read_order) is str else None)
    try:
        _require(type(allow_ocp) is bool and allow_ocp, "EXPLICIT_OCP_OPT_IN_REQUIRED")
        _require(type(read_order) is str and read_order in _READ_ORDERS, "READ_ORDER_SCHEMA")
        legacy, validate = _dependencies()
        mapping, shapes, entities, occurrences, _ = legacy._topology(shape)
        denominator = validate(entities, occurrences, [], native_valid=None)
        _require(denominator["status"] != "UNAVAILABLE", "TOPOLOGY_DENOMINATOR_UNAVAILABLE")
        _require(denominator["counts"]["required_self_records"] <= 1024,
                 "SELF_DETAIL_DENOMINATOR_BUDGET")
        expected = tuple((row["native_id"], row["context_id"])
                         for row in denominator["context_dispositions"])
        _require(len(expected) <= _MAX_CONTEXTS, "EXPECTED_CONTEXT_BUDGET")
        solid, owner_path = legacy._owner_solid(shape)
        status_type = _native_status_type()
        report["native_analyzer_count"] = 1
        analyzer = _make_analyzer(solid)
        native_valid = analyzer.IsValid()
        _require(type(native_valid) is bool, "NATIVE_GLOBAL_BOOLEAN_SCHEMA")
        report["native_valid"] = native_valid
        names, counters = {}, dict(status_values=0, cache_membership_calls=0,
                                   keyed_status_calls=0, iterator_status_calls=0)
        self_rows, results = _self_details(shapes, entities, analyzer, legacy, names, counters, status_type)
        if read_order == "KEYED_THEN_ITERATOR":
            keyed, queries, key_outcomes = _keyed(expected, self_rows, results, shapes, legacy, names, counters, status_type)
            iterator, iter_outcomes = _iterator(self_rows, results, mapping, legacy, names, counters, status_type)
        else:
            iterator, iter_outcomes = _iterator(self_rows, results, mapping, legacy, names, counters, status_type)
            keyed, queries, key_outcomes = _keyed(expected, self_rows, results, shapes, legacy, names, counters, status_type)
        comparison = _comparison(expected, keyed, iterator)
        consistent = not any(comparison[name] for name in
            ("keyed_only", "iterator_only", "differing_status_pairs", "unexpected_iterator_pairs"))
        key_complete = all(value[0] for value in key_outcomes.values())
        iter_complete = all(value[0] for value in iter_outcomes.values())
        read_complete = key_complete and iter_complete and all(row["detail_complete"] for row in self_rows)
        coverages = {name: _channel_coverage(entities, occurrences, self_rows, observations,
                      outcomes, report["native_valid"], validate)
                     for name, observations, outcomes in
                     (("keyed", keyed, key_outcomes), ("iterator", iterator, iter_outcomes))}
        _require(all(value["status"] != "UNAVAILABLE" for value in coverages.values()),
                 "CHANNEL_COVERAGE_UNAVAILABLE")
        report.update(status="COLLECTED" if read_complete else "PARTIAL_COLLECTION",
            topology_complete=True, read_sequence_complete=read_complete,
            keyed_reads_complete=key_complete, iterator_traversal_complete=iter_complete,
            channels_consistent=consistent, entities=entities, occurrences=occurrences,
            owner_path=owner_path, self_details=self_rows, keyed_observations=keyed,
            iterator_observations=iterator, cache_queries=queries,
            channel_details={name: [[identifier, *values] for identifier, values in sorted(outcomes.items())]
                             for name, outcomes in (("keyed", key_outcomes), ("iterator", iter_outcomes))},
            coverage=coverages, comparison=comparison, counters=counters, status_names=names)
        if not read_complete:
            report["reasons"].append("NATIVE_READ_SEQUENCE_INCOMPLETE")
        if not consistent:
            report["reasons"].append("READ_CHANNEL_DISAGREEMENT")
        for name, value in coverages.items():
            if not value["coverage_complete"]:
                report["reasons"].append(name.upper() + "_COVERAGE_NOT_COMPLETE")
    except _CollectionStop as error:
        report = _clear(report, str(error))
    except Exception:
        report = _clear(report, "COLLECTION_EXCEPTION")
    return _bounded(report)
