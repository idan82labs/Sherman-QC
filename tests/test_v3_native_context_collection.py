"""Independent cached-context observation controls, no physical inputs.

Ordinary collection is pure/mocked. The one native matrix is separately opt-in,
constructs eight fresh shapes, and retains each full two-channel report in a
parent-owned artifact destination rather than writing report-sized stdout.
"""
from copy import deepcopy
from enum import Enum, IntEnum
import hashlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("invented_context_collection_v3", ROOT / "scripts/v3_native_context_collection.py")
collection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collection)
READ_ORDERS = ("KEYED_THEN_ITERATOR", "ITERATOR_THEN_KEYED")
NATIVE_OPT_IN = "SHERMAN_RUN_NATIVE_CONTEXT_COLLECTION_V3"
ARTIFACT_ENV = "SHERMAN_NATIVE_CONTEXT_COLLECTION_V3_ARTIFACT_DIRECTORY"
NATIVE_CASES = ("box", "open_shell", "cylinder", "reversed_box")
CASE_BYTES, TOTAL_BYTES, SUMMARY_BYTES = 768*1024, 6*1024**2, 16*1024
FakeStatus = IntEnum("FakeStatus", {"CODE_"+str(i): i for i in range(256)})
CLEAR, ERROR = FakeStatus(0), FakeStatus(1)
EXPECTED_PAIRS = ((1, 0), (4, 2), (5, 3), (6, 2), (6, 3),
                  (7, 2), (7, 3), (7, 6), (8, 2), (8, 3), (8, 6))


def _symbolic_tree(seam=False):
    kinds = ("SOLID", "SHELL", "FACE", "FACE", "WIRE", "WIRE", "EDGE", "VERTEX", "VERTEX")
    indices, counts = (0, 0, 0, 1, 0, 1, 0, 0, 1), [1, 1, 1, 1, 1, 1, 2, 2, 2]
    occurrences = [[0, -1, 0], [1, 0, 0], [2, 1, 0], [4, 2, 0],
                   [6, 3, 0], [7, 4, 0], [8, 4, 1], [3, 1, 1],
                   [5, 7, 0], [6, 8, 1], [8, 9, 0], [7, 9, 1]]
    if seam:
        occurrences.extend([[6, 3, 2], [8, 12, 2], [7, 12, 3]])
        counts[6:] = [3, 3, 3]
    entities = [dict(native_id=i, type=kind, type_index=indices[i],
                     occurrence_count=counts[i], location_is_identity=True)
                for i, kind in enumerate(kinds)]
    return entities, occurrences


class _Result:
    def __init__(self, identifier, owners, events):
        self.identifier, self.events = identifier, events
        self.self_values = [CLEAR]
        self.keyed = {owner: [CLEAR] for owner in owners}
        self.iterator = [[owner, [CLEAR]] for owner in owners]
        self.faults, self.cursor = {}, 0

    def _read(self, name, owner=None):
        self.events.append((name, self.identifier, owner))
        value = self.faults.get((name, owner), self.faults.get(name))
        if isinstance(value, BaseException):
            raise value
        return value

    def Status(self):
        self._read("self")
        return self.self_values

    def IsStatusOnShape(self, owner):
        override = self._read("membership", owner.identifier)
        return owner.identifier in self.keyed if override is None else override

    def StatusOnShape(self, *args):
        if args:
            owner = args[0].identifier
            self._read("keyed", owner)
            assert owner in self.keyed, "Absent cache getter must never be called"
            return self.keyed[owner]
        owner, values = self.iterator[self.cursor]
        self._read("iterator_status", owner)
        return values

    def InitContextIterator(self):
        self._read("iterator_init")
        self.cursor = 0

    def MoreShapeInContext(self):
        override = self._read("iterator_more")
        return self.cursor < len(self.iterator) if override is None else override

    def ContextualShape(self):
        self._read("iterator_context")
        return SimpleNamespace(identifier=self.iterator[self.cursor][0])

    def NextShapeInContext(self):
        self._read("iterator_next")
        self.cursor += 1

    def __getattr__(self, name):
        if name in ("InContext", "Minimum", "Blind", "SetStatus", "SetFailStatus"):
            pytest.fail("Forbidden cache-mutating method: "+name)
        raise AttributeError(name)


@pytest.fixture
def mocked(monkeypatch):
    # These exact inert source dependencies are read/loaded, but their native
    # topology/property operations are replaced before the public call.
    legacy, validate = collection._dependencies()
    entities, occurrences = _symbolic_tree()
    events, shapes = [], [SimpleNamespace(identifier=i) for i in range(9)]
    results = {i: _Result(i, [owner for child, owner in EXPECTED_PAIRS if child == i], events)
               for i in range(9)}
    state = SimpleNamespace(entities=entities, occurrences=occurrences, events=events,
        shapes=shapes, results=results, native_valid=True, subtree={i: True for i in range(9)},
        result_errors={}, properties={}, analyzers=0, owner_path=[], legacy=legacy, validate=validate)

    class Analyzer:
        def IsValid(self, *args):
            events.append(("subtree" if args else "global", args[0].identifier if args else None))
            value = state.subtree[args[0].identifier] if args else state.native_valid
            if isinstance(value, BaseException): raise value
            return value
        def Result(self, shape):
            events.append(("result", shape.identifier))
            value = state.result_errors.get(shape.identifier, state.results[shape.identifier])
            if isinstance(value, BaseException): raise value
            return value

    def topology(shape):
        events.append(("topology",))
        mapping = SimpleNamespace(FindIndex=lambda value: value.identifier+1 if 0 <= value.identifier < 9 else 0)
        return mapping, shapes, deepcopy(state.entities), deepcopy(state.occurrences), {}
    def properties(shape, kind):
        value = state.properties.get(shape.identifier, {})
        if isinstance(value, BaseException): raise value
        return dict(value)
    def owner(shape):
        events.append(("owner",))
        return shapes[0], list(state.owner_path)
    def analyzer(shape):
        events.append(("analyzer",))
        state.analyzers += 1
        return Analyzer()
    def enum_type():
        events.append(("enum_type",))
        return FakeStatus
    monkeypatch.setattr(legacy, "_topology", topology)
    monkeypatch.setattr(legacy, "_properties", properties)
    monkeypatch.setattr(legacy, "_owner_solid", owner)
    monkeypatch.setattr(collection, "_dependencies", lambda: (legacy, validate))
    monkeypatch.setattr(collection, "_make_analyzer", analyzer)
    monkeypatch.setattr(collection, "_native_status_type", enum_type)
    state.call = lambda order=READ_ORDERS[0]: collection.collect_cached_native_contexts_v3(
        shapes[0], allow_ocp=True, read_order=order)
    return state


def _no_admission(report):
    assert all(report[key] is False for key in ("local_admission_allowed", "global_admission_allowed",
        "production_use_allowed", "feature_claims_produced", "physical_generalization_claim"))
    assert report["observed_bend_count"] is None


def _unavailable(report, reason=None):
    assert report["status"] == "UNAVAILABLE", report
    for key in ("entities", "occurrences", "owner_path", "self_details", "keyed_observations",
                "iterator_observations", "cache_queries"):
        assert report[key] == []
    assert report["channel_details"] == report["coverage"] == {}
    assert report["comparison"] is report["counters"] is None
    assert not report["topology_complete"] and not report["read_sequence_complete"]
    if reason is not None: assert report["reasons"] == [reason]
    _no_admission(report)


def _clear_mock(report, seam=False):
    assert report["status"] == "COLLECTED" and report["reasons"] == []
    assert report["read_sequence_complete"] and report["channels_consistent"]
    assert report["native_analyzer_count"] == 1
    expected = [[a, b, [0]] for a, b in EXPECTED_PAIRS]
    assert report["keyed_observations"] == report["iterator_observations"] == expected
    assert report["cache_queries"] == [[a, b, "PRESENT", "READ"] for a, b in EXPECTED_PAIRS]
    assert report["comparison"] == dict(expected_pairs=11, matching_pairs=11, keyed_only=[],
        iterator_only=[], differing_status_pairs=[], missing_both=[], unexpected_iterator_pairs=[])
    for channel in ("keyed", "iterator"):
        row = report["coverage"][channel]
        assert row["status"] == "COVERAGE_COMPLETE" and row["coverage_complete"]
        assert row["counts"]["expected_context_pairs"] == row["counts"]["observed_context_pairs"] == 11
        assert row["counts"]["occurrences"] == (15 if seam else 12)
        assert row["counts"]["ancestor_steps"] == (53 if seam else 39)
        assert row["counts"]["status_values"] == 20
    assert report["counters"] == dict(status_values=31, cache_membership_calls=11,
                                      keyed_status_calls=11, iterator_status_calls=11)
    _no_admission(report)


@pytest.mark.parametrize("order", READ_ORDERS)
@pytest.mark.parametrize("seam", (False, True))
def test_symbolic_incidence_oracle_one_analyzer_and_both_whole_read_orders(mocked, order, seam):
    if seam: mocked.entities, mocked.occurrences = _symbolic_tree(seam=True)
    report = mocked.call(order)
    _clear_mock(report, seam)
    assert mocked.analyzers == 1
    labels = [event[0] for event in mocked.events]
    assert labels.index("topology") < labels.index("analyzer") < labels.index("global")
    assert labels.count("analyzer") == labels.count("global") == 1
    keyed = [i for i, label in enumerate(labels) if label in ("membership", "keyed")]
    iterator = [i for i, label in enumerate(labels) if label.startswith("iterator_")]
    assert (max(keyed) < min(iterator)) if order == READ_ORDERS[0] else (max(iterator) < min(keyed))


@pytest.mark.parametrize("allow", (False, None, 0, 1, "1"))
def test_explicit_exact_opt_in_before_any_source_or_native_call(monkeypatch, allow):
    monkeypatch.setattr(collection, "_dependencies", lambda: pytest.fail("No dependency load without exact opt-in"))
    _unavailable(collection.collect_cached_native_contexts_v3(object(), allow_ocp=allow), "EXPLICIT_OCP_OPT_IN_REQUIRED")


@pytest.mark.parametrize("order", (None, "", "KEYED", 1, [], {}))
def test_read_order_admission_precedes_dependency_loading(monkeypatch, order):
    monkeypatch.setattr(collection, "_dependencies", lambda: pytest.fail("Bad order reached dependencies"))
    _unavailable(collection.collect_cached_native_contexts_v3(object(), allow_ocp=True, read_order=order), "READ_ORDER_SCHEMA")


def test_malformed_topology_denominator_precedes_native_enum_and_analyzer(mocked):
    mocked.occurrences[-1][1] = -1
    _unavailable(mocked.call(), "TOPOLOGY_DENOMINATOR_UNAVAILABLE")
    assert mocked.events == [("topology",)] and mocked.analyzers == 0


@pytest.mark.parametrize("count", (1024, 1025))
def test_self_detail_denominator_exact_boundary_before_owner_and_native(mocked, monkeypatch, count):
    # A structurally valid tree, not a geometry fixture: distinct edge leaves
    # under one wire. 1020/1021 edges are within the frozen 1024 edge cap.
    kinds = ("SOLID", "SHELL", "FACE", "WIRE")
    mocked.entities = [dict(native_id=i, type=kinds[i] if i < 4 else "EDGE",
        type_index=0 if i < 4 else i-4, occurrence_count=1, location_is_identity=True)
        for i in range(count)]
    mocked.occurrences = [[i, i-1 if i < 4 else 3, 0] for i in range(count)]
    def owner(shape):
        mocked.events.append(("owner",))
        raise collection._CollectionStop("INVENTED_POST_DENOMINATOR_STOP")
    monkeypatch.setattr(mocked.legacy, "_owner_solid", owner)
    _unavailable(mocked.call(), "INVENTED_POST_DENOMINATOR_STOP" if count == 1024
                 else "SELF_DETAIL_DENOMINATOR_BUDGET")
    assert mocked.events == ([("topology",), ("owner",)] if count == 1024 else [("topology",)])
    assert mocked.analyzers == 0


def test_normal_report_is_detached_from_inputs_and_other_reports(mocked):
    mocked.properties[8] = dict(native_tolerance_mm=0.125, native_tolerance_numeric_valid=True)
    mocked.owner_path = [0]
    first, second = mocked.call(), mocked.call()
    _clear_mock(first)
    _clear_mock(second)
    untouched = deepcopy(second)
    first["entities"][0]["type"] = "CHANGED"
    first["occurrences"][0][2] = 3
    first["owner_path"].append(9)
    first["self_details"][8]["self_status_codes"].append(9)
    first["self_details"][8]["native_tolerance_mm"] = 2.
    first["keyed_observations"][0][2].append(9)
    first["iterator_observations"][0][2].append(8)
    first["coverage"]["keyed"]["counts"]["contexts"]["CLEAR"] = 0
    first["source_bindings"][0]["sha256"] = "changed"
    assert second == untouched
    assert mocked.entities[0]["type"] == "SOLID" and mocked.occurrences[0][2] == 0
    assert mocked.owner_path == [0] and mocked.properties[8]["native_tolerance_mm"] == .125
    assert mocked.results[8].self_values == [CLEAR]
    assert all(values == [CLEAR] for result in mocked.results.values() for values in result.keyed.values())
    assert all(values == [CLEAR] for result in mocked.results.values() for _, values in result.iterator)
    assert second["inherited_tolerance_field_scope"] == (
        "native_tolerance_mm_IS_LEGACY_NAME_VALUES_ARE_UNVERIFIED_NATIVE_UNITS")


def _fake_status_binding(monkeypatch, no_error):
    package, binding = ModuleType("OCP"), ModuleType("OCP.BRepCheck")
    package.__path__ = []
    binding.BRepCheck_NoError = no_error
    package.BRepCheck = binding
    monkeypatch.setitem(sys.modules, "OCP", package)
    monkeypatch.setitem(sys.modules, "OCP.BRepCheck", binding)


@pytest.mark.parametrize("no_error", (False, 0, 0., 0j, "BRepCheck_NoError", b"BRepCheck_NoError"))
def test_native_status_reference_rejects_builtin_scalars_without_native_import(monkeypatch, no_error):
    _fake_status_binding(monkeypatch, no_error)
    with pytest.raises(collection._CollectionStop, match="^NATIVE_STATUS_ENUM_TYPE$"):
        collection._native_status_type()


def test_native_status_reference_accepts_exact_enum_contract_from_fake_binding(monkeypatch):
    class NativeStatus(Enum):
        BRepCheck_NoError = 0
        def __int__(self): return self.value
        def __str__(self): return "NativeStatus."+self.name
    _fake_status_binding(monkeypatch, NativeStatus.BRepCheck_NoError)
    assert collection._native_status_type() is NativeStatus


@pytest.mark.parametrize("mode", ("absent_keyed", "absent_both", "empty_keyed", "different_codes"))
def test_completed_reads_can_retain_missing_or_disagreeing_channels(mocked, mode):
    result = mocked.results[6]
    if mode.startswith("absent"):
        result.keyed.pop(3)
        if mode == "absent_both": result.iterator = [row for row in result.iterator if row[0] != 3]
    elif mode == "empty_keyed": result.keyed[3] = []
    else:
        result.keyed[3] = [ERROR]
        mocked.native_valid = False
    report = mocked.call()
    assert report["status"] == "COLLECTED" and report["read_sequence_complete"]
    query = next(row for row in report["cache_queries"] if row[:2] == [6, 3])
    assert query[2:] == (["ABSENT", "NOT_CALLED"] if mode.startswith("absent")
                        else ["PRESENT", "EMPTY" if mode == "empty_keyed" else "READ"])
    assert report["coverage"]["keyed"]["coverage_complete"] is (mode == "different_codes")
    assert report["coverage"]["iterator"]["coverage_complete"] is (mode != "absent_both")
    assert report["channels_consistent"] is (mode == "absent_both")
    if mode.startswith("absent"):
        assert ("keyed", 6, 3) not in mocked.events
    if mode == "absent_both": assert report["comparison"]["missing_both"] == [[6, 3]]
    if mode == "different_codes":
        assert report["comparison"]["differing_status_pairs"] == [[6, 3]]
        assert report["coverage"]["keyed"]["counts"]["contexts"]["ERROR"] == 1
    _no_admission(report)


@pytest.mark.parametrize("method", ("membership", "keyed", "iterator_init", "iterator_more",
                                    "iterator_context", "iterator_status", "iterator_next", "self"))
def test_native_read_exceptions_are_partial_not_false_clear(mocked, method):
    key = (method, 3) if method in ("membership", "keyed", "iterator_status") else method
    mocked.results[6].faults[key] = RuntimeError("invented read failure")
    report = mocked.call()
    assert report["status"] == "PARTIAL_COLLECTION" and not report["read_sequence_complete"]
    assert report["native_analyzer_count"] == 1 and report["entities"]
    assert "NATIVE_READ_SEQUENCE_INCOMPLETE" in report["reasons"]
    assert not all(row["coverage_complete"] for row in report["coverage"].values())
    _no_admission(report)


@pytest.mark.parametrize("kind", ("result_none", "result_exception", "property_exception", "subtree_exception"))
def test_missing_common_self_observation_affects_both_channels(mocked, kind):
    if kind == "result_none": mocked.result_errors[6] = None
    elif kind == "result_exception": mocked.result_errors[6] = RuntimeError("invented")
    elif kind == "property_exception": mocked.properties[6] = RuntimeError("invented")
    else: mocked.subtree[6] = RuntimeError("invented")
    report = mocked.call()
    assert report["status"] == "PARTIAL_COLLECTION"
    assert all(not row["coverage_complete"] for row in report["coverage"].values())
    if kind.startswith("result"):
        assert [row[2:] for row in report["cache_queries"] if row[0] == 6] == [["RESULT_UNAVAILABLE", "NOT_CALLED"]]*2
    _no_admission(report)


def test_legacy_property_budget_stop_is_fatal_not_downgraded(mocked):
    mocked.properties[8] = mocked.legacy.DiagnosticStop("NATIVE_GEOMETRY_NAME_BUDGET")
    report = mocked.call()
    _unavailable(report, "NATIVE_PROPERTY_CONTRACT")


@pytest.mark.parametrize("channel", ("self", "keyed", "iterator"))
@pytest.mark.parametrize("value", (False, .5, "0", b"0", 0))
def test_malformed_raw_status_types_cannot_be_coerced_to_clear(mocked, channel, value):
    if channel == "self": mocked.results[8].self_values = [value]
    elif channel == "keyed": mocked.results[8].keyed[6] = [value]
    else: mocked.results[8].iterator[-1][1] = [value]
    _unavailable(mocked.call())


def test_foreign_enum_and_arbitrary_integer_conversion_are_not_native_statuses(mocked):
    class Foreign(IntEnum): CLEAR = 0
    class Pretend:
        def __int__(self): return 0
    for value in (Foreign.CLEAR, Pretend()):
        mocked.results[8].keyed[6] = [value]
        _unavailable(mocked.call())


def test_exact_captured_enum_conversion_failure_whole_clears_late_keyed_prefix(mocked, monkeypatch):
    class NativeStatus(Enum):
        CLEAR = 0
        BROKEN = 1
        def __int__(self):
            if self is NativeStatus.BROKEN: raise ValueError("invented conversion error")
            return self.value
    monkeypatch.setattr(collection, "_native_status_type", lambda: NativeStatus)
    for result in mocked.results.values():
        result.self_values = [NativeStatus.CLEAR]
        result.keyed = {owner: [NativeStatus.CLEAR] for owner in result.keyed}
        result.iterator = [[owner, [NativeStatus.CLEAR]] for owner, _ in result.iterator]
    mocked.results[8].keyed[6] = [NativeStatus.BROKEN]
    report = mocked.call()
    _unavailable(report, "NATIVE_STATUS_ENUM_CONVERSION")
    assert ("keyed", 8, 6) in mocked.events and report["native_analyzer_count"] == 1


@pytest.mark.parametrize("values", ([CLEAR, ERROR], [ERROR, ERROR], [FakeStatus(i) for i in range(1, 34)]))
def test_mixed_duplicate_and_overlength_native_status_lists_whole_refuse(mocked, values):
    mocked.results[8].keyed[6] = values
    _unavailable(mocked.call())


@pytest.mark.parametrize("fault", ("duplicate", "foreign", "unexpected", "membership_bool", "iterator_bool", "global_bool", "subtree_bool"))
def test_late_malformed_native_structure_whole_refuses(mocked, fault):
    if fault == "duplicate": mocked.results[8].iterator.append(deepcopy(mocked.results[8].iterator[-1]))
    elif fault == "foreign": mocked.results[8].iterator.append([99, [CLEAR]])
    elif fault == "unexpected": mocked.results[8].iterator.append([0, [CLEAR]])
    elif fault == "membership_bool": mocked.results[8].faults[("membership", 6)] = 1
    elif fault == "iterator_bool": mocked.results[8].faults["iterator_more"] = 1
    elif fault == "global_bool": mocked.native_valid = 1
    else: mocked.subtree[8] = 1
    _unavailable(mocked.call())


@pytest.mark.parametrize("cap,value,complete", (("_MAX_CONTEXTS", 11, True), ("_MAX_CONTEXTS", 10, False),
    ("_MAX_CONTEXTS_PER_ENTITY", 3, True), ("_MAX_CONTEXTS_PER_ENTITY", 2, False),
    ("_MAX_STATUS_VALUES", 31, True), ("_MAX_STATUS_VALUES", 30, False)))
def test_fixed_work_caps_exact_and_one_below(mocked, monkeypatch, cap, value, complete):
    monkeypatch.setattr(collection, cap, value)
    report = mocked.call()
    if complete: _clear_mock(report)
    else: _unavailable(report)


def test_values_visited_before_iteration_exception_are_charged(mocked):
    def partial():
        yield CLEAR
        yield CLEAR
        raise RuntimeError("invented interrupted list")
    mocked.results[8].keyed[6] = partial()
    report = mocked.call()
    assert report["status"] == "PARTIAL_COLLECTION"
    assert report["counters"]["status_values"] == 32
    assert report["counters"]["keyed_status_calls"] == 11
    _no_admission(report)


def test_report_budget_clears_both_channels_after_completed_work(mocked, monkeypatch):
    monkeypatch.setattr(collection, "_MAX_REPORT_BYTES", 100)
    report = mocked.call()
    _unavailable(report, "REPORT_BYTE_BUDGET")
    assert report["native_analyzer_count"] == 1 and report["native_valid"] is True


def test_report_encoder_exception_whole_clears_completed_observations(mocked, monkeypatch):
    def fail_encoding(*args, **kwargs): raise ValueError("invented encoder failure")
    monkeypatch.setattr(collection, "json", SimpleNamespace(dumps=fail_encoding))
    report = mocked.call()
    _unavailable(report, "REPORT_ENCODING_FAILURE")
    assert report["native_analyzer_count"] == 1 and report["native_valid"] is True
    assert ("keyed", 8, 6) in mocked.events and ("iterator_status", 8, 6) in mocked.events


@pytest.mark.parametrize("failure", ("digest", "oversize", "execution"))
def test_private_bound_source_loader_rejects_wrong_bytes_and_restores_module(tmp_path, monkeypatch, failure):
    source = b"raise RuntimeError('invented import failure')\n" if failure == "execution" else b"VALUE = 7\n"
    path = tmp_path / "invented_source.py"
    path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    name = "_context_v3_bound_"+digest
    previous = object()
    monkeypatch.setitem(sys.modules, name, previous)
    if failure == "oversize": monkeypatch.setattr(collection, "_MAX_SOURCE_BYTES", len(source)-1)
    with pytest.raises((collection._CollectionStop, RuntimeError)):
        collection._load_bound_source(str(path), "0"*64 if failure == "digest" else digest)
    assert sys.modules[name] is previous


def test_exact_source_bytes_are_executed_and_previous_registration_is_restored(tmp_path, monkeypatch):
    source = b"VALUE = ('invented', 7)\n"
    path = tmp_path / "invented_source.py"
    path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    name = "_context_v3_bound_"+digest
    previous = object()
    monkeypatch.setitem(sys.modules, name, previous)
    loaded = collection._load_bound_source(str(path), digest)
    assert loaded.VALUE == ("invented", 7) and sys.modules[name] is previous


def test_dependency_mismatch_publicly_refuses_before_native_calls(monkeypatch):
    monkeypatch.setattr(collection, "_SOURCE_BINDINGS", (("scripts/v3_native_context_collection.py", "0"*64),))
    monkeypatch.setattr(collection, "_make_analyzer", lambda *a: pytest.fail("No native call on source mismatch"))
    _unavailable(collection.collect_cached_native_contexts_v3(object(), allow_ocp=True), "DEPENDENCY_SOURCE_BINDING_MISMATCH")


def _box_vertex(bits):
    return "V"+"".join(str(bit) for bit in bits)


def _box_face(axis, bit):
    return "F"+str(axis)+str(bit)


def _box_edge(first, second):
    return "E"+"-".join(sorted((_box_vertex(first), _box_vertex(second))))


def _box_combinatorics(omitted=None):
    vertices = tuple(product((0, 1), repeat=3))
    faces = tuple((axis, bit) for axis in range(3) for bit in (0, 1)
                  if _box_face(axis, bit) != omitted)
    edges = [(first, second) for i, first in enumerate(vertices) for second in vertices[i+1:]
             if sum(a != b for a, b in zip(first, second)) == 1]
    pairs = {("SHELL", "SOLID")}
    multiplicities = {"SOLID": 1, "SHELL": 1}
    for axis, bit in faces:
        face = _box_face(axis, bit)
        pairs.add(("W"+face, face))
        multiplicities[face] = multiplicities["W"+face] = 1
        for vertex in vertices:
            if vertex[axis] == bit: pairs.add((_box_vertex(vertex), face))
        for first, second in edges:
            if first[axis] == second[axis] == bit: pairs.add((_box_edge(first, second), face))
    for first, second in edges:
        edge = _box_edge(first, second)
        pairs.update(((_box_vertex(first), edge), (_box_vertex(second), edge)))
        multiplicities[edge] = sum(first[axis] == second[axis] == bit for axis, bit in faces)
    for vertex in vertices:
        multiplicities[_box_vertex(vertex)] = 2*sum(vertex[axis] == bit for axis, bit in faces)
    assert len(pairs) == (79 if omitted is None else 70)
    assert len(multiplicities) == (34 if omitted is None else 32)
    assert sum(multiplicities.values()) == (86 if omitted is None else 72)
    return pairs, multiplicities


CYLINDER_PAIRS = {("SHELL", "SOLID"), ("WSIDE", "FSIDE"), ("WBOTTOM", "FBOTTOM"), ("WTOP", "FTOP"),
    ("ESEAM", "FSIDE"), ("EBOTTOM", "FSIDE"), ("EBOTTOM", "FBOTTOM"),
    ("ETOP", "FSIDE"), ("ETOP", "FTOP"),
    ("V0", "ESEAM"), ("V1", "ESEAM"), ("V0", "EBOTTOM"), ("V1", "ETOP"),
    ("V0", "FSIDE"), ("V0", "FBOTTOM"), ("V1", "FSIDE"), ("V1", "FTOP")}
CYLINDER_MULTIPLICITIES = dict(SOLID=1, SHELL=1, FSIDE=1, FBOTTOM=1, FTOP=1,
    WSIDE=1, WBOTTOM=1, WTOP=1, ESEAM=2, EBOTTOM=2, ETOP=2, V0=6, V1=6)


def test_independent_native_combinatorial_counts_do_not_need_ocp():
    _box_combinatorics()
    for face in (_box_face(axis, bit) for axis in range(3) for bit in (0, 1)):
        _box_combinatorics(face)
    assert len(CYLINDER_PAIRS) == 17 and len(CYLINDER_MULTIPLICITIES) == 13
    assert sum(CYLINDER_MULTIPLICITIES.values()) == 26


def _native_vertices(shape):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    explorer, result = TopExp_Explorer(shape, TopAbs_VERTEX), set()
    while explorer.More():
        point = BRep_Tool.Pnt_s(TopoDS.Vertex_s(explorer.Current()))
        result.add((point.X(), point.Y(), point.Z()))
        explorer.Next()
    return result


def _classify_native_shape(shape, kind, cylinder=False):
    # Geometry labels bind IDs only; expected contexts come exclusively from
    # fixed cube/cylinder combinatorics above, never an occurrence ancestor walk.
    if kind in ("SOLID", "SHELL"): return kind
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Circle, GeomAbs_Line
    from OCP.TopoDS import TopoDS
    if kind == "VERTEX":
        point = BRep_Tool.Pnt_s(TopoDS.Vertex_s(shape))
        coords = {(point.X(), point.Y(), point.Z())}
    else:
        coords = _native_vertices(shape)
    assert coords
    if cylinder:
        for x, y, z in coords:
            assert abs(x-4.) < 1e-8 and abs(y) < 1e-8
            assert min(abs(z), abs(z-9.)) < 1e-8
        levels = {0 if abs(z) < 1e-8 else 1 for _, _, z in coords}
        if kind == "VERTEX": return "V"+str(next(iter(levels)))
        if kind == "WIRE": return "WSIDE" if len(levels) == 2 else ("WBOTTOM" if 0 in levels else "WTOP")
        if kind == "EDGE":
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(shape))
            if len(levels) == 2:
                assert curve.GetType() == GeomAbs_Line
                return "ESEAM"
            assert curve.GetType() == GeomAbs_Circle and abs(curve.Circle().Radius()-4.) < 1e-8
            return "EBOTTOM" if 0 in levels else "ETOP"
        surface = BRepAdaptor_Surface(TopoDS.Face_s(shape), True)
        if len(levels) == 2:
            assert surface.GetType() == GeomAbs_Cylinder and abs(surface.Cylinder().Radius()-4.) < 1e-8
            return "FSIDE"
        assert surface.GetType() == GeomAbs_Plane
        return "FBOTTOM" if 0 in levels else "FTOP"
    bits = set()
    for coordinate in coords:
        value = []
        for component, extent in zip(coordinate, (10., 12., 14.)):
            assert min(abs(component), abs(component-extent)) < 1e-8
            value.append(0 if abs(component) < 1e-8 else 1)
        bits.add(tuple(value))
    if kind == "VERTEX":
        assert len(bits) == 1
        return _box_vertex(next(iter(bits)))
    if kind == "EDGE":
        assert len(bits) == 2
        return _box_edge(*sorted(bits))
    fixed = [(axis, next(iter({b[axis] for b in bits}))) for axis in range(3)
             if len({b[axis] for b in bits}) == 1]
    assert len(bits) == 4 and len(fixed) == 1
    face = _box_face(*fixed[0])
    return "W"+face if kind == "WIRE" else face


def _fresh_native_case(case):
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid
    if case == "cylinder": return BRepPrimAPI_MakeCylinder(4., 9.).Shape(), None
    shape, omitted = BRepPrimAPI_MakeBox(10., 12., 14.).Shape(), None
    if case == "open_shell":
        # Source-equivalent reuse of the reviewed five-face fixture: omit the
        # first face, add the remaining original faces to NEW owned containers.
        shells = TopExp_Explorer(shape, TopAbs_SHELL)
        faces = TopExp_Explorer(shells.Current(), TopAbs_FACE)
        assert faces.More()
        omitted = _classify_native_shape(faces.Current(), "FACE")
        faces.Next()
        builder, shell, solid = BRep_Builder(), TopoDS_Shell(), TopoDS_Solid()
        builder.MakeShell(shell)
        count = 0
        while faces.More():
            value = faces.Current()
            builder.Add(shell, value.Oriented(value.Orientation()))
            count += 1
            faces.Next()
        assert count == 5
        builder.MakeSolid(solid)
        builder.Add(solid, shell)
        shape = solid
    if case == "reversed_box": shape = shape.Reversed()
    return shape, omitted


def _native_labels(shape, report, cylinder):
    from OCP.TopAbs import (TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE,
                           TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX)
    from OCP.TopoDS import TopoDS_Iterator
    # Independent identity matching by IsSame (TShape+location, ignoring use
    # orientation). Traversal establishes ID labels ONLY, not expected pairs.
    ordered, stack = [], [shape]
    while stack:
        current = stack.pop()
        if not any(current.IsSame(prior) for prior in ordered): ordered.append(current)
        children, iterator = [], TopoDS_Iterator(current, True, True)
        while iterator.More():
            value = iterator.Value()
            children.append(value.Oriented(value.Orientation()))
            iterator.Next()
        assert len(ordered) <= 40 and len(stack)+len(children) <= 100
        stack.extend(reversed(children))
    assert len(ordered) == len(report["entities"])
    kinds = {TopAbs_SOLID: "SOLID", TopAbs_SHELL: "SHELL", TopAbs_FACE: "FACE",
             TopAbs_WIRE: "WIRE", TopAbs_EDGE: "EDGE", TopAbs_VERTEX: "VERTEX"}
    assert all(entity["type"] == kinds[ordered[entity["native_id"]].ShapeType()]
               for entity in report["entities"])
    labels = {entity["native_id"]: _classify_native_shape(ordered[entity["native_id"]], entity["type"], cylinder)
              for entity in report["entities"]}
    assert len(set(labels.values())) == len(labels)
    return labels


def _native_contract(report, shape, case, omitted):
    assert report["status"] == "COLLECTED" and report["read_sequence_complete"]
    assert report["topology_complete"] and report["native_analyzer_count"] == 1
    assert report["keyed_reads_complete"] and report["iterator_traversal_complete"]
    assert type(report["native_valid"]) is bool
    if case in ("box", "cylinder"): assert report["native_valid"] is True
    if case == "open_shell": assert report["native_valid"] is False
    labels = _native_labels(shape, report, case == "cylinder")
    expected, multiplicities = ((CYLINDER_PAIRS, CYLINDER_MULTIPLICITIES) if case == "cylinder"
                                 else _box_combinatorics(omitted))
    assert set(labels.values()) == set(multiplicities)
    assert len(report["occurrences"]) == sum(multiplicities.values())
    assert {labels[row["native_id"]]: row["occurrence_count"] for row in report["entities"]} == multiplicities
    queries = {(labels[a], labels[b]): (membership, getter) for a, b, membership, getter in report["cache_queries"]}
    assert len(queries) == len(report["cache_queries"]) == len(expected) and set(queries) == expected
    for membership, getter in queries.values():
        assert (membership == "ABSENT" and getter == "NOT_CALLED") or (membership == "PRESENT" and getter in ("READ", "EMPTY"))
    channels = {}
    for name in ("keyed", "iterator"):
        values = {(labels[a], labels[b]): tuple(codes) for a, b, codes in report[name+"_observations"]}
        assert len(values) == len(report[name+"_observations"]) and set(values) <= expected
        assert report["coverage"][name]["counts"]["expected_context_pairs"] == len(expected)
        assert report["coverage"][name]["counts"]["observed_context_pairs"] == len(values)
        channels[name] = values
    assert set(channels["keyed"]) == {pair for pair, (membership, _) in queries.items() if membership == "PRESENT"}
    assert report["comparison"]["matching_pairs"] == sum(channels["keyed"][pair] == channels["iterator"][pair]
        for pair in channels["keyed"].keys() & channels["iterator"].keys())
    _no_admission(report)
    return dict(native_valid=report["native_valid"], expected_pairs=len(expected),
        self_statuses=sorted((labels[row["native_id"]], row["self_status_codes"], row.get("subtree_valid"))
                             for row in report["self_details"]),
        keyed=sorted((a, b, list(codes)) for (a, b), codes in channels["keyed"].items()),
        iterator=sorted((a, b, list(codes)) for (a, b), codes in channels["iterator"].items()),
        queries=sorted((a, b, *values) for (a, b), values in queries.items()))


def _write_case(directory, index, record):
    path = directory / ("case_"+str(index).zfill(2)+".json")
    try:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > CASE_BYTES: raise ValueError("CASE_ARTIFACT_BYTE_BUDGET")
    except Exception as error:
        record = dict(case=record.get("case"), read_order=record.get("read_order"), contract_pass=False,
            phase="artifact_encoding", error_type=type(error).__name__,
            reason="CASE_ARTIFACT_ENCODING_OR_BYTE_FAILURE", evaluated_report_discarded_whole=True)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    with path.open("rb") as stream:
        retained_bytes = stream.read(CASE_BYTES+1)
    if retained_bytes != encoded:
        raise ValueError("CASE_ARTIFACT_RETAINED_BYTES_MISMATCH")
    return record, dict(path=path.name, bytes=len(retained_bytes),
                        sha256=hashlib.sha256(retained_bytes).hexdigest())


@pytest.mark.parametrize("bad", ("nonfinite", "oversize"))
def test_artifact_encoding_failure_is_a_retained_failure_not_a_partial_report(tmp_path, bad):
    record = dict(case="invented", read_order=READ_ORDERS[0], contract_pass=True,
                  report=float("nan") if bad == "nonfinite" else "x"*CASE_BYTES)
    retained, reference = _write_case(tmp_path, 0, record)
    assert retained["contract_pass"] is False and retained["evaluated_report_discarded_whole"]
    assert reference["bytes"] < CASE_BYTES
    with pytest.raises(FileExistsError): _write_case(tmp_path, 0, record)


def test_artifact_reference_binds_fsynced_readback_bytes(tmp_path, monkeypatch):
    fsync, calls = os.fsync, []
    def witnessed_fsync(fd):
        calls.append(fd)
        return fsync(fd)
    monkeypatch.setattr(os, "fsync", witnessed_fsync)
    record = dict(case="invented", read_order=READ_ORDERS[0], contract_pass=True,
                  report=dict(keyed_observations=[[1, 0, [0]]], iterator_observations=[]))
    retained, reference = _write_case(tmp_path, 0, record)
    raw = (tmp_path / reference["path"]).read_bytes()
    assert len(calls) == 1 and json.loads(raw) == retained == record
    assert reference["bytes"] == len(raw) and reference["sha256"] == hashlib.sha256(raw).hexdigest()


def run_native_context_collection_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1": raise RuntimeError("EXPLICIT_CONSTRUCTED_NATIVE_OPT_IN_REQUIRED")
    destination = os.environ.get(ARTIFACT_ENV)
    if not destination: raise RuntimeError("PARENT_OWNED_ARTIFACT_DESTINATION_REQUIRED")
    directory = Path(destination)
    if not directory.is_absolute(): raise RuntimeError("ABSOLUTE_ARTIFACT_DESTINATION_REQUIRED")
    directory.mkdir(exist_ok=False)
    records, references, semantic = [], [], {}
    for case in NATIVE_CASES:
        for order in READ_ORDERS:
            record = dict(case=case, read_order=order, contract_pass=False, phase="fixture")
            try:
                shape, omitted = _fresh_native_case(case)
                record["phase"] = "collection"
                report = collection.collect_cached_native_contexts_v3(shape, allow_ocp=True, read_order=order)
                record["report"] = report
                record["phase"] = "independent_contract_oracle"
                normalized = _native_contract(report, shape, case, omitted)
                record.update(contract_pass=True, phase="complete", semantic_observation=normalized)
                semantic[(case, order)] = normalized
            except Exception as error:
                record.update(error_type=type(error).__name__, error=str(error)[:200])
            try:
                retained, reference = _write_case(directory, len(records), record)
                records.append(retained)
                references.append(reference)
            except Exception as error:
                records.append(dict(case=case, read_order=order, contract_pass=False,
                    phase="artifact_write", error_type=type(error).__name__, error=str(error)[:200]))
                references.append(None)
    rows = []
    for record, reference in zip(records, references):
        report = record.get("report", {})
        rows.append(dict(case=record["case"], read_order=record["read_order"], contract_pass=record["contract_pass"],
            phase=record["phase"], error_type=record.get("error_type"), artifact=reference,
            native_valid=report.get("native_valid"),
            keyed_coverage_complete=report.get("coverage", {}).get("keyed", {}).get("coverage_complete", False),
            iterator_coverage_complete=report.get("coverage", {}).get("iterator", {}).get("coverage_complete", False),
            channels_consistent=report.get("channels_consistent", False)))
    assert sum(reference["bytes"] for reference in references if reference is not None) <= TOTAL_BYTES
    order_agreement = {case: all((case, order) in semantic for order in READ_ORDERS)
        and semantic[(case, READ_ORDERS[0])] == semantic[(case, READ_ORDERS[1])] for case in NATIVE_CASES}
    summary = dict(scope="EIGHT_FRESH_INVENTED_SHAPES_CACHED_OBSERVATION_NOT_ADMISSION",
        expected_cases=8, attempted_cases=len(records), contract_passed=sum(row["contract_pass"] for row in rows),
        context_gap_closed_on_constructs=len(rows) == 8 and all(order_agreement.values())
            and all(row["contract_pass"] and row["keyed_coverage_complete"]
                and row["iterator_coverage_complete"] and row["channels_consistent"] for row in rows),
        fresh_read_orders_semantically_agree=order_agreement,
        records=rows, artifact_directory=str(directory), local_admission_allowed=False,
        global_admission_allowed=False, production_use_allowed=False, observed_bend_count=None)
    assert len(json.dumps(summary, allow_nan=False).encode()) <= SUMMARY_BYTES
    return summary


@pytest.mark.parametrize("mode", ("complete", "order_disagreement", "missing_coverage", "read_failure", "encode_failure"))
def test_native_manifest_bookkeeping_with_all_geometry_and_collection_mocked(tmp_path, monkeypatch, mode):
    monkeypatch.setenv(NATIVE_OPT_IN, "1")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path / "fresh_owned_reports"))
    attempts = []
    monkeypatch.setattr(sys.modules[__name__], "_fresh_native_case",
                        lambda case: (SimpleNamespace(case=case), None))
    def observe(shape, *, allow_ocp, read_order):
        assert allow_ocp is True
        attempts.append((shape.case, read_order))
        first = shape.case == "box" and read_order == READ_ORDERS[0]
        if first and mode == "read_failure": raise RuntimeError("invented one-case failure")
        return dict(native_valid=True, channels_consistent=True,
            coverage=dict(keyed=dict(coverage_complete=not (first and mode == "missing_coverage")),
                          iterator=dict(coverage_complete=True)),
            semantic_code=1 if first and mode == "order_disagreement" else 0,
            encoded_control=float("nan") if first and mode == "encode_failure" else 0)
    monkeypatch.setattr(collection, "collect_cached_native_contexts_v3", observe)
    monkeypatch.setattr(sys.modules[__name__], "_native_contract",
                        lambda report, *args: dict(code=report["semantic_code"]))
    summary = run_native_context_collection_matrix()
    assert attempts == [(case, order) for case in NATIVE_CASES for order in READ_ORDERS]
    assert summary["attempted_cases"] == len(summary["records"]) == 8
    assert summary["contract_passed"] == (7 if mode in ("read_failure", "encode_failure") else 8)
    assert summary["context_gap_closed_on_constructs"] is (mode == "complete")
    assert summary["fresh_read_orders_semantically_agree"]["box"] is (mode not in ("order_disagreement", "read_failure"))
    assert all(row["artifact"] is not None for row in summary["records"])
    for row in summary["records"]:
        raw = (Path(summary["artifact_directory"]) / row["artifact"]["path"]).read_bytes()
        assert row["artifact"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert json.loads(raw)["contract_pass"] is row["contract_pass"]


@pytest.mark.skipif(os.environ.get(NATIVE_OPT_IN) != "1", reason="Separate independently audited constructed-native gate")
def test_eight_fresh_native_cached_context_observations():
    summary = run_native_context_collection_matrix()
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    assert summary["attempted_cases"] == 8 and summary["contract_passed"] == 8, summary
