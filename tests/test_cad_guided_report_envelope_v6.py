"""Independent invented-only report custody controls; no geometry or file IO."""

import copy
import json

import pytest

from backend import cad_guided_report_envelope_v6 as subject


LIMIT = 96*1024**2
COUNT_LIMIT = 2**63-1
SCHEMA = "initialization_report_refusal_envelope_v6"
BUDGET = "INITIALIZATION_REPORT_BUDGET_EXCEEDED"
SERIALIZATION = "INITIALIZATION_REPORT_SERIALIZATION_FAILURE"
INVALID_CAP = "INVALID_INITIALIZATION_REPORT_CAP"
INVALID_CONTAINER = "INVALID_INITIALIZATION_REPORT_CONTAINER"
INVALID_ENVELOPE = "INVALID_INITIALIZATION_REFUSAL_ENVELOPE"
FIXED = dict(schema_version=SCHEMA, status="UNAVAILABLE", transform_scan_to_cad=None,
    search_complete=False, selected_policy_evaluation_complete=False,
    accepted_registration=False, production_use_allowed=False, observed_bend_count=None,
    conformance="INDETERMINATE", detailed_report_discarded_whole=True,
    evidence_scope="NO_PARTIAL_REPORT_OR_POSE_PUBLISHED",
    count_basis="RETAINED_LIST_LENGTHS_OR_PRIOR_CANONICAL_COUNTS_NOT_UNSEEN_WORK",
    byte_evidence_kind="ENCODED_PREFIX_COUNT_CAPPED_AT_LIMIT_PLUS_ONE")
EVENT_FIELDS = ("boundary_reason", "detailed_report_limit_bytes", "detailed_report_limit_valid",
    "encoded_bytes_examined", "source_schema_version", "source_reasons", "source_reasons_complete",
    "discarded_hypothesis_records", "discarded_hypothesis_count_status",
    "discarded_candidate_records", "discarded_candidate_count_status")
ENVELOPE_FIELDS = tuple(FIXED)+tuple(key for key in EVENT_FIELDS if key != "boundary_reason")+(
    "reasons", "hypotheses", "start_groups", "first_refusal")


class StringSubclass(str):
    pass


class ListSubclass(list):
    pass


class DictSubclass(dict):
    pass


class IntSubclass(int):
    pass


def _size(value):
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("ascii"))


def _fixture():
    """Construct the declared schema independently, not by calling the helper."""
    event = dict(boundary_reason=BUDGET, detailed_report_limit_bytes=7,
        detailed_report_limit_valid=True, encoded_bytes_examined=8,
        source_schema_version="Invented_source.v1", source_reasons=["SOURCE_REFUSAL"],
        source_reasons_complete=True, discarded_hypothesis_records=3,
        discarded_hypothesis_count_status="RETAINED_RECORD_COUNT", discarded_candidate_records=5,
        discarded_candidate_count_status="RETAINED_RECORD_COUNT")
    return dict(FIXED, **{key: value for key, value in event.items() if key != "boundary_reason"},
        reasons=[BUDGET], hypotheses=[], start_groups=[], first_refusal=copy.deepcopy(event))


def _safe(value, reason=None):
    assert type(value) is dict and set(value) == set(ENVELOPE_FIELDS)
    assert all(type(key) is str for key in value)
    for key, expected in FIXED.items():
        assert type(value[key]) is type(expected) and value[key] == expected
    assert type(value["reasons"]) is list and len(value["reasons"]) == 1
    assert value["reasons"][0] in (BUDGET, SERIALIZATION, INVALID_CAP, INVALID_CONTAINER, INVALID_ENVELOPE)
    if reason is not None:
        assert value["reasons"] == [reason]
    assert value["hypotheses"] == [] and value["start_groups"] == []
    assert type(value["hypotheses"]) is list and type(value["start_groups"]) is list
    assert type(value["first_refusal"]) is dict and set(value["first_refusal"]) == set(EVENT_FIELDS)
    assert _size(value) <= 4096
    return value


DETAILED_REPORTS = [
    {},
    {"message": "quotes\" and slash\\ and newline\n", "null": None},
    {"message": "invented Unicode: \u03b1 \U0001f9ea"},
    {"numbers": [False, True, 0, -2, 1.0, -0.0, [3.25]]},
    {"schema_version": "invented", "status": "COMPLETE", "transform_scan_to_cad": [[1, 0], [0, 1]],
     "hypotheses": [{"id": 0}, {"id": 1}], "evidence": {"candidates": [{"id": 9}]}},
]


@pytest.mark.parametrize("report", DETAILED_REPORTS)
def test_exact_detailed_cap_keeps_identity_and_one_byte_less_refuses(report):
    original = copy.deepcopy(report)
    size = _size(report)
    assert subject.bound_initialization_report_v6(report, size) is report
    refusal = _safe(subject.bound_initialization_report_v6(report, size-1), BUDGET)
    assert refusal["detailed_report_limit_bytes"] == size-1
    assert refusal["encoded_bytes_examined"] == size
    assert refusal["first_refusal"]["encoded_bytes_examined"] == size
    assert report == original


@pytest.mark.parametrize("repeat_cap", [0, 1, 4096, LIMIT])
def test_zero_cap_two_and_three_pass_idempotence_and_detachment(repeat_cap):
    report = {"schema_version": "invented", "reasons": ["FIRST"], "hypotheses": [1, 2, 3]}
    first = _safe(subject.bound_initialization_report_v6(report, 0), BUDGET)
    assert first["encoded_bytes_examined"] == 1
    second = _safe(subject.bound_initialization_report_v6(first, repeat_cap), BUDGET)
    third = _safe(subject.bound_initialization_report_v6(second, 0), BUDGET)
    assert first == second == third and first is not second and second is not third
    snapshot = copy.deepcopy(first)
    for key in ("reasons", "source_reasons", "hypotheses", "start_groups"):
        assert second[key] is not first[key]
        second[key].append("MUTATION")
    assert second["first_refusal"] is not first["first_refusal"]
    assert second["first_refusal"]["source_reasons"] is not first["first_refusal"]["source_reasons"]
    second["first_refusal"]["source_reasons"].append("MUTATION")
    second["first_refusal"]["encoded_bytes_examined"] = 99
    assert first == snapshot and third == snapshot


@pytest.mark.parametrize("maximum", [None, True, False, -1, LIMIT+1, 0.0, 1.0, "0", [], {}, float("inf"), float("nan"), IntSubclass(1)])
def test_invalid_cap_has_exact_safe_reason_and_preserves_prior_first_event(maximum):
    first = _fixture()
    result = _safe(subject.bound_initialization_report_v6(first, maximum), INVALID_CAP)
    assert result["detailed_report_limit_bytes"] == result["encoded_bytes_examined"] == 0
    assert result["detailed_report_limit_valid"] is False
    assert result["first_refusal"] == first["first_refusal"]
    assert result["discarded_hypothesis_records"] == 3 and result["discarded_candidate_records"] == 5
    assert subject.bound_initialization_report_v6(result, 0) == result


@pytest.mark.parametrize("report", [None, [], (), "report", 5, True, float("nan"), DictSubclass(_fixture())])
def test_non_dictionary_reports_have_unknown_not_zero_counts(report):
    result = _safe(subject.bound_initialization_report_v6(report, LIMIT), INVALID_CONTAINER)
    for name in ("hypothesis", "candidate"):
        assert result["discarded_"+name+"_records"] is None
        assert result["discarded_"+name+"_count_status"] == "UNKNOWN_MALFORMED_CONTAINER"
    assert result["source_schema_version"] is None and result["source_reasons"] == []
    assert result["source_reasons_complete"] is False


COUNTS = [
    ({}, (0, "NO_RETAINED_LIST_PRESENT"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": []}, (0, "RETAINED_RECORD_COUNT"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": [0, 1]}, (2, "RETAINED_RECORD_COUNT"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": None}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": (0, 1)}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"evidence": None}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"evidence": []}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"evidence": {}}, (0, "NO_RETAINED_LIST_PRESENT"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"evidence": {"candidates": []}}, (0, "NO_RETAINED_LIST_PRESENT"), (0, "RETAINED_RECORD_COUNT")),
    ({"evidence": {"candidates": [0, 1]}}, (0, "NO_RETAINED_LIST_PRESENT"), (2, "RETAINED_RECORD_COUNT")),
    ({"evidence": {"candidates": None}}, (0, "NO_RETAINED_LIST_PRESENT"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"evidence": {"initialization": None}}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"evidence": {"initialization": {}}}, (0, "NO_RETAINED_LIST_PRESENT"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": [0, 1, 2], "evidence": {"initialization": {"hypotheses": [0, 1, 2, 3]}}},
     (4, "RETAINED_RECORD_COUNT"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": [0, 1, 2], "evidence": {"initialization": ()}},
     (3, "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"hypotheses": None, "evidence": {"initialization": {"hypotheses": [0, 1, 2, 3]}}},
     (4, "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"hypotheses": [], "evidence": {"initialization": None, "candidates": [0, 1]}},
     (0, "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER"), (2, "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER")),
    ({"hypotheses": ListSubclass([0]), "evidence": DictSubclass(candidates=[0])},
     (None, "UNKNOWN_MALFORMED_CONTAINER"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"evidence": {"candidates": ()}}, (0, "NO_RETAINED_LIST_PRESENT"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"evidence": {"candidates": ListSubclass([1])}}, (0, "NO_RETAINED_LIST_PRESENT"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
    ({"hypotheses": {}}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (0, "NO_RETAINED_LIST_PRESENT")),
    ({"evidence": {"initialization": []}}, (None, "UNKNOWN_MALFORMED_CONTAINER"), (None, "UNKNOWN_MALFORMED_CONTAINER")),
]


@pytest.mark.parametrize("report,hypotheses,candidates", COUNTS)
def test_retained_container_denominators_are_not_laundered(report, hypotheses, candidates):
    result = _safe(subject.bound_initialization_report_v6(report, 0), BUDGET)
    for name, expected in (("hypothesis", hypotheses), ("candidate", candidates)):
        assert (result["discarded_"+name+"_records"], result["discarded_"+name+"_count_status"]) == expected


def test_nested_first_refusal_and_aliased_counts_use_max_never_sum():
    inner = _fixture()
    inner["detailed_report_limit_bytes"], inner["encoded_bytes_examined"] = 23, 24
    earliest = copy.deepcopy(inner["first_refusal"])
    pool = [dict(id=index) for index in range(11)]
    outer = {"schema_version": "Outer.v2", "hypotheses": pool[:7],
             "evidence": {"candidates": pool, "initialization": inner}}
    refused = _safe(subject.bound_initialization_report_v6(outer, 0), BUDGET)
    assert refused["first_refusal"] == earliest and refused["first_refusal"] is not inner["first_refusal"]
    assert refused["detailed_report_limit_bytes"] == 0 and refused["encoded_bytes_examined"] == 1
    assert refused["discarded_hypothesis_records"] == 7 and refused["discarded_candidate_records"] == 11
    assert subject.bound_initialization_report_v6(refused, 0) == refused
    third = _safe(subject.bound_initialization_report_v6({"hypotheses": [1, 2],
        "evidence": {"candidates": [1, 2, 3, 4], "initialization": refused}}, 0), BUDGET)
    assert third["first_refusal"] == earliest
    assert third["discarded_hypothesis_records"] == 7 and third["discarded_candidate_records"] == 11


def test_unknown_inner_count_preserves_known_outer_lower_bound_and_no_recursive_counts():
    inner = _fixture()
    inner.update(discarded_hypothesis_records=None, discarded_hypothesis_count_status="UNKNOWN_MALFORMED_CONTAINER",
                 discarded_candidate_records=None, discarded_candidate_count_status="UNKNOWN_MALFORMED_CONTAINER")
    report = {"hypotheses": [1, 2], "start_groups": list(range(20)), "statistics": {"attempted": 999999},
        "evidence": {"candidates": [1, 2, 3], "initialization": inner},
        "arbitrary": {"hypotheses": list(range(30))}}
    result = _safe(subject.bound_initialization_report_v6(report, 0), BUDGET)
    for name, count in (("hypothesis", 2), ("candidate", 3)):
        assert result["discarded_"+name+"_records"] == count
        assert result["discarded_"+name+"_count_status"] == "KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER"


@pytest.mark.parametrize("key", ENVELOPE_FIELDS)
def test_every_missing_outer_key_blocks_canonical_exemption(key):
    value = _fixture()
    del value[key]
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


@pytest.mark.parametrize("key", EVENT_FIELDS)
def test_every_missing_first_event_key_blocks_canonical_exemption(key):
    value = _fixture()
    del value["first_refusal"][key]
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


@pytest.mark.parametrize("key", ENVELOPE_FIELDS)
def test_equal_outer_str_subclass_key_is_not_a_builtin_schema_key(key):
    value = _fixture()
    item = value.pop(key)
    value[StringSubclass(key)] = item
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


@pytest.mark.parametrize("key", EVENT_FIELDS)
def test_equal_first_event_str_subclass_key_is_not_a_builtin_schema_key(key):
    value = _fixture()
    item = value["first_refusal"].pop(key)
    value["first_refusal"][StringSubclass(key)] = item
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


BAD_FIXED = [
    ("schema_version", "other"), ("status", "COMPLETE"), ("transform_scan_to_cad", [[1, 0], [0, 1]]),
    ("search_complete", True), ("selected_policy_evaluation_complete", True),
    ("accepted_registration", True), ("production_use_allowed", True), ("observed_bend_count", 0),
    ("conformance", "PASS"), ("detailed_report_discarded_whole", False),
    ("evidence_scope", "COMPLETE"), ("count_basis", "ALL_ATTEMPTED_WORK"), ("byte_evidence_kind", "FULL_SIZE"),
]


@pytest.mark.parametrize("key,bad", BAD_FIXED)
def test_every_fixed_value_is_required(key, bad):
    value = _fixture()
    value[key] = bad
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


@pytest.mark.parametrize("key", tuple(FIXED))
def test_every_fixed_value_uses_exact_builtin_type(key):
    value = _fixture()
    original = value[key]
    value[key] = StringSubclass(original) if type(original) is str else (int(original) if type(original) is bool else False)
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


BAD_EVENTS = [
    ("boundary_reason", "FORGED"), ("boundary_reason", StringSubclass(BUDGET)),
    ("detailed_report_limit_bytes", -1), ("detailed_report_limit_bytes", LIMIT+1),
    ("detailed_report_limit_bytes", True), ("detailed_report_limit_bytes", 7.0),
    ("detailed_report_limit_valid", 1), ("detailed_report_limit_valid", False),
    ("encoded_bytes_examined", -1), ("encoded_bytes_examined", 9),
    ("encoded_bytes_examined", 7), ("encoded_bytes_examined", True), ("encoded_bytes_examined", 8.0),
    ("source_schema_version", "A"*97), ("source_schema_version", "bad/schema"),
    ("source_schema_version", StringSubclass("valid")), ("source_schema_version", 1),
    ("source_schema_version", ""),
    ("source_reasons", ("VALID",)), ("source_reasons", ["A"]*4),
    ("source_reasons", ["A"*181]), ("source_reasons", ["lowercase"]),
    ("source_reasons", [StringSubclass("VALID")]), ("source_reasons_complete", 1),
    ("source_reasons", [""]), ("source_reasons", ListSubclass(["VALID"])),
    ("discarded_hypothesis_records", -1), ("discarded_hypothesis_records", COUNT_LIMIT+1),
    ("discarded_hypothesis_records", True), ("discarded_hypothesis_records", None),
    ("discarded_hypothesis_records", 3.0), ("discarded_hypothesis_count_status", "FORGED"),
    ("discarded_hypothesis_count_status", StringSubclass("RETAINED_RECORD_COUNT")),
    ("discarded_hypothesis_count_status", "UNKNOWN_MALFORMED_CONTAINER"),
    ("discarded_hypothesis_count_status", "NO_RETAINED_LIST_PRESENT"),
    ("discarded_candidate_records", -1), ("discarded_candidate_records", COUNT_LIMIT+1),
    ("discarded_candidate_records", True), ("discarded_candidate_records", None),
    ("discarded_candidate_records", 5.0), ("discarded_candidate_count_status", "FORGED"),
    ("discarded_candidate_count_status", StringSubclass("RETAINED_RECORD_COUNT")),
    ("discarded_candidate_count_status", "UNKNOWN_MALFORMED_CONTAINER"),
    ("discarded_candidate_count_status", "NO_RETAINED_LIST_PRESENT"),
]


@pytest.mark.parametrize("key,bad", BAD_EVENTS)
@pytest.mark.parametrize("nested", [False, True])
def test_every_event_value_type_and_relational_bound_is_validated(key, bad, nested):
    value = _fixture()
    target = value["first_refusal"] if nested else value
    target["reasons" if key == "boundary_reason" and not nested else key] = [bad] if key == "boundary_reason" and not nested else bad
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


@pytest.mark.parametrize("key,bad", [
    ("reasons", []), ("reasons", [BUDGET, SERIALIZATION]), ("reasons", (BUDGET,)),
    ("reasons", ListSubclass([BUDGET])),
    ("hypotheses", [dict(pose="FORGED")]), ("hypotheses", ()), ("start_groups", [1]),
    ("hypotheses", ListSubclass()), ("start_groups", ()),
    ("start_groups", ListSubclass()), ("first_refusal", None), ("first_refusal", DictSubclass(_fixture()["first_refusal"])),
])
def test_canonical_list_and_event_containers_are_exact_and_empty_where_required(key, bad):
    value = _fixture()
    value[key] = bad
    _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)


@pytest.mark.parametrize("nested,key", [(False, "pose"), (False, "transform"), (False, "extra"),
    (False, "evidence"), (False, "candidates"), (True, "pose"), (True, "first_refusal"), (True, "extra")])
def test_extra_pose_payload_or_recursive_event_never_gets_exemption(nested, key):
    value = _fixture()
    (value["first_refusal"] if nested else value)[key] = {"pose": [[1]], "SECRET_DYNAMIC_TEXT": "NOT_RETAINED"}
    result = _safe(subject.bound_initialization_report_v6(value, LIMIT), INVALID_ENVELOPE)
    assert "SECRET_DYNAMIC_TEXT" not in json.dumps(result)


@pytest.mark.parametrize("nested", [False, True])
def test_wrong_fixed_header_length_refuses_before_iterating_keys(monkeypatch, nested):
    value = _fixture()["first_refusal"] if nested else _fixture()
    value["one_extra_key"] = None
    def forbidden_iteration(*args, **kwargs):
        raise AssertionError("HEADER_KEY_ITERATION_MUST_NOT_START")
    monkeypatch.setattr(subject, "all", forbidden_iteration, raising=False)
    assert (subject._event_valid(value) if nested else subject._canonical(value)) is False


SOURCE_CASES = [
    ({}, None, [], True),
    ({"schema_version": "A"*96, "reasons": ["A"*180]*3}, "A"*96, ["A"*180]*3, True),
    ({"schema_version": "A"*97}, None, [], True),
    ({"schema_version": "bad/schema"}, None, [], True),
    ({"schema_version": StringSubclass("valid")}, None, [], True),
    ({"reasons": ["A", "B", "C", "D"]}, None, ["A", "B", "C"], False),
    ({"reasons": ["A"*181]}, None, [], False),
    ({"reasons": ["lowercase"]}, None, [], False),
    ({"reasons": ["BAD\nTEXT"]}, None, [], False),
    ({"reasons": ["BAD\"TEXT"]}, None, [], False),
    ({"reasons": ["\u03b1"]}, None, [], False),
    ({"reasons": [""]}, None, [], False),
    ({"reasons": ("A",)}, None, [], False),
    ({"reasons": [True, "VALID", StringSubclass("OTHER")]}, None, ["VALID"], False),
    ({"reasons": ["UPPER 1_.:-/ CODE"]}, None, ["UPPER 1_.:-/ CODE"], True),
]


@pytest.mark.parametrize("report,schema,reasons,complete", SOURCE_CASES)
def test_bounded_source_metadata_is_not_arbitrarily_stringified(report, schema, reasons, complete):
    result = _safe(subject.bound_initialization_report_v6(report, 0), BUDGET)
    assert result["source_schema_version"] == schema and result["source_reasons"] == reasons
    assert result["source_reasons_complete"] is complete


def test_maximal_canonical_envelope_is_below_four_kib_and_avoids_encoder_reentry(monkeypatch):
    value = _fixture()
    for event in (value, value["first_refusal"]):
        if event is value:
            event["reasons"] = [SERIALIZATION]
        else:
            event["boundary_reason"] = SERIALIZATION
        event.update(detailed_report_limit_bytes=LIMIT, encoded_bytes_examined=LIMIT+1,
            source_schema_version="S"*96, source_reasons=["R"*180]*3,
            discarded_hypothesis_records=COUNT_LIMIT, discarded_candidate_records=COUNT_LIMIT,
            discarded_hypothesis_count_status="KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER",
            discarded_candidate_count_status="KNOWN_LOWER_BOUND_WITH_MALFORMED_CONTAINER")
    encoded_size = _size(value)
    assert encoded_size <= 4096 and subject._canonical_bytes(value) == encoded_size
    def encoder_bomb(*args, **kwargs):
        raise RuntimeError("SECRET_ENCODER_MUST_NOT_RUN")
    monkeypatch.setattr(subject.json, "JSONEncoder", encoder_bomb)
    result = subject.bound_initialization_report_v6(value, 0)
    assert result == value and result is not value
    assert result["first_refusal"] is not value["first_refusal"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), object(), {1, 2}])
def test_nonfinite_or_unserializable_reports_have_no_dynamic_exception_payload(bad):
    result = _safe(subject.bound_initialization_report_v6({"bad": bad, "transform_scan_to_cad": [[1]]}, LIMIT), SERIALIZATION)
    assert 0 <= result["encoded_bytes_examined"] <= LIMIT+1
    assert "NaN" not in json.dumps(result) and "TypeError" not in json.dumps(result)


def test_cyclic_report_refuses_without_publishing_cycle_or_pose():
    report = {"transform_scan_to_cad": [[1]], "hypotheses": [1, 2]}
    report["cycle"] = report
    result = _safe(subject.bound_initialization_report_v6(report, LIMIT), SERIALIZATION)
    assert result["discarded_hypothesis_records"] == 2
    assert "Circular" not in json.dumps(result)


def test_no_arbitrary_string_or_container_methods_are_used_for_diagnostics():
    class Trap:
        def __str__(self):
            raise AssertionError("SECRET_STRING_METHOD")
        def __len__(self):
            raise AssertionError("SECRET_LENGTH_METHOD")
    report = {"schema_version": Trap(), "reasons": [Trap()], "hypotheses": Trap(),
              "evidence": {"candidates": Trap()}, "bad": Trap()}
    result = _safe(subject.bound_initialization_report_v6(report, LIMIT), SERIALIZATION)
    assert result["source_schema_version"] is None and result["source_reasons"] == []
    assert result["discarded_hypothesis_records"] is None and result["discarded_candidate_records"] is None
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("maximum,reason,examined", [(100, SERIALIZATION, 6), (4, BUDGET, 5), (0, BUDGET, 1)])
def test_encoder_prefix_count_is_capped_not_an_invented_total(monkeypatch, maximum, reason, examined):
    class BrokenEncoder:
        def __init__(self, **kwargs):
            assert kwargs == dict(ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        def iterencode(self, value):
            yield "abc"
            yield "def"
            raise RuntimeError("SECRET_RUNTIME_PAYLOAD")
    monkeypatch.setattr(subject.json, "JSONEncoder", BrokenEncoder)
    result = subject.bound_initialization_report_v6({"hypotheses": [1, 2]}, maximum)
    assert result["reasons"] == [reason] and result["encoded_bytes_examined"] == examined
    assert result["first_refusal"]["encoded_bytes_examined"] == examined
    assert "SECRET" not in repr(result) and "RuntimeError" not in repr(result)


def test_encoder_constructor_exception_is_caught_without_dynamic_text(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("SECRET_CONSTRUCTOR_TEXT")
    monkeypatch.setattr(subject.json, "JSONEncoder", fail)
    result = subject.bound_initialization_report_v6({"hypotheses": [1]}, LIMIT)
    assert result["reasons"] == [SERIALIZATION] and result["encoded_bytes_examined"] == 0
    assert "SECRET" not in repr(result) and result["transform_scan_to_cad"] is None


class ProcessControl(BaseException):
    pass


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit, ProcessControl])
def test_process_control_baseexceptions_propagate_unchanged(monkeypatch, exception_type):
    failure = exception_type("PROCESS_CONTROL")
    class InterruptedEncoder:
        def __init__(self, **kwargs):
            pass
        def iterencode(self, value):
            yield "{"
            raise failure
    monkeypatch.setattr(subject.json, "JSONEncoder", InterruptedEncoder)
    with pytest.raises(exception_type) as caught:
        subject.bound_initialization_report_v6({}, LIMIT)
    assert caught.value is failure


def _hostile_key(text, exception):
    class BadStr(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            raise exception
    return BadStr(text)


@pytest.mark.parametrize("mode", ["zero_cap", "invalid_cap", "encoder_failure"])
def test_hostile_hypotheses_key_cannot_escape_ordinary_refusal(monkeypatch, mode):
    report = {_hostile_key("hypotheses", RuntimeError("SECRET_COUNT_EQUALITY")): [1, 2],
              "reasons": ["KNOWN_SOURCE_REASON"]}
    maximum = 0 if mode == "zero_cap" else (None if mode == "invalid_cap" else LIMIT)
    if mode == "encoder_failure":
        def fail(**kwargs):
            raise RuntimeError("SECRET_ENCODER_FAILURE")
        monkeypatch.setattr(subject.json, "JSONEncoder", fail)
    result = subject.bound_initialization_report_v6(report, maximum)
    assert result["reasons"] == [{"zero_cap": BUDGET, "invalid_cap": INVALID_CAP, "encoder_failure": SERIALIZATION}[mode]]
    assert result["source_reasons"] == ["KNOWN_SOURCE_REASON"]
    for name in ("hypothesis", "candidate"):
        assert result["discarded_"+name+"_records"] is None
        assert result["discarded_"+name+"_count_status"] == "UNKNOWN_MALFORMED_CONTAINER"
    assert "SECRET" not in repr(result) and "RuntimeError" not in repr(result)
    assert result["transform_scan_to_cad"] is None and result["hypotheses"] == []


@pytest.mark.parametrize("maximum,expected_reason", [(0, SERIALIZATION), (None, INVALID_CAP)])
def test_source_metadata_hook_failure_does_not_erase_independently_known_counts(maximum, expected_reason):
    report = {"hypotheses": [1, 2, 3], "evidence": {"candidates": [1, 2]},
              _hostile_key("schema_version", RuntimeError("SECRET_SCHEMA_EQUALITY")): "invented"}
    result = _safe(subject.bound_initialization_report_v6(report, maximum), expected_reason)
    assert result["discarded_hypothesis_records"] == 3 and result["discarded_candidate_records"] == 2
    assert result["discarded_hypothesis_count_status"] == result["discarded_candidate_count_status"] == "RETAINED_RECORD_COUNT"
    assert result["source_schema_version"] is None and result["source_reasons"] == []
    assert result["source_reasons_complete"] is False and "SECRET" not in repr(result)


@pytest.mark.parametrize("mode", ["zero_cap", "invalid_cap", "encoder_failure"])
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, ProcessControl])
def test_hostile_retained_evidence_baseexceptions_still_propagate(monkeypatch, mode, exception_type):
    failure = exception_type("PROCESS_CONTROL")
    report = {_hostile_key("hypotheses", failure): [1]}
    maximum = 0 if mode == "zero_cap" else (None if mode == "invalid_cap" else LIMIT)
    if mode == "encoder_failure":
        def fail(**kwargs):
            raise RuntimeError("ORDINARY_ENCODER_FAILURE")
        monkeypatch.setattr(subject.json, "JSONEncoder", fail)
    with pytest.raises(exception_type) as caught:
        subject.bound_initialization_report_v6(report, maximum)
    assert caught.value is failure
