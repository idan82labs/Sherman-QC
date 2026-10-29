"""C10 Jev harness: state deny-list, JSON schema parity, questions and request limits.

Synthetic records only; the network is blocked for the whole module.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import httpx
import jsonschema
import pytest

from research.jev import questions_v1 as q
from research.jev import request_v1 as r
from research.jev import state_v1 as s
from research.jev.export_onboarding_v1 import export_onboarding, write_synthetic_inputs

SCHEMA_PATH = Path(s.__file__).parent / "schemas" / "jev_bend_candidate_state_v1.schema.json"


class NetworkBlocked(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*_a, **_k):
        raise NetworkBlocked("network access is forbidden in C10 tests")

    monkeypatch.setattr(httpx.Client, "send", _refuse)


def _record(**overrides):
    base = dict(
        source_sha256="a" * 64,
        candidate_key="onboarding|cad:1",
        candidate_source="CAD_ONBOARDING",
        candidate={"bend_line_length_mm": 123.456789, "nominal_sweep_deg": 90.0, "angle_convention": "NATIVE_SWEEP"},
        neighbours={"other_candidates_count": 3},
        context={"sheet_thickness_mm": 2.0},
        cad_relation={"nearest_offset_mm": 0.123456, "is_within_10mm_of_reference": True},
    )
    base.update(overrides)
    return s.CandidateRecord(**base)


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# build_state basics
# --------------------------------------------------------------------------


def test_build_state_rounds_to_four_significant_digits_and_keeps_id_outside():
    rid, state = s.build_state(_record())
    assert rid == s.record_id_for("a" * 64, "onboarding|cad:1")
    assert len(rid) == 24
    assert rid not in json.dumps(state)
    assert state["candidate"]["bend_line_length_mm"] == 123.5
    assert state["cad_relation"]["nearest_offset_mm"] == 0.1235
    assert "cad:1" not in json.dumps(state)
    s.check_state(state)


def test_round_sig_edge_cases():
    assert s.round_sig(-0.0) == 0.0 and math.copysign(1, s.round_sig(-0.0)) == 1
    assert s.round_sig(123456) == 123500 and isinstance(s.round_sig(123456), int)
    assert s.round_sig(0.000123456) == 0.0001235
    with pytest.raises(ValueError):
        s.round_sig(float("nan"))


def test_nan_becomes_null_with_missing_flag():
    _, state = s.build_state(_record(candidate={"bend_line_length_mm": float("nan"), "nominal_sweep_deg": None}))
    cand = state["candidate"]
    assert cand["bend_line_length_mm"] is None and cand["bend_line_length_mm_missing"] is True
    assert cand["nominal_sweep_deg"] is None and cand["nominal_sweep_deg_missing"] is True


def test_null_without_missing_flag_is_rejected_by_check():
    _, state = s.build_state(_record())
    state["candidate"]["nominal_radius_mm"] = None
    with pytest.raises(s.StateDenyError, match="_missing"):
        s.check_state(state)


@pytest.mark.parametrize("key", sorted(s.INFERENCE_FORBIDDEN_KEYS))
def test_inference_forbidden_keys_rejected_at_any_depth(key):
    for block in ("candidate", "neighbours", "context", "cad_relation"):
        raw = {"ok_count": 1, "nested": {key: 1}}
        with pytest.raises(s.StateDenyError):
            s.build_state(_record(**{block: {"bend_line_length_mm": 1.0, **raw}}))


@pytest.mark.parametrize(
    "key",
    ["sku", "part_number", "job_ref", "recipe_version_count", "scan_quality_score", "file_size_count",
     "scan_timestamp", "serial", "station", "bend_uid", "candidate_id", "cad_label", "case_score",
     "split", "sku_family_kind", "released_at", "cad_sha256", "operator_choice_kind", "truth_kind",
     "line_start", "line_end", "anchor", "span_endpoints", "axis_direction", "polyline_mm"],
)
def test_identity_and_geometry_keys_rejected(key):
    with pytest.raises(s.StateDenyError):
        s.build_state(_record(candidate={"bend_line_length_mm": 1.0, key: 1.0}))


@pytest.mark.parametrize(
    "value",
    [[1.0, 2.0, 3.0], [1.0, 2.0], [], [{"a_mm": 1.0}]],
)
def test_lists_and_vectors_rejected(value):
    with pytest.raises(s.StateDenyError, match="list"):
        s.build_state(_record(candidate={"bend_line_length_mm": 1.0, "vertex_mm": value}))


@pytest.mark.parametrize(
    "value",
    ["SYN-ONB-A", "23553000_B_FULL_SCAN 1.ply", "a" * 70, "deadbeef1234567890", "2026-10-17T00:00:00Z",
     "line_0123456789abcdef", "BEFORE BENDING  11979003_C", "B1234"],
)
def test_identity_like_string_values_rejected(value):
    with pytest.raises(s.StateDenyError):
        s.build_state(_record(candidate={"bend_line_length_mm": 1.0, "angle_convention": value}))


def test_enum_token_normalises_or_refuses():
    assert s.enum_token("NATIVE SWEEP") == "NATIVE_SWEEP"
    assert s.enum_token("SYN-ONB-A") == "SYN_ONB_A"  # enum_token is for enums; check_state still forbids digits runs
    assert s.enum_token("11979003_C") is None
    assert s.enum_token("") is None and s.enum_token(None) is None


@pytest.mark.parametrize(
    "candidate, match",
    [
        ({"bend_line_length": 1.0}, "unit suffix"),
        ({"bend_line_length_mm": 1.0, "flag": True}, "boolean"),
        ({"bend_line_length_mm": 1.0, "angle": "NATIVE"}, "category suffix"),
        ({"bend_line_length_mm": 1.0, "BadKey_mm": 1.0}, "KEY_FORMAT"),
    ],
)
def test_key_class_rules(candidate, match):
    with pytest.raises(s.StateDenyError, match=match):
        s.build_state(_record(candidate=candidate))


def test_depth_limit():
    deep: dict = {"leaf_mm": 1.0}
    for i in range(10):
        deep = {f"level{i}": deep}
    with pytest.raises(s.StateDenyError, match="nesting"):
        s.build_state(_record(candidate={"bend_line_length_mm": 1.0, "deep": deep}))


def test_unknown_candidate_source_and_top_level_keys_rejected():
    with pytest.raises(s.StateDenyError):
        s.build_state(_record(candidate_source="SOMETHING"))
    _, state = s.build_state(_record())
    state["split_kind"] = "DEV"
    with pytest.raises(s.StateDenyError, match="top-level"):
        s.check_state(state)


def test_ablate_drops_only_the_named_blocks():
    _, state = s.build_state(_record())
    cad_free = s.ablate(state, cad_free=True)
    assert "cad_relation" not in cad_free and cad_free["candidate"] == state["candidate"]
    assert "context" not in s.ablate(state, context_free=True)
    assert "cad_relation" in state  # original untouched
    s.check_state(cad_free)


def test_line_helpers_are_coordinate_free():
    a = s.line_from_endpoints([0, 0, 0], [100, 0, 0])
    b = s.line_from_endpoints([0, 3, 0], [100, 3, 0])
    c = s.line_from_endpoints([50, -50, 0], [50, 50, 0])
    assert s.line_from_endpoints([0, 0, 0], [0, 0, 0]) is None
    m = s.line_pair_metrics(a, b)
    assert m["axis_angle_deg"] == pytest.approx(0.0) and m["transverse_offset_mm"] == pytest.approx(3.0)
    summary = s.neighbour_summary(0, [a, b, c])
    assert summary["parallel_2deg_within_5mm_count"] == 1
    assert summary["nearest_parallel_offset_mm"] == pytest.approx(3.0)
    rel = s.relation_to_reference(c, [(a, {"nominal_sweep_deg": 90.0})])
    assert rel["nearest_axis_angle_deg"] == pytest.approx(90.0)
    assert rel["nearest_nominal_sweep_deg"] == 90.0
    for value in [*summary.values(), *rel.values()]:
        assert not isinstance(value, (list, tuple))


def test_segment_distance_is_the_true_closest_distance():
    L = s.line_from_endpoints
    ref = L([0, 0, 0], [100, 0, 0])
    assert s.segment_distance_mm(ref, L([80, -300, 0], [80, 20, 0])) == pytest.approx(0.0)  # crosses
    assert s.segment_distance_mm(ref, L([300, 0, 0], [400, 0, 0])) == pytest.approx(200.0)  # collinear, far
    assert s.segment_distance_mm(ref, L([50, 5, 3], [50, 5, 9])) == pytest.approx(math.hypot(5, 3))  # skew
    assert s.segment_distance_mm(ref, L([0, 7, 0], [100, 7, 0])) == pytest.approx(7.0)  # parallel
    assert s.segment_distance_mm(ref, L([-10, -10, 0], [-20, -30, 0])) == pytest.approx(math.hypot(10, 10))
    assert s.segment_distance_mm(L([80, -300, 0], [80, 20, 0]), ref) == pytest.approx(0.0)  # symmetric
    near = s.relation_to_reference(L([105, 2, 0], [200, 2, 0]), [(ref, {})])
    far = s.relation_to_reference(L([300, 0, 0], [400, 0, 0]), [(ref, {})])
    assert near["is_within_10mm_of_reference"] is True and far["is_within_10mm_of_reference"] is False


# --------------------------------------------------------------------------
# JSON schema parity
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported_states(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("jev_state")
    path = write_synthetic_inputs(tmp / "in")
    export_onboarding([path], tmp / "out")
    return [json.loads(line)["state"] for line in (tmp / "out" / "states.jsonl").read_text().splitlines()]


def test_every_exported_state_validates_against_the_schema(schema, exported_states):
    validator = jsonschema.Draft202012Validator(schema)
    assert exported_states
    for state in exported_states:
        s.check_state(state)
        errors = list(validator.iter_errors(state))
        assert not errors, errors[0].message


def _mutations():
    return [
        ("sku key", lambda st: st["candidate"].__setitem__("sku", "X")),
        ("vector", lambda st: st["candidate"].__setitem__("vertex_mm", [1.0, 2.0, 3.0])),
        ("forbidden key", lambda st: st["candidate"].__setitem__("points", 3)),
        ("string number", lambda st: st["candidate"].__setitem__("angle_convention", "B12345")),
        ("unsuffixed number", lambda st: st["candidate"].__setitem__("length", 3.0)),
        ("bool name", lambda st: st["candidate"].__setitem__("formed", True)),
        ("top-level extra", lambda st: st.__setitem__("split", "dev")),
        ("bad source", lambda st: st.__setitem__("candidate_source", "HOLDOUT")),
        ("missing false", lambda st: st["candidate"].__setitem__("x_mm_missing", False)),
        ("nested id", lambda st: st["candidate"].__setitem__("sub", {"bend_uid_kind": "A"})),
    ]


@pytest.mark.parametrize("name, mutate", _mutations(), ids=[m[0] for m in _mutations()])
def test_schema_and_code_both_reject_mutations(schema, exported_states, name, mutate):
    state = copy.deepcopy(exported_states[0])
    mutate(state)
    with pytest.raises(s.StateDenyError):
        s.check_state(state)
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(state)), f"schema accepted mutation {name}"


def test_schema_was_generated_from_the_code_constants(schema):
    assert schema["properties"]["candidate_source"]["enum"] == sorted(s.CANDIDATE_SOURCES)
    names = schema["$defs"]["block"]["propertyNames"]["not"]["anyOf"]
    exact = names[0]["pattern"][2:-2].split("|")
    tokens = names[1]["pattern"].split("(")[2].split(")")[0].split("|")
    assert set(exact) == set(s.INFERENCE_FORBIDDEN_KEYS | s.GEOMETRY_FORBIDDEN_KEYS)
    assert set(tokens) == set(s.DENY_TOKENS)
    patterns = schema["$defs"]["block"]["patternProperties"]
    num = next(p for p in patterns if "mm2" in p)
    assert set(num[2:-2].split("|")) == {x[1:] for x in s.UNIT_SUFFIXES}
    cat = next(p for p in patterns if "convention" in p)
    assert set(cat[2:-2].split("|")) == {x[1:] for x in s.CATEGORY_SUFFIXES}


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def test_primary_noul_lists_the_documented_confusers():
    noul = q.QUESTION_SET_V1["is_real_bend"]
    assert noul["type"] == "noul"
    assert noul["instructions"]["question"] == "Is this candidate a real press-brake bend line?"
    confusers = " ".join(noul["criteria"]["false"]["confusers"]).lower()
    for word in ("outer contour", "hole", "rolled", "joggle", "second copy", "tongue", "noise"):
        assert word in confusers
    choice = q.QUESTION_SET_V1["candidate_kind"]
    assert set(choice["criteria"]) == {"bend", "outline", "hole_wall", "rolled_band", "step", "duplicate_skin",
                                       "tongue_silhouette", "artifact", "other"}
    assert len(q.QUESTION_SET_V1["evidence_quality"]["criteria"]) == 4


def test_question_set_hashes_are_stable_and_variant_specific():
    v1 = q.select_questions("v1")
    verb = q.select_questions("verbalized_v1")
    assert q.question_set_sha256(v1, "v1") == q.question_set_sha256(q.select_questions("v1"), "v1")
    assert q.question_set_sha256(v1, "v1") != q.question_set_sha256(verb, "verbalized_v1")
    with_choice = q.select_questions("v1", ["is_real_bend", "candidate_kind"])
    assert q.question_set_sha256(with_choice) != q.question_set_sha256(v1)
    mutated = copy.deepcopy(v1)
    mutated["is_real_bend"]["criteria"]["true"] += " "
    assert q.question_set_sha256(mutated) != q.question_set_sha256(v1)
    with pytest.raises(ValueError):
        q.select_questions("v1", ["candidate_kind"])
    with pytest.raises(KeyError):
        q.select_questions("v2")
    # select_questions returns copies: mutating one must not change the frozen set
    v1["is_real_bend"]["type"] = "score"
    assert q.QUESTION_SET_V1["is_real_bend"]["type"] == "noul"


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


def _body(**overrides):
    _, state = s.build_state(_record())
    body = {"state": state, "model": "jev-1.13.0",
            "questions": q.select_questions("v1", ["is_real_bend", "candidate_kind", "evidence_quality"])}
    body.update(overrides)
    return body


def test_valid_request_passes_and_bytes_are_canonical():
    body = r.build_request(_body()["state"], "jev-1.13.0", q.select_questions())
    info = r.validate_request(body)
    raw = r.body_bytes(body)
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert info["bytes"] == len(raw)
    assert info["est_tokens"] == math.ceil(len(raw.decode("utf-8")) / 3)
    assert r.body_sha256(body) == r.body_sha256(json.loads(raw))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda b: b.pop("model"), "missing"),
        (lambda b: b.__setitem__("extra", 1), "unexpected top-level"),
        (lambda b: b.__setitem__("questions", {}), "non-empty map"),
        (lambda b: b["questions"]["is_real_bend"].__setitem__("type", "bool"), "type must be"),
        (lambda b: b["questions"]["is_real_bend"].__setitem__("instructions", " "), "instructions"),
        (lambda b: b["questions"]["is_real_bend"]["criteria"].__setitem__("maybe", "x"), "only 'true'/'false'"),
        (lambda b: b["questions"]["candidate_kind"].__setitem__("criteria", {f"o{i}": None for i in range(256)}), "max 255"),
        (lambda b: b["questions"]["evidence_quality"].__setitem__("criteria", ["only one"]), "levels"),
        (lambda b: b["questions"]["evidence_quality"].__setitem__("criteria", [str(i) for i in range(11)]), "levels"),
        (lambda b: b["questions"]["is_real_bend"].__setitem__("note", "x"), "unexpected keys"),
        (lambda b: b["state"]["candidate"].__setitem__("bend_line_length_mm", float("inf")), "non-finite"),
    ],
)
def test_request_rule_violations(mutate, match):
    body = _body()
    mutate(body)
    with pytest.raises(r.RequestInvalid, match=match):
        r.validate_request(body)


def test_nesting_limit_on_instructions():
    nested: object = "deep"
    for _ in range(65):
        nested = {"x": nested}
    body = _body()
    body["questions"]["is_real_bend"]["instructions"] = nested
    with pytest.raises(r.RequestInvalid, match="nesting"):
        r.validate_request(body)


def test_token_limits_per_request_and_per_question():
    big_state = {"text": "x" * (3 * 33_000)}
    body = {"state": big_state, "model": "jev-1.13.0", "questions": q.select_questions()}
    with pytest.raises(r.RequestInvalid, match="tokens with the state"):
        r.validate_request(body, check_state_rules=False)
    many = {f"q{i}": {"type": "noul", "instructions": "y" * 30_000} for i in range(8)}
    body = {"state": {"text": "short"}, "model": "jev-1.13.0", "questions": many}
    with pytest.raises(r.RequestInvalid, match="per request"):
        r.validate_request(body, check_state_rules=False)


def test_request_rejects_a_state_that_breaks_the_deny_list():
    body = _body()
    body["state"]["candidate"]["sku_kind"] = "X"
    with pytest.raises(r.RequestInvalid, match="state"):
        r.validate_request(body)
