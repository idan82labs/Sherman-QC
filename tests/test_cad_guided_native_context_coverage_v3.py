"""Independent symbolic coverage checks; no native geometry or physical input.

The expected incidences below are written out from a two-face drawing. Tests do
not call the validator's private topology/context helpers to construct an oracle.
"""

import ast
import builtins
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
import sys
from types import ModuleType

import pytest

from backend import cad_guided_native_context_coverage_v3 as coverage


RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
                       process_tree_rss_bytes=1024**3, output_bytes=1024**2)
SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_native_context_coverage_v3.py"

# Native IDs: solid0, shell1, faces2/3, wires4/5, shared edge6, vertices7/8.
# An edge use on either face has the SAME two endpoint identities. Its second
# use reverses endpoint order and orientation, without inventing new entities.
EXPECTED_PAIRS = (
    (1, 0, "SHELL->SOLID"),
    (4, 2, "WIRE->FACE"), (5, 3, "WIRE->FACE"),
    (6, 2, "EDGE->FACE"), (6, 3, "EDGE->FACE"),
    (7, 2, "VERTEX->FACE"), (7, 3, "VERTEX->FACE"), (7, 6, "VERTEX->EDGE"),
    (8, 2, "VERTEX->FACE"), (8, 3, "VERTEX->FACE"), (8, 6, "VERTEX->EDGE"),
)
HARD_MAXIMA = dict(max_entities=2048, max_occurrences=4096, max_depth=32,
    max_details=1024, max_expected_context_pairs=4096, max_observed_context_pairs=4096,
    max_statuses_per_list=32, max_total_status_values=32768, max_ancestor_steps=131072)


def _fixture():
    kinds = ("SOLID", "SHELL", "FACE", "FACE", "WIRE", "WIRE", "EDGE", "VERTEX", "VERTEX")
    type_indices = (0, 0, 0, 1, 0, 1, 0, 0, 1)
    occurrence_counts = (1, 1, 1, 1, 1, 1, 2, 2, 2)
    entities = [dict(native_id=i, type=kind, type_index=type_indices[i],
                     occurrence_count=occurrence_counts[i], location_is_identity=True)
                for i, kind in enumerate(kinds)]
    occurrences = [
        [0, -1, 0], [1, 0, 0], [2, 1, 0], [4, 2, 0],
        [6, 3, 0], [7, 4, 0], [8, 4, 1],
        [3, 1, 1], [5, 7, 0], [6, 8, 1], [8, 9, 0], [7, 9, 1],
    ]
    # This list is an explicit symbolic incidence oracle, not an ancestor walk.
    owners = ((), (0,), (), (), (2,), (3,), (2, 3), (2, 3, 6), (2, 3, 6))
    details = [dict(native_id=i, self_status_codes=[0], subtree_valid=True,
                    contexts=[[owner, [0]] for owner in owners[i]],
                    detail_complete=True, errors=[]) for i in range(9)]
    return entities, occurrences, details


def _run(inputs=None, **kwargs):
    return coverage.validate_native_context_coverage_v3(
        *(_fixture() if inputs is None else inputs), native_valid=kwargs.pop("native_valid", None), **kwargs)


def _no_admission(result):
    assert result["local_admission_allowed"] is False
    assert result["global_admission_allowed"] is False
    assert result["production_use_allowed"] is False
    assert result["feature_claims_produced"] is False
    assert result["observed_bend_count"] is None


def _unavailable(result, reason=None):
    assert result["status"] == "UNAVAILABLE", result
    assert result["coverage_complete"] is False and result["counts"] is None
    assert result["self_dispositions"] == result["context_dispositions"] == []
    assert result["reasons"]
    if reason is not None:
        assert result["reasons"] == [reason]
    _no_admission(result)


def _pair_rows(result):
    return {(row["native_id"], row["context_id"]): row for row in result["context_dispositions"]}


def _clear_symbolic_oracle(result):
    assert result["status"] == "COVERAGE_COMPLETE" and result["coverage_complete"] is True
    assert result["reasons"] == []
    assert [(row["native_id"], row["context_id"], row["relation"])
            for row in result["context_dispositions"]] == list(EXPECTED_PAIRS)
    assert [row["native_id"] for row in result["self_dispositions"]] == list(range(9))
    assert all(row["disposition"] == "CLEAR" and row["status_codes"] == [0]
               for row in result["self_dispositions"] + result["context_dispositions"])
    # Depths 0,1,2,3,4,5,5,2,3,4,5,5 sum to39, including irrelevant ancestors.
    assert result["counts"] == dict(entities=9, occurrences=12,
        required_self_records=9, observed_detail_records=9,
        expected_context_pairs=11, observed_context_pairs=11,
        self=dict(CLEAR=9, ERROR=0, MISSING=0), contexts=dict(CLEAR=11, ERROR=0, MISSING=0),
        incomplete_details=0, subtree_invalid=0, status_values=20, ancestor_steps=39)
    _no_admission(result)


@pytest.mark.parametrize("native_valid", (None, False, True))
def test_explicit_two_face_denominator_never_grants_admission(native_valid):
    result = _run(native_valid=native_valid, source_report_sha256="a1"*32)
    _clear_symbolic_oracle(result)
    assert result["native_valid"] is native_valid
    assert result["source_report_sha256"] == "a1"*32
    assert result["source_binding"] == "CALLER_DECLARED_DIGEST_NOT_AUTHENTICATED_BYTES_OR_NATIVE_IDENTITY"


@pytest.mark.parametrize("fault,missing_self,missing_pairs,status_values", (
    ("second_edge_face", (), ((6, 3),), 19),
    ("vertex_faces", (), ((7, 2), (7, 3), (8, 2), (8, 3)), 16),
    ("edge_detail", (6,), ((6, 2), (6, 3)), 17),
    ("empty_self", (6,), (), 19),
    ("empty_context", (), ((6, 3),), 19),
    ("incomplete_edge", (6,), ((6, 2), (6, 3)), 20),
    ("missing_subtree", (6,), ((6, 2), (6, 3)), 20),
    ("detail_error", (6,), ((6, 2), (6, 3)), 20),
    ("unavailable_tolerance", (6,), ((6, 2), (6, 3)), 20),
))
def test_missing_evidence_keeps_exact_complete_denominator(fault, missing_self, missing_pairs, status_values):
    inputs = _fixture()
    details = inputs[2]
    if fault == "second_edge_face":
        details[6]["contexts"].pop()
    elif fault == "vertex_faces":
        details[7]["contexts"] = [[6, [0]]]
        details[8]["contexts"] = [[6, [0]]]
    elif fault == "edge_detail":
        details.pop(6)
    elif fault == "empty_self":
        details[6]["self_status_codes"] = []
    elif fault == "empty_context":
        details[6]["contexts"][1][1] = []
    elif fault == "incomplete_edge":
        details[6]["detail_complete"] = False
    elif fault == "missing_subtree":
        details[6].pop("subtree_valid")
    elif fault == "detail_error":
        details[6]["errors"] = ["INVENTED_CACHE_READ_FAILURE"]
    else:
        details[6].update(native_tolerance_mm=None, native_tolerance_numeric_valid=False)
    result = _run(inputs, native_valid=False)
    assert result["status"] == "COVERAGE_INCOMPLETE" and not result["coverage_complete"]
    assert result["counts"]["expected_context_pairs"] == 11
    assert result["counts"]["required_self_records"] == 9
    assert result["counts"]["status_values"] == status_values
    assert {row["native_id"] for row in result["self_dispositions"]
            if row["disposition"] == "MISSING"} == set(missing_self)
    assert {pair for pair, row in _pair_rows(result).items() if row["disposition"] == "MISSING"} == set(missing_pairs)
    assert result["counts"]["self"]["ERROR"] == result["counts"]["contexts"]["ERROR"] == 0
    _no_admission(result)


@pytest.mark.parametrize("incomplete", (False, True))
def test_context_error_is_not_overridden_by_clear_edge_self_or_incomplete_details(incomplete):
    inputs = _fixture()
    inputs[2][6]["contexts"][1][1] = [255]
    if incomplete:
        inputs[2][6].update(detail_complete=False, errors=["INVENTED_OTHER_MISSING_CONTEXT"])
    result = _run(inputs, native_valid=False)
    assert result["status"] == ("COVERAGE_INCOMPLETE" if incomplete else "COVERAGE_COMPLETE")
    assert _pair_rows(result)[(6, 3)]["disposition"] == "ERROR"
    assert _pair_rows(result)[(6, 3)]["status_codes"] == [255]
    assert result["self_dispositions"][6]["subtree_valid"] is True
    assert result["self_dispositions"][6]["disposition"] == ("MISSING" if incomplete else "CLEAR")
    assert result["counts"]["contexts"]["ERROR"] == 1
    _no_admission(result)


def test_negative_self_statuses_are_sorted_not_suppressed():
    inputs = _fixture()
    inputs[2][8].update(self_status_codes=[255, 1, 7], subtree_valid=False)
    result = _run(inputs, native_valid=False)
    assert result["status"] == "COVERAGE_COMPLETE"
    assert result["self_dispositions"][8]["status_codes"] == [1, 7, 255]
    assert result["self_dispositions"][8]["disposition"] == "ERROR"
    assert result["counts"]["subtree_invalid"] == 1
    assert result["counts"]["status_values"] == 22
    _no_admission(result)


def test_entity_detail_context_branch_and_cumulative_orientation_permutations_are_semantic():
    original = _run()
    entities, occurrences, details = _fixture()
    order = [0, 1, 7, 8, 9, 11, 10, 2, 3, 4, 6, 5]
    inverse = {old: new for new, old in enumerate(order)}
    changed = [[occurrences[old][0], -1 if occurrences[old][1] == -1 else inverse[occurrences[old][1]],
                3-occurrences[old][2]] for old in order]
    for detail in details:
        detail["contexts"].reverse()
    result = _run((entities[::-1], changed, details[::-1]))
    assert result == original


def test_same_face_seam_occurrences_keep_multiplicity_without_duplicate_context_pairs():
    inputs = _fixture()
    inputs[1].extend([[6, 3, 2], [8, 12, 2], [7, 12, 3]])
    for identifier in (6, 7, 8):
        inputs[0][identifier]["occurrence_count"] += 1
    result = _run(inputs)
    assert result["status"] == "COVERAGE_COMPLETE"
    assert [(r["native_id"], r["context_id"], r["relation"])
            for r in result["context_dispositions"]] == list(EXPECTED_PAIRS)
    assert result["counts"]["occurrences"] == 15
    assert result["counts"]["ancestor_steps"] == 53
    assert result["counts"]["expected_context_pairs"] == result["counts"]["observed_context_pairs"] == 11
    assert result["counts"]["status_values"] == 20
    _no_admission(result)


def test_repeated_identity_cannot_change_child_identity_multiplicity():
    inputs = _fixture()
    inputs[1][-1][0] = 8
    inputs[0][7]["occurrence_count"] = 1
    inputs[0][8]["occurrence_count"] = 3
    _unavailable(_run(inputs), "INCONSISTENT_NATIVE_CHILD_IDENTITIES")


def test_recursive_native_container_refuses_even_with_contiguous_ids_and_counts():
    entity = dict(native_id=0, type="COMPOUND", type_index=0, occurrence_count=2,
                  location_is_identity=True)
    # Finite unrolling cannot preserve the same child multiset at the final use;
    # either this consistency check or an explicit recursive-ID check must refuse.
    _unavailable(_run(([entity], [[0, -1, 0], [0, 0, 1]], [])))


@pytest.mark.parametrize("native_valid", (None, False, True))
def test_container_only_empty_required_denominator_is_not_admission(native_valid):
    entity = dict(native_id=0, type="COMPOUND", type_index=0, occurrence_count=1,
                  location_is_identity=False)
    result = _run(([entity], [[0, -1, 0]], []), native_valid=native_valid)
    assert result["status"] == "COVERAGE_COMPLETE" and result["coverage_complete"]
    assert result["counts"]["required_self_records"] == result["counts"]["expected_context_pairs"] == 0
    assert result["self_dispositions"] == result["context_dispositions"] == []
    _no_admission(result)


def test_container_self_detail_is_not_an_expected_witness():
    entity = dict(native_id=0, type="COMPOUND", type_index=0, occurrence_count=1,
                  location_is_identity=True)
    detail = dict(native_id=0, self_status_codes=[0], contexts=[], subtree_valid=True,
                  detail_complete=True, errors=[])
    _unavailable(_run(([entity], [[0, -1, 0]], [detail])), "CONTAINER_DETAIL_UNSUPPORTED")


@pytest.mark.parametrize("argument", range(3))
def test_empty_inputs_distinguish_missing_details_from_absent_topology(argument):
    inputs = list(_fixture())
    inputs[argument] = []
    result = _run(inputs)
    if argument < 2:
        _unavailable(result)
    else:
        assert result["status"] == "COVERAGE_INCOMPLETE"
        assert result["counts"]["self"] == dict(CLEAR=0, ERROR=0, MISSING=9)
        assert result["counts"]["contexts"] == dict(CLEAR=0, ERROR=0, MISSING=11)
        assert result["counts"]["observed_detail_records"] == result["counts"]["status_values"] == 0
        _no_admission(result)


def test_optional_bounded_metadata_and_nonidentity_location_do_not_grant_or_block_admission():
    inputs = _fixture()
    inputs[0][-1]["location_is_identity"] = False
    inputs[2][-1].update(native_tolerance_mm=0., native_tolerance_numeric_valid=True,
        geometry_type="x"*128, degenerated=False, same_parameter=True, same_range=True)
    _clear_symbolic_oracle(_run(inputs, native_valid=False))


LATE_FAULTS = (
    "duplicate_entity", "entity_gap", "duplicate_type_index", "type_index_gap", "bad_count",
    "entity_extra", "entity_missing", "entity_bool_id", "entity_float_index", "entity_bad_location",
    "entity_unknown_type", "second_root", "forward_parent", "parent_self", "bad_transition",
    "orientation_bool", "orientation_float", "orientation_over", "occurrence_extra", "occurrence_tuple",
    "duplicate_detail", "detail_unknown_id", "detail_extra", "detail_missing", "detail_bool_id",
    "detail_nonbool_complete", "duplicate_context", "unknown_owner", "irrelevant_context",
    "context_tuple", "context_extra", "context_bool_owner", "duplicate_status", "mixed_zero_error",
    "status_bool", "status_float", "status_negative", "status_over", "status_tuple",
    "tolerance_nan", "tolerance_negative", "tolerance_bool", "tolerance_flag_contradiction",
    "metadata_nonbool", "geometry_empty", "geometry_oversize", "error_empty", "error_oversize", "errors_over",
)


@pytest.mark.parametrize("fault", LATE_FAULTS)
def test_late_invalid_input_clears_all_dispositions_and_counts(fault):
    entities, occurrences, details = _fixture()
    e, o, d = entities[-1], occurrences[-1], details[-1]
    if fault == "duplicate_entity": e["native_id"] = 7
    elif fault == "entity_gap": e["native_id"] = 9
    elif fault == "duplicate_type_index": e["type_index"] = 0
    elif fault == "type_index_gap": e["type_index"] = 2
    elif fault == "bad_count": e["occurrence_count"] = 1
    elif fault == "entity_extra": e["verified"] = True
    elif fault == "entity_missing": e.pop("location_is_identity")
    elif fault == "entity_bool_id": e["native_id"] = True
    elif fault == "entity_float_index": e["type_index"] = 1.
    elif fault == "entity_bad_location": e["location_is_identity"] = 1
    elif fault == "entity_unknown_type": e["type"] = "TRIANGLE"
    elif fault == "second_root": o[1] = -1
    elif fault == "forward_parent": o[1] = len(occurrences)
    elif fault == "parent_self": o[1] = len(occurrences)-1
    elif fault == "bad_transition": o[1] = 1
    elif fault == "orientation_bool": o[2] = True
    elif fault == "orientation_float": o[2] = 1.
    elif fault == "orientation_over": o[2] = 4
    elif fault == "occurrence_extra": o.append(0)
    elif fault == "occurrence_tuple": occurrences[-1] = tuple(o)
    elif fault == "duplicate_detail": details.append(deepcopy(d))
    elif fault == "detail_unknown_id": d["native_id"] = 9
    elif fault == "detail_extra": d["geometry_verified"] = True
    elif fault == "detail_missing": d.pop("errors")
    elif fault == "detail_bool_id": d["native_id"] = True
    elif fault == "detail_nonbool_complete": d["detail_complete"] = 1
    elif fault == "duplicate_context": d["contexts"].append(deepcopy(d["contexts"][-1]))
    elif fault == "unknown_owner": d["contexts"][-1][0] = 9
    elif fault == "irrelevant_context": d["contexts"][-1][0] = 0
    elif fault == "context_tuple": d["contexts"][-1] = tuple(d["contexts"][-1])
    elif fault == "context_extra": d["contexts"][-1].append(0)
    elif fault == "context_bool_owner": d["contexts"][-1][0] = True
    elif fault == "duplicate_status": d["contexts"][-1][1] = [1, 1]
    elif fault == "mixed_zero_error": d["contexts"][-1][1] = [0, 1]
    elif fault == "status_bool": d["contexts"][-1][1] = [True]
    elif fault == "status_float": d["contexts"][-1][1] = [1.]
    elif fault == "status_negative": d["contexts"][-1][1] = [-1]
    elif fault == "status_over": d["contexts"][-1][1] = [256]
    elif fault == "status_tuple": d["contexts"][-1][1] = (0,)
    elif fault == "tolerance_nan": d["native_tolerance_mm"] = float("nan")
    elif fault == "tolerance_negative": d["native_tolerance_mm"] = -1.
    elif fault == "tolerance_bool": d["native_tolerance_mm"] = False
    elif fault == "tolerance_flag_contradiction": d.update(native_tolerance_mm=None, native_tolerance_numeric_valid=True)
    elif fault == "metadata_nonbool": d["same_parameter"] = 1
    elif fault == "geometry_empty": d["geometry_type"] = ""
    elif fault == "geometry_oversize": d["geometry_type"] = "x"*129
    elif fault == "error_empty": d["errors"] = [""]
    elif fault == "error_oversize": d["errors"] = ["x"*161]
    else: d["errors"] = ["x"]*5
    result = _run((entities, occurrences, details), native_valid=False)
    _unavailable(result)
    assert result["native_valid"] is False


@pytest.mark.parametrize("fault", ("self_true_subtree", "global_true_context_error", "global_true_false_subtree"))
def test_contradictory_native_observations_are_whole_refusals(fault):
    inputs = _fixture()
    native_valid = False
    if fault == "self_true_subtree":
        inputs[2][8]["self_status_codes"] = [1]
    elif fault == "global_true_context_error":
        inputs[2][8]["contexts"][-1][1] = [1]
        native_valid = True
    else:
        inputs[2][8]["subtree_valid"] = False
        native_valid = True
    _unavailable(_run(inputs, native_valid=native_valid))


@pytest.mark.parametrize("argument", range(3))
@pytest.mark.parametrize("value", (None, (), {}, "invented"))
def test_top_level_containers_are_plain_lists(argument, value):
    inputs = list(_fixture())
    inputs[argument] = value
    _unavailable(_run(inputs))


@pytest.mark.parametrize("value", (0, 1, "true", [], {}))
def test_native_valid_requires_bool_or_none(value):
    _unavailable(_run(native_valid=value), "NATIVE_VALID_SCHEMA")


@pytest.mark.parametrize("value", ("", "a"*63, "a"*65, "A"*64, "g"*64, 0, True, []))
def test_malformed_declared_provenance_cannot_survive(value):
    result = _run(source_report_sha256=value)
    _unavailable(result, "SOURCE_REPORT_DIGEST_SCHEMA")
    assert result["source_report_sha256"] is None


EXACT_BUDGETS = (
    ("max_entities", 9), ("max_occurrences", 12), ("max_depth", 5),
    ("max_details", 9), ("max_expected_context_pairs", 11), ("max_observed_context_pairs", 11),
    ("max_statuses_per_list", 2), ("max_total_status_values", 20), ("max_ancestor_steps", 39),
)


@pytest.mark.parametrize("field,boundary", EXACT_BUDGETS)
@pytest.mark.parametrize("below", (False, True))
def test_every_budget_exact_boundary_and_one_below(field, boundary, below):
    inputs = _fixture()
    if field == "max_statuses_per_list":
        inputs[2][-1].update(self_status_codes=[1, 2], subtree_valid=False)
    limits = replace(coverage.NativeContextCoverageLimitsV3(), **{field: boundary-int(below)})
    result = _run(inputs, native_valid=False, limits=limits)
    if below:
        _unavailable(result)
    else:
        assert result["status"] == "COVERAGE_COMPLETE", result
        assert result["counts"]["expected_context_pairs"] == 11
        assert result["counts"]["status_values"] == (21 if field == "max_statuses_per_list" else 20)
        _no_admission(result)


def test_nine_public_limit_defaults_match_independent_declared_maxima():
    limits = coverage.NativeContextCoverageLimitsV3()
    assert {item.name: getattr(limits, item.name) for item in fields(limits)} == HARD_MAXIMA


@pytest.mark.parametrize("field", tuple(HARD_MAXIMA))
@pytest.mark.parametrize("kind", ("raised", "bool", "float", "zero"))
def test_mutated_limits_cannot_raise_or_change_exact_integer_contract(field, kind):
    limits = coverage.NativeContextCoverageLimitsV3()
    value = {"raised": HARD_MAXIMA[field]+1, "bool": True, "float": float(HARD_MAXIMA[field]), "zero": 0}[kind]
    object.__setattr__(limits, field, value)
    _unavailable(_run(limits=limits), "LIMITS_OUTSIDE_FIXED_CAPS")


@pytest.mark.parametrize("field", tuple(HARD_MAXIMA))
def test_bypass_mutating_public_default_cannot_mutate_hard_maximum(field):
    default = coverage.validate_native_context_coverage_v3.__kwdefaults__["limits"]
    saved = getattr(default, field)
    try:
        object.__setattr__(default, field, HARD_MAXIMA[field]+1)
        _unavailable(_run(), "LIMITS_OUTSIDE_FIXED_CAPS")
    finally:
        object.__setattr__(default, field, saved)


@pytest.mark.parametrize("admitted,changed", ((8, 9), (9, 8)))
def test_post_validation_caller_mutation_cannot_change_captured_limit(monkeypatch, admitted, changed):
    limits = replace(coverage.NativeContextCoverageLimitsV3(), max_entities=admitted)
    original = coverage._entities
    calls = []
    def mutate_caller(values, captured):
        calls.append(None)
        assert captured is not limits and captured.max_entities == admitted
        object.__setattr__(limits, "max_entities", changed)
        return original(values, captured)
    monkeypatch.setattr(coverage, "_entities", mutate_caller)
    result = _run(limits=limits)
    assert len(calls) == 1
    if admitted == 8:
        _unavailable(result, "ENTITY_BUDGET_OR_SCHEMA")
    else:
        _clear_symbolic_oracle(result)


def test_extended_limits_type_cannot_add_unreviewed_policy_fields():
    class Extended(coverage.NativeContextCoverageLimitsV3):
        native_admission = True
    _unavailable(_run(limits=Extended()), "LIMITS_SCHEMA")


def test_input_containers_are_not_output_aliases():
    inputs = _fixture()
    result = _run(inputs)
    saved = deepcopy(result)
    for detail in inputs[2]:
        detail["self_status_codes"].append(255)
        for context in detail["contexts"]:
            context[1].append(255)
            context[0] = -100
        detail["contexts"].clear()
        detail["errors"].append("MUTATED_AFTER_RETURN")
        detail.clear()
    for occurrence in inputs[1]:
        occurrence[:] = [False, False, False]
    for entity in inputs[0]:
        entity.clear()
    for container in inputs:
        container.clear()
    assert result == saved
    result["context_dispositions"][0]["status_codes"].append(255)
    assert saved["context_dispositions"][0]["status_codes"] == [0]


def test_import_and_public_paths_use_only_bounded_pure_python_contract(monkeypatch):
    # Read only this reviewed source BEFORE blocking open; no part/data/native
    # fixture is read. The executed module then receives an import allowlist and
    # an open() trap, including while its public entrypoint is exercised.
    source = SOURCE.read_text()
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert imports == {"collections", "dataclasses", "math"}
    original_import = builtins.__import__
    seen = []
    def import_only_standard_contract(name, globals=None, locals=None, fromlist=(), level=0):
        assert level == 0 and name in ("collections", "dataclasses", "math")
        seen.append(name)
        return original_import(name, globals, locals, fromlist, level)
    def forbidden_open(*args, **kwargs):
        pytest.fail("Pure coverage module attempted file access")
    name = "invented_native_context_coverage_no_io"
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": import_only_standard_contract, "open": forbidden_open}
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    _clear_symbolic_oracle(module.validate_native_context_coverage_v3(*_fixture(), native_valid=None))
    incomplete = _fixture()
    incomplete[2].pop()
    result = module.validate_native_context_coverage_v3(*incomplete, native_valid=False)
    assert result["status"] == "COVERAGE_INCOMPLETE"
    _no_admission(result)
    bad = _fixture()
    bad[1][-1][2] = True
    _unavailable(module.validate_native_context_coverage_v3(*bad, native_valid=False))
    assert set(seen) == {"collections", "dataclasses", "math"}
