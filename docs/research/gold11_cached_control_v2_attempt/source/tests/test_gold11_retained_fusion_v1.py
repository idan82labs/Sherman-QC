"""Independent pure retained-fusion contracts; invented records only.

The legacy script is parsed, never imported or invoked. The isolated metric
dependency executes ONLY its exact pair_metrics/Thresholds/is_match definitions;
no scorer, SciPy, defaults, physical payload, reference, inference or app starts.
Mocks are explicitly local admission/accounting fault controls, not cache replay.
A0 metadata omission, alias precedence and duplicate legacy IDs are intentional.
Per-case snapshots are not atomic authentication of all caller-owned payloads.
Prospective root-reviewed pure gate:60s wall/45s CPU/1GiB tree RSS/1MiB stdout.
"""
import ast
import copy
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"scripts/gold11_retained_fusion_v1.py"
LEGACY_SOURCE = ROOT/"scripts/diagnose_f1_cad_scan_fusion_gold11.py"
METRIC_SOURCE = ROOT/"scripts/evaluate_f1_against_manual_gold.py"
VARIANTS = ("scan_only_a0", "cad_only_gated", "intersection", "union")
CASE_IDS = tuple(f"invented_case_{index:02d}" for index in (7, 0, 10, 2, 9, 1, 6, 3, 8, 4, 5))
RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
    process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)


@pytest.fixture
def fusion():
    package = "_independent_retained_fusion_v1"
    names = (package, package+".evaluate_f1_against_manual_gold",
             package+".gold11_retained_fusion_v1")
    assert not any(name in sys.modules for name in names)
    parent = ModuleType(package)
    parent.__path__ = [str(SOURCE.parent)]
    sys.modules[package] = parent
    try:
        metric = ModuleType(names[1])
        sys.modules[names[1]] = metric
        metric.__dict__.update(np=np, math=math, dataclass=dataclass, Any=Any)
        tree = ast.parse(METRIC_SOURCE.read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name in ("Thresholds", "pair_metrics", "is_match")]
        assert [node.name for node in definitions] == ["Thresholds", "pair_metrics", "is_match"]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(METRIC_SOURCE), "exec"), metric.__dict__)
        spec = importlib.util.spec_from_file_location(names[2], SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[names[2]] = module
        spec.loader.exec_module(module)
        yield module, metric
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


def _line(identifier, y=0., start=0., end=10.):
    return dict(bend_id=identifier, line_start=[start, y, 0.], line_end=[end, y, 0.])


def _payloads(a_rows=None, c_rows=None, case_id=CASE_IDS[0]):
    a0 = dict(schema_version=1, case_id=case_id,
        kind="sherman_qc_f1_typed_union_counted_components",
        algorithm="f1_typed_union_exact_birth_v1", coordinate_frame="scan_raw",
        owned_bend_regions=[] if a_rows is None else a_rows)
    cad = dict(schema_version=1, case_id=case_id,
        kind="sherman_qc_f1_cad_guided_scan_line_predictions",
        route="blind_cad_ply_detected_lines", research_only=True, cad_guided_zero_shot=True,
        source_only=False, predictions=[] if c_rows is None else c_rows)
    return a0, cad


def _cohort():
    pairs = {case: _payloads(case_id=case) for case in reversed(CASE_IDS)}
    return ({case: value[0] for case, value in pairs.items()},
            {case: value[1] for case, value in pairs.items()})


def _one(fusion, a_rows=None, c_rows=None):
    helper, _ = fusion
    a0, cad = _payloads(a_rows, c_rows)
    return helper._case(a0, cad, CASE_IDS[0])


def _cleared(row, status=None, reason=None):
    assert row["status"] != "COMPLETE"
    if status is not None:
        assert row["status"] == status
    if reason is not None:
        assert row["reason"] == reason
    assert row["usable_variants"] is False
    assert row["variants"] == {key: [] for key in VARIANTS}
    assert row["cached_line_counts"] == {key: None for key in VARIANTS}
    assert row["observed_bend_count"] is None and row["production_use_allowed"] is False
    assert row["count_semantics"] == "CACHED_LINE_CARDINALITY_NOT_OBSERVED_BENDS"


def _ids(rows):
    return [row["bend_id"] for row in rows]


def _function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize("name", ("_line", "_duplicate", "_normalized_a0"))
def test_three_original_pure_functions_are_exact_ast_copies(name):
    assert ast.dump(_function(SOURCE, name), include_attributes=False) == ast.dump(
        _function(LEGACY_SOURCE, name), include_attributes=False)


def test_all_six_original_variant_statements_and_their_order_are_preserved():
    main = _function(LEGACY_SOURCE, "main")
    loop = next(node for node in main.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "entry")
    targets = {"scan_rows", "cad_rows", "intersection", "union", "rows_by_variant"}
    original = [node for node in loop.body if
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in targets
                                            for target in node.targets)
        or isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "cad"]
    function = _function(SOURCE, "_original_variants")
    assert len(original) == 6 and len(function.body) == 7
    assert [ast.dump(node, include_attributes=False) for node in function.body[:6]] == [
        ast.dump(node, include_attributes=False) for node in original]
    assert ast.unparse(function.body[-1]) == "return rows_by_variant"
    assert [argument.arg for argument in function.args.args] == ["a0_payload", "cad_payload", "_duplicate"]


def test_all_eleven_caller_ids_and_empty_variants_are_retained_in_declared_order(fusion):
    helper, _ = fusion
    a0, cad = _cohort()
    result = helper.reconstruct_gold11_retained_fusion_v1(
        a0_results_by_case=a0, cad_results_by_case=cad, case_ids=list(CASE_IDS))
    assert result["status"] == "COMPLETE" and result["all_cases_usable"] is True
    assert result["case_order"] == list(CASE_IDS) and list(result["cases"]) == list(CASE_IDS)
    assert result["case_count"] == 11 and result["failed_case_count"] == 0
    for case, row in result["cases"].items():
        assert row["case_id"] == case and row["status"] == "COMPLETE" and row["usable_variants"] is True
        assert row["variants"] == {name: [] for name in VARIANTS}
        assert row["cached_line_counts"] == {name: 0 for name in VARIANTS}
        assert row["work"]["pair_checks_attempted"] == row["work"]["pair_checks_preflight_upper_bound"] == 0


@pytest.mark.parametrize("field,value,expected", (
    ("axis_angle_deg", 5., True), ("axis_angle_deg", math.nextafter(5., math.inf), False),
    ("transverse_offset_mm", 4., True), ("transverse_offset_mm", math.nextafter(4., math.inf), False),
    ("shorter_span_overlap", .6, True), ("shorter_span_overlap", math.nextafter(.6, 0.), False),
))
def test_exact_duplicate_gate_boundaries_with_explicit_metric_stub(fusion, monkeypatch, field, value, expected):
    helper, _ = fusion
    metrics = dict(axis_angle_deg=0., transverse_offset_mm=0., shorter_span_overlap=1.)
    metrics[field] = value
    monkeypatch.setattr(helper, "pair_metrics", lambda left, right: metrics)
    assert helper._duplicate(_line("a"), _line("b")) is expected


@pytest.mark.parametrize("geometry,expected", (
    ("same", True), ("reversed", True), ("offset_equal", True), ("offset_above", False),
    ("overlap_equal", True), ("overlap_below", False), ("angle_four", True), ("angle_six", False),
))
def test_duplicate_uses_actual_original_metrics_on_invented_lines(fusion, geometry, expected):
    helper, _ = fusion
    left, right = _line("a"), _line("b")
    if geometry == "reversed":
        right["line_start"], right["line_end"] = right["line_end"], right["line_start"]
    elif geometry.startswith("offset"):
        y = 4. if geometry == "offset_equal" else 4.+2**-20
        right = _line("b", y=y)
    elif geometry.startswith("overlap"):
        start = 4. if geometry == "overlap_equal" else 4.+2**-20
        right = _line("b", start=start, end=start+10.)
    elif geometry.startswith("angle"):
        angle = math.radians(4. if geometry == "angle_four" else 6.)
        vector = np.array([5.*math.cos(angle), 5.*math.sin(angle), 0.])
        right.update(line_start=(np.array([5., 0., 0.])-vector).tolist(),
                     line_end=(np.array([5., 0., 0.])+vector).tolist())
    assert helper._duplicate(left, right) is expected


@pytest.mark.parametrize("reverse", (False, True))
def test_shorter_span_duplicate_is_not_the_scorers_both_span_match(fusion, reverse):
    helper, metric = fusion
    long, short = _line("long", end=100.), _line("short", end=10.)
    first, second = (short, long) if reverse else (long, short)
    values = metric.pair_metrics(helper._line(first), helper._line(second))
    assert values["shorter_span_overlap"] == 1.
    assert sorted((values["gold_span_overlap"], values["prediction_span_overlap"])) == [.1, 1.]
    assert helper._duplicate(first, second) is True
    assert metric.is_match(values, metric.Thresholds()) is False
    # This compares the exact pure gate, not a scoring run or a changed threshold.


@pytest.mark.parametrize("middle_first", (False, True))
def test_greedy_union_preserves_order_sensitive_nontransitive_chain(fusion, middle_first):
    rows = [_line("a", y=0.), _line("b", y=3.), _line("c", y=6.)]
    if middle_first:
        rows = [rows[1], rows[0], rows[2]]
    first, second = _one(fusion, [], rows), _one(fusion, [], copy.deepcopy(rows))
    assert first == second
    assert _ids(first["variants"]["cad_only_gated"]) == _ids(rows)
    assert _ids(first["variants"]["union"]) == (["CADF_b"] if middle_first else ["CADF_a", "CADF_c"])


def test_intersection_is_corroboration_not_one_to_one_assignment_and_a0_is_never_deduplicated(fusion):
    row = _one(fusion, [_line("a", 0.), _line("b", 6.), _line("a_again", 0.)], [_line("middle", 3.)])
    assert row["status"] == "COMPLETE"
    assert _ids(row["variants"]["intersection"]) == ["a", "b", "a_again"]
    assert _ids(row["variants"]["union"]) == ["a", "b", "a_again"]
    assert row["cached_line_counts"] == dict(scan_only_a0=3, cad_only_gated=1, intersection=3, union=3)


@pytest.mark.parametrize("metadata", (
    {"counted": False}, {"semantic_label": "not_a_bend"}, {"observability": "UNOBSERVED"},
    {"axis_direction": [0., 1., 0.], "span_mm": -50.},
    {"ownership": {"status": "NONE"}}, {"unused_nonjson_object": object()},
))
def test_a0_auxiliary_semantics_never_filter_or_rewrite_selected_line(fusion, metadata):
    source = {**_line(" kept verbatim "), **metadata}
    row = _one(fusion, [source], [])
    assert row["status"] == "COMPLETE"
    expected = {**_line(" kept verbatim "), "fusion_source": "scan_only_a0"}
    assert row["variants"]["scan_only_a0"] == row["variants"]["union"] == [expected]
    assert set(row["variants"]["scan_only_a0"][0]) == {"bend_id", "line_start", "line_end", "fusion_source"}


@pytest.mark.parametrize("span", ("absent", None, [], [[20., 1., 0.], [30., 1., 0.]]))
def test_legacy_span_alias_precedence_without_reconciling_direct_endpoints(fusion, span):
    source = _line("alias")
    if span != "absent":
        source["span_endpoints"] = span
    if isinstance(span, list) and span:
        source["line_start"], source["line_end"] = "ignored direct start", "ignored direct end"
    row = _one(fusion, [copy.deepcopy(source)], [copy.deepcopy(source)])
    assert row["status"] == "COMPLETE"
    expected = span if isinstance(span, list) and span else [[0., 0., 0.], [10., 0., 0.]]
    assert row["variants"]["scan_only_a0"][0]["line_start"] == expected[0]
    assert row["variants"]["scan_only_a0"][0]["line_end"] == expected[1]
    assert row["variants"]["cad_only_gated"] == [source]


def test_duplicate_ids_cad_prefix_collisions_and_existing_fusion_metadata_are_preserved(fusion):
    a0 = [_line("same", 0.), _line("same", 20.), _line("CADF_x", 40.)]
    cad = [{**_line("x", 60.), "fusion_source": "original_metadata"}, _line("x", 80.)]
    row = _one(fusion, a0, cad)
    assert row["status"] == "COMPLETE"
    assert _ids(row["variants"]["union"]) == ["same", "same", "CADF_x", "CADF_x", "CADF_x"]
    assert row["variants"]["cad_only_gated"][0]["fusion_source"] == "original_metadata"
    assert row["variants"]["union"][3]["fusion_source"] == "cad_guided_gated_rescue"


def test_maximum_admitted_id_is_not_truncated_after_original_cad_prefixing(fusion):
    identifier = "x"*512
    row = _one(fusion, [], [_line(identifier)])
    assert row["status"] == "COMPLETE"
    assert row["variants"]["union"][0]["bend_id"] == "CADF_"+identifier
    assert len(row["variants"]["union"][0]["bend_id"]) == 517
    # Original identity behavior; later scorer eligibility is a separate boundary.


def test_variants_and_nested_metadata_are_detached_from_each_other_and_inputs(fusion):
    helper, _ = fusion
    a0, cad = _payloads([_line("a")], [_line("match"), {**_line("rescue", 20.), "meta": {"items": [1, 2]}}])
    before = copy.deepcopy((a0, cad))
    row = helper._case(a0, cad, CASE_IDS[0])
    assert (a0, cad) == before
    variants = row["variants"]
    variants["scan_only_a0"][0]["line_start"][0] = 99.
    assert variants["intersection"][0]["line_start"][0] == variants["union"][0]["line_start"][0] == 0.
    variants["intersection"][0]["line_end"][0] = 88.
    assert variants["union"][0]["line_end"][0] == 10.
    variants["union"][1]["meta"]["items"].append(3)
    assert variants["cad_only_gated"][1]["meta"]["items"] == [1, 2]
    assert (a0, cad) == before


def test_caller_mutation_after_case_snapshot_cannot_change_fusion(fusion, monkeypatch):
    helper, _ = fusion
    a0, cad = _payloads([_line("a")], [{**_line("c", 20.), "meta": {"items": [1]}}])
    original = helper._original_variants
    def mutate_then_reconstruct(owned_a, owned_c, duplicate):
        a0["owned_bend_regions"][0]["line_end"][0] = float("nan")
        cad["predictions"][0]["line_start"][1] = 999.
        cad["predictions"][0]["meta"]["items"].append(2)
        return original(owned_a, owned_c, duplicate)
    monkeypatch.setattr(helper, "_original_variants", mutate_then_reconstruct)
    row = helper._case(a0, cad, CASE_IDS[0])
    assert row["status"] == "COMPLETE"
    assert row["variants"]["scan_only_a0"][0]["line_end"][0] == 10.
    assert row["variants"]["cad_only_gated"][0]["line_start"][1] == 20.
    assert row["variants"]["union"][1]["meta"]["items"] == [1]


def test_owned_snapshot_is_revalidated_before_any_duplicate_arithmetic(fusion, monkeypatch):
    helper, _ = fusion
    original = helper._copy_json
    changed = []
    def corrupt_owned(value, budget, depth=0):
        result = original(value, budget, depth)
        if depth == 0 and type(result) is dict and result.get("bend_id") == "late":
            result["line_end"][0] = float("nan")
            changed.append(True)
        return result
    monkeypatch.setattr(helper, "_copy_json", corrupt_owned)
    row = _one(fusion, [_line("late")], [_line("cad")])
    _cleared(row, "MALFORMED_RESULT", "INVALID_LINE_COORDINATE")
    assert changed == [True] and row["work"]["pair_checks_attempted"] == 0


def test_public_case_order_and_mapping_bindings_are_snapshotted_not_re_read(fusion, monkeypatch):
    helper, _ = fusion
    a0, cad = _cohort()
    order = list(CASE_IDS)
    original = helper._case
    def replace_later_mapping(first, second, case):
        if case == CASE_IDS[0]:
            cad[CASE_IDS[-1]] = None
            a0.clear()
            order[:] = ["mutated"]
        return original(first, second, case)
    monkeypatch.setattr(helper, "_case", replace_later_mapping)
    result = helper.reconstruct_gold11_retained_fusion_v1(
        a0_results_by_case=a0, cad_results_by_case=cad, case_ids=order)
    assert result["case_order"] == list(CASE_IDS) and list(result["cases"]) == list(CASE_IDS)
    assert result["all_cases_usable"] is True


@pytest.mark.parametrize("fault", ("a0_schema_bool", "cad_schema_two", "a0_case", "cad_case",
    "a0_kind", "a0_algorithm", "a0_frame", "cad_kind", "cad_route",
    "research_false", "zero_shot_false", "source_only_true", "status_present", "rows_tuple"))
def test_payload_schema_and_identity_conflicts_refuse_whole_case(fusion, fault):
    helper, _ = fusion
    a0, cad = _payloads([_line("a")], [_line("c")])
    if fault == "a0_schema_bool": a0["schema_version"] = True
    elif fault == "cad_schema_two": cad["schema_version"] = 2
    elif fault == "a0_case": a0["case_id"] = "wrong"
    elif fault == "cad_case": cad["case_id"] = "wrong"
    elif fault == "a0_kind": a0["kind"] = "wrong"
    elif fault == "a0_algorithm": a0["algorithm"] = "wrong"
    elif fault == "a0_frame": a0["coordinate_frame"] = "cad"
    elif fault == "cad_kind": cad["kind"] = "wrong"
    elif fault == "cad_route": cad["route"] = "wrong"
    elif fault == "research_false": cad["research_only"] = False
    elif fault == "zero_shot_false": cad["cad_guided_zero_shot"] = False
    elif fault == "source_only_true": cad["source_only"] = True
    elif fault == "status_present": cad["status"] = "COMPLETE"
    else: a0["owned_bend_regions"] = tuple(a0["owned_bend_regions"])
    row = helper._case(a0, cad, CASE_IDS[0])
    _cleared(row, "MALFORMED_RESULT")
    assert row["work"]["pair_checks_attempted"] == 0


@pytest.mark.parametrize("lane", ("a0", "cad"))
@pytest.mark.parametrize("fault", ("id_empty", "id_numeric", "id_long", "span_tuple",
    "span_bad_pair", "point_tuple", "bool_coordinate", "nan", "coordinate_overflow",
    "degenerate", "unit", "frame"))
def test_invalid_late_line_clears_every_variant_before_comparison(fusion, lane, fault):
    bad = _line("late", 40.)
    if fault == "id_empty": bad["bend_id"] = "  "
    elif fault == "id_numeric": bad["bend_id"] = 7
    elif fault == "id_long": bad["bend_id"] = "x"*513
    elif fault == "span_tuple": bad["span_endpoints"] = ([0., 0., 0.], [10., 0., 0.])
    elif fault == "span_bad_pair": bad["span_endpoints"] = [[0., 0., 0.]]
    elif fault == "point_tuple": bad["line_start"] = tuple(bad["line_start"])
    elif fault == "bool_coordinate": bad["line_end"][0] = True
    elif fault == "nan": bad["line_end"][0] = float("nan")
    elif fault == "coordinate_overflow": bad["line_end"][0] = 10**400
    elif fault == "degenerate": bad["line_end"] = list(bad["line_start"])
    elif fault == "unit": bad["unit"] = "inch"
    else: bad["reference_frame"] = "CAD_ALIGNED"
    a0, cad = [_line("good_a")], [_line("good_c", 20.)]
    (a0 if lane == "a0" else cad).append(bad)
    row = _one(fusion, a0, cad)
    _cleared(row, "MALFORMED_RESULT")
    assert row["work"]["pair_checks_attempted"] == 0
    assert row["work"]["a0_input_rows"] == len(a0) and row["work"]["cad_input_rows"] == len(cad)


@pytest.mark.parametrize("metadata", (float("nan"), np.float64(1.), object(), {1: "nonstring"},
                                      1 << 4096))
def test_invalid_cad_auxiliary_json_is_whole_case_failure_not_row_filter(fusion, metadata):
    row = _one(fusion, [_line("a")], [_line("good", 20.), {**_line("bad", 40.), "meta": metadata}])
    _cleared(row, "MALFORMED_RESULT")
    assert row["work"]["pair_checks_attempted"] == 0


def test_json_depth_and_single_string_boundary_are_explicit_without_copy_hooks(fusion):
    helper, _ = fusion
    value = None
    for _ in range(16):
        value = [value]
    assert helper._copy_json(value, helper._snapshot_budget()) == value
    with pytest.raises(helper._Budget, match="JSON_NODE_OR_DEPTH_BUDGET"):
        helper._copy_json([value], helper._snapshot_budget())
    assert helper._copy_json("x"*8192, helper._snapshot_budget()) == "x"*8192
    with pytest.raises(helper._Budget, match="JSON_STRING_BUDGET"):
        helper._copy_json("x"*8193, helper._snapshot_budget())


@pytest.mark.parametrize("kind", ("nodes", "strings", "depth", "cycle"))
def test_cad_json_resource_refusal_clears_all_variants(fusion, kind):
    helper, _ = fusion
    if kind == "nodes":
        metadata = [None]*helper.MAX_JSON_NODES
    elif kind == "strings":
        metadata = ["x"*8192]*512
    elif kind == "depth":
        metadata = None
        for _ in range(17): metadata = [metadata]
    else:
        metadata = []
        metadata.append(metadata)
    row = _one(fusion, [_line("a")], [{**_line("c", 20.), "meta": metadata}])
    _cleared(row, "RESOURCE_REFUSAL")
    assert row["work"]["pair_checks_attempted"] == 0


def test_a0_ignored_nonjson_auxiliary_is_not_accidentally_copied_like_cad_metadata(fusion):
    auxiliary = []
    auxiliary.append(auxiliary)
    row = _one(fusion, [{**_line("a"), "ignored": auxiliary}], [])
    assert row["status"] == "COMPLETE" and row["cached_line_counts"]["scan_only_a0"] == 1


@pytest.mark.parametrize("lane", ("a0", "cad"))
def test_input_row_cap_is_preflighted_before_any_row_projection(fusion, monkeypatch, lane):
    helper, _ = fusion
    a0, cad = _payloads()
    (a0["owned_bend_regions"] if lane == "a0" else cad["predictions"]).extend([None]*4097)
    def forbidden(_):
        raise AssertionError("ROW_PROJECTION_MUST_NOT_START_AFTER_CAP_REFUSAL")
    monkeypatch.setattr(helper, "_line_projection", forbidden)
    row = helper._case(a0, cad, CASE_IDS[0])
    _cleared(row, "RESOURCE_REFUSAL", "INPUT_ROW_BUDGET")
    assert row["work"]["pair_checks_preflight_upper_bound"] is None
    assert row["work"]["pair_checks_attempted"] == 0


def test_maximum_a0_only_rows_are_preserved_without_geometry_comparisons(fusion):
    row = _one(fusion, [_line("same_legacy_id")]*4096, [])
    assert row["status"] == "COMPLETE"
    assert row["cached_line_counts"] == dict(scan_only_a0=4096, cad_only_gated=0, intersection=0, union=4096)
    assert row["work"]["pair_checks_attempted"] == row["work"]["pair_checks_preflight_upper_bound"] == 0


@pytest.mark.parametrize("extra_a0", (0, 1))
def test_two_million_pair_preflight_boundary_with_explicit_stop_before_geometry(fusion, monkeypatch, extra_a0):
    helper, _ = fusion
    # 2*1444*625 + 625*624/2 = 2,000,000. One more A0 adds1250.
    # Only list lengths are real. This intentional seam stops the admitted case
    # at its FIRST projection; there are zero duplicate/metric calls.
    a0, cad = _payloads([None]*(1444+extra_a0), [None]*625)
    calls, work = [], {}
    class AdmittedPreflight(Exception): pass
    def stop(_):
        calls.append(True)
        raise AdmittedPreflight()
    monkeypatch.setattr(helper, "_line_projection", stop)
    if extra_a0:
        with pytest.raises(helper._Budget, match="PAIR_CHECK_PREFLIGHT_BUDGET"):
            helper._prepare(a0, cad, CASE_IDS[0], work)
        assert calls == []
    else:
        with pytest.raises(AdmittedPreflight):
            helper._prepare(a0, cad, CASE_IDS[0], work)
        assert calls == [True]
    assert work["pair_checks_preflight_upper_bound"] == 2_000_000+1250*extra_a0


def test_pair_counter_equals_bound_without_duplicates_and_short_circuits_when_possible(fusion):
    full = _one(fusion, [_line("a", 0.), _line("b", 20.)], [_line("c", 40.), _line("d", 60.), _line("e", 80.)])
    assert full["work"]["pair_checks_preflight_upper_bound"] == 15
    assert full["work"]["pair_checks_attempted"] == 15
    short = _one(fusion, [_line("a")], [_line("c"), _line("d")])
    assert short["work"]["pair_checks_preflight_upper_bound"] == 5
    assert short["work"]["pair_checks_attempted"] == 3


@pytest.mark.parametrize("failed_call", (1, 3, 5))
def test_failed_comparison_is_counted_and_never_publishes_earlier_variants(fusion, monkeypatch, failed_call):
    helper, _ = fusion
    attempted = []
    def injected(left, right):
        attempted.append((left["bend_id"], right["bend_id"]))
        if len(attempted) == failed_call:
            raise ArithmeticError("INJECTED_COMPARISON_FAILURE")
        return False
    monkeypatch.setattr(helper, "_duplicate", injected)
    row = _one(fusion, [_line("a")], [_line("c", 20.), _line("d", 40.)])
    _cleared(row, "RECONSTRUCTION_FAILURE", "UNEXPECTED_RECONSTRUCTION_ERROR")
    assert row["work"]["pair_checks_attempted"] == len(attempted) == failed_call


@pytest.mark.parametrize("extra_callback", (False, True))
def test_pair_attempt_counter_cap_with_explicit_primed_mock_not_millions_of_calls(fusion, monkeypatch, extra_callback):
    helper, _ = fusion
    comparisons = []
    def primed_prepare(a0, cad, case, work):
        work.update(a0_input_rows=1444, cad_input_rows=625,
                    pair_checks_preflight_upper_bound=2_000_000, pair_checks_attempted=1_999_999)
        return {"owned_bend_regions": []}, {"predictions": []}
    def one_or_two_callbacks(a0, cad, callback):
        callback(_line("a"), _line("b"))
        if extra_callback:
            callback(_line("a"), _line("b"))
        return {name: [] for name in VARIANTS}
    monkeypatch.setattr(helper, "_prepare", primed_prepare)
    monkeypatch.setattr(helper, "_original_variants", one_or_two_callbacks)
    monkeypatch.setattr(helper, "_duplicate", lambda left, right: comparisons.append(True) or False)
    row = _one(fusion)
    assert comparisons == [True]  # Exactly ONE mock comparison, not2M metric calls.
    assert row["work"]["pair_checks_attempted"] == 2_000_000+int(extra_callback)
    if extra_callback:
        _cleared(row, "RESOURCE_REFUSAL", "PAIR_CHECK_ACCOUNTING_BOUND_EXCEEDED")
    else:
        assert row["status"] == "COMPLETE"


def test_output_snapshot_budget_failure_clears_already_reconstructed_variants(fusion, monkeypatch):
    helper, _ = fusion
    original = helper._snapshot_budget
    monkeypatch.setattr(helper, "_snapshot_budget",
        lambda multiplier=1: original() if multiplier == 1 else {"nodes": 0, "chars": 0})
    row = _one(fusion, [_line("a")], [_line("c", 20.)])
    _cleared(row, "RESOURCE_REFUSAL", "JSON_NODE_OR_DEPTH_BUDGET")
    assert row["work"]["pair_checks_attempted"] == 2


def test_all_eleven_case_failures_are_retained_without_changing_other_cases(fusion):
    helper, _ = fusion
    a0, cad = _cohort()
    for index, case in enumerate(CASE_IDS):
        if index % 2:
            cad[case]["predictions"] = [_line("valid"), _line("")]
        else:
            a0[case]["owned_bend_regions"] = [_line("present", y=float(index)*20.)]
    result = helper.reconstruct_gold11_retained_fusion_v1(
        a0_results_by_case=a0, cad_results_by_case=cad, case_ids=CASE_IDS)
    assert result["status"] == "COMPLETE_WITH_CASE_REFUSALS" and result["failed_case_count"] == 5
    assert list(result["cases"]) == list(CASE_IDS) and result["case_count"] == 11
    for index, case in enumerate(CASE_IDS):
        if index % 2:
            _cleared(result["cases"][case], "MALFORMED_RESULT")
        else:
            assert result["cases"][case]["cached_line_counts"]["union"] == 1


@pytest.mark.parametrize("fault", ("ids_type", "ten", "twelve", "duplicate", "blank", "numeric", "long",
                                  "a0_missing", "cad_extra", "a0_type", "cad_type"))
def test_bad_cohort_declarations_raise_instead_of_running_a_partial_cohort(fusion, monkeypatch, fault):
    helper, _ = fusion
    a0, cad = _cohort()
    ids = list(CASE_IDS)
    if fault == "ids_type": ids = set(ids)
    elif fault == "ten": ids.pop()
    elif fault == "twelve": ids.append("extra")
    elif fault == "duplicate": ids[-1] = ids[0]
    elif fault == "blank": ids[-1] = " "
    elif fault == "numeric": ids[-1] = 7
    elif fault == "long": ids[-1] = "x"*513
    elif fault == "a0_missing": a0.pop(CASE_IDS[-1])
    elif fault == "cad_extra": cad["extra"] = cad[CASE_IDS[-1]]
    elif fault == "a0_type": a0 = list(a0.items())
    else: cad = None
    calls = []
    monkeypatch.setattr(helper, "_case", lambda *args: calls.append(True))
    with pytest.raises(ValueError):
        helper.reconstruct_gold11_retained_fusion_v1(
            a0_results_by_case=a0, cad_results_by_case=cad, case_ids=ids)
    assert calls == []


def test_result_explicitly_disclaims_scoring_authentication_observed_counts_and_release(fusion):
    helper, _ = fusion
    a0, cad = _cohort()
    result = helper.reconstruct_gold11_retained_fusion_v1(
        a0_results_by_case=a0, cad_results_by_case=cad, case_ids=CASE_IDS)
    assert result["inference_execution"] == result["scoring"] == "NOT_PERFORMED"
    assert result["historical_execution_attestation"] == "NOT_ESTABLISHED_BY_RECONSTRUCTION"
    assert result["historical_score_comparability"] == "UNESTABLISHED"
    assert result["cohort_authentication"] == "CALLER_FROZEN_CASE_AND_PAYLOAD_BYTES_REQUIRED"
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is result["floor_release_allowed"] is False
    assert result["duplicate_rule"] == dict(axis_angle_deg_max=5., transverse_offset_mm_max=4.,
        shorter_span_overlap_min=.6, order="ALL_A0_FIRST_THEN_INPUT_ORDER_CAD_GREEDY_APPEND", cad_added_id_prefix="CADF_")
    assert result["pair_check_cap_per_case"] == 2_000_000 and result["input_row_cap_per_case_per_source"] == 4096
    assert RESOURCE_LIMITS == dict(wall_seconds=60, cpu_seconds=45,
        process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
