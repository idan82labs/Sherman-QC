"""Independent cached-fusion scoring controls using invented 11/6/71 records.

No historical predictions, reference payloads, raw geometry or original CLI is
opened. At an authorized test run, a private package loads the inspected pure
helpers. ONLY the exact legacy matcher definitions are AST-extracted; SciPy's
real assignment is used for ordinary small invented cases, never inference.
The scorer-cap test explicitly mocks reconstruction AND matching admission to
avoid thousands of pair calculations; it makes no geometry qualification claim.
Snapshots are ownership controls, not atomic authentication of a caller's bundle.
Prospective gate:60s wall/45s CPU/1GiB tree RSS/1MiB stdout, subject to root freeze.
"""
import ast
import copy
from dataclasses import dataclass
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SOURCE = SCRIPTS / "gold11_cached_fusion_scoring_v1.py"
METRIC_SOURCE = SCRIPTS / "evaluate_f1_against_manual_gold.py"
VARIANTS = ("scan_only_a0", "cad_only_gated", "intersection", "union")
CASE_IDS = tuple(f"invented_case_{i:02d}" for i in range(11))
INSERTION_ORDER = (7, 0, 10, 2, 9, 1, 6, 3, 8, 4, 5)
RAW_FRAME = "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
RESOURCE_LIMITS = dict(wall_seconds=60, cpu_seconds=45,
    process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)


@pytest.fixture(scope="module")
def modules():
    package = "_independent_cached_fusion_scoring_v1"
    suffixes = ("evaluate_f1_against_manual_gold", "f1_physical_evaluation_contract_v2",
                "gold11_mesh_scoring_v1", "gold11_retained_prediction_scoring_v1",
                "gold11_retained_fusion_v1", "gold11_cached_fusion_scoring_v1")
    names = [package, *(package + "." + suffix for suffix in suffixes)]
    assert not any(name in sys.modules for name in names)
    parent = ModuleType(package)
    parent.__path__ = [str(SCRIPTS)]
    sys.modules[package] = parent
    loaded = {}
    try:
        metric = ModuleType(names[1])
        sys.modules[names[1]] = metric
        metric.__dict__.update(np=np, math=math, dataclass=dataclass, Any=Any,
            linear_sum_assignment=importlib.import_module("scipy.optimize").linear_sum_assignment)
        expected = ["Thresholds", "_endpoints", "pair_metrics", "is_match", "_match_cost", "match_lines"]
        definitions = [node for node in ast.parse(METRIC_SOURCE.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in expected]
        assert [node.name for node in definitions] == expected
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(METRIC_SOURCE), "exec"), metric.__dict__)
        loaded[suffixes[0]] = metric
        for suffix in suffixes[1:]:
            name = package + "." + suffix
            spec = importlib.util.spec_from_file_location(name, SCRIPTS / (suffix + ".py"))
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[suffix] = module
        yield SimpleNamespace(
            scorer=loaded[suffixes[-1]], fusion=loaded[suffixes[-2]],
            cached=loaded[suffixes[-3]], mesh=loaded[suffixes[-4]],
            contract=loaded[suffixes[1]], metric=metric)
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


def _line(identifier, y=0., start=0., end=10.):
    return dict(bend_id=identifier, line_start=[start, y, 0.], line_end=[end, y, 0.])


def _inputs(mode="perfect"):
    references, a0, cad, families = {}, {}, {}, {}
    for index in INSERTION_ORDER:
        case = CASE_IDS[index]
        count = 7 if index < 5 else 6
        families[case] = f"invented_family_{index % 6}"
        references[case] = [_line(f"reference_{j}", y=20.*j) for j in range(count)]
        rows = [] if mode == "empty" else [
            _line(f"legacy_{j}", y=20.*j + (1000. if mode == "wrong" else 0.))
            for j in range(count)]
        a0[case] = dict(schema_version=1, case_id=case,
            kind="sherman_qc_f1_typed_union_counted_components",
            algorithm="f1_typed_union_exact_birth_v1", coordinate_frame="scan_raw",
            owned_bend_regions=copy.deepcopy(rows))
        cad[case] = dict(schema_version=1, case_id=case,
            kind="sherman_qc_f1_cad_guided_scan_line_predictions",
            route="blind_cad_ply_detected_lines", research_only=True,
            cad_guided_zero_shot=True, source_only=False, predictions=copy.deepcopy(rows))
    return dict(reference_records_by_case=references, a0_results_by_case=a0,
                cad_results_by_case=cad, case_family_by_id=families)


def _score(modules, inputs):
    return modules.scorer.score_gold11_cached_fusion_v1(**inputs)


def _case(result, variant, case=CASE_IDS[0]):
    return result["variants"][variant]["case_dispositions"][case]


def _counts(summary, tp, fp, fn):
    assert (summary["true_positive"], summary["false_positive"], summary["false_negative"]) == (tp, fp, fn)
    assert summary["reference_line_count"] == tp + fn
    assert summary["cached_prediction_line_count"] == tp + fp


def test_unchanged_function_objects_and_no_fake_modern_or_cached_carrier_validation(modules, monkeypatch):
    scorer = modules.scorer
    assert scorer.match_lines is modules.metric.match_lines
    assert scorer.Thresholds is modules.metric.Thresholds
    assert scorer._cached_summary is modules.cached._cached_summary
    assert scorer._lines is modules.mesh._lines
    assert scorer.reconstruct_gold11_retained_fusion_v1 is modules.fusion.reconstruct_gold11_retained_fusion_v1
    assert modules.cached.score_localized_cases is modules.contract.score_localized_cases
    def forbidden(*args, **kwargs):
        raise AssertionError("UNRELATED_MEASURED_OR_CACHED_CARRIER_VALIDATOR_CALLED")
    for owner, names in ((modules.cached, ("_cached_predictions", "_cached_lane")),
                         (modules.mesh, ("_case_predictions", "_lane"))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    result = _score(modules, _inputs())
    assert all(lane["geometry"]["true_positive"] == 71 for lane in result["variants"].values())


def test_source_has_only_inspected_pure_imports_and_reconstruction_never_receives_references(modules, monkeypatch):
    tree = ast.parse(SOURCE.read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(isinstance(node, ast.ImportFrom) and node.level == 1 for node in imports)
    assert [node.module for node in imports] == [
        "evaluate_f1_against_manual_gold", "gold11_retained_fusion_v1",
        "gold11_retained_prediction_scoring_v1", "gold11_mesh_scoring_v1"]
    inputs, calls = _inputs(), []
    original = modules.scorer.reconstruct_gold11_retained_fusion_v1
    def checked(**kwargs):
        calls.append(kwargs)
        assert set(kwargs) == {"a0_results_by_case", "cad_results_by_case", "case_ids"}
        assert kwargs["case_ids"] == list(CASE_IDS)
        assert kwargs["a0_results_by_case"] == inputs["a0_results_by_case"]
        assert kwargs["cad_results_by_case"] == inputs["cad_results_by_case"]
        return original(**kwargs)
    monkeypatch.setattr(modules.scorer, "reconstruct_gold11_retained_fusion_v1", checked)
    _score(modules, inputs)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ("perfect", "empty", "wrong"))
def test_all_four_variants_complete_denominators_nil_conventions_and_distinct_exact_metrics(modules, mode):
    result = _score(modules, _inputs(mode))
    assert result["case_count"] == 11 and result["family_count"] == 6 and result["reference_line_count"] == 71
    assert list(result["variants"]) == list(VARIANTS)
    assert result["variant_selection"] == "NONE_ALL_FOUR_REPORTED"
    assert not {"winner", "best_variant", "selected_variant"}.intersection(result)
    for name, lane in result["variants"].items():
        assert list(lane["case_dispositions"]) == list(CASE_IDS)
        assert lane["lane_context"] == result["reconstruction"]["variant_contexts"][name]
        summary = lane["geometry"]
        assert summary["case_count"] == 11 and lane["refused_case_count"] == 0
        assert len(lane["families"]) == 6
        if mode == "perfect":
            _counts(summary, 71, 0, 0)
            assert summary["precision"] == summary["recall"] == summary["f1"] == 1.
            assert summary["exact_localized_target_set_cases"] == summary["exact_cached_line_count_cases"] == 11
        else:
            _counts(summary, 0, 0 if mode == "empty" else 71, 71)
            assert summary["precision"] == (None if mode == "empty" else 0.)
            assert summary["recall"] == 0. and summary["f1"] is None
            assert summary["exact_localized_target_set_cases"] == 0
            assert summary["exact_cached_line_count_cases"] == (0 if mode == "empty" else 11)
        for row in lane["case_dispositions"].values():
            assert row["disposition"] == "SCORED_CACHED_GEOMETRY" and row["observed_bend_count"] is None
            assert row["exact_cached_line_count"] is (mode != "empty")


@pytest.mark.parametrize("a0_is_short", (False, True))
def test_shorter_span_fusion_does_not_relax_either_side_of_the_scoring_gate(modules, a0_is_short):
    inputs, case = _inputs("empty"), CASE_IDS[0]
    inputs["reference_records_by_case"][case][0] = _line("reference_0", end=100.)
    short, long = _line("short", end=10.), _line("long", end=100.)
    inputs["a0_results_by_case"][case]["owned_bend_regions"] = [short if a0_is_short else long]
    inputs["cad_results_by_case"][case]["predictions"] = [long if a0_is_short else short]
    result = _score(modules, inputs)
    reconstructed = result["reconstruction"]["cases"][case]
    assert reconstructed["cached_line_counts"] == {variant: 1 for variant in VARIANTS}
    assert reconstructed["variants"]["union"][0]["bend_id"] == ("short" if a0_is_short else "long")
    for variant in VARIANTS:
        short_lane = (not a0_is_short) if variant == "cad_only_gated" else a0_is_short
        summary = result["variants"][variant]["geometry"]
        _counts(summary, 0 if short_lane else 1, 1 if short_lane else 0, 71 if short_lane else 70)
        if short_lane:
            assert summary["f1"] is None
        else:
            assert summary["f1"] == pytest.approx(1./36.)
    assert result["matching"]["gold_span_overlap_min"] == result["matching"]["prediction_span_overlap_min"] == .6


@pytest.mark.parametrize("mode", ("two_distinct", "same_geometry", "prefix_collision"))
def test_duplicate_display_ids_preserve_row_index_assignment_without_deduplication(modules, mode):
    inputs, case = _inputs("empty"), CASE_IDS[0]
    if mode == "prefix_collision":
        a0, cad = [_line("CADF_x")], [_line("x", 20.)]
        expected_ids, tp, fp = ["CADF_x", "CADF_x"], 2, 0
    else:
        a0, cad = [_line("same"), _line("same", 0. if mode == "same_geometry" else 20.)], []
        expected_ids, tp, fp = ["same", "same"], (1 if mode == "same_geometry" else 2), (1 if mode == "same_geometry" else 0)
    inputs["a0_results_by_case"][case]["owned_bend_regions"] = a0
    inputs["cad_results_by_case"][case]["predictions"] = cad
    result = _score(modules, inputs)
    assert [row["bend_id"] for row in result["reconstruction"]["cases"][case]["variants"]["union"]] == expected_ids
    row = _case(result, "union")
    assert (row["prediction_count"], row["true_positive"], row["false_positive"]) == (2, tp, fp)
    assert [match["prediction_id"] for match in row["matches"]] == expected_ids[:tp]
    assert row["unmatched_prediction_ids"] == (["same"] if mode == "same_geometry" else [])


@pytest.mark.parametrize("middle_first", (False, True))
def test_order_sensitive_union_is_reported_without_selecting_the_better_variant(modules, middle_first):
    inputs, case = _inputs("empty"), CASE_IDS[0]
    rows = [_line("a", 0.), _line("b", 3.), _line("c", 6.)]
    inputs["cad_results_by_case"][case]["predictions"] = [rows[1], rows[0], rows[2]] if middle_first else rows
    result = _score(modules, inputs)
    assert _case(result, "cad_only_gated")["prediction_count"] == 3
    assert _case(result, "union")["prediction_count"] == (1 if middle_first else 2)
    assert _case(result, "union")["true_positive"] == 1
    assert result["variants"]["union"]["geometry"]["f1"] == pytest.approx(2./(72 if middle_first else 73))
    assert result["variant_selection"] == "NONE_ALL_FOUR_REPORTED" and len(result["variants"]) == 4


@pytest.mark.parametrize("span", (None, [], [[0., 0., 0.], [10., 0., 0.]]))
def test_original_span_alias_precedence_survives_scoring_and_ids_are_not_renamed(modules, span):
    inputs, case = _inputs("empty"), CASE_IDS[0]
    row = _line(" legacy display id ")
    row["span_endpoints"] = span
    if span:
        row["line_start"], row["line_end"] = "ignored start", "ignored end"
    inputs["a0_results_by_case"][case]["owned_bend_regions"] = [copy.deepcopy(row)]
    inputs["cad_results_by_case"][case]["predictions"] = [copy.deepcopy(row)]
    result = _score(modules, inputs)
    for variant in VARIANTS:
        assert _case(result, variant)["true_positive"] == 1
        assert _case(result, variant)["matches"][0]["prediction_id"] == " legacy display id "
    assert result["reconstruction"]["cases"][case]["variants"]["cad_only_gated"] == [row]


def test_family_recall_retains_singleton_family_with_no_predictions(modules):
    inputs, case = _inputs(), CASE_IDS[5]
    inputs["a0_results_by_case"][case]["owned_bend_regions"] = []
    inputs["cad_results_by_case"][case]["predictions"] = []
    result = _score(modules, inputs)
    for lane in result["variants"].values():
        _counts(lane["geometry"], 65, 0, 6)
        assert lane["geometry"]["f1"] == pytest.approx(65./68.)
        assert lane["geometry"]["exact_localized_target_set_accuracy"] == pytest.approx(10./11.)
        assert lane["minimum_family_recall"] == 0.
        assert lane["families"]["invented_family_5"]["case_count"] == 1
        _counts(lane["families"]["invented_family_5"], 0, 0, 6)
        assert lane["families"]["invented_family_5"]["precision"] is None
        for family in range(5):
            _counts(lane["families"][f"invented_family_{family}"], 13, 0, 0)


@pytest.mark.parametrize("fault", ("a0_late", "cad_late", "a0_none", "cad_status", "a0_row_budget", "cad_schema"))
def test_reconstruction_refusal_zeroes_all_four_case_variants_and_retains_every_reference_fn(modules, fault):
    inputs, case = _inputs(), CASE_IDS[0]
    a0, cad = inputs["a0_results_by_case"], inputs["cad_results_by_case"]
    if fault == "a0_late": a0[case]["owned_bend_regions"].append(_line(""))
    elif fault == "cad_late": cad[case]["predictions"].append(_line(""))
    elif fault == "a0_none": a0[case] = None
    elif fault == "cad_status": cad[case]["status"] = "COMPLETE"
    elif fault == "a0_row_budget": a0[case]["owned_bend_regions"] = [None]*4097
    else: cad[case]["schema_version"] = True
    result = _score(modules, inputs)
    reconstruction = result["reconstruction"]["cases"][case]
    assert reconstruction["status"] != "COMPLETE"
    for variant, lane in result["variants"].items():
        row = _case(result, variant)
        assert row["disposition"] == "RECONSTRUCTION_REFUSED" and row["reason"] == reconstruction["reason"]
        assert row["prediction_count"] == row["true_positive"] == row["false_positive"] == 0
        assert row["false_negative"] == row["gold_count"] == 7
        assert row["precision"] is row["f1"] is None and row["recall"] == 0.
        assert lane["refused_case_count"] == 1 and len(lane["case_dispositions"]) == 11
        _counts(lane["geometry"], 64, 0, 7)
        assert lane["families"]["invented_family_0"]["recall"] == pytest.approx(6./13.)
        assert lane["minimum_family_recall"] == pytest.approx(6./13.)


@pytest.mark.parametrize("lane_key", ("a0_results_by_case", "cad_results_by_case"))
@pytest.mark.parametrize("declaration", ("absent", "correct", "wrong", None))
def test_embedded_family_identity_in_either_envelope_controls_all_lanes_not_reconstruction(modules, lane_key, declaration):
    inputs, case = _inputs(), CASE_IDS[0]
    if declaration != "absent":
        inputs[lane_key][case]["family_id"] = inputs["case_family_by_id"][case] if declaration == "correct" else declaration
    result = _score(modules, inputs)
    assert result["reconstruction"]["cases"][case]["status"] == "COMPLETE"
    assert all(result["reconstruction"]["cases"][case]["cached_line_counts"][variant] == 7 for variant in VARIANTS)
    refused = declaration not in ("absent", "correct")
    for variant, lane in result["variants"].items():
        row = _case(result, variant)
        assert row["disposition"] == ("SCORING_INPUT_REFUSED" if refused else "SCORED_CACHED_GEOMETRY")
        assert row["reason"] == ("EMBEDDED_FAMILY_IDENTITY_CONFLICT" if refused else None)
        assert row["false_negative"] == (7 if refused else 0)
        assert lane["refused_case_count"] == int(refused)
        _counts(lane["geometry"], 64 if refused else 71, 0, 7 if refused else 0)


@pytest.mark.parametrize("lane_key", ("a0_results_by_case", "cad_results_by_case"))
@pytest.mark.parametrize("kind", ("integer", "bool", "numpy_string", "array_one", "array_many", "equality_spoof", "equality_raises"))
def test_nonstring_embedded_family_is_refused_without_array_truth_or_custom_equality_hooks(modules, lane_key, kind):
    inputs, case, calls = _inputs(), CASE_IDS[0], []
    family = inputs["case_family_by_id"][case]
    class CustomEquality:
        def __eq__(self, other):
            calls.append(other)
            if kind == "equality_raises":
                raise AssertionError("CUSTOM_EQUALITY_MUST_NOT_RUN")
            return True
        def __ne__(self, other):
            return not self.__eq__(other)
    if kind == "integer": value = 0
    elif kind == "bool": value = True
    elif kind == "numpy_string": value = np.str_(family)
    elif kind == "array_one": value = np.array([family])
    elif kind == "array_many": value = np.array([family, family])
    else: value = CustomEquality()
    inputs[lane_key][case]["family_id"] = value
    result = _score(modules, inputs)
    assert calls == []
    assert result["reconstruction"]["cases"][case]["status"] == "COMPLETE"
    for variant, lane in result["variants"].items():
        row = _case(result, variant)
        assert row["disposition"] == "SCORING_INPUT_REFUSED"
        assert row["reason"] == "EMBEDDED_FAMILY_IDENTITY_CONFLICT"
        assert row["prediction_count"] == 0 and row["false_negative"] == 7
        _counts(lane["geometry"], 64, 0, 7)


@pytest.mark.parametrize("target", ("reference", "a0", "cad"))
@pytest.mark.parametrize("declaration", ("absent", "correct", "wrong_unit", "wrong_frame"))
def test_optional_units_and_frame_are_provisional_but_explicit_conflicts_are_never_scored(modules, target, declaration):
    inputs, case = _inputs(), CASE_IDS[0]
    row = (inputs["reference_records_by_case"][case][0] if target == "reference" else
           inputs["a0_results_by_case"][case]["owned_bend_regions"][0] if target == "a0" else
           inputs["cad_results_by_case"][case]["predictions"][0])
    if declaration != "absent":
        row.update(unit="inch" if declaration == "wrong_unit" else "mm",
                   reference_frame="CAD_ALIGNED" if declaration == "wrong_frame" else RAW_FRAME)
    conflict = declaration.startswith("wrong")
    if target == "reference" and conflict:
        with pytest.raises(ValueError, match="EXPLICIT_LINE_(UNIT|FRAME)_CONFLICT"):
            _score(modules, inputs)
    else:
        result = _score(modules, inputs)
        for variant in VARIANTS:
            assert _case(result, variant)["disposition"] == ("RECONSTRUCTION_REFUSED" if conflict else "SCORED_CACHED_GEOMETRY")
        assert result["reference_frame_compatibility"] == "PROVISIONAL_ASSUMPTION_NOT_VERIFIED"
        assert result["units_status"] == "PROVISIONAL_COMMON_MM_NOT_CALIBRATED"


def test_auxiliary_semantics_and_claims_neither_filter_lines_nor_grant_authority(modules):
    inputs = _inputs()
    for payload in inputs["a0_results_by_case"].values():
        for row in payload["owned_bend_regions"]:
            row.update(counted=False, semantic_label="not_a_bend", observed_bend_count=0)
    for payload in inputs["cad_results_by_case"].values():
        for row in payload["predictions"]:
            row.update(line_convention="UNAUTHENTICATED_MEASURED_CARRIER_CLAIM",
                       conformance="PASS", production_use_allowed=True)
    result = _score(modules, inputs)
    assert all(lane["geometry"]["true_positive"] == 71 for lane in result["variants"].values())
    assert result["production_use_allowed"] is False and result["conformance"] == "INDETERMINATE"
    assert result["observed_bend_count"] is None
    for lane in result["variants"].values():
        assert lane["observed_bend_count"] is None
        assert lane["geometry"]["count_metric_semantics"] == "CACHED_LINE_CARDINALITY_NOT_PHYSICAL_OBSERVED_BENDS"


def test_inputs_are_unmodified_repeated_results_deterministic_and_returned_reconstruction_owned(modules):
    inputs = _inputs()
    inputs["cad_results_by_case"][CASE_IDS[0]]["predictions"][0]["metadata"] = {"items": [1, 2]}
    before = copy.deepcopy(inputs)
    result, repeated = _score(modules, inputs), _score(modules, inputs)
    assert inputs == before and result == repeated
    reconstructed = result["reconstruction"]["cases"][CASE_IDS[0]]["variants"]
    reconstructed["cad_only_gated"][0]["metadata"]["items"].append(3)
    reconstructed["scan_only_a0"][0]["line_start"][0] = 999.
    assert inputs == before
    assert repeated["reconstruction"]["cases"][CASE_IDS[0]]["variants"]["scan_only_a0"][0]["line_start"][0] == 0.


def test_mapping_family_and_reference_snapshots_survive_post_validation_external_replacement(modules, monkeypatch):
    inputs = _inputs()
    original = modules.scorer.reconstruct_gold11_retained_fusion_v1
    def mutate_external_then_reconstruct(**kwargs):
        inputs["reference_records_by_case"][CASE_IDS[0]][0]["line_end"][0] = float("nan")
        inputs["reference_records_by_case"].clear()
        inputs["case_family_by_id"].clear()
        inputs["a0_results_by_case"].clear()
        inputs["cad_results_by_case"][CASE_IDS[-1]] = None
        return original(**kwargs)
    monkeypatch.setattr(modules.scorer, "reconstruct_gold11_retained_fusion_v1", mutate_external_then_reconstruct)
    result = _score(modules, inputs)
    assert all(lane["geometry"]["true_positive"] == 71 for lane in result["variants"].values())
    assert list(result["reconstruction"]["cases"]) == list(CASE_IDS)
    # Does not mutate a future case's nested payload before its own snapshot;
    # that stronger concurrent-authentication guarantee is explicitly absent.


@pytest.mark.parametrize("fault", ("container", "empty", "duplicate_id", "blank_id", "bool_coordinate",
    "nan", "huge_integer", "degenerate", "span_endpoints", "anchor", "axis_direction", "span_mm",
    "total_70", "total_72"))
def test_invalid_reference_geometry_or_denominator_aborts_before_any_reconstruction(modules, monkeypatch, fault):
    inputs, case = _inputs(), CASE_IDS[0]
    rows = inputs["reference_records_by_case"][case]
    if fault == "container": inputs["reference_records_by_case"][case] = tuple(rows)
    elif fault == "empty": rows.clear()
    elif fault == "duplicate_id": rows[-1]["bend_id"] = rows[0]["bend_id"]
    elif fault == "blank_id": rows[-1]["bend_id"] = " "
    elif fault == "bool_coordinate": rows[-1]["line_end"][0] = True
    elif fault == "nan": rows[-1]["line_end"][0] = float("nan")
    elif fault == "huge_integer": rows[-1]["line_end"][0] = 10**400
    elif fault == "degenerate": rows[-1]["line_end"] = list(rows[-1]["line_start"])
    elif fault == "total_70": rows.pop()
    elif fault == "total_72": rows.append(_line("extra_reference", y=999.))
    else: rows[-1][fault] = None
    calls = []
    monkeypatch.setattr(modules.scorer, "reconstruct_gold11_retained_fusion_v1", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError):
        _score(modules, inputs)
    assert calls == []


@pytest.mark.parametrize("fault", ("family_type", "ten_cases", "twelve_cases", "blank_case", "blank_family",
    "five_families", "seven_families", "reference_missing", "reference_extra", "a0_missing", "a0_extra",
    "cad_missing", "cad_extra", "a0_type", "cad_type", "reference_type"))
def test_invalid_complete_cohort_aborts_without_partial_reconstruction(modules, monkeypatch, fault):
    inputs, case = _inputs(), CASE_IDS[0]
    families = inputs["case_family_by_id"]
    if fault == "family_type": inputs["case_family_by_id"] = list(families.items())
    elif fault == "ten_cases": families.pop(case)
    elif fault == "twelve_cases": families["extra_case"] = "invented_family_0"
    elif fault == "blank_case": families[" "] = families.pop(case)
    elif fault == "blank_family": families[case] = " "
    elif fault == "five_families": families[CASE_IDS[5]] = "invented_family_0"
    elif fault == "seven_families": families[case] = "seventh_family"
    else:
        prefix, change = fault.split("_")
        key = {"reference": "reference_records_by_case", "a0": "a0_results_by_case", "cad": "cad_results_by_case"}[prefix]
        if change == "missing": inputs[key].pop(case)
        elif change == "extra": inputs[key]["extra_case"] = inputs[key][case]
        else: inputs[key] = list(inputs[key].items())
    calls = []
    monkeypatch.setattr(modules.scorer, "reconstruct_gold11_retained_fusion_v1", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError):
        _score(modules, inputs)
    assert calls == []


@pytest.mark.parametrize("failed_call", (1, 12, 44))
def test_unexpected_matcher_failure_propagates_instead_of_becoming_an_empty_case_or_report(modules, monkeypatch, failed_call):
    original, calls = modules.scorer.match_lines, []
    class InjectedMatcherFailure(RuntimeError): pass
    def fail_late(references, predictions, thresholds):
        calls.append(len(predictions))
        if len(calls) == failed_call:
            raise InjectedMatcherFailure("INJECTED_MATCHER_FAILURE")
        return original(references, predictions, thresholds)
    monkeypatch.setattr(modules.scorer, "match_lines", fail_late)
    with pytest.raises(InjectedMatcherFailure, match="INJECTED_MATCHER_FAILURE"):
        _score(modules, _inputs())
    assert len(calls) == failed_call


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("row_count", (4096, 4097))
def test_scorer_cap_admission_with_explicit_mocked_reconstruction_and_matching_boundaries(modules, monkeypatch, variant, row_count):
    inputs, case = _inputs("empty"), CASE_IDS[0]
    real_reconstruct = modules.scorer.reconstruct_gold11_retained_fusion_v1
    calls = []
    def inject_completed_variant(**kwargs):
        reconstructed = real_reconstruct(**kwargs)
        selected = reconstructed["cases"][case]
        selected["variants"][variant] = [_line("same_legacy_id", y=10000.) for _ in range(row_count)]
        selected["cached_line_counts"][variant] = row_count
        return reconstructed
    def count_only_match_admission(references, predictions, thresholds):
        # Explicit CAP-only stub: not an executed geometric matching claim.
        calls.append(len(predictions))
        return dict(gold_count=len(references), prediction_count=len(predictions), true_positive=0,
            false_positive=len(predictions), false_negative=len(references), exact=False, matches=[],
            unmatched_gold_ids=[row["bend_id"] for row in references],
            unmatched_prediction_ids=[row["bend_id"] for row in predictions])
    monkeypatch.setattr(modules.scorer, "reconstruct_gold11_retained_fusion_v1", inject_completed_variant)
    monkeypatch.setattr(modules.scorer, "match_lines", count_only_match_admission)
    result = _score(modules, inputs)
    assert len(calls) == 44
    refused = row_count > 4096
    row = _case(result, variant)
    assert row["disposition"] == ("SCORING_INPUT_REFUSED" if refused else "SCORED_CACHED_GEOMETRY")
    assert row["reason"] == ("PREDICTION_LINE_BUDGET" if refused else None)
    assert row["prediction_count"] == (0 if refused else 4096)
    assert row["false_negative"] == 7
    assert max(calls) == (0 if refused else 4096)
    assert result["reconstruction"]["cases"][case]["cached_line_counts"][variant] == row_count
    for name, lane in result["variants"].items():
        assert lane["geometry"]["reference_line_count"] == 71 and len(lane["case_dispositions"]) == 11
        assert lane["refused_case_count"] == int(refused and name == variant)


def test_high_level_authority_and_unknown_count_disclaimers_are_explicit_even_for_perfect_geometry(modules):
    result = _score(modules, _inputs())
    expected = {
        "schema": "GOLD11_CACHED_FUSION_GEOMETRY_DIAGNOSTIC_V1",
        "status": "COMPUTED_DEVELOPMENT_DIAGNOSTIC",
        "control_identity": "NEW_FROZEN_CACHED_FUSION_CONTROL_NOT_HISTORICAL_ENGINE_REPLAY",
        "historical_score_comparability": "UNESTABLISHED",
        "cohort_authentication": "CALLER_FROZEN_CASE_PAYLOAD_SOURCE_AND_REFERENCE_BYTES_REQUIRED",
        "comparison_scope": "PROVISIONAL_COMMON_MM_RAW_ENDPOINT_GEOMETRY_ONLY",
        "declared_common_frame": RAW_FRAME,
        "units_status": "PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        "reference_carrier_compatibility": "UNVERIFIED_NOT_ASSERTED_BY_SCORING",
        "reference_frame_compatibility": "PROVISIONAL_ASSUMPTION_NOT_VERIFIED",
        "carrier_semantics": "LEGACY_PREDICTED_LINES_NOT_ASSERTED_MEASURED_SURFACE_CARRIERS",
        "scoring_operations": "DIRECT_ENDPOINT_MATCHING_ONLY_NO_TRANSFORM_OR_UNIT_ESTIMATION",
        "inference_execution": "NOT_PERFORMED", "variant_selection": "NONE_ALL_FOUR_REPORTED",
        "failure_policy": "WHOLE_CASE_ZERO_PREDICTIONS_ALL_REFERENCE_FNS_NO_PREFIX_OR_DROPPED_CASE",
        "prediction_safety_cap_per_case": 4096, "observed_bend_count": None,
        "strict_physical_engine_targets_met": False, "production_use_allowed": False,
        "floor_release_allowed": False, "per_bend_measurement_accuracy": "NOT_EVALUATED_NO_MEASUREMENT_TRUTH",
        "conformance": "INDETERMINATE", "uncertainty_status": "UNCALIBRATED",
    }
    assert {key: result[key] for key in expected} == expected
    assert result["matching"] == dict(axis_angle_deg_max=5., transverse_offset_mm_max=4.,
        gold_span_overlap_min=.6, prediction_span_overlap_min=.6,
        method="UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST",
        assignment_identity="ROW_INDEX_ORIGINAL_DISPLAY_IDS_PRESERVED")
    assert RESOURCE_LIMITS == dict(wall_seconds=60, cpu_seconds=45,
        process_tree_rss_bytes=1024**3, stdout_bytes=1024**2)
