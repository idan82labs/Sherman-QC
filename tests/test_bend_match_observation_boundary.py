"""Source-extracted dataclass checks, not native geometry/runtime qualification.

Compile only the exact two named class definitions from the inspected source.
No module imports, detector methods, geometry algorithms or physical IO execute.
Minimal CAD/type stand-ins exercise the real initialization/summary methods.
"""
import ast
from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from domains.bend.application.finalize_bend_report import finalize_bend_report
from domains.bend.services.runtime_semantics import (
    feature_family_for_target,
    is_explicit_observability_evidence,
    legacy_observability_detail_state,
    measurement_primitive_for_feature_family,
    normalize_observability_state,
)


SOURCE = Path(__file__).resolve().parents[1] / "backend" / "feature_detection" / "bend_detector.py"
PURE_CALLS = {
    "field", "str", "bool", "round", "float", "int", "sum", "len", "max", "getattr", "isinstance",
    "normalize_observability_state", "legacy_observability_detail_state",
    "is_explicit_observability_evidence", "feature_family_for_target",
    "measurement_primitive_for_feature_family",
}
PURE_METHODS = {"upper", "tolist", "append", "get", "compute_summary", "to_dict", "join"}


@pytest.fixture
def extracted_classes(monkeypatch):
    source = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = [node for node in source.body if isinstance(node, ast.ClassDef)
                and node.name in {"BendMatch", "BendInspectionReport"}]
    assert [node.name for node in selected] == ["BendMatch", "BendInspectionReport"]
    for node in selected:
        assert len(node.decorator_list) == 1 and isinstance(node.decorator_list[0], ast.Name)
        assert node.decorator_list[0].id == "dataclass"
        # Fail before execution if a future class revision adds a new IO/native
        # call: the safety audit must then be repeated, never silently expanded.
        for child in ast.walk(node):
            assert not isinstance(child, (ast.Import, ast.ImportFrom))
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    assert child.func.id in PURE_CALLS, child.func.id
                else:
                    assert isinstance(child.func, ast.Attribute) and child.func.attr in PURE_METHODS
    module = ModuleType("_source_extracted_bend_match_observation")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update({
        "dataclass": dataclass, "field": field, "Any": Any, "Dict": Dict,
        "List": List, "Optional": Optional, "np": SimpleNamespace(ndarray=object),
        "BendSpecification": object, "DetectedBend": object,
        "normalize_observability_state": normalize_observability_state,
        "legacy_observability_detail_state": legacy_observability_detail_state,
        "is_explicit_observability_evidence": is_explicit_observability_evidence,
        "feature_family_for_target": feature_family_for_target,
        "measurement_primitive_for_feature_family": measurement_primitive_for_feature_family,
    })
    compiled = compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec", dont_inherit=True)
    exec(compiled, module.__dict__)
    return module.BendMatch, module.BendInspectionReport


def cad(identity="A", countable=True):
    return SimpleNamespace(
        bend_id=identity, bend_form="FOLDED", feature_type="BEND", countable_in_regression=countable,
        target_angle=90.0, target_radius=2.0, tolerance_angle=1.0, tolerance_radius=0.5,
        bend_line_start=SimpleNamespace(tolist=lambda: [0.0, 0.0, 0.0]),
        bend_line_end=SimpleNamespace(tolist=lambda: [0.0, 10.0, 0.0]),
    )


@pytest.mark.parametrize("status", ["PASS", "FAIL", "WARNING", "NOT_DETECTED"])
def test_source_extracted_unknown_cannot_be_filled_from_status_or_physical_claim(extracted_classes, status):
    BendMatch, _ = extracted_classes
    match = BendMatch(cad_bend=cad(), detected_bend=None, status=status,
                      observability_state="UNKNOWN", physical_completion_state="FORMED")
    assert match.observability_state == "UNKNOWN"
    assert match.to_dict()["observability_state"] == "UNKNOWN"


@pytest.mark.parametrize("state,detail", [("FORMED", "OBSERVED_FORMED"), ("UNFORMED", "OBSERVED_NOT_FORMED")])
def test_source_extracted_explicit_state_derives_missing_detail(extracted_classes, state, detail):
    BendMatch, _ = extracted_classes
    match = BendMatch(cad_bend=cad(), detected_bend=None, observability_state=state)
    assert match.observability_state == state
    assert match.observability_detail_state == detail


def test_source_extracted_explicit_unobserved_detail_survives_and_final_decision_abstains(extracted_classes):
    BendMatch, BendInspectionReport = extracted_classes
    match = BendMatch(cad_bend=cad(), detected_bend=None, status="PASS", observability_state="FORMED",
                      observability_detail_state="UNOBSERVED", physical_completion_state="FORMED")
    assert match.observability_detail_state == "UNOBSERVED"
    report = BendInspectionReport(part_id="SYNTHETIC", matches=[match])
    raw = report.to_dict()
    # The legacy summary is not the contradiction-resolution authority.
    assert raw["summary"]["detected"] == 1
    finalized = finalize_bend_report(raw)
    assert finalized["decision_contract"]["observed_counts"]["unknown"] == 1
    assert finalized["decision_contract"]["observed_counts"]["formed"] == 0


@pytest.mark.parametrize("detail", ["", "PARTIALLY_OBSERVED"])
def test_source_extracted_partial_observation_stays_unknown(extracted_classes, detail):
    BendMatch, _ = extracted_classes
    match = BendMatch(cad_bend=cad(), detected_bend=None, status="PASS", observability_state="PARTIALLY_OBSERVED",
                      observability_detail_state=detail, physical_completion_state="PARTIALLY_FORMED")
    assert match.observability_state == "UNKNOWN"
    assert match.observability_detail_state == "PARTIALLY_OBSERVED"
    assert match.physical_completion_state == "PARTIALLY_FORMED"


def test_source_extracted_legacy_detection_and_strict_observation_counts_remain_separate(extracted_classes):
    BendMatch, BendInspectionReport = extracted_classes
    matches = [
        BendMatch(cad_bend=cad("A"), detected_bend=None, status="PASS", observability_state="UNKNOWN"),
        BendMatch(cad_bend=cad("B"), detected_bend=None, status="PASS", observability_state="FORMED"),
        BendMatch(cad_bend=cad("C"), detected_bend=None, status="NOT_DETECTED", observability_state="UNFORMED"),
        BendMatch(cad_bend=cad("D"), detected_bend=None, status="WARNING", observability_state="PARTIALLY_OBSERVED",
                  observability_detail_state="PARTIALLY_OBSERVED"),
        BendMatch(cad_bend=cad("E"), detected_bend=None, status="NOT_DETECTED", observability_state="UNOBSERVED"),
        BendMatch(cad_bend=cad("P", countable=False), detected_bend=None, status="PASS", observability_state="FORMED"),
    ]
    report = BendInspectionReport(part_id="SYNTHETIC", matches=matches)
    raw = report.to_dict()
    assert report.total_cad_bends == 5
    assert report.detected_count == 3
    assert report.pass_count == 2
    assert report.observed_count == 2  # formed + unformed, never UNKNOWN+PASS
    assert report.detected_process_feature_count == 1
    assert raw["summary"]["completed_bends"] == 3  # deliberately preserved legacy field
    assert raw["summary"]["observed_bends"] == 2
    decision = finalize_bend_report(raw)["decision_contract"]
    assert decision["observed_counts"] == {
        "formed": 1, "unformed": 1, "unknown": 3, "total": 5, "partially_observed": 1,
    }
    assert decision["production_disposition"] == "NOT_AUTHORIZED"
