from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_f1_current_unified_gate_evidence_v1",
    ROOT / "scripts/build_f1_current_unified_gate_evidence_v1.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_current_evidence_is_lane_separated_and_development_only() -> None:
    evidence = MODULE.build(ROOT)
    cad = evidence["cad_assisted"]
    scan = evidence["scan_only"]
    assert cad["lane"] == "CAD_ASSISTED"
    assert scan["lane"] == "SCAN_ONLY"
    assert cad["evidence_class"] == scan["evidence_class"] == "CONSUMED_PHYSICAL"
    assert cad["truth"]["morphology_held_out"] is False
    assert scan["truth"]["dual_review_verified"] is False
    assert cad["baseline_metrics"]["f1"] == 0.4580152671755725
    assert cad["candidate_metrics"]["f1"] == 0.4556962025316456
    assert scan["baseline_metrics"]["f1"] == 0.5932203389830508
    assert scan["candidate_metrics"]["f1"] == 0.5179856115107914
    assert scan["counts"]["physical_scans"] == 12
    assert scan["counts"]["morphology_families"] == 12
    assert scan["execution_counts"] == {
        "physical_scans_executed": 15,
        "physical_scans_scored": 12,
        "physical_scans_visual_only": 3,
    }
    assert cad["visual"]["candidate_binding"] == "TOPOLOGY_ONLY_V9_NOT_AGGRESSIVE_TRIPLE_UNION"


def test_current_evidence_evaluates_fail_closed() -> None:
    evidence = MODULE.build(ROOT)
    gates = {key: MODULE.evaluate(value) for key, value in evidence.items()}
    assert gates["cad_assisted"]["grade"] == "C_NO_GENERALIZATION_EVIDENCE"
    assert gates["cad_assisted"]["direction"] == "NEGATIVE_DEVELOPMENT"
    assert gates["scan_only"]["grade"] == "C_NO_GENERALIZATION_EVIDENCE"
    assert gates["scan_only"]["direction"] == "NEGATIVE_DEVELOPMENT"
    assert gates["cad_assisted"]["operational_badge"] == "RED"
    assert gates["scan_only"]["operational_badge"] == "RED"
