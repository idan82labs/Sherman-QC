from __future__ import annotations

import copy
import hashlib

from research.f1_cad_line_ontology_consensus_v1 import _stable, compile_consensus


def _manifest():
    return {
        "schema": "F1_CAD_LINE_ONTOLOGY_AUTHORITY_PACK_V1",
        "cases": [{"events": [{"event_ref": "C001:E1"}, {"event_ref": "C001:E2"}]}],
    }


def _decision(manifest, role, name):
    return {
        "schema": "F1_CAD_LINE_ONTOLOGY_AUTHORITY_DECISION_V1",
        "review_pack_sha256": hashlib.sha256(_stable(manifest)).hexdigest(),
        "authority_role": role,
        "authority_name": name,
        "decision": "COMPLETE",
        "events": [
            {
                "event_ref": "C001:E1",
                "event_status": "QC_BEND",
                "line_convention": "NEUTRAL_HALF_TURN",
                "replacement_line": None,
                "rationale": "This is the formed bend axis.",
            },
            {
                "event_ref": "C001:E2",
                "event_status": "NOT_QC_BEND",
                "line_convention": "NONE",
                "replacement_line": None,
                "rationale": "This is not a controlled bend.",
            },
        ],
    }


def test_compiles_only_complete_resolved_consensus():
    manifest = _manifest()
    left = _decision(manifest, "MANUFACTURING_AUTHORITY", "Alice")
    right = _decision(manifest, "COUNTERSIGNING_AUTHORITY", "Bob")
    result = compile_consensus(manifest, left, right)
    assert result["status"] == "READY_TO_PREREGISTER_SIGNED_CIBEC_CONTRACTION"
    assert result["event_denominator"] == 2
    assert result["actionable_mapping"][0]["line_convention"] == "NEUTRAL_HALF_TURN"
    assert result["model_execution_authorized"] is False


def test_disagreement_emits_queue_without_mapping():
    manifest = _manifest()
    left = _decision(manifest, "MANUFACTURING_AUTHORITY", "Alice")
    right = _decision(manifest, "COUNTERSIGNING_AUTHORITY", "Bob")
    right["events"][0]["line_convention"] = "OBSERVED_SURFACE_INTERSECTION"
    result = compile_consensus(manifest, left, right)
    assert result["status"] == "AWAITING_ADJUDICATION"
    assert result["adjudication_event_refs"] == ["C001:E1"]
    assert result["actionable_mapping"] is None


def test_matching_unresolved_emits_queue_without_mapping():
    manifest = _manifest()
    left = _decision(manifest, "MANUFACTURING_AUTHORITY", "Alice")
    right = _decision(manifest, "COUNTERSIGNING_AUTHORITY", "Bob")
    for payload in (left, right):
        payload["events"][0]["event_status"] = "UNRESOLVED"
        payload["events"][0]["line_convention"] = "NONE"
    result = compile_consensus(manifest, left, right)
    assert result["unresolved_event_refs"] == ["C001:E1"]
    assert result["actionable_mapping"] is None
