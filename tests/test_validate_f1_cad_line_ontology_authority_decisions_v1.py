from __future__ import annotations

import copy
import hashlib

from scripts.validate_f1_cad_line_ontology_authority_decisions_v1 import _stable, validate


def _manifest():
    return {"schema":"F1_CAD_LINE_ONTOLOGY_AUTHORITY_PACK_V1","cases":[{"events":[{"event_ref":"C001:E1"},{"event_ref":"C001:E2"}]}]}


def _decision(manifest,role,name):
    return {"schema":"F1_CAD_LINE_ONTOLOGY_AUTHORITY_DECISION_V1","review_pack_sha256":hashlib.sha256(_stable(manifest)).hexdigest(),"authority_role":role,"authority_name":name,"decision":"COMPLETE","events":[{"event_ref":"C001:E1","event_status":"QC_BEND","line_convention":"NEUTRAL_HALF_TURN","replacement_line":None,"rationale":"This is a controlled formed bend."},{"event_ref":"C001:E2","event_status":"NOT_QC_BEND","line_convention":"NONE","replacement_line":None,"rationale":"This feature is not a controlled bend."}]}


def test_matching_distinct_authorities_are_ready():
    manifest=_manifest();left=_decision(manifest,"MANUFACTURING_AUTHORITY","Alice");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","Bob")
    assert validate(manifest,left,right)["status"]=="READY_TO_AMEND_CIBEC_CHARTER"


def test_disagreement_requires_adjudication():
    manifest=_manifest();left=_decision(manifest,"MANUFACTURING_AUTHORITY","Alice");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","Bob");right["events"][0]["line_convention"]="OBSERVED_SURFACE_INTERSECTION"
    result=validate(manifest,left,right)
    assert result["status"]=="AWAITING_ADJUDICATION" and result["disagreement_event_refs"]==["C001:E1"]


def test_matching_unresolved_decisions_fail_closed():
    manifest=_manifest();left=_decision(manifest,"MANUFACTURING_AUTHORITY","Alice");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","Bob")
    for payload in (left,right):
        payload["events"][0]["event_status"]="UNRESOLVED"
        payload["events"][0]["line_convention"]="NONE"
    result=validate(manifest,left,right)
    assert result["status"]=="AWAITING_ADJUDICATION"
    assert result["disagreement_event_refs"]==[]
    assert result["unresolved_event_refs"]==["C001:E1"]


def test_pending_or_same_authority_fails_closed():
    manifest=_manifest();left=_decision(manifest,"MANUFACTURING_AUTHORITY","REQUIRED_REAL_NAME");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","REQUIRED_REAL_NAME")
    assert validate(manifest,left,right)["status"]=="AWAITING_AUTHORITY_REVIEW"
    left=_decision(manifest,"MANUFACTURING_AUTHORITY","Alice");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","Alice")
    assert "AUTHORITIES_MUST_BE_DISTINCT" in validate(manifest,left,right)["blockers"]


def test_incomplete_decision_and_missing_rationale_fail_closed():
    manifest=_manifest();left=_decision(manifest,"MANUFACTURING_AUTHORITY","Alice");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","Bob")
    left["decision"]="PENDING"
    left["events"][0]["rationale"]=""
    result=validate(manifest,left,right)
    assert "MANUFACTURING_AUTHORITY_DECISION_MUST_BE_COMPLETE" in result["blockers"]
    assert "manufacturing:C001:E1:RATIONALE_REQUIRED" in result["blockers"]


def test_authority_drawn_line_requires_finite_numeric_coordinates():
    manifest=_manifest();left=_decision(manifest,"MANUFACTURING_AUTHORITY","Alice");right=_decision(manifest,"COUNTERSIGNING_AUTHORITY","Bob")
    for payload in (left,right):
        payload["events"][0]["line_convention"]="OTHER_AUTHORITY_DRAWN"
        payload["events"][0]["replacement_line"]=[[0,0,0],[1,float("nan"),0]]
    result=validate(manifest,left,right)
    assert any("REPLACEMENT_LINE_REQUIRED" in blocker for blocker in result["blockers"])
