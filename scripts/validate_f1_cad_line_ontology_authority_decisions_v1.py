#!/usr/bin/env python3
"""Validate two independent decisions over the frozen line-ontology pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PENDING_NAMES = {"", "REQUIRED_REAL_NAME", "PENDING"}
CONVENTIONS = {"CURVATURE_CENTER","NEUTRAL_HALF_TURN","OBSERVED_SURFACE_INTERSECTION","OTHER_AUTHORITY_DRAWN"}
STATUSES = {"QC_BEND","NOT_QC_BEND","UNRESOLVED"}


def _stable(payload: Any) -> bytes:
    return (json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def validate(manifest: dict, left: dict, right: dict) -> dict:
    if manifest.get("schema") != "F1_CAD_LINE_ONTOLOGY_AUTHORITY_PACK_V1":
        raise ValueError("invalid authority pack schema")
    pack_sha = hashlib.sha256(_stable(manifest)).hexdigest()
    expected_refs = [event["event_ref"] for case in manifest["cases"] for event in case["events"]]
    if len(expected_refs) != len(set(expected_refs)):
        raise ValueError("authority pack contains duplicate event_ref values")
    blockers = []
    decisions = []
    names = []
    for label, payload, role in (
        ("manufacturing",left,"MANUFACTURING_AUTHORITY"),
        ("countersigning",right,"COUNTERSIGNING_AUTHORITY"),
    ):
        if payload.get("schema") != "F1_CAD_LINE_ONTOLOGY_AUTHORITY_DECISION_V1" or payload.get("authority_role") != role:
            raise ValueError(f"invalid {label} decision schema or role")
        if payload.get("decision") != "COMPLETE":
            blockers.append(f"{role}_DECISION_MUST_BE_COMPLETE")
        if payload.get("review_pack_sha256") != pack_sha:
            raise ValueError(f"{label} decision is bound to the wrong pack")
        name = str(payload.get("authority_name") or "").strip()
        names.append(name)
        if name in PENDING_NAMES:
            blockers.append(f"{role}_NAME_REQUIRED")
        rows = payload.get("events")
        if not isinstance(rows,list) or [row.get("event_ref") for row in rows] != expected_refs:
            raise ValueError(f"{label} event denominator/order mismatch")
        indexed = {}
        for row in rows:
            status, convention = row.get("event_status"), row.get("line_convention")
            rationale = row.get("rationale")
            if not isinstance(rationale, str) or len(rationale.strip()) < 8 or len(rationale) > 2000:
                blockers.append(f"{label}:{row['event_ref']}:RATIONALE_REQUIRED")
            if status not in STATUSES:
                blockers.append(f"{label}:{row['event_ref']}:EVENT_STATUS_REQUIRED")
                continue
            if status == "QC_BEND" and convention not in CONVENTIONS:
                blockers.append(f"{label}:{row['event_ref']}:LINE_CONVENTION_REQUIRED")
                continue
            if status != "QC_BEND" and convention not in {None,"NONE"}:
                blockers.append(f"{label}:{row['event_ref']}:NON_BEND_MUST_NOT_SELECT_LINE")
                continue
            if convention == "OTHER_AUTHORITY_DRAWN":
                replacement = row.get("replacement_line")
                if (
                    not isinstance(replacement,list)
                    or len(replacement)!=2
                    or any(not isinstance(point,list) or len(point)!=3 for point in replacement)
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        for point in replacement
                        for value in point
                    )
                ):
                    blockers.append(f"{label}:{row['event_ref']}:REPLACEMENT_LINE_REQUIRED")
                    continue
            indexed[row["event_ref"]] = {"event_status":status,"line_convention":convention,"replacement_line":row.get("replacement_line")}
        decisions.append(indexed)
    if len(set(names)) != 2 and not any(name in PENDING_NAMES for name in names):
        blockers.append("AUTHORITIES_MUST_BE_DISTINCT")
    disagreements = []
    unresolved = []
    if not blockers:
        for ref in expected_refs:
            if decisions[0][ref] != decisions[1][ref]:
                disagreements.append(ref)
            elif decisions[0][ref]["event_status"] == "UNRESOLVED":
                # Agreement that an event is unresolved is not an actionable
                # ontology decision.  Failing this closed prevents a partially
                # defined target from being treated as execution authority.
                unresolved.append(ref)
    status = (
        "READY_TO_AMEND_CIBEC_CHARTER"
        if not blockers and not disagreements and not unresolved
        else (
            "AWAITING_ADJUDICATION"
            if (disagreements or unresolved) and not blockers
            else "AWAITING_AUTHORITY_REVIEW"
        )
    )
    return {
        "schema":"F1_CAD_LINE_ONTOLOGY_AUTHORITY_VALIDATION_V1",
        "status":status,
        "review_pack_sha256":pack_sha,
        "event_denominator":len(expected_refs),
        "blockers":blockers,
        "disagreement_event_refs":disagreements,
        "unresolved_event_refs":unresolved,
        "physical_assets_opened":0,
        "truth_opened":False,
    }


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--manufacturing",type=Path,required=True);parser.add_argument("--countersigning",type=Path,required=True);parser.add_argument("--output",type=Path);args=parser.parse_args()
    result=validate(json.loads(args.manifest.read_text()),json.loads(args.manufacturing.read_text()),json.loads(args.countersigning.read_text()))
    if args.output: args.output.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True));return 0 if result["status"]=="READY_TO_AMEND_CIBEC_CHARTER" else 2


if __name__=="__main__":raise SystemExit(main())
