"""Compile two independent CAD line-ontology reviews fail-closed.

This module does not infer manufacturing intent.  It only binds two complete,
distinct authority decisions to the frozen review pack.  An actionable mapping
is emitted only when every event agrees and none remains unresolved; otherwise
the result is a bounded adjudication queue with no executable mapping.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from scripts.validate_f1_cad_line_ontology_authority_decisions_v1 import validate


def _stable(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(payload: Any) -> str:
    return hashlib.sha256(_stable(payload)).hexdigest()


def compile_consensus(
    manifest: Mapping[str, Any],
    manufacturing: Mapping[str, Any],
    countersigning: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate(dict(manifest), dict(manufacturing), dict(countersigning))
    authority_evidence = [
        {
            "authority_role": manufacturing.get("authority_role"),
            "authority_name": manufacturing.get("authority_name"),
            "decision_sha256": _sha(manufacturing),
        },
        {
            "authority_role": countersigning.get("authority_role"),
            "authority_name": countersigning.get("authority_name"),
            "decision_sha256": _sha(countersigning),
        },
    ]
    if validation["status"] != "READY_TO_AMEND_CIBEC_CHARTER":
        refs = sorted(
            set(validation.get("disagreement_event_refs", []))
            | set(validation.get("unresolved_event_refs", []))
        )
        return {
            "schema": "F1_CAD_LINE_ONTOLOGY_ADJUDICATION_QUEUE_V1",
            "status": validation["status"],
            "review_pack_sha256": validation["review_pack_sha256"],
            "event_denominator": validation["event_denominator"],
            "authority_evidence": authority_evidence,
            "blockers": validation["blockers"],
            "disagreement_event_refs": validation.get("disagreement_event_refs", []),
            "unresolved_event_refs": validation.get("unresolved_event_refs", []),
            "adjudication_event_refs": refs,
            "actionable_mapping": None,
            "model_execution_authorized": False,
            "truth_opened": False,
            "physical_assets_opened": 0,
        }

    left = {row["event_ref"]: row for row in manufacturing["events"]}
    right = {row["event_ref"]: row for row in countersigning["events"]}
    ordered_refs = [
        event["event_ref"] for case in manifest["cases"] for event in case["events"]
    ]
    mapping = []
    for ref in ordered_refs:
        a, b = left[ref], right[ref]
        mapping.append(
            {
                "event_ref": ref,
                "event_status": a["event_status"],
                "line_convention": a["line_convention"],
                "replacement_line": a.get("replacement_line"),
                "manufacturing_rationale": a.get("rationale", "").strip(),
                "countersigning_rationale": b.get("rationale", "").strip(),
            }
        )
    return {
        "schema": "F1_CAD_LINE_ONTOLOGY_CONSENSUS_V1",
        "status": "READY_TO_PREREGISTER_SIGNED_CIBEC_CONTRACTION",
        "review_pack_sha256": validation["review_pack_sha256"],
        "event_denominator": len(mapping),
        "authority_evidence": authority_evidence,
        "actionable_mapping": mapping,
        "actionable_mapping_sha256": _sha(mapping),
        "model_execution_authorized": False,
        "next_action": "PREREGISTER_ONE_BOUNDED_SIGNED_CIBEC_CONTRACTION_EXPERIMENT",
        "truth_opened": False,
        "physical_assets_opened": 0,
    }

