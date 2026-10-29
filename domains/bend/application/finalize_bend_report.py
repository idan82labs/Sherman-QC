"""Pure, additive operator decision boundary; no geometry, models or file IO.

Legacy detection/summary fields are retained for compatibility, not endorsed as
physical completion. Version 1 cannot authorize disposition: registration and
native render/measurement qualification are not yet established end to end.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from domains.bend.services.runtime_semantics import normalize_observability_state


def _token(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _invalid_token(match: Mapping[str, Any], key: str, allowed: set[str]) -> bool:
    value = match.get(key)
    return value is not None and (not isinstance(value, str) or _token(value) not in allowed)


def _countability_value(match: Mapping[str, Any]) -> Any:
    cad = match.get("cad_bend")
    cad = cad if isinstance(cad, Mapping) else {}
    value = match.get("countable_in_regression")
    if value is None:
        value = cad.get("countable_in_regression")
    return value


def _countable(match: Mapping[str, Any]) -> bool:
    # Malformed flags retain an unknown target, never positive evidence or an
    # implicit exclusion from the denominator. Explicit False excludes it.
    return _countability_value(match) is not False


def finalize_bend_report(
    report: Mapping[str, Any], *, render_artifacts_available: bool = False,
) -> dict[str, Any]:
    """Attach conservative, versioned evidence without changing legacy fields.

    Identity is a reported CAD/event identity, not proof of event ontology.
    Duplicate IDs count once and abstain. No count prior, filename, tolerance
    status, confidence heuristic or expected-count field creates an observation.
    """
    raw_alignment = report.get("alignment")
    alignment = raw_alignment if isinstance(raw_alignment, Mapping) else {}
    raw_abstention = alignment.get("alignment_abstained")
    malformed_registration = (
        raw_alignment is not None and not isinstance(raw_alignment, Mapping)
    ) or (raw_abstention is not None and type(raw_abstention) is not bool)
    abstained = raw_abstention is True or malformed_registration
    registration = "ABSTAINED" if abstained else "UNVERIFIED"
    reasons = {"NO_PRODUCTION_DISPOSITION", "REGISTRATION_" + registration}
    if malformed_registration:
        reasons.add("MALFORMED_REGISTRATION_EVIDENCE")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    raw_matches = report.get("matches")
    if not isinstance(raw_matches, list):
        raw_matches = []
        reasons.add("MATCH_RECORDS_UNAVAILABLE")
    for index, match in enumerate(raw_matches):
        if not isinstance(match, Mapping):
            match = {}
        cad = match.get("cad_bend")
        cad = cad if isinstance(cad, Mapping) else {}
        identity = match.get("bend_id", cad.get("bend_id"))
        if not isinstance(identity, str) or not identity.strip():
            identity = f"UNIDENTIFIED_RECORD_{index + 1}"
            reasons.add("EVENT_ID_UNAVAILABLE")
        groups[identity].append(match)

    counts = {"formed": 0, "unformed": 0, "unknown": 0, "total": 0, "partially_observed": 0}
    bends = []
    for identity, records in sorted(groups.items()):
        match = records[0]
        state = normalize_observability_state(match.get("observability_state"))
        detail = _token(match.get("observability_detail_state")) or _token(match.get("observability_state"))
        physical = _token(match.get("physical_completion_state"))
        row_reasons: set[str] = set()
        if any(value is not None and type(value) is not bool
               for value in (_countability_value(record) for record in records)):
            state = "UNKNOWN"
            row_reasons.add("MALFORMED_COUNTABILITY_EVIDENCE")
        observation_tokens = {"", "NONE", "FORMED", "UNFORMED", "UNKNOWN", "UNOBSERVED", "PARTIALLY_OBSERVED", "OBSERVED_FORMED", "OBSERVED_NOT_FORMED"}
        physical_tokens = {"", "NONE", "UNKNOWN", "FORMED", "UNFORMED", "NOT_FORMED", "PARTIALLY_FORMED", "PARTIAL"}
        if any(_invalid_token(record, key, allowed) for record in records for key, allowed in (
            ("observability_state", observation_tokens),
            ("observability_detail_state", observation_tokens),
            ("physical_completion_state", physical_tokens),
        )):
            state = "UNKNOWN"
            row_reasons.add("MALFORMED_STATE_EVIDENCE")
        if len(records) > 1:
            state = "UNKNOWN"
            detail = "UNKNOWN"
            physical = ""
            row_reasons.add("DUPLICATE_BEND_ID")
        if identity.startswith("UNIDENTIFIED_RECORD_"):
            state = "UNKNOWN"
            row_reasons.add("EVENT_ID_UNAVAILABLE")
        if detail in {"UNOBSERVED", "PARTIALLY_OBSERVED", "UNKNOWN", "NONE"}:
            state = "UNKNOWN"
            row_reasons.add("INCOMPLETE_OBSERVATION")
        if physical in {"PARTIALLY_FORMED", "PARTIAL"}:
            state = "UNKNOWN"
            row_reasons.add("PHYSICAL_STAGE_NOT_QUALIFIED")
        physical_state = {"FORMED": "FORMED", "UNFORMED": "UNFORMED", "NOT_FORMED": "UNFORMED"}.get(physical)
        detail_state = {"FORMED": "FORMED", "UNFORMED": "UNFORMED", "OBSERVED_FORMED": "FORMED", "OBSERVED_NOT_FORMED": "UNFORMED"}.get(detail)
        if state != "UNKNOWN" and any(
            candidate is not None and candidate != state
            for candidate in (physical_state, detail_state)
        ):
            state = "UNKNOWN"
            row_reasons.add("CONFLICTING_STATE_EVIDENCE")
        if abstained:
            state = "UNKNOWN"
            row_reasons.add("REGISTRATION_ABSTAINED")
            if malformed_registration:
                row_reasons.add("MALFORMED_REGISTRATION_EVIDENCE")
        if state == "UNKNOWN":
            row_reasons.add("EXPLICIT_OBSERVATION_REQUIRED")
        if any(_countable(record) for record in records):
            counts["total"] += 1
            counts[state.lower()] += 1
            counts["partially_observed"] += int(detail == "PARTIALLY_OBSERVED")
        bends.append({
            "bend_id": identity,
            "state": state,
            "observation_detail": detail or "UNSPECIFIED",
            "reasons": sorted(row_reasons),
        })
        reasons.update(row_reasons)

    render_evidence = "UNVERIFIED" if render_artifacts_available is True else "UNAVAILABLE"
    reasons.add("RENDER_EVIDENCE_" + render_evidence)
    return {
        **report,
        "decision_contract": {
            "schema_version": 1,
            "kind": "BEND_DECISION_CONTRACT_V1",
            "observed_counts": counts,
            "registration": registration,
            "render_evidence": render_evidence,
            "production_disposition": "NOT_AUTHORIZED",
            "reasons": sorted(reasons),
            "bends": bends,
        },
    }


def format_bend_decision_table(report: Mapping[str, Any]) -> str:
    """Text export from the same finalized evidence as JSON, not legacy counts."""
    decision = report["decision_contract"]
    counts = decision["observed_counts"]
    lines = [
        "RESEARCH ONLY - NOT QUALIFIED FOR PRODUCTION ACCEPT/REJECT",
        "Observation states are experimental evidence, not part disposition.",
        f"Observed formed: {counts['formed']}/{counts['total']}",
        f"Observed unformed: {counts['unformed']}; Unknown: {counts['unknown']}",
        f"Registration: {decision['registration']}; Render evidence: {decision['render_evidence']}",
        "A missing observation is not an unfinished or defective bend.",
        "",
        "Bend | Observation | Evidence detail | Review reasons",
    ]
    for row in decision["bends"]:
        # Keep one record per line even if an external part/event ID has controls.
        identity = " ".join(row["bend_id"].split())
        detail = " ".join(row["observation_detail"].split())
        lines.append(f"{identity} | {row['state']} | {detail} | {', '.join(row['reasons']) or '-'}")
    return "\n".join(lines) + "\n"
