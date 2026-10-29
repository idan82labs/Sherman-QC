"""AI-adjudicated CAD event grouping with capacity-preserving scan ownership."""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Mapping, Sequence

from research.f1_cad_contextual_assignment_v1 import AssignmentConfig, assign


_EVENT_PATTERN = re.compile(r"CAD_EVENT_\d{4}")


def event_id(value: object) -> str:
    match = _EVENT_PATTERN.search(str(value))
    if match is None:
        raise ValueError(f"CAD event id absent from {value!r}")
    return match.group(0)


def ontology_filter_parent(
    parent: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted_ids = sorted(
        str(row["event_id"])
        for row in parent.get("records") or []
        if row.get("accepted") is True
    )
    binding_ids = sorted(str(row["event_id"]) for row in bindings)
    if accepted_ids != binding_ids:
        raise ValueError(f"ontology coverage gap: parent={accepted_ids} binding={binding_ids}")
    by_id = {str(row["event_id"]): row for row in bindings}
    predictions = [
        dict(row) for row in parent.get("predictions") or []
        if event_id(row.get("bend_id")) in by_id
        and by_id[event_id(row.get("bend_id"))]["status"] == "QC_BEND"
    ]
    records = [
        dict(row) for row in parent.get("records") or []
        if row.get("accepted") is True
        and by_id[str(row["event_id"])]["status"] == "QC_BEND"
    ]
    audit = []
    for row in sorted(bindings, key=lambda value: str(value["event_id"])):
        status = str(row["status"])
        if status not in {"QC_BEND", "NOT_QC_BEND"}:
            raise ValueError(f"unsupported ontology status {status}")
        audit.append({
            "event_ref": str(row["event_ref"]),
            "event_id": str(row["event_id"]),
            "status": status,
            "physical_event_group_id": row.get("physical_event_group_id"),
            "decision": "CONTINUE_TO_VISIBILITY" if status == "QC_BEND" else "TERMINATE_NOT_QC_BEND",
        })
    filtered = dict(parent)
    filtered["predictions"] = predictions
    filtered["records"] = records
    return filtered, audit


def _supported_span(row: Mapping[str, Any]) -> float:
    support = row.get("scan_support") or {}
    value = support.get("supported_span_mm", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def contract_physical_groups(
    finite_cad_rows: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    binding_by_id = {str(row["event_id"]): row for row in bindings}
    group_members: dict[str, list[dict[str, Any]]] = {}
    group_binding_members: dict[str, list[Mapping[str, Any]]] = {}
    for binding in bindings:
        if binding["status"] != "QC_BEND":
            continue
        group_id = str(binding["physical_event_group_id"])
        group_binding_members.setdefault(group_id, []).append(binding)
    for row in finite_cad_rows:
        identifier = event_id(row.get("bend_id"))
        binding = binding_by_id.get(identifier)
        if binding is None or binding["status"] != "QC_BEND":
            raise ValueError(f"finite CAD row lacks QC binding: {identifier}")
        group_members.setdefault(str(binding["physical_event_group_id"]), []).append(dict(row))
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for group_id in sorted(group_binding_members):
        declared = sorted(group_binding_members[group_id], key=lambda row: str(row["event_ref"]))
        surviving = group_members.get(group_id, [])
        owner_ref = str(declared[0]["canonical_line_owner_ref"])
        owner_id = event_id(owner_ref)
        owner_rows = [row for row in surviving if event_id(row.get("bend_id")) == owner_id]
        if owner_rows:
            representative = owner_rows[0]
            rule = "CANONICAL_OWNER_SURVIVED"
        elif surviving:
            representative = sorted(
                surviving,
                key=lambda row: (-_supported_span(row), event_id(row.get("bend_id"))),
            )[0]
            rule = "MAX_SUPPORTED_SPAN_FALLBACK"
        else:
            audit.append({
                "physical_event_group_id": group_id,
                "decision": "NULL_NO_FINITE_VISIBLE_MEMBER",
                "declared_event_refs": [str(row["event_ref"]) for row in declared],
            })
            continue
        representative_id = event_id(representative.get("bend_id"))
        emitted = dict(representative)
        emitted.update({
            "bend_id": f"GROUP_{group_id.replace(':', '_')}",
            "physical_event_group_id": group_id,
            "canonical_line_owner_ref": owner_ref,
            "representative_event_id": representative_id,
            "group_member_event_refs": [str(row["event_ref"]) for row in declared],
            "surviving_finite_event_ids": sorted(event_id(row.get("bend_id")) for row in surviving),
            "group_representative_rule": rule,
            "canonical_line_convention": "NEUTRAL_HALF_TURN",
            "overlay_line_convention": "FINITE_SCAN_RELOCATED_CAD_LINE",
            "assignment_source": "FINITE_SIGNED_PHYSICAL_EVENT_GROUP",
        })
        output.append(emitted)
        audit.append({
            "physical_event_group_id": group_id,
            "decision": "FINITE_GROUP_ROW",
            "representative_event_id": representative_id,
            "representative_rule": rule,
            "declared_event_refs": [str(row["event_ref"]) for row in declared],
            "surviving_finite_event_ids": emitted["surviving_finite_event_ids"],
            "contracted_event_ids": sorted(
                event_id(row.get("bend_id")) for row in surviving
                if event_id(row.get("bend_id")) != representative_id
            ),
        })
    return output, audit


def fuse_capacity_preserving(
    group_rows: Sequence[Mapping[str, Any]],
    scan_rows: Sequence[Mapping[str, Any]],
    config: AssignmentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = sorted((dict(row) for row in group_rows), key=lambda row: str(row["physical_event_group_id"]))
    scans = sorted((dict(row) for row in scan_rows), key=lambda row: str(row["bend_id"]))
    assignment = assign(groups, scans, config)
    matched_by_group = {row["cad_index"]: row for row in assignment["matches"]}
    matched_by_scan = {row["scan_index"]: row for row in assignment["matches"]}
    fused: list[dict[str, Any]] = []
    for scan_index, scan in enumerate(scans):
        added = dict(scan)
        match = matched_by_scan.get(scan_index)
        if match is not None:
            group = groups[match["cad_index"]]
            added.update({
                "assignment_source": "SCAN_OWNS_SIGNED_EVENT_GROUP",
                "physical_event_group_id": group["physical_event_group_id"],
                "canonical_line_owner_ref": group["canonical_line_owner_ref"],
                "group_member_event_refs": group["group_member_event_refs"],
            })
        fused.append(added)
    decisions = []
    for group_index, group in enumerate(groups):
        match = matched_by_group.get(group_index)
        if match is not None:
            decisions.append({
                "physical_event_group_id": group["physical_event_group_id"],
                "decision": "SCAN_OWNS_ASSIGNED_GROUP",
                "scan_owner_id": scans[match["scan_index"]]["bend_id"],
                "assignment_cost": match["cost"],
                "pair_geometry": match["geometry"],
            })
            continue
        eligible = [
            row for row in assignment["pair_records"]
            if row["cad_index"] == group_index and row["eligible"]
        ]
        added = dict(group)
        added["bend_id"] = "CAD_" + str(group["bend_id"])
        added["assignment_source"] = "UNMATCHED_FINITE_SIGNED_EVENT_GROUP"
        fused.append(added)
        decisions.append({
            "physical_event_group_id": group["physical_event_group_id"],
            "decision": "UNMATCHED_GROUP_ADMITTED",
            "eligible_scan_indices": [row["scan_index"] for row in eligible],
        })
    return fused, {**assignment, "decisions": decisions, "null_one_to_one_conflict_count": 0}


def contract_payload() -> dict[str, Any]:
    return {
        "schema": "V2_ONTOLOGY_GROUPED_CAPACITY_FUSION_V1",
        "assignment": asdict(AssignmentConfig()),
        "status_before_visibility": True,
        "one_output_capacity_per_physical_group": True,
        "unmatched_finite_groups_admitted": True,
        "null_one_to_one_conflict_forbidden": True,
        "ai_adjudicated_case_mapping_used": True,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }
