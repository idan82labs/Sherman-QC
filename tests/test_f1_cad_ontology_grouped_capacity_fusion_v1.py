from __future__ import annotations

import pytest

from research.f1_cad_contextual_assignment_v1 import AssignmentConfig
from research.f1_cad_ontology_grouped_capacity_fusion_v1 import (
    contract_physical_groups,
    fuse_capacity_preserving,
    ontology_filter_parent,
)


def _line(identifier: str, y: float, span: float = 10.0) -> dict:
    return {
        "bend_id": identifier,
        "line_start": [0.0, y, 0.0],
        "line_end": [span, y, 0.0],
        "scan_support": {"supported_span_mm": span},
    }


def _binding(event_id: str, group: str, owner: str | None = None, status: str = "QC_BEND") -> dict:
    ref = f"C001:{event_id}"
    return {
        "event_ref": ref,
        "event_id": event_id,
        "status": status,
        "physical_event_group_id": group if status == "QC_BEND" else None,
        "canonical_line_owner_ref": owner or ref if status == "QC_BEND" else None,
    }


def test_contracts_multiple_primitives_to_one_physical_group() -> None:
    bindings = [
        _binding("CAD_EVENT_0002", "C001:MG_001"),
        _binding("CAD_EVENT_0003", "C001:MG_001", owner="C001:CAD_EVENT_0002"),
    ]
    rows, audit = contract_physical_groups(
        [_line("CAD_CAD_EVENT_0002", 0.0, 8.0), _line("CAD_CAD_EVENT_0003", 0.0, 12.0)],
        bindings,
    )
    assert len(rows) == 1
    assert rows[0]["representative_event_id"] == "CAD_EVENT_0002"
    assert rows[0]["surviving_finite_event_ids"] == ["CAD_EVENT_0002", "CAD_EVENT_0003"]
    assert audit[0]["contracted_event_ids"] == ["CAD_EVENT_0003"]


def test_unmatched_group_is_admitted_even_with_eligible_scan_conflict() -> None:
    groups = [
        {**_line("GROUP_A", 0.0), "physical_event_group_id": "A", "canonical_line_owner_ref": "A:1", "group_member_event_refs": ["A:1"]},
        {**_line("GROUP_B", 1.0), "physical_event_group_id": "B", "canonical_line_owner_ref": "B:1", "group_member_event_refs": ["B:1"]},
    ]
    scans = [_line("SCAN_1", 0.2)]
    fused, audit = fuse_capacity_preserving(groups, scans, AssignmentConfig())
    assert len(fused) == 2
    assert sum(row["assignment_source"] == "SCAN_OWNS_SIGNED_EVENT_GROUP" for row in fused) == 1
    assert sum(row["assignment_source"] == "UNMATCHED_FINITE_SIGNED_EVENT_GROUP" for row in fused) == 1
    assert audit["null_one_to_one_conflict_count"] == 0


def test_not_qc_events_terminate_before_visibility() -> None:
    parent = {
        "records": [{"event_id": "CAD_EVENT_0001", "accepted": True}],
        "predictions": [{**_line("CAD_EVENT_0001", 0.0)}],
    }
    filtered, audit = ontology_filter_parent(
        parent, [_binding("CAD_EVENT_0001", "unused", status="NOT_QC_BEND")]
    )
    assert filtered["records"] == []
    assert filtered["predictions"] == []
    assert audit[0]["decision"] == "TERMINATE_NOT_QC_BEND"


def test_ontology_coverage_gap_fails_closed() -> None:
    parent = {
        "records": [{"event_id": "CAD_EVENT_0001", "accepted": True}],
        "predictions": [{**_line("CAD_EVENT_0001", 0.0)}],
    }
    with pytest.raises(ValueError, match="ontology coverage gap"):
        ontology_filter_parent(parent, [])
