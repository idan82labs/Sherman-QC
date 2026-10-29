from research.f1_cad_visibility_event_fusion_v1 import (
    VisibilityGates,
    filter_cad_events,
    late_fuse,
)


def record(event_id="E1", **overrides):
    row = {
        "event_id": event_id,
        "accepted": True,
        "scan_support": {
            "support_fraction": .9,
            "longest_support_run_fraction": .8,
            "supported_span_mm": 40.,
            "median_distance_mm": .8,
        },
        "normal_turn_verification": {"normal_turn_deg": 60., "minimum_side_fraction": .4},
    }
    row.update(overrides)
    return row


def prediction(event_id="E1", y=0.):
    return {"bend_id": event_id, "line_start": [0., y, 0.], "line_end": [40., y, 0.]}


def test_visibility_fails_closed_and_emits_explicit_null():
    parent = {
        "predictions": [prediction("E1"), prediction("E2", 10.)],
        "records": [record("E1"), record("E2", scan_support={
            "support_fraction": .4,
            "longest_support_run_fraction": .4,
            "supported_span_mm": 40.,
            "median_distance_mm": .8,
        })],
    }
    accepted, audit = filter_cad_events(parent, VisibilityGates())
    assert [row["bend_id"] for row in accepted] == ["E1"]
    assert audit[1]["decision"] == "NULL_UNOBSERVED_OR_WEAK"
    assert "CONTINUOUS_SUPPORT" in audit[1]["failed_gates"]


def test_scan_candidate_owns_duplicate_and_unique_cad_event_is_admitted():
    scan = [prediction("S1")]
    cad = [prediction("E1", .5), prediction("E2", 10.)]
    fused, audit = late_fuse(scan, cad, VisibilityGates())
    assert [row["assignment_source"] for row in fused] == [
        "SCAN_ONLY_A0_FIRST_OWNER",
        "CAD_VISIBILITY_VERIFIED_EVENT",
    ]
    assert audit[0]["decision"] == "NULL_DUPLICATE_SCAN_OR_PRIOR_EVENT"
    assert audit[1]["decision"] == "CAD_EVENT_ADMITTED"
