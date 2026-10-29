from backend.f1_corridor_completion import (
    complete_scan_supported_corridors,
    corridor_completion_applicability,
)


def _hypothesis(
    bend_id,
    source_kind,
    anchor,
    span,
    *,
    axis=(1.0, 0.0, 0.0),
    atoms=12,
    pair=("F1", "F2"),
    confidence=1.0,
):
    return {
        "bend_id": bend_id,
        "source_kind": source_kind,
        "anchor": anchor,
        "axis_direction": axis,
        "span_mm": span,
        "cross_width_mm": 8.0,
        "confidence": confidence,
        "atom_ids": [f"{bend_id}_A{index}" for index in range(atoms)],
        "candidate_incident_flange_pairs": [pair],
    }


def test_corridor_completion_dedupes_strips_and_suppresses_covered_births():
    strips = [
        _hypothesis("B1", "zero_shot_strip_seed", (0.0, 0.0, 0.0), 80.0),
        _hypothesis("B1_DUP", "zero_shot_strip_seed", (2.0, 1.0, 0.0), 76.0),
    ]
    births = [
        _hypothesis("I1", "interface_birth", (-10.0, 2.0, 0.0), 20.0),
        _hypothesis("I2", "interface_birth", (10.0, 2.0, 0.0), 20.0),
    ]

    result = complete_scan_supported_corridors(
        strip_hypotheses=strips,
        interface_birth_hypotheses=births,
        local_spacing_mm=2.0,
    )

    assert result["selected_strip_count"] == 1
    assert result["selected_uncovered_birth_count"] == 0
    assert len(result["covered_birth_components"]) == 1


def test_corridor_completion_splits_uncovered_birth_line_only_at_observed_gap():
    strips = [_hypothesis("B1", "zero_shot_strip_seed", (0.0, 0.0, 0.0), 80.0)]
    births = [
        _hypothesis("I1", "interface_birth", (-45.0, 30.0, 0.0), 18.0, pair=("F3", "F4")),
        _hypothesis("I2", "interface_birth", (-27.0, 30.0, 0.0), 18.0, pair=("F3", "F5")),
        _hypothesis("I3", "interface_birth", (27.0, 30.0, 0.0), 18.0, pair=("F3", "F4")),
        _hypothesis("I4", "interface_birth", (45.0, 30.0, 0.0), 18.0, pair=("F3", "F5")),
    ]

    result = complete_scan_supported_corridors(
        strip_hypotheses=strips,
        interface_birth_hypotheses=births,
        local_spacing_mm=2.0,
    )

    assert result["selected_strip_count"] == 1
    assert result["selected_uncovered_birth_count"] == 2
    births_out = [item for item in result["selected"] if item["source_kind"] == "uncovered_interface_corridor"]
    assert [item["support_ids"] for item in births_out] == [["I1", "I2"], ["I3", "I4"]]


def test_corridor_completion_is_deterministic_and_has_no_target_count_input():
    strips = [
        _hypothesis("B2", "zero_shot_strip_seed", (0.0, 20.0, 0.0), 40.0),
        _hypothesis("B1", "zero_shot_strip_seed", (0.0, 0.0, 0.0), 40.0),
    ]
    births = [
        _hypothesis("I2", "interface_birth", (8.0, 40.0, 0.0), 16.0),
        _hypothesis("I1", "interface_birth", (-8.0, 40.0, 0.0), 16.0),
    ]
    forward = complete_scan_supported_corridors(
        strip_hypotheses=strips,
        interface_birth_hypotheses=births,
        local_spacing_mm=2.0,
    )
    reverse = complete_scan_supported_corridors(
        strip_hypotheses=list(reversed(strips)),
        interface_birth_hypotheses=list(reversed(births)),
        local_spacing_mm=2.0,
    )

    assert forward == reverse
    assert forward["selected_count"] == 3


def test_corridor_completion_gate_rejects_partial_and_nonfragmented_routes():
    eligible = corridor_completion_applicability(
        route_payload={
            "route_id": "fragmented_repeat90_interface_birth",
            "feature_snapshot": {
                "scan_state_hint": "",
                "repeated_near90_strength": 0.82,
                "candidate_fragmentation": 1.625,
            },
        },
        strip_hypothesis_count=8,
        interface_birth_hypothesis_count=24,
    )
    assert eligible == {"applicable": True, "reason_codes": []}

    partial = corridor_completion_applicability(
        route_payload={
            "route_id": "fragmented_repeat90_interface_birth",
            "feature_snapshot": {
                "scan_state_hint": "partial",
                "repeated_near90_strength": 0.82,
                "candidate_fragmentation": 1.625,
            },
        },
        strip_hypothesis_count=8,
        interface_birth_hypothesis_count=24,
    )
    assert partial["applicable"] is False
    assert "partial_scan_not_eligible" in partial["reason_codes"]

    stable = corridor_completion_applicability(
        route_payload={
            "route_id": "zero_shot_control_stable",
            "feature_snapshot": {
                "scan_state_hint": "",
                "repeated_near90_strength": 0.82,
                "candidate_fragmentation": 1.0,
            },
        },
        strip_hypothesis_count=12,
        interface_birth_hypothesis_count=0,
    )
    assert stable["applicable"] is False
    assert "route_not_fragmented_repeat90_interface_birth" in stable["reason_codes"]
