import json
from pathlib import Path

from scripts.analyze_position_holds import analyze_position_holds


def test_analyze_position_holds_counts_formed_confident_unknown_position(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "part_key": "P1",
                        "scan_name": "scan_a.ply",
                        "matches": [
                            {
                                "bend_id": "CAD_B1",
                                "countable_in_regression": True,
                                "completion_state": "FORMED",
                                "correspondence_state": "CONFIDENT",
                                "positional_state": "UNKNOWN_POSITION",
                                "bend_form": "ROLLED",
                                "assignment_source": "GLOBAL_DETECTION",
                                "measurement_method": "detected_bend_match",
                                "metrology_state": "OUT_OF_TOL",
                                "angle_deviation": 0.1,
                                "radius_deviation": 0.9,
                                "claim_gate_reasons": ["observability_partial", "position_signal_unavailable"],
                                "position_evidence": {"reason": "no_position_signal"},
                            },
                            {
                                "bend_id": "CAD_B2",
                                "countable_in_regression": True,
                                "completion_state": "FORMED",
                                "correspondence_state": "CONFIDENT",
                                "positional_state": "ON_POSITION",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_position_holds(summary_path)

    assert analysis["formed_confident_unknown_position_bends"] == 1
    assert analysis["assignment_source_breakdown"] == {"GLOBAL_DETECTION": 1}
    assert analysis["measurement_method_breakdown"] == {"detected_bend_match": 1}
    assert analysis["bend_form_breakdown"] == {"ROLLED": 1}
    assert analysis["claim_gate_reason_breakdown"] == {
        "observability_partial": 1,
        "position_signal_unavailable": 1,
    }
    assert analysis["sample_bends"][0]["position_evidence_reason"] == "no_position_signal"
