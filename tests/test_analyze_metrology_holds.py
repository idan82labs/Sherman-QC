import json
from pathlib import Path

from scripts.analyze_metrology_holds import analyze_metrology_holds


def test_analyze_metrology_holds_reads_embedded_results(tmp_path: Path) -> None:
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
                                "metrology_state": "OUT_OF_TOL",
                                "correspondence_state": "CONFIDENT",
                                "assignment_source": "GLOBAL_DETECTION",
                                "measurement_method": "edge_fit",
                                "bend_form": "FOLDED",
                                "position_claim_eligible": True,
                                "metrology_claim_eligible": True,
                                "angle_deviation": 1.25,
                                "radius_deviation": 0.0,
                                "tolerance_angle": 1.0,
                                "tolerance_radius": 0.5,
                                "target_angle": 90.0,
                                "measured_angle": 91.25,
                                "claim_gate_reasons": [],
                            },
                            {
                                "bend_id": "CAD_B2",
                                "metrology_state": "OUT_OF_TOL",
                                "correspondence_state": "AMBIGUOUS",
                                "assignment_source": "GLOBAL_DETECTION",
                                "measurement_method": "edge_fit",
                                "bend_form": "FOLDED",
                                "position_claim_eligible": False,
                                "metrology_claim_eligible": False,
                                "angle_deviation": 2.0,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_metrology_holds(summary_path)

    assert analysis["out_of_tol_confident_bends"] == 1
    assert analysis["near_threshold_out_of_tol_bends"] == 1
    assert analysis["assignment_source_breakdown"] == {"GLOBAL_DETECTION": 1}
    assert analysis["measurement_method_breakdown"] == {"edge_fit": 1}
    assert analysis["position_readiness_breakdown"] == {"position_ready": 1}
    assert analysis["bend_form_breakdown"] == {"FOLDED": 1}
    assert analysis["failure_driver_breakdown"] == {"FOLDED:angle": 1}
    assert analysis["closest_out_of_tol_bends"][0]["bend_id"] == "CAD_B1"


def test_analyze_metrology_holds_falls_back_to_case_payload_files(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    part_dir = tmp_path / "P2"
    part_dir.mkdir()
    (part_dir / "BEFORE_BENDING_11979003_C.ply--abcd1234.json").write_text(
        json.dumps(
            {
                "part_key": "P2",
                "scan_name": "BEFORE BENDING 11979003_C.ply",
                "matches": [
                    {
                        "bend_id": "CAD_B3",
                        "metrology_state": "OUT_OF_TOL",
                        "correspondence_state": "CONFIDENT",
                        "assignment_source": "CAD_LOCAL_NEIGHBORHOOD",
                        "measurement_method": "local_fit",
                        "bend_form": "ROLLED",
                        "position_claim_eligible": False,
                        "metrology_claim_eligible": True,
                        "angle_deviation": -0.2,
                        "radius_deviation": 0.9,
                        "arc_length_deviation_mm": 1.1,
                        "tolerance_angle": 1.0,
                        "tolerance_radius": 0.5,
                        "target_angle": 135.0,
                        "measured_angle": 134.8,
                        "claim_gate_reasons": ["position_unknown"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"results": [{"part_key": "P2", "scan_name": "BEFORE BENDING 11979003_C.ply"}]}),
        encoding="utf-8",
    )

    analysis = analyze_metrology_holds(summary_path)

    assert analysis["out_of_tol_confident_bends"] == 1
    assert analysis["near_threshold_out_of_tol_bends"] == 1
    assert analysis["assignment_source_breakdown"] == {"CAD_LOCAL_NEIGHBORHOOD": 1}
    assert analysis["position_readiness_breakdown"] == {"position_unknown": 1}
    assert analysis["bend_form_breakdown"] == {"ROLLED": 1}
    assert analysis["failure_driver_breakdown"] == {"ROLLED:radius+arc": 1}
    assert analysis["closest_out_of_tol_bends"][0]["scan_name"] == "BEFORE BENDING 11979003_C.ply"
