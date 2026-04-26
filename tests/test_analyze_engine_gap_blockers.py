from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_engine_gap_blockers import analyze_engine_gap_blockers


def test_analyze_engine_gap_blockers_filters_full_scans_and_builds_slice(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "part_key": "A",
                        "scan_name": "A_full.ply",
                        "scan_path": "/tmp/A_full.ply",
                        "expectation": {"state": "full", "expected_total_bends": 8, "expected_completed_bends": 8},
                        "metrics": {
                            "engine_gap_position_unknown_bends": 4,
                            "scan_limited_position_unknown_bends": 0,
                            "unknown_position_bends": 4,
                            "position_eligible_bends": 1,
                            "correspondence_decision_ready_bends": 8,
                            "release_decision": "HOLD",
                        },
                        "details": {
                            "scan_state": "full",
                            "scan_quality_status": "GOOD",
                            "dense_local_refinement_decision": "primary_retained",
                            "dense_local_engine_gap_bends_before": 4,
                            "dense_local_engine_gap_bends_after": 4,
                            "dense_local_unresolved_engine_gaps": 4,
                            "dense_local_position_signal_recoveries": 0,
                        },
                    },
                    {
                        "part_key": "B",
                        "scan_name": "B_partial.ply",
                        "scan_path": "/tmp/B_partial.ply",
                        "expectation": {"state": "partial", "expected_total_bends": 6, "expected_completed_bends": None},
                        "metrics": {
                            "engine_gap_position_unknown_bends": 7,
                            "scan_limited_position_unknown_bends": 0,
                            "unknown_position_bends": 7,
                            "position_eligible_bends": 0,
                            "correspondence_decision_ready_bends": 6,
                            "release_decision": "HOLD",
                        },
                        "details": {
                            "scan_state": "partial",
                            "scan_quality_status": "GOOD",
                        },
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    analysis = analyze_engine_gap_blockers(summary_path, top_n=3)

    assert analysis["engine_gap_scans_analyzed"] == 1
    assert analysis["scan_state_breakdown"] == {"full": 1}
    assert analysis["top_engine_gap_scans"][0]["part_key"] == "A"
    assert analysis["slice_entries"][0]["scan_path"] == "/tmp/A_full.ply"
    assert analysis["slice_entries"][0]["expectation"]["state"] == "full"
    assert analysis["unique_part_slice"] is True


def test_analyze_engine_gap_blockers_can_include_partials(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "part_key": "A",
                        "scan_name": "A_partial.ply",
                        "scan_path": "/tmp/A_partial.ply",
                        "expectation": {"state": "partial"},
                        "metrics": {
                            "engine_gap_position_unknown_bends": 2,
                            "unknown_position_bends": 2,
                        },
                        "details": {"scan_state": "partial", "scan_quality_status": "GOOD"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_engine_gap_blockers(summary_path, full_only=False, top_n=1)

    assert analysis["engine_gap_scans_analyzed"] == 1
    assert analysis["scan_state_breakdown"] == {"partial": 1}
    assert analysis["top_engine_gap_scans"][0]["scan_state"] == "partial"


def test_analyze_engine_gap_blockers_deduplicates_slice_by_part(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "part_key": "A",
                        "scan_name": "A_1.ply",
                        "scan_path": "/tmp/A_1.ply",
                        "expectation": {"state": "full"},
                        "metrics": {"engine_gap_position_unknown_bends": 4, "unknown_position_bends": 4},
                        "details": {"scan_state": "full", "scan_quality_status": "GOOD"},
                    },
                    {
                        "part_key": "A",
                        "scan_name": "A_2.ply",
                        "scan_path": "/tmp/A_2.ply",
                        "expectation": {"state": "full"},
                        "metrics": {"engine_gap_position_unknown_bends": 3, "unknown_position_bends": 3},
                        "details": {"scan_state": "full", "scan_quality_status": "GOOD"},
                    },
                    {
                        "part_key": "B",
                        "scan_name": "B_1.ply",
                        "scan_path": "/tmp/B_1.ply",
                        "expectation": {"state": "full"},
                        "metrics": {"engine_gap_position_unknown_bends": 2, "unknown_position_bends": 2},
                        "details": {"scan_state": "full", "scan_quality_status": "GOOD"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_engine_gap_blockers(summary_path, top_n=3)

    assert [entry["scan_path"] for entry in analysis["slice_entries"]] == ["/tmp/A_1.ply", "/tmp/B_1.ply"]
