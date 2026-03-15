from pathlib import Path

import bend_analysis_worker as worker


def test_promote_structured_count_reporting_adds_operator_fields(monkeypatch):
    report = {
        "part_id": "PART",
        "summary": {
            "completed_bends": 4,
            "expected_bends": 10,
            "remaining_bends": 6,
        },
        "operator_brief": {
            "headline": "4/10 completed, 2 in spec, 2 out of spec, 6 remaining",
        },
        "matches": [],
    }

    monkeypatch.setattr(worker, "_latest_unary_model_path", lambda: Path("/tmp/fake.joblib"))
    monkeypatch.setattr(worker, "load_unary_models", lambda _path: {"fake": True})
    monkeypatch.setattr(
        worker,
        "score_case_payload",
        lambda payload, _bundle: {
            **payload,
            "structured_context": {
                "count_posterior": {
                    "median_completed_bends": 6,
                    "map_completed_bends": 6,
                    "mean_completed_bends": 5.7,
                }
            },
        },
    )

    promoted = worker._promote_structured_count_reporting(report)

    assert promoted["summary"]["match_evidence_completed_bends"] == 4
    assert promoted["summary"]["structured_completed_bends"] == 6
    assert promoted["summary"]["structured_remaining_bends"] == 4
    assert promoted["summary"]["structured_count_delta_vs_match"] == 2
    assert "Count model: 6/10 complete" in promoted["operator_brief"]["headline"]
    assert "Direct evidence: 4 explicit bends" in promoted["operator_brief"]["headline"]
