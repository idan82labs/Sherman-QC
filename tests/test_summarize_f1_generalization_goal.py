from __future__ import annotations

from scripts.summarize_f1_generalization_goal import evaluate_generalization_goal


def _payloads() -> dict:
    return {
        "score": {
            "corpus_freeze_id": "freeze",
            "metrics": {
                "scan_count": 30,
                "exact_scans": 27,
                "precision": 0.90,
                "recall": 0.90,
            },
            "part_family_metrics": {"family_count": 15, "minimum_family_recall": 0.80},
        },
        "operations": {
            "deterministic_repeat_rate": 1.0,
            "maximum_solver_child_peak_rss_bytes": 1024,
            "p95_latency_sec": 300.0,
            "job_success_rate": 0.995,
            "global_working_point_budget": 200_000,
        },
        "visual_review": [
            {"true_positive": 1, "false_positive": 0} for _ in range(30)
        ],
        "split_audit": {"passed": True, "inference_metadata_exposure": False},
        "pool_manifest": {
            "working_point_budget": 200_000,
            "source_point_serialization": False,
        },
    }


def test_goal_readiness_passes_only_with_all_thresholds() -> None:
    report = evaluate_generalization_goal(**_payloads())

    assert report["goal_achieved"] is True
    assert report["failed_gate_count"] == 0
    assert report["runtime_promotion_authorized"] is False


def test_goal_readiness_fails_semantic_family_and_integrity_regressions() -> None:
    payloads = _payloads()
    payloads["score"]["metrics"]["recall"] = 0.89
    payloads["score"]["part_family_metrics"]["minimum_family_recall"] = 0.79
    payloads["pool_manifest"]["source_point_serialization"] = True

    report = evaluate_generalization_goal(**payloads)

    assert report["goal_achieved"] is False
    assert set(report["failed_gates"]) == {
        "semantic.line_recall",
        "generalization.minimum_family_recall",
        "integrity.no_source_point_serialization",
    }


def test_goal_readiness_requires_visual_review_of_every_scan() -> None:
    payloads = _payloads()
    payloads["visual_review"].pop()

    report = evaluate_generalization_goal(**payloads)
    gate = next(
        row
        for row in report["gates"]
        if row["gate_id"] == "visual.render_admissible_marker_accuracy"
    )

    assert gate["passed"] is False
    assert gate["observed"]["reviewed_scans"] == 29
