import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_f1_merge_readiness.py"
    spec = importlib.util.spec_from_file_location("summarize_f1_merge_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


readiness = _load_module()


def _record(part, truth, count_range, exact=None, excluded=False):
    return {
        "part_key": part,
        "scan_name": f"{part}.ply",
        "track": "known",
        "human_observed_bend_count": truth,
        "algorithm_range": count_range,
        "algorithm_exact": exact,
        "algorithm_observed_prediction": None,
        "exclude_from_accuracy": excluded,
    }


def test_manual_metrics_counts_range_and_exact_hits():
    report = readiness.summarize_f1_merge_readiness(
        manual_validation={
            "records": [
                _record("exact", 4, [4, 4], exact=4),
                _record("range", 5, [4, 6]),
                _record("miss", 3, [4, 6]),
                _record("excluded", 7, [1, 2], excluded=True),
            ]
        },
        raw_crease_review={"cases": [], "recommendation_counts": {}},
    )

    manual = report["manual_validation_metrics"]
    assert manual["scored_case_count"] == 3
    assert manual["range_hit_count"] == 2
    assert manual["range_hit_rate"] == 0.6667
    assert manual["exact_hit_count"] == 1
    assert manual["exact_hit_rate"] == 1.0
    assert manual["range_miss_cases"][0]["part_key"] == "miss"


def test_merge_decision_allows_only_offline_diagnostics_when_raw_candidate_exists():
    report = readiness.summarize_f1_merge_readiness(
        manual_validation={"records": [_record("clean", 4, [4, 4], exact=4)]},
        raw_crease_review={
            "recommendation_counts": {"candidate_for_sparse_arrangement_review": 1},
            "cases": [
                {
                    "part_id": "clean",
                    "target_count": 4,
                    "final_offline_recommendation": "candidate_for_sparse_arrangement_review",
                    "selected_family_count": 4,
                    "safe_candidate_count": 6,
                    "final_recommendation_reasons": ["sparse_arrangement_count_matches_target"],
                }
            ],
        },
    )

    decision = report["merge_decision"]
    assert decision["main_branch_runtime_promotion"] == "not_ready"
    assert decision["offline_pr_merge_readiness"] == "safe_as_diagnostic_artifacts"
    assert "sparse_arrangement_candidate_exists_for_offline_review" in decision["allowed_next_steps"]


def test_raw_hold_cases_are_reported_as_blockers():
    report = readiness.summarize_f1_merge_readiness(
        manual_validation={"records": [_record("partial", 5, [1, 6])]},
        raw_crease_review={
            "recommendation_counts": {"hold_exact_count_scan_quality_limited": 1},
            "cases": [
                {
                    "part_id": "partial",
                    "target_count": 5,
                    "final_offline_recommendation": "hold_exact_count_scan_quality_limited",
                    "quality_decision": "limited_observed_support",
                    "sparse_status": "sparse_arrangement_underfit",
                    "final_recommendation_reasons": ["scan_support_below_target"],
                }
            ],
        },
    )

    assert "raw_crease_lane_has_hold_or_abstain_cases" in report["merge_decision"]["blockers"]
    assert report["raw_crease_review_gates"]["held_or_abstained"][0]["part_id"] == "partial"


def test_markdown_renders_practical_merge_path():
    report = readiness.summarize_f1_merge_readiness(
        manual_validation={"records": [_record("clean", 4, [4, 4], exact=4)]},
        raw_crease_review={"cases": [], "recommendation_counts": {}},
    )

    markdown = readiness.render_markdown(report)

    assert "F1 Merge Readiness Status" in markdown
    assert "Do not route production counts through sparse raw creases yet." in markdown
