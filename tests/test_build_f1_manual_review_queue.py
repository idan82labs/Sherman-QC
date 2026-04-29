import importlib.util
from pathlib import Path


def _load_module(name, filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


queue = _load_module("build_f1_manual_review_queue", "build_f1_manual_review_queue.py")
summary = _load_module("summarize_f1_manual_validation", "summarize_f1_manual_validation.py")


def test_priority_marks_broad_partial_as_p1():
    review = {
        "track": "partial",
        "algorithm": {"accepted_bend_count_range": [4, 14]},
        "manual_review": {"required": True, "reason_codes": ["broad_partial_observed_count_range"]},
    }

    assert queue._priority_for_review(review) == 1


def test_priority_marks_count_sweep_promotion_as_p1():
    review = {
        "track": "known",
        "algorithm": {"accepted_bend_count_range": [4, 4]},
        "manual_review": {"required": True, "reason_codes": ["raw_f1_manual_promotion_candidate"]},
    }

    assert queue._priority_for_review(review) == 1


def test_labels_template_preserves_observed_prediction_and_contact_sheet():
    payload = queue._labels_template(
        [
            {
                "part_key": "demo",
                "scan_name": "demo.ply",
                "track": "partial",
                "range": [2, 2],
                "observed_prediction": 2,
                "exact": None,
                "known_total_context": 10,
                "review_queue_contact_sheet": "/tmp/demo.png",
            }
        ]
    )

    label = payload["labels"][0]
    assert label["algorithm_observed_prediction"] == 2
    assert label["human_full_part_bend_count"] == 10
    assert label["contact_sheet"] == "/tmp/demo.png"


def test_manual_validation_summary_scores_observed_partial_labels():
    result = summary.summarize_manual_validation(
        {
            "labels": [
                {
                    "track": "partial",
                    "part_key": "demo",
                    "scan_name": "demo.ply",
                    "algorithm_range": [2, 2],
                    "algorithm_observed_prediction": 2,
                    "human_observed_bend_count": 2,
                },
                {
                    "track": "partial",
                    "part_key": "wide",
                    "scan_name": "wide.ply",
                    "algorithm_range": [4, 14],
                    "algorithm_observed_prediction": None,
                    "human_observed_bend_count": 8,
                },
            ]
        }
    )

    assert result["pending_label_count"] == 0
    assert result["exact_accuracy"] == 1.0
    assert result["range_contains_truth_rate"] == 1.0
    assert result["by_track"]["partial"]["range_contains_truth_rate"] == 1.0


def test_manual_validation_summary_accepts_records_schema():
    result = summary.summarize_manual_validation(
        {
            "records": [
                {
                    "track": "known",
                    "part_key": "demo",
                    "scan_name": "demo.ply",
                    "algorithm_range": [4, 4],
                    "algorithm_exact": 4,
                    "human_full_part_bend_count": 4,
                }
            ]
        }
    )

    assert result["case_count"] == 1
    assert result["exact_accuracy"] == 1.0
    assert result["range_contains_truth_rate"] == 1.0


def test_manual_validation_summary_separates_clean_disputed_and_excluded_cases():
    result = summary.summarize_manual_validation(
        {
            "labels": [
                {
                    "track": "partial",
                    "part_key": "clean",
                    "scan_name": "clean.ply",
                    "algorithm_range": [4, 6],
                    "human_observed_bend_count": 5,
                },
                {
                    "track": "known",
                    "part_key": "rolled",
                    "scan_name": "rolled.ply",
                    "algorithm_range": [2, 3],
                    "human_observed_bend_count": 2,
                    "part_family": "rolled_or_rounded_mixed_transition",
                },
                {
                    "track": "known",
                    "part_key": "bad",
                    "scan_name": "bad.ply",
                    "algorithm_range": [9, 12],
                    "exclude_from_accuracy": True,
                    "exclusion_reason": "bad_scan_quality",
                },
            ]
        }
    )

    assert result["clean_accuracy"]["cases"] == 1
    assert result["clean_accuracy"]["range_contains_truth_rate"] == 1.0
    assert result["disputed_or_special_accuracy"]["cases"] == 1
    assert result["excluded_accuracy"]["cases"] == 1
    assert result["excluded_accuracy"]["pending"] == 1
