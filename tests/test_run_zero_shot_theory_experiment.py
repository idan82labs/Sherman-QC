import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_zero_shot_theory_experiment.py"
    spec = importlib.util.spec_from_file_location("run_zero_shot_theory_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


theory = _load_module()


def _summary_row(part_key: str, *, exact_count, count_range, profile_family="orthogonal_dense_sheetmetal"):
    return {
        "part_key": part_key,
        "result": {
            "estimated_bend_count": exact_count,
            "estimated_bend_count_range": list(count_range),
            "profile_family": profile_family,
        },
    }


def test_decide_experiment_rejects_when_hard_case_broadens():
    control_summaries = {
        "zero_shot_development": {"aggregate": {"exact_count_accuracy": 1.0, "count_range_contains_truth_rate": 1.0, "abstained_rate": 0.571}},
        "zero_shot_holdout": {"aggregate": {"count_range_contains_truth_rate": 1.0, "profile_family_match_rate": 1.0, "abstention_correctness_rate": 1.0, "abstained_rate": 0.8}},
        "known_count_plys": {"aggregate": {"count_range_contains_truth_rate": 1.0, "abstained_rate": 0.5}, "results": [
            _summary_row("10839000", exact_count=10, count_range=(10, 10)),
            _summary_row("47959001", exact_count=6, count_range=(6, 6)),
        ]},
        "repeated_near_90": {"aggregate": {"count_range_contains_truth_rate": 1.0, "dominant_angle_family_match_rate": 1.0, "abstained_rate": 0.333}, "results": [
            _summary_row("10839000", exact_count=10, count_range=(10, 10)),
            _summary_row("ysherman", exact_count=None, count_range=(10, 12)),
            _summary_row("49204020_04", exact_count=None, count_range=(11, 12), profile_family="rolled_or_mixed_transition"),
        ]},
        "small_tight_bends": {"aggregate": {"count_range_contains_truth_rate": 1.0, "abstained_rate": 0.8}},
        "partial_total_bends": {"aggregate": {"abstained_rate": 1.0}},
    }
    candidate_summaries = {
        **control_summaries,
        "repeated_near_90": {"aggregate": {"count_range_contains_truth_rate": 1.0, "dominant_angle_family_match_rate": 1.0, "abstained_rate": 0.333}, "results": [
            _summary_row("10839000", exact_count=10, count_range=(10, 10)),
            _summary_row("ysherman", exact_count=None, count_range=(9, 13)),
            _summary_row("49204020_04", exact_count=None, count_range=(11, 12), profile_family="rolled_or_mixed_transition"),
        ]},
    }
    decision = theory._decide_experiment(
        method_config={"hypothesis_statement": "x", "conceptual_change": "y"},
        control_summaries=control_summaries,
        candidate_summaries=candidate_summaries,
    )
    assert decision["decision"] == "reject"
    assert "ysherman broadened" in decision["regressions"]


def test_decide_experiment_keeps_when_hard_case_tightens_without_regression():
    control_summaries = {
        "zero_shot_development": {"aggregate": {"exact_count_accuracy": 1.0, "count_range_contains_truth_rate": 1.0, "abstained_rate": 0.571}},
        "zero_shot_holdout": {"aggregate": {"count_range_contains_truth_rate": 1.0, "profile_family_match_rate": 1.0, "abstention_correctness_rate": 1.0, "abstained_rate": 0.8}},
        "known_count_plys": {"aggregate": {"count_range_contains_truth_rate": 1.0, "abstained_rate": 0.5}, "results": [
            _summary_row("10839000", exact_count=10, count_range=(10, 10)),
            _summary_row("47959001", exact_count=6, count_range=(6, 6)),
            _summary_row("48963008", exact_count=None, count_range=(3, 4)),
            _summary_row("48991006", exact_count=None, count_range=(8, 12), profile_family="rolled_or_mixed_transition"),
        ]},
        "repeated_near_90": {"aggregate": {"count_range_contains_truth_rate": 1.0, "dominant_angle_family_match_rate": 1.0, "abstained_rate": 0.333}, "results": [
            _summary_row("10839000", exact_count=10, count_range=(10, 10)),
            _summary_row("ysherman", exact_count=None, count_range=(10, 12)),
            _summary_row("49204020_04", exact_count=None, count_range=(11, 12), profile_family="rolled_or_mixed_transition"),
        ]},
        "small_tight_bends": {"aggregate": {"count_range_contains_truth_rate": 1.0, "abstained_rate": 0.8}},
        "partial_total_bends": {"aggregate": {"abstained_rate": 1.0}},
    }
    candidate_summaries = {
        **control_summaries,
        "repeated_near_90": {"aggregate": {"count_range_contains_truth_rate": 1.0, "dominant_angle_family_match_rate": 1.0, "abstained_rate": 0.333}, "results": [
            _summary_row("10839000", exact_count=10, count_range=(10, 10)),
            _summary_row("ysherman", exact_count=None, count_range=(11, 12)),
            _summary_row("49204020_04", exact_count=None, count_range=(11, 12), profile_family="rolled_or_mixed_transition"),
        ]},
    }
    decision = theory._decide_experiment(
        method_config={"hypothesis_statement": "x", "conceptual_change": "y"},
        control_summaries=control_summaries,
        candidate_summaries=candidate_summaries,
    )
    assert decision["decision"] == "keep"
    assert "ysherman tightened" in decision["improvements"]
