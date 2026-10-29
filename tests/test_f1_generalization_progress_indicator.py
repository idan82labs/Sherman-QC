from scripts.f1_generalization_progress_indicator import (
    GPIEvidence,
    GRADE_A,
    GRADE_B,
    GRADE_C,
    GRADE_D,
    GRADE_F,
    grade,
)


def _clean(**overrides):
    values = {
        "version": "VTEST",
        "protocol_valid": True,
        "leakage_free": True,
        "physical_scans": 10,
        "physical_families": 5,
        "untouched_family_holdout": True,
        "independent_line_truth": True,
        "matched_frozen_baseline": True,
        "f1_delta": 0.05,
        "exact_count_accuracy_delta": 0.02,
        "minimum_family_recall_delta": 0.01,
        "visual_gate_passed": True,
        "deterministic_replay_passed": True,
        "resource_gate_passed": True,
    }
    values.update(overrides)
    return GPIEvidence(**values)


def test_terminal_source_free_failure_is_f_not_negative_physical_evidence():
    report = grade(GPIEvidence(version="V73", protocol_valid=False, leakage_free=True))
    assert report["grade"] == GRADE_F
    assert report["direction"] == "NON_EVALUABLE"


def test_synthetic_or_fixture_only_success_is_c():
    report = grade(GPIEvidence(version="V52", protocol_valid=True, leakage_free=True))
    assert report["grade"] == GRADE_C


def test_clean_underpowered_physical_win_is_b():
    assert grade(_clean())["grade"] == GRADE_B


def test_clean_visual_failure_is_d_even_if_f1_rises():
    report = grade(_clean(visual_gate_passed=False))
    assert report["grade"] == GRADE_D


def test_minimum_family_regression_is_d_even_if_aggregate_f1_rises():
    report = grade(_clean(minimum_family_recall_delta=-0.01))
    assert report["grade"] == GRADE_D


def test_full_scale_all_gate_win_is_a():
    report = grade(
        _clean(
            physical_scans=30,
            physical_families=15,
            all_target_gates_passed=True,
        )
    )
    assert report["grade"] == GRADE_A


def test_same_count_but_no_f1_gain_is_c_not_progress():
    report = grade(_clean(f1_delta=0.0))
    assert report["grade"] == GRADE_C
