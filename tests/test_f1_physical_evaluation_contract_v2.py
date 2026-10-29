import pytest

from scripts.f1_physical_evaluation_contract_v2 import (
    adjudicate_visual_cases,
    score_localized_cases,
)


def test_count_accuracy_is_distinct_from_localized_target_set_accuracy():
    report = score_localized_cases(
        [
            {
                "case_id": "P01",
                "gold_count": 2,
                "prediction_count": 2,
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 1,
            }
        ]
    )
    assert report["exact_count_accuracy"] == 1.0
    assert report["exact_localized_target_set_accuracy"] == 0.0
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5


def test_localized_target_set_requires_zero_fp_and_fn():
    report = score_localized_cases(
        [
            {
                "case_id": "P01",
                "gold_count": 2,
                "prediction_count": 2,
                "true_positive": 2,
                "false_positive": 0,
                "false_negative": 0,
            }
        ]
    )
    assert report["exact_count_accuracy"] == 1.0
    assert report["exact_localized_target_set_accuracy"] == 1.0


def test_inconsistent_matching_accounting_is_rejected():
    with pytest.raises(ValueError, match=r"TP \+ FN"):
        score_localized_cases(
            [
                {
                    "case_id": "P01",
                    "gold_count": 2,
                    "prediction_count": 1,
                    "true_positive": 1,
                    "false_positive": 0,
                    "false_negative": 0,
                }
            ]
        )


def _visual_row(*, ruling="coherent", marker_admissible=True):
    return {
        "case_id": "P01",
        "reviewer": "manual-auditor-1",
        "ruling": ruling,
        "artifact_checks": {"native_overlay": True, "axis_stack": True},
        "markers": [
            {"marker_id": "m1", "render_admissible": marker_admissible},
            {"marker_id": "m2", "render_admissible": True},
        ],
    }


def test_marker_rate_is_per_marker_and_does_not_use_gold_count():
    report = adjudicate_visual_cases(
        [_visual_row(marker_admissible=False)],
        expected_marker_ids={"P01": ["m1", "m2"]},
    )
    assert report["render_admissible_marker_rate"] == 0.5
    assert report["visual_evidence_complete"] is True
    assert report["visual_gate_passed"] is True


@pytest.mark.parametrize("ruling", ["geometric_failure", "render_failure"])
def test_failure_ruling_never_passes_visual_gate(ruling):
    report = adjudicate_visual_cases(
        [_visual_row(ruling=ruling)],
        expected_marker_ids={"P01": ["m1", "m2"]},
    )
    assert report["visual_evidence_complete"] is True
    assert report["all_cases_coherent"] is False
    assert report["visual_gate_passed"] is False


def test_missing_marker_adjudication_fails_completeness():
    row = _visual_row()
    row["markers"] = row["markers"][:1]
    report = adjudicate_visual_cases(
        [row],
        expected_marker_ids={"P01": ["m1", "m2"]},
    )
    assert report["marker_denominator"] == 0
    assert report["render_admissible_marker_rate"] is None
    assert report["visual_evidence_complete"] is False
    assert report["visual_gate_passed"] is False
