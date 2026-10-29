from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.build_f1_v2_cross_parent_opposite_fragment_rung_probe_bundle import (
    LOCK_CRITERION_KEYS,
    PRE_GOLD_DECISION_VALUES,
    build_lock_attestation,
    decide_pre_gold,
    initialize_lock_criteria,
    validate_lock_attestation,
)


@pytest.mark.parametrize(
    ("integrity", "added", "reached", "advanced", "novel", "expected"),
    [
        (False, 1, 1, 1, 1, "INVALID_EXPERIMENT_STOP"),
        (True, 0, 0, 0, 0, "STOP_NO_ELIGIBLE_CROSS_PARENT_RUNGS"),
        (True, 1, 0, 0, 0, "STOP_RUNGS_DO_NOT_REACH_FROZEN_TARGETS"),
        (True, 1, 1, 0, 0, "STOP_OPPOSITE_SIDE_SUPPORT_NOT_ADVANCED"),
        (True, 1, 1, 1, 0, "STOP_NO_NOVEL_TARGET_RUNG_CERTIFICATE"),
        (True, 1, 1, 1, 1, "GO_NEXT_PROFESSOR_PACKET"),
    ],
)
def test_exact_ordered_pre_gold_decision_values(
    integrity, added, reached, advanced, novel, expected
) -> None:
    assert PRE_GOLD_DECISION_VALUES == (
        "INVALID_EXPERIMENT_STOP",
        "STOP_NO_ELIGIBLE_CROSS_PARENT_RUNGS",
        "STOP_RUNGS_DO_NOT_REACH_FROZEN_TARGETS",
        "STOP_OPPOSITE_SIDE_SUPPORT_NOT_ADVANCED",
        "STOP_NO_NOVEL_TARGET_RUNG_CERTIFICATE",
        "GO_NEXT_PROFESSOR_PACKET",
    )
    assert decide_pre_gold(
        integrity_valid=integrity,
        aggregate_added_cross_parent_rung_count=added,
        target_reached_by_added_rung_count=reached,
        target_side_gate_advanced_count=advanced,
        target_novel_rung_attributable_certificate_count=novel,
    ) == expected


def test_even_go_lock_never_authorizes_gold() -> None:
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    lock = build_lock_attestation(
        criteria,
        decision_metrics={
            "aggregate_added_cross_parent_rung_count": 1,
            "target_reached_by_added_rung_count": 1,
            "target_side_gate_advanced_count": 1,
            "target_novel_rung_attributable_certificate_count": 1,
        },
    )
    validate_lock_attestation(lock)
    assert lock["pre_gold_decision"] == "GO_NEXT_PROFESSOR_PACKET"
    assert lock["next_professor_packet_authorized"] is True
    assert lock["gold_open_authorized"] is False


def test_future_empirical_lock_criteria_default_to_false() -> None:
    criteria = initialize_lock_criteria()
    assert set(criteria) == set(LOCK_CRITERION_KEYS)
    assert criteria and all(value is False for value in criteria.values())


def test_missing_or_extra_lock_criterion_forces_invalid_experiment() -> None:
    criteria = {key: True for key in LOCK_CRITERION_KEYS}
    for changed in (
        {key: value for key, value in criteria.items() if key != "every_frozen_target_mapping_unique"},
        {**criteria, "not_preregistered": True},
    ):
        lock = build_lock_attestation(
            changed,
            decision_metrics={
                "aggregate_added_cross_parent_rung_count": 1,
                "target_reached_by_added_rung_count": 1,
                "target_side_gate_advanced_count": 1,
                "target_novel_rung_attributable_certificate_count": 1,
            },
        )
        validate_lock_attestation(lock)
        assert lock["integrity_valid"] is False
        assert lock["pre_gold_decision"] == "INVALID_EXPERIMENT_STOP"
        assert lock["gold_open_authorized"] is False


def test_lock_validator_rejects_any_gold_authorization() -> None:
    lock = build_lock_attestation(
        {key: True for key in LOCK_CRITERION_KEYS},
        decision_metrics={
            "aggregate_added_cross_parent_rung_count": 1,
            "target_reached_by_added_rung_count": 1,
            "target_side_gate_advanced_count": 1,
            "target_novel_rung_attributable_certificate_count": 1,
        },
    )
    changed = deepcopy(lock)
    changed["gold_open_authorized"] = True
    with pytest.raises(ValueError, match="never authorize gold"):
        validate_lock_attestation(changed)
