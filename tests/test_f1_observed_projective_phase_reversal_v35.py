from __future__ import annotations

import math

from backend.f1_observed_projective_phase_reversal_v35 import (
    direct_transition_phase_certificate,
    projective_delta,
)


def visual(phases: list[float], transition: list[int] | None = None) -> dict[str, object]:
    return {
        "free_phases": phases,
        "phase_path_strips": list(range(len(phases))),
        "transition_strips": transition if transition is not None else list(range(1, len(phases) - 1)),
    }


def test_projective_delta_is_wrap_invariant() -> None:
    assert math.isclose(projective_delta(math.radians(85), math.radians(-85)), math.radians(10))
    assert math.isclose(projective_delta(math.radians(-85), math.radians(85)), math.radians(-10))


def test_monotone_decreasing_transition_passes() -> None:
    certificate = direct_transition_phase_certificate(
        visual([math.radians(value) for value in [60, 60, 55, 40, 20, 0, 0]])
    )
    assert certificate["passed"] is True
    assert certificate["positive_step_count"] == 0
    assert certificate["negative_step_count"] > 0


def test_monotone_transition_through_projective_wrap_passes() -> None:
    certificate = direct_transition_phase_certificate(
        visual([math.radians(value) for value in [70, 80, 89, -82, -70, -60]])
    )
    assert certificate["passed"] is True
    assert certificate["positive_step_count"] > 0
    assert certificate["negative_step_count"] == 0


def test_nonmonotone_bead_reversal_fails() -> None:
    certificate = direct_transition_phase_certificate(
        visual([math.radians(value) for value in [0, 25, 70, 35, 60, 90]])
    )
    assert certificate["passed"] is False
    assert certificate["positive_step_count"] > 0
    assert certificate["negative_step_count"] > 0
    assert certificate["direction_reversal_count"] >= 2


def test_subresolution_jitter_does_not_create_reversal() -> None:
    certificate = direct_transition_phase_certificate(
        visual([0.0, 0.001, -0.001, 0.0, math.radians(20)])
    )
    assert certificate["passed"] is True
    assert certificate["meaningful_step_count"] == 1
