from __future__ import annotations

from typing import Optional


TERNARY_OBSERVABILITY_STATES = {"FORMED", "UNFORMED", "UNKNOWN"}
LEGACY_OBSERVABILITY_STATES = {
    "OBSERVED_FORMED",
    "OBSERVED_NOT_FORMED",
    "PARTIALLY_OBSERVED",
    "UNOBSERVED",
}


def normalize_observability_state(
    observability_state: Optional[str],
    *,
    physical_completion_state: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Normalize observation evidence, never a tolerance verdict or a state prior.

    The keyword arguments remain for call-site compatibility. In particular,
    PASS/WARNING/FAIL describes a measurement comparison, not whether both sides
    of a bend were observed. Explicit unknown/partial evidence is not completed
    by a physical-completion claim from another stage.
    """
    state = str(observability_state or "").strip().upper()
    if state in TERNARY_OBSERVABILITY_STATES:
        return state
    if state == "OBSERVED_FORMED":
        return "FORMED"
    if state == "OBSERVED_NOT_FORMED":
        return "UNFORMED"
    return "UNKNOWN"


def legacy_observability_detail_state(
    observability_state: Optional[str],
    *,
    physical_completion_state: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    state = str(observability_state or "").strip().upper()
    if state in LEGACY_OBSERVABILITY_STATES:
        return state

    ternary = normalize_observability_state(
        observability_state,
        physical_completion_state=physical_completion_state,
        status=status,
    )
    if ternary == "FORMED":
        return "OBSERVED_FORMED"
    if ternary == "UNFORMED":
        return "OBSERVED_NOT_FORMED"
    return "UNOBSERVED"


def is_explicit_observability_evidence(
    observability_state: Optional[str],
    *,
    physical_completion_state: Optional[str] = None,
    status: Optional[str] = None,
) -> bool:
    return normalize_observability_state(
        observability_state,
        physical_completion_state=physical_completion_state,
        status=status,
    ) in {"FORMED", "UNFORMED"}


def feature_family_for_target(
    *,
    feature_type: Optional[str],
    bend_form: Optional[str],
    countable_in_regression: bool,
) -> str:
    raw_feature_type = str(feature_type or "").strip().upper()
    bend_form_norm = str(bend_form or "").strip().upper()
    if countable_in_regression and raw_feature_type not in {"ROLLED_SECTION", "PROCESS_FEATURE"} and bend_form_norm != "ROLLED":
        return "COUNTABLE_DISCRETE_BEND"
    if raw_feature_type in {"ROLLED_SECTION", "PROCESS_FEATURE"} or bend_form_norm == "ROLLED":
        return "ROLLED_PROCESS_FEATURE"
    return "NON_BEND_TRANSITION"


def measurement_primitive_for_feature_family(feature_family: Optional[str]) -> str:
    family = str(feature_family or "").strip().upper()
    if family == "ROLLED_PROCESS_FEATURE":
        return "ROLLED_PROFILE"
    if family == "NON_BEND_TRANSITION":
        return "PROFILE"
    return "ANGLE_RADIUS"
