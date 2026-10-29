"""Characteristic vocabulary, identities and the guard-band decision rule.

A *characteristic* is one toleranced quantity the recipe asks the engine to
check: a flange length per side, an included angle, a bend-line position, a
mid-sequence backgauge length, or an operator-tagged general measurement.

Decision rule (plan §3.2 step 6), with deviation ``d = actual - expected``,
expanded uncertainty ``U > 0`` and tolerance band ``[-minus, +plus]``:

- PASS   iff ``d + U <= plus``  and ``d - U >= -minus``  (symmetric: ``|d| + U <= tol``)
- FAIL   iff ``d - U >  plus``  or  ``d + U <  -minus``  (symmetric: ``|d| - U > tol``)
- REVIEW otherwise (the uncertainty interval straddles a tolerance limit).

UNKNOWN is never produced by this rule: it is reserved for characteristics
that were not measured (or have no nominal/tolerance) and never becomes PASS
or FAIL. The nominal is never copied into ``actual``.
"""
from __future__ import annotations

from typing import Literal

__all__ = [
    "CharacteristicKind",
    "CHARACTERISTIC_KINDS",
    "BEND_CHARACTERISTIC_ORDER",
    "FlangeSide",
    "FLANGE_SIDES",
    "MeasurementState",
    "MEASUREMENT_STATES",
    "Decision",
    "DECISIONS",
    "GuardBandDecision",
    "UnknownReason",
    "UNKNOWN_REASONS",
    "PLAN_UNKNOWN_REASONS",
    "NotApplicableReason",
    "NOT_APPLICABLE_REASONS",
    "BoundaryKind",
    "BOUNDARY_KINDS",
    "ExpectedSource",
    "EXPECTED_SOURCES",
    "unit_for_kind",
    "make_characteristic_id",
    "guard_band_decision",
    "exceedance_ratio",
    "GUARD_BAND_EPS",
]

CharacteristicKind = Literal[
    "FLANGE_LENGTH",
    "INCLUDED_ANGLE",
    "BEND_LINE_POSITION",
    "BACKGAUGE_LENGTH",
    "GENERAL_MEASUREMENT",
]
CHARACTERISTIC_KINDS: frozenset[str] = frozenset(CharacteristicKind.__args__)  # type: ignore[attr-defined]

#: Fixed per-bend order: flange first (A then B), then angle, then position,
#: then the optional backgauge length.
BEND_CHARACTERISTIC_ORDER: tuple[tuple[str, str | None], ...] = (
    ("FLANGE_LENGTH", "A"),
    ("FLANGE_LENGTH", "B"),
    ("INCLUDED_ANGLE", None),
    ("BEND_LINE_POSITION", None),
)

FlangeSide = Literal["A", "B"]
FLANGE_SIDES: frozenset[str] = frozenset(FlangeSide.__args__)  # type: ignore[attr-defined]

MeasurementState = Literal["MEASURED", "NOT_MEASURED", "NOT_APPLICABLE"]
MEASUREMENT_STATES: frozenset[str] = frozenset(MeasurementState.__args__)  # type: ignore[attr-defined]

Decision = Literal["PASS", "FAIL", "REVIEW", "UNKNOWN", "NOT_APPLICABLE"]
DECISIONS: frozenset[str] = frozenset(Decision.__args__)  # type: ignore[attr-defined]
GuardBandDecision = Literal["PASS", "FAIL", "REVIEW"]

UnknownReason = Literal[
    # plan §3.2 step 6
    "OCCLUDED",
    "EDGE_NOT_SEEN",
    "LOW_DENSITY",
    "AMBIGUOUS_SEGMENTATION",
    "REGISTRATION_UNTRUSTED",
    "AMBIGUOUS_POSE",
    "TIMEOUT",
    "UNSUPPORTED_FEATURE",
    # contract additions
    "INVALID_INSPECTION",
    "BEND_STATE_UNKNOWN",
    "ADJACENT_BEND_NOT_MEASURED",
    "NOMINAL_UNAVAILABLE",
    "NO_TOLERANCE",
]
UNKNOWN_REASONS: frozenset[str] = frozenset(UnknownReason.__args__)  # type: ignore[attr-defined]
PLAN_UNKNOWN_REASONS: frozenset[str] = frozenset(
    {
        "OCCLUDED",
        "EDGE_NOT_SEEN",
        "LOW_DENSITY",
        "AMBIGUOUS_SEGMENTATION",
        "REGISTRATION_UNTRUSTED",
        "AMBIGUOUS_POSE",
        "TIMEOUT",
        "UNSUPPORTED_FEATURE",
    }
)

NotApplicableReason = Literal[
    "BEND_NOT_YET_FORMED",
    "NOT_INSPECTED_BY_RECIPE",
    "NOT_APPLICABLE_AT_STAGE",
]
NOT_APPLICABLE_REASONS: frozenset[str] = frozenset(NotApplicableReason.__args__)  # type: ignore[attr-defined]

BoundaryKind = Literal["FREE_EDGE", "ADJACENT_BEND"]
BOUNDARY_KINDS: frozenset[str] = frozenset(BoundaryKind.__args__)  # type: ignore[attr-defined]

ExpectedSource = Literal[
    "COMPILED_NOMINAL",  # CompiledPart final-stage nominal
    "STAGE_MODEL_NOMINAL",  # CompiledPart stage S_k nominal (backgauge)
    "RECIPE_TARGET",  # operator target from the recipe (general measurement)
    "CAD_DISTANCE",  # CAD anchor distance (general measurement)
    "NOMINAL_LOCUS_REFERENCE",  # bend-line position: offset from the nominal OML locus, expected 0
]
EXPECTED_SOURCES: frozenset[str] = frozenset(ExpectedSource.__args__)  # type: ignore[attr-defined]

#: Absolute slack for float comparisons in the guard-band rule.
GUARD_BAND_EPS = 1e-9


def unit_for_kind(kind: str) -> Literal["mm", "deg"]:
    if kind not in CHARACTERISTIC_KINDS:
        raise ValueError(f"unknown characteristic kind: {kind!r}")
    return "deg" if kind == "INCLUDED_ANGLE" else "mm"


def make_characteristic_id(
    kind: str,
    *,
    bend_uid: str | None = None,
    side: str | None = None,
    stage_index: int | None = None,
    measurement_id: str | None = None,
) -> str:
    """Deterministic characteristic id shared by CompiledPart and results.

    - ``<bend_uid>/FLANGE_LENGTH/<A|B>``
    - ``<bend_uid>/INCLUDED_ANGLE``
    - ``<bend_uid>/BEND_LINE_POSITION``
    - ``<bend_uid>/BACKGAUGE_LENGTH/S<k>``
    - ``gm/<measurement_id>``
    """
    if kind == "GENERAL_MEASUREMENT":
        if not measurement_id or bend_uid is not None or side is not None or stage_index is not None:
            raise ValueError("GENERAL_MEASUREMENT ids take only measurement_id")
        return f"gm/{measurement_id}"
    if kind not in CHARACTERISTIC_KINDS or not bend_uid or measurement_id is not None:
        raise ValueError(f"bend characteristic {kind!r} needs a bend_uid and no measurement_id")
    if kind == "FLANGE_LENGTH":
        if side not in FLANGE_SIDES or stage_index is not None:
            raise ValueError("FLANGE_LENGTH needs side A|B and no stage_index")
        return f"{bend_uid}/FLANGE_LENGTH/{side}"
    if kind == "BACKGAUGE_LENGTH":
        if stage_index is None or stage_index < 0:
            raise ValueError("BACKGAUGE_LENGTH needs a stage_index >= 0")
        return f"{bend_uid}/BACKGAUGE_LENGTH/S{stage_index}"
    if stage_index is not None:
        raise ValueError(f"{kind} takes no stage_index")
    return f"{bend_uid}/{kind}"


def guard_band_decision(deviation: float, uncertainty: float, plus: float, minus: float) -> GuardBandDecision:
    """Apply the guard-band rule (module docstring). ``uncertainty`` must be > 0."""
    if not (uncertainty > 0):
        raise ValueError("uncertainty must be strictly positive for a measured characteristic")
    if plus < 0 or minus < 0 or not (plus + minus > 0):
        raise ValueError("invalid tolerance band")
    if deviation + uncertainty <= plus + GUARD_BAND_EPS and deviation - uncertainty >= -minus - GUARD_BAND_EPS:
        return "PASS"
    if deviation - uncertainty > plus + GUARD_BAND_EPS or deviation + uncertainty < -minus - GUARD_BAND_EPS:
        return "FAIL"
    return "REVIEW"


def exceedance_ratio(deviation: float, uncertainty: float, plus: float, minus: float) -> float | None:
    """``(|d| + U) / tol_side`` where ``tol_side`` is the limit on the deviation's side.

    Returns ``None`` when that side's limit is zero (a unilateral band violated
    on its closed side); callers rank ``None`` as the worst possible value.
    """
    side = plus if deviation >= 0 else minus
    if side <= 0:
        return None
    return (abs(deviation) + uncertainty) / side
