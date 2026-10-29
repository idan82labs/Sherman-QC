"""State vocabularies shared by results, jobs and the station feed.

Kept in their own module so ``line_job`` and ``inspection_result`` can both use
them without an import cycle.
"""
from __future__ import annotations

from typing import Literal

__all__ = [
    "BendState",
    "BEND_STATES",
    "FORMED_LIKE_STATES",
    "ValidityState",
    "VALIDITY_STATES",
    "SummaryState",
    "SUMMARY_STATES",
    "StageStatus",
    "STAGE_STATUSES",
]

BendState = Literal["FORMED", "NOT_YET_FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED", "UNKNOWN"]
BEND_STATES: frozenset[str] = frozenset(BendState.__args__)  # type: ignore[attr-defined]
#: States in which the bend has (at least partly) been formed.
FORMED_LIKE_STATES: frozenset[str] = frozenset({"FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED"})

ValidityState = Literal[
    "VALID",
    "INSUFFICIENT_CAPTURE",
    "UNTRUSTED_REGISTRATION",
    "UNIT_MISMATCH_SUSPECTED",
    "UNSUPPORTED_INPUT",
    "WRONG_PART_SUSPECTED",
    "MIRRORED_PART_SUSPECTED",
    "TIMEOUT",
]
VALIDITY_STATES: frozenset[str] = frozenset(ValidityState.__args__)  # type: ignore[attr-defined]

#: ALL_MEASURED_IN_TOLERANCE: every planned bend formed and every required
#: characteristic MEASURED and PASS. STAGE_IN_TOLERANCE: everything measured so
#: far passes but the part is not finished (IN_PROCESS only); never green.
SummaryState = Literal[
    "ALL_MEASURED_IN_TOLERANCE",
    "STAGE_IN_TOLERANCE",
    "OUT_OF_TOLERANCE_FOUND",
    "REVIEW_REQUIRED",
    "INCOMPLETE",
    "INVALID_INSPECTION",
]
SUMMARY_STATES: frozenset[str] = frozenset(SummaryState.__args__)  # type: ignore[attr-defined]

StageStatus = Literal[
    "CONSISTENT",
    "STAGE_MISMATCH",
    "OUT_OF_ORDER_SUSPECTED",
    "PARTIAL_BEND_FOUND",
    "FINAL_INCOMPLETE",
    "UNDETERMINED",
    "AMBIGUOUS_POSE",
    "NOT_EVALUATED",
]
STAGE_STATUSES: frozenset[str] = frozenset(StageStatus.__args__)  # type: ignore[attr-defined]
