"""Display vocabulary: legacy -> production terms, and colour/label tokens.

The survey-era contract (and the legacy worker) used IN/OUT/NOT_MEASURED for
values and FORMED/UNFORMED/UNKNOWN for observations. The production engine
uses PASS/FAIL/REVIEW/UNKNOWN (+ NOT_APPLICABLE) for decisions and the five
bend states. Every consumer (station UI, PDF, Jev export, evaluation) maps
through :data:`LEGACY_TO_PRODUCTION` and renders with :data:`DISPLAY_TOKENS`.

Colour rule: PASS green, FAIL red, **REVIEW amber**, UNKNOWN neutral grey,
NOT_YET_FORMED / NOT_APPLICABLE muted slate (greyed, never red or green).
STAGE_IN_TOLERANCE (an unfinished part whose measured values pass) is blue
"info", never green: only ALL_MEASURED_IN_TOLERANCE is green.
The frontend mirrors these tokens (C08); the parity check compares them.

UNKNOWN is labelled neutrally ("Undetermined"): it is the decision of a
NOT_MEASURED value *and* of a MEASURED value without nominal or tolerance, so
the UI appends the reason (e.g. "Undetermined - EDGE_NOT_SEEN").

The Hebrew labels have not yet been reviewed by a native speaker.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = ["LEGACY_TO_PRODUCTION", "DISPLAY_TOKENS", "to_production_term", "display_token"]

#: Legacy / survey term -> production term. Unmapped legacy terms raise.
LEGACY_TO_PRODUCTION: Mapping[str, str] = MappingProxyType(
    {
        # value conformance
        "IN": "PASS",
        "IN_TOLERANCE": "PASS",
        "OUT": "FAIL",
        "OUT_OF_TOLERANCE": "FAIL",
        "NOT_MEASURED": "UNKNOWN",
        "NOT_APPLICABLE": "NOT_APPLICABLE",
        # bend observation
        "UNFORMED": "NOT_YET_FORMED",
        "OBSERVED_NOT_FORMED": "NOT_YET_FORMED",
        "FORMED": "FORMED",
        "OBSERVED_FORMED": "FORMED",
        "UNKNOWN": "UNKNOWN",
    }
)

_TOKENS = {
    # decisions
    "PASS": {"tone": "pass", "color": "#16A34A", "en": "Pass", "he": "תקין"},
    "FAIL": {"tone": "fail", "color": "#DC2626", "en": "Fail", "he": "חריגה"},
    "REVIEW": {"tone": "review", "color": "#F59E0B", "en": "Review", "he": "לבדיקה"},
    "UNKNOWN": {"tone": "unknown", "color": "#6B7280", "en": "Undetermined", "he": "לא הוכרע"},
    "NOT_APPLICABLE": {"tone": "inactive", "color": "#94A3B8", "en": "Not applicable", "he": "לא רלוונטי"},
    # bend states
    "FORMED": {"tone": "neutral", "color": "#334155", "en": "Formed", "he": "כופף"},
    "NOT_YET_FORMED": {"tone": "inactive", "color": "#94A3B8", "en": "Not yet formed", "he": "טרם כופף"},
    "PARTIALLY_FORMED": {"tone": "review", "color": "#F59E0B", "en": "Partially formed", "he": "כופף חלקית"},
    "UNEXPECTED_FORMED": {"tone": "review", "color": "#F59E0B", "en": "Formed out of order", "he": "כופף שלא לפי הסדר"},
    # summary states
    "ALL_MEASURED_IN_TOLERANCE": {"tone": "pass", "color": "#16A34A", "en": "All measured in tolerance", "he": "כל המידות בטולרנס"},
    "STAGE_IN_TOLERANCE": {"tone": "info", "color": "#2563EB", "en": "Stage in tolerance (part not complete)", "he": "השלב בטולרנס (החלק טרם הושלם)"},
    "OUT_OF_TOLERANCE_FOUND": {"tone": "fail", "color": "#DC2626", "en": "Out of tolerance found", "he": "נמצאה חריגה"},
    "REVIEW_REQUIRED": {"tone": "review", "color": "#F59E0B", "en": "Review required", "he": "נדרשת בדיקה"},
    "INCOMPLETE": {"tone": "unknown", "color": "#6B7280", "en": "Incomplete", "he": "לא הושלם"},
    "INVALID_INSPECTION": {"tone": "unknown", "color": "#6B7280", "en": "Invalid inspection", "he": "בדיקה לא תקפה"},
}
#: term -> {tone, color (hex), en, he}. Read-only.
DISPLAY_TOKENS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {k: MappingProxyType(v) for k, v in _TOKENS.items()}
)


def to_production_term(term: str) -> str:
    """Map a legacy/survey term to the production vocabulary; production terms pass through."""
    if term in DISPLAY_TOKENS:
        return term
    try:
        return LEGACY_TO_PRODUCTION[term]
    except KeyError:
        raise ValueError(f"no production mapping for term {term!r}") from None


def display_token(term: str) -> Mapping[str, str]:
    return DISPLAY_TOKENS[to_production_term(term)]
