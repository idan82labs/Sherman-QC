"""Candidate gates for Z3 (selection from the candidate stream). Rules are geometric and part-agnostic."""
from __future__ import annotations

from typing import Any


def g_simple(d: dict, sid: str) -> list[dict[str, Any]]:
    out = []
    for c in d.get("candidates", []):
        if c["length"] < 8.0 or c["n_support"] < 4 or c["support_frac"] < 0.5:
            continue
        if c.get("strip_frac", 0) < 0.5:
            continue
        if 180 - c["incl"] < 9.0:
            continue
        out.append(c)
    return out


def g_birth(d: dict, sid: str) -> list[dict[str, Any]]:
    """Liberal, unfitted birth filter: the Z3 candidate stream handed to selectors / integrators."""
    return [c for c in d.get("candidates", []) if c["length"] >= 5.0 and c["n_support"] >= 3
            and 180 - c["incl"] >= 8.0 and c["support_frac"] >= 0.4]


GATES = {"simple": g_simple, "birth": g_birth}
