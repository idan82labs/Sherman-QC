"""Stable bend identity and draft-to-draft bend remapping.

``bend_uid`` (CONTRACTS §0)::

    "bu_" + sha256("<cad_sha256>|<source>|<endpoints>|<radius>")[:20]

- endpoints: both line endpoints rounded to 1e-3 mm (``-0.000`` folded to
  ``0.000``), sorted lexicographically so the uid is invariant to endpoint
  order, formatted ``x,y,z;x,y,z`` with three decimals;
- radius: rounded to 1e-3 mm, or ``none``.

The same CAD bytes and the same confirmed line give the same uid across draft
edits; a new CAD file is a new recipe version with new uids. Rounding makes
the uid insensitive to sub-micron jitter except exactly at a rounding
boundary; :func:`remap_bends` is the robust way to carry identities between
drafts or CAD revisions.

:func:`remap_bends` matches previous recipe bends to new marks on feasible
pairs only (direction, midpoint and length within tolerances) with the
Hungarian algorithm. Pairs whose best and runner-up costs are too close
(``best >= ambiguity_ratio * second``) are **never** auto-assigned: they are
reported as ambiguous groups for the operator.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BEND_UID_SOURCES",
    "bend_uid",
    "bend_uid_for_mark",
    "RemapMatch",
    "AmbiguousGroup",
    "RemapResult",
    "remap_bends",
]

BEND_UID_SOURCES = frozenset({"cad_candidate", "manual"})


def _q(value: float) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("bend geometry must be finite numbers")
    r = round(float(value), 3)
    if r == 0.0:
        r = 0.0
    return f"{r:.3f}"


def _point(p: Sequence[float]) -> tuple[str, str, str]:
    if len(p) != 3:
        raise ValueError("bend line endpoints must be 3D points")
    return (_q(p[0]), _q(p[1]), _q(p[2]))


def bend_uid(
    cad_sha256: str,
    source: str,
    line_start: Sequence[float],
    line_end: Sequence[float],
    radius: float | None,
) -> str:
    """Stable ``bu_<20 hex>`` identity of a bend line on a given CAD file."""
    if not isinstance(cad_sha256, str) or len(cad_sha256) != 64 or any(c not in "0123456789abcdef" for c in cad_sha256):
        raise ValueError("cad_sha256 must be 64 lowercase hex characters")
    if source not in BEND_UID_SOURCES:
        raise ValueError(f"source must be one of {sorted(BEND_UID_SOURCES)}")
    a, b = sorted((_point(line_start), _point(line_end)), key=lambda t: tuple(float(v) for v in t))
    if a == b:
        raise ValueError("bend line endpoints coincide at 1e-3 mm")
    radius_text = "none" if radius is None else _q(radius)
    payload = f"{cad_sha256}|{source}|{','.join(a)};{','.join(b)}|{radius_text}"
    return "bu_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def bend_uid_for_mark(cad_sha256: str, mark: Mapping[str, Any]) -> str:
    """``bend_uid`` of a Vendor Setup draft mark (keys source/line_start/line_end/target_radius)."""
    if mark.get("line_start") is None or mark.get("line_end") is None:
        raise ValueError("mark has no line")
    return bend_uid(cad_sha256, mark["source"], mark["line_start"], mark["line_end"], mark.get("target_radius"))


@dataclass(frozen=True)
class RemapMatch:
    previous_uid: str
    feature_id: str
    cost: float
    midpoint_distance_mm: float
    direction_angle_deg: float
    length_difference_mm: float


@dataclass(frozen=True)
class AmbiguousGroup:
    previous_uids: tuple[str, ...]
    feature_ids: tuple[str, ...]


@dataclass(frozen=True)
class RemapResult:
    matched: tuple[RemapMatch, ...]
    new: tuple[str, ...]
    removed: tuple[str, ...]
    ambiguous: tuple[AmbiguousGroup, ...]


def _get(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, Mapping):
            if name in item and item[name] is not None:
                return item[name]
        elif getattr(item, name, None) is not None:
            return getattr(item, name)
    return None


def _line(item: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    start = _get(item, "line_start_mm", "line_start")
    end = _get(item, "line_end_mm", "line_end")
    if start is None or end is None:
        raise ValueError("every bend/mark passed to remap_bends needs a line")
    s = tuple(float(v) for v in start)
    e = tuple(float(v) for v in end)
    if len(s) != 3 or len(e) != 3 or not all(math.isfinite(v) for v in s + e) or s == e:
        raise ValueError("bend lines must be finite, 3D and non-degenerate")
    return s, e  # type: ignore[return-value]


def _pair_metrics(a, b) -> tuple[float, float, float]:
    (a0, a1), (b0, b1) = a, b
    da = [y - x for x, y in zip(a0, a1)]
    db = [y - x for x, y in zip(b0, b1)]
    la, lb = math.sqrt(sum(v * v for v in da)), math.sqrt(sum(v * v for v in db))
    cos = abs(sum(x * y for x, y in zip(da, db))) / (la * lb)
    angle = math.degrees(math.acos(min(1.0, cos)))
    ma = [(x + y) / 2 for x, y in zip(a0, a1)]
    mb = [(x + y) / 2 for x, y in zip(b0, b1)]
    return math.dist(ma, mb), angle, abs(la - lb)


def remap_bends(
    previous_bends: Sequence[Any],
    new_marks: Sequence[Any],
    dir_tol_deg: float = 1.0,
    mid_tol_mm: float = 0.5,
    len_tol_mm: float = 1.0,
    ambiguity_ratio: float = 0.8,
) -> RemapResult:
    """Carry identities from ``previous_bends`` (``bend_uid`` + line) to
    ``new_marks`` (``feature_id`` + line). See the module docstring."""
    if min(dir_tol_deg, mid_tol_mm, len_tol_mm) <= 0 or not 0 < ambiguity_ratio <= 1:
        raise ValueError("tolerances must be positive and 0 < ambiguity_ratio <= 1")
    raw_prev = [_get(b, "bend_uid") for b in previous_bends]
    raw_new = [_get(m, "feature_id") for m in new_marks]
    if any(v is None for v in raw_prev) or any(v is None for v in raw_new):
        raise ValueError("previous bends need bend_uid and new marks need feature_id")
    prev_ids = [str(v) for v in raw_prev]
    new_ids = [str(v) for v in raw_new]
    if len(set(prev_ids)) != len(prev_ids) or len(set(new_ids)) != len(new_ids):
        raise ValueError("ids must be unique")
    prev_lines = [_line(b) for b in previous_bends]
    new_lines = [_line(m) for m in new_marks]

    edges: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for i, pl in enumerate(prev_lines):
        for j, nl in enumerate(new_lines):
            mid, ang, dlen = _pair_metrics(pl, nl)
            if mid <= mid_tol_mm and ang <= dir_tol_deg and dlen <= len_tol_mm:
                cost = mid / mid_tol_mm + ang / dir_tol_deg + dlen / len_tol_mm
                edges[(i, j)] = (cost, mid, ang, dlen)

    def ambiguous(costs: list[float]) -> bool:
        if len(costs) < 2:
            return False
        best, second = sorted(costs)[:2]
        return best >= ambiguity_ratio * second

    amb_prev = {i for i in range(len(prev_ids)) if ambiguous([c[0] for (a, _), c in edges.items() if a == i])}
    amb_new = {j for j in range(len(new_ids)) if ambiguous([c[0] for (_, b), c in edges.items() if b == j])}

    clear = {k: v for k, v in edges.items() if k[0] not in amb_prev and k[1] not in amb_new}
    matched: list[RemapMatch] = []
    matched_prev: set[int] = set()
    matched_new: set[int] = set()
    if clear:
        from scipy.optimize import linear_sum_assignment  # local: keep module import light

        rows = sorted({i for i, _ in clear})
        cols = sorted({j for _, j in clear})
        big = 1e9
        matrix = [[clear.get((i, j), (big,))[0] for j in cols] for i in rows]
        r_idx, c_idx = linear_sum_assignment(matrix)
        for r, c in zip(r_idx, c_idx):
            i, j = rows[r], cols[c]
            if (i, j) in clear:
                cost, mid, ang, dlen = clear[(i, j)]
                matched.append(RemapMatch(prev_ids[i], new_ids[j], cost, mid, ang, dlen))
                matched_prev.add(i)
                matched_new.add(j)

    # Unmatched items that could have taken an ambiguous partner join the
    # ambiguity (closure over feasible edges); they are neither new nor removed.
    changed = True
    while changed:
        changed = False
        for (i, j) in edges:
            if j in amb_new and i not in matched_prev and i not in amb_prev:
                amb_prev.add(i)
                changed = True
            if i in amb_prev and j not in matched_new and j not in amb_new:
                amb_new.add(j)
                changed = True

    # Connected components of the ambiguous nodes over feasible edges.
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in amb_prev:
        find(("p", i))
    for j in amb_new:
        find(("n", j))
    for (i, j) in edges:
        if i in amb_prev and j in amb_new:
            parent[find(("p", i))] = find(("n", j))
    comps: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(parent):
        comps.setdefault(find(node), []).append(node)
    groups = sorted(
        (
            AmbiguousGroup(
                tuple(sorted(prev_ids[i] for kind, i in nodes if kind == "p")),
                tuple(sorted(new_ids[j] for kind, j in nodes if kind == "n")),
            )
            for nodes in comps.values()
        ),
        key=lambda g: (g.previous_uids, g.feature_ids),
    )

    removed = tuple(prev_ids[i] for i in range(len(prev_ids)) if i not in matched_prev and i not in amb_prev)
    new = tuple(new_ids[j] for j in range(len(new_ids)) if j not in matched_new and j not in amb_new)
    matched.sort(key=lambda m: (m.previous_uid, m.feature_id))
    return RemapResult(tuple(matched), new, removed, tuple(groups))
