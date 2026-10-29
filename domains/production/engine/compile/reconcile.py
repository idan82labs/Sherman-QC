"""Reconcile operator-confirmed recipe marks with flange-graph bends (ENGINE_IO §2.1).

A recipe bend's ``line_start_mm``/``line_end_mm`` is its *identity* line (for
CAD candidates: the native mid-radius half-turn line). A (mark, graph bend)
pair is **feasible** when

- the mark is parallel to the cylinder axis within ``reconcile_parallel_deg`` (2 deg),
- its midpoint lies inside the bend region: within ``Ri + t + 1 mm`` of the
  cylinder axis, inside the bend's axial extent (+-1 mm) and inside the bend's
  angular sector (+-10 deg). The axial and angular conditions make "inside the
  bend region" precise for collinear tabs and coaxial returns.

The cost of a feasible pair is the distance (mm) from the mark midpoint to the
bend's mid-radius apex line plus 0.5 mm per ``reconcile_parallel_deg`` of
direction error, so an exact native mark costs ~0. A mark (or a graph bend)
whose runner-up feasible partner is within ``reconcile_ambiguity_margin_mm``
of its best is **ambiguous** and is never auto-assigned (MARK_AMBIGUOUS). The
remaining feasible pairs are assigned one-to-one with the Hungarian algorithm.

Outcomes: MARK_NOT_IN_FLANGE_GRAPH (a mark is feasible for no graph bend),
UNMARKED_BEND (a *supported* graph bend has no mark; listed in
``unmarked_graph_bends``) and MARK_AMBIGUOUS, all BLOCKERs. MARK_AMBIGUOUS
covers a close runner-up on either side and a **duplicate**: a mark that is
feasible only for graph bends another mark already matched (the graph bend is
"matched by several marks", whatever their cost gap).

An *unsupported* graph bend without a mark is not a blocker: it can never be a
recipe bend and becomes a rigid, uninspected feature, formed in every stage
(see :mod:`.unsupported`). That is wrong for a hem formed mid-sequence: the
operator should mark it, and a marked unsupported bend is staged like any
recipe bend with every characteristic UNKNOWN(UNSUPPORTED_FEATURE).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.geometry import signed_angle_about, unit
from domains.production.engine.compile.part_model import ModelBend, PartModel

__all__ = ["Mark", "Reconciliation", "reconcile_marks", "pair_metrics"]

_AXIAL_MARGIN_MM = 1.0
_ANGULAR_MARGIN_DEG = 10.0
_INFEASIBLE = 1e9


@dataclass(frozen=True)
class Mark:
    """A recipe/draft bend mark: identity uid plus its confirmed line (cad_model_mm)."""

    bend_uid: str
    line_start: tuple[float, float, float]
    line_end: tuple[float, float, float]


@dataclass
class Reconciliation:
    matches: dict[str, int] = field(default_factory=dict)  # bend_uid -> graph bend key
    costs: dict[str, float] = field(default_factory=dict)
    unmatched_marks: list[str] = field(default_factory=list)
    ambiguous_marks: list[tuple[str, list[str]]] = field(default_factory=list)  # uid, graph ids
    ambiguous_graph: list[tuple[int, list[str]]] = field(default_factory=list)  # graph key, uids
    unmarked_supported: list[int] = field(default_factory=list)
    unmarked_unsupported: list[int] = field(default_factory=list)
    candidates: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return bool(self.unmatched_marks or self.ambiguous_marks or self.ambiguous_graph or self.unmarked_supported)


def pair_metrics(bend: ModelBend, mark: Mark, cfg: CompileConfig) -> tuple[bool, float, dict]:
    """Feasibility and cost of matching ``mark`` to ``bend`` (module docstring)."""
    a = np.asarray(mark.line_start, dtype=np.float64)
    b = np.asarray(mark.line_end, dtype=np.float64)
    d = b - a
    if float(np.linalg.norm(d)) <= 1e-9:
        return False, _INFEASIBLE, {"reason": "degenerate mark"}
    d = unit(d)
    angle = math.degrees(math.acos(min(1.0, abs(float(d @ bend.axis_dir)))))
    mid = 0.5 * (a + b)
    rel = mid - bend.axis_point
    z = float(rel @ bend.axis_dir)
    radial_vec = rel - z * bend.axis_dir
    radial = float(np.linalg.norm(radial_vec))
    info = {"angle_deg": angle, "radial_mm": radial, "axial_mm": z}
    if angle > cfg.reconcile_parallel_deg:
        return False, _INFEASIBLE, {**info, "reason": "not parallel"}
    if radial > bend.ri + bend.thickness + cfg.reconcile_radial_margin_mm:
        return False, _INFEASIBLE, {**info, "reason": "outside the bend radius"}
    if not (bend.z0 - _AXIAL_MARGIN_MM <= z <= bend.z1 + _AXIAL_MARGIN_MM):
        return False, _INFEASIBLE, {**info, "reason": "outside the bend's axial extent"}
    if radial > 1e-6:
        phi = math.degrees(signed_angle_about(bend.r_parent, unit(radial_vec), bend.axis_dir))
        if phi < -_ANGULAR_MARGIN_DEG and bend.sweep_deg > 180.0 - _ANGULAR_MARGIN_DEG:
            phi += 360.0  # a reflex (teardrop) sector extends past 180 deg from the parent end
        if not (-_ANGULAR_MARGIN_DEG <= phi <= bend.sweep_deg + _ANGULAR_MARGIN_DEG):
            return False, _INFEASIBLE, {**info, "phi_deg": phi, "reason": "outside the bend's angular sector"}
        info["phi_deg"] = phi
    m = bend.bisector()
    expected = bend.axis_point + z * bend.axis_dir + (bend.ri + bend.thickness / 2.0) * m
    cost = float(np.linalg.norm(mid - expected)) + 0.5 * angle / cfg.reconcile_parallel_deg
    return True, cost, {**info, "cost": cost}


def reconcile_marks(model: PartModel, marks: Sequence[Mark], cfg: CompileConfig) -> Reconciliation:
    """Match recipe marks to graph bends one-to-one (module docstring)."""
    from scipy.optimize import linear_sum_assignment

    out = Reconciliation()
    gkeys = sorted(model.bends, key=lambda k: model.bends[k].graph_bend_id)
    feas: dict[tuple[int, int], float] = {}
    for i, mark in enumerate(marks):
        cands = []
        for j, gk in enumerate(gkeys):
            ok, cost, _ = pair_metrics(model.bends[gk], mark, cfg)
            if ok:
                feas[(i, j)] = cost
                cands.append((model.bends[gk].graph_bend_id, round(cost, 6)))
        out.candidates[mark.bend_uid] = sorted(cands, key=lambda c: (c[1], c[0]))

    margin = cfg.reconcile_ambiguity_margin_mm
    amb_marks: set[int] = set()
    amb_graph: set[int] = set()
    for i, mark in enumerate(marks):
        costs = sorted((c, j) for (a, j), c in feas.items() if a == i)
        if len(costs) >= 2 and costs[1][0] - costs[0][0] <= margin:
            amb_marks.add(i)
            out.ambiguous_marks.append((mark.bend_uid, [model.bends[gkeys[j]].graph_bend_id for _, j in costs]))
    for j, gk in enumerate(gkeys):
        costs = sorted((c, i) for (i, b), c in feas.items() if b == j and i not in amb_marks)
        if len(costs) >= 2 and costs[1][0] - costs[0][0] <= margin:
            amb_graph.add(j)
            out.ambiguous_graph.append((gk, sorted(marks[i].bend_uid for _, i in costs)))

    clear = {k: v for k, v in feas.items() if k[0] not in amb_marks and k[1] not in amb_graph}
    assigned_marks: set[int] = set()
    assigned_graph: set[int] = set()
    graph_owner: dict[int, int] = {}  # graph column -> assigned mark row
    if clear:
        rows = sorted({i for i, _ in clear})
        cols = sorted({j for _, j in clear})
        matrix = np.full((len(rows), len(cols)), _INFEASIBLE)
        for (i, j), c in clear.items():
            matrix[rows.index(i), cols.index(j)] = c
        r_idx, c_idx = linear_sum_assignment(matrix)
        for r, c in zip(r_idx, c_idx):
            i, j = rows[r], cols[c]
            if (i, j) in clear:
                out.matches[marks[i].bend_uid] = gkeys[j]
                out.costs[marks[i].bend_uid] = clear[(i, j)]
                assigned_marks.add(i)
                assigned_graph.add(j)
                graph_owner[j] = i
    duplicates: dict[int, list[str]] = {}
    for i, mark in enumerate(marks):
        if i in assigned_marks or i in amb_marks:
            continue
        feasible = sorted(j for (a, j) in feas if a == i)
        if not feasible:
            out.unmatched_marks.append(mark.bend_uid)
            continue
        # feasible but unassigned: every graph bend it fits already has a mark (a duplicate)
        for j in feasible:
            duplicates.setdefault(j, []).append(mark.bend_uid)
    for j, uids in sorted(duplicates.items()):
        holder = [marks[graph_owner[j]].bend_uid] if j in graph_owner else []
        entry = next((e for e in out.ambiguous_graph if e[0] == gkeys[j]), None)
        if entry is None:
            out.ambiguous_graph.append((gkeys[j], sorted(set(holder + uids))))
            amb_graph.add(j)
        else:
            entry[1][:] = sorted(set(entry[1]) | set(uids))
    involved_graph = assigned_graph | amb_graph | {j for (i, j) in feas if i in amb_marks}
    for j, gk in enumerate(gkeys):
        if j in involved_graph:
            continue
        if model.bends[gk].unsupported is None:
            out.unmarked_supported.append(gk)
        else:
            out.unmarked_unsupported.append(gk)
    return out
