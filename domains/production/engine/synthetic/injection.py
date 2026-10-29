"""Scenarios and injected errors: nominal design -> as-built part.

An :class:`AsBuiltPart` is what the virtual scanner sees. It is derived from
the nominal :class:`~.geometry.PartDesign` by a :class:`Scenario` (which bends
are formed, i.e. the stage) and a list of :class:`Injection` records:

==========================  ==============================================================
Kind                        Effect (all exact, truth follows analytically)
==========================  ==============================================================
``PROCESS_VARIATION``       known-good scatter: every formed bend gets an included-angle
                            delta ~ N(0, ``angle_sigma_deg``) and a bend-line shift ~
                            N(0, ``shift_sigma_mm``); every non-datum *cut line* (the
                            collinear free edges of one straight cut, e.g. a far edge
                            split by a notch) that does not touch a bend attachment
                            edge gets one cut offset ~ N(0, ``cut_sigma_mm``)
``ANGLE``                   ``delta_included_deg`` on one bend (+ = more open)
``SPRINGBACK_GROUP``        the same ``delta_included_deg`` on every bend of one מהלך group
``FLANGE_LENGTH``           cut error ``delta_mm`` on the far end of one side: every free
                            edge parallel to the bend in the side's far half (the far
                            edge and any notch bottom) moves by ``delta_mm``, so the
                            whole far boundary shifts (never on the datum, whose edges
                            register the part)
``BEND_LINE_SHIFT``         backgauge error: the bend line moves ``delta_mm`` towards side
                            B (side A grows, side B shrinks; the blank is conserved)
``PARTIAL_BEND``            one bend formed to ``fraction`` (0.3..0.7) of its sweep
``ORDER_SWAP``              stage ``k`` with sequence bend ``i <= k`` replaced by ``j > k``
``MISSING_BEND``            a FINAL part with one bend left flat
``EXTRA_BEND``              an unplanned crease splitting a free flange (``fraction`` of
                            its length, ``sweep_deg``, ``direction``)
``MIRRORED``                the physical part is the mirror image of the design
``WRONG_SKU``               a different part (same family, other parameters) is scanned
==========================  ==============================================================

Bend states follow the physical model of :mod:`.geometry`: the blank and its
holes are fixed, a bend's arc consumes ``sweep_act (Ri + K t)`` of its strip,
the datum panel's cut features stay at their nominal CAD position (the truth
frame). Partial forming keeps the bend centred on its line.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .geometry import (
    FREE,
    BendDesign,
    BendState,
    FormState,
    GeometryError,
    PanelDesign,
    PartDesign,
    neutral_length,
)

__all__ = [
    "INJECTION_KINDS",
    "Injection",
    "Scenario",
    "AsBuiltPart",
    "apply_scenario",
    "order_consistent",
    "out_of_order_formed",
    "free_far_edge",
    "far_parallel_edges",
    "cut_lines",
    "side_panels",
    "PARTIAL_FRACTION_RANGE",
]

INJECTION_KINDS = frozenset(
    {
        "PROCESS_VARIATION",
        "ANGLE",
        "SPRINGBACK_GROUP",
        "FLANGE_LENGTH",
        "BEND_LINE_SHIFT",
        "PARTIAL_BEND",
        "ORDER_SWAP",
        "MISSING_BEND",
        "EXTRA_BEND",
        "MIRRORED",
        "WRONG_SKU",
    }
)
PARTIAL_FRACTION_RANGE = (0.3, 0.7)


@dataclass(frozen=True)
class Injection:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in INJECTION_KINDS:
            raise ValueError(f"unknown injection kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}


@dataclass(frozen=True)
class Scenario:
    """Which bends are formed: ``stage`` = number of sequence-prefix bends formed
    (``None`` = all, the complete part). Injections may change the formed set."""

    name: str
    stage: int | None
    inspection_stage: str  # FINAL | IN_PROCESS (recipe policy for this case)
    injections: tuple[Injection, ...] = ()


@dataclass
class AsBuiltPart:
    """The physical part of a case."""

    design: PartDesign  # may differ from nominal (extra bend, wrong SKU)
    state: FormState
    datum: str
    mirror: np.ndarray | None
    formed_nominal: tuple[str, ...]  # nominal bend keys with sweep > 0, in sequence order
    partial: dict[str, float]  # bend key -> fraction
    extra_bends: tuple[str, ...]
    wrong_part: bool
    split_panel: tuple[str, str] | None = None  # (inner key, outer key) of an EXTRA_BEND split
    notes: list[str] = field(default_factory=list)


def _ranks(design: PartDesign) -> dict[str, int]:
    """Order rank: all bends of one מהלך group share a rank (ENGINE_IO §4.2)."""
    rank: dict[str, int] = {}
    r = 0
    i = 0
    seq = list(design.sequence)
    group_of = {k: g for g in design.groups for k in g}
    while i < len(seq):
        g = group_of.get(seq[i])
        if g:
            for k in g:
                rank[k] = r
            i += len(g)
        else:
            rank[seq[i]] = r
            i += 1
        r += 1
    return rank


def order_consistent(design: PartDesign, formed: set[str]) -> bool:
    """No unformed bend has a strictly lower rank than a formed one (bends of one
    מהלך group share a rank, so order inside a group is free)."""
    rank = _ranks(design)
    unformed = [rank[k] for k in design.sequence if k not in formed]
    formed_ranks = [rank[k] for k in design.sequence if k in formed]
    if not unformed or not formed_ranks:
        return True
    return max(formed_ranks) <= min(unformed)


def out_of_order_formed(design: PartDesign, formed: set[str]) -> tuple[str, ...]:
    """Formed bends whose rank is above the lowest unformed rank (would be UNEXPECTED_FORMED)."""
    rank = _ranks(design)
    unformed = [rank[k] for k in design.sequence if k not in formed]
    if not unformed:
        return ()
    lowest = min(unformed)
    return tuple(k for k in design.sequence if k in formed and rank[k] > lowest)


def free_far_edge(design: PartDesign, panel_key: str, bend_key: str) -> int | None:
    """Index of the free edge of ``panel_key`` opposite (and parallel) to its
    attachment edge for ``bend_key``, when exactly one such edge exists."""
    p = design.panel(panel_key)
    b = design.bend(bend_key)
    m = np.asarray(b.normal_ab, dtype=float)
    pts = np.asarray(p.outline, dtype=float)
    n = len(pts)
    i_att = p.edge_index(bend_key)
    att_mid = 0.5 * (pts[i_att] + pts[(i_att + 1) % n])
    best = None
    best_dist = -1.0
    for i, tag in enumerate(p.edge_tags):
        if tag != FREE:
            continue
        e = pts[(i + 1) % n] - pts[i]
        if abs(float(e @ m)) > 1e-9 * np.linalg.norm(e):
            continue  # not parallel to the bend
        mid = 0.5 * (pts[i] + pts[(i + 1) % n])
        dist = abs(float((mid - att_mid) @ m))
        if dist > best_dist + 1e-9:
            best, best_dist = i, dist
    return best


def far_parallel_edges(design: PartDesign, panel_key: str, bend_key: str) -> tuple[int, ...]:
    """Free edges of ``panel_key`` parallel to ``bend_key`` in the panel's far half
    (distance from the attachment edge >= half the farthest one), excluding edges
    that touch a bend attachment edge: the far end a flange-length cut error moves."""
    p = design.panel(panel_key)
    b = design.bend(bend_key)
    m = np.asarray(b.normal_ab, dtype=float)
    pts = np.asarray(p.outline, dtype=float)
    n = len(pts)
    i_att = p.edge_index(bend_key)
    att_mid = 0.5 * (pts[i_att] + pts[(i_att + 1) % n])
    touching = p.edges_touching_attachments()
    cand: list[tuple[int, float]] = []
    for i, tag in enumerate(p.edge_tags):
        if tag != FREE or i in touching:
            continue
        e = pts[(i + 1) % n] - pts[i]
        if abs(float(e @ m)) > 1e-9 * np.linalg.norm(e):
            continue
        mid = 0.5 * (pts[i] + pts[(i + 1) % n])
        cand.append((i, abs(float((mid - att_mid) @ m))))
    if not cand:
        return ()
    far = max(dist for _, dist in cand)
    return tuple(i for i, dist in cand if dist >= 0.5 * far - 1e-9)


def cut_lines(panel: PanelDesign) -> list[tuple[int, ...]]:
    """The panel's free edges that do not touch a bend attachment edge, grouped by
    supporting line (collinear edges belong to one straight cut), ordered by their
    first edge index."""
    pts = np.asarray(panel.outline, dtype=float)
    n = len(pts)
    touching = panel.edges_touching_attachments()
    groups: list[tuple[np.ndarray, float, list[int]]] = []
    for i, tag in enumerate(panel.edge_tags):
        if tag != FREE or i in touching:
            continue
        e = pts[(i + 1) % n] - pts[i]
        d = e / np.linalg.norm(e)
        normal = np.array([d[1], -d[0]])  # outward for a CCW outline
        offset = float(normal @ pts[i])
        for g_normal, g_offset, members in groups:
            if float(g_normal @ normal) > 1.0 - 1e-12 and abs(g_offset - offset) < 1e-9:
                members.append(i)
                break
        else:
            groups.append((normal, offset, [i]))
    return [tuple(members) for _, _, members in groups]


def _split_for_extra_bend(
    design: PartDesign, panel_key: str, parent_bend: str, fraction: float, sweep_deg: float, direction: str
) -> tuple[PartDesign, str, str]:
    """Split a rectangular free flange with a new bend parallel to ``parent_bend``."""
    p = design.panel(panel_key)
    b = design.bend(parent_bend)
    pts = np.asarray(p.outline, dtype=float)
    if len(pts) != 4:
        raise GeometryError("extra bends split rectangular flanges only")
    m = np.asarray(b.normal_ab, dtype=float) * (1.0 if panel_key == b.panel_b else -1.0)  # into the flange
    d = np.asarray(b.line_dir, dtype=float)
    i_att = p.edge_index(parent_bend)
    base = pts[i_att]
    u = (pts - base) @ m
    length = float(u.max())
    ri = b.inner_radius
    ba = neutral_length(sweep_deg, ri, design.thickness, design.k_factor)
    cut = fraction * length
    lo, hi = cut - ba / 2.0, cut + ba / 2.0
    if lo < 3.0 or length - hi < 3.0:
        raise GeometryError("flange too short for an extra bend")
    for h in p.holes:
        hu = float((np.asarray(h.center) - base) @ m)
        if lo - h.radius - 1.0 < hu < hi + h.radius + 1.0:
            raise GeometryError("extra bend would cross a hole")
    v = (pts - base) @ d
    v0, v1 = float(v.min()), float(v.max())
    key_x = "BNDX1"
    outer_key = f"{panel_key}X"

    def P(uu: float, vv: float) -> tuple[float, float]:
        q = base + uu * m + vv * d
        return (float(q[0]), float(q[1]))

    # orientation: CCW requires (m, d) or (d, m) ordering; build then fix
    def rect(u0: float, u1: float, tag_low: str, tag_high: str) -> tuple[tuple, tuple]:
        poly = [P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)]
        tags = [FREE, tag_high, FREE, tag_low]
        area = 0.5 * sum(poly[k][0] * poly[(k + 1) % 4][1] - poly[(k + 1) % 4][0] * poly[k][1] for k in range(4))
        if area < 0:
            poly = [poly[0], poly[3], poly[2], poly[1]]
            tags = [tag_low, FREE, tag_high, FREE]
        return tuple(poly), tuple(tags)

    inner_poly, inner_tags = rect(0.0, lo, parent_bend, key_x)
    outer_poly, outer_tags = rect(hi, length, key_x, FREE)
    inner_holes = tuple(h for h in p.holes if float((np.asarray(h.center) - base) @ m) < lo)
    outer_holes = tuple(h for h in p.holes if float((np.asarray(h.center) - base) @ m) > hi)
    inner = PanelDesign(panel_key, inner_poly, inner_tags, inner_holes, p.role)
    outer = PanelDesign(outer_key, outer_poly, outer_tags, outer_holes, p.role + "_extra")
    lp = base + cut * m + v0 * d
    # line parameterised from v0
    new_bend = BendDesign(
        key=key_x,
        panel_a=panel_key,
        panel_b=outer_key,
        line_point=(float(lp[0]), float(lp[1])),
        line_dir=(float(d[0]), float(d[1])),
        normal_ab=(float(m[0]), float(m[1])),
        extent=(0.0, v1 - v0),
        direction=direction,
        inner_radius=ri,
        sweep_deg=sweep_deg,
    )
    panels = tuple(inner if q.key == panel_key else q for q in design.panels) + (outer,)
    new = replace(design, panels=panels, bends=design.bends + (new_bend,), sequence=design.sequence + (key_x,))
    new.validate()
    return new, key_x, outer_key


def apply_scenario(
    design: PartDesign,
    scenario: Scenario,
    rng: np.random.Generator,
    *,
    wrong_part_design: PartDesign | None = None,
) -> AsBuiltPart:
    """Build the as-built part for ``scenario`` (see module docstring)."""
    datum = design.default_datum()
    seq = list(design.sequence)
    n = len(seq)
    k = n if scenario.stage is None else int(scenario.stage)
    if not 0 <= k <= n:
        raise GeometryError(f"stage {k} out of range 0..{n}")
    formed = list(seq[:k])
    partial: dict[str, float] = {}
    delta_incl: dict[str, float] = {b.key: 0.0 for b in design.bends}
    shift: dict[str, float] = {b.key: 0.0 for b in design.bends}
    cuts: dict[tuple[str, int], float] = {}
    mirror = None
    wrong = False
    notes: list[str] = []
    extra: list[tuple[float, float, str]] = []
    for inj in scenario.injections:
        p = inj.params
        if inj.kind == "ORDER_SWAP":
            i, j = int(p["formed_out"]), int(p["formed_in"])
            if not (1 <= i <= k < j <= n):
                raise GeometryError("ORDER_SWAP needs 1 <= formed_out <= stage < formed_in <= N")
            formed = [b for b in formed if b != seq[i - 1]] + [seq[j - 1]]
        elif inj.kind == "MISSING_BEND":
            key = seq[int(p["sequence_index"]) - 1]
            formed = [b for b in formed if b != key]
        elif inj.kind == "PARTIAL_BEND":
            key = seq[int(p["sequence_index"]) - 1]
            frac = float(p["fraction"])
            lo, hi = PARTIAL_FRACTION_RANGE
            if not lo <= frac <= hi:
                raise GeometryError(f"partial forming fraction must be in [{lo}, {hi}]")
            if key not in formed:
                formed.append(key)
            partial[key] = frac
        elif inj.kind == "EXTRA_BEND":
            extra.append((float(p["fraction"]), float(p["sweep_deg"]), str(p["direction"])))
        elif inj.kind == "MIRRORED":
            mirror = True
        elif inj.kind == "WRONG_SKU":
            wrong = True
    formed_set = set(formed)
    for inj in scenario.injections:
        p = inj.params
        if inj.kind == "PROCESS_VARIATION":
            sa, ss, sc = float(p.get("angle_sigma_deg", 0.15)), float(p.get("shift_sigma_mm", 0.03)), float(p.get("cut_sigma_mm", 0.03))
            for b in design.bends:
                delta_incl[b.key] += float(rng.normal(0.0, sa))
                shift[b.key] += float(rng.normal(0.0, ss))
            for panel in design.panels:
                if panel.key == datum:
                    continue
                for line in cut_lines(panel):
                    off = float(rng.normal(0.0, sc))
                    for ei in line:
                        cuts[(panel.key, ei)] = cuts.get((panel.key, ei), 0.0) + off
        elif inj.kind == "ANGLE":
            delta_incl[seq[int(p["sequence_index"]) - 1]] += float(p["delta_included_deg"])
        elif inj.kind == "SPRINGBACK_GROUP":
            g = design.groups[int(p["group_index"]) - 1]
            for key in g:
                delta_incl[key] += float(p["delta_included_deg"])
        elif inj.kind == "BEND_LINE_SHIFT":
            key = seq[int(p["sequence_index"]) - 1]
            shift[key] += _toward_b_sign(design, key, datum) * float(p["delta_mm"])
        elif inj.kind == "FLANGE_LENGTH":
            key = seq[int(p["sequence_index"]) - 1]
            panel_key = str(p["panel"])
            if panel_key == datum:
                raise GeometryError("FLANGE_LENGTH errors are never injected on the datum panel")
            far_edges = far_parallel_edges(design, panel_key, key)
            if not far_edges:
                raise GeometryError(f"panel {panel_key} has no free far edge for bend {key}")
            for ei in far_edges:
                cuts[(panel_key, ei)] = cuts.get((panel_key, ei), 0.0) + float(p["delta_mm"])
    states: dict[str, BendState] = {}
    for b in design.bends:
        if b.key not in formed_set:
            states[b.key] = BendState(0.0, shift[b.key])
            continue
        sweep = b.sweep_deg - delta_incl[b.key]
        if b.key in partial:
            sweep = partial[b.key] * b.sweep_deg
        if not 0.0 < sweep < 179.0:
            raise GeometryError(f"bend {b.key}: actual sweep {sweep:.3f} out of range")
        states[b.key] = BendState(sweep, shift[b.key])
    as_design = design
    extra_keys: list[str] = []
    split = None
    if extra:
        fraction, sweep_x, direction = extra[0]
        target = _extra_bend_target(design, datum, formed_set)
        if target is None:
            raise GeometryError("no free rectangular flange for an extra bend")
        panel_key, parent_bend = target
        as_design, key_x, outer = _split_for_extra_bend(design, panel_key, parent_bend, fraction, sweep_x, direction)
        states[key_x] = BendState(sweep_x, 0.0)
        extra_keys.append(key_x)
        split = (panel_key, outer)
        # cut offsets on the split panel's edges no longer index the same edges
        cuts = {kk: v for kk, v in cuts.items() if kk[0] != panel_key}
        notes.append(f"extra crease on {panel_key} at {fraction:.3f} of its length")
    mirror_m = None
    if mirror:
        mirror_m = np.diag([-1.0, 1.0, 1.0, 1.0])
    if wrong:
        if wrong_part_design is None:
            raise GeometryError("WRONG_SKU needs a wrong_part_design")
        as_design = wrong_part_design
        w_datum = wrong_part_design.default_datum()
        states = {b.key: BendState(b.sweep_deg - float(rng.normal(0.0, 0.15)), float(rng.normal(0.0, 0.03))) for b in wrong_part_design.bends}
        cuts = {}
        notes.append("physical part is a different SKU's geometry")
        return AsBuiltPart(as_design, FormState(states, cuts), w_datum, None, (), {}, (), True, None, notes)
    formed_nominal = tuple(key for key in seq if states[key].sweep_deg > 0.0)
    return AsBuiltPart(as_design, FormState(states, cuts), datum, mirror_m, formed_nominal, partial, tuple(extra_keys), False, split, notes)


def side_panels(design: PartDesign, bend_key: str, datum: str) -> tuple[str, str]:
    """``(side A panel, side B panel)`` of a bend, named on the nominal final CAD
    pose by the +X rule (ENGINE_IO §1)."""
    from .geometry import FoldedPart, assign_sides_rule

    fp = FoldedPart(design, FormState.nominal(design), datum=datum)
    b = design.bend(bend_key)
    oa = fp.outward_cad(bend_key, b.panel_a)
    ob = fp.outward_cad(bend_key, b.panel_b)
    ia, ib = assign_sides_rule(oa, ob)
    pair = (b.panel_a, b.panel_b)
    return pair[ia], pair[ib]


def _toward_b_sign(design: PartDesign, bend_key: str, datum: str) -> float:
    """Sign turning a shift 'towards side B' into a shift along ``normal_ab``
    (``normal_ab`` points from ``panel_a`` to ``panel_b``)."""
    _, side_b = side_panels(design, bend_key, datum)
    return 1.0 if side_b == design.bend(bend_key).panel_b else -1.0


def _extra_bend_target(design: PartDesign, datum: str, formed: set[str]) -> tuple[str, str] | None:
    """A rectangular non-datum flange with a free far edge whose parent bend is formed."""
    from .geometry import FoldedPart

    fp = FoldedPart(design, FormState.nominal(design), datum=datum)
    candidates = []
    for p in design.panels:
        if p.key == datum or len(p.outline) != 4 or any(t != FREE for t in p.edge_tags if t != fp.parent_bend[p.key]):
            continue
        parent = fp.parent_bend[p.key]
        if parent is None or parent not in formed:
            continue
        if free_far_edge(design, p.key, parent) is None:
            continue
        candidates.append((-p.area(), p.key, parent))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]
