"""CAD roll solver: B-rep -> nominal roll features, in the style of E1's bend compile (research prototype).

Reads the STEP through E1's own reader (:func:`domains.production.engine.compile.brep.load_step`) and builds
E1's own flange graph (:func:`...flange_graph.build_flange_graph`), both imported read-only and unchanged.
On top of them it adds what E1 lacks: a **roll** feature.

Pipeline (one function each, all pure numpy on the BrepModel):

1. ``_curved_surfaces``  every cylindrical face and every analytic cone face, as a surface of revolution
   ``r(z) = r0 + k z`` about a canonical axis (k = 0 for a cylinder, k = tan(semi-angle) for a cone).
2. ``_pair_skins``        coaxial grouping (E1's tolerances: 0.25 deg, 0.02 mm), radius classes (0.01 mm),
   and concave/convex skin pairing: same axis, same taper, normal offset ``(r_out - r_in) cos(semi)`` = t,
   overlapping axially (> 1.5 x the offset, which rejects hole walls) and angularly. Connected components of
   the paired faces are the curved *features* (one per physical bend or roll).
3. ``_classify``          the bend/roll rule (module constant :data:`RULE_VERSION`, explained in SCHEMA.md):
   (B) an E1 flange-graph bend keeps E1's class and stays a bend, unless E1 itself declared it
       ROLLED (a) (Ri > 10 t, E1's frozen ``ROLLED_RI_OVER_T``); then it is a roll;
   (R) any other paired feature with Ri > 10 t is a roll (cylinder or cone);
   (J) a paired feature with Ri <= 10 t that is no E1 graph bend but is tangent to a roll is a bend
       (a *junction bend*: flange-to-roll or roll-to-roll), which E1 cannot place in its graph;
   anything else stays whatever E1 made of it (reported, never re-classified).
4. ``_runouts``           per roll end and per skin: the generator edges (straight, parallel to the axis)
   at the angular extremes of the skin, their neighbour face and its class (E1 panel, solver flange,
   bend, roll, free cut edge, edge round, free-form), merged into one run-out line per end and skin.
5. ``_checks``            consistency checks in the spirit of contracts/production/consistency.py.
6. ``_graph``             E1's panels and bends plus roll nodes, junction-bend edges and run-out edges.

Everything is in the STEP frame (``cad_model_mm``), mm and degrees. No part-specific logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from domains.production.engine.compile.brep import BrepModel
from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.flange_graph import FlangeGraph, GraphError, build_flange_graph
from domains.production.engine.compile.geometry import canonical_direction, line_distance_parallel, unit
from domains.production.engine.compile.unsupported import (
    ROLLED_RI_OVER_T,
    SWEEP_CONSISTENCY_TOL_DEG,
    TANGENCY_TOL_MM,
    UNSUPPORTED_RULE_VERSION,
)

__all__ = ["SCHEMA", "RULE_VERSION", "SolverConfig", "solve", "solve_step"]

SCHEMA = "cad_roll_nominal_v0"
RULE_VERSION = ("ROLL-RULE-0 (2026-10-25): E1 graph bend keeps E1 class unless E1 ROLLED(a); "
                f"paired feature with Ri > {ROLLED_RI_OVER_T:g} t is a ROLL; Ri <= {ROLLED_RI_OVER_T:g} t tangent "
                "to a roll is a junction BEND; thresholds reused from " + UNSUPPORTED_RULE_VERSION)

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class SolverConfig:
    """Tolerances. Every value is E1's (CompileConfig / unsupported.py) unless stated."""

    coaxial_angle_deg: float = 0.25        # E1 CompileConfig.coaxial_angle_deg
    coaxial_offset_mm: float = 0.02        # E1 CompileConfig.coaxial_offset_mm (absolute) ...
    coaxial_offset_rel: float = 1e-5       # ... plus this x R for very large radii (solver addition)
    same_radius_mm: float = 0.01           # E1 _SAME_RADIUS_MM
    taper_tol_deg: float = 0.25            # cones: same semi-angle (solver addition, = coaxial angle)
    min_offset_mm: float = 0.05            # E1 skin pairing: 0.05 <= Ro - Ri <= 30
    max_offset_mm: float = 30.0
    min_axial_overlap_t: float = 1.5       # solver addition: pair overlap > 1.5 t (a hole wall is t long)
    generator_angle_deg: float = 0.5       # E1 _GENERATOR_ANGLE_DEG
    end_cluster_deg: float = 1.0           # E1 _END_CLUSTER_DEG
    arc_merge_deg: float = 0.2             # angular gap closed when uniting the faces of one skin
    smooth_dihedral_deg: float = 1.0       # E1 CompileConfig.smooth_dihedral_deg
    tangency_tol_mm: float = TANGENCY_TOL_MM
    sweep_tol_deg: float = SWEEP_CONSISTENCY_TOL_DEG
    thickness_tol_mm: float = 0.05         # E1 CompileConfig.thickness_tol_mm
    thickness_tol_rel: float = 0.02        # E1 CompileConfig.thickness_tol_rel
    rolled_ri_over_t: float = ROLLED_RI_OVER_T
    station_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)


# --------------------------------------------------------------------------------------------- helpers

def _r6(x) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r6(v) for v in x]
    if isinstance(x, (float, np.floating)):
        v = round(float(x), 6)
        return 0.0 if v == 0 else v
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal (e1, e2) with e1 x e2 = axis."""
    k = int(np.argmin(np.abs(axis)))
    hint = np.zeros(3)
    hint[k] = 1.0
    e1 = unit(hint - float(hint @ axis) * axis)
    e2 = np.cross(axis, e1)
    return e1, e2


def _boundary_points(face) -> np.ndarray:
    pts = [u.xyz for lp in face.loops for u in lp.uses if len(u.xyz)]
    return np.concatenate(pts) if pts else face.centroid[None, :]


def _wrap(a: float) -> float:
    return a % TWO_PI


def _covering_arc(angles: np.ndarray) -> tuple[float, float]:
    """Smallest arc (start, span) containing every angle (rad)."""
    a = np.sort(np.mod(angles, TWO_PI))
    if len(a) < 2:
        return (float(a[0]) if len(a) else 0.0), 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + TWO_PI]))
    g = int(np.argmax(gaps))
    start = float(a[(g + 1) % len(a)])
    return start, float(TWO_PI - gaps[g])


def _union_arcs(arcs: list[tuple[float, float]], merge_tol: float) -> list[tuple[float, float]]:
    """Union of circular arcs (start, span); returns disjoint arcs. A full circle is [(0, 2 pi)]."""
    segs = []
    for s, w in arcs:
        if w >= TWO_PI - 1e-9:
            return [(0.0, TWO_PI)]
        s = _wrap(s)
        if s + w > TWO_PI:
            segs += [(s, TWO_PI), (0.0, s + w - TWO_PI)]
        else:
            segs.append((s, s + w))
    segs.sort()
    out: list[list[float]] = []
    for a, b in segs:
        if out and a <= out[-1][1] + merge_tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    if len(out) > 1 and out[0][0] <= merge_tol and out[-1][1] >= TWO_PI - merge_tol:
        out[0] = [out[-1][0], out[0][1] + TWO_PI]
        out.pop()
    if len(out) == 1 and out[0][1] - out[0][0] >= TWO_PI - merge_tol:
        return [(0.0, TWO_PI)]
    return [(a % TWO_PI, b - a) for a, b in out]


def _arcs_cover(arcs: list[tuple[float, float]]) -> tuple[float, float]:
    """Covering arc (start, span) of a set of disjoint arcs: 2 pi minus the largest gap."""
    if not arcs:
        return 0.0, 0.0
    if len(arcs) == 1:
        return arcs[0]
    arcs = sorted(arcs)
    gaps = []
    for i, (s, w) in enumerate(arcs):
        ns = arcs[(i + 1) % len(arcs)][0] + (TWO_PI if i == len(arcs) - 1 else 0.0)
        gaps.append(ns - (s + w))
    g = int(np.argmax(gaps))
    start = arcs[(g + 1) % len(arcs)][0]
    return start, TWO_PI - gaps[g]


def _arc_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Angular measure (rad) of the intersection of two arc sets."""
    def segs(arcs):
        out = []
        for s, w in arcs:
            if s + w > TWO_PI:
                out += [(s, TWO_PI), (0.0, s + w - TWO_PI)]
            else:
                out.append((s, s + w))
        return out
    tot = 0.0
    for a0, a1 in segs(a):
        for b0, b1 in segs(b):
            tot += max(0.0, min(a1, b1) - max(a0, b0))
    return tot


def _ang_dist(a: float, b: float) -> float:
    d = abs(_wrap(a) - _wrap(b))
    return min(d, TWO_PI - d)


# --------------------------------------------------------------------------------- 1. curved surfaces

@dataclass
class _Surf:
    face: int
    family: str  # cylinder | cone
    axis: np.ndarray  # canonical unit
    origin: np.ndarray  # point on the axis
    r0: float  # radius at origin
    k: float  # dr/dz along the canonical axis
    convex: bool
    area: float

    def radius_at(self, z: float, origin: np.ndarray) -> float:
        """Radius at axial coordinate z measured from ``origin`` (a point on the same axis)."""
        return self.r0 + self.k * (z + float((origin - self.origin) @ self.axis))


def _cone_surface(face) -> _Surf | None:
    ev = face._evaluator
    if ev is None:
        return None
    try:
        cone = ev._a.Cone()
    except Exception:
        return None
    apex = cone.Apex()
    apex = np.array([apex.X(), apex.Y(), apex.Z()])
    d = cone.Axis().Direction()
    d = np.array([d.X(), d.Y(), d.Z()])
    semi = float(cone.SemiAngle())  # radius grows along +d when semi > 0
    axis = canonical_direction(d)
    k = math.tan(semi) * (1.0 if float(axis @ d) > 0 else -1.0)
    u0, u1, v0, v1 = face.uv_bounds
    q = ev.point(0.5 * (u0 + u1), 0.5 * (v0 + v1))
    n = ev.normal(0.5 * (u0 + u1), 0.5 * (v0 + v1), face.reversed)
    rad = q - apex
    rad = rad - float(rad @ axis) * axis
    return _Surf(face.index, "cone", axis, apex, 0.0, k, bool(float(n @ rad) > 0), float(face.area))


def _curved_surfaces(model: BrepModel) -> dict[int, _Surf]:
    out: dict[int, _Surf] = {}
    for f in model.faces:
        if f.kind == "cylinder" and f.axis is not None and f.radius:
            out[f.index] = _Surf(f.index, "cylinder", canonical_direction(f.axis), f.origin.copy(), float(f.radius),
                                 0.0, bool(f.convex), float(f.area))
        elif f.surface == "cone":
            s = _cone_surface(f)
            if s is not None:
                out[f.index] = s
    return out


# ------------------------------------------------------------------------------------- 2. skin pairing

@dataclass
class Feature:
    """One paired curved feature (a bend or a roll): concave inner skin + convex outer skin."""

    family: str
    axis: np.ndarray
    origin: np.ndarray  # reference point on the axis (z measured from here)
    e1: np.ndarray
    e2: np.ndarray
    inner: list[int]
    outer: list[int]
    r_in0: float  # inner radius at origin
    r_out0: float
    k: float
    z_in: tuple[float, float]
    z_out: tuple[float, float]
    arcs_in: list[tuple[float, float]]
    arcs_out: list[tuple[float, float]]
    kind: str | None = None  # ROLL | BEND | None
    subtype: str = ""
    e1_bend: str | None = None
    e1_class: str | None = None
    fid: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def faces(self) -> list[int]:
        return sorted(self.inner + self.outer)

    @property
    def cos_semi(self) -> float:
        return 1.0 / math.sqrt(1.0 + self.k * self.k)

    @property
    def thickness(self) -> float:
        return (self.r_out0 - self.r_in0) * self.cos_semi

    @property
    def z_range(self) -> tuple[float, float]:
        return min(self.z_in[0], self.z_out[0]), max(self.z_in[1], self.z_out[1])

    @property
    def z_mid(self) -> float:
        a, b = self.z_range
        return 0.5 * (a + b)

    def r_in(self, z: float) -> float:
        return self.r_in0 + self.k * z

    def r_out(self, z: float) -> float:
        return self.r_out0 + self.k * z

    @property
    def cover(self) -> tuple[float, float]:
        return _arcs_cover(_union_arcs(self.arcs_in + self.arcs_out, math.radians(0.2)))


def _face_geometry(model, s: _Surf, origin, axis, e1, e2, closed: bool):
    p = _boundary_points(model.faces[s.face])
    rel = p - origin
    z = rel @ axis
    if closed:
        arc = (0.0, TWO_PI)
    else:
        arc = _covering_arc(np.arctan2(rel @ e2, rel @ e1))
    return (float(z.min()), float(z.max())), arc


def _pair_skins(model: BrepModel, surfs: dict[int, _Surf], t_ref: float | None, cfg: SolverConfig) -> list[Feature]:
    faces = model.faces
    keys = sorted(surfs)
    cos_ax = math.cos(math.radians(cfg.coaxial_angle_deg))
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(keys):
        sa = surfs[a]
        for b in keys[i + 1:]:
            sb = surfs[b]
            if sa.family != sb.family or abs(float(sa.axis @ sb.axis)) < cos_ax:
                continue
            rmax = max(sa.r0, sb.r0, 1.0)
            if line_distance_parallel(sa.origin, sa.axis, sb.origin) > cfg.coaxial_offset_mm + cfg.coaxial_offset_rel * rmax:
                continue
            if sa.family == "cone" and abs(math.degrees(math.atan(sa.k) - math.atan(sb.k))) > cfg.taper_tol_deg:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)

    features: list[Feature] = []
    for members in groups.values():
        ref = surfs[members[0]]
        axis, origin = ref.axis, ref.origin
        e1, e2 = _frame(axis)
        closed = {m: any(model.edges[e].seam for e in faces[m].edges) for m in members}
        geo = {m: _face_geometry(model, surfs[m], origin, axis, e1, e2, closed[m] or
                                 (surfs[m].family == "cylinder" and faces[m].angular_span >= TWO_PI - 1e-6))
               for m in members}
        r_at0 = {m: surfs[m].radius_at(0.0, origin) for m in members}
        # radius classes per convexity
        classes: list[dict] = []
        for m in sorted(members, key=lambda q: (surfs[q].convex, r_at0[q])):
            c = classes[-1] if classes else None
            if c and c["convex"] == surfs[m].convex and abs(r_at0[m] - c["r"]) <= cfg.same_radius_mm:
                c["faces"].append(m)
            else:
                classes.append({"convex": surfs[m].convex, "r": r_at0[m], "faces": [m]})
        for c in classes:
            c["z"] = (min(geo[m][0][0] for m in c["faces"]), max(geo[m][0][1] for m in c["faces"]))
            c["arcs"] = _union_arcs([geo[m][1] for m in c["faces"]], math.radians(cfg.arc_merge_deg))
        cos_semi = 1.0 / math.sqrt(1.0 + ref.k * ref.k)
        cands = []
        for ci, c in enumerate(classes):
            if c["convex"]:
                continue
            for vi, v in enumerate(classes):
                if not v["convex"]:
                    continue
                off = (v["r"] - c["r"]) * cos_semi
                if not (cfg.min_offset_mm <= off <= cfg.max_offset_mm):
                    continue
                ov = min(c["z"][1], v["z"][1]) - max(c["z"][0], v["z"][0])
                if ov <= cfg.min_axial_overlap_t * off:
                    continue
                if _arc_overlap(c["arcs"], v["arcs"]) <= math.radians(0.5):
                    continue
                score = abs(off - t_ref) if t_ref else off
                cands.append((score, ci, vi, off))
        used: set[int] = set()
        for _, ci, vi, off in sorted(cands):
            if ci in used or vi in used:
                continue
            used.update((ci, vi))
            c, v = classes[ci], classes[vi]
            # connected components: same-skin faces sharing an edge, inner/outer faces overlapping in z and angle
            fs = c["faces"] + v["faces"]
            par = {m: m for m in fs}

            def f2(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x

            for i, a in enumerate(fs):
                for b in fs[i + 1:]:
                    same_skin = surfs[a].convex == surfs[b].convex
                    if same_skin:
                        link = bool(set(faces[a].edges) & set(faces[b].edges))
                    else:
                        (za, aa), (zb, ab) = geo[a], geo[b]
                        link = (min(za[1], zb[1]) - max(za[0], zb[0]) > 0.05 and
                                _arc_overlap([aa], [ab]) > math.radians(0.5))
                    if link:
                        par[f2(a)] = f2(b)
            comps: dict[int, list[int]] = {}
            for m in fs:
                comps.setdefault(f2(m), []).append(m)
            for comp in comps.values():
                ins = sorted(m for m in comp if not surfs[m].convex)
                outs = sorted(m for m in comp if surfs[m].convex)
                if not ins or not outs:
                    continue
                features.append(Feature(
                    family=ref.family, axis=axis, origin=origin, e1=e1, e2=e2, inner=ins, outer=outs,
                    r_in0=c["r"], r_out0=v["r"], k=ref.k,
                    z_in=(min(geo[m][0][0] for m in ins), max(geo[m][0][1] for m in ins)),
                    z_out=(min(geo[m][0][0] for m in outs), max(geo[m][0][1] for m in outs)),
                    arcs_in=_union_arcs([geo[m][1] for m in ins], math.radians(cfg.arc_merge_deg)),
                    arcs_out=_union_arcs([geo[m][1] for m in outs], math.radians(cfg.arc_merge_deg))))
    return features


# ------------------------------------------------------------------------------------- 3. bend / roll

def _e1_face_bend(graph: FlangeGraph | None) -> dict[int, int]:
    out: dict[int, int] = {}
    if graph is None:
        return out
    for bk, b in graph.bends.items():
        for f in b.inner_faces + b.outer_faces:
            out[f] = bk
    return out


def _smooth_neighbours(model: BrepModel, faces: list[int], cfg: SolverConfig) -> set[int]:
    own = set(faces)
    out = set()
    for f in faces:
        for e, g in model.neighbors(f):
            d = model.edges[e].dihedral_deg
            if g not in own and d is not None and d <= cfg.smooth_dihedral_deg:
                out.add(g)
    return out


def _classify(model: BrepModel, graph: FlangeGraph | None, feats: list[Feature], cfg: SolverConfig) -> None:
    f2b = _e1_face_bend(graph)
    for ft in feats:
        bks = {f2b[f] for f in ft.faces if f in f2b}
        if len(bks) == 1:
            b = graph.bends[next(iter(bks))]
            ft.e1_bend, ft.e1_class = b.graph_bend_id, b.unsupported
        elif len(bks) > 1:
            ft.notes.append("faces span several E1 graph bends: " + ",".join(sorted(graph.bends[k].graph_bend_id for k in bks)))
        ri = ft.r_in(ft.z_mid)
        rolled = ri > cfg.rolled_ri_over_t * ft.thickness
        if ft.e1_bend is not None and ft.e1_class != "ROLLED":
            ft.kind, ft.subtype = "BEND", "E1_GRAPH_BEND"
        elif rolled:
            ft.kind = "ROLL"
            ft.subtype = "E1_ROLLED_A_GRAPH_BEND" if ft.e1_bend else "ROLL"
    # junction bends: Ri <= 10 t, no E1 bend, smooth-tangent to a roll (iterate: roll-junction-junction chains)
    changed = True
    while changed:
        changed = False
        roll_faces = {f for ft in feats if ft.kind == "ROLL" for f in ft.faces}
        jb_faces = {f for ft in feats if ft.subtype == "JUNCTION_BEND" for f in ft.faces}
        for ft in feats:
            if ft.kind is not None:
                continue
            nb = _smooth_neighbours(model, ft.faces, cfg)
            if nb & roll_faces:
                ft.kind, ft.subtype = "BEND", "JUNCTION_BEND"
                changed = True
            elif nb & jb_faces and any(model.faces[g].kind == "cylinder" for g in nb & jb_faces):
                pass  # a bend chained to a junction bend only (no roll contact) stays E1's
    rolls = sorted([f for f in feats if f.kind == "ROLL"], key=lambda f: (-f.r_in(f.z_mid) * f.cover[1], f.faces[0]))
    for i, f in enumerate(rolls, 1):
        f.fid = f"R{i}"
    jbs = sorted([f for f in feats if f.subtype == "JUNCTION_BEND"], key=lambda f: f.faces[0])
    for i, f in enumerate(jbs, 1):
        f.fid = f"J{i}"


# ----------------------------------------------------------------------------------------- 4. run-outs

def _plane_skin_partner(model: BrepModel, p: int, t: float | None) -> bool:
    """True when plane ``p`` has an opposite parallel plane at distance ~t (a sheet skin, not a wall)."""
    if t is None:
        return False
    fp = model.faces[p]
    for g in model.faces:
        if g.kind != "plane" or g.index == p or g.normal is None:
            continue
        if float(g.normal @ fp.normal) > -math.cos(math.radians(1.0)):
            continue
        d = abs(float((g.centroid - fp.centroid) @ fp.normal))
        if abs(d - t) <= 0.05 + 0.02 * t:
            return True
    return False


class _Ctx:
    def __init__(self, model, graph, feats, t_part, cfg, t_independent=True):
        self.model, self.graph, self.cfg, self.t = model, graph, cfg, t_part
        self.feats, self.t_independent = feats, t_independent
        self.face_feat = {f: ft for ft in feats for f in ft.faces}
        self.flanges: dict[frozenset, str] = {}
        self._skin_cache: dict[int, bool] = {}

    def is_skin_plane(self, p: int) -> bool:
        if p not in self._skin_cache:
            self._skin_cache[p] = _plane_skin_partner(self.model, p, self.t)
        return self._skin_cache[p]

    def neighbour(self, g: int) -> dict[str, Any]:
        """Class and id of the face across a run-out generator edge."""
        model, graph = self.model, self.graph
        face = model.faces[g]
        role = graph.face_role.get(g) if graph is not None else None
        if g in self.face_feat:
            ft = self.face_feat[g]
            if ft.kind == "ROLL":
                return {"class": "ROLL", "id": ft.fid}
            if ft.kind == "BEND":
                return {"class": "BEND", "id": ft.e1_bend or ft.fid}
        if role and role[0] == "bend":
            b = graph.bends[role[1]]
            return {"class": "BEND", "id": b.graph_bend_id, "e1_class": b.unsupported}
        if face.kind == "plane":
            if role and role[0] == "panel":
                return {"class": "FLANGE", "id": graph.panels[role[1]].panel_id, "source": "E1_PANEL"}
            if self.is_skin_plane(g):
                return {"class": "FLANGE", "id": None, "source": "SOLVER_FLANGE"}
            return {"class": "WALL_TANGENT", "id": None}
        if face.kind == "cylinder":
            r = face.radius or 0.0
            if self.t is not None and r <= 1.5 * self.t:
                return {"class": "EDGE_ROUND", "id": None}
            return {"class": "CURVED_UNPAIRED", "id": None}
        return {"class": "FREEFORM", "id": None, "surface": face.surface}


def _runouts(ctx: _Ctx, ft: Feature) -> dict[str, Any]:
    model, cfg = ctx.model, ctx.cfg
    start, span = ft.cover
    closed = span >= TWO_PI - math.radians(cfg.arc_merge_deg)
    ends_ang = [start, start + span]
    sin_gen = math.sin(math.radians(cfg.generator_angle_deg))
    # the slant of a cone generator against the axis is its semi-angle (0 for a cylinder)
    cos_slant = math.cos(math.atan(abs(ft.k)) + math.radians(cfg.generator_angle_deg))
    out: dict[str, Any] = {"closed": closed, "ends": []}
    per_end: list[dict[str, Any]] = [{"which": "start", "angle": ends_ang[0]}, {"which": "end", "angle": ends_ang[1]}]
    interior = 0
    for skin, fl in (("inner", ft.inner), ("outer", ft.outer)):
        own = set(fl)
        hits: list[list[dict]] = [[], []]
        for f in fl:
            for e in model.faces[f].edges:
                edge = model.edges[e]
                if edge.degenerate or edge.seam or edge.curve != "line" or edge.direction is None:
                    continue
                mid = 0.5 * (edge.points[0] + edge.points[-1]) - ft.origin
                phi = math.atan2(float(mid @ ft.e2), float(mid @ ft.e1))
                # a generator lies in a meridian plane: perpendicular to the circumferential direction (for a
                # cylinder that is "parallel to the axis", E1's test; for a cone it is the slant line)
                circ = -math.sin(phi) * ft.e1 + math.cos(phi) * ft.e2
                if abs(float(edge.direction @ circ)) > sin_gen or abs(float(edge.direction @ ft.axis)) < cos_slant:
                    continue
                others = [g for g in edge.faces if g not in own]
                if not others:
                    continue  # split between faces of one skin
                g = others[0]
                d = [_ang_dist(phi, a) for a in ends_ang]
                j = int(np.argmin(d))
                if closed or d[j] > math.radians(cfg.end_cluster_deg):
                    interior += 1
                    continue
                z = (edge.points - ft.origin) @ ft.axis
                smooth = edge.dihedral_deg is not None and edge.dihedral_deg <= cfg.smooth_dihedral_deg
                nb = ctx.neighbour(g) if smooth else {"class": "FREE_EDGE", "id": None}
                hits[j].append({"edge": int(e), "face": int(g), "z": (float(z.min()), float(z.max())), "phi": phi,
                                "length": float(edge.length), "smooth": bool(smooth),
                                "dihedral_deg": edge.dihedral_deg, **nb})
        for j in (0, 1):
            per_end[j][skin] = hits[j]
    if closed:
        out["interior_generator_edges"] = interior
        return out
    zr = {"inner": ft.z_in, "outer": ft.z_out}
    for j, pe in enumerate(per_end):
        end: dict[str, Any] = {"which": pe["which"], "angle_rad": _wrap(pe["angle"])}
        for skin in ("inner", "outer"):
            hs = pe[skin]
            radial = math.cos(pe["angle"]) * ft.e1 + math.sin(pe["angle"]) * ft.e2
            rfun = ft.r_in if skin == "inner" else ft.r_out
            if hs:
                z0, z1 = min(h["z"][0] for h in hs), max(h["z"][1] for h in hs)
                covered = sum(h["z"][1] - h["z"][0] for h in hs)
                byclass: dict[str, float] = {}
                for h in hs:
                    byclass[h["class"]] = byclass.get(h["class"], 0.0) + (h["z"][1] - h["z"][0])
                cls = max(sorted(byclass), key=lambda c: byclass[c])
                ids = sorted({h["id"] for h in hs if h.get("id")})
            else:
                z0, z1 = zr[skin]
                covered, cls, ids = 0.0, "NO_GENERATOR_EDGE", []
            p0 = ft.origin + z0 * ft.axis + rfun(z0) * radial
            p1 = ft.origin + z1 * ft.axis + rfun(z1) * radial
            end[skin] = {"start_mm": p0, "end_mm": p1, "length_mm": z1 - z0, "class": cls,
                         "neighbour_ids": ids, "neighbour_faces": sorted({h["face"] for h in hs}),
                         "edges": sorted({h["edge"] for h in hs}),
                         "coverage": covered / max(zr[skin][1] - zr[skin][0], 1e-9),
                         "_hits": hs, "_radial": radial}
        out["ends"].append(end)
    out["interior_generator_edges"] = interior
    return out


def _assign_solver_flanges(ctx: _Ctx, rolls_ro: list[tuple[Feature, dict]]) -> None:
    """Ids for planar flanges at run-outs that are not E1 panels: one F<n> per (inner plane set, outer plane set)."""
    n = 0
    for ft, ro in rolls_ro:
        for end in ro.get("ends", []):
            planes = frozenset(h["face"] for skin in ("inner", "outer") for h in end[skin]["_hits"]
                               if h["class"] == "FLANGE" and h.get("source") == "SOLVER_FLANGE")
            if not planes:
                continue
            key = None
            for k in ctx.flanges:
                if k & planes:
                    key = k
                    break
            if key is None:
                n += 1
                ctx.flanges[planes] = f"F{n}"
                fid = f"F{n}"
            else:
                fid = ctx.flanges.pop(key)
                ctx.flanges[key | planes] = fid
            for skin in ("inner", "outer"):
                for h in end[skin]["_hits"]:
                    if h["face"] in planes:
                        h["id"] = fid
                ids = sorted({h["id"] for h in end[skin]["_hits"] if h.get("id")})
                end[skin]["neighbour_ids"] = ids


# ------------------------------------------------------------------------------------------- 5. checks

def _check(name: str, ok: bool | None, value=None, tol=None, detail: str = "") -> dict[str, Any]:
    status = "NOT_APPLICABLE" if ok is None else ("PASS" if ok else "FAIL")
    return {"check": name, "status": status, "value": _r6(value), "tol": _r6(tol), "detail": detail}


def _checks(ctx: _Ctx, ft: Feature, ro: dict, surfs: dict[int, _Surf]) -> list[dict[str, Any]]:
    cfg, model = ctx.cfg, ctx.model
    out = []
    # skins coaxial: every face's axis vs the feature axis
    worst_ang, worst_off = 0.0, 0.0
    for f in ft.faces:
        s = surfs[f]
        worst_ang = max(worst_ang, math.degrees(math.acos(min(1.0, abs(float(s.axis @ ft.axis))))))
        worst_off = max(worst_off, line_distance_parallel(ft.origin, ft.axis, s.origin))
    out.append(_check("SKINS_COAXIAL_ANGLE_DEG", worst_ang <= cfg.coaxial_angle_deg, worst_ang, cfg.coaxial_angle_deg))
    tol_off = cfg.coaxial_offset_mm + cfg.coaxial_offset_rel * max(ft.r_out0, 1.0)
    out.append(_check("SKINS_COAXIAL_OFFSET_MM", worst_off <= tol_off, worst_off, tol_off))
    # thickness vs the part
    t_ind, src = ctx.t, "E1 flange graph"
    if not ctx.t_independent:
        others = [o.thickness for o in ctx.feats if o is not ft]
        t_ind, src = (float(np.median(others)), "other paired features") if others else (None, "")
    if t_ind is not None:
        tol = cfg.thickness_tol_mm + cfg.thickness_tol_rel * t_ind
        out.append(_check("THICKNESS_CONSISTENT_MM", abs(ft.thickness - t_ind) <= tol, ft.thickness - t_ind, tol,
                          f"roll t {ft.thickness:.4f} vs part t {t_ind:.4f} ({src})"))
    else:
        out.append(_check("THICKNESS_CONSISTENT_MM", None, detail="no independent part thickness (E1 has none and "
                          "no other paired feature)"))
    # skins sweep agreement
    si, so = _arcs_cover(ft.arcs_in)[1], _arcs_cover(ft.arcs_out)[1]
    out.append(_check("SKIN_SWEEPS_AGREE_DEG", abs(math.degrees(si - so)) <= cfg.sweep_tol_deg,
                      math.degrees(si - so), cfg.sweep_tol_deg))
    # axial overlap of the skins
    zi, zo = ft.z_in, ft.z_out
    ov = (min(zi[1], zo[1]) - max(zi[0], zo[0])) / max(1e-9, max(zi[1], zo[1]) - min(zi[0], zo[0]))
    out.append(_check("SKIN_AXIAL_OVERLAP_FRACTION", ov >= 0.9, ov, 0.9))
    if ft.family == "cone":
        ks = [math.degrees(math.atan(surfs[f].k)) for f in ft.faces]
        out.append(_check("CONE_SEMI_ANGLES_AGREE_DEG", max(ks) - min(ks) <= cfg.taper_tol_deg, max(ks) - min(ks),
                          cfg.taper_tol_deg))
    # run-outs tangent
    if ro.get("closed"):
        out.append(_check("RUNOUT_TANGENT", None, detail="closed roll: no run-outs"))
        return out
    worst, n_tan, normals = 0.0, 0, {}
    for end in ro["ends"]:
        for skin in ("inner", "outer"):
            for h in end[skin]["_hits"]:
                if not h["smooth"]:
                    continue
                g = model.faces[h["face"]]
                z = 0.5 * (h["z"][0] + h["z"][1])
                radial = math.cos(h["phi"]) * ft.e1 + math.sin(h["phi"]) * ft.e2
                r_skin = (ft.r_in if skin == "inner" else ft.r_out)(z)
                if g.kind == "plane":
                    # the skin's outward normal along the generator (cone: tilted by the semi-angle); E1 convention:
                    # an outer-skin plane has the same outward normal, an inner-skin plane the opposite one
                    sign = 1.0 if skin == "outer" else -1.0
                    n_skin = unit(radial - ft.k * ft.axis)
                    p_gen = ft.origin + z * ft.axis + r_skin * radial
                    aligned = float(g.normal @ n_skin) * sign > math.cos(math.radians(2.0))
                    err = abs(float((g.centroid - p_gen) @ g.normal)) if aligned else float("inf")
                    normals.setdefault(end["which"], []).append(g.normal * sign)
                elif g.kind == "cylinder" and g.radius:
                    ga = canonical_direction(g.axis)
                    if abs(float(ga @ ft.axis)) < math.cos(math.radians(cfg.coaxial_angle_deg)):
                        err = float("inf")
                    else:
                        dc = line_distance_parallel(ft.origin, ft.axis, g.origin)
                        err = min(abs(dc - abs(r_skin - g.radius)), abs(dc - (r_skin + g.radius)))
                else:
                    continue
                worst = max(worst, err)
                n_tan += 1
    if n_tan:
        out.append(_check("RUNOUT_TANGENT_MM", worst <= cfg.tangency_tol_mm, worst, cfg.tangency_tol_mm,
                          f"{n_tan} smooth run-out edges"))
    else:
        out.append(_check("RUNOUT_TANGENT_MM", None, detail="no smooth run-out (free cut ends)"))
    if len(normals) == 2:
        na = unit(np.mean(normals["start"], axis=0))
        nb = unit(np.mean(normals["end"], axis=0))
        dihedral = math.degrees(math.acos(float(np.clip(na @ nb, -1, 1))))
        span = math.degrees(ft.cover[1])
        sweep_planes = dihedral if abs(span - dihedral) <= abs(span - (360 - dihedral)) else 360 - dihedral
        out.append(_check("SWEEP_VS_RUNOUT_PLANES_DEG", abs(sweep_planes - span) <= cfg.sweep_tol_deg,
                          sweep_planes - span, cfg.sweep_tol_deg, "flange-plane dihedral vs skin angular span"))
    return out


# ---------------------------------------------------------------------------------------------- export

def _line(p0, p1) -> dict[str, Any]:
    return {"start_mm": _r6(p0), "end_mm": _r6(p1)}


def _feature_record(ctx: _Ctx, ft: Feature, ro: dict, surfs, cfg: SolverConfig) -> dict[str, Any]:
    start, span = ft.cover
    z0, z1 = ft.z_range
    zm = 0.5 * (z0 + z1)
    mid_ang = start + 0.5 * span
    bulge = math.cos(mid_ang) * ft.e1 + math.sin(mid_ang) * ft.e2
    ri, rout = ft.r_in(zm), ft.r_out(zm)
    s = min(span, TWO_PI)
    chord_in, chord_out = 2 * ri * math.sin(s / 2), 2 * rout * math.sin(s / 2)
    role = ctx.graph.face_role if ctx.graph is not None else {}
    e1_roles = sorted({"/".join(str(x) for x in role.get(f, ("unassigned",))) for f in ft.faces})
    rec: dict[str, Any] = {
        "id": ft.fid,
        "kind": ft.kind,
        "subtype": ft.subtype,
        "surface": ft.family,
        "is_cone": ft.family == "cone",
        "axis": {"point_mm": _r6(ft.origin + zm * ft.axis), "direction": _r6(ft.axis),
                 "segment": _line(ft.origin + z0 * ft.axis, ft.origin + z1 * ft.axis)},
        "frame": {"e1": _r6(ft.e1), "e2": _r6(ft.e2),
                  "note": "angles about the axis: atan2((p-axis.point)·e2, (p-axis.point)·e1), e1 x e2 = direction"},
        "r_inner_mm": _r6(ri),
        "r_outer_mm": _r6(rout),
        "thickness_mm": _r6(ft.thickness),
        "r_inner_over_t": _r6(ri / ft.thickness),
        "r_mid_mm": _r6(0.5 * (ri + rout)),
        "sweep_deg": _r6(math.degrees(span)),
        "sweep_inner_deg": _r6(math.degrees(_arcs_cover(ft.arcs_in)[1])),
        "sweep_outer_deg": _r6(math.degrees(_arcs_cover(ft.arcs_out)[1])),
        "start_angle_deg": _r6(math.degrees(_wrap(start))),
        "closed": bool(ro.get("closed")),
        "axial_length_mm": _r6(z1 - z0),
        "axial_length_inner_mm": _r6(ft.z_in[1] - ft.z_in[0]),
        "axial_length_outer_mm": _r6(ft.z_out[1] - ft.z_out[0]),
        "chord_inner_mm": _r6(chord_in),
        "chord_outer_mm": _r6(chord_out),
        "opening_mm": _r6(chord_in),
        "opening_gap_deg": _r6(max(0.0, 360.0 - math.degrees(span))),
        "arc_length_mid_mm": _r6(0.5 * (ri + rout) * span),
        "taper": {"dr_dz": _r6(ft.k), "mm_per_m": _r6(ft.k * 1000.0),
                  "semi_angle_deg": _r6(math.degrees(math.atan(ft.k))),
                  "r_inner_at_axis_ends_mm": _r6([ft.r_in(z0), ft.r_in(z1)]),
                  "r_outer_at_axis_ends_mm": _r6([ft.r_out(z0), ft.r_out(z1)])},
        "convexity": {"concave_skin": "inner", "convex_skin": "outer",
                      "bulge_direction": _r6(bulge),
                      "note": "unit radial from the axis to the angular middle of the sweep (the convex side)"},
        "faces": {"inner": ft.inner, "outer": ft.outer},
        "e1": {"graph_bend_id": ft.e1_bend, "class": ft.e1_class, "face_roles": e1_roles},
        "runouts": [],
        "neighbours": [],
        "stations": [],
        "interior_generator_edges": ro.get("interior_generator_edges", 0),
        "notes": ft.notes,
    }
    if not ro.get("closed"):
        nbs: set[str] = set()
        for end in ro["ends"]:
            r = {"which": end["which"], "angle_deg": _r6(math.degrees(end["angle_rad"]))}
            for skin in ("inner", "outer"):
                e = end[skin]
                r[skin] = {**_line(e["start_mm"], e["end_mm"]), "length_mm": _r6(e["length_mm"]),
                           "class": e["class"], "neighbour_ids": e["neighbour_ids"],
                           "neighbour_faces": e["neighbour_faces"], "edges": e["edges"],
                           "coverage": _r6(min(1.0, e["coverage"]))}
                nbs.update(e["neighbour_ids"])
            r["class"] = r["inner"]["class"] if r["inner"]["class"] == r["outer"]["class"] else \
                f"{r['inner']['class']}|{r['outer']['class']}"
            tangent = {"FLANGE", "BEND", "ROLL", "WALL_TANGENT", "CURVED_UNPAIRED", "FREEFORM"}
            r["end_type"] = ("TANGENT_RUNOUT" if r["inner"]["class"] in tangent and r["outer"]["class"] in tangent
                             else "FREE_END" if r["inner"]["class"] not in tangent and r["outer"]["class"] not in tangent
                             else "MIXED")
            rec["runouts"].append(r)
        rec["neighbours"] = sorted(nbs)
    for fr in cfg.station_fractions:
        z = z0 + fr * (z1 - z0)
        rec["stations"].append({"fraction": fr, "z_mm": _r6(z), "centre_mm": _r6(ft.origin + z * ft.axis),
                                "r_inner_mm": _r6(ft.r_in(z)), "r_outer_mm": _r6(ft.r_out(z)),
                                "start_angle_deg": _r6(math.degrees(_wrap(start))),
                                "sweep_deg": _r6(math.degrees(span))})
    rec["checks"] = _checks(ctx, ft, ro, surfs)
    rec["checks_pass"] = all(c["status"] != "FAIL" for c in rec["checks"])
    return rec


def _graph_view(graph: FlangeGraph | None, rolls: list[dict], jbends: list[dict], e1_bends: list[dict]) -> dict:
    nodes, edges = [], []
    if graph is not None:
        for p in sorted(graph.panels.values(), key=lambda p: int(p.panel_id[1:])):
            nodes.append({"id": p.panel_id, "type": "PANEL", "source": "E1"})
    seen = {n["id"] for n in nodes}
    for r in rolls:
        nodes.append({"id": r["id"], "type": "ROLL", "source": "ROLL_SOLVER"})
        seen.add(r["id"])
    for b in e1_bends:
        if b["class"] == "ROLLED":  # E1 ROLLED (a): replaced by the roll node and its two run-out edges
            continue
        edges.append({"id": b["graph_bend_id"], "type": "BEND", "source": "E1", "ends": [b["parent"], b["child"]],
                      "e1_class": b["class"]})
    def node(nid: str) -> None:
        if nid not in seen and nid[:1] in ("P", "F", "R"):
            nodes.append({"id": nid, "type": {"P": "PANEL", "F": "FLANGE", "R": "ROLL"}[nid[:1]],
                          "source": "E1" if nid[:1] == "P" else "ROLL_SOLVER"})
            seen.add(nid)

    for r in rolls:
        for ro in r["runouts"]:
            ids = sorted(set(ro["inner"]["neighbour_ids"]) | set(ro["outer"]["neighbour_ids"]))
            for nid in ids:
                if nid[:1] in ("P", "F", "R"):  # junction / E1 bends carry their own edge
                    node(nid)
                    edges.append({"id": f"{r['id']}:{ro['which']}", "type": "RUNOUT", "source": "ROLL_SOLVER",
                                  "ends": [r["id"], nid], "runout_class": ro["class"]})
    for j in jbends:
        ends = []
        for ro in j["runouts"]:
            ids = [i for i in sorted(set(ro["inner"]["neighbour_ids"]) | set(ro["outer"]["neighbour_ids"]))
                   if i[:1] in ("P", "F", "R")]
            for nid in ids:
                node(nid)
            ends.append(ids[0] if ids else None)
        while len(ends) < 2:
            ends.append(None)
        edges.append({"id": j["id"], "type": "JUNCTION_BEND", "source": "ROLL_SOLVER", "ends": ends[:2]})
    # connectivity
    par = {n["id"]: n["id"] for n in nodes}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for e in edges:
        a, b = e["ends"]
        if a in par and b in par:
            par[f(a)] = f(b)
    # a roll-to-roll run-out is listed from both rolls: keep one edge per unordered pair
    uniq, keyset = [], set()
    for e in edges:
        k = (e["type"], tuple(sorted(str(x) for x in e["ends"])), e["id"] if e["type"] != "RUNOUT" else "")
        if e["type"] == "RUNOUT" and str(e["ends"][1]).startswith("R"):
            if k in keyset:
                continue
        keyset.add(k)
        uniq.append(e)
    edges = uniq
    comps = sorted({f(n["id"]) for n in nodes})
    return {"nodes": nodes, "edges": edges, "components": len(comps),
            "coherent": len(comps) <= 1 and all(e["ends"][0] and e["ends"][1] for e in edges)}


def solve(model: BrepModel, cfg: SolverConfig | None = None, compile_cfg: CompileConfig | None = None,
          source: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve one B-rep. Returns ``(cad_roll_nominal dict, private render data)``."""
    cfg = cfg or SolverConfig()
    ccfg = compile_cfg or CompileConfig()
    graph, graph_error = None, None
    try:
        graph = build_flange_graph(model, ccfg)
    except GraphError as exc:
        graph_error = f"{exc.code}: {exc.detail}"
    datum_error = None
    if graph is not None:
        from domains.production.engine.compile.part_model import build_part_model
        try:
            build_part_model(graph)
        except GraphError as exc:
            datum_error = f"{exc.code}: {exc.detail}"
    t_e1 = graph.thickness if graph is not None else None
    surfs = _curved_surfaces(model)
    feats = _pair_skins(model, surfs, t_e1, cfg)
    # part thickness: E1's when E1 has one, else the area-weighted median offset of the paired features
    t_part, t_src = t_e1, "E1_FLANGE_GRAPH"
    if t_part is None and feats:
        w = np.array([sum(surfs[f].area for f in ft.faces) for ft in feats])
        v = np.array([ft.thickness for ft in feats])
        o = np.argsort(v)
        t_part = float(v[o][np.searchsorted(np.cumsum(w[o]), 0.5 * w.sum())])
        t_src = "PAIRED_SKINS_WEIGHTED_MEDIAN (no independent source)"
    _classify(model, graph, feats, cfg)
    ctx = _Ctx(model, graph, feats, t_part, cfg, t_independent=t_e1 is not None)
    curved = [ft for ft in feats if ft.kind == "ROLL" or ft.subtype == "JUNCTION_BEND"]
    ros = [(ft, _runouts(ctx, ft)) for ft in curved]
    _assign_solver_flanges(ctx, ros)
    # junction bends name rolls/flanges; rolls name junction bends: ids are now final
    rolls, jbends, render = [], [], {"features": []}
    for ft, ro in ros:
        rec = _feature_record(ctx, ft, ro, surfs, cfg)
        (rolls if ft.kind == "ROLL" else jbends).append(rec)
        render["features"].append({"id": ft.fid, "kind": ft.kind, "origin": ft.origin, "axis": ft.axis, "e1": ft.e1,
                                   "e2": ft.e2, "r_in0": ft.r_in0, "r_out0": ft.r_out0, "k": ft.k,
                                   "z": ft.z_range, "cover": ft.cover, "runouts": ro})
    rolls.sort(key=lambda r: r["id"])
    jbends.sort(key=lambda r: r["id"])
    # E1 bends: kept verbatim (minus E1 ROLLED (a), which became rolls); own values cross-checked
    e1_bends, solver_bends = [], []
    by_e1 = {ft.e1_bend: ft for ft in feats if ft.e1_bend}
    if graph is not None:
        for b in sorted(graph.bends.values(), key=lambda b: b.graph_bend_id):
            a, e = b.oml_sharp()
            parent, child = graph.panels[b.ends[0].panel].panel_id, graph.panels[b.ends[1].panel].panel_id
            rec = {"graph_bend_id": b.graph_bend_id, "class": b.unsupported, "parent": parent, "child": child,
                   "inner_radius_mm": _r6(b.ri), "thickness_mm": _r6(b.thickness), "sweep_deg": _r6(b.sweep_deg),
                   "length_mm": _r6(b.length), "oml_virtual_sharp": _line(a, e),
                   "faces": {"inner": b.inner_faces, "outer": b.outer_faces}}
            ft = by_e1.get(b.graph_bend_id)
            if ft is not None:
                rec["solver_cross_check"] = {"paired": True, "ri_diff_mm": _r6(ft.r_in(ft.z_mid) - b.ri),
                                             "t_diff_mm": _r6(ft.thickness - b.thickness),
                                             "sweep_diff_deg": _r6(math.degrees(ft.cover[1]) - b.cylinder_sweep_deg)}
            else:
                rec["solver_cross_check"] = {"paired": False}
            e1_bends.append(rec)
            if not (ft is not None and ft.kind == "ROLL"):
                solver_bends.append({"id": b.graph_bend_id, "source": "E1", "class": b.unsupported})
    for j in jbends:
        solver_bends.append({"id": j["id"], "source": "ROLL_SOLVER_JUNCTION", "class": None})
    roll_faces = {f for r in rolls for k in ("inner", "outer") for f in r["faces"][k]}
    e1_bend_faces = {f for b in e1_bends if b["class"] != "ROLLED" for k in ("inner", "outer") for f in b["faces"][k]}
    unclassified = []
    for ft in feats:
        if ft.kind is None:
            role = graph.face_role if graph is not None else {}
            unclassified.append({"faces": ft.faces, "surface": ft.family, "r_inner_mm": _r6(ft.r_in(ft.z_mid)),
                                 "thickness_mm": _r6(ft.thickness), "sweep_deg": _r6(math.degrees(ft.cover[1])),
                                 "e1_roles": sorted({"/".join(str(x) for x in role.get(f, ("unassigned",)))
                                                     for f in ft.faces})})
    # unpaired large curved skins (a roll modelled with one skin, or offset skins): reported, not rolls
    paired = {f for ft in feats for f in ft.faces}
    unpaired = []
    if t_part:
        for f, s in sorted(surfs.items()):
            if f in paired:
                continue
            p = _boundary_points(model.faces[f])
            zz = (p - s.origin) @ s.axis
            r = s.radius_at(float(np.mean(zz)), s.origin)
            if r > cfg.rolled_ri_over_t * t_part and float(zz.max() - zz.min()) > cfg.min_axial_overlap_t * t_part:
                unpaired.append({"face": f, "surface": s.family, "radius_mm": _r6(r), "convex": s.convex,
                                 "area_mm2": _r6(s.area)})
    # census of curved faces whose radius exceeds the roll threshold (screening): what each one became
    census = []
    if t_part:
        role = graph.face_role if graph is not None else {}
        owner = {f: (r["id"], r["kind"]) for r in rolls + jbends for k in ("inner", "outer") for f in r["faces"][k]}
        for f, s in sorted(surfs.items()):
            p = _boundary_points(model.faces[f])
            zz = (p - s.origin) @ s.axis
            r = s.radius_at(float(0.5 * (zz.min() + zz.max())), s.origin)
            if r > cfg.rolled_ri_over_t * t_part:
                census.append({"face": f, "surface": s.family, "radius_mm": _r6(r), "r_over_t": _r6(r / t_part),
                               "convex": s.convex, "area_mm2": _r6(s.area), "axial_extent_mm": _r6(float(zz.max() - zz.min())),
                               "solver": owner.get(f, ("not a roll skin", None))[0],
                               "e1_role": "/".join(str(x) for x in role.get(f, ("unassigned",)))})
    gv = _graph_view(graph, rolls, jbends, e1_bends)
    doc = {
        "schema": SCHEMA,
        "rule_version": RULE_VERSION,
        "source": source or {},
        "frame": "cad_model_mm (STEP frame, no transform)",
        "units": {"length": "mm", "angle": "deg"},
        "e1": {"flange_graph_error": graph_error, "part_model_error": datum_error,
               "thickness_mm": _r6(t_e1), "panels": len(graph.panels) if graph is not None else 0,
               "graph_bends": len(graph.bends) if graph is not None else 0,
               "features": [{"kind": f.kind, "reason": f.reason, "faces": list(f.face_indices)}
                            for f in (graph.features if graph is not None else [])]},
        "thickness": {"t_mm": _r6(t_part), "source": t_src},
        "summary": {"rolls": len(rolls), "junction_bends": len(jbends), "e1_bends": len(e1_bends),
                    "solver_bends": len(solver_bends), "cones": sum(r["is_cone"] for r in rolls),
                    "all_checks_pass": all(r["checks_pass"] for r in rolls + jbends),
                    "roll_faces_overlap_e1_bend_faces": sorted(roll_faces & e1_bend_faces)},
        "rolls": rolls,
        "junction_bends": jbends,
        "e1_bends": e1_bends,
        "solver_bends": solver_bends,
        "graph": gv,
        "unclassified_paired_features": unclassified,
        "unpaired_large_curved_skins": unpaired,
        "large_radius_curved_faces": census,
        "curved_faces": {"cylinder": sum(v.family == "cylinder" for v in surfs.values()),
                         "cone": sum(v.family == "cone" for v in surfs.values())},
    }
    return doc, render


def solve_step(path, cfg: SolverConfig | None = None, compile_cfg: CompileConfig | None = None):
    """STEP file -> (doc, render data, BrepModel)."""
    import hashlib
    from pathlib import Path

    from domains.production.engine.compile.brep import load_step

    ccfg = compile_cfg or CompileConfig()
    p = Path(path)
    model = load_step(p, linear_deflection=ccfg.mesh_linear_deflection_mm,
                      angular_deflection=ccfg.mesh_angular_deflection_rad)
    src = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    doc, render = solve(model, cfg, ccfg, src)
    return doc, render, model
