"""Flange graph of a folded sheet-metal solid (E1, plan §3.1).

Input: a :class:`~domains.production.engine.compile.brep.BrepModel`. Output: a
:class:`FlangeGraph` with

- **panels**: planar flange panels with both skins (outward normals, planes,
  measured thickness), boundary loops whose segments are labelled FREE (cut
  edge), BEND (tangent line of a graph bend), FEATURE (an unsupported feature)
  or INTERNAL (a split between coplanar faces of the same skin, removed);
- **graph bends**: coaxial inner/outer cylinder skins (radii ``Ri`` and
  ``Ro = Ri + t``) tangent to exactly two panels, with the cylinder axis, the
  plane-derived sweep, the axial extent and the per-feature UNSUPPORTED class;
- **walls**: thickness / cut-edge faces (they meet the skins at sharp edges);
- **features**: unsupported skin features (see :mod:`.unsupported`).

Topology, not heuristics on areas: a bend skin is a partial cylinder that meets
a planar face *smoothly along a generator* (the edge is a straight line parallel
to the axis and the outward normals agree). Inner and outer skins of one bend
are coaxial with radii differing by the thickness; the planes they touch at the
same angular end are the two skins of one panel. Everything that touches a
skin smoothly but is not a panel or bend skin is an unsupported feature; what
remains are walls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from domains.production.engine.compile.brep import BrepModel
from domains.production.engine.compile.config import CompileConfig
from domains.production.engine.compile.geometry import (
    bend_bisector,
    canonical_direction,
    line_distance_parallel,
    unit,
)
from domains.production.engine.compile.unsupported import (
    HEM_MIN_SWEEP_DEG,
    FeatureRecord,
    classify_bend,
    classify_skin_cluster,
    panel_feature_dominates,
)

__all__ = [
    "GraphError",
    "PanelSide",
    "BoundaryLoop",
    "GraphPanel",
    "BendEnd",
    "GraphBend",
    "FlangeGraph",
    "build_flange_graph",
    "LABEL_FREE",
    "LABEL_BEND",
    "LABEL_FEATURE",
]

LABEL_FREE = 0
LABEL_BEND = 1
LABEL_FEATURE = 2

_GENERATOR_ANGLE_DEG = 0.5
_END_CLUSTER_DEG = 1.0
_PARALLEL_FACE_DEG = 1.0
_COPLANAR_TOL_MM = 0.02


class GraphError(Exception):
    """The flange graph cannot be built (maps to a compile BLOCKER code)."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _UF:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for x in sorted(self.parent):
            out.setdefault(self.find(x), []).append(x)
        return out


@dataclass
class BoundaryLoop:
    """A closed boundary polygon of one panel skin (first point not repeated).

    ``labels[i]`` and ``refs[i]`` describe the segment from ``points[i]`` to
    ``points[i + 1]`` (wrapping): LABEL_FREE, LABEL_BEND (``refs`` = graph bend
    key) or LABEL_FEATURE (``refs`` = feature index).
    """

    points: np.ndarray
    labels: np.ndarray
    refs: np.ndarray
    is_outer: bool


@dataclass
class PanelSide:
    faces: list[int]
    normal: np.ndarray  # outward unit normal
    point: np.ndarray  # area-weighted centroid of the side's faces
    area: float
    loops: list[BoundaryLoop] = field(default_factory=list)


@dataclass
class GraphPanel:
    key: int
    sides: list[PanelSide]
    thickness: float | None
    panel_id: str = ""
    #: index into ``sides`` of skin 0 (assigned once the datum is chosen)
    skin0: int = 0
    #: one-sided footprint (mm2) of the unsupported features attached to this panel
    feature_area: float = 0.0
    #: kind with the largest footprint when the features dominate the panel (unsupported panel)
    dominant_feature: str | None = None

    @property
    def area(self) -> float:
        return float(np.mean([s.area for s in self.sides]))

    @property
    def mid_centroid(self) -> np.ndarray:
        return np.mean([s.point for s in self.sides], axis=0)

    def side_with_normal(self, n: np.ndarray) -> int | None:
        best, best_dot = None, math.cos(math.radians(5.0))
        for i, s in enumerate(self.sides):
            d = float(s.normal @ n)
            if d > best_dot:
                best, best_dot = i, d
        return best


@dataclass
class BendEnd:
    panel: int  # panel key
    radial: np.ndarray  # unit vector from the axis to this end's tangent lines
    inner_planes: list[int]
    outer_planes: list[int]


@dataclass
class GraphBend:
    key: int
    inner_faces: list[int]
    outer_faces: list[int]
    axis_point: np.ndarray
    axis_dir: np.ndarray  # canonical (sign-normalised) cylinder axis direction
    ri: float
    ro: float
    ends: tuple[BendEnd, BendEnd]
    z0: float
    z1: float
    sweep_deg: float
    cylinder_sweep_deg: float
    tangency_error_mm: float
    #: unit radial toward the angular middle of the bend's material (centroid of
    #: its cylinder faces); orients near-reverse (180 deg) and reflex bends
    mid_radial: np.ndarray | None = None
    unsupported: str | None = None
    unsupported_reason: str = ""
    graph_bend_id: str = ""
    #: offset-skin bends only (see _offset_skin_pieces): the outer skin cylinder's own axis point and radius;
    #: ``ro`` is then the effective ``Ri + t`` about the inner axis. None for coaxial skins.
    outer_axis_point: np.ndarray | None = None
    outer_radius: float | None = None

    @property
    def thickness(self) -> float:
        return self.ro - self.ri

    @property
    def length(self) -> float:
        return self.z1 - self.z0

    def bisector(self) -> np.ndarray:
        """Unit apex direction (angular middle of the bend) at the final pose."""
        return bend_bisector(self.ends[0].radial, self.ends[1].radial, self.axis_dir, self.sweep_deg,
                             self.mid_radial)

    def oml_sharp(self) -> tuple[np.ndarray, np.ndarray]:
        """OML virtual sharp line (start, end) at the final pose.

        For a hem (sweep >= ``HEM_MIN_SWEEP_DEG``) the virtual sharp lies at
        ``Ro / cos(sweep / 2)`` >= 11.5 Ro from the axis, or at infinity for
        180 deg, so the outer apex line stands in for it (ids and display).
        """
        m = self.bisector()
        if self.sweep_deg >= HEM_MIN_SWEEP_DEG:
            offset = self.ro * m
        else:
            offset = self.ro / math.cos(math.radians(self.sweep_deg) / 2.0) * m
        a = self.axis_point + self.z0 * self.axis_dir + offset
        b = self.axis_point + self.z1 * self.axis_dir + offset
        return a, b

    def end_for_panel(self, panel_key: int) -> BendEnd:
        for end in self.ends:
            if end.panel == panel_key:
                return end
        raise KeyError(panel_key)


@dataclass
class FlangeGraph:
    panels: dict[int, GraphPanel]
    bends: dict[int, GraphBend]
    walls: list[int]
    features: list[FeatureRecord]
    #: face index -> ('panel', key, side) | ('bend', key, 'inner'|'outer') | ('wall',) | ('feature', idx)
    face_role: dict[int, tuple]
    thickness: float | None
    thickness_spread: float
    notes: list[str] = field(default_factory=list)

    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        """panel key -> [(bend key, other panel key)]."""
        adj: dict[int, list[tuple[int, int]]] = {k: [] for k in self.panels}
        for bk, b in self.bends.items():
            p, q = b.ends[0].panel, b.ends[1].panel
            adj[p].append((bk, q))
            adj[q].append((bk, p))
        return adj


@dataclass
class _Tangency:
    cyl: int
    plane: int
    edge: int
    radial: np.ndarray
    length: float


def _radial(face, point: np.ndarray) -> np.ndarray:
    v = point - face.origin
    v = v - float(v @ face.axis) * face.axis
    return unit(v)


def _mid_radial(faces, face_ids: list[int], origin: np.ndarray, axis: np.ndarray) -> np.ndarray | None:
    """Unit radial toward the area-weighted centroid of a bend's cylinder faces (its angular middle).

    The centroid of a partial cylinder lies on the bisector of its angular range
    (for any range below 360 deg), so this radial is inside the bend's sweep.
    """
    w = np.array([faces[m].area for m in face_ids], dtype=np.float64)
    if not len(w) or float(w.sum()) <= 0:
        return None
    c = np.average([faces[m].centroid for m in face_ids], axis=0, weights=w)
    v = c - origin
    v = v - float(v @ axis) * axis
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _footprint(model: BrepModel, face_ids, normal: np.ndarray) -> float:
    """One-sided footprint (mm2) of a feature on a panel plane.

    The projected area of the tessellation triangles whose outward normals face
    one side of the plane, for the side where it is larger: a feature formed on
    one skin shows its footprint there, one through both skins about equally on
    both, and the total area of every face (both skins plus side walls) is
    never counted.
    """
    up = down = 0.0
    n = np.asarray(normal, dtype=np.float64)
    for f in face_ids:
        face = model.faces[f]
        if not len(face.mesh_triangles):
            continue
        tri = face.mesh_vertices[face.mesh_triangles]
        proj = 0.5 * (np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ n)
        up += float(np.sum(proj[proj > 0]))
        down -= float(np.sum(proj[proj < 0]))
    return max(up, down)


def _face_axial_range(model: BrepModel, face_index: int, origin: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    face = model.faces[face_index]
    pts = [u.xyz for loop in face.loops for u in loop.uses if len(u.xyz)]
    if not pts:
        z = float((face.centroid - origin) @ axis)
        return z, z
    p = np.concatenate(pts)
    z = (p - origin) @ axis
    return float(z.min()), float(z.max())


def _find_tangencies(model: BrepModel, cfg: CompileConfig) -> tuple[list[_Tangency], list[tuple[int, int, int]]]:
    """Smooth cylinder-plane junctions along generators, and smooth cylinder-cylinder junctions."""
    cos_gen = math.cos(math.radians(_GENERATOR_ANGLE_DEG))
    sin_gen = math.sin(math.radians(_GENERATOR_ANGLE_DEG))
    tangencies: list[_Tangency] = []
    cyl_cyl: list[tuple[int, int, int]] = []
    for edge in model.edges:
        if len(edge.faces) != 2 or edge.dihedral_deg is None or edge.dihedral_deg > cfg.smooth_dihedral_deg:
            continue
        fa, fb = edge.faces
        A, B = model.faces[fa], model.faces[fb]
        if A.kind == "cylinder" and B.kind == "cylinder":
            cyl_cyl.append((fa, fb, edge.index))
            continue
        for c, p in ((A, B), (B, A)):
            if c.kind != "cylinder" or p.kind != "plane" or c.axis is None or p.normal is None:
                continue
            if edge.curve != "line" or edge.direction is None:
                continue
            if abs(float(edge.direction @ c.axis)) < cos_gen or abs(float(p.normal @ c.axis)) > sin_gen:
                continue
            mid = 0.5 * (edge.points[0] + edge.points[-1])
            tangencies.append(_Tangency(c.index, p.index, edge.index, _radial(c, mid), edge.length))
    return tangencies, cyl_cyl


_SAME_RADIUS_MM = 0.01


def _offset_skin_pieces(model: BrepModel, cand: list[int], pieces: list[dict], features: list,
                        by_cyl: dict, cfg: CompileConfig) -> list[dict]:
    """Bends whose inner and outer skins are parallel cylinders that are **not coaxial**.

    Some CAD systems model a bend with an outer radius that is not ``Ri + t`` about the inner axis (the outer arc
    has its own centre, e.g. a fixed outer die radius), so the coaxial grouping leaves both skins without a
    partner. Such a pair is still one bend when the concave and the convex cylinder have parallel axes, overlap
    axially, and are tangent at **both** angular ends to planes with the same end radials (the two skins of the
    two flanges): the flanges, the sweep and the OML virtual sharp come from those planes, not from the arcs.
    The bend is described by its inner cylinder (axis, Ri) and ``ro_eff`` = the outer skin planes' distance from
    that axis (``Ri + t``), which must agree at both ends; anything less consistent stays unpaired (a feature
    or a wall, as before).
    """
    faces = model.faces
    used = {m for pc in pieces for m in pc["faces"]} | {m for f in features for m in f.face_indices}
    orphans = [c for c in cand if c not in used and len(by_cyl.get(c, ())) >= 2]
    cos_ax = math.cos(math.radians(cfg.coaxial_angle_deg))
    cos_end = math.cos(math.radians(_END_CLUSTER_DEG))

    def end_radials(c: int) -> list[np.ndarray] | None:
        rs = [t.radial for t in by_cyl[c]]
        labels = _cluster_radials(rs)
        if len(set(labels)) != 2:
            return None
        return [unit(np.mean([r for r, lab in zip(rs, labels) if lab == k], axis=0)) for k in sorted(set(labels))]

    out: list[dict] = []
    taken: set[int] = set()
    inners = [c for c in orphans if not faces[c].convex]
    outers = [c for c in orphans if faces[c].convex]
    for i in inners:
        fi = faces[i]
        ri_ends = end_radials(i)
        if ri_ends is None or i in taken:
            continue
        axis = canonical_direction(fi.axis)
        for o in outers:
            fo = faces[o]
            if o in taken or abs(float(fi.axis @ fo.axis)) < cos_ax:
                continue
            ro_ends = end_radials(o)
            if ro_ends is None:
                continue
            if not all(any(float(a @ b) > cos_end for b in ro_ends) for a in ri_ends):
                continue
            (a0, a1), (b0, b1) = _face_axial_range(model, i, fi.origin, axis), _face_axial_range(model, o, fi.origin, axis)
            if min(a1, b1) - max(a0, b0) <= 0.05:
                continue
            # the outer skin planes' distance from the inner axis, per end: one Ri + t for the whole bend
            dists = []
            for t in by_cyl[o]:
                pl = faces[t.plane]
                if pl.normal is None:
                    continue
                dists.append(abs(float((pl.centroid - fi.origin) @ pl.normal)))
            if len(dists) < 2 or max(dists) - min(dists) > _COPLANAR_TOL_MM + 1e-3 * max(dists):
                continue
            ro_eff = float(np.mean(dists))
            if not (0.05 <= ro_eff - fi.radius <= 30.0):
                continue
            ends: dict[int, dict] = {}
            for k, r in enumerate(ri_ends):
                ends[k] = {"inner": {t.plane for t in by_cyl[i] if float(t.radial @ r) > cos_end},
                           "outer": {t.plane for t in by_cyl[o] if float(t.radial @ r) > cos_end}, "radial": r}
            if not all(e["inner"] and e["outer"] for e in ends.values()):
                continue
            taken.update((i, o))
            z_range = {i: (a0, a1), o: (b0, b1)}
            out.append({"faces": [i, o], "inner": [i], "outer": [o], "axis": axis, "origin": fi.origin,
                        "ends": ends, "z": z_range, "ro_eff": ro_eff, "offset_skins": True})
            break
    return out


def _dominant_radius_class(faces, members: list[int]) -> list[int]:
    """The faces of ``members`` whose radius equals (within ``_SAME_RADIUS_MM``) that of the radius class with
    the largest total area (all of them when they share one radius)."""
    classes: list[list[int]] = []
    for m in sorted(members, key=lambda k: faces[k].radius):
        if classes and abs(faces[m].radius - faces[classes[-1][-1]].radius) <= _SAME_RADIUS_MM:
            classes[-1].append(m)
        else:
            classes.append([m])
    if len(classes) <= 1:
        return list(members)
    best = max(classes, key=lambda c: (sum(faces[k].area for k in c), -min(c)))
    return [m for m in members if m in best]


def _cluster_radials(radials: list[np.ndarray]) -> list[int]:
    """Greedy clustering of unit radials (cos > cos(1 deg)); returns a label per radial."""
    cos_t = math.cos(math.radians(_END_CLUSTER_DEG))
    centers: list[np.ndarray] = []
    labels = []
    for r in radials:
        for i, c in enumerate(centers):
            if float(r @ c) > cos_t:
                labels.append(i)
                break
        else:
            centers.append(r)
            labels.append(len(centers) - 1)
    return labels


def build_flange_graph(model: BrepModel, cfg: CompileConfig) -> FlangeGraph:
    """Build the flange graph (module docstring). Raises :class:`GraphError` on failure."""
    faces = model.faces
    notes: list[str] = []
    tangencies, cyl_cyl = _find_tangencies(model, cfg)
    by_cyl: dict[int, list[_Tangency]] = {}
    for t in tangencies:
        by_cyl.setdefault(t.cyl, []).append(t)

    # -- bend-skin candidates and coaxial groups --------------------------------
    cand = sorted(c for c in by_cyl if faces[c].angular_span < 2 * math.pi - 1e-6)
    cos_ax = math.cos(math.radians(cfg.coaxial_angle_deg))
    uf = _UF()
    for c in cand:
        uf.find(c)
    for i, c1 in enumerate(cand):
        f1 = faces[c1]
        for c2 in cand[i + 1:]:
            f2 = faces[c2]
            if abs(float(f1.axis @ f2.axis)) < cos_ax:
                continue
            if line_distance_parallel(f1.origin, f1.axis, f2.origin) > cfg.coaxial_offset_mm:
                continue
            uf.union(c1, c2)
    groups = list(uf.groups().values())

    features: list[FeatureRecord] = []
    pieces: list[dict] = []
    for members in groups:
        ref = faces[members[0]]
        axis = canonical_direction(ref.axis)
        origin = ref.origin
        inner = [m for m in members if not faces[m].convex]
        outer = [m for m in members if faces[m].convex]
        z_range = {m: _face_axial_range(model, m, origin, axis) for m in members}
        # pair inner/outer faces sharing a tangent end and overlapping axially
        pair_uf = _UF()
        for m in members:
            pair_uf.find(m)
        cos_end = math.cos(math.radians(_END_CLUSTER_DEG))
        for i in inner:
            for o in outer:
                ri, ro = faces[i].radius, faces[o].radius
                if not (ro > ri and 0.05 <= ro - ri <= 30.0):
                    continue
                (a0, a1), (b0, b1) = z_range[i], z_range[o]
                if min(a1, b1) - max(a0, b0) <= 0.05:
                    continue
                if any(float(ti.radial @ to.radial) > cos_end for ti in by_cyl[i] for to in by_cyl[o]):
                    pair_uf.union(i, o)
        for comp in pair_uf.groups().values():
            ins = [m for m in comp if not faces[m].convex]
            outs = [m for m in comp if faces[m].convex]
            if not ins or not outs:
                # no coaxial partner skin: left to the skin-cluster pass. Tangent to a panel skin it is a
                # curl (ROLLED b) or part of the formed feature around it; tangent only to walls (an
                # outline corner or slot round) it stays a wall
                continue
            # one skin radius per side: coaxial cylinders of another radius on the same side are a stepped
            # skin (a thinner or thicker strip across the width, e.g. a machined or coined step). The sheet's
            # own skin is the radius class with the largest area; the other levels (and the planes they are
            # tangent to) are an unsupported feature, so they never pair non-coplanar planes into one panel.
            stepped: list[int] = []
            for side in (ins, outs):
                keep = _dominant_radius_class(faces, side)
                stepped += [m for m in side if m not in keep]
                side[:] = keep
            if stepped:
                features.append(FeatureRecord(kind="OTHER", reason="stepped bend skin: coaxial cylinders of "
                                              "another radius on one side", face_indices=tuple(sorted(stepped)),
                                              area_mm2=sum(faces[m].area for m in stepped)))
                comp = [m for m in comp if m not in stepped]
            radials = [t.radial for m in comp for t in by_cyl[m]]
            owners = [(m, t) for m in comp for t in by_cyl[m]]
            labels = _cluster_radials(radials)
            ends: dict[int, dict[str, set[int] | np.ndarray]] = {}
            for (m, t), lab in zip(owners, labels):
                e = ends.setdefault(lab, {"inner": set(), "outer": set(), "radial": t.radial})
                (e["outer"] if faces[m].convex else e["inner"]).add(t.plane)  # type: ignore[union-attr]
            pieces.append({"faces": comp, "inner": ins, "outer": outs, "axis": axis, "origin": origin,
                           "ends": ends, "z": z_range})
    pieces += _offset_skin_pieces(model, cand, pieces, features, by_cyl, cfg)

    # -- panels: planes paired through bend ends, plus coplanar splits ---------------
    puf = _UF()
    valid_pieces = []
    for piece in pieces:
        ends = piece["ends"]
        ok = len(ends) == 2 and all(e["inner"] and e["outer"] for e in ends.values())
        if not ok:
            reason = (f"bend cylinders meet {len(ends)} tangent ends" if len(ends) != 2
                      else "a bend end lacks a tangent plane on one skin")
            kind = "ROLLED" if len(ends) == 1 else "OTHER"
            features.append(FeatureRecord(kind=kind, reason=reason, face_indices=tuple(piece["faces"]),
                                          area_mm2=sum(faces[m].area for m in piece["faces"])))
            continue
        for e in ends.values():
            planes = sorted(e["inner"] | e["outer"])
            for p in planes:
                puf.find(p)
            for p in planes[1:]:
                puf.union(planes[0], p)
        valid_pieces.append(piece)
    panel_planes = set(puf.parent)
    for edge in model.edges:
        if len(edge.faces) != 2 or edge.dihedral_deg is None or edge.dihedral_deg > cfg.smooth_dihedral_deg:
            continue
        fa, fb = edge.faces
        A, B = faces[fa], faces[fb]
        if A.kind == "plane" and B.kind == "plane" and (fa in panel_planes or fb in panel_planes):
            if float(A.normal @ B.normal) > math.cos(math.radians(_PARALLEL_FACE_DEG)) and abs(
                float((B.centroid - A.centroid) @ A.normal)) <= _COPLANAR_TOL_MM:
                puf.union(fa, fb)
                panel_planes.update((fa, fb))
    # iterate coplanar growth until stable (chains of split faces)
    changed = True
    while changed:
        changed = False
        for edge in model.edges:
            if len(edge.faces) != 2 or edge.dihedral_deg is None or edge.dihedral_deg > cfg.smooth_dihedral_deg:
                continue
            fa, fb = edge.faces
            A, B = faces[fa], faces[fb]
            if A.kind != "plane" or B.kind != "plane" or (fa in panel_planes) == (fb in panel_planes):
                continue
            if float(A.normal @ B.normal) > math.cos(math.radians(_PARALLEL_FACE_DEG)) and abs(
                float((B.centroid - A.centroid) @ A.normal)) <= _COPLANAR_TOL_MM:
                puf.union(fa, fb)
                panel_planes.update((fa, fb))
                changed = True

    panels: dict[int, GraphPanel] = {}
    plane_to_panel: dict[int, int] = {}
    for key, members in sorted(puf.groups().items()):
        ref = max(members, key=lambda m: (faces[m].area, -m))
        n_ref = faces[ref].normal
        cos_p = math.cos(math.radians(_PARALLEL_FACE_DEG))
        side_a = [m for m in members if float(faces[m].normal @ n_ref) > cos_p]
        side_b = [m for m in members if float(faces[m].normal @ n_ref) < -cos_p]
        if len(side_a) + len(side_b) != len(members):
            raise GraphError("FLANGE_GRAPH_NOT_A_TREE",
                             f"panel faces {members} are not parallel (bend ends pair non-parallel skins)")
        sides = []
        for group in (side_a, side_b):
            if not group:
                continue
            w = np.array([faces[m].area for m in group])
            point = np.average([faces[m].centroid for m in group], axis=0, weights=w)
            normal = unit(np.average([faces[m].normal for m in group], axis=0, weights=w))
            for m in group:
                off = abs(float((faces[m].centroid - point) @ normal))
                if off > max(_COPLANAR_TOL_MM, 1e-4 * max(1.0, float(np.linalg.norm(point)))):
                    raise GraphError("FLANGE_GRAPH_NOT_A_TREE", f"panel skin faces {group} are not coplanar")
            sides.append(PanelSide(faces=sorted(group), normal=normal, point=point, area=float(w.sum())))
        thickness = None
        if len(sides) == 2:
            thickness = abs(float((sides[1].point - sides[0].point) @ sides[0].normal))
        panel = GraphPanel(key=len(panels), sides=sides, thickness=thickness)
        panels[panel.key] = panel
        for m in members:
            plane_to_panel[m] = panel.key

    # -- graph bends ---------------------------------------------------------------
    raw_bends: list[GraphBend] = []
    for piece in valid_pieces:
        ends_raw = list(piece["ends"].values())
        ends = []
        for e in ends_raw:
            pk = {plane_to_panel[p] for p in (e["inner"] | e["outer"])}
            if len(pk) != 1:
                raise GraphError("FLANGE_GRAPH_NOT_A_TREE", "a bend end touches several panels")
            ends.append(BendEnd(panel=pk.pop(), radial=np.asarray(e["radial"], dtype=np.float64),
                                inner_planes=sorted(e["inner"]), outer_planes=sorted(e["outer"])))
        if ends[0].panel == ends[1].panel:
            features.append(FeatureRecord(kind="OTHER", reason="bend cylinders return to the same panel",
                                          face_indices=tuple(piece["faces"]),
                                          area_mm2=sum(faces[m].area for m in piece["faces"])))
            continue
        ins, outs = piece["inner"], piece["outer"]
        ri = float(np.mean([faces[m].radius for m in ins]))
        # an offset-skin bend (see _offset_skin_pieces) is described by its inner cylinder and the outer skin
        # planes' distance from that axis (Ri + t), the coaxial bend with the same skin planes
        ro = float(piece["ro_eff"]) if "ro_eff" in piece else float(np.mean([faces[m].radius for m in outs]))
        axis, origin = piece["axis"], piece["origin"]
        zs = [piece["z"][m] for m in piece["faces"]]
        z0, z1 = min(z for z, _ in zs), max(z for _, z in zs)
        r1, r2 = ends[0].radial, ends[1].radial
        cyl_sweep = max(math.degrees(faces[m].angular_span) for m in piece["faces"])
        # the end radials give the sweep up to the reflex ambiguity (acos is in [0, 180]):
        # a teardrop hem sweeps 360 - acos, which its cylinder faces' angular span decides
        acute = math.degrees(math.acos(float(np.clip(r1 @ r2, -1.0, 1.0))))
        sweep = acute if abs(cyl_sweep - acute) <= abs(cyl_sweep - (360.0 - acute)) else 360.0 - acute
        raw_bends.append(GraphBend(key=-1, inner_faces=sorted(ins), outer_faces=sorted(outs), axis_point=origin,
                                   axis_dir=axis, ri=ri, ro=ro, ends=(ends[0], ends[1]), z0=z0, z1=z1,
                                   sweep_deg=sweep, cylinder_sweep_deg=cyl_sweep, tangency_error_mm=0.0,
                                   mid_radial=_mid_radial(faces, piece["faces"], origin, axis),
                                   outer_axis_point=faces[outs[0]].origin if "ro_eff" in piece else None,
                                   outer_radius=float(faces[outs[0]].radius) if "ro_eff" in piece else None))
    # merge split pieces of one bend (same panels, coaxial, same radii)
    merged = _merge_same_panel_pieces(raw_bends, cfg)
    merged = _break_cycles(merged, panels, plane_to_panel, faces, cfg, notes)
    bends: dict[int, GraphBend] = {}
    for b in merged:
        b.key = len(bends)
        # tangency of the panels against the cylinder radii
        errs = []
        for end in b.ends:
            for plane_faces, radius, sign in ((end.outer_planes, b.ro, 1.0), (end.inner_planes, b.ri, -1.0)):
                for f in plane_faces:
                    face = faces[f]
                    # outer skin: outward normal = +radial, plane at +Ro; inner skin: normal = -radial, at Ri
                    if float(face.normal @ end.radial) * sign < math.cos(math.radians(2.0)):
                        errs.append(float("inf"))
                        continue
                    d = float((face.centroid - b.axis_point) @ face.normal)
                    errs.append(abs(d - sign * radius))
        b.tangency_error_mm = max(errs) if errs else 0.0
        kind, reason = classify_bend(sweep_deg=b.sweep_deg, inner_radius_mm=b.ri, thickness_mm=b.thickness,
                                     tangency_error_mm=b.tangency_error_mm,
                                     sweep_mismatch_deg=abs(b.cylinder_sweep_deg - b.sweep_deg))
        b.unsupported, b.unsupported_reason = kind, reason
        bends[b.key] = b

    # -- roles of every face ------------------------------------------------------
    face_role: dict[int, tuple] = {}
    for pk, panel in panels.items():
        for si, side in enumerate(panel.sides):
            for f in side.faces:
                face_role[f] = ("panel", pk, si)
    for bk, b in bends.items():
        for f in b.inner_faces:
            face_role[f] = ("bend", bk, "inner")
        for f in b.outer_faces:
            face_role[f] = ("bend", bk, "outer")
    for fi, feat in enumerate(features):
        for f in feat.face_indices:
            face_role.setdefault(f, ("feature", fi))
    # skin-attached clusters of unassigned faces (smooth junctions) -> features
    unassigned = [f.index for f in faces if f.index not in face_role]
    cuf = _UF()
    for f in unassigned:
        cuf.find(f)
    attached: dict[int, list[tuple[int, float]]] = {}
    for edge in model.edges:
        if len(edge.faces) != 2 or edge.dihedral_deg is None or edge.dihedral_deg > cfg.smooth_dihedral_deg:
            continue
        fa, fb = edge.faces
        ra, rb = face_role.get(fa), face_role.get(fb)
        if ra is None and rb is None:
            cuf.union(fa, fb)
        elif (ra is None) != (rb is None):
            free, owner = (fa, fb) if ra is None else (fb, fa)
            attached.setdefault(free, []).append((owner, edge.length))
    for root, members in cuf.groups().items():
        links = [(o, ln) for m in members for (o, ln) in attached.get(m, [])]
        skin_links = [(o, ln) for o, ln in links if face_role.get(o, ("",))[0] in ("panel", "bend")]
        if not skin_links:
            continue
        fams = {faces[m].surface if faces[m].kind != "plane" else "plane" for m in members}
        has_cyl = any(faces[m].kind == "cylinder" for m in members)
        has_plane = any(faces[m].kind == "plane" for m in members)
        kind, reason = classify_skin_cluster(fams - {"plane", "cylinder"}, has_unpaired_cylinder=has_cyl,
                                             has_offset_plane=has_plane)
        weight: dict[tuple, float] = {}
        for o, ln in skin_links:
            weight[face_role[o][:2]] = weight.get(face_role[o][:2], 0.0) + ln
        owner_role = max(sorted(weight), key=lambda k: weight[k])
        rec = FeatureRecord(kind=kind, reason=reason, face_indices=tuple(sorted(members)),
                            area_mm2=float(sum(faces[m].area for m in members)))
        if owner_role[0] == "panel":
            rec.panel_key = owner_role[1]
        else:
            rec.bend_key = owner_role[1]
        features.append(rec)
        for m in members:
            face_role[m] = ("feature", len(features) - 1)
    # owners of the early features (bend pieces that are no graph bend): the panel of a tangent plane
    for fi, feat in enumerate(features):
        if feat.panel_key is None and feat.bend_key is None:
            owners = []
            for f in feat.face_indices:
                for _, g in model.neighbors(f):
                    role = face_role.get(g)
                    if role and role[0] == "panel":
                        owners.append(role[1])
            if owners:
                feat.panel_key = max(sorted(set(owners)), key=owners.count)
    walls = sorted(f.index for f in faces if f.index not in face_role)
    for f in walls:
        face_role[f] = ("wall",)
    footprints: dict[int, dict[str, float]] = {}
    for feat in features:
        if feat.panel_key is not None:
            feat.footprint_mm2 = _footprint(model, feat.face_indices, panels[feat.panel_key].sides[0].normal)
            kinds = footprints.setdefault(feat.panel_key, {})
            kinds[feat.kind] = kinds.get(feat.kind, 0.0) + feat.footprint_mm2
    for pk, kinds in sorted(footprints.items()):
        panel = panels[pk]
        panel.feature_area = float(sum(kinds.values()))
        if panel_feature_dominates(panel.feature_area, panel.area):
            panel.dominant_feature = max(sorted(kinds), key=lambda k: kinds[k])

    # -- thickness ----------------------------------------------------------------
    values = [p.thickness for p in panels.values() if p.thickness is not None]
    values += [b.thickness for b in bends.values() if b.unsupported is None]
    thickness = float(np.median(values)) if values else None
    spread = float(max(values) - min(values)) if values else 0.0

    graph = FlangeGraph(panels=panels, bends=bends, walls=walls, features=features, face_role=face_role,
                        thickness=thickness, thickness_spread=spread, notes=notes)
    _label_boundaries(model, graph)
    _assign_ids(graph)
    if cyl_cyl:
        notes.append(f"{len(cyl_cyl)} smooth cylinder-cylinder junctions")
    return graph


def _collinear(a: GraphBend, b: GraphBend, cfg: CompileConfig) -> bool:
    if abs(float(a.axis_dir @ b.axis_dir)) < math.cos(math.radians(cfg.coaxial_angle_deg)):
        return False
    if line_distance_parallel(a.axis_point, a.axis_dir, b.axis_point) > cfg.coaxial_offset_mm:
        return False
    return abs(a.ri - b.ri) <= 0.01 and abs(a.ro - b.ro) <= 0.01


def _absorb(m: GraphBend, b: GraphBend) -> None:
    """Merge bend piece ``b`` into ``m`` (same two panels, collinear)."""
    shift = float((b.axis_point - m.axis_point) @ m.axis_dir)
    if float(b.axis_dir @ m.axis_dir) < 0:
        lo, hi = shift - b.z1, shift - b.z0
    else:
        lo, hi = b.z0 + shift, b.z1 + shift
    m.z0, m.z1 = min(m.z0, lo), max(m.z1, hi)
    m.inner_faces = sorted(set(m.inner_faces) | set(b.inner_faces))
    m.outer_faces = sorted(set(m.outer_faces) | set(b.outer_faces))
    for end in m.ends:
        other = b.end_for_panel(end.panel)
        end.inner_planes = sorted(set(end.inner_planes) | set(other.inner_planes))
        end.outer_planes = sorted(set(end.outer_planes) | set(other.outer_planes))
    m.cylinder_sweep_deg = max(m.cylinder_sweep_deg, b.cylinder_sweep_deg)


def _merge_same_panel_pieces(raw: list[GraphBend], cfg: CompileConfig) -> list[GraphBend]:
    merged: list[GraphBend] = []
    for b in raw:
        for m in merged:
            if {b.ends[0].panel, b.ends[1].panel} == {m.ends[0].panel, m.ends[1].panel} and _collinear(m, b, cfg):
                _absorb(m, b)
                break
        else:
            merged.append(b)
    return merged


def _coplanar_panels(p: GraphPanel, q: GraphPanel) -> bool:
    if len(p.sides) != len(q.sides):
        return False
    cos_p = math.cos(math.radians(0.5))
    for sp in p.sides:
        match = [sq for sq in q.sides if float(sp.normal @ sq.normal) > cos_p
                 and abs(float((sq.point - sp.point) @ sp.normal)) <= _COPLANAR_TOL_MM]
        if not match:
            return False
    return True


def _connected_without(bends: list[GraphBend], a: int, b: int, skip: set[int]) -> bool:
    adj: dict[int, list[int]] = {}
    for i, g in enumerate(bends):
        if i in skip:
            continue
        p, q = g.ends[0].panel, g.ends[1].panel
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)
    seen, stack = {a}, [a]
    while stack:
        cur = stack.pop()
        if cur == b:
            return True
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _break_cycles(bends: list[GraphBend], panels: dict[int, GraphPanel], plane_to_panel: dict[int, int], faces,
                  cfg: CompileConfig, notes: list[str]) -> list[GraphBend]:
    """Resolve cycles made by a panel cut into coplanar pieces (a full-width window).

    When two collinear bend pieces (same axis line, same radii and sweep) share
    one panel S and their far panels X1, X2 are coplanar *and* still connected
    without those two pieces, the pieces are one press stroke and X1, X2 one
    rigid panel with a window: X2 is merged into X1 and the pieces into one bend.
    Collinear tabs that are connected only through S (no cycle) stay separate.
    """
    changed = True
    while changed:
        changed = False
        for i, b1 in enumerate(bends):
            for j in range(i + 1, len(bends)):
                b2 = bends[j]
                if not _collinear(b1, b2, cfg) or abs(b1.sweep_deg - b2.sweep_deg) > 0.5:
                    continue
                shared = {b1.ends[0].panel, b1.ends[1].panel} & {b2.ends[0].panel, b2.ends[1].panel}
                if len(shared) != 1:
                    continue
                s_key = next(iter(shared))
                x1 = next(e.panel for e in b1.ends if e.panel != s_key)
                x2 = next(e.panel for e in b2.ends if e.panel != s_key)
                if not _coplanar_panels(panels[x1], panels[x2]):
                    continue
                if not _connected_without(bends, x1, x2, {i, j}):
                    continue
                _merge_panels(panels, plane_to_panel, faces, x1, x2, bends)
                _absorb(b1, b2)
                del bends[j]
                notes.append(f"cycle broken: coplanar panels merged through collinear bend pieces")
                bends[:] = _merge_same_panel_pieces(bends, cfg)
                changed = True
                break
            if changed:
                break
    return bends


def _merge_panels(panels: dict[int, GraphPanel], plane_to_panel: dict[int, int], faces, keep: int, drop: int,
                  bends: list[GraphBend]) -> None:
    pk, pd = panels[keep], panels[drop]
    cos_p = math.cos(math.radians(0.5))
    for sd in pd.sides:
        sk = next(s for s in pk.sides if float(s.normal @ sd.normal) > cos_p)
        members = sorted(set(sk.faces) | set(sd.faces))
        w = np.array([faces[m].area for m in members])
        sk.point = np.average([faces[m].centroid for m in members], axis=0, weights=w)
        sk.faces = members
        sk.area = float(w.sum())
    for f, k in list(plane_to_panel.items()):
        if k == drop:
            plane_to_panel[f] = keep
    for g in bends:
        for end in g.ends:
            if end.panel == drop:
                end.panel = keep
    del panels[drop]


def _label_boundaries(model: BrepModel, graph: FlangeGraph) -> None:
    """Merge each panel side's faces into closed boundary loops with segment labels."""
    faces = model.faces
    for panel in graph.panels.values():
        for si, side in enumerate(panel.sides):
            members = set(side.faces)
            uses = []
            for f in side.faces:
                for loop in faces[f].loops:
                    for use in loop.uses:
                        if use.degenerate or use.edge < 0 or len(use.xyz) < 2:
                            continue
                        others = [g for g in model.edges[use.edge].faces if g != f]
                        if others and all(g in members for g in others):
                            continue  # internal split between coplanar faces of this skin
                        label, ref = LABEL_FREE, -1
                        for g in others:
                            role = graph.face_role.get(g)
                            if role and role[0] == "bend":
                                label, ref = LABEL_BEND, role[1]
                            elif role and role[0] == "feature" and label == LABEL_FREE:
                                label, ref = LABEL_FEATURE, role[1]
                        uses.append((use, label, ref))
            side.loops = _chain_loops(uses, side.normal)


def _chain_loops(uses, normal: np.ndarray) -> list[BoundaryLoop]:
    """Chain oriented edge uses into closed loops through shared B-rep vertices."""
    by_start: dict[int, list[int]] = {}
    for i, (use, _, _) in enumerate(uses):
        by_start.setdefault(use.v_start, []).append(i)
    done = [False] * len(uses)
    loops: list[BoundaryLoop] = []
    for i in range(len(uses)):
        if done[i]:
            continue
        chain = []
        cur = i
        while cur is not None and not done[cur]:
            done[cur] = True
            chain.append(cur)
            nxt = [j for j in by_start.get(uses[cur][0].v_end, []) if not done[j]]
            cur = nxt[0] if nxt else None
        pts, labels, refs = [], [], []
        for j in chain:
            use, label, ref = uses[j]
            seg = use.xyz[:-1]
            pts.append(seg)
            labels.extend([label] * len(seg))
            refs.extend([ref] * len(seg))
        if not pts:
            continue
        p = np.concatenate(pts)
        if len(p) < 3:
            continue
        loops.append(BoundaryLoop(points=p, labels=np.asarray(labels, dtype=np.int64),
                                  refs=np.asarray(refs, dtype=np.int64), is_outer=False))
    if loops:
        x_axis = unit(np.cross(normal, [1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0.0, 1.0, 0.0]))
        y_axis = np.cross(normal, x_axis)
        areas = []
        for loop in loops:
            q = np.stack([loop.points @ x_axis, loop.points @ y_axis], axis=1)
            x, y = q[:, 0], q[:, 1]
            areas.append(abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))))
        outer = int(np.argmax(areas))
        loops[outer].is_outer = True
        loops.sort(key=lambda lp: not lp.is_outer)
    return loops


def _assign_ids(graph: FlangeGraph) -> None:
    """Deterministic ids: panels by mid-thickness centroid, bends by OML sharp midpoint."""
    def key3(v: np.ndarray) -> tuple[float, float, float]:
        return tuple(round(float(c), 3) + 0.0 for c in v)  # type: ignore[return-value]

    order = sorted(graph.panels.values(), key=lambda p: (key3(p.mid_centroid), p.key))
    for i, panel in enumerate(order):
        panel.panel_id = f"P{i}"
    border = sorted(graph.bends.values(), key=lambda b: (key3(0.5 * sum(b.oml_sharp())), b.key))
    for i, bend in enumerate(border, start=1):
        bend.graph_bend_id = f"G{i}"
