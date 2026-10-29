"""Rooted flange tree with skins, oriented bend axes and loci (E1).

Built from a :class:`~.flange_graph.FlangeGraph`:

- **datum**: the panel that never moves in the stage models (the tree root).
  Every panel is rigid at every stage, so all qualify; the largest (mean skin
  area) is chosen, ties broken by the lowest panel id (plan §3.1). When some
  graph bends will not be recipe bends (unmarked unsupported bends, kept
  rigid), the datum is restricted to the component that holds every recipe
  bend, because a compiled bend's parent chain must reach the datum through
  compiled bends only; if no such component exists the graph is refused.
- **tree**: breadth-first from the datum; each graph bend gets a parent and a
  child panel. Cycles or a disconnected graph -> ``FLANGE_GRAPH_NOT_A_TREE``.
- **skins** (ENGINE_IO §1): skin 0 of the datum is the side whose outward
  normal has the larger dot with +Z (ties +Y, then +X); skin 0 continues across
  every bend (it stays on the same cylinder side), so each bend's ``oml_skin``
  is the skin index of its convex (outer-cylinder) side and ``direction`` is
  UP when the OML is skin 1 (the child rises toward skin 0's normal), else DOWN.
- **oriented axis**: ``unit(cross(-p_out, c_out))`` where ``p_out``/``c_out`` are
  the parent/child in-plane outward directions (perpendicular to the axis,
  pointing away from the bend): a +sweep right-hand rotation about it takes
  the child from flat to formed (the contract's ``AxisV1`` rule). At sweeps
  >= ``REGULAR_SWEEP_MAX_DEG`` (179 deg: near-reverse hems, 180 deg hems,
  reflex teardrops) ``cross(-p_out, c_out)`` vanishes or flips, so the axis is
  oriented toward the material's angular middle (``GraphBend.mid_radial``) and
  ``p_out = -(axis x r_parent)``, ``c_out = axis x r_child``, which equals the
  regular construction wherever both are defined.
- **loci** at the final pose: cylinder axis, inner/outer tangent lines of both
  panels, IML/OML virtual sharps and inner/outer apex, all spanning the bend's
  axial extent. The OML virtual sharp is the published bend line; the cylinder
  axis never is. For a **hem** (sweep >= ``HEM_MIN_SWEEP_DEG`` = 170 deg, always
  unsupported) the virtual sharps lie at ``R / cos(sweep / 2)`` >= 11.5 R from
  the axis, or at infinity at 180 deg; the inner/outer apex lines stand in for
  the IML/OML virtual sharps there. Such a bend never has a BEND_LINE_POSITION
  nominal (UNSUPPORTED_FEATURE), so no measurement depends on the substitute.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from domains.production.engine.compile.flange_graph import FlangeGraph, GraphBend, GraphError, GraphPanel
from domains.production.engine.compile.geometry import (
    REGULAR_SWEEP_MAX_DEG,
    bend_bisector,
    oriented_bend_axis,
    unit,
)
from domains.production.engine.compile.unsupported import HEM_MIN_SWEEP_DEG

__all__ = ["ModelBend", "ModelPanel", "PartModel", "build_part_model", "skin0_rule"]


def skin0_rule(n_a: np.ndarray, n_b: np.ndarray) -> int:
    """Index (0 or 1) of the datum side that is skin 0: larger +Z, ties +Y then +X."""
    for axis in (2, 1, 0):
        d = float(n_a[axis] - n_b[axis])
        if abs(d) >= 1e-6:
            return 0 if d > 0 else 1
    return 0


@dataclass
class ModelPanel:
    key: int
    panel_id: str
    graph: GraphPanel
    #: skin index (0/1) -> index into graph.sides, or None when that skin face is absent
    skin_side: dict[int, int | None]
    parent_bend: int | None = None  # graph bend key
    parent_panel: int | None = None
    depth: int = 0

    def side(self, skin: int):
        idx = self.skin_side.get(skin)
        return None if idx is None else self.graph.sides[idx]

    def skin_of_side(self, side_index: int) -> int:
        for skin, idx in self.skin_side.items():
            if idx == side_index:
                return skin
        raise KeyError(side_index)

    @property
    def area(self) -> float:
        s0 = self.side(0) or self.side(1)
        return float(s0.area)


@dataclass
class ModelBend:
    key: int
    graph_bend_id: str
    graph: GraphBend
    parent: int
    child: int
    axis_point: np.ndarray
    axis_dir: np.ndarray  # oriented (+sweep: flat -> formed)
    z0: float
    z1: float
    ri: float
    ro: float
    sweep_deg: float
    r_parent: np.ndarray  # radial unit vector to the parent's tangent lines
    r_child: np.ndarray
    p_out: np.ndarray  # parent in-plane outward direction (away from the bend)
    c_out: np.ndarray
    oml_skin: int
    direction: str
    unsupported: str | None
    unsupported_reason: str
    #: unit apex direction (angular middle of the bend) at the final pose
    mid_dir: np.ndarray | None = None
    bend_uid: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def thickness(self) -> float:
        return self.ro - self.ri

    @property
    def sweep_rad(self) -> float:
        return math.radians(self.sweep_deg)

    @property
    def length(self) -> float:
        return self.z1 - self.z0

    def neutral_length(self, k_factor: float) -> float:
        return self.sweep_rad * (self.ri + k_factor * self.thickness)

    def outward_for(self, panel_key: int) -> np.ndarray:
        if panel_key == self.parent:
            return self.p_out
        if panel_key == self.child:
            return self.c_out
        raise KeyError(panel_key)

    def radial_for(self, panel_key: int) -> np.ndarray:
        if panel_key == self.parent:
            return self.r_parent
        if panel_key == self.child:
            return self.r_child
        raise KeyError(panel_key)

    def _line(self, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.axis_point + self.z0 * self.axis_dir + offset
        b = self.axis_point + self.z1 * self.axis_dir + offset
        return a, b

    @property
    def has_virtual_sharp(self) -> bool:
        """False for hems (sweep >= HEM_MIN_SWEEP_DEG): the apex stands in for the virtual sharps."""
        return self.sweep_deg < HEM_MIN_SWEEP_DEG

    def _sharp_scale(self) -> float:
        return 1.0 / math.cos(self.sweep_rad / 2.0) if self.has_virtual_sharp else 1.0

    def bisector(self) -> np.ndarray:
        if self.mid_dir is not None:
            return self.mid_dir
        return unit(self.r_parent + self.r_child)

    def loci(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        m = self.bisector()
        sec = self._sharp_scale()
        return {
            "cylinder_axis": self._line(np.zeros(3)),
            "inner_tangent_parent": self._line(self.ri * self.r_parent),
            "inner_tangent_child": self._line(self.ri * self.r_child),
            "outer_tangent_parent": self._line(self.ro * self.r_parent),
            "outer_tangent_child": self._line(self.ro * self.r_child),
            "iml_virtual_sharp": self._line(self.ri * sec * m),
            "oml_virtual_sharp": self._line(self.ro * sec * m),
            "inner_apex": self._line(self.ri * m),
            "outer_apex": self._line(self.ro * m),
        }

    def oml_sharp(self) -> tuple[np.ndarray, np.ndarray]:
        """The OML virtual sharp (the outer apex for a hem; module docstring)."""
        return self._line(self.ro * self._sharp_scale() * self.bisector())

    def skin_side_index(self, panel_key: int, convex: bool, graph: FlangeGraph) -> int:
        """Side index (into graph.sides) of ``panel_key`` on this bend's outer (convex) or inner side."""
        end = self.graph.end_for_panel(panel_key)
        faces = end.outer_planes if convex else end.inner_planes
        role = graph.face_role[faces[0]]
        return int(role[2])


@dataclass
class PartModel:
    graph: FlangeGraph
    panels: dict[int, ModelPanel]
    bends: dict[int, ModelBend]
    datum: int
    order: list[int]  # panel keys in BFS order from the datum
    thickness: float

    def panel_by_id(self, panel_id: str) -> ModelPanel:
        for p in self.panels.values():
            if p.panel_id == panel_id:
                return p
        raise KeyError(panel_id)

    def path_bends(self, panel_key: int) -> list[int]:
        """Graph bend keys from the datum to ``panel_key`` (datum side first)."""
        out = []
        cur = panel_key
        while self.panels[cur].parent_bend is not None:
            out.append(self.panels[cur].parent_bend)
            cur = self.panels[cur].parent_panel
        return out[::-1]

    def children(self, panel_key: int) -> list[int]:
        return [b.key for b in self.bends.values() if b.parent == panel_key]

    def subtree_panels(self, bend_key: int) -> list[int]:
        """Panels on the child side of a bend (including the child)."""
        out, stack = [], [self.bends[bend_key].child]
        while stack:
            p = stack.pop()
            out.append(p)
            stack.extend(self.bends[b].child for b in self.children(p))
        return sorted(out)


def _datum_candidates(graph: FlangeGraph, rigid: set[int]) -> list[int]:
    """Panels allowed as datum when the ``rigid`` graph bends will not be compiled.

    A compiled bend's parent chain must reach the datum through compiled bends
    only (``CompiledPartV1`` tree rule), so after removing the rigid bends all
    compiled bends must lie in one component, and the datum must be in it.
    """
    if not rigid:
        return sorted(graph.panels)
    parent = {k: k for k in graph.panels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bk, b in graph.bends.items():
        if bk not in rigid:
            parent[find(b.ends[0].panel)] = find(b.ends[1].panel)
    comps = {find(b.ends[0].panel) for bk, b in graph.bends.items() if bk not in rigid}
    if len(comps) > 1:
        names = sorted(graph.bends[k].graph_bend_id for k in rigid)
        raise GraphError("FLANGE_GRAPH_NOT_A_TREE",
                         f"recipe bends are separated by unmarked unsupported bend(s) {names}: no datum lets every "
                         "recipe bend reach it through recipe bends; mark the unsupported bend(s)")
    if not comps:
        return sorted(graph.panels)
    root = comps.pop()
    return sorted(k for k in graph.panels if find(k) == root)


def build_part_model(graph: FlangeGraph, rigid: set[int] | None = None) -> PartModel:
    """Root the flange graph at the datum and orient everything (module docstring).

    ``rigid``: graph bends that will not be recipe bends (unmarked unsupported);
    the datum is then the largest panel of the component their removal leaves
    holding every recipe bend (see :func:`_datum_candidates`).
    """
    if not graph.panels:
        raise GraphError("DATUM_PANEL_UNDETERMINED", "the CAD has no planar flange panel joined by a bend")
    if graph.thickness is None:
        raise GraphError("THICKNESS_INCONSISTENT", "no panel shows both skins and no bend gives a thickness")
    adj = graph.adjacency()
    n_edges = len(graph.bends)
    allowed = set(_datum_candidates(graph, set(rigid or ())))
    # connectivity + acyclicity
    ordered = sorted((p for p in graph.panels.values() if p.key in allowed), key=lambda p: int(p.panel_id[1:]))
    datum_graph = max(ordered, key=lambda p: (round(p.area, 6), -int(p.panel_id[1:])))
    datum = datum_graph.key
    seen = {datum}
    queue = [datum]
    parent_of: dict[int, tuple[int, int]] = {}
    order = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for bk, other in sorted(adj[cur], key=lambda x: graph.bends[x[0]].graph_bend_id):
            if other in seen:
                if parent_of.get(cur, (None, None))[0] != bk:
                    raise GraphError("FLANGE_GRAPH_NOT_A_TREE",
                                     f"cycle through bend {graph.bends[bk].graph_bend_id}")
                continue
            seen.add(other)
            parent_of[other] = (bk, cur)
            queue.append(other)
    if len(seen) != len(graph.panels) or n_edges != len(graph.panels) - 1:
        missing = sorted(graph.panels[k].panel_id for k in graph.panels if k not in seen)
        raise GraphError("FLANGE_GRAPH_NOT_A_TREE",
                         f"{len(graph.panels)} panels, {n_edges} bends; unreachable from the datum: {missing}")

    panels: dict[int, ModelPanel] = {}
    dg = graph.panels[datum]
    if len(dg.sides) == 2:
        s0 = skin0_rule(dg.sides[0].normal, dg.sides[1].normal)
        panels[datum] = ModelPanel(datum, dg.panel_id, dg, {0: s0, 1: 1 - s0})
    else:
        up = skin0_rule(dg.sides[0].normal, -dg.sides[0].normal) == 0
        panels[datum] = ModelPanel(datum, dg.panel_id, dg, {0: 0, 1: None} if up else {0: None, 1: 0})

    bends: dict[int, ModelBend] = {}
    for cur in order[1:]:
        bk, par = parent_of[cur]
        gb = graph.bends[bk]
        p_end, c_end = gb.end_for_panel(par), gb.end_for_panel(cur)
        r_p, r_c = p_end.radial, c_end.radial
        # parent skin on the convex side decides oml_skin; the child inherits skin continuity
        p_convex_side = int(graph.face_role[p_end.outer_planes[0]][2])
        c_convex_side = int(graph.face_role[c_end.outer_planes[0]][2])
        oml_skin = panels[par].skin_of_side(p_convex_side) if p_convex_side in panels[par].skin_side.values() else (
            1 - panels[par].skin_of_side(int(graph.face_role[p_end.inner_planes[0]][2])))
        cg = graph.panels[cur]
        other_side = None if len(cg.sides) < 2 else 1 - c_convex_side
        skin_side = {oml_skin: c_convex_side, 1 - oml_skin: other_side}
        depth = panels[par].depth + 1
        panels[cur] = ModelPanel(cur, cg.panel_id, cg, skin_side, parent_bend=bk, parent_panel=par, depth=depth)
        # outward directions: away from the bend, inside each panel's plane
        axis_c = gb.axis_dir
        if gb.sweep_deg < REGULAR_SWEEP_MAX_DEG:
            p_out = _outward(r_p, r_c, axis_c, graph.panels[par].mid_centroid, gb, p_end)
            c_out = _outward(r_c, r_p, axis_c, graph.panels[cur].mid_centroid, gb, c_end)
            axis = unit(np.cross(-p_out, c_out))
            mid_dir = unit(r_p + r_c)
        else:
            # near-reverse / reflex sweep (hems): cross(-p_out, c_out) vanishes or flips, so orient
            # the axis toward the material's angular middle and derive the outward directions from it
            if gb.mid_radial is None:
                raise GraphError("FLANGE_GRAPH_NOT_A_TREE", f"bend {gb.graph_bend_id} has no material mid radial")
            axis = oriented_bend_axis(r_p, axis_c, gb.mid_radial)
            p_out = unit(-np.cross(axis, r_p))
            c_out = unit(np.cross(axis, r_c))
            mid_dir = bend_bisector(r_p, r_c, axis, gb.sweep_deg, gb.mid_radial)
        if abs(float(axis @ axis_c)) < 0.999:
            raise GraphError("FLANGE_GRAPH_NOT_A_TREE", f"bend {gb.graph_bend_id} axis does not match its panels")
        sign = 1.0 if float(axis @ axis_c) > 0 else -1.0
        z0, z1 = (gb.z0, gb.z1) if sign > 0 else (-gb.z1, -gb.z0)
        bends[bk] = ModelBend(
            key=bk, graph_bend_id=gb.graph_bend_id, graph=gb, parent=par, child=cur, axis_point=gb.axis_point,
            axis_dir=axis, z0=z0, z1=z1, ri=gb.ri, ro=gb.ro, sweep_deg=gb.sweep_deg, r_parent=r_p, r_child=r_c,
            p_out=p_out, c_out=c_out, oml_skin=oml_skin, direction="UP" if oml_skin == 1 else "DOWN",
            unsupported=gb.unsupported, unsupported_reason=gb.unsupported_reason, mid_dir=mid_dir)
    return PartModel(graph=graph, panels=panels, bends=bends, datum=datum, order=order,
                     thickness=float(graph.thickness))


def _outward(r_self: np.ndarray, r_other: np.ndarray, axis: np.ndarray, centroid: np.ndarray, gb: GraphBend,
             end) -> np.ndarray:
    """In-plane unit vector of a panel at a bend end, perpendicular to the axis, pointing away from the bend."""
    tangent = unit(np.cross(axis, r_self))
    toward_other = r_other - float(r_other @ r_self) * r_self
    if float(np.linalg.norm(toward_other)) > 1e-6:
        # the bend lies on the side of the other end: outward points the other way
        out = -tangent if float(tangent @ toward_other) > 0 else tangent
    else:
        tangent_point = gb.axis_point + gb.ro * r_self
        out = tangent if float((centroid - tangent_point) @ tangent) > 0 else -tangent
    return out
