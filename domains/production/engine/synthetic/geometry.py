"""Analytic sheet-metal part model: flat blank, exact fold kinematics and loci.

This module is the single source of geometric truth for the E2 synthetic suite.
Everything else (the OCP BRep, the flat DXF, the virtual scan and ``truth.json``)
is derived from the objects defined here, so truth is computed from parameters,
never measured back from generated geometry.

Model
-----
A part is a **flat blank** made of planar *panels* (2D polygons with circular
holes, in the *flat frame*: the blank lies in ``0 <= z <= t``, skin 1 at
``z = 0`` with normal ``-Z``, skin 0 at ``z = t`` with normal ``+Z``) joined
by straight *bends*. Each bend owns a strip of the blank of width
``BA = sweep_rad * (Ri + K t)`` (the neutral length) centred on its *bend
line*; the two panels it joins end on the strip edges (their *attachment
edges*, one polygon edge each).

Folding (:class:`FoldedPart`) is exact rigid kinematics. For a bend formed to
an actual sweep ``alpha`` the arc consumes ``alpha (Ri + K t)`` of the strip,
centred on the (possibly shifted) bend line; the rest of the strip stays flat
and is added to the two panels. The child panel (the side away from the datum)
is mapped by::

    T_b = Rot(axis_point, axis_dir, alpha) . Trans(-BA_act * m)

in the flat frame, where ``m`` is the in-plane unit normal from parent to
child, ``axis_point`` the centre of curvature (at the parent's tangent line,
``Ri`` beyond the inner skin) and ``axis_dir`` is oriented so that ``+alpha``
(right-hand rule) goes from flat to formed (the CompiledPart convention).
World transforms compose from the datum outwards: ``W_child = W_parent . T_b``.
An unformed bend (``alpha = 0``) is the identity, so the stage-``k`` model of a
sequence prefix is exactly the contract's Method A stage model (rotate back by
the sweep about the axis, then translate by the neutral strip length).

Directions are given in the flat frame: ``UP`` means the child rotates towards
the skin-0 side (+Z of the flat), so skin 0 is inside the bend and the OML
(outside mould line) is skin 1; ``DOWN`` is the opposite. The contract's skin
and UP/DOWN labels are defined on the datum panel in the CAD frame; the CAD
placement may flip them, see :meth:`FoldedPart.contract_skin`.

Conventions match ``contracts/production/ENGINE_IO.md`` §1: included angle =
180 - sweep; OML flange length from the OML virtual sharp; side A by the +X
rule (:func:`contracts.production.compiled_part.assign_sides` logic, duplicated
here so this package imports nothing from E1/E3).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "FREE",
    "UP",
    "DOWN",
    "GeometryError",
    "Hole",
    "PanelDesign",
    "BendDesign",
    "PartDesign",
    "BendState",
    "FormState",
    "BendGeometry",
    "FoldedPart",
    "rotation_about_axis",
    "translation",
    "apply_points",
    "apply_dirs",
    "rigid_inverse",
    "polygon_signed_area",
    "assign_sides_rule",
    "neutral_length",
    "outside_setback",
]

FREE = "FREE"
UP = "UP"
DOWN = "DOWN"
Direction = Literal["UP", "DOWN"]
_EZ = np.array([0.0, 0.0, 1.0])
_PARALLEL_EPS = 1e-12
_ON_LINE_TOL = 1e-7


class GeometryError(ValueError):
    """A part design or forming state is inconsistent or physically infeasible."""


# ---------------------------------------------------------------------------
# small linear-algebra helpers (4x4 homogeneous, row-major numpy)
# ---------------------------------------------------------------------------


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 0.0:
        raise GeometryError("zero or non-finite vector")
    return a / n


def rotation_about_axis(point: Sequence[float], direction: Sequence[float], angle_rad: float) -> np.ndarray:
    """4x4 rotation by ``angle_rad`` (right-hand rule) about the line ``point + s*direction``."""
    d = _unit(direction)
    k = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    r = np.eye(3) + math.sin(angle_rad) * k + (1.0 - math.cos(angle_rad)) * (k @ k)
    p = np.asarray(point, dtype=float)
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = p - r @ p
    return out


def translation(v: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = np.asarray(v, dtype=float)
    return out


def apply_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return pts @ m[:3, :3].T + m[:3, 3]


def apply_dirs(m: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    return np.asarray(dirs, dtype=float) @ m[:3, :3].T


def rigid_inverse(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    r = m[:3, :3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ m[:3, 3]
    return out


def polygon_signed_area(poly: Sequence[Sequence[float]]) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def assign_sides_rule(outward_1: Sequence[float], outward_2: Sequence[float]) -> tuple[int, int]:
    """Side naming (ENGINE_IO §1): side A has the larger +X component of its in-plane
    outward unit vector; ties (|delta| < 1e-6) broken by +Y, then +Z.

    Duplicate of ``contracts.production.compiled_part.assign_sides`` (kept local so
    the synthetic suite stays independent; a test asserts both agree)."""
    for axis in range(3):
        delta = float(outward_1[axis]) - float(outward_2[axis])
        if abs(delta) >= 1e-6:
            return (0, 1) if delta > 0 else (1, 0)
    raise GeometryError("flange outward directions are identical; sides are undefined")


def neutral_length(sweep_deg: float, inner_radius: float, thickness: float, k_factor: float) -> float:
    """Developed (neutral-fibre) length of a bend: ``sweep_rad * (Ri + K t)``."""
    return math.radians(sweep_deg) * (inner_radius + k_factor * thickness)


def outside_setback(sweep_deg: float, inner_radius: float, thickness: float) -> float:
    """OSSB: distance from the outer tangent line to the OML virtual sharp, ``Ro tan(sweep/2)``."""
    return (inner_radius + thickness) * math.tan(math.radians(sweep_deg) / 2.0)


# ---------------------------------------------------------------------------
# design objects (flat frame)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hole:
    """A circular through-hole in a panel (flat frame, mm)."""

    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class PanelDesign:
    """A planar panel of the blank.

    ``outline`` is a counter-clockwise simple polygon (flat frame). ``edge_tags[i]``
    describes edge ``outline[i] -> outline[i+1]``: :data:`FREE` (a cut edge) or
    the key of the bend this edge is the attachment edge of.
    """

    key: str
    outline: tuple[tuple[float, float], ...]
    edge_tags: tuple[str, ...]
    holes: tuple[Hole, ...] = ()
    role: str = ""

    def area(self) -> float:
        return abs(polygon_signed_area(self.outline)) - sum(math.pi * h.radius**2 for h in self.holes)

    def edges_touching_attachments(self) -> frozenset[int]:
        """Indices of free edges sharing a vertex with an attachment (bend) edge.

        Moving such an edge would change a bend's span, so cut offsets may not
        touch them (the bend strip would have to widen with the blank)."""
        n = len(self.outline)
        out = set()
        for i, tag in enumerate(self.edge_tags):
            if tag == FREE:
                continue
            for j in ((i - 1) % n, (i + 1) % n):
                if self.edge_tags[j] == FREE:
                    out.add(j)
        return frozenset(out)

    def edge_index(self, tag: str) -> int:
        hits = [i for i, t in enumerate(self.edge_tags) if t == tag]
        if len(hits) != 1:
            raise GeometryError(f"panel {self.key} must have exactly one edge tagged {tag!r}, has {len(hits)}")
        return hits[0]


@dataclass(frozen=True)
class BendDesign:
    """A straight bend joining ``panel_a`` and ``panel_b``.

    ``line_point`` + ``line_dir`` define the bend line (the centre of the bend
    strip) in the flat frame; ``normal_ab`` is the in-plane unit normal pointing
    from ``panel_a`` to ``panel_b``; ``extent`` is the bend span along
    ``line_dir`` measured from ``line_point``. ``direction`` is relative to the
    flat frame's +Z (skin 0). ``sweep_deg`` is the nominal sweep (included angle
    = 180 - sweep).
    """

    key: str
    panel_a: str
    panel_b: str
    line_point: tuple[float, float]
    line_dir: tuple[float, float]
    normal_ab: tuple[float, float]
    extent: tuple[float, float]
    direction: Direction
    inner_radius: float
    sweep_deg: float

    @property
    def included_deg(self) -> float:
        return 180.0 - self.sweep_deg


@dataclass(frozen=True)
class PartDesign:
    """A complete synthetic part: blank, bends, bending order and מהלך groups.

    ``placement`` maps the flat/folding frame to the CAD frame (the STEP model
    coordinates, ``cad_model_mm``). ``sequence`` is the tagged bending order
    (bend keys); ``groups`` are contiguous slices of it (machine moves).
    """

    family: str
    thickness: float
    k_factor: float
    panels: tuple[PanelDesign, ...]
    bends: tuple[BendDesign, ...]
    sequence: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] = ()
    placement: tuple[tuple[float, ...], ...] = tuple(tuple(r) for r in np.eye(4).tolist())
    datum_key: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)

    # -- lookups -----------------------------------------------------------

    def panel(self, key: str) -> PanelDesign:
        for p in self.panels:
            if p.key == key:
                return p
        raise KeyError(key)

    def bend(self, key: str) -> BendDesign:
        for b in self.bends:
            if b.key == key:
                return b
        raise KeyError(key)

    @property
    def placement_matrix(self) -> np.ndarray:
        return np.asarray(self.placement, dtype=float)

    def default_datum(self) -> str:
        """The datum panel: ``datum_key`` when set, else the largest panel by area
        (ties broken by key, deterministic). Matches the plan rule 'the panel that
        never moves; the largest when several qualify'."""
        if self.datum_key is not None:
            return self.datum_key
        return sorted(self.panels, key=lambda p: (-round(p.area(), 6), p.key))[0].key

    def bend_allowance(self, key: str, sweep_deg: float | None = None) -> float:
        b = self.bend(key)
        return neutral_length(b.sweep_deg if sweep_deg is None else sweep_deg, b.inner_radius, self.thickness, self.k_factor)

    def sequence_index(self, key: str) -> int:
        return self.sequence.index(key) + 1

    def group_index(self, key: str) -> int | None:
        for i, g in enumerate(self.groups, start=1):
            if key in g:
                return i
        return None

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`GeometryError` unless the design is a consistent tree."""
        t, k = self.thickness, self.k_factor
        if not (t > 0 and math.isfinite(t)):
            raise GeometryError("thickness must be positive")
        if not 0.0 < k < 1.0:
            raise GeometryError("K-factor must be in (0, 1)")
        keys = [p.key for p in self.panels]
        if len(set(keys)) != len(keys) or not keys:
            raise GeometryError("panel keys must be unique and non-empty")
        bkeys = [b.key for b in self.bends]
        if len(set(bkeys)) != len(bkeys) or set(bkeys) & set(keys) or FREE in bkeys:
            raise GeometryError("bend keys must be unique and distinct from panel keys")
        if len(self.panels) != len(self.bends) + 1:
            raise GeometryError("a tree of panels has exactly one more panel than bends")
        pl = self.placement_matrix
        if pl.shape != (4, 4) or not np.allclose(pl[:3, :3].T @ pl[:3, :3], np.eye(3), atol=1e-9) or abs(
            np.linalg.det(pl[:3, :3]) - 1.0
        ) > 1e-9 or not np.allclose(pl[3], [0, 0, 0, 1]):
            raise GeometryError("placement must be a proper rigid transform")
        for p in self.panels:
            if len(p.outline) < 3 or len(p.edge_tags) != len(p.outline):
                raise GeometryError(f"panel {p.key}: outline/edge_tags mismatch")
            if polygon_signed_area(p.outline) <= 0:
                raise GeometryError(f"panel {p.key}: outline must be counter-clockwise")
            for tag in p.edge_tags:
                if tag != FREE and tag not in bkeys:
                    raise GeometryError(f"panel {p.key}: unknown edge tag {tag!r}")
            _check_simple_polygon(p)
            for h in p.holes:
                _check_hole(p, h, t)
        for b in self.bends:
            self._validate_bend(b)
        # tree connectivity
        adj: dict[str, set[str]] = {key: set() for key in keys}
        for b in self.bends:
            adj[b.panel_a].add(b.panel_b)
            adj[b.panel_b].add(b.panel_a)
        seen = {keys[0]}
        stack = [keys[0]]
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if seen != set(keys):
            raise GeometryError("panels are not connected by bends into one tree")
        if sorted(self.sequence) != sorted(bkeys) or len(set(self.sequence)) != len(self.sequence):
            raise GeometryError("sequence must list every bend exactly once")
        cursor = 0
        for g in self.groups:
            if not g:
                raise GeometryError("machine-move groups must be non-empty")
            try:
                start = self.sequence.index(g[0], cursor)
            except ValueError:
                raise GeometryError("groups must be ordered contiguous slices of the sequence") from None
            if tuple(self.sequence[start:start + len(g)]) != tuple(g):
                raise GeometryError("groups must be ordered contiguous slices of the sequence")
            cursor = start + len(g)
        if self.datum_key is not None and self.datum_key not in keys:
            raise GeometryError("datum_key must name a panel")

    def _validate_bend(self, b: BendDesign) -> None:
        d = np.asarray(b.line_dir, dtype=float)
        m = np.asarray(b.normal_ab, dtype=float)
        if abs(np.linalg.norm(d) - 1) > 1e-9 or abs(np.linalg.norm(m) - 1) > 1e-9 or abs(d @ m) > 1e-9:
            raise GeometryError(f"bend {b.key}: line_dir and normal_ab must be perpendicular unit vectors")
        if not 0.0 < b.sweep_deg < 180.0:
            raise GeometryError(f"bend {b.key}: sweep must be in (0, 180) degrees")
        if not b.inner_radius > 0:
            raise GeometryError(f"bend {b.key}: inner radius must be positive")
        if b.direction not in (UP, DOWN):
            raise GeometryError(f"bend {b.key}: direction must be UP or DOWN")
        e0, e1 = b.extent
        if not e1 > e0:
            raise GeometryError(f"bend {b.key}: extent must be increasing")
        ba = self.bend_allowance(b.key)
        lp = np.asarray(b.line_point, dtype=float)
        for panel_key, sign in ((b.panel_a, -1.0), (b.panel_b, 1.0)):
            p = self.panel(panel_key)
            i = p.edge_index(b.key)
            v0 = np.asarray(p.outline[i], dtype=float)
            v1 = np.asarray(p.outline[(i + 1) % len(p.outline)], dtype=float)
            for v in (v0, v1):
                off = float((v - lp) @ m)
                if abs(off - sign * ba / 2.0) > 1e-6:
                    raise GeometryError(
                        f"bend {b.key}: attachment edge of {panel_key} must lie on the strip edge "
                        f"(offset {off:.6f}, expected {sign * ba / 2:.6f})"
                    )
            span = sorted(float((v - lp) @ d) for v in (v0, v1))
            if abs(span[0] - e0) > 1e-6 or abs(span[1] - e1) > 1e-6:
                raise GeometryError(f"bend {b.key}: attachment edge of {panel_key} must span the bend extent")
            # neighbours must be perpendicular or collinear (keeps the span fixed under offsets)
            n = len(p.outline)
            prev_v = np.asarray(p.outline[(i - 1) % n], dtype=float)
            next_v = np.asarray(p.outline[(i + 2) % n], dtype=float)
            e = v1 - v0
            for a, c in ((prev_v, v0), (v1, next_v)):
                f = c - a
                cosang = abs(float(e @ f)) / (np.linalg.norm(e) * np.linalg.norm(f))
                if not (cosang < 1e-9 or cosang > 1 - 1e-12):
                    raise GeometryError(
                        f"bend {b.key}: edges adjacent to the attachment edge of {panel_key} must be perpendicular or collinear"
                    )
            # outward normal of the attachment edge must point towards the strip
            # (panel_a's edge sits at -BA/2 so it faces +m; panel_b's faces -m)
            out = np.array([e[1], -e[0]]) / np.linalg.norm(e)
            if float(out @ (-sign * m)) < 1 - 1e-9:
                raise GeometryError(f"bend {b.key}: panel {panel_key} must lie on its own side of the strip")


def _check_simple_polygon(p: PanelDesign) -> None:
    pts = np.asarray(p.outline, dtype=float)
    n = len(pts)
    for i in range(n):
        a0, a1 = pts[i], pts[(i + 1) % n]
        if np.linalg.norm(a1 - a0) < 1e-6:
            raise GeometryError(f"panel {p.key}: zero-length edge {i}")
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b0, b1 = pts[j], pts[(j + 1) % n]
            if _segments_intersect(a0, a1, b0, b1):
                raise GeometryError(f"panel {p.key}: outline self-intersects (edges {i}, {j})")


def _segments_intersect(a0, a1, b0, b1) -> bool:
    def orient(p, q, r):
        return _cross2(q - p, r - p)

    d1, d2 = orient(b0, b1, a0), orient(b0, b1, a1)
    d3, d4 = orient(a0, a1, b0), orient(a0, a1, b1)
    return (d1 * d2 < -1e-12) and (d3 * d4 < -1e-12)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    s = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


def point_in_polygon(pt: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xc > x:
                inside = not inside
    return inside


def _check_hole(p: PanelDesign, h: Hole, t: float) -> None:
    c = np.asarray(h.center, dtype=float)
    if not h.radius > 0:
        raise GeometryError(f"panel {p.key}: hole radius must be positive")
    if not point_in_polygon(c, p.outline):
        raise GeometryError(f"panel {p.key}: hole centre {h.center} outside the outline")
    pts = np.asarray(p.outline, dtype=float)
    n = len(pts)
    for i in range(n):
        dist = _point_segment_distance(c, pts[i], pts[(i + 1) % n])
        margin = max(0.5 * t, 0.5)
        if dist < h.radius + margin:
            raise GeometryError(f"panel {p.key}: hole at {h.center} too close to edge {i}")


# ---------------------------------------------------------------------------
# forming state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BendState:
    """Actual forming of one bend: ``sweep_deg`` (0 = not formed) and the bend-line
    shift along ``normal_ab`` (as-built backgauge error, mm)."""

    sweep_deg: float
    shift_mm: float = 0.0


@dataclass(frozen=True)
class FormState:
    """Per-bend forming plus per-free-edge cut offsets (outward positive, mm), keyed by
    ``(panel_key, design edge index)``."""

    bends: Mapping[str, BendState]
    cut_offsets: Mapping[tuple[str, int], float] = field(default_factory=dict)

    @classmethod
    def nominal(cls, design: PartDesign, formed: Iterable[str] | None = None) -> "FormState":
        """Nominal geometry with ``formed`` bends at their nominal sweep (all when None)."""
        formed_set = set(design.sequence if formed is None else formed)
        unknown = formed_set - {b.key for b in design.bends}
        if unknown:
            raise GeometryError(f"unknown bends {sorted(unknown)}")
        return cls({b.key: BendState(b.sweep_deg if b.key in formed_set else 0.0) for b in design.bends})

    def formed_keys(self) -> tuple[str, ...]:
        return tuple(k for k, s in self.bends.items() if s.sweep_deg > 0.0)


# ---------------------------------------------------------------------------
# evaluated (folded) part
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BendGeometry:
    """Evaluated geometry of one bend in the flat frame of its *parent* panel."""

    key: str
    parent: str
    child: str
    m: np.ndarray  # 2D unit, parent -> child
    d: np.ndarray  # 2D unit, line direction
    line_point: np.ndarray  # 2D effective (shifted) bend line point
    extent: tuple[float, float]
    direction: str  # flat-frame UP/DOWN
    inner_radius: float
    outer_radius: float
    thickness: float
    sweep_nominal_deg: float
    sweep_deg: float  # actual
    ba_nominal: float
    ba_actual: float
    axis_point: np.ndarray  # 3D flat frame (at e = 0 along d)
    axis_dir: np.ndarray  # 3D flat frame unit
    transform: np.ndarray  # 4x4 child relative transform (flat frame)

    @property
    def formed(self) -> bool:
        return self.sweep_deg > 0.0

    @property
    def alpha(self) -> float:
        return math.radians(self.sweep_deg)

    @property
    def ossb(self) -> float:
        return self.outer_radius * math.tan(self.alpha / 2.0)

    @property
    def issb(self) -> float:
        return self.inner_radius * math.tan(self.alpha / 2.0)

    @property
    def sigma(self) -> float:
        """Position of the OML virtual sharp along either side's outward direction,
        measured from the effective bend line, in that side's flat coordinates."""
        return self.ba_actual / 2.0 - self.ossb

    @property
    def z_oml(self) -> float:
        """Flat-frame z of the OML skin (outside of the bend) on the parent panel."""
        return 0.0 if self.direction == UP else self.thickness

    @property
    def z_iml(self) -> float:
        return self.thickness if self.direction == UP else 0.0

    @property
    def oml_flat_skin(self) -> int:
        """Flat-frame skin index on the outside of the bend (1 for UP, 0 for DOWN)."""
        return 1 if self.direction == UP else 0

    def outward(self, panel: str) -> np.ndarray:
        """2D in-plane unit vector from the bend into ``panel`` (flat frame)."""
        if panel == self.child:
            return self.m.copy()
        if panel == self.parent:
            return -self.m
        raise KeyError(panel)


class FoldedPart:
    """A :class:`PartDesign` evaluated at a :class:`FormState`.

    Attributes are all exact functions of the parameters:

    - ``world[panel]``: 4x4 flat-frame -> CAD-frame transform (includes the
      design placement and, when ``mirror`` is given, a final reflection);
    - ``bend_geo[bend]``: :class:`BendGeometry` in the parent's flat frame;
    - ``outline[panel]`` / ``tags[panel]``: the effective panel polygon (design
      polygon with attachment edges moved for partial forming and bend-line
      shifts, and free edges moved by cut offsets).
    """

    def __init__(
        self,
        design: PartDesign,
        state: FormState,
        datum: str | None = None,
        mirror: np.ndarray | None = None,
    ) -> None:
        design.validate()
        self.design = design
        self.state = state
        self.datum = datum or design.default_datum()
        if self.datum not in {p.key for p in design.panels}:
            raise GeometryError(f"datum {self.datum!r} is not a panel")
        missing = {b.key for b in design.bends} - set(state.bends)
        if missing:
            raise GeometryError(f"form state lacks bends {sorted(missing)}")
        self.mirror = None if mirror is None else np.asarray(mirror, dtype=float)
        t = design.thickness
        # orient the tree from the datum
        adj: dict[str, list[tuple[str, str]]] = {p.key: [] for p in design.panels}
        for b in design.bends:
            adj[b.panel_a].append((b.key, b.panel_b))
            adj[b.panel_b].append((b.key, b.panel_a))
        self.parent_bend: dict[str, str | None] = {self.datum: None}
        self.panel_order: list[str] = [self.datum]
        parent_of: dict[str, str] = {}
        child_of: dict[str, str] = {}
        queue = [self.datum]
        while queue:
            cur = queue.pop(0)
            for bkey, nb in sorted(adj[cur]):
                if nb in self.parent_bend:
                    continue
                self.parent_bend[nb] = bkey
                parent_of[bkey] = cur
                child_of[bkey] = nb
                self.panel_order.append(nb)
                queue.append(nb)
        self.bend_geo: dict[str, BendGeometry] = {}
        for b in design.bends:
            bs = state.bends[b.key]
            if not 0.0 <= bs.sweep_deg < 180.0:
                raise GeometryError(f"bend {b.key}: actual sweep must be in [0, 180)")
            parent, child = parent_of[b.key], child_of[b.key]
            m_ab = np.asarray(b.normal_ab, dtype=float)
            m = m_ab if parent == b.panel_a else -m_ab
            d = np.asarray(b.line_dir, dtype=float)
            lp = np.asarray(b.line_point, dtype=float) + bs.shift_mm * m_ab
            ba_nom = design.bend_allowance(b.key)
            # An over-bend (ba_act > ba_nom) consumes more strip than the design
            # allotted; the excess comes symmetrically out of the two panels
            # (negative extensions in _effective_outline), conserving the blank.
            ba_act = neutral_length(bs.sweep_deg, b.inner_radius, t, design.k_factor)
            m3 = np.array([m[0], m[1], 0.0])
            tangent = lp - (ba_act / 2.0) * m
            if b.direction == UP:
                axis_dir = np.cross(m3, _EZ)
                h = t + b.inner_radius
            else:
                axis_dir = np.cross(_EZ, m3)
                h = -b.inner_radius
            axis_point = np.array([tangent[0], tangent[1], h])
            rel = rotation_about_axis(axis_point, axis_dir, math.radians(bs.sweep_deg)) @ translation(-ba_act * m3)
            self.bend_geo[b.key] = BendGeometry(
                key=b.key,
                parent=parent,
                child=child,
                m=m,
                d=d,
                line_point=lp,
                extent=(float(b.extent[0]), float(b.extent[1])),
                direction=b.direction,
                inner_radius=b.inner_radius,
                outer_radius=b.inner_radius + t,
                thickness=t,
                sweep_nominal_deg=b.sweep_deg,
                sweep_deg=bs.sweep_deg,
                ba_nominal=ba_nom,
                ba_actual=ba_act,
                axis_point=axis_point,
                axis_dir=_unit(axis_dir),
                transform=rel,
            )
        base = design.placement_matrix
        if self.mirror is not None:
            base = self.mirror @ base
        self.world: dict[str, np.ndarray] = {self.datum: base.copy()}
        for panel in self.panel_order[1:]:
            bkey = self.parent_bend[panel]
            assert bkey is not None
            g = self.bend_geo[bkey]
            self.world[panel] = self.world[g.parent] @ g.transform
        self.outline: dict[str, np.ndarray] = {}
        self.tags: dict[str, tuple[str, ...]] = {}
        for p in design.panels:
            pts, tags = self._effective_outline(p)
            self.outline[p.key] = pts
            self.tags[p.key] = tags

    # -- effective polygons --------------------------------------------------

    def _effective_outline(self, p: PanelDesign) -> tuple[np.ndarray, tuple[str, ...]]:
        pts = np.asarray(p.outline, dtype=float)
        n = len(pts)
        disp = np.zeros((n, 2))
        for i, tag in enumerate(p.edge_tags):
            if tag == FREE:
                off = float(self.state.cut_offsets.get((p.key, i), 0.0))
                if off:
                    e = pts[(i + 1) % n] - pts[i]
                    disp[i] = off * np.array([e[1], -e[0]]) / np.linalg.norm(e)
                continue
            b = self.design.bend(tag)
            g = self.bend_geo[tag]
            m_ab = np.asarray(b.normal_ab, dtype=float)
            shift = self.state.bends[tag].shift_mm
            ext = (g.ba_nominal - g.ba_actual) / 2.0
            delta = shift + ext if p.key == b.panel_a else shift - ext
            disp[i] = delta * m_ab
        touching = p.edges_touching_attachments()
        for (key, idx) in self.state.cut_offsets:
            if key != p.key:
                continue
            if p.edge_tags[idx] != FREE:
                raise GeometryError(f"cut offsets apply to free edges only ({key}, edge {idx})")
            if idx in touching and self.state.cut_offsets[(key, idx)] != 0.0:
                raise GeometryError(f"cut offset on {key} edge {idx} would move a bend attachment edge's end")
        if not np.any(disp):
            return pts.copy(), tuple(p.edge_tags)
        out_pts: list[np.ndarray] = []
        out_tags: list[str] = []
        for i in range(n):
            ip = (i - 1) % n
            v = pts[i]
            e_prev = pts[i] - pts[ip]
            e_cur = pts[(i + 1) % n] - pts[i]
            dp, dc = disp[ip], disp[i]
            if not np.any(dp) and not np.any(dc):
                out_pts.append(v.copy())
            elif abs(_cross2(e_prev, e_cur)) > _PARALLEL_EPS * np.linalg.norm(e_prev) * np.linalg.norm(e_cur):
                # intersection of the two (displaced) edge lines
                a0 = pts[ip] + dp
                b0 = pts[i] + dc
                s = _cross2(b0 - a0, e_cur) / _cross2(e_prev, e_cur)
                out_pts.append(a0 + s * e_prev)
            elif np.allclose(dp, dc, atol=1e-12):
                out_pts.append(v + dc)
            else:
                # collinear neighbours displaced differently: insert a step edge
                out_pts.append(v + dp)
                out_tags.append(FREE)
                out_pts.append(v + dc)
            out_tags.append(p.edge_tags[i])
        # out_tags currently has, per vertex, possibly an inserted FREE before the
        # edge tag; rebuild so that tag k describes out_pts[k] -> out_pts[k+1].
        result = np.asarray(out_pts)
        if len(out_tags) != len(result):
            raise GeometryError("internal: effective outline tag mismatch")
        if polygon_signed_area(result) <= 0:
            raise GeometryError(f"panel {p.key}: effective outline collapsed (offsets too large)")
        return result, tuple(out_tags)

    # -- frames / conventions ---------------------------------------------------

    def panel_normal_skin0(self, panel: str) -> np.ndarray:
        """CAD-frame unit normal of the panel's flat skin 0 (flat +Z) at this state."""
        return _unit(apply_dirs(self.world[panel], _EZ))

    def contract_skin0_is_flat_skin0(self) -> bool:
        """True when the contract's datum skin 0 (normal with the larger +Z, ties +Y
        then +X) is this model's flat skin 0."""
        n = self.panel_normal_skin0(self.datum)
        for axis in (2, 1, 0):
            if abs(n[axis]) >= 1e-6:
                return bool(n[axis] > 0)
        raise GeometryError("datum normal is degenerate")

    def contract_skin(self, flat_skin: int) -> int:
        """Map a flat-frame skin index (0 = +Z of the flat) to the contract skin index."""
        if flat_skin not in (0, 1):
            raise ValueError(flat_skin)
        return flat_skin if self.contract_skin0_is_flat_skin0() else 1 - flat_skin

    def contract_direction(self, bend: str) -> str:
        """UP/DOWN relative to the datum's contract skin-0 normal (UP iff oml_skin == 1)."""
        oml = self.contract_skin(self.bend_geo[bend].oml_flat_skin)
        return UP if oml == 1 else DOWN

    # -- loci ------------------------------------------------------------------

    def _line_flat(self, g: BendGeometry, offset_m: float, z: float) -> np.ndarray:
        """Two 3D flat-frame points of a line parallel to the bend at m-offset ``offset_m``
        from the effective bend line and height ``z``, spanning the bend extent."""
        base = g.line_point + offset_m * g.m
        pts = [np.array([*(base + e * g.d), z]) for e in g.extent]
        return np.asarray(pts)

    def loci(self, bend: str) -> dict[str, np.ndarray]:
        """All loci of a formed bend (CAD frame), each as a (2, 3) array spanning the bend.

        Keys: cylinder_axis, inner_tangent_parent, inner_tangent_child,
        outer_tangent_parent, outer_tangent_child, iml_virtual_sharp,
        oml_virtual_sharp, inner_apex, outer_apex. For an unformed bend only
        ``strip_centre_oml`` (the bend line on the OML skin; the stage-model
        convention) and ``strip_centre_mid`` are returned.
        """
        g = self.bend_geo[bend]
        w = self.world[g.parent]
        if not g.formed:
            return {
                "strip_centre_oml": apply_points(w, self._line_flat(g, 0.0, g.z_oml)),
                "strip_centre_mid": apply_points(w, self._line_flat(g, 0.0, g.thickness / 2.0)),
            }
        tan_off = -g.ba_actual / 2.0
        outer_tp = self._line_flat(g, tan_off, g.z_oml)
        inner_tp = self._line_flat(g, tan_off, g.z_iml)
        rot_full = rotation_about_axis(g.axis_point, g.axis_dir, g.alpha)
        rot_half = rotation_about_axis(g.axis_point, g.axis_dir, g.alpha / 2.0)
        axis_line = np.asarray([g.axis_point + np.array([*(e * g.d), 0.0]) for e in g.extent])
        flat = {
            "cylinder_axis": axis_line,
            "inner_tangent_parent": inner_tp,
            "inner_tangent_child": apply_points(rot_full, inner_tp),
            "outer_tangent_parent": outer_tp,
            "outer_tangent_child": apply_points(rot_full, outer_tp),
            "iml_virtual_sharp": self._line_flat(g, tan_off + g.issb, g.z_iml),
            "oml_virtual_sharp": self._line_flat(g, tan_off + g.ossb, g.z_oml),
            "inner_apex": apply_points(rot_half, inner_tp),
            "outer_apex": apply_points(rot_half, outer_tp),
            "mid_radius_apex": apply_points(rot_half, self._line_flat(g, tan_off, g.thickness / 2.0)),
        }
        return {k: apply_points(w, v) for k, v in flat.items()}

    def bend_line_cad(self, bend: str) -> np.ndarray:
        """The stage-model bend line: OML virtual sharp when formed, else the strip
        centre on the OML skin (ENGINE_IO §2 stage ``bend_lines``)."""
        loci = self.loci(bend)
        return loci["oml_virtual_sharp"] if "oml_virtual_sharp" in loci else loci["strip_centre_oml"]

    def axis_cad(self, bend: str) -> tuple[np.ndarray, np.ndarray]:
        """Cylinder axis point (at extent start) and direction (CAD frame). For an
        unformed bend this is the axis it *would* rotate about at its nominal sweep."""
        g = self.bend_geo[bend]
        w = self.world[g.parent]
        if g.formed:
            p = g.axis_point + np.array([*(g.extent[0] * g.d), 0.0])
            return apply_points(w, p[None])[0], _unit(apply_dirs(w, g.axis_dir))
        # axis of the nominal-sweep fold at this bend's current place
        tangent = g.line_point - (g.ba_nominal / 2.0) * g.m
        h = g.thickness + g.inner_radius if g.direction == UP else -g.inner_radius
        p = np.array([*(tangent + g.extent[0] * g.d), h])
        return apply_points(w, p[None])[0], _unit(apply_dirs(w, g.axis_dir))

    def outward_cad(self, bend: str, panel: str) -> np.ndarray:
        """CAD-frame in-plane unit vector from the bend into ``panel``."""
        g = self.bend_geo[bend]
        o = g.outward(panel)
        return _unit(apply_dirs(self.world[panel], np.array([o[0], o[1], 0.0])))

    def panel_pose_change(self, other: "FoldedPart", panel: str) -> np.ndarray:
        """Rigid transform taking ``other``'s pose of ``panel`` to this part's pose."""
        return self.world[panel] @ rigid_inverse(other.world[panel])

    # -- misc ------------------------------------------------------------------

    def is_mirrored(self) -> bool:
        return self.mirror is not None

    def path_to_datum(self, panel: str) -> list[str]:
        """Bends from the datum to ``panel`` (datum side first)."""
        path: list[str] = []
        cur = panel
        while True:
            bkey = self.parent_bend[cur]
            if bkey is None:
                break
            path.append(bkey)
            cur = self.bend_geo[bkey].parent
        return path[::-1]
