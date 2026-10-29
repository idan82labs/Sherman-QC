"""Parametric sheet-metal fixtures built as real OCP B-reps (tests and dev self-checks).

Not used on the production compile path. The E2 synthetic suite is the
qualification generator; these fixtures exist so the E1 tests do not depend on
E2 (built concurrently) and so expected nominals can be derived *analytically*
from the fixture parameters, independently of the compile's B-rep analysis.

Profile parts (:func:`profile_part`) are defined by the **skin-0 virtual-sharp
polyline** in a 2D profile plane (``(x, z)``), walked so that skin 0 is on the
left and the material (thickness ``t``) on the right. Each interior vertex is a
bend with its own inner radius. The profile is extruded along +Y by ``width``.
At a left (CCW) turn skin 0 is the inside of the bend (its arc radius is
``Ri``); at a right (CW) turn skin 0 is the outside (``Ri + t``). The
:class:`ProfileInfo` returned with the shape gives, per bend, the cylinder axis,
the OML virtual sharp, the native mid-radius half-turn mark line (the
``NATIVE_MID_RADIUS_HALF_ANGULAR_TURN`` rule of the CAD upload), the sweep and
the analytic OML flange lengths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ProfileBendInfo",
    "ProfileInfo",
    "profile_part",
    "two_direction_part",
    "cut_hole",
    "cut_box",
    "round_vertical_edges",
    "write_step",
    "profile_flat",
    "candidates_from_info",
    "hem_part",
    "rolled_part",
    "window_u_part",
    "write_flat_dxf",
    "write_flat_step",
    "l_part",
    "u_part",
    "z_part",
    "hat_part",
    "ArcBendInfo",
    "arc_profile_part",
    "arc_candidates",
    "hem_180_part",
    "teardrop_part",
    "nurbs_convert",
    "compound_of",
    "add_rounded_boss",
]


@dataclass
class ProfileBendInfo:
    index: int
    sweep_deg: float
    ri: float
    ro: float
    turn: str  # "LEFT" (skin 0 inside) | "RIGHT" (skin 0 outside)
    center: np.ndarray  # 3D point on the cylinder axis (y = 0)
    axis: np.ndarray  # +Y
    oml_sharp: np.ndarray  # 3D point (y = 0)
    iml_sharp: np.ndarray
    mark_start: np.ndarray  # native mid-radius half-turn line
    mark_end: np.ndarray
    flange_in_dir: np.ndarray  # 3D direction of the incoming flange (walking direction)
    flange_out_dir: np.ndarray


@dataclass
class ProfileInfo:
    thickness: float
    width: float
    bends: list[ProfileBendInfo]
    #: OML flange lengths per straight segment: segment i spans vertex i -> i+1.
    segment_oml_lengths: list[float]
    skin0_sharps: np.ndarray
    skin1_sharps: np.ndarray
    notes: list[str] = field(default_factory=list)


def _rot2(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _left(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]])


def _line_intersection(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray) -> np.ndarray:
    m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    s, _ = np.linalg.solve(m, q - p)
    return p + s * d


def _to3(p2: np.ndarray, y: float = 0.0) -> np.ndarray:
    return np.array([float(p2[0]), y, float(p2[1])])


def profile_part(vertices: list[tuple[float, float]], thickness: float, radii: list[float] | float, width: float):
    """Extruded sheet profile from the skin-0 sharp polyline (module docstring).

    Returns ``(shape, ProfileInfo)``.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.gp import gp_Pnt, gp_Vec

    v = [np.asarray(p, dtype=np.float64) for p in vertices]
    n_bends = len(v) - 2
    if n_bends < 0:
        raise ValueError("need at least two vertices")
    radii_list = [float(radii)] * n_bends if isinstance(radii, (int, float)) else [float(r) for r in radii]
    if len(radii_list) != n_bends:
        raise ValueError("one inner radius per interior vertex")
    t = float(thickness)
    dirs = [(v[i + 1] - v[i]) / np.linalg.norm(v[i + 1] - v[i]) for i in range(len(v) - 1)]
    # skin 1 sharp polyline: offset by t to the right of skin 0
    rights = [-_left(d) for d in dirs]
    s1 = [v[0] + t * rights[0]]
    for i in range(1, len(v) - 1):
        s1.append(_line_intersection(v[i - 1] + t * rights[i - 1], dirs[i - 1], v[i] + t * rights[i], dirs[i]))
    s1.append(v[-1] + t * rights[-1])

    arcs = []  # per bend: (center, r0, r1, tangent points skin0 in/out, skin1 in/out, mid points)
    infos: list[ProfileBendInfo] = []
    for b in range(n_bends):
        i = b + 1
        d_in, d_out = dirs[i - 1], dirs[i]
        cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
        theta = math.atan2(cross, float(d_in @ d_out))
        if abs(theta) < 1e-9:
            raise ValueError("collinear vertices are not a bend")
        left_turn = theta > 0
        ri = radii_list[b]
        r0 = ri if left_turn else ri + t
        r1 = ri + t if left_turn else ri
        n_in = _left(d_in) if left_turn else -_left(d_in)
        n_out = _left(d_out) if left_turn else -_left(d_out)
        center = _line_intersection(v[i] + r0 * n_in, d_in, v[i] + r0 * n_out, d_out)
        t0_in, t0_out = center - r0 * n_in, center - r0 * n_out
        t1_in, t1_out = center - r1 * n_in, center - r1 * n_out
        m = -(n_in + n_out)
        m = m / np.linalg.norm(m)
        mid0, mid1 = center + r0 * m, center + r1 * m
        arcs.append((center, r0, r1, t0_in, t0_out, t1_in, t1_out, mid0, mid1))
        oml = v[i] if not left_turn else s1[i]
        iml = s1[i] if not left_turn else v[i]
        mark = center + (ri + t / 2.0) * m
        infos.append(ProfileBendInfo(
            index=b + 1, sweep_deg=math.degrees(abs(theta)), ri=ri, ro=ri + t,
            turn="LEFT" if left_turn else "RIGHT", center=_to3(center), axis=np.array([0.0, 1.0, 0.0]),
            oml_sharp=_to3(oml), iml_sharp=_to3(iml), mark_start=_to3(mark, 0.0), mark_end=_to3(mark, width),
            flange_in_dir=_to3(d_in), flange_out_dir=_to3(d_out)))

    # straight OML lengths per segment (between OML sharps, or to the free end)
    oml_pts = []
    for k in range(len(v)):
        if k == 0 or k == len(v) - 1:
            oml_pts.append(None)
        else:
            oml_pts.append(np.array([infos[k - 1].oml_sharp[0], infos[k - 1].oml_sharp[2]]))
    seg_lengths = []
    for k in range(len(v) - 1):
        a = oml_pts[k] if oml_pts[k] is not None else v[k]
        bpt = oml_pts[k + 1] if oml_pts[k + 1] is not None else v[k + 1]
        seg_lengths.append(float(abs((bpt - a) @ dirs[k])))

    def P(p):
        return gp_Pnt(float(p[0]), 0.0, float(p[1]))

    edges = []

    def seg(p, q):
        if np.linalg.norm(np.asarray(q) - np.asarray(p)) > 1e-9:
            edges.append(BRepBuilderAPI_MakeEdge(GC_MakeSegment(P(p), P(q)).Value()).Edge())

    def arc(p, mid, q):
        edges.append(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(P(p), P(mid), P(q)).Value()).Edge())

    # skin 0 forward
    cur = v[0]
    for b in range(n_bends):
        center, r0, r1, t0_in, t0_out, t1_in, t1_out, mid0, mid1 = arcs[b]
        seg(cur, t0_in)
        arc(t0_in, mid0, t0_out)
        cur = t0_out
    seg(cur, v[-1])
    # end cap, skin 1 backward, start cap
    seg(v[-1], s1[-1])
    cur = s1[-1]
    for b in reversed(range(n_bends)):
        center, r0, r1, t0_in, t0_out, t1_in, t1_out, mid0, mid1 = arcs[b]
        seg(cur, t1_out)
        arc(t1_out, mid1, t1_in)
        cur = t1_in
    seg(cur, s1[0])
    seg(s1[0], v[0])
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
    face = BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()
    shape = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, float(width), 0.0)).Shape()
    info = ProfileInfo(thickness=t, width=float(width), bends=infos, segment_oml_lengths=seg_lengths,
                       skin0_sharps=np.array([_to3(p) for p in v]), skin1_sharps=np.array([_to3(p) for p in s1]))
    return shape, info


def l_part(a: float = 40.0, b: float = 30.0, t: float = 2.0, ri: float = 2.0, width: float = 50.0, sweep: float = 90.0):
    """L bracket: base along +X, flange turning up by ``sweep`` (skin 0 = top, inside)."""
    ang = math.radians(sweep)
    p0 = (0.0, t)
    p1 = (a - t * math.tan(ang / 2.0), t)  # skin-0 sharp so that the OML base length is ~a
    p2 = (p1[0] + b * math.cos(ang), t + b * math.sin(ang))
    return profile_part([p0, p1, p2], t, ri, width)


def u_part(base: float = 60.0, h1: float = 30.0, h2: float = 30.0, t: float = 2.0, ri: float = 2.0, width: float = 80.0):
    """U channel: walls up at both ends of the base. Walking down, right, up makes two
    left turns, so skin 0 is the inside of the U and the material lies outside it."""
    pts = [(0.0, h1), (0.0, 0.0), (base, 0.0), (base, h2)]
    return profile_part(pts, t, ri, width)


def z_part(a: float = 30.0, h: float = 25.0, b: float = 35.0, t: float = 1.5, ri: float = 1.5, width: float = 60.0):
    """Z section: one left turn and one right turn."""
    return profile_part([(0.0, 0.0), (a, 0.0), (a, h), (a + b, h)], t, ri, width)


def hat_part(flange_l: float = 20.0, wall: float = 30.0, top: float = 40.0, flange_r: float = 30.0, t: float = 2.0,
             ri: float = 2.0, width: float = 100.0):
    """Asymmetric hat section (like the contract example SYN-HAT-1)."""
    pts = [(-flange_l, 0.0), (0.0, 0.0), (0.0, wall), (top, wall), (top, 0.0), (top + flange_r, 0.0)]
    return profile_part(pts, t, ri, width)


def _transform(shape, rotation_z_deg: float = 0.0, translate=(0.0, 0.0, 0.0)):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

    trsf = gp_Trsf()
    if rotation_z_deg:
        trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(rotation_z_deg))
    move = gp_Trsf()
    move.SetTranslation(gp_Vec(*[float(c) for c in translate]))
    return BRepBuilderAPI_Transform(shape, move.Multiplied(trsf), True).Shape()


def _fuse(*shapes):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    out = shapes[0]
    for s in shapes[1:]:
        op = BRepAlgoAPI_Fuse(out, s)
        op.Build()
        out = op.Shape()
    return out


def two_direction_part(a: float = 80.0, b: float = 60.0, t: float = 2.0, ri: float = 2.0, h1: float = 25.0,
                       h2: float = 20.0, gap: float = 4.0):
    """Base plate (x in [0,a], y in [0,b], z in [0,t]) with an up-flange along the +X edge
    (bend axis along Y) and an up-flange along the +Y edge (bend axis along X), with a corner gap.
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    base = BRepPrimAPI_MakeBox(a, b, t).Shape()
    # flange 1: profile starts at x = a on the top skin; extruded along +Y over [0, b - gap]
    f1, _ = profile_part([(a, t), (a + ri + 0.0, t), (a + ri, t + h1)], t, ri, b - gap)
    # flange 2: built along +X then rotated to run along the +Y edge
    f2, _ = profile_part([(b, t), (b + ri, t), (b + ri, t + h2)], t, ri, a - gap)
    f2 = _transform(f2, rotation_z_deg=90.0)  # (x, y) -> (-y, x): profile now runs along +Y at x in [-(a-gap), 0]
    f2 = _transform(f2, translate=(a - gap, 0.0, 0.0))
    return _fuse(base, f1, f2)


def cut_hole(shape, center, normal, diameter: float, depth: float = 50.0):
    """Subtract a through cylinder (axis ``normal`` through ``center``)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    start = np.asarray(center, dtype=np.float64) - n * depth / 2.0
    tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*start), gp_Dir(*n)), diameter / 2.0, depth).Shape()
    op = BRepAlgoAPI_Cut(shape, tool)
    op.Build()
    return op.Shape()


def cut_box(shape, lo, hi):
    """Subtract an axis-aligned box (notches, reliefs)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    tool = BRepPrimAPI_MakeBox(gp_Pnt(*[float(c) for c in lo]), gp_Pnt(*[float(c) for c in hi])).Shape()
    op = BRepAlgoAPI_Cut(shape, tool)
    op.Build()
    return op.Shape()


def write_step(shape, path: str | Path) -> Path:
    """Write ``shape`` to a STEP file (mm) and return the path."""
    from domains.production.engine.compile.brep import quiet_occ
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    quiet_occ()
    out = Path(path)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(out)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {out}")
    return out


def profile_flat(info: ProfileInfo, k_factor: float | list[float]
                 ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], float]:
    """Analytic flat blank of a profile part for ``k_factor``.

    Returns ``(outer rectangle (4,2), bend lines [(start, end)] (strip centre
    lines), flat length)`` in a frame with x along the developed profile from the
    first free edge and y along the extrusion (0..width). Independent of the
    compile: setback ``Ro tan(sweep/2)``, strip ``sweep (Ri + K t)``. A list gives
    one K per bend (profile order), as a shop's bend table would.
    """
    t = info.thickness
    n = len(info.bends)
    ks = [float(k_factor)] * n if isinstance(k_factor, (int, float)) else [float(k) for k in k_factor]
    if len(ks) != n:
        raise ValueError("one K per bend")
    x = 0.0
    lines = []
    for k, oml in enumerate(info.segment_oml_lengths):
        left = info.bends[k - 1] if k >= 1 else None
        right = info.bends[k] if k < n else None
        flat_k = oml - (left.ro * math.tan(math.radians(left.sweep_deg) / 2) if left else 0.0) - (
            right.ro * math.tan(math.radians(right.sweep_deg) / 2) if right else 0.0)
        x += flat_k
        if right is not None:
            ln = math.radians(right.sweep_deg) * (right.ri + ks[k] * t)
            lines.append((np.array([x + ln / 2, 0.0]), np.array([x + ln / 2, info.width])))
            x += ln
    outer = np.array([[0.0, 0.0], [x, 0.0], [x, info.width], [0.0, info.width]])
    return outer, lines, x


def _place(pts: np.ndarray, rotate_deg: float, offset, mirror: bool) -> np.ndarray:
    q = np.asarray(pts, dtype=np.float64).copy()
    if mirror:
        q[:, 1] = -q[:, 1]
    a = math.radians(rotate_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return q @ r.T + np.asarray(offset, dtype=np.float64)


def write_flat_dxf(path: str | Path, outer: np.ndarray, lines, *, holes=(), rotate_deg: float = 0.0,
                   offset=(0.0, 0.0), mirror: bool = False, layer: str = "BEND", linetype: str = "DASHED",
                   annotations: list[str] | None = None, units_mm: bool = True) -> Path:
    """Write a flat DXF: outline LWPOLYLINE, circles for holes, bend lines on ``layer``/``linetype``."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    if units_mm:
        doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    if layer and layer not in doc.layers:
        doc.layers.add(layer)
    msp.add_lwpolyline([tuple(p) for p in _place(outer, rotate_deg, offset, mirror)], close=True,
                       dxfattribs={"layer": "0"})
    for (c, r) in holes:
        cc = _place(np.array([c]), rotate_deg, offset, mirror)[0]
        msp.add_circle(tuple(cc), r, dxfattribs={"layer": "0"})
    for i, (a, b) in enumerate(lines):
        pa, pb = _place(np.stack([a, b]), rotate_deg, offset, mirror)
        attribs = {"layer": layer or "0"}
        if linetype:
            attribs["linetype"] = linetype
        msp.add_line(tuple(pa), tuple(pb), dxfattribs=attribs)
        if annotations and i < len(annotations):
            mid = 0.5 * (pa + pb)
            msp.add_text(annotations[i], dxfattribs={"insert": (float(mid[0]) + 2.0, float(mid[1]) + 2.0),
                                                     "height": 2.5, "layer": "0"})
    out = Path(path)
    doc.saveas(str(out))
    return out


def write_flat_step(path: str | Path, outer: np.ndarray, thickness: float, holes=()) -> Path:
    """Flat blank as a planar STEP solid (outline extruded by ``thickness`` along +Z)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Pnt, gp_Vec

    poly = BRepBuilderAPI_MakePolygon()
    for p in outer:
        poly.Add(gp_Pnt(float(p[0]), float(p[1]), 0.0))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
    shape = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, float(thickness))).Shape()
    for (c, r) in holes:
        shape = cut_hole(shape, (c[0], c[1], thickness / 2), (0, 0, 1), 2 * r, depth=4 * thickness + 2)
    return write_step(shape, path)


def candidates_from_info(info: ProfileInfo, *, pieces: list[tuple[float, float]] | None = None) -> list[dict]:
    """Native-style CAD candidates (mid-radius half-turn lines) of a profile part.

    ``pieces`` splits every line into axial pieces ``[(y0, y1), ...]`` the way the
    CAD extractors report a bend interrupted by a cutout.
    """
    out = []
    for b in info.bends:
        spans = pieces or [(0.0, info.width)]
        for y0, y1 in spans:
            start, end = b.mark_start.copy(), b.mark_end.copy()
            start[1], end[1] = y0, y1
            out.append({"line_start": start.tolist(), "line_end": end.tolist(), "target_angle": b.sweep_deg,
                        "target_radius": b.ri})
    return out


def hem_part(base: float = 40.0, ret: float = 15.0, t: float = 1.0, ri: float = 0.5, sweep: float = 172.0,
             width: float = 30.0):
    """A base with a near-closed hem (sweep >= 170 deg: HEM under the UNSUPPORTED rule) and an up flange.

    Built through :func:`profile_part` (skin-1 sharps by line intersection), which degenerates
    as the hem's two flanges become parallel: keep ``sweep`` below 178 deg and use
    :func:`hem_180_part` / :func:`teardrop_part` (exact arcs) at and beyond 180 deg.
    """
    ang = math.radians(sweep)
    p1 = (base, t)
    p2 = (p1[0] + ret * math.cos(ang), t + ret * math.sin(ang))
    # an ordinary 90 deg flange at the other end keeps a supported bend in the part
    return profile_part([(0.0, t + 20.0), (0.0, t), p1, p2], t, [1.0, ri], width)


def rolled_part(a: float = 60.0, b: float = 50.0, t: float = 2.0, ri: float = 25.0, width: float = 40.0):
    """An L with a rolled radius (Ri = 12.5 t > 10 t: ROLLED under the UNSUPPORTED rule) plus a normal bend."""
    return profile_part([(0.0, 0.0), (a, 0.0), (a, b), (a + 30.0, b)], t, [ri, 2.0], width)


def window_u_part(base: float = 60.0, h: float = 30.0, t: float = 2.0, ri: float = 2.0, width: float = 80.0,
                  window: tuple[float, float] = (35.0, 45.0)):
    """U channel whose base is cut by a full-width window: two base pieces joined only through the walls."""
    shape, info = u_part(base=base, h1=h, h2=h, t=t, ri=ri, width=width)
    y0, y1 = window
    shape = cut_box(shape, (-t - 5.0, y0, -t - 5.0), (base + t + 5.0, y1, ri + 1.0))
    return shape, info


def round_vertical_edges(shape, points_xy, radius: float):
    """Fillet the straight edges parallel to Z whose midpoints are at ``points_xy`` (outline corner rounds)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.GeomAbs import GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    fillet = BRepFilletAPI_MakeFillet(shape)
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    added = 0
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() == GeomAbs_Line:
            a, b = curve.Value(curve.FirstParameter()), curve.Value(curve.LastParameter())
            if abs(a.X() - b.X()) < 1e-9 and abs(a.Y() - b.Y()) < 1e-9 and abs(a.Z() - b.Z()) > 1e-9:
                if any(abs(a.X() - x) < 1e-6 and abs(a.Y() - y) < 1e-6 for x, y in points_xy):
                    fillet.Add(float(radius), edge)
                    added += 1
        explorer.Next()
    if not added:
        raise ValueError("no vertical edge found at the given points")
    fillet.Build()
    return fillet.Shape()


@dataclass
class ArcBendInfo:
    """One arc of :func:`arc_profile_part` (3D at y = 0; axis +Y)."""

    index: int
    sweep_deg: float  # unsigned
    ri: float
    ro: float
    turn: str  # LEFT (skin 0 inside) | RIGHT (skin 0 outside)
    center: np.ndarray
    mid_dir: np.ndarray  # unit apex direction (angular middle of the arc)
    mark_start: np.ndarray  # native mid-radius half-turn line, y = 0 .. width
    mark_end: np.ndarray


def arc_profile_part(ops: list[tuple], thickness: float, width: float, start=(0.0, 0.0), heading_deg: float = 0.0):
    """Extruded sheet profile from a turtle walk of skin 0, for **any** sweep (hems at and past 180 deg).

    ``ops``: ``("line", length)`` or ``("arc", sweep_deg, inner_radius)`` with
    sweep > 0 a left (CCW) turn, where skin 0 is the inside (radius Ri), and
    sweep < 0 a right turn, where skin 0 is the outside (radius Ri + t). The
    profile lies in the (x, z) plane, skin 1 is ``t`` to the right of skin 0,
    and the solid is extruded along +Y by ``width``. Arcs are built from their
    angles (no line intersections), so 180 deg open/closed hems and reflex
    teardrops are exact. Returns ``(shape, [ArcBendInfo])``.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.gp import gp_Pnt, gp_Vec

    t = float(thickness)
    p = np.asarray(start, dtype=np.float64)
    a0 = math.radians(heading_deg)
    h = np.array([math.cos(a0), math.sin(a0)])

    def right(d):
        return np.array([d[1], -d[0]])

    walk0 = [("pt", p.copy())]  # skin-0 vertices and arc mids, forward
    walk1 = [("pt", p + t * right(h))]  # skin-1 counterparts
    infos: list[ArcBendInfo] = []
    for op in ops:
        if op[0] == "line":
            p = p + float(op[1]) * h
            walk0.append(("line", p.copy()))
            walk1.append(("line", p + t * right(h)))
            continue
        sweep, ri = math.radians(float(op[1])), float(op[2])
        left_turn = sweep > 0
        r0 = ri if left_turn else ri + t
        r1 = ri + t if left_turn else ri
        c = p + r0 * (-right(h) if left_turn else right(h))
        u = (p - c) / r0  # unit radial of the start point (both skins)

        def rot(v, ang):
            ca, sa = math.cos(ang), math.sin(ang)
            return np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]])

        u_mid, u_end = rot(u, sweep / 2.0), rot(u, sweep)
        walk0.append(("arc", c + r0 * u_mid, c + r0 * u_end))
        walk1.append(("arc", c + r1 * u_mid, c + r1 * u_end))
        p = c + r0 * u_end
        h = rot(h, sweep)
        mark = c + (ri + t / 2.0) * u_mid
        infos.append(ArcBendInfo(index=len(infos) + 1, sweep_deg=abs(math.degrees(sweep)), ri=ri, ro=ri + t,
                                 turn="LEFT" if left_turn else "RIGHT", center=np.array([c[0], 0.0, c[1]]),
                                 mid_dir=np.array([u_mid[0], 0.0, u_mid[1]]),
                                 mark_start=np.array([mark[0], 0.0, mark[1]]),
                                 mark_end=np.array([mark[0], float(width), mark[1]])))

    def P(q):
        return gp_Pnt(float(q[0]), 0.0, float(q[1]))

    edges = []

    def seg(a, b):
        edges.append(BRepBuilderAPI_MakeEdge(GC_MakeSegment(P(a), P(b)).Value()).Edge())

    def arc(a, m, b):
        edges.append(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(P(a), P(m), P(b)).Value()).Edge())

    cur = walk0[0][1]
    for item in walk0[1:]:
        if item[0] == "line":
            seg(cur, item[1])
            cur = item[1]
        else:
            arc(cur, item[1], item[2])
            cur = item[2]
    ends1 = [walk1[0][1]] + [it[1] if it[0] == "line" else it[2] for it in walk1[1:]]
    seg(cur, ends1[-1])  # end cap
    for k in range(len(walk1) - 1, 0, -1):
        item, prev = walk1[k], ends1[k - 1]
        if item[0] == "line":
            seg(ends1[k], prev)
        else:
            arc(ends1[k], item[1], prev)
    seg(ends1[0], walk0[0][1])  # start cap
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
    face = BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()
    shape = BRepPrimAPI_MakePrism(face, gp_Vec(0.0, float(width), 0.0)).Shape()
    return shape, infos


def arc_candidates(infos: list[ArcBendInfo], width: float, *, include=None) -> list[dict]:
    """Native-style candidates (mid-radius half-turn lines) of :func:`arc_profile_part` arcs.

    ``include``: indices (1-based) of the arcs to mark; all by default.
    """
    out = []
    for b in infos:
        if include is not None and b.index not in include:
            continue
        out.append({"line_start": b.mark_start.tolist(), "line_end": b.mark_end.tolist(),
                    "target_angle": min(b.sweep_deg, 179.0), "target_radius": b.ri})
    return out


def hem_180_part(ri: float = 0.5, t: float = 1.0, base: float = 40.0, ret: float = 15.0, wall: float = 20.0,
                 width: float = 30.0):
    """An up flange, a base and an exact 180 deg hem folded back over the base (open: ri ~ t/2, closed: ri << t)."""
    ops = [("line", wall), ("arc", 90.0, 1.0), ("line", base), ("arc", 180.0, ri), ("line", ret)]
    return arc_profile_part(ops, t, width, start=(0.0, wall), heading_deg=-90.0)


def teardrop_part(sweep: float = 200.0, ri: float = 1.0, t: float = 1.0, base: float = 40.0, ret: float = 4.0,
                  wall: float = 20.0, width: float = 30.0):
    """Like :func:`hem_180_part` with a reflex (teardrop) hem: the return flange leans back toward the base."""
    ops = [("line", wall), ("arc", 90.0, 1.0), ("line", base), ("arc", float(sweep), ri), ("line", ret)]
    return arc_profile_part(ops, t, width, start=(0.0, wall), heading_deg=-90.0)


def nurbs_convert(shape):
    """The same solid with every surface and curve re-encoded as a B-spline (some exporters do this)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert

    return BRepBuilderAPI_NurbsConvert(shape, True).Shape()


def compound_of(*shapes):
    """A compound of separate solids (a multi-body STEP)."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for sh in shapes:
        builder.Add(comp, sh)
    return comp


def add_rounded_boss(shape, lo_xy, hi_xy, z: float, height: float, corner_radius: float, fillet_radius: float):
    """Fuse a box boss on the skin plane ``z`` and round it: vertical corner rounds plus a fillet
    along its foot, so the boss meets the skin smoothly (a formed feature: cylinders and tori)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    (x0, y0), (x1, y1) = lo_xy, hi_xy
    box = BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z), gp_Pnt(x1, y1, z + height)).Shape()
    box = round_vertical_edges(box, [(x0, y0), (x1, y0), (x0, y1), (x1, y1)], corner_radius)
    op = BRepAlgoAPI_Fuse(shape, box)
    op.Build()
    fused = op.Shape()
    fillet = BRepFilletAPI_MakeFillet(fused)
    explorer = TopExp_Explorer(fused, TopAbs_EDGE)
    seen = set()
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        a, b = curve.Value(curve.FirstParameter()), curve.Value(curve.LastParameter())
        m = curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()))
        on_foot = all(abs(q.Z() - z) < 1e-6 for q in (a, b, m))
        inside = x0 - 1e-6 <= m.X() <= x1 + 1e-6 and y0 - 1e-6 <= m.Y() <= y1 + 1e-6
        key = (round(m.X(), 4), round(m.Y(), 4))
        if on_foot and inside and key not in seen:
            seen.add(key)
            fillet.Add(float(fillet_radius), edge)
        explorer.Next()
    fillet.Build()
    return fillet.Shape()
