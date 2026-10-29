"""OCP B-rep reader for the onboarding compile: STEP file -> :class:`BrepModel`.

This is the only module of the compile that talks to OCP (OpenCascade). It
turns the folded STEP solid into plain numpy records so the flange-graph,
nominal, stage and sampling code stay pure and testable:

- **faces**: surface family (plane, cylinder, ...), outward normal convention
  (OCC: ``du x dv``, negated for REVERSED faces), analytic frames for planes and
  cylinders, area/centroid, the ordered boundary **edge uses** of every wire
  (UV and 3D points, B-rep vertex ids for chaining), and a tessellation;
- **edges**: adjacent faces, curve family, a discretisation and the **dihedral**
  between the two adjacent faces' outward normals (0 deg = smooth: tangent or
  coplanar junction; ~90 deg = a sheet skin meeting a thickness wall).

Coplanar / co-cylindrical face splits are merged first with
``ShapeUpgrade_UnifySameDomain`` (geometry unchanged); the original shape is
kept when unification fails or yields an invalid solid.

**Canonical recognition.** Some exporters write planes, bend cylinders and
straight edges as B-splines. A free-form face that ``GeomLib_IsPlanarSurface``
proves planar (1e-4 mm) is a plane; one that
``ShapeAnalysis_CanonicalRecognition`` proves cylindrical within
``_CYLINDER_RECOGNITION_TOL_MM`` (1 um) is a cylinder (axis, radius and
convexity from the recognised surface; its own (u, v) parametrisation is kept
for sampling, and its angular span comes from the boundary). A free-form edge
whose samples lie within ``_LINE_RECOGNITION_TOL_MM`` (or the edge tolerance)
of their chord is a line. Recognition never changes the geometry, only the
family the flange graph sees; each recognition is noted.

Units are the STEP model units, taken as mm (the recipe records whether the CAD
units were verified). The frame is ``cad_model_mm``: no transform is applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "CadReadError",
    "EdgeUse",
    "FaceLoop",
    "BrepFace",
    "BrepEdge",
    "BrepModel",
    "quiet_occ",
    "load_step",
    "shape_to_model",
    "SURFACE_FAMILIES",
]

#: Surface family names used throughout the compile.
SURFACE_FAMILIES = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bezier",
    "bspline",
    "revolution",
    "extrusion",
    "offset",
    "other",
)

_CIRCLE_STEP_RAD = math.radians(5.0)
_MAX_EDGE_POINTS = 400
_PLANAR_RECOGNITION_TOL_MM = 1e-4
_CYLINDER_RECOGNITION_TOL_MM = 1e-3
_LINE_RECOGNITION_TOL_MM = 1e-4
#: A STEP with several solids is read as its largest solid when every other solid is smaller than this
#: share of it: the part is then the clear majority body and the others are attached hardware, weld-on
#: brackets or stray bodies (noted, not inspected). At or above it, which body is "the part" is ambiguous
#: and the read is refused (MULTIPLE_SOLID_BODIES).
_DOMINANT_SOLID_RATIO = 0.5


class CadReadError(Exception):
    """The STEP file cannot be turned into a single sheet-metal solid."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class EdgeUse:
    """One oriented use of a B-rep edge inside a face wire (traversal order)."""

    edge: int
    reversed: bool
    v_start: int
    v_end: int
    degenerate: bool
    uv: np.ndarray  # (K,2) including both endpoints, traversal order
    xyz: np.ndarray  # (K,3) surface(uv)


@dataclass
class FaceLoop:
    uses: list[EdgeUse]
    is_outer: bool

    def uv_polygon(self) -> np.ndarray:
        parts = [u.uv[:-1] for u in self.uses if len(u.uv) >= 2]
        return np.concatenate(parts) if parts else np.zeros((0, 2))


@dataclass
class BrepFace:
    index: int
    surface: str
    kind: str  # 'plane' | 'cylinder' | 'other'
    reversed: bool
    area: float
    centroid: np.ndarray
    uv_bounds: tuple[float, float, float, float]
    loops: list[FaceLoop]
    edges: list[int]
    #: plane: point on plane and outward unit normal; plane frame (x_dir, y_dir)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray | None = None
    x_dir: np.ndarray | None = None
    y_dir: np.ndarray | None = None
    #: cylinder: Ax3 main direction, radius, Ax3 handedness, convexity of the outward normal
    axis: np.ndarray | None = None
    radius: float | None = None
    direct: bool = True
    convex: bool | None = None
    #: True when sampling may use the analytic (u, v) map of plane/cylinder
    analytic_uv: bool = False
    #: cylinder: angular extent about the axis (rad); 2 pi for a closed (seamed) face
    angular_span: float = 0.0
    mesh_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    mesh_triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    #: OCC surface handle evaluator (kept for generic sampling); not pickled
    _evaluator: object | None = None

    def outward_normal_at(self, u: float, v: float) -> np.ndarray:
        """Outward unit normal at surface parameters (u, v)."""
        if self.kind == "plane" and self.normal is not None:
            return self.normal.copy()
        if self.kind == "cylinder" and self.analytic_uv:
            radial = math.cos(u) * self.x_dir + math.sin(u) * self.y_dir
            return radial if self.convex else -radial
        ev = self._evaluator
        if ev is None:
            raise ValueError("face has no evaluator")
        return ev.normal(u, v, self.reversed)

    def point_at(self, u: float, v: float) -> np.ndarray:
        if self.analytic_uv and self.kind == "plane":
            return self.origin + u * self.x_dir + v * self.y_dir
        if self.analytic_uv and self.kind == "cylinder":
            return self.origin + self.radius * (math.cos(u) * self.x_dir + math.sin(u) * self.y_dir) + v * self.axis
        ev = self._evaluator
        if ev is None:
            raise ValueError("face has no evaluator")
        return ev.point(u, v)


@dataclass
class BrepEdge:
    index: int
    faces: tuple[int, ...]
    curve: str  # 'line' | 'circle' | 'other'
    points: np.ndarray  # (K,3)
    length: float
    direction: np.ndarray | None
    degenerate: bool
    seam: bool
    dihedral_deg: float | None = None


@dataclass
class BrepModel:
    faces: list[BrepFace]
    edges: list[BrepEdge]
    vertex_count: int
    solid_count: int
    volume: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    unified: bool
    notes: list[str] = field(default_factory=list)

    def neighbors(self, face_index: int) -> list[tuple[int, int]]:
        """``(edge index, other face index)`` pairs of a face."""
        out = []
        for e in self.faces[face_index].edges:
            for g in self.edges[e].faces:
                if g != face_index:
                    out.append((e, g))
        return out


class _SurfaceEvaluator:
    """Point/normal evaluation through an OCC adaptor (generic surfaces)."""

    def __init__(self, adaptor) -> None:
        from OCP.gp import gp_Pnt, gp_Vec

        self._a = adaptor
        self._p = gp_Pnt()
        self._du = gp_Vec()
        self._dv = gp_Vec()

    def point(self, u: float, v: float) -> np.ndarray:
        p = self._a.Value(u, v)
        return np.array([p.X(), p.Y(), p.Z()])

    def normal(self, u: float, v: float, reversed_face: bool) -> np.ndarray:
        self._a.D1(u, v, self._p, self._du, self._dv)
        n = np.cross([self._du.X(), self._du.Y(), self._du.Z()], [self._dv.X(), self._dv.Y(), self._dv.Z()])
        norm = float(np.linalg.norm(n))
        if norm <= 1e-14:
            # degenerate point (e.g. cone apex): nudge inside the domain
            self._a.D1(u + 1e-6, v + 1e-6, self._p, self._du, self._dv)
            n = np.cross([self._du.X(), self._du.Y(), self._du.Z()], [self._dv.X(), self._dv.Y(), self._dv.Z()])
            norm = float(np.linalg.norm(n))
            if norm <= 1e-14:
                return np.array([0.0, 0.0, 1.0])
        n = n / norm
        return -n if reversed_face else n


_QUIET = False


def quiet_occ() -> None:
    """Silence OCC messenger printers (the STEP reader/writer talk on stdout)."""
    global _QUIET
    if _QUIET:
        return
    try:
        from OCP.Message import Message, Message_Gravity

        printers = Message.DefaultMessenger_s().Printers()
        for i in range(1, printers.Size() + 1):
            printers.Value(i).SetTraceLevel(Message_Gravity.Message_Fail)
    except Exception:  # pragma: no cover - cosmetic only
        pass
    _QUIET = True


def _vec(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)


def load_step(path: str | Path, *, linear_deflection: float = 0.1, angular_deflection: float = 0.35,
              unify: bool = True) -> BrepModel:
    """Read a STEP file and return its single solid as a :class:`BrepModel`."""
    quiet_occ()
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    p = Path(path)
    if not p.is_file():
        raise CadReadError("CAD_FILE_MISSING", str(p))
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(p))
    if status != IFSelect_RetDone:
        raise CadReadError("STEP_PARSE_FAILED", f"reader status {status}")
    if int(reader.TransferRoots()) <= 0:
        raise CadReadError("STEP_NO_TRANSFERABLE_ROOTS")
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise CadReadError("STEP_EMPTY_SHAPE")
    return shape_to_model(shape, linear_deflection=linear_deflection, angular_deflection=angular_deflection,
                          unify=unify)


def _select_solid(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid = TopoDS.Solid_s(explorer.Current())
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, props)
        solids.append((abs(float(props.Mass())), solid))
        explorer.Next()
    if not solids:
        raise CadReadError("NO_SOLID_BODY", "the STEP holds no solid (surface model or empty)")
    solids.sort(key=lambda item: -item[0])
    if len(solids) > 1 and solids[1][0] >= _DOMINANT_SOLID_RATIO * solids[0][0]:
        raise CadReadError("MULTIPLE_SOLID_BODIES", f"{len(solids)} solids; the second has "
                           f"{solids[1][0] / max(solids[0][0], 1e-12):.1%} of the largest volume")
    return solids[0][1], len(solids), solids[0][0]


def _extra_solid_note(solid_count: int, shape, volume: float) -> str | None:
    """A note naming the bodies ignored next to the inspected (dominant) solid, or None."""
    if solid_count <= 1:
        return None
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    vols = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(TopoDS.Solid_s(explorer.Current()), props)
        vols.append(abs(float(props.Mass())))
        explorer.Next()
    vols.sort(reverse=True)
    second = vols[1] / max(volume, 1e-12) if len(vols) > 1 else 0.0
    return (f"MULTIPLE_SOLID_BODIES_DOMINANT: {solid_count} solids; the largest is the part, the others (largest "
            f"{second:.1%} of its volume) are attached or stray bodies and are not inspected")


def _unify(solid):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    try:
        tool = ShapeUpgrade_UnifySameDomain(solid, True, True, False)
        tool.Build()
        out = tool.Shape()
        if out is None or out.IsNull() or not BRepCheck_Analyzer(out).IsValid():
            return solid, False
        return out, True
    except Exception:
        return solid, False


def _surface_family(stype) -> str:
    from OCP.GeomAbs import (
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_OffsetSurface,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfExtrusion,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )

    return {
        GeomAbs_Plane: "plane",
        GeomAbs_Cylinder: "cylinder",
        GeomAbs_Cone: "cone",
        GeomAbs_Sphere: "sphere",
        GeomAbs_Torus: "torus",
        GeomAbs_BezierSurface: "bezier",
        GeomAbs_BSplineSurface: "bspline",
        GeomAbs_SurfaceOfRevolution: "revolution",
        GeomAbs_SurfaceOfExtrusion: "extrusion",
        GeomAbs_OffsetSurface: "offset",
    }.get(stype, "other")


def _curve_params(curve) -> tuple[str, np.ndarray]:
    """Curve family and parameter samples (including both ends) for an edge adaptor."""
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line

    first, last = float(curve.FirstParameter()), float(curve.LastParameter())
    ctype = curve.GetType()
    if ctype == GeomAbs_Line:
        return "line", np.array([first, last])
    if ctype == GeomAbs_Circle:
        n = max(2, int(math.ceil(abs(last - first) / _CIRCLE_STEP_RAD)))
        return "circle", np.linspace(first, last, n + 1)
    try:
        length = float(GCPnts_AbscissaPoint.Length_s(curve))
    except Exception:
        length = 10.0
    n = int(min(_MAX_EDGE_POINTS, max(4, math.ceil(length / 0.5))))
    return "other", np.linspace(first, last, n + 1)


def _is_straight(pts: np.ndarray, tol: float) -> bool:
    """True when every sample lies within ``tol`` of the chord (a free-form edge that is a line)."""
    if len(pts) < 2:
        return False
    chord = pts[-1] - pts[0]
    length = float(np.linalg.norm(chord))
    if length <= 1e-9:
        return False
    d = chord / length
    rel = pts - pts[0]
    along = rel @ d
    off = np.linalg.norm(rel - np.outer(along, d), axis=1)
    return bool(float(off.max()) <= tol and float(along.min()) >= -tol and float(along.max()) <= length + tol)


def _recognize_cylinder(face, tol: float):
    """``gp_Cylinder`` of a free-form face that is a cylinder within ``tol`` mm, else ``None``."""
    from OCP.gp import gp_Cylinder
    from OCP.ShapeAnalysis import ShapeAnalysis_CanonicalRecognition

    try:
        cyl = gp_Cylinder()
        if ShapeAnalysis_CanonicalRecognition(face).IsCylinder(float(tol), cyl) and cyl.Radius() > 0:
            return cyl
    except Exception:
        return None
    return None


def _angular_span(points: np.ndarray, origin: np.ndarray, x_dir: np.ndarray, y_dir: np.ndarray) -> float:
    """Smallest angular range (rad) about a cylinder axis that contains every boundary point."""
    rel = points - origin
    ang = np.sort(np.arctan2(rel @ y_dir, rel @ x_dir))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2.0 * math.pi]))
    return float(2.0 * math.pi - gaps.max())


def shape_to_model(shape, *, linear_deflection: float = 0.1, angular_deflection: float = 0.35,
                   unify: bool = True) -> BrepModel:
    """Convert an OCC shape holding one solid into a :class:`BrepModel`."""
    quiet_occ()
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.GeomLib import GeomLib_IsPlanarSurface
    from OCP.GProp import GProp_GProps
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX, TopAbs_WIRE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

    solid, solid_count, volume = _select_solid(shape)
    notes: list[str] = []
    extra = _extra_solid_note(solid_count, shape, volume)
    if extra:
        notes.append(extra)
    unified = False
    if unify:
        solid, unified = _unify(solid)
        if not unified:
            notes.append("UnifySameDomain unavailable or invalid; original faces kept")

    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(solid, TopAbs_FACE, fmap)
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(solid, TopAbs_EDGE, TopAbs_FACE, emap)
    vmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(solid, TopAbs_VERTEX, vmap)
    n_faces, n_edges = fmap.Extent(), emap.Extent()
    if n_faces == 0:
        raise CadReadError("STEP_HAS_NO_FACES")
    if n_faces > 20000 or n_edges > 80000:
        raise CadReadError("GEOMETRY_BUDGET_EXCEEDED", f"{n_faces} faces / {n_edges} edges")

    mesher = BRepMesh_IncrementalMesh(solid, float(linear_deflection), False, float(angular_deflection), False)
    mesher.Perform()

    # -- edges: adjacency, curve family, discretisation ------------------------
    recognized_lines = 0
    edges: list[BrepEdge] = []
    edge_shapes = []
    edge_params: list[np.ndarray] = []
    for ei in range(1, n_edges + 1):
        edge = TopoDS.Edge_s(emap.FindKey(ei))
        edge_shapes.append(edge)
        face_ids: list[int] = []
        for f in emap.FindFromIndex(ei):
            fi = fmap.FindIndex(f) - 1
            if fi >= 0 and fi not in face_ids:
                face_ids.append(fi)
        degenerate = bool(BRep_Tool.Degenerated_s(edge))
        if degenerate:
            edges.append(BrepEdge(ei - 1, tuple(face_ids), "other", np.zeros((0, 3)), 0.0, None, True, False))
            edge_params.append(np.zeros(0))
            continue
        curve = BRepAdaptor_Curve(edge)
        family, params = _curve_params(curve)
        pts = np.array([_vec(curve.Value(float(t))) for t in params])
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
        if family == "other" and _is_straight(pts, max(_LINE_RECOGNITION_TOL_MM, float(BRep_Tool.Tolerance_s(edge)))):
            family = "line"
            recognized_lines += 1
        direction = None
        if family == "line" and length > 1e-9:
            direction = (pts[-1] - pts[0]) / np.linalg.norm(pts[-1] - pts[0])
        seam = False
        if len(face_ids) == 1:
            seam = bool(BRep_Tool.IsClosed_s(edge, TopoDS.Face_s(fmap.FindKey(face_ids[0] + 1))))
        edges.append(BrepEdge(ei - 1, tuple(face_ids), family, pts, length, direction, False, seam))
        edge_params.append(params)

    # -- faces ---------------------------------------------------------------
    faces: list[BrepFace] = []
    projectors: list[ShapeAnalysis_Surface] = []
    for fi in range(1, n_faces + 1):
        face = TopoDS.Face_s(fmap.FindKey(fi))
        adaptor = BRepAdaptor_Surface(face, True)
        family = _surface_family(adaptor.GetType())
        reversed_face = face.Orientation() == TopAbs_REVERSED
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = abs(float(props.Mass()))
        centroid = _vec(props.CentreOfMass())
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        rec = BrepFace(fi - 1, family, "other", reversed_face, area, centroid, (u0, u1, v0, v1), [], [])
        rec._evaluator = _SurfaceEvaluator(adaptor)
        if family == "plane":
            pos = adaptor.Plane().Position()
            x_dir, y_dir = _vec(pos.XDirection()), _vec(pos.YDirection())
            n_param = np.cross(x_dir, y_dir)
            rec.kind = "plane"
            rec.origin = _vec(pos.Location())
            rec.x_dir, rec.y_dir = x_dir, y_dir
            rec.normal = -n_param if reversed_face else n_param
            rec.analytic_uv = True
        elif family == "cylinder":
            cyl = adaptor.Cylinder()
            pos = cyl.Position()
            rec.kind = "cylinder"
            rec.origin = _vec(pos.Location())
            rec.axis = _vec(pos.Direction())
            rec.x_dir, rec.y_dir = _vec(pos.XDirection()), _vec(pos.YDirection())
            rec.radius = float(cyl.Radius())
            rec.direct = bool(pos.Direct())
            # du x dv points away from the axis for a direct Ax3; outward flips for REVERSED faces.
            away = rec.direct != reversed_face
            rec.convex = bool(away)
            rec.analytic_uv = True
            rec.angular_span = float(u1 - u0)
        else:
            # Free-form faces that are exactly planar are recognised as planes.
            try:
                surf = BRep_Tool.Surface_s(face)
                planar = GeomLib_IsPlanarSurface(surf, _PLANAR_RECOGNITION_TOL_MM)
                if planar.IsPlanar():
                    pln = planar.Plan()
                    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                    n_out = rec._evaluator.normal(um, vm, reversed_face)
                    n_pln = _vec(pln.Axis().Direction())
                    rec.kind = "plane"
                    rec.origin = _vec(pln.Location())
                    rec.normal = n_pln if float(n_pln @ n_out) >= 0 else -n_pln
                    pos = pln.Position()
                    rec.x_dir, rec.y_dir = _vec(pos.XDirection()), _vec(pos.YDirection())
                    rec.analytic_uv = False
                    notes.append(f"face {fi - 1}: {family} recognised as plane")
            except Exception:
                pass
            if rec.kind == "other":
                cyl = _recognize_cylinder(face, _CYLINDER_RECOGNITION_TOL_MM)
                if cyl is not None:
                    pos = cyl.Position()
                    rec.kind = "cylinder"
                    rec.origin = _vec(pos.Location())
                    rec.axis = _vec(pos.Direction())
                    rec.x_dir, rec.y_dir = _vec(pos.XDirection()), _vec(pos.YDirection())
                    rec.radius = float(cyl.Radius())
                    rec.direct = bool(pos.Direct())
                    # convexity from the face's own outward normal at its parametric centre
                    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                    q = rec._evaluator.point(um, vm)
                    radial = q - rec.origin
                    radial = radial - float(radial @ rec.axis) * rec.axis
                    rec.convex = bool(float(rec._evaluator.normal(um, vm, reversed_face) @ radial) > 0)
                    rec.analytic_uv = False  # sampled through the face's own (u, v)
                    notes.append(f"face {fi - 1}: {family} recognised as cylinder (r {rec.radius:.4f} mm)")
        # boundary loops
        outer = BRepTools.OuterWire_s(face)
        wires = TopExp_Explorer(face, TopAbs_WIRE)
        loops: list[FaceLoop] = []
        face_edges: list[int] = []
        while wires.More():
            wire = TopoDS.Wire_s(wires.Current())
            uses: list[EdgeUse] = []
            wexp = BRepTools_WireExplorer(wire, face)
            while wexp.More():
                oedge = wexp.Current()
                ei = emap.FindIndex(oedge) - 1
                rev = oedge.Orientation() == TopAbs_REVERSED
                vs = vmap.FindIndex(TopExp.FirstVertex_s(oedge, True)) - 1
                ve = vmap.FindIndex(TopExp.LastVertex_s(oedge, True)) - 1
                c2 = BRepAdaptor_Curve2d(oedge, face)
                f2, l2 = float(c2.FirstParameter()), float(c2.LastParameter())
                base = edge_params[ei] if ei >= 0 else np.zeros(0)
                if ei >= 0 and edges[ei].degenerate or len(base) < 2:
                    t2 = np.linspace(f2, l2, 9)
                else:
                    t3a, t3b = float(base[0]), float(base[-1])
                    span = (t3b - t3a) if abs(t3b - t3a) > 1e-15 else 1.0
                    t2 = f2 + (base - t3a) / span * (l2 - f2)
                if rev:
                    t2 = t2[::-1]
                uv = np.array([(lambda q: (q.X(), q.Y()))(c2.Value(float(t))) for t in t2])
                xyz = np.array([rec.point_at(float(a), float(b)) for a, b in uv])
                uses.append(EdgeUse(ei, rev, vs, ve, bool(ei >= 0 and edges[ei].degenerate), uv, xyz))
                if ei >= 0 and ei not in face_edges:
                    face_edges.append(ei)
                wexp.Next()
            loops.append(FaceLoop(uses, bool(wire.IsSame(outer))))
            wires.Next()
        rec.loops = loops
        rec.edges = face_edges
        if rec.kind == "cylinder" and not rec.analytic_uv:
            if any(edges[e].seam for e in face_edges):
                rec.angular_span = 2.0 * math.pi
            else:
                bpts = [u.xyz for lp in loops for u in lp.uses if len(u.xyz)]
                rec.angular_span = (_angular_span(np.concatenate(bpts), rec.origin, rec.x_dir, rec.y_dir)
                                    if bpts else 0.0)
        # tessellation (outward-oriented triangles)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None and tri.NbTriangles() > 0:
            trsf = loc.Transformation()
            verts = np.array([_vec(tri.Node(i).Transformed(trsf)) for i in range(1, tri.NbNodes() + 1)])
            tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
            if reversed_face:
                tris = tris[:, [0, 2, 1]]
            rec.mesh_vertices, rec.mesh_triangles = verts, tris
        faces.append(rec)
        projectors.append(ShapeAnalysis_Surface(BRep_Tool.Surface_s(face)))

    if recognized_lines:
        notes.append(f"{recognized_lines} free-form edges recognised as lines")

    # -- edge dihedrals --------------------------------------------------------
    for edge in edges:
        if edge.degenerate or len(edge.faces) != 2 or len(edge.points) < 2:
            continue
        fa, fb = edge.faces
        pts = edge.points
        if len(pts) == 2:
            samples = [pts[0] + s * (pts[1] - pts[0]) for s in (0.25, 0.5, 0.75)]
        else:
            idx = sorted({len(pts) // 4, len(pts) // 2, (3 * len(pts)) // 4})
            samples = [pts[i] for i in idx]
        worst = 0.0
        ok = True
        for q in samples:
            normals = []
            for f in (fa, fb):
                try:
                    from OCP.gp import gp_Pnt

                    uv = projectors[f].ValueOfUV(gp_Pnt(float(q[0]), float(q[1]), float(q[2])), 1e-7)
                    normals.append(faces[f].outward_normal_at(float(uv.X()), float(uv.Y())))
                except Exception:
                    ok = False
                    break
            if not ok:
                break
            c = float(np.clip(normals[0] @ normals[1], -1.0, 1.0))
            worst = max(worst, math.degrees(math.acos(c)))
        if ok:
            edge.dihedral_deg = worst

    bbox = Bnd_Box()
    BRepBndLib.Add_s(solid, bbox)
    x0, y0, z0, x1, y1, z1 = bbox.Get()
    return BrepModel(
        faces=faces,
        edges=edges,
        vertex_count=vmap.Extent(),
        solid_count=solid_count,
        volume=volume,
        bbox_min=np.array([x0, y0, z0]),
        bbox_max=np.array([x1, y1, z1]),
        unified=unified,
        notes=notes,
    )
