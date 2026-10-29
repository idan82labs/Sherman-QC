"""Real OCP (OpenCascade) BReps for synthetic parts, STEP I/O and labelled tessellation.

Construction (:func:`build_pieces`): every panel of a :class:`~.geometry.FoldedPart`
becomes a prism of thickness ``t`` (its effective polygon with circular holes,
so hole walls are true cylinders) and every formed bend becomes the
revolution of the sheet cross-section at the parent's tangent line about the
bend axis (``BRepPrimAPI_MakeRevol``), which yields **true cylindrical** inner
and outer bend faces of radius ``Ri`` and ``Ri + t``. Pieces meet exactly on
shared *joint* faces.

- :func:`fused_solid` fuses the pieces and unifies same-domain faces into one
  valid closed solid (planes + cylinders only); that solid is what
  :func:`write_step` writes (AP214, mm), byte-deterministic: the header time
  stamp and the product name (which OCCT numbers per process) are normalised.
- :func:`tessellate` meshes the pieces *without* their joint faces, so the
  union of the returned triangles is exactly the closed outer surface of the
  fused solid, and every triangle carries an exact label
  (segment kind / index / flat-frame skin) for scanner truth.

OCCT's console printers are silenced at import (transfer statistics would
otherwise go to stdout, which the CLI keeps clean).
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Message import Message, Message_Gravity
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from .geometry import FREE, FoldedPart, GeometryError, apply_dirs, apply_points

__all__ = [
    "SEG_PANEL",
    "SEG_BEND",
    "SEG_EDGE",
    "SEG_BEND_EDGE",
    "SKIN_EDGE",
    "FaceRole",
    "Piece",
    "TessellatedSurface",
    "build_pieces",
    "fused_solid",
    "solid_volume",
    "analytic_volume",
    "check_single_valid_solid",
    "write_step",
    "read_step",
    "tessellate",
    "mirror_shape",
    "face_type_counts",
]

SEG_PANEL = 0
SEG_BEND = 1
SEG_EDGE = 2  # panel edge wall (cut edge, hole wall)
SEG_BEND_EDGE = 3  # side wall of a bend sector
SKIN_EDGE = 2
_FUZZY = 1e-5
_STEP_TIMESTAMP = "2026-01-01T00:00:00"


def _silence_occt() -> None:
    try:
        printers = Message.DefaultMessenger_s().Printers()
        for printer in printers:
            printer.SetTraceLevel(Message_Gravity.Message_Fail)
    except Exception:  # pragma: no cover - printers API differs across OCP builds
        pass


_silence_occt()


@dataclass(frozen=True)
class FaceRole:
    """Label of a piece face: segment kind (0 panel skin, 1 bend skin, 2 panel edge
    wall, 3 bend side wall), segment index (panel index for kinds 0/2, bend index
    for kinds 1/3, in design order), flat-frame skin (0, 1, or 2 for walls) and
    whether the face is an internal joint (never on the surface)."""

    kind: int
    index: int
    skin: int
    joint: bool
    detail: str = ""


@dataclass
class Piece:
    kind: str  # "panel" | "bend"
    key: str
    shape: TopoDS_Shape
    faces: list[tuple[TopoDS_Shape, FaceRole]]


@dataclass(frozen=True)
class TessellatedSurface:
    """Closed outer surface of a part as labelled triangles (CAD frame, mm)."""

    vertices: np.ndarray  # (V, 3) float64
    triangles: np.ndarray  # (T, 3) int64, outward winding
    seg_kind: np.ndarray  # (T,) uint8
    seg_index: np.ndarray  # (T,) int16
    skin: np.ndarray  # (T,) uint8 (flat-frame skin; 2 = edge wall)

    @property
    def triangle_normals(self) -> np.ndarray:
        v = self.vertices
        t = self.triangles
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(norm > 0, norm, 1.0)

    @property
    def triangle_areas(self) -> np.ndarray:
        v = self.vertices
        t = self.triangles
        return 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)

    def transformed(self, m: np.ndarray) -> "TessellatedSurface":
        """Apply a rigid (or mirror) 4x4; a mirror flips the winding to stay outward."""
        v = apply_points(m, self.vertices)
        tris = self.triangles
        if np.linalg.det(m[:3, :3]) < 0:
            tris = tris[:, [0, 2, 1]]
        return TessellatedSurface(v, tris.copy(), self.seg_kind.copy(), self.seg_index.copy(), self.skin.copy())


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _pnt(p) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _edge(a, b):
    return BRepBuilderAPI_MakeEdge(_pnt(a), _pnt(b)).Edge()


def _wire_from_points(points3: np.ndarray):
    """Closed polygon wire; returns (wire, [edge per polygon edge in order])."""
    mk = BRepBuilderAPI_MakeWire()
    edges = []
    n = len(points3)
    for i in range(n):
        mk.Add(_edge(points3[i], points3[(i + 1) % n]))
        # MakeWire may rebuild an edge to share vertices: keep the edge actually in the wire
        edges.append(mk.Edge())
    if not mk.IsDone():
        raise GeometryError("polygon wire construction failed")
    return mk.Wire(), edges


def _panel_piece(fp: FoldedPart, panel_key: str, panel_index: int) -> Piece:
    w = fp.world[panel_key]
    outline = fp.outline[panel_key]
    tags = fp.tags[panel_key]
    pts3 = apply_points(w, np.column_stack([outline, np.zeros(len(outline))]))
    normal = apply_dirs(w, np.array([0.0, 0.0, 1.0]))
    t = fp.design.thickness
    wire, edges = _wire_from_points(pts3)
    mf = BRepBuilderAPI_MakeFace(wire, True)
    hole_edges = []
    for hole in fp.design.panel(panel_key).holes:
        c = apply_points(w, np.array([[hole.center[0], hole.center[1], 0.0]]))[0]
        circ = gp_Circ(gp_Ax2(_pnt(c), gp_Dir(*normal)), float(hole.radius))
        e = BRepBuilderAPI_MakeEdge(circ).Edge()
        hw = BRepBuilderAPI_MakeWire(e).Wire()
        hw.Reverse()
        mf.Add(hw)
        hole_edges.append(e)
    if not mf.IsDone():
        raise GeometryError(f"panel {panel_key}: face construction failed")
    face = mf.Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(*(normal * t)))
    prism.Build()
    if not prism.IsDone():
        raise GeometryError(f"panel {panel_key}: prism failed")
    faces: list[tuple[TopoDS_Shape, FaceRole]] = []
    faces.append((prism.FirstShape(), FaceRole(SEG_PANEL, panel_index, 1, False, "skin1")))
    faces.append((prism.LastShape(), FaceRole(SEG_PANEL, panel_index, 0, False, "skin0")))
    for e, tag in zip(edges, tags):
        gen = prism.Generated(e)
        for s in gen:
            joint = tag != FREE
            faces.append((s, FaceRole(SEG_EDGE, panel_index, SKIN_EDGE, joint, f"wall:{tag}")))
    for e in hole_edges:
        for s in prism.Generated(e):
            faces.append((s, FaceRole(SEG_EDGE, panel_index, SKIN_EDGE, False, "hole")))
    return Piece("panel", panel_key, prism.Shape(), faces)


def _bend_piece(fp: FoldedPart, bend_key: str, bend_index: int) -> Piece:
    g = fp.bend_geo[bend_key]
    w = fp.world[g.parent]
    t = g.thickness
    tangent = g.line_point - (g.ba_actual / 2.0) * g.m
    e0, e1 = g.extent
    p0 = np.array([*(tangent + e0 * g.d), 0.0])
    p1 = np.array([*(tangent + e1 * g.d), 0.0])
    rect_flat = np.array([p0, p1, p1 + [0, 0, t], p0 + [0, 0, t]])
    rect = apply_points(w, rect_flat)
    wire, edges = _wire_from_points(rect)
    face = BRepBuilderAPI_MakeFace(wire, True).Face()
    axis_p = apply_points(w, g.axis_point[None])[0]
    axis_d = apply_dirs(w, g.axis_dir)
    rev = BRepPrimAPI_MakeRevol(face, gp_Ax1(_pnt(axis_p), gp_Dir(*axis_d)), g.alpha)
    rev.Build()
    if not rev.IsDone():
        raise GeometryError(f"bend {bend_key}: revolution failed")
    # edges: 0: z=0 along d (flat skin 1), 1: at e1 vertical, 2: z=t (flat skin 0), 3: at e0 vertical
    faces: list[tuple[TopoDS_Shape, FaceRole]] = [
        (rev.FirstShape(), FaceRole(SEG_BEND_EDGE, bend_index, SKIN_EDGE, True, "joint:parent")),
        (rev.LastShape(), FaceRole(SEG_BEND_EDGE, bend_index, SKIN_EDGE, True, "joint:child")),
    ]
    roles = [
        FaceRole(SEG_BEND, bend_index, 1, False, "skin1"),
        FaceRole(SEG_BEND_EDGE, bend_index, SKIN_EDGE, False, "side:e1"),
        FaceRole(SEG_BEND, bend_index, 0, False, "skin0"),
        FaceRole(SEG_BEND_EDGE, bend_index, SKIN_EDGE, False, "side:e0"),
    ]
    for e, role in zip(edges, roles):
        for s in rev.Generated(e):
            faces.append((s, role))
    return Piece("bend", bend_key, rev.Shape(), faces)


def build_pieces(fp: FoldedPart) -> list[Piece]:
    """One prism per panel and one revolved sector per *formed* bend (CAD frame).

    Mirrored parts are not built directly (a reflection flips wire orientation
    and rotation sense): build the unmirrored part, then reflect the fused
    shape with :func:`mirror_shape` and the tessellation with
    :meth:`TessellatedSurface.transformed`."""
    if fp.is_mirrored():
        raise GeometryError("build pieces from the unmirrored part and reflect the result")
    panel_index = {p.key: i for i, p in enumerate(fp.design.panels)}
    bend_index = {b.key: i for i, b in enumerate(fp.design.bends)}
    pieces = [_panel_piece(fp, p.key, panel_index[p.key]) for p in fp.design.panels]
    for b in fp.design.bends:
        if fp.bend_geo[b.key].formed:
            pieces.append(_bend_piece(fp, b.key, bend_index[b.key]))
    return pieces


def mirror_shape(shape: TopoDS_Shape, m: np.ndarray) -> TopoDS_Shape:
    """Apply a 4x4 (rigid or reflection) to an OCP shape (copying geometry)."""
    tr = gp_Trsf()
    r = m[:3, :3]
    tr.SetValues(
        float(r[0, 0]), float(r[0, 1]), float(r[0, 2]), float(m[0, 3]),
        float(r[1, 0]), float(r[1, 1]), float(r[1, 2]), float(m[1, 3]),
        float(r[2, 0]), float(r[2, 1]), float(r[2, 2]), float(m[2, 3]),
    )
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def fused_solid(pieces: list[Piece]) -> TopoDS_Shape:
    """Fuse the pieces into one solid and unify same-domain faces/edges."""
    if not pieces:
        raise GeometryError("no pieces to fuse")
    if len(pieces) == 1:
        shape = pieces[0].shape
    else:
        fuse = BRepAlgoAPI_Fuse()
        args = TopTools_ListOfShape()
        args.Append(pieces[0].shape)
        tools = TopTools_ListOfShape()
        for p in pieces[1:]:
            tools.Append(p.shape)
        fuse.SetArguments(args)
        fuse.SetTools(tools)
        fuse.SetFuzzyValue(_FUZZY)
        fuse.SetRunParallel(False)
        fuse.Build()
        if not fuse.IsDone():
            raise GeometryError("boolean fuse failed")
        shape = fuse.Shape()
    unify = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    unify.Build()
    return unify.Shape()


def _count(shape: TopoDS_Shape, kind) -> int:
    ex = TopExp_Explorer(shape, kind)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def check_single_valid_solid(shape: TopoDS_Shape) -> None:
    """Raise unless ``shape`` is one valid solid with one closed shell."""
    if not BRepCheck_Analyzer(shape).IsValid():
        raise GeometryError("BRep is not valid")
    if _count(shape, TopAbs_SOLID) != 1 or _count(shape, TopAbs_SHELL) != 1:
        raise GeometryError("BRep is not a single solid with one shell (pieces disconnected or self-intersecting)")


def solid_volume(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def analytic_volume(fp: FoldedPart) -> float:
    """Exact solid volume from the model: panel areas x t plus annular sectors."""
    t = fp.design.thickness
    vol = 0.0
    for p in fp.design.panels:
        pts = fp.outline[p.key]
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        area -= sum(math.pi * h.radius**2 for h in p.holes)
        vol += area * t
    for g in fp.bend_geo.values():
        if g.formed:
            span = g.extent[1] - g.extent[0]
            vol += 0.5 * g.alpha * (g.outer_radius**2 - g.inner_radius**2) * span
    return vol


def face_type_counts(shape: TopoDS_Shape) -> dict[str, int]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane

    counts = {"plane": 0, "cylinder": 0, "other": 0}
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        st = BRepAdaptor_Surface(TopoDS.Face_s(ex.Current())).GetType()
        if st == GeomAbs_Plane:
            counts["plane"] += 1
        elif st == GeomAbs_Cylinder:
            counts["cylinder"] += 1
        else:
            counts["other"] += 1
        ex.Next()
    return counts


# ---------------------------------------------------------------------------
# STEP I/O
# ---------------------------------------------------------------------------

_PRODUCT_RE = re.compile(r"(#\d+\s*=\s*PRODUCT\()'[^']*'\s*,\s*'[^']*'", re.S)


def write_step(shape: TopoDS_Shape, path: str | os.PathLike, product_name: str) -> str:
    """Write ``shape`` as AP214 STEP in mm, byte-deterministic; returns the sha256.

    The product name must be printable ASCII without quotes."""
    import hashlib

    if not re.fullmatch(r"[A-Za-z0-9._ -]{1,64}", product_name):
        raise ValueError("product_name must be 1-64 chars of [A-Za-z0-9._ -]")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise GeometryError("STEP transfer failed")
    from OCP.APIHeaderSection import APIHeaderSection_MakeHeader
    from OCP.TCollection import TCollection_HAsciiString

    header = APIHeaderSection_MakeHeader(writer.Model())
    # header name from the product (not the output path) so bytes depend on content only
    header.SetName(TCollection_HAsciiString(f"{product_name}.step"))
    header.SetTimeStamp(TCollection_HAsciiString(_STEP_TIMESTAMP))
    header.SetAuthorValue(1, TCollection_HAsciiString("sherman_qc_e2_synthetic"))
    header.SetOrganizationValue(1, TCollection_HAsciiString("sherman_qc"))
    header.SetPreprocessorVersion(TCollection_HAsciiString("OCP 7.9"))
    header.SetOriginatingSystem(TCollection_HAsciiString("sherman_qc_e2_synthetic"))
    header.SetAuthorisation(TCollection_HAsciiString(""))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".step_", suffix=".step", dir=str(target.parent))
    os.close(fd)
    try:
        if writer.Write(tmp) != IFSelect_RetDone:
            raise GeometryError("STEP write failed")
        text = Path(tmp).read_text(encoding="ascii")
        text, n = _PRODUCT_RE.subn(lambda m: f"{m.group(1)}'{product_name}','{product_name}'", text)
        if n != 1:
            raise GeometryError("unexpected STEP product records")
        data = text.encode("ascii")
        Path(tmp).write_bytes(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(data).hexdigest()


def read_step(path: str | os.PathLike) -> TopoDS_Shape:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise GeometryError(f"cannot read STEP {path}")
    reader.TransferRoots()
    return reader.OneShape()


# ---------------------------------------------------------------------------
# tessellation
# ---------------------------------------------------------------------------


def _face_triangles(face_shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray] | None:
    face = TopoDS.Face_s(face_shape)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None
    trsf = loc.Transformation()
    nodes = np.empty((tri.NbNodes(), 3))
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i).Transformed(trsf)
        nodes[i - 1] = (p.X(), p.Y(), p.Z())
    tris = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
    for i in range(1, tri.NbTriangles() + 1):
        tris[i - 1] = tri.Triangle(i).Get()
    tris -= 1
    if face.Orientation() == TopAbs_REVERSED:
        tris = tris[:, [0, 2, 1]]
    return nodes, tris


def _oriented_faces(piece: Piece) -> list[tuple[TopoDS_Shape, FaceRole]]:
    """The piece's faces as they appear in its solid (correct orientation), each
    matched to its role. Raises if a face has no role or a role no face."""
    out: list[tuple[TopoDS_Shape, FaceRole]] = []
    used = [False] * len(piece.faces)
    ex = TopExp_Explorer(piece.shape, TopAbs_FACE)
    while ex.More():
        face = ex.Current()
        for i, (cand, role) in enumerate(piece.faces):
            if not used[i] and face.IsSame(cand):
                used[i] = True
                out.append((face, role))
                break
        else:
            raise GeometryError(f"{piece.kind} {piece.key}: face without a role")
        ex.Next()
    if not all(used):
        raise GeometryError(f"{piece.kind} {piece.key}: role without a face")
    return out


def tessellate(
    pieces: list[Piece],
    linear_deflection: float = 0.01,
    angular_deflection: float = 0.08,
) -> TessellatedSurface:
    """Mesh every piece and keep only non-joint faces (the part's outer surface)."""
    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    idx: list[np.ndarray] = []
    skins: list[np.ndarray] = []
    offset = 0
    for piece in pieces:
        mesher = BRepMesh_IncrementalMesh(piece.shape, linear_deflection, False, angular_deflection, False)
        mesher.Perform()
        for face_shape, role in _oriented_faces(piece):
            if role.joint:
                continue
            got = _face_triangles(face_shape)
            if got is None:
                raise GeometryError(f"{piece.kind} {piece.key}: face without triangulation ({role.detail})")
            nodes, ft = got
            verts.append(nodes)
            tris.append(ft + offset)
            n = len(ft)
            kinds.append(np.full(n, role.kind, dtype=np.uint8))
            idx.append(np.full(n, role.index, dtype=np.int16))
            skins.append(np.full(n, role.skin, dtype=np.uint8))
            offset += len(nodes)
    return TessellatedSurface(
        np.vstack(verts),
        np.vstack(tris),
        np.concatenate(kinds),
        np.concatenate(idx),
        np.concatenate(skins),
    )
