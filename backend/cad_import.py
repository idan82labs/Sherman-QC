"""
CAD Import Module

Imports STEP and IGES CAD files using PythonOCC or OCP and converts them
to mesh/point cloud formats for QC comparison.

Supported formats:
- STEP (.step, .stp) - ISO 10303 standard
- IGES (.iges, .igs) - Initial Graphics Exchange Specification
- BREP (.brep) - OpenCASCADE native format

Features:
- Import CAD geometry
- Tessellate to mesh
- Extract point cloud from mesh
- Calculate bounding box
- Export to STL/PLY
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

MAX_TESSELLATION_FACE_RECORDS = 4096
MAX_NATIVE_TOPOLOGY_EDGE_USES = 32768
MAX_NATIVE_TOPOLOGY_TREE_NODES = 65536
MAX_NATIVE_TOPOLOGY_DEPTH = 64
MAX_NATIVE_TOPOLOGY_IDENTITY_COMPARISONS = 1000000


def _select_native_cad_binding(*, allow_ocp: bool = False) -> Optional[str]:
    """Select an installed binding before loading it or attempting any file.

    A broken but present OCC package is not absence and must not silently select
    another kernel binding. OCP is opt-in, not an automatic backend promotion.
    Top-level discovery does not import either package.
    """
    if type(allow_ocp) is not bool:
        raise TypeError("allow_ocp must be a bool")
    from importlib.util import find_spec

    if find_spec("OCC") is not None:
        return "pythonocc"
    if allow_ocp and find_spec("OCP") is not None:
        return "ocp"
    return None


def _tessellation_provenance(
    backend: str,
    vertex_count: int,
    triangle_count: int,
    *,
    source_face_count: Optional[int] = None,
    face_records: Optional[List[Dict[str, Any]]] = None,
    reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Describe emitted-mesh bookkeeping, never complete native root topology.

    Face ordinals are local to this import's enumeration. This V1 field
    intentionally excludes the separately captured CADMesh.native_topology
    owner/edge incidence. Its legacy limitations describe this field alone;
    source-byte binding and verified units/frame provenance remain unknown.
    """
    available = face_records is not None
    missing = (sum(record["tessellation_status"] != "TESSELLATED" for record in face_records)
               if available else None)
    mapping_reasons = list(reasons or [])
    if available and missing:
        mapping_reasons.append("SOURCE_FACES_WITHOUT_TRIANGLES")
    return {
        "schema_version": 1,
        "kind": "CAD_TESSELLATION_PROVENANCE_V1",
        "backend": backend,
        "mapping_status": "FACE_OCCURRENCE_RANGES" if available else "UNAVAILABLE",
        "mapping_scope": "EMITTED_TRIANGLES_TO_IMPORT_LOCAL_SOURCE_FACE_OCCURRENCE",
        "source_face_identity_scope": "IMPORT_LOCAL_ENUMERATION_ORDINAL_NOT_PERSISTENT_NATIVE_ID",
        "source_face_count": source_face_count,
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "face_records": face_records if available else [],
        "max_face_records": MAX_TESSELLATION_FACE_RECORDS,
        "source_faces_without_triangles": missing,
        "native_topology_status": "UNKNOWN",
        "native_owner_identity_status": "UNKNOWN",
        "native_edge_identity_status": "UNKNOWN",
        "source_id": None,
        "source_identity_is_verified": False,
        "frame_id": None,
        "frame_is_verified": False,
        "units": None,
        "units_verified": False,
        "reasons": mapping_reasons,
        "limitations": [
            "FACE_OCCURRENCE_BOOKKEEPING_NOT_PERSISTENT_NATIVE_IDENTITY",
            "NATIVE_OWNER_EDGE_AND_ASSEMBLY_OCCURRENCE_IDENTITY_NOT_CAPTURED",
            "NO_NATIVE_TOPOLOGY_COMPLETENESS_OR_MANUFACTURING_EVENT_AUTHORITY",
            "LOCATION_APPLIED_TO_COORDINATES_NOT_RETAINED_AS_VERIFIED_FRAME_PROVENANCE",
            "SOURCE_BYTES_KERNEL_VERSION_AND_UNIT_BASIS_NOT_VERIFIED",
            "NOT_ELIGIBLE_AS_COMPLETE_CAD_ROOT_GROUPING_TOPOLOGY",
        ],
    }


class CADFormat(Enum):
    """Supported CAD file formats"""
    STEP = "step"
    IGES = "iges"
    BREP = "brep"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """3D bounding box"""
    min_point: Tuple[float, float, float]
    max_point: Tuple[float, float, float]

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2]
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_point[0] + self.max_point[0]) / 2,
            (self.min_point[1] + self.max_point[1]) / 2,
            (self.min_point[2] + self.max_point[2]) / 2
        )

    @property
    def diagonal(self) -> float:
        s = self.size
        return np.sqrt(s[0]**2 + s[1]**2 + s[2]**2)


@dataclass
class CADMesh:
    """Mesh representation of CAD geometry"""
    vertices: np.ndarray  # Nx3 array
    faces: np.ndarray  # Mx3 array of vertex indices
    normals: Optional[np.ndarray] = None  # Nx3 or Mx3 array
    # Additive fourth field; existing three-argument constructors are unchanged.
    tessellation_provenance: Optional[Dict[str, Any]] = field(default=None, repr=False)
    # Separate fifth field: import-local kernel incidence, never verified roots.
    native_topology: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def to_point_cloud(self, density: float = 1.0) -> np.ndarray:
        """
        Convert mesh to point cloud by sampling surface.

        Args:
            density: Points per square mm (approximate)

        Returns:
            Nx3 array of points
        """
        points = []

        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            # Calculate triangle area
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

            # Number of points to sample
            n_points = max(1, int(area * density))

            # Sample random points using barycentric coordinates
            for _ in range(n_points):
                r1, r2 = np.random.random(2)
                sqrt_r1 = np.sqrt(r1)
                u = 1 - sqrt_r1
                v = sqrt_r1 * (1 - r2)
                w = sqrt_r1 * r2
                point = u * v0 + v * v1 + w * v2
                points.append(point)

        return np.array(points)


@dataclass
class CADImportResult:
    """Result of CAD import operation"""
    success: bool
    format: CADFormat
    mesh: Optional[CADMesh] = None
    bounding_box: Optional[BoundingBox] = None
    units: str = "mm"

    # Metadata from CAD file
    num_solids: int = 0
    num_shells: int = 0
    num_faces: int = 0
    num_edges: int = 0

    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    # Opt-in, in-process handoff only. These are never serialized and do not
    # verify source bytes, units, frame or manufacturing-event identity.
    native_shape: Optional[Any] = field(default=None, repr=False, compare=False)
    native_binding: Optional[str] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "format": self.format.value,
            "units": self.units,
            "num_vertices": self.mesh.num_vertices if self.mesh else 0,
            "num_faces": self.mesh.num_faces if self.mesh else 0,
            "num_solids": self.num_solids,
            "bounding_box": {
                "min": self.bounding_box.min_point if self.bounding_box else None,
                "max": self.bounding_box.max_point if self.bounding_box else None,
                "size": self.bounding_box.size if self.bounding_box else None,
            } if self.bounding_box else None,
            "error": self.error,
            "warnings": self.warnings
        }


class CADImporter:
    """
    Import STEP/IGES CAD files and convert to mesh.

    Usage:
        importer = CADImporter()
        result = importer.import_file("part.step")
        if result.success:
            points = result.mesh.to_point_cloud(density=1.0)
    """

    def __init__(self, linear_deflection: float = 0.1, angular_deflection: float = 0.5,
                 *, allow_ocp: bool = False):
        """
        Initialize CAD importer.

        Args:
            linear_deflection: Maximum deviation from true surface (mm)
            angular_deflection: Maximum angle between facets (radians)
            allow_ocp: Explicitly permit the OCP binding when PythonOCC is absent.
        """
        if type(allow_ocp) is not bool:
            raise TypeError("allow_ocp must be a bool")
        self.linear_deflection = linear_deflection
        self.angular_deflection = angular_deflection
        self.allow_ocp = allow_ocp

        # Lazy load OCC modules
        self._occ_loaded = False
        self._occ_available = None
        self.native_binding = None

    def _check_occ_available(self) -> bool:
        """Check the selected installed OCCT binding without backend retry."""
        if self._occ_available is not None:
            return self._occ_available

        self.native_binding = _select_native_cad_binding(allow_ocp=self.allow_ocp)
        if self.native_binding is None:
            self._occ_available = False
            return False
        try:
            if self.native_binding == "pythonocc":
                from OCC.Core.STEPControl import STEPControl_Reader
            else:
                from OCP.STEPControl import STEPControl_Reader
            self._occ_available = True
        except (ImportError, OSError):
            self._occ_available = False
            logger.warning("Selected native CAD binding is unavailable: %s", self.native_binding)

        return self._occ_available

    def _load_occ(self):
        """Lazy load OCC modules"""
        if self._occ_loaded:
            return

        if not self._check_occ_available():
            raise ImportError("Selected native CAD binding is unavailable")

        if self.native_binding == "ocp":
            self._load_ocp()
            self._occ_loaded = True
            return

        # Import OCC modules
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.BRepTools import breptools
        from OCC.Core.BRep import BRep_Builder, BRep_Tool
        from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.Poly import Poly_Triangulation
        from OCC.Core.IFSelect import IFSelect_RetDone

        self._STEPControl_Reader = STEPControl_Reader
        self._IGESControl_Reader = IGESControl_Reader
        self._breptools = breptools
        self._BRep_Builder = BRep_Builder
        self._BRep_Tool = BRep_Tool
        self._TopoDS_Shape = TopoDS_Shape
        self._TopoDS_Compound = TopoDS_Compound
        self._TopExp_Explorer = TopExp_Explorer
        self._TopAbs_SOLID = TopAbs_SOLID
        self._TopAbs_SHELL = TopAbs_SHELL
        self._TopAbs_FACE = TopAbs_FACE
        self._TopAbs_EDGE = TopAbs_EDGE
        self._BRepMesh_IncrementalMesh = BRepMesh_IncrementalMesh
        self._Bnd_Box = Bnd_Box
        self._brepbndlib = brepbndlib
        self._TopLoc_Location = TopLoc_Location
        self._Poly_Triangulation = Poly_Triangulation
        self._IFSelect_RetDone = IFSelect_RetDone

        # Optional provenance APIs must not disable the existing geometry path.
        self._native_topology_api_available = False
        try:
            from OCC.Core.TopoDS import TopoDS_Iterator, topods
            from OCC.Core.TopAbs import (
                TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL, TopAbs_EXTERNAL,
            )
            self._TopoDS_Iterator = TopoDS_Iterator
            self._native_edge_cast = topods.Edge
            self._native_orientations = {
                TopAbs_FORWARD: "FORWARD", TopAbs_REVERSED: "REVERSED",
                TopAbs_INTERNAL: "INTERNAL", TopAbs_EXTERNAL: "EXTERNAL",
            }
            self._native_topology_api_available = True
        except (ImportError, AttributeError):
            pass

        self._occ_loaded = True

    def _load_ocp(self):
        """Adapt the installed OCP OCCT binding to the existing importer seam.

        OCP exposes static C++ methods with an _s suffix and requires explicit
        shape casts. Oriented(current orientation) returns a value copy retaining
        TShape/location, without assuming an OCP shape-copy constructor exists.
        No mesh policy, native-incidence schema or file retry is introduced.
        """
        from types import SimpleNamespace
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IGESControl import IGESControl_Reader
        from OCP.BRepTools import BRepTools
        from OCP.BRep import BRep_Builder, BRep_Tool
        from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Compound, TopoDS_Iterator
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import (
            TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE,
            TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL, TopAbs_EXTERNAL,
        )
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.TopLoc import TopLoc_Location
        from OCP.Poly import Poly_Triangulation
        from OCP.IFSelect import IFSelect_RetDone

        self._STEPControl_Reader = STEPControl_Reader
        self._IGESControl_Reader = IGESControl_Reader
        self._breptools = SimpleNamespace(Read=BRepTools.Read_s)
        self._BRep_Builder = BRep_Builder
        self._BRep_Tool = SimpleNamespace(
            Triangulation=lambda face, location: BRep_Tool.Triangulation_s(TopoDS.Face_s(face), location),
            Degenerated=BRep_Tool.Degenerated_s,
        )
        self._TopoDS_Shape = lambda shape=None: (
            TopoDS_Shape() if shape is None else shape.Oriented(shape.Orientation()))
        self._TopoDS_Compound = TopoDS_Compound
        self._TopExp_Explorer = TopExp_Explorer
        self._TopAbs_SOLID, self._TopAbs_SHELL = TopAbs_SOLID, TopAbs_SHELL
        self._TopAbs_FACE, self._TopAbs_EDGE = TopAbs_FACE, TopAbs_EDGE
        self._BRepMesh_IncrementalMesh = BRepMesh_IncrementalMesh
        self._Bnd_Box = Bnd_Box
        self._brepbndlib = SimpleNamespace(Add=BRepBndLib.Add_s)
        self._TopLoc_Location = TopLoc_Location
        self._Poly_Triangulation = Poly_Triangulation
        self._IFSelect_RetDone = IFSelect_RetDone
        self._TopoDS_Iterator = TopoDS_Iterator
        self._native_edge_cast = TopoDS.Edge_s
        self._native_orientations = {
            TopAbs_FORWARD: "FORWARD", TopAbs_REVERSED: "REVERSED",
            TopAbs_INTERNAL: "INTERNAL", TopAbs_EXTERNAL: "EXTERNAL",
        }
        self._native_topology_api_available = True

    def detect_format(self, file_path: str) -> CADFormat:
        """Detect CAD file format from extension"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in [".step", ".stp"]:
            return CADFormat.STEP
        elif ext in [".iges", ".igs"]:
            return CADFormat.IGES
        elif ext == ".brep":
            return CADFormat.BREP
        else:
            return CADFormat.UNKNOWN

    def import_file(self, file_path: str, *, retain_native_shape: bool = False) -> CADImportResult:
        """
        Import CAD file and convert to mesh.

        Args:
            file_path: Path to STEP/IGES/BREP file
            retain_native_shape: Explicitly retain this successful read's exact
                native shape and binding for a synchronous offline consumer.
                The default result and serialized contract remain unchanged.

        Returns:
            CADImportResult with mesh and metadata
        """
        if type(retain_native_shape) is not bool:
            raise TypeError("retain_native_shape must be a bool")
        path = Path(file_path)
        if not path.exists():
            return CADImportResult(
                success=False,
                format=CADFormat.UNKNOWN,
                error=f"File not found: {file_path}"
            )

        format = self.detect_format(file_path)
        if format == CADFormat.UNKNOWN:
            return CADImportResult(
                success=False,
                format=format,
                error=f"Unsupported file format: {path.suffix}"
            )

        try:
            self._load_occ()
        except (ImportError, OSError, AttributeError) as e:
            return CADImportResult(
                success=False,
                format=format,
                error=str(e)
            )

        # Read the shape
        try:
            if format == CADFormat.STEP:
                shape = self._read_step(file_path)
            elif format == CADFormat.IGES:
                shape = self._read_iges(file_path)
            else:  # BREP
                shape = self._read_brep(file_path)

            if shape is None:
                return CADImportResult(
                    success=False,
                    format=format,
                    error="Failed to read CAD geometry"
                )

            # Analyze topology
            topology = self._analyze_topology(shape)

            # Calculate bounding box
            bbox = self._get_bounding_box(shape)

            # Tessellate to mesh
            mesh = self._tessellate(shape)

            return CADImportResult(
                success=True,
                format=format,
                mesh=mesh,
                bounding_box=bbox,
                num_solids=topology["solids"],
                num_shells=topology["shells"],
                num_faces=topology["faces"],
                num_edges=topology["edges"],
                native_shape=shape if retain_native_shape else None,
                native_binding=self.native_binding if retain_native_shape else None,
            )

        except Exception as e:
            logger.error(f"CAD import error: {e}")
            return CADImportResult(
                success=False,
                format=format,
                error=str(e)
            )

    def _read_step(self, file_path: str):
        """Read STEP file"""
        reader = self._STEPControl_Reader()
        status = reader.ReadFile(file_path)

        if status != self._IFSelect_RetDone:
            return None

        reader.TransferRoots()
        return reader.OneShape()

    def _read_iges(self, file_path: str):
        """Read IGES file"""
        reader = self._IGESControl_Reader()
        status = reader.ReadFile(file_path)

        if status != self._IFSelect_RetDone:
            return None

        reader.TransferRoots()
        return reader.OneShape()

    def _read_brep(self, file_path: str):
        """Read BREP file"""
        shape = self._TopoDS_Shape()
        builder = self._BRep_Builder()
        success = self._breptools.Read(shape, file_path, builder)

        if not success:
            return None

        return shape

    def _analyze_topology(self, shape) -> Dict[str, int]:
        """Count topological entities"""
        counts = {"solids": 0, "shells": 0, "faces": 0, "edges": 0}

        for topo_type, key in [
            (self._TopAbs_SOLID, "solids"),
            (self._TopAbs_SHELL, "shells"),
            (self._TopAbs_FACE, "faces"),
            (self._TopAbs_EDGE, "edges")
        ]:
            explorer = self._TopExp_Explorer(shape, topo_type)
            while explorer.More():
                counts[key] += 1
                explorer.Next()

        return counts

    def _get_bounding_box(self, shape) -> BoundingBox:
        """Calculate bounding box of shape"""
        bbox = self._Bnd_Box()
        self._brepbndlib.Add(shape, bbox)

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        return BoundingBox(
            min_point=(xmin, ymin, zmin),
            max_point=(xmax, ymax, zmax)
        )

    def _tessellate(self, shape) -> CADMesh:
        """Convert shape to triangular mesh"""
        # Perform tessellation
        mesh = self._BRepMesh_IncrementalMesh(
            shape,
            self.linear_deflection,
            False,  # relative
            self.angular_deflection,
            True   # parallel
        )
        mesh.Perform()

        # Extract triangles from faces
        all_vertices = []
        all_faces = []
        all_normals = []
        vertex_offset = 0
        source_face_count = 0
        face_records = []
        mapping_failures = set()
        native_faces = []
        native_capture_failure = None

        explorer = self._TopExp_Explorer(shape, self._TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            location = self._TopLoc_Location()
            face_index = source_face_count
            source_face_count += 1
            if native_capture_failure is None:
                if source_face_count > MAX_TESSELLATION_FACE_RECORDS:
                    native_capture_failure = "SOURCE_FACE_RECORD_BUDGET_EXCEEDED"
                    native_faces.clear()
                else:
                    try:
                        # Retain the exact explorer occurrence before Next().
                        native_faces.append(self._TopoDS_Shape(face))
                    except Exception:
                        native_capture_failure = "NATIVE_FACE_HANDLE_CAPTURE_UNAVAILABLE"
                        native_faces.clear()
            vertex_start, triangle_start = len(all_vertices), len(all_faces)
            if source_face_count > MAX_TESSELLATION_FACE_RECORDS:
                mapping_failures.add("SOURCE_FACE_RECORD_BUDGET_EXCEEDED")
                face_records.clear()

            triangulation = self._BRep_Tool.Triangulation(face, location)
            tessellation_status = "NO_TRIANGULATION"

            if triangulation is not None:
                # Get vertices
                nb_nodes = triangulation.NbNodes()
                vertices = []
                for i in range(1, nb_nodes + 1):
                    pnt = triangulation.Node(i)
                    # Apply transformation
                    if not location.IsIdentity():
                        pnt = pnt.Transformed(location.Transformation())
                    vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

                all_vertices.extend(vertices)

                # Get triangles
                nb_triangles = triangulation.NbTriangles()
                # Preserve existing geometry behavior, including its exception
                # handling, but never bless malformed native bookkeeping.
                if any(isinstance(count, (bool, np.bool_))
                       or not isinstance(count, (int, np.integer)) or count < 0
                       for count in (nb_nodes, nb_triangles)):
                    mapping_failures.add("INVALID_NATIVE_FACE_TESSELLATION_COUNTS")
                    face_records.clear()
                for i in range(1, nb_triangles + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    # A globally indexable flattened triangle can still escape
                    # its source face's node block (for example local index 0).
                    # Reject only the sidecar claim; preserve legacy geometry
                    # construction and its exceptions. For valid native counts,
                    # len(vertices) is exactly nb_nodes.
                    if any(isinstance(index, (bool, np.bool_))
                           or not isinstance(index, (int, np.integer))
                           or index < 1 or index > len(vertices)
                           for index in (n1, n2, n3)):
                        mapping_failures.add("INVALID_NATIVE_FACE_TRIANGLE_INDICES")
                        face_records.clear()
                    # Adjust indices for global array
                    all_faces.append([
                        n1 - 1 + vertex_offset,
                        n2 - 1 + vertex_offset,
                        n3 - 1 + vertex_offset
                    ])

                vertex_offset += nb_nodes
                tessellation_status = "TESSELLATED" if nb_triangles > 0 else "EMPTY_TRIANGULATION"

                if (len(all_vertices)-vertex_start != nb_nodes
                        or len(all_faces)-triangle_start != nb_triangles
                        or vertex_offset != len(all_vertices)):
                    mapping_failures.add("INCONSISTENT_NATIVE_FACE_TESSELLATION_RANGES")
                    face_records.clear()

            if not mapping_failures:
                face_records.append({
                    "source_face_index": face_index,
                    "vertex_range": [vertex_start, len(all_vertices)],
                    "triangle_range": [triangle_start, len(all_faces)],
                    "tessellation_status": tessellation_status,
                    "owner_id": None,
                    "edge_ids": None,
                })

            explorer.Next()

        vertices_array = np.array(all_vertices, dtype=np.float64)
        faces_array = np.array(all_faces, dtype=np.int32)

        # Calculate normals if we have data
        normals = None
        if len(all_faces) > 0:
            normals = self._calculate_face_normals(vertices_array, faces_array)

        native_topology = {
            "schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1",
            "status": "UNAVAILABLE", "face_records": [], "owner_records": [],
            "reasons": [native_capture_failure or "TESSELLATION_MAPPING_UNAVAILABLE"],
        }
        if not mapping_failures and native_capture_failure is None:
            try:
                native_topology = self._capture_native_topology(shape, native_faces)
            except Exception:
                # Incidence is additive. A missing API or unexpected native
                # failure never changes mesh geometry or its V1 range sidecar.
                native_topology["reasons"] = ["NATIVE_TOPOLOGY_CAPTURE_FAILED"]

        return CADMesh(
            vertices=vertices_array,
            faces=faces_array,
            normals=normals,
            tessellation_provenance=_tessellation_provenance(
                getattr(self, "native_binding", None) or "pythonocc", len(vertices_array), len(faces_array),
                source_face_count=source_face_count,
                face_records=None if mapping_failures else face_records,
                reasons=sorted(mapping_failures),
            ),
            native_topology=native_topology,
        )

    def _capture_native_topology(self, shape, tessellation_faces) -> Dict[str, Any]:
        """Capture bounded native occurrence incidence, not physical topology.

        TopoDS_Iterator cumulative orientation/location preserves hierarchy
        occurrences. IsSame compares TShape plus location, not coordinates; its
        orientation independence permits oppositely oriented uses of one edge.
        Every face must match the exact tessellation enumeration uniquely.
        Repeated coincident native faces, failures and caps clear the whole map.
        This neither verifies source bytes/units/frame nor validates a manifold,
        persistent assembly identity or a manufacturing-event interpretation.
        """
        unavailable = {
            "schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1",
            "status": "UNAVAILABLE", "face_records": [], "owner_records": [],
            "reasons": [],
        }
        try:
            if not getattr(self, "_native_topology_api_available", False):
                raise ValueError("NATIVE_TOPOLOGY_APIS_UNAVAILABLE")
            if len(tessellation_faces) > MAX_TESSELLATION_FACE_RECORDS:
                raise ValueError("SOURCE_FACE_RECORD_BUDGET_EXCEEDED")
            if not tessellation_faces:
                raise ValueError("NO_NATIVE_FACES")
            faces, owners, owner_keys, edges_by_owner = [], [], {}, {}
            tree_nodes = edge_uses = identity_comparisons = 0

            def same(left, right):
                nonlocal identity_comparisons
                identity_comparisons += 1
                if identity_comparisons > MAX_NATIVE_TOPOLOGY_IDENTITY_COMPARISONS:
                    raise ValueError("NATIVE_IDENTITY_COMPARISON_BUDGET_EXCEEDED")
                result = left.IsSame(right)
                if type(result) is not bool:
                    raise ValueError("INVALID_NATIVE_IDENTITY_RESULT")
                return result

            def owner_id(kind, path):
                key = (kind, path)
                if key not in owner_keys:
                    value = "owner:" + str(len(owners))
                    owner_keys[key] = value
                    owners.append({"owner_id": value, "kind": kind,
                                   "occurrence_path": list(path)})
                    edges_by_owner[value] = []
                return owner_keys[key]

            def walk(current, path, solid=None, shell=None, active_face=None):
                nonlocal tree_nodes, edge_uses
                tree_nodes += 1
                if tree_nodes > MAX_NATIVE_TOPOLOGY_TREE_NODES:
                    raise ValueError("NATIVE_TREE_NODE_BUDGET_EXCEEDED")
                if len(path) > MAX_NATIVE_TOPOLOGY_DEPTH:
                    raise ValueError("NATIVE_TREE_DEPTH_BUDGET_EXCEEDED")
                if current.IsNull():
                    raise ValueError("NULL_NATIVE_SHAPE")
                kind = current.ShapeType()
                if kind == self._TopAbs_SOLID:
                    if active_face is not None:
                        raise ValueError("INVALID_NATIVE_OCCURRENCE_HIERARCHY")
                    solid = ("SOLID", path)
                elif kind == self._TopAbs_SHELL:
                    if active_face is not None:
                        raise ValueError("INVALID_NATIVE_OCCURRENCE_HIERARCHY")
                    shell = ("SHELL", path)
                elif kind == self._TopAbs_FACE:
                    if active_face is not None:
                        raise ValueError("INVALID_NATIVE_OCCURRENCE_HIERARCHY")
                    if len(faces) >= MAX_TESSELLATION_FACE_RECORDS:
                        raise ValueError("SOURCE_FACE_RECORD_BUDGET_EXCEEDED")
                    if any(same(current, item[0]) for item in faces):
                        raise ValueError("AMBIGUOUS_REPEATED_NATIVE_FACE_OCCURRENCE")
                    active_face = {
                        "owner_id": owner_id(*(solid or shell or ("FACE", path))),
                        "edge_uses": [],
                    }
                    faces.append((self._TopoDS_Shape(current), active_face))
                elif kind == self._TopAbs_EDGE:
                    if active_face is not None:
                        edge_uses += 1
                        if edge_uses > MAX_NATIVE_TOPOLOGY_EDGE_USES:
                            raise ValueError("NATIVE_EDGE_USE_BUDGET_EXCEEDED")
                        edge = self._native_edge_cast(current)
                        orientation = self._native_orientations.get(edge.Orientation())
                        if orientation is None:
                            raise ValueError("INVALID_NATIVE_EDGE_ORIENTATION")
                        degenerate = self._BRep_Tool.Degenerated(edge)
                        if type(degenerate) is not bool:
                            raise ValueError("INVALID_NATIVE_EDGE_DEGENERATE_FLAG")
                        owner = active_face["owner_id"]
                        known = edges_by_owner[owner]
                        matches = [index for index, (prior, _) in enumerate(known)
                                   if same(edge, prior)]
                        if len(matches) > 1:
                            raise ValueError("AMBIGUOUS_NATIVE_EDGE_IDENTITY")
                        if matches:
                            index = matches[0]
                            if known[index][1] != degenerate:
                                raise ValueError("INCONSISTENT_NATIVE_EDGE_DEGENERATE_FLAG")
                        else:
                            index = len(known)
                            known.append((self._TopoDS_Shape(edge), degenerate))
                        active_face["edge_uses"].append({
                            "edge_id": owner + ":edge:" + str(index),
                            "orientation": orientation, "degenerate": degenerate,
                        })
                    # Edge vertices have no further face/owner incidence.
                    return

                iterator = self._TopoDS_Iterator(current, True, True)
                child_index = 0
                while iterator.More():
                    child = self._TopoDS_Shape(iterator.Value())
                    walk(child, path + (child_index,), solid, shell, active_face)
                    child_index += 1
                    iterator.Next()

            walk(self._TopoDS_Shape(shape), ())
            if len(faces) != len(tessellation_faces):
                raise ValueError("NATIVE_TESSELLATION_FACE_COUNT_MISMATCH")
            ordered, matched = [], set()
            for index, tessellation_face in enumerate(tessellation_faces):
                matches = [position for position, (face, _) in enumerate(faces)
                           if same(tessellation_face, face)]
                if len(matches) != 1 or matches[0] in matched:
                    raise ValueError("NATIVE_TESSELLATION_FACE_IDENTITY_MISMATCH")
                if tessellation_face.Orientation() != faces[matches[0]][0].Orientation():
                    raise ValueError("NATIVE_TESSELLATION_FACE_ORIENTATION_MISMATCH")
                matched.add(matches[0])
                record = faces[matches[0]][1]
                ordered.append({"source_face_index": index, "face_id": "face:" + str(index),
                                "owner_id": record["owner_id"],
                                "edge_uses": record["edge_uses"]})
            return {
                "schema_version": 1, "kind": "CAD_NATIVE_TOPOLOGY_V1",
                "status": "CAPTURED_IMPORT_LOCAL_INCIDENCE",
                "face_records": ordered, "owner_records": owners, "reasons": [],
            }
        except ValueError as exc:
            # Only our bounded reason vocabulary may cross the metadata seam.
            reason = str(exc)
            allowed = {
                "NATIVE_TOPOLOGY_APIS_UNAVAILABLE", "SOURCE_FACE_RECORD_BUDGET_EXCEEDED",
                "NO_NATIVE_FACES", "NATIVE_IDENTITY_COMPARISON_BUDGET_EXCEEDED",
                "INVALID_NATIVE_IDENTITY_RESULT", "NATIVE_TREE_NODE_BUDGET_EXCEEDED",
                "NATIVE_TREE_DEPTH_BUDGET_EXCEEDED", "NULL_NATIVE_SHAPE",
                "INVALID_NATIVE_OCCURRENCE_HIERARCHY", "AMBIGUOUS_REPEATED_NATIVE_FACE_OCCURRENCE",
                "NATIVE_EDGE_USE_BUDGET_EXCEEDED", "INVALID_NATIVE_EDGE_ORIENTATION",
                "INVALID_NATIVE_EDGE_DEGENERATE_FLAG", "AMBIGUOUS_NATIVE_EDGE_IDENTITY",
                "INCONSISTENT_NATIVE_EDGE_DEGENERATE_FLAG", "NATIVE_TESSELLATION_FACE_COUNT_MISMATCH",
                "NATIVE_TESSELLATION_FACE_IDENTITY_MISMATCH",
                "NATIVE_TESSELLATION_FACE_ORIENTATION_MISMATCH",
            }
            unavailable["reasons"] = [reason if reason in allowed else "NATIVE_TOPOLOGY_CAPTURE_FAILED"]
        except Exception:
            unavailable["reasons"] = ["NATIVE_TOPOLOGY_CAPTURE_FAILED"]
        return unavailable

    def _calculate_face_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate face normals"""
        normals = []
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            normals.append(normal)

        return np.array(normals, dtype=np.float64)

    def export_stl(self, mesh: CADMesh, file_path: str, binary: bool = True) -> bool:
        """
        Export mesh to STL file.

        Args:
            mesh: CADMesh to export
            file_path: Output STL file path
            binary: Use binary STL format (default True)

        Returns:
            Success status
        """
        try:
            if binary:
                return self._write_binary_stl(mesh, file_path)
            else:
                return self._write_ascii_stl(mesh, file_path)
        except Exception as e:
            logger.error(f"STL export error: {e}")
            return False

    def _write_binary_stl(self, mesh: CADMesh, file_path: str) -> bool:
        """Write binary STL file"""
        import struct

        with open(file_path, 'wb') as f:
            # Header (80 bytes)
            header = b'Sherman QC Binary STL' + b'\0' * (80 - 21)
            f.write(header)

            # Number of triangles
            f.write(struct.pack('<I', len(mesh.faces)))

            # Write triangles
            for i, face in enumerate(mesh.faces):
                v0 = mesh.vertices[face[0]]
                v1 = mesh.vertices[face[1]]
                v2 = mesh.vertices[face[2]]

                # Normal
                if mesh.normals is not None and i < len(mesh.normals):
                    normal = mesh.normals[i]
                else:
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm

                # Write normal and vertices
                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))

                # Attribute byte count
                f.write(struct.pack('<H', 0))

        return True

    def _write_ascii_stl(self, mesh: CADMesh, file_path: str) -> bool:
        """Write ASCII STL file"""
        with open(file_path, 'w') as f:
            f.write("solid ShermanQC\n")

            for i, face in enumerate(mesh.faces):
                v0 = mesh.vertices[face[0]]
                v1 = mesh.vertices[face[1]]
                v2 = mesh.vertices[face[2]]

                # Normal
                if mesh.normals is not None and i < len(mesh.normals):
                    normal = mesh.normals[i]
                else:
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm

                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            f.write("endsolid ShermanQC\n")

        return True

    def export_ply(self, mesh: CADMesh, file_path: str, binary: bool = True) -> bool:
        """
        Export mesh to PLY file.

        Args:
            mesh: CADMesh to export
            file_path: Output PLY file path
            binary: Use binary PLY format (default True)

        Returns:
            Success status
        """
        try:
            with open(file_path, 'wb' if binary else 'w') as f:
                # Header
                header = f"""ply
format {'binary_little_endian' if binary else 'ascii'} 1.0
element vertex {mesh.num_vertices}
property float x
property float y
property float z
element face {mesh.num_faces}
property list uchar int vertex_indices
end_header
"""
                if binary:
                    f.write(header.encode('ascii'))

                    # Vertices
                    for v in mesh.vertices:
                        f.write(np.array(v, dtype='<f4').tobytes())

                    # Faces
                    for face in mesh.faces:
                        f.write(np.array([3], dtype='<u1').tobytes())
                        f.write(np.array(face, dtype='<i4').tobytes())
                else:
                    f.write(header)

                    # Vertices
                    for v in mesh.vertices:
                        f.write(f"{v[0]} {v[1]} {v[2]}\n")

                    # Faces
                    for face in mesh.faces:
                        f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

            return True

        except Exception as e:
            logger.error(f"PLY export error: {e}")
            return False


# Convenience functions
def import_cad_file(file_path: str, deflection: float = 0.1, *, allow_ocp: bool = False) -> CADImportResult:
    """
    Import CAD file with default settings.

    Defaults to PythonOCC then gmsh. Only explicit allow_ocp=True admits the OCP
    binding between them. A selected binding/file failure is not retried.

    Args:
        file_path: Path to STEP/IGES file
        deflection: Tessellation accuracy in mm
        allow_ocp: Explicitly permit OCP; no production equivalence is implied.

    Returns:
        CADImportResult with mesh and metadata
    """
    importer = CADImporter(linear_deflection=deflection, allow_ocp=allow_ocp)
    if importer._check_occ_available():
        return importer.import_file(file_path)
    if importer.native_binding is not None:
        return CADImportResult(success=False, format=CADFormat.UNKNOWN,
                               error="Selected native CAD binding is unavailable: " + importer.native_binding)

    # Fall back to gmsh
    return _import_with_gmsh(file_path, deflection)


def cad_to_point_cloud(file_path: str, density: float = 1.0) -> Optional[np.ndarray]:
    """
    Import CAD file and convert directly to point cloud.

    Args:
        file_path: Path to STEP/IGES file
        density: Points per square mm

    Returns:
        Nx3 array of points, or None on failure
    """
    result = import_cad_file(file_path)
    if not result.success or result.mesh is None:
        return None
    return result.mesh.to_point_cloud(density)


def is_cad_import_available(*, allow_ocp: bool = False) -> bool:
    """Check the same default/explicitly opted-in binding policy as import."""
    importer = CADImporter(allow_ocp=allow_ocp)
    if importer._check_occ_available():
        return True
    if importer.native_binding is not None:
        return False

    # Check gmsh as fallback
    try:
        import gmsh
        return True
    except ImportError:
        pass

    return False


def _import_with_gmsh(file_path: str, deflection: float = 0.1) -> CADImportResult:
    """
    Import CAD using gmsh (fallback only when no permitted native binding exists).

    Args:
        file_path: Path to STEP/IGES file
        deflection: Mesh size factor (smaller = finer mesh)

    Returns:
        CADImportResult with mesh data
    """
    try:
        import gmsh
    except ImportError:
        return CADImportResult(
            success=False,
            format=CADFormat.UNKNOWN,
            error="No permitted native CAD binding or gmsh is available"
        )

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in ['.step', '.stp']:
        format = CADFormat.STEP
    elif ext in ['.iges', '.igs']:
        format = CADFormat.IGES
    else:
        return CADImportResult(
            success=False,
            format=CADFormat.UNKNOWN,
            error=f"Unsupported format: {ext}"
        )

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

        # Import the CAD file
        gmsh.model.add("import")

        try:
            gmsh.model.occ.importShapes(str(file_path))
            gmsh.model.occ.synchronize()
        except Exception as e:
            gmsh.finalize()
            return CADImportResult(
                success=False,
                format=format,
                error=f"Failed to import CAD file: {e}"
            )

        # Get bounding box
        try:
            bbox = gmsh.model.getBoundingBox(-1, -1)
            bounding_box = BoundingBox(
                min_point=(bbox[0], bbox[1], bbox[2]),
                max_point=(bbox[3], bbox[4], bbox[5])
            )
            # Use bounding box diagonal for mesh sizing
            diag = bounding_box.diagonal
            mesh_size = diag * deflection * 0.1  # Scale deflection to mesh size
        except:
            bounding_box = None
            mesh_size = deflection

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)

        # Generate 2D mesh (surface mesh)
        gmsh.model.mesh.generate(2)

        # Extract mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Build vertex array
        vertices = np.array(node_coords).reshape(-1, 3)

        # Build node tag to index mapping
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

        # Get triangles
        faces = []
        element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements(2)  # dim=2 for surfaces

        for elem_type, elem_tags, elem_nodes in zip(element_types, element_tags, node_tags_per_element):
            # Get element properties
            elem_name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)

            if num_nodes == 3:  # Triangles
                elem_nodes = np.array(elem_nodes).reshape(-1, 3)
                for tri_nodes in elem_nodes:
                    try:
                        face = [tag_to_idx[int(n)] for n in tri_nodes]
                        faces.append(face)
                    except KeyError:
                        continue

        gmsh.finalize()

        if len(faces) == 0:
            return CADImportResult(
                success=False,
                format=format,
                error="No triangles generated from mesh"
            )

        faces_array = np.array(faces, dtype=np.int32)

        # Calculate normals
        normals = []
        for face in faces_array:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            normals.append(normal)

        mesh = CADMesh(
            vertices=vertices,
            faces=faces_array,
            normals=np.array(normals, dtype=np.float64),
            tessellation_provenance=_tessellation_provenance(
                "gmsh", len(vertices), len(faces_array),
                reasons=["GMSH_NATIVE_FACE_ENTITY_MAPPING_UNAVAILABLE"],
            ),
        )

        return CADImportResult(
            success=True,
            format=format,
            mesh=mesh,
            bounding_box=bounding_box,
            num_faces=len(faces_array)
        )

    except Exception as e:
        try:
            gmsh.finalize()
        except:
            pass
        logger.error(f"gmsh import error: {e}")
        return CADImportResult(
            success=False,
            format=format,
            error=str(e)
        )


def get_supported_formats() -> List[Dict[str, str]]:
    """Get list of supported CAD formats"""
    return [
        {
            "format": "STEP",
            "extensions": [".step", ".stp"],
            "description": "ISO 10303 STEP file",
            "available": is_cad_import_available()
        },
        {
            "format": "IGES",
            "extensions": [".iges", ".igs"],
            "description": "Initial Graphics Exchange Specification",
            "available": is_cad_import_available()
        },
        {
            "format": "BREP",
            "extensions": [".brep"],
            "description": "OpenCASCADE Boundary Representation",
            "available": is_cad_import_available()
        }
    ]
