"""
CAD Import Module

Imports STEP and IGES CAD files using PythonOCC and converts them
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

    def __init__(self, linear_deflection: float = 0.1, angular_deflection: float = 0.5):
        """
        Initialize CAD importer.

        Args:
            linear_deflection: Maximum deviation from true surface (mm)
            angular_deflection: Maximum angle between facets (radians)
        """
        self.linear_deflection = linear_deflection
        self.angular_deflection = angular_deflection

        # Lazy load OCC modules
        self._occ_loaded = False
        self._occ_available = None

    def _check_occ_available(self) -> bool:
        """Check if PythonOCC is available"""
        if self._occ_available is not None:
            return self._occ_available

        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            self._occ_available = True
        except ImportError:
            self._occ_available = False
            logger.warning("PythonOCC not available. CAD import disabled.")

        return self._occ_available

    def _load_occ(self):
        """Lazy load OCC modules"""
        if self._occ_loaded:
            return

        if not self._check_occ_available():
            raise ImportError("PythonOCC is not installed. Install with: pip install pythonocc-core")

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

        self._occ_loaded = True

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

    def import_file(self, file_path: str) -> CADImportResult:
        """
        Import CAD file and convert to mesh.

        Args:
            file_path: Path to STEP/IGES/BREP file

        Returns:
            CADImportResult with mesh and metadata
        """
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
        except ImportError as e:
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
                num_edges=topology["edges"]
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

        explorer = self._TopExp_Explorer(shape, self._TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            location = self._TopLoc_Location()

            triangulation = self._BRep_Tool.Triangulation(face, location)

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
                for i in range(1, nb_triangles + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    # Adjust indices for global array
                    all_faces.append([
                        n1 - 1 + vertex_offset,
                        n2 - 1 + vertex_offset,
                        n3 - 1 + vertex_offset
                    ])

                vertex_offset += nb_nodes

            explorer.Next()

        vertices_array = np.array(all_vertices, dtype=np.float64)
        faces_array = np.array(all_faces, dtype=np.int32)

        # Calculate normals if we have data
        normals = None
        if len(all_faces) > 0:
            normals = self._calculate_face_normals(vertices_array, faces_array)

        return CADMesh(
            vertices=vertices_array,
            faces=faces_array,
            normals=normals
        )

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
def import_cad_file(file_path: str, deflection: float = 0.1) -> CADImportResult:
    """
    Import CAD file with default settings.

    Args:
        file_path: Path to STEP/IGES file
        deflection: Tessellation accuracy in mm

    Returns:
        CADImportResult with mesh and metadata
    """
    importer = CADImporter(linear_deflection=deflection)
    return importer.import_file(file_path)


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


def is_cad_import_available() -> bool:
    """Check if PythonOCC is available for CAD import"""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        return True
    except ImportError:
        return False


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
