"""
Tests for CAD Import Module (STEP/IGES)
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from cad_import import (
    CADFormat,
    BoundingBox,
    CADMesh,
    CADImportResult,
    CADImporter,
    import_cad_file,
    cad_to_point_cloud,
    is_cad_import_available,
    get_supported_formats,
)


class TestCADFormat:
    """Test CADFormat enum"""

    def test_all_formats_defined(self):
        """Test all expected formats exist"""
        formats = [f.value for f in CADFormat]
        assert "step" in formats
        assert "iges" in formats
        assert "brep" in formats
        assert "unknown" in formats


class TestBoundingBox:
    """Test BoundingBox dataclass"""

    def test_bounding_box_creation(self):
        """Test bounding box creation"""
        bbox = BoundingBox(
            min_point=(0, 0, 0),
            max_point=(10, 20, 30)
        )
        assert bbox.min_point == (0, 0, 0)
        assert bbox.max_point == (10, 20, 30)

    def test_bounding_box_size(self):
        """Test bounding box size calculation"""
        bbox = BoundingBox(
            min_point=(0, 0, 0),
            max_point=(10, 20, 30)
        )
        assert bbox.size == (10, 20, 30)

    def test_bounding_box_center(self):
        """Test bounding box center calculation"""
        bbox = BoundingBox(
            min_point=(0, 0, 0),
            max_point=(10, 20, 30)
        )
        assert bbox.center == (5, 10, 15)

    def test_bounding_box_diagonal(self):
        """Test bounding box diagonal calculation"""
        bbox = BoundingBox(
            min_point=(0, 0, 0),
            max_point=(3, 4, 0)
        )
        # Diagonal of 3-4-0 is 5
        assert abs(bbox.diagonal - 5.0) < 0.001


class TestCADMesh:
    """Test CADMesh dataclass"""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple triangular mesh (single triangle)"""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return CADMesh(vertices=vertices, faces=faces)

    @pytest.fixture
    def cube_mesh(self):
        """Create a cube mesh"""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
        ], dtype=np.float64)
        # 12 triangles for 6 faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ], dtype=np.int32)
        return CADMesh(vertices=vertices, faces=faces)

    def test_mesh_num_vertices(self, simple_mesh):
        """Test vertex count"""
        assert simple_mesh.num_vertices == 3

    def test_mesh_num_faces(self, simple_mesh):
        """Test face count"""
        assert simple_mesh.num_faces == 1

    def test_cube_mesh_counts(self, cube_mesh):
        """Test cube mesh counts"""
        assert cube_mesh.num_vertices == 8
        assert cube_mesh.num_faces == 12

    def test_to_point_cloud(self, simple_mesh):
        """Test point cloud generation"""
        points = simple_mesh.to_point_cloud(density=10.0)
        # Triangle area is 0.5, so ~5 points at density 10
        assert len(points) >= 1
        assert points.shape[1] == 3

    def test_to_point_cloud_density(self, cube_mesh):
        """Test point cloud density"""
        points_low = cube_mesh.to_point_cloud(density=1.0)
        points_high = cube_mesh.to_point_cloud(density=10.0)
        # Higher density should produce more points
        assert len(points_high) > len(points_low)


class TestCADImportResult:
    """Test CADImportResult dataclass"""

    def test_success_result(self):
        """Test successful result"""
        result = CADImportResult(
            success=True,
            format=CADFormat.STEP,
            num_solids=1,
            num_faces=12
        )
        assert result.success is True
        assert result.format == CADFormat.STEP
        assert result.error is None

    def test_failure_result(self):
        """Test failure result"""
        result = CADImportResult(
            success=False,
            format=CADFormat.UNKNOWN,
            error="File not found"
        )
        assert result.success is False
        assert "not found" in result.error

    def test_to_dict(self):
        """Test result serialization"""
        bbox = BoundingBox((0, 0, 0), (10, 10, 10))
        mesh = CADMesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=np.array([[0, 1, 2]])
        )
        result = CADImportResult(
            success=True,
            format=CADFormat.STEP,
            mesh=mesh,
            bounding_box=bbox,
            num_solids=1,
            num_faces=6
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["format"] == "step"
        assert data["num_vertices"] == 3
        assert data["num_faces"] == 1
        assert data["bounding_box"]["size"] == (10, 10, 10)


class TestCADImporter:
    """Test CADImporter class"""

    @pytest.fixture
    def importer(self):
        return CADImporter()

    def test_importer_creation(self, importer):
        """Test importer creation"""
        assert importer.linear_deflection == 0.1
        assert importer.angular_deflection == 0.5

    def test_importer_custom_settings(self):
        """Test importer with custom settings"""
        importer = CADImporter(linear_deflection=0.05, angular_deflection=0.3)
        assert importer.linear_deflection == 0.05
        assert importer.angular_deflection == 0.3

    def test_detect_format_step(self, importer):
        """Test STEP format detection"""
        assert importer.detect_format("part.step") == CADFormat.STEP
        assert importer.detect_format("part.stp") == CADFormat.STEP
        assert importer.detect_format("PART.STEP") == CADFormat.STEP

    def test_detect_format_iges(self, importer):
        """Test IGES format detection"""
        assert importer.detect_format("part.iges") == CADFormat.IGES
        assert importer.detect_format("part.igs") == CADFormat.IGES

    def test_detect_format_brep(self, importer):
        """Test BREP format detection"""
        assert importer.detect_format("part.brep") == CADFormat.BREP

    def test_detect_format_unknown(self, importer):
        """Test unknown format detection"""
        assert importer.detect_format("part.stl") == CADFormat.UNKNOWN
        assert importer.detect_format("part.obj") == CADFormat.UNKNOWN

    def test_import_nonexistent_file(self, importer):
        """Test importing nonexistent file"""
        result = importer.import_file("/nonexistent/file.step")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_import_unsupported_format(self, importer):
        """Test importing unsupported format"""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        result = importer.import_file(temp_path)
        assert result.success is False
        assert "unsupported" in result.error.lower()

        Path(temp_path).unlink()


class TestSTLExport:
    """Test STL export functionality"""

    @pytest.fixture
    def mesh(self):
        """Create test mesh"""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1]
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3]
        ], dtype=np.int32)
        return CADMesh(vertices=vertices, faces=faces)

    def test_export_ascii_stl(self, mesh):
        """Test ASCII STL export"""
        importer = CADImporter()
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            temp_path = f.name

        success = importer.export_stl(mesh, temp_path, binary=False)
        assert success is True

        # Verify file content
        with open(temp_path, 'r') as f:
            content = f.read()
        assert "solid ShermanQC" in content
        assert "facet normal" in content
        assert "vertex" in content
        assert "endsolid" in content

        Path(temp_path).unlink()

    def test_export_binary_stl(self, mesh):
        """Test binary STL export"""
        importer = CADImporter()
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            temp_path = f.name

        success = importer.export_stl(mesh, temp_path, binary=True)
        assert success is True

        # Verify file size (header + count + triangles)
        file_size = Path(temp_path).stat().st_size
        # Binary STL: 80 header + 4 count + 50 per triangle
        expected_size = 80 + 4 + (50 * mesh.num_faces)
        assert file_size == expected_size

        Path(temp_path).unlink()


class TestPLYExport:
    """Test PLY export functionality"""

    @pytest.fixture
    def mesh(self):
        """Create test mesh"""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0]
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return CADMesh(vertices=vertices, faces=faces)

    def test_export_ascii_ply(self, mesh):
        """Test ASCII PLY export"""
        importer = CADImporter()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            temp_path = f.name

        success = importer.export_ply(mesh, temp_path, binary=False)
        assert success is True

        # Verify file content
        with open(temp_path, 'r') as f:
            content = f.read()
        assert "ply" in content
        assert "format ascii" in content
        assert "element vertex 3" in content
        assert "element face 1" in content

        Path(temp_path).unlink()

    def test_export_binary_ply(self, mesh):
        """Test binary PLY export"""
        importer = CADImporter()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            temp_path = f.name

        success = importer.export_ply(mesh, temp_path, binary=True)
        assert success is True

        # Verify file exists and has content
        assert Path(temp_path).stat().st_size > 0

        # Check header
        with open(temp_path, 'rb') as f:
            header = f.read(100).decode('ascii', errors='ignore')
        assert "ply" in header
        assert "binary_little_endian" in header

        Path(temp_path).unlink()


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_is_cad_import_available(self):
        """Test OCC availability check"""
        available = is_cad_import_available()
        assert isinstance(available, bool)

    def test_get_supported_formats(self):
        """Test supported formats list"""
        formats = get_supported_formats()
        assert len(formats) >= 3

        step = next((f for f in formats if f["format"] == "STEP"), None)
        assert step is not None
        assert ".step" in step["extensions"]
        assert ".stp" in step["extensions"]

    def test_import_cad_file_nonexistent(self):
        """Test import_cad_file with nonexistent file"""
        result = import_cad_file("/nonexistent/file.step")
        assert result.success is False

    def test_cad_to_point_cloud_nonexistent(self):
        """Test cad_to_point_cloud with nonexistent file"""
        points = cad_to_point_cloud("/nonexistent/file.step")
        assert points is None


def _has_pythonocc():
    """Check if PythonOCC is specifically available"""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_pythonocc(),
    reason="PythonOCC not installed"
)
class TestWithOCC:
    """Tests that require PythonOCC to be installed"""

    def test_occ_import_available(self):
        """Verify OCC import works"""
        from OCC.Core.STEPControl import STEPControl_Reader
        reader = STEPControl_Reader()
        assert reader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
