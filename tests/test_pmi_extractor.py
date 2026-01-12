"""
Tests for PMI (Product Manufacturing Information) Extractor
"""

import pytest
import json
from pathlib import Path
import sys
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pmi_extractor import (
    DimensionType,
    GDTSymbol,
    MaterialCondition,
    DatumReference,
    Dimension,
    GDTCallout,
    DatumFeature,
    Note,
    PMIData,
    PMIExtractor,
    extract_pmi,
    get_inspection_requirements,
    parse_gdt_text,
)


class TestDimensionType:
    """Test DimensionType enum"""

    def test_all_types_defined(self):
        """Test all expected types exist"""
        types = [t.value for t in DimensionType]
        assert "linear" in types
        assert "angular" in types
        assert "radial" in types
        assert "diameter" in types


class TestGDTSymbol:
    """Test GDTSymbol enum"""

    def test_form_tolerances(self):
        """Test form tolerance symbols"""
        assert GDTSymbol.FLATNESS.value == "flatness"
        assert GDTSymbol.CYLINDRICITY.value == "cylindricity"
        assert GDTSymbol.CIRCULARITY.value == "circularity"
        assert GDTSymbol.STRAIGHTNESS.value == "straightness"

    def test_orientation_tolerances(self):
        """Test orientation tolerance symbols"""
        assert GDTSymbol.PARALLELISM.value == "parallelism"
        assert GDTSymbol.PERPENDICULARITY.value == "perpendicularity"
        assert GDTSymbol.ANGULARITY.value == "angularity"

    def test_location_tolerances(self):
        """Test location tolerance symbols"""
        assert GDTSymbol.POSITION.value == "position"
        assert GDTSymbol.CONCENTRICITY.value == "concentricity"
        assert GDTSymbol.SYMMETRY.value == "symmetry"


class TestMaterialCondition:
    """Test MaterialCondition enum"""

    def test_conditions(self):
        """Test material condition values"""
        assert MaterialCondition.MMC.value == "M"
        assert MaterialCondition.LMC.value == "L"
        assert MaterialCondition.RFS.value == "S"


class TestDatumReference:
    """Test DatumReference dataclass"""

    def test_datum_reference_creation(self):
        """Test creating datum reference"""
        ref = DatumReference(
            datum_letter="A",
            material_condition=MaterialCondition.MMC,
            precedence=1
        )
        assert ref.datum_letter == "A"
        assert ref.material_condition == MaterialCondition.MMC
        assert ref.precedence == 1

    def test_datum_reference_to_dict(self):
        """Test datum reference serialization"""
        ref = DatumReference(datum_letter="B", precedence=2)
        data = ref.to_dict()
        assert data["datum_letter"] == "B"
        assert data["material_condition"] is None
        assert data["precedence"] == 2


class TestDimension:
    """Test Dimension dataclass"""

    def test_dimension_creation(self):
        """Test creating dimension"""
        dim = Dimension(
            id="dim_1",
            dimension_type=DimensionType.LINEAR,
            nominal_value=25.4,
            tolerance_plus=0.1,
            tolerance_minus=0.1
        )
        assert dim.nominal_value == 25.4
        assert dim.tolerance_plus == 0.1

    def test_dimension_to_dict(self):
        """Test dimension serialization"""
        dim = Dimension(
            id="dim_1",
            dimension_type=DimensionType.DIAMETER,
            nominal_value=10.0,
            unit="mm"
        )
        data = dim.to_dict()
        assert data["type"] == "diameter"
        assert data["nominal"] == 10.0
        assert data["unit"] == "mm"


class TestGDTCallout:
    """Test GDTCallout dataclass"""

    def test_gdt_callout_creation(self):
        """Test creating GD&T callout"""
        callout = GDTCallout(
            id="gdt_1",
            symbol=GDTSymbol.POSITION,
            tolerance_value=0.25,
            material_condition=MaterialCondition.MMC
        )
        assert callout.symbol == GDTSymbol.POSITION
        assert callout.tolerance_value == 0.25
        assert callout.material_condition == MaterialCondition.MMC

    def test_gdt_callout_with_datums(self):
        """Test GD&T callout with datum references"""
        callout = GDTCallout(
            id="gdt_1",
            symbol=GDTSymbol.POSITION,
            tolerance_value=0.1,
            datums=[
                DatumReference("A", precedence=1),
                DatumReference("B", precedence=2),
                DatumReference("C", precedence=3)
            ]
        )
        assert len(callout.datums) == 3
        assert callout.datums[0].datum_letter == "A"

    def test_gdt_callout_to_dict(self):
        """Test GD&T callout serialization"""
        callout = GDTCallout(
            id="gdt_1",
            symbol=GDTSymbol.FLATNESS,
            tolerance_value=0.05
        )
        data = callout.to_dict()
        assert data["symbol"] == "flatness"
        assert data["tolerance"] == 0.05


class TestDatumFeature:
    """Test DatumFeature dataclass"""

    def test_datum_feature_creation(self):
        """Test creating datum feature"""
        datum = DatumFeature(
            letter="A",
            feature_id="face_1",
            feature_type="plane"
        )
        assert datum.letter == "A"
        assert datum.feature_type == "plane"

    def test_datum_feature_to_dict(self):
        """Test datum feature serialization"""
        datum = DatumFeature(letter="B", feature_id="axis_1")
        data = datum.to_dict()
        assert data["letter"] == "B"
        assert data["feature_id"] == "axis_1"


class TestNote:
    """Test Note dataclass"""

    def test_note_creation(self):
        """Test creating note"""
        note = Note(
            id="note_1",
            text="BREAK ALL SHARP EDGES",
            note_type="general"
        )
        assert note.text == "BREAK ALL SHARP EDGES"
        assert note.note_type == "general"


class TestPMIData:
    """Test PMIData dataclass"""

    @pytest.fixture
    def sample_pmi(self):
        """Create sample PMI data"""
        return PMIData(
            dimensions=[
                Dimension("dim_1", DimensionType.LINEAR, 25.0, tolerance_plus=0.1, tolerance_minus=0.1),
                Dimension("dim_2", DimensionType.DIAMETER, 10.0, tolerance_plus=0.05, tolerance_minus=0.05)
            ],
            gdt_callouts=[
                GDTCallout("gdt_1", GDTSymbol.FLATNESS, 0.05),
                GDTCallout("gdt_2", GDTSymbol.POSITION, 0.25, datums=[DatumReference("A")])
            ],
            datum_features=[
                DatumFeature("A", "face_1", "plane")
            ],
            notes=[
                Note("note_1", "ALL DIMS IN MM")
            ],
            source_file="test.step"
        )

    def test_pmi_to_dict(self, sample_pmi):
        """Test PMI serialization"""
        data = sample_pmi.to_dict()
        assert len(data["dimensions"]) == 2
        assert len(data["gdt_callouts"]) == 2
        assert data["summary"]["num_dimensions"] == 2
        assert data["summary"]["num_gdt_callouts"] == 2

    def test_pmi_to_json(self, sample_pmi):
        """Test PMI JSON serialization"""
        json_str = sample_pmi.to_json()
        data = json.loads(json_str)
        assert "dimensions" in data
        assert "gdt_callouts" in data

    def test_get_inspection_requirements(self, sample_pmi):
        """Test conversion to inspection requirements"""
        requirements = sample_pmi.get_inspection_requirements()
        assert len(requirements) == 4  # 2 dims + 2 GD&T

        dim_reqs = [r for r in requirements if r["type"] == "dimension"]
        gdt_reqs = [r for r in requirements if r["type"] == "gdt"]

        assert len(dim_reqs) == 2
        assert len(gdt_reqs) == 2

        # Check dimension requirement
        linear_dim = next(r for r in dim_reqs if r["characteristic"] == "linear")
        assert linear_dim["nominal"] == 25.0
        assert linear_dim["tolerance_plus"] == 0.1

        # Check GD&T requirement
        flatness = next(r for r in gdt_reqs if r["characteristic"] == "flatness")
        assert flatness["tolerance"] == 0.05


class TestPMIExtractor:
    """Test PMIExtractor class"""

    @pytest.fixture
    def extractor(self):
        return PMIExtractor()

    def test_extractor_creation(self, extractor):
        """Test extractor creation"""
        assert extractor._occ_available is None

    def test_extract_nonexistent_file(self, extractor):
        """Test extracting from nonexistent file"""
        pmi = extractor.extract("/nonexistent/file.step")
        assert pmi.is_complete is False
        assert "not found" in pmi.warnings[0].lower()

    def test_extract_unsupported_format(self, extractor):
        """Test extracting from unsupported format"""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        pmi = extractor.extract(temp_path)
        assert pmi.is_complete is False
        assert "not supported" in pmi.warnings[0].lower()

        Path(temp_path).unlink()

    def test_extract_from_text_dimensions(self, extractor):
        """Test extracting dimensions from text"""
        text = """
        Part dimensions:
        Length: 25.4 +/- 0.1 mm
        Width: 12.7 ± 0.05 mm
        Height: 6.35 +- 0.02 mm
        """
        pmi = extractor.extract_from_text(text)

        assert len(pmi.dimensions) == 3
        assert pmi.dimensions[0].nominal_value == 25.4
        assert pmi.dimensions[0].tolerance_plus == 0.1

    def test_extract_from_text_gdt(self, extractor):
        """Test extracting GD&T from text"""
        text = """
        GD&T Requirements:
        Flatness: 0.05 mm
        Position: 0.25 mm
        Perpendicularity: 0.10 mm
        """
        pmi = extractor.extract_from_text(text)

        assert len(pmi.gdt_callouts) == 3

        symbols = [c.symbol for c in pmi.gdt_callouts]
        assert GDTSymbol.FLATNESS in symbols
        assert GDTSymbol.POSITION in symbols
        assert GDTSymbol.PERPENDICULARITY in symbols

    def test_extract_from_text_datums(self, extractor):
        """Test extracting datums from text"""
        text = """
        Datum A: Bottom surface
        Datum B: Left edge
        Position tolerance 0.1 relative to Datum A and Datum B
        """
        pmi = extractor.extract_from_text(text)

        assert len(pmi.datum_features) == 2
        letters = [d.letter for d in pmi.datum_features]
        assert "A" in letters
        assert "B" in letters


class TestTextBasedParsing:
    """Test text-based PMI parsing"""

    @pytest.fixture
    def extractor(self):
        return PMIExtractor()

    def test_parse_datums_from_step(self, extractor):
        """Test parsing datum features from STEP-like content"""
        content = """
        #100=DATUM_FEATURE('A');
        #101=DATUM_FEATURE('B');
        #102=DATUM_TARGET('A1');
        """
        datums = extractor._parse_datums(content)

        assert len(datums) >= 2
        letters = [d.letter for d in datums]
        assert "A" in letters
        assert "B" in letters

    def test_parse_dimensions_from_step(self, extractor):
        """Test parsing dimensions from STEP-like content"""
        content = """
        #200=LINEAR_DIMENSION('dim1', 25.4);
        #201=DIAMETER_DIMENSION('dim2', 10.0);
        """
        dimensions = extractor._parse_dimensions(content)

        assert len(dimensions) == 2

    def test_parse_gdt_from_step(self, extractor):
        """Test parsing GD&T from STEP-like content"""
        content = """
        #300=FLATNESS_TOLERANCE('tol1', 0.05);
        #301=POSITION_TOLERANCE('tol2', 0.25);
        #302=CYLINDRICITY_TOLERANCE('tol3', 0.1);
        """
        callouts = extractor._parse_gdt_callouts(content)

        assert len(callouts) == 3
        symbols = [c.symbol for c in callouts]
        assert GDTSymbol.FLATNESS in symbols
        assert GDTSymbol.POSITION in symbols
        assert GDTSymbol.CYLINDRICITY in symbols


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_extract_pmi_nonexistent(self):
        """Test extract_pmi with nonexistent file"""
        pmi = extract_pmi("/nonexistent/file.step")
        assert pmi.is_complete is False

    def test_get_inspection_requirements_nonexistent(self):
        """Test get_inspection_requirements with nonexistent file"""
        reqs = get_inspection_requirements("/nonexistent/file.step")
        assert reqs == []

    def test_parse_gdt_text(self):
        """Test parse_gdt_text function"""
        text = "Flatness: 0.05 mm, Position: 0.1 mm"
        pmi = parse_gdt_text(text)

        assert len(pmi.gdt_callouts) >= 2


class TestGDTKeywordMapping:
    """Test GD&T keyword mapping"""

    def test_keyword_variations(self):
        """Test various GD&T keyword variations"""
        extractor = PMIExtractor()

        # Test different ways to express flatness
        text = """
        flatness: 0.05
        flat: 0.06
        """
        pmi = extractor.extract_from_text(text)

        flatness_callouts = [c for c in pmi.gdt_callouts if c.symbol == GDTSymbol.FLATNESS]
        assert len(flatness_callouts) == 2

    def test_true_position(self):
        """Test true position keyword"""
        extractor = PMIExtractor()
        text = "true position: 0.25 mm"
        pmi = extractor.extract_from_text(text)

        # Both "position" and "true position" keywords match
        assert len(pmi.gdt_callouts) >= 1
        # All matches should be POSITION
        assert all(c.symbol == GDTSymbol.POSITION for c in pmi.gdt_callouts)

    def test_roundness_alias(self):
        """Test roundness as alias for circularity"""
        extractor = PMIExtractor()
        text = "roundness = 0.02"
        pmi = extractor.extract_from_text(text)

        assert len(pmi.gdt_callouts) == 1
        assert pmi.gdt_callouts[0].symbol == GDTSymbol.CIRCULARITY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
