"""
Tests for FAI/PPAP Report Generator (AS9102)
"""

import pytest
import json
from pathlib import Path
import sys
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fai_report import (
    FAIReportGenerator,
    FAIReport,
    PartInfo,
    MaterialInfo,
    SpecialProcess,
    FunctionalTest,
    Characteristic,
    CharacteristicType,
    ResultCode,
    FAIType,
    create_fai_generator,
)


class TestFAIReportGeneratorCreation:
    """Test FAI report generator initialization"""

    def test_create_generator(self):
        """Test generator creation"""
        generator = create_fai_generator()
        assert generator is not None
        assert isinstance(generator, FAIReportGenerator)

    def test_create_generator_with_org_info(self):
        """Test generator creation with organization info"""
        generator = create_fai_generator(
            organization_name="ACME Manufacturing",
            cage_code="ABC12"
        )
        assert generator.organization_name == "ACME Manufacturing"
        assert generator.cage_code == "ABC12"


class TestPartInfo:
    """Test PartInfo dataclass"""

    def test_part_info_defaults(self):
        """Test part info with defaults"""
        part = PartInfo(
            part_number="PART-001",
            part_name="Test Bracket"
        )
        assert part.part_number == "PART-001"
        assert part.revision == "A"
        assert part.fai_type == FAIType.FULL

    def test_part_info_to_dict(self):
        """Test part info serialization"""
        part = PartInfo(
            part_number="PART-001",
            part_name="Test Bracket",
            revision="B",
            drawing_number="DWG-001",
            serial_number="SN12345",
            customer_name="Boeing"
        )
        data = part.to_dict()
        assert data["part_number"] == "PART-001"
        assert data["revision"] == "B"
        assert data["customer_name"] == "Boeing"


class TestMaterialInfo:
    """Test MaterialInfo dataclass"""

    def test_material_info_defaults(self):
        """Test material info with defaults"""
        mat = MaterialInfo(
            material_name="6061-T6 Aluminum",
            specification="AMS-QQ-A-250/11"
        )
        assert mat.result == ResultCode.PASS

    def test_material_info_to_dict(self):
        """Test material info serialization"""
        mat = MaterialInfo(
            material_name="6061-T6 Aluminum",
            specification="AMS-QQ-A-250/11",
            supplier="Alcoa",
            lot_number="LOT-12345",
            cert_number="CERT-67890"
        )
        data = mat.to_dict()
        assert data["material_name"] == "6061-T6 Aluminum"
        assert data["supplier"] == "Alcoa"
        assert data["result"] == "C"  # Conforming


class TestCharacteristic:
    """Test Characteristic dataclass"""

    def test_characteristic_defaults(self):
        """Test characteristic with defaults"""
        char = Characteristic(
            char_number=1,
            char_designator="A1"
        )
        assert char.char_type == CharacteristicType.DIMENSIONAL
        assert char.result == ResultCode.PASS
        assert char.unit == "mm"

    def test_characteristic_with_gdt(self):
        """Test characteristic with GD&T info"""
        char = Characteristic(
            char_number=1,
            char_designator="A1",
            nominal=10.0,
            tolerance_plus=0.1,
            tolerance_minus=0.1,
            measured_value=10.05,
            deviation=0.05,
            gdt_symbol="⌖",
            datum_reference="A|B|C"
        )
        data = char.to_dict()
        assert data["measured_value"] == 10.05
        assert data["gdt_symbol"] == "⌖"


class TestFAIReport:
    """Test FAIReport dataclass"""

    def test_report_creation(self):
        """Test basic report creation"""
        part_info = PartInfo(
            part_number="PART-001",
            part_name="Test Bracket"
        )
        report = FAIReport(part_info=part_info)
        assert report.part_info.part_number == "PART-001"
        assert len(report.characteristics) == 0

    def test_report_summary_calculation(self):
        """Test report summary calculation"""
        part_info = PartInfo(part_number="P001", part_name="Test")
        report = FAIReport(part_info=part_info)

        # Add some characteristics
        report.characteristics = [
            Characteristic(1, "A1", result=ResultCode.PASS),
            Characteristic(2, "A2", result=ResultCode.PASS),
            Characteristic(3, "A3", result=ResultCode.FAIL),
        ]

        report.calculate_summary()
        assert report.total_characteristics == 3
        assert report.conforming_count == 2
        assert report.nonconforming_count == 1
        assert report.overall_result == ResultCode.FAIL

    def test_report_to_dict(self):
        """Test report serialization"""
        part_info = PartInfo(part_number="P001", part_name="Test")
        report = FAIReport(
            part_info=part_info,
            report_number="FAI-001"
        )
        report.characteristics = [
            Characteristic(1, "A1", result=ResultCode.PASS),
        ]

        data = report.to_dict()
        assert data["report_number"] == "FAI-001"
        assert data["summary"]["total_characteristics"] == 1
        assert data["summary"]["pass_rate"] == 100.0


class TestFAIReportFromQCResult:
    """Test creating FAI reports from QC results"""

    @pytest.fixture
    def generator(self):
        return FAIReportGenerator(
            organization_name="Sherman Manufacturing",
            cage_code="SHM01"
        )

    @pytest.fixture
    def part_info(self):
        return PartInfo(
            part_number="BRACKET-001",
            part_name="Mounting Bracket",
            revision="A",
            customer_name="Test Customer"
        )

    @pytest.fixture
    def qc_result(self):
        return {
            "material": "aluminum",
            "tolerance": 0.1,
            "quality_score": 95.5,
            "statistics": {
                "mean_deviation": 0.02,
                "max_deviation": 0.08,
                "min_deviation": -0.05,
                "std_deviation": 0.015
            },
            "gdt_results": {
                "flatness": {
                    "measured_value_mm": 0.05,
                    "tolerance_mm": 0.1,
                    "conformance": "PASS"
                },
                "position": {
                    "measured_value_mm": 0.08,
                    "tolerance_mm": 0.15,
                    "conformance": "PASS"
                }
            },
            "regions": [
                {"name": "Top Surface", "mean_deviation": 0.03, "status": "OK"},
                {"name": "Side Surface", "mean_deviation": 0.05, "status": "OK"},
            ]
        }

    def test_create_report_from_qc(self, generator, part_info, qc_result):
        """Test creating report from QC results"""
        report = generator.create_report_from_qc_result(qc_result, part_info)

        assert report.part_info.part_number == "BRACKET-001"
        assert report.part_info.organization_name == "Sherman Manufacturing"
        assert len(report.characteristics) > 0
        assert report.overall_result in [ResultCode.PASS, ResultCode.FAIL]

    def test_report_has_gdt_characteristics(self, generator, part_info, qc_result):
        """Test that GD&T results are converted to characteristics"""
        report = generator.create_report_from_qc_result(qc_result, part_info)

        # Should have characteristics for GD&T results
        gdt_chars = [c for c in report.characteristics if c.gdt_symbol]
        assert len(gdt_chars) >= 2  # flatness and position

    def test_report_has_material(self, generator, part_info, qc_result):
        """Test that material is included"""
        report = generator.create_report_from_qc_result(qc_result, part_info)

        assert len(report.materials) > 0
        assert report.materials[0].material_name == "aluminum"

    def test_report_with_custom_materials(self, generator, part_info, qc_result):
        """Test report with custom material info"""
        materials = [
            MaterialInfo(
                material_name="6061-T6",
                specification="AMS-4027",
                supplier="Alcoa",
                result=ResultCode.PASS
            )
        ]

        report = generator.create_report_from_qc_result(
            qc_result,
            part_info,
            material_info=materials
        )

        assert report.materials[0].material_name == "6061-T6"
        assert report.materials[0].supplier == "Alcoa"


class TestFAIReportExport:
    """Test FAI report export functionality"""

    @pytest.fixture
    def generator(self):
        return FAIReportGenerator()

    @pytest.fixture
    def sample_report(self):
        part_info = PartInfo(
            part_number="TEST-001",
            part_name="Test Part"
        )
        report = FAIReport(
            part_info=part_info,
            report_number="FAI-TEST-001"
        )
        report.materials = [
            MaterialInfo("Steel", "ASTM-A36", result=ResultCode.PASS)
        ]
        report.characteristics = [
            Characteristic(
                char_number=1,
                char_designator="DIM-1",
                nominal=10.0,
                tolerance_plus=0.1,
                tolerance_minus=0.1,
                measured_value=10.05,
                deviation=0.05,
                result=ResultCode.PASS
            )
        ]
        return report

    def test_save_json(self, generator, sample_report):
        """Test saving report as JSON"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        generator.save_json(sample_report, filepath)

        # Verify file exists and is valid JSON
        assert Path(filepath).exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["report_number"] == "FAI-TEST-001"

        Path(filepath).unlink()

    def test_generate_html(self, generator, sample_report):
        """Test generating HTML report"""
        html = generator.generate_html(sample_report)

        assert "First Article Inspection Report" in html.upper() or "FIRST ARTICLE" in html.upper()
        assert "TEST-001" in html
        assert "FORM 1" in html
        assert "FORM 3" in html

    def test_save_html(self, generator, sample_report):
        """Test saving report as HTML"""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            filepath = f.name

        generator.save_html(sample_report, filepath)

        # Verify file exists
        assert Path(filepath).exists()
        with open(filepath) as f:
            content = f.read()
        assert "TEST-001" in content

        Path(filepath).unlink()


class TestResultCodes:
    """Test result code enum values"""

    def test_result_code_values(self):
        """Test result code values per AS9102"""
        assert ResultCode.PASS.value == "C"
        assert ResultCode.FAIL.value == "NC"
        assert ResultCode.WAIVER.value == "W"
        assert ResultCode.NA.value == "N/A"


class TestGDTSymbols:
    """Test GD&T symbol mapping"""

    def test_gdt_symbols_present(self):
        """Test that GD&T symbols are defined"""
        generator = FAIReportGenerator()
        assert "flatness" in generator.GDT_SYMBOLS
        assert "position" in generator.GDT_SYMBOLS
        assert generator.GDT_SYMBOLS["flatness"] == "⏥"
        assert generator.GDT_SYMBOLS["position"] == "⌖"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
