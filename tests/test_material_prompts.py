"""
Tests for Material-Specific AI Analysis Prompts
"""

import pytest
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from material_prompts import (
    MaterialCategory,
    MaterialProperties,
    MATERIAL_DATABASE,
    MATERIAL_ALIASES,
    get_material_properties,
    build_material_specific_prompt,
    get_root_cause_hints,
    list_supported_materials,
)


class TestMaterialCategory:
    """Test MaterialCategory enum"""

    def test_all_categories_defined(self):
        """Test all expected categories exist"""
        categories = [cat.value for cat in MaterialCategory]
        assert "aluminum" in categories
        assert "steel" in categories
        assert "stainless_steel" in categories
        assert "titanium" in categories
        assert "copper" in categories
        assert "plastic" in categories
        assert "composite" in categories
        assert "unknown" in categories


class TestMaterialProperties:
    """Test MaterialProperties dataclass"""

    def test_properties_creation(self):
        """Test creating material properties"""
        props = MaterialProperties(
            name="Test Material",
            category=MaterialCategory.ALUMINUM,
            density_g_cm3=2.7,
            youngs_modulus_gpa=70
        )
        assert props.name == "Test Material"
        assert props.category == MaterialCategory.ALUMINUM
        assert props.density_g_cm3 == 2.7

    def test_default_values(self):
        """Test default property values"""
        props = MaterialProperties(
            name="Test",
            category=MaterialCategory.UNKNOWN
        )
        assert props.machinability_rating == "medium"
        assert props.formability_rating == "medium"
        assert props.springback_tendency == "medium"
        assert props.common_defects == []


class TestMaterialDatabase:
    """Test the material database"""

    def test_database_not_empty(self):
        """Test database contains materials"""
        assert len(MATERIAL_DATABASE) > 0

    def test_aluminum_alloys_present(self):
        """Test aluminum alloys in database"""
        assert "6061" in MATERIAL_DATABASE
        assert "7075" in MATERIAL_DATABASE
        assert "5052" in MATERIAL_DATABASE

    def test_steel_alloys_present(self):
        """Test steel alloys in database"""
        assert "304ss" in MATERIAL_DATABASE
        assert "316ss" in MATERIAL_DATABASE
        assert "mild_steel" in MATERIAL_DATABASE

    def test_titanium_present(self):
        """Test titanium alloys in database"""
        assert "ti6al4v" in MATERIAL_DATABASE
        assert "cp_titanium" in MATERIAL_DATABASE

    def test_6061_properties(self):
        """Test 6061 aluminum properties"""
        al6061 = MATERIAL_DATABASE["6061"]
        assert al6061.category == MaterialCategory.ALUMINUM
        assert al6061.density_g_cm3 == 2.70
        assert al6061.machinability_rating == "excellent"
        assert len(al6061.common_defects) > 0

    def test_304ss_high_springback(self):
        """Test 304SS has high springback"""
        ss304 = MATERIAL_DATABASE["304ss"]
        assert ss304.springback_tendency == "high"
        assert ss304.work_hardening == "high"
        assert ss304.galling_risk == "high"

    def test_titanium_difficult_to_machine(self):
        """Test titanium has poor machinability"""
        ti64 = MATERIAL_DATABASE["ti6al4v"]
        assert ti64.machinability_rating == "poor"
        assert ti64.formability_rating == "poor"


class TestMaterialAliases:
    """Test material name aliases"""

    def test_aliases_not_empty(self):
        """Test aliases exist"""
        assert len(MATERIAL_ALIASES) > 0

    def test_aluminum_aliases(self):
        """Test aluminum aliases"""
        assert MATERIAL_ALIASES["aluminum"] == "6061"
        assert MATERIAL_ALIASES["al6061"] == "6061"
        assert MATERIAL_ALIASES["6061-t6"] == "6061"

    def test_stainless_aliases(self):
        """Test stainless steel aliases"""
        assert MATERIAL_ALIASES["stainless"] == "304ss"
        assert MATERIAL_ALIASES["ss304"] == "304ss"
        assert MATERIAL_ALIASES["304"] == "304ss"

    def test_titanium_aliases(self):
        """Test titanium aliases"""
        assert MATERIAL_ALIASES["titanium"] == "cp_titanium"
        assert MATERIAL_ALIASES["ti64"] == "ti6al4v"


class TestGetMaterialProperties:
    """Test get_material_properties function"""

    def test_direct_lookup(self):
        """Test direct material lookup"""
        props = get_material_properties("6061")
        assert props.name == "6061-T6 Aluminum"
        assert props.category == MaterialCategory.ALUMINUM

    def test_alias_lookup(self):
        """Test alias lookup"""
        props = get_material_properties("aluminum")
        assert props.name == "6061-T6 Aluminum"

    def test_case_insensitive(self):
        """Test case insensitive lookup"""
        props = get_material_properties("ALUMINUM")
        assert props.category == MaterialCategory.ALUMINUM

        props = get_material_properties("SS304")
        assert props.category == MaterialCategory.STAINLESS_STEEL

    def test_unknown_material(self):
        """Test unknown material returns generic"""
        props = get_material_properties("unknown_metal_xyz")
        assert props.category == MaterialCategory.UNKNOWN
        assert props.name == "unknown_metal_xyz"

    def test_partial_match(self):
        """Test partial name matching"""
        props = get_material_properties("6061 aluminum")
        # Should fuzzy match to 6061
        assert props.category == MaterialCategory.ALUMINUM


class TestBuildMaterialSpecificPrompt:
    """Test prompt building function"""

    def test_prompt_contains_material_name(self):
        """Test prompt includes material name"""
        prompt = build_material_specific_prompt("6061", 0.1)
        assert "6061-T6 Aluminum" in prompt

    def test_prompt_contains_properties(self):
        """Test prompt includes material properties"""
        prompt = build_material_specific_prompt("6061", 0.1)
        assert "Density" in prompt
        assert "Young's Modulus" in prompt
        assert "Yield Strength" in prompt

    def test_prompt_contains_defects(self):
        """Test prompt includes common defects"""
        prompt = build_material_specific_prompt("6061", 0.1)
        assert "Defect Patterns" in prompt

    def test_prompt_contains_tolerance_comparison(self):
        """Test prompt compares tolerance to achievable"""
        # Tight tolerance
        prompt = build_material_specific_prompt("6061", 0.01)
        assert "tighter than typically achievable" in prompt.lower() or "tight tolerance" in prompt.lower()

    def test_sheet_metal_process(self):
        """Test sheet metal specific guidance"""
        prompt = build_material_specific_prompt("304ss", 0.1, process="sheet_metal")
        assert "Sheet Metal" in prompt
        assert "springback" in prompt.lower()

    def test_cnc_machining_process(self):
        """Test CNC machining specific guidance"""
        prompt = build_material_specific_prompt("ti6al4v", 0.1, process="cnc_machining")
        assert "CNC Machining" in prompt

    def test_high_springback_material(self):
        """Test high springback material warning"""
        prompt = build_material_specific_prompt("304ss", 0.1, process="sheet_metal")
        assert "HIGH SPRINGBACK" in prompt

    def test_galling_risk_warning(self):
        """Test galling risk warning"""
        prompt = build_material_specific_prompt("304ss", 0.1, process="sheet_metal")
        assert "GALLING" in prompt


class TestGetRootCauseHints:
    """Test root cause hints function"""

    def test_springback_hints_for_high_springback_material(self):
        """Test springback hints for stainless steel"""
        hints = get_root_cause_hints("304ss", "springback")
        assert len(hints) > 0
        # Should have high confidence for SS
        springback_hint = next((h for h in hints if "overbend" in h["cause"].lower()), None)
        assert springback_hint is not None
        assert springback_hint["confidence"] > 0.7

    def test_scratch_hints_for_aluminum(self):
        """Test scratch hints for soft aluminum"""
        hints = get_root_cause_hints("6061", "surface scratch")
        assert len(hints) > 0
        soft_hint = next((h for h in hints if "soft" in h["cause"].lower()), None)
        assert soft_hint is not None

    def test_cracking_hints(self):
        """Test cracking hints for work hardening materials"""
        hints = get_root_cause_hints("304ss", "cracking")
        assert len(hints) > 0
        # Should mention work hardening
        wh_hint = next((h for h in hints if "hardening" in h["cause"].lower()), None)
        assert wh_hint is not None

    def test_warping_hints_for_high_expansion(self):
        """Test warping hints for high thermal expansion"""
        hints = get_root_cause_hints("abs", "warping")
        assert len(hints) > 0
        # ABS has high thermal expansion
        thermal_hint = next((h for h in hints if "thermal" in h["cause"].lower()), None)
        assert thermal_hint is not None


class TestListSupportedMaterials:
    """Test listing supported materials"""

    def test_list_returns_materials(self):
        """Test list returns material info"""
        materials = list_supported_materials()
        assert len(materials) > 0
        assert len(materials) == len(MATERIAL_DATABASE)

    def test_list_format(self):
        """Test list item format"""
        materials = list_supported_materials()
        first = materials[0]
        assert "id" in first
        assert "name" in first
        assert "category" in first
        assert "machinability" in first
        assert "formability" in first

    def test_list_includes_all_materials(self):
        """Test list includes all database materials"""
        materials = list_supported_materials()
        ids = [m["id"] for m in materials]
        for key in MATERIAL_DATABASE.keys():
            assert key in ids


class TestMaterialSpecificScenarios:
    """Test specific material scenarios"""

    def test_7075_high_strength_warnings(self):
        """Test 7075 has appropriate warnings"""
        al7075 = MATERIAL_DATABASE["7075"]
        assert al7075.springback_tendency == "high"
        assert any("springback" in d.lower() for d in al7075.common_defects)
        assert any("crack" in d.lower() for d in al7075.common_defects)

    def test_titanium_hot_forming_note(self):
        """Test titanium mentions hot forming"""
        ti64 = MATERIAL_DATABASE["ti6al4v"]
        assert any("hot" in n.lower() for n in ti64.notes)

    def test_carbon_fiber_delamination(self):
        """Test carbon fiber mentions delamination"""
        cf = MATERIAL_DATABASE["carbon_fiber"]
        assert any("delam" in d.lower() for d in cf.common_defects)

    def test_brass_machinability(self):
        """Test brass has excellent machinability"""
        brass = MATERIAL_DATABASE["brass"]
        assert brass.machinability_rating == "excellent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
