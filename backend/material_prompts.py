"""
Material-Specific AI Analysis Prompts

Provides domain-specific knowledge for different materials used in
CNC machining and sheet metal forming operations.

Supported materials:
- Aluminum alloys (6061, 7075, 5052, etc.)
- Steel alloys (304SS, 316SS, mild steel, tool steel)
- Titanium alloys (Ti-6Al-4V, CP titanium)
- Copper alloys (C110, brass)
- Plastics (ABS, PEEK, nylon, acetal)
- Composites (carbon fiber, fiberglass)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class MaterialCategory(Enum):
    """Material categories for manufacturing analysis"""
    ALUMINUM = "aluminum"
    STEEL = "steel"
    STAINLESS_STEEL = "stainless_steel"
    TITANIUM = "titanium"
    COPPER = "copper"
    BRASS = "brass"
    PLASTIC = "plastic"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


@dataclass
class MaterialProperties:
    """Physical and manufacturing properties of a material"""
    name: str
    category: MaterialCategory

    # Physical properties
    density_g_cm3: float = 0.0
    youngs_modulus_gpa: float = 0.0
    yield_strength_mpa: float = 0.0
    thermal_expansion_um_m_c: float = 0.0
    thermal_conductivity_w_m_k: float = 0.0

    # Manufacturing characteristics
    machinability_rating: str = "medium"  # poor, fair, medium, good, excellent
    formability_rating: str = "medium"
    springback_tendency: str = "medium"  # low, medium, high
    work_hardening: str = "medium"  # none, low, medium, high
    galling_risk: str = "low"  # low, medium, high

    # Typical tolerances achievable
    typical_tolerance_mm: float = 0.1
    best_tolerance_mm: float = 0.025

    # Common defect patterns
    common_defects: List[str] = field(default_factory=list)

    # Manufacturing notes
    notes: List[str] = field(default_factory=list)


# Material database with manufacturing-specific properties
MATERIAL_DATABASE: Dict[str, MaterialProperties] = {
    # Aluminum alloys
    "6061": MaterialProperties(
        name="6061-T6 Aluminum",
        category=MaterialCategory.ALUMINUM,
        density_g_cm3=2.70,
        youngs_modulus_gpa=68.9,
        yield_strength_mpa=276,
        thermal_expansion_um_m_c=23.6,
        thermal_conductivity_w_m_k=167,
        machinability_rating="excellent",
        formability_rating="good",
        springback_tendency="medium",
        work_hardening="low",
        typical_tolerance_mm=0.08,
        best_tolerance_mm=0.025,
        common_defects=[
            "Edge burrs from cutting",
            "Springback in bends (typically 1-3°)",
            "Surface scratching (soft material)",
            "Built-up edge marks",
            "Thermal distortion on thin sections"
        ],
        notes=[
            "Heat treated condition affects springback",
            "Soft material requires sharp tooling",
            "Good for complex geometries"
        ]
    ),
    "7075": MaterialProperties(
        name="7075-T6 Aluminum",
        category=MaterialCategory.ALUMINUM,
        density_g_cm3=2.81,
        youngs_modulus_gpa=71.7,
        yield_strength_mpa=503,
        thermal_expansion_um_m_c=23.4,
        thermal_conductivity_w_m_k=130,
        machinability_rating="good",
        formability_rating="fair",
        springback_tendency="high",
        work_hardening="medium",
        typical_tolerance_mm=0.1,
        best_tolerance_mm=0.03,
        common_defects=[
            "High springback (2-5°) in bends",
            "Cracking at tight bend radii",
            "Stress corrosion cracking risk",
            "Chatter marks from high hardness",
            "Residual stress warping"
        ],
        notes=[
            "Higher strength requires more overbending",
            "Minimum bend radius 2-3x thickness",
            "Consider stress relief after machining"
        ]
    ),
    "5052": MaterialProperties(
        name="5052-H32 Aluminum",
        category=MaterialCategory.ALUMINUM,
        density_g_cm3=2.68,
        youngs_modulus_gpa=70.3,
        yield_strength_mpa=193,
        thermal_expansion_um_m_c=23.8,
        thermal_conductivity_w_m_k=138,
        machinability_rating="good",
        formability_rating="excellent",
        springback_tendency="low",
        work_hardening="medium",
        typical_tolerance_mm=0.08,
        best_tolerance_mm=0.025,
        common_defects=[
            "Gummy chip formation",
            "Orange peel surface in formed areas",
            "Stretcher strain marks",
            "Soft material denting"
        ],
        notes=[
            "Best aluminum for deep drawing",
            "Low springback good for precision bends",
            "Use high rake angles for machining"
        ]
    ),

    # Steel alloys
    "304ss": MaterialProperties(
        name="304 Stainless Steel",
        category=MaterialCategory.STAINLESS_STEEL,
        density_g_cm3=8.0,
        youngs_modulus_gpa=193,
        yield_strength_mpa=215,
        thermal_expansion_um_m_c=17.2,
        thermal_conductivity_w_m_k=16.2,
        machinability_rating="fair",
        formability_rating="good",
        springback_tendency="high",
        work_hardening="high",
        galling_risk="high",
        typical_tolerance_mm=0.12,
        best_tolerance_mm=0.04,
        common_defects=[
            "Significant springback (3-6°)",
            "Work hardening at cut edges",
            "Galling on punched holes",
            "Heat discoloration at welds",
            "Tool wear marks",
            "Burrs requiring secondary ops"
        ],
        notes=[
            "Work hardens rapidly - use sharp tools",
            "Galling prevention: lubrication critical",
            "Overbend 3-5° to compensate springback",
            "Consider annealing for tight tolerances"
        ]
    ),
    "316ss": MaterialProperties(
        name="316 Stainless Steel",
        category=MaterialCategory.STAINLESS_STEEL,
        density_g_cm3=8.0,
        youngs_modulus_gpa=193,
        yield_strength_mpa=290,
        thermal_expansion_um_m_c=16.0,
        thermal_conductivity_w_m_k=16.3,
        machinability_rating="poor",
        formability_rating="fair",
        springback_tendency="high",
        work_hardening="high",
        galling_risk="high",
        typical_tolerance_mm=0.15,
        best_tolerance_mm=0.05,
        common_defects=[
            "Extreme springback (4-7°)",
            "Severe work hardening",
            "Built-up edge on tools",
            "Surface tearing at bends",
            "Galling in hole forming"
        ],
        notes=[
            "More difficult than 304SS",
            "Requires very sharp tooling",
            "Consider laser cutting over punching",
            "May need multiple forming passes"
        ]
    ),
    "mild_steel": MaterialProperties(
        name="Mild Steel (A36/1018)",
        category=MaterialCategory.STEEL,
        density_g_cm3=7.85,
        youngs_modulus_gpa=200,
        yield_strength_mpa=250,
        thermal_expansion_um_m_c=12.0,
        thermal_conductivity_w_m_k=51.9,
        machinability_rating="good",
        formability_rating="excellent",
        springback_tendency="low",
        work_hardening="low",
        typical_tolerance_mm=0.1,
        best_tolerance_mm=0.03,
        common_defects=[
            "Scale/oxide on surface",
            "Rust formation",
            "Minor springback (1-2°)",
            "Edge deformation from shearing",
            "Die marking on formed surfaces"
        ],
        notes=[
            "Easy to form and machine",
            "Predictable springback",
            "Rust protection required",
            "Good baseline for process setup"
        ]
    ),

    # Titanium
    "ti6al4v": MaterialProperties(
        name="Ti-6Al-4V (Grade 5)",
        category=MaterialCategory.TITANIUM,
        density_g_cm3=4.43,
        youngs_modulus_gpa=114,
        yield_strength_mpa=880,
        thermal_expansion_um_m_c=8.6,
        thermal_conductivity_w_m_k=6.7,
        machinability_rating="poor",
        formability_rating="poor",
        springback_tendency="high",
        work_hardening="high",
        galling_risk="high",
        typical_tolerance_mm=0.15,
        best_tolerance_mm=0.05,
        common_defects=[
            "Extreme springback (5-10°)",
            "Cracking at cold bends",
            "Severe tool wear",
            "Heat damage (alpha case)",
            "Galling on all contact surfaces",
            "Residual stress distortion"
        ],
        notes=[
            "Hot forming often required",
            "Very low thermal conductivity - heat builds",
            "Use titanium-specific tooling",
            "Minimum bend radius 4-6x thickness",
            "Stress relief critical"
        ]
    ),
    "cp_titanium": MaterialProperties(
        name="CP Titanium (Grade 2)",
        category=MaterialCategory.TITANIUM,
        density_g_cm3=4.51,
        youngs_modulus_gpa=103,
        yield_strength_mpa=275,
        thermal_expansion_um_m_c=8.6,
        thermal_conductivity_w_m_k=16.4,
        machinability_rating="fair",
        formability_rating="fair",
        springback_tendency="medium",
        work_hardening="medium",
        galling_risk="medium",
        typical_tolerance_mm=0.12,
        best_tolerance_mm=0.04,
        common_defects=[
            "Moderate springback (3-5°)",
            "Surface galling",
            "Tool wear",
            "Hydrogen pickup (welding)",
            "Orange peel surface"
        ],
        notes=[
            "Easier than Ti-6Al-4V",
            "Can be cold formed carefully",
            "Still needs lubricant for galling",
            "Better machinability than Grade 5"
        ]
    ),

    # Copper alloys
    "c110": MaterialProperties(
        name="C110 Copper (ETP)",
        category=MaterialCategory.COPPER,
        density_g_cm3=8.94,
        youngs_modulus_gpa=117,
        yield_strength_mpa=69,
        thermal_expansion_um_m_c=17.0,
        thermal_conductivity_w_m_k=388,
        machinability_rating="fair",
        formability_rating="excellent",
        springback_tendency="low",
        work_hardening="medium",
        typical_tolerance_mm=0.08,
        best_tolerance_mm=0.025,
        common_defects=[
            "Surface scratching (soft)",
            "Burr formation",
            "Gummy chips",
            "Orange peel at deep draws",
            "Thinning at stretch areas"
        ],
        notes=[
            "Very soft - handle carefully",
            "Excellent formability",
            "Work hardens significantly",
            "May need annealing between ops"
        ]
    ),
    "brass": MaterialProperties(
        name="Brass (C260/C360)",
        category=MaterialCategory.BRASS,
        density_g_cm3=8.53,
        youngs_modulus_gpa=110,
        yield_strength_mpa=124,
        thermal_expansion_um_m_c=20.0,
        thermal_conductivity_w_m_k=120,
        machinability_rating="excellent",
        formability_rating="good",
        springback_tendency="low",
        work_hardening="low",
        typical_tolerance_mm=0.08,
        best_tolerance_mm=0.02,
        common_defects=[
            "Season cracking risk",
            "Dezincification",
            "Surface tarnishing",
            "Burrs on free-machining grades"
        ],
        notes=[
            "Best machinability of copper alloys",
            "Free-machining grades chip well",
            "Stress relief prevents cracking",
            "Good for precision parts"
        ]
    ),

    # Plastics
    "abs": MaterialProperties(
        name="ABS Plastic",
        category=MaterialCategory.PLASTIC,
        density_g_cm3=1.05,
        youngs_modulus_gpa=2.3,
        yield_strength_mpa=45,
        thermal_expansion_um_m_c=95,
        thermal_conductivity_w_m_k=0.17,
        machinability_rating="excellent",
        formability_rating="good",
        springback_tendency="high",
        work_hardening="none",
        typical_tolerance_mm=0.15,
        best_tolerance_mm=0.05,
        common_defects=[
            "Melting/burning at high speeds",
            "High springback",
            "Stress whitening at bends",
            "Chipping on brittle grades",
            "Moisture absorption warping"
        ],
        notes=[
            "Low melting point - control heat",
            "Very high springback - thermal forming better",
            "Keep cutters sharp to prevent melting",
            "Dry before thermoforming"
        ]
    ),
    "peek": MaterialProperties(
        name="PEEK",
        category=MaterialCategory.PLASTIC,
        density_g_cm3=1.32,
        youngs_modulus_gpa=3.6,
        yield_strength_mpa=100,
        thermal_expansion_um_m_c=47,
        thermal_conductivity_w_m_k=0.25,
        machinability_rating="good",
        formability_rating="poor",
        springback_tendency="medium",
        work_hardening="none",
        typical_tolerance_mm=0.1,
        best_tolerance_mm=0.03,
        common_defects=[
            "Surface crazing",
            "Heat damage zones",
            "Residual stress warping",
            "Delamination (reinforced grades)"
        ],
        notes=[
            "High performance engineering plastic",
            "Better machinability than most plastics",
            "Anneal to relieve machining stress",
            "Expensive - minimize scrap"
        ]
    ),

    # Composites
    "carbon_fiber": MaterialProperties(
        name="Carbon Fiber Composite",
        category=MaterialCategory.COMPOSITE,
        density_g_cm3=1.55,
        youngs_modulus_gpa=70,
        yield_strength_mpa=600,
        thermal_expansion_um_m_c=2.0,
        thermal_conductivity_w_m_k=5,
        machinability_rating="poor",
        formability_rating="poor",
        springback_tendency="low",
        work_hardening="none",
        typical_tolerance_mm=0.15,
        best_tolerance_mm=0.05,
        common_defects=[
            "Delamination at edges",
            "Fiber pullout",
            "Dust/fiber hazard",
            "Resin burning",
            "Ply separation"
        ],
        notes=[
            "Requires diamond/carbide tooling",
            "Dust is hazardous - extraction required",
            "Support workpiece to prevent delam",
            "Use compression cutting where possible"
        ]
    ),
}


# Alias mappings for common material names
MATERIAL_ALIASES: Dict[str, str] = {
    # Aluminum
    "aluminum": "6061",
    "aluminium": "6061",
    "al6061": "6061",
    "al-6061": "6061",
    "6061-t6": "6061",
    "al7075": "7075",
    "al-7075": "7075",
    "7075-t6": "7075",
    "al5052": "5052",
    "5052-h32": "5052",

    # Steel
    "stainless": "304ss",
    "stainless steel": "304ss",
    "ss304": "304ss",
    "304": "304ss",
    "ss316": "316ss",
    "316": "316ss",
    "steel": "mild_steel",
    "carbon steel": "mild_steel",
    "a36": "mild_steel",
    "1018": "mild_steel",

    # Titanium
    "titanium": "cp_titanium",
    "ti64": "ti6al4v",
    "ti-6al-4v": "ti6al4v",
    "grade 5": "ti6al4v",
    "grade 2": "cp_titanium",

    # Copper
    "copper": "c110",
    "cu": "c110",

    # Plastics
    "plastic": "abs",
    "acrylic": "abs",

    # Composites
    "cfrp": "carbon_fiber",
    "composite": "carbon_fiber",
}


def get_material_properties(material_name: str) -> MaterialProperties:
    """
    Look up material properties by name.

    Args:
        material_name: Material name or alias

    Returns:
        MaterialProperties for the material, or generic properties if not found
    """
    name_lower = material_name.lower().strip()

    # Direct lookup
    if name_lower in MATERIAL_DATABASE:
        return MATERIAL_DATABASE[name_lower]

    # Alias lookup
    if name_lower in MATERIAL_ALIASES:
        return MATERIAL_DATABASE[MATERIAL_ALIASES[name_lower]]

    # Fuzzy match on key terms
    for key, alias in MATERIAL_ALIASES.items():
        if key in name_lower or name_lower in key:
            return MATERIAL_DATABASE[alias]

    # Return generic properties
    return MaterialProperties(
        name=material_name,
        category=MaterialCategory.UNKNOWN,
        common_defects=["Unknown material - generic defect analysis"],
        notes=["Material not in database - using generic parameters"]
    )


def build_material_specific_prompt(
    material_name: str,
    tolerance: float,
    process: str = "sheet_metal"
) -> str:
    """
    Build material-specific analysis instructions for AI prompt.

    Args:
        material_name: Name of the material
        tolerance: Part tolerance in mm
        process: Manufacturing process ("sheet_metal", "cnc_machining", "both")

    Returns:
        Material-specific prompt section
    """
    props = get_material_properties(material_name)

    prompt = f"""
## Material-Specific Analysis Guidelines: {props.name}

### Material Properties
- Category: {props.category.value.replace('_', ' ').title()}
- Density: {props.density_g_cm3} g/cm³
- Young's Modulus: {props.youngs_modulus_gpa} GPa
- Yield Strength: {props.yield_strength_mpa} MPa
- Thermal Expansion: {props.thermal_expansion_um_m_c} µm/m/°C

### Manufacturing Characteristics
- Machinability: {props.machinability_rating.capitalize()}
- Formability: {props.formability_rating.capitalize()}
- Springback Tendency: {props.springback_tendency.capitalize()}
- Work Hardening: {props.work_hardening.capitalize()}
"""

    if props.galling_risk != "low":
        prompt += f"- Galling Risk: {props.galling_risk.capitalize()} ⚠️\n"

    prompt += f"""
### Typical Tolerances
- Standard achievable: ±{props.typical_tolerance_mm}mm
- Best case: ±{props.best_tolerance_mm}mm
- This part specifies: ±{tolerance}mm"""

    if tolerance < props.best_tolerance_mm:
        prompt += f"\n  ⚠️ WARNING: Specified tolerance ({tolerance}mm) is tighter than typically achievable for this material!"
    elif tolerance < props.typical_tolerance_mm:
        prompt += f"\n  Note: Tight tolerance - extra process control required"

    prompt += f"""

### Known Defect Patterns for {props.name}
Look specifically for these material-specific issues:
"""

    for i, defect in enumerate(props.common_defects, 1):
        prompt += f"{i}. {defect}\n"

    if props.notes:
        prompt += f"""
### Manufacturing Notes
"""
        for note in props.notes:
            prompt += f"- {note}\n"

    # Add process-specific guidance
    if process in ["sheet_metal", "both"]:
        prompt += _get_sheet_metal_guidance(props)

    if process in ["cnc_machining", "both"]:
        prompt += _get_cnc_guidance(props)

    return prompt


def _get_sheet_metal_guidance(props: MaterialProperties) -> str:
    """Get sheet metal specific guidance for material"""
    guidance = """
### Sheet Metal Forming Considerations
"""

    if props.springback_tendency == "high":
        guidance += """- **HIGH SPRINGBACK EXPECTED**:
  - Look for under-bent angles (actual angle larger than nominal)
  - Check bend radii - may have opened up
  - Flanges may not meet specified angles
  - Recommend: Verify overbend compensation was applied
"""
    elif props.springback_tendency == "medium":
        guidance += """- **MODERATE SPRINGBACK**:
  - Check bend angles for 1-3° deviation
  - Verify radius hasn't opened excessively
"""

    if props.work_hardening == "high":
        guidance += """- **WORK HARDENING**:
  - Check for cracking at bend lines
  - Look for edge fracturing
  - Surface may show hardening marks
"""

    if props.galling_risk in ["medium", "high"]:
        guidance += f"""- **GALLING RISK ({props.galling_risk.upper()})**:
  - Check hole edges for material pickup
  - Look for torn/rough surfaces in formed areas
  - Scratch patterns from die contact
"""

    return guidance


def _get_cnc_guidance(props: MaterialProperties) -> str:
    """Get CNC machining specific guidance for material"""
    guidance = """
### CNC Machining Considerations
"""

    if props.machinability_rating in ["poor", "fair"]:
        guidance += f"""- **DIFFICULT TO MACHINE ({props.machinability_rating.upper()})**:
  - Check for tool marks and chatter
  - Look for built-up edge deposits
  - Surface finish may be inconsistent
  - Tool deflection more likely
"""

    if props.category == MaterialCategory.PLASTIC:
        guidance += """- **PLASTIC MATERIAL**:
  - Check for melting/burning at high-speed areas
  - Look for stress whitening
  - Surface may show heat damage
  - Dimensional changes from moisture possible
"""

    if props.category == MaterialCategory.COMPOSITE:
        guidance += """- **COMPOSITE MATERIAL**:
  - Check edges for delamination
  - Look for fiber pullout
  - Ply separation indicators
  - Resin damage from heat
"""

    if props.thermal_conductivity_w_m_k < 20:
        guidance += """- **LOW THERMAL CONDUCTIVITY**:
  - Heat concentrates at cut zone
  - Check for heat discoloration
  - Thermal expansion distortion risk
  - May see burn marks at high speeds
"""

    return guidance


def get_root_cause_hints(material_name: str, defect_type: str) -> List[Dict[str, Any]]:
    """
    Get likely root causes for a defect type based on material.

    Args:
        material_name: Material name
        defect_type: Type of defect observed

    Returns:
        List of likely root causes with confidence
    """
    props = get_material_properties(material_name)
    defect_lower = defect_type.lower()
    hints = []

    # Springback related
    if "springback" in defect_lower or "angle" in defect_lower or "bend" in defect_lower:
        if props.springback_tendency == "high":
            hints.append({
                "cause": f"Insufficient overbend compensation for {props.name}",
                "confidence": 0.85,
                "recommendation": f"Increase overbend by 3-5° for {props.name}"
            })
        hints.append({
            "cause": "Die opening too large for material thickness",
            "confidence": 0.6,
            "recommendation": "Use tighter V-die opening"
        })

    # Surface defects
    if "scratch" in defect_lower or "mark" in defect_lower:
        if props.category == MaterialCategory.ALUMINUM:
            hints.append({
                "cause": "Soft material surface damage during handling",
                "confidence": 0.7,
                "recommendation": "Use protective film, soft jaw inserts"
            })
        if props.galling_risk == "high":
            hints.append({
                "cause": f"Galling - {props.name} prone to material transfer",
                "confidence": 0.75,
                "recommendation": "Increase lubrication, use coated tooling"
            })

    # Cracking
    if "crack" in defect_lower or "fracture" in defect_lower:
        if props.work_hardening == "high":
            hints.append({
                "cause": f"Work hardening induced cracking in {props.name}",
                "confidence": 0.8,
                "recommendation": "Anneal before forming, increase bend radius"
            })
        hints.append({
            "cause": "Bend radius too tight for material",
            "confidence": 0.7,
            "recommendation": f"Increase bend radius to minimum {2 if props.formability_rating == 'poor' else 1.5}x thickness"
        })

    # Warping/distortion
    if "warp" in defect_lower or "distort" in defect_lower:
        if props.thermal_expansion_um_m_c > 20:
            hints.append({
                "cause": f"Thermal expansion ({props.thermal_expansion_um_m_c} µm/m/°C) causing distortion",
                "confidence": 0.65,
                "recommendation": "Allow cooling between operations, use coolant"
            })
        hints.append({
            "cause": "Residual stress release",
            "confidence": 0.6,
            "recommendation": "Stress relief before final machining"
        })

    return hints


# Export material categories for API
def list_supported_materials() -> List[Dict[str, str]]:
    """List all supported materials for API"""
    materials = []
    for key, props in MATERIAL_DATABASE.items():
        materials.append({
            "id": key,
            "name": props.name,
            "category": props.category.value,
            "machinability": props.machinability_rating,
            "formability": props.formability_rating
        })
    return materials
