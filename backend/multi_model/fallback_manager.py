"""
Fallback Manager for Multi-Model AI Pipeline

Provides local, rule-based fallback implementations when AI models
are unavailable. Ensures the system degrades gracefully while still
providing useful analysis.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

from .exchange_schema import (
    DrawingExtraction,
    DrawingDimension,
    DrawingBend,
    BendRegionAnalysis,
    BendFeature,
    EnhancedRootCause,
)

logger = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    enable_ocr: bool = True
    enable_heuristics: bool = True
    min_confidence: float = 0.3


class FallbackManager:
    """
    Manages fallback strategies when AI models are unavailable.

    Provides:
    1. Local drawing analysis using OCR + heuristics
    2. Rule-based root cause generation
    3. Pattern-based recommendations
    """

    # Common sheet metal bend angles
    COMMON_ANGLES = [30, 45, 60, 90, 120, 135, 150]

    # Springback compensation factors by material (approximate)
    SPRINGBACK_FACTORS = {
        "aluminum": 0.02,  # 2% springback
        "steel": 0.01,    # 1% springback
        "stainless": 0.015,
        "default": 0.015,
    }

    # Root cause templates
    ROOT_CAUSE_TEMPLATES = {
        "springback": {
            "category": "process",
            "description": "Springback in bend region - material elasticity causing under-bend",
            "evidence_pattern": "measured angle < nominal angle",
            "recommendations": [
                "Increase overbend compensation in press brake program",
                "Consider coining or bottoming die technique",
                "Verify material tensile strength matches specification",
            ],
        },
        "overbend": {
            "category": "process",
            "description": "Overbend in bend region - excessive forming force or wrong die opening",
            "evidence_pattern": "measured angle > nominal angle",
            "recommendations": [
                "Reduce overbend compensation",
                "Check die opening width",
                "Verify press tonnage settings",
            ],
        },
        "position_error": {
            "category": "tooling",
            "description": "Bend position deviation - likely backgauge calibration issue",
            "evidence_pattern": "apex position offset",
            "recommendations": [
                "Calibrate backgauge positioning",
                "Check for material slip during forming",
                "Verify blank dimensions before bending",
            ],
        },
        "radius_deviation": {
            "category": "tooling",
            "description": "Bend radius variation - punch tip wear or wrong tooling",
            "evidence_pattern": "radius deviation > 10%",
            "recommendations": [
                "Inspect punch tip for wear",
                "Verify correct punch radius is installed",
                "Check die V-opening matches specification",
            ],
        },
        "thickness_variation": {
            "category": "material",
            "description": "Material thickness variation affecting bend angle",
            "evidence_pattern": "inconsistent deviations across part",
            "recommendations": [
                "Measure incoming material thickness",
                "Adjust bend program for actual thickness",
                "Tighten incoming material specifications",
            ],
        },
        "grain_direction": {
            "category": "material",
            "description": "Grain direction effects - different springback parallel vs perpendicular to grain",
            "evidence_pattern": "directional deviation pattern",
            "recommendations": [
                "Orient blanks consistently relative to grain",
                "Apply different compensation for each bend direction",
                "Consider material with finer grain structure",
            ],
        },
    }

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()

    def extract_drawing_local(
        self,
        drawing_path: str,
    ) -> Optional[Dict]:
        """
        Extract drawing information using local methods.

        Uses OCR (if available) and heuristics to extract basic information.
        """
        result = {
            "dimensions": [],
            "gdt_callouts": [],
            "bends": [],
            "material": None,
            "thickness": None,
            "notes": [],
            "overall_confidence": 0.3,
        }

        if not os.path.exists(drawing_path):
            return result

        # Try OCR extraction
        if self.config.enable_ocr:
            ocr_result = self._try_ocr_extraction(drawing_path)
            if ocr_result:
                result = self._merge_results(result, ocr_result)

        # Apply heuristics
        if self.config.enable_heuristics:
            result = self._apply_heuristics(result)

        return result

    def _try_ocr_extraction(self, drawing_path: str) -> Optional[Dict]:
        """Attempt OCR extraction from drawing."""
        try:
            # Try pytesseract
            import pytesseract
            from PIL import Image

            if drawing_path.lower().endswith('.pdf'):
                # Convert PDF to image first
                try:
                    from pdf2image import convert_from_path

                    images = convert_from_path(drawing_path, first_page=1, last_page=1)
                    if images:
                        image = images[0]
                    else:
                        return None
                except ImportError:
                    return None
            else:
                image = Image.open(drawing_path)

            # Run OCR
            text = pytesseract.image_to_string(image)

            # Extract information from text
            return self._parse_ocr_text(text)

        except ImportError:
            logger.debug("pytesseract not available for OCR fallback")
            return None
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None

    def _parse_ocr_text(self, text: str) -> Dict:
        """Parse OCR text to extract drawing information."""
        result = {
            "dimensions": [],
            "bends": [],
            "notes": [],
        }

        # Look for angle specifications
        # Patterns: "90°", "90 DEG", "BEND 90"
        angle_patterns = [
            r'(\d+(?:\.\d+)?)\s*[°˚]',
            r'(\d+(?:\.\d+)?)\s*DEG',
            r'BEND\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*DEGREES',
        ]

        found_angles = set()
        for pattern in angle_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                angle = float(match)
                if 10 < angle < 180:
                    found_angles.add(angle)

        # Create bend entries
        for i, angle in enumerate(sorted(found_angles), 1):
            result["bends"].append({
                "id": f"B{i}",
                "sequence": i,
                "angle_nominal": angle,
                "angle_tolerance": 1.0,
                "confidence": 0.5,
            })

        # Look for dimensions
        # Patterns: "100.0", "50 ±0.5"
        dim_pattern = r'(\d+(?:\.\d+)?)\s*(?:±\s*(\d+(?:\.\d+)?))?'
        dim_matches = re.findall(dim_pattern, text)

        for i, (value, tolerance) in enumerate(dim_matches[:10]):  # Limit
            if value:
                val = float(value)
                if 1 < val < 10000:  # Reasonable dimension range (mm)
                    result["dimensions"].append({
                        "id": f"D{i+1}",
                        "type": "linear",
                        "nominal": val,
                        "tolerance_plus": float(tolerance) if tolerance else 0.5,
                        "tolerance_minus": float(tolerance) if tolerance else 0.5,
                        "confidence": 0.4,
                    })

        # Look for thickness
        thickness_patterns = [
            r'(?:THK|THICKNESS|T=)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:MM|mm)\s*(?:THK|THICK)',
        ]

        for pattern in thickness_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["thickness"] = float(match.group(1))
                break

        return result

    def _apply_heuristics(self, result: Dict) -> Dict:
        """Apply heuristics to improve extraction results."""
        # Snap angles to common values if close
        for bend in result.get("bends", []):
            angle = bend.get("angle_nominal", 0)
            for common in self.COMMON_ANGLES:
                if abs(angle - common) < 3:
                    bend["angle_nominal"] = common
                    bend["confidence"] = min(1.0, bend.get("confidence", 0.5) + 0.2)
                    break

        return result

    def _merge_results(self, base: Dict, overlay: Dict) -> Dict:
        """Merge two extraction results."""
        for key in ["dimensions", "bends", "gdt_callouts", "notes"]:
            if key in overlay and overlay[key]:
                base[key] = overlay[key]

        for key in ["material", "thickness"]:
            if key in overlay and overlay[key]:
                base[key] = overlay[key]

        return base

    def generate_basic_root_causes(
        self,
        bend_results: List[BendRegionAnalysis],
        deviations: np.ndarray,
        tolerance: float,
    ) -> List[EnhancedRootCause]:
        """
        Generate root causes using rule-based analysis.

        Analyzes bend results and deviation patterns to identify
        likely manufacturing issues.
        """
        root_causes = []
        cause_id = 0

        # Analyze each problematic bend
        for bend_result in bend_results:
            if bend_result.status == "pass":
                continue

            cause_id += 1

            # Springback detection
            if bend_result.springback_indicator > 0.3:
                template = self.ROOT_CAUSE_TEMPLATES["springback"]
                root_causes.append(EnhancedRootCause(
                    cause_id=f"RC{cause_id}",
                    category=template["category"],
                    description=template["description"],
                    severity="major" if bend_result.springback_indicator > 0.5 else "minor",
                    confidence=bend_result.springback_indicator,
                    affected_features=[],
                    affected_bends=[bend_result.bend.bend_id],
                    evidence=[
                        f"Bend {bend_result.bend.bend_id} shows {bend_result.angle_deviation_deg:.1f}° under-bend",
                        template["evidence_pattern"],
                    ],
                    recommendations=template["recommendations"][:2],
                    estimated_impact="Reduce angle deviation by 50-70%",
                    priority=1 if bend_result.springback_indicator > 0.5 else 2,
                ))

            # Overbend detection
            elif bend_result.over_bend_indicator > 0.3:
                cause_id += 1
                template = self.ROOT_CAUSE_TEMPLATES["overbend"]
                root_causes.append(EnhancedRootCause(
                    cause_id=f"RC{cause_id}",
                    category=template["category"],
                    description=template["description"],
                    severity="major" if bend_result.over_bend_indicator > 0.5 else "minor",
                    confidence=bend_result.over_bend_indicator,
                    affected_features=[],
                    affected_bends=[bend_result.bend.bend_id],
                    evidence=[
                        f"Bend {bend_result.bend.bend_id} shows {bend_result.angle_deviation_deg:.1f}° over-bend",
                        template["evidence_pattern"],
                    ],
                    recommendations=template["recommendations"][:2],
                    estimated_impact="Reduce angle deviation by 50-70%",
                    priority=1 if bend_result.over_bend_indicator > 0.5 else 2,
                ))

            # Position error
            if abs(bend_result.apex_deviation_mm) > tolerance:
                cause_id += 1
                template = self.ROOT_CAUSE_TEMPLATES["position_error"]
                root_causes.append(EnhancedRootCause(
                    cause_id=f"RC{cause_id}",
                    category=template["category"],
                    description=template["description"],
                    severity="major" if abs(bend_result.apex_deviation_mm) > tolerance * 2 else "minor",
                    confidence=0.6,
                    affected_features=[],
                    affected_bends=[bend_result.bend.bend_id],
                    evidence=[
                        f"Bend {bend_result.bend.bend_id} apex offset by {bend_result.apex_deviation_mm:.2f}mm",
                        template["evidence_pattern"],
                    ],
                    recommendations=template["recommendations"][:2],
                    estimated_impact="Reduce position error by 60-80%",
                    priority=2,
                ))

        # Analyze global patterns
        if len(deviations) > 0:
            # Check for systematic offset
            mean_dev = float(np.mean(deviations))
            if abs(mean_dev) > tolerance * 0.5:
                cause_id += 1
                root_causes.append(EnhancedRootCause(
                    cause_id=f"RC{cause_id}",
                    category="measurement",
                    description="Systematic offset detected - possible alignment or datum issue",
                    severity="major",
                    confidence=0.5,
                    affected_features=[],
                    affected_bends=[],
                    evidence=[
                        f"Mean deviation is {mean_dev:.3f}mm (non-zero offset)",
                        "Consistent bias across part suggests systematic error",
                    ],
                    recommendations=[
                        "Verify scan alignment with reference geometry",
                        "Check datum surface condition",
                        "Re-calibrate measurement system",
                    ],
                    estimated_impact="Eliminate systematic offset",
                    priority=1,
                ))

            # Check for high variability
            std_dev = float(np.std(deviations))
            if std_dev > tolerance:
                cause_id += 1
                template = self.ROOT_CAUSE_TEMPLATES["thickness_variation"]
                root_causes.append(EnhancedRootCause(
                    cause_id=f"RC{cause_id}",
                    category=template["category"],
                    description=template["description"],
                    severity="minor",
                    confidence=0.4,
                    affected_features=[],
                    affected_bends=[],
                    evidence=[
                        f"High deviation variability (σ = {std_dev:.3f}mm)",
                        template["evidence_pattern"],
                    ],
                    recommendations=template["recommendations"],
                    estimated_impact="Reduce variation by 30-50%",
                    priority=3,
                ))

        # Sort by priority
        root_causes.sort(key=lambda x: x.priority)

        return root_causes

    def generate_bend_recommendations(
        self,
        bend_result: BendRegionAnalysis,
        nominal_angle: Optional[float] = None,
    ) -> List[str]:
        """Generate specific recommendations for a bend."""
        recommendations = []

        if bend_result.springback_indicator > 0.3:
            compensation = bend_result.angle_deviation_deg * 1.1  # Slight overcompensation
            recommendations.append(
                f"Increase bend angle by {abs(compensation):.1f}° to compensate for springback"
            )

        if bend_result.over_bend_indicator > 0.3:
            adjustment = bend_result.angle_deviation_deg * 0.9
            recommendations.append(
                f"Reduce bend angle by {abs(adjustment):.1f}° to correct overbend"
            )

        if abs(bend_result.apex_deviation_mm) > 0.5:
            direction = "forward" if bend_result.apex_deviation_mm > 0 else "backward"
            recommendations.append(
                f"Adjust backgauge position {abs(bend_result.apex_deviation_mm):.2f}mm {direction}"
            )

        if bend_result.twist_angle_deg > 1.0:
            recommendations.append(
                "Check material hold-down pressure - twist indicates slipping during bend"
            )

        if not recommendations:
            recommendations.append("Bend is within acceptable tolerance - no action required")

        return recommendations

    def estimate_springback(
        self,
        material: str,
        thickness: float,
        bend_angle: float,
        bend_radius: float,
    ) -> float:
        """Estimate expected springback for given parameters."""
        factor = self.SPRINGBACK_FACTORS.get(
            material.lower() if material else "default",
            self.SPRINGBACK_FACTORS["default"],
        )

        # Springback increases with:
        # - Higher yield strength (material factor)
        # - Smaller bend radius
        # - Larger bend angle
        # - Thinner material (less constraint)

        r_t_ratio = bend_radius / thickness if thickness > 0 else 2.0
        angle_factor = bend_angle / 90.0  # Normalize to 90°

        estimated_springback = factor * angle_factor * (1 + 1 / r_t_ratio) * bend_angle

        return min(estimated_springback, bend_angle * 0.1)  # Cap at 10%
