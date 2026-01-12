"""
PMI (Product Manufacturing Information) Extractor

Extracts dimensional and GD&T information from CAD models:
- Dimensions (linear, angular, radial)
- GD&T callouts (position, flatness, etc.)
- Datum references
- Notes and annotations
- Tolerance specifications

Supports:
- STEP AP242 with embedded PMI
- STEP AP214 (limited PMI support)
- Model-Based Definition (MBD) extraction
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Types of dimensions"""
    LINEAR = "linear"
    ANGULAR = "angular"
    RADIAL = "radial"
    DIAMETER = "diameter"
    DEPTH = "depth"
    ORDINATE = "ordinate"


class GDTSymbol(Enum):
    """GD&T symbols per ASME Y14.5"""
    # Form tolerances
    STRAIGHTNESS = "straightness"
    FLATNESS = "flatness"
    CIRCULARITY = "circularity"
    CYLINDRICITY = "cylindricity"

    # Profile tolerances
    PROFILE_LINE = "profile_line"
    PROFILE_SURFACE = "profile_surface"

    # Orientation tolerances
    ANGULARITY = "angularity"
    PERPENDICULARITY = "perpendicularity"
    PARALLELISM = "parallelism"

    # Location tolerances
    POSITION = "position"
    CONCENTRICITY = "concentricity"
    SYMMETRY = "symmetry"

    # Runout tolerances
    CIRCULAR_RUNOUT = "circular_runout"
    TOTAL_RUNOUT = "total_runout"


class MaterialCondition(Enum):
    """Material condition modifiers"""
    MMC = "M"  # Maximum Material Condition
    LMC = "L"  # Least Material Condition
    RFS = "S"  # Regardless of Feature Size (implied)


@dataclass
class DatumReference:
    """Reference to a datum feature"""
    datum_letter: str  # A, B, C, etc.
    material_condition: Optional[MaterialCondition] = None
    precedence: int = 1  # 1=primary, 2=secondary, 3=tertiary

    def to_dict(self) -> Dict:
        return {
            "datum_letter": self.datum_letter,
            "material_condition": self.material_condition.value if self.material_condition else None,
            "precedence": self.precedence
        }


@dataclass
class Dimension:
    """Extracted dimension from PMI"""
    id: str
    dimension_type: DimensionType
    nominal_value: float
    unit: str = "mm"

    # Tolerances
    tolerance_plus: Optional[float] = None
    tolerance_minus: Optional[float] = None

    # Geometry reference
    feature_ids: List[str] = field(default_factory=list)
    location: Optional[Tuple[float, float, float]] = None

    # Display
    text: str = ""
    prefix: str = ""
    suffix: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.dimension_type.value,
            "nominal": self.nominal_value,
            "unit": self.unit,
            "tolerance_plus": self.tolerance_plus,
            "tolerance_minus": self.tolerance_minus,
            "text": self.text or f"{self.prefix}{self.nominal_value:.3f}{self.suffix}",
            "feature_ids": self.feature_ids
        }


@dataclass
class GDTCallout:
    """GD&T feature control frame"""
    id: str
    symbol: GDTSymbol
    tolerance_value: float
    unit: str = "mm"

    # Modifiers
    material_condition: Optional[MaterialCondition] = None
    projected_tolerance_zone: bool = False
    statistical_tolerance: bool = False
    tangent_plane: bool = False

    # Datum references
    datums: List[DatumReference] = field(default_factory=list)

    # Geometry reference
    feature_ids: List[str] = field(default_factory=list)
    location: Optional[Tuple[float, float, float]] = None

    # Display text
    text: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol.value,
            "tolerance": self.tolerance_value,
            "unit": self.unit,
            "material_condition": self.material_condition.value if self.material_condition else None,
            "datums": [d.to_dict() for d in self.datums],
            "feature_ids": self.feature_ids,
            "text": self.text or self._build_fcf_text()
        }

    def _build_fcf_text(self) -> str:
        """Build feature control frame text representation"""
        parts = [f"⌀{self.tolerance_value}" if self.symbol == GDTSymbol.POSITION else str(self.tolerance_value)]

        if self.material_condition and self.material_condition != MaterialCondition.RFS:
            parts.append(f"({self.material_condition.value})")

        for datum in self.datums:
            datum_text = datum.datum_letter
            if datum.material_condition:
                datum_text += f"({datum.material_condition.value})"
            parts.append(datum_text)

        return " | ".join(parts)


@dataclass
class DatumFeature:
    """Datum feature definition"""
    letter: str  # A, B, C, etc.
    feature_id: str
    feature_type: str = ""  # plane, axis, point
    location: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict:
        return {
            "letter": self.letter,
            "feature_id": self.feature_id,
            "type": self.feature_type,
            "location": self.location
        }


@dataclass
class Note:
    """General note or annotation"""
    id: str
    text: str
    note_type: str = "general"  # general, revision, material, process
    location: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.note_type
        }


@dataclass
class PMIData:
    """Complete PMI data extracted from CAD"""
    dimensions: List[Dimension] = field(default_factory=list)
    gdt_callouts: List[GDTCallout] = field(default_factory=list)
    datum_features: List[DatumFeature] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)

    # Metadata
    source_file: str = ""
    extraction_method: str = ""
    is_complete: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "gdt_callouts": [g.to_dict() for g in self.gdt_callouts],
            "datum_features": [d.to_dict() for d in self.datum_features],
            "notes": [n.to_dict() for n in self.notes],
            "summary": {
                "num_dimensions": len(self.dimensions),
                "num_gdt_callouts": len(self.gdt_callouts),
                "num_datums": len(self.datum_features),
                "num_notes": len(self.notes)
            },
            "metadata": {
                "source_file": self.source_file,
                "extraction_method": self.extraction_method,
                "is_complete": self.is_complete,
                "warnings": self.warnings
            }
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def get_inspection_requirements(self) -> List[Dict]:
        """
        Convert PMI to inspection requirements for QC.

        Returns:
            List of requirements with feature, tolerance, and GD&T specs
        """
        requirements = []

        # Add dimensions as requirements
        for dim in self.dimensions:
            req = {
                "id": dim.id,
                "type": "dimension",
                "characteristic": dim.dimension_type.value,
                "nominal": dim.nominal_value,
                "tolerance_plus": dim.tolerance_plus or 0.1,
                "tolerance_minus": dim.tolerance_minus or dim.tolerance_plus or 0.1,
                "unit": dim.unit,
                "feature_ids": dim.feature_ids
            }
            requirements.append(req)

        # Add GD&T callouts as requirements
        for gdt in self.gdt_callouts:
            req = {
                "id": gdt.id,
                "type": "gdt",
                "characteristic": gdt.symbol.value,
                "tolerance": gdt.tolerance_value,
                "unit": gdt.unit,
                "datums": [d.datum_letter for d in gdt.datums],
                "material_condition": gdt.material_condition.value if gdt.material_condition else None,
                "feature_ids": gdt.feature_ids
            }
            requirements.append(req)

        return requirements


class PMIExtractor:
    """
    Extract PMI from CAD files.

    Supports:
    - STEP AP242 (full semantic PMI)
    - STEP AP214 (limited, representation-based PMI)
    - Fallback parsing for non-semantic PMI

    Usage:
        extractor = PMIExtractor()
        pmi = extractor.extract("part.step")
        requirements = pmi.get_inspection_requirements()
    """

    # GD&T symbol Unicode mapping
    GDT_SYMBOLS = {
        "⏥": GDTSymbol.FLATNESS,
        "⌭": GDTSymbol.CYLINDRICITY,
        "○": GDTSymbol.CIRCULARITY,
        "⌀": GDTSymbol.POSITION,  # Often used with position
        "⌖": GDTSymbol.POSITION,
        "⫽": GDTSymbol.PARALLELISM,
        "⊥": GDTSymbol.PERPENDICULARITY,
        "∠": GDTSymbol.ANGULARITY,
        "⌓": GDTSymbol.CONCENTRICITY,
        "⌯": GDTSymbol.SYMMETRY,
        "↗": GDTSymbol.CIRCULAR_RUNOUT,
        "↗↗": GDTSymbol.TOTAL_RUNOUT,
        "⌒": GDTSymbol.PROFILE_LINE,
        "⌓": GDTSymbol.PROFILE_SURFACE,
        "⏤": GDTSymbol.STRAIGHTNESS,
    }

    # Keyword to symbol mapping for text-based extraction
    GDT_KEYWORDS = {
        "flatness": GDTSymbol.FLATNESS,
        "flat": GDTSymbol.FLATNESS,
        "cylindricity": GDTSymbol.CYLINDRICITY,
        "circularity": GDTSymbol.CIRCULARITY,
        "roundness": GDTSymbol.CIRCULARITY,
        "position": GDTSymbol.POSITION,
        "true position": GDTSymbol.POSITION,
        "parallelism": GDTSymbol.PARALLELISM,
        "parallel": GDTSymbol.PARALLELISM,
        "perpendicularity": GDTSymbol.PERPENDICULARITY,
        "perpendicular": GDTSymbol.PERPENDICULARITY,
        "angularity": GDTSymbol.ANGULARITY,
        "concentricity": GDTSymbol.CONCENTRICITY,
        "concentric": GDTSymbol.CONCENTRICITY,
        "symmetry": GDTSymbol.SYMMETRY,
        "runout": GDTSymbol.CIRCULAR_RUNOUT,
        "total runout": GDTSymbol.TOTAL_RUNOUT,
        "profile": GDTSymbol.PROFILE_SURFACE,
        "straightness": GDTSymbol.STRAIGHTNESS,
    }

    def __init__(self):
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

        return self._occ_available

    def extract(self, file_path: str) -> PMIData:
        """
        Extract PMI from CAD file.

        Args:
            file_path: Path to STEP file

        Returns:
            PMIData with extracted dimensions, GD&T, and notes
        """
        path = Path(file_path)
        if not path.exists():
            return PMIData(
                source_file=str(path),
                is_complete=False,
                warnings=[f"File not found: {file_path}"]
            )

        # Determine extraction method
        ext = path.suffix.lower()
        if ext not in [".step", ".stp"]:
            return PMIData(
                source_file=str(path),
                extraction_method="unsupported",
                is_complete=False,
                warnings=[f"PMI extraction not supported for format: {ext}"]
            )

        # Try semantic PMI extraction first
        if self._check_occ_available():
            try:
                pmi = self._extract_semantic_pmi(file_path)
                if pmi.dimensions or pmi.gdt_callouts:
                    return pmi
            except Exception as e:
                logger.warning(f"Semantic PMI extraction failed: {e}")

        # Fall back to text-based parsing
        return self._extract_text_based_pmi(file_path)

    def _extract_semantic_pmi(self, file_path: str) -> PMIData:
        """
        Extract semantic PMI from STEP AP242.

        This requires PythonOCC and a STEP file with embedded semantic PMI.
        """
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone

        pmi = PMIData(
            source_file=file_path,
            extraction_method="semantic_ap242"
        )

        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)

        if status != IFSelect_RetDone:
            pmi.warnings.append("Failed to read STEP file")
            pmi.is_complete = False
            return pmi

        # Transfer all roots
        reader.TransferRoots()

        # Get the model for PMI extraction
        # Note: Full AP242 PMI extraction requires additional OCC modules
        # that may not be available in all installations

        try:
            from OCC.Core.XSControl import XSControl_WorkSession
            from OCC.Core.STEPCAFControl import STEPCAFControl_Reader

            # Try to use XCAF reader for better PMI support
            xcaf_reader = STEPCAFControl_Reader()
            xcaf_reader.SetGDTMode(True)  # Enable GD&T reading
            xcaf_reader.ReadFile(file_path)

            # Extract PMI entities
            # This is a simplified implementation - full AP242 PMI
            # extraction is complex and depends on the file structure

            pmi.warnings.append("Semantic PMI extraction limited - using basic mode")

        except (ImportError, AttributeError) as e:
            pmi.warnings.append(f"XCAF PMI reader not available: {e}")

        return pmi

    def _extract_text_based_pmi(self, file_path: str) -> PMIData:
        """
        Extract PMI by parsing STEP file text.

        This is a fallback method that looks for PMI-like entities
        in the STEP file as text patterns.
        """
        pmi = PMIData(
            source_file=file_path,
            extraction_method="text_parsing"
        )

        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            pmi.warnings.append(f"Failed to read file: {e}")
            pmi.is_complete = False
            return pmi

        # Extract datum features
        datums = self._parse_datums(content)
        pmi.datum_features = datums

        # Extract dimensions
        dimensions = self._parse_dimensions(content)
        pmi.dimensions = dimensions

        # Extract GD&T callouts
        gdt_callouts = self._parse_gdt_callouts(content)
        pmi.gdt_callouts = gdt_callouts

        # Extract notes
        notes = self._parse_notes(content)
        pmi.notes = notes

        if not (dimensions or gdt_callouts or datums):
            pmi.warnings.append("No PMI data found in file - may not contain embedded PMI")
            pmi.is_complete = False

        return pmi

    def _parse_datums(self, content: str) -> List[DatumFeature]:
        """Parse datum features from STEP content"""
        datums = []

        # Look for DATUM_FEATURE entities
        datum_pattern = r"DATUM_FEATURE\s*\(\s*'([A-Z])'"
        matches = re.finditer(datum_pattern, content, re.IGNORECASE)

        for i, match in enumerate(matches):
            letter = match.group(1).upper()
            datums.append(DatumFeature(
                letter=letter,
                feature_id=f"datum_{letter}_{i}"
            ))

        # Also look for datum target pattern
        target_pattern = r"DATUM_TARGET\s*\(\s*'([A-Z]\d?)'"
        target_matches = re.finditer(target_pattern, content, re.IGNORECASE)

        for match in target_matches:
            target = match.group(1).upper()
            letter = target[0]
            if not any(d.letter == letter for d in datums):
                datums.append(DatumFeature(
                    letter=letter,
                    feature_id=f"datum_{letter}"
                ))

        return datums

    def _parse_dimensions(self, content: str) -> List[Dimension]:
        """Parse dimensions from STEP content"""
        dimensions = []
        dim_id = 0

        # Look for dimensional entities
        patterns = [
            # Linear dimension
            (r"LINEAR_DIMENSION\s*\([^)]*,\s*([0-9.]+)\s*\)", DimensionType.LINEAR),
            # Angular dimension
            (r"ANGULAR_DIMENSION\s*\([^)]*,\s*([0-9.]+)\s*\)", DimensionType.ANGULAR),
            # Diameter dimension
            (r"DIAMETER_DIMENSION\s*\([^)]*,\s*([0-9.]+)\s*\)", DimensionType.DIAMETER),
            # Radial dimension
            (r"RADIAL_DIMENSION\s*\([^)]*,\s*([0-9.]+)\s*\)", DimensionType.RADIAL),
            # Generic value association
            (r"VALUE_FORMAT_TYPE\s*\(\s*'NR1[^']*'\s*,\s*([0-9.]+)\s*\)", DimensionType.LINEAR),
        ]

        for pattern, dim_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    dim_id += 1
                    dimensions.append(Dimension(
                        id=f"dim_{dim_id}",
                        dimension_type=dim_type,
                        nominal_value=value
                    ))
                except ValueError:
                    continue

        # Look for tolerance specifications
        tol_pattern = r"TOLERANCE_VALUE\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)"
        tol_matches = re.finditer(tol_pattern, content, re.IGNORECASE)

        for i, match in enumerate(tol_matches):
            if i < len(dimensions):
                try:
                    dimensions[i].tolerance_plus = float(match.group(1))
                    dimensions[i].tolerance_minus = float(match.group(2))
                except ValueError:
                    continue

        return dimensions

    def _parse_gdt_callouts(self, content: str) -> List[GDTCallout]:
        """Parse GD&T callouts from STEP content"""
        callouts = []
        gdt_id = 0

        # Look for geometric tolerance entities
        gdt_patterns = [
            (r"FLATNESS_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.FLATNESS),
            (r"CYLINDRICITY_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.CYLINDRICITY),
            (r"CIRCULARITY_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.CIRCULARITY),
            (r"STRAIGHTNESS_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.STRAIGHTNESS),
            (r"POSITION_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.POSITION),
            (r"PERPENDICULARITY_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.PERPENDICULARITY),
            (r"PARALLELISM_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.PARALLELISM),
            (r"ANGULARITY_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.ANGULARITY),
            (r"SURFACE_PROFILE_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.PROFILE_SURFACE),
            (r"LINE_PROFILE_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.PROFILE_LINE),
            (r"CONCENTRICITY_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.CONCENTRICITY),
            (r"SYMMETRY_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.SYMMETRY),
            (r"CIRCULAR_RUNOUT_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.CIRCULAR_RUNOUT),
            (r"TOTAL_RUNOUT_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", GDTSymbol.TOTAL_RUNOUT),
            # Generic geometric tolerance
            (r"GEOMETRIC_TOLERANCE\s*\([^)]*,\s*([0-9.]+)\s*\)", None),
        ]

        for pattern, symbol in gdt_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    gdt_id += 1

                    # If no specific symbol, try to determine from context
                    actual_symbol = symbol
                    if actual_symbol is None:
                        actual_symbol = GDTSymbol.FLATNESS  # Default

                    callouts.append(GDTCallout(
                        id=f"gdt_{gdt_id}",
                        symbol=actual_symbol,
                        tolerance_value=value
                    ))
                except ValueError:
                    continue

        return callouts

    def _parse_notes(self, content: str) -> List[Note]:
        """Parse notes and annotations from STEP content"""
        notes = []
        note_id = 0

        # Look for text entities
        text_patterns = [
            r"DRAUGHTING_TEXT_LITERAL\s*\([^)]*'([^']+)'",
            r"TEXT_LITERAL\s*\([^)]*'([^']+)'",
            r"ANNOTATION_TEXT\s*\([^)]*'([^']+)'",
        ]

        for pattern in text_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                text = match.group(1).strip()
                if len(text) > 2:  # Filter out short garbage
                    note_id += 1
                    notes.append(Note(
                        id=f"note_{note_id}",
                        text=text
                    ))

        return notes

    def extract_from_text(self, text: str) -> PMIData:
        """
        Extract PMI from arbitrary text (e.g., drawing notes).

        Useful for parsing specifications provided as text.

        Args:
            text: Text containing dimensional and GD&T info

        Returns:
            PMIData with parsed specifications
        """
        pmi = PMIData(
            extraction_method="text_input"
        )

        # Parse dimensions from text
        # Pattern: 10.5 +/- 0.1 or 10.5 ± 0.1
        dim_pattern = r'(\d+\.?\d*)\s*(?:±|\+/-|\+-)\s*(\d+\.?\d*)\s*(mm|in)?'
        dim_matches = re.finditer(dim_pattern, text, re.IGNORECASE)

        dim_id = 0
        for match in dim_matches:
            dim_id += 1
            pmi.dimensions.append(Dimension(
                id=f"dim_{dim_id}",
                dimension_type=DimensionType.LINEAR,
                nominal_value=float(match.group(1)),
                tolerance_plus=float(match.group(2)),
                tolerance_minus=float(match.group(2)),
                unit=match.group(3) or "mm"
            ))

        # Parse GD&T from text
        for keyword, symbol in self.GDT_KEYWORDS.items():
            pattern = rf'{keyword}\s*[:=]?\s*(\d+\.?\d*)\s*(mm|in)?'
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for i, match in enumerate(matches):
                pmi.gdt_callouts.append(GDTCallout(
                    id=f"gdt_{keyword}_{i}",
                    symbol=symbol,
                    tolerance_value=float(match.group(1)),
                    unit=match.group(2) or "mm"
                ))

        # Parse datum references
        datum_pattern = r'datum\s*([A-Z])'
        datum_matches = re.finditer(datum_pattern, text, re.IGNORECASE)

        for match in datum_matches:
            letter = match.group(1).upper()
            if not any(d.letter == letter for d in pmi.datum_features):
                pmi.datum_features.append(DatumFeature(
                    letter=letter,
                    feature_id=f"datum_{letter}"
                ))

        return pmi


# Convenience functions
def extract_pmi(file_path: str) -> PMIData:
    """
    Extract PMI from CAD file.

    Args:
        file_path: Path to STEP file

    Returns:
        PMIData with extracted specifications
    """
    extractor = PMIExtractor()
    return extractor.extract(file_path)


def get_inspection_requirements(file_path: str) -> List[Dict]:
    """
    Get inspection requirements from CAD file.

    Args:
        file_path: Path to STEP file

    Returns:
        List of inspection requirements for QC
    """
    pmi = extract_pmi(file_path)
    return pmi.get_inspection_requirements()


def parse_gdt_text(text: str) -> PMIData:
    """
    Parse GD&T specifications from text.

    Args:
        text: Text containing dimensional/GD&T info

    Returns:
        PMIData with parsed specifications
    """
    extractor = PMIExtractor()
    return extractor.extract_from_text(text)
