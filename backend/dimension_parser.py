"""
Dimension Parser Module

Parses XLSX dimension specification files to extract:
- Linear dimensions
- Bend angles
- Radii
- Diameters

Supports Hebrew column headers as used in manufacturing specs.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Types of dimensions that can be specified."""
    LINEAR = "linear"
    ANGLE = "angle"
    RADIUS = "radius"
    DIAMETER = "diameter"
    UNKNOWN = "unknown"


# Hebrew to English type mapping
HEBREW_TYPE_MAP = {
    "ליניארית": DimensionType.LINEAR,
    "לינארית": DimensionType.LINEAR,  # Alternative spelling
    "זווית": DimensionType.ANGLE,
    "רדיוס": DimensionType.RADIUS,
    "קוטר": DimensionType.DIAMETER,
}

# English type keywords (fallback)
ENGLISH_TYPE_MAP = {
    "linear": DimensionType.LINEAR,
    "angle": DimensionType.ANGLE,
    "radius": DimensionType.RADIUS,
    "diameter": DimensionType.DIAMETER,
    "dia": DimensionType.DIAMETER,
}

# Tolerance column detection patterns
TOLERANCE_PATTERNS = ["טולרנס", "tolerance", "tol", "±"]

# Diameter symbol variants (including Unicode variants)
DIAMETER_SYMBOLS = ["Ø", "∅", "⌀"]


@dataclass
class DimensionSpec:
    """A single dimension specification from the XLSX."""
    dim_id: int
    dim_type: DimensionType
    value: float
    unit: str  # "mm" or "deg"
    tolerance_plus: float = 0.5  # Default tolerance
    tolerance_minus: float = 0.5
    description: str = ""
    raw_value: str = ""  # Original string value
    axis: Optional[str] = None  # Measurement axis: "X", "Y", "Z", or None

    def is_bend(self) -> bool:
        """Check if this dimension is a bend angle."""
        return self.dim_type == DimensionType.ANGLE

    def is_radius(self) -> bool:
        """Check if this dimension is a radius (often bend radius)."""
        return self.dim_type == DimensionType.RADIUS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dim_id": self.dim_id,
            "dim_type": self.dim_type.value,
            "value": self.value,
            "unit": self.unit,
            "tolerance_plus": self.tolerance_plus,
            "tolerance_minus": self.tolerance_minus,
            "description": self.description,
            "raw_value": self.raw_value,
            "is_bend": self.is_bend(),
            "axis": self.axis,
        }


@dataclass
class DimensionParseResult:
    """Result of parsing an XLSX file."""
    success: bool
    dimensions: List[DimensionSpec] = field(default_factory=list)
    bend_angles: List[DimensionSpec] = field(default_factory=list)
    linear_dims: List[DimensionSpec] = field(default_factory=list)
    radii: List[DimensionSpec] = field(default_factory=list)
    diameters: List[DimensionSpec] = field(default_factory=list)
    bend_radius: Optional[float] = None  # Common bend radius if specified
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "total_dimensions": len(self.dimensions),
            "bend_count": len(self.bend_angles),
            "linear_count": len(self.linear_dims),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "bend_angles": [d.to_dict() for d in self.bend_angles],
            "bend_radius": self.bend_radius,
            "warnings": self.warnings,
            "error": self.error,
        }


class DimensionParser:
    """
    Parser for XLSX dimension specification files.

    Expected format:
    - Column for dimension ID (מספר מידה)
    - Column for value (ערך)
    - Column for type (סיווג המידה)

    Handles Hebrew headers and various value formats.
    """

    # Known column name patterns (Hebrew and English)
    ID_PATTERNS = ["מספר מידה", "dim_id", "id", "#", "מספר"]
    VALUE_PATTERNS = ["ערך", "value", "ערך המידה", "nominal", "mm", "deg"]
    TYPE_PATTERNS = ["סיווג", "type", "סוג", "classification"]
    TOLERANCE_COL_PATTERNS = ["טולרנס", "tolerance", "tol"]
    AXIS_PATTERNS = ["axis", "ציר", "direction", "כיוון"]
    DESCRIPTION_PATTERNS = ["description", "תיאור", "desc", "notes", "הערות"]

    def __init__(self, default_tolerance: float = 0.5):
        """
        Initialize parser.

        Args:
            default_tolerance: Default tolerance for dimensions without specified tolerance
        """
        self.default_tolerance = default_tolerance

    def parse(self, file_path: str) -> DimensionParseResult:
        """
        Parse an XLSX file containing dimension specifications.

        Args:
            file_path: Path to XLSX file

        Returns:
            DimensionParseResult with parsed dimensions
        """
        path = Path(file_path)

        if not path.exists():
            return DimensionParseResult(
                success=False,
                error=f"File not found: {file_path}"
            )

        if path.suffix.lower() not in ['.xlsx', '.xls']:
            return DimensionParseResult(
                success=False,
                error=f"Unsupported file format: {path.suffix}"
            )

        try:
            # Read Excel file
            df = pd.read_excel(file_path, header=None)

            # Find header row and data
            header_row, columns = self._find_header_row(df)

            if header_row is None:
                return DimensionParseResult(
                    success=False,
                    error="Could not identify column headers in XLSX"
                )

            # Parse dimensions
            dimensions = self._parse_dimensions(df, header_row, columns)

            # Categorize dimensions
            bend_angles = [d for d in dimensions if d.is_bend()]
            linear_dims = [d for d in dimensions if d.dim_type == DimensionType.LINEAR]
            radii = [d for d in dimensions if d.dim_type == DimensionType.RADIUS]
            diameters = [d for d in dimensions if d.dim_type == DimensionType.DIAMETER]

            # Extract common bend radius if specified
            bend_radius = None
            if radii:
                # Use the first radius as bend radius (usually R3, etc.)
                bend_radius = radii[0].value

            warnings = []
            if not bend_angles:
                warnings.append("No bend angles found in XLSX")

            logger.info(f"Parsed {len(dimensions)} dimensions: "
                       f"{len(bend_angles)} angles, {len(linear_dims)} linear, "
                       f"{len(radii)} radii, {len(diameters)} diameters")

            return DimensionParseResult(
                success=True,
                dimensions=dimensions,
                bend_angles=bend_angles,
                linear_dims=linear_dims,
                radii=radii,
                diameters=diameters,
                bend_radius=bend_radius,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Error parsing XLSX: {e}")
            return DimensionParseResult(
                success=False,
                error=str(e)
            )

    def _find_header_row(self, df: pd.DataFrame) -> Tuple[Optional[int], Dict[str, int]]:
        """
        Find the header row in the dataframe.

        Returns:
            Tuple of (header_row_index, column_mapping)
        """
        for row_idx in range(min(10, len(df))):  # Check first 10 rows
            row = df.iloc[row_idx]

            columns = {}

            for col_idx, cell in enumerate(row):
                if pd.isna(cell):
                    continue

                cell_str = str(cell).strip().lower()

                # Check for ID column
                for pattern in self.ID_PATTERNS:
                    if pattern.lower() in cell_str:
                        columns['id'] = col_idx
                        break

                # Check for value column
                for pattern in self.VALUE_PATTERNS:
                    if pattern.lower() in cell_str:
                        columns['value'] = col_idx
                        break

                # Check for type column
                for pattern in self.TYPE_PATTERNS:
                    if pattern.lower() in cell_str:
                        columns['type'] = col_idx
                        break

                # Check for tolerance column
                for pattern in self.TOLERANCE_COL_PATTERNS:
                    if pattern.lower() in cell_str:
                        columns['tolerance'] = col_idx
                        break

                # Check for axis column
                for pattern in self.AXIS_PATTERNS:
                    if pattern.lower() in cell_str:
                        columns['axis'] = col_idx
                        break

                # Check for description column
                for pattern in self.DESCRIPTION_PATTERNS:
                    if pattern.lower() in cell_str:
                        columns['description'] = col_idx
                        break

            # Need at least ID and value columns
            if 'id' in columns and 'value' in columns:
                return row_idx, columns

        return None, {}

    def _parse_dimensions(
        self,
        df: pd.DataFrame,
        header_row: int,
        columns: Dict[str, int]
    ) -> List[DimensionSpec]:
        """
        Parse dimension rows from the dataframe.
        """
        dimensions = []

        # Start from row after header
        for row_idx in range(header_row + 1, len(df)):
            row = df.iloc[row_idx]

            # Get dimension ID
            id_val = row.iloc[columns['id']]
            if pd.isna(id_val):
                continue

            try:
                dim_id = int(float(id_val))
            except (ValueError, TypeError):
                continue

            # Get value
            value_val = row.iloc[columns['value']]
            if pd.isna(value_val):
                continue

            raw_value = str(value_val).strip()
            value, unit = self._parse_value(raw_value)

            if value is None:
                continue

            # Get type
            dim_type = DimensionType.UNKNOWN
            if 'type' in columns:
                type_val = row.iloc[columns['type']]
                if not pd.isna(type_val):
                    dim_type = self._parse_type(str(type_val).strip())

            # If type not specified, infer from value/unit
            if dim_type == DimensionType.UNKNOWN:
                dim_type = self._infer_type(raw_value, unit)

            # Parse tolerance from dedicated column if available
            tolerance_plus = self.default_tolerance
            tolerance_minus = self.default_tolerance

            if 'tolerance' in columns:
                tol_val = row.iloc[columns['tolerance']]
                if not pd.isna(tol_val):
                    parsed_tol = self._parse_tolerance(str(tol_val).strip())
                    if parsed_tol:
                        tolerance_plus, tolerance_minus = parsed_tol

            # Fallback: Set default tolerance based on type if not specified
            if tolerance_plus == self.default_tolerance and tolerance_minus == self.default_tolerance:
                if dim_type == DimensionType.ANGLE:
                    tolerance_plus = tolerance_minus = 0.5  # ±0.5° for angles

            # Parse axis column if available
            axis = None
            if 'axis' in columns:
                axis_val = row.iloc[columns['axis']]
                if not pd.isna(axis_val):
                    axis_str = str(axis_val).strip().upper()
                    if axis_str in ['X', 'Y', 'Z']:
                        axis = axis_str

            # Parse description column if available
            description = ""
            if 'description' in columns:
                desc_val = row.iloc[columns['description']]
                if not pd.isna(desc_val):
                    description = str(desc_val).strip()

            dim = DimensionSpec(
                dim_id=dim_id,
                dim_type=dim_type,
                value=value,
                unit=unit,
                tolerance_plus=tolerance_plus,
                tolerance_minus=tolerance_minus,
                raw_value=raw_value,
                axis=axis,
                description=description,
            )

            dimensions.append(dim)

        return dimensions

    def _parse_value(self, value_str: str) -> Tuple[Optional[float], str]:
        """
        Parse a value string like "15 mm", "163.9°", "R3 mm".

        Returns:
            Tuple of (numeric_value, unit)
        """
        value_str = value_str.strip()

        # Handle degree symbol
        if '°' in value_str or '◦' in value_str:
            # Extract number before degree symbol
            match = re.search(r'([\d.]+)\s*[°◦]', value_str)
            if match:
                return float(match.group(1)), "deg"

        # Handle radius format (R3, R3.5, etc.)
        if value_str.upper().startswith('R'):
            match = re.search(r'R\s*([\d.]+)', value_str, re.IGNORECASE)
            if match:
                return float(match.group(1)), "mm"

        # Handle diameter format (Ø12, ∅12, ⌀12, etc.)
        for sym in DIAMETER_SYMBOLS:
            if value_str.startswith(sym):
                match = re.search(r'[Ø∅⌀]\s*([\d.]+)', value_str)
                if match:
                    return float(match.group(1)), "mm"
                break

        # Handle plain number with optional mm
        match = re.search(r'([\d.]+)\s*(mm)?', value_str)
        if match:
            return float(match.group(1)), "mm"

        return None, ""

    def _parse_tolerance(self, tol_str: str) -> Optional[Tuple[float, float]]:
        """
        Parse tolerance strings like:
        - "±0.25" → (0.25, 0.25)
        - "±1°" or "±1◦" → (1.0, 1.0)
        - "+0.1/-0.2" → (0.1, 0.2)
        - "0.25" → (0.25, 0.25)

        Returns:
            Tuple of (tolerance_plus, tolerance_minus) or None if parsing fails
        """
        if not tol_str:
            return None

        tol_str = tol_str.strip()

        # Remove degree symbols if present (tolerance value is still numeric)
        tol_str = tol_str.replace('°', '').replace('◦', '')

        # Pattern: ±value (symmetric tolerance)
        match = re.match(r'[±]\s*([\d.]+)', tol_str)
        if match:
            val = float(match.group(1))
            return (val, val)

        # Pattern: +value/-value (asymmetric tolerance)
        match = re.match(r'\+\s*([\d.]+)\s*/\s*-\s*([\d.]+)', tol_str)
        if match:
            plus = float(match.group(1))
            minus = float(match.group(2))
            return (plus, minus)

        # Pattern: -value/+value (asymmetric tolerance, reversed order)
        match = re.match(r'-\s*([\d.]+)\s*/\s*\+\s*([\d.]+)', tol_str)
        if match:
            minus = float(match.group(1))
            plus = float(match.group(2))
            return (plus, minus)

        # Pattern: plain number (treat as symmetric)
        match = re.match(r'^([\d.]+)$', tol_str)
        if match:
            val = float(match.group(1))
            return (val, val)

        return None

    def _parse_type(self, type_str: str) -> DimensionType:
        """
        Parse dimension type from string.
        """
        type_str = type_str.strip().lower()

        # Check Hebrew types
        for hebrew, dim_type in HEBREW_TYPE_MAP.items():
            if hebrew in type_str:
                return dim_type

        # Check English types
        for english, dim_type in ENGLISH_TYPE_MAP.items():
            if english in type_str:
                return dim_type

        return DimensionType.UNKNOWN

    def _infer_type(self, raw_value: str, unit: str) -> DimensionType:
        """
        Infer dimension type from value format.
        """
        raw_value = raw_value.upper()

        if '°' in raw_value or '◦' in raw_value or unit == "deg":
            return DimensionType.ANGLE

        if raw_value.startswith('R'):
            return DimensionType.RADIUS

        for sym in DIAMETER_SYMBOLS:
            if raw_value.startswith(sym):
                return DimensionType.DIAMETER

        # Default to linear
        return DimensionType.LINEAR


def parse_dimension_file(file_path: str) -> DimensionParseResult:
    """
    Convenience function to parse a dimension file.

    Args:
        file_path: Path to XLSX file

    Returns:
        DimensionParseResult
    """
    parser = DimensionParser()
    return parser.parse(file_path)
