# Enhanced QC System - Implementation Plan

## Overview

Transform Sherman QC from percentage-based accuracy reporting to dimension-specific analysis showing:
- Expected value (from XLSX truth source)
- Actual value (from scan)
- Delta (with % or mm)
- Per-bend failure analysis with bend numbering from reference documents

## Phase 1: Data Schema Enhancement

### 1.1 New XLSX Parser Module
**File:** `/backend/dimension_parser.py` (NEW)

```python
@dataclass
class DimensionSpec:
    dim_id: int
    dim_type: str  # bend_angle, bend_radius, hole_dia, linear, thickness
    expected_value: float
    unit: str  # mm, degree
    tolerance_plus: float
    tolerance_minus: float
    description: str
    cad_feature_id: Optional[str] = None  # For manual mapping

class DimensionParser:
    def parse_xlsx(self, path: str) -> List[DimensionSpec]
    def validate_specs(self, specs: List[DimensionSpec]) -> List[str]  # Returns warnings
```

### 1.2 Extended XLSX Format
Required columns:
| Column | Type | Description |
|--------|------|-------------|
| dim_id | int | Unique dimension identifier (matches balloon #) |
| dim_type | enum | bend_angle, bend_radius, hole_dia, linear, thickness |
| expected_value | float | Nominal value |
| unit | enum | mm, degree |
| tolerance_plus | float | Upper tolerance |
| tolerance_minus | float | Lower tolerance |
| description | str | Human-readable description |
| cad_feature_id | str | Optional: manual feature mapping |

## Phase 2: Feature Extraction Enhancement

### 2.1 CAD Feature Extractor
**File:** `/backend/cad_feature_extractor.py` (NEW)

Extract from CAD:
- **Holes**: Detect circular curves, compute diameter, store center position
- **Bends**: Use existing bend_detector, enhance with CAD-specific parameters
- **Linear dimensions**: Edge lengths, face dimensions
- **Surfaces**: Planar surfaces with area and orientation

```python
@dataclass
class CADFeature:
    feature_id: str
    feature_type: str  # hole, bend, edge, surface
    value: float  # diameter for holes, angle for bends, length for edges
    unit: str
    position: Tuple[float, float, float]
    metadata: Dict[str, Any]

class CADFeatureExtractor:
    def extract_holes(self, mesh) -> List[CADFeature]
    def extract_bends(self, points) -> List[CADFeature]
    def extract_edges(self, mesh) -> List[CADFeature]
    def extract_all(self, cad_path: str) -> List[CADFeature]
```

### 2.2 Scan Feature Extractor
**File:** `/backend/scan_feature_extractor.py` (NEW)

Same interface as CAD extractor but for point clouds:
- Hole detection via circle fitting
- Bend detection (existing algorithm, enhanced)
- Edge detection via boundary analysis

## Phase 3: Dimension Matching Engine

### 3.1 Feature-to-Dimension Matcher
**File:** `/backend/dimension_matcher.py` (NEW)

```python
@dataclass
class DimensionMatch:
    spec: DimensionSpec
    cad_feature: Optional[CADFeature]
    scan_feature: Optional[CADFeature]
    cad_value: Optional[float]
    scan_value: Optional[float]
    delta: Optional[float]
    delta_percent: Optional[float]
    status: str  # PASS, FAIL, WARNING, NOT_FOUND
    confidence: float

class DimensionMatcher:
    def match_features_to_specs(
        self,
        specs: List[DimensionSpec],
        cad_features: List[CADFeature],
        scan_features: List[CADFeature]
    ) -> List[DimensionMatch]
```

**Matching Strategy:**
1. **By value proximity**: For holes, find CAD hole closest to expected diameter
2. **By position**: For bends, order by spatial position and match by sequence
3. **By type**: Only match features of same type (hole-to-hole, bend-to-bend)
4. **Confidence scoring**: Based on how close the CAD value is to expected

## Phase 4: Enhanced Output

### 4.1 Dimension Report Generator
**File:** `/backend/dimension_report.py` (NEW)

```python
@dataclass
class DimensionReport:
    dimensions: List[DimensionMatch]
    summary: DimensionSummary
    bend_analysis: List[BendAnalysisResult]

@dataclass
class DimensionSummary:
    total: int
    passed: int
    failed: int
    warnings: int
    pass_rate: float
    critical_failures: List[str]
    worst_deviations: List[DimensionMatch]

class DimensionReportGenerator:
    def generate_report(self, matches: List[DimensionMatch]) -> DimensionReport
    def to_json(self, report: DimensionReport) -> Dict
    def to_html_table(self, report: DimensionReport) -> str
```

### 4.2 PDF Report Enhancement
**File:** `/backend/pdf_generator.py` (MODIFY)

Add new section: "Dimension Analysis"
- Table with columns: Dim#, Type, Description, Expected, Actual, Delta, Status
- Color coding: Green=Pass, Red=Fail, Yellow=Warning
- Highlight failed dimensions with root cause hints

### 4.3 Frontend Components
**Files:**
- `/frontend/react/src/components/DimensionAnalysisTable.tsx` (NEW)
- `/frontend/react/src/components/BendComparisonView.tsx` (NEW)

## Phase 5: Integration

### 5.1 API Endpoint Updates
**File:** `/backend/server.py` (MODIFY)

New endpoint parameters:
```python
@app.post("/api/analyze-enhanced")
async def analyze_enhanced(
    reference_file: UploadFile,  # CAD
    scan_file: UploadFile,       # PLY
    dimension_file: UploadFile,  # XLSX (required)
    drawing_file: Optional[UploadFile] = None,  # PDF (optional for AI enhancement)
    ...
)
```

New response fields:
```python
{
    "dimension_analysis": {
        "dimensions": [...],
        "summary": {...},
        "bend_analysis": [...]
    },
    # Existing fields...
}
```

### 5.2 QC Engine Integration
**File:** `/backend/qc_engine.py` (MODIFY)

Add to `run_analysis()`:
1. Parse XLSX specs
2. Extract CAD features
3. Extract scan features
4. Run dimension matching
5. Generate dimension report
6. Include in QCReport

## Implementation Order

1. **Week 1**: Phase 1 (XLSX Parser) + Phase 2.1 (CAD Feature Extractor)
2. **Week 2**: Phase 2.2 (Scan Feature Extractor) + Phase 3 (Dimension Matcher)
3. **Week 3**: Phase 4 (Output/Reports) + Phase 5 (Integration)
4. **Week 4**: Testing, refinement, edge cases

## File Summary

### New Files (7)
| File | Purpose |
|------|---------|
| `/backend/dimension_parser.py` | Parse XLSX dimension specs |
| `/backend/cad_feature_extractor.py` | Extract features from CAD |
| `/backend/scan_feature_extractor.py` | Extract features from scan |
| `/backend/dimension_matcher.py` | Match features to specs |
| `/backend/dimension_report.py` | Generate dimension reports |
| `/frontend/react/src/components/DimensionAnalysisTable.tsx` | Dimension table UI |
| `/frontend/react/src/components/BendComparisonView.tsx` | Bend comparison UI |

### Modified Files (3)
| File | Changes |
|------|---------|
| `/backend/qc_engine.py` | Integrate dimension analysis |
| `/backend/pdf_generator.py` | Add dimension section |
| `/backend/server.py` | New API parameters |

## Testing Strategy

1. **Unit tests** for each new module
2. **Integration test** with provided sample files
3. **Edge cases**: Missing dimensions, no bends, partial scans
4. **Validation**: Compare automated results with manual measurements

## Open Questions

1. How to handle multiple instances of same dimension (e.g., 4 identical holes)?
   - Option A: Report average
   - Option B: Report all instances
   - Option C: Report worst case

2. Bend numbering when CAD and scan have different number of detected bends?
   - Need fallback strategy for partial matches

3. Should AI (Gemini) be used to enhance XLSX from PDF?
   - If yes: Add optional AI enrichment step
   - If no: Require complete XLSX
