"""
Feature Detection Module

Provides feature-by-feature dimension measurement for accurate three-way comparison:
XLSX specification <-> CAD design <-> Scan actual

Components:
- feature_types: Data structures for features, matches, and reports
- hole_detector: Cylindrical hole detection in CAD mesh
- cad_feature_extractor: Extract all measurable features from CAD
- feature_matcher: Match XLSX dimensions to CAD features
- scan_feature_measurer: Measure features in scan point cloud
- bend_detector: Progressive bend inspection for sheet metal QC
"""

from .feature_types import (
    FeatureType,
    MeasurableFeature,
    HoleFeature,
    LinearFeature,
    AngleFeature,
    FeatureRegistry,
    FeatureMatch,
    ThreeWayReport,
)

from .hole_detector import HoleDetector

from .cad_feature_extractor import CADFeatureExtractor

from .feature_matcher import (
    FeatureMatcher,
    match_xlsx_to_cad_features,
)

from .scan_feature_measurer import ScanFeatureMeasurer

from .bend_detector import (
    PlaneSegment,
    DetectedBend,
    BendSpecification,
    BendMatch,
    BendInspectionReport,
    BendDetector,
    ProgressiveBendMatcher,
    CADBendExtractor,
)

__all__ = [
    # Types
    "FeatureType",
    "MeasurableFeature",
    "HoleFeature",
    "LinearFeature",
    "AngleFeature",
    "FeatureRegistry",
    "FeatureMatch",
    "ThreeWayReport",
    # Detectors/Extractors
    "HoleDetector",
    "CADFeatureExtractor",
    # Matching
    "FeatureMatcher",
    "match_xlsx_to_cad_features",
    # Measurement
    "ScanFeatureMeasurer",
    # Bend Detection (Progressive Inspection)
    "PlaneSegment",
    "DetectedBend",
    "BendSpecification",
    "BendMatch",
    "BendInspectionReport",
    "BendDetector",
    "ProgressiveBendMatcher",
    "CADBendExtractor",
]
