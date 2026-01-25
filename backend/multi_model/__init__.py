"""
Multi-Model AI Analysis Module for Sherman QC

This module implements a revolutionary multi-model AI orchestration system that:
1. Bridges 2D technical drawings to 3D point cloud analysis
2. Detects and numbers bends in sheet metal parts
3. Provides precise deviation localization with manufacturing-specific root causes

Models Used:
- Gemini 3 Pro: Visual reasoning for drawing analysis
- Claude Opus 4.5: Feature detection and root cause analysis
- GPT-5.2: 2D-3D correlation and spatial reasoning
"""

from .exchange_schema import (
    DrawingExtraction,
    DrawingDimension,
    GDTCallout,
    DrawingBend,
    Feature3D,
    FeatureType,
    PointCloudFeatures,
    CorrelationMapping,
    Correlation2D3D,
    BendFeature,
    BendRegionAnalysis,
    EnhancedRootCause,
    EnhancedAnalysisResult,
    BendDetectionResult,
    PipelineStage,
    PipelineStatus,
)

from .orchestrator import MultiModelOrchestrator
from .fallback_manager import FallbackManager

__all__ = [
    # Data structures
    "DrawingExtraction",
    "DrawingDimension",
    "GDTCallout",
    "DrawingBend",
    "Feature3D",
    "FeatureType",
    "PointCloudFeatures",
    "CorrelationMapping",
    "Correlation2D3D",
    "BendFeature",
    "BendRegionAnalysis",
    "EnhancedRootCause",
    "EnhancedAnalysisResult",
    "BendDetectionResult",
    "PipelineStage",
    "PipelineStatus",
    # Orchestration
    "MultiModelOrchestrator",
    "FallbackManager",
]
