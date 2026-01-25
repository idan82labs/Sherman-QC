"""
AI Model Prompts for Multi-Model Analysis Pipeline

This module contains carefully crafted prompts for each AI model:
- Gemini 3 Pro: Drawing analysis and visual reasoning
- Claude Opus 4.5: Feature detection and root cause analysis
- GPT-5.2: 2D-3D correlation and spatial reasoning
"""

from .drawing_analysis import DRAWING_ANALYSIS_PROMPT, DRAWING_ANALYSIS_SYSTEM
from .feature_detection import FEATURE_DETECTION_PROMPT, FEATURE_VALIDATION_PROMPT
from .correlation import CORRELATION_PROMPT, CRITICAL_DEVIATION_PROMPT
from .root_cause import ROOT_CAUSE_PROMPT, BEND_SPECIFIC_PROMPT

__all__ = [
    "DRAWING_ANALYSIS_PROMPT",
    "DRAWING_ANALYSIS_SYSTEM",
    "FEATURE_DETECTION_PROMPT",
    "FEATURE_VALIDATION_PROMPT",
    "CORRELATION_PROMPT",
    "CRITICAL_DEVIATION_PROMPT",
    "ROOT_CAUSE_PROMPT",
    "BEND_SPECIFIC_PROMPT",
]
