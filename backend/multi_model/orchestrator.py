"""
Multi-Model AI Orchestrator for Enhanced QC Analysis

This orchestrator coordinates multiple AI models through a 4-stage pipeline:
1. Drawing Analysis (Gemini 3 Pro) - Extract specs from technical drawings
2. Feature Detection (Claude Opus 4.5) - Validate and enhance bend detection
3. 2D-3D Correlation (GPT-5.2) - Map drawing specs to point cloud features
4. Root Cause Analysis (Claude Opus 4.5) - Deep manufacturing insights

Each stage has fallback mechanisms for resilience.
"""

import asyncio
import json
import logging
import os
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np

from .exchange_schema import (
    DrawingExtraction,
    DrawingDimension,
    GDTCallout,
    DrawingBend,
    PointCloudFeatures,
    Feature3D,
    FeatureType,
    BendFeature,
    BendRegionAnalysis,
    Correlation2D3D,
    CorrelationMapping,
    CriticalDeviation,
    EnhancedRootCause,
    EnhancedAnalysisResult,
    PipelineStage,
    PipelineStatus,
    StageStatus,
    BendDetectionResult,
)
from .prompts import (
    DRAWING_ANALYSIS_PROMPT,
    DRAWING_ANALYSIS_SYSTEM,
    FEATURE_DETECTION_PROMPT,
    CORRELATION_PROMPT,
    ROOT_CAUSE_PROMPT,
)
from .fallback_manager import FallbackManager

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    provider: str  # "anthropic", "google", "openai"
    model_id: str
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.1


class MultiModelOrchestrator:
    """
    Orchestrates multi-model AI analysis pipeline.

    Model Assignments (using Gemini for all stages - excellent vision capabilities):
    - Stage 1 (Drawing Analysis): Gemini Pro - Visual understanding of technical drawings
    - Stage 2 (Feature Detection): Gemini Pro - Analytical reasoning for features
    - Stage 3 (2D-3D Correlation): Gemini Pro - Spatial correlation
    - Stage 4 (Root Cause): Gemini Pro - Manufacturing insights
    """

    # Model configurations - All using Gemini for unified vision-based analysis
    MODELS = {
        "drawing_analysis": ModelConfig(
            provider="google",
            model_id="gemini-2.0-flash",
            api_key_env="GOOGLE_API_KEY",
            max_tokens=8192,
            temperature=0.1,
        ),
        "feature_detection": ModelConfig(
            provider="google",
            model_id="gemini-2.0-flash",
            api_key_env="GOOGLE_API_KEY",
            max_tokens=4096,
            temperature=0.0,
        ),
        "correlation": ModelConfig(
            provider="google",
            model_id="gemini-2.0-flash",
            api_key_env="GOOGLE_API_KEY",
            max_tokens=4096,
            temperature=0.1,
        ),
        "root_cause": ModelConfig(
            provider="google",
            model_id="gemini-2.0-flash",
            api_key_env="GOOGLE_API_KEY",
            max_tokens=8192,
            temperature=0.2,
        ),
    }

    # Fallback model order for each stage
    FALLBACK_ORDER = {
        "drawing_analysis": ["google", "local"],
        "feature_detection": ["google", "local"],
        "correlation": ["google", "local"],
        "root_cause": ["google", "local"],
    }

    def __init__(
        self,
        enable_fallbacks: bool = True,
        max_retries: int = 2,
        timeout_seconds: float = 60.0,
    ):
        self.enable_fallbacks = enable_fallbacks
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.fallback_manager = FallbackManager()

        # Track pipeline status
        self._pipeline_status: Optional[PipelineStatus] = None
        self._models_used: List[str] = []
        self._fallbacks_triggered: List[str] = []

        # Import AI clients lazily
        self._anthropic_client = None
        self._google_client = None
        self._openai_client = None

    async def analyze(
        self,
        job_id: str,
        points: np.ndarray,
        deviations: np.ndarray,
        bend_detection_result: Optional[BendDetectionResult] = None,
        drawing_path: Optional[str] = None,
        part_info: Optional[Dict] = None,
        regional_analysis: Optional[Dict] = None,
        tolerance: float = 0.5,
    ) -> EnhancedAnalysisResult:
        """
        Run the complete multi-model analysis pipeline.

        Args:
            job_id: Unique job identifier
            points: Nx3 point cloud array
            deviations: Nx1 deviation values
            bend_detection_result: Pre-computed bend detection results
            drawing_path: Path to technical drawing (PDF)
            part_info: Part metadata
            regional_analysis: Pre-computed regional analysis
            tolerance: Deviation tolerance in mm

        Returns:
            EnhancedAnalysisResult with complete analysis
        """
        start_time = time.time()

        # Initialize pipeline status
        self._pipeline_status = PipelineStatus(
            job_id=job_id,
            current_stage=PipelineStage.DRAWING_ANALYSIS,
            started_at=datetime.now().isoformat(),
        )
        self._models_used = []
        self._fallbacks_triggered = []

        # Stage 1: Drawing Analysis
        drawing_extraction = None
        if drawing_path and os.path.exists(drawing_path):
            drawing_extraction = await self._stage1_drawing_analysis(drawing_path)
        else:
            self._update_stage_status(
                PipelineStage.DRAWING_ANALYSIS,
                "skipped",
                error_message="No drawing provided",
            )

        # Stage 2: Feature Detection (validate/enhance bend detection)
        point_cloud_features = await self._stage2_feature_detection(
            points=points,
            deviations=deviations,
            bend_detection_result=bend_detection_result,
            regional_analysis=regional_analysis,
        )

        # Stage 3: 2D-3D Correlation
        correlation_result = None
        if drawing_extraction:
            correlation_result = await self._stage3_correlation(
                drawing_extraction=drawing_extraction,
                point_cloud_features=point_cloud_features,
                deviations=deviations,
            )
        else:
            self._update_stage_status(
                PipelineStage.CORRELATION_2D_3D,
                "skipped",
                error_message="No drawing extraction available",
            )

        # Stage 4: Root Cause Analysis
        bend_results, root_causes = await self._stage4_root_cause_analysis(
            bend_detection_result=bend_detection_result,
            point_cloud_features=point_cloud_features,
            correlation_result=correlation_result,
            drawing_extraction=drawing_extraction,
            deviations=deviations,
            tolerance=tolerance,
            part_info=part_info,
        )

        # Compile final result
        processing_time = (time.time() - start_time) * 1000

        # Compute summary statistics
        bends_in_tol = sum(1 for b in bend_results if b.status == "pass")
        bends_out_tol = sum(1 for b in bend_results if b.status == "fail")
        critical_count = sum(1 for r in root_causes if r.severity == "critical")

        self._pipeline_status.current_stage = PipelineStage.COMPLETE
        self._pipeline_status.completed_at = datetime.now().isoformat()
        self._pipeline_status.total_duration_ms = processing_time
        self._pipeline_status.overall_progress = 100.0

        return EnhancedAnalysisResult(
            job_id=job_id,
            analysis_timestamp=datetime.now().isoformat(),
            drawing_extraction=drawing_extraction,
            point_cloud_features=point_cloud_features,
            correlation_2d_3d=correlation_result,
            bend_results=bend_results,
            enhanced_root_causes=root_causes,
            pipeline_status=self._pipeline_status,
            total_bends_detected=len(bend_results),
            bends_in_tolerance=bends_in_tol,
            bends_out_of_tolerance=bends_out_tol,
            critical_issues_count=critical_count,
            models_used=self._models_used,
            fallbacks_triggered=self._fallbacks_triggered,
            processing_time_ms=processing_time,
        )

    async def _stage1_drawing_analysis(
        self,
        drawing_path: str,
    ) -> Optional[DrawingExtraction]:
        """Stage 1: Analyze technical drawing with Gemini 3 Pro."""
        stage = PipelineStage.DRAWING_ANALYSIS
        self._update_stage_status(stage, "running")
        start_time = time.time()

        try:
            # Read drawing file
            drawing_data = self._read_drawing_file(drawing_path)
            if not drawing_data:
                raise ValueError("Could not read drawing file")

            # Try primary model (Gemini 3 Pro)
            result = await self._call_model_with_fallback(
                stage_name="drawing_analysis",
                prompt=DRAWING_ANALYSIS_PROMPT,
                system=DRAWING_ANALYSIS_SYSTEM,
                image_data=drawing_data,
            )

            if not result:
                # Use local fallback
                result = self.fallback_manager.extract_drawing_local(drawing_path)
                self._fallbacks_triggered.append("drawing_analysis_local")

            # Parse result into DrawingExtraction
            extraction = self._parse_drawing_result(result, drawing_path)

            duration = (time.time() - start_time) * 1000
            self._update_stage_status(stage, "completed", duration_ms=duration)

            return extraction

        except Exception as e:
            logger.error(f"Drawing analysis failed: {e}")
            self._update_stage_status(stage, "failed", error_message=str(e))
            return None

    async def _stage2_feature_detection(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        bend_detection_result: Optional[BendDetectionResult],
        regional_analysis: Optional[Dict],
    ) -> PointCloudFeatures:
        """Stage 2: Validate and enhance feature detection with Claude."""
        stage = PipelineStage.FEATURE_DETECTION
        self._update_stage_status(stage, "running")
        start_time = time.time()

        try:
            # Prepare context for AI validation
            context = self._prepare_feature_context(
                points=points,
                deviations=deviations,
                bend_detection_result=bend_detection_result,
                regional_analysis=regional_analysis,
            )

            # Call Claude for validation/enhancement
            prompt = FEATURE_DETECTION_PROMPT.format(**context)
            result = await self._call_model_with_fallback(
                stage_name="feature_detection",
                prompt=prompt,
            )

            # Process AI feedback
            enhanced_bends = self._process_feature_validation(
                bend_detection_result=bend_detection_result,
                ai_feedback=result,
            )

            # Build PointCloudFeatures
            features = PointCloudFeatures(
                scan_id=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                detection_timestamp=datetime.now().isoformat(),
                bends=enhanced_bends,
                surfaces=[],  # Could be populated from bend detector
                edges=[],
                other_features=[],
                total_points=len(points),
                processed_points=len(points),
                detection_confidence=0.85 if result else 0.7,
                model_used=self._models_used[-1] if self._models_used else "local",
            )

            duration = (time.time() - start_time) * 1000
            self._update_stage_status(stage, "completed", duration_ms=duration)

            return features

        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            self._update_stage_status(stage, "failed", error_message=str(e))

            # Return basic features from algorithmic detection
            bends = bend_detection_result.bends if bend_detection_result else []
            return PointCloudFeatures(
                scan_id=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                detection_timestamp=datetime.now().isoformat(),
                bends=bends,
                total_points=len(points),
                processed_points=len(points),
                detection_confidence=0.6,
                model_used="local_fallback",
            )

    async def _stage3_correlation(
        self,
        drawing_extraction: DrawingExtraction,
        point_cloud_features: PointCloudFeatures,
        deviations: np.ndarray,
    ) -> Optional[Correlation2D3D]:
        """Stage 3: Correlate 2D drawing with 3D features using GPT-5.2."""
        stage = PipelineStage.CORRELATION_2D_3D
        self._update_stage_status(stage, "running")
        start_time = time.time()

        try:
            # Prepare data for correlation
            context = {
                "drawing_data": json.dumps(drawing_extraction.to_dict(), indent=2),
                "detected_features": json.dumps(point_cloud_features.to_dict(), indent=2),
                "deviation_summary": self._summarize_deviations(deviations),
            }

            prompt = CORRELATION_PROMPT.format(**context)
            result = await self._call_model_with_fallback(
                stage_name="correlation",
                prompt=prompt,
            )

            # Parse correlation result
            correlation = self._parse_correlation_result(
                result=result,
                drawing_extraction=drawing_extraction,
                point_cloud_features=point_cloud_features,
            )

            duration = (time.time() - start_time) * 1000
            self._update_stage_status(stage, "completed", duration_ms=duration)

            return correlation

        except Exception as e:
            logger.error(f"Correlation failed: {e}")
            self._update_stage_status(stage, "failed", error_message=str(e))
            return None

    async def _stage4_root_cause_analysis(
        self,
        bend_detection_result: Optional[BendDetectionResult],
        point_cloud_features: PointCloudFeatures,
        correlation_result: Optional[Correlation2D3D],
        drawing_extraction: Optional[DrawingExtraction],
        deviations: np.ndarray,
        tolerance: float,
        part_info: Optional[Dict],
    ) -> Tuple[List[BendRegionAnalysis], List[EnhancedRootCause]]:
        """Stage 4: Deep root cause analysis with Claude Opus 4.5."""
        stage = PipelineStage.ROOT_CAUSE_ANALYSIS
        self._update_stage_status(stage, "running")
        start_time = time.time()

        bend_results = []
        root_causes = []

        try:
            # Build bend region analysis
            if bend_detection_result:
                for bend in bend_detection_result.bends:
                    # Find nominal from drawing if available
                    nominal_angle = self._find_nominal_angle(
                        bend, drawing_extraction
                    )

                    # Analyze this bend
                    analysis = self._analyze_single_bend(
                        bend=bend,
                        nominal_angle=nominal_angle,
                        deviations=deviations,
                        tolerance=tolerance,
                    )
                    bend_results.append(analysis)

            # Prepare context for root cause analysis
            context = {
                "part_info": json.dumps(part_info or {}, indent=2),
                "bend_results": json.dumps(
                    [b.to_dict() for b in bend_results], indent=2
                ),
                "deviation_summary": self._summarize_deviations(deviations),
                "correlation_results": json.dumps(
                    correlation_result.to_dict() if correlation_result else {},
                    indent=2,
                ),
                "critical_deviations": json.dumps(
                    [d.to_dict() for d in (correlation_result.critical_deviations if correlation_result else [])],
                    indent=2,
                ),
            }

            prompt = ROOT_CAUSE_PROMPT.format(**context)
            result = await self._call_model_with_fallback(
                stage_name="root_cause",
                prompt=prompt,
            )

            # Parse root cause results
            root_causes = self._parse_root_cause_result(result, bend_results)

            # Enhance bend results with root cause info
            bend_results = self._enhance_bend_results_with_causes(
                bend_results, root_causes, result
            )

            duration = (time.time() - start_time) * 1000
            self._update_stage_status(stage, "completed", duration_ms=duration)

        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}")
            self._update_stage_status(stage, "failed", error_message=str(e))

            # Generate basic root causes from bend analysis
            root_causes = self.fallback_manager.generate_basic_root_causes(
                bend_results, deviations, tolerance
            )

        return bend_results, root_causes

    async def _call_model_with_fallback(
        self,
        stage_name: str,
        prompt: str,
        system: Optional[str] = None,
        image_data: Optional[bytes] = None,
    ) -> Optional[Dict]:
        """Call AI model with fallback support."""
        config = self.MODELS.get(stage_name)
        fallback_order = self.FALLBACK_ORDER.get(stage_name, [])

        for provider in fallback_order:
            if provider == "local":
                continue  # Local fallback handled separately

            try:
                result = await self._call_provider(
                    provider=provider,
                    config=config,
                    prompt=prompt,
                    system=system,
                    image_data=image_data,
                )

                if result:
                    model_name = f"{provider}/{config.model_id if config else 'default'}"
                    self._models_used.append(model_name)

                    if provider != fallback_order[0]:
                        self._fallbacks_triggered.append(f"{stage_name}_{provider}")

                    return result

            except Exception as e:
                logger.warning(f"{provider} failed for {stage_name}: {e}")
                continue

        return None

    async def _call_provider(
        self,
        provider: str,
        config: Optional[ModelConfig],
        prompt: str,
        system: Optional[str] = None,
        image_data: Optional[bytes] = None,
    ) -> Optional[Dict]:
        """Call a specific AI provider."""
        if provider == "anthropic":
            return await self._call_anthropic(prompt, system, config)
        elif provider == "google":
            return await self._call_google(prompt, system, image_data, config)
        elif provider == "openai":
            return await self._call_openai(prompt, system, config)
        return None

    async def _call_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        config: Optional[ModelConfig],
    ) -> Optional[Dict]:
        """Call Anthropic Claude API."""
        try:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None

            if self._anthropic_client is None:
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)

            model_id = config.model_id if config else "claude-opus-4-5-20251101"

            messages = [{"role": "user", "content": prompt}]

            response = self._anthropic_client.messages.create(
                model=model_id,
                max_tokens=config.max_tokens if config else 4096,
                system=system or "",
                messages=messages,
            )

            text = response.content[0].text

            # Try to parse as JSON
            return self._extract_json(text)

        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            raise

    async def _call_google(
        self,
        prompt: str,
        system: Optional[str],
        image_data: Optional[bytes],
        config: Optional[ModelConfig],
    ) -> Optional[Dict]:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return None

            genai.configure(api_key=api_key)

            model_id = config.model_id if config else "gemini-3-pro"
            model = genai.GenerativeModel(model_id)

            # Build content
            content_parts = []
            if system:
                content_parts.append(system + "\n\n")

            if image_data:
                import PIL.Image
                import io

                image = PIL.Image.open(io.BytesIO(image_data))
                content_parts.append(image)

            content_parts.append(prompt)

            response = model.generate_content(content_parts)

            # Try to parse as JSON
            return self._extract_json(response.text)

        except Exception as e:
            logger.error(f"Google call failed: {e}")
            raise

    async def _call_openai(
        self,
        prompt: str,
        system: Optional[str],
        config: Optional[ModelConfig],
    ) -> Optional[Dict]:
        """Call OpenAI API."""
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None

            if self._openai_client is None:
                self._openai_client = openai.OpenAI(api_key=api_key)

            model_id = config.model_id if config else "gpt-5.2-turbo"

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self._openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=config.max_tokens if config else 4096,
                temperature=config.temperature if config else 0.1,
            )

            text = response.choices[0].message.content

            # Try to parse as JSON
            return self._extract_json(text)

        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from model response."""
        if not text:
            return None

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        import re

        # Look for JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _read_drawing_file(self, drawing_path: str) -> Optional[bytes]:
        """Read a drawing file (PDF or image)."""
        try:
            if drawing_path.lower().endswith('.pdf'):
                # Convert PDF to image
                try:
                    import fitz  # PyMuPDF

                    doc = fitz.open(drawing_path)
                    page = doc[0]  # First page
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    return pix.tobytes("png")
                except ImportError:
                    # Try pdf2image
                    from pdf2image import convert_from_path

                    images = convert_from_path(drawing_path, first_page=1, last_page=1)
                    if images:
                        import io

                        buf = io.BytesIO()
                        images[0].save(buf, format='PNG')
                        return buf.getvalue()
            else:
                # Direct image read
                with open(drawing_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read drawing: {e}")
            return None

    def _parse_drawing_result(
        self,
        result: Optional[Dict],
        drawing_path: str,
    ) -> DrawingExtraction:
        """Parse AI result into DrawingExtraction."""
        if not result:
            return DrawingExtraction(
                drawing_id=os.path.basename(drawing_path),
                extraction_timestamp=datetime.now().isoformat(),
                extraction_confidence=0.0,
            )

        dimensions = []
        for dim in result.get("dimensions", []):
            dimensions.append(DrawingDimension(
                id=dim.get("id", f"D{len(dimensions)+1}"),
                dimension_type=dim.get("type", "linear"),
                nominal=float(dim.get("nominal", 0)),
                tolerance_plus=float(dim.get("tolerance_plus", 0.5)),
                tolerance_minus=float(dim.get("tolerance_minus", 0.5)),
                unit=dim.get("unit", "mm"),
                feature_ref=dim.get("feature_ref"),
                location_hint=dim.get("location_hint"),
            ))

        gdt_callouts = []
        for gdt in result.get("gdt_callouts", []):
            gdt_callouts.append(GDTCallout(
                id=gdt.get("id", f"G{len(gdt_callouts)+1}"),
                gdt_type=gdt.get("type", "flatness"),
                tolerance=float(gdt.get("tolerance", 0.1)),
                unit=gdt.get("unit", "mm"),
                feature_ref=gdt.get("feature_ref"),
                datums=gdt.get("datums", []),
                modifier=gdt.get("modifier"),
            ))

        bends = []
        for bend in result.get("bends", []):
            bends.append(DrawingBend(
                id=bend.get("id", f"B{len(bends)+1}"),
                sequence=int(bend.get("sequence", len(bends)+1)),
                angle_nominal=float(bend.get("angle_nominal", 90)),
                angle_tolerance=float(bend.get("angle_tolerance", 1.0)),
                radius=float(bend.get("radius", 2.0)),
                direction=bend.get("direction", "up"),
                bend_line_location=bend.get("bend_line_location"),
            ))

        material_info = result.get("material", {})
        part_info = result.get("part_info", {})

        return DrawingExtraction(
            drawing_id=os.path.basename(drawing_path),
            extraction_timestamp=datetime.now().isoformat(),
            dimensions=dimensions,
            gdt_callouts=gdt_callouts,
            bends=bends,
            material=material_info.get("specification"),
            thickness=material_info.get("thickness"),
            part_number=part_info.get("part_number"),
            revision=part_info.get("revision"),
            notes=result.get("notes", []),
            extraction_confidence=float(result.get("overall_confidence", 0.8)),
            model_used=self._models_used[-1] if self._models_used else "",
            raw_extraction=result,
        )

    def _prepare_feature_context(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        bend_detection_result: Optional[BendDetectionResult],
        regional_analysis: Optional[Dict],
    ) -> Dict:
        """Prepare context for feature detection prompt."""
        bend_data = "No bends detected"
        if bend_detection_result and bend_detection_result.bends:
            bend_data = json.dumps(
                [b.to_dict() for b in bend_detection_result.bends],
                indent=2,
            )

        return {
            "bend_detection_results": bend_data,
            "total_points": len(points),
            "min_deviation": float(np.min(deviations)),
            "max_deviation": float(np.max(deviations)),
            "mean_deviation": float(np.mean(deviations)),
            "std_deviation": float(np.std(deviations)),
            "regional_analysis": json.dumps(regional_analysis or {}, indent=2),
        }

    def _process_feature_validation(
        self,
        bend_detection_result: Optional[BendDetectionResult],
        ai_feedback: Optional[Dict],
    ) -> List[BendFeature]:
        """Process AI validation feedback on detected features."""
        if not bend_detection_result:
            return []

        bends = list(bend_detection_result.bends)

        if not ai_feedback:
            return bends

        # Apply AI suggestions
        validation = ai_feedback.get("validation", {})
        flagged = validation.get("flagged_bends", [])

        for flag in flagged:
            bend_id = flag.get("bend_id")
            # Could adjust confidence or mark bends based on AI feedback
            for bend in bends:
                if bend.bend_id == bend_id:
                    # Reduce confidence for flagged bends
                    bend.detection_confidence *= 0.7
                    break

        return bends

    def _summarize_deviations(self, deviations: np.ndarray) -> str:
        """Create a text summary of deviations."""
        return json.dumps({
            "count": len(deviations),
            "min_mm": float(np.min(deviations)),
            "max_mm": float(np.max(deviations)),
            "mean_mm": float(np.mean(deviations)),
            "std_mm": float(np.std(deviations)),
            "percentile_95": float(np.percentile(deviations, 95)),
            "percentile_5": float(np.percentile(deviations, 5)),
        }, indent=2)

    def _parse_correlation_result(
        self,
        result: Optional[Dict],
        drawing_extraction: DrawingExtraction,
        point_cloud_features: PointCloudFeatures,
    ) -> Correlation2D3D:
        """Parse correlation AI result."""
        if not result:
            return Correlation2D3D(
                correlation_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                correlation_timestamp=datetime.now().isoformat(),
                overall_correlation_score=0.0,
            )

        mappings = []
        for m in result.get("bend_mappings", []):
            mappings.append(CorrelationMapping(
                mapping_id=f"M{len(mappings)+1}",
                drawing_element_id=str(m.get("drawing_bend_id", "")),
                drawing_element_type="bend",
                feature_3d_id=str(m.get("detected_bend_id", "")),
                feature_3d_type="bend",
                confidence=float(m.get("confidence", 0.5)),
                nominal_value=m.get("nominal_angle"),
                measured_value=m.get("measured_angle"),
                deviation=m.get("angle_deviation"),
                is_in_tolerance=m.get("is_in_tolerance"),
            ))

        critical_devs = []
        for cd in result.get("critical_deviations", []):
            loc = cd.get("location_3d", [0, 0, 0])
            critical_devs.append(CriticalDeviation(
                deviation_id=cd.get("id", f"CD{len(critical_devs)+1}"),
                feature_id=cd.get("affected_id", ""),
                feature_type=cd.get("type", "unknown"),
                location=tuple(loc) if isinstance(loc, list) else (0, 0, 0),
                deviation_mm=float(cd.get("deviation_mm", 0)),
                severity=cd.get("severity", "minor"),
                description=cd.get("description", ""),
            ))

        return Correlation2D3D(
            correlation_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            correlation_timestamp=datetime.now().isoformat(),
            mappings=mappings,
            critical_deviations=critical_devs,
            unmatched_drawing_elements=result.get("unmatched_drawing_elements", []),
            unmatched_3d_features=result.get("unmatched_3d_features", []),
            overall_correlation_score=float(result.get("overall_correlation_score", 0.7)),
            model_used=self._models_used[-1] if self._models_used else "",
        )

    def _find_nominal_angle(
        self,
        bend: BendFeature,
        drawing_extraction: Optional[DrawingExtraction],
    ) -> Optional[float]:
        """Find nominal angle for a bend from drawing."""
        if not drawing_extraction or not drawing_extraction.bends:
            return None

        # Match by sequence/position
        if bend.bend_id <= len(drawing_extraction.bends):
            return drawing_extraction.bends[bend.bend_id - 1].angle_nominal

        return None

    def _analyze_single_bend(
        self,
        bend: BendFeature,
        nominal_angle: Optional[float],
        deviations: np.ndarray,
        tolerance: float,
    ) -> BendRegionAnalysis:
        """Analyze a single bend region."""
        # Get deviations in bend region
        if bend.region_point_indices:
            bend_devs = deviations[bend.region_point_indices]
            apex_dev = float(np.mean(bend_devs))
        else:
            apex_dev = 0.0

        # Compute angle deviation
        angle_dev = 0.0
        springback = 0.0
        overbend = 0.0

        if nominal_angle is not None:
            angle_dev = bend.angle_degrees - nominal_angle
            if angle_dev < 0:
                springback = min(1.0, abs(angle_dev) / 5.0)
            else:
                overbend = min(1.0, angle_dev / 5.0)

        # Determine status
        status = "pass"
        if nominal_angle:
            if abs(angle_dev) > tolerance * 2:
                status = "fail"
            elif abs(angle_dev) > tolerance:
                status = "warning"

        return BendRegionAnalysis(
            bend=bend,
            nominal_angle=nominal_angle,
            angle_deviation_deg=angle_dev,
            springback_indicator=springback,
            over_bend_indicator=overbend,
            apex_deviation_mm=apex_dev,
            radius_deviation_mm=0.0,  # Would need nominal radius
            twist_angle_deg=0.0,
            status=status,
            recommendations=[],
            root_causes=[],
        )

    def _parse_root_cause_result(
        self,
        result: Optional[Dict],
        bend_results: List[BendRegionAnalysis],
    ) -> List[EnhancedRootCause]:
        """Parse root cause analysis result."""
        if not result:
            return []

        root_causes = []
        for rc in result.get("root_causes", []):
            root_causes.append(EnhancedRootCause(
                cause_id=rc.get("id", f"RC{len(root_causes)+1}"),
                category=rc.get("category", "process"),
                description=rc.get("description", ""),
                severity=rc.get("severity", "minor"),
                confidence=float(rc.get("confidence", 0.5)),
                affected_features=rc.get("affected_features", []),
                affected_bends=rc.get("affected_bends", []),
                evidence=rc.get("evidence", []),
                recommendations=[],
                priority=len(root_causes) + 1,
            ))

        # Add recommendations
        for rec in result.get("recommendations", []):
            cause_ids = rec.get("addresses_causes", [])
            for cause in root_causes:
                if cause.cause_id in cause_ids:
                    cause.recommendations.append(rec.get("action", ""))
                    cause.estimated_impact = rec.get("expected_improvement")

        return root_causes

    def _enhance_bend_results_with_causes(
        self,
        bend_results: List[BendRegionAnalysis],
        root_causes: List[EnhancedRootCause],
        ai_result: Optional[Dict],
    ) -> List[BendRegionAnalysis]:
        """Add root cause info to bend results."""
        if not ai_result:
            return bend_results

        bend_analysis = ai_result.get("bend_specific_analysis", [])

        for bend_result in bend_results:
            # Find matching analysis
            for ba in bend_analysis:
                if ba.get("bend_id") == bend_result.bend.bend_id:
                    bend_result.root_causes = ba.get("contributing_factors", [])

                    # Get recommendations for this bend
                    for rec in ai_result.get("recommendations", []):
                        if bend_result.bend.bend_id in rec.get("affected_bends", []):
                            bend_result.recommendations.append(rec.get("action", ""))
                    break

        return bend_results

    def _update_stage_status(
        self,
        stage: PipelineStage,
        status: str,
        model_used: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """Update pipeline stage status."""
        if not self._pipeline_status:
            return

        stage_status = StageStatus(
            stage=stage,
            status=status,
            model_used=model_used or (self._models_used[-1] if self._models_used else None),
            fallback_used=len(self._fallbacks_triggered) > 0,
            start_time=datetime.now().isoformat() if status == "running" else None,
            end_time=datetime.now().isoformat() if status in ["completed", "failed", "skipped"] else None,
            duration_ms=duration_ms,
            error_message=error_message,
        )

        self._pipeline_status.stages[stage.value] = stage_status
        self._pipeline_status.current_stage = stage

        # Update progress
        stage_weights = {
            PipelineStage.DRAWING_ANALYSIS: 25,
            PipelineStage.FEATURE_DETECTION: 25,
            PipelineStage.CORRELATION_2D_3D: 25,
            PipelineStage.ROOT_CAUSE_ANALYSIS: 25,
        }

        progress = 0
        for s, weight in stage_weights.items():
            stage_data = self._pipeline_status.stages.get(s.value)
            if stage_data and stage_data.status in ["completed", "skipped"]:
                progress += weight
            elif stage_data and stage_data.status == "running":
                progress += weight * 0.5

        self._pipeline_status.overall_progress = progress
