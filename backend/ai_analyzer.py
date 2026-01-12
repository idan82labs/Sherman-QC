"""
AI-Powered QC Analyzer
Uses SOTA multimodal vision models (Claude/Gemini/GPT) for real defect analysis.
Replaces rule-based heuristics with actual AI understanding.

Supports:
- Claude 4.5 Opus (Anthropic) - Best for precision inspection
- Gemini 3 Pro (Google) - Best for 3D spatial reasoning
- GPT-5.2 (OpenAI) - Best for abstract reasoning
"""

import os
import json
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from material_prompts import (
    build_material_specific_prompt,
    get_material_properties,
    get_root_cause_hints,
    MaterialCategory,
)

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class AIAnalysisResult:
    """Result from AI analysis"""
    verdict: str  # PASS, FAIL, WARNING
    quality_score: float  # 0-100
    confidence: float  # 0-1

    defects_found: List[Dict[str, Any]]
    root_causes: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]

    summary: str
    detailed_analysis: str

    # Material-specific analysis factors
    material_factors: Dict[str, Any] = None

    raw_response: str = ""
    model_used: str = ""
    tokens_used: int = 0

    def __post_init__(self):
        if self.material_factors is None:
            self.material_factors = {
                "springback_observed": False,
                "work_hardening_effects": False,
                "thermal_effects": False,
                "material_specific_issues": []
            }


class DeviationHeatmapRenderer:
    """Render 3D point cloud deviations as 2D heatmap images for AI analysis"""
    
    def __init__(self):
        self._plt = None
        self._o3d = None
    
    @property
    def plt(self):
        if self._plt is None:
            import matplotlib.pyplot as plt
            self._plt = plt
        return self._plt
    
    @property  
    def o3d(self):
        if self._o3d is None:
            import open3d as o3d
            self._o3d = o3d
        return self._o3d
    
    def render_heatmap(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        tolerance: float,
        view: str = "top"
    ) -> bytes:
        """
        Render deviation heatmap as PNG image.
        
        Args:
            points: Nx3 array of point coordinates
            deviations: N array of signed deviations
            tolerance: Tolerance threshold in mm
            view: "top", "front", "side", or "iso"
        
        Returns:
            PNG image as bytes
        """
        fig, ax = self.plt.subplots(1, 1, figsize=(10, 8))
        
        # Select view axes
        if view == "top":
            x_idx, y_idx = 0, 1  # X-Y plane
            xlabel, ylabel = "X (mm)", "Y (mm)"
        elif view == "front":
            x_idx, y_idx = 0, 2  # X-Z plane
            xlabel, ylabel = "X (mm)", "Z (mm)"
        elif view == "side":
            x_idx, y_idx = 1, 2  # Y-Z plane
            xlabel, ylabel = "Y (mm)", "Z (mm)"
        else:  # iso - use PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(points)
            x_idx, y_idx = None, None
            xlabel, ylabel = "PC1", "PC2"
        
        if x_idx is not None:
            x = points[:, x_idx]
            y = points[:, y_idx]
        else:
            x = projected[:, 0]
            y = projected[:, 1]
        
        # Normalize deviations for coloring
        vmax = max(abs(deviations.min()), abs(deviations.max()), tolerance * 2)
        
        scatter = ax.scatter(
            x, y, 
            c=deviations, 
            cmap='RdYlGn_r',  # Red=positive (outward), Green=negative (inward)
            vmin=-vmax, 
            vmax=vmax,
            s=1, 
            alpha=0.7
        )
        
        cbar = self.plt.colorbar(scatter, ax=ax, label='Deviation (mm)')
        
        # Add tolerance lines to colorbar
        cbar.ax.axhline(y=tolerance, color='black', linestyle='--', linewidth=1)
        cbar.ax.axhline(y=-tolerance, color='black', linestyle='--', linewidth=1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Deviation Heatmap ({view.capitalize()} View) - Tolerance: ±{tolerance}mm')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = (
            f"Points: {len(deviations):,}\n"
            f"Mean: {np.mean(deviations):.4f}mm\n"
            f"Max: {np.max(deviations):.4f}mm\n"
            f"Min: {np.min(deviations):.4f}mm\n"
            f"Std: {np.std(deviations):.4f}mm\n"
            f"Pass Rate: {np.mean(np.abs(deviations) <= tolerance)*100:.1f}%"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Save to bytes
        buf = BytesIO()
        self.plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        self.plt.close(fig)
        
        return buf.getvalue()
    
    def render_multi_view(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        tolerance: float
    ) -> bytes:
        """Render 4-view heatmap (top, front, side, iso)"""
        fig, axes = self.plt.subplots(2, 2, figsize=(16, 14))
        
        views = [("top", 0, 1), ("front", 0, 2), ("side", 1, 2), ("iso", None, None)]
        titles = ["Top View (X-Y)", "Front View (X-Z)", "Side View (Y-Z)", "Isometric"]
        
        vmax = max(abs(deviations.min()), abs(deviations.max()), tolerance * 2)
        
        for ax, (view, x_idx, y_idx), title in zip(axes.flat, views, titles):
            if x_idx is not None:
                x = points[:, x_idx]
                y = points[:, y_idx]
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                projected = pca.fit_transform(points)
                x = projected[:, 0]
                y = projected[:, 1]
            
            scatter = ax.scatter(
                x, y, c=deviations, cmap='RdYlGn_r',
                vmin=-vmax, vmax=vmax, s=0.5, alpha=0.7
            )
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Deviation (mm)', shrink=0.6)
        fig.suptitle(f'Multi-View Deviation Analysis - Tolerance: ±{tolerance}mm', fontsize=14)
        
        buf = BytesIO()
        self.plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        self.plt.close(fig)
        
        return buf.getvalue()


class MultimodalQCAnalyzer:
    """
    AI-powered QC analyzer using multimodal vision models.
    
    Analyzes:
    - Deviation heatmap images
    - Statistical data
    - Technical drawings (optional)
    
    Returns structured defect analysis with real confidence scores.
    """
    
    def __init__(
        self,
        provider: AIProvider = AIProvider.CLAUDE,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()
        self.renderer = DeviationHeatmapRenderer()
        
        # Initialize client
        self._client = None
    
    def _get_api_key(self) -> str:
        """Get API key from environment"""
        key_map = {
            AIProvider.CLAUDE: "ANTHROPIC_API_KEY",
            AIProvider.GEMINI: "GOOGLE_API_KEY",
            AIProvider.OPENAI: "OPENAI_API_KEY"
        }
        key = os.environ.get(key_map[self.provider])
        if not key:
            raise ValueError(f"Missing API key: {key_map[self.provider]}")
        return key
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            AIProvider.CLAUDE: "claude-opus-4-5-20251101",
            AIProvider.GEMINI: "gemini-3-pro",
            AIProvider.OPENAI: "gpt-5.2"
        }
        return defaults[self.provider]
    
    @property
    def client(self):
        """Lazy-load API client"""
        if self._client is None:
            if self.provider == AIProvider.CLAUDE:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            elif self.provider == AIProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            elif self.provider == AIProvider.OPENAI:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def analyze(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        tolerance: float,
        part_info: Dict[str, str],
        regions: Optional[List[Dict]] = None,
        technical_drawing: Optional[bytes] = None
    ) -> AIAnalysisResult:
        """
        Perform AI-powered QC analysis.
        
        Args:
            points: Nx3 point coordinates
            deviations: N signed deviation values
            tolerance: Tolerance in mm
            part_info: Dict with part_id, part_name, material
            regions: Optional regional analysis data
            technical_drawing: Optional PDF bytes of technical drawing
        
        Returns:
            AIAnalysisResult with AI-generated analysis
        """
        # Generate heatmap images
        heatmap_multi = self.renderer.render_multi_view(points, deviations, tolerance)
        heatmap_top = self.renderer.render_heatmap(points, deviations, tolerance, "top")
        
        # Calculate statistics
        abs_devs = np.abs(deviations)
        stats = {
            "total_points": len(deviations),
            "tolerance_mm": tolerance,
            "pass_rate": float(np.mean(abs_devs <= tolerance)),
            "mean_deviation_mm": float(np.mean(deviations)),
            "max_deviation_mm": float(np.max(deviations)),
            "min_deviation_mm": float(np.min(deviations)),
            "std_deviation_mm": float(np.std(deviations)),
            "points_exceeding_tolerance": int(np.sum(abs_devs > tolerance)),
            "max_positive_deviation": float(np.max(deviations)),
            "max_negative_deviation": float(np.min(deviations))
        }
        
        # Build prompt
        prompt = self._build_analysis_prompt(part_info, stats, regions)
        
        # Call AI model
        if self.provider == AIProvider.CLAUDE:
            result = self._call_claude(prompt, heatmap_multi, heatmap_top, technical_drawing)
        elif self.provider == AIProvider.GEMINI:
            result = self._call_gemini(prompt, heatmap_multi, heatmap_top, technical_drawing)
        else:
            result = self._call_openai(prompt, heatmap_multi, heatmap_top, technical_drawing)
        
        return result
    
    def _build_analysis_prompt(
        self,
        part_info: Dict[str, str],
        stats: Dict[str, Any],
        regions: Optional[List[Dict]] = None,
        process: str = "both"
    ) -> str:
        """Build the analysis prompt with material-specific guidance"""

        material = part_info.get('material', 'Unknown')
        tolerance = stats['tolerance_mm']

        # Get material properties for context
        mat_props = get_material_properties(material)

        prompt = f"""You are an expert manufacturing quality control engineer specializing in sheet metal forming and CNC machining. You are analyzing a LiDAR scan comparison between a manufactured part and its CAD reference model.

## Part Information
- Part ID: {part_info.get('part_id', 'Unknown')}
- Part Name: {part_info.get('part_name', 'Unknown')}
- Material: {material}
- Tolerance Specification: ±{tolerance}mm

## Deviation Statistics
- Total Points Measured: {stats['total_points']:,}
- Points Within Tolerance: {stats['total_points'] - stats['points_exceeding_tolerance']:,}
- Points Exceeding Tolerance: {stats['points_exceeding_tolerance']:,}
- Pass Rate: {stats['pass_rate']*100:.2f}%
- Mean Deviation: {stats['mean_deviation_mm']:.4f}mm
- Max Positive Deviation (outward): {stats['max_positive_deviation']:.4f}mm
- Max Negative Deviation (inward): {stats['min_deviation_mm']:.4f}mm
- Standard Deviation: {stats['std_deviation_mm']:.4f}mm
"""

        if regions:
            prompt += "\n## Regional Analysis\n"
            for r in regions:
                prompt += f"- **{r['name']}**: Mean={r['mean_deviation_mm']:.4f}mm, Max={r['max_deviation_mm']:.4f}mm, Pass Rate={r['pass_rate']:.1f}%\n"

        # Add material-specific analysis guidance
        material_guidance = build_material_specific_prompt(material, tolerance, process)
        prompt += material_guidance

        prompt += """
## Images Provided
1. Multi-view deviation heatmap (Top, Front, Side, Isometric views)
2. Top view detail

Color scale: Red = positive deviation (outward/bulging), Green = negative deviation (inward/depressed)

## Your Task
Analyze the deviation heatmaps and statistics, applying the material-specific knowledge above:

1. **Identify Defects**: Look for patterns indicating manufacturing issues:
   - Material-specific defects listed above for this material
   - Springback (uniform outward deviation in bent areas)
   - Edge curl/deformation
   - Thickness variation
   - Surface waviness
   - Tool marks or damage
   - Localized anomalies
   - Process-specific defects (galling, work hardening, thermal damage)

2. **Determine Root Causes**: For each defect pattern:
   - Consider the material properties (springback tendency, work hardening, etc.)
   - Identify the likely manufacturing root cause
   - Provide confidence level (0-100%) based on material behavior match

3. **Make Recommendations**: Provide specific, actionable recommendations:
   - Consider material-specific solutions (overbend compensation, tooling changes)
   - Account for material behavior in your suggestions
   - Prioritize based on defect severity and material constraints

4. **Verdict**: Determine if the part should PASS, FAIL, or receive a WARNING.
   - Consider achievable tolerances for this material
   - Factor in material-specific acceptance criteria

## Response Format
Respond ONLY with valid JSON in this exact structure:
```json
{
    "verdict": "PASS|FAIL|WARNING",
    "quality_score": 0-100,
    "confidence": 0.0-1.0,
    "material_factors": {
        "springback_observed": true|false,
        "work_hardening_effects": true|false,
        "thermal_effects": true|false,
        "material_specific_issues": ["list of material-specific issues found"]
    },
    "defects_found": [
        {
            "type": "defect type",
            "location": "where on part",
            "severity": "minor|moderate|severe",
            "deviation_mm": 0.0,
            "material_related": true|false,
            "description": "detailed description"
        }
    ],
    "root_causes": [
        {
            "issue": "what is wrong",
            "cause": "why it happened",
            "confidence": 0.0-1.0,
            "evidence": "what in the data supports this",
            "material_factor": "how material properties contribute"
        }
    ],
    "recommendations": [
        {
            "priority": "critical|high|medium|low",
            "action": "what to do",
            "expected_improvement": "expected result",
            "material_consideration": "material-specific notes"
        }
    ],
    "summary": "2-3 sentence executive summary including material considerations",
    "detailed_analysis": "Detailed technical analysis paragraph with material-specific insights"
}
```

Analyze the provided images now, applying your knowledge of {material} behavior, and respond with JSON only."""

        return prompt
    
    def _call_claude(
        self,
        prompt: str,
        heatmap_multi: bytes,
        heatmap_top: bytes,
        drawing: Optional[bytes]
    ) -> AIAnalysisResult:
        """Call Claude API"""
        
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(heatmap_multi).decode()
                }
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(heatmap_top).decode()
                }
            }
        ]
        
        if drawing:
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(drawing).decode()
                }
            })
        
        content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}]
        )
        
        raw_text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return self._parse_response(raw_text, self.model, tokens)
    
    def _call_gemini(
        self,
        prompt: str,
        heatmap_multi: bytes,
        heatmap_top: bytes,
        drawing: Optional[bytes]
    ) -> AIAnalysisResult:
        """Call Gemini API"""
        from PIL import Image
        
        img_multi = Image.open(BytesIO(heatmap_multi))
        img_top = Image.open(BytesIO(heatmap_top))
        
        content = [img_multi, img_top]
        
        if drawing:
            # Gemini handles PDFs differently
            content.append({"mime_type": "application/pdf", "data": drawing})
        
        content.append(prompt)
        
        response = self.client.generate_content(content)
        
        return self._parse_response(response.text, self.model, 0)
    
    def _call_openai(
        self,
        prompt: str,
        heatmap_multi: bytes,
        heatmap_top: bytes,
        drawing: Optional[bytes]
    ) -> AIAnalysisResult:
        """Call OpenAI API"""
        
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(heatmap_multi).decode()}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(heatmap_top).decode()}"
                }
            },
            {"type": "text", "text": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096
        )
        
        raw_text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        return self._parse_response(raw_text, self.model, tokens)
    
    def _parse_response(self, raw_text: str, model: str, tokens: int) -> AIAnalysisResult:
        """Parse AI response into structured result"""

        # Extract JSON from response
        try:
            # Find JSON block
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                json_str = raw_text.split("```")[1].split("```")[0]
            else:
                json_str = raw_text

            data = json.loads(json_str.strip())

            # Extract material factors with defaults
            material_factors = data.get("material_factors", {})
            if not material_factors:
                material_factors = {
                    "springback_observed": False,
                    "work_hardening_effects": False,
                    "thermal_effects": False,
                    "material_specific_issues": []
                }

            return AIAnalysisResult(
                verdict=data.get("verdict", "FAIL"),
                quality_score=float(data.get("quality_score", 0)),
                confidence=float(data.get("confidence", 0.5)),
                defects_found=data.get("defects_found", []),
                root_causes=data.get("root_causes", []),
                recommendations=data.get("recommendations", []),
                summary=data.get("summary", "Analysis failed to parse"),
                detailed_analysis=data.get("detailed_analysis", ""),
                material_factors=material_factors,
                raw_response=raw_text,
                model_used=model,
                tokens_used=tokens
            )

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            return AIAnalysisResult(
                verdict="FAIL",
                quality_score=0,
                confidence=0,
                defects_found=[],
                root_causes=[{"issue": "Analysis parsing failed", "cause": str(e), "confidence": 0}],
                recommendations=[],
                summary=f"Failed to parse AI response: {e}",
                detailed_analysis=raw_text[:500],
                material_factors={
                    "springback_observed": False,
                    "work_hardening_effects": False,
                    "thermal_effects": False,
                    "material_specific_issues": []
                },
                raw_response=raw_text,
                model_used=model,
                tokens_used=tokens
            )


# Convenience function for integration
def create_ai_analyzer(
    provider: str = "claude",
    api_key: Optional[str] = None
) -> MultimodalQCAnalyzer:
    """
    Create AI analyzer with specified provider.
    
    Args:
        provider: "claude", "gemini", or "openai"
        api_key: Optional API key (defaults to environment variable)
    
    Returns:
        Configured MultimodalQCAnalyzer
    """
    provider_enum = AIProvider(provider.lower())
    return MultimodalQCAnalyzer(provider=provider_enum, api_key=api_key)
