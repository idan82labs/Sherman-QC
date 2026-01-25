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
import time
import threading
from io import BytesIO
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
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


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for AI API calls.

    Prevents cascading failures by tracking consecutive failures and
    temporarily blocking calls to a failing service.
    """
    failure_threshold: int = 5  # Open circuit after this many failures
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    success_threshold: int = 2  # Successes needed to close from half-open

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("Circuit breaker entering half-open state")
            return self._state

    def record_success(self):
        """Record a successful API call"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker closed - service recovered")
            else:
                self._failure_count = 0

    def record_failure(self):
        """Record a failed API call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Immediate open on failure in half-open state
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker opened from half-open state")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures"
                )

    def is_call_permitted(self) -> bool:
        """Check if a call should be allowed through"""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            return True  # Allow test call
        else:  # OPEN
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring"""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "recovery_timeout": self.recovery_timeout
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls"""
    pass


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

    # Hallucination detection results
    validation_warnings: List[str] = field(default_factory=list)
    hallucination_detected: bool = False

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


class Model3DSnapshotRenderer:
    """
    Render 3D model snapshots with heatmap overlay using Open3D.
    Creates high-quality 3D rendered views for PDF reports.
    """

    def __init__(self):
        self._o3d = None
        # Set environment for headless rendering before Open3D import
        import os
        os.environ.setdefault('OPEN3D_CPU_RENDERING', 'true')

    @property
    def o3d(self):
        if self._o3d is None:
            import open3d as o3d
            self._o3d = o3d
        return self._o3d

    def _deviation_to_color(self, deviation: float, tolerance: float) -> list:
        """Convert deviation value to RGB color (green->yellow->red)."""
        # Normalize to 0-1 range based on tolerance
        abs_dev = abs(deviation)
        normalized = min(abs_dev / (tolerance * 2), 1.0)

        if normalized < 0.25:
            # Green (in tolerance)
            return [0.2, 0.8, 0.2]
        elif normalized < 0.5:
            # Green to Yellow
            t = (normalized - 0.25) * 4
            return [0.2 + t * 0.8, 0.8, 0.2 * (1 - t)]
        elif normalized < 0.75:
            # Yellow to Orange
            t = (normalized - 0.5) * 4
            return [1.0, 0.8 - t * 0.4, 0.0]
        else:
            # Orange to Red (out of tolerance)
            t = (normalized - 0.75) * 4
            return [1.0, 0.4 - t * 0.4, 0.0]

    def render_3d_snapshot(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        tolerance: float,
        reference_mesh_path: Optional[str] = None,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        show_mesh: bool = True
    ) -> bytes:
        """
        Render a 3D snapshot with heatmap colors.

        Args:
            points: Nx3 array of scan point coordinates
            deviations: N array of deviation values
            tolerance: Tolerance threshold for coloring
            reference_mesh_path: Optional path to reference mesh PLY file
            view: Camera view - "front", "side", "top", or "iso"
            width: Image width
            height: Image height
            show_mesh: Whether to show reference mesh (semi-transparent)

        Returns:
            PNG image as bytes
        """
        o3d = self.o3d

        # Create point cloud with deviation colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Assign colors based on deviations
        colors = np.array([self._deviation_to_color(d, tolerance) for d in deviations])
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)

        # Add point cloud
        vis.add_geometry(pcd)

        # Optionally add reference mesh (semi-transparent)
        if show_mesh and reference_mesh_path:
            try:
                mesh = o3d.io.read_triangle_mesh(reference_mesh_path)
                if mesh.has_triangles():
                    mesh.compute_vertex_normals()
                    # Make mesh semi-transparent gray
                    mesh.paint_uniform_color([0.7, 0.7, 0.7])
                    vis.add_geometry(mesh)
            except Exception as e:
                logger.warning(f"Could not load reference mesh: {e}")

        # Get render options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.95, 0.95, 0.95])  # Light gray background
        opt.point_size = 2.0

        # Set camera view
        ctr = vis.get_view_control()

        # Get bounding box for camera positioning
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = max(extent) * 1.5

        if view == "front":
            # Front view (looking at X-Z plane from +Y)
            ctr.set_front([0, -1, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_lookat(center)
            ctr.set_zoom(0.7)
        elif view == "side":
            # Side view (looking at Y-Z plane from +X)
            ctr.set_front([-1, 0, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_lookat(center)
            ctr.set_zoom(0.7)
        elif view == "top":
            # Top view (looking at X-Y plane from +Z)
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            ctr.set_lookat(center)
            ctr.set_zoom(0.7)
        else:  # iso
            # Isometric view
            ctr.set_front([-0.5, -0.5, -0.7])
            ctr.set_up([0, 0, 1])
            ctr.set_lookat(center)
            ctr.set_zoom(0.6)

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert to PNG bytes
        img_array = (np.asarray(img) * 255).astype(np.uint8)

        # Use PIL or matplotlib to save as PNG
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_array)
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            buf.seek(0)
            return buf.getvalue()
        except ImportError:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
            ax.imshow(img_array)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)
            return buf.getvalue()

    def render_multi_view_3d(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        tolerance: float,
        reference_mesh_path: Optional[str] = None,
        width: int = 1600,
        height: int = 1200
    ) -> bytes:
        """
        Render 4-view 3D snapshot (front, side, top, isometric).

        Returns a single image with 4 views arranged in a 2x2 grid.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        views = ["front", "side", "top", "iso"]
        titles = ["Front View", "Side View", "Top View", "Isometric View"]

        # Render individual views
        images = []
        for view in views:
            img_bytes = self.render_3d_snapshot(
                points, deviations, tolerance,
                reference_mesh_path=reference_mesh_path,
                view=view,
                width=width // 2,
                height=height // 2,
                show_mesh=True
            )
            img = Image.open(BytesIO(img_bytes))
            images.append(np.array(img))

        # Create 2x2 grid with colorbar on the right side (separate from views)
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('white')

        # Use GridSpec with proper spacing - give colorbar its own space
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, wspace=0.08, hspace=0.12,
                      left=0.02, right=0.88, top=0.92, bottom=0.05)

        axes = [
            fig.add_subplot(gs[0, 0]),  # Front View
            fig.add_subplot(gs[0, 1]),  # Side View
            fig.add_subplot(gs[1, 0]),  # Top View
            fig.add_subplot(gs[1, 1]),  # Isometric View
        ]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold', color='#333', pad=8)
            ax.axis('off')

        # Add colorbar in a separate axes on the right
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cmap = LinearSegmentedColormap.from_list('deviation', [
            (0.2, 0.8, 0.2),    # Green
            (1.0, 1.0, 0.0),    # Yellow
            (1.0, 0.4, 0.0),    # Orange
            (1.0, 0.0, 0.0),    # Red
        ])
        norm = Normalize(vmin=-tolerance * 2, vmax=tolerance * 2)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Create colorbar axes on the right side, vertically centered
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Deviation (mm)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        fig.suptitle(f'3D Model Analysis - Tolerance: +/-{tolerance}mm', fontsize=14, fontweight='bold', y=0.97)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue()


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decorator for retrying API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Don't retry on authentication/authorization errors
                    if any(code in error_str for code in ['401', '403', 'invalid_api_key', 'authentication']):
                        logger.error(f"API authentication error (not retrying): {e}")
                        raise

                    # Retry on rate limits, timeouts, and server errors
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"API call failed after {max_retries + 1} attempts: {e}")

            raise last_exception
        return wrapper
    return decorator


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
        model: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()
        self.renderer = DeviationHeatmapRenderer()

        # Circuit breaker for API resilience
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

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
        """Lazy-load API client with timeout configuration"""
        if self._client is None:
            # Default timeout: 60 seconds for response, 10 seconds for connection
            timeout_seconds = 60.0

            if self.provider == AIProvider.CLAUDE:
                import anthropic
                import httpx
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    timeout=httpx.Timeout(timeout_seconds, connect=10.0)
                )
            elif self.provider == AIProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            elif self.provider == AIProvider.OPENAI:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=timeout_seconds
                )
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

        # Check circuit breaker before calling AI
        if not self.circuit_breaker.is_call_permitted():
            logger.warning("Circuit breaker is open - skipping AI analysis")
            raise CircuitBreakerOpenError(
                f"AI service circuit breaker is open. "
                f"Status: {self.circuit_breaker.get_status()}"
            )

        # Call AI model with circuit breaker tracking
        try:
            if self.provider == AIProvider.CLAUDE:
                result = self._call_claude(prompt, heatmap_multi, heatmap_top, technical_drawing)
            elif self.provider == AIProvider.GEMINI:
                result = self._call_gemini(prompt, heatmap_multi, heatmap_top, technical_drawing)
            else:
                result = self._call_openai(prompt, heatmap_multi, heatmap_top, technical_drawing)

            self.circuit_breaker.record_success()
        except CircuitBreakerOpenError:
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

        # Validate AI response for hallucinations
        is_valid, warnings = self._validate_ai_response(result, stats)
        result.validation_warnings = warnings
        result.hallucination_detected = not is_valid

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
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_claude(
        self,
        prompt: str,
        heatmap_multi: bytes,
        heatmap_top: bytes,
        drawing: Optional[bytes]
    ) -> AIAnalysisResult:
        """Call Claude API with retry logic"""
        
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
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_gemini(
        self,
        prompt: str,
        heatmap_multi: bytes,
        heatmap_top: bytes,
        drawing: Optional[bytes]
    ) -> AIAnalysisResult:
        """Call Gemini API with retry logic"""
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
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_openai(
        self,
        prompt: str,
        heatmap_multi: bytes,
        heatmap_top: bytes,
        drawing: Optional[bytes]
    ) -> AIAnalysisResult:
        """Call OpenAI API with retry logic"""
        
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

    def _validate_ai_response(
        self,
        result: 'AIAnalysisResult',
        stats: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate AI response for hallucinations by checking consistency with input data.

        Returns:
            Tuple of (is_valid, list of warning messages)
        """
        warnings = []
        is_valid = True

        # Check quality score bounds
        if not 0 <= result.quality_score <= 100:
            warnings.append(f"Quality score {result.quality_score} outside valid range 0-100")
            is_valid = False

        # Check confidence bounds
        if not 0 <= result.confidence <= 1:
            warnings.append(f"Confidence {result.confidence} outside valid range 0-1")
            is_valid = False

        # Check verdict consistency with pass rate
        pass_rate = stats.get("pass_rate", 0)
        if result.verdict == "PASS" and pass_rate < 0.5:
            warnings.append(
                f"Verdict PASS inconsistent with low pass rate ({pass_rate*100:.1f}%)"
            )
        elif result.verdict == "FAIL" and pass_rate > 0.95:
            warnings.append(
                f"Verdict FAIL inconsistent with high pass rate ({pass_rate*100:.1f}%)"
            )

        # Check that reported defect deviations are plausible
        max_deviation = stats.get("max_deviation_mm", float('inf'))
        min_deviation = stats.get("min_deviation_mm", float('-inf'))
        for defect in result.defects_found:
            dev = defect.get("deviation_mm", 0)
            if isinstance(dev, (int, float)):
                if dev > max_deviation * 1.5 or dev < min_deviation * 1.5:
                    warnings.append(
                        f"Defect deviation {dev}mm exceeds actual data range "
                        f"[{min_deviation:.3f}, {max_deviation:.3f}]mm"
                    )

        # Check that quality score is roughly consistent with pass rate
        expected_min_score = max(0, (pass_rate - 0.2) * 100)
        expected_max_score = min(100, (pass_rate + 0.2) * 100)
        if not expected_min_score <= result.quality_score <= expected_max_score:
            warnings.append(
                f"Quality score {result.quality_score} may be inconsistent with "
                f"pass rate {pass_rate*100:.1f}% (expected ~{pass_rate*100:.0f})"
            )

        # Log warnings
        for warning in warnings:
            logger.warning(f"AI hallucination check: {warning}")

        return is_valid, warnings

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


    def get_circuit_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring"""
        return self.circuit_breaker.get_status()


# Global circuit breaker shared across analyzer instances for each provider
_circuit_breakers: Dict[AIProvider, CircuitBreaker] = {}


def get_circuit_breaker(provider: AIProvider) -> CircuitBreaker:
    """Get or create a shared circuit breaker for a provider"""
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker()
    return _circuit_breakers[provider]


# Convenience function for integration
def create_ai_analyzer(
    provider: str = "claude",
    api_key: Optional[str] = None,
    shared_circuit_breaker: bool = True
) -> MultimodalQCAnalyzer:
    """
    Create AI analyzer with specified provider.

    Args:
        provider: "claude", "gemini", or "openai"
        api_key: Optional API key (defaults to environment variable)
        shared_circuit_breaker: If True, use a shared circuit breaker per provider

    Returns:
        Configured MultimodalQCAnalyzer
    """
    provider_enum = AIProvider(provider.lower())
    circuit_breaker = get_circuit_breaker(provider_enum) if shared_circuit_breaker else None
    return MultimodalQCAnalyzer(
        provider=provider_enum,
        api_key=api_key,
        circuit_breaker=circuit_breaker
    )
