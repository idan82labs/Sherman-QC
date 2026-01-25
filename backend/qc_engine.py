"""
Scan QC Engine - Core Analysis Module
Handles mesh loading, alignment, deviation computation, and AI analysis.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path
import json
import logging
import time
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced analysis modules
try:
    from bend_detector import BendDetector, BendDetectionResult, analyze_bend_deviations
    from multi_model import (
        MultiModelOrchestrator,
        EnhancedAnalysisResult,
        BendRegionAnalysis as EnhancedBendAnalysis,
    )
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced analysis modules not available: {e}")
    ENHANCED_ANALYSIS_AVAILABLE = False

# AI Integration flag - set via environment or auto-detect
AI_ENABLED = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")


class QCResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ProgressUpdate:
    """Progress update for UI"""
    stage: str
    progress: float  # 0-100
    message: str
    

@dataclass
class RegionConfig:
    """Configuration for regional analysis"""
    # Percentage thresholds for defining regions (0.0-1.0)
    top_surface_threshold: float = 0.25  # Points above center + threshold*(max-center)
    bottom_surface_threshold: float = 0.25  # Points below center - threshold*(center-min)
    edge_threshold: float = 0.30  # Threshold for front/rear edges
    side_threshold: float = 0.25  # Threshold for left/right sides
    center_threshold: float = 0.30  # Threshold for center region
    # Minimum points required for valid region analysis
    min_points: int = 50
    # Pass rate threshold for OK status
    pass_rate_threshold: float = 0.95

    @classmethod
    def default(cls) -> 'RegionConfig':
        """Return default configuration"""
        return cls()


@dataclass
class RegionAnalysis:
    """Analysis of a specific region"""
    name: str
    point_count: int
    mean_deviation: float
    max_deviation: float
    min_deviation: float
    std_deviation: float
    pass_rate: float
    status: str
    deviation_direction: str
    ai_interpretation: str = ""


@dataclass 
class RootCause:
    """Root cause analysis result"""
    issue: str
    likely_cause: str
    technical_explanation: str
    confidence: float
    recommendation: str
    priority: str


@dataclass
class QCReport:
    """Complete QC inspection report"""
    part_id: str
    part_name: str
    material: str
    tolerance: float
    timestamp: str

    # Overall result
    overall_result: QCResult = QCResult.PASS
    quality_score: float = 0.0
    confidence: float = 0.0

    # Statistics
    total_points: int = 0
    points_in_tolerance: int = 0
    points_out_of_tolerance: int = 0
    pass_rate: float = 0.0

    mean_deviation: float = 0.0
    max_deviation: float = 0.0
    min_deviation: float = 0.0
    std_deviation: float = 0.0

    # Alignment quality
    alignment_fitness: float = 0.0
    alignment_rmse: float = 0.0

    # Detailed analysis
    regions: List[RegionAnalysis] = field(default_factory=list)
    root_causes: List[RootCause] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)

    # AI summary
    ai_summary: str = ""
    ai_detailed_analysis: str = ""
    ai_model_used: str = ""
    ai_tokens_used: int = 0

    # Heatmap images (paths or URLs)
    heatmap_multi_view: str = ""
    heatmap_top: str = ""
    heatmap_front: str = ""
    heatmap_side: str = ""

    # 3D model snapshots (paths or URLs)
    snapshot_3d_multi_view: str = ""
    snapshot_3d_iso: str = ""
    snapshot_3d_front: str = ""
    snapshot_3d_side: str = ""
    snapshot_3d_top: str = ""

    # Drawing context
    drawing_context: str = ""

    # Enhanced analysis (bend detection + multi-model)
    enhanced_analysis: Optional[Dict] = None
    bend_results: List[Dict] = field(default_factory=list)
    bend_detection_result: Optional[Dict] = None
    correlation_2d_3d: Optional[Dict] = None
    pipeline_status: Optional[Dict] = None

    # Dimension analysis (XLSX-based comparison)
    dimension_analysis: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "material": self.material,
            "tolerance": self.tolerance,
            "timestamp": self.timestamp,
            "overall_result": self.overall_result.value,
            "quality_score": round(self.quality_score, 1),
            "confidence": round(self.confidence * 100, 1),
            "statistics": {
                "total_points": self.total_points,
                "points_in_tolerance": self.points_in_tolerance,
                "points_out_of_tolerance": self.points_out_of_tolerance,
                "pass_rate": round(self.pass_rate * 100, 2),
                "mean_deviation_mm": round(self.mean_deviation, 4),
                "max_deviation_mm": round(self.max_deviation, 4),
                "min_deviation_mm": round(self.min_deviation, 4),
                "std_deviation_mm": round(self.std_deviation, 4),
            },
            "alignment": {
                "fitness": round(self.alignment_fitness * 100, 2),
                "rmse_mm": round(self.alignment_rmse, 4),
            },
            "regions": [
                {
                    "name": r.name,
                    "point_count": r.point_count,
                    "mean_deviation_mm": round(r.mean_deviation, 4),
                    "max_deviation_mm": round(r.max_deviation, 4),
                    "pass_rate": round(r.pass_rate * 100, 2),
                    "status": r.status,
                    "direction": r.deviation_direction,
                    "ai_interpretation": r.ai_interpretation,
                }
                for r in self.regions
            ],
            "root_causes": [
                {
                    "issue": rc.issue,
                    "likely_cause": rc.likely_cause,
                    "explanation": rc.technical_explanation,
                    "confidence": round(rc.confidence * 100, 0),
                    "recommendation": rc.recommendation,
                    "priority": rc.priority,
                }
                for rc in self.root_causes
            ],
            "recommendations": self.recommendations,
            "ai_summary": self.ai_summary,
            "ai_detailed_analysis": self.ai_detailed_analysis,
            "ai_model_used": self.ai_model_used,
            "ai_tokens_used": self.ai_tokens_used,
            "heatmaps": {
                "multi_view": self.heatmap_multi_view,
                "top": self.heatmap_top,
                "front": self.heatmap_front,
                "side": self.heatmap_side,
            },
            "snapshots_3d": {
                "multi_view": self.snapshot_3d_multi_view,
                "iso": self.snapshot_3d_iso,
                "front": self.snapshot_3d_front,
                "side": self.snapshot_3d_side,
                "top": self.snapshot_3d_top,
            },
            "drawing_context": self.drawing_context,
            # Enhanced analysis results
            "enhanced_analysis": self.enhanced_analysis,
            "bend_results": self.bend_results,
            "bend_detection_result": self.bend_detection_result,
            "correlation_2d_3d": self.correlation_2d_3d,
            "pipeline_status": self.pipeline_status,
            "dimension_analysis": self.dimension_analysis,
        }


class ScanQCEngine:
    """Main QC analysis engine"""

    def __init__(
        self,
        progress_callback: Callable[[ProgressUpdate], None] = None,
        output_dir: Optional[str] = None,
        region_config: Optional[RegionConfig] = None
    ):
        self.progress_callback = progress_callback or (lambda x: None)
        self.reference_mesh = None
        self.scan_pcd = None
        self.aligned_scan = None
        self.deviations = None
        self.output_dir = output_dir  # For saving heatmaps
        self.region_config = region_config or RegionConfig.default()
        self._o3d = None
        self._trimesh = None
        self._heatmap_renderer = None
        self._snapshot_renderer = None

    @property
    def heatmap_renderer(self):
        """Lazy-load heatmap renderer"""
        if self._heatmap_renderer is None:
            try:
                from ai_analyzer import DeviationHeatmapRenderer
                self._heatmap_renderer = DeviationHeatmapRenderer()
            except ImportError:
                self._heatmap_renderer = None
        return self._heatmap_renderer

    @property
    def snapshot_renderer(self):
        """Lazy-load 3D snapshot renderer"""
        if self._snapshot_renderer is None:
            try:
                from ai_analyzer import Model3DSnapshotRenderer
                self._snapshot_renderer = Model3DSnapshotRenderer()
            except ImportError:
                self._snapshot_renderer = None
        return self._snapshot_renderer

    @property
    def bend_detector(self):
        """Lazy-load bend detector"""
        if not hasattr(self, '_bend_detector'):
            self._bend_detector = None
        if self._bend_detector is None and ENHANCED_ANALYSIS_AVAILABLE:
            try:
                self._bend_detector = BendDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize bend detector: {e}")
                self._bend_detector = None
        return self._bend_detector

    def detect_bends(self, points: np.ndarray = None, deviations: np.ndarray = None) -> Optional[Dict]:
        """
        Detect bends in the point cloud.

        Args:
            points: Optional point cloud (uses self.aligned_scan if not provided)
            deviations: Optional deviations array

        Returns:
            BendDetectionResult as dict, or None if detection fails
        """
        if self.bend_detector is None:
            logger.warning("Bend detector not available")
            return None

        if points is None:
            if self.aligned_scan is None:
                logger.warning("No point cloud available for bend detection")
                return None
            points = np.asarray(self.aligned_scan.points)

        if deviations is None:
            deviations = self.deviations

        try:
            self._update_progress("bend_detect", 88, "Detecting bends in point cloud...")
            result = self.bend_detector.detect_bends(points, deviations)
            logger.info(f"Detected {result.total_bends_detected} bends")
            return result.to_dict()
        except Exception as e:
            logger.error(f"Bend detection failed: {e}")
            return None

    def _detect_bends_by_surface_clustering(
        self,
        points: np.ndarray,
        n_clusters: int = 8,
        min_bend_angle: float = 20.0,
        max_bend_angle: float = 175.0,
    ) -> List[Dict]:
        """
        Fallback bend detection using surface normal clustering.

        This method uses KMeans clustering on surface normals to identify
        distinct planar surfaces, then computes angles between them.
        More robust than curvature-based detection for some geometries.
        """
        try:
            import open3d as o3d
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("Open3D or sklearn not available for surface clustering")
            return []

        try:
            # Sample points for speed
            if len(points) > 25000:
                idx = np.random.choice(len(points), 25000, replace=False)
                sample_points = points[idx]
            else:
                sample_points = points

            # Create point cloud and estimate normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sample_points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
            )
            normals = np.asarray(pcd.normals)

            # Cluster normals to find distinct surfaces
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(normals)

            # Get cluster info
            cluster_info = []
            for i in range(n_clusters):
                mask = labels == i
                if mask.sum() < 30:
                    continue
                cluster_normals = normals[mask]
                cluster_pts = sample_points[mask]
                avg_n = cluster_normals.mean(axis=0)
                avg_n = avg_n / (np.linalg.norm(avg_n) + 1e-10)
                center = cluster_pts.mean(axis=0)
                cluster_info.append({
                    'id': i,
                    'count': int(mask.sum()),
                    'normal': avg_n,
                    'center': center
                })

            # Calculate angles between all surface pairs
            bend_angles = []
            for i in range(len(cluster_info)):
                for j in range(i + 1, len(cluster_info)):
                    n1, n2 = cluster_info[i]['normal'], cluster_info[j]['normal']
                    dot = np.clip(np.dot(n1, n2), -1, 1)
                    angle_between_normals = np.degrees(np.arccos(abs(dot)))
                    bend_angle = 180 - angle_between_normals

                    if min_bend_angle < bend_angle < max_bend_angle:
                        # Midpoint between surface centers as bend apex
                        center = (cluster_info[i]['center'] + cluster_info[j]['center']) / 2
                        confidence = min(cluster_info[i]['count'], cluster_info[j]['count']) / 1000.0

                        bend_angles.append({
                            'bend_id': len(bend_angles) + 1,
                            'angle_degrees': float(bend_angle),
                            'radius_mm': 3.0,  # Default estimate
                            'bend_apex': center.tolist(),
                            'detection_confidence': float(min(1.0, confidence)),
                            'surfaces': (cluster_info[i]['id'], cluster_info[j]['id']),
                        })

            # Sort by confidence and return top candidates
            bend_angles.sort(key=lambda x: x['detection_confidence'], reverse=True)

            # Remove duplicates (similar angles from different surface pairs)
            unique_bends = []
            for ba in bend_angles:
                is_duplicate = False
                for ub in unique_bends:
                    if abs(ba['angle_degrees'] - ub['angle_degrees']) < 5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_bends.append(ba)

            logger.info(f"Surface clustering found {len(unique_bends)} unique bend angles")
            return unique_bends[:10]  # Return top 10

        except Exception as e:
            logger.error(f"Surface clustering bend detection failed: {e}")
            return []

    def run_dimension_analysis(
        self,
        xlsx_path: str,
        part_id: str = "",
        part_name: str = "",
    ) -> Optional[Dict]:
        """
        Run dimension analysis comparing XLSX specs to CAD and scan measurements.

        This implements the bend matching pipeline:
        1. Parse XLSX dimension specs
        2. Detect bends in CAD
        3. Match XLSX bends to CAD bends by angle proximity
        4. Measure bends at matched positions in aligned scan
        5. Generate comparison report

        Args:
            xlsx_path: Path to XLSX dimension specification file
            part_id: Part identifier
            part_name: Part name

        Returns:
            Dimension analysis report as dict, or None on failure
        """
        try:
            from dimension_parser import parse_dimension_file
            from bend_matcher import BendMatcher
            from dimension_report import DimensionReportGenerator

            self._update_progress("dimension_analysis", 85, "Parsing dimension specifications...")

            # Step 1: Parse XLSX
            xlsx_result = parse_dimension_file(xlsx_path)
            if not xlsx_result.success:
                logger.error(f"Failed to parse XLSX: {xlsx_result.error}")
                return None

            logger.info(f"Parsed {len(xlsx_result.dimensions)} dimensions, "
                       f"{len(xlsx_result.bend_angles)} bend angles")

            # Step 2: Detect bends in scan (CAD detection often fails due to sparse points)
            self._update_progress("dimension_analysis", 87, "Detecting bends...")
            cad_bends = []
            bends_from_scan = False

            if self.aligned_scan is not None:
                aligned_points = np.asarray(self.aligned_scan.points)

                # Try primary bend detector first
                if ENHANCED_ANALYSIS_AVAILABLE:
                    detector = BendDetector(
                        normal_radius=10.0,
                        curvature_radius=5.0,
                        min_surface_points=200,
                    )
                    # Subsample for speed
                    if len(aligned_points) > 100000:
                        idx = np.random.choice(len(aligned_points), 100000, replace=False)
                        sample_points = aligned_points[idx]
                    else:
                        sample_points = aligned_points

                    bend_result = detector.detect_bends(sample_points)
                    cad_bends = bend_result.bends if bend_result else []

                    # Convert to dict format for matcher
                    cad_bends = [
                        {
                            'bend_id': b.bend_id,
                            'angle_degrees': b.angle_degrees,
                            'radius_mm': b.radius_mm,
                            'bend_apex': list(b.bend_apex),
                            'detection_confidence': b.detection_confidence,
                        }
                        for b in cad_bends
                    ]

                # If primary detector found no bends, use fallback surface-normal clustering
                # Mark bends as detected from scan (so we use detected angle directly, not re-measure)
                bends_from_scan = False
                if not cad_bends:
                    logger.info("Primary bend detector found no bends, trying surface-normal clustering...")
                    cad_bends = self._detect_bends_by_surface_clustering(aligned_points)
                    bends_from_scan = True  # These are scan measurements, not CAD

                logger.info(f"Detected {len(cad_bends)} bends in scan")

            # Step 3: Match XLSX to detected bends
            self._update_progress("dimension_analysis", 90, "Matching dimensions to features...")
            matcher = BendMatcher(angle_match_threshold=15.0)
            match_result = matcher.match_xlsx_to_cad(xlsx_result, cad_bends)

            # Step 4: Handle scan measurement
            # If bends were detected from scan data (surface clustering), use detected angles directly
            # Otherwise, measure bends at matched positions
            if bends_from_scan and match_result.success:
                # Surface clustering already measured from scan - use CAD angle as scan angle
                self._update_progress("dimension_analysis", 93, "Using detected bend angles...")
                for match in match_result.matches:
                    if match.cad_matched and match.cad_angle is not None:
                        match.scan_angle = match.cad_angle
                        match.scan_measured = True
                        match.scan_confidence = match.cad_confidence
                        match.compute_deviations()
                        logger.info(f"  Dim {match.dim_id}: Using detected angle {match.scan_angle:.1f}° "
                                   f"(deviation: {match.scan_deviation:+.2f}°, status: {match.status})")
                match_result.compute_summary()
            elif self.aligned_scan is not None and match_result.success:
                self._update_progress("dimension_analysis", 93, "Measuring bend angles...")
                aligned_points = np.asarray(self.aligned_scan.points)
                match_result = matcher.measure_scan_bends(match_result, aligned_points)

            # Step 5: Generate report
            self._update_progress("dimension_analysis", 95, "Generating dimension report...")
            generator = DimensionReportGenerator()
            report = generator.generate_report(
                xlsx_result=xlsx_result,
                bend_match_result=match_result,
                part_id=part_id,
                part_name=part_name,
            )

            logger.info(f"Dimension analysis complete: {report.pass_count}/{report.measured_count} passed")

            return report.to_dict()

        except ImportError as e:
            logger.warning(f"Dimension analysis modules not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Dimension analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_enhanced_analysis(
        self,
        job_id: str,
        tolerance: float,
        drawing_path: Optional[str] = None,
        part_info: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Run the complete enhanced multi-model analysis pipeline.

        This includes:
        1. Bend detection (local algorithm)
        2. Drawing analysis (Gemini 3 Pro)
        3. Feature detection validation (Claude Opus 4.5)
        4. 2D-3D correlation (GPT-5.2)
        5. Root cause analysis (Claude Opus 4.5)

        Args:
            job_id: Unique job identifier
            tolerance: Deviation tolerance in mm
            drawing_path: Path to technical drawing (PDF)
            part_info: Part metadata

        Returns:
            EnhancedAnalysisResult as dict, or None if not available
        """
        if not ENHANCED_ANALYSIS_AVAILABLE:
            logger.warning("Enhanced analysis modules not available")
            return None

        if self.aligned_scan is None or self.deviations is None:
            logger.warning("No scan data available for enhanced analysis")
            return None

        try:
            self._update_progress("enhanced", 88, "Running enhanced multi-model analysis...")

            points = np.asarray(self.aligned_scan.points)
            deviations = self.deviations

            # Step 1: Run bend detection
            self._update_progress("enhanced", 89, "Detecting bends...")
            bend_result = None
            if self.bend_detector:
                bend_result_obj = self.bend_detector.detect_bends(points, deviations)
                bend_result = bend_result_obj

            # Step 2: Prepare regional analysis for context
            regional_data = {}
            if hasattr(self, 'region_config'):
                # Already computed in analyze_regions
                pass

            # Step 3: Run multi-model orchestrator
            self._update_progress("enhanced", 90, "Running multi-model AI analysis...")
            orchestrator = MultiModelOrchestrator()

            result = await orchestrator.analyze(
                job_id=job_id,
                points=points,
                deviations=deviations,
                bend_detection_result=bend_result,
                drawing_path=drawing_path,
                part_info=part_info,
                regional_analysis=regional_data,
                tolerance=tolerance,
            )

            self._update_progress("enhanced", 95, "Enhanced analysis complete")

            return result.to_dict()

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_enhanced_analysis_sync(
        self,
        job_id: str,
        tolerance: float,
        drawing_path: Optional[str] = None,
        part_info: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Synchronous wrapper for run_enhanced_analysis."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.run_enhanced_analysis(job_id, tolerance, drawing_path, part_info)
        )

    def _update_progress(self, stage: str, progress: float, message: str):
        """Send progress update"""
        self.progress_callback(ProgressUpdate(stage, progress, message))
        
    @property
    def o3d(self):
        if self._o3d is None:
            import open3d as o3d
            self._o3d = o3d
        return self._o3d
    
    @property
    def trimesh(self):
        if self._trimesh is None:
            import trimesh
            self._trimesh = trimesh
        return self._trimesh
    
    def load_reference(self, filepath: str) -> bool:
        """Load reference model (STL, OBJ, PLY, STEP, or IGES)"""
        self._update_progress("load", 5, f"Loading reference model: {Path(filepath).name}")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Reference not found: {filepath}")

        ext = filepath.suffix.lower()

        # Handle CAD formats (STEP/IGES) - convert to mesh first
        if ext in ['.step', '.stp', '.iges', '.igs']:
            self._update_progress("load", 8, f"Converting CAD file to mesh...")
            from cad_import import import_cad_file, is_cad_import_available

            if not is_cad_import_available():
                raise ImportError(
                    "PythonOCC is required for STEP/IGES import. "
                    "Install with: conda install -c conda-forge pythonocc-core"
                )

            result = import_cad_file(str(filepath), deflection=0.05)
            if not result.success:
                raise ValueError(f"Failed to import CAD file: {result.error}")

            # Convert CADMesh to Open3D mesh
            self.reference_mesh = self.o3d.geometry.TriangleMesh()
            self.reference_mesh.vertices = self.o3d.utility.Vector3dVector(result.mesh.vertices)
            self.reference_mesh.triangles = self.o3d.utility.Vector3iVector(result.mesh.faces)
            self.reference_mesh.compute_vertex_normals()

            self._update_progress("load", 15,
                f"Converted CAD: {result.num_faces} faces → {len(self.reference_mesh.vertices)} vertices")
        else:
            # Direct mesh formats (STL, OBJ, PLY)
            self.reference_mesh = self.o3d.io.read_triangle_mesh(str(filepath))
            if not self.reference_mesh.has_vertex_normals():
                self.reference_mesh.compute_vertex_normals()

            self._update_progress("load", 15, f"Loaded {len(self.reference_mesh.vertices)} vertices")

        return True
    
    def load_scan(self, filepath: str) -> bool:
        """Load scan file (STL, PLY, or converted STEP)"""
        self._update_progress("load", 20, f"Loading scan: {Path(filepath).name}")
        
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        
        if ext in ['.ply', '.pcd']:
            self.scan_pcd = self.o3d.io.read_point_cloud(str(filepath))
        elif ext in ['.stl', '.obj']:
            mesh = self.o3d.io.read_triangle_mesh(str(filepath))
            self.scan_pcd = mesh.sample_points_uniformly(number_of_points=200000)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        self._update_progress("load", 30, f"Loaded {len(self.scan_pcd.points)} points")
        return True
    
    def preprocess(self, voxel_size: float = 0.3):
        """Preprocess scan data"""
        self._update_progress("preprocess", 35, "Downsampling point cloud...")
        
        original = len(self.scan_pcd.points)
        self.scan_pcd = self.scan_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        self._update_progress("preprocess", 40, "Removing outliers...")
        self.scan_pcd, _ = self.scan_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        self._update_progress("preprocess", 45, f"Preprocessed: {original} → {len(self.scan_pcd.points)} points")
        
        if not self.scan_pcd.has_normals():
            self.scan_pcd.estimate_normals()
    
    def _compute_pca_rotation(self, points: np.ndarray) -> np.ndarray:
        """Compute PCA-based rotation matrix to align principal axes with coordinate axes.

        Returns rotation matrix that aligns the point cloud's principal axes
        with the coordinate system (largest variance along X, smallest along Z).
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # Compute covariance matrix and eigenvectors
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (largest first) - eigenvectors are columns
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        return eigenvectors.T  # Rotation matrix (3x3)

    def _try_alignment_with_rotation(self, scan_pcd, ref_pcd, rotation_matrix, scale_factors):
        """Try ICP alignment with a given initial rotation and return fitness score."""
        import copy

        scan_test = copy.deepcopy(scan_pcd)

        # Apply rotation
        points = np.asarray(scan_test.points)
        rotated = points @ rotation_matrix.T
        scan_test.points = self.o3d.utility.Vector3dVector(rotated)
        scan_test.estimate_normals()

        # Run quick ICP
        reg = self.o3d.pipelines.registration.registration_icp(
            scan_test, ref_pcd, 20.0, np.eye(4),
            self.o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )

        return reg.fitness, reg.transformation, rotation_matrix

    def align(self, auto_scale: bool = True) -> tuple:
        """Align scan to reference using PCA pre-alignment, auto-scaling, and ICP.

        This method uses a professional-grade alignment pipeline:
        1. PCA-based pre-alignment to handle arbitrary initial orientations
        2. Auto-scaling to handle different unit systems or scale differences
        3. Multi-hypothesis rotation search to find best initial orientation
        4. Coarse-to-fine ICP for precise final alignment

        Args:
            auto_scale: Automatically scale scan to match reference size (default True)
        """
        import copy

        self._update_progress("align", 50, "Converting reference to point cloud...")
        ref_pcd = self.reference_mesh.sample_points_uniformly(
            number_of_points=max(100000, len(self.scan_pcd.points))
        )
        ref_pcd.estimate_normals()

        # Store scale factor for reporting
        self.scale_factor = 1.0
        self.scale_factors = np.array([1.0, 1.0, 1.0])

        # Center both point clouds at origin
        self._update_progress("align", 52, "Centering point clouds...")
        scan_center = self.scan_pcd.get_center()
        ref_center = ref_pcd.get_center()

        scan_working = copy.deepcopy(self.scan_pcd)
        scan_working.translate(-scan_center)
        ref_centered = copy.deepcopy(ref_pcd)
        ref_centered.translate(-ref_center)

        # === PCA-BASED PRE-ALIGNMENT ===
        self._update_progress("align", 54, "Computing PCA alignment...")

        scan_points = np.asarray(scan_working.points)
        ref_points = np.asarray(ref_centered.points)

        # Get PCA rotation matrices
        scan_pca = self._compute_pca_rotation(scan_points)
        ref_pca = self._compute_pca_rotation(ref_points)

        # Rotation to align scan's principal axes with reference's principal axes
        # R_align = R_ref @ R_scan.T (from scan PCA space to reference PCA space)
        pca_rotation = ref_pca.T @ scan_pca

        # Apply PCA rotation to scan
        scan_rotated = scan_points @ pca_rotation.T
        scan_working.points = self.o3d.utility.Vector3dVector(scan_rotated)

        # Auto-scale AFTER PCA rotation (now axes are aligned, so per-axis scaling makes sense)
        if auto_scale:
            self._update_progress("align", 56, "Computing scale factors...")
            scan_bbox = scan_working.get_axis_aligned_bounding_box()
            ref_bbox = ref_centered.get_axis_aligned_bounding_box()
            scan_size = np.asarray(scan_bbox.max_bound) - np.asarray(scan_bbox.min_bound)
            ref_size = np.asarray(ref_bbox.max_bound) - np.asarray(ref_bbox.min_bound)

            # Uniform scale to preserve shape (use geometric mean)
            # Per-axis can distort the shape, uniform is safer
            scale_ratio = ref_size / np.maximum(scan_size, 1e-6)
            self.scale_factor = float(np.cbrt(np.prod(scale_ratio)))  # Geometric mean
            self.scale_factors = np.array([self.scale_factor, self.scale_factor, self.scale_factor])

            logger.info(f"Auto-scaling scan by uniform factor {self.scale_factor:.4f}")

            # Apply uniform scaling
            points = np.asarray(scan_working.points)
            points_scaled = points * self.scale_factor
            scan_working.points = self.o3d.utility.Vector3dVector(points_scaled)

        # === TRY MULTIPLE ROTATIONS ===
        # PCA can have 180-degree ambiguity, so try flipping axes
        self._update_progress("align", 58, "Testing rotation hypotheses...")

        # Generate rotation variants (flips around each axis)
        flip_matrices = [
            np.eye(3),  # No flip
            np.diag([-1, -1, 1]),   # Flip X and Y
            np.diag([-1, 1, -1]),   # Flip X and Z
            np.diag([1, -1, -1]),   # Flip Y and Z
            np.diag([-1, 1, 1]),    # Flip X only
            np.diag([1, -1, 1]),    # Flip Y only
            np.diag([1, 1, -1]),    # Flip Z only
            np.diag([-1, -1, -1]),  # Flip all
        ]

        best_fitness = 0
        best_icp_transform = np.eye(4)
        best_pre_rotation = np.eye(3)

        scan_base = copy.deepcopy(scan_working)

        for flip in flip_matrices:
            combined_rotation = flip @ np.eye(3)  # Apply flip after PCA alignment
            fitness, icp_transform, _ = self._try_alignment_with_rotation(
                scan_base, ref_centered, combined_rotation, self.scale_factors
            )
            if fitness > best_fitness:
                best_fitness = fitness
                best_icp_transform = icp_transform
                best_pre_rotation = combined_rotation

        logger.info(f"Best pre-alignment fitness: {best_fitness:.2%}")

        # Apply best pre-rotation
        points = np.asarray(scan_working.points)
        rotated = points @ best_pre_rotation.T
        scan_working.points = self.o3d.utility.Vector3dVector(rotated)
        scan_working.estimate_normals()

        # === FINE ICP ===
        self._update_progress("align", 65, "Running fine alignment (ICP)...")
        reg_fine = self.o3d.pipelines.registration.registration_icp(
            scan_working, ref_centered, 5.0, best_icp_transform,
            self.o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )

        # === APPLY FULL TRANSFORMATION TO ORIGINAL SCAN ===
        self._update_progress("align", 68, "Applying transformation...")

        # Build combined transformation pipeline
        self.aligned_scan = copy.deepcopy(self.scan_pcd)

        # Step 1: Center at origin
        self.aligned_scan.translate(-scan_center)

        # Step 2: Apply PCA rotation
        points = np.asarray(self.aligned_scan.points)
        points = points @ pca_rotation.T
        self.aligned_scan.points = self.o3d.utility.Vector3dVector(points)

        # Step 3: Apply uniform scaling
        if auto_scale:
            points = np.asarray(self.aligned_scan.points)
            points = points * self.scale_factor
            self.aligned_scan.points = self.o3d.utility.Vector3dVector(points)

        # Step 4: Apply best flip rotation
        points = np.asarray(self.aligned_scan.points)
        points = points @ best_pre_rotation.T
        self.aligned_scan.points = self.o3d.utility.Vector3dVector(points)

        # Step 5: Apply ICP transformation
        self.aligned_scan.transform(reg_fine.transformation)

        # Step 6: Move to reference center
        self.aligned_scan.translate(ref_center)

        self._update_progress("align", 70, f"Alignment complete. Fitness: {reg_fine.fitness:.2%}")
        logger.info(f"Final alignment: fitness={reg_fine.fitness:.2%}, RMSE={reg_fine.inlier_rmse:.3f}mm")

        return reg_fine.fitness, reg_fine.inlier_rmse
    
    def compute_deviations(self, chunk_size: int = 10000, use_kdtree: bool = True) -> np.ndarray:
        """Compute point-to-mesh distances with chunked processing for memory efficiency.

        Args:
            chunk_size: Number of points to process at once (default 10000)
            use_kdtree: Use scipy KDTree (memory efficient) vs Trimesh BVH (more accurate)
        """
        self._update_progress("analyze", 75, "Computing deviations...")

        scan_points = np.asarray(self.aligned_scan.points)
        n_points = len(scan_points)

        if use_kdtree:
            # Memory-efficient approach using scipy KDTree
            return self._compute_deviations_kdtree(scan_points, n_points, chunk_size)
        else:
            # Original approach using Trimesh BVH (more accurate but memory intensive)
            return self._compute_deviations_trimesh(scan_points, n_points, chunk_size)

    def _compute_deviations_kdtree(self, scan_points: np.ndarray, n_points: int, chunk_size: int) -> np.ndarray:
        """Memory-efficient deviation computation using scipy cKDTree."""
        from scipy.spatial import cKDTree

        self._update_progress("analyze", 76, "Sampling reference surface...")

        # Sample dense points from reference mesh for KDTree
        ref_sample = self.reference_mesh.sample_points_uniformly(
            number_of_points=max(200000, n_points * 2)
        )
        ref_points = np.asarray(ref_sample.points)
        ref_sample.estimate_normals()
        ref_normals = np.asarray(ref_sample.normals)

        self._update_progress("analyze", 77, "Building spatial index...")

        # Build KDTree (much more memory efficient than Trimesh BVH)
        tree = cKDTree(ref_points)

        # Process in chunks
        all_deviations = []
        n_chunks = (n_points + chunk_size - 1) // chunk_size

        for i in range(0, n_points, chunk_size):
            chunk_idx = i // chunk_size + 1
            end_idx = min(i + chunk_size, n_points)
            chunk_points = scan_points[i:end_idx]

            self._update_progress("analyze", 77 + int(8 * chunk_idx / n_chunks),
                                  f"Computing deviations... ({chunk_idx}/{n_chunks})")

            # Query nearest neighbors
            distances, indices = tree.query(chunk_points, k=1)

            # Compute signed distances using normals
            nearest_normals = ref_normals[indices]
            vectors = chunk_points - ref_points[indices]
            signs = np.sign(np.sum(vectors * nearest_normals, axis=1))

            all_deviations.append(distances * signs)

        self.deviations = np.concatenate(all_deviations)
        self._update_progress("analyze", 85, f"Computed {len(self.deviations)} deviation values")
        return self.deviations

    def _compute_deviations_trimesh(self, scan_points: np.ndarray, n_points: int, chunk_size: int) -> np.ndarray:
        """Original deviation computation using Trimesh BVH (more accurate, more memory)."""
        mesh = self.trimesh.Trimesh(
            vertices=np.asarray(self.reference_mesh.vertices),
            faces=np.asarray(self.reference_mesh.triangles)
        )

        all_distances = []
        all_signs = []
        n_chunks = (n_points + chunk_size - 1) // chunk_size

        for i in range(0, n_points, chunk_size):
            chunk_idx = i // chunk_size + 1
            end_idx = min(i + chunk_size, n_points)
            chunk_points = scan_points[i:end_idx]

            self._update_progress("analyze", 75 + int(10 * chunk_idx / n_chunks),
                                  f"Computing deviations... ({chunk_idx}/{n_chunks})")

            closest_points, distances, triangle_ids = mesh.nearest.on_surface(chunk_points)

            face_normals = mesh.face_normals[triangle_ids]
            vectors = chunk_points - closest_points
            signs = np.sign(np.sum(vectors * face_normals, axis=1))

            all_distances.append(distances)
            all_signs.append(signs)

        distances = np.concatenate(all_distances)
        signs = np.concatenate(all_signs)
        self.deviations = distances * signs

        self._update_progress("analyze", 85, f"Computed {len(self.deviations)} deviation values")
        return self.deviations
    
    def generate_heatmaps(self, tolerance: float, output_dir: str) -> Dict[str, str]:
        """Generate and save deviation heatmap images.

        Args:
            tolerance: Tolerance threshold in mm
            output_dir: Directory to save heatmap images

        Returns:
            Dict mapping view name to file path
        """
        if self.aligned_scan is None or self.deviations is None:
            logger.warning("Cannot generate heatmaps: no scan data available")
            return {}

        if self.heatmap_renderer is None:
            logger.warning("Heatmap renderer not available")
            return {}

        self._update_progress("analyze", 86, "Generating heatmap visualizations...")

        points = np.asarray(self.aligned_scan.points)
        deviations = self.deviations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        heatmap_paths = {}

        try:
            # Generate multi-view heatmap
            multi_view_bytes = self.heatmap_renderer.render_multi_view(points, deviations, tolerance)
            multi_path = output_path / "heatmap_multi_view.png"
            with open(multi_path, "wb") as f:
                f.write(multi_view_bytes)
            heatmap_paths["multi_view"] = str(multi_path)

            # Generate individual views
            for view in ["top", "front", "side"]:
                view_bytes = self.heatmap_renderer.render_heatmap(points, deviations, tolerance, view)
                view_path = output_path / f"heatmap_{view}.png"
                with open(view_path, "wb") as f:
                    f.write(view_bytes)
                heatmap_paths[view] = str(view_path)

            logger.info(f"Generated {len(heatmap_paths)} heatmap images")

        except Exception as e:
            logger.error(f"Failed to generate heatmaps: {e}")

        return heatmap_paths

    def generate_3d_snapshots(self, tolerance: float, output_dir: str, reference_mesh_path: str = None) -> Dict[str, str]:
        """Generate 3D model snapshots with heatmap overlay.

        Args:
            tolerance: Tolerance threshold in mm
            output_dir: Directory to save snapshot images
            reference_mesh_path: Optional path to reference mesh PLY file

        Returns:
            Dict mapping view name to file path
        """
        if self.aligned_scan is None or self.deviations is None:
            logger.warning("Cannot generate 3D snapshots: no scan data available")
            return {}

        if self.snapshot_renderer is None:
            logger.warning("3D snapshot renderer not available")
            return {}

        self._update_progress("analyze", 87, "Generating 3D model snapshots...")

        points = np.asarray(self.aligned_scan.points)
        deviations = self.deviations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        snapshot_paths = {}

        try:
            # Generate multi-view 3D snapshot
            multi_view_bytes = self.snapshot_renderer.render_multi_view_3d(
                points, deviations, tolerance,
                reference_mesh_path=reference_mesh_path
            )
            multi_path = output_path / "snapshot_3d_multi_view.png"
            with open(multi_path, "wb") as f:
                f.write(multi_view_bytes)
            snapshot_paths["multi_view_3d"] = str(multi_path)

            # Generate individual 3D views
            for view in ["iso", "front", "side", "top"]:
                view_bytes = self.snapshot_renderer.render_3d_snapshot(
                    points, deviations, tolerance,
                    reference_mesh_path=reference_mesh_path,
                    view=view,
                    width=1200,
                    height=900
                )
                view_path = output_path / f"snapshot_3d_{view}.png"
                with open(view_path, "wb") as f:
                    f.write(view_bytes)
                snapshot_paths[f"3d_{view}"] = str(view_path)

            logger.info(f"Generated {len(snapshot_paths)} 3D snapshot images")

        except Exception as e:
            logger.error(f"Failed to generate 3D snapshots: {e}")
            import traceback
            traceback.print_exc()

        return snapshot_paths

    def analyze_regions(self, tolerance: float) -> List[RegionAnalysis]:
        """Analyze deviations by region using configurable thresholds"""
        self._update_progress("analyze", 88, "Analyzing regions...")

        points = np.asarray(self.aligned_scan.points)
        deviations = self.deviations
        cfg = self.region_config

        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        center = (min_pt + max_pt) / 2

        regions = []
        # Region definitions using configurable thresholds
        region_defs = [
            ("Top Surface", lambda p: p[:, 2] > center[2] + cfg.top_surface_threshold * (max_pt[2] - center[2])),
            ("Bottom Surface", lambda p: p[:, 2] < center[2] - cfg.bottom_surface_threshold * (max_pt[2] - center[2])),
            ("Front Edge", lambda p: p[:, 1] > center[1] + cfg.edge_threshold * (max_pt[1] - center[1])),
            ("Rear Edge", lambda p: p[:, 1] < center[1] - cfg.edge_threshold * (max_pt[1] - center[1])),
            ("Left Side", lambda p: p[:, 0] < center[0] - cfg.side_threshold * (max_pt[0] - center[0])),
            ("Right Side", lambda p: p[:, 0] > center[0] + cfg.side_threshold * (max_pt[0] - center[0])),
            ("Center/Curved", lambda p: (
                (np.abs(p[:, 0] - center[0]) < cfg.center_threshold * (max_pt[0] - min_pt[0])) &
                (np.abs(p[:, 1] - center[1]) < cfg.center_threshold * (max_pt[1] - min_pt[1]))
            )),
        ]

        for name, mask_fn in region_defs:
            mask = mask_fn(points)
            if np.sum(mask) < cfg.min_points:
                continue

            region_devs = deviations[mask]
            abs_devs = np.abs(region_devs)
            pass_rate = np.sum(abs_devs <= tolerance) / len(region_devs)

            regions.append(RegionAnalysis(
                name=name,
                point_count=int(np.sum(mask)),
                mean_deviation=float(np.mean(region_devs)),
                max_deviation=float(np.max(abs_devs)),
                min_deviation=float(np.min(region_devs)),
                std_deviation=float(np.std(region_devs)),
                pass_rate=pass_rate,
                status="OK" if pass_rate >= cfg.pass_rate_threshold else "ATTENTION",
                deviation_direction="outward" if np.mean(region_devs) > 0 else "inward"
            ))

        return regions
    
    def run_analysis(
        self,
        part_id: str,
        part_name: str,
        material: str,
        tolerance: float,
        drawing_context: str = "",
        drawing_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        enable_enhanced_analysis: bool = True,
        job_id: Optional[str] = None,
    ) -> QCReport:
        """Run complete QC analysis.

        Args:
            part_id: Part number/ID
            part_name: Part name/description
            material: Material specification
            tolerance: Tolerance in mm
            drawing_context: Extracted text from technical drawing PDF
            drawing_path: Path to technical drawing file (PDF or image)
            output_dir: Directory to save heatmap images

        Returns:
            Complete QCReport with analysis results
        """
        from datetime import datetime

        # Use provided output_dir or instance default
        heatmap_dir = output_dir or self.output_dir

        # Initialize report
        report = QCReport(
            part_id=part_id,
            part_name=part_name,
            material=material,
            tolerance=tolerance,
            timestamp=datetime.now().isoformat(),
            drawing_context=drawing_context
        )

        try:
            # Preprocess
            self.preprocess()

            # Align
            fitness, rmse = self.align()
            report.alignment_fitness = fitness
            report.alignment_rmse = rmse

            # Compute deviations
            deviations = self.compute_deviations()
            abs_devs = np.abs(deviations)

            # Statistics
            report.total_points = len(deviations)
            report.mean_deviation = float(np.mean(deviations))
            report.max_deviation = float(np.max(deviations))
            report.min_deviation = float(np.min(deviations))
            report.std_deviation = float(np.std(deviations))

            report.points_in_tolerance = int(np.sum(abs_devs <= tolerance))
            report.points_out_of_tolerance = int(np.sum(abs_devs > tolerance))
            report.pass_rate = report.points_in_tolerance / len(deviations)

            # Generate heatmaps if output directory provided
            if heatmap_dir:
                heatmap_paths = self.generate_heatmaps(tolerance, heatmap_dir)
                report.heatmap_multi_view = heatmap_paths.get("multi_view", "")
                report.heatmap_top = heatmap_paths.get("top", "")
                report.heatmap_front = heatmap_paths.get("front", "")
                report.heatmap_side = heatmap_paths.get("side", "")

                # Save deviations for 3D viewer heatmap
                try:
                    deviations_path = Path(heatmap_dir) / "deviations.npy"
                    np.save(str(deviations_path), deviations)
                    logger.info(f"Saved {len(deviations)} deviations to {deviations_path}")
                except Exception as e:
                    logger.warning(f"Failed to save deviations: {e}")

                # Export ALIGNED scan for 3D viewer (must show aligned version, not original)
                try:
                    if self.aligned_scan is not None:
                        import trimesh
                        aligned_scan_path = Path(heatmap_dir) / "aligned_scan.ply"
                        points = np.asarray(self.aligned_scan.points).astype(np.float32)
                        # Create point cloud with colors based on deviations
                        colors = np.zeros((len(points), 4), dtype=np.uint8)
                        for i, dev in enumerate(deviations):
                            # Green (in tolerance) -> Yellow -> Red (out of tolerance)
                            abs_dev = abs(dev)
                            normalized = min(abs_dev / tolerance, 1.0)
                            if normalized < 0.5:
                                t = normalized * 2
                                colors[i] = [int(t * 255), 255, 0, 255]  # Green to Yellow
                            else:
                                t = (normalized - 0.5) * 2
                                colors[i] = [255, int((1 - t) * 255), 0, 255]  # Yellow to Red
                        cloud = trimesh.PointCloud(points, colors=colors)
                        cloud.export(str(aligned_scan_path), file_type='ply')
                        logger.info(f"Exported aligned scan to {aligned_scan_path}")
                except Exception as e:
                    logger.warning(f"Failed to export aligned scan: {e}")

                # Export reference mesh for 3D viewer (convert CAD to PLY if needed)
                # Use trimesh to export with float32 (Three.js PLYLoader doesn't support float64)
                ref_mesh_path = None
                try:
                    if self.reference_mesh is not None:
                        ref_mesh_path = Path(heatmap_dir) / "reference_mesh.ply"
                        # Convert to trimesh for proper float32 PLY export
                        import trimesh
                        vertices = np.asarray(self.reference_mesh.vertices).astype(np.float32)
                        faces = np.asarray(self.reference_mesh.triangles)
                        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        tri_mesh.export(str(ref_mesh_path), file_type='ply')
                        logger.info(f"Exported reference mesh to {ref_mesh_path}")
                except Exception as e:
                    logger.warning(f"Failed to export reference mesh: {e}")

                # Generate 3D model snapshots with heatmap overlay
                try:
                    snapshot_paths = self.generate_3d_snapshots(
                        tolerance, heatmap_dir,
                        reference_mesh_path=str(ref_mesh_path) if ref_mesh_path else None
                    )
                    report.snapshot_3d_multi_view = snapshot_paths.get("multi_view_3d", "")
                    report.snapshot_3d_iso = snapshot_paths.get("3d_iso", "")
                    report.snapshot_3d_front = snapshot_paths.get("3d_front", "")
                    report.snapshot_3d_side = snapshot_paths.get("3d_side", "")
                    report.snapshot_3d_top = snapshot_paths.get("3d_top", "")
                except Exception as e:
                    logger.warning(f"Failed to generate 3D snapshots: {e}")

            # Regional analysis
            self._update_progress("analyze", 90, "Performing regional analysis...")
            report.regions = self.analyze_regions(tolerance)

            # AI Analysis
            self._update_progress("ai", 92, "Running AI interpretation...")
            self._run_ai_analysis(report, drawing_context, drawing_path)

            # Enhanced Analysis (bend detection + multi-model)
            if enable_enhanced_analysis and ENHANCED_ANALYSIS_AVAILABLE:
                self._update_progress("enhanced", 94, "Running enhanced bend detection...")
                try:
                    # Run bend detection
                    bend_result = self.detect_bends()
                    if bend_result:
                        report.bend_detection_result = bend_result
                        report.bend_results = bend_result.get("bends", [])
                        logger.info(f"Detected {len(report.bend_results)} bends")

                    # Run multi-model enhanced analysis if API keys available
                    if AI_ENABLED:
                        self._update_progress("enhanced", 96, "Running multi-model analysis...")
                        part_info = {
                            "part_id": part_id,
                            "part_name": part_name,
                            "material": material,
                        }
                        analysis_job_id = job_id or part_id

                        enhanced_result = self.run_enhanced_analysis_sync(
                            job_id=analysis_job_id,
                            tolerance=tolerance,
                            drawing_path=drawing_path,
                            part_info=part_info,
                        )

                        if enhanced_result:
                            report.enhanced_analysis = enhanced_result
                            report.correlation_2d_3d = enhanced_result.get("correlation_2d_3d")
                            report.pipeline_status = enhanced_result.get("pipeline_status")

                            # Merge enhanced bend results with better analysis
                            if enhanced_result.get("bend_results"):
                                report.bend_results = enhanced_result["bend_results"]

                            # Merge enhanced root causes
                            if enhanced_result.get("enhanced_root_causes"):
                                for erc in enhanced_result["enhanced_root_causes"]:
                                    if erc is None:
                                        continue
                                    report.root_causes.append(RootCause(
                                        issue=erc.get("description", ""),
                                        likely_cause=erc.get("category", ""),
                                        technical_explanation="; ".join(erc.get("evidence", []) or []),
                                        confidence=erc.get("confidence", 0.5),
                                        recommendation="; ".join(erc.get("recommendations", []) or []),
                                        priority=erc.get("severity", "MEDIUM").upper(),
                                    ))

                            logger.info("Enhanced multi-model analysis complete")
                except Exception as e:
                    logger.warning(f"Enhanced analysis failed (continuing with basic): {e}")

            # Calculate quality score and verdict
            self._calculate_verdict(report)

            self._update_progress("complete", 100, "Analysis complete!")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            report.overall_result = QCResult.FAIL
            report.ai_summary = f"Analysis error: {str(e)}"

        return report
    
    def _run_ai_analysis(self, report: QCReport, drawing_context: str, drawing_path: Optional[str] = None):
        """Generate AI interpretations for regions and root causes.

        Uses multimodal AI (Claude/Gemini/GPT) when API key is available,
        falls back to rule-based analysis otherwise.
        """
        # Try AI-powered analysis first if API key is available
        if AI_ENABLED and self.aligned_scan is not None and self.deviations is not None:
            try:
                self._update_progress("ai", 93, "Running AI vision analysis...")
                ai_result = self._run_multimodal_ai_analysis(report, drawing_context, drawing_path)
                if ai_result:
                    logger.info(f"AI analysis complete. Model: {ai_result.get('model_used', 'unknown')}")
                    return  # AI analysis succeeded
            except Exception as e:
                logger.warning(f"AI analysis failed, falling back to rule-based: {e}")

        # Fallback to rule-based analysis
        self._update_progress("ai", 93, "Running rule-based analysis...")

        # Interpret each region
        for region in report.regions:
            region.ai_interpretation = self._interpret_region(region, report.tolerance, report.material)

        # Identify root causes
        report.root_causes = self._identify_root_causes(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Generate AI summary
        report.ai_summary = self._generate_summary(report, drawing_context)
        report.ai_detailed_analysis = self._generate_detailed_analysis(report, drawing_context)

    def _run_multimodal_ai_analysis(self, report: QCReport, drawing_context: str, drawing_path: Optional[str] = None) -> Optional[Dict]:
        """Run actual multimodal AI analysis using vision models.

        Args:
            report: The QC report to populate
            drawing_context: Extracted text from technical drawing PDF
            drawing_path: Path to technical drawing file (PDF or image)

        Returns:
            Dict with AI results if successful, None otherwise
        """
        try:
            from ai_analyzer import create_ai_analyzer, AIProvider

            # Get points and deviations
            points = np.asarray(self.aligned_scan.points)
            deviations = self.deviations

            # Determine which provider to use
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "claude"
            elif os.environ.get("GOOGLE_API_KEY"):
                provider = "gemini"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                return None

            # Create analyzer
            analyzer = create_ai_analyzer(provider=provider)

            # Prepare part info
            part_info = {
                "part_id": report.part_id,
                "part_name": report.part_name,
                "material": report.material
            }

            # Prepare regional data for context
            regions_data = [
                {
                    "name": r.name,
                    "mean_deviation_mm": r.mean_deviation,
                    "max_deviation_mm": r.max_deviation,
                    "pass_rate": r.pass_rate * 100
                }
                for r in report.regions
            ]

            # Load technical drawing if available
            technical_drawing_bytes = None
            if drawing_path and os.path.exists(drawing_path):
                try:
                    with open(drawing_path, 'rb') as f:
                        technical_drawing_bytes = f.read()
                    logger.info(f"Loaded technical drawing: {drawing_path}")
                except Exception as e:
                    logger.warning(f"Failed to load technical drawing: {e}")

            # Call AI analyzer with drawing for visual analysis
            ai_result = analyzer.analyze(
                points=points,
                deviations=deviations,
                tolerance=report.tolerance,
                part_info=part_info,
                regions=regions_data,
                technical_drawing=technical_drawing_bytes,
                drawing_context=drawing_context
            )

            # Populate report from AI result
            report.overall_result = QCResult(ai_result.verdict) if ai_result.verdict in ["PASS", "FAIL", "WARNING"] else report.overall_result
            report.quality_score = ai_result.quality_score
            report.confidence = ai_result.confidence
            report.ai_summary = ai_result.summary
            report.ai_detailed_analysis = ai_result.detailed_analysis

            # Convert AI root causes to our format
            report.root_causes = [
                RootCause(
                    issue=rc.get("issue", "Unknown issue"),
                    likely_cause=rc.get("cause", "Unknown cause"),
                    technical_explanation=rc.get("evidence", ""),
                    confidence=rc.get("confidence", 0.5),
                    recommendation="",  # Will be filled from recommendations
                    priority="HIGH" if rc.get("confidence", 0) > 0.7 else "MEDIUM"
                )
                for rc in ai_result.root_causes
            ]

            # Convert AI recommendations
            report.recommendations = [
                {
                    "priority": rec.get("priority", "MEDIUM").upper(),
                    "action": rec.get("action", ""),
                    "expected_improvement": rec.get("expected_improvement", "")
                }
                for rec in ai_result.recommendations
            ]

            # Update regions with AI defect locations if available
            for defect in ai_result.defects_found:
                location = defect.get("location", "").lower()
                for region in report.regions:
                    if region.name.lower() in location or location in region.name.lower():
                        region.ai_interpretation = defect.get("description", region.ai_interpretation)

            # Fill any regions without AI interpretation with rule-based
            for region in report.regions:
                if not region.ai_interpretation:
                    region.ai_interpretation = self._interpret_region(region, report.tolerance, report.material)

            # Store AI metadata
            report.ai_model_used = ai_result.model_used
            report.ai_tokens_used = ai_result.tokens_used

            return {
                "model_used": ai_result.model_used,
                "tokens_used": ai_result.tokens_used
            }

        except ImportError as e:
            logger.warning(f"AI analyzer module not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Multimodal AI analysis failed: {e}")
            return None
    
    def _interpret_region(self, region: RegionAnalysis, tolerance: float, material: str) -> str:
        """AI interpretation of a single region"""
        
        if region.status == "OK":
            return f"Region within specification. {region.pass_rate*100:.1f}% of points meet ±{tolerance}mm tolerance."
        
        severity = "significant" if region.max_deviation > tolerance * 2 else "moderate"
        direction = "bulging outward" if region.deviation_direction == "outward" else "depressed inward"
        
        interpretations = {
            "Top Surface": f"Top surface shows {severity} {direction} deviation. This may indicate {'springback after forming' if direction == 'outward' else 'over-forming or material thinning'}.",
            "Bottom Surface": f"Bottom surface deviation suggests {'incomplete forming or springback' if direction == 'outward' else 'excessive pressure during forming'}.",
            "Front Edge": f"Front edge deviation of {region.max_deviation:.3f}mm indicates {'edge curl from trimming stress' if direction == 'outward' else 'edge compression'}.",
            "Rear Edge": f"Rear edge shows {direction} trend, typical of {'material flow during forming' if region.mean_deviation > 0 else 'clamping effects'}.",
            "Left Side": f"Left side deviation pattern suggests {'asymmetric tool wear or positioning error' if region.max_deviation > tolerance * 1.5 else 'minor forming variation'}.",
            "Right Side": f"Right side shows {severity} deviation, may indicate {'tool alignment issue' if region.max_deviation > tolerance * 2 else 'normal process variation'}.",
            "Center/Curved": f"Curved section shows classic {'springback pattern - elastic recovery after bend release' if direction == 'outward' else 'over-bend condition'}. {material} at this geometry typically requires bend compensation.",
        }
        
        return interpretations.get(region.name, f"Region shows {severity} {direction} deviation of {region.max_deviation:.3f}mm.")
    
    def _identify_root_causes(self, report: QCReport) -> List[RootCause]:
        """Identify root causes from analysis"""
        causes = []
        
        # Check for systematic offset
        if abs(report.mean_deviation) > report.tolerance * 0.3:
            causes.append(RootCause(
                issue=f"Systematic {'positive' if report.mean_deviation > 0 else 'negative'} offset of {report.mean_deviation:.3f}mm",
                likely_cause="Material thickness variation or tooling setup error",
                technical_explanation=f"A uniform {report.mean_deviation:.3f}mm offset across the part indicates either: (1) incoming material thickness differs from nominal, (2) die gap is incorrectly set, or (3) systematic measurement alignment error.",
                confidence=0.75,
                recommendation=f"Verify incoming {report.material} thickness with micrometer. Check die gap = material thickness + 10%.",
                priority="HIGH"
            ))
        
        # Check for springback
        curved_regions = [r for r in report.regions if "Curved" in r.name or "Center" in r.name]
        for region in curved_regions:
            if region.deviation_direction == "outward" and region.mean_deviation > report.tolerance * 0.5:
                causes.append(RootCause(
                    issue=f"Springback on {region.name}: {region.mean_deviation:.3f}mm mean outward deviation",
                    likely_cause="Elastic recovery after bending",
                    technical_explanation=f"{report.material} exhibits elastic recovery after forming. The stored elastic energy causes the material to spring back toward its original flat state after tool release.",
                    confidence=0.85,
                    recommendation="Increase overbend angle by 2-4°. Use springback compensation factor of 1.02-1.05 for this material/thickness combination.",
                    priority="HIGH"
                ))
        
        # Check edges
        edge_regions = [r for r in report.regions if "Edge" in r.name]
        for region in edge_regions:
            if region.status == "ATTENTION":
                causes.append(RootCause(
                    issue=f"{region.name} deviation: max {region.max_deviation:.3f}mm",
                    likely_cause="Edge forming defect or trimming stress",
                    technical_explanation="Edge regions often show deviation due to: material flow during forming, residual stress from trimming, or insufficient support during bending causing edge curl.",
                    confidence=0.70,
                    recommendation="Add edge support fixtures during bending. Review trim sequence and allowances.",
                    priority="MEDIUM"
                ))
        
        # Check for extreme deviations
        if report.max_deviation > report.tolerance * 3:
            causes.append(RootCause(
                issue=f"Extreme deviation detected: {report.max_deviation:.3f}mm (3× tolerance)",
                likely_cause="Manufacturing defect or handling damage",
                technical_explanation="Extreme localized deviations typically indicate: tool damage/gouging, material defect (inclusion, lamination), handling damage, or debris during forming.",
                confidence=0.80,
                recommendation="Inspect forming tools for damage. Review material handling. Check for debris in die cavity.",
                priority="CRITICAL"
            ))
        
        return causes
    
    def _generate_recommendations(self, report: QCReport) -> List[Dict]:
        """Generate actionable recommendations"""
        recs = []
        seen = set()
        
        for rc in report.root_causes:
            if rc.recommendation not in seen:
                seen.add(rc.recommendation)
                recs.append({
                    "priority": rc.priority,
                    "action": rc.recommendation,
                    "expected_improvement": self._estimate_improvement(rc)
                })
        
        if not recs:
            recs.append({
                "priority": "LOW",
                "action": "Continue monitoring process. Current variation within acceptable limits.",
                "expected_improvement": "Maintain quality level"
            })
        
        return recs
    
    def _estimate_improvement(self, root_cause: RootCause) -> str:
        """Estimate improvement from fixing root cause"""
        if "springback" in root_cause.likely_cause.lower():
            return "60-80% reduction in springback deviation"
        elif "thickness" in root_cause.likely_cause.lower():
            return "Eliminate systematic offset"
        elif "edge" in root_cause.likely_cause.lower():
            return "40-60% reduction in edge deviation"
        else:
            return "Significant quality improvement expected"
    
    def _generate_summary(self, report: QCReport, drawing_context: str) -> str:
        """Generate executive summary"""
        result_text = {
            QCResult.PASS: "PASSES quality inspection",
            QCResult.WARNING: "CONDITIONALLY PASSES with review required",
            QCResult.FAIL: "FAILS quality inspection"
        }
        
        summary = f"Part {report.part_id} ({report.part_name}) {result_text[report.overall_result]}. "
        summary += f"{report.pass_rate*100:.1f}% of {report.total_points:,} measured points are within ±{report.tolerance}mm tolerance. "
        
        if report.overall_result != QCResult.PASS:
            summary += f"Maximum deviation of {report.max_deviation:.3f}mm detected. "
            summary += f"{len(report.root_causes)} manufacturing issue(s) identified requiring attention."
        
        if drawing_context:
            summary += f" Analysis performed with reference to technical drawing specifications."
        
        return summary
    
    def _generate_detailed_analysis(self, report: QCReport, drawing_context: str) -> str:
        """Generate detailed AI analysis"""
        analysis = []
        
        analysis.append("## Detailed Analysis\n")
        
        # Alignment assessment
        if report.alignment_fitness < 0.8:
            analysis.append(f"⚠️ **Alignment Quality Warning**: Fitness score of {report.alignment_fitness:.1%} is below optimal (>80%). Results may have reduced accuracy. Consider improving scan coverage or part positioning.\n")
        else:
            analysis.append(f"✓ **Alignment Quality**: Good ({report.alignment_fitness:.1%} fitness). Scan properly registered to reference model.\n")
        
        # Statistical interpretation
        analysis.append(f"\n### Statistical Overview\n")
        analysis.append(f"The part shows a mean deviation of {report.mean_deviation:.4f}mm with standard deviation of {report.std_deviation:.4f}mm. ")
        
        if abs(report.mean_deviation) > report.tolerance * 0.2:
            bias = "positive (outward)" if report.mean_deviation > 0 else "negative (inward)"
            analysis.append(f"There is a systematic {bias} bias indicating potential tooling or material issues. ")
        else:
            analysis.append("No significant systematic bias detected. ")
        
        analysis.append(f"The deviation range spans from {report.min_deviation:.4f}mm to {report.max_deviation:.4f}mm.\n")
        
        # Regional breakdown
        analysis.append(f"\n### Regional Analysis\n")
        for region in report.regions:
            status_icon = "✓" if region.status == "OK" else "⚠️"
            analysis.append(f"\n**{status_icon} {region.name}** ({region.point_count:,} points)\n")
            analysis.append(f"- Pass rate: {region.pass_rate*100:.1f}%\n")
            analysis.append(f"- Mean: {region.mean_deviation:.4f}mm | Max: {region.max_deviation:.4f}mm\n")
            analysis.append(f"- {region.ai_interpretation}\n")
        
        # Drawing context integration
        if drawing_context:
            analysis.append(f"\n### Drawing Specification Context\n")
            analysis.append(f"The technical drawing specifies: {drawing_context[:500]}...\n")
            analysis.append("Analysis has been performed with these specifications in mind.\n")
        
        # Manufacturing insights
        analysis.append(f"\n### Manufacturing Insights\n")
        if report.root_causes:
            for rc in report.root_causes:
                analysis.append(f"\n**{rc.priority} Priority Issue**: {rc.issue}\n")
                analysis.append(f"- Root Cause: {rc.likely_cause} (Confidence: {rc.confidence*100:.0f}%)\n")
                analysis.append(f"- Technical Detail: {rc.technical_explanation}\n")
        else:
            analysis.append("No significant manufacturing issues identified. Process appears stable.\n")
        
        return "".join(analysis)
    
    def _calculate_verdict(self, report: QCReport):
        """Calculate final verdict and quality score"""
        # Quality score calculation
        pass_rate_score = report.pass_rate * 60
        
        max_dev_ratio = abs(report.max_deviation) / report.tolerance
        if max_dev_ratio <= 1:
            max_dev_score = 20
        elif max_dev_ratio <= 2:
            max_dev_score = 10
        elif max_dev_ratio <= 3:
            max_dev_score = 5
        else:
            max_dev_score = 0
        
        issue_count = len(report.root_causes)
        critical_issues = sum(1 for rc in report.root_causes if rc.priority == "CRITICAL")
        
        if critical_issues > 0:
            issue_score = 0
        elif issue_count == 0:
            issue_score = 20
        elif issue_count <= 2:
            issue_score = 15
        else:
            issue_score = 5
        
        report.quality_score = min(100, pass_rate_score + max_dev_score + issue_score)
        
        # Verdict
        if report.pass_rate >= 0.99 and max_dev_ratio <= 1.5:
            report.overall_result = QCResult.PASS
            report.confidence = 0.95
        elif report.pass_rate >= 0.95 and max_dev_ratio <= 2.0:
            report.overall_result = QCResult.WARNING
            report.confidence = 0.85
        else:
            report.overall_result = QCResult.FAIL
            report.confidence = 0.92
