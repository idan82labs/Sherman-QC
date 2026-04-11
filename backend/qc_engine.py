"""
Scan QC Engine - Core Analysis Module
Handles mesh loading, alignment, deviation computation, and AI analysis.
"""

import os
import numpy as np
import random
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
    from bend_detector import BendDetector, BendDetectionResult, analyze_bend_deviations, measure_all_scan_bends
    from cad_bend_extractor import CADBendExtractor, CADBendExtractionResult
    from cad_dimension_extractor import CADDimensionExtractor, CADDimensionExtractionResult, measure_scan_dimension, measure_scan_dimensions_batch
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
class AlignmentMetadata:
    """Traceable alignment provenance for downstream metrology."""
    source: str = "GLOBAL_ICP"
    parent_island_id: str = "NONE"
    confidence: float = 0.0
    abstained: bool = False
    conditioning: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alignment_source": self.source,
            "parent_island_id": self.parent_island_id,
            "alignment_confidence": round(float(self.confidence or 0.0), 4),
            "alignment_abstained": bool(self.abstained),
            "alignment_conditioning": dict(self.conditioning or {}),
        }


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

    # Bend analysis images (paths)
    bend_snapshot_paths: Dict[str, str] = field(default_factory=dict)

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
                **self.bend_snapshot_paths,
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
        self.reference_pcd = None
        self.scan_pcd = None
        self.raw_scan_pcd = None
        self.alignment_metadata = AlignmentMetadata()
        self.aligned_scan = None
        self.aligned_scan_raw = None
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
        Detect bends using dual approach: CAD mesh first, point cloud fallback.

        Primary: Extract bends from the CAD reference mesh (clean, structured data).
        Fallback: Detect bends from the scan point cloud (noisy but always available).
        Scan measurement: Measure actual bend angles in scan at CAD-specified locations.

        Args:
            points: Optional point cloud (uses self.aligned_scan if not provided)
            deviations: Optional deviations array

        Returns:
            BendDetectionResult-compatible dict, or None if detection fails
        """
        if points is None:
            if self.aligned_scan is None:
                logger.warning("No point cloud available for bend detection")
                return None
            points = np.asarray(self.aligned_scan.points)

        if deviations is None:
            deviations = self.deviations

        # Primary approach: extract bends from CAD reference mesh
        cad_bends_result = None
        if self.reference_mesh is not None and ENHANCED_ANALYSIS_AVAILABLE:
            try:
                self._update_progress("bend_detect", 88, "Extracting bends from CAD mesh...")
                extractor = CADBendExtractor()
                cad_bends_result = extractor.extract_from_mesh(self.reference_mesh)
                logger.info(f"CAD mesh bend extraction: {cad_bends_result.total_bends} bends "
                           f"from {cad_bends_result.total_major_surfaces} surfaces "
                           f"({cad_bends_result.processing_time_ms:.0f}ms)")
            except Exception as e:
                logger.warning(f"CAD mesh bend extraction failed: {e}")

        # If CAD extraction found bends, measure them in the scan
        if cad_bends_result and cad_bends_result.total_bends > 0:
            try:
                self._update_progress("bend_detect", 90, "Measuring scan bends at CAD locations...")
                cad_bend_dicts = [b.to_dict() for b in cad_bends_result.bends]

                # Build surface face map for surface classification measurement
                cad_vertices = np.asarray(self.reference_mesh.vertices)
                cad_triangles = np.asarray(self.reference_mesh.triangles)
                surface_face_map = {
                    s.surface_id: s.face_indices
                    for s in cad_bends_result.surfaces
                }

                scan_measurements = measure_all_scan_bends(
                    points, cad_bend_dicts, search_radius=35.0,
                    cad_vertices=cad_vertices,
                    cad_triangles=cad_triangles,
                    surface_face_map=surface_face_map,
                    deviations=deviations,
                )

                # Build combined result with CAD angles + scan measurements
                bends = []
                for cad_bend, measurement in zip(cad_bends_result.bends, scan_measurements):
                    bend_dict = cad_bend.to_dict()
                    bend_dict["detection_source"] = "cad_mesh"
                    if measurement["success"]:
                        bend_dict["scan_angle"] = measurement["measured_angle"]
                        bend_dict["scan_deviation"] = measurement["deviation"]
                        bend_dict["scan_measured"] = True
                        bend_dict["measurement_method"] = measurement.get("measurement_method", "unknown")
                        # Use confidence from the measurement method if provided
                        # (gradient method computes its own R²-based confidence)
                        if "measurement_confidence" in measurement and measurement["measurement_confidence"]:
                            bend_dict["measurement_confidence"] = measurement["measurement_confidence"]
                        else:
                            # Fallback: heuristic based on point count and CAD alignment
                            pts = measurement.get("side1_points", 0) + measurement.get("side2_points", 0)
                            align = (measurement.get("cad_alignment1", 0) + measurement.get("cad_alignment2", 0)) / 2
                            if pts >= 100 and align >= 0.95:
                                bend_dict["measurement_confidence"] = "high"
                            elif pts >= 50 and align >= 0.9:
                                bend_dict["measurement_confidence"] = "medium"
                            else:
                                bend_dict["measurement_confidence"] = "low"
                    else:
                        bend_dict["scan_angle"] = None
                        bend_dict["scan_deviation"] = None
                        bend_dict["scan_measured"] = False
                        bend_dict["measurement_method"] = None
                        bend_dict["measurement_confidence"] = None
                    bends.append(bend_dict)

                measured_count = sum(1 for m in scan_measurements if m["success"])
                logger.info(f"Scan measurement: {measured_count}/{len(scan_measurements)} bends measured")

                return {
                    "bends": bends,
                    "total_bends_detected": len(bends),
                    "detection_method": "cad_mesh_extraction",
                    "processing_time_ms": cad_bends_result.processing_time_ms,
                    "scan_measured_count": measured_count,
                    "cad_surfaces": cad_bends_result.total_major_surfaces,
                    "warnings": cad_bends_result.warnings,
                }
            except Exception as e:
                logger.warning(f"Scan bend measurement failed: {e}")
                # Still return CAD-only results
                return {
                    "bends": [b.to_dict() for b in cad_bends_result.bends],
                    "total_bends_detected": cad_bends_result.total_bends,
                    "detection_method": "cad_mesh_extraction",
                    "processing_time_ms": cad_bends_result.processing_time_ms,
                    "scan_measured_count": 0,
                    "warnings": cad_bends_result.warnings + ["Scan measurement failed"],
                }

        # Fallback: point cloud bend detection
        if self.bend_detector is not None:
            try:
                self._update_progress("bend_detect", 88, "Detecting bends in point cloud (fallback)...")
                result = self.bend_detector.detect_bends(points, deviations)
                logger.info(f"Point cloud fallback: {result.total_bends_detected} bends")
                result_dict = result.to_dict()
                result_dict["detection_method"] = "point_cloud_curvature"
                return result_dict
            except Exception as e:
                logger.error(f"Point cloud bend detection failed: {e}")

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
                stride = max(1, len(points) // 25000)
                sample_points = points[::stride][:25000]
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

                    # For sheet metal with outward-pointing normals:
                    # - Parallel surfaces (flat): dot ≈ 1 → angle ≈ 0° → not a bend
                    # - 90° bend: dot ≈ 0 → angle = 90°
                    # - Normals pointing opposite (180° apart): dot ≈ -1 → angle = 180°
                    #
                    # The bend angle is how much the sheet is bent from flat (180°).
                    # If surfaces are nearly parallel (dot close to 1 or -1), it's not a bend.
                    # Using abs(dot) incorrectly treats opposite-pointing normals as parallel.
                    angle_between_normals = np.degrees(np.arccos(dot))

                    # Bend angle: supplement of the angle between normals
                    # For outward normals: bend_angle = 180° - angle_between_normals
                    # But we need to handle cases where normals might be inconsistently oriented.
                    # If angle > 90°, the bend angle is 180° - angle (convex bend)
                    # If angle < 90°, the bend angle is 180° - angle (concave bend)
                    bend_angle = 180 - angle_between_normals

                    # Skip near-parallel surfaces (flat regions) - not actual bends
                    if abs(dot) > 0.95:  # Within ~18° of parallel
                        continue

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

            # Step 2: Detect bends - use CAD mesh extraction (primary) or point cloud (fallback)
            self._update_progress("dimension_analysis", 87, "Detecting bends...")
            cad_bends = []
            bends_from_scan = False

            # Primary: Extract bends from CAD reference mesh
            if self.reference_mesh is not None and ENHANCED_ANALYSIS_AVAILABLE:
                try:
                    extractor = CADBendExtractor()
                    cad_extraction = extractor.extract_from_mesh(self.reference_mesh)
                    if cad_extraction.total_bends > 0:
                        cad_bends = [
                            {
                                'bend_id': b.bend_id,
                                'angle_degrees': b.bend_angle,
                                'radius_mm': 5.0,  # Default radius estimate
                                'bend_apex': list(b.bend_center),
                                'detection_confidence': b.confidence,
                            }
                            for b in cad_extraction.bends
                        ]
                        logger.info(f"CAD mesh: extracted {len(cad_bends)} bends")
                except Exception as e:
                    logger.warning(f"CAD mesh bend extraction failed: {e}")

            # Fallback: point cloud detection
            if not cad_bends and self.aligned_scan is not None:
                aligned_points = np.asarray(self.aligned_scan.points)

                if ENHANCED_ANALYSIS_AVAILABLE:
                    detector = BendDetector(
                        normal_radius=10.0,
                        curvature_radius=5.0,
                        min_surface_points=200,
                    )
                    if len(aligned_points) > 100000:
                        stride = max(1, len(aligned_points) // 100000)
                        sample_points = aligned_points[::stride][:100000]
                    else:
                        sample_points = aligned_points

                    bend_result = detector.detect_bends(sample_points)
                    cad_bends = bend_result.bends if bend_result else []
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

                if not cad_bends:
                    logger.info("Primary detectors found no bends, trying surface-normal clustering...")
                    if self.aligned_scan is not None:
                        cad_bends = self._detect_bends_by_surface_clustering(aligned_points)
                        bends_from_scan = True

            logger.info(f"Total bends detected: {len(cad_bends)}")

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

    def run_cad_dimension_analysis(
        self,
        part_id: str = "",
        part_name: str = "",
    ) -> Optional[Dict]:
        """
        Run dimension analysis extracting dimensions from CAD when XLSX is not available.

        This is a fallback for when no XLSX specification file is provided.
        It extracts linear dimensions directly from the CAD mesh geometry:
        1. Bounding box dimensions (overall length, width, height)
        2. Distances between parallel planar surfaces (step heights, etc.)

        Args:
            part_id: Part identifier
            part_name: Part name

        Returns:
            Dimension analysis report as dict, or None on failure
        """
        if not ENHANCED_ANALYSIS_AVAILABLE:
            logger.warning("CAD dimension extraction modules not available")
            return None

        if self.reference_mesh is None:
            logger.warning("No CAD reference mesh available for dimension extraction")
            return None

        try:
            self._update_progress("dimension_analysis", 85, "Extracting dimensions from CAD...")

            # Extract dimensions from CAD mesh with comprehensive settings
            extractor = CADDimensionExtractor(
                min_surface_area_pct=0.5,  # Lower threshold for more surfaces
                num_slices=8,  # Cross-section slices per axis
                edge_angle_threshold=25.0,  # Detect feature edges
                min_dimension=2.0,  # Minimum dimension to report
                max_dimensions=80,  # Maximum dimensions to extract
            )
            cad_result = extractor.extract_from_mesh(self.reference_mesh)

            logger.info(f"CAD dimension extraction: {cad_result.total_dimensions} dimensions "
                       f"({cad_result.processing_time_ms:.0f}ms)")
            if cad_result.extraction_summary:
                logger.info(f"  Breakdown: {cad_result.extraction_summary}")

            if cad_result.total_dimensions == 0:
                logger.warning("No dimensions extracted from CAD mesh")
                return {
                    "source": "cad_extraction",
                    "status": "no_dimensions",
                    "message": "No linear dimensions could be extracted from CAD geometry",
                    "bounding_box": cad_result.bounding_box,
                }

            # Measure corresponding dimensions in scan if available
            scan_measurements = []
            if self.aligned_scan is not None:
                self._update_progress("dimension_analysis", 90, "Measuring dimensions in scan...")
                aligned_points = np.asarray(self.aligned_scan.points)

                # Use batch measurement for efficiency
                scan_measurements = measure_scan_dimensions_batch(
                    aligned_points, cad_result.dimensions, search_radius=8.0
                )

                logger.info(f"Measured {len(scan_measurements)}/{cad_result.total_dimensions} "
                           "dimensions in scan")

            # Build report in the same schema as XLSX-based DimensionReport.to_dict()
            status_map = {
                "PASS": "pass",
                "FAIL": "fail",
                "WARNING": "warning",
                "NOT_MEASURED": "not_measured",
                "PENDING": "pending",
            }
            dimension_comparisons = []
            for m in scan_measurements:
                raw_status = str(m.get("status", "pending")).upper()
                status = status_map.get(raw_status, raw_status.lower())
                cad_value = m.get("cad_value")
                scan_value = m.get("scan_value")
                scan_deviation = m.get("deviation")

                scan_dev_pct = None
                if (
                    cad_value is not None and cad_value != 0
                    and scan_deviation is not None
                ):
                    scan_dev_pct = (scan_deviation / cad_value) * 100.0

                tol = float(m.get("tolerance", 0.5))
                dimension_comparisons.append({
                    "dim_id": m["dim_id"],
                    "type": m["type"],
                    "description": m["description"],
                    "expected": cad_value,
                    "reference_source": "CAD",
                    "xlsx_value": None,
                    "cad_value": cad_value,
                    "scan_value": scan_value,
                    "measured_output": scan_value,
                    "unit": "mm",
                    "tolerance": f"+{tol}/-{tol}",
                    "tolerance_plus": tol,
                    "tolerance_minus": tol,
                    "xlsx_vs_cad_deviation": None,
                    "scan_vs_cad_deviation": scan_deviation,
                    "scan_deviation": scan_deviation,
                    "scan_deviation_percent": scan_dev_pct,
                    "status": status,
                    "measurement_axis": m.get("measurement_axis", ""),
                    "confidence": m.get("confidence", 1.0),
                })

            measured_count = sum(
                1 for d in dimension_comparisons if d["status"] != "not_measured"
            )
            pass_count = sum(1 for d in dimension_comparisons if d["status"] == "pass")
            fail_count = sum(1 for d in dimension_comparisons if d["status"] == "fail")
            warning_count = sum(1 for d in dimension_comparisons if d["status"] == "warning")
            pass_rate = (pass_count / measured_count * 100.0) if measured_count > 0 else 0.0

            # Get failed dimensions
            failed_dimensions = [d for d in dimension_comparisons if d["status"] == "fail"]

            # Get worst deviations (sorted by absolute deviation)
            worst_deviations = sorted(
                dimension_comparisons,
                key=lambda d: abs(d["scan_deviation"]) if d["scan_deviation"] is not None else 0,
                reverse=True
            )[:10]

            from datetime import datetime

            report = {
                "source": "cad_extraction",
                "metadata": {
                    "part_id": part_id,
                    "part_name": part_name,
                    "timestamp": datetime.now().isoformat(),
                },
                "summary": {
                    "total_dimensions": cad_result.total_dimensions,
                    "measured": measured_count,
                    "passed": pass_count,
                    "failed": fail_count,
                    "warnings": warning_count + len(cad_result.warnings),
                    "pass_rate": round(pass_rate, 1),
                },
                "bend_summary": {
                    "total_bends": 0,
                    "passed": 0,
                    "failed": 0,
                },
                "dimensions": dimension_comparisons,
                "bend_analysis": None,
                "failed_dimensions": failed_dimensions,
                "worst_deviations": worst_deviations,
                "bounding_box": cad_result.bounding_box,
                "extraction_summary": cad_result.extraction_summary,
                "cad_dimensions": [d.to_dict() for d in cad_result.dimensions],
                "processing_time_ms": cad_result.processing_time_ms,
            }

            self._update_progress("dimension_analysis", 95, "Dimension analysis complete")

            return report

        except Exception as e:
            logger.error(f"CAD dimension analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_feature_dimension_analysis(
        self,
        xlsx_path: str,
        part_id: str = "",
        part_name: str = "",
    ) -> Optional[Dict]:
        """
        Complete feature-based dimension analysis with three-way comparison.

        This implements the feature detection -> matching -> measurement pipeline:
        1. Parse XLSX -> DimensionSpec list
        2. Extract CAD features -> FeatureRegistry
        3. Match XLSX to CAD features -> FeatureMatch list
        4. Measure each feature in scan
        5. Compute deviations and status
        6. Generate ThreeWayReport

        This method enables accurate comparison of:
        - XLSX specification (engineering spec)
        - CAD design (nominal geometry)
        - Scan actual (measured part)

        Args:
            xlsx_path: Path to XLSX dimension specification file
            part_id: Part identifier
            part_name: Part name

        Returns:
            ThreeWayReport as dict, or None on failure
        """
        try:
            from dimension_parser import parse_dimension_file
            from feature_detection import (
                CADFeatureExtractor,
                FeatureMatcher,
                ScanFeatureMeasurer,
                ThreeWayReport,
            )
            import time

            start_time = time.time()

            self._update_progress("feature_analysis", 85, "Parsing dimension specifications...")

            # Step 1: Parse XLSX
            xlsx_result = parse_dimension_file(xlsx_path)
            if not xlsx_result.success:
                logger.error(f"Failed to parse XLSX: {xlsx_result.error}")
                return None

            logger.info(f"Parsed {len(xlsx_result.dimensions)} dimensions from XLSX")

            # Step 2: Extract CAD features
            self._update_progress("feature_analysis", 87, "Extracting features from CAD mesh...")

            if self.reference_mesh is None:
                logger.error("No CAD reference mesh available")
                return None

            extractor = CADFeatureExtractor()
            cad_features = extractor.extract(self.reference_mesh)

            logger.info(
                f"Extracted CAD features: {len(cad_features.holes)} holes, "
                f"{len(cad_features.linear_dims)} linear, "
                f"{len(cad_features.angles)} angles"
            )

            # Step 3: Match XLSX to CAD features
            self._update_progress("feature_analysis", 90, "Matching dimensions to CAD features...")

            matcher = FeatureMatcher()
            matches, unmatched_xlsx, unmatched_cad = matcher.match(
                xlsx_result.dimensions,
                cad_features
            )

            logger.info(
                f"Feature matching: {len(matches)} matched, "
                f"{len(unmatched_xlsx)} unmatched XLSX, "
                f"{len(unmatched_cad)} unmatched CAD"
            )

            # Step 4: Measure in scan
            self._update_progress("feature_analysis", 93, "Measuring features in scan...")

            if self.aligned_scan is not None:
                scan_points = np.asarray(self.aligned_scan.points)

                measurer = ScanFeatureMeasurer()
                matches = measurer.measure_all(
                    scan_points,
                    matches,
                    deviations=self.deviations
                )

                measured_count = sum(1 for m in matches if m.scan_value is not None)
                logger.info(f"Measured {measured_count}/{len(matches)} features in scan")
            else:
                logger.warning("No aligned scan available for measurement")

            # Step 5: Compute status (already done in measure_all via compute_status)
            # Step 6: Generate report
            self._update_progress("feature_analysis", 95, "Generating three-way comparison report...")

            processing_time = (time.time() - start_time) * 1000

            report = ThreeWayReport(
                part_id=part_id or part_name,
                matches=matches,
                unmatched_xlsx=unmatched_xlsx,
                unmatched_cad=unmatched_cad,
                processing_time_ms=processing_time,
            )
            report.compute_summary()

            logger.info(
                f"Feature dimension analysis complete: "
                f"{report.pass_count}/{report.measured_count} passed "
                f"({processing_time:.0f}ms)"
            )

            # Log table for debugging
            logger.info("\n" + report.to_table_string())

            return report.to_dict()

        except ImportError as e:
            logger.warning(f"Feature detection modules not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Feature dimension analysis failed: {e}")
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

            # Step 2: Run multi-model orchestrator
            self._update_progress("enhanced", 90, "Running multi-model AI analysis...")
            orchestrator = MultiModelOrchestrator()

            result = await orchestrator.analyze(
                job_id=job_id,
                points=points,
                deviations=deviations,
                bend_detection_result=bend_result,
                drawing_path=drawing_path,
                part_info=part_info,
                regional_analysis={},
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

        self.reference_mesh = None
        self.reference_pcd = None
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
            if len(self.reference_mesh.triangles) == 0:
                raise ValueError(
                    f"Converted CAD has no triangles: {filepath.name}. "
                    "Check CAD export settings/tessellation."
                )

            self._update_progress("load", 15,
                f"Converted CAD: {result.num_faces} faces → {len(self.reference_mesh.vertices)} vertices")
        else:
            # Direct mesh formats (STL, OBJ, PLY)
            mesh = self.o3d.io.read_triangle_mesh(str(filepath))
            mesh_vertices = len(mesh.vertices)
            mesh_triangles = len(mesh.triangles)

            if mesh_triangles > 0 and mesh_vertices > 0:
                self.reference_mesh = mesh
                if not self.reference_mesh.has_vertex_normals():
                    self.reference_mesh.compute_vertex_normals()
                self._update_progress(
                    "load",
                    15,
                    f"Loaded reference mesh: {mesh_vertices} vertices, {mesh_triangles} triangles",
                )
            else:
                # Fallback for point-cloud-like references (e.g. PLY without faces).
                pcd = self.o3d.io.read_point_cloud(str(filepath))
                pcd_points = len(pcd.points)
                if pcd_points <= 0:
                    raise ValueError(
                        f"Unable to parse reference geometry: {filepath.name}. "
                        f"Mesh vertices={mesh_vertices}, triangles={mesh_triangles}, point_cloud_points={pcd_points}. "
                        "File may be empty/corrupt or extension/content mismatch."
                    )
                if not pcd.has_normals():
                    pcd.estimate_normals()
                self.reference_pcd = pcd
                self._update_progress(
                    "load",
                    15,
                    f"Loaded reference point cloud: {pcd_points} points (mesh triangles unavailable)",
                )

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
            mesh_vertices = len(mesh.vertices)
            mesh_triangles = len(mesh.triangles)
            if mesh_triangles > 0 and mesh_vertices > 0:
                self.scan_pcd = mesh.sample_points_uniformly(number_of_points=200000)
            else:
                pcd = self.o3d.io.read_point_cloud(str(filepath))
                if len(pcd.points) <= 0:
                    raise ValueError(
                        f"Unable to parse scan geometry: {filepath.name}. "
                        f"Mesh vertices={mesh_vertices}, triangles={mesh_triangles}, point_cloud_points={len(pcd.points)}. "
                        "File may be empty/corrupt or extension/content mismatch."
                    )
                self.scan_pcd = pcd
        else:
            raise ValueError(f"Unsupported format: {ext}")

        if len(self.scan_pcd.points) <= 0:
            raise ValueError(f"Scan has zero points after loading: {filepath.name}")

        import copy
        self.raw_scan_pcd = copy.deepcopy(self.scan_pcd)

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

    def _set_alignment_random_seed(self, random_seed: Optional[int]) -> None:
        """Seed numpy/Python/Open3D RNGs so alignment retries are reproducible."""
        if random_seed is None:
            return
        try:
            random.seed(int(random_seed))
        except Exception:
            pass
        try:
            np.random.seed(int(random_seed))
        except Exception:
            pass
        try:
            if hasattr(self.o3d.utility, "random") and hasattr(self.o3d.utility.random, "seed"):
                self.o3d.utility.random.seed(int(random_seed))
        except Exception:
            logger.debug("Failed to seed Open3D RNG", exc_info=True)

    def align(self, auto_scale: bool = True, tolerance: float = 1.0, random_seed: Optional[int] = None) -> tuple:
        """ISO 5459-compliant scan-to-CAD alignment pipeline.

        Pipeline:
        1. PCA pre-alignment (resolve arbitrary orientation)
        2. Auto-scaling (detect unit mismatch only)
        3. Flip hypothesis search (resolve PCA 180° ambiguity)
        4. Fine ICP (point-to-plane with Tukey robust kernel)
        5. RANSAC datum alignment (ISO 5459 3-2-1)
        6. Apply final transformation
        """
        import copy

        self._set_alignment_random_seed(random_seed)
        self.alignment_metadata = AlignmentMetadata()
        ref_pcd = self._prepare_reference()
        scan_working, ref_centered, scan_center, ref_center = self._center_clouds(ref_pcd)
        pca_rotation = self._pca_pre_align(scan_working, ref_centered)
        self._auto_scale(scan_working, ref_centered, auto_scale)
        best_flip, best_icp_init = self._search_flip_hypotheses(scan_working, ref_centered)
        self._apply_flip(scan_working, best_flip)
        reg_fine = self._fine_icp(scan_working, ref_centered, best_icp_init)
        final_transform, final_fitness, final_rmse = self._ransac_datum_alignment(
            scan_working, ref_centered, reg_fine, tolerance)
        self._apply_final_transform(
            scan_center, ref_center, pca_rotation, best_flip, final_transform, auto_scale)

        self._update_progress("align", 70, f"Alignment complete. Fitness: {final_fitness:.2%}")
        logger.info(f"Final alignment: fitness={final_fitness:.2%}, RMSE={final_rmse:.3f}mm")

        return final_fitness, final_rmse

    # --- Alignment sub-methods ---

    def _prepare_reference(self):
        """Sample reference mesh uniformly and estimate normals."""
        self._update_progress("align", 50, "Converting reference to point cloud...")
        n_points = max(100_000, len(self.scan_pcd.points))
        if self.reference_mesh is not None and len(self.reference_mesh.triangles) > 0:
            ref_pcd = self.reference_mesh.sample_points_uniformly(number_of_points=n_points)
        elif self.reference_pcd is not None and len(self.reference_pcd.points) > 0:
            import copy
            ref_pcd = copy.deepcopy(self.reference_pcd)
        else:
            raise ValueError(
                "Reference geometry is unavailable for alignment. "
                "Provide a valid mesh (with triangles) or point cloud (with points)."
            )
        if len(ref_pcd.points) <= 0:
            raise ValueError("Reference point cloud is empty after preparation")
        ref_pcd.estimate_normals()
        return ref_pcd

    def _center_clouds(self, ref_pcd):
        """Center both clouds at origin. Returns working copies and original centers."""
        import copy
        self._update_progress("align", 52, "Centering point clouds...")
        scan_center = np.asarray(self.scan_pcd.get_center())
        ref_center = np.asarray(ref_pcd.get_center())

        scan_working = copy.deepcopy(self.scan_pcd)
        scan_points = np.asarray(scan_working.points)
        scan_working.points = self.o3d.utility.Vector3dVector(scan_points - scan_center)
        ref_centered = copy.deepcopy(ref_pcd)
        ref_points = np.asarray(ref_centered.points)
        ref_centered.points = self.o3d.utility.Vector3dVector(ref_points - ref_center)

        return scan_working, ref_centered, scan_center, ref_center

    def _pca_pre_align(self, scan_working, ref_centered):
        """Apply PCA rotation to align scan principal axes with reference. Returns rotation matrix."""
        self._update_progress("align", 54, "Computing PCA alignment...")
        scan_points = np.asarray(scan_working.points)
        ref_points = np.asarray(ref_centered.points)

        scan_pca = self._compute_pca_rotation(scan_points)
        ref_pca = self._compute_pca_rotation(ref_points)

        # R_align maps scan PCA space → reference PCA space
        pca_rotation = ref_pca.T @ scan_pca

        # Apply in-place via numpy (no deep copy)
        rotated = scan_points @ pca_rotation.T
        scan_working.points = self.o3d.utility.Vector3dVector(rotated)

        return pca_rotation

    def _auto_scale(self, scan_working, ref_centered, auto_scale):
        """Detect and apply uniform scaling for unit mismatch (e.g. mm vs inches).

        Only scales if per-axis ratios are consistent (CV < 0.03) AND significantly
        different from 1.0 (> 5%). Non-uniform ratios indicate genuine manufacturing
        deviations and are NOT scaled.
        """
        self.scale_factor = 1.0

        if not auto_scale:
            return

        self._update_progress("align", 56, "Computing scale factors...")
        scan_bbox = scan_working.get_axis_aligned_bounding_box()
        ref_bbox = ref_centered.get_axis_aligned_bounding_box()
        scan_size = np.asarray(scan_bbox.max_bound) - np.asarray(scan_bbox.min_bound)
        ref_size = np.asarray(ref_bbox.max_bound) - np.asarray(ref_bbox.min_bound)

        scale_ratio = ref_size / np.maximum(scan_size, 1e-6)
        geo_mean = float(np.cbrt(np.prod(scale_ratio)))

        scale_cv = np.std(scale_ratio) / np.mean(scale_ratio) if np.mean(scale_ratio) > 0 else 0
        scale_deviation_from_1 = abs(geo_mean - 1.0)

        if scale_cv < 0.03 and scale_deviation_from_1 > 0.05:
            self.scale_factor = geo_mean
            logger.info(f"Auto-scaling: uniform factor {geo_mean:.4f} (consistent across axes, CV={scale_cv:.3f})")
            pts = np.asarray(scan_working.points)
            scan_working.points = self.o3d.utility.Vector3dVector(pts * self.scale_factor)
        else:
            logger.info(f"Auto-scaling: SKIPPED — per-axis ratios are non-uniform "
                       f"(X={scale_ratio[0]:.4f}, Y={scale_ratio[1]:.4f}, Z={scale_ratio[2]:.4f}, "
                       f"CV={scale_cv:.3f}, geo_mean={geo_mean:.4f}). "
                       f"Size differences are genuine manufacturing deviations.")

    def _search_flip_hypotheses(self, scan_working, ref_centered):
        """Test 8 PCA flip variants to resolve 180° ambiguity. Returns (best_flip, best_icp_transform).

        Uses numpy array operations instead of deep copies for memory efficiency.
        """
        self._update_progress("align", 58, "Testing rotation hypotheses...")

        flip_matrices = [
            np.eye(3),
            np.diag([-1, -1, 1]),  np.diag([-1, 1, -1]), np.diag([1, -1, -1]),
            np.diag([-1, 1, 1]),   np.diag([1, -1, 1]),   np.diag([1, 1, -1]),
            np.diag([-1, -1, -1]),
        ]

        base_points = np.asarray(scan_working.points).copy()
        best_fitness = 0
        best_transform = np.eye(4)
        best_flip = np.eye(3)

        for flip in flip_matrices:
            rotated_pts = base_points @ flip.T
            test_pcd = self.o3d.geometry.PointCloud()
            test_pcd.points = self.o3d.utility.Vector3dVector(rotated_pts)
            test_pcd.estimate_normals()

            reg = self.o3d.pipelines.registration.registration_icp(
                test_pcd, ref_centered, 20.0, np.eye(4),
                self.o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )

            if reg.fitness > best_fitness:
                best_fitness = reg.fitness
                best_transform = reg.transformation
                best_flip = flip

        logger.info(f"Best pre-alignment fitness: {best_fitness:.2%}")
        return best_flip, best_transform

    def _apply_flip(self, scan_working, flip):
        """Apply flip rotation to scan_working in-place and estimate normals."""
        pts = np.asarray(scan_working.points)
        scan_working.points = self.o3d.utility.Vector3dVector(pts @ flip.T)
        scan_working.estimate_normals()

    def _fine_icp(self, scan_working, ref_centered, initial_transform):
        """Point-to-plane ICP with Tukey robust kernel for outlier rejection.

        The Tukey biweight function completely rejects outliers beyond 3*k,
        matching the approach used by GOM ATOS and PolyWorks internally.
        """
        self._update_progress("align", 62, "Running fine alignment (ICP)...")
        
        loss = self.o3d.pipelines.registration.TukeyLoss(k=2.0)
        estimation = self.o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        criteria = self.o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7,
            relative_rmse=1e-7,
            max_iteration=100
        )

        reg = self.o3d.pipelines.registration.registration_icp(
            scan_working, ref_centered, 5.0, initial_transform, estimation, criteria
        )

        logger.info(f"Initial fine ICP: fitness={reg.fitness:.2%}, RMSE={reg.inlier_rmse:.3f}mm")
        return reg

    def _ransac_datum_alignment(self, scan_working, ref_centered, reg_fine, tolerance: float = 1.0):
        """RANSAC-based datum alignment per ISO 5459.

        1. Sequential RANSAC extracts planes from reference
        2. Rank by spatial extent (bounding box area), not point count
        3. Select primary datum (largest area) + secondary (largest non-parallel)
        4. Run datum-constrained ICP with Tukey robust kernel
        5. Evaluate vs. global ICP on primary datum surface
        """
        self._update_progress("align", 64, "Refining alignment (RANSAC datum)...")

        best_transform = reg_fine.transformation.copy()
        best_fitness = reg_fine.fitness
        best_rmse = reg_fine.inlier_rmse
        self.alignment_metadata = AlignmentMetadata(
            source="GLOBAL_ICP",
            parent_island_id="NONE",
            confidence=float(best_fitness or 0.0),
            abstained=False,
            conditioning={
                "global_fitness": round(float(best_fitness or 0.0), 4),
                "global_rmse_mm": round(float(best_rmse or 0.0), 4),
            },
        )

        try:
            from scipy.spatial import cKDTree

            ref_planes = self._extract_planes_ransac(ref_centered)
            if not ref_planes:
                return best_transform, best_fitness, best_rmse

            primary, secondary = self._select_datums(ref_planes)
            datum_ref_pts = self._build_datum_point_set(primary, secondary)
            datum_scan_pts = self._find_datum_scan_points(
                scan_working, datum_ref_pts, best_transform)

            datum_transform, datum_fitness, datum_rmse = self._datum_icp(
                datum_scan_pts, datum_ref_pts, best_transform)

            use_datum, datum_stats = self._evaluate_datum_vs_global(
                scan_working, primary, best_transform, datum_transform, tolerance)

            if use_datum:
                self.alignment_metadata = AlignmentMetadata(
                    source="PLANE_DATUM_REFINEMENT",
                    parent_island_id="PRIMARY_PLANE",
                    confidence=float(
                        min(
                            1.0,
                            max(
                                datum_fitness,
                                1.0 / (1.0 + max(float(datum_stats.get("median_datum", 999.0)), 0.0)),
                            ),
                        )
                    ),
                    abstained=False,
                    conditioning={
                        "primary_plane_area": round(float(primary.get("bbox_area") or 0.0), 3),
                        "secondary_plane_present": bool(secondary is not None),
                        "datum_scan_point_count": int(datum_stats.get("datum_scan_point_count") or 0),
                        "median_primary_datum_residual_mm": round(float(datum_stats.get("median_datum") or 0.0), 4),
                        "p90_primary_datum_residual_mm": round(float(datum_stats.get("p90_datum") or 0.0), 4),
                        "global_median_primary_datum_residual_mm": round(float(datum_stats.get("median_global") or 0.0), 4),
                        "global_p90_primary_datum_residual_mm": round(float(datum_stats.get("p90_global") or 0.0), 4),
                        "global_fitness": round(float(best_fitness or 0.0), 4),
                        "datum_fitness": round(float(datum_fitness or 0.0), 4),
                    },
                )
                return datum_transform, datum_fitness, datum_rmse
            self.alignment_metadata = AlignmentMetadata(
                source="GLOBAL_ICP",
                parent_island_id="NONE",
                confidence=float(best_fitness or 0.0),
                abstained=False,
                conditioning={
                    "primary_plane_area": round(float(primary.get("bbox_area") or 0.0), 3),
                    "secondary_plane_present": bool(secondary is not None),
                    "datum_scan_point_count": int(datum_stats.get("datum_scan_point_count") or 0),
                    "median_primary_datum_residual_mm": round(float(datum_stats.get("median_datum") or 0.0), 4),
                    "p90_primary_datum_residual_mm": round(float(datum_stats.get("p90_datum") or 0.0), 4),
                    "global_median_primary_datum_residual_mm": round(float(datum_stats.get("median_global") or 0.0), 4),
                    "global_p90_primary_datum_residual_mm": round(float(datum_stats.get("p90_global") or 0.0), 4),
                    "global_fitness": round(float(best_fitness or 0.0), 4),
                    "datum_candidate_fitness": round(float(datum_fitness or 0.0), 4),
                },
            )

        except Exception as e:
            logger.warning(f"RANSAC datum alignment failed: {e}, using initial ICP")
            import traceback
            logger.debug(traceback.format_exc())
            self.alignment_metadata = AlignmentMetadata(
                source="GLOBAL_ICP",
                parent_island_id="NONE",
                confidence=float(best_fitness or 0.0),
                abstained=True,
                conditioning={
                    "global_fitness": round(float(best_fitness or 0.0), 4),
                    "global_rmse_mm": round(float(best_rmse or 0.0), 4),
                    "reason": str(e),
                },
            )

        return best_transform, best_fitness, best_rmse

    def _extract_planes_ransac(self, ref_pcd, max_planes=6, distance_threshold=0.15):
        """Sequential RANSAC plane extraction from reference point cloud.

        distance_threshold: 0.15mm (2-3x typical structured-light scanner noise of 0.05mm)
        min_inliers: max(500, 2% of total points) to reject spurious micro-planes
        """
        import copy

        total_points = len(np.asarray(ref_pcd.points))
        min_inliers = max(500, int(0.02 * total_points))

        remaining = copy.deepcopy(ref_pcd)
        planes = []

        for i in range(max_planes):
            pts = np.asarray(remaining.points)
            if len(pts) < min_inliers:
                break

            model, inliers = remaining.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=2000  # p=0.999, w=0.3 → N=254, with margin
            )

            if len(inliers) < min_inliers:
                break

            normal = np.array(model[:3])
            normal = normal / np.linalg.norm(normal)
            inlier_pts = pts[inliers]
            bbox_area = self._compute_plane_extent(inlier_pts)

            planes.append({
                'normal': normal,
                'd': model[3],
                'inlier_pts': inlier_pts,
                'n_points': len(inliers),
                'bbox_area': bbox_area,
                'centroid': inlier_pts.mean(axis=0),
            })

            logger.info(f"  RANSAC plane {i}: normal=[{normal[0]:.4f}, {normal[1]:.4f}, "
                       f"{normal[2]:.4f}], pts={len(inliers)}, area={bbox_area:.0f}mm²")

            # Remove inliers for next iteration
            mask = np.ones(len(pts), dtype=bool)
            mask[inliers] = False
            remaining = remaining.select_by_index(np.where(mask)[0])

        return planes

    def _compute_plane_extent(self, points):
        """Compute bounding box area of points projected onto their best-fit plane.

        Uses PCA to find the two directions of maximum spread.
        Eigenvalues and eigenvectors are sorted together (fixes B1 desync bug).
        """
        centroid = points.mean(axis=0)
        centered = points - centroid

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort BOTH eigenvalues and eigenvectors together (largest first)
        sort_idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sort_idx]

        # The two largest eigenvectors span the plane
        proj_1 = centered @ eigvecs[:, 0]
        proj_2 = centered @ eigvecs[:, 1]

        return float(np.ptp(proj_1) * np.ptp(proj_2))

    def _select_datums(self, planes):
        """Select primary and secondary datums per ISO 5459.

        Primary: largest spatial extent (bounding box area).
        Secondary: largest non-parallel plane (>30° from primary normal).
        """
        planes.sort(key=lambda p: p['bbox_area'], reverse=True)
        primary = planes[0]
        logger.info(f"  Primary datum: area={primary['bbox_area']:.0f}mm², "
                   f"normal=[{primary['normal'][0]:.4f}, {primary['normal'][1]:.4f}, "
                   f"{primary['normal'][2]:.4f}]")

        secondary = None
        for p in planes[1:]:
            cos_angle = abs(np.dot(p['normal'], primary['normal']))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
            if angle_deg > 30:
                secondary = p
                logger.info(f"  Secondary datum: area={p['bbox_area']:.0f}mm², "
                           f"angle_from_primary={angle_deg:.1f}°")
                break

        return primary, secondary

    def _build_datum_point_set(self, primary, secondary):
        """Build combined datum reference point set from primary (+ optional secondary)."""
        datum_pts = primary['inlier_pts'].copy()
        if secondary is not None:
            datum_pts = np.vstack([datum_pts, secondary['inlier_pts']])
        return datum_pts

    def _find_datum_scan_points(self, scan_working, datum_ref_pts, transform):
        """Find scan points near datum surfaces (within 5mm after alignment)."""
        from scipy.spatial import cKDTree

        # Apply transform via numpy (no deep copy needed)
        scan_pts = np.asarray(scan_working.points)
        R, t = transform[:3, :3], transform[:3, 3]
        scan_pts_aligned = scan_pts @ R.T + t

        datum_tree = cKDTree(datum_ref_pts)
        dists, _ = datum_tree.query(scan_pts_aligned, k=1)

        datum_mask = dists < 5.0
        datum_count = int(np.sum(datum_mask))
        logger.info(f"  Scan points near datum surfaces: {datum_count} "
                   f"({100*datum_count/len(scan_pts):.1f}%)")

        if datum_count < 500:
            raise ValueError(f"Too few scan points ({datum_count}) near datum surfaces")

        return scan_pts[datum_mask]

    def _datum_icp(self, datum_scan_pts, datum_ref_pts, initial_transform):
        """Run point-to-plane ICP on datum points with Tukey robust kernel."""
        datum_ref_pcd = self.o3d.geometry.PointCloud()
        datum_ref_pcd.points = self.o3d.utility.Vector3dVector(datum_ref_pts)
        datum_ref_pcd.estimate_normals()

        datum_scan_pcd = self.o3d.geometry.PointCloud()
        datum_scan_pcd.points = self.o3d.utility.Vector3dVector(datum_scan_pts)
        datum_scan_pcd.estimate_normals()

        loss = self.o3d.pipelines.registration.TukeyLoss(k=2.0)
        estimation = self.o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        # Coarse pass
        reg = self.o3d.pipelines.registration.registration_icp(
            datum_scan_pcd, datum_ref_pcd, 5.0, initial_transform,
            estimation,
            self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )

        logger.info(f"  Datum ICP: fitness={reg.fitness:.2%}, RMSE={reg.inlier_rmse:.3f}mm")

        # Tight refinement
        reg_tight = self.o3d.pipelines.registration.registration_icp(
            datum_scan_pcd, datum_ref_pcd, 2.0, reg.transformation,
            estimation,
            self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        if reg_tight.fitness > 0.3:
            logger.info(f"  → Tight pass: fitness={reg_tight.fitness:.2%}, RMSE={reg_tight.inlier_rmse:.3f}mm")
            return reg_tight.transformation, reg_tight.fitness, reg_tight.inlier_rmse

        return reg.transformation, reg.fitness, reg.inlier_rmse

    def _evaluate_datum_vs_global(self, scan_working, primary, global_transform, datum_transform,
                                   tolerance: float = 1.0):
        """Compare datum vs global alignment quality on primary datum surface.

        Uses numpy transforms (no deep copies) for memory efficiency.
        Returns (should_use_datum, stats).

        Criterion: datum alignment must be within 20% of global median AND
        below 3x tolerance on the primary datum surface.
        """
        from scipy.spatial import cKDTree

        primary_tree = cKDTree(primary['inlier_pts'])
        scan_pts = np.asarray(scan_working.points)

        def apply_4x4(pts, T):
            return pts @ T[:3, :3].T + T[:3, 3]

        pts_global = apply_4x4(scan_pts, global_transform)
        dists_global, _ = primary_tree.query(pts_global, k=1)
        near_g = dists_global < 10.0
        median_global = np.median(dists_global[near_g]) if near_g.sum() > 0 else 999
        p90_global = np.percentile(dists_global[near_g], 90) if near_g.sum() > 0 else 999

        pts_datum = apply_4x4(scan_pts, datum_transform)
        dists_datum, _ = primary_tree.query(pts_datum, k=1)
        near_d = dists_datum < 10.0
        median_datum = np.median(dists_datum[near_d]) if near_d.sum() > 0 else 999
        p90_datum = np.percentile(dists_datum[near_d], 90) if near_d.sum() > 0 else 999

        mean_global = float(np.mean(dists_global[near_g])) if near_g.sum() > 0 else 999
        mean_datum = float(np.mean(dists_datum[near_d])) if near_d.sum() > 0 else 999

        logger.info(f"  Primary datum evaluation (ref plane inliers only):")
        logger.info(f"    Original ICP:      median={median_global:.3f}mm, p90={p90_global:.3f}mm, "
                   f"mean={mean_global:+.3f}mm")
        logger.info(f"    Plane-aligned ICP: median={median_datum:.3f}mm, p90={p90_datum:.3f}mm, "
                   f"mean={mean_datum:+.3f}mm")

        # Tolerance-relative criterion:
        # Use datum if it's within 20% of global AND below 3x tolerance
        # Also require p90 not be dramatically worse
        use_datum = (median_datum < median_global * 1.2
                     and median_datum < tolerance * 3.0
                     and p90_datum < p90_global * 1.5)
        if use_datum:
            logger.info(f"  → Using PLANE-ALIGNED ICP (better on primary datum)")
        else:
            logger.info(f"  → Keeping ORIGINAL ICP (datum alignment not sufficiently better)")

        return use_datum, {
            "median_global": float(median_global),
            "p90_global": float(p90_global),
            "median_datum": float(median_datum),
            "p90_datum": float(p90_datum),
            "mean_global": float(mean_global),
            "mean_datum": float(mean_datum),
            "datum_scan_point_count": int(near_d.sum()),
        }

    def _apply_final_transform(self, scan_center, ref_center, pca_rotation,
                               flip, icp_transform, auto_scale):
        """Apply the complete transformation pipeline to the original scan.

        Composes: center → PCA → scale → flip → ICP → re-center
        into a single matrix multiplication for numerical precision.
        """
        import copy
        self._update_progress("align", 68, "Applying transformation...")

        R_icp = icp_transform[:3, :3]
        t_icp = icp_transform[:3, 3]

        # Composed rotation: ICP_R @ flip @ PCA
        R_composed = R_icp @ flip @ pca_rotation
        scale = self.scale_factor if auto_scale else 1.0

        # p' = scale * R_composed @ (p - scan_center) + t_icp + ref_center
        # = scale * R_composed @ p + (- scale * R_composed @ scan_center + t_icp + ref_center)
        self.aligned_scan = copy.deepcopy(self.scan_pcd)
        points = np.asarray(self.aligned_scan.points)

        t_combined = -scale * (R_composed @ scan_center) + t_icp + ref_center
        transformed = scale * (points @ R_composed.T) + t_combined
        self.aligned_scan.points = self.o3d.utility.Vector3dVector(transformed)

        self.aligned_scan_raw = None
        if self.raw_scan_pcd is not None and len(self.raw_scan_pcd.points) > 0:
            self.aligned_scan_raw = copy.deepcopy(self.raw_scan_pcd)
            raw_points = np.asarray(self.aligned_scan_raw.points)
            raw_transformed = scale * (raw_points @ R_composed.T) + t_combined
            self.aligned_scan_raw.points = self.o3d.utility.Vector3dVector(raw_transformed)
    
    def compute_deviations(self, chunk_size: int = 10000, use_trimesh: bool = True) -> np.ndarray:
        """Compute point-to-mesh distances with chunked processing for memory efficiency.

        Args:
            chunk_size: Number of points to process at once (default 10000)
            use_trimesh: Use Trimesh BVH (geometrically exact) vs KDTree (faster but ~0.5mm error).
                         Defaults to True for accuracy — Trimesh computes true closest point on
                         the mesh surface, while KDTree uses sampled points as approximation.
        """
        self._update_progress("analyze", 75, "Computing deviations...")

        scan_points = np.asarray(self.aligned_scan.points)
        n_points = len(scan_points)

        if use_trimesh:
            # Trimesh BVH: exact closest-point-on-mesh (preferred for accuracy)
            try:
                return self._compute_deviations_trimesh(scan_points, n_points, chunk_size)
            except Exception as e:
                logger.warning(f"Trimesh deviation computation failed, falling back to KDTree: {e}")
                return self._compute_deviations_kdtree(scan_points, n_points, chunk_size)
        else:
            # KDTree: approximate (sampled points), ~sqrt(area/N) error
            return self._compute_deviations_kdtree(scan_points, n_points, chunk_size)

    def _compute_deviations_kdtree(self, scan_points: np.ndarray, n_points: int, chunk_size: int) -> np.ndarray:
        """Memory-efficient deviation computation using scipy cKDTree."""
        from scipy.spatial import cKDTree

        self._update_progress("analyze", 76, "Sampling reference surface...")

        # Sample dense points from reference geometry for KDTree.
        if self.reference_mesh is not None and len(self.reference_mesh.triangles) > 0:
            ref_sample = self.reference_mesh.sample_points_uniformly(
                number_of_points=max(200000, n_points * 2)
            )
            ref_sample.estimate_normals()
            ref_points = np.asarray(ref_sample.points)
            ref_normals = np.asarray(ref_sample.normals)
        elif self.reference_pcd is not None and len(self.reference_pcd.points) > 0:
            ref_sample = self.reference_pcd
            if not ref_sample.has_normals():
                ref_sample.estimate_normals()
            ref_points = np.asarray(ref_sample.points)
            ref_normals = np.asarray(ref_sample.normals)
            # Keep KDTree memory bounded for very dense scans.
            target = max(200000, n_points * 2)
            if len(ref_points) > target:
                idx = np.random.choice(len(ref_points), size=target, replace=False)
                ref_points = ref_points[idx]
                if len(ref_normals) == len(self.reference_pcd.points):
                    ref_normals = ref_normals[idx]
        else:
            raise ValueError(
                "Reference geometry unavailable for deviation computation. "
                "No mesh triangles and no reference point cloud."
            )

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
        if self.reference_mesh is None or len(self.reference_mesh.triangles) == 0:
            return self._compute_deviations_kdtree(scan_points, n_points, chunk_size)
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

    def generate_bend_snapshots(
        self,
        tolerance: float,
        output_dir: str,
        bend_results: List[Dict],
        reference_mesh_path: str = None,
        angle_tolerance: float = 2.0,
    ) -> Dict[str, str]:
        """Generate 3D snapshots with bend markers and deviation labels.

        Args:
            tolerance: Distance tolerance in mm
            output_dir: Directory to save images
            bend_results: List of bend result dicts
            reference_mesh_path: Path to reference mesh PLY
            angle_tolerance: Angle tolerance in degrees

        Returns:
            Dict mapping view name to file path
        """
        if self.aligned_scan is None or self.deviations is None:
            return {}

        if self.snapshot_renderer is None:
            return {}

        if not bend_results:
            return {}

        points = np.asarray(self.aligned_scan.points)
        deviations = self.deviations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        snapshot_paths = {}

        try:
            # Generate multi-view bend analysis
            multi_bytes = self.snapshot_renderer.render_bend_multi_view(
                points, deviations, tolerance, bend_results,
                reference_mesh_path=reference_mesh_path,
                angle_tolerance=angle_tolerance,
            )
            multi_path = output_path / "bend_analysis_multi_view.png"
            with open(multi_path, "wb") as f:
                f.write(multi_bytes)
            snapshot_paths["bend_multi_view"] = str(multi_path)

            # Generate individual bend analysis views
            for view in ["iso", "top", "front"]:
                view_bytes = self.snapshot_renderer.render_bend_analysis_snapshot(
                    points, deviations, tolerance, bend_results,
                    reference_mesh_path=reference_mesh_path,
                    view=view,
                    width=1600,
                    height=1200,
                    angle_tolerance=angle_tolerance,
                )
                view_path = output_path / f"bend_analysis_{view}.png"
                with open(view_path, "wb") as f:
                    f.write(view_bytes)
                snapshot_paths[f"bend_{view}"] = str(view_path)

            logger.info(f"Generated {len(snapshot_paths)} bend analysis images")

        except Exception as e:
            logger.error(f"Failed to generate bend snapshots: {e}")
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
            fitness, rmse = self.align(tolerance=tolerance)
            report.alignment_fitness = fitness
            report.alignment_rmse = rmse

            # Alignment quality gate — reject clearly failed alignments
            MIN_FITNESS_THRESHOLD = 0.3  # 30% overlap minimum
            WARN_FITNESS_THRESHOLD = 0.6  # 60% overlap for warning
            if fitness < MIN_FITNESS_THRESHOLD:
                logger.error(f"Alignment quality too low: fitness={fitness:.2%} (minimum {MIN_FITNESS_THRESHOLD:.0%}). "
                            f"Results would be unreliable. Check scan coverage and part orientation.")
                report.overall_result = QCResult.FAIL
                report.quality_score = 0.0
                report.confidence = 0.0
                report.ai_summary = (
                    f"ALIGNMENT FAILED: ICP alignment fitness of {fitness:.1%} is below the "
                    f"minimum threshold of {MIN_FITNESS_THRESHOLD:.0%}. The scan could not be "
                    f"reliably registered to the reference model. This typically indicates: "
                    f"(1) wrong reference file, (2) insufficient scan coverage, or "
                    f"(3) the part is significantly different from the reference. "
                    f"All deviation measurements are unreliable and should be discarded."
                )
                return report
            elif fitness < WARN_FITNESS_THRESHOLD:
                logger.warning(f"Alignment quality marginal: fitness={fitness:.2%}. "
                              f"Results may have reduced accuracy.")

            # Compute deviations (Trimesh BVH for exact point-to-mesh distance)
            deviations = self.compute_deviations()
            abs_devs = np.abs(deviations)

            # Statistics
            report.total_points = len(deviations)
            report.mean_deviation = float(np.mean(deviations))
            report.max_deviation = float(np.max(np.abs(deviations)))
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
                        # Create point cloud with colors based on deviations (vectorized)
                        abs_devs_color = np.abs(deviations)
                        normalized = np.clip(abs_devs_color / tolerance, 0.0, 1.0)
                        colors = np.full((len(points), 4), 255, dtype=np.uint8)
                        # Green-to-Yellow (normalized < 0.5): R = t*255, G = 255
                        # Yellow-to-Red (normalized >= 0.5): R = 255, G = (1-t)*255
                        low = normalized < 0.5
                        high = ~low
                        t_low = normalized[low] * 2
                        t_high = (normalized[high] - 0.5) * 2
                        colors[low, 0] = (t_low * 255).astype(np.uint8)
                        colors[low, 1] = 255
                        colors[low, 2] = 0
                        colors[high, 0] = 255
                        colors[high, 1] = ((1 - t_high) * 255).astype(np.uint8)
                        colors[high, 2] = 0
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

            # Enhanced Analysis (CAD mesh bend detection + multi-model)
            if enable_enhanced_analysis and ENHANCED_ANALYSIS_AVAILABLE:
                self._update_progress("enhanced", 94, "Running enhanced bend detection...")
                try:
                    # Run dual-approach bend detection (CAD mesh primary, point cloud fallback)
                    bend_result = self.detect_bends()
                    if bend_result:
                        report.bend_detection_result = bend_result
                        report.bend_results = bend_result.get("bends", [])
                        detection_method = bend_result.get("detection_method", "unknown")
                        logger.info(f"Detected {len(report.bend_results)} bends via {detection_method}")

                        # Generate bend analysis snapshots with markers
                        try:
                            self._update_progress("enhanced", 95, "Generating bend analysis images...")
                            bend_snapshot_paths = self.generate_bend_snapshots(
                                tolerance=tolerance,
                                output_dir=output_dir,
                                bend_results=report.bend_results,
                                reference_mesh_path=str(Path(output_dir) / "reference_mesh.ply"),
                            )
                            if bend_snapshot_paths:
                                report.bend_snapshot_paths.update(bend_snapshot_paths)
                                logger.info(f"Generated {len(bend_snapshot_paths)} bend analysis images")
                        except Exception as e:
                            logger.warning(f"Bend snapshot generation failed: {e}")

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
