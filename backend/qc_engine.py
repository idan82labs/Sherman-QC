"""
Scan QC Engine - Core Analysis Module
Handles mesh loading, alignment, deviation computation, and AI analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # Drawing context
    drawing_context: str = ""
    
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
            "drawing_context": self.drawing_context,
        }


class ScanQCEngine:
    """Main QC analysis engine"""
    
    def __init__(self, progress_callback: Callable[[ProgressUpdate], None] = None):
        self.progress_callback = progress_callback or (lambda x: None)
        self.reference_mesh = None
        self.scan_pcd = None
        self.aligned_scan = None
        self.deviations = None
        self._o3d = None
        self._trimesh = None
        
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
        """Load reference STL model"""
        self._update_progress("load", 5, f"Loading reference model: {Path(filepath).name}")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Reference not found: {filepath}")
        
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
    
    def align(self) -> tuple:
        """Align scan to reference using ICP"""
        self._update_progress("align", 50, "Converting reference to point cloud...")
        
        ref_pcd = self.reference_mesh.sample_points_uniformly(
            number_of_points=max(100000, len(self.scan_pcd.points))
        )
        ref_pcd.estimate_normals()
        
        self._update_progress("align", 55, "Running coarse alignment (ICP)...")
        reg_coarse = self.o3d.pipelines.registration.registration_icp(
            self.scan_pcd, ref_pcd, 5.0, np.eye(4),
            self.o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        self._update_progress("align", 65, "Running fine alignment...")
        reg_fine = self.o3d.pipelines.registration.registration_icp(
            self.scan_pcd, ref_pcd, 1.0, reg_coarse.transformation,
            self.o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        self.aligned_scan = self.scan_pcd.transform(reg_fine.transformation)
        
        self._update_progress("align", 70, f"Alignment complete. Fitness: {reg_fine.fitness:.2%}")
        return reg_fine.fitness, reg_fine.inlier_rmse
    
    def compute_deviations(self) -> np.ndarray:
        """Compute point-to-mesh distances"""
        self._update_progress("analyze", 75, "Computing deviations...")
        
        scan_points = np.asarray(self.aligned_scan.points)
        
        mesh = self.trimesh.Trimesh(
            vertices=np.asarray(self.reference_mesh.vertices),
            faces=np.asarray(self.reference_mesh.triangles)
        )
        
        closest_points, distances, triangle_ids = mesh.nearest.on_surface(scan_points)
        
        # Signed distances
        face_normals = mesh.face_normals[triangle_ids]
        vectors = scan_points - closest_points
        signs = np.sign(np.sum(vectors * face_normals, axis=1))
        
        self.deviations = distances * signs
        
        self._update_progress("analyze", 85, f"Computed {len(self.deviations)} deviation values")
        return self.deviations
    
    def analyze_regions(self, tolerance: float) -> List[RegionAnalysis]:
        """Analyze deviations by region"""
        self._update_progress("analyze", 88, "Analyzing regions...")
        
        points = np.asarray(self.aligned_scan.points)
        deviations = self.deviations
        
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        center = (min_pt + max_pt) / 2
        
        regions = []
        region_defs = [
            ("Top Surface", lambda p: p[:, 2] > center[2] + 0.25 * (max_pt[2] - center[2])),
            ("Bottom Surface", lambda p: p[:, 2] < center[2] - 0.25 * (max_pt[2] - center[2])),
            ("Front Edge", lambda p: p[:, 1] > center[1] + 0.3 * (max_pt[1] - center[1])),
            ("Rear Edge", lambda p: p[:, 1] < center[1] - 0.3 * (max_pt[1] - center[1])),
            ("Left Side", lambda p: p[:, 0] < center[0] - 0.25 * (max_pt[0] - center[0])),
            ("Right Side", lambda p: p[:, 0] > center[0] + 0.25 * (max_pt[0] - center[0])),
            ("Center/Curved", lambda p: (
                (np.abs(p[:, 0] - center[0]) < 0.3 * (max_pt[0] - min_pt[0])) &
                (np.abs(p[:, 1] - center[1]) < 0.3 * (max_pt[1] - min_pt[1]))
            )),
        ]
        
        for name, mask_fn in region_defs:
            mask = mask_fn(points)
            if np.sum(mask) < 50:
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
                status="OK" if pass_rate >= 0.95 else "ATTENTION",
                deviation_direction="outward" if np.mean(region_devs) > 0 else "inward"
            ))
        
        return regions
    
    def run_analysis(
        self,
        part_id: str,
        part_name: str,
        material: str,
        tolerance: float,
        drawing_context: str = ""
    ) -> QCReport:
        """Run complete QC analysis"""
        from datetime import datetime
        
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
            
            # Regional analysis
            self._update_progress("analyze", 90, "Performing regional analysis...")
            report.regions = self.analyze_regions(tolerance)
            
            # AI Analysis
            self._update_progress("ai", 92, "Running AI interpretation...")
            self._run_ai_analysis(report, drawing_context)
            
            # Calculate quality score and verdict
            self._calculate_verdict(report)
            
            self._update_progress("complete", 100, "Analysis complete!")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            report.overall_result = QCResult.FAIL
            report.ai_summary = f"Analysis error: {str(e)}"
        
        return report
    
    def _run_ai_analysis(self, report: QCReport, drawing_context: str):
        """Generate AI interpretations for regions and root causes"""
        
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
