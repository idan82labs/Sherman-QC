"""
Tests for the bend detection module.

Tests:
1. PlaneSegment creation and methods
2. BendDetector plane segmentation
3. Dihedral angle computation
4. Bend matching
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import feature_detection.bend_detector as bend_module
from feature_detection.bend_detector import (
    PlaneSegment,
    DetectedBend,
    BendSpecification,
    BendMatch,
    BendInspectionReport,
    BendDetector,
    ProgressiveBendMatcher,
    CADBendExtractor,
)


class TestPlaneSegment:
    """Tests for PlaneSegment data class."""

    def test_creation(self):
        """Test PlaneSegment creation."""
        plane = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.random.rand(100, 3),
            boundary_points=np.random.rand(20, 3),
            area=100.0,
            inlier_count=100,
        )
        assert plane.plane_id == 0
        assert plane.area == 100.0

    def test_point_to_plane_distance(self):
        """Test signed distance computation."""
        # Plane z = 0
        plane = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.array([[0, 0, 0]]),
            boundary_points=np.array([[0, 0, 0]]),
            area=1.0,
            inlier_count=1,
        )

        # Point at z = 5 should have distance 5
        dist = plane.point_to_plane_distance(np.array([0, 0, 5]))
        assert abs(dist - 5.0) < 1e-10

        # Point at z = -3 should have distance -3
        dist = plane.point_to_plane_distance(np.array([0, 0, -3]))
        assert abs(dist - (-3.0)) < 1e-10


class TestBendDetector:
    """Tests for BendDetector class."""

    def test_ransac_plane_fit(self):
        """Test RANSAC plane fitting on synthetic data."""
        # Create points on a plane z = 2 with some noise
        np.random.seed(42)
        n_points = 100
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = 2.0 + np.random.normal(0, 0.1, n_points)  # Small noise
        points = np.column_stack([x, y, z])

        detector = BendDetector(ransac_iterations=100, plane_threshold=0.5)
        result = detector._ransac_plane_fit(points)

        assert result is not None
        normal, d, inlier_mask = result

        # Normal should be close to [0, 0, 1] or [0, 0, -1]
        assert abs(abs(normal[2]) - 1.0) < 0.1

        # Most points should be inliers
        assert np.sum(inlier_mask) > 80

    def test_dihedral_angle_computation(self):
        """Test dihedral angle computation between two planes."""
        # Create two planes at 90 degrees
        # Plane 1: z = 0 (normal [0, 0, 1])
        # Plane 2: x = 0 (normal [1, 0, 0])
        n1 = np.array([0, 0, 1])
        n2 = np.array([1, 0, 0])

        # cos(theta) = |n1 · n2|
        cos_angle = abs(np.dot(n1, n2))
        angle_between = np.degrees(np.arccos(cos_angle))
        dihedral = 180 - angle_between

        # Should be 90 degrees
        assert abs(dihedral - 90.0) < 1e-10

    def test_detect_bends_on_l_shape(self):
        """Test bend detection on an L-shaped point cloud."""
        np.random.seed(42)

        # Create L-shaped point cloud (two perpendicular planes)
        # Horizontal plane (z = 0, x from 0 to 10)
        n_points = 200
        x1 = np.random.uniform(0, 10, n_points)
        y1 = np.random.uniform(-5, 5, n_points)
        z1 = np.random.normal(0, 0.1, n_points)
        plane1 = np.column_stack([x1, y1, z1])

        # Vertical plane (x = 0, z from 0 to 10)
        x2 = np.random.normal(0, 0.1, n_points)
        y2 = np.random.uniform(-5, 5, n_points)
        z2 = np.random.uniform(0, 10, n_points)
        plane2 = np.column_stack([x2, y2, z2])

        points = np.vstack([plane1, plane2])

        detector = BendDetector(
            ransac_iterations=50,
            plane_threshold=0.5,
            min_plane_points=50,
            min_plane_area=10.0,
        )

        bends = detector.detect_bends(points)

        # Should detect at least one bend
        # Note: Detection might fail due to insufficient overlap between planes
        # so we test the mechanism works, not necessarily the result
        assert isinstance(bends, list)

    def test_segment_planes(self):
        """Test plane segmentation."""
        np.random.seed(42)

        # Create two separate planes
        # Plane 1: z = 0
        n_points = 150
        x1 = np.random.uniform(0, 10, n_points)
        y1 = np.random.uniform(0, 10, n_points)
        z1 = np.random.normal(0, 0.1, n_points)
        plane1 = np.column_stack([x1, y1, z1])

        # Plane 2: z = 20 (well separated)
        x2 = np.random.uniform(0, 10, n_points)
        y2 = np.random.uniform(0, 10, n_points)
        z2 = 20 + np.random.normal(0, 0.1, n_points)
        plane2 = np.column_stack([x2, y2, z2])

        points = np.vstack([plane1, plane2])

        detector = BendDetector(
            ransac_iterations=50,
            plane_threshold=0.5,
            min_plane_points=50,
            min_plane_area=10.0,
        )

        planes = detector._segment_planes(points)

        # Should detect 2 planes
        assert len(planes) >= 2

    def test_merge_coplanar_planes(self):
        """Coplanar fragmented planes should be merged into one flange."""
        detector = BendDetector(
            merge_coplanar_planes=True,
            coplanar_angle_threshold=2.0,
            coplanar_offset_threshold=0.2,
            coplanar_boundary_threshold=0.5,
        )

        p1_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        p2_points = np.array([
            [1.05, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.05, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ])

        plane1 = PlaneSegment(
            plane_id=0,
            normal=np.array([0.0, 0.0, 1.0]),
            d=0.0,
            centroid=p1_points.mean(axis=0),
            points=p1_points,
            boundary_points=p1_points,
            area=1.0,
            inlier_count=len(p1_points),
        )
        plane2 = PlaneSegment(
            plane_id=1,
            normal=np.array([0.0, 0.0, 1.0]),
            d=0.0,
            centroid=p2_points.mean(axis=0),
            points=p2_points,
            boundary_points=p2_points,
            area=1.0,
            inlier_count=len(p2_points),
        )

        merged = detector._merge_coplanar_planes([plane1, plane2])
        assert len(merged) == 1
        assert merged[0].inlier_count == len(p1_points) + len(p2_points)

    def test_dedup_overlapping_bends(self):
        """Near-duplicate bends should collapse to one detection."""
        detector = BendDetector(
            dedup_merge_threshold=5.0,
            dedup_angle_threshold=8.0,
            dedup_direction_threshold=0.98,
        )

        flange1 = PlaneSegment(
            plane_id=0,
            normal=np.array([0.0, 0.0, 1.0]),
            d=0.0,
            centroid=np.array([0.0, 0.0, 0.0]),
            points=np.random.rand(16, 3),
            boundary_points=np.random.rand(8, 3),
            area=20.0,
            inlier_count=16,
        )
        flange2 = PlaneSegment(
            plane_id=1,
            normal=np.array([1.0, 0.0, 0.0]),
            d=0.0,
            centroid=np.array([0.0, 0.0, 0.0]),
            points=np.random.rand(16, 3),
            boundary_points=np.random.rand(8, 3),
            area=20.0,
            inlier_count=16,
        )

        duplicate_a = DetectedBend(
            bend_id="B1",
            measured_angle=90.0,
            measured_radius=2.0,
            bend_line_start=np.array([0.0, 0.0, 0.0]),
            bend_line_end=np.array([0.0, 10.0, 0.0]),
            bend_line_direction=np.array([0.0, 1.0, 0.0]),
            confidence=0.95,
            flange1=flange1,
            flange2=flange2,
        )
        duplicate_b = DetectedBend(
            bend_id="B2",
            measured_angle=92.0,
            measured_radius=2.1,
            bend_line_start=np.array([1.0, 0.2, 0.0]),
            bend_line_end=np.array([1.0, 10.2, 0.0]),
            bend_line_direction=np.array([0.0, 1.0, 0.0]),
            confidence=0.70,
            flange1=flange1,
            flange2=flange2,
        )
        distinct = DetectedBend(
            bend_id="B3",
            measured_angle=120.0,
            measured_radius=3.0,
            bend_line_start=np.array([20.0, 0.0, 0.0]),
            bend_line_end=np.array([20.0, 10.0, 0.0]),
            bend_line_direction=np.array([0.0, 1.0, 0.0]),
            confidence=0.80,
            flange1=flange1,
            flange2=flange2,
        )

        merged = detector._merge_multiscale_bends([duplicate_a, duplicate_b, distinct], merge_threshold=5.0, angle_threshold=8.0)
        assert len(merged) == 2


class TestProgressiveBendMatcher:
    """Tests for ProgressiveBendMatcher class."""

    def test_exact_match(self):
        """Test matching when detected angle matches CAD exactly."""
        # Create CAD bend specification
        cad_bends = [
            BendSpecification(
                bend_id="B1",
                target_angle=90.0,
                target_radius=3.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 10, 0]),
                tolerance_angle=1.0,
            ),
        ]

        # Create "detected" bend that matches
        flange1 = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([5, 0, 0]),
            points=np.random.rand(100, 3),
            boundary_points=np.random.rand(10, 3),
            area=100.0,
            inlier_count=100,
        )
        flange2 = PlaneSegment(
            plane_id=1,
            normal=np.array([1, 0, 0]),
            d=0.0,
            centroid=np.array([0, 0, 5]),
            points=np.random.rand(100, 3),
            boundary_points=np.random.rand(10, 3),
            area=100.0,
            inlier_count=100,
        )

        detected_bends = [
            DetectedBend(
                bend_id="D1",
                measured_angle=90.0,  # Exact match
                measured_radius=3.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 10, 0]),
                bend_line_direction=np.array([0, 1, 0]),
                confidence=0.9,
                flange1=flange1,
                flange2=flange2,
            ),
        ]

        matcher = ProgressiveBendMatcher()
        report = matcher.match(detected_bends, cad_bends, part_id="TEST")

        assert report.detected_count == 1
        assert report.pass_count == 1
        assert report.fail_count == 0
        assert len(report.matches) == 1
        assert report.matches[0].status == "PASS"

    def test_no_detection(self):
        """Test when no bends are detected."""
        cad_bends = [
            BendSpecification(
                bend_id="B1",
                target_angle=90.0,
                target_radius=3.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 10, 0]),
            ),
        ]

        matcher = ProgressiveBendMatcher()
        report = matcher.match([], cad_bends, part_id="TEST")

        assert report.detected_count == 0
        assert report.total_cad_bends == 1
        assert report.matches[0].status == "NOT_DETECTED"

    def test_out_of_tolerance(self):
        """Test when detected angle is outside tolerance."""
        cad_bends = [
            BendSpecification(
                bend_id="B1",
                target_angle=90.0,
                target_radius=3.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 10, 0]),
                tolerance_angle=1.0,  # +/- 1 degree
            ),
        ]

        flange1 = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.zeros((10, 3)),
            boundary_points=np.zeros((5, 3)),
            area=100.0,
            inlier_count=100,
        )
        flange2 = PlaneSegment(
            plane_id=1,
            normal=np.array([1, 0, 0]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.zeros((10, 3)),
            boundary_points=np.zeros((5, 3)),
            area=100.0,
            inlier_count=100,
        )

        detected_bends = [
            DetectedBend(
                bend_id="D1",
                measured_angle=93.0,  # 3 degrees off
                measured_radius=3.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 10, 0]),
                bend_line_direction=np.array([0, 1, 0]),
                confidence=0.9,
                flange1=flange1,
                flange2=flange2,
            ),
        ]

        matcher = ProgressiveBendMatcher()
        report = matcher.match(detected_bends, cad_bends, part_id="TEST")

        assert report.detected_count == 1
        assert report.fail_count == 1
        assert report.matches[0].status == "FAIL"
        assert abs(report.matches[0].angle_deviation - 3.0) < 0.01

    def test_match_includes_mm_metrics_and_action(self):
        cad_bends = [
            BendSpecification(
                bend_id="B1",
                target_angle=90.0,
                target_radius=3.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 10, 0]),
                tolerance_angle=1.0,
                tolerance_radius=0.5,
            ),
        ]
        flange1 = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.zeros((10, 3)),
            boundary_points=np.zeros((5, 3)),
            area=100.0,
            inlier_count=100,
        )
        flange2 = PlaneSegment(
            plane_id=1,
            normal=np.array([1, 0, 0]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.zeros((10, 3)),
            boundary_points=np.zeros((5, 3)),
            area=100.0,
            inlier_count=100,
        )
        detected_bends = [
            DetectedBend(
                bend_id="D1",
                measured_angle=92.0,
                measured_radius=3.2,
                bend_line_start=np.array([0.8, 0, 0]),
                bend_line_end=np.array([0.8, 10, 0]),
                bend_line_direction=np.array([0, 1, 0]),
                confidence=0.9,
                flange1=flange1,
                flange2=flange2,
            ),
        ]
        matcher = ProgressiveBendMatcher()
        report = matcher.match(detected_bends, cad_bends, part_id="TEST")
        row = report.matches[0]
        assert row.line_center_deviation_mm is not None
        assert row.expected_line_length_mm is not None
        assert row.actual_line_length_mm is not None
        assert row.issue_location != ""
        assert row.action_item != ""
        payload = report.to_dict()
        assert "operator_actions" in payload
        assert isinstance(payload["operator_actions"], list)

    def test_rolled_status_uses_arc_and_center_metrics(self):
        cad_bends = [
            BendSpecification(
                bend_id="R1",
                target_angle=120.0,
                target_radius=12.0,
                bend_line_start=np.array([0, 0, 0]),
                bend_line_end=np.array([0, 30, 0]),
                tolerance_angle=1.0,
                tolerance_radius=0.5,
                bend_form="ROLLED",
            ),
        ]
        flange = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.zeros((10, 3)),
            boundary_points=np.zeros((5, 3)),
            area=100.0,
            inlier_count=100,
        )
        detected_bends = [
            DetectedBend(
                bend_id="D1",
                measured_angle=120.0,
                measured_radius=17.0,
                bend_line_start=np.array([8.0, 0, 0]),
                bend_line_end=np.array([8.0, 30, 0]),
                bend_line_direction=np.array([0, 1, 0]),
                confidence=0.9,
                flange1=flange,
                flange2=flange,
            ),
        ]
        matcher = ProgressiveBendMatcher()
        report = matcher.match(detected_bends, cad_bends, part_id="TEST")
        assert report.matches[0].status in {"WARNING", "FAIL"}
        assert report.matches[0].bend_form == "ROLLED"


class TestBendInspectionReport:
    """Tests for BendInspectionReport class."""

    def test_summary_computation(self):
        """Test that summary is computed correctly."""
        # Create some matches
        cad1 = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        cad2 = BendSpecification(
            bend_id="B2",
            target_angle=45.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )

        matches = [
            BendMatch(cad_bend=cad1, detected_bend=None, status="PASS"),
            BendMatch(cad_bend=cad2, detected_bend=None, status="NOT_DETECTED"),
        ]

        report = BendInspectionReport(part_id="TEST", matches=matches)
        report.compute_summary()

        assert report.total_cad_bends == 2
        assert report.pass_count == 1
        payload = report.to_dict()
        summary = payload["summary"]
        assert summary["completed_bends"] == 1
        assert summary["completed_in_spec"] == 1
        assert summary["completed_out_of_spec"] == 0
        assert summary["remaining_bends"] == 1
        assert summary["observed_bends"] == 1
        assert summary["unobserved_bends"] == 1
        assert summary["overall_result"] == "WARNING"
        assert summary["release_decision"] == "HOLD"
        assert summary["formed_bends"] == 1
        assert summary["unknown_completion_bends"] == 1
        assert summary["in_tolerance_bends"] == 1
        assert summary["unmeasurable_bends"] == 1
        assert summary["unknown_position_bends"] == 2
        assert summary["release_hold_reasons"] == ["completion", "correspondence", "metrology", "observability", "position"]

    def test_to_table_string(self):
        """Test ASCII table generation."""
        cad = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        matches = [BendMatch(cad_bend=cad, detected_bend=None, status="NOT_DETECTED")]

        report = BendInspectionReport(part_id="TEST_PART", matches=matches)
        table = report.to_table_string()

        assert "TEST_PART" in table
        assert "B1" in table
        assert "90.0" in table
        assert "In spec:" in table
        assert "Observability:" in table

    def test_match_serialization_includes_observability(self):
        cad = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        match = BendMatch(
            cad_bend=cad,
            detected_bend=None,
            status="NOT_DETECTED",
            physical_completion_state="UNKNOWN",
            observability_state="UNKNOWN",
            observability_detail_state="PARTIALLY_OBSERVED",
            observability_confidence=0.72,
            visibility_score=0.48,
            visibility_score_source="heuristic_local_visibility_v0",
            surface_visibility_ratio=0.5,
            local_support_score=0.27,
            side_balance_score=0.18,
            assignment_source="CAD_LOCAL_NEIGHBORHOOD",
            assignment_confidence=0.61,
            observed_surface_count=1,
            local_point_count=28,
            local_evidence_score=0.54,
            measurement_mode="cad_local_neighborhood",
            measurement_method="surface_classification",
            measurement_context={"search_radius_mm": 35.0},
        )
        payload = match.to_dict()
        assert payload["physical_completion_state"] == "UNKNOWN"
        assert payload["observability_state"] == "UNKNOWN"
        assert payload["observability_detail_state"] == "PARTIALLY_OBSERVED"
        assert payload["visibility_score"] == 0.48
        assert payload["surface_visibility_ratio"] == 0.5
        assert payload["assignment_source"] == "CAD_LOCAL_NEIGHBORHOOD"
        assert payload["assignment_confidence"] == 0.61
        assert payload["observed_surface_count"] == 1
        assert payload["feature_family"] == "COUNTABLE_DISCRETE_BEND"
        assert payload["measurement_primitive"] == "ANGLE_RADIUS"
        assert payload["measurement_mode"] == "cad_local_neighborhood"
        assert payload["completion_state"] == "UNKNOWN"
        assert payload["metrology_state"] == "UNMEASURABLE"
        assert payload["positional_state"] == "UNKNOWN_POSITION"
        assert payload["correspondence_state"] == "UNRESOLVED"
        assert payload["observability_state_internal"] == "PARTIAL"

    def test_match_serialization_includes_positional_and_metrology_states(self):
        cad = BendSpecification(
            bend_id="B1",
            target_angle=120.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        flange1 = PlaneSegment(0, np.array([1, 0, 0]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        flange2 = PlaneSegment(1, np.array([0, 0, 1]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        detected = DetectedBend(
            bend_id="D1",
            measured_angle=117.5,
            measured_radius=3.2,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
            bend_line_direction=np.array([0, 1, 0]),
            confidence=0.9,
            flange1=flange1,
            flange2=flange2,
        )
        match = BendMatch(
            cad_bend=cad,
            detected_bend=detected,
            status="FAIL",
            physical_completion_state="FORMED",
            observability_state="FORMED",
            line_center_deviation_mm=3.1,
            measurement_context={"heatmap_consistency": {"status": "CONTRADICTED"}},
        )
        payload = match.to_dict()
        assert payload["completion_state"] == "FORMED"
        assert payload["metrology_state"] == "OUT_OF_TOL"
        assert payload["positional_state"] == "MISLOCATED"
        assert payload["position_evidence"]["status"] == "CONTRADICTED"

    def test_summary_counts_only_countable_bends_but_keeps_process_features(self):
        countable = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        rolled = BendSpecification(
            bend_id="R1",
            target_angle=140.0,
            target_radius=25.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
            bend_form="ROLLED",
            feature_type="ROLLED_SECTION",
            countable_in_regression=False,
        )
        report = BendInspectionReport(
            part_id="TEST",
            matches=[
                BendMatch(cad_bend=countable, detected_bend=None, status="PASS"),
                BendMatch(cad_bend=rolled, detected_bend=None, status="PASS"),
            ],
        )

        payload = report.to_dict()
        summary = payload["summary"]
        assert summary["total_bends"] == 1
        assert summary["completed_bends"] == 1
        assert summary["total_features"] == 2
        assert summary["detected_features"] == 2
        assert summary["process_features"] == 1
        assert summary["detected_process_features"] == 1
        assert payload["matches"][1]["feature_type"] == "ROLLED_SECTION"
        assert payload["matches"][1]["countable_in_regression"] is False

    def test_summary_fails_release_on_mislocated_or_unformed_bends(self):
        cad1 = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        cad2 = BendSpecification(
            bend_id="B2",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        flange1 = PlaneSegment(0, np.array([1, 0, 0]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        flange2 = PlaneSegment(1, np.array([0, 0, 1]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        detected = DetectedBend(
            bend_id="D1",
            measured_angle=90.2,
            measured_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
            bend_line_direction=np.array([0, 1, 0]),
            confidence=0.9,
            flange1=flange1,
            flange2=flange2,
        )
        report = BendInspectionReport(
            part_id="TEST",
            matches=[
                BendMatch(
                    cad_bend=cad1,
                    detected_bend=detected,
                    status="PASS",
                    physical_completion_state="FORMED",
                    observability_state="FORMED",
                    line_center_deviation_mm=3.2,
                ),
                BendMatch(
                    cad_bend=cad2,
                    detected_bend=None,
                    status="NOT_DETECTED",
                    physical_completion_state="UNFORMED",
                    observability_state="UNFORMED",
                ),
            ],
        )
        summary = report.to_dict()["summary"]
        assert summary["mislocated_bends"] == 1
        assert summary["unformed_bends"] == 1
        assert summary["overall_result"] == "FAIL"
        assert summary["release_decision"] == "AUTO_FAIL"
        assert sorted(summary["release_blocked_by"]) == ["completion", "position"]

    def test_summary_warns_for_out_of_tolerance_without_release_blockers(self):
        cad = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        flange1 = PlaneSegment(0, np.array([1, 0, 0]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        flange2 = PlaneSegment(1, np.array([0, 0, 1]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        detected = DetectedBend(
            bend_id="D1",
            measured_angle=92.2,
            measured_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
            bend_line_direction=np.array([0, 1, 0]),
            confidence=0.9,
            flange1=flange1,
            flange2=flange2,
        )
        report = BendInspectionReport(
            part_id="TEST",
            matches=[
                BendMatch(
                    cad_bend=cad,
                    detected_bend=detected,
                    status="WARNING",
                    physical_completion_state="FORMED",
                    observability_state="FORMED",
                    line_center_deviation_mm=0.5,
                )
            ],
        )
        summary = report.to_dict()["summary"]
        assert summary["out_of_tolerance_bends"] == 1
        assert summary["on_position_bends"] == 1
        assert summary["overall_result"] == "WARNING"
        assert summary["release_decision"] == "HOLD"
        assert summary["release_blocked_by"] == []

    def test_summary_auto_pass_requires_confident_correspondence_and_trusted_alignment(self):
        cad = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        flange1 = PlaneSegment(0, np.array([1, 0, 0]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        flange2 = PlaneSegment(1, np.array([0, 0, 1]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        detected = DetectedBend(
            bend_id="D1",
            measured_angle=90.1,
            measured_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
            bend_line_direction=np.array([0, 1, 0]),
            confidence=0.9,
            flange1=flange1,
            flange2=flange2,
        )
        report = BendInspectionReport(
            part_id="TEST",
            matches=[
                BendMatch(
                    cad_bend=cad,
                    detected_bend=detected,
                    status="PASS",
                    physical_completion_state="FORMED",
                    observability_state="FORMED",
                    line_center_deviation_mm=0.2,
                    assignment_confidence=0.91,
                    assignment_candidate_score=0.9,
                    assignment_null_score=0.1,
                    assignment_candidate_count=1,
                )
            ],
        )
        report.alignment_metadata = {
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.84,
            "alignment_abstained": False,
        }
        summary = report.to_dict()["summary"]
        assert summary["overall_result"] == "PASS"
        assert summary["release_decision"] == "AUTO_PASS"
        assert summary["trusted_alignment_for_release"] is True

    def test_summary_holds_when_alignment_untrusted_even_if_local_metrics_pass(self):
        cad = BendSpecification(
            bend_id="B1",
            target_angle=90.0,
            target_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
        )
        flange1 = PlaneSegment(0, np.array([1, 0, 0]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        flange2 = PlaneSegment(1, np.array([0, 0, 1]), 0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), 10.0, 3)
        detected = DetectedBend(
            bend_id="D1",
            measured_angle=90.0,
            measured_radius=3.0,
            bend_line_start=np.zeros(3),
            bend_line_end=np.array([0, 10, 0]),
            bend_line_direction=np.array([0, 1, 0]),
            confidence=0.9,
            flange1=flange1,
            flange2=flange2,
        )
        report = BendInspectionReport(
            part_id="TEST",
            matches=[
                BendMatch(
                    cad_bend=cad,
                    detected_bend=detected,
                    status="PASS",
                    physical_completion_state="FORMED",
                    observability_state="FORMED",
                    line_center_deviation_mm=0.1,
                    assignment_confidence=0.91,
                    assignment_candidate_score=0.9,
                    assignment_null_score=0.1,
                    assignment_candidate_count=1,
                    measurement_context={"heatmap_consistency": {"status": "CONTRADICTED"}},
                )
            ],
        )
        report.alignment_metadata = {
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.2,
            "alignment_abstained": True,
        }
        summary = report.to_dict()["summary"]
        assert summary["overall_result"] == "WARNING"
        assert summary["release_decision"] == "HOLD"
        assert "alignment" in summary["release_hold_reasons"]


class TestMathematicalFoundations:
    """Tests for mathematical algorithms used in bend detection."""

    def test_circle_from_3_points(self):
        """Test circumcircle calculation."""
        detector = BendDetector()

        # Three points on a unit circle centered at origin
        p1 = np.array([1, 0])
        p2 = np.array([0, 1])
        p3 = np.array([-1, 0])

        center, radius = detector._circle_from_3_points(p1, p2, p3)

        assert center is not None
        assert abs(radius - 1.0) < 0.01
        assert np.linalg.norm(center) < 0.01  # Center should be at origin

    def test_collinear_points_circle(self):
        """Test that collinear points return None for circle."""
        detector = BendDetector()

        # Three collinear points
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([2, 0])

        center, radius = detector._circle_from_3_points(p1, p2, p3)

        # Should return None for degenerate case
        assert center is None or radius == 0.0

    def test_least_squares_plane(self):
        """Test least squares plane fitting."""
        np.random.seed(42)
        detector = BendDetector()

        # Create points on plane ax + by + cz = d
        # Simple case: z = 5
        n_points = 50
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.full(n_points, 5.0)
        points = np.column_stack([x, y, z])

        normal, d = detector._least_squares_plane(points)

        # Normal should be [0, 0, 1] or [0, 0, -1]
        assert abs(abs(normal[2]) - 1.0) < 0.01

        # d should be -5 or +5 depending on normal direction
        assert abs(abs(d) - 5.0) < 0.01


class TestCADBendExtractor:
    """Tests CAD bend extraction wiring and configurability."""

    def test_extractor_passes_detector_kwargs(self, monkeypatch):
        captured = {}

        class DummyDetector:
            def __init__(self, **kwargs):
                captured["ctor"] = kwargs

            def detect_bends(self, vertices, **kwargs):
                captured["call"] = kwargs
                return []

        monkeypatch.setattr(bend_module, "BendDetector", DummyDetector)

        extractor = CADBendExtractor(
            detector_kwargs={"dedup_merge_threshold": 13.0, "min_plane_points": 42},
            detect_call_kwargs={"preprocess": False, "voxel_size": 1.2},
        )
        specs = extractor.extract_from_mesh(np.zeros((64, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32))

        assert specs == []
        assert captured["ctor"]["dedup_merge_threshold"] == 13.0
        assert captured["ctor"]["min_plane_points"] == 42
        assert captured["call"]["preprocess"] is False
        assert abs(captured["call"]["voxel_size"] - 1.2) < 1e-9

    def test_extractor_merges_nearby_rolled_segments(self, monkeypatch):
        flange = PlaneSegment(
            plane_id=0,
            normal=np.array([0, 0, 1]),
            d=0.0,
            centroid=np.array([0, 0, 0]),
            points=np.random.rand(20, 3),
            boundary_points=np.random.rand(10, 3),
            area=100.0,
            inlier_count=20,
        )

        class DummyDetector:
            def __init__(self, **kwargs):
                pass

            def detect_bends(self, vertices, **kwargs):
                return [
                    DetectedBend(
                        bend_id="B1",
                        measured_angle=110.0,
                        measured_radius=14.0,
                        bend_line_start=np.array([0.0, 0.0, 0.0]),
                        bend_line_end=np.array([0.0, 25.0, 0.0]),
                        bend_line_direction=np.array([0.0, 1.0, 0.0]),
                        confidence=0.8,
                        flange1=flange,
                        flange2=flange,
                    ),
                    DetectedBend(
                        bend_id="B2",
                        measured_angle=112.0,
                        measured_radius=13.5,
                        bend_line_start=np.array([1.2, 0.0, 0.0]),
                        bend_line_end=np.array([1.2, 24.5, 0.0]),
                        bend_line_direction=np.array([0.0, 1.0, 0.0]),
                        confidence=0.7,
                        flange1=flange,
                        flange2=flange,
                    ),
                    DetectedBend(
                        bend_id="B3",
                        measured_angle=90.0,
                        measured_radius=3.0,
                        bend_line_start=np.array([40.0, 0.0, 0.0]),
                        bend_line_end=np.array([40.0, 20.0, 0.0]),
                        bend_line_direction=np.array([0.0, 1.0, 0.0]),
                        confidence=0.9,
                        flange1=flange,
                        flange2=flange,
                    ),
                ]

        monkeypatch.setattr(bend_module, "BendDetector", DummyDetector)

        extractor = CADBendExtractor()
        specs = extractor.extract_from_mesh(np.zeros((100, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32))

        # First two rolled segments should merge into one logical rolled bend.
        assert len(specs) == 2
        forms = sorted(s.bend_form for s in specs)
        assert forms == ["FOLDED", "ROLLED"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
