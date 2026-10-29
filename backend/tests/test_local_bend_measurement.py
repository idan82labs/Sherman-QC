import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import bend_detector as local_bend_detector


def _make_plane_points(center: np.ndarray, normal: np.ndarray, count: int, spread: float = 2.0) -> np.ndarray:
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    basis_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(basis_a, normal))) > 0.9:
        basis_a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_a = basis_a - normal * float(np.dot(basis_a, normal))
    basis_a = basis_a / (np.linalg.norm(basis_a) + 1e-12)
    basis_b = np.cross(normal, basis_a)
    pts = []
    steps = max(2, int(math.ceil(math.sqrt(count))))
    for i in range(steps):
        for j in range(steps):
            if len(pts) >= count:
                break
            u = (i - (steps - 1) / 2.0) * spread * 0.5
            v = (j - (steps - 1) / 2.0) * spread * 0.5
            pts.append(center + basis_a * u + basis_b * v)
    return np.asarray(pts, dtype=np.float64)


def test_probe_measurement_adapts_for_shallow_bends():
    cad_center = np.zeros(3, dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 25.0
    # Two surface normals separated by 25 degrees.
    surf1_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surf2_normal = np.array(
        [math.cos(math.radians(cad_angle)), math.sin(math.radians(cad_angle)), 0.0],
        dtype=np.float64,
    )

    tang1 = np.cross(cad_bend_direction, surf1_normal)
    tang1 /= np.linalg.norm(tang1)
    tang2 = np.cross(cad_bend_direction, surf2_normal)
    tang2 /= np.linalg.norm(tang2)
    if float(np.dot(tang1, tang2)) > 0.0:
        tang2 = -tang2

    probe1_center = cad_center + tang1 * 5.0
    probe2_center = cad_center + tang2 * 5.0
    # Only 8 points per side: this would fail under the previous fixed >=15 threshold.
    scan_points = np.vstack(
        [
            _make_plane_points(probe1_center, surf1_normal, 8),
            _make_plane_points(probe2_center, surf2_normal, 8),
        ]
    )

    result = local_bend_detector.measure_scan_bend_at_cad_location(
        scan_points=scan_points,
        cad_bend_center=cad_center,
        cad_surf1_normal=surf1_normal,
        cad_surf2_normal=surf2_normal,
        cad_bend_direction=cad_bend_direction,
        search_radius=30.0,
        probe_offset=15.0,
        probe_radius=10.0,
        cad_bend_angle=cad_angle,
    )

    assert result["success"] is True
    assert result["measurement_method"] == "probe_region"
    assert abs(result["measured_angle"] - cad_angle) < 2.0
    assert result["side1_points"] >= 6
    assert result["side2_points"] >= 6


def test_profile_section_measurement_handles_obtuse_bend_from_local_slice():
    cad_center = np.zeros(3, dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 160.0
    surface_normal_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surface_normal_b = np.array(
        [math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0],
        dtype=np.float64,
    )

    tangent_a = np.cross(cad_bend_direction, surface_normal_a)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(cad_bend_direction, surface_normal_b)
    tangent_b /= np.linalg.norm(tangent_b)
    if float(np.dot(tangent_a, tangent_b)) > 0.0:
        tangent_b = -tangent_b

    pts = []
    for z in np.linspace(-4.0, 4.0, 7):
        for s in np.linspace(3.0, 12.0, 7):
            pts.append(cad_center + tangent_a * s + cad_bend_direction * z)
            pts.append(cad_center + tangent_b * s + cad_bend_direction * z)
    scan_points = np.asarray(pts, dtype=np.float64)

    result = local_bend_detector.measure_scan_bend_via_profile_section(
        scan_points=scan_points,
        cad_bend_center=cad_center,
        cad_surf1_normal=surface_normal_a,
        cad_surf2_normal=surface_normal_b,
        cad_bend_direction=cad_bend_direction,
        cad_bend_angle=cad_angle,
        search_radius=20.0,
        section_half_width=5.0,
    )

    assert result["success"] is True
    assert result["measurement_method"] == "profile_section"
    assert abs(result["measured_angle"] - cad_angle) < 2.0
    assert result["side1_points"] >= 10
    assert result["side2_points"] >= 10


def test_parent_frame_profile_section_aggregates_multiple_valid_slices():
    line_start = np.array([0.0, 0.0, -12.0], dtype=np.float64)
    line_end = np.array([0.0, 0.0, 12.0], dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 160.0
    surface_normal_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surface_normal_b = np.array(
        [math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0],
        dtype=np.float64,
    )

    tangent_a = np.cross(cad_bend_direction, surface_normal_a)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(cad_bend_direction, surface_normal_b)
    tangent_b /= np.linalg.norm(tangent_b)
    if float(np.dot(tangent_a, tangent_b)) > 0.0:
        tangent_b = -tangent_b

    pts = []
    for z in np.linspace(-10.0, 10.0, 9):
        center = np.array([0.0, 0.0, z], dtype=np.float64)
        for s in np.linspace(3.0, 11.0, 8):
            pts.append(center + tangent_a * s)
            pts.append(center + tangent_b * s)
    scan_points = np.asarray(pts, dtype=np.float64)

    result = local_bend_detector.measure_scan_bend_via_parent_frame_profile_section(
        scan_points=scan_points,
        cad_bend_line_start=line_start,
        cad_bend_line_end=line_end,
        cad_surf1_normal=surface_normal_a,
        cad_surf2_normal=surface_normal_b,
        cad_bend_direction=cad_bend_direction,
        cad_bend_angle=cad_angle,
        search_radius=20.0,
        section_half_width=5.0,
        alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.91,
            "alignment_abstained": False,
        },
    )

    assert result["success"] is True
    assert result["measurement_method"] == "parent_frame_profile_section"
    assert result["slice_count"] >= 4
    assert result["planned_slice_count"] >= 5
    assert result["effective_slice_count"] >= 3.8
    assert result["angle_dispersion_deg"] <= 2.0
    assert abs(result["aggregate_angle_deg"] - cad_angle) < 2.0


def test_parent_frame_profile_section_uses_visible_span_instead_of_nominal_span():
    line_start = np.array([0.0, 0.0, -12.0], dtype=np.float64)
    line_end = np.array([0.0, 0.0, 12.0], dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 160.0
    surface_normal_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surface_normal_b = np.array(
        [math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0],
        dtype=np.float64,
    )

    tangent_a = np.cross(cad_bend_direction, surface_normal_a)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(cad_bend_direction, surface_normal_b)
    tangent_b /= np.linalg.norm(tangent_b)
    if float(np.dot(tangent_a, tangent_b)) > 0.0:
        tangent_b = -tangent_b

    pts = []
    for z in np.linspace(-5.0, 5.0, 9):
        center = np.array([0.0, 0.0, z], dtype=np.float64)
        for s in np.linspace(3.0, 11.0, 8):
            pts.append(center + tangent_a * s)
            pts.append(center + tangent_b * s)
    scan_points = np.asarray(pts, dtype=np.float64)

    result = local_bend_detector.measure_scan_bend_via_parent_frame_profile_section(
        scan_points=scan_points,
        cad_bend_line_start=line_start,
        cad_bend_line_end=line_end,
        cad_surf1_normal=surface_normal_a,
        cad_surf2_normal=surface_normal_b,
        cad_bend_direction=cad_bend_direction,
        cad_bend_angle=cad_angle,
        search_radius=20.0,
        section_half_width=5.0,
        alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.91,
            "alignment_abstained": False,
        },
    )

    assert result["success"] is True
    assert result["visible_span_ratio"] < 0.8
    assert result["supported_span_start_mm"] > 0.0
    assert result["supported_span_end_mm"] < 24.0


def test_profile_section_uses_line_arc_fallback_for_one_sided_slice():
    cad_center = np.zeros(3, dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 160.0
    surface_normal_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surface_normal_b = np.array(
        [math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0],
        dtype=np.float64,
    )

    tangent_a = np.cross(cad_bend_direction, surface_normal_a)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(cad_bend_direction, surface_normal_b)
    tangent_b /= np.linalg.norm(tangent_b)
    if float(np.dot(tangent_a, tangent_b)) > 0.0:
        tangent_b = -tangent_b

    pts = []
    for z in np.linspace(-4.0, 4.0, 7):
        for s in np.linspace(3.0, 12.0, 8):
            pts.append(cad_center + tangent_a * s + cad_bend_direction * z)
        for s in np.linspace(3.0, 4.2, 3):
            pts.append(cad_center + tangent_b * s + cad_bend_direction * z)
        for s in np.linspace(0.5, 2.0, 4):
            pts.append(cad_center + tangent_b * s + cad_bend_direction * z)
    scan_points = np.asarray(pts, dtype=np.float64)

    result = local_bend_detector.measure_scan_bend_via_profile_section(
        scan_points=scan_points,
        cad_bend_center=cad_center,
        cad_surf1_normal=surface_normal_a,
        cad_surf2_normal=surface_normal_b,
        cad_bend_direction=cad_bend_direction,
        cad_bend_angle=cad_angle,
        search_radius=20.0,
        section_half_width=5.0,
    )

    assert result["success"] is True


def test_surface_classification_recovers_local_faces_for_selected_baseline():
    cad_vertices = np.asarray(
        [
            [5.0, -6.0, -6.0],
            [5.0, 6.0, -6.0],
            [5.0, 6.0, 6.0],
            [5.0, -6.0, 6.0],
            [5.0, 0.0, 0.0],
            [-6.0, 5.0, -6.0],
            [6.0, 5.0, -6.0],
            [6.0, 5.0, 6.0],
            [-6.0, 5.0, 6.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    cad_triangles = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [5, 6, 9],
            [6, 7, 9],
            [7, 8, 9],
            [8, 5, 9],
        ],
        dtype=np.int32,
    )
    all_face_centers, all_face_normals = local_bend_detector._compute_triangle_centers_and_normals(
        cad_vertices,
        cad_triangles,
    )

    surf1_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surf2_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    recovered = local_bend_detector._recover_local_surface_face_centers(
        all_face_centers=all_face_centers,
        all_face_normals=all_face_normals,
        cad_bend_center=np.zeros(3, dtype=np.float64),
        cad_surface_normal=surf1_normal,
        search_radius=20.0,
        cad_bend_line_start=np.array([0.0, 0.0, -8.0], dtype=np.float64),
        cad_bend_line_end=np.array([0.0, 0.0, 8.0], dtype=np.float64),
    )

    assert recovered.shape[0] >= 4
    assert np.allclose(recovered[:, 0], 5.0)


def test_measure_all_scan_bends_disables_generic_station_retries_for_partials():
    original_profile = local_bend_detector.measure_scan_bend_via_profile_section
    original_probe = local_bend_detector.measure_scan_bend_at_cad_location
    calls = []

    def fake_profile(*args, **kwargs):
        calls.append(("profile", kwargs.get("allow_station_retries")))
        return {"success": False, "error": "no_profile"}

    def fake_probe(*args, **kwargs):
        calls.append(("probe", kwargs.get("allow_station_retries")))
        return {
            "success": True,
            "measured_angle": 90.0,
            "measurement_method": "probe_region",
            "measurement_confidence": "low",
            "side1_points": 6,
            "side2_points": 6,
        }

    local_bend_detector.measure_scan_bend_via_profile_section = fake_profile
    local_bend_detector.measure_scan_bend_at_cad_location = fake_probe
    try:
        scan_points = np.zeros((20, 3), dtype=np.float64)
        cad_bends = [
            {
                "bend_id": "CAD_B1",
                "bend_name": "CAD_B1",
                "bend_center": [0.0, 0.0, 0.0],
                "surface1_normal": [1.0, 0.0, 0.0],
                "surface2_normal": [0.0, 1.0, 0.0],
                "bend_direction": [0.0, 0.0, 1.0],
                "bend_angle": 90.0,
                "surface1_id": 1,
                "surface2_id": 2,
                "bend_line_start": [0.0, 0.0, -20.0],
                "bend_line_end": [0.0, 0.0, 20.0],
            }
        ]
        local_bend_detector.measure_all_scan_bends(
            scan_points=scan_points,
            cad_bends=cad_bends,
            scan_state="partial",
        )
        assert ("profile", False) in calls
        assert ("probe", False) in calls
    finally:
        local_bend_detector.measure_scan_bend_via_profile_section = original_profile
        local_bend_detector.measure_scan_bend_at_cad_location = original_probe


def test_profile_section_retries_off_center_station_along_bend_line():
    line_start = np.array([0.0, 0.0, -12.0], dtype=np.float64)
    line_end = np.array([0.0, 0.0, 12.0], dtype=np.float64)
    station_center = np.array([0.0, 0.0, 8.4], dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 160.0
    surface_normal_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surface_normal_b = np.array(
        [math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0],
        dtype=np.float64,
    )

    tangent_a = np.cross(cad_bend_direction, surface_normal_a)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(cad_bend_direction, surface_normal_b)
    tangent_b /= np.linalg.norm(tangent_b)
    if float(np.dot(tangent_a, tangent_b)) > 0.0:
        tangent_b = -tangent_b

    pts = []
    for z in np.linspace(-1.0, 1.0, 7):
        center = station_center + cad_bend_direction * z
        for s in np.linspace(3.0, 12.0, 7):
            pts.append(center + tangent_a * s)
            pts.append(center + tangent_b * s)
    scan_points = np.asarray(pts, dtype=np.float64)

    result = local_bend_detector.measure_scan_bend_via_profile_section(
        scan_points=scan_points,
        cad_bend_center=np.zeros(3, dtype=np.float64),
        cad_surf1_normal=surface_normal_a,
        cad_surf2_normal=surface_normal_b,
        cad_bend_direction=cad_bend_direction,
        cad_bend_angle=cad_angle,
        search_radius=20.0,
        section_half_width=3.0,
        cad_bend_line_start=line_start,
        cad_bend_line_end=line_end,
    )

    assert result["success"] is True
    assert abs(result["measured_angle"] - cad_angle) < 2.0
    assert "station_center" in result


def test_probe_measurement_retries_off_center_station_along_bend_line():
    line_start = np.array([0.0, 0.0, -12.0], dtype=np.float64)
    line_end = np.array([0.0, 0.0, 12.0], dtype=np.float64)
    station_center = np.array([0.0, 0.0, 8.0], dtype=np.float64)
    cad_bend_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cad_angle = 25.0
    surf1_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surf2_normal = np.array(
        [math.cos(math.radians(cad_angle)), math.sin(math.radians(cad_angle)), 0.0],
        dtype=np.float64,
    )

    tang1 = np.cross(cad_bend_direction, surf1_normal)
    tang1 /= np.linalg.norm(tang1)
    tang2 = np.cross(cad_bend_direction, surf2_normal)
    tang2 /= np.linalg.norm(tang2)
    if float(np.dot(tang1, tang2)) > 0.0:
        tang2 = -tang2

    probe1_center = station_center + tang1 * 5.0
    probe2_center = station_center + tang2 * 5.0
    scan_points = np.vstack(
        [
            _make_plane_points(probe1_center, surf1_normal, 8),
            _make_plane_points(probe2_center, surf2_normal, 8),
        ]
    )

    result = local_bend_detector.measure_scan_bend_at_cad_location(
        scan_points=scan_points,
        cad_bend_center=np.zeros(3, dtype=np.float64),
        cad_surf1_normal=surf1_normal,
        cad_surf2_normal=surf2_normal,
        cad_bend_direction=cad_bend_direction,
        search_radius=30.0,
        probe_offset=15.0,
        probe_radius=10.0,
        cad_bend_angle=cad_angle,
        cad_bend_line_start=line_start,
        cad_bend_line_end=line_end,
    )

    assert result["success"] is True
    assert abs(result["measured_angle"] - cad_angle) < 2.0
    assert "station_center" in result


def test_parent_frame_profile_section_requires_trusted_alignment():
    result = local_bend_detector.measure_scan_bend_via_parent_frame_profile_section(
        scan_points=np.zeros((10, 3), dtype=np.float64),
        cad_bend_line_start=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        cad_bend_line_end=np.array([0.0, 0.0, 10.0], dtype=np.float64),
        cad_surf1_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        cad_surf2_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        cad_bend_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        alignment_metadata={
            "alignment_source": "GLOBAL_ICP",
            "alignment_confidence": 0.99,
            "alignment_abstained": False,
        },
    )

    assert result["success"] is False
    assert result["measurement_method"] == "parent_frame_profile_section"
    assert "not trusted" in result["error"].lower()


def test_measure_all_scan_bends_skips_parent_frame_sections_for_rolled_features(monkeypatch):
    calls = {"count": 0}

    def _unexpected_parent_frame(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("parent-frame section should not run for rolled features")

    monkeypatch.setattr(
        local_bend_detector,
        "measure_scan_bend_via_parent_frame_profile_section",
        _unexpected_parent_frame,
    )

    results = local_bend_detector.measure_all_scan_bends(
        scan_points=np.zeros((40, 3), dtype=np.float64),
        cad_bends=[
            {
                "bend_id": "B1",
                "bend_name": "B1",
                "bend_center": [0.0, 0.0, 0.0],
                "surface1_normal": [1.0, 0.0, 0.0],
                "surface2_normal": [0.0, 1.0, 0.0],
                "bend_direction": [0.0, 0.0, 1.0],
                "bend_angle": 90.0,
                "surface1_id": 1,
                "surface2_id": 2,
                "bend_line_start": [0.0, 0.0, -5.0],
                "bend_line_end": [0.0, 0.0, 5.0],
                "feature_type": "ROLLED_PROCESS_FEATURE",
                "countable_in_regression": True,
            }
        ],
        search_radius=20.0,
        alignment_metadata={
            "alignment_source": "PLANE_DATUM_REFINEMENT",
            "alignment_confidence": 0.91,
            "alignment_abstained": False,
        },
        scan_state="partial",
    )

    assert calls["count"] == 0
    assert len(results) == 1
