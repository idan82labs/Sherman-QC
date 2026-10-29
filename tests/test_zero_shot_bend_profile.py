import hashlib

import numpy as np
import zero_shot_bend_profile as profile_module

from zero_shot_bend_profile import (
    DETECTOR_POINT_BUDGET,
    BendSupportStrip,
    _bounded_detector_input,
    _build_candidate_bundles,
    _build_prismatic_plane_lattice_strips,
    _detect_interior_formed_transition_strips,
    _prune_low_support_parallel_long_candidates,
    _resolve_scan_state_hint,
    run_zero_shot_bend_profile,
    run_zero_shot_bend_profile_from_geometry,
)


def test_scan_state_hint_never_uses_part_identity_or_filename():
    assert _resolve_scan_state_hint("KNOWN_PART", "/tmp/full_final_complete.ply") is None


def _grid_points(nx: int, ny: int, nz: int) -> np.ndarray:
    return np.stack(
        np.meshgrid(
            np.arange(nx, dtype=np.float64),
            np.arange(ny, dtype=np.float64),
            np.arange(nz, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)


def test_detector_input_enforces_default_budget_deterministically_without_source_mutation():
    points = _grid_points(60, 60, 60)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (len(points), 1))
    original_points = points.copy()
    original_normals = normals.copy()

    first_points, first_normals, first_voxel = _bounded_detector_input(points, normals)
    second_points, second_normals, second_voxel = _bounded_detector_input(points, normals)

    assert len(points) > DETECTOR_POINT_BUDGET
    assert 0 < len(first_points) <= DETECTOR_POINT_BUDGET
    assert first_voxel == second_voxel
    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_normals, second_normals)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(normals, original_normals)


def test_candidate_proposals_are_stable_and_preserve_original_call_contract():
    points = _grid_points(25, 20, 20)
    normals = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (len(points), 1))
    detect_kwargs = {"preprocess": True, "voxel_size": 1.6}

    class RecordingDetector:
        def detect_bends(self, detector_points, *, normals, **kwargs):
            digest = hashlib.sha256(np.ascontiguousarray(detector_points).tobytes()).hexdigest()
            return [(len(detector_points), digest, dict(kwargs))]

    first = _build_candidate_bundles(RecordingDetector(), points, normals, detect_kwargs)
    second = _build_candidate_bundles(RecordingDetector(), points, normals, detect_kwargs)

    assert first == second
    assert detect_kwargs == {"preprocess": True, "voxel_size": 1.6}
    assert first["original"]["single_scale"][0][2] == {
        "preprocess": True,
        "voxel_size": 1.6,
        "multiscale": False,
    }
    assert first["original"]["multiscale"][0][2]["multiscale"] is True
    for mode in ("voxel_0.8", "voxel_1.2"):
        assert first[mode]["single_scale"][0][2] == {"preprocess": False, "multiscale": False}


def test_loaded_profile_reuses_prebuilt_candidate_bundles_without_semantic_change(monkeypatch):
    points = _grid_points(4, 3, 2)
    bundles = {
        "original": {"single_scale": [], "multiscale": []},
        "voxel_0.8": {"single_scale": [], "multiscale": []},
        "voxel_1.2": {"single_scale": [], "multiscale": []},
    }
    direct = run_zero_shot_bend_profile_from_geometry(
        scan_path="scan.ply",
        points=points,
        candidate_bundles=bundles,
        part_id="P1",
    )

    monkeypatch.setattr(profile_module, "_load_scan_geometry", lambda _path: (points, None))
    monkeypatch.setattr(profile_module, "_build_candidate_bundles", lambda **_kwargs: bundles)

    loaded = run_zero_shot_bend_profile(
        scan_path="scan.ply",
        runtime_config=type("Runtime", (), {"scan_detector_kwargs": {}, "scan_detect_call_kwargs": {}})(),
        part_id="P1",
    )

    assert loaded.to_dict() == direct.to_dict()


def test_interior_formed_transition_completion_finds_two_internal_tabs_deterministically():
    axis = np.arange(-80.0, 81.0, 1.0)
    plate_x, plate_y = np.meshgrid(axis, axis, indexing="ij")
    plate = np.column_stack((plate_x.ravel(), plate_y.ravel(), np.zeros(plate_x.size)))
    slot_mask = ~(
        ((plate[:, 0] >= -32.0) & (plate[:, 0] <= 32.0) & (plate[:, 1] >= 22.0) & (plate[:, 1] < 30.0))
        | ((plate[:, 0] >= -32.0) & (plate[:, 0] <= 32.0) & (plate[:, 1] > -30.0) & (plate[:, 1] <= -22.0))
    )
    plate = plate[slot_mask]
    wall_x, wall_z = np.meshgrid(
        np.arange(-32.0, 33.0, 1.0),
        np.arange(-12.0, 0.1, 1.0),
        indexing="ij",
    )
    upper_wall = np.column_stack(
        (wall_x.ravel(), np.full(wall_x.size, 30.0), wall_z.ravel())
    )
    lower_wall = np.column_stack(
        (wall_x.ravel(), np.full(wall_x.size, -30.0), wall_z.ravel())
    )
    points = np.vstack((plate, upper_wall, lower_wall))

    first = _detect_interior_formed_transition_strips(points, spacing_mm=1.0, start_index=9)
    second = _detect_interior_formed_transition_strips(points, spacing_mm=1.0, start_index=9)

    assert [strip.to_dict() for strip in first] == [strip.to_dict() for strip in second]
    assert len(first) == 2
    assert [strip.strip_id for strip in first] == ["S9", "S10"]
    assert all(strip.strip_family == "interior_formed_transition_strip" for strip in first)
    assert all(50.0 <= strip.along_axis_span_mm <= 65.0 for strip in first)


def test_interior_formed_transition_completion_abstains_without_dominant_plane(monkeypatch):
    points = np.zeros((600, 3), dtype=np.float64)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (len(points), 1))

    monkeypatch.setattr(
        profile_module,
        "_voxel_downsample",
        lambda *_args, **_kwargs: (points, normals),
    )

    def missing_plane(*_args, **_kwargs):
        raise ValueError("insufficient dominant-normal support")

    monkeypatch.setattr(profile_module, "_deterministic_dominant_plane", missing_plane)

    assert _detect_interior_formed_transition_strips(
        points,
        spacing_mm=1.0,
        start_index=1,
    ) == []


def test_prismatic_completion_keeps_long_junctions_and_rejects_medium_end_residual(monkeypatch):
    points = np.vstack(
        (
            np.zeros((500, 3), dtype=np.float64),
            np.tile(np.asarray([[300.0, 100.0, 50.0]]), (500, 1)),
        )
    )

    def strip(strip_id: str, midpoint_x: float) -> BendSupportStrip:
        return BendSupportStrip(
            strip_id=strip_id,
            strip_family="discrete_fold_strip",
            axis_direction=[0.0, 0.0, 1.0],
            axis_line_hint={"midpoint": [midpoint_x, 0.0, 0.0]},
            adjacent_flange_normals=[],
            dominant_angle_deg=90.0,
            radius_mm=None,
            support_patch_ids=[strip_id],
            support_point_count=100,
            cross_fold_width_mm=5.0,
            along_axis_span_mm=60.0,
            support_bbox_mm=[1.0, 1.0, 60.0],
            observability_status="supported",
            merge_provenance={},
            stability_score=1.0,
            confidence=1.0,
        )

    monkeypatch.setattr(profile_module, "_voxel_downsample", lambda *args, **kwargs: (points, None))
    monkeypatch.setattr(
        profile_module,
        "_line_supported_by_reference",
        lambda start, end, reference: (
            True,
            {
                "supported_axial_interval_mm": [40.0, 55.0]
                if float(start[0]) < 5.0
                else [2.0, 58.0],
                "near_reference_point_count": 100,
            },
        ),
    )
    import plane_lattice_bend_recovery

    monkeypatch.setattr(plane_lattice_bend_recovery, "extract_deterministic_planes", lambda *a, **k: [])
    monkeypatch.setattr(
        plane_lattice_bend_recovery,
        "build_supported_plane_intersections",
        lambda *a, **k: {
            "candidates": [
                {
                    "candidate_id": "L1",
                    "span_mm": 200.0,
                    "span_endpoints": [[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
                    "angle_deg": 90.0,
                    "incident_plane_support_counts": [500, 500],
                    "incident_plane_cross_support_widths_mm": [10.0, 10.0],
                },
                {
                    "candidate_id": "L2",
                    "span_mm": 190.0,
                    "span_endpoints": [[0.0, 30.0, 0.0], [190.0, 30.0, 0.0]],
                    "angle_deg": 90.0,
                    "incident_plane_support_counts": [450, 450],
                    "incident_plane_cross_support_widths_mm": [9.0, 9.0],
                },
            ]
        },
    )

    result = _build_prismatic_plane_lattice_strips(
        points,
        selected_strips=[strip("SHORT", 0.0), strip("RESIDUAL", 10.0)],
        spacing_mm=1.0,
    )

    assert len(result) == 3
    assert [item.strip_family for item in result].count("prismatic_plane_lattice_long") == 2
    assert [item.strip_family for item in result].count("prismatic_scan_supported_short_end") == 1
    assert result[-1].support_patch_ids == ["SHORT"]


def test_prismatic_long_family_prunes_only_low_support_parallel_replica():
    def candidate(candidate_id: str, y: float, support: int) -> dict:
        return {
            "candidate_id": candidate_id,
            "span_endpoints": [[0.0, y, 0.0], [200.0, y, 0.0]],
            "incident_plane_support_counts": [support, support + 10],
        }

    result = _prune_low_support_parallel_long_candidates(
        [candidate("A", -100.0, 2000), candidate("ARTIFACT", -80.0, 1000), candidate("B", 100.0, 2100)]
    )

    assert [item["candidate_id"] for item in result] == ["A", "B"]
