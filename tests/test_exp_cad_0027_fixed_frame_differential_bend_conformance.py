from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_exp_cad_0027_fixed_frame_differential_bend_conformance.py"
SPEC = importlib.util.spec_from_file_location("exp27_conformance", PATH)
EXP27 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP27)


def test_density_cap_keeps_first_raw_identity_deterministically() -> None:
    points = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [1.10, 0.10, 0.10],
            [-0.10, -0.10, -0.10],
        ],
        dtype=np.float64,
    )
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    capped_points, capped_normals, indices = EXP27.density_cap_fixed_identities(
        points, normals, 1.0
    )
    assert indices.tolist() == [0, 2, 3]
    assert np.allclose(capped_points, points[indices])
    assert np.allclose(capped_normals, normals[indices])
    assert EXP27.canonical_int64_array_sha256(indices) == EXP27.canonical_int64_array_sha256(
        indices.copy()
    )


def test_minor_arc_coordinates_are_fixed_and_oriented() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([0.0, 1.0, 0.0])
    values = np.array(
        [
            first,
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            second,
            [-1.0, 0.0, 0.0],
        ]
    )
    angles, theta = EXP27.angle_coordinates(values, first, second, axis)
    assert np.isclose(theta, np.pi / 2.0)
    assert np.allclose(angles[:3], [0.0, np.pi / 4.0, np.pi / 2.0])
    assert angles[3] > theta


def test_constant_planar_normals_cannot_fill_native_cylinder_angle_cells() -> None:
    side = {
        "axis": np.array([0.0, 0.0, 1.0]),
        "axis_foot_mm": np.zeros(3),
        "line_start_mm": np.zeros(3),
        "line_end_mm": np.array([0.0, 0.0, 40.0]),
        "span_mm": 40.0,
        "radius_mm": 3.0,
        "guards": [
            {"radial_unit": np.array([1.0, 0.0, 0.0])},
            {"radial_unit": np.array([0.0, 1.0, 0.0])},
        ],
    }
    points = []
    for axial in np.linspace(3.0, 37.0, 16):
        for angle in np.linspace(0.10, np.pi / 2.0 - 0.10, 24):
            points.append([3.0 * np.cos(angle), 3.0 * np.sin(angle), axial])
    points = np.asarray(points)
    # A planar strip inside the old radial annulus has one constant normal.
    normals = np.tile([1.0, 0.0, 0.0], (len(points), 1))
    arrays = EXP27.cylinder_base_arrays(points, normals, side, 2.0)
    accepted_angles = arrays["angular_fraction"][arrays["base_mask"]]
    angular_indices = EXP27._bin_indices(
        accepted_angles,
        EXP27.CYLINDER_ANGLE_EDGE_FRACTION,
        1.0 - EXP27.CYLINDER_ANGLE_EDGE_FRACTION,
        EXP27.ANGULAR_BIN_COUNT,
    )
    occupied = np.bincount(angular_indices, minlength=EXP27.ANGULAR_BIN_COUNT)
    assert np.mean(occupied >= EXP27.MIN_POINTS_PER_CELL) < 0.60


def synthetic_event_and_scan(*, local_bump_mm: float = 0.0) -> tuple[dict, np.ndarray, np.ndarray]:
    thickness = 2.0
    span = 40.0
    axis = np.array([0.0, 0.0, 1.0])
    translation = np.array([0.20, -0.10, 0.0])

    def side(radius: float, ordinal: int) -> dict:
        return {
            "cylinder_face_id": f"CYL_{ordinal}",
            "cylinder_surface_id": f"SURF_{ordinal}",
            "radius_mm": radius,
            "axis": axis,
            "axis_foot_mm": np.zeros(3),
            "line_start_mm": np.zeros(3),
            "line_end_mm": np.array([0.0, 0.0, span]),
            "span_mm": span,
            "theta_cad_rad": np.pi / 2.0,
            "guards": [
                {
                    "face_id": f"PX_{ordinal}",
                    "surface_id": f"PSX_{ordinal}",
                    "shared_edge_id": f"EX_{ordinal}",
                    "edge_midpoint_mm": np.array([radius, 0.0, span / 2.0]),
                    "plane_origin_mm": np.array([radius, 0.0, 0.0]),
                    "plane_normal": np.array([1.0, 0.0, 0.0]),
                    "radial_unit": np.array([1.0, 0.0, 0.0]),
                    "interior_direction": np.array([0.0, -1.0, 0.0]),
                },
                {
                    "face_id": f"PY_{ordinal}",
                    "surface_id": f"PSY_{ordinal}",
                    "shared_edge_id": f"EY_{ordinal}",
                    "edge_midpoint_mm": np.array([0.0, radius, span / 2.0]),
                    "plane_origin_mm": np.array([0.0, radius, 0.0]),
                    "plane_normal": np.array([0.0, 1.0, 0.0]),
                    "radial_unit": np.array([0.0, 1.0, 0.0]),
                    "interior_direction": np.array([-1.0, 0.0, 0.0]),
                },
            ],
        }

    event = {
        "event_id": "SYNTHETIC_EVENT",
        "sheet_thickness_mm": thickness,
        "line_start_mm": np.zeros(3),
        "line_end_mm": np.array([0.0, 0.0, span]),
        "axis": axis,
        "span_mm": span,
        "sides": [side(1.0, 1), side(3.0, 2)],
    }
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    axial_centers = np.linspace(3.0, 37.0, 8)
    angular_centers = np.linspace(0.14, np.pi / 2.0 - 0.14, 8)
    for radius in (1.0, 3.0):
        for axial_index, axial_value in enumerate(axial_centers):
            bump = local_bump_mm if axial_index == 4 else 0.0
            for angle in angular_centers:
                radial = np.array([np.cos(angle), np.sin(angle), 0.0])
                for perturbation in (-0.03, 0.0, 0.03):
                    points.append(translation + (radius + bump + perturbation) * radial + [0.0, 0.0, axial_value])
                    normals.append(radial)
        # Fixed plane guards and near-seam witness strips for each sheet side.
        for axial_value in axial_centers:
            for inward in (4.2, 6.2, 8.2, 10.2, 11.8, 1.2, 2.0, 2.8):
                points.append(np.array([radius, -inward, axial_value]) + translation)
                normals.append(np.array([1.0, 0.0, 0.0]))
                points.append(np.array([-inward, radius, axial_value]) + translation)
                normals.append(np.array([0.0, 1.0, 0.0]))
    return event, np.asarray(points), np.asarray(normals)


def test_fixed_plane_translation_correction_does_not_create_anomaly() -> None:
    event, points, normals = synthetic_event_and_scan()
    result, _arrays = EXP27.event_conformance(points, normals, event)
    assert result["status"] == "MEASURED"
    assert result["measured_surface_count"] == 2
    for surface in result["surfaces"]:
        assert surface["status"] == "MEASURED_FIXED_FRAME"
        assert surface["bounded_cross_section_translation_norm_mm"] < 0.5
        assert abs(surface["corrected_radius_bias_mm"]) < 0.05
        assert surface["angle_error_p90_deg"] < 0.1
    assert result["event_anomaly_score"] < 0.05


def test_localized_radius_bump_increases_fixed_frame_form_score() -> None:
    baseline_event, baseline_points, baseline_normals = synthetic_event_and_scan()
    bumped_event, bumped_points, bumped_normals = synthetic_event_and_scan(local_bump_mm=0.8)
    baseline, _ = EXP27.event_conformance(baseline_points, baseline_normals, baseline_event)
    bumped, _ = EXP27.event_conformance(bumped_points, bumped_normals, bumped_event)
    assert bumped["event_anomaly_score"] > baseline["event_anomaly_score"] + 0.10
    assert max(
        surface["components"]["axial_form_over_thickness"]
        for surface in bumped["surfaces"]
    ) > 0.10


def test_real_cadir_has_complete_analytic_tangent_guard_topology() -> None:
    source = ROOT / "data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/50476000_A.SLDPRT"
    cadmpeg = Path(
        "/Users/idantal/.local/share/codex-tools/cadmpeg/v0.5.5/"
        "cadmpeg-aarch64-apple-darwin/cadmpeg"
    )
    cadir, _report = EXP27.EXP25.decode_cadir(source, cadmpeg)
    extraction = EXP27.EXP17.extract_native_roundbend_patch_axes(cadir)
    bundles = EXP27.native_event_bundles(cadir, extraction)
    assert len(bundles) == 4
    assert all(len(event["sides"]) == 2 for event in bundles)
    assert all(len(side["guards"]) == 2 for event in bundles for side in event["sides"])
    assert {
        guard["face_id"]
        for event in bundles
        for side in event["sides"]
        for guard in side["guards"]
    } >= {"sldprt:brep:face#859", "sldprt:brep:face#871"}
