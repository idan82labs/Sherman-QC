from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_exp_cad_0028_ntrc_pairwise_source_only.py"
SPEC = importlib.util.spec_from_file_location("exp28_ntrc", SCRIPT)
EXP28 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP28)


def _cad_radials() -> np.ndarray:
    # Eight centers partition [.1,.9] of a 90-degree minor arc: .15 ... .85.
    angles = np.linspace(0.15, 0.85, 8) * (math.pi / 2.0)
    return np.column_stack((np.cos(angles), np.sin(angles), np.zeros(8)))


def _normal_grid(turn_scale: float = 1.0, *, reverse_sign: bool = False) -> tuple[np.ndarray, np.ndarray]:
    cad = _cad_radials()
    base_angles = np.arctan2(cad[:, 1], cad[:, 0])
    center = float(base_angles[0])
    observed = center + turn_scale * (base_angles - center)
    row = np.column_stack((np.cos(observed), np.sin(observed), np.zeros(8)))
    if reverse_sign:
        row = -row
    return np.repeat(row[None, :, :], 8, axis=0), np.full((8, 8), 3, dtype=np.int64)


def _guard_fixture(distortion_translation: np.ndarray, distortion_rotation: np.ndarray):
    values = np.linspace(-12.0, 12.0, 9)
    guards = []
    definitions = (
        (np.array([1.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), "X"),
        (np.array([-1.0, 0.0, 0.0]), np.array([-5.0, 0.0, 0.0]), "X"),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 5.0, 0.0]), "Y"),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, -5.0, 0.0]), "Y"),
    )
    for normal, origin, kind in definitions:
        points = []
        for first in values:
            for second in values:
                if kind == "X":
                    points.append([origin[0], first, second])
                else:
                    points.append([first, origin[1], second])
        ideal = np.asarray(points, dtype=np.float64)
        distorted, _ = EXP28.apply_event_transform(
            ideal,
            np.broadcast_to(normal, ideal.shape),
            center_mm=np.zeros(3),
            translation_mm=distortion_translation,
            rotation_vector_rad=distortion_rotation,
        )
        guards.append({
            "points_mm": distorted,
            "plane_normal": normal,
            "plane_origin_mm": origin,
            "axial_bin_counts": np.full(8, 8, dtype=np.int64),
        })
    return guards


def test_preregistration_and_exact_robustness_inventory() -> None:
    prereg = ROOT / "output" / "research" / "autoresearch-loop" / "EXP_CAD_0028_PREREGISTRATION_V1.json"
    payload = EXP28.validate_preregistration(prereg)
    assert payload["source_execution_state_at_freeze"] == "NOT_RUN"
    rows = EXP28.robustness_variants()
    assert len(rows) == 39
    assert len({row["variant_id"] for row in rows}) == 39
    assert sum(row["perturbed_member"] is None for row in rows) == 15
    assert sum(row["perturbed_member"] is not None for row in rows) == 24
    assert sum(
        row["voxel_thickness_fraction"] == 0.125
        and row["partition"] == "FULL"
        and row["perturbed_member"] is None
        for row in rows
    ) == 1


def test_native_roundbend_parameter_parser_is_explicit() -> None:
    assert EXP28.validate_roundbend_native_parameters({"D1": "R1.0", "D4": "1.57079632679rad"}) == {
        "D1_token": "R1.0",
        "D1_radius": 1.0,
        "D4_token": "1.57079632679rad",
        "D4_rad": pytest.approx(math.pi / 2.0),
    }
    assert EXP28.validate_roundbend_native_parameters({"parameters": {"D1": "R1"}})["D4_rad"] is None
    with pytest.raises(ValueError):
        EXP28.validate_roundbend_native_parameters({"D1": "1.0", "D4": "90deg"})
    with pytest.raises(ValueError):
        audit = EXP28.validate_roundbend_native_parameters({"D1": "R2"})
        if abs(audit["D1_radius"] - 1.0) > EXP28.RADIUS_TOLERANCE_MM:
            raise ValueError("D1 radius gate")


def test_authentic_cad_selector_recovers_exact_six_pairs_without_scans() -> None:
    source = ROOT / "data" / "bend_corpus" / "incoming" / "2026-08-17_20_shaked_new_good_bad_pairs" / "authenticated_source" / "cad" / "42637000_B.SLDPRT"
    cadmpeg = Path.home() / ".local" / "share" / "codex-tools" / "cadmpeg" / "v0.5.5" / "cadmpeg-aarch64-apple-darwin" / "cadmpeg"
    assert EXP28.sha256_path(source) == "fdb2b5ca0a9a572f5320a3644543fca595330b0800d0e0d2ecd04e41f313e571"
    assert EXP28.sha256_path(cadmpeg) == "ee5d574a68eae7112774514a1a250658ac19818035435cf8e3757ae0ea8e3b67"
    cadir, _report = EXP28.EXP25.decode_cadir(source, cadmpeg)
    extraction = EXP28.select_geometry_consensus_events(cadir)
    assert extraction["proposal_status"] == "GEOMETRY_CONSENSUS_PROPOSAL_ONLY"
    assert extraction["active_roundbend_count"] == extraction["event_count"] == 6
    assert extraction["stale_bends_property_audit_only"] == 0
    assert sorted(round(row["span_mm"]) for row in extraction["events"]) == [45, 45, 130, 178, 178, 210]
    actual = sorted(
        (
            *sorted(int(skin["face_id"].rsplit("#", 1)[1]) for skin in row["skins"]),
            round(row["span_mm"]),
        )
        for row in extraction["events"]
    )
    assert actual == sorted([
        (1330, 1346, 178),
        (1235, 1465, 45),
        (552, 1251, 45),
        (2259, 2269, 178),
        (1379, 2327, 210),
        (1485, 1499, 130),
    ])
    assert all(row["minor_arc_deg"] == pytest.approx(90.0, abs=1e-6) for row in extraction["events"])
    _body, _region, _shell, shell_faces = EXP28._active_shell_face_ids(cadir["model"])
    panels = [
        EXP28.build_event_panels(cadir["model"], event, shell_faces)
        for event in extraction["events"]
    ]
    actionability = EXP28.cad_guard_actionability_preflight(extraction, panels)
    assert actionability["status"] == "TERMINAL_FOUR_FINITE_GUARDS_UNAVAILABLE"
    assert actionability["required_guard_count"] == 24
    assert actionability["actionable_guard_count"] == 12
    assert actionability["unavailable_guard_count"] == 12
    failed_extents = sorted(round(row["interior_extent_mm"], 6) for row in actionability["rows"] if not row["actionable"])
    assert failed_extents == [1.0] * 6 + [2.5] * 6
    assert actionability["source_scans_loaded"] is False


def test_active_shell_allows_empty_region_but_requires_one_shell_total() -> None:
    model = {
        "configurations": [{"active": True, "bodies": ["B"]}],
        "bodies": [{"id": "B", "regions": ["EMPTY", "SOLID"]}],
        "regions": [
            {"id": "EMPTY", "body": "B", "shells": []},
            {"id": "SOLID", "body": "B", "shells": ["S"]},
        ],
        "shells": [{"id": "S", "region": "SOLID", "faces": ["F"]}],
        "faces": [{"id": "F"}],
    }
    body, region, shell, faces = EXP28._active_shell_face_ids(model)
    assert (body, region, shell, faces) == ("B", "SOLID", "S", {"F"})
    model["shells"].append({"id": "S2", "region": "EMPTY", "faces": ["F"]})
    model["regions"][0]["shells"] = ["S2"]
    with pytest.raises(ValueError):
        EXP28._active_shell_face_ids(model)


def test_voxel_is_centroid_and_summed_unit_normal_not_first_point() -> None:
    points = np.asarray([[0.01, 0.02, 0.03], [0.09, 0.08, 0.07], [1.01, 0.0, 0.0]])
    normals = np.asarray([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    out_points, out_normals, ids = EXP28.canonical_voxel_aggregate(points, normals, 1.0)
    np.testing.assert_allclose(out_points[0], [0.05, 0.05, 0.05])
    np.testing.assert_allclose(out_normals[0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0])
    np.testing.assert_array_equal(ids, [0, 2])


def test_partitions_use_original_ids_and_fixed_cad_origin() -> None:
    points = np.asarray([[0.01, 0.01, 0.01], [0.20, 0.01, 0.01], [0.01, 0.20, 0.01]])
    ids = np.asarray([10, 11, 12])
    even = EXP28.partition_mask(points, ids, "RAW_EVEN", body_bbox_min_mm=np.zeros(3), thickness_mm=1.6)
    np.testing.assert_array_equal(even, [True, False, True])
    parity = EXP28.partition_mask(points, ids, "SPATIAL_PARITY_0", body_bbox_min_mm=np.zeros(3), thickness_mm=1.6)
    np.testing.assert_array_equal(parity, [True, False, False])


def test_guard_fit_has_no_axial_translation_and_three_nonincreasing_steps() -> None:
    translation = np.asarray([0.03, -0.02, 0.0])
    rotation = np.radians(np.asarray([0.04, -0.03, 0.02]))
    result = EXP28.solve_guard_only_nuisance(
        _guard_fixture(translation, rotation),
        event_axis=np.asarray([0.0, 0.0, 1.0]),
        event_center_mm=np.zeros(3),
        thickness_mm=1.5,
    )
    assert result["status"] == "FIT", result
    assert len(result["iterations"]) == 3
    assert abs(result["axial_translation_mm_exact"]) <= 1e-12
    assert all(row["rank"] == 5 and row["condition"] <= 1e4 for row in result["iterations"])
    assert all(row["guard_weight_totals"] == [0.25, 0.25, 0.25, 0.25] for row in result["iterations"])
    assert all(
        row["huber_objective_after"] <= row["huber_objective_before"] * (1 + 1e-12) + 1e-15
        for row in result["iterations"]
    )


def test_guard_huber_mass_is_balanced_with_unequal_counts_and_outlier() -> None:
    guards = _guard_fixture(np.asarray([0.02, -0.01, 0.0]), np.zeros(3))
    guards[0]["points_mm"] = guards[0]["points_mm"][:64]
    guards[1]["points_mm"] = np.vstack([guards[1]["points_mm"], [[-4.0, 100.0, 100.0]]])
    result = EXP28.solve_guard_only_nuisance(
        guards, event_axis=np.asarray([0, 0, 1]), event_center_mm=np.zeros(3), thickness_mm=1.5
    )
    assert result["status"] == "FIT", result
    assert all(row["guard_weight_totals"] == [0.25, 0.25, 0.25, 0.25] for row in result["iterations"])


def test_guard_fit_abstains_on_axial_occupancy_and_saturation() -> None:
    guards = _guard_fixture(np.zeros(3), np.zeros(3))
    guards[0]["axial_bin_counts"] = np.asarray([1, 1, 1, 1, 1, 0, 0, 0])
    assert EXP28.solve_guard_only_nuisance(
        guards, event_axis=np.asarray([0, 0, 1]), event_center_mm=np.zeros(3), thickness_mm=1.5
    )["reason"] == "GUARD_AXIAL_OCCUPANCY"
    saturated = EXP28.solve_guard_only_nuisance(
        _guard_fixture(np.asarray([0.19, 0.0, 0.0]), np.zeros(3)),
        event_axis=np.asarray([0, 0, 1]),
        event_center_mm=np.zeros(3),
        thickness_mm=1.5,
    )
    assert saturated["status"] == "ABSTAIN"


@pytest.mark.parametrize("scale,expected_q", [(1.0, 0.0), (60.0 / 63.0, 3.0), (66.0 / 63.0, 3.0)])
def test_cylinder_normal_turn_analytic_under_over(scale: float, expected_q: float) -> None:
    normals, counts = _normal_grid(scale)
    result = EXP28.skin_turn_statistic(
        normals, counts, cad_radial_cell_vectors=_cad_radials(), axis=np.asarray([0, 0, 1])
    )
    assert result["status"] == "MEASURED"
    assert result["Q_median_absolute_signed_error_deg"] == pytest.approx(expected_q, abs=1e-9)


def test_one_global_flip_and_axis_reversal_are_invariant() -> None:
    normals, counts = _normal_grid(1.0, reverse_sign=True)
    first = EXP28.skin_turn_statistic(
        normals, counts, cad_radial_cell_vectors=_cad_radials(), axis=np.asarray([0, 0, 1])
    )
    second = EXP28.skin_turn_statistic(
        normals, counts, cad_radial_cell_vectors=_cad_radials(), axis=np.asarray([0, 0, -1])
    )
    assert first["global_normal_flip"] == -1
    assert first["Q_median_absolute_signed_error_deg"] == pytest.approx(0.0, abs=1e-10)
    assert second["Q_median_absolute_signed_error_deg"] == pytest.approx(
        first["Q_median_absolute_signed_error_deg"], abs=1e-10
    )


def test_nonzero_underbend_survives_axis_and_endpoint_reversal() -> None:
    normals, counts = _normal_grid(60.0 / 63.0)
    cad = _cad_radials()
    forward = EXP28.skin_turn_statistic(
        normals, counts, cad_radial_cell_vectors=cad, axis=np.asarray([0, 0, 1])
    )
    reversed_result = EXP28.skin_turn_statistic(
        normals[:, ::-1], counts[:, ::-1], cad_radial_cell_vectors=cad[::-1], axis=np.asarray([0, 0, -1])
    )
    assert forward["Q_median_absolute_signed_error_deg"] == pytest.approx(3.0, abs=1e-9)
    assert reversed_result["Q_median_absolute_signed_error_deg"] == pytest.approx(3.0, abs=1e-9)
    assert reversed_result["signed_median_error_deg"] == pytest.approx(
        forward["signed_median_error_deg"], abs=1e-9
    )


def test_common_axial_normal_tilt_is_projected_out_of_turn() -> None:
    normals, counts = _normal_grid(1.0)
    angle = math.radians(20.0)
    tilted = math.cos(angle) * normals + math.sin(angle) * np.asarray([0.0, 0.0, 1.0])
    result = EXP28.skin_turn_statistic(
        tilted, counts, cad_radial_cell_vectors=_cad_radials(), axis=np.asarray([0, 0, 1])
    )
    assert result["status"] == "MEASURED"
    assert result["median_radial_normal_error_deg"] == pytest.approx(20.0, abs=1e-9)
    assert result["Q_median_absolute_signed_error_deg"] == pytest.approx(0.0, abs=1e-9)


def test_q_is_median_absolute_error_not_absolute_median() -> None:
    cad = _cad_radials()
    normals = np.empty((8, 8, 3), dtype=np.float64)
    base_angles = np.arctan2(cad[:, 1], cad[:, 0])
    for axial in range(8):
        target = 2.0 if axial % 2 == 0 else -2.0
        scale = (63.0 + target) / 63.0
        angles = base_angles[0] + scale * (base_angles - base_angles[0])
        normals[axial] = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(8)))
    result = EXP28.skin_turn_statistic(
        normals, np.full((8, 8), 3), cad_radial_cell_vectors=cad, axis=np.asarray([0, 0, 1])
    )
    assert abs(result["signed_median_error_deg"]) <= 1e-10
    assert result["Q_median_absolute_signed_error_deg"] == pytest.approx(2.0, abs=1e-9)


def test_common_rigid_rotation_preserves_turn_and_q() -> None:
    normals, counts = _normal_grid(60.0 / 63.0)
    cad = _cad_radials()
    rotation = EXP28.rodrigues(np.radians(np.asarray([12.0, -7.0, 4.0])))
    baseline = EXP28.skin_turn_statistic(
        normals, counts, cad_radial_cell_vectors=cad, axis=np.asarray([0, 0, 1])
    )
    rotated = EXP28.skin_turn_statistic(
        normals @ rotation.T,
        counts,
        cad_radial_cell_vectors=cad @ rotation.T,
        axis=np.asarray([0, 0, 1]) @ rotation.T,
    )
    assert rotated["Q_median_absolute_signed_error_deg"] == pytest.approx(
        baseline["Q_median_absolute_signed_error_deg"], abs=1e-9
    )
    half_turn = EXP28.rodrigues(np.asarray([math.pi, 0.0, 0.0]))
    rotated_180 = EXP28.skin_turn_statistic(
        normals @ half_turn.T,
        counts,
        cad_radial_cell_vectors=cad @ half_turn.T,
        axis=np.asarray([0, 0, 1]) @ half_turn.T,
    )
    assert rotated_180["signed_median_error_deg"] == pytest.approx(
        baseline["signed_median_error_deg"], abs=1e-9
    )
    assert rotated_180["Q_median_absolute_signed_error_deg"] == pytest.approx(
        baseline["Q_median_absolute_signed_error_deg"], abs=1e-9
    )


def test_scan_swap_exchanges_q_and_negates_d() -> None:
    variants = [np.ones(39), np.ones(39)]
    forward = EXP28.paired_event_direction(
        [2.0, 2.0], [1.0, 1.0], variants,
        first_skin_signed_median_e=[-2.0, -2.0], second_skin_signed_median_e=[-1.0, -1.0],
    )
    reverse = EXP28.paired_event_direction(
        [1.0, 1.0], [2.0, 2.0], [-variants[0], -variants[1]],
        first_skin_signed_median_e=[-1.0, -1.0], second_skin_signed_median_e=[-2.0, -2.0],
    )
    assert forward["direction_sign_n009_minus_n010"] == 1
    assert reverse["direction_sign_n009_minus_n010"] == -1
    assert [row["Q_n009_deg"] for row in forward["skins"]] == [
        row["Q_n010_deg"] for row in reverse["skins"]
    ]


def test_skin_list_swap_invariant_and_opposing_within_case_signs_abstain() -> None:
    variants = [np.ones(39), np.full(39, 1.8)]
    first = EXP28.paired_event_direction(
        [2.0, 3.0], [1.0, 1.2], variants,
        first_skin_signed_median_e=[-2.0, -3.0], second_skin_signed_median_e=[-1.0, -1.2],
    )
    swapped = EXP28.paired_event_direction(
        [3.0, 2.0], [1.2, 1.0], variants[::-1],
        first_skin_signed_median_e=[-3.0, -2.0], second_skin_signed_median_e=[-1.2, -1.0],
    )
    assert first["direction_sign_n009_minus_n010"] == swapped["direction_sign_n009_minus_n010"] == 1
    conflict = EXP28.paired_event_direction(
        [2.0, 2.0], [1.0, 1.0], [np.ones(39), np.ones(39)],
        first_skin_signed_median_e=[-2.0, 2.0], second_skin_signed_median_e=[-1.0, -1.0],
    )
    assert conflict["status"] == "ABSTAIN"
    assert conflict["n009_within_case_skin_signed_error_consensus"]["reason"] == "OPPOSING_INNER_OUTER_SIGNED_ERROR"
    mixed = EXP28.paired_event_direction(
        [2.0, 2.0], [1.0, 1.0], [np.ones(39), np.ones(39)],
        first_skin_signed_median_e=[0.0, 2.0], second_skin_signed_median_e=[-1.0, -1.0],
    )
    assert mixed["status"] == "ABSTAIN"
    assert mixed["n009_within_case_skin_signed_error_consensus"]["reason"] == "MIXED_NEUTRAL_AND_DIRECTIONAL_SKINS"


def test_all_39_robustness_rows_must_match_central_direction() -> None:
    variants = EXP28.robustness_variants()
    rows = [
        {"variant_id": row["variant_id"], "status": "DIRECTIONAL", "direction_sign_n009_minus_n010": 1}
        for row in variants
    ]
    assert EXP28.robustness_direction_gate(rows)["status"] == "PASS"
    rows[-1]["direction_sign_n009_minus_n010"] = -1
    failed = EXP28.robustness_direction_gate(rows)
    assert failed["status"] == "FAIL_TERMINAL_ABSTAIN"
    assert failed["direction_invariant"] is False
    rows[-1]["variant_id"] = "BOGUS"
    with pytest.raises(ValueError, match="exact frozen 39-ID inventory"):
        EXP28.robustness_direction_gate(rows)


def test_missing_cells_abstain_and_do_not_become_defect_evidence() -> None:
    normals, counts = _normal_grid(1.0)
    counts[:3] = 0
    result = EXP28.skin_turn_statistic(
        normals, counts, cad_radial_cell_vectors=_cad_radials(), axis=np.asarray([0, 0, 1])
    )
    assert result["status"] == "ABSTAIN"
    assert result["Q_median_absolute_signed_error_deg"] is None
