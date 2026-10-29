from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_exp_cad_0027_v2_paired_rotating_normal_conformance.py"
SPEC = importlib.util.spec_from_file_location("exp27_v2", PATH)
EXP27 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(EXP27)


def synthetic_event_and_scan(
    *,
    normal_turn_scale: float = 1.0,
    radial_bias_mm: float = 0.0,
    omit_outer_skin: bool = False,
    omit_guards: bool = False,
    replicates_per_cell: int = 3,
    guard_translation: tuple[float, float] = (0.0, 0.0),
) -> tuple[dict, np.ndarray, np.ndarray]:
    span = 40.0
    axis = np.array([0.0, 0.0, 1.0])

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
        "sheet_thickness_mm": 2.0,
        "line_start_mm": np.zeros(3),
        "line_end_mm": np.array([0.0, 0.0, span]),
        "axis": axis,
        "span_mm": span,
        "sides": [side(1.0, 1), side(3.0, 2)],
    }
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    angular_centres = EXP27.ANGULAR_LOW_FRACTION + (
        np.arange(EXP27.ANGULAR_BIN_COUNT) + 0.5
    ) / EXP27.ANGULAR_BIN_COUNT * (
        EXP27.ANGULAR_HIGH_FRACTION - EXP27.ANGULAR_LOW_FRACTION
    )
    angles = angular_centres * (np.pi / 2.0)
    axial_centres = (
        np.arange(EXP27.AXIAL_BIN_COUNT) + 0.5
    ) / EXP27.AXIAL_BIN_COUNT * span
    for side_index, radius in enumerate((1.0, 3.0)):
        if side_index == 1 and omit_outer_skin:
            continue
        expected_sign = -1.0 if side_index == 0 else 1.0
        for axial in axial_centres:
            for angle in angles:
                # Scale about the arc midpoint.  This preserves the normal
                # sign/radial-error gate while creating a controlled error in
                # total normal turn.
                normal_angle = np.pi / 4.0 + normal_turn_scale * (angle - np.pi / 4.0)
                normal = expected_sign * np.array(
                    [np.cos(normal_angle), np.sin(normal_angle), 0.0]
                )
                radial = np.array([np.cos(angle), np.sin(angle), 0.0])
                for perturbation in np.linspace(-0.02, 0.02, replicates_per_cell):
                    points.append((radius + radial_bias_mm + perturbation) * radial + [0.0, 0.0, axial])
                    normals.append(normal)

    # Four fixed tangent-plane guards.  Three witnesses per axial cell are
    # enough for the inherited V1 actionability gate.
    dx, dy = guard_translation
    if not omit_guards:
        for side_row in event["sides"]:
            radius = float(side_row["radius_mm"])
            for axial in axial_centres:
                for jitter in np.linspace(-0.02, 0.02, replicates_per_cell):
                    points.append(np.array([radius + dx, -8.0 + jitter, axial]))
                    normals.append(np.array([1.0, 0.0, 0.0]))
                    points.append(np.array([-8.0 + jitter, radius + dy, axial]))
                    normals.append(np.array([0.0, 1.0, 0.0]))
    return event, np.asarray(points), np.asarray(normals)


def measured_event(**kwargs: object) -> dict:
    event, points, normals = synthetic_event_and_scan(**kwargs)
    result, _arrays = EXP27.event_normal_turn_measurement(points, normals, event)
    return result


def rewrite_skin_normals(
    event: dict,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    skin_index: int,
    angular_offsets_deg: list[float],
) -> np.ndarray:
    result = normals.copy()
    side = sorted(event["sides"], key=lambda row: float(row["radius_mm"]))[skin_index]
    geometry = EXP27._radial_geometry(points, side)
    mask = (
        (geometry["axial_mm"] >= 0.0)
        & (geometry["axial_mm"] <= float(side["span_mm"]))
        & (geometry["angular_fraction"] >= EXP27.ANGULAR_LOW_FRACTION)
        & (geometry["angular_fraction"] <= EXP27.ANGULAR_HIGH_FRACTION)
        & (np.abs(geometry["radial_residual_mm"]) <= EXP27.CENTRAL_TUBE_MM)
    )
    angular_indices = EXP27._bin_indices(
        geometry["angular_fraction"][mask],
        EXP27.ANGULAR_LOW_FRACTION,
        EXP27.ANGULAR_HIGH_FRACTION,
        EXP27.ANGULAR_BIN_COUNT,
    )
    radial_angles = np.arctan2(
        geometry["radial_unit"][mask, 1], geometry["radial_unit"][mask, 0]
    )
    sign = -1.0 if skin_index == 0 else 1.0
    output_ids = np.flatnonzero(mask)
    for local_index, output_id in enumerate(output_ids):
        angle = radial_angles[local_index] + np.radians(angular_offsets_deg[angular_indices[local_index]])
        result[output_id] = sign * np.array([np.cos(angle), np.sin(angle), 0.0])
    return result


def test_density_selection_is_nearest_centre_and_permutation_invariant() -> None:
    points = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.49, 0.49, 0.49],  # closest to centre of voxel [0,0,0]
            [1.49, 0.49, 0.49],
            [2.49, 0.49, 0.49],
            [3.49, 0.49, 0.49],
        ]
    )
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 1.0],  # invalid normal is removed first
            [0.0, 0.0, 0.0],     # zero normal is removed first
        ]
    )
    selected_points, selected_normals, _ids = EXP27.density_cap_canonical(
        points, normals, 1.0
    )
    permutation = np.array([4, 2, 0, 3, 1])
    shuffled_points, shuffled_normals, _shuffled_ids = EXP27.density_cap_canonical(
        points[permutation], normals[permutation], 1.0
    )
    assert np.allclose(selected_points, [[0.49, 0.49, 0.49], [1.49, 0.49, 0.49]])
    assert np.allclose(selected_points, shuffled_points)
    assert np.allclose(selected_normals, shuffled_normals)


def test_uniform_radial_bias_is_diagnostic_only_and_cannot_create_direction() -> None:
    biased = measured_event(radial_bias_mm=0.20)
    nominal = measured_event(radial_bias_mm=0.0)
    paired = EXP27.paired_event_direction(biased, nominal)
    assert biased["status"] == "MEASURED_BOTH_SKINS"
    assert nominal["status"] == "MEASURED_BOTH_SKINS"
    assert all(
        abs(surface["absolute_radius_bias_mm_diagnostic_only"] - 0.20) < 1e-8
        for surface in biased["surfaces"]
    )
    assert paired["status"] == "ABSTAIN"
    assert paired["direction_sign_n011_minus_n012"] == 0


def test_signed_normal_turn_is_monotone_and_matches_frozen_formula() -> None:
    nominal = measured_event(normal_turn_scale=1.0)
    contracted = measured_event(normal_turn_scale=0.75)
    assert nominal["status"] == "MEASURED_BOTH_SKINS"
    assert contracted["status"] == "MEASURED_BOTH_SKINS"
    for surface in nominal["surfaces"]:
        assert np.allclose(
            surface["observed_normal_turn_deg_by_axial_bin"],
            [66.15] * EXP27.AXIAL_BIN_COUNT,
            atol=1e-8,
        )
        assert surface["monotone_increment_count_by_axial_bin"] == [7] * 8
    for surface in contracted["surfaces"]:
        assert np.allclose(
            surface["observed_normal_turn_deg_by_axial_bin"],
            [49.6125] * EXP27.AXIAL_BIN_COUNT,
            atol=1e-8,
        )
        assert surface["eligible_axial_bin_count"] == 8


def test_pair_direction_is_swap_antisymmetric() -> None:
    contracted = measured_event(normal_turn_scale=0.75)
    nominal = measured_event(normal_turn_scale=1.0)
    forward = EXP27.paired_event_direction(contracted, nominal)
    reverse = EXP27.paired_event_direction(nominal, contracted)
    assert forward["status"] == "QUALIFIED_EVENT_DIRECTION"
    assert reverse["status"] == "QUALIFIED_EVENT_DIRECTION"
    assert forward["direction_sign_n011_minus_n012"] == 1
    assert reverse["direction_sign_n011_minus_n012"] == -1
    assert np.isclose(
        forward["surfaces"][0]["paired_delta_median_deg"],
        -reverse["surfaces"][0]["paired_delta_median_deg"],
    )


def test_tight_tubes_do_not_overlap_or_claim_mid_skin_points() -> None:
    event, points, normals = synthetic_event_and_scan()
    points = np.vstack([points, [2.0, 0.0, 20.0]])
    normals = np.vstack([normals, [1.0, 0.0, 0.0]])
    result, arrays = EXP27.event_normal_turn_measurement(
        points, normals, event, tube_mm=0.45
    )
    assert result["strict_nonoverlap_verified"] is True
    first = arrays["surfaces"][0]["owner_mask"]
    second = arrays["surfaces"][1]["owner_mask"]
    assert not np.any(first & second)
    assert not first[-1] and not second[-1]


def test_missing_skin_forces_event_and_pair_abstention() -> None:
    missing = measured_event(omit_outer_skin=True)
    nominal = measured_event()
    assert missing["status"] == "ABSTAIN"
    assert "BOTH_SKINS_NOT_MEASURED" in missing["abstention_reasons"]
    paired = EXP27.paired_event_direction(missing, nominal)
    assert paired["status"] == "ABSTAIN"
    assert len(paired["surfaces"]) == 2
    assert "BOTH_CASE_EVENTS_MUST_BE_MEASURED_BOTH_SKINS" in paired["abstention_reasons"]


def test_topology_contract_rejects_nonconcentric_skin_and_endpoint_mismatch() -> None:
    event, _points, _normals = synthetic_event_and_scan()
    shifted = copy.deepcopy(event)
    shifted["sides"][1]["axis_foot_mm"] = np.array([0.001, 0.0, 0.0])
    with pytest.raises(ValueError, match="axis foot"):
        EXP27.validate_native_event_contract(shifted)
    endpoint = copy.deepcopy(event)
    endpoint["sides"][1]["guards"][1]["radial_unit"] = np.array([0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="arc endpoints"):
        EXP27.validate_native_event_contract(endpoint)


def test_shared_delta_is_diagnostic_and_abstains_above_frozen_norm() -> None:
    observed = measured_event(guard_translation=(0.20, -0.10))
    diagnostic = observed["shared_event_plane_translation"]
    assert diagnostic["status"] == "OBSERVED_DIAGNOSTIC_ONLY"
    assert diagnostic["rank"] == 2
    assert diagnostic["nonzero_singular_condition"] <= 10.0
    assert diagnostic["used_for_score_correction"] is False
    too_large = measured_event(guard_translation=(1.30, 1.30))
    assert too_large["status"] == "ABSTAIN"
    assert too_large["shared_event_plane_translation"]["status"] == "ABSTAIN"


def test_pose_variant_inventory_is_exactly_24() -> None:
    variants = EXP27.individual_pose_variants()
    assert len(variants) == 24
    assert len({row["variant_id"] for row in variants}) == 24
    assert {row["case_index"] for row in variants} == {0, 1}


@pytest.mark.parametrize(
    "offsets",
    [
        # Alternating normals are locally radial but not monotone in CAD order.
        [15.0, -15.0, 15.0, -15.0, 15.0, -15.0, 15.0, -15.0],
        # A four-facet staircase has zero turn inside every paired angular cell.
        [4.725, -4.725, 4.725, -4.725, 4.725, -4.725, 4.725, -4.725],
    ],
)
def test_nonmonotone_and_faceted_impostors_abstain(offsets: list[float]) -> None:
    event, points, normals = synthetic_event_and_scan()
    normals = rewrite_skin_normals(
        event, points, normals, skin_index=0, angular_offsets_deg=offsets
    )
    result, _arrays = EXP27.event_normal_turn_measurement(points, normals, event)
    assert result["status"] == "ABSTAIN"
    assert result["surfaces"][0]["status"] == "ABSTAIN"
    assert result["surfaces"][0]["eligible_axial_bin_count"] < 6


def test_planar_constant_normal_impostor_abstains() -> None:
    event, points, normals = synthetic_event_and_scan()
    geometry = EXP27._radial_geometry(points, event["sides"][0])
    inner = np.abs(geometry["radial_residual_mm"]) <= EXP27.CENTRAL_TUBE_MM
    normals[inner] = np.array([-1.0, 0.0, 0.0])
    result, _arrays = EXP27.event_normal_turn_measurement(points, normals, event)
    assert result["status"] == "ABSTAIN"
    assert result["surfaces"][0]["surface_median_radial_normal_error_deg"] > 20.0


def test_sign_consistency_below_98_percent_fails_surface() -> None:
    event, points, normals = synthetic_event_and_scan()
    geometry = EXP27._radial_geometry(points, event["sides"][0])
    inner_ids = np.flatnonzero(
        (np.abs(geometry["radial_residual_mm"]) <= EXP27.CENTRAL_TUBE_MM)
        & (geometry["angular_fraction"] >= EXP27.ANGULAR_LOW_FRACTION)
        & (geometry["angular_fraction"] <= EXP27.ANGULAR_HIGH_FRACTION)
    )
    normals[inner_ids[:4]] *= -1.0  # 188/192 = 97.9167%
    result, _arrays = EXP27.event_normal_turn_measurement(points, normals, event)
    assert result["surfaces"][0]["surface_normal_sign_consistency"] < 0.98
    assert "NORMAL_SIGN_CONSISTENCY_BELOW_0.98" in result["surfaces"][0]["abstention_reasons"]
    assert result["status"] == "ABSTAIN"


def test_median_radial_normal_error_above_20_degrees_fails_surface() -> None:
    event, points, normals = synthetic_event_and_scan()
    normals = rewrite_skin_normals(
        event, points, normals, skin_index=1, angular_offsets_deg=[25.0] * 8
    )
    result, _arrays = EXP27.event_normal_turn_measurement(points, normals, event)
    assert result["surfaces"][1]["surface_median_radial_normal_error_deg"] > 20.0
    assert "RADIAL_NORMAL_ERROR_ABOVE_20_DEG" in result["surfaces"][1]["abstention_reasons"]
    assert result["status"] == "ABSTAIN"


def test_guard_failure_cannot_enter_paired_direction() -> None:
    missing_guards = measured_event(omit_guards=True)
    nominal = measured_event()
    assert missing_guards["shared_event_plane_translation"]["status"] == "ABSTAIN"
    paired = EXP27.paired_event_direction(missing_guards, nominal)
    assert paired["status"] == "ABSTAIN"
    assert "BOTH_CASE_EVENTS_MUST_BE_MEASURED_BOTH_SKINS" in paired["abstention_reasons"]


def test_opposing_qualified_events_force_pair_abstention() -> None:
    contracted = measured_event(normal_turn_scale=0.75)
    nominal = measured_event(normal_turn_scale=1.0)
    event_a = (copy.deepcopy(contracted), copy.deepcopy(nominal))
    event_b = (copy.deepcopy(nominal), copy.deepcopy(contracted))
    event_a[0]["event_id"] = event_a[1]["event_id"] = "EVENT_A"
    event_b[0]["event_id"] = event_b[1]["event_id"] = "EVENT_B"
    paired = EXP27.pair_direction([event_a, event_b])
    assert paired["status"] == "ABSTAIN"
    assert "OPPOSING_QUALIFIED_EVENTS" in paired["abstention_reasons"]


def test_robustness_inventory_is_81_and_flip_or_abstention_fails_invariance() -> None:
    event, first_points, first_normals = synthetic_event_and_scan(
        normal_turn_scale=0.75, replicates_per_cell=6
    )
    _event2, second_points, second_normals = synthetic_event_and_scan(
        normal_turn_scale=1.0, replicates_per_cell=6
    )
    # One repeated event is sufficient to exercise the complete variant
    # cross-product inventory without pretending this is the physical run.
    result = EXP27.robustness_matrix(
        first_points, first_normals, second_points, second_normals, [event]
    )
    assert result["individual_pose_variant_count"] == 24
    assert result["variant_count_excluding_tube_cross_product"] == 27
    assert result["tube_cross_product_result_count"] == 81
    assert len(result["rows"]) == 81
    rows = copy.deepcopy(result["rows"])
    central = copy.deepcopy(result["central"])
    rows[-1]["status"] = "ABSTAIN"
    rows[-1]["direction_sign_n011_minus_n012"] = 0
    assert EXP27._rows_are_direction_invariant(rows, central) is False
    rows[-1]["status"] = "DIRECTIONAL"
    rows[-1]["direction_sign_n011_minus_n012"] = -central[
        "direction_sign_n011_minus_n012"
    ]
    assert EXP27._rows_are_direction_invariant(rows, central) is False


def test_real_cadir_and_frozen_pose_inputs_match_amended_preregistration() -> None:
    prereg_path = ROOT / "output/research/autoresearch-loop/EXP_CAD_0027_PREREGISTRATION_V2.json"
    prereg = EXP27._validate_preregistration(prereg_path)
    frozen = prereg["frozen_inputs"]
    source = ROOT / "data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/cad/50476000_A.SLDPRT"
    scan_root = ROOT / "data/bend_corpus/incoming/2026-08-17_20_shaked_new_good_bad_pairs/authenticated_source/scans"
    cadmpeg = Path(
        "/Users/idantal/.local/share/codex-tools/cadmpeg/v0.5.5/"
        "cadmpeg-aarch64-apple-darwin/cadmpeg"
    )
    exp17_summary_path = ROOT / "output/research/autoresearch-loop/exp-cad-0017-native-roundbend-patch-axes-v1/family-50476000_A/SOURCE_ONLY_SUMMARY.json"
    exp24_result_path = ROOT / "output/research/autoresearch-loop/EXP_CAD_0024_RESULT.json"
    exp25_result_path = ROOT / "output/research/autoresearch-loop/EXP_CAD_0025_RESULT.json"
    for path, expected in (
        (source, frozen["source_sldprt_sha256"]),
        (cadmpeg, frozen["cadmpeg_sha256"]),
        (exp17_summary_path, frozen["exp17_summary_sha256"]),
        (exp24_result_path, frozen["exp24_result_sha256"]),
        (exp25_result_path, frozen["exp25_result_sha256"]),
    ):
        assert EXP27.EXP21.sha256_path(path) == expected
    cadir, report = EXP27.EXP25.decode_cadir(source, cadmpeg)
    decoder_provenance = EXP27.validate_decoder_provenance(report)
    assert decoder_provenance["warning_count"] == 6
    assert decoder_provenance["error_severity_count"] == 0
    assert decoder_provenance["unresolved_displaylist_count_from_bound_warning"] == 24
    extraction = EXP27.EXP17.extract_native_roundbend_patch_axes(cadir)
    bundles = EXP27.V1.native_event_bundles(cadir, extraction)
    assert len(bundles) == 4
    for bundle in bundles:
        EXP27.validate_native_event_contract(bundle)
    exp25_result = __import__("json").loads(exp25_result_path.read_text())
    for case_token in EXP27.EXPECTED_CASE_TOKENS:
        points, normals, provenance = EXP27.V1.load_frozen_case(
            case_token,
            scan_root / f"{case_token}.ply",
            ROOT / "output/research/autoresearch-loop/exp-cad-0024-unresolved-native-orientation-landmarks-v1",
            ROOT / "output/research/autoresearch-loop/exp-cad-0025-analytic-brep-carrier-orientation-v1",
            exp25_result,
        )
        transform = np.asarray(provenance["selected_absolute_transform"])
        assert len(points) == len(normals) > 0
        assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-8)
        assert provenance["scan_sha256"] == frozen["scan_sha256"][case_token]


def test_cli_rejects_any_evaluator_manifest_argument() -> None:
    prereg_path = ROOT / "output/research/autoresearch-loop/EXP_CAD_0027_PREREGISTRATION_V2.json"
    with pytest.raises(SystemExit):
        EXP27.main(
            [
                "--preregistration", str(prereg_path),
                "--preflight-only",
                "--evaluator-manifest", "FORBIDDEN.json",
            ]
        )
