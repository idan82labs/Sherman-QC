"""Independent mathematics checks for the bounded adjacent-bend fixture.

No geometry analyzer is imported or called. Closed-form Frenet integration is
checked against independent Gauss-Legendre integration, circle identities,
extrusion planes, seam ownership, hand-computed right-angle chains and SE(3).
"""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.adjacent_bend_constructions import (
    FIRST_HALF_ARC, FULL_ARC, MAX_RAW_POINTS, MIDDLE_GAP_ARC, SUPPORT_NAMES,
    make_adjacent_bend_fixture,
)


def test_raw_sampling_is_constructively_bounded_and_supports_are_disjoint():
    result = make_adjacent_bend_fixture()
    assert result["points"].shape == (145, 3)
    assert result["sampling"]["max_raw_points"] == MAX_RAW_POINTS == 145 < 256
    assert tuple(result["support_indices"]) == SUPPORT_NAMES
    claimed = np.concatenate(list(result["support_indices"].values()))
    assert np.array_equal(claimed, np.arange(145))
    assert len(np.unique(result["points"], axis=0)) == 145
    for name, indices in result["support_indices"].items():
        assert np.array_equal(result["supports"][name], result["points"][indices])
        assert len(indices) == (25 if name.startswith("flange") else 35)
    assert result["sampling"]["posthoc_row_truncation"] is False
    assert result["truth"]["material_thickness"] is None
    assert result["truth"]["evidence_scope"] == "ANALYTIC_TEST_CONSTRUCTION_ONLY_NOT_PHYSICAL_CAPTURE"


@pytest.mark.parametrize("angles, expected", [
    ((90, 90), [(0, 0, 0), (2, 0, 0), (4, 0, 2), (4, 0, 4), (2, 0, 6), (0, 0, 6)]),
    ((90, -90), [(0, 0, 0), (2, 0, 0), (4, 0, 2), (4, 0, 4), (6, 0, 6), (8, 0, 6)]),
])
def test_hand_derived_right_angle_u_and_z_chains(angles, expected):
    result = make_adjacent_bend_fixture(angles_degrees=angles)
    flanges, bends = result["truth"]["flanges"], result["truth"]["bends"]
    chain = [flanges[0]["start"], bends[0]["start"], bends[0]["end"],
             bends[1]["start"], bends[1]["end"], flanges[2]["end"]]
    assert_allclose(chain, expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize("angles", [(60, 90), (-60, -90), (30, -120), (-30, 120), (135, -45)])
@pytest.mark.parametrize("radii", [(2, 2), (2, 3)])
def test_frenet_displacements_match_independent_numerical_tangent_integral(angles, radii):
    result = make_adjacent_bend_fixture(angles_degrees=angles, radii=radii)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    heading = 0.0
    for bend in result["truth"]["bends"]:
        length = bend["arc_length"]
        stations = (nodes+1)*length/2
        theta = heading+bend["signed_curvature"]*stations
        # Independent quadrature of dp/ds=(cos(theta),0,sin(theta)).
        integrated = length/2*np.array([weights@np.cos(theta), 0.0, weights@np.sin(theta)])
        assert_allclose(np.asarray(bend["end"])-bend["start"], integrated, atol=8e-14, rtol=0)
        assert math.isclose(bend["signed_curvature"]*length, bend["signed_angle_radians"], abs_tol=1e-14)
        heading += bend["signed_angle_radians"]
    assert_allclose(result["truth"]["final_tangent"], [math.cos(heading), 0, math.sin(heading)], atol=1e-14)


@pytest.mark.parametrize("angles", [(90, 90), (-90, -90), (60, -90), (-60, 90)])
def test_circle_radius_and_oriented_tangent_constraints_are_independently_satisfied(angles):
    result = make_adjacent_bend_fixture(angles_degrees=angles, radii=(2, 3))
    axis = np.asarray(result["truth"]["extrusion_axis"])
    for bend in result["truth"]["bends"]:
        points = result["supports"][bend["arc_support_name"]]
        radial = points-np.asarray(bend["center"])
        radial -= (radial@axis)[:, None]*axis
        assert_allclose(np.linalg.norm(radial, axis=1), bend["radius"], atol=3e-14, rtol=0)
        for end in ("start", "end"):
            radius = np.asarray(bend[end])-bend["center"]
            tangent = np.asarray(bend["tangent_"+end])
            assert abs(radius@tangent) <= 2e-14
            assert math.isclose(np.linalg.norm(radius), bend["radius"], abs_tol=1e-14)
            assert math.isclose(np.linalg.norm(tangent), 1.0, abs_tol=1e-14)
            expected = math.copysign(1, bend["signed_angle_radians"])*np.cross(-axis, radius)/bend["radius"]
            assert_allclose(tangent, expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("middle_length", [0.25, 2, 6])
def test_both_arcs_share_one_central_flange_and_all_seams_are_c1(middle_length):
    result = make_adjacent_bend_fixture(angles_degrees=(60, -90), radii=(2, 3),
                                        flange_lengths=(2, middle_length, 2))
    truth = result["truth"]
    first, second = truth["bends"]
    central = truth["flanges"][1]
    assert first["adjacent_flange_names"][1] == second["adjacent_flange_names"][0] == "flange_1"
    assert central["length"] == middle_length
    assert_allclose(first["end"], central["start"], atol=1e-14, rtol=0)
    assert_allclose(central["end"], second["start"], atol=1e-14, rtol=0)
    assert_allclose(first["tangent_end"], central["tangent"], atol=1e-14, rtol=0)
    assert_allclose(central["tangent"], second["tangent_start"], atol=1e-14, rtol=0)
    assert len(truth["seams"]) == 4
    for seam in truth["seams"]:
        seam_center = np.asarray(seam["position"])
        axial_rows = seam_center + np.asarray(truth["axial_stations"])[:, None]*truth["extrusion_axis"]
        owner = result["supports"][seam["sampled_owner"]]
        for point in axial_rows:
            assert np.count_nonzero(np.linalg.norm(owner-point, axis=1) < 1e-13) == 1
            assert np.count_nonzero(np.linalg.norm(result["points"]-point, axis=1) < 1e-13) == 1


def test_each_flange_is_an_extruded_plane_with_the_declared_normal():
    result = make_adjacent_bend_fixture(angles_degrees=(55, -110), radii=(2, 3))
    axis = np.asarray(result["truth"]["extrusion_axis"])
    for flange in result["truth"]["flanges"]:
        points = result["supports"][flange["support_name"]]
        normal, tangent = np.asarray(flange["normal"]), np.asarray(flange["tangent"])
        assert_allclose((points-flange["start"])@normal, 0, atol=2e-14, rtol=0)
        assert_allclose(np.cross(tangent, axis), normal, atol=1e-14, rtol=0)
        assert abs(axis@normal) < 1e-14
        longitudinal = (points-flange["start"])@tangent
        assert math.isclose(float(longitudinal.min()), 0, abs_tol=1e-14)
        assert math.isclose(float(longitudinal.max()), flange["length"], abs_tol=1e-14)


@pytest.mark.parametrize("visibility, expected_count", [
    ((FULL_ARC, MIDDLE_GAP_ARC), 140), ((FIRST_HALF_ARC, FULL_ARC), 125),
    (((), FULL_ARC), 110), (((), ()), 75),
])
def test_partial_visibility_changes_observed_arc_rows_not_the_analytic_chain(visibility, expected_count):
    full = make_adjacent_bend_fixture(radii=(2, 3))
    partial = make_adjacent_bend_fixture(radii=(2, 3), arc_sample_indices=visibility)
    assert len(partial["points"]) == expected_count
    assert partial["truth"]["chain_end"] == full["truth"]["chain_end"]
    assert partial["truth"]["total_profile_arclength"] == full["truth"]["total_profile_arclength"]
    for index in range(2):
        name = f"arc_{index}"
        selected = [row for station in visibility[index] for row in range((station-1)*5, station*5)]
        assert np.array_equal(partial["supports"][name], full["supports"][name][selected])
        assert partial["truth"]["bends"][index]["visible_sample_indices"] == list(visibility[index])
        assert partial["truth"]["bends"][index]["radius"] == full["truth"]["bends"][index]["radius"]


@pytest.mark.parametrize("seed", [37, 211, 2026090702])
def test_proper_se3_transforms_positions_and_truth_without_changing_support_membership(seed):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = rng.uniform(-10, 10, size=3)
    arguments = dict(angles_degrees=(-60, 90), radii=(2, 3), flange_lengths=(2, 0.25, 2))
    base = make_adjacent_bend_fixture(**arguments)
    moved = make_adjacent_bend_fixture(**arguments, rotation=rotation, translation=translation)
    assert_allclose(moved["points"], base["points"]@rotation.T+translation, atol=0, rtol=0)
    for name in SUPPORT_NAMES:
        assert np.array_equal(moved["support_indices"][name], base["support_indices"][name])
    for one, two in zip(base["truth"]["bends"], moved["truth"]["bends"]):
        for field in ("start", "end", "center"):
            assert_allclose(two[field], rotation@one[field]+translation, atol=1e-14, rtol=0)
        for field in ("tangent_start", "tangent_end", "axis_direction"):
            assert_allclose(two[field], rotation@one[field], atol=1e-14, rtol=0)
        assert one["radius"] == two["radius"]


def test_explicit_meter_values_scale_all_lengths_not_angles_or_membership():
    millimeters = make_adjacent_bend_fixture(angles_degrees=(60, -90), radii=(2, 3),
                                             flange_lengths=(2, 0.25, 2), axial_extent=2)
    meters = make_adjacent_bend_fixture(angles_degrees=(60, -90), radii=(0.002, 0.003),
                                        flange_lengths=(0.002, 0.00025, 0.002), axial_extent=0.002, unit="m")
    assert_allclose(meters["points"], millimeters["points"]/1000, atol=2e-17, rtol=0)
    for one, two in zip(millimeters["truth"]["bends"], meters["truth"]["bends"]):
        assert one["signed_angle_radians"] == two["signed_angle_radians"]
        assert math.isclose(two["radius"], one["radius"]/1000, abs_tol=1e-18)
        assert math.isclose(two["arc_length"], one["arc_length"]/1000, abs_tol=1e-18)
        assert math.isclose(two["signed_curvature"], 1000*one["signed_curvature"], abs_tol=1e-10)
        assert one["visible_sample_indices"] == two["visible_sample_indices"]


@pytest.mark.parametrize("changes", [
    {"angles_degrees": (0, 90)}, {"angles_degrees": (180, 90)}, {"angles_degrees": (90, -180)},
    {"angles_degrees": (True, 90)}, {"radii": (0, 2)}, {"radii": (2, math.inf)},
    {"flange_lengths": (2, -1, 2)}, {"axial_extent": 0}, {"unit": "cm"},
    {"arc_sample_indices": ((0, 1, 2), FULL_ARC)},
    {"arc_sample_indices": ((1, 2, 8), FULL_ARC)},
    {"arc_sample_indices": ((1, 1, 2), FULL_ARC)},
    {"arc_sample_indices": ((2, 1), FULL_ARC)},
    {"arc_sample_indices": ((True,), FULL_ARC)},
    {"rotation": np.diag([1, 1, -1])}, {"rotation": np.diag([1, 1, 2])},
    {"translation": [0, 0, math.nan]},
])
def test_unsupported_constructions_are_rejected_not_silently_repaired(changes):
    with pytest.raises(ValueError):
        make_adjacent_bend_fixture(**changes)
