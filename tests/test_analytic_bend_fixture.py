"""Source-free analytical oracles and metamorphic checks; no detector imports."""

from dataclasses import replace
import math

import numpy as np
from numpy.testing import assert_allclose as numpy_assert_allclose, assert_array_equal
import pytest

from research.geometry.analytic_bend_fixture import (
    BendSpec,
    MAX_SAMPLES,
    sample_surfaces,
    surface_generator,
)


def assert_allclose(actual, expected, *, atol=1e-12):
    """Keep absolute numerical test bounds explicit, without NumPy's default rtol."""
    numpy_assert_allclose(actual, expected, atol=atol, rtol=0)


def bend(**changes):
    return replace(BendSpec(2, 1, math.pi / 2, (-2, 4), (3, 5)), **changes)


def patch(samples, name, surface="inner", field="points"):
    return getattr(samples, field)[(samples.patches == name) & (samples.surfaces == surface)]


@pytest.mark.parametrize("sign", [-1, 1])
def test_quarter_turn_matches_hand_computed_coordinates_and_normals(sign):
    samples = sample_surfaces(bend(angle_rad=sign * math.pi / 2),
                              axial_fractions=(0,), flange_fractions=(0, 1),
                              bend_fractions=(0, 0.5, 1))
    root2 = math.sqrt(2)
    # Independent coordinate oracle in the x-z section, not calls to a fixture inverse.
    assert_allclose(patch(samples, "start"), [[0, -2, 2], [-3 * sign, -2, 2]], atol=1e-14)
    assert_allclose(patch(samples, "bend"),
                    [[0, -2, 2], [root2 * sign, -2, root2], [2 * sign, -2, 0]], atol=1e-14)
    assert_allclose(patch(samples, "end"), [[2 * sign, -2, 0], [2 * sign, -2, -5]], atol=1e-14)
    assert_allclose(patch(samples, "end", "outer"),
                    [[3 * sign, -2, 0], [3 * sign, -2, -5]], atol=1e-14)
    expected_normals = [[0, 0, 1], [sign / root2, 0, 1 / root2], [sign, 0, 0]]
    assert_allclose(patch(samples, "bend", "outer", "normals"), expected_normals, atol=1e-14)
    assert_allclose(patch(samples, "bend", "inner", "normals"),
                    -np.asarray(expected_normals), atol=1e-14)


@pytest.mark.parametrize("angle", [-math.pi, -2.4, -1e-7, 1e-7, 0.6, math.pi])
def test_implicit_cylinder_plane_thickness_and_finite_span_oracles(angle):
    spec = bend(angle_rad=angle)
    samples = sample_surfaces(spec)
    assert samples.points.shape == (66, 3)
    assert_allclose(np.linalg.norm(samples.normals, axis=1), 1, atol=1e-14)
    assert_array_equal(samples.normals[:, 1], 0)
    assert np.min(samples.points[:, 1]) == -2
    assert np.max(samples.points[:, 1]) == 4
    for surface, radius, normal_sign in (("inner", 2, -1), ("outer", 3, 1)):
        arc = patch(samples, "bend", surface)
        # Implicit x^2+z^2=r^2 and exact flange plane equations are separate oracles.
        assert_allclose(arc[:, 0] ** 2 + arc[:, 2] ** 2, radius ** 2, atol=1e-12)
        assert_allclose(patch(samples, "start", surface)[:, 2], radius, atol=1e-12)
        end = patch(samples, "end", surface)
        assert_allclose(end[:, 0] * math.sin(angle) + end[:, 2] * math.cos(angle),
                        radius, atol=1e-12)
        assert_allclose(patch(samples, "bend", surface, "normals")[:, [0, 2]],
                        normal_sign * arc[:, [0, 2]] / radius, atol=1e-14)
    for name in ("start", "bend", "end"):
        assert_allclose(patch(samples, name, "outer") - patch(samples, name, "inner"),
                        patch(samples, name, "outer", "normals"), atol=1e-14)
        assert_allclose(patch(samples, name, "inner", "normals"),
                        -patch(samples, name, "outer", "normals"), atol=1e-14)


@pytest.mark.parametrize("angle", [-2.1, -0.73, 0.73, 2.1])
def test_seams_have_matching_points_normals_and_arclength_tangents(angle):
    h = 1e-6
    samples = sample_surfaces(bend(angle_rad=angle), axial_fractions=(0.5,),
                              bend_fractions=(0, h, 1 - h, 1), flange_fractions=(0, h))
    for side in ("inner", "outer"):
        start, arc, end = (patch(samples, name, side) for name in ("start", "bend", "end"))
        assert_allclose(start[0], arc[0], atol=1e-14)
        assert_allclose(end[0], arc[-1], atol=1e-14)
        for name, index in (("start", 0), ("end", -1)):
            assert_allclose(patch(samples, name, side, "normals")[0],
                            patch(samples, "bend", side, "normals")[index], atol=1e-14)
        # Directed secants converge to the shared tangent; no generator derivatives used.
        before, first = start[0] - start[1], arc[1] - arc[0]
        after, last = end[1] - end[0], arc[-1] - arc[-2]
        for one, two in ((before, first), (after, last)):
            assert_allclose(one / np.linalg.norm(one), two / np.linalg.norm(two), atol=2e-6)


@pytest.mark.parametrize("angle", [-2.5, -math.pi / 6, math.pi / 6, 2.5])
def test_signed_normal_turn_is_not_flange_opening(angle):
    samples = sample_surfaces(bend(angle_rad=angle), axial_fractions=(0,),
                              flange_fractions=(0, 1), surfaces=("outer",))
    n0 = patch(samples, "start", "outer", "normals")[0]
    n1 = patch(samples, "end", "outer", "normals")[0]
    signed_turn = math.atan2(np.dot([0, 1, 0], np.cross(n0, n1)), n0 @ n1)
    start, end = (patch(samples, name, "outer") for name in ("start", "end"))
    rays = [v[1] - v[0] for v in (start, end)]
    opening = math.acos(np.clip(rays[0] @ rays[1] /
                                (np.linalg.norm(rays[0]) * np.linalg.norm(rays[1])), -1, 1))
    assert signed_turn == pytest.approx(angle, abs=1e-14)
    assert opening == pytest.approx(math.pi - abs(angle), abs=1e-14)


def test_proper_rigid_motion_preserves_geometry_normals_finite_axes_and_metadata():
    spec = bend(angle_rad=-1.2)
    # Rational orthogonal matrix from quaternion (1,2,3,4)/sqrt(30); not an API helper.
    q = np.array([[-2 / 3, 2 / 15, 11 / 15], [2 / 3, -1 / 3, 2 / 3],
                  [1 / 3, 14 / 15, 2 / 15]])
    d = np.array([11, -7, 3])
    original = sample_surfaces(spec)
    moved_spec = spec.rigidly_transformed(q, d)
    moved = sample_surfaces(moved_spec)
    assert_allclose(moved.points, original.points @ q.T + d, atol=1e-12)
    assert_allclose(moved.normals, original.normals @ q.T, atol=1e-12)
    assert_allclose(moved_spec.axis_segment(), spec.axis_segment() @ q.T + d, atol=1e-12)
    for field in ("patches", "surfaces", "component_ids", "axial_fractions", "patch_fractions"):
        assert_array_equal(getattr(moved, field), getattr(original, field))
    arc = patch(moved, "bend")
    assert_allclose(np.linalg.norm(np.cross(arc - d, moved_spec.axis), axis=1), 2, atol=1e-12)
    recovered = moved_spec.rigidly_transformed(q.T, -q.T @ d)
    assert_allclose(sample_surfaces(recovered).points, original.points, atol=1e-12)


def test_axis_reversal_is_a_signed_angle_gauge_change_not_a_new_surface():
    spec = bend(angle_rad=0.9)
    reverse = replace(spec, axis=(0, -1, 0), angle_rad=-spec.angle_rad,
                      span=(-spec.span[1], -spec.span[0]))
    original = sample_surfaces(spec)
    same = sample_surfaces(reverse, axial_fractions=(1, 0.5, 0))
    assert_allclose(same.points, original.points, atol=1e-14)
    assert_allclose(same.normals, original.normals, atol=1e-14)
    assert_array_equal(reverse.axis_segment(), spec.axis_segment()[::-1])


def test_unit_conversion_scales_every_length_but_not_normals_angles_or_identity():
    spec = bend(origin=(11, -3, 7), component_id="tab-A")
    converted = spec.in_units("m")
    original, metres = sample_surfaces(spec), sample_surfaces(converted)
    assert_allclose(metres.points, original.points / 1000, atol=1e-16)
    assert_allclose(converted.axis_segment(), spec.axis_segment() / 1000, atol=1e-16)
    assert_allclose(surface_generator(converted), surface_generator(spec) / 1000, atol=1e-16)
    assert_array_equal(metres.normals, original.normals)
    assert_array_equal(metres.component_ids, original.component_ids)
    assert converted.angle_rad == spec.angle_rad
    assert converted.radius("inner") == pytest.approx(0.002)
    assert converted.radius("outer") == pytest.approx(0.003)
    assert_allclose(converted.flange_lengths, [0.003, 0.005], atol=1e-16)
    assert metres.unit == "m"
    assert_allclose(sample_surfaces(converted.in_units("mm")).points, original.points, atol=1e-12)


def test_surface_ruling_is_finite_and_distinct_from_the_centre_axis():
    spec = bend()
    assert_array_equal(spec.axis_segment(), [[0, -2, 0], [0, 4, 0]])
    expected = [[3 / math.sqrt(2), u, 3 / math.sqrt(2)] for u in (-2, 4)]
    assert_allclose(surface_generator(spec), expected, atol=1e-14)
    assert_allclose(np.linalg.norm(surface_generator(spec) - spec.axis_segment(), axis=1), 3)
    assert np.linalg.norm(surface_generator(spec)[1] - surface_generator(spec)[0]) == 6


def test_virtual_flange_intersection_has_a_radius_dependent_offset_from_the_axis():
    spec = bend(angle_rad=math.pi / 3)
    samples = sample_surfaces(spec)
    for side, radius in (("inner", 2), ("outer", 3)):
        # Solve z=r and sin(theta)*x+cos(theta)*z=r independently in the cross-section.
        virtual_corner = np.array([radius / math.sqrt(3), 0, radius])
        for name in ("start", "end"):
            point = patch(samples, name, side)[0]
            normal = patch(samples, name, side, "normals")[0]
            assert (virtual_corner - point) @ normal == pytest.approx(0, abs=1e-14)
        assert np.linalg.norm(virtual_corner) == pytest.approx(2 * radius / math.sqrt(3))
        assert virtual_corner[0] > np.max(patch(samples, "start", side)[:, 0])
    assert_array_equal(spec.axis_segment()[:, [0, 2]], np.zeros((2, 2)))


def test_hidden_transition_allows_same_visible_flange_faces_at_different_radii():
    observations, specs = [], []
    for radius in (2.0, 4.0):
        # Both inner planes are x=0/z=0; outer planes x=1/z=1. Only x/z=-8..-10 visible.
        spec = bend(inner_radius=radius, origin=(-radius, 0, -radius),
                    flange_lengths=(10 - radius, 10 - radius))
        specs.append(spec)
        observations.append(sample_surfaces(spec, patches=("start", "end"),
                                            flange_fractions=((8 - radius) / (10 - radius), 1)))
    assert_allclose(observations[0].points, observations[1].points, atol=1e-14)
    assert_allclose(observations[0].normals, observations[1].normals, atol=1e-14)
    assert specs[0].inner_radius != specs[1].inner_radius
    assert not np.allclose(specs[0].axis_segment(), specs[1].axis_segment())
    assert np.all(observations[0].patches != "bend")
    # This is an equality of observation channels, not proof by metadata equality.


def test_one_visible_flange_does_not_identify_turn_angle():
    shallow = sample_surfaces(bend(angle_rad=math.pi / 6), patches=("start",))
    deep = sample_surfaces(bend(angle_rad=3 * math.pi / 4), patches=("start",))
    assert_array_equal(shallow.points, deep.points)
    assert_array_equal(shallow.normals, deep.normals)


def test_broad_radius_small_visible_arc_can_be_flattened_within_a_known_error_bound():
    radius, delta = 1000.0, 0.002
    samples = sample_surfaces(bend(inner_radius=radius, angle_rad=1), patches=("bend",),
                              surfaces=("inner",), bend_fractions=(0.5 - delta / 2, 0.5, 0.5 + delta / 2))
    grid = samples.points.reshape(3, 3, 3)
    flat = grid.copy()
    flat[:, 1] = (grid[:, 0] + grid[:, 2]) / 2
    # Stable, independent chord/sagitta formula; local arc is ~2 mm wide, ~0.0005 mm deep.
    sagitta = 2 * radius * math.sin(delta / 4) ** 2
    assert np.linalg.norm(grid[0, 2] - grid[0, 0]) == pytest.approx(2 * radius * math.sin(delta / 2))
    assert np.max(np.linalg.norm(flat - grid, axis=2)) == pytest.approx(sagitta, abs=2e-13)
    plane_normal = np.cross(flat[0, 2] - flat[0, 0], flat[2, 0] - flat[0, 0])
    plane_normal /= np.linalg.norm(plane_normal)
    assert_allclose((flat - flat[0, 0]) @ plane_normal, 0, atol=1e-12)
    assert sagitta < 0.001  # A stipulated bound, not a scanner's measured accuracy.


@pytest.mark.parametrize("height_ratio", [0.5, 1.0, 2.0])
def test_free_plane_position_error_boundary_has_an_explicit_radius_oracle(height_ratio):
    chord, epsilon = 2.0, 0.001
    height = 2 * epsilon * height_ratio
    # Solve the circle chord/sagitta equations independently of the surface generator.
    radius = chord ** 2 / (8 * height) + height / 2
    angle = 2 * math.asin(chord / (2 * radius))
    samples = sample_surfaces(bend(inner_radius=radius, angle_rad=angle),
                              patches=("bend",), surfaces=("inner",), bend_fractions=(0, 0.5, 1))
    grid = samples.points.reshape(3, 3, 3)
    chord_mid = (grid[0, 0] + grid[0, 2]) / 2
    direction = grid[0, 1] - chord_mid
    measured_height = np.linalg.norm(direction)
    direction /= measured_height
    # The plane halfway between chord and apex needs only h/2 positional error.
    planar = grid - (((grid - chord_mid) @ direction - height / 2)[..., None] * direction)
    assert_allclose((planar - chord_mid) @ direction, height / 2, atol=1e-12)
    residual = np.max(np.linalg.norm(grid - planar, axis=2))
    assert measured_height == pytest.approx(height, abs=1e-12)
    assert residual == pytest.approx(epsilon * height_ratio, abs=1e-12)
    critical_radius = 250.001  # L=2 mm and h=2 epsilon=0.002 mm.
    assert (radius >= critical_radius) == (height_ratio <= 1)
    if height_ratio <= 1:
        assert residual <= epsilon + 1e-12
    else:
        assert residual > epsilon
    # If BOTH hypotheses allow error, a shared observation needs only h/4 on each.
    shared_observation = (grid + planar) / 2
    for hypothesis in (grid, planar):
        assert np.max(np.linalg.norm(shared_observation - hypothesis, axis=2)) <= epsilon + 1e-12


def test_disconnected_collinear_components_keep_separate_finite_support_and_identity():
    left = bend(span=(-8, -3), component_id="left-tab")
    right = bend(span=(3, 8), component_id="right-tab")
    pieces = [sample_surfaces(spec) for spec in (left, right)]
    assert np.max(pieces[0].points[:, 1]) == -3
    assert np.min(pieces[1].points[:, 1]) == 3
    assert_array_equal(left.axis_segment()[:, [0, 2]], right.axis_segment()[:, [0, 2]])
    ids = np.concatenate([piece.component_ids for piece in pieces])
    assert set(ids) == {"left-tab", "right-tab"}
    combined_points = np.concatenate([piece.points for piece in pieces])
    assert not np.any(np.abs(combined_points[:, 1]) < 3)
    # One contracted interval would invent support in the 6 mm gap.
    invented = surface_generator(bend(span=(-8, 8))).mean(axis=0)
    assert invented[1] == 0
    for spec in (left, right):
        transformed = spec.rigidly_transformed(np.eye(3), (4, 7, -2)).in_units("m")
        assert set(sample_surfaces(transformed).component_ids) == {spec.component_id}


def test_resampling_permutation_repetition_and_partial_arc_are_explicit():
    spec = bend()
    small = sample_surfaces(spec, patches=("bend",), surfaces=("inner",),
                            axial_fractions=(0.5,), bend_fractions=(0, 0.5, 1))
    fine = sample_surfaces(spec, patches=("bend",), surfaces=("inner",),
                           axial_fractions=(0.5,), bend_fractions=(0, 0.25, 0.5, 0.75, 1))
    assert_array_equal(small.points, fine.points[::2])
    partial = sample_surfaces(spec, patches=("bend",), surfaces=("inner",),
                              axial_fractions=(0.5,), bend_fractions=(0.75, 0.25, 0.25))
    assert_array_equal(partial.points, fine.points[[3, 1, 1]])
    assert_array_equal(partial.patch_fractions, [0.75, 0.25, 0.25])
    assert_array_equal(sample_surfaces(spec).points, sample_surfaces(spec).points)


@pytest.mark.parametrize("changes", [
    {"inner_radius": 0}, {"inner_radius": -1}, {"inner_radius": math.inf},
    {"thickness": 0}, {"thickness": -1}, {"thickness": math.nan},
    {"inner_radius": 1e308, "thickness": 1e308},
    {"angle_rad": 0}, {"angle_rad": -0.0}, {"angle_rad": math.pi + 1e-6},
    {"angle_rad": -math.pi - 1e-6}, {"angle_rad": math.nan},
    {"span": (2, 2)}, {"span": (3, -1)}, {"span": (0, math.inf)},
    {"span": (-1e308, 1e308)}, {"flange_lengths": (0, 1)}, {"flange_lengths": (1, -1)},
    {"origin": (1, 2)}, {"origin": (0, math.nan, 0)}, {"axis": (0, 2, 0)},
    {"axis": (0, 0, 1)}, {"start_radial": (0, 0, 0)},
    {"unit": "inch"}, {"component_id": " "},
])
def test_invalid_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        bend(**changes)


@pytest.mark.parametrize("rotation", [np.diag([-1, 1, 1]), np.diag([2, 2, 2]),
                                      np.diag([1, 1, 2]), [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]],
                                      np.eye(2), np.full((3, 3), math.nan)])
def test_reflection_scale_shear_and_invalid_matrices_are_not_rigid_motions(rotation):
    with pytest.raises(ValueError, match="proper orthogonal"):
        bend().rigidly_transformed(rotation, (0, 0, 0))


@pytest.mark.parametrize("kwargs", [{"axial_fractions": ()}, {"bend_fractions": (-0.1, 1)},
                                     {"flange_fractions": (0, 1.1)}, {"bend_fractions": (math.nan,)},
                                     {"axial_fractions": [[0, 1]]}, {"patches": ()},
                                     {"patches": ("bend", "bend")}, {"patches": ("cap",)},
                                     {"surfaces": ("mid",)}, {"surfaces": ()}])
def test_invalid_sampling_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sample_surfaces(bend(), **kwargs)


def test_sample_budget_is_checked_before_grid_allocation():
    with pytest.raises(ValueError, match="limited to"):
        sample_surfaces(bend(), axial_fractions=np.linspace(0, 1, MAX_SAMPLES))
    exact_limit = sample_surfaces(bend(), axial_fractions=np.linspace(0, 1, MAX_SAMPLES),
                                  bend_fractions=(0.5,), patches=("bend",), surfaces=("outer",))
    assert len(exact_limit.points) == MAX_SAMPLES


def test_invalid_units_face_and_transform_translation_are_rejected():
    with pytest.raises(ValueError):
        bend().in_units("inch")
    with pytest.raises(ValueError):
        bend().radius("neutral")
    with pytest.raises(ValueError):
        bend().rigidly_transformed(np.eye(3), (0, math.inf, 0))
    with pytest.raises(ValueError):
        surface_generator(bend(), bend_fraction=1.01)
