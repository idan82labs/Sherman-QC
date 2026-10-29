"""Independent analytic two-bend extrusion for source-only fixture science.

This module imports no geometry engine. It integrates a planar Frenet frame
through straight/constant-curvature/straight/constant-curvature/straight pieces,
then extrudes the profile along +Y. Positive profile rotation is about -Y.
Every seam row belongs only to its planar flange; arc samples are strictly
interior. Fixed station counts bound the FULL construction at 145 raw rows.
Visibility omissions are declared inputs, not a post-result point-budget repair.

Named supports, centers and event counts are test-only construction truth. Pass
only ``result['points']`` to a position-only analyzer. This is not scanner data,
calibration, a physical part observation, or a verified non-self-intersecting
material solid; it has no thickness model.
"""
from __future__ import annotations

import math

import numpy as np


FULL_ARC = (1, 2, 3, 4, 5, 6, 7)
MIDDLE_GAP_ARC = (1, 2, 3, 5, 6, 7)
FIRST_HALF_ARC = (1, 2, 3)
SUPPORT_NAMES = ("flange_0", "arc_0", "flange_1", "arc_1", "flange_2")
FLANGE_STATION_COUNT = 5
AXIAL_STATION_COUNT = 5
MAX_RAW_POINTS = 3*FLANGE_STATION_COUNT*AXIAL_STATION_COUNT + 2*len(FULL_ARC)*AXIAL_STATION_COUNT


def _real(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as a finite real number") from error
    if not math.isfinite(value) or positive and value <= 0:
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return value


def _numbers(values, count, name, *, positive=False):
    if (not isinstance(values, (tuple, list, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1 or len(values) != count):
        raise ValueError(f"{name} must contain exactly {count} numbers")
    return tuple(_real(value, name, positive=positive) for value in values)


def _visibility(values):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError("arc_sample_indices must contain exactly two index sequences")
    result = []
    for indices in values:
        if not isinstance(indices, (tuple, list)) or len(indices) > len(FULL_ARC):
            raise ValueError("each arc has at most the seven declared interior stations")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               or not 1 <= value <= 7 for value in indices):
            raise ValueError("arc station indices must be integers from 1 through 7")
        canonical = tuple(int(value) for value in indices)
        if canonical != tuple(sorted(set(canonical))):
            raise ValueError("arc station indices must be strictly increasing and unique")
        result.append(canonical)
    return tuple(result)


def _pose(rotation, translation):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
        raise ValueError("pose must contain finite 3x3 rotation and length-three translation")
    if (not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-10, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10, rel_tol=0)):
        raise ValueError("rotation must be a proper orthonormal rotation, not a reflection or scale")
    return rotation, translation


def _frame(heading):
    # In the (x,z) profile plane: dT/dtheta=N, dN/dtheta=-T.
    return (np.array([math.cos(heading), math.sin(heading)]),
            np.array([-math.sin(heading), math.cos(heading)]))


def _arc_profile(start, heading, radius, angle, fractions):
    """Integrate dp/ds=T, dT/ds=kappa*N with kappa=sign(angle)/R.

    For u=abs(angle)*fraction, integral displacement is
    R*sin(u)*T0 + sign(angle)*R*(1-cos(u))*N0. The half-angle form avoids
    subtractive cancellation in 1-cos(u). Fractions include endpoints only when
    constructing analytic truth, never when sampling an observed arc support.
    """
    tangent, normal = _frame(heading)
    u = np.asarray(fractions, dtype=float)*abs(angle)
    sign = math.copysign(1.0, angle)
    along = radius*np.sin(u)
    across = sign*radius*2*np.sin(u/2)**2
    return np.asarray(start)[None, :] + along[:, None]*tangent + across[:, None]*normal


def _extrude(profile, axial_stations):
    return np.array([[point[0], axial, point[1]] for point in profile for axial in axial_stations],
                    dtype=float).reshape((-1, 3))


def make_adjacent_bend_fixture(
    *, angles_degrees=(90.0, 90.0), radii=(2.0, 2.0), flange_lengths=(2.0, 2.0, 2.0),
    axial_extent=2.0, arc_sample_indices=(FULL_ARC, FULL_ARC), unit="mm",
    rotation=None, translation=None,
):
    """Return a bounded two-bend profile cloud plus separate test-only truth.

    Length arguments and translation are expressed in the explicitly supplied
    unit; changing the unit string alone DOES NOT convert numeric lengths.
    Same-sign turns describe a U-like profile; opposite signs a Z/S-like profile.
    Nonzero turns satisfy |angle|<180 degrees. Arbitrary profiles may intersect;
    no collision/material identity assertion is implied. Coincident sampled rows
    are rejected, never deduplicated, trimmed or resampled to fit the row budget.
    """
    if not isinstance(unit, str) or unit not in ("m", "mm"):
        raise ValueError("unit must explicitly be 'm' or 'mm'")
    degrees = _numbers(angles_degrees, 2, "angles_degrees")
    if any(angle == 0 or abs(angle) >= 180 for angle in degrees):
        raise ValueError("each signed angle must be nonzero and have magnitude below 180 degrees")
    angles = tuple(math.radians(angle) for angle in degrees)
    radii = _numbers(radii, 2, "radii", positive=True)
    lengths = _numbers(flange_lengths, 3, "flange_lengths", positive=True)
    extent = _real(axial_extent, "axial_extent", positive=True)
    visibility = _visibility(arc_sample_indices)
    rotation, translation = _pose(rotation, translation)
    axial = np.linspace(-extent/2, extent/2, AXIAL_STATION_COUNT)
    flange_fractions = np.linspace(0.0, 1.0, FLANGE_STATION_COUNT)
    profile_supports, flange_truth, bend_truth, seams = {}, [], [], []
    position, heading, chain_length = np.zeros(2), 0.0, 0.0

    def point_world(profile_point):
        value = np.array([profile_point[0], 0.0, profile_point[1]])
        return (rotation@value + translation).tolist()

    def vector_world(profile_vector):
        return (rotation@np.array([profile_vector[0], 0.0, profile_vector[1]])).tolist()

    axis = (rotation@np.array([0.0, 1.0, 0.0])).tolist()
    for index in range(3):
        tangent, normal = _frame(heading)
        end = position + lengths[index]*tangent
        profile_supports[f"flange_{index}"] = position + flange_fractions[:, None]*lengths[index]*tangent
        flange_truth.append(dict(
            support_name=f"flange_{index}", start=point_world(position), end=point_world(end),
            tangent=vector_world(tangent), normal=vector_world(normal), length=lengths[index],
            profile_arclength_interval=[chain_length, chain_length+lengths[index]],
        ))
        position, chain_length = end, chain_length+lengths[index]
        if index == 2:
            continue
        angle, radius = angles[index], radii[index]
        sign = math.copysign(1.0, angle)
        center = position + sign*radius*normal
        fractions = np.asarray(visibility[index], dtype=float)/8.0
        profile_supports[f"arc_{index}"] = _arc_profile(position, heading, radius, angle, fractions)
        arc_end = _arc_profile(position, heading, radius, angle, [1.0])[0]
        tangent_end, _ = _frame(heading+angle)
        length = radius*abs(angle)
        bend_truth.append(dict(
            arc_support_name=f"arc_{index}", adjacent_flange_names=[f"flange_{index}", f"flange_{index+1}"],
            radius=radius, signed_angle_radians=angle, signed_curvature=sign/radius,
            start=point_world(position), end=point_world(arc_end), center=point_world(center),
            tangent_start=vector_world(tangent), tangent_end=vector_world(tangent_end),
            axis_direction=axis, profile_rotation_axis=(-np.asarray(axis)).tolist(),
            arc_length=length, profile_arclength_interval=[chain_length, chain_length+length],
            visible_sample_indices=list(visibility[index]), visible_sample_fractions=fractions.tolist(),
        ))
        seams.extend([
            dict(name=f"flange_{index}_to_arc_{index}", position=point_world(position),
                 tangent=vector_world(tangent), sampled_owner=f"flange_{index}"),
            dict(name=f"arc_{index}_to_flange_{index+1}", position=point_world(arc_end),
                 tangent=vector_world(tangent_end), sampled_owner=f"flange_{index+1}"),
        ])
        position, heading, chain_length = arc_end, heading+angle, chain_length+length

    arrays, support_indices, offset = [], {}, 0
    for name in SUPPORT_NAMES:
        sampled = _extrude(profile_supports[name], axial)
        arrays.append(sampled)
        support_indices[name] = np.arange(offset, offset+len(sampled), dtype=int)
        offset += len(sampled)
    # Bound is constructive: 75 flange rows plus at most 70 arc rows. No
    # posthoc spatial subsampling, row clipping, padding or seam deduplication.
    assert offset <= MAX_RAW_POINTS == 145
    raw = np.concatenate(arrays)
    points = raw@rotation.T+translation
    if not np.isfinite(points).all():
        raise ValueError("constructed positions exceed the supported finite numeric range")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("construction has coincident sampled rows; no deduplication or resampling is permitted")
    supports = {name: points[indices].copy() for name, indices in support_indices.items()}
    truth = dict(
        evidence_scope="ANALYTIC_TEST_CONSTRUCTION_ONLY_NOT_PHYSICAL_CAPTURE",
        flanges=flange_truth, bends=bend_truth, seams=seams, shared_flange_name="flange_1",
        extrusion_axis=axis, axial_stations=axial.tolist(), axial_extent=extent,
        chain_start=flange_truth[0]["start"], chain_end=flange_truth[-1]["end"],
        total_profile_arclength=chain_length, final_tangent=flange_truth[-1]["tangent"],
        self_intersection_is_checked=False, material_thickness=None,
        forming_or_conformance_state_is_established=False,
    )
    return dict(
        kind="ANALYTIC_ADJACENT_BEND_CONSTRUCTION_V1", unit=unit, points=points,
        supports=supports, support_indices=support_indices, truth=truth,
        sampling=dict(max_raw_points=MAX_RAW_POINTS, raw_point_count=len(points),
                      flange_station_count=FLANGE_STATION_COUNT, axial_station_count=AXIAL_STATION_COUNT,
                      arc_grid_denominator=8, seam_rows_owned_by_flange_only=True,
                      full_profile_exists_independently_of_visible_arc_samples=True,
                      posthoc_row_truncation=False, inferred_support_labels_are_supplied=False),
    )
