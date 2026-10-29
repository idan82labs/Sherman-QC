"""Fixed exact-arithmetic audit of sharp-corner versus cylindrical-axis semantics.

This is a mathematical simulation, not a native CAD, metrology, root-role or
manufacturing classifier. It imports only stdlib, reads no files and invokes no
kernel. Every fixture is rational and invented; no float tolerance can rescue an
identity failure. All four signed offsets remain alternatives.

For two unit plane normals n1,n2 with n1.n2=d and planes through the origin,
solve n_i.c=s_i*R in the cross-section by Cramer's rule. Independently check
|c|^2=2*R^2/(1+s1*s2*d). Thus the cylinder axis is generally displaced from
the planes' sharp intersection line (at 90 degrees, |c|=sqrt(2)*R).
The contact generator is c-s_i*R*n_i+t*a, where a is their unit cross-axis.
Finite t intervals are supplied separately: infinite tangency does not infer them.

Geometric basis: OCCT's cylinder parameterization and tangent-generator UIso:
https://occt3d.com/dev/doc/refman/html/class_geom___cylindrical_surface.html
Topology versus geometry and trimmed faces:
https://occt3d.com/dev/doc/overview/html/occt_user_guides__modeling_data.html
Manufacturing-feature interpretation is separately nonunique; see Tseng & Joshi
(1994), doi:10.1016/0010-4485(94)90018-3. Our same-descriptor/different-process
control demonstrates information loss, not the manufacturability of a fixture.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
from itertools import product
import json
import time


HALF_ANGLES = (F(1, 1_000_000), F(1, 10), F(1, 4), F(1, 2), F(3, 4),
               F(1), F(4, 3), F(2), F(10), F(1_000_000))
SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
RADII = (F(1, 4), F(1), F(7, 2), F(11))
SCALES = (F(1, 1000), F(1), F(5, 2), F(1000))
FINITE_INTERVALS = ((F(-5, 2), F(7, 3)), (F(10), F(11)))
EXPECTED_CASES = 5120
MAX_CASES = 8192
MAX_NUMERATOR_OR_DENOMINATOR_BITS = 512


def dot(first, second):
    return sum((a * b for a, b in zip(first, second)), F(0))


def add(first, second):
    return tuple(a + b for a, b in zip(first, second))


def sub(first, second):
    return tuple(a - b for a, b in zip(first, second))


def mul(scale, vector):
    return tuple(scale * value for value in vector)


def cross(first, second):
    return (first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0])


def rotate(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def determinant(matrix):
    return dot(matrix[0], cross(matrix[1], matrix[2]))


def stereo_rotation(parameters):
    """Rational unit quaternion via stereographic projection, then proper Q."""
    square = dot(parameters, parameters)
    w = (1 - square) / (1 + square)
    x, y, z = mul(2 / (1 + square), parameters)
    return ((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))


FRAMES = (
    ("identity", ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))), (F(0),) * 3),
    ("pythagorean_z", ((F(3, 5), F(-4, 5), F(0)), (F(4, 5), F(3, 5), F(0)), (F(0), F(0), F(1))),
     (F(7, 3), F(-5, 2), F(11, 7))),
    ("cyclic_large_translation", ((F(0), F(0), F(1)), (F(1), F(0), F(0)), (F(0), F(1), F(0))),
     (F(10**12), F(-10**12), F(1, 3))),
    ("rational_oblique", stereo_rotation((F(1, 2), F(1, 3), F(1, 4))), (F(-13, 5), F(17, 3), F(19, 11))),
)


def plane_normals(half_angle):
    """Both normals are rational, non-axis-aligned; determinant is positive."""
    if not isinstance(half_angle, F) or half_angle <= 0:
        raise ValueError("strictly positive exact rational half-angle required")
    cosine = (1 - half_angle*half_angle) / (1 + half_angle*half_angle)
    sine = 2 * half_angle / (1 + half_angle*half_angle)
    first = (F(3, 5), F(4, 5), F(0))
    second = (first[0] * cosine - first[1] * sine,
              first[1] * cosine + first[0] * sine, F(0))
    return first, second


def solve_axis_displacement(first, second, radius, signs):
    """Independent 2x2 Cramer's solve, not the norm formula under test."""
    if not isinstance(radius, F) or radius <= 0 or signs not in SIGNS:
        raise ValueError("positive exact radius and one of four signed alternatives required")
    det = first[0] * second[1] - first[1] * second[0]
    if det == 0:
        raise ValueError("parallel planes have no unique cross-sectional intersection")
    rhs1, rhs2 = signs[0] * radius, signs[1] * radius
    return ((rhs1 * second[1] - first[1] * rhs2) / det,
            (first[0] * rhs2 - rhs1 * second[0]) / det, F(0))


def normal_system_condition_squared(first, second):
    """Exact kappa_2(A)^2=kappa_2(A A^T), not a floating condition estimate."""
    correlation = abs(dot(first, second))
    if correlation >= 1:
        raise ValueError("nonparallel unit normals required")
    return (1 + correlation) / (1 - correlation)


def _bounded(values):
    return all(isinstance(value, (int, F)) and not isinstance(value, bool)
               and abs(F(value).numerator).bit_length() <= MAX_NUMERATOR_OR_DENOMINATOR_BITS
               and F(value).denominator.bit_length() <= MAX_NUMERATOR_OR_DENOMINATOR_BITS
               for value in values)


def _check(condition, name, checks):
    checks[name] += 1
    if not condition:
        raise ArithmeticError("exact identity failed: " + name)


def verify_case(half_angle, signs, radius, frame, scale, interval, checks):
    first, second = plane_normals(half_angle)
    center = solve_axis_displacement(first, second, radius, signs)
    _, matrix, translation = frame
    axis = (F(0), F(0), F(1))
    correlation = dot(first, second)
    determinant2 = cross(first, second)[2]
    _check(dot(first, first) == dot(second, second) == 1, "unit_input_normals", checks)
    _check(dot(first, center) == signs[0] * radius and dot(second, center) == signs[1] * radius,
           "independent_cramer_plane_equations", checks)
    _check(dot(center, axis) == 0 and cross(first, second) == mul(determinant2, axis),
           "sharp_intersection_axis_direction", checks)
    _check(determinant2*determinant2 == 1 - correlation*correlation,
           "normal_gram_determinant", checks)
    formula = 2 * radius*radius / (1 + signs[0]*signs[1]*correlation)
    _check(dot(center, center) == formula, "closed_form_offset_squared", checks)
    if half_angle == 1:
        _check(dot(center, center) == 2 * radius*radius, "right_angle_sqrt2_offset_squared", checks)

    normals = (rotate(matrix, first), rotate(matrix, second))
    transformed_axis = rotate(matrix, axis)
    displacement = rotate(matrix, mul(scale, center))
    transformed_center = add(translation, displacement)
    transformed_radius = scale * radius
    _check(all(dot(value, value) == 1 for value in (*normals, transformed_axis)),
           "transformed_unit_normals_and_axis", checks)
    _check(all(dot(normal, transformed_axis) == 0 for normal in normals)
           and cross(*normals) == mul(determinant2, transformed_axis),
           "proper_transformed_cross_axis", checks)
    _check(dot(displacement, displacement) == scale*scale*formula,
           "rigid_similarity_offset_identity", checks)
    _check(sub(transformed_center, translation) == displacement,
           "translation_cancellation_exact", checks)
    low, high = (scale * value for value in interval)
    _check(low < high, "finite_interval_separately_declared", checks)
    parameters = (low, (low + high) / 2, high)
    for normal, sign in zip(normals, signs):
        plane_constant = dot(normal, translation)
        _check(dot(normal, transformed_center) - plane_constant == sign * transformed_radius,
               "transformed_signed_plane_offset", checks)
        base_contact = sub(transformed_center, mul(sign * transformed_radius, normal))
        for parameter in parameters:
            contact = add(base_contact, mul(parameter, transformed_axis))
            center_at_parameter = add(transformed_center, mul(parameter, transformed_axis))
            radial = sub(contact, center_at_parameter)
            _check(dot(normal, contact) == plane_constant, "contact_on_analytic_plane", checks)
            _check(dot(radial, radial) == transformed_radius*transformed_radius
                   and dot(radial, transformed_axis) == 0, "contact_on_cylinder_at_finite_parameter", checks)
            _check(dot(sub(contact, transformed_center), transformed_axis) == parameter,
                   "contact_axial_parameter_preserved", checks)
        outside = high + (high - low)
        contact_outside = add(base_contact, mul(outside, transformed_axis))
        outside_radial = sub(contact_outside, add(transformed_center, mul(outside, transformed_axis)))
        _check(outside > high and dot(normal, contact_outside) == plane_constant
               and dot(outside_radial, outside_radial) == transformed_radius*transformed_radius,
               "infinite_tangency_does_not_certify_finite_interval", checks)
    _check(_bounded((*first, *second, *center, *displacement, *transformed_center,
                     transformed_radius, formula, low, high)), "fixed_exact_arithmetic_bit_budget", checks)


def run_checks():
    """Run the frozen 5,120-case matrix once and return a JSON-safe aggregate."""
    count = len(HALF_ANGLES) * len(SIGNS) * len(RADII) * len(FRAMES) * len(SCALES) * len(FINITE_INTERVALS)
    if count != EXPECTED_CASES or count > MAX_CASES:
        raise ValueError("frozen case matrix changed or exceeds cap")
    started = time.perf_counter()
    checks, strata, sign_counts, frame_counts, scale_counts, interval_counts = (Counter() for _ in range(6))
    for _, matrix, _ in FRAMES:
        _check(all(dot(matrix[i], matrix[j]) == int(i == j) for i in range(3) for j in range(3)),
               "frame_exact_orthogonality", checks)
        _check(determinant(matrix) == 1, "frame_proper_determinant", checks)
    maximum_condition = F(0)
    condition_by_half_angle = {}
    for half_angle in HALF_ANGLES:
        first, second = plane_normals(half_angle)
        condition = normal_system_condition_squared(first, second)
        condition_by_half_angle[str(half_angle)] = str(condition)
        maximum_condition = max(maximum_condition, condition)
        for radius in RADII:
            alternatives = [solve_axis_displacement(first, second, radius, signs) for signs in SIGNS]
            _check(len(set(alternatives)) == 4, "all_four_signed_axes_remain_distinct_alternatives", checks)
            _check(alternatives[0] == mul(-1, alternatives[3]) and alternatives[1] == mul(-1, alternatives[2]),
                   "signed_axis_central_symmetry", checks)
    completed = 0
    for half_angle, signs, radius, frame, scale, indexed_interval in product(
            HALF_ANGLES, SIGNS, RADII, FRAMES, SCALES, enumerate(FINITE_INTERVALS)):
        interval_index, interval = indexed_interval
        verify_case(half_angle, signs, radius, frame, scale, interval, checks)
        completed += 1
        stratum = ("near_parallel" if half_angle == F(1, 1_000_000) else
                   "near_antiparallel" if half_angle == 1_000_000 else
                   "right_angle" if half_angle == 1 else "other_oblique")
        strata[stratum] += 1
        sign_counts[f"{signs[0]:+d},{signs[1]:+d}"] += 1
        frame_counts[frame[0]] += 1
        scale_counts[str(scale)] += 1
        interval_counts[str(interval_index)] += 1
    _check(completed == EXPECTED_CASES, "complete_frozen_matrix_no_prefix", checks)
    _check(maximum_condition == 10**12, "near_parallel_exact_normal_system_condition_squared", checks)

    # The process tag is not an input geometric property. Equal geometric
    # descriptors admit different external process labels; no label is inferred.
    geometry = dict(radius="1", plane_normals=(("1", "0", "0"), ("0", "1", "0")),
                    finite_interval=("0", "1"), cylinder_axis_point=("1", "1", "0"))
    labelled = [dict(geometry=geometry, external_process_label=label) for label in ("formed", "machined")]
    _check(labelled[0]["geometry"] == labelled[1]["geometry"]
           and labelled[0]["external_process_label"] != labelled[1]["external_process_label"],
           "same_geometry_does_not_identify_manufacturing_process", checks)
    return {
        "schema_version": 1, "kind": "CAD_ROOT_AXIS_EXACT_MATH_AUDIT_V1", "status": "EXACT_IDENTITIES_PASSED",
        "case_count": completed, "expected_case_count": EXPECTED_CASES, "max_case_count": MAX_CASES,
        "arithmetic": "STDLIB_FRACTION_EXACT_NO_NUMERICAL_TOLERANCE",
        "max_numerator_or_denominator_bits": MAX_NUMERATOR_OR_DENOMINATOR_BITS,
        "strata": dict(sorted(strata.items())), "signed_offset_cases": dict(sorted(sign_counts.items())),
        "frame_cases": dict(sorted(frame_counts.items())), "scale_cases": dict(sorted(scale_counts.items())),
        "declared_finite_interval_cases": dict(sorted(interval_counts.items())),
        "exact_check_counts": dict(sorted(checks.items())), "exact_identity_check_count": sum(checks.values()),
        "gram_condition_number_by_half_angle": condition_by_half_angle,
        "maximum_gram_condition_number": str(maximum_condition),
        "maximum_normal_system_condition_squared": str(maximum_condition),
        "elapsed_seconds": time.perf_counter() - started,
        "finite_interval_status": "SEPARATELY_STIPULATED_NOT_INFERRED_FROM_INFINITE_TANGENCY",
        "root_role_status": "NOT_NOMINATED", "grouping_eligible": False,
        "event_identity_is_verified": False, "physical_geometry_is_verified": False,
        "native_runtime_verified": False, "observed_state": "UNKNOWN",
        "limitations": ["INVENTED_RATIONAL_GEOMETRY_NOT_NATIVE_OR_PHYSICAL_INPUT",
                        "NO_ARC_BRANCH_OR_INNER_OUTER_SELECTION_FROM_SIGN_ALTERNATIVES",
                        "NO_FINITE_TRIM_COMPLETENESS_OR_MANUFACTURABILITY_PROOF",
                        "NO_FLOATING_POINT_IMPLEMENTATION_OR_METROLOGY_QUALIFICATION"],
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True, indent=2))
