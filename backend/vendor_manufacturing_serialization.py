"""Fail-closed, JSON-only serialization of a native compensated draft.

No CAD files, profiles, native modules, preferences or geometry jobs are loaded
on import. The caller supplies the exact source shape and the pinned unfolder's
native result. This is a draft interchange seam, not manufacturing authority.
"""
from __future__ import annotations

import json
import math
import re


MAX_FACES = 1024
MAX_EDGES = 8192
MAX_BENDS = 32
MAX_COORDINATE_MM = 1_000_000.0
LINEAR_TOL_MM = 1e-7  # Native numerical tolerance, never a supplier tolerance.
ANGULAR_TOL_RAD = 1e-10
TAU = 2.0 * math.pi
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
COVERAGE_GROUPS = ("selected_source_face_indices", "opposite_source_face_indices",
                   "cap_source_face_indices", "hole_wall_source_face_indices")
REQUIRED_PROOFS = (
    "all_native_faces_accounted", "all_native_bend_cylinders_paired", "tree_skin_graph",
    "thickness_matches_opposite_planes_and_cylinders", "no_cleanup_or_curve_approximation",
    "patch_overlap_checked", "native_patch_and_cutout_coverage_checked",
    "independent_neutral_area_oracle_checked", "original_native_face_edge_vertex_order_preserved",
    "no_extraneous_or_duplicate_wrapper_topology", "geometry_checks_are_numerical_not_supplier_tolerance",
)


class ManufacturingSerializationUnavailable(ValueError):
    """No partial geometry or releaseable output is returned after refusal."""


def _require(condition, reason):
    if not condition:
        raise ManufacturingSerializationUnavailable(reason)


def _number(value, reason="NONFINITE_NUMBER", limit=MAX_COORDINATE_MM):
    _require(type(value) in (int, float), reason)
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ManufacturingSerializationUnavailable(reason) from None
    _require(math.isfinite(number) and abs(number) <= limit, reason)
    return number


def _point(vector, *, planar=False):
    values = [_number(getattr(vector, axis), "INVALID_NATIVE_COORDINATE") for axis in ("x", "y", "z")]
    if planar:
        _require(abs(values[2]) <= LINEAR_TOL_MM, "NONPLANAR_CUT_GEOMETRY")
    return values


def _sub(left, right):
    return [a - b for a, b in zip(left, right)]


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _length(vector):
    return math.hypot(*vector)


def _unit(vector):
    length = _length(vector)
    _require(math.isfinite(length) and length > LINEAR_TOL_MM, "INVALID_NATIVE_AXIS")
    return [value / length for value in vector]


def _cross(left, right):
    return [left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


def _native_bool(value, reason):
    _require(type(value) is bool and value, reason)


def _parameters(edge):
    first = _number(edge.FirstParameter, "INVALID_NATIVE_CURVE_PARAMETERS")
    last = _number(edge.LastParameter, "INVALID_NATIVE_CURVE_PARAMETERS")
    _require(last > first, "INVALID_NATIVE_CURVE_PARAMETERS")
    orientation = edge.Orientation
    _require(orientation in ("Forward", "Reversed"), "UNSUPPORTED_NATIVE_EDGE_ORIENTATION")
    return first, last, orientation


def _edge_length(edge):
    length = _number(edge.Length, "INVALID_NATIVE_EDGE_LENGTH", 10 * MAX_COORDINATE_MM)
    _require(length > LINEAR_TOL_MM, "DEGENERATE_NATIVE_EDGE")
    return length


def _native_vertex_correspondence(edge, start, finish):
    """Unique finite endpoint correspondence, never a closest-available match."""
    vertices = edge.Vertexes
    _require(len(vertices) == 2 and vertices[0].isSame(vertices[1]) is False,
             "TWO_DISTINCT_NATIVE_CUT_VERTICES_REQUIRED")
    points = [_point(vertex.Point, planar=True) for vertex in vertices]
    assignments = [(left, right) for left, right in ((0, 1), (1, 0))
                   if _length(_sub(points[left], start)) <= LINEAR_TOL_MM
                   and _length(_sub(points[right], finish)) <= LINEAR_TOL_MM]
    _require(len(assignments) == 1, "MISSING_OR_AMBIGUOUS_NATIVE_TRIM_VERTEX_CORRESPONDENCE")
    return vertices, points, assignments[0]


def _certified_cut_line(edge, start, finish, native_length):
    vertices, points, indices = _native_vertex_correspondence(edge, start, finish)
    del vertices  # Only the caller's native identities are used by wire closure below.
    projected = [[points[index][0], points[index][1], 0.0] for index in indices]
    displacement = max(_length(_sub(point, trim)) for point, trim in zip(projected, (start, finish)))
    projection_error = max(_length(_sub(point, points[index]))
                           for point, index in zip(projected, indices))
    # The exported affine segment is bounded against both the native trimmed
    # carrier and the retained finite vertex segment, including discarded z.
    endpoint_error = max(displacement, projection_error)
    projected_length = _length(_sub(projected[1], projected[0]))
    parametric_length = _length(_sub(finish, start))
    exported_length_error = abs(projected_length - native_length)
    parametric_length_error = abs(parametric_length - native_length)
    total_error = endpoint_error + exported_length_error + parametric_length_error
    _require(projected_length > LINEAR_TOL_MM and math.isfinite(total_error)
             and total_error <= LINEAR_TOL_MM, "NATIVE_CUT_VERTEX_LINE_PHYSICAL_ERROR_BUDGET")
    return projected, {
        "native_vertex_indices_in_parameter_traversal": list(indices),
        "maximum_projected_endpoint_to_native_trim_deviation_mm": displacement,
        "maximum_native_vertex_projection_deviation_mm": projection_error,
        "maximum_native_endpoint_carrier_deviation_mm": endpoint_error,
        "exported_chord_native_length_error_mm": exported_length_error,
        "parametric_chord_native_length_error_mm": parametric_length_error,
        "total_physical_error_bound_mm": total_error,
        "exact_native_vertex_xy_used": True, "proximity_snapping_used": False,
    }


def _analytic_edge(edge, layer):
    """Preserve native parameter traversal, including reflected circle axes."""
    _require(edge.ShapeType == "Edge", "NATIVE_EDGE_REQUIRED")
    _native_bool(edge.isValid(), "INVALID_NATIVE_EDGE")
    first, last, orientation = _parameters(edge)
    start_parameter, finish_parameter = (first, last) if orientation == "Forward" else (last, first)
    start, finish = _point(edge.valueAt(start_parameter), planar=True), _point(edge.valueAt(finish_parameter), planar=True)
    native_length = _edge_length(edge)
    curve = edge.Curve
    metadata = {"native_curve_type": curve.TypeId, "native_edge_orientation": orientation,
                "native_first_parameter": first, "native_last_parameter": last}
    if curve.TypeId == "Part::GeomLine":
        length = _length(_sub(finish, start))
        _require(length > LINEAR_TOL_MM and abs(length - native_length) <= LINEAR_TOL_MM,
                 "NATIVE_LINE_LENGTH_MISMATCH")
        if layer in ("CUT_OUTER", "CUT_INNER"):
            projected, certificate = _certified_cut_line(edge, start, finish, native_length)
            start, finish = projected
            metadata["native_line_endpoint_certificate"] = certificate
        return {"type": "LINE", "layer": layer, "start": start[:2], "end": finish[:2]}, metadata
    _require(curve.TypeId == "Part::GeomCircle", "UNSUPPORTED_ANALYTIC_CUT_CURVE")
    center = _point(curve.Center, planar=True)
    axis = _unit(_point(curve.Axis))
    _require(math.hypot(axis[0], axis[1]) <= ANGULAR_TOL_RAD, "NONPLANAR_NATIVE_CIRCLE_AXIS")
    radius = _number(curve.Radius, "INVALID_NATIVE_CIRCLE_RADIUS")
    _require(radius > LINEAR_TOL_MM, "INVALID_NATIVE_CIRCLE_RADIUS")
    span = last - first
    parameter_error = 16 * (math.ulp(first) + math.ulp(last) + math.ulp(TAU))
    _require(span <= TAU + parameter_error, "MULTITURN_NATIVE_CIRCLE_UNSUPPORTED")
    parameter_physical_error = radius * parameter_error
    length_error = abs(native_length - radius * span)
    carrier_error = max(abs(_length(_sub(point, center)) - radius) for point in (start, finish))
    physical_error = parameter_physical_error + length_error + carrier_error
    _require(math.isfinite(physical_error) and physical_error <= LINEAR_TOL_MM,
             "ILL_CONDITIONED_NATIVE_CIRCLE_PHYSICAL_ERROR_BUDGET")
    closed = edge.isClosed()
    _require(type(closed) is bool, "INVALID_NATIVE_CLOSURE_CERTIFICATE")
    periodic = curve.isPeriodic()
    _require(type(periodic) is bool and periodic, "NATIVE_PERIODIC_CIRCLE_REQUIRED")
    vertices = edge.Vertexes
    _require(0 < len(vertices) <= 2, "INVALID_NATIVE_CIRCLE_VERTEX_TABLE")
    closed_vertices = len(vertices) == 1 or vertices[0].isSame(vertices[1]) is True
    full_parameters = abs(span - TAU) <= parameter_error
    closure_deviation = radius * abs(span - TAU) if closed else 0.0
    total_physical_error = physical_error + closure_deviation
    metadata["native_circle_certificate"] = {
        "native_edge_is_closed": closed, "native_closed_vertex_identity": closed_vertices,
        "native_periodic_curve": periodic, "native_full_period_parameters": full_parameters,
        "parameter_roundoff_bound_rad": parameter_error,
        "parameter_roundoff_bound_mm": parameter_physical_error,
        "total_physical_error_bound_mm": total_physical_error,
        "endpoint_gap_heuristic_used": False,
    }
    if closed:
        _require(closed_vertices and full_parameters and total_physical_error <= LINEAR_TOL_MM,
                 "UNPROVEN_OR_ILL_CONDITIONED_NATIVE_FULL_CIRCLE")
        return {"type": "CIRCLE", "layer": layer, "center": center[:2], "radius_mm": radius,
                "sweep_rad": TAU, "full_circle_verified": True}, metadata
    _require(not closed_vertices and not full_parameters and span > ANGULAR_TOL_RAD,
             "AMBIGUOUS_NATIVE_PARTIAL_ARC")
    sign = (1.0 if axis[2] > 0 else -1.0) * (1.0 if orientation == "Forward" else -1.0)
    angle = math.atan2(start[1] - center[1], start[0] - center[0])
    return {"type": "ARC", "layer": layer, "center": center[:2], "radius_mm": radius,
            "start_angle_rad": angle, "sweep_rad": sign * span}, metadata


def _ordered_wire_edges(wire):
    _native_bool(wire.isValid(), "INVALID_NATIVE_CUT_WIRE")
    _native_bool(wire.isClosed(), "OPEN_NATIVE_CUT_WIRE")
    edges, ordered = wire.Edges, wire.OrderedEdges
    _require(0 < len(edges) <= MAX_EDGES and len(ordered) == len(edges), "INCOMPLETE_NATIVE_WIRE_EDGE_TABLE")
    hashes = {edge.hashCode(): edge for edge in edges}
    _require(len(hashes) == len(edges), "AMBIGUOUS_NATIVE_WIRE_EDGE_IDENTITY")
    seen = set()
    for edge in ordered:
        key = edge.hashCode()
        _require(key in hashes and key not in seen and edge.isSame(hashes[key]) is True,
                 "INCOMPLETE_NATIVE_ORDERED_WIRE_COVERAGE")
        seen.add(key)
    return ordered


def _cut_edge_endpoint_vertices(edge, metadata):
    """Retain analytic arc/circle rows; use native identities only for topology."""
    if edge.Curve.TypeId == "Part::GeomLine":
        indices = metadata["native_line_endpoint_certificate"]["native_vertex_indices_in_parameter_traversal"]
        return tuple(edge.Vertexes[index] for index in indices)
    circle = metadata["native_circle_certificate"]
    if circle["native_edge_is_closed"]:
        vertices = edge.Vertexes
        return vertices[0], vertices[0] if len(vertices) == 1 else vertices[1]
    first, last, orientation = _parameters(edge)
    parameters = (first, last) if orientation == "Forward" else (last, first)
    start, finish = (_point(edge.valueAt(parameter), planar=True) for parameter in parameters)
    vertices, _, indices = _native_vertex_correspondence(edge, start, finish)
    return vertices[indices[0]], vertices[indices[1]]


def _native_wire_vertex_closure(endpoints):
    for index, (_, finish) in enumerate(endpoints):
        start = endpoints[(index + 1) % len(endpoints)][0]
        _require(finish.isSame(start) is True, "CUT_WIRE_ENDPOINT_NATIVE_IDENTITY_GAP")
        # Native equality also requires the exact retained point. A façade that
        # claims shared identity but publishes inconsistent points must abstain.
        _require(_point(finish.Point, planar=True) == _point(start.Point, planar=True),
                 "INCONSISTENT_SHARED_NATIVE_VERTEX_POINT")


def _cut_rows(flat):
    _require(flat.ShapeType == "Face" and len(flat.Faces) == 1, "ONE_NATIVE_FLAT_FACE_REQUIRED")
    _native_bool(flat.isValid(), "INVALID_NATIVE_FLAT_FACE")
    _require(flat.Surface.TypeId == "Part::GeomPlane", "PLANAR_NATIVE_FLAT_FACE_REQUIRED")
    axis = _unit(_point(flat.Surface.Axis))
    _require(math.hypot(axis[0], axis[1]) <= ANGULAR_TOL_RAD, "NONPLANAR_FLAT_FACE_AXIS")
    bounds = flat.BoundBox
    for name in ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"):
        _number(getattr(bounds, name), "INVALID_NATIVE_FLAT_BOUNDS")
    _require(abs(bounds.ZMin) <= LINEAR_TOL_MM and abs(bounds.ZMax) <= LINEAR_TOL_MM
             and bounds.XMax > bounds.XMin and bounds.YMax > bounds.YMin,
             "DEGENERATE_OR_NONPLANAR_NATIVE_FLAT_BOUNDS")
    area = _number(flat.Area, "INVALID_NATIVE_FLAT_AREA", MAX_COORDINATE_MM ** 2 * 4)
    _require(area > LINEAR_TOL_MM ** 2, "DEGENERATE_NATIVE_FLAT_FACE")
    wires = flat.Wires
    _require(0 < len(wires) <= MAX_FACES, "NATIVE_CUT_WIRE_BUDGET_EXCEEDED")
    outer = [index for index, wire in enumerate(wires) if wire.isSame(flat.OuterWire) is True]
    _require(len(outer) == 1, "UNPROVEN_NATIVE_OUTER_WIRE")
    rows, provenance, count = [], [], 0
    for wire_index, wire in enumerate(wires):
        layer = "CUT_OUTER" if wire_index == outer[0] else "CUT_INNER"
        endpoints = []
        for edge_index, edge in enumerate(_ordered_wire_edges(wire)):
            count += 1
            _require(count <= MAX_EDGES, "ANALYTIC_CUT_EDGE_BUDGET_EXCEEDED")
            row, native = _analytic_edge(edge, layer)
            endpoints.append(_cut_edge_endpoint_vertices(edge, native))
            provenance.append({"row_index": len(rows), "wire_index": wire_index,
                               "edge_index": edge_index, "exact_native_wire_vertex_identity_closure": True, **native})
            rows.append(row)
        _native_wire_vertex_closure(endpoints)
    _require(len(flat.Edges) == count, "INCOMPLETE_NATIVE_FLAT_EDGE_COVERAGE")
    return rows, provenance, {"min": [float(bounds.XMin), float(bounds.YMin)],
                              "max": [float(bounds.XMax), float(bounds.YMax)], "area_mm2": area}


def _coverage(shape, proof):
    _require(type(proof) is dict, "NATIVE_COVERAGE_PROOF_REQUIRED")
    faces = shape.Faces
    _require(0 < len(faces) <= MAX_FACES and type(proof.get("native_face_count")) is int
             and proof["native_face_count"] == len(faces), "NATIVE_SOURCE_FACE_COUNT_MISMATCH")
    _require(all(proof.get(name) is True for name in REQUIRED_PROOFS), "INCOMPLETE_NATIVE_COVERAGE_PROOF")
    categories, all_indices = {}, []
    for category in COVERAGE_GROUPS:
        indices = proof.get(category)
        _require(type(indices) is list and len(indices) <= MAX_FACES
                 and all(type(index) is int and 0 <= index < len(faces) for index in indices)
                 and len(set(indices)) == len(indices), "INVALID_SOURCE_FACE_COVERAGE_INDICES")
        categories[category] = set(indices)
        all_indices.extend(indices)
    _require(len(all_indices) == len(faces) and len(set(all_indices)) == len(faces)
             and set(all_indices) == set(range(len(faces))), "SOURCE_FACE_COVERAGE_NOT_COMPLETE_AND_UNIQUE")
    root = proof.get("root_source_face_index")
    _require(type(root) is int and root in categories[COVERAGE_GROUPS[0]]
             and faces[root].Surface.TypeId == "Part::GeomPlane", "INVALID_COVERAGE_ROOT_FACE")
    return faces, categories, [{"source_face_index": index, "coverage_role": category}
                              for category in COVERAGE_GROUPS for index in sorted(categories[category])]


def _cylinder_marker(face, paired, row):
    _require(face.Surface.TypeId == paired.Surface.TypeId == "Part::GeomCylinder", "NATIVE_BEND_CYLINDER_PAIR_REQUIRED")
    cylinder, opposite = face.Surface, paired.Surface
    axis = _unit(_point(cylinder.Axis))
    paired_axis = _unit(_point(opposite.Axis))
    _require(_length(_cross(axis, paired_axis)) <= ANGULAR_TOL_RAD, "NONCOAXIAL_NATIVE_BEND_PAIR")
    center, paired_center = _point(cylinder.Center), _point(opposite.Center)
    displacement = _sub(paired_center, center)
    _require(_length(_sub(displacement, [value * _dot(displacement, axis) for value in axis])) <= LINEAR_TOL_MM,
             "NONCOAXIAL_NATIVE_BEND_PAIR")
    radius = _number(cylinder.Radius, "INVALID_NATIVE_BEND_RADIUS")
    paired_radius = _number(opposite.Radius, "INVALID_NATIVE_BEND_RADIUS")
    thickness = row["thickness_mm"]
    _require(min(radius, paired_radius) > LINEAR_TOL_MM
             and abs(abs(radius - paired_radius) - thickness) <= LINEAR_TOL_MM
             and abs(min(radius, paired_radius) - row["inner_radius_mm"]) <= LINEAR_TOL_MM,
             "NATIVE_BEND_PAIR_RADIUS_MISMATCH")
    parameters = list(face.ParameterRange)
    paired_parameters = list(paired.ParameterRange)
    _require(len(parameters) == len(paired_parameters) == 4, "INVALID_NATIVE_BEND_PARAMETER_RANGE")
    parameters = [_number(value, "INVALID_NATIVE_BEND_PARAMETER_RANGE") for value in parameters]
    paired_parameters = [_number(value, "INVALID_NATIVE_BEND_PARAMETER_RANGE") for value in paired_parameters]
    sweep = parameters[1] - parameters[0]
    _require(0 < sweep <= math.radians(175) and parameters[3] > parameters[2]
             and abs(sweep - row["sweep_rad"]) <= ANGULAR_TOL_RAD
             and abs(paired_parameters[1] - paired_parameters[0] - sweep) <= ANGULAR_TOL_RAD,
             "NATIVE_BEND_PAIR_SWEEP_MISMATCH")
    middle_u = (parameters[0] + parameters[1]) / 2
    native_points = [_point(face.valueAt(middle_u, value)) for value in parameters[2:]]
    axial = [_dot(_sub(point, center), axis) for point in native_points]
    radial = [_sub(_sub(point, center), [value * position for value in axis])
              for point, position in zip(native_points, axial)]
    _require(all(abs(_length(vector) - radius) <= LINEAR_TOL_MM for vector in radial)
             and _length(_sub(radial[0], radial[1])) <= LINEAR_TOL_MM,
             "INVALID_NATIVE_MID_SKIN_CARRIER")
    radial_unit = _unit(radial[0])
    paired_u = (paired_parameters[0] + paired_parameters[1]) / 2
    paired_point = _point(paired.valueAt(paired_u, paired_parameters[2]))
    paired_offset = _sub(paired_point, paired_center)
    paired_radial = _unit(_sub(paired_offset, [value * _dot(paired_offset, paired_axis) for value in paired_axis]))
    _require(_length(_sub(radial_unit, paired_radial)) <= ANGULAR_TOL_RAD, "NATIVE_BEND_PAIR_HALF_TURN_MISMATCH")
    middle_radius = (radius + paired_radius) / 2.0
    points = [[center[index] + axis[index] * position + radial_unit[index] * middle_radius
               for index in range(3)] for position in axial]
    _require(_length(_sub(points[1], points[0])) > LINEAR_TOL_MM, "DEGENERATE_FOLDED_MID_SKIN_MARKER")
    return {"pair_id": row["pair_id"], "source_face_index": row["source_face_index"],
            "paired_source_face_index": row["paired_source_face_index"],
            "line_start": points[0], "line_end": points[1],
            "marker_convention": "SOURCE_CAD_MID_SKIN_RADIUS_HALF_NATIVE_SWEEP",
            "mid_skin_radius_mm": middle_radius, "compensated_k_location": False,
            "vendor_feature_id_mapping": "UNMAPPED"}


def _bend_rows(ledger, flat, faces, categories, proof):
    _require(type(ledger) is list and 0 < len(ledger) <= MAX_BENDS
             and type(proof.get("native_bend_count")) is int and proof["native_bend_count"] == len(ledger),
             "INVALID_NATIVE_BEND_LEDGER")
    expected = {index for index in categories[COVERAGE_GROUPS[0]]
                if faces[index].Surface.TypeId == "Part::GeomCylinder"}
    expected_pairs = {index for index in categories[COVERAGE_GROUPS[1]]
                      if faces[index].Surface.TypeId == "Part::GeomCylinder"}
    seen, paired_seen, rows, provenance, markers = set(), set(), [], [], []
    for native in ledger:
        _require(type(native) is dict, "INVALID_NATIVE_BEND_LEDGER_ROW")
        index, paired = native.get("source_face_index"), native.get("paired_source_face_index")
        _require(type(index) is int and index in expected and index not in seen
                 and type(paired) is int and paired in expected_pairs and paired not in paired_seen,
                 "INCOMPLETE_OR_DUPLICATE_NATIVE_BEND_PAIR_MAPPING")
        seen.add(index); paired_seen.add(paired)
        row = {key: _number(native.get(key), "INVALID_NATIVE_BEND_LEDGER_SCALAR")
               for key in ("sweep_rad", "included_angle_deg", "inner_radius_mm", "bend_allowance_mm", "thickness_mm", "k_factor_ansi")}
        _require(0 < row["sweep_rad"] <= math.radians(175) and 0 < abs(row["included_angle_deg"]) <= 180
                 and row["inner_radius_mm"] > LINEAR_TOL_MM and row["thickness_mm"] > LINEAR_TOL_MM
                 and 0 <= row["k_factor_ansi"] <= 1 and row["bend_allowance_mm"] > LINEAR_TOL_MM,
                 "INVALID_NATIVE_BEND_LEDGER_SCALAR")
        allowance = row["sweep_rad"] * (row["inner_radius_mm"] + row["k_factor_ansi"] * row["thickness_mm"])
        _require(abs(allowance - row["bend_allowance_mm"]) <= LINEAR_TOL_MM, "BEND_ALLOWANCE_LEDGER_MISMATCH")
        direction = native.get("bend_direction")
        _require(direction in ("up", "down"), "UNKNOWN_NATIVE_BEND_DIRECTION")
        expected_included = (180.0 - math.degrees(row["sweep_rad"])) * (1.0 if direction == "up" else -1.0)
        _require(abs(math.radians(row["included_angle_deg"] - expected_included)) <= ANGULAR_TOL_RAD,
                 "SIGNED_INCLUDED_ANGLE_SWEEP_DIRECTION_MISMATCH")
        line = native.get("line")
        _require(line is not None and line.Curve.TypeId == "Part::GeomLine", "FINITE_NATIVE_FLAT_BEND_LINE_REQUIRED")
        geometry, _ = _analytic_edge(line, "BEND_UP" if direction == "up" else "BEND_DOWN")
        length = _edge_length(line)
        common_length = _number(line.common(flat).Length, "INVALID_NATIVE_BEND_CONTAINMENT", 10 * MAX_COORDINATE_MM)
        outside_length = _number(line.cut(flat).Length, "INVALID_NATIVE_BEND_CONTAINMENT", 10 * MAX_COORDINATE_MM)
        _require(common_length >= 0 and outside_length >= 0 and abs(common_length - length) <= LINEAR_TOL_MM
                 and outside_length <= LINEAR_TOL_MM, "FLAT_BEND_LINE_OVERHANG_OR_CUTOUT_INTERSECTION")
        pair_id = f"native_face_pair:{min(index, paired)}:{max(index, paired)}"
        row.update({"pair_id": pair_id, "source_face_index": index, "paired_source_face_index": paired,
                    "bend_direction": direction, "whole_native_line_inside_cut_face": True,
                    "line_was_trimmed": False, "vendor_feature_id_mapping": "UNMAPPED"})
        row.update({"included_angle_convention": "UPSTREAM_SIGNED_INCLUDED_ANGLE_DEG",
                    "included_angle_abs_deg": abs(row["included_angle_deg"])})
        rows.append(geometry)
        provenance.append({"row_index": len(rows) - 1, **row})
        markers.append(_cylinder_marker(faces[index], faces[paired], row))
    _require(seen == expected and paired_seen == expected_pairs, "OMITTED_NATIVE_BEND_CYLINDER_PAIR")
    return rows, provenance, markers


def _pure_json(value, budget, depth=0):
    budget[0] += 1
    _require(budget[0] <= 500_000 and depth <= 12, "SERIALIZED_JSON_BUDGET_EXCEEDED")
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        _number(value, "NONFINITE_SERIALIZED_JSON", 1e18)
        return value
    if type(value) is str:
        _require(len(value) <= 4096, "SERIALIZED_STRING_BUDGET_EXCEEDED")
        return value
    if type(value) in (list, tuple):
        return [_pure_json(item, budget, depth + 1) for item in value]
    if type(value) is dict:
        _require(all(type(key) is str and len(key) <= 128 for key in value), "INVALID_SERIALIZED_JSON_KEY")
        _require(all(value.get(key, False) is False for key in ("release_allowed", "manufacturing_ready", "units_verified")),
                 "NESTED_RELEASE_OR_UNIT_APPROVAL_FORBIDDEN")
        return {key: _pure_json(item, budget, depth + 1) for key, item in value.items()}
    raise ManufacturingSerializationUnavailable("NON_JSON_NATIVE_PAYLOAD")


def serialize_compensated_draft(shape, result, source_sha256, profile_sha256):
    """Return analytic JSON rows and exact native provenance, or refuse wholly.

    Source/profile hashes bind the caller's controlled input; they do not prove
    material approval or units. All release flags stay false. No tessellation is
    generated here: an optional visual-only mesh must be separately bounded.
    """
    _require(type(source_sha256) is str and HASH_PATTERN.fullmatch(source_sha256) is not None,
             "VALID_SOURCE_SHA256_REQUIRED")
    _require(type(profile_sha256) is str and HASH_PATTERN.fullmatch(profile_sha256) is not None,
             "VALID_PROFILE_SHA256_REQUIRED")
    _require(type(result) is dict and result.get("status") == "DRAFT_COMPUTED"
             and result.get("release_allowed") is False and result.get("manufacturing_ready") is False
             and result.get("units_verified") is False and result.get("units") == "FreeCAD_internal_mm"
             and result.get("source_sha256") in (None, source_sha256), "DRAFT_ONLY_NATIVE_RESULT_REQUIRED")
    try:
        _require(shape.ShapeType in ("Solid", "Compound") and len(shape.Solids) == 1, "ONE_SOURCE_SOLID_REQUIRED")
        _native_bool(shape.isValid(), "INVALID_SOURCE_NATIVE_SHAPE")
        _require(0 < len(shape.Vertexes) <= 2 * MAX_EDGES, "SOURCE_NATIVE_VERTEX_BUDGET_EXCEEDED")
        for vertex in shape.Vertexes:
            _point(vertex.Point)
        proof = result.get("coverage_proof")
        faces, categories, source_coverage = _coverage(shape, proof)
        flat = result.get("flat_face")
        _require(flat is not None, "NATIVE_FLAT_FACE_REQUIRED")
        cut_rows, cut_provenance, bounds = _cut_rows(flat)
        numerical_tolerance = _number(proof.get("linear_numerical_tolerance_mm"), "INVALID_NATIVE_PROOF_TOLERANCE")
        area_tolerance = _number(proof.get("area_numerical_tolerance_mm2"), "INVALID_NATIVE_PROOF_TOLERANCE")
        _require(numerical_tolerance == LINEAR_TOL_MM and area_tolerance > 0,
                 "INVALID_NATIVE_PROOF_TOLERANCE")
        for name in ("flat_face_area_mm2", "independent_neutral_area_mm2"):
            reference_area = _number(proof.get(name), "INVALID_NATIVE_AREA_PROOF", MAX_COORDINATE_MM ** 2 * 4)
            _require(reference_area > 0 and abs(reference_area - bounds["area_mm2"]) <= area_tolerance,
                     "NATIVE_FLAT_AREA_PROOF_MISMATCH")
        bend_rows, bend_provenance, markers = _bend_rows(result.get("bend_ledger"), flat, faces, categories, proof)
        output = {
            "schema": "VENDOR_COMPENSATED_DRAFT_ANALYTIC_V1", "status": "DRAFT_SERIALIZED",
            "source_sha256": source_sha256, "profile_sha256": profile_sha256,
            "release_allowed": False, "manufacturing_ready": False, "units_verified": False,
            "units": "internal_mm", "cut_geometry_is_analytic": True,
            "cut_rows": cut_rows, "bend_rows": bend_rows,
            "cut_row_provenance": cut_provenance, "bend_row_provenance": bend_provenance,
            "folded_bend_markers": markers, "source_face_coverage": source_coverage,
            "vendor_feature_id_mapping": "UNMAPPED", "coverage_proof": proof,
            "flat_bounds": bounds, "visual_mesh": None,
            "visual_mesh_status": "OMITTED_NO_UNBOUNDED_TESSELLATION",
            "linear_numerical_tolerance_mm": LINEAR_TOL_MM,
            "geometry_checks_are_numerical_not_supplier_tolerance": True,
            "upstream_commit": result.get("upstream_commit"),
            "freecad_source_commit": result.get("freecad_source_commit"),
            "freecad_version": result.get("freecad_version"),
        }
        output = _pure_json(output, [0])
        # An independent stdlib encoder check forbids any accidental NaN/native object.
        json.dumps(output, allow_nan=False)
        return output
    except ManufacturingSerializationUnavailable:
        raise
    except Exception as error:
        raise ManufacturingSerializationUnavailable("NATIVE_SERIALIZATION_EXCEPTION: " + type(error).__name__) from error
