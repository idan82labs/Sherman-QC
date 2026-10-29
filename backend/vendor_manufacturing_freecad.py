"""Strict draft-only adapter for a pinned FreeCAD SheetMetal unfolder.

This module does not read CAD files, set preferences, export DXF, infer material
compensation, or authorize manufacture. The caller owns source/unit/profile
provenance. Invoke only inside a supervised, serial FreeCAD Python process.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import threading


UPSTREAM_COMMIT = "3141e205ea340d68e69ff0e257a44e27d9bcb591"
FREECAD_SOURCE_COMMIT = "145529fe741292ff0b3977a01195bf0247425794"
UPSTREAM_SOURCE_HASHES = {
    "SheetMetalNewUnfolder.py": "295ddf32da7f85e59be30b08bb48399c8b4acd23c0e18adb302eab33797f1ff7",
    "SheetMetalTools.py": "e0f3abe620f59b1ddeccaef34a9f1ebb34d7f24db4412c728c8ecab6c4fd5a44",
    "SheetMetalText.py": "9512f5d5cae601a6bbd89e7418487648a932dc0e51b3b53da6d7afe192001fb6",
}
MAX_FACES = 1024
MAX_EDGES = 8192
MAX_VERTICES = 2 * MAX_EDGES
MAX_BENDS = 32
MAX_PAIR_TESTS = 65536
MAX_IDENTITY_COMPARISONS = 1_000_000
MAX_COORDINATE_MM = 1_000_000.
LINEAR_TOL_MM = 1e-7  # OCCT confusion precision, not a material/supplier tolerance.
ANGULAR_TOL = 1e-10
MAX_BEND_SWEEP_RAD = math.radians(175.)
_LOCK = threading.Lock()


class ManufacturingUnfoldUnavailable(ValueError):
    """Fail-closed reason; no partial flat geometry is returned."""


def _require(condition, reason):
    if not condition:
        raise ManufacturingUnfoldUnavailable(reason)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _unit(vector):
    from FreeCAD import Vector
    result = Vector(vector)
    _require(_finite(result.Length) and result.Length > LINEAR_TOL_MM, "INVALID_NATIVE_VECTOR")
    result.normalize()
    return result


def _parallel(a, b):
    return _unit(a).cross(_unit(b)).Length <= ANGULAR_TOL


def _area_tolerance(*faces):
    return LINEAR_TOL_MM * max(1., sum(float(face.Length) for face in faces))


def _source_index(face, faces, counter):
    matches = []
    for index, candidate in enumerate(faces):
        counter[0] += 1
        _require(counter[0] <= MAX_IDENTITY_COMPARISONS, "NATIVE_IDENTITY_BUDGET_EXCEEDED")
        if face.isSame(candidate):
            matches.append(index)
    _require(len(matches) == 1, "AMBIGUOUS_NATIVE_FACE_IDENTITY")
    return matches[0]


def _normalize_single_solid(shape, root):
    """Admit only an identity-preserving, direct single-solid wrapper.

    FreeCAD boolean operations may return a Compound around one Solid. Neither
    filtering extra topology nor renumbering native tables is permitted here:
    the caller's original face index must still identify exactly the same face.
    Nested/other wrapper forms conservatively abstain.
    """
    _require(shape.ShapeType in {"Solid", "Compound"} and len(shape.Solids) == 1
             and shape.isValid(), "ONE_VALID_SOLID_REQUIRED")
    solid = shape.Solids[0]
    _require(solid.ShapeType == "Solid" and solid.isValid(), "ONE_VALID_SOLID_REQUIRED")
    original_faces = shape.Faces
    _require(type(root) is int and 0 <= root < len(original_faces), "INVALID_ROOT_FACE_INDEX")
    for name, cap in (("Faces", MAX_FACES), ("Edges", MAX_EDGES), ("Vertexes", MAX_VERTICES)):
        original, normalized = getattr(shape, name), getattr(solid, name)
        _require(0 < len(original) <= cap and 0 < len(normalized) <= cap,
                 "NATIVE_GEOMETRY_BUDGET_EXCEEDED")
        _require(len(original) == len(normalized), "WRAPPER_NATIVE_TABLE_MISMATCH")
        # A hash collision also abstains: hashes are never used to infer native
        # equality. isSame below proves every original row against its Solid row.
        _require(len({item.hashCode() for item in original}) == len(original)
                 and len({item.hashCode() for item in normalized}) == len(normalized),
                 "DUPLICATE_OR_AMBIGUOUS_WRAPPER_NATIVE_TABLE")
        _require(all(left.isSame(right) and left.Orientation == right.Orientation
                     for left, right in zip(original, normalized)),
                 "WRAPPER_NATIVE_TABLE_ORDER_MISMATCH")
    if shape.ShapeType == "Compound":
        children = shape.childShapes()
        _require(len(children) == 1 and children[0].ShapeType == "Solid" and children[0].isSame(solid)
                 and children[0].Orientation == solid.Orientation,
                 "EXTRANEOUS_OR_DUPLICATE_WRAPPER_TOPOLOGY")
    else:
        _require(shape.isSame(solid) and shape.Orientation == solid.Orientation,
                 "SOLID_NATIVE_IDENTITY_MISMATCH")
    return solid, {"input_native_shape_type": shape.ShapeType,
                   "single_solid_wrapper_normalized": shape.ShapeType == "Compound",
                   "original_native_face_edge_vertex_order_preserved": True,
                   "no_extraneous_or_duplicate_wrapper_topology": True}


def _native_preflight(shape, root):
    _require(shape.ShapeType == "Solid" and len(shape.Solids) == 1 and shape.isValid(), "ONE_VALID_SOLID_REQUIRED")
    faces, edges = shape.Faces, shape.Edges
    _require(0 < len(faces) <= MAX_FACES and 0 < len(edges) <= MAX_EDGES, "NATIVE_GEOMETRY_BUDGET_EXCEEDED")
    _require(type(root) is int and 0 <= root < len(faces), "INVALID_ROOT_FACE_INDEX")
    _require(faces[root].Surface.TypeId == "Part::GeomPlane", "PLANAR_ROOT_REQUIRED")
    _require(len({f.hashCode() for f in faces}) == len(faces), "NATIVE_FACE_HASH_COLLISION")
    _require(len({e.hashCode() for e in edges}) == len(edges), "NATIVE_EDGE_HASH_COLLISION")
    for vertex in shape.Vertexes:
        point = vertex.Point
        _require(all(_finite(v) and abs(v) <= MAX_COORDINATE_MM for v in (point.x, point.y, point.z)),
                 "NONFINITE_OR_ILL_CONDITIONED_COORDINATES")
    for face in faces:
        _require(face.Surface.TypeId in {"Part::GeomPlane", "Part::GeomCylinder"}, "UNSUPPORTED_NATIVE_FORMING_SURFACE")
        _require(_finite(face.Area) and face.Area > LINEAR_TOL_MM ** 2, "DEGENERATE_NATIVE_FACE")
        if face.Surface.TypeId == "Part::GeomCylinder":
            _require(_finite(face.Surface.Radius) and LINEAR_TOL_MM < face.Surface.Radius <= MAX_COORDINATE_MM
                     and all(_finite(v) for v in face.ParameterRange), "INVALID_NATIVE_CYLINDER")
    for edge in edges:
        _require(edge.Curve.TypeId in {"Part::GeomLine", "Part::GeomCircle"}, "UNSUPPORTED_NATIVE_TRIM")
        _require(_finite(edge.Length) and edge.Length > LINEAR_TOL_MM, "DEGENERATE_NATIVE_EDGE")
    return faces, edges


def _validate_tree(graph, faces, edges, root):
    nodes = set(graph.nodes)
    _require(root in nodes and graph.number_of_edges() == len(nodes) - 1, "NON_TREE_SKIN_GRAPH")
    cylinders = set()
    for index in nodes:
        _require(type(index) is int and 0 <= index < len(faces), "INVALID_UPSTREAM_GRAPH_NODE")
        if faces[index].Surface.TypeId == "Part::GeomCylinder":
            cylinders.add(index)
            _require(graph.degree(index) == 2, "CYLINDER_REQUIRES_TWO_FLANGE_SEAMS")
            sweep = faces[index].ParameterRange[1] - faces[index].ParameterRange[0]
            _require(LINEAR_TOL_MM < sweep <= MAX_BEND_SWEEP_RAD, "UNSUPPORTED_BEND_SWEEP")
    _require(0 < len(cylinders) <= MAX_BENDS, "BEND_COUNT_UNSUPPORTED")
    for a, b, data in graph.edges(data=True):
        _require({faces[a].Surface.TypeId, faces[b].Surface.TypeId} == {"Part::GeomPlane", "Part::GeomCylinder"},
                 "UNCERTAIN_SKIN_ADJACENCY")
        label = data.get("label")
        _require(type(label) is int and 0 <= label < len(edges) and edges[label].Curve.TypeId == "Part::GeomLine",
                 "STRAIGHT_NATIVE_BEND_SEAM_REQUIRED")
        cylinder = faces[a] if a in cylinders else faces[b]
        _require(_parallel(edges[label].Curve.Direction, cylinder.Surface.Axis), "BEND_SEAM_NOT_AXIS_PARALLEL")
    return cylinders


def _translated_planes_match(left, right, thickness):
    if not _parallel(left.Surface.Axis, right.Surface.Axis):
        return False
    normal = _unit(left.Surface.Axis)
    separation = (right.Surface.Position - left.Surface.Position).dot(normal)
    if abs(abs(separation) - thickness) > LINEAR_TOL_MM:
        return False
    translated = left.translated(normal * separation)
    tolerance = _area_tolerance(left, right)
    if abs(left.Area - right.Area) > tolerance:
        return False
    return abs(translated.common(right).Area - left.Area) <= tolerance


def _cylinder_pair_matches(a, b, graph_a, graph_b, faces, edges, thickness):
    ca, cb = a.Surface, b.Surface
    if not _parallel(ca.Axis, cb.Axis):
        return False
    axis = _unit(ca.Axis)
    if cb.Center.distanceToLine(ca.Center, axis) > LINEAR_TOL_MM:
        return False
    if abs(abs(ca.Radius - cb.Radius) - thickness) > LINEAR_TOL_MM:
        return False
    if min(ca.Radius, cb.Radius) <= LINEAR_TOL_MM:
        return False
    ra, rb = a.ParameterRange, b.ParameterRange
    if max(ca.Radius, cb.Radius) * abs((ra[1] - ra[0]) - (rb[1] - rb[0])) > LINEAR_TOL_MM:
        return False
    va = sorted((v.Point - ca.Center).dot(axis) for v in a.Vertexes)
    vb = sorted((v.Point - ca.Center).dot(axis) for v in b.Vertexes)
    if max(abs(va[0] - vb[0]), abs(va[-1] - vb[-1])) > LINEAR_TOL_MM:
        return False
    ia = next(i for i, f in enumerate(faces) if f.isSame(a))
    ib = next(i for i, f in enumerate(faces) if f.isSame(b))
    radial_sets = []
    for index, graph, surface in ((ia, graph_a, ca), (ib, graph_b, cb)):
        radial = []
        for neighbor in graph.neighbors(index):
            point = edges[graph.get_edge_data(index, neighbor)["label"]].CenterOfMass
            vector = point - surface.Center
            radial.append(_unit(vector - axis * vector.dot(axis)))
        radial_sets.append(radial)
    radius = max(ca.Radius, cb.Radius)
    return min(max((radial_sets[0][i] - radial_sets[1][i]).Length for i in (0, 1)),
               max((radial_sets[0][i] - radial_sets[1][1 - i]).Length for i in (0, 1))) * radius <= LINEAR_TOL_MM


def _line_rails_translate(a, b, offset):
    if a.Curve.TypeId != "Part::GeomLine" or b.Curve.TypeId != "Part::GeomLine":
        return False
    left = [a.valueAt(a.FirstParameter) + offset, a.valueAt(a.LastParameter) + offset]
    right = [b.valueAt(b.FirstParameter), b.valueAt(b.LastParameter)]
    return min(max(left[i].distanceToPoint(right[i]) for i in (0, 1)),
               max(left[i].distanceToPoint(right[1 - i]) for i in (0, 1))) <= LINEAR_TOL_MM


def _circle_rails_cover_wall(rails, wall, plane):
    """Whole-circle interval/carrier bound; caller shares both sides' budget."""
    _require(0 < len(rails) <= 32, "CIRCULAR_RAIL_SET_BUDGET_EXCEEDED")
    surface, normal = wall.Surface, _unit(wall.Surface.Axis)
    span = wall.ParameterRange[1] - wall.ParameterRange[0]
    center = surface.Center + normal * (plane.Surface.Position - surface.Center).dot(normal)
    initial_radial = rails[0].valueAt(rails[0].FirstParameter) - center
    reference_x = _unit(initial_radial - normal * initial_radial.dot(normal))
    reference_y = _unit(normal.cross(reference_x))
    tau, intervals, carrier_error = 2 * math.pi, [], 0.
    coordinate_scale = max(1., *(abs(v) for v in (center.x, center.y, center.z)))
    parameter_scale = max(1., *(abs(v) for v in wall.ParameterRange))
    for rail in rails:
        _require(rail.Curve.TypeId == "Part::GeomCircle", "NONCIRCULAR_HOLE_RAIL")
        circle = rail.Curve
        radial = _unit(rail.valueAt(rail.FirstParameter) - circle.Center)
        tangent = _unit(rail.tangentAt(rail.FirstParameter))
        sign = 1. if radial.cross(tangent).dot(normal) > 0. else -1.
        phase = math.atan2(radial.dot(reference_y), radial.dot(reference_x)) % tau
        expected_radial = reference_x * math.cos(phase) + reference_y * math.sin(phase)
        expected_tangent = (reference_x * -math.sin(phase) + reference_y * math.cos(phase)) * sign
        physical_error = (circle.Center.distanceToPoint(center)
            + (radial * circle.Radius - expected_radial * surface.Radius).Length
            + (tangent * circle.Radius - expected_tangent * surface.Radius).Length)
        carrier_error = max(carrier_error, physical_error)
        sweep = rail.LastParameter - rail.FirstParameter
        start = phase if sign > 0. else (phase - sweep) % tau
        intervals.append((start, sweep))
        coordinate_scale = max(coordinate_scale, *(abs(v) for v in (circle.Center.x, circle.Center.y, circle.Center.z)))
        parameter_scale = max(parameter_scale, abs(rail.FirstParameter), abs(rail.LastParameter))
    epsilon = 2.220446049250313e-16
    operations = 128 + 32 * len(rails)
    gamma = operations * epsilon / (1. - operations * epsilon)
    numerical_error = gamma * (coordinate_scale + surface.Radius * (parameter_scale + tau + 1.))
    _require(_finite(numerical_error) and numerical_error < LINEAR_TOL_MM, "ILL_CONDITIONED_CIRCULAR_RAIL_SET")
    closure_error = surface.Radius * abs(span - tau)
    _require(closure_error <= numerical_error, "PARTIAL_HOLE_WALL_UNSUPPORTED")
    pieces = []
    for start, sweep in intervals:
        _require(numerical_error / surface.Radius < sweep <= tau + numerical_error / surface.Radius,
                 "DEGENERATE_OR_MULTITURN_CIRCULAR_RAIL")
        end = start + sweep
        pieces.extend([(start, tau), (0., end - tau)] if end > tau else [(start, end)])
    cursor, gap, overlap = 0., 0., 0.
    for start, stop in sorted(pieces):
        gap += max(0., start - cursor)
        overlap += max(0., min(stop, cursor) - start)
        cursor = max(cursor, stop)
    gap += max(0., tau - cursor)
    defects = surface.Radius * (gap + overlap)
    _require(defects <= numerical_error, "INCOMPLETE_OR_DUPLICATE_CIRCULAR_RAIL")
    return carrier_error + defects + numerical_error + closure_error


def _coverage(shape, graph, opposite_graph, pairs, plane_pairs, thickness, faces, edges):
    import Part
    selected, opposite = set(graph.nodes), set(opposite_graph.nodes)
    _require(not selected & opposite, "SKIN_LAYER_OVERLAP")
    counter = [0]
    owners = {}
    for index, edge in enumerate(edges):
        owners[index] = {_source_index(face, faces, counter) for face in shape.ancestorsOfType(edge, Part.Face)}
    native_edge_indices = {edge.hashCode(): i for i, edge in enumerate(edges)}
    adjacency = {index: set().union(*(owners[native_edge_indices[e.hashCode()]] for e in face.Edges))
                 for index, face in enumerate(faces)}
    explained, caps, holes = selected | opposite, set(), set()
    paired = dict(pairs)
    paired.update({value: key for key, value in pairs.items()})
    planar = dict(plane_pairs)
    planar.update({value: key for key, value in plane_pairs.items()})
    for layer, other_layer in ((selected, opposite), (opposite, selected)):
        for index in sorted(layer):
            face = faces[index]
            for rail in face.Edges:
                edge_index = native_edge_indices[rail.hashCode()]
                neighbors = owners[edge_index] - {index}
                _require(len(neighbors) == 1, "OPEN_OR_AMBIGUOUS_SKIN_BOUNDARY")
                neighbor_index = next(iter(neighbors))
                if neighbor_index in layer:
                    continue
                _require(neighbor_index not in other_layer, "SKINS_JOIN_WITHOUT_BOUNDARY_WALL")
                neighbor = faces[neighbor_index]
                if neighbor.Surface.TypeId == "Part::GeomPlane":
                    if face.Surface.TypeId == "Part::GeomCylinder":
                        _require(_parallel(neighbor.Surface.Axis, face.Surface.Axis)
                                 and paired[index] in adjacency[neighbor_index], "UNKNOWN_CYLINDRICAL_END_CAP")
                    else:
                        normal = _unit(face.Surface.Axis)
                        _require(abs(_unit(neighbor.Surface.Axis).dot(normal)) <= ANGULAR_TOL,
                                 "UNKNOWN_PLANAR_BOUNDARY_CAP")
                        counterpart = faces[planar[index]]
                        separation = (counterpart.Surface.Position - face.Surface.Position).dot(normal)
                        counterpart_edges = [e for e in counterpart.Edges
                            if neighbor_index in owners[native_edge_indices[e.hashCode()]]]
                        _require(any(_line_rails_translate(rail, edge, normal * separation) for edge in counterpart_edges),
                                 "UNKNOWN_PLANAR_BOUNDARY_CAP")
                    caps.add(neighbor_index)
                else:
                    _require(face.Surface.TypeId == "Part::GeomPlane", "UNKNOWN_CYLINDRICAL_BOUNDARY")
                    counterpart_index = planar[index]
                    _require(counterpart_index in adjacency[neighbor_index], "UNPAIRED_HOLE_WALL")
                    _require(_parallel(neighbor.Surface.Axis, face.Surface.Axis), "OBLIQUE_HOLE_WALL")
                    pr = neighbor.ParameterRange
                    _require(abs(pr[3] - pr[2] - thickness) <= LINEAR_TOL_MM, "HOLE_WALL_THICKNESS_MISMATCH")
                    circular_error = 0.
                    for plane_index in (index, counterpart_index):
                        rails = [e for e in faces[plane_index].Edges
                            if neighbor_index in owners[native_edge_indices[e.hashCode()]]]
                        circular_error += _circle_rails_cover_wall(rails, neighbor, faces[plane_index])
                    _require(circular_error <= LINEAR_TOL_MM, "HOLE_RAIL_CARRIER_MISMATCH")
                    normal = _unit(face.Surface.Axis)
                    axial_positions = [(v.Point - face.Surface.Position).dot(normal) for v in neighbor.Vertexes]
                    separation = (faces[counterpart_index].Surface.Position - face.Surface.Position).dot(normal)
                    sign = math.copysign(1., separation)
                    _require(min(v * sign for v in axial_positions) >= -LINEAR_TOL_MM
                             and max(v * sign for v in axial_positions) <= thickness + LINEAR_TOL_MM,
                             "HOLE_WALL_OUTSIDE_THICKNESS_SLAB")
                    holes.add(neighbor_index)
                explained.add(neighbor_index)
    _require(explained == set(range(len(faces))), "UNEXPLAINED_NATIVE_MATERIAL_SURFACE")
    return {"selected_source_face_indices": sorted(selected), "opposite_source_face_indices": sorted(opposite),
            "cap_source_face_indices": sorted(caps), "hole_wall_source_face_indices": sorted(holes),
            "all_native_faces_accounted": True, "all_native_bend_cylinders_paired": True,
            "tree_skin_graph": True, "thickness_matches_opposite_planes_and_cylinders": True}


def _strict_flat_face(edges, Part):
    _require(0 < len(edges) <= MAX_EDGES, "FLAT_EDGE_BUDGET_EXCEEDED")
    for edge in edges:
        _require(edge.Curve.TypeId in {"Part::GeomLine", "Part::GeomCircle"}, "UNSUPPORTED_FLAT_TRIM")
        for vertex in edge.Vertexes:
            _require(abs(vertex.Point.z) <= LINEAR_TOL_MM, "NONPLANAR_FLAT_EDGE")
        if edge.Curve.TypeId == "Part::GeomCircle":
            _require(abs(edge.Curve.Center.z) <= LINEAR_TOL_MM
                     and abs(abs(_unit(edge.Curve.Axis).z) - 1.) <= ANGULAR_TOL, "NONPLANAR_FLAT_CIRCLE")
    groups = Part.sortEdges(edges)
    wires = [Part.Wire(group) for group in groups]
    _require(all(wire.isClosed() and wire.isValid() for wire in wires), "OPEN_OR_INVALID_FLAT_WIRE")
    face = Part.makeFace(wires, "Part::FaceMakerBullseye")
    _require(face.isValid() and len(face.Faces) == 1 and _finite(face.Area)
             and face.Area > LINEAR_TOL_MM ** 2, "INVALID_OR_DISCONNECTED_FLAT_FACE")
    return face


def compensated_unfold(shape, root_face_index, thickness_mm, k_factor_ansi):
    """Return a native flat face and source-indexed bend ledger; always draft.

    No material/profile/unit approval is inferred. The positive finite thickness
    and explicit ANSI K-factor must come from the caller's approved profile.
    Exceptions produce no result; hooks/preferences/native input remain intact.
    """
    _require(isinstance(thickness_mm, (int, float)) and not isinstance(thickness_mm, bool) and _finite(thickness_mm)
             and float(thickness_mm) > LINEAR_TOL_MM, "EXPLICIT_POSITIVE_THICKNESS_REQUIRED")
    _require(isinstance(k_factor_ansi, (int, float)) and not isinstance(k_factor_ansi, bool) and _finite(k_factor_ansi)
             and 0. <= float(k_factor_ansi) <= 1., "EXPLICIT_ANSI_K_FACTOR_REQUIRED")
    thickness, k = float(thickness_mm), float(k_factor_ansi)
    _require(_LOCK.acquire(blocking=False), "SERIAL_UNFOLDER_PROCESS_REQUIRED")
    try:
        return _compensated_unfold_locked(shape, root_face_index, thickness, k)
    finally:
        _LOCK.release()


def _compensated_unfold_locked(shape, root, thickness, k):
    import FreeCAD
    import Part
    import SheetMetalNewUnfolder as upstream
    import networkx as nx
    version = list(FreeCAD.Version())
    _require(version[:3] == ["1", "1", "3"], "FREECAD_VERSION_PIN_MISMATCH")
    _require(len(version) >= 8 and version[7] == FREECAD_SOURCE_COMMIT,
             "FREECAD_SOURCE_COMMIT_PIN_MISMATCH")
    for module in (upstream, upstream.SheetMetalTools, upstream.SheetMetalText):
        source = Path(module.__file__)
        _require(source.name in UPSTREAM_SOURCE_HASHES
                 and hashlib.sha256(source.read_bytes()).hexdigest() == UPSTREAM_SOURCE_HASHES[source.name],
                 "UPSTREAM_SOURCE_PIN_MISMATCH")
    shape, normalization = _normalize_single_solid(shape, root)
    faces, edges = _native_preflight(shape, root)
    original_graph = upstream.build_graph_of_tangent_faces
    original_unroll = upstream.unroll_cylinder
    original_warning = FreeCAD.Console.PrintWarning
    counter, seen = [0], {}
    def refuse_warning(message):
        raise ManufacturingUnfoldUnavailable("UPSTREAM_WARNING: " + str(message)[:240])
    def checked_graph(native_shape, native_root):
        _require(native_shape.isSame(shape) and native_root == root, "UPSTREAM_GRAPH_INPUT_CHANGED")
        result = original_graph(native_shape, native_root)
        _require(set(result.nodes) == set(graph.nodes)
                 and {(min(a,b), max(a,b), d["label"]) for a,b,d in result.edges(data=True)}
                 == {(min(a,b), max(a,b), d["label"]) for a,b,d in graph.edges(data=True)}, "UPSTREAM_GRAPH_CHANGED")
        return result
    def checked_unroll(face, uvref, calculator, inferred, seam_edges):
        index = _source_index(face, faces, counter)
        _require(index in selected_cylinders and index not in seen, "UNEXPECTED_OR_DUPLICATE_UNROLLED_CYLINDER")
        _require(abs(inferred - thickness) <= LINEAR_TOL_MM, "UPSTREAM_THICKNESS_CHANGED")
        result = original_unroll(face, uvref, calculator, inferred, seam_edges)
        result[1].source_face_index = index
        seen[index] = result[1]
        return result
    FreeCAD.Console.PrintWarning = refuse_warning
    try:
        graph = original_graph(shape, root)
        selected_cylinders = _validate_tree(graph, faces, edges, root)
        # A native 3D circle/line can have a BSpline pcurve. The pinned engine's
        # BSpline2d reconstruction omits native basis data, so no such branch is
        # admitted by this exact-analytic draft adapter.
        for index in selected_cylinders:
            for edge in faces[index].Edges:
                pcurve, first, last = faces[index].curveOnSurface(edge)
                _require(isinstance(pcurve, (Part.Geom2d.Line2d, Part.Geom2d.Line2dSegment))
                         and _finite(first) and _finite(last) and last > first,
                         "UNSUPPORTED_SELECTED_CYLINDER_PCURVE")
        estimate = upstream.EstimateThickness.using_best_method(shape, root)
        _require(_finite(estimate) and abs(estimate - thickness) <= LINEAR_TOL_MM, "ESTIMATED_THICKNESS_MISMATCH")
        opposite_roots = [i for i, face in enumerate(faces) if i not in graph.nodes
            and face.Surface.TypeId == "Part::GeomPlane" and _translated_planes_match(faces[root], face, thickness)]
        _require(len(opposite_roots) == 1, "AMBIGUOUS_OPPOSITE_ROOT_PLANE")
        opposite_graph = original_graph(shape, opposite_roots[0])
        opposite_cylinders = _validate_tree(opposite_graph, faces, edges, opposite_roots[0])
        _require(not set(graph.nodes) & set(opposite_graph.nodes), "SKIN_LAYER_OVERLAP")
        pair_tests, pairs = 0, {}
        for index in sorted(selected_cylinders):
            matches = []
            for other in sorted(opposite_cylinders):
                pair_tests += 1
                _require(pair_tests <= MAX_PAIR_TESTS, "COUNTERPART_PROOF_BUDGET_EXCEEDED")
                if _cylinder_pair_matches(faces[index], faces[other], graph, opposite_graph, faces, edges, thickness):
                    matches.append(other)
            _require(len(matches) == 1, "UNPAIRED_OR_AMBIGUOUS_NATIVE_BEND")
            pairs[index] = matches[0]
        _require(set(pairs.values()) == opposite_cylinders and len(pairs) == len(opposite_cylinders),
                 "MISSING_OR_DUPLICATE_OPPOSITE_BEND")
        plane_pairs = {}
        for index in sorted(set(graph.nodes) - selected_cylinders):
            matches = []
            for other in sorted(set(opposite_graph.nodes) - opposite_cylinders):
                pair_tests += 1
                _require(pair_tests <= MAX_PAIR_TESTS, "COUNTERPART_PROOF_BUDGET_EXCEEDED")
                if _translated_planes_match(faces[index], faces[other], thickness):
                    matches.append(other)
            _require(len(matches) == 1, "UNPAIRED_OR_AMBIGUOUS_FLANGE")
            plane_pairs[index] = matches[0]
        _require(set(plane_pairs.values()) == set(opposite_graph.nodes) - opposite_cylinders,
                 "INCOMPLETE_OPPOSITE_FLANGE_LAYER")
        proof = _coverage(shape, graph, opposite_graph, pairs, plane_pairs, thickness, faces, edges)
        calculator = upstream.BendAllowanceCalculator.from_single_value(k, "ansi")
        upstream.build_graph_of_tangent_faces, upstream.unroll_cylinder = checked_graph, checked_unroll
        unfolded_edges, bend_infos = upstream.unfold(shape, root, calculator)
        _require(set(seen) == selected_cylinders and len(bend_infos) == len(selected_cylinders)
                 and {info.source_face_index for info in bend_infos} == selected_cylinders,
                 "MISSING_UNROLLED_CYLINDER")
        alignment = upstream.SketchExtraction.move_to_origin(Part.makeCompound(unfolded_edges), faces[root])
        flat = _strict_flat_face([edge.transformed(alignment) for edge in unfolded_edges], Part)
        # Independently retain every complete patch, including the cylinder's
        # two seam rails; upstream's outline omits these internal seams.
        tree = nx.bfs_tree(graph, root)
        transforms, patches = {}, []
        for index in nx.topological_sort(tree):
            ancestry = nx.shortest_path(tree, root, index)[:-1]
            total = FreeCAD.Matrix()
            for ancestor in ancestry:
                if ancestor in transforms:
                    total = total * transforms[ancestor]
            if index in selected_cylinders:
                parent = next(iter(tree.predecessors(index)))
                seam = edges[graph.get_edge_data(parent, index)["label"]]
                local, after, uvref = upstream.compute_unbend_transform(faces[index], seam, thickness, calculator)
                transforms[index] = after
                complete, _ = original_unroll(faces[index], uvref, calculator, thickness, set())
                patch_edges = [edge.transformed(alignment * total * local) for edge in complete]
                patch = _strict_flat_face(patch_edges, Part)
                neutral_radius = min(faces[index].Surface.Radius, faces[pairs[index]].Surface.Radius) + k * thickness
                native_area_oracle = faces[index].Area * neutral_radius / faces[index].Surface.Radius
                _require(abs(patch.Area - native_area_oracle) <= _area_tolerance(patch, faces[index]),
                         "CYLINDER_NEUTRAL_AREA_ORACLE_MISMATCH")
            else:
                patch = faces[index].transformed(alignment * total)
                _require(patch.isValid() and len(patch.Faces) == 1, "INVALID_FLATTENED_NATIVE_FLANGE")
            _require(abs(patch.BoundBox.ZMin) <= LINEAR_TOL_MM and abs(patch.BoundBox.ZMax) <= LINEAR_TOL_MM,
                     "NONPLANAR_DEVELOPED_PATCH")
            patches.append(patch)
        overlap_tests = 0
        for index, patch in enumerate(patches):
            _require(patch.cut(flat).Area <= _area_tolerance(patch, flat), "FLAT_OUTLINE_OMITS_NATIVE_PATCH")
            for other in patches[:index]:
                overlap_tests += 1
                _require(overlap_tests <= MAX_PAIR_TESTS, "PATCH_OVERLAP_BUDGET_EXCEEDED")
                _require(patch.common(other).Area <= _area_tolerance(patch, other), "DEVELOPED_PATCH_OVERLAP")
        _require(abs(sum(patch.Area for patch in patches) - flat.Area) <= _area_tolerance(flat, *patches),
                 "FLAT_AREA_OR_CUTOUT_COVERAGE_MISMATCH")
        independent_area = sum(faces[index].Area for index in set(graph.nodes) - selected_cylinders)
        independent_area += sum(faces[index].Area * (
            min(faces[index].Surface.Radius, faces[pairs[index]].Surface.Radius) + k * thickness)
            / faces[index].Surface.Radius for index in selected_cylinders)
        _require(abs(flat.Area - independent_area) <= _area_tolerance(flat, *patches),
                 "WHOLE_NEUTRAL_AREA_ORACLE_MISMATCH")
        ledger = []
        for info in bend_infos:
            index = info.source_face_index
            native_face = faces[index]
            direction = upstream.BendDirection.from_face(native_face)
            sweep = native_face.ParameterRange[1] - native_face.ParameterRange[0]
            radius = min(native_face.Surface.Radius, faces[pairs[index]].Surface.Radius)
            allowance = sweep * (radius + k * thickness)
            actual = calculator.get_bend_allowance(direction, native_face.Surface.Radius, thickness, sweep)
            _require(abs(actual - allowance) <= LINEAR_TOL_MM, "COMPENSATION_DIRECTION_MISMATCH")
            ledger.append({"source_face_index": index, "paired_source_face_index": pairs[index],
                "line": info.line.transformed(alignment),
                "bend_direction": "up" if direction == upstream.BendDirection.UP else "down",
                "sweep_rad": sweep, "included_angle_deg": float(info.angle),
                "inner_radius_mm": float(radius), "bend_allowance_mm": allowance,
                "thickness_mm": thickness, "k_factor_ansi": k})
        proof.update(normalization)
        proof.update({"root_source_face_index": root, "native_face_count": len(faces),
            "native_bend_count": len(ledger), "estimated_thickness_mm": float(estimate),
            "native_trim_types": ["Part::GeomLine", "Part::GeomCircle"],
            "no_cleanup_or_curve_approximation": True, "patch_overlap_checked": True,
            "native_patch_and_cutout_coverage_checked": True, "flat_face_area_mm2": float(flat.Area),
            "independent_neutral_area_mm2": float(independent_area),
            "independent_neutral_area_oracle_checked": True,
            "linear_numerical_tolerance_mm": LINEAR_TOL_MM,
            "area_numerical_tolerance_mm2": float(_area_tolerance(flat, *patches)),
            "geometry_checks_are_numerical_not_supplier_tolerance": True})
        return {"status": "DRAFT_COMPUTED", "flat_face": flat, "bend_ledger": ledger,
            "coverage_proof": proof, "upstream_commit": UPSTREAM_COMMIT,
            "freecad_source_commit": FREECAD_SOURCE_COMMIT, "freecad_version": version,
            "release_allowed": False, "manufacturing_ready": False,
            "source_sha256": None, "units": "FreeCAD_internal_mm", "units_verified": False}
    except ManufacturingUnfoldUnavailable:
        raise
    except Exception as error:
        raise ManufacturingUnfoldUnavailable("NATIVE_UNFOLDER_EXCEPTION: " + type(error).__name__) from error
    finally:
        upstream.build_graph_of_tangent_faces = original_graph
        upstream.unroll_cylinder = original_unroll
        FreeCAD.Console.PrintWarning = original_warning
