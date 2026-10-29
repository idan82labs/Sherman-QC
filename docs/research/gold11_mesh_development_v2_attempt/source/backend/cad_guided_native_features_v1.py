"""Bounded, source-preserving native transition ROIs, successor to frozen R0.

No reader, tessellation, repair, scan, registration or manufacturing truth enters
this module. Native topology identifies an inner cylindrical transition between
two tangent planes. A compatible outer cylinder is corroborating geometry, not
proof of a manufacturing event. Complete native boundary-curve enclosures and
an interior witness admit conservative finite search windows, including on
nonrectangular faces. A window is NOT a physical trim or an observed bend.

The enclosure argument is topological: a connected open rectangle with one
inside witness and no boundary crossing lies inside the face. Cylinder windows
are unwrapped in (radius * angle, axial distance); flange windows cover the
ENTIRE root-to-search strip (0, maximum search distance), not merely the points
later used for fitting. Bounds are conservative native floating-point bounds,
not interval-arithmetic or calibrated metrology certificates. Unsupported and
unresolved candidates survive in the inventory; unrelated bores do not erase
supported transitions. General bend/event inventory completeness is never claimed.

Analytic proof keeps a strict native tolerance gate. Larger finite trim-edge or
incident-vertex tolerances are exclusion-only: their complete edge tube and
vertex balls are enclosed before any strip/axial rejection. They never establish
a generator contact or bypass a root boundary. These are native representation
tolerances, not calibrated uncertainty about the manufactured part.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from .cad_guided_native_features import NativeGuidedPolicy
from .cad_guided_measurement import CADGuidedFeature, MAX_FEATURES as KERNEL_MAX_FEATURES


GEOMETRY_TOL_MM = 2e-6
DIRECTION_TOL = 1e-7
BOUNDARY_GUARD_MM = 4 * GEOMETRY_TOL_MM
MAX_FACES = 1024
MAX_TREE_NODES = 32768
MAX_TREE_DEPTH = 32
MAX_EDGE_USES = 16384
MAX_UNIQUE_EDGES = 8192
MAX_WINDOWS = 4096
MAX_CONTACT_PAIRS = 4096


class _Unsupported(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise _Unsupported(reason)


def _xyz(value):
    result = np.array([value.X(), value.Y(), value.Z()], dtype=float)
    _require(np.isfinite(result).all() and np.max(np.abs(result)) <= 1e6,
             "UNSUPPORTED_NUMERICAL_SCALE")
    return result


def _unit(vector):
    length = float(np.linalg.norm(vector))
    _require(math.isfinite(length) and length > 1e-12, "DEGENERATE_NATIVE_DIRECTION")
    return vector / length


def _union(intervals):
    """Closed interval union; never close a positive gap using a tolerance."""
    values = []
    for lo, hi in sorted(intervals):
        _require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi,
                 "INVALID_NATIVE_INTERVAL")
        if values and lo <= values[-1][1]:
            values[-1] = (values[-1][0], max(hi, values[-1][1]))
        else:
            values.append((float(lo), float(hi)))
    return values


def _intersect(first, second):
    first, second = _union(first), _union(second)
    result, i, j = [], 0, 0
    while i < len(first) and j < len(second):
        lo, hi = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if lo < hi:
            result.append((lo, hi))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return result


def _without_boundaries(intervals, forbidden, guard=BOUNDARY_GUARD_MM):
    """Remove complete axial enclosures, with a strict numerical separation."""
    _require(math.isfinite(guard) and guard > 0, "INVALID_BOUNDARY_GUARD")
    result = _union(intervals)
    for lo, hi in _union(forbidden):
        lo, hi = lo - guard, hi + guard
        next_result = []
        for start, stop in result:
            if stop <= lo or start >= hi:
                next_result.append((start, stop))
            else:
                if start < lo:
                    next_result.append((start, lo))
                if hi < stop:
                    next_result.append((hi, stop))
        result = next_result
    return [(lo, hi) for lo, hi in result if hi - lo > 2 * guard]


def _box_projection(bounds, origin, direction):
    """Support function of a world AABB; encloses every point, not samples."""
    bounds = np.asarray(bounds, dtype=float)
    origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    _require(bounds.shape == (6,) and np.isfinite(bounds).all()
             and origin.shape == direction.shape == (3,)
             and np.isfinite(origin).all() and np.isfinite(direction).all()
             and np.all(bounds[:3] <= bounds[3:]), "INVALID_NATIVE_BOUNDING_BOX")
    low, high = bounds[:3] - origin, bounds[3:] - origin
    return (float(np.minimum(low * direction, high * direction).sum()),
            float(np.maximum(low * direction, high * direction).sum()))


def _vector3(value):
    vector = np.asarray(value, dtype=float)
    _require(vector.shape == (3,) and np.isfinite(vector).all()
             and np.max(np.abs(vector)) <= 1e6, "UNSUPPORTED_NUMERICAL_SCALE")
    return vector


def _native_tolerance(value):
    """Numeric admission only; this does NOT relax the analytic proof gate."""
    _require(math.isfinite(value) and value >= 0, "INVALID_NATIVE_TOLERANCE")
    return float(value)


def _strict_contact_tolerances(edge_tolerance, vertex_tolerances):
    return (_native_tolerance(edge_tolerance) <= GEOMETRY_TOL_MM
            and all(_native_tolerance(value) <= GEOMETRY_TOL_MM for value in vertex_tolerances))


def _native_box(bounds):
    bounds = np.asarray(bounds, dtype=float)
    _require(bounds.shape == (6,) and np.isfinite(bounds).all()
             and np.all(bounds[:3] <= bounds[3:]), "INVALID_NATIVE_BOUNDING_BOX")
    _require(np.max(np.abs(bounds)) <= 1e6, "UNSUPPORTED_NUMERICAL_SCALE")
    return bounds.copy()


def _include_vertex_uncertainty(edge_bounds, vertex_data):
    """Union an already tolerance-inflated edge box with located vertex balls.

    OCCT 7.9 AddOptimal(edge, False, True) bounds the edge with its tolerance,
    but its separate vertex pass excludes vertices belonging to edges. A
    vertex's own position/tolerance therefore needs an explicit box union; a
    larger edge tolerance or curve-endpoint substitution is not equivalent.
    """
    bounds = _native_box(edge_bounds)
    for point, tolerance in vertex_data:
        point, tolerance = _vector3(point), _native_tolerance(tolerance)
        _require(tolerance <= 1e6, "UNSUPPORTED_NUMERICAL_SCALE")
        bounds[:3] = np.minimum(bounds[:3], point - tolerance)
        bounds[3:] = np.maximum(bounds[3:], point + tolerance)
    return _native_box(bounds)


def _plane_distance(point, plane_point, normal):
    """Euclidean distance to the underlying plane, not its UV projection."""
    point, plane_point, normal = map(_vector3, (point, plane_point, normal))
    return abs(float((point - plane_point) @ _unit(normal)))


def _cylinder_distance(point, center, axis, radius):
    """Euclidean distance to the infinite cylindrical surface, in millimetres."""
    point, center, axis = map(_vector3, (point, center, axis))
    _require(math.isfinite(radius) and 0 < radius <= 1e6, "UNSUPPORTED_NUMERICAL_SCALE")
    axis = _unit(axis)
    offset = point - center
    return abs(float(np.linalg.norm(offset - (offset @ axis) * axis)) - radius)


def _plane_rectangle_max_distance(origin, axis, ray, span, width, plane_point, normal):
    """Exact affine-distance supremum over the finite root-to-search rectangle.

    The absolute value of an affine signed plane distance is convex. On this
    rectangle its maximum is at a corner. This proves underlying-plane distance
    only: corners NEVER certify inclusion in a trimmed face (holes still need
    the complete-boundary exclusions and inside witness below).
    """
    origin, axis, ray = map(_vector3, (origin, axis, ray))
    _require(len(span) == 2 and all(math.isfinite(value) for value in span)
             and span[0] < span[1] and math.isfinite(width) and width > 0,
             "INVALID_NATIVE_RECTANGLE")
    return max(_plane_distance(origin + z * axis + t * ray, plane_point, normal)
               for z in span for t in (0., width))


def extract_ocp_guided_features_v1(shape, *, policy, unit_status="UNVERIFIED",
                                   allow_ocp=False, native_binding="ocp"):
    """Return native candidate inventory and bounded executable nominal ROIs.

    ``AVAILABLE`` means this adapter has executable ROIs and no unresolved
    partial-inner cylinder candidates; it is NOT complete bend detection.
    ``PARTIAL`` retains executable ROIs alongside explicit unresolved candidates.
    ``UNAVAILABLE`` has no executable ROI. ``batch_required`` forbids passing an
    oversized feature tuple directly to the existing bounded measurement kernel.
    Per-native-candidate grouping and every rejected candidate are retained.
    """
    metadata = dict(
        schema_version=1, adapter_version="NATIVE_TRANSITION_ROI_V1",
        scope="SINGLE_SOLID_GENERAL_TRIM_INNER_TRANSITION_WINDOWS",
        native_binding=native_binding if isinstance(native_binding, str) else None,
        unit_status=unit_status if isinstance(unit_status, str) else "UNVERIFIED",
        source_identity_status="IMPORT_LOCAL_NATIVE_TOPOLOGY_NOT_PERSISTENT",
        frame_status="NATIVE_COORDINATES_NOT_SCAN_REGISTRATION",
        feature_identity_status="NATIVE_TRANSITION_ROI_NOT_MANUFACTURING_EVENT",
        candidate_inventory_complete=False, exhaustive_bend_inventory=False,
        physical_observed_bend_count=None, production_use_allowed=False,
        boundary_inclusion_status="NATIVE_CURVE_ENCLOSURES_AND_INSIDE_WITNESS",
        physical_trim_endpoints="NOT_ESTABLISHED",
        geometry_tolerance_mm=GEOMETRY_TOL_MM, boundary_guard_mm=BOUNDARY_GUARD_MM,
        trim_uncertainty_policy="LOCAL_EDGE_TUBE_AND_INCIDENT_VERTEX_BALL_EXCLUSION_V1",
        native_tolerance_meaning="REPRESENTATION_TOLERANCE_NOT_CALIBRATED_METROLOGY",
        kernel_feature_limit=KERNEL_MAX_FEATURES, batch_required=False,
        candidates=[], features=[], reasons=[])

    def unavailable(reason):
        metadata["status"] = "UNAVAILABLE"
        metadata["reasons"].append(reason)
        metadata["features"] = []
        metadata["batch_required"] = False
        return dict(status="UNAVAILABLE", features=(), metadata=metadata)

    if (type(allow_ocp) is not bool or not allow_ocp
            or type(native_binding) is not str or native_binding != "ocp"):
        return unavailable("EXPLICIT_OCP_BINDING_REQUIRED")
    if type(unit_status) is not str or unit_status != "DECLARED_MM":
        return unavailable("EXPLICIT_NATIVE_MILLIMETRE_DECLARATION_REQUIRED")
    if not isinstance(policy, NativeGuidedPolicy):
        return unavailable("VALIDATED_SEARCH_POLICY_REQUIRED")

    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepClass import BRepClass_FaceClassifier
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.Bnd import Bnd_Box
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Line, GeomAbs_Circle
        from OCP.TopAbs import (TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_COMPSOLID,
                               TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
                               TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_OUT, TopAbs_IN)
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Iterator
        from OCP.gp import gp_Pnt, gp_Vec

        _require(isinstance(shape, TopoDS_Shape) and not shape.IsNull(), "NATIVE_SHAPE_REQUIRED")
        # Bound topology before any whole-shape native geometry/validation call.
        stack, nodes = [(shape, 0)], 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            _require(nodes <= MAX_TREE_NODES and depth <= MAX_TREE_DEPTH, "NATIVE_TREE_BUDGET")
            iterator = TopoDS_Iterator(current, True, True)
            while iterator.More():
                _require(nodes + len(stack) < MAX_TREE_NODES, "NATIVE_TREE_BUDGET")
                child = iterator.Value()
                stack.append((child.Oriented(child.Orientation()), depth + 1))
                iterator.Next()
        metadata["native_tree_nodes"] = nodes
        # STEP readers may retain a chain of one-child containers. Do not flatten
        # an assembly, discard loose geometry or merge distinct material owners.
        owner_path, solid = [], shape
        while solid.ShapeType() in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
            iterator = TopoDS_Iterator(solid, True, True)
            _require(iterator.More(), "SINGLE_SOLID_OWNER_REQUIRED")
            child = iterator.Value()
            child = child.Oriented(child.Orientation())
            iterator.Next()
            _require(not iterator.More(), "SINGLE_SOLID_OWNER_REQUIRED")
            solid, owner_path = child, owner_path + [0]
        _require(solid.ShapeType() == TopAbs_SOLID, "SINGLE_SOLID_OWNER_REQUIRED")
        metadata["owner_path"] = owner_path

        def explore(item, kind, cap):
            found, explorer = [], TopExp_Explorer(item, kind)
            while explorer.More():
                _require(len(found) < cap, "NATIVE_INCIDENCE_BUDGET")
                value = explorer.Current()
                found.append(value.Oriented(value.Orientation()))
                explorer.Next()
            return found

        faces = [TopoDS.Face_s(value) for value in explore(solid, TopAbs_FACE, MAX_FACES)]
        _require(bool(faces), "NO_NATIVE_FACES")
        face_map = TopTools_IndexedMapOfShape()

        vertex_map = TopTools_IndexedMapOfShape()
        vertices, vertex_tolerances = [], []
        for value in explore(solid, TopAbs_VERTEX, MAX_TREE_NODES):
            vertex = TopoDS.Vertex_s(value)
            if vertex_map.FindIndex(vertex) == 0:
                index = vertex_map.Add(vertex)
                _require(index == len(vertices) + 1, "NATIVE_IDENTITY_MAP_MISMATCH")
                vertices.append(vertex)
                vertex_tolerances.append(_native_tolerance(float(BRep_Tool.Tolerance_s(vertex))))
        face_tolerances = []
        for face in faces:
            _require(face_map.FindIndex(face) == 0, "REPEATED_FACE_OCCURRENCE")
            face_map.Add(face)
            face_tolerances.append(_native_tolerance(float(BRep_Tool.Tolerance_s(face))))
            _require(face.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED),
                     "UNSUPPORTED_FACE_ORIENTATION")
        edge_map = TopTools_IndexedMapOfShape()
        edges, edge_tolerances, edge_owners, face_edges, total_uses = [], [], [], [], 0
        for fi, face in enumerate(faces):
            indices = []
            for value in explore(face, TopAbs_EDGE, MAX_EDGE_USES):
                total_uses += 1
                _require(total_uses <= MAX_EDGE_USES, "NATIVE_INCIDENCE_BUDGET")
                edge = TopoDS.Edge_s(value)
                index = edge_map.FindIndex(edge)
                if index == 0:
                    _require(len(edges) < MAX_UNIQUE_EDGES, "NATIVE_EDGE_BUDGET")
                    index = edge_map.Add(edge)
                    _require(index == len(edges) + 1, "NATIVE_IDENTITY_MAP_MISMATCH")
                    edges.append(edge)
                    edge_tolerances.append(_native_tolerance(float(BRep_Tool.Tolerance_s(edge))))
                    edge_owners.append([])
                indices.append(index - 1)
                edge_owners[index - 1].append((fi, edge.Orientation()))
            face_edges.append(indices)
        edge_vertices, strict_edges, vertex_uses = [], [], 0
        for ei, edge in enumerate(edges):
            indices = set()
            for value in explore(edge, TopAbs_VERTEX, MAX_TREE_NODES):
                vertex_uses += 1
                _require(vertex_uses <= MAX_TREE_NODES, "NATIVE_INCIDENCE_BUDGET")
                index = vertex_map.FindIndex(value)
                _require(index > 0, "NATIVE_VERTEX_OWNERSHIP_REQUIRED")
                indices.add(index - 1)
            edge_vertices.append(sorted(indices))
            strict_edges.append(_strict_contact_tolerances(
                edge_tolerances[ei], [vertex_tolerances[vi] for vi in indices]))
        metadata.update(
            high_tolerance_face_indices=[fi for fi, value in enumerate(face_tolerances)
                                         if value > GEOMETRY_TOL_MM],
            high_tolerance_vertex_indices=[vi for vi, value in enumerate(vertex_tolerances)
                                           if value > GEOMETRY_TOL_MM],
            exclusion_only_edge_indices=[ei for ei, strict in enumerate(strict_edges) if not strict],
            native_tolerance_maxima_mm=dict(face=max(face_tolerances, default=0.),
                                           edge=max(edge_tolerances, default=0.),
                                           vertex=max(vertex_tolerances, default=0.)))
        seams, degenerate = [], []
        for ei, owners in enumerate(edge_owners):
            if BRep_Tool.Degenerated_s(edges[ei]):
                # Valid poles can occur on unrelated surfaces. They are never
                # admitted as transition/contact boundaries below.
                degenerate.append(ei)
                continue
            _require(len(owners) == 2 and {item[1] for item in owners}
                     == {TopAbs_FORWARD, TopAbs_REVERSED}, "CLOSED_ORIENTED_EDGE_USES_REQUIRED")
            if owners[0][0] == owners[1][0]:
                _require(BRep_Tool.IsClosed_s(edges[ei], faces[owners[0][0]]),
                         "UNCERTIFIED_REPEATED_FACE_EDGE")
                seams.append(ei)
        _require(BRepCheck_Analyzer(solid, True, False, True).IsValid(), "INVALID_NATIVE_SOLID")
        solid_classifier = BRepClass3d_SolidClassifier(solid)
        solid_classifier.PerformInfinitePoint(GEOMETRY_TOL_MM)
        _require(solid_classifier.State() == TopAbs_OUT, "BOUNDED_POSITIVE_MATERIAL_SOLID_REQUIRED")
        metadata.update(native_face_count=len(faces), native_edge_count=len(edges),
                        native_edge_use_count=total_uses, certified_seam_edge_indices=seams,
                        degenerate_edge_indices=degenerate)
        surfaces = [BRepAdaptor_Surface(face, True) for face in faces]
        curve_cache, bound_cache = {}, {}

        def curve_for(ei):
            if ei not in curve_cache:
                edge = edges[ei]
                _require(not BRep_Tool.Degenerated_s(edge), "DEGENERATE_CANDIDATE_BOUNDARY")
                _require(BRep_Tool.SameParameter_s(edge) and BRep_Tool.SameRange_s(edge),
                         "NATIVE_CURVE_PARAMETER_CORRESPONDENCE_REQUIRED")
                curve_cache[ei] = BRepAdaptor_Curve(edge)
            return curve_cache[ei]

        def endpoints(curve):
            first, last = float(curve.FirstParameter()), float(curve.LastParameter())
            _require(math.isfinite(first) and math.isfinite(last) and first < last,
                     "FINITE_NATIVE_EDGE_REQUIRED")
            return np.array([_xyz(curve.Value(first)), _xyz(curve.Value(last))])

        def differential(fi, u=None, v=None):
            surface = surfaces[fi]
            if u is None:
                u = (surface.FirstUParameter() + surface.LastUParameter()) / 2
                v = (surface.FirstVParameter() + surface.LastVParameter()) / 2
            _require(math.isfinite(u) and math.isfinite(v), "FINITE_NATIVE_SURFACE_DOMAIN_REQUIRED")
            p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
            surface.D1(u, v, p, du, dv)
            point, first, second = _xyz(p), _xyz(du), _xyz(dv)
            normal = _unit(np.cross(first, second))
            if faces[fi].Orientation() == TopAbs_REVERSED:
                normal = -normal
            return point, first, second, normal

        def edge_projection(ei, origin, direction):
            """Entire native curve enclosure, never an endpoint/sample proxy."""
            curve = curve_for(ei)
            if strict_edges[ei] and curve.GetType() == GeomAbs_Line:
                values = (endpoints(curve) - origin) @ direction
                return float(values.min()), float(values.max())
            if strict_edges[ei] and curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                center = _xyz(circle.Location())
                x = _xyz(circle.Position().XDirection())
                y = _xyz(circle.Position().YDirection())
                radius = float(circle.Radius())
                _require(math.isfinite(radius) and 0 < radius <= 1e6,
                         "UNSUPPORTED_NUMERICAL_SCALE")
                mid = float((center - origin) @ direction)
                amplitude = radius * math.hypot(float(x @ direction), float(y @ direction))
                return mid - amplitude, mid + amplitude  # full circle encloses any trimmed arc
            if ei not in bound_cache:
                _require(edge_tolerances[ei] <= 1e6, "UNSUPPORTED_NUMERICAL_SCALE")
                box = Bnd_Box()
                BRepBndLib.AddOptimal_s(edges[ei], box, False, True)
                _require(not box.IsVoid() and not box.IsWhole(), "FINITE_NATIVE_EDGE_BOUNDS_REQUIRED")
                bound_cache[ei] = _include_vertex_uncertainty(
                    box.Get(), [(_xyz(BRep_Tool.Pnt_s(vertices[vi])), vertex_tolerances[vi])
                                for vi in edge_vertices[ei]])
            return _box_projection(bound_cache[ei], origin, direction)

        def is_inside(fi, point):
            # OCCT's gp_Pnt classifier projects to UV without bounding the
            # projection distance. Check the actual underlying surface first.
            # See OCCT 7.9 BRepClass_FaceClassifier.cxx Perform(gp_Pnt).
            surface = surfaces[fi]
            _require(face_tolerances[fi] <= GEOMETRY_TOL_MM,
                     "ANALYTIC_FACE_TOLERANCE_OUTSIDE_SCOPE")
            if surface.GetType() == GeomAbs_Plane:
                plane = surface.Plane()
                distance = _plane_distance(point, _xyz(plane.Location()), _xyz(plane.Axis().Direction()))
            elif surface.GetType() == GeomAbs_Cylinder:
                cylinder = surface.Cylinder()
                distance = _cylinder_distance(point, _xyz(cylinder.Location()),
                                              _xyz(cylinder.Axis().Direction()), float(cylinder.Radius()))
            else:
                raise _Unsupported("UNSUPPORTED_WITNESS_SURFACE")
            if distance > GEOMETRY_TOL_MM:
                return False
            native_point = gp_Pnt(*map(float, point))
            return BRepClass_FaceClassifier(faces[fi], native_point, GEOMETRY_TOL_MM).State() == TopAbs_IN

        def cylinder_data(fi):
            _require(face_tolerances[fi] <= GEOMETRY_TOL_MM,
                     "ANALYTIC_CYLINDER_TOLERANCE_OUTSIDE_SCOPE")
            surface = surfaces[fi]
            u0, u1 = float(surface.FirstUParameter()), float(surface.LastUParameter())
            v0, v1 = float(surface.FirstVParameter()), float(surface.LastVParameter())
            _require(all(math.isfinite(value) for value in (u0, u1, v0, v1))
                     and u0 < u1 and v0 < v1, "FINITE_CYLINDER_DOMAIN_REQUIRED")
            cylinder = surface.Cylinder()
            center, radius = _xyz(cylinder.Location()), float(cylinder.Radius())
            _require(math.isfinite(radius) and 1e-8 <= radius <= 1e6,
                     "UNSUPPORTED_NUMERICAL_SCALE")
            p0, du, dv, _ = differential(fi, u0, v0)
            axis = _unit(dv)  # do not canonical-flip before evaluating native V
            radial0 = _unit(p0 - center - ((p0 - center) @ axis) * axis)
            sign = 1 if du @ np.cross(axis, radial0) > 0 else -1
            sweep = sign * (u1 - u0)
            radial1 = math.cos(sweep) * radial0 + math.sin(sweep) * np.cross(axis, radial0)
            mid, _, _, normal = differential(fi)
            radial_mid = _unit(mid - center - ((mid - center) @ axis) * axis)
            polarity = float(normal @ radial_mid)
            _require(abs(abs(polarity) - 1.) <= DIRECTION_TOL, "AMBIGUOUS_CYLINDER_ORIENTATION")
            span = tuple(sorted(float((_xyz(surface.Value(u0, value)) - center) @ axis)
                                for value in (v0, v1)))
            _require(span[1] - span[0] > 2 * BOUNDARY_GUARD_MM,
                     "CYLINDER_AXIAL_DOMAIN_TOO_SMALL")
            return dict(fi=fi, center=center, radius=radius, axis=axis, radial0=radial0,
                        radial1=radial1, radial_mid=radial_mid, sweep=sweep, sign=sign,
                        span=span, polarity=polarity)

        cylinders, unresolved = {}, 0
        for fi, surface in enumerate(surfaces):
            if surface.GetType() != GeomAbs_Cylinder:
                continue
            record = dict(candidate_id=f"solid:0:face:{fi}", owner_id="solid:0",
                          root_face_index=fi, classification="UNRESOLVED_CYLINDER",
                          reasons=[], window_feature_ids=[])
            metadata["candidates"].append(record)
            try:
                data = cylinder_data(fi)
                cylinders[fi] = data
                data["record"] = record
                record.update(nominal_radius_mm=data["radius"],
                              native_sweep_deg=math.degrees(data["sweep"]),
                              native_axial_span_mm=list(data["span"]),
                              native_center=data["center"].tolist(), native_axis=data["axis"].tolist())
                if data["polarity"] > 0:
                    record["classification"] = "OUTER_CONVEX_CYLINDRICAL_SKIN"
                elif abs(abs(data["sweep"]) - 2 * math.pi) <= DIRECTION_TOL:
                    record["classification"] = "FULL_WRAP_CONCAVE_CAVITY_OR_ROLLED_WALL"
                    record["reasons"].append("OUTSIDE_OPEN_FOLD_MODEL_NOT_MANUFACTURING_CLASSIFIED")
                else:
                    record["classification"] = "UNRESOLVED_PARTIAL_INNER_CYLINDER"
            except _Unsupported as error:
                record["reasons"].append(str(error))
                unresolved += 1
            except Exception as error:
                record["reasons"].append("NATIVE_CYLINDER_ERROR:" + type(error).__name__)
                unresolved += 1
        metadata["candidate_inventory_complete"] = True

        def generator_side(ei, data):
            if not strict_edges[ei]:
                return None  # Larger uncertainty can only remove support.
            curve = curve_for(ei)
            if curve.GetType() != GeomAbs_Line:
                return None
            points = endpoints(curve)
            local = points - data["center"]
            z = local @ data["axis"]
            transverse = local - z[:, None] * data["axis"]
            sides = [side for side, radial in enumerate((data["radial0"], data["radial1"]))
                     if np.max(np.linalg.norm(transverse - data["radius"] * radial, axis=1))
                     <= GEOMETRY_TOL_MM]
            if len(sides) == 1 and float(np.ptp(z)) > 2 * BOUNDARY_GUARD_MM:
                return sides[0], (float(z.min()), float(z.max()))
            return None

        def cylinder_boundaries(data):
            forbidden, generators = [], [[], []]
            for ei in sorted(set(face_edges[data["fi"]])):
                contact = generator_side(ei, data)
                if contact is not None:
                    side, interval = contact
                    generators[side].append((ei, interval))
                else:
                    forbidden.append(edge_projection(ei, data["center"], data["axis"]))
            _require(all(generators), "TWO_NATIVE_GENERATOR_CONTACTS_REQUIRED")
            return forbidden, generators

        def compatible_outer(inner, outer):
            if outer["polarity"] <= 0 or outer["radius"] <= inner["radius"] + GEOMETRY_TOL_MM:
                return False
            axis, offset = inner["axis"], outer["center"] - inner["center"]
            alignment_error = min(np.linalg.norm(axis - outer["axis"]),
                                  np.linalg.norm(axis + outer["axis"]))
            # A dimensionless angular tolerance alone is insufficient on long
            # parts: constrain the complete finite axis segment in millimetres.
            axis_points = [inner["center"] + z * axis for z in inner["span"]]
            line_errors = []
            for point in axis_points:
                delta = point - outer["center"]
                line_errors.append(np.linalg.norm(delta - (delta @ outer["axis"]) * outer["axis"]))
            angle_error = abs(abs(inner["sweep"]) - abs(outer["sweep"]))
            middle_error = np.linalg.norm(inner["radial_mid"] - outer["radial_mid"])
            if (alignment_error > DIRECTION_TOL or max(line_errors) > GEOMETRY_TOL_MM
                    or np.linalg.norm(offset - float(offset @ axis) * axis) > GEOMETRY_TOL_MM
                    or angle_error > DIRECTION_TOL or angle_error * outer["radius"] > GEOMETRY_TOL_MM
                    or middle_error > DIRECTION_TOL or middle_error * outer["radius"] > GEOMETRY_TOL_MM):
                return False
            end_error = min(max(np.linalg.norm(inner["radial0"] - outer["radial0"]),
                                np.linalg.norm(inner["radial1"] - outer["radial1"])),
                            max(np.linalg.norm(inner["radial0"] - outer["radial1"]),
                                np.linalg.norm(inner["radial1"] - outer["radial0"])))
            return end_error <= DIRECTION_TOL and end_error * outer["radius"] <= GEOMETRY_TOL_MM

        def flange_contacts(data, generators):
            grouped = [{}, {}]
            for side, contacts in enumerate(generators):
                radial = data["radial0"] if side == 0 else data["radial1"]
                contact_point = data["center"] + data["radius"] * radial
                for ei, interval in contacts:
                    owners = [fi for fi, _ in edge_owners[ei] if fi != data["fi"]]
                    _require(len(owners) == 1, "UNAMBIGUOUS_NATIVE_FLANGE_ADJACENCY_REQUIRED")
                    fi = owners[0]
                    _require(face_tolerances[fi] <= GEOMETRY_TOL_MM,
                             "ANALYTIC_FLANGE_TOLERANCE_OUTSIDE_SCOPE")
                    _require(surfaces[fi].GetType() == GeomAbs_Plane,
                             "PLANAR_ADJACENT_FLANGE_REQUIRED")
                    point, _, _, normal = differential(fi)
                    _require(np.linalg.norm(normal + radial) <= DIRECTION_TOL
                             and abs(float((point - contact_point) @ radial)) <= GEOMETRY_TOL_MM,
                             "SIGNED_INNER_TANGENCY_REQUIRED")
                    grouped[side].setdefault(fi, []).append((ei, interval))
            return grouped

        def flange_boundaries(fi, data, side):
            radial = data["radial0"] if side == 0 else data["radial1"]
            ray = (-data["sign"] if side == 0 else data["sign"]) * np.cross(data["axis"], radial)
            origin = data["center"] + data["radius"] * radial
            forbidden = []
            for ei in sorted(set(face_edges[fi])):
                curve = curve_for(ei)
                if strict_edges[ei] and curve.GetType() == GeomAbs_Line:
                    points = endpoints(curve)
                    if np.max(np.abs((points - origin) @ ray)) <= GEOMETRY_TOL_MM:
                        # Complete straight root boundary; it cannot cross the
                        # OPEN strip. Native contact intervals bound its reach.
                        continue
                low, high = edge_projection(ei, origin, ray)
                if high < -BOUNDARY_GUARD_MM or low > policy.flange_window_mm[1] + BOUNDARY_GUARD_MM:
                    continue
                forbidden.append(edge_projection(ei, data["center"], data["axis"]))
            return forbidden, origin, ray

        features = []
        for fi, data in cylinders.items():
            record = data["record"]
            if record["classification"] != "UNRESOLVED_PARTIAL_INNER_CYLINDER":
                continue
            try:
                _require(20 <= abs(math.degrees(data["sweep"])) <= 160,
                         "INNER_CYLINDER_SWEEP_OUTSIDE_MEASUREMENT_MODEL")
                outer_matches = [outer for outer in cylinders.values() if compatible_outer(data, outer)]
                # A separated coaxial cylinder is not a counterpart. Native
                # spans are evaluated physically before axes are compared.
                overlapping = []
                for outer in outer_matches:
                    positions = [outer["center"] + z * outer["axis"] for z in outer["span"]]
                    span = tuple(sorted(float((p - data["center"]) @ data["axis"]) for p in positions))
                    if _intersect([data["span"]], [span]):
                        overlapping.append((outer, span))
                _require(len(overlapping) == 1, "UNAMBIGUOUS_COAXIAL_OUTER_SKIN_REQUIRED")
                outer, outer_span = overlapping[0]
                forbidden, generators = cylinder_boundaries(data)
                outer_forbidden, _ = cylinder_boundaries(outer)
                # Convert complete outer boundary enclosures to the INNER axis
                # frame, including possible opposite native axis direction.
                outer_bad = []
                for low, high in outer_forbidden:
                    values = [float((outer["center"] + z * outer["axis"] - data["center"])
                                    @ data["axis"]) for z in (low, high)]
                    outer_bad.append((min(values), max(values)))
                contacts = flange_contacts(data, generators)
                _require(len(contacts[0]) * len(contacts[1]) <= MAX_CONTACT_PAIRS,
                         "NATIVE_CONTACT_PAIR_BUDGET")
                base = _intersect([data["span"]], [outer_span])
                base = [(lo + BOUNDARY_GUARD_MM, hi - BOUNDARY_GUARD_MM)
                        for lo, hi in base if hi - lo > 4 * BOUNDARY_GUARD_MM]
                base = _without_boundaries(base, forbidden + outer_bad)
                windows, strip_cache = [], {}
                for first, first_edges in sorted(contacts[0].items()):
                    for last, last_edges in sorted(contacts[1].items()):
                        _require(first != last, "TWO_DISTINCT_NATIVE_FLANGES_REQUIRED")
                        spans = _intersect(base, [interval for _, interval in first_edges])
                        spans = _intersect(spans, [interval for _, interval in last_edges])
                        # Exclude contact endpoints as well as all noncontact
                        # boundaries; numerical inclusion must not bridge gaps.
                        spans = [(lo + BOUNDARY_GUARD_MM, hi - BOUNDARY_GUARD_MM)
                                 for lo, hi in spans if hi - lo > 4 * BOUNDARY_GUARD_MM]
                        for side, flange in enumerate((first, last)):
                            if (flange, side) not in strip_cache:
                                strip_cache[flange, side] = flange_boundaries(flange, data, side)
                            spans = _without_boundaries(spans, strip_cache[flange, side][0])
                        for span in spans:
                            z = (span[0] + span[1]) / 2
                            inner_witness = data["center"] + z * data["axis"] + data["radius"] * data["radial_mid"]
                            outer_witness = data["center"] + z * data["axis"] + outer["radius"] * data["radial_mid"]
                            _require(is_inside(fi, inner_witness) and is_inside(outer["fi"], outer_witness),
                                     "CYLINDER_WINDOW_INSIDE_WITNESS_FAILED")
                            for side, flange in enumerate((first, last)):
                                _, origin, ray = strip_cache[flange, side]
                                plane = surfaces[flange].Plane()
                                _require(_plane_rectangle_max_distance(
                                    origin, data["axis"], ray, span, policy.flange_window_mm[1],
                                    _xyz(plane.Location()), _xyz(plane.Axis().Direction())) <= GEOMETRY_TOL_MM,
                                    "ROOT_TO_SEARCH_STRIP_OFF_UNDERLYING_PLANE")
                                witness = origin + z * data["axis"] + .5 * policy.flange_window_mm[1] * ray
                                _require(is_inside(flange, witness), "ROOT_TO_SEARCH_STRIP_INSIDE_WITNESS_FAILED")
                            _require(len(features) + len(windows) < MAX_WINDOWS, "NATIVE_WINDOW_BUDGET")
                            windows.append((span, first, last, first_edges, last_edges))
                _require(bool(windows), "NO_BOUNDARY_CERTIFIED_COMMON_AXIAL_WINDOW")
                # Overlapping windows from multiple adjacency assignments would
                # duplicate support. Keep the candidate unresolved, not greedy.
                ordered = sorted(windows, key=lambda item: (item[0], item[1], item[2]))
                _require(all(a[0][1] <= b[0][0] for a, b in zip(ordered, ordered[1:])),
                         "AMBIGUOUS_OVERLAPPING_NATIVE_WINDOWS")
                _require(len(features) + len(ordered) <= MAX_WINDOWS, "NATIVE_WINDOW_BUDGET")
                candidate_features, descriptions = [], []
                for wi, (span, first, last, first_edges, last_edges) in enumerate(ordered):
                    feature_id = f"{record['candidate_id']}:window:{wi}"
                    feature = CADGuidedFeature(
                        feature_id=feature_id, nominal_radius_mm=data["radius"],
                        nominal_sweep_deg=math.degrees(data["sweep"]), origin=tuple(data["center"]),
                        axis=tuple(data["axis"]), start_radial=tuple(data["radial0"]), span_mm=span,
                        nominal_source=f"OCP_IMPORT_LOCAL_V1:{record['candidate_id']}", **asdict(policy))
                    candidate_features.append(feature)
                    descriptions.append(dict(
                        feature_id=feature_id, candidate_id=record["candidate_id"], owner_id="solid:0",
                        root_face_index=fi, outer_skin_face_index=outer["fi"],
                        flange_face_indices=[first, last],
                        contact_edge_indices=[[ei for ei, _ in first_edges], [ei for ei, _ in last_edges]],
                        certified_search_span_mm=list(span),
                        root_to_search_strip_mm=[0., policy.flange_window_mm[1]],
                        nominal_radius_mm=data["radius"], nominal_sweep_deg=feature.nominal_sweep_deg,
                        physical_trim_endpoints=None, observed_bend_count=None))
                features.extend(candidate_features)
                metadata["features"].extend(descriptions)
                record.update(classification="SUPPORTED_INNER_TANGENT_TRANSITION_ROI",
                              outer_skin_face_index=outer["fi"],
                              skin_pair_status="GEOMETRIC_CORRESPONDENCE_NOT_MANUFACTURING_PROOF",
                              window_feature_ids=[feature.feature_id for feature in candidate_features],
                              native_full_trim_coverage="NOT_CLAIMED")
            except _Unsupported as error:
                record["reasons"].append(str(error))
                unresolved += 1
            except Exception as error:
                record["reasons"].append("NATIVE_CANDIDATE_ERROR:" + type(error).__name__)
                unresolved += 1
        metadata["unresolved_candidate_count"] = unresolved
        metadata["supported_native_candidate_count"] = sum(bool(item["window_feature_ids"])
                                                           for item in metadata["candidates"])
        metadata["expected_roi_count"] = len(features)
        metadata["native_complete_curve_box_edge_indices"] = sorted(bound_cache)
        metadata["native_trim_exclusion_records"] = [dict(
            edge_index=ei, edge_tolerance_mm=edge_tolerances[ei],
            vertex_indices=edge_vertices[ei],
            incident_vertex_tolerances_mm=[vertex_tolerances[vi] for vi in edge_vertices[ei]],
            world_enclosure_mm=bound_cache[ei].tolist())
            for ei in sorted(bound_cache) if not strict_edges[ei]]
        metadata["batch_required"] = len(features) > KERNEL_MAX_FEATURES
        metadata["batch_grouping"] = "NATIVE_CANDIDATE_THEN_WINDOW_ORDER_NO_TRUNCATION"
        if not features:
            return unavailable("NO_SUPPORTED_NATIVE_TRANSITION_ROI")
        status = "PARTIAL" if unresolved else "AVAILABLE"
        metadata["status"] = status
        return dict(status=status, features=tuple(features), metadata=metadata)
    except _Unsupported as error:
        return unavailable(str(error))
    except Exception as error:
        return unavailable("NATIVE_EXTRACTION_ERROR:" + type(error).__name__)
