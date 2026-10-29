"""Opt-in OCP finite inner-cylinder features for conditional offline inspection.

Native geometry supplies nominal values only. The deliberately narrow contract
is a single bounded solid, rectangular cylindrical trim, and two natively shared
rectangular tangent flanges. This is not an exhaustive bend/event detector.
Source units and scan alignment remain declarations; no tolerance is invented.
OCCT adaptors already apply location; D1 does NOT apply face orientation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

if __package__:
    from .cad_guided_measurement import CADGuidedFeature
else:  # Existing backend-directory launcher convention, without backend retry.
    from cad_guided_measurement import CADGuidedFeature


GEOMETRY_TOL_MM = 2e-6
DIRECTION_TOL = 1e-7
MAX_FACES = 128
MAX_EDGE_USES = 1024
MAX_TREE_NODES = 4096
MAX_TREE_DEPTH = 16


@dataclass(frozen=True)
class NativeGuidedPolicy:
    """Search/fit engineering policy, separate from native CAD nominal geometry."""
    flange_window_mm: tuple[float, float]
    search_margin_mm: float
    max_fit_residual_mm: float
    plane_exclusion_mm: float
    max_tangency_error_mm: float
    fit_model: str = "measured_flange_tangent"

    def __post_init__(self):
        # Reuse the actual kernel admission policy without accepting CAD values
        # from the caller. Snapshot and canonicalize scalars/tuples as it does.
        check = CADGuidedFeature(
            feature_id="policy-check", nominal_radius_mm=5, nominal_sweep_deg=90,
            origin=(0, 0, 0), axis=(0, 0, 1), start_radial=(1, 0, 0),
            span_mm=(0, 10), nominal_source="POLICY_VALIDATION_ONLY", **asdict(self))
        for key in asdict(self):
            object.__setattr__(self, key, getattr(check, key))


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


def _unit(value):
    length = float(np.linalg.norm(value))
    _require(math.isfinite(length) and length > 1e-12, "DEGENERATE_NATIVE_DIRECTION")
    return value / length


def extract_ocp_guided_features(shape, *, policy, unit_status="UNVERIFIED",
                                allow_ocp=False, native_binding="ocp"):
    """Return features plus truthful extraction metadata; never repair geometry.

    No files are read here. The caller must pass the same native shape retained
    by its reader. IDs are local to this invocation, not persistent source IDs.
    Any unsupported inner cylindrical candidate invalidates the feature set;
    other noncylindrical geometry is not claimed to have been bend-classified.
    """
    metadata = dict(
        schema_version=1, scope="SINGLE_SOLID_RECTANGULAR_INNER_CYLINDERS",
        native_binding=native_binding if isinstance(native_binding, str) else None,
        unit_status=unit_status if isinstance(unit_status, str) else "UNVERIFIED",
        source_identity_status="IMPORT_LOCAL_NOT_PERSISTENT",
        frame_status="NATIVE_COORDINATES_NOT_SCAN_REGISTRATION",
        feature_identity_status="NATIVE_INSPECTION_FEATURE_NOT_MANUFACTURING_EVENT",
        exhaustive_bend_inventory=False, production_use_allowed=False,
        geometry_tolerance_mm=GEOMETRY_TOL_MM, features=[], reasons=[])

    def unavailable(reason):
        metadata["status"] = "UNAVAILABLE"
        metadata["reasons"].append(reason)
        metadata["features"] = []
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
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Line, GeomAbs_Circle
        from OCP.TopAbs import (TopAbs_SOLID, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX,
                               TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_OUT)
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Iterator
        from OCP.gp import gp_Pnt, gp_Vec

        _require(isinstance(shape, TopoDS_Shape) and not shape.IsNull(), "NATIVE_SHAPE_REQUIRED")
        _require(shape.ShapeType() == TopAbs_SOLID, "SINGLE_TOP_LEVEL_SOLID_REQUIRED")
        # Bound the topology BEFORE invoking whole-shape native validators.
        stack, nodes = [(shape, 0)], 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            _require(nodes <= MAX_TREE_NODES and depth <= MAX_TREE_DEPTH, "NATIVE_TREE_BUDGET")
            it = TopoDS_Iterator(current, True, True)
            while it.More():
                _require(nodes + len(stack) < MAX_TREE_NODES, "NATIVE_TREE_BUDGET")
                child = it.Value()
                stack.append((child.Oriented(child.Orientation()), depth + 1))
                it.Next()

        def explore(item, kind, cap):
            items, explorer = [], TopExp_Explorer(item, kind)
            while explorer.More():
                _require(len(items) < cap, "NATIVE_INCIDENCE_BUDGET")
                occurrence = explorer.Current()
                items.append(occurrence.Oriented(occurrence.Orientation()))
                explorer.Next()
            return items

        faces = [TopoDS.Face_s(f) for f in explore(shape, TopAbs_FACE, MAX_FACES)]
        _require(bool(faces), "NO_NATIVE_FACES")
        def tolerance_in_scope(item):
            tolerance = float(BRep_Tool.Tolerance_s(item))
            return math.isfinite(tolerance) and 0 <= tolerance <= GEOMETRY_TOL_MM

        for vertex in explore(shape, TopAbs_VERTEX, MAX_EDGE_USES * 2):
            _require(tolerance_in_scope(TopoDS.Vertex_s(vertex)), "NATIVE_TOLERANCE_OUTSIDE_SCOPE")
        for i, face in enumerate(faces):
            _require(not any(face.IsSame(other) for other in faces[:i]), "REPEATED_FACE_OCCURRENCE")
            _require(tolerance_in_scope(face), "NATIVE_TOLERANCE_OUTSIDE_SCOPE")
            _require(face.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED), "UNSUPPORTED_FACE_ORIENTATION")
        uses, unique_edges, face_edges = [], [], []
        for fi, face in enumerate(faces):
            indices = []
            for edge_shape in explore(face, TopAbs_EDGE, MAX_EDGE_USES):
                _require(len(uses) < MAX_EDGE_USES, "NATIVE_INCIDENCE_BUDGET")
                edge = TopoDS.Edge_s(edge_shape)
                _require(not BRep_Tool.Degenerated_s(edge), "DEGENERATE_EDGE_OUTSIDE_SCOPE")
                _require(tolerance_in_scope(edge),
                         "NATIVE_TOLERANCE_OUTSIDE_SCOPE")
                _require(BRep_Tool.SameParameter_s(edge) and BRep_Tool.SameRange_s(edge),
                         "NATIVE_CURVE_PARAMETER_CORRESPONDENCE_REQUIRED")
                matches = [j for j, other in enumerate(unique_edges) if edge.IsSame(other)]
                if not matches:
                    matches = [len(unique_edges)]
                    unique_edges.append(edge)
                indices.append(matches[0])
                uses.append((matches[0], fi, edge.Orientation()))
            face_edges.append(indices)
        for index in range(len(unique_edges)):
            owners = [(fi, orientation) for ei, fi, orientation in uses if ei == index]
            _require(len(owners) == 2 and owners[0][0] != owners[1][0]
                     and {orientation for _, orientation in owners} == {TopAbs_FORWARD, TopAbs_REVERSED},
                     "CLOSED_TWO_FACE_MANIFOLD_REQUIRED")
        _require(BRepCheck_Analyzer(shape, True, False, True).IsValid(), "INVALID_NATIVE_SOLID")
        classifier = BRepClass3d_SolidClassifier(shape)
        classifier.PerformInfinitePoint(GEOMETRY_TOL_MM)
        _require(classifier.State() == TopAbs_OUT, "BOUNDED_POSITIVE_MATERIAL_SOLID_REQUIRED")

        surfaces = [BRepAdaptor_Surface(face, True) for face in faces]

        def differential(index, u=None, v=None):
            surface = surfaces[index]
            if u is None:
                u = (surface.FirstUParameter() + surface.LastUParameter()) / 2
                v = (surface.FirstVParameter() + surface.LastVParameter()) / 2
            point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
            surface.D1(u, v, point, du, dv)
            p, x, y = _xyz(point), _xyz(du), _xyz(dv)
            n = _unit(np.cross(x, y))
            if faces[index].Orientation() == TopAbs_REVERSED:
                n = -n
            return p, x, y, n

        def four_edges(index):
            _require(len(explore(faces[index], TopAbs_WIRE, 2)) == 1,
                     "SINGLE_RECTANGULAR_WIRE_REQUIRED")
            ids = face_edges[index]
            _require(len(ids) == len(set(ids)) == 4, "FOUR_DISTINCT_TRIM_EDGES_REQUIRED")
            return [(ei, BRepAdaptor_Curve(unique_edges[ei])) for ei in ids]

        def endpoints(curve):
            first, last = curve.FirstParameter(), curve.LastParameter()
            _require(math.isfinite(first) and math.isfinite(last) and first < last,
                     "FINITE_NATIVE_EDGE_REQUIRED")
            return np.array([_xyz(curve.Value(first)), _xyz(curve.Value(last))])

        def matches_pair(points, expected):
            return min(np.max(np.linalg.norm(points - expected, axis=1)),
                       np.max(np.linalg.norm(points[::-1] - expected, axis=1))) <= GEOMETRY_TOL_MM

        def flange(index, contact, axis, radial, ray, span, shared):
            _require(surfaces[index].GetType() == GeomAbs_Plane, "PLANAR_ADJACENT_FLANGE_REQUIRED")
            _, _, _, normal = differential(index)
            _require(np.linalg.norm(normal + radial) <= DIRECTION_TOL, "SIGNED_INNER_TANGENCY_REQUIRED")
            edges = four_edges(index)
            points = []
            for _, curve in edges:
                _require(curve.GetType() == GeomAbs_Line, "RECTANGULAR_LINEAR_FLANGE_REQUIRED")
                points.extend(endpoints(curve))
            local = np.asarray(points) - contact
            z, t = local @ axis, local @ ray
            _require(np.max(np.abs(local @ radial)) <= GEOMETRY_TOL_MM, "FLANGE_PLANE_OFFSET_MISMATCH")
            length = float(t.max())
            _require(abs(float(t.min())) <= GEOMETRY_TOL_MM and length >= policy.flange_window_mm[1]
                     and abs(float(z.min()) - span[0]) <= GEOMETRY_TOL_MM
                     and abs(float(z.max()) - span[1]) <= GEOMETRY_TOL_MM,
                     "FINITE_FLANGE_WINDOW_NOT_CONTAINED")
            corners = np.array([contact + zz * axis + tt * ray
                                for zz, tt in ((span[0], 0), (span[1], 0),
                                               (span[1], length), (span[0], length))])
            expected = [corners[[j, (j + 1) % 4]] for j in range(4)]
            match_ids = []
            for ei, curve in edges:
                hits = [j for j, pair in enumerate(expected) if matches_pair(endpoints(curve), pair)]
                _require(len(hits) == 1, "NONRECTANGULAR_FLANGE_TRIM")
                match_ids.append(hits[0])
                if ei == shared:
                    _require(hits[0] == 0, "NATIVE_CONTACT_EDGE_MISMATCH")
            _require(len(set(match_ids)) == 4, "NONRECTANGULAR_FLANGE_TRIM")
            return length

        features = []
        for fi, surface in enumerate(surfaces):
            if surface.GetType() != GeomAbs_Cylinder:
                continue
            cylinder = surface.Cylinder()
            center = _xyz(cylinder.Location())
            axis = _unit(_xyz(cylinder.Axis().Direction()))
            midpoint, _, _, normal = differential(fi)
            radial_mid = _unit(midpoint - center - ((midpoint - center) @ axis) * axis)
            if normal @ radial_mid > 1 - DIRECTION_TOL:
                continue  # convex outer cylinder is not an inner feature
            _require(np.linalg.norm(normal + radial_mid) <= DIRECTION_TOL, "AMBIGUOUS_CYLINDER_ORIENTATION")
            u0, u1 = surface.FirstUParameter(), surface.LastUParameter()
            v0, v1 = surface.FirstVParameter(), surface.LastVParameter()
            _require(all(math.isfinite(x) for x in (u0, u1, v0, v1)) and v0 < v1,
                     "FINITE_CYLINDER_DOMAIN_REQUIRED")
            _require(20 <= math.degrees(u1 - u0) <= 160, "CYLINDER_SWEEP_OUTSIDE_SCOPE")
            p0, du, dv, _ = differential(fi, u0, v0)
            axis = _unit(dv)
            radius = float(cylinder.Radius())
            radial0 = _unit(p0 - center - ((p0 - center) @ axis) * axis)
            sign = 1 if du @ np.cross(axis, radial0) > 0 else -1
            sweep = sign * (u1 - u0)
            radial1 = math.cos(sweep) * radial0 + math.sin(sweep) * np.cross(axis, radial0)
            # World evaluations handle location scale; raw V is not a mm span.
            span = tuple(sorted(float((_xyz(surface.Value(u0, v)) - center) @ axis) for v in (v0, v1)))
            contacts = [center + radius * radial0, center + radius * radial1]
            expected_lines = [np.array([contact + z * axis for z in span]) for contact in contacts]
            generators, arc_ends = {}, []
            for ei, curve in four_edges(fi):
                if curve.GetType() == GeomAbs_Line:
                    hits = [j for j, expected in enumerate(expected_lines) if matches_pair(endpoints(curve), expected)]
                    _require(len(hits) == 1 and hits[0] not in generators, "INVALID_CYLINDER_GENERATOR")
                    generators[hits[0]] = ei
                elif curve.GetType() == GeomAbs_Circle:
                    circle = curve.Circle()
                    c = _xyz(circle.Location())
                    z = float((c - center) @ axis)
                    hits = [j for j, end in enumerate(span) if abs(z - end) <= GEOMETRY_TOL_MM]
                    _require(len(hits) == 1 and hits[0] not in arc_ends
                             and np.linalg.norm(c - center - z * axis) <= GEOMETRY_TOL_MM
                             and abs(float(circle.Radius()) - radius) <= GEOMETRY_TOL_MM
                             and abs(abs(_xyz(circle.Axis().Direction()) @ axis) - 1) <= DIRECTION_TOL
                             and abs((curve.LastParameter() - curve.FirstParameter()) - abs(sweep)) <= DIRECTION_TOL,
                             "INVALID_CYLINDER_END_ARC")
                    _require(matches_pair(endpoints(curve), np.array([c + radius * radial0, c + radius * radial1])),
                             "CYLINDER_END_ARC_ENDPOINT_MISMATCH")
                    mid = _xyz(curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2))
                    expected_mid = c + radius * (math.cos(sweep / 2) * radial0
                                                 + math.sin(sweep / 2) * np.cross(axis, radial0))
                    _require(np.linalg.norm(mid - expected_mid) <= GEOMETRY_TOL_MM, "CYLINDER_END_ARC_BRANCH_MISMATCH")
                    arc_ends.append(hits[0])
                else:
                    raise _Unsupported("UNSUPPORTED_CYLINDER_TRIM_CURVE")
            _require(len(generators) == len(arc_ends) == 2, "RECTANGULAR_CYLINDER_TRIM_REQUIRED")
            adjacent, lengths = [], []
            for j, radial in enumerate((radial0, radial1)):
                ei = generators[j]
                neighbors = [owner for edge_id, owner, _ in uses if edge_id == ei and owner != fi]
                _require(len(neighbors) == 1, "UNAMBIGUOUS_NATIVE_FLANGE_ADJACENCY_REQUIRED")
                adjacent.append(neighbors[0])
                ray = (-sign if j == 0 else sign) * np.cross(axis, radial)
                lengths.append(flange(neighbors[0], contacts[j], axis, radial, ray, span, ei))
            _require(adjacent[0] != adjacent[1], "TWO_DISTINCT_NATIVE_FLANGES_REQUIRED")
            _require(len(features) < 16, "FEATURE_BUDGET")
            feature = CADGuidedFeature(
                feature_id=f"solid:0:face:{fi}", nominal_radius_mm=radius,
                nominal_sweep_deg=math.degrees(sweep), origin=tuple(center), axis=tuple(axis),
                start_radial=tuple(radial0), span_mm=span,
                nominal_source=f"OCP_IMPORT_LOCAL:solid:0:face:{fi}", **asdict(policy))
            features.append(feature)
            metadata["features"].append(dict(
                feature_id=feature.feature_id, owner_id="solid:0", root_face_index=fi,
                flange_face_indices=adjacent, contact_edge_indices=[generators[0], generators[1]],
                finite_flange_lengths_mm=lengths, nominal_radius_mm=radius,
                nominal_sweep_deg=feature.nominal_sweep_deg, native_axial_span_mm=list(span)))
        _require(bool(features), "NO_SUPPORTED_INNER_CYLINDRICAL_FEATURE")
        metadata["native_face_count"] = len(faces)
        metadata["native_edge_count"] = len(unique_edges)
        metadata["status"] = "AVAILABLE"
        return dict(status="AVAILABLE", features=tuple(features), metadata=metadata)
    except _Unsupported as error:
        return unavailable(str(error))
    except Exception as error:
        # Native binding failures do not activate an alternate reader or model.
        return unavailable("NATIVE_EXTRACTION_ERROR:" + type(error).__name__)
