"""Pure fake-native boundary checks; no FreeCAD import, CAD or geometry jobs.

These tests establish serialization/refusal contracts only. They are not native
kernel, physical fixture, compensated unfolding or manufacturing evidence.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "backend" / "vendor_manufacturing_serialization.py"
SPEC = importlib.util.spec_from_file_location("isolated_manufacturing_serialization", MODULE_PATH)
serializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serializer)
SOURCE_HASH, PROFILE_HASH = "a" * 64, "b" * 64


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Vertex:
    def __init__(self, point, identity=None):
        self.Point = Vector(*point)
        self.identity = identity if identity is not None else id(self)

    def isSame(self, other):
        return self.identity == other.identity


class Edge:
    ShapeType = "Edge"

    def __init__(self, start, end, *, orientation="Forward", identity=None):
        self.start, self.end = list(start), list(end)
        self.Orientation = orientation
        self.FirstParameter, self.LastParameter = 0.0, 1.0
        self.Curve = SimpleNamespace(TypeId="Part::GeomLine")
        self.Length = math.dist(self.start, self.end)
        self.Vertexes = [Vertex(start), Vertex(end)]
        self.identity = identity if identity is not None else id(self)
        self.common_length = self.Length
        self.outside_length = 0.0
        self.native_boolean_calls = []
        self.evaluation_delta_start = (0.0, 0.0, 0.0)
        self.evaluation_delta_end = (0.0, 0.0, 0.0)

    def valueAt(self, parameter):
        return Vector(*(a + parameter * (b - a) + da + parameter * (db - da)
                        for a, b, da, db in zip(self.start, self.end,
                                               self.evaluation_delta_start, self.evaluation_delta_end)))

    def isValid(self):
        return True

    def isClosed(self):
        return False

    def hashCode(self):
        return self.identity

    def isSame(self, other):
        return self.identity == other.identity

    def common(self, face):
        self.native_boolean_calls.append(("common", face))
        return SimpleNamespace(Length=self.common_length)

    def cut(self, face):
        self.native_boolean_calls.append(("cut", face))
        return SimpleNamespace(Length=self.outside_length)


class CircleCurve:
    TypeId = "Part::GeomCircle"

    def __init__(self, center, radius, axis_sign):
        self.Center, self.Radius, self.Axis = Vector(*center), radius, Vector(0, 0, axis_sign)

    def isPeriodic(self):
        return True


class CircleEdge(Edge):
    def __init__(self, first=0.0, last=2 * math.pi, *, closed=True, axis_sign=1,
                 orientation="Forward", radius=1.0, center=(4.0, 4.0, 0.0)):
        super().__init__((0, 0, 0), (1, 0, 0), orientation=orientation)
        self.FirstParameter, self.LastParameter = first, last
        self.Curve = CircleCurve(center, radius, axis_sign)
        self.Length = radius * (last - first)
        self.closed, self.axis_sign = closed, axis_sign
        self.Vertexes = [Vertex((0, 0, 0))] if closed else [Vertex((0, 0, 0)), Vertex((1, 1, 0))]

    def valueAt(self, parameter):
        c = self.Curve.Center
        return Vector(c.x + self.Curve.Radius * math.cos(parameter),
                      c.y + self.axis_sign * self.Curve.Radius * math.sin(parameter), c.z)

    def isClosed(self):
        return self.closed


class Wire:
    def __init__(self, edges):
        self.Edges, self.OrderedEdges = list(edges), list(edges)
        self.closed, self.identity = True, id(self)
        # This constructs an exact fake-native topology only when coordinates
        # are identical. It never glues distinct/near points or repairs a gap.
        native_vertices = {}
        for edge in self.Edges:
            for index, vertex in enumerate(edge.Vertexes):
                point = vertex.Point
                key = (point.x, point.y, point.z)
                if key in native_vertices:
                    edge.Vertexes[index] = native_vertices[key]
                else:
                    native_vertices[key] = vertex

    def isSame(self, other):
        return self.identity == other.identity

    def isValid(self):
        return True

    def isClosed(self):
        return self.closed


class FlatFace:
    ShapeType = "Face"

    def __init__(self):
        points = [(0, 0, 0), (10, 0, 0), (10, 20, 0), (0, 20, 0)]
        self.OuterWire = Wire([Edge(a, b) for a, b in zip(points, points[1:] + points[:1])])
        self.Wires = [self.OuterWire]
        self.Faces = [self]
        self.Area = 200.0
        self.Surface = SimpleNamespace(TypeId="Part::GeomPlane", Axis=Vector(0, 0, 1))
        self.BoundBox = SimpleNamespace(XMin=0.0, XMax=10.0, YMin=0.0, YMax=20.0, ZMin=0.0, ZMax=0.0)

    @property
    def Edges(self):
        return [edge for wire in self.Wires for edge in wire.Edges]

    def isValid(self):
        return True


class CylinderFace:
    def __init__(self, radius, axis_sign=1):
        self.Surface = SimpleNamespace(TypeId="Part::GeomCylinder", Center=Vector(),
                                       Axis=Vector(0, 0, axis_sign), Radius=radius)
        self.ParameterRange = (0.0, math.pi / 2, 0.0, 20.0)
        self.axis_sign = axis_sign

    def valueAt(self, u, v):
        return Vector(self.Surface.Radius * math.cos(u),
                      self.axis_sign * self.Surface.Radius * math.sin(u), self.axis_sign * v)


class SourceShape:
    ShapeType = "Solid"

    def __init__(self):
        plane = lambda: SimpleNamespace(Surface=SimpleNamespace(TypeId="Part::GeomPlane"))
        self.Faces = [plane(), CylinderFace(2.0), plane(), CylinderFace(3.0)]
        self.Solids = [self]
        self.Vertexes = [Vertex((0, 0, 0)), Vertex((3, 0, 20))]

    def isValid(self):
        return True


def fixture():
    shape, flat = SourceShape(), FlatFace()
    proof = {name: True for name in serializer.REQUIRED_PROOFS}
    proof.update({"native_face_count": 4, "native_bend_count": 1, "root_source_face_index": 0,
                  "selected_source_face_indices": [0, 1], "opposite_source_face_indices": [2, 3],
                  "cap_source_face_indices": [], "hole_wall_source_face_indices": [],
                  "flat_face_area_mm2": 200.0, "independent_neutral_area_mm2": 200.0,
                  "linear_numerical_tolerance_mm": 1e-7, "area_numerical_tolerance_mm2": 1e-5})
    ledger = [{"source_face_index": 1, "paired_source_face_index": 3,
               "line": Edge((5, 0, 0), (5, 20, 0)), "bend_direction": "up",
               "sweep_rad": math.pi / 2, "included_angle_deg": 90.0,
               "inner_radius_mm": 2.0, "thickness_mm": 1.0, "k_factor_ansi": 0.4,
               "bend_allowance_mm": math.pi / 2 * 2.4}]
    result = {"status": "DRAFT_COMPUTED", "flat_face": flat, "bend_ledger": ledger,
              "coverage_proof": proof, "release_allowed": False, "manufacturing_ready": False,
              "units_verified": False, "source_sha256": None, "units": "FreeCAD_internal_mm",
              "upstream_commit": "fake-only", "freecad_source_commit": "fake-only",
              "freecad_version": ["fake", "not-native"]}
    return shape, result


class SerializationContractTests(unittest.TestCase):
    def serialize(self, shape=None, result=None):
        if shape is None or result is None:
            shape, result = fixture()
        return serializer.serialize_compensated_draft(shape, result, SOURCE_HASH, PROFILE_HASH)

    def refuses(self, mutate, reason=None):
        shape, result = fixture()
        mutate(shape, result)
        with self.assertRaises(serializer.ManufacturingSerializationUnavailable) as caught:
            self.serialize(shape, result)
        if reason:
            self.assertIn(reason, str(caught.exception))

    def test_pure_json_draft_rows_and_complete_face_provenance(self):
        output = self.serialize()
        self.assertEqual(json.loads(json.dumps(output, allow_nan=False)), output)
        self.assertEqual(len(output["cut_rows"]), 4)
        self.assertEqual({row["layer"] for row in output["cut_rows"]}, {"CUT_OUTER"})
        self.assertEqual(output["bend_rows"], [{"type": "LINE", "layer": "BEND_UP", "start": [5.0, 0.0], "end": [5.0, 20.0]}])
        self.assertEqual(sorted(row["source_face_index"] for row in output["source_face_coverage"]), [0, 1, 2, 3])
        self.assertEqual(output["units"], "internal_mm")
        for key in ("release_allowed", "manufacturing_ready", "units_verified"):
            self.assertIs(output[key], False)
        self.assertEqual(output["vendor_feature_id_mapping"], "UNMAPPED")
        self.assertIsNone(output["visual_mesh"])
        for row in output["cut_rows"] + output["bend_rows"]:
            self.assertNotIn("source_face_index", row)
            self.assertNotIn("wire_index", row)

    def test_folded_marker_is_native_mid_skin_not_k_location(self):
        output = self.serialize()
        marker = output["folded_bend_markers"][0]
        self.assertEqual(marker["pair_id"], "native_face_pair:1:3")
        self.assertEqual(marker["source_face_index"], 1)
        self.assertEqual(marker["paired_source_face_index"], 3)
        self.assertAlmostEqual(marker["line_start"][0], 2.5 / math.sqrt(2))
        self.assertAlmostEqual(marker["line_start"][1], 2.5 / math.sqrt(2))
        self.assertEqual(marker["line_start"][2], 0.0)
        self.assertEqual(marker["line_end"][2], 20.0)
        self.assertIs(marker["compensated_k_location"], False)
        shape, result = fixture()
        result["bend_ledger"][0]["k_factor_ansi"] = 0.7
        result["bend_ledger"][0]["bend_allowance_mm"] = math.pi / 2 * 2.7
        self.assertEqual(self.serialize(shape, result)["folded_bend_markers"], output["folded_bend_markers"])

    def test_signed_down_included_angle_is_preserved(self):
        shape, result = fixture()
        result["bend_ledger"][0].update({"bend_direction": "down", "included_angle_deg": -90.0})
        output = self.serialize(shape, result)
        self.assertEqual(output["bend_rows"][0]["layer"], "BEND_DOWN")
        provenance = output["bend_row_provenance"][0]
        self.assertEqual(provenance["included_angle_deg"], -90.0)
        self.assertEqual(provenance["included_angle_abs_deg"], 90.0)
        self.assertEqual(provenance["included_angle_convention"], "UPSTREAM_SIGNED_INCLUDED_ANGLE_DEG")
        for direction, angle in (("up", -90.0), ("down", 90.0), ("up", 89.0), ("down", -91.0)):
            with self.subTest(direction=direction, angle=angle):
                self.refuses(lambda _, result: result["bend_ledger"][0].update(
                    bend_direction=direction, included_angle_deg=angle), "SIGNED_INCLUDED_ANGLE")

    def test_native_circle_certificate_not_endpoint_gap(self):
        shape, result = fixture()
        result["flat_face"].Wires.append(Wire([CircleEdge()]))
        output = self.serialize(shape, result)
        circle = output["cut_rows"][-1]
        self.assertEqual(circle, {"type": "CIRCLE", "layer": "CUT_INNER", "center": [4.0, 4.0],
                                  "radius_mm": 1.0, "sweep_rad": 2 * math.pi, "full_circle_verified": True})
        certificate = output["cut_row_provenance"][-1]["native_circle_certificate"]
        self.assertTrue(certificate["native_closed_vertex_identity"])
        self.assertFalse(certificate["endpoint_gap_heuristic_used"])
        self.assertLessEqual(certificate["total_physical_error_bound_mm"], 1e-7)
        for mutate in (
            lambda edge: setattr(edge, "closed", False),
            lambda edge: setattr(edge, "Vertexes", [Vertex((0, 0, 0)), Vertex((0, 0, 0))]),
            lambda edge: setattr(edge, "LastParameter", 2 * math.pi - 1e-6),
        ):
            edge = CircleEdge()
            mutate(edge)
            edge.Length = edge.Curve.Radius * (edge.LastParameter - edge.FirstParameter)
            with self.assertRaises(serializer.ManufacturingSerializationUnavailable):
                serializer._analytic_edge(edge, "CUT_INNER")

    def test_circle_total_physical_error_is_not_relaxed_by_large_parameter_origin(self):
        # A native-closed certificate is necessary but cannot waive the same
        # 1e-7 mm total numerical budget used by all other geometry.
        edge = CircleEdge(first=100_000.0, last=100_000.0 + 2 * math.pi, radius=100_000.0)
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "PHYSICAL_ERROR_BUDGET"):
            serializer._analytic_edge(edge, "CUT_INNER")
        edge = CircleEdge(radius=1.0)
        edge.Length += 1e-7
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "PHYSICAL_ERROR_BUDGET"):
            serializer._analytic_edge(edge, "CUT_INNER")

    def test_partial_arc_mirror_and_reverse_signed_sweep(self):
        for axis_sign in (1, -1):
            for orientation in ("Forward", "Reversed"):
                with self.subTest(axis=axis_sign, orientation=orientation):
                    edge = CircleEdge(first=0.2, last=4.4, closed=False,
                                      axis_sign=axis_sign, orientation=orientation)
                    row, metadata = serializer._analytic_edge(edge, "CUT_INNER")
                    expected_sign = axis_sign * (1 if orientation == "Forward" else -1)
                    self.assertEqual(row["type"], "ARC")
                    self.assertAlmostEqual(row["sweep_rad"], expected_sign * 4.2)
                    parameter = edge.FirstParameter if orientation == "Forward" else edge.LastParameter
                    endpoint = edge.valueAt(parameter)
                    self.assertAlmostEqual(row["center"][0] + row["radius_mm"] * math.cos(row["start_angle_rad"]), endpoint.x)
                    self.assertAlmostEqual(row["center"][1] + row["radius_mm"] * math.sin(row["start_angle_rad"]), endpoint.y)
                    opposite = edge.valueAt(edge.LastParameter if orientation == "Forward" else edge.FirstParameter)
                    angle = row["start_angle_rad"] + row["sweep_rad"]
                    self.assertAlmostEqual(row["center"][0] + row["radius_mm"] * math.cos(angle), opposite.x)
                    self.assertAlmostEqual(row["center"][1] + row["radius_mm"] * math.sin(angle), opposite.y)
                    self.assertFalse(metadata["native_circle_certificate"]["native_full_period_parameters"])

    def test_near_period_partial_arc_is_not_promoted_to_circle(self):
        edge = CircleEdge(last=2 * math.pi - 1e-6, closed=False)
        row, _ = serializer._analytic_edge(edge, "CUT_INNER")
        self.assertEqual(row["type"], "ARC")
        self.assertLess(row["sweep_rad"], 2 * math.pi)

    def test_cut_line_uses_exact_shared_native_vertices_not_parameter_drift(self):
        shape, result = fixture()
        edge = result["flat_face"].OuterWire.Edges[0]
        edge.evaluation_delta_start = (2e-9, 3e-9, 0.0)
        edge.evaluation_delta_end = (2e-9, 3e-9, 0.0)
        output = self.serialize(shape, result)
        self.assertEqual(output["cut_rows"][0]["start"], [0.0, 0.0])
        self.assertEqual(output["cut_rows"][0]["end"], [10.0, 0.0])
        certificate = output["cut_row_provenance"][0]["native_line_endpoint_certificate"]
        self.assertTrue(certificate["exact_native_vertex_xy_used"])
        self.assertFalse(certificate["proximity_snapping_used"])
        self.assertLessEqual(certificate["total_physical_error_bound_mm"], 1e-7)
        self.assertGreater(certificate["maximum_projected_endpoint_to_native_trim_deviation_mm"], 0)
        self.assertTrue(all(row["exact_native_wire_vertex_identity_closure"] for row in output["cut_row_provenance"]))
        # Native vertex list order is not an orientation convention.
        edge.Vertexes.reverse()
        reordered = self.serialize(shape, result)
        self.assertEqual(reordered["cut_rows"], output["cut_rows"])
        self.assertEqual(reordered["cut_row_provenance"][0]["native_line_endpoint_certificate"]
                         ["native_vertex_indices_in_parameter_traversal"], [1, 0])

    def test_cut_line_native_vertex_drift_projection_and_lengths_share_one_budget(self):
        edge = Edge((0, 0, 0), (10, 0, 0))
        edge.Vertexes = [Vertex((0, 0, 8e-8)), Vertex((10, 0, 8e-8))]
        row, metadata = serializer._analytic_edge(edge, "CUT_OUTER")
        certificate = metadata["native_line_endpoint_certificate"]
        self.assertEqual(row["start"], [0.0, 0.0])
        self.assertEqual(certificate["maximum_projected_endpoint_to_native_trim_deviation_mm"], 0.0)
        self.assertEqual(certificate["maximum_native_vertex_projection_deviation_mm"], 8e-8)
        self.assertEqual(certificate["total_physical_error_bound_mm"], 8e-8)
        # The source-vertex projection cannot receive an independent allowance
        # from either native length discrepancy.
        edge.Length += 2e-8
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "PHYSICAL_ERROR_BUDGET"):
            serializer._analytic_edge(edge, "CUT_OUTER")
        edge = Edge((0, 0, 8e-8), (10, 0, 8e-8))
        edge.evaluation_delta_start = (7e-8, 0.0, 0.0)
        edge.evaluation_delta_end = (7e-8, 0.0, 0.0)
        # Raw vertex-to-trim distance and |z| separately pass, but the actual
        # exported XY endpoint-to-3D-trim norm exceeds the total physical budget.
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "PHYSICAL_ERROR_BUDGET"):
            serializer._analytic_edge(edge, "CUT_OUTER")
        edge = Edge((0, 0, 0), (10, 0, 0))
        edge.evaluation_delta_start = edge.evaluation_delta_end = (0.0, 3e-8, 0.0)
        edge.Length += 6e-8
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "PHYSICAL_ERROR_BUDGET"):
            serializer._analytic_edge(edge, "CUT_OUTER")
        edge = Edge((0, 0, 0), (10, 0, 0))
        edge.evaluation_delta_start = edge.evaluation_delta_end = (0.0, 2e-7, 0.0)
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "VERTEX_CORRESPONDENCE"):
            serializer._analytic_edge(edge, "CUT_OUTER")

    def test_cut_line_ambiguous_short_edge_is_not_nearest_matched(self):
        edge = Edge((-6e-8, 0, 0), (6e-8, 0, 0))
        edge.Vertexes = [Vertex((0, -6e-8, 0)), Vertex((0, 6e-8, 0))]
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "AMBIGUOUS"):
            serializer._analytic_edge(edge, "CUT_OUTER")

    def test_cut_wire_true_gap_coincident_distinct_identity_and_wrong_orientation_refuse(self):
        for gap in (0.0, 5e-8):
            shape, result = fixture()
            next_edge = result["flat_face"].OuterWire.Edges[1]
            native = next_edge.Vertexes[0].Point
            next_edge.Vertexes[0] = Vertex((native.x + gap, native.y, native.z))
            with self.subTest(gap=gap), self.assertRaisesRegex(
                    serializer.ManufacturingSerializationUnavailable, "NATIVE_IDENTITY_GAP"):
                self.serialize(shape, result)
        self.refuses(lambda _, result: setattr(result["flat_face"].OuterWire.Edges[0],
                                               "Orientation", "Reversed"), "NATIVE_IDENTITY_GAP")

    def test_claimed_shared_native_identity_with_inconsistent_point_refuses(self):
        shape, result = fixture()
        next_edge = result["flat_face"].OuterWire.Edges[1]
        shared = next_edge.Vertexes[0]
        next_edge.Vertexes[0] = Vertex((shared.Point.x + 2e-9, shared.Point.y, shared.Point.z),
                                     identity=shared.identity)
        with self.assertRaisesRegex(serializer.ManufacturingSerializationUnavailable, "INCONSISTENT_SHARED_NATIVE_VERTEX"):
            self.serialize(shape, result)

    def test_flat_line_whole_edge_containment_and_no_trimming(self):
        shape, result = fixture()
        line = result["bend_ledger"][0]["line"]
        before = copy.deepcopy((line.start, line.end))
        output = self.serialize(shape, result)
        self.assertEqual((line.start, line.end), before)
        self.assertEqual([name for name, _ in line.native_boolean_calls], ["common", "cut"])
        self.assertTrue(output["bend_row_provenance"][0]["whole_native_line_inside_cut_face"])
        self.assertFalse(output["bend_row_provenance"][0]["line_was_trimmed"])
        self.refuses(lambda _, result: setattr(result["bend_ledger"][0]["line"], "outside_length", 1.0), "OVERHANG")
        self.refuses(lambda _, result: setattr(result["bend_ledger"][0]["line"], "common_length", 19.0), "OVERHANG")

    def test_hash_syntax_and_source_binding(self):
        shape, result = fixture()
        for value in (None, "", "x" * 64, "A" * 64, "a" * 63, "a" * 64 + "\n"):
            with self.subTest(hash=value), self.assertRaises(serializer.ManufacturingSerializationUnavailable):
                serializer.serialize_compensated_draft(shape, result, value, PROFILE_HASH)
        with self.assertRaises(serializer.ManufacturingSerializationUnavailable):
            serializer.serialize_compensated_draft(shape, result, SOURCE_HASH, "bad")
        result["source_sha256"] = "c" * 64
        with self.assertRaises(serializer.ManufacturingSerializationUnavailable):
            self.serialize(shape, result)

    def test_complete_unique_native_source_face_mapping(self):
        changes = (
            lambda proof: proof.update(native_face_count=3),
            lambda proof: proof.update(selected_source_face_indices=[0, 1, 1]),
            lambda proof: proof.update(opposite_source_face_indices=[2]),
            lambda proof: proof.update(cap_source_face_indices=[1]),
            lambda proof: proof.update(selected_source_face_indices=[False, 1]),
            lambda proof: proof.update(root_source_face_index=1),
            lambda proof: proof.update(original_native_face_edge_vertex_order_preserved=False),
        )
        for change in changes:
            with self.subTest(change=change):
                self.refuses(lambda _, result: change(result["coverage_proof"]))
        self.refuses(lambda _, result: result["bend_ledger"][0].update(paired_source_face_index=2))
        self.refuses(lambda _, result: result["bend_ledger"][0].update(source_face_index=0))

    def test_closed_wire_edge_coverage_and_outer_identity(self):
        self.refuses(lambda _, result: setattr(result["flat_face"].OuterWire, "closed", False), "OPEN_NATIVE_CUT_WIRE")
        self.refuses(lambda _, result: result["flat_face"].OuterWire.OrderedEdges.pop(), "INCOMPLETE_NATIVE_WIRE")
        self.refuses(lambda _, result: setattr(result["flat_face"], "OuterWire", Wire([])), "OUTER_WIRE")
        self.refuses(lambda _, result: result["flat_face"].Wires.append(result["flat_face"].OuterWire), "OUTER_WIRE")

    def test_nonfinite_native_payload_and_flags_refuse(self):
        cases = (
            lambda shape, _: setattr(shape.Vertexes[0].Point, "x", math.nan),
            lambda _, result: result["bend_ledger"][0].update(k_factor_ansi=math.nan),
            lambda _, result: result["bend_ledger"][0].update(bend_allowance_mm=math.inf),
            lambda _, result: result["bend_ledger"][0].update(bend_allowance_mm=1.0),
            lambda _, result: setattr(result["flat_face"].BoundBox, "ZMax", 1.0),
            lambda _, result: result.update(release_allowed=True),
            lambda _, result: result.update(units_verified=True),
            lambda _, result: result["coverage_proof"].update(unexpected_scalar=math.nan),
            lambda _, result: result["coverage_proof"].update(unexpected_native=object()),
            lambda _, result: result["coverage_proof"].update(release_allowed=True),
            lambda _, result: result["coverage_proof"].update(independent_neutral_area_mm2=199.0),
            lambda _, result: result["coverage_proof"].update(linear_numerical_tolerance_mm=1e-3),
            lambda shape, _: setattr(shape.Faces[3].Surface, "Radius", 3.1),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                self.refuses(mutate)

    def test_no_native_import_or_job_on_module_import(self):
        self.assertNotIn("FreeCAD", serializer.__dict__)
        self.assertNotIn("Part", serializer.__dict__)
        self.assertNotIn("FreeCAD", sys.modules)
        self.assertNotIn("Part", sys.modules)


if __name__ == "__main__":
    unittest.main()
