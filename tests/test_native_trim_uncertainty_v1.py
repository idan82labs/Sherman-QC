"""Separate local-trim uncertainty controls; no native execution by default.

The original 16-case matrix is untouched. Root must explicitly opt in to this
new fixed nine-case matrix under its existing native-worker limits. Every shape
is invented afresh; tolerance changes below are adversarial fixture construction,
never adapter-side repair. No physical input, CAD/scan file or reference is read.
"""
from __future__ import annotations

import ast
import builtins
import itertools
import json
import math
import os
from pathlib import Path
import runpy

import numpy as np
import pytest

from backend import cad_guided_native_features_v1 as adapter


NATIVE_OPT_IN = "SHERMAN_RUN_NATIVE_TRIM_UNCERTAINTY_V1"
NATIVE_CASES = (
    "unchanged_ellipse", "located_ellipse", "high_generator_edge",
    "high_generator_vertex", "outside_trim_uncertainty_enters_strip",
    "trim_vertex_exceeds_edge", "uncertainty_covers_all_support",
    "unsupported_boundary_keeps_other_transition", "high_root_cylinder_face",
)
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SOURCE = ROOT / "tests/test_cad_guided_native_features_v1.py"
ADAPTER_SOURCE = ROOT / "backend/cad_guided_native_features_v1.py"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1., -float("inf")))
def test_invalid_native_tolerance_is_not_an_exclusion_radius(value):
    with pytest.raises(adapter._Unsupported, match="INVALID_NATIVE_TOLERANCE"):
        adapter._native_tolerance(value)


def test_strict_contact_requires_edge_and_every_incident_vertex():
    assert adapter.GEOMETRY_TOL_MM == 2e-6
    assert adapter._strict_contact_tolerances(2e-6, (1e-7, 2e-6))
    assert not adapter._strict_contact_tolerances(5e-6, (1e-7, 1e-7))
    assert not adapter._strict_contact_tolerances(1e-7, (1e-7, 5e-6))
    assert adapter._native_tolerance(5e-6) == 5e-6  # numeric, not proof admission


def test_edge_tube_is_unioned_with_actual_vertex_position_and_larger_ball():
    # This vertex is not substituted with either nominal curve endpoint.
    edge_box = (0., 0., 0., 1., 1., 1.)
    bounds = adapter._include_vertex_uncertainty(edge_box, [((1.1, .5, .5), .3)])
    assert bounds == pytest.approx((0., 0., 0., 1.4, 1., 1.))
    assert edge_box == (0., 0., 0., 1., 1., 1.)


def test_oblique_projection_contains_every_vertex_ball_not_just_its_center():
    point, tolerance = np.array([2., -3., 4.]), .25
    bounds = adapter._include_vertex_uncertainty((0., 0., 0., 1., 1., 1.), [(point, tolerance)])
    origin, direction = np.array([7., -11., 13.]), np.array([2., -3., 5.]) / math.sqrt(38.)
    low, high = adapter._box_projection(bounds, origin, direction)
    center = float((point - origin) @ direction)
    assert low <= center - tolerance and high >= center + tolerance
    corners = np.asarray(list(itertools.product(*zip(bounds[:3], bounds[3:]))))
    projected = (corners - origin) @ direction
    assert low == pytest.approx(projected.min(), abs=1e-13)
    assert high == pytest.approx(projected.max(), abs=1e-13)


def test_fixed_enclosure_inflation_is_monotone():
    original = (0., 0., 0., 1., 1., 1.)
    previous = np.asarray(original)
    for tolerance in (0., .01, .2, 2.):
        current = adapter._include_vertex_uncertainty(original, [((1., 0., .5), tolerance)])
        assert np.all(current[:3] <= previous[:3])
        assert np.all(current[3:] >= previous[3:])
        previous = current


def test_nominal_outside_strip_is_not_outside_after_uncertainty_inflation():
    # Native edge box here already includes the edge tolerance. The larger
    # vertex ball enters Wmax=12 even though the entire nominal box is outside.
    nominal = (13., -1., 14., 14., 1., 16.)
    inflated = adapter._include_vertex_uncertainty(nominal, [((13., 0., 15.), 1.25)])
    low, high = adapter._box_projection(inflated, np.zeros(3), np.array([1., 0., 0.]))
    assert nominal[0] > 12. + adapter.BOUNDARY_GUARD_MM
    assert low < 12. < high
    axial = adapter._box_projection(inflated, np.zeros(3), np.array([0., 0., 1.]))
    survivors = adapter._without_boundaries([(0., 30.)], [axial])
    assert len(survivors) == 2
    assert survivors[0][1] < 13.75 and survivors[1][0] > 16.25


def test_uncertainty_enclosure_covering_all_support_has_no_surviving_roi():
    inflated = adapter._include_vertex_uncertainty((4., -8., 13., 6., -7., 17.),
                                                    [((5., -7.5, 17.), 40.)])
    axial = adapter._box_projection(inflated, np.zeros(3), np.array([0., 0., 1.]))
    assert axial == pytest.approx((-23., 57.))
    assert adapter._without_boundaries([(0., 30.)], [axial]) == []


@pytest.mark.parametrize("bounds", (
    (0., 0., 0., float("inf"), 1., 1.), (2., 0., 0., 1., 1., 1.),
    (0., 0., 0., float("nan"), 1., 1.), (0., 0., 0., 1., 1.),
))
def test_invalid_complete_bound_cannot_be_replaced_with_endpoints(bounds):
    with pytest.raises(adapter._Unsupported, match="INVALID_NATIVE_BOUNDING_BOX"):
        adapter._include_vertex_uncertainty(bounds, [])


def test_inflated_bound_outside_numerical_scope_refuses():
    with pytest.raises(adapter._Unsupported, match="UNSUPPORTED_NUMERICAL_SCALE"):
        adapter._include_vertex_uncertainty((0., 0., 0., 1., 1., 1.), [((1e6, 0., 0.), .1)])


def test_native_entry_refuses_before_native_import_or_fixture_source_read(monkeypatch):
    monkeypatch.delenv(NATIVE_OPT_IN, raising=False)
    calls, original = [], builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise AssertionError("unexpected native import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(runpy, "run_path", lambda *args, **kwargs: pytest.fail("fixture source read"))
    with pytest.raises(RuntimeError, match="EXPLICIT_ROOT_NATIVE_OPT_IN_REQUIRED"):
        run_invented_native_trim_uncertainty_checks()
    assert calls == []


def test_source_keeps_strict_proof_and_both_exclusion_only_shortcut_guards():
    source = ADAPTER_SOURCE.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    generator = ast.get_source_segment(source, functions["generator_side"])
    flange = ast.get_source_segment(source, functions["flange_boundaries"])
    assert "if not strict_edges[ei]:" in generator
    assert "if strict_edges[ei] and curve.GetType() == GeomAbs_Line:" in flange
    for name in ("cylinder_data", "flange_contacts", "is_inside"):
        assert "face_tolerances[fi] <= GEOMETRY_TOL_MM" in ast.get_source_segment(source, functions[name])
    assert "BRepBndLib.AddOptimal_s(edges[ei], box, False, True)" in source
    assert "BRepCheck_Analyzer(solid, True, False, True).IsValid()" in source
    assert "BRep_Tool.SameParameter_s(edge) and BRep_Tool.SameRange_s(edge)" in source
    assert "len(features) + len(windows) < MAX_WINDOWS" in source
    assert "tolerance_in_scope" not in source


def _unique_shapes(shape, kind, cast):
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    mapping, result, explorer, count = TopTools_IndexedMapOfShape(), [], TopExp_Explorer(shape, kind), 0
    while explorer.More():
        count += 1
        assert count <= 4096
        value = cast(explorer.Current())
        if mapping.FindIndex(value) == 0:
            mapping.Add(value)
            result.append(value)
        explorer.Next()
    return result


def _vertices(edge):
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopoDS import TopoDS
    return _unique_shapes(edge, TopAbs_VERTEX, TopoDS.Vertex_s)


def _fixture_owner_path(shape):
    """Read only the invented fixture's bounded, single-child owner chain.

    A Boolean result can retain a container before the location wrapper adds
    another. This oracle reads the input before adapter execution and requires
    one solid owner; it never takes the expected path from adapter metadata.
    Location/orientation accumulation is immaterial to this topology-only path.
    """
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_COMPSOLID
    from OCP.TopoDS import TopoDS_Iterator
    current, path = shape, []
    while current.ShapeType() != TopAbs_SOLID:
        assert len(path) < 16, "INVENTED_OWNER_CHAIN_BUDGET"
        assert current.ShapeType() in (TopAbs_COMPOUND, TopAbs_COMPSOLID)
        iterator = TopoDS_Iterator(current, False, False)
        assert iterator.More(), "INVENTED_EMPTY_OWNER_CONTAINER"
        value = iterator.Value()
        current = value.Oriented(value.Orientation())
        iterator.Next()
        assert not iterator.More(), "INVENTED_MULTIPLE_MATERIAL_OWNERS"
        path.append(0)
    return path


def _point(value):
    return np.array([value.X(), value.Y(), value.Z()])


def _generator(shape):
    """Select independent constructor truth x=5,y=0,z=0..30, not adapter IDs."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    found = []
    for edge in _unique_shapes(shape, TopAbs_EDGE, TopoDS.Edge_s):
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() == GeomAbs_Line:
            points = np.array([_point(curve.Value(t)) for t in (curve.FirstParameter(), curve.LastParameter())])
            if (np.max(np.abs(points[:, :2] - np.array([5., 0.]))) < 1e-8
                    and np.allclose(sorted(points[:, 2]), (0., 30.), rtol=0, atol=1e-8)):
                found.append(edge)
    assert len(found) == 1
    return found[0]


def _hole_edges(shape, *, circles=False):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_Circle
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    result = []
    for edge in _unique_shapes(shape, TopAbs_EDGE, TopoDS.Edge_s):
        curve = BRepAdaptor_Curve(edge)
        if not circles and curve.GetType() == GeomAbs_BSplineCurve:
            result.append(edge)
        elif circles and curve.GetType() == GeomAbs_Circle:
            circle = curve.Circle()
            if abs(circle.Radius() - .5) < 1e-8 and abs(circle.Location().Y() + 13.5) < 1e-8:
                result.append(edge)
    assert len(result) == 2
    return result


def _raise_edge_and_vertex_tolerances(edges, tolerance):
    from OCP.BRep import BRep_Builder
    builder = BRep_Builder()
    for edge in edges:
        builder.UpdateEdge(edge, tolerance)
        for vertex in _vertices(edge):
            builder.UpdateVertex(vertex, tolerance + 1e-12)


def _two_transition_strip():
    """Independent connected inverted-U strip, centers (0,0),(-30,0)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

    def p(x, y):
        return gp_Pnt(float(x), float(y), 0.)

    wire = BRepBuilderAPI_MakeWire()

    def line(a, b):
        edge = BRepBuilderAPI_MakeEdge(p(*a), p(*b))
        assert edge.IsDone()
        wire.Add(edge.Edge())

    def arc(cx, radius, start, stop):
        points = [p(cx + radius * math.cos(theta), radius * math.sin(theta))
                  for theta in (start, (start + stop) / 2, stop)]
        curve = GC_MakeArcOfCircle(*points)
        assert curve.IsDone()
        edge = BRepBuilderAPI_MakeEdge(curve.Value())
        assert edge.IsDone()
        wire.Add(edge.Edge())

    line((6., -15.), (6., 0.))
    arc(0., 6., 0., math.pi / 2)
    line((0., 6.), (-30., 6.))
    arc(-30., 6., math.pi / 2, math.pi)
    line((-36., 0.), (-36., -15.))
    line((-36., -15.), (-35., -15.))
    line((-35., -15.), (-35., 0.))
    arc(-30., 5., math.pi, math.pi / 2)
    line((-30., 5.), (0., 5.))
    arc(0., 5., math.pi / 2, 0.)
    line((5., 0.), (5., -15.))
    line((5., -15.), (6., -15.))
    assert wire.IsDone()
    face = BRepBuilderAPI_MakeFace(gp_Pln(p(0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True)
    assert face.IsDone()
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 30.), True, True)
    assert prism.IsDone()
    return prism.Shape()


def run_invented_native_trim_uncertainty_checks():
    """Fixed nine-case, root-owned worker entry; every failure stays scored."""
    if os.environ.get(NATIVE_OPT_IN) != "1":
        raise RuntimeError("EXPLICIT_ROOT_NATIVE_OPT_IN_REQUIRED")
    # This exact named source contains only definitions at top level; reading
    # it is explicit fixture reuse, not raw/physical input or application load.
    fixtures = runpy.run_path(str(FIXTURE_SOURCE))
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from backend.cad_guided_measurement import inspect_cad_guided_scan

    records = []
    for case in NATIVE_CASES:
        record = dict(case=case, passed=False, phase="fixture")
        original_enclosure = adapter._include_vertex_uncertainty
        try:
            shape = _two_transition_strip() if case == "unsupported_boundary_keeps_other_transition" else fixtures["_invented_bend"](90.)
            ellipse_cases = {"unchanged_ellipse", "located_ellipse", "trim_vertex_exceeds_edge",
                             "uncertainty_covers_all_support", "unsupported_boundary_keeps_other_transition"}
            if case in ellipse_cases:
                shape = fixtures["_modify_invented"](shape, "ellipse_trim_90")
            if case == "outside_trim_uncertainty_enters_strip":
                shape = fixtures["_modify_invented"](shape, "unrelated_bore_90")
                _raise_edge_and_vertex_tolerances(_hole_edges(shape, circles=True), 1.25)
            if case == "high_generator_edge":
                _raise_edge_and_vertex_tolerances([_generator(shape)], 5e-6)
            if case == "high_generator_vertex":
                generator = _generator(shape)
                BRep_Builder().UpdateVertex(_vertices(generator)[0], 5e-6)
                assert BRep_Tool.Tolerance_s(generator) <= 2e-6
            if case in ("trim_vertex_exceeds_edge", "uncertainty_covers_all_support"):
                tolerance = .25 if case == "trim_vertex_exceeds_edge" else 40.
                vertex_z = []
                for edge in _hole_edges(shape):
                    assert BRep_Tool.Tolerance_s(edge) <= 5.000001e-6
                    for vertex in _vertices(edge):
                        # The Boolean may choose either seam position. This
                        # native fixture carrier is independent of the adapter;
                        # never assume its closed-edge vertex is at z=17.
                        vertex_z.append(float(BRep_Tool.Pnt_s(vertex).Z()))
                        BRep_Builder().UpdateVertex(vertex, tolerance)
                record["invented_vertex_axial_positions"] = sorted(set(vertex_z))
            if case == "high_root_cylinder_face":
                inner = [face for face in _unique_shapes(shape, TopAbs_FACE, TopoDS.Face_s)
                         if BRepAdaptor_Surface(face, True).GetType() == GeomAbs_Cylinder
                         and abs(BRepAdaptor_Surface(face, True).Cylinder().Radius() - 5.) < 1e-8]
                assert len(inner) == 1
                BRep_Builder().UpdateFace(inner[0], 5e-6)
            rotation, translation = fixtures["_pose"]("located_container_90" if case == "located_ellipse" else "plain_90")
            if case == "located_ellipse":
                record["fixture_owner_path_before_location"] = _fixture_owner_path(shape)
                record["expected_owner_path"] = [0] + record["fixture_owner_path_before_location"]
                shape = fixtures["_located_container"](shape, rotation, translation)
            if case == "unsupported_boundary_keeps_other_transition":
                # Deliberate local bound failure, not fabricated geometry or a
                # passing endpoint proxy. The second transition has no such
                # boundary. This isolates exception scope from native behavior.
                def fail_right_hole(edge_bounds, vertex_data):
                    if edge_bounds[0] > 4.:
                        raise adapter._Unsupported("INVENTED_UNSUPPORTED_LOCAL_BOUNDARY")
                    return original_enclosure(edge_bounds, vertex_data)
                adapter._include_vertex_uncertainty = fail_right_hole
                record["failure_injection"] = "RIGHT_ELLIPSE_COMPLETE_BOUND_UNAVAILABLE"
            record["phase"] = "native_adapter"
            result = adapter.extract_ocp_guided_features_v1(shape, policy=fixtures["_policy"](),
                                                           unit_status="DECLARED_MM", allow_ocp=True)
            metadata = result["metadata"]
            record.update(native_status=result["status"], native_metadata=metadata)
            json.dumps(metadata, allow_nan=False)
            assert metadata["geometry_tolerance_mm"] == 2e-6
            assert metadata["production_use_allowed"] is False
            assert metadata["physical_observed_bend_count"] is None
            assert metadata["exhaustive_bend_inventory"] is False
            if case in ("high_generator_edge", "high_generator_vertex", "high_root_cylinder_face"):
                assert result["status"] == "UNAVAILABLE" and result["features"] == ()
                assert metadata["candidate_inventory_complete"] is True
                assert metadata["unresolved_candidate_count"] >= 1
                expected_reason = ("ANALYTIC_CYLINDER_TOLERANCE_OUTSIDE_SCOPE" if case == "high_root_cylinder_face"
                                   else "TWO_NATIVE_GENERATOR_CONTACTS_REQUIRED")
                assert any(expected_reason in item["reasons"] for item in metadata["candidates"])
            elif case == "uncertainty_covers_all_support":
                assert result["status"] == "UNAVAILABLE" and result["features"] == ()
                # This stronger attribution ensures the positive pure-box
                # exclusion proof isn't confused with global invalid topology.
                assert metadata["candidate_inventory_complete"] is True
                assert any("NO_BOUNDARY_CERTIFIED_COMMON_AXIAL_WINDOW" in item["reasons"]
                           for item in metadata["candidates"])
            elif case == "unsupported_boundary_keeps_other_transition":
                assert result["status"] == "PARTIAL" and len(result["features"]) == 1
                assert metadata["supported_native_candidate_count"] == 1
                assert metadata["unresolved_candidate_count"] == 1
                assert any("INVENTED_UNSUPPORTED_LOCAL_BOUNDARY" in item["reasons"]
                           for item in metadata["candidates"])
                assert abs(result["features"][0].origin[0] + 30.) < 1e-8
                record["phase"] = "actual_measurement_kernel"
                scan, _, _ = fixtures["_analytic_scan"]("plain_90", 90.)
                left_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
                scan = scan @ left_rotation.T + np.array([-30., 0., 0.])
                measured = inspect_cad_guided_scan(scan, result["features"],
                                                   registration_status="DECLARED_ALIGNED")
                assert measured["measured_conditional_feature_count"] == 1
                row = measured["features"][0]
                assert abs(row["measurements"]["measured_radius_mm"] - 6.) <= 2e-6
                assert abs(row["nominal_radius_mm"] - 5.) <= 2e-6
                assert row["conformance"] == "INDETERMINATE"
                assert measured["observed_bend_count"] is None
                record["measured_conditional_feature_count"] = 1
            else:
                assert result["status"] == "AVAILABLE" and len(result["features"]) == 2
                assert metadata["candidate_inventory_complete"] is True
                assert metadata["unresolved_candidate_count"] == 0
                assert metadata["exclusion_only_edge_indices"]
                assert metadata["native_trim_exclusion_records"]
                assert all(set(item["contact_edge_indices"][0] + item["contact_edge_indices"][1]).isdisjoint(
                    metadata["exclusion_only_edge_indices"]) for item in metadata["features"])
                spans = fixtures["_construction_axial_intervals"](result["features"], rotation, translation)
                if case == "outside_trim_uncertainty_enters_strip":
                    assert spans[0][1] < 13.25 and spans[1][0] > 16.75
                    scan_case = "plain_90"
                else:
                    assert spans[0][1] < 13. and spans[1][0] > 17.
                    scan_case = "ellipse_trim_90"
                if case == "trim_vertex_exceeds_edge":
                    assert max(max(item["incident_vertex_tolerances_mm"])
                               for item in metadata["native_trim_exclusion_records"]) == .25
                    assert all(stop < z - .25 or start > z + .25
                               for start, stop in spans for z in vertex_z)
                if case == "located_ellipse":
                    assert metadata["owner_path"] == record["expected_owner_path"]
                record["phase"] = "actual_measurement_kernel"
                scan, expected_radius, expected_sweep = fixtures["_analytic_scan"](scan_case, 90.)
                measured = inspect_cad_guided_scan(scan @ rotation.T + translation, result["features"],
                                                   registration_status="DECLARED_ALIGNED")
                assert measured["measured_conditional_feature_count"] == 2
                for row in measured["features"]:
                    output = row["measurements"]
                    assert abs(output["measured_radius_mm"] - expected_radius) <= 2e-6
                    assert abs(abs(output["measured_sweep_deg"]) - expected_sweep) <= 2e-6
                    assert abs(row["nominal_radius_mm"] - 5.) <= 2e-6
                    assert row["conformance"] == "INDETERMINATE"
                    assert row["physical_completion_state"] == "UNKNOWN"
                assert measured["observed_bend_count"] is None
                record["measured_conditional_feature_count"] = 2
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error))
        finally:
            adapter._include_vertex_uncertainty = original_enclosure
        records.append(record)
    return dict(scope="INVENTED_LOCAL_TRIM_UNCERTAINTY_V1_ENGINEERING_ONLY",
                expected_case_count=len(NATIVE_CASES), scored_case_count=len(records),
                passed=sum(row["passed"] for row in records), records=records,
                physical_generalization_claim=False)


@pytest.mark.skipif(os.environ.get(NATIVE_OPT_IN) != "1", reason="Root-owned native worker opt-in required")
def test_fixed_native_trim_uncertainty_matrix():
    report = run_invented_native_trim_uncertainty_checks()
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    assert report["expected_case_count"] == report["scored_case_count"] == 9
    assert report["passed"] == 9, report
