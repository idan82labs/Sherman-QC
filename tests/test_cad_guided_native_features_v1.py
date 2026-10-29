"""V1 source/pure-math contracts and opt-in, invented-only native fixtures.

Default collection never invokes a native library. Root may explicitly enable
the fixed native matrix after reviewing this entire file and the adapter under
its one-worker CPU/RSS/wall budget. No CAD/scan files are read or written. Native
geometry and analytic scan positions are independently constructed, not derived
from the adapter's output. This is engineering validation, not a corpus score.
"""
from __future__ import annotations

import ast
import builtins
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend import cad_guided_native_features_v1 as adapter


SOURCE = Path(__file__).resolve().parents[1] / "backend/cad_guided_native_features_v1.py"
NATIVE_CASES = (
    "plain_60", "plain_90", "plain_135", "oblique_90", "axial_flip_90",
    "flange_hole_90", "flange_notch_90", "angle_deviation_90",
    "unrelated_bore_90", "many_unrelated_bores_90", "unresolved_open_notch_90",
    "ellipse_trim_90", "located_container_90",
    "no_outer_skin", "two_material_owners", "reversed_solid",
)
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_NATIVE_V1"


def _policy():
    return adapter.NativeGuidedPolicy(
        flange_window_mm=(3., 12.), search_margin_mm=2.,
        max_fit_residual_mm=.01, plane_exclusion_mm=.05,
        max_tangency_error_mm=.05)


@pytest.fixture
def guarded_native(monkeypatch):
    calls, original = [], builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise RuntimeError("NATIVE_IMPORT_FORBIDDEN_IN_SOURCE_CHECK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return calls


def test_whole_module_import_does_not_load_native_or_legacy_app(guarded_native):
    spec = importlib.util.spec_from_file_location("backend._native_v1_source_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert guarded_native == []
    assert module.NativeGuidedPolicy is adapter.NativeGuidedPolicy
    tree = ast.parse(SOURCE.read_text())
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {node.module for node in top_imports if isinstance(node, ast.ImportFrom)}
    assert names == {"__future__", "dataclasses", "cad_guided_native_features", "cad_guided_measurement"}
    assert {alias.name for node in top_imports if isinstance(node, ast.Import) for alias in node.names} == {"math", "numpy"}


@pytest.mark.parametrize("flag", (False, None, 1, "true", np.bool_(True)))
def test_explicit_opt_in_before_native_import(guarded_native, flag):
    result = adapter.extract_ocp_guided_features_v1(object(), policy=_policy(),
                                                   unit_status="DECLARED_MM", allow_ocp=flag)
    assert result["status"] == "UNAVAILABLE"
    assert result["metadata"]["reasons"] == ["EXPLICIT_OCP_BINDING_REQUIRED"]
    assert guarded_native == []


@pytest.mark.parametrize("units", (None, "mm", "UNVERIFIED", True))
def test_explicit_units_before_native_import(guarded_native, units):
    result = adapter.extract_ocp_guided_features_v1(object(), policy=_policy(),
                                                   unit_status=units, allow_ocp=True)
    assert result["metadata"]["reasons"] == ["EXPLICIT_NATIVE_MILLIMETRE_DECLARATION_REQUIRED"]
    assert guarded_native == []


def test_policy_failure_before_native_import(guarded_native):
    result = adapter.extract_ocp_guided_features_v1(object(), policy={},
                                                   unit_status="DECLARED_MM", allow_ocp=True)
    assert result["metadata"]["reasons"] == ["VALIDATED_SEARCH_POLICY_REQUIRED"]
    assert guarded_native == []


def test_native_import_failure_preserves_unknown_not_defect(guarded_native):
    result = adapter.extract_ocp_guided_features_v1(object(), policy=_policy(),
                                                   unit_status="DECLARED_MM", allow_ocp=True)
    assert result["features"] == ()
    assert result["metadata"]["reasons"] == ["NATIVE_EXTRACTION_ERROR:RuntimeError"]
    assert guarded_native == ["OCP.BRep"]
    assert result["metadata"]["physical_observed_bend_count"] is None
    assert result["metadata"]["candidate_inventory_complete"] is False
    assert result["metadata"]["exhaustive_bend_inventory"] is False
    assert result["metadata"]["production_use_allowed"] is False
    json.dumps(result["metadata"], allow_nan=False)


def test_interval_union_preserves_positive_gaps_even_below_geometry_tolerance():
    assert adapter._union([(8., 10.), (0., 2.), (1., 3.), (3.000000001, 5.)]) == [
        (0., 3.), (3.000000001, 5.), (8., 10.)]


def test_interval_intersection_preserves_native_contact_gaps():
    assert adapter._intersect([(0., 5.), (8., 12.)], [(2., 9.), (10., 15.)]) == [
        (2., 5.), (8., 9.), (10., 12.)]


def test_hole_boundary_band_splits_a_window_instead_of_corner_certification():
    assert adapter._without_boundaries([(0., 30.)], [(14., 16.)], guard=.001) == [
        (0., 13.999), (16.001, 30.)]


def test_enclosure_covering_whole_window_abstains():
    assert adapter._without_boundaries([(0., 30.)], [(-1., 31.)]) == []


@pytest.mark.parametrize("intervals", (((float("nan"), 1.),), ((2., 1.),), ((0., float("inf")),)))
def test_invalid_interval_rejected(intervals):
    with pytest.raises(adapter._Unsupported, match="INVALID_NATIVE_INTERVAL"):
        adapter._union(intervals)


def test_world_aabb_projection_is_complete_under_oblique_axis():
    bounds = (-2., -3., -5., 7., 11., 13.)
    origin = np.array([4., -9., 2.])
    axis = np.array([-2., 3., -5.]) / math.sqrt(38.)
    low, high = adapter._box_projection(bounds, origin, axis)
    corners = np.asarray(list(itertools.product(*zip(bounds[:3], bounds[3:]))))
    projected = (corners - origin) @ axis
    assert low == pytest.approx(projected.min(), abs=1e-13)
    assert high == pytest.approx(projected.max(), abs=1e-13)


def test_off_surface_plane_and_cylinder_witness_distances_are_not_uv_membership():
    assert adapter._plane_distance((0., 0., 100.), (0., 0., 0.), (0., 0., 1.)) == 100.
    assert adapter._cylinder_distance((105., 0., 17.), (0., 0., 0.), (0., 0., 1.), 5.) == 100.
    assert adapter._plane_distance((13., -4., 0.), (0., 0., 0.), (0., 0., 1.)) == 0.
    assert adapter._cylinder_distance((3., 4., 17.), (0., 0., 0.), (0., 0., 1.), 5.) == 0.


def test_long_flange_normal_error_cannot_hide_behind_exact_center_witness():
    # The center is exactly on the plane and the angular error is below 1e-7,
    # but a 100 mm axial strip leaves that plane by 4.5e-6 mm at either end.
    normal = np.array([1., 0., 9e-8])
    normal /= np.linalg.norm(normal)
    origin, center = np.array([5., 0., 0.]), np.array([5., -6., 0.])
    assert adapter._plane_distance(center, origin, normal) == 0.
    assert np.linalg.norm(normal - np.array([1., 0., 0.])) < adapter.DIRECTION_TOL
    bound = adapter._plane_rectangle_max_distance(
        origin, (0., 0., 1.), (0., -1., 0.), (-50., 50.), 12., origin, normal)
    assert bound == pytest.approx(4.5e-6, abs=1e-18)
    assert bound > adapter.GEOMETRY_TOL_MM


def test_plane_rectangle_distance_covers_width_and_both_axial_endpoints():
    normal = np.array([1., 2e-8, -3e-8])
    normal /= np.linalg.norm(normal)
    origin = np.array([5., 0., 0.])
    value = adapter._plane_rectangle_max_distance(
        origin, (0., 0., 1.), (0., -1., 0.), (-10., 30.), 12., origin, normal)
    assert value == pytest.approx(1.14e-6, abs=1e-18)
    assert value < adapter.GEOMETRY_TOL_MM


def _construction_axial_intervals(features, rotation, translation):
    """Map native parameter endpoints back into the independent fixture frame.

    p = origin + parameter * axis; fixture_z = e_z . R.T . (p - t).
    Neither an absolute-value shortcut nor a native-axis sign assumption is
    valid when the parameter origin or the native location is nonzero.
    """
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    intervals = []
    for feature in features:
        origin, axis = np.asarray(feature.origin), np.asarray(feature.axis)
        values = [float((rotation.T @ (origin + parameter * axis - translation))[2])
                  for parameter in feature.span_mm]
        intervals.append(tuple(sorted(values)))
    return sorted(intervals)


def test_construction_interval_oracle_handles_negative_axis_offset_and_known_pose():
    rotation, translation = _pose("oblique_90")
    origin = translation + rotation @ np.array([2., -3., 10.])
    negative = SimpleNamespace(origin=origin, axis=-(rotation @ np.array([0., 0., 1.])),
                               span_mm=(-6., -2.))
    positive = SimpleNamespace(origin=origin, axis=rotation @ np.array([0., 0., 1.]),
                               span_mm=(2., 6.))
    later = SimpleNamespace(origin=translation + rotation @ np.array([2., -3., 30.]),
                            axis=negative.axis, span_mm=(-2., 0.))
    negative_result = _construction_axial_intervals((later, negative), rotation, translation)
    positive_result = _construction_axial_intervals((positive,), rotation, translation)
    assert negative_result[0] == pytest.approx((12., 16.), abs=1e-12)
    assert negative_result[1] == pytest.approx((30., 32.), abs=1e-12)
    assert positive_result[0] == pytest.approx(negative_result[0], abs=1e-12)


def test_source_caps_and_native_boundary_not_mesh_contract():
    assert (adapter.MAX_FACES, adapter.MAX_TREE_NODES, adapter.MAX_TREE_DEPTH) == (1024, 32768, 32)
    assert adapter.MAX_WINDOWS == 4096 and adapter.KERNEL_MAX_FEATURES == 16
    source = SOURCE.read_text()
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and any(word in name.lower() for word in (
        "mesh", "registration", "pipeline", "scorer", "cad_import", "open3d")) for name in imports)
    assert "BRep_Tool.IsClosed_s(edges[ei], faces[owners[0][0]])" in source
    assert "BRepBndLib.AddOptimal_s(edges[ei], box, False, True)" in source
    assert "NATIVE_CANDIDATE_THEN_WINDOW_ORDER_NO_TRUNCATION" in source
    assert "if distance > GEOMETRY_TOL_MM:" in source
    assert "len(features) + len(windows) < MAX_WINDOWS" in source


def _invented_bend(sweep_deg, *, thickness=1., flange_length=15.):
    """Independent 2-D strip construction, then an exact 30 mm extrusion."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

    phi, radius = math.radians(sweep_deg), 5.

    def polar(r, theta):
        return np.array([r * math.cos(theta), r * math.sin(theta), 0.])

    def point(value):
        return gp_Pnt(*map(float, value))

    def line(a, b):
        builder = BRepBuilderAPI_MakeEdge(point(a), point(b))
        assert builder.IsDone()
        return builder.Edge()

    def arc(r, first, last):
        curve = GC_MakeArcOfCircle(point(polar(r, first)), point(polar(r, (first + last) / 2)),
                                  point(polar(r, last)))
        assert curve.IsDone()
        builder = BRepBuilderAPI_MakeEdge(curve.Value())
        assert builder.IsDone()
        return builder.Edge()

    a, b = polar(radius, 0.), polar(radius, phi)
    c, d = polar(radius + thickness, 0.), polar(radius + thickness, phi)
    first_ray, last_ray = np.array([0., -1., 0.]), np.array([-math.sin(phi), math.cos(phi), 0.])
    edges = [line(c + flange_length * first_ray, c), arc(radius + thickness, 0., phi),
             line(d, d + flange_length * last_ray),
             line(d + flange_length * last_ray, b + flange_length * last_ray),
             line(b + flange_length * last_ray, b), arc(radius, phi, 0.),
             line(a, a + flange_length * first_ray),
             line(a + flange_length * first_ray, c + flange_length * first_ray)]
    wire = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire.Add(edge)
        assert wire.IsDone()
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True)
    assert face.IsDone()
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 30.), True, True)
    assert prism.IsDone()
    return prism.Shape()


def _cut(shape, cutter):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    result = BRepAlgoAPI_Cut(shape, cutter)
    assert result.IsDone()
    return result.Shape()


def _modify_invented(shape, case):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    if case == "flange_hole_90":
        # Through-thickness bore crosses the root-to-search strip at z=14..16.
        bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(4., -7.5, 15.), gp_Dir(1., 0., 0.)), 1., 3.)
        return _cut(shape, bore.Shape())
    if case == "unrelated_bore_90":
        # Same periodic cavity, entirely beyond the declared Wmax=12 strip.
        bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(4., -13.5, 15.), gp_Dir(1., 0., 0.)), .5, 3.)
        return _cut(shape, bore.Shape())
    if case == "ellipse_trim_90":
        # Elliptical hole edges exercise AddOptimal's complete non-line/circle
        # curve enclosure, not the analytic line/circle shortcuts. The ellipse
        # major axis is native world Z; its excluded z band is [13,17].
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Elips, gp_Vec
        axes = gp_Ax2(gp_Pnt(4., -7.5, 15.), gp_Dir(1., 0., 0.), gp_Dir(0., 0., 1.))
        edge = BRepBuilderAPI_MakeEdge(gp_Elips(axes, 2., 1.))
        assert edge.IsDone()
        wire = BRepBuilderAPI_MakeWire(edge.Edge())
        assert wire.IsDone()
        face = BRepBuilderAPI_MakeFace(wire.Wire(), True)
        assert face.IsDone()
        prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(3., 0., 0.), True, True)
        assert prism.IsDone()
        return _cut(shape, prism.Shape())
    if case == "many_unrelated_bores_90":
        # One Boolean operation with 130 invented disjoint tiny bores gives
        # >128 faces, exercising the successor bound without any physical CAD.
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        compound, builder = TopoDS_Compound(), BRep_Builder()
        builder.MakeCompound(compound)
        for index in range(130):
            axis = gp_Ax2(gp_Pnt(4., -13.5, .2 + .225 * index), gp_Dir(1., 0., 0.))
            builder.Add(compound, BRepPrimAPI_MakeCylinder(axis, .04, 3.).Shape())
        return _cut(shape, compound)
    if case == "unresolved_open_notch_90":
        # A half-cylindrical cavity reaches the flange's free end. It lies
        # outside this feature's strip, but is an unresolved partial cylinder
        # and must force PARTIAL without deleting the valid main transition.
        notch = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(4., -15., 15.), gp_Dir(1., 0., 0.)), 1., 3.)
        return _cut(shape, notch.Shape())
    if case == "flange_notch_90":
        # A concave nonrectangular trim removes interior strip material at one
        # end. UV bounds and most original corners cannot certify this face.
        notch = BRepPrimAPI_MakeBox(gp_Pnt(4., -10., -1.), 3., 5., 6.)
        return _cut(shape, notch.Shape())
    if case == "two_material_owners":
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        compound, builder = TopoDS_Compound(), BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, shape)
        builder.Add(compound, BRepPrimAPI_MakeBox(gp_Pnt(100., 100., 100.), 1., 1., 1.).Shape())
        return compound
    if case == "reversed_solid":
        return shape.Reversed()
    return shape


def _pose(case):
    if case == "axial_flip_90":
        return np.diag([1., -1., -1.]), np.array([-33., 24., 42.])
    if case in ("oblique_90", "located_container_90"):
        axis, theta = np.array([2., -1., 3.]) / math.sqrt(14.), 1.23
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rotation = math.cos(theta) * np.eye(3) + (1 - math.cos(theta)) * np.outer(axis, axis) + math.sin(theta) * skew
        return rotation, np.array([31., -47., 19.])
    return np.eye(3), np.zeros(3)


def _move_native(shape, rotation, translation):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    transform = gp_Trsf()
    transform.SetValues(*[float(value) for row, shift in zip(rotation, translation) for value in (*row, shift)])
    builder = BRepBuilderAPI_Transform(shape, transform, True)
    assert builder.IsDone()
    return builder.Shape()


def _located_container(shape, rotation, translation):
    """One-child owner wrapper with native Location, no geometry baking."""
    from OCP.BRep import BRep_Builder
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Trsf
    container, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(container)
    builder.Add(container, shape)
    transform = gp_Trsf()
    transform.SetValues(*[float(value) for row, shift in zip(rotation, translation) for value in (*row, shift)])
    return container.Located(TopLoc_Location(transform))


def _analytic_scan(case, sweep):
    radius = 5. if case == "angle_deviation_90" else 6.
    actual_sweep = sweep + (5. if case == "angle_deviation_90" else 0.)
    phi = math.radians(actual_sweep)
    axial = np.linspace(1., 29., 29)
    if case == "flange_hole_90":
        axial = axial[(axial <= 13.) | (axial >= 17.)]
    if case == "ellipse_trim_90":
        axial = axial[(axial <= 12.) | (axial >= 18.)]
    if case == "flange_notch_90":
        axial = axial[axial >= 6.]
    along = np.linspace(3., 12., 19)
    points = [[radius * math.cos(theta), radius * math.sin(theta), z]
              for theta in np.linspace(0., phi, 101) for z in axial]
    points += [[radius, -t, z] for t in along for z in axial]
    points += [[radius * math.cos(phi) - t * math.sin(phi),
                radius * math.sin(phi) + t * math.cos(phi), z] for t in along for z in axial]
    return np.asarray(points, dtype=float), radius, actual_sweep


def run_invented_native_checks():
    """Root-owned bounded worker entry; every fixed case retains its outcome."""
    from backend.cad_guided_measurement import inspect_cad_guided_scan

    records = []
    for case in NATIVE_CASES:
        record = dict(case=case, phase="fixture", passed=False)
        try:
            sweep = 60. if case == "plain_60" else 135. if case == "plain_135" else 90.
            # The no-skin control keeps the inner cylindrical/planar support but
            # changes the outer circular skin into a triangulated polyline in
            # the independent constructor below; no detector output is edited.
            shape = _invented_without_circular_outer() if case == "no_outer_skin" else _invented_bend(sweep)
            shape = _modify_invented(shape, case)
            rotation, translation = _pose(case)
            if case in ("oblique_90", "axial_flip_90"):
                shape = _move_native(shape, rotation, translation)
            if case == "located_container_90":
                shape = _located_container(shape, rotation, translation)
            record["phase"] = "native_adapter"
            result = adapter.extract_ocp_guided_features_v1(
                shape, policy=_policy(), unit_status="DECLARED_MM", allow_ocp=True)
            record["native_status"] = result["status"]
            record["native_metadata"] = result["metadata"]
            json.dumps(result["metadata"], allow_nan=False)
            if case in ("two_material_owners", "reversed_solid", "no_outer_skin"):
                assert result["status"] == "UNAVAILABLE" and result["features"] == ()
                if case == "no_outer_skin":
                    assert result["metadata"]["candidate_inventory_complete"] is True
                    assert result["metadata"]["unresolved_candidate_count"] == 1
                    assert any("UNAMBIGUOUS_COAXIAL_OUTER_SKIN_REQUIRED" in item["reasons"]
                               for item in result["metadata"]["candidates"])
            else:
                expected_status = "PARTIAL" if case == "unresolved_open_notch_90" else "AVAILABLE"
                assert result["status"] == expected_status, result["metadata"]
                expected_windows = 2 if case in ("flange_hole_90", "ellipse_trim_90") else 1
                assert len(result["features"]) == expected_windows
                assert result["metadata"]["supported_native_candidate_count"] == 1
                assert result["metadata"]["candidate_inventory_complete"] is True
                if case == "unresolved_open_notch_90":
                    assert result["metadata"]["unresolved_candidate_count"] >= 1
                    assert any(item["classification"] == "UNRESOLVED_PARTIAL_INNER_CYLINDER"
                               and item["reasons"] for item in result["metadata"]["candidates"])
                else:
                    assert result["metadata"]["unresolved_candidate_count"] == 0
                assert result["metadata"]["physical_observed_bend_count"] is None
                assert result["metadata"]["exhaustive_bend_inventory"] is False
                if case in ("flange_hole_90", "unrelated_bore_90", "many_unrelated_bores_90"):
                    assert result["metadata"]["certified_seam_edge_indices"]
                    assert any(item["classification"] == "FULL_WRAP_CONCAVE_CAVITY_OR_ROLLED_WALL"
                               for item in result["metadata"]["candidates"])
                if case == "many_unrelated_bores_90":
                    assert 128 < result["metadata"]["native_face_count"] <= 1024
                if case == "located_container_90":
                    assert result["metadata"]["owner_path"] == [0]
                if case == "ellipse_trim_90":
                    assert result["metadata"]["native_complete_curve_box_edge_indices"]
                    spans = _construction_axial_intervals(result["features"], rotation, translation)
                    assert spans[0][1] < 13. and spans[1][0] > 17.
                if case == "flange_hole_90":
                    spans = _construction_axial_intervals(result["features"], rotation, translation)
                    assert spans[0][1] < 14. and spans[1][0] > 16.
                    assert len({feature.nominal_source for feature in result["features"]}) == 1
                if case == "flange_notch_90":
                    spans = _construction_axial_intervals(result["features"], rotation, translation)
                    assert spans[0][0] > 5.
                record["phase"] = "actual_measurement_kernel"
                points, expected_radius, expected_sweep = _analytic_scan(case, sweep)
                points = points @ rotation.T + translation
                measured = inspect_cad_guided_scan(points, result["features"], registration_status="DECLARED_ALIGNED")
                assert measured["measured_conditional_feature_count"] == expected_windows, measured
                errors = []
                for row in measured["features"]:
                    output = row["measurements"]
                    errors.append(abs(output["measured_radius_mm"] - expected_radius))
                    errors.append(abs(abs(output["measured_sweep_deg"]) - expected_sweep))
                    assert abs(row["nominal_radius_mm"] - 5.) <= 2e-6
                    assert abs(abs(row["nominal_sweep_deg"]) - sweep) <= 2e-6
                    assert row["conformance"] == "INDETERMINATE"
                    assert row["physical_completion_state"] == "UNKNOWN"
                    # Independent expected axis line, not the adapter origin.
                    fitted_axis = np.asarray(output["measured_axis_direction"])
                    expected_axis = rotation @ np.array([0., 0., 1.])
                    assert abs(abs(float(fitted_axis @ expected_axis)) - 1.) <= 2e-6
                    center = np.asarray(output["measured_axis_point"])
                    offset = center - translation
                    errors.append(float(np.linalg.norm(offset - (offset @ expected_axis) * expected_axis)))
                assert max(errors) <= 2e-6, errors
                record["maximum_geometry_error_mm_or_deg"] = max(errors)
                record["measured_conditional_feature_count"] = measured["measured_conditional_feature_count"]
                assert measured["observed_bend_count"] is None
            record.update(phase="complete", passed=True)
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error))
        records.append(record)
    return dict(scope="INVENTED_NATIVE_V1_ENGINEERING_ONLY", expected_case_count=len(NATIVE_CASES),
                scored_case_count=len(records), passed=sum(row["passed"] for row in records),
                records=records, physical_generalization_claim=False)


def _invented_without_circular_outer():
    """Same inner arc/flanges but a polygonal outer skin: unsupported, retained."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

    def p(x, y):
        return gp_Pnt(float(x), float(y), 0.)

    points = [p(6., -15.), p(6., 0.), p(6., 6.), p(0., 6.), p(-15., 6.), p(-15., 5.), p(0., 5.)]
    wire = BRepBuilderAPI_MakeWire()
    for first, last in zip(points, points[1:]):
        wire.Add(BRepBuilderAPI_MakeEdge(first, last).Edge())
    arc = GC_MakeArcOfCircle(p(0., 5.), p(5. / math.sqrt(2.), 5. / math.sqrt(2.)), p(5., 0.))
    assert arc.IsDone()
    wire.Add(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(p(5., 0.), p(5., -15.)).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(p(5., -15.), p(6., -15.)).Edge())
    assert wire.IsDone()
    face = BRepBuilderAPI_MakeFace(gp_Pln(p(0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True)
    assert face.IsDone()
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 30.), True, True)
    assert prism.IsDone()
    return prism.Shape()


def run_invented_native_tolerance_probe():
    """Separate fixed diagnostic, not a rerun or fixture/acceptance repair.

    Root owns the 20 s wall / 15 s CPU / 512 MiB RSS / 1 MiB output supervisor.
    This function constructs only the unchanged invented base and ellipse cut.
    It calls neither the feature adapter nor the measurement/registration engine.
    No shape repair, tolerance mutation, geometry coordinates or file I/O occurs
    here. The unchanged Boolean constructor may share/update native carriers;
    therefore base-before and base-after inventories are both retained.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS, TopoDS_Iterator

    phases = ("base_before_cut", "unchanged_ellipse_result", "base_after_cut")
    records = []

    def require(condition, reason):
        if not condition:
            raise RuntimeError(reason)

    def inventory(shape):
        stack, nodes = [(shape, 0)], 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            require(nodes <= 512 and depth <= 16, "TOLERANCE_PROBE_TREE_BUDGET")
            iterator = TopoDS_Iterator(current, True, True)
            while iterator.More():
                require(nodes + len(stack) < 512, "TOLERANCE_PROBE_TREE_BUDGET")
                child = iterator.Value()
                stack.append((child.Oriented(child.Orientation()), depth + 1))
                iterator.Next()

        def occurrences(item, kind):
            count, explorer = 0, TopExp_Explorer(item, kind)
            while explorer.More():
                count += 1
                require(count <= 512, "TOLERANCE_PROBE_OCCURRENCE_BUDGET")
                current = explorer.Current()
                yield current.Oriented(current.Orientation())
                explorer.Next()

        maps, shapes, tables, summaries = {}, {}, {}, {}
        for label, kind, cast in (("vertex", TopAbs_VERTEX, TopoDS.Vertex_s),
                                  ("edge", TopAbs_EDGE, TopoDS.Edge_s),
                                  ("face", TopAbs_FACE, TopoDS.Face_s)):
            mapping, unique, count = TopTools_IndexedMapOfShape(), [], 0
            for occurrence in occurrences(shape, kind):
                count += 1
                value = cast(occurrence)
                index = mapping.Add(value)
                if index > len(unique):
                    require(index == len(unique) + 1, "TOLERANCE_PROBE_IDENTITY_MAP_MISMATCH")
                    unique.append(value)
            rows = []
            for index, value in enumerate(unique, start=1):
                tolerance = float(BRep_Tool.Tolerance_s(value))
                valid = math.isfinite(tolerance) and tolerance >= 0
                row = dict(native_id=index, tolerance_mm=tolerance if valid else None,
                           tolerance_numeric_valid=valid,
                           outside_frozen_scope=not valid or tolerance > 2e-6)
                if label in ("edge", "face"):
                    try:
                        if label == "edge":
                            row["geometry_type"] = ("DEGENERATE" if BRep_Tool.Degenerated_s(value)
                                                    else str(BRepAdaptor_Curve(value).GetType()))
                        else:
                            row["geometry_type"] = str(BRepAdaptor_Surface(value, True).GetType())
                    except Exception as error:
                        row["geometry_type_error"] = type(error).__name__
                rows.append(row)
            values = [row["tolerance_mm"] for row in rows if row["tolerance_numeric_valid"]]
            summaries[label] = dict(occurrence_count=count, unique_count=len(unique),
                                   min_tolerance_mm=min(values) if values else None,
                                   max_tolerance_mm=max(values) if values else None,
                                   outside_frozen_scope_ids=[row["native_id"] for row in rows
                                                             if row["outside_frozen_scope"]])
            maps[label], shapes[label], tables[label] = mapping, unique, rows
        relationships = 0
        for owner_label, child_label, child_kind in (("edge", "vertex", TopAbs_VERTEX),
                                                      ("face", "edge", TopAbs_EDGE)):
            for owner, row in zip(shapes[owner_label], tables[owner_label]):
                children = set()
                for child in occurrences(owner, child_kind):
                    relationships += 1
                    require(relationships <= 1024, "TOLERANCE_PROBE_OWNERSHIP_BUDGET")
                    native_id = maps[child_label].FindIndex(child)
                    require(native_id > 0, "TOLERANCE_PROBE_OWNER_CHILD_MISMATCH")
                    children.add(native_id)
                row[child_label + "_ids"] = sorted(children)
        return dict(native_tree_nodes=nodes, ownership_occurrence_count=relationships,
                    identity_scope="IMPORT_LOCAL_PER_PHASE_NOT_PERSISTENT",
                    summary=summaries, topology=tables)

    try:
        base = _invented_bend(90.)
        records.append(dict(phase=phases[0], status="COMPLETE", inventory=inventory(base)))
    except Exception as error:
        return dict(scope="INVENTED_NATIVE_TOLERANCE_DIAGNOSTIC_ONLY", frozen_tolerance_mm=2e-6,
                    records=[dict(phase=phase, status="UNAVAILABLE", error_type=type(error).__name__)
                             for phase in phases], physical_generalization_claim=False)
    try:
        ellipse = _modify_invented(base, "ellipse_trim_90")
        records.append(dict(phase=phases[1], status="COMPLETE", inventory=inventory(ellipse)))
    except Exception as error:
        records.append(dict(phase=phases[1], status="UNAVAILABLE", error_type=type(error).__name__))
    try:
        records.append(dict(phase=phases[2], status="COMPLETE", inventory=inventory(base)))
    except Exception as error:
        records.append(dict(phase=phases[2], status="UNAVAILABLE", error_type=type(error).__name__))
    return dict(scope="INVENTED_NATIVE_TOLERANCE_DIAGNOSTIC_ONLY", frozen_tolerance_mm=2e-6,
                records=records, physical_generalization_claim=False)


@pytest.mark.skipif(os.environ.get(NATIVE_OPT_IN) != "1", reason="Root-reviewed invented-native worker opt-in required")
def test_fixed_invented_native_matrix():
    report = run_invented_native_checks()
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    print(json.dumps({"native_matrix_compact": [
        {key: row.get(key) for key in ("case", "passed", "phase", "native_status", "error_type", "error")}
        for row in report["records"] if not row["passed"]]}, sort_keys=True), flush=True)
    assert report["scored_case_count"] == report["expected_case_count"] == 16
    assert report["passed"] == 16, report
