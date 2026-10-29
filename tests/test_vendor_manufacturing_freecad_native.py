"""Fresh in-memory FreeCAD geometry only; no CAD/file reads or exports.

Run mainrun_suite() inside the separately supervised pinned FreeCAD runtime.
Stdlib assertions/JSON only; pytest, GUI, mutable preferences and fixtures from
the physical corpus are not needed. This module does no native work on import.
"""
from __future__ import annotations

import json
import math
import time


def _profile(points, arc_midpoints):
    import FreeCAD as App
    import Part
    edges = []
    for index, (a, b) in enumerate(zip(points, points[1:] + points[:1])):
        pa, pb = App.Vector(*a, 0.), App.Vector(*b, 0.)
        if index in arc_midpoints:
            edges.append(Part.Arc(pa, App.Vector(*arc_midpoints[index], 0.), pb).toShape())
        else:
            edges.append(Part.makeLine(pa, pb))
    solid = Part.Face(Part.Wire(edges)).extrude(App.Vector(0., 0., 20.))
    assert solid.ShapeType == "Solid" and solid.isValid()
    return solid


def _l_shape(*, hole=False, sharp_return=False):
    root2 = math.sqrt(2.)
    points = [(4., -30.), (4., 0.), (0., 4.), (-10., 4.), (-10., 5.), (0., 5.), (5., 0.), (5., -30.)]
    if sharp_return:
        points = [(4., -31.), (4., 0.), (0., 4.), (-10., 4.), (-10., 5.),
                  (0., 5.), (5., 0.), (5., -30.), (12., -30.), (12., -31.)]
    shape = _profile(points, {1: (4. / root2, 4. / root2), 5: (5. / root2, 5. / root2)})
    if hole:
        import FreeCAD as App
        import Part
        cutter = Part.makeCylinder(1., 3., App.Vector(3., -15., 10.), App.Vector(1., 0., 0.))
        shape = shape.cut(cutter)
        assert shape.isValid() and len(shape.Solids) == 1
    return shape


def _angled_l_shape(angle_deg, *, hole=False):
    import FreeCAD as App
    import Part
    theta, radius, outer, leg_a, leg_b, width = math.radians(angle_deg), 2., 3.5, 30., 20., 40.
    radial = (math.cos(theta), math.sin(theta))
    tangent = (-math.sin(theta), math.cos(theta))
    inner_end = tuple(radius * v for v in radial)
    outer_end = tuple(outer * v for v in radial)
    points = [(radius, -leg_a), (radius, 0.), inner_end,
        tuple(inner_end[i] + leg_b * tangent[i] for i in (0, 1)),
        tuple(outer_end[i] + leg_b * tangent[i] for i in (0, 1)), outer_end,
        (outer, 0.), (outer, -leg_a)]
    half = theta / 2.
    # _profile first extrudes20; re-extrude the very same native section toW40.
    intermediate = _profile(points, {1: (radius * math.cos(half), radius * math.sin(half)),
                                    5: (outer * math.cos(half), outer * math.sin(half))})
    endcap = next(face for face in intermediate.Faces if face.Surface.TypeId == "Part::GeomPlane"
        and abs(face.Surface.Axis.z) > .999999 and abs(face.CenterOfMass.z) < 1e-8)
    shape = endcap.extrude(App.Vector(0., 0., width))
    assert shape.isValid() and shape.ShapeType == "Solid"
    if hole:
        shape = shape.cut(Part.makeCylinder(1., 3., App.Vector(1.5, -15., 20.), App.Vector(1., 0., 0.)))
        assert shape.isValid()
    return shape


def _u_shape():
    root2 = math.sqrt(2.)
    points = [(-4., 20.), (-4., 0.), (0., -4.), (20., -4.), (24., 0.), (24., 20.),
              (25., 20.), (25., 0.), (20., -5.), (0., -5.), (-5., 0.), (-5., 20.)]
    return _profile(points, {1: (-4. / root2, -4. / root2), 3: (20. + 4. / root2, -4. / root2),
                            7: (20. + 5. / root2, -5. / root2), 9: (-5. / root2, -5. / root2)})


def _l_roots(shape):
    return [i for i, face in enumerate(shape.Faces) if face.Surface.TypeId == "Part::GeomPlane"
            and abs(abs(face.Surface.Axis.x) - 1.) < 1e-10 and face.Area > 500.]


def _u_root(shape):
    return next(i for i, face in enumerate(shape.Faces) if face.Surface.TypeId == "Part::GeomPlane"
                and abs(face.Surface.Position.y + 4.) < 1e-8 and abs(face.Area - 400.) < 1e-7)


def _assert_draft(result, count, area, allowance, inner_radius=4.):
    assert result["status"] == "DRAFT_COMPUTED"
    assert result["release_allowed"] is False and result["manufacturing_ready"] is False
    assert result["units_verified"] is False and result["source_sha256"] is None
    assert len(result["bend_ledger"]) == count
    assert abs(result["flat_face"].Area - area) < 1e-6
    assert abs(result["flat_face"].BoundBox.ZMin) < 1e-7
    assert abs(result["flat_face"].BoundBox.ZMax) < 1e-7
    assert result["flat_face"].isValid() and len(result["flat_face"].Faces) == 1
    for bend in result["bend_ledger"]:
        assert type(bend["source_face_index"]) is int
        assert type(bend["paired_source_face_index"]) is int
        assert abs(bend["bend_allowance_mm"] - allowance) < 1e-7
        assert abs(bend["inner_radius_mm"] - inner_radius) < 1e-7
        assert bend["bend_direction"] in {"up", "down"}
        assert abs(bend["line"].BoundBox.ZMin) < 1e-7
    proof = result["coverage_proof"]
    assert proof["all_native_faces_accounted"] and proof["all_native_bend_cylinders_paired"]
    assert proof["tree_skin_graph"] and proof["no_cleanup_or_curve_approximation"]
    assert proof["patch_overlap_checked"] and proof["native_patch_and_cutout_coverage_checked"]
    assert proof["independent_neutral_area_oracle_checked"]
    assert proof["original_native_face_edge_vertex_order_preserved"]
    assert proof["no_extraneous_or_duplicate_wrapper_topology"]
    assert abs(proof["independent_neutral_area_mm2"] - area) < 1e-6
    covered = set().union(*(set(proof[name]) for name in ("selected_source_face_indices",
        "opposite_source_face_indices", "cap_source_face_indices", "hole_wall_source_face_indices")))
    assert covered == set(range(proof["native_face_count"]))


def _run_cases(passed, failed):
    import FreeCAD as App
    import Part
    import SheetMetalNewUnfolder as upstream
    from backend import vendor_manufacturing_freecad as adapter
    def case(name, run):
        try:
            run()
        except Exception as error:
            failed.append({"case": name[:160], "exception_type": type(error).__name__,
                           "reason": str(error)[:400]})
        else:
            passed.append(name[:160])
    def refuses(call, expected=None):
        try:
            call()
        except adapter.ManufacturingUnfoldUnavailable as error:
            if expected is not None:
                assert expected in str(error), (expected, str(error))
        else:
            raise AssertionError("unavailable was required")
    expected_lengths = {30.: 51.3613568166, 90.: 54.0840704497, 135.: 56.1261056745}
    for angle, expected_length in expected_lengths.items():
        allowance = math.radians(angle) * (2. + .4 * 1.5)
        assert abs(50. + allowance - expected_length) < 5e-11
        for hole in (False, True):
            shape = _angled_l_shape(angle, hole=hole)
            roots = _l_roots(shape)
            assert len(roots) == 2
            area = 40. * (50. + allowance) - (math.pi if hole else 0.)
            for root in roots:
                case(f"angle{angle}_hole{hole}_root{root}", lambda shape=shape, root=root, area=area,
                     allowance=allowance: _assert_draft(adapter.compensated_unfold(shape, root, 1.5, .4),
                                                        1, area, allowance, 2.))
    for k in (.2, .4, .7):
        for hole in (False, True):
            shape = _l_shape(hole=hole)
            roots = _l_roots(shape)
            assert len(roots) == 2
            allowance = math.pi / 2. * (4. + k)
            area = 20. * (40. + allowance) - (math.pi if hole else 0.)
            for root in roots:
                def run(shape=shape, root=root, k=k, area=area, allowance=allowance):
                    result = adapter.compensated_unfold(shape, root, 1., k)
                    _assert_draft(result, 1, area, allowance)
                    assert result["coverage_proof"]["root_source_face_index"] == root
                    assert bool(result["coverage_proof"]["hole_wall_source_face_indices"]) == hole
                case(f"L_k{k}_hole{hole}_root{root}", run)
    shape = _u_shape()
    root = _u_root(shape)
    allowance = math.pi / 2. * 4.4
    case("two_bend_U", lambda: _assert_draft(adapter.compensated_unfold(shape, root, 1., .4),
        2, 20. * (60. + 2. * allowance), allowance))
    base = _l_shape(hole=True)
    root = _l_roots(base)[0]
    placement = App.Placement(App.Vector(43., -21., 15.), App.Rotation(App.Vector(1., 2., 3.), 71.))
    posed = base.transformed(placement.toMatrix())
    case("posed_hole_L", lambda: _assert_draft(adapter.compensated_unfold(posed, root, 1., .4),
        1, 20. * (40. + allowance) - math.pi, allowance))
    mirror_source = _l_shape(hole=True)
    before_mirror = (mirror_source.Volume, tuple(face.hashCode() for face in mirror_source.Faces),
                     mirror_source.BoundBox.XMin, mirror_source.BoundBox.XMax)
    mirrored = mirror_source.copy()
    mirror_result = mirrored.mirror(App.Vector(0., 0., 0.), App.Vector(1., 0., 0.))
    if mirror_result is not None:
        mirrored = mirror_result
    mirror_root = _l_roots(mirrored)[0]
    def mirrored_case():
        _assert_draft(adapter.compensated_unfold(mirrored, mirror_root, 1., .4),
                      1, 20. * (40. + allowance) - math.pi, allowance)
        assert before_mirror == (mirror_source.Volume, tuple(face.hashCode() for face in mirror_source.Faces),
                                 mirror_source.BoundBox.XMin, mirror_source.BoundBox.XMax)
    case("mirrored_hole_L_input_immutable", mirrored_case)
    base = _l_shape()
    root = _l_roots(base)[0]
    for value in (None, math.nan, math.inf, -.1, 1.1, True, "0.4", 10 ** 400):
        case(f"invalid_K_{value}", lambda value=value: refuses(
            lambda: adapter.compensated_unfold(base, root, 1., value), "EXPLICIT_ANSI_K_FACTOR_REQUIRED"))
    for value in (None, math.nan, 0., -.5, True, "1.0", 10 ** 400):
        case(f"invalid_thickness_{value}", lambda value=value: refuses(
            lambda: adapter.compensated_unfold(base, root, value, .4), "EXPLICIT_POSITIVE_THICKNESS_REQUIRED"))
    case("thickness_mismatch", lambda: refuses(lambda: adapter.compensated_unfold(base, root, 1.1, .4),
                                               "ESTIMATED_THICKNESS_MISMATCH"))
    case("invalid_root", lambda: refuses(lambda: adapter.compensated_unfold(base, 10000, 1., .4),
                                         "INVALID_ROOT_FACE_INDEX"))
    cylindrical = next(i for i, face in enumerate(base.Faces) if face.Surface.TypeId == "Part::GeomCylinder")
    case("cylindrical_root", lambda: refuses(lambda: adapter.compensated_unfold(base, cylindrical, 1., .4),
                                              "PLANAR_ROOT_REQUIRED"))
    case("open_face", lambda: refuses(lambda: adapter.compensated_unfold(Part.makePlane(10., 10.), 0, 1., .4),
                                      "ONE_VALID_SOLID_REQUIRED"))
    compound = Part.makeCompound([base, base.translated(App.Vector(100., 0., 0.))])
    case("multiple_solids", lambda: refuses(lambda: adapter.compensated_unfold(compound, 0, 1., .4),
                                            "ONE_VALID_SOLID_REQUIRED"))
    # No geometry may be silently filtered from an otherwise single-solid
    # wrapper. Existing boolean-cut hole and rigid-pose fixtures stay unchanged.
    stray_face = Part.makePlane(5., 5., App.Vector(200., 0., 0.))
    stray_wire = Part.Wire([Part.makeLine(App.Vector(200., 0., 0.), App.Vector(205., 0., 0.))])
    stray_vertex = Part.Vertex(App.Vector(200., 0., 0.))
    for kind, extra in (("face", stray_face), ("wire", stray_wire), ("vertex", stray_vertex)):
        wrapped = Part.makeCompound([base, extra])
        case(f"single_solid_plus_stray_{kind}", lambda wrapped=wrapped: refuses(
            lambda: adapter.compensated_unfold(wrapped, root, 1., .4)))
    # FreeCAD's compound builder deduplicates repeated references to the SAME
    # native Solid. A deep native copy is required to construct two distinct
    # solids occupying identical coordinates; certify that fixture population.
    duplicate = Part.makeCompound([base, base.copy()])
    def duplicate_solid_case():
        children = duplicate.childShapes()
        assert len(children) == 2 and len(duplicate.Solids) == 2
        assert all(child.ShapeType == "Solid" for child in children)
        assert not children[0].isSame(children[1])
        refuses(lambda: adapter.compensated_unfold(duplicate, root, 1., .4),
                "ONE_VALID_SOLID_REQUIRED")
    case("distinct_same_coordinates_solid_wrapper", duplicate_solid_case)
    wrapped = Part.makeCompound([base])
    case("direct_single_solid_wrapper", lambda: _assert_draft(
        adapter.compensated_unfold(wrapped, root, 1., .4), 1,
        20. * (40. + math.pi / 2. * 4.4), math.pi / 2. * 4.4))
    class ReorderedWrapper:
        ShapeType, Solids = "Compound", [base]
        Faces, Edges, Vertexes = list(reversed(base.Faces)), base.Edges, base.Vertexes
        def isValid(self):
            return True
        def childShapes(self):
            return [base]
    case("reordered_original_face_table_refuses", lambda: refuses(
        lambda: adapter._normalize_single_solid(ReorderedWrapper(), root),
        "WRAPPER_NATIVE_TABLE_ORDER_MISMATCH"))
    class ReorientedWrapper(ReorderedWrapper):
        Faces = list(base.Faces)
    ReorientedWrapper.Faces[0].reverse()
    case("reoriented_original_face_table_refuses", lambda: refuses(
        lambda: adapter._normalize_single_solid(ReorientedWrapper(), root),
        "WRAPPER_NATIVE_TABLE_ORDER_MISMATCH"))
    sharp = _l_shape(sharp_return=True)
    case("sharp_return_refuses", lambda: refuses(lambda: adapter.compensated_unfold(sharp, _l_roots(sharp)[0], 1., .4)))
    originals = upstream.build_graph_of_tangent_faces, upstream.unroll_cylinder, App.Console.PrintWarning
    def unroll_failure(*args, **kwargs):
        raise RuntimeError("authored synthetic unroll refusal")
    upstream.unroll_cylinder = unroll_failure
    try:
        case("upstream_warning_is_fatal", lambda: refuses(lambda: adapter.compensated_unfold(base, root, 1., .4),
                                                          "UPSTREAM_WARNING"))
        assert upstream.unroll_cylinder is unroll_failure
        assert upstream.build_graph_of_tangent_faces is originals[0]
        assert App.Console.PrintWarning is originals[2]
    finally:
        upstream.unroll_cylinder = originals[1]
    def cyclic_graph(native, selected):
        graph = originals[0](native, selected)
        planes = [i for i in graph if native.Faces[i].Surface.TypeId == "Part::GeomPlane"]
        graph.add_edge(planes[0], planes[1], label=0)
        return graph
    upstream.build_graph_of_tangent_faces = cyclic_graph
    try:
        case("cyclic_skin_graph_refuses", lambda: refuses(lambda: adapter.compensated_unfold(base, root, 1., .4),
                                                          "NON_TREE_SKIN_GRAPH"))
        assert upstream.build_graph_of_tangent_faces is cyclic_graph
        assert upstream.unroll_cylinder is originals[1] and App.Console.PrintWarning is originals[2]
    finally:
        upstream.build_graph_of_tangent_faces = originals[0]
    assert (upstream.build_graph_of_tangent_faces, upstream.unroll_cylinder, App.Console.PrintWarning) == originals
    return {"freecad_version": list(App.Version()), "upstream_commit": adapter.UPSTREAM_COMMIT}


def mainrun_suite():
    """Bounded CLI seam with durable per-case metadata, including failures.

    Caller enforces wall/RSS and exclusive native-job caps. Native process kills
    remain supervisor failures; caught assertions/refusals retain earlier cases.
    """
    start, passed, failed, metadata = time.monotonic(), [], [], {}
    try:
        metadata = _run_cases(passed, failed)
    except Exception as error:
        failed.append({"case": "suite_setup_or_postcondition", "exception_type": type(error).__name__,
                       "reason": str(error)[:400]})
    return {"status": "SYNTHETIC_SUITE_FAILED" if failed else "SYNTHETIC_SUITE_PASSED",
            "passed": len(passed), "cases": passed, "failed": len(failed), "failed_cases": failed,
            "elapsed_seconds": time.monotonic() - start, **metadata,
            "physical_inputs_read": False, "manufacturing_release_evidence": False}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = mainrun_suite()
    payload = json.dumps(result, allow_nan=False)
    if args.result is not None:
        # The supervisor supplies its own fresh temporary receipt path. This is
        # metadata transport only; no geometry/CAD files are serialized.
        with args.result.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload)
    raise SystemExit(1 if result["failed"] else 0)
