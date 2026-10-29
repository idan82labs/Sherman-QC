"""Factory-to-measurement integration for the native feature bridge V1 (V10 repair of the V9 MG defect).

Joins the REAL native factories (both accepted branches: r0 `cad_guided_native_features` and v1
`cad_guided_native_features_v1`) on a tiny independently constructed OCP bend to the REAL bridge, the
REAL context measurement V4 / kernel V3 / exact cover planner V1 / support cover V5 / publication
validator / contact projection V3 with independently constructed analytic scan rows, and proves that
the ACTUAL successor dispatch (`inspect_cad_guided_mesh_v18`) calls the bridge on the real factory
output before any costly registration.

Declared substitutes (labelled, not mocks of the audited seams): an identity registration fixture
and a keep-all surface-role selector in the factory-to-measurement case; in the dispatch case the
scan/topology/CAD-sampling/registration/cover boundaries are the engine-integration doubles. No
factory, bridge, crop, fit, cover or validator seam is mocked in the measurement case. This is
factory-to-measurement integration, not a full byte-workflow or registration qualification, and
it claims no physical accuracy. The analytic scan rows are generated FROM the extracted feature's
own nominal frame (radius, sweep, origin, axis, start radial, span), so the positive case is
FACTORY-CONDITIONED TRANSPORT of the bridged feature into the measurement chain, not independent
geometry truth about the CAD solid.

Named regression retained: a 1 mm skin leaves outer-arc rows inside the kernel's axis-side arc
filter on this r 5 / 90 deg geometry (nominal isolation bound 68.4 deg < 90 deg), so the free-circle
fit refuses ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL; the fixture's 2.5 mm skin (bound 97.5 deg)
is measurable. That is a documented kernel limitation, not something the bridge changes.
"""
import copy
from dataclasses import asdict
import importlib
import math
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import pytest

from .test_cad_guided_v11_connected_chain_v1 import (_mock_roles, _registration, _workflow, _project, _score,
                                                     _support_policy, SCAN_SHA)
from .test_cad_guided_completion_v1_engine_integration_v18 import (_byte_arrays, _registration as _plumbing_registration,
                                                                   _policy, SCAN_BYTES)

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("v1", "r0")


@pytest.fixture(scope="module")
def m():
    saved = {k: v for k, v in sys.modules.items() if k == "backend" or k.startswith("backend.") or k == "scripts" or k.startswith("scripts.")}
    for k in saved:
        del sys.modules[k]
    for name, folder in (("backend", "backend"), ("scripts", "scripts")):
        package = types.ModuleType(name); package.__path__ = [str(ROOT/folder)]; sys.modules[name] = package
    mapping = dict(workflow="cad_guided_mesh_offline_v18", init="cad_guided_plane_initialization_v16",
        bridge="cad_guided_native_feature_bridge_v2", native="cad_guided_native_features",
        native1="cad_guided_native_features_v1", kernel1="cad_guided_measurement", kernel="cad_guided_measurement_v3",
        context="cad_guided_context_measurement_v4", cover="cad_guided_support_cover_measurement_v5",
        mesh="cad_guided_mesh_input_v1", topology="cad_guided_mesh_topology_v4", cad_surface="cad_guided_cad_surface_v2",
        two_plane="cad_guided_two_plane_context_v4", offline2="cad_guided_mesh_offline_v2")
    try:
        ns = types.SimpleNamespace(**{name: importlib.import_module("backend."+path) for name, path in mapping.items()})
        ns.contact = importlib.import_module("scripts.gold11_contact_scoring_v5")
        ns.scripts_v3 = ns.contact
        yield ns
    finally:
        for k in tuple(sys.modules):
            if k == "backend" or k.startswith("backend.") or k == "scripts" or k.startswith("scripts."):
                del sys.modules[k]
        sys.modules.update(saved)


SKIN_THICKNESS_MM = 2.5   # outer arc at 7.5 mm: the kernel's axis-side filter can isolate the inner arc at r 5 / 90 deg
                           # (nominal heuristic bound 2*acos((R-e)/(R+t)) = 97.5 deg > 90 deg); a 1 mm skin would leave
                           # outer-arc rows in the free-circle fit (ARC_NOT_SINGLE_CIRCLE), an honest kernel refusal.


def _invented_bend(sweep_deg=90., *, thickness=SKIN_THICKNESS_MM, flange_length=15.):
    """Independent OCP 2-D strip (inner radius 5 mm, one skin thickness) extruded 30 mm; same construction
    as the native V1 test's fixture, reproduced here so this module owns its geometry."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
    phi, radius = math.radians(sweep_deg), 5.
    polar = lambda r, theta: np.array([r*math.cos(theta), r*math.sin(theta), 0.])
    point = lambda value: gp_Pnt(*map(float, value))
    def line(a, b):
        builder = BRepBuilderAPI_MakeEdge(point(a), point(b)); assert builder.IsDone(); return builder.Edge()
    def arc(r, first, last):
        curve = GC_MakeArcOfCircle(point(polar(r, first)), point(polar(r, (first+last)/2)), point(polar(r, last)))
        assert curve.IsDone(); builder = BRepBuilderAPI_MakeEdge(curve.Value()); assert builder.IsDone(); return builder.Edge()
    a, b = polar(radius, 0.), polar(radius, phi)
    c, d = polar(radius+thickness, 0.), polar(radius+thickness, phi)
    first_ray, last_ray = np.array([0., -1., 0.]), np.array([-math.sin(phi), math.cos(phi), 0.])
    edges = [line(c+flange_length*first_ray, c), arc(radius+thickness, 0., phi), line(d, d+flange_length*last_ray),
             line(d+flange_length*last_ray, b+flange_length*last_ray), line(b+flange_length*last_ray, b),
             arc(radius, phi, 0.), line(a, a+flange_length*first_ray), line(a+flange_length*first_ray, c+flange_length*first_ray)]
    wire = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire.Add(edge); assert wire.IsDone()
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0., 0., 0.), gp_Dir(0., 0., 1.)), wire.Wire(), True); assert face.IsDone()
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0., 0., 30.), True, True); assert prism.IsDone()
    return prism.Shape()


def _owned_step_bytes(shape):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    with tempfile.TemporaryDirectory(prefix="invented-bridge-") as directory:
        path = Path(directory)/"invented_bend.step"
        writer = STEPControl_Writer()
        assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
        assert writer.Write(str(path)) == IFSelect_RetDone
        return path.read_bytes()


def _native_policy(m):
    return m.native.NativeGuidedPolicy(flange_window_mm=(2.5, 12.5), search_margin_mm=3., max_fit_residual_mm=.03,
                                       plane_exclusion_mm=.04, max_tangency_error_mm=.1, fit_model="free_circle")


def _factory(m, branch):
    return m.native1.extract_ocp_guided_features_v1 if branch == "v1" else m.native.extract_ocp_guided_features


def _analytic_rows(feature, *, thickness=SKIN_THICKNESS_MM, spacing=.3, noise=.004, seed=3):
    """Inner surface (arc at the nominal radius plus tangent flanges) and an outer skin at radius +
    thickness, generated in the FEATURE's own frame with the kernel's crop conventions; independent
    of any CAD tessellation."""
    origin, axis, n0 = (np.asarray(v, float) for v in (feature.origin, feature.axis, feature.start_radial))
    k0 = np.cross(axis, n0); phi = math.radians(feature.nominal_sweep_deg); sign = math.copysign(1., phi)
    n1 = math.cos(phi)*n0+math.sin(phi)*k0
    rays = (-sign*k0, sign*np.cross(axis, n1))
    lo, hi = feature.span_mm
    zs = np.arange(lo, hi+1e-9, spacing)
    def layer(r):
        rows = []
        n = max(3, int(round(r*abs(phi)/spacing))+1)
        for t in np.linspace(0., phi, n):
            radial = math.cos(t)*n0+math.sin(t)*k0
            for z in zs:
                rows.append(origin+z*axis+r*radial)
        for contact_radial, ray in ((n0, rays[0]), (n1, rays[1])):
            for d in np.arange(.3, 14., spacing):
                for z in zs:
                    rows.append(origin+z*axis+r*contact_radial+d*ray)
        return rows
    rows = np.array(layer(feature.nominal_radius_mm)+layer(feature.nominal_radius_mm+thickness), float)
    rows = rows+np.random.default_rng(seed).normal(0, noise, rows.shape)
    return np.unique(rows, axis=0)


@pytest.mark.parametrize("branch", BRANCHES)
def test_real_factory_bridge_and_real_measurement_chain_yield_a_supported_generic_carrier(m, monkeypatch, branch):
    shape = _invented_bend()
    extracted = _factory(m, branch)(shape, policy=_native_policy(m), unit_status="DECLARED_MM", allow_ocp=True)
    assert extracted["status"] == "AVAILABLE" and len(extracted["features"]) == 1
    originals = extracted["features"]
    before = [asdict(f) for f in originals]
    ids_before = [id(f) for f in originals]
    metadata_before = copy.deepcopy(extracted["metadata"])
    assert all(type(f) is m.kernel1.CADGuidedFeature for f in originals)          # the unchanged factory still returns V1
    bridged, provenance = m.bridge.bridge_native_features_v2(extracted)
    # Lossless, ordered, class-exact, inputs untouched.
    assert type(bridged) is tuple and len(bridged) == len(originals)
    assert all(type(f) is m.kernel.CADGuidedFeature for f in bridged)
    assert [asdict(f) for f in bridged] == before and [tuple(asdict(f)) for f in bridged] == [tuple(m.bridge.FIELD_NAMES)]*len(bridged)
    assert [asdict(f) for f in originals] == before and [id(f) for f in extracted["features"]] == ids_before
    assert extracted["metadata"] == metadata_before and extracted["features"] is originals
    assert provenance["feature_count"] == 1 and provenance["source_class"].endswith("cad_guided_measurement.CADGuidedFeature")
    assert provenance["target_class"].endswith("cad_guided_measurement_v3.CADGuidedFeature")
    assert provenance["features"][0]["feature_id"] == originals[0].feature_id
    assert provenance["features"][0]["nominal_source"] == originals[0].nominal_source
    assert provenance["features"][0]["field_digest_sha256"] == m.bridge._field_digest({k: getattr(originals[0], k) for k in m.bridge.FIELD_NAMES})
    assert provenance["authority"].startswith("NO_MATERIAL_SIDE") and provenance["truncation"].startswith("NEVER")
    assert provenance["native_metadata_retained_unchanged"] is True and provenance["native_status"] == "AVAILABLE"
    # Real measurement chain on independently constructed analytic rows in the feature's frame.
    chain = types.SimpleNamespace(context=m.context, cover=m.cover, kernel=m.kernel, contact=m.contact)
    points = _analytic_rows(bridged[0])
    prepared = _mock_roles(chain, monkeypatch, points)           # declared keep-all role substitute
    registration = _registration(chain, points)                   # declared identity registration fixture
    measured = m.cover.inspect_support_cover_context_windows_v5(points, list(bridged), registration=registration,
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert measured["status"] == "PARTIAL", measured["reasons"]
    assert "INVALID_SUPPORT_COVER_FEATURES" not in str(measured["reasons"])
    receipt = m.cover.validate_support_cover_publication_v5(measured, raw_points=points, scan_sha256=SCAN_SHA)
    assert receipt["cover_plan_revalidated"] is True
    groups = [g for g in measured["support_groups"] if g["status"] == "MEASURED_CONDITIONAL"]
    assert len(groups) == 1 and groups[0]["finite_support"]["status"] == "SUPPORTED"
    group = groups[0]
    assert abs(group["measurements"]["measured_radius_mm"]-5.) < .05
    assert group["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    assert group["inner_face_claim_eligible"] is False and group["carrier_role"] == m.kernel.GENERIC_CARRIER_ROLE
    assert all(row["canonical_radius_deviation_mm"] is None for row in group["associations"])
    assert "radius_deviation_mm" not in group["measurements"]
    assert measured["observed_bend_count"] is None and measured["conformance"] == "INDETERMINATE"
    window = measured["source_context_result"]["context_windows"][0]
    assert window["feature_id"] == originals[0].feature_id and window["status"] == "MEASURED_CONTEXT"
    carrier = measured["predictions"][0]
    assert carrier["carrier_role"] == m.kernel.GENERIC_CARRIER_ROLE
    projection = _project(chain, _workflow(measured, receipt, registration))
    assert projection["disposition"] == "SCORED_CONDITIONAL", projection["reason"]
    assert projection["predictions"][0]["line_start"] == carrier["line_start"]
    assert projection["predictions"][0]["line_end"] == carrier["line_end"]
    assert projection["diagnostics"]["inner_face_eligible_carrier_count"] == 0
    assert projection["diagnostics"]["identity_unresolved_carrier_count"] == 1
    assert _score(chain, projection, carrier)["control"]["geometry"]["true_positive"] == 1   # transport only


@pytest.mark.parametrize("branch", BRANCHES)
def test_frozen_v1_objects_handed_directly_to_cover_v3_reproduce_the_v9_refusal(m, monkeypatch, branch):
    extracted = _factory(m, branch)(_invented_bend(), policy=_native_policy(m), unit_status="DECLARED_MM", allow_ocp=True)
    chain = types.SimpleNamespace(context=m.context, cover=m.cover, kernel=m.kernel, contact=m.contact)
    points = _analytic_rows(extracted["features"][0])
    prepared = _mock_roles(chain, monkeypatch, points)
    refused = m.cover.inspect_support_cover_context_windows_v5(points, list(extracted["features"]),
        registration=_registration(chain, points), scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.),
        surface_roles=prepared)
    assert refused["status"] == "UNAVAILABLE" and refused["predictions"] == [] and refused["support_groups"] == []
    assert any("INVALID_SUPPORT_COVER_FEATURES" in reason for reason in refused["reasons"])


def _construction_rows(*, thickness=SKIN_THICKNESS_MM, spacing=.3, noise=.004, seed=5, radius=5., sweep_deg=90., height=30.):
    """Scan rows generated from the INDEPENDENT construction constants of `_invented_bend` (inner arc of
    the given radius from +x to +y about +z, tangent flanges along -y at the +x contact and along -x at
    the +y contact, extruded 0..height along +z, outer skin at radius + thickness), never from any
    extracted feature. Used to check the extractor's published frame conventions against geometry it
    did not produce."""
    phi = math.radians(sweep_deg)
    zs = np.arange(.15, height-.15+1e-9, spacing)
    def layer(r):
        rows = []
        n = max(3, int(round(r*phi/spacing))+1)
        for th in np.linspace(0., phi, n):
            for z in zs:
                rows.append((r*math.cos(th), r*math.sin(th), z))
        for d in np.arange(.3, 14., spacing):
            for z in zs:
                rows.append((r, -d, z))                     # flange at the +x contact, along -y
                rows.append((-d, r, z))                     # flange at the +y contact, along -x
        return rows
    rows = np.array(layer(radius)+layer(radius+thickness), float)
    rows = rows+np.random.default_rng(seed).normal(0, noise, rows.shape)
    return np.unique(rows, axis=0)


@pytest.mark.parametrize("branch", BRANCHES)
def test_independently_constructed_geometry_confirms_the_extractor_frame_through_the_real_chain(m, monkeypatch, branch):
    """Rows come from the construction constants, not the returned feature: a wrong extractor
    origin/axis/radial/sweep-sign/span convention would crop or fit the wrong rows here."""
    extracted = _factory(m, branch)(_invented_bend(), policy=_native_policy(m), unit_status="DECLARED_MM", allow_ocp=True)
    bridged, _ = m.bridge.bridge_native_features_v2(extracted)
    chain = types.SimpleNamespace(context=m.context, cover=m.cover, kernel=m.kernel, contact=m.contact)
    points = _construction_rows()
    prepared = _mock_roles(chain, monkeypatch, points)
    measured = m.cover.inspect_support_cover_context_windows_v5(points, list(bridged), registration=_registration(chain, points),
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert measured["status"] == "PARTIAL", measured["reasons"]
    group = _measured_group = [g for g in measured["support_groups"] if g["status"] == "MEASURED_CONDITIONAL"][0]
    assert abs(group["measurements"]["measured_radius_mm"]-5.) < .05
    assert abs(abs(group["measurements"]["measured_sweep_deg"])-90.) < .5
    axis = np.asarray(group["measurements"]["measured_axis_direction"], float)
    assert abs(abs(axis[2])-1.) < 1e-3                                  # measured axis is the construction's +-z
    carrier = measured["predictions"][0]
    start, end = np.asarray(carrier["line_start"], float), np.asarray(carrier["line_end"], float)
    # The carrier is the measured inner-arc MID-ANGLE generator (not the axis): on the construction it lies at
    # radius 5 and 45 deg, i.e. x == y == 5/sqrt(2), parallel to +-z within the extruded height.
    for point in (start, end):
        assert abs(np.hypot(point[0], point[1])-5.) < .05 and abs(point[0]-point[1]) < .05
    assert min(start[2], end[2]) > -.5 and max(start[2], end[2]) < 30.5 and abs(end[2]-start[2]) > 20.
    assert abs(start[0]-end[0]) < .05 and abs(start[1]-end[1]) < .05
    assert group["inner_face_claim_eligible"] is False and carrier["carrier_role"] == m.kernel.GENERIC_CARRIER_ROLE


def test_actual_bridge_failure_through_the_dispatch_boundary_with_real_factory_output(plumbing17, m, monkeypatch):
    """A real factory result with one wrong-class feature reaches the REAL bridge through V17's own
    dispatch (no injected exception): the bridge refuses, nothing costly runs, metadata is retained."""
    state = plumbing17
    real = m.native1.extract_ocp_guided_features_v1
    def tampered_factory(shape, **kwargs):
        extracted = real(shape, **kwargs)
        wrong = m.kernel.CADGuidedFeature(**asdict(extracted["features"][0]))       # kernel-V3 object where V1 is required
        return dict(extracted, features=(wrong,)+tuple(extracted["features"][1:]))
    monkeypatch.setattr(m.native1, "extract_ocp_guided_features_v1", tampered_factory)
    answer = state.run("v1")
    assert answer["status"] == "UNAVAILABLE" and answer["phase"] == "NATIVE_FEATURE_BRIDGE"
    assert answer["reasons"] == ["NATIVE_FEATURE_BRIDGE_REFUSED:NATIVE_FEATURE_NOT_EXACT_KERNEL_V1_CLASS"]
    assert answer["native_feature_bridge"]["status"] == "REFUSED" and answer["native_features"]["candidate_inventory_complete"] is True
    assert len(state.bridge_calls) == 1 and "register_mock" not in state.events and state.measurement_calls == []
    assert answer["predictions"] == [] and answer["measurements"] is None


def test_one_mm_skin_remains_a_named_kernel_refusal_not_a_bridge_defect(m, monkeypatch):
    extracted = _factory(m, "v1")(_invented_bend(thickness=1.), policy=_native_policy(m), unit_status="DECLARED_MM", allow_ocp=True)
    bridged, _ = m.bridge.bridge_native_features_v2(extracted)
    chain = types.SimpleNamespace(context=m.context, cover=m.cover, kernel=m.kernel, contact=m.contact)
    points = _analytic_rows(bridged[0], thickness=1.)
    prepared = _mock_roles(chain, monkeypatch, points)
    measured = m.cover.inspect_support_cover_context_windows_v5(points, list(bridged), registration=_registration(chain, points),
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert "INVALID_SUPPORT_COVER_FEATURES" not in str(measured["reasons"])          # the bridge is not the refusal
    windows = measured["source_context_result"]["context_windows"]
    assert [w["reasons"] for w in windows] == [["ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL"]]
    assert windows[0]["diagnostics"]["failure_phase"] == "ARC_FIT_AND_TANGENCY" and measured["predictions"] == []
    bound = 2.*math.degrees(math.acos((5.-.04)/(5.+1.)))
    assert bound < 90. < 2.*math.degrees(math.acos((5.-.04)/(5.+SKIN_THICKNESS_MM)))


def test_bridge_limit_is_the_cover_and_context_boundary_of_64_not_the_kernel_entrypoint_cap(m, monkeypatch):
    extracted = _factory(m, "v1")(_invented_bend(), policy=_native_policy(m), unit_status="DECLARED_MM", allow_ocp=True)
    feature = extracted["features"][0]
    assert m.bridge.MAX_BRIDGED_FEATURES == 64 and m.context.MAX_SUPPORT_RECORDS == 512 and m.kernel.MAX_FEATURES == 16
    def variants(count):
        return tuple(m.kernel1.CADGuidedFeature(**dict(asdict(feature), feature_id="native:%d:window:0" % i)) for i in range(count))
    chain = types.SimpleNamespace(context=m.context, cover=m.cover, kernel=m.kernel, contact=m.contact)
    points = _analytic_rows(feature)
    prepared = _mock_roles(chain, monkeypatch, points)
    for count in (17, 64):     # 17 = the V9 CG4BDD native window count that a kernel-cap bridge would newly refuse
        bridged, provenance = m.bridge.bridge_native_features_v2(dict(extracted, features=variants(count)))
        assert len(bridged) == count == provenance["feature_count"] and [f.feature_id for f in bridged] == [f.feature_id for f in variants(count)]
        measured = m.cover.inspect_support_cover_context_windows_v5(points, list(bridged), registration=_registration(chain, points),
            scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
        assert "INVALID_SUPPORT_COVER_FEATURES" not in str(measured["reasons"])      # accepted by the real cover grammar
        assert len(measured["source_context_result"]["context_windows"]) == count
    with pytest.raises(m.bridge.BridgeRefusal, match="NATIVE_FEATURE_SET_OVER_LIMIT_NOT_TRUNCATED"):
        m.bridge.bridge_native_features_v2(dict(extracted, features=variants(65)))
    # The real cover boundary is exactly 64: 65 kernel-V3 objects are refused by the cover's own grammar,
    # so the bridge limit coincides with the consumer, not with the standalone kernel entrypoint cap.
    sixty_five = [m.kernel.CADGuidedFeature(**asdict(f)) for f in variants(65)]
    refused = m.cover.inspect_support_cover_context_windows_v5(points, sixty_five, registration=_registration(chain, points),
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert any("INVALID_SUPPORT_COVER_FEATURES" in reason for reason in refused["reasons"])


def test_bridge_refuses_wrong_class_duplicate_ids_over_limit_and_malformed_results_without_truncation(m):
    extracted = _factory(m, "v1")(_invented_bend(), policy=_native_policy(m), unit_status="DECLARED_MM", allow_ocp=True)
    feature = extracted["features"][0]
    def refuse(result, reason):
        snapshot = copy.deepcopy({k: v for k, v in result.items() if k != "features"})
        with pytest.raises(m.bridge.BridgeRefusal, match=reason):
            m.bridge.bridge_native_features_v2(result)
        assert {k: v for k, v in result.items() if k != "features"} == snapshot
    v2 = m.kernel.CADGuidedFeature(**asdict(feature))
    refuse(dict(extracted, features=(v2,)), "NATIVE_FEATURE_NOT_EXACT_KERNEL_V1_CLASS")
    refuse(dict(extracted, features=(asdict(feature),)), "NATIVE_FEATURE_NOT_EXACT_KERNEL_V1_CLASS")
    refuse(dict(extracted, features=(feature, m.kernel1.CADGuidedFeature(**asdict(feature)))), "DUPLICATE_NATIVE_FEATURE_ID")
    many = tuple(m.kernel1.CADGuidedFeature(**dict(asdict(feature), feature_id="f%d" % i)) for i in range(m.bridge.MAX_BRIDGED_FEATURES+1))
    refuse(dict(extracted, features=many), "NATIVE_FEATURE_SET_OVER_LIMIT_NOT_TRUNCATED")
    exact = tuple(m.kernel1.CADGuidedFeature(**dict(asdict(feature), feature_id="f%d" % i)) for i in range(m.bridge.MAX_BRIDGED_FEATURES))
    bridged, provenance = m.bridge.bridge_native_features_v2(dict(extracted, features=exact))
    assert len(bridged) == m.bridge.MAX_BRIDGED_FEATURES and [f.feature_id for f in bridged] == [f.feature_id for f in exact]
    refuse(dict(extracted, features=list(extracted["features"])), "NATIVE_FEATURES_NOT_A_TUPLE")
    refuse(dict(status="AVAILABLE", features=extracted["features"]), "INVALID_NATIVE_FACTORY_RESULT")
    with pytest.raises(m.bridge.BridgeRefusal, match="INVALID_NATIVE_FACTORY_RESULT"):
        m.bridge.bridge_native_features_v2(list(extracted.items()))
    empty, provenance = m.bridge.bridge_native_features_v2(dict(extracted, status="UNAVAILABLE", features=()))
    assert empty == () and provenance["feature_count"] == 0 and provenance["native_status"] == "UNAVAILABLE"


@pytest.fixture
def plumbing17(m, monkeypatch):
    """The engine-integration doubles around the REAL STEP reader, REAL native factories and REAL bridge."""
    scan, topology, cad = _byte_arrays()
    m17 = types.SimpleNamespace(workflow=m.workflow, init=m.init, context=m.two_plane, mesh=m.mesh, topology=m.topology,
                                native=m.native, native1=m.native1, cad_surface=m.cad_surface, cover=m.cover)
    state = types.SimpleNamespace(m=m17, scan=scan, topology=topology, cad=cad, roles=object(), events=[],
                                  registration_calls=[], measurement_calls=[], bridge_calls=[])
    state.registration = _plumbing_registration(m17, scan, topology, cad)
    state.measured = dict(status="PARTIAL", computation_status="COMPLETE", reasons=[],
        predictions=[dict(invented="sole_top_level_carrier")], source_context_result=dict(predictions=[]),
        observed_bend_count=None, production_use_allowed=False)
    monkeypatch.setattr(m.mesh, "prepare_mesh_scan_v1", lambda **kwargs: (state.events.append("prepare"), state.scan)[1])
    monkeypatch.setattr(m.topology, "parse_triangle_mesh_ply_v4", lambda blob: (state.events.append("parse"), state.topology)[1])
    monkeypatch.setattr(m.cad_surface, "sample_native_cad_surface_v2",
                        lambda shape, **kwargs: (state.events.append("native_sampling_mock"), state.cad)[1])
    def register(*args, **kwargs):
        state.events.append("register_mock"); state.registration_calls.append((args, kwargs))
        return state.registration, state.roles
    monkeypatch.setattr(m.init, "register_measurement_contexts_v16", register)
    original_bridge = m.bridge.bridge_native_features_v2
    def bridge(extracted):
        state.events.append("bridge"); state.bridge_calls.append(extracted)
        return original_bridge(extracted)
    monkeypatch.setattr(m.bridge, "bridge_native_features_v2", bridge)
    def measure(raw, features, **kwargs):
        state.events.append("measure_mock"); state.measurement_calls.append((raw, features, kwargs))
        return state.measured
    monkeypatch.setattr(m.cover, "inspect_support_cover_context_windows_v5", measure)
    monkeypatch.setattr(m.cover, "validate_support_cover_publication_v5",
                        lambda *a, **k: (state.events.append("validate_mock"), dict(fixture="PUBLICATION_BOUNDARY_STUB"))[1])
    state.cad_bytes = _owned_step_bytes(_invented_bend())
    def run(branch, **overrides):
        options = dict(scan_bytes=SCAN_BYTES, cad_bytes=state.cad_bytes, position_error_bound_mm=.025,
            workflow_policy=m.workflow.MeshWorkflowPolicyV18(max_registration_scan_rows=36, cad_sample_count=36,
                                                             native_adapter_version=branch),
            registration_policy=_policy(m17), measurement_policy=_native_policy(m),
            instance_policy=m.cover._v1.SupportPolicy(4., 30., 20., 3), allow_ocp=True,
            connected_measurement_contexts=True, two_plane_context_mode="v4")
        options.update(overrides)
        return m.workflow.inspect_cad_guided_mesh_v18(**options)
    state.run = run
    return state


@pytest.mark.parametrize("branch", BRANCHES)
def test_actual_v17_dispatch_bridges_the_real_factory_output_before_registration_and_hands_v2_objects_to_the_cover(plumbing17, m, branch):
    state = plumbing17
    answer = state.run(branch)
    assert answer["status"] == "PARTIAL" and answer["phase"] == "COMPLETE", answer["reasons"]
    assert answer["implementation"] == m.workflow.WORKFLOW_IMPLEMENTATION and "V18" in answer["implementation"]
    assert state.events.index("bridge") < state.events.index("native_sampling_mock") < state.events.index("register_mock")
    assert len(state.bridge_calls) == 1 and len(state.measurement_calls) == 1
    extracted = state.bridge_calls[0]
    assert all(type(f) is m.kernel1.CADGuidedFeature for f in extracted["features"]) and len(extracted["features"]) == 1
    handed = state.measurement_calls[0][1]
    assert type(handed) is tuple and all(type(f) is m.kernel.CADGuidedFeature for f in handed)
    assert [asdict(f) for f in handed] == [asdict(f) for f in extracted["features"]]
    assert answer["native_feature_bridge"]["feature_count"] == 1
    assert answer["native_feature_bridge"]["features"][0]["feature_id"] == extracted["features"][0].feature_id
    assert answer["native_features"] is extracted["metadata"] or answer["native_features"] == extracted["metadata"]
    assert answer["predictions"] is state.measured["predictions"]


def test_v17_bridge_refusal_returns_before_any_costly_work_with_native_diagnostics_retained(plumbing17, m, monkeypatch):
    state = plumbing17
    def refusing(extracted):
        state.events.append("bridge"); state.bridge_calls.append(extracted)
        raise m.bridge.BridgeRefusal("NATIVE_FEATURE_NOT_EXACT_KERNEL_V1_CLASS")
    monkeypatch.setattr(m.bridge, "bridge_native_features_v2", refusing)
    answer = state.run("v1")
    assert answer["status"] == "UNAVAILABLE" and answer["phase"] == "NATIVE_FEATURE_BRIDGE"
    assert answer["reasons"] == ["NATIVE_FEATURE_BRIDGE_REFUSED:NATIVE_FEATURE_NOT_EXACT_KERNEL_V1_CLASS"]
    assert answer["predictions"] == [] and answer["measurements"] is None
    assert answer["native_feature_bridge"]["status"] == "REFUSED" and answer["native_features"]["candidate_inventory_complete"] is True
    assert "register_mock" not in state.events and "native_sampling_mock" not in state.events and state.measurement_calls == []
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
