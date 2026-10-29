"""Real connected V9 measurement chain on synthetic bend points, no mocked schema success.

context measurement V2 (real kernel V2 fits, real local crop) -> exact cover planner V1 -> support
cover V3 -> an independently constructed V13-grammar workflow envelope -> contact projection V3
(direct) AND the V4 schedule/context wrapper on the same emitted bytes -> the UNCHANGED legacy
matcher/scorer through score_gold11_contact_workflows_v3.

Declared fixture substitutes (boundaries, not geometry): registration evidence is an identity pose
record, and the surface-role selector is a keep-all mock. Every fit, crop, plan, validator, projection
and score is live. The scoring reference of the positive case is COPIED from the emitted carrier, so
the numeric F1 here checks transport and matcher arithmetic on real emitted geometry, NOT independent
localization accuracy against any external truth.
"""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from .test_gold11_schedule_projection_v4 import _payload as _v4_payload, _args as _v4_args

ROOT = Path(__file__).resolve().parents[1]
SCAN_SHA = "a" * 64


@pytest.fixture(scope="module")
def chain():
    saved = {k: v for k, v in sys.modules.items() if k == "backend" or k.startswith("backend.") or k == "scripts" or k.startswith("scripts.")}
    for k in saved:
        del sys.modules[k]
    for name, folder in (("backend", "backend"), ("scripts", "scripts")):
        package = types.ModuleType(name); package.__path__ = [str(ROOT/folder)]; sys.modules[name] = package
    try:
        context = importlib.import_module("backend.cad_guided_context_measurement_v2")
        cover = importlib.import_module("backend.cad_guided_support_cover_measurement_v3")
        kernel = importlib.import_module("backend.cad_guided_measurement_v2")
        contact = importlib.import_module("scripts.gold11_contact_scoring_v3")
        wrapper = importlib.import_module("scripts.gold11_schedule_projection_v4")
        assert wrapper._contact3 is contact
        yield types.SimpleNamespace(context=context, cover=cover, kernel=kernel, contact=contact, wrapper=wrapper)
    finally:
        for k in tuple(sys.modules):
            if k == "backend" or k.startswith("backend.") or k == "scripts" or k.startswith("scripts."):
                del sys.modules[k]
        sys.modules.update(saved)


def _bend(radius=0.8, sweep=90.0, noise=0.004, seed=3, spacing=0.3, thickness=None, outer_only=False, axial=12.0,
          shift=(0., 0., 0.)):
    """Inner surface at r (five nominal angular columns for r 0.8 / 90 deg at 0.3 mm: 0, 22.5, 45, 67.5, 90 deg;
    this is a uniform-column model, not an exact 0.3 mm chord sampler) plus tangent flanges, optionally an
    outer skin at r+t, with independent per-coordinate noise."""
    rng = np.random.default_rng(seed)
    sign = math.copysign(1.0, sweep); phi = math.radians(sweep)
    zs = np.arange(0.0, axial+1e-9, spacing)

    def layer(r):
        out = []
        n = max(3, int(round(r*abs(phi)/spacing))+1)
        for th in np.linspace(0.0, sweep, n):
            for z in zs:
                out.append((r*math.cos(math.radians(th)), r*math.sin(math.radians(th)), z))
        c1 = np.array([r, 0., 0.]); c2 = r*np.array([math.cos(phi), math.sin(phi), 0.])
        ray1 = np.array([0., -sign, 0.]); ray2 = sign*np.array([-math.sin(phi), math.cos(phi), 0.])
        for c, ray in ((c1, ray1), (c2, ray2)):
            for d in np.arange(0.3, 14.0, spacing):
                for z in zs:
                    out.append(tuple(c+d*ray+np.array([0., 0., z])))
        return out
    points = [] if outer_only else layer(radius)
    if thickness:
        points += layer(radius+thickness)
    points = np.array(points, float)+np.asarray(shift, float)
    points = points+rng.normal(0, noise, points.shape)
    return np.unique(points, axis=0)


def _feature(chain, identity="native:0", origin=(0., 0., 0.), delta=.03, excl=.04):
    return chain.kernel.CADGuidedFeature(feature_id=identity, nominal_radius_mm=.8, nominal_sweep_deg=90., origin=origin,
        axis=(0., 0., 1.), start_radial=(1., 0., 0.), span_mm=(0., 12.), flange_window_mm=(2.5, 12.5), search_margin_mm=3.,
        max_fit_residual_mm=delta, plane_exclusion_mm=excl, max_tangency_error_mm=.1, nominal_source="INVENTED_NOMINAL")


def _support_policy(chain, gap=30.):
    return chain.context.SupportPolicy(max_axial_gap_mm=4., max_angular_gap_deg=gap, min_arc_sweep_deg=20., min_local_points=3)


def _sha(points):
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()


def _registration(chain, points):
    mod = chain.context
    r, t = np.eye(3), np.zeros(3)
    candidate = dict(index=0, execution_status="ATTEMPTED", numerical_consumer_status=mod._v3.CONSUMER_COMPLETE,
        numerical_support_policy=mod._v3.SUPPORT_POLICY, distance_model=mod._v3.DISTANCE_MODEL,
        optimizer_objective=mod._v3.SURROGATE, rotation=r.tolist(), translation=t.tolist(),
        conditional_context_support=dict(status="SUPPORTED_CONTEXT", proximity_policy=mod._stages.PROXIMITY,
            accepted_registration=False, pose_sha256=_sha(np.concatenate((r.ravel(), t)))))
    return dict(schema_version="noisy_plane_initialized_finite_registration_v12",
        status="CONDITIONAL_MEASUREMENT_CONTEXTS", transform_scan_to_cad=None,
        evidence=dict(candidate_evaluation_complete=True, numerical_consumer_complete=True,
            accepted_registration=False, connected_measurement_contexts=True, joint_configuration_status="NOT_ESTABLISHED",
            stage_policies=mod._stages.ablation_stage_policies_v1(connected=True),
            input_digests=dict(scan_points_sha256=_sha(points), scan_triangle_raw_rows_sha256="b"*64,
                              position_error_bounds_mm_sha256="c"*64),
            surface_role_preparation=dict(metadata_sha256="d"*64),
            candidates=[candidate], measurement_context_indices=[0], initialization={}))


def _mock_roles(chain, monkeypatch, points):
    """Declared boundary substitute: keep-all surface roles (no cap classification proof)."""
    mod = chain.context

    class Prepared:
        points_sha256 = _sha(points); triangle_raw_rows_sha256 = "b"*64
        position_error_bounds_mm_sha256 = "c"*64; metadata_sha256 = "d"*64; raw_row_count = len(points)
    prepared = Prepared()
    monkeypatch.setattr(mod._roles, "PreparedContextSurfaceRolesV1", Prepared)

    def select(owned, axis, **limits):
        kept, excluded = np.arange(len(points)), np.array([], dtype=np.int64)
        metadata = dict(schema_version="context_surface_roles_v1", coordinate_frame="RAW_SCAN_UNREGISTERED",
            scope="CONDITIONAL_PATCH_NORMAL_ROLE_NOT_PHYSICAL_CAP_CERTIFICATE",
            role_policy="CAP_AND_NO_COMPETING_NONCAP_OR_UNRESOLVED_ALL_INCIDENT_TRIANGLES",
            numeric_status="ENGINEERING_NORMALIZATION_AND_ANGULAR_INWARD_RESERVE_NOT_CERTIFIED",
            points_sha256=prepared.points_sha256, triangle_raw_rows_sha256=prepared.triangle_raw_rows_sha256,
            position_error_bounds_mm_sha256=prepared.position_error_bounds_mm_sha256,
            encoded_patch_metadata_sha256=prepared.metadata_sha256,
            keep_raw_rows_sha256=hashlib.sha256(np.asarray(kept, dtype="<i8").tobytes()).hexdigest(),
            excluded_raw_rows_sha256=hashlib.sha256(np.asarray(excluded, dtype="<i8").tobytes()).hexdigest(),
            raw_row_count=len(points), kept_row_count=len(kept), excluded_row_count=0,
            selected_proposal_evaluation_complete=True, production_use_allowed=False, observed_bend_count=None,
            attempted_triangle_membership_visits=7, attempted_incidence_corner_visits=9,
            disclosure="MOCK_SELECTOR_NO_CAP_CLASSIFICATION_PROOF")
        return dict(keep_raw_rows=kept, excluded_raw_rows=excluded, metadata=metadata)
    monkeypatch.setattr(mod._roles, "select_context_measurement_rows_v1", select)
    return prepared


def _measure(chain, monkeypatch, points, *, features=None, layer_policy=None, gap=30.):
    prepared = _mock_roles(chain, monkeypatch, points)
    registration = _registration(chain, points)
    features = features or [_feature(chain)]
    measured = chain.cover.inspect_support_cover_context_windows_v3(points, features, registration=registration,
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, gap), surface_roles=prepared, layer_policy=layer_policy)
    receipt = chain.cover.validate_support_cover_publication_v3(measured, raw_points=points, scan_sha256=SCAN_SHA)
    return measured, receipt, registration


def _workflow(measured, receipt, registration):
    status = measured["status"]
    return dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status=status, phase="COMPLETE",
        reasons=list(measured["reasons"]), predictions=copy.deepcopy(measured["predictions"]),
        measurements=measured, connected_measurement_contexts=True,
        prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS",
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE", policies={},
        registration=dict(status="CONDITIONAL_MEASUREMENT_CONTEXTS", transform_scan_to_cad=None,
            evidence=dict(accepted_registration=False, joint_configuration_status="NOT_ESTABLISHED",
                measurement_context_indices=[0], initialization={})),
        support_cover_publication_validation=dict(receipt, cover_refit_repeated=False))


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _project(chain, payload):
    blob = _wire(payload)
    return chain.contact.project_gold11_contact_workflow_v3(blob, expected_result_sha256=hashlib.sha256(blob).hexdigest(),
        expected_scan_plane_mode="v7", expected_scan_plane_configuration=None)


def _refusal():
    return dict(status="UNAVAILABLE", phase="SUPERVISOR", reasons=["INVENTED_REFUSAL"], predictions=[],
                observed_bend_count=None, production_use_allowed=False)


def _configuration():
    return dict(mode="composite_v1", seeded_budgets=dict(max_seeds=32, max_seed_selection_triangle_visits=64_000_000,
        max_proposal_raw_row_visits=32_000_000, max_proposal_triangle_visits=64_000_000, max_proposal_edge_visits=96_000_000,
        max_full_support_row_tests=32_000_000, max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000,
        max_fit_evaluations=32, max_retained_array_bytes=268_435_456, max_metadata_bytes=67_108_864,
        max_working_array_bytes=1_073_741_824), composite_budgets=dict(max_qualified_planes=65_552,
        max_triangle_memberships=20_000_000, max_raw_memberships=32_000_000, max_full_support_row_tests=64_000_000,
        max_raw_predicate_visits=48_000_000, max_triangle_predicate_visits=160_000_000, max_edge_predicate_visits=192_000_000,
        max_fit_evaluations=65_584, max_retained_array_bytes=536_870_912, max_metadata_bytes=67_108_864))


def _score(chain, projection, carrier):
    """Eleven invented cases; case-00 gets the real projection. The first reference of case-00 is COPIED from
    the emitted carrier (transport/arithmetic check, not independent localization)."""
    cases = ["case-"+str(i).zfill(2) for i in range(11)]
    families = {case: "family-"+str(i % 6) for i, case in enumerate(cases)}
    references, control, challenger = {}, {}, {}
    for index, case in enumerate(cases):
        count = 7 if index < 5 else 6
        references[case] = [dict(bend_id="ref-"+str(j), line_start=[j*50+500, 0, 0], line_end=[j*50+500, 0, 10]) for j in range(count)]
        control[case] = _refusal(); challenger[case] = _refusal()
    if carrier is not None:
        references[cases[0]][0] = dict(bend_id="ref-0", line_start=carrier["line_start"], line_end=carrier["line_end"])
    control[cases[0]] = projection
    return chain.contact.score_gold11_contact_workflows_v3(reference_records_by_case=references,
        control_results_by_case=control, challenger_results_by_case=challenger, case_family_by_id=families,
        scan_plane_requests_by_arm=dict(control=dict(mode="v7", configuration=None),
                                        challenger=dict(mode="composite_v1", configuration=_configuration())))


def _measured_group(measured):
    return [g for g in measured["support_groups"] if g["status"] == "MEASURED_CONDITIONAL"][0]


def test_two_skin_bend_yields_a_generic_carrier_with_a_two_layer_side_hypothesis_and_a_numeric_transport_f1(chain, monkeypatch):
    points = _bend(thickness=1.5)
    measured, receipt, registration = _measure(chain, monkeypatch, points)
    assert measured["status"] == "PARTIAL", measured["reasons"]
    assert receipt["cover_plan_revalidated"] is True
    group = _measured_group(measured)
    assert group["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"   # side hypothesis only
    assert group["inner_face_claim_eligible"] is False and group["layer_policy"] == measured["source_context_result"]["layer_policy"]
    assert all(row["canonical_radius_deviation_mm"] is None for row in group["associations"])
    assert "radius_deviation_mm" not in group["measurements"] and group["measurements"]["inner_face_claim_eligible"] is False
    assert set(group["fit_policy"]) == set(chain.context.FIT_FIELDS)
    assert abs(group["measurements"]["measured_radius_mm"]-.8) < .05
    assert group["finite_support"]["status"] == "SUPPORTED"
    window = measured["source_context_result"]["context_windows"][0]
    assert window["status"] == "MEASURED_CONTEXT"
    selection = window["diagnostics"]["flanges"][0]["layer_selection"]
    assert selection["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED" and selection["dropped_rows"] > 0
    assert window["diagnostics"]["phases_completed"][-1] == "ARC_FIT_AND_TANGENCY"
    carrier = measured["predictions"][0]
    assert carrier["carrier_role"] == chain.kernel.GENERIC_CARRIER_ROLE
    assert carrier["material_side_status"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
    projection = _project(chain, _workflow(measured, receipt, registration))
    assert projection["disposition"] == "SCORED_CONDITIONAL", projection["reason"]
    assert projection["diagnostics"]["inner_face_eligible_carrier_count"] == 0
    assert projection["diagnostics"]["identity_unresolved_carrier_count"] == 1
    assert projection["predictions"][0]["line_start"] == carrier["line_start"]
    score = _score(chain, projection, carrier)
    control = score["control"]["geometry"]
    assert (control["true_positive"], control["false_positive"], control["false_negative"]) == (1, 0, 70)
    assert control["f1"] == pytest.approx(2*(1/1)*(1/71)/((1/1)+(1/71)))
    assert score["challenger"]["geometry"]["f1"] is None
    assert score["scoring_scope"].startswith("GENERIC_CONDITIONAL_CONTACT_GENERATOR")
    assert score["observed_bend_count"] is None and score["conformance"] == "INDETERMINATE"


def test_single_skin_bend_yields_a_generic_carrier_with_unresolved_identity_still_scored_as_a_contact_line(chain, monkeypatch):
    points = _bend(thickness=None)
    measured, receipt, registration = _measure(chain, monkeypatch, points)
    assert measured["status"] == "PARTIAL", measured["reasons"]
    group = _measured_group(measured)
    assert group["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED" and group["inner_face_claim_eligible"] is False
    assert all(row["canonical_radius_deviation_mm"] is None for row in group["associations"])
    carrier = measured["predictions"][0]
    assert carrier["carrier_role"] == chain.kernel.GENERIC_CARRIER_ROLE and carrier["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED"
    projection = _project(chain, _workflow(measured, receipt, registration))
    assert projection["disposition"] == "SCORED_CONDITIONAL", projection["reason"]
    assert projection["diagnostics"]["identity_unresolved_carrier_count"] == 1 and projection["diagnostics"]["inner_face_eligible_carrier_count"] == 0
    assert projection["predictions"][0]["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED"
    score = _score(chain, projection, carrier)
    assert score["control"]["geometry"]["true_positive"] == 1
    assert score["control"]["case_dispositions"]["case-00"]["diagnostics"]["identity_unresolved_carrier_count"] == 1


def test_outer_only_scan_is_never_labelled_inner(chain, monkeypatch):
    outer = _bend(thickness=1.5, outer_only=True)
    measured, receipt, registration = _measure(chain, monkeypatch, outer)
    for group in measured["support_groups"]:
        assert group["inner_face_claim_eligible"] in (False, None) and "INNER_SELECTED" not in str(group["material_side_status"])
        if group["status"] == "MEASURED_CONDITIONAL":
            assert group["material_side_status"] == "SINGLE_LAYER_IDENTITY_UNRESOLVED"
            assert abs(group["measurements"]["measured_radius_mm"]-2.3) < .1
            assert all(row["canonical_radius_deviation_mm"] is None for row in group["associations"])
    for carrier in measured["predictions"]:
        assert carrier["carrier_role"] == chain.kernel.GENERIC_CARRIER_ROLE


def test_frozen_v8_measurement_policy_refuses_the_same_two_skin_geometry(chain, monkeypatch):
    points = _bend(thickness=1.5)
    frozen = [_feature(chain, delta=.05, excl=.075)]
    measured, receipt, registration = _measure(chain, monkeypatch, points, features=frozen)
    assert measured["predictions"] == [] and measured["support_groups"] == []
    windows = measured["source_context_result"]["context_windows"]
    assert windows and all(w["status"] == "UNAVAILABLE" for w in windows)
    assert all(w["diagnostics"] is not None and w["diagnostics"]["failure_phase"] is not None for w in windows)
    assert measured["status"] == "UNAVAILABLE"
    projection = _project(chain, _workflow(measured, receipt, registration))
    assert projection["disposition"] == "REFUSED" and projection["predictions"] == []   # honest whole-case refusal, retained
    score = _score(chain, projection, None)
    assert score["control"]["geometry"]["true_positive"] == 0 and score["control"]["geometry"]["f1"] is None
    assert score["control"]["case_dispositions"]["case-00"]["disposition"] == "REFUSED"


def test_frozen_5_deg_angular_gap_refuses_the_fixture_support_and_30_deg_admits_it(chain, monkeypatch):
    """Fixture evidence only for the declared support-policy delta: the zero-phase uniform-column model
    has 22.5 deg nominal angular steps, so the per-slab gap cap decides; no physical claim."""
    points = _bend(thickness=1.5)
    refused, receipt, registration = _measure(chain, monkeypatch, points, gap=5.)
    group = _measured_group(refused)
    assert group["finite_support"]["status"] == "UNAVAILABLE"
    assert group["finite_support"]["reasons"] == ["DISCONNECTED_LOCAL_ANGULAR_SUPPORT"] and group["finite_support"]["instance"] is None
    assert refused["predictions"] == [] and abs(group["measurements"]["measured_radius_mm"]-.8) < .05
    projection = _project(chain, _workflow(refused, receipt, registration))
    assert projection["disposition"] == "COMPLETE_NO_LOCALIZED_CARRIERS" and projection["predictions"] == []
    assert _score(chain, projection, None)["control"]["geometry"]["f1"] is None
    admitted, _, _ = _measure(chain, monkeypatch, points, gap=30.)
    instance = _measured_group(admitted)["finite_support"]["instance"]
    assert instance["observation_status"] == "SAMPLE_SUPPORTED_CONDITIONAL"
    assert all(20. <= slab["max_angular_gap_deg"] <= 30. for slab in instance["local_coverage"])
    assert instance["continuous_surface_coverage"] == "NOT_ESTABLISHED"


def test_ordinary_canonical_refusal_coexists_with_a_positive_and_the_whole_publication_validates(chain, monkeypatch):
    points = np.vstack((_bend(thickness=1.5), _bend(thickness=1.5, seed=5, shift=(0., 0., 30.))))
    features = [_feature(chain, "native:0"), _feature(chain, "native:1", origin=(0., 0., 30.))]
    original = chain.context._canonical_fit

    def forced(points_, arc_ids, flange_ids, feature, *args, **kwargs):
        if feature.feature_id == "native:1":
            raise chain.kernel._Unavailable("FORCED_ORDINARY_CANONICAL_REFUSAL")
        return original(points_, arc_ids, flange_ids, feature, *args, **kwargs)
    monkeypatch.setattr(chain.context, "_canonical_fit", forced)
    measured, receipt, registration = _measure(chain, monkeypatch, points, features=features)
    records = {r["associations"][0]["feature_id"]: r for r in measured["source_context_result"]["support_records"]}
    assert records["native:1"]["status"] == "UNAVAILABLE" and records["native:1"]["reasons"] == ["FORCED_ORDINARY_CANONICAL_REFUSAL"]
    assert records["native:1"]["measurements"] is None and records["native:1"]["finite_support"] is None
    for association in records["native:1"]["associations"]:
        assert association["inner_face_claim_eligible"] is records["native:1"]["inner_face_claim_eligible"]
        assert association["canonical_radius_deviation_mm"] is None and association["canonical_sweep_magnitude_deviation_deg"] is None
        assert association["per_context_measurements"]["material_side_status"] == records["native:1"]["material_side_status"]
    assert records["native:0"]["status"] == "MEASURED_CONDITIONAL"
    assert len(measured["predictions"]) == 1 and records["native:0"]["finite_support"]["status"] == "SUPPORTED"
    assert -.5 <= measured["predictions"][0]["line_start"][2] <= 12.5   # the z 0-12 bend, not the refused z 30-42 one
    projection = _project(chain, _workflow(measured, receipt, registration))
    assert projection["disposition"] == "SCORED_CONDITIONAL", projection["reason"]
    assert len(projection["predictions"]) == 1
    assert _score(chain, projection, measured["predictions"][0])["control"]["geometry"]["true_positive"] == 1


def test_disabled_layer_policy_is_the_v1_control_and_grants_no_inner_claim(chain, monkeypatch):
    points = _bend(thickness=1.5)
    disabled = chain.kernel.LayerSelectionPolicyV1(enabled=False)
    measured, receipt, registration = _measure(chain, monkeypatch, points, layer_policy=disabled)
    assert measured["source_context_result"]["layer_policy"]["enabled"] is False
    windows = measured["source_context_result"]["context_windows"]
    assert windows and all(w["status"] == "UNAVAILABLE" and w["reasons"] == ["FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY"] for w in windows)
    assert measured["predictions"] == []
    single = _bend(thickness=None)
    measured, receipt, registration = _measure(chain, monkeypatch, single, layer_policy=disabled)
    group = _measured_group(measured)
    assert group["material_side_status"] == "CONTEXT_ASSUMED_NOT_VERIFIED" and group["inner_face_claim_eligible"] is False
    assert measured["predictions"][0]["carrier_role"] == chain.kernel.GENERIC_CARRIER_ROLE
    assert all(row["canonical_radius_deviation_mm"] is None for row in group["associations"])


def test_emitted_measurement_bytes_transport_through_the_v4_wrapper_equal_to_the_direct_v3_projection(chain, monkeypatch):
    points = _bend(thickness=1.5)
    measured, receipt, registration = _measure(chain, monkeypatch, points)
    value = _v4_payload(schedule="v2", scan="v7", empty=False)
    value.update(status=measured["status"], reasons=list(measured["reasons"]), predictions=copy.deepcopy(measured["predictions"]),
                 measurements=measured, support_cover_publication_validation=dict(receipt, cover_refit_repeated=False))
    blob = _wire(value)
    args = _v4_args(blob, schedule="v2", scan="v7")
    projected = chain.wrapper.project_gold11_schedule_workflow_v4(blob, **args)
    assert projected["disposition"] == "ADMITTED", projected["reason"]
    direct = chain.contact.project_gold11_contact_workflow_v3(blob, expected_result_sha256=args["expected_result_sha256"],
        expected_scan_plane_mode="v7", expected_scan_plane_configuration=None)
    assert direct["disposition"] == "SCORED_CONDITIONAL", direct["reason"]
    assert _wire(projected["contact_projection"]) == _wire(direct)
    assert projected["contact_projection"]["predictions"][0]["line_start"] == measured["predictions"][0]["line_start"]
    assert projected["contact_projection"]["diagnostics"]["inner_face_eligible_carrier_count"] == 0
    assert projected["contact_projection"]["diagnostics"]["identity_unresolved_carrier_count"] == 1
    retained = _wire(projected)
    rebuilt = chain.wrapper.revalidate_gold11_schedule_projection_v4(retained,
        expected_projection_sha256=hashlib.sha256(retained).hexdigest(), result_bytes=blob, **args)
    assert _wire(rebuilt["scorer_input"]) == _wire(direct)
    frozen_v2 = importlib.import_module("scripts.gold11_contact_scoring_v2")
    assert frozen_v2.project_gold11_contact_workflow_v2(blob, expected_result_sha256=args["expected_result_sha256"],
        expected_scan_plane_mode="v7")["disposition"] == "MALFORMED_RESULT"


def test_stale_layer_policy_or_forged_identity_fails_closed_before_scoring(chain, monkeypatch):
    points = _bend(thickness=1.5)
    measured, receipt, registration = _measure(chain, monkeypatch, points)
    payload = _workflow(measured, receipt, registration)
    assert _project(chain, payload)["disposition"] == "SCORED_CONDITIONAL"
    forged = copy.deepcopy(payload)
    forged["measurements"]["support_groups"][0]["finite_support"]["instance"]["material_side_status"] = "VERIFIED_INNER"
    forged["predictions"][0]["material_side_status"] = "VERIFIED_INNER"
    forged["measurements"]["predictions"][0]["material_side_status"] = "VERIFIED_INNER"
    assert _project(chain, forged)["disposition"] == "MALFORMED_RESULT"
    promoted = copy.deepcopy(payload)
    for row in (promoted["measurements"]["support_groups"][0], promoted["measurements"]["support_groups"][0]["measurements"],
                promoted["measurements"]["support_groups"][0]["finite_support"]["instance"], promoted["predictions"][0],
                promoted["measurements"]["predictions"][0]):
        row["inner_face_claim_eligible"] = 1
    assert _project(chain, promoted)["disposition"] == "MALFORMED_RESULT"
    stale = copy.deepcopy(payload)
    stale["measurements"]["support_groups"][0]["layer_policy"]["layer_split_gap_mm"] = .31
    assert _project(chain, stale)["disposition"] == "MALFORMED_RESULT"
    downgraded = copy.deepcopy(payload)
    downgraded["measurements"]["schema"] = "CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2"
    assert _project(chain, downgraded)["disposition"] == "MALFORMED_RESULT"
    with pytest.raises(ValueError):
        tampered = copy.deepcopy(measured)
        tampered["source_context_result"]["support_records"][0]["associations"][0]["per_context_measurements"]["material_side_status"] = "SINGLE_LAYER_IDENTITY_UNRESOLVED"
        chain.cover.validate_support_cover_publication_v3(tampered, raw_points=points, scan_sha256=SCAN_SHA)
