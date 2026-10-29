"""Real connected V10 measurement chain (V9 chain port: context V3, cover V4, contact V4, projection V5) on synthetic bend points, no mocked schema success.

context measurement V4 (real kernel V3 fits, real local crop) -> exact cover planner V1 -> support
cover V3 -> an independently constructed V13-grammar workflow envelope -> contact projection V3
(direct) AND the V4 schedule/context wrapper on the same emitted bytes -> the UNCHANGED legacy
matcher/scorer through score_gold11_contact_workflows_v5.

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

from .test_gold11_schedule_projection_v6 import _payload as _v4_payload, _args as _v4_args

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
        context = importlib.import_module("backend.cad_guided_context_measurement_v4")
        cover = importlib.import_module("backend.cad_guided_support_cover_measurement_v5")
        kernel = importlib.import_module("backend.cad_guided_measurement_v3")
        contact = importlib.import_module("scripts.gold11_contact_scoring_v5")
        wrapper = importlib.import_module("scripts.gold11_schedule_projection_v6")
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
    measured = chain.cover.inspect_support_cover_context_windows_v5(points, features, registration=registration,
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, gap), surface_roles=prepared, layer_policy=layer_policy)
    receipt = chain.cover.validate_support_cover_publication_v5(measured, raw_points=points, scan_sha256=SCAN_SHA)
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
                candidate_evaluation_complete=registration["evidence"]["candidate_evaluation_complete"],
                measurement_context_indices=[0], initialization={})),
        support_cover_publication_validation=dict(receipt, cover_refit_repeated=False))


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _project(chain, payload):
    blob = _wire(payload)
    return chain.contact.project_gold11_contact_workflow_v5(blob, expected_result_sha256=hashlib.sha256(blob).hexdigest(),
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
    return chain.contact.score_gold11_contact_workflows_v5(reference_records_by_case=references,
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
    disabled = chain.kernel.LayerSelectionPolicyV2(enabled=False)
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
    projected = chain.wrapper.project_gold11_schedule_workflow_v6(blob, **args)
    assert projected["disposition"] == "ADMITTED", projected["reason"]
    direct = chain.contact.project_gold11_contact_workflow_v5(blob, expected_result_sha256=args["expected_result_sha256"],
        expected_scan_plane_mode="v7", expected_scan_plane_configuration=None)
    assert direct["disposition"] == "SCORED_CONDITIONAL", direct["reason"]
    assert _wire(projected["contact_projection"]) == _wire(direct)
    assert projected["contact_projection"]["predictions"][0]["line_start"] == measured["predictions"][0]["line_start"]
    assert projected["contact_projection"]["diagnostics"]["inner_face_eligible_carrier_count"] == 0
    assert projected["contact_projection"]["diagnostics"]["identity_unresolved_carrier_count"] == 1
    retained = _wire(projected)
    rebuilt = chain.wrapper.revalidate_gold11_schedule_projection_v6(retained,
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
    downgraded["measurements"]["schema"] = "CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V3"
    assert _project(chain, downgraded)["disposition"] == "MALFORMED_RESULT"
    with pytest.raises(ValueError):
        tampered = copy.deepcopy(measured)
        tampered["source_context_result"]["support_records"][0]["associations"][0]["per_context_measurements"]["material_side_status"] = "SINGLE_LAYER_IDENTITY_UNRESOLVED"
        chain.cover.validate_support_cover_publication_v5(tampered, raw_points=points, scan_sha256=SCAN_SHA)


def _subset_registration(chain, points, *, ledger_ok=True):
    """Initializer-V16 style evidence: two candidates, the second OMITTED by allowance; explicit contract."""
    registration = _registration(chain, points)
    evidence = registration["evidence"]
    first = evidence["candidates"][0]
    omitted = dict(index=1, execution_status="OMITTED", reasons=["OMITTED_BY_LANE_WORK_QUOTA"], allowance_lane="main",
                   allowance_reason="LANE_WORK_QUOTA:main", numerical_consumer_status="NOT_EVALUATED_OMITTED_UNKNOWN", cost=None)
    evidence["candidates"] = [first, omitted]
    evidence.update(candidate_evaluation_complete=False, evaluated_subset_complete=True,
        candidate_omission_status="ALLOWANCE_OMISSIONS_RETAINED_UNKNOWN",
        publication_contract="EVALUATED_ALLOWANCE_SUBSET_EXPLICIT_DISJOINT_DISPOSITION_LEDGER_V1",
        candidate_disposition_ledger=dict(schema="candidate_disposition_ledger_v1", completed=[0], failed=[], expired=[],
                                          omitted=[1], not_attempted=[], selected_count=2) if ledger_ok else None)
    return registration


def test_explicit_subset_registration_contract_is_measured_by_cover_v4_and_scored_while_context_v2_and_a_bare_claim_refuse(chain, monkeypatch):
    points = _bend(thickness=1.5)
    prepared = _mock_roles(chain, monkeypatch, points)
    registration = _subset_registration(chain, points)
    measured = chain.cover.inspect_support_cover_context_windows_v5(points, [_feature(chain)], registration=registration,
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert measured["status"] == "PARTIAL", measured["reasons"]
    receipt = chain.cover.validate_support_cover_publication_v5(measured, raw_points=points, scan_sha256=SCAN_SHA)
    payload = _workflow(measured, receipt, registration)
    payload["registration"] = registration                                # the explicit contract travels with the result
    projection = _project(chain, payload)
    assert projection["disposition"] == "SCORED_CONDITIONAL", projection["reason"]
    assert projection["diagnostics"]["subset_registration_count"] == 1
    assert _score(chain, projection, measured["predictions"][0])["control"]["geometry"]["true_positive"] == 1
    # A bare "incomplete" claim without the declared ledger is refused by cover V4 (context V3) and by contact V4.
    bare = _subset_registration(chain, points, ledger_ok=False)
    refused = chain.cover.inspect_support_cover_context_windows_v5(points, [_feature(chain)], registration=bare,
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert refused["status"] == "UNAVAILABLE" and refused["predictions"] == []
    stale = copy.deepcopy(payload); stale["registration"] = bare
    assert _project(chain, stale)["disposition"] == "MALFORMED_RESULT"
    # The frozen context V2 grammar requires strict whole-set completeness and refuses the altered meaning.
    context2 = importlib.import_module("backend.cad_guided_context_measurement_v2")
    kernel2 = importlib.import_module("backend.cad_guided_measurement_v2")           # the frozen V2 grammar takes kernel-V2 objects
    old = context2.inspect_generated_context_windows_v2(points, [kernel2.CADGuidedFeature(**chain.context.asdict(_feature(chain)))],
        registration=registration, scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared)
    assert old["status"] == "UNAVAILABLE" and any("INCOMPLETE_OR_UNBOUND_CONTEXT_EVIDENCE" in r for r in old["reasons"])


# ---- Codex acceptance family (V11): non-nominal radius, two skins 2.5 mm apart, bridging rows, oblique pose ----

def _acceptance_feature(chain, nominal_radius=2.0, delta=.03, excl=.04):
    return chain.kernel.CADGuidedFeature(feature_id="acceptance:0", nominal_radius_mm=nominal_radius, nominal_sweep_deg=90.,
        origin=(0., 0., 0.), axis=(0., 0., 1.), start_radial=(1., 0., 0.), span_mm=(0., 12.), flange_window_mm=(2.5, 12.5),
        search_margin_mm=3., max_fit_residual_mm=delta, plane_exclusion_mm=excl, max_tangency_error_mm=.1, nominal_source="CONSTRUCTOR_NOMINAL")


def _acceptance_rows(radius=2.06, thickness=2.5, noise=.005, seed=91, spacing=.3, edge_spacing=.3, axial=12.0,
                     drop_arc=False, dent=None, third_layer=False, return_dent=False):
    """Constructor geometry in the CAD frame: inner surface at the SCAN radius (2.06, never the 2.0 nominal), tangent
    flanges, an outer skin at +thickness, cut-edge rows at both axial ends spanning the thickness (bridging rows that
    fill every inter-mode gap below 0.3 mm), bounded deterministic noise. Expectations come from these constants."""
    rng = np.random.default_rng(seed)
    phi = math.pi/2
    zs = np.arange(0.0, axial+1e-9, spacing)
    rows = []
    def layer(r):
        out = []
        n = max(3, int(round(r*phi/spacing))+1)
        for th in np.linspace(0.0, 90.0, n):
            for z in zs:
                out.append((r*math.cos(math.radians(th)), r*math.sin(math.radians(th)), z))
        c1, c2 = np.array([r, 0., 0.]), np.array([0., r, 0.])
        ray1, ray2 = np.array([0., -1., 0.]), np.array([-1., 0., 0.])
        for c, ray in ((c1, ray1), (c2, ray2)):
            for d in np.arange(0.3, 14.0, spacing):
                for z in zs:
                    out.append(tuple(c+d*ray+np.array([0., 0., z])))
        return out
    inner = layer(radius)
    if drop_arc:
        inner = [p for p in inner if not (abs(math.hypot(p[0], p[1])-radius) < 1e-9 and p[0] > 1e-9 and p[1] > 1e-9)]
    rows += inner + layer(radius+thickness)
    if third_layer:
        rows += layer(radius+2*thickness)
    c1, c2 = np.array([radius, 0., 0.]), np.array([0., radius, 0.])
    for c, ray, radial in ((c1, np.array([0., -1., 0.]), np.array([1., 0., 0.])), (c2, np.array([-1., 0., 0.]), np.array([0., 1., 0.]))):
        for d in np.arange(0.3, 14.0, spacing):
            for s in np.arange(edge_spacing, thickness, edge_spacing):
                for z in (0.0, axial):
                    rows.append(tuple(c+d*ray+s*radial+np.array([0., 0., z])))
    rows = np.array(rows, float)
    chosen = np.zeros(0, dtype=int)
    if dent is not None:                                   # displace ORIGINAL inner first-flange rows (not extra rows)
        offset, fraction = dent
        on_flange = (np.abs(rows[:, 0]-radius) < 1e-6) & (rows[:, 1] < -.3)
        idx = np.flatnonzero(on_flange)
        chosen = np.sort(rng.choice(idx, size=int(round(fraction*len(idx))), replace=False))
        rows[chosen, 0] += offset
    rows = rows+rng.uniform(-noise, noise, rows.shape)      # bounded deterministic noise (|e| <= 0.005 mm)
    return (rows, chosen) if return_dent else rows


def _oblique_pose(seed=7):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    angle = .83
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
    return rotation, np.array([37., -23., 11.])


def _registration_posed(chain, raw, rotation, translation):
    """One completed, supported candidate at the declared pose (cad = R raw + t); strict whole-set contract."""
    mod = chain.context
    r, t = np.asarray(rotation, float), np.asarray(translation, float)
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
            input_digests=dict(scan_points_sha256=_sha(raw), scan_triangle_raw_rows_sha256="b"*64,
                              position_error_bounds_mm_sha256="c"*64),
            surface_role_preparation=dict(metadata_sha256="d"*64),
            candidates=[candidate], measurement_context_indices=[0], initialization={}))


def _to_raw(cad_rows, rotation, translation):
    return (np.asarray(cad_rows, float)-np.asarray(translation, float)) @ np.asarray(rotation, float)   # raw = R^T (cad - t)


def _line_distance(points, start, end):
    direction = (end-start)/np.linalg.norm(end-start)
    rel = points-start
    return np.linalg.norm(rel-np.outer(rel @ direction, direction), axis=1)


def _measure_acceptance(chain, monkeypatch, cad_rows, rotation, translation, *, feature=None, policy=None):
    raw = _to_raw(cad_rows, rotation, translation)
    prepared = _mock_roles(chain, monkeypatch, raw)
    registration = _registration_posed(chain, raw, rotation, translation)
    feature = _acceptance_feature(chain) if feature is None else feature
    policy = chain.kernel.LayerSelectionPolicyV2() if policy is None else policy
    measured = chain.cover.inspect_support_cover_context_windows_v5(raw, [feature], registration=registration,
        scan_sha256=SCAN_SHA, instance_policy=_support_policy(chain, 30.), surface_roles=prepared, layer_policy=policy)
    return raw, registration, measured


@pytest.mark.parametrize("pose_seed,shuffle", [(7, False), (11, False), (7, True)])
def test_acceptance_family_measures_the_non_nominal_radius_through_the_real_kernel_context_cover_and_contact_path(chain, monkeypatch, pose_seed, shuffle):
    rotation, translation = _oblique_pose(pose_seed)
    cad_rows = _acceptance_rows()
    if shuffle:
        cad_rows = cad_rows[np.random.default_rng(3).permutation(len(cad_rows))]
    raw, registration, measured = _measure_acceptance(chain, monkeypatch, cad_rows, rotation, translation)
    assert measured["status"] == "PARTIAL", measured["reasons"]
    window = measured["source_context_result"]["context_windows"][0]
    assert window["status"] == "MEASURED_CONTEXT" and window["diagnostics"]["phases_completed"][-1] == "ARC_FIT_AND_TANGENCY"
    for role in window["diagnostics"]["flanges"]:
        selection = role["layer_selection"]
        assert selection["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
        assert selection["bridging_rows"] > 0 and selection["dropped_other_skin_rows"] > 0          # distinct exclusions
        assert abs(selection["observed_layer_gap_mm"]-2.5) < .05 and selection["refit_converged"] is True
        assert role["after_selection"]["status"] == "PASS"
    support = window["diagnostics"]["flange_support"]
    assert support["semantics"].startswith("SKIN_MEMBERSHIP_SUBSET") and all(r["excluded_bridging_rows"] > 0 for r in support["per_role"])
    group = _measured_group(measured)
    radius = group["measurements"]["measured_radius_mm"]
    assert abs(radius-2.06) <= .02 and abs(radius-2.0) > .04                # scan radius, never the 2.0 nominal
    assert abs(abs(group["measurements"]["measured_sweep_deg"])-90.) < .3
    assert group["finite_support"]["status"] == "SUPPORTED" and group["inner_face_claim_eligible"] is False
    receipt = chain.cover.validate_support_cover_publication_v5(measured, raw_points=raw, scan_sha256=SCAN_SHA)
    assert receipt["cover_plan_revalidated"] is True
    # The emitted generic carrier is the constructor's mid-angle generator transported to the raw frame.
    expected = _to_raw(np.array([[2.06/math.sqrt(2), 2.06/math.sqrt(2), 0.], [2.06/math.sqrt(2), 2.06/math.sqrt(2), 12.]]), rotation, translation)
    prediction = measured["predictions"][0]
    endpoints = np.array([prediction["line_start"], prediction["line_end"]])
    assert np.max(_line_distance(endpoints, expected[0], expected[1])) <= .10          # predeclared synthetic limit
    direction, want = endpoints[1]-endpoints[0], expected[1]-expected[0]
    assert math.degrees(math.acos(min(1., abs(direction @ want)/np.linalg.norm(direction)/np.linalg.norm(want)))) <= 1.
    assert abs(np.linalg.norm(direction)-12.) <= .6                                    # sampled support spans the construction span
    # Contact V5 projection + unchanged matcher against the CONSTRUCTED reference line (not the prediction): TP 1.
    payload = _workflow(measured, receipt, registration)
    projection = _project(chain, payload)
    assert projection["disposition"] == "SCORED_CONDITIONAL", projection.get("reason")
    score = _score(chain, projection, dict(line_start=expected[0].tolist(), line_end=expected[1].tolist()))
    assert score["control"]["geometry"]["true_positive"] == 1 and score["control"]["geometry"]["false_positive"] == 0


@pytest.mark.parametrize("control", ["dent_between_skins", "dent_air_side", "dent_excess_fraction", "third_population",
                                     "excess_bridging", "no_arc_support", "thin_skin_mixture"])
def test_acceptance_family_controls_abstain_or_refuse_without_certifying_a_deformed_span(chain, monkeypatch, control):
    """Deformed / adversarial constructions through the same real chain. A displaced patch of ORIGINAL rows either
    stays in its group and is refused by the unchanged gate (air side), or is an explicit, exactly counted, unverified
    exclusion with coverage NOT_ESTABLISHED (between the skins, bounded by the declared fraction), never a whole-surface
    certificate; the remaining constructions abstain or refuse with their exact declared reasons."""
    rotation, translation = _oblique_pose(7)
    policy = chain.kernel.LayerSelectionPolicyV2()
    feature = None
    if control == "dent_between_skins":
        cad_rows, chosen = _acceptance_rows(dent=(.2, .12), return_dent=True)
    elif control == "dent_air_side":
        cad_rows, chosen = _acceptance_rows(dent=(-.2, .12), return_dent=True)
    elif control == "dent_excess_fraction":
        cad_rows, chosen = _acceptance_rows(dent=(.2, .12), return_dent=True)
        policy = chain.kernel.LayerSelectionPolicyV2(max_bridging_row_fraction=.05)       # the exclusion budget is declared, not free
    elif control == "third_population":
        cad_rows = _acceptance_rows(thickness=1.2, third_layer=True)                        # three skins inside the 3 mm crop margin
    elif control == "excess_bridging":
        cad_rows = _acceptance_rows(); policy = chain.kernel.LayerSelectionPolicyV2(max_bridging_row_fraction=.02)
    elif control == "no_arc_support":
        cad_rows = _acceptance_rows(drop_arc=True)
    else:
        cad_rows = _acceptance_rows(radius=5.0, thickness=1.0)           # the thin-skin / large-radius arc-mixture negative stays negative
        feature = _acceptance_feature(chain, nominal_radius=5.0)
    raw, registration, measured = _measure_acceptance(chain, monkeypatch, cad_rows, rotation, translation, feature=feature, policy=policy)
    window = measured["source_context_result"]["context_windows"][0]
    diagnostics = window["diagnostics"]
    if control == "dent_between_skins":
        # Permitted inlier-only measurement with EXPLICIT excluded-support semantics: the displaced original rows lie
        # between the skins, outside both bands, and are published as unassigned rows (not verified cut-edge identity);
        # every crop row is accounted; no displaced row is a measured row; coverage is never established.
        assert measured["status"] == "PARTIAL" and len(measured["predictions"]) == 1
        role0 = diagnostics["flange_support"]["per_role"][0]
        dented_in_crop = _rows_in_crop(chain, raw, rotation, translation, chosen, role=0)
        assert dented_in_crop > 100
        assert role0["excluded_bridging_rows"] >= dented_in_crop and role0["outside_sandwich_member_rows"] == 0
        assert role0["kept_rows"] + role0["excluded_opposite_skin_rows"] + role0["excluded_bridging_rows"] == role0["crop_rows"]
        assert diagnostics["flange_support"]["semantics"].endswith("COVERAGE_NOT_ESTABLISHED")
        group = _measured_group(measured)
        assert group["measurements"]["physical_trim_endpoints"] is None and group["inner_face_claim_eligible"] is False
        assert measured["conformance"] == "INDETERMINATE" and measured["production_use_allowed"] is False
        kept = set(np.asarray(measured["source_context_result"]["support_records"][0]["flange_row_indices"][0]).tolist()) if "flange_row_indices" in measured["source_context_result"]["support_records"][0] else None
        assert kept is None or not (kept & set(chosen.tolist()))
        return
    assert measured["predictions"] == [] and measured["support_groups"] == []
    assert window["status"] != "MEASURED_CONTEXT" and window["reasons"]
    if control == "dent_air_side":
        # The displaced rows are NOT between the skins: they stay in the nearest (inner) group by the membership rule
        # and the UNCHANGED residual gate refuses the dented flange; nothing is trimmed to certify it.
        selection = diagnostics["flanges"][0]["layer_selection"]
        assert selection["identity"] == "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED"
        assert selection["outside_sandwich_member_rows"] == _rows_in_crop(chain, raw, rotation, translation, chosen, role=0) > 100
        assert window["reasons"][0] == "FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL" and diagnostics["failure_phase"] == "PLANE_FIT_ROLE_0"
    elif control == "dent_excess_fraction":
        selection = diagnostics["flanges"][0]["layer_selection"]
        assert selection["identity"] == "TWO_MODES_MEMBERSHIP_UNRESOLVED"
        assert selection["unresolved_reason"] == "BRIDGING_FRACTION_ABOVE_DECLARED_MAXIMUM" and selection["kept_rows"] == selection["input_rows"]
    elif control == "third_population":
        assert diagnostics["flanges"][0]["layer_selection"]["identity"] == "AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED"
        assert diagnostics["flanges"][0]["layer_selection"]["modes"]["ambiguous_third"] is not None
    elif control == "excess_bridging":
        assert diagnostics["flanges"][0]["layer_selection"]["identity"] == "TWO_MODES_MEMBERSHIP_UNRESOLVED"
        assert diagnostics["flanges"][0]["layer_selection"]["unresolved_reason"] == "BRIDGING_FRACTION_ABOVE_DECLARED_MAXIMUM"
    elif control == "no_arc_support":
        assert diagnostics["failure_phase"] in ("ARC_SUPPORT_COUNT", "ARC_FIT_AND_TANGENCY") and "ARC" in window["reasons"][0]
    else:
        assert diagnostics["failure_phase"] == "ARC_FIT_AND_TANGENCY" and window["reasons"][0] == "ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL"


def _rows_in_crop(chain, raw, rotation, translation, chosen, role):
    """Number of the displaced original rows that fall inside the kernel's flange crop for `role`, rebuilt from the
    constructor geometry in the CAD frame (the crop is pose-invariant: the feature is transported with the pose)."""
    cad = np.asarray(rotation, float) @ raw.T
    cad = (cad.T + np.asarray(translation, float))
    feat = _acceptance_feature(chain)
    origin, a0, n0 = (np.asarray(v, float) for v in (feat.origin, feat.axis, feat.start_radial))
    k0 = np.cross(a0, n0); phi = math.radians(feat.nominal_sweep_deg); sign = math.copysign(1., phi)
    n1 = math.cos(phi) * n0 + math.sin(phi) * k0
    radial, ray = ((n0, -sign * k0), (n1, sign * np.cross(a0, n1)))[role]
    local = cad - origin; axial = local @ a0
    in_span = (axial >= feat.span_mm[0] - 1e-9) & (axial <= feat.span_mm[1] + 1e-9)
    from_contact = local - feat.nominal_radius_mm * radial
    along = from_contact @ ray
    crop = in_span & (along >= feat.flange_window_mm[0]) & (along <= feat.flange_window_mm[1]) & (np.abs(from_contact @ radial) <= feat.search_margin_mm)
    return int(crop[np.asarray(chosen)].sum())
