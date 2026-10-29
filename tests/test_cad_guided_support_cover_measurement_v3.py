"""Independent fixed-raw-support measurement/transport qualification source.

Only the V1 discovery function is mocked. Its complete returned ledger is
validated by the ACTUAL frozen V1 validator. Cover planning, canonical scan fit,
finite-support checks and the V2 publication validator remain real. No native
CAD, registration accuracy, context discovery, physical inputs or event count
is qualified here. Root owns a separate60wall/45CPU/1GiB/1MiB execution freeze.
"""

import base64
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"backend/cad_guided_support_cover_measurement_v3.py"
LAYER_POLICY = dict(enabled=True, layer_split_gap_mm=.3, min_layer_rows=12, max_layers_reported=8,
    max_layer_normal_angle_deg=3., max_layer_gap_mm=6., min_projected_interval_overlap=.5, max_layer_rms_mm=.1)
SCAN_SHA = "a"*64
EPSILON = 2e-7  # A fixed numeric check on these invented represented arrays.
OVERLAP_REASON = "AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT"


@pytest.fixture
def mod():
    name = "isolated_support_cover_measurement_v3_test"
    def owned(key):
        return key in (name, "backend") or key.startswith(name+".") or key.startswith("backend.")
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved:
        del sys.modules[key]
    for prefix in (name, "backend"):
        package = types.ModuleType(prefix)
        package.__path__ = [str(ROOT/"backend")]
        sys.modules[prefix] = package
    spec = importlib.util.spec_from_file_location(name+".cad_guided_support_cover_measurement_v3", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in tuple(sys.modules):
            if owned(key):
                del sys.modules[key]
        sys.modules.update(saved)


def _array_sha(value, dtype="<f8"):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _geometry():
    """Independent R5.3 cylinder samples and its two tangent planes,125 rows."""
    radius, heights = 5.3, (0., 3., 6., 9., 12.)
    angles = np.arange(10., 81., 5.)*math.pi/180.
    arc = [(radius*math.cos(a), radius*math.sin(a), z) for z in heights for a in angles]
    first = [(radius, -d, z) for z in heights for d in (3., 5., 7., 9., 11.)]
    second = [(-d, radius, z) for z in heights for d in (3., 5., 7., 9., 11.)]
    return np.array(arc+first+second, dtype=np.float64)


def _frame(index):
    if index == 0:
        return np.eye(3), np.zeros(3)
    if index == 1:
        return np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]), np.array([11., -7., 3.])
    axis = np.array([1., 2., -3.])/math.sqrt(14.)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = math.cos(.47)*np.eye(3)+(1-math.cos(.47))*np.outer(axis, axis)+math.sin(.47)*skew
    return rotation, np.array([-13., 17., 5.])


def _feature(mod, identity="nominal-a", radius=5.):
    return mod._v1.CADGuidedFeature(feature_id=identity, nominal_radius_mm=radius,
        nominal_sweep_deg=90., origin=(0., 0., 0.), axis=(0., 0., 1.),
        start_radial=(1., 0., 0.), span_mm=(0., 12.), flange_window_mm=(2.5, 12.5),
        search_margin_mm=3., max_fit_residual_mm=.05, plane_exclusion_mm=.075,
        max_tangency_error_mm=.1, nominal_source="INVENTED_NOMINAL_DECLARATION_NOT_SCAN_TRUTH",
        fit_model="free_circle")


def _policy(mod, max_gap=4.):
    return mod._v1.SupportPolicy(max_axial_gap_mm=max_gap, max_angular_gap_deg=5.,
        min_arc_sweep_deg=20., min_local_points=3)


def _roles(kind="nested", offset=0):
    def shifted(arc, first, second):
        return {"arc_rows": [i+offset for i in arc],
                "flange_rows": [[i+offset for i in first], [i+offset for i in second]]}
    cover = shifted(range(75), range(75, 100), range(100, 125))
    lower = shifted(range(45), range(75, 90), range(100, 115))
    upper = shifted(range(30, 75), range(85, 100), range(110, 125))
    if kind == "nested": return [upper, cover, lower]
    if kind == "nonnested": return [upper, lower]
    if kind == "single": return [cover]
    if kind == "cross_role":
        conflict = shifted(range(75, 90), range(15), range(100, 115))
        return [cover, conflict]
    raise AssertionError("UNDECLARED_SUPPORT_FIXTURE")


def _filter(points):
    n = len(points)
    binding = dict(points_sha256=_array_sha(points), triangle_raw_rows_sha256="b"*64,
        position_error_bounds_mm_sha256="c"*64, encoded_patch_metadata_sha256="d"*64)
    row = dict(binding, schema_version="context_surface_roles_v1", coordinate_frame="RAW_SCAN_UNREGISTERED",
        scope="CONDITIONAL_PATCH_NORMAL_ROLE_NOT_PHYSICAL_CAP_CERTIFICATE",
        role_policy="CAP_AND_NO_COMPETING_NONCAP_OR_UNRESOLVED_ALL_INCIDENT_TRIANGLES",
        numeric_status="ENGINEERING_NORMALIZATION_AND_ANGULAR_INWARD_RESERVE_NOT_CERTIFIED",
        selected_proposal_evaluation_complete=True, production_use_allowed=False, observed_bend_count=None,
        raw_row_count=n, kept_row_count=n, excluded_row_count=0,
        keep_raw_rows_sha256=_array_sha(np.arange(n), "<i8"),
        excluded_raw_rows_sha256=_array_sha(np.array([], dtype=np.int64), "<i8"),
        excluded_raw_row_bitmap=dict(encoding="BASE64_PACKBITS_LITTLE_ZERO_PADDED", bit_count=n,
            data=base64.b64encode(bytes((n+7)//8)).decode()),
        declaration="INVENTED_KEEP_ALL_SELECTOR_OUTPUT_NOT_ACTUAL_MESH_ROLE_DISCOVERY")
    return binding, row


def _source_ledger(mod, points, features, instance_policy, roles):
    """Legal V1 publication with explicit discovery mocks, never synthetic fit truth."""
    result = mod._v1._base()
    binding, filter_row = _filter(points)
    records, windows = [], []
    for context, raw in enumerate(roles):
        feature = features[context % len(features)]
        policy = mod._v1._fit_policy(feature)
        identity = _json_sha(dict(scan_sha256=SCAN_SHA, arc_rows=raw["arc_rows"],
            flange_rows=raw["flange_rows"], fit_policy=policy, instance_policy=asdict(instance_policy),
            surface_assumption="SINGLE_INNER_FACE_UNVERIFIED",
            canonical_frame="RAW_AXIS_MAX_COMPONENT_POSITIVE_AND_RAW_ROLE_ORDER"))
        ordered = copy.deepcopy(raw["flange_rows"])
        if context % 2: ordered.reverse()
        association = dict(context_index=context, feature_id=feature.feature_id,
            nominal_source=feature.nominal_source, nominal_radius_mm=feature.nominal_radius_mm,
            nominal_sweep_deg=feature.nominal_sweep_deg, ordered_flange_raw_rows=ordered,
            surface_role_filter_index=0, inner_face_claim_eligible=False, per_context_measurements={
                "measured_radius_mm": 111.+context, "mocked_discovery_answer_must_not_be_reused": True,
                "material_side_status": "TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED",
                "inner_face_claim_eligible": False, "carrier_role": "GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC"},
            transform_scan_to_cad=np.eye(4).tolist(), accepted_registration=False)
        records.append(dict(record_id=identity, status="UNAVAILABLE", reasons=[OVERLAP_REASON],
            raw_support=copy.deepcopy(raw), fit_policy=policy, layer_policy=dict(LAYER_POLICY), associations=[association],
            measurements=None, finite_support=None, material_side_status="TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED",
            inner_face_claim_eligible=False, carrier_role="GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC"))
        windows.append(dict(context_index=context, feature_id=feature.feature_id, status="MEASURED_CONTEXT",
            reasons=[], local_input_rows=len(points), pre_role_filter_roi_rows=len(points),
            excluded_cap_like_roi_rows=0, support_record_id=identity, surface_role_filter_index=0,
            material_side_status="TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED",
            inner_face_claim_eligible=False, carrier_role="GENERIC_CONDITIONAL_CONTACT_LINE_SURFACE_IDENTITY_UNRESOLVED_NOT_INNER_ARC"))
    result.update(computation_status="COMPLETE", scan_sha256=SCAN_SHA, raw_points_sha256=_array_sha(points),
        instance_policy=asdict(instance_policy), layer_policy=dict(LAYER_POLICY), support_records=records, context_windows=windows,
        surface_role_source_binding=binding, surface_role_filters=[filter_row],
        planned_window_count=len(windows), conditional_support_record_count=len(records),
        total_local_input_rows=len(points)*len(windows), reasons=["NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS"],
        discovery_fixture="MOCKED_V1_DISCOVERY_ONLY_ALL_ORIGINAL_REFUSALS_RETAINED")
    mod._v1.validate_context_measurement_publication_v2(result, raw_count=len(points),
        scan_sha256=SCAN_SHA, raw_points_sha256=_array_sha(points))
    return result


def _setup(mod, monkeypatch, *, kind="nested", frame=0, radius=5., max_gap=4., two_components=False):
    rotation, translation = _frame(frame)
    points = _geometry()@rotation.T+translation
    roles = _roles(kind)
    if two_components:
        points = np.vstack((points, _geometry()@rotation.T+translation+np.array([30., 0., 0.])))
        roles += _roles("single", 125)
    features, policy = [_feature(mod, radius=radius)], _policy(mod, max_gap)
    source = _source_ledger(mod, points, features, policy, roles)
    state = types.SimpleNamespace(points=points, features=features, instance_policy=policy,
        source=source, registration={"explicit_mock_discovery_boundary": True}, surface_roles=object(),
        discovery_calls=[], fit_calls=[], rotation=rotation, translation=translation)
    def discovery(owned, cloned, *, registration, scan_sha256, instance_policy, surface_roles, layer_policy=None):
        assert type(owned) is np.ndarray and not owned.flags.writeable
        assert _array_sha(owned) == _array_sha(state.points)
        assert registration is state.registration and surface_roles is state.surface_roles
        assert scan_sha256 == SCAN_SHA and asdict(instance_policy) == asdict(state.instance_policy)
        assert [asdict(item) for item in cloned] == [asdict(item) for item in state.features]
        state.discovery_calls.append(owned)
        return copy.deepcopy(state.source)
    real_fit = mod._v1._canonical_fit
    def fit(owned, arc, flanges, feature, instance_policy, identity, material_side_status="CONTEXT_ASSUMED_NOT_VERIFIED"):
        state.fit_calls.append(dict(raw_support=dict(arc_rows=list(arc), flange_rows=copy.deepcopy(flanges)),
            feature=feature, record_id=identity, owned_readonly=not owned.flags.writeable))
        return real_fit(owned, arc, flanges, feature, instance_policy, identity, material_side_status=material_side_status)
    monkeypatch.setattr(mod._v1, "inspect_generated_context_windows_v2", discovery)
    monkeypatch.setattr(mod._v1, "_canonical_fit", fit)
    state.invoke = lambda: mod.inspect_support_cover_context_windows_v3(state.points, state.features,
        registration=state.registration, scan_sha256=SCAN_SHA, instance_policy=state.instance_policy,
        surface_roles=state.surface_roles)
    return state


def _validate(mod, state, result):
    return mod.validate_support_cover_publication_v3(result, raw_points=state.points, scan_sha256=SCAN_SHA)


def _fatal(result):
    assert result["status"] == "UNAVAILABLE" and result["discarded_measurement_prefix"] is True
    assert result["support_groups"] == result["predictions"] == []
    assert result["source_context_result"] is None and result["support_cover_plan"] is None
    assert result["production_use_allowed"] is False and result["observed_bend_count"] is None
    assert result["reasons"] and result["attempted_work"]


def _measured(result):
    assert result["status"] == "PARTIAL" and result["computation_status"] == "COMPLETE", result
    assert len(result["support_groups"]) == 1
    group = result["support_groups"][0]
    assert group["status"] == "MEASURED_CONDITIONAL", group
    assert group["measurements"]["measured_radius_mm"] == pytest.approx(5.3, abs=EPSILON)
    assert group["measurements"]["measured_sweep_magnitude_deg"] == pytest.approx(90., abs=EPSILON)
    return group


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_actual_cover_refit_retains_source_refusals_and_recovers_scan_not_nominal(mod, monkeypatch, frame):
    state = _setup(mod, monkeypatch, frame=frame)
    source_before = copy.deepcopy(state.source)
    result = state.invoke()
    group = _measured(result)
    assert _validate(mod, state, result)["cover_plan_revalidated"] is True
    assert result["source_context_result"] == source_before == state.source
    assert result["source_context_result_sha256"] == _json_sha(source_before)
    assert all(row["status"] == "UNAVAILABLE" and row["reasons"] == [OVERLAP_REASON]
               for row in result["source_context_result"]["support_records"])
    assert result["source_context_result"]["predictions"] == []
    assert len(state.discovery_calls) == 1 and len(state.fit_calls) == 1
    assert state.fit_calls[0]["raw_support"] == group["raw_support"] == _roles()[1]
    assert state.fit_calls[0]["record_id"] == group["group_id"] and state.fit_calls[0]["owned_readonly"]
    assert result["cover_refit_row_visits"] == 125
    assert len(group["member_record_ids"]) == 3 and len(group["associations"]) == 3
    assert sum(row["is_full_cover_witness"] for row in group["associations"]) == 1
    assert all(row["canonical_radius_deviation_mm"] is None and row["inner_face_claim_eligible"] is False
               for row in group["associations"])   # side hypothesis only: no nominal inner deviation
    finite = group["finite_support"]
    assert finite["status"] == "SUPPORTED" and result["predictions"] == [finite["instance"]]
    instance = finite["instance"]
    expected = np.array([[5.3/math.sqrt(2), 5.3/math.sqrt(2), z] for z in (0., 12.)])
    expected = expected@state.rotation.T+state.translation
    actual = np.array([instance["line_start"], instance["line_end"]])
    assert min(np.max(np.abs(actual-expected)), np.max(np.abs(actual-expected[::-1]))) < EPSILON
    assert result["registration_status"] == "NOT_ACCEPTED_CONTEXT_PROPOSALS_ONLY"
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    assert group["measurements"]["physical_trim_endpoints"] is None
    assert group["measurements"]["datum_relative_deviation"] is None


def test_distinct_nominal_aliases_change_only_association_deviations(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    state.features = [_feature(mod, "nominal-a", 5.), _feature(mod, "nominal-b", 9.)]
    state.source = _source_ledger(mod, state.points, state.features, state.instance_policy, _roles())
    result = state.invoke()
    group = _measured(result)
    _validate(mod, state, result)
    assert len(group["associations"]) == 3
    for association in group["associations"]:
        assert association["canonical_radius_deviation_mm"] is None   # nominal aliases change only the retained nominal
    assert group["measurements"]["measured_radius_mm"] == pytest.approx(5.3, abs=EPSILON)
    assert {row["nominal_radius_mm"] for row in group["associations"]} == {5., 9.}
    assert all(row["per_context_measurements"]["measured_radius_mm"] >= 111.
        for record in result["source_context_result"]["support_records"] for row in record["associations"])


def test_member_prior_refusal_does_not_become_independent_member_fit_claim(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    state.source["support_records"][0]["reasons"] = ["INVENTED_MEMBER_FINITE_REFUSAL"]
    result = state.invoke()
    group = _measured(result)
    _validate(mod, state, result)
    assert result["source_context_result"] == state.source
    assert len(state.fit_calls) == 1
    assert result["member_geometry_policy"] == "COMMON_COVER_MODEL_WITNESS_NOT_INDEPENDENT_MEMBER_REFIT_CLAIM"
    assert all(row["source_support_role"] == "ORIGINAL_MEMBER_SUPPORT_NOT_FULL_COVER_UNDER_EVERY_CONTEXT"
               for row in group["associations"])


@pytest.mark.parametrize("kind", ["nonnested", "cross_role"])
def test_failed_component_retains_all_records_and_cannot_select_subset(mod, monkeypatch, kind):
    state = _setup(mod, monkeypatch, kind=kind)
    result = state.invoke()
    assert result["status"] == "UNAVAILABLE" and result["computation_status"] == "COMPLETE"
    assert len(result["support_groups"]) == 1 and len(result["source_context_result"]["support_records"]) == 2
    assert result["support_groups"][0]["cover_record_id"] is None
    assert result["predictions"] == [] and state.fit_calls == []
    assert result["source_context_result"] == state.source
    _validate(mod, state, result)


def test_nonconflicting_component_can_succeed_without_hiding_failed_component(mod, monkeypatch):
    state = _setup(mod, monkeypatch, kind="nonnested", two_components=True)
    result = state.invoke()
    assert result["status"] == "PARTIAL" and len(result["support_groups"]) == 2
    assert sorted(row["status"] for row in result["support_groups"]) == ["MEASURED_CONDITIONAL", "UNAVAILABLE"]
    assert len(result["predictions"]) == 1 and len(state.fit_calls) == 1
    assert len(result["source_context_result"]["support_records"]) == 3
    _validate(mod, state, result)


def test_duplicate_coordinate_cover_refuses_actual_fresh_fit_not_raw_set_plan(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    state.points[10] = state.points[9]
    state.source = _source_ledger(mod, state.points, state.features, state.instance_policy, _roles())
    result = state.invoke()
    assert result["support_cover_plan"]["groups"][0]["status"] == "COVERED"
    assert result["support_groups"][0]["status"] == "UNAVAILABLE"
    assert "DUPLICATE_POSITION" in result["support_groups"][0]["reasons"][0]
    assert result["predictions"] == [] and result["cover_refit_row_visits"] == 125
    _validate(mod, state, result)


def test_cover_finite_refusal_preserves_quantity_but_emits_no_carrier(mod, monkeypatch):
    state = _setup(mod, monkeypatch, max_gap=2.9)
    result = state.invoke()
    group = _measured(result)
    assert group["finite_support"]["status"] == "UNAVAILABLE"
    assert "DISCONNECTED_AXIAL_SUPPORT" in group["finite_support"]["reasons"]
    assert result["predictions"] == []
    _validate(mod, state, result)


def test_source_fatal_guard_never_recovers_even_if_other_fields_look_promising(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    state.source = mod._v1._base()
    state.source.update(discarded_measurement_prefix=True, reasons=["INVENTED_V1_FATAL"],
                        attempted_work={"canonical_refit_row_visits": 99})
    result = state.invoke()
    assert result["status"] == "UNAVAILABLE" and result["computation_status"] == "SOURCE_CONTEXT_REFUSED"
    assert result["source_context_result"] == state.source
    assert result["support_cover_plan"] is None and result["support_groups"] == result["predictions"] == []
    assert state.fit_calls == []
    assert _validate(mod, state, result)["source_refusal"] is True


def test_invalid_v1_publication_is_not_hidden_by_valid_cover_rows(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    state.source["surface_role_filters"][0]["kept_row_count"] -= 1
    result = state.invoke()
    _fatal(result)
    assert state.fit_calls == []
    assert result["discarded_source_context_sha256"] is None
    assert result["discarded_source_record_count"] == 0
    assert _validate(mod, state, result)["fatal_refusal"] is True


def test_plan_cap_refusal_is_complete_refusal_not_geometry_absence(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    monkeypatch.setattr(mod._cover, "MAX_RECORDS", 2)
    result = state.invoke()
    assert result["computation_status"] == "SUPPORT_COVER_PLAN_REFUSED"
    assert result["support_cover_plan"]["status"] == "UNAVAILABLE"
    assert result["support_groups"] == result["predictions"] == [] and state.fit_calls == []
    assert _validate(mod, state, result)["plan_refusal"] is True


def test_exact_refit_row_cap_and_one_over_clear_whole_report(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    monkeypatch.setattr(mod, "MAX_COVER_REFIT_ROW_VISITS", 125)
    result = state.invoke()
    _measured(result)
    _validate(mod, state, result)
    state.fit_calls.clear()
    monkeypatch.setattr(mod, "MAX_COVER_REFIT_ROW_VISITS", 124)
    failed = state.invoke()
    _fatal(failed)
    assert state.fit_calls == []
    assert failed["attempted_work"]["cover_refit_row_visits"] == 125
    assert failed["attempted_work"]["attempted_groups"] == 1
    assert failed["discarded_source_context_sha256"] == _json_sha(state.source)
    assert failed["discarded_source_record_count"] == 3


def test_second_component_ordinary_fit_exception_whole_clears_successful_prefix(mod, monkeypatch):
    state = _setup(mod, monkeypatch, two_components=True)
    real = mod._v1._canonical_fit
    calls = []
    def fit(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2: raise RuntimeError("INVENTED_LATE_FIT_ERROR")
        return real(*args, **kwargs)
    monkeypatch.setattr(mod._v1, "_canonical_fit", fit)
    result = state.invoke()
    _fatal(result)
    assert calls == [True, True]
    assert result["attempted_work"]["cover_refit_row_visits"] == 250
    assert result["attempted_work"]["attempted_groups"] == 2
    assert result["discarded_source_record_count"] == 4
    assert _validate(mod, state, result)["fatal_refusal"] is True


def test_report_byte_cap_exact_then_one_over_preserves_failure_work(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    result = state.invoke()
    _measured(result)
    size = len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode())
    monkeypatch.setattr(mod, "MAX_REPORT_BYTES", size)
    assert state.invoke() == result
    monkeypatch.setattr(mod, "MAX_REPORT_BYTES", size-1)
    failed = state.invoke()
    _fatal(failed)
    assert failed["attempted_work"]["cover_refit_row_visits"] == 125


def test_source_record_policy_cannot_disagree_with_actual_cover_feature(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    state.features = [replace(state.features[0], max_fit_residual_mm=.04)]
    result = state.invoke()
    _fatal(result)
    assert "SUPPORT_COVER_FIT_POLICY_BINDING" in result["reasons"][0]
    assert state.fit_calls == []


@pytest.mark.parametrize("fault", ["list", "complex", "nan", "range", "empty"])
def test_invalid_raw_input_refuses_before_discovery(mod, monkeypatch, fault):
    state = _setup(mod, monkeypatch)
    if fault == "list": state.points = state.points.tolist()
    elif fault == "complex": state.points = state.points.astype(complex)
    elif fault == "nan": state.points[0, 0] = np.nan
    elif fault == "range": state.points[0, 0] = 1e6+1
    elif fault == "empty": state.points = np.empty((0, 3))
    result = state.invoke()
    _fatal(result)
    assert state.discovery_calls == state.fit_calls == []


@pytest.mark.parametrize("fault", [
    "count", "production", "registration", "frame", "source_digest", "point_digest", "scan_digest",
    "drop_source_record", "drop_group", "group_count_bool", "cover_identity", "member_identity",
    "permutation_bool", "raw_arc_bool", "raw_subset", "fit_policy", "refit_work_bool", "refit_work_wrong",
    "association_deviation", "association_witness", "association_context", "association_nominal",
    "instance_eligibility_int", "group_eligibility_int",
    "radius_bool", "radius_nan", "radius_negative", "sweep_mismatch", "material_side",
    "radius_interval", "trim", "datum", "finite_status", "line_nan", "axis_shape", "line_degenerate",
    "carrier_material", "carrier_event", "carrier_frame", "duplicate_prediction", "removed_prediction",
    "status", "instance_policy",
    "circle_residual", "flange_residual", "tangency_residual", "circle_condition", "singular_zero",
    "axis_zero", "axis_noncanonical", "center_off_axis", "carrier_radius_offset", "carrier_length",
    "arc_count_bool", "flange_count", "slab_count", "slab_points", "slab_gap", "slab_sweep",
])
def test_publication_rejects_tampered_proof_sources_quantities_and_projection(mod, monkeypatch, fault):
    state = _setup(mod, monkeypatch)
    result = state.invoke()
    group = _measured(result)
    measured, finite = group["measurements"], group["finite_support"]
    instance = finite["instance"]
    if fault == "count": result["observed_bend_count"] = 1
    elif fault == "production": result["production_use_allowed"] = True
    elif fault == "registration": result["registration_status"] = "ACCEPTED"
    elif fault == "frame": result["geometry_frame"] = "CAD"
    elif fault == "source_digest": result["source_context_result_sha256"] = "f"*64
    elif fault == "point_digest": result["raw_points_sha256"] = "f"*64
    elif fault == "scan_digest": result["scan_sha256"] = "f"*64
    elif fault == "drop_source_record": result["source_context_result"]["support_records"].pop()
    elif fault == "drop_group": result["support_groups"].pop()
    elif fault == "group_count_bool": result["conditional_support_group_count"] = True
    elif fault == "cover_identity": group["cover_record_id"] = "f"*64
    elif fault == "member_identity": group["member_record_ids"][0] = "f"*64
    elif fault == "permutation_bool": group["member_flange_permutations"][group["member_record_ids"][0]] = [False, True]
    elif fault == "raw_arc_bool": group["raw_support"]["arc_rows"][0] = False
    elif fault == "raw_subset": group["raw_support"]["arc_rows"].pop()
    elif fault == "fit_policy": group["fit_policy"]["max_fit_residual_mm"] = .06
    elif fault == "refit_work_bool": result["cover_refit_row_visits"] = True
    elif fault == "refit_work_wrong": result["cover_refit_row_visits"] -= 1
    elif fault == "association_deviation": group["associations"][0]["canonical_radius_deviation_mm"] = .1
    elif fault == "instance_eligibility_int": group["finite_support"]["instance"]["inner_face_claim_eligible"] = 0
    elif fault == "group_eligibility_int": group["inner_face_claim_eligible"] = 0
    elif fault == "association_witness": group["associations"][0]["is_full_cover_witness"] = not group["associations"][0]["is_full_cover_witness"]
    elif fault == "association_context": group["associations"][0]["context_index"] = True
    elif fault == "association_nominal": group["associations"][0]["nominal_radius_mm"] = 5.3
    elif fault == "radius_bool": measured["measured_radius_mm"] = True
    elif fault == "radius_nan": measured["measured_radius_mm"] = float("nan")
    elif fault == "radius_negative": measured["measured_radius_mm"] = -1.
    elif fault == "sweep_mismatch": measured["measured_sweep_deg"] += 1.
    elif fault == "material_side": measured["material_side_status"] = "VERIFIED_INNER_SURFACE"
    elif fault == "radius_interval": measured["radius_uncertainty_interval_mm"] = [5.29, 5.31]
    elif fault == "trim": measured["physical_trim_endpoints"] = [[0, 0, 0], [0, 0, 12]]
    elif fault == "datum": measured["datum_relative_deviation"] = .01
    elif fault == "finite_status": finite["status"] = "UNKNOWN"
    elif fault == "line_nan": instance["line_start"][0] = float("nan")
    elif fault == "axis_shape": instance["axis_support_line_end"] = [0., 0.]
    elif fault == "line_degenerate": instance["line_end"] = list(instance["line_start"])
    elif fault == "carrier_material": instance["material_side_status"] = "VERIFIED"
    elif fault == "carrier_event": instance["manufacturing_event_identity"] = "VERIFIED_BEND"
    elif fault == "carrier_frame": instance["geometry_frame"] = "CAD"
    elif fault == "duplicate_prediction": result["predictions"].append(copy.deepcopy(instance))
    elif fault == "removed_prediction": result["predictions"] = []
    elif fault == "status": result["status"] = "UNAVAILABLE"
    elif fault == "instance_policy": result["instance_policy"]["max_axial_gap_mm"] = 5.
    elif fault == "circle_residual": measured["circle_max_residual_mm"] = .051
    elif fault == "flange_residual": measured["flange_max_residual_mm"][0] = .051
    elif fault == "tangency_residual": measured["measured_tangency_residual_mm"][0] = .101
    elif fault == "circle_condition": measured["free_circle_jacobian_condition"] = 1001.
    elif fault == "singular_zero": measured["free_circle_min_singular_per_sqrt_n"] = 0.
    elif fault == "axis_zero": measured["measured_axis_direction"] = [0., 0., 0.]
    elif fault == "axis_noncanonical": measured["measured_axis_direction"] = [-v for v in measured["measured_axis_direction"]]
    elif fault == "center_off_axis": measured["measured_axis_point"][0] += 1.
    elif fault == "carrier_radius_offset": instance["line_start"][0] += 1.
    elif fault == "carrier_length": instance["axial_support_length_mm"] += 1.
    elif fault == "arc_count_bool": instance["unique_arc_support_count"] = True
    elif fault == "flange_count": instance["unique_flange_support_counts"][0] -= 1
    elif fault == "slab_count": instance["local_coverage"].pop()
    elif fault == "slab_points": instance["local_coverage"][0]["arc_points"] = 0
    elif fault == "slab_gap": instance["local_coverage"][0]["max_angular_gap_deg"] = 6.
    elif fault == "slab_sweep": instance["local_coverage"][0]["arc_sweep_deg"] = 19.
    else: raise AssertionError("UNDECLARED_PUBLICATION_FAULT")
    with pytest.raises(ValueError):
        _validate(mod, state, result)


def test_fatal_publication_cannot_attach_any_scientific_prefix(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    monkeypatch.setattr(mod, "MAX_COVER_REFIT_ROW_VISITS", 0)
    result = state.invoke()
    _fatal(result)
    result["support_groups"] = [{"invented": True}]
    with pytest.raises(ValueError): _validate(mod, state, result)


def test_validation_rechecks_proof_but_does_not_refit_or_claim_geometry_authentication(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    result = state.invoke()
    _measured(result)
    def no_refit(*args, **kwargs):
        raise AssertionError("VALIDATION_MUST_NOT_HIDE_ANOTHER_GEOMETRY_RUN")
    monkeypatch.setattr(mod._v1, "_canonical_fit", no_refit)
    report = _validate(mod, state, result)
    assert report["cover_plan_revalidated"] is True and report["cover_refit_repeated"] is False


def test_base_exception_is_not_swallowed_as_ordinary_failed_measurement(mod, monkeypatch):
    state = _setup(mod, monkeypatch)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("INVENTED_INTERRUPT")
    monkeypatch.setattr(mod._v1, "_canonical_fit", interrupted)
    with pytest.raises(KeyboardInterrupt, match="INVENTED_INTERRUPT"):
        state.invoke()
