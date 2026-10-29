"""Prospective isolated array/contract controls; never an actual native gate.

Authored source-only. Root must independently review/freeze before the proposed
60 s monitored wall / 45 worker CPU / 1 GiB / 1 MiB supervised invocation.
The canonical fit tests use invented positions and real NumPy/SciPy fitting.
Aggregation tests mock crop, per-context role discovery and owned cap selection;
they do NOT qualify those discovery mechanisms, CAD import or physical accuracy.
No backend package startup, test-to-test import, file input or native geometry.
"""

import base64
import copy
from dataclasses import replace
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend/cad_guided_context_measurement_v1.py"
SCAN_SHA = "a" * 64
ARRAY_EPS = 2e-7  # Fixed forward-error check on these represented invented arrays.


@pytest.fixture
def mod():
    name = "isolated_context_measurement_v1_adversarial"
    prefixes = (name, "backend")
    def owned(key):
        return any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)
    saved = {key: value for key, value in sys.modules.items() if owned(key)}
    for key in saved:
        del sys.modules[key]
    # Frozen V7's one absolute V5 import needs a second inert namespace. Never
    # execute backend/__init__.py; restore any caller modules after the fixture.
    fallback = types.ModuleType("backend")
    fallback.__path__ = [str(ROOT / "backend")]
    sys.modules["backend"] = fallback
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "backend")]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".cad_guided_context_measurement_v1", SOURCE)
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


def _sha(points):
    return hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()


def _geometry():
    """125 unique rows: 75 curved + 25 + 25 tangent planar rows, no cap."""
    radius = 5.3
    theta = np.deg2rad(np.arange(10., 81., 5.))
    z = (0., 3., 6., 9., 12.)
    arc = [(radius * math.cos(a), radius * math.sin(a), h) for h in z for a in theta]
    f0 = [(radius, -d, h) for h in z for d in (3., 5., 7., 9., 11.)]
    f1 = [(-d, radius, h) for h in z for d in (3., 5., 7., 9., 11.)]
    return np.array(arc + f0 + f1, dtype=np.float64), np.arange(75), [np.arange(75, 100), np.arange(100, 125)]


def _feature(mod, identity="nominal-a", radius=5.):
    return mod.CADGuidedFeature(feature_id=identity, nominal_radius_mm=radius,
        nominal_sweep_deg=90., origin=(0., 0., 0.), axis=(0., 0., 1.),
        start_radial=(1., 0., 0.), span_mm=(0., 12.), flange_window_mm=(2.5, 12.5),
        search_margin_mm=3., max_fit_residual_mm=.05, plane_exclusion_mm=.075,
        max_tangency_error_mm=.1, nominal_source="INVENTED_DECLARATION_NOT_TRUTH", fit_model="free_circle")


def _policy(mod):
    return mod.SupportPolicy(max_axial_gap_mm=4., max_angular_gap_deg=5.,
                             min_arc_sweep_deg=20., min_local_points=3)


def _frame(index):
    if index == 0:
        return np.eye(3), np.zeros(3)
    if index == 1:
        return np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]), np.array([11., -7., 3.])
    axis = np.array([1., 2., -3.]) / math.sqrt(14.)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = math.cos(.47) * np.eye(3) + (1 - math.cos(.47)) * np.outer(axis, axis) + math.sin(.47) * cross
    return rotation, np.array([-13., 17., 5.])


def _query(lower, upper):
    lo, hi = np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)
    n = len(lo)
    result = dict(status="COMPLETE", reason="CONDITIONAL_LOCAL_FINITE_TRIANGLE_QUERY_COMPLETE",
        closest_points=np.zeros((n, 3)), distances=(lo + hi) / 2,
        triangle_indices=np.zeros(n, dtype=np.int64), source_face_ids=(0,) * n,
        oriented_normals=np.tile([0., 0., 1.], (n, 1)), closest_regions=np.full(n, "FACE", dtype="U6"),
        near_tie_counts=np.ones(n, dtype=np.int64), distance_lower=lo, distance_upper=hi,
        triangle_distance_lower=lo.copy(), triangle_distance_upper=hi.copy(),
        numerical_distance_guard=(hi - lo) / 2, closest_point_error=np.zeros(n),
        closest_point_restoration_error=np.zeros(n),
        distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
        closest_point_error_scope="SELECTED_TRIANGLE_OPTIMIZER_NOT_GLOBAL_UNIQUE_WITNESS",
        normal_region_status="ENGINEERING_NOT_INTERVAL_CERTIFIED",
        tie_semantics="POSSIBLE_COMPETITORS_AT_FIXED_GLOBAL_UPPER_PLUS_TAU",
        diagnostics=dict(node_visits=2*n, triangle_box_pairs=2*n, triangle_pairs=2*n,
            certificate_pairs=8*n, normal_pairs=0, work_units=28*n, passes_completed=2,
            maximum_pair_batch=n, maximum_certificate_batch=n, tie_batches_completed=int(n > 0),
            maximum_tie_storage_rows=n, interval_width_rejections=0, certificate_failures=0,
            proof_pruned_triangle_pairs=0))
    for key in ("normal_ambiguous", "closest_point_ambiguous", "face_identity_ambiguous",
                "normal_ambiguity_uncertain", "closest_point_ambiguity_uncertain", "region_uncertain"):
        result[key] = np.zeros(n, dtype=bool)
    return result


def _registration(mod, points, contexts=1):
    candidates = []
    for i in range(contexts):
        r, t = np.eye(3), np.zeros(3)
        candidates.append(dict(index=i, execution_status="ATTEMPTED",
            numerical_consumer_status=mod._v3.CONSUMER_COMPLETE,
            numerical_support_policy=mod._v3.SUPPORT_POLICY, distance_model=mod._v3.DISTANCE_MODEL,
            optimizer_objective=mod._v3.SURROGATE, rotation=r.tolist(), translation=t.tolist(),
            conditional_context_support=dict(status="SUPPORTED_CONTEXT", proximity_policy=mod._stages.PROXIMITY,
                accepted_registration=False, pose_sha256=_sha(np.concatenate((r.ravel(), t))))))
    return dict(schema_version="noisy_plane_initialized_finite_registration_v12",
        status="CONDITIONAL_MEASUREMENT_CONTEXTS", transform_scan_to_cad=None,
        evidence=dict(candidate_evaluation_complete=True, numerical_consumer_complete=True,
            accepted_registration=False, connected_measurement_contexts=True,
            stage_policies=mod._stages.ablation_stage_policies_v1(connected=True),
            input_digests=dict(scan_points_sha256=_sha(points), scan_triangle_raw_rows_sha256="b"*64,
                              position_error_bounds_mm_sha256="c"*64),
            surface_role_preparation=dict(metadata_sha256="d"*64),
            candidates=candidates, measurement_context_indices=list(range(contexts))))


def _mock_role_metadata(prepared, keep, excluded):
    """Published schema fixture, NOT proof of actual cap or incidence analysis."""
    return dict(schema_version="context_surface_roles_v1", coordinate_frame="RAW_SCAN_UNREGISTERED",
        scope="CONDITIONAL_PATCH_NORMAL_ROLE_NOT_PHYSICAL_CAP_CERTIFICATE",
        role_policy="CAP_AND_NO_COMPETING_NONCAP_OR_UNRESOLVED_ALL_INCIDENT_TRIANGLES",
        numeric_status="ENGINEERING_NORMALIZATION_AND_ANGULAR_INWARD_RESERVE_NOT_CERTIFIED",
        points_sha256=prepared.points_sha256, triangle_raw_rows_sha256=prepared.triangle_raw_rows_sha256,
        position_error_bounds_mm_sha256=prepared.position_error_bounds_mm_sha256,
        encoded_patch_metadata_sha256=prepared.metadata_sha256,
        keep_raw_rows_sha256=hashlib.sha256(np.asarray(keep, dtype="<i8").tobytes()).hexdigest(),
        excluded_raw_rows_sha256=hashlib.sha256(np.asarray(excluded, dtype="<i8").tobytes()).hexdigest(),
        raw_row_count=prepared.raw_row_count, kept_row_count=len(keep), excluded_row_count=len(excluded),
        selected_proposal_evaluation_complete=True, production_use_allowed=False, observed_bend_count=None,
        attempted_triangle_membership_visits=7, attempted_incidence_corner_visits=9,
        disclosure="MOCK_SELECTOR_NO_CAP_CLASSIFICATION_PROOF")


def _setup(mod, monkeypatch, *, contexts=1, features=None, role_choices=None):
    """Explicit MOCK discovery/binding boundary; actual canonical fits remain live."""
    points, arc, flanges = _geometry()
    registration = _registration(mod, points, contexts)
    features = [_feature(mod)] if features is None else features
    state = types.SimpleNamespace(points=points, arc=arc, flanges=flanges, registration=registration,
        features=features, instance_policy=_policy(mod), scan_sha256=SCAN_SHA,
        crop_calls=[], measure_calls=[], role_calls=[])

    class FakePrepared:
        def __init__(self):
            self.points_sha256 = _sha(points)
            self.triangle_raw_rows_sha256 = "b"*64
            self.position_error_bounds_mm_sha256 = "c"*64
            self.metadata_sha256 = "d"*64
            self.raw_row_count = len(points)

    state.prepared = FakePrepared()
    monkeypatch.setattr(mod._roles, "PreparedContextSurfaceRolesV1", FakePrepared)

    def select(prepared, axis, **limits):
        assert prepared is state.prepared
        state.role_calls.append((axis.copy(), dict(limits)))
        kept, excluded = np.arange(len(points)), np.array([], dtype=np.int64)
        return dict(keep_raw_rows=kept, excluded_raw_rows=excluded,
                    metadata=_mock_role_metadata(prepared, kept, excluded))

    def crop(owned, feature):
        assert not owned.flags.writeable
        state.crop_calls.append((owned, feature))
        return np.arange(len(owned))

    def measure(local, feature):
        index = len(state.measure_calls)
        state.measure_calls.append((local.copy(), feature))
        a, fs = (arc, flanges) if role_choices is None else role_choices(index, feature, arc, flanges)
        return dict(measured_radius_mm=100. + index, mocked_per_context_fit=True,
            support_row_indices=list(range(len(local))),
            _instance_support=dict(arc_row_indices=list(a), flange_row_indices=[list(x) for x in fs]))

    monkeypatch.setattr(mod._roles, "select_context_measurement_rows_v1", select)
    monkeypatch.setattr(mod, "_crop_rows", crop)
    monkeypatch.setattr(mod, "_measure", measure)
    state.invoke = lambda: mod.inspect_generated_context_windows_v1(state.points, state.features,
        registration=state.registration, scan_sha256=state.scan_sha256,
        instance_policy=state.instance_policy, surface_roles=state.prepared)
    return state


def _refused(result):
    assert result["status"] == "UNAVAILABLE"
    assert result["discarded_measurement_prefix"] is True
    assert result["support_records"] == result["context_windows"] == result["predictions"] == []
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False


def _publication(mod, state, result):
    return mod.validate_context_measurement_publication_v1(result, raw_count=len(state.points),
        scan_sha256=SCAN_SHA, raw_points_sha256=_sha(state.points))


def test_distance_proximity_does_not_certify_legacy_targets_or_tangent_rank(mod):
    query = _query([.1, .4, .49, .51], [.2, .5, .51, .6])
    query["closest_point_ambiguity_uncertain"][0] = True
    query["region_uncertain"][1] = True
    before = copy.deepcopy(query)
    rows = mod._stages.classify_stage_rows_v1(query, .5)
    assert rows["definite_proximity"].tolist() == [True, True, False, False]
    assert rows["proposal_target"].tolist() == [False, True, False, False]
    assert rows["tangent_constraint"].tolist() == [False]*4
    assert rows["unresolved_proximity"].tolist() == [False, False, True, False]
    assert rows["definite_outside"].tolist() == [False, False, False, True]
    assert rows["possible_proximity"].tolist() == [True, True, True, False]
    rows["definite_proximity"][:] = False
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(query[key], value)
    assert mod._stages.stage_row_counts_v1(query, .5)["definite_proximity"] == 2


@pytest.mark.parametrize("flag,target,tangent", [
    ("normal_ambiguous", False, False), ("closest_point_ambiguous", False, False),
    ("normal_ambiguity_uncertain", False, False), ("closest_point_ambiguity_uncertain", False, False),
    ("face_identity_ambiguous", True, True), ("region_uncertain", True, False)])
def test_each_uncertainty_flag_has_explicit_stage_role(mod, flag, target, tangent):
    query = _query([.1], [.2])
    query[flag][0] = True
    rows = mod._stages.classify_stage_rows_v1(query, .5)
    assert rows["definite_proximity"].tolist() == [True]
    assert rows["proposal_target"].tolist() == [target]
    assert rows["tangent_constraint"].tolist() == [tangent]


@pytest.mark.parametrize("fault", ["missing", "unavailable", "shape", "nan", "bounds", "work"])
def test_incomplete_or_malformed_distances_are_not_proximity_evidence(mod, fault):
    query = _query([.1], [.2])
    if fault == "missing": del query["distance_lower"]
    elif fault == "unavailable": query["status"] = "UNAVAILABLE"
    elif fault == "shape": query["region_uncertain"] = np.zeros(2, dtype=bool)
    elif fault == "nan": query["distance_upper"][0] = np.nan
    elif fault == "bounds": query["distance_lower"][0] = .3
    else: query["diagnostics"]["certificate_pairs"] -= 1
    with pytest.raises(ValueError):
        mod._stages.classify_stage_rows_v1(query, .5)


def test_stage_policy_has_exact_bound_tuple_and_explicit_ablation(mod):
    policy = mod._stages.StagePoliciesV1()
    c2 = mod._stages.snapshot_stage_policies_v1(policy)
    c1 = mod._stages.ablation_stage_policies_v1(connected=False)
    assert c2["proposal_proximity"] == c1["proposal_proximity"]
    assert c2["refinement_targets"] == c1["refinement_targets"]
    assert c2["tangent_constraints"] == c1["tangent_constraints"]
    assert c2["final_proximity"] != c1["final_proximity"]
    object.__setattr__(policy, "measurement", "UNSUPPORTED")
    with pytest.raises(ValueError): mod._stages.snapshot_stage_policies_v1(policy)
    with pytest.raises(ValueError): mod._stages.snapshot_stage_policies_v1(c2)
    with pytest.raises(ValueError): mod._stages.ablation_stage_policies_v1(connected=1)


@pytest.mark.parametrize("frame", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_actual_canonical_free_fit_fixed_roles_is_proper_pose_and_order_invariant(mod, frame, reverse):
    points, arc, flanges = _geometry()
    rotation, translation = _frame(frame)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14, rtol=0)
    assert abs(np.linalg.det(rotation) - 1) < 1e-14
    points = points @ rotation.T + translation
    if reverse: flanges = flanges[::-1]
    measured, finite = mod._canonical_fit(points, arc, flanges, _feature(mod), _policy(mod), "e"*64)
    assert measured["measured_radius_mm"] == pytest.approx(5.3, abs=ARRAY_EPS)
    assert measured["measured_sweep_magnitude_deg"] == pytest.approx(90., abs=ARRAY_EPS)
    assert abs(measured["measured_sweep_deg"]) == measured["measured_sweep_magnitude_deg"]
    assert measured["radius_uncertainty_interval_mm"] is None
    assert measured["material_side_status"] == "CONTEXT_ASSUMED_NOT_VERIFIED"
    assert finite["status"] == "SUPPORTED", finite
    instance = finite["instance"]
    expected = np.array([[5.3/math.sqrt(2), 5.3/math.sqrt(2), h] for h in (0., 12.)]) @ rotation.T + translation
    actual = np.array([instance["line_start"], instance["line_end"]])
    assert min(np.max(np.abs(actual-expected)), np.max(np.abs(actual-expected[::-1]))) < ARRAY_EPS
    expected_axis = np.array([[0., 0., h] for h in (0., 12.)]) @ rotation.T + translation
    actual_axis = np.array([instance["axis_support_line_start"], instance["axis_support_line_end"]])
    assert min(np.max(np.abs(actual_axis-expected_axis)), np.max(np.abs(actual_axis-expected_axis[::-1]))) < ARRAY_EPS
    assert instance["line_convention"] == "MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR"
    assert instance["physical_trim_endpoints"] is None
    assert instance["manufacturing_event_identity"] == "UNKNOWN"


def test_fixed_role_fit_has_no_nominal_radius_answer_and_gap_refusal_is_not_lost_quantity(mod):
    points, arc, flanges = _geometry()
    policy = replace(_policy(mod), max_axial_gap_mm=2.9)
    measured, finite = mod._canonical_fit(points, arc, flanges, _feature(mod, radius=9.), policy, "e"*64)
    assert measured["measured_radius_mm"] == pytest.approx(5.3, abs=ARRAY_EPS)
    assert finite["status"] == "UNAVAILABLE" and finite["instance"] is None
    assert "DISCONNECTED_AXIAL_SUPPORT" in finite["reasons"]


@pytest.mark.parametrize("fault", ["short_flat_arc", "eleven_rows", "duplicate_position", "overlapping_roles"])
def test_canonical_fixed_role_adversaries_refuse_without_changing_model(mod, fault):
    points, arc, flanges = _geometry()
    exception = ValueError
    if fault == "short_flat_arc":
        theta = np.deg2rad(np.linspace(5., 25., 15))
        points[:75] = [(5.3*math.cos(a), 5.3*math.sin(a), z) for z in (0., 3., 6., 9., 12.) for a in theta]
        exception = mod._Unavailable
    elif fault == "eleven_rows": arc = arc[:11]
    elif fault == "duplicate_position": points[1] = points[0]
    else: flanges[0][0] = arc[0]
    with pytest.raises(exception):
        mod._canonical_fit(points, arc, flanges, _feature(mod), _policy(mod), "e"*64)


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_raw_feature_inverse_pose_is_not_forward_transform(mod, frame):
    feature = _feature(mod)
    rotation, translation = _frame(frame)
    inverse = mod._raw_feature(feature, rotation, translation)
    np.testing.assert_allclose(rotation @ inverse.origin + translation, feature.origin, atol=1e-13, rtol=0)
    np.testing.assert_allclose(rotation @ inverse.axis, feature.axis, atol=1e-13, rtol=0)
    np.testing.assert_allclose(rotation @ inverse.start_radial, feature.start_radial, atol=1e-13, rtol=0)


ADMISSION_FAULTS = (
    "schema", "status", "published_pose", "incomplete", "numeric_incomplete", "accepted", "disconnected",
    "stage", "points_digest", "boolean_index", "omitted_context", "repeated_context", "candidate_index",
    "execution", "consumer", "support_policy", "distance_model", "surrogate", "rotation_bool",
    "rotation_string", "translation_bool", "translation_string", "pose_digest", "improper_rotation",
    "missing_roles", "roles_points", "roles_triangles", "roles_epsilon", "roles_metadata", "roles_count",
    "scan_digest", "instance_policy", "feature_model")


@pytest.mark.parametrize("fault", ADMISSION_FAULTS)
def test_strict_metadata_pose_and_owned_role_failures_refuse_before_crop(mod, monkeypatch, fault):
    s = _setup(mod, monkeypatch)
    r, e = s.registration, s.registration["evidence"]
    candidate = e["candidates"][0]
    if fault == "schema": r["schema_version"] = "noisy_plane_initialized_finite_registration_v11"
    elif fault == "status": r["status"] = "ACCEPTED"
    elif fault == "published_pose": r["transform_scan_to_cad"] = np.eye(4).tolist()
    elif fault in ("incomplete", "numeric_incomplete", "accepted", "disconnected"):
        key = dict(incomplete="candidate_evaluation_complete", numeric_incomplete="numerical_consumer_complete",
                   accepted="accepted_registration", disconnected="connected_measurement_contexts")[fault]
        e[key] = fault == "accepted"
    elif fault == "stage": e["stage_policies"]["measurement"] = "UNSUPPORTED"
    elif fault == "points_digest": e["input_digests"]["scan_points_sha256"] = "f"*64
    elif fault == "boolean_index": e["measurement_context_indices"] = [False]
    elif fault == "omitted_context": e["measurement_context_indices"] = []
    elif fault == "repeated_context": e["measurement_context_indices"] = [0, 0]
    elif fault == "candidate_index": candidate["index"] = False
    elif fault in ("execution", "consumer", "support_policy", "distance_model", "surrogate"):
        key = dict(execution="execution_status", consumer="numerical_consumer_status",
                   support_policy="numerical_support_policy", distance_model="distance_model",
                   surrogate="optimizer_objective")[fault]
        candidate[key] = "WRONG"
    elif fault == "rotation_bool": candidate["rotation"] = np.eye(3, dtype=bool).tolist()
    elif fault == "rotation_string": candidate["rotation"] = np.eye(3).astype(str).tolist()
    elif fault == "translation_bool": candidate["translation"] = [False]*3
    elif fault == "translation_string": candidate["translation"] = ["0"]*3
    elif fault == "pose_digest": candidate["conditional_context_support"]["pose_sha256"] = "f"*64
    elif fault == "improper_rotation": candidate["rotation"][0][0] = -1.
    elif fault == "missing_roles": s.prepared = None
    elif fault.startswith("roles_"):
        attr = dict(roles_points="points_sha256", roles_triangles="triangle_raw_rows_sha256",
            roles_epsilon="position_error_bounds_mm_sha256", roles_metadata="metadata_sha256",
            roles_count="raw_row_count")[fault]
        setattr(s.prepared, attr, 1 if fault == "roles_count" else "f"*64)
    elif fault == "scan_digest": s.scan_sha256 = "NOT_A_SHA"
    elif fault == "instance_policy": s.instance_policy = {}
    else: s.features = [replace(s.features[0], fit_model="measured_flange_tangent")]
    result = s.invoke()
    _refused(result)
    assert s.crop_calls == s.measure_calls == s.role_calls == []


def test_aliases_and_nominal_associations_survive_one_actual_canonical_refit(mod, monkeypatch):
    features = [_feature(mod, "a", 5.), _feature(mod, "b", 6.)]
    s = _setup(mod, monkeypatch, contexts=2, features=features,
        role_choices=lambda i, f, a, fs: (a, fs if i % 2 == 0 else fs[::-1]))
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    assert len(result["support_records"]) == len(result["predictions"]) == 1
    record = result["support_records"][0]
    assert len(record["associations"]) == len(result["context_windows"]) == 4
    assert [x["per_context_measurements"]["measured_radius_mm"] for x in record["associations"]] == [100., 101., 102., 103.]
    assert record["measurements"]["measured_radius_mm"] == pytest.approx(5.3, abs=ARRAY_EPS)
    assert [x["canonical_radius_deviation_mm"] for x in record["associations"]] == pytest.approx([.3, -.7, .3, -.7], abs=ARRAY_EPS)
    assert all(x["accepted_registration"] is False for x in record["associations"])
    assert record["associations"][0]["ordered_flange_raw_rows"] != record["associations"][1]["ordered_flange_raw_rows"]
    assert result["attempted_crop_row_visits"] == result["completed_crop_row_visits"] == 500
    assert result["canonical_refit_row_visits"] == 125
    assert len(s.role_calls) == 1
    assert result["role_membership_visits"] == 7 and result["role_incidence_corner_visits"] == 9
    bitmap = result["surface_role_filters"][0]["excluded_raw_row_bitmap"]
    assert bitmap["bit_count"] == 125 and bitmap["encoding"] == "BASE64_PACKBITS_LITTLE_ZERO_PADDED"
    assert base64.b64decode(bitmap["data"], validate=True) == bytes(16)
    assert result["observed_bend_count"] is None and result["joint_configuration_status"] == "NOT_ESTABLISHED"
    _publication(mod, s, result)


def test_mocked_role_exclusion_is_applied_after_full_crop_with_original_row_bitmap(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    # These extra rows are only a selector-plumbing marker, not geometric caps.
    s.points = np.vstack(([[20., 20., 0.], [21., 20., 0.], [22., 20., 0.]], s.points))
    s.prepared.points_sha256, s.prepared.raw_row_count = _sha(s.points), 128
    s.registration["evidence"]["input_digests"]["scan_points_sha256"] = _sha(s.points)
    def select(*args, **kwargs):
        kept, excluded = np.arange(3, 128), np.arange(3)
        return dict(keep_raw_rows=kept, excluded_raw_rows=excluded,
                    metadata=_mock_role_metadata(s.prepared, kept, excluded))
    monkeypatch.setattr(mod._roles, "select_context_measurement_rows_v1", select)
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    window = result["context_windows"][0]
    assert window["pre_role_filter_roi_rows"] == 128 and window["local_input_rows"] == 125
    assert window["excluded_cap_like_roi_rows"] == 3
    assert len(s.crop_calls[0][0]) == 128 and len(s.measure_calls[0][0]) == 125
    assert result["attempted_crop_row_visits"] == result["completed_crop_row_visits"] == 128
    raw = result["support_records"][0]["raw_support"]
    assert raw == dict(arc_rows=list(range(3, 78)), flange_rows=[list(range(78, 103)), list(range(103, 128))])
    bitmap = result["surface_role_filters"][0]["excluded_raw_row_bitmap"]
    decoded = np.unpackbits(np.frombuffer(base64.b64decode(bitmap["data"], validate=True), dtype=np.uint8), bitorder="little")
    np.testing.assert_array_equal(decoded, np.r_[np.ones(3, dtype=np.uint8), np.zeros(125, dtype=np.uint8)])
    assert bitmap["bit_count"] == 128
    _publication(mod, s, result)


def _overlap_choices(index, feature, arc, flanges):
    # Two different broad 65-degree subsets, not duplicate exact-role aliases.
    keep = arc[arc % 15 != (0 if index % 2 == 0 else 14)]
    return keep, flanges


def test_late_overlapping_arc_claims_invalidate_both_actual_fits_not_choose_best(mod, monkeypatch):
    s = _setup(mod, monkeypatch, contexts=2, role_choices=_overlap_choices)
    result = s.invoke()
    assert result["computation_status"] == "COMPLETE"
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert len(result["support_records"]) == 2 and len(result["context_windows"]) == 2
    assert all(x["status"] == "UNAVAILABLE" and x["measurements"] is None and x["finite_support"] is None
               for x in result["support_records"])
    assert result["ownership_work"]["invalidated_records"] == 2
    assert result["canonical_refit_row_visits"] == 240
    assert all(x["status"] == "MEASURED_CONTEXT" for x in result["context_windows"])
    _publication(mod, s, result)


def _ownership_record(arc, f0, f1):
    return dict(status="MEASURED_CONDITIONAL", measurements={"retained": True}, finite_support={"retained": True},
                raw_support=dict(arc_rows=list(arc), flange_rows=[list(f0), list(f1)]), reasons=[])


@pytest.mark.parametrize("kind,conflicts", [("arc_arc", 2), ("arc_flange", 2), ("flange_only", 0), ("duplicate_position", 2)])
def test_role_ownership_distinguishes_arc_conflict_from_flange_context(mod, kind, conflicts):
    points = np.array([[float(i), 0., 0.] for i in range(8)])
    first = _ownership_record([0], [1], [2])
    second = _ownership_record([3], [1], [4])
    if kind == "arc_arc": second["raw_support"]["arc_rows"] = [0]
    elif kind == "arc_flange": second["raw_support"]["flange_rows"][1] = [0]
    elif kind == "duplicate_position": points[3] = points[0]
    records = [first, second]
    work = mod._invalidate_overlapping_records(records, points)
    assert work["row_visits"] == 6 and work["invalidated_records"] == conflicts
    assert sum(r["status"] == "UNAVAILABLE" for r in records) == conflicts
    if not conflicts: assert all(r["measurements"] is not None for r in records)


def test_digest_collision_does_not_merge_different_raw_roles(mod, monkeypatch):
    s = _setup(mod, monkeypatch, contexts=2, role_choices=_overlap_choices)
    monkeypatch.setattr(mod, "_json_digest", lambda value: "e"*64)
    result = s.invoke()
    _refused(result)
    assert "CONTRADICTORY_SUPPORT_EQUIVALENCE_DIGEST" in result["reasons"][0]
    assert result["attempted_support_records"] == 1
    assert result["attempted_work"]["canonical_refit_row_visits"] == 0


def test_duplicate_position_in_real_canonical_fit_whole_clears_context_prefix(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    s.points[1] = s.points[0]
    s.registration["evidence"]["input_digests"]["scan_points_sha256"] = _sha(s.points)
    s.prepared.points_sha256 = _sha(s.points)
    result = s.invoke()
    _refused(result)
    assert "DUPLICATE_POSITION_IN_CANONICAL_SUPPORT" in result["reasons"][0]
    assert result["attempted_support_records"] == 1
    assert result["attempted_work"]["canonical_refit_row_visits"] == 125


@pytest.mark.parametrize("failed_call", [1, 2])
def test_crop_exception_retains_precharged_work_but_no_scientific_prefix(mod, monkeypatch, failed_call):
    s = _setup(mod, monkeypatch, contexts=2)
    original = mod._crop_rows
    calls = 0
    def crop(points, feature):
        nonlocal calls
        calls += 1
        if calls == failed_call: raise RuntimeError("INVENTED_CROP_FAULT")
        return original(points, feature)
    monkeypatch.setattr(mod, "_crop_rows", crop)
    result = s.invoke()
    _refused(result)
    assert result["attempted_work"]["attempted_crop_row_visits"] == 125*failed_call
    assert result["attempted_work"]["completed_crop_row_visits"] == 125*(failed_call-1)
    assert result["attempted_work"]["role_membership_visits"] == 7
    assert result["attempted_context_windows"] == failed_call-1


def test_late_canonical_exception_discards_previously_measured_class(mod, monkeypatch):
    s = _setup(mod, monkeypatch, contexts=2, role_choices=_overlap_choices)
    original = mod._canonical_fit
    calls = []
    def fit(*args):
        calls.append(args[1])
        if len(calls) == 2: raise RuntimeError("INVENTED_LATE_CANONICAL_FAULT")
        return original(*args)
    monkeypatch.setattr(mod, "_canonical_fit", fit)
    result = s.invoke()
    _refused(result)
    assert len(calls) == 2 and result["attempted_support_records"] == 2
    assert result["attempted_work"]["canonical_refit_row_visits"] == 240


@pytest.mark.parametrize("cap,limit,reason", [
    ("MAX_CROP_ROW_VISITS", 124, "FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET"),
    ("MAX_TOTAL_LOCAL_ROWS", 124, "AGGREGATE_CONTEXT_ROW_BUDGET"),
    ("MAX_CONTEXT_WINDOWS", 0, "CONTEXT_WINDOW_PREFLIGHT_BUDGET"),
    ("MAX_ROLE_CACHE_ENTRIES", 0, "CONTEXT_ROLE_CACHE_BUDGET"),
    ("MAX_REPORT_BYTES", 0, "CONTEXT_REPORT_BUDGET")])
def test_early_and_late_caps_whole_clear_without_widening(mod, monkeypatch, cap, limit, reason):
    s = _setup(mod, monkeypatch)
    monkeypatch.setattr(mod, cap, limit)
    result = s.invoke()
    _refused(result)
    assert reason in result["reasons"][0]
    expected = 125 if cap in ("MAX_TOTAL_LOCAL_ROWS", "MAX_REPORT_BYTES") else 0
    assert result["attempted_work"]["attempted_crop_row_visits"] == expected
    assert result["attempted_work"]["completed_crop_row_visits"] == expected


def test_exact_full_crop_budget_is_admitted(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    monkeypatch.setattr(mod, "MAX_CROP_ROW_VISITS", 125)
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    assert result["attempted_crop_row_visits"] == result["completed_crop_row_visits"] == 125


def test_late_class_limit_and_ownership_work_are_explicit(mod, monkeypatch):
    s = _setup(mod, monkeypatch, contexts=2, role_choices=_overlap_choices)
    monkeypatch.setattr(mod, "MAX_SUPPORT_RECORDS", 1)
    result = s.invoke()
    _refused(result)
    assert "SUPPORT_RECORD_BUDGET" in result["reasons"][0]
    assert result["attempted_work"]["completed_crop_row_visits"] == 250
    monkeypatch.setattr(mod, "MAX_SUPPORT_RECORDS", 512)
    monkeypatch.setattr(mod, "MAX_OWNERSHIP_PAIRS", 0)
    result = s.invoke()
    _refused(result)
    assert "CONTEXT_OWNERSHIP_PAIR_BUDGET" in result["reasons"][0]
    assert result["attempted_work"]["ownership"]["competing_claim_pairs"] == 1
    assert result["attempted_work"]["ownership"]["row_visits"] > 120


def test_role_selector_failure_retains_its_attempted_work_and_no_crop(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    stats = dict(attempted_triangle_membership_visits=13, attempted_incidence_corner_visits=0)
    def fail(*args, **kwargs):
        raise mod._roles.ContextSurfaceRolesUnavailableV1("INVENTED_ROLE_BUDGET", statistics=stats)
    monkeypatch.setattr(mod._roles, "select_context_measurement_rows_v1", fail)
    result = s.invoke()
    _refused(result)
    assert result["attempted_work"]["failed_role_selection"] == stats
    assert result["attempted_work"]["role_membership_visits"] == 0
    assert s.crop_calls == []


def test_raw_snapshot_and_pose_snapshot_survive_later_caller_mutation(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    original = mod._raw_feature
    def mutate(feature, rotation, translation):
        assert not rotation.flags.writeable and not translation.flags.writeable
        s.points[:] = np.nan
        s.registration["evidence"]["candidates"][0]["translation"][:] = [999., 999., 999.]
        return original(feature, rotation, translation)
    monkeypatch.setattr(mod, "_raw_feature", mutate)
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    assert result["support_records"][0]["measurements"]["measured_radius_mm"] == pytest.approx(5.3, abs=ARRAY_EPS)
    assert result["support_records"][0]["associations"][0]["transform_scan_to_cad"] == np.eye(4).tolist()


def test_post_copy_numeric_readmission_is_not_only_pre_snapshot_validation(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    original = mod.np.asarray
    fired = False
    def mutate(value, *args, **kwargs):
        nonlocal fired
        if value is s.points and kwargs.get("dtype") == "<f8" and not fired:
            fired = True
            value[0, 0] = np.nan
        return original(value, *args, **kwargs)
    monkeypatch.setattr(mod.np, "asarray", mutate)
    result = s.invoke()
    _refused(result)
    assert fired and "INVALID_OWNED_RAW_CONTEXT_POINTS" in result["reasons"][0]
    assert s.crop_calls == s.role_calls == []


PUBLICATION_FAULTS = (
    "schema", "accepted", "count", "production", "frame", "source", "points", "prefix",
    "empty_partial", "unavailable_measured", "record_id", "duplicate_record", "negative_row",
    "boolean_row", "out_of_range_row", "duplicate_row", "unsorted_row", "role_overlap",
    "fit_model", "uncertainty_claim", "radius_nan", "radius_negative", "radius_bool", "sweep_bool",
    "sweep_mismatch", "endpoint_nan", "endpoint_bool", "endpoint_string", "endpoint_shape",
    "endpoint_degenerate", "carrier_claim", "trim_claim", "known_event", "prediction_missing")


@pytest.mark.parametrize("fault", PUBLICATION_FAULTS)
def test_publication_boundary_rejects_malformed_or_overclaiming_result(mod, monkeypatch, fault):
    s = _setup(mod, monkeypatch)
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    result = copy.deepcopy(result)
    record = result["support_records"][0]
    measured, instance = record["measurements"], record["finite_support"]["instance"]
    if fault == "schema": result["schema"] = "WRONG"
    elif fault == "accepted": result["registration_status"] = "ACCEPTED"
    elif fault == "count": result["observed_bend_count"] = 1
    elif fault == "production": result["production_use_allowed"] = True
    elif fault == "frame": result["geometry_frame"] = "CAD"
    elif fault == "source": result["scan_sha256"] = "f"*64
    elif fault == "points": result["raw_points_sha256"] = "f"*64
    elif fault == "prefix": result["discarded_measurement_prefix"] = True
    elif fault == "empty_partial": result["support_records"], result["predictions"] = [], []
    elif fault == "unavailable_measured":
        result["status"], result["predictions"] = "UNAVAILABLE", []
        record["finite_support"] = dict(status="UNAVAILABLE", instance=None, reasons=["INVENTED"])
    elif fault == "record_id": record["record_id"] = "NOT_HEX"
    elif fault == "duplicate_record": result["support_records"].append(copy.deepcopy(record))
    elif fault in ("negative_row", "boolean_row", "out_of_range_row", "duplicate_row", "unsorted_row"):
        rows = record["raw_support"]["arc_rows"]
        if fault == "negative_row": rows[0] = -1
        elif fault == "boolean_row": rows[0] = False
        elif fault == "out_of_range_row": rows[-1] = len(s.points)
        elif fault == "duplicate_row": rows[1] = rows[0]
        else: rows[0], rows[1] = rows[1], rows[0]
    elif fault == "role_overlap": record["raw_support"]["flange_rows"][0][0] = 0
    elif fault == "fit_model": measured["fit_model"] = "measured_flange_tangent"
    elif fault == "uncertainty_claim": measured["radius_uncertainty_interval_mm"] = [5.2, 5.4]
    elif fault == "radius_nan": measured["measured_radius_mm"] = float("nan")
    elif fault == "radius_negative": measured["measured_radius_mm"] = -1.
    elif fault == "radius_bool": measured["measured_radius_mm"] = True
    elif fault == "sweep_bool": measured["measured_sweep_deg"] = True
    elif fault == "sweep_mismatch": measured["measured_sweep_deg"] = 89.
    elif fault == "endpoint_nan": instance["line_start"][0] = float("nan")
    elif fault == "endpoint_bool": instance["line_start"] = [False]*3
    elif fault == "endpoint_string": instance["line_start"] = ["0"]*3
    elif fault == "endpoint_shape": instance["line_start"] = [0., 0.]
    elif fault == "endpoint_degenerate": instance["line_end"] = list(instance["line_start"])
    elif fault == "carrier_claim": instance["line_convention"] = "VERIFIED_INNER_MATERIAL_CARRIER"
    elif fault == "trim_claim": instance["physical_trim_endpoints"] = [[0., 0., 0.], [0., 0., 12.]]
    elif fault == "known_event": instance["manufacturing_event_identity"] = "ONE_BEND"
    else: result["predictions"] = []
    with pytest.raises(ValueError): _publication(mod, s, result)


def test_publication_accepts_complete_quantity_with_unavailable_finite_support(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    s.instance_policy = replace(s.instance_policy, max_axial_gap_mm=2.9)
    result = s.invoke()
    assert result["status"] == "PARTIAL" and result["predictions"] == []
    assert result["support_records"][0]["finite_support"]["status"] == "UNAVAILABLE"
    assert result["support_records"][0]["measurements"]["measured_radius_mm"] == pytest.approx(5.3, abs=ARRAY_EPS)
    _publication(mod, s, result)


def test_publication_accepts_only_empty_scientific_prefix_after_fatal_failure(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    monkeypatch.setattr(mod, "MAX_REPORT_BYTES", 0)
    result = s.invoke()
    _refused(result)
    _publication(mod, s, result)


def test_expected_geometry_refusal_is_retained_not_retyped_as_absent_bend(mod, monkeypatch):
    s = _setup(mod, monkeypatch)
    def unsupported(*args): raise mod._Unavailable("INVENTED_EXPECTED_LOCAL_GEOMETRY_REFUSAL")
    monkeypatch.setattr(mod, "_measure", unsupported)
    result = s.invoke()
    assert result["computation_status"] == "COMPLETE" and result["status"] == "UNAVAILABLE"
    assert result["observed_bend_count"] is None and result["predictions"] == []
    assert len(result["context_windows"]) == 1
    assert result["context_windows"][0]["reasons"] == ["INVENTED_EXPECTED_LOCAL_GEOMETRY_REFUSAL"]
    _publication(mod, s, result)


def _add_second_association_filter(result):
    """A second complete bound filter; no different geometric claim is made."""
    result["surface_role_filters"].append(copy.deepcopy(result["surface_role_filters"][0]))
    result["support_records"][0]["associations"][1]["surface_role_filter_index"] = 1
    result["context_windows"][1]["surface_role_filter_index"] = 1
    result["role_membership_visits"] *= 2
    result["role_incidence_corner_visits"] *= 2


def _exclude_published_row(result, row, *, filter_index=0, window_index=0):
    """Make the bitmap/counts/SHA mutually consistent; only support use is bad."""
    filtered = result["surface_role_filters"][filter_index]
    n = filtered["raw_row_count"]
    mask = np.zeros(n, dtype=np.uint8)
    mask[row] = 1
    kept, excluded = np.flatnonzero(mask == 0), np.flatnonzero(mask)
    filtered.update(kept_row_count=n-1, excluded_row_count=1,
        keep_raw_rows_sha256=hashlib.sha256(np.asarray(kept, dtype="<i8").tobytes()).hexdigest(),
        excluded_raw_rows_sha256=hashlib.sha256(np.asarray(excluded, dtype="<i8").tobytes()).hexdigest(),
        excluded_raw_row_bitmap=dict(encoding="BASE64_PACKBITS_LITTLE_ZERO_PADDED", bit_count=n,
            data=base64.b64encode(np.packbits(mask, bitorder="little").tobytes()).decode()))
    window = result["context_windows"][window_index]
    window["local_input_rows"] -= 1
    window["excluded_cap_like_roi_rows"] += 1


FILTER_PUBLICATION_FAULTS = (
    "missing_filters", "filters_type", "empty_filters", "schema", "frame", "scope", "policy",
    "numeric", "complete", "production", "count", "source_binding_missing", "source_binding_hash",
    "source_binding_points", "filter_points", "filter_triangles", "filter_epsilon", "filter_patches",
    "bitmap_missing", "bitmap_encoding", "bitmap_base64", "bitmap_noncanonical", "bitmap_short",
    "bitmap_long", "bitmap_padding", "bitmap_count_bool", "bitmap_count_wrong", "raw_count_bool",
    "kept_count_wrong", "excluded_count_bool", "kept_sha", "excluded_sha", "window_filter_bool",
    "window_filter_negative", "window_filter_range", "window_context_bool", "window_context_range",
    "window_counts", "association_filter_bool", "association_filter_negative", "association_filter_range",
    "association_context_missing", "association_feature_missing", "association_flange_order_mismatch",
    "excluded_arc", "excluded_flange", "excluded_later_association")


@pytest.mark.parametrize("fault", FILTER_PUBLICATION_FAULTS)
def test_publication_revalidates_complete_role_filter_and_each_support_association(mod, monkeypatch, fault):
    s = _setup(mod, monkeypatch, contexts=2 if fault == "excluded_later_association" else 1)
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    _publication(mod, s, result)  # A complete valid precursor, not a pre-failed fixture.
    result = copy.deepcopy(result)
    filtered = result["surface_role_filters"][0]
    bitmap = filtered["excluded_raw_row_bitmap"]
    window = result["context_windows"][0]
    association = result["support_records"][0]["associations"][0]
    if fault == "missing_filters": del result["surface_role_filters"]
    elif fault == "filters_type": result["surface_role_filters"] = tuple(result["surface_role_filters"])
    elif fault == "empty_filters": result["surface_role_filters"] = []
    elif fault in ("schema", "frame", "scope", "policy", "numeric"):
        key = dict(schema="schema_version", frame="coordinate_frame", scope="scope",
                   policy="role_policy", numeric="numeric_status")[fault]
        filtered[key] = "WRONG_OR_OVERCLAIMING"
    elif fault == "complete": filtered["selected_proposal_evaluation_complete"] = False
    elif fault == "production": filtered["production_use_allowed"] = True
    elif fault == "count": filtered["observed_bend_count"] = 1
    elif fault == "source_binding_missing": del result["surface_role_source_binding"]
    elif fault == "source_binding_hash": result["surface_role_source_binding"]["triangle_raw_rows_sha256"] = "NOT_SHA"
    elif fault == "source_binding_points":
        result["surface_role_source_binding"]["points_sha256"] = "f"*64
        filtered["points_sha256"] = "f"*64
    elif fault.startswith("filter_"):
        key = dict(filter_points="points_sha256", filter_triangles="triangle_raw_rows_sha256",
            filter_epsilon="position_error_bounds_mm_sha256", filter_patches="encoded_patch_metadata_sha256")[fault]
        filtered[key] = "f"*64
    elif fault == "bitmap_missing": del filtered["excluded_raw_row_bitmap"]
    elif fault == "bitmap_encoding": bitmap["encoding"] = "BIG_ENDIAN_OR_UNKNOWN"
    elif fault == "bitmap_base64": bitmap["data"] = "*"*24
    elif fault == "bitmap_noncanonical": bitmap["data"] += "\n"
    elif fault == "bitmap_short": bitmap["data"] = base64.b64encode(bytes(15)).decode()
    elif fault == "bitmap_long": bitmap["data"] = base64.b64encode(bytes(17)).decode()
    elif fault == "bitmap_padding": bitmap["data"] = base64.b64encode(bytes(15) + b"\x80").decode()
    elif fault == "bitmap_count_bool": bitmap["bit_count"] = True
    elif fault == "bitmap_count_wrong": bitmap["bit_count"] = 124
    elif fault == "raw_count_bool": filtered["raw_row_count"] = True
    elif fault == "kept_count_wrong": filtered["kept_row_count"] = 124
    elif fault == "excluded_count_bool": filtered["excluded_row_count"] = False
    elif fault == "kept_sha": filtered["keep_raw_rows_sha256"] = "f"*64
    elif fault == "excluded_sha": filtered["excluded_raw_rows_sha256"] = "f"*64
    elif fault.startswith("window_filter_"):
        window["surface_role_filter_index"] = dict(window_filter_bool=False,
            window_filter_negative=-1, window_filter_range=1)[fault]
    elif fault == "window_context_bool": window["context_index"] = False
    elif fault == "window_context_range": window["context_index"] = 48
    elif fault == "window_counts": window["local_input_rows"] -= 1
    elif fault.startswith("association_filter_"):
        association["surface_role_filter_index"] = dict(association_filter_bool=False,
            association_filter_negative=-1, association_filter_range=1)[fault]
    elif fault == "association_context_missing": association["context_index"] = 1
    elif fault == "association_feature_missing": association["feature_id"] = "ABSENT_WINDOW"
    elif fault == "association_flange_order_mismatch": association["ordered_flange_raw_rows"][0][0] = 0
    elif fault in ("excluded_arc", "excluded_flange"):
        _exclude_published_row(result, 0 if fault == "excluded_arc" else 75)
    else:
        _add_second_association_filter(result)
        _publication(mod, s, result)  # Distinct fully kept filters themselves remain valid.
        _exclude_published_row(result, 0, filter_index=1, window_index=1)
    with pytest.raises(ValueError):
        _publication(mod, s, result)


def test_publication_checks_all_alias_filters_without_discarding_valid_alternatives(mod, monkeypatch):
    s = _setup(mod, monkeypatch, contexts=2)
    result = s.invoke()
    assert result["status"] == "PARTIAL", result["reasons"]
    _add_second_association_filter(result)
    _publication(mod, s, result)
    assert len(result["support_records"][0]["associations"]) == 2
    assert len(result["surface_role_filters"]) == 2
    assert len(result["predictions"]) == 1 and result["observed_bend_count"] is None
