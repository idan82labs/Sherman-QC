"""Fixed supplied-object ledger controls; every scientific validator is mocked.

No engine/native/fixture geometry is imported or invoked. The two 650/1176 row
buffers below are invented identity payloads, not a measurement-positive CAD
case. These tests qualify orchestration/common predicates and failure reporting,
not the correctness of the frozen validators or the native accuracy oracle.
"""

import ast
import copy
from contextlib import nullcontext
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest

from tests import native_depth_diagnostic_ledger_v1 as ledger


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = "tests/test_cad_guided_two_plane_depth_native_v1.py"
HELPER_SOURCE = "tests/native_depth_diagnostic_ledger_v1.py"
SOURCE_DEPENDENCIES = (NATIVE_SOURCE, HELPER_SOURCE, "tests/__init__.py")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def typed_equal(a, b):
    """Mock-validator oracle: serialized equality PLUS bool/int type identity."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


class TruthBomb:
    def __getattribute__(self, name):
        raise AssertionError("TRUTH_MUST_NOT_BE_INSPECTED")

    def __iter__(self):
        raise AssertionError("TRUTH_MUST_NOT_BE_ITERATED")

    def __repr__(self):
        raise AssertionError("TRUTH_MUST_NOT_BE_RENDERED")


@dataclass(frozen=True)
class InventedBudget:
    limit: int = 7


POLICY_KEYS = dict(workflow="workflow_policy", registration="registration_policy", measurement="measurement_policy",
    support="instance_policy", noisy="noisy_policy", noisy_budgets="noisy_budgets", cad_patch="cad_patch_policy",
    cad_patch_budgets="cad_patch_budgets", search="search_policy", search_budgets="search_budgets",
    initialization_budgets="initialization_budgets", surface_budgets="surface_budgets", registration_budgets="registration_budgets")


@pytest.fixture
def supplied(monkeypatch):
    """Complete *mock* source transport with two contexts and two native roles.

    Loaded-module seams are explicit stubs: no hidden scientific qualification.
    All actual new-helper source/common/context/association checks still run.
    """
    calls = []
    modules = {}
    for name in ledger.BACKENDS:
        module = types.ModuleType("backend."+name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, "backend."+name, module)
    monkeypatch.setattr(ledger, "_deadline", lambda seconds: nullcontext())
    stage = modules["cad_guided_stage_policies_v1"]
    stage.PROXIMITY = "MOCK_DISTANCE_ONLY_POLICY"
    stage.ablation_stage_policies_v1 = lambda *, connected: {"connected": connected}
    raw = np.arange(650*3, dtype=np.float64).reshape(650, 3)/100.
    triangles = np.array([[i % 648, i % 648+1, i % 648+2] for i in range(1176)], dtype=np.int64)
    cad_blob, scan_blob = b"INVENTED_STEP_IDENTITY_ONLY", b"INVENTED_PLY_IDENTITY_ONLY"
    hashes = dict(cad_sha256=sha(cad_blob), scan_sha256=sha(scan_blob), raw_xyz_sha256=sha(raw.astype("<f8").tobytes()),
        triangle_raw_rows_sha256=sha(triangles.astype("<i8").tobytes()))
    kwargs = {key: InventedBudget() for key in POLICY_KEYS.values()}
    kwargs.update(two_plane_schedule_budgets=InventedBudget(), seeded_plane_budgets=InventedBudget(),
        composite_plane_budgets=InventedBudget(), scan_plane_mode="composite_v1", two_plane_schedule_mode="v2",
        position_error_bound_mm=.05, allow_ocp=True, connected_measurement_contexts=True, two_plane_enabled=True)
    config = dict(mode="v2", budgets={"limit": 7})
    fixed_sources = {"tests/invented_source_only.py": "a"*64}
    fixture = types.ModuleType("INVENTED_LOADED_NATIVE_FIXTURE")
    fixture.SOURCE_HASHES = fixed_sources
    native_cache = dict(currsize=1, misses=1, hits=1, maxsize=3)
    fixture._fixture_module = types.SimpleNamespace(cache_info=lambda: types.SimpleNamespace(**native_cache))
    fixture._policy_objects = lambda: copy.deepcopy(kwargs)
    previous = types.SimpleNamespace(_fixture_module=types.SimpleNamespace(
        cache_info=lambda: types.SimpleNamespace(currsize=2, misses=2, hits=1, maxsize=2)))
    monkeypatch.setitem(sys.modules,
        "_two_plane_depth_native_v1_fixture_test_cad_guided_connected_v12_native_bytes", previous)
    for name in ("_connected_v12_native_fixture_test_cad_guided_mesh_connected_native_v1",
                 "_connected_v12_native_fixture_test_cad_guided_connected_v12"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    def forbidden(*args, **kw):
        raise AssertionError("NO_FILE_NATIVE_INFERENCE_REFIT_OR_ACCURACY_CALL")
    for name in ("_source_bytes", "_make_fixture", "_execute_workflow", "_evaluate", "_legacy"):
        setattr(fixture, name, forbidden)
    def rigid(matrix):
        calls.append("rigid")
        assert matrix.shape == (4, 4) and np.all(np.isfinite(matrix))
        assert np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-8, rtol=0)
        assert abs(np.linalg.det(matrix[:3, :3])-1) <= 1e-8
        return matrix[:3, :3], matrix[:3, 3]
    modules["cad_guided_mesh_offline_v2"]._rigid_transform = rigid
    digests = dict(scan_points_sha256=hashes["raw_xyz_sha256"], scan_triangle_raw_rows_sha256=hashes["triangle_raw_rows_sha256"],
        position_error_bounds_mm_sha256=sha(np.full(650, .05, dtype="<f8").tobytes()),
        cad_triangles_sha256="c"*64, cad_face_ids_sha256="d"*64, cad_source_triangle_ids_sha256="e"*64)
    labels = dict(numerical_support_policy="DEFINITE_UPPER_BOUND_WITH_UNCERTAINTY_EXCLUSION",
        distance_model="CONDITIONAL_REPRESENTED_TRIANGLE_HULL_INTERVAL",
        optimizer_objective="MATERIALIZED_CANDIDATE_FIXED_SET_HUBER_SURROGATE_NOT_CERTIFIED_SURFACE_LOSS")
    rotation = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    translation = [1., 2., 3.]
    pose_sha = sha(np.array([1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 2., 3.], dtype="<f8").tobytes())
    candidates, hypotheses, start_groups = [], [], []
    for index in range(2):
        candidates.append(dict(labels, index=index, execution_status="ATTEMPTED", exact_group_index=index,
            numerical_consumer_status="COMPLETE_V3_INTERVAL_SUPPORT_POLICY", rotation=copy.deepcopy(rotation),
            translation=list(translation), initialized_rotation=copy.deepcopy(rotation), initialized_translation_mm=list(translation),
            hypothesis_indices=[index], conditional_context_support=dict(status="SUPPORTED_CONTEXT",
                accepted_registration=False, pose_sha256=pose_sha)))
        start_groups.append(dict(index=index, exact_group_index=index, hypothesis_indices=[index],
            refinement_disposition="SCHEDULED_FOR_V3_REFINEMENT", initialized_rotation=copy.deepcopy(rotation),
            initialized_translation_mm=list(translation)))
        pairs = [dict(scan_plane_id=j, cad_plane_id=10+j, normal_sign=1) for j in range(3)]
        hypotheses.append(dict(hypothesis_index=index, exact_start_group_index=index, prefilter_status="PASSED",
            proposal_lane="TRIPLET", rotation=copy.deepcopy(rotation), translation_mm=list(translation),
            assigned_support=[dict(row, status="PASSED") for row in pairs], correspondences=pairs))
    init = dict(schema_version="noisy_plane_initialization_prefilter_v12", status="COMPLETE",
        numerical_consumer_complete=True, box_screening_complete=True,
        box_screening_policy="REJECTION_ONLY_COMPLETE_BOUND_FACE_AND_SUPPORT_V1", numerical_support_policy=stage.PROXIMITY,
        input_digests=copy.deepcopy(digests), hypotheses=hypotheses, start_groups=start_groups,
        all_passing_start_groups=copy.deepcopy(start_groups), budgets={"limit": 7},
        two_plane_schedule_configuration=copy.deepcopy(config),
        two_plane_schedule=dict(schema_version="two_plane_schedule_v2", status="COMPLETE",
            selected_policy_evaluation_complete=True, budgets={"limit": 7}, scheduled_matches=[], input_digests={"mock": "f"*64}),
        joint_proposal_accounting=dict(pair_lane_enabled=True, triplet_records=2, triplet_unsigned_reservation=2,
            ordered_pair_matches=0, pair_slot_reservation=0, total_hypothesis_records=2),
        mock_composite_configuration={"mode": "composite_v1", "complete": True})
    evidence = dict(labels, schema="NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V12", initialization=init,
        candidate_evaluation_complete=True, numerical_consumer_complete=True, accepted_registration=False,
        global_pose_uniqueness="NOT_ESTABLISHED", connected_measurement_contexts=True, stage_policies={"connected": True},
        input_digests=copy.deepcopy(digests), candidates=candidates, measurement_context_indices=[0, 1],
        cad_initialization_points_sha256="b"*64, two_plane_schedule_configuration=copy.deepcopy(config))
    features = [dict(feature_id="F"+str(i), nominal_radius_mm=5., nominal_sweep_deg=90.) for i in range(2)]
    association = lambda i: dict(context_index=i, feature_id="F"+str(i), nominal_source="OCP_IMPORT_LOCAL_V1:invented",
        nominal_radius_mm=5., nominal_sweep_deg=90., accepted_registration=False,
        canonical_radius_deviation_mm=.3, canonical_sweep_magnitude_deviation_deg=0.)
    quantity = dict(measured_radius_mm=5.3, measured_sweep_magnitude_deg=90., measured_sweep_deg=90.,
        geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM", status="CONDITIONAL_SCAN_POINT_ESTIMATE",
        radius_uncertainty_interval_mm=None, sweep_uncertainty_interval_deg=None, physical_trim_endpoints=None,
        datum_relative_deviation=None)
    groups = [dict(group_id=str(i)*64, status="MEASURED_CONDITIONAL", measurements=copy.deepcopy(quantity), reasons=[],
        associations=[association(i)], finite_support=dict(status="SUPPORTED", instance={"mock_carrier_id": i})) for i in range(2)]
    original = dict(computation_status="COMPLETE", support_records=[dict(record_id=str(i)*64, associations=[association(i)])
        for i in range(2)], context_windows=[dict(context_index=i, feature_id="F"+str(j)) for i in range(2) for j in range(2)])
    proof = dict(cover_plan_revalidated=True, cover_refit_repeated=False, cover_plan_work={"mock": 1})
    measured = dict(schema="CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2", status="PARTIAL", computation_status="COMPLETE",
        source_context_result=original, source_context_result_role="RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS",
        support_groups=groups, predictions=[copy.deepcopy(row["finite_support"]["instance"]) for row in groups], reasons=[],
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        joint_configuration_status="NOT_ESTABLISHED", continuous_pose_family_coverage="NOT_ESTABLISHED",
        context_quantifier="EXISTS_COMPLETED_EXISTING_COVER_CONTEXT", scan_sha256=hashes["scan_sha256"],
        raw_points_sha256=hashes["raw_xyz_sha256"])
    result = dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status="PARTIAL", phase="COMPLETE", reasons=[],
        connected_measurement_contexts=True, observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE", prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        measurement_geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        position_error_model=dict(uniform_euclidean_bound_mm=.05, status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
            scope="PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY"),
        cad_sha256=hashes["cad_sha256"], input_preparation=dict(source_bindings=dict(scan_sha256=hashes["scan_sha256"],
            raw_xyz_sha256=hashes["raw_xyz_sha256"]), coverage={"raw_vertex_count": 650}),
        scan_triangle_binding=dict(status="CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT",
            scope="SAME_VALIDATED_BYTES_AND_ORIGINAL_ROW_IDENTITIES_NOT_MATERIAL_TOPOLOGY", source_sha256=hashes["scan_sha256"],
            raw_points_sha256=hashes["raw_xyz_sha256"], triangle_raw_rows_sha256=hashes["triangle_raw_rows_sha256"],
            raw_vertex_count=650, triangle_count=1176), cad_units=dict(target_unit="mm", transferred_roots=1,
            target_unit_action="SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER", physical_calibration="NOT_ESTABLISHED",
            source_unit_names={"length": ["MM"]}), native_features=dict(adapter_version="NATIVE_TRANSITION_ROI_V1",
            candidate_inventory_complete=True, exhaustive_bend_inventory=False, physical_observed_bend_count=None,
            production_use_allowed=False, expected_roi_count=2, features=features),
        cad_surface_sampling=dict(status="AVAILABLE", sampling_policy={"sample_count": 32768},
            digests=dict(oriented_triangle_xyz_sha256="c"*64, source_face_ids_sha256="d"*64,
                source_triangle_ids_sha256="e"*64, sample_points_sha256="a"*64)),
        triangle_sample_binding=dict(initialization_sample_count=32768, oriented_triangle_xyz_sha256="c"*64),
        registration=dict(schema_version="noisy_plane_initialized_finite_registration_v12", status="CONDITIONAL_MEASUREMENT_CONTEXTS",
            transform_scan_to_cad=None, evidence=evidence), registration_array_binding=[dict(raw_scan_points_sha256=hashes["raw_xyz_sha256"],
            raw_triangle_rows_sha256=hashes["triangle_raw_rows_sha256"], position_error_bounds_mm_sha256=digests["position_error_bounds_mm_sha256"],
            cad_triangles_sha256="c"*64, cad_sampler_points_sha256="a"*64,
            registration_canonical_cad_points_sha256="b"*64, registration_selection="ALL_RAW_ROWS") for _ in range(2)],
        measurements=measured, predictions=copy.deepcopy(measured["predictions"]), support_cover_publication_validation=copy.deepcopy(proof),
        prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS",
        policies={key: asdict(kwargs[value]) for key, value in POLICY_KEYS.items()})
    result["policies"].update(two_plane_enabled=True, stage_policies={"connected": True},
        two_plane_schedule_configuration=copy.deepcopy(config),
        scan_plane_configuration=dict(mode="composite_v1", seeded_budgets={"limit": 7},
            composite_budgets={"limit": 7}))
    record = dict(status="RETURNED_PENDING_ORACLE", phase="RETURNED_PENDING_ORACLE", workflow=result,
        full_returned_workflow_retained=True, source_consistency=True, execution_error=None,
        source_sha256=fixed_sources, input_binding=dict(hashes, raw_rows=650, triangle_rows=1176,
            position_error_bound_mm=.05, cad_placement_scope="DECLARED_NOMINAL_CAD_FRAME_NOT_SCAN_POSE"),
        input_artifacts=[dict(filename="CAD.step", bytes=len(cad_blob), sha256=hashes["cad_sha256"]),
            dict(filename="SCAN.ply", bytes=len(scan_blob), sha256=hashes["scan_sha256"])])
    def composite(value, supplied_raw, supplied_triangles):
        calls.append("composite")
        assert value is result and supplied_raw is raw and supplied_triangles is triangles
        assert typed_equal(init["mock_composite_configuration"], {"mode": "composite_v1", "complete": True})
    fixture._assert_composite_binding = composite
    def schedule(registration, expected):
        calls.append("schedule")
        assert registration is result["registration"] and typed_equal(expected, config)
        assert typed_equal(init["two_plane_schedule_configuration"], config)
        assert init["two_plane_schedule"]["schema_version"] == "two_plane_schedule_v2"
        return {"mock_bound": True}
    modules["cad_guided_mesh_offline_v12"]._two_plane_schedule_binding = schedule
    def schedule_metadata(value, expected, budgets, bound):
        calls.append("schedule_metadata")
        assert value is result and typed_equal(expected, config) and typed_equal(budgets, {"limit": 7})
        assert bound == {"mock_bound": True}
    fixture._assert_schedule_metadata = schedule_metadata
    def cover(value, *, raw_points, scan_sha256):
        calls.append("cover")
        assert value is measured and raw_points is raw and scan_sha256 == hashes["scan_sha256"]
        return copy.deepcopy(proof)
    modules["cad_guided_support_cover_measurement_v2"].validate_support_cover_publication_v2 = cover
    def projection(value):
        calls.append("projection")
        expected = [g["finite_support"]["instance"] for g in groups
            if g["status"] == "MEASURED_CONDITIONAL" and g["finite_support"]["status"] == "SUPPORTED"]
        assert typed_equal(value["predictions"], measured["predictions"]) and typed_equal(value["predictions"], expected)
    fixture._assert_top_level_cover_projection = projection
    return types.SimpleNamespace(fixture=fixture, record=record, result=result, raw=raw, triangles=triangles,
        cad=cad_blob, scan=scan_blob, hashes=hashes, calls=calls, modules=modules, kwargs=kwargs, proof=proof,
        native_cache=native_cache)


def evaluate(supplied, artifact=None):
    if artifact is None:
        blob = encode(supplied.record)
        artifact = dict(filename="CASE_RESULT.json", bytes=len(blob), sha256=sha(blob))
    return ledger.evaluate_native_diagnostic_v1(supplied.fixture, supplied.record, artifact,
        supplied.cad, supplied.scan, supplied.raw, supplied.triangles, TruthBomb())


def criterion(answer, name):
    return next(row for row in answer["criteria"] if row["name"] == name)


@pytest.mark.parametrize("measured", [True, False])
def test_complete_mock_routes_keep_integrity_separate_from_accuracy(supplied, measured):
    if not measured:
        result = supplied.result
        value = result["measurements"]
        for group in value["support_groups"]:
            group.update(status="UNAVAILABLE", measurements=None, finite_support=None, reasons=["NO_EXISTING_COVER"])
        value.update(status="UNAVAILABLE", predictions=[], reasons=["NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS"])
        result.update(status="UNAVAILABLE", predictions=[], reasons=list(value["reasons"]))
    before = encode(supplied.record)
    answer = evaluate(supplied)
    assert answer["status"] == ("COMPLETE_CONDITIONAL_INTEGRITY" if measured else "COMPLETE_NO_MEASUREMENTS_INTEGRITY")
    assert len(answer["criteria"]) == 8 and all(row["status"] == "PASS" for row in answer["criteria"])
    assert answer["immutable_supplied_evidence"] is True and encode(supplied.record) == before
    assert answer["record_before"] == answer["record_after"] == dict(bytes=len(before), sha256=sha(before))
    assert answer["denominators"] == dict(native_features=2, contexts=2, windows=4, support_records=2, support_groups=2,
        predictions=2 if measured else 0)
    assert answer["accuracy_evaluated"] is False and answer["file_identity_reverified"] is False
    assert answer["observed_bend_count"] is None and answer["production_use_allowed"] is False
    assert supplied.calls.count("cover") == 1 and supplied.calls.count("projection") == 1
    assert len(encode(answer)) <= 32768


def test_truth_and_forbidden_function_bombs_never_called(supplied):
    answer = evaluate(supplied)
    assert answer["truth_accessed"] is False and answer["inference_calls"] == answer["refit_calls"] == 0
    assert set(supplied.calls) == {"rigid", "composite", "schedule", "schedule_metadata", "cover", "projection"}


@pytest.mark.parametrize("fault", [None, "unexpected_miss", "module_reload"])
def test_post_accuracy_named_two_entry_cache_allowed_but_new_load_refused(supplied, monkeypatch, fault):
    supplied.native_cache.update(currsize=2, misses=2, hits=3)
    array_name = "_two_plane_depth_native_v1_fixture_test_cad_guided_connected_v12"
    original_module = types.ModuleType(array_name)
    monkeypatch.setitem(sys.modules, array_name, original_module)
    original = supplied.fixture._policy_objects
    def policies():
        answer = original()
        supplied.native_cache["hits"] += 1
        if fault == "unexpected_miss": supplied.native_cache["misses"] += 1
        elif fault == "module_reload": monkeypatch.setitem(sys.modules, array_name, types.ModuleType(array_name))
        return answer
    monkeypatch.setattr(supplied.fixture, "_policy_objects", policies)
    answer = evaluate(supplied)
    if fault is None:
        assert answer["status"] == "COMPLETE_CONDITIONAL_INTEGRITY"
        assert sys.modules[array_name] is original_module
    else:
        assert answer["status"] in ("FAIL", "NOT_EVALUABLE")
        assert criterion(answer, "composite_schedule_profile")["status"] == "FAIL"
        assert supplied.calls == []


def test_int64_min_cannot_bypass_raw_absolute_range(supplied):
    supplied.raw = np.zeros((650, 3), dtype=np.int64)
    supplied.raw[0, 0] = np.iinfo(np.int64).min
    answer = evaluate(supplied)
    assert answer["status"] in ("FAIL", "NOT_EVALUABLE")
    assert answer["inputs_before"] is None and supplied.calls == []


@pytest.mark.parametrize("fault", ["cad_bytes", "scan_bytes", "source", "result_hash", "raw_hash", "pose_bool", "pose_string",
    "origin_late", "schedule_bool", "composite_bool", "projection", "deviation", "window_duplicate", "status_spoof",
    "policy_bool", "carrier_semantics", "validator_receipt"])
def test_independent_tampering_cannot_admit_integrity(supplied, fault):
    result, record = supplied.result, supplied.record
    evidence = result["registration"]["evidence"]
    artifact = None
    if fault == "cad_bytes": supplied.cad += b"changed"
    elif fault == "scan_bytes": supplied.scan += b"changed"
    elif fault == "source": record["source_sha256"] = {"tests/invented_source_only.py": "b"*64}
    elif fault == "result_hash": artifact = dict(filename="CASE_RESULT.json", bytes=len(encode(record)), sha256="0"*64)
    elif fault == "raw_hash": result["registration_array_binding"][-1]["raw_scan_points_sha256"] = "0"*64
    elif fault == "pose_bool": evidence["candidates"][-1]["rotation"][0][0] = True
    elif fault == "pose_string": evidence["candidates"][-1]["translation"][0] = "1"
    elif fault == "origin_late": evidence["initialization"]["hypotheses"][-1]["correspondences"][-1]["normal_sign"] = False
    elif fault == "schedule_bool": evidence["initialization"]["two_plane_schedule_configuration"]["budgets"]["limit"] = True
    elif fault == "composite_bool": evidence["initialization"]["mock_composite_configuration"]["complete"] = 1
    elif fault == "projection": result["predictions"].append({"nested_member": True})
    elif fault == "deviation": result["measurements"]["support_groups"][-1]["associations"][-1]["canonical_radius_deviation_mm"] = 0.
    elif fault == "window_duplicate": result["measurements"]["source_context_result"]["context_windows"][-1]["feature_id"] = "F0"
    elif fault == "status_spoof": result["status"] = "UNAVAILABLE"
    elif fault == "policy_bool": result["policies"]["measurement"]["limit"] = True
    elif fault == "carrier_semantics": result["measurements"]["support_groups"][-1]["measurements"]["physical_trim_endpoints"] = [[0., 0., 0.], [1., 1., 1.]]
    elif fault == "validator_receipt": result["support_cover_publication_validation"]["cover_refit_repeated"] = True
    answer = evaluate(supplied, artifact)
    assert answer["status"] in ("FAIL", "NOT_EVALUABLE")
    assert any(row["status"] == "FAIL" for row in answer["criteria"])
    if fault not in ("cad_bytes", "scan_bytes", "source", "result_hash"):
        assert supplied.calls.count("cover") == 1 and supplied.calls.count("projection") == 1


def test_all_late_associations_checked_and_bulk_exception_progress_unknown(supplied, monkeypatch):
    rows = supplied.result["measurements"]["support_groups"][1]["associations"]
    rows[:] = [dict(rows[0], canonical_radius_deviation_mm=99.) for _ in range(12)]
    def failed_cover(*args, **kwargs):
        supplied.calls.append("cover")
        raise RuntimeError("INVENTED_LATE_BULK_VALIDATOR_FAILURE")
    monkeypatch.setattr(supplied.modules["cad_guided_support_cover_measurement_v2"], "validate_support_cover_publication_v2", failed_cover)
    answer = evaluate(supplied)
    nominal = criterion(answer, "windows_nominals")
    assert nominal["failed"] == 12 and nominal["planned"] == nominal["visited"]
    assert len(nominal["errors"]) == 8 and nominal["errors_truncated"] is True
    bulk = criterion(answer, "raw_support_cover")["bulk_proofs"][0]
    assert bulk["validator_returned"] is False and bulk["internally_visited_rows"] is None
    assert bulk["unreached_rows_upper_bound"] == bulk["declared_rows"] == 10
    assert criterion(answer, "quantity_carrier_projection")["status"] == "PASS"


@pytest.mark.parametrize("target", ["record", "raw", "triangles"])
def test_any_late_input_mutation_is_reported_without_rewriting_evidence(supplied, monkeypatch, target):
    original = supplied.fixture._assert_top_level_cover_projection
    def mutate(result):
        original(result)
        if target == "record": supplied.record["late_mutation"] = True
        elif target == "raw": supplied.raw[0, 0] += 1.
        else: supplied.triangles[0, 0] = 4
    monkeypatch.setattr(supplied.fixture, "_assert_top_level_cover_projection", mutate)
    answer = evaluate(supplied)
    assert answer["status"] == "FAIL" and answer["immutable_supplied_evidence"] is False
    assert criterion(answer, "publication_source_input")["status"] == "FAIL"
    if target == "record":
        assert supplied.record["late_mutation"] is True


@pytest.mark.parametrize("limit", ["nodes", "depth", "record_bytes", "ledger_bytes"])
def test_whole_caps_never_return_successful_prefix(supplied, monkeypatch, limit):
    if limit == "nodes": monkeypatch.setattr(ledger, "MAX_NODES", 50)
    elif limit == "depth": monkeypatch.setattr(ledger, "MAX_DEPTH", 2)
    elif limit == "record_bytes": monkeypatch.setattr(ledger, "MAX_RECORD_BYTES", 64)
    else: monkeypatch.setattr(ledger, "MAX_LEDGER_BYTES", 256)
    if limit == "ledger_bytes":
        with pytest.raises(ValueError, match="DIAGNOSTIC_LEDGER_BYTE_LIMIT"):
            evaluate(supplied)
    else:
        answer = evaluate(supplied)
        assert answer["status"] in ("FAIL", "NOT_EVALUABLE") and supplied.calls == []


@pytest.mark.parametrize("fault", [None, "downstream_output", "discarded_result", "stale_binding",
    "stale_native_report", "null_registration", "unexpected_field", "missing_composite_request",
    "changed_schedule_request"])
def test_limited_early_refusal_does_not_claim_native_route(supplied, fault):
    # Exact real V13 base plus only its reached input-preparation publication.
    # No stale downstream keys, even null ones, belong to this source branch.
    result = dict(schema="CAD_GUIDED_MESH_OFFLINE_V13", status="UNAVAILABLE", phase="INPUT_PREPARATION",
        reasons=["MESH_INPUT_UNAVAILABLE", "INVENTED_FORMAT_REFUSAL"], predictions=[], measurements=None,
        observed_bend_count=None, production_use_allowed=False, conformance="INDETERMINATE",
        uncertainty_status="UNCALIBRATED", units_status="PROVISIONAL_COMMON_MM_NOT_CALIBRATED",
        registration_mode="FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE", datum_status="NO_VERIFIED_STABLE_DATUM",
        registration_surface="FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD",
        registration_initialization="TRIPLET_AND_TWO_PLANE_CONTEXT_FINITE_PREFILTER_V12",
        measurement_geometry_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM",
        prediction_frame="ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM", reference_frame_compatibility="NOT_ASSERTED_BY_INFERENCE",
        connected_measurement_contexts=True, position_error_model=copy.deepcopy(supplied.result["position_error_model"]),
        policies=copy.deepcopy(supplied.result["policies"]), input_preparation=dict(status="UNAVAILABLE",
            reasons=["INVENTED_FORMAT_REFUSAL"], source_bindings={}, coverage=None,
            measurement_cloud_contract="FULL_VALIDATED_RAW_XYZ_REQUIRED_FOR_LOCAL_FITTING",
            sample_cloud_contract="REGISTRATION_AND_PROPOSALS_ONLY", units_status="UNVERIFIED", coordinate_frame_status="UNVERIFIED"))
    supplied.record["workflow"] = result
    if fault == "downstream_output": result["predictions"] = [{"forbidden": True}]
    elif fault == "discarded_result": supplied.record["full_returned_workflow_retained"] = False
    elif fault == "stale_binding": result["registration_array_binding"] = [{"stale": True}]
    elif fault == "stale_native_report": result["native_features"] = {"status": "AVAILABLE"}
    elif fault == "null_registration": result["registration"] = None
    elif fault == "unexpected_field": result["future_report"] = None
    elif fault == "missing_composite_request": del result["policies"]["scan_plane_configuration"]
    elif fault == "changed_schedule_request": result["policies"]["two_plane_schedule_configuration"]["budgets"]["limit"] = True
    answer = evaluate(supplied)
    if fault is None:
        assert answer["status"] == "SAFE_EARLY_REFUSAL_LIMITED"
        assert criterion(answer, "raw_native_sampler")["status"] == "UNREACHED"
        assert criterion(answer, "raw_support_cover")["prerequisite"].startswith("NOT_APPLICABLE_BY_VALID_REFUSAL")
    else:
        assert answer["status"] in ("FAIL", "NOT_EVALUABLE")
    assert supplied.calls == [] and answer["accuracy_evaluated"] is False


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, ledger._BudgetInterrupt])
def test_baseexception_is_never_turned_into_completed_diagnostic(supplied, monkeypatch, interrupt):
    def stop(*args, **kwargs):
        raise interrupt("INVENTED_INTERRUPT")
    monkeypatch.setattr(supplied.fixture, "_assert_composite_binding", stop)
    with pytest.raises(interrupt):
        evaluate(supplied)


@pytest.mark.parametrize("initial", [0., 20., 200.])
def test_nested_deadline_never_extends_shorter_enclosing_timer(monkeypatch, initial):
    calls = []
    old_handler = object()
    clock = iter((10., 17.))
    monkeypatch.setattr(ledger.signal, "getitimer", lambda which: (initial, 0.))
    monkeypatch.setattr(ledger.signal, "getsignal", lambda which: old_handler)
    monkeypatch.setattr(ledger.signal, "signal", lambda *args: calls.append(("handler", *args)))
    monkeypatch.setattr(ledger.signal, "setitimer", lambda *args: calls.append(("timer", *args)))
    monkeypatch.setattr(ledger.time, "monotonic", lambda: next(clock))
    with ledger._deadline(120.):
        pass
    timers = [row[2] for row in calls if row[0] == "timer"]
    assert timers == ({0.: [120., 0.], 20.: [], 200.: [120., 0., 193.]}[initial])
    if initial != 20.:
        assert calls[-1][0] == ("timer" if initial else "handler")
        assert any(row[0] == "handler" and row[-1] is old_handler for row in calls)


def test_exact_float64_serialization_and_cpu_boundary(supplied, monkeypatch):
    supplied.record["represented_float64"] = np.float64(-0.)
    answer = evaluate(supplied)
    assert answer["status"] == "COMPLETE_CONDITIONAL_INTEGRITY"
    assert type(supplied.record["represented_float64"]) is np.float64
    assert np.signbit(supplied.record["represented_float64"])
    work = ledger._Work()
    monkeypatch.setattr(ledger.time, "process_time", lambda: work.cpu+90.)
    with pytest.raises(ledger._BudgetInterrupt):
        work.boundary()


@pytest.mark.parametrize("fault", [None, "boolean_slot_count", "wrong_input_digest", "wrong_schema"])
def test_context_v2_profile_has_typed_global_slot_and_input_contract(supplied, fault):
    """Pure report profile only: no pair scheduler or context geometry executes."""
    init = copy.deepcopy(supplied.result["registration"]["evidence"]["initialization"])
    init["hypotheses"] = [{} for _ in range(32)]
    init["joint_proposal_accounting"].update(triplet_records=0, triplet_unsigned_reservation=0,
        ordered_pair_matches=1, pair_slot_reservation=32, total_hypothesis_records=32)
    init["two_plane_schedule"]["scheduled_matches"] = [{"mock_match": True}]
    profile = dict(schema_version="two_plane_context_v2", status="COMPLETE", selected_policy_evaluation_complete=True,
        accepted_registration=False, finite_support_match_verified=False, statistics={"slot_records_completed": 32},
        input_digests={key+"_sha256": init["input_digests"][name+"_sha256"] for key, name in (
            ("scan_points", "scan_points"), ("scan_triangles", "scan_triangle_raw_rows"),
            ("cad_triangles", "cad_triangles"), ("cad_face_ids", "cad_face_ids"),
            ("cad_source_triangle_ids", "cad_source_triangle_ids"))})
    init["two_plane_context"] = profile
    def validate(value, mode):
        supplied.calls.append("context_profile")
        assert value is profile and mode == "v2" and value["schema_version"] == "two_plane_context_v2"
    supplied.modules["cad_guided_plane_initialization_v12"]._validate_two_plane_context_configuration = validate
    if fault == "boolean_slot_count": profile["statistics"]["slot_records_completed"] = True
    elif fault == "wrong_input_digest": profile["input_digests"]["scan_triangles_sha256"] = "0"*64
    elif fault == "wrong_schema": profile["schema_version"] = "two_plane_context_v1"
    if fault is None:
        ledger._profile(init)
    else:
        with pytest.raises((ValueError, AssertionError)):
            ledger._profile(init)
    assert supplied.calls == ["context_profile"]


@pytest.mark.parametrize("fault", ["nonfinite", "cycle", "string_subclass_key", "array_in_report"])
def test_unsupported_json_tree_refuses_before_loaded_scientific_predicates(supplied, fault):
    if fault == "nonfinite": supplied.record["invalid"] = float("nan")
    elif fault == "cycle": supplied.record["invalid"] = supplied.record
    elif fault == "array_in_report": supplied.record["invalid"] = np.array([1.])
    else:
        class StringSubclass(str): pass
        supplied.record[StringSubclass("invalid")] = 1
    artifact = dict(filename="CASE_RESULT.json", bytes=1, sha256="0"*64)
    answer = evaluate(supplied, artifact)
    assert answer["status"] in ("FAIL", "NOT_EVALUABLE")
    assert supplied.calls == [] and answer["immutable_supplied_evidence"] is False


def test_helper_source_has_no_truth_reads_or_file_engine_dispatch():
    tree = ast.parse((ROOT/HELPER_SOURCE).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_native_diagnostic_v1")
    assert [argument.arg for argument in function.args.args] == ["fixture_module", "record", "result_artifact",
        "cad_blob", "scan_blob", "raw", "triangles", "truth"]
    assert not any(isinstance(node, ast.Name) and node.id == "truth" for node in ast.walk(function))
    forbidden = {"open", "read_bytes", "read_text", "_execute_workflow", "_make_fixture", "_evaluate", "_canonical_fit",
        "inspect_cad_guided_mesh_v13", "register_measurement_contexts_v12", "import_module", "exec", "eval", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            assert name not in forbidden
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("backend", "OCP"))


@pytest.mark.parametrize("accuracy_fails,diagnostic_fails", [(False, False), (True, False), (False, True), (True, True),
    (False, "MUTATION")])
def test_original_recorder_durability_accuracy_first_and_always_supplemental(tmp_path, accuracy_fails, diagnostic_fails):
    """AST-isolated exact recorder, fake publication/inference, fresh empty dir.

    This is an ordering control, not a real durable-file or engine smoke.
    No original test fixture helper, native import or original accuracy executes.
    """
    tree = ast.parse((ROOT/NATIVE_SOURCE).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_single_native_byte_case")
    events, publications = [], {}
    raw, triangles = np.zeros((650, 3)), np.zeros((1176, 3), dtype=np.int64)
    fake_fixture = b"cad", b"scan", raw, triangles, TruthBomb()
    result = {"invented_returned_workflow": True}
    def write(directory, filename, value, cap):
        blob = encode(value)
        assert len(blob) <= cap
        publications[filename] = blob
        events.append(filename)
        return dict(filename=filename, bytes=len(blob), sha256=sha(blob))
    def write_bytes(directory, filename, blob, cap):
        publications[filename] = blob
        events.append(filename)
        return dict(filename=filename, bytes=len(blob), sha256=sha(blob))
    def accuracy(value, *args):
        assert "CASE_RESULT.json" in publications and value is result
        events.append("accuracy")
        if accuracy_fails: raise AssertionError("INVENTED_ORIGINAL_ACCURACY_FAILURE")
        return {"original": True}
    def diagnostic(record, artifact, *args):
        assert events[-1] == "accuracy" and record["workflow"] is result
        assert artifact["sha256"] == sha(publications["CASE_RESULT.json"])
        events.append("diagnostic")
        if diagnostic_fails == "MUTATION":
            result["mutated_after_original_accuracy"] = True
            return {"status": "FAIL", "immutable_supplied_evidence": False, "accuracy_evaluated": False}
        if diagnostic_fails: raise RuntimeError("INVENTED_SUPPLEMENTAL_FAILURE")
        return {"status": "COMPLETE_NO_MEASUREMENTS_INTEGRITY", "accuracy_evaluated": False}
    namespace = dict(os=os, Path=Path, time=time, OPT_IN="INVENTED_NO_NATIVE_OPT_IN", SOURCE_HASHES={"mock": "a"*64},
        CASE_ID="L-R", RESOURCE_LIMITS={}, RESOURCE_LIMIT_STATUS="MOCK", ARTIFACT_LIMITS=dict(start_json_bytes=4096,
            cad_bytes=1024, scan_bytes=1024, metadata_json_bytes=1024, result_json_bytes=65536, total_bytes=131072),
        _require_execution_limits=lambda: None, _source_bytes=lambda name: b"INVENTED_SOURCE",
        _make_fixture=lambda: fake_fixture, _execute_workflow=lambda *args: result, _write=write, _write_bytes=write_bytes,
        _check_result_metadata=lambda value: None, _sha=sha,
        _array_sha=lambda value, dtype: sha(np.asarray(value, dtype=dtype).tobytes()),
        _evaluate=accuracy, _diagnostic_evaluate=diagnostic,
        _error=lambda error: {"type": type(error).__name__, "message": str(error)[:200]})
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, NATIVE_SOURCE, "exec"), namespace)
    # No real access guard bypass: every source/inference/publication seam above
    # is fake and this isolated function only creates one fresh empty directory.
    prior = os.environ.get(namespace["OPT_IN"])
    os.environ[namespace["OPT_IN"]] = "1"
    try:
        receipt = namespace["run_single_native_byte_case"](tmp_path/"ARTIFACTS")
    finally:
        if prior is None: os.environ.pop(namespace["OPT_IN"], None)
        else: os.environ[namespace["OPT_IN"]] = prior
    assert events == ["CASE_START.json", "CAD.step", "SCAN.ply", "CASE_RESULT.json", "accuracy", "diagnostic"]
    assert receipt["positive_geometry_passed"] is (not accuracy_fails)
    assert (receipt["oracle_error"] is not None) is accuracy_fails
    raised = diagnostic_fails is True
    assert (receipt["diagnostic_error"] is not None) is raised
    if raised:
        assert receipt["diagnostic_ledger"] is None  # No supplemental admission even if original accuracy passed.
    else:
        assert receipt["diagnostic_ledger"] is not None
    assert receipt["status"] == ("FAIL" if accuracy_fails else "PASS")
    if diagnostic_fails == "MUTATION":
        assert receipt["oracle"] == {"original": True} and receipt["positive_geometry_passed"] is True
        assert receipt["diagnostic_ledger"]["status"] == "FAIL"
        assert json.loads(publications["CASE_RESULT.json"])["workflow"] == {"invented_returned_workflow": True}
        assert result["mutated_after_original_accuracy"] is True
    # The recorder deliberately has no new admission predicate: original PASS
    # alone cannot admit a missing/failed/oversized supplemental ledger. Exact
    # independent POST must enforce that separate criterion, without rerunning.
