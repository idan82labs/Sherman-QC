"""Fixed four-case native V7 successor; source/helper import performs no native IO.

The first three V4/legacy native cases and every original numeric assertion stay
unchanged. One added nominal case has independently generated row-ID-only 3D
noise of Euclidean norm <=.001 mm. All four declare epsilon=.005 mm. The added
case alone uses radius .01 mm, sweep .1 degree, generator/axis endpoints .05 mm
and pose action .20 mm engineering ceilings, not calibration guarantees.

Native work requires root's explicit opt-in and fresh ARTIFACTS child under its
180s wall /120s worker CPU /4GiB RSS /1MiB stdout supervisor. Four complete JSON
case artifacts each <=16MiB, total <=64MiB, precede a positive manifest. No native
or physical work is authorized by import, collection, or this authored source.
"""

import ast
import builtins
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
V4_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_planes_v4.py")
MOCK_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_offline_v2.py")
CONNECTED_CASES = (
    ("v1_nominal_oblique", "v1", False, 1),
    ("v1_missing_arc_oblique", "v1", True, 0),
    ("r0_common_wrapper_nominal_oblique", "r0", False, 1),
    ("v1_bounded_3d_noise_nominal_oblique", "v1", False, 1),
)
CRITERIA = dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
                generator_endpoint_error_mm=.02, pose_action_error_mm=.20)
NOISY_CRITERIA = dict(radius_error_mm=.01, absolute_sweep_error_deg=.1,
                      generator_endpoint_error_mm=.05, pose_action_error_mm=.20)
POSITION_ERROR_BOUND_MM = .005
NOISE_MAX_EUCLIDEAN_MM = .001
NOISE_ROUNDING_RESERVE_FACTOR = .999
RESOURCE_LIMITS = dict(wall_seconds=180, cpu_seconds=120,
    process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
ARTIFACT_LIMITS = dict(per_case_json_bytes=16*1024**2, total_case_json_bytes=64*1024**2)
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_CONNECTED_PLANES_V7"
ARTIFACT_DIRECTORY_ENV = "SHERMAN_MESH_CONNECTED_PLANES_V7_ARTIFACT_DIRECTORY"
NO_START_REASON = "NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS"


@lru_cache(maxsize=None)
def _load(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("_v7_"+path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v4():
    return _load(V4_SOURCE)


def _original():
    return _v4()._surface()._dense()._original()


def _policies(branch):
    from backend.cad_guided_mesh_offline_v7 import MeshWorkflowPolicyV7
    policies = _v4()._policies(branch)
    return {**policies, "workflow_policy": MeshWorkflowPolicyV7(**asdict(policies["workflow_policy"]))}


def _noisy_rows(raw):
    # Three independent modular patterns: no CAD, normals, geometry or truth.
    i = np.arange(len(raw), dtype=np.int64)
    direction = np.column_stack((((17*i+3) % 19-9)/9.,
                                 ((7*i+5) % 23-11)/11.,
                                 ((11*i+2) % 29-14)/14.))
    intended = NOISE_ROUNDING_RESERVE_FACTOR*NOISE_MAX_EUCLIDEAN_MM/math.sqrt(3.)*direction
    noisy = raw+intended
    # The evidence is the represented displacement, not the pre-addition ideal.
    # A fixed 0.1% reserve precedes all cases; never adapt it after observing errors.
    return noisy, noisy-raw


def _adapted_noisy_nominal_assertions():
    original = _original()._nominal_assertions
    # Post-inference oracle only; identical code, detached changed criteria.
    return types.FunctionType(original.__code__, {**original.__globals__, "CRITERIA": dict(NOISY_CRITERIA)},
                              name="_v7_noisy_post_inference_oracle")


def _adapted_missing_arc_assertions():
    original = _v4()._missing_arc_assertions
    return types.FunctionType(original.__code__, {**original.__globals__, "_execution_complete": _execution_complete})


def _execution_complete(registration, *, allow_prefilter_abstention=False):
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    assert initialization["status"] == "COMPLETE" and initialization["search_complete"] is True
    assert initialization["selected_policy_evaluation_complete"] is True
    assert initialization["search_exhaustive"] is False
    assert initialization["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    search = initialization["hypothesis_evidence"]
    assert search["status"] == "COMPLETE" and search["selected_policy_evaluation_complete"] is True
    assert search["search_exhaustive"] is False and search["accepted_registration"] is False
    ss = search["statistics"]
    assert ss["signed_assignments_attempted"] == ss["signed_assignments_evaluated"] == ss["signed_assignments_planned"]
    assert ss["scan_triplets_scheduled"] == ss["scan_triplets_evaluated"]
    hypotheses, stats = initialization["hypotheses"], initialization["statistics"]
    assert stats["hypothesis_count"] == len(hypotheses) == ss["signed_assignments_evaluated"]
    assert [row["hypothesis_index"] for row in hypotheses] == list(range(len(hypotheses)))
    selected = initialization["selected_unsigned_assignment_ids"]
    scheduled = [row for row in hypotheses if row["prefilter_scheduled"]]
    omitted = [row for row in hypotheses if row["status"] == "PROPOSED" and not row["prefilter_scheduled"]]
    refused = [row for row in hypotheses if row["status"] == "REJECTED"]
    assert stats["scheduled_hypothesis_count"] == stats["hypotheses_evaluated"] == len(scheduled)
    assert stats["omitted_proposed_hypothesis_count"] == len(omitted)
    assert stats["search_rejected_hypothesis_count"] == len(refused)
    assert stats["proposed_hypothesis_count"] == len(scheduled)+len(omitted)
    assert len(hypotheses) == len(scheduled)+len(omitted)+len(refused)
    assert stats["passing_hypothesis_count"]+stats["rejected_hypothesis_count"] == len(scheduled)
    for row in hypotheses:
        assert row["prefilter_scheduled"] == (row["status"] == "PROPOSED" and row["unsigned_assignment_id"] in selected)
        if row["prefilter_scheduled"]:
            assert row["prefilter_status"] in ("PASSED", "REJECTED") and len(row["assigned_support"]) == 3
            assert row["prefilter_reasons"] == ([] if row["prefilter_status"] == "PASSED"
                else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"])
            assert all(s["status"] in ("PASSED", "REJECTED")
                and s["query_rows"] == s["attempted_query_rows"] == s["support_count"] > 0
                for s in row["assigned_support"])
            assert all(s["reasons"] == ([] if s["status"] == "PASSED"
                else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"]) for s in row["assigned_support"])
        else:
            assert row["assigned_support"] == []
            assert row["prefilter_status"] == ("SEARCH_NOMINAL_WITNESS_REFUSAL" if row["status"] == "REJECTED"
                else "NOT_SCHEDULED_BY_DECLARED_UNSIGNED_QUOTA")
    groups, all_groups = initialization["start_groups"], initialization["all_passing_start_groups"]
    assert stats["exact_passing_start_group_count"] == len(all_groups)
    assert stats["surviving_start_group_count"] == len(groups)
    assert stats["omitted_passing_start_group_count"] == len(all_groups)-len(groups)
    assert len(groups) <= 48
    selected_group_ids = [i*(len(all_groups)-1)//(min(48, len(all_groups))-1)
        for i in range(min(48, len(all_groups)))] if len(all_groups) > 1 else list(range(len(all_groups)))
    assert [group["exact_group_index"] for group in groups] == selected_group_ids
    origins = []
    for index, group in enumerate(all_groups):
        assert group["exact_group_index"] == index
        assert group["refinement_disposition"] == ("SCHEDULED_FOR_V2_REFINEMENT" if index in selected_group_ids
            else "NOT_SCHEDULED_BY_DECLARED_GROUP_QUOTA")
        assert group["hypothesis_indices"] and all(type(i) is int and 0 <= i < len(hypotheses)
            and hypotheses[i]["prefilter_status"] == "PASSED"
            and hypotheses[i]["exact_start_group_index"] == index for i in group["hypothesis_indices"])
        origins.extend(group["hypothesis_indices"])
    assert sorted(origins) == [row["hypothesis_index"] for row in hypotheses if row["prefilter_status"] == "PASSED"]
    for group in groups:
        assert {key: value for key, value in group.items() if key != "index"} == all_groups[group["exact_group_index"]]
    candidates = evidence["candidates"]
    assert evidence["candidate_numerical_failure_count"] == 0
    if not groups:
        assert allow_prefilter_abstention and scheduled
        assert all(row["prefilter_status"] == "REJECTED" for row in scheduled)
        assert not all_groups and stats["passing_hypothesis_count"] == 0
        assert initialization["reasons"] == registration["reasons"] == [NO_START_REASON]
        assert registration["status"] == "UNAVAILABLE" and candidates == []
        assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == 0
        return "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE"
    assert initialization["reasons"] == []
    assert evidence["candidate_search_complete"] is True and evidence["candidate_evaluation_complete"] is True
    assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == len(groups) == len(candidates)
    for index, (row, group) in enumerate(zip(candidates, groups)):
        assert row["index"] == index and row["execution_status"] == "ATTEMPTED"
        for key in ("hypothesis_indices", "initialized_rotation", "initialized_translation_mm"):
            assert row[key] == group[key]
        assert all(key in row for key in ("rotation", "translation", "scan_coverage", "inlier_surface_p95_mm",
                                         "observability_ratio", "weighted_fit_observability_ratio"))
        assert all(reason in _v4()._GEOMETRIC_CANDIDATE_REFUSALS for reason in row["reasons"])
    return "ALL_SELECTED_PLANE_START_GROUPS_FULLY_EVALUATED"


def _missing_arc_assertions(result):
    disposition = _execution_complete(result["registration"], allow_prefilter_abstention=True)
    if disposition == "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE":
        assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
        assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
        assert result["phase"] == "WHOLE_PART_REGISTRATION" and result["measurements"] is None
        assert result["registration"]["transform_scan_to_cad"] is None
        assert result["reasons"] == ["PROVISIONAL_REGISTRATION_UNAVAILABLE", NO_START_REASON]
        return disposition
    return _adapted_missing_arc_assertions()(result)  # Original audited geometric reason/window guards.


def _common_assertions(result, cad_blob, scan_blob, *, missing_arc):
    _v4()._legacy_common_assertions(result, cad_blob, scan_blob)
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V7"
    assert result["registration_surface"] == "FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD"
    assert result["registration_initialization"] == "PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_V7"
    assert result["position_error_model"] == dict(uniform_euclidean_bound_mm=POSITION_ERROR_BOUND_MM,
        status="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        scope="PLANE_INITIALIZATION_ONLY_NOT_MEASUREMENT_UNCERTAINTY")
    binding, topology = result["triangle_sample_binding"], result["scan_triangle_binding"]
    assert binding["status"] == "CONSISTENT_IMMUTABLE_TRIANGLE_SAMPLE_SNAPSHOT"
    assert binding["initialization_sample_count"] == 32768
    assert topology["status"] == "CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT"
    assert topology["source_sha256"] == hashlib.sha256(scan_blob).hexdigest()
    assert topology["raw_points_sha256"] == result["input_preparation"]["source_bindings"]["raw_xyz_sha256"]
    assert topology["triangle_raw_rows_sha256"] == result["scan_topology"]["triangle_raw_rows_sha256"]
    evidence = result["registration"]["evidence"]
    assert evidence["schema"] == "NOISY_PLANE_INITIALIZED_FINITE_TRIANGLE_REGISTRATION_V7"
    assert evidence["initialization"]["hypothesis_evidence"]["schema_version"] == "plane_search_v6"
    assert evidence["registration_raw_rows"] == dict(selection="ALL_RAW_ROWS", count=topology["raw_vertex_count"], sha256=None)
    assert evidence["prefilter_full_scan_raw_rows"] == evidence["registration_input_rows"] == topology["raw_vertex_count"]
    initialized = evidence["initialization"]["input_digests"]
    assert initialized["scan_points_sha256"] == topology["raw_points_sha256"]
    assert initialized["scan_triangle_raw_rows_sha256"] == topology["triangle_raw_rows_sha256"]
    assert initialized["cad_triangles_sha256"] == binding["oriented_triangle_xyz_sha256"]
    epsilon_digest = hashlib.sha256(np.full(topology["raw_vertex_count"], POSITION_ERROR_BOUND_MM, dtype="<f8").tobytes()).hexdigest()
    assert initialized["position_error_bounds_mm_sha256"] == epsilon_digest
    assert binding["oriented_triangle_xyz_sha256"] == result["cad_surface_sampling"]["digests"]["oriented_triangle_xyz_sha256"]
    assert binding["initialization_points_sha256"] == result["cad_surface_sampling"]["digests"]["sample_points_sha256"]
    if result["registration"]["status"] == "CONDITIONAL_BEST_FIT":
        final = result["registration_array_binding"]
        assert final["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING"
        assert final["registration_selection"] == "ALL_RAW_ROWS"
        assert final["position_error_bounds_mm_sha256"] == epsilon_digest
        assert final["cad_sampler_points_sha256"] == binding["initialization_points_sha256"]
    return _execution_complete(result["registration"], allow_prefilter_abstention=missing_arc)


def _initialization_action_errors(result, raw, cad_frame):
    rows = []
    initialization = (result.get("registration") or {}).get("evidence", {}).get("initialization", {})
    for proposal in initialization.get("hypotheses", []):
        row = dict(hypothesis_index=proposal["hypothesis_index"], status=proposal["status"],
            prefilter_status=proposal["prefilter_status"], maximum_mm=None, rms_mm=None)
        if proposal.get("rotation") is not None and proposal.get("translation_mm") is not None:
            error = np.linalg.norm(raw@np.asarray(proposal["rotation"]).T+proposal["translation_mm"]-cad_frame, axis=1)
            row.update(maximum_mm=float(error.max()), rms_mm=float(np.sqrt(np.mean(error**2))))
        rows.append(row)
    return rows


def _source_digests():
    paths = (Path(__file__).resolve(), V4_SOURCE, _v4().SURFACE_SOURCE, _v4().DENSE_SOURCE,
        _v4().ORIGINAL_SOURCE, ROOT/"backend/cad_guided_mesh_offline_v7.py",
        ROOT/"docs/research/2026-09-09_mesh_connected_planes_v7_protocol.md")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _write_artifact(directory, index, record):
    assert type(index) is int and 0 <= index < 4
    encoded = _v4()._encode_case(record, ARTIFACT_LIMITS["per_case_json_bytes"])
    path = directory/f"case_{index:02d}.json"
    with path.open("xb") as stream:
        assert stream.write(encoded) == len(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.stat().st_size == len(encoded)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == hashlib.sha256(encoded).hexdigest()
    return dict(relative_path=path.name, bytes=len(encoded), sha256=actual, status="COMPLETE_CASE_RECORD_RETAINED")


def _receipt(records, before, after, directory):
    artifacts = all(row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED" for row in records)
    return dict(protocol="CAD_GUIDED_MESH_CONNECTED_PLANES_V7", cases=records, expected_case_count=4,
        matrix_passed=len(records) == 4 and before == after and artifacts and all(row["passed"] for row in records),
        criteria=CRITERIA, added_noisy_case_criteria=NOISY_CRITERIA, position_error_bound_mm=POSITION_ERROR_BOUND_MM,
        external_resource_limits=RESOURCE_LIMITS, artifact_limits=ARTIFACT_LIMITS, artifact_directory=str(directory),
        fixture_source_sha256_before=before, fixture_source_sha256_after=after,
        predecessor_receipt="mesh_connected_planes_v4-epppmtsj",
        policy_scope="FIXED_INVENTED_ERROR_BOUND_NOT_SCANNER_CALIBRATION_OR_PHYSICAL_POLICY",
        search_scope="SELECTED_POLICY_COMPLETE_NOT_EXHAUSTIVE_OR_GLOBALLY_UNIQUE",
        r0_scope="NEW_COMMON_WRAPPER_EXTRACTOR_CONTROL_NOT_FROZEN_HISTORICAL_PIPELINE",
        evidence_class="INVENTED_NATIVE_CONNECTED_NOT_PHYSICAL_GENERALIZATION")


def run_invented_mesh_workflow_planes_v7_checks(*, artifact_directory):
    from backend.cad_guided_mesh_offline_v7 import inspect_cad_guided_mesh_v7
    before = _source_digests()
    try:
        directory = _v4()._new_artifact_directory(artifact_directory)
    except Exception as error:
        records = [dict(case=name, passed=False, phase="artifact_setup", expected_predictions=count,
            error_type=type(error).__name__, error=str(error)[:1000]) for name, _, _, count in CONNECTED_CASES]
        return _receipt(records, before, _source_digests(), artifact_directory)
    original = _original()
    cad_blob, fixture_error = None, None
    try:
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
    except Exception as error:
        fixture_error = error
    records = []
    for index, (name, branch, missing_arc, expected) in enumerate(CONNECTED_CASES):
        record = dict(case=name, passed=False, phase="fixture", expected_predictions=expected,
            qualification_scope="CASE_ASSERTIONS_ONLY_MATRIX_REQUIRES_FINAL_SOURCE_AND_ARTIFACT_MANIFEST",
            production_use_allowed=False, observed_absence_claim=False)
        try:
            if fixture_error is not None:
                raise fixture_error
            raw, faces, cad_frame = original._analytic_mesh_scan(missing_arc)
            if index == 3:
                raw, delta = _noisy_rows(raw)
                assert np.max(np.linalg.norm(delta, axis=1)) <= NOISE_MAX_EUCLIDEAN_MM
                record["noise"] = dict(model="FIXED_ORIGINAL_ROW_ID_3D_MODULAR_PATTERN",
                    rounding_reserve_factor=NOISE_ROUNDING_RESERVE_FACTOR,
                    displacement_basis="REPRESENTED_NOISY_RAW_MINUS_CLEAN_RAW",
                    maximum_euclidean_mm=float(np.max(np.linalg.norm(delta, axis=1))),
                    delta_sha256=hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes()).hexdigest())
            scan_blob = original._ply_bytes(raw, faces)
            record.update(scan_rows=len(raw), mesh_faces=len(faces), cad_sha256=hashlib.sha256(cad_blob).hexdigest(),
                scan_sha256=hashlib.sha256(scan_blob).hexdigest(), phase="actual_byte_to_v7_workflow")
            result = inspect_cad_guided_mesh_v7(scan_bytes=scan_blob, cad_bytes=cad_blob,
                position_error_bound_mm=POSITION_ERROR_BOUND_MM, allow_ocp=True, **_policies(branch))
            record["workflow"] = result
            record["phase"] = "post_inference_evaluation"
            record["post_inference_candidate_action_errors"] = (_v4()._surface()._candidate_action_errors(result, raw, cad_frame)
                if isinstance(result.get("registration"), dict) else [])
            record["post_inference_initialization_action_errors"] = _initialization_action_errors(result, raw, cad_frame)
            record["phase"] = "frozen_assertions"
            _common_assertions(result, cad_blob, scan_blob, missing_arc=missing_arc)
            assert result["input_preparation"]["source_bindings"]["raw_xyz_sha256"] == hashlib.sha256(np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
            assert result["scan_triangle_binding"]["triangle_raw_rows_sha256"] == hashlib.sha256(np.asarray(faces, dtype="<i8").tobytes()).hexdigest()
            assert result["input_preparation"]["coverage"]["raw_vertex_count"] == len(raw)
            assert len(result["predictions"]) == expected
            if missing_arc:
                record["sentinel_disposition"] = _missing_arc_assertions(result)
            elif index == 3:
                record["errors"] = _adapted_noisy_nominal_assertions()(result, raw, cad_frame)
            else:
                record["errors"] = original._nominal_assertions(result, raw, cad_frame)
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        compact = _v4()._compact_case(record)
        try:
            compact["artifact"] = _write_artifact(directory, index, record)
        except Exception as error:
            compact.update(passed=False, phase="artifact_publication", error_type=type(error).__name__, error=str(error)[:1000],
                artifact=dict(relative_path=f"case_{index:02d}.json", status="FAILED_OR_PARTIAL_NOT_QUALIFIED"))
        records.append(compact)
    after = _source_digests()
    for record in records:
        if before != after:
            record.update(passed=False, source_consistency_failure="FIXTURE_OR_PROTOCOL_SOURCE_CHANGED")
        if sum(row.get("artifact", {}).get("bytes", 0) for row in records) > ARTIFACT_LIMITS["total_case_json_bytes"]:
            record.update(passed=False, artifact_budget_failure="TOTAL_CASE_ARTIFACT_BUDGET_EXCEEDED")
    return _receipt(records, before, after, directory)


@pytest.mark.parametrize("branch", ("v1", "r0"))
def test_source_policies_and_oracles_preserve_frozen_predecessor(monkeypatch, branch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    old, original = _v4(), _original()
    saved = copy.deepcopy((old.CONNECTED_CASES, old.CRITERIA, original.CRITERIA))
    assert CONNECTED_CASES[:3] == old.CONNECTED_CASES and CRITERIA == old.CRITERIA
    assert {key: asdict(value) for key, value in _policies(branch).items()} == {
        key: asdict(value) for key, value in old._policies(branch).items()}
    adapted = _adapted_noisy_nominal_assertions()
    assert adapted.__code__ is original._nominal_assertions.__code__
    assert {key for key in adapted.__globals__ if adapted.__globals__[key] is not original._nominal_assertions.__globals__[key]} == {"CRITERIA"}
    sentinel = _adapted_missing_arc_assertions()
    assert sentinel.__code__ is old._missing_arc_assertions.__code__
    assert {key for key in sentinel.__globals__ if sentinel.__globals__[key] is not old._missing_arc_assertions.__globals__[key]} == {"_execution_complete"}
    assert (old.CONNECTED_CASES, old.CRITERIA, original.CRITERIA) == saved


def test_source_error_bound_noise_and_resource_constants_are_fixed():
    assert POSITION_ERROR_BOUND_MM == .005 and NOISE_MAX_EUCLIDEAN_MM == .001
    assert NOISE_ROUNDING_RESERVE_FACTOR == .999
    raw = np.arange(300., dtype=float).reshape(100, 3)
    before = raw.copy()
    first, delta = _noisy_rows(raw)
    second, again = _noisy_rows(raw)
    assert np.array_equal(raw, before) and np.array_equal(first, second) and np.array_equal(delta, again)
    assert np.all(np.linalg.norm(delta, axis=1) <= .001) and np.all(np.std(delta, axis=0) > 0.)
    assert not np.array_equal(first, raw)
    assert np.array_equal(delta, first-raw)
    assert NOISY_CRITERIA == dict(radius_error_mm=.01, absolute_sweep_error_deg=.1,
        generator_endpoint_error_mm=.05, pose_action_error_mm=.20)
    assert RESOURCE_LIMITS == dict(wall_seconds=180, cpu_seconds=120, process_tree_rss_bytes=4*1024**3, stdout_bytes=1024**2)
    assert ARTIFACT_LIMITS == dict(per_case_json_bytes=16*1024**2, total_case_json_bytes=64*1024**2)
    assert NATIVE_OPT_IN != _v4().NATIVE_OPT_IN and ARTIFACT_DIRECTORY_ENV != _v4().ARTIFACT_DIRECTORY_ENV


def test_full_analytic_nominal_noise_uses_represented_delta_with_fixed_reserve(monkeypatch):
    # Pure analytic fixture only: no STEP writer, OCP, native shape or workflow.
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "OCP" and not name.startswith("OCP.")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    raw, _, _ = _original()._analytic_mesh_scan(False)
    assert raw.shape == (2752, 3)
    clean = raw.copy()
    noisy, actual = _noisy_rows(raw)
    assert np.array_equal(raw, clean) and np.array_equal(actual, noisy-clean)
    assert np.max(np.linalg.norm(actual, axis=1)) <= NOISE_MAX_EUCLIDEAN_MM
    i = 1550
    endpoint_direction = np.array([((17*i+3) % 19-9)/9.,
        ((7*i+5) % 23-11)/11., ((11*i+2) % 29-14)/14.])
    np.testing.assert_array_equal(endpoint_direction, [-1., 1., -1.])
    assert np.dot(endpoint_direction, endpoint_direction) == 3.  # Full unreserved modular amplitude.
    intended = NOISE_ROUNDING_RESERVE_FACTOR*NOISE_MAX_EUCLIDEAN_MM/math.sqrt(3.)*endpoint_direction
    assert not np.array_equal(actual[i], intended)
    assert hashlib.sha256(np.asarray(actual, dtype="<f8").tobytes()).hexdigest() == hashlib.sha256(
        np.asarray(noisy-clean, dtype="<f8").tobytes()).hexdigest()


def test_source_actual_worker_has_no_runtime_pose_or_geometry_patch():
    tree = ast.parse(Path(__file__).read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "run_invented_mesh_workflow_planes_v7_checks")
    forbidden = {"setattr", "monkeypatch", "_refine", "_starts", "_oracle_pose"}
    assert not any(isinstance(node, ast.Name) and node.id in forbidden or isinstance(node, ast.Attribute)
        and node.attr in forbidden for node in ast.walk(worker))
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    call = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "inspect_cad_guided_mesh_v7")
    assert [key.arg for key in call.keywords] == ["scan_bytes", "cad_bytes", "position_error_bound_mm", "allow_ocp", None]
    for node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("_nominal_assertions", "_candidate_action_errors"):
            assert node.lineno > call.lineno


def _mock_completed_sentinel(branch="registration"):
    """Injected ledger/report control only, not native or geometry execution."""
    result = _v4()._mock_missing_arc_workflow(branch)
    evidence = result["registration"]["evidence"]
    initialization = evidence["initialization"]
    first = initialization["hypotheses"][0]
    first.update(hypothesis_index=0, unsigned_assignment_id=0, status="PROPOSED", prefilter_scheduled=True)
    if first["prefilter_status"] == "PASSED":
        first["exact_start_group_index"] = 0
    first["prefilter_reasons"] = [] if first["prefilter_status"] == "PASSED" else ["ASSIGNED_FINITE_FACE_PREFILTER_REJECTED"]
    for support in first["assigned_support"]:
        support["reasons"] = [] if support["status"] == "PASSED" else ["INSUFFICIENT_ASSIGNED_FINITE_FACE_SUPPORT"]
    initialization["hypotheses"] += [dict(hypothesis_index=i, unsigned_assignment_id=0, status="REJECTED",
        prefilter_scheduled=False, prefilter_status="SEARCH_NOMINAL_WITNESS_REFUSAL", assigned_support=[]) for i in range(1, 8)]
    initialization.update(selected_policy_evaluation_complete=True, search_exhaustive=False,
        global_pose_uniqueness="NOT_ESTABLISHED", selected_unsigned_assignment_ids=[0],
        hypothesis_evidence=dict(status="COMPLETE", selected_policy_evaluation_complete=True,
            search_exhaustive=False, accepted_registration=False, statistics=dict(signed_assignments_attempted=8,
                signed_assignments_evaluated=8, signed_assignments_planned=8, scan_triplets_scheduled=1, scan_triplets_evaluated=1)))
    all_groups = []
    for group in initialization["start_groups"]:
        group.update(exact_group_index=0, refinement_disposition="SCHEDULED_FOR_V2_REFINEMENT")
        all_groups.append(copy.deepcopy({key: value for key, value in group.items() if key != "index"}))
    initialization["all_passing_start_groups"] = all_groups
    initialization["statistics"].update(hypothesis_count=8, proposed_hypothesis_count=1, scheduled_hypothesis_count=1,
        omitted_proposed_hypothesis_count=0, search_rejected_hypothesis_count=7,
        exact_passing_start_group_count=len(all_groups), omitted_passing_start_group_count=0)
    if branch == "prefilter":
        initialization["reasons"] = [NO_START_REASON]
        result["registration"]["reasons"] = [NO_START_REASON]
        result["reasons"] = ["PROVISIONAL_REGISTRATION_UNAVAILABLE", NO_START_REASON]
    return result


@pytest.mark.parametrize("branch", ("prefilter", "registration", "local_row", "local_finite"))
def test_adapted_original_sentinel_accepts_only_completed_declared_branches(branch):
    original = _v4()._missing_arc_assertions
    original_globals = dict(original.__globals__)
    result = _mock_completed_sentinel(branch)
    assert _missing_arc_assertions(result).endswith("NOT_OBSERVED_ABSENCE")
    assert all(original.__globals__[key] is value for key, value in original_globals.items())


@pytest.mark.parametrize("fault", ("search_incomplete", "signed_incomplete", "partial_support", "candidate_numeric",
    "candidate_incomplete", "candidate_unattempted", "binding_downgrade", "local_budget", "empty_native", "unknown_reason",
    "support_resource_reason"))
def test_scheduled_completion_never_turns_failures_into_missing_arc_success(fault):
    result = _mock_completed_sentinel("local_row")
    evidence = result["registration"]["evidence"]
    if fault == "search_incomplete":
        evidence["initialization"]["hypothesis_evidence"]["selected_policy_evaluation_complete"] = False
    elif fault == "signed_incomplete":
        evidence["initialization"]["hypothesis_evidence"]["statistics"]["signed_assignments_evaluated"] -= 1
    elif fault == "partial_support":
        evidence["initialization"]["hypotheses"][0]["assigned_support"][0]["query_rows"] -= 1
    elif fault == "candidate_numeric":
        evidence["candidate_numerical_failure_count"] = 1
    elif fault == "candidate_incomplete":
        evidence["candidate_evaluation_complete"] = False
    elif fault == "candidate_unattempted":
        evidence["candidates"][0]["execution_status"] = "NOT_ATTEMPTED"
    elif fault == "binding_downgrade":
        result["registration"].update(status="UNAVAILABLE", transform_scan_to_cad=None,
            reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
        result.update(phase="WHOLE_PART_REGISTRATION", measurements=None, reasons=["WORKFLOW_EXCEPTION:ValueError"])
    elif fault == "local_budget":
        result["measurements"]["features"][0]["reasons"] = ["LOCAL_OR_TOTAL_SUPPORT_BUDGET_NO_THINNING"]
    elif fault == "empty_native":
        result["native_features"]["features"] = []
    elif fault == "support_resource_reason":
        evidence["initialization"]["hypotheses"][0]["assigned_support"][0]["reasons"] = ["QUERY_BUDGET_EXCEEDED"]
    else:
        result["measurements"]["features"][0]["reasons"] = ["UNKNOWN_FAILURE"]
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_zero_start_sentinel_requires_nonempty_geometric_finite_work():
    result = _mock_completed_sentinel("prefilter")
    initialization = result["registration"]["evidence"]["initialization"]
    initialization.update(hypotheses=[], selected_unsigned_assignment_ids=[])
    for key in ("hypothesis_count", "hypotheses_evaluated", "proposed_hypothesis_count", "scheduled_hypothesis_count",
                "search_rejected_hypothesis_count", "rejected_hypothesis_count"):
        initialization["statistics"][key] = 0
    search = initialization["hypothesis_evidence"]["statistics"]
    search.update(signed_assignments_attempted=0, signed_assignments_evaluated=0, signed_assignments_planned=0)
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_group_origins_cover_every_passed_hypothesis_even_when_counts_are_coherent():
    result = _mock_completed_sentinel("registration")
    initialization = result["registration"]["evidence"]["initialization"]
    initialization["hypotheses"][1] = dict(copy.deepcopy(initialization["hypotheses"][0]), hypothesis_index=1)
    initialization["statistics"].update(proposed_hypothesis_count=2, scheduled_hypothesis_count=2,
        hypotheses_evaluated=2, passing_hypothesis_count=2, search_rejected_hypothesis_count=6)
    # The second passed sign coherently belongs to exact group0, but its origin
    # is lost from both the all-group and selected-group ledger. Cardinalities,
    # scheduled/refused totals and candidate/group equality alone still agree.
    assert initialization["hypotheses"][1]["exact_start_group_index"] == 0
    assert initialization["all_passing_start_groups"][0]["hypothesis_indices"] == [0]
    with pytest.raises(AssertionError):
        _execution_complete(result["registration"])


@pytest.fixture
def plumbing(monkeypatch):
    """Existing independently invented V2 stubs; NO native/pose-recovery claim."""
    old = _load(MOCK_SOURCE)
    state = old.fixture.__wrapped__(monkeypatch)
    package = old.workflow.__package__
    workflow = importlib.import_module(package+".cad_guided_mesh_offline_v7")
    topology_module = importlib.import_module(package+".cad_guided_mesh_topology_v4")
    initialize = importlib.import_module(package+".cad_guided_plane_initialization_v7")
    raw = state["scan"]["raw_points"]
    triangles = old.immutable(np.arange(72).reshape(24, 3), "<i8")
    state["scan"]["input_admission"] = dict(face_count=24, face_index_count=72)
    state["topology"] = dict(status="AVAILABLE", reasons=[], points=raw, triangle_raw_rows=triangles,
        vertex_count=72, triangle_count=24, source_sha256=hashlib.sha256(old.SCAN_BYTES).hexdigest(),
        points_sha256=old.digest(raw), triangle_raw_rows_sha256=old.digest(triangles),
        raw_row_order="ORIGINAL_PLY_VERTEX_ROWS", triangle_row_order="ORIGINAL_PLY_FACE_ROWS")
    monkeypatch.setattr(topology_module, "parse_triangle_mesh_ply_v4", lambda blob: state["topology"])
    def register(points, rows, cad, faces, local, **kwargs):
        state["events"].append("register_v7")
        state["register_v7_args"] = points, rows, cad, faces, local, kwargs
        digests = dict(scan_points_sha256=old.digest(points), scan_triangle_raw_rows_sha256=old.digest(rows),
            cad_triangles_sha256=old.digest(cad), cad_face_ids_sha256=old.digest(faces), cad_source_triangle_ids_sha256=old.digest(local),
            position_error_bounds_mm_sha256=old.digest(np.full(len(points), kwargs["position_error_bounds_mm"], dtype="<f8")))
        if state.get("binding_fault"):
            digests[state["binding_fault"]] = "0"*64
        evidence = state["registration"]["evidence"]
        evidence.update(initialization=dict(status="COMPLETE", search_complete=True, selected_policy_evaluation_complete=True,
            search_exhaustive=False, input_digests=dict(digests)), input_digests=dict(digests),
            candidate_evaluation_complete=not state.get("incomplete"), prefilter_full_scan_raw_rows=72, registration_input_rows=36,
            registration_input_points_sha256=old.digest(state["scan"]["points"]), registration_raw_rows=dict(
                selection="EXPLICIT_ORIGINAL_RAW_IDS_NO_HIDDEN_SAMPLING", count=36, sha256=old.digest(state["scan"]["engine_to_raw_row_indices"])),
            cad_initialization_points_sha256=old.digest(np.ascontiguousarray(np.unique(kwargs["cad_initialization_points"], axis=0), dtype="<f8")))
        return state["registration"]
    monkeypatch.setattr(initialize, "register_whole_part_planes_v7", register)
    def run(**overrides):
        policies = old.policies()
        policies["workflow_policy"] = workflow.MeshWorkflowPolicyV7(**asdict(policies["workflow_policy"]))
        args = dict(scan_bytes=old.SCAN_BYTES, cad_bytes=old.CAD_BYTES, position_error_bound_mm=POSITION_ERROR_BOUND_MM,
                    allow_ocp=True, **policies)
        args.update(overrides)
        return workflow.inspect_cad_guided_mesh_v7(**args)
    return old, workflow, state, run


def test_mock_bytes_epsilon_full_raw_measurement_and_inverse_publication(plumbing):
    old, _, state, run = plumbing
    result = run()
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE"
    assert state["read_bytes"] is old.CAD_BYTES
    assert state["extract_args"][0] is state["sample_args"][0] is state["shape"]
    points, rows, cad, faces, local, kwargs = state["register_v7_args"]
    assert points is state["scan"]["raw_points"] and len(points) == 72
    assert rows is state["topology"]["triangle_raw_rows"] and cad is state["cad"]["triangles"]
    assert faces is state["cad"]["source_face_indices"] and local is state["cad"]["source_triangle_indices"]
    assert kwargs["position_error_bounds_mm"] == .005
    assert kwargs["registration_raw_rows"] is state["scan"]["engine_to_raw_row_indices"]
    measured, _, args = state["measure_args"]
    np.testing.assert_array_equal(measured, points@old.ROTATION.T+old.TRANSLATION)
    assert len(measured) == 72 and args["scan_sha256"] == hashlib.sha256(old.SCAN_BYTES).hexdigest()
    assert "position_error_bounds_mm" not in args
    for i, key in enumerate(old.ENDPOINT_FIELDS):
        np.testing.assert_array_equal(result["predictions"][0][key], [i+.25, 2., 3.])
    assert result["measurements"]["features"][0]["raw_support"]["arc_rows"] == [1, 5, 69]
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ("epsilon_digest", "scan_byte_digest", "incomplete", "refusal", "late_publication"))
def test_mock_bad_binding_refusal_or_late_output_cannot_publish(plumbing, fault):
    _, _, state, run = plumbing
    if fault == "epsilon_digest":
        state["binding_fault"] = "position_error_bounds_mm_sha256"
    elif fault == "scan_byte_digest":
        state["topology"]["source_sha256"] = "0"*64
    elif fault == "incomplete":
        state["incomplete"] = True
    elif fault == "refusal":
        state["registration"].update(status="UNAVAILABLE", transform_scan_to_cad=None, reasons=["MOCK_GEOMETRIC_REFUSAL"])
    else:
        state["measured"]["finite_support_instances"].append(dict(state["measured"]["finite_support_instances"][0], line_end=[float("nan"), 0., 0.]))
    result = run()
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == [] and result["measurements"] is None
    if fault != "late_publication":
        assert "measure" not in state["events"]
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False


@pytest.mark.parametrize("epsilon", (None, False, -.1, float("nan"), float("inf"), "0.005"))
def test_mock_invalid_position_declaration_stops_before_loader(plumbing, epsilon):
    _, _, state, run = plumbing
    result = run(position_error_bound_mm=epsilon)
    assert result["reasons"] == ["EXPLICIT_FINITE_POSITION_ERROR_BOUND_REQUIRED"]
    assert state["events"] == [] and result["predictions"] == []


def test_fourth_artifact_is_exclusive_and_prior_artifacts_stay_frozen(tmp_path):
    directory = _v4()._new_artifact_directory(tmp_path/"ARTIFACTS")
    record = dict(case=CONNECTED_CASES[3][0], passed=False, reasons=["SOURCE_HELPER_ONLY"])
    artifact = _write_artifact(directory, 3, record)
    assert artifact["relative_path"] == "case_03.json"
    assert json.loads((directory/artifact["relative_path"]).read_bytes()) == record
    with pytest.raises(FileExistsError):
        _write_artifact(directory, 3, {"passed": True})


def test_fixed_invented_native_planes_v7_workflow_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must opt in under frozen serial180/120-second4-GiB protocol")
    directory = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    assert directory, "Root must supply its fresh retained-job ARTIFACTS child"
    receipt = run_invented_mesh_workflow_planes_v7_checks(artifact_directory=directory)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["stdout_bytes"]
    print(encoded, flush=True)
    assert len(receipt["cases"]) == 4 and receipt["matrix_passed"] is True
    assert receipt["fixture_source_sha256_before"] == receipt["fixture_source_sha256_after"]
    assert all(row["passed"] and row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED"
               for row in receipt["cases"]), receipt["cases"]
