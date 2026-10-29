"""Prospective fixed three-case native plane-initialized successor.

No native work at import/ordinary collection; root must opt in under the frozen
180s wall /120s CPU /2GiB process-tree RSS /1MiB stdout watchdog. The original
invented STEP, analytic scan, pose oracle and nominal assertions are reused
unchanged. All dense/V2 numeric policies and32768 CAD samples stay fixed.

V4 changes initialization, not radius/sweep/endpoint/pose ceilings. Both nominal
cases MUST measure one instance. The missing-arc negative may be a completed
geometric prefilter abstention, never a fatal/budget/numerically incomplete pass.
No observed-absence, production, historical-R0 or physical-generalization claim.

Root supplies a fresh retained-job ARTIFACTS path programmatically. Each complete
case record is exclusively created as a <=16MiB JSON file, <=48MiB total; only the
compact manifest reaches stdout. Failed/partial files are not overwritten or
published as successful records. No arbitrary CLI path or physical raw fixture
is accepted; the original helper still creates its owned temporary invented STEP.
"""

import ast
import builtins
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SURFACE_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_surface_v2.py")
DENSE_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_native_dense_v1.py")
ORIGINAL_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_native_v1.py")
CONNECTED_CASES = (
    ("v1_nominal_oblique", "v1", False, 1),
    ("v1_missing_arc_oblique", "v1", True, 0),
    ("r0_common_wrapper_nominal_oblique", "r0", False, 1),
)
CRITERIA = {"radius_error_mm": 2e-5, "absolute_sweep_error_deg": 2e-5,
            "generator_endpoint_error_mm": .02, "pose_action_error_mm": .20}
RESOURCE_LIMITS = {"wall_seconds": 180, "cpu_seconds": 120,
                   "process_tree_rss_bytes": 2 * 1024**3, "stdout_bytes": 1024**2}
ARTIFACT_LIMITS = {"per_case_json_bytes": 16 * 1024**2, "total_case_json_bytes": 48 * 1024**2}
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_CONNECTED_PLANES_V4"
ARTIFACT_DIRECTORY_ENV = "SHERMAN_MESH_CONNECTED_PLANES_V4_ARTIFACT_DIRECTORY"
PREDECESSOR_RECEIPTS = ("mesh_connected_native_v1-jkz08n45", "mesh_connected_native_dense_v1-j5p3bnlr",
                        "mesh_connected_surface_v2-lqfsnygb")


@lru_cache(maxsize=1)
def _surface():
    spec = importlib.util.spec_from_file_location("_planes_v4_surface_predecessor_helpers", SURFACE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CONNECTED_CASES == CONNECTED_CASES and module.CRITERIA == CRITERIA
    return module


def _policies(branch):
    from backend.cad_guided_mesh_offline_v4 import MeshWorkflowPolicyV4

    policies = _surface()._policies(branch)
    return {**policies, "workflow_policy": MeshWorkflowPolicyV4(**asdict(policies["workflow_policy"]))}


def _fixture_source_digests():
    paths = (Path(__file__).resolve(), SURFACE_SOURCE, DENSE_SOURCE, ORIGINAL_SOURCE,
             ROOT / "backend/cad_guided_mesh_offline_v4.py")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _legacy_common_assertions(result, cad_blob, scan_blob):
    # Exact dense predecessor checks except its PCA-specific candidate_count24.
    # A source-AST control below proves this precise, prospectively declared delta.
    json.dumps(result, allow_nan=False)
    assert result["observed_bend_count"] is None
    assert result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE"
    assert result["uncertainty_status"] == "UNCALIBRATED"
    assert result["cad_sha256"] == hashlib.sha256(cad_blob).hexdigest()
    assert result["input_preparation"]["source_bindings"]["scan_sha256"] == hashlib.sha256(scan_blob).hexdigest()
    units = result["cad_units"]
    assert units["target_unit"] == "mm" and units["transferred_roots"] == 1
    assert units["target_unit_action"] == "SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER"
    assert units["source_unit_names"]["length"]
    assert all(name.upper() in ("MILLIMETRE", "MILLIMETER", "MM")
               for name in units["source_unit_names"]["length"])
    assert units["physical_calibration"] == "NOT_ESTABLISHED"
    assert result["cad_surface_sampling"]["status"] == "AVAILABLE"
    assert result["cad_surface_sampling"]["native_surface_point_status"] == "NOT_CERTIFIED"
    assert result["cad_surface_sampling"]["coverage"]["sample_count"] == 32768
    assert result["reference_frame_compatibility"] == "NOT_ASSERTED_BY_INFERENCE"


def _execution_complete(registration, *, allow_prefilter_abstention=False):
    evidence = registration["evidence"]
    initialization = evidence["initialization"]
    assert initialization["status"] == "COMPLETE" and initialization["search_complete"] is True
    hypotheses, groups = initialization["hypotheses"], initialization["start_groups"]
    stats = initialization["statistics"]
    assert stats["hypothesis_count"] == stats["hypotheses_evaluated"] == len(hypotheses)
    assert stats["passing_hypothesis_count"] + stats["rejected_hypothesis_count"] == len(hypotheses)
    assert all(row["prefilter_status"] in ("PASSED", "REJECTED") for row in hypotheses)
    assert all(len(row["assigned_support"]) == 3 and all(
        support["status"] in ("PASSED", "REJECTED")
        and support["query_rows"] == support["support_count"]
        and support["attempted_query_rows"] == support["support_count"]
        for support in row["assigned_support"]) for row in hypotheses)
    rows = evidence["candidates"]
    assert evidence["candidate_numerical_failure_count"] == 0
    if not groups:
        assert allow_prefilter_abstention
        assert initialization["reasons"] == ["NO_FINITE_PREFILTERED_PLANE_STARTS"]
        assert all(row["prefilter_status"] == "REJECTED" for row in hypotheses)
        assert stats["passing_hypothesis_count"] == stats["surviving_start_group_count"] == 0
        assert registration["status"] == "UNAVAILABLE"
        assert registration["reasons"] == ["NO_FINITE_PREFILTERED_PLANE_STARTS"]
        assert rows == [] and evidence["candidate_count"] == evidence["attempted_candidate_count"] == 0
        return "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE"
    assert 1 <= len(groups) <= 24
    assert initialization["reasons"] == []
    assert stats["surviving_start_group_count"] == len(groups)
    assert evidence["candidate_search_complete"] is True and evidence["candidate_evaluation_complete"] is True
    assert evidence["candidate_count"] == evidence["attempted_candidate_count"] == len(groups) == len(rows)
    assert [row["index"] for row in rows] == list(range(len(groups)))
    for row, group in zip(rows, groups):
        assert row["execution_status"] == "ATTEMPTED"
        assert row["hypothesis_indices"] == group["hypothesis_indices"]
        assert row["initialized_rotation"] == group["initialized_rotation"]
        assert row["initialized_translation_mm"] == group["initialized_translation_mm"]
        assert all(key in row for key in ("rotation", "translation", "scan_coverage",
            "inlier_surface_p95_mm", "observability_ratio", "weighted_fit_observability_ratio"))
        assert not any(reason.startswith(("CANDIDATE_NUMERICAL_FAILURE", "NOT_ATTEMPTED",
            "REGISTRATION_QUERY_BUDGET_EXCEEDED", "FINITE_SURFACE_QUERY_UNAVAILABLE")) for reason in row["reasons"])
    return "ALL_PLANE_START_GROUPS_FULLY_EVALUATED"


def _common_assertions(result, cad_blob, scan_blob, *, missing_arc):
    _legacy_common_assertions(result, cad_blob, scan_blob)
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V4"
    assert result["registration_initialization"] == "PLANE_ONLY_FINITE_ASSIGNED_FACE_PREFILTER_V4"
    assert result["registration_surface"] == "FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD"
    binding = result["triangle_sample_binding"]
    assert binding["status"] == "CONSISTENT_IMMUTABLE_TRIANGLE_SAMPLE_SNAPSHOT"
    assert binding["initialization_sample_count"] == 32768
    assert binding["oriented_triangle_xyz_sha256"] == result["cad_surface_sampling"]["digests"]["oriented_triangle_xyz_sha256"]
    assert binding["initialization_points_sha256"] == result["cad_surface_sampling"]["digests"]["sample_points_sha256"]
    topology = result["scan_triangle_binding"]
    assert topology["status"] == "CONSISTENT_IMMUTABLE_SCAN_TRIANGLE_SNAPSHOT"
    assert topology["source_sha256"] == hashlib.sha256(scan_blob).hexdigest()
    assert topology["raw_points_sha256"] == result["input_preparation"]["source_bindings"]["raw_xyz_sha256"]
    assert topology["triangle_raw_rows_sha256"] == result["scan_topology"]["triangle_raw_rows_sha256"]
    assert result["registration"]["evidence"]["schema"] == "PLANE_INITIALIZED_FINITE_TRIANGLE_WHOLE_PART_REGISTRATION_V4"
    evidence = result["registration"]["evidence"]
    assert evidence["registration_raw_rows"]["selection"] == "ALL_RAW_ROWS"
    assert evidence["registration_raw_rows"]["sha256"] is None
    assert evidence["prefilter_full_scan_raw_rows"] == evidence["registration_input_rows"] == topology["raw_vertex_count"]
    initialized = evidence["initialization"]["input_digests"]
    assert initialized["scan_points_sha256"] == topology["raw_points_sha256"]
    assert initialized["scan_triangle_raw_rows_sha256"] == topology["triangle_raw_rows_sha256"]
    assert initialized["cad_triangles_sha256"] == binding["oriented_triangle_xyz_sha256"]
    if result["registration"]["status"] == "CONDITIONAL_BEST_FIT":
        final_binding = result["registration_array_binding"]
        assert final_binding["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING"
        assert final_binding["registration_selection"] == "ALL_RAW_ROWS"
        assert final_binding["cad_sampler_points_sha256"] == binding["initialization_points_sha256"]
    return _execution_complete(result["registration"], allow_prefilter_abstention=missing_arc)


_GEOMETRIC_REGISTRATION_REFUSALS = frozenset((
    "NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE", "AMBIGUOUS_GENERATED_WHOLE_PART_POSES"))
_GEOMETRIC_CANDIDATE_REFUSALS = frozenset(("INSUFFICIENT_FINITE_SURFACE_SUPPORT",
    "SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE", "LOCAL_POSE_UNOBSERVABLE"))
_GEOMETRIC_MEASUREMENT_REFUSALS = frozenset((
    "INSUFFICIENT_FLANGE_SUPPORT", "FLANGE_SUPPORT_RANK_DEFICIENT",
    "FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY", "FLANGE_ORIENTATION_OUTSIDE_SEARCH_SCOPE",
    "FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL", "INSUFFICIENT_ARC_SUPPORT", "ARC_SUPPORT_RANK_DEFICIENT",
    "ARC_COMPATIBLE_WITH_FLAT", "ARC_ILL_CONDITIONED", "ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL",
    "TANGENT_RADIUS_WEAKLY_IDENTIFIED", "ARC_NOT_TANGENT_CIRCLE_AT_DECLARED_RESIDUAL",
    "AMBIGUOUS_FLANGE_WINDOWS", "MEASURED_AXIS_OUTSIDE_SEARCH_SCOPE", "MEASURED_SWEEP_OUTSIDE_MODEL_SCOPE",
    "CIRCLE_NOT_SAME_INNER_FACE_TANGENT", "ARC_OUTSIDE_MEASURED_CONTACT_SECTOR",
    "INSUFFICIENT_ARC_ANGULAR_COVERAGE", "INSUFFICIENT_DISTINCT_AXIAL_STATIONS",
    "INSUFFICIENT_AXIAL_SAMPLE_SPAN", "AMBIGUOUS_SHARED_RAW_FEATURE_SUPPORT"))
_GEOMETRIC_FINITE_SUPPORT_REFUSALS = frozenset((
    "INSUFFICIENT_UNIQUE_SUPPORT", "INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT", "UNSUPPORTED_ARC_WINDING",
    "NO_FINITE_AXIAL_SUPPORT", "DISCONNECTED_AXIAL_SUPPORT", "INSUFFICIENT_LOCAL_ARC_OR_FLANGE_SUPPORT",
    "INSUFFICIENT_LOCAL_ANGULAR_STATIONS", "INSUFFICIENT_LOCAL_ANGULAR_COVERAGE",
    "DISCONNECTED_LOCAL_ANGULAR_SUPPORT", "AMBIGUOUS_SHARED_RAW_FEATURE_SUPPORT"))


def _geometric_reasons(reasons, allowed):
    # Exact reason allowlists, not permissive prefix filtering. Unknown/mixed,
    # input, numerical, solver, resource and carrier-undefined reasons fail closed.
    assert type(reasons) is list and reasons and all(type(reason) is str and reason in allowed for reason in reasons)


def _missing_arc_assertions(result):
    """The actual native sentinel: complete execution is necessary, not sufficient."""
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    registration = result["registration"]
    disposition = _execution_complete(registration, allow_prefilter_abstention=True)
    evidence = registration["evidence"]
    for row in evidence["candidates"]:
        assert type(row["reasons"]) is list
        assert all(reason in _GEOMETRIC_CANDIDATE_REFUSALS for reason in row["reasons"])
    if registration["status"] == "UNAVAILABLE":
        assert result["phase"] == "WHOLE_PART_REGISTRATION" and result["measurements"] is None
        assert registration["transform_scan_to_cad"] is None
        reasons = registration["reasons"]
        assert len(reasons) == 1
        assert result["reasons"] == ["PROVISIONAL_REGISTRATION_UNAVAILABLE", *reasons]
        if disposition == "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE":
            assert reasons == ["NO_FINITE_PREFILTERED_PLANE_STARTS"]
            return disposition
        _geometric_reasons(reasons, _GEOMETRIC_REGISTRATION_REFUSALS)
        if reasons == ["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"]:
            assert all(row["reasons"] for row in evidence["candidates"])
        else:
            assert evidence["comparable_distinct_pose_count"] >= 1
        return "COMPLETED_GEOMETRIC_REGISTRATION_ABSTENTION_NOT_OBSERVED_ABSENCE"
    assert registration["status"] == "CONDITIONAL_BEST_FIT" and registration["reasons"] == []
    assert result["phase"] == "COMPLETE" and result["reasons"] == ["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"]
    assert result["registration_array_binding"]["status"] == "CONSISTENT_REGISTRATION_ARRAY_BINDING"
    transform = np.asarray(registration["transform_scan_to_cad"])
    assert transform.shape == (4, 4) and transform.dtype.kind in "fiu" and np.isfinite(transform).all()
    assert np.array_equal(transform[3], [0., 0., 0., 1.])
    assert np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), rtol=0, atol=1e-8)
    assert abs(float(np.linalg.det(transform[:3, :3]))-1.) <= 1e-8
    measured = result["measurements"]
    assert measured["computation_status"] == "COMPLETE" and measured["reasons"] == []
    assert measured["finite_support_instances"] == [] and measured["observed_support_instance_count"] == 0
    assert measured["observed_bend_count"] is None
    rows = measured["features"]
    native_ids = [row["feature_id"] for row in result["native_features"]["features"]]
    assert native_ids and len(set(native_ids)) == len(native_ids)
    assert [row["feature_id"] for row in rows] == native_ids
    assert measured["expected_window_count"] == len(rows) == len(native_ids)
    if "expected_roi_count" in result["native_features"]:
        assert result["native_features"]["expected_roi_count"] == len(native_ids)
    measured_count = sum(row["status"] == "MEASURED_CONDITIONAL" for row in rows)
    assert measured["measured_conditional_window_count"] == measured_count
    assert measured["status"] == ("PARTIAL" if measured_count else "UNAVAILABLE")
    for row in rows:
        finite = row["finite_support"]
        if row["status"] == "UNAVAILABLE":
            assert row["measurements"] is None
            _geometric_reasons(row["reasons"], _GEOMETRIC_MEASUREMENT_REFUSALS)
        else:
            assert row["status"] == "MEASURED_CONDITIONAL" and type(row["measurements"]) is dict
            assert row["reasons"] == [] and finite is not None
        if finite is not None:
            assert finite["status"] == "UNAVAILABLE" and finite["instance"] is None
            _geometric_reasons(finite["reasons"], _GEOMETRIC_FINITE_SUPPORT_REFUSALS)
    return "COMPLETED_GEOMETRIC_LOCAL_SUPPORT_ABSTENTION_NOT_OBSERVED_ABSENCE"


def _initialization_action_errors(result, raw, cad_frame):
    """Oracle evaluation AFTER inference only; every retained hypothesis stays."""
    rows = []
    initialization = (result.get("registration") or {}).get("evidence", {}).get("initialization", {})
    for proposal in initialization.get("hypotheses", []):
        error = np.linalg.norm(raw@np.asarray(proposal["rotation"]).T
                               + proposal["translation_mm"]-cad_frame, axis=1)
        rows.append(dict(hypothesis_index=proposal["hypothesis_index"],
            prefilter_status=proposal["prefilter_status"], maximum_mm=float(error.max()),
            rms_mm=float(np.sqrt(np.mean(error**2)))))
    return rows


def _encode_case(record, maximum=ARTIFACT_LIMITS["per_case_json_bytes"]):
    assert type(maximum) is int and 0 <= maximum <= ARTIFACT_LIMITS["per_case_json_bytes"]
    encoded = bytearray()
    for piece in json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":")).iterencode(record):
        chunk = piece.encode("utf-8")
        if len(encoded)+len(chunk) > maximum:
            raise ValueError("COMPLETE_CASE_ARTIFACT_BUDGET_EXCEEDED")
        encoded.extend(chunk)
    return bytes(encoded)


def _new_artifact_directory(path):
    directory = Path(path)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    assert directory.resolve() == directory and directory.parent.is_dir()
    directory.mkdir(mode=0o700)  # Fresh only; refuse preexisting paths, never overwrite.
    return directory


def _write_case_artifact(directory, index, record):
    assert type(index) is int and 0 <= index < len(CONNECTED_CASES)
    encoded = _encode_case(record)
    name = f"case_{index:02d}.json"
    path = directory/name
    with path.open("xb") as stream:
        written = stream.write(encoded)
        assert written == len(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.stat().st_size == len(encoded)
    # Read only the file just exclusively created in this new owned directory.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == hashlib.sha256(encoded).hexdigest()
    return dict(relative_path=name, bytes=len(encoded), sha256=digest,
                status="COMPLETE_CASE_RECORD_RETAINED")


def _compact_case(record):
    keys = ("case", "passed", "phase", "expected_predictions", "error_type", "error", "errors",
            "scan_rows", "mesh_faces", "cad_sha256", "scan_sha256", "sentinel_disposition")
    compact = {key: record[key] for key in keys if key in record}
    workflow = record.get("workflow", {})
    evidence = (workflow.get("registration") or {}).get("evidence", {})
    compact.update(predictions=len(workflow.get("predictions", [])), workflow_status=workflow.get("status"),
        candidate_count=evidence.get("candidate_count"),
        candidate_numerical_failure_count=evidence.get("candidate_numerical_failure_count"),
        candidate_evaluation_complete=evidence.get("candidate_evaluation_complete"),
        initialization_statistics=evidence.get("initialization", {}).get("statistics"))
    return compact


def _receipt(records, before, after, artifact_directory):
    complete_artifacts = all(row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED"
                             for row in records)
    matrix_passed = (len(records) == 3 and before == after and complete_artifacts
                     and all(row["passed"] for row in records))
    return dict(protocol="CAD_GUIDED_MESH_CONNECTED_PLANES_V4", cases=records, matrix_passed=matrix_passed,
        expected_case_count=len(CONNECTED_CASES), criteria=CRITERIA,
        external_resource_limits=RESOURCE_LIMITS, artifact_limits=ARTIFACT_LIMITS,
        artifact_directory=str(artifact_directory), predecessor_receipts=PREDECESSOR_RECEIPTS,
        fixture_source_sha256_before=before, fixture_source_sha256_after=after,
        policy_delta="PLANE_ONLY_INITIALIZATION; DENSE_V2_NUMERIC_AND_MEASUREMENT_POLICIES_UNCHANGED",
        evidence_class="INVENTED_NATIVE_CONNECTED_NOT_PHYSICAL_GENERALIZATION",
        missing_arc_scope="NEGATIVE_CONTROL_ABSTENTION_NOT_OBSERVED_ABSENCE",
        r0_scope="NEW_COMMON_WRAPPER_EXTRACTOR_CONTROL_NOT_FROZEN_HISTORICAL_PIPELINE")


def run_invented_mesh_workflow_planes_v4_checks(*, artifact_directory):
    """Root-supervised serial entry; no command-line path or data lookup."""
    from backend.cad_guided_mesh_offline_v4 import inspect_cad_guided_mesh_v4

    before = _fixture_source_digests()
    try:
        directory = _new_artifact_directory(artifact_directory)
    except Exception as error:
        records = [dict(case=name, passed=False, phase="artifact_setup", expected_predictions=count,
                        error_type=type(error).__name__, error=str(error)[:1000])
                   for name, _, _, count in CONNECTED_CASES]
        return _receipt(records, before, _fixture_source_digests(), artifact_directory)
    original = _surface()._dense()._original()
    cad_blob, fixture_error = None, None
    try:
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
    except Exception as error:
        fixture_error = error
    records = []
    for index, (name, branch, missing_arc, expected_predictions) in enumerate(CONNECTED_CASES):
        record = dict(case=name, passed=False, phase="fixture", expected_predictions=expected_predictions,
            qualification_scope="CASE_ASSERTIONS_ONLY_MATRIX_REQUIRES_FINAL_SOURCE_AND_ARTIFACT_MANIFEST",
            production_use_allowed=False, observed_absence_claim=False)
        try:
            if fixture_error is not None:
                raise fixture_error
            raw, faces, cad_frame = original._analytic_mesh_scan(missing_arc)
            scan_blob = original._ply_bytes(raw, faces)
            record.update(scan_rows=len(raw), mesh_faces=len(faces),
                          cad_sha256=hashlib.sha256(cad_blob).hexdigest(), scan_sha256=hashlib.sha256(scan_blob).hexdigest())
            record["phase"] = "actual_byte_to_v4_workflow"
            result = inspect_cad_guided_mesh_v4(scan_bytes=scan_blob, cad_bytes=cad_blob,
                                               allow_ocp=True, **_policies(branch))
            record["workflow"] = result
            record["phase"] = "post_inference_evaluation"
            record["post_inference_candidate_action_errors"] = (_surface()._candidate_action_errors(result, raw, cad_frame)
                if isinstance(result.get("registration"), dict) else [])
            record["post_inference_initialization_action_errors"] = _initialization_action_errors(result, raw, cad_frame)
            record["phase"] = "frozen_assertions"
            _common_assertions(result, cad_blob, scan_blob, missing_arc=missing_arc)
            assert result["input_preparation"]["source_bindings"]["raw_xyz_sha256"] == hashlib.sha256(
                np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
            assert result["scan_triangle_binding"]["triangle_raw_rows_sha256"] == hashlib.sha256(
                np.asarray(faces, dtype="<i8").tobytes()).hexdigest()
            assert result["input_preparation"]["coverage"]["raw_vertex_count"] == len(raw)
            assert len(result["predictions"]) == expected_predictions
            if missing_arc:
                record["sentinel_disposition"] = _missing_arc_assertions(result)
            else:
                record["errors"] = original._nominal_assertions(result, raw, cad_frame)
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        compact = _compact_case(record)
        try:
            compact["artifact"] = _write_case_artifact(directory, index, record)
        except Exception as error:
            compact.update(passed=False, phase="artifact_publication", error_type=type(error).__name__,
                error=str(error)[:1000], artifact=dict(relative_path=f"case_{index:02d}.json",
                                                     status="FAILED_OR_PARTIAL_NOT_QUALIFIED"))
        records.append(compact)
    after = _fixture_source_digests()
    if before != after:
        for record in records:
            record.update(passed=False, source_consistency_failure="FIXTURE_OR_PROTOCOL_SOURCE_CHANGED")
    if sum(record.get("artifact", {}).get("bytes", 0) for record in records) > ARTIFACT_LIMITS["total_case_json_bytes"]:
        for record in records:
            record.update(passed=False, artifact_budget_failure="TOTAL_CASE_ARTIFACT_BUDGET_EXCEEDED")
    return _receipt(records, before, after, directory)


@pytest.mark.parametrize("branch", ("v1", "r0"))
def test_policy_values_and_predecessor_globals_unchanged_without_native(monkeypatch, branch):
    calls, real_import = [], builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise AssertionError("SOURCE_CONTROL_MUST_NOT_IMPORT_NATIVE")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    surface = _surface()
    dense, original = surface._dense(), surface._dense()._original()
    globals_before = copy.deepcopy([(module.CONNECTED_CASES, module.CRITERIA, module.RESOURCE_LIMITS)
                                   for module in (surface, dense, original)])
    previous = {key: asdict(value) for key, value in surface._policies(branch).items()}
    policies = _policies(branch)
    assert type(policies["workflow_policy"]).__name__ == "MeshWorkflowPolicyV4"
    assert {key: asdict(value) for key, value in policies.items()} == previous
    assert previous["workflow_policy"]["cad_sample_count"] == 32768
    assert previous["workflow_policy"]["max_registration_scan_rows"] == 3000
    assert [(module.CONNECTED_CASES, module.CRITERIA, module.RESOURCE_LIMITS)
            for module in (surface, dense, original)] == globals_before
    assert calls == []


def test_common_contract_differs_only_by_removed_pca_candidate_count_assertion():
    def function(path, name):
        return next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    previous = function(DENSE_SOURCE, "_common_assertions")
    current = function(Path(__file__), "_legacy_common_assertions")
    expected = ast.parse('assert result["registration"]["evidence"]["candidate_count"] == 24').body[0]
    matching = [node for node in previous.body
                if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    assert len(matching) == 1
    previous.body.remove(matching[0])
    current.name = previous.name
    assert ast.dump(current, include_attributes=False) == ast.dump(previous, include_attributes=False)


def test_cases_thresholds_budgets_and_opt_in_are_prospectively_fixed():
    surface = _surface()
    assert CONNECTED_CASES == surface.CONNECTED_CASES == surface._dense().CONNECTED_CASES
    assert CRITERIA == surface.CRITERIA == dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
                                               generator_endpoint_error_mm=.02, pose_action_error_mm=.20)
    assert RESOURCE_LIMITS == surface.RESOURCE_LIMITS == dict(wall_seconds=180, cpu_seconds=120,
        process_tree_rss_bytes=2*1024**3, stdout_bytes=1024**2)
    assert ARTIFACT_LIMITS == dict(per_case_json_bytes=16*1024**2, total_case_json_bytes=48*1024**2)
    assert NATIVE_OPT_IN != surface.NATIVE_OPT_IN


def test_fixture_oracles_only_evaluate_after_actual_workflow_without_runtime_patch():
    tree = ast.parse(Path(__file__).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    worker = functions["run_invented_mesh_workflow_planes_v4_checks"]
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    helpers = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "original"}
    assert helpers == {"_owned_step_bytes", "_asymmetric_native_strip", "_analytic_mesh_scan",
                       "_ply_bytes", "_nominal_assertions"}
    assert "_nominal_assertions" not in functions and "_analytic_mesh_scan" not in functions
    forbidden = {"_refine", "_starts", "_oracle_pose", "_pose", "setattr", "monkeypatch"}
    assert not any((isinstance(node, ast.Name) and node.id in forbidden)
                   or (isinstance(node, ast.Attribute) and node.attr in forbidden) for node in ast.walk(worker))
    workflow = next(node for node in calls if isinstance(node.func, ast.Name)
                    and node.func.id == "inspect_cad_guided_mesh_v4")
    assert [key.arg for key in workflow.keywords] == ["scan_bytes", "cad_bytes", "allow_ocp", None]
    sentinel = [node for node in calls if isinstance(node.func, ast.Name)
                and node.func.id == "_missing_arc_assertions"]
    assert len(sentinel) == 1 and sentinel[0].lineno > workflow.lineno
    for node in calls:
        if ((isinstance(node.func, ast.Name) and node.func.id == "_initialization_action_errors")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "_candidate_action_errors")):
            assert node.lineno > workflow.lineno


def test_workflow_source_reuses_v2_and_binds_raw_input_to_same_native_snapshot():
    source = ROOT/"backend/cad_guided_mesh_offline_v4.py"
    tree = ast.parse(source.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "inspect_cad_guided_mesh_v4")
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    reused = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "_v2"}
    assert {"_read_step_owned_snapshot", "_scan_binding", "_surface_binding",
            "_rigid_transform", "_publish_measurements"} <= reused
    snapshot = [node for node in calls if isinstance(node.func, ast.Attribute)
                and node.func.attr == "_read_step_owned_snapshot"]
    assert len(snapshot) == 1 and ast.unparse(snapshot[0].args[0]) == "cad_bytes"
    for name in ("extract", "sample_native_cad_surface_v2"):
        node = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == name)
        assert ast.unparse(node.args[0]) == "shape"
    registration = next(node for node in calls if isinstance(node.func, ast.Attribute)
                        and node.func.attr == "register_whole_part_planes_v4")
    assert [ast.unparse(arg) for arg in registration.args] == [
        "scan['raw_points']", "topology['triangle_raw_rows']", "cad['triangles']",
        "cad['source_face_indices']", "cad['source_triangle_indices']"]
    assert {item.arg: ast.unparse(item.value) for item in registration.keywords}["cad_initialization_points"] == "cad['points']"
    aligned = next(node for node in ast.walk(worker) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "aligned" for target in node.targets))
    assert ast.unparse(aligned.value) == "scan['raw_points'] @ rotation.T + translation"
    assert not any(isinstance(node, ast.Attribute) and node.attr in ("_refine", "_starts") for node in ast.walk(worker))


@pytest.mark.parametrize("allow", (False, 0, "true"))
def test_new_workflow_native_opt_in_refuses_before_any_native_import(monkeypatch, allow):
    from backend.cad_guided_mesh_offline_v4 import inspect_cad_guided_mesh_v4

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            raise AssertionError("ADMISSION_CONTROL_MUST_NOT_IMPORT_NATIVE")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    result = inspect_cad_guided_mesh_v4(scan_bytes=b"invented-invalid", cad_bytes=b"invented-invalid",
        workflow_policy=None, registration_policy=None, measurement_policy=None, instance_policy=None, allow_ocp=allow)
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert result["reasons"] == ["EXPLICIT_OCP_OPT_IN_REQUIRED"]


def test_new_workflow_whole_report_cap_discards_pose_and_predictions():
    from backend.cad_guided_mesh_offline_v4 import _bounded_workflow

    record = dict(status="PARTIAL", predictions=[{"line_start": [1., 2., 3.]}],
                  registration={"transform_scan_to_cad": np.eye(4).tolist()})
    result = _bounded_workflow(record, 1)
    assert result["status"] == "UNAVAILABLE" and result["predictions"] == []
    assert result["registration"] is None and result["measurements"] is None
    assert result["detailed_report_discarded_whole"] is True
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False


def _completed_registration(*, empty=False):
    support = dict(status="PASSED", query_rows=25, attempted_query_rows=25, support_count=25)
    hypothesis = dict(prefilter_status="PASSED", assigned_support=[dict(support) for _ in range(3)])
    group = dict(index=0, initialized_rotation=np.eye(3).tolist(), initialized_translation_mm=[0., 0., 0.],
                 hypothesis_indices=[0])
    candidate = dict(group, execution_status="ATTEMPTED", reasons=[], rotation=np.eye(3).tolist(),
        translation=[0., 0., 0.], scan_coverage=1., inlier_surface_p95_mm=0., observability_ratio=.1,
        weighted_fit_observability_ratio=.1)
    initialization = dict(status="COMPLETE", search_complete=True, reasons=[],
        hypotheses=[hypothesis], start_groups=[group], statistics=dict(hypothesis_count=1, hypotheses_evaluated=1,
        passing_hypothesis_count=1, rejected_hypothesis_count=0, surviving_start_group_count=1))
    result = dict(status="CONDITIONAL_BEST_FIT", reasons=[], evidence=dict(initialization=initialization,
        candidates=[candidate], candidate_count=1, attempted_candidate_count=1,
        candidate_search_complete=True, candidate_evaluation_complete=True, candidate_numerical_failure_count=0))
    if empty:
        initialization.update(reasons=["NO_FINITE_PREFILTERED_PLANE_STARTS"], start_groups=[])
        hypothesis["prefilter_status"] = "REJECTED"
        for row in hypothesis["assigned_support"]:
            row["status"] = "REJECTED"
        initialization["statistics"].update(passing_hypothesis_count=0, rejected_hypothesis_count=1,
                                             surviving_start_group_count=0)
        result.update(status="UNAVAILABLE", reasons=["NO_FINITE_PREFILTERED_PLANE_STARTS"])
        result["evidence"].update(candidates=[], candidate_count=0, attempted_candidate_count=0,
            candidate_search_complete=False, candidate_evaluation_complete=False)
    return result


@pytest.mark.parametrize("failure", ("numerical", "unattempted", "incomplete", "query", "partial_support"))
def test_incomplete_or_failed_candidates_never_pass_missing_arc_sentinel(failure):
    value = _completed_registration()
    evidence = value["evidence"]
    if failure == "numerical":
        evidence["candidate_numerical_failure_count"] = 1
    elif failure == "unattempted":
        evidence["candidates"][0]["execution_status"] = "NOT_ATTEMPTED"
    elif failure == "incomplete":
        evidence["candidate_evaluation_complete"] = False
    elif failure == "query":
        evidence["candidates"][0]["reasons"] = ["FINITE_SURFACE_QUERY_UNAVAILABLE:QUERY_BUDGET_EXCEEDED"]
    else:
        evidence["initialization"]["hypotheses"][0]["assigned_support"][0]["query_rows"] = 24
    with pytest.raises(AssertionError):
        _execution_complete(value, allow_prefilter_abstention=True)


def test_completed_prefilter_abstention_is_negative_only_and_not_nominal_pass():
    result = _completed_registration(empty=True)
    assert _execution_complete(result, allow_prefilter_abstention=True) == "PREFILTER_ABSTENTION_NOT_OBSERVED_ABSENCE"
    with pytest.raises(AssertionError):
        _execution_complete(result)
    result["evidence"]["initialization"]["status"] = "UNAVAILABLE"
    with pytest.raises(AssertionError):
        _execution_complete(result, allow_prefilter_abstention=True)


def _mock_missing_arc_workflow(branch="local_row"):
    """Sentinel-only injected report, explicitly NOT a runtime geometry claim."""
    registration = _completed_registration(empty=branch == "prefilter")
    registration["transform_scan_to_cad"] = None if branch == "prefilter" else np.eye(4).tolist()
    row = dict(feature_id="mock-window", status="UNAVAILABLE", measurements=None,
               finite_support=None, reasons=["INSUFFICIENT_ARC_SUPPORT"])
    measured = dict(status="UNAVAILABLE", computation_status="COMPLETE", reasons=[],
        finite_support_instances=[], observed_support_instance_count=0, observed_bend_count=None,
        features=[row], expected_window_count=1, measured_conditional_window_count=0)
    result = dict(status="UNAVAILABLE", predictions=[], observed_bend_count=None,
        production_use_allowed=False, phase="COMPLETE", reasons=["NO_FINITE_SCAN_SUPPORTED_PREDICTIONS"],
        registration=registration, measurements=measured,
        registration_array_binding=dict(status="CONSISTENT_REGISTRATION_ARRAY_BINDING"),
        native_features=dict(features=[{"feature_id": "mock-window"}], expected_roi_count=1))
    if branch in ("prefilter", "registration"):
        if branch == "registration":
            registration.update(status="UNAVAILABLE", transform_scan_to_cad=None,
                                reasons=["NO_SUPPORTED_OBSERVABLE_WHOLE_PART_POSE"])
            registration["evidence"]["candidates"][0]["reasons"] = ["INSUFFICIENT_FINITE_SURFACE_SUPPORT"]
        result.update(phase="WHOLE_PART_REGISTRATION", measurements=None,
                      reasons=["PROVISIONAL_REGISTRATION_UNAVAILABLE", *registration["reasons"]])
    elif branch == "local_finite":
        row.update(status="MEASURED_CONDITIONAL", measurements={"measured_radius_mm": 5.}, reasons=[],
                   finite_support=dict(status="UNAVAILABLE", instance=None, reasons=["DISCONNECTED_AXIAL_SUPPORT"]))
        measured.update(status="PARTIAL", measured_conditional_window_count=1)
    return result


@pytest.mark.parametrize("branch", ("prefilter", "registration", "local_row", "local_finite"))
def test_actual_missing_arc_helper_accepts_only_declared_geometric_report_branches(branch):
    result = _mock_missing_arc_workflow(branch)
    disposition = _missing_arc_assertions(result)
    assert disposition.endswith("NOT_OBSERVED_ABSENCE")
    assert ("PREFILTER" in disposition) == (branch == "prefilter")
    assert ("REGISTRATION" in disposition) == (branch == "registration")
    assert ("LOCAL_SUPPORT" in disposition) == branch.startswith("local_")


def test_binding_or_transform_downgrade_cannot_pass_despite_complete_candidate_evidence():
    result = _mock_missing_arc_workflow()
    registration = result["registration"]
    registration.update(status="UNAVAILABLE", transform_scan_to_cad=None,
        reasons=["INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT"])
    result.update(phase="WHOLE_PART_REGISTRATION", measurements=None, reasons=["WORKFLOW_EXCEPTION:ValueError"])
    assert _execution_complete(registration, allow_prefilter_abstention=True) == "ALL_PLANE_START_GROUPS_FULLY_EVALUATED"
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


@pytest.mark.parametrize("reason", ("INVALID_REGISTRATION_TRANSFORM_OR_BINDING_CONTRACT",
    "FINITE_SURFACE_QUERY_UNAVAILABLE:QUERY_BUDGET_EXCEEDED", "REGISTRATION_QUERY_BUDGET_EXCEEDED",
    "INVALID_OR_NUMERICAL_INITIALIZATION_INPUT", "INITIALIZATION_REPORT_BUDGET_EXCEEDED", "UNKNOWN_FAILURE"))
def test_unknown_or_failure_registration_reason_cannot_be_geometric_sentinel(reason):
    result = _mock_missing_arc_workflow("registration")
    result["registration"]["reasons"] = [reason]
    result["reasons"] = ["PROVISIONAL_REGISTRATION_UNAVAILABLE", reason]
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


@pytest.mark.parametrize("reason", ("LOCAL_OR_TOTAL_SUPPORT_BUDGET_NO_THINNING", "TANGENT_SOLVER_INCOMPLETE",
    "CIRCLE_SOLVER_INCOMPLETE", "Singular matrix", "FloatingPointError", "INVALID_FULL_RAW_SCAN", "UNKNOWN_FAILURE"))
def test_complete_local_computation_does_not_hide_row_resource_solver_or_numeric_failure(reason):
    result = _mock_missing_arc_workflow()
    result["measurements"]["features"][0]["reasons"] = [reason]
    assert result["measurements"]["computation_status"] == "COMPLETE"
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


@pytest.mark.parametrize("reason", ("AXIAL_SLAB_BUDGET", "NUMERIC_RESOLUTION_TOO_COARSE",
                                   "INVALID_SUPPORT_ROWS", "MEASURED_CARRIER_UNAVAILABLE", "UNKNOWN_FAILURE"))
def test_finite_support_resource_numeric_input_or_carrier_failure_is_not_geometric_abstention(reason):
    result = _mock_missing_arc_workflow("local_finite")
    result["measurements"]["features"][0]["finite_support"]["reasons"] = [reason]
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


@pytest.mark.parametrize("fault", ("empty_native", "missing_window", "wrong_feature_id", "mixed_reason",
                                  "missing_reason", "top_level_input_failure", "workflow_exception"))
def test_local_sentinel_requires_exact_nonempty_window_denominator_and_unmixed_geometric_reasons(fault):
    result = _mock_missing_arc_workflow()
    measured = result["measurements"]
    if fault == "empty_native":
        result["native_features"]["features"] = []
    elif fault == "missing_window":
        measured["expected_window_count"] = 2
    elif fault == "wrong_feature_id":
        measured["features"][0]["feature_id"] = "another-window"
    elif fault == "mixed_reason":
        measured["features"][0]["reasons"].append("TANGENT_SOLVER_INCOMPLETE")
    elif fault == "missing_reason":
        measured["features"][0]["reasons"] = []
    elif fault == "top_level_input_failure":
        measured["reasons"] = ["INVALID_FULL_RAW_SCAN"]
    else:
        result["reasons"] = ["WORKFLOW_EXCEPTION:ValueError"]
    with pytest.raises(AssertionError):
        _missing_arc_assertions(result)


def test_case_artifact_cap_never_returns_a_partial_json_prefix():
    record = {"all_candidates": [{"index": 0}, {"index": 1}], "passed": False}
    encoded = _encode_case(record)
    assert json.loads(encoded) == record
    assert _encode_case(record, len(encoded)) == encoded
    with pytest.raises(ValueError, match="COMPLETE_CASE_ARTIFACT_BUDGET_EXCEEDED"):
        _encode_case(record, len(encoded)-1)


def test_artifacts_are_fresh_exclusive_and_hash_bound(tmp_path):
    directory = _new_artifact_directory(tmp_path/"ARTIFACTS")
    with pytest.raises(FileExistsError):
        _new_artifact_directory(directory)
    record = dict(case="invented_helper_only", passed=False, candidates=[{"index": 0, "reasons": ["UNKNOWN"]}])
    artifact = _write_case_artifact(directory, 0, record)
    assert artifact["status"] == "COMPLETE_CASE_RECORD_RETAINED"
    assert json.loads((directory/artifact["relative_path"]).read_bytes()) == record
    with pytest.raises(FileExistsError):
        _write_case_artifact(directory, 0, {"passed": True})


def test_fixed_invented_native_planes_v4_workflow_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must opt in under the frozen serial180/120-second2-GiB protocol")
    directory = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    assert directory, "Root supervisor must supply its fresh retained-job ARTIFACTS path"
    receipt = run_invented_mesh_workflow_planes_v4_checks(artifact_directory=directory)
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["stdout_bytes"]
    print(encoded, flush=True)
    assert len(receipt["cases"]) == len(CONNECTED_CASES) == 3
    assert receipt["fixture_source_sha256_before"] == receipt["fixture_source_sha256_after"]
    assert receipt["matrix_passed"] is True
    assert all(row.get("artifact", {}).get("status") == "COMPLETE_CASE_RECORD_RETAINED" for row in receipt["cases"])
    assert all(row["passed"] for row in receipt["cases"]), receipt["cases"]
