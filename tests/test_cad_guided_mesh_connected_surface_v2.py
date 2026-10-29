"""Frozen three-case native finite-surface successor; predecessors stay intact.

The original invented STEP/analytic mesh and post-inference truth helpers are
reused directly, never copied or revised. All three original cases remain:
v1 nominal MUST publish one measured finite instance; missing arc MUST publish
zero; the r0-extractor common-wrapper nominal MUST publish one. This is NOT the
historic/frozen R0 engine. A nominal abstention remains a failure.

The 32,768 CAD samples and EVERY numeric inference policy value match the dense
V1 successor. Only the workflow policy type/entry change to V2, which introduces
finite-triangle correspondence and its declared execution budgets. The original
radius/sweep ceilings (2e-5 mm/degree), endpoint ceiling (.02 mm) and pose-action
ceiling (.20 mm) are applied by the unchanged original assertion function.
The missing-arc sentinel cannot pass from a fatal/incomplete query execution.
All 24 candidates must be attempted; individual numerical-failure rows remain
visible and do not disqualify an otherwise qualified nominal result.

This NEW protocol has prospectively assigned external 180 s wall / 120 s CPU /
2 GiB process-tree RSS / 1 MiB stdout limits, one serial heavy job. It does not
retry either retained negative V1 matrix or change their budgets/criteria.
Previous receipts are cited only; no retained output, physical
part, Gold, held-out, or email file is read. Native files exist only in the
original helper's freshly owned temporary directory and workflow byte snapshot.

No native or registration work occurs at import/ordinary collection. Root must
review/freeze source and explicitly opt in before running. The true pose is
used to construct the synthetic scan and evaluate returned candidates only;
it is never supplied to workflow inference. Full outputs and failures remain
in the receipt; successful synthetic checks do not establish floor readiness.
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
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_CONNECTED_SURFACE_V2"
PREDECESSOR_RECEIPTS = ("mesh_connected_native_v1-jkz08n45", "mesh_connected_native_dense_v1-j5p3bnlr")


@lru_cache(maxsize=1)
def _dense():
    spec = importlib.util.spec_from_file_location("_surface_v2_dense_predecessor_helpers", DENSE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CONNECTED_CASES == CONNECTED_CASES
    assert module.CRITERIA == CRITERIA
    return module


def _policies(branch):
    from backend.cad_guided_mesh_offline_v2 import MeshWorkflowPolicyV2

    original = _dense()._policies(branch)
    assert original["workflow_policy"].cad_sample_count == 32768
    return {**original, "workflow_policy": MeshWorkflowPolicyV2(**asdict(original["workflow_policy"]))}


def _fixture_source_digests():
    # Protocol/fixture consistency only; the root harness owns full source closure.
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), DENSE_SOURCE.resolve(), ORIGINAL_SOURCE.resolve())}


def _execution_complete(registration):
    rows = registration["evidence"]["candidates"]
    assert registration["evidence"]["candidate_count"] == 24
    assert [row["index"] for row in rows] == list(range(24))
    fatal = ("NOT_ATTEMPTED_AFTER_SURFACE_QUERY_FAILURE", "REGISTRATION_QUERY_BUDGET_EXCEEDED",
             "FINITE_SURFACE_QUERY_UNAVAILABLE:")
    assert not any(reason.startswith(fatal) for row in rows for reason in row["reasons"])
    # A completed geometric diagnosis must exist. All-numerical failure is not
    # evidence of missing scan support, while isolated failed starts are retained.
    assert any(all(key in row for key in ("rotation", "translation", "scan_coverage",
                                          "inlier_surface_p95_mm")) for row in rows)


def _common_assertions(result, cad_blob, scan_blob):
    _dense()._common_assertions(result, cad_blob, scan_blob)
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V2"
    assert result["registration_surface"] == "FINITE_ORIENTED_TRIANGLES_NOT_ANALYTIC_CAD"
    binding = result["triangle_sample_binding"]
    assert binding["status"] == "CONSISTENT_IMMUTABLE_TRIANGLE_SAMPLE_SNAPSHOT"
    assert binding["initialization_sample_count"] == 32768
    assert binding["oriented_triangle_xyz_sha256"] == result["cad_surface_sampling"]["digests"]["oriented_triangle_xyz_sha256"]
    assert binding["initialization_points_sha256"] == result["cad_surface_sampling"]["digests"]["sample_points_sha256"]
    assert result["registration"]["evidence"]["schema"] == "FINITE_TRIANGLE_WHOLE_PART_REGISTRATION_V2"
    _execution_complete(result["registration"])


def _candidate_action_errors(result, raw, cad_frame):
    """Post-inference evaluation only; retain refused/unattempted slots too."""
    rows = []
    for candidate in result.get("registration", {}).get("evidence", {}).get("candidates", []):
        row = dict(index=candidate["index"], maximum_mm=None, rms_mm=None)
        if "rotation" in candidate and "translation" in candidate:
            error = np.linalg.norm(raw @ np.asarray(candidate["rotation"]).T
                                   + candidate["translation"] - cad_frame, axis=1)
            row.update(maximum_mm=float(error.max()), rms_mm=float(np.sqrt(np.mean(error**2))))
        rows.append(row)
    return rows


def _receipt(records, source_before, source_after):
    return dict(protocol="CAD_GUIDED_MESH_CONNECTED_FINITE_SURFACE_V2", cases=records,
                expected_case_count=len(CONNECTED_CASES), criteria=CRITERIA,
                predecessor_receipts=PREDECESSOR_RECEIPTS, external_resource_limits=RESOURCE_LIMITS,
                fixture_source_sha256_before=source_before, fixture_source_sha256_after=source_after,
                policy_delta="V2_WORKFLOW_AND_FINITE_SURFACE_ENGINE; DENSE_V1_NUMERIC_VALUES_UNCHANGED",
                policy_scope="FIXED_INVENTED_NATIVE_SUCCESSOR_NOT_PHYSICAL_RUN_POLICY",
                evidence_class="INVENTED_NATIVE_CONNECTED_NOT_PHYSICAL_GENERALIZATION")


def run_invented_mesh_workflow_surface_v2_checks():
    """Root-supervised serial entry; every original case retains its denominator."""
    from backend.cad_guided_mesh_offline_v2 import inspect_cad_guided_mesh_v2

    before = _fixture_source_digests()
    original = _dense()._original()
    try:
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
    except Exception as error:
        records = [dict(case=name, passed=False, phase="fixture", expected_predictions=count,
                        error_type=type(error).__name__, error=str(error)[:1000])
                   for name, _, _, count in CONNECTED_CASES]
        return _receipt(records, before, _fixture_source_digests())
    records = []
    for name, branch, missing_arc, expected_predictions in CONNECTED_CASES:
        record = dict(case=name, passed=False, phase="fixture", expected_predictions=expected_predictions)
        try:
            raw, faces, cad_frame = original._analytic_mesh_scan(missing_arc)
            scan_blob = original._ply_bytes(raw, faces)
            record.update(scan_rows=len(raw), mesh_faces=len(faces),
                          cad_sha256=hashlib.sha256(cad_blob).hexdigest(),
                          scan_sha256=hashlib.sha256(scan_blob).hexdigest())
            record["phase"] = "actual_byte_to_v2_workflow"
            result = inspect_cad_guided_mesh_v2(scan_bytes=scan_blob, cad_bytes=cad_blob,
                                               allow_ocp=True, **_policies(branch))
            record["workflow"] = result
            record["phase"] = "post_inference_evaluation"
            record["post_inference_candidate_action_errors"] = _candidate_action_errors(result, raw, cad_frame)
            record["phase"] = "unchanged_frozen_assertions"
            _common_assertions(result, cad_blob, scan_blob)
            assert result["input_preparation"]["source_bindings"]["raw_xyz_sha256"] == hashlib.sha256(
                np.asarray(raw, dtype="<f8").tobytes()).hexdigest()
            assert result["input_preparation"]["coverage"]["raw_vertex_count"] == len(raw)
            assert len(result["predictions"]) == expected_predictions
            if missing_arc:
                assert result["status"] == "UNAVAILABLE"
                assert result["reasons"]
                assert result["phase"] in ("WHOLE_PART_REGISTRATION", "COMPLETE")
            else:
                record["errors"] = original._nominal_assertions(result, raw, cad_frame)
            record.update(passed=True, phase="complete")
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error)[:1000])
        records.append(record)
    after = _fixture_source_digests()
    if before != after:
        for record in records:
            record.update(passed=False, source_consistency_failure="FIXTURE_OR_PROTOCOL_SOURCE_CHANGED")
    return _receipt(records, before, after)


@pytest.mark.parametrize("branch", ("v1", "r0"))
def test_policy_values_exactly_match_dense_predecessor_without_native_or_mutation(monkeypatch, branch):
    calls, real_import = [], builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise AssertionError("SOURCE_CONTROL_MUST_NOT_IMPORT_NATIVE")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    dense = _dense()
    original = dense._original()
    before = {key: asdict(value) for key, value in dense._policies(branch).items()}
    globals_before = copy.deepcopy((dense.CONNECTED_CASES, dense.CRITERIA, dense.RESOURCE_LIMITS,
                                   original.CONNECTED_CASES, original.CRITERIA, original.RESOURCE_LIMITS))
    current = _policies(branch)
    assert type(current["workflow_policy"]).__name__ == "MeshWorkflowPolicyV2"
    assert {key: asdict(value) for key, value in current.items()} == before
    assert before["workflow_policy"]["cad_sample_count"] == 32768
    assert {key: asdict(value) for key, value in dense._policies(branch).items()} == before
    assert (dense.CONNECTED_CASES, dense.CRITERIA, dense.RESOURCE_LIMITS,
            original.CONNECTED_CASES, original.CRITERIA, original.RESOURCE_LIMITS) == globals_before
    assert calls == []


def test_original_cases_and_numerical_ceilings_are_unchanged_but_budget_is_new():
    dense = _dense()
    original = dense._original()
    assert CONNECTED_CASES == dense.CONNECTED_CASES == original.CONNECTED_CASES
    assert CRITERIA == dense.CRITERIA == original.CRITERIA
    assert CRITERIA == dict(radius_error_mm=2e-5, absolute_sweep_error_deg=2e-5,
                            generator_endpoint_error_mm=.02, pose_action_error_mm=.20)
    assert dense.RESOURCE_LIMITS == original.RESOURCE_LIMITS == dict(
        wall_seconds=60, cpu_seconds=45, process_tree_rss_bytes=1024**3)
    assert RESOURCE_LIMITS == dict(wall_seconds=180, cpu_seconds=120,
                                   process_tree_rss_bytes=2 * 1024**3, stdout_bytes=1024**2)


def test_original_fixture_and_truth_helpers_are_called_directly_without_pose_patch():
    tree = ast.parse(Path(__file__).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    worker = functions["run_invented_mesh_workflow_surface_v2_checks"]
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call)]
    helper_names = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "original"}
    assert helper_names == {"_owned_step_bytes", "_asymmetric_native_strip", "_analytic_mesh_scan",
                            "_ply_bytes", "_nominal_assertions"}
    forbidden = {"_oracle_pose", "_refine", "_starts", "register_whole_part_v2", "_weighted_rigid_increment"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(worker))
    assert not any(isinstance(node, ast.Name) and node.id in ("monkeypatch", "setattr") for node in ast.walk(worker))
    workflow_call = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "inspect_cad_guided_mesh_v2")
    assert [key.arg for key in workflow_call.keywords] == ["scan_bytes", "cad_bytes", "allow_ocp", None]
    assert isinstance(workflow_call.keywords[-1].value, ast.Call)
    assert workflow_call.keywords[-1].value.func.id == "_policies"
    oracle_call = next(node for node in calls if isinstance(node.func, ast.Name)
                       and node.func.id == "_candidate_action_errors")
    assert oracle_call.lineno > workflow_call.lineno
    assert "_nominal_assertions" not in functions and "_analytic_mesh_scan" not in functions
    common = functions["_common_assertions"]
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "_common_assertions" for node in ast.walk(common))


def _completed_registration():
    return dict(evidence=dict(candidate_count=24, candidates=[dict(index=index, reasons=[],
        rotation=np.eye(3).tolist(), translation=[0., 0., 0.], scan_coverage=1., inlier_surface_p95_mm=0.)
        for index in range(24)]))


@pytest.mark.parametrize("reason", ("NOT_ATTEMPTED_AFTER_SURFACE_QUERY_FAILURE",
    "REGISTRATION_QUERY_BUDGET_EXCEEDED", "FINITE_SURFACE_QUERY_UNAVAILABLE:QUERY_BUDGET_EXCEEDED_NODE_VISITS",
    "FINITE_SURFACE_QUERY_UNAVAILABLE:UNSUPPORTED_QUERY_ARITHMETIC_CONDITIONING"))
def test_fatal_or_unattempted_registration_cannot_satisfy_missing_arc_sentinel(reason):
    registration = _completed_registration()
    registration["evidence"]["candidates"][3]["reasons"] = [reason]
    with pytest.raises(AssertionError):
        _execution_complete(registration)


def test_individual_numerical_failure_is_retained_without_new_nominal_gate():
    registration = _completed_registration()
    failed = dict(index=3, reasons=["CANDIDATE_NUMERICAL_FAILURE"])
    registration["evidence"]["candidates"][3] = failed
    _execution_complete(registration)
    assert registration["evidence"]["candidates"][3] is failed
    registration["evidence"]["candidates"] = [dict(index=index, reasons=["CANDIDATE_NUMERICAL_FAILURE"])
                                               for index in range(24)]
    with pytest.raises(AssertionError):
        _execution_complete(registration)


def test_fixed_invented_native_surface_v2_workflow_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must explicitly opt in under the new serial 180/120-second, 2-GiB native protocol")
    receipt = run_invented_mesh_workflow_surface_v2_checks()
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    print(encoded, flush=True)
    assert len(encoded.encode()) <= RESOURCE_LIMITS["stdout_bytes"]
    assert len(receipt["cases"]) == 3
    assert receipt["fixture_source_sha256_before"] == receipt["fixture_source_sha256_after"]
    assert all(row["passed"] for row in receipt["cases"]), [
        {key: row.get(key) for key in ("case", "phase", "error_type", "error", "errors")}
        for row in receipt["cases"]]
