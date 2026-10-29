"""Separately frozen 32,768-CAD-sample successor; original failed run is intact.

Predecessor: mesh_connected_native_v1-jkz08n45, fixed 8192-sample matrix, 1/3.
Its two nominal cases found a generated pose with maximum oracle action error
.001048375807 mm, but scan-to-CAD-SAMPLE coverage was .9120639535 < .95. Its
nearest-sample distance p95 was .549465376565 mm. Distance to a fixed set is
1-Lipschitz: even correcting that entire pose error leaves p95 >= .548417 mm,
above the frozen .5 mm gate. CAD coverage, surface residual and observability
passed. Thus this successor tests sampling resolution, not weaker acceptance.

The independently constructed solid has area 3098 + 170.5*pi = 3633.64155 mm^2.
Fourfold samples halve the mean spacing scale; this motivates, but does not
guarantee, a successful finite-sample gate. The missing-arc predecessor also
found an inner/outer-flange ambiguity; denser samples do NOT qualify side
identity, and the same no-predictions sentinel remains required.

ONLY workflow.cad_sample_count changes, 8192 -> 32768. Exact same geometry,
three cases, 24-start registration policy, fitting policy, support policy,
thresholds, numerical error ceilings and external 60 s wall / 45 s CPU / 1 GiB
process-tree limit remain. Both nominal cases must measure; abstention is failure.
This is a new synthetic sampling-policy protocol, not a retry of the original,
not physical validation, and not the historical/frozen r0 engine comparison.

Original helpers are loaded without execution of their test/native entrypoints.
Policies are newly constructed and dataclasses.replace returns a local copy;
no original module global or configuration is modified. Native work starts only
on an explicit root-supervised worker/test call. No physical/raw/Gold file is
read; the predecessor is cited by identity, never opened by this module.
"""

import ast
import builtins
import copy
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


ORIGINAL_SOURCE = Path(__file__).with_name("test_cad_guided_mesh_connected_native_v1.py")
CONNECTED_CASES = (
    ("v1_nominal_oblique", "v1", False, 1),
    ("v1_missing_arc_oblique", "v1", True, 0),
    ("r0_common_wrapper_nominal_oblique", "r0", False, 1),
)
NATIVE_OPT_IN = "SHERMAN_RUN_INVENTED_MESH_CONNECTED_DENSE_V1"
RESOURCE_LIMITS = {"wall_seconds": 60, "cpu_seconds": 45, "process_tree_rss_bytes": 1024**3}
CRITERIA = {"radius_error_mm": 2e-5, "absolute_sweep_error_deg": 2e-5,
            "generator_endpoint_error_mm": .02, "pose_action_error_mm": .20}
PREDECESSOR_RECEIPT = "mesh_connected_native_v1-jkz08n45"


@lru_cache(maxsize=1)
def _original():
    spec = importlib.util.spec_from_file_location("_mesh_connected_8192_predecessor", ORIGINAL_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CONNECTED_CASES == CONNECTED_CASES
    assert module.CRITERIA == CRITERIA
    assert module.RESOURCE_LIMITS == RESOURCE_LIMITS
    return module


def _policies(branch):
    original = _original()
    request = original._policies(branch)
    assert request["workflow_policy"].cad_sample_count == 8192
    return {**request, "workflow_policy": replace(request["workflow_policy"], cad_sample_count=32768)}


def _common_assertions(result, cad_blob, scan_blob):
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
    assert result["registration"]["evidence"]["candidate_count"] == 24
    assert result["reference_frame_compatibility"] == "NOT_ASSERTED_BY_INFERENCE"


def _receipt(records):
    return dict(cases=records, expected_case_count=len(CONNECTED_CASES), criteria=CRITERIA,
                external_resource_limits=RESOURCE_LIMITS,
                protocol="CAD_GUIDED_MESH_CONNECTED_NATIVE_DENSE_V1",
                predecessor_receipt=PREDECESSOR_RECEIPT,
                inference_policy_delta={"workflow.cad_sample_count": {"before": 8192, "after": 32768}},
                policy_scope="FIXED_SYNTHETIC_SAMPLING_SUCCESSOR_NOT_PHYSICAL_RUN_POLICY",
                evidence_class="INVENTED_NATIVE_CONNECTED_NOT_PHYSICAL_GENERALIZATION")


def run_invented_mesh_workflow_dense_checks():
    """Explicit root-supervised worker entry; no inherited test is executed."""
    from backend.cad_guided_mesh_offline_v1 import inspect_cad_guided_mesh_v1

    original = _original()
    try:
        cad_blob = original._owned_step_bytes(original._asymmetric_native_strip())
    except Exception as error:
        return _receipt([dict(case=name, passed=False, phase="fixture",
                              error_type=type(error).__name__, error=str(error)[:1000])
                         for name, _, _, _ in CONNECTED_CASES])
    records = []
    for name, branch, missing_arc, expected_predictions in CONNECTED_CASES:
        record = dict(case=name, passed=False, phase="fixture", expected_predictions=expected_predictions)
        try:
            raw, faces, cad_frame = original._analytic_mesh_scan(missing_arc)
            scan_blob = original._ply_bytes(raw, faces)
            record.update(scan_rows=len(raw), mesh_faces=len(faces), cad_sha256=hashlib.sha256(cad_blob).hexdigest())
            record["phase"] = "actual_byte_to_workflow"
            result = inspect_cad_guided_mesh_v1(scan_bytes=scan_blob, cad_bytes=cad_blob,
                                               allow_ocp=True, **_policies(branch))
            record["workflow"] = result
            record["phase"] = "frozen_assertions"
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
    return _receipt(records)


@pytest.mark.parametrize("branch", ("r0", "v1"))
def test_only_cad_sampling_request_differs_without_original_mutation(monkeypatch, branch):
    calls, real_import = [], builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            calls.append(name)
            raise AssertionError("SOURCE_POLICY_CHECK_MUST_NOT_IMPORT_NATIVE")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    original = _original()
    previous = {key: asdict(value) for key, value in original._policies(branch).items()}
    globals_before = copy.deepcopy((original.CONNECTED_CASES, original.CRITERIA, original.RESOURCE_LIMITS))
    changed = {key: asdict(value) for key, value in _policies(branch).items()}
    expected = copy.deepcopy(previous)
    expected["workflow_policy"]["cad_sample_count"] = 32768
    assert changed == expected
    assert previous["workflow_policy"]["cad_sample_count"] == 8192
    assert {key: asdict(value) for key, value in original._policies(branch).items()} == previous
    assert (original.CONNECTED_CASES, original.CRITERIA, original.RESOURCE_LIMITS) == globals_before
    assert calls == []


def test_all_common_assertions_are_identical_except_expected_cad_sample_count():
    def function(path):
        return next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_common_assertions")

    original, current = function(ORIGINAL_SOURCE), function(Path(__file__))
    differences = [node for node in ast.walk(current) if isinstance(node, ast.Constant) and node.value == 32768]
    assert len(differences) == 1
    differences[0].value = 8192  # AST-only comparison; no runtime/source mutation.
    assert ast.dump(current, include_attributes=False) == ast.dump(original, include_attributes=False)


def test_cases_criteria_and_external_limits_are_exact_predecessor_contract():
    original = _original()
    assert CONNECTED_CASES == original.CONNECTED_CASES
    assert CRITERIA == original.CRITERIA
    assert RESOURCE_LIMITS == original.RESOURCE_LIMITS
    assert len(CONNECTED_CASES) == 3


def test_fixed_invented_dense_mesh_workflow_matrix():
    if os.environ.get(NATIVE_OPT_IN) != "1":
        pytest.skip("Root must explicitly opt in under the unchanged external native budget")
    receipt = run_invented_mesh_workflow_dense_checks()
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    print(json.dumps({"connected_dense_compact": [
        {key: row.get(key) for key in ("case", "passed", "phase", "error_type", "error", "errors")}
        for row in receipt["cases"]]}, sort_keys=True), flush=True)
    assert len(receipt["cases"]) == len(CONNECTED_CASES)
    assert all(row["passed"] for row in receipt["cases"]), receipt
