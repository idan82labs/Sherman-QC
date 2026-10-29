"""Prospective SINGLE invented L-R STEP/PLY-to-V12 connected qualification.

Source-authored only. No native execution on import/ordinary collection. Exact
array POST, new native protocol/source freeze and independent PRE are required.
The current array gate is negative: this is inactive future scaffolding, not
an execution release or a successor qualification claim.
The original strip/STEP/PLY functions and L-R scan/oracles are reused unchanged.
Only the nominal CAD solid receives the declared independent CAD placement;
scanner truth, explicit feature declarations and analytic CAD triangles never
enter the byte workflow. All returned workflow data survive ordinary failures.
Serialization/cap/IO failure is an explicit retention failure, not a good prefix.
"""

import ast
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OPT_IN = "SHERMAN_RUN_CONNECTED_V12_NATIVE_BYTES"
ARTIFACT_ENV = "SHERMAN_CONNECTED_V12_NATIVE_BYTES_ARTIFACT_DIRECTORY"
CASE_ID = "L-R"
NATIVE_SOURCE = "tests/test_cad_guided_mesh_connected_native_v1.py"
ARRAY_SOURCE = "tests/test_cad_guided_connected_v12.py"
SOURCE_HASHES = {
    NATIVE_SOURCE: "11455ceda7283bcd1b84dbd037b7bd16bd2517bd8d44985ef8f774d2e5c100e8",
    ARRAY_SOURCE: "fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd",
}
SOURCE_DEPENDENCIES = (
    NATIVE_SOURCE, ARRAY_SOURCE, "backend/cad_guided_mesh_offline_v12.py",
    "docs/research/2026-09-10_connected_v12_native_bytes_protocol.md",
)
RESOURCE_LIMITS = dict(wall_seconds=300, worker_cpu_seconds=240,
    sampled_process_group_rss_bytes=4*1024**3, stdout_bytes=1024**2)
ARTIFACT_LIMITS = dict(case_count=1, start_json_bytes=4096, cad_bytes=8*1024**2,
    scan_bytes=1024**2, result_json_bytes=16*1024**2, total_bytes=25*1024**2+4096)
ARTIFACT_NAMES = ("CASE_START.json", "CAD.step", "SCAN.ply", "CASE_RESULT.json")
WORKFLOW_FIELDS = dict(max_registration_scan_rows=50000, cad_sample_count=32768,
    cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
    native_adapter_version="v1", unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _array_sha(value, dtype):
    return _sha(np.asarray(value, dtype=dtype).tobytes())


def _source_bytes(relative):
    assert relative in SOURCE_HASHES, "UNDECLARED_FIXTURE_SOURCE"
    path = ROOT/relative
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and 1 <= before.st_size <= 2*1024**2
    with path.open("rb") as stream:
        blob = stream.read(2*1024**2+1)
    after = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    assert identity(before) == identity(after) and len(blob) == before.st_size
    assert _sha(blob) == SOURCE_HASHES[relative], "FIXTURE_SOURCE_DRIFT"
    return blob


@lru_cache(maxsize=2)
def _fixture_module(relative):
    """Compile the same verified buffer; no loader performs a second open."""
    blob = _source_bytes(relative)
    name = "_connected_v12_native_fixture_" + Path(relative).stem
    module = types.ModuleType(name)
    module.__file__ = str(ROOT/relative)
    sys.modules[name] = module  # dataclass needs its defining module identity.
    try:
        exec(compile(blob, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _move_nominal_cad(shape, rotation, translation):
    # Same native transform call pattern as the existing native feature tests;
    # these inputs are CAD placement, not the independently generated scan pose.
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    assert np.array_equal(rotation.T@rotation, np.eye(3)) and np.linalg.det(rotation) == 1.
    transform = gp_Trsf()
    transform.SetValues(*[float(value) for row, shift in zip(rotation, translation)
                          for value in (*row, shift)])
    expected = np.column_stack((rotation, translation))
    assert all(transform.Value(i+1, j+1) == expected[i, j] for i in range(3) for j in range(4))
    builder = BRepBuilderAPI_Transform(shape, transform, True)
    assert builder.IsDone() and not builder.Shape().IsNull()
    return builder.Shape()


def _make_fixture():
    """Exactly one original L-R scan, with independently constructed native CAD."""
    native, array = _fixture_module(NATIVE_SOURCE), _fixture_module(ARRAY_SOURCE)
    case = array.CURVED_CASES[2]
    assert (case.case_id, case.radius_mm, case.sweep_deg, case.localized, case.noisy, case.pose) == (
        "L-R", 5.3, 90., True, True, "B")
    inputs, truth = array.curved_packet(case)
    raw, triangles = inputs["scan_points"], inputs["scan_triangles"]
    assert raw.shape == (650, 3) and triangles.shape == (1176, 3)
    assert inputs["position_error_bounds_mm"] == .05
    assert np.max(np.linalg.norm(truth["actual_coordinate_errors"], axis=1)) <= .002
    cache = Path(os.environ["TMPDIR"])
    assert cache.is_absolute() and cache.is_dir(), "FRESH_SUPERVISOR_OWNED_CACHE_REQUIRED"
    # Original helpers create only a fresh owned STEP file plus bytes. The
    # workflow creates its own separate exact-byte STEP snapshot in this cache.
    shape = _move_nominal_cad(native._asymmetric_native_strip(), array.CAD_ROTATION, array.CAD_TRANSLATION)
    cad_blob = native._owned_step_bytes(shape)
    scan_blob = native._ply_bytes(raw, triangles)
    assert 1 <= len(cad_blob) <= ARTIFACT_LIMITS["cad_bytes"]
    assert 1 <= len(scan_blob) <= ARTIFACT_LIMITS["scan_bytes"]
    return cad_blob, scan_blob, raw, triangles, truth


def _policy_objects():
    array = _fixture_module(ARRAY_SOURCE)
    from backend.cad_guided_mesh_offline_v12 import MeshWorkflowPolicyV12
    from backend.cad_guided_native_features import NativeGuidedPolicy
    _, kwargs, instance_policy = array._policy_objects("C2", planar=False)
    kwargs = dict(kwargs)
    kwargs.pop("unit_assumption")  # Declared by the unchanged workflow policy.
    kwargs["initialization_budgets"] = kwargs.pop("budgets")
    measurement = copy.deepcopy(array.PHYSICAL_POLICIES["measurement"])
    measurement["flange_window_mm"] = tuple(measurement["flange_window_mm"])
    kwargs.update(workflow_policy=MeshWorkflowPolicyV12(**WORKFLOW_FIELDS),
        measurement_policy=NativeGuidedPolicy(**measurement), instance_policy=instance_policy,
        position_error_bound_mm=.05, allow_ocp=True, connected_measurement_contexts=True)
    return kwargs


def _execute_workflow(cad_blob, scan_blob):
    """The entire inference interface: bytes and fixed policies, never truth."""
    from backend.cad_guided_mesh_offline_v12 import inspect_cad_guided_mesh_v12
    return inspect_cad_guided_mesh_v12(cad_bytes=cad_blob, scan_bytes=scan_blob, **_policy_objects())


def _assert_byte_and_context_binding(result, cad_blob, scan_blob, raw, triangles):
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V12"
    assert result["status"] == "PARTIAL" and result["phase"] == "COMPLETE", result.get("reasons")
    assert result["observed_bend_count"] is None and result["production_use_allowed"] is False
    assert result["conformance"] == "INDETERMINATE"
    assert result["registration_mode"] == "FINITE_GENERATED_CONTEXTS_NOT_ACCEPTED_POSE"
    assert result["prediction_frame"] == result["measurement_geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
    assert result["cad_sha256"] == _sha(cad_blob)
    prepared = result["input_preparation"]
    source = prepared["source_bindings"]
    xyz_sha, triangle_sha = _array_sha(raw, "<f8"), _array_sha(triangles, "<i8")
    assert source["scan_sha256"] == _sha(scan_blob) and source["raw_xyz_sha256"] == xyz_sha
    assert prepared["coverage"]["raw_vertex_count"] == len(raw)
    topology = result["scan_triangle_binding"]
    assert topology["source_sha256"] == _sha(scan_blob)
    assert topology["raw_points_sha256"] == xyz_sha and topology["triangle_raw_rows_sha256"] == triangle_sha
    assert topology["raw_vertex_count"] == 650 and topology["triangle_count"] == 1176
    units = result["cad_units"]
    assert units["target_unit"] == "mm" and units["transferred_roots"] == 1
    assert units["target_unit_action"] == "SET_SYSTEM_LENGTH_UNIT_1_AFTER_READ_BEFORE_TRANSFER"
    assert units["physical_calibration"] == "NOT_ESTABLISHED"
    assert units["source_unit_names"]["length"]
    assert all(name.upper() in ("MILLIMETRE", "MILLIMETER", "MM") for name in units["source_unit_names"]["length"])
    native = result["native_features"]
    assert native["adapter_version"] == "NATIVE_TRANSITION_ROI_V1"
    assert native["candidate_inventory_complete"] is True and native["exhaustive_bend_inventory"] is False
    assert native["features"] and native["expected_roi_count"] == len(native["features"])
    assert native["physical_observed_bend_count"] is None and native["production_use_allowed"] is False
    cad, sample_binding = result["cad_surface_sampling"], result["triangle_sample_binding"]
    assert cad["status"] == "AVAILABLE" and cad["sampling_policy"]["sample_count"] == 32768
    assert sample_binding["initialization_sample_count"] == 32768
    assert sample_binding["oriented_triangle_xyz_sha256"] == cad["digests"]["oriented_triangle_xyz_sha256"]
    registration = result["registration"]
    assert registration["status"] == "CONDITIONAL_MEASUREMENT_CONTEXTS"
    assert registration["transform_scan_to_cad"] is None
    evidence = registration["evidence"]
    assert evidence["candidate_evaluation_complete"] is True and evidence["numerical_consumer_complete"] is True
    assert evidence["accepted_registration"] is False and evidence["global_pose_uniqueness"] == "NOT_ESTABLISHED"
    initialization = evidence["initialization"]
    assert initialization["status"] == "COMPLETE" and initialization["box_screening_complete"] is True
    assert initialization["numerical_consumer_complete"] is True
    assert initialization["joint_proposal_accounting"]["pair_lane_enabled"] is True
    expected = dict(scan_points_sha256=xyz_sha, scan_triangle_raw_rows_sha256=triangle_sha,
        position_error_bounds_mm_sha256=_array_sha(np.full(len(raw), .05), "<f8"),
        cad_triangles_sha256=cad["digests"]["oriented_triangle_xyz_sha256"],
        cad_face_ids_sha256=cad["digests"]["source_face_ids_sha256"],
        cad_source_triangle_ids_sha256=cad["digests"]["source_triangle_ids_sha256"])
    assert initialization["input_digests"] == evidence["input_digests"] == expected
    candidates, indices = evidence["candidates"], evidence["measurement_context_indices"]
    assert type(indices) is list and indices and all(type(i) is int for i in indices)
    assert indices == [i for i, candidate in enumerate(candidates)
        if candidate["conditional_context_support"]["status"] == "SUPPORTED_CONTEXT"]
    bindings = result["registration_array_binding"]
    assert len(bindings) == len(indices)
    for index, binding in zip(indices, bindings):
        candidate = candidates[index]
        assert type(candidate["index"]) is int and candidate["index"] == index
        assert candidate["execution_status"] == "ATTEMPTED"
        assert candidate["numerical_consumer_status"] == "COMPLETE_V3_INTERVAL_SUPPORT_POLICY"
        assert candidate["conditional_context_support"]["accepted_registration"] is False
        pose = np.r_[np.asarray(candidate["rotation"]).ravel(), candidate["translation"]]
        assert candidate["conditional_context_support"]["pose_sha256"] == _array_sha(pose, "<f8")
        assert binding["raw_scan_points_sha256"] == xyz_sha and binding["raw_triangle_rows_sha256"] == triangle_sha
        assert binding["position_error_bounds_mm_sha256"] == expected["position_error_bounds_mm_sha256"]
        assert binding["cad_triangles_sha256"] == expected["cad_triangles_sha256"]
        assert binding["cad_sampler_points_sha256"] == cad["digests"]["sample_points_sha256"]
        assert binding["registration_canonical_cad_points_sha256"] == evidence["cad_initialization_points_sha256"]
        assert binding["registration_selection"] == "ALL_RAW_ROWS"
        group = initialization["start_groups"][index]
        assert candidate["hypothesis_indices"] == group["hypothesis_indices"]
        assert candidate["exact_group_index"] == group["exact_group_index"]
        assert group["hypothesis_indices"], "MISSING_CONTEXT_ORIGINS"
        for origin_index in group["hypothesis_indices"]:
            origin = initialization["hypotheses"][origin_index]
            assert origin["prefilter_status"] == "PASSED"
            assert origin["exact_start_group_index"] == candidate["exact_group_index"]
            assert np.array_equal(origin["rotation"], candidate["initialized_rotation"])
            assert np.array_equal(origin["translation_mm"], candidate["initialized_translation_mm"])
    return dict(scan_sha256=_sha(scan_blob), raw_xyz_sha256=xyz_sha, triangle_raw_rows_sha256=triangle_sha,
                eligible_context_count=len(indices), native_roi_count=len(native["features"]),
                actual_native_cad_sample_count=32768, effective_scan_rows=len(raw),
                full_8192_refinement_rows_exercised=False)


def _evaluate(result, cad_blob, scan_blob, raw, triangles, truth):
    binding = _assert_byte_and_context_binding(result, cad_blob, scan_blob, raw, triangles)
    from backend.cad_guided_context_measurement_v1 import validate_context_measurement_publication_v1
    measured = result["measurements"]
    validate_context_measurement_publication_v1(measured, raw_count=len(raw),
        scan_sha256=_sha(scan_blob), raw_points_sha256=_array_sha(raw, "<f8"))
    assert result["predictions"] == measured["predictions"]
    assert measured["scan_sha256"] == _sha(scan_blob)  # File SHA, not array-only label.
    assert measured["raw_points_sha256"] == _array_sha(raw, "<f8")
    native_features = {row["feature_id"]: row for row in result["native_features"]["features"]}
    indices = result["registration"]["evidence"]["measurement_context_indices"]
    windows = measured["context_windows"]
    assert len(windows) == len(indices)*len(native_features)
    assert {(row["context_index"], row["feature_id"]) for row in windows} == {
        (index, feature) for index in indices for feature in native_features}
    for record in measured["support_records"]:
        for association in record["associations"]:
            nominal = native_features[association["feature_id"]]
            assert association["nominal_source"].startswith("OCP_IMPORT_LOCAL_V1:")
            assert association["nominal_radius_mm"] == nominal["nominal_radius_mm"]
            assert association["nominal_sweep_deg"] == nominal["nominal_sweep_deg"]
    array = _fixture_module(ARRAY_SOURCE)
    answer = dict(registration=result["registration"], measurements=measured,
                  publication_validator="PASSED_ACTUAL_VALIDATOR")
    array._assert_safety(answer, "C2")
    oracle = array._assert_curved_output(answer, truth, "C2")
    return dict(binding=binding, all_emitted_measurements_and_carriers=oracle,
                inference_inputs="ONLY_STEP_PLY_BYTES_AND_FIXED_POLICIES",
                declaration_scope="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION")


def _encode(value, maximum):
    blocks, size = [], 0
    for piece in json.JSONEncoder(allow_nan=False, sort_keys=True, separators=(",", ":")).iterencode(value):
        block = piece.encode()
        size += len(block)
        if size > maximum:
            raise ValueError("NATIVE_BYTE_ARTIFACT_LIMIT")
        blocks.append(block)
    return b"".join(blocks)


def _write_bytes(directory, name, blob, maximum):
    assert name in ARTIFACT_NAMES
    assert type(blob) is bytes and 1 <= len(blob) <= maximum
    target, stage = directory/name, directory/(name+".partial")
    with stage.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(stage, target, follow_symlinks=False)  # EEXIST is never benign success.
    stage.unlink()  # Only successful publication removes its staging hardlink.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return dict(filename=name, bytes=len(blob), sha256=_sha(blob))


def _write(directory, name, value, maximum):
    return _write_bytes(directory, name, _encode(value, maximum), maximum)


def _error(error):
    return dict(type=type(error).__name__, message=str(error)[:1000])


def run_single_native_byte_case(artifact_directory):
    assert os.environ.get(OPT_IN) == "1", "EXPLICIT_NATIVE_BYTE_SUPERVISOR_OPT_IN_REQUIRED"
    directory = Path(artifact_directory)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    directory.mkdir(exist_ok=False)
    start = dict(schema="CONNECTED_V12_SINGLE_NATIVE_BYTE_CASE", case_id=CASE_ID, arm="C2",
        status="STARTED", source_sha256=SOURCE_HASHES, resource_limits=RESOURCE_LIMITS,
        artifact_limits=ARTIFACT_LIMITS, required_positive=True,
        incomplete_execution_stage="UNKNOWN_UNLESS_COMPLETE_RESULT_RETAINED")
    start_artifact = _write(directory, ARTIFACT_NAMES[0], start, ARTIFACT_LIMITS["start_json_bytes"])
    record = dict(start, status="FAILED_BEFORE_RETURN", phase="FIXTURE", workflow=None,
        execution_error=None, input_artifacts=[],
        oracle_disposition="NOT_RUN_RESULT_MUST_BE_DURABLE_FIRST",
        full_returned_workflow_retained=False, source_consistency=False)
    fixture = None
    try:
        for relative in SOURCE_HASHES: _source_bytes(relative)
        cad_blob, scan_blob, raw, triangles, truth = _make_fixture()
        fixture = cad_blob, scan_blob, raw, triangles, truth
        record["input_binding"] = dict(cad_sha256=_sha(cad_blob), scan_sha256=_sha(scan_blob),
            raw_xyz_sha256=_array_sha(raw, "<f8"), triangle_raw_rows_sha256=_array_sha(triangles, "<i8"),
            raw_rows=len(raw), triangle_rows=len(triangles), position_error_bound_mm=.05,
            cad_placement_scope="DECLARED_NOMINAL_CAD_FRAME_NOT_SCAN_POSE")
        record["input_artifacts"].append(_write_bytes(directory, "CAD.step", cad_blob, ARTIFACT_LIMITS["cad_bytes"]))
        record["input_artifacts"].append(_write_bytes(directory, "SCAN.ply", scan_blob, ARTIFACT_LIMITS["scan_bytes"]))
        record["phase"] = "ACTUAL_BYTE_WORKFLOW"
        record["workflow"] = _execute_workflow(cad_blob, scan_blob)
        record["full_returned_workflow_retained"] = True
        record["status"] = "RETURNED_PENDING_ORACLE"
        record["phase"] = "RETURNED_SOURCE_RECHECK"
        for relative in SOURCE_HASHES: _source_bytes(relative)
        record["source_consistency"] = True
        record["phase"] = "RETURNED_PENDING_ORACLE"
    except Exception as error:
        record["execution_error"] = _error(error)
        record["status"] = "EXECUTION_OR_SOURCE_FAILURE"
    try:
        artifact = _write(directory, "CASE_RESULT.json", record, ARTIFACT_LIMITS["result_json_bytes"])
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        # Encoding failure happens before opening any staging file. A bounded
        # whole-discard receipt is explicitly negative and NOT full retention.
        record = dict(schema=start["schema"], case_id=CASE_ID, arm="C2", status="FAIL",
            phase="RESULT_RETENTION_UNAVAILABLE", workflow=None, discarded_workflow_whole=True,
            full_returned_workflow_retained=False, source_consistency=False,
            input_artifacts=record.get("input_artifacts", []), publication_error=_error(error))
        artifact = _write(directory, "CASE_RESULT.json", record, 4096)
    total = start_artifact["bytes"]+artifact["bytes"]+sum(row["bytes"] for row in record["input_artifacts"])
    assert total <= ARTIFACT_LIMITS["total_bytes"]
    # Full workflow is now immutable/durable. An oracle exception/interruption
    # cannot remove it; missing compact completion remains a negative gate.
    oracle = oracle_error = None
    positive = False
    if record["full_returned_workflow_retained"] and record["source_consistency"] and record.get("execution_error") is None:
        try:
            oracle = _evaluate(record["workflow"], *fixture)
            positive = True
        except Exception as error:
            oracle_error = _error(error)
    return dict(schema=start["schema"], case_id=CASE_ID, status="PASS" if positive else "FAIL",
        phase="POST_INFERENCE_ORACLE_COMPLETE" if positive else record["phase"], positive_geometry_passed=positive,
        full_returned_workflow_retained=record["full_returned_workflow_retained"],
        source_consistency=record["source_consistency"], start_artifact=start_artifact, result_artifact=artifact,
        input_artifacts=record["input_artifacts"], total_artifact_bytes=total,
        execution_error=record.get("execution_error"), oracle_error=oracle_error, oracle=oracle,
        publication_error=record.get("publication_error"),
        evidence_class="ONE_INVENTED_NATIVE_BYTE_CASE_NOT_PHYSICAL_GENERALIZATION",
        observed_bend_count=None, production_use_allowed=False)


def test_source_pins_and_single_case_contract_do_not_invoke_native():
    assert tuple(SOURCE_HASHES) == (NATIVE_SOURCE, ARRAY_SOURCE)
    for relative in SOURCE_HASHES:
        assert isinstance(ast.parse(_source_bytes(relative)), ast.Module)
    assert CASE_ID == "L-R" and ARTIFACT_NAMES == ("CASE_START.json", "CAD.step", "SCAN.ply", "CASE_RESULT.json")
    assert ARTIFACT_LIMITS == dict(case_count=1, start_json_bytes=4096, cad_bytes=8*1024**2,
        scan_bytes=1024**2, result_json_bytes=16*1024**2, total_bytes=25*1024**2+4096)
    assert RESOURCE_LIMITS == dict(wall_seconds=300, worker_cpu_seconds=240,
        sampled_process_group_rss_bytes=4*1024**3, stdout_bytes=1024**2)


def test_source_native_and_scan_helpers_are_exact_references_not_alternative_geometry():
    tree = ast.parse(Path(__file__).read_text())
    make = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_make_fixture")
    text = ast.unparse(make)
    for name in ("native._asymmetric_native_strip()", "native._owned_step_bytes(shape)",
                 "native._ply_bytes(raw, triangles)", "array.curved_packet(case)",
                 "array.CAD_ROTATION", "array.CAD_TRANSLATION"):
        assert name in text
    assert "nominal_feature_kwargs" not in text
    assert "_analytic_mesh_scan" not in text


def test_source_inference_boundary_cannot_receive_simulator_truth_or_array_cad():
    tree = ast.parse(Path(__file__).read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_execute_workflow")
    assert [arg.arg for arg in execute.args.args] == ["cad_blob", "scan_blob"]
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "inspect_cad_guided_mesh_v12"]
    assert len(calls) == 1 and [keyword.arg for keyword in calls[0].keywords] == ["cad_bytes", "scan_bytes", None]
    assert not any(isinstance(node, ast.Name) and node.id in ("truth", "scanner_pose", "cad_triangles") for node in ast.walk(execute))


def test_literal_workflow_native_settings_are_not_original_easy_native_policy():
    assert WORKFLOW_FIELDS == dict(max_registration_scan_rows=50000, cad_sample_count=32768,
        cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
        native_adapter_version="v1", unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")
    tree = ast.parse(Path(__file__).read_text())
    policy = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_policy_objects")
    text = ast.unparse(policy)
    assert "array._policy_objects('C2', planar=False)" in text
    assert "kwargs['initialization_budgets'] = kwargs.pop('budgets')" in text
    assert "position_error_bound_mm=0.05" in text and "connected_measurement_contexts=True" in text


def test_opt_in_refuses_before_sources_fixture_or_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(OPT_IN, raising=False)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(AssertionError, match="EXPLICIT_NATIVE_BYTE_SUPERVISOR"):
        run_single_native_byte_case(path)
    assert not path.exists()


def test_preexisting_artifact_directory_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    path = tmp_path/"ARTIFACTS"
    path.mkdir()
    with pytest.raises(FileExistsError): run_single_native_byte_case(path)
    assert list(path.iterdir()) == []


def _mock_recorder(monkeypatch, *, failure=None):
    """Recorder-only seam; no source/native/inference qualification claim."""
    namespace = sys.modules[__name__]
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setattr(namespace, "_source_bytes", lambda name: b"MOCK_VERIFIED_SOURCE")
    monkeypatch.setattr(namespace, "_make_fixture", lambda: (
        b"invented-CAD", b"invented-PLY", np.zeros((3, 3)), np.array([[0, 1, 2]]), {"mock": True}))
    workflow = dict(status="UNAVAILABLE", phase="WHOLE_PART_REGISTRATION", all_rows=[1, 2, 3],
                    marker="COMPLETE_RETURNED_MOCK_NOT_ENGINE_POSITIVE")
    def execute(*args):
        if failure == "execute": raise RuntimeError("INVENTED_EXECUTION_FAILURE")
        return copy.deepcopy(workflow)
    def evaluate(*args):
        if failure == "oracle": raise AssertionError("INVENTED_REQUIRED_POSITIVE_FAILED")
        return {"mock_evaluation_only": True}
    monkeypatch.setattr(namespace, "_execute_workflow", execute)
    monkeypatch.setattr(namespace, "_evaluate", evaluate)
    return workflow


@pytest.mark.parametrize("failure", [None, "oracle", "execute"])
def test_recorder_retains_returned_workflow_and_separates_execution_from_positive(tmp_path, monkeypatch, failure):
    workflow = _mock_recorder(monkeypatch, failure=failure)
    receipt = run_single_native_byte_case(tmp_path/"ARTIFACTS")
    blob = (tmp_path/"ARTIFACTS"/"CASE_RESULT.json").read_bytes()
    retained = json.loads(blob)
    assert receipt["result_artifact"]["sha256"] == _sha(blob)
    assert receipt["positive_geometry_passed"] is (failure is None)
    assert receipt["full_returned_workflow_retained"] is (failure != "execute")
    assert retained["workflow"] == (None if failure == "execute" else workflow)
    assert retained["oracle_disposition"] == "NOT_RUN_RESULT_MUST_BE_DURABLE_FIRST"
    assert (receipt["oracle_error"] is not None) is (failure == "oracle")
    assert (retained["execution_error"] is not None) is (failure == "execute")
    assert receipt["status"] == ("PASS" if failure is None else "FAIL")
    assert (tmp_path/"ARTIFACTS"/"CAD.step").read_bytes() == b"invented-CAD"
    assert (tmp_path/"ARTIFACTS"/"SCAN.ply").read_bytes() == b"invented-PLY"


def test_late_source_drift_keeps_returned_workflow_but_cannot_pass(tmp_path, monkeypatch):
    workflow = _mock_recorder(monkeypatch)
    calls = 0
    def source(name):
        nonlocal calls
        calls += 1
        if calls == 3: raise AssertionError("FIXTURE_SOURCE_DRIFT")
        return b"MOCK"
    monkeypatch.setattr(sys.modules[__name__], "_source_bytes", source)
    receipt = run_single_native_byte_case(tmp_path/"ARTIFACTS")
    retained = json.loads((tmp_path/"ARTIFACTS"/"CASE_RESULT.json").read_bytes())
    assert retained["workflow"] == workflow and retained["source_consistency"] is False
    assert receipt["status"] != "PASS" and receipt["execution_error"] is not None


def test_full_result_and_exact_inputs_are_durable_before_oracle(tmp_path, monkeypatch):
    workflow = _mock_recorder(monkeypatch)
    directory = tmp_path/"ARTIFACTS"
    evaluated = []
    def evaluate(result, cad_blob, scan_blob, raw, triangles, truth):
        assert json.loads((directory/"CASE_RESULT.json").read_bytes())["workflow"] == workflow == result
        assert (directory/"CAD.step").read_bytes() == cad_blob
        assert (directory/"SCAN.ply").read_bytes() == scan_blob
        assert sorted(path.name for path in directory.iterdir()) == sorted(ARTIFACT_NAMES)
        evaluated.append(True)
        raise AssertionError("INVENTED_ORACLE_FAILURE_AFTER_DURABLE_RESULT")
    monkeypatch.setattr(sys.modules[__name__], "_evaluate", evaluate)
    receipt = run_single_native_byte_case(directory)
    assert evaluated == [True] and receipt["status"] == "FAIL"
    assert receipt["full_returned_workflow_retained"] is True
    assert receipt["oracle_error"]["message"] == "INVENTED_ORACLE_FAILURE_AFTER_DURABLE_RESULT"


def test_report_cap_whole_discards_and_never_reports_full_retention(tmp_path, monkeypatch):
    _mock_recorder(monkeypatch)
    monkeypatch.setitem(ARTIFACT_LIMITS, "result_json_bytes", 1)
    receipt = run_single_native_byte_case(tmp_path/"ARTIFACTS")
    retained = json.loads((tmp_path/"ARTIFACTS"/"CASE_RESULT.json").read_bytes())
    assert retained["discarded_workflow_whole"] is True and retained["workflow"] is None
    assert receipt["full_returned_workflow_retained"] is False
    assert receipt["positive_geometry_passed"] is False and receipt["status"] == "FAIL"
    assert not list((tmp_path/"ARTIFACTS").glob("*.partial"))


def test_exclusive_publication_keeps_existing_target_and_failed_stage(tmp_path):
    name = ARTIFACT_NAMES[0]
    receipt = _write(tmp_path, name, {"first": True}, 4096)
    original = (tmp_path/name).read_bytes()
    with pytest.raises(FileExistsError): _write(tmp_path, name, {"second": True}, 4096)
    assert (tmp_path/name).read_bytes() == original and receipt["sha256"] == _sha(original)
    assert json.loads((tmp_path/(name+".partial")).read_bytes()) == {"second": True}


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1",
                   reason="Separate exact PRE and bounded native worker required after array POST")
def test_connected_v12_single_native_byte_case():
    directory = os.environ.get(ARTIFACT_ENV)
    assert directory, "FRESH_ROOT_OWNED_ARTIFACT_DIRECTORY_REQUIRED"
    receipt = run_single_native_byte_case(directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    assert receipt["status"] == "PASS", receipt
    assert receipt["positive_geometry_passed"] is True
    assert receipt["full_returned_workflow_retained"] is True and receipt["source_consistency"] is True
