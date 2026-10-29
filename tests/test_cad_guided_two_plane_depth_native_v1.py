"""Inactive native depth-scheduler V2 successor on the unchanged L-R fixture.

The old composite/native files remain unchanged. Only explicit scheduler V2,
its independently requested configuration/binding checks, new refusal controls
and this packet identity are added. Generated STEP/PLY geometry, original raw,
native, context, measurement and endpoint oracles are preserved. Native timing
is frozen at2520CPU/2815work/2820parent; the exact amended diagnostic PRE
is still required before dispatch. Full returned results
remain durable before oracle evaluation. No physical inputs or scoring wrapper.
"""

import ast
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCE = "tests/test_cad_guided_connected_v12_native_bytes.py"
NATIVE_SOURCE = "tests/test_cad_guided_mesh_connected_native_v1.py"
ARRAY_SOURCE = "tests/test_cad_guided_connected_v12.py"
SOURCE_HASHES = {
    LEGACY_SOURCE: "19c8a2b7c96c4629760e060efb09d180b84f9a6099760738aca10b2022207c75",
    NATIVE_SOURCE: "11455ceda7283bcd1b84dbd037b7bd16bd2517bd8d44985ef8f774d2e5c100e8",
    ARRAY_SOURCE: "fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd",
}
SOURCE_DEPENDENCIES = (*SOURCE_HASHES, "backend/cad_guided_mesh_offline_v13.py",
    "backend/cad_guided_mesh_offline_v12.py", "backend/cad_guided_support_cover_measurement_v2.py",
    "backend/cad_guided_exact_support_cover_v1.py", "backend/cad_guided_seeded_plane_proposals_v1.py",
    "backend/cad_guided_scan_plane_preparation_v1.py", "backend/cad_guided_composite_surface_roles_v1.py",
    "backend/cad_guided_noisy_mesh_planes_v7.py", "backend/cad_guided_two_plane_schedule_v2.py")
OPT_IN = "SHERMAN_RUN_TWO_PLANE_DEPTH_NATIVE_V1"
ARTIFACT_ENV = "SHERMAN_TWO_PLANE_DEPTH_NATIVE_V1_ARTIFACT_DIRECTORY"
CASE_ID = "L-R"
SOURCE_RESOURCE_PROPOSAL = dict(wall_seconds=None, work_wall_seconds=None, worker_cpu_seconds=None,
    sampled_process_group_rss_bytes=1024**3, stdout_bytes=1024**2)
RESOURCE_LIMITS = dict(wall_seconds=2820, work_wall_seconds=2815, worker_cpu_seconds=2520,
    sampled_process_group_rss_bytes=4*1024**3, stdout_bytes=1024**2)
RESOURCE_LIMIT_STATUS = "FROZEN_NATIVE_DIAGNOSTIC_PENDING_EXACT_PRE"
ARTIFACT_LIMITS = dict(case_count=1, start_json_bytes=4096, cad_bytes=8*1024**2,
    scan_bytes=1024**2, result_json_bytes=129*1024**2, metadata_json_bytes=1024**2,
    total_bytes=138*1024**2+4096)
ARTIFACT_NAMES = ("CASE_START.json", "CAD.step", "SCAN.ply", "CASE_RESULT.json")
WORKFLOW_FIELDS = dict(max_registration_scan_rows=50000, cad_sample_count=32768,
    cad_linear_deflection_mm=.1, cad_angular_deflection_radians=.2,
    native_adapter_version="v1", unit_assumption="PROVISIONAL_COMMON_MM_NOT_CALIBRATED")


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _array_sha(value, dtype):
    return _sha(np.asarray(value, dtype=dtype).tobytes())


def _source_bytes(relative):
    assert relative in SOURCE_HASHES, "UNDECLARED_DEPTH_NATIVE_FIXTURE_SOURCE"
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


@lru_cache(maxsize=3)
def _fixture_module(relative):
    blob = _source_bytes(relative)
    name = "_two_plane_depth_native_v1_fixture_"+Path(relative).stem
    module = types.ModuleType(name)
    module.__file__ = str(ROOT/relative)
    sys.modules[name] = module
    try:
        exec(compile(blob, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _legacy():
    return _fixture_module(LEGACY_SOURCE)


def _make_fixture():
    # Exact old single L-R construction, including real connected cap and
    # independently placed native CAD. Never call its old workflow/evaluator.
    return _legacy()._make_fixture()


def _policy_objects():
    from backend.cad_guided_mesh_offline_v13 import MeshWorkflowPolicyV13
    from backend.cad_guided_seeded_plane_proposals_v1 import SeededPlaneBudgetsV1
    from backend.cad_guided_scan_plane_preparation_v1 import CompositeScanPlaneBudgetsV1
    from backend.cad_guided_two_plane_schedule_v2 import TwoPlaneScheduleBudgetsV2
    kwargs = dict(_legacy()._policy_objects())
    kwargs["workflow_policy"] = MeshWorkflowPolicyV13(**WORKFLOW_FIELDS)
    kwargs.update(scan_plane_mode="composite_v1", seeded_plane_budgets=SeededPlaneBudgetsV1(),
                  composite_plane_budgets=CompositeScanPlaneBudgetsV1(),
                  two_plane_schedule_mode="v2", two_plane_schedule_budgets=TwoPlaneScheduleBudgetsV2())
    assert kwargs["connected_measurement_contexts"] is True
    return kwargs


def _execute_workflow(cad_blob, scan_blob):
    """Only generated bytes and fixed policies enter actual inference."""
    from backend.cad_guided_mesh_offline_v13 import inspect_cad_guided_mesh_v13
    return inspect_cad_guided_mesh_v13(cad_bytes=cad_blob, scan_bytes=scan_blob, **_policy_objects())


def _assert_byte_and_context_binding(result, cad_blob, scan_blob, raw, triangles):
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V13"
    assert result["connected_measurement_contexts"] is True
    assert result["position_error_model"]["scope"] == (
        "PLANE_INITIALIZATION_AND_CONTEXT_SURFACE_ROLE_POLICY_NOT_MEASUREMENT_UNCERTAINTY")
    # Oracle-only schema projection invokes the unchanged historical byte,
    # topology/native/sampler/context binding checks. The actual returned V13
    # result is neither modified nor passed through the old V12 workflow.
    historical_view = dict(result, schema="CAD_GUIDED_MESH_OFFLINE_V12")
    previous = _legacy()._assert_byte_and_context_binding(historical_view, cad_blob, scan_blob, raw, triangles)
    return dict(previous, composite_scan_planes=_assert_composite_binding(result, raw, triangles),
                two_plane_schedule=_assert_schedule_binding(result))


def _assert_schedule_binding(result):
    """Use the same qualified equation/match binder as the connected packet."""
    from backend.cad_guided_mesh_offline_v12 import _two_plane_schedule_binding
    kwargs = _policy_objects()  # Fresh source-requested values, not returned claims.
    expected = dict(mode=kwargs["two_plane_schedule_mode"],
                    budgets=asdict(kwargs["two_plane_schedule_budgets"]))
    initialization = asdict(kwargs["initialization_budgets"])
    bound = _two_plane_schedule_binding(result["registration"], expected)
    return _assert_schedule_metadata(result, expected, initialization, bound)


def _assert_schedule_metadata(result, expected, initialization_budgets, bound):
    """Report-copy oracle; owned-support proof remains in the original checks."""
    evidence = result["registration"]["evidence"]
    initialization = evidence["initialization"]
    assert expected["mode"] == "v2"
    for parent in (result["policies"], evidence, initialization):
        assert _exact_json(parent["two_plane_schedule_configuration"], expected)
    assert result["policies"]["two_plane_enabled"] is True
    assert _exact_json(result["policies"]["initialization_budgets"], initialization_budgets)
    assert _exact_json(initialization["budgets"], initialization_budgets)
    schedule = initialization["two_plane_schedule"]
    assert schedule["schema_version"] == "two_plane_schedule_v2"
    assert schedule["status"] == "COMPLETE" and schedule["selected_policy_evaluation_complete"] is True
    assert _exact_json(schedule["budgets"], expected["budgets"])
    keys = {"two_plane_schedule_mode", "two_plane_schedule_configuration_sha256",
            "two_plane_schedule_input_digests"}
    assert type(bound) is dict and set(bound) == keys and bound["two_plane_schedule_mode"] == "v2"
    assert bound["two_plane_schedule_configuration_sha256"] == _sha(_encode(expected, 65536))
    assert _exact_json(bound["two_plane_schedule_input_digests"], schedule["input_digests"])
    indices, rows = evidence["measurement_context_indices"], result["registration_array_binding"]
    assert type(indices) is list and 1 <= len(indices) <= 48
    assert all(type(index) is int and 0 <= index < 48 for index in indices)
    assert len(set(indices)) == len(indices)
    assert type(rows) is list and len(rows) == len(indices)
    for row in rows:
        assert type(row) is dict
        actual = {key for key in row if key.startswith("two_plane_schedule")}
        assert actual == keys
        assert _exact_json({key: row[key] for key in keys}, bound)
    return dict(mode="v2", configuration_sha256=bound["two_plane_schedule_configuration_sha256"],
        input_digests=bound["two_plane_schedule_input_digests"], checked_binding_rows=len(rows),
        binding_scope="REPORTED_COPY_CONSISTENCY_PLUS_QUALIFIED_RUNTIME_EQUATION_MATCH_BINDER",
        per_context_identity_proved_by_schedule_binding=False, raw_support_reconstructed=False)


def _assert_composite_binding(result, raw, triangles):
    """Additional report binding, not a claim that hashes reconstruct raw rows."""
    from backend.cad_guided_noisy_mesh_planes_v7 import decode_metadata_v7
    kwargs = _policy_objects()
    expected = dict(mode=kwargs["scan_plane_mode"], seeded_budgets=asdict(kwargs["seeded_plane_budgets"]),
                    composite_budgets=asdict(kwargs["composite_plane_budgets"]))
    envelope = result["registration"]["evidence"]["initialization"]["patch_evidence"]["scan"]
    # The production metadata digest uses insertion order, unlike artifact JSON.
    digest, size = hashlib.sha256(), 0
    for fragment in json.JSONEncoder(allow_nan=False, ensure_ascii=True, separators=(",", ":")).iterencode(envelope):
        block = fragment.encode()
        size += len(block)
        assert size <= expected["composite_budgets"]["max_metadata_bytes"]
        digest.update(block)
    meta = decode_metadata_v7(envelope)
    return _assert_composite_metadata(result, meta, digest.hexdigest(), expected, raw, triangles)


def _assert_composite_metadata(result, meta, digest, expected, raw, triangles):
    """Pure oracle seam; mocked records below qualify binding checks only."""
    evidence = result["registration"]["evidence"]
    initialization = evidence["initialization"]
    for container in (result["policies"], evidence, initialization):
        assert _exact_json(container["scan_plane_configuration"], expected)
    binding = dict(points_sha256=_array_sha(raw, "<f8"),
        triangle_raw_rows_sha256=_array_sha(triangles, "<i8"),
        position_error_bounds_mm_sha256=_array_sha(np.full(len(raw), .05), "<f8"))
    assert meta["schema_version"] == "composite_scan_planes_v1" and meta["status"] == "COMPLETE"
    assert _exact_json(meta["input_digests"], binding)
    assert _exact_json(meta["requested_lane_budgets"]["seeded_v1"], expected["seeded_budgets"])
    assert _exact_json(meta["budgets"], expected["composite_budgets"])
    assert tuple(meta["lanes"]) == ("v7", "seeded_v1")
    assert meta["lane_execution"] == dict(v7="COMPLETE", seeded_v1="COMPLETE")
    origins = []
    for lane, schema in (("v7", "noisy_mesh_planes_v7"), ("seeded_v1", "seeded_plane_proposals_v1")):
        source = meta["lanes"][lane]
        assert source["schema_version"] == schema and source["status"] == "COMPLETE"
        assert _exact_json(source["input_digests"], binding)
        for ref in source["proposals"]:
            kind, identifier = ref["kind"], ref["record_id"]
            assert type(identifier) is int and identifier >= 0
            assert kind in (("CORE", "CLOSURE") if lane == "v7" else ("SEEDED_REGION",))
            table = source["cores" if kind == "CORE" else "closures"] if lane == "v7" else source["regions"]
            record = table[identifier]
            assert record["analysis"]["status"] == "PROPOSED"
            origins.append(dict(plane_id=len(origins), lane=lane, kind=kind, record_id=identifier,
                triangle_rows_sha256=record["triangle_rows_sha256"],
                raw_vertex_rows_sha256=record["raw_vertex_rows_sha256"]))
    assert _exact_json(meta["planes"], origins)
    assert type(meta["statistics"]["qualified_plane_count"]) is int
    assert meta["statistics"]["qualified_plane_count"] == len(origins)
    assert meta["statistics"]["selected_policy_evaluation_complete"] is True
    # Original pair/triplet hypotheses retain their combined-ID support source;
    # neither proposal lane is relabelled as a physical plane count.
    for hypothesis in initialization["hypotheses"]:
        for support in hypothesis.get("assigned_support", []):
            identifier = support["scan_plane_id"]
            assert type(identifier) is int and 0 <= identifier < len(origins)
            origin = origins[identifier]
            assert type(support["source_record_id"]) is int and support["source_record_id"] == identifier
            assert support["proposal_kind"] == origin["lane"]+":"+origin["kind"]
            assert support["scan_triangle_rows_sha256"] == origin["triangle_rows_sha256"]
            assert support["raw_vertex_rows_sha256"] == origin["raw_vertex_rows_sha256"]
    role = evidence["surface_role_preparation"]
    assert all(role[key] == value for key, value in binding.items())
    assert role["metadata_sha256"] == digest
    assert type(role["raw_row_count"]) is int and role["raw_row_count"] == len(raw)
    for row in result["registration_array_binding"]:
        assert row["scan_plane_mode"] == "composite_v1" and row["scan_plane_metadata_sha256"] == digest
    measured_binding = result["measurements"]["source_context_result"]["surface_role_source_binding"]
    assert _exact_json(measured_binding, dict(binding, encoded_patch_metadata_sha256=digest))
    return dict(mode="composite_v1", metadata_sha256=digest, qualified_proposal_count=len(origins),
                lane_statuses={name: value["status"] for name, value in meta["lanes"].items()},
                physical_plane_count=None, observed_bend_count=None)


def _encode(value, maximum):
    blocks, size = [], 0
    for fragment in json.JSONEncoder(allow_nan=False, sort_keys=True, separators=(",", ":")).iterencode(value):
        block = fragment.encode()
        size += len(block)
        if size > maximum:
            raise ValueError("DEPTH_NATIVE_ARTIFACT_LIMIT")
        blocks.append(block)
    return b"".join(blocks)


def _exact_json(left, right):
    return _encode(left, ARTIFACT_LIMITS["result_json_bytes"]) == _encode(right, ARTIFACT_LIMITS["result_json_bytes"])


def _assert_top_level_cover_projection(result):
    measured = result["measurements"]
    assert measured["schema"] == "CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2"
    assert measured["source_context_result_role"] == "RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS"
    assert result["prediction_collection_status"] == "CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS"
    expected = [row["finite_support"]["instance"] for row in measured["support_groups"]
        if row["status"] == "MEASURED_CONDITIONAL" and row["finite_support"]["status"] == "SUPPORTED"]
    assert _exact_json(result["predictions"], measured["predictions"])
    assert _exact_json(result["predictions"], expected)
    return dict(cover_prediction_count=len(expected), nested_member_prediction_collection_used=False)


def _evaluate(result, cad_blob, scan_blob, raw, triangles, truth):
    binding = _assert_byte_and_context_binding(result, cad_blob, scan_blob, raw, triangles)
    from backend.cad_guided_support_cover_measurement_v2 import validate_support_cover_publication_v2
    measured = result["measurements"]
    validation = validate_support_cover_publication_v2(measured, raw_points=raw, scan_sha256=_sha(scan_blob))
    assert _exact_json(result["support_cover_publication_validation"], validation)
    projection = _assert_top_level_cover_projection(result)
    assert measured["scan_sha256"] == _sha(scan_blob)
    assert measured["raw_points_sha256"] == _array_sha(raw, "<f8")
    assert measured["observed_bend_count"] is None and measured["production_use_allowed"] is False
    assert measured["conformance"] == "INDETERMINATE"
    assert measured["joint_configuration_status"] == "NOT_ESTABLISHED"
    assert measured["continuous_pose_family_coverage"] == "NOT_ESTABLISHED"
    assert measured["context_quantifier"] == "EXISTS_COMPLETED_EXISTING_COVER_CONTEXT"
    native_features = {row["feature_id"]: row for row in result["native_features"]["features"]}
    indices = result["registration"]["evidence"]["measurement_context_indices"]
    original = measured["source_context_result"]
    windows = original["context_windows"]
    assert len(windows) == len(indices)*len(native_features)
    assert {(row["context_index"], row["feature_id"]) for row in windows} == {
        (index, feature) for index in indices for feature in native_features}
    for record in original["support_records"]:
        for association in record["associations"]:
            nominal = native_features[association["feature_id"]]
            assert association["nominal_source"].startswith("OCP_IMPORT_LOCAL_V1:")
            assert association["nominal_radius_mm"] == nominal["nominal_radius_mm"]
            assert association["nominal_sweep_deg"] == nominal["nominal_sweep_deg"]
    array = _fixture_module(ARRAY_SOURCE)
    groups = [row for row in measured["support_groups"] if row["status"] == "MEASURED_CONDITIONAL"]
    assert groups, measured.get("reasons")
    errors = [array.assert_scan_derived_measurement(row["measurements"], truth) for row in groups]
    for group in groups:
        value = group["measurements"]
        assert value["geometry_frame"] == "ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM"
        assert value["radius_uncertainty_interval_mm"] is None and value["sweep_uncertainty_interval_deg"] is None
        assert value["datum_relative_deviation"] is None
        for association in group["associations"]:
            nominal = native_features[association["feature_id"]]
            assert association["nominal_radius_mm"] == nominal["nominal_radius_mm"]
            assert association["nominal_sweep_deg"] == nominal["nominal_sweep_deg"]
            assert association["accepted_registration"] is False
            assert abs(association["canonical_radius_deviation_mm"]
                -(value["measured_radius_mm"]-nominal["nominal_radius_mm"])) <= 1e-12
            assert abs(association["canonical_sweep_magnitude_deviation_deg"]
                -(value["measured_sweep_magnitude_deg"]-abs(nominal["nominal_sweep_deg"]))) <= 1e-12
    assert result["predictions"], "MISSING_REQUIRED_LOCALIZED_RAW_CARRIER"
    carrier_errors = [array.assert_localized_raw_carrier(row, truth) for row in result["predictions"]]
    return dict(binding=binding, projection=projection, measurement_errors=errors, carrier_errors=carrier_errors,
        inference_inputs="ONLY_STEP_PLY_BYTES_AND_FIXED_POLICIES",
        declaration_scope="STIPULATED_UNCALIBRATED_NOT_SCANNER_CALIBRATION",
        manufacturing_count_claim=False)


def _write_bytes(directory, name, blob, maximum):
    assert name in ARTIFACT_NAMES
    assert type(blob) is bytes and 1 <= len(blob) <= maximum
    target, stage = directory/name, directory/(name+".partial")
    with stage.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(stage, target, follow_symlinks=False)
    stage.unlink()
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


def _require_execution_limits():
    wall, work, cpu = (RESOURCE_LIMITS[key] for key in
        ("wall_seconds", "work_wall_seconds", "worker_cpu_seconds"))
    if not (all(type(value) is int and value > 0 for value in (wall, work, cpu))
            and cpu <= work <= wall-5):
        raise ValueError("NATIVE_LIMITS_PENDING_CONNECTED_TIMING_AND_SEPARATE_EXACT_PRE")


def _check_result_metadata(record):
    # The actual V13 result is independently capped at128MiB. Replacing the
    # literal null in this <=1MiB envelope gives <=129MiB minus four bytes.
    # Do not encode or alter the returned workflow during this metadata check.
    return len(_encode(dict(record, workflow=None), ARTIFACT_LIMITS["metadata_json_bytes"]))


def _diagnostic_evaluate(record, artifact, *fixture):
    """Supplemental integrity evidence; never changes the original accuracy oracle.

    This runs only after the original full RESULT is durable. The separate
    helper receives existing returned values, not permission to rerun inference.
    Source qualification and the exact native PRE remain prerequisites.
    """
    from tests.native_depth_diagnostic_ledger_v1 import evaluate_native_diagnostic_v1
    return evaluate_native_diagnostic_v1(sys.modules[__name__], record, artifact, *fixture)


def run_single_native_byte_case(artifact_directory):
    assert os.environ.get(OPT_IN) == "1", "EXPLICIT_DEPTH_NATIVE_SUPERVISOR_OPT_IN_REQUIRED"
    _require_execution_limits()
    directory = Path(artifact_directory)
    assert directory.is_absolute() and directory.name == "ARTIFACTS"
    directory.mkdir(exist_ok=False)
    start = dict(schema="TWO_PLANE_DEPTH_V1_SINGLE_NATIVE_BYTE_CASE", case_id=CASE_ID, arm="DEPTH_V2_COMPOSITE_COVER_V2",
        status="STARTED", source_sha256=SOURCE_HASHES, resource_limits=RESOURCE_LIMITS,
        resource_limit_status=RESOURCE_LIMIT_STATUS, artifact_limits=ARTIFACT_LIMITS, required_positive=True,
        incomplete_execution_stage="UNKNOWN_UNLESS_COMPLETE_RESULT_RETAINED")
    start_artifact = _write(directory, "CASE_START.json", start, ARTIFACT_LIMITS["start_json_bytes"])
    record = dict(start, status="FAILED_BEFORE_RETURN", phase="FIXTURE", workflow=None,
        execution_error=None, input_artifacts=[], timings={},
        oracle_disposition="NOT_RUN_RESULT_MUST_BE_DURABLE_FIRST",
        full_returned_workflow_retained=False, source_consistency=False)
    fixture = None
    try:
        for relative in SOURCE_HASHES:
            _source_bytes(relative)
        wall, cpu = time.monotonic_ns(), time.process_time_ns()
        try:
            cad_blob, scan_blob, raw, triangles, truth = _make_fixture()
            fixture = cad_blob, scan_blob, raw, triangles, truth
        finally:
            record["timings"]["fixture"] = dict(wall_seconds=(time.monotonic_ns()-wall)/1e9,
                                                 cpu_seconds=(time.process_time_ns()-cpu)/1e9)
        record["input_binding"] = dict(cad_sha256=_sha(cad_blob), scan_sha256=_sha(scan_blob),
            raw_xyz_sha256=_array_sha(raw, "<f8"), triangle_raw_rows_sha256=_array_sha(triangles, "<i8"),
            raw_rows=len(raw), triangle_rows=len(triangles), position_error_bound_mm=.05,
            cad_placement_scope="DECLARED_NOMINAL_CAD_FRAME_NOT_SCAN_POSE")
        record["input_artifacts"].append(_write_bytes(directory, "CAD.step", cad_blob, ARTIFACT_LIMITS["cad_bytes"]))
        record["input_artifacts"].append(_write_bytes(directory, "SCAN.ply", scan_blob, ARTIFACT_LIMITS["scan_bytes"]))
        record["phase"] = "ACTUAL_BYTE_WORKFLOW"
        wall, cpu = time.monotonic_ns(), time.process_time_ns()
        try:
            record["workflow"] = _execute_workflow(cad_blob, scan_blob)
        finally:
            record["timings"]["workflow"] = dict(wall_seconds=(time.monotonic_ns()-wall)/1e9,
                                                  cpu_seconds=(time.process_time_ns()-cpu)/1e9)
        record["full_returned_workflow_retained"] = True
        record["status"], record["phase"] = "RETURNED_PENDING_ORACLE", "RETURNED_SOURCE_RECHECK"
        for relative in SOURCE_HASHES:
            _source_bytes(relative)
        record["source_consistency"] = True
        record["phase"] = "RETURNED_PENDING_ORACLE"
    except Exception as error:
        record["execution_error"] = _error(error)
        record["status"] = "EXECUTION_OR_SOURCE_FAILURE"
    try:
        _check_result_metadata(record)
        artifact = _write(directory, "CASE_RESULT.json", record, ARTIFACT_LIMITS["result_json_bytes"])
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        record = dict(schema=start["schema"], case_id=CASE_ID, arm="DEPTH_V2_COMPOSITE_COVER_V2", status="FAIL",
            phase="RESULT_RETENTION_UNAVAILABLE", workflow=None, discarded_workflow_whole=True,
            full_returned_workflow_retained=False, source_consistency=False, timings=record["timings"],
            input_artifacts=record.get("input_artifacts", []), publication_error=_error(error))
        artifact = _write(directory, "CASE_RESULT.json", record, 4096)
    total = start_artifact["bytes"]+artifact["bytes"]+sum(row["bytes"] for row in record["input_artifacts"])
    assert total <= ARTIFACT_LIMITS["total_bytes"]
    oracle = oracle_error = diagnostic_ledger = diagnostic_error = None
    positive = False
    if record["full_returned_workflow_retained"] and record["source_consistency"] and record.get("execution_error") is None:
        # Original accuracy runs before the supplemental helper can touch the
        # supplied objects. Its ordinary failure never skips the later ledger.
        try:
            oracle = _evaluate(record["workflow"], *fixture)
            positive = True
        except Exception as error:
            oracle_error = _error(error)
        try:
            diagnostic_ledger = _diagnostic_evaluate(record, artifact, *fixture)
        except Exception as error:
            diagnostic_error = _error(error)
    return dict(schema=start["schema"], case_id=CASE_ID, status="PASS" if positive else "FAIL",
        positive_geometry_passed=positive, full_returned_workflow_retained=record["full_returned_workflow_retained"],
        source_consistency=record["source_consistency"], start_artifact=start_artifact, result_artifact=artifact,
        input_artifacts=record["input_artifacts"], total_artifact_bytes=total, timings=record["timings"],
        execution_error=record.get("execution_error"), oracle_error=oracle_error, oracle=oracle,
        diagnostic_ledger=diagnostic_ledger, diagnostic_error=diagnostic_error,
        publication_error=record.get("publication_error"),
        evidence_class="ONE_INVENTED_NATIVE_BYTE_CASE_NOT_PHYSICAL_GENERALIZATION",
        observed_bend_count=None, production_use_allowed=False)


@pytest.fixture
def runtime(monkeypatch):
    """Actual pure adapter imports under an inert package; never native calls."""
    saved = {key: value for key, value in sys.modules.items() if key == "backend" or key.startswith("backend.")}
    for key in saved:
        monkeypatch.delitem(sys.modules, key)
    package = types.ModuleType("backend")
    package.__path__ = [str(ROOT/"backend")]
    monkeypatch.setitem(sys.modules, "backend", package)
    try:
        yield importlib.import_module("backend.cad_guided_mesh_offline_v13")
    finally:
        for key in tuple(sys.modules):
            if key == "backend" or key.startswith("backend."):
                sys.modules.pop(key, None)
        sys.modules.update(saved)


def test_source_pins_and_prospective_caps_do_not_invoke_native():
    for relative in SOURCE_HASHES:
        assert isinstance(ast.parse(_source_bytes(relative)), ast.Module)
    assert tuple(SOURCE_HASHES) == (LEGACY_SOURCE, NATIVE_SOURCE, ARRAY_SOURCE)
    assert ARTIFACT_NAMES == ("CASE_START.json", "CAD.step", "SCAN.ply", "CASE_RESULT.json")
    assert ARTIFACT_LIMITS["total_bytes"] == 138*1024**2+4096
    assert ARTIFACT_LIMITS["result_json_bytes"] == 129*1024**2
    assert ARTIFACT_LIMITS["metadata_json_bytes"] == 1024**2
    assert RESOURCE_LIMITS == dict(wall_seconds=2820, work_wall_seconds=2815, worker_cpu_seconds=2520,
        sampled_process_group_rss_bytes=4*1024**3, stdout_bytes=1024**2)
    assert RESOURCE_LIMIT_STATUS == "FROZEN_NATIVE_DIAGNOSTIC_PENDING_EXACT_PRE"


def test_original_fixture_is_delegated_without_geometry_replacement():
    tree = ast.parse(Path(__file__).read_text())
    make = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_make_fixture")
    assert ast.unparse(make.body[-1]) == "return _legacy()._make_fixture()"
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_execute_workflow")
    assert [arg.arg for arg in execute.args.args] == ["cad_blob", "scan_blob"]
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "inspect_cad_guided_mesh_v13"]
    assert len(calls) == 1 and [keyword.arg for keyword in calls[0].keywords] == ["cad_bytes", "scan_bytes", None]


def test_fixed_policy_adds_only_explicit_depth_to_original_composite_configuration(runtime):
    from backend.cad_guided_seeded_plane_proposals_v1 import SeededPlaneBudgetsV1
    from backend.cad_guided_scan_plane_preparation_v1 import CompositeScanPlaneBudgetsV1
    from backend.cad_guided_two_plane_schedule_v2 import TwoPlaneScheduleBudgetsV2
    old = _legacy()._policy_objects()
    new = _policy_objects()
    assert set(new) == set(old) | {"scan_plane_mode", "seeded_plane_budgets", "composite_plane_budgets",
                                   "two_plane_schedule_mode", "two_plane_schedule_budgets"}
    assert type(new["workflow_policy"]) is runtime.MeshWorkflowPolicyV13
    assert type(old["workflow_policy"]) is not type(new["workflow_policy"])
    assert asdict(new["workflow_policy"]) == asdict(old["workflow_policy"]) == WORKFLOW_FIELDS
    for key in old.keys()-{"workflow_policy"}:
        assert new[key] == old[key]
    assert new["scan_plane_mode"] == "composite_v1"
    assert type(new["seeded_plane_budgets"]) is SeededPlaneBudgetsV1
    assert type(new["composite_plane_budgets"]) is CompositeScanPlaneBudgetsV1
    assert asdict(new["seeded_plane_budgets"]) == asdict(SeededPlaneBudgetsV1())
    assert asdict(new["composite_plane_budgets"]) == asdict(CompositeScanPlaneBudgetsV1())
    assert new["two_plane_schedule_mode"] == "v2"
    assert type(new["two_plane_schedule_budgets"]) is TwoPlaneScheduleBudgetsV2
    assert asdict(new["two_plane_schedule_budgets"]) == asdict(TwoPlaneScheduleBudgetsV2())


def _mock_composite_binding():
    """Tiny structural report only: no extraction, fitting or native inference."""
    raw = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    binding = dict(points_sha256=_array_sha(raw, "<f8"),
        triangle_raw_rows_sha256=_array_sha(triangles, "<i8"),
        position_error_bounds_mm_sha256=_array_sha(np.full(3, .05), "<f8"))
    expected = dict(mode="composite_v1", seeded_budgets={"mock_budget": 1},
                    composite_budgets={"mock_budget": 2})
    record = dict(analysis={"status": "PROPOSED"}, triangle_rows_sha256="a"*64,
                  raw_vertex_rows_sha256="b"*64)
    lanes = dict(v7=dict(schema_version="noisy_mesh_planes_v7", status="COMPLETE", input_digests=binding,
        proposals=[dict(kind="CORE", record_id=0)], cores=[record], closures=[]),
        seeded_v1=dict(schema_version="seeded_plane_proposals_v1", status="COMPLETE", input_digests=binding,
        proposals=[dict(kind="SEEDED_REGION", record_id=0)], regions=[copy.deepcopy(record)]))
    origins = [dict(plane_id=i, lane=lane, kind=kind, record_id=0,
        triangle_rows_sha256="a"*64, raw_vertex_rows_sha256="b"*64)
        for i, (lane, kind) in enumerate((("v7", "CORE"), ("seeded_v1", "SEEDED_REGION")))]
    meta = dict(schema_version="composite_scan_planes_v1", status="COMPLETE", input_digests=binding,
        requested_lane_budgets=dict(seeded_v1=copy.deepcopy(expected["seeded_budgets"])),
        budgets=copy.deepcopy(expected["composite_budgets"]), lanes=lanes, planes=origins,
        lane_execution=dict(v7="COMPLETE", seeded_v1="COMPLETE"),
        statistics=dict(qualified_plane_count=2, selected_policy_evaluation_complete=True))
    digest = "c"*64
    initialization = dict(scan_plane_configuration=copy.deepcopy(expected), hypotheses=[dict(assigned_support=[
        dict(scan_plane_id=i, source_record_id=i, proposal_kind=origin["lane"]+":"+origin["kind"],
            scan_triangle_rows_sha256="a"*64, raw_vertex_rows_sha256="b"*64)
        for i, origin in enumerate(origins)])])
    result = dict(policies=dict(scan_plane_configuration=copy.deepcopy(expected)),
        registration=dict(evidence=dict(scan_plane_configuration=copy.deepcopy(expected), initialization=initialization,
            surface_role_preparation=dict(binding, metadata_sha256=digest, raw_row_count=3))),
        registration_array_binding=[dict(scan_plane_mode="composite_v1", scan_plane_metadata_sha256=digest)],
        measurements=dict(source_context_result=dict(surface_role_source_binding=
            dict(binding, encoded_patch_metadata_sha256=digest))))
    return result, meta, digest, expected, raw, triangles


@pytest.mark.parametrize("fault", [None, "missing_configuration", "boolean_budget", "incomplete_lane",
    "wrong_origin", "stale_context", "stale_role", "stale_measurement"])
def test_composite_binding_oracle_rejects_structural_faults_without_geometry(fault):
    result, meta, digest, expected, raw, triangles = _mock_composite_binding()
    evidence = result["registration"]["evidence"]
    if fault == "missing_configuration":
        del evidence["scan_plane_configuration"]
    elif fault == "boolean_budget":
        evidence["initialization"]["scan_plane_configuration"]["seeded_budgets"]["mock_budget"] = True
    elif fault == "incomplete_lane":
        meta["lanes"]["seeded_v1"]["status"] = "UNAVAILABLE"
    elif fault == "wrong_origin":
        evidence["initialization"]["hypotheses"][0]["assigned_support"][1]["raw_vertex_rows_sha256"] = "d"*64
    elif fault == "stale_context":
        result["registration_array_binding"][0]["scan_plane_metadata_sha256"] = "d"*64
    elif fault == "stale_role":
        evidence["surface_role_preparation"]["metadata_sha256"] = "d"*64
    elif fault == "stale_measurement":
        result["measurements"]["source_context_result"]["surface_role_source_binding"]["encoded_patch_metadata_sha256"] = "d"*64
    if fault is None:
        answer = _assert_composite_metadata(result, meta, digest, expected, raw, triangles)
        assert answer["qualified_proposal_count"] == 2 and answer["observed_bend_count"] is None
    else:
        with pytest.raises((AssertionError, KeyError)):
            _assert_composite_metadata(result, meta, digest, expected, raw, triangles)


@pytest.mark.parametrize("connected", [False, None, 1, "true"])
def test_real_adapter_rejects_non_true_connected_before_native(runtime, connected):
    result = runtime.inspect_cad_guided_mesh_v13(scan_bytes=b"unused", cad_bytes=b"unused",
        position_error_bound_mm=.05, workflow_policy=None, registration_policy=None,
        measurement_policy=None, instance_policy=None, allow_ocp=False,
        connected_measurement_contexts=connected)
    assert result["schema"] == "CAD_GUIDED_MESH_OFFLINE_V13"
    assert result["phase"] == "ADMISSION" and result["reasons"] == ["V13_CONNECTED_CONTEXTS_REQUIRED"]
    assert result["predictions"] == [] and result["measurements"] is None


def test_real_adapter_rejects_predecessor_policy_before_native(runtime):
    old_policy = runtime._v12.MeshWorkflowPolicyV12(**WORKFLOW_FIELDS)
    result = runtime.inspect_cad_guided_mesh_v13(scan_bytes=b"unused", cad_bytes=b"unused",
        position_error_bound_mm=.05, workflow_policy=old_policy, registration_policy=None,
        measurement_policy=None, instance_policy=None, allow_ocp=True)
    assert result["phase"] == "ADMISSION" and result["reasons"] == ["VALIDATED_WORKFLOW_POLICY_REQUIRED"]


def test_execution_boundary_forwards_exact_bytes_and_policies(runtime, monkeypatch):
    values = dict(workflow_policy=object(), sentinel="fixed-policy-only")
    monkeypatch.setattr(sys.modules[__name__], "_policy_objects", lambda: values)
    calls = []
    def spy(**kwargs):
        calls.append(kwargs)
        return {"mock_only": True}
    monkeypatch.setattr(runtime, "inspect_cad_guided_mesh_v13", spy)
    assert _execute_workflow(b"invented-step", b"invented-ply") == {"mock_only": True}
    assert calls == [dict(cad_bytes=b"invented-step", scan_bytes=b"invented-ply", **values)]


def test_top_projection_never_concatenates_nested_member_predictions():
    cover, member = {"feature_id": "cover"}, {"feature_id": "member"}
    value = dict(predictions=[cover], prediction_collection_status="CONDITIONAL_COVER_SUPPORT_GROUPS_NOT_MANUFACTURING_EVENTS",
        measurements=dict(schema="CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2",
            source_context_result_role="RETAINED_EVIDENCE_NOT_ADDITIONAL_PREDICTIONS",
            source_context_result=dict(predictions=[member]), predictions=[cover], support_groups=[
                dict(status="MEASURED_CONDITIONAL", finite_support=dict(status="SUPPORTED", instance=cover))]))
    assert _assert_top_level_cover_projection(value)["cover_prediction_count"] == 1
    for predictions in ([cover, member], [member], []):
        bad = copy.deepcopy(value)
        bad["predictions"] = predictions
        with pytest.raises(AssertionError):
            _assert_top_level_cover_projection(bad)


def test_opt_in_precedes_source_fixture_or_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(OPT_IN, raising=False)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(AssertionError, match="EXPLICIT_DEPTH_NATIVE_SUPERVISOR"):
        run_single_native_byte_case(path)
    assert not path.exists()


def test_preexisting_artifact_directory_is_never_reused(tmp_path, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setattr(sys.modules[__name__], "_require_execution_limits", lambda: None)
    path = tmp_path/"ARTIFACTS"
    path.mkdir()
    with pytest.raises(FileExistsError):
        run_single_native_byte_case(path)
    assert list(path.iterdir()) == []


def _mock_recorder(monkeypatch, failure=None):
    namespace = sys.modules[__name__]
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setattr(namespace, "_require_execution_limits", lambda: None)
    monkeypatch.setattr(namespace, "_source_bytes", lambda name: b"MOCK_VERIFIED_SOURCE")
    monkeypatch.setattr(namespace, "_make_fixture", lambda: (
        b"invented-CAD", b"invented-PLY", np.zeros((3, 3)), np.array([[0, 1, 2]]), {"mock": True}))
    workflow = dict(status="UNAVAILABLE", phase="WHOLE_PART_REGISTRATION", all_rows=[1, 2, 3],
        marker="COMPLETE_RETURNED_MOCK_NOT_ENGINE_POSITIVE")
    def execute(*args):
        if failure == "execute":
            raise RuntimeError("INVENTED_EXECUTION_FAILURE")
        return copy.deepcopy(workflow)
    def evaluate(*args):
        if failure == "oracle":
            raise AssertionError("INVENTED_REQUIRED_POSITIVE_FAILED")
        return {"mock_evaluation_only": True}
    monkeypatch.setattr(namespace, "_execute_workflow", execute)
    monkeypatch.setattr(namespace, "_evaluate", evaluate)
    monkeypatch.setattr(namespace, "_diagnostic_evaluate", lambda *args: {
        "status": "MOCK_LEDGER_NOT_NATIVE_INTEGRITY_EVIDENCE"})
    return workflow


@pytest.mark.parametrize("failure", [None, "oracle", "execute"])
def test_recorder_separates_durable_return_from_geometry_pass(tmp_path, monkeypatch, failure):
    workflow = _mock_recorder(monkeypatch, failure)
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


def test_inputs_precede_actual_entry_and_full_result_precedes_oracle(tmp_path, monkeypatch):
    workflow = _mock_recorder(monkeypatch)
    directory = tmp_path/"ARTIFACTS"
    def execute(cad_blob, scan_blob):
        assert (directory/"CAD.step").read_bytes() == cad_blob
        assert (directory/"SCAN.ply").read_bytes() == scan_blob
        return copy.deepcopy(workflow)
    def evaluate(result, cad_blob, scan_blob, *args):
        assert json.loads((directory/"CASE_RESULT.json").read_bytes())["workflow"] == workflow == result
        assert sorted(path.name for path in directory.iterdir()) == sorted(ARTIFACT_NAMES)
        raise AssertionError("INVENTED_FAILURE_AFTER_DURABLE_RESULT")
    monkeypatch.setattr(sys.modules[__name__], "_execute_workflow", execute)
    monkeypatch.setattr(sys.modules[__name__], "_evaluate", evaluate)
    receipt = run_single_native_byte_case(directory)
    assert receipt["status"] == "FAIL" and receipt["full_returned_workflow_retained"] is True
    assert receipt["oracle_error"]["message"] == "INVENTED_FAILURE_AFTER_DURABLE_RESULT"


def test_late_source_drift_preserves_workflow_but_blocks_positive(tmp_path, monkeypatch):
    workflow = _mock_recorder(monkeypatch)
    calls = 0
    def source(name):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise AssertionError("FIXTURE_SOURCE_DRIFT")
        return b"MOCK"
    monkeypatch.setattr(sys.modules[__name__], "_source_bytes", source)
    receipt = run_single_native_byte_case(tmp_path/"ARTIFACTS")
    retained = json.loads((tmp_path/"ARTIFACTS"/"CASE_RESULT.json").read_bytes())
    assert retained["workflow"] == workflow and receipt["source_consistency"] is False
    assert receipt["status"] == "FAIL"


def test_report_overflow_retains_no_successful_prefix(tmp_path, monkeypatch):
    _mock_recorder(monkeypatch)
    monkeypatch.setitem(ARTIFACT_LIMITS, "result_json_bytes", 1)
    receipt = run_single_native_byte_case(tmp_path/"ARTIFACTS")
    retained = json.loads((tmp_path/"ARTIFACTS"/"CASE_RESULT.json").read_bytes())
    assert retained["discarded_workflow_whole"] is True and retained["workflow"] is None
    assert receipt["full_returned_workflow_retained"] is False and receipt["status"] == "FAIL"
    assert not list((tmp_path/"ARTIFACTS").glob("*.partial"))


def test_metadata_byte_edge_never_traverses_the_returned_workflow():
    record = dict(workflow=object(), marker="")
    overhead = _check_result_metadata(record)
    record["marker"] = "x"*(ARTIFACT_LIMITS["metadata_json_bytes"]-overhead)
    assert _check_result_metadata(record) == ARTIFACT_LIMITS["metadata_json_bytes"]
    record["marker"] += "x"
    with pytest.raises(ValueError, match="DEPTH_NATIVE_ARTIFACT_LIMIT"):
        _check_result_metadata(record)


def test_metadata_overflow_retains_whole_refusal_before_geometry_oracle(tmp_path, monkeypatch):
    _mock_recorder(monkeypatch)
    monkeypatch.setitem(ARTIFACT_LIMITS, "metadata_json_bytes", 1)
    calls = []
    monkeypatch.setattr(sys.modules[__name__], "_evaluate", lambda *args: calls.append(args))
    receipt = run_single_native_byte_case(tmp_path/"ARTIFACTS")
    retained = json.loads((tmp_path/"ARTIFACTS"/"CASE_RESULT.json").read_bytes())
    assert retained["discarded_workflow_whole"] is True and retained["workflow"] is None
    assert receipt["full_returned_workflow_retained"] is False and receipt["status"] == "FAIL"
    assert calls == [] and not list((tmp_path/"ARTIFACTS").glob("*.partial"))


def test_exclusive_artifact_publication_never_overwrites_existing_target(tmp_path):
    receipt = _write(tmp_path, "CASE_START.json", {"first": True}, 4096)
    original = (tmp_path/"CASE_START.json").read_bytes()
    with pytest.raises(FileExistsError):
        _write(tmp_path, "CASE_START.json", {"second": True}, 4096)
    assert (tmp_path/"CASE_START.json").read_bytes() == original
    assert receipt["sha256"] == _sha(original)


def test_unset_limits_with_opt_in_refuse_before_source_fixture_or_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setitem(RESOURCE_LIMITS, "worker_cpu_seconds", None)
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError("UNSET_LIMITS_MUST_NOT_DISPATCH")
    monkeypatch.setattr(sys.modules[__name__], "_source_bytes", forbidden)
    monkeypatch.setattr(sys.modules[__name__], "_make_fixture", forbidden)
    path = tmp_path/"ARTIFACTS"
    with pytest.raises(ValueError, match="NATIVE_LIMITS_PENDING_CONNECTED_TIMING"):
        run_single_native_byte_case(path)
    assert calls == [] and not path.exists()


def _mock_schedule_binding():
    """Two invented binding rows; no schedule execution, equations or raw proof."""
    kwargs = _policy_objects()
    expected = dict(mode="v2", budgets=asdict(kwargs["two_plane_schedule_budgets"]))
    cap = asdict(kwargs["initialization_budgets"])
    digests = dict(scan_bank_equations_sha256="a"*64, cad_bank_equations_sha256="b"*64)
    bound = dict(two_plane_schedule_mode="v2",
        two_plane_schedule_configuration_sha256=_sha(_encode(expected, 65536)),
        two_plane_schedule_input_digests=digests)
    initialization = dict(two_plane_schedule_configuration=copy.deepcopy(expected), budgets=copy.deepcopy(cap),
        two_plane_schedule=dict(schema_version="two_plane_schedule_v2", status="COMPLETE",
            selected_policy_evaluation_complete=True, budgets=copy.deepcopy(expected["budgets"]),
            input_digests=copy.deepcopy(digests)))
    result = dict(policies=dict(two_plane_enabled=True, initialization_budgets=copy.deepcopy(cap),
        two_plane_schedule_configuration=copy.deepcopy(expected)), registration=dict(evidence=dict(
        two_plane_schedule_configuration=copy.deepcopy(expected), initialization=initialization,
        measurement_context_indices=[0, 1])), registration_array_binding=[copy.deepcopy(bound), copy.deepcopy(bound)])
    return result, expected, cap, bound


@pytest.mark.parametrize("fault", [None, "missing_configuration", "boolean_budget", "wrong_schedule",
    "stale_context", "stale_digest", "short_contexts", "unexpected_marker"])
def test_schedule_binding_copies_are_all_required_without_geometry(runtime, fault):
    result, expected, cap, bound = _mock_schedule_binding()
    evidence = result["registration"]["evidence"]
    if fault == "missing_configuration":
        del evidence["two_plane_schedule_configuration"]
    elif fault == "boolean_budget":
        evidence["initialization"]["budgets"]["max_refined_start_groups"] = True
    elif fault == "wrong_schedule":
        evidence["initialization"]["two_plane_schedule"]["schema_version"] = "two_plane_schedule_v1"
    elif fault == "stale_context":
        result["registration_array_binding"][-1]["two_plane_schedule_configuration_sha256"] = "c"*64
    elif fault == "stale_digest":
        result["registration_array_binding"][-1]["two_plane_schedule_input_digests"]["scan_bank_equations_sha256"] = "c"*64
    elif fault == "short_contexts":
        result["registration_array_binding"].pop()
    elif fault == "unexpected_marker":
        result["registration_array_binding"][-1]["two_plane_schedule_extra"] = None
    if fault is None:
        answer = _assert_schedule_metadata(result, expected, cap, bound)
        assert answer["checked_binding_rows"] == 2 and answer["raw_support_reconstructed"] is False
        assert answer["per_context_identity_proved_by_schedule_binding"] is False
    else:
        with pytest.raises((AssertionError, KeyError)):
            _assert_schedule_metadata(result, expected, cap, bound)


@pytest.mark.skipif(os.environ.get(OPT_IN) != "1",
    reason="Inactive native draft: accepted connected audit, exact diagnostic PRE and supervisor required")
def test_two_plane_depth_v1_single_native_byte_case():
    _require_execution_limits()
    directory = os.environ.get(ARTIFACT_ENV)
    assert directory, "FRESH_ROOT_OWNED_ARTIFACT_DIRECTORY_REQUIRED"
    receipt = run_single_native_byte_case(directory)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    assert receipt["status"] == "PASS", receipt
    assert receipt["positive_geometry_passed"] is True
    assert receipt["full_returned_workflow_retained"] is True and receipt["source_consistency"] is True
