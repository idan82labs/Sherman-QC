"""Run only the named invented-input CAD measurement tests in one bounded child.

No input/output path options, app startup or physical fixtures. Only the explicit
native_features_v1_constructed suite enables its fixed invented-native matrix.
The RSS limit is sampled process-group RSS, not an OS memory sandbox.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import secrets
import selectors
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
TESTS = {
    "report": "tests/test_cad_guided_measurement_report.py",
    "geometry": "tests/test_cad_guided_measurement_geometry.py",
    "stress": "tests/test_cad_guided_measurement_stress.py",
    "stress_free": "tests/test_cad_guided_measurement_stress.py::test_frozen_8192_case_position_only_stress",
    "stress_tangent": "tests/test_cad_guided_measurement_stress.py::test_same_8192_case_measured_flange_tangent_candidate",
    "tangent_math": "tests/test_cad_guided_tangent_math.py",
    "registration": "tests/test_cad_guided_registration.py",
    "instances": "tests/test_cad_guided_instances.py",
    "evaluation": "tests/test_cad_guided_evaluation.py",
    "registered_source": "tests/test_cad_registered_guided_entry_source.py",
    "native_worker_source": "tests/test_cad_native_worker_output_budget.py",
    "instance_integration": "tests/test_cad_guided_instance_integration.py",
    "offline_source": "tests/test_cad_guided_offline_source.py",
    "case_input": "tests/test_cad_guided_case_input.py",
    "paired_evaluation": "tests/test_cad_guided_paired_evaluation.py",
    "development_bundle": "tests/test_cad_guided_development_bundle.py",
    "row_lineage": "tests/test_cad_guided_row_lineage.py",
    "line_convention_math": "tests/test_cad_line_convention_math.py",
    "gold11_admission": "tests/test_gold11_admission_geometry.py",
    "gold11_review_controls": "tests/test_gold11_review_controls.py",
    "mesh_input_v1": "tests/test_cad_guided_mesh_input_v1.py",
    "whole_part_registration_v1": "tests/test_cad_guided_global_registration_v1.py",
    "native_features_v1": "tests/test_cad_guided_native_features_v1.py",
    "native_features_v1_constructed": "tests/test_cad_guided_native_features_v1.py",
    "full_raw_measurement_v1": "tests/test_cad_guided_local_measurement_v1.py",
    "cad_sampling_v1": "tests/test_cad_guided_cad_sampling_v1.py",
    "mesh_offline_v1": "tests/test_cad_guided_mesh_offline_v1.py",
    "native_tolerance_probe_v1": "tests/test_cad_guided_native_features_v1.py",
    "cad_sampling_native_v1": "tests/test_cad_guided_cad_sampling_v1.py",
    "gold11_mesh_scoring_v1": "tests/test_gold11_mesh_scoring_v1.py",
    "native_trim_uncertainty_v1": "tests/test_native_trim_uncertainty_v1.py",
    "native_trim_uncertainty_constructed_v1": "tests/test_native_trim_uncertainty_v1.py",
    "mesh_connected_native_v1": "tests/test_cad_guided_mesh_connected_native_v1.py",
    "mesh_connected_native_dense_v1": "tests/test_cad_guided_mesh_connected_native_dense_v1.py",
    "registration_stage_trace_v1": "tests/test_cad_guided_registration_stage_trace_v1.py",
    "gold11_mesh_runner_v1": "tests/test_gold11_mesh_runner_v1.py",
    "gold11_mesh_guard_smoke_v1": "tests/test_gold11_mesh_runner_v1.py",
    "cad_surface_v2": "tests/test_cad_guided_cad_surface_v2.py",
    "triangle_surface_v2": "tests/test_cad_guided_triangle_surface_v2.py",
    "triangle_surface_v2_independent": "tests/test_cad_guided_triangle_surface_v2_independent.py",
    "native_validity_v2_source": "tests/test_v2_native_validity_diagnostic.py",
    "native_validity_v2_constructed": "tests/test_v2_native_validity_diagnostic.py",
    "surface_registration_v2": "tests/test_cad_guided_global_registration_v2.py",
    "surface_registration_v2_matrix": "tests/test_cad_guided_global_registration_v2.py",
    "mesh_offline_v2": "tests/test_cad_guided_mesh_offline_v2.py",
    "mesh_connected_surface_v2_source": "tests/test_cad_guided_mesh_connected_surface_v2.py",
    "mesh_connected_surface_v2": "tests/test_cad_guided_mesh_connected_surface_v2.py",
    "finite_gauss_newton_v3": "tests/test_cad_guided_finite_gauss_newton_v3.py",
    "finite_fine_comparison_v3_source": "tests/test_cad_guided_finite_fine_comparison_v3.py",
    "finite_fine_comparison_v3": "tests/test_cad_guided_finite_fine_comparison_v3.py",
    "native_validity_v2_supervisor": "tests/test_v2_native_validity_supervisor.py",
    "native_validity_v2_guard_smoke": "tests/test_v2_native_validity_supervisor.py",
    "cad_smoke_group_custody": "tests/test_cad_smoke_group_custody.py",
    "mesh_topology_v4": "tests/test_cad_guided_mesh_topology_v4.py",
    "plane_hypotheses_v4": "tests/test_cad_guided_plane_hypotheses_v4.py",
    "mesh_plane_proposals_v4": "tests/test_cad_guided_mesh_plane_proposals_v4.py",
    "mesh_plane_capacity_v4_source": "tests/test_cad_guided_mesh_plane_capacity_v4.py",
    "mesh_plane_capacity_v4": "tests/test_cad_guided_mesh_plane_capacity_v4.py",
    "plane_initialization_v4": "tests/test_cad_guided_plane_initialization_v4.py",
    "mesh_connected_planes_v4_source": "tests/test_cad_guided_mesh_connected_planes_v4.py",
    "mesh_connected_planes_v4": "tests/test_cad_guided_mesh_connected_planes_v4.py",
    "native_context_coverage_v3": "tests/test_cad_guided_native_context_coverage_v3.py",
    "native_context_collection_v3_source": "tests/test_v3_native_context_collection.py",
    "native_context_collection_v3": "tests/test_v3_native_context_collection.py",
    "native_context_collection_supervision_v3": "tests/test_v3_native_context_test_supervision.py",
    "noisy_mesh_planes_v5": "tests/test_cad_guided_noisy_mesh_planes_v5.py",
    "plane_search_v5": "tests/test_cad_guided_plane_search_v5.py",
    "noisy_mesh_capacity_v5_source": "tests/test_cad_guided_noisy_mesh_capacity_v5.py",
    "noisy_mesh_capacity_v5": "tests/test_cad_guided_noisy_mesh_capacity_v5.py",
    "plane_initialization_v5": "tests/test_cad_guided_plane_initialization_v5.py",
    "noisy_mesh_evidence_v6": "tests/test_cad_guided_noisy_mesh_evidence_v6.py",
    "noisy_mesh_evidence_v7": "tests/test_cad_guided_noisy_mesh_evidence_v7.py",
    "report_envelope_v6": "tests/test_cad_guided_report_envelope_v6.py",
    "noisy_mesh_capacity_v6_source": "tests/test_cad_guided_noisy_mesh_capacity_v6.py",
    "noisy_mesh_capacity_v6": "tests/test_cad_guided_noisy_mesh_capacity_v6.py",
    "plane_initialization_v6": "tests/test_cad_guided_plane_initialization_v6.py",
}
WALL_SECONDS = 300
CPU_SECONDS = 240
RSS_BYTES = 2 * 1024**3
OUTPUT_BYTES = 1024**2
NATIVE_V1_SOURCES = ("backend/cad_guided_native_features_v1.py",
                     "backend/cad_guided_native_features.py",
                     "backend/cad_guided_measurement.py", "backend/cad_guided_instances.py")
LOCAL_V1_SOURCES = ("backend/cad_guided_local_measurement_v1.py",
                   "backend/cad_guided_measurement.py", "backend/cad_guided_instances.py")
SCORER_V1_SOURCES = ("scripts/gold11_mesh_scoring_v1.py", "scripts/evaluate_f1_against_manual_gold.py",
                    "scripts/f1_physical_evaluation_contract_v2.py")
V1_SOURCE_BINDINGS = {
    "mesh_input_v1": ("backend/cad_guided_mesh_input_v1.py",),
    "whole_part_registration_v1": ("backend/cad_guided_global_registration_v1.py",),
    "native_features_v1": NATIVE_V1_SOURCES,
    "native_features_v1_constructed": NATIVE_V1_SOURCES,
    "native_tolerance_probe_v1": NATIVE_V1_SOURCES,
    "full_raw_measurement_v1": LOCAL_V1_SOURCES,
    "cad_sampling_v1": ("backend/cad_guided_cad_sampling_v1.py",),
    "cad_sampling_native_v1": ("backend/cad_guided_cad_sampling_v1.py",),
    "mesh_offline_v1": tuple(dict.fromkeys((*NATIVE_V1_SOURCES, *LOCAL_V1_SOURCES,
        "backend/cad_guided_mesh_offline_v1.py", "backend/cad_guided_mesh_input_v1.py",
        "backend/cad_guided_global_registration_v1.py", "backend/cad_guided_cad_sampling_v1.py"))),
    "gold11_mesh_scoring_v1": SCORER_V1_SOURCES,
    "native_trim_uncertainty_v1": NATIVE_V1_SOURCES,
    "native_trim_uncertainty_constructed_v1": (*NATIVE_V1_SOURCES,
                                              "tests/test_cad_guided_native_features_v1.py"),
    "gold11_mesh_runner_v1": ("scripts/run_gold11_mesh_development_v1.py",),
    "gold11_mesh_guard_smoke_v1": ("scripts/run_gold11_mesh_development_v1.py",),
    "cad_surface_v2": ("backend/cad_guided_cad_surface_v2.py", "backend/cad_guided_cad_sampling_v1.py"),
    "triangle_surface_v2": ("backend/cad_guided_triangle_surface_v2.py",),
    "triangle_surface_v2_independent": ("backend/cad_guided_triangle_surface_v2.py",),
    "native_validity_v2_source": ("scripts/v2_native_validity_diagnostic.py",
        "docs/research/gold11_mesh_development_v1_attempt/source/backend/cad_guided_mesh_offline_v1.py",
        "docs/research/2026-09-09_v2_native_validity_diagnostic_protocol.md"),
    "surface_registration_v2": ("backend/cad_guided_global_registration_v2.py",
        "backend/cad_guided_global_registration_v1.py", "backend/cad_guided_triangle_surface_v2.py"),
}
V1_SOURCE_BINDINGS["mesh_connected_native_v1"] = V1_SOURCE_BINDINGS["mesh_offline_v1"]
V1_SOURCE_BINDINGS["mesh_connected_native_dense_v1"] = (
    *V1_SOURCE_BINDINGS["mesh_offline_v1"], "tests/test_cad_guided_mesh_connected_native_v1.py")
V1_SOURCE_BINDINGS["registration_stage_trace_v1"] = V1_SOURCE_BINDINGS["mesh_connected_native_dense_v1"]
V1_SOURCE_BINDINGS["native_validity_v2_constructed"] = V1_SOURCE_BINDINGS["native_validity_v2_source"]
V1_SOURCE_BINDINGS["surface_registration_v2_matrix"] = V1_SOURCE_BINDINGS["surface_registration_v2"]
V1_SOURCE_BINDINGS["mesh_offline_v2"] = tuple(dict.fromkeys((
    *V1_SOURCE_BINDINGS["mesh_offline_v1"], *V1_SOURCE_BINDINGS["surface_registration_v2"],
    "backend/cad_guided_mesh_offline_v2.py", "backend/cad_guided_cad_surface_v2.py")))
V1_SOURCE_BINDINGS["mesh_connected_surface_v2_source"] = (
    *V1_SOURCE_BINDINGS["mesh_offline_v2"],
    "tests/test_cad_guided_mesh_connected_native_v1.py",
    "tests/test_cad_guided_mesh_connected_native_dense_v1.py")
V1_SOURCE_BINDINGS["mesh_connected_surface_v2"] = V1_SOURCE_BINDINGS["mesh_connected_surface_v2_source"]
V1_SOURCE_BINDINGS["finite_gauss_newton_v3"] = (
    *V1_SOURCE_BINDINGS["surface_registration_v2"], "backend/cad_guided_finite_gauss_newton_v3.py",
    "docs/research/2026-09-09_finite_fine_v3_mechanism_protocol.md")
V1_SOURCE_BINDINGS["finite_fine_comparison_v3_source"] = (
    *V1_SOURCE_BINDINGS["mesh_connected_surface_v2_source"], "backend/cad_guided_finite_gauss_newton_v3.py",
    "docs/research/2026-09-09_finite_fine_v3_mechanism_protocol.md")
V1_SOURCE_BINDINGS["finite_fine_comparison_v3"] = V1_SOURCE_BINDINGS["finite_fine_comparison_v3_source"]
V1_SOURCE_BINDINGS["native_validity_v2_supervisor"] = (
    *V1_SOURCE_BINDINGS["native_validity_v2_source"],
    "scripts/run_v2_native_validity_diagnostic.py", "scripts/review_gold11_inputs_v1.py",
    "docs/research/gold11_mesh_development_v1_attempt/source/scripts/run_gold11_mesh_development_v1.py",
    "tests/test_v2_native_validity_diagnostic.py")
V1_SOURCE_BINDINGS["native_validity_v2_guard_smoke"] = (
    *V1_SOURCE_BINDINGS["native_validity_v2_supervisor"], "tests/test_cad_smoke_group_custody.py",
    "docs/research/2026-09-09_v2_smoke_custody_successor_protocol.md")
V1_SOURCE_BINDINGS["cad_smoke_group_custody"] = ("tests/test_v2_native_validity_supervisor.py",
    "docs/research/2026-09-09_v2_smoke_custody_successor_protocol.md")
V4_PLANE_PROTOCOLS = ("docs/research/2026-09-09_plane_initialization_v4_source_protocol.md",
                      "docs/research/2026-09-09_plane_hypotheses_v4_design.md")
V1_SOURCE_BINDINGS["mesh_topology_v4"] = (
    "backend/cad_guided_mesh_topology_v4.py", "backend/cad_guided_mesh_input_v1.py", *V4_PLANE_PROTOCOLS)
V1_SOURCE_BINDINGS["plane_hypotheses_v4"] = ("backend/cad_guided_plane_hypotheses_v4.py", *V4_PLANE_PROTOCOLS)
V1_SOURCE_BINDINGS["mesh_plane_proposals_v4"] = ("backend/cad_guided_mesh_plane_proposals_v4.py",
    "docs/research/2026-09-09_mesh_plane_proposals_v4_protocol.md")
V1_SOURCE_BINDINGS["mesh_plane_capacity_v4_source"] = (
    *V1_SOURCE_BINDINGS["mesh_plane_proposals_v4"],
    "docs/research/2026-09-09_mesh_plane_capacity_v4_protocol.md")
V1_SOURCE_BINDINGS["mesh_plane_capacity_v4"] = V1_SOURCE_BINDINGS["mesh_plane_capacity_v4_source"]
V1_SOURCE_BINDINGS["plane_initialization_v4"] = (
    *V1_SOURCE_BINDINGS["surface_registration_v2"],
    *V1_SOURCE_BINDINGS["mesh_plane_proposals_v4"],
    *V1_SOURCE_BINDINGS["plane_hypotheses_v4"],
    "backend/cad_guided_plane_initialization_v4.py",
    "docs/research/2026-09-09_plane_initialization_connected_v4_protocol.md")
V1_SOURCE_BINDINGS["mesh_connected_planes_v4_source"] = tuple(dict.fromkeys((
    *V1_SOURCE_BINDINGS["mesh_connected_surface_v2_source"],
    *V1_SOURCE_BINDINGS["mesh_topology_v4"],
    *V1_SOURCE_BINDINGS["plane_initialization_v4"],
    "tests/test_cad_guided_mesh_connected_surface_v2.py",
    "backend/cad_guided_mesh_offline_v4.py",
    "docs/research/2026-09-09_mesh_connected_planes_v4_protocol.md")))
V1_SOURCE_BINDINGS["mesh_connected_planes_v4"] = V1_SOURCE_BINDINGS["mesh_connected_planes_v4_source"]
V1_SOURCE_BINDINGS["native_context_coverage_v3"] = (
    "backend/cad_guided_native_context_coverage_v3.py",
    "docs/research/2026-09-09_native_context_coverage_v3_protocol.md")
V1_SOURCE_BINDINGS["native_context_collection_v3_source"] = (
    *V1_SOURCE_BINDINGS["native_context_coverage_v3"],
    "scripts/v3_native_context_collection.py", "scripts/v2_native_validity_diagnostic.py",
    "docs/research/2026-09-09_native_context_collection_v3_protocol.md")
V1_SOURCE_BINDINGS["native_context_collection_supervision_v3"] = (
    "scripts/v3_native_context_test_supervision.py",
    "docs/research/2026-09-09_native_context_collection_v3_protocol.md")
V1_SOURCE_BINDINGS["native_context_collection_v3"] = (
    *V1_SOURCE_BINDINGS["native_context_collection_v3_source"],
    "scripts/v3_native_context_test_supervision.py", "tests/test_v3_native_context_test_supervision.py")
V1_SOURCE_BINDINGS["noisy_mesh_planes_v5"] = (
    "backend/cad_guided_noisy_mesh_planes_v5.py",
    "docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md",
    "docs/research/2026-09-09_noisy_plane_registration_design_notes.md")
V1_SOURCE_BINDINGS["plane_search_v5"] = (
    "backend/cad_guided_plane_search_v5.py",
    "docs/research/2026-09-09_plane_search_v5_protocol.md")
V1_SOURCE_BINDINGS["noisy_mesh_capacity_v5_source"] = (
    *V1_SOURCE_BINDINGS["noisy_mesh_planes_v5"],
    "docs/research/2026-09-09_noisy_mesh_capacity_v5_protocol.md")
V1_SOURCE_BINDINGS["noisy_mesh_capacity_v5"] = V1_SOURCE_BINDINGS["noisy_mesh_capacity_v5_source"]
V1_SOURCE_BINDINGS["plane_initialization_v5"] = tuple(dict.fromkeys((
    *V1_SOURCE_BINDINGS["plane_initialization_v4"],
    *V1_SOURCE_BINDINGS["noisy_mesh_planes_v5"], *V1_SOURCE_BINDINGS["plane_search_v5"],
    "backend/cad_guided_plane_initialization_v5.py",
    "docs/research/2026-09-09_plane_initialization_v5_protocol.md")))
SMOKE_REGISTRY_ENV = "SHERMAN_SMOKE_GROUP_REGISTRY"
SMOKE_NONCE_ENV = "SHERMAN_SMOKE_GROUP_NONCE"
SMOKE_GROUP_CAP = 2
SMOKE_SCHEMA = "ACKNOWLEDGED_INVENTED_SMOKE_GROUP_V1"

V6_METADATA_SOURCES = (
    *V1_SOURCE_BINDINGS["noisy_mesh_planes_v5"],
    "backend/cad_guided_noisy_mesh_planes_v6.py",
    "docs/research/2026-09-09_noisy_mesh_evidence_v6_protocol.md")
V1_SOURCE_BINDINGS["noisy_mesh_evidence_v6"] = (
    *V6_METADATA_SOURCES, "tests/test_cad_guided_noisy_mesh_planes_v5.py",
    "docs/research/2026-09-09_noisy_mesh_evidence_v6_test_protocol.md")
V7_METADATA_SOURCES = (
    *V6_METADATA_SOURCES, "backend/cad_guided_noisy_mesh_planes_v7.py",
    "docs/research/2026-09-09_noisy_mesh_evidence_v7_protocol.md")
V1_SOURCE_BINDINGS["noisy_mesh_evidence_v7"] = (
    *V1_SOURCE_BINDINGS["noisy_mesh_evidence_v6"],
    "tests/test_cad_guided_noisy_mesh_evidence_v6.py",
    "backend/cad_guided_noisy_mesh_planes_v7.py",
    "docs/research/2026-09-09_noisy_mesh_evidence_v7_protocol.md",
    "docs/research/2026-09-09_noisy_mesh_evidence_v7_test_protocol.md")
V1_SOURCE_BINDINGS["report_envelope_v6"] = (
    "backend/cad_guided_report_envelope_v6.py",
    "docs/research/2026-09-09_report_envelope_v6_protocol.md")
V1_SOURCE_BINDINGS["noisy_mesh_capacity_v6_source"] = (
    *V7_METADATA_SOURCES, "tests/test_cad_guided_noisy_mesh_capacity_v5.py",
    "docs/research/2026-09-09_noisy_mesh_capacity_v5_protocol.md",
    "docs/research/2026-09-09_noisy_mesh_capacity_v6_protocol.md")
V1_SOURCE_BINDINGS["noisy_mesh_capacity_v6"] = V1_SOURCE_BINDINGS["noisy_mesh_capacity_v6_source"]
V1_SOURCE_BINDINGS["plane_initialization_v6"] = tuple(dict.fromkeys((
    *V1_SOURCE_BINDINGS["plane_initialization_v5"],
    *V7_METADATA_SOURCES, *V1_SOURCE_BINDINGS["report_envelope_v6"],
    "backend/cad_guided_plane_initialization_v6.py",
    "tests/test_cad_guided_plane_initialization_v5.py",
    "docs/research/2026-09-09_plane_initialization_v6_protocol.md")))



def smoke_encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def smoke_require(condition, reason):
    if not condition:
        error = RuntimeError(reason)
        error.smoke_reason = reason
        raise error


def smoke_failure(error, stage, reason="SMOKE_UNEXPECTED_EXCEPTION", **fields):
    """Retain the original type and a fixed reason, never arbitrary path text."""
    if isinstance(error, PermissionError):
        reason = "SMOKE_OS_PERMISSION_DENIED"
    elif isinstance(error, FileNotFoundError):
        reason = "SMOKE_REQUIRED_FILE_MISSING"
    elif isinstance(error, json.JSONDecodeError):
        reason = "SMOKE_REGISTRY_JSON_INVALID"
    elif isinstance(error, subprocess.TimeoutExpired):
        reason = "SMOKE_PROCESS_WAIT_TIMEOUT"
    return dict(exception_type=type(error).__name__, reason=getattr(error, "smoke_reason", reason),
                stage=stage, monotonic=time.monotonic(), **fields)


def first_smoke_failure(previous, error, stage):
    return previous if previous is not None else smoke_failure(error, stage)


def smoke_read(path):
    """Only fixed registry filenames; private regular descriptors, at most2KiB."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        smoke_require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                      and not before.st_mode & 0o077 and 0 < before.st_size <= 2048,
                      "SMOKE_REGISTRY_FILE_CONTRACT")
        blob = os.read(fd, 2049)
        after = os.fstat(fd)
        smoke_require(len(blob) == before.st_size and all(getattr(before, field) == getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")),
            "SMOKE_REGISTRY_FILE_CHANGED")
    finally:
        os.close(fd)
    def pairs(items):
        result = {}
        for key, value in items:
            smoke_require(key not in result, "SMOKE_REGISTRY_DUPLICATE_KEY")
            result[key] = value
        return result
    def nonfinite(_):
        smoke_require(False, "SMOKE_REGISTRY_NONFINITE")
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=nonfinite)


def smoke_publish(directory, filename, value):
    """Atomic exclusive target; KEEP the bounded stage hardlink until terminal.

    Unlinking stage after publishing target changes their shared inode ctime and
    races a concurrent descriptor-stability check. There are only eight fixed
    target/stage names total; terminal cache cleanup removes both together.
    """
    blob = smoke_encoded(value)
    smoke_require(len(blob) <= 2048, "SMOKE_REGISTRY_BYTE_CAP")
    stage, target = directory/("."+filename+".tmp"), directory/filename
    fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(stage, target)  # Never mutate this inode again before terminal cleanup.


def smoke_registry_identity(directory):
    row = directory.lstat()
    smoke_require(directory.is_absolute() and directory == directory.resolve()
                  and stat.S_ISDIR(row.st_mode) and row.st_uid == os.getuid()
                  and not row.st_mode & 0o077, "SMOKE_REGISTRY_NOT_PRIVATE")
    return row.st_dev, row.st_ino, row.st_uid


def register_smoke_child(child, directory, nonce, *, timeout=2.):
    """Smoke-only client barrier: caller MUST NOT write child stdin before ACK.

    The forwarding Popen proxy kills/reaps the directly owned blocked child on
    every failure. No native/physical driver uses this test-only capability.
    """
    identity = smoke_registry_identity(directory)
    smoke_require(type(nonce) is str and len(nonce) == 64
                  and all(c in "0123456789abcdef" for c in nonce), "SMOKE_NONCE_REQUIRED")
    parent, harness_pid = os.getpid(), os.getppid()
    record = None
    for slot in range(SMOKE_GROUP_CAP):
        candidate = dict(schema=SMOKE_SCHEMA, nonce=nonce, slot=slot,
                         pytest_pid=parent, pid=child.pid, pgid=child.pid)
        try:
            smoke_publish(directory, f"{slot}.request.json", candidate)
        except FileExistsError:
            continue
        record = candidate
        break
    smoke_require(record is not None, "SMOKE_GROUP_REGISTRATION_CAP")
    expected = dict(schema=SMOKE_SCHEMA, decision="ACK", nonce=nonce, slot=record["slot"],
        pytest_pid=parent, pid=child.pid, pgid=child.pid, harness_pid=harness_pid,
        registration_sha256=hashlib.sha256(smoke_encoded(record)).hexdigest())
    stop = time.monotonic()+timeout
    while time.monotonic() < stop:
        smoke_require(child.poll() is None and os.getppid() == harness_pid, "SMOKE_CHILD_OR_HARNESS_EXITED")
        smoke_require(smoke_registry_identity(directory) == identity, "SMOKE_REGISTRY_REPLACED")
        try:
            ack = smoke_read(directory/f"{record['slot']}.ack.json")
        except FileNotFoundError:
            time.sleep(.01)
            continue
        smoke_require(type(ack) is dict and set(ack) == set(expected) | {"ack_phase", "ack_monotonic"}
            and smoke_encoded({key: ack[key] for key in expected}) == smoke_encoded(expected)
            and ack["ack_phase"] == "MAIN_ADMISSION"
            and type(ack["ack_monotonic"]) in (int, float)
            and 0 <= ack["ack_monotonic"] <= time.monotonic(), "SMOKE_ACK_BINDING")
        return record["slot"]
    smoke_require(False, "SMOKE_ACK_TIMEOUT")


def smoke_process_table():
    result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="],
                            capture_output=True, text=True, timeout=1)
    rows = [line.split() for line in result.stdout.splitlines()]
    smoke_require(result.returncode == 0 and rows and all(len(row) == 5
        and all(value.isdigit() for value in row[:4]) and 0 < len(row[4]) <= 16 for row in rows),
        "SMOKE_PROCESS_MONITOR_MALFORMED")
    table = {}
    for pid, ppid, pgid, kib, state in rows:
        smoke_require(int(pid) not in table, "SMOKE_PROCESS_MONITOR_DUPLICATE_PID")
        table[int(pid)] = dict(ppid=int(ppid), pgid=int(pgid), rss=int(kib)*1024, state=state)
    return table


class SmokeGroupRegistry:
    """Two barrier-ACKed groups; sampled identity, NOT PID-reuse attestation.

    Once ACKed, membership follows PGID even after reparenting. Observed-empty
    groups retire permanently; a reappearing number is never re-admitted/killed.
    Requests/ACKs are retained until the harness writes its terminal receipt.
    """
    def __init__(self, directory, nonce, pytest_pid):
        self.directory, self.nonce, self.pytest_pid = directory, nonce, pytest_pid
        self.identity = smoke_registry_identity(directory)
        self.records, self.retired_pgids, self.pytest_empty = {}, set(), False
        self.stage = "REGISTRY_CREATED"

    def capture_requests(self):
        """Capture ONLY fixed slots, before sampling ps; no later slot reread."""
        smoke_require(smoke_registry_identity(self.directory) == self.identity, "SMOKE_REGISTRY_REPLACED")
        allowed = {f"{slot}.{kind}.json" for slot in range(SMOKE_GROUP_CAP) for kind in ("request", "ack")}
        allowed |= {"."+name+".tmp" for name in allowed}
        with os.scandir(self.directory) as entries:
            for count, entry in enumerate(entries, 1):
                smoke_require(count <= len(allowed) and entry.name in allowed, "SMOKE_REGISTRY_ENTRY_CAP")
        captured = {}
        for slot in range(SMOKE_GROUP_CAP):
            try:
                captured[slot] = smoke_read(self.directory/f"{slot}.request.json")
            except FileNotFoundError:
                smoke_require(slot not in self.records, "SMOKE_REGISTRATION_REMOVED")
        return captured

    def sample(self, child, *, phase):
        smoke_require(phase in ("MAIN_ADMISSION", "CLEANUP_CUSTODY"), "SMOKE_SAMPLE_PHASE")
        self.stage = phase+"_CAPTURE_REQUESTS"
        captured = self.capture_requests()
        self.stage = phase+"_PROCESS_SAMPLE"
        table = smoke_process_table()
        self.stage = phase+"_VALIDATE_CAPTURED"
        self.update(captured, table, pytest_exited=child.poll() is not None, phase=phase)
        return table

    def update(self, captured, table, *, pytest_exited=False, phase="MAIN_ADMISSION"):
        # No request filesystem reads here. A request published after capture is
        # deferred to the next fresh ps sample, never checked against old rows.
        smoke_require(phase in ("MAIN_ADMISSION", "CLEANUP_CUSTODY"), "SMOKE_SAMPLE_PHASE")
        smoke_require(type(captured) is dict and set(captured) <= set(range(SMOKE_GROUP_CAP)),
                      "SMOKE_CAPTURED_SLOT_SCHEMA")
        smoke_require(smoke_registry_identity(self.directory) == self.identity, "SMOKE_REGISTRY_REPLACED")
        present = {row["pgid"] for row in table.values()}
        smoke_require(self.pytest_pid in table or pytest_exited, "SMOKE_ACTIVE_PYTEST_ROW_MISSING")
        if self.pytest_pid not in present:
            smoke_require(pytest_exited, "SMOKE_ACTIVE_PYTEST_GROUP_MISSING")
            self.pytest_empty = True
        smoke_require(not (self.pytest_empty and self.pytest_pid in present), "SMOKE_PYTEST_GROUP_REAPPEARED")
        for record in self.records.values():
            if record["pgid"] not in present:
                record["empty_confirmed"] = True
                self.retired_pgids.add(record["pgid"])
        smoke_require(not self.retired_pgids.intersection(present), "SMOKE_RETIRED_GROUP_REAPPEARED")
        for slot, request in sorted(captured.items()):
            self.stage = phase+"_VALIDATE_CAPTURED"
            sha = hashlib.sha256(smoke_encoded(request)).hexdigest()
            if slot in self.records:
                smoke_require(sha == self.records[slot]["registration_sha256"], "SMOKE_REGISTRATION_CHANGED")
                if self.records[slot]["acknowledged"]:
                    self.stage = phase+"_VERIFY_PRIOR_ACK"
                    smoke_require(smoke_encoded(smoke_read(self.directory/f"{slot}.ack.json")) ==
                                  smoke_encoded(self.records[slot]["ack"]), "SMOKE_ACK_CHANGED")
                continue
            smoke_require(type(request) is dict and set(request) ==
                {"schema", "nonce", "slot", "pytest_pid", "pid", "pgid"}
                and request["schema"] == SMOKE_SCHEMA and request["nonce"] == self.nonce
                and type(request["slot"]) is int and request["slot"] == slot
                and request["pytest_pid"] == self.pytest_pid
                and type(request["pid"]) is int and request["pid"] > 1
                and type(request["pgid"]) is int and request["pgid"] == request["pid"]
                and request["pgid"] != self.pytest_pid
                and request["pgid"] not in {record["pgid"] for record in self.records.values()},
                "SMOKE_REGISTRATION_BINDING")
            row = table.get(request["pid"])
            smoke_require(row is not None and row["ppid"] == self.pytest_pid
                          and row["pgid"] == request["pid"] and not row["state"].startswith("Z"),
                          "SMOKE_COORDINATOR_NOT_LIVE_DIRECT_SESSION_CHILD")
            observed = time.monotonic()
            ack = dict(**request, decision="ACK", harness_pid=os.getpid(), registration_sha256=sha,
                       ack_phase="MAIN_ADMISSION", ack_monotonic=observed)
            # Own the verified group before publishing ACK, so ACK-write failure
            # cannot lose custody of a child whose stdin might be released.
            self.records[slot] = dict(slot=slot, pytest_pid=self.pytest_pid, pgid=request["pgid"],
                registration_sha256=sha, acknowledged=False, empty_confirmed=False, ack=None,
                first_observed_phase=phase, first_observed_monotonic=observed,
                ack_phase=None, ack_monotonic=None, ack_publication_completed_monotonic=None)
            if phase == "MAIN_ADMISSION":
                self.stage = "MAIN_ADMISSION_PUBLISH_ACK"
                self.records[slot]["ack"] = ack
                smoke_publish(self.directory, f"{slot}.ack.json", ack)
                self.records[slot].update(acknowledged=True, ack_phase="MAIN_ADMISSION", ack_monotonic=observed,
                                         ack_publication_completed_monotonic=time.monotonic())
            # CLEANUP_CUSTODY keeps a verified pending group only for kill. It
            # never publishes ACK and can never turn a failed run into admission.

    def active_groups(self):
        return {record["pgid"] for record in self.records.values() if not record["empty_confirmed"]}

    def rss(self, table):
        tree = {self.pytest_pid}
        while True:
            extra = {pid for pid, row in table.items() if row["ppid"] in tree}-tree
            if not extra:
                break
            tree |= extra
        groups = self.active_groups() | ({self.pytest_pid} if not self.pytest_empty else set())
        measured = {pid for pid, row in table.items() if pid in tree or row["pgid"] in groups}
        return sum(table[pid]["rss"] for pid in measured)

    def kill_owned(self):
        groups = self.active_groups() | ({self.pytest_pid} if not self.pytest_empty else set())
        errors = []
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(smoke_failure(error, "CLEANUP_GROUP_KILL", "SMOKE_GROUP_KILL_FAILED", pgid=pgid))
        return errors

    def receipt(self):
        return dict(cap=SMOKE_GROUP_CAP, identity_scope="ACK_BARRIER_PS_SAMPLE_NOT_PID_REUSE_OR_ADVERSARIAL_ATTESTATION",
            pytest_group_empty_confirmed=self.pytest_empty,
            registrations=[{key: value for key, value in record.items() if key != "ack"}
                           for _, record in sorted(self.records.items())],
            all_owned_groups_empty_confirmed=self.pytest_empty and not self.active_groups())


def stop_smoke_groups(registry, child, stop):
    """Scan before every kill; one bounded confirmation loop, no blocking wait."""
    errors = []
    while True:
        child.poll()  # Reap without waiting; absent ps is NOT exit evidence.
        try:
            registry.sample(child, phase="CLEANUP_CUSTODY")
        except Exception as error:
            errors.append(smoke_failure(error, registry.stage))
        if child.poll() is None:
            try:
                child.kill()  # Independently retain ownership of our live direct child.
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(smoke_failure(error, "CLEANUP_DIRECT_CHILD_KILL", "SMOKE_DIRECT_CHILD_KILL_FAILED"))
        errors.extend(registry.kill_owned())
        if registry.pytest_empty and not registry.active_groups() and child.poll() is not None:
            break
        if time.monotonic() >= stop:
            break
        time.sleep(min(.02, max(0., stop-time.monotonic())))
    return dict(**registry.receipt(), cleanup_errors=errors[:8],
                requires_group_attention=not (registry.pytest_empty and not registry.active_groups()))


class SmokeCache:
    """Keep the private registry if empty-group custody cannot be confirmed."""
    def __init__(self):
        self.name = tempfile.mkdtemp(prefix="cad-guided-smoke-")
        self.keep = True  # Only confirmed terminal custody authorizes cleanup.

    def __enter__(self):
        return self.name

    def __exit__(self, *_):
        if not self.keep:
            shutil.rmtree(self.name)


def source_bindings(suite):
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in (*V1_SOURCE_BINDINGS.get(suite, ()), TESTS[suite],
                         "scripts/check_cad_guided_measurement.py")}


def main():
    global WALL_SECONDS, CPU_SECONDS, RSS_BYTES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*TESTS, "all"), default="all")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.suite in ("native_tolerance_probe_v1", "cad_sampling_native_v1"):
        WALL_SECONDS, CPU_SECONDS, RSS_BYTES = 20, 15, 512*1024**2
    if args.suite in ("native_trim_uncertainty_constructed_v1", "mesh_connected_native_v1",
                      "mesh_connected_native_dense_v1", "registration_stage_trace_v1",
                      "gold11_mesh_runner_v1", "gold11_mesh_guard_smoke_v1",
                      "cad_surface_v2", "triangle_surface_v2", "triangle_surface_v2_independent",
                      "native_validity_v2_source", "native_validity_v2_constructed",
                      "surface_registration_v2", "surface_registration_v2_matrix",
                      "mesh_offline_v2", "mesh_connected_surface_v2_source", "finite_gauss_newton_v3",
                      "finite_fine_comparison_v3_source", "finite_fine_comparison_v3",
                      "native_validity_v2_supervisor", "native_validity_v2_guard_smoke",
                      "cad_smoke_group_custody", "mesh_topology_v4", "plane_hypotheses_v4",
                      "mesh_plane_proposals_v4", "mesh_plane_capacity_v4_source", "plane_initialization_v4",
                      "mesh_connected_planes_v4_source", "native_context_coverage_v3",
                      "native_context_collection_v3_source", "native_context_collection_supervision_v3",
                      "noisy_mesh_planes_v5", "plane_search_v5", "noisy_mesh_capacity_v5_source",
                      "plane_initialization_v5", "noisy_mesh_evidence_v6", "noisy_mesh_evidence_v7", "report_envelope_v6",
                      "noisy_mesh_capacity_v6_source", "plane_initialization_v6"):
        WALL_SECONDS, CPU_SECONDS, RSS_BYTES = 60, 45, 1024**3
    if args.suite in ("mesh_connected_surface_v2", "mesh_connected_planes_v4"):
        WALL_SECONDS, CPU_SECONDS, RSS_BYTES = 180, 120, 2*1024**3
    if args.suite in ("mesh_plane_capacity_v4", "noisy_mesh_capacity_v5", "noisy_mesh_capacity_v6"):
        WALL_SECONDS, CPU_SECONDS, RSS_BYTES = 120, 90, 4*1024**3
    if args.suite == "native_context_collection_v3":
        WALL_SECONDS, CPU_SECONDS, RSS_BYTES = 120, 90, 1024**3
    if args.worker:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        sys.path.insert(0, str(ROOT / ".venv/lib/python3.11/site-packages"))
        if args.suite in ("native_features_v1", "native_features_v1_constructed", "cad_sampling_v1",
                          "native_tolerance_probe_v1", "cad_sampling_native_v1",
                          "native_trim_uncertainty_v1", "native_trim_uncertainty_constructed_v1",
                          "mesh_connected_native_v1", "mesh_connected_native_dense_v1",
                          "registration_stage_trace_v1", "cad_surface_v2",
                          "triangle_surface_v2", "triangle_surface_v2_independent",
                          "surface_registration_v2", "surface_registration_v2_matrix",
                          "mesh_offline_v2", "mesh_connected_surface_v2_source", "mesh_connected_surface_v2",
                          "finite_gauss_newton_v3", "finite_fine_comparison_v3_source", "finite_fine_comparison_v3",
                          "mesh_topology_v4", "plane_hypotheses_v4", "mesh_plane_proposals_v4",
                          "mesh_plane_capacity_v4_source", "mesh_plane_capacity_v4", "plane_initialization_v4",
                          "mesh_connected_planes_v4_source", "mesh_connected_planes_v4", "native_context_coverage_v3",
                          "noisy_mesh_planes_v5", "plane_search_v5",
                          "noisy_mesh_capacity_v5_source", "noisy_mesh_capacity_v5", "plane_initialization_v5",
                      "noisy_mesh_evidence_v6", "noisy_mesh_evidence_v7", "report_envelope_v6", "noisy_mesh_capacity_v6_source",
                      "noisy_mesh_capacity_v6", "plane_initialization_v6"):
            # These reviewed fixtures import the source-only backend namespace;
            # no backend package initializer exists in this checkout.
            sys.path.insert(0, str(ROOT))
        if args.suite == "native_features_v1_constructed":
            os.environ["SHERMAN_RUN_INVENTED_NATIVE_V1"] = "1"
        if args.suite == "native_trim_uncertainty_constructed_v1":
            os.environ["SHERMAN_RUN_NATIVE_TRIM_UNCERTAINTY_V1"] = "1"
        if args.suite == "mesh_connected_native_v1":
            os.environ["SHERMAN_RUN_INVENTED_MESH_CONNECTED_V1"] = "1"
        if args.suite == "mesh_connected_native_dense_v1":
            os.environ["SHERMAN_RUN_INVENTED_MESH_CONNECTED_DENSE_V1"] = "1"
        if args.suite == "registration_stage_trace_v1":
            os.environ["SHERMAN_RUN_INVENTED_REGISTRATION_STAGE_TRACE_V1"] = "1"
        if args.suite == "gold11_mesh_guard_smoke_v1":
            os.environ["SHERMAN_RUN_GUARDED_RUNTIME_SMOKE"] = "1"
        if args.suite == "native_validity_v2_constructed":
            os.environ["SHERMAN_RUN_V2_NATIVE_VALIDITY_CONSTRUCTED"] = "1"
        if args.suite == "surface_registration_v2_matrix":
            os.environ["SHERMAN_RUN_SURFACE_REGISTRATION_V2_MATRIX"] = "1"
        if args.suite == "mesh_connected_surface_v2":
            os.environ["SHERMAN_RUN_INVENTED_MESH_CONNECTED_SURFACE_V2"] = "1"
        if args.suite == "finite_fine_comparison_v3":
            os.environ["SHERMAN_RUN_FIXED_FINITE_FINE_COMPARISON_V3"] = "1"
        if args.suite == "native_validity_v2_guard_smoke":
            os.environ["SHERMAN_RUN_V2_NATIVE_VALIDITY_GUARD_SMOKE"] = "1"
        if args.suite == "mesh_plane_capacity_v4":
            os.environ["SHERMAN_RUN_MESH_PLANE_CAPACITY_V4"] = "1"
        if args.suite == "noisy_mesh_capacity_v5":
            os.environ["SHERMAN_RUN_NOISY_MESH_CAPACITY_V5"] = "1"
        if args.suite == "noisy_mesh_capacity_v6":
            os.environ["SHERMAN_RUN_NOISY_MESH_CAPACITY_V6"] = "1"
        if args.suite == "mesh_connected_planes_v4":
            os.environ["SHERMAN_RUN_INVENTED_MESH_CONNECTED_PLANES_V4"] = "1"
        if args.suite == "native_context_collection_v3":
            os.environ["SHERMAN_RUN_NATIVE_CONTEXT_COLLECTION_V3"] = "1"
        import pytest
        if args.suite in ("native_tolerance_probe_v1", "cad_sampling_native_v1"):
            namespace = runpy.run_path(str(ROOT / TESTS[args.suite]))
            function = ("run_invented_native_tolerance_probe" if args.suite == "native_tolerance_probe_v1"
                        else "run_constructed_native_sampling_checks_v1")
            report = namespace[function]()
            print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
            result = 0 if args.suite == "native_tolerance_probe_v1" or report["passed"] == 4 else 1
        else:
            selected = list(dict.fromkeys(TESTS.values())) if args.suite == "all" else [TESTS[args.suite]]
            result = pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:cacheprovider",
                                  "-q", "-s", *[str(ROOT / name) for name in selected]])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(json.dumps({"worker_cpu_seconds": usage.ru_utime + usage.ru_stime,
                          "worker_peak_rss_bytes": usage.ru_maxrss}), flush=True)
        return int(result)
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    retained = None
    if args.suite in ("mesh_input_v1", "whole_part_registration_v1", "native_features_v1",
                      "native_features_v1_constructed", "full_raw_measurement_v1",
                      "cad_sampling_v1", "mesh_offline_v1", "native_tolerance_probe_v1",
                      "cad_sampling_native_v1", "gold11_mesh_scoring_v1",
                      "native_trim_uncertainty_v1", "native_trim_uncertainty_constructed_v1",
                      "mesh_connected_native_v1", "mesh_connected_native_dense_v1", "registration_stage_trace_v1",
                      "gold11_mesh_runner_v1",
                      "gold11_mesh_guard_smoke_v1", "cad_surface_v2", "triangle_surface_v2",
                      "triangle_surface_v2_independent", "native_validity_v2_source",
                      "native_validity_v2_constructed", "surface_registration_v2",
                      "surface_registration_v2_matrix", "mesh_offline_v2",
                      "mesh_connected_surface_v2_source", "mesh_connected_surface_v2", "finite_gauss_newton_v3",
                      "finite_fine_comparison_v3_source", "finite_fine_comparison_v3",
                      "native_validity_v2_supervisor", "native_validity_v2_guard_smoke",
                      "cad_smoke_group_custody", "mesh_topology_v4", "plane_hypotheses_v4",
                      "mesh_plane_proposals_v4", "mesh_plane_capacity_v4_source", "mesh_plane_capacity_v4",
                      "plane_initialization_v4", "mesh_connected_planes_v4_source", "mesh_connected_planes_v4",
                      "native_context_coverage_v3", "native_context_collection_v3_source",
                      "native_context_collection_v3", "native_context_collection_supervision_v3",
                      "noisy_mesh_planes_v5", "plane_search_v5",
                      "noisy_mesh_capacity_v5_source", "noisy_mesh_capacity_v5", "plane_initialization_v5",
                      "noisy_mesh_evidence_v6", "noisy_mesh_evidence_v7", "report_envelope_v6", "noisy_mesh_capacity_v6_source",
                      "noisy_mesh_capacity_v6", "plane_initialization_v6"):
        receipt_root = ROOT / "docs/research/cad_engine_v1_synthetic_checks"
        receipt_root.mkdir(exist_ok=True)
        retained = Path(tempfile.mkdtemp(prefix=args.suite + "-", dir=receipt_root))
        frozen = {"harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "test_sha256": hashlib.sha256((ROOT / TESTS[args.suite]).read_bytes()).hexdigest(),
                  "suite": args.suite, "status": "STARTED_SYNTHETIC_CHECK_NO_OVERWRITE",
                  "limits": {"wall_seconds": WALL_SECONDS, "worker_cpu_seconds": CPU_SECONDS,
                             "sampled_group_rss_bytes": RSS_BYTES, "output_bytes": OUTPUT_BYTES}}
        frozen["source_sha256_before"] = source_bindings(args.suite)
        if args.suite == "mesh_connected_planes_v4":
            # Parent-owned fresh destination; never inherit a caller's path.
            artifact_directory = retained / "ARTIFACTS"
            env["SHERMAN_MESH_CONNECTED_PLANES_V4_ARTIFACT_DIRECTORY"] = str(artifact_directory)
            frozen["artifact_directory"] = str(artifact_directory)
            frozen["artifact_limits"] = {"case_count": 3, "per_case_json_bytes": 16*1024**2,
                                         "total_case_json_bytes": 48*1024**2}
        if args.suite == "native_context_collection_v3":
            artifact_directory = retained / "ARTIFACTS"
            env["SHERMAN_NATIVE_CONTEXT_COLLECTION_V3_ARTIFACT_DIRECTORY"] = str(artifact_directory)
            frozen["artifact_directory"] = str(artifact_directory)
            frozen["artifact_limits"] = {"case_count": 8, "per_case_json_bytes": 768*1024,
                                         "total_case_json_bytes": 6*1024**2}
            frozen["cleanup_reserve_seconds"] = 5
        with (retained / "STARTED.json").open("x") as stream:
            json.dump(frozen, stream, sort_keys=True)
    if args.suite == "native_context_collection_v3":
        # This new native route has separate deadline/cleanup ownership; do not
        # inherit the legacy unbounded reap or TemporaryDirectory cleanup path.
        namespace = runpy.run_path(str(ROOT / "scripts/v3_native_context_test_supervision.py"))
        receipt = namespace["supervise_native_context_test_v3"](
            [str(ROOT / ".venv/bin/python"), "-I", "-S", "-B", "-u", str(Path(__file__).resolve()),
             "--worker", "--suite", args.suite], cwd=ROOT, env=env, retained=retained)
        receipt.update(suite=args.suite, artifact_directory=frozen["artifact_directory"],
            artifact_limits=frozen["artifact_limits"], cleanup_reserve_seconds=5,
            source_sha256_after=source_bindings(args.suite), retained_attempt_directory=str(retained))
        receipt["source_before_after_consistent"] = receipt["source_sha256_after"] == frozen["source_sha256_before"]
        with (retained / "COMPLETED.json").open("x") as stream:
            json.dump(receipt, stream, sort_keys=True)
        print(json.dumps(receipt))
        return receipt["exit_code"] if (receipt["exit_code"] is not None and not receipt["budget_stop"]
            and receipt["source_before_after_consistent"] and not receipt["cleanup_error"]) else 1
    cache_scope = (SmokeCache() if args.suite == "native_validity_v2_guard_smoke"
                   else tempfile.TemporaryDirectory(prefix="cad-guided-tests-"))
    with cache_scope as cache:
        env.update(HOME=cache, TMPDIR=cache, XDG_CACHE_HOME=cache)
        registry_path = nonce = None
        if args.suite == "native_validity_v2_guard_smoke":
            registry_path = Path(cache).resolve()/"acknowledged-smoke-groups"
            registry_path.mkdir(mode=0o700)
            nonce = secrets.token_hex(32)
            env.update({SMOKE_REGISTRY_ENV: str(registry_path), SMOKE_NONCE_ENV: nonce})
        registry = SmokeGroupRegistry(registry_path, nonce, 0) if registry_path is not None else None
        started = time.monotonic()
        child = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-I", "-S", "-B", "-u", str(Path(__file__).resolve()),
             "--worker", "--suite", args.suite], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        if registry is not None:
            registry.pytest_pid = child.pid
        selector = None
        group_receipt = None
        peak_rss = forwarded = 0
        stopped = None
        primary_error = None
        retained_output = None
        try:
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            retained_output = (retained / "OUTPUT.txt").open("xb") if retained else None
            while True:
                smoke_finished = False
                if registry is not None:
                    registry.stage = "HARNESS_OUTPUT_FORWARD"
                for key, _ in selector.select(timeout=.02):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = max(0, OUTPUT_BYTES - forwarded)
                    sys.stdout.buffer.write(block[:remaining])
                    sys.stdout.buffer.flush()
                    if retained_output is not None:
                        retained_output.write(block[:remaining])
                        retained_output.flush()
                    forwarded += len(block)
                if registry is None and child.poll() is not None and not selector.get_map():
                    break
                try:
                    if registry is not None:
                        table = registry.sample(child, phase="MAIN_ADMISSION")
                        registry.stage = "MAIN_ADMISSION_RSS"
                        rss = registry.rss(table)
                        if child.poll() is not None and not selector.get_map():
                            if registry.active_groups():
                                stopped = "SMOKE_GROUP_SURVIVED_PYTEST"
                            smoke_finished = True
                    else:
                        sample = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss="],
                                          capture_output=True, text=True, timeout=2)
                        rows = [line.split() for line in sample.stdout.splitlines()]
                        if sample.returncode != 0 or not rows or any(
                            len(row) != 3 or not all(value.isdigit() for value in row)
                            for row in rows):
                            raise RuntimeError("RSS_MONITOR_UNAVAILABLE_OR_MALFORMED")
                        if len({int(row[0]) for row in rows}) != len(rows):
                            raise RuntimeError("RSS_MONITOR_DUPLICATE_PID")
                        group = [row for row in rows if int(row[1]) == child.pid]
                        if child.pid not in {int(row[0]) for row in group} and child.poll() is None:
                            raise RuntimeError("ACTIVE_CHILD_RSS_MISSING")
                        rss = sum(int(row[2]) * 1024 for row in group)
                except Exception as error:
                    if registry is not None:
                        primary_error = first_smoke_failure(primary_error, error, registry.stage)
                        stopped = "SMOKE_GROUP_CUSTODY_FAILURE"
                    elif child.poll() is None:
                        stopped = "RSS_MONITOR_FAILURE"
                    rss = 0
                peak_rss = max(peak_rss, rss)
                # Only this smoke reserves5s inside its60s limit for group kill
                # and sampled-empty confirmation. Other suites are unchanged.
                if time.monotonic() - started > WALL_SECONDS-(5 if registry is not None else 0):
                    stopped = "WALL_LIMIT"
                elif rss > RSS_BYTES:
                    stopped = "SAMPLED_RSS_LIMIT"
                elif forwarded > OUTPUT_BYTES:
                    stopped = "OUTPUT_LIMIT"
                if stopped:
                    if registry is not None:
                        break  # Single cleanup owner below, including all ACKed groups.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                    break
                if smoke_finished:
                    break
        except Exception as error:
            if registry is None:
                raise
            primary_error = first_smoke_failure(primary_error, error, registry.stage)
            stopped = stopped or "SMOKE_HARNESS_EXCEPTION"
        finally:
            if registry is not None:
                group_receipt = stop_smoke_groups(registry, child, min(started+WALL_SECONDS, time.monotonic()+5))
                cache_scope.keep = bool(group_receipt["requires_group_attention"] or group_receipt["cleanup_errors"])
                if group_receipt["requires_group_attention"] or group_receipt["cleanup_errors"]:
                    stopped = stopped or "SMOKE_GROUP_CLEANUP_UNCONFIRMED"
                if len(group_receipt["registrations"]) != SMOKE_GROUP_CAP or not all(
                        record["acknowledged"] and record["ack_phase"] == "MAIN_ADMISSION"
                        for record in group_receipt["registrations"]):
                    stopped = stopped or "SMOKE_EXPECTED_TWO_ACKNOWLEDGED_GROUPS"
            if selector is not None:
                selector.close()
            if retained_output is not None:
                retained_output.close()
            if registry is None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        receipt = {"suite": args.suite, "exit_code": child.returncode,
                          "parent_wall_seconds": time.monotonic() - started,
                          "sampled_group_peak_rss_bytes": peak_rss, "budget_stop": stopped,
                          "limits": {"wall_seconds": WALL_SECONDS, "worker_cpu_seconds": CPU_SECONDS,
                                     "sampled_group_rss_bytes": RSS_BYTES, "output_bytes": OUTPUT_BYTES}}
        if registry is not None:
            receipt["smoke_group_custody"] = group_receipt
            receipt["primary_error"] = primary_error
            receipt["rss_scope"] = "PYTEST_TREE_AND_GROUP_PLUS_ALL_ACKNOWLEDGED_SMOKE_GROUPS"
            receipt["private_cache_retained_for_custody"] = str(Path(cache).resolve()) if cache_scope.keep else None
        if retained is not None:
            if args.suite == "mesh_connected_planes_v4":
                receipt["artifact_directory"] = frozen["artifact_directory"]
                receipt["artifact_limits"] = frozen["artifact_limits"]
            receipt["source_sha256_after"] = source_bindings(args.suite)
            receipt["source_before_after_consistent"] = (
                receipt["source_sha256_after"] == frozen["source_sha256_before"])
            receipt["retained_attempt_directory"] = str(retained)
            receipt["output_sha256"] = hashlib.sha256((retained / "OUTPUT.txt").read_bytes()).hexdigest()
            with (retained / "COMPLETED.json").open("x") as stream:
                json.dump(receipt, stream, sort_keys=True)
        print(json.dumps(receipt))
        return child.returncode if not stopped and (retained is None or receipt["source_before_after_consistent"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
