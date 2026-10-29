#!/usr/bin/env python3
"""Orchestrate the professor-authorized PWTS representation preflight.

The runner contains no PWTS geometry.  It binds immutable inputs and source
files, launches one fresh CPU-only process at a time, enforces resource and
serialization caps, and records complete process evidence.  Semantic and
independent-validator executables are supplied explicitly and are never
imported by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


PRELIGHT_ID = "PWTS_REPRESENTATION_PREFLIGHT_V1"
EXPECTED_CONFIGURATION_SHA256 = "5d512e51911cff6ec4b44ce9bb17eead5852eb3647b15aaf10fd1dea490efd49"
VALID_STATUS = "PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET"
INVALID_STATUS = "INVALID_NEW_REPRESENTATION_PREFLIGHT_STOP"
EXPECTED_CASE_TOKENS = tuple(f"C{index:03d}" for index in range(1, 38))
PROCESS_ROLES = ("BASE_A", "BASE_B", "ORDER_PERMUTED", "INDEPENDENT_VALIDATOR")

PER_PROCESS_CPU_SECONDS = 900
PER_PROCESS_WALL_SECONDS = 1200
PER_PROCESS_PEAK_RSS_BYTES = 12_884_901_888
PER_PROCESS_SCRATCH_BYTES = 4_294_967_296
PER_CASE_SERIALIZED_BYTES = 100_663_296
AGGREGATE_CPU_SECONDS = 133_200
AGGREGATE_WALL_SECONDS = 177_600
AGGREGATE_SERIALIZED_BYTES = 2_147_483_648

PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS = 0.01
RLIMIT_AS_LIMITER_MODE = "RLIMIT_AS_PLUS_PROCESS_TREE_RSS_WATCHDOG"
DARWIN_WATCHDOG_LIMITER_MODE = "DARWIN_PROCESS_TREE_RSS_WATCHDOG"
UNSUPPORTED_LIMITER_MODE = "UNSUPPORTED_FAIL_CLOSED"

TABLE_ROW_CAPS = {
    "seed_primitive": 1_024,
    "trace_primitive": 1_024,
    "trace_node": 131_072,
    "strip_membership": 1_000_000,
    "certificate": 1_024,
}
FORBIDDEN_SERIALIZATION_KEYS = frozenset(
    {
        "x", "y", "z", "xyz", "coordinates", "raw_coordinates",
        "raw_points", "source_points", "sampled_points", "point_array",
        "neighborhood_array", "rendered_geometry", "source_point_subset",
    }
)
SAFE_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _atomic_write(path: Path, value: Any) -> None:
    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _source_manifest_content_sha(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    asserted = content.pop("input_manifest_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if asserted != observed:
        raise ValueError("input manifest internal SHA mismatch")
    return observed


def validate_source_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _read_json(path)
    if (
        payload.get("preflight_id") != PRELIGHT_ID
        or payload.get("kind") != "pwts_source_lock_v1"
        or payload.get("semantic_work_performed") is not False
        or payload.get("sampling_performed") is not False
        or payload.get("source_points_serialized") is not False
    ):
        raise ValueError("invalid or semantic PWTS source manifest")
    _source_manifest_content_sha(payload)
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("input manifest cases must be a list")
    by_token: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("input manifest case rows must be objects")
        row = dict(raw)
        token = str(row.get("case_token") or "")
        if token in by_token:
            raise ValueError("duplicate input-manifest case token")
        required = {
            "raw_path", "raw_byte_count", "raw_ply_sha256", "ply_header_sha256",
            "parsed_vertex_count", "finite_coordinate_count", "exact_duplicate_count",
            "ignored_property_name_digest", "canonical_coordinate_multiset_sha256",
        }
        if not required.issubset(row):
            raise ValueError(f"{token}: incomplete input source binding")
        by_token[token] = row
    if tuple(sorted(by_token)) != EXPECTED_CASE_TOKENS:
        raise ValueError("input manifest must contain exactly C001-C037")
    return payload, by_token


def validate_configuration(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_json(path)
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError(
            "unauthorized configuration hash: "
            f"expected {EXPECTED_CONFIGURATION_SHA256}, observed {observed}"
        )
    execution = payload.get("execution")
    if (
        payload.get("preflight_id") != PRELIGHT_ID
        or payload.get("case_tokens") != list(EXPECTED_CASE_TOKENS)
        or not isinstance(execution, Mapping)
        or execution.get("semantic_variants") != list(PROCESS_ROLES)
        or execution.get("workers") != 1
        or execution.get("cpu_only") is not True
        or execution.get("fresh_process_per_case_variant") is not True
        or execution.get("per_process_peak_rss_bytes") != PER_PROCESS_PEAK_RSS_BYTES
    ):
        raise ValueError("configuration execution contract mismatch")
    return payload, observed


def _check_no_forbidden_serialization(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_SERIALIZATION_KEYS:
                raise ValueError(f"forbidden source-point serialization key at {location}.{key}")
            _check_no_forbidden_serialization(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_no_forbidden_serialization(child, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite serialized value at {location}")


def validate_worker_payload(path: Path, *, case_token: str, role: str) -> dict[str, Any]:
    size = path.stat().st_size
    if size > PER_CASE_SERIALIZED_BYTES:
        raise ValueError(f"{case_token}/{role}: serialized payload cap exceeded")
    payload = _read_json(path)
    if (
        payload.get("preflight_id") != PRELIGHT_ID
        or payload.get("case_token") != case_token
        or payload.get("process_role") != role
        or payload.get("source_points_serialized") is not False
    ):
        raise ValueError(f"{case_token}/{role}: worker payload binding mismatch")
    _check_no_forbidden_serialization(payload)
    tables = payload.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError(f"{case_token}/{role}: tables object missing")
    for table, cap in TABLE_ROW_CAPS.items():
        rows = tables.get(table)
        if not isinstance(rows, list):
            raise ValueError(f"{case_token}/{role}: required table missing: {table}")
        if len(rows) > cap:
            raise ValueError(f"{case_token}/{role}: {table} cap exceeded before acceptance")
    return payload


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass(frozen=True)
class ProcessSpec:
    case_token: str
    role: str
    executable: str
    expected_executable_sha256: str
    expected_runner_sha256: str
    raw_path: str
    source_manifest_path: str
    expected_source_manifest_file_sha256: str
    expected_input_manifest_sha256: str
    configuration_path: str
    expected_configuration_file_sha256: str
    expected_configuration_sha256: str
    schema_path: str
    expected_schema_sha256: str
    output_dir: str
    scratch_dir: str


@dataclass(frozen=True)
class ProcessResult:
    case_token: str
    process_role: str
    return_code: int
    cpu_seconds: float
    wall_seconds: float
    process_tree_peak_rss_bytes: int
    scratch_bytes: int
    serialized_bytes: int
    limiter_mode: str
    rss_sampling_interval_seconds: float
    monitor_available: bool
    full_lifetime_accounting: bool
    reported_passing: bool
    rss_cap_exceeded: bool
    scratch_cap_exceeded: bool
    wall_cap_exceeded: bool
    payload_sha256: str | None
    stdout_sha256: str
    stderr_sha256: str
    success: bool
    failure_code: str | None


def build_process_specs(
    *, source_rows: Mapping[str, Mapping[str, Any]], input_manifest_path: Path,
    input_manifest_sha256: str, configuration_path: Path, schema_path: Path,
    semantic_executable: Path, validator_executable: Path, output_root: Path,
) -> list[ProcessSpec]:
    schema_sha = sha256_file(schema_path)
    specs: list[ProcessSpec] = []
    for token in EXPECTED_CASE_TOKENS:
        for role in PROCESS_ROLES:
            executable = validator_executable if role == "INDEPENDENT_VALIDATOR" else semantic_executable
            role_root = output_root / "processes" / token / role
            specs.append(ProcessSpec(
                case_token=token,
                role=role,
                executable=str(executable.resolve()),
                expected_executable_sha256=sha256_file(executable.resolve()),
                expected_runner_sha256=sha256_file(Path(__file__).resolve()),
                raw_path=str(Path(str(source_rows[token]["raw_path"])).resolve()),
                source_manifest_path=str(input_manifest_path.resolve()),
                expected_source_manifest_file_sha256=sha256_file(input_manifest_path.resolve()),
                expected_input_manifest_sha256=input_manifest_sha256,
                configuration_path=str(configuration_path.resolve()),
                expected_configuration_file_sha256=sha256_file(configuration_path.resolve()),
                expected_configuration_sha256=EXPECTED_CONFIGURATION_SHA256,
                schema_path=str(schema_path.resolve()),
                expected_schema_sha256=schema_sha,
                output_dir=str((role_root / "output").resolve()),
                scratch_dir=str((role_root / "scratch").resolve()),
            ))
    return specs


def _preexec_limits(*, limiter_mode: str | None = None) -> None:
    if limiter_mode is None:
        if sys.platform == "darwin":
            limiter_mode = DARWIN_WATCHDOG_LIMITER_MODE
        elif hasattr(resource, "RLIMIT_AS"):
            limiter_mode = RLIMIT_AS_LIMITER_MODE
        else:
            raise RuntimeError("no supported process memory limiter")
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (PER_PROCESS_CPU_SECONDS, PER_PROCESS_CPU_SECONDS))
    if limiter_mode == RLIMIT_AS_LIMITER_MODE:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (PER_PROCESS_PEAK_RSS_BYTES, PER_PROCESS_PEAK_RSS_BYTES),
        )
    elif limiter_mode != DARWIN_WATCHDOG_LIMITER_MODE:
        raise RuntimeError(f"unsupported limiter mode: {limiter_mode}")
    resource.setrlimit(resource.RLIMIT_FSIZE, (PER_CASE_SERIALIZED_BYTES, PER_CASE_SERIALIZED_BYTES))


def _read_process_table() -> dict[int, tuple[int, int, int]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,rss="],
        check=True, capture_output=True, text=True,
    )
    rows: dict[int, tuple[int, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 4:
            pid, ppid, pgid, rss_kib = map(int, fields)
            rows[pid] = (ppid, pgid, rss_kib)
    return rows


def _verify_process_tree_monitor() -> bool:
    if shutil.which("ps") is None or not hasattr(os, "killpg"):
        return False
    try:
        rows = _read_process_table()
        self_row = rows.get(os.getpid())
        return bool(
            self_row is not None
            and self_row[1] == os.getpgrp()
            and self_row[2] >= 0
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


@dataclass(frozen=True)
class LimiterContract:
    limiter_mode: str
    rss_sampling_interval_seconds: float
    monitor_available: bool
    failure_code: str | None


def _limiter_contract(
    *, platform_name: str | None = None,
    monitor_verified: bool | None = None,
    rlimit_as_supported: bool | None = None,
) -> LimiterContract:
    platform_name = sys.platform if platform_name is None else platform_name
    monitor_verified = (
        _verify_process_tree_monitor()
        if monitor_verified is None
        else monitor_verified
    )
    rlimit_as_supported = (
        hasattr(resource, "RLIMIT_AS")
        if rlimit_as_supported is None
        else rlimit_as_supported
    )
    if platform_name == "darwin":
        mode = DARWIN_WATCHDOG_LIMITER_MODE
    elif rlimit_as_supported:
        mode = RLIMIT_AS_LIMITER_MODE
    else:
        return LimiterContract(
            limiter_mode=UNSUPPORTED_LIMITER_MODE,
            rss_sampling_interval_seconds=PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS,
            monitor_available=bool(monitor_verified),
            failure_code="UNSUPPORTED_PROCESS_TREE_LIMITER",
        )
    return LimiterContract(
        limiter_mode=mode,
        rss_sampling_interval_seconds=PROCESS_TREE_RSS_SAMPLE_INTERVAL_SECONDS,
        monitor_available=bool(monitor_verified),
        failure_code=None if monitor_verified else "PROCESS_TREE_MONITOR_UNAVAILABLE",
    )


def _process_tree_sample(root_pid: int, expected_pgid: int) -> tuple[int, tuple[int, ...], bool]:
    rows = _read_process_table()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _pgid, _rss) in rows.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    live_descendants = {pid for pid in descendants if pid in rows}
    group_members = {pid for pid, row in rows.items() if row[1] == expected_pgid}
    live = tuple(sorted(live_descendants | group_members))
    escaped_group = any(rows[pid][1] != expected_pgid for pid in live_descendants)
    return sum(rows[pid][2] for pid in live) * 1024, live, escaped_group


def _kill_supervised_processes(pgid: int, observed_pids: Sequence[int]) -> None:
    # The process group is the primary containment boundary.  Kill it first;
    # individually kill observed descendants as a fail-closed backstop for a
    # child caught changing its process group between samples.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in reversed(observed_pids):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _failed_process_result(
    spec: ProcessSpec, *, failure_code: str, limiter: LimiterContract,
    stdout_path: Path, stderr_path: Path, started: float,
    before: resource.struct_rusage, return_code: int = -1,
    peak_rss: int = 0, rss_exceeded: bool = False,
    scratch_exceeded: bool = False, wall_exceeded: bool = False,
) -> ProcessResult:
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ProcessResult(
        case_token=spec.case_token,
        process_role=spec.role,
        return_code=return_code,
        cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
        wall_seconds=time.monotonic() - started,
        process_tree_peak_rss_bytes=peak_rss,
        scratch_bytes=_directory_bytes(Path(spec.scratch_dir)),
        serialized_bytes=0,
        limiter_mode=limiter.limiter_mode,
        rss_sampling_interval_seconds=limiter.rss_sampling_interval_seconds,
        monitor_available=limiter.monitor_available,
        full_lifetime_accounting=False,
        reported_passing=False,
        rss_cap_exceeded=rss_exceeded,
        scratch_cap_exceeded=scratch_exceeded,
        wall_cap_exceeded=wall_exceeded,
        payload_sha256=None,
        stdout_sha256=sha256_file(stdout_path),
        stderr_sha256=sha256_file(stderr_path),
        success=False,
        failure_code=failure_code,
    )


def execute_supervised(spec: ProcessSpec) -> ProcessResult:
    output_dir = Path(spec.output_dir)
    scratch_dir = Path(spec.scratch_dir)
    if output_dir.exists() or scratch_dir.exists():
        raise ValueError(f"fresh-process directories already exist: {spec.case_token}/{spec.role}")
    output_dir.mkdir(parents=True)
    scratch_dir.mkdir(parents=True)
    stdout_path = output_dir.parent / "stdout.log"
    stderr_path = output_dir.parent / "stderr.log"
    command = [
        sys.executable, spec.executable,
        "--role", spec.role,
        "--case-token", spec.case_token,
        "--raw-ply", spec.raw_path,
        "--source-manifest", spec.source_manifest_path,
        "--expected-source-manifest-file-sha256", spec.expected_source_manifest_file_sha256,
        "--expected-input-manifest-sha256", spec.expected_input_manifest_sha256,
        "--configuration", spec.configuration_path,
        "--expected-configuration-file-sha256", spec.expected_configuration_file_sha256,
        "--expected-configuration-sha256", spec.expected_configuration_sha256,
        "--schema", spec.schema_path,
        "--expected-schema-sha256", spec.expected_schema_sha256,
        "--expected-executable-sha256", spec.expected_executable_sha256,
        "--runner-source", str(Path(__file__).resolve()),
        "--expected-runner-sha256", spec.expected_runner_sha256,
        "--output-dir", spec.output_dir,
        "--scratch-dir", spec.scratch_dir,
    ]
    environment = {**os.environ, **SAFE_ENVIRONMENT, "TMPDIR": spec.scratch_dir}
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    peak_rss = 0
    rss_exceeded = scratch_exceeded = wall_exceeded = False
    limiter = _limiter_contract()
    monitor_available = limiter.monitor_available
    failure_code: str | None = None
    last_tree_pids: tuple[int, ...] = ()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        if limiter.failure_code is not None:
            stderr.write((limiter.failure_code + "\n").encode("ascii"))
            stderr.flush()
            return _failed_process_result(
                spec, failure_code=limiter.failure_code, limiter=limiter,
                stdout_path=stdout_path, stderr_path=stderr_path,
                started=started, before=before,
            )
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=environment,
                preexec_fn=partial(_preexec_limits, limiter_mode=limiter.limiter_mode),
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            stderr.write(
                (f"PROCESS_LAUNCH_FAILED:{type(exc).__name__}:{exc}\n").encode(
                    "utf-8", errors="replace"
                )
            )
            stderr.flush()
            return _failed_process_result(
                spec, failure_code="PROCESS_LAUNCH_FAILED", limiter=limiter,
                stdout_path=stdout_path, stderr_path=stderr_path,
                started=started, before=before,
            )
        pgid = process.pid
        root_observed = False
        try:
            while True:
                wall = time.monotonic() - started
                try:
                    rss, last_tree_pids, escaped_group = _process_tree_sample(process.pid, pgid)
                    peak_rss = max(peak_rss, rss)
                    root_observed = root_observed or process.pid in last_tree_pids
                    if escaped_group:
                        failure_code = "CHILD_ESCAPED_MONITORED_PROCESS_GROUP"
                except (OSError, subprocess.SubprocessError, ValueError):
                    monitor_available = False
                    failure_code = "PROCESS_TREE_MONITOR_UNAVAILABLE"
                if peak_rss > PER_PROCESS_PEAK_RSS_BYTES:
                    rss_exceeded = True
                    failure_code = "PROCESS_TREE_RSS_CAP_EXCEEDED"
                if failure_code is not None:
                    _kill_supervised_processes(pgid, last_tree_pids)
                    break

                return_code = process.poll()
                if return_code is not None:
                    try:
                        final_rss, final_tree_pids, final_escaped = _process_tree_sample(
                            process.pid, pgid
                        )
                        peak_rss = max(peak_rss, final_rss)
                        last_tree_pids = final_tree_pids
                        if final_escaped:
                            failure_code = "CHILD_ESCAPED_MONITORED_PROCESS_GROUP"
                        if peak_rss > PER_PROCESS_PEAK_RSS_BYTES:
                            rss_exceeded = True
                            failure_code = "PROCESS_TREE_RSS_CAP_EXCEEDED"
                    except (OSError, subprocess.SubprocessError, ValueError):
                        monitor_available = False
                        failure_code = "PROCESS_TREE_MONITOR_UNAVAILABLE"
                    if not root_observed:
                        failure_code = "PROCESS_TREE_ROOT_NOT_OBSERVED"
                    elif last_tree_pids and failure_code is None:
                        failure_code = "PROCESS_GROUP_SURVIVED_ROOT_PROCESS"
                    if failure_code is not None:
                        _kill_supervised_processes(pgid, last_tree_pids)
                    break

                scratch_bytes = _directory_bytes(scratch_dir)
                if scratch_bytes > PER_PROCESS_SCRATCH_BYTES:
                    scratch_exceeded = True
                    failure_code = "SCRATCH_CAP_EXCEEDED"
                elif wall > PER_PROCESS_WALL_SECONDS:
                    wall_exceeded = True
                    failure_code = "WALL_CAP_EXCEEDED"
                if failure_code:
                    _kill_supervised_processes(pgid, last_tree_pids)
                    break
                time.sleep(limiter.rss_sampling_interval_seconds)
            return_code = process.wait()
        finally:
            if process.poll() is None:
                _kill_supervised_processes(pgid, last_tree_pids)
                process.wait()
    wall_seconds = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    scratch_bytes = _directory_bytes(scratch_dir)
    payload_path = output_dir / "case_payload.json"
    serialized_bytes = payload_path.stat().st_size if payload_path.is_file() else 0
    payload_sha: str | None = None
    if failure_code is None and return_code != 0:
        failure_code = "WORKER_NONZERO_EXIT"
    if failure_code is None and not payload_path.is_file():
        failure_code = "WORKER_PAYLOAD_MISSING"
    if failure_code is None:
        try:
            validate_worker_payload(payload_path, case_token=spec.case_token, role=spec.role)
            payload_sha = sha256_file(payload_path)
        except (OSError, ValueError, json.JSONDecodeError):
            failure_code = "WORKER_PAYLOAD_INVALID"
    full_lifetime_accounting = bool(
        monitor_available
        and root_observed
        and failure_code not in {
            "PROCESS_TREE_MONITOR_UNAVAILABLE",
            "PROCESS_TREE_ROOT_NOT_OBSERVED",
            "PROCESS_GROUP_SURVIVED_ROOT_PROCESS",
        }
    )
    reported_passing = bool(
        failure_code is None
        and full_lifetime_accounting
        and cpu_seconds <= PER_PROCESS_CPU_SECONDS
        and wall_seconds <= PER_PROCESS_WALL_SECONDS
        and peak_rss <= PER_PROCESS_PEAK_RSS_BYTES
        and scratch_bytes <= PER_PROCESS_SCRATCH_BYTES
        and serialized_bytes <= PER_CASE_SERIALIZED_BYTES
    )
    return ProcessResult(
        case_token=spec.case_token, process_role=spec.role, return_code=return_code,
        cpu_seconds=cpu_seconds, wall_seconds=wall_seconds,
        process_tree_peak_rss_bytes=peak_rss, scratch_bytes=scratch_bytes,
        serialized_bytes=serialized_bytes, monitor_available=monitor_available,
        limiter_mode=limiter.limiter_mode,
        rss_sampling_interval_seconds=limiter.rss_sampling_interval_seconds,
        full_lifetime_accounting=full_lifetime_accounting,
        reported_passing=reported_passing,
        rss_cap_exceeded=rss_exceeded, scratch_cap_exceeded=scratch_exceeded,
        wall_cap_exceeded=wall_exceeded, payload_sha256=payload_sha,
        stdout_sha256=sha256_file(stdout_path), stderr_sha256=sha256_file(stderr_path),
        success=reported_passing, failure_code=failure_code,
    )


ProcessExecutor = Callable[[ProcessSpec], ProcessResult]


CHECKPOINT_KIND = "pwts_progress_checkpoint_v1"
PROCESS_RECEIPT_NAME = "process_result.json"


def _content_bound_payload(value: Mapping[str, Any], *, hash_key: str) -> dict[str, Any]:
    payload = dict(value)
    payload.pop(hash_key, None)
    payload[hash_key] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _validate_content_binding(
    value: Mapping[str, Any], *, hash_key: str, label: str,
) -> None:
    content = dict(value)
    asserted = content.pop(hash_key, None)
    observed = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if asserted != observed:
        raise ValueError(f"{label} internal SHA mismatch")


def _process_receipt_path(spec: ProcessSpec) -> Path:
    return Path(spec.output_dir).parent / PROCESS_RECEIPT_NAME


def _write_process_receipt(spec: ProcessSpec, result: ProcessResult) -> None:
    payload = _content_bound_payload(
        {
            "schema_version": 1,
            "preflight_id": PRELIGHT_ID,
            "kind": "pwts_process_result_receipt_v1",
            "spec_index_binding": {
                "case_token": spec.case_token,
                "process_role": spec.role,
            },
            "resource_row": asdict(result),
        },
        hash_key="receipt_sha256",
    )
    _atomic_write(_process_receipt_path(spec), payload)


def _checkpoint_payload(
    *, execution_lock: Mapping[str, Any], process_results: Sequence[ProcessResult],
    next_spec_index: int, terminal_failure_code: str | None = None,
    orchestration_result_sha256: str | None = None,
) -> dict[str, Any]:
    return _content_bound_payload(
        {
            "schema_version": 1,
            "preflight_id": PRELIGHT_ID,
            "kind": CHECKPOINT_KIND,
            "execution_lock_sha256": execution_lock["execution_lock_sha256"],
            "immutable_source_sha256": execution_lock["source_sha256"],
            "raw_source_sha256": execution_lock["raw_source_sha256"],
            "completed_process_results": [asdict(row) for row in process_results],
            "next_spec_index": next_spec_index,
            "terminal_failure_code": terminal_failure_code,
            "orchestration_result_sha256": orchestration_result_sha256,
        },
        hash_key="checkpoint_sha256",
    )


def _write_checkpoint(
    path: Path, *, execution_lock: Mapping[str, Any],
    process_results: Sequence[ProcessResult], next_spec_index: int,
    terminal_failure_code: str | None = None,
    orchestration_result_sha256: str | None = None,
) -> None:
    _atomic_write(
        path,
        _checkpoint_payload(
            execution_lock=execution_lock,
            process_results=process_results,
            next_spec_index=next_spec_index,
            terminal_failure_code=terminal_failure_code,
            orchestration_result_sha256=orchestration_result_sha256,
        ),
    )


def _execution_lock(
    *, input_manifest_path: Path, configuration_path: Path, schema_path: Path,
    semantic_executable: Path, validator_executable: Path, tamper_test_source: Path,
) -> dict[str, Any]:
    limiter = _limiter_contract()
    if limiter.failure_code is not None:
        raise ValueError(limiter.failure_code)
    sources = {
        "runner_source": Path(__file__).resolve(),
        "semantic_core_source": semantic_executable.resolve(),
        "independent_validator_source": validator_executable.resolve(),
        "tamper_test_source": tamper_test_source.resolve(),
        "schema": schema_path.resolve(),
        "configuration": configuration_path.resolve(),
        "input_manifest": input_manifest_path.resolve(),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    source_sha256 = {label: sha256_file(path) for label, path in sources.items()}
    source_path = {label: str(path) for label, path in sources.items()}
    source_manifest, source_rows = validate_source_manifest(input_manifest_path)
    raw_source_sha256: dict[str, str] = {}
    for token in EXPECTED_CASE_TOKENS:
        raw_path = Path(str(source_rows[token]["raw_path"])).resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing raw source for {token}: {raw_path}")
        observed = sha256_file(raw_path)
        if observed != source_rows[token]["raw_ply_sha256"]:
            raise ValueError(f"{token}: raw source SHA mismatch")
        raw_source_sha256[token] = observed
    lock = {
        "preflight_id": PRELIGHT_ID,
        "kind": "pwts_execution_lock_v1",
        "expected_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "input_manifest_sha256": source_manifest["input_manifest_sha256"],
        "source_sha256": source_sha256,
        "source_path": source_path,
        "raw_source_sha256": raw_source_sha256,
        "limiter_mode": limiter.limiter_mode,
        "rss_sampling_interval_seconds": limiter.rss_sampling_interval_seconds,
        "process_tree_peak_rss_cap_bytes": PER_PROCESS_PEAK_RSS_BYTES,
        "process_tree_monitor_verified": limiter.monitor_available,
    }
    return lock


def _bind_execution_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    return _content_bound_payload(lock, hash_key="execution_lock_sha256")


def _validate_completed_process(
    *, spec: ProcessSpec, raw_row: Mapping[str, Any], spec_index: int,
) -> ProcessResult:
    try:
        result = ProcessResult(**dict(raw_row))
    except TypeError as exc:
        raise ValueError(f"checkpoint resource row {spec_index} schema mismatch") from exc
    if asdict(result) != dict(raw_row):
        raise ValueError(f"checkpoint resource row {spec_index} is not byte-exact")
    if (
        result.case_token != spec.case_token
        or result.process_role != spec.role
        or not result.success
        or not result.reported_passing
        or result.failure_code is not None
    ):
        raise ValueError(f"checkpoint process order/status mismatch at index {spec_index}")

    receipt = _read_json(_process_receipt_path(spec))
    _validate_content_binding(
        receipt, hash_key="receipt_sha256", label=f"process receipt {spec_index}"
    )
    if (
        receipt.get("preflight_id") != PRELIGHT_ID
        or receipt.get("kind") != "pwts_process_result_receipt_v1"
        or receipt.get("spec_index_binding") != {
            "case_token": spec.case_token, "process_role": spec.role,
        }
        or canonical_json_bytes(receipt.get("resource_row")) != canonical_json_bytes(raw_row)
    ):
        raise ValueError(f"checkpoint resource row/receipt mismatch at index {spec_index}")

    output_dir = Path(spec.output_dir)
    payload_path = output_dir / "case_payload.json"
    stdout_path = output_dir.parent / "stdout.log"
    stderr_path = output_dir.parent / "stderr.log"
    if not all(path.is_file() for path in (payload_path, stdout_path, stderr_path)):
        raise ValueError(f"checkpoint evidence file missing at index {spec_index}")
    if (
        result.payload_sha256 is None
        or sha256_file(payload_path) != result.payload_sha256
        or sha256_file(stdout_path) != result.stdout_sha256
        or sha256_file(stderr_path) != result.stderr_sha256
        or payload_path.stat().st_size != result.serialized_bytes
    ):
        raise ValueError(f"checkpoint evidence hash/size mismatch at index {spec_index}")
    validate_worker_payload(
        payload_path, case_token=spec.case_token, role=spec.role,
    )
    return result


def _load_and_validate_checkpoint(
    *, checkpoint_path: Path, execution_lock: Mapping[str, Any],
    specs: Sequence[ProcessSpec],
) -> tuple[list[ProcessResult], int, str | None, str | None]:
    checkpoint = _read_json(checkpoint_path)
    _validate_content_binding(
        checkpoint, hash_key="checkpoint_sha256", label="progress checkpoint"
    )
    if (
        checkpoint.get("preflight_id") != PRELIGHT_ID
        or checkpoint.get("kind") != CHECKPOINT_KIND
        or checkpoint.get("execution_lock_sha256") != execution_lock["execution_lock_sha256"]
        or canonical_json_bytes(checkpoint.get("immutable_source_sha256"))
        != canonical_json_bytes(execution_lock["source_sha256"])
        or canonical_json_bytes(checkpoint.get("raw_source_sha256"))
        != canonical_json_bytes(execution_lock["raw_source_sha256"])
    ):
        raise ValueError("checkpoint execution/source lock mismatch")
    rows = checkpoint.get("completed_process_results")
    next_index = checkpoint.get("next_spec_index")
    if (
        not isinstance(rows, list)
        or not isinstance(next_index, int)
        or isinstance(next_index, bool)
        or next_index != len(rows)
        or not 0 <= next_index <= len(specs)
    ):
        raise ValueError("checkpoint contains a process gap or invalid next index")
    results: list[ProcessResult] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"checkpoint resource row {index} is not an object")
        results.append(
            _validate_completed_process(
                spec=specs[index], raw_row=raw, spec_index=index,
            )
        )
    terminal_failure = checkpoint.get("terminal_failure_code")
    orchestration_sha = checkpoint.get("orchestration_result_sha256")
    if terminal_failure is not None and not isinstance(terminal_failure, str):
        raise ValueError("checkpoint terminal failure code is invalid")
    if orchestration_sha is not None and not isinstance(orchestration_sha, str):
        raise ValueError("checkpoint orchestration result SHA is invalid")
    return results, next_index, terminal_failure, orchestration_sha


def _validate_execution_lock_files(execution_lock: Mapping[str, Any]) -> None:
    paths = execution_lock.get("source_path")
    hashes = execution_lock.get("source_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("execution lock source maps are invalid")
    if set(paths) != set(hashes):
        raise ValueError("execution lock source maps differ")
    for label, raw_path in paths.items():
        path = Path(str(raw_path))
        if not path.is_file() or sha256_file(path) != hashes[label]:
            raise ValueError(f"locked source changed: {label}")
    manifest = _read_json(Path(str(paths["input_manifest"])))
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise ValueError("locked input manifest cases are invalid")
    raw_by_token = execution_lock.get("raw_source_sha256")
    if not isinstance(raw_by_token, Mapping):
        raise ValueError("execution lock raw-source map is invalid")
    manifest_by_token = {
        str(row.get("case_token")): row for row in rows if isinstance(row, Mapping)
    }
    if set(raw_by_token) != set(EXPECTED_CASE_TOKENS):
        raise ValueError("execution lock raw-source token set differs")
    for token in EXPECTED_CASE_TOKENS:
        row = manifest_by_token.get(token)
        if not isinstance(row, Mapping):
            raise ValueError(f"locked source row missing: {token}")
        raw_path = Path(str(row.get("raw_path"))).resolve()
        if not raw_path.is_file() or sha256_file(raw_path) != raw_by_token[token]:
            raise ValueError(f"locked raw source changed: {token}")


def run_preflight(
    *, input_manifest_path: Path, configuration_path: Path, schema_path: Path,
    semantic_executable: Path, validator_executable: Path, tamper_test_source: Path,
    output_root: Path, process_executor: ProcessExecutor = execute_supervised,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failure_code: str | None = None
    observed_configuration_sha: str | None = None
    process_results: list[ProcessResult] = []
    completed_results: list[ProcessResult] = []
    execution_lock: dict[str, Any] | None = None
    checkpoint_path = output_root / "progress_checkpoint.json"
    checkpoint_validated = False
    next_spec_index = 0
    try:
        _configuration, observed_configuration_sha = validate_configuration(configuration_path)
        source_manifest, source_rows = validate_source_manifest(input_manifest_path)
        execution_lock = _execution_lock(
            input_manifest_path=input_manifest_path, configuration_path=configuration_path,
            schema_path=schema_path, semantic_executable=semantic_executable,
            validator_executable=validator_executable, tamper_test_source=tamper_test_source,
        )
        execution_lock["observed_configuration_sha256"] = observed_configuration_sha
        execution_lock["configuration_hash_match"] = True
        execution_lock = _bind_execution_lock(execution_lock)
        lock_path = output_root / "execution_lock.json"
        if lock_path.exists():
            stored_lock = _read_json(lock_path)
            _validate_content_binding(
                stored_lock, hash_key="execution_lock_sha256", label="execution lock"
            )
            if canonical_json_bytes(stored_lock) != canonical_json_bytes(execution_lock):
                raise ValueError("existing execution lock differs from current immutable sources")
        elif checkpoint_path.exists():
            raise ValueError("checkpoint exists without its execution lock")
        else:
            _atomic_write(lock_path, execution_lock)
        specs = build_process_specs(
            source_rows=source_rows, input_manifest_path=input_manifest_path,
            input_manifest_sha256=str(source_manifest["input_manifest_sha256"]),
            configuration_path=configuration_path, schema_path=schema_path,
            semantic_executable=semantic_executable, validator_executable=validator_executable,
            output_root=output_root,
        )
        _validate_execution_lock_files(execution_lock)
        if checkpoint_path.exists():
            (
                completed_results, next_spec_index, prior_terminal_failure,
                bound_orchestration_sha,
            ) = _load_and_validate_checkpoint(
                checkpoint_path=checkpoint_path, execution_lock=execution_lock, specs=specs,
            )
            checkpoint_validated = True
            process_results = list(completed_results)
            orchestration_path = output_root / "orchestration_result.json"
            if bound_orchestration_sha is not None:
                if (
                    not orchestration_path.is_file()
                    or sha256_file(orchestration_path) != bound_orchestration_sha
                ):
                    raise ValueError("checkpoint-bound orchestration result differs")
                authoritative = _read_json(orchestration_path)
                if prior_terminal_failure is not None or next_spec_index == len(specs):
                    return authoritative
            if prior_terminal_failure is not None:
                failure_code = f"PREVIOUS_TERMINAL_FAILURE:{prior_terminal_failure}"
        else:
            # Establish a crash-safe zero-progress boundary before role 0.  This
            # also makes a first-role terminal failure permanently non-retryable.
            _write_checkpoint(
                checkpoint_path, execution_lock=execution_lock,
                process_results=(), next_spec_index=0,
            )
            checkpoint_validated = True

        for spec_index in range(next_spec_index, len(specs)):
            if failure_code is not None:
                break
            spec = specs[spec_index]  # Normative workers=1: intentionally sequential.
            try:
                _validate_execution_lock_files(execution_lock)
            except (OSError, ValueError, json.JSONDecodeError):
                failure_code = "LOCKED_SOURCE_CHANGED_BEFORE_PROCESS_START"
                break
            if Path(spec.output_dir).exists() or Path(spec.scratch_dir).exists():
                failure_code = "UNSTARTED_PROCESS_DIRECTORY_NOT_FRESH"
                break
            result = process_executor(spec)
            if not isinstance(result, ProcessResult):
                raise ValueError("process executor returned an invalid resource row")
            process_results.append(result)
            if not result.success:
                failure_code = result.failure_code or "PROCESS_FAILED"
                break
            _write_process_receipt(spec, result)
            # Validate the just-finished result with the same byte-exact checks used
            # during resume before committing it as completed progress.
            validated_result = _validate_completed_process(
                spec=spec, raw_row=asdict(result), spec_index=spec_index,
            )
            completed_results.append(validated_result)
            next_spec_index = spec_index + 1
            _write_checkpoint(
                checkpoint_path, execution_lock=execution_lock,
                process_results=completed_results, next_spec_index=next_spec_index,
            )
            checkpoint_validated = True
            if sum(row.cpu_seconds for row in completed_results) > AGGREGATE_CPU_SECONDS:
                failure_code = "AGGREGATE_CPU_CAP_EXCEEDED"
                break
            if sum(row.wall_seconds for row in completed_results) > AGGREGATE_WALL_SECONDS:
                failure_code = "AGGREGATE_WALL_CAP_EXCEEDED"
                break
            if sum(row.serialized_bytes for row in completed_results) > AGGREGATE_SERIALIZED_BYTES:
                failure_code = "AGGREGATE_SERIALIZATION_CAP_EXCEEDED"
                break
            if sum(
                row.serialized_bytes for row in completed_results
                if row.case_token == result.case_token
            ) > PER_CASE_SERIALIZED_BYTES:
                failure_code = "PER_CASE_SERIALIZATION_CAP_EXCEEDED"
                break
        if failure_code is None and len(completed_results) != len(specs):
            failure_code = "PROCESS_MATRIX_INCOMPLETE"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failure_code = "CHECKPOINT_VALIDATION_FAILED" if checkpoint_path.exists() else type(exc).__name__.upper()
        failure_detail = str(exc)
    else:
        failure_detail = None
    result_payload = {
        "schema_version": 1,
        "preflight_id": PRELIGHT_ID,
        "kind": "pwts_orchestration_result_v1",
        "terminal_status": INVALID_STATUS,
        "failure_code": failure_code or "AWAITING_INDEPENDENT_VALIDATION_AND_TAMPER_SUITE",
        "failure_detail": failure_detail,
        "expected_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "observed_configuration_sha256": observed_configuration_sha,
        "configuration_hash_match": observed_configuration_sha == EXPECTED_CONFIGURATION_SHA256,
        "semantic_process_started": bool(process_results),
        "workers": 1,
        "cpu_only": True,
        "fresh_process_per_case_variant": True,
        "process_role_order": list(PROCESS_ROLES),
        "expected_process_count": len(EXPECTED_CASE_TOKENS) * len(PROCESS_ROLES),
        "completed_process_count": len(process_results),
        "resource_rows": [asdict(row) for row in process_results],
        "aggregate_cpu_seconds": sum(row.cpu_seconds for row in process_results),
        "aggregate_wall_seconds": sum(row.wall_seconds for row in process_results),
        "aggregate_serialized_bytes": sum(row.serialized_bytes for row in process_results),
        "source_points_serialized": False,
        "gold_or_truth_consumed": False,
        "runtime_or_production_authorized": False,
        "next_professor_packet_authorized": False,
    }
    # The runner never grants the valid terminal status.  Only independent
    # evidence assembly after semantic equivalence and tamper checks may do so.
    orchestration_path = output_root / "orchestration_result.json"
    if execution_lock is not None and checkpoint_validated and failure_code is not None:
        _write_checkpoint(
            checkpoint_path, execution_lock=execution_lock,
            process_results=completed_results, next_spec_index=next_spec_index,
            terminal_failure_code=failure_code,
        )
    preserve_authoritative_result = orchestration_path.exists() and not checkpoint_validated
    if not preserve_authoritative_result:
        _atomic_write(orchestration_path, result_payload)
    if execution_lock is not None and checkpoint_validated and not preserve_authoritative_result:
        _write_checkpoint(
            checkpoint_path, execution_lock=execution_lock,
            process_results=completed_results, next_spec_index=next_spec_index,
            terminal_failure_code=failure_code,
            orchestration_result_sha256=sha256_file(orchestration_path),
        )
    return result_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--semantic-executable", required=True, type=Path)
    parser.add_argument("--validator-executable", required=True, type=Path)
    parser.add_argument("--tamper-test-source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_preflight(
        input_manifest_path=args.input_manifest.resolve(),
        configuration_path=args.configuration.resolve(), schema_path=args.schema.resolve(),
        semantic_executable=args.semantic_executable.resolve(),
        validator_executable=args.validator_executable.resolve(),
        tamper_test_source=args.tamper_test_source.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0 if result["failure_code"] == "AWAITING_INDEPENDENT_VALIDATION_AND_TAMPER_SUITE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
