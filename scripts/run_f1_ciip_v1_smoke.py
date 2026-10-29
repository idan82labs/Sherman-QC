#!/usr/bin/env python3
"""Prepare and execute the single user-authorized frozen V7 gold-free smoke."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_ciip_v1.config import (  # noqa: E402
    canonical_json_bytes,
    load_frozen_v7,
    select_smoke_source_by_content,
    validate_frozen_allocation_plan,
)
from research.f1_ciip_v1.container import (  # noqa: E402
    METADATA_REGION_BYTES,
    decode_container,
    encode_metadata_region,
    validate_container,
)
from research.f1_ciip_v1.execution_lock import (  # noqa: E402
    CLAIM_RELATIVE,
    LOCK_RELATIVE,
    NATIVE_BUILD_RELATIVE,
    OUTPUT_RELATIVE,
    ROLE_ORDER,
    SCRATCH_RELATIVE,
    code_record,
    dependency_record,
    hash_open_regular_source,
    load_execution_lock,
    prepare_execution_lock,
    sha256_file,
    write_new_execution_lock,
)
from research.f1_ciip_v1.execution_result import validate_execution_result  # noqa: E402
LOCK_PATH = ROOT / LOCK_RELATIVE
CLAIM_PATH = ROOT / CLAIM_RELATIVE
NATIVE_BUILD_DIR = ROOT / NATIVE_BUILD_RELATIVE
RESULT_FILENAME = "result.json"
PASS_TOKEN = "CII_PERSISTENCE_PREFLIGHT_VALID_NEXT_PROFESSOR_PACKET"
STOP_TOKEN = "INVALID_CII_PERSISTENCE_PREFLIGHT_STOP"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_FORBIDDEN_INDEPENDENT_MODULES = frozenset({
    "research.f1_ciip_v1.primary_parser",
    "research.f1_ciip_v1.partition",
    "research.f1_ciip_v1.fields",
    "research.f1_ciip_v1.graph",
    "research.f1_ciip_v1.semantic_core",
    "research.f1_ciip_v1.native_worker",
})


def _code_paths() -> tuple[Path, ...]:
    paths = [path for path in (ROOT / "research/f1_ciip_v1").rglob("*") if path.is_file() and path.suffix in {".py", ".cpp"}]
    paths.append(Path(__file__).resolve())
    return tuple(paths)


def _native_build_from_lock(lock: Mapping[str, Any], role: str = "BASE_A") -> Any:
    from research.f1_ciip_v1.native_worker import NativeBuild

    row = lock["native_binaries"][role]
    manifest = row["build_manifest"]
    build = NativeBuild(
        library_path=ROOT / row["path"],
        compiler_path=str(manifest["compiler_path"]),
        compiler_version=str(manifest["compiler_version"]),
        compiler_sha256=str(manifest["compiler_sha256"]),
        flags=tuple(str(item) for item in manifest["flags"]),
        source_sha256=str(manifest["source_sha256"]),
        binary_sha256=str(manifest["binary_sha256"]),
    )
    if build.binary_sha256 != row["sha256"] or sha256_file(build.library_path)[0] != row["sha256"]:
        raise RuntimeError("native binary lock mismatch")
    return build


def _validate_pregeometry_gates(lock: Mapping[str, Any]) -> None:
    """Run all source-free role gates before live source hashing or parsing."""

    from research.f1_ciip_v1.identifiers import validate_v7_known_answer_vectors
    from research.f1_ciip_v1.independent_semantic_rebuild import (
        validate_independent_known_answer_vectors,
    )
    from research.f1_ciip_v1.native_worker import NativeWorker

    frozen = load_frozen_v7()
    validate_frozen_allocation_plan(frozen)
    validate_v7_known_answer_vectors()
    validate_independent_known_answer_vectors()
    for role in ROLE_ORDER[:3]:
        NativeWorker(_native_build_from_lock(lock, role)).validate_known_answers()


def _write_new_durable(path: Path, raw: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode)
    try:
        written = 0
        while written < len(raw):
            count = os.write(fd, raw[written:])
            if count <= 0:
                raise OSError("durable write made no progress")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _create_execution_claim(lock: Mapping[str, Any]) -> str:
    capability = os.urandom(32).hex()
    claim = {
        "execution_lock_sha256": lock["execution_lock_sha256"],
        "capability_sha256": hashlib.sha256(bytes.fromhex(capability)).hexdigest(),
        "claim_state": "CONSUMED_BEFORE_FIRST_WORKER_LAUNCH",
        "one_shot": True,
        "rerun_authorized": False,
    }
    _write_new_durable(CLAIM_PATH, canonical_json_bytes(claim) + b"\n")
    return capability


def _validate_worker_capability(lock: Mapping[str, Any], role: str, role_dir: Path) -> None:
    raw = CLAIM_PATH.read_bytes()
    claim = json.loads(raw.decode("ascii"))
    if canonical_json_bytes(claim) + b"\n" != raw:
        raise RuntimeError("execution claim is not canonical")
    capability = os.environ.pop("CIIP_EXECUTION_CAPABILITY", "")
    try:
        capability_hash = hashlib.sha256(bytes.fromhex(capability)).hexdigest()
    except ValueError as exc:
        raise RuntimeError("worker capability is malformed") from exc
    if (
        len(capability) != 64
        or claim.get("capability_sha256") != capability_hash
        or claim.get("execution_lock_sha256") != lock["execution_lock_sha256"]
        or claim.get("claim_state") != "CONSUMED_BEFORE_FIRST_WORKER_LAUNCH"
    ):
        raise RuntimeError("worker capability does not match consumed execution claim")
    expected_role_dir = (ROOT / SCRATCH_RELATIVE / role).resolve()
    if role_dir.resolve() != expected_role_dir:
        raise RuntimeError("worker role directory escaped exact one-shot scratch path")


def prepare_lock() -> None:
    from research.f1_ciip_v1.identifiers import validate_v7_known_answer_vectors
    from research.f1_ciip_v1.independent_semantic_rebuild import (
        validate_independent_known_answer_vectors,
    )
    from research.f1_ciip_v1.native_worker import NativeWorker, build_native_worker

    if LOCK_PATH.exists() or CLAIM_PATH.exists():
        raise RuntimeError("immutable execution lock already exists")
    if (ROOT / OUTPUT_RELATIVE).exists() or (ROOT / SCRATCH_RELATIVE).exists():
        raise RuntimeError("reserved output or scratch path already exists")
    if NATIVE_BUILD_DIR.exists():
        raise RuntimeError("reserved correction-v4 native build path already exists")
    build = build_native_worker(NATIVE_BUILD_DIR)
    frozen = load_frozen_v7()
    validate_frozen_allocation_plan(frozen)
    validate_v7_known_answer_vectors()
    validate_independent_known_answer_vectors()
    NativeWorker(build).validate_known_answers()
    lock = prepare_execution_lock(
        root=ROOT,
        frozen=frozen,
        clang=Path(build.compiler_path),
        code_paths=_code_paths(),
        native_binaries={role: build.library_path for role in ROLE_ORDER[:3]},
        native_binary_manifests={role: build.manifest() for role in ROLE_ORDER[:3]},
    )
    write_new_execution_lock(LOCK_PATH, lock)
    print(canonical_json_bytes({
        "execution_lock": str(LOCK_PATH),
        "execution_lock_sha256": lock["execution_lock_sha256"],
        "native_binary_sha256": build.binary_sha256,
        "status": "PREEXECUTION_LOCK_READY",
    }).decode("ascii"))


def _validate_live_lock(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    if (ROOT / OUTPUT_RELATIVE).exists() or (ROOT / SCRATCH_RELATIVE).exists():
        raise RuntimeError("one-shot destination already exists; rerun is forbidden")
    frozen = load_frozen_v7()
    selected = select_smoke_source_by_content(frozen)
    observed_sha, observed_size = hash_open_regular_source(Path(str(selected["raw_path"])))
    expected_source = {
        "raw_byte_count": observed_size,
        "raw_ply_sha256": observed_sha,
        "ply_header_byte_count": selected["ply_header_byte_count"],
        "ply_header_sha256": selected["ply_header_sha256"],
        "ply_format": selected["ply_format"],
        "parsed_vertex_count": selected["parsed_vertex_count"],
        "finite_coordinate_count": selected["finite_coordinate_count"],
        "exact_duplicate_count": selected["exact_duplicate_count"],
        "ignored_property_name_count": selected["ignored_property_name_count"],
        "ignored_property_name_digest": selected["ignored_property_name_digest"],
        "canonical_coordinate_multiset_digest": selected["canonical_coordinate_multiset_digest"],
        "selection_rule": "UNIQUE_LEXICOGRAPHIC_MINIMUM_RAW_PLY_SHA256_OVER_EXACT_37_ROWS",
    }
    if lock["source"] != expected_source:
        raise RuntimeError("live content selection disagrees with immutable execution lock")
    paths = [ROOT / row["path"] for row in lock["code"]["files"]]
    observed_code, observed_code_sha = code_record(ROOT, paths)
    if observed_code != lock["code"] or observed_code_sha != lock["code_sha256"]:
        raise RuntimeError("code changed after immutable execution lock")
    builds = tuple(_native_build_from_lock(lock, role) for role in ROLE_ORDER[:3])
    build = builds[0]
    if any(candidate != build for candidate in builds[1:]):
        raise RuntimeError("semantic roles do not resolve to one locked native build")
    observed_dependency = dependency_record(Path(build.compiler_path))
    if observed_dependency != lock["dependency"]:
        raise RuntimeError("dependency/toolchain environment changed after lock")
    return selected


def _open_source(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeError("selected source is not regular")
    return fd


def _forbidden_independent_imports() -> tuple[str, ...]:
    return tuple(sorted(name for name in sys.modules if name in _FORBIDDEN_INDEPENDENT_MODULES))


def _assert_independent_import_graph_clean(stage: str) -> None:
    forbidden = _forbidden_independent_imports()
    if forbidden:
        raise RuntimeError(f"independent import graph contaminated at {stage}: {','.join(forbidden)}")


def _load_independent_functions() -> tuple[Any, Any]:
    _assert_independent_import_graph_clean("before-independent-import")
    from research.f1_ciip_v1.independent_validator import (
        parse_and_verify_ply_fd,
        rebuild_and_validate_container_bytes,
    )
    _assert_independent_import_graph_clean("after-independent-import")
    return parse_and_verify_ply_fd, rebuild_and_validate_container_bytes


def _probe_independent_import_graph() -> None:
    parse_function, rebuild_function = _load_independent_functions()
    if not callable(parse_function) or not callable(rebuild_function):
        raise RuntimeError("independent validation functions are not callable")
    _assert_independent_import_graph_clean("probe-complete")
    print(canonical_json_bytes({
        "forbidden_imports": list(_forbidden_independent_imports()),
        "status": "INDEPENDENT_IMPORT_GRAPH_CLEAN",
    }).decode("ascii"))


def _worker(role: str, source_fd: int, role_dir: Path, base_container: Path | None) -> None:
    lock = load_execution_lock(LOCK_PATH)
    _validate_worker_capability(lock, role, role_dir)
    if not stat.S_ISREG(os.fstat(source_fd).st_mode):
        raise RuntimeError("worker source descriptor is not regular")
    if role == "INDEPENDENT_VALIDATOR":
        expected_base = (ROOT / SCRATCH_RELATIVE / "BASE_A" / "container.bin").resolve()
        if base_container is None or base_container.resolve() != expected_base:
            raise RuntimeError("independent baseline path mismatch")
    elif base_container is not None:
        raise RuntimeError("semantic role unexpectedly received baseline container")
    expected_source = lock["source"]
    started_wall = time.monotonic_ns()
    started_cpu = time.process_time_ns()
    output = role_dir / "container.bin"
    if output.exists():
        raise RuntimeError("worker output exists")
    if role == "INDEPENDENT_VALIDATOR":
        if base_container is None:
            raise RuntimeError("independent validator has no baseline container")
        parse_and_verify_ply_fd, rebuild_and_validate_container_bytes = _load_independent_functions()
        parsed = parse_and_verify_ply_fd(source_fd, expected_source)
        raw = base_container.read_bytes()
        rebuild_and_validate_container_bytes(
            raw,
            parsed,
            expected_code_sha256=lock["code_sha256"],
            expected_dependency_lock_sha256=lock["dependency_lock_sha256"],
        )
        _assert_independent_import_graph_clean("after-independent-validation")
        container_bytes = raw
    else:
        # Import only in semantic roles.  The independent-validator process
        # must have no semantic-core module in its import graph.
        from research.f1_ciip_v1.native_worker import NativeWorker
        from research.f1_ciip_v1.semantic_core import execute_semantic_role

        worker = NativeWorker(_native_build_from_lock(lock, role))
        result = execute_semantic_role(
            source_fd,
            expected_source=expected_source,
            native_worker=worker,
            code_sha256=lock["code_sha256"],
            dependency_lock_sha256=lock["dependency_lock_sha256"],
            role=role,
            resource={
                "cpu_nanoseconds": max(0, time.process_time_ns() - started_cpu),
                "wall_nanoseconds": max(0, time.monotonic_ns() - started_wall),
                "peak_process_tree_rss_bytes": 0,
                "scratch_bytes": 0,
            },
        )
        container_bytes = result.container_bytes
    with output.open("xb") as handle:
        handle.write(container_bytes)
        handle.flush()
        os.fsync(handle.fileno())


def _preexec() -> None:
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))


def _process_tree_sample(root_pid: int, expected_pgid: int) -> tuple[int, tuple[int, ...], bool, bool]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="], check=True, capture_output=True, text=True
    )
    rows: dict[int, tuple[int, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 4:
            pid, ppid, pgid, rss = map(int, fields)
            rows[pid] = (ppid, pgid, rss)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _group, _rss) in rows.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    live_descendants = {pid for pid in descendants if pid in rows}
    group_members = {pid for pid, row in rows.items() if row[1] == expected_pgid}
    live = tuple(sorted(live_descendants | group_members))
    escaped = any(rows[pid][1] != expected_pgid for pid in live_descendants)
    return sum(rows[pid][2] for pid in live) * 1024, live, escaped, root_pid in rows


def _monitor_probe() -> None:
    rss, live, escaped, observed = _process_tree_sample(os.getpid(), os.getpgrp())
    if not observed or escaped or os.getpid() not in live or rss <= 0:
        raise RuntimeError("process-tree monitor preflight failed")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _patch_resource_metadata(path: Path, metrics: Mapping[str, int]) -> None:
    expected_metric_keys = {
        "cpu_nanoseconds", "wall_nanoseconds", "peak_process_tree_rss_bytes", "scratch_bytes",
    }
    if set(metrics) != expected_metric_keys:
        raise RuntimeError("resource patch keys are not the exact supervisor-owned set")
    raw = path.read_bytes()
    decoded = decode_container(raw, materialize_rows=False)
    metadata = dict(decoded.metadata)
    metadata["resource"] = dict(metadata["resource"])
    metadata["resource"].update(metrics)
    patched = encode_metadata_region(metadata) + raw[METADATA_REGION_BYTES:]
    validate_container(
        patched,
        expected_code_sha256=metadata["code_sha256"],
        expected_dependency_lock_sha256=metadata["dependency_lock_sha256"],
    )
    replacement = path.with_name("container.patched.tmp")
    _write_new_durable(replacement, patched, mode=0o600)
    os.replace(replacement, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _run_role(
    role: str,
    source_fd: int,
    scratch: Path,
    base_container: Path | None,
    capability: str,
) -> dict[str, Any]:
    role_dir = scratch / role
    role_dir.mkdir(mode=0o700)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-role",
        role,
        "--source-fd",
        str(source_fd),
        "--role-dir",
        str(role_dir),
    ]
    if base_container is not None:
        command += ["--base-container", str(base_container)]
    started_wall = time.monotonic_ns()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    stdout_path = role_dir / "stdout.bin"
    stderr_path = role_dir / "stderr.bin"
    output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    stdout_fd = os.open(stdout_path, output_flags, 0o400)
    try:
        stderr_fd = os.open(stderr_path, output_flags, 0o400)
    except BaseException:
        os.close(stdout_fd)
        raise
    stdout_handle = os.fdopen(stdout_fd, "wb", closefd=True)
    stderr_handle = os.fdopen(stderr_fd, "wb", closefd=True)
    try:
        process = subprocess.Popen(
            command,
            pass_fds=(source_fd,),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env={**os.environ, **THREAD_ENV, "CIIP_EXECUTION_CAPABILITY": capability},
            preexec_fn=_preexec,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    peak_rss = 0
    failure: str | None = None
    observed_root = False
    observed_pids: set[int] = set()
    escaped = False
    # Sample immediately so short-lived workers cannot evade accounting.
    try:
        sample_rss, live, sample_escaped, root_seen = _process_tree_sample(process.pid, process.pid)
        peak_rss = max(peak_rss, sample_rss)
        observed_pids.update(live)
        observed_root |= root_seen
        escaped |= sample_escaped
    except Exception:
        failure = "PROCESS_TREE_MONITOR_FAILURE"
    while process.poll() is None:
        elapsed = time.monotonic_ns() - started_wall
        try:
            sample_rss, live, sample_escaped, root_seen = _process_tree_sample(process.pid, process.pid)
            peak_rss = max(peak_rss, sample_rss)
            observed_pids.update(live)
            observed_root |= root_seen
            escaped |= sample_escaped
        except Exception:
            failure = "PROCESS_TREE_MONITOR_FAILURE"
        if elapsed > 300_000_000_000:
            failure = "WALL_CAP_EXCEEDED"
        if peak_rss > 12 * 1024**3:
            failure = "RSS_CAP_EXCEEDED"
        if _directory_bytes(role_dir) > 256 * 1024**2:
            failure = "SCRATCH_CAP_EXCEEDED"
        if escaped:
            failure = "PROCESS_ESCAPED_GROUP"
        if failure:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for pid in observed_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            break
        time.sleep(0.01)
    process.wait()
    stdout_handle.flush()
    stderr_handle.flush()
    os.fsync(stdout_handle.fileno())
    os.fsync(stderr_handle.fileno())
    stdout_handle.close()
    stderr_handle.close()
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_ns = time.monotonic_ns() - started_wall
    cpu_ns = int(
        ((usage_after.ru_utime + usage_after.ru_stime) - (usage_before.ru_utime + usage_before.ru_stime))
        * 1_000_000_000
    )
    # ru_maxrss is bytes on Darwin.  It supplies full-lifetime root-process
    # peak accounting across the final ps polling gap; the worker is forbidden
    # to spawn descendants, enforced below.
    peak_rss = max(peak_rss, int(usage_after.ru_maxrss))
    if not observed_root and failure is None:
        failure = "ROOT_PROCESS_NEVER_OBSERVED"
    unexpected_children = tuple(pid for pid in observed_pids if pid != process.pid)
    if unexpected_children and failure is None:
        failure = "UNEXPECTED_WORKER_DESCENDANT"
    output = role_dir / "container.bin"
    if process.returncode != 0 and failure is None:
        failure = f"WORKER_EXIT_{process.returncode}"
    if not output.exists() and failure is None:
        failure = "WORKER_OUTPUT_MISSING"
    scratch_bytes = _directory_bytes(role_dir)
    if wall_ns > 300_000_000_000 and failure is None:
        failure = "WALL_CAP_EXCEEDED_POSTEXIT"
    if cpu_ns > 300_000_000_000 and failure is None:
        failure = "CPU_CAP_EXCEEDED_POSTEXIT"
    if peak_rss > 12 * 1024**3 and failure is None:
        failure = "RSS_CAP_EXCEEDED_POSTEXIT"
    if scratch_bytes > 256 * 1024**2 and failure is None:
        failure = "SCRATCH_CAP_EXCEEDED_POSTEXIT"
    if failure is None:
        _patch_resource_metadata(
            output,
            {
                "cpu_nanoseconds": cpu_ns,
                "wall_nanoseconds": wall_ns,
                "peak_process_tree_rss_bytes": peak_rss,
                "scratch_bytes": scratch_bytes,
            },
        )
    evidence = {
        "role": role,
        "return_code": process.returncode,
        "cpu_nanoseconds": cpu_ns,
        "wall_nanoseconds": wall_ns,
        "peak_process_tree_rss_bytes": peak_rss,
        "scratch_bytes": scratch_bytes,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "failure": failure,
        "container_sha256": sha256_file(output)[0] if output.exists() else None,
    }
    _write_new_durable(role_dir / "process_evidence.json", canonical_json_bytes(evidence) + b"\n")
    return evidence


def _baseline_container_for_role(role: str, baseline: Path | None) -> Path | None:
    """Route BASE_A bytes only to the independent-validator worker."""

    if role == "INDEPENDENT_VALIDATOR":
        if baseline is None:
            raise RuntimeError("independent validator has no baseline container")
        return baseline
    return None


def _rename_exclusive(source: Path, destination: Path) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("exclusive directory publication is only locked for Darwin")
    libc = ctypes.CDLL(None, use_errno=True)
    renamex_np = libc.renamex_np
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004) != 0:  # RENAME_EXCL
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def execute_once() -> None:
    lock = load_execution_lock(LOCK_PATH)
    _validate_pregeometry_gates(lock)
    selected = _validate_live_lock(lock)
    _monitor_probe()
    capability = _create_execution_claim(lock)
    scratch = ROOT / SCRATCH_RELATIVE
    output = ROOT / OUTPUT_RELATIVE
    scratch.mkdir(parents=True, mode=0o700)
    fd: int | None = None
    evidence: list[dict[str, Any]] = []
    outcome = STOP_TOKEN
    failure: str | None = None
    try:
        fd = _open_source(Path(str(selected["raw_path"])))
        base: Path | None = None
        baseline_table_hashes: dict[str, str] | None = None
        for role in ROLE_ORDER:
            role_base = _baseline_container_for_role(role, base)
            row = _run_role(role, fd, scratch, role_base, capability)
            evidence.append(row)
            if row["failure"] is not None:
                failure = f"{role}:{row['failure']}"
                break
            container_path = scratch / role / "container.bin"
            decoded = decode_container(
                container_path.read_bytes(),
                expected_code_sha256=lock["code_sha256"],
                expected_dependency_lock_sha256=lock["dependency_lock_sha256"],
                materialize_rows=False,
            )
            table_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in decoded.table_bytes.items()}
            if baseline_table_hashes is None:
                baseline_table_hashes = table_hashes
                base = container_path
            elif table_hashes != baseline_table_hashes:
                failure = f"{role}:CANONICAL_TABLE_BYTES_MISMATCH"
                break
        if failure is None and len(evidence) == 4:
            outcome = PASS_TOKEN
    except Exception as exc:
        failure = f"SUPERVISOR:{type(exc).__name__}:{exc}"
    finally:
        if fd is not None:
            os.close(fd)
        result = {
            "outcome": outcome,
            "failure": failure,
            "execution_lock_sha256": lock["execution_lock_sha256"],
            "raw_ply_sha256": lock["source"]["raw_ply_sha256"],
            "roles": evidence,
            "rerun_authorized": False,
            "replacement_input_authorized": False,
            "gold_or_labels_accessed": False,
        }
        validate_execution_result(result)
        _write_new_durable(scratch / RESULT_FILENAME, canonical_json_bytes(result) + b"\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        _rename_exclusive(scratch, output)
    print(canonical_json_bytes(result).decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-lock-only", action="store_true")
    modes.add_argument("--execute-locked-smoke", action="store_true")
    modes.add_argument("--probe-independent-import-graph", action="store_true")
    modes.add_argument("--worker-role", choices=ROLE_ORDER)
    parser.add_argument("--source-fd", type=int)
    parser.add_argument("--role-dir", type=Path)
    parser.add_argument("--base-container", type=Path)
    args = parser.parse_args()
    if args.prepare_lock_only:
        prepare_lock()
    elif args.execute_locked_smoke:
        execute_once()
    elif args.probe_independent_import_graph:
        _probe_independent_import_graph()
    else:
        if args.source_fd is None or args.role_dir is None:
            parser.error("worker role requires --source-fd and --role-dir")
        _worker(args.worker_role, args.source_fd, args.role_dir, args.base_container)


if __name__ == "__main__":
    main()
