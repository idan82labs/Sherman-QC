"""Candidate14 Darwin sampled resource-ceiling capability.

This module is filesystem-blind and input-blind.  It binds the unchanged
Candidate13/Candidate9/Candidate6 scientific lineage and supplies only the
Darwin process-resource primitive required by a future guarded Candidate14
parent.  Memory enforcement is explicitly sampled: every observation sums the
parent and worker resident sizes, separately sums their physical footprints,
and compares the larger combined quantity with the fixed ceiling.  A 10 ms
poll can overshoot between samples; this is not a hard ``RLIMIT_AS`` claim.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import os
import select
import signal
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Callable


SCHEMA_VERSION = 1
ATTEMPT_ORDINAL = 14
ENGINEERING_ATTEMPT_ID = "CANDIDATE14_DARWIN_RESOURCE_CAPABILITY_V1"
SCIENTIFIC_HYPOTHESIS_ID = "F1_AUG30_CONSERVATIVE_RECALL_32_RULE_V1"
RESULT_KIND = "sherman_qc_f1_aug30_candidate14_darwin_resource_capability_v1"

RUSAGE_INFO_V2 = 2
RUSAGE_INFO_V2_SIZE_BYTES = 160
POLL_INTERVAL_SECONDS = 0.010
CHILD_CPU_SOFT_LIMIT_SECONDS = 240
CHILD_CPU_HARD_LIMIT_SECONDS = 241
MEMORY_ENFORCEMENT_KIND = (
    "SAMPLED_DARWIN_COMBINED_RESIDENT_OR_PHYS_FOOTPRINT_CEILING"
)
MEMORY_ENFORCEMENT_IS_HARD_RLIMIT_AS = False
SAMPLING_OVERSHOOT_BETWEEN_POLLS_POSSIBLE = True

FIXED_BINDINGS = MappingProxyType(
    {
        "c13_core": MappingProxyType(
            {
                "path": "backend/f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
                "sha256": "f6296d2fc48c0a6015862a9927030488777ba75eefc60d3eb67d72c355cc422b",
                "size_bytes": 65344,
            }
        ),
        "c13_core_tests": MappingProxyType(
            {
                "path": "tests/test_f1_train18_candidate13_pool_adapter_stage_localizer_v1.py",
                "sha256": "1a0982f8809c673deff711252c737bc565257f71070f09bce6e47b9cedbece15",
                "size_bytes": 43387,
            }
        ),
        "c13_terminal_disposition": MappingProxyType(
            {
                "path": "research/f1_aug30_train18_candidate13_terminal_disposition_v1.json",
                "sha256": "a0b2b04c9a3101e2a7228541c1003e34872ccb29c6c1c28318200060cfcf4abd",
                "size_bytes": 3865,
            }
        ),
        "c6_hypothesis": MappingProxyType(
            {
                "path": "backend/f1_train18_candidate6_conservative_recall_recovery_v1.py",
                "sha256": "4c02914840d9435b811150653c2c683b0a32e2c20105d308c4f51505471ad252",
                "size_bytes": 53837,
            }
        ),
        "c9_adapter": MappingProxyType(
            {
                "path": "backend/f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
                "sha256": "59368b3955dd4b8d86e6e1073a09a7bf4f0d274e42d5f623655aec04b1270096",
                "size_bytes": 22354,
            }
        ),
        "c9_adapter_tests": MappingProxyType(
            {
                "path": "tests/test_f1_train18_candidate9_lossless_pool_alias_adapter_v1.py",
                "sha256": "4551c84bf336490166967d1707b060a48db455325d9fe2c2fd5954d2ff9fa623",
                "size_bytes": 16598,
            }
        ),
    }
)


class RUsageInfoV2(ctypes.Structure):
    _fields_ = (
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    )


if (
    ctypes.sizeof(RUsageInfoV2) != RUSAGE_INFO_V2_SIZE_BYTES
    or RUsageInfoV2.ri_resident_size.offset != 64
    or RUsageInfoV2.ri_phys_footprint.offset != 72
    or RUsageInfoV2.ri_proc_start_abstime.offset != 80
    or RUsageInfoV2.ri_diskio_byteswritten.offset != 152
):
    raise RuntimeError("Candidate14 Darwin rusage_info_v2 ABI mismatch")


@dataclass(frozen=True)
class ProcessUsageSample:
    pid: int
    resident_size_bytes: int
    phys_footprint_bytes: int
    process_start_abstime: int


@dataclass(frozen=True)
class CombinedUsageSample:
    parent: ProcessUsageSample
    worker: ProcessUsageSample
    combined_resident_size_bytes: int
    combined_phys_footprint_bytes: int
    ceiling_comparison_bytes: int


class DarwinRusageError(RuntimeError):
    pass


class DarwinRusageSampler:
    """Exact ``proc_pid_rusage(RUSAGE_INFO_V2)`` sampler."""

    def __init__(self, function: Any | None = None) -> None:
        if sys.platform != "darwin":
            raise DarwinRusageError("Candidate14 requires Darwin libproc")
        library: Any | None = None
        if function is None:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            function = library.proc_pid_rusage
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(RUsageInfoV2),
            ]
            function.restype = ctypes.c_int
        if not callable(function):
            raise TypeError("Candidate14 proc_pid_rusage binding is not callable")
        self._library = library
        self._function = function

    def sample_pid(self, pid: int) -> ProcessUsageSample:
        if type(pid) is not int or pid <= 0:
            raise ValueError("Candidate14 sampled PID must be a positive exact integer")
        buffer = RUsageInfoV2()
        ctypes.set_errno(0)
        result = self._function(pid, RUSAGE_INFO_V2, ctypes.byref(buffer))
        if type(result) is not int or result != 0:
            error_number = ctypes.get_errno() or errno.ESRCH
            raise DarwinRusageError(
                f"Candidate14 proc_pid_rusage failed with errno {error_number}"
            )
        resident = int(buffer.ri_resident_size)
        footprint = int(buffer.ri_phys_footprint)
        start = int(buffer.ri_proc_start_abstime)
        if resident < 0 or footprint < 0 or start <= 0:
            raise DarwinRusageError("Candidate14 libproc returned an invalid sample")
        return ProcessUsageSample(
            pid=pid,
            resident_size_bytes=resident,
            phys_footprint_bytes=footprint,
            process_start_abstime=start,
        )


def combined_usage_sample(
    sampler: DarwinRusageSampler,
    *,
    parent_pid: int,
    worker_pid: int,
) -> CombinedUsageSample:
    if type(sampler) is not DarwinRusageSampler:
        raise TypeError("Candidate14 combined sampler identity mismatch")
    if parent_pid == worker_pid:
        raise ValueError("Candidate14 parent and worker PIDs must differ")
    parent = sampler.sample_pid(parent_pid)
    worker = sampler.sample_pid(worker_pid)
    combined_resident = parent.resident_size_bytes + worker.resident_size_bytes
    combined_footprint = parent.phys_footprint_bytes + worker.phys_footprint_bytes
    if combined_resident < 0 or combined_footprint < 0:
        raise DarwinRusageError("Candidate14 combined resource sample overflowed")
    return CombinedUsageSample(
        parent=parent,
        worker=worker,
        combined_resident_size_bytes=combined_resident,
        combined_phys_footprint_bytes=combined_footprint,
        ceiling_comparison_bytes=max(combined_resident, combined_footprint),
    )


class SampledDarwinCeiling:
    """One parent/worker sampled memory ceiling with an exact kill target."""

    def __init__(
        self,
        sampler: DarwinRusageSampler,
        *,
        parent_pid: int,
        worker_pid: int,
        ceiling_bytes: int,
        kill_function: Callable[[int, int], Any] = os.kill,
    ) -> None:
        if type(sampler) is not DarwinRusageSampler:
            raise TypeError("Candidate14 ceiling sampler identity mismatch")
        if type(parent_pid) is not int or parent_pid != os.getpid():
            raise ValueError("Candidate14 ceiling parent must be the current process")
        if type(worker_pid) is not int or worker_pid <= 0 or worker_pid == parent_pid:
            raise ValueError("Candidate14 ceiling worker PID is invalid")
        if type(ceiling_bytes) is not int or ceiling_bytes <= 0:
            raise ValueError("Candidate14 ceiling must be a positive exact integer")
        if not callable(kill_function):
            raise TypeError("Candidate14 kill function must be callable")
        baseline = combined_usage_sample(
            sampler, parent_pid=parent_pid, worker_pid=worker_pid
        )
        self._sampler = sampler
        self._parent_pid = parent_pid
        self._worker_pid = worker_pid
        self._parent_start = baseline.parent.process_start_abstime
        self._worker_start = baseline.worker.process_start_abstime
        self._ceiling_bytes = ceiling_bytes
        self._kill_function = kill_function
        self._kill_sent = False

    def sample_and_enforce(self) -> str:
        if self._kill_sent:
            raise RuntimeError("Candidate14 worker kill was already sent")
        sample = combined_usage_sample(
            self._sampler,
            parent_pid=self._parent_pid,
            worker_pid=self._worker_pid,
        )
        if (
            sample.parent.process_start_abstime != self._parent_start
            or sample.worker.process_start_abstime != self._worker_start
        ):
            raise DarwinRusageError("Candidate14 sampled process identity changed")
        if sample.ceiling_comparison_bytes <= self._ceiling_bytes:
            return "WITHIN_SAMPLED_CEILING"
        self._kill_function(self._worker_pid, signal.SIGKILL)
        self._kill_sent = True
        return "SAMPLED_CEILING_EXCEEDED_KILL_SENT"


RESULT_KEYS = frozenset(
    {
        "artifact_or_control_input_opened",
        "check_code",
        "child_cpu_backup_limit_verified",
        "combined_parent_worker_sampling_verified",
        "engineering_attempt_id",
        "kill_path_exercised",
        "kind",
        "label_evaluator_metric_fit_holdout_opened_or_started",
        "memory_enforcement_is_hard_rlimit_as",
        "memory_enforcement_kind",
        "poll_interval_seconds",
        "project_artifact_or_control_filesystem_discovery_or_open_performed",
        "sampling_overshoot_between_polls_possible",
        "scientific_hypothesis_id",
        "source_only",
        "status",
        "struct_size_bytes",
        "synthetic_child_only",
    }
)


def _capability_result() -> dict[str, Any]:
    value = {
        "artifact_or_control_input_opened": False,
        "check_code": "C14_DARWIN_RESOURCE_CAPABILITY_PASSED",
        "child_cpu_backup_limit_verified": True,
        "combined_parent_worker_sampling_verified": True,
        "engineering_attempt_id": ENGINEERING_ATTEMPT_ID,
        "kill_path_exercised": True,
        "kind": RESULT_KIND,
        "label_evaluator_metric_fit_holdout_opened_or_started": False,
        "memory_enforcement_is_hard_rlimit_as": (
            MEMORY_ENFORCEMENT_IS_HARD_RLIMIT_AS
        ),
        "memory_enforcement_kind": MEMORY_ENFORCEMENT_KIND,
        "poll_interval_seconds": "0.010",
        "project_artifact_or_control_filesystem_discovery_or_open_performed": False,
        "sampling_overshoot_between_polls_possible": (
            SAMPLING_OVERSHOOT_BETWEEN_POLLS_POSSIBLE
        ),
        "scientific_hypothesis_id": SCIENTIFIC_HYPOTHESIS_ID,
        "source_only": True,
        "status": "PASS_SOURCE_ONLY_NONAUTHORIZING",
        "struct_size_bytes": RUSAGE_INFO_V2_SIZE_BYTES,
        "synthetic_child_only": True,
    }
    if set(value) != RESULT_KEYS:
        raise RuntimeError("Candidate14 resource capability result schema mismatch")
    return value


def run_repeatable_source_only_capability_probe() -> dict[str, Any]:
    """Exercise self, synthetic-child sampling, CPU backup, and kill path."""

    sampler = DarwinRusageSampler()
    self_sample = sampler.sample_pid(os.getpid())
    if self_sample.resident_size_bytes <= 0 or self_sample.phys_footprint_bytes <= 0:
        raise DarwinRusageError("Candidate14 self sample lacks memory accounting")
    child_program = (
        "import os,resource,sys,time;"
        "(sys.flags.isolated==1 and sys.flags.no_site==1 and "
        "sys.flags.ignore_environment==1 and sys.flags.no_user_site==1)"
        " or sys.exit(90);"
        f"resource.setrlimit(resource.RLIMIT_CPU,({CHILD_CPU_SOFT_LIMIT_SECONDS},"
        f"{CHILD_CPU_HARD_LIMIT_SECONDS}));"
        f"limits=resource.getrlimit(resource.RLIMIT_CPU);"
        f"limits==({CHILD_CPU_SOFT_LIMIT_SECONDS},{CHILD_CPU_HARD_LIMIT_SECONDS})"
        " or sys.exit(91);"
        "payload=bytearray(8*1024*1024);"
        "os.write(1,b'C14_READY\\n');"
        "time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", child_program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    try:
        if child.stdout is None:
            raise DarwinRusageError("Candidate14 synthetic child pipe is absent")
        deadline = time.monotonic() + 5.0
        ready = b""
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise DarwinRusageError(
                    "Candidate14 synthetic child exited before its CPU-limit handshake"
                )
            readable, _, _ = select.select([child.stdout], [], [], POLL_INTERVAL_SECONDS)
            if readable:
                ready = child.stdout.readline()
                break
        if ready != b"C14_READY\n":
            raise DarwinRusageError(
                "Candidate14 synthetic child CPU-limit handshake failed"
            )
        monitor: SampledDarwinCeiling | None = None
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise DarwinRusageError(
                    "Candidate14 synthetic child exited before sampling"
                )
            try:
                monitor = SampledDarwinCeiling(
                    sampler,
                    parent_pid=os.getpid(),
                    worker_pid=child.pid,
                    ceiling_bytes=1,
                )
                break
            except DarwinRusageError:
                time.sleep(POLL_INTERVAL_SECONDS)
        if monitor is None:
            raise DarwinRusageError("Candidate14 synthetic child was not sampleable")
        outcome = monitor.sample_and_enforce()
        if outcome != "SAMPLED_CEILING_EXCEEDED_KILL_SENT":
            raise DarwinRusageError("Candidate14 synthetic kill path was not exercised")
        child.wait(timeout=5.0)
        if child.returncode != -signal.SIGKILL:
            raise DarwinRusageError("Candidate14 synthetic child kill status mismatch")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5.0)
    return _capability_result()
