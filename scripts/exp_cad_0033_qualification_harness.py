#!/usr/bin/env python3
"""One-shot source-free EXP_CAD_0033 qualification harness.

There is deliberately no preparation mode.  The harness accepts only the
pre-existing, immutable-lock-bound EXP32 synthetic envelope.  It dynamically
imports the hash-bound candidate harness only after the EXP33 lock and exact
CLI spellings have passed verification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

from scripts.exp_cad_0033_qualification_contract import (
    ROOT,
    canonical_bytes,
    canonical_json_write,
    sha256_path,
    verify_harness_cli,
    verify_lock,
    verify_detached_postlock_approval,
    verify_current_environment,
    verify_self_binding,
)


SCHEMA = "EXP_CAD_0033_SOURCE_FREE_QUALIFICATION_RESULT_V1"
EXPERIMENT_ID = "EXP_CAD_0033"
CANDIDATE_SCHEMA = "EXP_CAD_0032_PERFORMANCE_EQUIVALENCE_RESULT_V1"
CANDIDATE_EXPERIMENT_ID = "EXP_CAD_0032"
CANDIDATE_PASS = "PASS_PRODUCTION_SCALE_SOURCE_FREE"
OUTER_PASS = "PASS_SOURCE_FREE_QUALIFICATION"
OUTER_FAIL = "FAIL_TERMINAL_SOURCE_FREE_QUALIFICATION"
JOURNAL_SCHEMA = "EXP_CAD_0033_DURABLE_QUALIFICATION_ATTEMPT_JOURNAL_V1"
EXPECTED_LAUNCH_SCHEDULE = (
    ("source", "full/run_1/source_measurement"),
    ("render", "full/run_1/renderer_measurement"),
    ("source", "full/run_2/source_measurement"),
    ("render", "full/run_2/renderer_measurement"),
    ("source", "full/run_3/source_measurement"),
    ("render", "full/run_3/renderer_measurement"),
    ("source", "half/run_1/source_measurement"),
    ("reference", "reference_differential/frozen_parent/run_1/measurement"),
    ("source", "reference_differential/v5/run_1/measurement"),
    ("reference", "reference_differential/frozen_parent/run_2/measurement"),
    ("source", "reference_differential/v5/run_2/measurement"),
    ("reference", "reference_differential/frozen_parent/run_3/measurement"),
    ("source", "reference_differential/v5/run_3/measurement"),
)


class AttemptJournal:
    """Durable, fail-closed accounting for the one authorized attempt."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.result_root = path.parent

    @classmethod
    def consume(
        cls,
        *,
        result_root: Path,
        lock_path: Path,
        lock_sha256: str,
        approval_path: Path,
        approval_sha256: str,
    ) -> "AttemptJournal":
        if result_root.exists():
            raise ValueError("EXP33 one-shot result root already exists; attempt is consumed")
        result_root.mkdir(parents=True, exist_ok=False)
        parent_fd = os.open(result_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        payload = {
            "schema": JOURNAL_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "authorized_attempts": 1,
            "attempts_started": 1,
            "attempt_status": "IN_PROGRESS_CONSUMED",
            "qualification_lock_path": str(lock_path),
            "qualification_lock_sha256": lock_sha256,
            "detached_approval_path": str(approval_path),
            "detached_approval_sha256": approval_sha256,
            "started_monotonic_ns": time.monotonic_ns(),
            "candidate_import_started": False,
            "candidate_import_completed": False,
            "environment_verified": False,
            "child_launch_count": 0,
            "child_launches": [],
            "terminal_result_path": None,
            "physical_scan_or_evaluator_input_used": False,
        }
        journal = cls(result_root / "EXP_CAD_0033_ATTEMPT_JOURNAL.json", payload)
        journal.persist()
        return journal

    def persist(self) -> None:
        canonical_json_write(self.path, self.payload)

    def mark_candidate_import(self, *, completed: bool = False) -> None:
        self.payload["candidate_import_started"] = True
        if completed:
            self.payload["candidate_import_completed"] = True
        self.persist()

    def bind_verified_environment(
        self, environment: Mapping[str, str], *, worker_path: Path
    ) -> None:
        self.payload["environment_verified"] = True
        self.payload["verified_environment"] = dict(environment)
        self.payload["worker_path"] = str(worker_path)
        self.persist()

    def launch_started(self, command: Sequence[str], measurement_root: Path) -> int:
        launch_id = len(self.payload["child_launches"]) + 1
        if launch_id > len(EXPECTED_LAUNCH_SCHEDULE):
            raise ValueError("EXP33 refused a fourteenth child launch")
        mode = command[2] if len(command) > 2 else "UNKNOWN"
        relative = Path(measurement_root).relative_to(self.result_root).as_posix()
        if (mode, relative) != EXPECTED_LAUNCH_SCHEDULE[launch_id - 1]:
            raise ValueError(
                f"EXP33 child launch {launch_id} violates frozen FSM: {(mode, relative)}"
            )
        if not self.payload.get("environment_verified"):
            raise ValueError("EXP33 child launch preceded environment verification")
        if command[0] != self.payload["verified_environment"]["python_executable_invocation_path"]:
            raise ValueError("EXP33 child interpreter differs from environment attestation")
        if command[1] != self.payload["worker_path"]:
            raise ValueError("EXP33 child worker path differs from immutable lock")
        output_index = command.index("--output-root")
        actual_output = command[output_index + 1]
        measurement = self.result_root / relative
        if mode == "source" and relative.startswith(("full/", "half/")):
            expected_output = measurement.parent / "source"
        elif mode == "render":
            expected_output = measurement.parent / "renderer"
        else:
            expected_output = measurement.parent / "output"
        if actual_output != str(expected_output):
            raise ValueError("EXP33 child output root violates frozen FSM")
        expected_rows = (
            1_000_000
            if launch_id in {1, 3, 5}
            else 500_000
            if launch_id == 7
            else 100_000
            if mode in {"source", "reference"}
            else None
        )
        if expected_rows is not None:
            rows_index = command.index("--rows-per-member")
            if command[rows_index + 1] != str(expected_rows):
                raise ValueError("EXP33 child row count violates frozen FSM")
        for flag, expected in (
            ("--post-lock-approval", self.payload["detached_approval_path"]),
            ("--post-lock-approval-sha256", self.payload["detached_approval_sha256"]),
            ("--attempt-journal", str(self.path)),
            ("--launch-id", str(launch_id)),
        ):
            flag_index = command.index(flag)
            if command[flag_index + 1] != expected:
                raise ValueError(f"EXP33 child capability flag drift: {flag}")
        row = {
            "launch_id": launch_id,
            "mode": mode,
            "command": list(command),
            "command_sha256": hashlib.sha256(canonical_bytes(list(command))).hexdigest(),
            "measurement_root": str(measurement_root),
            "status": "STARTED",
            "started_monotonic_ns": time.monotonic_ns(),
            "pid": None,
            "result": None,
        }
        self.payload["child_launches"].append(row)
        self.payload["child_launch_count"] = launch_id
        self.persist()
        return launch_id

    def assert_full_schedule_complete(self) -> None:
        if len(self.payload["child_launches"]) != len(EXPECTED_LAUNCH_SCHEDULE):
            raise ValueError("EXP33 qualification did not complete exactly 13 launches")
        if any(row["status"] != "RETURNED" for row in self.payload["child_launches"]):
            raise ValueError("EXP33 qualification has a non-returned child launch")

    def launch_completed(
        self, launch_id: int, *, result: Mapping[str, Any] | None, error: BaseException | None
    ) -> None:
        disk = json.loads(self.path.read_text(encoding="utf-8"))
        if disk.get("schema") != JOURNAL_SCHEMA or disk.get("attempts_started") != 1:
            raise ValueError("EXP33 durable journal changed before child completion")
        self.payload = disk
        row = self.payload["child_launches"][launch_id - 1]
        if row.get("status") not in {"STARTED", "WORKER_CLAIMED"}:
            raise ValueError("EXP33 child completion found a replayed/closed journal row")
        if row.get("status") == "WORKER_CLAIMED":
            self._verify_worker_claim(row, launch_id)
        row["completed_monotonic_ns"] = time.monotonic_ns()
        if error is not None:
            row["status"] = "CONTROLLER_EXCEPTION"
            row["error_type"] = type(error).__name__
            row["error_message"] = str(error)
        else:
            assert result is not None
            if result.get("returncode") == 0:
                if row.get("status") != "WORKER_CLAIMED":
                    raise ValueError("successful EXP33 child returned without durable worker claim")
            row["status"] = "TIMED_OUT" if result.get("timed_out") else "RETURNED"
            row["pid"] = result.get("root_pid")
            row["result"] = {
                key: value
                for key, value in result.items()
                if key
                in {
                    "returncode",
                    "timed_out",
                    "rss_limit_exceeded",
                    "resource_kill_reason",
                    "wall_seconds",
                    "peak_process_tree_rss_bytes",
                    "sampled_process_tree_peak_rss_bytes",
                    "exact_worker_process_max_rss_bytes",
                    "stdout_sha256",
                    "stderr_sha256",
                    "time_l_sha256",
                }
            }
        self.persist()

    def _verify_worker_claim(self, row: Mapping[str, Any], launch_id: int) -> None:
        claim_path = self.result_root / f"EXP_CAD_0033_LAUNCH_{launch_id:02d}.claim.json"
        if row.get("claim_path") != str(claim_path) or not claim_path.is_file():
            raise ValueError("EXP33 worker claim path is absent or mismatched")
        block = claim_path.read_bytes()
        digest = hashlib.sha256(block).hexdigest()
        if row.get("claim_sha256") != digest:
            raise ValueError("EXP33 worker claim SHA differs from journal row")
        claim = json.loads(block)
        if set(claim) != {
            "schema",
            "experiment_id",
            "launch_id",
            "command_sha256",
            "qualification_lock_sha256",
            "detached_approval_sha256",
            "worker_pid",
        }:
            raise ValueError("EXP33 worker claim schema fields differ")
        if (
            claim.get("schema") != "EXP_CAD_0033_WORKER_LAUNCH_CLAIM_V1"
            or claim.get("experiment_id") != EXPERIMENT_ID
            or claim.get("launch_id") != launch_id
            or claim.get("command_sha256") != row.get("command_sha256")
            or claim.get("qualification_lock_sha256")
            != self.payload.get("qualification_lock_sha256")
            or claim.get("detached_approval_sha256")
            != self.payload.get("detached_approval_sha256")
            or not isinstance(claim.get("worker_pid"), int)
            or claim.get("worker_pid") <= 0
            or claim.get("worker_pid") != row.get("worker_pid")
        ):
            raise ValueError("EXP33 worker claim contents differ from durable authority")

    def terminal(self, *, status: str, result_path: Path | None, error: BaseException | None = None) -> None:
        self.payload["attempt_status"] = status
        self.payload["completed_monotonic_ns"] = time.monotonic_ns()
        self.payload["terminal_result_path"] = str(result_path) if result_path else None
        if error is not None:
            self.payload["terminal_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self.persist()


@contextmanager
def _journal_candidate_launches(candidate: Any, journal: AttemptJournal) -> Iterator[None]:
    original = candidate.run_measured_command

    def measured(
        command: Sequence[str], *, output_root: Path, hard_timeout_seconds: float
    ) -> Mapping[str, Any]:
        launch_id = len(journal.payload["child_launches"]) + 1
        command = [
            *command,
            "--post-lock-approval",
            journal.payload["detached_approval_path"],
            "--post-lock-approval-sha256",
            journal.payload["detached_approval_sha256"],
            "--attempt-journal",
            str(journal.path),
            "--launch-id",
            str(launch_id),
        ]
        launch_id = journal.launch_started(command, output_root)
        try:
            lock_flag = command.index("--qualification-lock")
            hash_flag = command.index("--qualification-lock-sha256")
            if command[lock_flag + 1] != journal.payload["qualification_lock_path"]:
                raise ValueError("child command qualification-lock path drift")
            if command[hash_flag + 1] != journal.payload["qualification_lock_sha256"]:
                raise ValueError("child command recomputed or changed qualification-lock hash")
            # Revalidate the original immutable bytes immediately before spawn.
            verify_lock(
                Path(journal.payload["qualification_lock_path"]),
                journal.payload["qualification_lock_sha256"],
            )
            result = original(
                command,
                output_root=output_root,
                hard_timeout_seconds=hard_timeout_seconds,
            )
        except BaseException as error:
            journal.launch_completed(launch_id, result=None, error=error)
            raise
        journal.launch_completed(launch_id, result=result, error=None)
        return result

    candidate.run_measured_command = measured
    try:
        yield
    finally:
        candidate.run_measured_command = original


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def _strict_measured_command(
    candidate: Any,
    command: Sequence[str],
    *,
    output_root: Path,
    hard_timeout_seconds: float,
) -> dict[str, Any]:
    """Measure one child and kill immediately on wall or process-tree RSS breach."""

    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "stdout.txt"
    stderr_path = output_root / "stderr.txt"
    time_path = output_root / "time_l.txt"
    wrapped = ["/usr/bin/time", "-l", "-o", str(time_path), *command]
    started = time.monotonic()
    timed_out = False
    rss_limit_exceeded = False
    peak_tree = 0
    samples = 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            wrapped,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                current_rss = candidate.process_tree_rss_bytes(process.pid)
                peak_tree = max(peak_tree, current_rss)
                samples += 1
                if current_rss > candidate.RSS_LIMIT_BYTES:
                    rss_limit_exceeded = True
                    _terminate_process_group(process)
                    break
                if time.monotonic() - started > hard_timeout_seconds:
                    timed_out = True
                    _terminate_process_group(process)
                    break
                time.sleep(candidate.MEMORY_SAMPLING_INTERVAL_SECONDS)
        except BaseException:
            # No controller/sampler/journal exception may orphan a child.
            if process.poll() is None:
                try:
                    _terminate_process_group(process)
                finally:
                    process.wait()
            else:
                process.wait()
            raise
        return_code = process.wait()
    wall = time.monotonic() - started
    time_text = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    exact_process_rss = candidate.parse_time_l_max_rss(time_text)
    peak_rss = max(peak_tree, exact_process_rss or 0)
    # A between-sample final RSS observation still fail-stops publication.
    rss_limit_exceeded = rss_limit_exceeded or peak_rss > candidate.RSS_LIMIT_BYTES
    return {
        "command": list(command),
        "root_pid": int(process.pid),
        "returncode": int(return_code),
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_limit_exceeded,
        "resource_kill_reason": (
            "RSS_LIMIT" if rss_limit_exceeded else "WALL_LIMIT" if timed_out else None
        ),
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": int(peak_rss),
        "sampled_process_tree_peak_rss_bytes": int(peak_tree),
        "exact_worker_process_max_rss_bytes": exact_process_rss,
        "process_tree_sample_count": samples,
        "process_tree_sampling_interval_seconds": candidate.MEMORY_SAMPLING_INTERVAL_SECONDS,
        "stdout_sha256": sha256_path(stdout_path),
        "stderr_sha256": sha256_path(stderr_path),
        "time_l_sha256": sha256_path(time_path) if time_path.exists() else None,
    }


@contextmanager
def _strict_candidate_measurements(candidate: Any) -> Iterator[None]:
    original = candidate.run_measured_command

    def measured(
        command: Sequence[str], *, output_root: Path, hard_timeout_seconds: float
    ) -> dict[str, Any]:
        return _strict_measured_command(
            candidate,
            command,
            output_root=output_root,
            hard_timeout_seconds=hard_timeout_seconds,
        )

    candidate.run_measured_command = measured
    try:
        yield
    finally:
        candidate.run_measured_command = original


def _load_candidate_harness(lock: Mapping[str, Any]) -> Any:
    relative = lock["bound_files"]["candidate_harness"]["path"]
    path = ROOT / relative
    if sha256_path(path) != lock["bound_files"]["candidate_harness"]["sha256"]:
        raise ValueError("candidate harness changed immediately before import")
    spec = importlib.util.spec_from_file_location("exp33_hash_bound_candidate_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load hash-bound candidate harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _candidate_worker(candidate: Any, lock: Mapping[str, Any]) -> Iterator[None]:
    original = candidate.WORKER
    worker = ROOT / lock["bound_files"]["exp33_worker"]["path"]
    if sha256_path(worker) != lock["bound_files"]["exp33_worker"]["sha256"]:
        raise ValueError("EXP33 worker changed immediately before child binding")
    candidate.WORKER = worker
    try:
        yield
    finally:
        candidate.WORKER = original


def _candidate_failure(status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "experiment_id": CANDIDATE_EXPERIMENT_ID,
        "status": status,
        **fields,
        "physical_scan_or_evaluator_input_used": False,
    }


def outer_result(
    *,
    candidate_payload: Mapping[str, Any],
    qualification_lock_sha256: str,
    candidate_harness_sha256: str,
    candidate_worker_sha256: str,
) -> dict[str, Any]:
    """Nest the untouched candidate identity; never rewrite it as EXP33."""

    if candidate_payload.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate result schema drift")
    if candidate_payload.get("experiment_id") != CANDIDATE_EXPERIMENT_ID:
        raise ValueError("candidate result experiment identity drift")
    nested = dict(candidate_payload)
    passed = nested.get("status") == CANDIDATE_PASS
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_SOURCE_FREE_QUALIFICATION_WRAPPER",
        "status": OUTER_PASS if passed else OUTER_FAIL,
        "candidate_identity": {
            "schema": CANDIDATE_SCHEMA,
            "experiment_id": CANDIDATE_EXPERIMENT_ID,
            "harness_sha256": candidate_harness_sha256,
            "worker_sha256": candidate_worker_sha256,
            "silently_relabelled": False,
        },
        "qualification_lock_sha256": qualification_lock_sha256,
        "candidate_payload": nested,
        "physical_scan_or_evaluator_input_used": False,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_EXECUTION_FEASIBILITY_NOT_F1_PROGRESS",
        "terminal_on_failure": True,
    }


def _publish(
    *,
    result_root: Path,
    candidate_payload: Mapping[str, Any],
    lock: Mapping[str, Any],
    lock_sha256: str,
) -> Mapping[str, Any]:
    outer = outer_result(
        candidate_payload=candidate_payload,
        qualification_lock_sha256=lock_sha256,
        candidate_harness_sha256=lock["bound_files"]["candidate_harness"]["sha256"],
        candidate_worker_sha256=lock["bound_files"]["candidate_worker"]["sha256"],
    )
    canonical_json_write(
        result_root / "EXP_CAD_0032_PERFORMANCE_RESULT.json", candidate_payload
    )
    canonical_json_write(result_root / "EXP_CAD_0033_QUALIFICATION_RESULT.json", outer)
    return outer


def _measurement_passes(
    measurement: Mapping[str, Any], *, wall_limit: float, rss_limit: int
) -> bool:
    return bool(
        measurement.get("returncode") == 0
        and not measurement.get("timed_out")
        and not measurement.get("rss_limit_exceeded")
        and float(measurement.get("wall_seconds", float("inf"))) <= wall_limit
        and int(measurement.get("peak_process_tree_rss_bytes", rss_limit + 1)) <= rss_limit
    )


def _median_three(values: Sequence[float]) -> float:
    if len(values) != 3:
        raise ValueError("EXP33 median requires exactly three fresh runs")
    return float(sorted(values)[1])


def _run_reference_differential(
    args: argparse.Namespace, manifest: Path, root: Path, candidate: Any
) -> dict[str, Any]:
    runs: dict[str, list[dict[str, Any]]] = {"FROZEN_PARENT": [], "V5": []}
    for index in range(1, 4):
        for engine in ("FROZEN_PARENT", "V5"):
            run_root = root / engine.lower() / f"run_{index}"
            output_root = run_root / "output"
            command = (
                candidate._reference_command(args, manifest, output_root)
                if engine == "FROZEN_PARENT"
                else candidate._source_command(
                    args,
                    manifest,
                    candidate.REFERENCE_DIFFERENTIAL_ROWS,
                    output_root,
                    run_root / "measurement/ENGINE_MONITORING_METRICS.json",
                )
            )
            measurement = candidate.run_measured_command(
                command,
                output_root=run_root / "measurement",
                hard_timeout_seconds=candidate.SOURCE_WALL_LIMIT_SECONDS,
            )
            if not _measurement_passes(
                measurement,
                wall_limit=candidate.SOURCE_WALL_LIMIT_SECONDS,
                rss_limit=candidate.RSS_LIMIT_BYTES,
            ):
                return {
                    "status": "FAIL",
                    "reason": f"{engine}_RUN_{index}_RESOURCE_OR_PROCESS_GATE",
                    "rows_per_member": candidate.REFERENCE_DIFFERENTIAL_ROWS,
                    "runs": runs,
                    "failed_measurement": measurement,
                }
            result = dict(candidate._load_worker_result(output_root / "WORKER_RESULT.json"))
            runs[engine].append(
                {
                    "run_index": index,
                    "result": result,
                    "measurement": measurement,
                    "full_tree_hash_map": candidate.full_tree_hash_map(output_root),
                }
            )
    parent_median = _median_three(
        [row["measurement"]["wall_seconds"] for row in runs["FROZEN_PARENT"]]
    )
    v5_median = _median_three(
        [row["measurement"]["wall_seconds"] for row in runs["V5"]]
    )
    semantic_hashes = {
        row["result"]["semantic_canonical_sha256"]
        for engine_rows in runs.values()
        for row in engine_rows
    }
    visual_hashes = {
        row["result"]["visual_canonical_sha256"]
        for engine_rows in runs.values()
        for row in engine_rows
    }
    parent_trees = {
        candidate.canonical_sha256(row["full_tree_hash_map"])
        for row in runs["FROZEN_PARENT"]
    }
    v5_trees = {
        candidate.canonical_sha256(row["full_tree_hash_map"])
        for row in runs["V5"]
    }
    checks = {
        "three_fresh_frozen_parent_processes": len(
            {row["measurement"]["root_pid"] for row in runs["FROZEN_PARENT"]}
        ) == 3,
        "three_fresh_v5_processes": len(
            {row["measurement"]["root_pid"] for row in runs["V5"]}
        ) == 3,
        "all_processes_wall_le_150_seconds": all(
            row["measurement"]["wall_seconds"] <= candidate.SOURCE_WALL_LIMIT_SECONDS
            for engine_rows in runs.values()
            for row in engine_rows
        ),
        "all_processes_peak_rss_le_4gib": all(
            row["measurement"]["peak_process_tree_rss_bytes"] <= candidate.RSS_LIMIT_BYTES
            for engine_rows in runs.values()
            for row in engine_rows
        ),
        "frozen_parent_three_run_complete_tree_exact": len(parent_trees) == 1,
        "v5_three_run_complete_tree_exact": len(v5_trees) == 1,
        "cross_engine_semantic_canonical_bytes_exact": len(semantic_hashes) == 1,
        "cross_engine_visual_canonical_bytes_exact": len(visual_hashes) == 1,
        "v5_median_strictly_faster_than_frozen_parent": v5_median < parent_median,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "rows_per_member": candidate.REFERENCE_DIFFERENTIAL_ROWS,
        "fixture_policy": "minimum_actionable_clean_source_interface_fixture_frozen_before_measurement",
        "runs": runs,
        "checks": checks,
        "frozen_parent_median_wall_seconds": parent_median,
        "v5_median_wall_seconds": v5_median,
        "median_speedup_ratio_parent_over_v5": parent_median / max(v5_median, 1e-12),
    }


def _run_qualification(
    args: argparse.Namespace,
    *,
    lock: Mapping[str, Any],
    lock_sha256: str,
    candidate: Any,
    journal: AttemptJournal,
) -> Mapping[str, Any]:
    result_root = args.result_root
    if not result_root.is_dir() or journal.path.parent != result_root:
        raise ValueError("EXP33 durable attempt was not consumed before execution")
    manifest = args.envelope_root / "BENCHMARK_MANIFEST.json"

    full_runs: list[dict[str, Any]] = []
    with (
        _candidate_worker(candidate, lock),
        _strict_candidate_measurements(candidate),
        _journal_candidate_launches(candidate, journal),
    ):
        for index in range(1, 4):
            run_root = result_root / "full" / f"run_{index}"
            source_root = run_root / "source"
            source_measurement = candidate.run_measured_command(
                candidate._source_command(
                    args,
                    manifest,
                    candidate.DEFAULT_ROWS,
                    source_root,
                    run_root / "source_measurement/ENGINE_MONITORING_METRICS.json",
                ),
                output_root=run_root / "source_measurement",
                hard_timeout_seconds=candidate.SOURCE_WALL_LIMIT_SECONDS,
            )
            if not _measurement_passes(
                source_measurement,
                wall_limit=candidate.SOURCE_WALL_LIMIT_SECONDS,
                rss_limit=candidate.RSS_LIMIT_BYTES,
            ):
                inner = _candidate_failure(
                    "FAIL_STAGED_SOURCE_RUN_STOP",
                    failed_full_run=index,
                    source_measurement=source_measurement,
                )
                return _publish(
                    result_root=result_root,
                    candidate_payload=inner,
                    lock=lock,
                    lock_sha256=lock_sha256,
                )
            source = dict(candidate._load_worker_result(source_root / "WORKER_RESULT.json"))
            source["scientific_tree_canonical_sha256"] = candidate._scientific_tree_identity(source)
            source_tree = candidate.full_tree_hash_map(source_root)
            renderer_root = run_root / "renderer"
            renderer_measurement = candidate.run_measured_command(
                candidate._render_command(
                    args, source_root / "VISUAL_PAYLOAD.json", renderer_root
                ),
                output_root=run_root / "renderer_measurement",
                hard_timeout_seconds=candidate.RENDER_WALL_LIMIT_SECONDS,
            )
            if not _measurement_passes(
                renderer_measurement,
                wall_limit=candidate.RENDER_WALL_LIMIT_SECONDS,
                rss_limit=candidate.RSS_LIMIT_BYTES,
            ):
                inner = _candidate_failure(
                    "FAIL_STAGED_RENDERER_RUN_STOP",
                    failed_full_run=index,
                    source=source,
                    source_measurement=source_measurement,
                    renderer_measurement=renderer_measurement,
                )
                return _publish(
                    result_root=result_root,
                    candidate_payload=inner,
                    lock=lock,
                    lock_sha256=lock_sha256,
                )
            renderer = dict(candidate._load_worker_result(renderer_root / "WORKER_RESULT.json"))
            full_runs.append(
                {
                    "run_index": index,
                    "source": source,
                    "source_measurement": source_measurement,
                    "source_full_tree_hash_map": source_tree,
                    "renderer": renderer,
                    "renderer_measurement": renderer_measurement,
                    "renderer_full_tree_hash_map": candidate.full_tree_hash_map(renderer_root),
                }
            )

        half_root = result_root / "half" / "run_1"
        half_measurement = candidate.run_measured_command(
            candidate._source_command(
                args,
                manifest,
                candidate.HALF_ROWS,
                half_root / "source",
                half_root / "source_measurement/ENGINE_MONITORING_METRICS.json",
            ),
            output_root=half_root / "source_measurement",
            hard_timeout_seconds=candidate.SOURCE_WALL_LIMIT_SECONDS,
        )
        if not _measurement_passes(
            half_measurement,
            wall_limit=candidate.SOURCE_WALL_LIMIT_SECONDS,
            rss_limit=candidate.RSS_LIMIT_BYTES,
        ):
            inner = _candidate_failure(
                "FAIL_STAGED_HALF_SCALE_RUN_STOP",
                full_runs=full_runs,
                half_source_measurement=half_measurement,
            )
            return _publish(
                result_root=result_root,
                candidate_payload=inner,
                lock=lock,
                lock_sha256=lock_sha256,
            )
        half_source = dict(
            candidate._load_worker_result(half_root / "source/WORKER_RESULT.json")
        )
        half_source["scientific_tree_canonical_sha256"] = (
            candidate._scientific_tree_identity(half_source)
        )
        half_run = {
            "source": half_source,
            "source_measurement": half_measurement,
            "source_full_tree_hash_map": candidate.full_tree_hash_map(half_root / "source"),
        }
        differential = _run_reference_differential(
            args, manifest, result_root / "reference_differential", candidate
        )
        if differential.get("status") == "PASS":
            journal.assert_full_schedule_complete()

    per_run_bundle_sizes = []
    for index in range(1, 4):
        run_root = result_root / "full" / f"run_{index}"
        source_logical, source_allocated = candidate._directory_sizes(run_root / "source")
        render_logical, render_allocated = candidate._directory_sizes(run_root / "renderer")
        per_run_bundle_sizes.append(
            {
                "run_index": index,
                "logical_bytes": source_logical + render_logical,
                "allocated_bytes": source_allocated + render_allocated,
            }
        )
    maximum_logical = max(row["logical_bytes"] for row in per_run_bundle_sizes)
    maximum_allocated = max(row["allocated_bytes"] for row in per_run_bundle_sizes)
    total_logical, total_allocated = candidate._directory_sizes(result_root)
    gates = candidate.evaluate_gates(
        full_runs=full_runs,
        half_run=half_run,
        differential=differential,
        bundle_logical_bytes=maximum_logical,
        bundle_allocated_bytes=maximum_allocated,
    )
    inner = {
        "schema": CANDIDATE_SCHEMA,
        "experiment_id": CANDIDATE_EXPERIMENT_ID,
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_EXECUTION_EQUIVALENT",
        "status": CANDIDATE_PASS if gates["status"] == "PASS" else "FAIL_PRODUCTION_SCALE_SOURCE_FREE",
        "retained_rows_per_member": candidate.DEFAULT_ROWS,
        "cad_node_count": candidate.EXPECTED_NODE_COUNT,
        "cad_edge_count": candidate.EXPECTED_EDGE_COUNT,
        "benchmark_manifest_sha256": sha256_path(manifest),
        "qualification_lock_sha256": lock_sha256,
        "full_runs": full_runs,
        "half_run": half_run,
        "reference_differential": differential,
        "artifact_bundle_accounting": {
            "gate_denominator": "maximum deterministic source-plus-renderer output bytes for any one full production-scale run",
            "per_full_run": per_run_bundle_sizes,
            "qualification_evidence_total_logical_bytes_not_gate_denominator": total_logical,
            "qualification_evidence_total_allocated_bytes_not_gate_denominator": total_allocated,
        },
        "gates": gates,
        "physical_scan_or_evaluator_input_used": False,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_EXECUTION_FEASIBILITY_NOT_F1_PROGRESS",
    }
    return _publish(
        result_root=result_root,
        candidate_payload=inner,
        lock=lock,
        lock_sha256=lock_sha256,
    )


def assert_frozen_candidate_gates(candidate: Any) -> None:
    expected = {
        "DEFAULT_ROWS": 1_000_000,
        "HALF_ROWS": 500_000,
        "REFERENCE_DIFFERENTIAL_ROWS": 100_000,
        "SOURCE_WALL_LIMIT_SECONDS": 150.0,
        "RENDER_WALL_LIMIT_SECONDS": 60.0,
        "RSS_LIMIT_BYTES": 4 * 1024**3,
        "BUNDLE_LIMIT_BYTES": 2 * 1024**3,
        "SCALING_LIMIT": 2.5,
    }
    drift = {key: getattr(candidate, key, None) for key, value in expected.items() if getattr(candidate, key, None) != value}
    if drift:
        raise ValueError(f"candidate qualification gate constants drifted: {drift}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--cad-nodes", required=True, type=Path)
    parser.add_argument("--cad-edges", required=True, type=Path)
    parser.add_argument("--edge-geodesic-mm", required=True, type=Path)
    parser.add_argument("--cad-mesh", required=True, type=Path)
    parser.add_argument("--v5-preregistration", required=True, type=Path)
    parser.add_argument("--parent-preregistration", required=True, type=Path)
    parser.add_argument("--cad-preflight", required=True, type=Path)
    parser.add_argument("--source-free-runner", required=True, type=Path)
    parser.add_argument("--frozen-parent-runner", required=True, type=Path)
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument("--qualification-lock", required=True, type=Path)
    parser.add_argument("--qualification-lock-sha256", required=True)
    parser.add_argument("--post-lock-approval", required=True, type=Path)
    parser.add_argument("--post-lock-approval-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = verify_lock(args.qualification_lock, args.qualification_lock_sha256)
    verify_self_binding(Path(__file__), "exp33_harness", lock)
    verify_harness_cli(args, lock)
    verify_detached_postlock_approval(
        args.post_lock_approval,
        args.post_lock_approval_sha256,
        qualification_lock=args.qualification_lock,
        qualification_lock_sha256=args.qualification_lock_sha256,
    )
    journal = AttemptJournal.consume(
        result_root=args.result_root,
        lock_path=args.qualification_lock,
        lock_sha256=args.qualification_lock_sha256,
        approval_path=args.post_lock_approval,
        approval_sha256=args.post_lock_approval_sha256,
    )
    result_path = args.result_root / "EXP_CAD_0033_QUALIFICATION_RESULT.json"
    try:
        environment = verify_current_environment(lock)
        worker_path = ROOT / lock["bound_files"]["exp33_worker"]["path"]
        if sha256_path(worker_path) != lock["bound_files"]["exp33_worker"]["sha256"]:
            raise ValueError("EXP33 worker drifted after durable claim")
        journal.bind_verified_environment(environment, worker_path=worker_path)
        journal.mark_candidate_import()
        candidate = _load_candidate_harness(lock)
        journal.mark_candidate_import(completed=True)
        assert_frozen_candidate_gates(candidate)
        result = _run_qualification(
            args,
            lock=lock,
            lock_sha256=args.qualification_lock_sha256,
            candidate=candidate,
            journal=journal,
        )
        journal.terminal(status=result["status"], result_path=result_path)
    except BaseException as error:
        inner = _candidate_failure(
            "FAIL_CONTROLLER_EXCEPTION",
            failure_stage="POST_LOCK_CONTROLLER",
            error_type=type(error).__name__,
            error_message=str(error),
            durable_attempt_consumed=True,
        )
        try:
            result = _publish(
                result_root=args.result_root,
                candidate_payload=inner,
                lock=lock,
                lock_sha256=args.qualification_lock_sha256,
            )
            journal.terminal(
                status=OUTER_FAIL, result_path=result_path, error=error
            )
        except BaseException as publication_error:
            journal.terminal(
                status="FAIL_TERMINAL_ACCOUNTING_PUBLICATION_EXCEPTION",
                result_path=result_path if result_path.exists() else None,
                error=publication_error,
            )
            raise
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == OUTER_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
