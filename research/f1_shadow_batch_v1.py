"""Sequential production-shaped shadow-batch orchestration for F1 jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from research.f1_isolated_job_v1 import JobLimits, run_isolated_job


SCHEMA = "F1_SHADOW_BATCH_SPEC_V1"
SUMMARY_SCHEMA = "F1_SHADOW_BATCH_SUMMARY_V1"
HEX = frozenset("0123456789abcdef")
PROHIBITED = frozenset({"part_id", "part_key", "filename", "expected_count", "truth", "gold", "family_id"})


class ShadowBatchError(ValueError):
    pass


def _hex(value: Any, length: int, path: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in HEX for char in value):
        raise ShadowBatchError(f"{path} must be lowercase hexadecimal length {length}")
    return value


def _reject_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in PROHIBITED:
                raise ShadowBatchError(f"{path}.{key} is prohibited")
            _reject_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_keys(child, f"{path}[{index}]")


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_batch(payload: Any, output: Path) -> tuple[int, dict[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ShadowBatchError(f"schema must be {SCHEMA}")
    _reject_keys(payload)
    batch_id = _hex(payload.get("batch_id"), 32, "batch_id")
    lane = payload.get("lane")
    if lane not in {"SCAN_ONLY", "CAD_ASSISTED"}:
        raise ShadowBatchError("invalid lane")
    raw_limits = payload.get("limits")
    if not isinstance(raw_limits, Mapping):
        raise ShadowBatchError("limits must be an object")
    try:
        limits = JobLimits(
            timeout_seconds=float(raw_limits["timeout_seconds"]),
            rss_limit_bytes=int(raw_limits["rss_limit_bytes"]),
            log_limit_bytes=int(raw_limits.get("log_limit_bytes", 16 * 1024**2)),
            poll_interval_seconds=float(raw_limits.get("poll_interval_seconds", 0.1)),
            termination_grace_seconds=float(raw_limits.get("termination_grace_seconds", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ShadowBatchError("invalid limits") from exc
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ShadowBatchError("jobs must be non-empty")
    seen = set()
    normalized = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            raise ShadowBatchError(f"jobs[{index}] must be an object")
        job_id = _hex(job.get("job_id"), 32, f"jobs[{index}].job_id")
        if job_id in seen:
            raise ShadowBatchError("duplicate job_id")
        seen.add(job_id)
        command = job.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(value, str) or not value for value in command):
            raise ShadowBatchError(f"jobs[{index}].command must be a non-empty string list")
        inputs = job.get("input_hashes")
        if not isinstance(inputs, Mapping) or not inputs:
            raise ShadowBatchError(f"jobs[{index}].input_hashes must be non-empty")
        normalized.append({
            "job_id": job_id,
            "command": command,
            "input_hashes": {str(key): _hex(value, 64, f"jobs[{index}].input_hashes.{key}") for key, value in inputs.items()},
            "detector_sha256": _hex(job.get("detector_sha256"), 64, "detector_sha256"),
            "config_sha256": _hex(job.get("config_sha256"), 64, "config_sha256"),
            "ontology_sha256": _hex(job.get("ontology_sha256"), 64, "ontology_sha256"),
            "artifacts": job.get("artifacts", {}),
        })
    receipts = []
    for job in normalized:  # Deliberately sequential: one memory-heavy bend job at a time.
        artifacts_raw = job["artifacts"]
        if not isinstance(artifacts_raw, Mapping) or any(not isinstance(role, str) or not isinstance(path, str) for role, path in artifacts_raw.items()):
            raise ShadowBatchError("artifacts must map roles to paths")
        job_root = output / "jobs" / job["job_id"]
        _code, receipt = run_isolated_job(
            job_id=job["job_id"],
            lane=lane,
            command=job["command"],
            receipt_path=job_root / "receipt.json",
            log_directory=job_root / "logs",
            input_hashes=job["input_hashes"],
            detector_sha256=job["detector_sha256"],
            config_sha256=job["config_sha256"],
            ontology_sha256=job["ontology_sha256"],
            artifacts={role: Path(path) for role, path in artifacts_raw.items()},
            limits=limits,
        )
        receipts.append(receipt)
    walls = sorted(float(receipt["wall_seconds"]) for receipt in receipts)
    p95 = walls[max(0, math.ceil(0.95 * len(walls)) - 1)]
    statuses = {status: sum(receipt["status"] == status for receipt in receipts) for status in ("COMPLETE", "ABSTAIN", "TIMEOUT", "RESOURCE_STOP", "FAILED")}
    compact = [{
        "job_id": receipt["job_id"],
        "status": receipt["status"],
        "reason": receipt["reason"],
        "wall_seconds": receipt["wall_seconds"],
        "peak_process_group_rss_bytes": receipt["peak_process_group_rss_bytes"],
        "receipt_sha256": hashlib.sha256((output / "jobs" / receipt["job_id"] / "receipt.json").read_bytes()).hexdigest(),
    } for receipt in receipts]
    successful_jobs = statuses["COMPLETE"] + statuses["ABSTAIN"]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "batch_id": batch_id,
        "lane": lane,
        "jobs_attempted": len(receipts),
        "jobs_complete": statuses["COMPLETE"],
        "jobs_abstained": statuses["ABSTAIN"],
        "jobs_succeeded": successful_jobs,
        "job_success_rate": successful_jobs / len(receipts),
        "decision_coverage_rate": statuses["COMPLETE"] / len(receipts),
        "abstention_rate": statuses["ABSTAIN"] / len(receipts),
        "status_counts": statuses,
        "p95_wall_seconds": p95,
        "peak_process_group_rss_bytes": max(receipt["peak_process_group_rss_bytes"] for receipt in receipts),
        "all_jobs_accounted": len(compact) == len(normalized),
        "jobs": compact,
    }
    _atomic(output / "summary.json", summary)
    return (0 if successful_jobs == len(receipts) else 2), summary
