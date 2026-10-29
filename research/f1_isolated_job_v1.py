"""Production-shaped isolated F1 job runner with process-group resource control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence


SCHEMA = "F1_ISOLATED_JOB_RECEIPT_V1"
STATUSES = frozenset({"COMPLETE", "ABSTAIN", "TIMEOUT", "RESOURCE_STOP", "FAILED"})
LANES = frozenset({"SCAN_ONLY", "CAD_ASSISTED"})
HEX = frozenset("0123456789abcdef")
PROHIBITED_ROLE_TOKENS = frozenset({"part", "family", "filename", "identity", "expected", "truth", "gold", "label", "prediction", "score"})
PROHIBITED_COMPACT_ROLE_TOKENS = frozenset({
    "partid", "partkey", "partname", "familyid", "filename", "identity",
    "expectedcount", "expectedbendcount", "truth", "gold", "label", "prediction", "score",
})
SCAN_ONLY_PROHIBITED_INPUT_TOKENS = frozenset({"cad", "step", "overlay"})
LOCKED_ENVIRONMENT = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
ABSTAIN_RETURN_CODE = 3
ABSTAIN_REASON = "DECLARED_ABSTENTION"


class IsolatedJobError(ValueError):
    pass


@dataclass(frozen=True)
class JobLimits:
    timeout_seconds: float = 300.0
    rss_limit_bytes: int = 16 * 1024 ** 3
    log_limit_bytes: int = 16 * 1024 ** 2
    poll_interval_seconds: float = 0.10
    termination_grace_seconds: float = 1.0


def _hex(value: Any, length: int, path: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(item not in HEX for item in value):
        raise IsolatedJobError(f"{path} must be lowercase hexadecimal length {length}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _role(value: Any, path: str, *, lane: str | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 64 or any(not (char.isalnum() or char == "_") for char in value):
        raise IsolatedJobError(f"{path} must use 1-64 alphanumeric/underscore characters")
    folded = value.casefold()
    components = set(folded.split("_"))
    compact = "".join(char for char in folded if char.isalnum())
    if components & PROHIBITED_ROLE_TOKENS or compact in PROHIBITED_COMPACT_ROLE_TOKENS:
        raise IsolatedJobError(f"{path} contains prohibited identity, truth, or feedback semantics")
    if lane == "SCAN_ONLY" and components & SCAN_ONLY_PROHIBITED_INPUT_TOKENS:
        raise IsolatedJobError(f"{path} violates the scan-only lane")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise IsolatedJobError(f"receipt schema must be {SCHEMA}")
    receipt = json.loads(json.dumps(value, allow_nan=False))
    _hex(receipt.get("job_id"), 32, "job_id")
    if receipt.get("lane") not in LANES or receipt.get("status") not in STATUSES:
        raise IsolatedJobError("invalid receipt lane or status")
    for key in ("detector_sha256", "config_sha256", "ontology_sha256", "command_sha256"):
        _hex(receipt.get(key), 64, key)
    inputs = receipt.get("input_hashes")
    if not isinstance(inputs, Mapping) or not inputs:
        raise IsolatedJobError("input_hashes must be a non-empty object")
    for key, digest in inputs.items():
        _role(key, f"input_hashes.{key}", lane=receipt["lane"])
        _hex(digest, 64, f"input_hashes.{key}")
    limits = receipt.get("limits")
    if not isinstance(limits, Mapping) or float(limits.get("timeout_seconds") or 0) <= 0 or int(limits.get("rss_limit_bytes") or 0) <= 0 or int(limits.get("log_limit_bytes") or 0) <= 0:
        raise IsolatedJobError("invalid receipt limits")
    wall = receipt.get("wall_seconds"); rss = receipt.get("peak_process_group_rss_bytes")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or float(wall) < 0:
        raise IsolatedJobError("invalid receipt wall_seconds")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss < 0:
        raise IsolatedJobError("invalid receipt peak RSS")
    status = receipt["status"]; return_code = receipt.get("return_code"); reason = receipt.get("reason")
    if status in {"ABSTAIN", "TIMEOUT", "RESOURCE_STOP", "FAILED"} and not isinstance(reason, str):
        raise IsolatedJobError("non-complete receipt requires a reason")
    for stream in ("stdout", "stderr"):
        row = receipt.get(stream)
        if not isinstance(row, Mapping) or not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
            raise IsolatedJobError(f"invalid {stream} receipt")
        _hex(row.get("sha256"), 64, f"{stream}.sha256")
    stream_bytes = sum(receipt[stream]["bytes"] for stream in ("stdout", "stderr"))
    if status == "COMPLETE" and (return_code != 0 or reason is not None or rss >= int(limits["rss_limit_bytes"]) or float(wall) > float(limits["timeout_seconds"]) or stream_bytes >= int(limits["log_limit_bytes"])):
        raise IsolatedJobError("COMPLETE receipt contradicts exit or resource evidence")
    if status == "ABSTAIN" and (
        return_code != ABSTAIN_RETURN_CODE
        or reason != ABSTAIN_REASON
        or rss >= int(limits["rss_limit_bytes"])
        or float(wall) > float(limits["timeout_seconds"])
        or stream_bytes >= int(limits["log_limit_bytes"])
    ):
        raise IsolatedJobError("ABSTAIN receipt contradicts exit or resource evidence")
    seen_roles: set[str] = set()
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise IsolatedJobError("artifacts must be an array")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("role"), str) or not artifact["role"] or artifact["role"] in seen_roles:
            raise IsolatedJobError("artifact roles must be non-empty and unique")
        _role(artifact["role"], f"artifacts[{index}].role")
        seen_roles.add(artifact["role"])
        if artifact.get("missing") is True:
            if status in {"COMPLETE", "ABSTAIN"}:
                raise IsolatedJobError("successful receipt cannot have missing artifacts")
        else:
            _hex(artifact.get("sha256"), 64, f"artifacts[{index}].sha256")
            if not isinstance(artifact.get("bytes"), int) or artifact["bytes"] < 0:
                raise IsolatedJobError("artifact bytes must be non-negative")
    serialized = json.dumps(receipt, sort_keys=True)
    if any(token in serialized.casefold() for token in ('"part_id"', '"part_key"', '"filename"', '"expected_count"', '"truth"', '"scan_path"', '"cad_path"')):
        raise IsolatedJobError("receipt contains prohibited identity or truth fields")
    return receipt


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _process_group_rss_bytes(pgid: int) -> int:
    result = subprocess.run(
        ["ps", "-axo", "pgid=,rss="], check=False, capture_output=True, text=True,
        env={"PATH": "/bin:/usr/bin"}, timeout=5.0,
    )
    total_kib = 0
    for raw in result.stdout.splitlines():
        values = raw.split()
        if len(values) == 2:
            try:
                row_pgid, rss_kib = int(values[0]), int(values[1])
            except ValueError:
                continue
            if row_pgid == pgid:
                total_kib += max(rss_kib, 0)
    return total_kib * 1024


def _parse_time_peak_rss(path: Path) -> int:
    """Read macOS /usr/bin/time -l post-exit maximum RSS, fail closed to zero."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        if "maximum resident set size" not in line:
            continue
        fields = line.strip().split()
        if fields:
            try:
                return max(0, int(fields[0]))
            except ValueError:
                return 0
    return 0


def _terminate_group(pgid: int, grace_seconds: float) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if _process_group_rss_bytes(pgid) == 0:
            return
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_isolated_job(
    *,
    job_id: str,
    lane: str,
    command: Sequence[str],
    receipt_path: Path,
    log_directory: Path,
    input_hashes: Mapping[str, str],
    detector_sha256: str,
    config_sha256: str,
    ontology_sha256: str,
    artifacts: Mapping[str, Path] | None = None,
    limits: JobLimits = JobLimits(),
    environment: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    _hex(job_id, 32, "job_id")
    if lane not in LANES:
        raise IsolatedJobError("invalid lane")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise IsolatedJobError("command must be a non-empty string array")
    if limits.timeout_seconds <= 0 or limits.rss_limit_bytes <= 0 or limits.log_limit_bytes <= 0 or limits.poll_interval_seconds <= 0:
        raise IsolatedJobError("job limits must be positive")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise IsolatedJobError("input_hashes must be a non-empty object")
    normalized_inputs = {
        _role(key, f"input_hashes.{key}", lane=lane): _hex(value, 64, f"input_hashes.{key}")
        for key, value in sorted(input_hashes.items())
    }
    bound = {
        "detector_sha256": _hex(detector_sha256, 64, "detector_sha256"),
        "config_sha256": _hex(config_sha256, 64, "config_sha256"),
        "ontology_sha256": _hex(ontology_sha256, 64, "ontology_sha256"),
    }
    if artifacts is not None and not isinstance(artifacts, Mapping):
        raise IsolatedJobError("artifacts must be an object")
    normalized_artifacts: dict[str, Path] = {}
    for role, path in (artifacts or {}).items():
        normalized_role = _role(role, f"artifacts.{role}")
        if not isinstance(path, Path):
            raise IsolatedJobError(f"artifacts.{normalized_role} must be a Path")
        normalized_artifacts[normalized_role] = path
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = log_directory / f"{job_id}.stdout.log"
    stderr_path = log_directory / f"{job_id}.stderr.log"
    resource_path = log_directory / f"{job_id}.resource.log"
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    peak_rss = 0
    stop_status: str | None = None
    stop_reason: str | None = None
    env = {"PATH": os.environ.get("PATH", "/bin:/usr/bin"), "LANG": "C", "LC_ALL": "C", **LOCKED_ENVIRONMENT}
    if environment:
        env.update({str(key): str(value) for key, value in environment.items()})
    env.update(LOCKED_ENVIRONMENT)
    command_fingerprint = hashlib.sha256(_canonical_bytes(list(command))).hexdigest()
    time_utility = Path("/usr/bin/time")
    launched_command = [str(time_utility), "-l", "-o", str(resource_path), *command] if platform.system() == "Darwin" and time_utility.is_file() else list(command)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(launched_command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, env=env, start_new_session=True, close_fds=True)
        pgid = process.pid
        while process.poll() is None:
            elapsed = time.monotonic() - started
            current_rss = _process_group_rss_bytes(pgid)
            peak_rss = max(peak_rss, current_rss)
            if current_rss >= limits.rss_limit_bytes:
                stop_status = "RESOURCE_STOP"; stop_reason = "PROCESS_GROUP_RSS_LIMIT"
                _terminate_group(pgid, limits.termination_grace_seconds); break
            if stdout_path.stat().st_size + stderr_path.stat().st_size >= limits.log_limit_bytes:
                stop_status = "RESOURCE_STOP"; stop_reason = "LOG_SIZE_LIMIT"
                _terminate_group(pgid, limits.termination_grace_seconds); break
            if elapsed >= limits.timeout_seconds:
                stop_status = "TIMEOUT"; stop_reason = "WALL_TIME_LIMIT"
                _terminate_group(pgid, limits.termination_grace_seconds); break
            time.sleep(limits.poll_interval_seconds)
        try:
            return_code = process.wait(timeout=max(limits.termination_grace_seconds, 0.1) + 1.0)
        except subprocess.TimeoutExpired:
            _terminate_group(pgid, 0.0); return_code = process.wait(timeout=2.0)
        lingering_group_rss = _process_group_rss_bytes(pgid)
        peak_rss = max(peak_rss, lingering_group_rss)
        orphan_process_group = lingering_group_rss > 0
        if orphan_process_group:
            _terminate_group(pgid, limits.termination_grace_seconds)
    wall_seconds = time.monotonic() - started
    post_exit_peak_rss = _parse_time_peak_rss(resource_path)
    peak_rss = max(peak_rss, post_exit_peak_rss)
    artifact_rows = []
    artifact_missing = False
    for role, path in sorted(normalized_artifacts.items()):
        if path.is_file():
            artifact_rows.append({"role": str(role), "sha256": _sha(path), "bytes": path.stat().st_size})
        else:
            artifact_missing = True
            artifact_rows.append({"role": str(role), "missing": True})
    if stop_status is not None:
        status = stop_status; reason = stop_reason
    elif orphan_process_group:
        status = "FAILED"; reason = "ORPHAN_PROCESS_GROUP"
    elif wall_seconds > limits.timeout_seconds:
        status = "TIMEOUT"; reason = "WALL_TIME_LIMIT_POST_EXIT"
    elif peak_rss >= limits.rss_limit_bytes:
        status = "RESOURCE_STOP"; reason = "PROCESS_GROUP_RSS_LIMIT_POST_EXIT"
    elif stdout_path.stat().st_size + stderr_path.stat().st_size >= limits.log_limit_bytes:
        status = "RESOURCE_STOP"; reason = "LOG_SIZE_LIMIT_POST_EXIT"
    elif artifact_missing:
        status = "FAILED"; reason = "REQUIRED_ARTIFACT_MISSING"
    elif return_code == ABSTAIN_RETURN_CODE:
        status = "ABSTAIN"; reason = ABSTAIN_REASON
    elif return_code != 0:
        status = "FAILED"; reason = "NONZERO_EXIT"
    else:
        status = "COMPLETE"; reason = None
    receipt = {
        "schema": SCHEMA,
        "job_id": job_id,
        "lane": lane,
        "status": status,
        "reason": reason,
        "started_at_utc": started_wall.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall_seconds, 6),
        "peak_process_group_rss_bytes": peak_rss,
        "limits": {"timeout_seconds": limits.timeout_seconds, "rss_limit_bytes": limits.rss_limit_bytes, "log_limit_bytes": limits.log_limit_bytes},
        "return_code": return_code,
        "command_sha256": command_fingerprint,
        "input_hashes": normalized_inputs,
        **bound,
        "stdout": {"sha256": _sha(stdout_path), "bytes": stdout_path.stat().st_size},
        "stderr": {"sha256": _sha(stderr_path), "bytes": stderr_path.stat().st_size},
        "artifacts": artifact_rows,
        "runtime": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "single_threaded": True},
    }
    validate_receipt(receipt)
    _atomic_json(receipt_path, receipt)
    return (0 if status in {"COMPLETE", "ABSTAIN"} else 2), receipt
