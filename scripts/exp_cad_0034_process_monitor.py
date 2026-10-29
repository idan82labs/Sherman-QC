#!/usr/bin/env python3
"""External wall/RSS watchdog for EXP_CAD_0034 child processes.

This module has no knowledge of source identities or scan paths.  It supervises
one fresh process group, samples its complete process-tree RSS, and writes all
non-deterministic telemetry beneath a caller-provided measurement directory.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence


SOURCE_WALL_LIMIT_SECONDS = 180.0
RENDERER_WALL_LIMIT_SECONDS = 60.0
RSS_LIMIT_BYTES = 4 * 1024**3
MEMORY_SAMPLING_INTERVAL_SECONDS = 0.05


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish strict canonical JSON plus one terminal newline."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
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


def _process_table() -> dict[int, tuple[int, int]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("process-table RSS sampler failed")
    rows: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            rows[int(fields[0])] = (int(fields[1]), int(fields[2]) * 1024)
        except ValueError:
            continue
    return rows


def process_tree_rss_bytes(root_pid: int) -> int:
    rows = _process_table()
    members = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in rows.items():
            if parent in members and pid not in members:
                members.add(pid)
                changed = True
    return sum(rows.get(pid, (0, 0))[1] for pid in members)


def parse_time_l_max_rss(text: str) -> int | None:
    mac = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", text, re.MULTILINE)
    if mac:
        return int(mac.group(1))
    linux = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if linux:
        return int(linux.group(1)) * 1024
    return None


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_measured_command(
    command: Sequence[str],
    *,
    measurement_root: Path,
    wall_limit_seconds: float,
    rss_limit_bytes: int = RSS_LIMIT_BYTES,
) -> dict[str, Any]:
    """Run one process group and fail closed on inclusive wall/RSS limits."""

    if wall_limit_seconds <= 0.0 or not math.isfinite(wall_limit_seconds):
        raise ValueError("wall limit must be finite and positive")
    if rss_limit_bytes <= 0:
        raise ValueError("RSS limit must be positive")
    root = Path(measurement_root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"measurement root must be absent or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    stdout_path = root / "stdout.txt"
    stderr_path = root / "stderr.txt"
    time_path = root / "time_l.txt"
    result_path = root / "PROCESS_MEASUREMENT.json"
    wrapped = ["/usr/bin/time", "-l", "-o", str(time_path), *map(str, command)]
    started = time.monotonic()
    timed_out = False
    rss_limit_exceeded = False
    sampler_failed = False
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
                try:
                    current_rss = process_tree_rss_bytes(process.pid)
                except Exception:
                    sampler_failed = True
                    _terminate_process_group(process)
                    break
                peak_tree = max(peak_tree, current_rss)
                samples += 1
                if current_rss > rss_limit_bytes:
                    rss_limit_exceeded = True
                    _terminate_process_group(process)
                    break
                if time.monotonic() - started > wall_limit_seconds:
                    timed_out = True
                    _terminate_process_group(process)
                    break
                time.sleep(MEMORY_SAMPLING_INTERVAL_SECONDS)
        except BaseException:
            _terminate_process_group(process)
            process.wait()
            raise
        returncode = process.wait()
    wall = time.monotonic() - started
    time_text = time_path.read_text(encoding="utf-8") if time_path.is_file() else ""
    exact_process_rss = parse_time_l_max_rss(time_text)
    peak_rss = max(peak_tree, exact_process_rss or 0)
    # A final /usr/bin/time observation can expose a between-sample RSS breach.
    rss_limit_exceeded = rss_limit_exceeded or peak_rss > rss_limit_bytes
    # A successful child must itself finish at or below the inclusive wall cap.
    timed_out = timed_out or wall > wall_limit_seconds
    passed = bool(
        returncode == 0
        and not timed_out
        and not rss_limit_exceeded
        and not sampler_failed
        and wall <= wall_limit_seconds
        and peak_rss <= rss_limit_bytes
    )
    result = {
        "schema": "EXP_CAD_0034_PROCESS_MEASUREMENT_V1",
        "status": "PASS" if passed else "FAIL_TERMINAL",
        "command": list(map(str, command)),
        "root_pid": int(process.pid),
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "rss_limit_exceeded": bool(rss_limit_exceeded),
        "sampler_failed": bool(sampler_failed),
        "resource_kill_reason": (
            "RSS_LIMIT"
            if rss_limit_exceeded
            else "WALL_LIMIT"
            if timed_out
            else "RSS_SAMPLER_FAILURE"
            if sampler_failed
            else None
        ),
        "wall_seconds": float(wall),
        "wall_limit_seconds": float(wall_limit_seconds),
        "peak_process_tree_rss_bytes": int(peak_rss),
        "rss_limit_bytes": int(rss_limit_bytes),
        "sampled_process_tree_peak_rss_bytes": int(peak_tree),
        "exact_worker_process_max_rss_bytes": exact_process_rss,
        "process_tree_sample_count": int(samples),
        "process_tree_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "time_l_path": str(time_path.resolve()),
    }
    canonical_json_write(result_path, result)
    return result


if __name__ == "__main__":
    raise SystemExit("EXP34 monitor is an import-only orchestration dependency")
