"""Shared helpers for the broad-baseline wrappers (run_lane, to_eval_format, score).

Research-only. Nothing here edits backend/, domains/bend/ or scripts/; those
modules are imported read-only so the evaluator, gate and lane code run exactly
as they exist on disk. Bytecode writing is disabled before any project import so
no __pycache__ appears inside the owned source trees.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Same precedence as scripts/run_f1_accuracy_validation.py (scripts, backend, ROOT
# prepended): legacy modules import each other as top-level names, and
# `scripts.*` / `backend.*` resolve through ROOT.
for _p in (str(ROOT), str(ROOT / "backend"), str(ROOT / "scripts")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

APP_PYTHON = ROOT / ".venv" / "bin" / "python"
SOLVER_PYTHON = ROOT / ".venv-f1-structured" / "bin" / "python"
DEFAULT_RUNLOG = ROOT / "docs" / "production" / "runlog" / "broad_baseline"
GOLD_CATALOG = ROOT / "data" / "bend_corpus" / "annotations" / "bend_line_gold_catalog_v1.json"
AI_GOLD_ROOT = ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_v1"
BENCHMARK_V2 = ROOT / "docs" / "production" / "runlog" / "benchmark_v2" / "BENCHMARK_V2.json"
CACHED_CONTROL_INPUTS = ROOT / "docs" / "research" / "gold11_cached_control_v2_attempt" / "work" / "inputs"
CACHED_CONTROL_REFERENCES = (
    ROOT / "docs" / "research" / "gold11_cached_control_v2_attempt" / "work" / "SELECTED_REFERENCES.json"
)

EVAL_SCHEMA = "sherman.broad_baseline.eval_lines.v1"
SCORE_SCHEMA = "sherman.broad_baseline.score_report.v1"
RAW_FRAME = "scan_raw"

# Frozen matcher constants of scripts/evaluate_f1_against_manual_gold.py.
MATCH_ANGLE_DEG = 5.0
MATCH_OFFSET_MM = 4.0
MATCH_GOLD_OVERLAP = 0.60
MATCH_PRED_OVERLAP = 0.60
# Frozen union duplicate rule of scripts/diagnose_f1_cad_scan_fusion_gold11.py.
FUSION_ANGLE_DEG = 5.0
FUSION_OFFSET_MM = 4.0
FUSION_SHORTER_OVERLAP = 0.60
# Frozen raw-scan support gate of scripts/apply_f1_cad_guided_scan_frame_gate.py.
GATE_CONSTANTS = {
    "support_sample_count": 201,
    "support_distance_mm": 3.0,
    "minimum_supported_fraction": 0.10,
    "minimum_supported_span_mm": 8.0,
    "scan_point_cap": 200_000,
}
TRANSFORM_RESIDUAL_MAX_MM = 0.01

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


# --------------------------------------------------------------------------- io

def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ resources

def python_rss_gib() -> float:
    """Total RSS of every process whose command name contains 'python'."""
    out = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True, check=False).stdout
    total_kib = 0
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and "python" in parts[1].lower():
            try:
                total_kib += int(parts[0])
            except ValueError:
                pass
    return total_kib / (1024.0 * 1024.0)


def wait_for_rss_budget(limit_gib: float, *, poll_s: float = 15.0, max_wait_s: float = 3600.0,
                        log=print) -> dict[str, Any]:
    started = time.monotonic()
    waits = 0
    while True:
        current = python_rss_gib()
        if current < limit_gib:
            return {"python_rss_gib_at_start": round(current, 3), "rss_waits": waits,
                    "rss_wait_s": round(time.monotonic() - started, 1), "rss_limit_gib": limit_gib}
        if time.monotonic() - started > max_wait_s:
            raise RuntimeError(f"python RSS stayed at {current:.2f} GiB >= {limit_gib} GiB for {max_wait_s}s")
        waits += 1
        log(f"[rss-gate] python RSS {current:.2f} GiB >= {limit_gib} GiB; waiting {poll_s:.0f}s", flush=True)
        time.sleep(poll_s)


def _pgid_rss_bytes(pgid: int) -> int:
    out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, check=False).stdout
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == str(pgid):
            try:
                total += int(parts[1]) * 1024
            except ValueError:
                pass
    return total


_TIME_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
_TIME_REAL = re.compile(r"^\s*([0-9.]+)\s+real\s+([0-9.]+)\s+user\s+([0-9.]+)\s+sys", re.MULTILINE)


def run_monitored(cmd: Sequence[str], *, name: str, log_dir: Path, env: Mapping[str, str] | None = None,
                  cwd: Path = ROOT, timeout_s: float = 600.0, rss_limit_gib: float = 9.0,
                  sample_s: float = 0.5, log=print) -> dict[str, Any]:
    """Run one heavy step under /usr/bin/time -l in its own process group.

    Before starting, wait until total python RSS < rss_limit_gib. The whole
    group is SIGKILLed at timeout_s. Reports wall time, /usr/bin/time's
    maximum RSS of the child, and the sampled peak RSS of the process group
    (includes grandchildren such as the CP-SAT solver).
    """
    gate = wait_for_rss_budget(rss_limit_gib, log=log)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    merged_env = dict(os.environ)
    merged_env.update(THREAD_ENV)
    if env:
        merged_env.update(env)
    full = ["/usr/bin/time", "-l", *[str(part) for part in cmd]]
    log(f"[{name}] start: {' '.join(str(c) for c in cmd)[:400]}", flush=True)
    started = time.monotonic()
    peak_group = 0
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(full, cwd=str(cwd), env=merged_env, stdout=out, stderr=err,
                                start_new_session=True)
        while True:
            try:
                rc = proc.wait(timeout=sample_s)
                break
            except subprocess.TimeoutExpired:
                peak_group = max(peak_group, _pgid_rss_bytes(proc.pid))
                if time.monotonic() - started > timeout_s:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    rc = proc.wait()
                    break
    wall = time.monotonic() - started
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    rss_match = _TIME_RSS.search(stderr_text)
    real_match = _TIME_REAL.search(stderr_text)
    record = {
        "name": name,
        "command": [str(part) for part in cmd],
        "returncode": rc,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "wall_s": round(wall, 3),
        "time_real_s": float(real_match.group(1)) if real_match else None,
        "time_user_s": float(real_match.group(2)) if real_match else None,
        "time_sys_s": float(real_match.group(3)) if real_match else None,
        "time_max_rss_bytes": int(rss_match.group(1)) if rss_match else None,
        "sampled_peak_group_rss_bytes": int(peak_group),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        **gate,
    }
    status = "TIMEOUT" if timed_out else ("OK" if rc == 0 else "FAILED")
    record["status"] = status
    peak = max(record["time_max_rss_bytes"] or 0, peak_group)
    log(f"[{name}] {status} rc={rc} wall={wall:.1f}s peak_rss={peak / 2**30:.2f}GiB", flush=True)
    return record


# ---------------------------------------------------------------- geometry

def endpoints(record: Mapping[str, Any]) -> np.ndarray:
    """Exactly scripts/evaluate_f1_against_manual_gold._endpoints precedence."""
    from scripts.evaluate_f1_against_manual_gold import _endpoints  # read-only import

    return _endpoints(dict(record))


def as_line(bend_id: str, start: Sequence[float], end: Sequence[float], **meta: Any) -> dict[str, Any]:
    a = [float(v) for v in start]
    b = [float(v) for v in end]
    return {"bend_id": str(bend_id), "line_start": a, "line_end": b, **meta}


def valid_line(start: Sequence[float] | None, end: Sequence[float] | None) -> bool:
    if start is None or end is None:
        return False
    try:
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return a.shape == (3,) and b.shape == (3,) and bool(np.isfinite(a).all() and np.isfinite(b).all()) \
        and float(np.linalg.norm(b - a)) > 1e-9


def apply_h(transform: np.ndarray, points: Iterable[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 3)
    homogeneous = np.c_[pts, np.ones(len(pts))]
    mapped = homogeneous @ matrix.T
    return mapped[:, :3] / mapped[:, 3:4]


def validate_h(transform: Any) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10):
        raise ValueError("transform must be affine (last row 0 0 0 1)")
    if np.linalg.det(matrix[:3, :3]) <= 1e-12:
        raise ValueError("transform must be orientation preserving and non-singular")
    return matrix


def similarity_report(transform: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(transform, dtype=np.float64)
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    rotation = linear / scale if scale > 0 else linear
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return {
        "scale": scale,
        "rotation_deg": math.degrees(math.acos(cos_angle)),
        "translation_mm": float(np.linalg.norm(matrix[:3, 3])),
        "orthogonality_error": float(np.abs(rotation.T @ rotation - np.eye(3)).max()),
    }


def load_scan_vertices(path: Path) -> np.ndarray:
    """Raw scan coordinates as stored in the file (points, else mesh vertices)."""
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        mesh = o3d.io.read_triangle_mesh(str(path))
        points = np.asarray(mesh.vertices, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError(f"scan contains no finite geometry: {path}")
    return points


# -------------------------------------------------------------- eval format

def eval_payload(*, lane: str, lines: list[dict[str, Any]], scan_path: Path | None, scan_sha256: str | None,
                 source_kind: str, source_path: Path | None, conversion: Mapping[str, Any],
                 dropped: list[dict[str, Any]] | None = None, status: str = "OK",
                 failure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema": EVAL_SCHEMA,
        "lane": lane,
        "status": status,
        "frame": RAW_FRAME,
        "frame_definition": "coordinates of the scan file exactly as stored (source_ply_raw == scan_raw)",
        "units": "scan file units (mm for every Sherman scan inspected so far; not re-estimated)",
        "scan": {"path": None if scan_path is None else str(scan_path), "sha256": scan_sha256},
        "source": {
            "kind": source_kind,
            "path": None if source_path is None else str(source_path),
            "sha256": None if source_path is None or not Path(source_path).is_file() else sha256_file(source_path),
        },
        "conversion": dict(conversion),
        "line_count": len(lines),
        # Key name kept so scripts/evaluate_f1_against_manual_gold.evaluate() can read the file.
        "owned_bend_regions": lines,
        "dropped": list(dropped or []),
        "failure": None if failure is None else dict(failure),
        "created_at": utc_now(),
    }
    return payload


def load_eval_lines(path: Path | str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != EVAL_SCHEMA:
        raise ValueError(f"{path}: not a {EVAL_SCHEMA} file (run to_eval_format.py first)")
    if payload.get("frame") != RAW_FRAME:
        raise ValueError(f"{path}: frame {payload.get('frame')!r} is not {RAW_FRAME}")
    return payload
