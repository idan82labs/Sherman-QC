#!/usr/bin/env python3
"""Deterministic source-free performance/equivalence harness for EXP_CAD_0032.

This harness deliberately has no physical scan or evaluator arguments.  It
prepares two deterministic one-million-row synthetic acquisition envelopes
from the frozen CAD nodes, launches the V5 source worker in three fresh
processes, launches the frozen renderer in three fresh processes, and executes
one half-scale source run.  Runtime, process-tree RSS, operation counters,
canonical scientific hashes, replay identity, scaling, and bundle size are all
fail-closed gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/exp_cad_0032_performance_worker.py"
SCHEMA = "EXP_CAD_0032_PERFORMANCE_EQUIVALENCE_RESULT_V1"
MANIFEST_SCHEMA = "EXP_CAD_0032_SYNTHETIC_ACQUISITION_ENVELOPE_V1"
EXPECTED_NODE_COUNT = 8_077
EXPECTED_EDGE_COUNT = 19_267
DEFAULT_ROWS = 1_000_000
HALF_ROWS = 500_000
REFERENCE_DIFFERENTIAL_ROWS = 100_000
SOURCE_WALL_LIMIT_SECONDS = 150.0
RENDER_WALL_LIMIT_SECONDS = 60.0
RSS_LIMIT_BYTES = 4 * 1024**3
BUNDLE_LIMIT_BYTES = 2 * 1024**3
SCALING_LIMIT = 2.5
MEMORY_SAMPLING_INTERVAL_SECONDS = 0.05
TOKENS = ("SYNTHETIC_BENCH_A", "SYNTHETIC_BENCH_B")
EXPECTED_CAD_SHA256 = {
    "cad_preflight": "e96daf9288f59a8d7f156981f6d13e9fd3204b344523d96bba587bbbd84c8f82",
    "cad_nodes": "575c19d404efd6733051d611dd46f123fbbee3117be13ef001514d658e2618b0",
    "cad_edges": "0bf773fc65af9152bb8491830f98faf2e14c10fe999e8e4bf81a3db687513e3d",
    "cad_edge_geodesic_mm": "92605624e786450ca260134555b6e71bbac56875e9e3e49b4aceac6d5e9de185",
    "cad_mesh": "9e6571704780383e99b6fa4f2800fa69a50a3e187467698fe5dd7414d9d2a716",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _relative_binding(path: Path, manifest_path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "relative_path": os.path.relpath(path, manifest_path.parent.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_path(path),
    }


def validate_frozen_cad_hashes(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for key in ("cad_nodes", "cad_edges", "cad_edge_geodesic_mm", "cad_mesh"):
        row = bindings.get(key)
        if not isinstance(row, Mapping) or row.get("sha256") != EXPECTED_CAD_SHA256[key]:
            raise ValueError(f"EXP32 frozen CAD hash mismatch: {key}")


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return z ^ (z >> np.uint64(31))


def synthetic_block(
    nodes: np.ndarray, start: int, stop: int, *, member_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic, morphology-neutral acquisition stress block."""

    row = np.arange(start, stop, dtype=np.uint64)
    seed = np.uint64(0x6A09E667F3BCC909 if member_index == 0 else 0xBB67AE8584CAA73B)
    # The coprime stride makes every complete 8,077-row cycle visit every CAD
    # node once; member B uses a phase shift but no planted geometric defect.
    node_id = ((row * np.uint64(7_919) + np.uint64(member_index * 2_003)) % np.uint64(len(nodes))).astype(np.int64)
    h1 = _splitmix64(row ^ seed)
    h2 = _splitmix64(h1 ^ np.uint64(0x3C6EF372FE94F82B))
    unit1 = ((h1 >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)) - 0.5
    unit2 = ((h2 >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)) - 0.5
    xyz = np.column_stack((nodes["x"][node_id], nodes["y"][node_id], nodes["z"][node_id]))
    e1 = np.column_stack((nodes["e1x"][node_id], nodes["e1y"][node_id], nodes["e1z"][node_id]))
    e2 = np.column_stack((nodes["e2x"][node_id], nodes["e2y"][node_id], nodes["e2z"][node_id]))
    normal = np.column_stack((nodes["nx"][node_id], nodes["ny"][node_id], nodes["nz"][node_id]))
    # +/-0.60 mm tangential jitter creates a conservative high-cardinality
    # voxel workload without asserting any synthetic bend morphology.
    point = xyz + (1.2 * unit1[:, None]) * e1 + (1.2 * unit2[:, None]) * e2
    sign = np.where((h2 & np.uint64(1)) == 0, 1.0, -1.0)
    normal = normal * sign[:, None]  # exercise unoriented-normal semantics
    return np.asarray(point, dtype="<f8"), np.asarray(normal, dtype="<f8")


def _atomic_memmap_array(
    path: Path,
    nodes: np.ndarray,
    rows: int,
    *,
    member_index: int,
    normals: bool,
    chunk_rows: int = 100_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        output = np.lib.format.open_memmap(
            temporary, mode="w+", dtype="<f8", shape=(rows, 3)
        )
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            points, normal = synthetic_block(
                nodes, start, stop, member_index=member_index
            )
            output[start:stop] = normal if normals else points
        output.flush()
        del output
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_envelope(
    *,
    output_root: Path,
    cad_nodes: Path,
    cad_edges: Path,
    edge_weights: Path,
    cad_mesh: Path,
    rows: int = DEFAULT_ROWS,
) -> Path:
    if rows != DEFAULT_ROWS:
        raise ValueError("EXP32 production-scale envelope is frozen at 1,000,000 rows/member")
    output_root = output_root.resolve()
    manifest_path = output_root / "BENCHMARK_MANIFEST.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != MANIFEST_SCHEMA or payload.get(
            "retained_rows_per_member"
        ) != DEFAULT_ROWS:
            raise ValueError("existing EXP32 benchmark manifest has incompatible identity")
        validate_frozen_cad_hashes(payload["files"])
        for row in payload["files"].values():
            bound = (manifest_path.parent / row["relative_path"]).resolve()
            if sha256_path(bound) != row["sha256"]:
                raise ValueError("existing EXP32 benchmark binding changed")
        return manifest_path
    output_root.mkdir(parents=True, exist_ok=True)
    nodes = np.load(cad_nodes, allow_pickle=False)
    edges = np.load(cad_edges, allow_pickle=False)
    weights = np.load(edge_weights, allow_pickle=False)
    if len(nodes) != EXPECTED_NODE_COUNT or len(edges) != EXPECTED_EDGE_COUNT:
        raise ValueError("EXP32 requires the frozen 8,077-node/19,267-edge graph")
    if weights.shape != (EXPECTED_EDGE_COUNT,):
        raise ValueError("EXP32 edge-geodesic shape mismatch")
    input_hashes = {
        "cad_nodes": {"sha256": sha256_path(cad_nodes)},
        "cad_edges": {"sha256": sha256_path(cad_edges)},
        "cad_edge_geodesic_mm": {"sha256": sha256_path(edge_weights)},
        "cad_mesh": {"sha256": sha256_path(cad_mesh)},
    }
    validate_frozen_cad_hashes(input_hashes)
    source_paths = {
        "first_points": output_root / "FIRST_POINTS.npy",
        "first_normals": output_root / "FIRST_NORMALS.npy",
        "second_points": output_root / "SECOND_POINTS.npy",
        "second_normals": output_root / "SECOND_NORMALS.npy",
    }
    _atomic_memmap_array(source_paths["first_points"], nodes, rows, member_index=0, normals=False)
    _atomic_memmap_array(source_paths["first_normals"], nodes, rows, member_index=0, normals=True)
    _atomic_memmap_array(source_paths["second_points"], nodes, rows, member_index=1, normals=False)
    _atomic_memmap_array(source_paths["second_normals"], nodes, rows, member_index=1, normals=True)
    files = {
        "cad_nodes": _relative_binding(cad_nodes, manifest_path),
        "cad_edges": _relative_binding(cad_edges, manifest_path),
        "cad_edge_geodesic_mm": _relative_binding(edge_weights, manifest_path),
        "cad_mesh": _relative_binding(cad_mesh, manifest_path),
        **{key: _relative_binding(path, manifest_path) for key, path in source_paths.items()},
    }
    payload = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "evidence_class": "SOURCE_FREE_SYNTHETIC_ACQUISITION_PERFORMANCE_ONLY",
        "retained_rows_per_member": rows,
        "half_scale_rows_per_member": rows // 2,
        "opaque_member_tokens": list(TOKENS),
        "cad_node_count": int(len(nodes)),
        "cad_edge_count": int(len(edges)),
        "generator": {
            "algorithm": "CAD_NODE_TANGENT_LOW_DISCREPANCY_V1",
            "node_stride_coprime": 7_919,
            "member_phase_shift": 2_003,
            "tangential_jitter_full_width_mm": 1.2,
            "planted_deformation": False,
            "normal_orientation_sign_is_deterministic_unoriented_stress": True,
            "prefix_nested_half_scale": True,
        },
        "scientific_workload": {
            "robustness_row_count": 32,
            "expected_unique_member_fit_count": 38,
            "thresholds_or_graph_workload_weakened": False,
        },
        "files": files,
        "physical_scan_paths_or_hashes_present": False,
        "evaluator_roles_labels_gold_expected_counts_quality_used": False,
        "generalization_evidence": False,
    }
    atomic_json(manifest_path, payload)
    return manifest_path


def _process_table() -> dict[int, tuple[int, int]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    rows: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
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


def run_measured_command(
    command: Sequence[str], *, output_root: Path, hard_timeout_seconds: float
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "stdout.txt"
    stderr_path = output_root / "stderr.txt"
    time_path = output_root / "time_l.txt"
    wrapped = ["/usr/bin/time", "-l", "-o", str(time_path), *command]
    started = time.monotonic()
    timed_out = False
    peak_tree = 0
    samples = 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            wrapped,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        while process.poll() is None:
            peak_tree = max(peak_tree, process_tree_rss_bytes(process.pid))
            samples += 1
            if time.monotonic() - started > hard_timeout_seconds:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(MEMORY_SAMPLING_INTERVAL_SECONDS)
        return_code = process.wait()
    wall = time.monotonic() - started
    time_text = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    exact_process_rss = parse_time_l_max_rss(time_text)
    peak_rss = max(peak_tree, exact_process_rss or 0)
    return {
        "command": list(command),
        "root_pid": int(process.pid),
        "returncode": int(return_code),
        "timed_out": timed_out,
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": int(peak_rss),
        "sampled_process_tree_peak_rss_bytes": int(peak_tree),
        "exact_worker_process_max_rss_bytes": exact_process_rss,
        "process_tree_sample_count": samples,
        "process_tree_sampling_interval_seconds": MEMORY_SAMPLING_INTERVAL_SECONDS,
        "stdout_sha256": sha256_path(stdout_path),
        "stderr_sha256": sha256_path(stderr_path),
        "time_l_sha256": sha256_path(time_path) if time_path.exists() else None,
    }


def _source_command(
    args: argparse.Namespace,
    manifest: Path,
    rows: int,
    output_root: Path,
    monitoring_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "source",
        "--benchmark-manifest",
        str(manifest),
        "--rows-per-member",
        str(rows),
        "--v5-preregistration",
        str(args.v5_preregistration.resolve()),
        "--parent-preregistration",
        str(args.parent_preregistration.resolve()),
        "--cad-preflight",
        str(args.cad_preflight.resolve()),
        "--source-free-runner",
        str(args.source_free_runner.resolve()),
        "--monitoring-output",
        str(monitoring_output.resolve()),
        "--qualification-lock",
        str(args.qualification_lock.resolve()),
        "--qualification-lock-sha256",
        sha256_path(args.qualification_lock.resolve()),
        "--output-root",
        str(output_root),
    ]


def _render_command(args: argparse.Namespace, visual: Path, output_root: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "render",
        "--visual-payload",
        str(visual),
        "--renderer",
        str(args.renderer.resolve()),
        "--qualification-lock",
        str(args.qualification_lock.resolve()),
        "--qualification-lock-sha256",
        sha256_path(args.qualification_lock.resolve()),
        "--output-root",
        str(output_root),
    ]


def _reference_command(
    args: argparse.Namespace, manifest: Path, output_root: Path
) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "reference",
        "--benchmark-manifest",
        str(manifest),
        "--rows-per-member",
        str(REFERENCE_DIFFERENTIAL_ROWS),
        "--parent-preregistration",
        str(args.parent_preregistration.resolve()),
        "--cad-preflight",
        str(args.cad_preflight.resolve()),
        "--frozen-parent-runner",
        str(args.frozen_parent_runner.resolve()),
        "--qualification-lock",
        str(args.qualification_lock.resolve()),
        "--qualification-lock-sha256",
        sha256_path(args.qualification_lock.resolve()),
        "--output-root",
        str(output_root),
    ]


def _load_worker_result(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "EXP_CAD_0032_PERFORMANCE_WORKER_RESULT_V1" or payload.get(
        "status"
    ) != "COMPLETE":
        raise ValueError(f"invalid performance worker result: {path}")
    return payload


def _scientific_tree_identity(source: Mapping[str, Any]) -> str:
    identity = {
        "semantic_canonical_sha256": source["semantic_canonical_sha256"],
        "visual_canonical_sha256": source["visual_canonical_sha256"],
        "semantic_schema": source["semantic_schema"],
        "visual_schema": source["visual_schema"],
        "retained_row_identity": source["retained_row_identity"],
        "worker_operation_counters": source["worker_operation_counters"],
        "engine_operation_counters": source["engine_operation_counters"],
        "equivalence_checks": source["equivalence_checks"],
    }
    return canonical_sha256(identity)


def _directory_sizes(root: Path) -> tuple[int, int]:
    logical = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    completed = subprocess.run(
        ["du", "-sk", str(root)], capture_output=True, text=True, check=False
    )
    allocated = int(completed.stdout.split()[0]) * 1024 if completed.returncode == 0 else logical
    return int(logical), int(allocated)


def full_tree_hash_map(root: Path) -> dict[str, dict[str, Any]]:
    """Hash every deterministic artifact under a worker output root."""

    return {
        path.relative_to(root).as_posix(): {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_path(path),
        }
        for path in sorted(row for row in root.rglob("*") if row.is_file())
    }


def _metric_bool(source: Mapping[str, Any], key: str) -> bool:
    checks = source.get("equivalence_checks", {})
    return bool(checks.get(key)) if isinstance(checks, Mapping) else False


def evaluate_gates(
    *,
    full_runs: Sequence[Mapping[str, Any]],
    half_run: Mapping[str, Any],
    differential: Mapping[str, Any],
    bundle_logical_bytes: int,
    bundle_allocated_bytes: int,
) -> dict[str, Any]:
    sources = [row["source"] for row in full_runs]
    source_measurements = [row["source_measurement"] for row in full_runs]
    renderers = [row["renderer"] for row in full_runs]
    renderer_measurements = [row["renderer_measurement"] for row in full_runs]
    full_median = float(np.median([row["wall_seconds"] for row in source_measurements]))
    half_wall = float(half_run["source_measurement"]["wall_seconds"])
    scaling_ratio = full_median / max(half_wall, 1e-12)
    semantic_hashes = {row["semantic_canonical_sha256"] for row in sources}
    visual_hashes = {row["visual_canonical_sha256"] for row in sources}
    tree_hashes = {row["scientific_tree_canonical_sha256"] for row in sources}
    literal_source_tree_hashes = {
        canonical_sha256(row["source_full_tree_hash_map"]) for row in full_runs
    }
    literal_render_tree_hashes = {
        canonical_sha256(row["renderer_full_tree_hash_map"]) for row in full_runs
    }
    render_hashes = {row["render_manifest_canonical_sha256"] for row in renderers}
    engine_counter_hashes = {
        canonical_sha256(row["engine_operation_counters"]) for row in sources
    }
    worker_counter_hashes = {
        canonical_sha256(row["worker_operation_counters"]) for row in sources
    }
    unique_fit_ok = all(
        row["engine_operation_counters"].get("unique_member_fit_count") == 38
        for row in sources
    )
    robustness_ok = all(
        row["engine_operation_counters"].get("robustness_row_count") == 32
        for row in sources
    )
    checks = {
        "three_fresh_source_processes": len({row["root_pid"] for row in source_measurements}) == 3,
        "three_fresh_renderer_processes": len({row["root_pid"] for row in renderer_measurements}) == 3,
        "source_wall_le_150_seconds_each": all(
            not row["timed_out"] and row["returncode"] == 0 and row["wall_seconds"] <= SOURCE_WALL_LIMIT_SECONDS
            for row in source_measurements
        ),
        "source_peak_process_tree_rss_le_4gib_each": all(
            row["peak_process_tree_rss_bytes"] <= RSS_LIMIT_BYTES for row in source_measurements
        ),
        "renderer_wall_le_60_seconds_each": all(
            not row["timed_out"] and row["returncode"] == 0 and row["wall_seconds"] <= RENDER_WALL_LIMIT_SECONDS
            for row in renderer_measurements
        ),
        "renderer_peak_process_tree_rss_le_4gib_each": all(
            row["peak_process_tree_rss_bytes"] <= RSS_LIMIT_BYTES for row in renderer_measurements
        ),
        "three_run_semantic_canonical_hash_exact": len(semantic_hashes) == 1,
        "three_run_visual_canonical_hash_exact": len(visual_hashes) == 1,
        "three_run_scientific_tree_hash_exact": len(tree_hashes) == 1,
        "three_run_render_tree_hash_exact": len(render_hashes) == 1,
        "three_run_literal_source_output_tree_exact": len(literal_source_tree_hashes) == 1,
        "three_run_literal_renderer_output_tree_exact": len(literal_render_tree_hashes) == 1,
        "three_run_engine_operation_counters_exact": len(engine_counter_hashes) == 1,
        "three_run_worker_operation_counters_exact": len(worker_counter_hashes) == 1,
        "all_38_unique_member_fits_executed": unique_fit_ok,
        "all_32_robustness_rows_executed": robustness_ok,
        "raycast_tie_differential_passed": all(
            _metric_bool(row, "raycast_tie_differential_passed") for row in sources
        ),
        "kd_tie_differential_passed": all(
            _metric_bool(row, "kd_tie_differential_passed") for row in sources
        ),
        "scaling_t_n_le_2_5_t_half": scaling_ratio <= SCALING_LIMIT,
        "final_bundle_logical_lt_2gib": bundle_logical_bytes < BUNDLE_LIMIT_BYTES,
        "final_bundle_allocated_lt_2gib": bundle_allocated_bytes < BUNDLE_LIMIT_BYTES,
        "frozen_parent_v5_identical_fixture_differential": differential.get("status") == "PASS",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "scaling": {
            "full_median_wall_seconds": full_median,
            "half_wall_seconds": half_wall,
            "ratio": scaling_ratio,
            "maximum_inclusive": SCALING_LIMIT,
        },
        "bundle": {
            "denominator": "maximum deterministic source-plus-renderer output bytes for any one full production-scale run",
            "maximum_full_run_logical_bytes": bundle_logical_bytes,
            "maximum_full_run_allocated_bytes": bundle_allocated_bytes,
            "maximum_bytes_strict": BUNDLE_LIMIT_BYTES,
        },
    }


def run_reference_differential(
    args: argparse.Namespace, manifest: Path, root: Path
) -> dict[str, Any]:
    runs: dict[str, list[dict[str, Any]]] = {"FROZEN_PARENT": [], "V5": []}
    # Alternate engines to reduce monotonic thermal/load bias while preserving
    # three independent fresh processes for each implementation.
    for index in range(1, 4):
        for engine in ("FROZEN_PARENT", "V5"):
            run_root = root / engine.lower() / f"run_{index}"
            output_root = run_root / "output"
            command = (
                _reference_command(args, manifest, output_root)
                if engine == "FROZEN_PARENT"
                else _source_command(
                    args,
                    manifest,
                    REFERENCE_DIFFERENTIAL_ROWS,
                    output_root,
                    run_root / "measurement/ENGINE_MONITORING_METRICS.json",
                )
            )
            measurement = run_measured_command(
                command,
                output_root=run_root / "measurement",
                hard_timeout_seconds=315.0,
            )
            if measurement["returncode"] != 0 or measurement["timed_out"]:
                return {
                    "status": "FAIL",
                    "reason": f"{engine}_RUN_{index}_DID_NOT_COMPLETE",
                    "rows_per_member": REFERENCE_DIFFERENTIAL_ROWS,
                    "runs": runs,
                    "failed_measurement": measurement,
                }
            result = dict(_load_worker_result(output_root / "WORKER_RESULT.json"))
            runs[engine].append(
                {
                    "run_index": index,
                    "result": result,
                    "measurement": measurement,
                    "full_tree_hash_map": full_tree_hash_map(output_root),
                }
            )
    parent_median = float(
        np.median([row["measurement"]["wall_seconds"] for row in runs["FROZEN_PARENT"]])
    )
    v5_median = float(
        np.median([row["measurement"]["wall_seconds"] for row in runs["V5"]])
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
        canonical_sha256(row["full_tree_hash_map"]) for row in runs["FROZEN_PARENT"]
    }
    v5_trees = {canonical_sha256(row["full_tree_hash_map"]) for row in runs["V5"]}
    checks = {
        "three_fresh_frozen_parent_processes": len(
            {row["measurement"]["root_pid"] for row in runs["FROZEN_PARENT"]}
        ) == 3,
        "three_fresh_v5_processes": len(
            {row["measurement"]["root_pid"] for row in runs["V5"]}
        ) == 3,
        "all_processes_peak_rss_le_4gib": all(
            row["measurement"]["peak_process_tree_rss_bytes"] <= RSS_LIMIT_BYTES
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
        "rows_per_member": REFERENCE_DIFFERENTIAL_ROWS,
        "fixture_policy": "minimum_actionable_clean_source_interface_fixture_frozen_before_measurement",
        "runs": runs,
        "checks": checks,
        "frozen_parent_median_wall_seconds": parent_median,
        "v5_median_wall_seconds": v5_median,
        "median_speedup_ratio_parent_over_v5": parent_median / max(v5_median, 1e-12),
    }
def run_benchmark(args: argparse.Namespace, manifest: Path) -> Mapping[str, Any]:
    # The worker repeats this complete verification in every fresh child.
    lock_sha256 = sha256_path(args.qualification_lock.resolve())
    lock_payload = json.loads(args.qualification_lock.read_text(encoding="utf-8"))
    if lock_payload.get("schema") != "EXP_CAD_0032_SOURCE_FREE_QUALIFICATION_LOCK_V1":
        raise ValueError("unexpected EXP32 qualification lock schema")
    result_root = args.result_root.resolve()
    if result_root.exists() and any(result_root.iterdir()):
        raise ValueError("EXP32 result root must be absent or empty")
    result_root.mkdir(parents=True, exist_ok=True)
    full_runs: list[dict[str, Any]] = []
    for index in range(1, 4):
        run_root = result_root / "full" / f"run_{index}"
        source_root = run_root / "source"
        measurement = run_measured_command(
            _source_command(
                args,
                manifest,
                DEFAULT_ROWS,
                source_root,
                run_root / "source_measurement/ENGINE_MONITORING_METRICS.json",
            ),
            output_root=run_root / "source_measurement",
            hard_timeout_seconds=SOURCE_WALL_LIMIT_SECONDS + 15.0,
        )
        if measurement["returncode"] != 0 or measurement["timed_out"]:
            failure = {
                "schema": SCHEMA,
                "experiment_id": "EXP_CAD_0032",
                "status": "FAIL_STAGED_SOURCE_RUN_STOP",
                "failed_full_run": index,
                "source_measurement": measurement,
                "physical_scan_or_evaluator_input_used": False,
            }
            atomic_json(result_root / "EXP_CAD_0032_PERFORMANCE_RESULT.json", failure)
            return failure
        source = dict(_load_worker_result(source_root / "WORKER_RESULT.json"))
        source["scientific_tree_canonical_sha256"] = _scientific_tree_identity(source)
        source_tree = full_tree_hash_map(source_root)
        renderer_root = run_root / "renderer"
        renderer_measurement = run_measured_command(
            _render_command(args, source_root / "VISUAL_PAYLOAD.json", renderer_root),
            output_root=run_root / "renderer_measurement",
            hard_timeout_seconds=RENDER_WALL_LIMIT_SECONDS + 15.0,
        )
        if renderer_measurement["returncode"] != 0 or renderer_measurement["timed_out"]:
            failure = {
                "schema": SCHEMA,
                "experiment_id": "EXP_CAD_0032",
                "status": "FAIL_STAGED_RENDERER_RUN_STOP",
                "failed_full_run": index,
                "source": source,
                "source_measurement": measurement,
                "renderer_measurement": renderer_measurement,
                "physical_scan_or_evaluator_input_used": False,
            }
            atomic_json(result_root / "EXP_CAD_0032_PERFORMANCE_RESULT.json", failure)
            return failure
        renderer = dict(_load_worker_result(renderer_root / "WORKER_RESULT.json"))
        renderer_tree = full_tree_hash_map(renderer_root)
        full_runs.append(
            {
                "run_index": index,
                "source": source,
                "source_measurement": measurement,
                "source_full_tree_hash_map": source_tree,
                "renderer": renderer,
                "renderer_measurement": renderer_measurement,
                "renderer_full_tree_hash_map": renderer_tree,
            }
        )

    half_root = result_root / "half" / "run_1"
    half_measurement = run_measured_command(
        _source_command(
            args,
            manifest,
            HALF_ROWS,
            half_root / "source",
            half_root / "source_measurement/ENGINE_MONITORING_METRICS.json",
        ),
        output_root=half_root / "source_measurement",
        hard_timeout_seconds=SOURCE_WALL_LIMIT_SECONDS + 15.0,
    )
    if half_measurement["returncode"] != 0 or half_measurement["timed_out"]:
        failure = {
            "schema": SCHEMA,
            "experiment_id": "EXP_CAD_0032",
            "status": "FAIL_STAGED_HALF_SCALE_RUN_STOP",
            "full_runs": full_runs,
            "half_source_measurement": half_measurement,
            "physical_scan_or_evaluator_input_used": False,
        }
        atomic_json(result_root / "EXP_CAD_0032_PERFORMANCE_RESULT.json", failure)
        return failure
    half_source = dict(_load_worker_result(half_root / "source/WORKER_RESULT.json"))
    half_source["scientific_tree_canonical_sha256"] = _scientific_tree_identity(half_source)
    half_run = {
        "source": half_source,
        "source_measurement": half_measurement,
        "source_full_tree_hash_map": full_tree_hash_map(half_root / "source"),
    }

    differential = run_reference_differential(
        args, manifest, result_root / "reference_differential"
    )

    per_run_bundle_sizes = []
    for index in range(1, 4):
        run_root = result_root / "full" / f"run_{index}"
        source_logical, source_allocated = _directory_sizes(run_root / "source")
        render_logical, render_allocated = _directory_sizes(run_root / "renderer")
        per_run_bundle_sizes.append(
            {
                "run_index": index,
                "logical_bytes": source_logical + render_logical,
                "allocated_bytes": source_allocated + render_allocated,
            }
        )
    logical = max(row["logical_bytes"] for row in per_run_bundle_sizes)
    allocated = max(row["allocated_bytes"] for row in per_run_bundle_sizes)
    qualification_total_logical, qualification_total_allocated = _directory_sizes(result_root)
    gates = evaluate_gates(
        full_runs=full_runs,
        half_run=half_run,
        differential=differential,
        bundle_logical_bytes=logical,
        bundle_allocated_bytes=allocated,
    )
    result = {
        "schema": SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "architecture": "CROSSFIT_PAIRWISE_DEFORMATION_GRAPH_V5_EXECUTION_EQUIVALENT",
        "status": "PASS_PRODUCTION_SCALE_SOURCE_FREE" if gates["status"] == "PASS" else "FAIL_PRODUCTION_SCALE_SOURCE_FREE",
        "retained_rows_per_member": DEFAULT_ROWS,
        "cad_node_count": EXPECTED_NODE_COUNT,
        "cad_edge_count": EXPECTED_EDGE_COUNT,
        "benchmark_manifest_sha256": sha256_path(manifest),
        "qualification_lock_sha256": lock_sha256,
        "full_runs": full_runs,
        "half_run": half_run,
        "reference_differential": differential,
        "artifact_bundle_accounting": {
            "gate_denominator": "maximum deterministic source-plus-renderer output bytes for any one full production-scale run",
            "per_full_run": per_run_bundle_sizes,
            "qualification_evidence_total_logical_bytes_not_gate_denominator": qualification_total_logical,
            "qualification_evidence_total_allocated_bytes_not_gate_denominator": qualification_total_allocated,
        },
        "gates": gates,
        "physical_scan_or_evaluator_input_used": False,
        "generalization_evidence": False,
        "bend_line_f1_authorized": False,
        "production_promotion_authorized": False,
        "gpi_ceiling": "GPI-C_SOURCE_ONLY_EXECUTION_FEASIBILITY_NOT_F1_PROGRESS",
    }
    atomic_json(result_root / "EXP_CAD_0032_PERFORMANCE_RESULT.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope-root", required=True, type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--cad-nodes", required=True, type=Path)
    parser.add_argument("--cad-edges", required=True, type=Path)
    parser.add_argument("--edge-geodesic-mm", required=True, type=Path)
    parser.add_argument("--cad-mesh", required=True, type=Path)
    parser.add_argument("--v5-preregistration", type=Path)
    parser.add_argument("--parent-preregistration", type=Path)
    parser.add_argument("--cad-preflight", type=Path)
    parser.add_argument("--source-free-runner", type=Path)
    parser.add_argument("--frozen-parent-runner", type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--qualification-lock", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_envelope(
        output_root=args.envelope_root,
        cad_nodes=args.cad_nodes.resolve(),
        cad_edges=args.cad_edges.resolve(),
        edge_weights=args.edge_geodesic_mm.resolve(),
        cad_mesh=args.cad_mesh.resolve(),
    )
    if args.prepare_only:
        print(json.dumps({"status": "PREPARED", "manifest": str(manifest)}))
        return 0
    required = (
        "result_root",
        "v5_preregistration",
        "parent_preregistration",
        "cad_preflight",
        "source_free_runner",
        "frozen_parent_runner",
        "renderer",
        "qualification_lock",
    )
    missing = [key for key in required if getattr(args, key) is None]
    if missing:
        raise ValueError(f"benchmark execution arguments missing: {missing}")
    result = run_benchmark(args, manifest)
    print(json.dumps({"status": result["status"]}, indent=2))
    return 0 if result["status"] == "PASS_PRODUCTION_SCALE_SOURCE_FREE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
