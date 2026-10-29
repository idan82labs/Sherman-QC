#!/usr/bin/env python3
"""Run one blind PLY-only F1 case repeatedly and verify result determinism."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _canonical_result_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return inference-semantic fields; exclude timing, cache-hit, and paths."""
    repair = dict(payload.get("repair_diagnostics") or {})
    return {
        "exact_bend_count": payload.get("exact_bend_count"),
        "bend_count_range": payload.get("bend_count_range"),
        "current_zero_shot_exact_count": payload.get("current_zero_shot_exact_count"),
        "current_zero_shot_count_range": payload.get("current_zero_shot_count_range"),
        "assignment": payload.get("assignment"),
        "owned_bend_regions": payload.get("owned_bend_regions"),
        "candidate_solution_counts": payload.get("candidate_solution_counts"),
        "candidate_solution_energies": payload.get("candidate_solution_energies"),
        "object_level_confidence_summary": payload.get("object_level_confidence_summary"),
        "solver_options": repair.get("solver_options"),
        "solver_iterations": repair.get("iterations"),
        "scan_family_route": repair.get("scan_family_route"),
        "owned_region_marker_admissibility": repair.get("owned_region_marker_admissibility"),
    }


def _result_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _canonical_result_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def run_audit(
    *,
    slice_json: Path,
    part_key: str,
    repeat_count: int,
    output_dir: Path,
    cache_dir: Path,
    timeout_sec: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = []
    runs = []
    for index in range(1, max(1, int(repeat_count)) + 1):
        run_dir = output_dir / f"run_{index:02d}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_patch_graph_f1_prototype.py"),
            "--slice-json", str(slice_json),
            "--output-dir", str(run_dir),
            "--cache-dir", str(cache_dir),
            "--route-mode", "apply",
            "--strict-blind-inference",
            "--no-render",
            "--part-key", str(part_key),
        ]
        started = perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=max(60, int(timeout_sec)),
            check=False,
        )
        wall_sec = perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "determinism run failed").strip())
        candidates = sorted(run_dir.glob(f"{part_key}/**/decomposition_result.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one decomposition result for {part_key}, found {len(candidates)}")
        result_path = candidates[0]
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        fingerprint = _result_fingerprint(payload)
        fingerprints.append(fingerprint)
        runs.append(
            {
                "run_index": index,
                "wall_time_sec": round(wall_sec, 3),
                "fingerprint": fingerprint,
                "result_path": str(result_path),
            }
        )
        print(f"[{index}/{repeat_count}] {fingerprint[:12]} wall={wall_sec:.2f}s", flush=True)

    counts = Counter(fingerprints)
    modal_count = max(counts.values()) if counts else 0
    summary = {
        "schema_version": 1,
        "part_key": part_key,
        "repeat_count": len(runs),
        "unique_fingerprint_count": len(counts),
        "deterministic_repeat_rate": round(modal_count / len(runs), 6) if runs else None,
        "all_results_identical": len(counts) == 1 and bool(runs),
        "p95_latency_sec": _percentile([item["wall_time_sec"] for item in runs], 0.95),
        "runs": runs,
    }
    (output_dir / "determinism_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-json", required=True, type=Path)
    parser.add_argument("--part-key", required=True)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args(argv)
    summary = run_audit(
        slice_json=args.slice_json.resolve(),
        part_key=str(args.part_key),
        repeat_count=args.repeat_count,
        output_dir=args.output_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({key: summary[key] for key in ("repeat_count", "unique_fingerprint_count", "deterministic_repeat_rate")}, indent=2))
    return 0 if summary["all_results_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
