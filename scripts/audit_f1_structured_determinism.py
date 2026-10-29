#!/usr/bin/env python3
"""Prove byte-stable replay for one exact typed-union graph."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--solver-python", type=Path, default=ROOT / ".venv-f1-structured/bin/python")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fingerprints = []
    for index in range(1, int(args.repeat_count) + 1):
        output = output_dir / f"solution_{index:02d}.json"
        started = perf_counter()
        completed = subprocess.run(
            [
                str(args.solver_python.absolute()),
                str(ROOT / "scripts/solve_f1_structured_union_cpsat.py"),
                "--graph",
                str(args.graph.resolve()),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(60, int(args.timeout_sec)),
        )
        wall = perf_counter() - started
        if completed.returncode:
            raise RuntimeError(completed.stderr[-2000:] or completed.stdout[-2000:])
        payload = json.loads(output.read_text(encoding="utf-8"))
        if str((payload.get("solver") or {}).get("status")) != "OPTIMAL":
            raise RuntimeError("determinism audit requires OPTIMAL on every replay")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        fingerprints.append(fingerprint)
        rows.append(
            {
                "run_index": index,
                "wall_time_sec": round(wall, 6),
                "fingerprint": fingerprint,
                "solution_path": str(output),
            }
        )
        print(f"[{index}/{args.repeat_count}] {fingerprint[:12]} wall={wall:.3f}s", flush=True)
    counts = Counter(fingerprints)
    modal = max(counts.values()) if counts else 0
    summary = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_structured_exact_determinism",
        "graph": str(args.graph.resolve()),
        "repeat_count": len(rows),
        "unique_fingerprint_count": len(counts),
        "deterministic_repeat_rate": modal / len(rows) if rows else None,
        "all_results_identical": len(counts) == 1 and bool(rows),
        "p95_latency_sec": _percentile([float(row["wall_time_sec"]) for row in rows], 0.95),
        "runs": rows,
    }
    (output_dir / "determinism_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "repeat_count": summary["repeat_count"],
                "unique_fingerprint_count": summary["unique_fingerprint_count"],
                "deterministic_repeat_rate": summary["deterministic_repeat_rate"],
                "p95_latency_sec": summary["p95_latency_sec"],
            },
            indent=2,
        )
    )
    return 0 if summary["all_results_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

