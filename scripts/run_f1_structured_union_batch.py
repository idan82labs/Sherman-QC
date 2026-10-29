#!/usr/bin/env python3
"""Run the research-only exact typed-union lane over a bounded pool manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "output/research/f1-gold27-20260803/candidate_pool_bounded_v2/manifest.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/structured_union_exact_v1"
DEFAULT_APP_PYTHON = ROOT / ".venv/bin/python"
DEFAULT_SOLVER_PYTHON = ROOT / ".venv-f1-structured/bin/python"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_case(task: dict[str, Any]) -> dict[str, Any]:
    token = str(task["case_token"])
    output_dir = Path(str(task["output_dir"]))
    command = [
        str(task["app_python"]),
        str(ROOT / "scripts/run_f1_structured_union_e2e.py"),
        "--candidate-pool",
        str(task["candidate_pool"]),
        "--output-dir",
        str(output_dir),
        "--solver-python",
        str(task["solver_python"]),
    ]
    if bool(task.get("disable_birth")):
        command.append("--disable-birth")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "batch.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        return {
            "case_token": token,
            "success": False,
            "returncode": completed.returncode,
            "error_tail": completed.stderr[-2000:],
        }
    summary = _read(output_dir / "inference_summary.json")
    return {
        "case_token": token,
        "success": summary.get("solver_status") == "OPTIMAL",
        "solver_status": summary.get("solver_status"),
        "counted_bend_component_count": int(
            summary.get("counted_bend_component_count") or 0
        ),
        "unsupported_selected_path_count": int(
            summary.get("unsupported_selected_path_count") or 0
        ),
        "birth_column_count": int(summary.get("birth_column_count") or 0),
        "solver_child_peak_rss_bytes": int(
            summary.get("solver_child_peak_rss_bytes") or 0
        ),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--app-python", type=Path, default=DEFAULT_APP_PYTHON)
    parser.add_argument("--solver-python", type=Path, default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--measured-peak-rss-bytes", type=int)
    parser.add_argument("--host-memory-budget-gib", type=float, default=96.0)
    parser.add_argument("--disable-birth", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    workers = int(args.workers)
    if workers < 1 or workers > 18:
        raise ValueError("workers must be between 1 and the 18-core host limit")
    if workers > 1:
        measured = int(args.measured_peak_rss_bytes or 0)
        if measured <= 0:
            raise ValueError("parallel exact solve requires measured-peak-rss-bytes")
        memory_budget = int(float(args.host_memory_budget_gib) * 1024**3)
        memory_safe = max(1, memory_budget // measured)
        if workers > memory_safe:
            raise ValueError(
                f"requested {workers} workers exceeds measured memory-safe cap {memory_safe}"
            )

    manifest = _read(args.pool_manifest.resolve())
    output_root = args.output_root.resolve()
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tasks = []
    for case in manifest.get("cases") or ():
        token = str(case["case_token"])
        output_dir = cases_root / token
        summary_path = output_dir / "inference_summary.json"
        if args.skip_existing and summary_path.is_file():
            summary = _read(summary_path)
            if (
                summary.get("solver_status") != "OPTIMAL"
                or int(summary.get("unsupported_selected_path_count") or 0) != 0
                or bool(summary.get("runtime_promotion_authorized"))
            ):
                raise ValueError(f"{token}: existing structured result is invalid")
            rows.append(
                {
                    "case_token": token,
                    "success": True,
                    "solver_status": "OPTIMAL",
                    "counted_bend_component_count": int(
                        summary.get("counted_bend_component_count") or 0
                    ),
                    "unsupported_selected_path_count": 0,
                    "birth_column_count": int(summary.get("birth_column_count") or 0),
                    "solver_child_peak_rss_bytes": int(
                        summary.get("solver_child_peak_rss_bytes") or 0
                    ),
                    "output_dir": str(output_dir),
                    "reused": True,
                }
            )
            continue
        tasks.append(
            {
                "case_token": token,
                "candidate_pool": str(Path(str(case["path"])).resolve()),
                "output_dir": str(output_dir),
                "app_python": str(args.app_python.absolute()),
                "solver_python": str(args.solver_python.absolute()),
                "disable_birth": bool(args.disable_birth),
            }
        )

    print(
        f"solving {len(tasks)} cases with workers={workers}; reusing {len(rows)}",
        flush=True,
    )
    if workers == 1:
        rows.extend(_run_case(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows.extend(executor.map(_run_case, tasks))
    rows.sort(key=lambda row: str(row["case_token"]))
    failed = [row for row in rows if not bool(row.get("success"))]
    summary = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_typed_union_batch_summary",
        "research_only": True,
        "runtime_promotion_authorized": False,
        "pool_manifest": str(args.pool_manifest.resolve()),
        "case_count": len(rows),
        "success_count": len(rows) - len(failed),
        "failure_count": len(failed),
        "workers": workers,
        "birth_enabled": not bool(args.disable_birth),
        "max_solver_child_peak_rss_bytes": max(
            (int(row.get("solver_child_peak_rss_bytes") or 0) for row in rows),
            default=0,
        ),
        "cases": rows,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "success_count": len(rows) - len(failed),
                "failure_count": len(failed),
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

