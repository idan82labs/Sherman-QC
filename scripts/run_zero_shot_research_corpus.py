#!/usr/bin/env python3
"""Run the frozen scan-only F1 corpus, one opaque isolated process per scan."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "output/research/f1-corpus-evidence-20260729-v1/freeze/inference_manifest.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-corpus-evidence-20260729-v1/inference"
RSS_PATTERN = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)


def _run_case(
    case: dict,
    *,
    index: int,
    total: int,
    output_dir: Path,
    timeout_sec: int,
    enable_interior: bool,
    enable_lattice: bool,
) -> tuple[dict, dict]:
    token = str(case["case_token"])
    case_dir = output_dir / token
    case_dir.mkdir(parents=True, exist_ok=True)
    alias_path = case_dir / str(case["opaque_input_name"])
    source_path = Path(str(case["source_scan_path"])).resolve()
    if alias_path.is_symlink() and alias_path.resolve() != source_path:
        raise RuntimeError(f"opaque alias collision: {alias_path}")
    if not alias_path.exists():
        os.symlink(source_path, alias_path)
    command = [
        "/usr/bin/time", "-l", sys.executable,
        str(ROOT / "scripts/render_zero_shot_bend_overlay.py"),
        "--scan-path", str(alias_path),
        "--output-dir", str(case_dir / "result"),
    ]
    if enable_interior:
        command.append("--enable-interior-formed-transition-completion")
    if enable_lattice:
        command.append("--enable-prismatic-plane-lattice-completion")
    print(f"[{index}/{total}] START {token}", flush=True)
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=max(60, timeout_sec),
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    wall = perf_counter() - started
    rss_match = RSS_PATTERN.search(completed.stderr or "")
    rss = int(rss_match.group(1)) if rss_match else None
    operation = {
        "case_token": token,
        "returncode": completed.returncode,
        "wall_time_sec": round(wall, 3),
        "maximum_resident_set_bytes": rss,
        "stderr_tail": (completed.stderr or "")[-2000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"{token} inference failed: {operation['stderr_tail']}")
    result = {
        "zero_shot_result": str(case_dir / "result/zero_shot_result.json"),
        "overlay_manifest": str(case_dir / "result/zero_shot_bend_overlay_manifest.json"),
        "overview_render": str(case_dir / "result/zero_shot_bend_overlay_overview.png"),
    }
    print(
        f"[{index}/{total}] DONE {token} wall={wall:.1f}s "
        f"rss_gib={(rss or 0)/(1024**3):.2f}",
        flush=True,
    )
    return operation, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--enable-interior-formed-transition-completion", action="store_true")
    parser.add_argument("--enable-prismatic-plane-lattice-completion", action="store_true")
    parser.add_argument(
        "--case-token",
        action="append",
        default=[],
        help="Run only the requested opaque case token; may be repeated",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(manifest.get("cases") or [])
    if args.case_token:
        requested = set(args.case_token)
        available = {str(case["case_token"]) for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested case tokens are absent from manifest: {missing}")
        cases = [case for case in cases if str(case["case_token"]) in requested]
    workers = max(1, min(int(args.max_workers), len(cases)))
    jobs = [
        {
            "case": case,
            "index": index,
            "total": len(cases),
            "output_dir": output_dir,
            "timeout_sec": args.timeout_sec,
            "enable_interior": args.enable_interior_formed_transition_completion,
            "enable_lattice": args.enable_prismatic_plane_lattice_completion,
        }
        for index, case in enumerate(cases, start=1)
    ]
    if workers == 1:
        completed_jobs = [_run_case(**job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            completed_jobs = list(executor.map(lambda job: _run_case(**job), jobs))
    operations = [operation for operation, _result in completed_jobs]
    result_map = {
        str(case["case_token"]): result
        for case, (_operation, result) in zip(cases, completed_jobs)
    }

    summary = {
        "schema_version": 1,
        "corpus_freeze_id": manifest["corpus_freeze_id"],
        "inference_contract": manifest["protocol"],
        "experiment_flags": [
            name
            for name, enabled in (
                ("interior_formed_transition_completion", args.enable_interior_formed_transition_completion),
                ("prismatic_plane_lattice_completion", args.enable_prismatic_plane_lattice_completion),
            )
            if enabled
        ],
        "execution_model": (
            "sequential_one_opaque_process_per_scan"
            if workers == 1
            else f"bounded_parallel_{workers}_opaque_processes"
        ),
        "operations": operations,
        "result_map": result_map,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"case_count": len(operations), "summary": str(output_dir / 'inference_summary.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
