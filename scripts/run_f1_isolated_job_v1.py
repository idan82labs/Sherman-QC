#!/usr/bin/env python3
"""CLI for the production-shaped isolated F1 job runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_isolated_job_v1 import JobLimits, run_isolated_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--lane", choices=("SCAN_ONLY", "CAD_ASSISTED"), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--input-hashes", type=Path, required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--ontology-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--rss-limit-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--log-limit-bytes", type=int, default=16 * 1024**2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    hashes = json.loads(args.input_hashes.read_text())
    code, receipt = run_isolated_job(
        job_id=args.job_id, lane=args.lane, command=command,
        receipt_path=args.receipt.resolve(), log_directory=args.log_directory.resolve(),
        input_hashes=hashes, detector_sha256=args.detector_sha256,
        config_sha256=args.config_sha256, ontology_sha256=args.ontology_sha256,
        limits=JobLimits(timeout_seconds=args.timeout_seconds, rss_limit_bytes=args.rss_limit_bytes, log_limit_bytes=args.log_limit_bytes),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
