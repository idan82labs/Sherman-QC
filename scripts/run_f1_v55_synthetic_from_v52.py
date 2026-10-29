#!/usr/bin/env python3
"""One-shot sealed V55 MSTLR primary/replay execution orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_mesh_section_turn_locus_reselection_v55 import (
    CONSTANTS_SHA256,
    MAXIMUM_PUBLIC_BYTES,
    PRECOMMIT_TOKEN,
    PROFILE,
    RSS_LIMIT_BYTES,
    TERMINAL_FAILURE_TOKEN,
    V55ScanAbstention,
    WALL_LIMIT_SECONDS,
    canonical_bytes,
    run_preflight_fixtures,
)


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))
MAXIMUM_COMPLETE_JSON_BYTES = 64 * 1024 * 1024


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload, newline=True))


def _terminal(output_root: Path, reason: str, **details: object) -> int:
    failure = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_terminal_execution_failure",
        "token": TOKEN,
        "terminal_failure_token": TERMINAL_FAILURE_TOKEN,
        "status": "TERMINAL_RETIRED",
        "reason": reason,
        "truth_opened": False,
        **details,
    }
    _write_new(output_root / "terminal_failure.json", failure)
    print(json.dumps(failure, sort_keys=True))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v52-execution-root", type=Path, required=True)
    parser.add_argument("--pool-execution-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("V55 output root is not empty")
    args.output_root.mkdir(parents=True, exist_ok=True)

    code_paths = [
        ROOT / "backend/f1_mesh_section_turn_locus_reselection_v55.py",
        ROOT / "scripts/run_f1_v55_mstlr_worker.py",
        ROOT / "scripts/run_f1_v55_synthetic_from_v52.py",
    ]
    audit_paths = [
        ROOT / "scripts/render_f1_v55_mstlr_source_only_audit.py",
        ROOT / "scripts/score_f1_v55_synthetic_stress.py",
        ROOT / "tests/test_f1_mesh_section_turn_locus_reselection_v55.py",
    ]
    frozen: dict[str, dict[str, dict[str, str]]] = {"primary": {}, "replay": {}}
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            v52_path = args.v52_execution_root / pass_name / case_id / "semantic_result.json"
            pool_path = args.pool_execution_root / pass_name / case_id / "pool" / f"{case_id}.json"
            scan_path = args.scan_root / f"{case_id}.ply"
            for path in (v52_path, pool_path, scan_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            frozen[pass_name][case_id] = {
                "v52": _sha(v52_path),
                "pool": _sha(pool_path),
                "source": _sha(scan_path),
            }
    for path in [*code_paths, *audit_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_preexecution_lock",
        "token": TOKEN,
        "profile": PROFILE,
        "precommit_token": PRECOMMIT_TOKEN,
        "status": "MECHANISM_LOCKED_PREFLIGHT_PENDING",
        "truth_opened": False,
        "code_sha256": {str(path): _sha(path) for path in code_paths},
        "audit_code_sha256": {str(path): _sha(path) for path in audit_paths},
        "constants_sha256": CONSTANTS_SHA256,
        "frozen_input_sha256": frozen,
        "primary_replay_order": [[pass_name, case_id] for pass_name in ("primary", "replay") for case_id in CASE_IDS],
        "maximum_public_json_bytes_per_scan": MAXIMUM_PUBLIC_BYTES,
        "maximum_complete_json_bytes": MAXIMUM_COMPLETE_JSON_BYTES,
        "maximum_wall_seconds_per_scan": WALL_LIMIT_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "compact_schema": "V55_MSTLR_FIXED_STATION_AGGREGATES_ONLY_V1",
    }
    lock_path = args.output_root / "preexecution_lock.json"
    _write_new(lock_path, lock)

    # Gate 0 occurs only after code/constants/input hashes have been sealed.
    try:
        preflight = run_preflight_fixtures()
    except V55ScanAbstention as error:
        return _terminal(args.output_root, error.reason, preflight_passed=False)
    if preflight.get("passed") is not True or preflight.get("constants_sha256") != CONSTANTS_SHA256:
        return _terminal(args.output_root, "PREFLIGHT_REPORT_INVALID", preflight_passed=False)
    _write_new(args.output_root / "source_free_preflight.json", preflight)

    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    worker = ROOT / "scripts/run_f1_v55_mstlr_worker.py"
    for pass_name in ("primary", "replay"):
        command = [
            sys.executable,
            str(worker),
            "--pass-name",
            pass_name,
            "--v52-root",
            str(args.v52_execution_root),
            "--pool-root",
            str(args.pool_execution_root),
            "--scan-root",
            str(args.scan_root),
            "--output-root",
            str(args.output_root / pass_name),
            "--preexecution-lock",
            str(lock_path),
        ]
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            return _terminal(
                args.output_root,
                "PRIMARY_WORKER_FAILURE" if pass_name == "primary" else "REPLAY_WORKER_FAILURE",
                failed_pass=pass_name,
                worker_returncode=completed.returncode,
            )

    identity: dict[str, bool] = {}
    for case_id in CASE_IDS:
        primary = args.output_root / "primary" / case_id / "semantic_result.json"
        replay = args.output_root / "replay" / case_id / "semantic_result.json"
        identity[case_id] = primary.read_bytes() == replay.read_bytes()
    if not all(identity.values()):
        return _terminal(
            args.output_root,
            "REPLAY_BYTE_MISMATCH",
            semantic_replay_identity=identity,
        )

    primary_manifest = json.loads((args.output_root / "primary" / "pass_manifest.json").read_text(encoding="utf-8"))
    replay_manifest = json.loads((args.output_root / "replay" / "pass_manifest.json").read_text(encoding="utf-8"))
    records = list(primary_manifest["records"]) + list(replay_manifest["records"])
    wall_times = sorted(float(row["wall_seconds"]) for row in records)
    p95 = wall_times[int(0.95 * (len(wall_times) - 1))]
    peak_rss = max(int(primary_manifest["peak_process_rss_bytes"]), int(replay_manifest["peak_process_rss_bytes"]))
    if p95 >= WALL_LIMIT_SECONDS or peak_rss >= RSS_LIMIT_BYTES:
        return _terminal(args.output_root, "RESOURCE_GATE", p95_case_wall_seconds=p95, peak_process_rss_bytes=peak_rss)
    json_paths = list(args.output_root.glob("**/*.json"))
    total_json_bytes = sum(path.stat().st_size for path in json_paths)
    if total_json_bytes > MAXIMUM_COMPLETE_JSON_BYTES:
        return _terminal(args.output_root, "COMPLETE_OUTPUT_SIZE_GATE", complete_json_bytes=total_json_bytes)
    execution = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_synthetic_execution",
        "token": TOKEN,
        "profile": PROFILE,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "status": "SEALED_PRIMARY_REPLAY",
        "truth_opened": False,
        "single_execution_consumed": True,
        "source_free_preflight_passed": True,
        "semantic_replay_identity": identity,
        "p95_case_wall_seconds": round(p95, 9),
        "maximum_case_wall_seconds": round(max(wall_times), 9),
        "peak_process_rss_bytes": peak_rss,
        "completed_jobs": len(records),
        "expected_jobs": 60,
        "job_success_fraction": len(records) / 60.0,
        "complete_json_bytes_before_manifest": total_json_bytes,
        "primary_records": primary_manifest["records"],
        "replay_records": replay_manifest["records"],
    }
    _write_new(args.output_root / "execution_manifest.json", execution)
    print(
        json.dumps(
            {
                "status": execution["status"],
                "replay_identity": sum(identity.values()),
                "p95_seconds": execution["p95_case_wall_seconds"],
                "peak_rss_bytes": peak_rss,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
