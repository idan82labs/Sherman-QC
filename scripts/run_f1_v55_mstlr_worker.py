#!/usr/bin/env python3
"""Clean-process worker for one frozen V55 primary or replay pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_mesh_section_turn_locus_reselection_v55 import (
    MAXIMUM_PUBLIC_BYTES,
    RSS_LIMIT_BYTES,
    V55ScanAbstention,
    WALL_LIMIT_SECONDS,
    canonical_bytes,
    compose_v55,
)


CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload, newline=True))


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-name", choices=("primary", "replay"), required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preexecution-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(args.output_root)
    lock = json.loads(args.preexecution_lock.read_text(encoding="utf-8"))
    if lock.get("truth_opened") is not False or lock.get("status") != "MECHANISM_LOCKED_PREFLIGHT_PENDING":
        raise ValueError("invalid V55 preexecution lock")
    expected = lock["frozen_input_sha256"]
    records = []
    try:
        for case_id in CASE_IDS:
            v52_path = args.v52_root / args.pass_name / case_id / "semantic_result.json"
            pool_path = args.pool_root / args.pass_name / case_id / "pool" / f"{case_id}.json"
            scan_path = args.scan_root / f"{case_id}.ply"
            actual = {
                "v52": _sha(v52_path),
                "pool": _sha(pool_path),
                "source": _sha(scan_path),
            }
            expected_case = expected[args.pass_name][case_id]
            if actual != expected_case:
                raise V55ScanAbstention("ABSTAIN_FROZEN_INPUT_HASH_MISMATCH")
            v52 = json.loads(v52_path.read_text(encoding="utf-8"))
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            started = time.perf_counter()
            result = compose_v55(v52, pool, scan_path, integrity=actual)
            elapsed = time.perf_counter() - started
            if elapsed >= WALL_LIMIT_SECONDS:
                raise V55ScanAbstention("ABSTAIN_WALL_TIME")
            if _rss_bytes() >= RSS_LIMIT_BYTES:
                raise V55ScanAbstention("ABSTAIN_ACTUAL_RSS")
            output = args.output_root / case_id / "semantic_result.json"
            _write_new(output, result)
            if output.stat().st_size > MAXIMUM_PUBLIC_BYTES:
                raise V55ScanAbstention("ABSTAIN_OUTPUT_SCHEMA_OVERFLOW")
            records.append(
                {
                    "case_id": case_id,
                    "semantic_sha256": _sha(output),
                    "wall_seconds": round(elapsed, 9),
                    "peak_process_rss_bytes": _rss_bytes(),
                    "public_json_bytes": output.stat().st_size,
                    "prediction_count": len(result["predictions"]),
                    "pruned_v52_count": result["mesh_section_turn_locus_reselection_audit"]["pruned_v52_count"],
                    "accepted_addition_count": result["mesh_section_turn_locus_reselection_audit"]["accepted_addition_count"],
                }
            )
    except V55ScanAbstention as error:
        failure = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v55_worker_failure",
            "pass": args.pass_name,
            "reason": error.reason,
            "completed_records": records,
            "truth_opened": False,
            "peak_process_rss_bytes": _rss_bytes(),
        }
        _write_new(args.output_root / "worker_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v55_worker_pass",
        "pass": args.pass_name,
        "truth_opened": False,
        "status": "COMPLETE",
        "records": records,
        "peak_process_rss_bytes": _rss_bytes(),
    }
    _write_new(args.output_root / "pass_manifest.json", manifest)
    print(json.dumps({"pass": args.pass_name, "completed": len(records), "peak_rss": manifest["peak_process_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
