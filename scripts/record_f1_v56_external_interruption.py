#!/usr/bin/env python3
"""Seal a non-resumable V56 external-interruption failure artifact.

This reporter never executes the detector and never opens truth.  It records
only already-written partial outputs after the one-shot process was interrupted
by its operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
TERMINAL_TOKEN = "V56_CTMLR_TERMINAL_RETIRE"


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_view(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload, allow_nan=False))
    audit = result["canonical_turn_median_line_reconstruction_audit"]
    audit.pop("wall_ms", None)
    audit.pop("peak_rss_bytes", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.execution_root / "execution_manifest.json").exists():
        raise ValueError("complete V56 execution manifest exists; interruption reporter is invalid")
    expected_primary = [f"S{index:03d}" for index in range(1, 31)]
    expected_replay = [f"S{index:03d}" for index in range(1, 9)]
    records: list[dict[str, Any]] = []
    for pass_name, case_ids in (("primary", expected_primary), ("replay", expected_replay)):
        for case_id in case_ids:
            result_path = args.execution_root / pass_name / case_id / "semantic_result.json"
            cache_path = args.execution_root / pass_name / case_id / "compact_cache.json"
            if not result_path.is_file() or not cache_path.is_file():
                raise FileNotFoundError(f"expected completed partial record missing: {pass_name}/{case_id}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            records.append(
                {
                    "pass": pass_name,
                    "case_id": case_id,
                    "semantic_result_sha256": _sha(result_path),
                    "deterministic_semantic_sha256": hashlib.sha256(_canonical(_semantic_view(result))).hexdigest(),
                    "prediction_sha256": hashlib.sha256(_canonical(result["predictions"])).hexdigest(),
                    "compact_cache_sha256": _sha(cache_path),
                    "wall_seconds": float(cache["wall_ms"]) / 1000.0,
                    "peak_rss_bytes": int(cache["peak_rss_bytes"]),
                    "output_count": int(cache["output_count"]),
                }
            )
    actual_primary = sorted(path.parent.name for path in (args.execution_root / "primary").glob("S*/semantic_result.json"))
    actual_replay = sorted(path.parent.name for path in (args.execution_root / "replay").glob("S*/semantic_result.json"))
    if actual_primary != expected_primary or actual_replay != expected_replay:
        raise ValueError(f"unexpected partial completion sets: primary={actual_primary}, replay={actual_replay}")
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    partial_identity = {
        case_id: bool(
            primary[case_id]["deterministic_semantic_sha256"] == replay[case_id]["deterministic_semantic_sha256"]
            and primary[case_id]["prediction_sha256"] == replay[case_id]["prediction_sha256"]
        )
        for case_id in expected_replay
    }
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_terminal_execution_failure",
        "token": TOKEN,
        "terminal_token": TERMINAL_TOKEN,
        "status": "TERMINAL_RETIRED_EXTERNAL_INTERRUPTION",
        "reason": "OPERATOR_SENT_KEYBOARD_INTERRUPT_DURING_REPLAY_S009_BEFORE_ANY_PROVEN_LIMIT_BREACH",
        "operator_assessment_error": True,
        "hard_latency_limit_breach_proven": False,
        "interrupted_pass": "replay",
        "interrupted_case_id": "S009",
        "interruption_stack_leaf": "build_station_contours:np.linalg.norm",
        "resume_or_rerun_authorized": False,
        "truth_opened": False,
        "score_consumed": False,
        "visual_gate_consumed": False,
        "completed_record_count": len(records),
        "completed_primary_count": len(primary),
        "completed_replay_count": len(replay),
        "partial_semantic_replay_identity": partial_identity,
        "all_completed_replays_identical": all(partial_identity.values()),
        "maximum_completed_case_wall_seconds": max(row["wall_seconds"] for row in records),
        "maximum_completed_peak_rss_bytes": max(row["peak_rss_bytes"] for row in records),
        "records": records,
    }
    args.output.write_bytes(_canonical(payload))
    print(json.dumps({key: payload[key] for key in ("status", "completed_record_count", "completed_primary_count", "completed_replay_count", "all_completed_replays_identical", "maximum_completed_case_wall_seconds", "maximum_completed_peak_rss_bytes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
