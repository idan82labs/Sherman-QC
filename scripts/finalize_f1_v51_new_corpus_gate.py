#!/usr/bin/env python3
"""Combine immutable V51 execution, score, and all-case visual audit gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


TOKEN = "V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))
MAX_RSS = 16 * 1024**3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, expected_kind: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("token") != TOKEN or payload.get("schema_version") != 1:
        raise ValueError(f"token/schema mismatch: {path}")
    if payload.get("kind") != expected_kind:
        raise ValueError(f"kind mismatch: {path}")
    return payload


def _percentile95(values: list[float]) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(execution_path: Path, score_path: Path, audit_path: Path) -> dict[str, Any]:
    execution = _load(execution_path, "sherman_qc_f1_v51_new_untouched_corpus_blind_execution")
    score = _load(score_path, "sherman_qc_f1_v51_new_untouched_corpus_locked_score")
    audit = _load(audit_path, "sherman_qc_f1_v51_new_untouched_corpus_visual_audit")
    records = list(execution.get("records") or [])
    primary = [row for row in records if row.get("pass") == "primary"]
    replay = [row for row in records if row.get("pass") == "replay"]
    audits = list(audit.get("cases") or [])
    if [row.get("case_id") for row in audits] != list(CASE_IDS):
        raise ValueError("visual audit must contain O001..O030 in exact order")
    score_by_case = {str(row["case_id"]): row for row in score.get("cases") or []}
    if set(score_by_case) != set(CASE_IDS):
        raise ValueError("score case set is not exactly O001..O030")

    visual_ok = True
    marker_correct = 0
    visual_evidence = []
    for row in audits:
        case_id = str(row["case_id"])
        required = ("prediction_only_path", "prediction_only_sha256", "scored_overlay_path", "scored_overlay_sha256", "axis_stack_path", "axis_stack_sha256")
        artifact_checks = {}
        for prefix in ("prediction_only", "scored_overlay", "axis_stack"):
            path = Path(str(row[f"{prefix}_path"])).resolve()
            artifact_checks[prefix] = path.is_file() and _sha(path) == row[f"{prefix}_sha256"]
        required_fields = all(key in row for key in required) and bool(row.get("reviewer")) and row.get("ruling") in {"coherent", "geometric_failure", "render_failure"}
        visual_ok &= required_fields and all(artifact_checks.values())
        observed_count = int(row["render_admissible_marker_count"])
        expected_count = int(score_by_case[case_id]["gold_count"])
        marker_correct += int(observed_count == expected_count)
        visual_evidence.append({"case_id": case_id, "artifacts": artifact_checks, "required_fields": required_fields, "marker_count": observed_count, "gold_count": expected_count})

    all_rows = primary + replay
    job_pass = len(primary) == 30 and len(replay) == 30 and execution.get("primary_success_count") == 30 and execution.get("replay_success_count") == 30
    replay_pass = set(execution.get("semantic_replay_identity") or {}) == set(CASE_IDS) and all((execution.get("semantic_replay_identity") or {}).values())
    point_pass = len(all_rows) == 60 and all(int(row.get("point_budget") or 0) <= 200_000 and not row.get("forbidden_cache_keys") for row in all_rows)
    rss = max((int(row.get("peak_stage_rss_bytes") or 0) for row in all_rows), default=0)
    latency = _percentile95([float(row.get("elapsed_seconds") or 0.0) for row in primary])
    ratios = score["aggregate_ratios"]
    integers = score["aggregate_integers"]
    marker_accuracy = marker_correct / 30
    gates = {
        "exact_bend_count": int(integers["exact_case_count"]) >= 27,
        "line_precision": float(ratios["precision"]) >= 0.90,
        "line_recall": float(ratios["recall"]) >= 0.90,
        "minimum_family_recall": float(ratios["minimum_family_recall"]) >= 0.80,
        "render_admissible_marker_accuracy": marker_accuracy >= 0.99,
        "deterministic_replay": replay_pass,
        "all_jobs": job_pass,
        "point_and_cache_contract": point_pass,
        "peak_rss_strictly_below_16_gib": rss < MAX_RSS,
        "primary_p95_latency_at_most_300_seconds": latency <= 300.0,
        "all_case_visual_audit": visual_ok,
    }
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_new_untouched_corpus_terminal_gate",
        "token": TOKEN,
        "research_only": True,
        "inputs": {"execution_sha256": _sha(execution_path), "score_sha256": _sha(score_path), "visual_audit_sha256": _sha(audit_path)},
        "metrics": {**integers, **ratios, "marker_correct_cases": marker_correct, "marker_denominator_cases": 30, "render_admissible_marker_accuracy": marker_accuracy, "peak_stage_rss_bytes": rss, "primary_p95_latency_seconds": latency},
        "gates": gates,
        "visual_evidence": visual_evidence,
        "passed": all(gates.values()),
        "terminal_disposition": "RESEARCH_PROMOTION_RULING_REQUIRED" if all(gates.values()) else "RETIRE_V51_OR_INVALIDATE_PER_PROTOCOL",
        "runtime_main_production_closed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"terminal gate is immutable and already exists: {output}")
    report = finalize(args.execution.resolve(), args.score.resolve(), args.visual_audit.resolve())
    _atomic(output, report)
    print(json.dumps({"passed": report["passed"], "terminal_disposition": report["terminal_disposition"], "output": str(output)}, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
