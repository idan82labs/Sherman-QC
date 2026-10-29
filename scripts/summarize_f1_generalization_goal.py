#!/usr/bin/env python3
"""Summarize the active Sherman QC F1 generalization target from one cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/research/f1-gold27-20260803"
DEFAULT_SCORE = BASE / "structured_union_exact_v1/gold27_score.json"
DEFAULT_OPERATIONS = BASE / "structured_union_exact_v1/operational_summary.json"
DEFAULT_VISUAL = BASE / "structured_union_exact_v1/visual_review/review.json"
DEFAULT_SPLIT_AUDIT = BASE / "freeze/grouped_five_fold_split_integrity.json"
DEFAULT_POOL = BASE / "candidate_pool_bounded_v2/manifest.json"
DEFAULT_OUTPUT = BASE / "structured_union_exact_v1/generalization_goal_readiness.json"

THRESHOLDS = {
    "minimum_scans": 30,
    "minimum_part_families": 15,
    "minimum_exact_count_accuracy": 0.90,
    "minimum_line_precision": 0.90,
    "minimum_line_recall": 0.90,
    "minimum_render_admissible_marker_accuracy": 0.99,
    "minimum_deterministic_repeat_rate": 1.0,
    "maximum_peak_rss_bytes": 16 * 1024**3,
    "maximum_p95_latency_sec": 300.0,
    "minimum_job_success_rate": 0.995,
    "minimum_family_recall": 0.80,
    "maximum_working_points": 200_000,
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(
    gate_id: str, observed: Any, required: Any, passed: bool, evidence: str
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
        "evidence": evidence,
    }


def evaluate_generalization_goal(
    *,
    score: Mapping[str, Any],
    operations: Mapping[str, Any],
    visual_review: list[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**THRESHOLDS, **dict(thresholds or {})}
    metrics = dict(score.get("metrics") or {})
    family = dict(score.get("part_family_metrics") or {})
    scan_count = int(metrics.get("scan_count") or 0)
    exact_accuracy = float(metrics.get("exact_scans") or 0) / max(scan_count, 1)
    visual_tp = sum(int(row.get("true_positive") or 0) for row in visual_review)
    visual_fp = sum(int(row.get("false_positive") or 0) for row in visual_review)
    visual_marker_accuracy = visual_tp / max(visual_tp + visual_fp, 1)
    visual_case_count = len(visual_review)
    family_count = int(family.get("family_count") or 0)
    minimum_family_recall = float(family.get("minimum_family_recall") or 0.0)
    peak_rss = int(operations.get("maximum_solver_child_peak_rss_bytes") or 0)
    working_budget = int(
        pool_manifest.get("working_point_budget")
        or operations.get("global_working_point_budget")
        or 0
    )
    source_points_absent = pool_manifest.get("source_point_serialization") is False

    gates = [
        _gate("corpus.scan_count", scan_count, cfg["minimum_scans"], scan_count >= int(cfg["minimum_scans"]), "gold27_score"),
        _gate("corpus.part_family_count", family_count, cfg["minimum_part_families"], family_count >= int(cfg["minimum_part_families"]), "gold27_score.part_family_metrics"),
        _gate("semantic.exact_count_accuracy", round(exact_accuracy, 6), cfg["minimum_exact_count_accuracy"], exact_accuracy >= float(cfg["minimum_exact_count_accuracy"]), "gold27_score.metrics"),
        _gate("semantic.line_precision", metrics.get("precision"), cfg["minimum_line_precision"], float(metrics.get("precision") or 0.0) >= float(cfg["minimum_line_precision"]), "gold27_score.metrics"),
        _gate("semantic.line_recall", metrics.get("recall"), cfg["minimum_line_recall"], float(metrics.get("recall") or 0.0) >= float(cfg["minimum_line_recall"]), "gold27_score.metrics"),
        _gate(
            "visual.render_admissible_marker_accuracy",
            {"accuracy": round(visual_marker_accuracy, 6), "reviewed_scans": visual_case_count},
            {"accuracy": cfg["minimum_render_admissible_marker_accuracy"], "reviewed_scans": scan_count},
            visual_case_count == scan_count
            and visual_marker_accuracy >= float(cfg["minimum_render_admissible_marker_accuracy"]),
            "all-case six-view review line matches; semantic marker precision proxy",
        ),
        _gate("operations.deterministic_repeat_rate", operations.get("deterministic_repeat_rate"), cfg["minimum_deterministic_repeat_rate"], float(operations.get("deterministic_repeat_rate") or 0.0) >= float(cfg["minimum_deterministic_repeat_rate"]), "operational_summary"),
        _gate("operations.peak_rss_bytes", peak_rss, f"< {cfg['maximum_peak_rss_bytes']}", 0 < peak_rss < int(cfg["maximum_peak_rss_bytes"]), "operational_summary maximum child solver RSS"),
        _gate("operations.p95_latency_sec", operations.get("p95_latency_sec"), f"<= {cfg['maximum_p95_latency_sec']}", 0.0 < float(operations.get("p95_latency_sec") or 0.0) <= float(cfg["maximum_p95_latency_sec"]), "operational_summary"),
        _gate("operations.job_success_rate", operations.get("job_success_rate"), cfg["minimum_job_success_rate"], float(operations.get("job_success_rate") or 0.0) >= float(cfg["minimum_job_success_rate"]), "operational_summary"),
        _gate("generalization.minimum_family_recall", minimum_family_recall, cfg["minimum_family_recall"], minimum_family_recall >= float(cfg["minimum_family_recall"]), "gold27_score.part_family_metrics"),
        _gate("integrity.family_held_out_split", split_audit.get("passed"), True, split_audit.get("passed") is True and split_audit.get("inference_metadata_exposure") is False, "grouped split integrity audit"),
        _gate("integrity.working_point_budget", working_budget, f"<= {cfg['maximum_working_points']}", 0 < working_budget <= int(cfg["maximum_working_points"]), "bounded candidate-pool manifest"),
        _gate("integrity.no_source_point_serialization", pool_manifest.get("source_point_serialization"), False, source_points_absent, "bounded candidate-pool manifest"),
    ]
    failed = [row["gate_id"] for row in gates if not row["passed"]]
    return {
        "schema_version": 1,
        "kind": "sherman_qc_f1_generalization_goal_readiness",
        "research_only": True,
        "runtime_promotion_authorized": False,
        "corpus_freeze_id": score.get("corpus_freeze_id"),
        "thresholds": cfg,
        "passed_gate_count": len(gates) - len(failed),
        "failed_gate_count": len(failed),
        "goal_achieved": not failed,
        "failed_gates": failed,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    parser.add_argument("--operations", type=Path, default=DEFAULT_OPERATIONS)
    parser.add_argument("--visual-review", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--pool-manifest", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_generalization_goal(
        score=_read(args.score.resolve()),
        operations=_read(args.operations.resolve()),
        visual_review=_read(args.visual_review.resolve()),
        split_audit=_read(args.split_audit.resolve()),
        pool_manifest=_read(args.pool_manifest.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["goal_achieved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
