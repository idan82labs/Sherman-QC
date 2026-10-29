#!/usr/bin/env python3
"""Evaluate whether the bend-inspection system is eligible for runtime promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_THRESHOLDS = {
    "minimum_truth_scans": 150,
    "minimum_truth_parts": 75,
    "minimum_locked_holdout_parts": 30,
    "minimum_family_scans": 30,
    "minimum_ply_exact_accuracy": 0.90,
    "minimum_cad_ply_exact_accuracy": 0.95,
    "minimum_cad_uplift": 0.05,
    "minimum_range_containment": 0.98,
    "minimum_visual_instance_accuracy": 0.99,
    "minimum_instance_truth_coverage": 0.95,
    "minimum_deterministic_repeat_rate": 1.0,
    "maximum_p95_latency_sec": 300.0,
    "minimum_job_success_rate": 0.995,
}


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if payload is None:
        return {}
    value = payload.get("aggregate")
    return value if isinstance(value, Mapping) else payload


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(
    checks: list[Dict[str, Any]],
    gate_id: str,
    *,
    observed: Any,
    required: Any,
    passed: bool,
    blocker: str,
) -> None:
    checks.append(
        {
            "gate_id": gate_id,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
            "blocker": None if passed else blocker,
        }
    )


def evaluate_promotion_gates(
    *,
    truth_registry: Mapping[str, Any],
    ply_summary: Optional[Mapping[str, Any]],
    cad_ply_summary: Optional[Mapping[str, Any]],
    operational_summary: Optional[Mapping[str, Any]],
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    checks: list[Dict[str, Any]] = []
    records = list(truth_registry.get("records") or ())
    eligible = [record for record in records if bool(record.get("release_metric_eligible"))]
    eligible_parts = {str(record.get("part_key") or "") for record in eligible if str(record.get("part_key") or "")}
    holdout_parts = {
        str(record.get("part_key") or "")
        for record in eligible
        if str(record.get("evaluation_partition") or "") == "locked_holdout"
    }
    by_family: Dict[str, int] = {}
    for record in eligible:
        family = str(record.get("part_family") or "unknown")
        by_family[family] = by_family.get(family, 0) + 1
    family_min = min(by_family.values()) if by_family else 0
    instance_truth_count = sum(bool(record.get("instance_geometry")) for record in eligible)
    instance_truth_coverage = instance_truth_count / len(eligible) if eligible else 0.0

    _check(
        checks,
        "truth.release_scan_count",
        observed=len(eligible),
        required=cfg["minimum_truth_scans"],
        passed=len(eligible) >= int(cfg["minimum_truth_scans"]),
        blocker="authoritative typed truth set is too small",
    )
    _check(
        checks,
        "truth.release_part_count",
        observed=len(eligible_parts),
        required=cfg["minimum_truth_parts"],
        passed=len(eligible_parts) >= int(cfg["minimum_truth_parts"]),
        blocker="too few independently reviewed parts",
    )
    _check(
        checks,
        "truth.locked_holdout_part_count",
        observed=len(holdout_parts),
        required=cfg["minimum_locked_holdout_parts"],
        passed=len(holdout_parts) >= int(cfg["minimum_locked_holdout_parts"]),
        blocker="no sufficiently large locked holdout exists",
    )
    _check(
        checks,
        "truth.minimum_family_coverage",
        observed={"minimum": family_min, "by_family": dict(sorted(by_family.items()))},
        required=cfg["minimum_family_scans"],
        passed=bool(by_family) and family_min >= int(cfg["minimum_family_scans"]),
        blocker="one or more scan families lack enough adjudicated examples",
    )
    _check(
        checks,
        "truth.instance_geometry_coverage",
        observed=round(instance_truth_coverage, 4),
        required=cfg["minimum_instance_truth_coverage"],
        passed=instance_truth_coverage >= float(cfg["minimum_instance_truth_coverage"]),
        blocker="count labels do not provide enough bend-instance localization truth",
    )

    ply = _aggregate(ply_summary)
    ply_typed = (ply_summary or {}).get("typed_release_metrics")
    ply_typed = ply_typed if isinstance(ply_typed, Mapping) else {}
    ply_exact = _number(ply_typed.get("exact_count_accuracy", ply.get("candidate_exact_accuracy")))
    ply_evaluated = int(ply_typed.get("scans_processed", ply.get("exact_count_evaluated")) or 0)
    ply_range = _number(ply_typed.get("range_containment_rate", ply.get("candidate_range_contains_truth_rate")))
    visual_accuracy = _number(
        ply_typed.get("render_admissible_exact_accuracy", ply.get("render_admissible_exact_accuracy"))
    )
    visual_evaluated = int(
        ply_typed.get("scans_processed", ply.get("render_admissible_exact_count_evaluated")) or 0
    )
    _check(
        checks,
        "ply_only.evaluated_scan_count",
        observed=ply_evaluated,
        required=cfg["minimum_truth_scans"],
        passed=ply_evaluated >= int(cfg["minimum_truth_scans"]),
        blocker="PLY-only benchmark is undersized",
    )
    _check(
        checks,
        "ply_only.exact_accuracy",
        observed=ply_exact,
        required=cfg["minimum_ply_exact_accuracy"],
        passed=ply_exact is not None and ply_exact >= float(cfg["minimum_ply_exact_accuracy"]),
        blocker="blind PLY-only exact-count accuracy is below threshold",
    )
    _check(
        checks,
        "ply_only.range_containment",
        observed=ply_range,
        required=cfg["minimum_range_containment"],
        passed=ply_range is not None and ply_range >= float(cfg["minimum_range_containment"]),
        blocker="PLY-only reported ranges exclude truth too often",
    )
    _check(
        checks,
        "ply_only.visual_instance_accuracy",
        observed={"accuracy": visual_accuracy, "evaluated": visual_evaluated},
        required={"accuracy": cfg["minimum_visual_instance_accuracy"], "evaluated": cfg["minimum_truth_scans"]},
        passed=(
            visual_accuracy is not None
            and visual_accuracy >= float(cfg["minimum_visual_instance_accuracy"])
            and visual_evaluated >= int(cfg["minimum_truth_scans"])
        ),
        blocker="render-admissible bend ownership is not sufficiently accurate or broadly evaluated",
    )

    cad = _aggregate(cad_ply_summary)
    cad_typed = (cad_ply_summary or {}).get("typed_release_metrics")
    cad_typed = cad_typed if isinstance(cad_typed, Mapping) else {}
    cad_mode = None if cad_ply_summary is None else cad_ply_summary.get("inference_mode")
    cad_contract_rate = _number(cad_typed.get("blind_inference_contract_rate")) if isinstance(cad_typed, Mapping) else None
    cad_exact = _number(cad_typed.get("exact_count_accuracy", cad.get("full_scan_exact_completion_rate")))
    cad_evaluated = int(cad_typed.get("scans_processed", cad.get("full_scans")) or 0)
    _check(
        checks,
        "cad_ply.blind_inference_contract",
        observed={"inference_mode": cad_mode, "explicit_contract_rate": cad_contract_rate},
        required={"inference_mode": "blind_cad_ply", "explicit_contract_rate": 1.0},
        passed=cad_mode == "blind_cad_ply" and cad_contract_rate == 1.0,
        blocker="CAD+PLY result is missing or used operator/ground-truth hints during inference",
    )
    _check(
        checks,
        "cad_ply.evaluated_scan_count",
        observed=cad_evaluated,
        required=cfg["minimum_truth_scans"],
        passed=cad_evaluated >= int(cfg["minimum_truth_scans"]),
        blocker="blind CAD+PLY benchmark is undersized",
    )
    _check(
        checks,
        "cad_ply.exact_accuracy",
        observed=cad_exact,
        required=cfg["minimum_cad_ply_exact_accuracy"],
        passed=cad_exact is not None and cad_exact >= float(cfg["minimum_cad_ply_exact_accuracy"]),
        blocker="blind CAD+PLY exact-count accuracy is below threshold",
    )
    uplift = None if cad_exact is None or ply_exact is None else cad_exact - ply_exact
    uplift_evidence_sized = (
        ply_evaluated >= int(cfg["minimum_truth_scans"])
        and cad_evaluated >= int(cfg["minimum_truth_scans"])
    )
    _check(
        checks,
        "cad_ply.uplift_over_ply_only",
        observed={
            "uplift": None if uplift is None else round(uplift, 4),
            "ply_evaluated_scans": ply_evaluated,
            "cad_ply_evaluated_scans": cad_evaluated,
        },
        required={
            "minimum_uplift": cfg["minimum_cad_uplift"],
            "minimum_aligned_scans_per_lane": cfg["minimum_truth_scans"],
        },
        passed=(
            uplift is not None
            and uplift >= float(cfg["minimum_cad_uplift"])
            and uplift_evidence_sized
        ),
        blocker="CAD does not yet provide the required measured uplift over PLY-only inference",
    )

    operations = dict(operational_summary or {})
    repeat_rate = _number(operations.get("deterministic_repeat_rate"))
    p95_latency = _number(operations.get("p95_latency_sec"))
    job_success = _number(operations.get("job_success_rate"))
    _check(
        checks,
        "operations.determinism",
        observed=repeat_rate,
        required=cfg["minimum_deterministic_repeat_rate"],
        passed=repeat_rate is not None and repeat_rate >= float(cfg["minimum_deterministic_repeat_rate"]),
        blocker="ten-run deterministic repeat evidence is missing or unstable",
    )
    _check(
        checks,
        "operations.p95_latency",
        observed=p95_latency,
        required=f"<= {cfg['maximum_p95_latency_sec']}",
        passed=p95_latency is not None and p95_latency <= float(cfg["maximum_p95_latency_sec"]),
        blocker="P95 processing latency is missing or above the release budget",
    )
    _check(
        checks,
        "operations.job_success_rate",
        observed=job_success,
        required=cfg["minimum_job_success_rate"],
        passed=job_success is not None and job_success >= float(cfg["minimum_job_success_rate"]),
        blocker="job-level reliability evidence is missing or below threshold",
    )

    blockers = [str(item["blocker"]) for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "policy_status": "LEGACY_AGGREGATE_DIAGNOSTIC_ONLY",
        "production_use_allowed": False,
        "requires_independent_verification": True,
        "promotion_status": "eligible" if not blockers else "blocked",
        "gate_count": len(checks),
        "passed_gate_count": sum(bool(item["passed"]) for item in checks),
        "blocked_gate_count": len(blockers),
        "blockers": blockers,
        "thresholds": cfg,
        "checks": checks,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-registry", required=True, type=Path)
    parser.add_argument("--ply-summary", type=Path)
    parser.add_argument("--cad-ply-summary", type=Path)
    parser.add_argument("--operational-summary", type=Path)
    parser.add_argument(
        "--determinism-summary",
        type=Path,
        help="Optional ten-run determinism evidence merged with operational metrics.",
    )
    parser.add_argument("--thresholds-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)

    operational_summary = _load_json(args.operational_summary) or {}
    determinism_summary = _load_json(args.determinism_summary) or {}
    if determinism_summary:
        operational_summary = {
            **operational_summary,
            "deterministic_repeat_rate": determinism_summary.get("deterministic_repeat_rate"),
            "determinism_repeat_count": determinism_summary.get("repeat_count"),
            "determinism_unique_fingerprint_count": determinism_summary.get("unique_fingerprint_count"),
        }
    report = evaluate_promotion_gates(
        truth_registry=_load_json(args.truth_registry) or {},
        ply_summary=_load_json(args.ply_summary),
        cad_ply_summary=_load_json(args.cad_ply_summary),
        operational_summary=operational_summary,
        thresholds=_load_json(args.thresholds_json),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("promotion_status", "passed_gate_count", "blocked_gate_count")}, indent=2))
    return 0 if report["promotion_status"] == "eligible" or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
