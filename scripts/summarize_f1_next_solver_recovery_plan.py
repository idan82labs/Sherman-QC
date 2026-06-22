#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _case_category(row: Mapping[str, Any]) -> tuple[str, str]:
    recommendation = str(row.get("final_offline_recommendation") or "")
    track = str(row.get("track") or "")
    target = int(row.get("target_count") or 0)
    selected = int(row.get("selected_family_count") or 0)
    safe = int(row.get("safe_candidate_count") or 0)

    if recommendation == "candidate_for_sparse_arrangement_review":
        return "sparse_arrangement_review_candidate", "review clean sparse arrangement before any runtime wiring"
    if recommendation == "hold_rolled_mixed_transition_semantics_needed":
        return "rolled_mixed_transition_semantics", "requires separate rolled/mixed transition semantics, not conventional bend-line promotion"
    if recommendation.startswith("preserve_existing_f1"):
        return "preserve_existing_f1_raw_lane_abstains", "raw lane abstains; keep existing F1/control result"
    if recommendation == "hold_exact_count_scan_quality_limited" and track == "known" and target >= 8:
        return "hard_known_latent_support_creation", "hard known case under-covered by raw creases; needs latent support creation/decomposition"
    if recommendation == "hold_exact_count_scan_quality_limited":
        return "scan_quality_or_observed_support_limited", "scan support below target; keep observed-support/scan-quality gate"
    if recommendation == "hold_underfit_support_creation_needed":
        if selected < target and safe >= target:
            return "same_family_underfit_or_semantics", "bridge-safe supports collapse into too few families; inspect semantics/same-pair split"
        return "support_creation_underfit", "support creation underfits target"
    return "manual_triage_required", "unclassified diagnostic state"


def _case_plan(row: Mapping[str, Any]) -> Dict[str, Any]:
    category, rationale = _case_category(row)
    target = int(row.get("target_count") or 0)
    selected = int(row.get("selected_family_count") or 0)
    safe = int(row.get("safe_candidate_count") or 0)
    gap = max(0, target - selected)
    next_experiment = {
        "sparse_arrangement_review_candidate": "visual_review_then_expand_clean_known_validation",
        "hard_known_latent_support_creation": "typed_latent_support_creation_or_junction_decomposition",
        "scan_quality_or_observed_support_limited": "scan_quality_observed_support_gate",
        "rolled_mixed_transition_semantics": "rolled_mixed_transition_router_and_metadata_class",
        "preserve_existing_f1_raw_lane_abstains": "preserve_existing_result_and_use_raw_lane_as_negative_control",
        "same_family_underfit_or_semantics": "same_pair_split_or_semantics_disambiguation",
        "support_creation_underfit": "support_creation_birth_recovery",
        "manual_triage_required": "manual_triage",
    }[category]
    return {
        "part_id": row.get("part_id"),
        "track": row.get("track"),
        "target_count": target,
        "selected_family_count": selected,
        "safe_candidate_count": safe,
        "underfit_gap": gap,
        "category": category,
        "rationale": rationale,
        "next_experiment": next_experiment,
        "current_recommendation": row.get("final_offline_recommendation"),
        "quality_decision": row.get("quality_decision"),
        "sparse_status": row.get("sparse_status"),
        "part_family": row.get("part_family"),
    }


def summarize_next_solver_recovery_plan(raw_review: Mapping[str, Any]) -> Dict[str, Any]:
    cases = [_case_plan(row) for row in raw_review.get("cases") or ()]
    category_counts = Counter(str(row["category"]) for row in cases)
    experiment_counts = Counter(str(row["next_experiment"]) for row in cases)
    tranches = [
        {
            "tranche_id": "T1_clean_sparse_arrangement_candidate",
            "purpose": "Validate the only current sparse-arrangement exact candidate before considering any runtime wiring.",
            "case_ids": [row["part_id"] for row in cases if row["category"] == "sparse_arrangement_review_candidate"],
            "exit_rule": "Requires more clean-known cases with visually correct owned bend-line support; current evidence is insufficient for runtime.",
        },
        {
            "tranche_id": "T2_hard_known_latent_support_creation",
            "purpose": "Recover hard repeated/raised-feature known scans where raw sparse creases under-cover the target.",
            "case_ids": [row["part_id"] for row in cases if row["category"] == "hard_known_latent_support_creation"],
            "exit_rule": "Do not promote raw sparse lane; prototype typed support creation / latent decomposition and require visual support coverage.",
        },
        {
            "tranche_id": "T3_scan_quality_observed_support_gate",
            "purpose": "Keep poor/partial scans behind observed-support gates rather than forcing exact full-count promotion.",
            "case_ids": [row["part_id"] for row in cases if row["category"] == "scan_quality_or_observed_support_limited"],
            "exit_rule": "Report observed support limits and escalate manual validation when scan coverage is insufficient.",
        },
        {
            "tranche_id": "T4_rolled_mixed_transition_semantics",
            "purpose": "Separate rounded/rolled mixed transitions from conventional bend-line counts.",
            "case_ids": [row["part_id"] for row in cases if row["category"] == "rolled_mixed_transition_semantics"],
            "exit_rule": "Requires an explicit rolled/mixed-transition class before any count promotion.",
        },
        {
            "tranche_id": "T5_preserve_existing_f1_exact",
            "purpose": "Use raw-crease abstain cases as negative controls and preserve existing exact F1/control results.",
            "case_ids": [row["part_id"] for row in cases if row["category"] == "preserve_existing_f1_raw_lane_abstains"],
            "exit_rule": "Raw lane should not replace exact accepted results when it abstains.",
        },
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Next solver recovery plan derived from offline raw-crease review gates.",
        "case_count": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "next_experiment_counts": dict(sorted(experiment_counts.items())),
        "cases": cases,
        "tranches": tranches,
        "primary_next_step": "T2_hard_known_latent_support_creation",
        "promotion_rule": "No runtime promotion until a tranche wins on visual support coverage and count accuracy without breaking preserve-existing and scan-quality gates.",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F1 Next Solver Recovery Plan",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"Primary next step: `{report.get('primary_next_step')}`",
        "",
        f"Promotion rule: {report.get('promotion_rule')}",
        "",
        "## Tranches",
        "",
    ]
    for tranche in report.get("tranches") or []:
        case_ids = ", ".join(f"`{case_id}`" for case_id in tranche.get("case_ids") or ()) or "None"
        lines.extend(
            [
                f"### {tranche['tranche_id']}",
                "",
                f"- Purpose: {tranche['purpose']}",
                f"- Cases: {case_ids}",
                f"- Exit rule: {tranche['exit_rule']}",
                "",
            ]
        )
    lines.extend(["## Case Routing", ""])
    lines.append("| Part | Target | Selected | Safe | Gap | Category | Next Experiment |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in report.get("cases") or []:
        lines.append(
            "| {part} | {target} | {selected} | {safe} | {gap} | `{category}` | `{experiment}` |".format(
                part=row["part_id"],
                target=row["target_count"],
                selected=row["selected_family_count"],
                safe=row["safe_candidate_count"],
                gap=row["underfit_gap"],
                category=row["category"],
                experiment=row["next_experiment"],
            )
        )
    lines.extend(
        [
            "",
            "## Engineering Conclusion",
            "",
            "- Stop expanding raw-crease sparse promotion for hard cases; it under-covers `10839000` and `ysherman`.",
            "- The next solver implementation should target typed latent support creation / decomposition for hard known cases.",
            "- Keep `37361005` as the only sparse-arrangement review candidate and use the other cases as gates/blockers.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the next solver recovery plan from raw-crease review gates.")
    parser.add_argument("--raw-crease-review-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    report = summarize_next_solver_recovery_plan(_load_json(args.raw_crease_review_json))
    _write_json(args.output_json, report)
    _write_text(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "primary_next_step": report["primary_next_step"],
                "category_counts": report["category_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
