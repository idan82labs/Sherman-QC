#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


CONTACT_GATING_REASONS = {
    "missing_designated_flange_contact",
    "contact_imbalance",
    "boundary_leakage",
    "attachment_penalty_too_high",
    "weak_incident_flange_pair",
    "unsupported_full_reattachment",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_repair_steps(repair_diagnostics: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for iteration in repair_diagnostics.get("iterations") or ():
        for step in iteration.get("repair_steps") or ():
            if isinstance(step, Mapping):
                yield step


def _collect_repair_pressure(decomposition: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = decomposition.get("repair_diagnostics") or {}
    reason_counts: Counter[str] = Counter()
    birth_reject_counts: Counter[str] = Counter()
    inactive_birth_counts: Counter[str] = Counter()
    pruned_labels: set[str] = set()
    duplicate_pruned_labels: set[str] = set()
    selected_pair_duplicate_candidates = 0
    fragment_pruned_atoms = 0
    activated_labels: set[str] = set()

    for step in _iter_repair_steps(diagnostics):
        prune = step.get("prune_diagnostics") or {}
        reason_counts.update({str(k): _as_int(v) for k, v in (prune.get("inadmissible_reason_counts") or {}).items()})
        pruned_labels.update(str(value) for value in prune.get("inadmissible_pruned_labels") or ())
        duplicate_pruned_labels.update(str(value) for value in prune.get("duplicate_pruned_labels") or ())
        fragment_pruned_atoms += _as_int(prune.get("fragment_pruned_atoms"))
        activated_labels.update(str(value) for value in step.get("activated_bend_labels") or ())

        selected_pair = step.get("selected_pair_prune_diagnostics") or {}
        selected_pair_duplicate_candidates += _as_int(selected_pair.get("selected_pair_duplicate_candidates"))
        duplicate_pruned_labels.update(str(value) for value in selected_pair.get("selected_pair_duplicate_pruned") or ())

        birth = step.get("birth_rescue_diagnostics") or {}
        birth_reject_counts.update(
            {str(k): _as_int(v) for k, v in (birth.get("admissibility_reason_counts") or {}).items()}
        )
        for detail in birth.get("skipped_birth_details") or ():
            if isinstance(detail, Mapping):
                inactive_birth_counts[str(detail.get("reason") or "unknown")] += 1

    contact_reason_total = sum(reason_counts.get(reason, 0) for reason in CONTACT_GATING_REASONS)
    birth_contact_reject_total = sum(birth_reject_counts.get(reason, 0) for reason in CONTACT_GATING_REASONS)
    return {
        "inadmissible_reason_counts": dict(sorted(reason_counts.items())),
        "contact_gating_reason_total": int(contact_reason_total),
        "birth_reject_reason_counts": dict(sorted(birth_reject_counts.items())),
        "birth_contact_reject_total": int(birth_contact_reject_total),
        "skipped_birth_reason_counts": dict(sorted(inactive_birth_counts.items())),
        "inadmissible_pruned_label_count": len(pruned_labels),
        "duplicate_pruned_label_count": len(duplicate_pruned_labels),
        "selected_pair_duplicate_candidate_count": int(selected_pair_duplicate_candidates),
        "fragment_pruned_atom_count": int(fragment_pruned_atoms),
        "activated_bend_label_count": len(activated_labels),
        "repair_step_count": sum(1 for _ in _iter_repair_steps(diagnostics)),
    }


def _route_snapshot(decomposition: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = decomposition.get("repair_diagnostics") or {}
    route = diagnostics.get("scan_family_route") or decomposition.get("scan_family_route") or {}
    return {
        "route_id": route.get("route_id"),
        "solver_profile": route.get("solver_profile"),
        "reason_codes": route.get("reason_codes") or [],
        "confidence": route.get("confidence"),
        "applied": route.get("applied"),
        "feature_snapshot": route.get("feature_snapshot") or {},
    }


def _sparse_snapshot(sparse_arrangement: Mapping[str, Any] | None) -> Dict[str, Any]:
    sparse_arrangement = sparse_arrangement or {}
    target = _as_int(sparse_arrangement.get("target_count"))
    selected = _as_int(sparse_arrangement.get("selected_family_count"))
    safe = _as_int(sparse_arrangement.get("safe_candidate_count"))
    return {
        "status": sparse_arrangement.get("status"),
        "target_count": target,
        "selected_family_count": selected,
        "safe_candidate_count": safe,
        "selected_underfit_gap": max(0, target - selected),
        "safe_underfit_gap": max(0, target - safe),
        "suppressed_duplicate_count": _as_int(sparse_arrangement.get("suppressed_duplicate_count")),
    }


def _quality_snapshot(quality: Mapping[str, Any] | None) -> Dict[str, Any]:
    quality = quality or {}
    features = quality.get("feature_snapshot") or {}
    return {
        "decision": quality.get("decision"),
        "recommended_gate": quality.get("recommended_gate"),
        "reason_codes": quality.get("reason_codes") or [],
        "raw_admissible_before_dedupe": features.get("raw_admissible_before_dedupe"),
        "raw_deduped_candidate_count": features.get("raw_deduped_candidate_count"),
        "bridge_safe_candidate_count": features.get("bridge_safe_candidate_count"),
        "bridge_safe_fraction": features.get("bridge_safe_fraction"),
        "safe_flange_pair_count": features.get("safe_flange_pair_count"),
    }


def classify_gap(
    *,
    sparse: Mapping[str, Any],
    repair_pressure: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
) -> tuple[str, list[str], str]:
    target = _as_int(sparse.get("target_count"))
    selected = _as_int(sparse.get("selected_family_count"))
    safe = _as_int(sparse.get("safe_candidate_count"))
    selected_gap = max(0, target - selected)
    safe_gap = max(0, target - safe)
    contact_total = _as_int(repair_pressure.get("contact_gating_reason_total"))
    birth_contact_total = _as_int(repair_pressure.get("birth_contact_reject_total"))
    skipped_births = repair_pressure.get("skipped_birth_reason_counts") or {}
    duplicate_candidates = _as_int(repair_pressure.get("selected_pair_duplicate_candidate_count"))
    fragment_atoms = _as_int(repair_pressure.get("fragment_pruned_atom_count"))
    reasons: list[str] = []

    if selected_gap <= 0:
        return "no_selected_underfit", ["selected_sparse_count_reaches_target"], "preserve_as_control_or_visual_review"

    if safe_gap > 0:
        reasons.append("raw_bridge_safe_support_under_covers_target")

    if contact_total >= 4 or birth_contact_total >= 2:
        reasons.append("many_candidates_die_on_flange_contact_or_attachment")

    if _as_int(skipped_births.get("active_count_cap")) > 0:
        reasons.append("birth_recovery_blocked_by_active_count_cap")

    if duplicate_candidates > 0:
        reasons.append("selected_pair_duplicate_pressure_present")

    if fragment_atoms > 0:
        reasons.append("fragment_pruning_present")

    if "many_candidates_die_on_flange_contact_or_attachment" in reasons and safe_gap > 0:
        return (
            "latent_support_creation_with_contact_recovery_gap",
            reasons,
            "prototype contact-aware latent support creation: recover candidate support first, then re-score incidence instead of pruning early",
        )

    if safe_gap > 0:
        return (
            "candidate_support_generation_gap",
            reasons,
            "add support creation/birth proposals before sparse arrangement selection",
        )

    if duplicate_candidates > 0 or fragment_atoms > 0:
        return (
            "same_family_or_fragmentation_gap",
            reasons,
            "improve selected-pair duplicate collapse and split/merge before count promotion",
        )

    return (
        "unclassified_underfit_gap",
        reasons or ["selected_sparse_count_below_target_without_clear_pressure"],
        "manual diagnostic review before implementation",
    )


def analyze_case(
    *,
    decomposition: Mapping[str, Any],
    sparse_arrangement: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None = None,
    case_label: str | None = None,
) -> Dict[str, Any]:
    sparse = _sparse_snapshot(sparse_arrangement)
    repair_pressure = _collect_repair_pressure(decomposition)
    gap_class, reason_codes, next_experiment = classify_gap(
        sparse=sparse,
        repair_pressure=repair_pressure,
        quality=quality,
    )
    route = _route_snapshot(decomposition)
    return {
        "case_label": case_label or decomposition.get("part_id") or sparse_arrangement.get("part_id") if sparse_arrangement else decomposition.get("part_id"),
        "part_id": decomposition.get("part_id") or (sparse_arrangement or {}).get("part_id"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Offline T2 diagnostic for hard known F1 underfit: support creation vs contact/incidence pruning vs duplicate fragmentation.",
        "promotion_safe": False,
        "gap_classification": gap_class,
        "reason_codes": reason_codes,
        "recommended_next_experiment": next_experiment,
        "route": route,
        "sparse_arrangement": sparse,
        "raw_support_quality": _quality_snapshot(quality),
        "decomposition_summary": {
            "exact_bend_count": decomposition.get("exact_bend_count"),
            "bend_count_range": decomposition.get("bend_count_range"),
            "candidate_solution_counts": decomposition.get("candidate_solution_counts"),
            "owned_bend_region_count": len(decomposition.get("owned_bend_regions") or ()),
            "bend_hypothesis_count": len(decomposition.get("bend_hypotheses") or ()),
            "flange_hypothesis_count": len(decomposition.get("flange_hypotheses") or ()),
            "atom_count": len(((decomposition.get("atom_graph") or {}).get("atoms") or ())),
        },
        "repair_pressure": repair_pressure,
        "engineering_conclusion": (
            "Do not promote this lane. The next implementation needs typed latent support creation/contact recovery before sparse count selection."
            if gap_class == "latent_support_creation_with_contact_recovery_gap"
            else "Do not promote this lane until the diagnosed underfit class is resolved and visually validated."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    if "cases" in report:
        lines = [
            "# Hard Known F1 Latent Support Gap Summary",
            "",
            f"Generated: `{report.get('generated_at')}`",
            "",
            "## Cases",
            "",
            "| Case | Target | Selected | Safe | Gap Class | Main Reasons | Next Experiment |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for case in report.get("cases") or ():
            sparse = case.get("sparse_arrangement") or {}
            lines.append(
                "| {case} | {target} | {selected} | {safe} | `{gap}` | {reasons} | {next} |".format(
                    case=case.get("part_id") or case.get("case_label"),
                    target=sparse.get("target_count"),
                    selected=sparse.get("selected_family_count"),
                    safe=sparse.get("safe_candidate_count"),
                    gap=case.get("gap_classification"),
                    reasons=", ".join(case.get("reason_codes") or ()),
                    next=case.get("recommended_next_experiment"),
                )
            )
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                "- Raw sparse arrangement under-covers both hard known cases.",
                "- Decomposition diagnostics show strong contact/incidence pruning pressure, so the next tranche should recover support/contact before pruning.",
                "- Runtime/main promotion remains blocked.",
            ]
        )
        return "\n".join(lines) + "\n"

    sparse = report.get("sparse_arrangement") or {}
    repair = report.get("repair_pressure") or {}
    lines = [
        f"# Latent Support Gap Diagnostic: {report.get('part_id') or report.get('case_label')}",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        f"- Gap classification: `{report.get('gap_classification')}`",
        f"- Target / selected / safe: `{sparse.get('target_count')}` / `{sparse.get('selected_family_count')}` / `{sparse.get('safe_candidate_count')}`",
        f"- Recommendation: {report.get('recommended_next_experiment')}",
        f"- Contact gating reason total: `{repair.get('contact_gating_reason_total')}`",
        f"- Pruned inadmissible labels: `{repair.get('inadmissible_pruned_label_count')}`",
        f"- Fragment pruned atoms: `{repair.get('fragment_pruned_atom_count')}`",
        "",
        "## Reason Codes",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in report.get("reason_codes") or ())
    lines.extend(["", "## Engineering Conclusion", "", str(report.get("engineering_conclusion") or "")])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose hard known F1 latent support underfit gaps.")
    parser.add_argument("--decomposition-json", action="append", required=True, type=Path)
    parser.add_argument("--sparse-arrangement-json", action="append", required=True, type=Path)
    parser.add_argument("--quality-json", action="append", type=Path, default=[])
    parser.add_argument("--case-label", action="append", default=[])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    if len(args.decomposition_json) != len(args.sparse_arrangement_json):
        raise SystemExit("--decomposition-json and --sparse-arrangement-json counts must match")
    if args.quality_json and len(args.quality_json) != len(args.decomposition_json):
        raise SystemExit("--quality-json count must match --decomposition-json count when provided")

    cases = []
    for index, decomp_path in enumerate(args.decomposition_json):
        quality = _load_json(args.quality_json[index]) if args.quality_json else None
        label = args.case_label[index] if index < len(args.case_label) else None
        cases.append(
            analyze_case(
                decomposition=_load_json(decomp_path),
                sparse_arrangement=_load_json(args.sparse_arrangement_json[index]),
                quality=quality,
                case_label=label,
            )
        )

    report: Dict[str, Any]
    if len(cases) == 1:
        report = cases[0]
    else:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "purpose": "Aggregate hard known F1 latent support gap diagnostic.",
            "case_count": len(cases),
            "promotion_safe": False,
            "cases": cases,
        }
    _write_json(args.output_json, report)
    _write_text(args.output_md, render_markdown(report))
    print(json.dumps({"case_count": len(cases), "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
