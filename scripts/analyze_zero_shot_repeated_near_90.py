#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


POLICIES = (
    "conservative_macro",
    "orthogonal_dense_sheetmetal",
    "repeated_90_sheetmetal",
    "rolled_or_mixed_transition",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_repeated_near_90(payload: Dict[str, Any]) -> bool:
    expectation = dict(payload.get("expectation") or {})
    angles = expectation.get("dominant_angle_families_deg") or []
    try:
        return any(abs(float(angle) - 90.0) <= 5.0 for angle in angles)
    except (TypeError, ValueError):
        return False


def _variant_metric(variant: Dict[str, Any], key: str) -> Optional[int]:
    value = variant.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _policy_snapshot(policy_name: str, policy: Dict[str, Any]) -> Dict[str, Any]:
    support_breakdown = dict(policy.get("support_breakdown") or {})
    perturbation = dict(policy.get("perturbation_summary") or support_breakdown.get("perturbation") or {})
    variants = dict(perturbation.get("variant_summaries") or {})
    variant_effective_counts = {
        name: _variant_metric(summary, "effective_count")
        for name, summary in variants.items()
    }
    variant_raw_candidate_counts = {
        name: _variant_metric(summary, "raw_candidate_count")
        for name, summary in variants.items()
    }
    variant_line_group_counts = {
        name: _variant_metric(summary, "line_family_group_count")
        for name, summary in variants.items()
    }
    variant_strip_merge_counts = {
        name: _variant_metric(summary, "strip_duplicate_merge_count") or _variant_metric(summary, "duplicate_merge_count")
        for name, summary in variants.items()
    }
    effective_values = [value for value in variant_effective_counts.values() if value is not None]
    raw_values = [value for value in variant_raw_candidate_counts.values() if value is not None]
    return {
        "policy_name": policy_name,
        "score": policy.get("score"),
        "effective_structural_count": policy.get("effective_structural_count", policy.get("estimated_bend_count")),
        "structural_count_range": policy.get("structural_count_range", policy.get("estimated_bend_count_range")),
        "dominant_angle_deg": (policy.get("dominant_angle_families_deg") or [None])[0],
        "dominant_angle_mass": policy.get("dominant_angle_mass"),
        "reference_strip_count": perturbation.get("reference_strip_count"),
        "mean_matched_reference_fraction": perturbation.get(
            "mean_matched_reference_fraction",
            perturbation.get("matched_reference_fraction_mean"),
        ),
        "identity_stability": perturbation.get("identity_stability"),
        "modal_support": perturbation.get("modal_support", perturbation.get("modal_count_support")),
        "effective_counts": variant_effective_counts,
        "raw_candidate_counts": variant_raw_candidate_counts,
        "line_family_group_counts": variant_line_group_counts,
        "strip_duplicate_merge_counts": variant_strip_merge_counts,
        "effective_count_span": (max(effective_values) - min(effective_values)) if len(effective_values) >= 2 else 0,
        "raw_candidate_count_span": (max(raw_values) - min(raw_values)) if len(raw_values) >= 2 else 0,
    }


def analyze_zero_shot_repeated_near_90(summary_path: Path | str) -> Dict[str, Any]:
    summary_path = Path(summary_path)
    run_dir = summary_path.parent

    repeated_rows: List[Dict[str, Any]] = []
    for payload_path in sorted(run_dir.glob("*.json")):
        if payload_path.name in {"summary.json", "manifest.json"}:
            continue
        payload = _load_json(payload_path)
        if not _is_repeated_near_90(payload):
            continue

        result = dict(payload.get("result") or {})
        policy_scores = dict(result.get("policy_scores") or {})
        policy_rows = [
            _policy_snapshot(policy_name, dict(policy_scores[policy_name]))
            for policy_name in POLICIES
            if policy_name in policy_scores
        ]
        policy_rows.sort(
            key=lambda row: (
                -(row.get("effective_count_span") or 0),
                -(row.get("raw_candidate_count_span") or 0),
                row["policy_name"],
            )
        )

        repeated_rows.append(
            {
                "part_key": payload.get("part_key"),
                "scan_name": payload.get("scan_name"),
                "scan_path": payload.get("scan_path"),
                "expected_bend_count": (payload.get("expectation") or {}).get("expected_bend_count"),
                "selected_profile_family": result.get("profile_family"),
                "selected_count_range": result.get("estimated_bend_count_range"),
                "selected_exact_count": result.get("estimated_bend_count"),
                "abstained": bool(result.get("abstained")),
                "abstain_reasons": list(result.get("abstain_reasons") or []),
                "dominant_angle_families_deg": list(result.get("dominant_angle_families_deg") or []),
                "candidate_generator_summary": {
                    "single_scale_count": (result.get("candidate_generator_summary") or {}).get("single_scale_count"),
                    "multiscale_count": (result.get("candidate_generator_summary") or {}).get("multiscale_count"),
                    "union_candidate_count": (result.get("candidate_generator_summary") or {}).get("union_candidate_count"),
                    "duplicate_line_family_rate": (result.get("candidate_generator_summary") or {}).get("duplicate_line_family_rate"),
                    "strip_duplicate_merge_count": (result.get("candidate_generator_summary") or {}).get("strip_duplicate_merge_count"),
                    "perturbation_variants": (result.get("candidate_generator_summary") or {}).get("perturbation_variants"),
                },
                "policy_rows": policy_rows,
            }
        )

    repeated_rows.sort(
        key=lambda row: (
            row["abstained"],
            -(len(row.get("abstain_reasons") or [])),
            row["part_key"] or "",
        ),
    )

    return {
        "summary_path": str(summary_path),
        "repeated_scan_count": len(repeated_rows),
        "repeated_scans": repeated_rows,
    }


def _markdown_report(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Repeated Near-90 Zero-Shot Analysis",
        "",
        f"- Summary: `{analysis['summary_path']}`",
        f"- Repeated near-90 scans: `{analysis['repeated_scan_count']}`",
        "",
    ]
    for row in analysis["repeated_scans"]:
        lines.extend(
            [
                f"## {row['part_key']} / {row['scan_name']}",
                "",
                f"- Expected bend count: `{row['expected_bend_count']}`",
                f"- Selected profile: `{row['selected_profile_family']}`",
                f"- Selected count range: `{row['selected_count_range']}`",
                f"- Selected exact count: `{row['selected_exact_count']}`",
                f"- Abstained: `{row['abstained']}`",
                f"- Abstain reasons: `{row['abstain_reasons']}`",
                f"- Generator summary: `{row['candidate_generator_summary']}`",
                "",
            ]
        )
        for policy in row["policy_rows"]:
            lines.append(
                f"- `{policy['policy_name']}`: "
                f"effective=`{policy['effective_structural_count']}`, "
                f"range=`{policy['structural_count_range']}`, "
                f"effective_span=`{policy['effective_count_span']}`, "
                f"raw_span=`{policy['raw_candidate_count_span']}`, "
                f"match_frac=`{policy['mean_matched_reference_fraction']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_text(path_str: str, contents: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze repeated near-90 zero-shot scans from an evaluation summary.")
    parser.add_argument("summary")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    analysis = analyze_zero_shot_repeated_near_90(args.summary)
    if args.output_json:
        _write_text(args.output_json, json.dumps(analysis, indent=2, ensure_ascii=False))
    if args.output_markdown:
        _write_text(args.output_markdown, _markdown_report(analysis))
    if not args.output_json and not args.output_markdown:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
