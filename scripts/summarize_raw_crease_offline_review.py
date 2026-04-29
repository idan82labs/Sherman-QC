#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


DEFAULT_ANNOTATIONS_DIR = Path("data/bend_corpus/annotations/f1_feature_region_labels_20260427")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_case(value: str) -> Dict[str, Any]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--case must use PART_ID:TARGET_COUNT:TRACK")
    part_id, target_text, track = parts
    if not part_id:
        raise argparse.ArgumentTypeError("case part id cannot be empty")
    try:
        target_count = int(target_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case target count must be an integer") from exc
    return {"part_id": part_id, "target_count": target_count, "track": track}


def _parse_part_count(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use PART_ID=COUNT")
    part_id, count_text = value.split("=", 1)
    if not part_id:
        raise argparse.ArgumentTypeError("part id cannot be empty")
    try:
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count must be an integer") from exc
    return part_id, count


def _quality_path(annotations_dir: Path, part_id: str) -> Path:
    return annotations_dir / f"{part_id}_raw_crease_scan_support_quality.json"


def _arrangement_path(annotations_dir: Path, part_id: str) -> Path:
    return annotations_dir / f"{part_id}_raw_crease_sparse_arrangement.json"


def _recommendation(
    *,
    target_count: int,
    quality: Mapping[str, Any] | None,
    arrangement: Mapping[str, Any] | None,
    existing_f1_exact: int | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    quality_decision = str(quality.get("decision") if quality else "missing_quality")
    arrangement_status = str(arrangement.get("status") if arrangement else "missing_arrangement")
    selected_count = int(arrangement.get("selected_family_count") or 0) if arrangement else 0

    if existing_f1_exact is not None and existing_f1_exact == target_count and quality_decision == "insufficient_support_abstain":
        return "preserve_existing_f1_exact_raw_lane_abstains", ["existing_f1_exact_matches_target", "raw_crease_lane_abstains"]

    if quality_decision == "limited_observed_support":
        reasons.append("scan_support_below_target")
        if existing_f1_exact is not None and existing_f1_exact == target_count:
            reasons.append("existing_f1_exact_matches_target")
            return "preserve_existing_f1_exact_raw_lane_limited", reasons
        return "hold_exact_count_scan_quality_limited", reasons

    if arrangement_status == "sparse_arrangement_exact_candidate" and selected_count == target_count:
        reasons.append("sparse_arrangement_count_matches_target")
        return "candidate_for_sparse_arrangement_review", reasons

    if arrangement_status == "sparse_arrangement_underfit":
        reasons.append("sparse_arrangement_below_target")
        return "hold_underfit_support_creation_needed", reasons

    if arrangement_status == "sparse_arrangement_overcomplete":
        reasons.append("sparse_arrangement_above_target")
        return "manual_review_required_overcomplete", reasons

    if arrangement_status == "no_safe_support" or quality_decision == "insufficient_support_abstain":
        reasons.append("raw_crease_lane_abstains")
        if existing_f1_exact is not None:
            reasons.append("existing_f1_available")
            return "preserve_existing_f1_result_raw_lane_abstains", reasons
        return "raw_crease_lane_abstains", reasons

    reasons.extend([quality_decision, arrangement_status])
    return "manual_review_required", reasons


def _safe_feature_snapshot(quality: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not quality:
        return {}
    snapshot = dict(quality.get("feature_snapshot") or {})
    return {
        "raw_admissible_before_dedupe": snapshot.get("raw_admissible_before_dedupe"),
        "raw_deduped_candidate_count": snapshot.get("raw_deduped_candidate_count"),
        "bridge_safe_candidate_count": snapshot.get("bridge_safe_candidate_count"),
        "bridge_safe_fraction": snapshot.get("bridge_safe_fraction"),
        "safe_flange_pair_count": snapshot.get("safe_flange_pair_count"),
        "safe_flange_pair_counts": snapshot.get("safe_flange_pair_counts") or {},
    }


def _case_row(
    *,
    case: Mapping[str, Any],
    annotations_dir: Path,
    existing_f1_exact_by_part: Mapping[str, int],
) -> Dict[str, Any]:
    part_id = str(case["part_id"])
    target_count = int(case["target_count"])
    quality_file = _quality_path(annotations_dir, part_id)
    arrangement_file = _arrangement_path(annotations_dir, part_id)
    quality = _load_json(quality_file) if quality_file.exists() else None
    arrangement = _load_json(arrangement_file) if arrangement_file.exists() else None
    existing_f1_exact = existing_f1_exact_by_part.get(part_id)
    recommendation, recommendation_reasons = _recommendation(
        target_count=target_count,
        quality=quality,
        arrangement=arrangement,
        existing_f1_exact=existing_f1_exact,
    )
    return {
        "part_id": part_id,
        "target_count": target_count,
        "track": case["track"],
        "existing_f1_exact": existing_f1_exact,
        "quality_decision": quality.get("decision") if quality else "missing",
        "recommended_gate": quality.get("recommended_gate") if quality else "missing",
        "quality_reason_codes": quality.get("reason_codes") if quality else ["missing_quality_report"],
        "feature_snapshot": _safe_feature_snapshot(quality),
        "sparse_status": arrangement.get("status") if arrangement else "missing",
        "sparse_reason_codes": arrangement.get("reason_codes") if arrangement else ["missing_sparse_arrangement_report"],
        "safe_candidate_count": arrangement.get("safe_candidate_count") if arrangement else 0,
        "selected_family_count": arrangement.get("selected_family_count") if arrangement else 0,
        "suppressed_duplicate_count": arrangement.get("suppressed_duplicate_count") if arrangement else 0,
        "selected_candidate_ids": arrangement.get("selected_candidate_ids") if arrangement else [],
        "selected_family_source_candidate_ids": arrangement.get("selected_family_source_candidate_ids") if arrangement else [],
        "final_offline_recommendation": recommendation,
        "final_recommendation_reasons": recommendation_reasons,
        "artifact_paths": {
            "quality_json": str(quality_file),
            "sparse_arrangement_json": str(arrangement_file),
        },
    }


def summarize_raw_crease_offline_review(
    *,
    cases: Iterable[Mapping[str, Any]],
    annotations_dir: Path,
    existing_f1_exact_by_part: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    existing_f1_exact_by_part = existing_f1_exact_by_part or {}
    rows = [
        _case_row(
            case=case,
            annotations_dir=annotations_dir,
            existing_f1_exact_by_part=existing_f1_exact_by_part,
        )
        for case in cases
    ]
    recommendations = Counter(str(row["final_offline_recommendation"]) for row in rows)
    sparse_statuses = Counter(str(row["sparse_status"]) for row in rows)
    quality_decisions = Counter(str(row["quality_decision"]) for row in rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Offline-only review summary for raw-crease support and sparse-arrangement diagnostics.",
        "artifact_contract": "Diagnostic gating only. Does not mutate runtime zero-shot or F1 solver behavior.",
        "case_count": len(rows),
        "recommendation_counts": dict(sorted(recommendations.items())),
        "quality_decision_counts": dict(sorted(quality_decisions.items())),
        "sparse_status_counts": dict(sorted(sparse_statuses.items())),
        "cases": rows,
    }


def _markdown_table(rows: list[Mapping[str, Any]]) -> str:
    lines = [
        "| Part | Target | Track | Quality | Sparse | Selected | Safe | Duplicates | Recommendation |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {part} | {target} | {track} | {quality} | {sparse} | {selected} | {safe} | {dupes} | {recommendation} |".format(
                part=row["part_id"],
                target=row["target_count"],
                track=row["track"],
                quality=row["quality_decision"],
                sparse=row["sparse_status"],
                selected=row["selected_family_count"],
                safe=row["safe_candidate_count"],
                dupes=row["suppressed_duplicate_count"],
                recommendation=row["final_offline_recommendation"],
            )
        )
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = list(report.get("cases") or ())
    lines = [
        "# Raw Crease Offline Review Summary",
        "",
        f"Generated: `{report.get('generated_at')}`",
        "",
        "This is an offline-only diagnostic gate. It summarizes raw-crease support quality and sparse-arrangement evidence; it does not promote any result into runtime.",
        "",
        "## Case Table",
        "",
        _markdown_table(rows),
        "",
        "## Recommendation Counts",
        "",
    ]
    for key, value in sorted((report.get("recommendation_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Case Notes", ""])
    for row in rows:
        reasons = ", ".join(str(reason) for reason in row.get("final_recommendation_reasons") or ())
        quality_reasons = ", ".join(str(reason) for reason in row.get("quality_reason_codes") or ())
        sparse_reasons = ", ".join(str(reason) for reason in row.get("sparse_reason_codes") or ())
        lines.extend(
            [
                f"### {row['part_id']}",
                "",
                f"- Target: `{row['target_count']}`; track: `{row['track']}`; existing F1 exact: `{row.get('existing_f1_exact')}`.",
                f"- Quality gate: `{row['quality_decision']}` / `{row['recommended_gate']}`; reasons: {quality_reasons}.",
                f"- Sparse arrangement: `{row['sparse_status']}`; selected `{row['selected_family_count']}` from `{row['safe_candidate_count']}` bridge-safe candidates; suppressed duplicates `{row['suppressed_duplicate_count']}`.",
                f"- Final recommendation: `{row['final_offline_recommendation']}`; reasons: {reasons}.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize offline raw-crease review gates across scan cases.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--case", action="append", type=_parse_case, required=True, help="PART_ID:TARGET_COUNT:TRACK")
    parser.add_argument("--existing-f1-exact", action="append", type=_parse_part_count, default=[], help="PART_ID=COUNT")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    existing = dict(args.existing_f1_exact)
    report = summarize_raw_crease_offline_review(
        cases=args.case,
        annotations_dir=args.annotations_dir,
        existing_f1_exact_by_part=existing,
    )
    _write_json(args.output_json, report)
    _write_text(args.output_md, render_markdown(report))
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "recommendation_counts": report["recommendation_counts"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
