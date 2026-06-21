from __future__ import annotations

from typing import Any, Dict


def augment_progressive_bend_report(report_dict: Dict[str, Any], details: Any) -> Dict[str, Any]:
    report_dict.setdefault("summary", {})
    report_dict["summary"]["expected_bends"] = details.expected_bend_count
    report_dict["summary"]["expected_progress_pct"] = round(details.expected_progress_pct, 2)
    report_dict["summary"]["overdetected_vs_expected"] = details.overdetected_count
    completed_bends = int(
        report_dict["summary"].get(
            "countable_completed_bends",
            report_dict["summary"].get("completed_bends", report_dict["summary"].get("detected", 0)),
        )
    )
    completed_in_spec = int(report_dict["summary"].get("completed_in_spec", report_dict["summary"].get("passed", 0)))
    completed_out_of_spec = max(0, completed_bends - completed_in_spec)
    remaining_expected = max(0, int(details.expected_bend_count) - min(completed_bends, int(details.expected_bend_count)))
    report_dict["summary"]["completed_bends"] = completed_bends
    report_dict["summary"]["countable_completed_bends"] = completed_bends
    report_dict["summary"]["completed_in_spec"] = completed_in_spec
    report_dict["summary"]["completed_out_of_spec"] = completed_out_of_spec
    report_dict["summary"]["remaining_bends"] = remaining_expected
    report_dict["summary"]["countable_remaining_bends"] = remaining_expected
    report_dict["summary"]["is_complete"] = remaining_expected == 0
    rolled_count = sum(1 for m in report_dict.get("matches", []) if m.get("bend_form") == "ROLLED")
    folded_count = sum(1 for m in report_dict.get("matches", []) if m.get("bend_form") == "FOLDED")
    report_dict["summary"]["rolled_bends"] = rolled_count
    report_dict["summary"]["folded_bends"] = folded_count
    if isinstance(report_dict.get("scan_quality"), dict):
        sq = report_dict["scan_quality"]
        report_dict["summary"]["scan_quality_status"] = sq.get("status")
        report_dict["summary"]["scan_coverage_pct"] = sq.get("coverage_pct")
        report_dict["summary"]["scan_density_pts_per_cm2"] = sq.get("density_pts_per_cm2")
    operator_actions = report_dict.get("operator_actions", [])
    matches = list(report_dict.get("matches") or [])
    def _match_countable(match: Dict[str, Any]) -> bool:
        if "countable_in_regression" in match:
            return bool(match.get("countable_in_regression", True))
        return bool((match.get("cad_bend") or {}).get("countable_in_regression", True))

    def _match_bend_id(match: Dict[str, Any]) -> str:
        if "bend_id" in match:
            return str(match.get("bend_id"))
        return str((match.get("cad_bend") or {}).get("bend_id"))

    countable_bend_ids = {
        _match_bend_id(match)
        for match in matches
        if _match_countable(match)
    }
    countable_operator_actions = [
        action
        for action in operator_actions
        if str(action.get("bend_id")) in countable_bend_ids
    ]
    process_feature_actions = [
        action
        for action in operator_actions
        if str(action.get("bend_id")) not in countable_bend_ids
    ]
    critical_actions = [
        a for a in countable_operator_actions
        if a.get("status") in {"FAIL", "NOT_DETECTED"}
    ]
    report_dict["summary"]["critical_actions"] = len(critical_actions)
    report_dict["summary"]["process_feature_actions"] = len(process_feature_actions)
    scan_quality = report_dict.get("scan_quality", {}) if isinstance(report_dict, dict) else {}
    quality_status = str(scan_quality.get("status", "")).strip().upper()
    quality_suffix = ""
    if quality_status:
        coverage_val = scan_quality.get("coverage_pct", None)
        density_val = scan_quality.get("density_pts_per_cm2", None)
        coverage_txt = f"{coverage_val:.0f}%" if isinstance(coverage_val, (int, float)) else "-"
        density_txt = f"{density_val:.0f}" if isinstance(density_val, (int, float)) else "-"
        quality_suffix = f" | Scan {quality_status} ({coverage_txt} cov, {density_txt} pts/cm^2)"
    report_dict["operator_brief"] = {
        "headline": (
            f"{completed_bends}/{details.expected_bend_count} completed, "
            f"{completed_in_spec} in spec, {completed_out_of_spec} out of spec, "
            f"{remaining_expected} remaining"
            f"{quality_suffix}"
        ),
        "actions": countable_operator_actions[:6],
    }
    if process_feature_actions:
        report_dict["process_feature_actions"] = process_feature_actions
    return report_dict
