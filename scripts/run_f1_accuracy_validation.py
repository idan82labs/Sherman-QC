#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from bend_inspection_pipeline import load_bend_runtime_config
from patch_graph_latent_decomposition import render_decomposition_artifacts, run_patch_graph_latent_decomposition
from run_patch_graph_f1_prototype import (
    _entry_result_payload,
    _family_residual_repair_promoted_count,
    _load_count_sweep_promotion_evidence,
    _load_promotion_evidence,
    _range_contains,
    _range_width,
    _result_with_supplemental_owned_regions,
    _supplemental_owned_bend_regions_from_evidence,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _slug_for_entry(entry: Mapping[str, Any], index: int) -> str:
    part_key = str(entry.get("part_key") or f"entry_{index}")
    scan_stem = Path(str(entry.get("scan_path") or "scan")).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in scan_stem)
    return f"{index:02d}_{part_key}_{safe_stem}"


def _default_review_output_dir() -> Path:
    icloud_root = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Sherman QC"
    if icloud_root.exists():
        return icloud_root / f"F1 Accuracy Validation Renders {datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    return ROOT / "data/bend_corpus/evaluation" / f"f1_accuracy_validation_renders_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _expected_total_bend_count(entry: Mapping[str, Any]) -> Optional[int]:
    expectation = dict(entry.get("expectation") or {})
    value = expectation.get("expected_bend_count")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _partial_observed_bend_count(entry: Mapping[str, Any]) -> Optional[int]:
    expectation = dict(entry.get("expectation") or {})
    audit_tags = {str(tag) for tag in entry.get("audit_tags") or ()}
    semantics = str(expectation.get("label_semantics") or "")
    if not (
        expectation.get("scan_limited")
        or "visible_observed" in semantics
        or "observed_visible_bends" in audit_tags
    ):
        return None
    return _expected_total_bend_count(entry)


def _partial_whole_part_context(entry: Mapping[str, Any]) -> Optional[int]:
    expectation = dict(entry.get("expectation") or {})
    explicit_context = expectation.get("known_whole_part_total_context")
    if explicit_context is not None:
        try:
            return int(explicit_context)
        except (TypeError, ValueError):
            return None
    if _partial_observed_bend_count(entry) is None:
        return _expected_total_bend_count(entry)
    return None


def _expected_feature_count_semantics(entry: Mapping[str, Any]) -> Dict[str, Any]:
    expectation = dict(entry.get("expectation") or {})
    conventional = expectation.get("expected_conventional_bend_count")
    raised = expectation.get("expected_raised_form_feature_count")
    total = expectation.get("expected_total_feature_count")
    if total is None:
        try:
            total = int(conventional) + int(raised or 0) if conventional is not None else None
        except (TypeError, ValueError):
            total = None
    return {
        "label_semantics": expectation.get("label_semantics"),
        "counted_target": "conventional_bends"
        if expectation.get("expected_conventional_bend_count") is not None
        else "bend_count",
        "expected_conventional_bend_count": conventional,
        "expected_raised_form_feature_count": raised,
        "expected_total_feature_count": total,
        "algorithm_feature_split_available": False,
        "algorithm_feature_split_status": "not_modeled",
    }


def _singleton_count(count_range: Sequence[Any]) -> Optional[int]:
    if len(count_range) < 2:
        return None
    try:
        low = int(count_range[0])
        high = int(count_range[1])
    except (TypeError, ValueError):
        return None
    return low if low == high else None


def _has_final_accepted_exact(algorithm: Mapping[str, Any]) -> bool:
    exact = algorithm.get("accepted_exact_bend_count")
    accepted_range = list(algorithm.get("accepted_bend_count_range") or ())
    singleton = _singleton_count(accepted_range)
    if exact is None or singleton is None:
        return False
    try:
        return int(exact) == int(singleton)
    except (TypeError, ValueError):
        return False


def _known_escalation_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    expected = _expected_total_bend_count(row)
    candidate = dict(row.get("candidate") or {})
    candidate_exact = candidate.get("exact_bend_count")
    candidate_range = list(candidate.get("bend_count_range") or ())
    final_exact = candidate_exact if _has_final_accepted_exact(
        {
            "accepted_exact_bend_count": candidate_exact,
            "accepted_bend_count_range": candidate_range,
        }
    ) else None
    if expected is None:
        reasons.append("known_truth_missing")
    elif final_exact is not None and int(final_exact) != int(expected):
        reasons.append("deterministic_count_conflicts_with_known_truth")
    elif _range_contains(expected, candidate_range) is False:
        reasons.append("candidate_range_excludes_known_truth")
    width = _range_width(candidate_range)
    if width is not None and width > 0:
        reasons.append("non_singleton_known_count_range")
    if dict(row.get("raw_f1_promotion_diagnostic") or {}).get("candidate"):
        reasons.append("raw_f1_manual_promotion_candidate")
    return reasons


def _partial_escalation_reasons(entry: Mapping[str, Any], row: Mapping[str, Any]) -> List[str]:
    reasons = ["partial_scan_observed_truth_required"]
    expected_observed = _partial_observed_bend_count(entry)
    candidate = dict(row.get("candidate") or {})
    candidate_range = list(candidate.get("bend_count_range") or ())
    singleton = _singleton_count(candidate_range)
    candidate_exact = candidate.get("exact_bend_count")
    if candidate_exact is not None or singleton is not None:
        reasons.append("deterministic_observed_count_needs_manual_confirmation")
    if expected_observed is not None:
        predicted = candidate_exact if candidate_exact is not None else singleton
        if predicted is not None and int(predicted) != int(expected_observed):
            reasons.append("deterministic_observed_count_conflicts_with_manual_truth")
        if _range_contains(int(expected_observed), candidate_range) is False:
            reasons.append("observed_count_range_excludes_manual_truth")
    width = _range_width(candidate_range)
    if width is not None and width > 2:
        reasons.append("broad_partial_observed_count_range")
    if _partial_whole_part_context(entry) is not None and expected_observed is None:
        reasons.append("whole_part_total_available_but_not_scored_as_observed_truth")
    return reasons


def _review_payload(
    *,
    track: str,
    entry: Mapping[str, Any],
    row: Mapping[str, Any],
    render_info: Optional[Mapping[str, Any]],
    escalation_reasons: Sequence[str],
) -> Dict[str, Any]:
    candidate = dict(row.get("candidate") or {})
    raw_f1 = dict(row.get("raw_f1_candidate") or {})
    accepted_range = list(candidate.get("bend_count_range") or ())
    final_exact = candidate.get("exact_bend_count") if _has_final_accepted_exact(
        {
            "accepted_exact_bend_count": candidate.get("exact_bend_count"),
            "accepted_bend_count_range": accepted_range,
        }
    ) else None
    observed_prediction = candidate.get("exact_bend_count")
    if observed_prediction is None:
        observed_prediction = _singleton_count(accepted_range)
    if track == "partial" and _singleton_count(accepted_range) is None:
        observed_prediction = None
    payload: Dict[str, Any] = {
        "track": track,
        "part_key": entry.get("part_key"),
        "scan_path": entry.get("scan_path"),
        "scan_limited": track == "partial",
        "validation_target": "visible_observed_bend_count" if track == "partial" else "known_full_scan_bend_count",
        "known_total_bend_count_context": _partial_whole_part_context(entry) if track == "partial" else None,
        "expected_bend_count_scored": _expected_total_bend_count(entry) if track == "known" else _partial_observed_bend_count(entry),
        "algorithm": {
            "accepted_exact_bend_count": final_exact,
            "raw_exact_bend_count_field": candidate.get("exact_bend_count"),
            "accepted_bend_count_range": accepted_range,
            "observed_bend_count_prediction": observed_prediction if track == "partial" else None,
            "candidate_source": candidate.get("candidate_source"),
            "guard_reason": candidate.get("guard_reason"),
            "raw_f1_map_bend_count": raw_f1.get("map_bend_count"),
            "raw_f1_bend_count_range": raw_f1.get("bend_count_range"),
            "owned_bend_region_count": candidate.get("owned_bend_region_count"),
        },
        "scan_family_route": row.get("scan_family_route"),
        "feature_count_semantics": _expected_feature_count_semantics(entry),
        "render_artifacts": dict(render_info or {}),
        "manual_review": {
            "required": bool(escalation_reasons),
            "reason_codes": list(escalation_reasons),
            "observed_bend_count": None,
            "verified_by": None,
            "notes": "",
            "false_positive_bend_ids": [],
            "missed_bend_notes": [],
        },
    }
    if track == "partial":
        payload["manual_review"]["required"] = True
    return payload


def _write_review_contact_sheet(
    *,
    review_dir: Path,
    review: Mapping[str, Any],
    render_info: Mapping[str, Any],
    width_px: int = 1920,
    height_px: int = 1080,
) -> Optional[str]:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None

    atom_projection = render_info.get("atom_projection_path")
    overview = render_info.get("overview_path")
    if not atom_projection and not overview:
        return None

    algorithm = dict(review.get("algorithm") or {})
    manual = dict(review.get("manual_review") or {})
    reasons = list(manual.get("reason_codes") or ())
    route = dict(review.get("scan_family_route") or {})
    title = (
        f"{review.get('part_key')} | {Path(str(review.get('scan_path') or '')).name} | "
        f"range {algorithm.get('accepted_bend_count_range')} | exact {algorithm.get('accepted_exact_bend_count')}"
    )
    subtitle = (
        f"track={review.get('track')} scan_limited={review.get('scan_limited')} "
        f"route={route.get('route_id')} source={algorithm.get('candidate_source')}"
    )
    reason_text = ", ".join(str(reason) for reason in reasons[:5]) if reasons else "no manual escalation"

    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100, facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[0.16, 0.84], width_ratios=[1.0, 1.0, 1.0])
    text_ax = fig.add_subplot(grid[0, :])
    text_ax.axis("off")
    text_ax.text(0.01, 0.72, title, fontsize=18, fontweight="bold", color="#0f172a", va="center")
    text_ax.text(0.01, 0.38, subtitle, fontsize=11.5, color="#334155", va="center")
    text_ax.text(0.01, 0.10, f"review: {reason_text}", fontsize=11.0, color="#9a3412", va="center")

    overview_views = dict(render_info.get("overview_view_paths") or {})
    context_views = dict(render_info.get("scan_context_view_paths") or {})
    side = overview_views.get("side")
    top = overview_views.get("top")
    raw_count = algorithm.get("owned_bend_region_count")
    accepted_range = list(algorithm.get("accepted_bend_count_range") or ())
    diagnostic_overlay = not _has_final_accepted_exact(algorithm)
    rejected_raw_overlay = False
    if diagnostic_overlay and raw_count is not None and len(accepted_range) >= 2:
        try:
            rejected_raw_overlay = not (int(accepted_range[0]) <= int(raw_count) <= int(accepted_range[1]))
        except (TypeError, ValueError):
            rejected_raw_overlay = False
    display_overview = render_info.get("scan_context_path") if diagnostic_overlay else overview
    display_side_or_top = (context_views.get("side") or context_views.get("top")) if diagnostic_overlay else (side or top)
    image_paths = [atom_projection, display_overview, display_side_or_top]
    if diagnostic_overlay:
        image_titles = [
            "Debug raw atom projection",
            "3D final context iso - no accepted dots",
            "3D final context side/top - no accepted dots",
        ]
    else:
        image_titles = ["Owned atom projection", "3D final overlay iso", "3D final overlay side/top"]
    for index, raw_path in enumerate(image_paths):
        ax = fig.add_subplot(grid[1, index])
        ax.axis("off")
        if raw_path and Path(str(raw_path)).exists():
            ax.imshow(mpimg.imread(str(raw_path)))
            ax.set_title(image_titles[index], fontsize=12, fontweight="bold", color="#111827", pad=5)
            if diagnostic_overlay and index > 0:
                warning_text = (
                    f"no final bend dots: accepted result is range {accepted_range}; raw F1 support ({raw_count}) is debug-only"
                    if rejected_raw_overlay
                    else f"no final bend dots: accepted result is range {accepted_range}; raw support is debug-only"
                )
                ax.text(
                    0.5,
                    0.04,
                    warning_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#9a3412",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff7ed", edgecolor="#fdba74", alpha=0.94),
                )
        else:
            ax.text(0.5, 0.5, "missing image", ha="center", va="center", fontsize=14, color="#64748b")

    fig.tight_layout(pad=0.55)
    output_path = review_dir / "review_contact_sheet_1080p.png"
    fig.savefig(output_path, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return str(output_path)


def _aggregate_validation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    known_rows = [row for row in rows if row.get("track") == "known"]
    partial_rows = [row for row in rows if row.get("track") == "partial"]
    known_exact_evaluated = 0
    known_exact_correct = 0
    known_contains_evaluated = 0
    known_contains = 0
    partial_exact_evaluated = 0
    partial_exact_correct = 0
    partial_contains_evaluated = 0
    partial_contains = 0
    render_count = 0
    escalation_count = 0
    for row in rows:
        if row.get("render_artifacts"):
            render_count += 1
        if (row.get("manual_review") or {}).get("required"):
            escalation_count += 1
    for row in known_rows:
        expected = row.get("expected_bend_count_scored")
        algorithm = dict(row.get("algorithm") or {})
        exact = algorithm.get("accepted_exact_bend_count")
        count_range = list(algorithm.get("accepted_bend_count_range") or ())
        if expected is not None and exact is not None and _has_final_accepted_exact(algorithm):
            known_exact_evaluated += 1
            known_exact_correct += int(int(exact) == int(expected))
        if expected is not None and len(count_range) >= 2:
            known_contains_evaluated += 1
            known_contains += int(bool(_range_contains(int(expected), count_range)))
    for row in partial_rows:
        expected = row.get("expected_bend_count_scored")
        algorithm = dict(row.get("algorithm") or {})
        count_range = list(algorithm.get("accepted_bend_count_range") or ())
        singleton = _singleton_count(count_range)
        if expected is not None and singleton is not None:
            partial_exact_evaluated += 1
            partial_exact_correct += int(int(singleton) == int(expected))
        if expected is not None and len(count_range) >= 2:
            partial_contains_evaluated += 1
            partial_contains += int(bool(_range_contains(int(expected), count_range)))
    return {
        "known_scans_processed": len(known_rows),
        "partial_scans_processed": len(partial_rows),
        "known_exact_evaluated": known_exact_evaluated,
        "known_exact_accuracy": round(known_exact_correct / known_exact_evaluated, 3) if known_exact_evaluated else None,
        "known_range_contains_truth_rate": round(known_contains / known_contains_evaluated, 3) if known_contains_evaluated else None,
        "partial_observed_exact_evaluated": partial_exact_evaluated,
        "partial_observed_exact_accuracy": round(partial_exact_correct / partial_exact_evaluated, 3) if partial_exact_evaluated else None,
        "partial_observed_range_contains_truth_rate": round(partial_contains / partial_contains_evaluated, 3) if partial_contains_evaluated else None,
        "scan_limited_partial_count": len([row for row in partial_rows if row.get("scan_limited")]),
        "render_pack_count": render_count,
        "manual_review_required_count": escalation_count,
    }


def _write_review_markdown(review_output_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# F1 Accuracy Validation Review",
        "",
        f"Generated: {summary.get('generated_at')}",
        f"Route mode: `{summary.get('route_mode')}`",
        "",
        "## Aggregate",
        "",
    ]
    aggregate = dict(summary.get("aggregate") or {})
    for key, value in aggregate.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Cases", ""])
    for row in summary.get("results") or []:
        algorithm = dict(row.get("algorithm") or {})
        manual = dict(row.get("manual_review") or {})
        render = dict(row.get("render_artifacts") or {})
        contact = render.get("contact_sheet_1080p_path")
        if contact:
            try:
                contact_link = Path(str(contact)).relative_to(review_output_dir)
            except ValueError:
                contact_link = Path(str(contact))
            contact_md = f"[1080p contact sheet]({contact_link})"
        else:
            contact_md = "no render"
        lines.append(
            f"- `{row.get('track')}` `{row.get('part_key')}` `{Path(str(row.get('scan_path') or '')).name}`: "
            f"range `{algorithm.get('accepted_bend_count_range')}`, exact `{algorithm.get('accepted_exact_bend_count')}`, "
            f"observed `{algorithm.get('observed_bend_count_prediction')}`, "
            f"review `{manual.get('required')}` - {contact_md}"
        )
    (review_output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_track(
    *,
    track: str,
    slice_payload: Mapping[str, Any],
    output_dir: Path,
    review_output_dir: Path,
    runtime_config: Any,
    route_mode: str,
    cache_dir: Path,
    promotion_evidence: Mapping[str, Dict[str, Any]],
    render_all_known: bool,
    limit: Optional[int],
    part_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    selected = list(slice_payload.get("entries") or [])
    part_key_filter = {str(value) for value in part_keys if str(value)}
    if part_key_filter:
        selected = [entry for entry in selected if str(entry.get("part_key") or "") in part_key_filter]
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    rows: List[Dict[str, Any]] = []
    for index, entry in enumerate(selected, start=1):
        slug = _slug_for_entry(entry, index)
        part_key = str(entry.get("part_key") or slug)
        scan_path = str(entry["scan_path"])
        print(f"[{track} {index}/{len(selected)}] START {part_key}: {scan_path}", flush=True)
        case_dir = output_dir / track / slug
        case_dir.mkdir(parents=True, exist_ok=True)
        result = run_patch_graph_latent_decomposition(
            scan_path=scan_path,
            runtime_config=runtime_config,
            part_id=part_key,
            cache_dir=cache_dir / track / slug,
            route_mode=route_mode,
        )
        _write_json(case_dir / "decomposition_result.json", result.to_dict())
        row = _entry_result_payload(
            entry=dict(entry),
            result=result,
            render_info=None,
            promotion_evidence=promotion_evidence.get(part_key),
        )
        part_promotion_evidence = promotion_evidence.get(part_key)
        supplemental_regions = tuple()
        if (
            _family_residual_repair_promoted_count(
                promotion_evidence=part_promotion_evidence,
                raw_exact=int(result.exact_bend_count),
                raw_range=result.bend_count_range,
                control_range=result.current_zero_shot_count_range,
            )
            is not None
        ):
            supplemental_regions = _supplemental_owned_bend_regions_from_evidence(
                promotion_evidence=part_promotion_evidence,
                existing_regions=result.owned_bend_regions,
            )
        render_result = _result_with_supplemental_owned_regions(
            result=result,
            supplemental_regions=supplemental_regions,
        )
        if track == "partial":
            escalation_reasons = _partial_escalation_reasons(entry, row)
            should_render = True
        else:
            escalation_reasons = _known_escalation_reasons({**row, "expectation": entry.get("expectation")})
            should_render = bool(render_all_known or escalation_reasons)
        render_info = None
        if should_render:
            render_dir = review_output_dir / track / slug
            if supplemental_regions:
                _write_json(case_dir / "decomposition_result_with_supplemental_regions.json", render_result.to_dict())
            render_info = render_decomposition_artifacts(result=render_result, output_dir=render_dir / "render")
        review = _review_payload(
            track=track,
            entry=entry,
            row=row,
            render_info=render_info,
            escalation_reasons=escalation_reasons,
        )
        _write_json(case_dir / "validation_review.json", review)
        if should_render:
            review_dir = review_output_dir / track / slug
            contact_sheet = _write_review_contact_sheet(
                review_dir=review_dir,
                review=review,
                render_info=dict(render_info or {}),
            )
            if contact_sheet:
                review["render_artifacts"]["contact_sheet_1080p_path"] = contact_sheet
            _write_json(review_dir / "validation_review.json", review)
            _write_json(case_dir / "validation_review.json", review)
        rows.append(review)
        print(
            f"[{track} {index}/{len(selected)}] DONE {part_key}: range={review['algorithm']['accepted_bend_count_range']} review={review['manual_review']['required']}",
            flush=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline F1 accuracy validation with 1080p rendered review evidence.")
    parser.add_argument("--known-slice-json", default="data/bend_corpus/evaluation_slices/zero_shot_known_count_plys.json")
    parser.add_argument("--partial-slice-json", default="data/bend_corpus/evaluation_slices/zero_shot_partial_total_bends.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--review-output-dir", default=None, help="Render-pack output root. Defaults to the Sherman QC iCloud folder when available.")
    parser.add_argument("--track", choices=("known", "partial", "both"), default="both")
    parser.add_argument("--route-mode", choices=("off", "observe", "apply"), default="apply")
    parser.add_argument("--part-key", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--render-all-known", action="store_true", help="Render all known-count cases instead of only mismatches/suspicious ranges.")
    parser.add_argument("--count-sweep-summary-json", action="append", default=[])
    parser.add_argument("--promotion-evidence-json", action="append", default=[])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "data/bend_corpus/evaluation" / f"f1_accuracy_validation_{timestamp}"
    review_output_dir = Path(args.review_output_dir).expanduser().resolve() if args.review_output_dir else _default_review_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = load_bend_runtime_config(None)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "_cache"
    promotion_evidence = _load_count_sweep_promotion_evidence(args.count_sweep_summary_json or ())
    promotion_evidence.update(_load_promotion_evidence(args.promotion_evidence_json or ()))

    rows: List[Dict[str, Any]] = []
    if args.track in ("known", "both"):
        rows.extend(
            _run_track(
                track="known",
                slice_payload=_load_json(Path(args.known_slice_json).resolve()),
                output_dir=output_dir,
                review_output_dir=review_output_dir,
                runtime_config=runtime_config,
                route_mode=args.route_mode,
                cache_dir=cache_dir,
                promotion_evidence=promotion_evidence,
                render_all_known=bool(args.render_all_known),
                limit=args.limit,
                part_keys=args.part_key,
            )
        )
    if args.track in ("partial", "both"):
        rows.extend(
            _run_track(
                track="partial",
                slice_payload=_load_json(Path(args.partial_slice_json).resolve()),
                output_dir=output_dir,
                review_output_dir=review_output_dir,
                runtime_config=runtime_config,
                route_mode=args.route_mode,
                cache_dir=cache_dir,
                promotion_evidence=promotion_evidence,
                render_all_known=bool(args.render_all_known),
                limit=args.limit,
                part_keys=args.part_key,
            )
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "review_output_dir": str(review_output_dir),
        "route_mode": args.route_mode,
        "validation_target": {
            "known": "known full-scan bend count",
            "partial": "visible observed bend count only",
        },
        "aggregate": _aggregate_validation(rows),
        "results": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(review_output_dir / "VALIDATION_SUMMARY.json", summary)
    _write_review_markdown(review_output_dir, summary)
    print(json.dumps({"output_dir": str(output_dir), "review_output_dir": str(review_output_dir), "aggregate": summary["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
