#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_engine_gap_blockers(
    summary_path: Path | str,
    *,
    full_only: bool = True,
    top_n: int = 8,
    unique_part_slice: bool = True,
) -> Dict[str, Any]:
    summary_path = Path(summary_path)
    summary = _load_json(summary_path)

    ranked_rows: List[Dict[str, Any]] = []
    scan_state_counter: Counter[str] = Counter()
    scan_quality_counter: Counter[str] = Counter()
    part_counter: Counter[str] = Counter()

    for result in summary.get("results") or []:
        metrics = dict(result.get("metrics") or {})
        details = dict(result.get("details") or {})
        expectation = dict(result.get("expectation") or {})
        engine_gap = int(metrics.get("engine_gap_position_unknown_bends") or 0)
        if engine_gap <= 0:
            continue

        scan_state = str(
            details.get("scan_state")
            or expectation.get("state")
            or "unknown"
        ).strip().lower() or "unknown"
        if full_only and scan_state != "full":
            continue

        row = {
            "part_key": result["part_key"],
            "scan_name": result["scan_name"],
            "scan_path": result["scan_path"],
            "scan_state": scan_state,
            "scan_quality_status": str(details.get("scan_quality_status") or "UNKNOWN"),
            "engine_gap_position_unknown_bends": engine_gap,
            "scan_limited_position_unknown_bends": int(metrics.get("scan_limited_position_unknown_bends") or 0),
            "unknown_position_bends": int(metrics.get("unknown_position_bends") or 0),
            "position_eligible_bends": int(metrics.get("position_eligible_bends") or 0),
            "correspondence_ready_bends": int(metrics.get("correspondence_decision_ready_bends") or 0),
            "release_decision": str(metrics.get("release_decision") or ""),
            "expected_total_bends": expectation.get("expected_total_bends"),
            "expected_completed_bends": expectation.get("expected_completed_bends"),
            "dense_local_refinement_decision": details.get("dense_local_refinement_decision"),
            "dense_local_engine_gap_bends_before": int(details.get("dense_local_engine_gap_bends_before") or 0),
            "dense_local_engine_gap_bends_after": int(details.get("dense_local_engine_gap_bends_after") or 0),
            "dense_local_unresolved_engine_gaps": int(details.get("dense_local_unresolved_engine_gaps") or 0),
            "dense_local_position_signal_recoveries": int(details.get("dense_local_position_signal_recoveries") or 0),
        }
        ranked_rows.append(row)
        scan_state_counter[scan_state] += 1
        scan_quality_counter[row["scan_quality_status"]] += 1
        part_counter[row["part_key"]] += engine_gap

    ranked_rows.sort(
        key=lambda item: (
            -item["engine_gap_position_unknown_bends"],
            -item["unknown_position_bends"],
            item["part_key"],
            item["scan_name"],
        )
    )

    slice_entries: List[Dict[str, Any]] = []
    seen_parts: set[str] = set()
    for row in ranked_rows:
        if unique_part_slice and row["part_key"] in seen_parts:
            continue
        seen_parts.add(row["part_key"])
        slice_entries.append(
            {
                "part_key": row["part_key"],
                "scan_path": row["scan_path"],
                "expectation": {
                    "state": row["scan_state"],
                    "expected_total_bends": row["expected_total_bends"],
                    "expected_completed_bends": row["expected_completed_bends"],
                },
                "notes": [
                    "engine_gap_position_unknown",
                    f"engine_gap={row['engine_gap_position_unknown_bends']}",
                    f"scan_quality={row['scan_quality_status'].lower()}",
                ],
            }
        )
        if len(slice_entries) >= top_n:
            break

    return {
        "summary_path": str(summary_path),
        "full_only": full_only,
        "top_n": top_n,
        "unique_part_slice": unique_part_slice,
        "engine_gap_scans_analyzed": len(ranked_rows),
        "scan_state_breakdown": dict(scan_state_counter),
        "scan_quality_breakdown": dict(scan_quality_counter),
        "part_engine_gap_breakdown": dict(part_counter.most_common()),
        "top_engine_gap_scans": ranked_rows[:top_n],
        "slice_entries": slice_entries,
    }


def _slice_payload(analysis: Dict[str, Any]) -> Dict[str, Any]:
    name = "engine_gap_position_unknown_full_scans" if analysis["full_only"] else "engine_gap_position_unknown_all_scans"
    description = (
        "Frozen targeted slice for engine-owned position-unknown blockers, "
        "ranked from the latest corpus blocker baseline."
    )
    return {
        "name": name,
        "description": description,
        "entries": analysis["slice_entries"],
    }


def _markdown_report(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Engine Gap Blocker Analysis",
        "",
        f"- Summary: `{analysis['summary_path']}`",
        f"- Full scans only: `{analysis['full_only']}`",
        f"- Unique-part slice: `{analysis['unique_part_slice']}`",
        f"- Engine-gap scans analyzed: `{analysis['engine_gap_scans_analyzed']}`",
        "",
        "## Breakdown",
        "",
        f"- Scan state: `{analysis['scan_state_breakdown']}`",
        f"- Scan quality: `{analysis['scan_quality_breakdown']}`",
        f"- Part engine-gap totals: `{analysis['part_engine_gap_breakdown']}`",
        "",
        "## Top Scans",
        "",
    ]
    for row in analysis["top_engine_gap_scans"]:
        lines.append(
            f"- `{row['part_key']} / {row['scan_name']}`: "
            f"engine_gap=`{row['engine_gap_position_unknown_bends']}`, "
            f"unknown_position=`{row['unknown_position_bends']}`, "
            f"position_eligible=`{row['position_eligible_bends']}`, "
            f"dense_local=`{row['dense_local_refinement_decision']}` "
            f"(before `{row['dense_local_engine_gap_bends_before']}` -> after `{row['dense_local_engine_gap_bends_after']}`)"
        )
    return "\n".join(lines) + "\n"


def _write_text(path_str: str, contents: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze engine-owned position-unknown blockers and build a targeted slice.")
    parser.add_argument("summary")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--include-partials", action="store_true")
    parser.add_argument("--allow-duplicate-parts", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument("--output-slice-json", default=None)
    args = parser.parse_args()

    analysis = analyze_engine_gap_blockers(
        Path(args.summary),
        full_only=not args.include_partials,
        top_n=args.top_n,
        unique_part_slice=not args.allow_duplicate_parts,
    )
    if args.output_json:
        _write_text(args.output_json, json.dumps(analysis, indent=2, ensure_ascii=False))
    if args.output_markdown:
        _write_text(args.output_markdown, _markdown_report(analysis))
    if args.output_slice_json:
        _write_text(
            args.output_slice_json,
            json.dumps(_slice_payload(analysis), indent=2, ensure_ascii=False),
        )
    if not args.output_json and not args.output_markdown and not args.output_slice_json:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
