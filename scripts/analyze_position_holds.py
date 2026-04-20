#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_position_holds(summary_path: Path | str) -> Dict[str, Any]:
    summary_path = Path(summary_path)
    summary = _load_json(summary_path)
    rows: List[Dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    method_counter: Counter[str] = Counter()
    form_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()

    for result in summary.get("results") or []:
        for match in result.get("matches") or []:
            if not match.get("countable_in_regression"):
                continue
            if str(match.get("completion_state") or "") != "FORMED":
                continue
            if str(match.get("correspondence_state") or "") != "CONFIDENT":
                continue
            if str(match.get("positional_state") or "") != "UNKNOWN_POSITION":
                continue
            reasons = list(match.get("claim_gate_reasons") or [])
            row = {
                "part_key": result["part_key"],
                "scan_name": result["scan_name"],
                "bend_id": match.get("bend_id"),
                "bend_form": match.get("bend_form"),
                "assignment_source": match.get("assignment_source"),
                "measurement_method": match.get("measurement_method"),
                "metrology_state": match.get("metrology_state"),
                "angle_deviation_deg": round(float(match.get("angle_deviation") or 0.0), 3),
                "radius_deviation_mm": round(float(match.get("radius_deviation") or 0.0), 3),
                "claim_gate_reasons": reasons,
                "position_evidence_reason": (match.get("position_evidence") or {}).get("reason"),
            }
            rows.append(row)
            source_counter[str(row["assignment_source"] or "UNKNOWN")] += 1
            method_counter[str(row["measurement_method"] or "UNKNOWN")] += 1
            form_counter[str(row["bend_form"] or "UNKNOWN")] += 1
            for reason in reasons:
                reason_counter[str(reason)] += 1

    rows.sort(key=lambda item: (item["part_key"], item["scan_name"], item["bend_id"]))
    return {
        "summary_path": str(summary_path),
        "formed_confident_unknown_position_bends": len(rows),
        "assignment_source_breakdown": dict(source_counter),
        "measurement_method_breakdown": dict(method_counter),
        "bend_form_breakdown": dict(form_counter),
        "claim_gate_reason_breakdown": dict(reason_counter),
        "sample_bends": rows[:40],
    }


def _markdown_report(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Position Hold Analysis",
        "",
        f"- Summary: `{analysis['summary_path']}`",
        f"- Formed + confident + unknown-position bends: `{analysis['formed_confident_unknown_position_bends']}`",
        "",
        "## Breakdown",
        "",
        f"- Assignment source: `{analysis['assignment_source_breakdown']}`",
        f"- Measurement method: `{analysis['measurement_method_breakdown']}`",
        f"- Bend form: `{analysis['bend_form_breakdown']}`",
        f"- Claim gate reasons: `{analysis['claim_gate_reason_breakdown']}`",
        "",
        "## Sample Bends",
        "",
    ]
    for row in analysis["sample_bends"]:
        lines.append(
            f"- `{row['part_key']} / {row['scan_name']} / {row['bend_id']}`: "
            f"`{row['bend_form']}` / `{row['assignment_source']}` / `{row['measurement_method']}`, "
            f"metrology=`{row['metrology_state']}`, reasons=`{row['claim_gate_reasons']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze formed/confident bends that still abstain on position.")
    parser.add_argument("summary")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    analysis = analyze_position_holds(Path(args.summary))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_markdown:
        Path(args.output_markdown).write_text(_markdown_report(analysis), encoding="utf-8")
    if not args.output_json and not args.output_markdown:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
