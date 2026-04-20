#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_payload_path(summary_path: Path, part_key: str, scan_name: str) -> Path:
    safe = f"{scan_name.replace(' ', '_')}--"
    part_dir = summary_path.parent / part_key
    matches = sorted(part_dir.glob(f"{safe}*.json"))
    if not matches:
        raise FileNotFoundError(f"Could not find case payload for {part_key} / {scan_name}")
    return matches[0]


def _iter_case_payloads(summary_path: Path, summary: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for result in summary.get("results") or []:
        if result.get("matches") is not None:
            yield result
            continue
        yield _load_json(_case_payload_path(summary_path, result["part_key"], result["scan_name"]))


def analyze_metrology_holds(summary_path: Path | str) -> Dict[str, Any]:
    summary_path = Path(summary_path)
    summary = _load_json(summary_path)
    rows: List[Dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    method_counter: Counter[str] = Counter()
    position_ready_counter: Counter[str] = Counter()
    bend_form_counter: Counter[str] = Counter()
    driver_counter: Counter[str] = Counter()
    near_threshold = 0

    for payload in _iter_case_payloads(summary_path, summary):
        for match in payload.get("matches") or []:
            if str(match.get("metrology_state") or "") != "OUT_OF_TOL":
                continue
            if str(match.get("correspondence_state") or "") != "CONFIDENT":
                continue
            angle_dev = float(match.get("angle_deviation") or 0.0)
            radius_dev = float(match.get("radius_deviation") or 0.0)
            tolerance_angle = float(match.get("tolerance_angle") or 0.0)
            tolerance_radius = float(match.get("tolerance_radius") or 0.0)
            arc_dev = match.get("arc_length_deviation_mm")
            arc_dev = float(arc_dev) if arc_dev is not None else None
            bend_form = str(match.get("bend_form") or "UNKNOWN")
            angle_fail = abs(angle_dev) > tolerance_angle
            radius_fail = abs(radius_dev) > tolerance_radius
            arc_fail = arc_dev is not None and abs(arc_dev) > tolerance_radius
            driver_parts: List[str] = []
            if angle_fail:
                driver_parts.append("angle")
            if radius_fail:
                driver_parts.append("radius")
            if arc_fail:
                driver_parts.append("arc")
            driver_label = "+".join(driver_parts) if driver_parts else "status_only"
            row = {
                "part_key": payload["part_key"],
                "scan_name": payload["scan_name"],
                "bend_id": match.get("bend_id"),
                "bend_form": bend_form,
                "assignment_source": match.get("assignment_source"),
                "measurement_method": match.get("measurement_method"),
                "position_claim_eligible": bool(match.get("position_claim_eligible")),
                "metrology_claim_eligible": bool(match.get("metrology_claim_eligible")),
                "angle_deviation_deg": round(angle_dev, 3),
                "radius_deviation_mm": round(radius_dev, 3),
                "arc_length_deviation_mm": round(arc_dev, 3) if arc_dev is not None else None,
                "tolerance_angle_deg": tolerance_angle,
                "tolerance_radius_mm": tolerance_radius,
                "failure_driver": driver_label,
                "target_angle_deg": match.get("target_angle"),
                "measured_angle_deg": match.get("measured_angle"),
                "claim_gate_reasons": list(match.get("claim_gate_reasons") or []),
            }
            rows.append(row)
            source_counter[str(row["assignment_source"] or "UNKNOWN")] += 1
            method_counter[str(row["measurement_method"] or "UNKNOWN")] += 1
            position_ready_counter["position_ready" if row["position_claim_eligible"] else "position_unknown"] += 1
            bend_form_counter[bend_form] += 1
            driver_counter[f"{bend_form}:{driver_label}"] += 1
            if abs(angle_dev) <= 1.5:
                near_threshold += 1

    rows.sort(key=lambda item: (abs(float(item["angle_deviation_deg"])), item["part_key"], item["scan_name"], item["bend_id"]))
    return {
        "summary_path": str(summary_path),
        "out_of_tol_confident_bends": len(rows),
        "near_threshold_out_of_tol_bends": near_threshold,
        "assignment_source_breakdown": dict(source_counter),
        "measurement_method_breakdown": dict(method_counter),
        "position_readiness_breakdown": dict(position_ready_counter),
        "bend_form_breakdown": dict(bend_form_counter),
        "failure_driver_breakdown": dict(driver_counter),
        "closest_out_of_tol_bends": rows[:25],
    }


def _markdown_report(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Metrology Hold Analysis",
        "",
        f"- Summary: `{analysis['summary_path']}`",
        f"- Confident out-of-tolerance bends: `{analysis['out_of_tol_confident_bends']}`",
        f"- Near-threshold out-of-tolerance bends (<= 1.5 deg): `{analysis['near_threshold_out_of_tol_bends']}`",
        "",
        "## Breakdown",
        "",
        f"- Assignment source: `{analysis['assignment_source_breakdown']}`",
        f"- Measurement method: `{analysis['measurement_method_breakdown']}`",
        f"- Position readiness: `{analysis['position_readiness_breakdown']}`",
        f"- Bend form: `{analysis['bend_form_breakdown']}`",
        f"- Failure driver: `{analysis['failure_driver_breakdown']}`",
        "",
        "## Closest Out-Of-Tol Bends",
        "",
    ]
    for row in analysis["closest_out_of_tol_bends"]:
        lines.append(
            f"- `{row['part_key']} / {row['scan_name']} / {row['bend_id']}`: "
            f"`{row['angle_deviation_deg']:+.3f} deg`, "
            f"`{row['radius_deviation_mm']:+.3f} mm`, "
            f"`{row['assignment_source']}` / `{row['measurement_method']}`, "
            f"driver=`{row['failure_driver']}`, "
            f"position_ready=`{row['position_claim_eligible']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze strict metrology hold drivers from a corpus evaluation summary.")
    parser.add_argument("summary")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    analysis = analyze_metrology_holds(Path(args.summary))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_markdown:
        Path(args.output_markdown).write_text(_markdown_report(analysis), encoding="utf-8")
    if not args.output_json and not args.output_markdown:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
