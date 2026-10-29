#!/usr/bin/env python3
"""Apply the frozen July scan-frame support gate to CAD-guided lines.

The gate is label-free and predates this experiment: 201 samples per axis,
nearest scan support within 3 mm, at least 10% support and an 8 mm longest
contiguous supported run. Accepted axes are cropped to that run; unsupported
or mixed-frame axes are removed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_additional_line_label_review import _scan_points, line_scan_support


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.input_summary.read_text(encoding="utf-8"))
    output = copy.deepcopy(summary)
    audit_cases = []
    accepted_total = rejected_total = 0
    for result in output["results"]:
        points = _scan_points(Path(result["scan_path"]))
        records = []
        for row in result.get("matches") or []:
            start = row.get("detected_line_start")
            end = row.get("detected_line_end")
            if start is None or end is None:
                continue
            support = line_scan_support(points, start, end)
            accepted = bool(support["frame_integrity_pass"])
            records.append(
                {
                    "bend_id": str(row.get("bend_id") or ""),
                    "accepted": accepted,
                    "source_line_start": start,
                    "source_line_end": end,
                    "scan_support": support,
                }
            )
            if accepted:
                row["detected_line_start"] = support["trimmed_line_start"]
                row["detected_line_end"] = support["trimmed_line_end"]
                row["cad_guided_scan_frame_gate"] = "accepted_and_cropped"
                accepted_total += 1
            else:
                row["detected_line_start"] = None
                row["detected_line_end"] = None
                row["cad_guided_scan_frame_gate"] = "rejected"
                rejected_total += 1
        audit_cases.append(
            {
                "part_key": str(result.get("part_key") or ""),
                "scan_path": str(result["scan_path"]),
                "accepted": sum(row["accepted"] for row in records),
                "rejected": sum(not row["accepted"] for row in records),
                "records": records,
            }
        )
    audit = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_guided_scan_frame_gate_audit",
        "label_free": True,
        "predates_current_experiment": True,
        "constants": {
            "support_sample_count": 201,
            "support_distance_mm": 3.0,
            "minimum_supported_fraction": 0.10,
            "minimum_supported_span_mm": 8.0,
        },
        "accepted_total": accepted_total,
        "rejected_total": rejected_total,
        "cases": audit_cases,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    args.audit.write_text(json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output_summary": str(args.output_summary.resolve()), "accepted": accepted_total, "rejected": rejected_total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
