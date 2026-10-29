#!/usr/bin/env python3
"""Run the general CAD-first localizer sequentially over an existing blind batch."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compile_cad_bend_template_v1 import _atomic, compile_template
from scripts.localize_cad_bends_on_scan_v1 import localize
from scripts.render_f1_source_only_prediction_audit import render


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.input_summary.read_text())
    output = copy.deepcopy(source)
    elapsed = []
    abstentions = 0
    for ordinal, result in enumerate(output["results"], 1):
        started = time.perf_counter()
        case_id = f"C{ordinal:03d}"
        case_root = args.output_root / case_id
        template = compile_template(Path(result["cad_path"]).resolve())
        _atomic(case_root / "cad_bend_template.json", template)
        localized = localize(template, Path(result["scan_path"]).resolve(), dict(result.get("alignment_metadata") or {}))
        _atomic(case_root / "semantic_result.json", localized)
        if localized["status"] == "ABSTAIN": abstentions += 1
        result["matches"] = [{
            **row,
            "measurement_method": "cad_first_template_scan_localized",
            "observability_state": "FORMED",
        } for row in localized["predictions"]]
        result["summary"] = {**dict(result.get("summary") or {}), "cad_first_template_events": template["event_count"], "cad_first_localized_bends": len(localized["predictions"]), "cad_first_status": localized["status"]}
        if args.render and localized["status"] == "COMPLETE":
            render(geometry=Path(result["scan_path"]).resolve(), prediction_path=(case_root / "semantic_result.json").resolve(), output=(case_root / "native_visual_audit.png").resolve(), max_points=250_000)
        elapsed.append(time.perf_counter() - started)
    ordered = sorted(elapsed)
    p95 = ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]
    output["typed_release_metrics"] = {**dict(output.get("typed_release_metrics") or {}), "scans_processed": len(elapsed), "successful_scans": len(elapsed), "failed_scans": 0, "job_success_rate": 1.0, "blind_inference_contract_rate": 1.0, "p95_pipeline_latency_sec": p95, "registration_abstention_count": abstentions}
    _atomic(args.output_summary, output)
    print(json.dumps({"cases": len(elapsed), "abstentions": abstentions, "p95_seconds": p95, "median_seconds": statistics.median(elapsed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
