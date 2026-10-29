#!/usr/bin/env python3
"""Apply count-invariant observed-surface localization to EXP_CAD_0002."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_cad_observed_intersection_v1 import relocate_predictions
from scripts.build_additional_line_label_review import _scan_points
from scripts.localize_cad_bends_on_scan_v1 import _atomic


def _stable(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.parent_summary.read_text(encoding="utf-8"))
    output_summary = copy.deepcopy(summary)
    records = []
    started = time.perf_counter()
    for ordinal, summary_row in enumerate(output_summary["results"], 1):
        case_id = f"C{ordinal:03d}"
        parent = json.loads((args.parent_root / "cases" / case_id / "semantic_result.json").read_text(encoding="utf-8"))
        points = _scan_points(Path(summary_row["scan_path"]).resolve(), max_points=200_000)
        predictions, evidence = relocate_predictions(points, list(parent["predictions"]))
        result = copy.deepcopy(parent)
        result.update({
            "schema": "SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V7",
            "route": "CAD_PRIOR_SCAN_OBSERVED_PLANE_INTERSECTION_V1",
            "predictions": predictions,
            "localized_bend_count": len(predictions),
            "observed_intersection_audit": {
                "parent_prediction_count": len(parent["predictions"]),
                "candidate_prediction_count": len(predictions),
                "relocated_count": sum(row["relocated"] for row in evidence),
                "abstained_relocation_count": sum(not row["relocated"] for row in evidence),
                "records": evidence,
            },
            "gold_or_expected_counts_used": False,
        })
        replay_predictions, replay_evidence = relocate_predictions(points, list(parent["predictions"]))
        replay = copy.deepcopy(result)
        replay["predictions"] = replay_predictions
        replay["observed_intersection_audit"]["records"] = replay_evidence
        if _stable(result) != _stable(replay):
            raise RuntimeError(f"{case_id}: replay mismatch")
        for stage, payload in (("primary", result), ("replay", replay)):
            _atomic(args.output_root / stage / case_id / "semantic_result.json", payload)
        summary_row["matches"] = [{**row, "measurement_method":"cad_prior_scan_observed_plane_intersection"} for row in predictions]
        records.append({
            "case_id": case_id,
            "parent_count": len(parent["predictions"]),
            "candidate_count": len(predictions),
            "relocated_count": result["observed_intersection_audit"]["relocated_count"],
            "abstained_relocation_count": result["observed_intersection_audit"]["abstained_relocation_count"],
            "replay_byte_identical": True,
        })
    if any(row["parent_count"] != row["candidate_count"] for row in records):
        raise RuntimeError("event-count invariant violated")
    _atomic(args.output_root / "summary.json", output_summary)
    manifest = {
        "schema":"SHERMAN_QC_EXP_CAD_0007_SOURCE_ONLY_EXECUTION_V1",
        "experiment_id":"EXP_CAD_0007",
        "status":"SOURCE_ONLY_COMPLETE",
        "case_count":len(records),
        "relocated_count":sum(row["relocated_count"] for row in records),
        "abstained_relocation_count":sum(row["abstained_relocation_count"] for row in records),
        "changed_case_ids":[row["case_id"] for row in records if row["relocated_count"]],
        "event_count_invariant":True,
        "semantic_replay_byte_identical":True,
        "gold_opened":False,
        "elapsed_seconds":time.perf_counter()-started,
        "cases":records,
    }
    _atomic(args.output_root / "source_only_manifest.json", manifest)
    print(json.dumps({key:manifest[key] for key in ("case_count","relocated_count","abstained_relocation_count","changed_case_ids","elapsed_seconds")},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
