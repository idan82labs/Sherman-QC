#!/usr/bin/env python3
"""Run joint finite-span localization and contextual CAD/scan assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_cad_contextual_assignment_v1 import AssignmentConfig, contract_payload, fuse
from research.f1_cad_visibility_event_fusion_v1 import VisibilityGates, filter_cad_events
from scripts.build_additional_line_label_review import _scan_points, line_scan_support
from scripts.compile_cad_bend_template_v1 import _atomic


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(row: dict) -> np.ndarray:
    endpoints = row.get("span_endpoints")
    if endpoints is None:
        endpoints = [row.get("detected_line_start", row.get("line_start")), row.get("detected_line_end", row.get("line_end"))]
    value = np.asarray(endpoints, dtype=np.float64)
    if value.shape != (2, 3) or not np.all(np.isfinite(value)):
        raise ValueError("candidate must provide finite 2x3 line")
    return value


def crop_candidates(points: np.ndarray, rows: list[dict], source: str) -> tuple[list[dict], list[dict]]:
    accepted = []
    audit = []
    for ordinal, row in enumerate(rows, 1):
        line = _line(row)
        support = line_scan_support(points, line[0].tolist(), line[1].tolist())
        event_id = str(row.get("bend_id") or f"{source}_{ordinal:04d}")
        if not support["frame_integrity_pass"]:
            audit.append({"bend_id": event_id, "decision": "NULL_NO_CONTIGUOUS_SUPPORT", "scan_support": support})
            continue
        accepted.append({
            **dict(row),
            "bend_id": f"{source}_{event_id}",
            "line_start": support["trimmed_line_start"],
            "line_end": support["trimmed_line_end"],
            "detected_line_start": support["trimmed_line_start"],
            "detected_line_end": support["trimmed_line_end"],
            "assignment_source": f"{source}_FINITE_CONTIGUOUS_SUPPORT",
            "scan_support": support,
        })
        audit.append({"bend_id": event_id, "decision": "FINITE_SPAN_ACCEPTED", "scan_support": support})
    return accepted, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--parent-cases-root", type=Path, required=True)
    parser.add_argument("--a0-pool-manifest", type=Path, required=True)
    parser.add_argument("--a0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    slice_payload = json.loads(args.slice.read_text(encoding="utf-8"))
    parent_summary = json.loads(args.parent_summary.read_text(encoding="utf-8"))
    pool = json.loads(args.a0_pool_manifest.read_text(encoding="utf-8"))
    token_by_sha = {str(row["source_scan_sha256"]): str(row["case_token"]) for row in pool["cases"]}
    entries = slice_payload["entries"]
    if len(entries) != len(parent_summary["results"]):
        raise ValueError("slice and parent summary case counts differ")
    visibility = VisibilityGates()
    assignment = AssignmentConfig()
    started = time.perf_counter()
    output_records = []
    for ordinal, (entry, parent_row) in enumerate(zip(entries, parent_summary["results"]), 1):
        case_id = str(entry["case_token"])
        scan_path = Path(parent_row["scan_path"]).resolve()
        if scan_path.name != Path(entry["scan_path"]).name:
            raise ValueError(f"case {case_id}: scan mismatch")
        points = _scan_points(scan_path)
        pool_token = token_by_sha.get(str(entry["scan_sha256"]))
        a0_path = args.a0_root / pool_token / "prediction_owned_regions.json" if pool_token is not None else None
        a0_payload = json.loads(a0_path.read_text(encoding="utf-8")) if a0_path is not None else {"owned_bend_regions": []}
        parent_path = args.parent_cases_root / f"C{ordinal:03d}" / "semantic_result.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        cad_visible, visibility_audit = filter_cad_events(parent, visibility)
        scan_finite, scan_crop_audit = crop_candidates(points, [dict(row) for row in a0_payload.get("owned_bend_regions") or []], "SCAN")
        cad_finite, cad_crop_audit = crop_candidates(points, cad_visible, "CAD")
        predictions, assignment_audit = fuse(cad_finite, scan_finite, assignment)
        payload = {
            "schema": "SHERMAN_QC_CAD_CONTEXTUAL_ASSIGNMENT_RESULT_V1",
            "status": "COMPLETE",
            "experiment_id": "EXP_CAD_0009",
            "lane": "CAD_ASSISTED",
            "route": "JOINT_FINITE_SPAN_CONTEXTUAL_CAD_SCAN_ASSIGNMENT_V1",
            "case_id": case_id,
            "scan_sha256": str(entry["scan_sha256"]),
            "parent_sha256": _sha(parent_path),
            "a0_sha256": _sha(a0_path) if a0_path is not None else None,
            "gold_or_expected_counts_used": False,
            "contract": contract_payload(),
            "predictions": predictions,
            "counts": {
                "raw_scan_candidates": len(a0_payload.get("owned_bend_regions") or []),
                "finite_scan_candidates": len(scan_finite),
                "visible_cad_events": len(cad_visible),
                "finite_cad_events": len(cad_finite),
                "assigned_pairs": len(assignment_audit["matches"]),
                "final_predictions": len(predictions),
            },
            "visibility_audit": visibility_audit,
            "scan_crop_audit": scan_crop_audit,
            "cad_crop_audit": cad_crop_audit,
            "assignment_audit": assignment_audit,
        }
        target = args.output_root / case_id / "semantic_result.json"
        _atomic(target, payload)
        output_records.append({
            "case_id": case_id,
            "scan_path": str(scan_path),
            "semantic_result": str(target.resolve()),
            "semantic_result_sha256": _sha(target),
            **payload["counts"],
        })
    summary = {
        "schema": "SHERMAN_QC_CAD_CONTEXTUAL_ASSIGNMENT_BATCH_V1",
        "status": "COMPLETE",
        "experiment_id": "EXP_CAD_0009",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
        "case_count": len(output_records),
        "elapsed_seconds": time.perf_counter() - started,
        "gold_or_expected_counts_used": False,
        "contract": contract_payload(),
        "records": output_records,
    }
    _atomic(args.output_summary, summary)
    print(json.dumps({
        "status": "COMPLETE",
        "cases": len(output_records),
        "assigned_pairs": sum(row["assigned_pairs"] for row in output_records),
        "predictions": sum(row["final_predictions"] for row in output_records),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
