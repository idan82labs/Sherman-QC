#!/usr/bin/env python3
"""Run EXP_CAD_0010 from its truth-denying case-binding manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_cad_contextual_assignment_v1 import AssignmentConfig
from research.f1_cad_ontology_grouped_capacity_fusion_v1 import (
    contract_payload,
    contract_physical_groups,
    fuse_capacity_preserving,
    ontology_filter_parent,
)
from research.f1_cad_visibility_event_fusion_v1 import VisibilityGates, filter_cad_events
from scripts.build_additional_line_label_review import _scan_points
from scripts.compile_cad_bend_template_v1 import _atomic
from scripts.run_cad_contextual_assignment_v1 import crop_candidates


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-manifest", type=Path, required=True)
    parser.add_argument("--a0-pool-manifest", type=Path, required=True)
    parser.add_argument("--a0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding_manifest.read_text(encoding="utf-8"))
    if binding.get("status") != "LOCKED" or binding.get("case_count") != 13:
        raise ValueError("locked 13-case binding manifest required")
    pool = json.loads(args.a0_pool_manifest.read_text(encoding="utf-8"))
    token_by_sha = {str(row["source_scan_sha256"]): str(row["case_token"]) for row in pool["cases"]}
    visibility = VisibilityGates()
    assignment = AssignmentConfig()
    started = time.perf_counter()
    output_records = []
    for record in binding["records"]:
        case_id = str(record["case_token"])
        scan_sha = str(record["scan_sha256"])
        scan_path = Path(record["scan_path"]).resolve()
        if _sha(scan_path) != scan_sha:
            raise ValueError(f"{case_id}: scan SHA mismatch")
        parent_path = Path(record["parent_semantic_path"]).resolve()
        if _sha(parent_path) != str(record["parent_semantic_sha256"]):
            raise ValueError(f"{case_id}: parent semantic SHA mismatch")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if str(parent["scan_sha256"]) != scan_sha or str(parent["step_sha256"]) != str(record["step_sha256"]):
            raise ValueError(f"{case_id}: bound scan/STEP identity mismatch")
        if str(parent["status"]) != str(record["parent_status"]):
            raise ValueError(f"{case_id}: parent status mismatch")
        filtered_parent, ontology_audit = ontology_filter_parent(parent, record["event_bindings"])
        points = _scan_points(scan_path)
        pool_token = token_by_sha.get(scan_sha)
        a0_path = args.a0_root / pool_token / "prediction_owned_regions.json" if pool_token is not None else None
        a0_payload = json.loads(a0_path.read_text(encoding="utf-8")) if a0_path is not None else {"owned_bend_regions": []}
        scan_finite, scan_crop_audit = crop_candidates(
            points, [dict(row) for row in a0_payload.get("owned_bend_regions") or []], "SCAN"
        )
        cad_visible, visibility_audit = filter_cad_events(filtered_parent, visibility)
        cad_finite, cad_crop_audit = crop_candidates(points, cad_visible, "CAD")
        group_rows, group_audit = contract_physical_groups(cad_finite, record["event_bindings"])
        predictions, assignment_audit = fuse_capacity_preserving(group_rows, scan_finite, assignment)
        if assignment_audit["null_one_to_one_conflict_count"] != 0:
            raise ValueError("NULL_ONE_TO_ONE_CONFLICT is forbidden")
        if any(
            row.get("physical_event_group_id") is not None
            and str(row["physical_event_group_id"]).startswith("C005:")
            for row in predictions
        ):
            raise ValueError("C005 CAD emission is forbidden")
        payload = {
            "schema": "SHERMAN_QC_CAD_ONTOLOGY_GROUPED_CAPACITY_RESULT_V1",
            "status": "COMPLETE",
            "experiment_id": "EXP_CAD_0010",
            "lane": "CAD_ASSISTED",
            "route": "V2_ONTOLOGY_GROUPED_CAPACITY_FUSION_V1",
            "case_id": case_id,
            "scan_sha256": scan_sha,
            "binding_manifest_sha256": _sha(args.binding_manifest),
            "parent_sha256": _sha(parent_path),
            "a0_sha256": _sha(a0_path) if a0_path is not None else None,
            "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
            "generalization_evidence": False,
            "promotion_eligible": False,
            "ai_adjudicated_case_mapping_used": True,
            "gold_or_expected_counts_used": False,
            "contract": contract_payload(),
            "predictions": predictions,
            "counts": {
                "raw_scan_candidates": len(a0_payload.get("owned_bend_regions") or []),
                "finite_scan_candidates": len(scan_finite),
                "bound_parent_events": len(record["event_bindings"]),
                "qc_bound_events": sum(row["status"] == "QC_BEND" for row in record["event_bindings"]),
                "not_qc_bound_events": sum(row["status"] == "NOT_QC_BEND" for row in record["event_bindings"]),
                "visible_cad_primitives": len(cad_visible),
                "finite_cad_primitives": len(cad_finite),
                "finite_physical_groups": len(group_rows),
                "assigned_groups": len(assignment_audit["matches"]),
                "final_predictions": len(predictions),
                "null_one_to_one_conflicts": 0,
            },
            "ontology_audit": ontology_audit,
            "visibility_audit": visibility_audit,
            "scan_crop_audit": scan_crop_audit,
            "cad_crop_audit": cad_crop_audit,
            "group_audit": group_audit,
            "assignment_audit": assignment_audit,
        }
        target = args.output_root / case_id / "semantic_result.json"
        _atomic(target, payload)
        output_records.append({
            "case_id": case_id,
            "scan_path": str(scan_path),
            "scan_sha256": scan_sha,
            "semantic_result": str(target.resolve()),
            "semantic_result_sha256": _sha(target),
            **payload["counts"],
        })
    summary = {
        "schema": "SHERMAN_QC_CAD_ONTOLOGY_GROUPED_CAPACITY_BATCH_V1",
        "status": "COMPLETE",
        "experiment_id": "EXP_CAD_0010",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
        "generalization_evidence": False,
        "promotion_eligible": False,
        "ai_adjudicated_case_mapping_used": True,
        "gold_or_expected_counts_used": False,
        "binding_manifest_sha256": _sha(args.binding_manifest),
        "case_count": len(output_records),
        "elapsed_seconds": time.perf_counter() - started,
        "contract": contract_payload(),
        "records": output_records,
    }
    _atomic(args.output_summary, summary)
    print(json.dumps({
        "status": "COMPLETE",
        "cases": len(output_records),
        "finite_cad_primitives": sum(row["finite_cad_primitives"] for row in output_records),
        "finite_physical_groups": sum(row["finite_physical_groups"] for row in output_records),
        "assigned_groups": sum(row["assigned_groups"] for row in output_records),
        "predictions": sum(row["final_predictions"] for row in output_records),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
