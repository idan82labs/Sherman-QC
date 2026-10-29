#!/usr/bin/env python3
"""Run source-only visibility filtering and late CAD/scan fusion."""

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

from research.f1_cad_visibility_event_fusion_v1 import (
    VisibilityGates,
    contract_payload,
    filter_cad_events,
    late_fuse,
)
from scripts.compile_cad_bend_template_v1 import _atomic


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_rows(payload: dict) -> list[dict]:
    return [dict(row) for row in payload.get("owned_bend_regions") or []]


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
    token_by_sha = {
        str(row["source_scan_sha256"]): str(row["case_token"])
        for row in pool["cases"]
    }
    entries = slice_payload["entries"]
    if len(entries) != len(parent_summary["results"]):
        raise ValueError("slice and parent summary case counts differ")
    gates = VisibilityGates()
    started = time.perf_counter()
    records = []
    for ordinal, (entry, parent_row) in enumerate(zip(entries, parent_summary["results"]), 1):
        scan_path = Path(parent_row["scan_path"]).resolve()
        if scan_path.name != Path(entry["scan_path"]).name:
            raise ValueError(f"case {ordinal}: slice and parent scan mismatch")
        case_id = str(entry["case_token"])
        pool_token = token_by_sha.get(str(entry["scan_sha256"]))
        a0_path = (
            args.a0_root / pool_token / "prediction_owned_regions.json"
            if pool_token is not None
            else None
        )
        a0_payload = (
            json.loads(a0_path.read_text(encoding="utf-8"))
            if a0_path is not None
            else {"owned_bend_regions": []}
        )
        parent_path = args.parent_cases_root / f"C{ordinal:03d}" / "semantic_result.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        cad_visible, visibility_audit = filter_cad_events(parent, gates)
        fused, ownership_audit = late_fuse(_scan_rows(a0_payload), cad_visible, gates)
        payload = {
            "schema": "SHERMAN_QC_CAD_VISIBILITY_EVENT_FUSION_RESULT_V1",
            "status": "COMPLETE",
            "lane": "CAD_ASSISTED",
            "route": "CAD_VISIBILITY_EVENT_CONTRACTION_LATE_FUSION_V1",
            "case_id": case_id,
            "scan_sha256": str(entry["scan_sha256"]),
            "parent_status": parent.get("status"),
            "parent_sha256": _sha(parent_path),
            "a0_sha256": _sha(a0_path) if a0_path is not None else None,
            "a0_availability": "AVAILABLE" if a0_path is not None else "NOT_IN_FROZEN_A0_POOL",
            "contract": contract_payload(),
            "gold_or_expected_counts_used": False,
            "predictions": fused,
            "counts": {
                "scan_first_candidates": len(_scan_rows(a0_payload)),
                "parent_cad_predictions": len(parent.get("predictions") or []),
                "visible_cad_events": len(cad_visible),
                "fused_predictions": len(fused),
                "explicit_null_visibility": sum(row["decision"].startswith("NULL_") for row in visibility_audit),
                "explicit_null_ownership": sum(row["decision"].startswith("NULL_") for row in ownership_audit),
            },
            "visibility_audit": visibility_audit,
            "ownership_audit": ownership_audit,
        }
        target = args.output_root / case_id / "semantic_result.json"
        _atomic(target, payload)
        records.append({
            "case_id": case_id,
            "scan_path": str(scan_path),
            "semantic_result": str(target.resolve()),
            "semantic_result_sha256": _sha(target),
            **payload["counts"],
        })
    summary = {
        "schema": "SHERMAN_QC_CAD_VISIBILITY_EVENT_FUSION_BATCH_V1",
        "status": "COMPLETE",
        "experiment_id": "EXP_CAD_0008",
        "evidence_class": "DEVELOPMENT_CONSUMED_PHYSICAL",
        "case_count": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "contract": contract_payload(),
        "gold_or_expected_counts_used": False,
        "records": records,
    }
    _atomic(args.output_summary, summary)
    print(json.dumps({
        "status": summary["status"],
        "cases": len(records),
        "predictions": sum(row["fused_predictions"] for row in records),
        "visible_cad_events": sum(row["visible_cad_events"] for row in records),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
