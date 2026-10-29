#!/usr/bin/env python3
"""Apply source-only spatial fold verification to frozen physical15 V51."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_spatial_two_side_fold_v1 import filter_predictions
from scripts.build_additional_line_label_review import _scan_points
from scripts.localize_cad_bends_on_scan_v1 import _atomic


def _stable(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _one(case_id: str, source_root: Path) -> tuple[dict, dict]:
    case_root = source_root / "primary" / case_id
    parent = json.loads((case_root / "semantic_result.json").read_text(encoding="utf-8"))
    slice_payload = json.loads((case_root / "slice.json").read_text(encoding="utf-8"))
    scan = Path(slice_payload["entries"][0]["scan_path"]).resolve()
    points = _scan_points(scan, max_points=200_000)
    predictions, evidence = filter_predictions(points, list(parent["predictions"]))
    result = copy.deepcopy(parent)
    result.update({
        "profile": "EXP_SCAN_0001_SPATIAL_TWO_SIDE_FOLD_VERIFIER_V1",
        "parent_profile": parent.get("profile"),
        "predictions": predictions,
        "candidate_count": len(predictions),
        "spatial_fold_audit": {
            "parent_prediction_count": len(parent["predictions"]),
            "retained_prediction_count": len(predictions),
            "removed_prediction_count": len(parent["predictions"]) - len(predictions),
            "unresolved_prediction_count": sum(row["state"] == "UNRESOLVED" for row in evidence),
            "records": evidence,
        },
        "source_only": True,
        "cad_used": False,
        "gold_or_expected_counts_used": False,
    })
    return result, {"scan_sha256": hashlib.sha256(scan.read_bytes()).hexdigest(), "scan_path": str(scan)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    case_ids = sorted(path.name for path in (args.source_root / "primary").glob("D*"))
    records = []
    started = time.perf_counter()
    for case_id in case_ids:
        first, source = _one(case_id, args.source_root)
        second, _ = _one(case_id, args.source_root)
        identical = _stable(first) == _stable(second)
        if not identical:
            raise RuntimeError(f"{case_id}: semantic replay mismatch")
        for stage, payload in (("primary", first), ("replay", second)):
            _atomic(args.output_root / stage / case_id / "semantic_result.json", payload)
        removed = first["spatial_fold_audit"]["removed_prediction_count"]
        records.append({
            "case_id": case_id,
            "scan_sha256": source["scan_sha256"],
            "parent_prediction_count": first["spatial_fold_audit"]["parent_prediction_count"],
            "candidate_prediction_count": len(first["predictions"]),
            "removed_prediction_count": removed,
            "unresolved_prediction_count": first["spatial_fold_audit"]["unresolved_prediction_count"],
            "semantic_replay_byte_identical": identical,
        })
    manifest = {
        "schema": "SHERMAN_QC_EXP_SCAN_0001_SOURCE_ONLY_EXECUTION_V1",
        "experiment_id": "EXP_SCAN_0001",
        "status": "SOURCE_ONLY_COMPLETE",
        "case_count": len(records),
        "changed_case_ids": [row["case_id"] for row in records if row["removed_prediction_count"]],
        "removed_prediction_count": sum(row["removed_prediction_count"] for row in records),
        "unresolved_prediction_count": sum(row["unresolved_prediction_count"] for row in records),
        "semantic_replay_byte_identical": all(row["semantic_replay_byte_identical"] for row in records),
        "gold_opened": False,
        "expected_counts_opened": False,
        "cad_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "cases": records,
    }
    _atomic(args.output_root / "source_only_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("case_count", "changed_case_ids", "removed_prediction_count", "unresolved_prediction_count", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
