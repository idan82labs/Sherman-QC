#!/usr/bin/env python3
"""Execute the source-only fixed CAD-event registration audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.qc_engine import AlignmentMetadata, ScanQCEngine  # noqa: E402
from research.f1_cad_registration_execution_audit_v1 import (  # noqa: E402
    AuditConfig,
    audit_transform_pair,
    contract_payload,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CaptureEngine(ScanQCEngine):
    audit_global_transform: np.ndarray | None = None
    audit_datum_transform: np.ndarray | None = None
    audit_datum_selected: bool | None = None
    audit_datum_stats: dict | None = None

    def _evaluate_datum_vs_global(self, scan_working, primary, global_transform, datum_transform, tolerance=1.0):
        use_datum, stats = super()._evaluate_datum_vs_global(
            scan_working, primary, global_transform, datum_transform, tolerance
        )
        self.audit_global_transform = np.asarray(global_transform, dtype=np.float64).copy()
        self.audit_datum_transform = np.asarray(datum_transform, dtype=np.float64).copy()
        self.audit_datum_selected = bool(use_datum)
        self.audit_datum_stats = dict(stats)
        return use_datum, stats


def execute_case(row: dict, seed: int) -> dict:
    scan = Path(row["scan_path"])
    cad = Path(row["cad_path"])
    template_path = Path(row["cad_template_path"])
    for path, expected in (
        (scan, row["scan_sha256"]),
        (cad, row["cad_sha256"]),
        (template_path, row["cad_template_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    template = json.loads(template_path.read_text())
    engine = CaptureEngine()
    engine.load_reference(str(cad))
    engine.load_scan(str(scan))
    raw_count = int(len(engine.scan_pcd.points))
    voxel = 2.0 if raw_count >= 650000 else 1.5 if raw_count >= 450000 else 1.0
    engine.preprocess(voxel_size=voxel)
    engine._set_alignment_random_seed(seed)
    engine.alignment_metadata = AlignmentMetadata()
    ref_pcd = engine._prepare_reference()
    scan_working, ref_centered, _scan_center, ref_center = engine._center_clouds(ref_pcd)
    engine._pca_pre_align(scan_working, ref_centered)
    engine._auto_scale(scan_working, ref_centered, True)
    best_flip, best_icp_init = engine._search_flip_hypotheses(scan_working, ref_centered)
    engine._apply_flip(scan_working, best_flip)
    reg_fine = engine._fine_icp(scan_working, ref_centered, best_icp_init)
    engine._ransac_datum_alignment(scan_working, ref_centered, reg_fine, 1.0)
    captured = bool(engine.audit_global_transform is not None and engine.audit_datum_transform is not None)
    base = {
        "schema": "EXP_CAD_0011_CASE_RESULT_V1",
        "experiment_id": "EXP_CAD_0011",
        "case_token": row["case_token"],
        "scan_sha256": row["scan_sha256"],
        "cad_sha256": row["cad_sha256"],
        "cad_template_sha256": row["cad_template_sha256"],
        "parent_alignment_source": row.get("parent_alignment_source"),
        "parent_alignment_abstained": bool(row.get("parent_alignment_abstained")),
        "rerun_datum_candidate_captured": captured,
        "rerun_datum_selected": engine.audit_datum_selected,
        "processed_scan_point_count": int(len(scan_working.points)),
        "reference_point_count": int(len(ref_centered.points)),
        "voxel_size_mm": voxel,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }
    if not captured:
        return {**base, "status": "NO_DATUM_CANDIDATE", "audit": None}
    audit = audit_transform_pair(
        processed_scan_points=np.asarray(scan_working.points, dtype=np.float64),
        reference_points=np.asarray(ref_centered.points, dtype=np.float64),
        reference_center=ref_center,
        global_transform=engine.audit_global_transform,
        datum_transform=engine.audit_datum_transform,
        events=template["events"],
        config=AuditConfig(),
    )
    return {
        **base,
        "status": "COMPLETE",
        "datum_selection_stats": engine.audit_datum_stats,
        "audit": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("gold_or_expected_counts_present") is not False:
        raise ValueError("truth-denying source manifest required")
    started = perf_counter()
    records = []
    for index, row in enumerate(manifest["records"], 1):
        print(f"[{index}/{len(manifest['records'])}] {row['case_token']}", flush=True)
        result = execute_case(copy.deepcopy(row), args.seed)
        case_path = args.output_root / "cases" / row["case_token"] / "semantic_result.json"
        atomic(case_path, result)
        records.append({
            "case_token": row["case_token"],
            "semantic_result_sha256": sha(case_path),
            "status": result["status"],
            "parent_alignment_source": result["parent_alignment_source"],
            "rerun_datum_selected": result["rerun_datum_selected"],
            "material_local_reversal_count": int((result.get("audit") or {}).get("material_local_reversal_count") or 0),
        })
    summary = {
        "schema": "EXP_CAD_0011_SOURCE_ONLY_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0011",
        "contract": contract_payload(),
        "manifest_sha256": sha(args.manifest),
        "case_count": len(records),
        "complete_case_count": sum(row["status"] == "COMPLETE" for row in records),
        "selected_datum_case_count": sum(row["rerun_datum_selected"] is True for row in records),
        "reversal_case_count": sum(row["material_local_reversal_count"] > 0 for row in records),
        "elapsed_seconds": perf_counter() - started,
        "records": records,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }
    atomic(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({key: summary[key] for key in ("case_count", "complete_case_count", "selected_datum_case_count", "reversal_case_count", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

