#!/usr/bin/env python3
"""Qualify deterministic CAD reference sampling through the frozen registration path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.qc_engine import AlignmentMetadata, ScanQCEngine  # noqa: E402
from research.f1_cad_registration_execution_audit_v1 import AuditConfig, audit_transform_pair  # noqa: E402
from research.f1_deterministic_reference_surface_v1 import (  # noqa: E402
    array_sha256,
    contract_payload,
    sample_surface,
    sorted_reference_points,
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


def peak_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 ** 3)


class DeterministicCaptureEngine(ScanQCEngine):
    audit_global_transform: np.ndarray | None = None
    audit_datum_transform: np.ndarray | None = None
    audit_datum_selected: bool | None = None
    audit_datum_stats: dict | None = None
    deterministic_reference_points_sha256: str | None = None
    deterministic_reference_normals_sha256: str | None = None

    def _prepare_reference(self):
        target = max(100_000, len(self.scan_pcd.points))
        point_cloud = self.o3d.geometry.PointCloud()
        if self.reference_mesh is not None and len(self.reference_mesh.triangles) > 0:
            vertices = np.asarray(self.reference_mesh.vertices, dtype=np.float64)
            triangles = np.asarray(self.reference_mesh.triangles, dtype=np.int64)
            points, normals = sample_surface(vertices, triangles, target)
        elif self.reference_pcd is not None and len(self.reference_pcd.points) > 0:
            base = sorted_reference_points(np.asarray(self.reference_pcd.points, dtype=np.float64))
            indices = np.arange(target, dtype=np.int64) % len(base)
            points = np.ascontiguousarray(base[indices], dtype=np.float64)
            normals = np.zeros_like(points)
        else:
            raise ValueError("reference geometry unavailable")
        point_cloud.points = self.o3d.utility.Vector3dVector(points)
        point_cloud.normals = self.o3d.utility.Vector3dVector(normals)
        self.deterministic_reference_points_sha256 = array_sha256(points)
        self.deterministic_reference_normals_sha256 = array_sha256(normals)
        return point_cloud

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
    for path, expected in ((scan, row["scan_sha256"]), (cad, row["cad_sha256"]), (template_path, row["cad_template_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"source hash mismatch: {path}")
    template = json.loads(template_path.read_text())
    engine = DeterministicCaptureEngine()
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
    audit = None
    if captured:
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
        "schema": "EXP_CAD_0012_DETERMINISTIC_REGISTRATION_CASE_V1",
        "experiment_id": "EXP_CAD_0012",
        "case_token": row["case_token"],
        "scan_sha256": row["scan_sha256"],
        "cad_sha256": row["cad_sha256"],
        "cad_template_sha256": row["cad_template_sha256"],
        "deterministic_reference_points_sha256": engine.deterministic_reference_points_sha256,
        "deterministic_reference_normals_sha256": engine.deterministic_reference_normals_sha256,
        "processed_scan_point_count": int(len(scan_working.points)),
        "reference_point_count": int(len(ref_centered.points)),
        "voxel_size_mm": voxel,
        "datum_candidate_captured": captured,
        "datum_selected": engine.audit_datum_selected,
        "global_transform": engine.audit_global_transform.tolist() if captured else None,
        "datum_transform": engine.audit_datum_transform.tolist() if captured else None,
        "datum_selection_stats": engine.audit_datum_stats,
        "material_local_reversal_event_ids": (audit or {}).get("material_local_reversal_event_ids", []),
        "global_metrics": (audit or {}).get("global_metrics"),
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("gold_or_expected_counts_present") is not False:
        raise ValueError("truth-denying manifest required")
    started = perf_counter()
    records = []
    for index, row in enumerate(manifest["records"], 1):
        print(f"[{index}/{len(manifest['records'])}] {row['case_token']}", flush=True)
        payload = execute_case(row, args.seed)
        output = args.output_root / "cases" / row["case_token"] / "semantic_result.json"
        atomic(output, payload)
        records.append({
            "case_token": row["case_token"],
            "semantic_result_sha256": sha(output),
            "datum_selected": payload["datum_selected"],
            "datum_candidate_captured": payload["datum_candidate_captured"],
        })
    summary = {
        "schema": "EXP_CAD_0012_DETERMINISTIC_REGISTRATION_SUMMARY_V1",
        "experiment_id": "EXP_CAD_0012",
        "contract": contract_payload(),
        "manifest_sha256": sha(args.manifest),
        "case_count": len(records),
        "complete_case_count": sum(row["datum_candidate_captured"] for row in records),
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_gib": peak_rss_gib(),
        "records": records,
        "gold_or_expected_counts_used": False,
        "generalization_evidence": False,
    }
    atomic(args.output_root / "SOURCE_ONLY_SUMMARY.json", summary)
    print(json.dumps({key: summary[key] for key in ("case_count", "complete_case_count", "elapsed_seconds", "peak_rss_gib")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

