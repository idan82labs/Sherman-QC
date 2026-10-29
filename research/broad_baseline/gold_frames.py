#!/usr/bin/env python3
"""Recover the raw-scan -> label-frame transform of every Gold11 label set.

9 of the 11 Gold11 label sets declare coordinate_frame ``reference_mesh_raw``
(they were drawn in the bend-label studio jobs output/bend_label_NN_<part>/);
2 (shaked_xv C05/C06) declare ``scan_raw``. Plan s10 amendment 6(a) requires a
recovered raw-to-reference transform with residual <= 0.01 mm before scoring in
the raw-scan frame, otherwise the case is listed and excluded.

Method (read-only):
  reference_mesh_raw  The label job's display cloud (aligned_scan.ply, the points
                      the labeller saw) is matched point-for-point to the raw scan
                      vertices with a KD-tree. If every display point coincides with
                      a raw vertex (max distance <= 0.01 mm) the transform is the
                      identity. Otherwise a rigid fit (ICP with Kabsch on nearest
                      pairs) is attempted and accepted only if its max residual is
                      <= 0.01 mm. The job's bend_inspection_meta.json provenance
                      (coordinate_frame_provenance, source_sha256) is recorded.
  scan_raw            The labelled file's sha256 must equal the catalog scan_sha256
                      (identity by construction).
For every case the gold lines' raw-scan support (3 mm, the frozen gate) is also
reported as an independent check that the lines sit on the raw scan.

Output: JSON consumed by score.py --frame-records.
  gold_frames.py --out docs/production/runlog/broad_baseline/gold11_frame_recovery.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import gold11_cases as g11  # noqa: E402

FRAME_SCHEMA = "sherman.broad_baseline.gold_frame_records.v1"


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    h = (source - cs).T @ (target - ct)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = ct - rotation @ cs
    return transform


def recover_reference_frame(display: np.ndarray, raw: np.ndarray, *, iterations: int = 30) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    tree = cKDTree(raw)
    distances, index = tree.query(display, k=1, workers=1)
    result: dict[str, Any] = {
        "n_display_points": int(len(display)),
        "n_raw_points": int(len(raw)),
        "identity_nn_max_mm": float(distances.max()),
        "identity_nn_p99_mm": float(np.quantile(distances, 0.99)),
        "identity_unique_raw_vertices_hit": int(len(np.unique(index))),
    }
    if distances.max() <= bb.TRANSFORM_RESIDUAL_MAX_MM:
        result.update({"method": "DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY",
                       "raw_to_reference": np.eye(4).tolist(),
                       "residual_max_mm": float(distances.max()),
                       "status": "RECOVERED"})
        return result
    # Rigid fit: transform raw -> display frame (raw_to_reference), ICP from identity.
    transform = np.eye(4)
    moved = raw.copy()
    for _ in range(iterations):
        moved_tree = cKDTree(moved)
        d, idx = moved_tree.query(display, k=1, workers=1)
        step = _kabsch(moved[idx], display)
        transform = step @ transform
        moved = bb.apply_h(transform, raw)
    d, idx = cKDTree(moved).query(display, k=1, workers=1)
    result.update({"method": "NN_ICP_KABSCH", "raw_to_reference": transform.tolist(),
                   "residual_max_mm": float(d.max()), "residual_p99_mm": float(np.quantile(d, 0.99)),
                   **bb.similarity_report(transform)})
    result["status"] = "RECOVERED" if d.max() <= bb.TRANSFORM_RESIDUAL_MAX_MM else "TRANSFORM_RESIDUAL_TOO_LARGE"
    return result


def gold_support(raw: np.ndarray, bends: list[dict[str, Any]], transform_raw_to_ref: np.ndarray) -> list[dict[str, Any]]:
    from scripts.build_additional_line_label_review import line_scan_support

    order = np.lexsort((raw[:, 2], raw[:, 1], raw[:, 0]))
    pts = raw[order]
    if len(pts) > bb.GATE_CONSTANTS["scan_point_cap"]:
        pts = pts[np.linspace(0, len(pts) - 1, bb.GATE_CONSTANTS["scan_point_cap"], dtype=np.int64)]
    inverse = np.linalg.inv(transform_raw_to_ref)
    rows = []
    for bend in bends:
        ends = bb.apply_h(inverse, [bend["line_start"], bend["line_end"]])
        support = line_scan_support(pts, ends[0].tolist(), ends[1].tolist())
        rows.append({"bend_id": bend["bend_id"], "support_fraction": round(support["support_fraction"], 4),
                     "median_distance_mm": round(support["median_distance_mm"], 4),
                     "supported_span_mm": round(support["supported_span_mm"], 3),
                     "frame_integrity_pass": support["frame_integrity_pass"]})
    return rows


def recover_case(case: dict[str, Any], *, gold11_id: str | None) -> dict[str, Any]:
    scan = Path(case["scan_path"])
    record: dict[str, Any] = {
        "gold11_id": gold11_id, "catalog_case_id": case["case_id"], "family_id": case["part_id"],
        "label_frame": case["coordinate_frame"], "scan_path": str(scan),
        "scan_sha256": case["scan_sha256"], "bend_count": case["bend_count"],
        "source_labels_path": case.get("source_labels_path"), "truth_tier": case.get("truth_tier"),
    }
    started = time.monotonic()
    if not scan.is_file():
        record.update({"status": "SCAN_MISSING", "raw_to_reference": None})
        return record
    actual = bb.sha256_file(scan)
    record["scan_sha256_actual"] = actual
    if actual != case["scan_sha256"]:
        record.update({"status": "SCAN_SHA_MISMATCH", "raw_to_reference": None})
        return record
    raw = bb.load_scan_vertices(scan)
    if case["coordinate_frame"] == "scan_raw":
        record.update({"method": "SHA256_BOUND_SCAN_RAW", "raw_to_reference": np.eye(4).tolist(),
                       "residual_max_mm": 0.0, "status": "RECOVERED"})
    elif case["coordinate_frame"] == "reference_mesh_raw":
        label_dir = Path(case["source_labels_path"]).parent
        display_path = label_dir / "aligned_scan.ply"
        meta_path = label_dir / "bend_inspection_meta.json"
        meta = bb.read_json(meta_path).get("seed_case", {}) if meta_path.is_file() else {}
        record["label_job_provenance"] = {k: meta.get(k) for k in (
            "coordinate_frame", "coordinate_frame_provenance", "labeling_mode", "display_point_count",
            "source_sha256", "source_scan")}
        record["label_job_source_sha_matches"] = meta.get("source_sha256") == case["scan_sha256"]
        if not display_path.is_file():
            record.update({"status": "NO_RECOVERABLE_TRANSFORM", "raw_to_reference": None,
                           "reason": "label display cloud missing"})
            return record
        display = bb.load_scan_vertices(display_path)
        record["display_cloud"] = {"path": str(display_path), "sha256": bb.sha256_file(display_path)}
        record.update(recover_reference_frame(display, raw))
    else:
        record.update({"status": "UNKNOWN_FRAME", "raw_to_reference": None})
        return record
    if record.get("status") == "RECOVERED":
        record["gold_line_raw_support"] = gold_support(raw, case["bends"], np.asarray(record["raw_to_reference"]))
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=bb.GOLD_CATALOG)
    parser.add_argument("--case", action="append", help="Gold11 id (CG...) to process; default all 11")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rss-limit-gib", type=float, default=9.0)
    args = parser.parse_args(argv)
    catalog_sha = bb.sha256_file(args.catalog)
    catalog = bb.read_json(args.catalog)
    by_sha = {row["scan_sha256"]: row for row in catalog["cases"]}
    wanted = args.case or sorted(g11.CASES)
    records = []
    for gold11_id in wanted:
        bb.wait_for_rss_budget(args.rss_limit_gib)
        case = by_sha[g11.scan_sha(gold11_id)]
        if case["case_id"] != g11.catalog_case(gold11_id):
            raise ValueError(f"{gold11_id}: catalog case mismatch {case['case_id']}")
        record = recover_case(case, gold11_id=gold11_id)
        records.append(record)
        print(f"{gold11_id} {case['case_id']:22s} {case['coordinate_frame']:18s} -> {record['status']} "
              f"method={record.get('method')} residual_max_mm={record.get('residual_max_mm')}", flush=True)
    payload = {
        "schema": FRAME_SCHEMA,
        "created_at": bb.utc_now(),
        "catalog": {"path": str(args.catalog), "sha256": catalog_sha,
                    "expected_sha256": g11.CATALOG_SHA256, "matches": catalog_sha == g11.CATALOG_SHA256},
        "residual_limit_mm": bb.TRANSFORM_RESIDUAL_MAX_MM,
        "convention": "raw_to_reference maps raw scan coordinates to the label frame; gold lines are "
                      "mapped to scan_raw with its inverse",
        "recovered": sum(r["status"] == "RECOVERED" for r in records),
        "excluded": [{"gold11_id": r["gold11_id"], "status": r["status"]} for r in records
                     if r["status"] != "RECOVERED"],
        "cases": records,
    }
    bb.write_json(args.out, payload)
    print(f"recovered {payload['recovered']}/{len(records)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
