#!/usr/bin/env python3
"""Diagnostic: why does the active runtime emit so few scan-fitted lines?

run_progressive_bend_inspection forces ``position_mode=enabled, strict_spatial_matching=True`` into
the matcher kwargs whenever a shared scan alignment is passed (backend/bend_inspection_pipeline.py,
``shared_frame`` branch) - and the worker always passes one. The blind CAD+PLY call path
(scripts/evaluate_bend_case.py, the CAD half of best_f1) does not, and detects on raw-frame points.

This probe replicates the pipeline's PRIMARY detect-and-match step on one scan with the same
runtime config, CAD bend extraction and scan runtime profile, then:
  1. detects bends on the raw points and on the worker's shared-frame points
     (raw points mapped with the worker report's raw_scan_to_reference_transform);
  2. matches the shared-frame detections twice: with the worker's forced kwargs and with the
     default kwargs; matches the raw-frame detections with the default kwargs (blind path).
Nothing is written except --out. Read-only imports of backend modules; one process.

  probe  : spatial_matching_probe.py probe --scan S --cad C --report bend_inspection_report.json --out O.json
  run    : spatial_matching_probe.py run ... (same args) -> runs `probe` as one monitored heavy step
           (engine flock, python-RSS gate, /usr/bin/time -l, 600 s cap)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402


def _summ(report) -> dict:
    report.compute_summary()
    matches = list(report.matches or [])
    detected = [m for m in matches if m.status != "NOT_DETECTED" and m.detected_bend is not None]
    return {"cad_bends_matched_with_detection": len(detected),
            "detected_count_summary": int(report.detected_count),
            "matched_cad_bend_ids": sorted(str(getattr(m.cad_bend, "bend_id", "")) for m in detected),
            "unmatched_detections": len(report.unmatched_detections or [])}


def cmd_probe(args: argparse.Namespace) -> int:
    import numpy as np
    from bend_inspection_pipeline import (
        _apply_scan_runtime_profile,
        detect_scan_bends,
        extract_cad_bend_specs_with_strategy,
        load_bend_runtime_config,
        load_cad_geometry,
        load_scan_points,
        match_detected_bends,
        select_cad_bends_for_expected,
    )

    t0 = perf_counter()
    cfg = load_bend_runtime_config(None)
    token = "PROBE"
    cad_vertices, cad_triangles, _src, cad_provenance = load_cad_geometry(
        args.cad, cad_import_deflection=cfg.cad_import_deflection, include_provenance=True,
        include_native_topology=True)
    cad_bends, strategy = extract_cad_bend_specs_with_strategy(
        cad_vertices=cad_vertices, cad_triangles=cad_triangles, tolerance_angle=1.0, tolerance_radius=0.5,
        cad_extractor_kwargs=cfg.cad_extractor_kwargs, cad_detector_kwargs=cfg.cad_detector_kwargs,
        cad_detect_call_kwargs=cfg.cad_detect_call_kwargs, expected_bend_count_hint=None,
        spec_bend_angles_hint=None, spec_schedule_type=None, feature_policy=None, part_id=token,
        cad_provenance=cad_provenance)
    cad_bends = select_cad_bends_for_expected(cad_bends, len(cad_bends))
    raw = load_scan_points(args.scan)
    report = bb.read_json(args.report)
    transform = bb.validate_h(report["alignment"]["raw_scan_to_reference_transform"])
    aligned = bb.apply_h(transform, raw)
    ctor, call, profile = _apply_scan_runtime_profile(cfg.scan_detector_kwargs, cfg.scan_detect_call_kwargs,
                                                      len(raw), scan_state=None, fallback=False)
    det_raw = detect_scan_bends(scan_points=raw, scan_detector_kwargs=ctor, scan_detect_call_kwargs=call)
    det_aligned = detect_scan_bends(scan_points=aligned, scan_detector_kwargs=ctor, scan_detect_call_kwargs=call)
    strict = {**cfg.matcher_kwargs, "position_mode": "enabled", "strict_spatial_matching": True}

    def nearest_cad(det):
        """Strict gate of ProgressiveBendMatcher._compute_match_cost: midpoint <= max_center_offset
        (40 mm default) and direction within 15 deg of the CAD bend line."""
        b = np.asarray([det.bend_line_start, det.bend_line_end], dtype=float)
        db = (b[1] - b[0]) / max(np.linalg.norm(b[1] - b[0]), 1e-9)
        best = None
        for cad in cad_bends:
            a = np.asarray([cad.bend_line_start, cad.bend_line_end], dtype=float)
            da = (a[1] - a[0]) / max(np.linalg.norm(a[1] - a[0]), 1e-9)
            mid = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(da, db)))))))
            angle_diff = abs(float(det.measured_angle) - float(cad.target_angle))
            row = {"cad_bend_id": str(getattr(cad, "bend_id", "")), "midpoint_distance_mm": round(mid, 2),
                   "direction_angle_deg": round(ang, 2), "bend_angle_diff_deg": round(angle_diff, 2),
                   "cad_line_length_mm": round(float(np.linalg.norm(a[1] - a[0])), 1),
                   "detected_line_length_mm": round(float(np.linalg.norm(b[1] - b[0])), 1),
                   "passes_strict_gate": bool(mid <= 40.0 and ang <= 15.0)}
            if best is None or (row["direction_angle_deg"] <= 15.0, -mid) > (best["direction_angle_deg"] <= 15.0,
                                                                               -best["midpoint_distance_mm"]):
                best = row
        return best
    out = {
        "schema": "sherman.broad_baseline.spatial_matching_probe.v1",
        "scan": args.scan, "cad": args.cad, "worker_report": args.report,
        "cad_bend_strategy": strategy, "cad_bends": len(cad_bends), "scan_points": int(len(raw)),
        "scan_runtime_profile": profile,
        "detections": {"raw_frame": len(det_raw), "shared_frame": len(det_aligned)},
        "match": {
            "shared_frame__worker_strict_spatial": _summ(match_detected_bends(
                detected_bends=det_aligned, cad_bends=cad_bends, part_id=token, matcher_kwargs=strict)),
            "shared_frame__default_kwargs": _summ(match_detected_bends(
                detected_bends=det_aligned, cad_bends=cad_bends, part_id=token, matcher_kwargs=cfg.matcher_kwargs)),
            "raw_frame__default_kwargs_blind_path": _summ(match_detected_bends(
                detected_bends=det_raw, cad_bends=cad_bends, part_id=token, matcher_kwargs=cfg.matcher_kwargs)),
        },
        "shared_frame_detections_vs_nearest_cad_bend": [nearest_cad(d) for d in det_aligned],
        "matcher_kwargs_default": cfg.matcher_kwargs,
        "seconds": round(perf_counter() - t0, 3),
        "note": ("primary detect+match step only (no fallback, no dense local refinement); counts, not lines"),
    }
    bb.write_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("cad_bends", "detections", "match", "seconds")}, default=str))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import run_lane

    out = bb.expand(args.out)
    cmd = [str(bb.APP_PYTHON), "-B", str(Path(__file__).resolve()), "probe", "--scan", args.scan, "--cad", args.cad,
           "--report", args.report, "--out", str(out)]
    env = {"PYTHONPATH": os.pathsep.join([str(bb.ROOT), str(bb.ROOT / "backend")])}
    with run_lane._engine_lock(run_lane.DEFAULT_LOCK) as lock_info:
        record = bb.run_monitored(cmd, name="spatial_matching_probe", log_dir=out.parent / (out.stem + "_logs"),
                                  env=env, timeout_s=600.0, rss_limit_gib=9.0)
    if out.is_file():
        payload = bb.read_json(out)
        payload["step"] = {k: record.get(k) for k in ("status", "wall_s", "time_max_rss_bytes",
                                                      "sampled_peak_group_rss_bytes", "timed_out")}
        payload.update(lock_info)
        bb.write_json(out, payload)
    print(json.dumps({"status": record["status"], "wall_s": record["wall_s"]}))
    return 0 if record["status"] == "OK" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("probe", cmd_probe), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--scan", required=True)
        p.add_argument("--cad", required=True)
        p.add_argument("--report", required=True)
        p.add_argument("--out", required=True)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
