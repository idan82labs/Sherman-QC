#!/usr/bin/env python3
"""Convert lane outputs to the evaluator's prediction format, in the raw-scan frame.

Output schema ``sherman.broad_baseline.eval_lines.v1``: one JSON object per scan
whose ``owned_bend_regions`` rows carry ``bend_id``, ``line_start``, ``line_end``
(raw scan file coordinates). The key name is the one
``scripts/evaluate_f1_against_manual_gold.evaluate`` reads, so these files can
also be fed to that script through ``--case-map``.

Kinds
  a0              prediction_owned_regions.json of scripts/run_f1_structured_union_e2e.py
                  (scan-only typed union, coordinate_frame scan_raw). Rows are
                  normalized exactly like diagnose_f1_cad_scan_fusion_gold11._normalized_a0.
  decomposition   decomposition_result.json (strict-blind zero-shot F1 decomposition,
                  provenance.coordinate_frame scan_raw); endpoints by the evaluator's
                  _endpoints precedence (span_endpoints, line_start/end, anchor+axis+span).
  cad_case        output JSON of scripts/evaluate_bend_case.py --inference-mode blind_cad_ply
                  (historical CAD half of the best-F1 lane). Every match with a detected
                  line, in report order, is passed through the frozen July raw-scan support
                  gate (201 samples, 3 mm, >=10 % support, >=8 mm longest run; accepted
                  lines cropped to that run) exactly as
                  scripts/apply_f1_cad_guided_scan_frame_gate.py + score_f1_cad_guided_gold13_e2e
                  did. --no-gate keeps the raw lines (historical "raw CAD-only" diagnostic).
  cad_semantic    retained semantic_result.json (already gated CAD predictions).
  runtime_report  bend_inspection_report.json of backend/bend_analysis_worker.py (active
                  runtime). Detected lines live in the reference (CAD) frame; they are mapped
                  to the raw scan frame with inv(alignment.raw_scan_to_reference_transform).
                  Only detected_line_geometry_source == SCAN_FIT rows are predictions by
                  default; CAD_REFERENCE_PLACEHOLDER rows (assignment_source
                  CAD_LOCAL_NEIGHBORHOOD: nominal CAD axis copied into the detection slot)
                  are dropped unless --include-placeholders.
  union           best-F1 fusion of an A0 eval file (--input) and a gated-CAD eval file
                  (--cad-input): all A0 lines, then each CAD line that is not a duplicate
                  (5 deg / 4 mm / 60 % SHORTER-span overlap) of any union member, in input
                  order, renamed CADF_<id>. Same code as diagnose_f1_cad_scan_fusion_gold11.
  intersection    A0 lines corroborated by a gated CAD line (diagnostic only).

Usage
  to_eval_format.py --kind a0 --input .../prediction_owned_regions.json --scan SCAN --out a0.json
  to_eval_format.py --kind cad_case --input .../cad_case.json --scan SCAN --out cad_gated.json
  to_eval_format.py --kind runtime_report --input .../bend_inspection_report.json --scan SCAN --out rt.json
  to_eval_format.py --kind union --input a0.json --cad-input cad_gated.json --out best_f1.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402


# ------------------------------------------------------------------ helpers

def _scan_identity(scan: Path | None) -> tuple[Path | None, str | None]:
    if scan is None:
        return None, None
    path = bb.expand(scan)
    return path, bb.sha256_file(path)


def _gate_points(scan: Path) -> np.ndarray:
    from scripts.build_additional_line_label_review import _scan_points  # frozen gate loader

    return _scan_points(Path(scan))


def _gate(points: np.ndarray, start: Any, end: Any) -> dict[str, Any]:
    from scripts.build_additional_line_label_review import line_scan_support  # frozen gate

    return line_scan_support(points, start, end)


# --------------------------------------------------------------- converters

def convert_a0(payload: Mapping[str, Any], *, source: Path | None, scan: Path | None,
               lane: str = "a0_scan_only_typed_union") -> dict[str, Any]:
    from scripts.diagnose_f1_cad_scan_fusion_gold11 import _normalized_a0

    frame = payload.get("coordinate_frame")
    if frame != "scan_raw":
        raise ValueError(f"A0 payload frame {frame!r} is not scan_raw")
    lines = []
    for row in payload.get("owned_bend_regions") or []:
        normalized = _normalized_a0(row)
        lines.append(bb.as_line(normalized["bend_id"], normalized["line_start"], normalized["line_end"],
                                origin="scan_only_a0",
                                semantic_label=row.get("semantic_label"),
                                span_mm=row.get("span_mm")))
    scan_path, scan_sha = _scan_identity(scan)
    return bb.eval_payload(lane=lane, lines=lines, scan_path=scan_path, scan_sha256=scan_sha,
                           source_kind="a0_prediction_owned_regions", source_path=source,
                           conversion={"rule": "diagnose_f1_cad_scan_fusion_gold11._normalized_a0",
                                       "algorithm": payload.get("algorithm"),
                                       "frame_in": frame, "transform": "identity"},
                           status="OK" if lines else "NO_PREDICTIONS")


def convert_decomposition(payload: Mapping[str, Any], *, source: Path | None, scan: Path | None,
                          proxy_sha256: str | None = None) -> dict[str, Any]:
    """proxy_sha256: sha of the lossless PLY proxy the decomposition was run on (non-PLY scans);
    the proxy holds the scan's own vertex coordinates, so the frame is still scan_raw of --scan."""
    frame = (payload.get("provenance") or {}).get("coordinate_frame")
    if frame != "scan_raw":
        raise ValueError(f"decomposition frame {frame!r} is not scan_raw")
    lines, dropped = [], []
    for index, row in enumerate(payload.get("owned_bend_regions") or [], start=1):
        bend_id = str(row.get("bend_id") or f"D{index:02d}")
        try:
            ends = bb.endpoints(row)
        except (KeyError, TypeError, ValueError) as error:
            dropped.append({"bend_id": bend_id, "reason": f"NO_ENDPOINTS:{type(error).__name__}"})
            continue
        if not bb.valid_line(ends[0], ends[1]):
            dropped.append({"bend_id": bend_id, "reason": "DEGENERATE_LINE"})
            continue
        lines.append(bb.as_line(bend_id, ends[0], ends[1], origin="f1_zero_shot_decomposition"))
    scan_path, scan_sha = _scan_identity(scan)
    if scan_sha and (payload.get("provenance") or {}).get("cache_identity", {}).get("scan_sha256") not in (
            None, scan_sha, proxy_sha256):
        raise ValueError("decomposition scan_sha256 differs from --scan")
    return bb.eval_payload(lane="f1_zero_shot_decomposition", lines=lines, scan_path=scan_path,
                           scan_sha256=scan_sha, source_kind="decomposition_result", source_path=source,
                           conversion={"rule": "evaluate_f1_against_manual_gold._endpoints",
                                       "frame_in": frame, "transform": "identity"},
                           dropped=dropped, status="OK" if lines else "NO_PREDICTIONS")


def convert_cad_case(payload: Mapping[str, Any], *, source: Path | None, scan: Path,
                     gate: bool = True) -> dict[str, Any]:
    """Historical CAD half: score_f1_cad_guided_gold13_e2e._cad_predictions after the July gate."""
    scan_path, scan_sha = _scan_identity(scan)
    points = _gate_points(scan_path) if gate else None
    lines, dropped, audit = [], [], []
    for row in payload.get("matches") or []:
        start, end = row.get("detected_line_start"), row.get("detected_line_end")
        if start is None or end is None:
            continue
        bend_id = str(row.get("bend_id") or f"CAD_SCAN_{len(lines) + 1}")
        meta = {
            "origin": "cad_guided_blind_runtime_corpus_path",
            "cad_target_id": str(row.get("bend_id") or ""),
            "assignment_source": row.get("assignment_source"),
            "measurement_method": row.get("measurement_method"),
            "observability_state": row.get("observability_state"),
            "detected_line_geometry_source": row.get("detected_line_geometry_source"),
            "status": row.get("status"),
        }
        if gate:
            support = _gate(points, start, end)
            audit.append({"bend_id": bend_id, "accepted": bool(support["frame_integrity_pass"]),
                          **{k: support[k] for k in ("support_fraction", "supported_span_mm",
                                                       "median_distance_mm", "source_span_mm")}})
            if not support["frame_integrity_pass"]:
                dropped.append({"bend_id": bend_id, "reason": "SCAN_FRAME_GATE_REJECTED",
                                "support_fraction": support["support_fraction"],
                                "supported_span_mm": support["supported_span_mm"], **meta})
                continue
            start, end = support["trimmed_line_start"], support["trimmed_line_end"]
        # Bend id assignment follows _cad_predictions: count AFTER gating (gated rows are nulled).
        bend_id = str(row.get("bend_id") or f"CAD_SCAN_{len(lines) + 1}")
        lines.append(bb.as_line(bend_id, start, end, **meta))
    lane = "cad_gated" if gate else "cad_raw_ungated"
    return bb.eval_payload(
        lane=lane, lines=lines, scan_path=scan_path, scan_sha256=scan_sha,
        source_kind="evaluate_bend_case_blind_cad_ply", source_path=source,
        conversion={
            "rule": ("apply_f1_cad_guided_scan_frame_gate + score_f1_cad_guided_gold13_e2e._cad_predictions"
                     if gate else "score_f1_cad_guided_gold13_e2e._cad_predictions (no gate)"),
            "gate_constants": bb.GATE_CONSTANTS if gate else None,
            "frame_in": "MIXED: GLOBAL_DETECTION rows are raw-scan, local-refinement rows are CAD-aligned; "
                        "the raw-scan support gate is the historical frame filter",
            "transform": "none (historical)",
            "gate_audit": audit,
            "inference_mode": payload.get("inference_mode"),
        },
        dropped=dropped, status="OK" if lines else "NO_PREDICTIONS")


def convert_cad_semantic(payload: Mapping[str, Any], *, source: Path | None, scan: Path | None) -> dict[str, Any]:
    if payload.get("kind") != "sherman_qc_f1_cad_guided_scan_line_predictions":
        raise ValueError("not a CAD semantic_result payload")
    lines = [bb.as_line(row["bend_id"], row["line_start"], row["line_end"], origin="cad_guided_gated_retained",
                        **{k: row.get(k) for k in ("cad_target_id", "assignment_source", "measurement_method",
                                                   "observability_state")})
             for row in payload.get("predictions") or []]
    scan_path, scan_sha = _scan_identity(scan)
    return bb.eval_payload(lane="cad_gated", lines=lines, scan_path=scan_path, scan_sha256=scan_sha,
                           source_kind="cad_semantic_result", source_path=source,
                           conversion={"rule": "retained gated CAD predictions, unchanged",
                                       "transform": "none (historical)"},
                           status="OK" if lines else "NO_PREDICTIONS")


def convert_runtime_report(payload: Mapping[str, Any], *, source: Path | None, scan: Path | None,
                           include_placeholders: bool = False, frame: str = "raw",
                           gate: bool = False) -> dict[str, Any]:
    alignment = payload.get("alignment") or {}
    transform = alignment.get("raw_scan_to_reference_transform")
    if transform is None:
        transform = (payload.get("alignment_metadata") or {}).get("raw_scan_to_cad_transform")
    inverse = None
    transform_info: dict[str, Any] = {"available": transform is not None}
    if transform is not None:
        matrix = bb.validate_h(transform)
        inverse = np.linalg.inv(matrix)
        transform_info.update({"raw_scan_to_reference_transform": matrix.tolist(),
                               "geometry_frame": alignment.get("geometry_frame"),
                               "fitness": alignment.get("fitness"), "rmse_mm": alignment.get("rmse_mm"),
                               **bb.similarity_report(matrix)})
    if frame == "raw" and inverse is None:
        raise ValueError("runtime report has no raw_scan_to_reference_transform; cannot map to scan_raw")
    scan_path, scan_sha = _scan_identity(scan)
    points = _gate_points(scan_path) if scan_path is not None else None
    lines, dropped, audit = [], [], []
    for row in payload.get("matches") or []:
        start, end = row.get("detected_line_start"), row.get("detected_line_end")
        bend_id = str(row.get("bend_id") or f"RT_{len(lines) + len(dropped) + 1}")
        source_tag = row.get("detected_line_geometry_source")
        meta = {"origin": "active_runtime_worker", "status": row.get("status"),
                "assignment_source": row.get("assignment_source"),
                "detected_line_geometry_source": source_tag,
                "measurement_method": row.get("measurement_method"),
                "observability_state": row.get("observability_state"),
                "measured_angle": row.get("measured_angle"), "target_angle": row.get("target_angle")}
        if not bb.valid_line(start, end):
            continue
        if source_tag == "CAD_REFERENCE_PLACEHOLDER" and not include_placeholders:
            dropped.append({"bend_id": bend_id, "reason": "CAD_REFERENCE_PLACEHOLDER_NOT_SCAN_LINE", **meta})
            continue
        if frame == "raw":
            mapped = bb.apply_h(inverse, [start, end])
            start, end = mapped[0].tolist(), mapped[1].tolist()
        if points is not None:
            support = _gate(points, start, end)
            audit.append({"bend_id": bend_id, "raw_scan_support_pass": bool(support["frame_integrity_pass"]),
                          **{k: support[k] for k in ("support_fraction", "supported_span_mm",
                                                       "median_distance_mm", "source_span_mm")}})
            if gate:
                if not support["frame_integrity_pass"]:
                    dropped.append({"bend_id": bend_id, "reason": "SCAN_FRAME_GATE_REJECTED", **meta})
                    continue
                start, end = support["trimmed_line_start"], support["trimmed_line_end"]
        lines.append(bb.as_line(bend_id, start, end, **meta))
    lane = "runtime" if not (gate or include_placeholders or frame != "raw") else (
        "runtime" + ("_gated" if gate else "") + ("_with_placeholders" if include_placeholders else "")
        + ("_reference_frame_UNMAPPED" if frame != "raw" else ""))
    return bb.eval_payload(
        lane=lane, lines=lines, scan_path=scan_path, scan_sha256=scan_sha,
        source_kind="bend_analysis_worker_report", source_path=source,
        conversion={"rule": "detected_line_* of matches; inv(raw_scan_to_reference_transform) to scan_raw",
                    "frame_in": "reference_mesh_raw (shared worker alignment)",
                    "frame_out": "scan_raw" if frame == "raw" else "reference_mesh_raw (NOT mapped)",
                    "transform": transform_info, "include_placeholders": include_placeholders,
                    "gate_applied": gate, "raw_scan_support_audit": audit},
        dropped=dropped, status="OK" if lines else "NO_PREDICTIONS")


def fuse(a0: Mapping[str, Any], cad: Mapping[str, Any], *, mode: str = "union") -> dict[str, Any]:
    from scripts.diagnose_f1_cad_scan_fusion_gold11 import _duplicate, _normalized_a0

    scan_rows = [_normalized_a0(row) for row in a0.get("owned_bend_regions") or []]
    cad_rows = [dict(row) for row in cad.get("owned_bend_regions") or []]
    if mode == "intersection":
        rows = [row for row in scan_rows if any(_duplicate(row, other) for other in cad_rows)]
    else:
        rows = list(scan_rows)
        for row in cad_rows:
            if not any(_duplicate(row, existing) for existing in rows):
                added = dict(row)
                added["bend_id"] = "CADF_" + str(added["bend_id"])
                added["fusion_source"] = "cad_guided_gated_rescue"
                rows.append(added)
    lines = [bb.as_line(row["bend_id"], row["line_start"], row["line_end"],
                        **{k: v for k, v in row.items() if k not in {"bend_id", "line_start", "line_end"}})
             for row in rows]
    shas = {a0.get("scan", {}).get("sha256"), cad.get("scan", {}).get("sha256")} - {None}
    if len(shas) > 1:
        raise ValueError("A0 and CAD eval files reference different scans")
    scan = a0.get("scan") or cad.get("scan") or {}
    payload = bb.eval_payload(
        lane="best_f1_union" if mode == "union" else "best_f1_intersection_diagnostic", lines=lines,
        scan_path=None if scan.get("path") is None else Path(scan["path"]),
        scan_sha256=next(iter(shas)) if shas else None,
        source_kind="fusion", source_path=None,
        conversion={"rule": f"diagnose_f1_cad_scan_fusion_gold11 {mode}",
                    "duplicate_rule": {"axis_angle_deg_max": bb.FUSION_ANGLE_DEG,
                                       "transverse_offset_mm_max": bb.FUSION_OFFSET_MM,
                                       "shorter_span_overlap_min": bb.FUSION_SHORTER_OVERLAP},
                    "a0_lines": len(scan_rows), "cad_lines": len(cad_rows),
                    "a0_status": a0.get("status"), "cad_status": cad.get("status")},
        status="OK" if lines else "NO_PREDICTIONS")
    return payload


def empty_lane(lane: str, *, scan: Path | None, reason: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scan_path, scan_sha = _scan_identity(scan)
    return bb.eval_payload(lane=lane, lines=[], scan_path=scan_path, scan_sha256=scan_sha,
                           source_kind="lane_failure", source_path=None, conversion={},
                           status="LANE_FAILED", failure={"reason": reason, **dict(detail or {})})


# --------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True,
                        choices=("a0", "decomposition", "cad_case", "cad_semantic", "runtime_report",
                                 "union", "intersection"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cad-input", type=Path, help="gated CAD eval file (union/intersection)")
    parser.add_argument("--scan", type=Path, help="raw scan file (needed for gating / sha binding)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-gate", action="store_true", help="cad_case: keep ungated lines (diagnostic)")
    parser.add_argument("--include-placeholders", action="store_true", help="runtime_report: keep CAD placeholders")
    parser.add_argument("--runtime-gate", action="store_true", help="runtime_report: apply the raw-scan gate")
    parser.add_argument("--runtime-frame", choices=("raw", "reference"), default="raw")
    args = parser.parse_args(argv)
    payload = bb.read_json(args.input)
    if args.kind == "a0":
        out = convert_a0(payload, source=args.input, scan=args.scan)
    elif args.kind == "decomposition":
        out = convert_decomposition(payload, source=args.input, scan=args.scan)
    elif args.kind == "cad_case":
        if args.scan is None:
            parser.error("--scan is required for cad_case")
        out = convert_cad_case(payload, source=args.input, scan=args.scan, gate=not args.no_gate)
    elif args.kind == "cad_semantic":
        out = convert_cad_semantic(payload, source=args.input, scan=args.scan)
    elif args.kind == "runtime_report":
        out = convert_runtime_report(payload, source=args.input, scan=args.scan,
                                     include_placeholders=args.include_placeholders,
                                     frame=args.runtime_frame, gate=args.runtime_gate)
    else:
        if args.cad_input is None:
            parser.error("--cad-input is required for union/intersection")
        out = fuse(bb.load_eval_lines(args.input), bb.load_eval_lines(args.cad_input), mode=args.kind)
    bb.write_json(args.out, out)
    print(f"{args.kind}: {out['line_count']} lines ({len(out['dropped'])} dropped) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
