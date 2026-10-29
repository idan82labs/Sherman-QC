#!/usr/bin/env python3
"""Retain only CAD-targeted local bend measurements on the aligned scan.

This lane deliberately excludes open-world/global bend proposals.  It consumes
the existing automatic CAD/scan overlay and the engine's per-CAD-target local
measurements, then applies the pre-existing scan-frame support crop.  Constants
are inherited from production code or the frozen line matcher; none are fitted
to line gold.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_additional_line_label_review import _scan_points, line_scan_support
from scripts.evaluate_f1_against_manual_gold import pair_metrics


TRUSTED_ALIGNMENT_SOURCE = "PLANE_DATUM_REFINEMENT"
MIN_ALIGNMENT_CONFIDENCE = 0.55
DUPLICATE_ANGLE_DEG = 5.0
DUPLICATE_OFFSET_MM = 4.0
DUPLICATE_SHORTER_OVERLAP = 0.60


def _line(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["detected_line_start"], row["detected_line_end"]],
        dtype=np.float64,
    )


def _trusted_parent_frame(row: dict[str, Any]) -> bool:
    context = row.get("measurement_context") or {}
    parent = context.get("parent_frame") or {}
    return bool(
        not parent.get("alignment_abstained")
        and str(parent.get("alignment_source") or "").upper()
        == TRUSTED_ALIGNMENT_SOURCE
        and float(parent.get("alignment_confidence") or 0.0)
        >= MIN_ALIGNMENT_CONFIDENCE
        and parent.get("raw_scan_to_cad_transform") is not None
    )


def _axis_in_raw_scan_frame(
    row: dict[str, Any], *, axis_source: str
) -> tuple[list[float], list[float]]:
    context = row.get("measurement_context") or {}
    parent = context.get("parent_frame") or {}
    raw_to_cad = np.asarray(parent["raw_scan_to_cad_transform"], dtype=np.float64)
    if raw_to_cad.shape != (4, 4) or not np.all(np.isfinite(raw_to_cad)):
        raise ValueError("invalid raw_scan_to_cad_transform")
    cad_to_raw = np.linalg.inv(raw_to_cad)
    if axis_source == "cad_nominal":
        endpoints = np.asarray(
            [row["cad_line_start"], row["cad_line_end"]], dtype=np.float64
        )
    else:
        endpoints = np.asarray(
            [row["detected_line_start"], row["detected_line_end"]], dtype=np.float64
        )
    homogeneous = np.column_stack([endpoints, np.ones(2, dtype=np.float64)])
    transformed = (homogeneous @ cad_to_raw.T)[:, :3]
    return transformed[0].tolist(), transformed[1].tolist()


def _duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    metrics = pair_metrics(_line(left), _line(right))
    return bool(
        metrics["axis_angle_deg"] <= DUPLICATE_ANGLE_DEG
        and metrics["transverse_offset_mm"] <= DUPLICATE_OFFSET_MM
        and metrics["shorter_span_overlap"] >= DUPLICATE_SHORTER_OVERLAP
    )


def _priority(row: dict[str, Any]) -> tuple[float, float, float, str]:
    context = row.get("measurement_context") or {}
    parent = context.get("parent_frame") or {}
    method = str(row.get("measurement_method") or "")
    method_rank = 1.0 if method == "profile_section" else 0.0
    return (
        float(row.get("local_evidence_score") or 0.0),
        float(parent.get("alignment_confidence") or 0.0),
        method_rank,
        str(row.get("bend_id") or ""),
    )


def _eligibility_reason(row: dict[str, Any]) -> str | None:
    if str(row.get("assignment_source") or "") != "CAD_LOCAL_NEIGHBORHOOD":
        return "not_cad_local_measurement"
    if str(row.get("assignment_candidate_kind") or "") != "MEASUREMENT":
        return "not_measurement_candidate"
    if str(row.get("observability_state") or "") != "FORMED":
        return "not_observed_formed"
    if row.get("detected_line_start") is None or row.get("detected_line_end") is None:
        return "missing_observed_axis"
    if not _trusted_parent_frame(row):
        return "untrusted_existing_overlay"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--axis-source",
        choices=("local_observed", "cad_nominal"),
        default="local_observed",
    )
    args = parser.parse_args()

    source = json.loads(args.input_summary.read_text(encoding="utf-8"))
    output = copy.deepcopy(source)
    audit_cases = []
    totals = {
        "source_cad_targets": 0,
        "eligible_local_measurements": 0,
        "scan_supported_measurements": 0,
        "deduplicated_measurements": 0,
        "retained_measurements": 0,
    }
    for result in output["results"]:
        points = _scan_points(Path(result["scan_path"]))
        records = []
        eligible = []
        for row in result.get("matches") or []:
            totals["source_cad_targets"] += 1
            reason = _eligibility_reason(row)
            record = {
                "bend_id": str(row.get("bend_id") or ""),
                "assignment_source": row.get("assignment_source"),
                "measurement_method": row.get("measurement_method"),
                "eligible": reason is None,
                "rejection_reason": reason,
            }
            if reason is None:
                totals["eligible_local_measurements"] += 1
                if args.axis_source == "cad_nominal":
                    overlay_start = list(row["cad_line_start"])
                    overlay_end = list(row["cad_line_end"])
                else:
                    overlay_start = list(row["detected_line_start"])
                    overlay_end = list(row["detected_line_end"])
                raw_start, raw_end = _axis_in_raw_scan_frame(
                    row, axis_source=args.axis_source
                )
                row["overlay_frame_detected_line_start"] = overlay_start
                row["overlay_frame_detected_line_end"] = overlay_end
                row["detected_line_start"] = raw_start
                row["detected_line_end"] = raw_end
                record["overlay_frame_line_start"] = overlay_start
                record["overlay_frame_line_end"] = overlay_end
                record["raw_scan_frame_line_start"] = raw_start
                record["raw_scan_frame_line_end"] = raw_end
                support = line_scan_support(
                    points,
                    raw_start,
                    raw_end,
                )
                record["scan_support"] = support
                if support["frame_integrity_pass"]:
                    totals["scan_supported_measurements"] += 1
                    row["detected_line_start"] = support["trimmed_line_start"]
                    row["detected_line_end"] = support["trimmed_line_end"]
                    row["cad_targeted_local_verifier"] = "accepted_scan_supported_local_measurement"
                    eligible.append(row)
                else:
                    record["rejection_reason"] = "failed_scan_frame_support"
            records.append(record)

        retained = []
        for row in sorted(eligible, key=_priority, reverse=True):
            duplicate = next((existing for existing in retained if _duplicate(row, existing)), None)
            if duplicate is not None:
                totals["deduplicated_measurements"] += 1
                for record in records:
                    if record["bend_id"] == str(row.get("bend_id") or ""):
                        record["rejection_reason"] = (
                            "duplicate_of_" + str(duplicate.get("bend_id") or "")
                        )
                        break
                continue
            retained.append(row)
        retained.sort(key=lambda row: str(row.get("bend_id") or ""))
        totals["retained_measurements"] += len(retained)
        result["matches"] = retained
        result["summary"] = {
            **dict(result.get("summary") or {}),
            "cad_targeted_local_verified_bends": len(retained),
        }
        audit_cases.append(
            {
                "part_key": str(result.get("part_key") or ""),
                "scan_path": str(result["scan_path"]),
                "retained_bend_ids": [str(row.get("bend_id") or "") for row in retained],
                "records": records,
            }
        )

    audit = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_cad_targeted_local_verifier_audit",
        "uses_existing_automatic_cad_scan_overlay": True,
        "axis_source": args.axis_source,
        "global_scan_proposals_allowed": False,
        "gold_or_expected_counts_used": False,
        "fitted_parameters": False,
        "contract": {
            "required_assignment_source": "CAD_LOCAL_NEIGHBORHOOD",
            "required_candidate_kind": "MEASUREMENT",
            "required_observability_state": "FORMED",
            "required_alignment_source": TRUSTED_ALIGNMENT_SOURCE,
            "minimum_alignment_confidence": MIN_ALIGNMENT_CONFIDENCE,
            "scan_support_sample_count": 201,
            "scan_support_distance_mm": 3.0,
            "minimum_scan_supported_fraction": 0.10,
            "minimum_scan_supported_span_mm": 8.0,
            "duplicate_angle_deg_max": DUPLICATE_ANGLE_DEG,
            "duplicate_transverse_offset_mm_max": DUPLICATE_OFFSET_MM,
            "duplicate_shorter_span_overlap_min": DUPLICATE_SHORTER_OVERLAP,
        },
        "totals": totals,
        "cases": audit_cases,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.audit.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
