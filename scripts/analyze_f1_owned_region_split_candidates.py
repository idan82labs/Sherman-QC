#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return float(sum(float(lhs[i]) * float(rhs[i]) for i in range(3)))


def _sub(lhs: Sequence[float], rhs: Sequence[float]) -> List[float]:
    return [float(lhs[i]) - float(rhs[i]) for i in range(3)]


def _norm(value: Sequence[float]) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in value)))


def _normalize(value: Sequence[float]) -> List[float]:
    n = _norm(value)
    if n <= 1e-9:
        return [1.0, 0.0, 0.0]
    return [float(v) / n for v in value]


def _atom_lookup(decomposition: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(atom.get("atom_id")): atom
        for atom in list((decomposition.get("atom_graph") or {}).get("atoms") or ())
        if atom.get("atom_id")
    }


def _split_metrics_for_region(region: Mapping[str, Any], atoms: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    atom_ids = [str(value) for value in list(region.get("owned_atom_ids") or ())]
    points = [
        list(atoms[atom_id].get("centroid") or ())
        for atom_id in atom_ids
        if atom_id in atoms and len(list(atoms[atom_id].get("centroid") or ())) == 3
    ]
    axis = _normalize(list(region.get("axis_direction") or (1.0, 0.0, 0.0)))
    anchor = list(region.get("anchor") or (0.0, 0.0, 0.0))
    if len(anchor) != 3:
        anchor = [0.0, 0.0, 0.0]
    projections = sorted(_dot(_sub(point, anchor), axis) for point in points)
    gaps = [projections[index + 1] - projections[index] for index in range(len(projections) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    median_gap = float(median(gaps)) if gaps else 0.0
    span = (max(projections) - min(projections)) if projections else 0.0
    split_index = gaps.index(max_gap) if gaps else -1
    left_count = split_index + 1 if split_index >= 0 else 0
    right_count = len(projections) - left_count
    gap_ratio = max_gap / max(median_gap, 1e-6)
    balanced_split = bool(min(left_count, right_count) >= 3)
    substantial = bool(len(points) >= 8 and span >= 40.0)
    strong_gap = bool(max_gap >= 12.0 and gap_ratio >= 2.5)
    split_candidate = bool(substantial and balanced_split and strong_gap)
    if split_candidate and gap_ratio >= 3.5:
        rank = "strong"
    elif split_candidate:
        rank = "moderate"
    elif substantial and balanced_split:
        rank = "weak"
    else:
        rank = "not_candidate"
    return {
        "bend_id": region.get("bend_id"),
        "owned_atom_count": len(points),
        "support_mass": region.get("support_mass"),
        "incident_flange_ids": list(region.get("incident_flange_ids") or ()),
        "angle_deg": region.get("angle_deg"),
        "span_projection_mm": round(float(span), 4),
        "max_projection_gap_mm": round(float(max_gap), 4),
        "median_projection_gap_mm": round(float(median_gap), 4),
        "max_gap_to_median_gap_ratio": round(float(gap_ratio), 4) if gaps else None,
        "largest_gap_left_atom_count": int(left_count),
        "largest_gap_right_atom_count": int(right_count),
        "balanced_split": balanced_split,
        "substantial_support": substantial,
        "strong_gap": strong_gap,
        "split_candidate": split_candidate,
        "split_candidate_rank": rank,
        "diagnostic_reason": (
            "large balanced along-axis support gap"
            if split_candidate
            else "no large balanced along-axis gap"
        ),
    }


def analyze_owned_region_split_candidates(
    *,
    decomposition: Mapping[str, Any],
    feature_label_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    atoms = _atom_lookup(decomposition)
    regions = list(decomposition.get("owned_bend_regions") or ())
    metrics = [_split_metrics_for_region(region, atoms) for region in regions]
    metrics.sort(
        key=lambda row: (
            {"strong": 0, "moderate": 1, "weak": 2, "not_candidate": 3}.get(str(row["split_candidate_rank"]), 9),
            -float(row.get("max_gap_to_median_gap_ratio") or 0.0),
            str(row.get("bend_id")),
        )
    )
    summary = dict(feature_label_summary or {})
    capacity = dict(summary.get("label_capacity") or {})
    deficit = _as_int(capacity.get("owned_region_capacity_deficit")) or 0
    split_candidates = [row for row in metrics if row.get("split_candidate")]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": decomposition.get("scan_path"),
        "part_id": decomposition.get("part_id"),
        "owned_region_count": len(regions),
        "capacity_deficit": deficit,
        "split_candidate_count": len(split_candidates),
        "top_split_candidate_bend_ids": [row["bend_id"] for row in split_candidates[:5]],
        "status": (
            "split_candidate_available"
            if deficit > 0 and split_candidates
            else "missing_region_recovery_needed"
            if deficit > 0
            else "no_capacity_deficit"
        ),
        "regions": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze F1 owned regions for split candidates when feature capacity is short.")
    parser.add_argument("--decomposition-json", required=True, type=Path)
    parser.add_argument("--feature-label-summary-json", type=Path, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    report = analyze_owned_region_split_candidates(
        decomposition=_load_json(args.decomposition_json),
        feature_label_summary=_load_json(args.feature_label_summary_json) if args.feature_label_summary_json else None,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "status": report["status"],
                "capacity_deficit": report["capacity_deficit"],
                "split_candidate_count": report["split_candidate_count"],
                "top_split_candidate_bend_ids": report["top_split_candidate_bend_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
