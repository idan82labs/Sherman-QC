#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _scan_name(value: Any) -> str:
    return Path(str(value or "")).name


def _case_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(row.get("part_key") or ""), _scan_name(row.get("scan_name")))


def _index_decompositions(root: Path) -> Dict[Tuple[str, str], Path]:
    indexed: Dict[Tuple[str, str], Path] = {}
    for path in root.rglob("decomposition_result.json"):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        key = (str(payload.get("part_id") or ""), _scan_name(payload.get("scan_path")))
        if key[0] and key[1]:
            indexed[key] = path
    return indexed


def _span_length(region: Mapping[str, Any]) -> Optional[float]:
    endpoints = list(region.get("span_endpoints") or ())
    if len(endpoints) != 2:
        return None
    try:
        return float(math.dist(endpoints[0], endpoints[1]))
    except (TypeError, ValueError):
        return None


def _support_mass(region: Mapping[str, Any]) -> float:
    try:
        return float(region.get("support_mass") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _role_tags_for_regions(regions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    masses = [_support_mass(region) for region in regions]
    med_mass = float(median(masses)) if masses else 0.0
    sorted_masses = sorted(masses)
    q25 = sorted_masses[max(0, min(len(sorted_masses) - 1, len(sorted_masses) // 4))] if sorted_masses else 0.0
    small_mass_threshold = max(q25, med_mass * 0.35)
    pair_counts: Dict[Tuple[str, ...], int] = {}
    for region in regions:
        pair = tuple(str(value) for value in region.get("incident_flange_ids") or ())
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    rows: List[Dict[str, Any]] = []
    for region in regions:
        atom_count = len(list(region.get("owned_atom_ids") or ()))
        mass = _support_mass(region)
        pair = tuple(str(value) for value in region.get("incident_flange_ids") or ())
        duplicate_pair = bool(pair and pair_counts.get(pair, 0) > 1)
        small_support = bool(atom_count <= 6 or (small_mass_threshold > 0 and mass <= small_mass_threshold))
        span = _span_length(region)
        compact_support = bool(span is not None and span <= 25.0 and atom_count <= 8)
        tags: List[str] = []
        if small_support:
            tags.append("small_support")
        if duplicate_pair:
            tags.append("duplicate_incident_pair")
        if compact_support:
            tags.append("compact_support")
        if mass >= med_mass * 0.5 and atom_count >= 8:
            tags.append("substantial_support")

        if "substantial_support" in tags and not compact_support:
            provisional_role = "likely_conventional_bend"
        elif duplicate_pair or small_support or compact_support:
            provisional_role = "candidate_raised_form_or_fragment"
        else:
            provisional_role = "ambiguous_support"

        rows.append(
            {
                "bend_id": region.get("bend_id"),
                "provisional_role": provisional_role,
                "diagnostic_tags": tags,
                "angle_deg": region.get("angle_deg"),
                "support_mass": mass,
                "owned_atom_count": atom_count,
                "span_length_mm": None if span is None else round(float(span), 4),
                "incident_flange_ids": list(pair),
                "anchor": region.get("anchor"),
                "post_bend_class": region.get("post_bend_class"),
                "debug_confidence": region.get("debug_confidence"),
            }
        )
    rows.sort(key=lambda row: (str(row["provisional_role"]), -float(row["support_mass"]), str(row["bend_id"])))
    return rows


def analyze_feature_semantic_blockers(
    *,
    feature_semantics_audit: Mapping[str, Any],
    decomposition_root: Path,
) -> Dict[str, Any]:
    decomp_index = _index_decompositions(decomposition_root)
    blocker_reports = []
    for blocker in list(feature_semantics_audit.get("blockers") or ()):
        key = _case_key(blocker)
        decomp_path = decomp_index.get(key)
        report: Dict[str, Any] = {
            "part_key": key[0],
            "scan_name": key[1],
            "feature_semantics_status": blocker.get("feature_semantics_status"),
            "truth_conventional_bends": blocker.get("truth_conventional_bends"),
            "truth_raised_form_features": blocker.get("truth_raised_form_features"),
            "truth_total_features": blocker.get("truth_total_features"),
            "algorithm_range": blocker.get("algorithm_range"),
            "algorithm_raw_exact_field": blocker.get("algorithm_raw_exact_field"),
            "decomposition_result_path": str(decomp_path) if decomp_path else None,
            "owned_region_count": None,
            "owned_region_diagnostics": [],
            "diagnostic_summary": {},
            "recommended_next_action": "locate decomposition_result.json before geometry diagnosis",
        }
        if decomp_path is not None:
            payload = _load_json(decomp_path)
            regions = list(payload.get("owned_bend_regions") or ())
            diagnostics = _role_tags_for_regions(regions)
            counts: Dict[str, int] = {}
            for item in diagnostics:
                role = str(item.get("provisional_role"))
                counts[role] = counts.get(role, 0) + 1
            report.update(
                {
                    "owned_region_count": len(regions),
                    "owned_region_diagnostics": diagnostics,
                    "diagnostic_summary": dict(sorted(counts.items())),
                    "recommended_next_action": (
                        "review candidate_raised_form_or_fragment regions against render; "
                        "do not auto-subtract until feature-class labels are confirmed"
                    ),
                }
            )
        blocker_reports.append(report)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decomposition_root": str(decomposition_root),
        "blocker_count": len(blocker_reports),
        "blockers": blocker_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze geometry behind F1 feature-semantics blockers.")
    parser.add_argument("--feature-semantics-audit-json", required=True, type=Path)
    parser.add_argument("--decomposition-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    report = analyze_feature_semantic_blockers(
        feature_semantics_audit=_load_json(args.feature_semantics_audit_json),
        decomposition_root=args.decomposition_root,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "blocker_count": report["blocker_count"],
                "decomposition_root": report["decomposition_root"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
