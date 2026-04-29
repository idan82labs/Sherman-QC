#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("raw_candidate_id") or candidate.get("candidate_id") or "")


def _pair_key(candidate: Mapping[str, Any]) -> Tuple[str, str]:
    pair = tuple(str(value) for value in candidate.get("best_flange_pair") or ())
    if len(pair) != 2:
        return ("unknown", "unknown")
    return tuple(sorted(pair))  # type: ignore[return-value]


def _rank(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(candidate.get("best_pair_score") or 0.0),
        float(candidate.get("linearity") or 0.0),
        float(candidate.get("span_mm") or 0.0),
        -float(candidate.get("thickness_mm") or 999.0),
    )


def _family_row(pair: Tuple[str, str], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(candidates, key=_rank, reverse=True)
    selected = ranked[0]
    suppressed = ranked[1:]
    merged_candidate = dict(selected)
    merged_candidate["raw_candidate_id"] = "MERGED_" + "_".join(_candidate_id(candidate) for candidate in ranked)
    merged_candidate["source_candidate_ids"] = [_candidate_id(candidate) for candidate in ranked]
    merged_candidate["sampled_points"] = [
        point
        for candidate in ranked
        for point in candidate.get("sampled_points") or ()
    ]
    return {
        "flange_pair": list(pair),
        "selected_candidate_id": _candidate_id(selected),
        "source_candidate_ids": [_candidate_id(candidate) for candidate in ranked],
        "selected_score": round(float(selected.get("best_pair_score") or 0.0), 6),
        "selected_span_mm": round(float(selected.get("span_mm") or 0.0), 6),
        "selected_thickness_mm": round(float(selected.get("thickness_mm") or 0.0), 6),
        "selected_linearity": round(float(selected.get("linearity") or 0.0), 6),
        "selected_candidate": selected,
        "merged_family_candidate": merged_candidate,
        "suppressed_duplicate_candidate_ids": [_candidate_id(candidate) for candidate in suppressed],
        "suppressed_duplicate_count": len(suppressed),
        "candidate_count": len(ranked),
        "candidate_ids": [_candidate_id(candidate) for candidate in ranked],
    }


def _decision(*, target_count: int | None, selected_count: int, safe_count: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if safe_count <= 0:
        return "no_safe_support", ["no_bridge_safe_support"]
    if safe_count > selected_count:
        reasons.append("duplicate_flange_pair_support_suppressed")
    if target_count is None:
        reasons.append("no_target_count_supplied")
        return "sparse_arrangement_candidate", reasons
    if selected_count == target_count:
        reasons.append("unique_flange_pair_count_matches_target")
        return "sparse_arrangement_exact_candidate", reasons
    if selected_count < target_count:
        reasons.append("unique_flange_pair_count_below_target")
        return "sparse_arrangement_underfit", reasons
    reasons.append("unique_flange_pair_count_above_target")
    return "sparse_arrangement_overcomplete", reasons


def solve_raw_crease_sparse_arrangement(
    *,
    bridge_validation: Mapping[str, Any],
    target_count: int | None = None,
) -> Dict[str, Any]:
    candidates = list(bridge_validation.get("candidates") or ())
    safe_candidates = [candidate for candidate in candidates if candidate.get("bridge_safe")]
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for candidate in safe_candidates:
        groups[_pair_key(candidate)].append(candidate)
    families = [_family_row(pair, group) for pair, group in sorted(groups.items())]
    selected_count = len(families)
    status, reason_codes = _decision(
        target_count=target_count,
        selected_count=selected_count,
        safe_count=len(safe_candidates),
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_path": bridge_validation.get("scan_path"),
        "part_id": bridge_validation.get("part_id"),
        "purpose": "Diagnostic-only sparse arrangement over bridge-safe raw crease candidates.",
        "target_count": target_count,
        "status": status,
        "reason_codes": reason_codes,
        "safe_candidate_count": len(safe_candidates),
        "selected_family_count": selected_count,
        "suppressed_duplicate_count": max(0, len(safe_candidates) - selected_count),
        "selected_candidate_ids": [row["selected_candidate_id"] for row in families],
        "selected_family_source_candidate_ids": [row["source_candidate_ids"] for row in families],
        "suppressed_duplicate_candidate_ids": [
            candidate_id
            for row in families
            for candidate_id in row["suppressed_duplicate_candidate_ids"]
        ],
        "families": families,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a sparse diagnostic arrangement over bridge-safe raw crease candidates.")
    parser.add_argument("--bridge-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--target-count", type=int, default=None)
    args = parser.parse_args()
    report = solve_raw_crease_sparse_arrangement(
        bridge_validation=_load_json(args.bridge_json),
        target_count=args.target_count,
    )
    _write_json(args.output_json, report)
    print(
        json.dumps(
            {
                "part_id": report["part_id"],
                "status": report["status"],
                "safe_candidate_count": report["safe_candidate_count"],
                "selected_family_count": report["selected_family_count"],
                "suppressed_duplicate_count": report["suppressed_duplicate_count"],
                "selected_candidate_ids": report["selected_candidate_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
