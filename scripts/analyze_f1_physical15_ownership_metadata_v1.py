#!/usr/bin/env python3
"""Measure whether frozen physical-pool candidates retain explicit surface ownership."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_f1_against_manual_gold import Thresholds, _endpoints, is_match, pair_metrics


BIRTH_PAIR = re.compile(r"(?:^|_)F(\d+)_F(\d+)(?:_|$)")
MATCH_THRESHOLDS = Thresholds()


def _pair(values: Any, prefix: str) -> str | None:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    left, right = sorted(str(item) for item in values)
    if not left or not right or left == right:
        return None
    return f"{prefix}:{left}|{right}"


def ownership_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("canonical_flange_pair", "incident_flange_ids", "seed_incident_flange_ids"):
        value = _pair(row.get(field), "FLANGE")
        if value:
            keys.add(value)
    for values in row.get("candidate_incident_flange_pairs") or []:
        value = _pair(values, "FLANGE")
        if value:
            keys.add(value)
    value = _pair(row.get("incident_plane_ids"), "PLANE")
    if value:
        keys.add(value)
    value = _pair([row.get("surface1_id"), row.get("surface2_id")], "SURFACE")
    if value and "None" not in value:
        keys.add(value)
    for seed in row.get("seed_ids") or []:
        match = BIRTH_PAIR.search(str(seed))
        if match:
            keys.add(_pair([f"F{match.group(1)}", f"F{match.group(2)}"], "FLANGE"))
    return {item for item in keys if item is not None}


def analyze_case(pool: dict[str, Any], parent: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    candidates_by_id = {str(row.get("candidate_id")): row for row in pool["candidates"] if row.get("candidate_id")}
    rows = []
    for line in pool["line_objects"]:
        members = [candidates_by_id[item] for item in line.get("cluster_member_ids") or [] if item in candidates_by_id]
        keys = ownership_keys(line)
        for member in members:
            keys.update(ownership_keys(member))
        matched_gold = [
            str(bend.get("bend_id"))
            for bend in gold["bends"]
            if is_match(pair_metrics(_endpoints(bend), _endpoints(line)), MATCH_THRESHOLDS)
        ]
        rows.append({
            "bend_id": str(line.get("bend_id")),
            "source": str(line.get("representative_source") or "unknown"),
            "ownership_keys": sorted(keys),
            "ownership_key_count": len(keys),
            "matched_gold_ids": matched_gold,
        })
    selected_source_ids = {str(row.get("source_hypothesis_id")) for row in parent["predictions"]}
    selected = [row for row, line in zip(rows, pool["line_objects"]) if str(line.get("candidate_id")) in selected_source_ids]
    matched = [row for row in rows if row["matched_gold_ids"]]
    return {
        "line_object_count": len(rows),
        "line_objects_with_any_ownership": sum(row["ownership_key_count"] > 0 for row in rows),
        "line_objects_with_unique_ownership": sum(row["ownership_key_count"] == 1 for row in rows),
        "line_objects_with_ambiguous_ownership": sum(row["ownership_key_count"] > 1 for row in rows),
        "gold_matching_line_object_count": len(matched),
        "gold_matching_with_any_ownership": sum(row["ownership_key_count"] > 0 for row in matched),
        "gold_matching_with_unique_ownership": sum(row["ownership_key_count"] == 1 for row in matched),
        "selected_line_object_count": len(selected),
        "selected_with_any_ownership": sum(row["ownership_key_count"] > 0 for row in selected),
        "source_coverage": {
            source: {
                "count": sum(row["source"] == source for row in rows),
                "with_any_ownership": sum(row["source"] == source and row["ownership_key_count"] > 0 for row in rows),
            }
            for source in sorted({row["source"] for row in rows})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--gold-catalog", type=Path, required=True)
    parser.add_argument("--v51-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = json.loads(args.public_manifest.read_text())
    catalog = json.loads(args.gold_catalog.read_text())
    gold_by_sha = {str(row["scan_sha256"]): row for row in catalog["cases"]}
    cases = []
    for public_row in public["cases"]:
        gold = gold_by_sha.get(str(public_row["source_sha256"]))
        if gold is None or gold.get("truth_tier") != "codex_manual_multiview_count_backed":
            continue
        case_id = str(public_row["opaque_case_id"])
        case_root = args.v51_root / "primary" / case_id
        pool = json.loads((case_root / "pool" / f"{case_id}.json").read_text())
        parent = json.loads((case_root / "semantic_result.json").read_text())
        cases.append({"case_id": case_id, **analyze_case(pool, parent, gold)})
    totals = {
        key: sum(int(row[key]) for row in cases)
        for key in (
            "line_object_count", "line_objects_with_any_ownership", "line_objects_with_unique_ownership",
            "line_objects_with_ambiguous_ownership", "gold_matching_line_object_count",
            "gold_matching_with_any_ownership", "gold_matching_with_unique_ownership",
            "selected_line_object_count", "selected_with_any_ownership",
        )
    }
    result = {
        "schema": "F1_PHYSICAL15_OWNERSHIP_METADATA_AUDIT_V1",
        "evidence_class": "CONSUMED_PHYSICAL_DEVELOPMENT_DIAGNOSTIC",
        "case_count": len(cases),
        "cases": cases,
        "totals": totals,
        "ownership_coverage_fraction": totals["line_objects_with_any_ownership"] / totals["line_object_count"],
        "gold_matching_ownership_coverage_fraction": totals["gold_matching_with_any_ownership"] / max(totals["gold_matching_line_object_count"], 1),
        "selected_ownership_coverage_fraction": totals["selected_with_any_ownership"] / max(totals["selected_line_object_count"], 1),
        "promotion_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print(json.dumps({"case_count": len(cases), **totals, "gold_matching_ownership_coverage_fraction": result["gold_matching_ownership_coverage_fraction"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

