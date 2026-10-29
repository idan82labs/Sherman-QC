#!/usr/bin/env python3
"""Build a deterministic multi-family hidden-oracle lock without reading truth rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK_ID = "F1_QUARANTINED_MULTI_FAMILY_SEMANTIC_ORACLE_V2_20260810"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
POOL_MANIFEST = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1/manifest.json"
OUTPUT = ROOT / "docs/research/F1_QUARANTINED_MULTI_FAMILY_SEMANTIC_ORACLE_V2_LOCK_20260810.json"
EXCLUDED_PART_IDS = {"10839000", "35705010", "38172000", "47959001", "ysherman"}
EXCLUDED_CASE_TOKENS = {"C006"}
ALLOWED_TRUTH_TIERS = {
    "authoritative_user_verified",
    "codex_manual_multiview_count_backed",
    "codex_manual_multiview_drawing_backed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(*values: str) -> str:
    return hashlib.sha256("\0".join((LOCK_ID, *values)).encode("utf-8")).hexdigest()


def _scan_sha(case: dict[str, Any]) -> str:
    return str(
        ((case.get("source") or {}).get("scan") or {}).get("sha256")
        or case.get("scan_sha256")
        or ""
    )


def build() -> dict[str, Any]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    manifest = json.loads(POOL_MANIFEST.read_text(encoding="utf-8"))
    catalog_by_sha = {_scan_sha(case): case for case in catalog.get("cases") or []}
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in manifest.get("cases") or []:
        token = str(row["case_token"])
        scan_sha = str(row["source_scan_sha256"])
        case = catalog_by_sha.get(scan_sha)
        if token in EXCLUDED_CASE_TOKENS:
            skipped.append({"case_token": token, "reason": "explicit_C006_seal"})
            continue
        if case is None:
            skipped.append({"case_token": token, "reason": "candidate_catalog_scan_hash_mismatch"})
            continue
        part_id = str(case["part_id"])
        if part_id in EXCLUDED_PART_IDS:
            skipped.append({"case_token": token, "reason": "excluded_consumed_or_leakage_part"})
            continue
        if case.get("truth_tier") not in ALLOWED_TRUTH_TIERS:
            skipped.append({"case_token": token, "reason": "truth_contract_ineligible"})
            continue
        pool_path = Path(str(row["path"])).resolve()
        eligible.append(
            {
                "case_token": token,
                "case_id": str(case["case_id"]),
                "part_id": part_id,
                "truth_tier": str(case["truth_tier"]),
                "promotion_eligible": bool(case.get("promotion_eligible")),
                "scan_sha256": scan_sha,
                "candidate_pool_path": str(pool_path.relative_to(ROOT)),
                "candidate_pool_sha256": _sha256(pool_path),
                "candidate_count": int(row["candidate_count"]),
                "line_object_count": int(row["line_object_count"]),
                "within_part_rank_sha256": _rank(part_id, str(case["case_id"]), scan_sha),
            }
        )

    by_part: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_part.setdefault(row["part_id"], []).append(row)
    representatives = [
        min(rows, key=lambda item: item["within_part_rank_sha256"])
        for _part_id, rows in sorted(by_part.items())
    ]
    for row in representatives:
        row["family_rank_sha256"] = _rank(row["part_id"], row["scan_sha256"])
    representatives.sort(key=lambda item: item["family_rank_sha256"])
    if len(representatives) < 13:
        raise RuntimeError("fewer than thirteen eligible unique part families")
    development = representatives[:8]
    transfer = representatives[8:13]
    reserve = representatives[13:]
    return {
        "schema_version": 1,
        "lock_id": LOCK_ID,
        "status": "LOCKED_BEFORE_ANY_V2_ORACLE_FEEDBACK",
        "objective": "Sherman QC scan-only F1 bend-line generalization",
        "selection": {
            "rule": "ascending_sha256(lock_id, part_id, scan_sha256), one deterministically selected scan per part",
            "development_family_count": len(development),
            "one_shot_transfer_family_count": len(transfer),
            "reserve_family_count": len(reserve),
            "expected_counts_used": False,
            "truth_geometry_used": False,
            "candidate_predictions_used": False,
            "identity_routing_used": False,
        },
        "catalog": {
            "path": str(CATALOG.relative_to(ROOT)),
            "sha256": _sha256(CATALOG),
            "allowed_truth_tiers": sorted(ALLOWED_TRUTH_TIERS),
            "promotion_eligible_required": False,
            "non_promotion_tiers_are_research_only": True,
        },
        "candidate_pool_manifest": {
            "path": str(POOL_MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(POOL_MANIFEST),
        },
        "matching": {
            "axis_angle_degrees_max": 5.0,
            "transverse_offset_mm_max": 4.0,
            "gold_span_overlap_min": 0.60,
            "prediction_span_overlap_min": 0.60,
            "assignment": "one_to_one_minimum_cost",
        },
        "development_cases": development,
        "transfer_cases": transfer,
        "reserve_cases": reserve,
        "skipped_cases": sorted(skipped, key=lambda item: item["case_token"]),
        "safety": {
            "development_feedback": "coarse ownership classes and aggregate metrics only",
            "transfer_execution": "exactly one claim-bound oracle opening per locked case",
            "gold_coordinates_returned": False,
            "gold_ids_returned": False,
            "expected_counts_returned": False,
            "cad_opened": False,
            "C006_opened": False,
            "runtime_main_or_production_write": False,
            "promotion_claim_from_v2": False,
        },
    }


def main() -> int:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "development": [row["case_token"] for row in payload["development_cases"]],
                "transfer": [row["case_token"] for row in payload["transfer_cases"]],
                "reserve": [row["case_token"] for row in payload["reserve_cases"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
