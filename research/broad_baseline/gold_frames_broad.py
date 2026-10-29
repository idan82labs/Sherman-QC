#!/usr/bin/env python3
"""Frame records for every human-gold (catalog) case of a broad case list.

Reuses gold_frames.recover_case (read-only display-cloud -> raw-vertex recovery,
residual <= 0.01 mm, else excluded) for the catalog cases that are in the case
list but not in the Gold11 frame-record file, and writes ONE merged record file
(Gold11 records copied verbatim + the new ones) for score.py --frame-records.

  gold_frames_broad.py --case-list docs/production/runlog/broad_baseline/case_list_broad_v2.json \
      --gold11-records docs/production/runlog/broad_baseline/gold11_frame_recovery.json \
      --out docs/production/runlog/broad_baseline/broad_frame_records.json
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import gold_frames as gf  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--gold11-records", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=bb.GOLD_CATALOG)
    parser.add_argument("--rss-limit-gib", type=float, default=9.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = bb.read_json(args.case_list)["cases"]
    base = bb.read_json(args.gold11_records)
    have = {r["scan_sha256"] for r in base["cases"]}
    catalog = {row["scan_sha256"]: row for row in bb.read_json(args.catalog)["cases"]}
    new_records = []
    for case in cases:
        if case["gold"]["kind"] != "gold11_catalog" or case["scan_sha256"] in have:
            continue
        bb.wait_for_rss_budget(args.rss_limit_gib)
        row = catalog[case["scan_sha256"]]
        record = gf.recover_case(row, gold11_id=None)
        record["broad_case_id"] = case["case_id"]
        new_records.append(record)
        print(f"{case['case_id']} {row['case_id']:22s} {row['coordinate_frame']:18s} -> {record['status']} "
              f"method={record.get('method')} residual_max_mm={record.get('residual_max_mm')}", flush=True)
    records = list(base["cases"]) + new_records
    payload = {
        "schema": gf.FRAME_SCHEMA,
        "created_at": bb.utc_now(),
        "catalog": {"path": str(args.catalog), "sha256": bb.sha256_file(args.catalog)},
        "residual_limit_mm": bb.TRANSFORM_RESIDUAL_MAX_MM,
        "convention": base.get("convention"),
        "sources": {"gold11_records": str(args.gold11_records), "gold11_records_sha256": bb.sha256_file(args.gold11_records),
                    "new_records": [r["catalog_case_id"] for r in new_records]},
        "recovered": sum(r["status"] == "RECOVERED" for r in records),
        "excluded": [{"catalog_case_id": r["catalog_case_id"], "status": r["status"]} for r in records
                     if r["status"] != "RECOVERED"],
        "cases": records,
    }
    bb.write_json(args.out, payload)
    print(f"recovered {payload['recovered']}/{len(records)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
