#!/usr/bin/env python3
"""Freeze exact visual selections from neutral candidate slates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="opaque_id:slate.json:selected_id_csv",
    )
    args = parser.parse_args()
    scans = []
    for value in args.case:
        opaque_id, raw_path, raw_ids = value.split(":", 2)
        slate = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        selected_ids = [item for item in raw_ids.split(",") if item]
        by_id = {str(row["bend_id"]): dict(row) for row in slate["predictions"]}
        missing = [item for item in selected_ids if item not in by_id]
        if missing:
            raise ValueError(f"{opaque_id}: unknown candidate ids: {missing}")
        manual_bends = []
        for item in selected_ids:
            row = by_id[item]
            manual_bends.append(
                {
                    "bend_id": item,
                    "candidate_id": row["candidate_id"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                }
            )
        scans.append(
            {
                "opaque_scan_id": opaque_id,
                "manual_bends": manual_bends,
                "selected_ids": selected_ids,
                "rejected_ids": sorted(set(by_id) - set(selected_ids)),
                "candidate_count": len(by_id),
                "manual_bend_count": len(manual_bends),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_native_visual_manual_precommit",
        "profile": "BOUNDED_MESH_ADJACENCY_V2_NATIVE_VISUAL_PRECOMMIT",
        "precommit_status": "frozen_before_selector_on_regenerated_pool",
        "source_only": True,
        "expected_count_opened": False,
        "selector_run": False,
        "scans": scans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
