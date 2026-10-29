#!/usr/bin/env python3
"""Promote the untouched V2 reserve into a final 4+4 V3 lock.

The split is fixed by the pre-existing reserve order: the first four cases are
development; the final four remain one-shot transfer.  Previously consumed V2
development and transfer cases are retained only as audit metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    reserve = [dict(row) for row in parent.get("reserve_cases") or []]
    if [row["case_token"] for row in reserve] != [
        "C018", "C027", "C005", "C033", "C029", "C013", "C015", "C019"
    ]:
        raise RuntimeError("parent reserve order or membership changed")
    result = {
        "schema_version": 1,
        "lock_id": "F1_QUARANTINED_MULTI_FAMILY_SEMANTIC_ORACLE_V3_20260811",
        "objective": parent["objective"],
        "parent_lock_path": str(args.parent),
        "parent_lock_sha256": sha(args.parent),
        "catalog": parent["catalog"],
        "candidate_pool_manifest": parent["candidate_pool_manifest"],
        "matching": parent["matching"],
        "development_cases": reserve[:4],
        "transfer_cases": reserve[4:],
        "reserve_cases": [],
        "consumed_parent_case_tokens": [
            row["case_token"]
            for row in [*(parent.get("development_cases") or []), *(parent.get("transfer_cases") or [])]
        ],
        "split_contract": {
            "rule": "first_four_and_final_four_in_parent_reserve_order",
            "development_feedback_may_be_used": True,
            "transfer_feedback_may_not_be_used_before_submission_freeze": True,
            "one_shot_transfer_claim_required": True,
            "rerun_or_replacement_submission_authorized": False,
        },
        "safety": {
            **parent["safety"],
            "development_feedback": "coarse ownership classes and aggregate metrics only",
            "transfer_execution": "exactly one claim-bound oracle opening per locked case",
            "promotion_claim_from_v3": False,
        },
        "c006_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "development": [row["case_token"] for row in result["development_cases"]],
        "transfer": [row["case_token"] for row in result["transfer_cases"]],
        "sha256": sha(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
