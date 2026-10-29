#!/usr/bin/env python3
"""Freeze a deterministic part-family split for F1 optimization evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "output/research/f1-corpus-evidence-20260729-v1/freeze/sealed_gold_manifest.json"
OUTPUT = ROOT / "output/research/f1-corpus-evidence-20260729-v1/freeze/grouped_development_holdout.json"
SALT = "sherman-f1-study-v1:"


def main() -> int:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    families = sorted({str(case["part_key"]) for case in gold["cases"]})
    ranked = sorted(
        (hashlib.sha256((SALT + family).encode("utf-8")).hexdigest(), family)
        for family in families
    )
    holdout = {family for _digest, family in ranked[:3]}
    cases = [
        {
            "case_token": case["case_token"],
            "part_key": case["part_key"],
            "split": "holdout" if case["part_key"] in holdout else "development",
        }
        for case in gold["cases"]
    ]
    payload = {
        "schema_version": 1,
        "split_id": "sherman-f1-family-hash-split-v1",
        "corpus_freeze_id": gold["corpus_freeze_id"],
        "rule": "three lexicographically smallest SHA256(salt + part_key) digests are holdout",
        "salt": SALT,
        "development_part_families": sorted(set(families) - holdout),
        "holdout_part_families": sorted(holdout),
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
