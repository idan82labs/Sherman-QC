#!/usr/bin/env python3
"""Execute V58 on the sealed prospective calibrated synthetic V1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_calibrated_multiview_turn_consensus_v58 import compose_v58


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-ids", default="all")
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("V58 output root is not empty")
    case_ids = CASE_IDS if args.case_ids == "all" else tuple(value.strip() for value in args.case_ids.split(",") if value.strip())
    if not case_ids or any(case_id not in CASE_IDS for case_id in case_ids):
        raise ValueError("invalid opaque case IDs")

    records = []
    for pass_name in ("primary", "replay"):
        for case_id in case_ids:
            bundle = args.corpus_root / "public" / case_id
            v52_path = args.v52_root / pass_name / case_id / "semantic_result.json"
            started = time.perf_counter()
            result = compose_v58(bundle, json.loads(v52_path.read_text(encoding="utf-8")))
            elapsed = time.perf_counter() - started
            output = args.output_root / pass_name / case_id / "semantic_result.json"
            _write(output, result)
            records.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "bundle_manifest_sha256": _sha(bundle / "manifest.json"),
                    "v52_sha256": _sha(v52_path),
                    "v58_sha256": _sha(output),
                    "elapsed_seconds": elapsed,
                    **result["audit"],
                }
            )
    identity = {}
    if len(case_ids):
        primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
        replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
        identity = {case_id: primary[case_id]["v58_sha256"] == replay[case_id]["v58_sha256"] for case_id in case_ids}
        if not all(identity.values()):
            raise RuntimeError("V58 primary/replay semantic bytes differ")
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v58_calibrated_raw_synthetic_v1_execution",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened": False,
        "case_ids": list(case_ids),
        "semantic_replay_identity": identity,
        "records": records,
    }
    _write(args.output_root / "execution_manifest.json", manifest)
    print(json.dumps({"case_count": len(case_ids), "replay_identity": True, "raw_new": sum(row["raw_new_count"] for row in records if row["pass"] == "primary"), "pruned": sum(row["pruned_v52_count"] for row in records if row["pass"] == "primary")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
