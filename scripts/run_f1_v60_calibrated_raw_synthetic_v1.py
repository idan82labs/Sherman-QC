#!/usr/bin/env python3
"""Execute V60 primary/replay on calibrated synthetic V1."""

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

from backend.f1_calibrated_per_view_segment_consensus_v60 import compose_v60


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("V60 output root is not empty")
    records = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            bundle = args.corpus_root / "public" / case_id
            v52_path = args.v52_root / pass_name / case_id / "semantic_result.json"
            started = time.perf_counter()
            result = compose_v60(bundle, json.loads(v52_path.read_text()))
            output = args.output_root / pass_name / case_id / "semantic_result.json"
            _write(output, result)
            records.append({"case_id": case_id, "pass": pass_name, "v60_sha256": _sha(output), "elapsed_seconds": time.perf_counter() - started, **result["audit"]})
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {case_id: primary[case_id]["v60_sha256"] == replay[case_id]["v60_sha256"] for case_id in CASE_IDS}
    if not all(identity.values()):
        raise RuntimeError("V60 primary/replay differ")
    manifest = {"schema_version": 1, "kind": "sherman_qc_f1_v60_calibrated_raw_synthetic_v1_execution", "token": TOKEN, "confirmatory_eligible": False, "truth_opened_by_inference": False, "semantic_replay_identity": identity, "records": records}
    _write(args.output_root / "execution_manifest.json", manifest)
    print(json.dumps({"case_count": 30, "replay_identity": True, "raw_new": sum(row["raw_new_count"] for row in records if row["pass"] == "primary"), "suppressed_matches": sum(row["suppressed_v52_match_count"] for row in records if row["pass"] == "primary")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
