#!/usr/bin/env python3
"""Apply source-only V52 to sealed synthetic V51 primary/replay artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_cylindrical_consensus_recovery_v52 import compose_v52


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v51-execution-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("V52 output root is not empty")
    records = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            case_root = args.v51_execution_root / pass_name / case_id
            v51_path = case_root / "semantic_result.json"
            pool_path = case_root / "pool" / f"{case_id}.json"
            result = compose_v52(
                json.loads(v51_path.read_text(encoding="utf-8")),
                json.loads(pool_path.read_text(encoding="utf-8")),
            )
            output = args.output_root / pass_name / case_id / "semantic_result.json"
            _write(output, result)
            records.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "v51_sha256": _sha(v51_path),
                    "pool_sha256": _sha(pool_path),
                    "v52_sha256": _sha(output),
                    "recovered_candidate_count": result["cylindrical_consensus_recovery_audit"]["recovered_candidate_count"],
                }
            )
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {case_id: primary[case_id]["v52_sha256"] == replay[case_id]["v52_sha256"] for case_id in CASE_IDS}
    if not all(identity.values()):
        raise RuntimeError("V52 primary/replay semantic bytes differ")
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v52_synthetic_execution",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened": False,
        "semantic_replay_identity": identity,
        "records": records,
    }
    _write(args.output_root / "execution_manifest.json", manifest)
    print(json.dumps({"case_count": 30, "replay_identity": True, "recovered": sum(row["recovered_candidate_count"] for row in records if row["pass"] == "primary")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
