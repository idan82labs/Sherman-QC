#!/usr/bin/env python3
"""Execute development-only V61 primary/replay on calibrated synthetic V1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_calibrated_birth_ownership_guard_v61 import compose_v61


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V1_NON_CONFIRMATORY"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v52-root", type=Path, required=True)
    parser.add_argument("--v60-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("V61 output root is not empty")
    records = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            v52 = json.loads((args.v52_root / pass_name / case_id / "semantic_result.json").read_text())
            v60 = json.loads((args.v60_root / pass_name / case_id / "semantic_result.json").read_text())
            points = np.asarray(o3d.io.read_point_cloud(str(args.corpus_root / "public" / case_id / "source.ply")).points)
            B = float(np.linalg.norm(np.ptp(points, axis=0)))
            result = compose_v61(v52, v60, B)
            output = args.output_root / pass_name / case_id / "semantic_result.json"
            _write(output, result)
            records.append({"case_id": case_id, "pass": pass_name, "semantic_file_sha256": _file_sha(output), **result["audit"]})
    primary = {row["case_id"]: row["semantic_file_sha256"] for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row["semantic_file_sha256"] for row in records if row["pass"] == "replay"}
    identity = {case_id: primary[case_id] == replay[case_id] for case_id in CASE_IDS}
    if not all(identity.values()):
        raise RuntimeError("V61 primary/replay differ")
    manifest = {"schema_version": 1, "kind": "sherman_qc_f1_v61_calibrated_raw_synthetic_v1_execution", "token": TOKEN, "confirmatory_eligible": False, "development_only_after_v1_truth_open": True, "semantic_replay_identity": identity, "records": records}
    _write(args.output_root / "execution_manifest.json", manifest)
    print(json.dumps({"case_count": 30, "replay_identity": True, "admitted_births": sum(row["admitted_birth_count"] for row in records if row["pass"] == "primary"), "suppressed_births": sum(row["ownership_suppressed_count"] for row in records if row["pass"] == "primary")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
