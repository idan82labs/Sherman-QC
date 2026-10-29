#!/usr/bin/env python3
"""Execute the frozen V53 ALBC primary/replay pass on the synthetic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_adjacency_lattice_bank_consensus_v53 import compose_v53


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v51-execution-root", type=Path, required=True)
    parser.add_argument("--v52-execution-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("V53 output root is not empty")

    code_paths = [
        ROOT / "backend/f1_adjacency_lattice_bank_consensus_v53.py",
        ROOT / "scripts/run_f1_v53_synthetic_from_v52.py",
    ]
    frozen_dependency_paths = [
        ROOT / "backend/cad_bend_extractor.py",
        ROOT / "backend/f1_cylindrical_consensus_recovery_v52.py",
        ROOT / "backend/plane_lattice_bend_recovery.py",
        ROOT / "scripts/build_f1_scan_only_candidate_pool.py",
        ROOT / "scripts/render_f1_source_only_prediction_audit.py",
        ROOT / "scripts/render_f1_v51_new_corpus_scored_overlay.py",
        ROOT / "scripts/score_f1_v53_synthetic_stress.py",
    ]
    pool_paths = [
        args.v51_execution_root / pass_name / case_id / "pool" / f"{case_id}.json"
        for pass_name in ("primary", "replay")
        for case_id in CASE_IDS
    ]
    v52_paths = [
        args.v52_execution_root / pass_name / case_id / "semantic_result.json"
        for pass_name in ("primary", "replay")
        for case_id in CASE_IDS
    ]
    scan_paths = [args.scan_root / f"{case_id}.ply" for case_id in CASE_IDS]
    for path in [*code_paths, *frozen_dependency_paths, *pool_paths, *v52_paths, *scan_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v53_preexecution_lock",
        "token": TOKEN,
        "precommit_token": "V53_ALBC_MECHANISM_LOCKED",
        "code_sha256": {str(path): _sha(path) for path in code_paths},
        "frozen_dependency_sha256": {
            str(path): _sha(path) for path in frozen_dependency_paths
        },
        "candidate_pool_tree_sha256": _tree_hash(pool_paths),
        "v52_output_tree_sha256": _tree_hash(v52_paths),
        "source_mesh_tree_sha256": _tree_hash(scan_paths),
        "primary_replay_order": [
            [pass_name, case_id]
            for pass_name in ("primary", "replay")
            for case_id in CASE_IDS
        ],
        "compact_result_schema": "V53_ALBC_AGGREGATES_ONLY_V1",
    }
    _write(args.output_root / "preexecution_lock.json", lock)

    records = []
    wall_times = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            v52_path = args.v52_execution_root / pass_name / case_id / "semantic_result.json"
            pool_path = args.v51_execution_root / pass_name / case_id / "pool" / f"{case_id}.json"
            scan_path = args.scan_root / f"{case_id}.ply"
            v52 = json.loads(v52_path.read_text(encoding="utf-8"))
            started = time.perf_counter()
            result = compose_v53(
                v52,
                json.loads(pool_path.read_text(encoding="utf-8")),
                scan_path,
            )
            elapsed = time.perf_counter() - started
            wall_times.append(elapsed)
            if result["predictions"][: len(v52.get("predictions") or ())] != v52.get("predictions"):
                raise RuntimeError(f"{case_id}: immutable V52 prefix changed")
            output = args.output_root / pass_name / case_id / "semantic_result.json"
            _write(output, result)
            audit = result["adjacency_lattice_bank_consensus_audit"]
            records.append(
                {
                    "case_id": case_id,
                    "pass": pass_name,
                    "v52_sha256": _sha(v52_path),
                    "pool_sha256": _sha(pool_path),
                    "scan_sha256": _sha(scan_path),
                    "v53_sha256": _sha(output),
                    "v52_prediction_prefix_sha256": hashlib.sha256(
                        json.dumps(v52.get("predictions") or [], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "v53_prediction_prefix_sha256": hashlib.sha256(
                        json.dumps(result["predictions"][: len(v52.get("predictions") or ())], sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "status": audit["status"],
                    "eligible_pair_count": audit["eligible_pair_count"],
                    "geometrically_eligible_pair_count": audit[
                        "geometrically_eligible_pair_count"
                    ],
                    "recovered_candidate_count": audit["recovered_candidate_count"],
                    "wall_seconds": round(elapsed, 9),
                }
            )
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {
        case_id: primary[case_id]["v53_sha256"] == replay[case_id]["v53_sha256"]
        for case_id in CASE_IDS
    }
    if not all(identity.values()):
        raise RuntimeError("V53 primary/replay semantic bytes differ")
    p95 = sorted(wall_times)[int(0.95 * (len(wall_times) - 1))]
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v53_synthetic_execution",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "truth_opened": False,
        "semantic_replay_identity": identity,
        "p95_case_wall_seconds": round(p95, 9),
        "peak_process_rss_bytes": peak_rss_bytes,
        "records": records,
    }
    _write(args.output_root / "execution_manifest.json", manifest)
    print(
        json.dumps(
            {
                "case_count": len(CASE_IDS),
                "replay_identity": True,
                "recovered": sum(
                    row["recovered_candidate_count"]
                    for row in records
                    if row["pass"] == "primary"
                ),
                "p95_case_wall_seconds": round(p95, 9),
                "peak_process_rss_bytes": peak_rss_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
