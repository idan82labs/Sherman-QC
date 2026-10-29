#!/usr/bin/env python3
"""One-shot primary/replay execution of frozen V54 ENTC on the synthetic corpus."""

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

from backend.f1_exclusive_normal_turn_coverage_v54 import (
    MAXIMUM_PUBLIC_BYTES,
    PRECOMMIT_TOKEN,
    PROFILE,
    RSS_LIMIT_BYTES,
    TERMINAL_FAILURE_TOKEN,
    V54ScanAbstention,
    WALL_LIMIT_SECONDS,
    compose_v54,
)


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))


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


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _research_inputs(case_root: Path) -> Path:
    matches = sorted((case_root / "cache").glob("**/research_inputs.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one frozen research-input cache under {case_root}, found {len(matches)}")
    return matches[0]


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
        raise FileExistsError("V54 output root is not empty")

    code_paths = [
        ROOT / "backend/f1_exclusive_normal_turn_coverage_v54.py",
        ROOT / "scripts/run_f1_v54_synthetic_from_v52.py",
    ]
    frozen_dependency_paths = [
        ROOT / "backend/cad_bend_extractor.py",
        ROOT / "backend/f1_cylindrical_consensus_recovery_v52.py",
        ROOT / "research/f1_multifamily_selector_v2/model.py",
        ROOT / "scripts/build_f1_scan_only_candidate_pool.py",
        ROOT / "scripts/score_f1_v54_synthetic_stress.py",
        ROOT / "scripts/render_f1_v54_entc_source_only_audit.py",
        ROOT / "scripts/finalize_f1_v54_entc_visual_gate.py",
    ]
    pool_paths = []
    v52_paths = []
    research_paths = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            case_root = args.v51_execution_root / pass_name / case_id
            pool_paths.append(case_root / "pool" / f"{case_id}.json")
            research_paths.append(_research_inputs(case_root))
            v52_paths.append(args.v52_execution_root / pass_name / case_id / "semantic_result.json")
    scan_paths = [args.scan_root / f"{case_id}.ply" for case_id in CASE_IDS]
    for path in [*code_paths, *frozen_dependency_paths, *pool_paths, *research_paths, *v52_paths, *scan_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v54_preexecution_lock",
        "token": TOKEN,
        "precommit_token": PRECOMMIT_TOKEN,
        "profile": PROFILE,
        "code_sha256": {str(path): _sha(path) for path in code_paths},
        "frozen_dependency_sha256": {str(path): _sha(path) for path in frozen_dependency_paths},
        "candidate_pool_tree_sha256": _tree_hash(pool_paths),
        "v52_output_tree_sha256": _tree_hash(v52_paths),
        "frozen_support_record_tree_sha256": _tree_hash(research_paths),
        "source_mesh_tree_sha256": _tree_hash(scan_paths),
        "primary_replay_order": [
            [pass_name, case_id]
            for pass_name in ("primary", "replay")
            for case_id in CASE_IDS
        ],
        "compact_result_schema": "V54_ENTC_AGGREGATES_ONLY_V1",
        "maximum_public_json_bytes_per_scan": MAXIMUM_PUBLIC_BYTES,
        "maximum_wall_seconds_per_scan": WALL_LIMIT_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "truth_opened": False,
    }
    _write(args.output_root / "preexecution_lock.json", lock)

    records: list[dict[str, object]] = []
    wall_times: list[float] = []
    try:
        for pass_name in ("primary", "replay"):
            for case_id in CASE_IDS:
                case_root = args.v51_execution_root / pass_name / case_id
                pool_path = case_root / "pool" / f"{case_id}.json"
                research_path = _research_inputs(case_root)
                v52_path = args.v52_execution_root / pass_name / case_id / "semantic_result.json"
                scan_path = args.scan_root / f"{case_id}.ply"
                v52 = json.loads(v52_path.read_text(encoding="utf-8"))
                started = time.perf_counter()
                result = compose_v54(
                    v52,
                    json.loads(pool_path.read_text(encoding="utf-8")),
                    json.loads(research_path.read_text(encoding="utf-8")),
                    scan_path,
                )
                elapsed = time.perf_counter() - started
                wall_times.append(elapsed)
                prefix_count = len(v52.get("predictions") or ())
                if result.get("predictions", [])[:prefix_count] != v52.get("predictions"):
                    raise V54ScanAbstention("ABSTAIN_V52_PREFIX_MUTATION")
                output = args.output_root / pass_name / case_id / "semantic_result.json"
                _write(output, result)
                if output.stat().st_size > MAXIMUM_PUBLIC_BYTES:
                    raise V54ScanAbstention("ABSTAIN_PUBLIC_JSON_SIZE")
                audit = result["exclusive_normal_turn_coverage_audit"]
                records.append(
                    {
                        "pass": pass_name,
                        "case_id": case_id,
                        "v52_sha256": _sha(v52_path),
                        "pool_sha256": _sha(pool_path),
                        "support_record_sha256": _sha(research_path),
                        "scan_sha256": _sha(scan_path),
                        "v54_sha256": _sha(output),
                        "v52_prediction_prefix_sha256": hashlib.sha256(
                            json.dumps(v52.get("predictions") or [], sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "v54_prediction_prefix_sha256": hashlib.sha256(
                            json.dumps(result["predictions"][:prefix_count], sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "candidate_decision_sha256": audit["candidate_decision_sha256"],
                        "candidate_bin_summary_sha256": audit["candidate_bin_summary_sha256"],
                        "recovered_candidate_count": audit["recovered_candidate_count"],
                        "wall_seconds": round(elapsed, 9),
                        "public_json_bytes": output.stat().st_size,
                    }
                )
    except V54ScanAbstention as error:
        failure = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v54_terminal_execution_failure",
            "token": TOKEN,
            "terminal_failure_token": TERMINAL_FAILURE_TOKEN,
            "status": "TERMINAL_RETIRED_SCAN_ABSTENTION",
            "reason": error.reason,
            "completed_records": records,
            "truth_opened": False,
            "peak_process_rss_bytes": _rss_bytes(),
        }
        _write(args.output_root / "terminal_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2

    primary = {str(row["case_id"]): row for row in records if row["pass"] == "primary"}
    replay = {str(row["case_id"]): row for row in records if row["pass"] == "replay"}
    identity: dict[str, bool] = {}
    for case_id in CASE_IDS:
        identity[case_id] = bool(
            primary[case_id]["v54_sha256"] == replay[case_id]["v54_sha256"]
            and primary[case_id]["candidate_decision_sha256"] == replay[case_id]["candidate_decision_sha256"]
            and primary[case_id]["candidate_bin_summary_sha256"] == replay[case_id]["candidate_bin_summary_sha256"]
            and primary[case_id]["v52_prediction_prefix_sha256"] == primary[case_id]["v54_prediction_prefix_sha256"]
            and replay[case_id]["v52_prediction_prefix_sha256"] == replay[case_id]["v54_prediction_prefix_sha256"]
        )
    if not all(identity.values()):
        failure = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v54_terminal_execution_failure",
            "token": TOKEN,
            "terminal_failure_token": TERMINAL_FAILURE_TOKEN,
            "status": "TERMINAL_RETIRED_REPLAY_DISAGREEMENT",
            "semantic_replay_identity": identity,
            "truth_opened": False,
        }
        _write(args.output_root / "terminal_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 3

    ordered_times = sorted(wall_times)
    p95 = ordered_times[int(0.95 * (len(ordered_times) - 1))]
    peak_rss = _rss_bytes()
    if p95 > WALL_LIMIT_SECONDS or peak_rss >= RSS_LIMIT_BYTES:
        failure = {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v54_terminal_execution_failure",
            "token": TOKEN,
            "terminal_failure_token": TERMINAL_FAILURE_TOKEN,
            "status": "TERMINAL_RETIRED_RESOURCE_GATE",
            "p95_case_wall_seconds": p95,
            "peak_process_rss_bytes": peak_rss,
            "truth_opened": False,
        }
        _write(args.output_root / "terminal_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 4
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v54_synthetic_execution",
        "token": TOKEN,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "status": "READY_FOR_SINGLE_SCORE",
        "truth_opened": False,
        "semantic_replay_identity": identity,
        "p95_case_wall_seconds": round(p95, 9),
        "peak_process_rss_bytes": peak_rss,
        "records": records,
    }
    _write(args.output_root / "execution_manifest.json", manifest)
    print(
        json.dumps(
            {
                "case_count": len(CASE_IDS),
                "replay_identity": True,
                "recovered": sum(
                    int(row["recovered_candidate_count"])
                    for row in records
                    if row["pass"] == "primary"
                ),
                "p95_case_wall_seconds": round(p95, 9),
                "peak_process_rss_bytes": peak_rss,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
