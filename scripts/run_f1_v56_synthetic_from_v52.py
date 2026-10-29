#!/usr/bin/env python3
"""One-shot primary/replay execution of frozen V56 CTMLR V1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any


for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_canonical_turn_median_line_reconstruction_v56 import (
    DUPLICATE_DIRECTION_DEGREES,
    DUPLICATE_OVERLAP,
    MAXIMUM_DIRECTION_DRIFT_DEGREES,
    MAXIMUM_INPUT_ROWS,
    MAXIMUM_INTERSECTION_SEGMENTS,
    MAXIMUM_TAIL_RMS,
    MAXIMUM_TURN_DEGREES,
    MAXIMUM_TURN_IQR_OVER_W,
    MAXIMUM_WALL_SECONDS,
    MINIMUM_CERTIFIED_STATIONS,
    MINIMUM_LONGEST_RUN,
    MINIMUM_PROMINENCE_DEGREES,
    MINIMUM_STATION_COVERAGE,
    MINIMUM_TURN_DEGREES,
    MINIMUM_TURN_PURITY,
    PRECOMMIT_TOKEN,
    PROFILE,
    RESULT_KIND,
    RSS_LIMIT_BYTES,
    STATION_FRACTIONS,
    TERMINAL_TOKEN,
    MeshData,
    SeedSchemaError,
    compose_v56,
    materialize_seed,
)


TOKEN = "SYNTHETIC_LIDAR_STRESS_V2_NON_CONFIRMATORY"
CASE_IDS = tuple(f"S{index:03d}" for index in range(1, 31))
SOURCE_FREE_KIND = "sherman_qc_f1_v56_source_free_preflight"


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


def _semantic_view(result: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(result, allow_nan=False))
    audit = result["canonical_turn_median_line_reconstruction_audit"]
    audit.pop("wall_ms", None)
    audit.pop("peak_rss_bytes", None)
    return result


def _schema_audit_record(
    mesh: MeshData,
    v52: dict[str, Any],
    pool: dict[str, Any],
) -> dict[str, Any]:
    rows: list[tuple[dict[str, Any], str, int, int]] = []
    for row_ordinal, row in enumerate(v52.get("predictions") or ()):
        rows.append((row, "V52", 0, row_ordinal))
    for row_ordinal, row in enumerate(pool.get("line_objects") or ()):
        rows.append((row, "POOL", 1, row_ordinal))
    if len(rows) > MAXIMUM_INPUT_ROWS:
        raise SeedSchemaError("INPUT_ROW_CAP_EXCEEDED")
    schema_counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    for row, origin_class, origin_file_ordinal, row_ordinal in rows:
        seed = materialize_seed(
            row,
            bbox_diagonal=mesh.bbox_diagonal,
            median_mesh_edge=mesh.median_mesh_edge,
            origin_class=origin_class,
            origin_file_ordinal=origin_file_ordinal,
            row_ordinal=row_ordinal,
        )
        schema_counts[seed.schema_code] += 1
        normalized.append(
            {
                "direction": seed.direction.tolist(),
                "midpoint": seed.midpoint.tolist(),
                "span": seed.span,
                "line_start": seed.line_start.tolist(),
                "line_end": seed.line_end.tolist(),
                "origin_class": seed.origin_class,
                "origin_file_ordinal": seed.origin_file_ordinal,
                "row_ordinal": seed.row_ordinal,
                "schema_code": seed.schema_code,
            }
        )
    return {
        "input_row_count": len(rows),
        "materialized_row_count": len(normalized),
        "malformed_row_count": 0,
        "schema_counts": dict(sorted(schema_counts.items())),
        "normalized_sha256": hashlib.sha256(_canonical(normalized)).hexdigest(),
        "bbox_diagonal": mesh.bbox_diagonal,
        "median_mesh_edge": mesh.median_mesh_edge,
        "all_finite": True,
    }


def _terminal(output_root: Path, status: str, reason: str, *, completed_records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> int:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_terminal_execution_failure",
        "token": TOKEN,
        "terminal_token": TERMINAL_TOKEN,
        "status": status,
        "reason": reason,
        "completed_record_count": len(completed_records),
        "completed_records": completed_records,
        "truth_opened": False,
        "peak_process_rss_bytes": _rss_bytes(),
    }
    if extra:
        payload.update(extra)
    _write(output_root / "terminal_failure.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v52-execution-root", type=Path, required=True)
    parser.add_argument("--pool-execution-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--source-free-preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit synthetic engineering token required")
    if not args.source_free_preflight.is_file():
        raise FileNotFoundError(args.source_free_preflight)
    source_free = json.loads(args.source_free_preflight.read_text(encoding="utf-8"))
    if source_free.get("kind") != SOURCE_FREE_KIND or source_free.get("status") != "PASS" or source_free.get("precommit_token") != PRECOMMIT_TOKEN:
        raise ValueError("V56 source-free preflight is not a complete PASS")
    expected_preflight_hashes = {
        "backend": _sha(ROOT / "backend/f1_canonical_turn_median_line_reconstruction_v56.py"),
        "source_free_preflight": _sha(ROOT / "scripts/run_f1_v56_source_free_preflight.py"),
    }
    if source_free.get("implementation_sha256") != expected_preflight_hashes:
        raise ValueError("V56 implementation changed after source-free preflight")
    args.output_root.mkdir(parents=True, exist_ok=True)
    allowed_existing = {args.source_free_preflight.resolve()}
    unexpected = [path for path in args.output_root.iterdir() if path.resolve() not in allowed_existing]
    if unexpected:
        raise FileExistsError(f"V56 output root contains unexpected files: {unexpected}")

    code_paths = [
        ROOT / "backend/f1_canonical_turn_median_line_reconstruction_v56.py",
        ROOT / "scripts/run_f1_v56_source_free_preflight.py",
        ROOT / "scripts/run_f1_v56_synthetic_from_v52.py",
        ROOT / "scripts/score_f1_v56_synthetic_stress.py",
        ROOT / "scripts/render_f1_v56_ctmlr_source_only_audit.py",
        ROOT / "scripts/finalize_f1_v56_ctmlr_visual_gate.py",
        ROOT / "tests/test_f1_canonical_turn_median_line_reconstruction_v56.py",
    ]
    pool_paths: list[Path] = []
    v52_paths: list[Path] = []
    source_paths: list[Path] = []
    ordered_input_hashes: list[dict[str, Any]] = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            pool_path = args.pool_execution_root / pass_name / case_id / "pool" / f"{case_id}.json"
            v52_path = args.v52_execution_root / pass_name / case_id / "semantic_result.json"
            source_path = args.scan_root / f"{case_id}.ply"
            for path in (pool_path, v52_path, source_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            pool_paths.append(pool_path)
            v52_paths.append(v52_path)
            source_paths.append(source_path)
            ordered_input_hashes.append(
                {
                    "pass": pass_name,
                    "case_id": case_id,
                    "source_sha256": _sha(source_path),
                    "v52_sha256": _sha(v52_path),
                    "pool_sha256": _sha(pool_path),
                }
            )
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    fixed_constants = {
        "station_fractions": STATION_FRACTIONS,
        "maximum_input_rows": MAXIMUM_INPUT_ROWS,
        "maximum_intersection_segments": MAXIMUM_INTERSECTION_SEGMENTS,
        "minimum_turn_degrees": MINIMUM_TURN_DEGREES,
        "maximum_turn_degrees": MAXIMUM_TURN_DEGREES,
        "minimum_peak_prominence_degrees": MINIMUM_PROMINENCE_DEGREES,
        "minimum_signed_turn_purity": MINIMUM_TURN_PURITY,
        "maximum_normalized_tail_rms": MAXIMUM_TAIL_RMS,
        "maximum_turn_iqr_over_W": MAXIMUM_TURN_IQR_OVER_W,
        "minimum_certified_stations": MINIMUM_CERTIFIED_STATIONS,
        "minimum_station_coverage": MINIMUM_STATION_COVERAGE,
        "minimum_longest_station_run": MINIMUM_LONGEST_RUN,
        "maximum_fitted_line_seed_angle_degrees": MAXIMUM_DIRECTION_DRIFT_DEGREES,
        "duplicate_direction_degrees": DUPLICATE_DIRECTION_DEGREES,
        "duplicate_overlap": DUPLICATE_OVERLAP,
        "maximum_wall_seconds_per_scan": MAXIMUM_WALL_SECONDS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "dynamic_scale_formulas": {
            "minimum_seed_span": "max(12h,0.015B)",
            "W": "clamp(20h,0.012B,0.035B)",
            "delta": "clamp(1.5h,0.0005B,0.002B)",
            "l1": "clamp(4h,0.0025B,0.005B)",
            "l2": "clamp(8h,0.005B,0.010B)",
            "l3": "clamp(16h,0.010B,0.020B)",
            "rho": "max(3delta,0.002B)",
            "tau_f": "max(3h,0.002B)",
            "tau_c": "max(2h,0.0015B)",
            "tau_dup": "max(4h,0.003B)",
        },
    }
    lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_preexecution_lock",
        "token": TOKEN,
        "precommit_token": PRECOMMIT_TOKEN,
        "terminal_token": TERMINAL_TOKEN,
        "profile": PROFILE,
        "implementation_sha256": {str(path): _sha(path) for path in code_paths},
        "fixture_sha256": _sha(args.source_free_preflight),
        "fixed_constants": fixed_constants,
        "schema_rules": {
            "supported_complete_encodings": ["E", "S", "A"],
            "precedence": ["E", "S", "A"],
            "origin_file_ordinals": {"V52": 0, "POOL": 1},
            "partial_redundant_fields_must_be_consistent": True,
            "malformed_span_endpoints_terminal": True,
            "all_rows_required": True,
        },
        "primary_replay_order": [[pass_name, case_id] for pass_name in ("primary", "replay") for case_id in CASE_IDS],
        "ordered_input_hashes": ordered_input_hashes,
        "source_mesh_tree_sha256": _tree_hash(source_paths),
        "v52_output_tree_sha256": _tree_hash(v52_paths),
        "union_pool_tree_sha256": _tree_hash(pool_paths),
        "output_schema": {
            "semantic_result_kind": RESULT_KIND,
            "compact_cache_fields": [
                "version", "input_sha256", "bbox_diagonal", "median_mesh_edge", "schema_counts",
                "input_row_count", "exact_duplicate_count", "canonical_seed_count",
                "candidate_rejection_counts", "proposal_count", "output_count", "outputs", "wall_ms", "peak_rss_bytes",
            ],
            "source_geometry_serialization_forbidden": True,
        },
        "visualizer_sha256": _sha(ROOT / "scripts/render_f1_v56_ctmlr_source_only_audit.py"),
        "truth_opened": False,
    }
    _write(args.output_root / "preexecution_lock.json", lock)

    completed_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    try:
        for pass_name in ("primary", "replay"):
            for case_id in CASE_IDS:
                pool_path = args.pool_execution_root / pass_name / case_id / "pool" / f"{case_id}.json"
                v52_path = args.v52_execution_root / pass_name / case_id / "semantic_result.json"
                source_path = args.scan_root / f"{case_id}.ply"
                mesh = MeshData.load(source_path)
                v52 = json.loads(v52_path.read_text(encoding="utf-8"))
                pool = json.loads(pool_path.read_text(encoding="utf-8"))
                record = _schema_audit_record(mesh, v52, pool)
                audit_records.append({"pass": pass_name, "case_id": case_id, **record})
    except (SeedSchemaError, Exception) as error:
        reason = error.reason if isinstance(error, SeedSchemaError) else f"{type(error).__name__}:{error}"
        _write(
            args.output_root / "schema_audit.json",
            {"schema_version": 1, "profile": PROFILE, "status": "FAIL", "reason": reason, "records": audit_records},
        )
        return _terminal(args.output_root, "TERMINAL_RETIRED_SCHEMA_AUDIT", reason, completed_records=completed_records)

    primary_audit = {row["case_id"]: {key: value for key, value in row.items() if key not in ("pass", "case_id")} for row in audit_records if row["pass"] == "primary"}
    replay_audit = {row["case_id"]: {key: value for key, value in row.items() if key not in ("pass", "case_id")} for row in audit_records if row["pass"] == "replay"}
    identity = {case_id: _canonical(primary_audit[case_id]) == _canonical(replay_audit[case_id]) for case_id in CASE_IDS}
    audit_payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_total_schema_audit",
        "profile": PROFILE,
        "status": "PASS" if all(identity.values()) else "FAIL",
        "case_identity": identity,
        "records": audit_records,
    }
    _write(args.output_root / "schema_audit.json", audit_payload)
    if not all(identity.values()):
        return _terminal(args.output_root, "TERMINAL_RETIRED_SCHEMA_AUDIT", "PRIMARY_REPLAY_SCHEMA_AUDIT_MISMATCH", completed_records=completed_records)
    if any(row["materialized_row_count"] != row["input_row_count"] or row["malformed_row_count"] != 0 or not row["all_finite"] for row in audit_records):
        return _terminal(args.output_root, "TERMINAL_RETIRED_SCHEMA_AUDIT", "INCOMPLETE_SCHEMA_COVERAGE", completed_records=completed_records)

    if {str(path): _sha(path) for path in code_paths} != lock["implementation_sha256"]:
        return _terminal(args.output_root, "TERMINAL_RETIRED_HASH_MISMATCH", "IMPLEMENTATION_CHANGED_AFTER_SCHEMA_AUDIT", completed_records=completed_records)

    wall_times: list[float] = []
    try:
        for pass_name in ("primary", "replay"):
            for case_id in CASE_IDS:
                pool_path = args.pool_execution_root / pass_name / case_id / "pool" / f"{case_id}.json"
                v52_path = args.v52_execution_root / pass_name / case_id / "semantic_result.json"
                source_path = args.scan_root / f"{case_id}.ply"
                expected_hashes = next(row for row in ordered_input_hashes if row["pass"] == pass_name and row["case_id"] == case_id)
                if _sha(source_path) != expected_hashes["source_sha256"] or _sha(v52_path) != expected_hashes["v52_sha256"] or _sha(pool_path) != expected_hashes["pool_sha256"]:
                    raise RuntimeError("INPUT_HASH_MISMATCH")
                started = time.perf_counter()
                mesh = MeshData.load(source_path)
                result, _ = compose_v56(
                    json.loads(v52_path.read_text(encoding="utf-8")),
                    json.loads(pool_path.read_text(encoding="utf-8")),
                    mesh,
                    include_visual_audit=False,
                )
                elapsed = time.perf_counter() - started
                wall_times.append(elapsed)
                if elapsed >= MAXIMUM_WALL_SECONDS:
                    raise RuntimeError("SCAN_WALL_TIME_LIMIT")
                if _rss_bytes() >= RSS_LIMIT_BYTES:
                    raise RuntimeError("RSS_LIMIT")
                cache = result["canonical_turn_median_line_reconstruction_audit"]
                cache["input_sha256"] = {
                    "source": expected_hashes["source_sha256"],
                    "v52": expected_hashes["v52_sha256"],
                    "pool": expected_hashes["pool_sha256"],
                }
                output = args.output_root / pass_name / case_id / "semantic_result.json"
                cache_path = args.output_root / pass_name / case_id / "compact_cache.json"
                _write(output, result)
                _write(cache_path, cache)
                semantic_sha = hashlib.sha256(_canonical(_semantic_view(result))).hexdigest()
                prediction_sha = hashlib.sha256(_canonical(result["predictions"])).hexdigest()
                completed_records.append(
                    {
                        "pass": pass_name,
                        "case_id": case_id,
                        "source_sha256": expected_hashes["source_sha256"],
                        "v52_sha256": expected_hashes["v52_sha256"],
                        "pool_sha256": expected_hashes["pool_sha256"],
                        "semantic_result_sha256": _sha(output),
                        "deterministic_semantic_sha256": semantic_sha,
                        "prediction_sha256": prediction_sha,
                        "compact_cache_sha256": _sha(cache_path),
                        "output_count": len(result["predictions"]),
                        "wall_seconds": elapsed,
                        "peak_process_rss_bytes": _rss_bytes(),
                    }
                )
    except Exception as error:
        reason = error.reason if hasattr(error, "reason") else f"{type(error).__name__}:{error}"
        return _terminal(args.output_root, "TERMINAL_RETIRED_EXECUTION_FAILURE", reason, completed_records=completed_records)

    if len(completed_records) != 60:
        return _terminal(args.output_root, "TERMINAL_RETIRED_INCOMPLETE_EXECUTION", "EXPECTED_60_COMPLETED_RECORDS", completed_records=completed_records)
    primary = {row["case_id"]: row for row in completed_records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in completed_records if row["pass"] == "replay"}
    replay_identity = {
        case_id: bool(
            primary[case_id]["deterministic_semantic_sha256"] == replay[case_id]["deterministic_semantic_sha256"]
            and primary[case_id]["prediction_sha256"] == replay[case_id]["prediction_sha256"]
        )
        for case_id in CASE_IDS
    }
    if not all(replay_identity.values()):
        return _terminal(
            args.output_root,
            "TERMINAL_RETIRED_REPLAY_MISMATCH",
            "PRIMARY_REPLAY_BYTE_DIFFERENCE",
            completed_records=completed_records,
            extra={"semantic_replay_identity": replay_identity},
        )
    ordered_times = sorted(wall_times)
    p95 = ordered_times[int(0.95 * (len(ordered_times) - 1))]
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v56_synthetic_execution",
        "token": TOKEN,
        "profile": PROFILE,
        "evidence_class": "engineering_only_non_confirmatory",
        "confirmatory_eligible": False,
        "status": "READY_FOR_SINGLE_SCORE",
        "truth_opened": False,
        "completed_execution_count": len(completed_records),
        "semantic_replay_identity": replay_identity,
        "p95_case_wall_seconds": p95,
        "peak_process_rss_bytes": max(int(row["peak_process_rss_bytes"]) for row in completed_records),
        "records": completed_records,
    }
    _write(args.output_root / "execution_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "completed": 60, "replay_identity": True, "p95": p95}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
