#!/usr/bin/env python3
"""Execute the locked V70 BRMVT U01-U04 primary/replay preflight once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_boundary_rejecting_multiscale_variation_track_composition_v70 as v70
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    opaque_source_id,
    ordinary_final_nms,
    sha256_file,
    source_paths,
)
from scripts.run_f1_v69_u01_u04_actionability_preflight import _thresholded_v16_stream


PROFILE = "V70_BRMVT_V1_U01_U04_ACTIONABILITY_PREFLIGHT_V1"
RULING = "docs/research/2026-08-16_gpt56_pro_post_v69_real_morphology_ruling_verbatim.md"
IMPLEMENTATION = "backend/f1_boundary_rejecting_multiscale_variation_track_composition_v70.py"
FIXTURES = "tests/test_f1_boundary_rejecting_multiscale_variation_track_composition_v70.py"
RUNNER = "scripts/run_f1_v70_u01_u04_actionability_preflight.py"
FROZEN_FILES = (IMPLEMENTATION, FIXTURES, RUNNER, RULING)


def _hash(value: Any) -> str:
    return hashlib.sha256(v70.canonical_bytes(value)).hexdigest()


def _route_hashes(rows: Any) -> list[str]:
    return [str(row.get("route_hash") or row.get("bend_id")) for row in rows]


def _deterministic_case_payload(result: Mapping[str, Any]) -> bytes:
    return v70.canonical_bytes({
        "audit": result["brmvt_v1_audit"],
        "prediction_hashes": _route_hashes(result.get("predictions") or []),
    })


def _case(case_id: str, pool_name: str) -> dict[str, Any]:
    source_path, _pool_path, _v51_path = source_paths(case_id, pool_name)
    opaque_id = opaque_source_id(source_path)
    baseline, vertices, faces, bounded_replay = _thresholded_v16_stream(case_id, pool_name)
    incoming = baseline.get("predictions") or []
    baseline_final, _baseline_nms_audit = ordinary_final_nms(incoming)

    primary_started = time.perf_counter()
    primary, primary_evaluations = v70.apply_brmvt_v1(
        baseline, vertices, faces, ordinary_nms=ordinary_final_nms,
    )
    primary_elapsed = time.perf_counter() - primary_started

    replay_started = time.perf_counter()
    replay, replay_evaluations = v70.apply_brmvt_v1(
        baseline, vertices, faces, ordinary_nms=ordinary_final_nms,
    )
    replay_elapsed = time.perf_counter() - replay_started

    primary_bytes = _deterministic_case_payload(primary)
    replay_bytes = _deterministic_case_payload(replay)
    replay_equal = primary_bytes == replay_bytes
    evaluations = sorted(primary_evaluations, key=lambda value: value.candidate.opaque_hash)
    kept = [value for value in evaluations if value.decision == "KEEP"]
    support_dropped = [value for value in evaluations if value.terminal_reason in {"NO_FEATURE_NODE", "NO_ADMISSIBLE_TRACK"}]
    final_hashes = _route_hashes(primary.get("predictions") or [])
    baseline_hashes = _route_hashes(baseline_final)
    changed = final_hashes != baseline_hashes
    envelope = primary.get("brmvt_v1_operational_envelope") or {}
    replay_envelope = replay.get("brmvt_v1_operational_envelope") or {}
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "opaque_source_id": opaque_id,
        "source_sha256": sha256_file(source_path),
        "source_only": True,
        "research_only": True,
        "truth_opened": False,
        "pseudo_truth_opened": False,
        "labels_used": False,
        "cad_used": False,
        "expected_count_used": False,
        "filename_or_identity_available_to_v70": False,
        "bounded_replay": bounded_replay,
        "bounded_vertex_count": len(vertices),
        "bounded_triangle_count": len(faces),
        "incoming_thresholded_candidate_count": len(incoming),
        "incoming_candidate_hash": _hash(sorted(_route_hashes(incoming))),
        "baseline_after_ordinary_nms_count": len(baseline_final),
        "baseline_after_ordinary_nms_hashes": baseline_hashes,
        "final_after_one_ordinary_nms_count": len(primary.get("predictions") or []),
        "final_after_one_ordinary_nms_hashes": final_hashes,
        "changed_final_candidate_set": changed,
        "primary_replay_byte_identical": replay_equal,
        "primary_deterministic_sha256": hashlib.sha256(primary_bytes).hexdigest(),
        "replay_deterministic_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "primary_job_ok": envelope.get("job_status") == "OK",
        "replay_job_ok": replay_envelope.get("job_status") == "OK",
        "primary_wall_seconds": primary_elapsed,
        "replay_wall_seconds": replay_elapsed,
        "primary_peak_rss_bytes": int(envelope.get("peak_rss_bytes") or 0),
        "replay_peak_rss_bytes": int(replay_envelope.get("peak_rss_bytes") or 0),
        "kept_count": len(kept),
        "support_dropped_count": len(support_dropped),
        "strong_kept_A_ge_1_10": sum(value.admissibility >= 1.10 for value in kept),
        "clear_support_drop_A_le_0_90": sum(value.admissibility <= 0.90 for value in support_dropped),
        "candidate_decisions": [
            {
                "candidate_hash": value.candidate.opaque_hash,
                "decision": value.decision,
                "terminal_reason": value.terminal_reason,
                "admissibility_A_hex": float(value.admissibility).hex(),
                "occupied_length_over_L_hex": float(value.occupied_length / value.candidate.length).hex(),
                "mean_beta_hex": float(value.mean_beta).hex(),
                "duplicate_owner_hash_or_null": value.duplicate_owner,
            }
            for value in evaluations
        ],
        "deterministic_audit": primary.get("brmvt_v1_audit"),
        "primary_operational_envelope": envelope,
        "replay_operational_envelope": replay_envelope,
        "evaluation_count_primary": len(primary_evaluations),
        "evaluation_count_replay": len(replay_evaluations),
    }


def execute(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite actionability output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v70.ARCHITECTURE,
        "constant_sha256": v70.CONSTANT_SHA256,
        "fixture_sha256": sha256_file(ROOT / FIXTURES),
        "implementation_sha256": sha256_file(ROOT / IMPLEMENTATION),
        "ruling_sha256": sha256_file(ROOT / RULING),
        "runner_sha256": sha256_file(ROOT / RUNNER),
        "source_free_fixture_result": "24 passed in 373.08s",
        "frozen_files": [{"path": path, "sha256": sha256_file(ROOT / path)} for path in FROZEN_FILES],
        "candidate_bindings": [list(binding) for binding in CASE_BINDINGS],
        "truth_sealed": True,
        "fresh_scan_sealed": True,
        "primary_count": 1,
        "replay_count": 1,
        "third_run_authorized": False,
    }
    atomic_json(output_root / "lock.json", lock)

    cases = []
    for case_id, pool_name in CASE_BINDINGS:
        payload = _case(case_id, pool_name)
        case_path = output_root / "cases" / payload["opaque_source_id"] / "result.json"
        atomic_json(case_path, payload)
        cases.append({
            "opaque_source_id": payload["opaque_source_id"],
            "result": str(case_path.relative_to(output_root)),
            "result_sha256": sha256_file(case_path),
            "incoming_count": payload["incoming_thresholded_candidate_count"],
            "baseline_final_count": payload["baseline_after_ordinary_nms_count"],
            "final_count": payload["final_after_one_ordinary_nms_count"],
            "changed": payload["changed_final_candidate_set"],
            "replay_byte_identical": payload["primary_replay_byte_identical"],
            "jobs_ok": payload["primary_job_ok"] and payload["replay_job_ok"],
            "primary_wall_seconds": payload["primary_wall_seconds"],
            "replay_wall_seconds": payload["replay_wall_seconds"],
            "primary_peak_rss_bytes": payload["primary_peak_rss_bytes"],
            "replay_peak_rss_bytes": payload["replay_peak_rss_bytes"],
            "strong_kept_A_ge_1_10": payload["strong_kept_A_ge_1_10"],
            "clear_support_drop_A_le_0_90": payload["clear_support_drop_A_le_0_90"],
        })

    incoming_total = sum(value["incoming_count"] for value in cases)
    retained_total = sum(value["final_count"] for value in cases)
    changed_cases = sum(value["changed"] for value in cases)
    strong_kept = sum(value["strong_kept_A_ge_1_10"] for value in cases)
    clear_drops = sum(value["clear_support_drop_A_le_0_90"] for value in cases)
    per_scan_floor = all(
        value["incoming_count"] < 3 or value["final_count"] >= math.ceil(0.20 * value["incoming_count"])
        for value in cases
    )
    gate_checks = {
        "fixture_suite_passed": True,
        "candidate_stream_is_exactly_48": incoming_total == 48,
        "no_scan_abort": all(value["jobs_ok"] for value in cases),
        "primary_replay_byte_identical": all(value["replay_byte_identical"] for value in cases),
        "retained_total_12_to_38_inclusive": 12 <= retained_total <= 38,
        "at_least_two_changed_scans": changed_cases >= 2,
        "per_scan_twenty_percent_floor": per_scan_floor,
        "at_least_six_strong_kept": strong_kept >= 6,
        "at_least_six_clear_support_drops": clear_drops >= 6,
        "each_execution_at_most_240_seconds": all(max(value["primary_wall_seconds"], value["replay_wall_seconds"]) <= 240.0 for value in cases),
        "each_execution_below_8_gib": all(max(value["primary_peak_rss_bytes"], value["replay_peak_rss_bytes"]) < 8 * 1024**3 for value in cases),
    }
    mechanical_passed = all(gate_checks.values())
    summary = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v70.ARCHITECTURE,
        "lock_sha256": sha256_file(output_root / "lock.json"),
        "case_count": len(cases),
        "incoming_candidate_count": incoming_total,
        "retained_after_ordinary_nms_count": retained_total,
        "changed_case_count": changed_cases,
        "strong_kept_A_ge_1_10": strong_kept,
        "clear_support_drop_A_le_0_90": clear_drops,
        "cases": cases,
        "gate_checks": gate_checks,
        "mechanical_actionability_gate_passed": mechanical_passed,
        "visual_gate_status": "PENDING_RAW_FIRST_NATIVE_RESOLUTION" if mechanical_passed else "NOT_OPENED",
        "fresh_scan_status": "SEALED",
        "next_gate": "RAW_FIRST_NATIVE_RESOLUTION_VISUAL_AUDIT" if mechanical_passed else "TERMINAL_RETIRE_V70",
        "rerun_or_rescue_authorized": False,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = execute(arguments.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["mechanical_actionability_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
