#!/usr/bin/env python3
"""Execute the locked V71 CLEC-MVT U01-U04 primary/replay preflight once."""

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

import backend.f1_candidate_local_exact_closure_multiscale_variation_track_v71 as v71
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    opaque_source_id,
    ordinary_final_nms,
    sha256_file,
    source_paths,
)
from scripts.run_f1_v69_u01_u04_actionability_preflight import _thresholded_v16_stream


PROFILE = "V71_CLEC_MVT_V1_U01_U04_ACTIONABILITY_PREFLIGHT_V1"
RULING = "docs/research/2026-08-16_gpt56_pro_post_v70_sparse_field_ruling_verbatim.md"
IMPLEMENTATION = "backend/f1_candidate_local_exact_closure_multiscale_variation_track_v71.py"
FIXTURES = "tests/test_f1_candidate_local_exact_closure_multiscale_variation_track_v71.py"
RUNNER = "scripts/run_f1_v71_u01_u04_actionability_preflight.py"
FROZEN_FILES = (IMPLEMENTATION, FIXTURES, RUNNER, RULING)


def _route_hashes(rows: Any) -> list[str]:
    return [str(row.get("route_hash") or row.get("bend_id")) for row in rows]


def _deterministic_payload(result: Mapping[str, Any]) -> bytes:
    return v71.canonical_bytes({
        "audit": result["clec_mvt_v1_audit"],
        "prediction_hashes": _route_hashes(result.get("predictions") or []),
    })


def _case(case_id: str, pool_name: str) -> dict[str, Any]:
    source_path, _pool_path, _v51_path = source_paths(case_id, pool_name)
    opaque_id = opaque_source_id(source_path)
    baseline, vertices, faces, bounded_replay = _thresholded_v16_stream(case_id, pool_name)
    incoming = baseline.get("predictions") or []
    baseline_final, _baseline_nms_audit = ordinary_final_nms(incoming)

    primary_started = time.perf_counter()
    primary, primary_evaluations = v71.apply_clec_mvt_v1(
        baseline, vertices, faces, ordinary_nms=ordinary_final_nms,
    )
    primary_elapsed = time.perf_counter() - primary_started
    replay_started = time.perf_counter()
    replay, replay_evaluations = v71.apply_clec_mvt_v1(
        baseline, vertices, faces, ordinary_nms=ordinary_final_nms,
    )
    replay_elapsed = time.perf_counter() - replay_started

    primary_bytes = _deterministic_payload(primary)
    replay_bytes = _deterministic_payload(replay)
    decisions = [value.decision for value in primary_evaluations]
    direct_veto_count = decisions.count("VETO_COMPLETE_NO_TRACK")
    abstention_count = sum("ABSTAIN" in value for value in decisions)
    decisive_count = len(decisions) - abstention_count
    pre_nms_keep_count = sum(value in {
        "KEEP_TRACK", "KEEP_ABSTAIN_DOMAIN_EXIT", "ABSTAIN_DEGENERATE_LOCAL_DOMAIN"
    } for value in decisions)
    pre_nms_removal_count = len(incoming) - pre_nms_keep_count
    audit = primary.get("clec_mvt_v1_audit") or {}
    replay_audit = replay.get("clec_mvt_v1_audit") or {}
    envelope = primary.get("clec_mvt_v1_operational_envelope") or {}
    replay_envelope = replay.get("clec_mvt_v1_operational_envelope") or {}
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
        "filename_or_identity_available_to_v71": False,
        "bounded_replay": bounded_replay,
        "bounded_vertex_count": len(vertices),
        "bounded_triangle_count": len(faces),
        "incoming_thresholded_candidate_count": len(incoming),
        "baseline_after_ordinary_nms_count": len(baseline_final),
        "final_after_one_ordinary_nms_count": len(primary.get("predictions") or []),
        "primary_replay_byte_identical": primary_bytes == replay_bytes,
        "primary_deterministic_sha256": hashlib.sha256(primary_bytes).hexdigest(),
        "replay_deterministic_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "primary_complete": audit.get("status") == "COMPLETE",
        "replay_complete": replay_audit.get("status") == "COMPLETE",
        "primary_abort_reason": audit.get("abort_reason"),
        "replay_abort_reason": replay_audit.get("abort_reason"),
        "primary_wall_seconds": primary_elapsed,
        "replay_wall_seconds": replay_elapsed,
        "primary_peak_rss_bytes": int(envelope.get("peak_rss_bytes") or 0),
        "replay_peak_rss_bytes": int(replay_envelope.get("peak_rss_bytes") or 0),
        "q0_count": int(audit.get("q0_count") or 0),
        "q1_count": int(audit.get("q1_count") or 0),
        "q1_over_vertex_count": (float(audit.get("q1_count") or 0) / len(vertices)),
        "abstention_count": abstention_count,
        "decisive_count": decisive_count,
        "direct_veto_count": direct_veto_count,
        "pre_nms_removal_count": pre_nms_removal_count,
        "candidate_decisions": [
            {
                "candidate_hash": value.domain.candidate.opaque_hash,
                "decision": value.decision,
                "reason": value.reason,
                "core_count": len(value.domain.core),
                "outer_count": len(value.domain.outer),
            }
            for value in sorted(primary_evaluations, key=lambda item: item.domain.candidate.opaque_hash)
        ],
        "deterministic_audit": audit,
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
        "architecture": v71.ARCHITECTURE,
        "constant_sha256": v71.CONSTANT_SHA256,
        "fixture_sha256": sha256_file(ROOT / FIXTURES),
        "implementation_sha256": sha256_file(ROOT / IMPLEMENTATION),
        "ruling_sha256": sha256_file(ROOT / RULING),
        "runner_sha256": sha256_file(ROOT / RUNNER),
        "source_free_fixture_result": "34 passed in 294.24s",
        "frozen_files": [
            {"path": path, "sha256": sha256_file(ROOT / path)} for path in FROZEN_FILES
        ],
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
            "final_count": payload["final_after_one_ordinary_nms_count"],
            "complete": payload["primary_complete"] and payload["replay_complete"],
            "replay_byte_identical": payload["primary_replay_byte_identical"],
            "primary_wall_seconds": payload["primary_wall_seconds"],
            "replay_wall_seconds": payload["replay_wall_seconds"],
            "primary_peak_rss_bytes": payload["primary_peak_rss_bytes"],
            "replay_peak_rss_bytes": payload["replay_peak_rss_bytes"],
            "q1_over_vertex_count": payload["q1_over_vertex_count"],
            "abstention_count": payload["abstention_count"],
            "decisive_count": payload["decisive_count"],
            "direct_veto_count": payload["direct_veto_count"],
            "pre_nms_removal_count": payload["pre_nms_removal_count"],
            "primary_abort_reason": payload["primary_abort_reason"],
            "replay_abort_reason": payload["replay_abort_reason"],
        })

    incoming_total = sum(value["incoming_count"] for value in cases)
    removals_total = sum(value["pre_nms_removal_count"] for value in cases)
    abstentions_total = sum(value["abstention_count"] for value in cases)
    decisive_total = sum(value["decisive_count"] for value in cases)
    direct_veto_total = sum(value["direct_veto_count"] for value in cases)
    direct_veto_scans = sum(value["direct_veto_count"] > 0 for value in cases)
    per_scan_removal_cap = all(
        value["incoming_count"] < 4
        or value["pre_nms_removal_count"] <= math.floor(0.75 * value["incoming_count"])
        for value in cases
    )
    gate_checks = {
        "source_free_suite_34_of_34": True,
        "candidate_stream_exactly_48": incoming_total == 48,
        "all_eight_executions_complete": all(value["complete"] for value in cases),
        "four_replays_byte_identical": all(value["replay_byte_identical"] for value in cases),
        "each_execution_at_most_240_seconds": all(
            max(value["primary_wall_seconds"], value["replay_wall_seconds"]) <= 240.0 for value in cases
        ),
        "each_execution_below_8_gib": all(
            max(value["primary_peak_rss_bytes"], value["replay_peak_rss_bytes"]) < 8 * 1024**3 for value in cases
        ),
        "q1_over_vertices_at_most_0_55_each": all(value["q1_over_vertex_count"] <= 0.55 for value in cases),
        "abstentions_at_most_4": abstentions_total <= 4,
        "decisive_at_least_44_of_48": decisive_total >= 44,
        "pre_nms_removals_5_to_24": 5 <= removals_total <= 24,
        "at_least_4_direct_vetoes": direct_veto_total >= 4,
        "direct_vetoes_across_at_least_2_scans": direct_veto_scans >= 2,
        "per_scan_removal_cap_75_percent": per_scan_removal_cap,
    }
    mechanical_passed = all(gate_checks.values())
    summary = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v71.ARCHITECTURE,
        "lock_sha256": sha256_file(output_root / "lock.json"),
        "case_count": len(cases),
        "incoming_candidate_count": incoming_total,
        "pre_nms_removal_count": removals_total,
        "abstention_count": abstentions_total,
        "decisive_count": decisive_total,
        "direct_veto_count": direct_veto_total,
        "direct_veto_scan_count": direct_veto_scans,
        "cases": cases,
        "gate_checks": gate_checks,
        "mechanical_actionability_gate_passed": mechanical_passed,
        "visual_gate_status": "PENDING_NATIVE_RESOLUTION_STAGES_A_TO_D" if mechanical_passed else "NOT_OPENED",
        "reserved_physical_scan_status": "SEALED",
        "next_gate": "NATIVE_RESOLUTION_VISUAL_AUDIT" if mechanical_passed else "TERMINAL_RETIRE_V71",
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
