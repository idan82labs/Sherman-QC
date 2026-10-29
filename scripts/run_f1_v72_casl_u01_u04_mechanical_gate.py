#!/usr/bin/env python3
"""Run the frozen V72 CASL source-free lock and U01-U04 mechanical gate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_candidate_centered_annular_secant_law_v72 as v72
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    opaque_source_id,
    sha256_file,
    source_paths,
)
from scripts.run_f1_v69_u01_u04_actionability_preflight import _thresholded_v16_stream


PROFILE = "V72_CASL_V1_U01_U04_MECHANICAL_GATE_V1"
RULING = "docs/research/2026-08-16_gpt56_pro_post_v71_successor_ruling_verbatim.md"
IMPLEMENTATION = "backend/f1_candidate_centered_annular_secant_law_v72.py"
FIXTURES = "tests/test_f1_candidate_centered_annular_secant_law_v72.py"
RUNNER = "scripts/run_f1_v72_casl_u01_u04_mechanical_gate.py"
FROZEN_FILES = (IMPLEMENTATION, FIXTURES, RUNNER, RULING)
REQUIRED_COUNTS = {"U01": 13, "U02": 20, "U03": 3, "U04": 12}
SUPPRESSION_CEILINGS = {"U01": 4, "U02": 6, "U03": 1, "U04": 4}


def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    tests = int(root.attrib.get("tests", 0))
    failures = int(root.attrib.get("failures", 0))
    errors = int(root.attrib.get("errors", 0))
    skipped = int(root.attrib.get("skipped", 0))
    if root.tag == "testsuites" and tests == 0:
        suites = list(root)
        tests = sum(int(value.attrib.get("tests", 0)) for value in suites)
        failures = sum(int(value.attrib.get("failures", 0)) for value in suites)
        errors = sum(int(value.attrib.get("errors", 0)) for value in suites)
        skipped = sum(int(value.attrib.get("skipped", 0)) for value in suites)
    return tests, failures, errors, skipped


def _candidate_hash(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_opaque_hash") or row.get("route_hash")
    if value is None:
        raise RuntimeError("frozen handoff row lacks opaque hash")
    return str(value)


def _endpoint_bytes(row: Mapping[str, Any]) -> bytes:
    return v72.canonical_bytes({"line_start": row["line_start"], "line_end": row["line_end"]})


def _case(case_id: str, pool_name: str, output_root: Path) -> dict[str, Any]:
    source_path, _pool_path, _v51_path = source_paths(case_id, pool_name)
    opaque_id = opaque_source_id(source_path)
    baseline, vertices, faces, bounded_replay = _thresholded_v16_stream(case_id, pool_name)
    incoming = [dict(row) for row in baseline.get("predictions") or ()]

    primary_started = time.perf_counter()
    primary, primary_evaluations = v72.apply_casl_v1(baseline, vertices, faces)
    primary_elapsed = time.perf_counter() - primary_started
    replay_started = time.perf_counter()
    replay, replay_evaluations = v72.apply_casl_v1(baseline, vertices, faces)
    replay_elapsed = time.perf_counter() - replay_started

    primary_artifact = v72.canonical_bytes(primary["casl_v1_audit"])
    replay_artifact = v72.canonical_bytes(replay["casl_v1_audit"])
    artifact_root = output_root / "cases" / opaque_id
    primary_sha = v72.commit_canonical_artifact(artifact_root / "primary.scientific.json", primary_artifact)
    replay_sha = v72.commit_canonical_artifact(artifact_root / "replay.scientific.json", replay_artifact)

    primary_by_hash = {value.candidate.opaque_hash: value for value in primary_evaluations}
    replay_by_hash = {value.candidate.opaque_hash: value for value in replay_evaluations}
    incoming_by_hash = {_candidate_hash(row): row for row in incoming}
    survivor_by_hash = {_candidate_hash(row): row for row in primary.get("predictions") or ()}
    suppressed = sorted(hash_value for hash_value, value in primary_by_hash.items() if value.decision == v72.DECISION_SUPPRESS)
    retained = sorted(hash_value for hash_value, value in primary_by_hash.items() if value.decision != v72.DECISION_SUPPRESS)
    endpoint_preserved = all(
        _endpoint_bytes(survivor_by_hash[hash_value]) == _endpoint_bytes(incoming_by_hash[hash_value])
        for hash_value in survivor_by_hash
    )
    suppression_comparator_valid = all(
        primary_by_hash[hash_value].summary["positive_count"] == 0
        and int(primary_by_hash[hash_value].summary["anti_total"]) >= 13
        and int(primary_by_hash[hash_value].summary["dominant_anti_count"]) >= 11
        and int(primary_by_hash[hash_value].summary["dominant_anti_run"]) >= 8
        for hash_value in suppressed
    )
    audit = primary.get("casl_v1_audit") or {}
    replay_audit = replay.get("casl_v1_audit") or {}
    envelope = primary.get("casl_v1_operational_envelope") or {}
    replay_envelope = replay.get("casl_v1_operational_envelope") or {}
    counters = audit.get("work_counters") or {}
    audit_lower = primary_artifact.lower()
    no_source_material = not any(
        token in audit_lower
        for token in (b"source_coordinate", b"source_index", b"point_id", b"shell_membership", b"triangle", b"normal_vector")
    )
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "opaque_source_id": opaque_id,
        "source_sha256": sha256_file(source_path),
        "source_only": True,
        "research_only": True,
        "truth_opened": False,
        "labels_used": False,
        "cad_used": False,
        "expected_count_used": False,
        "filename_or_identity_available_to_v72": False,
        "bounded_replay": bounded_replay,
        "bounded_vertex_count": len(vertices),
        "incoming_candidate_count": len(incoming),
        "required_candidate_count": REQUIRED_COUNTS[case_id],
        "primary_complete": audit.get("status") == "COMPLETE",
        "replay_complete": replay_audit.get("status") == "COMPLETE",
        "primary_abort_reason": audit.get("abort_reason"),
        "replay_abort_reason": replay_audit.get("abort_reason"),
        "evaluation_count_primary": len(primary_evaluations),
        "evaluation_count_replay": len(replay_evaluations),
        "primary_replay_byte_identical": primary_artifact == replay_artifact,
        "primary_scientific_sha256": primary_sha,
        "replay_scientific_sha256": replay_sha,
        "primary_wall_seconds": primary_elapsed,
        "replay_wall_seconds": replay_elapsed,
        "primary_peak_rss_bytes": int(envelope.get("peak_rss_bytes") or 0),
        "replay_peak_rss_bytes": int(replay_envelope.get("peak_rss_bytes") or 0),
        "primary_result_bytes": len(primary_artifact),
        "replay_result_bytes": len(replay_artifact),
        "model_window_evaluation_count": sum(value.model_window_evaluated for value in primary_evaluations),
        "suppression_count": len(suppressed),
        "suppression_ceiling": SUPPRESSION_CEILINGS[case_id],
        "suppressed_hashes": suppressed,
        "retained_hashes": retained,
        "final_hash_subset": set(survivor_by_hash).issubset(incoming_by_hash),
        "endpoint_bytes_preserved": endpoint_preserved,
        "suppression_comparator_valid": suppression_comparator_valid,
        "no_source_material_in_scientific_artifact": no_source_material,
        "work_counters": counters,
        "candidate_decisions": [
            {
                "candidate_hash": value.candidate.opaque_hash,
                "decision": value.decision,
                "station_classes": list(value.station_classes),
                "selected_model_windows": list(value.selected_codes),
                "summary": dict(value.summary),
                "model_window_evaluated": value.model_window_evaluated,
            }
            for value in sorted(primary_evaluations, key=lambda item: item.candidate.opaque_hash)
        ],
        "replay_decisions_identical": {
            hash_value: (
                primary_by_hash[hash_value].decision == replay_by_hash[hash_value].decision
                and primary_by_hash[hash_value].station_classes == replay_by_hash[hash_value].station_classes
            )
            for hash_value in sorted(primary_by_hash)
        },
    }


def execute(output_root: Path, source_free_junit: Path) -> Mapping[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite V72 output: {output_root}")
    tests, failures, errors, skipped = _junit_counts(source_free_junit)
    if (tests, failures, errors, skipped) != (43, 0, 0, 0):
        raise RuntimeError(
            f"V72 source-free lock is not 43/43 clean: tests={tests}, failures={failures}, errors={errors}, skipped={skipped}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    junit_copy = output_root / "source_free_lock.junit.xml"
    junit_copy.write_bytes(source_free_junit.read_bytes())
    lock = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v72.ARCHITECTURE,
        "constant_sha256": v72.CONSTANT_SHA256,
        "source_free_fixture_result": "43 passed",
        "source_free_junit_sha256": sha256_file(junit_copy),
        "frozen_files": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in FROZEN_FILES
        ],
        "candidate_bindings": [list(binding) for binding in CASE_BINDINGS],
        "required_candidate_counts": REQUIRED_COUNTS,
        "suppression_ceilings": SUPPRESSION_CEILINGS,
        "truth_sealed": True,
        "reserved_physical_scan_sealed": True,
        "primary_count": 1,
        "replay_count": 1,
        "third_run_authorized": False,
        "rescue_or_constant_edit_authorized": False,
    }
    atomic_json(output_root / "lock.json", lock)

    case_payloads = []
    case_summaries = []
    for case_id, pool_name in CASE_BINDINGS:
        payload = _case(case_id, pool_name, output_root)
        case_path = output_root / "cases" / payload["opaque_source_id"] / "result.json"
        atomic_json(case_path, payload)
        case_payloads.append(payload)
        case_summaries.append(
            {
                "opaque_source_id": payload["opaque_source_id"],
                "result": str(case_path.relative_to(output_root)),
                "result_sha256": sha256_file(case_path),
                "incoming_count": payload["incoming_candidate_count"],
                "complete": payload["primary_complete"] and payload["replay_complete"],
                "replay_byte_identical": payload["primary_replay_byte_identical"],
                "model_window_evaluation_count": payload["model_window_evaluation_count"],
                "suppression_count": payload["suppression_count"],
                "suppression_ceiling": payload["suppression_ceiling"],
                "primary_wall_seconds": payload["primary_wall_seconds"],
                "replay_wall_seconds": payload["replay_wall_seconds"],
                "primary_peak_rss_bytes": payload["primary_peak_rss_bytes"],
                "replay_peak_rss_bytes": payload["replay_peak_rss_bytes"],
            }
        )

    incoming_total = sum(value["incoming_candidate_count"] for value in case_payloads)
    model_total = sum(value["model_window_evaluation_count"] for value in case_payloads)
    suppression_total = sum(value["suppression_count"] for value in case_payloads)
    gate_checks = {
        "source_free_lock_43_of_43": True,
        "required_per_scan_decision_counts": all(
            value["incoming_candidate_count"] == value["required_candidate_count"]
            and value["evaluation_count_primary"] == value["required_candidate_count"]
            and value["evaluation_count_replay"] == value["required_candidate_count"]
            for value in case_payloads
        ),
        "candidate_stream_exactly_48": incoming_total == 48,
        "all_eight_runs_complete": all(value["primary_complete"] and value["replay_complete"] for value in case_payloads),
        "four_replays_byte_identical": all(value["primary_replay_byte_identical"] for value in case_payloads),
        "all_replay_decisions_identical": all(all(value["replay_decisions_identical"].values()) for value in case_payloads),
        "zero_graph_mesh_q0_closure_counters": all(
            all(
                int(value["work_counters"].get(key, 0)) == 0
                for key in (
                    "graph_vertices_visited", "graph_edges_visited", "mesh_adjacency_reads", "q0_count", "closure_count"
                )
            )
            for value in case_payloads
        ),
        "at_least_36_model_window_evaluations": model_total >= 36,
        "total_suppressions_1_to_15": 1 <= suppression_total <= 15,
        "per_scan_suppression_ceilings": all(
            value["suppression_count"] <= value["suppression_ceiling"] for value in case_payloads
        ),
        "every_suppression_satisfies_0_13_11_8": all(value["suppression_comparator_valid"] for value in case_payloads),
        "hash_subset_and_endpoint_bytes_unchanged": all(
            value["final_hash_subset"] and value["endpoint_bytes_preserved"] for value in case_payloads
        ),
        "each_run_at_most_300_seconds": all(
            max(value["primary_wall_seconds"], value["replay_wall_seconds"]) <= 300.0 for value in case_payloads
        ),
        "each_run_below_16_gib": all(
            max(value["primary_peak_rss_bytes"], value["replay_peak_rss_bytes"]) < 16 * 1024**3
            for value in case_payloads
        ),
        "result_bytes_at_most_262144": all(
            max(value["primary_result_bytes"], value["replay_result_bytes"]) <= 262_144 for value in case_payloads
        ),
        "no_source_material_in_artifacts": all(value["no_source_material_in_scientific_artifact"] for value in case_payloads),
        "truth_and_reserved_scan_remained_sealed": all(
            not value["truth_opened"] and not value["labels_used"] and not value["cad_used"] and not value["expected_count_used"]
            for value in case_payloads
        ),
    }
    passed = all(gate_checks.values())
    summary = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v72.ARCHITECTURE,
        "lock_sha256": sha256_file(output_root / "lock.json"),
        "case_count": len(case_payloads),
        "incoming_candidate_count": incoming_total,
        "model_window_evaluation_count": model_total,
        "suppression_count": suppression_total,
        "cases": case_summaries,
        "gate_checks": gate_checks,
        "mechanical_gate_passed": passed,
        "visual_gate_status": "PENDING_BLIND_NATIVE_RESOLUTION_AUDIT" if passed else "NOT_OPENED",
        "reserved_physical_scan_status": "SEALED",
        "next_gate": "BLIND_NATIVE_RESOLUTION_VISUAL_AUDIT" if passed else "TERMINAL_RETIRE_V72",
        "rerun_or_rescue_authorized": False,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-free-junit", type=Path, required=True)
    arguments = parser.parse_args()
    summary = execute(arguments.output_root, arguments.source_free_junit)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["mechanical_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
