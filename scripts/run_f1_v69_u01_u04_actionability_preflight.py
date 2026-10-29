#!/usr/bin/env python3
"""Run the frozen V69 U01-U04 source-only shadow actionability preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_candidate_aligned_multiscale_normal_turn_persistence_v69 as v69
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    load_identity,
    opaque_source_id,
    ordinary_final_nms,
    sha256_file,
    source_paths,
)
from scripts.select_f1_mesh_route_v16 import PROFILE as V16_PROFILE, build_result, select_unified
from research.f1_multifamily_selector_v2.model import endpoints


PROFILE = "V69_CAMNTP_V1_U01_U04_SHADOW_ACTIONABILITY_V1"
RULING = "docs/research/2026-08-15_gpt56_pro_post_v68_candidate_validity_ruling_verbatim.md"
V9_MODEL = "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib"
V4_MODEL = "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib"
FROZEN_FILES = (
    "backend/f1_candidate_aligned_multiscale_normal_turn_persistence_v69.py",
    "tests/test_f1_candidate_aligned_multiscale_normal_turn_persistence_v69.py",
    "scripts/run_f1_v69_u01_u04_actionability_preflight.py",
    RULING,
    V9_MODEL,
    V4_MODEL,
)


def _route_hashes(rows: Any) -> list[str]:
    return [str(row.get("route_hash") or row.get("bend_id")) for row in rows]


def _hash(value: Any) -> str:
    return hashlib.sha256(v69.canonical_bytes(value)).hexdigest()


def _thresholded_v16_stream(case_id: str, pool_name: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Reconstruct the frozen V16 score-threshold stream before ordinary NMS.

    The 0.45 V9 threshold yields 13/20/0/12 candidates on U01-U04.  U03's
    frozen no-output retry contributes its three already-frozen V16 rows,
    producing the professor-bound 48-row stream (13/20/3/12).
    """

    vertices, faces, pool, loaded = load_identity(case_id, pool_name)
    v9 = joblib.load(ROOT / V9_MODEL)
    v4 = joblib.load(ROOT / V4_MODEL)
    selection = select_unified(pool, v9, v4)
    scores = np.asarray(selection["v9_scores"], dtype=np.float64)
    threshold = float(v9.get("score_threshold", 0.45))
    indices = [int(index) for index in np.flatnonzero(scores >= threshold)]
    rows = []
    if indices:
        for index in indices:
            line_object = selection["rows"][index]
            line = endpoints(line_object)
            source_id = str(line_object["candidate_id"])
            digest = hashlib.sha256((V16_PROFILE + "\0" + source_id).encode()).hexdigest()[:20]
            bend_id = f"TRV16-{digest}"
            score = round(float(scores[index]), 12)
            rows.append({
                "bend_id": bend_id,
                "route_hash": hashlib.sha256(bend_id.encode("utf-8")).hexdigest()[:32],
                "better_first_key": [-score, bend_id],
                "line_start": line[0].tolist(),
                "line_end": line[1].tolist(),
                "source_hypothesis_id": source_id,
                "source_kind": line_object.get("representative_source"),
                "selection_route": f"{selection['route']}_thresholded_pre_nms",
                "selector_score": score,
            })
    else:
        # Frozen V16 no-output retry path; no alternate threshold is invented.
        frozen = build_result(pool, selection)
        for original in frozen["predictions"]:
            row = dict(original)
            bend_id = str(row["bend_id"])
            row["route_hash"] = hashlib.sha256(bend_id.encode("utf-8")).hexdigest()[:32]
            row["better_first_key"] = [-float(row.get("selector_score") or 0.0), bend_id]
            rows.append(row)
    rows.sort(key=lambda row: str(row["bend_id"]))
    baseline = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v16_thresholded_pre_nms_stream",
        "profile": V16_PROFILE,
        "predictions": rows,
        "candidate_count": len(rows),
        "threshold": threshold,
        "no_output_retry_used": not bool(indices),
        "source_only": True,
        "research_only": True,
        "labels_used": False,
        "cad_used": False,
        "expected_count_used": False,
    }
    return baseline, vertices, faces, loaded["bounded_replay"]


def _case(case_id: str, pool_name: str) -> dict[str, Any]:
    source_path, _pool_path, _v51_path = source_paths(case_id, pool_name)
    opaque_id = opaque_source_id(source_path)
    baseline, vertices, faces, bounded_replay = _thresholded_v16_stream(case_id, pool_name)
    rows = baseline.get("predictions") or []
    started = time.perf_counter()
    primary, primary_evaluations = v69.apply_camntp_v1(
        baseline, vertices, faces, ordinary_nms=ordinary_final_nms,
    )
    replay, replay_evaluations = v69.apply_camntp_v1(
        baseline, vertices, faces, ordinary_nms=ordinary_final_nms,
    )
    elapsed = time.perf_counter() - started
    primary_record = primary["camntp_v1_decision_record"]
    replay_record = replay["camntp_v1_decision_record"]
    replay_equal = v69.canonical_bytes(primary_record) == v69.canonical_bytes(replay_record)
    replay_equal = replay_equal and _route_hashes(primary.get("predictions") or []) == _route_hashes(replay.get("predictions") or [])
    decisions = {
        evaluation.candidate.opaque_hash: evaluation.decision
        for evaluation in primary_evaluations
    }
    vetoed = [
        dict(evaluation.candidate.original)
        for evaluation in sorted(primary_evaluations, key=lambda row: row.candidate.opaque_hash)
        if evaluation.decision == "VETO"
    ]
    retained = [
        dict(evaluation.candidate.original)
        for evaluation in sorted(primary_evaluations, key=lambda row: row.candidate.opaque_hash)
        if evaluation.decision == "KEEP"
    ]
    envelope = primary["camntp_v1_operational_envelope"]
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "opaque_source_id": opaque_id,
        "source_sha256": sha256_file(source_path),
        "source_only": True,
        "research_only": True,
        "shadow_only": True,
        "modified_detector_output_committed": False,
        "truth_opened": False,
        "pseudo_truth_opened": False,
        "labels_used": False,
        "cad_used": False,
        "expected_count_used": False,
        "filename_or_identity_available_to_v69": False,
        "bounded_replay": bounded_replay,
        "bounded_vertex_count": len(vertices),
        "bounded_triangle_count": len(faces),
        "incoming_thresholded_candidate_count": len(rows),
        "incoming_candidate_hash": _hash(sorted(decisions)),
        "preprocess_status": primary_record.get("preprocess_status"),
        "scan_decision_status": primary_record.get("scan_decision_status"),
        "decision_sha256": primary_record.get("decision_sha256"),
        "replay_decision_sha256": replay_record.get("decision_sha256"),
        "replay_byte_identical": replay_equal,
        "veto_count": len(vetoed),
        "retain_count": len(retained),
        "vetoed_candidates": vetoed,
        "retained_candidates": retained,
        "final_after_one_ordinary_nms_count": len(primary.get("predictions") or []),
        "final_after_one_ordinary_nms_hashes": _route_hashes(primary.get("predictions") or []),
        "ordinary_nms_call_count_primary": envelope.get("ordinary_nms_call_count"),
        "decision_record": primary_record,
        "operational_envelope": envelope,
        "primary_plus_replay_wall_seconds": elapsed,
        "primary_job_ok": primary_record.get("preprocess_status") == "OK",
        "replay_job_ok": replay_record.get("preprocess_status") == "OK",
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
        "architecture": v69.ARCHITECTURE,
        "constant_sha256": v69.CONSTANT_SHA256,
        "fixture_sha256": v69.FIXTURE_SHA256,
        "implementation_sha256": sha256_file(ROOT / FROZEN_FILES[0]),
        "ruling_sha256": sha256_file(ROOT / RULING),
        "frozen_files": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in FROZEN_FILES
        ],
        "candidate_bindings": [list(binding) for binding in CASE_BINDINGS],
        "truth_sealed": True,
        "execution_authorized": True,
        "fresh_scan_authorized": False,
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
            "candidate_count": payload["incoming_thresholded_candidate_count"],
            "veto_count": payload["veto_count"],
            "retain_count": payload["retain_count"],
            "replay_byte_identical": payload["replay_byte_identical"],
            "job_ok": payload["primary_job_ok"] and payload["replay_job_ok"],
            "primary_wall_microseconds": payload["operational_envelope"].get("wall_time_microseconds"),
            "peak_rss_bytes": payload["operational_envelope"].get("peak_rss_bytes"),
        })
    candidate_count = sum(row["candidate_count"] for row in cases)
    veto_count = sum(row["veto_count"] for row in cases)
    changed_cases = sum(row["veto_count"] > 0 for row in cases)
    mechanical_gate = (
        candidate_count == 48
        and veto_count >= 2
        and changed_cases >= 2
        and veto_count <= 24
        and all(row["replay_byte_identical"] and row["job_ok"] for row in cases)
        and all(row["primary_wall_microseconds"] <= 120_000_000 for row in cases)
        and all(row["peak_rss_bytes"] < v69.MAX_TOTAL_RSS_BYTES for row in cases)
    )
    summary = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v69.ARCHITECTURE,
        "lock_sha256": sha256_file(output_root / "lock.json"),
        "case_count": len(cases),
        "candidate_count": candidate_count,
        "veto_count": veto_count,
        "retain_count": candidate_count - veto_count,
        "changed_case_count": changed_cases,
        "cases": cases,
        "mechanical_actionability_gate_passed": mechanical_gate,
        "visual_gate_status": "PENDING_NATIVE_4096_AUDIT" if mechanical_gate else "NOT_OPENED",
        "fresh_scan_status": "SEALED",
        "next_gate": "NATIVE_4096_VISUAL_AUDIT" if mechanical_gate else "TERMINAL_RETIRE_V69",
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = execute(args.output_root.resolve())
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["mechanical_actionability_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
