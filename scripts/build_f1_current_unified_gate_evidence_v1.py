#!/usr/bin/env python3
"""Build lane-separated unified-gate evidence from frozen summary artifacts only.

This adapter deliberately reads compact reports, never point clouds, CAD geometry,
prediction payloads, or line truth.  It converts the latest retained CAD-assisted
and scan-only development comparisons into the common gate schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_unified_gate_v1 import SCHEMA, evaluate


CAD_GPI = Path("docs/research/artifacts/2026-08-22_cad_assisted_production_v9_gpi.json")
CAD_METRICS = Path("output/research/f1-step-topology-normal-verified-v9-20260822/MULTISOURCE_GOLD11_DIAGNOSTIC.json")
CAD_SUMMARY = Path("output/research/f1-step-topology-normal-verified-v9-20260822/summary.json")
SCAN_COMPARISON = Path("output/research/f1-physical15-development-v52-20260822/GOLD12_DEVELOPMENT_COMPARISON.json")
SCAN_VISUAL = Path("output/research/f1-physical15-development-v52-20260822/NATIVE_VISUAL_AUDIT.json")
STAGE_GRAPH = Path("docs/research/artifacts/2026-08-25_f1_retained_stage_graph_v1.json")
CAD_SCORER = Path("scripts/diagnose_f1_cad_assisted_multisource_gold11.py")
SCAN_SCORER = Path("scripts/score_f1_physical15_development_gold12.py")


def _load(root: Path, relative: Path) -> Any:
    with (root / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha(root: Path, relative: Path) -> str:
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def _metrics(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "exact_count_accuracy": row.get("exact_count_accuracy"),
        "exact_localized_target_set_accuracy": row.get("exact_localized_target_set_accuracy"),
        "minimum_family_recall": row.get("minimum_family_recall"),
        "marker_admissibility": None,
    }


def build(root: Path) -> dict[str, dict[str, Any]]:
    cad_gpi = _load(root, CAD_GPI)
    cad_metrics = _load(root, CAD_METRICS)["metrics"]
    cad_summary = _load(root, CAD_SUMMARY)["typed_release_metrics"]
    scan = _load(root, SCAN_COMPARISON)
    scan_visual = _load(root, SCAN_VISUAL)

    cad_evidence = {
        "schema": SCHEMA,
        "version": "CAD_ASSISTED_PRODUCTION_V9_RETAINED_COMPARISON_20260825",
        "lane": "CAD_ASSISTED",
        "evidence_class": "CONSUMED_PHYSICAL",
        "hashes": {
            "detector": _sha(root, CAD_METRICS),
            "baseline": _sha(root, CAD_GPI),
            "scorer": _sha(root, CAD_SCORER),
            "ontology": _sha(root, STAGE_GRAPH),
            "dataset": _sha(root, CAD_METRICS),
        },
        "counts": {
            "physical_scans": int(cad_gpi["evidence"]["physical_scans"]),
            "morphology_families": int(cad_gpi["evidence"]["physical_families"]),
            "independent_instances": 0,
            "jobs_attempted": 26,
            "jobs_succeeded": 26,
        },
        "truth": {
            "independent_line_truth": False,
            "dual_review_verified": False,
            "morphology_held_out": False,
        },
        "comparison": {
            "matched_frozen_baseline": True,
            "baseline_lane": "CAD_ASSISTED",
            "candidate_lane": "CAD_ASSISTED",
        },
        "baseline_metrics": _metrics(cad_metrics["incumbent_union"]),
        "candidate_metrics": _metrics(cad_metrics["triple_union"]),
        "visual": {
            "complete": True,
            "passed": bool(cad_gpi["evidence"]["autonomous_production_visual_gate_passed"]),
            "candidate_binding": "TOPOLOGY_ONLY_V9_NOT_AGGRESSIVE_TRIPLE_UNION",
        },
        "operations": {
            "deterministic_replay_rate": 1.0 if cad_gpi["evidence"]["deterministic_replay_passed"] else 0.0,
            "p95_latency_sec": float(cad_summary["p95_pipeline_latency_sec"]),
            "peak_process_tree_rss_bytes": None,
            "job_success_rate": float(cad_summary["job_success_rate"]),
        },
        "protocol_valid": bool(cad_gpi["evidence"]["protocol_valid"]),
        "leakage_free": bool(cad_gpi["evidence"]["leakage_free"]),
    }

    baseline = scan["baseline"]["metrics"]
    candidate = scan["candidate"]["metrics"]
    operational = scan_visual["operational"]
    scan_evidence = {
        "schema": SCHEMA,
        "version": "SCAN_ONLY_V52_VS_RETAINED_V51_20260825",
        "lane": "SCAN_ONLY",
        "evidence_class": "CONSUMED_PHYSICAL",
        "hashes": {
            "detector": _sha(root, SCAN_VISUAL),
            "baseline": str(scan_visual["v51_execution_manifest_sha256"]),
            "scorer": _sha(root, SCAN_SCORER),
            "ontology": _sha(root, STAGE_GRAPH),
            "dataset": _sha(root, SCAN_COMPARISON),
        },
        "counts": {
            # Accuracy is defined on the 12 scored Gold12 cases.  D013-D015
            # were executed and visually audited, but have no line truth and
            # must not inflate the accuracy denominator or family count.
            "physical_scans": int(candidate["case_count"]),
            "morphology_families": len(candidate["family_recalls"]),
            "independent_instances": 0,
            "jobs_attempted": 30,
            "jobs_succeeded": 30,
        },
        "execution_counts": {
            "physical_scans_executed": 15,
            "physical_scans_scored": int(candidate["case_count"]),
            "physical_scans_visual_only": 15 - int(candidate["case_count"]),
        },
        "truth": {
            "independent_line_truth": False,
            "dual_review_verified": False,
            "morphology_held_out": False,
        },
        "comparison": {
            "matched_frozen_baseline": True,
            "baseline_lane": "SCAN_ONLY",
            "candidate_lane": "SCAN_ONLY",
        },
        "baseline_metrics": _metrics(baseline),
        "candidate_metrics": _metrics(candidate),
        "visual": {
            "complete": True,
            "passed": bool(scan_visual["visual_gate_passed"]),
        },
        "operations": {
            "deterministic_replay_rate": 1.0 if operational["deterministic_replay"] else 0.0,
            "p95_latency_sec": float(operational["primary_p95_elapsed_seconds"]),
            "peak_process_tree_rss_bytes": int(operational["peak_stage_rss_bytes"]),
            "job_success_rate": 1.0 if operational["all_15_primary_completed"] and operational["all_15_replay_completed"] else 0.0,
        },
        "protocol_valid": bool(scan["protocol_valid"]),
        "leakage_free": bool(scan["leakage_free_model_path"]),
    }
    return {"cad_assisted": cad_evidence, "scan_only": scan_evidence}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = build(root)
    gates = {lane: evaluate(payload) for lane, payload in evidence.items()}
    result = {
        "schema_version": 1,
        "kind": "F1_CURRENT_LANE_SEPARATED_UNIFIED_GATE",
        "evidence": evidence,
        "gates": gates,
    }
    _atomic_json(args.output, result)
    print(json.dumps({lane: {"grade": gate["grade"], "direction": gate["direction"], "operational_badge": gate["operational_badge"]} for lane, gate in gates.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
