#!/usr/bin/env python3
"""Source-only real-scan actionability reconnaissance for V67.

This deliberately reuses already exposed U01--U04 solely to falsify whether
V67 can act on real corpus morphology.  It is not fresh generalization evidence
and never opens pseudo-truth.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_similarity_invariant_normal_cycle_mode_match_nms_v67 import apply_ncmm_nms_v1
from scripts.f1_tftoc_v66_protocol import CASE_BINDINGS, frozen_v52, opaque_source_id, ordinary_final_nms, source_paths


OUTPUT = ROOT / "output/research/f1-ncmm-v67-real-scan-actionability-recon-v1-20260815/summary.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def main() -> int:
    cases = []
    total_eligible = total_profile = total_proposed = changed = 0
    for case_id, pool_name in CASE_BINDINGS:
        source_path, _pool_path, _v51_path = source_paths(case_id, pool_name)
        opaque_id = opaque_source_id(source_path)
        baseline, vertices, faces, _bounded_replay = frozen_v52(case_id, pool_name)
        baseline_final, _nms_audit = ordinary_final_nms(baseline["predictions"])
        started = time.perf_counter()
        result, pairs = apply_ncmm_nms_v1(baseline, vertices, faces)
        elapsed = time.perf_counter() - started
        audit = result["ncmm_nms_v1_audit"]
        reasons = Counter(pair.reason for pair in pairs)
        eligible = int(audit.get("geometry_or_exact_pair_count") or 0)
        profile = sum(1 for pair in pairs if pair.reason not in ("PAIR_GEOMETRY", "EXACT_COINCIDENT"))
        proposed = len(audit.get("selected_matching") or ())
        is_changed = proposed > 0
        total_eligible += eligible
        total_profile += profile
        total_proposed += proposed
        changed += int(is_changed)
        cases.append(
            {
                "opaque_source_id": opaque_id,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "vertex_count": len(vertices),
                "triangle_count": len(faces),
                "incoming_pre_nms_count": len(baseline["predictions"]),
                "frozen_v52_final_count": len(baseline_final),
                "status": audit["status"],
                "detail": audit.get("detail"),
                "geometry_or_exact_pair_count": eligible,
                "profile_evaluable_pair_count": profile,
                "reason_histogram": dict(sorted(reasons.items())),
                "proposed_suppression_count": proposed,
                "proposed_changed": is_changed,
                "primary_elapsed_seconds": round(elapsed, 6),
                "audit_semantic_sha256": audit.get("semantic_sha256"),
            }
        )
        print(opaque_id, cases[-1], flush=True)

    gates = {
        "at_least_two_geometry_eligible": total_eligible >= 2,
        "at_least_two_profile_evaluable": total_profile >= 2,
        "at_least_two_disjoint_proposed_suppressions": total_proposed >= 2,
        "at_least_one_changed_case": changed >= 1,
        "no_abort": all(row["status"] == "OK" for row in cases),
    }
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v67_real_scan_actionability_recon",
        "profile": "V67_SIMILARITY_INVARIANT_NORMAL_CYCLE_MODE_MATCH_NMS_V1",
        "status": "TERMINAL_RETIRE_ACTIONABILITY_ZERO" if not all(gates.values()) else "ACTIONABLE_NONQUALIFYING_RECON",
        "scientific_scope": "already_exposed_real_scan_source_only_reconnaissance_not_fresh_generalization_evidence",
        "case_count": len(cases),
        "aggregate": {
            "geometry_or_exact_pair_count": total_eligible,
            "profile_evaluable_pair_count": total_profile,
            "proposed_suppression_count": total_proposed,
            "changed_case_count": changed,
        },
        "gates": gates,
        "cases": sorted(cases, key=lambda row: row["opaque_source_id"]),
        "candidate_output_committed": False,
        "variants_skipped_after_zero_primary_actionability": total_proposed == 0,
        "visual_change_audit_required": total_proposed > 0,
        "synthetic_morphology_used": False,
        "gold_used": False,
        "labels_used": False,
        "expected_count_used": False,
        "cad_used": False,
        "pseudo_truth_opened": False,
        "new_physical_part_used": False,
    }
    payload["semantic_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_bytes(canonical(payload) + b"\n")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
