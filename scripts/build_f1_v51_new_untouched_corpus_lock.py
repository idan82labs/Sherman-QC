#!/usr/bin/env python3
"""Seal the V51 new-corpus source-free precommit exactly once."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED"
LOCK_PATH = ROOT / "research/f1_v51_new_untouched_corpus_lock_v2.json"
V1_LOCK_PATH = ROOT / "research/f1_v51_new_untouched_corpus_lock.json"
EVENTS_PATH = ROOT / "output/research/f1-v51-new-untouched-corpus-20260814/invocation_events.jsonl"

PRE_EXPOSURE_PACKET_PATHS = {
    "backend/f1_elongated_lane_consolidation_v46.py",
    "backend/f1_route_independent_elongated_lane_v47.py",
    "backend/f1_nested_partial_duplicate_nms_v48.py",
    "backend/f1_short_span_cross_source_duplicate_nms_v49.py",
    "backend/f1_longest_incumbent_ownership_v50.py",
    "backend/f1_single_axis_parallel_return_recovery_v51.py",
    "scripts/run_f1_v47_from_v45.py",
    "scripts/run_f1_v48_from_v47.py",
    "scripts/run_f1_v49_from_v48.py",
    "scripts/run_f1_v50_from_v44.py",
    "scripts/run_f1_v51_from_v49.py",
    "tests/test_f1_elongated_lane_consolidation_v46.py",
    "tests/test_f1_route_independent_elongated_lane_v47.py",
    "tests/test_f1_nested_partial_duplicate_nms_v48.py",
    "tests/test_f1_short_span_cross_source_duplicate_nms_v49.py",
    "tests/test_f1_longest_incumbent_ownership_v50.py",
    "tests/test_f1_single_axis_parallel_return_recovery_v51.py",
}

SOURCE_PATHS = (
    "backend/ai_analyzer.py",
    "backend/bend_detector.py",
    "backend/bend_inspection_pipeline.py",
    "backend/bend_matcher.py",
    "backend/cad_bend_extractor.py",
    "backend/cad_dimension_extractor.py",
    "backend/cad_import.py",
    "backend/dimension_parser.py",
    "backend/dimension_report.py",
    "backend/f1_axial_causal_precedence_v37.py",
    "backend/f1_candidate_prism_geometry.py",
    "backend/f1_collinear_fragment_stitch.py",
    "backend/f1_corridor_completion.py",
    "backend/f1_cylindrical_band_completion.py",
    "backend/f1_deterministic_paired_return_completion_v45.py",
    "backend/f1_distributed_harmonic_fold_flux.py",
    "backend/f1_elongated_lane_consolidation_v46.py",
    "backend/f1_final_line_accounting_v39.py",
    "backend/f1_fragment_tolerant_parent_bridge_v41.py",
    "backend/f1_line_object_clustering.py",
    "backend/f1_line_topology_completion.py",
    "backend/f1_longest_incumbent_ownership_v50.py",
    "backend/f1_nested_partial_duplicate_nms_v48.py",
    "backend/f1_observed_projective_phase_reversal_v35.py",
    "backend/f1_observed_projective_phase_reversal_v36.py",
    "backend/f1_obtuse_parent_bridge_v40.py",
    "backend/f1_ordered_projective_gauss_transport.py",
    "backend/f1_paired_return_completion.py",
    "backend/f1_route_independent_elongated_lane_v47.py",
    "backend/f1_sharp_transition_route_certificate_v38.py",
    "backend/f1_short_span_cross_source_duplicate_nms_v49.py",
    "backend/f1_single_axis_parallel_return_recovery_v51.py",
    "backend/f1_single_shell_parent_bridge_v42.py",
    "backend/f1_strip_detector_completion.py",
    "backend/f1_two_sided_parent_support_guard_v43.py",
    "backend/f1_v34_artifact_scalar_normalizer.py",
    "backend/f1_v43_fragment_stitch_v44.py",
    "backend/feature_detection/bend_detector.py",
    "backend/material_prompts.py",
    "backend/multi_model/exchange_schema.py",
    "backend/patch_graph_latent_decomposition.py",
    "backend/plane_lattice_bend_recovery.py",
    "backend/qc_engine.py",
    "backend/scan_family_router.py",
    "backend/zero_shot_bend_overlay.py",
    "backend/zero_shot_bend_profile.py",
    "domains/bend/services/runtime_semantics.py",
    "research/f1_multifamily_selector_v2/model.py",
    "scripts/audit_f1_v51_new_untouched_corpus_preflight.py",
    "scripts/build_f1_scan_only_candidate_pool.py",
    "scripts/evaluate_f1_against_manual_gold.py",
    "scripts/finalize_f1_v51_new_corpus_gate.py",
    "scripts/prepare_f1_v51_new_corpus_visual_audit.py",
    "scripts/render_f1_v51_axis_normal_stack.py",
    "scripts/render_f1_v51_new_corpus_prediction_only.py",
    "scripts/render_f1_v51_new_corpus_scored_overlay.py",
    "scripts/render_manual_bend_review.py",
    "scripts/run_f1_v42_v16_real_scan_compatibility.py",
    "scripts/run_f1_v43_v16_real_scan_compatibility.py",
    "scripts/run_f1_v44_from_v43.py",
    "scripts/run_f1_v47_from_v45.py",
    "scripts/run_f1_v48_from_v47.py",
    "scripts/run_f1_v49_from_v48.py",
    "scripts/run_f1_v50_from_v44.py",
    "scripts/run_f1_v51_from_v49.py",
    "scripts/run_f1_v51_new_untouched_corpus_blind_once.py",
    "scripts/run_isolated_eyal_f1_baseline.py",
    "scripts/run_patch_graph_f1_prototype.py",
    "scripts/score_f1_v51_new_untouched_corpus.py",
    "scripts/select_f1_bounded_mesh_router_v9.py",
    "scripts/select_f1_full_mesh_consensus_v10.py",
    "scripts/select_f1_full_mesh_consensus_v11.py",
    "scripts/select_f1_mesh_route_v12.py",
    "scripts/select_f1_mesh_route_v13.py",
    "scripts/select_f1_mesh_route_v14.py",
    "scripts/select_f1_mesh_route_v15.py",
    "scripts/select_f1_mesh_route_v16.py",
    "scripts/select_f1_topology_bounded_mesh_v7.py",
    "scripts/select_f1_topology_coverage_v5.py",
    "scripts/select_f1_topology_plane_coverage_v6.py",
    "scripts/train_f1_bounded_mesh_router_v9.py",
    "scripts/train_f1_topology_ranker_v3.py",
    "scripts/train_f1_topology_recall_v4.py",
    "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib",
    "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib",
    "research/f1_v51_new_untouched_corpus_dependencies.json",
    "research/f1_v51_new_untouched_corpus_detector.sb",
    "docs/research/2026-08-14_f1_v51_new_untouched_corpus_custody_protocol.md",
    "docs/research/2026-08-14_f1_v51_new_corpus_precommit_v1_invalidation.md",
    "docs/research/2026-08-14_gpt56_sol_pro_post_v51_protocol_invalidation_ruling_verbatim.md",
    "tests/test_build_professor_f1_v51_post_prospective_package_20260814.py",
    "tests/test_f1_elongated_lane_consolidation_v46.py",
    "tests/test_f1_route_independent_elongated_lane_v47.py",
    "tests/test_f1_nested_partial_duplicate_nms_v48.py",
    "tests/test_f1_short_span_cross_source_duplicate_nms_v49.py",
    "tests/test_f1_longest_incumbent_ownership_v50.py",
    "tests/test_f1_single_axis_parallel_return_recovery_v51.py",
    "tests/test_run_isolated_eyal_f1_baseline.py",
    "tests/test_run_isolated_f1_blind_ply_benchmark.py",
    "tests/test_f1_fragment_tolerant_parent_bridge_v41.py",
    "tests/test_run_f1_v41_v16_real_scan_compatibility.py",
    "tests/test_f1_single_shell_parent_bridge_v42.py",
    "tests/test_f1_v43_fragment_stitch_v44.py",
    "tests/test_f1_deterministic_paired_return_completion_v45.py",
    "tests/test_select_f1_mesh_route_v16.py",
    "tests/test_run_f1_v51_new_untouched_corpus_blind_once.py",
    "tests/test_score_f1_v51_new_untouched_corpus.py",
    "tests/test_finalize_f1_v51_new_corpus_gate.py",
    "tests/test_render_f1_v51_axis_normal_stack.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anchor(path: str) -> str:
    if path in PRE_EXPOSURE_PACKET_PATHS:
        return "pre_exposure_v51_packet_manifest"
    if path == "scripts/build_f1_scan_only_candidate_pool.py":
        return "pre_exposure_20260812_candidate_generator_report"
    if path == "scripts/select_f1_mesh_route_v16.py":
        return "pre_exposure_v16_records_and_dependency_locks"
    if path in {
        "backend/f1_two_sided_parent_support_guard_v43.py",
        "backend/f1_v43_fragment_stitch_v44.py",
    }:
        return "pre_exposure_v42_v45_action_cycle"
    if path.endswith("router_v9.joblib") or path.endswith("selector_v4.joblib"):
        return "pre_exposure_v16_result_model_hashes"
    if path == "scripts/evaluate_f1_against_manual_gold.py":
        return "pre_exposure_filesystem_identity_20260716_plus_incident_packet_copy"
    return "frozen_after_ruling_before_new_corpus_assembly"


def main() -> int:
    if LOCK_PATH.exists():
        raise FileExistsError(f"lock already exists: {LOCK_PATH}")
    if not V1_LOCK_PATH.is_file():
        raise FileNotFoundError("the immutable V1 lock must be preserved before V2 sealing")
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot seal missing paths: {missing}")
    bindings = [
        {"path": path, "sha256": _sha(ROOT / path), "anchor": _anchor(path)}
        for path in SOURCE_PATHS
    ]
    lock = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_new_untouched_corpus_blind_once_lock",
        "token": TOKEN,
        "status": "LOCKED_BEFORE_NEW_CORPUS_ASSEMBLY",
        "lock_revision": 2,
        "supersedes_invalid_preflight_lock": {
            "path": str(V1_LOCK_PATH.relative_to(ROOT)),
            "sha256": _sha(V1_LOCK_PATH),
            "reason": "preflight launcher omitted repository root from sys.path; import failed before corpus assembly or detector invocation",
        },
        "research_only": True,
        "gold37_permanently_closed": True,
        "new_corpus_detector_invocation_count": 0,
        "pre_exposure_packet": {
            "path": "output/professor_package/Sherman_QC_F1_V51_Post_Prospective_20260814_v1.zip",
            "sha256": "03cfa7e568de5b4ff533a7544ebed84186e5eae27b69c4ff24c253d2d4272be8",
        },
        "pre_exposure_manifest": {
            "path": "output/professor_package/Sherman_QC_F1_V51_Post_Prospective_20260814_v1/MANIFEST.json",
            "sha256": "39b60b156b91ac210e291beb47bedd84d6218100fefc1ba85a3921ca541414b0",
        },
        "model_bundles": {
            "v9_path": "output/research/f1-new-parts-20260812/v9/model/router_v9.joblib",
            "v9_sha256": "ac48c8b9ed48d352ca5a125137a6128a73f793cef012c967eb3a998e460a39ae",
            "v4_path": "output/research/f1-multifamily-semantic-oracle-v3/selector_v4/selector_v4.joblib",
            "v4_sha256": "412946c78a63d8550f9abf1d476c94dcf01b7ed1215cfef28053a6dfd7c966f1",
            "feature_count": 76,
        },
        "composition": ["V16", "V43", "V44", "V50", "V47", "V48", "V49", "V51"],
        "case_contract": {"case_count": 30, "family_count": 15, "scans_per_family": 2},
        "matcher": {"axis_angle_deg": 5.0, "transverse_offset_mm": 4.0, "gold_span_overlap": 0.60, "prediction_span_overlap": 0.60, "assignment": "one_to_one_minimum_cost"},
        "resource_contract": {"working_point_budget": 200000, "peak_process_tree_rss_bytes_exclusive_max": 17179869184, "p95_latency_seconds_max": 300, "source_points_serialized": False},
        "semantic_gates": {"exact_case_count_min": 27, "precision_min": 0.90, "recall_min": 0.90, "minimum_family_recall": 0.80, "marker_accuracy_min": 0.99, "deterministic_replay": 1.0, "job_success": 1.0},
        "source_bindings": bindings,
    }
    encoded = (json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n").encode()
    LOCK_PATH.write_bytes(encoded)
    lock_sha = hashlib.sha256(encoded).hexdigest()
    invalidation_event = {
        "architecture": "V51",
        "detector_invocation_count": 0,
        "event": "PRECOMMIT_LOCK_V1_INVALIDATED_NO_CORPUS",
        "event_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "invalidated_lock_path": str(V1_LOCK_PATH),
        "invalidated_lock_sha256": _sha(V1_LOCK_PATH),
        "reason": "preflight launcher import-path failure before corpus assembly",
        "research_only": True,
        "token": TOKEN,
    }
    event = {
        "architecture": "V51",
        "detector_invocation_count": 0,
        "event": "PRECOMMIT_BINDINGS_V2_SEALED",
        "event_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lock_path": str(LOCK_PATH),
        "lock_sha256": lock_sha,
        "research_only": True,
        "token": TOKEN,
    }
    with EVENTS_PATH.open("ab", buffering=0) as handle:
        handle.write((json.dumps(invalidation_event, sort_keys=True, separators=(",", ":")) + "\n").encode())
        handle.write((json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode())
        os.fsync(handle.fileno())
    print(json.dumps({"lock": str(LOCK_PATH), "lock_sha256": lock_sha, "binding_count": len(bindings)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
