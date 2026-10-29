#!/usr/bin/env python3
"""Build the immutable source-free PGRR-V1 preexecution lock payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_pgrr_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402


CODE_PATHS = (
    "research/f1_pgrr_v1/__init__.py",
    "research/f1_pgrr_v1/config.py",
    "research/f1_pgrr_v1/geometry.py",
    "research/f1_pgrr_v1/independent_validator.py",
    "research/f1_pgrr_v1/frozen_handoff.py",
    "research/f1_pgrr_v1/tamper_suite.py",
    "research/f1_pgrr_v1/visual.py",
    "scripts/run_f1_pgrr_v1_candidate_worker.py",
    "scripts/run_f1_pgrr_v1_causal_preflight.py",
    "scripts/build_f1_pgrr_v1_preexecution_lock.py",
    "tests/test_f1_pgrr_v1.py",
)

INPUT_PATHS = (
    "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply",
    "output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v4/result.json",
    "output/research/ciip-digital-occupied-shell-h0-v1/smoke-correction-v4/BASE_A/container.bin",
    "output/research/f1-gold27-20260803/candidate_pool_transition_v2_probe/C006.json",
    "output/research/f1-gold27-20260803/local_plane_fragment_probe_v2/C006.json",
    "output/research/f1-gold27-20260803/v2_complete_same_surface_edge_probe_v1/C006.json",
    "output/research/f1-gold27-20260803/v2_cross_parent_rung_mapping_preflight_v1.json",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocation_proof() -> dict[str, Any]:
    n = CONFIG.sample_cap
    neighbor = CONFIG.first_neighbor_count
    edges = CONFIG.mutual_edge_maximum
    candidates = CONFIG.raw_ribbon_maximum
    deduplicated = CONFIG.deduplicated_ribbon_maximum
    byte_count = (n + 7) // 8
    ledger = {
        "sampled_xyz": n * 3 * 8,
        "sampled_point_ids": n * 32,
        "sampled_multiplicity": n * 4,
        "neighbor_ids": n * neighbor * 4,
        "neighbor_distances": n * neighbor * 8,
        "two_normal_fields": 2 * n * 3 * 8,
        "two_eigenvalue_fields": 2 * n * 3 * 8,
        "mean_normal": n * 3 * 8,
        "stable_and_condition_masks": 3 * n,
        "mutual_edge_endpoints": edges * 2 * 4,
        "pole_working_arrays": n * (8 + 8 + 2),
        "raw_membership_bitsets": candidates * byte_count,
        "deduplicated_bin_bitsets": deduplicated * CONFIG.angular_bin_maximum * byte_count,
        "target_region_bitsets": CONFIG.c006_target_count * 3 * byte_count,
        "union_find_and_sparse_graph_work": n * 24 + edges * 12,
        "renderer_and_serializer_reserve": 256 * 1024 * 1024,
        "python_numpy_scipy_runtime_reserve": 768 * 1024 * 1024,
        "allocator_fragmentation_reserve": 512 * 1024 * 1024,
    }
    total = sum(ledger.values())
    if total >= CONFIG.peak_rss_ceiling_bytes:
        raise ValueError("PGRR static allocation proof exceeds RSS ceiling")
    proof = {
        "ledger_bytes": ledger,
        "calculated_maximum_bytes": total,
        "rss_ceiling_bytes": CONFIG.peak_rss_ceiling_bytes,
        "headroom_bytes": CONFIG.peak_rss_ceiling_bytes - total,
        "source_points": n,
        "stored_neighbor_references": n * neighbor,
        "projective_axes": CONFIG.projective_pole_count,
        "two_scale_pole_normal_band_tests": 2 * n * CONFIG.projective_pole_count,
        "mutual_source_edges": edges,
        "pole_edge_eligibility_tests": edges * CONFIG.projective_pole_count,
        "raw_ribbon_maximum": candidates,
        "deduplicated_ribbon_maximum": deduplicated,
        "ribbon_target_handoff_rows": deduplicated * CONFIG.c006_target_count,
        "adjacent_boundary_checks": deduplicated * CONFIG.c006_target_count * (CONFIG.angular_bin_maximum - 1),
        "roles_sequential": True,
        "blas_threads_per_role": 1,
    }
    proof["resource_proof_sha256"] = sha_json(proof)
    return proof


def build_lock() -> dict[str, Any]:
    response = ROOT / "docs/research/PROFESSOR_F1_POST_CIIP_NEXT_ARCHITECTURE_GPT56_PRO_RESPONSE_20260809.txt"
    lock = {
        "schema_version": 1,
        "profile": CONFIG.tranche_id,
        "authorization": {
            "ruling": "AUTHORIZE_PGRR_V1_C006_FROZEN_TARGET_CAUSAL_PREFLIGHT_V1",
            "professor_response_sha256": _sha_file(response),
            "execution_count": "EXACTLY_ONE",
            "claim_consumption": "BEFORE_FIRST_WORKER_LAUNCH",
        },
        "configuration": CONFIG.as_frozen_dict(),
        "configuration_sha256": CONFIG_SHA256,
        "resource_proof": allocation_proof(),
        "raw_source_relative": INPUT_PATHS[0],
        "code_sha256": {relative: _sha_file(ROOT / relative) for relative in CODE_PATHS},
        "input_sha256": {relative: _sha_file(ROOT / relative) for relative in INPUT_PATHS},
        "safety": {
            "research_only": True,
            "runtime_main_write": False,
            "production_access": False,
            "production_import": False,
            "gold_access": False,
            "truth_opening": False,
            "cad_access": False,
            "expected_count_access": False,
            "manual_adjudication": False,
            "benchmark_feedback": False,
            "family_routing": False,
            "target_access_before_candidate_freeze": False,
            "filename_feature_access": False,
            "randomness": False,
            "gpu": False,
            "parameter_sweep": False,
            "ranking": False,
            "top_k": False,
            "threshold_mutation": False,
            "source_point_serialization": False,
            "replacement_input": False,
            "replay": False,
            "synthetic_scientific_evidence": False,
            "visual_review_read_only": True,
        },
        "failure_policy": "EARLIEST_APPLICABLE_TERMINAL_TOKEN_NO_RERUN_NO_REPLACEMENT_NO_RESCUE",
    }
    lock["execution_lock_sha256"] = sha_json(lock)
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("lock builder accepts no arguments")
    print(canonical_json_bytes(build_lock()).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
