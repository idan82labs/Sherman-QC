#!/usr/bin/env python3
"""Build the immutable one-shot GCDF-V1 execution lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_gcdf_v1.config import CONFIG, CONFIG_SHA256, canonical_json_bytes, sha_json  # noqa: E402

CODE_PATHS = (
    "research/f1_gcdf_v1/__init__.py",
    "research/f1_gcdf_v1/config.py",
    "research/f1_gcdf_v1/core.py",
    "research/f1_gcdf_v1/tamper.py",
    "research/f1_gcdf_v1/visual.py",
    "scripts/build_f1_gcdf_v1_lock.py",
    "scripts/build_f1_gcdf_v1_source_binding.py",
    "scripts/run_f1_gcdf_v1_causal_preflight.py",
    "scripts/run_f1_gcdf_v1_role.py",
    "scripts/run_f1_gcdf_v1_source_free_regressions.py",
    "tests/test_f1_gcdf_v1.py",
)
INPUT_PATHS = (
    "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply",
    "docs/research/PROFESSOR_F1_POST_PGRR_NEXT_ARCHITECTURE_GPT56_PRO_RESPONSE_20260809.txt",
    "docs/research/GCDF_V1_SOURCE_BINDING_V1_20260809.json",
    "docs/research/GCDF_V1_SOURCE_FREE_REGRESSIONS_V1_20260809.json",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocation_proof() -> dict[str, Any]:
    n = CONFIG.sample_count
    arcs = n * CONFIG.graph_neighbors * 2
    edges = n * CONFIG.graph_neighbors // 2
    ledger = {
        "sample_binary64_xyz": n * 3 * 8,
        "sample_point_ids": n * 32,
        "sample_ordinals_and_multiplicity": n * 12,
        "canonical_integer_and_metric_xyz": n * 3 * (4 + 8),
        "neighbor_ids_squared_weights": n * CONFIG.tangent_neighbors * (4 + 8 + 8),
        "temporary_tangent_offsets_covariance_eigensystem": n * (32 * 3 * 8 + 3 * 3 * 8 * 3),
        "directed_edge_codes_and_weights": n * CONFIG.graph_neighbors * 16,
        "csr_arcs": arcs * 16,
        "undirected_edges": edges * 16,
        "one_shortest_path_workspace": n * 24,
        "vote_link_cap": 64 * 12_720 * 16,
        "two_full_resolution_raster_buffers": 2 * 6144 * 4096 * 4,
        "python_scipy_matplotlib_runtime_reserve": 1_610_612_736,
        "allocator_fragmentation_reserve": 805_306_368,
    }
    total = sum(ledger.values())
    if total >= CONFIG.peak_rss_ceiling_bytes:
        raise ValueError("GCDF static allocation proof exceeds RSS ceiling")
    proof = {
        "ledger_bytes": ledger, "calculated_maximum_bytes": total,
        "rss_ceiling_bytes": CONFIG.peak_rss_ceiling_bytes,
        "headroom_bytes": CONFIG.peak_rss_ceiling_bytes - total,
        "roles_sequential": True, "neighbor_record_count": n * CONFIG.tangent_neighbors,
        "undirected_mutual_edge_upper_bound": edges, "csr_arc_upper_bound": arcs,
        "landmark_pair_upper_bound": 12_720, "n_by_n_matrix_allocated": False,
        "source_points_serialized": False,
    }
    proof["resource_proof_sha256"] = sha_json(proof)
    return proof


def build_lock() -> dict[str, Any]:
    binding = json.loads((ROOT / INPUT_PATHS[2]).read_text(encoding="utf-8"))
    regression = json.loads((ROOT / INPUT_PATHS[3]).read_text(encoding="utf-8"))
    if not regression.get("passed") or regression.get("source_opened"):
        raise ValueError("GCDF source-free regression record invalid")
    lock: dict[str, Any] = {
        "schema_version": 1, "profile": CONFIG.attempt_id,
        "authorization": {
            "ruling": "AUTHORIZE_F1_POST_PGRR_SINGLE_ARCHITECTURE_PREFLIGHT_V1",
            "professor_response_relative": INPUT_PATHS[1],
            "professor_response_sha256": _sha_file(ROOT / INPUT_PATHS[1]),
            "execution_count": "EXACTLY_ONE", "claim_consumption": "BEFORE_FIRST_ROLE_OPENS_RAW_SOURCE",
            "pgrr_stop_remains_final": True,
        },
        "configuration": CONFIG.as_frozen_dict(), "configuration_sha256": CONFIG_SHA256,
        "raw_source_relative": INPUT_PATHS[0], "source_binding": binding,
        "source_free_regression": regression, "resource_proof": allocation_proof(),
        "code_sha256": {relative: _sha_file(ROOT / relative) for relative in CODE_PATHS},
        "input_sha256": {relative: _sha_file(ROOT / relative) for relative in INPUT_PATHS},
        "output_relative": "output/research/gcdf-v1-c006-frozen-source-causal-preflight-v1",
        "safety": {
            "research_only": True, "source_only": True, "target_capability_granted": False,
            "target_capability_opened": False, "target_process_launched": False,
            "gold_access": False, "labels_access": False, "cad_access": False,
            "expected_count_access": False, "benchmark_feedback": False,
            "runtime_main_access": False, "production_access": False, "replay": False,
            "replacement_input": False, "parameter_sweep": False, "threshold_mutation": False,
            "source_point_serialization": False, "visual_review_mandatory": True,
        },
        "failure_policy": "EARLIEST_APPLICABLE_WALL_NO_RERUN_NO_REPLACEMENT_NO_THRESHOLD_RESCUE",
    }
    lock["execution_lock_sha256"] = sha_json(lock)
    return lock


if __name__ == "__main__":
    print(canonical_json_bytes(build_lock()).decode("ascii"))
