from __future__ import annotations

import json
import hashlib
import itertools
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import f1_v2_complete_same_surface_edge_probe as edge


SYNTHETIC_SCHEMA_VERSION = 1
SYNTHETIC_KIND = "f1_v2_complete_same_surface_edge_probe_synthetic_worst_cap_resource_v1"
SYNTHETIC_RESOURCE_PROFILE = "V2_COMPLETE_EDGE_ABSTRACT_26_REGULAR_CAP_ENVELOPE_V1"
SYNTHETIC_OUTPUT = Path(
    "output/research/f1-gold27-20260803/"
    "v2_complete_same_surface_edge_probe_v1_replays/"
    "synthetic_worst_cap_resource.json"
)
SYNTHETIC_RUN_FIELDS = {
    "run_index", "status", "parent_count", "mode_count", "modes_per_parent",
    "adjacent_parent_pair_count", "eligible_mode_pair_evaluation_count",
    "complete_same_surface_edge_count", "inherited_cross_mode_edge_count",
    "effective_mode_edge_count", "complete_same_surface_edge_table_sha256",
    "inherited_cross_mode_edge_table_sha256", "effective_mode_edge_semantic_sha256",
    "canonical_resource_semantic_sha256", "wall_seconds", "peak_rss_bytes",
}
SYNTHETIC_TOP_FIELDS = {
    "schema_version", "kind", "profile", "resource_profile",
    "resource_envelope_only", "physical_voxel_realizable", "resource_limits",
    "runs", "semantic_hashes_identical", "all_runs_below_16_gib",
    "attestation_sha256",
}


def _sha_json(value) -> str:
    return hashlib.sha256(edge.canonical_json_bytes(value)).hexdigest()


def _stream_sorted_record_hash(records: list[tuple[str, bytes]]) -> str:
    """Hash a canonical JSON table without materializing its joined bytes."""
    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, (_record_id, encoded) in enumerate(records):
        if index:
            digest.update(b",")
        digest.update(encoded)
    digest.update(b"]")
    return digest.hexdigest()


def _synthetic_worker(run_index: int) -> dict:
    """Exercise the exact algebraic cap, independent of finite voxel boundaries.

    A finite induced voxel set cannot contain 4,000 vertices all having 26
    occupied neighbours.  The resource-only envelope therefore uses the unique
    26-regular circulant graph on 4,000 parent IDs (offsets 1..13).  The core's
    physical 26-neighbour enumeration is tested separately in the core suite.
    """
    started = time.perf_counter()
    parent_count = 4_000
    modes_per_parent = 4
    mode_count = parent_count * modes_per_parent
    parent_pairs = [
        tuple(sorted((left, (left + offset) % parent_count)))
        for left in range(parent_count)
        for offset in range(1, 14)
    ]
    assert len(parent_pairs) == 52_000
    assert len(set(parent_pairs)) == 52_000

    complete_records: list[tuple[str, bytes]] = []
    for left_parent, right_parent in parent_pairs:
        for left_mode_offset in range(modes_per_parent):
            left_mode = f"M_{left_parent:04d}_{left_mode_offset}"
            for right_mode_offset in range(modes_per_parent):
                right_mode = f"M_{right_parent:04d}_{right_mode_offset}"
                mode_ids = sorted((left_mode, right_mode))
                edge_id = "E_" + _sha_json([mode_ids, "SAME_SURFACE"])[:16]
                record = {
                    "mode_edge_id": edge_id,
                    "mode_ids": mode_ids,
                    "edge_kind": "SAME_SURFACE",
                    # The envelope uses coplanar normals/centroids: every one of
                    # the 832k binary64 evaluations is eligible.
                    "spatial_residual": float(abs(right_parent - left_parent)),
                    "normal_residual": 0.0,
                    "plane_residual": 0.0,
                }
                complete_records.append((edge_id, edge.canonical_json_bytes(record)))
    assert len(complete_records) == 832_000
    assert len({record_id for record_id, _encoded in complete_records}) == 832_000
    complete_hash = _stream_sorted_record_hash(complete_records)

    cross_records: list[tuple[str, bytes]] = []
    for parent in range(parent_count):
        mode_ids = [f"M_{parent:04d}_{offset}" for offset in range(modes_per_parent)]
        for left_mode, right_mode in itertools.combinations(mode_ids, 2):
            pair = [left_mode, right_mode]
            edge_id = "E_" + _sha_json([pair, "CROSS_MODE_LOCAL"])[:16]
            record = {
                "mode_edge_id": edge_id, "mode_ids": pair,
                "edge_kind": "CROSS_MODE_LOCAL", "spatial_residual": 0.0,
                "normal_residual": 0.0, "plane_residual": 0.0,
            }
            cross_records.append((edge_id, edge.canonical_json_bytes(record)))
    assert len(cross_records) == 24_000
    cross_hash = _stream_sorted_record_hash(cross_records)

    semantic = {
        "parent_count": parent_count,
        "mode_count": mode_count,
        "modes_per_parent": modes_per_parent,
        "adjacent_parent_pair_count": len(parent_pairs),
        "eligible_mode_pair_evaluation_count": len(complete_records),
        "complete_same_surface_edge_count": len(complete_records),
        "inherited_cross_mode_edge_count": len(cross_records),
        "effective_mode_edge_count": len(complete_records) + len(cross_records),
        "complete_same_surface_edge_table_sha256": complete_hash,
        "inherited_cross_mode_edge_table_sha256": cross_hash,
    }
    effective_hash = _sha_json({
        "complete_same_surface_edge_table_sha256": complete_hash,
        "inherited_cross_mode_edge_table_sha256": cross_hash,
        "effective_mode_edge_count": semantic["effective_mode_edge_count"],
    })
    semantic["effective_mode_edge_semantic_sha256"] = effective_hash
    semantic_hash = _sha_json({
        "resource_profile": SYNTHETIC_RESOURCE_PROFILE,
        **semantic,
    })
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    return {
        "run_index": run_index, "status": "SUCCESS", **semantic,
        "canonical_resource_semantic_sha256": semantic_hash,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_bytes": peak_rss,
    }


def _run_synthetic_three_replays(output_path: Path = SYNTHETIC_OUTPUT) -> dict:
    runs = []
    for run_index in range(1, 4):
        completed = subprocess.run(
            [sys.executable, __file__, "--synthetic-worker", str(run_index)],
            cwd=Path.cwd(), text=True, capture_output=True, check=True, timeout=600,
        )
        run = json.loads(completed.stdout.strip().splitlines()[-1])
        if set(run) != SYNTHETIC_RUN_FIELDS:
            raise AssertionError("synthetic run schema mismatch")
        runs.append(run)
    semantic_hash_fields = (
        "complete_same_surface_edge_table_sha256",
        "inherited_cross_mode_edge_table_sha256",
        "effective_mode_edge_semantic_sha256",
        "canonical_resource_semantic_sha256",
    )
    semantic_hashes_identical = all(
        len({run[field] for run in runs}) == 1 for field in semantic_hash_fields
    )
    attestation = {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "kind": SYNTHETIC_KIND,
        "profile": edge.PRIMITIVE_PROFILE,
        "resource_profile": SYNTHETIC_RESOURCE_PROFILE,
        "resource_envelope_only": True,
        "physical_voxel_realizable": False,
        "resource_limits": {
            "peak_rss_limit_bytes": edge.PEAK_RSS_LIMIT_BYTES,
            "parent_cap": 4_000, "mode_cap": 16_000,
            "modes_per_parent_cap": 4, "adjacent_parent_pair_cap": 52_000,
            "complete_edge_evaluation_cap": 832_000,
            "complete_same_surface_edge_cap": 832_000,
            "inherited_cross_mode_edge_cap": 24_000,
            "effective_mode_edge_cap": 856_000,
        },
        "runs": runs,
        "semantic_hashes_identical": semantic_hashes_identical,
        "all_runs_below_16_gib": all(
            run["peak_rss_bytes"] < edge.PEAK_RSS_LIMIT_BYTES for run in runs
        ),
    }
    attestation["attestation_sha256"] = _sha_json(attestation)
    if set(attestation) != SYNTHETIC_TOP_FIELDS:
        raise AssertionError("synthetic attestation schema mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(edge.canonical_json_bytes(attestation))
    return attestation


def test_professor_hard_caps_are_frozen() -> None:
    assert edge.HARD_CAPS["input_working_point_cap"] == 200_000
    assert edge.HARD_CAPS["transition_parent_cap"] == 4_000
    assert edge.HARD_CAPS["transition_mode_cap"] == 16_000
    assert edge.HARD_CAPS["adjacent_parent_pair_cap"] == 52_000
    assert edge.HARD_CAPS["complete_edge_evaluation_cap"] == 832_000
    assert edge.HARD_CAPS["complete_same_surface_edge_cap"] == 832_000
    assert edge.HARD_CAPS["inherited_cross_mode_edge_cap"] == 24_000
    assert edge.HARD_CAPS["effective_mode_edge_cap"] == 856_000


def test_real_c001_fresh_process_peak_rss_below_16_gib() -> None:
    code = r'''
import json, resource
from pathlib import Path
from backend import f1_v2_complete_same_surface_edge_probe as edge
root=Path("output/research/f1-gold27-20260803")
row=json.loads((root/"local_plane_fragment_probe_v2/manifest.json").read_text())["cases"][0]
projected=edge.project_edge_probe_input(json.loads(Path(row["candidate_pool_path"]).read_text()))
payload=edge.build_v2_complete_same_surface_edge_probe(
    projected, scan_sha256=row["scan_sha256"],
    candidate_pool_sha256=row["candidate_pool_sha256"],
    a0_prediction_sha256=row["a0_prediction_sha256"],
    frozen_control_probe_sha256=row["probe_sha256"])
edge.validate_v2_complete_same_surface_edge_probe_payload(payload, projected_input=projected)
print(json.dumps({
    "canonical_sha256": __import__("hashlib").sha256(edge.canonical_json_bytes(payload)).hexdigest(),
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "serialized_output_bytes": len(edge.canonical_json_bytes(payload)),
    "one_parent_finite_span_count": payload["case_counters"]["one_parent_finite_span_count"],
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=Path.cwd(), text=True,
        capture_output=True, check=True, timeout=180,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["peak_rss_bytes"] < edge.PEAK_RSS_LIMIT_BYTES
    assert result["serialized_output_bytes"] > 0
    assert result["one_parent_finite_span_count"] == 0
    assert len(result["canonical_sha256"]) == 64


def test_synthetic_worst_cap_three_replays_are_identical_and_bounded() -> None:
    attestation = _run_synthetic_three_replays()
    assert attestation["semantic_hashes_identical"] is True
    assert attestation["all_runs_below_16_gib"] is True
    assert len(attestation["runs"]) == 3
    for run in attestation["runs"]:
        assert run["parent_count"] == 4_000
        assert run["mode_count"] == 16_000
        assert run["adjacent_parent_pair_count"] == 52_000
        assert run["eligible_mode_pair_evaluation_count"] == 832_000
        assert run["complete_same_surface_edge_count"] == 832_000
        assert run["inherited_cross_mode_edge_count"] == 24_000
        assert run["effective_mode_edge_count"] == 856_000
        assert run["peak_rss_bytes"] < edge.PEAK_RSS_LIMIT_BYTES


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--synthetic-worker":
        print(json.dumps(_synthetic_worker(int(sys.argv[2])), sort_keys=True))
    elif len(sys.argv) == 3 and sys.argv[1] == "--emit":
        print(json.dumps(_run_synthetic_three_replays(Path(sys.argv[2])), sort_keys=True))
    else:
        raise SystemExit(
            "usage: test_f1_v2_complete_same_surface_edge_probe_memory.py "
            "--synthetic-worker N | --emit OUTPUT"
        )
