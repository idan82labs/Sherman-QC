#!/usr/bin/env python3
"""Hash and freeze the sealed V66 U01-U04 one-shot preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.f1_transverse_face_turn_ownership_certificate_v66 as v66
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    frozen_v52,
    opaque_source_id,
    sha256_file,
    source_paths,
)


PROFILE = "V66_TFTOC_U01_U04_PROSPECTIVE_ONE_SHOT_V1"
FROZEN_CODE = (
    "backend/f1_cylindrical_consensus_recovery_v52.py",
    "backend/f1_transverse_face_turn_ownership_certificate_v66.py",
    "tests/test_f1_transverse_face_turn_ownership_certificate_v66.py",
    "tests/test_f1_tftoc_v66_protocol.py",
    "scripts/f1_tftoc_v66_protocol.py",
    "scripts/build_f1_tftoc_v66_u01_u04_preflight.py",
    "scripts/run_f1_tftoc_v66_u01_u04_one_shot.py",
    "scripts/render_f1_tftoc_v66_visual_packet.py",
    "scripts/score_f1_tftoc_v66_u01_u04_once.py",
)


def build(output_root: Path) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite preflight: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    constants_path = output_root / "constants.json"
    constants = {
        "architecture": v66.PROFILE,
        "mechanism_lock": v66.LOCK_TOKEN,
        "fixture_lock": v66.FIXTURE_TOKEN,
        "configuration_sha256": v66.CONFIGURATION_SHA256,
        "configuration": v66.frozen_configuration(),
    }
    atomic_json(constants_path, constants)

    cases = []
    for case_id, pool_name in CASE_BINDINGS:
        source_path, pool_path, v51_path = source_paths(case_id, pool_name)
        opaque_id = opaque_source_id(source_path)
        v52_result, vertices, faces, bounded_replay = frozen_v52(case_id, pool_name)
        # The topology hash is order/winding invariant and contains no source
        # coordinate serialization in the persisted manifest.
        mesh = v66.SourceMesh(vertices, faces)
        candidate_path = output_root / "candidates" / f"{opaque_id}.json"
        atomic_json(candidate_path, v52_result)
        cases.append({
            "opaque_source_id": opaque_id,
            "source_sha256": sha256_file(source_path),
            "pool_sha256": sha256_file(pool_path),
            "v51_input_sha256": sha256_file(v51_path),
            "bounded_vertex_count": len(vertices),
            "bounded_triangle_count": len(faces),
            "bounded_replay": bounded_replay,
            "bounded_mesh_topology_sha256": mesh.topology_hash,
            "candidate_count": len(v52_result.get("predictions") or ()),
            "candidate_snapshot": str(candidate_path.relative_to(output_root)),
            "candidate_snapshot_sha256": sha256_file(candidate_path),
        })
    cases.sort(key=lambda row: row["source_sha256"])

    frozen_files = [{"path": path, "sha256": sha256_file(ROOT / path)} for path in FROZEN_CODE]
    frozen_files.append({
        "path": str(constants_path.relative_to(ROOT)),
        "sha256": sha256_file(constants_path),
    })
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "architecture": v66.PROFILE,
        "mechanism_lock": v66.LOCK_TOKEN,
        "fixture_lock": v66.FIXTURE_TOKEN,
        "source_only": True,
        "research_only": True,
        "pseudo_truth_sealed": True,
        "labels_used": False,
        "expected_count_used": False,
        "cad_used": False,
        "filename_or_identity_available_to_decision": False,
        "source_point_cache_serialized": False,
        "case_names_replaced_by_opaque_source_hashes": True,
        "processing_order": "ascending_source_sha256",
        "variants": ["identity_primary", "identity_replay", "locked_similarity", "order_winding", "exact_pl_refinement"],
        "v52_rerun_on_variants": False,
        "ordinary_final_nms_call_count_per_variant": 1,
        "frozen_files": frozen_files,
        "cases": cases,
        "pseudo_truth_gate": "must remain sealed until 4096px visual adjudication is complete and hashed",
        "no_rerun_replacement_or_rescue": True,
        "execution_authorized": True,
    }
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.output_root.resolve())
    print(json.dumps({
        "profile": manifest["profile"],
        "case_count": len(manifest["cases"]),
        "opaque_source_ids": [row["opaque_source_id"] for row in manifest["cases"]],
        "candidate_counts": [row["candidate_count"] for row in manifest["cases"]],
        "execution_authorized": manifest["execution_authorized"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
