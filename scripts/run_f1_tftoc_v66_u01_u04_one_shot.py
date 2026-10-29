#!/usr/bin/env python3
"""Execute the hash-bound V66 U01-U04 prospective lane exactly once."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_transverse_face_turn_ownership_certificate_v66 import apply_tftoc_v1, canonical_bytes
from scripts.f1_tftoc_v66_protocol import (
    CASE_BINDINGS,
    atomic_json,
    canonical_json_bytes,
    load_identity,
    opaque_source_id,
    ordinary_final_nms,
    order_winding_variant,
    pl_refinement_variant,
    sha256_file,
    similarity_variant,
    source_paths,
    tftoc_semantic_signature,
)


PROFILE = "V66_TFTOC_U01_U04_PROSPECTIVE_ONE_SHOT_V1"


def _verify_preflight(root: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("execution_authorized") is not True or manifest.get("pseudo_truth_sealed") is not True:
        raise RuntimeError("preflight does not authorize sealed execution")
    for row in manifest.get("frozen_files") or ():
        path = ROOT / str(row["path"])
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"frozen file drift: {path}: {observed} != {row['sha256']}")
    for case in manifest.get("cases") or ():
        candidate_path = root / str(case["candidate_snapshot"])
        if sha256_file(candidate_path) != case["candidate_snapshot_sha256"]:
            raise RuntimeError(f"candidate snapshot drift: {case['opaque_source_id']}")


def _run_variant(
    baseline: Mapping[str, Any], vertices: np.ndarray, faces: np.ndarray
) -> tuple[dict[str, Any], tuple[Any, ...], list[dict[str, Any]], list[Mapping[str, Any]]]:
    result, pairs = apply_tftoc_v1(baseline, vertices, faces)
    if result["tftoc_v1_audit"]["status"] != "OK":
        raise RuntimeError(f"V66 atomic no-op: {result['tftoc_v1_audit']}")
    final_rows, final_nms_audit = ordinary_final_nms(result.get("predictions") or ())
    return result, pairs, final_rows, final_nms_audit


def _hash_semantic(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def execute(preflight_root: Path, output_root: Path) -> Mapping[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"one-shot output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((preflight_root / "manifest.json").read_text(encoding="utf-8"))
    _verify_preflight(preflight_root, manifest)
    case_by_opaque = {row["opaque_source_id"]: row for row in manifest["cases"]}
    binding_by_opaque = {}
    for case_id, pool_name in CASE_BINDINGS:
        source_path, _pool_path, _v51_path = source_paths(case_id, pool_name)
        binding_by_opaque[opaque_source_id(source_path)] = (case_id, pool_name)

    started = time.perf_counter()
    records = []
    terminal_failure = None
    for opaque_id in sorted(case_by_opaque):
        case_manifest = case_by_opaque[opaque_id]
        case_id, pool_name = binding_by_opaque[opaque_id]
        vertices, faces, _pool, loaded = load_identity(case_id, pool_name)
        baseline = json.loads((preflight_root / case_manifest["candidate_snapshot"]).read_text(encoding="utf-8"))
        if len(vertices) != case_manifest["bounded_vertex_count"] or len(faces) != case_manifest["bounded_triangle_count"]:
            raise RuntimeError(f"bounded source drift: {opaque_id}")

        identity_primary, primary_pairs, primary_final, primary_nms = _run_variant(baseline, vertices, faces)
        identity_replay, replay_pairs, replay_final, replay_nms = _run_variant(baseline, vertices, faces)
        primary_signature = tftoc_semantic_signature(identity_primary, primary_pairs)
        replay_signature = tftoc_semantic_signature(identity_replay, replay_pairs)
        replay_equal = canonical_bytes(identity_primary["tftoc_v1_audit"]) == canonical_bytes(identity_replay["tftoc_v1_audit"])
        replay_equal = replay_equal and canonical_bytes(primary_signature) == canonical_bytes(replay_signature)
        replay_equal = replay_equal and canonical_bytes(primary_nms) == canonical_bytes(replay_nms)
        replay_equal = replay_equal and [row["route_hash"] for row in primary_final] == [row["route_hash"] for row in replay_final]

        similar_vertices, _matrix, similar_rows = similarity_variant(vertices, baseline["predictions"])
        similar_baseline = deepcopy(dict(baseline)); similar_baseline["predictions"] = similar_rows
        similar, similar_pairs, similar_final, similar_nms = _run_variant(similar_baseline, similar_vertices, faces)

        order_vertices, order_faces = order_winding_variant(vertices, faces)
        ordered, ordered_pairs, ordered_final, ordered_nms = _run_variant(baseline, order_vertices, order_faces)

        refined_vertices, refined_faces, refinement_count = pl_refinement_variant(vertices, faces)
        refined, refined_pairs, refined_final, refined_nms = _run_variant(baseline, refined_vertices, refined_faces)

        signatures = {
            "identity": primary_signature,
            "similarity": tftoc_semantic_signature(similar, similar_pairs),
            "order_winding": tftoc_semantic_signature(ordered, ordered_pairs),
            "pl_refinement": tftoc_semantic_signature(refined, refined_pairs),
        }
        mutation_invariant = all(canonical_bytes(value) == canonical_bytes(primary_signature) for value in signatures.values())
        final_hashes = {
            "identity": [row["route_hash"] for row in primary_final],
            "similarity": [row["route_hash"] for row in similar_final],
            "order_winding": [row["route_hash"] for row in ordered_final],
            "pl_refinement": [row["route_hash"] for row in refined_final],
        }
        final_invariant = all(value == final_hashes["identity"] for value in final_hashes.values())
        nms_audits = {
            "identity": primary_nms,
            "similarity": similar_nms,
            "order_winding": ordered_nms,
            "pl_refinement": refined_nms,
        }
        nms_invariant = all(
            [(row["route_hash"], row["accepted"], row["duplicate_of"]) for row in value]
            == [(row["route_hash"], row["accepted"], row["duplicate_of"]) for row in primary_nms]
            for value in nms_audits.values()
        )
        passed = replay_equal and mutation_invariant and final_invariant and nms_invariant
        if not passed and terminal_failure is None:
            terminal_failure = f"METAMORPHIC_OR_REPLAY_MISMATCH:{opaque_id}"

        baseline_final, baseline_nms = ordinary_final_nms(baseline["predictions"])
        case_payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "opaque_source_id": opaque_id,
            "source_only": True,
            "research_only": True,
            "pseudo_truth_opened": False,
            "identity": {
                "incoming_pre_nms_count": len(baseline["predictions"]),
                "incoming_pre_nms_predictions": baseline["predictions"],
                "v52_final_count": len(baseline_final),
                "v66_pre_nms_count": len(identity_primary["predictions"]),
                "v66_final_count": len(primary_final),
                "v52_final_predictions": baseline_final,
                "v66_final_predictions": primary_final,
                "v52_final_nms_audit": baseline_nms,
                "v66_final_nms_audit": primary_nms,
                "tftoc_audit": identity_primary["tftoc_v1_audit"],
                "tftoc_audit_sha256": identity_primary["tftoc_v1_audit_sha256"],
                "semantic_signature": primary_signature,
                "semantic_signature_sha256": _hash_semantic(primary_signature),
            },
            "variants": {
                "replay_byte_identical": replay_equal,
                "similarity_signature_sha256": _hash_semantic(signatures["similarity"]),
                "order_winding_signature_sha256": _hash_semantic(signatures["order_winding"]),
                "pl_refinement_signature_sha256": _hash_semantic(signatures["pl_refinement"]),
                "pl_refined_face_count": refinement_count,
                "mutation_invariant": mutation_invariant,
                "final_output_invariant": final_invariant,
                "ordinary_nms_invariant": nms_invariant,
                "ordinary_nms_call_count_per_variant": 1,
            },
            "passed": passed,
        }
        case_path = output_root / "cases" / opaque_id / "result.json"
        atomic_json(case_path, case_payload)
        records.append({
            "opaque_source_id": opaque_id,
            "result": str(case_path.relative_to(output_root)),
            "result_sha256": sha256_file(case_path),
            "incoming_pre_nms_count": len(baseline["predictions"]),
            "v52_final_count": len(baseline_final),
            "v66_final_count": len(primary_final),
            "suppression_count": len(identity_primary["tftoc_v1_audit"].get("suppressed_hashes") or ()),
            "positive_pair_count": sum(pair.pair_class == "POSITIVE" for pair in primary_pairs),
            "passed": passed,
        })

    elapsed = time.perf_counter() - started
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    summary = {
        "schema_version": 1,
        "profile": PROFILE,
        "preflight_manifest_sha256": sha256_file(preflight_root / "manifest.json"),
        "pseudo_truth_opened": False,
        "case_count": len(records),
        "cases": records,
        "total_suppression_count": sum(row["suppression_count"] for row in records),
        "changed_case_count": sum(row["suppression_count"] > 0 for row in records),
        "wall_time_seconds": elapsed,
        "peak_rss_native": peak_rss,
        "terminal_failure": terminal_failure,
        "execution_passed": terminal_failure is None and all(row["passed"] for row in records),
        "next_gate": "NATIVE_VISUAL_AUDIT_BEFORE_PSEUDO_TRUTH" if terminal_failure is None else "TERMINAL_RETIRE_V66",
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = execute(args.preflight_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["execution_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
