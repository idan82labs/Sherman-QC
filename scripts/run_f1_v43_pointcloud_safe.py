#!/usr/bin/env python3
"""Fail-closed V43 compatibility for source PLYs without triangle faces.

V43's measurement sidecar requires a retained triangle mesh.  A scanner-like
point cloud cannot supply that capability.  This adapter therefore preserves
the V16 incumbents unchanged and records the unavailable sidecar; it does not
reconstruct a mesh or use calibrated raw frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    retained = json.loads(args.retained.read_text(encoding="utf-8"))
    predictions = list(retained.get("predictions") or [])
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v43_pointcloud_safe_result",
        "profile": "F1_V43_POINTCLOUD_SAFE_ABSTENTION_V1",
        "route": retained.get("route"),
        "candidate_pool_sha256": _sha(args.pool),
        "retained_v16_sha256": _sha(args.retained),
        "scan_sha256": _sha(args.scan),
        "pointcloud_measurement_sidecar_available": False,
        "pointcloud_measurement_sidecar_action": "PRESERVE_V16_NO_V43_ADDITION",
        "compatible_rejected_candidate_count": 0,
        "candidate_mappings": [],
        "v43_results": [],
        "v43_result_sha256": [],
        "v43_resource_stops": [{"token": "V43_TRIANGLE_MESH_CAPABILITY_UNAVAILABLE"}],
        "v43_resource_stop_count": 1,
        "v43_pre_merge_addition_count": 0,
        "v43_post_merge_addition_count": 0,
        "incumbent_first_merge_audit": [],
        "retained_v16_prediction_count": len(predictions),
        "candidate_count": len(predictions),
        "predictions": predictions,
        "research_only": True,
        "source_only": True,
        "labels_used": False,
        "expected_count_used": False,
        "part_identity_used": False,
        "cad_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(predictions), "sidecar_available": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
