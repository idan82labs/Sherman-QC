#!/usr/bin/env python3
"""Assemble the frozen 27-part candidate corpus from scan-only pool artifacts.

The join key is the frozen record ID, never the part identifier or bend count.
Candidate geometry was generated without access to labels/CAD; this script only
renames cases to their opaque freeze tokens and verifies source scan hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "output/research/f1-gold27-20260803/freeze/sealed_gold_manifest.json"
DEFAULT_INFERENCE = ROOT / "output/research/f1-gold27-20260803/freeze/inference_manifest.json"
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"
DEFAULT_ROOTS = (
    ROOT / "output/experiments/f1-10of13/strict-blind-uniform-candidate-pool-mesh-v4",
    ROOT / "output/manual_labels/shaked-count-backed-12-20260717/candidate_pool_mesh_v4",
    ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/candidate_pool",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_sha256(path: Path) -> str:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices):
        points = np.asarray(mesh.vertices, dtype="<f8")
    else:
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype="<f8")
    if not len(points):
        raise ValueError(f"geometry is empty: {path}")
    points = np.unique(points, axis=0)
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    return hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()


def _source_stem(record_id: str) -> str:
    namespace, local_id = record_id.split("/", 1)
    if namespace in {"corpus", "shaked12"}:
        return local_id
    if namespace == "shaked_xv":
        return f"{int(local_id[1:]):02d}_{local_id}"
    raise ValueError(f"unsupported record namespace: {namespace}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--inference-manifest", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-root", type=Path, action="append")
    args = parser.parse_args()

    gold = _read(args.gold_manifest.resolve())
    inference = _read(args.inference_manifest.resolve())
    if inference["corpus_freeze_id"] != gold["corpus_freeze_id"]:
        raise ValueError("freeze ID mismatch")
    frozen_inputs = {
        str(case["case_token"]): Path(str(case["source_scan_path"])).resolve()
        for case in inference.get("cases") or []
    }
    roots = tuple(path.resolve() for path in (args.candidate_root or DEFAULT_ROOTS))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for case in gold.get("cases") or []:
        token = str(case["case_token"])
        stem = _source_stem(str(case["record_id"]))
        matches = [root / f"{stem}.json" for root in roots if (root / f"{stem}.json").is_file()]
        if len(matches) != 1:
            raise ValueError(f"{token}: expected exactly one candidate pool for {stem}, found {matches}")
        source_path = matches[0]
        payload = _read(source_path)
        contract = payload.get("inference_contract") or {}
        if any(
            bool(contract.get(key))
            for key in ("part_identifier_exposed", "expected_count_exposed", "manual_truth_exposed", "cad_exposed")
        ):
            raise ValueError(f"{token}: candidate pool violates blind inference contract")
        scan_path = Path(str(payload["scan_path"])).resolve()
        actual_scan_hash = _sha256(scan_path)
        expected_scan_hash = str(case["source_scan_sha256"])
        if actual_scan_hash != expected_scan_hash:
            # The labeling studio requires PLY, so one historical STL was
            # converted without changing its vertex coordinates. Accept that
            # representation-only difference only after a canonical geometry
            # digest proves the point sets are identical.
            if _geometry_sha256(scan_path) != _geometry_sha256(frozen_inputs[token]):
                raise ValueError(
                    f"{token}: source scan hash and geometry mismatch: "
                    f"{actual_scan_hash} != {expected_scan_hash}"
                )
        payload["case_id"] = token
        payload["source_case_id"] = stem
        payload["corpus_freeze_id"] = gold["corpus_freeze_id"]
        destination = output_dir / f"{token}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                "case_token": token,
                "path": str(destination),
                "source_path": str(source_path),
                "source_scan_sha256": expected_scan_hash,
                "candidate_count": int(payload.get("candidate_count") or 0),
                "line_object_count": int(payload.get("line_object_count") or 0),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_gold27_scan_only_candidate_pool",
        "corpus_freeze_id": gold["corpus_freeze_id"],
        "case_count": len(rows),
        "source_point_serialization": False,
        "join_contract": "frozen_record_id_to_opaque_case_token",
        "candidate_generation_uses_manual_truth": False,
        "cases": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
