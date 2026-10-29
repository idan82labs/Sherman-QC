#!/usr/bin/env python3
"""Add bounded TRANSITION_ATOM_V2 support to a frozen scan-only pool.

Candidate geometries, source lanes, and line objects are copied byte-for-byte
as parsed JSON values.  Only scan-derived support tables are added.  This
command accepts no gold, expected-count, CAD, part, or split-group input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_scan_only_candidate_pool import (  # noqa: E402
    _load_geometry_views,
    augment_case_pool_with_transition_v2,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def augment_manifest(
    *,
    pool_manifest_path: Path,
    inference_manifest_path: Path,
    output_dir: Path,
    case_tokens: Sequence[str] = (),
    plane_voxel_mm: float = 1.8,
) -> dict[str, Any]:
    pool_manifest = _read(pool_manifest_path)
    inference_manifest = _read(inference_manifest_path)
    wanted = {str(value) for value in case_tokens}
    inference_by_case = {
        str(record["case_token"]): dict(record)
        for record in inference_manifest.get("cases") or ()
    }
    pool_rows = list(pool_manifest.get("cases") or ())
    if not pool_rows:
        raise ValueError("pool manifest contains no cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(pool_rows, start=1):
        case_token = str(raw_row.get("case_token") or raw_row.get("case_id") or "")
        if not case_token or (wanted and case_token not in wanted):
            continue
        inference_row = inference_by_case.get(case_token)
        if inference_row is None:
            raise ValueError(f"{case_token}: missing inference manifest row")
        source_pool_path = Path(str(raw_row["path"])).resolve()
        scan_path = Path(str(inference_row["source_scan_path"])).resolve()
        expected_scan_sha256 = str(inference_row["source_scan_sha256"])
        actual_scan_sha256 = _sha256_file(scan_path)
        if actual_scan_sha256 != expected_scan_sha256:
            raise ValueError(f"{case_token}: scan hash mismatch")
        source_pool = _read(source_pool_path)
        frozen_candidates_sha256 = _json_sha256(
            {
                "candidates": source_pool.get("candidates") or [],
                "line_objects": source_pool.get("line_objects") or [],
            }
        )
        print(
            f"[{index}/{len(pool_rows)}] {case_token}: build TRANSITION_ATOM_V2",
            flush=True,
        )
        points, normals, plane_points, plane_normals = _load_geometry_views(
            scan_path,
            plane_voxel_mm=float(plane_voxel_mm),
        )
        augmented = augment_case_pool_with_transition_v2(
            source_pool,
            points=points,
            normals=normals,
            plane_points=plane_points,
            plane_normals=plane_normals,
        )
        if frozen_candidates_sha256 != _json_sha256(
            {
                "candidates": augmented.get("candidates") or [],
                "line_objects": augmented.get("line_objects") or [],
            }
        ):
            raise AssertionError(f"{case_token}: frozen candidate geometry changed")
        augmented["case_id"] = case_token
        augmented["scan_path"] = str(scan_path)
        augmented["coordinate_frame"] = "scan_raw"
        output_path = output_dir / f"{case_token}.json"
        output_path.write_text(
            json.dumps(augmented, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "case_token": case_token,
                "path": str(output_path),
                "output_sha256": _sha256_file(output_path),
                "source_pool_path": str(source_pool_path),
                "source_pool_sha256": _sha256_file(source_pool_path),
                "source_scan_sha256": actual_scan_sha256,
                "frozen_candidates_sha256": frozen_candidates_sha256,
                "working_point_count": int(augmented["transient_sampled_point_count"]),
                "plane_point_count": int(augmented["transient_plane_point_count"]),
                "transition_parent_count": len(augmented["support_parent_table"]),
                "transition_mode_count": len(augmented["support_mode_table"]),
                "transition_mode_edge_count": len(augmented["support_mode_edges"]),
            }
        )
    if wanted - {row["case_token"] for row in rows}:
        raise ValueError(f"requested cases were not found: {sorted(wanted - {row['case_token'] for row in rows})}")
    manifest = {
        "schema_version": 1,
        "kind": "f1_transition_atom_v2_augmented_candidate_pool",
        "source_pool_manifest_sha256": _sha256_file(pool_manifest_path),
        "inference_manifest_sha256": _sha256_file(inference_manifest_path),
        "case_count": len(rows),
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "candidate_geometry_frozen": True,
        "working_point_budget": 200_000,
        "plane_voxel_mm": float(plane_voxel_mm),
        "cases": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def finalize_existing_outputs(
    *,
    pool_manifest_path: Path,
    inference_manifest_path: Path,
    output_dir: Path,
    plane_voxel_mm: float = 1.8,
) -> dict[str, Any]:
    """Rebuild the deterministic batch manifest after parallel case jobs."""

    pool_manifest = _read(pool_manifest_path)
    inference_manifest = _read(inference_manifest_path)
    inference_by_case = {
        str(record["case_token"]): dict(record)
        for record in inference_manifest.get("cases") or ()
    }
    rows: list[dict[str, Any]] = []
    for raw_row in pool_manifest.get("cases") or ():
        case_token = str(raw_row.get("case_token") or raw_row.get("case_id") or "")
        output_path = output_dir / f"{case_token}.json"
        if not output_path.exists():
            raise ValueError(f"{case_token}: missing augmented pool output")
        payload = _read(output_path)
        source_pool_path = Path(str(raw_row["path"])).resolve()
        source_pool = _read(source_pool_path)
        frozen_candidates_sha256 = _json_sha256(
            {
                "candidates": source_pool.get("candidates") or [],
                "line_objects": source_pool.get("line_objects") or [],
            }
        )
        if frozen_candidates_sha256 != _json_sha256(
            {
                "candidates": payload.get("candidates") or [],
                "line_objects": payload.get("line_objects") or [],
            }
        ):
            raise AssertionError(f"{case_token}: frozen candidate geometry changed")
        inference_row = inference_by_case.get(case_token)
        if inference_row is None:
            raise ValueError(f"{case_token}: missing inference manifest row")
        rows.append(
            {
                "case_token": case_token,
                "path": str(output_path.resolve()),
                "output_sha256": _sha256_file(output_path),
                "source_pool_path": str(source_pool_path),
                "source_pool_sha256": _sha256_file(source_pool_path),
                "source_scan_sha256": str(inference_row["source_scan_sha256"]),
                "frozen_candidates_sha256": frozen_candidates_sha256,
                "working_point_count": int(payload["transient_sampled_point_count"]),
                "plane_point_count": int(payload["transient_plane_point_count"]),
                "transition_parent_count": len(payload["support_parent_table"]),
                "transition_mode_count": len(payload["support_mode_table"]),
                "transition_mode_edge_count": len(payload["support_mode_edges"]),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "f1_transition_atom_v2_augmented_candidate_pool",
        "source_pool_manifest_sha256": _sha256_file(pool_manifest_path),
        "inference_manifest_sha256": _sha256_file(inference_manifest_path),
        "case_count": len(rows),
        "support_representation_profile": "TRANSITION_ATOM_V2",
        "source_point_serialization": False,
        "candidate_geometry_frozen": True,
        "working_point_budget": 200_000,
        "plane_voxel_mm": float(plane_voxel_mm),
        "cases": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-token", action="append", default=[])
    parser.add_argument("--plane-voxel-mm", type=float, default=1.8)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        manifest = finalize_existing_outputs(
            pool_manifest_path=args.pool_manifest.resolve(),
            inference_manifest_path=args.inference_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            plane_voxel_mm=args.plane_voxel_mm,
        )
    else:
        manifest = augment_manifest(
            pool_manifest_path=args.pool_manifest.resolve(),
            inference_manifest_path=args.inference_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            case_tokens=args.case_token,
            plane_voxel_mm=args.plane_voxel_mm,
        )
    print(json.dumps({"case_count": manifest["case_count"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
