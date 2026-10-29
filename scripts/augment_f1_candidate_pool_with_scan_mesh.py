#!/usr/bin/env python3
"""Add identity-free scan-mesh adjacency candidates to a frozen F1 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_scan_only_candidate_pool import (  # noqa: E402
    _attach_case_relative_features,
    _attach_finite_scan_support,
    _attach_local_bend_support,
    _load_geometry_views,
    _scan_mesh_adjacency_candidates,
    _tag,
    _trim_to_longest_scan_supported_run,
    cluster_line_candidates,
)


FEATURE_SCHEMA_VERSION = "f1_line_selector_features_v3"
REQUIRED_LINE_OBJECT_FEATURES = (
    "normal_second_eigenvalue",
    "two_sided_transition_score",
    "two_sided_normal_angle_deg",
    "two_sided_support_balance",
    "two_sided_min_support_fraction",
    "two_sided_normal_coherence",
    "cluster_max_two_sided_transition_score",
    "cluster_max_two_sided_normal_angle_deg",
    "cluster_max_two_sided_support_balance",
    "cluster_max_two_sided_min_support_fraction",
    "cluster_max_two_sided_normal_coherence",
    "tight_normal_second_eigenvalue_r2mm",
    "tight_normal_second_eigenvalue_r6mm",
    "tight_to_wide_normal_second_ratio",
    "cluster_max_tight_normal_second_eigenvalue_r2mm",
    "cluster_max_tight_to_wide_normal_second_ratio",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_line_object_feature_schema(line_objects: list[dict[str, Any]]) -> None:
    for index, line_object in enumerate(line_objects):
        missing = [name for name in REQUIRED_LINE_OBJECT_FEATURES if name not in line_object]
        if missing:
            raise ValueError(
                f"line object {index} is missing required {FEATURE_SCHEMA_VERSION} fields: {missing}"
            )


def _namespace_existing_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    source = str(result.get("candidate_source") or "unknown")
    source_id = str(
        result.get("source_candidate_id")
        or result.get("candidate_id")
        or result.get("bend_id")
        or "candidate"
    )
    prefix = f"{source}::"
    if source_id.startswith(prefix):
        source_id = source_id[len(prefix):]
    result["source_candidate_id"] = source_id
    result["candidate_id"] = f"{source}::{source_id}"
    result["bend_id"] = result["candidate_id"]
    return result


def augment_case(pool: dict[str, Any], scan_path: Path, *, voxel_mm: float) -> dict[str, Any]:
    points, normals, plane_points, plane_normals = _load_geometry_views(
        scan_path,
        plane_voxel_mm=voxel_mm,
    )
    mesh_candidates, diagnostics = _scan_mesh_adjacency_candidates(scan_path)
    tagged = _trim_to_longest_scan_supported_run(
        _tag(mesh_candidates, "scan_mesh_adjacency"),
        points,
    )
    candidates = [
        *[
            _namespace_existing_candidate(dict(candidate))
            for candidate in pool.get("candidates") or ()
            if str(candidate.get("candidate_source")) != "scan_mesh_adjacency"
        ],
        *tagged,
    ]
    candidate_ids = [str(candidate.get("candidate_id")) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique after source namespacing")
    # Recompute the complete selector feature schema for every source.  Frozen
    # input pools were produced by multiple historical enrichment versions;
    # enriching only the new mesh candidates makes missingness identify the
    # corpus/cache version instead of geometry.
    _attach_finite_scan_support(candidates, plane_points)
    _attach_local_bend_support(candidates, plane_points, plane_normals)
    line_objects = cluster_line_candidates(candidates)
    _attach_case_relative_features(line_objects)
    validate_line_object_feature_schema(line_objects)
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.get("candidate_source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    result = dict(pool)
    result.update(
        {
            "algorithm": "f1_uniform_scan_only_candidate_pool_mesh_v4",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_point_serialization": False,
            "transient_sampled_point_count": int(len(points)),
            "transient_plane_point_count": int(len(plane_points)),
            "candidate_count": len(candidates),
            "line_object_count": len(line_objects),
            "candidate_source_counts": dict(sorted(source_counts.items())),
            "candidates": candidates,
            "line_objects": line_objects,
        }
    )
    generator_diagnostics = dict(result.get("generator_diagnostics") or {})
    generator_diagnostics["scan_mesh_adjacency"] = diagnostics
    result["generator_diagnostics"] = generator_diagnostics
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-mm", type=float, default=1.8)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read(input_dir / "manifest.json")
    rows = []
    skipped = []
    for index, row in enumerate(manifest.get("cases") or (), start=1):
        # Frozen corpus manifests use opaque ``case_token`` while standalone
        # pool manifests use ``case_id``.  Both identify the input filename;
        # accepting either preserves the same source-only augmentation path.
        case_id = str(row.get("case_id") or row["case_token"])
        pool = _read(input_dir / f"{case_id}.json")
        scan_path = Path(str(pool["scan_path"])).resolve()
        print(f"[{index}/{len(manifest['cases'])}] {case_id}: {scan_path.name}", flush=True)
        try:
            result = augment_case(pool, scan_path, voxel_mm=float(args.voxel_mm))
        except ValueError as exc:
            # Batch research augmentation is allowed to fail a case closed.
            # Never silently substitute a larger resource budget or a changed
            # geometry threshold after seeing an input-specific failure.
            skipped.append({"case_id": case_id, "reason": str(exc)})
            print(f"[{index}/{len(manifest['cases'])}] {case_id}: SKIP {exc}", flush=True)
            continue
        output_path = output_dir / f"{case_id}.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                "case_id": case_id,
                "path": str(output_path),
                "candidate_count": result["candidate_count"],
                "line_object_count": result["line_object_count"],
                "candidate_source_counts": result["candidate_source_counts"],
            }
        )
    output_manifest = {
        "schema_version": 1,
        "algorithm": "f1_uniform_scan_only_candidate_pool_mesh_v4",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "case_count": len(rows),
        "skipped_case_count": len(skipped),
        "skipped_cases": skipped,
        "source_point_serialization": False,
        "voxel_mm": float(args.voxel_mm),
        "cases": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
