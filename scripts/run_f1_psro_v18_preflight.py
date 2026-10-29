#!/usr/bin/env python3
"""Run the frozen PSRO_V1 sidecar on one native-mesh candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_paired_skin_ribbon_ownership_v18 import (  # noqa: E402
    PROFILE,
    SCAN_HEADER_BYTES,
    NativeMeshProbeContext,
    collapse_ribbon_duplicates,
    probe_candidate,
)
from scripts.select_f1_mesh_route_v16 import ROUTE_FULL, select_unified  # noqa: E402
from scripts.train_f1_topology_recall_v4 import _similar  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_adjacency_rows(pool: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for raw in pool.get("candidates") or ():
        row = dict(raw)
        if str(row.get("candidate_source") or "") != "scan_mesh_adjacency":
            continue
        source_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "")
        if source_id:
            result[source_id] = row
    return result


def run(pool: dict, v9: dict, v4: dict) -> tuple[dict, bytes]:
    started = time.perf_counter()
    selection = select_unified(pool, v9, v4)
    if selection["route"] != ROUTE_FULL:
        report = {
            "schema_version": 1,
            "profile": PROFILE,
            "route": selection["route"],
            "eligible_count": 0,
            "retained_count": 0,
            "records": [],
            "probe_cache_bytes": SCAN_HEADER_BYTES,
            "probe_cache_bound_bytes": SCAN_HEADER_BYTES,
            "reason": "non_native_full_mesh_route",
            "elapsed_seconds": time.perf_counter() - started,
        }
        return report, bytes(SCAN_HEADER_BYTES)

    scan_path = Path(str(pool["scan_path"]))
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    context = NativeMeshProbeContext(mesh)
    source_rows = _source_adjacency_rows(pool)
    incumbents = [dict(row) for row in selection["prediction_rows"]]
    incumbent_ids = {str(row.get("candidate_id") or "") for row in incumbents}
    incumbent_parent_pairs = []
    for row in incumbents:
        if str(row.get("representative_source") or "") == "scan_mesh_adjacency":
            try:
                incumbent_parent_pairs.append((int(row["surface1_id"]), int(row["surface2_id"])))
            except (KeyError, TypeError, ValueError):
                pass

    eligible = []
    for clustered in selection["rows"]:
        if str(clustered.get("representative_source") or "") != "scan_mesh_adjacency":
            continue
        if str(clustered.get("candidate_id") or "") in incumbent_ids:
            continue
        source_id = str(clustered.get("source_candidate_id") or "")
        source = source_rows.get(source_id)
        if source is None:
            continue
        # The cropped line is the frozen V11 line geometry; raw mesh ownership
        # fields come from the source candidate generated in the same pass.
        bound = dict(source)
        for key in ("candidate_id", "line_start", "line_end", "span_endpoints", "anchor", "axis_direction", "span_mm"):
            if clustered.get(key) is not None:
                bound[key] = clustered[key]
        eligible.append((clustered, bound))

    records = [
        probe_candidate(context, bound, incumbent_parent_pairs=incumbent_parent_pairs)
        for _clustered, bound in eligible
    ]
    canonical = collapse_ribbon_duplicates(records)
    surviving = []
    for record in canonical:
        clustered = next(row for row, _bound in eligible if str(row.get("candidate_id")) == record.candidate_id)
        if any(_similar(clustered, incumbent) for incumbent in incumbents):
            continue
        if any(_similar(clustered, prior) for prior in surviving):
            continue
        surviving.append(clustered)
    retained_ids = {str(row.get("candidate_id") or "") for row in surviving}
    encoded = bytes(SCAN_HEADER_BYTES) + b"".join(record.encode() for record in records)
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "route": selection["route"],
        "scan_sha256": _sha(scan_path),
        "candidate_pool_sha256": hashlib.sha256(
            json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "mesh_vertex_count": len(context.vertices),
        "mesh_triangle_count": len(context.triangles),
        "raw_owner_count": len(context.surfaces),
        "transition_parent_count": len(context.transition_parents),
        "mesh_scale": context.mesh_scale,
        "eligible_count": len(records),
        "retained_count": len(retained_ids),
        "retained_candidate_ids": sorted(retained_ids),
        "records": [record.to_summary() for record in records],
        "probe_cache_bytes": len(encoded),
        "probe_cache_bound_bytes": SCAN_HEADER_BYTES + 224 * len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "research_only": True,
        "source_only": True,
        "gold_used": False,
        "labels_used": False,
        "identity_used": False,
    }
    return report, encoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.input.read_text(encoding="utf-8"))
    report, cache = run(pool, joblib.load(args.v9_model), joblib.load(args.v4_model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    args.cache.write_bytes(cache)
    print(json.dumps({key: report[key] for key in ("route", "eligible_count", "retained_count", "probe_cache_bytes", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
