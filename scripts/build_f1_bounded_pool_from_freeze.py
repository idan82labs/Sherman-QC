#!/usr/bin/env python3
"""Rebuild frozen scan-only candidate pools under the global 200k contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_f1_scan_only_candidate_pool import (  # noqa: E402
    GLOBAL_POINT_BUDGET,
    _load_geometry_views,
    _scan_mesh_adjacency_candidates,
    build_case_pool,
)
from scripts.assemble_f1_gold27_candidate_pool import _geometry_sha256  # noqa: E402


DEFAULT_INFERENCE = ROOT / "output/research/f1-gold27-20260803/freeze/inference_manifest.json"
DEFAULT_DECOMPOSITIONS = (
    ROOT / "output/enterprise_benchmark/blind-ply-bounded-20260804-v2",
    ROOT / "output/enterprise_benchmark/shaked-count-backed-12-blind-baseline-20260717",
    ROOT / "output/research/f1-new-shaked-pairs-20260730-v1/baseline",
)
DEFAULT_OUTPUT = ROOT / "output/research/f1-gold27-20260803/candidate_pool_bounded_v2"
DEFAULT_SOURCE_POOL = ROOT / "output/research/f1-gold27-20260803/candidate_pool_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_ply_vertex_count(path: Path) -> int | None:
    """Read the ASCII PLY header without materializing scan geometry."""

    if path.suffix.lower() != ".ply":
        return None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            return None
        vertex_count: int | None = None
        for _line_index in range(4096):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path}: truncated PLY header")
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[-1])
                if vertex_count < 0:
                    raise ValueError(f"{path}: negative PLY vertex count")
            if line == "end_header":
                return vertex_count
        raise ValueError(f"{path}: PLY header exceeds deterministic read cap")


def _decomposition_path(
    roots: list[Path],
    case_token: str,
    *,
    source_scan_sha256: str | None = None,
    source_case_id: str | None = None,
) -> Path:
    matches = sorted(
        {
            path.resolve()
            for root in roots
            for path in (root / case_token / "result").glob("**/decomposition_result.json")
        }
    )
    if not matches and source_case_id:
        matches = sorted(
            {
                path.resolve()
                for root in roots
                for path in root.glob(
                    f"**/{source_case_id}/**/decomposition_result.json"
                )
            }
        )
    if not matches and source_scan_sha256:
        for root in roots:
            for path in sorted(root.glob("**/decomposition_result.json")):
                payload = _read(path)
                cache_identity = dict((payload.get("provenance") or {}).get("cache_identity") or {})
                if str(cache_identity.get("scan_sha256") or "") == source_scan_sha256:
                    matches.append(path.resolve())
        matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError(
            f"{case_token}: expected one frozen decomposition, found {len(matches)}"
        )
    return matches[0].resolve()


def build_one(
    *,
    case: dict[str, Any],
    decomposition_roots: list[Path],
    source_case_id: str | None,
    output_dir: Path,
    voxel_mm: float,
) -> dict[str, Any]:
    token = str(case["case_token"])
    decomposition_path = _decomposition_path(
        decomposition_roots,
        token,
        source_scan_sha256=str(case["source_scan_sha256"]),
        source_case_id=source_case_id,
    )
    decomposition = _read(decomposition_path)
    provenance = dict(decomposition.get("provenance") or {})
    if provenance.get("coordinate_frame") != "scan_raw":
        raise ValueError(f"{token}: decomposition is not in scan_raw")
    stripped_part_identifier = decomposition.pop("part_id", None)
    decomposition_scan_path = Path(str(decomposition["scan_path"])).resolve()
    expected_source = Path(str(case["source_scan_path"])).resolve()
    expected_hash = str(case["source_scan_sha256"])
    actual_hash = _sha256(decomposition_scan_path)
    representation_proxy_used = False
    if actual_hash != expected_hash:
        if decomposition_scan_path == expected_source:
            raise ValueError(
                f"{token}: decomposition scan hash does not match frozen source "
                f"({actual_hash} != {expected_hash})"
            )
        if not expected_source.is_file():
            raise ValueError(f"{token}: frozen representation proxy is missing")
        if _geometry_sha256(decomposition_scan_path) != _geometry_sha256(expected_source):
            raise ValueError(
                f"{token}: decomposition scan hash and geometry do not match frozen source "
                f"({actual_hash} != {expected_hash})"
            )
        scan_path = expected_source
        representation_proxy_used = True
    else:
        scan_path = decomposition_scan_path
    if expected_source.is_file() and _sha256(expected_source) != expected_hash:
        raise ValueError(f"{token}: frozen source hash changed")
    declared_vertex_count = _declared_ply_vertex_count(scan_path)

    points, normals, plane_points, plane_normals = _load_geometry_views(
        scan_path,
        plane_voxel_mm=float(voxel_mm),
    )
    if declared_vertex_count is not None and declared_vertex_count > GLOBAL_POINT_BUDGET:
        # Mesh adjacency cannot be decimated without changing connectivity.
        # Disable that lane before its second full-file read; every remaining
        # generator consumes the one deterministic bounded working cloud.
        mesh_candidates = []
        mesh_diagnostics = {
            "available": False,
            "reason": "disabled_by_global_vertex_budget",
            "declared_vertex_count": declared_vertex_count,
            "consumed_vertex_count": 0,
            "candidate_count": 0,
        }
    else:
        mesh_candidates, mesh_diagnostics = _scan_mesh_adjacency_candidates(scan_path)
    if int(mesh_diagnostics.get("vertex_count") or 0) > GLOBAL_POINT_BUDGET:
        raise ValueError(
            f"{token}: scan-mesh adjacency consumed "
            f"{int(mesh_diagnostics['vertex_count'])} vertices, exceeding the strict "
            f"global {GLOBAL_POINT_BUDGET}-point contract"
        )
    payload = build_case_pool(
        decomposition=decomposition,
        points=points,
        normals=normals,
        plane_points=plane_points,
        plane_normals=plane_normals,
        scan_mesh_adjacency=mesh_candidates,
    )
    if int(payload["transient_sampled_point_count"]) > GLOBAL_POINT_BUDGET:
        raise AssertionError("global working-point contract regressed")
    payload["case_id"] = token
    payload["scan_path"] = str(scan_path)
    payload["coordinate_frame"] = "scan_raw"
    payload["source_scan_sha256"] = expected_hash
    payload["storage_metadata_sanitization"] = {
        "part_identifier_removed_before_candidate_generation": (
            stripped_part_identifier is not None
        ),
        "removed_identifier_value_serialized": False,
        "frozen_representation_proxy_used": representation_proxy_used,
        "decomposition_geometry_equivalence_verified": True,
    }
    payload["generator_diagnostics"]["scan_mesh_adjacency"] = mesh_diagnostics
    destination = output_dir / f"{token}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "case_token": token,
        "path": str(destination),
        "candidate_count": int(payload["candidate_count"]),
        "line_object_count": int(payload["line_object_count"]),
        "working_point_count": int(payload["transient_sampled_point_count"]),
        "plane_point_count": int(payload["transient_plane_point_count"]),
        "source_scan_sha256": expected_hash,
    }


def _build_task(arguments: dict[str, Any]) -> dict[str, Any]:
    return build_one(**arguments)


def _memory_safe_worker_cap(
    *, measured_peak_rss_bytes: int, memory_budget_bytes: int, host_core_limit: int = 18
) -> int:
    measured = int(measured_peak_rss_bytes)
    budget = int(memory_budget_bytes)
    core_limit = int(host_core_limit)
    if measured <= 0 or budget <= 0 or core_limit <= 0:
        raise ValueError("memory and core limits must be positive")
    cap = min(core_limit, budget // measured)
    if cap < 1:
        raise ValueError(
            "measured per-job peak RSS exceeds the configured host memory budget"
        )
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--decomposition-root", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-candidate-root", type=Path, default=DEFAULT_SOURCE_POOL)
    parser.add_argument("--case-token", action="append")
    parser.add_argument("--voxel-mm", type=float, default=1.8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--measured-peak-rss-bytes", type=int)
    parser.add_argument("--host-memory-budget-gib", type=float, default=96.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest = _read(args.inference_manifest.resolve())
    selected = set(args.case_token or ())
    cases = [
        dict(case)
        for case in manifest.get("cases") or ()
        if not selected or str(case["case_token"]) in selected
    ]
    if selected - {str(case["case_token"]) for case in cases}:
        raise ValueError("one or more requested case tokens are absent from the freeze")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decomposition_roots = [
        path.resolve() for path in (args.decomposition_root or DEFAULT_DECOMPOSITIONS)
    ]
    source_candidate_root = args.source_candidate_root.resolve()
    workers = int(args.workers)
    if workers < 1 or workers > 18:
        raise ValueError("workers must be between 1 and the 18-core host limit")
    if workers > 1:
        measured = int(args.measured_peak_rss_bytes or 0)
        if measured <= 0:
            raise ValueError("parallel execution requires measured-peak-rss-bytes")
        memory_budget = int(float(args.host_memory_budget_gib) * 1024**3)
        memory_safe_workers = _memory_safe_worker_cap(
            measured_peak_rss_bytes=measured,
            memory_budget_bytes=memory_budget,
        )
        if workers > memory_safe_workers:
            raise ValueError(
                f"requested {workers} workers exceeds measured memory-safe cap "
                f"{memory_safe_workers}"
            )

    tasks = []
    rows = []
    for case in cases:
        existing_path = output_dir / f"{case['case_token']}.json"
        if args.skip_existing and existing_path.is_file():
            existing = _read(existing_path)
            if (
                existing.get("source_point_serialization") is not False
                or int(existing.get("transient_sampled_point_count") or 0)
                > GLOBAL_POINT_BUDGET
                or str(existing.get("working_point_contract_version") or "")
                != "GLOBAL_POINT_BUDGET_V1"
                or int(existing.get("working_point_budget") or 0)
                != GLOBAL_POINT_BUDGET
                or str(existing.get("source_scan_sha256") or "")
                != str(case["source_scan_sha256"])
            ):
                raise ValueError(f"{case['case_token']}: existing bounded pool is invalid")
            rows.append(
                {
                    "case_token": str(case["case_token"]),
                    "path": str(existing_path),
                    "candidate_count": int(existing.get("candidate_count") or 0),
                    "line_object_count": int(existing.get("line_object_count") or 0),
                    "working_point_count": int(existing.get("transient_sampled_point_count") or 0),
                    "plane_point_count": int(existing.get("transient_plane_point_count") or 0),
                    "source_scan_sha256": str(case["source_scan_sha256"]),
                }
            )
            continue
        source_candidate_path = source_candidate_root / f"{case['case_token']}.json"
        source_case_id = (
            str(_read(source_candidate_path).get("source_case_id") or "")
            if source_candidate_path.is_file()
            else ""
        )
        tasks.append(
            {
                "case": case,
                "decomposition_roots": decomposition_roots,
                "source_case_id": source_case_id or None,
                "output_dir": output_dir,
                "voxel_mm": float(args.voxel_mm),
            }
        )
    print(
        f"building {len(tasks)} bounded pools with workers={workers}; "
        f"reusing {len(rows)} verified outputs",
        flush=True,
    )
    if workers == 1:
        rows.extend(_build_task(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows.extend(executor.map(_build_task, tasks))
    rows.sort(key=lambda row: str(row["case_token"]))
    output_manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_bounded_scan_only_candidate_pool",
        "corpus_freeze_id": manifest.get("corpus_freeze_id"),
        "case_count": len(rows),
        "source_point_serialization": False,
        "working_point_contract_version": "GLOBAL_POINT_BUDGET_V1",
        "working_point_budget": GLOBAL_POINT_BUDGET,
        "cases": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
