#!/usr/bin/env python3
"""Fresh-child worker for the EXP_CAD_0032 source-free performance lock.

The worker has no physical-source CLI.  It consumes only the deterministic
synthetic acquisition envelope prepared by ``exp_cad_0032_performance_harness``
and the frozen 8,077-node/19,267-edge CAD artifacts.  SOURCE mode executes the
V5 source-free entry point once and publishes canonical semantic/visual bytes.
RENDER mode executes the frozen native renderer once against those bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "EXP_CAD_0032_PERFORMANCE_WORKER_RESULT_V1"
EXPECTED_NODE_COUNT = 8_077
EXPECTED_EDGE_COUNT = 19_267
SYNTHETIC_TOKENS = ("SYNTHETIC_BENCH_A", "SYNTHETIC_BENCH_B")
EXPECTED_CAD_SHA256 = {
    "cad_preflight": "e96daf9288f59a8d7f156981f6d13e9fd3204b344523d96bba587bbbd84c8f82",
    "cad_nodes": "575c19d404efd6733051d611dd46f123fbbee3117be13ef001514d658e2618b0",
    "cad_edges": "0bf773fc65af9152bb8491830f98faf2e14c10fe999e8e4bf81a3db687513e3d",
    "cad_edge_geodesic_mm": "92605624e786450ca260134555b6e71bbac56875e9e3e49b4aceac6d5e9de185",
    "cad_mesh": "9e6571704780383e99b6fa4f2800fa69a50a3e187467698fe5dd7414d9d2a716",
}
QUALIFICATION_LOCK_SCHEMA = "EXP_CAD_0032_SOURCE_FREE_QUALIFICATION_LOCK_V1"
QUALIFICATION_LOCK_KEYS = frozenset(
    {
        "v5_preregistration", "parent_preregistration", "cad_preflight",
        "cad_nodes", "cad_edges", "cad_edge_geodesic_mm", "cad_mesh",
        "frozen_science", "frozen_execute_source", "frozen_interface",
        "v5_engine", "source_free_runner", "performance_worker",
        "performance_harness", "engine_tests", "performance_tests",
        "renderer", "benchmark_manifest",
    }
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        # Frozen parent publication semantics map diagnostic NaN/Inf to null.
        # This is output canonicalization only; scientific arrays are unchanged.
        return None
    return value


def canonical_json_write(path: Path, value: Any) -> str:
    """Atomically stream canonical JSON to disk and return its byte hash."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for text in encoder.iterencode(_jsonable(value)):
                block = text.encode("utf-8")
                handle.write(block)
                digest.update(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def verify_qualification_lock(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    path = path.resolve()
    if sha256_path(path) != expected_sha256:
        raise ValueError("EXP32 qualification lock packet hash mismatch")
    payload = _load_json(path)
    if (
        payload.get("schema") != QUALIFICATION_LOCK_SCHEMA
        or payload.get("experiment_id") != "EXP_CAD_0032"
        or payload.get("status") != "PASS_SOURCE_FREE_LOCK_EXECUTION_AUTHORIZED"
        or payload.get("physical_source_execution_authorized") is not False
    ):
        raise ValueError("EXP32 qualification lock identity/status mismatch")
    files = payload.get("bound_files")
    if not isinstance(files, Mapping) or set(files) != set(QUALIFICATION_LOCK_KEYS):
        raise ValueError("EXP32 qualification lock bound-file key mismatch")
    for key, row in files.items():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError(f"EXP32 qualification lock malformed binding: {key}")
        bound = (ROOT / row["path"]).resolve()
        if sha256_path(bound) != row.get("sha256"):
            raise ValueError(f"EXP32 qualification lock file mismatch: {key}")
    return payload


def _manifest_path(manifest_path: Path, manifest: Mapping[str, Any], key: str) -> Path:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(key), Mapping):
        raise ValueError(f"benchmark manifest lacks files.{key}")
    row = files[key]
    relative = row.get("relative_path")
    expected = row.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"invalid benchmark manifest file binding: {key}")
    path = (manifest_path.parent / relative).resolve()
    if sha256_path(path) != expected:
        raise ValueError(f"benchmark file hash mismatch: {key}")
    return path


def _retained_identity(values: Any) -> tuple[int, str]:
    array = np.asarray(values, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError("retained row identity must be one-dimensional")
    canonical = np.ascontiguousarray(array.astype("<u8", copy=False))
    return int(len(canonical)), hashlib.sha256(canonical.tobytes()).hexdigest()


def _exact_operation_counters(metrics: Mapping[str, Any]) -> dict[str, int]:
    nested = metrics.get("operation_counters")
    source = nested if isinstance(nested, Mapping) else metrics
    counters: dict[str, int] = {}
    for key, value in source.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("elapsed", "second", "time", "rss", "byte")):
            continue
        if not any(token in lowered for token in ("count", "call", "build", "hit", "query")):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            continue
        counters[str(key)] = int(value)
    if not counters:
        raise ValueError("engine metrics expose no exact operation counters")
    return dict(sorted(counters.items()))


def _source(args: argparse.Namespace) -> int:
    verify_qualification_lock(args.qualification_lock, args.qualification_lock_sha256)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("source output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.benchmark_manifest.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "EXP_CAD_0032_SYNTHETIC_ACQUISITION_ENVELOPE_V1":
        raise ValueError("unexpected EXP32 benchmark manifest schema")
    if manifest.get("cad_node_count") != EXPECTED_NODE_COUNT or manifest.get(
        "cad_edge_count"
    ) != EXPECTED_EDGE_COUNT:
        raise ValueError("EXP32 CAD graph cardinality is not frozen")
    if manifest.get("opaque_member_tokens") != list(SYNTHETIC_TOKENS):
        raise ValueError("EXP32 synthetic token contract mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("EXP32 manifest lacks exact file bindings")
    for key in ("cad_nodes", "cad_edges", "cad_edge_geodesic_mm", "cad_mesh"):
        row = files.get(key)
        if not isinstance(row, Mapping) or row.get("sha256") != EXPECTED_CAD_SHA256[key]:
            raise ValueError(f"EXP32 frozen CAD hash mismatch: {key}")
    if sha256_path(args.cad_preflight.resolve()) != EXPECTED_CAD_SHA256["cad_preflight"]:
        raise ValueError("EXP32 frozen CAD preflight hash mismatch")
    maximum_rows = int(manifest["retained_rows_per_member"])
    rows = int(args.rows_per_member)
    if rows < 1 or rows > maximum_rows:
        raise ValueError("rows-per-member outside prepared envelope")

    file_load_counters = {
        "json_load_call_count": 4,
        "npy_load_call_count": 7,
        "triangle_mesh_load_call_count": 1,
        "scientific_execute_call_count": 1,
        "semantic_canonical_write_call_count": 1,
        "visual_canonical_write_call_count": 1,
    }
    nodes = np.load(_manifest_path(manifest_path, manifest, "cad_nodes"), allow_pickle=False)
    edges = np.load(_manifest_path(manifest_path, manifest, "cad_edges"), allow_pickle=False)
    weights = np.load(
        _manifest_path(manifest_path, manifest, "cad_edge_geodesic_mm"),
        allow_pickle=False,
    )
    if len(nodes) != EXPECTED_NODE_COUNT or len(edges) != EXPECTED_EDGE_COUNT:
        raise ValueError("loaded CAD graph cardinality drift")
    mesh_path = _manifest_path(manifest_path, manifest, "cad_mesh")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError("benchmark CAD mesh is empty")

    def source_array(key: str) -> np.ndarray:
        path = _manifest_path(manifest_path, manifest, key)
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if value.shape != (maximum_rows, 3) or value.dtype != np.dtype("<f8"):
            raise ValueError(f"unexpected synthetic source array: {key}")
        return np.asarray(value[:rows], dtype=np.float64)

    raw_first = source_array("first_points")
    normals_first = source_array("first_normals")
    raw_second = source_array("second_points")
    normals_second = source_array("second_normals")
    v5_prereg = _load_json(args.v5_preregistration)
    parent_prereg = _load_json(args.parent_preregistration)
    cad_preflight = _load_json(args.cad_preflight)
    runner = _load_module(args.source_free_runner, "exp32_source_free_runner_bound")
    execute = getattr(runner, "execute_source_v5", None)
    if not callable(execute):
        raise ValueError("source-free runner lacks execute_source_v5")
    semantic, visual, filtered, metrics = execute(
        v5_prereg=v5_prereg,
        parent_prereg=parent_prereg,
        cad_preflight=cad_preflight,
        nodes=nodes,
        edges=edges,
        edge_weights=weights,
        mesh=mesh,
        raw_first=raw_first,
        normals_first=normals_first,
        raw_second=raw_second,
        normals_second=normals_second,
        transform_first=np.eye(4, dtype=np.float64),
        transform_second=np.eye(4, dtype=np.float64),
        first_token=SYNTHETIC_TOKENS[0],
        second_token=SYNTHETIC_TOKENS[1],
    )
    if not isinstance(semantic, Mapping) or not isinstance(visual, Mapping):
        raise ValueError("source-free runner returned invalid scientific payloads")
    if not isinstance(filtered, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("source-free runner returned invalid audit payloads")
    first_filtered = filtered.get("first")
    second_filtered = filtered.get("second")
    if not isinstance(first_filtered, Mapping) or not isinstance(second_filtered, Mapping):
        raise ValueError("source-free runner filtered payload lacks member namespaces")
    first_count, first_hash = _retained_identity(first_filtered["original_row_ids"])
    second_count, second_hash = _retained_identity(second_filtered["original_row_ids"])
    if first_count != rows or second_count != rows:
        raise ValueError("clean synthetic envelope did not retain every source row")

    semantic_hash = canonical_json_write(output_root / "SEMANTIC_PAYLOAD.json", semantic)
    visual_hash = canonical_json_write(output_root / "VISUAL_PAYLOAD.json", visual)
    operation_counters = _exact_operation_counters(metrics)
    raw_checks = metrics.get("equivalence_checks", {})
    equivalence_checks = (
        {str(key): bool(value) for key, value in raw_checks.items()}
        if isinstance(raw_checks, Mapping)
        else {}
    )
    # Elapsed-time and profiling fields are intentionally segregated from the
    # byte-compared scientific tree.  The external harness owns authoritative
    # wall/RSS measurements; this file is monitor evidence only.
    canonical_json_write(
        args.monitoring_output.resolve(),
        {
            "schema": "EXP_CAD_0032_NONCANONICAL_ENGINE_MONITORING_METRICS_V1",
            "excluded_from_scientific_tree_identity": True,
            "metrics": _jsonable(metrics),
        },
    )
    result = {
        "schema": SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "mode": "SOURCE",
        "status": "COMPLETE",
        "rows_per_member": rows,
        "cad_node_count": int(len(nodes)),
        "cad_edge_count": int(len(edges)),
        "semantic_canonical_sha256": semantic_hash,
        "visual_canonical_sha256": visual_hash,
        "semantic_schema": semantic.get("schema"),
        "visual_schema": visual.get("schema"),
        "retained_row_identity": {
            "first": {"count": first_count, "sha256_uint64_le": first_hash},
            "second": {"count": second_count, "sha256_uint64_le": second_hash},
        },
        "worker_operation_counters": file_load_counters,
        "engine_operation_counters": operation_counters,
        "equivalence_checks": equivalence_checks,
        "noncanonical_engine_monitoring_metrics_excluded_from_scientific_tree_identity": True,
        "physical_scan_or_evaluator_input_used": False,
    }
    canonical_json_write(output_root / "WORKER_RESULT.json", result)
    print(json.dumps({"status": "COMPLETE", "mode": "SOURCE"}))
    return 0


def _renderer_manifest(native_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(row for row in native_root.rglob("*") if row.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(native_root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
            }
        )
    return {"file_count": len(files), "files": files}


def _reference(args: argparse.Namespace) -> int:
    """Run the exact frozen parent on the preregistered smaller differential."""

    verify_qualification_lock(args.qualification_lock, args.qualification_lock_sha256)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("reference output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.benchmark_manifest.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "EXP_CAD_0032_SYNTHETIC_ACQUISITION_ENVELOPE_V1":
        raise ValueError("unexpected EXP32 benchmark manifest schema")
    if manifest.get("cad_node_count") != EXPECTED_NODE_COUNT or manifest.get(
        "cad_edge_count"
    ) != EXPECTED_EDGE_COUNT:
        raise ValueError("EXP32 CAD graph cardinality is not frozen")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("EXP32 manifest lacks exact file bindings")
    for key in ("cad_nodes", "cad_edges", "cad_edge_geodesic_mm", "cad_mesh"):
        row = files.get(key)
        if not isinstance(row, Mapping) or row.get("sha256") != EXPECTED_CAD_SHA256[key]:
            raise ValueError(f"EXP32 frozen CAD hash mismatch: {key}")
    if sha256_path(args.cad_preflight.resolve()) != EXPECTED_CAD_SHA256["cad_preflight"]:
        raise ValueError("EXP32 frozen CAD preflight hash mismatch")
    maximum_rows = int(manifest["retained_rows_per_member"])
    rows = int(args.rows_per_member)
    if rows != 100_000:
        raise ValueError("frozen-parent differential is fixed at 100000 rows/member")
    nodes = np.load(_manifest_path(manifest_path, manifest, "cad_nodes"), allow_pickle=False)
    edges = np.load(_manifest_path(manifest_path, manifest, "cad_edges"), allow_pickle=False)
    weights = np.load(
        _manifest_path(manifest_path, manifest, "cad_edge_geodesic_mm"), allow_pickle=False
    )
    mesh = o3d.io.read_triangle_mesh(
        str(_manifest_path(manifest_path, manifest, "cad_mesh"))
    )

    def source_array(key: str) -> np.ndarray:
        value = np.load(
            _manifest_path(manifest_path, manifest, key),
            allow_pickle=False,
            mmap_mode="r",
        )
        if value.shape != (maximum_rows, 3) or value.dtype != np.dtype("<f8"):
            raise ValueError(f"unexpected synthetic source array: {key}")
        return np.asarray(value[:rows], dtype=np.float64)

    raw_first = source_array("first_points")
    normals_first = source_array("first_normals")
    raw_second = source_array("second_points")
    normals_second = source_array("second_normals")
    prereg = _load_json(args.parent_preregistration)
    cad_preflight = _load_json(args.cad_preflight)
    frozen = _load_module(args.frozen_parent_runner, "exp32_frozen_parent_bound")
    execute = getattr(frozen, "execute_source", None)
    if not callable(execute):
        raise ValueError("frozen parent runner lacks execute_source")
    semantic, visual = execute(
        prereg=prereg,
        cad_preflight=cad_preflight,
        nodes=nodes,
        edges=edges,
        edge_weights=weights,
        mesh=mesh,
        raw_first=raw_first,
        normals_first=normals_first,
        raw_second=raw_second,
        normals_second=normals_second,
        transform_first=np.eye(4, dtype=np.float64),
        transform_second=np.eye(4, dtype=np.float64),
        first_token=SYNTHETIC_TOKENS[0],
        second_token=SYNTHETIC_TOKENS[1],
    )
    semantic_hash = canonical_json_write(output_root / "SEMANTIC_PAYLOAD.json", semantic)
    visual_hash = canonical_json_write(output_root / "VISUAL_PAYLOAD.json", visual)
    identity = np.arange(rows, dtype="<u8")
    identity_hash = hashlib.sha256(identity.tobytes()).hexdigest()
    result = {
        "schema": SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "mode": "FROZEN_PARENT_REFERENCE",
        "status": "COMPLETE",
        "rows_per_member": rows,
        "cad_node_count": int(len(nodes)),
        "cad_edge_count": int(len(edges)),
        "semantic_canonical_sha256": semantic_hash,
        "visual_canonical_sha256": visual_hash,
        "semantic_schema": semantic.get("schema"),
        "visual_schema": visual.get("schema"),
        "retained_row_identity": {
            "first": {"count": rows, "sha256_uint64_le": identity_hash},
            "second": {"count": rows, "sha256_uint64_le": identity_hash},
        },
        "worker_operation_counters": {
            "json_load_call_count": 3,
            "npy_load_call_count": 7,
            "triangle_mesh_load_call_count": 1,
            "scientific_execute_call_count": 1,
            "semantic_canonical_write_call_count": 1,
            "visual_canonical_write_call_count": 1,
        },
        "physical_scan_or_evaluator_input_used": False,
    }
    canonical_json_write(output_root / "WORKER_RESULT.json", result)
    print(json.dumps({"status": "COMPLETE", "mode": "FROZEN_PARENT_REFERENCE"}))
    return 0


def _render(args: argparse.Namespace) -> int:
    verify_qualification_lock(args.qualification_lock, args.qualification_lock_sha256)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("renderer output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    payload_path = args.visual_payload.resolve()
    payload = _load_json(payload_path)
    renderer = _load_module(args.renderer, "exp32_renderer_bound")
    render = getattr(renderer, "render_native_visuals", None)
    if not callable(render):
        raise ValueError("renderer lacks render_native_visuals")
    native_root = output_root / "native"
    render(payload, native_root)
    manifest = _renderer_manifest(native_root)
    if manifest["file_count"] < 1:
        raise ValueError("renderer emitted no files")
    manifest_hash = canonical_json_write(output_root / "RENDER_MANIFEST.json", manifest)
    result = {
        "schema": SCHEMA,
        "experiment_id": "EXP_CAD_0032",
        "mode": "RENDER",
        "status": "COMPLETE",
        "visual_payload_sha256": sha256_path(payload_path),
        "render_manifest_canonical_sha256": manifest_hash,
        "rendered_file_count": manifest["file_count"],
        "worker_operation_counters": {
            "visual_json_load_call_count": 1,
            "native_renderer_call_count": 1,
            "rendered_file_hash_call_count": manifest["file_count"],
        },
        "physical_scan_or_evaluator_input_used": False,
    }
    canonical_json_write(output_root / "WORKER_RESULT.json", result)
    print(json.dumps({"status": "COMPLETE", "mode": "RENDER"}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--benchmark-manifest", required=True, type=Path)
    source.add_argument("--rows-per-member", required=True, type=int)
    source.add_argument("--v5-preregistration", required=True, type=Path)
    source.add_argument("--parent-preregistration", required=True, type=Path)
    source.add_argument("--cad-preflight", required=True, type=Path)
    source.add_argument("--source-free-runner", required=True, type=Path)
    source.add_argument("--monitoring-output", required=True, type=Path)
    source.add_argument("--qualification-lock", required=True, type=Path)
    source.add_argument("--qualification-lock-sha256", required=True)
    source.add_argument("--output-root", required=True, type=Path)
    render = subparsers.add_parser("render")
    render.add_argument("--visual-payload", required=True, type=Path)
    render.add_argument("--renderer", required=True, type=Path)
    render.add_argument("--output-root", required=True, type=Path)
    render.add_argument("--qualification-lock", required=True, type=Path)
    render.add_argument("--qualification-lock-sha256", required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--benchmark-manifest", required=True, type=Path)
    reference.add_argument("--rows-per-member", required=True, type=int)
    reference.add_argument("--parent-preregistration", required=True, type=Path)
    reference.add_argument("--cad-preflight", required=True, type=Path)
    reference.add_argument("--frozen-parent-runner", required=True, type=Path)
    reference.add_argument("--output-root", required=True, type=Path)
    reference.add_argument("--qualification-lock", required=True, type=Path)
    reference.add_argument("--qualification-lock-sha256", required=True)
    args = parser.parse_args(argv)
    if args.mode == "source":
        return _source(args)
    if args.mode == "reference":
        return _reference(args)
    return _render(args)


if __name__ == "__main__":
    raise SystemExit(main())
