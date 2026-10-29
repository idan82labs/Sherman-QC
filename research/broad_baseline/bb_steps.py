#!/usr/bin/env python3
"""Single heavy steps of the best-F1 lane, each run in its own process by run_lane.py.

Subcommands (all read-only with respect to the repo; they only write --out paths):

  decompose  strict-blind scan-only F1 decomposition, exactly what
             scripts/run_f1_accuracy_validation.py --strict-blind-inference
             --route-mode apply computes per scan (part_id=None):
             backend.patch_graph_latent_decomposition.run_patch_graph_latent_decomposition
  pool       bounded scan-only candidate pool, via the unchanged
             scripts/build_f1_bounded_pool_from_freeze.build_one (200k point
             contract, voxel 1.8 mm, mesh adjacency disabled above 200k PLY vertices)
  ply-proxy  lossless STL/OBJ -> PLY source-frame proxy (trimesh, process=False),
             the same conversion scripts/sync_bend_line_gold_corpus.py used
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402


def _decompose(args: argparse.Namespace) -> int:
    from bend_inspection_pipeline import load_bend_runtime_config
    from patch_graph_latent_decomposition import run_patch_graph_latent_decomposition

    started = perf_counter()
    runtime_config = load_bend_runtime_config(None)
    result = run_patch_graph_latent_decomposition(
        scan_path=str(Path(args.scan).resolve()),
        runtime_config=runtime_config,
        part_id=None,  # strict blind: no part identity reaches inference
        cache_dir=str(Path(args.cache_dir).resolve()),
        route_mode="apply",
    )
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "decompose_done", "seconds": round(perf_counter() - started, 3),
                      "owned_bend_regions": len(result.owned_bend_regions),
                      "exact_bend_count": int(result.exact_bend_count), "out": str(out)}))
    return 0


def _pool(args: argparse.Namespace) -> int:
    from scripts.build_f1_bounded_pool_from_freeze import build_one

    started = perf_counter()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)  # build_one expects main() to have created it
    row = build_one(
        case={
            "case_token": args.token,
            "source_scan_sha256": args.source_sha,
            "source_scan_path": str(Path(args.source_scan).resolve()),
        },
        decomposition_roots=[Path(args.decomposition_root).resolve()],
        source_case_id=None,
        output_dir=Path(args.out_dir).resolve(),
        voxel_mm=float(args.voxel_mm),
    )
    row["seconds"] = round(perf_counter() - started, 3)
    Path(args.out_dir, f"{args.token}.row.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "pool_done", **row}))
    return 0


def _ply_proxy(args: argparse.Namespace) -> int:
    import trimesh

    source = Path(args.scan).resolve()
    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"{source}: scene contains no mesh")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{source}: unsupported mesh type {type(loaded).__name__}")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    loaded.export(target, file_type="ply", encoding="binary_little_endian")
    print(json.dumps({"event": "ply_proxy_done", "vertices": int(len(loaded.vertices)),
                      "faces": int(len(loaded.faces)), "out": str(target)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("decompose")
    p.add_argument("--scan", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--cache-dir", required=True)
    p.set_defaults(func=_decompose)
    p = sub.add_parser("pool")
    p.add_argument("--decomposition-root", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--source-scan", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--voxel-mm", type=float, default=1.8)
    p.set_defaults(func=_pool)
    p = sub.add_parser("ply-proxy")
    p.add_argument("--scan", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_ply_proxy)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
