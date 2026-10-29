#!/usr/bin/env python3
"""Source-only ordinary, permuted, or independent GCDF-V1 role."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_gcdf_v1.config import canonical_json_bytes  # noqa: E402
from research.f1_gcdf_v1.core import (  # noqa: E402
    build_surface_graph, build_vote_candidates, canonicalize_sample,
    evaluate_causality, evaluate_landmark_paths, load_bound_sample,
    select_landmarks, semantic_bundle,
)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("PRODUCER_A", "PRODUCER_A_PERMUTED", "VALIDATOR_B"))
    parser.add_argument("--raw-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visual-scratch", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    sample = load_bound_sample(args.raw_source, permute_input=args.role == "PRODUCER_A_PERMUTED")
    nodes = canonicalize_sample(sample)
    graph = build_surface_graph(nodes, validator_backend=args.role == "VALIDATOR_B")
    landmarks, by_component = select_landmarks(nodes, graph)
    path_result = evaluate_landmark_paths(nodes, graph, by_component)
    candidate_result = build_vote_candidates(nodes, graph, path_result)
    causal_result = evaluate_causality(nodes, path_result, candidate_result)
    bundle = semantic_bundle(
        sample=sample, nodes=nodes, graph=graph, landmarks=landmarks,
        path_result=path_result, candidate_result=candidate_result,
        causal_result=causal_result, role=args.role,
    )
    bundle["wall_seconds"] = perf_counter() - start
    bundle["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _atomic_write(args.output, canonical_json_bytes(bundle) + b"\n")
    if args.visual_scratch is not None and args.role == "PRODUCER_A":
        scratch = {
            "path_result": path_result,
            "candidate_result": candidate_result,
            "causal_result": causal_result,
        }
        _atomic_write(args.visual_scratch, canonical_json_bytes(scratch) + b"\n")
    print(json.dumps({
        "role": args.role, "decision": bundle["decision"],
        "semantic_sha256": bundle["semantic_sha256"],
        "wall_seconds": bundle["wall_seconds"], "peak_rss_bytes": bundle["peak_rss_bytes"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
