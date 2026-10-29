#!/usr/bin/env python3
"""Execute one frozen SDRA-V1 producer, permutation, or validator role."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from research.f1_sdra_v1.config import canonical_json_bytes  # noqa:E402


def _atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor,temp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,path)
    except BaseException:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise


def _atomic_npz(path:Path,arrays:dict[str,np.ndarray])->None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor,temp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".npz",dir=path.parent); os.close(descriptor)
    try:
        np.savez_compressed(temp,**arrays); os.replace(temp,path)
    except BaseException:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--role",required=True,choices=("PRODUCER_A","PRODUCER_A_PERMUTED","VALIDATOR_B")); parser.add_argument("--raw-source",required=True,type=Path); parser.add_argument("--output",required=True,type=Path); parser.add_argument("--visual-scratch",type=Path); args=parser.parse_args()
    start=perf_counter()
    if args.role=="VALIDATOR_B":
        from research.f1_sdra_v1.independent_validator import validate_source
        semantic,scratch=validate_source(args.raw_source)
    else:
        from research.f1_sdra_v1.core import evaluate_full,load_bound_source,visual_scratch_arrays
        source=load_bound_source(args.raw_source,permute_input=args.role=="PRODUCER_A_PERMUTED"); evaluation=evaluate_full(source); semantic=evaluation.semantic_payload; scratch=visual_scratch_arrays(evaluation)
    bundle={**semantic,"role":args.role,"wall_seconds":perf_counter()-start,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    _atomic(args.output,canonical_json_bytes(bundle)+b"\n")
    if args.visual_scratch is not None and args.role=="PRODUCER_A": _atomic_npz(args.visual_scratch,scratch)
    print(json.dumps({"role":args.role,"decision":semantic["decision"],"semantic_sha256":semantic["semantic_sha256"],"wall_seconds":bundle["wall_seconds"],"peak_rss_bytes":bundle["peak_rss_bytes"]},sort_keys=True,separators=(",",":")))
    return 0


if __name__=="__main__": raise SystemExit(main())
