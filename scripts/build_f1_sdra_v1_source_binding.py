#!/usr/bin/env python3
"""Build the no-source-open SDRA-V1 source/canonicalizer binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from research.f1_sdra_v1.config import CONFIG,CONFIG_SHA256,canonical_json_bytes,sha_json  # noqa:E402

RAW_RELATIVE="data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/35705010_A/scans/35705010_A_X.ply"
CANONICALIZER_FILES=("research/f1_ciip_v1/partition.py","research/f1_ciip_v1/primary_parser.py","research/f1_sdra_v1/core.py","research/f1_sdra_v1/independent_validator.py")


def _sha(path:Path)->str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def _atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor,temp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    with os.fdopen(descriptor,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def build()->dict[str,object]:
    raw=ROOT/RAW_RELATIVE
    if raw.stat().st_size!=CONFIG.raw_source_bytes: raise ValueError("SDRA source byte-count metadata mismatch")
    value={"schema_version":1,"binding_id":"SDRA_V1_C006_CONTENT_SELECTED_SOURCE_BINDING_V1","raw_source_relative":RAW_RELATIVE,"raw_source_sha256":CONFIG.raw_source_sha256,"raw_source_bytes":CONFIG.raw_source_bytes,"selection_rule":"UNIQUE_LEXICOGRAPHIC_MINIMUM_RAW_PLY_SHA256_OVER_EXACT_37_ROWS","bounded_canonicalizer":"F1_CIIP_V1_CANONICAL_SAMPLE_200000","quantized_point_ids":"LEXICOGRAPHIC_NEAREST_EVEN_L_OVER_2_POW_30","configuration_sha256":CONFIG_SHA256,"canonicalizer_code_sha256":{name:_sha(ROOT/name) for name in CANONICALIZER_FILES},"source_opened":False,"raw_source_hashed_during_binding":False}
    value["source_binding_sha256"]=sha_json(value); return value


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path); args=parser.parse_args(); value=build()
    if args.output: _atomic(args.output,canonical_json_bytes(value)+b"\n")
    else: print(canonical_json_bytes(value).decode("ascii"))
    return 0


if __name__=="__main__": raise SystemExit(main())
