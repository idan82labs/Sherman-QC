#!/usr/bin/env python3
"""Build the immutable SDRA-V1 preexecution lock and one-shot claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from research.f1_sdra_v1.config import CONFIG,CONFIG_SHA256,canonical_json_bytes,sha_json  # noqa:E402

CODE_PATHS=("research/f1_sdra_v1/__init__.py","research/f1_sdra_v1/config.py","research/f1_sdra_v1/core.py","research/f1_sdra_v1/independent_validator.py","research/f1_sdra_v1/tamper.py","research/f1_sdra_v1/visual.py","scripts/run_f1_sdra_v1_role.py","scripts/run_f1_sdra_v1_causal_preflight.py","scripts/run_f1_sdra_v1_source_free_regressions.py","scripts/build_f1_sdra_v1_source_binding.py","scripts/build_f1_sdra_v1_lock.py")
INPUT_PATHS=("docs/research/PROFESSOR_F1_POST_GCDF_NEXT_ARCHITECTURE_GPT56_PRO_RESPONSE_20260809.txt","docs/research/SDRA_V1_SOURCE_BINDING_V1_20260809.json","docs/research/SDRA_V1_SOURCE_FREE_REGRESSIONS_V1_20260809.json")


def _sha(path:Path)->str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def _atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor,temp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    with os.fdopen(descriptor,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def allocation_proof()->dict[str,Any]:
    n=CONFIG.sample_count;k=384
    ledger={"integer_and_metric_xyz":n*3*16,"neighbor_ids":n*k*4,"neighbor_squared":n*k*8,"moment_working_blocks":CONFIG.block_size*k*3*8*2,"per_point_metrics":n*160,"kdtree_and_parser_allowance":1_200_000_000,"python_numpy_overhead_allowance":1_800_000_000,"single_render_canvas_and_projection":CONFIG.global_image_pixels**2*4+CONFIG.sample_count*64}
    total=sum(ledger.values())
    if total>=CONFIG.peak_rss_ceiling_bytes: raise ValueError("SDRA static allocation proof exceeds ceiling")
    value={"ledger_bytes":ledger,"calculated_maximum_bytes":total,"rss_ceiling_bytes":CONFIG.peak_rss_ceiling_bytes,"headroom_bytes":CONFIG.peak_rss_ceiling_bytes-total,"roles_sequential":True,"workers_per_role":1,"neighbor_tables_serialized":False,"source_points_serialized":False,"n_by_n_matrix":False}
    value["resource_proof_sha256"]=sha_json(value); return value


def build()->dict[str,Any]:
    binding=json.loads((ROOT/INPUT_PATHS[1]).read_text(encoding="utf-8")); regression=json.loads((ROOT/INPUT_PATHS[2]).read_text(encoding="utf-8"))
    if binding.get("source_opened") or not regression.get("passed") or regression.get("source_opened"): raise ValueError("SDRA prelock evidence invalid")
    value={"schema_version":1,"profile":CONFIG.attempt_id,"authorization":{"token":CONFIG.authorization_token,"professor_response_relative":INPUT_PATHS[0],"professor_response_sha256":_sha(ROOT/INPUT_PATHS[0]),"execution_count":"EXACTLY_ONE","claim_consumption":"BEFORE_FIRST_ROLE_OPENS_RAW_SOURCE","gcdf_and_pgrr_closed":True},"configuration":CONFIG.as_frozen_dict(),"configuration_sha256":CONFIG_SHA256,"source_binding":binding,"source_free_regression":regression,"resource_proof":allocation_proof(),"code_sha256":{name:_sha(ROOT/name) for name in CODE_PATHS},"input_sha256":{name:_sha(ROOT/name) for name in INPUT_PATHS},"output_relative":"output/research/sdra-v1-c006-frozen-source-causal-preflight-v1","safety":{"research_only":True,"source_only":True,"target":False,"gold":False,"labels":False,"cad":False,"expected_counts":False,"benchmark_feedback":False,"gcdf_artifacts":False,"runtime_main":False,"production":False,"replay":False,"replacement_input":False,"threshold_mutation":False,"source_point_serialization":False,"visual_review_mandatory":True},"failure_policy":"EARLIEST_APPLICABLE_WALL_NO_RERUN_NO_REPAIR_NO_REPLACEMENT"}
    value["execution_lock_sha256"]=sha_json(value); return value


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True,type=Path); parser.add_argument("--claim-output",required=True,type=Path); args=parser.parse_args(); lock=build(); _atomic(args.output,canonical_json_bytes(lock)+b"\n")
    claim={"claim_state":"READY_FOR_FIRST_AND_ONLY_EXECUTION","execution_lock_sha256":lock["execution_lock_sha256"],"one_shot":True,"rerun_authorized":False}; _atomic(args.claim_output,canonical_json_bytes(claim)+b"\n"); print(canonical_json_bytes({"lock_sha256":lock["execution_lock_sha256"],"claim":claim}).decode("ascii")); return 0


if __name__=="__main__": raise SystemExit(main())
