#!/usr/bin/env python3
"""Invoke one immutable V33 PNFM source-free fixture exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[name]="1"

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research"/"f1_v33_pnfm_v1_lock.json"
OUT=ROOT/"output"/"research"/"f1-v33-pnfm-v1"
FIX=OUT/"source_free_fixtures";RESULTS=OUT/"source_free_results";POST=OUT/"postinvoke_visuals"
PRECOMMIT=OUT/"precommit_manifest.json";COUNTER=OUT/"invocation_events.jsonl"
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from backend.f1_projective_normal_fan_moment import canonical_result_bytes,select_unified_v33  # noqa:E402
from scripts.render_f1_v33_pnfm_source_free import render  # noqa:E402


def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_ids()->tuple[str,...]:return tuple(load(LOCK)["ordered_fixture_ids"])


def require_lock()->None:
    manifest=load(PRECOMMIT)
    if manifest.get("passed") is not True or manifest.get("token")!="V33_PNFM_MECHANISM_LOCKED":raise SystemExit("V33 mechanism is not frozen")
    for row in manifest["source_hashes"]+manifest["fixture_hashes"]+manifest["preinvoke_visual_hashes"]:
        path=ROOT/row["path"]
        if not path.exists() or sha(path)!=row["sha256"]:raise SystemExit(f"V33 frozen hash drift: {row['path']}")


def prior_invocations(fixture_id:str)->int:
    if not COUNTER.exists():return 0
    return sum(json.loads(line).get("event")=="PNFM_INVOKE" and json.loads(line).get("fixture_id")==fixture_id for line in COUNTER.read_text(encoding="utf-8").splitlines() if line.strip())


def append_invocation(fixture_id:str,run_tag:str,workers:int)->None:
    if prior_invocations(fixture_id):raise SystemExit(f"V33 fixture is terminal and cannot be rerun: {fixture_id}")
    event={"event":"PNFM_INVOKE","fixture_id":fixture_id,"run_tag":run_tag,"worker_count":workers,"lock_sha256":sha(LOCK),"precommit_sha256":sha(PRECOMMIT)}
    COUNTER.parent.mkdir(parents=True,exist_ok=True)
    with COUNTER.open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(event,sort_keys=True,separators=(",",":"))+"\n");handle.flush();os.fsync(handle.fileno())


def run(fixture_id:str,run_tag:str,workers:int=1)->dict[str,Any]:
    require_lock()
    if fixture_id not in fixture_ids():raise SystemExit("unregistered V33 fixture")
    descriptor_path=FIX/f"{fixture_id}.json";descriptor=load(descriptor_path)
    if sha(ROOT/descriptor["source_payload_path"])!=descriptor["source_payload_sha256"] or sha(ROOT/descriptor["triangle_binding_path"])!=descriptor["triangle_binding_sha256"]:raise SystemExit("V33 binding hash drift")
    raw=load(ROOT/descriptor["triangle_binding_path"])
    append_invocation(fixture_id,run_tag,workers)
    result=select_unified_v33(raw,descriptor["candidate_records"],config_sha256=sha(LOCK),worker_count=workers)
    canonical=json.loads(canonical_result_bytes(result));cache=result["_ephemeral"]["fixed_cache"]
    output=RESULTS/run_tag;output.mkdir(parents=True,exist_ok=True)
    result_path=output/f"{fixture_id}.json";result_path.write_bytes(canonical_result_bytes(result))
    cache_path=output/f"{fixture_id}.cache.bin";cache_path.write_bytes(cache)
    post_path=POST/f"{fixture_id}_post_4096.png"
    visual_row=render(fixture_id,post_path,result["decisions"][0],result["_ephemeral"]["candidate_visuals"][0])
    return {"fixture_id":fixture_id,"run_tag":run_tag,"result_path":str(result_path.relative_to(ROOT)),"result_sha256":sha(result_path),"cache_path":str(cache_path.relative_to(ROOT)),"cache_sha256":sha(cache_path),"post_visual":visual_row,"result":canonical}


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--fixture",required=True,choices=fixture_ids());parser.add_argument("--run-tag",required=True);parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args();print(json.dumps(run(args.fixture,args.run_tag,args.workers),sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
