#!/usr/bin/env python3
"""Strict ordered numeric+native-visual V33 one-shot gate driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research"/"f1_v33_pnfm_v1_lock.json";OUT=ROOT/"output"/"research"/"f1-v33-pnfm-v1"
STATE=OUT/"source_free_gate_result.json";POST_MANIFEST=OUT/"postinvoke_visual_manifest.json"
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_f1_v33_pnfm_source_free import run  # noqa:E402


def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def write(path:Path,value:Any)->None:path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")


def initial(lock:dict[str,Any])->dict[str,Any]:
    return {"profile":lock["profile"],"terminal":False,"passed":False,"token":"V33_NOT_COMPLETE","awaiting_manual_visual_fixture":None,"rows":[{"fixture_id":fid,"status":"NOT_RUN","visual_signed":False} for fid in lock["ordered_fixture_ids"]]}


def state(lock:dict[str,Any])->dict[str,Any]:return load(STATE) if STATE.exists() else initial(lock)


def stage_token(fid:str,index:int)->str:
    if index==0:return "V33_CLEAN_FOLD_STOP"
    if index<4:return "V33_REMESH_INVARIANCE_STOP"
    if index==4:return "V33_JOINT_MEASURE_ADVERSARY_STOP"
    if index<7:return "V33_FRAGMENT_STOP"
    if index<21:return "V33_NUISANCE_STOP"
    return "V33_DUPLICATE_ORDER_STOP"


def terminal(value:dict[str,Any],token:str,stage:str,detail:Any)->int:
    value.update({"terminal":True,"passed":False,"token":token,"terminal_stage":stage,"terminal_detail":detail,"awaiting_manual_visual_fixture":None})
    for row in value["rows"]:
        if row["status"]=="NOT_RUN":row["status"]="NOT_RUN_AFTER_TERMINAL_FAILURE"
    write(STATE,value);print(json.dumps({"terminal":True,"token":token,"stage":stage,"detail":detail},sort_keys=True));return 2


def next_fixture()->int:
    lock=load(LOCK);value=state(lock)
    if value.get("terminal"):raise SystemExit("V33 is terminal and may not be rerun")
    if value.get("awaiting_manual_visual_fixture"):raise SystemExit(f"native visual review required for {value['awaiting_manual_visual_fixture']}")
    pending=next((row for row in value["rows"] if row["status"]=="NOT_RUN"),None)
    if pending is None:
        value.update({"passed":True,"terminal":False,"token":lock["success_token"]});write(STATE,value);print(json.dumps({"passed":True,"token":value["token"]},sort_keys=True));return 0
    fid=pending["fixture_id"];index=value["rows"].index(pending);descriptor=load(OUT/"source_free_fixtures"/f"{fid}.json")
    row=run(fid,f"ordered_{index+1:02d}",1);result=row["result"]
    observed=[int(result["pre_nms_rescue_count"]),len(result["final_lines"])];expected=[int(v) for v in descriptor["expected_pre_final"]]
    pending.update({"status":"NUMERIC_PASS_AWAITING_VISUAL" if observed==expected else "NUMERIC_MISMATCH","expected":expected,"observed":observed,"reasons":[decision["reason"] for decision in result["decisions"]],"result_path":row["result_path"],"result_sha256":row["result_sha256"],"cache_sha256":row["cache_sha256"],"post_visual":row["post_visual"]})
    manifest=load(POST_MANIFEST) if POST_MANIFEST.exists() else {"schema_version":1,"kind":"v33_pnfm_postinvoke_visual_manifest","all_signed":False,"rows":[]}
    manifest["rows"].append(dict(row["post_visual"],signed=False,decision=None));write(POST_MANIFEST,manifest)
    if observed!=expected:return terminal(value,stage_token(fid,index),f"numeric:{fid}",{"expected":expected,"observed":observed,"reasons":pending["reasons"]})
    value["awaiting_manual_visual_fixture"]=fid;write(STATE,value);print(json.dumps({"fixture_id":fid,"numeric_pass":True,"observed":observed,"awaiting_manual_native_visual_review":True,"post_visual":row["post_visual"]},sort_keys=True));return 0


def sign_current(decision:str)->int:
    lock=load(LOCK);value=state(lock);fid=value.get("awaiting_manual_visual_fixture")
    if not fid:raise SystemExit("no V33 result awaits visual review")
    index=next(i for i,row in enumerate(value["rows"]) if row["fixture_id"]==fid)
    if decision!="accept":return terminal(value,"V33_VISUAL_CONTRADICTION_STOP",f"visual:{fid}","native artifact contradicts fixture or numeric decision")
    row=value["rows"][index];row["status"]="PASS";row["visual_signed"]=True;value["awaiting_manual_visual_fixture"]=None
    manifest=load(POST_MANIFEST)
    for visual in manifest["rows"]:
        if visual["fixture_id"]==fid:visual["signed"]=True;visual["decision"]="ACCEPT_SOURCE_GEOMETRY_AND_NUMERIC_OUTCOME"
    manifest["all_signed"]=len(manifest["rows"])==len(lock["ordered_fixture_ids"]) and all(row["signed"] for row in manifest["rows"])
    write(POST_MANIFEST,manifest);write(STATE,value);print(json.dumps({"fixture_id":fid,"visual_signed":True,"next":"next-fixture"},sort_keys=True));return 0


def main()->int:
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--next-fixture",action="store_true");group.add_argument("--sign-current",choices=("accept","reject"));args=parser.parse_args()
    return next_fixture() if args.next_fixture else sign_current(args.sign_current)


if __name__=="__main__":raise SystemExit(main())
