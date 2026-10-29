#!/usr/bin/env python3
"""Consume, execute, render, and finalize the one authorized SDRA-V1 attempt."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from research.f1_sdra_v1.config import CONFIG,CONFIG_SHA256,canonical_json_bytes,sha_json  # noqa:E402
from research.f1_sdra_v1.core import normalized_semantic,sha_file  # noqa:E402
from research.f1_sdra_v1.independent_validator import verify_visual_manifest  # noqa:E402
from research.f1_sdra_v1.tamper import assert_allowed_path,assert_tamper_suite  # noqa:E402
from research.f1_sdra_v1.visual import build_visual_package  # noqa:E402

OUTPUT_RELATIVE=Path("output/research/sdra-v1-c006-frozen-source-causal-preflight-v1")
DEFAULT_LOCK=ROOT/"docs/research/SDRA_V1_C006_FROZEN_SOURCE_CAUSAL_PREFLIGHT_V1_LOCK_20260809.json"
DEFAULT_CLAIM=ROOT/"docs/research/SDRA_V1_C006_FROZEN_SOURCE_CAUSAL_PREFLIGHT_V1_CLAIM_20260809.json"
THREAD_ENV={"BLIS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1"}
MANUAL_FIELDS=("reviewed_image_count","reviewed_image_sha256s","all_images_opened_native_resolution","source_complete_and_unclipped","stage_state_labels_consistent","broad_smooth_panel_activation","long_scan_boundary_activation","hole_rim_activation","end_geometry_activation","isolated_speckle_activation","sparse_interior_bendlike_bands","cross_scale_profiles_visually_consistent","veto","veto_reason_tokens","evidence_image_ids","free_text_observations")
BOOLEAN_REVIEW={"YES","NO","UNCERTAIN"}; NUISANCE={"NONE","MINOR","DOMINANT","UNCERTAIN"}; BANDS={"PRESENT","ABSENT","UNCERTAIN"}; PROFILES={"CONSISTENT","INCONSISTENT","UNCERTAIN"}


def _atomic(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor,temporary=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _read(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,Mapping): raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _verify_consume(lock_path:Path,claim_path:Path)->tuple[dict[str,Any],str]:
    lock=_read(lock_path); asserted=str(lock.get("execution_lock_sha256","")); observed=sha_json({key:value for key,value in lock.items() if key!="execution_lock_sha256"})
    if asserted!=observed or lock.get("configuration_sha256")!=CONFIG_SHA256: raise ValueError("INVALID_SDRA_V1_GOVERNANCE_STOP: lock/config hash")
    for relative,expected in lock["code_sha256"].items():
        if sha_file(ROOT/relative)!=expected: raise ValueError(f"INVALID_SDRA_V1_GOVERNANCE_STOP: code changed {relative}")
    for relative,expected in lock["input_sha256"].items():
        if sha_file(ROOT/relative)!=expected: raise ValueError(f"INVALID_SDRA_V1_GOVERNANCE_STOP: input changed {relative}")
    claim=_read(claim_path); ready={"claim_state":"READY_FOR_FIRST_AND_ONLY_EXECUTION","execution_lock_sha256":asserted,"one_shot":True,"rerun_authorized":False}
    if claim!=ready: raise ValueError("INVALID_SDRA_V1_GOVERNANCE_STOP: claim missing, changed, or consumed")
    consumed={**claim,"claim_state":"CONSUMED_BEFORE_FIRST_ROLE_OPENS_RAW_SOURCE"}; _atomic(claim_path,canonical_json_bytes(consumed)+b"\n")
    return lock,asserted


def _role(role:str,source:Path,output:Path,scratch:Path|None,log:Path)->dict[str,Any]:
    command=[sys.executable,"scripts/run_f1_sdra_v1_role.py","--role",role,"--raw-source",str(source),"--output",str(output)]
    if scratch is not None: command.extend(("--visual-scratch",str(scratch)))
    environment=os.environ.copy(); environment.update(THREAD_ENV); ceiling=CONFIG.validator_wall_seconds if role=="VALIDATOR_B" else CONFIG.producer_wall_seconds; start=perf_counter()
    completed=subprocess.run(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=ceiling,check=False)
    process={"role":role,"command":command,"return_code":completed.returncode,"wall_seconds":perf_counter()-start,"stdout_sha256":hashlib.sha256(completed.stdout).hexdigest(),"stderr_sha256":hashlib.sha256(completed.stderr).hexdigest(),"stdout_bytes":len(completed.stdout),"stderr_bytes":len(completed.stderr),"stderr_tail":completed.stderr.decode("utf-8",errors="replace")[-4000:]}
    _atomic(log,canonical_json_bytes(process)+b"\n")
    if completed.returncode: raise RuntimeError(f"role {role} failed: {process['stderr_tail']}")
    value=_read(output)
    if float(value["wall_seconds"])>ceiling or int(value["peak_rss_bytes"])>CONFIG.peak_rss_ceiling_bytes: raise RuntimeError("SDRA_V1_RESOURCE_STOP")
    return value


def _manual_template(manifest:Mapping[str,Any])->dict[str,Any]:
    return {"reviewed_image_count":0,"reviewed_image_sha256s":[],"all_images_opened_native_resolution":None,"source_complete_and_unclipped":None,"stage_state_labels_consistent":None,"broad_smooth_panel_activation":None,"long_scan_boundary_activation":None,"hole_rim_activation":None,"end_geometry_activation":None,"isolated_speckle_activation":None,"sparse_interior_bendlike_bands":None,"cross_scale_profiles_visually_consistent":None,"veto":None,"veto_reason_tokens":[],"evidence_image_ids":[],"free_text_observations":""}


def execute(lock_path:Path,claim_path:Path)->int:
    output=ROOT/OUTPUT_RELATIVE
    if output.exists(): raise ValueError(f"immutable SDRA output already exists: {output}")
    lock,lock_sha=_verify_consume(lock_path,claim_path); output.mkdir(parents=True,exist_ok=False); total_start=perf_counter(); roles=[]; scratch=output/".producer-visual-scratch.npz"
    source=ROOT/lock["source_binding"]["raw_source_relative"]
    assert_allowed_path(source,{source})
    try:
        roles.append(_role("PRODUCER_A",source,output/"producer-a.json",scratch,output/"producer-a-process.json"))
        roles.append(_role("PRODUCER_A_PERMUTED",source,output/"producer-a-permuted.json",None,output/"producer-a-permuted-process.json"))
        roles.append(_role("VALIDATOR_B",source,output/"validator-b.json",None,output/"validator-b-process.json"))
        normalized=[normalized_semantic(row) for row in roles]
        consensus=all(row==normalized[0] for row in normalized[1:])
        if not consensus: raise RuntimeError("INVALID_SDRA_V1_REPRODUCIBILITY_STOP")
        tamper=assert_tamper_suite(); render_start=perf_counter(); manifest=build_visual_package(output,raw_source=source,scratch_path=scratch,semantic=normalized[0],roles=roles); renderer_wall=perf_counter()-render_start
        if renderer_wall>CONFIG.renderer_wall_seconds: raise RuntimeError("SDRA_V1_RESOURCE_STOP: renderer")
        _atomic(output/"visual-manifest.json",canonical_json_bytes(manifest)+b"\n")
        visual_verification=verify_visual_manifest(output)
        _atomic(output/"manual-visual-review-template.json",canonical_json_bytes(_manual_template(manifest))+b"\n")
        scratch.unlink()
        result={"schema_version":1,"attempt_id":CONFIG.attempt_id,"execution_lock_sha256":lock_sha,"roles":[{key:row[key] for key in ("role","semantic_sha256","decision","wall_seconds","peak_rss_bytes")} for row in roles],"role_consensus":consensus,"tamper_results":tamper,"visual_verification":visual_verification,"visual_manifest_sha256":manifest["visual_manifest_sha256"],"visual_image_count":manifest["image_count"],"renderer_wall_seconds":renderer_wall,"machine_decision":roles[0]["decision"],"final_decision":"PENDING_MANDATORY_MANUAL_VISUAL_REVIEW","claim_consumed":True,"rerun_authorized":False,"target_gold_cad_benchmark_gcdf_opened":False,"runtime_main_production_opened":False,"total_wall_seconds":perf_counter()-total_start}
        result["machine_result_sha256"]=sha_json(result); _atomic(output/"machine-result-pending-manual-review.json",canonical_json_bytes(result)+b"\n"); print(canonical_json_bytes(result).decode("ascii")); return 0
    except BaseException as error:
        failure={"schema_version":1,"attempt_id":CONFIG.attempt_id,"execution_lock_sha256":lock_sha,"decision":"INVALID_SDRA_V1_BINDING_OR_EXECUTION_STOP","error_type":type(error).__name__,"error":str(error),"roles_completed":[row.get("role") for row in roles],"claim_consumed":True,"rerun_authorized":False,"total_wall_seconds":perf_counter()-total_start}; failure["result_sha256"]=sha_json(failure); _atomic(output/"result.json",canonical_json_bytes(failure)+b"\n")
        if scratch.exists(): scratch.unlink()
        raise


def _validate_manual(manual:Mapping[str,Any],manifest:Mapping[str,Any],output:Path)->bool:
    if set(manual)!=set(MANUAL_FIELDS): raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: exact fields")
    expected=[row["sha256"] for row in manifest["images"]]
    if manual["reviewed_image_count"]!=77 or manual["reviewed_image_sha256s"]!=expected: raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: image census")
    for row in manifest["images"]:
        if sha_file(output/row["path"])!=row["sha256"]: raise ValueError("INVALID_SDRA_V1_VISUAL_PACKAGE_STOP: post-review hash")
    for key in ("all_images_opened_native_resolution","source_complete_and_unclipped","stage_state_labels_consistent"):
        if manual[key] not in BOOLEAN_REVIEW: raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: boolean enum")
    for key in ("broad_smooth_panel_activation","long_scan_boundary_activation","hole_rim_activation","end_geometry_activation","isolated_speckle_activation"):
        if manual[key] not in NUISANCE: raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: nuisance enum")
    if manual["sparse_interior_bendlike_bands"] not in BANDS or manual["cross_scale_profiles_visually_consistent"] not in PROFILES or not isinstance(manual["veto"],bool): raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: terminal enum")
    if not isinstance(manual["veto_reason_tokens"],list) or not isinstance(manual["evidence_image_ids"],list) or not isinstance(manual["free_text_observations"],str): raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: evidence fields")
    mandatory=(manual["all_images_opened_native_resolution"]!="YES" or manual["source_complete_and_unclipped"]!="YES" or manual["stage_state_labels_consistent"]!="YES" or any(manual[key] in {"DOMINANT","UNCERTAIN"} for key in ("broad_smooth_panel_activation","long_scan_boundary_activation","hole_rim_activation","end_geometry_activation","isolated_speckle_activation")) or manual["sparse_interior_bendlike_bands"]!="PRESENT" or manual["cross_scale_profiles_visually_consistent"]!="CONSISTENT")
    if mandatory and not manual["veto"]: raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: mandatory veto omitted")
    if any(manual[key]!="NONE" for key in ("broad_smooth_panel_activation","long_scan_boundary_activation","hole_rim_activation","end_geometry_activation","isolated_speckle_activation")) and (not manual["evidence_image_ids"] or not manual["free_text_observations"].strip()): raise ValueError("INVALID_SDRA_V1_MANUAL_AUDIT_STOP: nuisance evidence missing")
    return bool(manual["veto"])


def finalize(manual_path:Path)->int:
    output=ROOT/OUTPUT_RELATIVE; machine=_read(output/"machine-result-pending-manual-review.json"); manifest=_read(output/"visual-manifest.json"); manual=_read(manual_path); veto=_validate_manual(manual,manifest,output); machine_decision=str(machine["machine_decision"])
    final="SDRA_V1_VISUAL_NUISANCE_ARCHITECTURE_STOP" if machine_decision=="SDRA_V1_SOURCE_REPRESENTATION_PREFLIGHT_SURVIVES" and veto else machine_decision
    result={**machine,"manual_review":manual,"manual_review_sha256":sha_file(manual_path),"final_decision":final,"source_candidate_bundle_frozen":final=="SDRA_V1_SOURCE_REPRESENTATION_PREFLIGHT_SURVIVES"}; result.pop("machine_result_sha256",None); result["result_sha256"]=sha_json(result); _atomic(output/"result.json",canonical_json_bytes(result)+b"\n"); print(canonical_json_bytes(result).decode("ascii")); return 0


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True); run=sub.add_parser("execute"); run.add_argument("--lock",type=Path,default=DEFAULT_LOCK); run.add_argument("--claim",type=Path,default=DEFAULT_CLAIM); finish=sub.add_parser("finalize"); finish.add_argument("--manual-review",required=True,type=Path); args=parser.parse_args(argv)
    return execute(args.lock,args.claim) if args.command=="execute" else finalize(args.manual_review)


if __name__=="__main__": raise SystemExit(main())
