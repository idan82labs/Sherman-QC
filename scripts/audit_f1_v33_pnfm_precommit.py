#!/usr/bin/env python3
"""Static, non-invoking freeze audit for the one-shot V33 PNFM study."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"research"/"f1_v33_pnfm_v1_lock.json"
ENGINE=ROOT/"backend"/"f1_projective_normal_fan_moment.py"
BUILDER=ROOT/"scripts"/"build_f1_v33_pnfm_source_free_fixtures.py"
RENDERER=ROOT/"scripts"/"render_f1_v33_pnfm_source_free.py"
RUNNER=ROOT/"scripts"/"run_f1_v33_pnfm_source_free.py"
GATES=ROOT/"scripts"/"run_f1_v33_pnfm_gates.py"
OUT=ROOT/"output"/"research"/"f1-v33-pnfm-v1"
FIXTURE_MANIFEST=OUT/"fixture_manifest.json"
VISUAL_MANIFEST=OUT/"preinvoke_visual_manifest.json"
SIGNOFF=OUT/"preinvoke_manual_visual_signoff.json"
PRECOMMIT=OUT/"precommit_manifest.json"
COUNTER=OUT/"invocation_events.jsonl"

if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import backend.f1_projective_normal_fan_moment as engine  # noqa:E402


def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path:Path,value:Any)->None:path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")


def invocation_count()->int:
    if not COUNTER.exists():return 0
    return sum(json.loads(line).get("event")=="PNFM_INVOKE" for line in COUNTER.read_text(encoding="utf-8").splitlines() if line.strip())


def dummy(index:int)->dict[str,Any]:
    digest=hashlib.sha256(f"v33-dummy-{index}".encode()).hexdigest()
    return {"candidate_id":index,"frozen_v16_order":index,"candidate_sha256":digest,"passthrough_line_sha256":digest,"action":engine.ACTION_NONE,"reason":"PNFM_AREA_DENSITY","metrics_i32":[0]*24,"nms_reason":""}


def main()->int:
    lock=load(LOCK);fixtures=load(FIXTURE_MANIFEST);visuals=load(VISUAL_MANIFEST);signoff=load(SIGNOFF)
    source=ENGINE.read_text(encoding="utf-8");runner=RUNNER.read_text(encoding="utf-8");renderer=RENDERER.read_text(encoding="utf-8")
    tree=ast.parse(source);calls=[node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    imports={alias.name for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names}
    checks=[]
    def check(name:str,passed:bool,evidence:Any)->None:checks.append({"name":name,"passed":bool(passed),"evidence":evidence})
    check("profile_exact",engine.PROFILE==lock["profile"],engine.PROFILE)
    check("primitive_exact",engine.PRIMITIVE==lock["primitive"],engine.PRIMITIVE)
    constants={"MAX_VERTICES":"vertices_cap","MAX_TRIANGLES":"unique_triangles_cap","MAX_CANDIDATES":"candidates_cap","MAX_SUPPORT_TRIANGLES":"local_support_triangles_cap","MAX_CLIPPED_PIECES":"clipped_pieces_cap","MAX_INTERVAL_ENDPOINTS":"interval_endpoints_cap","CACHE_HEADER_BYTES":"cache_header_bytes","CACHE_RECORD_BYTES":"cache_record_bytes"}
    for symbol,key in constants.items():check(f"constant_{symbol.lower()}",getattr(engine,symbol)==lock["constants"][key],{"source":getattr(engine,symbol),"lock":lock["constants"][key]})
    check("original_positive_byte_exact",sha(ROOT/lock["original_v32_positive_path"])==lock["original_v32_positive_sha256"],sha(ROOT/lock["original_v32_positive_path"]))
    check("engine_no_file_io","pathlib" not in imports and not any(name in calls for name in ("open","read_text","write_text","read_bytes","write_bytes")),sorted(imports))
    canonical_source=inspect.getsource(engine.canonical_mesh)
    analysis_source=inspect.getsource(engine._candidate_analysis)
    check("engine_reads_only_positions_triangles_candidates",'allowed = {"positions", "triangles"}' in canonical_source and all(token not in analysis_source for token in ("mesh_adjac","stable_source_keys","expected_count","filename","part_identity")),"input-access source scan")
    check("ordinary_nms_exactly_one_callsite",calls.count("ordinary_nms")==1,calls.count("ordinary_nms"))
    check("exact_segment_triangle_support",all(name in source for name in ("_point_triangle_distance","_segment_segment_distance","_segment_triangle_distance","broad_start","broad_end")),"source scan")
    check("fixed_jacobi_16_sweeps","for _ in range(16)" in inspect.getsource(engine._jacobi_principal),sha(ENGINE))
    check("renderer_native_surface_and_all_candidates",all(token in renderer for token in ("Poly3DCollection","Line3DCollection","candidate_records")[0:]) and "[1:]" in renderer,sha(RENDERER))
    check("runner_records_before_mechanism",runner.index("append_invocation(fixture_id")<runner.index("select_unified_v33("),"lexical order")
    check("no_prior_mechanism_invocation",invocation_count()==0,invocation_count())
    rows=fixtures.get("rows",[]);ids=[row["fixture_id"] for row in rows]
    check("fixture_manifest_complete",ids==lock["ordered_fixture_ids"] and len(rows)==25,ids)
    fixture_hashes=[];fixtures_ok=True
    for row in rows:
        for key in ("descriptor_path",):
            path=ROOT/row[key];fixtures_ok &= path.exists() and sha(path)==row["descriptor_sha256"]
            fixture_hashes.append({"path":row[key],"sha256":row["descriptor_sha256"]})
        descriptor=load(ROOT/row["descriptor_path"])
        for path_key,hash_key in (("source_payload_path","source_payload_sha256"),("triangle_binding_path","triangle_binding_sha256")):
            path=ROOT/descriptor[path_key];fixtures_ok &= path.exists() and sha(path)==descriptor[hash_key]
            fixture_hashes.append({"path":descriptor[path_key],"sha256":descriptor[hash_key]})
    check("all_fixture_bindings_current",fixtures_ok,len(fixture_hashes))
    vrows=visuals.get("rows",[]);visuals_ok=len(vrows)==25 and [row["fixture_id"] for row in vrows]==ids and all((ROOT/row["path"]).exists() and sha(ROOT/row["path"])==row["sha256"] and row["width"]==4096 and row["height"]==4096 for row in vrows)
    check("all_native_visuals_current",visuals_ok,len(vrows))
    signed_by={row["path"]:row for row in signoff.get("rows",[])}
    signed=signoff.get("all_signed") is True and signoff.get("decision")=="ACCEPT_ALL_PREINVOKE_SOURCE_GEOMETRY" and signoff.get("visual_manifest_sha256")==sha(VISUAL_MANIFEST) and len(signed_by)==25 and all(signed_by.get(row["path"],{}).get("signed") is True and signed_by[row["path"]]["sha256"]==row["sha256"] for row in vrows)
    check("all_native_visuals_manually_signed",signed,{"count":len(signed_by),"manifest":sha(VISUAL_MANIFEST)})
    cache=engine.build_fixed_cache([dummy(i) for i in range(128)],"00"*32,"11"*32)
    cache_ok=len(cache)==lock["constants"]["stress_cache_bytes"] and cache[:8]==b"PNFMV1\0\0" and all(cache[64+i*192+164:64+(i+1)*192]==b"\0"*28 for i in range(128))
    check("fixed_cache_layout_and_zero_reserve",cache_ok,len(cache))
    source_paths=(ENGINE,LOCK,BUILDER,RENDERER,RUNNER,GATES,Path(__file__))
    source_hashes=[{"path":str(path.relative_to(ROOT)),"sha256":sha(path)} for path in source_paths]
    visual_hashes=[{"path":row["path"],"sha256":row["sha256"]} for row in vrows]+[{"path":str(VISUAL_MANIFEST.relative_to(ROOT)),"sha256":sha(VISUAL_MANIFEST)},{"path":str(SIGNOFF.relative_to(ROOT)),"sha256":sha(SIGNOFF)},{"path":str(FIXTURE_MANIFEST.relative_to(ROOT)),"sha256":sha(FIXTURE_MANIFEST)}]
    passed=all(row["passed"] for row in checks)
    manifest={"schema_version":1,"kind":"v33_pnfm_static_precommit_manifest","passed":passed,"mechanism_invocation_count":invocation_count(),"checks":checks,"source_hashes":source_hashes,"fixture_hashes":fixture_hashes,"preinvoke_visual_hashes":visual_hashes,"token":"V33_PNFM_MECHANISM_LOCKED" if passed else "V33_STATIC_PREFLIGHT_STOP"}
    write(PRECOMMIT,manifest)
    print(json.dumps({"passed":passed,"checks":len(checks),"failed":[row["name"] for row in checks if not row["passed"]],"token":manifest["token"]},sort_keys=True))
    return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
