#!/usr/bin/env python3
"""Build the immutable V32 source-free fixture/visual preflight without invoking V32."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V31_ROOT = ROOT / "output" / "research" / "f1-v31-dhff-v1" / "source_free_fixtures"
LOCK_PATH = ROOT / "research" / "f1_v32_avhfm_v1_lock.json"
OUTPUT_ROOT = ROOT / "output" / "research" / "f1-v32-avhfm-v1"
FIXTURE_ROOT = OUTPUT_ROOT / "source_free_fixtures"
RAW_VISUAL_ROOT = OUTPUT_ROOT / "preinvoke_visuals"
STRESS_VISUAL_ROOT = OUTPUT_ROOT / "preinvoke_stress_visuals"
COUNTER_PATH = OUTPUT_ROOT / "invocation_events.jsonl"
HARNESS_PATH = OUTPUT_ROOT / "preinvoke_harness_manifest.json"
VISUAL_PATH = OUTPUT_ROOT / "preinvoke_visual_manifest.json"
STRESS_PATH = FIXTURE_ROOT / "STRESS_200K_128.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_f1_v32_avhfm_source_free import render_raw_fixture, render_stress_page  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invocations() -> int:
    if not COUNTER_PATH.exists():
        return 0
    return sum(1 for line in COUNTER_PATH.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event") == "AVHFM_INVOKE")


def _phase_fixture(base: dict[str, Any], fixture_id: str, offset: float) -> dict[str, Any]:
    value = json.loads(json.dumps(base))
    source = value["source"]
    positions = np.asarray(source["positions"], dtype=np.float64)
    keep = np.flatnonzero((positions[:, 1] + offset >= -16.0) & (positions[:, 1] + offset <= 16.0))
    inverse = np.full(len(positions), -1, dtype=np.int64)
    inverse[keep] = np.arange(len(keep), dtype=np.int64)
    positions = positions[keep]
    positions[:, 1] += offset
    source["positions"] = positions.tolist()
    for field in ("source_normals", "parent_fragment_ids", "stable_source_keys"):
        source[field] = np.asarray(source[field], dtype=object)[keep].tolist()
    source["mesh_adjacencies"] = [[int(inverse[a]), int(inverse[b])] for a, b in source["mesh_adjacencies"] if inverse[a] >= 0 and inverse[b] >= 0]
    value["fixture_id"] = fixture_id
    value["kind"] = "v32_avhfm_source_free_fixture"
    value["construction"] = f"exact SF01 continuous equations with analytic axial sampling lattice origin +{offset:.1f}s; fixed [-16s,+16s] crop"
    return value


def _remove_vertices(value: dict[str, Any], keep_mask: np.ndarray, fixture_id: str, construction: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(value))
    source = payload["source"]
    keep = np.flatnonzero(keep_mask)
    inverse = np.full(len(keep_mask), -1, dtype=np.int64)
    inverse[keep] = np.arange(len(keep), dtype=np.int64)
    for field in ("positions", "source_normals", "parent_fragment_ids", "stable_source_keys"):
        source[field] = np.asarray(source[field], dtype=object)[keep].tolist()
    source["mesh_adjacencies"] = [[int(inverse[a]), int(inverse[b])] for a, b in source["mesh_adjacencies"] if inverse[a] >= 0 and inverse[b] >= 0]
    payload["fixture_id"] = fixture_id
    payload["kind"] = "v32_avhfm_source_free_fixture"
    payload["construction"] = construction
    return payload


def _axial_bypass(base: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(base["source"]["positions"], dtype=np.float64)
    stable = np.asarray(base["source"]["stable_source_keys"], dtype=np.int64)
    ci = stable % 10_000
    transition = (ci >= 28) & (ci <= 44)
    keep = ~transition | (np.abs(positions[:, 1] - 8.0) <= 0.75)
    return _remove_vertices(base, keep, "SF26_AXIAL_BYPASS_BRIDGE", "broad tails joined only by a complete fold bridge near z=+8s; current must travel axially at least 6s before crossing")


def _staggered_halves(base: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(base["source"]["positions"], dtype=np.float64)
    stable = np.asarray(base["source"]["stable_source_keys"], dtype=np.int64)
    ci = stable % 10_000
    early_half = (ci >= 28) & (ci <= 36)
    late_half = (ci >= 36) & (ci <= 44)
    transition = (ci >= 28) & (ci <= 44)
    # Keep the early and late transition lobes more than one complete 2.4s
    # local-window diameter apart.  The shared midpoint strip (ci == 36)
    # remains across the full axis so the candidate-wide graph is one global
    # component, while no local window can pool both transition halves.
    allowed = ((positions[:, 1] <= -1.5) & early_half) | ((positions[:, 1] >= 1.5) & late_half) | (ci == 36)
    keep = ~transition | allowed
    return _remove_vertices(base, keep, "SF27_STAGGERED_HALF_FOLDS", "negative axial half contains only early fold transition and positive half only late transition; shared midpoint strip preserves one global component")


def _sf27_analytic_truth(payload: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(payload["source"]["positions"], dtype=np.float64)
    stable = np.asarray(payload["source"]["stable_source_keys"], dtype=np.int64)
    ci = stable % 10_000
    z = positions[:, 1]
    transition = (ci >= 28) & (ci <= 44)
    event_coordinates = sorted(set(float(value) for value in z[transition] if -12.0 <= value <= 12.0))
    windows = []
    for center in event_coordinates:
        present = sorted(set(int(value) for value in ci[transition & (np.abs(z - center) <= 1.2)]))
        windows.append({"center_s": center, "transition_bins_present": present, "complete_transition": all(value in present for value in range(28, 45))})
    shared_midpoint_full_axis = all(np.any((ci == 36) & np.isclose(z, coordinate)) for coordinate in sorted(set(float(value) for value in z if -13.2 <= value <= 13.2)))
    passed = bool(windows) and not any(row["complete_transition"] for row in windows) and shared_midpoint_full_axis
    return {
        "kind": "v32_sf27_pure_analytic_truth",
        "passed": passed,
        "expected_first_local_coverage_token": "V32_POTENTIAL_COVERAGE_MEASURE",
        "window_half_s": 1.2,
        "event_count": len(windows),
        "maximum_transition_bins_in_any_window": max((len(row["transition_bins_present"]) for row in windows), default=0),
        "complete_transition_window_count": sum(row["complete_transition"] for row in windows),
        "shared_midpoint_full_axis": shared_midpoint_full_axis,
    }


def _stress_fixture(base_candidate: dict[str, Any]) -> dict[str, Any]:
    x = np.arange(500, dtype=np.float64) * 0.5
    y = np.arange(400, dtype=np.float64) * 0.5
    xx, yy = np.meshgrid(x, y, indexing="xy")
    positions = np.column_stack((xx.ravel() + 20_000.0, yy.ravel(), np.full(xx.size, 5_000.0)))
    source = {"positions": positions.tolist(), "source_normals": np.tile([0.0, 0.0, 1.0], (len(positions), 1)).tolist(), "parent_fragment_ids": np.full(len(positions), 999, dtype=np.int64).tolist(), "stable_source_keys": (800_000_000 + np.arange(len(positions), dtype=np.int64)).tolist(), "mesh_adjacencies": []}
    candidates=[]
    for index in range(128):
        candidate=json.loads(json.dumps(base_candidate)); shift=np.asarray([index*50.0,0.0,0.0])
        for field in ("center_xyz","render_line_start","render_line_end"):
            candidate[field]=(np.asarray(candidate[field],dtype=float)+shift).tolist()
        for parent_key in ("parent_A","parent_B"):
            parent=candidate[parent_key]; normal=np.asarray(parent["unit_normal_xyz"],dtype=float); parent["centroid_xyz"]=(np.asarray(parent["centroid_xyz"],dtype=float)+shift).tolist(); parent["offset"]=float(parent["offset"]+np.dot(normal,shift))
        candidate["candidate_id"]=9_000_000+index; candidate["frozen_v16_order"]=index; candidates.append(candidate)
    return {"schema_version":1,"kind":"v32_avhfm_source_free_fixture","fixture_id":"STRESS_200K_128","source_free":True,"construction":"exact 200000 sample remote planar source with 128 empty-corridor candidates","source":source,"candidate_records":candidates}


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    source=payload["source"]; positions=np.asarray(source["positions"],dtype=float); normals=np.asarray(source["source_normals"],dtype=float); parents=np.asarray(source["parent_fragment_ids"]); stable=np.asarray(source["stable_source_keys"]); edges=np.asarray(source["mesh_adjacencies"],dtype=np.int64)
    checks={"finite":bool(np.isfinite(positions).all() and np.isfinite(normals).all()),"schema_lengths":bool(len(positions)==len(normals)==len(parents)==len(stable)),"unit_normals":bool(np.allclose(np.linalg.norm(normals,axis=1),1.0,atol=1e-12)),"stable_keys_unique":bool(len(np.unique(stable))==len(stable)),"edges_valid":bool(edges.size==0 or (edges.ndim==2 and edges.shape[1]==2 and edges.min()>=0 and edges.max()<len(positions))),"source_cap":len(positions)<=200_000,"candidate_cap":len(payload["candidate_records"])<=128}
    return {"passed":all(checks.values()),"checks":checks,"samples":len(positions),"edges":len(edges),"candidates":len(payload["candidate_records"])}


def main() -> int:
    if _invocations()!=0:
        raise SystemExit("post-invocation V32 fixture rebuild is forbidden")
    lock=_load(LOCK_PATH)
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); FIXTURE_ROOT.mkdir(parents=True,exist_ok=True); COUNTER_PATH.touch(exist_ok=True)
    fixtures: dict[str,dict[str,Any]]={}
    for fixture_id in lock["v31_fixture_ids"]:
        source_path=V31_ROOT/f"{fixture_id}.json"
        if _sha(source_path)!=lock["v31_fixture_hashes"][fixture_id]:
            raise SystemExit(f"frozen V31 fixture drift: {fixture_id}")
        fixtures[fixture_id]=_load(source_path)
    base=fixtures["SF01_FOLD60_MESH_C2"]
    for fixture_id,offset in (("SF22_FOLD60_MESH_C2_PHASE_01",0.1),("SF23_FOLD60_MESH_C2_PHASE_02",0.2),("SF24_FOLD60_MESH_C2_PHASE_03",0.3),("SF25_FOLD60_MESH_C2_PHASE_04",0.4)):
        fixtures[fixture_id]=_phase_fixture(base,fixture_id,offset)
    fixtures["SF26_AXIAL_BYPASS_BRIDGE"]=_axial_bypass(base)
    fixtures["SF27_STAGGERED_HALF_FOLDS"]=_staggered_halves(base)
    fixtures={fixture_id:fixtures[fixture_id] for fixture_id in lock["physical_fixture_ids"]}
    if tuple(fixtures)!=tuple(lock["physical_fixture_ids"]):
        raise SystemExit("V32 physical fixture order drift")
    sf27_truth = _sf27_analytic_truth(fixtures["SF27_STAGGERED_HALF_FOLDS"])
    if not sf27_truth["passed"] or sf27_truth["expected_first_local_coverage_token"] != lock["expected_results"]["SF27_STAGGERED_HALF_FOLDS"]["reason"]:
        raise SystemExit(f"SF27 analytic truth mismatch: {sf27_truth}")
    fixture_rows=[]; visual_rows=[]
    for fixture_id,payload in fixtures.items():
        validation=_validate(payload)
        if not validation["passed"]: raise SystemExit(f"invalid fixture {fixture_id}: {validation}")
        path=FIXTURE_ROOT/f"{fixture_id}.json"
        if fixture_id in lock["v31_fixture_ids"]:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((V31_ROOT/f"{fixture_id}.json").read_bytes())
        else: _write(path,payload)
        first,second=render_raw_fixture(payload,RAW_VISUAL_ROOT/fixture_id)
        fixture_rows.append({"fixture_id":fixture_id,"path":str(path.relative_to(ROOT)),"sha256":_sha(path),"validation":validation})
        for kind,image in (("raw_multiview",first),("axial_unwrapped_input",second)):
            visual_rows.append({"fixture_id":fixture_id,"kind":kind,"path":str(image.relative_to(ROOT)),"sha256":_sha(image),"width":4096,"height":4096,"signed":False,"decision":None})
    stress=_stress_fixture(base["candidate_records"][0]); validation=_validate(stress)
    if not validation["passed"] or validation["samples"]!=200_000 or validation["candidates"]!=128: raise SystemExit(f"stress invalid: {validation}")
    _write(STRESS_PATH,stress); fixture_rows.append({"fixture_id":"STRESS_200K_128","path":str(STRESS_PATH.relative_to(ROOT)),"sha256":_sha(STRESS_PATH),"validation":validation})
    for page in range(8):
        image=render_stress_page(stress,page,STRESS_VISUAL_ROOT/f"stress_candidates_{page*16:03d}_{page*16+15:03d}.png")
        visual_rows.append({"fixture_id":"STRESS_200K_128","kind":f"stress_page_{page+1}","path":str(image.relative_to(ROOT)),"sha256":_sha(image),"width":4096,"height":4096,"signed":False,"decision":None})
    _write(HARNESS_PATH,{"kind":"v32_avhfm_preinvoke_harness_manifest","architecture_execution_count":0,"passed_analytic_checks":True,"visuals_signed":False,"fixture_rows":fixture_rows,"sf27_expected_token":lock["expected_results"]["SF27_STAGGERED_HALF_FOLDS"]["reason"],"sf27_analytic_truth":sf27_truth})
    _write(VISUAL_PATH,{"kind":"v32_avhfm_preinvoke_visual_manifest","all_signed":False,"rows":visual_rows})
    print(json.dumps({"physical_fixtures":27,"preinvoke_images":len(visual_rows),"architecture_execution_count":0,"status":"AWAITING_STATIC_AUDIT_AND_MANUAL_VISUAL_SIGNOFF"},sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
