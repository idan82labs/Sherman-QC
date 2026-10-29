#!/usr/bin/env python3
"""Complete mandatory source-free geometry and tamper package for SDRA-V1."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from research.f1_sdra_v1.config import CONFIG,CONFIG_SHA256,canonical_json_bytes,sha_json  # noqa:E402
from research.f1_sdra_v1.core import (  # noqa:E402
    SDRAStop, canonicalize_sampled_xyz, evaluate_fixture_geometry, evaluate_point_features,
    exact_knn_kdtree,
)
from research.f1_sdra_v1 import independent_validator as validator  # noqa:E402
from research.f1_sdra_v1.tamper import assert_allowed_path,assert_tamper_suite  # noqa:E402


def _plane(n:int=35,spacing:float=1.0)->np.ndarray:
    u=np.arange(n,dtype=float)-(n-1)/2; x,y=np.meshgrid(u*spacing,u*spacing,indexing="ij")
    return np.stack((x.ravel(),y.ravel(),np.zeros(x.size)),axis=1)


def _hole()->np.ndarray:
    xyz=_plane(43); radius=np.linalg.norm(xyz[:,:2],axis=1)
    return xyz[radius>=5.5]


def _density_step()->np.ndarray:
    left=np.stack(np.meshgrid(np.arange(-22,0,.75),np.arange(-18,19,1.0),[0.0],indexing="ij"),axis=-1).reshape(-1,3)
    right=np.stack(np.meshgrid(np.arange(0,23,1.5),np.arange(-18,19,1.0),[0.0],indexing="ij"),axis=-1).reshape(-1,3)
    return np.vstack((left,right))


def _noise_plane()->np.ndarray:
    xyz=_plane(37); xyz[:,2]=0.025*np.sin(1.731*xyz[:,0]+0.913*xyz[:,1])
    return xyz


def _double_layer()->np.ndarray:
    base=_plane(29,1.0); top=base.copy(); top[:,2]=0.06
    return np.vstack((base,top))


def _cylinder()->np.ndarray:
    theta=np.linspace(-.7,.7,41); axis=np.linspace(-20,20,41); t,z=np.meshgrid(theta,axis,indexing="ij"); r=35.0
    return np.stack((r*np.sin(t).ravel(),z.ravel(),r*(1-np.cos(t)).ravel()),axis=1)


def _fold(angle_degrees:float,*,fillet:bool=False)->np.ndarray:
    angle=math.radians(angle_degrees); along=np.arange(-22,23,dtype=float); across=np.arange(0,23,dtype=float)
    points=[]
    for x in along:
        points.append((x,0.0,0.0))
        for s in across[1:]:
            points.append((x,s,0.0))
            points.append((x,-s*math.cos(angle),s*math.sin(angle)))
        if fillet:
            for step in range(1,6):
                theta=angle*step/6.0; radius=3.0
                points.append((x,-radius*math.sin(theta),radius*(1-math.cos(theta))))
    return np.unique(np.asarray(points,dtype=float),axis=0)


def _volume()->np.ndarray:
    grid=np.stack(np.meshgrid(np.arange(-8,9),np.arange(-8,9),np.arange(-8,9),indexing="ij"),axis=-1).reshape(-1,3).astype(float)
    return grid[np.linalg.norm(grid,axis=1)<=8.1]


def _metric_signature(result:dict[str,object])->str:
    return sha_json({key:result[key] for key in ("point_count","scale_eligible_count","candidate_count","supported_candidate_count","median_d_q32","p90_d_q32")})


def _run_geometry(name:str,factory:Callable[[],np.ndarray])->dict[str,object]:
    xyz=factory(); result=evaluate_fixture_geometry(xyz); result.pop("candidate_integer_xyz",None)
    result["fixture_name"]=name; return result


def run_regressions()->dict[str,object]:
    geometry:dict[str,dict[str,object]]={}
    geometry["complete_planar_lattice"]=_run_geometry("complete_planar_lattice",_plane)
    geometry["planar_straight_boundary"]=_run_geometry("planar_straight_boundary",_plane)
    geometry["planar_circular_hole"]=_run_geometry("planar_circular_hole",_hole)
    geometry["planar_density_step"]=_run_geometry("planar_density_step",_density_step)
    geometry["deterministic_normal_noise_plane"]=_run_geometry("deterministic_normal_noise_plane",_noise_plane)
    geometry["close_parallel_double_layer"]=_run_geometry("close_parallel_double_layer",_double_layer)
    geometry["smooth_cylindrical_patch"]=_run_geometry("smooth_cylindrical_patch",_cylinder)
    geometry["thirty_degree_sharp_fold"]=_run_geometry("thirty_degree_sharp_fold",lambda:_fold(30))
    geometry["ninety_degree_radius_three_fillet"]=_run_geometry("ninety_degree_radius_three_fillet",lambda:_fold(90,fillet=True))
    geometry["ten_degree_fold"]=_run_geometry("ten_degree_fold",lambda:_fold(10))
    geometry["volumetric_isotropic_cloud"]=_run_geometry("volumetric_isotropic_cloud",_volume)
    end=_run_geometry("end_cap_confound",lambda:_fold(90)); end["known_geometric_confuser_present"]=end["candidate_count"]>0
    geometry["end_cap_confound"]=end
    checks:dict[str,bool]={}
    for name in ("complete_planar_lattice","planar_straight_boundary","planar_circular_hole","planar_density_step"):
        checks[name]=geometry[name]["candidate_count"]==0
    checks["deterministic_normal_noise_plane"]=geometry["deterministic_normal_noise_plane"]["candidate_count"]==0
    checks["close_parallel_double_layer"]=geometry["close_parallel_double_layer"]["candidate_count"]==0
    checks["smooth_cylindrical_patch"]=geometry["smooth_cylindrical_patch"]["candidate_count"]==0
    row=geometry["thirty_degree_sharp_fold"]
    checks["thirty_degree_sharp_fold"]=0<row["candidate_count"]<.15*row["point_count"] and row["supported_candidate_count"]>0 and row["median_d_q32"]<=CONFIG.support_median_d_maximum_q32 and row["p90_d_q32"]<=CONFIG.support_p90_d_maximum_q32
    row=geometry["ninety_degree_radius_three_fillet"]
    checks["ninety_degree_radius_three_fillet"]=row["candidate_count"]>0 and row["supported_candidate_count"]>0 and row["median_d_q32"]<=CONFIG.support_median_d_maximum_q32 and row["p90_d_q32"]<=CONFIG.support_p90_d_maximum_q32
    checks["ten_degree_fold"]=geometry["ten_degree_fold"]["candidate_count"]==0
    volume=geometry["volumetric_isotropic_cloud"]
    checks["volumetric_isotropic_cloud"]=volume["candidate_count"]==0 or volume["median_d_q32"]>CONFIG.support_median_d_maximum_q32 or volume["p90_d_q32"]>CONFIG.support_p90_d_maximum_q32
    checks["end_cap_confound_recorded"]=bool(end["known_geometric_confuser_present"])
    base=_fold(30); base_result=evaluate_fixture_geometry(base); order=np.arange(len(base))[::-1]
    checks["input_order_permutation"]=_metric_signature(base_result)==_metric_signature(evaluate_fixture_geometry(base[order]))
    transforms=(
        base[:,[1,2,0]]*np.asarray([-1,1,1])+np.asarray([11,-7,3]),
        base*3.25+np.asarray([100,-80,25]),
    )
    checks["signed_axis_translation_uniform_scale"] = all(_metric_signature(base_result)==_metric_signature(evaluate_fixture_geometry(value)) for value in transforms)
    source=canonicalize_sampled_xyz(_plane())
    ids,sq=exact_knn_kdtree(source,anchors=np.asarray([0],np.uint32),k=384)
    pairs=list(zip(sq[0].tolist(),ids[0].tolist())); checks["equal_distance_neighbor_ring"]=pairs==sorted(pairs)
    vsource=validator._Source(source.integer_xyz,source.metric_xyz,source.lattice_step,source.origin,"FIX",0,source.sampled_coordinate_sha256,source.canonical_table_sha256,source.morton_order)
    vids,vsq=validator.exact_knn_morton(vsource,anchors=np.asarray([0,17,400,1224],np.uint32))
    pids,psq=exact_knn_kdtree(source,anchors=np.asarray([0,17,400,1224],np.uint32))
    checks["producer_validator_exact_knn"] = np.array_equal(vids,pids) and np.array_equal(vsq,psq)
    fold_source=canonicalize_sampled_xyz(base)
    fold_ids=np.arange(len(fold_source.integer_xyz),dtype=np.uint32)
    producer_features,_producer_neighbors,_producer_squared=evaluate_point_features(fold_source)
    validator_source=validator._Source(fold_source.integer_xyz,fold_source.metric_xyz,fold_source.lattice_step,fold_source.origin,"FIX",0,fold_source.sampled_coordinate_sha256,fold_source.canonical_table_sha256,fold_source.morton_order)
    validator_features,_validator_neighbors=validator._point_features(validator_source,fold_ids)
    feature_pairs=(("q_q32","q"),("x_q32","x"),("mu1_min_q32","mu1"),("a_q32","a"),("c_q32","c"),("b_q32","b"),("gain_q32","gain"),("error_q32","error"),("phi_q32","phi"),("scale_eligible","eligible"),("candidate","candidate"))
    checks["producer_validator_quantized_semantics"] = all(np.array_equal(getattr(producer_features,left),validator_features[right]) for left,right in feature_pairs)
    nan_fixture=_plane(); nan_fixture[0,0]=np.nan
    inf_fixture=_plane(); inf_fixture[0,0]=np.inf
    bad_inputs={
        "duplicate_quantized_point":np.vstack((_plane(),_plane()[0])),
        "nan":nan_fixture,
        "infinity":inf_fixture,
        "fewer_than_385":_plane(19),
    }
    for name,value in bad_inputs.items():
        try: canonicalize_sampled_xyz(value)
        except SDRAStop: checks[f"source_contract_{name}"]=True
        else: checks[f"source_contract_{name}"]=False
    checks.update(assert_tamper_suite())
    allowed=ROOT/"tmp"/"sdra-source-free-allowed.bin"; forbidden=ROOT/"data"/"gold.bin"
    assert_allowed_path(allowed,{allowed})
    try: assert_allowed_path(forbidden,{allowed})
    except PermissionError: checks["governance_forbidden_path_stop"]=True
    else: checks["governance_forbidden_path_stop"]=False
    passed=all(checks.values())
    result={"schema_version":1,"regression_id":"SDRA_V1_SOURCE_FREE_REGRESSIONS_V1","configuration_sha256":CONFIG_SHA256,"source_opened":False,"target_gold_cad_benchmark_gcdf_opened":False,"geometry":geometry,"checks":checks,"passed":passed}
    result["result_sha256"]=sha_json(result); return result


def main()->int:
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path); args=parser.parse_args()
    result=run_regressions(); encoded=canonical_json_bytes(result)+b"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix=f".{args.output.name}.",suffix=".tmp",dir=args.output.parent)
        try:
            with os.fdopen(descriptor,"wb") as stream: stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,args.output)
        except BaseException:
            try: os.unlink(temporary)
            except FileNotFoundError: pass
            raise
    print(encoded.decode("ascii").rstrip()); return 0 if result["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
