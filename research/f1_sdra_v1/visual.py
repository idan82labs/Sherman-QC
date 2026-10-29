"""Authoritative deterministic 77-image SDRA-V1 visual package."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from research.f1_ciip_v1 import partition, primary_parser
from .config import CAMERAS, CONFIG, GEOMETRY_LAYERS, K_VALUES, sha_json


VIEW_AXES = {
    "+X": (1, 2, 1, 1), "-X": (1, 2, -1, 1),
    "+Y": (0, 2, 1, 1), "-Y": (0, 2, -1, 1),
    "+Z": (0, 1, 1, 1), "-Z": (0, 1, 1, -1),
}


def _color_table() -> np.ndarray:
    index = np.arange(256, dtype=np.float64) / 255.0
    red = np.clip(1.8 * index - 0.35, 0, 1)
    green = np.clip(1.65 - 2.2 * np.abs(index - 0.55), 0, 1)
    blue = np.clip(1.35 - 1.9 * index, 0, 1)
    return np.rint(255 * np.stack((red, green, blue), axis=1)).astype(np.uint8)


COLOR_TABLE = _color_table()
COLOR_TABLE_SHA256 = hashlib.sha256(COLOR_TABLE.tobytes()).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_points(raw_source: Path) -> tuple[np.ndarray, np.ndarray, float]:
    if _sha_file(raw_source) != CONFIG.raw_source_sha256 or raw_source.stat().st_size != CONFIG.raw_source_bytes:
        raise ValueError("SDRA renderer source binding mismatch")
    descriptor = os.open(raw_source, os.O_RDONLY)
    try:
        xyz = np.asarray(primary_parser.parse_ply_fd(descriptor).xyz, dtype="<f8")
    finally:
        os.close(descriptor)
    packed = np.ascontiguousarray(xyz).view("V24").reshape(len(xyz))
    _unique, first, _inverse, counts = np.unique(packed, return_index=True, return_inverse=True, return_counts=True)
    sampled = partition.canonical_sample(
        np.ascontiguousarray(xyz[first], dtype="<f8"), np.asarray(counts, dtype=np.uint32), sample_cap=200_000
    )
    sampled_xyz = np.asarray(sampled.xyz, dtype=np.float64)
    origin = np.min(sampled_xyz, axis=0)
    largest = float(np.max(np.max(sampled_xyz, axis=0) - origin))
    lattice = largest / float(1 << 30)
    integer = np.rint((sampled_xyz - origin) / lattice).astype(np.int64)
    order = np.lexsort((integer[:, 2], integer[:, 1], integer[:, 0]))
    integer = np.ascontiguousarray(integer[order], dtype=np.int64)
    if np.any(np.all(integer[1:] == integer[:-1], axis=1)):
        raise ValueError("SDRA renderer duplicate quantized point")
    return origin + lattice * integer.astype(np.float64), integer, lattice


def _state_for_layer(layer: str, ledger: Mapping[str, Mapping[str, str]], full: bool) -> tuple[str, Mapping[str, str]]:
    if layer == "complete-source": stage = "source"
    elif layer == "supported-candidates-d": stage = "support"
    elif full and layer == "candidate-mask": stage = "candidate_mask"
    elif full: stage = "full_features"
    else: stage = "sentinel"
    if stage not in ledger:
        return stage, {"execution_state":"NOT_EVALUABLE","entity_state":"NOT_APPLICABLE","upstream_stop":"MISSING_STAGE"}
    return stage, ledger[stage]


def _metadata(stage_id: str, state: Mapping[str, str], semantic_hash: str, ledger_hash: str) -> dict[str, str]:
    return {
        "STAGE_ID": stage_id,
        "EXECUTION_STATE": str(state["execution_state"]),
        "ENTITY_STATE": str(state["entity_state"]),
        "UPSTREAM_STOP": str(state["upstream_stop"]),
        "SEMANTIC_RESULT_HASH": semantic_hash,
        "STAGE_LEDGER_HASH": ledger_hash,
    }


def _save_png(image: Image.Image, path: Path, metadata: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    for key, value in metadata.items(): info.add_text(key, value)
    image.save(path, format="PNG", compress_level=9, optimize=False, pnginfo=info)


def _header(image: Image.Image, title: str, metadata: Mapping[str, str]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,image.width,92), fill=(250,250,250))
    font = ImageFont.load_default()
    draw.text((16,12), title, fill=(5,5,5), font=font)
    draw.text((16,34), f"{metadata['STAGE_ID']} | {metadata['EXECUTION_STATE']} | {metadata['ENTITY_STATE']}", fill=(25,25,25), font=font)
    draw.text((16,54), f"UPSTREAM_STOP={metadata['UPSTREAM_STOP']}", fill=(60,60,60), font=font)
    draw.text((16,72), f"SEM={metadata['SEMANTIC_RESULT_HASH'][:24]} LEDGER={metadata['STAGE_LEDGER_HASH'][:24]}", fill=(60,60,60), font=font)


def _project(points: np.ndarray, view: str, pixels: int) -> tuple[np.ndarray, tuple[float,float,float,float]]:
    a,b,sa,sb = VIEW_AXES[view]
    projected = np.stack((sa*points[:,a], sb*points[:,b]), axis=1)
    low=np.min(projected,axis=0); high=np.max(projected,axis=0); span=np.maximum(high-low,1e-12)
    low=low-0.02*span; high=high+0.02*span
    usable_y=pixels-112
    x=np.rint(12+(projected[:,0]-low[0])/(high[0]-low[0])*(pixels-24)).astype(np.int32)
    y=np.rint(100+(high[1]-projected[:,1])/(high[1]-low[1])*(usable_y-12)).astype(np.int32)
    xy=np.stack((np.clip(x,0,pixels-1),np.clip(y,96,pixels-1)),axis=1)
    return xy,(float(low[0]),float(high[0]),float(low[1]),float(high[1]))


def _scalar_colors(values: np.ndarray) -> np.ndarray:
    value=np.asarray(values,dtype=np.float64)
    finite=np.isfinite(value)
    output=np.tile(np.asarray([180,180,180],dtype=np.uint8),(len(value),1))
    if not np.any(finite): return output
    subset=value[finite]; low,high=np.quantile(subset,[.01,.99])
    if not math.isfinite(low) or not math.isfinite(high) or high<=low:
        indices=np.full(len(subset),128,dtype=np.uint8)
    else:
        indices=np.rint(255*np.clip((subset-low)/(high-low),0,1)).astype(np.uint8)
    output[finite]=COLOR_TABLE[indices]
    return output


def _raster_points(pixels: int, xy: np.ndarray, colors: np.ndarray, *, radius: int = 0) -> Image.Image:
    canvas=np.full((pixels,pixels,3),255,dtype=np.uint8)
    canvas[96:]=248
    x,y=xy[:,0],xy[:,1]
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            xx=np.clip(x+dx,0,pixels-1); yy=np.clip(y+dy,96,pixels-1); canvas[yy,xx]=colors
    return Image.fromarray(canvas,"RGB")


def _placeholder(pixels: int, title: str, metadata: Mapping[str,str]) -> Image.Image:
    image=Image.new("RGB",(pixels,pixels),(248,248,248)); _header(image,title,metadata)
    draw=ImageDraw.Draw(image); font=ImageFont.load_default()
    draw.line((200,200,pixels-200,pixels-200),fill=(180,180,180),width=8)
    draw.line((pixels-200,200,200,pixels-200),fill=(180,180,180),width=8)
    draw.text((pixels//2-180,pixels//2),f"{metadata['EXECUTION_STATE']} / {metadata['ENTITY_STATE']}",fill=(50,50,50),font=font)
    return image


def _geometry_values(layer: str, scratch: Mapping[str,np.ndarray], n: int) -> tuple[np.ndarray,np.ndarray,int]:
    ids=np.asarray(scratch["point_ids"],dtype=np.uint32)
    if layer=="complete-source": return np.arange(n,dtype=np.uint32),np.zeros(n),0
    if layer.startswith("q"):
        scale={"q24":0,"q48":1,"q96":2,"q192":3,"q384":4}[layer]
        return ids,np.asarray(scratch["q_q32"])[:,scale].astype(float),1
    if layer=="fitted-c": return ids,np.asarray(scratch["c_q32"]).astype(float),1
    if layer=="model-gain-g": return ids,np.asarray(scratch["gain_q32"]).astype(float),1
    if layer=="candidate-mask":
        mask=np.asarray(scratch["candidate"],dtype=bool); return ids[mask],np.ones(np.count_nonzero(mask)),2
    if layer=="supported-candidates-d":
        mask=np.asarray(scratch["supported"],dtype=bool); return ids[mask],np.asarray(scratch["d_q32"])[mask].astype(float),3
    raise KeyError(layer)


def _diagnostic_image(pixels: int, title: str, metadata: Mapping[str,str], lines: Sequence[str]) -> Image.Image:
    image=Image.new("RGB",(pixels,pixels),(248,248,248)); _header(image,title,metadata)
    draw=ImageDraw.Draw(image); font=ImageFont.load_default(); y=120
    for line in lines:
        draw.text((24,y),line,fill=(20,20,20),font=font); y+=18
        if y>pixels-24: break
    return image


def _select_context_cells(integer: np.ndarray, candidate_ids: np.ndarray) -> list[tuple[int,int,int]]:
    if not len(candidate_ids): return []
    cells=np.minimum(63,(integer[candidate_ids].astype(np.uint64)*64)//((1<<30)+1)).astype(np.int16)
    counts: dict[tuple[int,int,int],int]={}
    for row in cells:
        key=(int(row[0]),int(row[1]),int(row[2])); counts[key]=counts.get(key,0)+1
    def morton(key: tuple[int,int,int]) -> int:
        value=0
        for bit in range(6):
            for axis in range(3): value|=((key[axis]>>bit)&1)<<(3*bit+axis)
        return value
    chosen=[]
    for key,_count in sorted(counts.items(),key=lambda item:(-item[1],morton(item[0]))):
        if all(max(abs(key[i]-old[i]) for i in range(3))>=2 for old in chosen): chosen.append(key)
        if len(chosen)==12: break
    return chosen


def build_visual_package(
    output: Path,
    *,
    raw_source: Path,
    scratch_path: Path,
    semantic: Mapping[str,Any],
    roles: Sequence[Mapping[str,Any]],
) -> dict[str,Any]:
    points,integer,lattice=_load_points(raw_source)
    loaded=np.load(scratch_path,allow_pickle=False)
    scratch={key:loaded[key] for key in loaded.files}
    ledger=dict(semantic["stage_ledger"]); ledger_hash=str(semantic["stage_ledger_sha256"]); semantic_hash=str(semantic["semantic_sha256"])
    stage_state={"schema_version":1,"attempt_id":CONFIG.attempt_id,"semantic_result_hash":semantic_hash,"stage_ledger_hash":ledger_hash,"stages":ledger}
    (output/"stage_state.json").write_text(json.dumps(stage_state,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    images=[]; full=bool(semantic["full_evaluated"]); n=len(points)
    source_gray=np.tile(np.asarray([160,160,160],dtype=np.uint8),(n,1))
    projections={view:_project(points,view,CONFIG.global_image_pixels)[0] for view in CAMERAS}
    for view in CAMERAS:
        xy=projections[view]
        for layer in GEOMETRY_LAYERS:
            stage,state=_state_for_layer(layer,ledger,full); metadata=_metadata(stage,state,semantic_hash,ledger_hash)
            title=f"SDRA V1 | {view} | {layer}"
            if state["execution_state"]=="NOT_EVALUABLE": image=_placeholder(CONFIG.global_image_pixels,title,metadata)
            else:
                ids,values,kind=_geometry_values(layer,scratch,n)
                image=_raster_points(CONFIG.global_image_pixels,xy,source_gray,radius=0)
                if layer!="complete-source" and len(ids):
                    if kind==2: colors=np.tile(np.asarray([230,20,20],np.uint8),(len(ids),1))
                    else: colors=_scalar_colors(values)
                    overlay=np.array(image); ox,oy=xy[ids,0],xy[ids,1]
                    for dx in (-1,0,1):
                        for dy in (-1,0,1): overlay[np.clip(oy+dy,96,CONFIG.global_image_pixels-1),np.clip(ox+dx,0,CONFIG.global_image_pixels-1)]=colors
                    image=Image.fromarray(overlay,"RGB")
                _header(image,title,metadata)
            relative=Path("geometry")/f"{view.replace('+','pos').replace('-','neg')}-{layer}.png"; path=output/relative; _save_png(image,path,metadata)
            images.append({"path":relative.as_posix(),"sha256":_sha_file(path),"state":stage,"state_tuple":dict(state),"pixels":[image.width,image.height]})
    candidate_local=np.flatnonzero(np.asarray(scratch["candidate"],dtype=bool)); processed_ids=np.asarray(scratch["point_ids"],dtype=np.uint32); candidate_ids=processed_ids[candidate_local]
    cells=_select_context_cells(integer,candidate_ids) if full else []
    context_stage="candidate_mask" if full else "support"; context_state=ledger.get(context_stage,{"execution_state":"NOT_EVALUABLE","entity_state":"NOT_APPLICABLE","upstream_stop":str(semantic["decision"])})
    for index in range(12):
        metadata=_metadata(context_stage,context_state,semantic_hash,ledger_hash); title=f"SDRA V1 | candidate context {index+1:02d}"
        if index>=len(cells) or context_state["execution_state"]=="NOT_EVALUABLE": image=_placeholder(CONFIG.context_image_pixels,title,metadata)
        else:
            cell=np.asarray(cells[index]); center=((cell.astype(float)+.5)/64.0)*((1<<30)+1)
            local_candidate=candidate_ids[np.all(np.abs(integer[candidate_ids]-center)<=((1<<30)+1)/128,axis=1)]
            if not len(local_candidate): local_candidate=candidate_ids
            r=np.asarray(scratch["r384_integer"])[local_candidate if len(scratch["r384_integer"])==n else np.searchsorted(processed_ids,local_candidate)]
            radius=max(float(np.median(r))*4.0,((1<<30)+1)/128.0)
            center=np.mean(integer[local_candidate].astype(float),axis=0)
            source_mask=np.all(np.abs(integer-center)<=radius,axis=1); source_ids=np.flatnonzero(source_mask); candidates=local_candidate[np.all(np.abs(integer[local_candidate]-center)<=radius,axis=1)]
            subset=points[source_ids]; low=np.min(subset,axis=0); high=np.max(subset,axis=0); axes=np.argsort(high-low)[-2:]; projected=subset[:,axes]
            plo=np.min(projected,axis=0); phi=np.max(projected,axis=0); span=np.maximum(phi-plo,1e-12)
            x=np.rint(12+(projected[:,0]-plo[0])/span[0]*(CONFIG.context_image_pixels-24)).astype(int); y=np.rint(100+(phi[1]-projected[:,1])/span[1]*(CONFIG.context_image_pixels-112)).astype(int)
            xy=np.stack((np.clip(x,0,CONFIG.context_image_pixels-1),np.clip(y,96,CONFIG.context_image_pixels-1)),axis=1); colors=np.tile(np.asarray([155,155,155],np.uint8),(len(source_ids),1)); image=_raster_points(CONFIG.context_image_pixels,xy,colors)
            lookup={int(pid):i for i,pid in enumerate(source_ids)}; local=[lookup[int(pid)] for pid in candidates if int(pid) in lookup]
            arr=np.array(image); red=np.asarray([230,20,20],np.uint8)
            if local:
                ox,oy=xy[local,0],xy[local,1]
                for dx in (-2,-1,0,1,2):
                    for dy in (-2,-1,0,1,2): arr[np.clip(oy+dy,96,CONFIG.context_image_pixels-1),np.clip(ox+dx,0,CONFIG.context_image_pixels-1)]=red
            image=Image.fromarray(arr,"RGB"); _header(image,title,metadata)
        relative=Path("contexts")/f"context-{index+1:02d}.png"; path=output/relative; _save_png(image,path,metadata); images.append({"path":relative.as_posix(),"sha256":_sha_file(path),"state":context_stage,"state_tuple":dict(context_state),"pixels":[image.width,image.height]})
    diagnostic_specs=[
        ("histograms","full_features",[f"q{K_VALUES[i]} finite={int(np.count_nonzero(np.isfinite(np.asarray(scratch['q_q32'])[:,i])))}" for i in range(5)]+[f"candidate_count={semantic.get('candidate_count',0)}"]),
        ("top64-profiles","full_features",[f"top candidate point_id={int(pid)}" for pid in candidate_ids[:64]] or ["NO COMPLETED CANDIDATES"]),
        ("stage-state-matrix","source",[f"{key}: {value['execution_state']} / {value['entity_state']} / {value['upstream_stop']}" for key,value in ledger.items()]),
        ("morton-coverage","sentinel",[f"processed={semantic['processed_point_count']}",f"source={semantic['point_count']}",f"sentinel_eligible={semantic['sentinel_scale_eligible_count']}",f"sentinel_candidates={semantic['sentinel_candidate_count']}"]),
        ("role-agreement","source",[f"{row['role']} semantic={row['semantic_sha256']} decision={row['decision']}" for row in roles]),
    ]
    for name,stage,lines in diagnostic_specs:
        state=ledger.get(stage,{"execution_state":"NOT_EVALUABLE","entity_state":"NOT_APPLICABLE","upstream_stop":str(semantic["decision"])}); metadata=_metadata(stage,state,semantic_hash,ledger_hash); title=f"SDRA V1 | diagnostic | {name}"
        image=_placeholder(CONFIG.global_image_pixels,title,metadata) if state["execution_state"]=="NOT_EVALUABLE" else _diagnostic_image(CONFIG.global_image_pixels,title,metadata,lines)
        relative=Path("diagnostics")/f"{name}.png"; path=output/relative; _save_png(image,path,metadata); images.append({"path":relative.as_posix(),"sha256":_sha_file(path),"state":stage,"state_tuple":dict(state),"pixels":[image.width,image.height]})
    if len(images)!=77: raise ValueError(f"SDRA visual cardinality {len(images)} != 77")
    total=sum((output/row["path"]).stat().st_size for row in images)
    if total>CONFIG.visual_package_ceiling_bytes: raise ValueError("SDRA visual package byte ceiling exceeded")
    manifest={"schema_version":1,"attempt_id":CONFIG.attempt_id,"image_count":len(images),"image_bytes":total,"images":images,"renderer_sha256":_sha_file(Path(__file__)),"color_table_sha256":COLOR_TABLE_SHA256,"semantic_result_hash":semantic_hash,"stage_ledger_hash":ledger_hash}
    manifest["visual_manifest_sha256"]=sha_json(manifest)
    return manifest
