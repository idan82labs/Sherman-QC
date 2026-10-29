"""Frozen research-only V33 projective normal fan moment measure.

The primitive is a candidate-wide, area-weighted triangle measure.  It does
not allocate or inspect mesh adjacency, graph edges, paths, stations, labels,
CAD, filenames, or expected counts.  Orchestration and immutable fixture
bindings live under ``scripts/`` and ``research/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import struct
from typing import Any, Iterable, Sequence

import numpy as np

from backend.f1_distributed_harmonic_fold_flux import (
    ACTION_LEGACY,
    Candidate,
    ordinary_nms,
    parse_candidates,
)


PROFILE = "V33_CANDIDATE_ALIGNED_SURFEL_VARIFOLD_V1"
PRIMITIVE = "PROJECTIVE_NORMAL_FAN_MOMENT_V1"
SCHEMA_VERSION = 1

MAX_VERTICES = 250_000
MAX_TRIANGLES = 500_000
MAX_CANDIDATES = 512
MAX_SUPPORT_TRIANGLES = 65_536
MAX_CLIPPED_PIECES = 131_072
MAX_INTERVAL_ENDPOINTS = 262_144
MAX_WORKERS = 8

CACHE_HEADER_BYTES = 64
CACHE_RECORD_BYTES = 192
ACTION_RESCUE = "PNFM_PASS"
ACTION_NONE = "NO_RESCUE"

REASONS = (
    "PNFM_INPUT_CAP",
    "PNFM_AREA_DENSITY",
    "PNFM_FRAME_GAP",
    "PNFM_AXIS_NORMALITY",
    "PNFM_FAN_SPAN",
    "PNFM_FAN_FILL",
    "PNFM_TRANSVERSE_COUPLING",
    "PNFM_AXIAL_DOMINANCE",
    "PNFM_TAIL_MASS",
    "PNFM_TAIL_WIDTH",
    "PNFM_TAIL_SEPARATION",
    "PNFM_AXIAL_COVERAGE",
    "PNFM_RADIAL_MULTIPLICITY",
    "PNFM_PASS",
    "ORDINARY_NMS_SUPPRESSED",
)
REASON_CODE = {name: index for index, name in enumerate(REASONS)}


class PNFMContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    return _canonical({key: value for key, value in result.items() if key != "_ephemeral"}) + b"\n"


def _round_even(value: float) -> int:
    return int(round(float(value)))


def _q(value: float, quantum: float) -> float:
    return _round_even(float(value) / quantum) * quantum


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise PNFMContractError("PNFM_INPUT_CAP", "zero/nonfinite vector")
    return value / norm


def _jacobi_principal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinned three-by-three symmetric Jacobi solve: 16 fixed sweeps."""
    a = np.asarray(matrix, dtype=np.float64).copy()
    v = np.eye(3, dtype=np.float64)
    for _ in range(16):
        for p, q in ((0, 1), (0, 2), (1, 2)):
            apq = float(a[p, q])
            if apq == 0.0:
                continue
            angle = 0.5 * math.atan2(2.0 * apq, float(a[q, q] - a[p, p]))
            c, s = math.cos(angle), math.sin(angle)
            j = np.eye(3, dtype=np.float64)
            j[p, p] = c; j[q, q] = c
            j[p, q] = s; j[q, p] = -s
            a = j.T @ a @ j
            v = v @ j
    values = np.diag(a)
    order = np.argsort(values)[::-1]
    return values[order], v[:, order]


@dataclass(frozen=True)
class CanonicalMesh:
    triangles: tuple[np.ndarray, ...]
    normals: tuple[np.ndarray, ...]
    areas: tuple[float, ...]
    source_sha256: str


def canonical_mesh(raw_source: dict[str, Any], scale: float) -> CanonicalMesh:
    allowed = {"positions", "triangles"}
    if set(raw_source) - allowed:
        raise PNFMContractError("PNFM_INPUT_CAP", f"forbidden source fields: {sorted(set(raw_source)-allowed)}")
    positions = np.asarray(raw_source.get("positions", []), dtype=np.float64)
    faces = np.asarray(raw_source.get("triangles", []), dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) > MAX_VERTICES:
        raise PNFMContractError("PNFM_INPUT_CAP", "vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) > MAX_TRIANGLES:
        raise PNFMContractError("PNFM_INPUT_CAP", "triangles")
    if not np.isfinite(positions).all() or np.any(faces < 0) or np.any(faces >= len(positions)):
        raise PNFMContractError("PNFM_INPUT_CAP", "numeric/index")
    if not math.isfinite(scale) or scale <= 0.0:
        raise PNFMContractError("PNFM_INPUT_CAP", "scale")
    dx = scale / (1 << 16)
    qpos = np.rint(positions / dx).astype(np.int64)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for face in faces:
        vertices = sorted((tuple(int(v) for v in qpos[int(index)]) for index in face))
        key = tuple(component for vertex in vertices for component in vertex)
        unique.setdefault(key, np.asarray(vertices, dtype=np.float64) * dx)
    triangles: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    areas: list[float] = []
    area_quantum = scale * scale / (1 << 24)
    minimum_area = scale * scale / (1 << 20)
    for key in sorted(unique):
        tri = unique[key]
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(cross))
        area = _q(0.5 * norm, area_quantum)
        if area < minimum_area or norm <= 0.0:
            continue
        normal = np.rint((cross / norm) * (1 << 30)).astype(np.int64).astype(np.float64) / (1 << 30)
        normal = _unit(normal)
        triangles.append(tri)
        normals.append(normal)
        areas.append(area)
    digest_payload = {
        "positions_q16": qpos.tolist(),
        "triangles": [list(key) for key in sorted(unique)],
        "scale": scale,
    }
    return CanonicalMesh(tuple(triangles), tuple(normals), tuple(areas), hashlib.sha256(_canonical(digest_payload)).hexdigest())


def _local(triangle: np.ndarray, p0: np.ndarray, origin: np.ndarray, a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.column_stack(((triangle - p0) @ a, (triangle - origin) @ b, (triangle - origin) @ m))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    t = 0.0 if denominator <= 0.0 else float(np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * direction)))


def _segment_segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    """Deterministic closest distance between two closed 3-D segments."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(np.dot(u, u)), float(np.dot(u, v)), float(np.dot(v, v))
    dd, ee = float(np.dot(u, w)), float(np.dot(v, w))
    denominator = aa * cc - bb * bb
    if aa <= 0.0:
        return _point_segment_distance(a0, b0, b1)
    if cc <= 0.0:
        return _point_segment_distance(b0, a0, a1)
    s_num, s_den = denominator, denominator
    t_num, t_den = denominator, denominator
    if denominator <= 1e-30:
        s_num, s_den = 0.0, 1.0
        t_num, t_den = ee, cc
    else:
        s_num = bb * ee - cc * dd
        t_num = aa * ee - bb * dd
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0.0:
        t_num = 0.0
        if -dd < 0.0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0.0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd + bb, aa
    sc = 0.0 if abs(s_num) <= 1e-30 else s_num / s_den
    tc = 0.0 if abs(t_num) <= 1e-30 else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def _point_triangle_distance(point: np.ndarray, tri: np.ndarray) -> float:
    """Ericson closest-point test, including all vertex/edge regions."""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + t * ab)))
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + t * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + t * (c - b))))
    normal = np.cross(ab, ac)
    return abs(float(np.dot(ap, normal))) / max(float(np.linalg.norm(normal)), 1e-30)


def _segment_triangle_distance(start: np.ndarray, end: np.ndarray, tri: np.ndarray) -> float:
    """Exact closed segment-to-triangle distance without topology queries."""
    a, b, c = tri
    normal = np.cross(b - a, c - a)
    denominator = float(np.dot(normal, end - start))
    if abs(denominator) > 1e-30:
        t = float(np.dot(normal, a - start) / denominator)
        if 0.0 <= t <= 1.0:
            hit = start + t * (end - start)
            if _point_triangle_distance(hit, tri) <= 1e-12:
                return 0.0
    distances = [_point_triangle_distance(start, tri), _point_triangle_distance(end, tri)]
    distances.extend(_segment_segment_distance(start, end, edge0, edge1) for edge0, edge1 in ((a, b), (b, c), (c, a)))
    return min(distances)


def _clip(poly: np.ndarray, axis: int, bound: float, keep_greater: bool, dx: float) -> np.ndarray:
    if len(poly) == 0:
        return poly
    out: list[np.ndarray] = []
    previous = poly[-1]
    prev_inside = bool(previous[axis] >= bound) if keep_greater else bool(previous[axis] <= bound)
    for current in poly:
        cur_inside = bool(current[axis] >= bound) if keep_greater else bool(current[axis] <= bound)
        if cur_inside != prev_inside:
            denominator = float(current[axis] - previous[axis])
            t = 0.0 if denominator == 0.0 else (bound - float(previous[axis])) / denominator
            point = previous + t * (current - previous)
            point = np.rint(point / dx) * dx
            point[axis] = bound
            out.append(point)
        if cur_inside:
            out.append(current.copy())
        previous, prev_inside = current, cur_inside
    return np.asarray(out, dtype=np.float64).reshape((-1, 3))


def _clip_prism(poly: np.ndarray, scale: float, length: float, coverage: bool = False) -> np.ndarray:
    dx = scale / (1 << 16)
    planes = (
        (0, 0.0 if coverage else -scale, True),
        (0, length if coverage else length + scale, False),
        (1, -12.0 * scale, True), (1, 12.0 * scale, False),
        (2, -12.0 * scale, True), (2, 12.0 * scale, False),
    )
    result = poly
    for axis, bound, greater in planes:
        result = _clip(result, axis, bound, greater, dx)
        if len(result) < 3:
            return np.empty((0, 3), dtype=np.float64)
    return result


def _polygon_moments(poly: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    area = iq = iq2 = iu = iu2 = ir = 0.0
    q_min = math.inf; q_max = -math.inf
    anchor = poly[0]
    for index in range(1, len(poly) - 1):
        tri = np.asarray((anchor, poly[index], poly[index + 1]))
        tri_area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
        if tri_area <= 0.0:
            continue
        u, q, r = tri[:, 0], tri[:, 1], tri[:, 2]
        area += tri_area
        iq += tri_area * float(np.sum(q)) / 3.0
        iu += tri_area * float(np.sum(u)) / 3.0
        ir += tri_area * float(np.sum(r)) / 3.0
        iq2 += tri_area * (float(np.sum(q*q)) + float(q[0]*q[1]+q[0]*q[2]+q[1]*q[2])) / 6.0
        iu2 += tri_area * (float(np.sum(u*u)) + float(u[0]*u[1]+u[0]*u[2]+u[1]*u[2])) / 6.0
        q_min = min(q_min, float(np.min(q))); q_max = max(q_max, float(np.max(q)))
    return area, iq, iq2, iu, iu2, ir, q_min, q_max


def _weighted_percentile(rows: Sequence[tuple[float, float]], fraction: float) -> float:
    ordered = sorted((float(value), float(weight)) for value, weight in rows if weight > 0.0)
    total = sum(weight for _value, weight in ordered)
    if total <= 0.0:
        return math.nan
    target = fraction * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _union_metrics(intervals: Iterable[tuple[float, float]], length: float, quantum: float) -> tuple[float, float]:
    values = sorted((_q(max(0.0, left), quantum), _q(min(length, right), quantum)) for left, right in intervals if right > 0.0 and left < length)
    merged: list[list[float]] = []
    for left, right in values:
        if right <= left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    coverage = sum(right-left for left, right in merged)
    cursor = 0.0; gap = 0.0
    for left, right in merged:
        gap = max(gap, left-cursor); cursor = right
    gap = max(gap, length-cursor)
    return coverage, gap


def _candidate_analysis(candidate: Candidate, mesh: CanonicalMesh) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_sha = hashlib.sha256(_canonical(candidate.raw)).hexdigest()
    passthrough_sha = hashlib.sha256(_canonical({"line_start": candidate.raw["render_line_start"], "line_end": candidate.raw["render_line_end"], "score": candidate.raw.get("score")})).hexdigest()
    base = {
        "candidate_id": candidate.candidate_id,
        "frozen_v16_order": candidate.frozen_order,
        "candidate_sha256": raw_sha,
        "passthrough_line_sha256": passthrough_sha,
        "anchor_xyz": candidate.center.tolist(), "axis_xyz": candidate.axis.tolist(),
        "line_start_xyz": candidate.line_start.tolist(), "line_end_xyz": candidate.line_end.tolist(), "scale": candidate.scale,
    }
    if candidate.selected_legacy:
        decision = dict(base, action=ACTION_LEGACY, reason="PNFM_PASS", metrics_i32=[0]*24, nms_reason="")
        return decision, {"pieces": []}
    if not candidate.eligible or not (8.0*candidate.scale <= candidate.length <= 512.0*candidate.scale):
        decision = dict(base, action=ACTION_NONE, reason="PNFM_INPUT_CAP", metrics_i32=[0]*24, nms_reason="")
        return decision, {"pieces": []}

    s = candidate.scale; p0 = candidate.line_start; p1 = candidate.line_end
    a = _unit(p1-p0); length = float(np.linalg.norm(p1-p0)); origin = 0.5*(p0+p1)
    support: list[int] = []
    projector = np.eye(3)-np.outer(a,a)
    broad_start, broad_end = p0 - 2.0 * s * a, p1 + 2.0 * s * a
    tensor = np.zeros((3,3), dtype=np.float64); broad_area = 0.0
    for index, (tri, normal, area) in enumerate(zip(mesh.triangles, mesh.normals, mesh.areas)):
        u = (tri-p0)@a
        if float(np.max(u)) < -2*s or float(np.min(u)) > length+2*s:
            continue
        if _segment_triangle_distance(broad_start, broad_end, tri) > 16*s:
            continue
        support.append(index)
        pn = projector@normal
        tensor += area*np.outer(pn,pn); broad_area += area
    if len(support)>MAX_SUPPORT_TRIANGLES or broad_area<=0:
        decision=dict(base,action=ACTION_NONE,reason="PNFM_INPUT_CAP",metrics_i32=[len(support)]+[0]*23,nms_reason="")
        return decision,{"pieces":[]}
    values,vectors=_jacobi_principal(tensor/broad_area)
    m=_unit(projector@vectors[:,0])
    for component in m:
        if abs(float(component))>2**-30:
            if component<0:m=-m
            break
    b=_unit(np.cross(a,m))
    gap=max(0.0,float(values[0]-values[1]))/max(1e-30,float(values[0]+values[1]))

    pieces=[]; total_area=axis_energy=0.0
    for index in support:
        tri,normal=mesh.triangles[index],mesh.normals[index]
        local=_local(tri,p0,origin,a,b,m)
        clipped=_clip_prism(local,s,length,False)
        if len(clipped)<3:continue
        area,iq,iq2,iu,iu2,ir,qmin,qmax=_polygon_moments(clipped)
        if area<=0:continue
        pn=projector@normal
        if float(np.linalg.norm(pn))<=0:continue
        nt=_unit(pn)
        if float(np.dot(nt,m))<0 or (float(np.dot(nt,m))==0 and float(np.dot(nt,b))<0):nt=-nt
        z=float(np.dot(nt,b)); theta=math.degrees(math.asin(float(np.clip(z,-1,1))))
        cov=_clip_prism(local,s,length,True)
        interval=(float(np.min(cov[:,0])),float(np.max(cov[:,0]))) if len(cov)>=3 else None
        pieces.append({"face":index,"area":area,"iq":iq,"iq2":iq2,"iu":iu,"iu2":iu2,"ir":ir,"q":iq/area,"u":iu/area,"r":ir/area,"z":z,"theta":theta,"interval":interval,"qmin":qmin,"qmax":qmax})
        total_area+=area; axis_energy+=area*float(np.dot(normal,a))**2
    if len(pieces)>MAX_CLIPPED_PIECES or len(pieces)*2>MAX_INTERVAL_ENDPOINTS or total_area<=0:
        decision=dict(base,action=ACTION_NONE,reason="PNFM_INPUT_CAP",metrics_i32=[len(support),len(pieces)]+[0]*22,nms_reason="")
        return decision,{"pieces":pieces,"frame":{"a":a.tolist(),"m":m.tolist(),"b":b.tolist()}}

    theta05=_weighted_percentile([(p["theta"],p["area"]) for p in pieces],.05)
    theta95=_weighted_percentile([(p["theta"],p["area"]) for p in pieces],.95)
    span=theta95-theta05
    low_limit=theta05+.10*span; core_lo=theta05+.35*span; core_hi=theta05+.65*span; high_limit=theta95-.10*span
    groups={"low":[p for p in pieces if p["theta"]<=low_limit],"core":[p for p in pieces if core_lo<=p["theta"]<=core_hi],"high":[p for p in pieces if p["theta"]>=high_limit]}
    def mass(rows):return sum(p["area"] for p in rows)
    def corr(xkey):
        ex=sum(p["iq"] if xkey=="q" else p["iu"] for p in pieces)/total_area
        ex2=sum(p["iq2"] if xkey=="q" else p["iu2"] for p in pieces)/total_area
        ez=sum(p["area"]*p["z"] for p in pieces)/total_area
        ez2=sum(p["area"]*p["z"]**2 for p in pieces)/total_area
        exz=sum((p["iq"] if xkey=="q" else p["iu"])*p["z"] for p in pieces)/total_area
        den=math.sqrt(max(0.0,ex2-ex*ex)*max(0.0,ez2-ez*ez))
        return math.nan if den<=0 else (exz-ex*ez)/den
    rhoq=corr("q"); rhou=corr("u")
    fill=np.zeros(16,dtype=np.float64)
    if span>0:
        for p in pieces:
            k=min(15,max(0,int(16*(p["theta"]-theta05)/span)))
            fill[k]+=p["area"]
    active=fill>=.0025*total_area; fill_count=int(np.sum(active))
    longest_empty=0;run=0
    for value in active[1:-1]:
        run=0 if value else run+1;longest_empty=max(longest_empty,run)
    widths={};medians={};cover={};gaps={}
    quantum=s/4096
    for name,rows in groups.items():
        qrows=[(p["q"],p["area"]) for p in rows]
        widths[name]=_weighted_percentile(qrows,.90)-_weighted_percentile(qrows,.10) if rows else math.nan
        medians[name]=_weighted_percentile(qrows,.50) if rows else math.nan
        c,g=_union_metrics((p["interval"] for p in rows if p["interval"] is not None),length,quantum);cover[name]=c/length;gaps[name]=g/length
    overall_c,overall_g=_union_metrics((p["interval"] for p in pieces if p["interval"] is not None),length,quantum)
    overall_cover=overall_c/length; overall_gap=overall_g/length

    occupancy=np.zeros((16,48),dtype=np.float64)
    for p in pieces:
        qi=min(15,max(0,int((p["q"]+12*s)/(24*s)*16)))
        ri=min(47,max(0,int((p["r"]+12*s)/(24*s)*48)))
        occupancy[qi,ri]+=p["area"]
    max_runs=0
    for qi in range(2,14):
        aq=float(np.sum(occupancy[qi]))
        if aq<.01*total_area:continue
        flags=occupancy[qi]>=max(.005*aq,.0002*total_area)
        filled=flags.copy()
        for ri in range(1,47):
            if not flags[ri] and flags[ri-1] and flags[ri+1]:filled[ri]=True
        runs=[];start=None
        for ri,value in enumerate(np.r_[filled,False]):
            if value and start is None:start=ri
            elif not value and start is not None:
                if float(np.sum(occupancy[qi,start:ri]))>=.05*aq:runs.append((start,ri))
                start=None
        max_runs=max(max_runs,len(runs))

    density=total_area/(length*s); axis_residual=axis_energy/total_area
    margin=abs(rhoq)-abs(rhou) if math.isfinite(rhoq) and math.isfinite(rhou) else -math.inf
    sep=abs(medians["high"]-medians["low"]) if all(math.isfinite(medians[k]) for k in ("low","high")) else math.nan
    predicates=(
        True,density>=6,gap>=.12,axis_residual<=.10,24<=span<=135,
        fill_count>=12 and longest_empty<=2,math.isfinite(rhoq) and abs(rhoq)>=.80,
        math.isfinite(margin) and margin>=.45,mass(groups["low"])>=.12*total_area and mass(groups["high"])>=.12*total_area and mass(groups["core"])>=.04*total_area,
        math.isfinite(widths["low"]) and widths["low"]>=2.5*s and math.isfinite(widths["high"]) and widths["high"]>=2.5*s,
        math.isfinite(sep) and sep>=4*s,overall_cover>=.80 and cover["low"]>=.55 and cover["core"]>=.45 and cover["high"]>=.55 and overall_gap<=.20 and max(gaps.values())<=.40,
        max_runs<=2,
    )
    failed=next((i for i,value in enumerate(predicates) if not value),None)
    reason="PNFM_PASS" if failed is None else REASONS[failed]
    mask=sum((1<<i) for i,value in enumerate(predicates) if value)
    metrics=[
        len(support),len(pieces),_round_even(density*1e6),_round_even(gap*1e6),_round_even(axis_residual*1e6),_round_even(span*1e4),fill_count,
        _round_even((rhoq if math.isfinite(rhoq) else 0)*1e6),_round_even((abs(rhou) if math.isfinite(rhou) else 0)*1e6),_round_even((margin if math.isfinite(margin) else -1)*1e6),
        _round_even(mass(groups["low"])/total_area*1e6),_round_even(mass(groups["high"])/total_area*1e6),_round_even(mass(groups["core"])/total_area*1e6),
        _round_even((widths["low"] if math.isfinite(widths["low"]) else 0)/s*(1<<16)),_round_even((widths["high"] if math.isfinite(widths["high"]) else 0)/s*(1<<16)),_round_even((sep if math.isfinite(sep) else 0)/s*(1<<16)),
        _round_even(overall_cover*1e6),_round_even(cover["low"]*1e6),_round_even(cover["core"]*1e6),_round_even(cover["high"]*1e6),
        _round_even(overall_gap*length/s*(1<<16)),_round_even(max(gaps.values())*length/s*(1<<16)),max_runs,mask,
    ]
    decision=dict(base,action=ACTION_RESCUE if reason=="PNFM_PASS" else ACTION_NONE,reason=reason,metrics_i32=metrics,nms_reason="")
    visual={"pieces":pieces,"groups":{name:[p["face"] for p in rows] for name,rows in groups.items()},"frame":{"a":a.tolist(),"m":m.tolist(),"b":b.tolist()},"theta05":theta05,"theta95":theta95,"occupancy":occupancy.tolist()}
    return decision,visual


def build_fixed_cache(decisions: Sequence[dict[str, Any]], source_sha256: str, config_sha256: str) -> bytes:
    header=bytearray(CACHE_HEADER_BYTES);header[:8]=b"PNFMV1\0\0"
    struct.pack_into("<HHI",header,8,1,CACHE_RECORD_BYTES,len(decisions))
    header[16:48]=bytes.fromhex(source_sha256);header[48:64]=bytes.fromhex(config_sha256)[:16]
    records=bytearray()
    for decision in sorted(decisions,key=lambda d:(int(d["frozen_v16_order"]),str(d["candidate_sha256"]))):
        record=bytearray(CACHE_RECORD_BYTES);record[:32]=bytes.fromhex(decision["candidate_sha256"])
        flags=(1 if decision["action"]==ACTION_RESCUE else 0)|(2 if decision["action"]==ACTION_LEGACY else 0)|(4 if decision.get("nms_reason")=="ORDINARY_NMS_SUPPRESSED" else 0)
        struct.pack_into("<HH",record,32,REASON_CODE.get(decision["reason"],65535),flags)
        struct.pack_into("<24i",record,36,*[int(np.clip(value,-2147483648,2147483647)) for value in decision["metrics_i32"]])
        record[132:164]=bytes.fromhex(decision["passthrough_line_sha256"])
        records.extend(record)
    return bytes(header+records)


def select_unified_v33(raw_source: dict[str,Any], raw_candidates: Sequence[dict[str,Any]], *, config_sha256: str, worker_count: int=1) -> dict[str,Any]:
    if len(raw_candidates)>MAX_CANDIDATES or worker_count<1 or worker_count>MAX_WORKERS:
        raise PNFMContractError("PNFM_INPUT_CAP","candidates/workers")
    candidates=parse_candidates(raw_candidates)
    scales={float(c.scale) for c in candidates}
    if len(scales)!=1:raise PNFMContractError("PNFM_INPUT_CAP","mixed scale")
    mesh=canonical_mesh(raw_source,next(iter(scales)))
    analyzed=[_candidate_analysis(candidate,mesh) for candidate in candidates]
    decisions=[row[0] for row in analyzed];visuals=[row[1] for row in analyzed]
    legacy=[(c,d) for c,d in zip(candidates,decisions) if d["action"]==ACTION_LEGACY]
    additions=[(c,d) for c,d in zip(candidates,decisions) if d["action"]==ACTION_RESCUE]
    ordered=sorted(legacy,key=lambda row:(row[0].frozen_order,row[0].candidate_id))+sorted(additions,key=lambda row:(row[0].frozen_order,row[1]["candidate_sha256"]))
    accepted,suppressed=ordinary_nms(ordered)
    for decision in decisions:
        if decision["candidate_id"] in suppressed:decision["nms_reason"]="ORDINARY_NMS_SUPPRESSED"
    final_lines=[{"candidate_id":c.candidate_id,"line_start":c.raw["render_line_start"],"line_end":c.raw["render_line_end"],"score":c.raw.get("score"),"source_lane":"legacy" if d["action"]==ACTION_LEGACY else "v33_pnfm_rescue"} for c,d in accepted]
    cache=build_fixed_cache(decisions,mesh.source_sha256,config_sha256)
    return {"schema_version":1,"kind":"sherman_qc_f1_v33_pnfm_research_result","profile":PROFILE,"primitive":PRIMITIVE,"research_only":True,"source_only":True,"rescue_only":True,"expected_count_used":False,"part_identity_used":False,"filename_identity_used":False,"labels_used":False,"cad_used":False,"decisions":decisions,"final_lines":final_lines,"pre_nms_rescue_count":sum(d["action"]==ACTION_RESCUE for d in decisions),"rescue_count":sum(v["source_lane"]=="v33_pnfm_rescue" for v in final_lines),"ordinary_nms_call_count":1,"candidate_added_after_nms":False,"cache_bytes":len(cache),"cache_sha256":hashlib.sha256(cache).hexdigest(),"source_sha256":mesh.source_sha256,"_ephemeral":{"candidate_visuals":visuals,"fixed_cache":cache,"mesh":mesh,"candidates":candidates}}
