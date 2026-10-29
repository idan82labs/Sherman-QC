"""Materially independent SDRA-V1 validator.

This module intentionally imports no producer geometry, neighbor, eigensolver,
model-fit, candidate, or renderer-state implementation.  It uses an exact
Morton-cell search, a closed-form symmetric eigensolver, and its own active-set
enumerator before rebuilding the canonical semantic payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from research.f1_ciip_v1 import partition, primary_parser
from .config import ACTIVE_SET_ORDER, CONFIG, CONFIG_SHA256, K_VALUES, canonical_json_bytes, sha_json


_Q32 = float(1 << 32)
_Q48 = float(1 << 48)
_EPS48 = 2.0 ** -48
_SPAN_UNITS = (1 << CONFIG.quantization_bits) + 1


class ValidatorStop(RuntimeError):
    def __init__(self, token: str, message: str) -> None:
        super().__init__(message)
        self.token = token


@dataclass(slots=True)
class _Source:
    integer: np.ndarray
    metric: np.ndarray
    lattice: float
    origin: np.ndarray
    raw_sha: str
    raw_bytes: int
    sampled_sha: str
    table_sha: str
    morton_order: np.ndarray


def _sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _sha_arrays(tag: bytes, arrays: Iterable[np.ndarray | bytes]) -> str:
    value = hashlib.sha256(tag)
    for array in arrays:
        raw = array if isinstance(array, bytes) else np.ascontiguousarray(array).tobytes()
        value.update(len(raw).to_bytes(8, "little"))
        value.update(raw)
    return value.hexdigest()


def _q32(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=(2**31 - 1), neginf=-(2**31))
    return np.rint(array * _Q32).astype(np.int64)


def _morton(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.uint64)
    code = np.zeros(len(cells), dtype=np.uint64)
    for bit in range(6):
        code |= ((cells[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((cells[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((cells[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


def _cells(integer: np.ndarray) -> np.ndarray:
    return np.minimum(63, (np.asarray(integer, dtype=np.uint64) * 64) // _SPAN_UNITS).astype(np.uint8)


def _canonicalize(sampled: np.ndarray, raw_sha: str, raw_bytes: int) -> _Source:
    xyz = np.asarray(sampled, dtype=np.float64)
    if xyz.shape != (CONFIG.sample_count, 3) or not np.all(np.isfinite(xyz)):
        raise ValidatorStop("SDRA_V1_SOURCE_CONTRACT_STOP", "validator sample contract failure")
    origin = np.min(xyz, axis=0)
    largest = float(np.max(np.max(xyz, axis=0) - origin))
    if largest <= 0 or not math.isfinite(largest):
        raise ValidatorStop("SDRA_V1_SOURCE_CONTRACT_STOP", "validator span failure")
    lattice = largest / float(1 << 30)
    integer = np.rint((xyz - origin) / lattice).astype(np.int64)
    order = np.lexsort((integer[:, 2], integer[:, 1], integer[:, 0]))
    integer = np.ascontiguousarray(integer[order], dtype="<i8")
    if np.any(np.all(integer[1:] == integer[:-1], axis=1)):
        raise ValidatorStop("SDRA_V1_SOURCE_CONTRACT_STOP", "validator duplicate quantized point")
    codes = _morton(_cells(integer))
    ids = np.arange(len(integer), dtype=np.uint32)
    morton_order = np.lexsort((ids, codes)).astype(np.uint32)
    return _Source(
        integer=integer,
        metric=origin + lattice * integer.astype(np.float64),
        lattice=lattice,
        origin=origin,
        raw_sha=raw_sha,
        raw_bytes=raw_bytes,
        sampled_sha=_sha_arrays(b"SDRA_V1_SAMPLED_BINARY64_XYZ\0", (np.asarray(xyz, dtype="<f8"),)),
        table_sha=_sha_arrays(b"SDRA_V1_CANONICAL_INTEGER_XYZ\0", (integer,)),
        morton_order=morton_order,
    )


def _permute(xyz: np.ndarray, source_sha: str) -> np.ndarray:
    rows = np.ascontiguousarray(xyz, dtype="<f8").view(np.uint8).reshape(len(xyz), 24)
    keys = np.empty((len(rows), 32), dtype=np.uint8)
    prefix = b"SDRA_V1_PERM" + bytes.fromhex(source_sha)
    for index, row in enumerate(rows):
        keys[index] = np.frombuffer(hashlib.sha256(prefix + bytes(row)).digest(), dtype=np.uint8)
    order = np.lexsort((np.ascontiguousarray(rows).view("V24").reshape(len(rows)), np.ascontiguousarray(keys).view("V32").reshape(len(keys))))
    return np.ascontiguousarray(xyz[order], dtype="<f8")


def rebuild_source(raw_source: Path, *, permute_input: bool = False) -> _Source:
    raw_sha, raw_bytes = _sha_file(raw_source), raw_source.stat().st_size
    if raw_sha != CONFIG.raw_source_sha256 or raw_bytes != CONFIG.raw_source_bytes:
        raise ValidatorStop("INVALID_SDRA_V1_GOVERNANCE_STOP", "validator source binding mismatch")
    descriptor = os.open(raw_source, os.O_RDONLY)
    try:
        xyz = np.asarray(primary_parser.parse_ply_fd(descriptor).xyz, dtype="<f8")
    finally:
        os.close(descriptor)
    if not np.all(np.isfinite(xyz)):
        raise ValidatorStop("SDRA_V1_SOURCE_CONTRACT_STOP", "validator nonfinite raw coordinate")
    if permute_input:
        xyz = _permute(xyz, raw_sha)
    packed = np.ascontiguousarray(xyz).view("V24").reshape(len(xyz))
    _unique, first, _inverse, counts = np.unique(packed, return_index=True, return_inverse=True, return_counts=True)
    sampled = partition.canonical_sample(
        np.ascontiguousarray(xyz[first], dtype="<f8"), np.asarray(counts, dtype=np.uint32), sample_cap=200_000
    )
    return _canonicalize(np.asarray(sampled.xyz, dtype=np.float64), raw_sha, raw_bytes)


def _cell_bounds(cell: int) -> tuple[int, int]:
    low = (cell * _SPAN_UNITS + 63) // 64
    high_exclusive = ((cell + 1) * _SPAN_UNITS + 63) // 64
    return low, high_exclusive


def _outside_lower_bound_squared(anchor: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
    distances: list[int] = []
    for axis in range(3):
        if int(lo[axis]) > 0:
            boundary, _ = _cell_bounds(int(lo[axis]))
            distances.append(int(anchor[axis]) - (boundary - 1))
        if int(hi[axis]) < 63:
            boundary, _ = _cell_bounds(int(hi[axis]) + 1)
            distances.append(boundary - int(anchor[axis]))
    if not distances:
        return 1 << 126
    distance = max(0, min(distances))
    return distance * distance


def exact_knn_morton(
    source: _Source, *, anchors: np.ndarray | None = None, k: int = 384
) -> tuple[np.ndarray, np.ndarray]:
    integer = source.integer
    anchor_ids = np.arange(len(integer), dtype=np.uint32) if anchors is None else np.asarray(anchors, dtype=np.uint32)
    cells = _cells(integer)
    bucket: dict[tuple[int, int, int], list[int]] = {}
    for point_id, cell in enumerate(cells):
        bucket.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(point_id)
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for point_id in anchor_ids:
        cell = cells[int(point_id)]
        grouped.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(int(point_id))
    out_ids = np.empty((len(anchor_ids), k), dtype=np.uint32)
    out_sq = np.empty((len(anchor_ids), k), dtype=np.uint64)
    output_row = {int(point_id): row for row, point_id in enumerate(anchor_ids)}
    for cell_key in sorted(grouped):
        cell = np.asarray(cell_key, dtype=np.int16)
        group_ids = np.asarray(grouped[cell_key], dtype=np.int64)
        completed = np.zeros(len(group_ids), dtype=bool)
        for radius in range(64):
            lo = np.maximum(0, cell - radius)
            hi = np.minimum(63, cell + radius)
            candidates: list[int] = []
            for x in range(int(lo[0]), int(hi[0]) + 1):
                for y in range(int(lo[1]), int(hi[1]) + 1):
                    for z in range(int(lo[2]), int(hi[2]) + 1):
                        candidates.extend(bucket.get((x, y, z), ()))
            if len(candidates) <= k:
                continue
            candidate_ids = np.asarray(sorted(candidates), dtype=np.int64)
            pending = np.flatnonzero(~completed)
            for local in pending:
                anchor = int(group_ids[local])
                usable = candidate_ids[candidate_ids != anchor]
                delta = integer[usable] - integer[anchor]
                sq = np.einsum("ij,ij->i", delta, delta, dtype=np.int64).astype(np.uint64)
                rank = np.lexsort((usable, sq))
                chosen, chosen_sq = usable[rank][:k], sq[rank][:k]
                outside = _outside_lower_bound_squared(integer[anchor], lo, hi)
                if int(chosen_sq[-1]) < outside or radius == 63:
                    row = output_row[anchor]
                    out_ids[row], out_sq[row] = chosen.astype(np.uint32), chosen_sq
                    completed[local] = True
            if np.all(completed):
                break
        if not np.all(completed):
            raise ValidatorStop("SDRA_V1_SOURCE_CONTRACT_STOP", "Morton exact search did not complete")
    return out_ids, out_sq


def _eigenvalues_symmetric3(matrix: np.ndarray) -> np.ndarray:
    """Closed-form symmetric 3x3 eigenvalues, ascending."""
    result = np.empty((len(matrix), 3), dtype=np.float64)
    identity = np.eye(3)
    for index, value in enumerate(matrix):
        off = float(value[0, 1] ** 2 + value[0, 2] ** 2 + value[1, 2] ** 2)
        if off == 0.0:
            result[index] = np.sort(np.diag(value))
            continue
        center = float(np.trace(value) / 3.0)
        spread2 = float(np.sum((np.diag(value) - center) ** 2) + 2.0 * off)
        spread = math.sqrt(max(spread2 / 6.0, 0.0))
        if spread == 0.0:
            result[index] = center
            continue
        normalized = (value - center * identity) / spread
        half_det = float(np.linalg.det(normalized) / 2.0)
        angle = math.acos(max(-1.0, min(1.0, half_det))) / 3.0
        largest = center + 2.0 * spread * math.cos(angle)
        smallest = center + 2.0 * spread * math.cos(angle + 2.0 * math.pi / 3.0)
        middle = 3.0 * center - smallest - largest
        result[index] = (smallest, middle, largest)
    return result


def _models(x: np.ndarray, y: np.ndarray, include_c: bool) -> tuple[np.ndarray, np.ndarray]:
    names = ("a", "c", "b")
    basis = np.stack((1.0 / x, np.ones_like(x), x), axis=2)
    order = [row for row in ACTIVE_SET_ORDER if include_c or "c" not in row]
    best = np.full(len(x), np.iinfo(np.int64).max, dtype=np.int64)
    output = np.zeros((len(x), 3), dtype=np.int64)
    for active in order:
        columns = [names.index(name) for name in active]
        design = basis[:, :, columns]
        gram = np.einsum("nki,nkj->nij", design, design)
        rhs = np.einsum("nki,nk->ni", design, y)
        try:
            coefficient = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficient = np.asarray([np.linalg.lstsq(design[i], y[i], rcond=None)[0] for i in range(len(x))])
        valid = np.all(coefficient >= -(2.0 ** -40), axis=1)
        fixed = _q32(np.maximum(coefficient, 0.0))
        prediction = np.einsum("nki,ni->nk", design, fixed.astype(np.float64) / _Q32)
        sse = _q32(np.sum((y - prediction) ** 2, axis=1))
        improve = valid & (sse < best)
        best[improve] = sse[improve]
        for local, column in enumerate(columns):
            output[improve, column] = fixed[improve, local]
    if np.any(best == np.iinfo(np.int64).max):
        raise ValidatorStop("SDRA_V1_SOURCE_CONTRACT_STOP", "validator active-set failure")
    return output, best


def _point_features(source: _Source, anchors: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    neighbors, squared = exact_knn_morton(source, anchors=anchors)
    n = len(anchors)
    q = np.zeros((n, 5), dtype=np.float64)
    x = np.zeros((n, 5), dtype=np.float64)
    mu1 = np.full(n, np.inf)
    r384 = np.zeros(n)
    for start in range(0, n, 4096):
        stop = min(n, start + 4096)
        ids = anchors[start:stop]
        sq = squared[start:stop].astype(np.float64)
        base = np.mean(sq[:, :24], axis=1)
        for scale, k in enumerate(K_VALUES):
            chosen = neighbors[start:stop, :k]
            delta = source.integer[chosen].astype(np.float64) - source.integer[ids, None, :].astype(np.float64)
            unit = delta / np.sqrt(sq[:, :k])[..., None]
            moment = np.einsum("nki,nkj->nij", unit, unit) / float(k)
            moment = np.rint(moment * _Q48) / _Q48
            eigen = _eigenvalues_symmetric3(moment)
            eigen = _q32(eigen).astype(np.float64) / _Q32
            q[start:stop, scale] = eigen[:, 0]
            mu1[start:stop] = np.minimum(mu1[start:stop], eigen[:, 1])
            radius2 = np.mean(sq[:, :k], axis=1)
            x[start:stop, scale] = _q32(radius2 / base).astype(np.float64) / _Q32
            if k == 384:
                r384[start:stop] = np.sqrt(radius2)
    qq, xq, muq = _q32(q), _q32(x), _q32(mu1)
    ratio = _q32(x[:, 1:] / x[:, :-1])
    eligible = (
        np.all(ratio >= CONFIG.successive_x_ratio_low_q32, axis=1)
        & np.all(ratio <= CONFIG.successive_x_ratio_high_q32, axis=1)
        & (xq[:, -1] >= CONFIG.x384_low_q32) & (xq[:, -1] <= CONFIG.x384_high_q32)
        & (muq >= CONFIG.mu1_minimum_q32)
    )
    aq = np.zeros(n, np.int64); cq = np.zeros(n, np.int64); bq = np.zeros(n, np.int64)
    gq = np.zeros(n, np.int64); eq = np.zeros(n, np.int64); pq = np.zeros(n, np.int64)
    h0q = np.zeros(n, np.int64); h1q = np.zeros(n, np.int64)
    rows = np.flatnonzero(eligible)
    if len(rows):
        xf, yf = xq[rows].astype(float) / _Q32, qq[rows].astype(float) / _Q32
        h0c, h0 = _models(xf, yf, False); h1c, h1 = _models(xf, yf, True)
        gain = np.maximum(0, (h0 - h1).astype(float) / _Q32 / np.maximum(h0.astype(float) / _Q32, _EPS48))
        error = (h1.astype(float) / _Q32) / np.maximum(np.sum(yf * yf, axis=1), _EPS48)
        af, cf, bf = h1c[:, 0] / _Q32, h1c[:, 1] / _Q32, h1c[:, 2] / _Q32
        phi = cf / np.maximum(af / xf[:, -1] + cf + bf * xf[:, -1] + 2.0 ** -24, _EPS48)
        aq[rows], cq[rows], bq[rows] = h1c.T
        gq[rows], eq[rows], pq[rows], h0q[rows], h1q[rows] = _q32(gain), _q32(error), _q32(phi), h0, h1
    candidate = eligible & (cq >= CONFIG.c_low_q32) & (cq <= CONFIG.c_high_q32) & (gq >= CONFIG.gain_minimum_q32) & (eq <= CONFIG.error_maximum_q32) & (pq >= CONFIG.phi_minimum_q32)
    return {
        "point_ids": anchors.astype(np.uint32), "q": qq, "x": xq, "mu1": muq,
        "a": aq, "c": cq, "b": bq, "gain": gq, "error": eq, "phi": pq,
        "eligible": eligible, "candidate": candidate, "r384": r384, "h0": h0q, "h1": h1q,
    }, neighbors


def _rank(values: np.ndarray, probability: float) -> int:
    values = np.sort(np.asarray(values, dtype=np.int64), kind="stable")
    if not len(values): return 0
    return int(values[max(0, min(len(values)-1, math.ceil(probability * len(values))-1))])


def _state(execution: str, entity: str, upstream: str = "NONE") -> dict[str, str]:
    return {"execution_state": execution, "entity_state": entity, "upstream_stop": upstream}


def _point_hash(f: Mapping[str, np.ndarray], supported: np.ndarray, m48: np.ndarray, m192: np.ndarray, d: np.ndarray) -> str:
    return _sha_arrays(b"SDRA_V1_PER_POINT_SEMANTICS\0", (
        f["point_ids"].astype("<u4"), f["q"].astype("<i8"), f["x"].astype("<i8"), f["mu1"].astype("<i8"),
        f["a"].astype("<i8"), f["c"].astype("<i8"), f["b"].astype("<i8"), f["gain"].astype("<i8"),
        f["error"].astype("<i8"), f["phi"].astype("<i8"), f["eligible"].astype(np.uint8),
        f["candidate"].astype(np.uint8), supported.astype(np.uint8), m48.astype("<u2"), m192.astype("<u2"), d.astype("<i8"),
    ))


def validate_source(raw_source: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = rebuild_source(raw_source)
    ranks = np.linspace(0, len(source.integer)-1, 8192, dtype=np.int64)
    sentinel_ids = source.morton_order[ranks]
    sentinel, _ = _point_features(source, sentinel_ids)
    se, sc = int(np.count_nonzero(sentinel["eligible"])), int(np.count_nonzero(sentinel["candidate"]))
    if se < CONFIG.sentinel_scale_eligible_minimum:
        decision = "SDRA_V1_SAMPLING_REGIME_ARCHITECTURE_STOP"
    elif sc > CONFIG.sentinel_candidate_maximum:
        decision = "SDRA_V1_NONSELECTIVE_ARCHITECTURE_STOP"
    else:
        decision = "CONTINUE_FULL_SOURCE"
    if decision != "CONTINUE_FULL_SOURCE":
        zeros_b=np.zeros(len(sentinel_ids),bool); zeros_u=np.zeros(len(sentinel_ids),np.uint16); zeros_i=np.zeros(len(sentinel_ids),np.int64)
        ledger={"source":_state("OBSERVED_COMPLETE","PRESENT"),"sentinel":_state("OBSERVED_COMPLETE","PRESENT"),"full_features":_state("NOT_EVALUABLE","NOT_APPLICABLE",decision),"support":_state("NOT_EVALUABLE","NOT_APPLICABLE",decision)}
        payload={"schema_version":1,"attempt_id":CONFIG.attempt_id,"configuration_sha256":CONFIG_SHA256,"source_sha256":source.raw_sha,"source_bytes":source.raw_bytes,"sampled_coordinate_sha256":source.sampled_sha,"canonical_table_sha256":source.table_sha,"point_count":len(source.integer),"processed_point_count":len(sentinel_ids),"sentinel_scale_eligible_count":se,"sentinel_candidate_count":sc,"full_evaluated":False,"candidate_count":0,"supported_candidate_count":0,"median_d_q32":0,"p90_d_q32":0,"decision":decision,"stage_ledger":ledger}
        payload["stage_ledger_sha256"]=sha_json(ledger); payload["per_point_semantic_sha256"]=_point_hash(sentinel,zeros_b,zeros_u,zeros_u,zeros_i); payload["semantic_sha256"]=sha_json(payload)
        return payload, sentinel
    f, neighbors = _point_features(source, np.arange(len(source.integer),dtype=np.uint32))
    candidate=f["candidate"]; m48=np.sum(candidate[neighbors[:,:48]],axis=1,dtype=np.uint16); m192=np.sum(candidate[neighbors[:,:192]],axis=1,dtype=np.uint16)
    supported=candidate&(m48>=2); support_ids=np.flatnonzero(supported); d=np.zeros(len(candidate),np.int64)
    if len(support_ids): d[support_ids]=_q32(np.log2((m192[support_ids].astype(float)+1)/(m48[support_ids].astype(float)+1)))
    med,p90=_rank(d[support_ids],.5),_rank(d[support_ids],.9); ec=int(np.count_nonzero(f["eligible"])); cc=int(np.count_nonzero(candidate)); ss=int(np.count_nonzero(supported))
    if ec<CONFIG.full_scale_eligible_minimum: decision="SDRA_V1_SAMPLING_REGIME_ARCHITECTURE_STOP"
    elif cc>CONFIG.candidate_maximum: decision="SDRA_V1_NONSELECTIVE_ARCHITECTURE_STOP"
    elif cc==0: decision="SDRA_V1_NO_SIGNAL_ARCHITECTURE_STOP"
    elif ss*2<cc: decision="SDRA_V1_INCOHERENT_SUPPORT_ARCHITECTURE_STOP"
    elif med>CONFIG.support_median_d_maximum_q32 or p90>CONFIG.support_p90_d_maximum_q32: decision="SDRA_V1_AREA_LIKE_SUPPORT_ARCHITECTURE_STOP"
    else: decision="SDRA_V1_SOURCE_REPRESENTATION_PREFLIGHT_SURVIVES"
    ledger={"source":_state("OBSERVED_COMPLETE","PRESENT"),"sentinel":_state("OBSERVED_COMPLETE","PRESENT"),"full_features":_state("OBSERVED_COMPLETE","PRESENT"),"candidate_mask":_state("OBSERVED_COMPLETE","PRESENT" if cc else "NOT_PRESENT"),"support":_state("OBSERVED_COMPLETE","PRESENT" if ss else "NOT_PRESENT")}
    ids=np.flatnonzero(candidate); rows=[{"point_id":int(i),"c_q32":int(f["c"][i]),"gain_q32":int(f["gain"][i]),"error_q32":int(f["error"][i]),"phi_q32":int(f["phi"][i]),"m48":int(m48[i]),"m192":int(m192[i]),"supported":bool(supported[i]),"d_q32":int(d[i])} for i in ids]
    payload={"schema_version":1,"attempt_id":CONFIG.attempt_id,"configuration_sha256":CONFIG_SHA256,"source_sha256":source.raw_sha,"source_bytes":source.raw_bytes,"sampled_coordinate_sha256":source.sampled_sha,"canonical_table_sha256":source.table_sha,"point_count":len(source.integer),"processed_point_count":len(source.integer),"sentinel_scale_eligible_count":se,"sentinel_candidate_count":sc,"full_evaluated":True,"scale_eligible_count":ec,"candidate_count":cc,"supported_candidate_count":ss,"median_d_q32":med,"p90_d_q32":p90,"candidate_rows":rows,"decision":decision,"stage_ledger":ledger}
    payload["stage_ledger_sha256"]=sha_json(ledger); payload["per_point_semantic_sha256"]=_point_hash(f,supported,m48,m192,d); payload["semantic_sha256"]=sha_json(payload)
    scratch={"point_ids":f["point_ids"],"q_q32":f["q"],"x_q32":f["x"],"c_q32":f["c"],"gain_q32":f["gain"],"error_q32":f["error"],"phi_q32":f["phi"],"candidate":candidate.astype(np.uint8),"supported":supported.astype(np.uint8),"d_q32":d,"r384_integer":f["r384"]}
    return payload,scratch


def verify_stage_ledger(ledger: Mapping[str, Mapping[str, str]]) -> None:
    for stage, state in ledger.items():
        execution, entity = state.get("execution_state"), state.get("entity_state")
        if execution not in ("OBSERVED_COMPLETE","OBSERVED_PARTIAL","NOT_EVALUABLE"):
            raise ValueError(f"invalid execution state: {stage}")
        allowed={"OBSERVED_COMPLETE":{"PRESENT","NOT_PRESENT"},"OBSERVED_PARTIAL":{"PRESENT","NOT_APPLICABLE"},"NOT_EVALUABLE":{"NOT_APPLICABLE"}}
        if entity not in allowed[execution]: raise ValueError(f"invalid state tuple: {stage}")


def verify_visual_manifest(output: Path) -> dict[str, Any]:
    import json
    manifest=json.loads((output/"visual-manifest.json").read_text(encoding="utf-8")); ledger=json.loads((output/"stage_state.json").read_text(encoding="utf-8"))
    verify_stage_ledger(ledger["stages"])
    if len(manifest["images"]) != CONFIG.authoritative_png_count: raise ValueError("visual image cardinality mismatch")
    for row in manifest["images"]:
        path=output/row["path"]
        if _sha_file(path)!=row["sha256"]: raise ValueError(f"visual hash mismatch: {row['path']}")
        state=row["state"]
        if state not in ledger["stages"]: raise ValueError("manifest references unknown stage")
        if row["state_tuple"]!=ledger["stages"][state]: raise ValueError("manifest stage tuple mismatch")
    return {"visual_manifest_verified":True,"image_count":len(manifest["images"]),"visual_manifest_sha256":sha_json(manifest)}

