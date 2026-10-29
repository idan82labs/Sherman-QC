"""zs_lane_v3 line fixes, applied after zs_lane_v2's selection and span post-processing (postproc.postprocess).

Two generic rules. Neither uses CAD, labels, file names or any part-specific input; both act only on the pooled
candidates, the Z2 lines and the per-candidate evidence that the v2 pipeline already computes.

EXT (adopt a longer Z2 extent at pooling)
    Pooling turns a Z2 line that duplicates a Z3 candidate (<= 4 deg, <= 2.5 mm, > 50 % of the shorter; the
    pool rule of z3_slicing.pooled.pool_scan, best match per Z2 line) into an agreement flag and discards its
    geometry. For every output line, each Z2 line absorbed by one of the line's candidates is considered in
    turn: when the Z2 line is at least ``ext_ratio`` x longer than the line's current extent and contains at
    least ``ext_min_in`` of that extent (measured along the output line's own axis), the line takes the union of
    the two extents along its own axis (its axis and transverse placement are kept). Output lines that an
    extended line now duplicates (same 4 deg / 2.5 mm / 50 % rule) and that are not longer are dropped. EXT
    never adds a line.

Z3R (rescue a strong Z3-only candidate)
    A candidate that only Z3 produced (no Z2 agreement), that is not part of any output line, whose Z3-alone
    selector probability is >= ``z3r_tau`` and whose station normal-sweep fraction (v2 evidence, the same
    measure as v2's sweep gate) is >= ``z3r_sweep_min`` is added at its snapped placement, unless it duplicates an
    output line. Candidates are visited in index order.

Line dicts follow postproc.postprocess: start_s / end_s (snapped placement), score, src, incl, parts, op. EXT
marks an extended line with ``fix = "EXT"`` (and keeps its op); a rescued line has ``op = "keep"``,
``fix = "Z3R"`` and its Z3-alone probability ``p_z3``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

#: Duplicate rule shared with pooling, v2's join dedupe and the Z3 NMS: angle / transverse offset / overlap.
DUP_ANGLE_DEG, DUP_OFFSET_MM, DUP_OVERLAP = 4.0, 2.5, 0.5


@dataclass(frozen=True)
class FixParams:
    ext: bool = True
    ext_ratio: float = 1.3          # the Z2 line is at least this many times longer than the line's extent
    ext_min_in: float = 0.6         # ... and contains at least this fraction of it (DEV plateau 0.5-0.75, middle)
    ext_dedupe: bool = True         # drop output lines an extended line now duplicates
    z3r: bool = True
    z3r_tau: float = 0.9            # Z3-alone GBM probability floor
    z3r_sweep_min: float = 0.25     # station normal-sweep fraction floor (v2's sweep gate value)

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())


def pair_geom(a0, a1, b0, b1) -> tuple[float, float, float]:
    """(axis angle deg, transverse offset mm, overlap as a fraction of the shorter) of segments a and b."""
    a0, a1, b0, b1 = (np.asarray(p, float) for p in (a0, a1, b0, b1))
    da, db = a1 - a0, b1 - b0
    la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    da, db = da / max(la, 1e-12), db / max(lb, 1e-12)
    ang = math.degrees(math.acos(min(1.0, abs(float(da @ db)))))
    ma, mb = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
    dm = mb - ma
    off = max(float(np.linalg.norm(dm - da * (dm @ da))), float(np.linalg.norm(dm - db * (dm @ db))))
    ia = sorted([float((a0 - ma) @ da), float((a1 - ma) @ da)])
    ib = sorted([float((b0 - ma) @ da), float((b1 - ma) @ da)])
    ov = max(0.0, min(ia[1], ib[1]) - max(ia[0], ib[0]))
    return ang, off, ov / max(min(la, lb), 1e-9)


def is_dup(a0, a1, b0, b1) -> bool:
    ang, off, ov = pair_geom(a0, a1, b0, b1)
    return ang < DUP_ANGLE_DEG and off < DUP_OFFSET_MM and ov > DUP_OVERLAP


def _len(line: dict) -> float:
    return float(np.linalg.norm(np.subtract(line["end_s"], line["start_s"])))


def z2_attachment(cands: list[dict], z2_lines: list[dict]) -> dict[int, list[int]]:
    """Candidate index -> indices of the Z2 lines pooling attached to it (raw geometry, pool rule, best match:
    smallest angle/4 + offset/2.5 - overlap among the Z3 candidates)."""
    att: dict[int, list[int]] = {}
    z3 = [j for j, c in enumerate(cands) if c.get("has_z3")]
    for li, ln in enumerate(z2_lines or []):
        best, bj = None, -1
        for j in z3:
            ang, off, ov = pair_geom(ln["start"], ln["end"], cands[j]["start"], cands[j]["end"])
            if ang < DUP_ANGLE_DEG and off < DUP_OFFSET_MM and ov > DUP_OVERLAP:
                sc = ang / DUP_ANGLE_DEG + off / DUP_OFFSET_MM - ov
                if best is None or sc < best:
                    best, bj = sc, j
        if bj >= 0:
            att.setdefault(bj, []).append(li)
    return att


def extend(lines: list[dict], cands: list[dict], z2_lines: list[dict], prm: FixParams) -> list[dict]:
    """EXT (module docstring). Returns new line dicts; the input is not modified."""
    att = z2_attachment(cands, z2_lines)
    out = []
    for ln in lines:
        a = np.asarray(ln["start_s"], float)
        b = np.asarray(ln["end_s"], float)
        L = float(np.linalg.norm(b - a))
        u = (b - a) / max(L, 1e-9)
        lo, hi = 0.0, L
        for i in ln.get("parts", []):
            for li in att.get(int(i), []):
                z = z2_lines[li]
                za, zb = np.asarray(z["start"], float), np.asarray(z["end"], float)
                Lz = float(np.linalg.norm(zb - za))
                s0, s1 = sorted([float((za - a) @ u), float((zb - a) @ u)])
                inside = max(0.0, min(hi, s1) - max(lo, s0)) / max(hi - lo, 1e-9)
                if Lz >= prm.ext_ratio * (hi - lo) and inside >= prm.ext_min_in:
                    lo, hi = min(lo, s0), max(hi, s1)
        n = dict(ln)
        if lo < -1e-9 or hi > L + 1e-9:
            n["start_s"] = (a + lo * u).tolist()
            n["end_s"] = (a + hi * u).tolist()
            n["fix"] = "EXT"
            n["ext_mm"] = [round(-lo, 3), round(hi - L, 3)]
        out.append(n)
    if not prm.ext_dedupe:
        return out
    drop: set[int] = set()
    extended = [k for k, o in enumerate(out) if o.get("fix") == "EXT"]
    for k in sorted(extended, key=lambda k: -_len(out[k])):
        if k in drop:
            continue
        Lk = _len(out[k])
        for m, o in enumerate(out):
            if m != k and m not in drop and _len(o) <= Lk and is_dup(out[k]["start_s"], out[k]["end_s"],
                                                                     o["start_s"], o["end_s"]):
                drop.add(m)
    return [o for m, o in enumerate(out) if m not in drop]


def rescue(lines: list[dict], cands: list[dict], prob_zs, prob_z3, sweep: list | None,
           prm: FixParams) -> list[dict]:
    """Z3R (module docstring). ``prob_z3`` is NaN for candidates without Z3 features; ``sweep`` is the v2
    evidence list (``[sw_full, ...]`` or None per candidate), or None when the evidence stage did not run."""
    if sweep is None:
        return list(lines)
    used = {int(i) for ln in lines for i in ln.get("parts", [])}
    out = list(lines)
    p3 = np.asarray(prob_z3, float)
    for k, c in enumerate(cands):
        if list(c.get("src") or []) != ["z3"] or k in used:
            continue
        if not (p3[k] >= prm.z3r_tau):          # NaN-safe
            continue
        sw = sweep[k] if k < len(sweep) else None
        if sw is None or not (float(sw[0]) >= prm.z3r_sweep_min):
            continue
        a, b = c.get("start_s", c["start"]), c.get("end_s", c["end"])
        if any(is_dup(a, b, o["start_s"], o["end_s"]) for o in out):
            continue
        out.append({"start_s": [float(v) for v in a], "end_s": [float(v) for v in b],
                    "score": float(np.asarray(prob_zs, float)[k]), "src": list(c.get("src") or []),
                    "incl": c.get("incl"), "parts": [int(k)], "op": "keep", "fix": "Z3R", "p_z3": float(p3[k])})
    return out


def apply(lines: list[dict], cands: list[dict], z2_lines: list[dict], prob_zs, prob_z3, sweep: list | None,
          prm: FixParams) -> list[dict]:
    """EXT, then Z3R (each only when enabled)."""
    out = extend(lines, cands, z2_lines, prm) if prm.ext else list(lines)
    if prm.z3r:
        out = rescue(out, cands, prob_zs, prob_z3, sweep, prm)
    return out
