"""zs_lane_v4 compact support profiles (label-free, part-agnostic).

For every pooled candidate with flank directions (da, db: the Z3 candidates), per 1 mm station along its raw line,
extended to cover every Z2 line that pooling attached to it (+ ``PAD_MM`` on both ends):

* ``bend``  : scanned surface touches the apex line within the bend zone (zs_lane_v2.evidence.station_profile);
* ``fa``/``fb`` : plate material of flank A / flank B just beyond the bend zone;
* ``sw``    : the normal-sweep test (>= 3 of the 4 inner bins of the arc between the two flank normals hold a
              normal: a formed bend, not a seam or a sharp cut edge);
* ``touch`` : the tight apex-touch test at 0.5 mm stations (zs_lane_v2.evidence.bend_touch: a slot between tabs
              cuts through the bend, so the apex line leaves the surface there).

Profiles are stored as '0'/'1' strings with their start station ``s0`` (mm along the raw line from its start).
They let the selector and the post-processing test "is this extent continuously supported by a formed bend?"
without re-running geometry. The station tests are the frozen v2 functions, unchanged.
"""
from __future__ import annotations

import numpy as np

from research.zeroshot.zs_lane_v2 import evidence as E

PAD_MM = 5.0
MAX_EXT_MM = 800.0       # never profile beyond this many mm past a candidate end (bounded work)


def mesh_ctx(mesh, t: float, evidence: dict | None = None) -> E.MeshCtx:
    sp = (evidence or {}).get("spacing")
    return E.MeshCtx(mesh.V, mesh.vn, mesh.F, float(t), spacing=sp)


def bits(a) -> str:
    return "".join("1" if bool(x) else "0" for x in np.asarray(a).ravel())


def unbits(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def _line(c: dict) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(c["start"], float)
    b = np.asarray(c["end"], float)
    L = float(np.linalg.norm(b - a))
    return a, (b - a) / max(L, 1e-9), L


def attached_extents(cands: list[dict], z2_lines: list[dict]) -> dict[int, list[tuple[float, float, int]]]:
    """Candidate index -> [(s0, s1, z2 index)] extents (along the candidate's raw axis) of the Z2 lines pooling
    attached to it (zs_lane_v3.fixes.z2_attachment: the pool rule, best match per Z2 line)."""
    from research.zeroshot.zs_lane_v3.fixes import z2_attachment
    att = z2_attachment(cands, [{"start": l["start"], "end": l["end"]} for l in z2_lines])
    out: dict[int, list[tuple[float, float, int]]] = {}
    for i, lis in att.items():
        p0, d, L = _line(cands[i])
        for li in lis:
            z = z2_lines[li]
            s = sorted([float((np.asarray(z["start"], float) - p0) @ d), float((np.asarray(z["end"], float) - p0) @ d)])
            out.setdefault(int(i), []).append((s[0], s[1], int(li)))
    return out


def candidate_profiles(ctx: E.MeshCtx, cands: list[dict], z2_lines: list[dict]) -> dict[str, dict]:
    ext = attached_extents(cands, z2_lines)
    out: dict[str, dict] = {}
    for i, c in enumerate(cands):
        if "da" not in c or "db" not in c:
            continue
        p0, d, L = _line(c)
        if L < 1.0:
            continue
        lo, hi = 0.0, L
        for s0, s1, _ in ext.get(i, []):
            lo, hi = min(lo, s0), max(hi, s1)
        lo = max(lo, -MAX_EXT_MM) - PAD_MM
        hi = min(hi, L + MAX_EXT_MM) + PAD_MM
        prof = E.station_profile(ctx, c, s_lo=lo, s_hi=hi)
        touch = E.bend_touch(ctx, c, lo, hi, 0.5)
        out[str(i)] = {"s0": round(float(lo), 4), "L": round(L, 4), "bend": bits(prof["bend"]), "fa": bits(prof["fa"]),
                       "fb": bits(prof["fb"]), "sw": bits(prof["sweep"]), "touch": bits(touch),
                       "theta": round(float(prof.get("theta", float("nan"))), 3),
                       "z2_ext": [[round(a, 3), round(b, 3), k] for a, b, k in ext.get(i, [])]}
    return out


def frac(prof: dict, key: str, s_lo: float, s_hi: float) -> float:
    """Fraction of stations of profile ``key`` set within [s_lo, s_hi] (mm along the candidate's raw line)."""
    arr = unbits(prof[key])
    step = 0.5 if key == "touch" else 1.0
    if not len(arr) or s_hi <= s_lo:
        return float("nan")
    i0 = int(np.floor((s_lo - prof["s0"]) / step))
    i1 = int(np.ceil((s_hi - prof["s0"]) / step))
    i0, i1 = max(i0, 0), min(i1, len(arr))
    if i1 <= i0:
        return float("nan")
    return float(arr[i0:i1].mean())


def longest_gap(prof: dict, key: str, s_lo: float, s_hi: float) -> float:
    """Longest run (mm) of unset stations of ``key`` within [s_lo, s_hi]."""
    arr = unbits(prof[key])
    step = 0.5 if key == "touch" else 1.0
    i0 = max(int(np.floor((s_lo - prof["s0"]) / step)), 0)
    i1 = min(int(np.ceil((s_hi - prof["s0"]) / step)), len(arr))
    best = run = 0
    for x in arr[i0:i1]:
        run = 0 if x else run + 1
        best = max(best, run)
    return best * step
