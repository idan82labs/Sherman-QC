"""Option 4 (DEV prototype): rigid registration seeds from zs bend lines matched to CAD bend lines.

Rotation-invariant graph matching: a correspondence pair (zs i, zs j) <-> (CAD a, CAD b) is admitted when the
undirected inter-line angle agrees within ANG_TOL, the inter-line distance within DIST_TOL and each zs span is not
longer than its CAD span (+ slack); it fixes a proper rigid transform (Kabsch on the two directions and their cross
product; translation by least squares of the perpendicular point-to-line residuals). All-parallel pairs use the
perpendicular offset vector as the second direction and the span midpoints for the along-axis slide. Each hypothesis
is scored by one-to-one inliers over all lines, refined on its inliers and deduplicated. Mirror images are NOT
produced (proper rotations only); a mirrored scan gets its own seeds from its mirrored lines.
Inputs: zs lines (raw frame), CAD formed-bend OUTER_APEX lines per stage model (cad_model_mm). No file names, no ids.
"""
import math
import numpy as np

ANG_TOL = 4.0       # deg, inter-line angle agreement
DIST_TOL = 3.0      # mm, inter-line distance agreement
IN_ANG = 4.0        # deg, inlier axis angle
IN_OFF = 3.0        # mm, inlier perpendicular offset
LEN_SLACK = 1.15


def _seg(l):
    a, b = np.asarray(l[0], float), np.asarray(l[1], float)
    d = b - a
    L = float(np.linalg.norm(d))
    return a, b, d / max(L, 1e-9), L, (a + b) / 2


def _acute(u, v):
    return math.degrees(math.acos(min(1.0, abs(float(u @ v)))))


def _line_dist(m1, u1, m2, u2):
    n = np.cross(u1, u2)
    nn = np.linalg.norm(n)
    w = m2 - m1
    if nn < 1e-6:
        return float(np.linalg.norm(w - u1 * (w @ u1)))
    return abs(float(w @ n)) / nn


def _kabsch(A, B):
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _trans(R, mids, qs, dirs, along=None):
    S = np.zeros((3, 3)); r = np.zeros(3)
    for m, q, c in zip(mids, qs, dirs):
        P = np.eye(3) - np.outer(c, c)
        S += P; r += P @ (q - R @ m)
    if along is not None:          # all parallel: fix the slide along the common axis by the span midpoints
        c, dm = along
        S += np.outer(c, c); r += np.outer(c, c) @ dm
    return np.linalg.lstsq(S, r, rcond=None)[0]


def _inliers(R, t, Z, C):
    pairs, used = [], set()
    cand = []
    for i, (a, b, u, L, m) in enumerate(Z):
        ma, mb = R @ a + t, R @ b + t
        uu = R @ u
        for k, (ca, cb, cu, CL, cm) in enumerate(C):
            ang = _acute(uu, cu)
            if ang > IN_ANG:
                continue
            off = max(np.linalg.norm((x - ca) - cu * ((x - ca) @ cu)) for x in (ma, mb))
            if off > IN_OFF:
                continue
            s = sorted(((ma - ca) @ cu, (mb - ca) @ cu))
            ov = max(0.0, min(s[1], CL) - max(s[0], 0.0)) / max(L, 1e-9)
            if ov < 0.3:
                continue
            cand.append((off / IN_OFF + ang / IN_ANG - ov, i, k))
    cand.sort()
    ui = set()
    for _, i, k in cand:
        if i in ui or k in used:
            continue
        ui.add(i); used.add(k); pairs.append((i, k))
    return pairs


def _refine(R, t, Z, C, pairs, iters=8):
    for _ in range(iters):
        P, Q = [], []
        for i, k in pairs:
            a, b, u, L, m = Z[i]
            ca, cb, cu, CL, cm = C[k]
            for s in np.linspace(0, 1, 7):
                x = a + (b - a) * s
                y = R @ x + t
                Q.append(ca + cu * ((y - ca) @ cu)); P.append(x)
        P, Q = np.asarray(P), np.asarray(Q)
        # point-to-line Procrustes step (closest points on the CAD lines)
        pc, qc = P.mean(0), Q.mean(0)
        R2 = _kabsch(P - pc, Q - qc)
        t2 = qc - R2 @ pc
        if len({tuple(np.round(C[k][2], 3)) for _, k in pairs}) < 2 and len(pairs) >= 1:
            break          # all inliers parallel: the slide along the axis is unobservable, keep the seed
        R, t = R2, t2
    return R, t


def _rigid(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def same_pose(T1, T2, ang=3.0, dist=3.0, probe=None):
    dR = T1[:3, :3] @ T2[:3, :3].T
    a = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    p = np.zeros((1, 3)) if probe is None else probe
    d = float(np.max(np.linalg.norm((p @ T1[:3, :3].T + T1[:3, 3]) - (p @ T2[:3, :3].T + T2[:3, 3]), axis=1)))
    return a <= ang and d <= dist, a, d


def seeds(zs_lines, cad_stages, keep=4, min_inliers=3):
    """zs_lines: [[start, end]] raw; cad_stages: {stage_id: [[start, end], ...]} cad frame.
    Returns [{'T': 4x4 raw->cad, 'inliers': n, 'stage': id, 'rms': mm}] best first (proper rotations only)."""
    Z = [_seg(l) for l in zs_lines if np.linalg.norm(np.subtract(l[1], l[0])) > 3]
    out = []
    if len(Z) < 2:
        return out
    probe = np.asarray([z[4] for z in Z])
    for sid, lines in cad_stages.items():
        C = [_seg(l) for l in lines]
        if len(C) < 2:
            continue
        hyps = []
        for i in range(len(Z)):
            for j in range(len(Z)):
                if j <= i:
                    continue
                ai, aj = Z[i], Z[j]
                angz = _acute(ai[2], aj[2]); dz = _line_dist(ai[4], ai[2], aj[4], aj[2])
                par = angz < 8.0
                if par:
                    w = (aj[4] - ai[4]); w = w - ai[2] * (w @ ai[2])
                    if np.linalg.norm(w) < 5.0:
                        continue
                for a in range(len(C)):
                    if ai[3] > C[a][3] * LEN_SLACK + 5:
                        continue
                    for b in range(len(C)):
                        if b == a or aj[3] > C[b][3] * LEN_SLACK + 5:
                            continue
                        ca, cb = C[a], C[b]
                        if abs(_acute(ca[2], cb[2]) - angz) > ANG_TOL or abs(_line_dist(ca[4], ca[2], cb[4], cb[2]) - dz) > DIST_TOL:
                            continue
                        for si in (1, -1):
                            for sj in (1, -1):
                                if not par:
                                    A = np.stack([ai[2], aj[2], np.cross(ai[2], aj[2])])
                                    v1, v2 = si * ca[2], sj * cb[2]
                                    B = np.stack([v1, v2, np.cross(v1, v2)])
                                    if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
                                        continue
                                    R = _kabsch(A, B)
                                    if max(_acute(R @ ai[2], v1), _acute(R @ aj[2], v2)) > ANG_TOL:
                                        continue
                                    t = _trans(R, [ai[4], aj[4]], [ca[4], cb[4]], [ca[2], cb[2]])
                                else:
                                    if sj == -1:
                                        continue
                                    wc = cb[4] - ca[4]; wc = wc - ca[2] * (wc @ ca[2])
                                    if abs(np.linalg.norm(wc) - np.linalg.norm(w)) > DIST_TOL:
                                        continue
                                    v1 = si * ca[2]
                                    A = np.stack([ai[2], w / np.linalg.norm(w), np.cross(ai[2], w / np.linalg.norm(w))])
                                    wn = wc / np.linalg.norm(wc)
                                    B = np.stack([v1, wn, np.cross(v1, wn)])
                                    R = _kabsch(A, B)
                                    mz = (ai[4] + aj[4]) / 2; mc = (ca[4] + cb[4]) / 2
                                    t = _trans(R, [ai[4], aj[4]], [ca[4], cb[4]], [ca[2], cb[2]], along=(ca[2], mc - R @ mz))
                                hyps.append((R, t))
        best = []
        for R, t in hyps:
            pr = _inliers(R, t, Z, C)
            if len(pr) < min(min_inliers, len(Z), len(C)):
                continue
            best.append((len(pr), R, t, pr))
        best.sort(key=lambda x: -x[0])
        seen = []
        for n, R, t, pr in best[:200]:
            T0 = _rigid(R, t)
            if any(same_pose(T0, s, probe=probe)[0] for s in seen):
                continue
            seen.append(T0)
            R2, t2 = _refine(R, t, Z, C, pr)
            pr2 = _inliers(R2, t2, Z, C)
            if len(pr2) < len(pr):
                R2, t2, pr2 = R, t, pr
            res = []
            for i, k in pr2:
                a, b, u, L, m = Z[i]; ca, cb, cu, CL, cm = C[k]
                for x in (a, b):
                    y = R2 @ x + t2
                    res.append(np.linalg.norm((y - ca) - cu * ((y - ca) @ cu)))
            out.append({"T": _rigid(R2, t2), "inliers": len(pr2), "n_zs": len(Z), "n_cad": len(C), "stage": sid,
                        "rms": float(np.sqrt(np.mean(np.square(res)))) if res else None})
            if len(seen) >= keep * 3:
                break
    out.sort(key=lambda s: (-s["inliers"] / max(1, min(s["n_zs"], s["n_cad"])), -s["inliers"], s["rms"] or 9))
    final = []
    for s in out:
        if any(same_pose(s["T"], f["T"], probe=probe)[0] for f in final):
            continue
        final.append(s)
        if len(final) >= keep:
            break
    return final
