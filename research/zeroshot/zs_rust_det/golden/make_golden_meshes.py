"""Synthetic golden meshes for `zs_rust selftest --golden` (no scan data): extruded sheet-metal profiles with
known bends, written with a plain numpy writer (no open3d) in the reader paths the station sees: binary PLY (float32 and
float64), ASCII PLY and binary STL. Deterministic: numpy's PCG64 with fixed seeds; the files are committed with
their sha256 in golden.json, so this script only documents how they were made.
usage: .venv/bin/python -B research/zeroshot/zs_rust/golden/make_golden_meshes.py OUT_DIR"""
import math, struct, sys
from pathlib import Path
import numpy as np


def profile(segments, t, ri, step):
    p = np.zeros(2); ang = 0.0; mids, nrms = [], []; rm = ri + t / 2
    for kind, val in segments:
        if kind == "L":
            n = max(2, int(val / step))
            for s in np.linspace(0, val, n, endpoint=False):
                d = np.array([math.cos(ang), math.sin(ang)])
                mids.append(p + s * d); nrms.append(np.array([-d[1], d[0]]))
            p = p + val * np.array([math.cos(ang), math.sin(ang)])
        else:
            b = math.radians(val); sgn = 1 if b > 0 else -1
            d = np.array([math.cos(ang), math.sin(ang)])
            c = p + sgn * rm * np.array([-d[1], d[0]])
            n = max(4, int(abs(b) * rm / step))
            for k in range(n):
                a = ang + b * k / n
                mids.append(c - sgn * rm * np.array([-math.sin(a), math.cos(a)])); nrms.append(np.array([-math.sin(a), math.cos(a)]))
            ang += b
            p = c - sgn * rm * np.array([-math.sin(ang), math.cos(ang)])
    mids.append(p); d = np.array([math.cos(ang), math.sin(ang)]); nrms.append(np.array([-d[1], d[0]]))
    M, N = np.array(mids), np.array(nrms)
    return M + N * t / 2, M - N * t / 2


def extrude(segments, t, ri, z1, step, zstep, noise, seed, rot_deg=0.0, shift=(0, 0, 0)):
    up, lo = profile(segments, t, ri, step)
    loop = np.vstack([up, lo[::-1]])
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    loop = loop[::-1] if area < 0 else loop
    zs = np.arange(0, z1 + 1e-9, zstep); n = len(loop)
    V = np.array([[x, y, z] for z in zs for (x, y) in loop], float)
    F = []
    for k in range(len(zs) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
    F = np.array(F, np.int64)
    if noise:
        V = V + np.random.default_rng(seed).normal(scale=noise, size=V.shape)
    if rot_deg:
        r = math.radians(rot_deg); c, s = math.cos(r), math.sin(r)
        R = np.array([[c, -s, 0], [s * 0.8, c * 0.8, 0.6], [-s * 0.6, -c * 0.6, 0.8]])
        V = V @ R.T
    V = V + np.array(shift, float)
    fc = V[F].mean(1); cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.mean(np.sum(cr * (fc - V.mean(0)), axis=1)) < 0:
        F = F[:, ::-1].copy()
    return V.astype(np.float32), F


def ply_binary(path, V, F, typ="float"):
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty {typ} x\nproperty {typ} y\n"
            f"property {typ} z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode()
    fb = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)]); fb["n"] = 3; fb["i"] = F
    Path(path).write_bytes(head + V.astype("<f4" if typ == "float" else "<f8").tobytes() + fb.tobytes())


def ply_ascii(path, V, F):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(V)}", "property float x", "property float y",
             "property float z", f"element face {len(F)}", "property list uchar int vertex_indices", "end_header"]
    lines += [f"{float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in V.astype(np.float64)]
    lines += [f"3 {a} {b} {c}" for a, b, c in F]
    Path(path).write_text("\n".join(lines) + "\n")


def stl_binary(path, V, F):
    T = V[F].astype("<f4")
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]).astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n.astype("<f4"); rec["v"] = T
    Path(path).write_bytes(b"zs_rust golden synthetic".ljust(80, b" ") + struct.pack("<I", len(F)) + rec.tobytes())


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
V, F = extrude([("L", 25), ("B", 90), ("L", 40), ("B", 90), ("L", 25)], 2.0, 2.0, 40, 0.5, 1.0, 0.01, 1)
ply_binary(out / "g1_channel.ply", V, F)
V, F = extrude([("L", 22), ("B", -60), ("L", 30), ("B", 120), ("L", 24)], 1.5, 3.0, 36, 0.5, 1.0, 0.02, 2, rot_deg=27.0, shift=(120.0, -40.0, 15.0))
ply_binary(out / "g2_zprofile_rot.ply", V, F, "double")
V, F = extrude([("L", 18), ("B", 90), ("L", 22), ("B", -90), ("L", 26), ("B", -90), ("L", 22), ("B", 90), ("L", 18)], 3.0, 1.5, 30, 0.6, 1.2, 0.015, 3)
ply_ascii(out / "g3_hat.ply", V, F)
# STL is read as stored (open3d's reader: no vertex join), i.e. a triangle soup; small, for the reader path
V, F = extrude([("L", 12), ("B", 90), ("L", 15), ("B", 90), ("L", 12)], 2.0, 2.0, 12, 0.6, 1.2, 0.01, 4)
stl_binary(out / "g4_channel_soup.stl", V, F)
for p in sorted(out.iterdir()):
    print(p.name, p.stat().st_size)
