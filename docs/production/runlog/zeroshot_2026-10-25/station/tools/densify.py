"""Dense-scan stand-in: conformal n-subdivision of a scanned triangle mesh with lattice-breaking jitter and
scanner-card noise. Measures compute scaling only (it adds no geometric detail a denser sensor would resolve).

usage: densify.py IN.ply OUT.ply N [--seed S] [--sigma MM] [--jitter FRAC]

Method (per original triangle v0 v1 v2, lattice points (i, j), i + j <= N, weights w1 = i/N, w2 = j/N):
  * every edge gets N-1 shared points (conformal: adjacent faces use the same points, no T-junctions);
    every face gets (N-1)(N-2)/2 interior points; each face becomes N^2 triangles with the original winding;
  * original vertices (and their normals) are kept bit-exact;
  * new points are jittered inside the surface to break the regular lattice: edge points along their edge by
    U(-J, J)/N of the edge, interior points by U(-J, J)/N in both barycentric directions (J = 0.2, which can
    never invert a sub-triangle: the worst-case approach 2*0.2*sqrt(2) = 0.57 lattice units < 0.71);
  * new points are then displaced along their (barycentric-interpolated, renormalised) normal by
    N(0, sigma), sigma = scanner_model_card noise_sigma_mm (0.004564 mm), i.e. independent sensor noise;
  * output: binary little-endian PLY, float32 x y z nx ny nz, uchar/int32 faces (the Geomagic export layout).
Deterministic for a given (input, N, seed).
"""
import argparse
import sys
import time

import numpy as np


def read_ply(path):
    with open(path, "rb") as fh:
        hdr = []
        while True:
            line = fh.readline().decode("latin1").strip()
            hdr.append(line)
            if line == "end_header":
                break
        off = fh.tell()
    assert "format binary_little_endian 1.0" in hdr, hdr[:3]
    nv = nf = None
    vprops, fprop = [], None
    cur = None
    for l in hdr:
        t = l.split()
        if t[:2] == ["element", "vertex"]:
            nv, cur = int(t[2]), "v"
        elif t[:2] == ["element", "face"]:
            nf, cur = int(t[2]), "f"
        elif t[:1] == ["property"]:
            if cur == "v":
                vprops.append((t[1], t[2]))
            elif cur == "f":
                fprop = t
    tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4"}
    vdt = np.dtype([(n, tmap[ty]) for ty, n in vprops])
    assert fprop[2] in ("uchar", "uint8") and fprop[3] in ("int", "int32", "uint", "uint32"), fprop
    raw = np.memmap(path, mode="r", offset=off, dtype=np.uint8)
    vb = nv * vdt.itemsize
    V = np.frombuffer(raw[:vb].tobytes(), dtype=vdt, count=nv)
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    Fr = np.frombuffer(raw[vb:vb + nf * fdt.itemsize].tobytes(), dtype=fdt, count=nf)
    assert (Fr["n"] == 3).all()
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    nrm = np.stack([V["nx"], V["ny"], V["nz"]], 1).astype(np.float64) if "nx" in vdt.names else None
    return xyz, nrm, Fr["i"].astype(np.int64)


def write_ply(path, xyz, nrm, F):
    nv, nf = len(xyz), len(F)
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
    V = np.empty(nv, vdt)
    V["x"], V["y"], V["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    V["nx"], V["ny"], V["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    Fo = np.empty(nf, fdt)
    Fo["n"] = 3
    Fo["i"] = F
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment densify.py dense-scan stand-in\n"
           f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\n"
           f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(hdr.encode())
        fh.write(V.tobytes())
        fh.write(Fo.tobytes())


def unit(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("n", type=int)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--sigma", type=float, default=0.004564)
    ap.add_argument("--jitter", type=float, default=0.2)
    a = ap.parse_args()
    t0 = time.time()
    N, J = a.n, a.jitter
    rng = np.random.default_rng(a.seed)
    X, Nr, F = read_ply(a.inp)
    if Nr is None:
        raise SystemExit("needs vertex normals")
    nv, nf = len(X), len(F)
    # unique edges
    e_all = np.concatenate([F[:, [0, 1]], F[:, [0, 2]], F[:, [1, 2]]])  # face-edge order: 01, 02, 12
    lo, hi = np.minimum(e_all[:, 0], e_all[:, 1]), np.maximum(e_all[:, 0], e_all[:, 1])
    key = lo * nv + hi
    uk, inv = np.unique(key, return_inverse=True)
    ne = len(uk)
    ea, eb = uk // nv, uk % nv
    eid = inv.reshape(3, nf).T  # (nf, 3): edge ids of 01, 02, 12
    # edge points: k = 1..N-1 from ea to eb
    k = np.arange(1, N, dtype=np.float64)
    t = (k[None, :] + rng.uniform(-J, J, size=(ne, N - 1))) / N  # jitter along the edge
    PE = X[ea][:, None, :] * (1 - t)[..., None] + X[eb][:, None, :] * t[..., None]
    NE = Nr[ea][:, None, :] * (1 - t)[..., None] + Nr[eb][:, None, :] * t[..., None]
    PE, NE = PE.reshape(-1, 3), unit(NE.reshape(-1, 3))
    # interior points
    ij = [(i, j) for i in range(1, N) for j in range(1, N) if i + j <= N - 1]
    m = len(ij)
    if m:
        I = np.array([p[0] for p in ij], float); Jj = np.array([p[1] for p in ij], float)
        w1 = (I[None, :] + rng.uniform(-J, J, size=(nf, m))) / N
        w2 = (Jj[None, :] + rng.uniform(-J, J, size=(nf, m))) / N
        w0 = 1 - w1 - w2
        PI = (X[F[:, 0]][:, None, :] * w0[..., None] + X[F[:, 1]][:, None, :] * w1[..., None]
              + X[F[:, 2]][:, None, :] * w2[..., None]).reshape(-1, 3)
        NI = unit((Nr[F[:, 0]][:, None, :] * w0[..., None] + Nr[F[:, 1]][:, None, :] * w1[..., None]
                   + Nr[F[:, 2]][:, None, :] * w2[..., None]).reshape(-1, 3))
        del w0, w1, w2
    else:
        PI = np.zeros((0, 3)); NI = np.zeros((0, 3))
    newP = np.concatenate([PE, PI]); newN = np.concatenate([NE, NI])
    del PE, PI, NE, NI
    newP += newN * rng.normal(0.0, a.sigma, size=(len(newP), 1))  # sensor noise along the normal
    XO = np.concatenate([X, newP]); NO = np.concatenate([Nr, newN])
    del newP, newN
    base_e, base_i = nv, nv + ne * (N - 1)
    # lattice local ids
    L = {}
    for i in range(N + 1):
        for j in range(N + 1 - i):
            L[(i, j)] = len(L)
    G = np.empty((nf, len(L)), np.int64)

    def edge_pt(col_edge, p_vert, tpos):  # point at fraction tpos/N from vertex p along face edge col_edge
        e = eid[:, col_edge]
        fwd = ea[e] == F[:, p_vert]
        kk = np.where(fwd, tpos, N - tpos)
        return base_e + e * (N - 1) + (kk - 1)

    for (i, j), c in L.items():
        if (i, j) == (0, 0):
            G[:, c] = F[:, 0]
        elif (i, j) == (N, 0):
            G[:, c] = F[:, 1]
        elif (i, j) == (0, N):
            G[:, c] = F[:, 2]
        elif j == 0:
            G[:, c] = edge_pt(0, 0, i)       # v0 -> v1
        elif i == 0:
            G[:, c] = edge_pt(1, 0, j)       # v0 -> v2
        elif i + j == N:
            G[:, c] = edge_pt(2, 1, j)       # v1 -> v2
        else:
            G[:, c] = base_i + np.arange(nf) * m + ij.index((i, j))
    tris = []
    for i in range(N):
        for j in range(N - i):
            tris.append((L[(i, j)], L[(i + 1, j)], L[(i, j + 1)]))
            if i + j <= N - 2:
                tris.append((L[(i + 1, j)], L[(i + 1, j + 1)], L[(i, j + 1)]))
    T = np.array(tris)
    FO = G[:, T].reshape(-1, 3)
    del G
    assert len(FO) == nf * N * N and FO.max() < len(XO)
    write_ply(a.out, XO, NO, FO.astype(np.int32))
    print(f"{a.inp} -> {a.out}: N={N} V {nv}->{len(XO)} F {nf}->{len(FO)} edges={ne} wall={time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
