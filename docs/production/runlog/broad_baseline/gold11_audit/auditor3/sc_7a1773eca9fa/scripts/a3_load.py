"""Auditor 3 loader: parse the raw Geomagic PLY (vertices+normals+faces) into a cached npz."""
import os
import sys

import numpy as np

SCAN = "/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.ply"
CACHE = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_scan.npz"


def load(path=SCAN, cache=CACHE):
    if os.path.exists(cache):
        z = np.load(cache)
        return z["v"], z["n"], z["f"]
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.index(b"end_header") + len(b"end_header")
    while raw[end:end + 1] in (b"\r", b"\n"):
        end += 1
        if raw[end - 1:end] == b"\n":
            break
    header = raw[:end].decode("ascii", "replace")
    nv = int([l for l in header.splitlines() if l.startswith("element vertex")][0].split()[-1])
    nf = int([l for l in header.splitlines() if l.startswith("element face")][0].split()[-1])
    vd = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
    vert = np.frombuffer(raw, dtype=vd, count=nv, offset=end)
    off = end + nv * vd.itemsize
    fd = np.dtype([("c", "u1"), ("i", "<i4", (3,))])
    faces = np.frombuffer(raw, dtype=fd, count=nf, offset=off)
    assert (faces["c"] == 3).all(), "non-triangle faces"
    assert off + nf * fd.itemsize == len(raw), (off + nf * fd.itemsize, len(raw))
    v = np.stack([vert["x"], vert["y"], vert["z"]], 1).astype(np.float64)
    n = np.stack([vert["nx"], vert["ny"], vert["nz"]], 1).astype(np.float64)
    f = faces["i"].astype(np.int64)
    np.savez(cache, v=v, n=n, f=f)
    return v, n, f


if __name__ == "__main__":
    v, n, f = load()
    print("verts", v.shape, "faces", f.shape)
    print("bbox min", v.min(0), "max", v.max(0))
    nn = np.linalg.norm(n, axis=1)
    print("normal norms", nn.min(), nn.max())
    # Orientation sanity: face normals from winding vs vertex normals
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vn_mean = n[f].mean(1)
    agree = np.einsum("ij,ij->i", fn, vn_mean)
    print("winding vs vertex-normal agreement: frac>0", float((agree > 0).mean()))
    c0 = v.mean(0)
    print("centroid", c0)
    w, U = np.linalg.eigh(np.cov((v - c0).T))
    print("pca eigvals", w, "\naxes\n", U)
