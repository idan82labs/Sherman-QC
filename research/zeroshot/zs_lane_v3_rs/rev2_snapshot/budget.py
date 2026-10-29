"""zs_lane_v3 revision 2: a face budget for very large scans (feature-preserving, memory-light decimation).

Why. Peak RSS and wall time of the lane grow with the mesh size (DEV: about 1.1 GiB per million vertices at peak,
2.74 GiB at 2.4 M vertices with Z2 run before Z3). Customer scans on disk reach 5.6 M vertices, which would exceed
the runner's 4 GiB zs limit (the step is killed: no overlay). Above the budget the lane therefore runs on a
decimated copy of the scan; at or below it the scan is used unchanged (bit-identical to v3).

Budget. ``max_faces`` triangles (frozen in config_v3.json; 3 000 000, about 1.5 M vertices). It is checked from the
file header (PLY ``element face``, binary STL triangle count) before anything is loaded. Point clouds (no faces) are
not budgeted: Z3 and Z2 already subsample them.

Decimation (no part-specific input; only the mesh):
  * **vertex clustering on a uniform grid of cell size h**, h chosen deterministically as the smallest value on a
    geometric ladder (start: mean edge length x sqrt(F / max_faces)) whose clustered mesh is within the budget;
  * **normal-aware cells**: the cluster key also contains the vertex normal's dominant signed axis (6 bins), so the
    two skins of a sheet (opposite normals) are never merged, whatever the thickness;
  * **quadric placement** (Lindstrom 2000): each cluster's vertex minimises the area-weighted sum of squared
    distances to the planes of its incident faces, regularised towards the cluster mean and clamped to the cell, so
    flats stay on their plane and sharp bend edges stay on the plane intersection;
  * faces are re-indexed; degenerate (repeated cluster) and duplicate faces are dropped; per-cluster file normals
    (mean, re-normalised) are kept when the scan has them.
The decimated mesh is written as a binary PLY into a private temporary directory and every lane stage (Z3, Z2, the
snap and the evidence) runs on it. Output lines stay in the raw scan frame (decimation does not move the frame).
"""
from __future__ import annotations

import math
import struct
import time
from pathlib import Path

import numpy as np

MAX_FACES = 3_000_000     # the frozen budget (config_v3.json "budget"); about 1.5 M vertices
CHUNK = 2_000_000          # faces per accumulation chunk (bounds the temporary arrays)
LADDER = 1.12              # cell-size ladder ratio
MAX_STEPS = 12
REG = 0.05                 # quadric regularisation (fraction of trace / 3) towards the cluster mean


def face_count(path: str) -> int | None:
    """Triangle count from the file header (None: unknown / no faces / ASCII STL)."""
    p = Path(path)
    with open(p, "rb") as f:
        if p.suffix.lower() == ".stl":
            head = f.read(84)
            if len(head) < 84:
                return None
            n = struct.unpack("<I", head[80:84])[0]
            if 84 + 50 * n == p.stat().st_size:
                return int(n)
            return None
        if f.read(3) != b"ply":
            return None
        f.seek(0)
        nf = 0
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            s = line.decode("latin1").strip()
            if s.startswith("element face"):
                nf = int(s.split()[2])
            if s == "end_header":
                break
        return nf


def _normals(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted vertex normals (from the face winding) and the per-face doubled-area cross products."""
    vn = np.zeros_like(V)
    for s in range(0, len(F), CHUNK):
        f = F[s:s + CHUNK]
        cr = np.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]])
        for k in range(3):
            for ax in range(3):
                vn[:, ax] += np.bincount(f[:, k], weights=cr[:, ax], minlength=len(V))
    n = np.linalg.norm(vn, axis=1)
    vn /= np.maximum(n, 1e-30)[:, None]
    return vn, n > 0


def _keys(V: np.ndarray, nbin: np.ndarray, h: float) -> np.ndarray:
    """Grid cell x normal bin key per vertex (one axis at a time: V-sized temporaries only)."""
    lo = V.min(0)
    g = [np.floor((V[:, k] - lo[k]) / h).astype(np.int64) for k in range(3)]
    m = [int(x.max()) + 1 for x in g]
    key = g[0] * m[1]
    key += g[1]
    del g[:2]
    key *= m[2]
    key += g[0]
    key *= 6
    key += nbin
    return key


def _clustered_faces(F: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Faces re-indexed to clusters, degenerate ones dropped (chunked, so only the surviving rows are held), then
    duplicates dropped keeping the first occurrence in face order."""
    parts = []
    for s in range(0, len(F), CHUNK):
        G = inv[F[s:s + CHUNK]]
        ok = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])
        parts.append(G[ok])
    G = np.concatenate(parts) if parts else np.zeros((0, 3), inv.dtype)
    del parts
    srt = np.sort(G, axis=1)
    o = np.lexsort((srt[:, 2], srt[:, 1], srt[:, 0]))         # stable: the first of equal rows keeps its rank
    s = srt[o]
    new = np.ones(len(s), bool)
    new[1:] = (s[1:] != s[:-1]).any(axis=1)
    return G[np.sort(o[new])]


def decimate(V: np.ndarray, F: np.ndarray, max_faces: int, file_normals: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Normal-aware quadric vertex clustering to at most ``max_faces`` faces (module docstring)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F)
    if F.dtype != np.int32:
        F = F.astype(np.int32 if len(V) < 2 ** 31 else np.int64)
    vn, has = _normals(V, F)
    dom = np.argmax(np.abs(vn), axis=1)
    nbin = dom * 2 + (vn[np.arange(len(V)), dom] < 0)
    nbin[~has] = 0
    e = V[F[: min(len(F), 200_000), 1]] - V[F[: min(len(F), 200_000), 0]]
    edge = float(np.mean(np.linalg.norm(e, axis=1)))
    h = edge * math.sqrt(max(1.0, len(F) / max_faces))
    steps = []
    for _ in range(MAX_STEPS):
        keys = _keys(V, nbin, h)
        uk, inv = np.unique(keys, return_inverse=True)
        del keys
        inv = inv.astype(F.dtype)
        G = _clustered_faces(F, inv)
        steps.append([round(h, 5), len(uk), len(G)])
        if len(G) <= max_faces:
            break
        h *= LADDER
    C = len(uk)
    cnt = np.bincount(inv, minlength=C).astype(np.float64)
    mean = np.stack([np.bincount(inv, weights=V[:, ax], minlength=C) for ax in range(3)], 1) / cnt[:, None]
    # quadrics: A = sum w n n^T, b = sum w n (n . p), over the faces incident to each cluster's vertices
    A = np.zeros((C, 6))
    b = np.zeros((C, 3))
    for s in range(0, len(F), CHUNK):
        f = F[s:s + CHUNK]
        cr = np.cross(V[f[:, 1]] - V[f[:, 0]], V[f[:, 2]] - V[f[:, 0]])
        w = np.linalg.norm(cr, axis=1)
        n = cr / np.maximum(w, 1e-30)[:, None]
        d = np.einsum("ij,ij->i", n, V[f[:, 0]])
        terms = [w * n[:, 0] * n[:, 0], w * n[:, 0] * n[:, 1], w * n[:, 0] * n[:, 2],
                 w * n[:, 1] * n[:, 1], w * n[:, 1] * n[:, 2], w * n[:, 2] * n[:, 2]]
        bt = [w * n[:, ax] * d for ax in range(3)]
        for k in range(3):
            c = inv[f[:, k]]
            for j in range(6):
                A[:, j] += np.bincount(c, weights=terms[j], minlength=C)
            for ax in range(3):
                b[:, ax] += np.bincount(c, weights=bt[ax], minlength=C)
    M = np.empty((C, 3, 3))
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = A[:, 0], A[:, 1], A[:, 2]
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = A[:, 1], A[:, 3], A[:, 4]
    M[:, 2, 0], M[:, 2, 1], M[:, 2, 2] = A[:, 2], A[:, 4], A[:, 5]
    lam = REG * (A[:, 0] + A[:, 3] + A[:, 5]) / 3.0 + 1e-12
    rhs = b + lam[:, None] * mean
    M[:, 0, 0] += lam
    M[:, 1, 1] += lam
    M[:, 2, 2] += lam
    X = np.linalg.solve(M, rhs[..., None])[..., 0]
    far = np.linalg.norm(X - mean, axis=1) > h          # clamp to the cell: fall back to the mean
    X[far] = mean[far]
    Fo = G                                                  # the last ladder step's faces
    No = None
    if file_normals is not None and len(file_normals) == len(V):
        fnm = np.asarray(file_normals, np.float64)
        No = np.stack([np.bincount(inv, weights=fnm[:, ax], minlength=C) for ax in range(3)], 1)
        No /= np.maximum(np.linalg.norm(No, axis=1), 1e-30)[:, None]
    info = {"faces_in": int(len(F)), "vertices_in": int(len(V)), "faces_out": int(len(Fo)), "vertices_out": int(C),
            "cell_mm": round(float(h), 5), "mean_edge_mm": round(edge, 5), "ladder": steps,
            "clamped_frac": round(float(far.mean()), 5)}
    return X, Fo, No, info


def budget_scan(path: str, max_faces: int | None, workdir: str) -> tuple[str, dict]:
    """``path`` itself when within the budget (or unbudgeted), else a decimated binary PLY in ``workdir``."""
    if not max_faces:
        return path, {"applied": False, "reason": "NO_BUDGET"}
    nf = face_count(path)
    if nf is not None and nf <= max_faces:
        return path, {"applied": False, "faces": nf, "max_faces": int(max_faces)}
    import open3d as o3d
    t0 = time.time()
    m = o3d.io.read_triangle_mesh(str(path))
    F = np.asarray(m.triangles)                                         # int32 views of the o3d buffers
    if len(F) <= max_faces:              # point cloud or header unknown but small
        return path, {"applied": False, "faces": int(len(F)), "max_faces": int(max_faces)}
    V = np.asarray(m.vertices, dtype=np.float64)
    fnorm = np.asarray(m.vertex_normals, dtype=np.float64) if m.has_vertex_normals() else None
    X, Fo, No, info = decimate(V, F, int(max_faces), fnorm)
    del V, F, fnorm, m
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(X), o3d.utility.Vector3iVector(Fo.astype(np.int32)))
    if No is not None:
        out.vertex_normals = o3d.utility.Vector3dVector(No)
    dst = str(Path(workdir) / ("scan" + ".ply"))
    if not o3d.io.write_triangle_mesh(dst, out, write_ascii=False, write_vertex_normals=No is not None):
        raise RuntimeError("could not write the decimated scan")
    info.update(applied=True, max_faces=int(max_faces), wall=round(time.time() - t0, 3))
    return dst, info
