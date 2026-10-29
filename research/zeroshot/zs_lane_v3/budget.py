"""zs_lane_v3 revision 3: a density-invariant face budget for very large scans.

Why (revision 2 -> 3). Revision 2 decimated a scan above 3 000 000 faces by vertex clustering on a grid whose cell
grew with the input density (start: mean edge x sqrt(F / max_faces)). On densified copies of DEV scans its lines
drifted from the native-density lines above about 10 M faces (station/DENSE_READINESS.md: 7/7 -> 2/7 + 3 FP at 40 M,
16/16 -> 6/16 at 46 M). Two causes, both measured: (1) the cell depended on the density; (2) even at a fixed cell,
vertex clustering of a mesh much denser than the cell builds its output faces from whichever fine triangles happen
to straddle 3 cells, which leaves overlapping and missing faces around every 4-cell junction (on 40 M faces about
2 % of the faces fold over); Z3's slices then break up at the bends (support fraction 0.93 -> 0.43, the selector
drops the line). Revision 3 removes both: the output depends on the surface, not on how finely it was sampled.

Budget. ``max_faces`` triangles (frozen in config_v3.json; 3 000 000, unchanged) is the trigger, checked from the
file header (PLY ``element face``, binary STL triangle count) before anything is loaded. At or below it the scan is
used unchanged (bit-identical to revisions 1 and 2). Point clouds (no faces) are not budgeted.

Resampling above the trigger (no part-specific input; only the mesh):
  * **fixed absolute grid**: cells of size h with corners at integer multiples of h in the scan frame (no origin
    taken from the data). h is the first value of the ladder ``cell_mm`` x 1.12^k (cell_mm = 0.30 mm, frozen) whose
    output has at most ``max_faces`` faces (2 per grid-edge crossing, a count fixed by the surface, so the chosen h
    does not depend on the density either);
  * **crossings**: every point where a scan triangle crosses a grid line (a line parallel to one axis through grid
    corners), with that triangle's unit normal (from its winding) and, when the scan has vertex normals, the normal
    interpolated at the point. A crossing lies on one grid edge, shared by 4 cells;
  * **skins** (the two skins of a sheet never merge): within a cell, a crossing belongs to the other skin when its
    normal makes more than 120 deg with the cell's reference normal (dot < -0.5; the reference is the normal of the
    cell's first crossing in the canonical crossing order), otherwise to the reference skin. One output vertex per
    (cell, skin). A bend apex (normals up to 90 deg apart, also a sharp corner) stays one skin; opposite skins
    (about 180 deg) are always split;
  * **vertex placement (bend edges preserved)**: each (cell, skin) vertex minimises the sum of squared distances to
    the planes of its crossings (point + triangle normal; the dual-contouring QEF), regularised towards the mean
    crossing point (weight 0.05 x trace / 3, as revision 2) and replaced by that mean if it lands more than h from
    it; flats stay on their plane and a bend's two planes meet at their intersection;
  * **faces**: one quad per crossing, joining the 4 (cell, skin) vertices around its grid edge, oriented by the
    sign of the triangle normal along the edge axis (the scan's winding is kept) and split along its shorter
    diagonal; faces repeated by several crossings of one edge are kept once (first in canonical order).
    Every crossing is ordered canonically (axis, grid line, edge, point, normal), so the output does not depend on
    the scan's vertex or face order.
The resampled mesh is written as a binary PLY into a private temporary directory and every lane stage (Z3, Z2, the
snap and the evidence) runs on it. Output lines stay in the raw scan frame (resampling does not move the frame).
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

MAX_FACES = 3_000_000     # the frozen trigger and output ceiling (config_v3.json "budget"); unchanged from rev 2
CELL_MM = 0.30            # revision 3: the fixed grid cell (config_v3.json "budget"."cell_mm")
CHUNK = 2_000_000          # faces per chunk (bounds the temporary arrays)
LADDER = 1.12              # cell ladder ratio above the ceiling
MAX_STEPS = 24
SKIN_COS = -0.5            # a crossing more than 120 deg from the cell's reference normal is on the other skin
REG = 0.05                 # QEF regularisation (fraction of trace / 3) towards the mean crossing point
PERM = {0: (1, 2), 1: (2, 0), 2: (0, 1)}                         # (b, c) for the grid lines parallel to axis a
QUAD = ((-1, -1), (0, -1), (0, 0), (-1, 0))                      # cells around an edge, counter-clockwise from +a


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


def crossings(V: np.ndarray, F: np.ndarray, h: float, file_normals: np.ndarray | None = None,
              count_only: bool = False) -> dict | int:
    """Every crossing of a scan triangle with a grid line, in canonical order (``count_only``: just their number).

    Returns axis ``a``, grid line ``B``, ``C`` (integer coordinates along the two other axes), edge index ``G``
    along ``a``, point ``P``, unit triangle normal ``N`` and (optional) interpolated file normal ``FN``."""
    g = np.floor(V / h).astype(np.int64)
    parts = []
    total = 0
    for a in (0, 1, 2):
        b, c = PERM[a]
        for s in range(0, len(F), CHUNK):
            f = F[s:s + CHUNK]
            gb, gc = g[f, b], g[f, c]
            lo_b, hi_b = gb.min(1), gb.max(1)
            lo_c, hi_c = gc.min(1), gc.max(1)
            cand = np.nonzero((hi_b > lo_b) & (hi_c > lo_c))[0]   # the projection can contain a grid line
            if not len(cand):
                continue
            fc = f[cand]
            p0, p1, p2 = V[fc[:, 0]], V[fc[:, 1]], V[fc[:, 2]]
            lb, lc = lo_b[cand], lo_c[cand]
            sb, sc = hi_b[cand] - lb, hi_c[cand] - lc
            for ob in range(1, int(sb.max()) + 1):
                for oc in range(1, int(sc.max()) + 1):
                    idx = np.nonzero((sb >= ob) & (sc >= oc))[0]
                    if not len(idx):
                        continue
                    Bv, Cv = lb[idx] + ob, lc[idx] + oc
                    qb, qc = Bv * h, Cv * h
                    x0b, x0c = p0[idx, b], p0[idx, c]
                    e1b, e1c = p1[idx, b] - x0b, p1[idx, c] - x0c
                    e2b, e2c = p2[idx, b] - x0b, p2[idx, c] - x0c
                    den = e1b * e2c - e2b * e1c
                    ok = np.abs(den) > 1e-18                       # a triangle parallel to the line never crosses it
                    den = np.where(ok, den, 1.0)
                    l1 = ((qb - x0b) * e2c - e2b * (qc - x0c)) / den
                    l2 = (e1b * (qc - x0c) - (qb - x0b) * e1c) / den
                    l0 = 1.0 - l1 - l2
                    ok &= (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                    if not ok.any():
                        continue
                    if count_only:
                        total += int(ok.sum())
                        continue
                    k = idx[ok]
                    L0, L1, L2 = l0[ok, None], l1[ok, None], l2[ok, None]
                    P = L0 * p0[k] + L1 * p1[k] + L2 * p2[k]
                    P[:, b], P[:, c] = qb[ok], qc[ok]
                    cr = np.cross(p1[k] - p0[k], p2[k] - p0[k])
                    N = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-30)[:, None]
                    FN = None
                    if file_normals is not None:
                        fk = fc[k]
                        FN = L0 * file_normals[fk[:, 0]] + L1 * file_normals[fk[:, 1]] + L2 * file_normals[fk[:, 2]]
                    parts.append((np.full(len(k), a, np.int64), Bv[ok], Cv[ok],
                                  np.floor(P[:, a] / h).astype(np.int64), P, N, FN))
    if count_only:
        return total
    if not parts:
        z = np.zeros(0, np.int64)
        return {"a": z, "B": z, "C": z, "G": z, "P": np.zeros((0, 3)), "N": np.zeros((0, 3)),
                "FN": None if file_normals is None else np.zeros((0, 3))}
    cat = lambda j: np.concatenate([p[j] for p in parts])  # noqa: E731
    x = {"a": cat(0), "B": cat(1), "C": cat(2), "G": cat(3), "P": cat(4), "N": cat(5),
         "FN": cat(6) if file_normals is not None else None}
    o = np.lexsort((x["N"][:, 2], x["N"][:, 1], x["N"][:, 0], x["P"][:, 2], x["P"][:, 1], x["P"][:, 0],
                    x["G"], x["C"], x["B"], x["a"]))
    return {k: (None if v is None else v[o]) for k, v in x.items()}


def resample(cx: dict, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """The (cell, skin) vertices and the quads of a crossing set (module docstring)."""
    a, P, N = cx["a"], cx["P"], cx["N"]
    n = len(a)
    cells = np.empty((n, 4, 3), np.int64)
    for ax in (0, 1, 2):
        b, c = PERM[ax]
        m = a == ax
        for j, (db, dc) in enumerate(QUAD):
            cells[m, j, ax] = cx["G"][m]
            cells[m, j, b] = cx["B"][m] + db
            cells[m, j, c] = cx["C"][m] + dc
    rel = cells.reshape(-1, 3)
    rel = rel - rel.min(0)
    dim = rel.max(0) + 1
    ckey = (rel[:, 0] * dim[1] + rel[:, 1]) * dim[2] + rel[:, 2]
    del rel, cells
    Nr = np.repeat(N, 4, axis=0)
    _u, first, cinv = np.unique(ckey, return_index=True, return_inverse=True)
    skin = (np.einsum("ij,ij->i", Nr, Nr[first][cinv]) < SKIN_COS).astype(np.int64)
    del first, cinv
    _uv, vinv = np.unique(ckey * 2 + skin, return_inverse=True)
    del ckey, skin
    C = len(_uv)
    Pr = np.repeat(P, 4, axis=0)
    cnt = np.bincount(vinv, minlength=C).astype(np.float64)
    mean = np.stack([np.bincount(vinv, weights=Pr[:, k], minlength=C) for k in range(3)], 1) / cnt[:, None]
    d = np.einsum("ij,ij->i", Nr, Pr)
    M = np.empty((C, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            M[:, i, j] = np.bincount(vinv, weights=Nr[:, i] * Nr[:, j], minlength=C)
            M[:, j, i] = M[:, i, j]
    rhs = np.stack([np.bincount(vinv, weights=Nr[:, k] * d, minlength=C) for k in range(3)], 1)
    del Pr, d
    lam = REG * (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]) / 3.0 + 1e-12
    rhs += lam[:, None] * mean
    for i in range(3):
        M[:, i, i] += lam
    X = np.linalg.solve(M, rhs[..., None])[..., 0]
    far = np.linalg.norm(X - mean, axis=1) > h
    X[far] = mean[far]
    No = None
    if cx.get("FN") is not None:
        FNr = np.repeat(cx["FN"], 4, axis=0)
        No = np.stack([np.bincount(vinv, weights=FNr[:, k], minlength=C) for k in range(3)], 1)
        No /= np.maximum(np.linalg.norm(No, axis=1), 1e-30)[:, None]
        del FNr
    del Nr
    q = vinv.reshape(n, 4)
    flip = N[np.arange(n), a] < 0                                      # winding: the triangle normal's side
    q[flip] = q[flip][:, ::-1]
    d02 = np.linalg.norm(X[q[:, 0]] - X[q[:, 2]], axis=1)
    d13 = np.linalg.norm(X[q[:, 1]] - X[q[:, 3]], axis=1)
    use02 = d02 <= d13
    T = np.empty((2 * n, 3), np.int64)
    T[0::2][use02], T[1::2][use02] = q[use02][:, [0, 1, 2]], q[use02][:, [0, 2, 3]]
    T[0::2][~use02], T[1::2][~use02] = q[~use02][:, [0, 1, 3]], q[~use02][:, [1, 2, 3]]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    s = np.sort(T, axis=1)
    o = np.lexsort((s[:, 2], s[:, 1], s[:, 0]))                       # stable: the first of equal rows keeps its rank
    ss = s[o]
    new = np.ones(len(ss), bool)
    new[1:] = (ss[1:] != ss[:-1]).any(axis=1)
    T = T[np.sort(o[new])]
    info = {"crossings": int(n), "vertices_out": int(C), "faces_out": int(len(T)),
            "clamped_frac": round(float(far.mean()), 5) if C else 0.0}
    return X, T, No, info


def decimate(V: np.ndarray, F: np.ndarray, max_faces: int, file_normals: np.ndarray | None = None,
             cell_mm: float = CELL_MM) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Resample to at most ``max_faces`` faces on the fixed grid (module docstring)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F)
    if F.dtype != np.int32:
        F = F.astype(np.int32 if len(V) < 2 ** 31 else np.int64)
    fnm = None if file_normals is None or len(file_normals) != len(V) else np.asarray(file_normals, np.float64)
    # the smallest k with 2 x crossings(cell_mm x 1.12^k) <= max_faces; the crossing count scales about as 1 / h^2,
    # so k is first predicted from the count at cell_mm, then corrected up / down with exact counts
    counts: dict[int, int] = {}

    def count(k: int) -> int:
        if k not in counts:
            counts[k] = crossings(V, F, float(cell_mm) * LADDER ** k, count_only=True)
        return counts[k]
    n0 = count(0)
    k = 0 if 2 * n0 <= max_faces else min(MAX_STEPS, int(np.ceil(np.log(2 * n0 / max_faces) / (2 * np.log(LADDER)))))
    while k < MAX_STEPS and 2 * count(k) > max_faces:
        k += 1
    while k > 0 and 2 * count(k - 1) <= max_faces:
        k -= 1
    h = float(cell_mm) * LADDER ** k
    cx = crossings(V, F, h, fnm)
    steps = [[round(float(cell_mm) * LADDER ** j, 5), counts[j], 2 * counts[j]] for j in sorted(counts)]
    X, T, No, info = resample(cx, h)
    info = {"faces_in": int(len(F)), "vertices_in": int(len(V)), "faces_out": info["faces_out"],
            "vertices_out": info["vertices_out"], "cell_mm": round(h, 5), "crossings": info["crossings"],
            "ladder": steps, "clamped_frac": info["clamped_frac"]}
    return X, T, No, info


def budget_scan(path: str, max_faces: int | None, workdir: str, cell_mm: float | None = None) -> tuple[str, dict]:
    """``path`` itself when within the budget (or unbudgeted), else a resampled binary PLY in ``workdir``."""
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
    X, Fo, No, info = decimate(V, F, int(max_faces), fnorm, CELL_MM if cell_mm is None else float(cell_mm))
    del V, F, fnorm, m
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(X), o3d.utility.Vector3iVector(Fo.astype(np.int32)))
    if No is not None:
        out.vertex_normals = o3d.utility.Vector3dVector(No)
    dst = str(Path(workdir) / ("scan" + ".ply"))
    if not o3d.io.write_triangle_mesh(dst, out, write_ascii=False, write_vertex_normals=No is not None):
        raise RuntimeError("could not write the resampled scan")
    info.update(applied=True, max_faces=int(max_faces), method="fixed_grid_dual_contouring", wall=round(time.time() - t0, 3))
    return dst, info
