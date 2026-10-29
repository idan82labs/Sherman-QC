"""Voxel downsampling, normals and robust planar region growing.

Sheet metal has two parallel skins a thickness apart. They are kept apart by
(a) oriented normals when the scan has them (the skins face opposite ways and
are never put in the same voxel or region), and (b) a local point-to-tangent-
plane test that rejects neighbours one thickness away.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SegmentParams:
    voxel_mm: float | None = None      # None = auto
    max_points: int = 200_000
    k_normals: int = 20
    k_grow: int = 12
    angle_deg: float = 8.0             # point normal vs region normal
    local_dist_mm: float = 0.3         # neighbour vs point tangent plane
    merge_angle_deg: float = 3.0
    merge_dist_mm: float = 0.5
    min_area_mm2: float = 30.0
    min_points: int = 30
    min_width_mm: float = 2.0          # drop slivers (bend-arc facets) narrower than this


@dataclass
class Segmentation:
    points: np.ndarray                 # (M, 3) picked real scan points (raw coords)
    normals: np.ndarray                # (M, 3) unit normals (oriented if `oriented`)
    curvature: np.ndarray              # (M,)
    labels: np.ndarray                 # (M,) int, 0 = unsegmented, patches 1..P
    voxel_mm: float
    oriented: bool
    patches: list[dict] = field(default_factory=list)
    thickness_mm: float | None = None


def _voxel_keys(points: np.ndarray, voxel: float, normals: np.ndarray | None) -> np.ndarray:
    grid = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    dims = grid.max(axis=0) + 1
    key = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    if normals is not None:
        dominant = np.argmax(np.abs(normals), axis=1)
        sign = (normals[np.arange(len(normals)), dominant] >= 0).astype(np.int64)
        key = key * 6 + dominant * 2 + sign
    return key


def voxel_pick(points: np.ndarray, voxel: float, normals: np.ndarray | None = None) -> np.ndarray:
    """Indices of one real point per voxel (the one nearest the voxel centre)."""
    key = _voxel_keys(points, voxel, normals)
    origin = points.min(axis=0)
    centre = (np.floor((points - origin) / voxel) + 0.5) * voxel + origin
    dist = np.einsum("ij,ij->i", points - centre, points - centre)
    order = np.lexsort((dist, key))
    sorted_key = key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    return np.sort(order[first])


def choose_voxel(points: np.ndarray, normals: np.ndarray | None, params: SegmentParams) -> float:
    if params.voxel_mm:
        return float(params.voxel_mm)
    voxel = 0.5
    for _ in range(40):
        if len(np.unique(_voxel_keys(points, voxel, normals))) <= params.max_points:
            return voxel
        voxel *= 1.2
    return voxel


def _weighted_pca(points, idx, weights):
    normals = np.empty_like(points)
    curvature = np.empty(len(points))
    step = 50_000
    for start in range(0, len(points), step):
        block = idx[start:start + step]
        w = weights[start:start + step][..., None]
        nb = points[block]
        wsum = np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
        nb = nb - (nb * w).sum(axis=1, keepdims=True) / wsum
        cov = np.einsum("nki,nkj->nij", nb * w, nb) / wsum[:, 0, :, None]
        evals, evecs = np.linalg.eigh(cov)
        normals[start:start + step] = evecs[:, :, 0]
        total = np.maximum(evals.sum(axis=1), 1e-12)
        curvature[start:start + step] = evals[:, 0] / total
    return normals, curvature


def pca_normals(points: np.ndarray, k: int, source_normals: np.ndarray | None = None,
                skin_tol_mm: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point normals robust to the second sheet skin in the neighbourhood.

    Pass 1 uses the 8 nearest points (and, when the scan has oriented normals,
    only same-facing neighbours). Pass 2 refits with the k nearest points that
    lie within ``skin_tol_mm`` of the pass-1 tangent plane.
    """
    tree = cKDTree(points)
    k = min(k, len(points))
    _, idx = tree.query(points, k=k)
    weights = np.ones(idx.shape)
    if source_normals is not None:
        agree = np.einsum("nkj,nj->nk", source_normals[idx], source_normals) > 0.3
        weights = agree.astype(float)
    first = np.zeros_like(weights)
    first[:, : min(8, k)] = weights[:, : min(8, k)]
    first[:, 0] = 1.0
    normals, _ = _weighted_pca(points, idx, first)
    offset = np.abs(np.einsum("nkj,nj->nk", points[idx] - points[:, None, :], normals))
    weights = weights * (offset <= skin_tol_mm)
    weights[:, 0] = 1.0
    normals, curvature = _weighted_pca(points, idx, weights)
    return normals, curvature, idx


def fit_plane(points: np.ndarray, iterations: int = 3, k_sigma: float = 2.5):
    """Trimmed least-squares plane. Returns (normal, centroid, rms, inlier_mask)."""
    mask = np.ones(len(points), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    centroid = points.mean(axis=0)
    for _ in range(max(1, iterations)):
        sel = points[mask]
        if len(sel) < 3:
            break
        centroid = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
        normal = vt[2]
        resid = (points - centroid) @ normal
        sigma = 1.4826 * np.median(np.abs(resid[mask] - np.median(resid[mask])))
        new_mask = np.abs(resid) <= max(k_sigma * sigma, 1e-4)
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            mask = new_mask if new_mask.sum() >= 3 else mask
            break
        mask = new_mask
    sel = points[mask]
    centroid = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
    normal = vt[2]
    rms = float(np.sqrt(np.mean(((sel - centroid) @ normal) ** 2)))
    return normal, centroid, rms, mask


def _grow(points, normals, curvature, idx, oriented, params) -> np.ndarray:
    n = len(points)
    labels = np.zeros(n, dtype=np.int64)
    cos_thr = np.cos(np.radians(params.angle_deg))
    nbrs = idx[:, 1:params.k_grow + 1]
    order = np.argsort(curvature, kind="stable")
    current = 0
    for seed in order:
        if labels[seed]:
            continue
        current += 1
        labels[seed] = current
        region_sum = normals[seed].copy()
        region_n = normals[seed].copy()
        count = 1
        queue = deque([seed])
        while queue:
            p = queue.popleft()
            cand = nbrs[p]
            cand = cand[labels[cand] == 0]
            if not len(cand):
                continue
            dots = normals[cand] @ region_n
            if not oriented:
                dots = np.abs(dots)
            local = np.abs((points[cand] - points[p]) @ normals[p])
            ok = cand[(dots >= cos_thr) & (local <= params.local_dist_mm)]
            if not len(ok):
                continue
            labels[ok] = current
            add = normals[ok]
            if not oriented:
                add = add * np.sign(add @ region_n + 1e-12)[:, None]
            region_sum += add.sum(axis=0)
            count += len(ok)
            region_n = region_sum / np.linalg.norm(region_sum)
            queue.extend(ok.tolist())
    return labels


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _groups(labels, min_size):
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    bounds = np.flatnonzero(np.diff(sorted_labels)) + 1
    for chunk in np.split(order, bounds):
        lab = int(labels[chunk[0]])
        if lab == 0 or len(chunk) < min_size:
            continue
        yield lab, chunk


def _plane_table(points, normals, labels, oriented, min_size=10):
    table = {}
    for lab, sel in _groups(labels, min_size):
        n, c, rms, _ = fit_plane(points[sel], iterations=2)
        if oriented and (normals[sel] @ n).sum() < 0:
            n = -n
        table[int(lab)] = (n, c, rms, sel)
    return table


def _adjacent_pairs(labels, idx, k):
    a = np.repeat(labels, k)
    b = labels[idx[:, 1:k + 1]].ravel()
    keep = (a != b) & (a > 0) & (b > 0)
    pairs = np.stack([np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])], axis=1)
    if not len(pairs):
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(pairs, axis=0)


def _merge(points, normals, labels, idx, oriented, params):
    cos_thr = np.cos(np.radians(params.merge_angle_deg))
    for _ in range(4):
        table = _plane_table(points, normals, labels, oriented)
        pairs = _adjacent_pairs(labels, idx, params.k_grow)
        uf = _UnionFind(int(labels.max()) + 1)
        merged = 0
        for a, b in pairs:
            if a not in table or b not in table:
                continue
            na, ca, _, sa = table[a]
            nb, cb, _, sb = table[b]
            dot = float(na @ nb)
            if not oriented:
                dot = abs(dot)
            if dot < cos_thr:
                continue
            small, (nl, cl) = (sb, (na, ca)) if len(sa) >= len(sb) else (sa, (nb, cb))
            offset = float(np.median(np.abs((points[small] - cl) @ nl)))
            if offset <= params.merge_dist_mm:
                uf.union(int(a), int(b))
                merged += 1
        if not merged:
            break
        roots = np.array([uf.find(i) for i in range(int(labels.max()) + 1)])
        labels = roots[labels]
    return labels


def _absorb(points, normals, labels, idx, oriented, params, rounds=4):
    """Give unlabelled points to a neighbouring region whose plane and normal they fit."""
    cos_thr = np.cos(np.radians(params.angle_deg))
    for _ in range(rounds):
        table = _plane_table(points, normals, labels, oriented, min_size=params.min_points)
        if not table:
            break
        free = np.flatnonzero(labels == 0)
        if not len(free):
            break
        changed = 0
        nb_labels = labels[idx[free, 1:]]
        best_lab = np.zeros(len(free), dtype=np.int64)
        best_dist = np.full(len(free), np.inf)
        for lab in np.unique(nb_labels):
            if lab == 0 or int(lab) not in table:
                continue
            n, c, _, _ = table[int(lab)]
            rows = np.flatnonzero((nb_labels == lab).any(axis=1))
            q = free[rows]
            dist = np.abs((points[q] - c) @ n)
            dots = normals[q] @ n
            if not oriented:
                dots = np.abs(dots)
            ok = (dots >= cos_thr) & (dist <= 2.0 * params.local_dist_mm) & (dist < best_dist[rows])
            best_lab[rows[ok]] = lab
            best_dist[rows[ok]] = dist[ok]
        hit = best_lab > 0
        labels[free[hit]] = best_lab[hit]
        changed = int(hit.sum())
        if not changed:
            break
    return labels


def _patch_area(points_2d: np.ndarray, cell: float) -> float:
    cells = np.unique(np.floor(points_2d / cell).astype(np.int64), axis=0)
    return float(len(cells) * cell * cell)


def in_plane_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Robust (1-99 %) length and width of a planar patch."""
    basis = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, :2]
    q = (points - points.mean(axis=0)) @ basis
    _, _, vt = np.linalg.svd(q - q.mean(axis=0), full_matrices=False)
    r = q @ vt.T
    lo, hi = np.percentile(r, [1, 99], axis=0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1])


def _find_twins(points, table, oriented, voxel):
    """Pair each patch with its other-skin partner (parallel, 0.3-12 mm away, overlapping)."""
    trees = {lab: cKDTree(points[sel]) for lab, (_, _, _, sel) in table.items()}
    twins = {}
    labs = sorted(table, key=lambda lab: -len(table[lab][3]))
    for a in labs:
        na, ca, _, sa = table[a]
        best = None
        for b in labs:
            if b == a:
                continue
            nb, cb, _, sb = table[b]
            if abs(float(na @ nb)) < np.cos(np.radians(4.0)):
                continue
            if oriented and float(na @ nb) > 0:
                continue  # the other skin faces the other way
            offset = float(np.median((points[sb] - ca) @ na))
            if not 0.3 <= abs(offset) <= 12.0:
                continue
            probe = sb if len(sb) <= 4000 else sb[np.linspace(0, len(sb) - 1, 4000).astype(int)]
            dist, _ = trees[a].query(points[probe], k=1, distance_upper_bound=abs(offset) + 3 * voxel)
            overlap = float(np.isfinite(dist).mean())
            if overlap < 0.25:
                continue
            score = overlap * min(len(sa), len(sb))
            if best is None or score > best[0]:
                best = (score, b, offset, overlap)
        if best is not None:
            twins[a] = {"twin": int(best[1]), "offset_mm": best[2], "overlap": best[3]}
    return twins


def segment(points: np.ndarray, normals: np.ndarray | None, oriented: bool,
            params: SegmentParams | None = None) -> Segmentation:
    params = params or SegmentParams()
    use_normals = normals if oriented else None
    voxel = choose_voxel(points, use_normals, params)
    pick = voxel_pick(points, voxel, use_normals)
    pts = points[pick]
    pca_n, curvature, idx = pca_normals(pts, params.k_normals,
                                        normals[pick] if oriented else None,
                                        skin_tol_mm=1.5 * params.local_dist_mm)
    if oriented:
        src = normals[pick]
        flip = np.einsum("ij,ij->i", pca_n, src) < 0
        pca_n[flip] *= -1.0
    labels = _grow(pts, pca_n, curvature, idx, oriented, params)
    labels = _merge(pts, pca_n, labels, idx, oriented, params)
    labels = _absorb(pts, pca_n, labels, idx, oriented, params)
    table = _plane_table(pts, pca_n, labels, oriented)
    keep = {}
    for lab, (n, c, rms, sel) in table.items():
        if len(sel) < params.min_points:
            continue
        basis = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2]
        area = _patch_area((pts[sel] - c) @ basis, max(voxel, 0.5))
        if area < params.min_area_mm2:
            continue
        length, width = in_plane_extent(pts[sel], n)
        if width < params.min_width_mm:
            continue
        keep[lab] = (n, c, rms, sel, area, length, width)
    ranked = sorted(keep, key=lambda lab: -keep[lab][4])
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    for new, lab in enumerate(ranked, start=1):
        remap[lab] = new
    labels = remap[labels]
    table = {int(remap[lab]): keep[lab][:4] for lab in ranked}
    areas = {int(remap[lab]): keep[lab][4] for lab in ranked}
    extents = {int(remap[lab]): keep[lab][5:7] for lab in ranked}
    twins = _find_twins(pts, table, oriented, voxel)
    patches = []
    for pid in sorted(table):
        n, c, rms, sel = table[pid]
        patch_pts = pts[sel]
        medoid = patch_pts[np.argmin(np.linalg.norm(patch_pts - patch_pts.mean(axis=0), axis=1))]
        patches.append({
            "id": pid,
            "n_points": int(len(sel)),
            "area_mm2": round(areas[pid], 1),
            "length_mm": round(extents[pid][0], 2),
            "width_mm": round(extents[pid][1], 2),
            "normal": [round(float(v), 6) for v in n],
            "centroid": [round(float(v), 4) for v in c],
            "anchor": [round(float(v), 4) for v in medoid],
            "plane_rms_mm": round(rms, 4),
            "normal_oriented": bool(oriented),
            "twin": twins.get(pid, {}).get("twin"),
            "twin_offset_mm": (round(twins[pid]["offset_mm"], 3) if pid in twins else None),
        })
    offsets = [abs(t["offset_mm"]) for t in twins.values()]
    thickness = float(np.median(offsets)) if offsets else None
    return Segmentation(points=pts, normals=pca_n, curvature=curvature, labels=labels,
                        voxel_mm=voxel, oriented=oriented, patches=patches,
                        thickness_mm=thickness)
