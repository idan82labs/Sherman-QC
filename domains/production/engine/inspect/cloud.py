"""Point-cloud preparation (E3 step 1): voxel cloud, normals, neighbour graph, spacing.

Two clouds are kept (plan §3.2 step 1):

- the **full-resolution** cloud, KD-tree indexed, used for every measurement;
- a **voxel cloud** (centroid per occupied voxel, 0.8-2.5 mm adaptive) with
  PCA normals, a planarity measure and a radius neighbour graph, used for
  registration and segmentation. ``inverse[i]`` maps full-resolution point
  ``i`` to its voxel, so voxel-level labels propagate to every point.

All functions are deterministic (sorted voxel keys, fixed sampling seeds) and
single-threaded (``workers=1``), so results do not depend on the machine's core
count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = ["VoxelCloud", "voxelize", "estimate_normals", "median_spacing", "neighbour_graph", "components_of",
           "adaptive_voxel", "gross_inliers"]


def gross_inliers(points: np.ndarray, trim_percent: float, factor: float) -> np.ndarray:
    """Mask of the points within ``factor`` robust extents of the trimmed bounding box's centre.

    The robust extent is the diagonal of the per-axis ``[trim, 100 - trim]``
    percentile box (at least 1 mm). A scanner's sentinel or overflow vertex
    (finite, but metres or 1e38 away) is dropped at ingest exactly like a
    non-finite vertex, so the voxel grid stays bounded; everything that is
    plausibly part of the scan (even a stray fixture far beside the part) is kept.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    lo = np.percentile(p, trim_percent, axis=0)
    hi = np.percentile(p, 100.0 - trim_percent, axis=0)
    centre = 0.5 * (lo + hi)
    reach = factor * max(float(np.linalg.norm(hi - lo)), 1.0)
    return np.linalg.norm(p - centre, axis=1) <= reach


@dataclass
class VoxelCloud:
    points: np.ndarray  # (M, 3) voxel centroids
    inverse: np.ndarray  # (N,) full-resolution point -> voxel index
    counts: np.ndarray  # (M,) full-resolution points per voxel
    voxel: float
    normals: np.ndarray | None = None  # (M, 3), unoriented
    curvature: np.ndarray | None = None  # (M,) lambda_min / sum
    #: (M,) standard deviation of the neighbourhood along its normal (mm). A thin
    #: sheet scanned on both skins reads about t/2 here, a single skin ~ noise.
    spread: np.ndarray | None = None
    #: True when the sheet is thin against the PCA neighbourhood (both skins share it)
    slab: bool = False
    tree: cKDTree | None = None
    graph: sparse.csr_matrix | None = None


def voxelize(points: np.ndarray, voxel: float) -> VoxelCloud:
    """Centroid per occupied voxel (keys sorted, so the order is deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    lo = p.min(axis=0)
    q = np.floor((p - lo) / voxel).astype(np.int64)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    m = len(uniq)
    cent = np.empty((m, 3))
    for k in range(3):
        cent[:, k] = np.bincount(inverse, weights=p[:, k], minlength=m) / counts
    return VoxelCloud(points=cent, inverse=inverse.astype(np.int64), counts=counts.astype(np.int64), voxel=float(voxel))


def _pca(pts: np.ndarray, tree: cKDTree, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kk = int(min(k, len(pts)))
    _, idx = tree.query(pts, k=kk, workers=1)
    if kk == 1:
        idx = idx[:, None]
    normals = np.empty((len(pts), 3))
    curv = np.empty(len(pts))
    spread = np.empty(len(pts))
    step = 200_000
    for s in range(0, len(pts), step):
        nb = pts[idx[s:s + step]]
        c = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", c, c) / kk
        vals, vecs = np.linalg.eigh(cov)
        normals[s:s + step] = vecs[:, :, 0]
        tot = vals.sum(axis=1)
        curv[s:s + step] = np.where(tot > 1e-18, vals[:, 0] / np.maximum(tot, 1e-18), 1.0)
        spread[s:s + step] = np.sqrt(np.maximum(vals[:, 0], 0.0))
    return normals, curv, spread


def estimate_normals(cloud: VoxelCloud, k: int = 14, normal_voxel: float | None = None) -> None:
    """PCA normals, curvature and spread on ``cloud`` (in place); builds its KD-tree.

    For thin sheets scanned on both skins a neighbourhood must be much wider
    than the thickness, or the two skins make it nearly isotropic. With
    ``normal_voxel`` larger than the cloud's voxel, the PCA runs on a coarser
    voxel cloud (where the sheet reads as one slab) and every point takes the
    normal of its nearest coarse point.
    """
    pts = cloud.points
    cloud.tree = cKDTree(pts, balanced_tree=False, compact_nodes=False)
    if normal_voxel is None or normal_voxel <= cloud.voxel * 1.01:
        cloud.normals, cloud.curvature, cloud.spread = _pca(pts, cloud.tree, k)
        return
    coarse = voxelize(pts, normal_voxel)
    ctree = cKDTree(coarse.points, balanced_tree=False, compact_nodes=False)
    n, c, sp = _pca(coarse.points, ctree, k)
    _, near = ctree.query(pts, k=1, workers=1)
    cloud.normals, cloud.curvature, cloud.spread = n[near], c[near], sp[near]


def neighbour_graph(cloud: VoxelCloud, radius: float) -> sparse.csr_matrix:
    """Symmetric CSR adjacency of voxel points closer than ``radius``."""
    tree = cloud.tree if cloud.tree is not None else cKDTree(cloud.points)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    m = len(cloud.points)
    if len(pairs) == 0:
        g = sparse.csr_matrix((m, m), dtype=np.int8)
    else:
        data = np.ones(len(pairs) * 2, dtype=np.int8)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        g = sparse.csr_matrix((data, (rows, cols)), shape=(m, m))
    cloud.graph = g
    return g


def components_of(graph: sparse.csr_matrix, mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Indices of ``mask`` points connected (inside the mask) to any of ``seeds``."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return idx
    sub = graph[idx][:, idx]
    _, labels = connected_components(sub, directed=False)
    pos = np.full(graph.shape[0], -1, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    seed_pos = pos[np.asarray(seeds, dtype=np.int64)]
    seed_pos = seed_pos[seed_pos >= 0]
    if len(seed_pos) == 0:
        return np.zeros(0, dtype=np.int64)
    wanted = np.unique(labels[seed_pos])
    return idx[np.isin(labels, wanted)]


def median_spacing(points: np.ndarray, sample: int = 20000, seed: int = 7) -> float:
    """Median nearest-other-point distance (the scanner model-card statistic) on a fixed subsample."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return float("nan")
    tree = cKDTree(p, balanced_tree=False, compact_nodes=False)
    if len(p) > sample:
        rng = np.random.default_rng(seed)
        q = p[np.sort(rng.choice(len(p), size=sample, replace=False))]
    else:
        q = p
    d, _ = tree.query(q, k=2, workers=1)
    vals = d[:, 1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if len(vals) else float("nan")


def adaptive_voxel(points: np.ndarray, spacing: float, vmin: float, vmax: float, target: int) -> float:
    """Voxel edge that keeps the voxel cloud near ``target`` points (area estimated from the spacing)."""
    n = len(points)
    if not math.isfinite(spacing) or spacing <= 0:
        spacing = 0.5
    area = n * spacing * spacing  # mm^2 of sampled surface (roughly one point per spacing^2)
    v = math.sqrt(max(area, 1.0) / max(target, 1))
    return float(min(vmax, max(vmin, v)))
