"""Truth-side symmetry test: does a stage model register onto itself?

Mirrors the contract's ``SymmetryTestV1`` semantics (plan §10 amendment 5) on
E2's own geometry so E4 can cross-check E1's compile-time test:

- area-weighted samples of the stage surface;
- candidates from the principal-axis frame: the three 180° rotations about
  the principal axes (``ROTATION``) and the three principal-plane reflections
  plus the point inversion (``MIRROR``);
- each candidate refined by point-to-surface ICP (proper Kabsch updates; a
  mirror keeps its reflection) against the tessellated surface;
- ``rms_mm`` = RMS point-to-surface distance after refinement (exact
  symmetry gives ~tessellation error, far below the 0.3 mm threshold);
- bend permutation: each formed bend line's midpoint/direction mapped by the
  candidate onto the nearest parallel bend line within 2 mm.

``is_symmetric`` follows the contract (any candidate within threshold);
``rotation_symmetric`` / ``mirror_symmetric`` split it by kind. Ambiguous bend
groups are the orbits (size >= 2) under the within-threshold candidates.
The candidate set does not enumerate 90° rotations of square-like frames;
the generator's families never rely on those.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .brep import TessellatedSurface

__all__ = ["symmetry_test", "sample_surface", "SYMMETRY_THRESHOLD_MM"]

SYMMETRY_THRESHOLD_MM = 0.3


def sample_surface(surface: TessellatedSurface, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` area-weighted random points on the surface."""
    areas = surface.triangle_areas
    p = areas / areas.sum()
    tri = rng.choice(len(areas), size=n, p=p)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    v = surface.vertices[surface.triangles[tri]]
    return (1 - r1)[:, None] * v[:, 0] + (r1 * (1 - r2))[:, None] * v[:, 1] + (r1 * r2)[:, None] * v[:, 2]


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Proper rigid 4x4 minimising |R a + t - b|."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = cb - r @ ca
    return m


def symmetry_test(
    surface: TessellatedSurface,
    bend_lines: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    n_samples: int = 8000,
    threshold_mm: float = SYMMETRY_THRESHOLD_MM,
    icp_iterations: int = 12,
    nthreads: int | None = None,
) -> dict[str, Any]:
    """Self-registration symmetry test (see module docstring).

    ``bend_lines``: formed bends' lines (2,3) in the surface's frame, keyed by id.
    ``nthreads``: Open3D query threads (``None``: from ``OMP_NUM_THREADS``, see
    :func:`.scanner.default_ray_threads`); results do not depend on it."""
    import open3d as o3d

    from .scanner import default_ray_threads

    threads = default_ray_threads() if nthreads is None else int(nthreads)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(surface.vertices.astype(np.float32)), o3d.core.Tensor(surface.triangles.astype(np.uint32))
    )
    pts = sample_surface(surface, n_samples, rng)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    _, vecs = np.linalg.eigh(cov)
    e = vecs  # columns = principal axes
    cands: list[tuple[str, np.ndarray]] = []
    for s in ((1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        cands.append(("ROTATION", np.diag(s).astype(float)))
    for s in ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1)):
        cands.append(("MIRROR", np.diag(s).astype(float)))
    results = []
    for kind, s in cands:
        lin = e @ s @ e.T
        m0 = np.eye(4)
        m0[:3, :3] = lin
        m0[:3, 3] = c - lin @ c
        cur = m0.copy()
        for _ in range(icp_iterations):
            moved = pts @ cur[:3, :3].T + cur[:3, 3]
            closest = scene.compute_closest_points(o3d.core.Tensor(moved.astype(np.float32)), nthreads=threads)["points"].numpy().astype(np.float64)
            step = _kabsch(moved, closest)
            cur = step @ cur
        moved = pts @ cur[:3, :3].T + cur[:3, 3]
        dist = scene.compute_distance(o3d.core.Tensor(moved.astype(np.float32)), nthreads=threads).numpy().astype(np.float64)
        rms = float(math.sqrt(float(np.mean(dist**2))))
        perm = _bend_permutation(cur, bend_lines)
        results.append({"kind": kind, "transform": cur.tolist(), "rms_mm": rms, "bend_permutation": perm})
    hits = [r for r in results if r["rms_mm"] <= threshold_mm]
    groups = _orbits([r["bend_permutation"] for r in hits])
    return {
        "method": "e2_pca_candidates_point_to_surface_icp_v1",
        "threshold_rms_mm": threshold_mm,
        "samples": n_samples,
        "candidates": results,
        "is_symmetric": bool(hits),
        "rotation_symmetric": any(r["kind"] == "ROTATION" for r in hits),
        "mirror_symmetric": any(r["kind"] == "MIRROR" for r in hits),
        "ambiguous_bend_groups": groups,
    }


def _bend_permutation(m: np.ndarray, lines: dict[str, np.ndarray]) -> dict[str, str]:
    keys = sorted(lines)
    mids = {k: lines[k].mean(axis=0) for k in keys}
    dirs = {}
    for k in keys:
        d = lines[k][1] - lines[k][0]
        dirs[k] = d / np.linalg.norm(d)
    out: dict[str, str] = {}
    for k in keys:
        pm = m[:3, :3] @ mids[k] + m[:3, 3]
        pd = m[:3, :3] @ dirs[k]
        best, best_d = None, 2.0
        for k2 in keys:
            if abs(float(pd @ dirs[k2])) < math.cos(math.radians(5.0)):
                continue
            dd = float(np.linalg.norm(pm - mids[k2]))
            if dd < best_d:
                best, best_d = k2, dd
        if best is not None:
            out[k] = best
    return out


def _orbits(perms: list[dict[str, str]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for perm in perms:
        for a, b in perm.items():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    comps: dict[str, list[str]] = {}
    for x in list(parent):
        comps.setdefault(find(x), []).append(x)
    return sorted(sorted(g) for g in comps.values() if len(g) >= 2)
