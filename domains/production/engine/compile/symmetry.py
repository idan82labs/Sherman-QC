"""Stage-model symmetry test: the stage registers onto itself (plan §10 amendment 5).

``is_symmetric`` iff some *non-identity* self-map of the stage sample cloud
reaches an RMS distance of at most 0.3 mm (``SYMMETRY_RMS_THRESHOLD_MM``).

Method ``SELF_REGISTRATION_PCA_ICP_V1``:

1. The stage cloud is voxel-decimated to ``symmetry_target_spacing_mm``
   (target, with normals) and deterministically subsampled (source).
2. Candidate self-maps about the centroid come from the principal axes: the
   180 deg rotations about each axis (ROTATION) and the reflections across each
   principal plane plus the point inversion (MIRROR). A rotation axis of order
   3 or more forces two equal principal moments, so only when two moments
   agree within 8 % are the rotations by +-60, +-90 and +-120 deg about the third
   axis added (orders 3, 4 and 6), with the 180 deg flips and reflections about
   in-plane directions every 15 deg (the in-plane principal axes are arbitrary
   there; ICP refines the nearest one). Coverage is therefore: every symmetry
   of order 2 (any part), and orders 3, 4 and 6 about the axis of two equal
   moments. Symmetries of other orders (5, 8, ...) are **not** generated: such
   a part would report fewer candidates than it has, which makes a stage look
   less ambiguous than it is (none in the authorized corpus).
3. Each candidate is refined by point-to-plane ICP (the linear part stays a
   rotation for ROTATION candidates and a fixed reflection times a rotation for
   MIRROR candidates) and scored over the source sample. A point's distance is
   point-to-plane when it lands on the target surface (in-plane offset within
   1.5 target spacings and agreeing normals), otherwise point-to-point, so a
   point mapped into a hole or beyond an edge is never scored as "on surface".
   Distances are capped at 10 mm (a capped RMS above the threshold is a lower
   bound; one at or below it is exact), and ICP abandons a candidate once fewer
   than half of its points reach the surface.
4. Refined rotations within 5 deg / 1 mm of the identity are dropped;
   near-duplicate maps are merged.
5. Bend permutations: each formed bend's stage line is mapped by the candidate
   and matched to a formed bend line (parallel within 3 deg, midpoints within
   2 mm). ``ambiguous_bend_groups`` are the orbits (size >= 2) under the
   ROTATION candidates within the threshold: E3 registers with proper rigid
   transforms only, so a mirror-only symmetry never makes a bend identity
   ambiguous; a MIRROR candidate within the threshold means a mirrored part is
   indistinguishable from the CAD (E3 must not claim MIRRORED_PART_SUSPECTED).

Scope: each stage S_k is registered onto *itself*. Congruence *between*
hypotheses is not tested: an equal-wall U with one wall formed (S_1) has no
self-symmetry, yet it is congruent to the out-of-order hypothesis {B2}, so a
scan of either is ambiguous between them. E3 must test competing formed sets
itself (``stage_points_from_compiled`` builds any of them).
"""
from __future__ import annotations

import math

import numpy as np

from contracts.production.compiled_part import SYMMETRY_RMS_THRESHOLD_MM
from domains.production.engine.compile.geometry import rotation_angle_deg, rotation_matrix

__all__ = ["SYMMETRY_METHOD", "symmetry_test", "voxel_decimate"]

SYMMETRY_METHOD = "SELF_REGISTRATION_PCA_ICP_V1"
_DEGENERATE_RATIO = 0.92
_IDENTITY_DEG, _IDENTITY_MM = 5.0, 1.0
_ICP_POINTS = 3000
_MAX_CANDIDATES_RECORDED = 10
#: ICP abandons a candidate once fewer than this fraction of points find the surface.
_MIN_INLIER_FRACTION = 0.5
#: Point distances are capped here when scoring: a capped RMS above the 0.3 mm
#: threshold is a lower bound, and an RMS at or below it is exact (no point was capped).
_SCORE_CAP_MM = 10.0


def voxel_decimate(points: np.ndarray, spacing: float) -> np.ndarray:
    """Indices of one point per voxel (the first in input order): deterministic."""
    keys = np.floor(points / spacing).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def _candidates(evecs: np.ndarray, evals: np.ndarray) -> list[tuple[str, np.ndarray]]:
    e = [evecs[:, i] for i in range(3)]
    out: list[tuple[str, np.ndarray]] = []
    for v in e:
        out.append(("ROTATION", rotation_matrix(v, math.pi)))
    for v in e:
        out.append(("MIRROR", np.eye(3) - 2.0 * np.outer(v, v)))
    out.append(("MIRROR", -np.eye(3)))
    for i in range(3):
        for j in range(i + 1, 3):
            hi, lo = max(evals[i], evals[j]), min(evals[i], evals[j])
            if hi <= 0 or lo / hi < _DEGENERATE_RATIO:
                continue
            k = 3 - i - j
            # an axis of order 3, 4 or 6 is only possible about the axis of the two equal moments
            for deg in (90.0, 60.0, 120.0):
                out.append(("ROTATION", rotation_matrix(e[k], math.radians(deg))))
                out.append(("ROTATION", rotation_matrix(e[k], -math.radians(deg))))
            # in-plane 2-fold axes and mirror planes: the in-plane principal axes are arbitrary here
            for step in range(1, 12):
                if step == 6:
                    continue  # e[j] itself is a base candidate
                a = math.radians(15.0 * step)
                d = math.cos(a) * e[i] + math.sin(a) * e[j]
                d = d / np.linalg.norm(d)
                out.append(("ROTATION", rotation_matrix(d, math.pi)))
                out.append(("MIRROR", np.eye(3) - 2.0 * np.outer(d, d)))
    return out


def _icp(src: np.ndarray, src_n: np.ndarray, tree, tgt: np.ndarray, tgt_n: np.ndarray, linear: np.ndarray,
         center: np.ndarray, iterations: int, max_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Refine x -> L (x - c) + c + t by point-to-plane ICP; returns (L, t)."""
    lin = linear.copy()
    trans = np.zeros(3)
    rel = src - center
    for it in range(iterations):
        q = rel @ lin.T + center + trans
        qn = src_n @ lin.T
        dist, idx = tree.query(q, distance_upper_bound=max_dist)
        found = np.isfinite(dist)
        if it >= 2 and float(found.mean()) < _MIN_INLIER_FRACTION:
            break  # not converging onto the surface: not a self-map
        idx = np.where(found, idx, 0)
        n = tgt_n[idx]
        ok = found & (np.einsum("ij,ij->i", qn, n) > 0.5)
        if int(ok.sum()) < 12:
            break
        qo, no, to = q[ok], n[ok], tgt[idx[ok]]
        r = np.einsum("ij,ij->i", qo - to, no)
        a = np.hstack([np.cross(qo, no), no])
        try:
            x, *_ = np.linalg.lstsq(a, -r, rcond=None)
        except np.linalg.LinAlgError:
            break
        w, tau = x[:3], x[3:]
        ang = float(np.linalg.norm(w))
        dr = rotation_matrix(w / ang, ang) if ang > 1e-12 else np.eye(3)
        # new map: dr @ (L rel + c + t) + tau
        lin = dr @ lin
        trans = dr @ (center + trans) + tau - center
        if ang < 1e-7 and float(np.linalg.norm(tau)) < 1e-6:
            break
    return lin, trans


def _score(src: np.ndarray, src_n: np.ndarray, tree, tgt: np.ndarray, tgt_n: np.ndarray, lin: np.ndarray,
           trans: np.ndarray, center: np.ndarray, spacing: float) -> float:
    q = (src - center) @ lin.T + center + trans
    qn = src_n @ lin.T
    dist, idx = tree.query(q, distance_upper_bound=_SCORE_CAP_MM)
    far = ~np.isfinite(dist)
    dist = np.where(far, _SCORE_CAP_MM, dist)
    idx = np.where(far, 0, idx)
    n = tgt_n[idx]
    diff = q - tgt[idx]
    normal_off = np.einsum("ij,ij->i", diff, n)
    in_plane = np.sqrt(np.maximum(dist * dist - normal_off * normal_off, 0.0))
    agree = np.einsum("ij,ij->i", qn, n) > 0.5
    on_surface = (in_plane <= 1.5 * spacing) & agree & ~far
    d = np.where(on_surface, np.abs(normal_off), dist)
    return float(math.sqrt(float(np.mean(d * d))))


def _to_mat4(lin: np.ndarray, trans: np.ndarray, center: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = lin
    m[:3, 3] = center + trans - lin @ center
    return m


def _permutation(m4: np.ndarray, lines: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, str]:
    out: dict[str, str] = {}
    items = sorted(lines.items())
    for uid, (a, b) in items:
        ta, tb = m4[:3, :3] @ a + m4[:3, 3], m4[:3, :3] @ b + m4[:3, 3]
        d = tb - ta
        d = d / max(np.linalg.norm(d), 1e-12)
        mid = 0.5 * (ta + tb)
        best = None
        for uid2, (a2, b2) in items:
            d2 = b2 - a2
            d2 = d2 / max(np.linalg.norm(d2), 1e-12)
            if abs(float(d @ d2)) < math.cos(math.radians(3.0)):
                continue
            dist = float(np.linalg.norm(mid - 0.5 * (a2 + b2)))
            if dist <= 2.0 and (best is None or dist < best[0]):
                best = (dist, uid2)
        if best is not None:
            out[uid] = best[1]
    return out


def symmetry_test(points: np.ndarray, normals: np.ndarray, formed_lines: dict[str, tuple[np.ndarray, np.ndarray]],
                  *, target_spacing: float, source_points: int, iterations: int) -> dict:
    """Self-registration test of one stage cloud; returns ``SymmetryTestV1`` fields."""
    from scipy.spatial import cKDTree

    threshold = SYMMETRY_RMS_THRESHOLD_MM
    if len(points) < 100:
        return {"method": SYMMETRY_METHOD, "threshold_rms_mm": threshold, "candidates": [], "is_symmetric": False,
                "ambiguous_bend_groups": []}
    keep = voxel_decimate(points, target_spacing)
    tgt, tgt_n = points[keep], normals[keep]
    tree = cKDTree(tgt)
    step = max(1, len(tgt) // source_points)
    src, src_n = tgt[::step], tgt_n[::step]
    icp_step = max(1, len(src) // _ICP_POINTS)
    icp_src, icp_n = src[::icp_step], src_n[::icp_step]
    center = tgt.mean(axis=0)
    cov = np.cov((tgt - center).T)
    evals, evecs = np.linalg.eigh(cov)
    for i in range(3):
        k = int(np.argmax(np.abs(evecs[:, i])))
        if evecs[k, i] < 0:
            evecs[:, i] = -evecs[:, i]
    found: list[tuple[float, str, np.ndarray]] = []
    for kind, lin0 in _candidates(evecs, evals):
        lin, trans = _icp(icp_src, icp_n, tree, tgt, tgt_n, lin0, center, iterations, max_dist=4.0 * target_spacing)
        if kind == "ROTATION" and np.linalg.det(lin) < 0:
            continue
        m4 = _to_mat4(lin, trans, center)
        if kind == "ROTATION" and rotation_angle_deg(m4) < _IDENTITY_DEG and float(
                np.linalg.norm(m4[:3, 3] - (center - m4[:3, :3] @ center))) < _IDENTITY_MM:
            continue
        rms = _score(src, src_n, tree, tgt, tgt_n, lin, trans, center, target_spacing)
        dup = False
        for i, (r2, k2, m2) in enumerate(found):
            if k2 == kind and float(np.max(np.abs(m2[:3, :3] - m4[:3, :3]))) < 0.02 and float(
                    np.linalg.norm(m2[:3, 3] - m4[:3, 3])) < 0.5:
                if rms < r2:
                    found[i] = (rms, kind, m4)
                dup = True
                break
        if not dup:
            found.append((rms, kind, m4))
    found.sort(key=lambda item: (item[0], item[1]))
    passing = [f for f in found if f[0] <= threshold]
    recorded = passing + [f for f in found if f[0] > threshold][: max(0, _MAX_CANDIDATES_RECORDED - len(passing))]
    candidates = []
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rms, kind, m4 in recorded:
        perm = _permutation(m4, formed_lines)
        candidates.append({"kind": kind, "transform": m4, "rms_mm": rms, "bend_permutation": perm})
        if kind == "ROTATION" and rms <= threshold:
            for a, b in perm.items():
                if a != b:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for uid in parent:
        groups.setdefault(find(uid), []).append(uid)
    orbit_groups = sorted(tuple(sorted(g)) for g in groups.values() if len(g) >= 2)
    return {"method": SYMMETRY_METHOD, "threshold_rms_mm": threshold, "candidates": candidates,
            "is_symmetric": bool(passing), "ambiguous_bend_groups": orbit_groups}
