"""D1 two-skin analysis of a scanned sheet-metal mesh (research only; no engine imports).

Geometry: a sheet of thickness t has two skins. Mesh faces are wound outward (away from the
material), so a ray cast from a skin point along -n (into the material) leaves the material
through the opposite skin, whose outward normal is ~ -n, at distance ~ t. At a bend of included
angle theta the two skins' virtual sharps are t/sin(theta/2) apart along the bisector; the two
arc apexes are exactly t apart (concentric arcs).

Public API
  load_mesh(path)                        -> Mesh (vertices, faces, face/vertex normals, areas)
  probe_inward(mesh, points, normals)    -> distance to the opposite skin + antiparallel test
  estimate_thickness(mesh)               -> ThicknessEstimate (t, confidence, stats)
  label_skins(mesh, t)                   -> SkinLabels (per vertex two_sided, side A/B, mid point)
  mid_surface(mesh, labels, t)           -> (points, normals) of the collapsed sheet
  bend_profile(mesh_pts, mesh_nrm, line, t) -> BendProfile (outer/inner loci in the cross-section)
  snap_line_to_outer_apex(...)           -> moved line + diagnostics

No part-specific logic: every threshold is in mm or a multiple of the measured t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------- mesh


@dataclass
class Mesh:
    vertices: np.ndarray            # (V,3) float64
    faces: np.ndarray               # (F,3) int64
    face_normals: np.ndarray        # (F,3) unit, from winding
    face_areas: np.ndarray          # (F,)
    vertex_normals: np.ndarray      # (V,3) unit, area-weighted face normals (same winding)
    file_normal_agreement: float | None = None   # mean sign agreement with the file's vertex normals


def mesh_from_arrays(vertices: np.ndarray, faces: np.ndarray, file_normals: np.ndarray | None = None) -> Mesh:
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm = np.linalg.norm(cr, axis=1)
    area = 0.5 * nrm
    fn = cr / np.maximum(nrm, 1e-300)[:, None]
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], cr)          # area-weighted (|cr| = 2*area)
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-300)[:, None]
    agree = None
    if file_normals is not None and len(file_normals) == len(V):
        fnn = np.asarray(file_normals, dtype=np.float64)
        ok = np.linalg.norm(fnn, axis=1) > 0.5
        if ok.any():
            agree = float(np.mean(np.sign(np.sum(fnn[ok] * vn[ok], axis=1))))
    return Mesh(V, F, fn, area, vn, agree)


def load_mesh(path: str) -> Mesh:
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(path))
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)
    if len(F) == 0:
        raise ValueError(f"{path}: no faces (D1 needs a mesh)")
    fnorm = np.asarray(m.vertex_normals, dtype=np.float64) if m.has_vertex_normals() else None
    return mesh_from_arrays(V, F, fnorm)


def _scene(mesh: Mesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                        o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    return scene


def cast(scene, origins: np.ndarray, dirs: np.ndarray, chunk: int = 400_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_hit, hit primitive id); t_hit = inf on a miss."""
    import open3d as o3d

    th = np.empty(len(origins), dtype=np.float64)
    pid = np.empty(len(origins), dtype=np.int64)
    for s in range(0, len(origins), chunk):
        rays = np.hstack([origins[s:s + chunk], dirs[s:s + chunk]]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        th[s:s + chunk] = hit["t_hit"].numpy()
        pid[s:s + chunk] = hit["primitive_ids"].numpy().astype(np.int64)
    return th, pid


@dataclass
class Probe:
    distance: np.ndarray        # (n,) distance to the first surface along -n (inf on miss)
    hit_face: np.ndarray        # (n,) face id (-1 on miss)
    cos_hit: np.ndarray         # (n,) dot(hit face normal, n); ~ -1 for the opposite skin
    exits: np.ndarray           # (n,) bool, the ray leaves the material at the hit (dir . n_hit > 0)


def probe_inward(mesh: Mesh, points: np.ndarray, normals: np.ndarray, scene=None, eps: float = 0.02) -> Probe:
    scene = scene if scene is not None else _scene(mesh)
    d = -normals
    th, pid = cast(scene, points + eps * d, d)
    miss = ~np.isfinite(th) | (pid < 0) | (pid >= len(mesh.faces))
    pid = np.where(miss, -1, pid)
    hn = np.where(miss[:, None], 0.0, mesh.face_normals[np.clip(pid, 0, None)])
    cos_hit = np.sum(hn * normals, axis=1)
    exits = np.sum(hn * d, axis=1) > 0
    dist = np.where(miss, np.inf, th + eps)
    return Probe(dist, pid, cos_hit, exits)


# ------------------------------------------------------------------------ thickness

ANTIPARALLEL_COS = -0.85   # opposite skin normal within ~32 deg of antiparallel


@dataclass
class ThicknessEstimate:
    t_mm: float | None
    confidence: float
    n_samples: int
    n_valid: int
    two_sided_fraction: float      # samples whose inward probe lands on the opposite skin at ~t
    peak_fraction: float           # valid hits within +-20 % of t
    mad_mm: float | None
    exit_fraction: float           # orientation sanity: inward first hits that leave the material
    second_mode_mm: float | None   # e.g. hem / double wall (reported, not used)
    histogram: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def sample_surface(mesh: Mesh, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = mesh.face_areas / mesh.face_areas.sum()
    fi = rng.choice(len(mesh.faces), size=n, p=p)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    V, F = mesh.vertices, mesh.faces[fi]
    pts = a[:, None] * V[F[:, 0]] + b[:, None] * V[F[:, 1]] + c[:, None] * V[F[:, 2]]
    return pts, mesh.face_normals[fi]


def robust_mode(values: np.ndarray, lo: float, hi: float, bin_mm: float = 0.02, smooth_mm: float = 0.06):
    """Histogram mode on [lo,hi] with Gaussian smoothing; returns (mode, counts, edges)."""
    edges = np.arange(lo, hi + bin_mm, bin_mm)
    counts, edges = np.histogram(values, bins=edges)
    k = max(1, int(round(3 * smooth_mm / bin_mm)))
    x = np.arange(-k, k + 1) * bin_mm
    w = np.exp(-0.5 * (x / smooth_mm) ** 2)
    sm = np.convolve(counts.astype(np.float64), w / w.sum(), mode="same")
    i = int(np.argmax(sm))
    return 0.5 * (edges[i] + edges[i + 1]), sm, edges


def estimate_thickness(mesh: Mesh, n_samples: int = 120_000, dmin: float = 0.25, dmax: float = 12.0,
                       seed: int = 0, scene=None) -> ThicknessEstimate:
    rng = np.random.default_rng(seed)
    scene = scene if scene is not None else _scene(mesh)
    pts, nrm = sample_surface(mesh, n_samples, rng)
    pr = probe_inward(mesh, pts, nrm, scene=scene)
    hit = np.isfinite(pr.distance)
    exit_frac = float(np.mean(pr.exits[hit])) if hit.any() else 0.0
    ok = hit & pr.exits & (pr.cos_hit < ANTIPARALLEL_COS) & (pr.distance > dmin) & (pr.distance < dmax)
    d = pr.distance[ok]
    if len(d) < 200:
        return ThicknessEstimate(None, 0.0, n_samples, int(len(d)), float(len(d) / n_samples), 0.0, None,
                                 exit_frac, None)
    mode, sm, edges = robust_mode(d, dmin, dmax)
    near = np.abs(d - mode) < 0.2 * mode
    t = float(np.median(d[near]))
    near = np.abs(d - t) < 0.2 * t
    t = float(np.median(d[near]))
    mad = float(np.median(np.abs(d[near] - t)))
    two_sided = float(np.mean(ok & (np.abs(pr.distance - t) < 0.25 * t)))
    peak = float(np.mean(near))
    # second mode (outside +-35 % of t), for hems / double walls / gaps
    far = np.abs(d - t) > 0.35 * t
    second = None
    if far.sum() > 0.1 * near.sum():
        m2, _, _ = robust_mode(d[far], dmin, dmax)
        second = float(m2)
    # confidence: dominance of the peak, two-sided coverage, tightness of the peak
    conf = peak * min(1.0, two_sided / 0.25) * math.exp(-(mad / max(t, 1e-6)) / 0.15)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = sm > 0.01 * sm.max()
    hist = {"bin_mm": 0.02, "centres_mm": [round(float(c), 3) for c in centres[keep]],
            "smoothed_counts": [round(float(c), 2) for c in sm[keep]]}
    return ThicknessEstimate(t, float(conf), n_samples, int(ok.sum()), two_sided, peak, mad, exit_frac, second, hist)


# ----------------------------------------------------------------------- skin labels


@dataclass
class SkinLabels:
    two_sided: np.ndarray       # (V,) bool, inward probe lands on the opposite skin at t +- 35 %
    pair_distance: np.ndarray   # (V,) inward distance (inf if none)
    side: np.ndarray            # (V,) int8: 0 = side A, 1 = side B, -1 = unlabelled (edge band / one-sided)
    consistency: float          # fraction of pair links that join opposite sides after 2-colouring
    n_components: int

    def summary(self) -> dict[str, Any]:
        lab = self.side >= 0
        return {"two_sided_fraction": round(float(self.two_sided.mean()), 4),
                "labelled_fraction": round(float(lab.mean()), 4),
                "side_a_fraction": round(float(np.mean(self.side == 0)), 4),
                "side_b_fraction": round(float(np.mean(self.side == 1)), 4),
                "pair_link_consistency": round(self.consistency, 4),
                "components": self.n_components}


def label_skins(mesh: Mesh, t: float, scene=None, tol: float = 0.35) -> SkinLabels:
    """Pair skins and 2-colour them into the sheet's two sides.

    Vertices whose inward probe reaches an antiparallel surface at t*(1 +- tol) are two-sided.
    Mesh edges between two-sided vertices form skin components (the cut-edge band and holes'
    walls are not two-sided, so they separate the skins). Each pair link joins a vertex to the
    component of the face it hit; components are 2-coloured greedily along the heaviest links.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    scene = scene if scene is not None else _scene(mesh)
    V, F = mesh.vertices, mesh.faces
    pr = probe_inward(mesh, V, mesh.vertex_normals, scene=scene)
    two = (np.isfinite(pr.distance) & pr.exits & (pr.cos_hit < ANTIPARALLEL_COS)
           & (np.abs(pr.distance - t) < tol * t))
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    N = mesh.vertex_normals
    # skin edges only: both ends two-sided and normals agreeing (an edge across a coarse cut-edge
    # band joins antiparallel normals and must not merge the two skins)
    e = e[two[e[:, 0]] & two[e[:, 1]] & (np.sum(N[e[:, 0]] * N[e[:, 1]], axis=1) > 0.97)]
    n = len(V)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    ncomp, comp = connected_components(g, directed=False)
    comp = np.where(two, comp, -1)
    # pair links component -> component (hit face's first two-sided vertex)
    src = np.where(two)[0]
    hf = F[pr.hit_face[src]]
    hv = np.where(two[hf[:, 0]], hf[:, 0], np.where(two[hf[:, 1]], hf[:, 1], hf[:, 2]))
    ca, cb = comp[src], comp[hv]
    good = cb >= 0
    ca, cb = ca[good], cb[good]
    side_c = -np.ones(ncomp, dtype=np.int8)
    if len(ca):
        key = np.minimum(ca, cb).astype(np.int64) * ncomp + np.maximum(ca, cb)
        uk, cnt = np.unique(key, return_counts=True)
        a_, b_ = uk // ncomp, uk % ncomp
        same = a_ == b_
        csize = np.bincount(comp[comp >= 0], minlength=ncomp)
        adj: dict[int, list[tuple[int, int]]] = {}
        for x, y, w in zip(a_[~same], b_[~same], cnt[~same]):
            adj.setdefault(int(x), []).append((int(y), int(w)))
            adj.setdefault(int(y), []).append((int(x), int(w)))
        import heapq
        for seed in np.argsort(-csize):
            if side_c[seed] >= 0 or csize[seed] == 0 or int(seed) not in adj:
                continue
            side_c[seed] = 0
            heap = [(-w, int(seed), y) for y, w in adj[int(seed)]]
            heapq.heapify(heap)
            while heap:
                _, x, y = heapq.heappop(heap)
                if side_c[y] >= 0:
                    continue
                side_c[y] = 1 - side_c[x]
                for z, w in adj.get(y, []):
                    if side_c[z] < 0:
                        heapq.heappush(heap, (-w, y, z))
        sa, sb = side_c[ca], side_c[cb]
        lab = (sa >= 0) & (sb >= 0)
        consistency = float(np.mean(sa[lab] != sb[lab])) if lab.any() else 0.0
    else:
        consistency = 0.0
    side = np.where(comp >= 0, side_c[np.clip(comp, 0, None)], -1).astype(np.int8)
    return SkinLabels(two, pr.distance, side, consistency, int(ncomp))


def mid_surface(mesh: Mesh, labels: SkinLabels, t: float, keep_one_sided: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the sheet to one surface: two-sided vertices move inward by half their pair distance,
    one-sided skin vertices (optional) by t/2; edge-band / unpaired-wall vertices are dropped.
    Each pair is represented twice (once from each skin); the downstream voxeliser merges them."""
    V, N = mesh.vertices, mesh.vertex_normals
    two = labels.two_sided
    out_p = [V[two] - 0.5 * labels.pair_distance[two, None] * N[two]]
    out_n = [N[two]]
    if keep_one_sided:
        # one-sided: inward probe hit nothing within 3t (an unscanned opposite skin), not an edge band
        one = ~two & ~(labels.pair_distance < 3.0 * t)
        out_p.append(V[one] - 0.5 * t * N[one])
        out_n.append(N[one])
    return np.vstack(out_p), np.vstack(out_n)


def side_points(mesh: Mesh, labels: SkinLabels, side: int) -> tuple[np.ndarray, np.ndarray]:
    m = labels.side == side
    return mesh.vertices[m], mesh.vertex_normals[m]


# ------------------------------------------------------------------- bend profile


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = _unit(axis)
    ref = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(u, ref))
    e2 = np.cross(u, e1)
    return e1, e2


@dataclass
class BendProfile:
    ok: bool
    reason: str = ""
    included_angle_deg: float | None = None
    bisector_2d: list[float] | None = None          # unit, points to the convex (outer) side
    outer_sharp_2d: list[float] | None = None       # outer-skin virtual sharp, in the line's 2D frame
    inner_sharp_2d: list[float] | None = None
    mid_sharp_2d: list[float] | None = None
    outer_apex_2d: list[float] | None = None        # outer arc apex (point of the outer arc on the bisector)
    inner_apex_2d: list[float] | None = None
    outer_apex_support: int = 0
    inner_apex_support: int = 0
    arm_points: list[int] | None = None
    arm_outer_fraction: list[float] | None = None
    local_t_mm: float | None = None                 # outer-to-inner skin offset measured on the arms
    e1: list[float] | None = None
    e2: list[float] | None = None
    origin: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def to3d(self, p2: list[float] | None) -> np.ndarray | None:
        if p2 is None:
            return None
        return np.asarray(self.origin) + p2[0] * np.asarray(self.e1) + p2[1] * np.asarray(self.e2)


def _line_intersect(m1, o1, m2, o2):
    A = np.array([m1, m2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([o1, o2]))


def bend_profile(points: np.ndarray, normals: np.ndarray, start: np.ndarray, end: np.ndarray, t: float,
                 radius_mm: float | None = None, trim: float = 0.1, min_arm_deg: float = 8.0) -> BendProfile:
    """Cross-section analysis around a candidate bend line (the line need not be on the bend exactly).

    Points within the trimmed span and within radius_mm of the line are projected onto the plane
    perpendicular to the line axis. The two flange arms are the two dominant 2-D normal
    directions outside the bend zone; oriented normals give each arm point's skin (outer when its
    outward normal points to the convex side). Loci: outer/inner/mid virtual sharps (intersections
    of the arms' skin lines) and outer/inner arc apexes (surface points on the bisector).
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    axis = end - start
    L = float(np.linalg.norm(axis))
    u = axis / L
    e1, e2 = _frame(u)
    mid = 0.5 * (start + end)
    R = radius_mm if radius_mm is not None else max(14.0, 5.0 * t)
    rel = points - mid
    s = rel @ u
    sel = np.abs(s) < (0.5 - trim) * L
    rel = rel[sel]
    nn = normals[sel]
    q = np.c_[rel @ e1, rel @ e2]
    r = np.linalg.norm(q, axis=1)
    sel2 = (r < R) & (np.abs(nn @ u) < 0.5)
    q, nn, r = q[sel2], nn[sel2], r[sel2]
    prof = BendProfile(False, e1=e1.tolist(), e2=e2.tolist(), origin=mid.tolist())
    if len(q) < 40:
        prof.reason = "TOO_FEW_POINTS"
        return prof
    m2 = np.c_[nn @ e1, nn @ e2]
    m2 /= np.maximum(np.linalg.norm(m2, axis=1), 1e-12)[:, None]
    # arm points: outside the bend zone
    r_in = max(2.5, 1.5 * t)
    arm = r > r_in
    if arm.sum() < 30:
        prof.reason = "NO_ARMS"
        return prof
    ang = np.mod(np.degrees(np.arctan2(m2[arm, 1], m2[arm, 0])), 180.0)   # normal direction mod 180
    # nearer points weigh more: the arms of THIS bend dominate over other flanges in the window
    hist, edges = np.histogram(ang, bins=180, range=(0, 180), weights=np.exp(-r[arm] / 6.0))
    k = np.array([0.25, 0.5, 1, 0.5, 0.25])
    sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode="same")[2:-2]
    p1 = int(np.argmax(sm))
    dist_deg = np.minimum(np.abs(np.arange(180) - p1), 180 - np.abs(np.arange(180) - p1))
    sm2 = np.where(dist_deg > min_arm_deg, sm, -1)
    p2 = int(np.argmax(sm2))
    if sm2[p2] < max(0.5, 0.08 * sm[p1]):
        prof.reason = "ONE_ARM"
        return prof
    phis = []
    for p in (p1, p2):
        dd = np.minimum(np.abs(ang - (p + 0.5)), 180 - np.abs(ang - (p + 0.5)))
        mem = dd < 4.0
        # circular mean of doubled angles
        a2 = np.radians(2 * ang[mem])
        phis.append(0.5 * math.degrees(math.atan2(np.sin(a2).mean(), np.cos(a2).mean())) % 180)
    arm_idx = np.where(arm)[0]
    arms = []
    for phi in phis:
        nvec = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
        tvec = np.array([-nvec[1], nvec[0]])
        dd = np.abs(m2[arm_idx] @ nvec)
        mem = arm_idx[dd > math.cos(math.radians(6.0))]
        along = q[mem] @ tvec
        # the arm leaves the bend on one side: majority sign of the along-coordinate
        sgn = 1.0 if np.median(along) >= 0 else -1.0
        a_dir = sgn * tvec
        mem = mem[(q[mem] @ a_dir) > 0]
        if len(mem) >= 10:
            # keep the sheet nearest the line (drop parallel flanges further away)
            o = q[mem] @ nvec
            near = np.argsort(r[mem])[:max(5, len(mem) // 5)]
            o_near = float(np.median(o[near]))
            mem = mem[np.abs(o - o_near) < 1.5 * t + 0.5]
        arms.append((a_dir, nvec, mem))
    a1, a2 = arms[0][0], arms[1][0]
    ssum = a1 + a2
    if np.linalg.norm(ssum) < 1e-3:          # flat (180 deg) -> not a bend
        prof.reason = "FLAT"
        return prof
    b = -_unit(ssum)
    cos_inc = float(np.clip(a1 @ a2, -1, 1))
    theta = math.degrees(math.acos(cos_inc))       # included angle between the arms
    outer_lines, inner_lines, fr, npts = [], [], [], []
    loc_t = []
    for a_dir, nvec, mem in arms:
        m_out = nvec if nvec @ b > 0 else -nvec           # arm normal toward the convex side
        o = q[mem] @ m_out
        on = m2[mem] @ m_out
        is_out = on > 0.7
        is_in = on < -0.7
        npts.append(int(len(mem)))
        fr.append(float(is_out.mean()) if len(mem) else 0.0)
        oo = float(np.median(o[is_out])) if is_out.sum() >= 8 else None
        oi = float(np.median(o[is_in])) if is_in.sum() >= 8 else None
        if oo is not None and oi is not None:
            loc_t.append(oo - oi)
        if oo is None and oi is None:
            prof.reason = "ARM_WITHOUT_SKIN"
            return prof
        if oo is None:
            oo = oi + t
        if oi is None:
            oi = oo - t
        outer_lines.append((m_out, oo))
        inner_lines.append((m_out, oi))
    vo = _line_intersect(outer_lines[0][0], outer_lines[0][1], outer_lines[1][0], outer_lines[1][1])
    vi = _line_intersect(inner_lines[0][0], inner_lines[0][1], inner_lines[1][0], inner_lines[1][1])
    if vo is None or vi is None:
        prof.reason = "PARALLEL_ARMS"
        return prof
    vm = 0.5 * (vo + vi)
    perp = np.array([-b[1], b[0]])
    w = max(0.75, 0.4 * t)

    def apex(v0, sign):
        lat = (q - v0) @ perp
        ax = (q - v0) @ b
        on = m2 @ b
        cand = (np.abs(lat) < w) & (sign * on > 0.7) & (ax < 1.0) & (ax > -R / 2)
        if cand.sum() < 3:
            return None, int(cand.sum())
        return (v0 + b * float(np.median(ax[cand]))).tolist(), int(cand.sum())

    ao, nao = apex(vo, +1)
    ai, nai = apex(vo, -1)
    prof.ok = True
    prof.included_angle_deg = theta
    prof.bisector_2d = b.tolist()
    prof.outer_sharp_2d, prof.inner_sharp_2d, prof.mid_sharp_2d = vo.tolist(), vi.tolist(), vm.tolist()
    prof.outer_apex_2d, prof.inner_apex_2d = ao, ai
    prof.outer_apex_support, prof.inner_apex_support = nao, nai
    prof.arm_points, prof.arm_outer_fraction = npts, fr
    prof.local_t_mm = float(np.median(loc_t)) if loc_t else None
    return prof


def expected_skin_offsets(t: float, theta_deg: float) -> dict[str, float]:
    s = math.sin(math.radians(theta_deg) / 2)
    return {"virtual_sharp_skin_to_skin": t / max(s, 1e-6), "arc_apex_skin_to_skin": t}


def snap_line_to_outer_apex(start, end, profile: BendProfile, max_shift_mm: float,
                            prefer: str = "apex") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Translate the line (axis and span kept) onto the outer-skin locus of its profile."""
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    info: dict[str, Any] = {"moved": False}
    if not profile.ok:
        info["reason"] = profile.reason
        return start, end, info
    target = profile.outer_apex_2d if (prefer == "apex" and profile.outer_apex_2d is not None) else None
    used = "outer_apex"
    if target is None:
        # no surface on the bisector (scan gap at the bend): fall back to the outer virtual sharp
        target, used = profile.outer_sharp_2d, "outer_sharp"
    shift = float(np.hypot(*target))
    info.update(locus=used, shift_mm=round(shift, 4))
    if shift > max_shift_mm:
        info["reason"] = "SHIFT_TOO_LARGE"
        return start, end, info
    d3 = target[0] * np.asarray(profile.e1) + target[1] * np.asarray(profile.e2)
    info["moved"] = True
    return start + d3, end + d3, info
