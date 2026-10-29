"""Z2 step 1: planar skin regions and flanges from the scan alone.

Input: a triangle mesh (outward winding, as every Sherman scan inspected). Work set: mesh vertices
with area-weighted vertex normals (randomly thinned above ``max_points``). Sheet thickness t from
the D1 two-skin probe (``research.zeroshot.d1_two_skin.two_skin.estimate_thickness``, read-only).

1. Normals smoothed over kNN neighbours that face the same way (the opposite skin, antiparallel,
   never mixes in).
2. Seeds = connected components of the kNN graph restricted to *flat* points joined by edges
   whose smoothed normals agree within ``seed_deg`` and whose offset along the normal is small.
3. Robust plane per seed (E3 ``geom.robust_plane``), then plane-guided growth: an unlabelled
   point joins a neighbouring region when it lies within ``grow_dist`` of its plane and within
   ``grow_deg`` of its normal.
4. Adjacent coplanar regions are merged.
5. Skin regions are paired into flanges: antiparallel, one thickness apart, overlapping
   footprints (union-find). Narrow unpaired strips about one thickness wide are cut walls, not
   flanges.

Every threshold is in mm, degrees or a multiple of the measured t.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from research.zeroshot.z2_self_recipe import common  # noqa: F401  (sys.path)
from research.zeroshot.d1_two_skin import two_skin as D1
from domains.production.engine.inspect.geom import robust_plane, Plane


@dataclass
class SegParams:
    max_points: int = 450_000
    knn: int = 16
    smooth_iters: int = 2
    seed_deg: float = 2.5
    flat_deg: float = 4.0          # max deviation of a seed point's smoothed normal from its neighbours'
    seed_offset_mm: float = 0.12
    min_seed_points: int = 30
    grow_deg: float = 12.0
    grow_dist_mm: float = 0.30
    grow_iters: int = 60
    merge_deg: float = 2.5
    merge_dist_mm: float = 0.35
    min_region_area_mm2: float = 12.0
    pair_deg: float = 6.0
    wall_width_t: float = 1.6
    seed_release_deg: float = 6.0       # an unpaired region narrower than this many t is a cut wall


@dataclass
class Region:
    rid: int
    idx: np.ndarray            # point indices
    plane: Plane               # outward normal (oriented with the point normals)
    area: float
    width: float               # in-plane minor extent (robust)
    length: float              # in-plane major extent (robust)
    flange: int = -1


@dataclass
class Flange:
    fid: int
    regions: list[int]
    idx: np.ndarray
    normal: np.ndarray          # unit, sign of its largest skin
    skins: list[tuple[int, Plane]] = field(default_factory=list)   # (region id, plane with outward normal)
    area: float = 0.0
    two_sided: bool = False
    width: float = 0.0
    length: float = 0.0
    virtual: bool = False       # a riser placed from its step (no flat material of its own): E3 "model plane"


@dataclass
class Scan:
    path: str
    P: np.ndarray
    N: np.ndarray              # raw (area-weighted) vertex normals, outward
    Ns: np.ndarray             # smoothed normals
    nbr: np.ndarray            # kNN indices (excluding self)
    t: float
    t_conf: float
    spacing: float
    point_area: float          # mm^2 represented by one point
    label: np.ndarray          # region id per point, -1 = none
    regions: list[Region]
    flanges: list[Flange]
    flange_of_point: np.ndarray
    timing: dict


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]


def load_scan(path: str, prm: SegParams | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Vertices, vertex normals, t, t confidence, total area, area per point."""
    prm = prm or SegParams()
    import open3d as o3d
    if len(o3d.io.read_triangle_mesh(str(path)).triangles) == 0:
        return load_point_cloud(path, prm, seed)
    mesh = D1.load_mesh(path)
    te = D1.estimate_thickness(mesh, n_samples=min(120_000, max(20_000, len(mesh.faces) // 2)), seed=seed)
    t = float(te.t_mm) if te.t_mm else 1.5
    P, N = mesh.vertices, mesh.vertex_normals
    area = float(mesh.face_areas.sum())
    ok = np.isfinite(P).all(1) & (np.linalg.norm(N, axis=1) > 0.5)
    P, N = P[ok], N[ok]
    if len(P) > prm.max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(P), prm.max_points, replace=False))
        P, N = P[keep], N[keep]
    return P, N, t, float(te.confidence), area, area / len(P)


def load_point_cloud(path: str, prm: SegParams, seed: int = 0):
    """A scan stored without faces: PCA normals, oriented and t measured by probing for the opposite skin.

    For each point the line along +-n is probed at steps of 0.15 mm (0.35 .. 8 mm); the first side that meets
    material (a point within 0.6 spacing whose PCA normal is parallel) is the inside, its distance a thickness
    sample. t is the robust mode of the samples. Points with no hit take the orientation of their oriented
    neighbours (majority, iterated); any left face away from the part centre.
    """
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(path))
    P = np.asarray(pc.points, dtype=np.float64)
    P = P[np.isfinite(P).all(1)]
    rng = np.random.default_rng(seed)
    if len(P) > prm.max_points:
        P = P[np.sort(rng.choice(len(P), prm.max_points, replace=False))]
    tree = cKDTree(P)
    dd, nbr = tree.query(P, k=20)
    spacing = float(np.median(dd[:, 1]))
    Q = P[nbr] - P[nbr].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", Q, Q)
    _, vecs = np.linalg.eigh(cov)
    N = vecs[:, :, 0]
    rho = 0.6 * spacing
    dist = np.full(len(P), np.inf)
    sign = np.zeros(len(P))
    todo = np.ones(len(P), bool)
    for st in np.arange(0.35, 8.0, 0.15):
        idx = np.nonzero(todo)[0]
        if len(idx) == 0:
            break
        for sg in (1.0, -1.0):
            q = P[idx] + sg * st * N[idx]
            d, j = tree.query(q, k=1, distance_upper_bound=rho)
            hit = np.isfinite(d)
            hit[hit] &= np.abs(np.einsum("nd,nd->n", N[j[hit]], N[idx[hit]])) > 0.8
            h = idx[hit & todo[idx]]
            dist[h] = st
            sign[h] = -sg            # material along sg -> outward is -sg
            todo[h] = False
    ok = np.isfinite(dist)
    t = 1.5
    conf = 0.0
    if ok.sum() > 200:
        mode, _, _ = D1.robust_mode(dist[ok], 0.3, 8.0, bin_mm=0.05, smooth_mm=0.1)
        near = np.abs(dist[ok] - mode) < 0.25 * mode
        t = float(np.median(dist[ok][near]))
        conf = float(near.mean())
        bad = ok & (np.abs(dist - t) > 0.4 * t)
        sign[bad] = 0.0
    for _ in range(30):
        un = sign == 0
        if not un.any():
            break
        nb = sign[nbr[un]] * np.sign(np.einsum("nkd,nd->nk", N[nbr[un]], N[un]))
        vote = nb.sum(1)
        upd = np.nonzero(un)[0][vote != 0]
        if len(upd) == 0:
            break
        sign[upd] = np.sign(vote[vote != 0])
    c = P.mean(0)
    rest = sign == 0
    sign[rest] = np.sign(np.einsum("nd,nd->n", P[rest] - c, N[rest])) + (np.einsum("nd,nd->n", P[rest] - c, N[rest]) == 0)
    N = N * sign[:, None]
    # area per point from the spacing (one skin sample per spacing^2)
    pa = spacing ** 2
    return P, N, t, conf, pa * len(P), pa


def smooth_normals(P, N, nbr, iters: int) -> np.ndarray:
    Ns = N.copy()
    for _ in range(iters):
        nb = Ns[nbr]                                     # (n,k,3)
        w = (np.einsum("nkd,nd->nk", nb, Ns) > 0.5).astype(np.float64)
        acc = Ns + np.einsum("nk,nkd->nd", w, nb)
        Ns = _unit_rows(acc)
    return Ns


def _region_geometry(P: np.ndarray, plane: Plane) -> tuple[float, float]:
    c = P.mean(0)
    q = P - c
    q = q - np.outer(q @ plane.n, plane.n)
    if len(q) < 3:
        return 0.0, 0.0
    cov = q.T @ q / len(q)
    vals, vecs = np.linalg.eigh(cov)
    e_major, e_minor = vecs[:, 2], vecs[:, 1]
    a, b = q @ e_major, q @ e_minor
    length = float(np.quantile(a, 0.99) - np.quantile(a, 0.01))
    width = float(np.quantile(b, 0.99) - np.quantile(b, 0.01))
    return width, length


def segment(path: str, prm: SegParams | None = None) -> Scan:
    import time
    prm = prm or SegParams()
    tm = {}
    t0 = time.time()
    P, N, t, tconf, _area, pa = load_scan(path, prm)
    tm["load"] = time.time() - t0
    t1 = time.time()
    tree = cKDTree(P)
    dd, nbr = tree.query(P, k=prm.knn + 1)
    nbr, dd = nbr[:, 1:], dd[:, 1:]
    spacing = float(np.median(dd[:, 0]))
    Ns = smooth_normals(P, N, nbr, prm.smooth_iters)
    tm["normals"] = time.time() - t1
    t1 = time.time()
    n = len(P)
    nb = Ns[nbr]
    cosn = np.einsum("nkd,nd->nk", nb, Ns)
    same = cosn > 0.5
    # flatness: worst same-side neighbour deviation
    dev = np.where(same, cosn, 1.0).min(1)
    flat = dev > math.cos(math.radians(prm.flat_deg))
    off = np.abs(np.einsum("nkd,nd->nk", P[nbr] - P[:, None, :], Ns))
    ok = same & (cosn > math.cos(math.radians(prm.seed_deg))) & (off < prm.seed_offset_mm) \
        & flat[:, None] & flat[nbr]
    label = _components(ok, nbr, np.ones(n, bool), prm.min_seed_points, 0)
    # seeds chain across gentle arcs (a 15 deg joggle) on dense meshes: points off their seed's robust plane are
    # released and re-seeded on their own, a few rounds
    for _ in range(3):
        m = int(label.max()) + 1 if (label >= 0).any() else 0
        planes = _orient(_fit_planes(P, label, m), P, Ns, label)
        rel = np.zeros(n, bool)
        cos_s = math.cos(math.radians(prm.seed_release_deg))
        for r, pl in enumerate(planes):
            if pl is None:
                continue
            idx = np.nonzero(label == r)[0]
            # normal test only: a bowed panel (a degree or two, millimetres off its global plane) stays whole
            bad = Ns[idx] @ pl.n < cos_s
            rel[idx[bad]] = True
        if rel.sum() < prm.min_seed_points:
            break
        label[rel] = -1
        # releasing a riser can disconnect what it joined (the far plate of a joggle): re-split the kept seeds
        same = ok & (label[:, None] == label[nbr]) & (label[:, None] >= 0)
        kept = _components(same, nbr, label >= 0, prm.min_seed_points, 0)
        m2 = int(kept.max()) + 1 if (kept >= 0).any() else 0
        new = _components(ok, nbr, rel, prm.min_seed_points, m2)
        label = np.where(new >= 0, new, kept)
    uniq = np.unique(label[label >= 0])
    remap = -np.ones(int(label.max()) + 2 if len(uniq) else 1, np.int64)
    remap[uniq] = np.arange(len(uniq))
    label = np.where(label >= 0, remap[np.clip(label, 0, None)], -1)
    tm["seeds"] = time.time() - t1
    t1 = time.time()
    planes = _fit_planes(P, label, len(uniq))
    label = _grow(P, Ns, nbr, label, planes, prm)
    planes = _fit_planes(P, label, len(uniq))
    planes = _orient(planes, P, Ns, label)
    label, planes = _merge(P, nbr, label, planes, prm)
    label, planes = _purify(P, Ns, nbr, label, planes, prm)
    tm["grow_merge"] = time.time() - t1
    t1 = time.time()
    regions = []
    for r, pl in enumerate(planes):
        idx = np.nonzero(label == r)[0]
        if pl is None or len(idx) == 0:
            continue
        area = len(idx) * pa
        if area < prm.min_region_area_mm2:
            label[idx] = -1
            continue
        w, L = _region_geometry(P[idx], pl)
        regions.append(Region(len(regions), idx, pl, area, w, L))
    regions = _merge_strips(P, Ns, regions, t)
    label = -np.ones(n, np.int64)
    for reg in regions:
        label[reg.idx] = reg.rid
    flanges, fop = _pair_flanges(P, Ns, regions, t, prm, n)
    tm["flanges"] = time.time() - t1
    return Scan(path, P, N, Ns, nbr, t, tconf, spacing, pa, label, regions, flanges, fop, tm)


def _components(ok, nbr, mask, min_pts, offset):
    """Connected components of the seed edge graph restricted to ``mask`` (labels from ``offset``, -1 small)."""
    n = len(mask)
    k = nbr.shape[1]
    okm = ok & mask[:, None] & mask[nbr]
    ii = np.repeat(np.arange(n), k)[okm.ravel()]
    jj = nbr.ravel()[okm.ravel()]
    g = coo_matrix((np.ones(len(ii), np.int8), (ii, jj)), shape=(n, n))
    ncomp, comp = connected_components(g, directed=False)
    cnt = np.bincount(comp, minlength=ncomp)
    big = (cnt >= min_pts)
    lab = np.where(mask & big[comp], comp, -1)
    uniq = np.unique(lab[lab >= 0])
    remap = -np.ones(ncomp, np.int64)
    remap[uniq] = np.arange(len(uniq)) + offset
    return np.where(lab >= 0, remap[np.clip(lab, 0, None)], -1)


def _fit_planes(P, label, m) -> list[Plane | None]:
    out: list[Plane | None] = []
    order = np.argsort(label, kind="stable")
    lab_sorted = label[order]
    starts = np.searchsorted(lab_sorted, np.arange(m))
    ends = np.searchsorted(lab_sorted, np.arange(m), side="right")
    for r in range(m):
        idx = order[starts[r]:ends[r]]
        if len(idx) < 3:
            out.append(None)
            continue
        sub = idx if len(idx) <= 20000 else idx[:: int(math.ceil(len(idx) / 20000))]
        out.append(robust_plane(P[sub], iterations=3))
    return out


def _orient(planes, P, Ns, label):
    """Orient each plane's normal with its points' outward normals."""
    for r, pl in enumerate(planes):
        if pl is None:
            continue
        idx = np.nonzero(label == r)[0][:2000]
        if len(idx) and float(np.mean(Ns[idx] @ pl.n)) < 0:
            planes[r] = pl.flipped()
    return planes


def _grow(P, Ns, nbr, label, planes, prm: SegParams) -> np.ndarray:
    m = len(planes)
    nvec = np.array([pl.n if pl is not None else np.zeros(3) for pl in planes]) if m else np.zeros((0, 3))
    dvec = np.array([pl.d if pl is not None else 0.0 for pl in planes]) if m else np.zeros(0)
    # orient normals with the points
    for r, pl in enumerate(planes):
        if pl is None:
            continue
        idx = np.nonzero(label == r)[0][:2000]
        if len(idx) and float(np.mean(Ns[idx] @ nvec[r])) < 0:
            nvec[r], dvec[r] = -nvec[r], -dvec[r]
    cos_g = math.cos(math.radians(prm.grow_deg))
    label = label.copy()
    for _ in range(prm.grow_iters):
        un = np.nonzero(label < 0)[0]
        if len(un) == 0 or m == 0:
            break
        nl = label[nbr[un]]                           # (u,k)
        has = (nl >= 0).any(1)
        un, nl = un[has], nl[has]
        if len(un) == 0:
            break
        best = -np.ones(len(un), np.int64)
        bestd = np.full(len(un), np.inf)
        for k in range(nl.shape[1]):
            r = nl[:, k]
            v = r >= 0
            rr = np.clip(r, 0, None)
            dist = np.abs(np.einsum("nd,nd->n", P[un], nvec[rr]) - dvec[rr])
            cs = np.einsum("nd,nd->n", Ns[un], nvec[rr])
            good = v & (dist < prm.grow_dist_mm) & (cs > cos_g) & (dist < bestd)
            best[good] = r[good]
            bestd[good] = dist[good]
        add = best >= 0
        if not add.any():
            break
        label[un[add]] = best[add]
    return label


def _purify(P, Ns, nbr, label, planes, prm: SegParams):
    """Seeds can chain across a smooth arc (dense meshes) or along a rolled surface. Points off their region's
    robust plane (beyond the growth tolerances) are released; a region that keeps under half its points is a
    curved surface, not a planar skin, and is released whole. Released points may be re-grown by other regions."""
    cos_g = math.cos(math.radians(prm.grow_deg))
    changed = False
    for r, pl in enumerate(planes):
        if pl is None:
            continue
        idx = np.nonzero(label == r)[0]
        if len(idx) == 0:
            continue
        n = pl.n if float(np.mean(Ns[idx[:2000]] @ pl.n)) >= 0 else -pl.n
        good = Ns[idx] @ n > cos_g
        if good.mean() < 0.5:
            label[idx] = -1
            changed = True
        elif not good.all():
            label[idx[~good]] = -1
            changed = True
    if changed:
        planes = _fit_planes(P, label, len(planes))
        planes = _orient(planes, P, Ns, label)
        label = _grow(P, Ns, nbr, label, planes, prm)
        planes = _fit_planes(P, label, len(planes))
        planes = _orient(planes, P, Ns, label)
    return label, planes


def _merge(P, nbr, label, planes, prm: SegParams):
    m = len(planes)
    if m == 0:
        return label, planes
    n = len(P)
    li = np.repeat(label, nbr.shape[1])
    lj = label[nbr.ravel()]
    ok = (li >= 0) & (lj >= 0) & (li != lj)
    a, b = np.minimum(li[ok], lj[ok]), np.maximum(li[ok], lj[ok])
    key, cnt = np.unique(a * m + b, return_counts=True)
    parent = np.arange(m)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    cos_m = math.cos(math.radians(prm.merge_deg))
    cents = {}
    for r, pl in enumerate(planes):
        if pl is not None:
            idx = np.nonzero(label == r)[0]
            cents[r] = P[idx].mean(0) if len(idx) else None
    for k, c in zip(key, cnt):
        x, y = int(k // m), int(k % m)
        px, py = planes[x], planes[y]
        if px is None or py is None or c < 3:
            continue
        cs = float(px.n @ py.n)
        if cs < cos_m:
            continue
        # same side: normals agree (sign of the fitted normals is arbitrary -> use the points)
        cx, cy = cents.get(x), cents.get(y)
        if cx is None or cy is None:
            continue
        if abs(float(px.distance(cy[None])[0])) > prm.merge_dist_mm or abs(float(py.distance(cx[None])[0])) > prm.merge_dist_mm:
            continue
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    roots = np.array([find(x) for x in range(m)])
    uniq, newid = np.unique(roots, return_inverse=True)
    label = np.where(label >= 0, newid[np.clip(label, 0, None)], -1)
    planes = _fit_planes(P, label, len(uniq))
    return label, planes


def _strip_axis(Q, n):
    q = Q - Q.mean(0)
    q = q - np.outer(q @ n, n)
    _, vecs = np.linalg.eigh(q.T @ q)
    return vecs[:, 2]


def _merge_strips(P, Ns, regions: list[Region], t: float) -> list[Region]:
    """Narrow strips (a short web or riser) break into collinear fragments at scan dropouts, and a strip's plane
    is poorly fixed about its long axis. Same-facing narrow fragments that are collinear (axes within 5 deg, lateral
    and out-of-plane offsets under 1 mm / 0.6 mm) with an axial gap under 6 mm are joined into one region."""
    narrow = [r for r in regions if r.width < 4.0 * t + 2.0]
    if len(narrow) < 2:
        return regions
    info = {}
    for r in narrow:
        Q = P[r.idx]
        info[r.rid] = (Q.mean(0), _strip_axis(Q, r.plane.n), Q)
    parent = {r.rid: r.rid for r in narrow}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(narrow):
        ca, ua, Qa = info[a.rid]
        za = (Qa - ca) @ ua
        for b in narrow[i + 1:]:
            if float(a.plane.n @ b.plane.n) < math.cos(math.radians(10.0)):
                continue
            cb, ub, Qb = info[b.rid]
            if abs(float(ua @ ub)) < math.cos(math.radians(5.0)):
                continue
            dv = cb - ca
            lat = dv - ua * float(dv @ ua)
            if abs(float(lat @ a.plane.n)) > 0.6 or float(np.linalg.norm(lat - a.plane.n * float(lat @ a.plane.n))) > 1.0:
                continue
            zb = (Qb - ca) @ ua
            gap = max(float(zb.min() - za.max()), float(za.min() - zb.max()))
            if gap > 6.0:
                continue
            ra, rb = find(a.rid), find(b.rid)
            if ra != rb:
                parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for r in narrow:
        groups.setdefault(find(r.rid), []).append(r.rid)
    if all(len(g) == 1 for g in groups.values()):
        return regions
    byid = {r.rid: r for r in regions}
    out: list[Region] = []
    done = set()
    for r in regions:
        if r.rid in done:
            continue
        g = groups.get(find(r.rid)) if r.rid in parent else None
        if g and len(g) > 1:
            idx = np.concatenate([byid[k].idx for k in g])
            sub = idx if len(idx) <= 20000 else idx[:: int(math.ceil(len(idx) / 20000))]
            pl = robust_plane(P[sub], iterations=3)
            if float(np.mean(Ns[idx[:2000]] @ pl.n)) < 0:
                pl = pl.flipped()
            w, L = _region_geometry(P[idx], pl)
            out.append(Region(len(out), idx, pl, float(sum(byid[k].area for k in g)), w, L))
            done.update(g)
        else:
            out.append(Region(len(out), r.idx, r.plane, r.area, r.width, r.length))
            done.add(r.rid)
    return out


def _pair_flanges(P, Ns, regions: list[Region], t: float, prm: SegParams, n: int):
    """Union antiparallel skin regions one thickness apart into flanges; drop cut walls."""
    m = len(regions)
    # orient each region's plane with its points' normals
    for reg in regions:
        if float(np.mean(Ns[reg.idx[:3000]] @ reg.plane.n)) < 0:
            reg.plane = reg.plane.flipped()
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    paired = np.zeros(m, bool)
    cos_p = math.cos(math.radians(prm.pair_deg))
    samples = []
    for reg in regions:
        s = reg.idx if len(reg.idx) <= 4000 else reg.idx[np.linspace(0, len(reg.idx) - 1, 4000).astype(int)]
        samples.append(s)
    trees = [cKDTree(P[s]) for s in samples]
    for i in range(m):
        for j in range(i + 1, m):
            a, b = regions[i], regions[j]
            if float(a.plane.n @ b.plane.n) > -cos_p:
                continue
            # b's points should sit about t behind a's plane (inside the material: negative distance)
            db = a.plane.distance(P[samples[j]])
            near = np.abs(db + t) < max(0.35 * t, 0.3)
            if near.sum() < 20:
                continue
            # footprint overlap: b points within ~t of a's points must sit one thickness behind a's plane
            # (a joggle's offset plate touches a only around the riser, at other depths)
            dist, _ = trees[i].query(P[samples[j]], k=1, distance_upper_bound=1.6 * t + 1.0)
            close = np.isfinite(dist)
            ov = close & near
            if ov.sum() < max(20, 0.05 * min(len(samples[i]), len(samples[j]))) or ov.sum() < 0.6 * close.sum():
                continue
            paired[i] = paired[j] = True
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    flanges: list[Flange] = []
    fop = -np.ones(n, np.int64)
    for root, members in groups.items():
        two = any(paired[i] for i in members)
        big = max(members, key=lambda i: regions[i].area)
        if not two:
            reg = regions[big]
            if reg.width < prm.wall_width_t * t:
                continue                              # cut wall / narrow edge band
        idx = np.concatenate([regions[i].idx for i in members])
        f = Flange(len(flanges), list(members), idx, regions[big].plane.n.copy(),
                   [(i, regions[i].plane) for i in members], float(sum(regions[i].area for i in members)), two,
                   regions[big].width, regions[big].length)
        for i in members:
            regions[i].flange = f.fid
        fop[idx] = f.fid
        flanges.append(f)
    return flanges, fop
