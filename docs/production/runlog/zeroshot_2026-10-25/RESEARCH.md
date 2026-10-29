# Zero-shot bend-line detection for press-brake sheet metal: research notes (2026-10-25)

Scope: research only. No engine edits, no installs, and no forbidden data opened (HOLDOUT10, N001–N016, EHO4, *_SEALED*, face scans).

Target: line-F1 >= 0.85 on 40+ real scans. A predicted line matches when it is within 5 deg and 4 mm of a gold line and the span overlap is >= 0.6. Current no-CAD lane: about 0.39. CAD-assisted per-flange plane fit: about 0.94.

## 0. Where we are (what the repo does today)

**Lanes** (`research/broad_baseline/run_lane.py`)
- `best_f1` is the union of:
  - the A0 strict-blind scan-only typed union (`backend/patch_graph_latent_decomposition.py` → `scripts/run_f1_structured_union_e2e.py`, CP-SAT in `.venv-f1-structured`);
  - CAD+PLY runtime detections.
- Without CAD it degrades to A0 only.

**Scan-only detectors**
- `backend/feature_detection/bend_detector.py::_segment_planes` uses iterative (sequential, greedy) RANSAC on a voxelized cloud. It is multi-scale, merges coplanar planes, and builds bends from adjacent plane pairs (intersection line, Taubin circle for radius).
- `backend/zero_shot_bend_profile.py` (3.2k lines) adds policy scoring on top: orthogonal-dense, repeated-90, rolled/mixed, perturbation stability, and many count "rescues".
- `patch_graph_latent_decomposition.py` (10.4k lines) builds:
  - a surfel atom graph and transition-atom graph;
  - flange and bend hypotheses;
  - a CP-SAT typed labelling;
  - many junction/corridor/birth recovery passes.
- No module models sheet **thickness / the two skins** explicitly. A grep for thickness, mid-surface or double-wall in the zero-shot detectors finds nothing.

**Error budget** (`docs/production/runlog/broad_baseline/BROAD_BASELINE_SUMMARY.json`, best_f1 lane, 57 scans, 305 gold lines)

| Cause | Count |
|---|---|
| FN: MISSED | 116 |
| FN: SPAN_LT_60PCT | 40 |
| FN: OFFSET_GT_4MM | 32 |
| FN: ANGLE_GT_5 | 2 |
| FP: SPURIOUS | 47 |
| FP: span | 24 |
| FP: offset | 23 |

Angles are already good (MAE 0.56 deg when matched). Recall, span and outer-side offset are the problem. Angle is not.

**Python env** (`/Users/idantal/ShermanQC/FullProject/.venv`, checked, nothing installed)
- Present: numpy 2.4.6, scipy 1.17.1, open3d 0.19.0, scikit-learn 1.9.0, trimesh 4.12.2, onnxruntime 1.27.0.
- `.venv-f1-structured` has ortools 9.15 (CP-SAT).
- MISSING: torch, networkx, scikit-image, cv2, numba, pymeshlab, pygco/maxflow, hdbscan, cvxpy.
- Useful built-ins that do exist:
  - `o3d.geometry.PointCloud.detect_planar_patches` (robust-statistics planar patches; see D7);
  - `segment_plane`, `cluster_dbscan`, `estimate_covariances`;
  - `o3d.t.geometry.PointCloud.compute_boundary_points`;
  - `o3d.t.geometry.RaycastingScene`;
  - `scipy.sparse.csgraph.maximum_flow`, so α-expansion min-cuts are possible without gco;
  - `scipy.optimize.linear_sum_assignment`.

**Literature finding:** there is very little published work on bend-line recovery from scans of *formed sheet metal* specifically:
- a Chinese patent that slices normal to the bend and fits two lines (CN106295058A);
- an Aalborg MSc thesis;
- roller-forming in-process measurement.

The general machinery (multi-model fitting, crease detection, B-rep reverse engineering) is mature. The sheet-metal structure (piecewise-planar, straight cylindrical bends, constant thickness t, developable) is a strong prior that generic methods do not use. Every direction below exploits that structure.

### Geometry fact that drives direction D1

Two skins are offset by t. For included angle θ, their virtual-sharp apex lines are separated along the bisector by

    d = t / sin(θ/2)

| θ | d |
|---|---|
| 90 deg | 1.41 t |
| 60 deg | 2 t |
| 30 deg | 3.9 t |

So for t >= 2.9 mm at 90 deg, or t >= 1 mm at 30 deg, reporting the inner-skin apex instead of the outer one fails the 4 mm gate by itself. This is a plausible root cause for a large part of the 32 OFFSET FNs and 23 OFFSET FPs. It also explains "doubled" lines: one line per skin, where one is a TP and the other is SPURIOUS or OFFSET.

---

## Ranked directions

### D1. Thickness-aware two-skin flange decomposition (mid-surface / skin pairing). Rank 1

**Method**
1. Orient normals: mesh normals or the scanner viewpoint. For point clouds, use `orient_normals_consistent_tangent_plane`, then flip outward against the part's convex hull.
2. For each surfel, cast a ray along −n (`RaycastingScene` on the mesh) or query a thin cylinder in a KD-tree (point clouds). Look for an antiparallel surfel at distance in [0.3, 8] mm.
3. The robust mode of those distances is **t**. A second, independent estimate comes from the width of the narrow cut-edge band whose normals are perpendicular to the flange.
4. Pair the skins. Assign every surfel a skin label (outer/inner) per flange and a "two-sided / one-sided" flag.
5. Collapse to a mid-surface (x − n·t/2 for outer, x + n·t/2 for inner) for **segmentation**: one sheet, not two parallel planes 0.5–6 mm apart.
6. Then **report** the line on the **outer** skin:
   - fit planes to each flange's outer skin, if seen;
   - otherwise use the mid-plane offset by t/2 toward the convex side of the bend;
   - intersect adjacent outer planes.
7. Outer = convex side of the bend cylinder. Its normals diverge along the profile.

**Key references**
- Woo, "Abstraction of mid-surfaces from solid models of thin-walled parts: a divide-and-conquer approach", CAD 2014. https://doi.org/10.1016/j.cad.2013.08.010
- MidSurfer (2024), parameter-free mid-surface extraction (volumetric, but the pairing logic transfers). https://arxiv.org/abs/2405.19339
- Thin-surface laser-scan reconstruction pitfalls (TSDF collapse of thin walls). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014519/
- Open3D RaycastingScene. https://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html

**Why it fits.** Constant t is the strongest invariant of sheet metal. Today's RANSAC and atom-graph lanes treat the two skins as independent planes. That creates:
- parallel near-duplicate planes, which confuse adjacency and the bend count;
- apex lines on the wrong skin (the OFFSET failures);
- hems and joggles that look like noise.

**Failure modes**
- One-sided scans (fixture side unseen): t cannot be measured locally. Fall back to the global t, then a CAD prior.
- Thin sheets (0.5–1 mm) at about 0.3–0.5 mm scanner noise: skins are inseparable. That is fine, because the offset error is then < 1.5 mm and below the gate anyway.
- Hems: four parallel skins within 2t. The pairing must be nearest-antiparallel, not nearest.
- Weld plates and studs locally double the thickness. Reject pairs where d > 1.5 t_mode.

**CPU:** good. Raycasting is about 1–3 M rays/s on CPU in Open3D. A KD-tree on 200k points is under 1 s.

**Licence:** own code (Open3D MIT, SciPy BSD).

**Sketch**
```python
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree

def estimate_thickness(mesh_legacy, n_samples=60000, dmin=0.3, dmax=8.0):
    m = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(m)
    pcd = mesh_legacy.sample_points_uniformly(n_samples); pcd.estimate_normals()
    P, N = np.asarray(pcd.points), np.asarray(pcd.normals)
    # mesh normals point outward from material; probe inward
    rays = np.hstack([P - 1e-3 * N, -N]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    d = hit['t_hit'].numpy(); fn = hit['primitive_normals'].numpy()
    ok = (d > dmin) & (d < dmax) & ((fn * N).sum(1) > 0.85)   # opposite skin's outward normal is antiparallel to the probe dir => parallel to -(-N)
    t = np.median(d[ok]) if ok.sum() > 200 else None
    two_sided = np.zeros(len(P), bool); two_sided[ok] = np.abs(d[ok] - t) < 0.35 * t
    mid = P.copy(); mid[two_sided] -= 0.5 * d[two_sided, None] * N[two_sided]
    return t, P, N, two_sided, mid

def outer_apex(plane_a, plane_b, t, convex_sign_a, convex_sign_b):
    """plane = (n, c) with n·x = c, n outward. Offset each mid-plane to its outer skin then intersect."""
    (na, ca), (nb, cb) = plane_a, plane_b
    ca, cb = ca + 0.5 * t, cb + 0.5 * t          # mid-plane -> outer skin along outward normal
    u = np.cross(na, nb); u /= np.linalg.norm(u)
    A = np.vstack([na, nb, u]); x0 = np.linalg.solve(A, np.array([ca, cb, 0.0]))
    return x0, u
```
Integration: run D1 as a pre-pass producing `mid` points and per-point `skin`/`two_sided` labels. Feed `mid` into the existing BendDetector or atom graph, so segmentation sees one sheet. Do the final line placement on the outer skin.

---

### D2. Axis-family cross-section profile analysis ("slice and polyline"). Rank 2

**Method.** Press-brake bends are straight lines, and most parts have only a few bend-axis direction families.

1. **Find axis families.** Bend axes are perpendicular to the normals of both adjacent flanges. Cluster normals on the Gauss sphere (mean-shift or k-means on S²; no new libraries needed), take pairwise cross products of the dominant normal clusters, and cluster those directions.
   - Alternatively, take the great circles traced by bend-zone normals: points on a cylinder have normals ⊥ axis, so the axis is the smallest-eigenvalue eigenvector of the bend-zone normals' scatter matrix. This is Gauss-map axis estimation as used for cylinder detection.
2. **Slice.** For each axis family u, slice the cloud with planes ⊥ u every 2–5 mm. Project each slice to 2D. That gives a 2D polyline of the profile, with two skins = two parallel polylines t apart.
3. **Fit the 2D profile.** Use 2D line RANSAC/Douglas–Peucker on the outer chain, robust to holes because slices through holes simply lose segments. Vertices = bends; the turn angle gives the included angle.
4. **Track across slices.** Link vertices across consecutive slices (1D Hungarian on vertex position + angle). The tracked run = the bend line's extent. Gaps in the run from holes or studs are bridged when they are shorter than k slices.
5. Fit a 3D line to the tracked apex points (total least squares). The span = first-to-last slice with support.

**Key references**
- CN106295058A: section-normal slicing + two-line fit for bend angle. https://eureka.patsnap.com/patent-CN106295058A
- Aalborg thesis, "Computer-Vision Based Bend Quality Inspection for Sheet Metal Parts". https://projekter.aau.dk/computer-vision-based-bend-quality-inspection-for-sheet-metal-parts-06fddb1d.html
- Gauss map for cylinder axes: Chaperon & Goulette, "Extracting cylinders in full 3D data using a random sampling method and the Gaussian image", VMV 2001.
- Rabbani et al., "Efficient Hough transform for automatic detection of cylinders in point clouds", ISPRS 2005. https://www.isprs.org/proceedings/xxxvi/3-w19/papers/060.pdf

**Why it fits**
- It turns a 3D segmentation problem into many 1D/2D problems where joggles (two close opposite bends), small flanges (2–3 points per slice still define a segment) and hems (a 180 deg U-turn in the profile) are explicit and easy.
- Span comes directly from slice support, which attacks SPAN_LT_60PCT.
- Short bends on small flanges become visible because they are tested along the known axis family, not discovered by global RANSAC. That attacks MISSED.

**Failure modes**
- Parts whose bend axes are not few families (box parts with many directions are fine; freeform or rolled parts are not). Restrict each family's slicing to the bounding region of the flanges that define it.
- Slices that cut obliquely through a perpendicular bend family produce spurious vertices. Require the vertex turn axis ≈ u (check with the 3D normals of the vertex's neighbours).
- Partial (mid-process) parts are fine, because it is local.
- Large bend radius (rolled) breaks the vertex model. Detect this as a sequence of small turns and merge them.

**CPU:** excellent. O(N log N) per family, about 1 s for 200k points.

**Licence:** own code.

**Sketch**
```python
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

def axis_families(N, flange_normals, ang_tol=4.0):
    cands = []
    for i in range(len(flange_normals)):
        for j in range(i + 1, len(flange_normals)):
            c = np.cross(flange_normals[i], flange_normals[j]); s = np.linalg.norm(c)
            if s > np.sin(np.radians(8)): cands.append(c / s)
    cands = np.array(cands); cands *= np.sign(cands @ cands[0])[:, None]   # projective
    # greedy angular clustering
    fam = []
    for c in cands:
        if not any(abs(c @ f) > np.cos(np.radians(ang_tol)) for f in fam): fam.append(c)
    return fam

def profile_vertices_2d(Q, tol, min_turn_deg=8):
    """Q: (m,2) slice points (outer chain). Order by a 1-D embedding (PCA of a kNN graph) then DP-simplify."""
    order = np.argsort(Q @ np.linalg.svd(Q - Q.mean(0))[2][0]); Q = Q[order]
    def dp(a, b):
        if b <= a + 1: return [a, b]
        d = np.abs(np.cross(Q[b] - Q[a], Q[a + 1:b] - Q[a])) / (np.linalg.norm(Q[b] - Q[a]) + 1e-9)
        k = int(np.argmax(d)); return dp(a, a + 1 + k)[:-1] + dp(a + 1 + k, b) if d[k] > tol else [a, b]
    idx = dp(0, len(Q) - 1); V = []
    for p, q, r in zip(idx, idx[1:], idx[2:]):
        e1, e2 = Q[q] - Q[p], Q[r] - Q[q]
        turn = np.degrees(np.arccos(np.clip(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2), -1, 1)))
        if turn > min_turn_deg: V.append((Q[q], 180 - turn))     # apex, included angle
    return V

def track(slices_vertices, step, max_gap=3, pos_tol=3.0, ang_tol=6.0):
    """slices_vertices: list over slice index of [(xy, angle)]. Returns runs = bend lines."""
    runs, active = [], []
    for s, verts in enumerate(slices_vertices):
        if active and verts:
            C = np.array([[np.linalg.norm(r['xy'] - v[0]) + abs(r['ang'] - v[1]) for v in verts] for r in active])
            ri, vi = linear_sum_assignment(C)
            used = set()
            for a, b in zip(ri, vi):
                if np.linalg.norm(active[a]['xy'] - verts[b][0]) < pos_tol and abs(active[a]['ang'] - verts[b][1]) < ang_tol:
                    active[a]['pts'].append((s, verts[b][0])); active[a]['xy'] = verts[b][0]; active[a]['last'] = s; used.add(b)
            new = [v for k, v in enumerate(verts) if k not in used]
        else: new = verts
        for v in new: active.append({'xy': v[0], 'ang': v[1], 'pts': [(s, v[0])], 'last': s})
        runs += [r for r in active if s - r['last'] > max_gap]; active = [r for r in active if s - r['last'] <= max_gap]
    return runs + active
```
Each run becomes a 3D line `x = o + s*step*u + xy[0]*e1 + xy[1]*e2`, fitted with SVD. Keep runs with length > 15 mm and support > 60 % of slices in the span.

---

### D3. Global multi-model energy fitting (PEARL / Multi-X / Progressive-X style), replacing sequential RANSAC. Rank 3

**Method.** Replace greedy "fit best plane, remove inliers, repeat" with a global labelling.

1. **Proposals.** Thousands of plane and cylinder hypotheses from:
   - RANSAC, Schnabel-style octree-local sampling;
   - region-growing seeds;
   - D7 patches;
   - D2 slices.
2. **Energy.** E = Σ data (point-to-model distance, normal deviation, truncated at about 3σ) + λ·Σ smoothness (Potts on the kNN graph, weighted by normal similarity) + h·|labels| (label cost/MDL).
3. **Optimise** with α-expansion (min-cut via `scipy.sparse.csgraph.maximum_flow` on integer capacities), or ICM + label-cost greedy as a first version. Iterate: re-fit models to their labelled points, re-propose, re-solve (PEARL).
4. Use cylinders with **axis constrained ⊥ both neighbour plane normals** for bend zones.

Small flanges survive because label cost is per model, not a minimum-inlier threshold, and smoothness stops one plane from swallowing a coplanar distant flange or a bend strip.

**Key references**
- Isack & Boykov, "Energy-based geometric multi-model fitting" (PEARL), IJCV 2012. https://doi.org/10.1007/s11263-011-0474-7
- Delong, Osokin, Isack, Boykov, "Fast approximate energy minimization with label costs", IJCV 2012. https://doi.org/10.1007/s11263-011-0437-z
- Barath & Matas, Multi-X (ICCV 2017) and Progressive-X (ICCV 2019). https://arxiv.org/abs/1906.02290. Code (MIT, C++ with a Python binding that needs compiling, so not usable here without an install): https://github.com/danini/progressive-x
- Barath & Matas, Graph-Cut RANSAC (CVPR 2018), BSD. https://github.com/danini/graph-cut-ransac
- Schnabel, Wahl, Klein, "Efficient RANSAC for point-cloud shape detection", CGF 2007. https://doi.org/10.1111/j.1467-8659.2007.01016.x
- Rabbani et al., "Segmentation of point clouds using smoothness constraint", ISPRS 2006 (region growing). https://www.isprs.org/proceedings/xxxvi/part5/paper/RABB_639.pdf
- Vo et al., "Octree-based region growing for point cloud segmentation" (OBR), ISPRS J. 2015. https://doi.org/10.1016/j.isprsjprs.2015.01.011
- Limberger & Oliveira, "Real-time detection of planar regions in unorganized point clouds" (kernel Hough), PR 2015. https://doi.org/10.1016/j.patcog.2014.12.020

**Why it fits.** The existing sequential RANSAC has known failure modes on sheet metal:
- the first big plane steals inliers from coplanar or parallel flanges (two skins!) and from shallow bends;
- small flanges fall below the minimum-inlier count;
- the greedy order is unstable. This is why the repo needed perturbation-stability rescues.

A joint energy with label costs is the standard fix. It gives one principled knob (h) that CAD can set (D5).

**Failure modes**
- λ and h tuning. Calibrate h in units of "points a 5 mm × 20 mm flange would have at current density".
- Two skins still produce two labels unless D1's mid-surface is used. Run D3 on `mid`.
- Runtime grows with the number of proposals. Prune to about 300 proposals after a first pass.

**CPU:** OK. ICM/label-cost greedy on 100k points × 300 labels is seconds with vectorised numpy. Python α-expansion with scipy maxflow is about 0.2–1 s per expansion; restrict expansions to labels touching the current boundary.

**Licence:** own code. Progressive-X/GC-RANSAC are MIT/BSD if compiled later.

**Sketch** (label-cost ICM; α-expansion later)
```python
import numpy as np
from scipy.spatial import cKDTree

def data_cost(P, N, models, sigma, ang_sigma_deg=10):
    C = np.empty((len(P), len(models)), np.float32)
    for k, (kind, prm) in enumerate(models):
        if kind == 'plane':
            n, c = prm; r = P @ n - c; ca = np.abs(N @ n)
        else:                                     # cylinder: axis point a, dir u, radius R
            a, u, R = prm; v = P - a; v -= (v @ u)[:, None] * u; rad = np.linalg.norm(v, axis=1)
            r = rad - R; ca = np.abs((N * (v / rad[:, None])).sum(1))
        C[:, k] = np.minimum((r / sigma) ** 2 + ((1 - ca) / (1 - np.cos(np.radians(ang_sigma_deg)))) ** 2, 9.0)
    return C

def fit_labels(P, N, models, sigma, lam=1.0, h=None, k=10, iters=8):
    nbr = cKDTree(P).query(P, k + 1)[1][:, 1:]
    w = np.clip((N[:, None, :] * N[nbr]).sum(-1), 0, 1) ** 4            # weak smoothing across creases
    D = data_cost(P, N, models, sigma); L = np.argmin(D, 1)
    h = h if h is not None else 60.0
    for _ in range(iters):
        counts = np.bincount(L, minlength=len(models))
        for kk in np.where((counts > 0) & (counts * 1.0 < h))[0]:       # label-cost greedy removal
            m = L == kk; D2 = D[m].copy(); D2[:, kk] = np.inf; L[m] = np.argmin(D2, 1)
        S = np.zeros_like(D)
        for j in range(k):
            np.add.at(S, (np.arange(len(P)), L[nbr[:, j]]), -lam * w[:, j])
        L = np.argmin(D + S, 1)
        # refit (PEARL step): least-squares per label (plane: SVD, cylinder: axis-constrained circle fit)
    return L
```

---

### D4. Explicit bend-zone (cylinder-strip) model for line, angle and span. Rank 4

**Method.** Treat each bend as a cylindrical strip between two flanges.

1. Take a candidate flange pair (A, B) that is adjacent in the region graph (D1/D3) or from a D2 run. Set axis u = n_A × n_B.
2. Collect the points between the flanges whose normals lie in the plane ⊥ u and between n_A and n_B (the Gauss-map great-circle arc).
3. Fit a 2D circle (Taubin, already in `bend_detector.py`) in the plane ⊥ u to the outer-skin points. This gives inner radius r and the outer radius r + t.
4. The line to report is the **outer virtual sharp**: the intersection of the outer-skin planes of A and B, as in D1.
5. The **span** is the extent along u of the strip points, plus the extent of the shared boundary of A and B projected on u. Use the union with gaps < 10 mm bridged, because holes near bends are common.
6. The included angle comes from the plane normals, which are already accurate.

Also add a per-pair "strip existence" test: the fraction of slices along u with great-circle-arc normals. This removes SPURIOUS lines between flanges that are merely adjacent across a cut edge or step.

**Key references**
- Chaperon & Goulette (Gaussian image cylinders, 2001).
- Schnabel 2007 (cylinder primitive with normals).
- Lukács, Martin, Marshall, "Faithful least-squares fitting of spheres, cylinders, cones and tori for reliable segmentation", ECCV 1998. https://doi.org/10.1007/BFb0055704
- Taubin, "Estimation of planar curves, surfaces…", PAMI 1991. https://doi.org/10.1109/34.103273
- Point2CAD's surface-intersection-for-edges idea. https://arxiv.org/abs/2312.04962

**Why it fits**
- Press-brake bends are exactly circular-arc strips with a straight axis. The strip is the most direct evidence that a bend exists, and it is distinct from a sharp cut-edge or a weld seam.
- It gives the span from the evidence (attacks SPAN_LT_60PCT) and a principled FP test (attacks SPURIOUS).

**Failure modes**
- Tiny radius on thin sheet (r ≈ t ≈ 0.5 mm) gives fewer than 3 points across the strip. Fall back to plane intersection + boundary adjacency.
- Partial bends (mid-process, springback) are still cylindrical, which is fine.
- Hems: θ ≈ 0 means n_A ≈ −n_B, so u from the cross product is unstable. Get u from the hem's U-turn edge direction (the PCA of the fold points) instead.

**CPU:** trivial.

**Licence:** own code.

**Sketch**
```python
def bend_strip(P, N, nA, nB, region_mask, slice_step=3.0, arc_tol_deg=12):
    u = np.cross(nA, nB); u /= np.linalg.norm(u)
    # normals in plane ⊥ u and angularly between nA and nB
    perp = np.abs(N @ u) < np.sin(np.radians(arc_tol_deg))
    a, b = np.arctan2(N @ np.cross(u, nA), N @ nA), np.arctan2(nB @ np.cross(u, nA), nB @ nA)
    between = (np.sign(b) * a > -0.1) & (np.abs(a) < np.abs(b) + 0.1)
    S = P[region_mask & perp & between]
    s = S @ u; bins = np.floor((s - s.min()) / slice_step).astype(int)
    occ = np.bincount(bins) > 2
    # span = longest run of occupied bins allowing gaps <= 3 bins
    runs, start, gap = [], None, 0
    for i, o in enumerate(occ):
        if o: start = i if start is None else start; gap = 0
        elif start is not None:
            gap += 1
            if gap > 3: runs.append((start, i - gap)); start, gap = None, 0
    if start is not None: runs.append((start, len(occ) - 1 - gap))
    lo, hi = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, -1)
    support = occ[lo:hi + 1].mean() if hi >= lo else 0.0
    return u, s.min() + lo * slice_step, s.min() + (hi + 1) * slice_step, support
```

---

### D5. CAD as priors: constrained fitting + flange-graph matching (registration-free). Rank 5

The ranking is for zero-shot F1. For the CAD-available lane this is the highest-value item.

**Method**
1. **Parse CAD into a flange graph.** Nodes = planar faces grouped into flanges, with area, outline extents and normal. Edges = bends, with included angle, inner radius, bend length and t. `cad_bend_extractor.py` already has most of this.
2. **Detected flange graph** from D1/D3/D4: nodes = flanges with area, extents and normal; edges = bends with angle, length and strip support.
3. **Attributed graph matching.** Use spectral matching (Leordeanu–Hebert) as a fast numpy initialiser. Use exact QAP / maximum common subgraph with CP-SAT (ortools exists in `.venv-f1-structured`) for ≤ 25 nodes. Allow unmatched nodes, because partial and mid-process parts are subgraphs of CAD and unbent bends are θ = 180.
   - Unary affinity: area ratio, extent ratios, number of bends.
   - Pairwise affinity: angle between normals vs CAD dihedral (rotation-invariant), bend length, distances between flange centroids.
4. **Feed the match back into fitting.** It replaces CAD registration as the source of the per-flange plane fit that gives 0.94:
   - label cost per CAD flange (D3 h) lowered for expected flanges;
   - an exact or bounded bend-count prior (min/max over process states);
   - angle snapping windows;
   - known t for D1 on one-sided scans;
   - flange-size gates that reject tiny spurious planes;
   - search restricted to axis families present in CAD (D2).
5. **Constrained multi-model fit.** Fit all flange planes jointly with the CAD dihedral constraints as soft penalties: a small nonlinear least-squares with `scipy.optimize.least_squares` over plane parameters.

**Key references**
- Leordeanu & Hebert, "A spectral technique for correspondence problems using pairwise constraints", ICCV 2005. https://doi.org/10.1109/ICCV.2005.20
- Cho, Lee, Lee, "Reweighted random walks for graph matching", ECCV 2010. https://doi.org/10.1007/978-3-642-15555-0_36
- Zhou & De la Torre, "Factorized graph matching", CVPR 2012. https://doi.org/10.1109/CVPR.2012.6247777
- Li et al., GlobFit, "Consistently fitting primitives by discovering global relations", SIGGRAPH 2011 (relations as constraints on primitives). https://doi.org/10.1145/2010324.1964947
- Ma & Yang, sheet-metal B-rep face-edge graph features (GNN), Sci. Rep. 2024. https://www.nature.com/articles/s41598-024-61443-2

**Why it fits**
- The CAD lane gets 0.94 because it knows which flanges exist. Graph matching recovers that knowledge without needing a good global registration, which fails on partial or mid-process parts.
- The graph is tiny (≤ 21 nodes), so exact solving is cheap.
- Sheet-metal attributes (dihedrals, t, bend lengths) are rotation-invariant and highly discriminative.

**Failure modes**
- Symmetric parts give multiple equal matches. That is harmless for line-F1 if matches are equivalent, but break ties with a hole/stud pattern if needed.
- Mid-process parts: unbent CAD bends show up as coplanar flanges merged into one detected flange. Allow 1-to-many node merges, i.e. CAD flanges joined by a "flat" bend.
- A bad detected graph (merged skins) makes the matching garbage-in. D1 has to come first.

**CPU:** spectral matching is ms. CP-SAT is under 1 s for these sizes.

**Licence:** own code; ortools Apache-2.0.

**Sketch**
```python
def spectral_match(A_nodes, B_nodes, A_edges, B_edges, unary, pair, iters=50):
    """A_edges/B_edges: dict (i,j)->attr. unary(i,a)->[0,1], pair((i,j),(a,b))->[0,1]."""
    cand = [(i, a) for i in range(len(A_nodes)) for a in range(len(B_nodes)) if unary(i, a) > 0.2]
    idx = {c: k for k, c in enumerate(cand)}; M = np.zeros((len(cand),) * 2)
    for k, (i, a) in enumerate(cand): M[k, k] = unary(i, a)
    for (i, j), ea in A_edges.items():
        for (a, b), eb in B_edges.items():
            for (p, q) in (((i, a), (j, b)), ((i, b), (j, a))):
                if p in idx and q in idx: M[idx[p], idx[q]] = M[idx[q], idx[p]] = pair(ea, eb)
    x = np.ones(len(cand)) / np.sqrt(len(cand))
    for _ in range(iters): x = M @ x; x /= np.linalg.norm(x) + 1e-12
    S = np.zeros((len(A_nodes), len(B_nodes)))
    for (i, a), v in zip(cand, x): S[i, a] = v
    r, c = linear_sum_assignment(-S); return [(i, a) for i, a in zip(r, c) if S[i, a] > 0.05]
```
Pairwise affinity example: `exp(-(θ_A-θ_B)^2/(2·3°^2)) * exp(-((L_A-L_B)/max(L_B,10))^2/0.1)`. CAD-pruned search: give D2 only CAD axis families (in the matched frame), set D3 h from the CAD minimum flange area, and use the CAD bend count as a CP-SAT cardinality constraint in the existing A0 solver.

---

### D6. Robust crease scoring + line fitting as a recall-booster proposal channel. Rank 6

**Method.** Compute a per-point crease likelihood with noise-robust integral estimators:
- **multi-scale covariance**: surface variation λ0/(λ0+λ1+λ2) at radii 1.5t … 6t;
- **Difference of Normals (DoN)** across two radii;
- or **VCM** (Voronoi covariance measure, provably robust to noise; CGAL has it, but use a numpy kNN approximation here).

Threshold, then cluster crease points with DBSCAN (`sklearn`), and fit 3D lines with RANSAC per cluster, direction-constrained to the D2 axis families when available. Each line is a *proposal*, not an output. It must pass D4 (two flanges on both sides with dihedral > 8°) to be emitted. This attacks MISSED small flanges and short bends that plane segmentation never isolates.

**Key references**
- Mérigot, Ovsjanikov, Guibas, "Voronoi-based curvature and feature estimation from point clouds", TVCG 2011. https://www.lix.polytechnique.fr/~maks/papers/voronoi.pdf. CGAL `compute_vcm`: https://doc.cgal.org/latest/Point_set_processing_3/
- Weber, Hahmann, Hagen, "Sharp feature detection in point clouds", SMI 2010 (Gauss-map clustering). https://doi.org/10.1109/SMI.2010.32
- Bazazian, Casas, Ruiz-Hidalgo, "Fast and robust edge extraction in unorganized point clouds", DICTA 2015. https://doi.org/10.1109/DICTA.2015.7371262
- Ioannou et al., "Difference of Normals as a multi-scale operator in unorganized point clouds", 3DIMPVT 2012. https://arxiv.org/abs/1209.1759
- Hackel, Wegner, Schindler, "Contour detection in unstructured 3D point clouds", CVPR 2016. https://doi.org/10.1109/CVPR.2016.178
- Edge/corner detection for robotic welding (sheet-like parts). https://arxiv.org/abs/1809.10468

**Why it fits.** Bends are the only *smooth-but-high-curvature* creases on the part. Cut edges are boundaries, not creases, so they can be told apart with `compute_boundary_points` and the two-skin test. Multi-scale means a bend with radius r shows up at radius ≈ r..3r.

**Failure modes**
- Holes, studs and weld beads are strong curvature responses. Reject them because they are not straight (line RANSAC inlier ratio) and lack flange pairs (D4).
- Cut edges and the thickness band of the sheet look like 90° creases. Reject points within t of a boundary, and edges whose two "flanges" are a skin and its cut band (band width ≈ t).
- Noise at 0.5 mm sheet. Use the larger radii.

**CPU:** good. `estimate_covariances` is in Open3D; kNN eigen decomposition is vectorised numpy, about 1–2 s per 200k points per scale.

**Licence:** own code. CGAL is GPL/commercial, so do not use it.

**Sketch**
```python
def crease_score(pcd, radii=(2.0, 4.0, 8.0)):
    P = np.asarray(pcd.points); tree = cKDTree(P); out = []
    for r in radii:
        nb = tree.query_ball_point(P, r, return_sorted=False)
        sv = np.zeros(len(P))
        for i, ix in enumerate(nb):                      # vectorise by chunking in practice
            if len(ix) < 8: continue
            X = P[ix] - P[ix].mean(0); w = np.linalg.eigvalsh(X.T @ X)
            sv[i] = w[0] / w.sum()
        out.append(sv)
    return np.max(out, 0)                                # 0 = planar, 1/3 = isotropic
# then: m = score > tau; labels = DBSCAN(eps=2*spacing, min_samples=8).fit_predict(P[m]);
# per cluster: 3D line RANSAC (sample 2 pts, direction optionally snapped to axis family), keep inlier ratio > 0.6, length > 15 mm
```

---

### D7. Quick win: Open3D `detect_planar_patches` (robust-statistics split-grow-merge) as the proposal generator. Rank 7

**Method.** Open3D ≥ 0.17 ships `PointCloud.detect_planar_patches`, a deterministic robust-statistics planar-patch detector (Araújo & Oliveira). It is O(n log n), has auto-adapted parameters, and handles small planes. Use its oriented boxes as plane proposals for D3, and as a drop-in A/B against the sequential RANSAC in `bend_detector._segment_planes`.

**Key references**
- Araújo & Oliveira, "A robust statistics approach for plane detection in unorganized point clouds", Pattern Recognition 2020. https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/SM/index.html
- Open3D API (`detect_planar_patches`, checked present in the installed 0.19). https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html

**Why it fits.** It is deterministic, so it does not need the perturbation-stability machinery, and it is already installed. Run it on the D1 mid-surface so the skins do not produce twin patches.

**Failure modes**
- It returns boxes, not labels. Assign points to patches by box containment + plane distance.
- Bend strips are split into slivers that must be discarded (aspect ratio along the axis, width ≈ πr·φ/180).
- Default thresholds (60°/75°) are loose for 0.5 mm sheet. Sweep `normal_variance_threshold_deg` 20–45 and `coplanarity_deg` 60–80.

**CPU:** fast, C++ under 1 s.

**Licence:** MIT.

**Sketch**
```python
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
boxes = pcd.detect_planar_patches(normal_variance_threshold_deg=35, coplanarity_deg=75,
                                  outlier_ratio=0.75, min_plane_edge_length=8.0, min_num_points=40,
                                  search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
planes = []
for b in boxes:
    n = b.R[:, 2]; c = n @ b.center
    idx = b.get_point_indices_within_bounding_box(pcd.points)
    planes.append((n, c, np.asarray(idx)))
```

---

### D8. Self-supervised-by-simulation light classifier (synthetic press-brake parts + scanner model). Rank 8

**Method**
1. Procedurally generate press-brake parts with a sheet-metal generator: random t in 0.5–6 mm, r, 1–20 bends, joggles, hems, holes, studs, partial bend states. Each part is a 2D outline extruded and folded with an exact bend-zone geometry.
2. Simulate the ZEISS-style scan using `scanner_model_card.json` (already in `docs/production/runlog/`): one-sided visibility via raycasting from N viewpoints, noise, dropouts and density.
3. Train a **small CPU model** with sklearn GradientBoosting/HistGradientBoosting:
   - features: multi-scale eigen features, DoN, D1 thickness-pair flags, D2 slice-vertex flags;
   - labels: flange / bend-strip / cut-edge / hole-rim / other.
   Or train an MLP and export to ONNX for onnxruntime (training needs torch elsewhere, so sklearn first).
4. The output reweights D3 data costs and D6 crease proposals.

**Key references**
- BenDFM synthetic sheet-metal bending CAD dataset/taxonomy (2026). https://arxiv.org/abs/2603.13102
- Hackel et al. 2016 (hand-crafted multi-scale features + RF for contours). https://doi.org/10.1109/CVPR.2016.178
- Weinmann et al., "Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers", ISPRS J. 2015. https://doi.org/10.1016/j.isprsjprs.2015.01.016

**Why it fits.** This is the only learned option with a clean licence and no domain gap from CAD datasets (ABC), because it is trained on our part family and scanner model. It is "zero-shot" with respect to real scans.

**Failure modes**
- Sim-to-real gap: reflections, multipath on shiny steel, scanner-specific edge artefacts. Validate on the gold set only; do not tune on HOLDOUT.
- The generator effort is about 1–2 days.

**CPU:** training minutes; inference about 1 s.

**Licence:** own (sklearn BSD).

**Sketch:** `features = np.c_[sv_r1, sv_r2, sv_r3, linearity, planarity, don, two_sided, dist_to_boundary/t, slice_vertex_flag]`; `HistGradientBoostingClassifier(max_iter=300).fit(F_syn, y_syn)`.

---

### D9. Learned primitive / edge networks (ParSeNet, HPNet, SPFN, SED-Net, Point2CAD, PIE-NET, EC-Net, DEF, NerVE, ComplexGen) and point foundation models. Rank 9: research only, not for production

**Methods and licences** (checked with the GitHub API, 2026-10-25)

| Model | Reference | Code licence |
|---|---|---|
| SPFN | Li et al., CVPR 2019, https://arxiv.org/abs/1812.07144 | MIT, https://github.com/lingxiaoli94/SPFN |
| ParSeNet | Sharma et al., ECCV 2020, https://arxiv.org/abs/2003.12181 | no licence file, https://github.com/Hippogriff/parsenet-codebase |
| HPNet | Yan et al., ICCV 2021, https://arxiv.org/abs/2105.10620 | no licence file, https://github.com/SimingYan/HPNet |
| SED-Net | Li et al., CVPR 2023 | no licence file, https://github.com/yuanqili78/SED-Net |
| Point2CAD | Liu et al., CVPR 2024, https://arxiv.org/abs/2312.04962 | README says CC-BY-NC 4.0 (repo metadata says Apache-2.0; README governs), https://github.com/prs-eth/point2cad |
| PIE-NET | Wang et al., NeurIPS 2020, https://arxiv.org/abs/2007.04883 | Apache-2.0, https://github.com/wangxiaogang866/PIE-NET |
| EC-Net | Yu et al., ECCV 2018, https://arxiv.org/abs/1807.06010 | no licence file |
| DEF | Matveev et al., SIGGRAPH 2022, https://arxiv.org/abs/2011.15081 | no licence file, https://github.com/artonson/def |
| NerVE | Zhu et al., CVPR 2023, https://arxiv.org/abs/2303.16465 | not checked |
| ComplexGen | Guo et al., SIGGRAPH 2022, https://arxiv.org/abs/2205.14573 | MIT, https://github.com/guohaoxiang/ComplexGen |

Foundation models:
- Sonata / PTv3 (CVPR 2025): code Apache-2.0, **weights CC-BY-NC 4.0**. https://github.com/facebookresearch/sonata
- PartField (NVIDIA): NVIDIA licence, non-commercial. https://github.com/nv-tlabs/PartField
- Pointcept framework: MIT.

**Why it is ranked last**
1. **Licence.** Almost all of these are trained on ABC (Onshape-derived, research terms). Pretrained weights are either unlicensed, NC, or ABC-derived. I found **no pretrained primitive/edge weights clearly licensed for commercial use**. The only commercially clean code (SPFN MIT, PIE-NET Apache, ComplexGen MIT) would need retraining on our own synthetic data (see D8).
2. **Domain gap.** They are trained on clean, normalised, about 10k-point CAD samples. They are not trained on two-skin thin sheet with 0.5 mm features at 0.3 mm noise. Published follow-ups report HPNet/SED-Net generalise poorly to densified or real clouds.
3. **Runtime.** torch is not installed. Running them would need ONNX export done elsewhere, and their custom ops (kNN, mean-shift, differentiable fitting) do not export cleanly.
4. **Thin sheet.** At normalised scale a 1 mm sheet is below their resolution, so both skins collapse into one noisy surface.

Use them only as offline comparators if a GPU box is available.

**Point2CAD's idea worth stealing:** segment, then fit each cluster, then recover edges by *intersecting neighbouring fitted surfaces*. That is what D1/D4 do with sheet-metal-specific surfaces.

**CPU:** poor to moderate. **Licence:** see above, mostly not commercial.

---

### D10 (cross-cutting). Global consistency / developability verification to prune false positives

**Method.** After proposals from D2/D3/D6, solve a selection problem. The existing A0 CP-SAT already does this and should get these new terms:
- every emitted bend must separate two distinct flanges with strip evidence (D4);
- no two emitted lines within d = t/sin(θ/2) + 1 mm with the same axis and angle (the two-skin duplicate rule from D1);
- the flange graph must be a tree, or nearly a tree for closed boxes, and **unfoldable**: unfold the flanges about the bend lines using t and a K-factor of about 0.33–0.45, and check that the flat pattern has no self-overlap and every flange is connected;
- the bend count is bounded by the CAD prior when it is available (D5).

**Refs**
- Unfolding/flat-pattern: OCCT sheet unfolding docs. https://occt3d.com/components/sheet-metal-operations/
- Unfolding of non-developable sheet-metal surfaces. https://www.researchgate.net/publication/271975459

**CPU / licence:** trivial; own code + ortools.

This targets the 47 SPURIOUS FPs and the duplicate-skin FPs.

---

## Suggested execution order and measurement

1. **D1 thickness pre-pass alone.** Measure how many current OFFSET FNs/FPs flip, using the existing scorer `research/broad_baseline/score.py` on the gold set. Cheap, and high information value.
2. **D2 slicing lane** as a new scan-only lane, unioned with A0 through D4 gating. Expect the largest recall gain (MISSED + SPAN).
3. **D7 → D3** to replace the sequential RANSAC proposals.
4. **D5** for the CAD-prior lane, reusing `cad_bend_extractor.py` and the CP-SAT env.
5. **D6 and D10** for the last recall and precision points. Then D8 if the sim-to-real budget exists.

For each step, ablate on the dev cohorts only, keep the matcher fixed (5°/4 mm/0.6), and report the FN/FP cause histograms as in BROAD_BASELINE_SUMMARY.

## Sources consulted (web)
- https://eureka.patsnap.com/patent-CN106295058A
- https://projekter.aau.dk/computer-vision-based-bend-quality-inspection-for-sheet-metal-parts-06fddb1d.html
- https://link.springer.com/chapter/10.1007/978-3-031-84744-8_19
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12349652/
- https://arxiv.org/abs/2603.13102
- https://www.nature.com/articles/s41598-024-61443-2
- https://arxiv.org/abs/2312.04962 and https://github.com/prs-eth/point2cad
- https://arxiv.org/abs/1906.02290 and https://github.com/danini/progressive-x
- https://github.com/danini/graph-cut-ransac
- https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/SM/index.html
- https://www.lix.polytechnique.fr/~maks/papers/voronoi.pdf
- https://arxiv.org/abs/2405.19339
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7014519/
- https://github.com/facebookresearch/sonata
- https://github.com/nv-tlabs/PartField
- https://github.com/lingxiaoli94/SPFN, https://github.com/wangxiaogang866/PIE-NET, https://github.com/guohaoxiang/ComplexGen
- https://github.com/Hippogriff/parsenet-codebase, https://github.com/SimingYan/HPNet, https://github.com/artonson/def, https://github.com/yuanqili78/SED-Net
