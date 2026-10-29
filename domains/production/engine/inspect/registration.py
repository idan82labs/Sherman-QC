"""Stage-independent datum registration (E3 step 2; plan §3.2 and §10 amendment 9).

The datum panel never moves in any stage model, so it is the one part of the
scan whose pose does not depend on how far the part has been bent. The
registration uses **only** the datum's plane, its **free (cut) edges** and its
**holes**. A boundary of the datum that is a bend line is never used: when that
bend is not yet formed, the scan shows a flat continuation there.

1. **Planes** (:func:`extract_planes`). Planar voxel points (PCA curvature)
   are binned by normal direction; the densest direction's offset histogram
   gives a plane; its inliers are split into connected segments with the voxel
   neighbour graph. Repeated until no dense direction remains.
2. **Datum candidates**: the largest segments (area >= a fraction of the
   datum skin area). A parallel segment one thickness away is the other skin,
   which fixes the material side.
3. **In-plane search** (:func:`_coarse_search`). For each candidate and each
   of the two ways the plane can map onto the datum (face up or face down,
   which mirror the outline as seen in the plane), the scan segment is
   rasterised and correlated (FFT) with a datum *kernel*: +1 inside the datum
   outline (holes excluded), -w in *forbidden* zones (the interior of every
   hole and a band outside every free edge), 0 beyond bend edges (don't care).
   Angles are swept every few degrees; the best peaks are refined on a finer
   raster.
4. **Fine fit** (:func:`refine_datum`): the datum skin plane(s) are fitted
   robustly (height and tilt, from local skin tiles), then the in-plane pose is
   solved by weighted least squares on per-bin free-edge observations
   (point-to-edge-line) and per angular-bin hole boundaries (radial), iterated.
   Only round holes are binned radially; a slot, a rectangular window or any
   other cut-out contributes its straight boundary runs as cut edges
   (:meth:`DatumModel.hole_segments`), where the hole is open at every compiled
   stage (a tab lying flat in its window at S_0 blocks that part). A hole is
   "filled" (WRONG_PART evidence) by its inner disc when round, else when the
   scan covers its open core about as densely as the datum around it.
   A round hole's fill is judged only where its disc is clear of compiled stage
   material (a stepped or countersunk hole's smaller bore lies inside the loop
   E1 reports on skin 0).
   A bin whose **cut wall** is seen (points between the skins) is observed at
   the wall; a bare skin edge sits at a range-image depth discontinuity, where
   the scanner erodes it inward, so it gets a low weight and its possible
   erosion (measured on this scan: walled minus unwalled offsets) is carried as
   a worst-case pose bias (:meth:`InPlaneFit.position_bias_mm`) into the
   bend-line position U. The information matrix of the constraints says
   whether the in-plane pose is determined; if not, ``in_plane_degenerate``
   (only bend-line position and the heatmap become untrusted; angles and
   flange lengths do not depend on the pose). The undetermined directions are
   kept (``InPlaneFit.undetermined``): a bend-line position measured across
   them (along the datum normal, or along the determined axis) stays trusted
   (:meth:`InPlaneFit.position_determined`). After the articulated fit, the
   undetermined directions are fixed, when they can be, by the **cut ends of
   fitted flanges** that no bend moves along them (:func:`child_edge_rows`,
   :func:`resolve_free_directions`): a flange reached from the datum only
   through bends parallel to a direction ``a`` of the datum plane keeps, formed
   or flat, its blank's cut edges perpendicular to ``a`` exactly where the blank
   put them along ``a``. At least two consistent such ends are required, and
   the directions the datum itself determines are never moved.
5. **Sheet thickness** (:func:`tile_thickness`): where a 12 mm tile of the
   registered datum shows both skins, their separation is the real sheet's
   thickness. The second level is searched within +-25 % of the compiled ``t``,
   so a sheet off its CAD thickness by a stock tolerance is still measured; the
   median over the tiles (``InPlaneFit.thickness_mm``) replaces the compiled
   ``t`` wherever a single visible skin is moved to the other one.

Every hypothesis keeps its score, so the caller can keep the top-K poses,
including the symmetric and mirrored ones (computed on the mirrored scan), and
report the margin between them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from domains.production.engine.inspect.cloud import VoxelCloud, components_of
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import (
    apply,
    orthonormal_frame,
    rigid,
    robust_plane,
    rotation_between,
    skin_peaks,
    unit,
)
from domains.production.engine.inspect.part import PartView

__all__ = [
    "PlaneSegment",
    "DatumModel",
    "PoseHypothesis",
    "InPlaneFit",
    "extract_planes",
    "coarse_hypotheses",
    "datum_hole_targets",
    "scan_holes",
    "hole_pair_hypotheses",
    "refine_datum",
    "fine_search",
    "tile_thickness",
    "ChildEdgeRows",
    "child_edge_rows",
    "resolve_free_directions",
    "MIRROR",
]

#: Reflection used to build improper (mirrored-part) hypotheses on the mirrored scan.
MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])
_REF_DIR = np.array([0.37, 0.53, 0.76]) / np.linalg.norm([0.37, 0.53, 0.76])


@dataclass
class PlaneSegment:
    n: np.ndarray
    d: float
    indices: np.ndarray  # voxel indices
    area: float
    centroid: np.ndarray
    rms: float


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def mean_normal(normals: np.ndarray, ref: np.ndarray, trim_deg: float = 6.0) -> np.ndarray:
    """Trimmed mean of unoriented unit normals, oriented along ``ref``."""
    n = np.asarray(normals, dtype=np.float64)
    n = n * np.sign(n @ ref)[:, None]
    m = unit(n.sum(axis=0))
    for _ in range(2):
        keep = (n @ m) > math.cos(math.radians(trim_deg))
        if keep.sum() < 3:
            break
        m = unit(n[keep].sum(axis=0))
    return m


def _footprint_area(pts: np.ndarray, n: np.ndarray, pixel: float) -> float:
    e1, e2 = orthonormal_frame(n)
    uv = np.stack([pts @ e1, pts @ e2], axis=1)
    q = np.floor(uv / pixel).astype(np.int64)
    key = q[:, 0] * 1_000_003 + q[:, 1]
    return float(len(np.unique(key))) * pixel * pixel


def extract_planes(vox: VoxelCloud, cfg: InspectConfig, min_area: float, thickness: float) -> list[PlaneSegment]:
    """Greedy normal-histogram plane extraction split into connected segments."""
    assert vox.normals is not None and vox.spread is not None and vox.graph is not None
    pts, nrm = vox.points, vox.normals.copy()
    slab = bool(vox.slab)
    planar = vox.spread < cfg.planar_spread(thickness, slab)
    offset_tol = cfg.plane_offset_tol_mm + (0.5 * thickness if slab else 0.0)
    flip = (nrm @ _REF_DIR) < 0
    nrm[flip] *= -1.0
    bin_rad = math.radians(cfg.plane_normal_bin_deg)
    grid = _fibonacci_sphere(max(200, int(4 * math.pi / (bin_rad * bin_rad))))
    _, bins = cKDTree(grid).query(nrm, k=1, workers=1)
    remaining = planar.copy()
    cos_tol = math.cos(math.radians(cfg.plane_normal_tol_deg))
    min_pts = max(30, int(min_area / (vox.voxel * vox.voxel) * 0.3))
    out: list[PlaneSegment] = []
    for _ in range(cfg.max_planes * 2):
        if len(out) >= cfg.max_planes:
            break
        counts = np.bincount(bins[remaining], minlength=len(grid))
        best = int(np.argmax(counts))
        if counts[best] < min_pts:
            break
        nd = grid[best]
        for _ in range(3):
            sel = remaining & (np.abs(nrm @ nd) > cos_tol)
            if not sel.any():
                break
            m = nrm[sel] * np.sign(nrm[sel] @ nd)[:, None]
            nd = unit(m.sum(axis=0))
        sel = remaining & (np.abs(nrm @ nd) > cos_tol)
        if sel.sum() < min_pts:
            remaining[bins == best] = False
            continue
        dvals = pts[sel] @ nd
        hb = 0.25
        lo = float(dvals.min())
        hist = np.bincount(np.floor((dvals - lo) / hb).astype(np.int64))
        w = max(1, int(round(offset_tol / hb)))
        smooth = np.convolve(hist, np.ones(2 * w + 1), mode="same")
        peak = int(np.argmax(smooth))
        d0 = lo + (peak + 0.5) * hb
        inl = np.nonzero(sel)[0][np.abs(dvals - d0) < offset_tol]
        if len(inl) < 3:
            remaining[sel] = False
            continue
        plane = robust_plane(pts[inl], iterations=3)
        dist = np.abs(plane.distance(pts))
        mask = remaining & (dist < offset_tol) & (np.abs(nrm @ plane.n) > cos_tol)
        if mask.sum() < 3:
            remaining[sel] = False
            continue
        comp_idx = np.nonzero(mask)[0]
        sub = vox.graph[comp_idx][:, comp_idx]
        from scipy.sparse.csgraph import connected_components

        _, labels = connected_components(sub, directed=False)
        remaining[mask] = False
        for lab in np.unique(labels):
            idx = comp_idx[labels == lab]
            if len(idx) < min_pts:
                continue
            area = _footprint_area(pts[idx], plane.n, vox.voxel)
            if area < min_area:
                continue
            p = robust_plane(pts[idx], iterations=2)
            if float(p.n @ plane.n) < 0:
                p = p.flipped()
            # the normal from the local (voxel) normals: a least-squares plane through a thin sheet
            # whose two skins are seen in different areas tilts by about t / distance between them
            n_loc = mean_normal(vox.normals[idx], p.n)
            d_loc = float(np.median(pts[idx] @ n_loc))
            out.append(PlaneSegment(n_loc, d_loc, idx, area, pts[idx].mean(axis=0), p.rms))
    out.sort(key=lambda s: -s.area)
    return out


# --------------------------------------------------------------------------- datum model (2D)
@dataclass
class DatumModel:
    """The datum panel in its own 2D frame (origin ``p0``, axes ``f1``, ``f2``, normal ``n0``)."""

    p0: np.ndarray
    n0: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    t: float
    polygon: np.ndarray  # (K, 2) outer boundary, counter-clockwise
    free: np.ndarray  # (K,) bool: edge i -> i+1 is a free (cut) edge
    holes: list[np.ndarray]
    area: float
    bbox: tuple[float, float, float, float]
    #: Per hole: 2D points of compiled stage material (S_0 and S_N, inside the datum slab) that lie
    #: in the hole polygon. A tab cut out of a window lies flat in it at S_0 and leaves its bend zone
    #: there at S_N: the hole is not open in every stage, so that area neither forbids scan
    #: material nor anchors the in-plane pose (stage-independent registration, amendment 9).
    hole_blocked: list = field(default_factory=list)

    def round_hole(self, k: int) -> bool:
        """A circular hole (radial bins about its centre); any other shape is used through its straight runs."""
        return _round_hole(self.holes[k])

    def blocked_points(self, k: int) -> np.ndarray:
        blocked = getattr(self, "hole_blocked", None) or []
        return blocked[k] if k < len(blocked) else np.zeros((0, 2))

    def open_cells(self, k: int, cells: np.ndarray, clearance: float = 1.5) -> np.ndarray:
        """Mask of ``cells`` (2D) farther than ``clearance`` from stage material inside hole ``k``."""
        blocked = self.blocked_points(k)
        if len(blocked) == 0 or len(cells) == 0:
            return np.ones(len(cells), dtype=bool)
        d, _ = cKDTree(blocked).query(cells, k=1, distance_upper_bound=clearance, workers=1)
        return ~np.isfinite(d)

    def hole_segments(self, min_length: float, reach: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """``(a, b, normal into the hole, band limit)`` of the straight, open runs of every non-round hole.

        A slot, a rectangular window or a cut-out has no centre to bin radially;
        its straight boundary runs are cut edges like the outline's free edges
        (the material lies behind them, the normal points into the hole). A run
        is used where the hole in front of it is open in every compiled stage:
        stage material (a tab flat in its window at S_0) closer than ``reach``
        plus a margin ends the usable part of the run, and the band that may
        hold the edge's own material stops short of the hole's opposite side and
        of any stage material (the returned limit, mm). Runs shorter than
        ``min_length`` are dropped.
        """
        key = (round(float(min_length), 6), round(float(reach), 6))
        cache = self.__dict__.setdefault("_hole_segs", {})
        if key in cache:
            return cache[key]
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for k, hp in enumerate(self.holes):
            if len(hp) < 3 or self.round_hole(k):
                continue
            ccw = _signed_area(hp) > 0
            blocked = self.blocked_points(k)
            for a, b in _straight_runs(hp):
                d = b - a
                length = float(np.linalg.norm(d))
                if length < min_length:
                    continue
                e = d / length
                n_in = np.array([-e[1], e[0]]) if ccw else np.array([e[1], -e[0]])
                clear = _clearance(hp, a, b, n_in)
                lim_c = 0.45 * clear
                if lim_c < 1.0:
                    continue  # a narrow slot: its opposite side is inside any useful band
                ss = np.arange(0.0, length + 1e-9, 1.0)
                lim = np.full(len(ss), lim_c)
                if len(blocked):
                    rel = blocked - a
                    al, of = rel @ e, rel @ n_in
                    near = (of > -0.5) & (of < reach + 2.0) & (al > -1.5) & (al < length + 1.5)
                    if near.any():
                        order = np.argsort(al[near])
                        alv = al[near][order]
                        ofv = np.maximum(of[near][order], 0.0) - 0.6  # the band stops 0.6 mm short of it
                        lo_i = np.searchsorted(alv, ss - 1.25, side="left")
                        hi_i = np.searchsorted(alv, ss + 1.25, side="right")
                        for si in range(len(ss)):
                            if hi_i[si] > lo_i[si]:
                                lim[si] = min(lim[si], float(ofv[lo_i[si]:hi_i[si]].min()))
                ok = lim >= 1.0
                i = 0
                while i < len(ss):
                    if not ok[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < len(ss) and ok[j + 1]:
                        j += 1
                    s0, s1 = float(ss[i]), float(ss[j])
                    if s1 - s0 >= min_length:
                        out.append((a + e * s0, a + e * s1, n_in, float(lim[i:j + 1].min())))
                    i = j + 1
        cache[key] = out
        return out

    def hole_fill_cells(self, k: int, cell: float, inset: float, ring: float) -> tuple[np.ndarray, np.ndarray]:
        """(open core cells, surrounding material ring cells) of non-round hole ``k`` (cached 2D cell centres).

        Core: inside the hole, at least ``inset`` from its boundary and clear of
        stage material. Ring: datum material within ``inset .. inset + ring`` of
        the hole boundary (it shows whether this part of the datum was seen).
        """
        key = (k, round(float(cell), 6), round(float(inset), 6), round(float(ring), 6))
        cache = self.__dict__.setdefault("_hole_fill", {})
        if key in cache:
            return cache[key]
        hp = self.holes[k]
        closed = np.vstack([hp, hp[:1]])
        lo, hi = hp.min(axis=0) - (inset + ring + cell), hp.max(axis=0) + (inset + ring + cell)
        gx = np.arange(lo[0], hi[0], cell) + 0.5 * cell
        gy = np.arange(lo[1], hi[1], cell) + 0.5 * cell
        xx, yy = np.meshgrid(gx, gy)
        cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
        inside = MplPath(hp).contains_points(cells)
        dist = _distance_to_polyline(cells, closed)
        core = cells[inside & (dist >= inset)]
        core = core[self.open_cells(k, core)]
        cand = cells[~inside & (dist >= inset) & (dist <= inset + ring)]
        mat, _ = self.material_lookup(cand)
        cache[key] = (core, cand[mat])
        return cache[key]

    @classmethod
    def from_part(cls, part: PartView) -> "DatumModel":
        panel = part.panels[part.datum].model
        n0 = np.asarray(panel.skin0_plane.normal, dtype=np.float64)
        p0 = np.asarray(panel.skin0_plane.point_mm, dtype=np.float64)
        f1, f2 = orthonormal_frame(n0)
        outer = np.asarray(panel.outer_boundary_mm, dtype=np.float64)
        poly = np.stack([(outer - p0) @ f1, (outer - p0) @ f2], axis=1)
        free_segments = []
        for fe in panel.free_edges:
            if not fe.registration_usable:
                continue  # E1 says this boundary must not drive the stage-independent registration
            line = np.asarray(fe.polyline_mm, dtype=np.float64)
            l2 = np.stack([(line - p0) @ f1, (line - p0) @ f2], axis=1)
            free_segments.extend(zip(l2[:-1], l2[1:]))
        k = len(poly)
        free = np.zeros(k, dtype=bool)
        for i in range(k):
            a, b = poly[i], poly[(i + 1) % k]
            if np.linalg.norm(b - a) < 1e-9:
                continue
            mid = 0.5 * (a + b)
            for s0, s1 in free_segments:
                if _point_segment_distance(mid, s0, s1) < 1e-3 and _point_segment_distance(a, s0, s1) < 1e-3 \
                        and _point_segment_distance(b, s0, s1) < 1e-3:
                    free[i] = True
                    break
        area = _signed_area(poly)
        if area < 0:  # make counter-clockwise; edge i (i -> i+1) becomes edge k-2-i of the reversed list
            poly = poly[::-1].copy()
            free = np.roll(free[::-1], -1)
            area = -area
        holes = []
        for h in panel.holes_mm:
            hp = np.asarray(h, dtype=np.float64)
            holes.append(np.stack([(hp - p0) @ f1, (hp - p0) @ f2], axis=1))
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        return cls(p0, n0, f1, f2, part.t, poly, free, holes, area, (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                   hole_blocked=_stage_material_in_holes(part, p0, n0, f1, f2, holes))

    def material_lookup(self, uv: np.ndarray, pixel: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """(inside the outline and outside every hole, distance to that boundary in mm) per 2D point."""
        cache = self.__dict__.setdefault("_material", {})
        if pixel not in cache:
            from scipy.ndimage import distance_transform_edt

            x0, y0, x1, y1 = self.bbox
            pad = 2.0
            ox, oy = x0 - pad, y0 - pad
            w = int(math.ceil((x1 - x0 + 2 * pad) / pixel)) + 1
            hgt = int(math.ceil((y1 - y0 + 2 * pad) / pixel)) + 1
            gx = ox + np.arange(w) * pixel
            gy = oy + np.arange(hgt) * pixel
            xx, yy = np.meshgrid(gx, gy)
            cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
            mask = MplPath(self.polygon).contains_points(cells)
            for hp in self.holes:
                mask &= ~MplPath(hp).contains_points(cells)
            mask = mask.reshape(hgt, w)
            depth = distance_transform_edt(mask) * pixel
            cache[pixel] = (mask, depth, np.array([ox, oy]))
        mask, depth, origin = cache[pixel]
        q = np.floor((uv - origin) / pixel + 0.5).astype(np.int64)
        hh, ww = mask.shape
        ok = (q[:, 0] >= 0) & (q[:, 0] < ww) & (q[:, 1] >= 0) & (q[:, 1] < hh)
        inside = np.zeros(len(uv), dtype=bool)
        dist = np.zeros(len(uv))
        inside[ok] = mask[q[ok, 1], q[ok, 0]]
        dist[ok] = depth[q[ok, 1], q[ok, 0]]
        return inside, dist

    def boundary_tree(self, which: str = "all") -> cKDTree:
        """KD-tree of boundary samples every 0.2 mm: ``all`` (outline + holes) or ``free`` (free edges + holes)."""
        cache = self.__dict__.setdefault("_btrees", {})
        if which not in cache:
            segs = []
            if which == "all":
                closed = np.vstack([self.polygon, self.polygon[:1]])
                segs.extend(zip(closed[:-1], closed[1:]))
            else:
                segs.extend((a, b) for a, b, _n in self.free_segments())
            for hp in self.holes:
                closed = np.vstack([hp, hp[:1]])
                segs.extend(zip(closed[:-1], closed[1:]))
            samples = []
            for a, b in segs:
                n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.2)) + 1)
                samples.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
            pts = np.vstack(samples) if samples else np.zeros((1, 2)) + 1e9
            cache[which] = cKDTree(pts)
        return cache[which]

    def kernel_cached(self, pixel: float, band: float, weight: float, hole_weight: float = 0.0,
                      clearance: float = 1.5, clearance_window: float = 1.5):
        key = (round(pixel, 6), round(band, 6), round(weight, 6), round(hole_weight, 6), round(clearance, 6),
               round(clearance_window, 6))
        cache = self.__dict__.setdefault("_kernels", {})
        if key not in cache:
            cache[key] = self.kernel(pixel, band, weight, 4.0, hole_weight=hole_weight, clearance=clearance,
                                     clearance_window=clearance_window)
        return cache[key]

    def rotational_symmetries(self, pixel: float = 0.5, overlap_min: float = 0.98) -> tuple[list[float], np.ndarray]:
        """``(angles, centre)``: the in-plane rotations (90, 180, 270 deg about the material's area centroid) that map
        the datum's material (outline minus holes) onto itself, with at least ``overlap_min`` of the rotated material
        landing on material. Such a datum cannot tell those images of a pose apart by its own outline."""
        cache = self.__dict__.setdefault("_rotsym", {})
        key = (round(float(pixel), 6), round(float(overlap_min), 6))
        if key not in cache:
            x0, y0, x1, y1 = self.bbox
            gx = np.arange(x0, x1 + pixel, pixel)
            gy = np.arange(y0, y1 + pixel, pixel)
            xx, yy = np.meshgrid(gx, gy)
            cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
            ins, _ = self.material_lookup(cells, pixel)
            mat = cells[ins]
            angles: list[float] = []
            centre = mat.mean(axis=0) if len(mat) else np.zeros(2)
            if len(mat):
                for phi in (0.5 * math.pi, math.pi, 1.5 * math.pi):
                    rot = (mat - centre) @ _rot2(phi).T + centre
                    on, _ = self.material_lookup(rot, pixel)
                    if float(on.mean()) >= overlap_min:
                        angles.append(phi)
            cache[key] = (angles, centre)
        return cache[key]

    def symmetric_images(self, T: np.ndarray) -> list[np.ndarray]:
        """The poses ``S @ T`` for every in-plane rotation ``S`` of :meth:`rotational_symmetries` (CAD frame)."""
        sym, centre = self.rotational_symmetries()
        out = []
        F = np.stack([self.f1, self.f2, self.n0], axis=1)
        c3 = self.p0 + self.f1 * centre[0] + self.f2 * centre[1]
        for phi in sym:
            r3 = np.eye(3)
            r3[:2, :2] = _rot2(phi)
            R = F @ r3 @ F.T
            S = np.eye(4)
            S[:3, :3] = R
            S[:3, 3] = c3 - R @ c3
            out.append(S @ T)
        return out

    def to2d(self, pts: np.ndarray) -> np.ndarray:
        q = np.asarray(pts, dtype=np.float64) - self.p0
        return np.stack([q @ self.f1, q @ self.f2], axis=1)

    def free_segments(self, min_length: float = 0.0) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(a, b, outward normal) of every free polygon edge (counter-clockwise polygon)."""
        out = []
        k = len(self.polygon)
        for i in range(k):
            if not self.free[i]:
                continue
            a, b = self.polygon[i], self.polygon[(i + 1) % k]
            d = b - a
            length = float(np.linalg.norm(d))
            if length < max(min_length, 1e-9):
                continue
            out.append((a, b, np.array([d[1], -d[0]]) / length))
        return out

    def kernel(self, pixel: float, band: float, weight: float, margin: float, hole_weight: float = 0.0,
               clearance: float = 1.5, clearance_window: float = 1.5):
        """Rasterised score kernel and its grid origin (see module docstring).

        With ``hole_weight`` > 0 the hole cores get their own normalised penalty
        (small holes otherwise vanish against the free-edge bands). A hole's core
        counts where it lies farther than ``clearance`` from compiled stage
        material inside the hole (:meth:`open_cells`): a stepped or countersunk
        hole's bore is open in every stage. A non-round hole (a window, a slot)
        keeps at least ``clearance_window``: the material in it is a tab cut on
        its own contour, whose kerf, edge flare and kink put scan points a
        millimetre or more beside its compiled footprint.
        """
        x0, y0, x1, y1 = self.bbox
        pad = band + margin
        ox, oy = x0 - pad, y0 - pad
        w = int(math.ceil((x1 - x0 + 2 * pad) / pixel)) + 1
        h = int(math.ceil((y1 - y0 + 2 * pad) / pixel)) + 1
        gx = ox + (np.arange(w) + 0.0) * pixel
        gy = oy + (np.arange(h) + 0.0) * pixel
        xx, yy = np.meshgrid(gx, gy)
        cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
        inside = MplPath(self.polygon).contains_points(cells)
        hole_any = np.zeros(len(cells), dtype=bool)
        forb = np.zeros(len(cells), dtype=bool)
        for k, hp in enumerate(self.holes):
            hin = MplPath(hp).contains_points(cells)
            hole_any |= hin
            if hin.any():
                dist = _distance_to_polyline(cells[hin], np.vstack([hp, hp[:1]]))
                core = np.zeros(len(cells), dtype=bool)
                idx = np.nonzero(hin)[0][dist > max(0.75, pixel)]
                c_k = clearance if self.round_hole(k) else max(clearance, clearance_window)
                core[idx[self.open_cells(k, cells[idx], c_k)]] = True  # only where open at every stage
                forb |= core
        for a, b, nout in self.free_segments():
            d = b - a
            length = float(np.linalg.norm(d))
            e = d / length
            rel = cells - a
            along = rel @ e
            off = rel @ nout
            band_mask = (along >= 0) & (along <= length) & (off > max(0.75, pixel)) & (off < band)
            forb |= band_mask & ~inside
        inside &= ~hole_any
        kin = inside.reshape(h, w).astype(np.float64)
        kf = forb.reshape(h, w).astype(np.float64)
        n_in, n_f = max(kin.sum(), 1.0), max(kf.sum(), 1.0)
        kernel = kin / n_in - weight * kf / n_f
        if hole_weight > 0 and self.holes:
            core = np.zeros(len(cells), dtype=bool)
            for k, hp in enumerate(self.holes):
                hin = MplPath(hp).contains_points(cells)
                if hin.any():
                    dist = _distance_to_polyline(cells[hin], np.vstack([hp, hp[:1]]))
                    idx = np.nonzero(hin)[0][dist > max(0.5, pixel)]
                    c_k = clearance if self.round_hole(k) else max(clearance, clearance_window)
                    core[idx[self.open_cells(k, cells[idx], c_k)]] = True
            kh = core.reshape(h, w).astype(np.float64)
            if kh.sum() > 0:
                kernel = kernel - hole_weight * kh / kh.sum()
        return kernel, np.array([ox, oy]), kin, kf


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_segment_distance(p, a, b) -> float:
    d = b - a
    ll = float(d @ d)
    if ll < 1e-18:
        return float(np.linalg.norm(p - a))
    s = min(1.0, max(0.0, float((p - a) @ d) / ll))
    return float(np.linalg.norm(p - (a + s * d)))


def _distance_to_polyline(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    best = np.full(len(pts), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        d = b - a
        ll = float(d @ d)
        if ll < 1e-18:
            continue
        s = np.clip((pts - a) @ d / ll, 0.0, 1.0)
        q = a + s[:, None] * d
        best = np.minimum(best, np.linalg.norm(pts - q, axis=1))
    return best


def _round_hole(hp: np.ndarray) -> bool:
    """Circular: binned radially about its centre.

    At least 8 vertices whose radii scatter by less than 8 % of their mean
    radius ``r``, enclosing an area within 12 % of ``pi r^2``. A rectangle's
    corners are equidistant from its centre, and a rounded window's arcs crowd
    its vertex list, but neither encloses the area of a circle.
    """
    if len(hp) < 8:
        return False
    c = hp.mean(axis=0)
    rads = np.linalg.norm(hp - c, axis=1)
    r = max(float(np.mean(rads)), 1e-6)
    if float(np.std(rads)) >= 0.08 * r:
        return False
    return bool(abs(abs(_signed_area(hp)) / (math.pi * r * r) - 1.0) < 0.12)


def _straight_runs(poly: np.ndarray, ang_tol_deg: float = 2.0, dev_tol: float = 0.05) -> list[tuple[np.ndarray, np.ndarray]]:
    """Maximal straight runs ``(a, b)`` of a closed polygon (consecutive edges along one chord).

    An edge joins the current run while its direction stays within
    ``ang_tol_deg`` of the run's chord and every vertex of the run stays within
    ``dev_tol`` of it, so arcs (fillets, slot ends) break into short pieces.
    The walk starts at a corner, so no straight edge is split at the polygon's
    first vertex; a finely tessellated curve gives only short pieces, and a
    polygon without any corner none at all.
    """
    pts = np.asarray(poly, dtype=np.float64)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
            keep.append(i)
    pts = pts[keep]
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    k = len(pts)
    if k < 3:
        return []
    d = np.roll(pts, -1, axis=0) - pts
    d = d / np.linalg.norm(d, axis=1)[:, None]
    cos_tol = math.cos(math.radians(ang_tol_deg))
    turn = np.einsum("ij,ij->i", np.roll(d, 1, axis=0), d)  # turn at vertex i (edge i-1 -> edge i)
    corners = np.nonzero(turn < cos_tol)[0]
    if len(corners) == 0:
        return []
    start = int(corners[0])
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    i = 0
    while i < k:
        i0 = (start + i) % k
        j = i + 1  # the run covers vertices start+i .. start+j
        while j < k:
            a, b = pts[i0], pts[(start + j + 1) % k]
            chord = b - a
            cl = float(np.linalg.norm(chord))
            if cl < 1e-9:
                break
            ce = chord / cl
            if float(d[(start + j) % k] @ ce) < cos_tol:
                break
            inner = pts[[(start + m) % k for m in range(i, j + 2)]] - a
            dev = np.abs(inner @ np.array([-ce[1], ce[0]]))
            if float(dev.max()) > dev_tol:
                break
            j += 1
        runs.append((pts[i0].copy(), pts[(start + j) % k].copy()))
        i = j
    return runs


def _clearance(hp: np.ndarray, a: np.ndarray, b: np.ndarray, n_in: np.ndarray) -> float:
    """Distance from the run's midpoint along ``n_in`` across the hole to its boundary (the opposite side)."""
    o = 0.5 * (a + b) + n_in * 1e-3
    closed = np.vstack([hp, hp[:1]])
    best = np.inf
    for p, q in zip(closed[:-1], closed[1:]):
        e = q - p
        m = np.array([[n_in[0], -e[0]], [n_in[1], -e[1]]])
        det = float(np.linalg.det(m))
        if abs(det) < 1e-12:
            continue
        s, u = np.linalg.solve(m, p - o)
        if s > 1e-3 and -1e-9 <= u <= 1.0 + 1e-9:
            best = min(best, float(s))
    return float(best) if math.isfinite(best) else 0.0


def _stage_material_in_holes(part: PartView, p0: np.ndarray, n0: np.ndarray, f1: np.ndarray, f2: np.ndarray,
                             holes: list[np.ndarray]) -> list[np.ndarray]:
    """Per datum hole: 2D compiled stage material (S_0, S_N; datum slab) inside it, 0.3 mm clear of its boundary."""
    if not holes:
        return []
    chunks = []
    for sid in ("S0", f"S{part.n}"):
        if sid not in part.stages:
            continue
        pts = part.cloud(sid).points
        h = (pts - p0) @ n0
        q = pts[(h > -part.t - 0.5) & (h < 0.5)] - p0
        chunks.append(np.stack([q @ f1, q @ f2], axis=1))
    uv = np.vstack(chunks) if chunks else np.zeros((0, 2))
    out = []
    for hp in holes:
        lo, hi = hp.min(axis=0), hp.max(axis=0)
        cand = uv[(uv[:, 0] > lo[0]) & (uv[:, 0] < hi[0]) & (uv[:, 1] > lo[1]) & (uv[:, 1] < hi[1])]
        if len(cand):
            cand = cand[MplPath(hp).contains_points(cand)]
        if len(cand):
            cand = cand[_distance_to_polyline(cand, np.vstack([hp, hp[:1]])) > 0.3]
        out.append(np.asarray(cand, dtype=np.float64).reshape(-1, 2))
    return out


# --------------------------------------------------------------------------- hypotheses
@dataclass
class PoseHypothesis:
    #: raw (unit-scaled) scan -> cad_model_mm; det -1 for mirrored hypotheses
    T: np.ndarray
    score: float
    segment: int
    family: int
    height: float
    mirrored: bool
    theta_deg: float
    coverage: float = 0.0
    violation: float = 0.0
    partner: bool = False
    info: dict = field(default_factory=dict)


def _scan_frame(seg: PlaneSegment) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e1, e2 = orthonormal_frame(seg.n)
    return seg.centroid, e1, e2, seg.n


def _rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _build_transform(model: DatumModel, seg: PlaneSegment, family: int, theta: float, tau: np.ndarray,
                     height: float) -> np.ndarray:
    q, e1, e2, m = _scan_frame(seg)
    e = np.stack([e1, e2, m], axis=1)
    f = np.stack([model.f1, model.f2, model.n0], axis=1)
    r2 = _rot2(theta)
    local = np.eye(3)
    local[:2, :2] = r2
    mf = np.diag([1.0, float(family), float(family)])
    r = f @ local @ mf @ e.T
    origin_model = model.p0 + model.f1 * tau[0] + model.f2 * tau[1] + model.n0 * height
    t = origin_model - r @ q
    return rigid(r, t)


def _raster(points2d: np.ndarray, pixel: float) -> tuple[np.ndarray, np.ndarray]:
    lo = points2d.min(axis=0) - pixel
    q = np.floor((points2d - lo) / pixel).astype(np.int64)
    h, w = int(q[:, 1].max()) + 2, int(q[:, 0].max()) + 2
    img = np.zeros((h, w), dtype=np.float64)
    img[q[:, 1], q[:, 0]] = 1.0
    return img, lo


def _correlate(img: np.ndarray, kernel_fft: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    a = np.fft.rfft2(img, s=shape)
    return np.fft.irfft2(np.conj(a) * kernel_fft, s=shape)


def _lag_to_tau(i: int, j: int, shape, pixel: float, origin_k: np.ndarray, origin_s: np.ndarray) -> np.ndarray:
    hh, ww = shape
    li = i if i < hh // 2 else i - hh
    lj = j if j < ww // 2 else j - ww
    return np.array([lj * pixel, li * pixel]) + origin_k - origin_s


def _score_direct(model_kin: np.ndarray, model_kf: np.ndarray, origin_k: np.ndarray, pixel: float,
                  pts2d: np.ndarray, weight: float) -> tuple[float, float, float]:
    """(score, coverage, violation) of scan points already mapped into the model 2D frame."""
    q = np.floor((pts2d - origin_k) / pixel + 0.5).astype(np.int64)
    h, w = model_kin.shape
    ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    q = q[ok]
    if len(q) == 0:
        return -1.0, 0.0, 0.0
    key = np.unique(q[:, 1] * w + q[:, 0])
    rows, cols = np.divmod(key, w)
    cov = float(model_kin[rows, cols].sum()) / max(float(model_kin.sum()), 1.0)
    viol = float(model_kf[rows, cols].sum()) / max(float(model_kf.sum()), 1.0)
    return cov - weight * viol, cov, viol


def coarse_hypotheses(vox: VoxelCloud, part: PartView, model: DatumModel, cfg: InspectConfig, *,
                      mirrored: bool, budget_left, segments: list[PlaneSegment] | None = None,
                      plane_seeds: bool = False) -> tuple[list[PoseHypothesis], list[PlaneSegment]]:
    """Coarse datum poses for the scan (or the mirrored scan): see the module docstring.

    Peaks of every (candidate plane, family, angle) correlation are pooled and
    the global best ones (non-maximum suppressed) are angle-polished and turned
    into 3D transforms (both skin heights unless the other skin was seen).

    ``plane_seeds`` (a datum that fixes none of its in-plane directions, ``engine._register``): the
    best peak of every (candidate plane, family) that no global peak represents is returned instead
    of the global peaks. Such a datum's outline correlates with the large planes of the formed part
    as well as with its own plane, so the global peaks can all land on those planes.
    """
    datum_area = float(part.panels[part.datum].area)
    if segments is None:
        segments = extract_planes(vox, cfg, min_area=max(30.0, cfg.datum_min_area_fraction * datum_area),
                                  thickness=part.t)
    cands = segments[: cfg.datum_candidates]
    if not cands:
        return [], segments
    pix = cfg.coarse_pixel_mm
    kc, ok_c, _, _ = model.kernel(pix, cfg.free_edge_band_mm, cfg.violation_weight, 4.0,
                                  clearance=cfg.hole_core_clearance_mm,
                                  clearance_window=cfg.window_core_clearance_mm)
    _, ok_f, kin_f, kf_f = model.kernel_cached(cfg.fine_pixel_mm, cfg.free_edge_band_mm, cfg.violation_weight,
                                               clearance=cfg.hole_core_clearance_mm,
                                               clearance_window=cfg.window_core_clearance_mm)
    pooled: list[tuple[float, int, int, float, np.ndarray]] = []
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]] = {}
    for si, seg in enumerate(cands):
        if budget_left() <= 0:
            break
        q, e1, e2, _m = _scan_frame(seg)
        p = vox.points[seg.indices] - q
        base2d = np.stack([p @ e1, p @ e2], axis=1)
        # one point per half-pixel cell keeps the raster exact and the rotation cheap
        cell = np.floor(base2d / (0.5 * pix)).astype(np.int64)
        _, first = np.unique(cell[:, 0] * 2_000_003 + cell[:, 1], return_index=True)
        reduced = base2d[np.sort(first)]
        for family in (1, -1):
            pts = reduced.copy()
            pts[:, 1] *= family
            full = base2d.copy()
            full[:, 1] *= family
            diam = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) + 4 * pix
            ns = int(math.ceil(diam / pix)) + 3
            shape = (_fft_size(kc.shape[0] + ns), _fft_size(kc.shape[1] + ns))
            kfft = np.fft.rfft2(kc, s=shape)
            prepared[(si, family)] = (pts, full, shape, kfft)
            for th in np.arange(0.0, 2 * math.pi, math.radians(cfg.coarse_angle_step_deg)):
                val, tau = _corr_peak(pts, float(th), kfft, shape, pix, ok_c)
                pooled.append((val, si, family, float(th), tau))
    pooled.sort(key=lambda r: -r[0])
    chosen: list[tuple[float, int, int, float, np.ndarray]] = []
    for row in pooled:
        if len(chosen) >= cfg.coarse_candidates_fine:
            break
        val, si, family, th, tau = row
        if any(c[1] == si and c[2] == family and _angle_diff(th, c[3]) < math.radians(10)
               and np.linalg.norm(tau - c[4]) < 10.0 for c in chosen):
            continue
        chosen.append(row)
    if plane_seeds:
        seen = {(c[1], c[2]) for c in chosen}
        seeds = []
        for row in pooled:  # sorted by value: the first row of a (plane, family) is its best peak
            if (row[1], row[2]) not in seen:
                seen.add((row[1], row[2]))
                seeds.append(row)
        chosen = seeds
    hyps: list[PoseHypothesis] = []
    for val, si, family, th, tau in chosen:
        seg = cands[si]
        pts, full, shape, kfft = prepared[(si, family)]
        # angle polish on the coarse raster
        best = (val, th, tau)
        for dth in (-1.5, -0.75, 0.75, 1.5):
            v2, t2 = _corr_peak(pts, th + math.radians(dth), kfft, shape, pix, ok_c)
            if v2 > best[0]:
                best = (v2, th + math.radians(dth), t2)
        _, th_b, tau_b = best
        rp = full @ _rot2(th_b).T + tau_b
        sc, cov, viol = _score_direct(kin_f, kf_f, ok_f, cfg.fine_pixel_mm, rp, cfg.violation_weight)
        partner = _partner_side(seg, segments, part.t)
        heights = [0.0, -model.t]
        if partner is not None:
            face_is_skin0 = (partner < 0) == (family > 0)
            heights = [0.0] if face_is_skin0 else [-model.t]
        for h in heights:
            t_raw = _build_transform(model, seg, family, th_b, tau_b, h)
            T = t_raw @ MIRROR if mirrored else t_raw
            hyps.append(PoseHypothesis(T=T, score=sc, segment=si, family=family, height=h, mirrored=mirrored,
                                       theta_deg=math.degrees(th_b), coverage=cov, violation=viol,
                                       partner=partner is not None))
    hyps.sort(key=lambda hp: -hp.score)
    return _dedupe(hyps), segments


def datum_hole_targets(model: DatumModel) -> list[tuple[np.ndarray, float]]:
    """``(centre, open radius)`` of every round datum hole: the radius of the disc open in every stage.

    A through-hole is open to its loop; a stepped or countersunk hole only to its
    bore, where the compiled stage material inside the loop (:meth:`DatumModel.blocked_points`)
    begins. That open disc is what a scan of the datum plane shows empty.
    """
    out = []
    for k, hp in enumerate(model.holes):
        if len(hp) < 3 or not _round_hole(hp):
            continue
        c = hp.mean(axis=0)
        r = _hole_radius(hp)
        blocked = model.blocked_points(k)
        if len(blocked):
            r = min(r, float(np.min(np.linalg.norm(blocked - c, axis=1))))
        out.append((c, float(r)))
    return out


def scan_holes(uv: np.ndarray, spacing: float, r_lo: float, r_hi: float, pixel: float = 0.4
               ) -> list[tuple[np.ndarray, float]]:
    """Round empty discs enclosed by 2D scan points: ``(centre, radius)`` with ``r_lo <= radius <= r_hi``.

    A cell is empty when no point lies within ``max(0.8, 2 spacing)`` of it.
    Every empty region that does not reach the raster's border is a candidate;
    its radius is the largest distance from a cell to the nearest point (the
    inscribed disc), its centre the centroid of the cells within 60 % of that.
    A slot or a gap between panels holds far more cells than a disc of that
    radius and is dropped.
    """
    from scipy import ndimage

    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) < 50 or r_hi <= 0:
        return []
    gap = max(0.8, 2.0 * float(spacing))
    lo = uv.min(axis=0) - 2.0 * pixel
    hi = uv.max(axis=0) + 2.0 * pixel
    nx = int(math.ceil((hi[0] - lo[0]) / pixel)) + 1
    ny = int(math.ceil((hi[1] - lo[1]) / pixel)) + 1
    if nx * ny > 6_000_000:
        pixel *= math.sqrt(nx * ny / 6_000_000)
        nx = int(math.ceil((hi[0] - lo[0]) / pixel)) + 1
        ny = int(math.ceil((hi[1] - lo[1]) / pixel)) + 1
    gx = lo[0] + (np.arange(nx) + 0.5) * pixel
    gy = lo[1] + (np.arange(ny) + 0.5) * pixel
    xx, yy = np.meshgrid(gx, gy)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    cap = r_hi + 1.0
    d, _ = cKDTree(uv).query(cells, k=1, distance_upper_bound=cap, workers=1)
    d = np.where(np.isfinite(d), d, cap).reshape(ny, nx)
    lab, n = ndimage.label(d > gap)
    if n == 0:
        return []
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or i in border:
            continue
        comp = lab[sl] == i
        dv = d[sl]
        r = float(dv[comp].max())
        if r < r_lo or r > r_hi:
            continue
        if comp.sum() * pixel * pixel > 2.5 * math.pi * max(r - gap + pixel, pixel) ** 2:
            continue  # elongated: a slot or a gap, not a hole
        core = comp & (dv >= 0.6 * r)
        jy, jx = np.nonzero(core)
        centre = np.array([gx[sl[1]][jx].mean(), gy[sl[0]][jy].mean()])
        out.append((centre, r))
    return out


def hole_pair_hypotheses(points: np.ndarray, segments: list[PlaneSegment], part: PartView, model: DatumModel,
                         cfg: InspectConfig, spacing: float, *, mirrored: bool) -> list[PoseHypothesis]:
    """Datum poses from pairs of the datum's round holes found in the scan (stage independent: plane + holes).

    The coarse raster cannot see holes of a few millimetres; on a datum whose
    outline barely fixes the pose (a strip bounded by bends, scanned while its
    neighbours still lie flat beside it) its peaks can sit degrees and
    millimetres off. For each candidate plane and each way it can map onto the
    datum (``family``), the scan's round holes (:func:`scan_holes`, slab points
    of the plane) are matched to the datum's (:func:`datum_hole_targets`, open
    radius within ``max(0.6, 25 %)``): a pair of datum holes and a pair of scan
    holes the same distance apart (within 0.8 mm) fix the in-plane pose. Every
    pose is counted by how many datum holes land on a scan hole (1 mm); among
    the best-supported distinct poses (at least two holes) the
    ``hole_pair_hypotheses`` best by the datum kernel's score (as the coarse
    peaks are scored) are returned. Used for datums with 2 to
    ``hole_pair_max_holes`` round holes.
    """
    targets = datum_hole_targets(model)
    if len(targets) < 2 or len(targets) > cfg.hole_pair_max_holes or cfg.hole_pair_hypotheses <= 0:
        return []
    radii = np.array([r for _, r in targets])
    tol_r = np.maximum(0.6, 0.25 * radii)
    r_lo = float(max(0.5, np.min(radii - tol_r)))
    r_hi = float(np.max(radii + tol_r))
    cad = np.array([c for c, _ in targets])
    _, ok_f, kin_f, kf_f = model.kernel_cached(cfg.fine_pixel_mm, cfg.free_edge_band_mm, cfg.violation_weight,
                                               clearance=cfg.hole_core_clearance_mm,
                                               clearance_window=cfg.window_core_clearance_mm)
    found: list[tuple[int, float, int, int, float, np.ndarray]] = []
    for si, seg in enumerate(segments[: cfg.datum_candidates]):
        q, e1, e2, n = _scan_frame(seg)
        dist = points @ n - float(seg.d)
        slab = np.abs(dist) < model.t + 0.8
        if slab.sum() < 50:
            continue
        p = points[slab] - q
        uv = np.stack([p @ e1, p @ e2], axis=1)
        holes = scan_holes(uv, spacing, r_lo, r_hi)
        if len(holes) < 2 or len(holes) > 4 * cfg.hole_pair_max_holes:
            continue
        hc = np.array([c for c, _ in holes])
        hr = np.array([r for _, r in holes])
        compat = np.abs(hr[:, None] - radii[None, :]) <= tol_r[None, :]  # scan hole x datum hole
        for family in (1, -1):
            sc2 = hc.copy()
            sc2[:, 1] *= family
            for i in range(len(cad)):
                for j in range(i + 1, len(cad)):
                    dij = cad[j] - cad[i]
                    lij = float(np.linalg.norm(dij))
                    if lij < 3.0:
                        continue
                    for a in np.nonzero(compat[:, i])[0]:
                        for b in np.nonzero(compat[:, j])[0]:
                            if a == b:
                                continue
                            dab = sc2[b] - sc2[a]
                            if abs(float(np.linalg.norm(dab)) - lij) > 0.8:
                                continue
                            th = math.atan2(dij[1], dij[0]) - math.atan2(dab[1], dab[0])
                            rot = _rot2(th)
                            tau = 0.5 * (cad[i] + cad[j]) - rot @ (0.5 * (sc2[a] + sc2[b]))
                            mapped = sc2 @ rot.T + tau
                            dd = np.linalg.norm(mapped[:, None, :] - cad[None, :, :], axis=2)
                            hits = int(np.sum(np.any((dd <= 1.0) & compat, axis=0)))
                            if hits >= 2:
                                found.append((hits, th, si, family, 0.0, tau))
    if not found:
        return []
    found.sort(key=lambda r: -r[0])
    best_hits = found[0][0]
    # the best-supported distinct poses, ranked by the datum kernel's score (a spurious pair of empty discs
    # the right distance apart has the same two hits as the true pair)
    kept: list[tuple[float, float, float, float, int, int, float, np.ndarray]] = []
    slabs: dict[tuple[int, int], np.ndarray] = {}
    for hits, th, si, family, _, tau in found:
        if hits < best_hits or len(kept) >= 64:
            break
        if any(k[4] == si and k[5] == family and _angle_diff(th, k[6]) < math.radians(2.0)
               and float(np.linalg.norm(tau - k[7])) < 2.0 for k in kept):
            continue
        if (si, family) not in slabs:
            seg = segments[si]
            q, e1, e2, _n = _scan_frame(seg)
            pv = np.asarray(points) - q
            near = np.abs(np.asarray(points) @ seg.n - float(seg.d)) < model.t + 0.8
            slabs[(si, family)] = np.stack([pv[near] @ e1, pv[near] @ e2 * family], axis=1)
        sc, cov, viol = _score_direct(kin_f, kf_f, ok_f, cfg.fine_pixel_mm, slabs[(si, family)] @ _rot2(th).T + tau,
                                      cfg.violation_weight)
        kept.append((sc, cov, viol, float(hits), si, family, th, tau))
    kept.sort(key=lambda k: -k[0])
    hyps: list[PoseHypothesis] = []
    for sc, cov, viol, hits, si, family, th, tau in kept[: cfg.hole_pair_hypotheses]:
        seg = segments[si]
        partner = _partner_side(seg, segments, part.t)
        heights = [0.0, -model.t]
        if partner is not None:
            face_is_skin0 = (partner < 0) == (family > 0)
            heights = [0.0] if face_is_skin0 else [-model.t]
        for h in heights:
            t_raw = _build_transform(model, seg, family, th, tau, h)
            T = t_raw @ MIRROR if mirrored else t_raw
            hyps.append(PoseHypothesis(T=T, score=sc, segment=si, family=family, height=h, mirrored=mirrored,
                                       theta_deg=math.degrees(th), coverage=cov, violation=viol,
                                       partner=partner is not None, info={"hole_pair_hits": int(hits)}))
    hyps.sort(key=lambda hp: -hp.score)
    return _dedupe(hyps)


def fine_search(points: np.ndarray, T: np.ndarray, model: DatumModel, cfg: InspectConfig,
                with_score: bool = False, walks: int | None = None):
    """Exhaustive in-plane search around ``T`` on a fine raster (holes weighted).

    The coarse raster cannot see holes of a few millimetres, and when the
    datum's outline leaves a direction free (a datum whose cut edges run one
    way only, scanned mid-sequence inside a larger flat region) the coarse peak
    can sit anywhere along it. This search correlates the datum-plane scan
    points with a 0.5 mm kernel that penalises occupied hole cores, over
    +-``fine_search_mm`` and +-``fine_search_deg``. A best angle on the edge of
    that window is followed: the window is centred on it again, up to
    ``fine_search_walks`` times while the score improves (a coarse peak several
    degrees off, on a datum whose outline barely fixes the rotation, would
    otherwise leave the search, and the pose, on its window's edge). ``walks``
    overrides that count (0: the window around ``T`` only).
    """
    pc = apply(T, points)
    h = (pc - model.p0) @ model.n0
    sel = (h < 0.6) & (h > -model.t - 0.6)
    if sel.sum() < 30:
        return (T, -np.inf) if with_score else T
    uv = model.to2d(pc[sel])
    x0, y0, x1, y1 = model.bbox
    m = cfg.fine_search_mm + cfg.free_edge_band_mm + 2.0
    keep = (uv[:, 0] > x0 - m) & (uv[:, 0] < x1 + m) & (uv[:, 1] > y0 - m) & (uv[:, 1] < y1 + m)
    uv = uv[keep]
    if len(uv) < 30:
        return (T, -np.inf) if with_score else T
    pix = cfg.fine_search_pixel_mm
    kern, ok, _, _ = model.kernel_cached(pix, cfg.free_edge_band_mm, cfg.violation_weight, hole_weight=1.0,
                                         clearance=cfg.hole_core_clearance_mm,
                                         clearance_window=cfg.window_core_clearance_mm)
    cen2 = model.polygon.mean(axis=0)
    rel = uv - cen2
    win = int(math.ceil(cfg.fine_search_mm / pix))
    ns = int(math.ceil((rel.max(axis=0) - rel.min(axis=0)).max() / pix)) + 8 + 2 * win
    shape = (_fft_size(kern.shape[0] + ns), _fft_size(kern.shape[1] + ns))
    kfft = np.fft.rfft2(kern, s=shape)

    def window(centre: float) -> tuple[float, float, np.ndarray]:
        out = (-np.inf, centre, np.zeros(2))
        for dth in centre + np.radians(np.arange(-cfg.fine_search_deg, cfg.fine_search_deg + 1e-9, 0.5)):
            rp = rel @ _rot2(float(dth)).T + cen2
            img, lo = _raster(rp, pix)
            corr = _correlate(img, kfft, shape)
            # lag that leaves the points where they are (tau = 0)
            base = (lo + 0.5 * pix - ok) / pix
            bi, bj = int(round(base[1])), int(round(base[0]))
            ii = (np.arange(bi - win, bi + win + 1)) % shape[0]
            jj = (np.arange(bj - win, bj + win + 1)) % shape[1]
            sub = corr[np.ix_(ii, jj)]
            k = int(np.argmax(sub))
            a, b = divmod(k, sub.shape[1])
            val = float(sub[a, b])
            if val > out[0]:
                tau = np.array([(b - win) * pix, (a - win) * pix]) - (base - np.array([bj, bi])) * pix
                out = (val, float(dth), tau)
        return out

    centre = 0.0
    best = window(centre)
    edge = math.radians(max(0.0, cfg.fine_search_deg - 0.25))
    for _ in range(int(cfg.fine_search_walks if walks is None else walks)):
        if abs(best[1] - centre) < edge or not math.isfinite(best[0]):
            break
        centre = best[1]
        nxt = window(centre)
        if nxt[0] <= best[0]:
            break
        best = nxt
    val, dth, tau = best
    r3 = np.eye(3)
    r3[:2, :2] = _rot2(dth)
    F = np.stack([model.f1, model.f2, model.n0], axis=1)
    R = F @ r3 @ F.T
    c3 = model.p0 + model.f1 * cen2[0] + model.f2 * cen2[1]
    out = rigid(R, c3 - R @ c3 + model.f1 * tau[0] + model.f2 * tau[1]) @ T
    return (out, float(val)) if with_score else out


def _corr_peak(pts: np.ndarray, theta: float, kfft: np.ndarray, shape, pix: float, origin_k: np.ndarray
               ) -> tuple[float, np.ndarray]:
    rp = pts @ _rot2(theta).T
    img, lo = _raster(rp, pix)
    corr = _correlate(img, kfft, shape)
    idx = int(np.argmax(corr))
    i, j = divmod(idx, corr.shape[1])
    return float(corr.ravel()[idx]), _lag_to_tau(i, j, corr.shape, pix, origin_k, lo + 0.5 * pix)


def _fft_size(n: int) -> int:
    size = 1
    while size < n:
        size *= 2
    # allow 3 * 2^k sizes for less padding
    if size // 4 * 3 >= n:
        return size // 4 * 3
    return size


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def _partner_side(seg: PlaneSegment, segments: list[PlaneSegment], t: float) -> float | None:
    """Sign (along ``seg.n``) of a parallel segment one thickness away (the other skin), else None."""
    best = None
    for other in segments:
        if other is seg:
            continue
        if abs(float(other.n @ seg.n)) < math.cos(math.radians(3.0)):
            continue
        nn = other.n if float(other.n @ seg.n) > 0 else -other.n
        dd = other.d if float(other.n @ seg.n) > 0 else -other.d
        gap = dd - seg.d
        if abs(abs(gap) - t) > max(0.3, 0.25 * t):
            continue
        # overlapping footprints
        if np.linalg.norm((other.centroid - seg.centroid) - ((other.centroid - seg.centroid) @ seg.n) * seg.n) > \
                0.5 * math.sqrt(max(seg.area, other.area)):
            continue
        if best is None or other.area > best[1]:
            best = (math.copysign(1.0, gap), other.area)
        _ = nn
    return None if best is None else best[0]


def _dedupe(hyps: list[PoseHypothesis]) -> list[PoseHypothesis]:
    out: list[PoseHypothesis] = []
    for h in hyps:
        # the two skin heights of one in-plane pose differ by a thickness: both are kept
        if any(_same_pose(h.T, o.T) and abs(h.height - o.height) < 1e-9 for o in out):
            continue
        out.append(h)
    return out


def _same_pose(a: np.ndarray, b: np.ndarray, ang_deg: float = 2.0, dist: float = 1.5) -> bool:
    r = a[:3, :3] @ b[:3, :3].T
    if np.linalg.det(r) < 0:
        return False
    c = (float(np.trace(r)) - 1.0) / 2.0
    ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    if ang > ang_deg:
        return False
    probe = np.array([0.0, 0.0, 0.0])
    pa = a[:3, :3] @ probe + a[:3, 3]
    pb = b[:3, :3] @ probe + b[:3, 3]
    return float(np.linalg.norm(pa - pb)) < dist + 0.0


# --------------------------------------------------------------------------- fine datum fit
@dataclass
class InPlaneFit:
    T: np.ndarray
    #: 3x3 information matrix of (rotation [rad * L], tx, ty) in effective observations
    information: np.ndarray
    length_scale: float
    edge_bins: int
    edge_bins_expected: int
    holes_seen: int
    holes_total: int
    edge_rms: float
    hole_rms: float
    degenerate: bool
    #: 2x2 standard covariance (mm^2) of the in-plane translation in the datum 2D frame
    translation_cov: np.ndarray
    rotation_sigma_rad: float
    skin_heights: dict[int, float]
    datum_points: int
    datum_coverage: float
    hole_mismatch: int
    #: Stage-independent datum score after the fine fit (outline coverage, free-edge and
    #: hole agreement, minus forbidden-zone violation); higher is better.
    score: float = 0.0
    two_skins: bool = False
    #: 3 x m pose shift (rotation * L, tx, ty; mm) when each of the m unwalled rows (skin
    #: edges and hole bins seen without their cut wall) is eroded by its erosion scale.
    bias_gain: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    #: Erosion scale of unwalled skin edges in this scan (mm), see ``InspectConfig.erosion_*``.
    erosion_mm: float = 0.0
    #: The sheet thickness measured on the datum (two skins seen in at least
    #: ``thickness_min_tiles`` tiles), its standard uncertainty and the tile count; None when
    #: the datum shows one skin only.
    thickness_mm: float | None = None
    thickness_sigma_mm: float | None = None
    thickness_tiles: int = 0
    detail: dict = field(default_factory=dict)
    #: 3x3 sandwich covariance (mm^2) of (rotation * L, tx, ty). Blind along undetermined directions
    #: (no row constrains them, so the sandwich has nothing there): see ``undetermined``.
    pose_cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    #: Unit (rotation * L, tx, ty) directions with fewer than ``degenerate_min_information``
    #: effective observations (rows): the solve never moves the pose along them.
    undetermined: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def _sensitivity(self, model: "DatumModel", point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Shift (mm) at ``point`` along ``direction`` per unit (rotation * L, tx, ty) pose change."""
        x = model.to2d(np.asarray(point, dtype=np.float64)[None, :])[0]
        m = np.array([float(direction @ model.f1), float(direction @ model.f2)])
        xr = x - model.polygon.mean(axis=0)
        return np.array([float(m @ np.array([-xr[1], xr[0]])) / self.length_scale, m[0], m[1]])

    def position_determined(self, model: "DatumModel", point: np.ndarray, direction: np.ndarray,
                            tol: float = 0.05) -> bool:
        """True when a position at ``point`` measured along ``direction`` does not depend on an undetermined
        pose direction (at most ``tol`` mm per mm of pose error along each of them).

        A datum whose cut edges and holes leave one in-plane direction free (a
        panel bounded by bends on two opposite sides) still fixes every position
        measured across it: along the datum normal, or along the determined axis.
        """
        if self.undetermined.size == 0:
            return True
        q = self._sensitivity(model, point, direction)
        return bool(np.all(np.abs(self.undetermined @ q) <= tol))

    def position_sigma_mm(self, model: "DatumModel", point: np.ndarray, direction: np.ndarray) -> float:
        """Standard uncertainty (mm) of the datum frame at ``point`` along ``direction``, within the determined
        pose directions (``pose_cov`` projected; the undetermined directions are left to
        :meth:`position_determined`)."""
        q = self._sensitivity(model, point, direction)
        for v in self.undetermined:
            q = q - float(q @ v) * v
        cov = np.asarray(self.pose_cov, dtype=np.float64)
        if not np.all(np.isfinite(cov)):
            return 1.0
        return float(math.sqrt(max(0.0, float(q @ cov @ q))))

    def position_bias_mm(self, model: "DatumModel", point: np.ndarray, direction: np.ndarray) -> float:
        """Worst-case in-plane registration bias (mm) at ``point`` along ``direction`` (CAD frame).

        Every unwalled row may be eroded by anything from 0 to its erosion scale; the
        bound is the larger of the positive and the negative sensitivities' sums.
        """
        if self.bias_gain.size == 0:
            return 0.0
        x = model.to2d(np.asarray(point, dtype=np.float64)[None, :])[0]
        m = np.array([float(direction @ model.f1), float(direction @ model.f2)])
        xr = x - model.polygon.mean(axis=0)
        q = np.array([float(m @ np.array([-xr[1], xr[0]])) / self.length_scale, m[0], m[1]])
        s = q @ self.bias_gain
        return float(max(float(s[s > 0].sum()), float(-s[s < 0].sum())))


def _datum_points(pts_cad: np.ndarray, model: DatumModel, slack: float, shrink: float):
    """Indices of points on the datum skins (either skin) inside the (shrunk) outline."""
    h = (pts_cad - model.p0) @ model.n0
    near = (np.abs(h) < slack) | (np.abs(h + model.t) < slack)
    idx = np.nonzero(near)[0]
    if len(idx) == 0:
        return idx, h
    uv = model.to2d(pts_cad[idx])
    # material area (outline minus holes) and the distance to its boundary, on a cached raster:
    # the outline's and the holes' walls span the thickness and stay out of the skin heights
    inside, depth = model.material_lookup(uv)
    if shrink > 0:
        inside &= depth > shrink
    return idx[inside], h


def refine_datum(points: np.ndarray, T0: np.ndarray, model: DatumModel, cfg: InspectConfig, spacing: float,
                 *, fix_height: float | None = None, vox_points: np.ndarray | None = None,
                 vox_normals: np.ndarray | None = None) -> InPlaneFit:
    """Fine datum registration of the full-resolution ``points`` (raw, unit-scaled) from ``T0``.

    Returns the refined raw->CAD transform with the in-plane information.
    """
    T = T0.copy()
    gap = cfg.edge_gap_factor * spacing
    # 1. plane (height + tilt) from the datum skin points. When both skins are seen the
    #    upper one (along n0) is skin 0; with a single skin the hypothesis's height decides.
    heights: dict[int, float] = {}
    two_skins = False
    for it in range(8):
        pc = apply(T, points)
        slack = model.t + 1.5 if it == 0 else 0.6
        idx, h = _datum_points(pc, model, slack, 1.5)
        if it == 0:
            idx = idx[np.abs(h[idx] + 0.5 * model.t) < 0.5 * model.t + 1.5]
        if len(idx) < 50:
            break
        uv = model.to2d(pc[idx])
        skin, certain = _local_skins(h[idx], uv, model.t, cfg)
        two_skins = bool((skin == 0).sum() >= 30 and (skin == 1).sum() >= 30)
        if (skin >= 0).sum() < 30:
            break
        if not two_skins and it == 0:
            # one skin only: the cut walls show on which side of it the material lies
            side = _material_side(pc, h, model, cfg)
            if side is not None:
                visible = 0 if side < 0 else 1  # material below the visible face: it is skin 0
                skin = np.where(skin >= 0, visible, -1)
        # joint plane of both skins: skin-1 points moved up by t along the current normal. Tiles
        # that show both skins are unambiguous; while the plane is still tilted, single-skin
        # tiles far from the centre can be assigned the wrong skin, so they wait.
        q = pc[idx].copy()
        q[skin == 1] += model.n0 * model.t
        use = skin >= 0
        if certain.sum() >= max(200, 0.2 * use.sum()):
            use = use & certain
        pl = robust_plane(q[use], iterations=3)
        nmean = pl.n if float(pl.n @ model.n0) > 0 else -pl.n
        cen = pc[idx][use].mean(axis=0)
        r = rotation_between(nmean, model.n0)
        T = rigid(r, cen - r @ cen) @ T
        pc = apply(T, points)
        hh = (pc[idx] - model.p0) @ model.n0
        heights = {}
        # tiles that show both skins fix the heights; single-skin tiles were only assigned by
        # proximity to the current plane and would keep a wrong height
        ref = (skin >= 0) & certain if certain.sum() >= 200 else skin >= 0
        for sk in (0, 1):
            if ((skin == sk) & ref).sum() >= 30:
                heights[sk] = float(np.median(hh[(skin == sk) & ref]))
        if 0 in heights:
            dz = -heights[0]
        elif 1 in heights:
            dz = -(heights[1] + model.t)
        else:
            dz = 0.0
        if fix_height is not None:
            dz = 0.0
        T = rigid(np.eye(3), model.n0 * dz) @ T
        if abs(dz) < 1e-4 and float(np.linalg.norm(r - np.eye(3))) < 1e-7:
            break
    # 2. in-plane: free-edge extremes + hole boundaries (cut walls included: the slab)
    pc = apply(T, points)
    h = (pc - model.p0) @ model.n0
    cand = np.nonzero((h < 0.5) & (h > -model.t - 0.5))[0]
    L = max(10.0, 0.5 * math.sqrt(model.area))
    cen2 = model.polygon.mean(axis=0)
    info = np.zeros((3, 3))
    min_seg = 2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm
    segs = list(model.free_segments(min_length=min_seg))
    if hasattr(model, "hole_segments"):  # the straight, open runs of slots and windows are cut edges too
        segs += model.hole_segments(min_seg, 2.0 * cfg.edge_band_mm)
    F = np.stack([model.f1, model.f2, model.n0], axis=1)
    c3 = model.p0 + model.f1 * cen2[0] + model.f2 * cen2[1]
    # cut-wall points lie between the skins (heights are invariant under the in-plane updates)
    interior = None
    if model.t >= cfg.wall_min_thickness_mm and len(cand):
        margin = cfg.wall_skin_margin_fraction * model.t
        interior = (h[cand] < -margin) & (h[cand] > -model.t + margin)
    for it in range(cfg.refine_iterations):
        uv = model.to2d(pc[cand]) if len(cand) else np.zeros((0, 2))
        con = _inplane_constraints(uv, model, segs, cfg, gap, spacing, cen2, L, wide=(it == 0), interior=interior)
        if len(con["rows"]) < 3:
            break
        A = np.asarray(con["rows"])
        bvec = np.asarray(con["rhs"])
        scale = max(0.05, 1.4826 * float(np.median(np.abs(bvec))))
        hub = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(bvec), 1e-12)) * np.asarray(con["weights"])
        H = (A * hub[:, None]).T @ A
        g = (A * hub[:, None]).T @ bvec
        info = A.T @ A
        try:
            delta = np.linalg.lstsq(H + 1e-9 * np.eye(3), g, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        # only update constrained directions. A direction the rows determine only weakly (fewer than
        # ``degenerate_min_information`` effective observations: a strip datum whose rotation rests on two
        # holes) is still moved to its least-squares optimum once it has ``refine_min_information``: left
        # frozen, it kept the fine search's raster pose, up to that search's window (2 deg) off. It stays
        # undetermined below (degenerate, position trust), so no position depends on it unchecked.
        evals, evecs = np.linalg.eigh(info)
        keep = evals >= min(cfg.refine_min_information, cfg.degenerate_min_information)
        delta = evecs @ (np.where(keep, evecs.T @ delta, 0.0))
        dth = float(delta[0]) / L
        dtx, dty = float(delta[1]), float(delta[2])
        r3 = np.eye(3)
        r3[:2, :2] = _rot2(dth)
        R = F @ r3 @ F.T
        T = rigid(R, c3 - R @ c3 + model.f1 * dtx + model.f2 * dty) @ T
        pc = apply(T, points)
        if abs(dth) * L < 1e-4 and math.hypot(dtx, dty) < 1e-4:
            break
    uv = model.to2d(pc[cand]) if len(cand) else np.zeros((0, 2))
    # material standing proud of the datum slab (>= _HARDWARE_PROUD_MM beyond either skin): a stud, standoff or
    # clinch nut pressed into a CAD hole shows there, concentric with the hole; a hole with no hole cut never does
    hp_ = (pc - model.p0) @ model.n0
    proud_idx = np.nonzero((hp_ >= _HARDWARE_PROUD_MM) | (hp_ <= -model.t - _HARDWARE_PROUD_MM))[0]
    proud = model.to2d(pc[proud_idx]) if len(proud_idx) else np.zeros((0, 2))
    con = _inplane_constraints(uv, model, segs, cfg, gap, spacing, cen2, L, wide=False, interior=interior,
                               proud=proud)
    wts = np.asarray(con["weights"], dtype=np.float64)
    walled = np.asarray(con["walled"], dtype=bool)
    if len(con["rows"]) >= 1:
        A = np.asarray(con["rows"])
        info = A.T @ A
    else:
        A = np.zeros((0, 3))
        info = np.zeros((3, 3))
    edge_res, hole_res = con["edge_res"], con["hole_res"]
    edge_arr = np.asarray(edge_res, dtype=np.float64)
    edge_wall = np.asarray(con["edge_walled"], dtype=bool)
    hole_arr = np.asarray(hole_res, dtype=np.float64)
    hole_wall = np.asarray([w for g in con["hole_walled"] for w in g], dtype=bool)
    # hole residuals carry each hole's radius bias (the sampling gap inside the hole wall),
    # which does not move its centre: the scatter is taken about each hole's own mean. Rows
    # seen without their cut wall are eroded by a view-dependent amount: a bias (bounded
    # below), not noise, so the scatter comes from the walled rows when there are enough.
    walled_centred = list(edge_arr[edge_wall])
    for g, gw in zip(con["hole_groups"], con["hole_walled"]):
        gv = np.asarray(g)[np.asarray(gw, dtype=bool)]
        if len(gv) >= 3:
            walled_centred += list(gv - float(np.mean(gv)))
    if len(walled_centred) >= 10:
        centred = walled_centred
    else:
        centred = list(edge_arr) + [r - float(np.mean(g)) for g in con["hole_groups"] for r in g]
    evals, evecs = np.linalg.eigh(info)
    degenerate = bool(evals.min() < cfg.degenerate_min_information)
    # the solve never updates these directions (see the loop above): the pose along them is the
    # fine search's, with no measured uncertainty
    undetermined = evecs[:, evals < cfg.degenerate_min_information].T.copy()
    if centred:
        c = np.asarray(centred)
        sigma_obs = max(0.02, 1.4826 * float(np.median(np.abs(c - np.median(c)))))
    else:
        sigma_obs = 0.1
    # weighted least squares: covariance by the sandwich rule, and the gain of every unwalled
    # row on the pose (a row eroded by e moves the pose by gain * e)
    # erosion scales of unwalled rows (edges inward, holes outward), measured on this scan
    erosion = _erosion_scale(edge_arr[edge_wall], edge_arr[~edge_wall], spacing, cfg)
    erosion_hole = _erosion_scale(-hole_arr[hole_wall], -hole_arr[~hole_wall], spacing, cfg)
    bias_gain = np.zeros((3, 0))
    try:
        if len(A):
            Aw = A * wts[:, None]
            Minv = np.linalg.inv(Aw.T @ A + 1e-6 * np.eye(3))
            cov_full = Minv @ ((A * (wts * wts)[:, None]).T @ A) @ Minv * sigma_obs ** 2
            scale = np.r_[np.full(len(edge_arr), erosion), np.full(len(hole_arr), erosion_hole)]
            bias_gain = ((Minv @ Aw.T) * scale[None, :])[:, ~walled]
        else:
            cov_full = np.eye(3) * 1e6
    except np.linalg.LinAlgError:
        cov_full = np.eye(3) * 1e6
    didx, _ = _datum_points(pc, model, 0.5, 0.0)
    cov_cells = _coverage_fraction(model, pc[didx], max(1.0, 2 * spacing))
    # the real sheet's thickness: both skins inside one tile, searched +-25 % around the CAD t
    tidx, th = _datum_points(pc, model, cfg.thickness_search_fraction * model.t + 0.3, 1.5)
    thick = tile_thickness(th[tidx], model.to2d(pc[tidx]), model.t, cfg) if len(tidx) else None
    edge_in = sum(1 for r in edge_res if abs(r) < 0.3) / max(con["expected_bins"], 1)
    # round holes are scored by their radial bins; slots and windows by their straight runs (edges)
    hole_bins_total = sum(_hole_bins(_hole_radius(hp), spacing, cfg) for hp in model.holes
                          if _hole_radius(hp) >= 1.0 and _round_hole(hp))
    hole_hits = sum(1 for g in con["hole_groups"] for r in g if abs(r - float(np.mean(g))) < 0.45 and abs(r) < 1.5)
    hole_in = (hole_hits / hole_bins_total) if hole_bins_total else None
    _, ok_f, kin_f, kf_f = model.kernel_cached(cfg.fine_pixel_mm, cfg.free_edge_band_mm, cfg.violation_weight,
                                               clearance=cfg.hole_core_clearance_mm,
                                               clearance_window=cfg.window_core_clearance_mm)
    _, cov_raster, viol = _score_direct(kin_f, kf_f, ok_f, cfg.fine_pixel_mm, uv, cfg.violation_weight)
    if hole_in is None:
        score = 0.55 * cov_raster + 0.45 * min(1.0, edge_in) - cfg.violation_weight * viol
    else:
        score = 0.4 * cov_raster + 0.3 * min(1.0, edge_in) + 0.3 * min(1.0, hole_in) - cfg.violation_weight * viol
    return InPlaneFit(
        T=T, information=info, length_scale=L, edge_bins=con["edge_bins"], edge_bins_expected=con["expected_bins"],
        holes_seen=con["holes_seen"], holes_total=len(model.holes),
        edge_rms=float(np.sqrt(np.mean(np.square(edge_res)))) if edge_res else float("nan"),
        hole_rms=float(np.sqrt(np.mean(np.square(hole_res)))) if hole_res else float("nan"),
        degenerate=degenerate, translation_cov=cov_full[1:, 1:],
        rotation_sigma_rad=float(math.sqrt(max(cov_full[0, 0], 0.0))) / L,
        pose_cov=np.asarray(cov_full, dtype=np.float64), undetermined=undetermined,
        skin_heights=heights, datum_points=int(len(didx)), datum_coverage=cov_cells,
        hole_mismatch=con["hole_mismatch"], score=float(score), two_skins=two_skins,
        bias_gain=bias_gain, erosion_mm=float(erosion),
        thickness_mm=None if thick is None else thick[0], thickness_sigma_mm=None if thick is None else thick[1],
        thickness_tiles=0 if thick is None else thick[2],
        detail={"min_information": float(evals.min()), "sigma_obs": sigma_obs, "edge_inlier": edge_in,
                "holes_hardware": int(con.get("holes_hardware", 0)),
                "hole_inlier": hole_in, "raster_coverage": cov_raster, "violation": viol,
                "walled_share": float(walled.mean()) if len(walled) else 0.0, "rows": int(len(walled)),
                "erosion_mm": float(erosion),
                "erosion_hole_mm": float(erosion_hole)})


# --------------------------------------------------------------------------- free directions: flange ends
@dataclass
class ChildEdgeRows:
    """In-plane pose rows observed on the cut ends of fitted child flanges (:func:`child_edge_rows`).

    Row aligned: ``rows`` are the sensitivities to (rotation * L, tx, ty) in the
    datum 2D frame (the convention of the datum's own edge rows), ``rhs`` minus
    the observed edge offset along its outward normal (mm), ``weights`` 1 for a
    bin observed at its cut wall and ``edge_unwalled_weight`` for a bare skin
    edge, ``edge`` the index of the edge (into ``labels`` / ``spread``) the row
    belongs to. ``spread`` is each edge's MAD-sigma of its bin offsets.
    """

    rows: np.ndarray
    rhs: np.ndarray
    weights: np.ndarray
    walled: np.ndarray
    edge: np.ndarray
    labels: list
    spread: list

    @classmethod
    def empty(cls) -> "ChildEdgeRows":
        return cls(np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64),
                   [], [])


def child_edge_rows(points: np.ndarray, point_owner: np.ndarray, part: PartView, model: DatumModel, panels,
                    cfg: InspectConfig, spacing: float, blockers: np.ndarray | None = None) -> ChildEdgeRows:
    """Rows of the datum's in-plane pose observed on the cut ends of fitted child flanges.

    A child panel reached from the datum only through bends whose axes are all
    parallel to one direction ``a`` of the datum plane moves, whatever the
    angles and bend-line positions of those bends, only across ``a``: the
    blank's cut edges of that panel that are perpendicular to ``a`` (the ends
    of a flange) stay where the blank put them along ``a``, formed or still
    flat. Their scanned positions therefore observe the datum frame along
    ``a`` exactly as the datum's own free edges do, and no bend-line position,
    angle or flange length the engine reports can move them.

    ``points`` are the registered full-resolution points and ``point_owner``
    their articulated panel (-1 none; points of other panels are never used);
    ``panels`` the articulated fit's :class:`PanelFit` list (duck-typed:
    ``state``, ``A``, ``n0``, ``d0``, ``t``). An edge qualifies when its
    outward in-panel normal lies within ``lateral_axis_tol_deg`` of the path's
    common axis and it is at least ``2 edge_end_trim + edge_bin`` long. It is
    binned like a datum free edge (``edge_bin_mm``; a bin seen at its cut wall
    is observed at the wall, else at the ``far_edge_rank``-th extreme skin
    point). ``blockers`` are articulated model points (registered frame) of
    every placed panel and bend zone: model material within the band beyond an
    edge (a slot's far side, a neighbour across a relief, the window a flat tab
    sits in) shortens the band to 0.6 mm short of it, and an edge whose band
    would be under 1 mm is not used.
    """
    if len(points) == 0 or not len(panels):
        return ChildEdgeRows.empty()
    tree = cKDTree(points)
    btree = cKDTree(blockers) if blockers is not None and len(blockers) else None
    cen2 = model.polygon.mean(axis=0)
    L = max(10.0, 0.5 * math.sqrt(model.area))
    gap = cfg.edge_gap_factor * spacing
    min_seg = 2 * cfg.edge_end_trim_mm + cfg.edge_bin_mm
    cos_tol = math.cos(math.radians(cfg.lateral_axis_tol_deg))
    band = cfg.edge_band_mm
    rows: list[list[float]] = []
    rhs: list[float] = []
    wts: list[float] = []
    wl: list[bool] = []
    eidx: list[int] = []
    labels: list[str] = []
    spread: list[float] = []
    for c, pf in enumerate(panels):
        if c == part.datum or c in part.rigid_followers:
            continue
        if getattr(pf, "state", None) not in ("FITTED", "FLAT") or pf.A is None or pf.n0 is None or pf.d0 is None:
            continue
        path = part.path_bends(c)
        if not path:
            continue
        a0 = unit(part.bends[path[0]].axis_dir)
        if abs(float(a0 @ model.n0)) > 1e-3:
            continue
        if any(abs(float(unit(part.bends[j].axis_dir) @ a0)) < 1.0 - 1e-6 for j in path):
            continue  # a bend across ``a`` on the way moves the panel along it
        pv = part.panels[c]
        pm = pv.model
        e1, e2 = orthonormal_frame(pv.n0)
        outer = np.asarray(pm.outer_boundary_mm, dtype=np.float64)
        poly = MplPath(np.stack([outer @ e1, outer @ e2], axis=1))
        hole_paths = []
        for hm in pm.holes_mm:
            hv = np.asarray(hm, dtype=np.float64)
            hole_paths.append(MplPath(np.stack([hv @ e1, hv @ e2], axis=1)))
        R = np.asarray(pf.A, dtype=np.float64)[:3, :3]
        nz = unit(np.asarray(pf.n0, dtype=np.float64))
        d0 = float(pf.d0)
        tc = float(pf.t if pf.t is not None else part.t)
        for fe in pm.free_edges:
            if not fe.registration_usable:
                continue
            line = np.asarray(fe.polyline_mm, dtype=np.float64)
            for si, (a, b) in enumerate(zip(line[:-1], line[1:])):
                d = b - a
                length = float(np.linalg.norm(d))
                if length < min_seg:
                    continue
                e = d / length
                n_in = np.cross(pv.n0, e)
                if float(np.linalg.norm(n_in)) < 1e-9:
                    continue
                n_in = unit(n_in)
                mid = 0.5 * (a + b)
                probes = np.array([mid + n_in * 0.5, mid - n_in * 0.5])
                p2 = np.stack([probes @ e1, probes @ e2], axis=1)
                ins = poly.contains_points(p2)
                for hp in hole_paths:
                    ins &= ~hp.contains_points(p2)
                if bool(ins[0]) == bool(ins[1]):
                    continue  # not a boundary of the material here
                n_out = -n_in if ins[0] else n_in
                if abs(float(n_out @ a0)) < cos_tol:
                    continue
                # the edge in the registered frame (the panel's fitted pose), normal kept in its fitted plane
                ar = apply(pf.A, a[None])[0]
                nr = R @ n_out
                nr = nr - float(nr @ nz) * nz
                if float(np.linalg.norm(nr)) < 0.5:
                    continue
                nr = unit(nr)
                er = R @ e
                er = er - float(er @ nz) * nz - float(er @ nr) * nr
                if float(np.linalg.norm(er)) < 0.5:
                    continue
                er = unit(er)
                n2 = np.array([float(nr @ model.f1), float(nr @ model.f2)])
                if float(np.linalg.norm(n2)) < cos_tol:
                    continue  # it must observe the datum frame within the datum plane
                n2 = n2 / float(np.linalg.norm(n2))
                lo, hi = cfg.edge_end_trim_mm, length - cfg.edge_end_trim_mm
                midr = ar + er * (0.5 * length)
                reach_out = band
                if btree is not None:
                    bsel = btree.query_ball_point(midr, r=0.5 * length + band + 2.0)
                    if bsel:
                        bq = blockers[np.asarray(bsel, dtype=np.int64)] - ar
                        b_al, b_of = bq @ er, bq @ nr
                        b_h = (bq + ar) @ nz - d0
                        near = (b_al > lo - 1.0) & (b_al < hi + 1.0) & (b_of > 0.3) & (b_of < band + 0.6) & \
                               (b_h < 0.5) & (b_h > -tc - 0.5)
                        if near.any():
                            reach_out = min(band, float(b_of[near].min()) - 0.6)
                if reach_out < 1.0:
                    continue  # material of the part lies just beyond: the extreme would be its
                sel = tree.query_ball_point(midr, r=0.5 * length + 2.0 * band + 1.0)
                if not sel:
                    continue
                sel = np.asarray(sel, dtype=np.int64)
                q = points[sel]
                rel = q - ar
                along, off = rel @ er, rel @ nr
                hh = q @ nz - d0
                own = point_owner[sel]
                keep = (along >= lo) & (along <= hi) & (off > -band) & (off < reach_out) & (hh < 0.5) & \
                       (hh > -tc - 0.5) & ((own < 0) | (own == c))
                if int(keep.sum()) < 3:
                    continue
                along, off, hh = along[keep], off[keep], hh[keep]
                if tc >= cfg.wall_min_thickness_mm:
                    margin = cfg.wall_skin_margin_fraction * tc
                    wall_pt = (hh < -margin) & (hh > -tc + margin)
                else:
                    wall_pt = np.zeros(len(hh), dtype=bool)
                bins = np.floor((along - lo) / cfg.edge_bin_mm).astype(np.int64)
                e_rows, e_obs, e_wall = [], [], []
                for bi in np.unique(bins):
                    m = bins == bi
                    cnt = int(m.sum())
                    if cnt < 3:
                        continue
                    om = off[m]
                    order = np.argsort(-om)
                    k = min(cfg.far_edge_rank, cnt) - 1
                    extreme = float(om[order[k]])
                    wsel = wall_pt[m] & (om > extreme - cfg.wall_zone_mm)
                    walled = int(wsel.sum()) >= cfg.reg_wall_min_points
                    obs = (float(np.median(om[wsel])) if walled else extreme) + gap
                    xb = ar + er * float(np.median(along[m])) + nr * obs
                    xr = model.to2d(xb[None])[0] - cen2
                    e_rows.append([float(n2 @ np.array([-xr[1], xr[0]])) / L, float(n2[0]), float(n2[1])])
                    e_obs.append(obs)
                    e_wall.append(walled)
                if len(e_obs) < 2:
                    continue
                ov = np.asarray(e_obs)
                k_edge = len(labels)
                labels.append(f"{pv.panel_id}/{fe.edge_id}#{si}")
                spread.append(float(1.4826 * np.median(np.abs(ov - np.median(ov)))))
                for r_, o_, w_ in zip(e_rows, e_obs, e_wall):
                    rows.append(r_)
                    rhs.append(-o_)
                    wts.append(1.0 if w_ else cfg.edge_unwalled_weight)
                    wl.append(bool(w_))
                    eidx.append(k_edge)
    if not rows:
        return ChildEdgeRows.empty()
    return ChildEdgeRows(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64),
                         np.asarray(wts, dtype=np.float64), np.asarray(wl, dtype=bool),
                         np.asarray(eidx, dtype=np.int64), labels, spread)


def resolve_free_directions(fit: InPlaneFit, T: np.ndarray, rows: ChildEdgeRows, model: DatumModel,
                            cfg: InspectConfig) -> tuple[InPlaneFit | None, dict]:
    """``fit`` with the in-plane directions its datum left undetermined fixed by flange-end rows.

    ``rows`` (:func:`child_edge_rows`) were observed with the scan registered by
    ``T`` (``fit.T``, or an earlier resolution of it). Only ``fit.undetermined``
    is moved: the directions the datum's own free edges and holes determine stay
    exactly as the datum fit left them (a flange can never pull them).

    An edge is used when its rows are sensitive to the undetermined directions
    (mean ``|row . u|`` at least ``lateral_min_sensitivity``) and consistent
    along its length (bin MAD-sigma at most ``lateral_edge_max_spread_mm``). The
    robust (Huber) solution is checked edge by edge: the edge whose median
    residual is worst is dropped while it exceeds ``lateral_outlier_mm``. A
    direction is resolved when at least ``lateral_min_edges`` consistent edges
    observe it and the datum's and the edges' information along it reach
    ``degenerate_min_information``; a solution moving the frame by more than
    ``lateral_max_shift_mm`` is refused. The covariance along the resolved
    directions is the edges' sandwich covariance plus their disagreement (the
    blank's cut tolerance: the edges' median residuals); unwalled rows add
    their erosion gain to ``bias_gain``. Returns ``(fit or None, detail)``.
    """
    U = np.asarray(fit.undetermined, dtype=np.float64).reshape(-1, 3)
    info: dict = {"edges_offered": len(rows.labels), "rejected": []}
    if len(U) == 0 or len(rows.rhs) == 0:
        info["reason"] = "NO_FREE_DIRECTION" if len(U) == 0 else "NO_FLANGE_END_EDGES"
        return None, info
    B_all = rows.rows @ U.T
    use: list[int] = []
    for k_e, label in enumerate(rows.labels):
        m = rows.edge == k_e
        sens = float(np.mean(np.linalg.norm(B_all[m], axis=1))) if m.any() else 0.0
        if sens < cfg.lateral_min_sensitivity:
            continue
        if rows.spread[k_e] > cfg.lateral_edge_max_spread_mm:
            info["rejected"].append(f"{label}:spread={rows.spread[k_e]:.3f}")
            continue
        use.append(k_e)
    info_u = U @ np.asarray(fit.information, dtype=np.float64) @ U.T
    while True:
        if len(use) < cfg.lateral_min_edges:
            info["reason"] = f"TOO_FEW_CONSISTENT_EDGES:{len(use)}"
            return None, info
        sel = np.isin(rows.edge, use)
        B, b, w, ed = B_all[sel], rows.rhs[sel], rows.weights[sel], rows.edge[sel]
        M = info_u + B.T @ B
        evals, evecs = np.linalg.eigh(M)
        keep = evals >= cfg.degenerate_min_information
        if not keep.any():
            info["reason"] = f"TOO_LITTLE_INFORMATION:{float(evals.max()):.2f}"
            return None, info
        V = evecs[:, keep]
        Br = B @ V
        r = Br.shape[1]
        wh = w.copy()
        z = np.zeros(r)
        for _ in range(5):
            H = (Br * wh[:, None]).T @ Br
            g = (Br * wh[:, None]).T @ b
            try:
                z = np.linalg.solve(H + 1e-9 * np.eye(r), g)
            except np.linalg.LinAlgError:
                info["reason"] = "SINGULAR"
                return None, info
            res = b - Br @ z
            scale = max(0.05, 1.4826 * float(np.median(np.abs(res))))
            wh = w * np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(res), 1e-12))
        res = b - Br @ z
        med = {k_e: float(np.median(res[ed == k_e])) for k_e in use}
        worst = max(use, key=lambda k_e: abs(med[k_e]))
        if abs(med[worst]) > cfg.lateral_outlier_mm:
            info["rejected"].append(f"{rows.labels[worst]}:residual={med[worst]:+.3f}")
            use.remove(worst)
            continue
        break
    Ur = U.T @ V  # 3 x r resolved pose directions (orthonormal columns)
    delta = Ur @ z
    if float(np.linalg.norm(delta)) > cfg.lateral_max_shift_mm:
        info["reason"] = f"SHIFT_TOO_LARGE:{float(np.linalg.norm(delta)):.3f}"
        return None, info
    # covariance along the resolved directions: scan noise (sandwich) + the edges' own disagreement
    walled_sel = rows.walled[sel]
    rr = res[walled_sel] if int(walled_sel.sum()) >= 5 else res
    sigma = max(0.02, 1.4826 * float(np.median(np.abs(rr - np.median(rr)))))
    try:
        Minv = np.linalg.inv((Br * w[:, None]).T @ Br + 1e-9 * np.eye(r))
    except np.linalg.LinAlgError:
        info["reason"] = "SINGULAR"
        return None, info
    cov_r = Minv @ ((Br * (w * w)[:, None]).T @ Br) @ Minv * sigma ** 2
    E = np.asarray([Br[ed == k_e].mean(axis=0) for k_e in use])
    r_e = np.asarray([med[k_e] for k_e in use])
    s_e = max(0.02, math.sqrt(float(np.sum(r_e ** 2)) / max(1, len(use) - r)))
    cov_sys = s_e ** 2 * np.linalg.pinv(E.T @ E)
    cov_z = cov_r + cov_sys
    P = np.eye(3) - Ur @ Ur.T
    pose_cov = P @ np.asarray(fit.pose_cov, dtype=np.float64) @ P + Ur @ cov_z @ Ur.T
    gain = Ur @ (Minv @ (Br * w[:, None]).T)
    unw = ~walled_sel
    bias_old = P @ fit.bias_gain if fit.bias_gain.size else np.zeros((3, 0))
    bias_gain = np.hstack([bias_old, gain[:, unw] * float(fit.erosion_mm)])
    remaining = (U.T @ evecs[:, ~keep]).T.copy()
    L = float(fit.length_scale)
    cen2 = model.polygon.mean(axis=0)
    F = np.stack([model.f1, model.f2, model.n0], axis=1)
    c3 = model.p0 + model.f1 * cen2[0] + model.f2 * cen2[1]
    r3 = np.eye(3)
    r3[:2, :2] = _rot2(float(delta[0]) / L)
    Rm = F @ r3 @ F.T
    T_new = rigid(Rm, c3 - Rm @ c3 + model.f1 * float(delta[1]) + model.f2 * float(delta[2])) @ np.asarray(T)
    information = np.asarray(fit.information, dtype=np.float64) + U.T @ (B.T @ B) @ U
    info.update({"edges_used": [rows.labels[k_e] for k_e in use], "rows": int(len(b)),
                 "information": [float(v) for v in evals], "shift": [float(v) for v in delta],
                 "sigma_obs": float(sigma), "edge_scatter": float(s_e),
                 "resolved": int(r), "remaining": int(len(remaining))})
    from dataclasses import replace

    out = replace(fit, T=T_new, information=information, degenerate=bool(len(remaining)),
                  translation_cov=pose_cov[1:, 1:].copy(),
                  rotation_sigma_rad=float(math.sqrt(max(float(pose_cov[0, 0]), 0.0))) / L,
                  pose_cov=pose_cov, undetermined=remaining, bias_gain=bias_gain,
                  detail={**fit.detail, "flange_end_edges": dict(info)})
    return out, info


def tile_thickness(h: np.ndarray, uv: np.ndarray, t_nominal: float, cfg: InspectConfig
                   ) -> tuple[float, float, int] | None:
    """Sheet thickness from the two skin levels seen inside the same 2D tile: (t, sigma, tiles) or None.

    ``h`` are heights along the panel normal, ``uv`` the in-plane coordinates.
    Per ``skin_tile_mm`` tile the densest 0.02 mm level (0.1 mm smoothed) is one
    skin; the other is the densest level whose separation lies within
    ``+-thickness_search_fraction`` of ``t_nominal`` and holds at least 15 % of
    the first's count. Each level is the median of the heights within 0.12 mm
    of its peak, so a residual tilt or bow of the panel does not mix the skins
    (a tile is small). The result is the median separation over the tiles that
    show both skins (at least ``thickness_min_tiles``), with the standard error
    of that median (floored at ``thickness_floor_mm``).
    """
    h = np.asarray(h, dtype=np.float64)
    if len(h) < 60:
        return None
    tile = cfg.skin_tile_mm
    key = np.floor(np.asarray(uv) / tile).astype(np.int64)
    k = key[:, 0] * 1_000_003 + key[:, 1]
    order = np.argsort(k, kind="stable")
    ks = k[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], len(ks)]
    lo_sep = (1.0 - cfg.thickness_search_fraction) * t_nominal
    hi_sep = (1.0 + cfg.thickness_search_fraction) * t_nominal
    hb, tol = 0.02, 0.12
    seps: list[float] = []
    for st, en in zip(starts, ends):
        if en - st < 40:
            continue
        hv = h[order[st:en]]
        base = float(hv.min()) - 0.5
        hist = np.bincount(np.floor((hv - base) / hb).astype(np.int64))
        sm = np.convolve(hist, np.ones(5), mode="same")
        p1 = int(np.argmax(sm))
        c1 = float(sm[p1])
        best = None
        for sign in (1.0, -1.0):
            j_lo = p1 + sign * lo_sep / hb
            j_hi = p1 + sign * hi_sep / hb
            a, b = int(math.ceil(min(j_lo, j_hi))), int(math.floor(max(j_lo, j_hi)))
            a, b = max(a, 0), min(b, len(sm) - 1)
            if b < a:
                continue
            jj = a + int(np.argmax(sm[a:b + 1]))
            if sm[jj] >= max(0.15 * c1, 8.0) and (best is None or sm[jj] > best[1]):
                best = (jj, float(sm[jj]))
        if best is None:
            continue
        l1 = base + (p1 + 0.5) * hb
        l2 = base + (best[0] + 0.5) * hb
        m1, m2 = np.abs(hv - l1) < tol, np.abs(hv - l2) < tol
        if m1.sum() < 8 or m2.sum() < 8:
            continue
        seps.append(abs(float(np.median(hv[m1])) - float(np.median(hv[m2]))))
    if len(seps) < cfg.thickness_min_tiles:
        return None
    arr = np.asarray(seps)
    med = float(np.median(arr))
    mad = 1.4826 * float(np.median(np.abs(arr - med)))
    sigma = max(cfg.thickness_floor_mm, 1.2533 * mad / math.sqrt(len(arr)))
    return med, float(sigma), int(len(arr))


def _erosion_scale(walled: np.ndarray, unwalled: np.ndarray, spacing: float, cfg: InspectConfig) -> float:
    """Inward erosion (mm) of rows seen without their wall: walled median minus unwalled median, clipped."""
    if len(walled) >= 5 and len(unwalled) >= 5:
        e = float(np.median(walled) - np.median(unwalled))
        return float(min(cfg.erosion_max_spacing * spacing, max(cfg.erosion_min_spacing * spacing, e)))
    return float(cfg.erosion_default_spacing * spacing)


def _hole_radius(hp: np.ndarray) -> float:
    c = hp.mean(axis=0)
    return float(np.mean(np.linalg.norm(hp - c, axis=1)))


def _inplane_constraints(uv: np.ndarray, model: DatumModel, segs, cfg: InspectConfig, gap: float, spacing: float,
                         cen2: np.ndarray, L: float, *, wide: bool, interior: np.ndarray | None = None,
                         proud: np.ndarray | None = None) -> dict:
    """Point-to-edge-line and radial hole rows (and residual statistics) for the 2D pose solve.

    ``interior`` flags the points between the two skins (cut walls). A bin whose
    wall is seen is observed at the wall's median (weight 1); otherwise at the
    skin's extreme, which a scanner erodes inward at a depth discontinuity, with
    weight ``edge_unwalled_weight``. ``walled`` / ``weights`` are row aligned.

    ``segs`` are ``(a, b, outward normal)`` cut edges, optionally with a fourth
    item: how far (mm) past the edge the band may reach (a non-round hole's
    straight run, :meth:`DatumModel.hole_segments`, stops short of the hole's
    opposite side). Round holes are binned radially about their centre;
    ``hole_mismatch`` counts holes filled with material (round ones by their
    inner disc where it is clear of compiled stage material, the others by
    their open core, :func:`_nonround_hole_filled`). ``proud`` (2D points of material
    standing proud of the datum slab) marks a hole holding installed hardware (a stud,
    standoff or clinch nut, :func:`_hardware_in_hole`): such a hole is counted in
    ``holes_hardware``, never as filled.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    weights: list[float] = []
    walled_rows: list[bool] = []
    edge_res: list[float] = []
    edge_walled: list[bool] = []
    hole_res: list[float] = []
    edge_bins = expected_bins = holes_seen = hole_mismatch = holes_hardware = 0
    proud_tree = cKDTree(proud) if proud is not None and len(proud) else None
    hole_groups: list[list[float]] = []
    hole_walled: list[list[bool]] = []
    tree = cKDTree(uv) if len(uv) else None
    inner_pts = interior if interior is not None else np.zeros(len(uv), dtype=bool)
    for seg in segs:
        a, b, nout = seg[0], seg[1], seg[2]
        reach = float(seg[3]) if len(seg) > 3 and seg[3] is not None else math.inf
        d = b - a
        length = float(np.linalg.norm(d))
        e = d / length
        lo, hi = cfg.edge_end_trim_mm, length - cfg.edge_end_trim_mm
        expected_bins += max(1, int((hi - lo) // cfg.edge_bin_mm))
        if tree is None:
            continue
        mid = a + e * (0.5 * length)
        sel = tree.query_ball_point(mid, r=0.5 * length + 2 * cfg.edge_band_mm + 1.0)
        if not sel:
            continue
        sel = np.asarray(sel)
        rel = uv[sel] - a
        along = rel @ e
        off = rel @ nout
        band = cfg.edge_band_mm * (2.0 if wide else 1.0)
        keep = (along >= lo) & (along <= hi) & (off > -band) & (off < min(band, reach))
        if keep.sum() < 3:
            continue
        along, off, wall_pt = along[keep], off[keep], inner_pts[sel][keep]
        bins = np.floor((along - lo) / cfg.edge_bin_mm).astype(np.int64)
        order = np.lexsort((-off, bins))
        bins_s, off_s, along_s, wall_s = bins[order], off[order], along[order], wall_pt[order]
        starts = np.r_[0, np.nonzero(np.diff(bins_s))[0] + 1]
        ends = np.r_[starts[1:], len(bins_s)]
        for st, en in zip(starts, ends):
            cnt = en - st
            if cnt < 3:
                continue
            k = min(cfg.far_edge_rank, cnt) - 1
            extreme = float(off_s[st + k])
            wall = wall_s[st:en] & (off_s[st:en] > extreme - cfg.wall_zone_mm)
            walled = int(wall.sum()) >= cfg.reg_wall_min_points
            obs = (float(np.median(off_s[st:en][wall])) if walled else extreme) + gap
            x_b = a + e * float(np.median(along_s[st:en])) + nout * obs
            xr = x_b - cen2
            rows.append([float(nout @ np.array([-xr[1], xr[0]])) / L, float(nout[0]), float(nout[1])])
            rhs.append(-obs)
            weights.append(1.0 if walled else cfg.edge_unwalled_weight)
            walled_rows.append(walled)
            edge_res.append(obs)
            edge_walled.append(walled)
            edge_bins += 1
    for k_hole, hp in enumerate(model.holes):
        c = hp.mean(axis=0)
        rads = np.linalg.norm(hp - c, axis=1)
        r_h = float(np.mean(rads))
        if not _round_hole(hp):
            # a slot or a window has no centre to bin about: its straight runs come in ``segs``
            # (DatumModel.hole_segments) and its fill is judged on its open core
            if tree is not None and hasattr(model, "hole_fill_cells") and \
                    _nonround_hole_filled(uv, model, k_hole, spacing):
                if _hardware_in_hole(proud_tree, hp, spacing):
                    holes_hardware += 1
                else:
                    hole_mismatch += 1
            continue
        if tree is None or r_h < 1.0:
            continue
        sel = np.asarray(tree.query_ball_point(c, r=r_h + cfg.hole_annulus_mm), dtype=np.int64)
        if len(sel) == 0:
            continue
        rel = uv[sel] - c
        rr = np.linalg.norm(rel, axis=1)
        # the fill test's disc lies inside the hole: 1 mm inside its *inscribed* circle. The mean radius r_h
        # exceeds the inscribed one on a near-round hole (an obround within the round test's 8 % radius
        # scatter), and a disc of r_h - 1 then reaches the bore's own cut walls on the straight sides
        r_ins = min(r_h, float(_distance_to_polyline(c[None, :], np.vstack([hp, hp[:1]]))[0]))
        r_in = max(0.5, r_ins - 1.0)
        inner = rr < r_in
        open_share = 1.0
        blocked_of = getattr(model, "blocked_points", None)
        if blocked_of is not None and len(blocked_of(k_hole)) and inner.any():
            # compiled stage material inside the hole's loop (the smaller bore of a stepped or countersunk
            # hole, a tab lying flat in it): scan material there is the part's own, never a filled hole
            inner_idx = np.nonzero(inner)[0]
            inner = np.zeros(len(rr), dtype=bool)
            inner[inner_idx[model.open_cells(k_hole, uv[sel][inner_idx])]] = True
            open_share = _open_disc_share(model, k_hole, c, r_in)
        if open_share >= _HOLE_FILL_MIN_OPEN_SHARE and \
                inner.sum() > max(8, 0.05 * open_share * np.pi * r_ins * r_ins / max(spacing * spacing, 1e-6)):
            if _hardware_in_hole(proud_tree, hp, spacing):
                holes_hardware += 1  # a stud / nut pressed into the hole: the hole is there, occupied
            else:
                hole_mismatch += 1  # material where the CAD has a hole
        ring = (rr > r_h - 1.5) & (rr < r_h + cfg.hole_annulus_mm)
        if ring.sum() < 6:
            continue
        nbins = _hole_bins(r_h, spacing, cfg)
        ang = np.arctan2(rel[ring, 1], rel[ring, 0])
        ab = np.floor((ang + math.pi) / (2 * math.pi) * nbins).astype(np.int64) % nbins
        rv, relv, wv = rr[ring], rel[ring], inner_pts[sel][ring]
        used = 0
        group: list[float] = []
        group_walled: list[bool] = []
        for bi in range(nbins):
            m = ab == bi
            if m.sum() < 2:
                continue
            order = np.argsort(rv[m])
            k = min(cfg.far_edge_rank, int(m.sum())) - 1
            rk = float(rv[m][order[k]])
            wall = wv[m] & (rv[m] < rk + cfg.wall_zone_mm)
            walled = int(wall.sum()) >= cfg.reg_wall_min_points_hole
            rb = (float(np.median(rv[m][wall])) if walled else rk) - gap
            u = relv[m][order[k]] / max(rk, 1e-9)
            x_b = c + u * rb
            res, nvec = rb - r_h, u
            xr = x_b - cen2
            rows.append([float(nvec @ np.array([-xr[1], xr[0]])) / L, float(nvec[0]), float(nvec[1])])
            rhs.append(-res)
            weights.append(1.0 if walled else cfg.edge_unwalled_weight)
            walled_rows.append(walled)
            hole_res.append(res)
            group.append(res)
            group_walled.append(walled)
            used += 1
        if group:
            hole_groups.append(group)
            hole_walled.append(group_walled)
        if used >= nbins // 2:
            holes_seen += 1
    return {"rows": rows, "rhs": rhs, "weights": weights, "walled": walled_rows, "edge_res": edge_res,
            "edge_walled": edge_walled, "hole_res": hole_res, "edge_bins": edge_bins,
            "expected_bins": expected_bins, "holes_seen": holes_seen, "hole_mismatch": hole_mismatch,
            "holes_hardware": holes_hardware,
            "hole_groups": hole_groups, "hole_walled": hole_walled}


#: A round hole whose inner disc is open (clear of compiled stage material) over less than this share is
#: not judged for fill: what the scan shows there is the part's own material.
_HOLE_FILL_MIN_OPEN_SHARE = 0.25


#: Material at least this far beyond either datum skin (twice the datum slab's own 0.5 mm skin allowance) stands
#: proud of the sheet: inside a CAD hole it is installed hardware, not sheet material.
_HARDWARE_PROUD_MM = 1.0


def _hardware_in_hole(proud_tree, hp: np.ndarray, spacing: float) -> bool:
    """Installed hardware (stud, standoff, clinch nut) in hole ``hp``: material proud of the datum slab inside
    the hole's inscribed circle, concentric with it.

    A press-fit fastener fills the hole's footprint at the skin heights (its shank or barrel) *and* stands
    out of the sheet (its head, body or thread); a part without that hole is flat sheet there and has nothing
    proud of it. Required: at least as many proud points inside the inscribed circle as the round fill test
    needs (8, or 5 % of the disc at the scan spacing), with their centroid within half that radius of the
    hole's centre (a fastener is concentric; a stray flange crossing the hole is not)."""
    if proud_tree is None:
        return False
    c = hp.mean(axis=0)
    r_ins = float(_distance_to_polyline(c[None, :], np.vstack([hp, hp[:1]]))[0])
    if r_ins <= 0.5:
        return False
    idx = proud_tree.query_ball_point(c, r=r_ins)
    need = max(8, 0.05 * np.pi * r_ins * r_ins / max(spacing * spacing, 1e-6))
    if len(idx) < need:
        return False
    cen = np.asarray(proud_tree.data)[idx].mean(axis=0)
    return bool(np.linalg.norm(cen - c) <= 0.5 * r_ins)


def _open_disc_share(model: "DatumModel", k: int, c: np.ndarray, r: float, cell: float = 0.25) -> float:
    """Share of the disc of radius ``r`` about ``c`` that is clear of compiled stage material in hole ``k``."""
    g = np.arange(-r, r + 1e-9, cell)
    xx, yy = np.meshgrid(g, g)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    cells = cells[np.hypot(cells[:, 0], cells[:, 1]) <= r] + c
    if len(cells) == 0:
        return 1.0
    return float(np.mean(model.open_cells(k, cells)))


def _hole_bins(r_h: float, spacing: float, cfg: InspectConfig) -> int:
    """Angular bins around a hole: about 1.5 scan spacings of arc each (8 .. hole_angular_bins)."""
    return int(max(8, min(cfg.hole_angular_bins, int(2 * math.pi * r_h / (1.5 * max(spacing, 0.1))))))


def _nonround_hole_filled(uv: np.ndarray, model: DatumModel, k: int, spacing: float) -> bool:
    """Scan material fills the open core of non-round hole ``k`` (a wrong part, or a cut that is not there).

    Judged on cells of two scan spacings (at least 1 mm). The core is the hole's
    interior at least one cell inside its boundary and clear of stage material;
    the ring is the datum material within 4 mm around the hole. A cell is
    covered when a datum-slab point lies within half a cell of its centre. The
    hole is filled when the scan covers at least 30 % of the core cells and at
    least half the ring's coverage. A ring covered below 30 % (that part of the
    datum was not seen) or a core of fewer than 4 cells (a narrow slot) is not
    judged: a cut wall's flare or a slot's own walls never read as material.
    """
    cell = max(1.0, 2.0 * float(spacing))
    core, ring = model.hole_fill_cells(k, cell, cell, 4.0)
    if len(core) < 4 or len(ring) < 4 or len(uv) == 0:
        return False
    hp = model.holes[k]
    lo, hi = hp.min(axis=0) - (cell + 6.0), hp.max(axis=0) + (cell + 6.0)
    near = uv[(uv[:, 0] > lo[0]) & (uv[:, 0] < hi[0]) & (uv[:, 1] > lo[1]) & (uv[:, 1] < hi[1])]
    if len(near) == 0:
        return False
    tree = cKDTree(near)
    d_ring, _ = tree.query(ring, k=1, distance_upper_bound=0.5 * cell, workers=1)
    ring_cov = float(np.mean(np.isfinite(d_ring)))
    if ring_cov < 0.3:
        return False
    d_core, _ = tree.query(core, k=1, distance_upper_bound=0.5 * cell, workers=1)
    core_cov = float(np.mean(np.isfinite(d_core)))
    return core_cov >= max(0.3, 0.5 * ring_cov)


def _material_side(pc: np.ndarray, h: np.ndarray, model: DatumModel, cfg: InspectConfig) -> int | None:
    """-1 when the datum's cut walls lie below its visible skin (material below), +1 above, None unknown.

    With a single visible skin the height alone cannot say whether it is the
    upper or the lower face. The walls of the free edges and holes span the
    thickness on the material side of that face, and oblique views see them.
    """
    near = np.abs(h) < model.t + 1.0
    idx = np.nonzero(near)[0]
    if len(idx) < 50:
        return None
    hv = h[idx]
    top, bottom = skin_peaks(hv, model.t)
    if bottom is not None:
        return None
    uv = model.to2d(pc[idx])
    dist, _ = model.boundary_tree("free").query(uv, k=1, workers=1)
    edge = dist < cfg.wall_side_band_mm
    rel = hv[edge] - top
    lo, hi = 0.25 * model.t, 0.9 * model.t
    below = int(((rel < -lo) & (rel > -hi)).sum())
    above = int(((rel > lo) & (rel < hi)).sum())
    if max(below, above) < cfg.wall_side_min_points or max(below, above) < 3 * min(below, above):
        return None
    return -1 if below > above else 1


def _segment_distance(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    ll = float(d @ d)
    if ll < 1e-18:
        return np.linalg.norm(q - a, axis=1)
    s = np.clip((q - a) @ d / ll, 0.0, 1.0)
    return np.linalg.norm(q - (a + s[:, None] * d), axis=1)


def _local_skins(h: np.ndarray, uv: np.ndarray, t: float, cfg: InspectConfig) -> np.ndarray:
    """Per-point skin (0 upper, 1 lower, -1 undecided) from local height peaks in 2D tiles.

    A small tilt of the current plane spreads the heights over the panel, but
    inside a tile the two skins stay one thickness apart, so the upper one
    (along n0) is skin 0. A tile with a single skin takes the side its peak is
    nearer to (relative to the current plane).
    """
    tile = cfg.skin_tile_mm
    key = np.floor(uv / tile).astype(np.int64)
    k = key[:, 0] * 1_000_003 + key[:, 1]
    order = np.argsort(k, kind="stable")
    ks = k[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], len(ks)]
    out = np.full(len(h), -1, dtype=np.int64)
    certain = np.zeros(len(h), dtype=bool)
    tol = max(0.15, min(0.4, 0.35 * t))
    for st, en in zip(starts, ends):
        sel = order[st:en]
        if len(sel) < 12:
            continue
        hv = h[sel]
        top, bottom = skin_peaks(hv, t)
        if bottom is not None:
            m0, m1 = np.abs(hv - top) < tol, np.abs(hv - bottom) < tol
            out[sel[m0]] = 0
            out[sel[m1]] = 1
            certain[sel[m0 | m1]] = True
        else:
            near0 = abs(top) <= abs(top + t)
            out[sel[np.abs(hv - top) < tol]] = 0 if near0 else 1
    return out, certain


def _nearest_on_polyline(p: np.ndarray, line: np.ndarray) -> np.ndarray:
    best, bd = line[0], np.inf
    for a, b in zip(line[:-1], line[1:]):
        d = b - a
        ll = float(d @ d)
        if ll < 1e-18:
            continue
        s = min(1.0, max(0.0, float((p - a) @ d) / ll))
        q = a + s * d
        dd = float(np.linalg.norm(p - q))
        if dd < bd:
            best, bd = q, dd
    return best


def _coverage_fraction(model: DatumModel, pts_cad: np.ndarray, pixel: float) -> float:
    """Fraction of the datum outline area (holes excluded) occupied by scan points."""
    if len(pts_cad) == 0:
        return 0.0
    x0, y0, x1, y1 = model.bbox
    gx = np.arange(x0, x1 + pixel, pixel)
    gy = np.arange(y0, y1 + pixel, pixel)
    xx, yy = np.meshgrid(gx, gy)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = MplPath(model.polygon).contains_points(cells)
    for hp in model.holes:
        inside &= ~MplPath(hp).contains_points(cells)
    n_in = int(inside.sum())
    if n_in == 0:
        return 0.0
    uv = model.to2d(pts_cad)
    q = np.floor((uv - np.array([x0, y0])) / pixel + 0.5).astype(np.int64)
    w, h = len(gx), len(gy)
    ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    occ = np.zeros(h * w, dtype=bool)
    occ[q[ok, 1] * w + q[ok, 0]] = True
    return float((occ & inside).sum()) / n_in


def points_on_plane(vox: VoxelCloud, n: np.ndarray, d: float, tol: float, seeds: np.ndarray) -> np.ndarray:
    """Voxel indices within ``tol`` of the plane and connected to ``seeds`` (region of one plane)."""
    dist = np.abs(vox.points @ n - d)
    return components_of(vox.graph, dist < tol, seeds)
