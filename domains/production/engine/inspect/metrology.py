"""Section metrology for formed bends (E3 step 5; plan §3.2, ENGINE_IO §1).

For every bend whose child flange was fitted (FORMED, PARTIALLY_FORMED or
UNEXPECTED_FORMED) the visible axial length is cut into 5-11 contiguous
**stations**. In each station the parent's and the child's OML planes are
fitted locally (robust Tukey IRLS on full-resolution points, the bend region
excluded up to each side's own flat material -- past the CAD tangent line, or
past the formed arc's tail when the real radius or its springback transition
runs on beyond it (:func:`_flat_start`) -- and never past an adjacent bend's
tangent line: a short riser is fitted on its own flat material, not on the arc
and flange beyond it) and give:

- the **included angle** ``180 - angle(n_parent_out, n_child_out)``. The angle is
  signed by the side the child lies on (:func:`~domains.production.engine.inspect.geom.signed_sweep_deg`,
  about the hinge axis): a child formed the wrong way, on the other side of its
  parent's plane, reads ``180 + angle`` (above 180, never its nominal), and its OML
  sharp, which sits on the wrong skin, carries no flange length
  (``BendMeasure.reversed``). Its bend line is the fold's own outer sharp (where
  the CAD's inner skins meet), and its position is measured by mirror symmetry
  (:func:`reversed_position_measurement`);
- the **OML virtual sharp**: the two OML planes intersected, evaluated at the
  station's axial position. When only the inner skin of a flange is visible,
  its OML is that skin moved by ``t``: the sheet thickness measured on the
  datum (both skins seen, :func:`~domains.production.engine.inspect.registration.tile_thickness`),
  else the compiled ``t``. The uncertainty of that offset (:class:`Thickness`)
  enters the U of every characteristic whose sharp it moves: by ``u_t /
  sin(sweep)`` along the other flange;
- the **bend-line position**: the sharp's offset from the decoded stage
  model's nominal OML sharp, perpendicular to the nominal line, projected on
  side A's (stage-pose) outward direction;
- the **flange length** per side, in that side's OML plane along the in-plane
  direction away from the bend: to the farthest material of the side panel at
  that station (``FREE_EDGE``; the visible cut wall, else a high order
  statistic of the skin so a stray flare point does not move it), to the
  measured OML sharp of the adjacent bend (``ADJACENT_BEND``, OML sharp to OML
  sharp), or, when an unplanned crease along the bend splits a FREE_EDGE side,
  to the crease's OML virtual sharp (UNKNOWN when that sharp is unresolved);
- mid-sequence, the **backgauge length** of the just-formed bend across the
  still-flat material (the side panel and every panel reached through unformed
  bends) to its far boundary. A not-yet-formed bend may carry a small kink, so
  every island panel is first **unfolded** onto the side panel's plane about
  the line where their fitted planes meet, with the developed-length
  correction ``theta (r_skin - R_n)`` of the kink (:func:`unfold_island`); the
  far edge is then found on the developed island. A run that stops while the
  panel's grown region continues beyond it (a plane the run could not follow)
  is **truncated**: that station is dropped, and too few stations left make the
  value UNKNOWN(AMBIGUOUS_SEGMENTATION), never a short length.

A bend whose median station sweep disagrees with its region-wide panel planes by
more than ``station_region_max_deg`` is not measured (AMBIGUOUS_SEGMENTATION): its
stations settled on another sheet, and every value from them would be confidently
wrong.

Stations are aggregated by the median (a dominant level when a minority of
stations sits on a notch or tab), with min/max reported. The expanded
uncertainty (:func:`u_budget`) combines the station scatter (standard error
of the median from the MAD), the plane systematics (tilt; the OML sharp's
displacement when a plane sits off), the far-edge localisation (a fraction of
the scan spacing) and its erosion when the cut wall is not seen, the
registration (bend-line position only: random, the worst-case erosion bias
and, when the datum showed only its skin 1, the frame height from ``t``), the
single-skin offsets (``u_t / sin(sweep)``, per side, in proportion to the
stations that used the inner skin) and, for a backgauge, the stage-model
nominal's K uncertainty (``sigma_K |dN/dK|``).

A FREE_EDGE far edge seen without its cut wall at most stations is a lower
bound of the length (a clamp or an out-of-view edge can end the visible
material short of the edge): it can prove a side too long, never too short, so
a short-side FAIL there is UNKNOWN(EDGE_NOT_SEEN) (:func:`bare_edge_untrusted`).
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from domains.production.engine.inspect.articulated import ArticulatedFit, HingeFrame, PanelFit
from domains.production.engine.inspect.cloud import VoxelCloud
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import (
    Plane,
    apply,
    rigid,
    robust_plane,
    rotation_between,
    signed_sweep_deg,
    skin_peaks,
    unit,
)
from domains.production.engine.inspect.part import BendView, PartView

__all__ = ["Measurement", "BendMeasure", "Thickness", "measure_bend_geometry", "measure_flange", "measure_backgauge",
           "SideProfile",
           "PointIndex", "unfold_island", "bare_edge_untrusted", "reversed_position_measurement"]


@dataclass(frozen=True)
class Thickness:
    """The sheet thickness every single-skin offset uses, and its standard uncertainty.

    ``u_mm`` combines the single-skin level systematics (``u_thickness_fraction
    * t``) with the uncertainty of ``t_mm`` itself: the datum's measured
    two-skin separation, or the compiled ``t`` with the stock tolerance
    (``u_thickness_unmeasured_fraction``). ``frame_mm`` is the uncertainty of the
    datum frame's height when the datum showed only its skin 1 (placed ``t``
    below skin 0), else 0.
    """

    t_mm: float
    u_mm: float
    measured: bool
    frame_mm: float = 0.0


@dataclass
class Measurement:
    value: float | None
    u: float | None = None
    count: int = 0
    median: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    reason: str | None = None
    method: str = ""
    support: int = 0
    coverage: float | None = None
    fit_rms: float | None = None
    notes: tuple[str, ...] = ()
    components: dict = field(default_factory=dict)
    #: FREE_EDGE far edges: stations whose cut wall was seen, of the stations with a far edge
    walls_seen: int = 0
    wall_stations: int = 0
    #: station-nominal comparison (``..._VS_STATION_NOMINAL``): the range of the station nominals and the median of
    #: the raw station values, so a caller can tell whether the decision depends on where the stations lie
    profile_range: tuple[float, float] | None = None
    raw_median: float | None = None

    @staticmethod
    def unknown(reason: str, method: str, note: str = "") -> "Measurement":
        return Measurement(None, reason=reason, method=method, notes=(note,) if note else ())


@dataclass
class BendMeasure:
    """Per-bend measured geometry shared by all of the bend's characteristics."""

    bend: int
    ok: bool
    reason: str | None = None
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    width: float = 0.0
    sharp: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # per valid station
    valid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: global OML planes (outward normals) of parent and child near the bend
    plane_parent: Plane | None = None
    plane_child: Plane | None = None
    line_point: np.ndarray | None = None
    line_dir: np.ndarray | None = None
    sweep_deg: float | None = None
    one_skin: bool = False
    #: share of the valid stations whose parent / child OML plane came from the inner skin + t
    inner_parent: float = 0.0
    inner_child: float = 0.0
    support_parent: int = 0
    support_child: int = 0
    coverage: float = 0.0
    fit_rms: float = 0.0
    visible: tuple[float, float] = (0.0, 0.0)
    #: per-station in-plane directions away from the bend, per panel index
    w: dict = field(default_factory=dict)
    #: residual scale of the station plane fits and the in-plane fit extent, per side
    rms_parent: float = 0.0
    rms_child: float = 0.0
    depth_parent: float = 10.0
    depth_child: float = 10.0
    sin_sweep: float = 1.0
    w_parent: np.ndarray | None = None
    w_child: np.ndarray | None = None
    apex: tuple[np.ndarray, np.ndarray] | None = None
    #: standard uncertainty (deg) of the angle from a joggle riser placed by its step (``PanelFit.model_plane``)
    model_u_deg: float = 0.0
    #: standard displacement (mm) of the bend's sharp along the other side from that angle uncertainty
    model_sharp_mm: float = 0.0
    #: the region-wide OML planes (parent, child) when a side was re-seated on its station strip
    #: (:func:`_reseat_plane`): a long FREE_EDGE side's far edge is retried on them
    region_planes: tuple | None = None
    #: the child turned the other way about the hinge axis (it lies on the other side of its parent's plane:
    #: formed the wrong way). ``sweep_deg`` stays the magnitude; the station angles are ``180 - signed sweep``
    #: (above 180), so the angle can never read its nominal; the OML sharp sits on the wrong skin
    reversed: bool = False
    #: a fold formed the wrong way: its own outer virtual sharp (where the CAD's inner skins meet) per station
    #: (NaN where unresolved) and its region-wide line point (axial origin of ``line_point``, direction
    #: ``line_dir``), and the parent's mid-plane ``(n, d)`` its bend-line position is mirrored across
    #: (:func:`reversed_position_measurement`)
    rev_sharp: np.ndarray | None = None
    rev_line_point: np.ndarray | None = None
    rev_mirror: tuple[np.ndarray, float] | None = None
    #: e3-1.13: the same bend measured with every near-tie skin / window choice taken the other way
    #: (:class:`_BranchRec`), None when no choice was within ``branch_tie_ratio`` of its alternative
    alt: "BendMeasure | None" = None
    #: e3-1.13: the near-tie choices met (kind strings) and the one-thickness skin swaps applied (side names)
    ties: tuple[str, ...] = ()
    skin_swaps: tuple[str, ...] = ()
    #: how far (mm) each side's station window was moved out past the CAD tangent line + 0.5 mm, onto the side's
    #: own flat material (:func:`_flat_start`; parent, child)
    flat_shift: tuple[float, float] = (0.0, 0.0)


class PointIndex:
    """Full-resolution registered points with per-point voxel ids and cached region masks."""

    def __init__(self, pts: np.ndarray, vox: VoxelCloud) -> None:
        self.pts = pts
        self.vox = vox
        self._masks: dict[tuple, np.ndarray] = {}

    def region_mask(self, key, voxels: np.ndarray) -> np.ndarray:
        if key not in self._masks:
            vm = np.zeros(len(self.vox.points), dtype=bool)
            if len(voxels):
                vm[voxels] = True
            self._masks[key] = vm[self.vox.inverse]
        return self._masks[key]


class _BranchRec:
    """Near-tie choices met while one bend is measured (e3-1.13 ``branch_withhold``).

    A discrete choice between two candidate skin levels (the densest level against the second one, the denser
    skin of a window) whose counts lie within ``branch_tie_ratio`` of each other is a near-tie: a 5 % change in
    the points can take it either way. ``flip`` takes every such choice the other way (the alternative reading).
    """

    def __init__(self, flip: bool, ratio: float) -> None:
        self.flip = flip
        self.ratio = ratio
        self.ties: list[str] = []
        self.swaps: list[str] = []


_BRANCH = threading.local()


def _branch() -> _BranchRec | None:
    return getattr(_BRANCH, "rec", None)


def _tie(kind: str, c1: float, c2: float) -> bool:
    """Record a near-tie between counts ``c1 >= c2`` and return True when the alternative is to be taken."""
    rec = _branch()
    if rec is None or c1 <= 0 or c2 < rec.ratio * c1:
        return False
    rec.ties.append(kind)
    return rec.flip


def _two_levels(h: np.ndarray, t: float, sep_min: float) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """The densest 0.05 mm level ``(level, count)`` of heights ``h`` and the densest other *peak* (a local maximum
    of the smoothed histogram) at least ``max(sep_min, 0.3)`` from it and not its partner skin (one thickness
    ``t`` away, +-0.3 mm), or None."""
    hb = 0.05
    lo = float(h.min()) - 1.0
    hist = np.bincount(np.floor((h - lo) / hb).astype(np.int64))
    sm = np.convolve(hist, np.ones(5), mode="same")
    p1 = int(np.argmax(sm))
    h1 = lo + (p1 + 0.5) * hb
    centres = lo + (np.arange(len(sm)) + 0.5) * hb
    pad = np.r_[np.zeros(3), sm, np.zeros(3)]
    peak = np.ones(len(sm), dtype=bool)
    for k in range(1, 4):
        peak &= (sm >= pad[3 - k:3 - k + len(sm)]) & (sm >= pad[3 + k:3 + k + len(sm)])
    sep = np.abs(centres - h1)
    ok = peak & (sep >= max(sep_min, 0.3)) & (np.abs(sep - t) > 0.3) & (sm > 0)
    if not ok.any():
        return (h1, float(sm[p1])), None
    p2 = int(np.argmax(np.where(ok, sm, -1.0)))
    return (h1, float(sm[p1])), (float(centres[p2]), float(sm[p2]))


def _oml_plane(panel: PanelFit, skin: int) -> Plane:
    return panel.skin_plane(skin)


def _skin_tol(cfg: InspectConfig) -> float:
    return 0.15


def _station_layout(zlo: float, zhi: float, cfg: InspectConfig) -> tuple[np.ndarray, float]:
    length = zhi - zlo
    n = int(round(length / cfg.station_target_mm))
    n = max(cfg.stations_min, min(cfg.stations_max, n))
    w = length / n
    z = zlo + w * (np.arange(n) + 0.5)
    return z, w


def _skin_level(d: np.ndarray, target: float, reach: float, tol: float, min_points: int) -> float | None:
    """Level of one skin near ``target`` (signed distances ``d`` to the panel plane), or None.

    The panel plane says where the skin should be; it is fitted over the whole
    region and can sit a few tenths off near one bend. When fewer than half of the
    densest level's points lie within ``tol`` of ``target``, that level (0.05 mm
    histogram within ``reach``) is taken instead.
    """
    win = d[np.abs(d - target) < reach]
    if len(win) < min_points:
        return None
    hb = 0.05
    lo = target - reach
    hist = np.bincount(np.clip(np.floor((win - lo) / hb).astype(np.int64), 0, None))
    sm = np.convolve(hist, np.ones(3), mode="same")
    peak = lo + (int(np.argmax(sm)) + 0.5) * hb
    at_peak = int((np.abs(win - peak) < tol).sum())
    at_target = int((np.abs(win - target) < tol).sum())
    take_target = at_target >= max(min_points, 0.5 * at_peak)
    # e3-1.13: the target and a distinct densest level (more than ``tol`` apart, both supported) nearly tie
    if abs(peak - target) > tol and min(at_target, at_peak) >= min_points:
        a, b = float(at_target), 0.5 * float(at_peak)
        if _tie("station_skin_level", max(a, b), min(a, b)):
            take_target = not take_target
    if take_target and at_target >= min_points:
        return target
    if at_peak < min_points:
        return None
    return float(np.median(win[np.abs(win - peak) < tol]))


def _fit_side(pts: np.ndarray, plane: Plane, t: float, cfg: InspectConfig) -> tuple[Plane | None, bool, int, float]:
    """Local OML plane from OML points (or IML points moved by t). Returns (plane, one_skin, support, rms)."""
    d = plane.distance(pts)
    tol = _skin_tol(cfg)
    reach = min(cfg.station_skin_reach_mm, 0.4 * t)
    for target, inner in ((0.0, False), (-t, True)):
        level = _skin_level(d, target, reach, tol, cfg.min_station_points)
        if level is None:
            continue
        sel = pts[np.abs(d - level) < tol]
        if len(sel) < cfg.min_station_points:
            continue
        pl = robust_plane(sel, iterations=3)
        if float(pl.n @ plane.n) < 0:
            pl = pl.flipped()
        return (pl.offset(t) if inner else pl), inner, len(sel), pl.rms
    return None, False, 0, 0.0


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """``(a, b, kept)`` of ``y = a + b x`` by least squares, three rounds rejecting residuals beyond 3 MAD-sigmas."""
    keep = np.ones(len(x), dtype=bool)
    sol = None
    for _ in range(3):
        if int(keep.sum()) < 10:
            return None
        A = np.c_[np.ones(int(keep.sum())), x[keep]]
        sol = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (sol[0] + sol[1] * x)
        mad = 1.4826 * float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        keep = np.abs(r) < max(3.0 * mad, 0.01)
    return float(sol[0]), float(sol[1]), keep


def _sagitta(x: np.ndarray, y: np.ndarray) -> float:
    """Signed sagitta (mm) of the parabola fitted to ``(x, y)`` over its span: ``c L^2 / 8``."""
    u = x - 0.5 * (float(x.min()) + float(x.max()))
    c = float(np.polyfit(u, y, 2)[0])
    return c * float(np.ptp(x)) ** 2 / 8.0


def _flat_start(s: np.ndarray, d: np.ndarray, z: np.ndarray, start0: float, depth: float, t: float,
                zlo: float, zhi: float, cfg: InspectConfig) -> float:
    """Where a side's flat material starts (at least ``start0``): past the formed arc's own tangent line.

    The station windows start 0.5 mm past the tangent line of the CAD radius. A real
    bend whose inner radius exceeds the CAD's (an air-bent 2.6 mm radius drawn as
    0.8 mm), or whose springback transition runs on past its arc, keeps bending
    beyond that line, and that tail, bending toward the bend's centre, tilts the
    station planes toward an underbent reading and moves their OML sharp. ``s`` is
    the distance from the bend's OML sharp line along the side, ``d`` the signed
    distance to the side's OML plane (outward positive) and ``z`` the axial position,
    for the side's slab points. Both skins' arcs end on the same line, so the side's
    denser skin (the OML one near 0 or the inner one near ``-t``) is tested, from
    ``start0`` outward in steps of ``flat_start_step_mm``. In each axial chunk
    (``flat_start_chunk_mm``) a line is fitted on up to ``flat_start_ref_mm`` of skin
    that starts ``flat_start_gap_mm`` beyond a ``flat_start_test_mm`` test strip (the
    gap keeps a slowly departing tail out of that line), chunks pooled:

    - a reference that is not straight (its parabola's sagitta beyond
      ``flat_start_dev_mm``: still on the tail, or a side curved throughout) or a strip
      that lost its skin (a deep tail leaves the skin band; a hole) cannot judge the
      strip, and the window moves on;
    - a strip whose median residual lies more than ``flat_start_dev_mm`` toward the
      centre (below the line) is on the tail, and the window moves on;
    - otherwise the strip is flat and the window starts there.

    A side with no such flat strip within ``flat_start_max_shift_mm`` (and the window
    keeping half its reference length) keeps the CAD start: a short tab curved
    throughout has no flat material to move to. A panel's bow (a radius of hundreds
    of millimetres) bends a few-millimetre reference by microns, and a tilted panel is
    a line too, so neither moves the window.
    """
    g, gap, step = cfg.flat_start_test_mm, cfg.flat_start_gap_mm, cfg.flat_start_step_mm
    ref_min = 0.5 * cfg.flat_start_ref_mm
    limit = min(start0 + cfg.flat_start_max_shift_mm, depth - g - gap - ref_min)
    if cfg.flat_start_max_shift_mm <= 0.0 or limit < start0 + step:
        return start0
    band = 0.45 * t
    sel = (s >= start0) & (s <= depth) & (z >= zlo) & (z <= zhi)
    ss, dd, zz = s[sel], d[sel], z[sel]
    need = cfg.min_station_points * cfg.stations_min
    skins = [(m, lv) for m, lv in ((np.abs(dd) < band, 0.0), (np.abs(dd + t) < band, -t)) if int(m.sum()) >= need]
    if not skins:
        return start0
    skins = sorted(skins, key=lambda k: -int(k[0].sum()))
    first = _flat_scan(ss, dd, zz, skins[0][0], skins[0][1], start0, limit, depth, band, zlo, zhi, cfg)
    if len(skins) == 2 and _branch() is not None:
        c1, c2 = float(skins[0][0].sum()), float(skins[1][0].sum())
        rec = _branch()
        if c2 >= rec.ratio * c1:
            # e3-1.13: the two skins nearly tie for the test; it matters only when the other skin flattens elsewhere
            other = _flat_scan(ss, dd, zz, skins[1][0], skins[1][1], start0, limit, depth, band, zlo, zhi, cfg)
            if abs(other - first) >= 2.0 * step - 1e-9 and _tie("flat_start_skin", c1, c2):
                return other
    return first


def _flat_scan(ss: np.ndarray, dd: np.ndarray, zz: np.ndarray, keep: np.ndarray, level: float, start0: float,
               limit: float, depth: float, band: float, zlo: float, zhi: float, cfg: InspectConfig) -> float:
    """:func:`_flat_start`'s outward walk on one skin (``keep``, at ``level``)."""
    g, gap, step = cfg.flat_start_test_mm, cfg.flat_start_gap_mm, cfg.flat_start_step_mm
    ss, dd, zz = ss[keep], dd[keep] - level, zz[keep]
    n_chunks = max(1, int(round((zhi - zlo) / cfg.flat_start_chunk_mm)))
    chunk = np.clip(((zz - zlo) / max(zhi - zlo, 1e-9) * n_chunks).astype(np.int64), 0, n_chunks - 1)
    groups = [np.nonzero(chunk == k)[0] for k in range(n_chunks)]
    s0 = start0
    while s0 <= limit + 1e-9:
        r0 = s0 + g + gap
        ref_len = min(cfg.flat_start_ref_mm, depth - r0)
        res, sags = [], []
        expected = 0.0
        for gi in groups:
            sg, dg = ss[gi], dd[gi]
            ref = (sg >= r0) & (sg < r0 + ref_len)
            if int(ref.sum()) < 20 or float(np.ptp(sg[ref])) < 0.6 * ref_len:
                continue
            line = _robust_line(sg[ref], dg[ref])
            if line is None:
                continue
            kept = line[2] if int(line[2].sum()) >= 10 else np.ones(int(ref.sum()), dtype=bool)
            sags.append(_sagitta(sg[ref][kept], dg[ref][kept]))
            tst = (sg >= s0) & (sg < s0 + g)
            r = dg[tst] - (line[0] + line[1] * sg[tst])
            res.append(r[(r > -band) & (r < 0.15)])
            expected += float(ref.sum()) * g / ref_len
        if expected < 20.0:
            break  # no reference to judge against
        r = np.concatenate(res)
        judged = abs(float(np.median(sags))) <= cfg.flat_start_dev_mm and len(r) >= 0.5 * expected
        if judged and float(np.median(r)) >= -cfg.flat_start_dev_mm:
            return float(s0)
        s0 += step
    return start0


def _in_plane_dir(plane: Plane, a: np.ndarray, toward: np.ndarray) -> np.ndarray:
    w = np.cross(a, plane.n)
    w = unit(w - float(w @ plane.n) * plane.n)
    return w if float(w @ toward) > 0 else -w


def measure_bend_geometry(idx: PointIndex, part: PartView, art: ArticulatedFit, j: int, cfg: InspectConfig
                          ) -> BendMeasure:
    """Station planes, angles and OML sharps of bend ``j`` (its child must be FITTED).

    e3-1.13: with ``branch_withhold`` the bend is measured with a :class:`_BranchRec`; when a near-tie choice was
    met, it is measured again with every such choice taken the other way, and that reading is ``alt``.
    """
    prev = _branch()
    try:
        rec = _BranchRec(False, cfg.branch_tie_ratio)
        _BRANCH.rec = rec
        out = _measure_bend_geometry(idx, part, art, j, cfg)
        out.ties, out.skin_swaps = tuple(rec.ties), tuple(rec.swaps)
        if cfg.branch_withhold and rec.ties and out.ok:
            alt_rec = _BranchRec(True, cfg.branch_tie_ratio)
            _BRANCH.rec = alt_rec
            alt = _measure_bend_geometry(idx, part, art, j, cfg)
            alt.ties, alt.skin_swaps = tuple(alt_rec.ties), tuple(alt_rec.swaps)
            out.alt = alt
        return out
    finally:
        _BRANCH.rec = prev


def _measure_bend_geometry(idx: PointIndex, part: PartView, art: ArticulatedFit, j: int, cfg: InspectConfig
                           ) -> BendMeasure:
    """Station planes, angles and OML sharps of bend ``j`` (its child must be FITTED)."""
    bend = part.bends[j]
    bf = art.bends[j]
    parent, child = art.panels[bend.parent], art.panels[bend.child]
    out = BendMeasure(bend=j, ok=False)
    if bf.frame is None or child.state != "FITTED" or parent.n0 is None or child.n0 is None:
        out.reason = bf.reason or "OCCLUDED"
        return out
    frame: HingeFrame = bf.frame
    t = float(parent.t or part.t)
    skin = bend.oml_skin
    pp = _oml_plane(parent, skin)
    pc = _oml_plane(child, skin)
    line = _intersect(pp, pc)
    if line is None:
        out.reason = "AMBIGUOUS_SEGMENTATION"
        return out
    lp, ld = line
    a = unit(ld if float(ld @ frame.a) > 0 else -ld)
    # signed by the side the child lies on (negative: formed the wrong way); its magnitude is the unsigned dihedral
    sweep = signed_sweep_deg(pp.n, pc.n, a, cfg.reversed_wrap_margin_deg)
    if abs(sweep) < 2.0:
        out.reason = "AMBIGUOUS_SEGMENTATION"
        return out
    ro = bend.ro
    depth = cfg.local_fit_depth_mm
    pts = idx.pts
    # a joggle riser placed by its step (no flat material of its own) keeps that plane at every station
    model_p, model_c = bool(parent.model_plane), bool(child.model_plane)
    need = cfg.min_station_points * cfg.stations_min
    pp0, pc0 = pp, pc
    for attempt in range(2):
        d_t = ro * math.tan(math.radians(min(abs(sweep), 170.0)) / 2.0)  # sharp -> tangent line along each flange
        rel = pts - lp
        z = rel @ a
        z_frame0 = float((lp - frame.o) @ frame.a)  # axial origin shift between the line point and the frame
        zlo_nom, zhi_nom = frame.z0 - z_frame0, frame.z1 - z_frame0
        w_p = _in_plane_dir(pp, a, -frame.u)  # into the parent (away from the bend)
        w_c = _in_plane_dir(pc, a, frame.line_dir(math.radians(bf.theta_deg or 0.0)))
        s_p = rel @ w_p
        s_c = rel @ w_c
        # the plane fits use the flat part of each flange only: past this bend's tangent line and
        # short of the side's far boundary (another bend's tangent line, or the cut edge and its wall)
        depth_p = _flat_depth(part, bend, bend.parent, d_t, depth, cfg)
        depth_c = _flat_depth(part, bend, bend.child, d_t, depth, cfg)
        inz = (z > zlo_nom - 1) & (z < zhi_nom + 1)
        if attempt == 1:
            break
        # a long or bowed panel sits off its region-wide plane near one bend (a 460 mm datum bowed by a few
        # millimetres at its ends): each fitted side's plane is re-seated on its own material in the strip the
        # stations use, before the stations look for it within a fraction of a millimetre of that plane
        moved = False
        for is_parent in (True, False):
            if (model_p if is_parent else model_c):
                continue
            strip = ((s_p > d_t + 0.5) & (s_p < depth_p) if is_parent else (s_c > d_t + 0.5) & (s_c < depth_c)) & inz
            panel = parent if is_parent else child
            q = _reseat_plane(pts[strip], pp if is_parent else pc, t, cfg, one_skin=len(panel.skins_seen) == 1,
                              side="parent" if is_parent else "child")
            if q is not None:
                moved = True
                if is_parent:
                    pp = q
                else:
                    pc = q
        if not moved:
            break
        line = _intersect(pp, pc)
        if line is None:
            out.reason = "AMBIGUOUS_SEGMENTATION"
            return out
        lp, ld = line
        a = unit(ld if float(ld @ frame.a) > 0 else -ld)
        sweep = signed_sweep_deg(pp.n, pc.n, a, cfg.reversed_wrap_margin_deg)
        if abs(sweep) < 2.0:
            out.reason = "AMBIGUOUS_SEGMENTATION"
            return out
    reversed_fold = sweep < 0.0
    dp = pp.distance(pts)
    dc = pc.distance(pts)
    slab = 0.5 * t + 0.6
    # each side's window starts on its own flat material: past the formed arc's tangent line, which a real radius
    # larger than the CAD's moves beyond the CAD one
    start_p = start_c = d_t + 0.5
    if not model_p:
        start_p = _flat_start(s_p, dp, z, start_p, depth_p, t, zlo_nom, zhi_nom, cfg)
    if not model_c:
        start_c = _flat_start(s_c, dc, z, start_c, depth_c, t, zlo_nom, zhi_nom, cfg)
    near_p = (np.abs(dp + 0.5 * t) < slab) & (s_p > start_p) & (s_p < depth_p) & inz
    near_c = (np.abs(dc + 0.5 * t) < slab) & (s_c > start_c) & (s_c < depth_c) & inz
    ip, ic = np.nonzero(near_p)[0], np.nonzero(near_c)[0]
    if (model_p and model_c) or (not model_p and len(ip) < need) or (not model_c and len(ic) < need):
        out.reason = "OCCLUDED"
        return out
    # visible axial extent: both flanges seen (the fitted side alone next to a model riser)
    ends = [np.sort(z[ii]) for ii, m in ((ip, model_p), (ic, model_c)) if not m]
    lo = max([float(np.quantile(zz, 0.005)) for zz in ends] + [zlo_nom])
    hi = min([float(np.quantile(zz, 0.995)) for zz in ends] + [zhi_nom])
    trim = min(2.0, 0.05 * (hi - lo))  # keep the lateral cut walls and corner rounds out of the stations
    lo, hi = lo + trim, hi - trim
    if hi - lo < cfg.min_visible_length_mm:
        out.reason = "OCCLUDED"
        return out
    zs, w = _station_layout(lo, hi, cfg)
    n = len(zs)
    sharp = np.full((n, 3), np.nan)
    rev_sharp = np.full((n, 3), np.nan) if reversed_fold else None
    ang = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    one_skin = False
    n_inner_p = n_inner_c = 0
    rms = []
    rms_p, rms_c = [], []
    for i, zi in enumerate(zs):
        mp = np.abs(z[ip] - zi) <= 0.5 * w
        mc = np.abs(z[ic] - zi) <= 0.5 * w
        fp, op, _, rp = (pp, False, 0, 0.0) if model_p else _fit_side(pts[ip[mp]], pp, t, cfg)
        fc, oc, _, rc = (pc, False, 0, 0.0) if model_c else _fit_side(pts[ic[mc]], pc, t, cfg)
        if fp is None or fc is None:
            continue
        li = _intersect(fp, fc)
        if li is None:
            continue
        p0, d0 = li
        da = float(d0 @ a)
        if abs(da) < 0.5:
            continue
        q = p0 + ((zi - float((p0 - lp) @ a)) / da) * d0
        sharp[i] = q
        if reversed_fold:
            # a child on the other side of its parent's plane: the included angle on the CAD's inner side is
            # 180 - (negative sweep), above 180, never the nominal
            ang[i] = 180.0 - signed_sweep_deg(fp.n, fc.n, a, cfg.reversed_wrap_margin_deg)
            # the fold's own outer virtual sharp: where the CAD's inner skins (the OML planes moved by t) meet
            q_r = _station_sharp(Plane(fp.n, float(fp.d) - t), Plane(fc.n, float(fc.d) - t), lp, a, zi)
            if q_r is not None:
                rev_sharp[i] = q_r
        else:
            c = max(-1.0, min(1.0, float(fp.n @ fc.n)))
            ang[i] = 180.0 - math.degrees(math.acos(c))
        valid[i] = True
        one_skin |= op or oc
        n_inner_p += int(op)
        n_inner_c += int(oc)
        rms.extend([rp, rc])
        rms_p.append(rp)
        rms_c.append(rc)
    if valid.sum() < max(3, int(math.ceil(cfg.min_station_fraction * n))):
        out.reason = "LOW_DENSITY"
        return out
    # the stations must describe the two panels the region-wide planes do: a station fit that settles on
    # another sheet (the flange beyond a short riser, seen on one skin) disagrees with them by tens of
    # degrees, and every value measured from it would be confidently wrong
    if abs((180.0 - float(np.median(ang[valid]))) - sweep) > cfg.station_region_max_deg:
        out.reason = "AMBIGUOUS_SEGMENTATION"
        return out
    # the bend's OML sharp line (the far boundary of its neighbours' ADJACENT_BEND lengths) is
    # the intersection of the region-wide panel planes: fitted over whole flanges, it is more
    # precise than the station sharps (on DEV, P95 0.058 vs 0.078 mm). When the station sharps
    # disagree with it by more than ``sharp_line_consistency_mm`` a panel plane is misfitted
    # near this bend (one visible skin, a twist) and the line is moved onto the station median
    # (direction and axial origin unchanged, so the station layout above stays valid).
    vs = sharp[valid]
    rel_s = vs - lp
    off = np.median(rel_s - np.outer(rel_s @ a, a), axis=0)
    if float(np.linalg.norm(off)) > cfg.sharp_line_consistency_mm:
        lp = lp + off
    # apex (outer surface at mid-sweep) from the sharp: back along the bisector by Ro (sec(theta/2) - 1)
    bis = unit(-(pp.n + pc.n)) if float(np.linalg.norm(pp.n + pc.n)) > 1e-9 else -pp.n
    back = ro * (1.0 / math.cos(math.radians(min(abs(sweep), 170.0)) / 2.0) - 1.0)
    a_lo, a_hi = lp + a * lo, lp + a * hi
    apex = (a_lo + bis * back, a_hi + bis * back)
    out.ok = True
    out.z = zs
    out.width = w
    out.sharp = sharp
    out.valid = valid
    out.angles = ang
    out.plane_parent, out.plane_child = pp, pc
    out.line_point, out.line_dir = lp, a
    out.sweep_deg = abs(sweep)
    out.reversed = reversed_fold
    if reversed_fold:
        out.rev_sharp = rev_sharp
        rl = _intersect(Plane(pp.n, float(pp.d) - t), Plane(pc.n, float(pc.d) - t))
        if rl is not None:
            out.rev_line_point = rl[0] - float((rl[0] - lp) @ a) * a
        out.rev_mirror = (np.asarray(pp.n, dtype=np.float64).copy(), float(pp.d) - 0.5 * t)
    out.one_skin = one_skin
    nv = max(int(valid.sum()), 1)
    out.inner_parent, out.inner_child = n_inner_p / nv, n_inner_c / nv
    out.support_parent, out.support_child = int(len(ip)), int(len(ic))
    out.coverage = float(min(1.0, (hi - lo) / max(zhi_nom - zlo_nom, 1e-9)))
    out.fit_rms = float(np.median(rms)) if rms else 0.0
    out.rms_parent = float(np.median(rms_p)) if rms_p else out.fit_rms
    out.rms_child = float(np.median(rms_c)) if rms_c else out.fit_rms
    out.depth_parent = float(max(depth_p - start_p, cfg.local_fit_min_depth_mm))
    out.depth_child = float(max(depth_c - start_c, cfg.local_fit_min_depth_mm))
    out.flat_shift = (float(start_p - (d_t + 0.5)), float(start_c - (d_t + 0.5)))
    out.sin_sweep = float(max(math.sin(math.radians(min(abs(sweep), 179.0))), 0.05))
    out.visible = (float(lo), float(hi))
    out.w = {bend.parent: w_p, bend.child: w_c}
    out.w_parent, out.w_child = w_p, w_c
    out.apex = apex
    out.model_u_deg = float(math.hypot(parent.model_u_deg if model_p else 0.0, child.model_u_deg if model_c else 0.0))
    if out.model_u_deg > 0.0:
        # the riser plane turns about its own middle: the sharp moves along the other side by lever * angle / sin
        lever = max(parent.model_lever_mm if model_p else 0.0, child.model_lever_mm if model_c else 0.0)
        out.model_sharp_mm = float(lever * math.radians(out.model_u_deg) / out.sin_sweep)
    if pp is not pp0 or pc is not pc0:
        out.region_planes = (pp0, pc0)
    _ = vs
    return out


def _reseat_plane(pts: np.ndarray, plane: Plane, t: float, cfg: InspectConfig, one_skin: bool = False,
                  side: str = "") -> Plane | None:
    """``plane`` (an OML plane, outward normal) re-seated on the material of a station strip, else None.

    The strip's points within ``local_reseat_max_mm`` outside and ``t +
    local_reseat_max_mm`` inside the plane give the skin levels (``skin_peaks``):
    with two skins the outer one is the OML, with one the reading nearer the
    plane's own levels (OML at 0, inner skin at ``-t``).

    e3-1.13: on a side whose panel region holds one skin (``one_skin``), two skins one thickness apart in the
    strip with the plane on the inner one (OML level near ``+t``) or one thickness outside the outer one (near
    ``-t``) mean the region took one skin for the other: the plane moves onto the outer skin (``+-t``), and the
    share guard counts that two-skin band (the strip also holds the arc and other sheets). With
    ``branch_withhold``, a densest level that a second, non-partner level nearly ties is a near-tie
    (:func:`_tie`); the alternative reading starts from the second level. When the OML level lies
    more than ``local_reseat_min_mm`` off, the plane is refitted on that skin's
    points (or the inner skin's, moved out by ``t``), three times on the band the
    last refit selects; a refit holding less than ``local_reseat_min_share`` of the
    strip's points, or tilted more than ``local_reseat_max_tilt_deg`` from the
    plane, is refused.
    """
    need = cfg.min_station_points * cfg.stations_min
    if len(pts) < need:
        return None
    d = plane.distance(pts)
    w = cfg.local_reseat_max_mm
    win = (d > -t - w) & (d < w)
    if int(win.sum()) < need:
        return None
    top, bottom = skin_peaks(d[win], t)
    if cfg.branch_withhold and _branch() is not None:
        first, second = _two_levels(d[win], t, cfg.local_reseat_min_mm)
        if second is not None and _tie("reseat_level", first[1], second[1]):
            dw = d[win]
            # the alternative: the second level's sheet (that level and its partner skin, if seen), the first
            # level's band left out
            near2 = dw[(np.abs(dw - second[0]) < t + 0.3) & (np.abs(dw - first[0]) >= 0.15)]
            top, bottom = skin_peaks(near2, t) if len(near2) >= 10 else (second[0], None)
    oml = top if (bottom is not None or abs(top) <= abs(top + t)) else top + t
    tol = min(0.25, 0.3 * t)
    swap = bool(cfg.one_skin_swap and one_skin and bottom is not None and abs(abs(oml) - t) < 0.3 * t
                and abs((top - bottom) - t) < 0.25 * t)
    if abs(oml) <= cfg.local_reseat_min_mm:
        return None
    inner = int((win & (np.abs(d - oml) < tol)).sum()) < need
    pl = plane.offset(oml - t if inner else oml)  # the seen skin's plane, refitted on its band (it widens as it turns)
    sel = None
    for _ in range(3):
        sel = np.abs(pl.distance(pts)) < tol
        if int(sel.sum()) < need:
            return None
        q = robust_plane(pts[sel], iterations=3)
        pl = q if float(q.n @ plane.n) > 0 else q.flipped()
    sel = np.abs(pl.distance(pts)) < tol
    # the refit must stand for the strip's material (a skin holds about half of a two-skin strip), not a narrow
    # band across a sheet that runs off at an angle
    base = len(pts)
    if swap:
        base = int((win & ((np.abs(d - top) < tol) | (np.abs(d - bottom) < tol))).sum())
    if int(sel.sum()) < cfg.local_reseat_min_share * base:
        return None
    if math.degrees(math.acos(max(-1.0, min(1.0, float(pl.n @ plane.n))))) > cfg.local_reseat_max_tilt_deg:
        return None
    if swap and _branch() is not None:
        _branch().swaps.append(side)
    return pl.offset(t) if inner else pl


def _flat_depth(part: PartView, bend: BendView, panel_idx: int, d_t: float, depth: float, cfg: InspectConfig) -> float:
    """Largest distance from the sharp (along the side) used for the plane fits.

    At least ``local_fit_min_depth_mm`` of flat flange, except toward an adjacent
    bend: the fit never runs past that bend's tangent line (its arc and the panel
    beyond, a joggle's far flange, would tilt a short side's planes), so a side
    shorter than the floor is fitted on what flat material it has (at least
    0.5 mm past this bend's tangent zone).
    """
    side = bend.side_of_panel(panel_idx)
    nom = part.nominals.get(f"{bend.uid}/FLANGE_LENGTH/{side}")
    limit = d_t + depth
    floor = d_t + 0.5 + cfg.local_fit_min_depth_mm
    if nom is not None and nom.nominal is not None:
        far = float(nom.nominal)
        if nom.boundary_kind == "ADJACENT_BEND" and nom.adjacent_bend_uid in part.bend_index:
            other = part.bends[part.bend_index[nom.adjacent_bend_uid]]
            far -= other.ro * math.tan(min(other.sweep, math.radians(170.0)) / 2.0) + 0.5
            floor = min(floor, max(far, d_t + 1.0))
        else:
            far -= 1.0
        limit = min(limit, far)
    return max(limit, floor)


def _station_sharp(p1: Plane, p2: Plane, lp: np.ndarray, a: np.ndarray, zi: float) -> np.ndarray | None:
    """Where the line of planes ``p1`` and ``p2`` crosses axial position ``zi`` (along ``a`` from ``lp``), else None."""
    li = _intersect(p1, p2)
    if li is None:
        return None
    p0, d0 = li
    da = float(d0 @ a)
    if abs(da) < 0.5:
        return None
    return p0 + ((zi - float((p0 - lp) @ a)) / da) * d0


def _intersect(a: Plane, b: Plane):
    d = np.cross(a.n, b.n)
    s = float(np.linalg.norm(d))
    if s < 1e-6:
        return None
    d = d / s
    m = np.array([a.n, b.n, d])
    p = np.linalg.solve(m, np.array([a.d, b.d, 0.0]))
    return p, d


def robust_location(values: np.ndarray) -> tuple[float, float, str]:
    """(value, MAD-sigma, method): the dominant level when a majority shares one, else the median."""
    value, sigma, how, _ = robust_levels(values)
    return value, sigma, how


def robust_levels(values: np.ndarray) -> tuple[float, float, str, np.ndarray]:
    """:func:`robust_location` and the station values it stands for (the dominant level's, else all)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    tol = max(0.1, 3.0 * mad)
    levels: list[list[float]] = [[float(v[0])]]
    for x in v[1:]:
        if x - levels[-1][-1] <= tol:
            levels[-1].append(float(x))
        else:
            levels.append([float(x)])
    top = max(levels, key=len)
    if len(top) >= 0.6 * len(v) and len(top) < len(v):
        lv = np.asarray(top)
        m = float(np.median(lv))
        return m, float(np.median(np.abs(lv - m))) * 1.4826, "DOMINANT_LEVEL", lv
    return med, mad, "MEDIAN", v


def station_spread(used: np.ndarray) -> float:
    """Standard uncertainty of a station median that stands for the whole visible bend: ``(q90 - q10) / (2 sqrt 3)``.

    The station values a median stands for can disagree systematically (a
    trend or a crowning along the bend, real or from the plane fits of a narrow
    flange); the standard error of the median covers random scatter only. A
    tolerance holds along the whole bend, so the 10-90 % range of those values,
    taken as a uniform bound, joins U (a lone outlier station, outside the
    dominant level or beyond the quantiles, does not).
    """
    u = np.asarray(used, dtype=np.float64)
    if len(u) < 3:
        return 0.0
    q10, q90 = np.quantile(u, [0.1, 0.9])
    return float(max(0.0, q90 - q10) / (2.0 * math.sqrt(3.0)))


def _se_median(sigma: float, n: int) -> float:
    return 1.2533 * sigma / math.sqrt(max(n, 1))


#: Floors (standard uncertainty) per measured quantity.
def _floor(kind: str, cfg: InspectConfig) -> float:
    return {"angle": cfg.u_floor_angle_deg, "position": cfg.u_floor_position_mm}.get(kind, cfg.u_floor_length_mm)


def u_budget(kind: str, comps: dict, cfg: InspectConfig) -> float:
    """Expanded uncertainty ``U = k * sqrt(sum (factor_i * c_i)^2)`` of a measurement.

    ``comps`` holds the unit components (standard uncertainties before their
    calibration factors): ``station`` (standard error of the station median),
    ``spread`` (the systematic disagreement of the stations the median stands
    for, :func:`station_spread`; factor 1),
    ``tilt_deg`` (plane residual over fit extent, both flanges), ``sharp_mm``
    and ``far_sharp_mm`` (the other plane's residual over ``sin(sweep)``: where
    the OML intersection moves when a plane is off), ``edge_mm`` (the scan
    spacing), ``pose_mm`` (datum frame, analytic: the in-plane fit's sandwich
    covariance), ``pose_bias_mm`` (the datum frame's worst-case shift from skin
    edges seen without their cut wall), ``erosion_mm`` (a far edge seen without
    its wall) and ``k_sigma_mm`` (stage-model K). ``thickness_mm`` (single-skin
    offsets: :class:`Thickness` ``u_mm`` carried through the geometry) and
    ``frame_height_mm`` are already standard contributions (factor 1), as is
    ``riser_model_deg`` (a joggle riser placed by its step: the angle that step
    implies under the riser's nominal length, ``tan(theta)`` times that length's
    relative uncertainty) and ``riser_model_mm`` (where that angle moves the
    riser's sharps). Factors live in the config.
    """
    terms = [
        comps.get("station", 0.0),
        comps.get("spread", 0.0),
        cfg.u_plane_tilt_factor * comps.get("tilt_deg", 0.0),
        cfg.u_sharp_factor * comps.get("sharp_mm", 0.0),
        cfg.u_sharp_factor * comps.get("far_sharp_mm", 0.0),
        cfg.u_edge_factor * comps.get("edge_mm", 0.0),
        comps.get("thickness_mm", 0.0),
        comps.get("frame_height_mm", 0.0),
        cfg.u_pose_factor * comps.get("pose_mm", 0.0),
        cfg.u_pose_bias_factor * comps.get("pose_bias_mm", 0.0),
        cfg.u_erosion_factor * comps.get("erosion_mm", 0.0),
        comps.get("k_sigma_mm", 0.0),
        comps.get("riser_model_deg", 0.0),
        comps.get("riser_model_mm", 0.0),
        _floor(kind, cfg),
    ]
    return float(cfg.coverage_factor * math.sqrt(sum(t * t for t in terms)))


def _rms(v: float, cfg: InspectConfig) -> float:
    return max(v, cfg.u_plane_rms_floor_mm)


def sharp_unit_along(bm: BendMeasure, panel_is_parent: bool, cfg: InspectConfig) -> float:
    """Sharp displacement along one side's in-plane direction (mm, 1 sigma).

    The *other* plane's residual and the skin-level uncertainty (setup
    misregistration, ``u_skin_level_mm``) over ``sin(sweep)``: where the OML
    intersection moves when that plane sits off.
    """
    rms = bm.rms_child if panel_is_parent else bm.rms_parent
    return float(math.hypot(_rms(rms, cfg), cfg.u_skin_level_mm) / bm.sin_sweep)


def _inner_share(share: float) -> float:
    """Weight of single-skin offsets in a station median: full once half the stations used the inner skin."""
    return float(min(1.0, 2.0 * max(0.0, share)))


def thickness_along(bm: BendMeasure, panel_is_parent: bool, thick: Thickness | None) -> float:
    """Sharp displacement along one side's in-plane direction from the *other* flange's inner skin + t (mm, 1 sigma).

    The other flange's OML plane moves by ``u_t`` along its normal when it came
    from the inner skin; the intersection then moves by ``u_t / sin(sweep)``
    inside this side's plane.
    """
    if thick is None:
        return 0.0
    share = bm.inner_child if panel_is_parent else bm.inner_parent
    return float(thick.u_mm * _inner_share(share) / bm.sin_sweep)


def angle_measurement(bm: BendMeasure, cfg: InspectConfig) -> Measurement:
    vals = bm.angles[bm.valid]
    value, sig, how, used = robust_levels(vals)
    comps = {"station": _se_median(sig, len(vals)),
             "tilt_deg": math.degrees(math.hypot(_rms(bm.rms_parent, cfg) / max(bm.depth_parent, 1.0),
                                                 _rms(bm.rms_child, cfg) / max(bm.depth_child, 1.0))),
             "spread": station_spread(used)}
    method = "STATION_OML_PLANES_"
    # diagnostic only (not a U term): how far each side's window moved onto its own flat material
    for key, shift in (("flat_shift_parent_mm", bm.flat_shift[0]), ("flat_shift_child_mm", bm.flat_shift[1])):
        if shift > 0.0:
            comps[key] = float(shift)
    if bm.model_u_deg > 0.0:
        # a joggle riser placed by its far flange's step: the angle is the one that step implies
        comps["riser_model_deg"] = float(bm.model_u_deg)
        method = "STATION_OML_PLANES_JOGGLE_STEP_RISER_"
    return Measurement(value=value, u=u_budget("angle", comps, cfg), count=int(len(vals)), median=float(np.median(vals)),
                       vmin=float(vals.min()), vmax=float(vals.max()), method=method + how,
                       support=bm.support_parent + bm.support_child, coverage=bm.coverage, fit_rms=bm.fit_rms,
                       components=comps)


def position_measurement(bm: BendMeasure, nominal_line: tuple[np.ndarray, np.ndarray], out_a: np.ndarray,
                         pose_sigma_mm: float, cfg: InspectConfig,
                         pose_bias: Callable[[np.ndarray, np.ndarray], float] | None = None,
                         thick: Thickness | None = None, datum_normal: np.ndarray | None = None) -> Measurement:
    """Bend-line position: station sharps vs the nominal OML sharp, along side A's outward direction.

    ``pose_bias(point, direction)`` bounds the datum frame's systematic shift there (mm).
    ``thick`` carries the single-skin offsets into U (a flange seen on its inner
    skin moves the sharp along the other flange) and, with ``datum_normal``, the
    frame height when the datum showed only its skin 1.
    """
    q0, q1 = nominal_line
    dn = unit(q1 - q0)
    oa = unit(out_a - float(out_a @ dn) * dn)
    vs = bm.sharp[bm.valid]
    rel = vs - q0
    off = rel - np.outer(rel @ dn, dn)
    vals = off @ oa
    value, sig, how = robust_location(vals)
    s_p = sharp_unit_along(bm, True, cfg) * abs(float(bm.w_parent @ oa))
    s_c = sharp_unit_along(bm, False, cfg) * abs(float(bm.w_child @ oa))
    comps = {"station": _se_median(sig, len(vals)), "pose_mm": pose_sigma_mm, "sharp_mm": math.hypot(s_p, s_c)}
    if bm.model_sharp_mm > 0.0:
        comps["riser_model_mm"] = float(bm.model_sharp_mm)
    if pose_bias is not None:
        comps["pose_bias_mm"] = float(pose_bias(0.5 * (q0 + q1), oa))
    t_p = thickness_along(bm, True, thick) * abs(float(bm.w_parent @ oa))
    t_c = thickness_along(bm, False, thick) * abs(float(bm.w_child @ oa))
    if t_p or t_c:
        comps["thickness_mm"] = math.hypot(t_p, t_c)
    if thick is not None and thick.frame_mm > 0 and datum_normal is not None:
        comps["frame_height_mm"] = float(thick.frame_mm * abs(float(oa @ datum_normal)))
    m = Measurement(value=value, u=u_budget("position", comps, cfg), count=int(len(vals)), median=float(np.median(vals)),
                    vmin=float(vals.min()), vmax=float(vals.max()), method="STATION_OML_SHARP_VS_STAGE_NOMINAL_" + how,
                    support=bm.support_parent + bm.support_child, coverage=bm.coverage, fit_rms=bm.fit_rms,
                    components=comps)
    m.notes = (f"normal_component_mm={float(np.median(off @ unit(np.cross(dn, oa)))):.4f}",)
    return m


def reversed_position_measurement(bm: BendMeasure, nominal_line: tuple[np.ndarray, np.ndarray], out_a: np.ndarray,
                                  pose_sigma_mm: float, cfg: InspectConfig,
                                  pose_bias: Callable[[np.ndarray, np.ndarray], float] | None = None,
                                  thick: Thickness | None = None, datum_normal: np.ndarray | None = None
                                  ) -> Measurement | None:
    """Bend-line position of a fold formed the wrong way, by mirror symmetry; None when unresolved.

    Reflected across the parent's mid-plane, the wrong-way fold is a fold of the
    same bend line turned the CAD way, and its own outer sharps (``rev_sharp``)
    become that fold's OML sharps. Those are compared with the nominal like any
    bend's (:func:`position_measurement`). The reflection keeps every offset
    within the parent's plane and removes only the side of the sheet the fold
    turned to, so a bend line placed off nominal reads off nominal either way.
    """
    if not bm.reversed or bm.rev_sharp is None or bm.rev_mirror is None:
        return None
    ok = bm.valid & np.isfinite(bm.rev_sharp).all(axis=1)
    if int(ok.sum()) < max(3, int(math.ceil(cfg.min_station_fraction * len(bm.z)))):
        return None
    n, d = bm.rev_mirror
    rs = np.where(ok[:, None], bm.rev_sharp, 0.0)
    refl = rs - 2.0 * (rs @ n - d)[:, None] * n[None, :]
    w_c = None if bm.w_child is None else bm.w_child - 2.0 * float(bm.w_child @ n) * n
    from dataclasses import replace

    m = position_measurement(replace(bm, sharp=refl, valid=ok, w_child=w_c), nominal_line, out_a, pose_sigma_mm, cfg,
                             pose_bias, thick=thick, datum_normal=datum_normal)
    m.method = m.method.replace("STATION_OML_SHARP", "STATION_REVERSED_FOLD_SHARP_MIRRORED", 1)
    m.notes = m.notes + ("BEND_FORMED_WRONG_WAY: own outer sharp mirrored across the parent mid-plane",)
    return m


@dataclass
class _FarEdge:
    value: float | None
    support: int
    walled: bool
    #: the panel's grown region continues beyond the run's end outside the material slab
    truncated: bool = False


def _far_free(pts: np.ndarray, region_mask: np.ndarray, plane: Plane, t: float, origin: np.ndarray, w: np.ndarray,
              a: np.ndarray, width: float, s_min: float, spacing: float, voxel: float, cfg: InspectConfig,
              flatten: tuple[np.ndarray, np.ndarray] | None = None) -> _FarEdge:
    """Farthest material of the side panel at one station (skins + cut wall), as a high order statistic.

    ``pts`` are the registered points (the developed island for a backgauge);
    the station is the slab ``|(x - origin) . a| <= width / 2``. The grown region
    establishes which material belongs to the panel; it may stop short of the
    cut edge (normals there are blurred by the wall), so the run is continued
    over contiguous material-slab points (gaps up to ``far_edge_gap_factor``
    spacings) before the order statistic is taken. ``walled`` when the cut wall
    gave the edge; otherwise the value is a skin extreme, eroded inward by the
    scanner. ``truncated`` when at least ``truncation_min_points`` points of the
    grown region lie more than ``max(truncation_gap_mm, 1.5 voxels)`` beyond the
    run's end outside the material slab: the panel continues where its plane
    does not (a bow, or a kink nothing unfolded), so the run's end is not its edge.

    ``flatten`` ``(z, offset)``: the CAD far boundary's offset from its value at
    the station centre across the station (a slanted or stepped far edge); each
    point's ``s`` is reduced by it first, so the far edge is taken at the centre.
    """
    rel = pts - origin
    z = rel @ a
    sl = np.abs(z) <= 0.5 * width
    if not sl.any():
        return _FarEdge(None, 0, False)
    sel = np.nonzero(sl)[0]
    d = plane.distance(pts[sel])
    mat = (d < cfg.far_edge_material_slack_mm) & (d > -t - cfg.far_edge_material_slack_mm)
    s = rel[sel] @ w
    if flatten is not None:
        s = s - np.interp(z[sel], flatten[0], flatten[1])
    reg_any = region_mask[sel] & (s > s_min)
    reg = mat & reg_any
    if reg.sum() < 3:
        return _FarEdge(None, int(reg.sum()), False)
    r_max = float(np.max(s[reg]))
    cand = np.sort(s[mat & (s > s_min)])
    gap = cfg.far_edge_gap_factor * max(spacing, 0.2) + 0.3
    far = r_max
    beyond = cand[cand > r_max]
    for v in beyond:
        if v - far > gap:
            break
        far = float(v)
    run = cand[(cand <= far + 1e-9) & (cand > r_max - 2.0)]
    k = min(cfg.far_edge_rank, len(run))
    if k == 0:
        return _FarEdge(None, 0, False)
    extreme = float(run[-k])
    reach = max(cfg.truncation_gap_mm, 1.5 * voxel)
    truncated = int((reg_any & ~mat & (s > far + reach)).sum()) >= cfg.truncation_min_points
    # the cut wall, when seen, is the edge itself: points between the two skins near the extreme
    if t >= cfg.wall_min_thickness_mm:
        sm = s[mat]
        dm = d[mat]
        margin = cfg.wall_skin_margin_fraction * t
        wall = (sm > extreme - cfg.wall_zone_mm) & (sm <= far + 1e-9) & (dm < -margin) & (dm > -t + margin)
        if wall.sum() >= cfg.wall_min_points:
            return _FarEdge(float(np.median(sm[wall])), int(reg.sum()), True, truncated)
    return _FarEdge(extreme, int(reg.sum()), False, truncated)


def _free_edge_stations(pts: np.ndarray, mask: np.ndarray, bm: BendMeasure, plane: Plane, t: float, w: np.ndarray,
                        d_t: float, spacing: float, voxel: float, cfg: InspectConfig, used: list[int] | None = None,
                        flatten_of: Callable[[int], tuple[np.ndarray, np.ndarray] | None] | None = None
                        ) -> tuple[list[float], list[bool], int, int]:
    """Per valid station of ``bm``: far-edge values and walls; also (support, truncated stations).

    ``used`` (when given) receives the station index of every returned value;
    ``flatten_of(i)`` the CAD far-edge offsets across station ``i`` (:func:`_far_free`).
    """
    vals: list[float] = []
    walls: list[bool] = []
    support = truncated = 0
    for i in np.nonzero(bm.valid)[0]:
        fl = flatten_of(int(i)) if flatten_of is not None else None
        fe = _far_free(pts, mask, plane, t, bm.sharp[i], w, bm.line_dir, bm.width, d_t, spacing, voxel, cfg,
                       flatten=fl)
        support += fe.support
        if fe.truncated:
            truncated += 1
            continue
        if fe.value is not None:
            vals.append(fe.value)
            walls.append(fe.walled)
            if used is not None:
                used.append(int(i))
    return vals, walls, support, truncated


# --------------------------------------------------------------------------- nominal station profile
#: A CAD far boundary varying by more than this (mm) across a station's width is flattened before the
#: station's far edge is taken, and a profile varying by more than this along the stations is compared
#: station by station; a straight far boundary parallel to the bend keeps the plain station statistic.
_PROFILE_MIN_SPREAD_MM = 0.05
#: The compiled nominal must lie within the CAD profile's range along the corridor (plus this, mm).
_PROFILE_RANGE_SLACK_MM = 0.05
#: Resolution (mm) of the profile samples across a station's width.
_PROFILE_STEP_MM = 0.1


def _uw_crossings(starts: np.ndarray, ends: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
    """``w`` where the station line ``u`` crosses each segment (E1's half-open rule) and the segment indices."""
    u0, u1 = starts[:, 0], ends[:, 0]
    lo, hi = np.minimum(u0, u1), np.maximum(u0, u1)
    idx = np.nonzero((lo <= u) & (u < hi))[0]
    if len(idx) == 0:
        return np.zeros(0), idx
    a, b = starts[idx], ends[idx]
    frac = (u - a[:, 0]) / (b[:, 0] - a[:, 0])
    return a[:, 1] + frac * (b[:, 1] - a[:, 1]), idx


class SideProfile:
    """The nominal flange length of one side as a function of the axial station position (final-pose CAD).

    E1 compiles one number per side over the whole corridor: the dominant level
    of a stepped far edge, the median of a taper, the median crossing of a
    non-parallel adjacent bend. A scan measures the *visible* stations, each
    over a station width, so on a stepped, notched or tapered far boundary its
    station values differ from that number on a perfect part. ``at(u)`` gives
    the nominal at axial position ``u`` (along the bend from the start of its
    OML sharp), with E1's definitions: FREE_EDGE, the farthest crossing of the
    side panel's boundary loops by the station line ``sharp(u) + s w``;
    ADJACENT_BEND, that line's crossing with the adjacent bend's OML sharp.
    None where no material crosses. It is used only when the compiled nominal
    lies within its range along the corridor (:meth:`covers`).
    """

    def __init__(self, kind: str, origin: np.ndarray, a_dir: np.ndarray, length: float, z0: float,
                 starts: np.ndarray | None = None, ends: np.ndarray | None = None,
                 adjacent: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> None:
        self.kind = kind
        self.origin = origin
        self.a_dir = a_dir
        self.length = length
        self.z0 = z0
        self.starts, self.ends = starts, ends
        self.adjacent = adjacent

    @classmethod
    def build(cls, part: PartView, j: int, side: str, boundary_kind: str | None,
              adjacent_uid: str | None) -> "SideProfile | None":
        bend = part.bends[j]
        a_dir = unit(bend.axis_dir)
        w_dir = unit(bend.side_out[side] - float(bend.side_out[side] @ a_dir) * a_dir)
        q0, _ = bend.locus("oml_virtual_sharp")
        origin = q0 + float((bend.axis_point + bend.z0 * a_dir - q0) @ a_dir) * a_dir
        length = float(bend.z1 - bend.z0)
        if length <= 1e-6:
            return None
        if boundary_kind == "ADJACENT_BEND":
            jb = part.bend_index.get(adjacent_uid or "")
            if jb is None:
                return None
            oq0, oq1 = part.bends[jb].locus("oml_virtual_sharp")
            od = oq1 - oq0
            if abs(float(od @ a_dir)) < 0.1 * float(np.linalg.norm(od)):
                return None  # the far bend runs along the station direction (E1: no nominal either)
            return cls("ADJACENT_BEND", origin, a_dir, length, float(bend.z0), adjacent=(oq0, od, w_dir))
        panel = part.panels[bend.side_panel[side]].model
        loops = [np.asarray(panel.outer_boundary_mm, dtype=np.float64)]
        loops += [np.asarray(h, dtype=np.float64) for h in panel.holes_mm]
        st, en = [], []
        for lp in loops:
            p = lp - origin
            uw = np.stack([p @ a_dir, p @ w_dir], axis=1)
            st.append(uw)
            en.append(np.roll(uw, -1, axis=0))
        return cls("FREE_EDGE", origin, a_dir, length, float(bend.z0), starts=np.concatenate(st),
                   ends=np.concatenate(en))

    def at(self, u: float) -> float | None:
        if self.kind == "ADJACENT_BEND":
            oq0, od, w_dir = self.adjacent
            tau = (u - float((oq0 - self.origin) @ self.a_dir)) / float(od @ self.a_dir)
            return float((oq0 + tau * od - self.origin) @ w_dir)
        w, _ = _uw_crossings(self.starts, self.ends, u)
        return float(w.max()) if len(w) else None

    def covers(self, nominal: float) -> bool:
        """The compiled nominal lies within this profile's range along the corridor."""
        vals = [v for v in (self.at(float(u)) for u in np.linspace(0.0, self.length, 201)) if v is not None]
        return bool(vals) and min(vals) - _PROFILE_RANGE_SLACK_MM <= nominal <= max(vals) + _PROFILE_RANGE_SLACK_MM

    def station_u(self, bm: BendMeasure, frame: HingeFrame, i: int) -> float:
        """Axial position of station ``i`` (its sharp carried into the bend's hinge frame)."""
        return float((bm.sharp[i] - frame.o) @ frame.a) - self.z0

    def across(self, u: float, half: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Profile samples ``(z, value)`` across a station of half width ``half`` at ``u`` (None: no material)."""
        zs = np.arange(-half, half + 1e-9, _PROFILE_STEP_MM)
        vals = [self.at(u + float(z)) for z in zs]
        ok = np.array([v is not None for v in vals])
        if not ok.any():
            return None
        return zs[ok], np.array([v for v in vals if v is not None], dtype=np.float64)


def _edge_unknown(vals: list[float], truncated: int, n_stations: int, method: str, cfg: InspectConfig
                  ) -> Measurement | None:
    """The UNKNOWN a FREE_EDGE far edge gets when too few of its stations can be trusted, else None.

    AMBIGUOUS_SEGMENTATION when truncated runs (material the run could not
    follow) took the stations it needs, else EDGE_NOT_SEEN. Far edges seen
    without their wall are judged once the nominal is known
    (:func:`bare_edge_untrusted`).
    """
    need = max(3, int(math.ceil(cfg.min_station_fraction * n_stations)))
    if len(vals) < need:
        if truncated and len(vals) + truncated >= need:
            return Measurement.unknown("AMBIGUOUS_SEGMENTATION", method,
                                       f"the far-edge run stops while the panel continues at {truncated} of "
                                       f"{n_stations} stations")
        return Measurement.unknown("EDGE_NOT_SEEN", method, f"{len(vals)} of {n_stations} stations")
    return None


def bare_edge_untrusted(m: Measurement, nominal: float | None, plus: float | None, minus: float | None,
                        cfg: InspectConfig) -> Measurement | None:
    """UNKNOWN(EDGE_NOT_SEEN) for a FREE_EDGE far edge that would FAIL short while its cut wall was not seen.

    A far edge seen without its wall (at ``bare_edge_station_share`` of its
    stations or more) only shows where the visible material ends: a scanner
    erodes such an edge by a few tenths (in U), and a clamp, a fixture or an
    out-of-view edge can end the visible material millimetres short of it. The
    reading is therefore a lower bound of the length: it can prove a side too
    long (material is there), never too short. A guard-band FAIL on the short
    side becomes UNKNOWN(EDGE_NOT_SEEN); a PASS, a REVIEW and a long-side FAIL
    stand. Only a seen cut wall proves where the material ends.
    """
    if m.value is None or m.u is None or m.wall_stations == 0 or nominal is None or plus is None or minus is None:
        return None
    bare = 1.0 - m.walls_seen / m.wall_stations
    if bare < cfg.bare_edge_station_share:
        return None
    d = round(float(m.value) - float(nominal), 6)
    if d < 0 and d + max(float(m.u), 1e-6) < -float(minus):
        return Measurement.unknown("EDGE_NOT_SEEN", m.method.rsplit("_", 1)[0] if m.method else "FREE_EDGE",
                                   f"short by {-d:.3f} mm to the visible end, but no cut wall at {bare:.0%} of the "
                                   f"stations: the material may continue where the scan does not see it")
    return None


def measure_flange(idx: PointIndex, part: PartView, art: ArticulatedFit, geoms: dict[int, BendMeasure], j: int,
                   side: str, boundary_kind: str | None, adjacent_uid: str | None, spacing: float,
                   cfg: InspectConfig, erosion_mm: float = 0.0, creases=(), thick: Thickness | None = None
                   ) -> Measurement:
    """Flange length of bend ``j`` on ``side`` (module docstring).

    ``erosion_mm`` is the scan's erosion scale of skin edges seen without their
    cut wall; it enters U in proportion to the stations without a visible wall.
    ``creases`` are (point, direction, U, id[, polyline]) OML virtual sharps of unplanned
    creases on the side panel: one running along this bend ends a FREE_EDGE side
    there (the contract measures to a bend's OML sharp, planned or not), at the
    stations its polyline spans (:func:`_crease_covers`).
    ``thick`` carries single-skin offsets into U. The measurement records how
    many FREE_EDGE stations saw the cut wall (``walls_seen``), for the bare-edge
    rule (:func:`bare_edge_untrusted`).
    """
    bm = geoms.get(j)
    bend = part.bends[j]
    if bm is None or not bm.ok:
        return Measurement.unknown((bm.reason if bm is not None else None) or "OCCLUDED", "STATION_OML")
    if bm.reversed:
        # formed the wrong way: the CAD's OML skin is the inner one of this fold, so the sharp the length starts
        # from is not the characteristic's locus
        return Measurement.unknown("UNSUPPORTED_FEATURE", "STATION_OML", "bend formed the wrong way")
    panel_idx = bend.side_panel[side]
    panel = art.panels[panel_idx]
    if panel.model_plane:
        # a joggle riser without flat material, placed by its far flange's step: its two OML sharps follow from
        # that step and the riser's nominal length, so its own length is not a measurement
        return Measurement.unknown("UNSUPPORTED_FEATURE", "STATION_OML",
                                   "joggle riser placed by its far flange's step; its own length is not measured")
    plane = bm.plane_parent if panel_idx == bend.parent else bm.plane_child
    w = bm.w[panel_idx]
    t = float(panel.t or part.t)
    ro = bend.ro
    d_t = ro * math.tan(math.radians(min(bm.sweep_deg or 90.0, 170.0)) / 2.0)
    vals: list[float] = []
    walls: list[bool] = []
    used: list[int] = []  # station index of every value (the nominal station profile follows them)
    flattened: set[int] = set()  # stations whose far edge was taken across a slanted / stepped CAD boundary
    support = 0
    t_far = 0.0
    frame = art.bends[j].frame
    nom_side = part.nominals.get(f"{bend.uid}/FLANGE_LENGTH/{side}")
    profile = None
    if frame is not None and nom_side is not None and nom_side.nominal is not None:
        profile = SideProfile.build(part, j, side, boundary_kind, adjacent_uid)
        if profile is not None and not profile.covers(float(nom_side.nominal)):
            profile = None
    if boundary_kind == "ADJACENT_BEND":
        jb = part.bend_index.get(adjacent_uid or "")
        other = geoms.get(jb) if jb is not None else None
        if other is None or not other.ok or other.reversed:
            return Measurement.unknown("ADJACENT_BEND_NOT_MEASURED", "OML_SHARP_TO_OML_SHARP")
        for i in np.nonzero(bm.valid)[0]:
            origin = bm.sharp[i]
            ls = _station_crossing(origin, bm.line_dir, w, other.line_point, other.line_dir)
            if ls is not None:
                vals.append(ls)
                used.append(int(i))
        method = "OML_SHARP_TO_OML_SHARP"
        u_edge = 0.0
        # the far sharp moves along this side when the far bend's other plane moves
        far_is_parent = part.bends[jb].parent == panel_idx
        u_far = sharp_unit_along(other, far_is_parent, cfg)
        t_far = thickness_along(other, far_is_parent, thick)
    else:
        u_far = 0.0
        along = [c for c in creases if abs(float(unit(c[1]) @ bm.line_dir)) >= 0.8 and _crease_covers(c, bm) is not False]
        if any(c[0] is None for c in along):
            return Measurement.unknown("AMBIGUOUS_SEGMENTATION", "OML_SHARP_TO_UNPLANNED_CREASE_SHARP",
                                       "an unplanned crease splits this side; its OML sharp is unresolved")
        crossed = _crease_crossings(bm, w, d_t, along)
        if len(crossed) >= max(3, int(math.ceil(cfg.min_station_fraction * len(bm.z)))):
            vals = [c[0] for c in crossed]
            u_far = max(c[1] for c in crossed)
            method = "OML_SHARP_TO_UNPLANNED_CREASE_SHARP"
            u_edge = 0.0
        else:
            mask = _panel_footprint_region(idx, part, art, panel_idx, cfg.free_edge_footprint_margin_mm)
            method = "OML_SHARP_TO_FREE_EDGE"
            flatten_of = None
            if profile is not None:
                half = 0.5 * float(bm.width)

                def flatten_of(i: int) -> tuple[np.ndarray, np.ndarray] | None:
                    u = profile.station_u(bm, frame, i)
                    acr, centre = profile.across(u, half), profile.at(u)
                    if acr is None or centre is None or float(np.ptp(acr[1])) <= _PROFILE_MIN_SPREAD_MM:
                        return None
                    flattened.add(i)
                    return acr[0], acr[1] - centre

            vals, walls, support, truncated = _free_edge_stations(idx.pts, mask, bm, plane, t, w, d_t, spacing,
                                                                  idx.vox.voxel, cfg, used=used,
                                                                  flatten_of=flatten_of)
            bad = _edge_unknown(vals, truncated, len(bm.z), method, cfg)
            if bad is not None and bm.region_planes is not None:
                # a side re-seated near its bend (a bowed panel) may leave that local plane over a long run: its
                # far edge is followed on the region-wide plane instead
                plane0 = bm.region_planes[0] if panel_idx == bend.parent else bm.region_planes[1]
                if plane0 is not plane:
                    used.clear()
                    flattened.clear()
                    vals, walls, support, truncated = _free_edge_stations(idx.pts, mask, bm, plane0, t, w, d_t,
                                                                          spacing, idx.vox.voxel, cfg, used=used,
                                                                          flatten_of=flatten_of)
                    bad = _edge_unknown(vals, truncated, len(bm.z), method, cfg)
            if bad is not None:
                return bad
            u_edge = spacing
    if len(vals) < max(3, int(math.ceil(cfg.min_station_fraction * len(bm.z)))):
        return Measurement.unknown("EDGE_NOT_SEEN", method, f"{len(vals)} of {len(bm.z)} stations")
    arr = np.asarray(vals)
    # a stepped, notched or tapered far boundary: each station against the nominal at that station
    prof = None
    if profile is not None and method != "OML_SHARP_TO_UNPLANNED_CREASE_SHARP" and len(used) == len(arr):
        exp = [profile.at(profile.station_u(bm, frame, i)) for i in used]
        if all(e is not None for e in exp):
            exp_arr = np.asarray(exp, dtype=np.float64)
            if flattened or float(np.ptp(exp_arr)) > _PROFILE_MIN_SPREAD_MM:
                prof = exp_arr
    prof_range = raw_med = None
    if prof is not None:
        prof_range, raw_med = (float(prof.min()), float(prof.max())), float(np.median(arr))
        nom_v = float(nom_side.nominal)
        dev, sig, how = robust_location(arr - prof)
        value = nom_v + dev
        arr = nom_v + (arr - prof)
        how = how + "_VS_STATION_NOMINAL"
    else:
        value, sig, how = robust_location(arr)
    comps = {"station": _se_median(sig, len(arr)), "edge_mm": u_edge,
             "sharp_mm": sharp_unit_along(bm, panel_idx == bend.parent, cfg), "far_sharp_mm": u_far,
             "erosion_mm": _erosion_term(walls, erosion_mm)}
    far_model = 0.0
    if boundary_kind == "ADJACENT_BEND" and other is not None:
        far_model = other.model_sharp_mm
    if bm.model_sharp_mm > 0.0 or far_model > 0.0:
        comps["riser_model_mm"] = float(math.hypot(bm.model_sharp_mm, far_model))
    t_near = thickness_along(bm, panel_idx == bend.parent, thick)
    if t_near or t_far:
        comps["thickness_mm"] = math.hypot(t_near, t_far)
    u = u_budget("length", comps, cfg)
    return Measurement(value=value, u=u, count=int(len(arr)), median=float(np.median(arr)), vmin=float(arr.min()),
                       vmax=float(arr.max()), method=method + "_" + how, support=support,
                       coverage=float(len(arr) / max(len(bm.z), 1)), fit_rms=bm.fit_rms,
                       components=comps, walls_seen=int(sum(walls)), wall_stations=len(walls),
                       profile_range=prof_range, raw_median=raw_med)


def _crease_covers(crease, bm: BendMeasure, z: float | None = None) -> bool | None:
    """Does the crease's own extent reach the bend's stations (or station position ``z``)? None: extent unknown.

    A crease is found where a segment hinges off the panel; it ends a FREE_EDGE side only
    over the length it actually runs, never along the whole flange as an unbounded line.
    """
    if len(crease) < 5 or crease[4] is None or len(crease[4]) < 2 or bm.z is None or len(bm.z) == 0:
        return None
    zc = (np.asarray(crease[4], dtype=np.float64) - bm.line_point) @ bm.line_dir
    half = 0.5 * float(bm.width)
    if z is None:
        return bool(zc.max() >= float(np.min(bm.z)) - half and zc.min() <= float(np.max(bm.z)) + half)
    return bool(zc.min() - half <= z <= zc.max() + half)


def _crease_crossings(bm: BendMeasure, w: np.ndarray, d_t: float, creases) -> list[tuple[float, float]]:
    """Per valid station: (distance to the nearest crease OML sharp along ``w``, its U) for creases along the bend
    whose extent covers that station (:func:`_crease_covers`)."""
    out: list[tuple[float, float]] = []
    along = [c for c in creases if c[0] is not None and abs(float(unit(c[1]) @ bm.line_dir)) >= 0.8]
    if not along:
        return out
    for i in np.nonzero(bm.valid)[0]:
        best = None
        for c in along:
            if _crease_covers(c, bm, float(bm.z[i])) is False:
                continue
            cp, cd, cu = c[0], c[1], c[2]
            ls = _station_crossing(bm.sharp[i], bm.line_dir, w, cp, unit(cd))
            if ls is not None and ls > d_t and (best is None or ls < best[0]):
                best = (ls, cu)
        if best is not None:
            out.append(best)
    return out


def _erosion_term(walls: list[bool], erosion_mm: float) -> float:
    """Erosion bias bound of a station median: full once half the stations lack a visible wall."""
    if not walls:
        return 0.0
    unwalled = 1.0 - float(np.mean(walls))
    return float(erosion_mm * min(1.0, 2.0 * unwalled))


def _station_crossing(origin, a, w, q0, d) -> float | None:
    """Station-line parameter ``s`` (origin + s w, in the plane normal to ``a``) where line (q0, d) crosses."""
    da = float(d @ a)
    if abs(da) < 0.1:
        return None
    tau = -float((q0 - origin) @ a) / da
    q = q0 + tau * d
    return float((q - origin) @ w)


def _panel_region(idx: PointIndex, art: ArticulatedFit, panel_idx: int) -> np.ndarray:
    panel = art.panels[panel_idx]
    vox = panel.region if len(panel.region) else panel.voxels
    return idx.region_mask(("panel", panel_idx), vox)


def _panel_footprint_region(idx: PointIndex, part: PartView, art: ArticulatedFit, panel_idx: int,
                            margin: float) -> np.ndarray:
    """The panel's grown region, kept within its fitted outline grown by ``margin`` (holes' cores excluded).

    A grown region follows coplanar material over the voxel graph, so it can
    run through a touching coplanar panel into another one far away (the lips and
    tabs on one face of a box): a FREE_EDGE far edge taken from it would be that
    other panel's edge. The run past the region's last point still follows
    contiguous material (:func:`_far_free`), so a flange cut long is measured long.
    """
    panel = art.panels[panel_idx]
    base = _panel_region(idx, art, panel_idx)
    if panel.A is None or panel.n0 is None:
        return base
    key = ("panel_footprint", panel_idx, float(margin))
    if key not in idx._masks:
        sel = np.nonzero(base)[0]
        out = np.zeros(len(base), dtype=bool)
        if len(sel):
            from domains.production.engine.inspect.articulated import in_footprint

            out[sel[in_footprint(idx.pts[sel], part, panel_idx, panel.A, panel.n0, margin)]] = True
        idx._masks[key] = out
    return idx._masks[key]


def unfold_island(idx: PointIndex, part: PartView, art: ArticulatedFit, side_panel: int, island: list[int],
                  crossed: list[int], skin: int, cfg: InspectConfig) -> tuple[np.ndarray, tuple[str, ...]]:
    """Registered points with every island panel unfolded onto the side panel's ``skin`` plane (developed lengths).

    The island is the side panel plus the panels reached through the crossed
    (not yet formed) bends. Walking out from the side panel, each panel ``q``
    reached from ``p`` through bend ``c`` is mapped onto ``p``'s plane by the
    rotation about the line where their fitted ``skin`` planes meet (a shift
    along the normal when they are parallel within ``unfold_min_angle_deg``),
    then moved toward that line by the kink's developed-length correction
    ``2 r_skin tan(theta / 2) - theta R_n``: along a skin through its virtual
    sharp a kinked strip is longer (``skin`` convex there, ``r_skin = Ri + t``) or
    shorter (concave, ``r_skin = Ri``) than along the neutral fibre the flat
    stage model S_k keeps. Composed outward from the side panel, the maps put
    the island where S_k has it, so its far edge is measured on developed
    material. The rotation's line lies on both fitted planes, so an error in
    where it sits along them moves the unfolded points by a second-order amount
    only. A point belongs to ``q`` when it lies in ``q``'s material slab, past the
    bend's hinge (the articulated fit's hinge frame) and within ``q``'s grown
    extent along it.

    Returns (points, notes); the points are ``idx.pts`` itself when nothing moved.
    """
    pts = idx.pts
    members = set(island)
    adj: dict[int, list[tuple[int, int]]] = {}
    for c in crossed:
        b = part.bends[c]
        adj.setdefault(b.parent, []).append((c, b.child))
        adj.setdefault(b.child, []).append((c, b.parent))
    order: list[tuple[int, int, int]] = []
    seen, queue = {side_panel}, [side_panel]
    while queue:
        cur = queue.pop(0)
        for c, other in sorted(adj.get(cur, ())):
            if other in seen or other not in members:
                continue
            seen.add(other)
            order.append((other, cur, c))
            queue.append(other)
    if not order:
        return pts, ()
    maps: dict[int, np.ndarray] = {side_panel: np.eye(4)}
    owner = np.full(len(pts), -1, dtype=np.int64)
    notes: list[str] = []
    for q, p, c in order:
        pf, qf = art.panels[p], art.panels[q]
        bf = art.bends[c]
        if p not in maps or pf.n0 is None or qf.n0 is None or qf.t is None or bf.frame is None:
            continue  # q stays where it is: a run that cannot follow it is truncated, never extended
        pl_p, pl_q = pf.skin_plane(skin), qf.skin_plane(skin)
        t = float(qf.t)
        frame = bf.frame
        into_q = frame.u if q == part.bends[c].child else -frame.u
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(pl_p.n @ pl_q.n)))))
        if ang > 30.0:
            continue  # no not-yet-formed kink is that steep (12 % of a sweep): the planes disagree, nothing moves
        line = _intersect(pl_p, pl_q) if ang >= cfg.unfold_min_angle_deg else None
        if line is None:
            m = rigid(np.eye(3), pl_p.n * (pl_p.d - pl_q.d))
        else:
            h0, e = line
            r3 = rotation_between(pl_q.n, pl_p.n)
            m = rigid(r3, h0 - r3 @ h0)
            u_p = unit(np.cross(e, pl_p.n))
            u_p = u_p if float(u_p @ into_q) < 0 else -u_p  # in p's plane, from the line into p
            u_q = unit(np.cross(e, pl_q.n))
            u_q = u_q if float(u_q @ into_q) > 0 else -u_q  # in q's plane, from the line into q
            bend_c = part.bends[c]
            theta = math.radians(ang)
            convex = float(u_q @ pl_p.n) < 0.0  # q bends away from the skin's outward side: the skin is outside
            r_skin = bend_c.ri + t if convex else bend_c.ri
            r_n = bend_c.ri + part.k_factor * t
            corr = 2.0 * r_skin * math.tan(0.5 * theta) - theta * r_n
            m = rigid(np.eye(3), u_p * corr) @ m
            notes.append(f"unfolded_{bend_c.label}_deg={ang:.3f}")
        maps[q] = maps[p] @ m
        # q's points: its material slab, past the bend's hinge (the articulated fit's frame: well placed
        # even when two nearly parallel planes meet far away), within its grown extent along the hinge
        d_mid = pl_q.d - 0.5 * t
        slab = np.abs(pts @ pl_q.n - d_mid) < 0.5 * t + cfg.grow_plane_tol_mm + cfg.far_edge_material_slack_mm
        rel = pts - frame.o
        mine = slab & (rel @ into_q > 0.0)
        regm = _panel_region(idx, art, q)
        if regm.any():
            ze_all = rel @ frame.a
            ze = ze_all[regm]
            lo, hi = float(ze.min()) - 2.0, float(ze.max()) + 2.0
            mine &= (ze_all >= lo) & (ze_all <= hi)
        owner[mine] = q
    moved = [q for q in maps if q != side_panel and (owner == q).any()]
    if not moved:
        return pts, tuple(notes)
    out = pts.copy()
    for q in moved:
        sel = owner == q
        out[sel] = apply(maps[q], pts[sel])
    return out, tuple(notes)


def measure_backgauge(idx: PointIndex, part: PartView, art: ArticulatedFit, geoms: dict[int, BendMeasure], j: int,
                      side: str, boundary_kind: str | None, adjacent_uid: str | None, island: list[int],
                      crossed: list[int], k_sigma_mm: float, spacing: float, cfg: InspectConfig,
                      erosion_mm: float = 0.0, thick: Thickness | None = None) -> Measurement:
    """Backgauge length of the just-formed bend ``j`` across the flat ``island`` panels (module docstring).

    ``crossed`` are the not-yet-formed bends between the island panels; the
    island is unfolded about them (:func:`unfold_island`) before the far edge is found.
    """
    bm = geoms.get(j)
    bend = part.bends[j]
    if bm is None or not bm.ok:
        return Measurement.unknown((bm.reason if bm is not None else None) or "OCCLUDED", "BACKGAUGE")
    if bm.reversed:
        return Measurement.unknown("UNSUPPORTED_FEATURE", "BACKGAUGE", "bend formed the wrong way")
    panel_idx = bend.side_panel[side]
    plane = bm.plane_parent if panel_idx == bend.parent else bm.plane_child
    w = bm.w[panel_idx]
    t = float(art.panels[panel_idx].t or part.t)
    d_t = bend.ro * math.tan(math.radians(min(bm.sweep_deg or 90.0, 170.0)) / 2.0)
    vals: list[float] = []
    walls: list[bool] = []
    support = 0
    t_far = 0.0
    notes: tuple[str, ...] = ()
    if boundary_kind == "ADJACENT_BEND":
        jb = part.bend_index.get(adjacent_uid or "")
        other = geoms.get(jb) if jb is not None else None
        if other is None or not other.ok or other.reversed:
            return Measurement.unknown("ADJACENT_BEND_NOT_MEASURED", "BACKGAUGE_TO_OML_SHARP")
        for i in np.nonzero(bm.valid)[0]:
            ls = _station_crossing(bm.sharp[i], bm.line_dir, w, other.line_point, other.line_dir)
            if ls is not None:
                vals.append(ls)
        method, u_edge = "BACKGAUGE_TO_OML_SHARP", 0.0
        far_is_parent = part.bends[jb].parent == panel_idx
        u_far = sharp_unit_along(other, far_is_parent, cfg)
        t_far = thickness_along(other, far_is_parent, thick)
    else:
        masks = [_panel_region(idx, art, p) for p in island if art.panels[p].state != "UNKNOWN"]
        if not masks:
            return Measurement.unknown("OCCLUDED", "BACKGAUGE_TO_FREE_EDGE")
        mask = np.logical_or.reduce(masks)
        method = "BACKGAUGE_TO_FREE_EDGE"
        pts, notes = unfold_island(idx, part, art, panel_idx, island, crossed, bend.oml_skin, cfg)
        vals, walls, support, truncated = _free_edge_stations(pts, mask, bm, plane, t, w, d_t, spacing,
                                                              idx.vox.voxel, cfg)
        bad = _edge_unknown(vals, truncated, len(bm.z), method, cfg)
        if bad is not None:
            return bad
        u_edge = spacing
        u_far = 0.0
    if len(vals) < max(3, int(math.ceil(cfg.min_station_fraction * len(bm.z)))):
        return Measurement.unknown("EDGE_NOT_SEEN", method, f"{len(vals)} of {len(bm.z)} stations")
    arr = np.asarray(vals)
    value, sig, how = robust_location(arr)
    comps = {"station": _se_median(sig, len(arr)), "edge_mm": u_edge, "k_sigma_mm": k_sigma_mm,
             "sharp_mm": sharp_unit_along(bm, panel_idx == bend.parent, cfg), "far_sharp_mm": u_far,
             "erosion_mm": _erosion_term(walls, erosion_mm)}
    t_near = thickness_along(bm, panel_idx == bend.parent, thick)
    if t_near or t_far:
        comps["thickness_mm"] = math.hypot(t_near, t_far)
    u = u_budget("length", comps, cfg)
    return Measurement(value=value, u=u, count=int(len(arr)), median=float(np.median(arr)), vmin=float(arr.min()),
                       vmax=float(arr.max()), method=method + "_" + how, support=support,
                       coverage=float(len(arr) / max(len(bm.z), 1)), fit_rms=bm.fit_rms, notes=notes[:4],
                       components=comps, walls_seen=int(sum(walls)), wall_stations=len(walls))
