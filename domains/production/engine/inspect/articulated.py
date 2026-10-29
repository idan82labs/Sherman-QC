"""Articulated fit (E3 step 3; plan §3.2 and §10 amendment 9).

The flange tree is walked from the registered datum, parents before children.
For every bend the child flange is **found in the scan**, not gated by its
nominal position, so a flange that is under- or over-bent, not yet formed or
cut too long is measured rather than dropped:

1. **Hinge band.** The bend's axis is carried to the scan by the parent's
   fitted pose. Full-resolution points within ``Ri + 2t + 5 mm`` of the axis
   (over the bend's axial span) are expressed in the cross-section frame
   ``(u, v)``: ``u`` is the parent's flat continuation (away from the parent),
   ``v = axis x u`` the side the child rotates toward.
2. **Theta search (Hough).** For every angle ``theta`` from slightly negative
   through flat (0) to the nominal sweep + 25 deg, the points on the child side
   (``s = u cos theta + v sin theta > 0``) are histogrammed by their offset
   ``rho = -u sin theta + v cos theta``. A flange is two parallel lines (both
   skins, one thickness apart) tangent to the bend's inner/outer cylinders;
   the score of ``theta`` is the best two-skin line count. Support starts past
   the bend's own arc tangent zone (every intermediate line touches the arc
   there). The best angle is refined on a 0.1 deg grid. The inner-skin line is
   searched around the offset the fold kinematics give at that angle
   (:meth:`HingeFrame.rho_iml_at`; a partially formed bend has not consumed its
   strip), within ``hinge_rho_slack_flat_mm + hinge_rho_slack_mm |sin theta|``
   (at most the slack, :func:`rho_window`): a bend line placed ``d`` off moves
   the child's line by ``d sin theta`` only, so a flat continuation must be the
   parent's own plane. A child that carries a bend parallel to its own within
   the fine reach (a joggle's 3 mm riser) is followed by a parallel sheet about
   one thickness off the parent; near flat, only material continuous from the
   hinge counts for it (``continuity`` in :func:`_hough`). When that
   leaves a bend of at least ``displaced_min_sweep_deg`` flat or partial, or
   formed but more than ``displaced_check_dev_deg`` off its nominal sweep, the
   formed-like angles are searched again with ``hinge_rho_slack_wide_mm`` over
   the fine reach, and a clearly better supported line (a formed flange whose
   bend line sits millimetres off nominal) replaces the narrow one, so the
   defect is measured instead of read as a half-formed bend (or as a line pair
   crossing that flange's two skins obliquely). On a child that carries a
   parallel bend the window stops, on the side toward the parallel bend, halfway to that bend's
   hinge (``grand_u`` in :func:`_hough`): a formed tab beyond a still flat riser
   is the parallel bend's line, not this one's displaced. Its line support also
   stops where the riser ends, at the parallel bend's tangent lines (``s_cut``,
   the limit the child's growth already uses): the arc and the sheet beyond them
   are that bend's, and a shallow line crossing them must not outscore the riser's
   own line. A line seen on one skin
   reads alike as either skin; the reading nearer the nominal kinematics is
   taken (:func:`_one_skin_tie`). A formed-like line the window reads on one
   skin is also read the other way, its partner one thickness past the window
   (:func:`_one_skin_reread`): a hinge frame placed a few millimetres off (an
   ancestor's bend line moved along the blank) leaves only the flange's near
   skin inside the window, and that skin must not be taken for the inner one.
3. **Region growing.** Seeds are the points on the winning line. The child's
   material slab (mid-plane +- t/2 + tolerance, voxel normals within a cone)
   is grown over the voxel neighbour graph from the seeds, the plane is refitted
   robustly and the growth repeated. Points already owned by another panel are
   never taken; a panel owns nothing inside the core of its own holes (a tab cut
   out of a window lies flat in it until formed, :func:`in_footprint`). The
   mid-plane follows the slab's skin height levels
   (:func:`~domains.production.engine.inspect.geom.skin_peaks`) at every step,
   so cut walls, flare and fused setups between the skins neither bias it nor
   walk it onto one skin; a slab seen on one skin is named by the hinge line
   pair it grew from (:func:`_one_skin_mid`).
4. **Pose.** The child's pose follows the fold kinematics
   (:func:`fold_transform`): the arc consumes ``theta * R_n`` of the bend's
   neutral strip, centred on the bend line (E1's Method A when flat, the
   identity at the nominal sweep), corrected along its normal to the fitted
   slab. Grandchildren are searched from that pose.

A child that shows neither a formed flange nor a flat continuation near its
hinge is UNKNOWN (OCCLUDED or LOW_DENSITY); everything below it in the tree is
UNKNOWN (BEND_STATE_UNKNOWN).

5. **Wrong-way fold (mirrored hinge).** A bend formed the wrong way turns its
   child toward ``-v``, about an axis on the other side of the sheet: the CAD
   axis reflected across the parent's mid-plane, with the parent's other skin as
   the fold's inner one (:class:`MirroredBend`, :func:`mirrored_hinge_frame`).
   In that mirrored frame the fold is the CAD fold's mirror image, so the same
   two-skin search reads it (:func:`_reversed_reading`, angles past the flat
   limits up to the wrap margin below 180, material no fitted panel owns). It
   replaces the CAD-direction reading only when it shows a sheet (a second skin
   parallel to its line, 0.4 to 1.6 thicknesses away: :func:`_sheet_skins`) and
   its support is at least ``reversed_support_ratio`` times the CAD reading's,
   and its child then grows with the mirrored fold kinematics. The child lies on
   the other side of its parent's plane, so the decode reads a negative sweep
   (formed the wrong way) and its angle FAILs (:mod:`.metrology`). The engine
   runs this search on its chosen pose only (``reversed_search``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

from domains.production.engine.inspect.cloud import VoxelCloud, components_of
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.geom import (
    Plane,
    apply,
    apply_dir,
    rigid,
    robust_plane,
    rotation_about_line,
    rotation_between,
    skin_peaks,
    unit,
)
from domains.production.engine.inspect.part import BendView, PartView

__all__ = ["PanelFit", "BendFit", "ArticulatedFit", "fit_articulated", "fold_transform", "hinge_frame", "HingeFrame",
           "inside_grown", "in_footprint", "rho_window", "pose_on_plane", "MirroredBend", "mirrored_bend",
           "mirrored_hinge_frame"]


@dataclass
class PanelFit:
    index: int
    state: str  # DATUM | FITTED | FLAT | UNKNOWN
    #: final-pose -> registered-frame transform of this panel (hinge kinematics)
    A: np.ndarray | None
    #: skin-0 outward normal and offset in the registered frame (n . x = d0)
    n0: np.ndarray | None = None
    d0: float | None = None
    t: float | None = None
    #: voxel indices owned by this panel
    voxels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    skins_seen: tuple[int, ...] = ()
    reason: str | None = None
    #: the whole grown region (may run past the panel into coplanar flat material)
    region: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: a joggle's riser placed by its far flange's step (:func:`_model_riser`), not by its own material: the
    #: metrology uses this plane on the riser's side instead of station fits, and ``model_u_deg`` (standard
    #: uncertainty of its angle) joins the U of the riser's bends' angles
    model_plane: bool = False
    model_u_deg: float = 0.0
    #: distance (mm) from the riser's own anchor (the S's middle) to either of its bend lines: the lever by
    #: which ``model_u_deg`` moves those bends' sharps
    model_lever_mm: float = 0.0

    def skin_plane(self, skin: int) -> Plane:
        """The fitted plane of contract skin ``skin`` with its outward normal."""
        assert self.n0 is not None and self.d0 is not None and self.t is not None
        if skin == 0:
            return Plane(self.n0.copy(), float(self.d0))
        return Plane(-self.n0, -(float(self.d0) - float(self.t)))


@dataclass
class BendFit:
    index: int
    found: bool
    flat: bool
    theta_deg: float | None = None
    rho_iml: float | None = None
    score: float = 0.0
    child_line_points: int = 0
    axial_coverage: float = 0.0
    reason: str | None = None
    frame: "HingeFrame | None" = None
    one_skin: bool = False
    detail: dict = field(default_factory=dict)
    #: the child was found formed the wrong way, about the mirrored hinge (``theta_deg`` is then negative: the
    #: angle in the CAD hinge frame ``frame``, which stays the CAD one for the decode and the metrology)
    reversed: bool = False


@dataclass
class HingeFrame:
    o: np.ndarray
    a: np.ndarray
    u: np.ndarray
    v: np.ndarray
    z0: float
    z1: float
    rho_iml: float  # signed nominal offset of the child's inner skin line
    delta: float  # rho_oml - rho_iml (signed thickness in rho)
    band: float
    #: ``(parent pose A, bend, inner tangent point of the child at its final pose)``: the fold kinematics
    #: that place the child's inner skin line at any angle (:meth:`rho_iml_at`); None keeps ``rho_iml``.
    fold: tuple | None = None

    def rho_iml_at(self, theta: float) -> float:
        """Expected offset of the child's inner skin line at angle ``theta`` (rad), by the fold kinematics.

        Equal to ``rho_iml`` at the nominal sweep. A partially formed bend has not
        consumed all of its strip, so its arc (and with it the child's line) sits
        farther along the parent (:func:`fold_transform`); a flat child is the
        parent's own plane continued.
        """
        if self.fold is None:
            return float(self.rho_iml)
        A, bend, it0 = self.fold
        p = apply(A @ fold_transform(bend, theta), it0[None])[0] - self.o
        return float(-math.sin(theta) * float(p @ self.u) + math.cos(theta) * float(p @ self.v))

    def coords(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = np.asarray(pts, dtype=np.float64) - self.o
        return rel @ self.a, rel @ self.u, rel @ self.v

    def line_normal(self, theta: float) -> np.ndarray:
        """3D unit normal of the child line at ``theta`` (pointing to +rho)."""
        return -math.sin(theta) * self.u + math.cos(theta) * self.v

    def line_dir(self, theta: float) -> np.ndarray:
        return math.cos(theta) * self.u + math.sin(theta) * self.v


@dataclass
class ArticulatedFit:
    panels: list[PanelFit]
    bends: list[BendFit]
    #: voxel -> owning panel index (-1 none)
    owner: np.ndarray


def hinge_frame(bend: BendView, parent: PanelFit, cfg: InspectConfig) -> HingeFrame:
    """Cross-section frame of ``bend`` carried to the scan by the parent's pose (module docstring)."""
    A = parent.A
    o = apply(A, bend.axis_point[None])[0]
    a = unit(apply_dir(A, bend.axis_dir[None])[0])
    n_p = parent.n0 if parent.n0 is not None else unit(apply_dir(A, bend.r_parent[None])[0])
    a = unit(a - float(a @ n_p) * n_p)
    u0 = -apply_dir(A, bend.p_out[None])[0]
    u = u0 - float(u0 @ n_p) * n_p
    u = unit(u - float(u @ a) * a)
    v = np.cross(a, u)
    # nominal child skin lines (final pose): tangent to the Ri / Ro cylinders
    it0, _ = bend.locus("inner_tangent_child")
    ot0, _ = bend.locus("outer_tangent_child")
    th = bend.sweep
    nrm2 = np.array([-math.sin(th), math.cos(th)])

    def rho_of(p_final: np.ndarray) -> float:
        p = apply(A, p_final[None])[0] - o
        return float(np.array([p @ u, p @ v]) @ nrm2)

    rho_i, rho_o = rho_of(it0), rho_of(ot0)
    band = bend.ri + 2.0 * bend.t + cfg.hinge_band_extra_mm
    return HingeFrame(o=o, a=a, u=u, v=v, z0=bend.z0, z1=bend.z1, rho_iml=rho_i, delta=rho_o - rho_i, band=band,
                      fold=(np.asarray(A, dtype=np.float64), bend, np.asarray(it0, dtype=np.float64)))


@dataclass(eq=False)
class MirroredBend:
    """A CAD bend folded the wrong way: its fold kinematics mirrored across the parent's mid-plane.

    The flat blank is the CAD's (:func:`fold_transform` at 0 is unchanged); from
    flat the child turns toward ``-v`` about ``axis_point`` (the CAD axis
    reflected across the parent's mid-plane, final-pose frame), consuming the
    same neutral strip. Skin 0 still continues across the bend, so the fold's
    outer skin is the CAD's inner one: ``oml_skin`` is the CAD's other skin.
    Everything else (panels, nominals, loci) is the CAD bend's (``base``).
    """

    base: BendView
    axis_point: np.ndarray

    @property
    def parent(self) -> int:
        return self.base.parent

    @property
    def child(self) -> int:
        return self.base.child

    @property
    def oml_skin(self) -> int:
        return 1 - int(self.base.oml_skin)

    @property
    def sweep(self) -> float:
        return float(self.base.sweep)


def mirrored_bend(part: PartView, bend: BendView) -> MirroredBend:
    """``bend`` folded the wrong way: its axis reflected across the parent's mid-plane (CAD final pose)."""
    pv = part.panels[bend.parent]
    n = unit(np.asarray(pv.n0, dtype=np.float64))
    mid = np.asarray(pv.p0, dtype=np.float64) - 0.5 * float(part.t) * n
    ap = np.asarray(bend.axis_point, dtype=np.float64)
    return MirroredBend(bend, ap - 2.0 * float((ap - mid) @ n) * n)


def mirrored_hinge_frame(part: PartView, bend: BendView, parent: PanelFit, frame: HingeFrame
                         ) -> tuple[HingeFrame, MirroredBend]:
    """The hinge frame of ``bend`` folded the wrong way, and its :class:`MirroredBend`.

    Its axis is the mirrored axis carried by the parent's pose, ``a' = -a``,
    ``u' = u`` and ``v' = -v``: in it the wrong-way fold has the CAD fold's
    cross-section exactly (the reflection keeps ``(u, v)`` coordinates and turns
    ``z`` over), so the CAD fold's line offsets and band apply unchanged, and the
    child's inner line follows the mirrored kinematics (its inner tangent line is
    the CAD's outer one).
    """
    mb = mirrored_bend(part, bend)
    A = np.asarray(parent.A, dtype=np.float64)
    o = apply(A, mb.axis_point[None])[0]
    ot0, _ = bend.locus("outer_tangent_child")
    fm = HingeFrame(o=o, a=-frame.a, u=frame.u.copy(), v=-frame.v, z0=-frame.z1, z1=-frame.z0,
                    rho_iml=frame.rho_iml, delta=frame.delta, band=frame.band,
                    fold=(A, mb, np.asarray(ot0, dtype=np.float64)))
    return fm, mb


def rho_window(frame: HingeFrame, theta: float, slack: float, cfg: InspectConfig) -> tuple[float, float]:
    """(centre, half-width) of the child's inner-skin line offset searched at angle ``theta`` (rad).

    The centre follows the fold kinematics (:meth:`HingeFrame.rho_iml_at`). A
    bend line placed ``d`` off its nominal moves the child's line by ``d sin
    theta`` only, so the half-width grows from ``hinge_rho_slack_flat_mm`` at
    flat with ``slack |sin theta|`` up to ``slack``: a flat continuation must be
    the parent's own plane, never a parallel sheet millimetres off it (the far
    flange of a joggle, a return beyond a short step).
    """
    half = min(float(slack), float(cfg.hinge_rho_slack_flat_mm) + float(slack) * abs(math.sin(theta)))
    return frame.rho_iml_at(theta), half


def _hough(zc, uc, vc, frame: HingeFrame, thetas: np.ndarray, cfg: InspectConfig, s_max: float,
           continuity: bool = False, grand_u: float | None = None, s_cut=None):
    """Best two-skin line per angle: (score, rho_iml, n_iml, n_oml) arrays.

    ``continuity``: at angles within ``flat_continuity_max_deg`` of flat, a
    candidate line counts only when the material on it is continuous from the
    hinge: every ``flat_continuity_step_mm`` sub-band of the first
    ``flat_continuity_mm`` past the arc's tangent zone holds at least a fifth of
    the line's mean support per sub-band (and 3 points), on either skin. A flat
    or barely kinked strip is the parent's sheet continued; a parallel sheet one
    thickness off (a joggle's far flange) starts millimetres out, and a shallow
    line pair crossing it touches it far from the hinge only.

    ``grand_u``: the child carries a parallel bend whose axis lies ``grand_u``
    along ``u`` from this one's (the child flat). A line at ``theta`` whose hinge
    sits ``d`` farther along ``u`` is offset by ``-d sin theta`` in ``rho``; a hinge
    nearer that bend than this one is that bend's line (this one flat, that one
    formed: a joggle or a hook half made), so on that side the window reaches at
    most ``max(hinge_rho_slack_flat_mm, grand_u / 2 |sin theta|)``.

    ``s_cut``: ``(s_limits, z_lo, z_hi)``, per angle the distance along the line
    where that child ends (the parallel bend's tangent lines placed at ``theta``,
    :func:`_short_child_limit`). Within the parallel bend's axial span
    ``z_lo..z_hi`` no point beyond it supports the line: past it lie that bend's
    arc and the sheet beyond, which a shallow line crosses at length.

    With one skin visible, its line is also the other skin's line one thickness
    off, and both readings score alike: the tie (within ``one_skin_tie_fraction``)
    goes to the reading nearer the window centre (the nominal kinematics), never
    to whichever lies first in ``rho``.

    A formed-like line (``|theta| >= one_skin_reread_min_deg``) that the window
    reads on one skin only may be a sheet whose *other* skin lies just outside
    the window: a hinge frame placed a few millimetres off (an ancestor's bend
    line moved along its blank) leaves the flange's skin nearer the nominal
    inside the window and the far one outside, and the window then takes the
    near skin for the inner one, one thickness off. The other reading of that
    visible skin (its partner on the other side) is read even outside the window
    (never past the grandchild limit): when that partner skin holds at least
    ``one_skin_reread_min_share`` of the visible skin's support, the two-skin
    pair stands. With the partner unseen nothing changes.
    """
    hb = cfg.hough_bin_mm
    slack = cfg.hinge_rho_slack_mm
    # candidate IML positions per angle: within the window of the fold kinematics' offset (rho_window)
    win = [rho_window(frame, float(th), slack, cfg) for th in thetas]
    centres = np.asarray([c for c, _ in win]) if win else np.asarray([frame.rho_iml])
    halves = np.asarray([h for _, h in win]) if win else np.asarray([slack])
    lows, highs = halves.copy(), halves.copy()
    # the one-skin re-reading may reach one thickness past the window, except on the grandchild's side
    re_lo = halves + abs(frame.delta)
    re_hi = halves + abs(frame.delta)
    if grand_u is not None and grand_u > 0:
        for k, th in enumerate(thetas):
            sth = math.sin(float(th))
            lim = max(float(cfg.hinge_rho_slack_flat_mm), 0.5 * float(grand_u) * abs(sth))
            if sth >= 0:
                lows[k] = min(lows[k], lim)
                re_lo[k] = lows[k]
            else:
                highs[k] = min(highs[k], lim)
                re_hi[k] = highs[k]
    # the histogram reaches one thickness past every pair's skins: the re-reading's pairs lie inside that too
    lo_r = float(np.min(np.minimum(centres, centres + frame.delta) - halves)) - abs(frame.delta) - 1.0
    hi_r = float(np.max(np.maximum(centres, centres + frame.delta) + halves)) + abs(frame.delta) + 1.0
    nb = int(math.ceil((hi_r - lo_r) / hb)) + 1
    w = max(1, int(round(cfg.hough_window_mm / hb)))
    kernel = np.ones(2 * w + 1)
    shift = int(round(frame.delta / hb))
    scores = np.zeros(len(thetas))
    rhos = np.zeros(len(thetas))
    n_i = np.zeros(len(thetas))
    n_o = np.zeros(len(thetas))
    # every line between the parent's and the child's tangent touches the bend's own arc
    # within sqrt(2 R window) of its tangent point: that support belongs to no angle
    r_out = max(abs(frame.rho_iml), abs(frame.rho_iml + frame.delta))
    s_min = max(0.3, math.sqrt(2.0 * r_out * cfg.hough_window_mm) + cfg.hough_arc_clearance_mm)
    for k, th in enumerate(thetas):
        c, s = math.cos(th), math.sin(th)
        sv = uc * c + vc * s
        sel = (sv > s_min) & (sv < s_max)
        if s_cut is not None:
            sel &= ~((zc > s_cut[1]) & (zc < s_cut[2]) & (sv >= s_cut[0][k]))
        if not sel.any():
            continue
        rho = -uc[sel] * s + vc[sel] * c
        idx = np.floor((rho - lo_r) / hb).astype(np.int64)
        ok = (idx >= 0) & (idx < nb)
        hist = np.bincount(idx[ok], minlength=nb).astype(np.float64)
        sm = np.convolve(hist, kernel, mode="same")
        ci_lo = int(math.floor((centres[k] - lows[k] - lo_r) / hb))
        ci_hi = int(math.ceil((centres[k] + highs[k] - lo_r) / hb))
        cand = np.arange(max(0, ci_lo), min(nb, ci_hi + 1))
        if len(cand) == 0:
            continue
        other = cand + shift
        valid = (other >= 0) & (other < nb)
        h_o = np.where(valid, sm[np.clip(other, 0, nb - 1)], 0.0)
        comb = sm[cand] + h_o
        if continuity and abs(th) <= math.radians(cfg.flat_continuity_max_deg):
            comb = np.where(_continuous_at_hinge(sv[sel][ok], idx[ok], cand, shift, nb, kernel, comb, s_min, s_max, cfg),
                            comb, 0.0)
        j = int(np.argmax(comb))
        if continuity and comb[j] <= 0.0:
            continue
        j = _one_skin_tie(j, cand, comb, sm, h_o, shift, centres[k], lo_r, hb, cfg)
        scores[k] = comb[j]
        rhos[k] = lo_r + (cand[j] + 0.5) * hb
        n_i[k] = sm[cand[j]]
        n_o[k] = h_o[j]
        if abs(th) >= math.radians(cfg.one_skin_reread_min_deg):
            alt = _one_skin_reread(int(cand[j]), float(n_i[k]), float(n_o[k]), hist, sm, shift, nb, centres[k],
                                   re_lo[k], re_hi[k], lo_r, hb, cfg)
            if alt is not None and alt[1] + alt[2] > scores[k]:
                rhos[k] = lo_r + (alt[0] + 0.5) * hb
                n_i[k], n_o[k] = alt[1], alt[2]
                scores[k] = alt[1] + alt[2]
    return scores, rhos, n_i, n_o


def _one_skin_reread(ci: int, ni: float, no: float, hist: np.ndarray, sm: np.ndarray, shift: int, nb: int,
                     centre: float, reach_lo: float, reach_hi: float, lo_r: float, hb: float, cfg: InspectConfig
                     ) -> tuple[int, float, float] | None:
    """``(inner-skin bin, n_iml, n_oml)`` of the other reading of a one-skin line when its partner skin is seen.

    The window's winner at inner bin ``ci`` reads one skin (``ni`` or ``no`` at most
    ``one_skin_fraction`` of the other). Its visible skin taken as the other skin puts
    the pair one thickness over (``shift`` bins). That pair is read within
    ``reach_lo`` / ``reach_hi`` of the window centre (the window plus one thickness,
    the grandchild's side excepted) and stands for the line when its partner skin holds
    at least ``one_skin_reread_min_share`` of the visible skin's support.
    """
    if shift == 0 or max(ni, no) <= 0.0 or min(ni, no) > cfg.one_skin_fraction * max(ni, no):
        return None
    vis = ci if ni >= no else ci + shift
    # the window's candidate may sit up to the smoothing half-width off the visible skin's own level (the
    # centroid of its raw counts there): the other reading is built on that level, so the offset is not
    # carried twice onto the partner skin
    w = max(1, int(round(cfg.hough_window_mm / hb)))
    lo, hi = max(0, vis - w), min(nb, vis + w + 1)
    mass = hist[lo:hi]
    if hi <= lo or float(mass.sum()) <= 0.0:
        return None
    vis = int(round(float((np.arange(lo, hi) * mass).sum() / mass.sum())))
    # visible skin read as the inner one: the other reading takes it for the outer skin, and the reverse
    new_i = vis - shift if ni >= no else vis
    new_o = new_i + shift
    if not (0 <= new_i < nb and 0 <= new_o < nb):
        return None
    off = lo_r + (new_i + 0.5) * hb - centre
    if off < -reach_lo or off > reach_hi:
        return None
    partner = float(sm[new_i] if vis == new_o else sm[new_o])
    if partner < max(3.0, cfg.one_skin_reread_min_share * float(sm[vis])):
        return None
    return int(new_i), float(sm[new_i]), float(sm[new_o])


def _one_skin_tie(j: int, cand: np.ndarray, comb: np.ndarray, sm: np.ndarray, h_o: np.ndarray, shift: int,
                  centre: float, lo_r: float, hb: float, cfg: InspectConfig) -> int:
    """The candidate index for one visible skin: the reading nearer ``centre`` when both readings tie.

    A line seen on one skin only is read two ways that score alike: as the inner
    skin (its outer partner empty) or as the outer skin (its inner partner empty),
    one thickness apart. When neither reading's weaker skin holds more than
    ``one_skin_fraction`` of the stronger one's support and the two scores agree
    within ``one_skin_tie_fraction``, the reading whose inner-skin offset lies
    nearer the window centre (the nominal kinematics) is taken. With the second
    skin seen, the reading that uses it scores higher and stands.
    """
    if shift == 0 or comb[j] <= 0.0:
        return j
    n_i, n_o = float(sm[cand[j]]), float(h_o[j])
    big = max(n_i, n_o)
    target = int(cand[j] - shift) if n_i >= n_o else int(cand[j] + shift)
    if target < int(cand[0]) or target > int(cand[-1]):
        return j
    ja = target - int(cand[0])
    p_a = min(float(sm[cand[ja]]), float(h_o[ja]))
    if max(min(n_i, n_o), p_a) > cfg.one_skin_fraction * big:
        return j
    if comb[ja] < (1.0 - cfg.one_skin_tie_fraction) * comb[j]:
        return j
    d_j = abs(lo_r + (cand[j] + 0.5) * hb - centre)
    d_a = abs(lo_r + (cand[ja] + 0.5) * hb - centre)
    return ja if d_a < d_j else j


def _continuous_at_hinge(sv: np.ndarray, idx: np.ndarray, cand: np.ndarray, shift: int, nb: int, kernel: np.ndarray,
                         comb: np.ndarray, s_min: float, s_max: float, cfg: InspectConfig) -> np.ndarray:
    """Per candidate rho bin: is the line's material continuous from the hinge (see :func:`_hough`)?"""
    step = cfg.flat_continuity_step_mm
    n_sub = max(1, int(round(cfg.flat_continuity_mm / step)))
    sub = np.floor((sv - s_min) / step).astype(np.int64)
    near = (sub >= 0) & (sub < n_sub)
    grid = np.zeros((nb, n_sub))
    np.add.at(grid, (idx[near], sub[near]), 1.0)
    for c in range(n_sub):
        grid[:, c] = np.convolve(grid[:, c], kernel, mode="same")
    other = cand + shift
    valid = (other >= 0) & (other < nb)
    both = grid[cand] + np.where(valid[:, None], grid[np.clip(other, 0, nb - 1)], 0.0)
    per_sub = comb * step / max(s_max - s_min, step)
    need = np.maximum(3.0, 0.2 * per_sub)
    return np.all(both >= need[:, None], axis=1)


def _displaced_formed(zc, uc, vc, frame: HingeFrame, sweep_deg: float, cfg: InspectConfig, reach: float,
                      grand_u: float | None = None, s_cut_fn=None) -> tuple[float, float, float, float, float] | None:
    """``(theta, rho_iml, score, n_iml, n_oml)`` of the best formed-like two-skin line whose inner skin
    lies beyond the narrow slack (up to ``hinge_rho_slack_wide_mm`` from nominal), else None.

    Angles from half the nominal sweep to the overbend limit; support counted
    over the fine reach, like the narrow fine search it is compared with. A
    line inside the narrow window was already weighed by the narrow search.
    ``grand_u`` limits the side toward a parallel bend the child carries, and
    ``s_cut_fn(thetas)`` gives the support cut, as in :func:`_hough`.
    """
    from dataclasses import replace

    wide = replace(cfg, hinge_rho_slack_mm=cfg.hinge_rho_slack_wide_mm)
    lo = max(0.5 * sweep_deg, cfg.flat_sweep_max_deg + 1.0)
    hi = sweep_deg + cfg.theta_overbend_deg
    coarse = np.radians(np.arange(lo, hi + 1e-9, cfg.theta_coarse_step_deg))
    if len(coarse) == 0:
        return None
    sc, _, _, _ = _hough(zc, uc, vc, frame, coarse, wide, reach, grand_u=grand_u,
                         s_cut=None if s_cut_fn is None else s_cut_fn(coarse))
    if float(sc.max()) <= 0.0:
        return None
    k = int(np.argmax(sc))
    fine = np.radians(np.arange(math.degrees(coarse[k]) - 1.5, math.degrees(coarse[k]) + 1.51, 0.1))
    sc2, rh2, ni2, no2 = _hough(zc, uc, vc, frame, fine, wide, reach, grand_u=grand_u,
                                s_cut=None if s_cut_fn is None else s_cut_fn(fine))
    k2 = int(np.argmax(sc2))
    centre, half = rho_window(frame, float(fine[k2]), cfg.hinge_rho_slack_mm, cfg)
    if abs(float(rh2[k2]) - centre) <= half:
        return None
    return float(fine[k2]), float(rh2[k2]), float(sc2[k2]), float(ni2[k2]), float(no2[k2])


def _wide_search_needed(narrow_deg: float, sweep_deg: float, cfg: InspectConfig) -> tuple[bool, bool]:
    """``(search, astray)``: does a bend whose narrow line lies at ``narrow_deg`` get the wide (displaced) search?

    Bends of at least ``displaced_min_sweep_deg`` do when the narrow line reads flat or
    partial (below ``formed_fraction_min`` of the sweep), and when it reads formed but more
    than ``displaced_check_dev_deg`` off the nominal sweep (``astray``).
    """
    if sweep_deg < cfg.displaced_min_sweep_deg:
        return False, False
    if narrow_deg < cfg.formed_fraction_min * sweep_deg:
        return True, False
    astray = abs(narrow_deg - sweep_deg) > cfg.displaced_check_dev_deg
    return astray, astray


def _short_child(part: PartView, bend: BendView, reach: float) -> int | None:
    """The nearest bend of the child parallel to this one within ``reach`` of its axis, else None.

    Such a child is short (a joggle's riser, a short step between two parallel bends).
    """
    a0 = unit(bend.axis_dir)
    best, best_d = None, math.inf
    for kb in part.panels[bend.child].children:
        gb = part.bends[kb]
        if abs(abs(float(unit(gb.axis_dir) @ a0)) - 1.0) > 1e-3:
            continue
        d = gb.axis_point - bend.axis_point
        dist = float(np.linalg.norm(d - float(d @ a0) * a0))
        if dist <= reach and dist < best_d:
            best, best_d = kb, dist
    return best


def _grand_offset(part: PartView, bend: BendView, grand: int, parent: PanelFit, frame: HingeFrame) -> float | None:
    """How far along ``u`` from this bend's axis the child's parallel bend ``grand`` has its axis, the child flat."""
    A = parent.A @ fold_transform(bend, 0.0)
    p = apply(A, np.asarray(part.bends[grand].axis_point, dtype=np.float64)[None])[0]
    u = float((p - frame.o) @ frame.u)
    return u if u > 0 else None


def _short_child_limit(part: PartView, bend: BendView, grand: int, parent: PanelFit, frame: HingeFrame, theta: float,
                       flat: bool, cfg: InspectConfig) -> tuple[float, float, float]:
    """``(s_limit, z_lo, z_hi)``: a short child grows, over the grandchild bend's axial span, only so far along its line.

    Up to the grandchild bend's tangent lines on the child, placed by the fold
    kinematics at ``theta``, plus ``short_child_margin_mm``: past them lie that
    bend's arc and the flange beyond it, whose voxels (normals smeared over a
    riser a few millimetres wide) would otherwise pull the riser's plane onto it.
    The limit applies within that bend's axial span (+- the margin, along the
    hinge frame's axis ``a``) only: beside a tab the child panel runs on to its
    own far edge.
    """
    A = parent.A @ fold_transform(bend, 0.0 if flat else theta)
    gb = part.bends[grand]
    d = frame.line_dir(theta)
    s_vals = [float((apply(A, np.asarray(gb.locus(name)[0], dtype=np.float64)[None])[0] - frame.o) @ d)
              for name in ("inner_tangent_parent", "outer_tangent_parent")]
    ends = apply(A, np.asarray(gb.locus("cylinder_axis"), dtype=np.float64))
    zs = (ends - frame.o) @ frame.a
    m = cfg.short_child_margin_mm
    return max(s_vals) + m, float(zs.min()) - m, float(zs.max()) + m


@dataclass
class JoggleStep:
    """The far flange of a joggle found beyond its riser, in the parent bend's hinge frame (:func:`_joggle_step`)."""

    #: offset of the far flange's mid-plane from the parent's along the frame's ``v`` (mm): measured, nominal
    step: float
    step_nominal: float
    #: the riser angle (rad) whose fold kinematics (both bends turned alike, the far flange kept parallel)
    #: put the far flange at ``step``
    theta: float
    #: the far flange's angle to the parent in the window (deg) and the points on its dominant skin
    slope_deg: float
    points: int
    two_skins: bool
    #: offset (``rho``) at ``theta`` of the riser's mid-plane through the S's middle (where each skin is half way
    #: up the step), None when neither skin's crossing was seen
    rho_mid: float | None = None


def _joggle_pair(part: PartView, bend: BendView, grand: int) -> bool:
    """The child's parallel bend ``grand`` turns the sheet back to the parent's orientation (a joggle, not a hook).

    Skin 0 continues across every bend, so the far flange's skin-0 normal at the
    final pose equals the parent's for a joggle and opposes it for a U-turn.
    """
    n_p = np.asarray(part.panels[bend.parent].n0, dtype=np.float64)
    n_g = np.asarray(part.panels[part.bends[grand].child].n0, dtype=np.float64)
    return float(n_p @ n_g) > math.cos(math.radians(1.0))


def _joggle_offset(part: PartView, bend: BendView, grand: int, parent: PanelFit, frame: HingeFrame,
                   theta: float) -> float:
    """Offset (frame ``v``, mm) of the far flange's mid-plane from the parent's, the riser folded to ``theta``.

    Both bends turn alike (``grand`` by ``theta + sweep_grand - sweep``), so the far
    flange keeps its nominal orientation to the parent and only its step changes.
    """
    gb = part.bends[grand]
    t = float(parent.t if parent.t is not None else part.t)
    a_c = parent.A @ fold_transform(bend, theta)
    a_g = a_c @ fold_transform(gb, theta + (gb.sweep - bend.sweep))
    pg, pp = part.panels[gb.child], part.panels[bend.parent]
    mid_g = apply(a_g, (np.asarray(pg.p0, dtype=np.float64) - 0.5 * t * np.asarray(pg.n0, dtype=np.float64))[None])[0]
    mid_p = apply(parent.A, (np.asarray(pp.p0, dtype=np.float64) - 0.5 * t * np.asarray(pp.n0, dtype=np.float64))[None])[0]
    return float((mid_g - mid_p) @ frame.v)


def _joggle_step(part: PartView, bend: BendView, grand: int, parent: PanelFit, frame: HingeFrame,
                 zc: np.ndarray, uc: np.ndarray, vc: np.ndarray, cfg: InspectConfig) -> JoggleStep | None:
    """The formed joggle's far flange beyond the riser and the riser angle its step implies, else None.

    In the parent bend's hinge frame the far flange's flat material starts past the
    far bend's child-side tangent lines (nominal pose): a window of
    ``joggle_window_mm`` along ``u`` from ``joggle_window_gap_mm`` beyond them holds
    it. Its skin levels (``skin_peaks``) must run parallel to the parent (within
    ``joggle_parallel_deg``) with at least ``joggle_min_points`` points on the
    dominant skin; the mid-plane is the two levels' middle, or, with one skin seen,
    that skin moved by ``t/2`` toward the nominal mid-plane. Its offset from the
    parent's mid-plane is the step; a step below half the nominal one (same sign) is
    not a formed joggle (None). The riser angle is the one at which the fold
    kinematics (:func:`_joggle_offset`) put the far flange at that step, the
    crossing nearest the nominal sweep.
    """
    gb = part.bends[grand]
    t = float(parent.t if parent.t is not None else part.t)
    if parent.n0 is None or parent.d0 is None:
        return None
    h_nom = _joggle_offset(part, bend, grand, parent, frame, bend.sweep)
    if abs(h_nom) < 0.3 * t:
        return None  # a step this small leaves the far flange inside the parent's own slab
    us = [float((apply(parent.A, np.asarray(gb.locus(name)[0], dtype=np.float64)[None])[0] - frame.o) @ frame.u)
          for name in ("inner_tangent_child", "outer_tangent_child")]
    u_lo = max(us) + cfg.joggle_window_gap_mm
    inside = (zc > frame.z0) & (zc < frame.z1)
    n_v = float(np.asarray(parent.n0, dtype=np.float64) @ frame.v)
    v_p = (float(parent.d0) - 0.5 * t - float(np.asarray(parent.n0, dtype=np.float64) @ frame.o)) * n_v
    # the far flange beyond the riser, and the parent's own sheet just before the hinge: both read locally, so a
    # bow or twist of the parent's region-wide plane does not enter the step
    far = _sheet_line(uc[inside], vc[inside], u_lo, u_lo + cfg.joggle_window_mm, t, v_p + h_nom, cfg)
    if far is None:
        return None
    near = _sheet_line(uc[inside], vc[inside], -cfg.joggle_window_gap_mm - cfg.joggle_window_mm,
                       -cfg.joggle_window_gap_mm, t, v_p, cfg)
    p_mid, p_slope = (v_p, 0.0) if near is None else (near[0], near[1])
    slope_deg = math.degrees(math.atan(far[1]) - math.atan(p_slope))
    if abs(slope_deg) > cfg.joggle_parallel_deg:
        return None
    u_m = 0.5 * u_lo  # the riser's middle: both mid-lines are compared there
    step = (far[0] + far[1] * u_m) - (p_mid + p_slope * u_m)
    if step * h_nom <= 0.0 or abs(step) < 0.5 * abs(h_nom):
        return None
    grid = np.radians(np.arange(0.0, math.degrees(bend.sweep) + cfg.theta_overbend_deg + 1e-9, 0.25))
    offs = np.asarray([_joggle_offset(part, bend, grand, parent, frame, float(th)) for th in grid]) - step
    cross = np.nonzero(np.sign(offs[:-1]) != np.sign(offs[1:]))[0]
    if len(cross) == 0:
        return None
    ths = [float(grid[k] + (grid[k + 1] - grid[k]) * offs[k] / (offs[k] - offs[k + 1])) for k in cross]
    theta = min(ths, key=lambda th: abs(th - bend.sweep))
    # the riser's middle: each skin crosses half the step there. A skin's half level lies beyond the other skin's
    # whole run whenever the step is below 2 t, so the points at it belong to that skin's S alone
    rho_mid = None
    if abs(step) < 2.0 * t:
        tr = (uc > -1.0) & (uc < u_lo)
        crossings = []
        for side in (0.5, -0.5):
            lv = p_mid + p_slope * uc + side * t + 0.5 * step
            at = inside & tr & (np.abs(vc - lv) < min(0.12, 0.1 * t))
            if int(at.sum()) >= cfg.joggle_min_points // 2:
                crossings.append(float(np.median(uc[at])))
        if crossings:
            u_i = float(np.mean(crossings))
            v_i = p_mid + p_slope * u_i + 0.5 * step
            rho_mid = -u_i * math.sin(theta) + v_i * math.cos(theta)
    return JoggleStep(step=float(step), step_nominal=float(h_nom), theta=theta, slope_deg=slope_deg,
                      points=int(far[2]), two_skins=bool(far[3]), rho_mid=rho_mid)


def _sheet_line(uc: np.ndarray, vc: np.ndarray, u_lo: float, u_hi: float, t: float, target: float,
                cfg: InspectConfig) -> tuple[float, float, int, bool] | None:
    """``(mid at u = 0, slope, points, two skins)`` of the sheet seen between ``u_lo`` and ``u_hi`` in a hinge frame.

    The skin levels are ``skin_peaks``; the line ``v = a + b u`` is fitted on the
    denser skin's points and moved to the mid-plane: by half the two levels'
    separation toward the other skin, or, with one skin seen, by ``t/2`` toward
    ``target`` (the expected mid-plane level). None with fewer than
    ``joggle_min_points`` points on that skin or less than 1 mm of it along ``u``.
    """
    sel = (uc > u_lo) & (uc < u_hi)
    if int(sel.sum()) < cfg.joggle_min_points:
        return None
    uu, vv = uc[sel], vc[sel]
    top, bottom = skin_peaks(vv, t)
    tol = min(0.25, 0.3 * t)
    lv, other = top, bottom
    near = np.abs(vv - top) < tol
    if bottom is not None:
        near_b = np.abs(vv - bottom) < tol
        if int(near_b.sum()) > int(near.sum()):
            near, lv, other = near_b, bottom, top
    if int(near.sum()) < cfg.joggle_min_points or float(np.ptp(uu[near])) < 1.0:
        return None
    b, a = (float(c) for c in np.polyfit(uu[near], vv[near], 1))
    u_c = 0.5 * (u_lo + u_hi)
    if other is not None:
        a += 0.5 * (other - lv)
    else:
        a += min((-0.5 * t, 0.5 * t), key=lambda d: abs(a + b * u_c + d - target))
    return a, b, int(near.sum()), other is not None


def _plane_angle_deg(a: PanelFit, b: PanelFit) -> float | None:
    """Angle (deg) between two fitted panels' skin-0 planes, None when either is not fitted."""
    if a.n0 is None or b.n0 is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.asarray(a.n0) @ np.asarray(b.n0)))))))


def _model_riser(part: PartView, bend: BendView, parent: PanelFit, frame: HingeFrame, theta: float, rho_i: float,
                 voxels: np.ndarray, skins: tuple[int, ...], u_deg: float, lever_mm: float = 0.0) -> PanelFit:
    """A joggle's riser placed on its line at ``theta`` (the step's angle): the plane through the line pair.

    The skin conventions are :func:`_grow_child`'s (inner line at ``rho_i``, outer
    one ``delta`` off); the pose follows the fold kinematics at ``theta``, moved
    onto the plane without a turn (a riser). ``voxels`` are its line's material.
    """
    t = float(parent.t if parent.t is not None else part.t)
    nrm = frame.line_normal(theta)
    sig = 1.0 if frame.delta > 0 else -1.0
    d_mid = float(nrm @ frame.o) + rho_i + 0.5 * frame.delta
    if bend.oml_skin == 0:
        n0, d0 = sig * nrm, sig * (d_mid + sig * 0.5 * t)
    else:
        n0, d0 = -sig * nrm, -sig * (d_mid - sig * 0.5 * t)
    A = parent.A @ fold_transform(bend, theta)
    A = pose_on_plane(A, part.panels[bend.child].p0, part.panels[bend.child].n0, n0, float(d0), frame.o, 0.0)
    fit = PanelFit(bend.child, "FITTED", A, n0=unit(n0), d0=float(d0), t=t, skins_seen=skins)
    fit.voxels = np.asarray(voxels, dtype=np.int64)
    fit.region = fit.voxels
    fit.model_plane = True
    fit.model_u_deg = float(u_deg)
    fit.model_lever_mm = float(lever_mm)
    return fit


def _band_points(pts: np.ndarray, frame: HingeFrame, margin: float, radius: float | None = None):
    z, uc, vc = frame.coords(pts)
    r = np.hypot(uc, vc)
    sel = (z > frame.z0 - margin) & (z < frame.z1 + margin) & (r < (frame.band if radius is None else radius))
    idx = np.nonzero(sel)[0]
    return idx, z[idx], uc[idx], vc[idx]


def _line_support(zc: np.ndarray, uc: np.ndarray, vc: np.ndarray, frame: HingeFrame, theta: float, rho_i: float,
                  cfg: InspectConfig) -> tuple[np.ndarray, int, bool, float]:
    """``(line mask, points, one skin, axial coverage)`` of the two-skin line at ``theta`` / ``rho_i`` in a band.

    The line's points lie on either skin past ``s = 0.3`` mm; the axial coverage is
    the share of 2 mm bins of the bend's span ``z0 .. z1`` that hold one.
    """
    c, s = math.cos(theta), math.sin(theta)
    sv = uc * c + vc * s
    rho = -uc * s + vc * c
    on_i = (sv > 0.3) & (np.abs(rho - rho_i) < cfg.hough_window_mm)
    on_o = (sv > 0.3) & (np.abs(rho - (rho_i + frame.delta)) < cfg.hough_window_mm)
    line = on_i | on_o
    one_skin = bool(min(on_i.sum(), on_o.sum()) < max(5, 0.1 * max(on_i.sum(), on_o.sum())))
    zl = zc[line]
    if len(zl):
        bins = np.floor((zl - frame.z0) / 2.0).astype(np.int64)
        nb = max(1, int(math.ceil((frame.z1 - frame.z0) / 2.0)))
        cov = len(np.unique(bins[(bins >= 0) & (bins < nb)])) / nb
    else:
        cov = 0.0
    return line, int(line.sum()), one_skin, float(cov)


def _sheet_skins(uc: np.ndarray, vc: np.ndarray, theta: float, rho_v: float, t: float, s_max: float,
                 cfg: InspectConfig) -> tuple[float, float]:
    """``(visible, partner)``: support of the line's visible skin at offset ``rho_v`` (angle ``theta``) and of the
    strongest parallel level 0.4 t to 1.6 t from it, on either side.

    A sheet shows two skins about one thickness apart; a thin lip's skins often
    read closer than its nominal thickness (edge-blurred points), so the partner
    is looked for over a band of separations rather than at exactly ``t``. Counts
    are smoothed over ``hough_window_mm`` like the line search's, past 1 mm along
    the line and within ``s_max``.
    """
    c, s = math.cos(theta), math.sin(theta)
    sv = uc * c + vc * s
    rho = (-uc * s + vc * c)[(sv > 1.0) & (sv < s_max)]
    if len(rho) == 0:
        return 0.0, 0.0
    hb = cfg.hough_bin_mm
    lo = rho_v - 1.7 * t - 2 * cfg.hough_window_mm
    nb = int(math.ceil((3.4 * t + 4 * cfg.hough_window_mm) / hb)) + 1
    idx = np.floor((rho - lo) / hb).astype(np.int64)
    ok = (idx >= 0) & (idx < nb)
    hist = np.bincount(idx[ok], minlength=nb).astype(np.float64)
    w = max(1, int(round(cfg.hough_window_mm / hb)))
    sm = np.convolve(hist, np.ones(2 * w + 1), mode="same")
    centres = lo + (np.arange(nb) + 0.5) * hb
    d = np.abs(centres - rho_v)
    vis = float(sm[d <= cfg.hough_window_mm].max(initial=0.0))
    band = (d >= 0.4 * t) & (d <= 1.6 * t)
    return vis, float(sm[band].max(initial=0.0))


def _reversed_reading(full_reg, vox: VoxelCloud, vox_reg, vox_nrm, owner, part: PartView, bend: BendView,
                      parent: PanelFit, frame: HingeFrame, sweep_deg: float, cad_score: float, reach: float,
                      cfg: InspectConfig):
    """The child of ``bend`` found formed the wrong way, about the mirrored hinge, else None.

    In :func:`mirrored_hinge_frame` the two-skin line search runs over the angles a
    wrong-way fold reads as formed (past the flat limits and the CAD search's reach,
    ``-theta_min_deg``) up to ``180 - reversed_wrap_margin_deg`` (a reading nearer 180
    keeps its positive sign: :func:`~domains.production.engine.inspect.geom.signed_sweep_deg`),
    on material no fitted panel owns. The reading stands only when its support is
    at least ``reversed_support_ratio`` times ``cad_score`` (the CAD-direction
    reading's; 0 when that one finds no child), it shows a sheet (a parallel level
    0.4 to 1.6 thicknesses from its visible skin holding at least
    ``reversed_two_skin_share`` of that skin's support: :func:`_sheet_skins`), its
    line meets the child-line minimums and the child grows from it. Returns
    ``(child, theta (rad, mirrored frame), rho_iml, score, line points, one skin,
    axial coverage, detail)``.
    """
    fm, mb = mirrored_hinge_frame(part, bend, parent, frame)
    lo = max(-cfg.theta_min_deg, min(cfg.flat_sweep_max_deg, cfg.flat_fraction_max * sweep_deg)) + 1.0
    hi = min(sweep_deg + cfg.theta_overbend_deg, 180.0 - cfg.reversed_wrap_margin_deg - 1.0)
    if hi <= lo:
        return None
    widx, wz, wu, wv = _band_points(full_reg, fm, 2.0, radius=reach)
    free = owner[vox.inverse[widx]] < 0
    widx, wz, wu, wv = widx[free], wz[free], wu[free], wv[free]
    inband = np.hypot(wu, wv) < fm.band
    if int(inband.sum()) < cfg.min_child_points:
        return None
    zc, uc, vc = wz[inband], wu[inband], wv[inband]
    coarse = np.radians(np.arange(lo, hi + 1e-9, cfg.theta_coarse_step_deg))
    sc, _, _, _ = _hough(zc, uc, vc, fm, coarse, cfg, fm.band)
    if float(sc.max()) <= 0.0:
        return None
    k = int(np.argmax(sc))
    c_deg = math.degrees(coarse[k])
    fine = np.radians(np.arange(max(lo, c_deg - 1.5), min(hi, c_deg + 1.5) + 0.01, 0.1))
    sc2, rh2, ni2, no2 = _hough(wz, wu, wv, fm, fine, cfg, reach)
    k2 = int(np.argmax(sc2))
    theta, rho_i, score = float(fine[k2]), float(rh2[k2]), float(sc2[k2])
    n_i, n_o = float(ni2[k2]), float(no2[k2])
    if score < cfg.reversed_support_ratio * max(cad_score, 1.0):
        return None
    t = float(parent.t if parent.t is not None else part.t)
    vis, partner = _sheet_skins(wu, wv, theta, rho_i if n_i >= n_o else rho_i + fm.delta, t, reach, cfg)
    if partner < max(5.0, cfg.reversed_two_skin_share * vis):
        return None
    line, n_line, one_skin, cov = _line_support(wz[inband], wu[inband], wv[inband], fm, theta, rho_i, cfg)
    if n_line < cfg.min_child_points or cov < cfg.min_child_axial_coverage:
        return None
    seeds_full = widx[inband][line]
    child = _grow_child(full_reg, vox, vox_reg, vox_nrm, owner, part, mb, parent, fm, theta, rho_i, seeds_full,
                        False, cfg)
    if child.state == "UNKNOWN" or child.n0 is None:
        return None
    detail = {"reversed": 1.0, "reversed_theta_deg": math.degrees(theta), "reversed_score": score,
              "n_iml": n_i, "n_oml": n_o}
    return child, theta, rho_i, score, n_line, one_skin, cov, detail


def fit_articulated(full_reg: np.ndarray, vox: VoxelCloud, vox_reg: np.ndarray, vox_nrm: np.ndarray,
                    part: PartView, datum_n0: np.ndarray, datum_d0: float, datum_t: float,
                    datum_skins: tuple[int, ...], cfg: InspectConfig, budget_left,
                    reversed_search: bool = False) -> ArticulatedFit:
    """Walk the flange tree from the registered datum (module docstring).

    ``reversed_search`` adds the wrong-way (mirrored hinge) search (module docstring,
    step 5). The engine runs it on the chosen pose only: on every candidate pose it
    would let a datum turned over read each of its bends as formed the wrong way and
    explain the scan as well as the true pose.

    ``full_reg`` / ``vox_reg`` / ``vox_nrm`` are the full-resolution points,
    voxel points and voxel normals already in the registered frame; ``vox``
    carries the neighbour graph and the point -> voxel map.
    """
    n_panels = len(part.panels)
    panels: list[PanelFit] = [PanelFit(i, "UNKNOWN", None) for i in range(n_panels)]
    bends: list[BendFit] = [BendFit(j, False, False) for j in range(part.n)]
    owner = np.full(len(vox_reg), -1, dtype=np.int64)
    # datum: slab points inside the (slightly grown) outline
    dat = PanelFit(part.datum, "DATUM", np.eye(4), n0=datum_n0, d0=datum_d0, t=datum_t, skins_seen=datum_skins)
    dv = _slab_voxels_in_outline(vox_reg, part, part.datum, np.eye(4), datum_n0, datum_d0, datum_t, grow=0.6)
    dat.voxels = dv
    owner[dv] = part.datum
    # the datum's grown region (cut errors may put its real edges beyond the nominal outline)
    h = vox_reg @ datum_n0 - datum_d0
    slab = (h < 0.6) & (h > -datum_t - 0.6) & (np.abs(vox_nrm @ datum_n0) > math.cos(math.radians(cfg.grow_normal_tol_deg)))
    dat.region = components_of(vox.graph, slab, dv) if len(dv) else dv
    panels[part.datum] = dat
    for j in part.bfs:
        bend = part.bends[j]
        parent = panels[bend.parent]
        bf = bends[j]
        if budget_left() <= 0:
            bf.reason = "TIMEOUT"
            continue
        if parent.state == "UNKNOWN" or parent.A is None:
            bf.reason = "BEND_STATE_UNKNOWN"
            panels[bend.child] = PanelFit(bend.child, "UNKNOWN", None, reason="BEND_STATE_UNKNOWN")
            continue
        frame = hinge_frame(bend, parent, cfg)
        bf.frame = frame
        reach = frame.band * cfg.hough_fine_reach
        widx, wz, wu, wv = _band_points(full_reg, frame, 2.0, radius=reach)
        inband = np.hypot(wu, wv) < frame.band
        bidx, zc, uc, vc = widx[inband], wz[inband], wu[inband], wv[inband]
        if len(bidx) < cfg.min_child_points:
            bf.reason = "OCCLUDED"
            panels[bend.child] = PanelFit(bend.child, "UNKNOWN", None, reason="OCCLUDED")
            continue
        sweep_deg = math.degrees(bend.sweep)
        coarse = np.radians(np.arange(cfg.theta_min_deg, sweep_deg + cfg.theta_overbend_deg + 1e-9,
                                      cfg.theta_coarse_step_deg))
        # a child too short to stand on its own line (a joggle's riser) is followed by a parallel sheet
        # about one thickness off the parent: near flat, only material continuous from the hinge counts
        grand = _short_child(part, bend, reach)
        short = grand is not None
        grand_u = _grand_offset(part, bend, grand, parent, frame) if short else None

        def _cut(ths, _g=grand, _b=bend, _p=parent, _f=frame):
            """Per angle, where the short child ends along its line (None without one)."""
            if _g is None:
                return None
            lims = [_short_child_limit(part, _b, _g, _p, _f, float(t), False, cfg) for t in ths]
            return np.asarray([x[0] for x in lims]), lims[0][1], lims[0][2]
        sc, rh, ni, no = _hough(zc, uc, vc, frame, coarse, cfg, frame.band, continuity=short, grand_u=grand_u,
                                s_cut=_cut(coarse))
        k = int(np.argmax(sc))
        fine = np.radians(np.arange(math.degrees(coarse[k]) - 1.5, math.degrees(coarse[k]) + 1.51, 0.1))
        sc2, rh2, ni2, no2 = _hough(wz, wu, wv, frame, fine, cfg, reach, continuity=short, grand_u=grand_u,
                                    s_cut=_cut(fine))
        k2 = int(np.argmax(sc2))
        theta, rho_i = float(fine[k2]), float(rh2[k2])
        score = float(sc2[k2])
        detail = {"n_iml": float(ni2[k2]), "n_oml": float(no2[k2]), "coarse_profile_max": float(sc.max())}
        if short:
            detail["short_child"] = 1.0
        # a formed flange whose bend line sits beyond the slack (a backgauge error, a flange cut long)
        # leaves the narrow search on clutter at a flat or partial angle: a formed-like line found with
        # the wide slack over the full reach replaces it when its support is clearly stronger. So it does
        # when the narrow line reads formed but strays more than ``displaced_check_dev_deg`` from the
        # nominal sweep: a pair of lines crossing a displaced sheet's two skins obliquely is such clutter
        wide, astray = _wide_search_needed(math.degrees(theta), sweep_deg, cfg)
        if wide:
            alt = _displaced_formed(wz, wu, wv, frame, sweep_deg, cfg, reach, grand_u=grand_u, s_cut_fn=_cut)
            if alt is not None and alt[2] >= cfg.displaced_support_ratio * max(score, 1.0):
                detail.update({"narrow_theta_deg": math.degrees(theta), "narrow_score": score,
                               "displaced_mm": float(alt[1] - frame.rho_iml_at(alt[0])), "n_iml": alt[3],
                               "n_oml": alt[4]})
                if astray:
                    detail["narrow_astray"] = 1.0
                theta, rho_i, score = alt[0], alt[1], alt[2]
        # a joggle's riser: the far flange beyond it (found directly, parallel to the parent) gives the step,
        # and the step the riser's angle. On a riser without flat material (a curved S) the convex skin offers
        # a tangent line at any angle and a shallow pair across the far flange's start can outscore the riser:
        # when the line strays from the step's angle, the step decides (``_joggle_step``)
        jog = None
        jog_forced = False
        if short and sweep_deg > cfg.flat_sweep_max_deg and _joggle_pair(part, bend, grand):
            jog = _joggle_step(part, bend, grand, parent, frame, wz, wu, wv, cfg)
            if jog is not None:
                detail.update({"joggle_step_mm": jog.step, "joggle_step_nominal_mm": jog.step_nominal,
                               "joggle_theta_deg": math.degrees(jog.theta), "joggle_slope_deg": jog.slope_deg})
                if abs(theta - jog.theta) > math.radians(cfg.joggle_theta_tol_deg):
                    th1 = np.array([jog.theta])
                    sc1, rh1, ni1, no1 = _hough(wz, wu, wv, frame, th1, cfg, reach, grand_u=grand_u, s_cut=_cut(th1))
                    detail.update({"line_theta_deg": math.degrees(theta), "line_score": score,
                                   "n_iml": float(ni1[0]), "n_oml": float(no1[0])})
                    theta = jog.theta
                    rho_i = float(rh1[0]) if sc1[0] > 0 else frame.rho_iml_at(jog.theta)
                    score = float(sc1[0])
                    jog_forced = True
        bf.theta_deg = math.degrees(theta)
        bf.rho_iml = rho_i
        bf.score = score
        bf.detail = detail
        # the line points: both skins at theta
        line, n_line, one_skin, cov = _line_support(zc, uc, vc, frame, theta, rho_i, cfg)
        bf.child_line_points = n_line
        bf.one_skin = one_skin
        bf.axial_coverage = float(cov)
        cad_ok = jog is not None or (n_line >= cfg.min_child_points and cov >= cfg.min_child_axial_coverage)
        # a fold formed the wrong way: its child lies on the other side of the parent's plane, about the mirrored
        # hinge, outside the CAD-direction search. Its reading replaces this one only when clearly stronger
        if reversed_search and jog is None and not short and not parent.model_plane and parent.n0 is not None:
            rev = _reversed_reading(full_reg, vox, vox_reg, vox_nrm, owner, part, bend, parent, frame, sweep_deg,
                                    score if cad_ok else 0.0, reach, cfg)
            if rev is not None:
                child, th_r, rho_r, sc_r, n_r, one_r, cov_r, det_r = rev
                detail.update({"cad_theta_deg": bf.theta_deg, "cad_score": score, "cad_line_points": float(n_line),
                               **det_r})
                bf.theta_deg = -math.degrees(th_r)
                bf.rho_iml = rho_r
                bf.score = sc_r
                bf.child_line_points = n_r
                bf.one_skin = one_r
                bf.axial_coverage = float(cov_r)
                bf.found, bf.flat, bf.reversed = True, False, True
                panels[bend.child] = child
                if child.voxels.size:
                    free = owner[child.voxels] < 0
                    owner[child.voxels[free]] = bend.child
                continue
        if jog is None and (bf.child_line_points < cfg.min_child_points or cov < cfg.min_child_axial_coverage):
            bf.reason = "OCCLUDED" if bf.child_line_points < cfg.min_child_points else "LOW_DENSITY"
            panels[bend.child] = PanelFit(bend.child, "UNKNOWN", None, reason=bf.reason)
            continue
        flat = abs(bf.theta_deg) <= min(cfg.flat_sweep_max_deg, cfg.flat_fraction_max * sweep_deg)
        bf.found = True
        bf.flat = flat
        seeds_full = bidx[line]
        s_limit = None if grand is None else _short_child_limit(part, bend, grand, parent, frame, theta, flat, cfg)
        child = _grow_child(full_reg, vox, vox_reg, vox_nrm, owner, part, bend, parent, frame, theta, rho_i,
                            seeds_full, flat, cfg, s_limit=s_limit)
        if jog is not None and not flat:
            # the riser grew no plane at the step's angle (a curved S: its voxel normals leave every cone, or the
            # region settled on the S's steepest part, away from its own line): it is placed on its line at the
            # step's angle instead
            ang = _plane_angle_deg(parent, child) if child.state != "UNKNOWN" else None
            if jog_forced or ang is None or abs(ang - math.degrees(jog.theta)) > cfg.joggle_theta_tol_deg \
                    or abs(ang - bf.theta_deg) > cfg.joggle_theta_tol_deg:
                if not jog_forced:
                    th1 = np.array([jog.theta])
                    sc1, rh1, ni1, no1 = _hough(wz, wu, wv, frame, th1, cfg, reach, grand_u=grand_u, s_cut=_cut(th1))
                    detail.update({"line_theta_deg": bf.theta_deg, "line_score": bf.score,
                                   "n_iml": float(ni1[0]), "n_oml": float(no1[0])})
                    rho_i = float(rh1[0]) if sc1[0] > 0 else frame.rho_iml_at(jog.theta)
                    theta = jog.theta
                    bf.theta_deg, bf.score = math.degrees(theta), float(sc1[0])
                if jog.rho_mid is not None:
                    # the riser's plane passes through the S's middle (a curved S touches a line at its angle on one
                    # skin only, near one of its ends): both bends' sharps then sit where the flat flanges meet it
                    detail["joggle_rho_line"] = rho_i
                    rho_i = float(jog.rho_mid) - 0.5 * frame.delta
                bf.rho_iml = rho_i
                if ang is not None:
                    detail["joggle_grown_angle_deg"] = ang
                s_lim = _short_child_limit(part, bend, grand, parent, frame, theta, False, cfg)
                c, s = math.cos(theta), math.sin(theta)
                sv_all = uc * c + vc * s
                rho_all = -uc * s + vc * c
                on = (sv_all > 0.3) & (sv_all < s_lim[0]) & ((np.abs(rho_all - rho_i) < cfg.hough_window_mm)
                                                              | (np.abs(rho_all - rho_i - frame.delta) < cfg.hough_window_mm))
                rv = np.unique(vox.inverse[bidx[on]])
                rv = rv[owner[rv] < 0]
                n_i, n_o = detail.get("n_iml", 0.0), detail.get("n_oml", 0.0)
                skins = (0, 1) if min(n_i, n_o) >= 0.1 * max(n_i, n_o, 1.0) else (
                    (bend.oml_skin,) if n_o >= n_i else (1 - bend.oml_skin,))
                u_deg = math.degrees(math.tan(min(abs(theta), math.radians(80.0))) * cfg.joggle_riser_length_u_fraction)
                child = _model_riser(part, bend, parent, frame, theta, rho_i, rv, skins, u_deg,
                                     lever_mm=0.5 * max(float(s_lim[0]), 0.0))
                detail["joggle_model"] = 1.0
                detail["joggle_model_u_deg"] = u_deg
        panels[bend.child] = child
        if child.voxels.size:
            free = owner[child.voxels] < 0
            owner[child.voxels[free]] = bend.child
    return ArticulatedFit(panels=panels, bends=bends, owner=owner)


def _slab_voxels_in_outline(vox_reg: np.ndarray, part: PartView, panel_index: int, A: np.ndarray, n0: np.ndarray,
                            d0: float, t: float, grow: float) -> np.ndarray:
    """Voxels inside the panel's slab (skin 0 .. skin 1, +- tolerance) whose projection is inside its outline.

    The cores of the panel's holes are not the panel (:func:`in_footprint`): a
    tab cut out of a window lies flat inside it until it is formed.
    """
    h = vox_reg @ n0 - d0
    slab = (h < 0.6) & (h > -t - 0.6)
    idx = np.nonzero(slab)[0]
    if len(idx) == 0:
        return idx
    return idx[in_footprint(vox_reg[idx], part, panel_index, A, n0, grow)]


def in_footprint(pts: np.ndarray, part: PartView, panel_index: int, A: np.ndarray, n0: np.ndarray,
                 grow: float) -> np.ndarray:
    """Points whose projection on the panel's plane (pose ``A``) lies inside its outline grown by ``grow``
    and outside the core of every hole (the hole shrunk by ``grow``).

    Material inside a hole's core is never the panel's own: a tab cut out of a
    window lies flat in it until its bend is formed, and it belongs to that tab
    (the child), not to the panel around it.
    """
    from domains.production.engine.inspect.geom import orthonormal_frame

    panel = part.panels[panel_index].model
    e1, e2 = orthonormal_frame(n0)
    outer = apply(A, np.asarray(panel.outer_boundary_mm, dtype=np.float64))
    q = np.stack([pts @ e1, pts @ e2], axis=1)
    keep = inside_grown(np.stack([outer @ e1, outer @ e2], axis=1), q, grow)
    for hole in panel.holes_mm:
        if len(hole) < 3 or not keep.any():
            continue
        hw = apply(A, np.asarray(hole, dtype=np.float64))
        keep &= ~_in_core(np.stack([hw @ e1, hw @ e2], axis=1), q, grow)
    return keep


def _in_core(poly: np.ndarray, q: np.ndarray, margin: float) -> np.ndarray:
    """Points inside the 2D polygon by more than ``margin`` (distance to its boundary)."""
    inside = MplPath(poly).contains_points(q)
    if margin <= 0 or not inside.any():
        return inside
    cand = np.nonzero(inside)[0]
    best = np.full(len(cand), np.inf)
    closed = np.vstack([poly, poly[:1]])
    for a, b in zip(closed[:-1], closed[1:]):
        d = b - a
        ll = float(d @ d)
        if ll < 1e-18:
            continue
        sp = np.clip((q[cand] - a) @ d / ll, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(q[cand] - (a + sp[:, None] * d), axis=1))
    inside[cand[best <= margin]] = False
    return inside


def inside_grown(poly: np.ndarray, q: np.ndarray, grow: float) -> np.ndarray:
    """Points inside the 2D polygon or within ``grow`` of its boundary."""
    inside = MplPath(poly).contains_points(q)
    if grow <= 0 or inside.all():
        return inside
    rest = np.nonzero(~inside)[0]
    best = np.full(len(rest), np.inf)
    closed = np.vstack([poly, poly[:1]])
    for a, b in zip(closed[:-1], closed[1:]):
        d = b - a
        ll = float(d @ d)
        if ll < 1e-18:
            continue
        sp = np.clip((q[rest] - a) @ d / ll, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(q[rest] - (a + sp[:, None] * d), axis=1))
    inside[rest[best <= grow]] = True
    return inside


def _grow_child(full_reg, vox: VoxelCloud, vox_reg, vox_nrm, owner, part: PartView, bend: BendView,
                parent: PanelFit, frame: HingeFrame, theta: float, rho_i: float, seeds_full: np.ndarray, flat: bool,
                cfg: InspectConfig, s_limit: tuple[float, float, float] | None = None) -> PanelFit:
    """Region-grow the child's material slab from the Hough line seeds and fit its skins.

    ``s_limit``: ``(s, z_lo, z_hi)``, a short child grows only up to ``s`` along
    its line within the axial span ``z_lo .. z_hi`` (:func:`_short_child_limit`).

    Skins: in the cross-section the inner (IML) line is at ``rho_i`` and the
    outer (OML) line at ``rho_i + delta``, so along ``+n_line`` the OML lies on
    the side of ``sign(delta)``. The bend's OML is contract skin
    ``bend.oml_skin`` of the child (skin 0 continues across every bend).
    """
    t = float(parent.t if parent.t is not None else part.t)
    n_line = frame.line_normal(theta)  # +rho direction
    sig = 1.0 if frame.delta > 0 else -1.0
    rho_mid = rho_i + 0.5 * frame.delta
    d_mid = float(n_line @ frame.o) + rho_mid
    seeds = np.unique(vox.inverse[seeds_full])
    region = np.zeros(0, dtype=np.int64)
    cos_n = math.cos(math.radians(cfg.grow_normal_tol_deg))
    nrm = n_line.copy()
    seen = (0, 1)
    last = -1
    for _ in range(cfg.grow_iterations):
        dist = np.abs(vox_reg @ nrm - d_mid)
        okn = np.abs(vox_nrm @ nrm) > cos_n
        mask = (dist < 0.5 * t + cfg.grow_plane_tol_mm) & okn & (owner != parent.index)
        if flat:
            # coplanar with the parent: only the child side of the hinge
            mask &= ((vox_reg - frame.o) @ frame.line_dir(theta)) > 0.0
        if s_limit is not None:
            rel = vox_reg - frame.o
            za = rel @ frame.a
            mask &= ~((rel @ frame.line_dir(theta) >= s_limit[0]) & (za >= s_limit[1]) & (za <= s_limit[2]))
        region = components_of(vox.graph, mask, seeds)
        if len(region) < 3:
            break
        if len(region) == last:
            break
        last = len(region)
        sel = np.isin(vox.inverse, region)
        pts = full_reg[sel]
        if len(pts) < 20:
            break
        dd = pts @ nrm - d_mid
        inslab = np.abs(dd) < 0.5 * t + cfg.grow_plane_tol_mm
        pts, dd = pts[inslab], dd[inslab]
        if len(dd) < 30:
            break
        # the skins are the slab's height levels, not its two halves: cut walls, flare and fused setups
        # fill the slab between them, and a half's median pulled by that clutter walks the mid-plane
        # onto one skin, which then drops the other out of the slab
        top, bottom = skin_peaks(dd, t)
        tol = min(cfg.grow_plane_tol_mm, 0.4 * t)
        groups = [pts[np.abs(dd - lv) < tol] for lv in ((top, bottom) if bottom is not None else (top,))]
        groups = [g for g in groups if len(g) >= 30]
        if not groups:
            break
        planes = []
        for grp in groups:
            pl = robust_plane(grp, iterations=3)
            planes.append((pl if float(pl.n @ nrm) > 0 else pl.flipped(), len(grp)))
        nrm = unit(sum(pl.n * cnt for pl, cnt in planes))
        if len(groups) == 2:
            d_mid = 0.5 * (float(np.median(groups[0] @ nrm)) + float(np.median(groups[1] @ nrm)))
            seen = (0, 1)
        else:
            d_mid, seen = _one_skin_mid(float(np.median(groups[0] @ nrm)), nrm, frame, rho_i, t, sig, bend)
    if len(region) == 0:
        return PanelFit(bend.child, "UNKNOWN", None, reason="AMBIGUOUS_SEGMENTATION")
    # the skins are the converged slab's height levels: cut-wall points, flare and fused setups
    # fill the slab between them. A single visible skin is named by the hinge line pair.
    d_mid, seen = _refine_skin_levels(full_reg[np.isin(vox.inverse, region)], nrm, d_mid, t, sig, bend, seen, cfg,
                                      ref=(frame, rho_i))
    d_oml = d_mid + sig * 0.5 * t
    d_iml = d_mid - sig * 0.5 * t
    if bend.oml_skin == 0:
        n0, d0 = sig * nrm, sig * d_oml
    else:
        n0, d0 = -sig * nrm, -sig * d_iml
    # hinge kinematics for grandchildren, moved onto the fitted plane (a riser's plane, a few millimetres up to
    # the next parallel bend, does not turn it)
    A = parent.A @ fold_transform(bend, 0.0 if flat else theta)
    riser = s_limit is not None and s_limit[0] < cfg.child_orient_min_depth_mm
    A = pose_on_plane(A, part.panels[bend.child].p0, part.panels[bend.child].n0, n0, float(d0), frame.o,
                      0.0 if riser else cfg.child_orient_max_deg)
    fit = PanelFit(bend.child, "FLAT" if flat else "FITTED", A, n0=n0, d0=float(d0), t=t, skins_seen=seen)
    fit.voxels = _restrict_to_footprint(vox_reg, region, part, bend.child, A, n0, grow=3.0)
    fit.region = region
    return fit


def pose_on_plane(A: np.ndarray, p0: np.ndarray, n0_cad: np.ndarray, n0: np.ndarray, d0: float, hinge: np.ndarray,
                  max_turn_deg: float) -> np.ndarray:
    """The kinematic pose ``A`` of a child (final pose -> registered frame) moved onto its fitted skin-0 plane.

    ``n0 . x = d0`` is the fitted plane, ``p0`` / ``n0_cad`` the panel's skin-0
    reference point and normal at its final pose. The fold kinematics take the
    hinge's Hough angle (none at all for a flat child), so a kink of a
    not-yet-formed bend, springback or the angle's quantisation, carried down
    the tree, would put a grandchild's hinge a millimetre off its sheet across
    a wide panel: when the kinematic normal lies within ``max_turn_deg`` of the
    fitted one, ``A`` is first turned about ``hinge`` (a point on the bend's
    axis) onto it. Then ``A`` is moved along the normal so that ``p0`` lies on
    the plane. A kinematic normal more than ~25 deg off the fitted one is left
    as it is (the fit and the kinematics disagree; nothing is moved).
    """
    n_pred = unit(apply_dir(A, np.asarray(n0_cad, dtype=np.float64)[None])[0])
    if float(n_pred @ n0) <= 0.9:
        return A
    if max_turn_deg > 0.0 and float(n_pred @ n0) > math.cos(math.radians(max_turn_deg)):
        R = rotation_between(n_pred, n0)
        hinge = np.asarray(hinge, dtype=np.float64)
        A = rigid(R, hinge - R @ hinge) @ A
    ref_pt = apply(A, np.asarray(p0, dtype=np.float64)[None])[0]
    return rigid(np.eye(3), n0 * (float(d0) - float(ref_pt @ n0))) @ A


def fold_transform(bend: BendView | MirroredBend, theta: float) -> np.ndarray:
    """Child at its final pose -> child folded to ``theta`` (rad), in the parent's final-pose frame.

    The arc consumes ``theta * R_n`` of the bend's neutral strip (``BA`` long), centred
    on the bend line: the tangent line moves toward the child by half the unconsumed
    strip and the other half stays flat on the child (the E2 fold model). At
    ``theta = 0`` this is E1's Method A (rotate back by the sweep, translate by the
    neutral strip length); at the nominal sweep it is the identity.

    A :class:`MirroredBend` is folded from the same flat blank the other way: about
    its mirrored axis, toward ``-v`` (at its nominal sweep, the CAD fold's mirror
    image across the parent's mid-plane, with the child's skins kept).
    """
    mirrored = isinstance(bend, MirroredBend)
    b = bend.base if mirrored else bend
    ba = float(b.model.neutral_length_mm)
    used = ba * theta / b.sweep if b.sweep > 1e-9 else 0.0
    m = -b.p_out  # in the parent's plane, from the parent toward the child
    to_flat = rigid(np.eye(3), m * ba) @ rotation_about_line(b.axis_point, b.axis_dir, -b.sweep)
    axis_t = (bend.axis_point if mirrored else b.axis_point) + m * (0.5 * (ba - used))
    fold = rotation_about_line(axis_t, b.axis_dir, -theta if mirrored else theta) @ rigid(np.eye(3), -m * used)
    return fold @ to_flat


def _refine_skin_levels(pts: np.ndarray, nrm: np.ndarray, d_mid: float, t: float, sig: float, bend: BendView,
                        seen: tuple[int, ...], cfg: InspectConfig, ref: tuple | None = None
                        ) -> tuple[float, tuple[int, ...]]:
    """Mid-plane offset and seen skins of a child slab from its height levels (``skin_peaks``).

    ``ref`` ``(frame, rho_iml)``: the hinge line pair the slab was grown from;
    a single visible skin is the one of that pair it lies on at the hinge
    (:func:`_one_skin_mid`). Without it, the side of the current mid-plane decides.
    """
    dd = pts @ nrm - d_mid
    dd = dd[np.abs(dd) < 0.5 * t + cfg.grow_plane_tol_mm]
    if len(dd) < 30:
        return d_mid, seen
    top, bottom = skin_peaks(dd, t)
    tol = min(cfg.grow_plane_tol_mm, 0.4 * t)
    if bottom is not None:
        lv_top = float(np.median(dd[np.abs(dd - top) < tol]))
        lv_bot = float(np.median(dd[np.abs(dd - bottom) < tol]))
        return d_mid + 0.5 * (lv_top + lv_bot), (0, 1)
    near = dd[np.abs(dd - top) < tol]
    if len(near) < 30:
        return d_mid, seen
    lv = float(np.median(near))
    if ref is not None:
        return _one_skin_mid(d_mid + lv, nrm, ref[0], ref[1], t, sig, bend)
    if lv > 0:  # the upper skin (along +nrm) is the one seen
        return d_mid + lv - 0.5 * t, ((bend.oml_skin,) if sig > 0 else (1 - bend.oml_skin,))
    return d_mid + lv + 0.5 * t, ((1 - bend.oml_skin,) if sig > 0 else (bend.oml_skin,))


def _one_skin_mid(level: float, nrm: np.ndarray, frame: HingeFrame, rho_i: float, t: float, sig: float,
                  bend: BendView) -> tuple[float, tuple[int, ...]]:
    """``(mid-plane offset, seen skins)`` of a child slab seen on one skin at offset ``level`` along ``nrm``.

    The skin is named by the hinge line pair the slab grew from: at the hinge
    point the inner line lies at ``rho_i`` and the outer at ``rho_i + delta``
    (``nrm`` keeps the line normal's orientation), and the visible skin's plane
    passes nearer one of them. A flange's plane is evaluated where its lines
    were found, so a tilt of the region-wide fit does not move the comparison.
    """
    off = level - float(nrm @ frame.o)
    is_oml = abs(off - (rho_i + frame.delta)) < abs(off - rho_i)
    if is_oml:
        return level - sig * 0.5 * t, (bend.oml_skin,)
    return level + sig * 0.5 * t, (1 - bend.oml_skin,)


def _restrict_to_footprint(vox_reg: np.ndarray, region: np.ndarray, part: PartView, panel_index: int, A: np.ndarray,
                           n0: np.ndarray, grow: float) -> np.ndarray:
    if len(region) == 0:
        return region
    return region[in_footprint(vox_reg[region], part, panel_index, A, n0, grow)]
