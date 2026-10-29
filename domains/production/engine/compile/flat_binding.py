"""Flat binding, K from the flat and developed-length QA (plan §3.1 "Flat binding").

**CAD development.** The compiled flange tree is unfolded to stage S_0 (Method
A) for a trial K: every panel at its S_0 pose plus the flat strip of every
bend, projected into the datum's skin-0 plane. Its cut boundary (panel FREE
segments and the strips' axial ends) is sampled every 0.5 mm, and each bend's
predicted flat line is its strip centre line. Because the development is
generated *from the panel tree*, assigning flat lines to development lines is
a graph-consistent match (each flat line lands between the two panels its bend
joins).

**Fit.** The flat outline is registered onto the development by a 2D rigid
transform, optionally **mirrored** (the drawing may show the other skin): eight
starts from the principal axes (4 rotations x mirror), each refined by trimmed
point-to-point ICP. A blank whose outline is symmetric (every rectangle) fits
several ways equally well: the fits within ``max(0.05 mm, best RMS)`` of the
best are *outline-equivalent*, and the drawing's own evidence chooses among
them, never the order in which tied RMS values happen to sort (that order
changes with a mere translation of the drawing):

1. with a released map (compile check): agreement with the released mirror
   flag, then the most released lines within tolerance, then their mean offset;
2. otherwise, with drawn bend lines: the most lines assigned to bends, then the
   smallest line residual (ties within 0.05 mm), then the most UP/DOWN and
   angle annotations that agree with the compiled direction and sweep (UP means
   toward the viewer, i.e. toward skin 0 on an unmirrored fit; a mirrored fit
   views skin 1, so UP and DOWN swap);
3. then unmirrored before mirrored, then the smaller rotation angle.

When candidates with *different* maps survive every step the binding is
ambiguous (``fit_choice.ambiguous``; confidence halved, operator review). The
choice is made at every trial K over the whole outline-equivalent family.

**K.** K* in [0.05, 0.75] (0.02 grid, then golden-section) minimises the outline
residual plus the line term: the RMS over *every* drawn line of its offset
(assigned) or of ``flat_line_offset_tol_mm`` (unassigned), so a fit cannot score
well by leaving lines unassigned. K* is only *used* inside the plausible band
``shop K +- 3 x shop_k_uncertainty`` (default [0.18, 0.48]): a uniform length
error in a flat is otherwise indistinguishable from a different K and would be
absorbed silently. Outside the band the bound K is the band edge (its
uncertainty covers the clamp) and the QA is evaluated there.

**Binding.** Flat bend lines (``L<n>`` from the DXF parser; ``I<n>`` inferred
from the development for a flat STEP) are assigned to compiled bends by the
Hungarian algorithm over (perpendicular offset <= ``flat_line_offset_tol_mm``,
angle <= ``flat_line_angle_tol_deg``) pairs. The proposal carries a confidence
in [0, 1] and ``needs_operator_review`` (any doubt: incomplete map, ambiguous
fit, inferred lines, clamped K, failed QA); the plan still requires operator
confirmation in the wizard before a binding counts as OPERATOR_CONFIRMED.

**Compile use.** For a recipe whose binding is AUTO_PROPOSED or
OPERATOR_CONFIRMED, every map entry is checked against the fit
(FLAT_BINDING_INCONSISTENT on any disagreement or unknown line id).
**Developed-length QA**: the largest of the bound lines' offsets and the
outline extents' differences across every bend direction, flat vs development
at the bound K (the nearest plausible K); above
``developed_length_threshold_mm`` (0.5 mm) the compile is BLOCKED with
FLAT_DEVELOPED_LENGTH_MISMATCH. It is therefore the smallest disagreement any
plausible K leaves, not the residual at a K fitted to absorb it. A binding that
passes gives K (``BOUND_FLAT``) only when it is OPERATOR_CONFIRMED, or
AUTO_PROPOSED in a DEV_AUTO_CONFIRMED recipe (the DEV stand-in for operator
confirmation); an unconfirmed proposal leaves the shop K (FLAT_NOT_BOUND). An
UNBOUND flat is parsed and a proposal is reported in diagnostics, but K is the
shop default.

**Method B (fold the bound flat, :func:`fold_flat`).** A shop flat carries the
shop's own bend allowance per bend, which a single K only approximates (up to
the QA threshold per strip). Once a binding is bound, every recipe bend's strip
length is measured from the flat itself: the S_0 development is linear in the
strip lengths (a longer strip of bend *a* moves everything beyond *a* along
*a*'s strip direction, and *a*'s own drawn line by half as much), so the strips
and a small correction of the flat's 2D pose are solved by Gauss-Newton from
the outline (normal residuals of every outline sample against the segment of
the panel that carries it) and the bound drawn lines (strip centre lines). The
stage models then use these strips (``METHOD_B_FOLD_FLAT``; the published
``neutral_length_mm`` of each bend is its strip from the flat, and
``k_factor`` stays the bound K). The fold is used only when every strip is
observable (drawn lines are needed: a flat STEP's lines are inferred from the
development itself), it converges, it explains the flat to drafting precision
(afterwards every bound line and every extent across a bend direction within
0.03 mm, and 99 % of the outline within 0.05 mm: a panel or line drawn off the
CAD is never absorbed into the strips). The strips follow the drawn lines
(weight 100 against one outline sample); the outline fixes the pose and
whatever the lines leave open. A strip whose equivalent K lies outside the
plausible band is the flat's defect, not the shop's process: its stage strip
is the nearest plausible one (the band edge), and the clamp is reported.
Otherwise
the stages stay ``METHOD_A_ROTATE_BACK`` with the bound K, and the reason is in
``diagnostics.flat.method_b`` and the developed-length check's detail. When the
fold explains the flat its strips are known, so the developed-length QA compares
the flat with the nearest plausible per-bend development (every strip clamped
into the band) instead of a single K: a wide but plausible bend table is no
mismatch, and one implausible strip is not hidden by being spread over every
strip (it moves its own drawn line by only half of its error). Otherwise the
single-K check stands. Under Method B, ``k_factor.uncertainty`` is what the fold leaves unexplained
(floored at 0.005), not the uniform fit's residual. The policy is pinned by
:data:`FOLD_POLICY_VERSION`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.production.part_recipe import PartRecipeV1
from domains.production.engine.compile.config import DEFAULT_SHOP_K, CompileConfig
from domains.production.engine.compile.flange_graph import LABEL_FEATURE, LABEL_FREE
from domains.production.engine.compile.flat import FlatBendLine, FlatParseError, ParsedFlat, parse_flat
from domains.production.engine.compile.geometry import apply4, orthonormal_frame, unit
from domains.production.engine.compile.part_model import PartModel
from domains.production.engine.compile.stages import stage_bend_line, stage_panel_transforms

__all__ = [
    "Development",
    "FlatFit",
    "FlatBindingProposal",
    "FlatFold",
    "FlatOutcome",
    "development_2d",
    "fit_flat",
    "fold_flat",
    "fold_flat_or_reason",
    "propose_binding",
    "compile_flat",
    "parse_recipe_flat",
    "binding_counts_for_k",
    "k_plausible_band",
    "K_SEARCH_RANGE",
    "K_PLAUSIBLE_SIGMAS",
    "FOLD_POLICY_VERSION",
]

K_SEARCH_RANGE = (0.05, 0.75)
#: Half-width of the plausible K band, in standard uncertainties of the shop K.
K_PLAUSIBLE_SIGMAS = 3.0
_TRIM = 0.95
#: Fits whose outline RMS is within max(_EQUIVALENT_MM, best RMS) of the best are outline-equivalent.
_EQUIVALENT_MM = 0.05
#: Line residuals within this of the best are tied (annotations decide next).
_LINE_TIE_MM = 0.05
#: Annotated angle within this of a bend's sweep or included angle agrees with it.
_ANNOTATION_ANGLE_TOL_DEG = 1.0
#: K scan step before the golden-section refinement.
_K_GRID_STEP = 0.02


def k_plausible_band(shop_k: float, shop_k_uncertainty: float) -> tuple[float, float]:
    """``shop K +- K_PLAUSIBLE_SIGMAS * uncertainty``, inside the search range."""
    lo = max(K_SEARCH_RANGE[0], shop_k - K_PLAUSIBLE_SIGMAS * shop_k_uncertainty)
    hi = min(K_SEARCH_RANGE[1], shop_k + K_PLAUSIBLE_SIGMAS * shop_k_uncertainty)
    if lo > hi:
        lo = hi = min(max(shop_k, K_SEARCH_RANGE[0]), K_SEARCH_RANGE[1])
    return lo, hi


@dataclass
class Development:
    k_factor: float
    boundary: np.ndarray  # (M,2) cut-boundary samples in the datum skin-0 frame
    lines: dict[int, tuple[np.ndarray, np.ndarray]]  # graph key -> predicted flat line (2D)
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    #: (M,2) start / end of the boundary segment each sample lies on
    seg_a: np.ndarray | None = None
    seg_b: np.ndarray | None = None
    #: (M,) panel key that carries each boundary sample (a strip end: its bend's parent panel)
    owner: np.ndarray | None = None


@dataclass
class FlatFit:
    rms: float
    rotation: np.ndarray  # 2x2 (det -1 when mirrored): flat -> development
    translation: np.ndarray
    mirrored: bool

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.rotation.T + self.translation

    def matrix3(self) -> list[list[float]]:
        m = np.eye(3)
        m[:2, :2] = self.rotation
        m[:2, 2] = self.translation
        return [[round(float(v), 6) for v in row] for row in m]


@dataclass
class FlatBindingProposal:
    bend_line_map: dict[str, str]
    mirrored: bool
    confidence: float
    needs_operator_review: bool
    #: K the flat supports inside the plausible band (the K a binding would use)
    k_factor: float
    k_uncertainty: float
    fit: FlatFit
    runner_up_rms: float | None
    line_offsets_mm: dict[str, float]
    lines: list[FlatBendLine]
    developed_max_abs_delta_mm: float | None
    developed_detail: str
    method: str = "GRAPH_MATCH_RIGID_FIT"
    #: unconstrained best-fit K of the flat (K*), and the plausible band it was clipped to
    k_flat_fit: float | None = None
    k_band: tuple[float, float] | None = None
    #: how the fit was chosen among the outline-equivalent ones
    fit_choice: dict[str, Any] = field(default_factory=dict)
    #: the bound lines' offsets at the K the QA is evaluated at (``line_offsets_mm`` is at K*)
    qa_line_offsets_mm: dict[str, float] = field(default_factory=dict)

    @property
    def k_clamped(self) -> bool:
        return self.k_flat_fit is not None and abs(self.k_flat_fit - self.k_factor) > 1e-9

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "bend_line_map": dict(sorted(self.bend_line_map.items())),
            "mirrored": self.mirrored,
            "confidence": round(self.confidence, 4),
            "needs_operator_review": self.needs_operator_review,
            "k_factor": round(self.k_factor, 6),
            "k_uncertainty": round(self.k_uncertainty, 6),
            "k_flat_fit": None if self.k_flat_fit is None else round(self.k_flat_fit, 6),
            "k_plausible_band": None if self.k_band is None else [round(v, 6) for v in self.k_band],
            "k_clamped": self.k_clamped,
            "outline_rms_mm": round(self.fit.rms, 6),
            "runner_up_rms_mm": None if self.runner_up_rms is None else round(self.runner_up_rms, 6),
            "flat_to_development": self.fit.matrix3(),
            "fit_choice": self.fit_choice,
            "line_offsets_mm": {k: (round(v, 6) if math.isfinite(v) else None)
                                for k, v in sorted(self.line_offsets_mm.items())},
            "qa_line_offsets_mm": {k: (round(v, 6) if math.isfinite(v) else None)
                                   for k, v in sorted(self.qa_line_offsets_mm.items())},
            "lines": [ln.to_json() for ln in self.lines],
            "developed_max_abs_delta_mm": (None if self.developed_max_abs_delta_mm is None
                                           else round(self.developed_max_abs_delta_mm, 6)),
            "developed_detail": self.developed_detail,
        }


@dataclass
class FlatOutcome:
    record: dict[str, Any]
    bound: bool
    k_factor: float | None
    k_uncertainty: float
    neutral_lengths: dict[int, float] | None
    issues: list[tuple[str, str]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    proposal: FlatBindingProposal | None = None
    #: why a parsed, consistent flat still gives no K (FLAT_NOT_BOUND detail), when it does not
    unbound_reason: str | None = None
    #: stage method the bound flat supports: METHOD_B_FOLD_FLAT when :func:`fold_flat` measured every strip
    method: str = "METHOD_A_ROTATE_BACK"
    fold: FlatFold | None = None


def _sample_segment(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / spacing)))
    s = np.arange(n + 1)[:, None] / n
    return a + s * (b - a)


def development_2d(model: PartModel, order: list[int], k_factor: float, spacing: float = 0.5,
                   neutral: dict[int, float] | None = None) -> Development:
    """CAD development at stage S_0 for ``k_factor`` (module docstring).

    ``neutral`` overrides the strip length of the given graph bends (Method B,
    :func:`fold_flat`); every other strip is ``sweep (Ri + k_factor t)``.
    """
    rigid = set(model.bends) - set(order)
    strips = {k: b.neutral_length(k_factor) for k, b in model.bends.items()}
    if neutral:
        strips.update(neutral)
    neutral = strips
    transforms = stage_panel_transforms(model, rigid, neutral)
    datum = model.panels[model.datum]
    side0 = datum.side(0) or datum.side(1)
    normal = side0.normal if datum.side(0) is not None else -side0.normal
    origin = side0.point
    x_axis, y_axis = orthonormal_frame(normal, np.array([1.0, 0.0, 0.0]))

    def to2(p3: np.ndarray) -> np.ndarray:
        rel = p3 - origin
        return np.stack([rel @ x_axis, rel @ y_axis], axis=-1)

    pts, seg_a, seg_b, owner = [], [], [], []

    def add(a2: np.ndarray, b2: np.ndarray, panel_key: int) -> None:
        q = _sample_segment(a2, b2, spacing)
        pts.append(q)
        seg_a.append(np.broadcast_to(a2, q.shape))
        seg_b.append(np.broadcast_to(b2, q.shape))
        owner.append(np.full(len(q), panel_key, dtype=np.int64))

    for key, panel in model.panels.items():
        side = panel.side(0)
        if side is None:
            continue
        for loop in side.loops:
            p3 = apply4(transforms[key], loop.points)
            p2 = to2(p3)
            nxt = np.roll(p2, -1, axis=0)
            for i in range(len(p2)):
                if loop.labels[i] in (LABEL_FREE, LABEL_FEATURE):
                    add(p2[i], nxt[i], key)
    lines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for gk in order:
        b = model.bends[gk]
        t_parent = transforms[b.parent]
        loci = b.loci()
        tangent = loci["outer_tangent_parent"] if b.oml_skin == 0 else loci["inner_tangent_parent"]
        a3, e3 = tangent
        shift = -b.p_out * neutral[gk]
        for end in (a3, e3):
            # a strip end runs along the strip (parent -> child): it moves with the parent panel
            add(to2(apply4(t_parent, end)), to2(apply4(t_parent, end + shift)), b.parent)
        la, lb = stage_bend_line(b, False, t_parent, neutral[gk])
        lines[gk] = (to2(la), to2(lb))
    if not pts:
        empty = np.zeros((0, 2))
        return Development(k_factor, empty, lines, origin, x_axis, y_axis, empty, empty, np.zeros(0, dtype=np.int64))
    return Development(k_factor, np.concatenate(pts), lines, origin, x_axis, y_axis, np.concatenate(seg_a),
                       np.concatenate(seg_b), np.concatenate(owner))


def _pca_frame(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = p.mean(axis=0)
    cov = np.cov((p - c).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    if major[int(np.argmax(np.abs(major)))] < 0:
        major = -major
    return c, major


def _rot2(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _trimmed_rms(d: np.ndarray) -> float:
    if not len(d):
        return float("inf")
    k = max(1, int(_TRIM * len(d)))
    part = np.sort(d)[:k]
    return float(math.sqrt(float(np.mean(part * part))))


def _closest_on_segments(q: np.ndarray, tree, seg_a: np.ndarray, seg_b: np.ndarray, k: int = 2
                         ) -> tuple[np.ndarray, np.ndarray]:
    """``(distance, closest point)`` of each point on the segments of its ``k`` nearest samples."""
    dist, proj, _ = _closest_on_segments_idx(q, tree, seg_a, seg_b, k)
    return dist, proj


def _closest_on_segments_idx(q: np.ndarray, tree, seg_a: np.ndarray, seg_b: np.ndarray, k: int = 2
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As :func:`_closest_on_segments`, plus the index of the sample whose segment is closest."""
    k = max(1, min(k, len(seg_a)))
    _, idx = tree.query(q, k=k)
    idx = np.asarray(idx).reshape(len(q), k)
    a, b = seg_a[idx], seg_b[idx]
    ab = b - a
    denom = np.einsum("nki,nki->nk", ab, ab)
    t = np.einsum("nki,nki->nk", q[:, None, :] - a, ab) / np.where(denom > 1e-18, denom, 1.0)
    proj = a + np.clip(t, 0.0, 1.0)[..., None] * ab
    dist = np.linalg.norm(q[:, None, :] - proj, axis=2)
    best = np.argmin(dist, axis=1)
    rows = np.arange(len(q))
    return dist[rows, best], proj[rows, best], idx[rows, best]


def _segment_distances(q: np.ndarray, tree, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Distance of each point to the nearest of the segments of its two nearest samples."""
    return _closest_on_segments(q, tree, seg_a, seg_b)[0]


def _icp2(src: np.ndarray, tgt_tree, tgt: np.ndarray, rot: np.ndarray, trans: np.ndarray, iterations: int,
          tgt_segs: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Trimmed ICP of ``src`` onto the target; with ``tgt_segs`` each point pairs with its closest
    point on the target *polyline* (no sampling-phase pull along the segments)."""
    mirrored = np.linalg.det(rot) < 0
    flip = np.diag([1.0, -1.0]) if mirrored else np.eye(2)
    base = src @ flip.T
    r = rot @ flip  # proper part
    for _ in range(iterations):
        q = base @ r.T + trans
        if tgt_segs is not None:
            # two nearest samples' segments: one is not enough near corners (ICP then stalls)
            d, target = _closest_on_segments(q, tgt_tree, tgt_segs[0], tgt_segs[1], k=2)
        else:
            d, idx = tgt_tree.query(q)
            target = tgt[idx]
        keep = d <= np.quantile(d, _TRIM) if len(d) > 10 else np.ones(len(d), dtype=bool)
        a, b = base[keep], target[keep]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        h = (a - ca).T @ (b - cb)
        u, _, vt = np.linalg.svd(h)
        rr = vt.T @ u.T
        if np.linalg.det(rr) < 0:
            vt[1] *= -1
            rr = vt.T @ u.T
        r, trans = rr, cb - rr @ ca
    q = base @ r.T + trans
    if tgt_segs is not None:
        d = _segment_distances(q, tgt_tree, tgt_segs[0], tgt_segs[1])
    else:
        d, _ = tgt_tree.query(q)
    return r @ flip, trans, _trimmed_rms(d)


def fit_flat(flat_pts: np.ndarray, dev_pts: np.ndarray, iterations: int = 25, seeds: list[FlatFit] | None = None,
             flat_segs: tuple[np.ndarray, np.ndarray] | None = None,
             dev_segs: tuple[np.ndarray, np.ndarray] | None = None) -> list[FlatFit]:
    """Distinct 2D rigid/mirror fits of the flat onto the development, best first.

    With the segments given (the compile always gives them), ICP pairs each
    flat sample with its closest point on the development *polyline* and the
    score is the symmetric trimmed RMS of point-to-segment distances, so
    neither carries a sampling-phase floor: between two 0.5 mm samplings,
    point-to-point pairs pull along the edges by up to a quarter spacing, the
    pull changes with K (segment lengths move the sample phase), and it biased
    the pose and K* on parts with few bends (K* 0.357 for a 0.33 single-bend L).
    """
    from scipy.spatial import cKDTree

    tree_d = cKDTree(dev_pts)
    tree_f = cKDTree(flat_pts)
    cf, mf = _pca_frame(flat_pts)
    cd, md = _pca_frame(dev_pts)
    ang_f, ang_d = math.atan2(mf[1], mf[0]), math.atan2(md[1], md[0])
    starts: list[np.ndarray] = []
    if seeds:
        starts = [s.rotation for s in seeds]
    else:
        for mirror in (False, True):
            flip = np.diag([1.0, -1.0]) if mirror else np.eye(2)
            af = ang_f if not mirror else -ang_f
            for k in range(4):
                starts.append(_rot2(ang_d - af + k * math.pi / 2) @ flip)
    fits: list[FlatFit] = []
    for idx, rot0 in enumerate(starts):
        trans0 = seeds[idx].translation if seeds else cd - rot0 @ cf
        rot, trans, _ = _icp2(flat_pts, tree_d, dev_pts, rot0, trans0, iterations, dev_segs)
        q = flat_pts @ rot.T + trans
        back = (dev_pts - trans) @ rot  # R^T (p - t): development samples in the flat's frame
        if flat_segs is not None and dev_segs is not None:
            d1 = _segment_distances(q, tree_d, dev_segs[0], dev_segs[1])
            d2 = _segment_distances(back, tree_f, flat_segs[0], flat_segs[1])
        else:
            d1, _ = tree_d.query(q)
            d2, _ = tree_f.query(back)
        rms = math.sqrt(0.5 * (_trimmed_rms(d1) ** 2 + _trimmed_rms(d2) ** 2))
        fit = FlatFit(rms, rot, trans, bool(np.linalg.det(rot) < 0))
        if any(np.allclose(f.rotation, fit.rotation, atol=1e-3) and np.linalg.norm(f.translation - fit.translation) < 0.5
               for f in fits):
            continue
        fits.append(fit)
    fits.sort(key=lambda f: f.rms)
    return fits


def _assign_lines(lines: list[FlatBendLine], fit: FlatFit, dev: Development, order: list[int],
                  uid_of: dict[int, str], cfg: CompileConfig) -> tuple[dict[str, str], dict[str, float]]:
    from scipy.optimize import linear_sum_assignment

    if not lines or not order:
        return {}, {}
    big = 1e9
    cost = np.full((len(lines), len(order)), big)
    offs = np.full((len(lines), len(order)), np.inf)
    for i, ln in enumerate(lines):
        a, b = fit.apply(np.stack([ln.start, ln.end]))
        d = unit(np.append(b - a, 0.0))[:2]
        mid = 0.5 * (a + b)
        for j, gk in enumerate(order):
            pa, pb = dev.lines[gk]
            pd = unit(np.append(pb - pa, 0.0))[:2]
            ang = math.degrees(math.acos(min(1.0, abs(float(d @ pd)))))
            normal = np.array([-pd[1], pd[0]])
            off = abs(float((mid - pa) @ normal))
            if ang <= cfg.flat_line_angle_tol_deg and off <= cfg.flat_line_offset_tol_mm:
                cost[i, j] = off + 0.1 * ang
                offs[i, j] = off
    rows, cols = linear_sum_assignment(cost)
    mapping, offsets = {}, {}
    for i, j in zip(rows, cols):
        if cost[i, j] < big:
            mapping[lines[i].line_id] = uid_of[order[j]]
            offsets[lines[i].line_id] = float(offs[i, j])
    return mapping, offsets


def _line_offsets_for(mapping: dict[str, str], lines: list[FlatBendLine], fit: FlatFit, dev: Development,
                      gk_of: dict[str, int]) -> dict[str, float]:
    by_id = {ln.line_id: ln for ln in lines}
    out = {}
    for lid, uid in mapping.items():
        ln = by_id.get(lid)
        if ln is None or uid not in gk_of:
            out[lid] = float("inf")
            continue
        a, b = fit.apply(np.stack([ln.start, ln.end]))
        pa, pb = dev.lines[gk_of[uid]]
        pd = unit(np.append(pb - pa, 0.0))[:2]
        d = unit(np.append(b - a, 0.0))[:2]
        if abs(float(d @ pd)) < math.cos(math.radians(5.0)):
            out[lid] = float("inf")
            continue
        normal = np.array([-pd[1], pd[0]])
        out[lid] = abs(float((0.5 * (a + b) - pa) @ normal))
    return out


def _extent_deltas(flat_pts: np.ndarray, fit: FlatFit, dev: Development, order: list[int]) -> float:
    q = fit.apply(flat_pts)
    worst = 0.0
    seen: list[np.ndarray] = []
    for gk in order:
        pa, pb = dev.lines[gk]
        d = unit(np.append(pb - pa, 0.0))[:2]
        n = np.array([-d[1], d[0]])
        if any(abs(float(n @ s)) > math.cos(math.radians(2.0)) for s in seen):
            continue
        seen.append(n)
        ef = float(np.ptp(q @ n))
        ed = float(np.ptp(dev.boundary @ n))
        worst = max(worst, abs(ef - ed))
    return worst


def _infer_lines(dev: Development, fit: FlatFit, order: list[int]) -> list[FlatBendLine]:
    inv_r = fit.rotation.T
    out = []
    for gk in order:
        pa, pb = dev.lines[gk]
        a = (pa - fit.translation) @ inv_r.T
        b = (pb - fit.translation) @ inv_r.T
        out.append(FlatBendLine("", a, b, "INFERRED"))
    out.sort(key=lambda ln: (round(float(ln.midpoint[0]), 3), round(float(ln.midpoint[1]), 3)))
    for i, ln in enumerate(out, start=1):
        ln.line_id = f"I{i}"
    return out


@dataclass
class _Candidate:
    """One outline-equivalent fit with the line evidence it implies."""

    fit: FlatFit
    lines: list[FlatBendLine]
    mapping: dict[str, str]
    offsets: dict[str, float]
    residual: float
    annotation: int
    released: tuple[int, int, float] | None


def _rotation_angle_deg(fit: FlatFit) -> float:
    """Angle (deg, [0, 360)) of the proper rotation part of a fit (translation-free tie-break)."""
    r = fit.rotation @ np.diag([1.0, -1.0]) if fit.mirrored else fit.rotation
    return round(math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0]))), 6) % 360.0


def _annotation_agreement(lines: list[FlatBendLine], mapping: dict[str, str], mirrored: bool, model: PartModel,
                          gk_of: dict[str, int]) -> int:
    """(# agreeing - # disagreeing) UP/DOWN and angle annotations of the mapped lines.

    UP is toward the viewer: toward skin 0 on an unmirrored fit (the compiled
    UP), toward skin 1 on a mirrored one. An angle agrees with either the sweep
    or the included angle (drawing conventions differ), within 1 deg.
    """
    by_id = {ln.line_id: ln for ln in lines}
    score = 0
    for lid, uid in mapping.items():
        ln, gk = by_id.get(lid), gk_of.get(uid)
        if ln is None or gk is None:
            continue
        bend = model.bends[gk]
        if ln.direction in ("UP", "DOWN") and bend.direction in ("UP", "DOWN"):
            expected = bend.direction if not mirrored else ("DOWN" if bend.direction == "UP" else "UP")
            score += 1 if ln.direction == expected else -1
        if ln.angle_deg is not None:
            gap = min(abs(ln.angle_deg - bend.sweep_deg), abs(ln.angle_deg - (180.0 - bend.sweep_deg)))
            score += 1 if gap <= _ANNOTATION_ANGLE_TOL_DEG else -1
    return score


def _equivalent(fits: list[FlatFit]) -> list[FlatFit]:
    """The outline-equivalent fits (module docstring), best first."""
    if not fits:
        return []
    best = fits[0].rms
    return [f for f in fits if f.rms <= best + max(_EQUIVALENT_MM, best)][:8]


def _choose_fit(fits: list[FlatFit], dev: Development, parsed: ParsedFlat, model: PartModel, order: list[int],
                uid_of: dict[int, str], gk_of: dict[str, int], cfg: CompileConfig,
                recipe_map: dict[str, str] | None, recipe_mirrored: bool | None) -> tuple[_Candidate, dict[str, Any]]:
    """Pick among the outline-equivalent fits by the drawing's evidence (module docstring)."""
    drawn = parsed.bend_lines
    tol = cfg.flat_line_offset_tol_mm
    cands: list[_Candidate] = []
    for f in _equivalent(fits):
        if drawn:
            lines = drawn
            mapping, offsets = _assign_lines(drawn, f, dev, order, uid_of, cfg)
        else:
            lines = _infer_lines(dev, f, order)
            mapping = {ln.line_id: uid_of[gk] for ln, gk in zip(lines, _match_inferred(lines, f, dev, order))}
            offsets = {lid: 0.0 for lid in mapping}
        residual = math.sqrt(float(np.mean(np.square(list(offsets.values()))))) if offsets else math.inf
        released = None
        if recipe_map:
            offs = _line_offsets_for(recipe_map, lines, f, dev, gk_of)
            within = [v for v in offs.values() if math.isfinite(v) and v <= tol]
            mirror_miss = int(recipe_mirrored is not None and f.mirrored != bool(recipe_mirrored))
            released = (mirror_miss, len(offs) - len(within), float(np.mean(within)) if within else 1e9)
        cands.append(_Candidate(f, lines, mapping, offsets, residual,
                                _annotation_agreement(drawn, mapping, f.mirrored, model, gk_of) if drawn else 0,
                                released))
    pool, decided, narrowed = cands, "outline", []

    def narrow(keep: list[_Candidate], label: str) -> None:
        nonlocal pool, decided
        if keep and len(keep) < len(pool):
            pool, decided = keep, label
            narrowed.append(label)

    if recipe_map:
        best = min(c.released for c in pool)
        narrow([c for c in pool if c.released[:2] == best[:2]
                and c.released[2] <= best[2] + _LINE_TIE_MM], "released_map")
    if drawn:
        n_best = max(len(c.mapping) for c in pool)
        narrow([c for c in pool if len(c.mapping) == n_best], "line_count")
        r_best = min(c.residual for c in pool)
        narrow([c for c in pool if c.residual <= r_best + _LINE_TIE_MM], "line_residual")
        a_best = max(c.annotation for c in pool)
        narrow([c for c in pool if c.annotation == a_best], "annotations")
    pool = sorted(pool, key=lambda c: (c.fit.mirrored, c.fit.rms, _rotation_angle_deg(c.fit)))
    chosen = pool[0]
    maps = {tuple(sorted(c.mapping.items())) for c in pool}
    info = {
        "equivalent_fits": len(cands),
        "narrowed_by": narrowed,
        "decided_by": decided if len(pool) == 1 or len(maps) == 1 else "default_order",
        "ambiguous": len(maps) > 1,
        "mirror_determined": all(c.fit.mirrored == chosen.fit.mirrored for c in pool),
        "alternative_maps": [dict(m) for m in sorted(maps) if m != tuple(sorted(chosen.mapping.items()))][:3],
    }
    return chosen, info


def _line_term(candidate: _Candidate, n_drawn: int, tol: float) -> float:
    """RMS over every drawn line of its offset, or ``tol`` when it is not assigned."""
    if not n_drawn:
        return 0.0
    sq = sum(v * v for v in candidate.offsets.values()) + (n_drawn - len(candidate.offsets)) * tol * tol
    return math.sqrt(sq / n_drawn)


def propose_binding(model: PartModel, order: list[int], uid_of: dict[int, str], parsed: ParsedFlat,
                    cfg: CompileConfig, recipe_map: dict[str, str] | None = None,
                    recipe_mirrored: bool | None = None, shop_k: float = DEFAULT_SHOP_K) -> FlatBindingProposal:
    """Fit the flat to the K-parametrised development and bind its lines (module docstring)."""
    flat_pts, flat_a, flat_b = parsed.outline_segments(0.5)
    if len(flat_pts) < 10:
        raise FlatParseError("NO_CLOSED_OUTLINE", "flat outline too small to fit")
    gk_of = {uid: gk for gk, uid in uid_of.items()}
    n_drawn = len(parsed.bend_lines)
    tol = cfg.flat_line_offset_tol_mm

    def fit(dev: Development, iterations: int, seeds: list[FlatFit] | None) -> list[FlatFit]:
        return fit_flat(flat_pts, dev.boundary, iterations=iterations, seeds=seeds, flat_segs=(flat_a, flat_b),
                        dev_segs=(dev.seg_a, dev.seg_b))

    def evaluate(k: float, seeds: list[FlatFit] | None):
        dev = development_2d(model, order, k)
        fits = fit(dev, 25 if seeds is None else 8, seeds)
        chosen, info = _choose_fit(fits, dev, parsed, model, order, uid_of, gk_of, cfg, recipe_map, recipe_mirrored)
        score = math.sqrt(chosen.fit.rms ** 2 + _line_term(chosen, n_drawn, tol) ** 2)
        return score, chosen, info, fits, dev

    lo, hi = K_SEARCH_RANGE
    band = k_plausible_band(shop_k, cfg.shop_k_uncertainty)
    # the outline-equivalent family is established once from all eight starts and carried through the scan
    seeds = _equivalent(evaluate(min(max(shop_k, lo), hi), None)[3])
    results = []
    for k in np.round(np.arange(lo, hi + 1e-9, _K_GRID_STEP), 4):
        score, _, _, fits, _ = evaluate(float(k), seeds)
        seeds = _equivalent(fits) or seeds
        results.append((score, float(k), seeds))
    _, best_k, best_seeds = min(results, key=lambda r: (r[0], r[1]))
    # golden-section refinement on [best_k - step, best_k + step] (the objective has no sampling noise)
    a, b = max(lo, best_k - _K_GRID_STEP), min(hi, best_k + _K_GRID_STEP)
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc = evaluate(c, best_seeds)[0]
    fd = evaluate(d, best_seeds)[0]
    for _ in range(14):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = evaluate(c, best_seeds)[0]
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = evaluate(d, best_seeds)[0]
    k_star = round(0.5 * (a + b), 6)
    # full re-evaluation at K* with all starts: the binding and the ambiguity measure
    _, chosen, info, fits, dev = evaluate(k_star, None)
    best = chosen.fit
    runner = next((f.rms for f in fits if f is not best and (f.mirrored != best.mirrored or not np.allclose(
        f.rotation, best.rotation, atol=0.05))), None)
    lines, mapping = chosen.lines, chosen.mapping
    # K is used only inside the plausible band; the QA is the disagreement left at the K actually used
    k_used = round(min(max(k_star, band[0]), band[1]), 6)
    qa_fit, qa_dev = best, dev
    if abs(k_used - k_star) > 1e-9:
        qa_dev = development_2d(model, order, k_used)
        qa_fit = fit(qa_dev, 25, [best])[0]
    use_map = recipe_map if recipe_map else mapping
    # the map is checked where the binding was established (K*); the QA where K is used
    bind_offsets = _line_offsets_for(use_map, lines, best, dev, gk_of) if use_map else {}
    qa_offsets = (bind_offsets if qa_dev is dev else
                  (_line_offsets_for(use_map, lines, qa_fit, qa_dev, gk_of) if use_map else {}))
    extent = _extent_deltas(flat_pts, qa_fit, qa_dev, order)
    finite = [v for v in qa_offsets.values() if math.isfinite(v)]
    max_delta = max(finite + [extent])
    clamp_note = ("" if k_used == k_star else
                  f"; K*={k_star:.4f} from the flat lies outside the plausible band [{band[0]:.3f}, {band[1]:.3f}] "
                  f"(shop K {shop_k:g} +- {K_PLAUSIBLE_SIGMAS:g} x {cfg.shop_k_uncertainty:g}), evaluated at the edge")
    detail = (f"K={k_used:.4f} (flat best fit K*={k_star:.4f}) outline rms {qa_fit.rms:.4f} mm, max line offset "
              f"{max(finite) if finite else float('nan'):.4f} mm, max extent delta {extent:.4f} mm{clamp_note}")
    sweep_t = [model.bends[gk].sweep_rad * model.bends[gk].thickness for gk in order] or [1.0]
    resid = math.sqrt(qa_fit.rms ** 2 + (float(np.mean(np.square(finite))) if finite else 0.0))
    k_unc = float(min(0.2, max(0.005, 2.0 * resid / max(float(np.mean(sweep_t)), 1e-6))))
    k_unc = max(k_unc, abs(k_star - k_used))
    complete = len(mapping) == len(order)
    ambiguous = bool(info["ambiguous"])
    confidence = 1.0
    confidence *= 1.0 if complete else len(mapping) / max(1, len(order))
    confidence *= 1.0 / (1.0 + max(0.0, best.rms - 0.05) / 0.25)
    if ambiguous:
        confidence *= 0.5
    if not parsed.bend_lines:
        confidence *= 0.8  # lines inferred, not drawn
    if k_used != k_star:
        confidence *= 0.5
    needs_review = (not complete) or ambiguous or confidence < 0.9 or k_used != k_star or (
        max_delta > cfg.developed_length_threshold_mm)
    return FlatBindingProposal(mapping, best.mirrored, float(confidence), bool(needs_review), k_used, k_unc, best,
                               runner, bind_offsets or chosen.offsets, lines, max_delta, detail,
                               k_flat_fit=k_star, k_band=band, fit_choice=info, qa_line_offsets_mm=qa_offsets)


def _match_inferred(lines: list[FlatBendLine], fit: FlatFit, dev: Development, order: list[int]) -> list[int]:
    """Graph key of each inferred line (they were generated from ``order``)."""
    out = []
    for ln in lines:
        a = fit.apply(np.stack([ln.start, ln.end]))
        mid = a.mean(axis=0)
        best = min(order, key=lambda gk: float(np.linalg.norm(0.5 * (dev.lines[gk][0] + dev.lines[gk][1]) - mid)))
        out.append(best)
    return out


# ------------------------------------------------------------------ Method B: fold the bound flat

#: Method B policy (module docstring, "Method B"). Module constants rather than CompileConfig
#: fields, so the configuration hash of every compile without a bound flat is unchanged.
FOLD_POLICY_VERSION = "E1-FOLD-1 (2026-10-17)"
_FOLD_MAX_ITERATIONS = 8
#: Gauss-Newton stops once no strip length changes by more than this (mm).
_FOLD_CONVERGED_MM = 1e-4
#: A fold whose last step still moved a strip by more than this (mm) did not converge.
_FOLD_NOT_CONVERGED_MM = 1e-3
#: Weight of one drawn-line residual against one 0.5 mm outline sample (both in mm). The strips
#: follow the drawn lines (the flat's own statement of each strip centre); the outline fixes the
#: pose and what the lines leave open, so a panel drawn off size shows as a residual instead of
#: being absorbed into the strips (at weight 10 a 0.1 mm end error moved a strip by 0.16 mm).
_FOLD_LINE_WEIGHT = 100.0
#: Outline samples farther than this from the development are not the same edge (mm).
_FOLD_OUTLIER_MM = 1.0
#: The per-bend fold must explain the flat to drafting precision (mm): after it, every bound line
#: and every extent across a bend direction agree within this. A shop flat built from per-bend
#: allowances is explained to ~1e-6 mm; a panel drawn 0.05 mm off size is not, and is never
#: absorbed into the strips (the stages then stay Method A).
_FOLD_MAX_RESIDUAL_MM = 0.03
#: ... and the 99th percentile of the outline's distance to the folded development stays within
#: this (mm): looser than the lines, for the 5 deg chords of CAD arcs and the DXF arc flattening.
_FOLD_MAX_OUTLINE_P99_MM = 0.05
#: Smallest singular value of the strip block, pose projected out, for every strip to be observable.
_FOLD_MIN_SINGULAR = 1e-3


@dataclass
class FlatFold:
    """Method B result (:func:`fold_flat`): the bound flat's strip length per recipe bend.

    ``explained``: every strip is observable, the fold converged and it explains
    the flat to drafting precision, so ``strips_mm`` are the flat's strips.
    ``used``: the stages fold the flat (``neutral_lengths``: the flat's strips,
    each clamped into the plausible K band). ``excess_mm`` is, per bend, how far
    the flat's strip lies outside that band (0 inside); the compile adds it to
    the developed-length QA.
    """

    used: bool
    reason: str
    #: graph key -> strip length (mm) the stages use (the flat's, clamped into the plausible band);
    #: empty when not used
    neutral_lengths: dict[int, float] = field(default_factory=dict)
    #: graph key -> strip length minus ``sweep (Ri + K t)`` at the bound K (mm)
    deltas_mm: dict[int, float] = field(default_factory=dict)
    #: graph key -> the K that gives this strip, ``(L / sweep - Ri) / t``
    k_equivalent: dict[int, float] = field(default_factory=dict)
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    min_singular: float | None = None
    explained: bool = False
    #: graph key -> strip length solved from the flat (mm), whether used or not
    strips_mm: dict[int, float] = field(default_factory=dict)
    #: graph key -> distance (mm of strip) beyond the plausible K band, 0 inside it
    excess_mm: dict[int, float] = field(default_factory=dict)
    #: when explained: the flat against the development of the stage strips (clamped into the
    #: plausible band), i.e. the nearest plausible per-bend development: the per-bend QA
    qa: dict[str, float] = field(default_factory=dict)

    @property
    def qa_mm(self) -> float | None:
        """Per-bend developed-length disagreement: the larger of bound-line offset and extent delta."""
        if not self.qa:
            return None
        return max(self.qa.get("max_line_offset_mm", math.inf), self.qa.get("max_extent_delta_mm", math.inf))

    def to_json(self, uid_of: dict[int, str]) -> dict[str, Any]:
        def r6(v: float) -> float | None:
            return round(float(v), 6) if math.isfinite(float(v)) else None

        keys = sorted(self.deltas_mm, key=lambda g: uid_of.get(g, str(g)))
        return {
            "policy": FOLD_POLICY_VERSION, "used": self.used, "explained": self.explained, "reason": self.reason,
            "iterations": self.iterations,
            "min_singular": None if self.min_singular is None else r6(self.min_singular),
            "before": {k: r6(v) for k, v in sorted(self.before.items())},
            "after": {k: r6(v) for k, v in sorted(self.after.items())},
            "qa_against_stage_strips": {k: r6(v) for k, v in sorted(self.qa.items())},
            "per_bend": {uid_of.get(g, str(g)): {"strip_mm": r6(self.strips_mm.get(g, math.nan)),
                                                  "stage_strip_mm": r6(self.neutral_lengths.get(g, math.nan)),
                                                  "delta_mm": r6(self.deltas_mm[g]),
                                                  "k_equivalent": r6(self.k_equivalent.get(g, math.nan)),
                                                  "excess_mm": r6(self.excess_mm.get(g, math.nan))}
                         for g in keys},
        }


def _fold_metrics(parsed: ParsedFlat, flat_pts: np.ndarray, line_map: dict[str, str], fit: FlatFit,
                  dev: Development, order: list[int], gk_of: dict[str, int]) -> dict[str, float]:
    """Agreement of the flat with a development: bound lines, extents and the whole outline."""
    from scipy.spatial import cKDTree

    offsets = _line_offsets_for(line_map, parsed.bend_lines, fit, dev, gk_of)
    finite = [v for v in offsets.values() if math.isfinite(v)]
    dist = _segment_distances(fit.apply(flat_pts), cKDTree(dev.boundary), dev.seg_a, dev.seg_b)
    return {"outline_rms_mm": float(fit.rms),
            "outline_p99_mm": float(np.quantile(dist, 0.99)) if len(dist) else math.inf,
            "max_line_offset_mm": max(finite) if len(finite) == len(offsets) and finite else math.inf,
            "max_extent_delta_mm": _extent_deltas(flat_pts, fit, dev, order)}


def fold_flat(model: PartModel, order: list[int], uid_of: dict[int, str], parsed: ParsedFlat,
              proposal: FlatBindingProposal, line_map: dict[str, str], cfg: CompileConfig,
              shop_k: float = DEFAULT_SHOP_K) -> FlatFold:
    """Method B: the strip length of every recipe bend, measured from a bound flat (module docstring).

    The S_0 development is linear in the strip lengths: lengthening bend *a*'s
    strip by ``delta_a`` moves every panel beyond it (and every strip and drawn
    line beyond it) by ``delta_a`` along *a*'s in-plane strip direction, and *a*'s
    own drawn line by half of that. The strips and a small correction of the
    flat's 2D pose are solved by Gauss-Newton from the outline (point-to-segment
    normal residuals, each sample moving with the panel that carries it) and the
    bound drawn lines (weighted ``_FOLD_LINE_WEIGHT``). The fold is used only when
    every strip is observable (the strip block, pose projected out, has full rank),
    it converged, it explains the flat (bound-line offsets and extent deltas at most
    ``_FOLD_MAX_RESIDUAL_MM``, outline 99th percentile at most
    ``_FOLD_MAX_OUTLINE_P99_MM``). A strip whose equivalent K lies outside the
    plausible band is then clamped to the band edge (the flat's defect is never the
    nominal; ``excess_mm`` records it, and the compile's QA blocks one beyond the
    threshold). Otherwise the reason is returned and the stages stay Method A with
    the bound K.
    """
    from scipy.spatial import cKDTree

    k = float(proposal.k_factor)
    gk_of = {uid: gk for gk, uid in uid_of.items()}
    base = {gk: model.bends[gk].neutral_length(k) for gk in order}
    if not parsed.bend_lines:
        return FlatFold(False, "the flat has no drawn bend lines (they are inferred from the development), so its "
                               "per-bend strips are not observable")
    bound = [(ln, gk_of[line_map[ln.line_id]]) for ln in parsed.bend_lines
             if line_map.get(ln.line_id) in gk_of]
    if not bound:
        return FlatFold(False, "no drawn bend line is bound to a recipe bend")
    flat_pts, flat_a, flat_b = parsed.outline_segments(0.5)
    n = len(order)
    col = {gk: i for i, gk in enumerate(order)}
    order_set = set(order)
    strips = dict(base)
    dev = development_2d(model, order, k, neutral=strips)
    # in-plane strip directions (parent -> child) at S_0: rotations do not depend on strip lengths
    transforms = stage_panel_transforms(model, set(model.bends) - order_set,
                                        {b: model.bends[b].neutral_length(k) for b in model.bends})
    d2: dict[int, np.ndarray] = {}
    for gk in order:
        b = model.bends[gk]
        d3 = transforms[b.parent][:3, :3] @ (-b.p_out)
        v = np.array([float(d3 @ dev.x_axis), float(d3 @ dev.y_axis)])
        d2[gk] = v / max(float(np.linalg.norm(v)), 1e-12)
    path_cols = {p: [col[b] for b in model.path_bends(p) if b in order_set] for p in model.panels}

    def refit(dev_: Development, seed: FlatFit, iterations: int) -> FlatFit:
        return fit_flat(flat_pts, dev_.boundary, iterations=iterations, seeds=[seed], flat_segs=(flat_a, flat_b),
                        dev_segs=(dev_.seg_a, dev_.seg_b))[0]

    fit = refit(dev, proposal.fit, 25)
    before = _fold_metrics(parsed, flat_pts, line_map, fit, dev, order, gk_of)
    min_sv: float | None = None
    step = math.inf
    iterations = 0
    for iterations in range(1, _FOLD_MAX_ITERATIONS + 1):
        q = fit.apply(flat_pts)
        _, proj, sidx = _closest_on_segments_idx(q, cKDTree(dev.boundary), dev.seg_a, dev.seg_b)
        seg = dev.seg_b[sidx] - dev.seg_a[sidx]
        seg_len = np.linalg.norm(seg, axis=1)
        tang = seg / np.where(seg_len > 1e-9, seg_len, 1.0)[:, None]
        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        res = np.einsum("ij,ij->i", nrm, q - proj)
        keep = (seg_len > 1e-9) & (np.abs(res) <= _FOLD_OUTLIER_MM)
        if int(keep.sum()) < 10:
            return FlatFold(False, "the flat outline does not follow the development", before=before,
                            iterations=iterations)
        centre = q[keep].mean(axis=0)
        owner = dev.owner[sidx][keep]
        nk, qk, rk = nrm[keep], q[keep], res[keep]
        a_out = np.zeros((len(rk), n + 3))
        for p, cols in path_cols.items():
            sel = owner == p
            if np.any(sel):
                for c in cols:
                    a_out[sel, c] = -(nk[sel] @ d2[order[c]])
        rel = qk - centre
        a_out[:, n] = nk[:, 0] * -rel[:, 1] + nk[:, 1] * rel[:, 0]
        a_out[:, n + 1:n + 3] = nk
        line_rows, line_rhs = [], []
        for ln, gk in bound:
            ends = fit.apply(np.stack([ln.start, ln.end]))
            mid = 0.5 * (ends[0] + ends[1])
            pa, pb = dev.lines[gk]
            u = (pb - pa) / max(float(np.linalg.norm(pb - pa)), 1e-12)
            nu = np.array([-u[1], u[0]])
            if float(nu @ d2[gk]) < 0:
                nu = -nu
            row = np.zeros(n + 3)
            row[col[gk]] -= 0.5 * float(nu @ d2[gk])
            for c in path_cols[model.bends[gk].parent]:
                row[c] -= float(nu @ d2[order[c]])
            rl = mid - centre
            row[n] = float(nu[0] * -rl[1] + nu[1] * rl[0])
            row[n + 1:n + 3] = nu
            line_rows.append(_FOLD_LINE_WEIGHT * row)
            line_rhs.append(-_FOLD_LINE_WEIGHT * float(nu @ (mid - pa)))
        a_mat = np.vstack([a_out, np.asarray(line_rows)])
        rhs = np.concatenate([-rk, np.asarray(line_rhs)])
        # observability: the strip columns with the pose columns projected out must have full rank
        a_strip, a_pose = a_mat[:, :n], a_mat[:, n:]
        perp = a_strip - a_pose @ (np.linalg.pinv(a_pose) @ a_strip)
        sv = np.linalg.svd(perp, compute_uv=False)
        min_sv = float(sv.min()) if len(sv) == n else 0.0
        if min_sv < _FOLD_MIN_SINGULAR:
            return FlatFold(False, f"the bound lines and the outline do not determine every strip (smallest singular "
                                   f"value {min_sv:.2e} < {_FOLD_MIN_SINGULAR:g})", before=before,
                            iterations=iterations, min_singular=min_sv)
        x, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
        for gk in order:
            strips[gk] += float(x[col[gk]])
        rot = _rot2(float(x[n]))
        fit = FlatFit(fit.rms, rot @ fit.rotation, rot @ (fit.translation - centre) + centre + x[n + 1:n + 3],
                      fit.mirrored)
        step = float(np.max(np.abs(x[:n])))
        dev = development_2d(model, order, k, neutral=strips)
        fit = refit(dev, fit, 8)
        if step < _FOLD_CONVERGED_MM:
            break
    after = _fold_metrics(parsed, flat_pts, line_map, fit, dev, order, gk_of)
    deltas = {gk: strips[gk] - base[gk] for gk in order}
    k_eq = {gk: (strips[gk] / model.bends[gk].sweep_rad - model.bends[gk].ri) / model.bends[gk].thickness
            for gk in order}
    lo, hi = k_plausible_band(shop_k, cfg.shop_k_uncertainty)
    excess = {}
    for gk in order:
        per_k = model.bends[gk].sweep_rad * model.bends[gk].thickness
        excess[gk] = max(0.0, (lo - k_eq[gk]) * per_k, (k_eq[gk] - hi) * per_k)
    out = FlatFold(False, "", {}, deltas, k_eq, before, after, iterations, min_sv, strips_mm=dict(strips),
                   excess_mm=excess)
    if step > _FOLD_NOT_CONVERGED_MM:
        out.reason = f"the per-bend fold did not converge (last step {step:.4f} mm)"
        return out
    if (after["max_line_offset_mm"] > _FOLD_MAX_RESIDUAL_MM or after["max_extent_delta_mm"] > _FOLD_MAX_RESIDUAL_MM
            or after["outline_p99_mm"] > _FOLD_MAX_OUTLINE_P99_MM):
        out.reason = (f"per-bend strips do not explain the flat (after the fold: max line offset "
                      f"{after['max_line_offset_mm']:.3f} mm, max extent delta {after['max_extent_delta_mm']:.3f} mm "
                      f"(limit {_FOLD_MAX_RESIDUAL_MM:g}), outline p99 {after['outline_p99_mm']:.3f} mm (limit "
                      f"{_FOLD_MAX_OUTLINE_P99_MM:g})): a panel or line is drawn off the CAD, which strips must not absorb")
        return out
    out.explained = True
    out.used = True
    # a strip no plausible K gives is the flat's defect, not the process: its nominal is the nearest
    # plausible strip (the compile's QA blocks one beyond the threshold), every other strip is the flat's
    used = {}
    for gk in order:
        b = model.bends[gk]
        lo_strip, hi_strip = b.neutral_length(lo), b.neutral_length(hi)
        used[gk] = min(max(strips[gk], lo_strip), hi_strip)
    out.neutral_lengths = used
    if any(abs(used[gk] - strips[gk]) > 1e-9 for gk in order):
        # the flat against the nearest plausible per-bend development (the clamped stage strips):
        # a clamp shows in its own line by half and accumulates in every extent beyond it
        dev_q = development_2d(model, order, k, neutral=used)
        out.qa = _fold_metrics(parsed, flat_pts, line_map, refit(dev_q, fit, 8), dev_q, order, gk_of)
    else:
        out.qa = dict(after)
    out.reason = (f"per-bend strips measured from the bound flat (max |strip - sweep (Ri + K t)| "
                  f"{max(abs(v) for v in deltas.values()):.4f} mm at K = {k:.4f})")
    clamped = [gk for gk in order if excess[gk] > 1e-6]
    if clamped:
        worst = max(clamped, key=lambda g: excess[g])
        out.reason += (f"; {len(clamped)} strip(s) clamped into the plausible band [{lo:.3f}, {hi:.3f}] (largest: bend "
                       f"{uid_of.get(worst, model.bends[worst].graph_bend_id)} needs K = {k_eq[worst]:.3f}, "
                       f"{excess[worst]:.3f} mm beyond it)")
    return out


def fold_flat_or_reason(model: PartModel, order: list[int], uid_of: dict[int, str], parsed: ParsedFlat,
                        proposal: FlatBindingProposal, line_map: dict[str, str], cfg: CompileConfig,
                        shop_k: float = DEFAULT_SHOP_K) -> FlatFold:
    """:func:`fold_flat`, with an unexpected failure reported as an unused fold (Method A) and its reason.

    Method B only refines the stage strips of a bound flat; a geometry the solver
    does not handle must fall back to Method A visibly, never block the compile
    under an unrelated code or break the wizard proposal.
    """
    try:
        return fold_flat(model, order, uid_of, parsed, proposal, line_map, cfg, shop_k=shop_k)
    except Exception as exc:  # noqa: BLE001 - recorded in diagnostics and the QA detail
        return FlatFold(False, f"the per-bend fold failed ({type(exc).__name__}: {exc})"[:300])


def parse_recipe_flat(recipe: PartRecipeV1, flat_path: Path) -> tuple[ParsedFlat | None, dict[str, Any], list[tuple[str, str]]]:
    """Parse the recipe's flat: ``(parsed | None, FlatCompileV1 fields, issues)``."""
    flat = recipe.flat
    record: dict[str, Any] = {"provided": True, "format": flat.format, "sha256": flat.sha256,
                              "binding_state": flat.binding.state, "parse_status": "PARSE_FAILED"}
    try:
        parsed = parse_flat(flat_path, flat.format)
    except FlatParseError as exc:
        return None, record, [("FLAT_PARSE_FAILED", f"{exc.reason}: {exc.detail}"[:900])]
    record["parse_status"] = "PARSED"
    record["bend_lines_found"] = len(parsed.bend_lines)
    return parsed, record, []


def binding_counts_for_k(recipe: PartRecipeV1) -> bool:
    """Whether the recipe's flat binding is confirmed enough to supply K (module docstring)."""
    state = recipe.flat.binding.state if recipe.flat is not None else "UNBOUND"
    return state == "OPERATOR_CONFIRMED" or (state == "AUTO_PROPOSED" and recipe.confirmation == "DEV_AUTO_CONFIRMED")


def compile_flat(model: PartModel, recipe: PartRecipeV1, parsed: ParsedFlat, record: dict[str, Any],
                 order: list[int], uid_of: dict[int, str], cfg: CompileConfig,
                 shop_k: float = DEFAULT_SHOP_K) -> FlatOutcome:
    """Bind the recipe's parsed flat for the compile (module docstring)."""
    flat = recipe.flat
    record = dict(record)
    diag: dict[str, Any] = {"parsed": parsed.summary()}
    recipe_map = dict(flat.binding.bend_line_map) if flat.binding.state != "UNBOUND" else None
    try:
        proposal = propose_binding(model, order, uid_of, parsed, cfg, recipe_map,
                                   flat.binding.mirrored if recipe_map else None, shop_k=shop_k)
    except FlatParseError as exc:
        record["parse_status"] = "PARSE_FAILED"
        record.pop("bend_lines_found", None)
        return FlatOutcome(record, False, None, cfg.shop_k_uncertainty, None,
                           [("FLAT_PARSE_FAILED", f"{exc.reason}: {exc.detail}"[:900])], diag)
    diag["proposal"] = proposal.to_json()
    issues: list[tuple[str, str]] = []
    if recipe_map is None:
        record["developed_length_check"] = {"status": "NOT_EVALUATED", "threshold_mm": cfg.developed_length_threshold_mm,
                                            "detail": "flat not bound; proposal in diagnostics"}
        return FlatOutcome(record, False, None, cfg.shop_k_uncertainty, None, issues, diag, proposal,
                           unbound_reason="flat binding state UNBOUND; a proposal is in diagnostics")
    # verify the released map against the fit
    known_ids = {ln.line_id for ln in proposal.lines}
    bad = []
    for lid, uid in sorted(recipe_map.items()):
        off = proposal.line_offsets_mm.get(lid, float("inf"))
        if lid not in known_ids:
            bad.append(f"{lid}: unknown flat line")
        elif not math.isfinite(off) or off > cfg.flat_line_offset_tol_mm:
            bad.append(f"{lid}->{uid}: offset {off:.3f} mm")
    if flat.binding.mirrored is not None and bool(flat.binding.mirrored) != proposal.mirrored:
        bad.append(f"recipe mirrored={flat.binding.mirrored} but the fit is mirrored={proposal.mirrored}")
    missing = sorted(set(uid_of.values()) - set(recipe_map.values()))
    if missing:
        diag["unbound_bends"] = missing
    if bad:
        issues.append(("FLAT_BINDING_INCONSISTENT", "; ".join(bad)[:900]))
    status = "NOT_EVALUATED"
    max_delta = proposal.developed_max_abs_delta_mm
    if max_delta is not None and math.isfinite(max_delta):
        status = "MISMATCH" if max_delta > cfg.developed_length_threshold_mm else "OK"
    k_star = proposal.k_flat_fit if proposal.k_flat_fit is not None else proposal.k_factor
    k_edge = k_star <= K_SEARCH_RANGE[0] + 1e-6 or k_star >= K_SEARCH_RANGE[1] - 1e-6
    if k_edge:
        status = "MISMATCH"
        max_delta = max(max_delta or 0.0, cfg.developed_length_threshold_mm + 1e-6)
    detail = proposal.developed_detail + ("; K* at the search bound" if k_edge else "")
    # Method B: the flat's own strip per bend, when the flat determines them. When the fold explains
    # the flat, its strips are known, so the QA compares the flat with the nearest plausible per-bend
    # development (strips clamped into the band) instead of a single K: a bend table with a wide K
    # spread is not a mismatch, and one strip no plausible K gives is not hidden by being spread over
    # every strip (a strip error moves its own drawn line by only half of it).
    fold = None
    if not bad and status != "NOT_EVALUATED" and not k_edge:
        fold = fold_flat_or_reason(model, order, uid_of, parsed, proposal, recipe_map, cfg, shop_k=shop_k)
        diag["method_b"] = fold.to_json(uid_of)
        if fold.explained and fold.qa_mm is not None and math.isfinite(fold.qa_mm):
            uniform = float(max_delta)
            max_delta = float(fold.qa_mm)
            status = "MISMATCH" if max_delta > cfg.developed_length_threshold_mm else "OK"
            detail += (f"; per-bend QA (the fold explains the flat; it replaces the single-K check at {uniform:.3f} "
                       f"mm): {max_delta:.3f} mm from the nearest plausible per-bend development")
            worst = max(fold.excess_mm.items(), key=lambda kv: kv[1], default=(None, 0.0))
            if worst[1] > 1e-6:
                detail += (f", the flat's strip of bend {uid_of.get(worst[0], worst[0])} lies {worst[1]:.3f} mm "
                           "beyond every plausible K")
    record["developed_length_check"] = {
        "status": status, "threshold_mm": cfg.developed_length_threshold_mm,
        "max_abs_delta_mm": None if status == "NOT_EVALUATED" else round(float(max_delta), 6),
        "detail": detail[:500],
    }
    if status == "MISMATCH":
        issues.append(("FLAT_DEVELOPED_LENGTH_MISMATCH", record["developed_length_check"]["detail"]))
    consistent = not bad and status != "MISMATCH"
    confirmed = binding_counts_for_k(recipe)
    bound = consistent and confirmed
    unbound_reason = None
    if consistent and not confirmed:
        unbound_reason = (f"flat binding {flat.binding.state} is not operator-confirmed (recipe confirmation "
                          f"{recipe.confirmation}); the flat is consistent but K stays the shop default")
    neutral = None
    method = "METHOD_A_ROTATE_BACK"
    k_unc = proposal.k_uncertainty
    if bound:
        neutral = {gk: model.bends[gk].neutral_length(proposal.k_factor) for gk in model.bends}
        check = record["developed_length_check"]
        if fold is not None and fold.used:
            neutral.update(fold.neutral_lengths)
            method = "METHOD_B_FOLD_FLAT"
            k_unc = _fold_k_uncertainty(model, order, fold)
            note = f"; stages fold the flat (Method B): {fold.reason}"
        else:
            why = fold.reason if fold is not None else "the per-bend fold was not evaluated"
            note = f"; stages use K={proposal.k_factor:.4f} uniformly (Method A): {why}"
        check["detail"] = (check["detail"] + note)[:500]
    if "method_b" in diag:
        # whether the compile staged it: a fold can be usable while the binding gives no K (unconfirmed, QA)
        diag["method_b"]["applied"] = method == "METHOD_B_FOLD_FLAT"
    return FlatOutcome(record, bound, proposal.k_factor if bound else None, k_unc, neutral, issues,
                       diag, proposal, unbound_reason=unbound_reason, method=method, fold=fold)


def _fold_k_uncertainty(model: PartModel, order: list[int], fold: FlatFold) -> float:
    """K-equivalent standard uncertainty of Method B strips: what the fold leaves unexplained.

    The strips are the flat's own, so the uniform-K residual no longer applies;
    the remaining disagreement (outline RMS and bound-line offsets after the fold)
    is expressed per unit of ``sweep t`` as for a uniform fit, floored at 0.005.
    """
    resid = math.hypot(fold.after.get("outline_rms_mm", 0.0), fold.after.get("max_line_offset_mm", 0.0))
    sweep_t = [model.bends[gk].sweep_rad * model.bends[gk].thickness for gk in order] or [1.0]
    return float(min(0.2, max(0.005, 2.0 * resid / max(float(np.mean(sweep_t)), 1e-6))))
