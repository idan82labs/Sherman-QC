"""Per-scan inspection orchestration (E3): ingest -> registration -> articulated fit -> decode -> metrology.

:func:`inspect_scan` runs the plan §3.2 pipeline on verified inputs
(:func:`domains.production.engine.inspect.part.load_inputs`) and writes the
ENGINE_IO §4.1 output directory. Every inspection outcome is a result, never
an exception: an unreadable scan, a suspected unit mismatch, no datum plane,
a mirrored or wrong part and a timeout all produce a valid
``ShopfloorInspectionResultV1`` with a non-VALID ``inspection_validity`` and
nothing MEASURED.

Validity decisions (in order):

1. ``UNSUPPORTED_INPUT``: the scan bytes are not a readable PLY/STL.
2. ``INSUFFICIENT_CAPTURE``: fewer than ``min_scan_points`` finite points.
3. ``UNIT_MISMATCH_SUSPECTED``: the scan extent is off the compiled part's (both
   its blank S_0 and its final shape S_N) by more than ``unit_ratio_limit``
   (an inch or metre export read as mm, or a wrong ``--unit-factor``). The
   scan is never rescaled.
4. ``UNTRUSTED_REGISTRATION``: no datum plane matches, or the datum is barely
   covered.
5. ``MIRRORED_PART_SUSPECTED``: the mirrored scan registers clearly better
   (its datum outline, holes and articulated fit) than any proper pose. The
   improper transform is never published.
6. ``WRONG_PART_SUSPECTED``: the datum's holes are filled with material, or too
   little of the scan is explained by the fitted flanges, after every refined
   datum pose (not only the best few) got its articulated fit.
7. ``TIMEOUT``: the internal budget ran out before the registration finished:
   the datum fit, or the pose selection (every competing pose's articulated
   fit, and the mirrored scan's when the best pose looked suspicious). A cut
   articulation explains less of the scan, so it is never judged a wrong or
   mirrored part. A budget that runs out later (metrology) leaves a VALID
   result whose unmeasured characteristics are UNKNOWN(TIMEOUT).

Ingest drops non-finite and gross-outlier vertices (``gross_outlier_extent_factor``
robust extents from the trimmed box), counting them in a warning; every
index-aligned output file describes the kept points.

Symmetric twins: a pose whose motion to a top pose is a ROTATION self-map of
some compiled stage (E1's symmetry test, rms within its threshold) is always
articulated, and a twin whose total score or explained shares are within
``ambiguity_margin`` of the best competes with it. Among competing twins the
planned order is the soft prior (an order-consistent decode wins). Pose
ambiguity (``pose_ambiguous``) is declared only when E1's symmetry test of the
chosen decode's own stage matches the motion to a competing twin and the two
disagree on a bend's identity or state.

Folds formed the wrong way (the mirrored hinge search, :func:`~domains.production.engine.inspect.articulated.fit_articulated`)
are looked for on the chosen pose only, after the pose decisions above: on every
candidate pose that search would let a datum turned over read each of its bends
as formed the wrong way and explain the scan as well as the true pose.

Unplanned creases are found before the metrology: a crease along a bend ends
the FREE_EDGE flange it splits at its OML virtual sharp (UNKNOWN when that
sharp cannot be resolved).
"""
from __future__ import annotations

import math
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from contracts.production.common import round6
from contracts.production.consistency import check_result_against_compiled
from contracts.production.inspection_result import (
    BendGeometryFileV1,
    CoverageV1,
    EngineInfoV1,
    RegistrationV1,
    ResultContextV1,
    ShopfloorInspectionResultV1,
    UnplannedCreaseV1,
    ViewerAssetsV1,
)
from contracts.production.layout import ENGINE_OUTPUT_FILES
from domains.production.engine.inspect.articulated import ArticulatedFit, fit_articulated
from domains.production.engine.inspect.cloud import (
    VoxelCloud,
    adaptive_voxel,
    estimate_normals,
    gross_inliers,
    median_spacing,
    neighbour_graph,
    voxelize,
)
from domains.production.engine.inspect.config import ALGORITHM_VERSIONS, ENGINE_VERSION, InspectConfig
from domains.production.engine.inspect.decode import BendDecode, choose_stage_model, decode_states
from domains.production.engine.inspect.geom import apply, rigid, robust_plane, signed_sweep_deg, skin_peaks, unit
from domains.production.engine.inspect.metrology import (
    BendMeasure,
    Measurement,
    PointIndex,
    Thickness,
    angle_measurement,
    bare_edge_untrusted,
    measure_backgauge,
    measure_bend_geometry,
    measure_flange,
    position_measurement,
    reversed_position_measurement,
    sharp_unit_along,
    u_budget,
)
from domains.production.engine.inspect.outputs import copy_file, write_array, write_json, write_points_ply
from domains.production.engine.inspect.part import Inputs, PartView
from domains.production.engine.inspect.registration import (
    MIRROR,
    DatumModel,
    InPlaneFit,
    coarse_hypotheses,
    fine_search,
    hole_pair_hypotheses,
    refine_datum,
)
from domains.production.engine.inspect.result import BendOutcome, ResultInputs, build_result
from domains.production.engine.inspect.scan_io import ScanData, ScanReadError, read_scan

__all__ = ["InspectOptions", "InspectOutcome", "InspectInternalError", "inspect_scan"]

_FORMED_LIKE = ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED")


class InspectInternalError(RuntimeError):
    """The engine produced a result that fails its own contract checks (CLI exit 3)."""


@dataclass
class InspectOptions:
    context: ResultContextV1 = field(default_factory=ResultContextV1)
    declared_stage: int | None = None
    declared_source: str | None = None
    unit_factor: float = 1.0
    engine_bundle_id: str | None = None
    time_budget_s: float | None = None
    created_at: str | None = None
    write_outputs: bool = True
    #: e3-1.13 safety net (internal): inspect a seeded subsample ``(fraction, seed)`` of the scan's points
    subsample: tuple[float, int] | None = None
    #: e3-1.13 safety net (internal): build the result but defer writing it (``_Ctx.finish_args`` keeps the inputs)
    defer_finish: bool = False
    #: e3-1.13 dense ingest (internal): the primary's thinning decision, inherited by its resamples (None: decide)
    dense_thin: bool | None = None


@dataclass
class InspectOutcome:
    result: ShopfloorInspectionResultV1
    files: dict[str, str]
    log: list[str]


class _Budget:
    def __init__(self, seconds: float) -> None:
        self.t0 = time.monotonic()
        self.seconds = seconds

    def left(self) -> float:
        return self.seconds - (time.monotonic() - self.t0)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


def _peak_rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _robust_extent(pts: np.ndarray, trim: float) -> float:
    lo = np.percentile(pts, trim, axis=0)
    hi = np.percentile(pts, 100.0 - trim, axis=0)
    return float(np.linalg.norm(hi - lo))


@dataclass(eq=False)  # identity semantics: hypotheses hold numpy arrays
class _Hyp:
    fit: InPlaneFit
    art: ArticulatedFit | None = None
    decodes: list[BendDecode] | None = None
    explained: float = 0.0
    explained_tight: float = 0.0
    total: float = -1.0
    mirrored: bool = False
    #: set on the chosen hypothesis by :func:`_ambiguity`: the certified symmetric twins it ties with, and per
    #: twin the bend permutations of the matching E1 symmetry candidates (twin label = perm[chosen label])
    twins: list = field(default_factory=list)
    twin_perms: list = field(default_factory=list)
    #: added by the free-datum search (``_free_datum_poses``): always articulated, among the best ``free_datum_rank_k``
    free_extra: bool = False


#: e3-1.13 safety net: fixed seeds of the extra resamples (the first ``resample_check_k`` are used)
RESAMPLE_SEEDS = (7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108)


def inspect_scan(inputs: Inputs, out_dir: Path, cfg: InspectConfig, opts: InspectOptions,
                 progress: Callable[[str, int], None] | None = None) -> InspectOutcome:
    """Run the per-scan pipeline and write the output directory (module docstring).

    e3-1.13 safety net (``resample_check_k`` > 0): the primary inspection is built but not yet written; then
    ``resample_check_k`` deterministic subsamples of the scan (``resample_check_fraction``, seeds
    :data:`RESAMPLE_SEEDS`) are inspected the same way, within what is left of the time budget, serially or in
    forked worker processes (``resample_check_parallel``). Their readings widen the U of the primary's decided
    characteristics (:func:`_resample_policy`), and the primary result is built again and written. A check that
    cannot finish (budget, or a resample that is not VALID while the primary is) withholds every decided
    characteristic: the check never lets an unchecked decision through, and it never creates a PASS.
    """
    k = int(cfg.resample_check_k)
    if k <= 0 or opts.subsample is not None:
        return _inspect_once(inputs, out_dir, cfg, opts, progress)
    import dataclasses
    budget = _Budget(opts.time_budget_s if opts.time_budget_s is not None else cfg.time_budget_s)
    popts = dataclasses.replace(opts, defer_finish=True)
    primary, ctx = _inspect_once(inputs, out_dir, cfg, popts, progress, keep_ctx=True)
    if ctx is None or ctx.finish_args is None:
        return primary
    seeds = RESAMPLE_SEEDS[:k]
    readings: list[dict | None] = []
    t_primary = budget.elapsed()
    valid = primary.result.inspection_validity.state == "VALID"
    if valid:
        if progress is not None:
            progress("resample_check", 90)
        ropts = dataclasses.replace(opts, dense_thin=bool(getattr(ctx, "dense_thinned", False)))
        readings = _resample_readings(inputs, out_dir, cfg, ropts, seeds, budget, t_primary)
    ctx.note(f"resample check: k={k} policy={cfg.resample_check_policy} parallel={cfg.resample_check_parallel} "
             f"primary {t_primary:.1f}s total {budget.elapsed():.1f}s; "
             + " ".join(f"s{sd}={'-' if r is None else r['validity']}" for sd, r in zip(seeds, readings)))
    rec = None
    if valid:
        rec = _resample_policy(ctx, readings, cfg)
    ctx.opts = dataclasses.replace(opts, defer_finish=False)
    ctx.budget = budget
    out = ctx.finish(*ctx.finish_args[0], **ctx.finish_args[1])
    if cfg.resample_check_record and opts.write_outputs:
        rec = rec or {}
        rec.update(seeds=list(seeds), primary_s=round(t_primary, 3), total_s=round(budget.elapsed(), 3),
                   resamples=readings, primary_raw=_readings_of(primary.result))
        write_json(Path(out_dir) / "resample_check.json", rec)
    return out


def _readings_of(result) -> dict:
    """Per characteristic (decision, actual, U) and the validity of a result (safety-net bookkeeping)."""
    chars = {}
    for b in result.bends:
        for c in b.characteristics:
            chars[c.characteristic_id] = (c.decision, c.actual, c.uncertainty)
    for g in result.general_measurements:
        c = g.characteristic
        chars[c.characteristic_id] = (c.decision, c.actual, c.uncertainty)
    return {"validity": result.inspection_validity.state, "summary": result.summary.state, "chars": chars}


def _one_resample(inputs: Inputs, out_dir: Path, cfg: InspectConfig, opts: InspectOptions, seed: int,
                  seconds: float) -> dict:
    import dataclasses
    t0 = time.monotonic()
    o2 = dataclasses.replace(opts, write_outputs=False, defer_finish=False, time_budget_s=max(seconds, 1.0),
                             subsample=(float(cfg.resample_check_fraction), int(seed)))
    res = _inspect_once(inputs, out_dir, cfg, o2, None)
    r = _readings_of(res.result)
    r["seconds"] = round(time.monotonic() - t0, 3)
    r["peak_rss_bytes"] = _peak_rss()
    return r


def _resample_readings(inputs, out_dir, cfg: InspectConfig, opts: InspectOptions, seeds, budget: _Budget,
                       t_primary: float) -> list[dict | None]:
    """Readings of the resamples (None: not run, the budget would not hold it)."""
    out: list[dict | None] = []
    if cfg.resample_check_parallel:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        left = budget.left() - 2.0
        if left < 0.5 * t_primary:
            return [None for _ in seeds]
        # spawn on every platform: Windows has no fork, and a spawned worker shares no state with the parent
        with ProcessPoolExecutor(max_workers=len(seeds), mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_one_resample, inputs, out_dir, cfg, opts, sd, left) for sd in seeds]
            for f in futs:
                try:
                    out.append(f.result(timeout=max(left + 5.0, 1.0)))
                except Exception:  # noqa: BLE001 - a failed check run withholds (fail safe)
                    out.append(None)
        return out
    for i, sd in enumerate(seeds):
        left = budget.left() - 2.0
        share = left / (len(seeds) - i)
        # a resample takes about as long as the primary: one that cannot get that much is not started
        if share < 0.8 * t_primary or left <= 1.0:
            out.append(None)
            continue
        try:
            out.append(_one_resample(inputs, out_dir, cfg, opts, sd, share))
        except Exception:  # noqa: BLE001 - a failed check run withholds (fail safe)
            out.append(None)
    return out


def _resample_policy(ctx: "_Ctx", readings: list[dict | None], cfg: InspectConfig) -> dict:
    """Widen the primary's decided characteristics' U from the resamples (``resample_check_policy``).

    1 (self-check): a PASS or FAIL that any resample decides otherwise (another decision, UNKNOWN, a resample
    that could not run or is not VALID) becomes REVIEW (U = U + spread, at least the guard-band width).
    2 (honest U): U' = sqrt(U^2 + (2 sigma)^2), sigma the sample deviation of the primary and resample values;
    a resample without a value (or not run) withholds. Characteristics already REVIEW / UNKNOWN are untouched.
    """
    part = ctx.part
    args = ctx.finish_args[0]
    outcomes, general = args[4], ctx.general[0]
    incomplete = any(r is None or r["validity"] != "VALID" for r in readings)
    if incomplete:
        ctx.warnings.append("RESAMPLE_CHECK_INCOMPLETE: decided characteristics withheld (REVIEW)")
    changed: dict[str, list] = {}
    primary_raw = {}
    items = [(cid, m, o) for o in outcomes for cid, m in o.measurements.items()] + \
        [(cid, m, None) for cid, m in general.items()]
    for cid, m, o in items:
        if m.value is None or m.u is None or (o is not None and (cid in o.forced_unknown or cid in o.stage_na)):
            continue
        if o is not None and o.state in ("UNKNOWN", "NOT_YET_FORMED"):
            continue
        nom = part.nominals.get(cid)
        d0 = _gb(nom, m.value, m.u)
        primary_raw[cid] = (d0, float(m.value), float(m.u))
        if d0 not in ("PASS", "FAIL"):
            continue
        others = [None if (r is None or r["validity"] != "VALID") else r["chars"].get(cid) for r in readings]
        vals = [float(x[1]) for x in others if x is not None and x[1] is not None]
        u0 = float(m.u)
        if cfg.resample_check_policy == 1:
            agree = all(x is not None and x[0] == d0 for x in others)
            if agree:
                continue
            spread = max([abs(v - float(m.value)) for v in vals] + [0.0])
            m.u = withhold_u(nom, m.value, u0, spread)
        else:
            if len(vals) < len(others):
                m.u = withhold_u(nom, m.value, u0, 0.0)
            else:
                allv = np.asarray([float(m.value)] + vals)
                sig = float(allv.std(ddof=1)) if len(allv) > 1 else 0.0
                m.u = float(math.hypot(u0, 2.0 * sig))
        d1 = _gb(nom, m.value, m.u)
        if d1 != d0:
            m.notes = tuple(m.notes) + ("RESAMPLE_CHECK_WITHHELD",)
            changed[cid] = [d0, d1, round(u0, 6), round(float(m.u), 6)]
    ctx.primary_raw = primary_raw
    if changed:
        ctx.note("resample check withheld: " + " ".join(changed))
    return {"policy": int(cfg.resample_check_policy), "incomplete": incomplete, "changed": changed}


def _inspect_once(inputs: Inputs, out_dir: Path, cfg: InspectConfig, opts: InspectOptions,
                  progress: Callable[[str, int], None] | None = None, keep_ctx: bool = False):
    """One pass of the per-scan pipeline (``inspect_scan`` without the safety net)."""
    out = _inspect_core(inputs, out_dir, cfg, opts, progress)
    if keep_ctx:
        return out, getattr(out, "_ctx", None)
    return out


def _inspect_core(inputs: Inputs, out_dir: Path, cfg: InspectConfig, opts: InspectOptions,
                  progress: Callable[[str, int], None] | None = None) -> InspectOutcome:
    """Run the per-scan pipeline and write the output directory (module docstring)."""
    budget = _Budget(opts.time_budget_s if opts.time_budget_s is not None else cfg.time_budget_s)
    log: list[str] = []

    def note(msg: str) -> None:
        log.append(f"[{budget.elapsed():7.2f}s] {msg}")

    def step(stage: str, pct: int) -> None:
        note(stage)
        if progress is not None:
            progress(stage, pct)

    part = PartView(inputs.compiled, inputs.compiled_dir)
    recipe = inputs.recipe
    out_dir = Path(out_dir)
    warnings: list[str] = []
    ctx = _Ctx(part=part, recipe=recipe, inputs=inputs, cfg=cfg, opts=opts, out_dir=out_dir, warnings=warnings,
               note=note, budget=budget)
    step("ingest", 5)
    try:
        scan = read_scan(inputs.scan_path)
    except ScanReadError as exc:
        return ctx.invalid("UNSUPPORTED_INPUT", (f"SCAN_UNREADABLE: {exc}"[:200],), None, None, log)
    ctx.scan = scan
    pts = scan.points * float(opts.unit_factor)
    if opts.subsample is not None:  # e3-1.13 safety net: a seeded subsample of the points
        rng = np.random.default_rng(int(opts.subsample[1]))
        pts = pts[np.sort(rng.choice(len(pts), size=int(round(float(opts.subsample[0]) * len(pts))), replace=False))]
    if scan.dropped_nonfinite:
        warnings.append(f"SCAN_NONFINITE_POINTS_DROPPED:{scan.dropped_nonfinite}")
    # gross outliers (a finite sentinel vertex metres or 1e38 away) are dropped like non-finite
    # vertices: the registered scan and every index-aligned file hold the kept points only
    keep = gross_inliers(pts, cfg.extent_trim_percent, cfg.gross_outlier_extent_factor)
    if not keep.all():
        warnings.append(f"SCAN_GROSS_OUTLIERS_DROPPED:{int((~keep).sum())}")
        pts = pts[keep]
    ctx.points = pts
    if len(pts) < cfg.min_scan_points:
        return ctx.invalid("INSUFFICIENT_CAPTURE", (f"SCAN_POINTS:{len(pts)}",), None, None, log)
    # -- unit check (no auto-scale, ever) ------------------------------------------------
    ext = _robust_extent(pts, cfg.extent_trim_percent)
    ratios = (ext / max(part.outline_extent(), 1e-9), ext / max(part.blank_extent(), 1e-9))
    if not any(1.0 / cfg.unit_ratio_limit <= r <= cfg.unit_ratio_limit for r in ratios):
        r = ratios[0]
        hint = ""
        for f, name in ((25.4, "INCH"), (1000.0, "METRE"), (1 / 25.4, "MM_AS_INCH"), (1e-3, "MM_AS_METRE")):
            if abs(math.log(r * f)) < math.log(1.5):
                hint = f";LIKELY_{name}"
        return ctx.invalid("UNIT_MISMATCH_SUSPECTED", (f"SCAN_TO_CAD_EXTENT_RATIO:{r:.4g}{hint}",), None, None, log)
    # e3-1.14 dense ingest (after the unit check): a cloud denser than the validated native envelope
    # (``density_guard_ppc`` points per occupied ``density_guard_cell_mm`` cell) is thinned to one actual point per
    # cell when ``dense_thin`` is on (its resamples inherit the decision); otherwise, and for a cloud that still holds
    # more than ``density_guard_max_points`` points, the e3-1.13 density guard refuses it (fail closed)
    thin = opts.dense_thin
    if thin is None:
        thin = bool(cfg.dense_thin) and cfg.density_guard_ppc > 0 and len(pts) > 0 and \
            _cell_density(pts, cfg.density_guard_cell_mm) > cfg.density_guard_ppc
    if thin:
        n0 = len(pts)
        ppc = _cell_density(pts, cfg.density_guard_cell_mm)
        pts = _dense_thin(pts, cfg.density_guard_cell_mm)
        ctx.points = pts
        note(f"dense input thinned: {n0} -> {len(pts)} points (one per {cfg.density_guard_cell_mm:g} mm cell; "
             f"{ppc:.2f} points per cell, trigger {cfg.density_guard_ppc:g})")
    ctx.dense_thinned = bool(thin)
    dense = _density_guard(pts, cfg)
    if dense is not None:
        note(f"density guard: {dense}")
        return ctx.invalid("UNSUPPORTED_INPUT", (dense,), None, None, log)
    # -- clouds ----------------------------------------------------------------------------
    step("clouds", 10)
    spacing = median_spacing(pts)
    voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points),
                min(cfg.voxel_max_mm, spacing))
    vox = voxelize(pts, voxel)
    slab = cfg.slab_mode(part.t, voxel)
    vox.slab = slab
    normal_voxel = max(voxel, cfg.thin_normal_voxel_factor * part.t) if slab else None
    estimate_normals(vox, cfg.normal_neighbours, normal_voxel=normal_voxel)
    neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
    note(f"points={len(pts)} spacing={spacing:.3f} voxel={voxel:.2f} voxels={len(vox.points)}")
    ctx.spacing = spacing
    # -- registration -----------------------------------------------------------------------
    step("registration", 20)
    model = DatumModel.from_part(part)
    ctx.model = model
    # the datum fit needs the scanner's native density, not more: dense exports are thinned
    # to one point per ``registration_decimate_mm`` cell (deterministic) for the registration only
    reg_src = _decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
    reg_spacing = median_spacing(reg_src) if reg_src is not pts else spacing
    proper = _register(reg_src, vox, part, model, cfg, budget, note, reg_spacing, mirrored=False)
    if budget.left() <= 0:
        return ctx.invalid("TIMEOUT", ("TIMEOUT_DURING_REGISTRATION",), None, None, log)
    if not proper:
        return ctx.invalid("UNTRUSTED_REGISTRATION", ("NO_DATUM_PLANE_MATCHED",), None, None, log, abstained=True)
    step("articulated", 40)
    # the best datum poses get the articulated fit, and so does every symmetric twin of one of
    # them (a motion matching a compiled stage's symmetry candidate, or a near symmetry within
    # ``near_twin_rms_mm``): a twin explains the scan as well, so the decode must see it, whatever
    # its noisy datum score
    scored = list(proper[:cfg.articulated_hypotheses])
    for h in proper[cfg.articulated_hypotheses:]:
        if any(_matching_symmetry(h.fit.T @ np.linalg.inv(g.fit.T), part, cfg, rms_max=cfg.near_twin_rms_mm)
               for g in scored):
            scored.append(h)
    # poses the free-datum search added are articulated too (the best ``free_datum_rank_k`` of them)
    for h in [h for h in proper if h.free_extra and not any(h is g for g in scored)][:cfg.free_datum_rank_k]:
        scored.append(h)
    for h in scored:
        if budget.left() <= 0:
            break
        _articulate(h, pts, vox, part, model, cfg, budget)
    # before the scan is called a wrong part, every refined datum pose gets its articulated fit: the best few
    # by datum score can all be one wrong image of a datum that barely tells its poses apart
    if max(h.explained for h in scored) < cfg.wrong_part_explained_min:
        for h in proper:
            if h.art is None and budget.left() > 0:
                _articulate(h, pts, vox, part, model, cfg, budget)
    proper = sorted(proper, key=lambda h: -h.total)
    if any(h.free_extra for h in proper):
        proper = _free_datum_rank(proper, pts, vox, part, model, cfg, budget, note, spacing)
    best = proper[0]
    # e3-1.14 pose wells: the chosen pose refined again from tilted starts (a datum plane fit can stop in a
    # self-consistent tilted plane); a clearly better well replaces it, a near one is kept as an alternative pose
    wells: list[_Hyp] = []
    if cfg.pose_wells and best.art is not None and budget.left() > 0:
        wells = _pose_wells(best, reg_src, reg_spacing, pts, vox, part, model, cfg, budget, note)
        if wells and wells[0].total > best.total + cfg.pose_well_switch_margin:
            note(f"pose well replaces the chosen pose: total {best.total:.3f} -> {wells[0].total:.3f} "
                 f"(datum score {best.fit.score:.3f} -> {wells[0].fit.score:.3f})")
            old_best, best = best, wells[0]
            wells = sorted(wells[1:] + [old_best], key=lambda h: -h.total)
            proper = sorted(proper + [best], key=lambda h: -h.total)
            ctx.pose_switched = True
    note(f"best datum score={best.fit.score:.3f} explained={best.explained:.3f} total={best.total:.3f}"
         + (f" holes_with_hardware={best.fit.detail.get('holes_hardware')}" if best.fit.detail.get("holes_hardware")
            else ""))
    # -- mirrored part check -----------------------------------------------------------------
    suspicious = best.fit.hole_mismatch > 0 or best.explained < cfg.mirror_check_explained or \
        best.fit.score < cfg.mirror_check_score
    if suspicious and budget.left() > 0:
        step("mirror_check", 45)
        flip = np.array([-1.0, 1.0, 1.0])
        mvox = _mirror_voxels(vox)
        mirrored = _register(reg_src * flip, mvox, part, model, cfg, budget, note, reg_spacing, mirrored=True)
        mpts = pts * flip
        m_scored = mirrored[:cfg.articulated_hypotheses]
        m_scored += [h for h in mirrored[cfg.articulated_hypotheses:] if h.free_extra][:cfg.free_datum_rank_k]
        for h in m_scored:
            if budget.left() <= 0:
                break
            _articulate(h, mpts, mvox, part, model, cfg, budget)
        if mirrored and max(h.explained for h in mirrored[:cfg.articulated_hypotheses]) < cfg.wrong_part_explained_min:
            for h in mirrored:  # the same chance the proper poses had
                if h.art is None and budget.left() > 0:
                    _articulate(h, mpts, mvox, part, model, cfg, budget)
        mirrored.sort(key=lambda h: -h.total)
        if any(h.free_extra for h in mirrored):
            mirrored = _free_datum_rank(mirrored, mpts, mvox, part, model, cfg, budget, note, spacing)
        if mirrored:
            mb = mirrored[0]
            sym_maps: list[np.ndarray] = []
            note(f"mirrored best total={mb.total:.3f} holes_mismatch={mb.fit.hole_mismatch} vs proper {best.total:.3f}"
                 f"/{best.fit.hole_mismatch}")
            if mb.total > best.total + cfg.mirror_margin and mb.fit.hole_mismatch <= best.fit.hole_mismatch:
                # a CAD that is its own mirror image (a certified MIRROR self-map of a compiled stage) cannot tell
                # a mirrored part from a proper one: its mirrored fit is a proper pose the proper search missed.
                # That pose (the self-map composed with the mirrored fit) joins the proper hypotheses and competes
                # like any other; only a chiral CAD, or a converted pose that still explains clearly less, refuses
                # the symmetry that matters is that of the shape the scan shows: the stage the mirrored fit decoded
                # (a flat blank is often mirror-symmetric while the formed part is not)
                sym_maps = _mirror_self_maps(part, _decoded_stage_id(part, mb))
                conv = []
                for S in sym_maps:
                    if budget.left() <= 0:
                        break
                    Tp = S @ mb.fit.T @ MIRROR
                    h = _Hyp(fit=refine_datum(reg_src, Tp, model, cfg, reg_spacing), mirrored=False)
                    if any(_close(h.fit.T, o.fit.T, dist=min(1.0, 0.4 * part.t)) for o in proper + conv):
                        continue  # the proper search had it (or another self-map led to it)
                    _articulate(h, pts, vox, part, model, cfg, budget)
                    if h.art is not None:
                        conv.append(h)
                if conv:
                    proper = sorted(proper + conv, key=lambda h: -h.total)
                    best = proper[0]
                    note(f"mirror-symmetric CAD: mirrored fit converted to {len(conv)} proper pose(s); best proper "
                         f"total={best.total:.3f} explained={best.explained:.3f}")
            why = _mirror_refusal(mb.total, best.total, mb.fit.hole_mismatch, best.fit.hole_mismatch,
                                  symmetric=bool(sym_maps), margin=cfg.mirror_margin)
            if why is not None:
                return ctx.invalid("MIRRORED_PART_SUSPECTED", (why,), None, None, log, abstained=True)
    # the pose is chosen by comparing complete articulated fits (and, when the best looks suspicious,
    # the mirrored scan's): a budget that ran out on the way left that comparison unfinished, and a
    # cut articulation explains less of the scan, which must never read as a wrong part. Nothing could
    # be measured after the budget anyway: the registration did not finish (ENGINE_IO §4).
    if budget.left() <= 0 or best.art is None:
        return ctx.invalid("TIMEOUT", ("TIMEOUT_DURING_ARTICULATED_FIT",), None, None, log)
    # -- pose ambiguity (E1's symmetry test of the decoded stage is the authority) ----------------
    ambiguous, amb_states, margin, chosen = _ambiguity(
        proper, part, cfg, evidence=lambda h: _hole_evidence(h, pts, vox, part, cfg, spacing), note=note)
    if chosen is not best:
        note(f"symmetric rival chosen: total {chosen.total:.3f} vs {best.total:.3f}")
        best = chosen
    note(f"pose margin={margin:.4f} ambiguous={ambiguous}")
    # a datum whose cut edges and holes leave an in-plane direction free: the cut ends of flanges that
    # no bend moves along that direction fix it (the determined directions stay the datum's own)
    if not ambiguous and best.fit.undetermined.size and budget.left() > 0:
        refined = _refine_free_directions(best, pts, vox, part, model, cfg, budget, note, spacing)
        if refined is not None:
            best = refined
    # folds formed the wrong way are looked for on the chosen pose only (``fit_articulated``): on every candidate
    # pose the mirrored hinge search would let a datum turned over read its bends as formed the wrong way
    if not ambiguous and budget.left() > 0:
        rev = _Hyp(fit=best.fit, mirrored=best.mirrored)
        _articulate(rev, pts, vox, part, model, cfg, budget, reversed_search=True)
        if rev.art is not None and any(b.reversed for b in rev.art.bends) \
                and not any(b.reason == "TIMEOUT" for b in rev.art.bends):
            note("formed the wrong way: " + " ".join(part.bends[j].label for j, b in enumerate(rev.art.bends)
                                                      if b.reversed)
                 + f" (explained {best.explained:.3f} -> {rev.explained:.3f})")
            best = rev
    T = best.fit.T
    reg_pts = apply(T, pts)
    ctx.reg_pts = reg_pts
    # -- validity of the registration ----------------------------------------------------------------
    if best.fit.datum_coverage < cfg.min_datum_coverage:
        return ctx.invalid("UNTRUSTED_REGISTRATION", (f"DATUM_COVERAGE:{best.fit.datum_coverage:.3f}",), T, best,
                           log)
    if best.fit.hole_mismatch >= cfg.wrong_part_hole_mismatch_min:
        return ctx.invalid("WRONG_PART_SUSPECTED", (f"DATUM_HOLES_FILLED:{best.fit.hole_mismatch}",), T, best, log)
    if best.explained < cfg.wrong_part_explained_min:
        return ctx.invalid("WRONG_PART_SUSPECTED", (f"SCAN_EXPLAINED:{best.explained:.3f}",), T, best, log)
    # -- decode -------------------------------------------------------------------------------------
    step("decode", 55)
    decodes = best.decodes or []
    states = [d.state for d in decodes]
    chosen_states = list(states)
    if ambiguous:
        states = [s if s == "NOT_YET_FORMED" and all(o[j] == "NOT_YET_FORMED" for o in amb_states) else "UNKNOWN"
                  for j, s in enumerate(states)]
    stage_id, exact = choose_stage_model(part, states)
    if ambiguous:
        stage_id, exact = None, False
    formed_uids = [part.bends[j].uid for j, s in enumerate(states) if s in _FORMED_LIKE]
    if ambiguous:
        # nothing is measured and no identity is claimed, but the grey nominal lines must still lie on
        # the part: the chosen decode's stage (any twin puts its bends at the same places)
        display_id, _ = choose_stage_model(part, chosen_states)
        lines = part.stage_lines(display_id, [part.bends[j].uid for j, s in enumerate(chosen_states)
                                              if s in _FORMED_LIKE])
    else:
        lines = part.stage_lines(stage_id, formed_uids)
    transforms = part.stage_transforms(stage_id, formed_uids)
    in_plane_degenerate = bool(best.fit.degenerate) or _pose_sigma(best.fit) > cfg.degenerate_max_sigma_mm
    thick = _thickness(best.fit, part, cfg, warnings)
    note(f"thickness used {thick.t_mm:.4f} mm ({'measured on ' + str(best.fit.thickness_tiles) + ' datum tiles' if thick.measured else 'compiled'}"
         f", sigma {best.fit.thickness_sigma_mm}) u {thick.u_mm:.4f} frame {thick.frame_mm:.4f}; compiled {part.t:.4f}")
    # -- metrology ----------------------------------------------------------------------------------
    step("metrology", 65)
    idx = PointIndex(reg_pts, vox)
    geoms: dict[int, BendMeasure] = {}
    for j in range(part.n):
        if budget.left() <= 0:
            break
        if states[j] in _FORMED_LIKE and not ambiguous:
            geoms[j] = measure_bend_geometry(idx, part, best.art, j, cfg)
    # unplanned creases first: a crease ends the flange it splits (it is a bend, not the cut edge)
    unknown_any = any(s == "UNKNOWN" for s in states)
    found: list[_Crease] = []
    crease_search = not ambiguous and not unknown_any
    if crease_search and budget.left() > 0:
        found = _creases(part, best.art, vox, apply(T, vox.points), vox.normals @ T[:3, :3].T, reg_pts, cfg, thick)
    # a FREE_EDGE side is only measured once the crease search has looked at it
    ctx.free_edge_blocked = "TIMEOUT" if crease_search and budget.left() <= 0 else None
    creases: tuple[UnplannedCreaseV1, ...] = tuple(c.model for c in found)
    ctx.crease_sharps = {}
    for c in found:
        point = None if c.sharp is None else c.sharp[0]
        ctx.crease_sharps.setdefault(c.panel, []).append((point, c.direction, c.sharp_u, c.model.crease_id,
                                                          np.asarray(c.model.polyline_mm, dtype=np.float64)))
    # a datum whose edges and holes leave an in-plane direction free still fixes every bend-line
    # position measured across that direction: such positions are measured, the others withheld
    pos_trust = _position_trust(part, best.fit, model, states, geoms, lines, transforms, in_plane_degenerate, cfg)
    reg_degenerate = in_plane_degenerate and not any(ok for ok, _ in pos_trust.values())
    if in_plane_degenerate:
        note("in-plane pose degenerate: positions " + (" ".join(
            f"{part.bends[j].label}={'ok' if ok else 'withheld'}" for j, (ok, _) in sorted(pos_trust.items())) or "none"))
    outcomes = _outcomes(part, best, states, decodes, geoms, lines, transforms, stage_id, exact, reg_degenerate,
                         ambiguous, idx, cfg, budget, thick, ctx, pos_trust)
    # e3-1.13: one-thickness skin swaps and near-tie skin / window choices (``metrology._BranchRec``)
    swaps = [f"{part.bends[j].label}:{s}" for j, g in sorted(geoms.items()) if g is not None for s in g.skin_swaps]
    if swaps:
        note("one-skin side re-seated one thickness onto its outer skin: " + " ".join(swaps))
    if cfg.branch_withhold and not ambiguous:
        _branch_withhold(outcomes, part, best, states, decodes, geoms, lines, transforms, stage_id, exact,
                         reg_degenerate, ambiguous, idx, cfg, budget, thick, ctx, pos_trust)
    ctx.general = _general_measurements(part, recipe, states, geoms, best.art, cfg, budget)
    # e3-1.14: alternative poses (near wells of the datum fit) measured too; a decision they do not share is withheld
    alts = [h for h in wells if h is not best and h.art is not None and h.total >= best.total - cfg.pose_well_alt_margin
            and _pose_gap_mm(h.fit.T, best.fit.T, model) > cfg.pose_well_min_gap_mm]
    if alts and not ambiguous:
        _alternative_pose_withhold(outcomes, ctx.general[0], part, recipe, best, alts, pts, vox, cfg, budget, thick,
                                   ctx, note)
    # -- symmetric twins that tie (amendment ENGINE_IO-A2): publish only what every twin reads the same ---------
    identity_ambiguous = False
    if ambiguous and cfg.twin_agreement and best.twins and budget.left() > 0:
        merged = _twin_agreement(part, recipe, best, pts, vox, cfg, budget, thick, ctx, outcomes, note)
        if merged is not None:
            outcomes, ctx.general = merged
            states = [o.state for o in outcomes]
            identity_ambiguous = True
            warnings.append("IDENTITY_AMBIGUOUS_TWINS: certified symmetric twin poses tie; only readings identical "
                            "under every twin are published, everything else is UNKNOWN(AMBIGUOUS_POSE)")
    # -- heatmap, labels --------------------------------------------------------------------------------
    step("heatmap", 85)
    heat_reason = None
    if ambiguous:
        heat_reason = "AMBIGUOUS_POSE"
    elif in_plane_degenerate:
        heat_reason = "REGISTRATION_UNTRUSTED"
    elif stage_id is None or not exact:
        heat_reason = "STAGE_MODEL_UNDETERMINED"
    deviations = None
    surface_cov = None
    if heat_reason is None and budget.left() > 0:
        deviations, surface_cov = _heatmap(reg_pts, part, stage_id, cfg)
    elif heat_reason is None:
        heat_reason = "TIMEOUT"
    labels = _segment_labels(part, best.art, vox, reg_pts, geoms, found, ambiguous)
    registration = _registration_block(best, len(proper), margin, ambiguous, reg_degenerate, cfg,
                                       float(opts.unit_factor), identity_ambiguous=identity_ambiguous)
    coverage = CoverageV1(surface_coverage_ratio=None if surface_cov is None else float(min(1.0, surface_cov)),
                          datum_coverage_ratio=float(min(1.0, max(0.0, best.fit.datum_coverage))),
                          scan_point_count=int(len(pts)))
    detail = _stage_detail(part, states, stage_id, exact, ambiguous)
    if part.compiled.k_factor is not None and part.compiled.k_factor.source == "SHOP_DEFAULT":
        warnings.append("K_FACTOR_DEFAULTED: backgauge U includes sigma_K * |dN/dK|")
    lateral = (best.fit.detail or {}).get("flange_end_edges") or {}
    if lateral.get("resolved"):
        warnings.append("IN_PLANE_POSE_FROM_FLANGE_ENDS: the datum's cut edges and holes leave an in-plane direction "
                        f"free; {len(lateral.get('edges_used') or [])} cut flange ends that no bend moves along it "
                        "fix it")
    if reg_degenerate:
        warnings.append("IN_PLANE_POSE_DEGENERATE: bend-line position and heatmap withheld")
    elif in_plane_degenerate:
        withheld = sum(1 for ok, _ in pos_trust.values() if not ok)
        warnings.append("IN_PLANE_POSE_PARTIAL: the datum's cut edges and holes leave an in-plane direction "
                        f"undetermined; heatmap and {withheld} bend-line position(s) depending on it withheld")
    return ctx.finish("VALID", (), registration, coverage, outcomes, stage_id, detail, creases, deviations, labels,
                      heat_reason, log)


# ----------------------------------------------------------------------------------------------- helpers
@dataclass
class _Ctx:
    part: PartView
    recipe: object
    inputs: Inputs
    cfg: InspectConfig
    opts: InspectOptions
    out_dir: Path
    warnings: list
    note: Callable
    budget: _Budget
    scan: ScanData | None = None
    #: the ingested points (unit factor applied, gross outliers dropped): what every
    #: index-aligned output file describes
    points: np.ndarray | None = None
    spacing: float = 0.5
    reg_pts: np.ndarray | None = None
    #: the datum model of the registration (for the pose-bias bound of bend-line positions)
    model: DatumModel | None = None
    #: panel index -> (OML virtual sharp point or None when unresolved, direction, U component,
    #: crease id) of the unplanned creases hinged on it
    crease_sharps: dict = field(default_factory=dict)
    #: (measurements, anchors, not applicable ids) of the recipe's general measurements
    general: tuple = ({}, {}, set())
    #: UNKNOWN reason for every FREE_EDGE side when the unplanned-crease search could not run
    free_edge_blocked: str | None = None
    #: e3-1.13 safety net: the arguments of the deferred :meth:`finish`, and the primary's own readings
    finish_args: tuple | None = None
    primary_raw: dict = field(default_factory=dict)
    #: e3-1.13 dense ingest: whether the primary's points were thinned (its resamples inherit it)
    dense_thinned: bool = False
    #: e3-1.14 pose wells: the chosen pose was replaced by a better well; alternative-pose withholding record
    pose_switched: bool = False
    pose_alt: dict = field(default_factory=dict)

    def _engine(self) -> EngineInfoV1:
        return EngineInfoV1(engine_version=ENGINE_VERSION, config_sha256=self.cfg.sha256(),
                            bundle_id=self.opts.engine_bundle_id,
                            compiled_part_sha256=self.part.compiled.compiled_part_sha256,
                            algorithm_versions=dict(ALGORITHM_VERSIONS),
                            duration_s=round6(self.budget.elapsed()), peak_rss_bytes=_peak_rss())

    def invalid(self, state: str, reasons: tuple[str, ...], T: np.ndarray | None, hyp: _Hyp | None, log: list,
                abstained: bool = False) -> InspectOutcome:
        self.note(f"validity {state}: {', '.join(reasons)}")
        publish = T is not None and not abstained and float(np.linalg.det(T[:3, :3])) > 0
        reg = RegistrationV1(method="datum_plane_outline_edges_holes", raw_to_frame=_mat(T) if publish else None,
                             unit_factor=float(self.opts.unit_factor), abstained=not publish,
                             datum_fitness=None if hyp is None else float(min(1.0, max(0.0, hyp.fit.datum_coverage))),
                             pose_candidates_considered=None)
        n = len(self.points) if self.points is not None else (len(self.scan.points) if self.scan is not None else 0)
        cov = CoverageV1(scan_point_count=n)
        outcomes = [BendOutcome(state="UNKNOWN", state_reasons=("INVALID_INSPECTION",)) for _ in self.part.bends]
        if publish and self.reg_pts is None and self.points is not None:
            self.reg_pts = apply(T, self.points)
        return self.finish(state, reasons, reg, cov, outcomes, None, f"not evaluated: {state}", (), None, None,
                           f"VALIDITY_{state}", log, publish_scan=publish)

    def finish(self, state, reasons, registration, coverage, outcomes, stage_id, detail, creases, deviations, labels,
               heat_reason, log, publish_scan: bool = True) -> InspectOutcome:
        if self.opts.defer_finish:
            self.finish_args = ((state, reasons, registration, coverage, outcomes, stage_id, detail, creases,
                                 deviations, labels, heat_reason, log), {"publish_scan": publish_scan})
        files: dict[str, str] = {}
        out = self.out_dir
        write = self.opts.write_outputs and not self.opts.defer_finish
        part = self.part
        reg_rel = dev_rel = lab_rel = stage_rel = final_rel = None
        npts = None
        if publish_scan and self.reg_pts is not None and registration.raw_to_frame is not None:
            reg_rel = ENGINE_OUTPUT_FILES["registered_scan"]
            npts = int(len(self.reg_pts))
            if write:
                write_points_ply(out / reg_rel, self.reg_pts)
            if deviations is not None and state == "VALID":
                dev_rel = ENGINE_OUTPUT_FILES["deviations"]
                if write:
                    write_array(out / dev_rel, deviations, "<f4")
            if labels is not None:
                lab_rel = ENGINE_OUTPUT_FILES["segment_labels"]
                if write:
                    write_array(out / lab_rel, labels, "<u2")
        if stage_id is not None and state == "VALID":
            stage_rel = ENGINE_OUTPUT_FILES["stage_mesh"]
            if write:
                copy_file(part.mesh_path(stage_id), out / stage_rel)
        final_rel = ENGINE_OUTPUT_FILES["final_mesh"]
        if write:
            copy_file(part.mesh_path(f"S{part.n}"), out / final_rel)
        viewer = ViewerAssetsV1(
            registered_scan=reg_rel, point_count=npts, deviations=dev_rel,
            deviation_reference_stage=stage_id if dev_rel is not None else None,
            heatmap_band_mm=float(self.recipe.default_tolerances.surface_profile_mm),
            heatmap_unavailable_reason=None if dev_rel is not None else (heat_reason or "UNAVAILABLE"),
            segment_labels=lab_rel, stage_mesh=stage_rel, final_mesh=final_rel,
            bend_geometry=ENGINE_OUTPUT_FILES["bend_geometry"],
        )
        scan = self.scan
        inp = ResultInputs(
            part=part, recipe=self.recipe, scan_sha256=self.inputs.scan_sha256,
            scan_format=scan.format if scan is not None else _sniff_format(self.inputs.scan_path),
            vertex_count=scan.vertex_count if scan is not None else 0,
            face_count=scan.face_count if scan is not None else 0,
            context=self.opts.context, engine=self._engine(), validity_state=state, validity_reasons=tuple(reasons),
            registration=registration, coverage=coverage, declared_stage=self.opts.declared_stage,
            declared_source=self.opts.declared_source, stage_model_id=stage_id, stage_detail=detail,
            bends=outcomes, viewer=viewer, creases=tuple(creases), warnings=tuple(self.warnings),
            created_at=self.opts.created_at, general_measurements=self.general[0], general_anchors=self.general[1],
            general_not_applicable=self.general[2],
        )
        result = build_result(inp)
        problems = check_result_against_compiled(result, part.compiled, self.recipe)
        if problems:
            raise InspectInternalError("result disagrees with the compiled part/recipe: " + ", ".join(problems))
        if write:
            geo = BendGeometryFileV1.from_result(result)
            write_json(out / ENGINE_OUTPUT_FILES["bend_geometry"], geo)
            (out / ENGINE_OUTPUT_FILES["log"]).write_text("\n".join(log) + "\n", encoding="utf-8")
            write_json(out / ENGINE_OUTPUT_FILES["result"], result)
        for key, rel in (("registered_scan", reg_rel), ("deviations", dev_rel), ("segment_labels", lab_rel),
                         ("stage_mesh", stage_rel), ("final_mesh", final_rel)):
            if rel is not None:
                files[key] = rel
        files["bend_geometry"] = ENGINE_OUTPUT_FILES["bend_geometry"]
        files["result"] = ENGINE_OUTPUT_FILES["result"]
        res = InspectOutcome(result=result, files=files, log=log)
        res._ctx = self  # e3-1.13 safety net: the deferred finish is redone on this context
        return res


def _sniff_format(path: Path) -> str:
    head = Path(path).read_bytes()[:512]
    if head.startswith(b"ply"):
        if b"binary_big_endian" in head:
            return "PLY_BINARY_BE"
        if b"ascii" in head:
            return "PLY_ASCII"
        return "PLY_BINARY_LE"
    if head.lstrip().lower().startswith(b"solid"):
        return "STL_ASCII"
    return "STL_BINARY"


def _mat(T: np.ndarray) -> tuple:
    m = np.asarray(T, dtype=np.float64)
    rows = [tuple(round6(float(v)) for v in m[i]) for i in range(3)]
    return tuple(rows) + ((0.0, 0.0, 0.0, 1.0),)


def _density_guard(pts: np.ndarray, cfg: InspectConfig) -> str | None:
    """e3-1.13 density guard: the ``DENSITY_ABOVE_VALIDATED_RANGE`` reason when the cleaned points lie above the
    validated native envelope (``density_guard_ppc`` points per occupied ``density_guard_cell_mm`` cell, or
    ``density_guard_max_points`` points), else None. Cheap (one integer key per point); never changes the points."""
    n = len(pts)
    if cfg.density_guard_max_points and n > cfg.density_guard_max_points:
        return f"DENSITY_ABOVE_VALIDATED_RANGE:points={n}>{int(cfg.density_guard_max_points)}"
    if cfg.density_guard_ppc <= 0 or n == 0:
        return None
    q = np.floor((pts - pts.min(axis=0)) / float(cfg.density_guard_cell_mm)).astype(np.int64)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    ppc = float(n) / float(len(np.unique(key)))
    if ppc > cfg.density_guard_ppc:
        return (f"DENSITY_ABOVE_VALIDATED_RANGE:points_per_{cfg.density_guard_cell_mm:g}mm_cell={ppc:.2f}"
                f">{cfg.density_guard_ppc:g}")
    return None


def _decimate(pts: np.ndarray, cell: float) -> np.ndarray:
    """One point (the first in file order) per ``cell`` voxel: deterministic thinning of dense exports."""
    q = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, first = np.unique(key, return_index=True)
    return pts[np.sort(first)]


def _cell_keys(pts: np.ndarray, cell: float) -> np.ndarray:
    q = np.floor((pts - pts.min(axis=0)) / cell).astype(np.int64)
    dims = q.max(axis=0) + 1
    return (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]


def _cell_density(pts: np.ndarray, cell: float) -> float:
    """Points per occupied ``cell`` voxel (grid anchored at the cloud's minimum corner)."""
    if len(pts) == 0:
        return 0.0
    return float(len(pts)) / float(len(np.unique(_cell_keys(pts, cell))))


def _dense_thin(pts: np.ndarray, cell: float) -> np.ndarray:
    """One actual point per ``cell`` voxel: the one nearest the cell's mean (ties by coordinates, then file order);
    the kept points stay in file order (deterministic)."""
    key = _cell_keys(pts, cell)
    order = np.argsort(key, kind="stable")  # file order within a cell
    ks = key[order]
    del key
    starts = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
    del ks
    counts = np.diff(np.r_[starts, len(order)])
    grp = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
    d2 = np.zeros(len(order))
    for ax in range(3):
        c = pts[order, ax]
        m = np.add.reduceat(c, starts) / counts
        d2 += np.square(c - m[grp])
    best = np.minimum.reduceat(d2, starts)
    tie = d2 == best[grp]
    # the first tied point of each cell in file order; cells with several tied points are resolved by coordinates
    cand = np.nonzero(tie)[0]
    gc = grp[cand]
    pick = cand[np.r_[True, gc[1:] != gc[:-1]]]
    multi = np.nonzero(np.bincount(gc, minlength=len(starts)) > 1)[0]
    if len(multi):
        sel = {}
        for i in cand[np.isin(gc, multi)]:
            g = int(grp[i])
            p = tuple(pts[order[i]])
            if g not in sel or p < sel[g][0]:
                sel[g] = (p, i)
        pick = pick.copy()
        pos = np.searchsorted(grp[pick], np.asarray(sorted(sel)))
        pick[pos] = [sel[g][1] for g in sorted(sel)]
    return pts[np.sort(order[pick])]


def _pose_gap_mm(a: np.ndarray, b: np.ndarray, model: DatumModel) -> float:
    """Largest displacement (mm) between two poses over the datum outline's vertices (in the CAD frame)."""
    poly = np.asarray(model.polygon, dtype=np.float64)
    q = model.p0[None, :] + (poly[:, :1] - float(model.to2d(model.p0[None])[0][0])) * model.f1[None, :] \
        + (poly[:, 1:2] - float(model.to2d(model.p0[None])[0][1])) * model.f2[None, :]
    rel = a @ np.linalg.inv(b)
    return float(np.linalg.norm(q @ rel[:3, :3].T + rel[:3, 3] - q, axis=1).max())


def _tilt_about(T: np.ndarray, centre: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    R = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    return rigid(R, centre - R @ centre) @ T


def _pose_wells(best: "_Hyp", reg_src, reg_spacing: float, pts, vox, part: PartView, model: DatumModel,
                cfg: InspectConfig, budget: "_Budget", note: Callable[[str], None]) -> list["_Hyp"]:
    """e3-1.14 ``pose_wells``: other optima of the datum fit near the chosen pose, articulated, best total first.

    The chosen pose is refined again (:func:`refine_datum`, the same fit every hypothesis gets) from tilted
    starts: about each in-plane datum axis through the outline's centroid, by the angles that move the datum's
    far edge (its half-extent across the axis) by +-``pose_well_edge_mm`` and +-2x that, at most
    ``pose_well_max_deg``. Optima within ``pose_well_min_gap_mm`` of the chosen pose or of each other are one
    well. Wells whose datum score is not below the chosen one's by more than ``pose_well_score_slack`` get the
    articulated fit, best datum score first, at most ``pose_well_articulate_max``. Geometry only: no part logic."""
    t_start = budget.elapsed()
    poly = np.asarray(model.polygon, dtype=np.float64)
    cen2 = poly.mean(axis=0)
    o2 = model.to2d(model.p0[None])[0]
    centre = model.p0 + model.f1 * float(cen2[0] - o2[0]) + model.f2 * float(cen2[1] - o2[1])
    half = 0.5 * (poly.max(axis=0) - poly.min(axis=0))  # half-extents along f1, f2
    starts = []
    for axis, reach in ((model.f1, float(half[1])), (model.f2, float(half[0]))):
        if reach <= 1e-6:
            continue
        for k in (1.0, -1.0, 2.0, -2.0):
            deg = math.degrees(math.atan(k * cfg.pose_well_edge_mm / reach))
            deg = max(-cfg.pose_well_max_deg, min(cfg.pose_well_max_deg, deg))
            starts.append(_tilt_about(best.fit.T, centre, axis, deg))
    fits: list[InPlaneFit] = []
    for T0 in starts:
        if budget.left() <= 0:
            break
        f = refine_datum(reg_src, T0, model, cfg, reg_spacing)
        if _pose_gap_mm(f.T, best.fit.T, model) <= cfg.pose_well_min_gap_mm:
            continue
        if any(_pose_gap_mm(f.T, o.T, model) <= cfg.pose_well_min_gap_mm for o in fits):
            continue
        fits.append(f)
    fits.sort(key=lambda f: -f.score)
    cand = [f for f in fits if f.score >= best.fit.score - cfg.pose_well_score_slack][:cfg.pose_well_articulate_max]
    out: list[_Hyp] = []
    for f in cand:
        if budget.left() <= 0:
            break
        h = _Hyp(fit=f, mirrored=best.mirrored)
        _articulate(h, pts, vox, part, model, cfg, budget)
        if h.art is not None:
            out.append(h)
    out.sort(key=lambda h: -h.total)
    note(f"pose wells: {len(starts)} tilted starts, {len(fits)} other well(s), {len(out)} articulated: "
         + " ".join(f"score {h.fit.score:.3f} total {h.total:.3f} gap {_pose_gap_mm(h.fit.T, best.fit.T, model):.2f}mm"
                    for h in out) + f" (chosen score {best.fit.score:.3f} total {best.total:.3f}; {budget.elapsed() - t_start:.1f}s)")
    return out


def _alternative_pose_withhold(outcomes: list[BendOutcome], general: dict, part: PartView, recipe, best: "_Hyp",
                               alts: list["_Hyp"], pts, vox, cfg: InspectConfig, budget: "_Budget", thick,
                               ctx: "_Ctx", note: Callable[[str], None]) -> None:
    """e3-1.14: the scan measured under each alternative pose (a near datum-fit well, ``pose_well_alt_margin``).

    Each alternative is decoded and measured as the chosen pose is (no crease search, no wrong-way search, no
    free-direction refinement). A decided characteristic (PASS or FAIL) of the chosen pose that an alternative
    reads with another decision (REVIEW included) is withheld: REVIEW, with the difference added to its U. One
    that the alternative cannot measure, or decodes as another bend state, is withheld too (the pose that decides
    it is not established). Nothing becomes PASS: U only grows."""
    model = ctx.model
    t_start = budget.elapsed()
    withheld: dict[str, list] = {}
    saved = (ctx.free_edge_blocked, ctx.crease_sharps)
    for h in alts:
        if budget.left() <= 0:
            ctx.warnings.append("POSE_ALTERNATIVE_CHECK_INCOMPLETE: decided characteristics withheld (REVIEW)")
            h_out, h_gen = None, None
        else:
            decodes = h.decodes or []
            states = [d.state for d in decodes]
            stage_id, exact = choose_stage_model(part, states)
            formed = [part.bends[j].uid for j, st in enumerate(states) if st in _FORMED_LIKE]
            lines = part.stage_lines(stage_id, formed)
            transforms = part.stage_transforms(stage_id, formed)
            idx = PointIndex(apply(h.fit.T, pts), vox)
            geoms = {j: measure_bend_geometry(idx, part, h.art, j, cfg) for j, st in enumerate(states)
                     if st in _FORMED_LIKE and budget.left() > 0}
            deg = bool(h.fit.degenerate) or _pose_sigma(h.fit) > cfg.degenerate_max_sigma_mm
            pos_trust = _position_trust(part, h.fit, model, states, geoms, lines, transforms, deg, cfg)
            reg_deg = deg and not any(ok for ok, _ in pos_trust.values())
            h_out = _outcomes(part, h, states, decodes, geoms, lines, transforms, stage_id, exact, reg_deg, False, idx,
                              cfg, budget, thick, ctx, pos_trust)
            h_gen = _general_measurements(part, recipe, states, geoms, h.art, cfg, budget)[0]
            ctx.free_edge_blocked, ctx.crease_sharps = saved
            if budget.left() <= 0:
                ctx.warnings.append("POSE_ALTERNATIVE_CHECK_INCOMPLETE: decided characteristics withheld (REVIEW)")
                h_out, h_gen = None, None
        items = [(cid, m, o, None if h_out is None else h_out[j]) for j, o in enumerate(outcomes)
                 for cid, m in o.measurements.items()] + [(cid, m, None, None) for cid, m in general.items()]
        for cid, m, o, oa in items:
            if m.value is None or m.u is None or (o is not None and (cid in o.forced_unknown or cid in o.stage_na)):
                continue
            if o is not None and o.state in ("UNKNOWN", "NOT_YET_FORMED"):
                continue
            nom = part.nominals.get(cid)
            d0 = _gb(nom, m.value, m.u)
            if d0 not in ("PASS", "FAIL"):
                continue
            if h_out is None:
                ma = None
            elif o is None:
                ma = h_gen.get(cid)
            else:
                ma = None if (oa is None or oa.state != o.state or cid in oa.forced_unknown) else oa.measurements.get(cid)
            if ma is not None and ma.value is not None and ma.u is not None and _gb(nom, ma.value, ma.u) == d0:
                continue
            delta = 0.0 if ma is None or ma.value is None else abs(float(ma.value) - float(m.value))
            u0 = float(m.u)
            m.u = withhold_u(nom, m.value, u0, delta)
            m.notes = tuple(m.notes) + ("POSE_ALTERNATIVE_WITHHELD",)
            withheld[cid] = [d0, None if ma is None or ma.value is None else round(float(ma.value), 6), round(u0, 6)]
    ctx.pose_alt = {"alternatives": [{"total": round(h.total, 4), "score": round(h.fit.score, 4)} for h in alts],
                    "withheld": withheld}
    note(f"alternative poses: {len(alts)} within {cfg.pose_well_alt_margin} of total {best.total:.3f} "
         + " ".join(f"[{h.total:.3f}]" for h in alts) + f" ({budget.elapsed() - t_start:.1f}s); withheld "
         + (" ".join(withheld) if withheld else "none"))


def _mirror_voxels(vox: VoxelCloud) -> VoxelCloud:
    flip = np.array([-1.0, 1.0, 1.0])
    out = VoxelCloud(points=vox.points * flip, inverse=vox.inverse, counts=vox.counts, voxel=vox.voxel,
                     normals=None if vox.normals is None else vox.normals * flip, curvature=vox.curvature,
                     spread=vox.spread, slab=vox.slab, tree=None, graph=vox.graph)
    out.tree = cKDTree(out.points, balanced_tree=False, compact_nodes=False)
    return out


def _register(pts, vox, part, model, cfg, budget, note, spacing: float, *, mirrored: bool) -> list[_Hyp]:
    hyps, segs = coarse_hypotheses(vox, part, model, cfg, mirrored=False, budget_left=budget.left)
    # poses from pairs of the datum's round holes found in the scan join the coarse peaks: the coarse
    # raster cannot see them, and on a datum whose outline barely fixes the pose they are what does
    holes = hole_pair_hypotheses(pts, segs, part, model, cfg, spacing, mirrored=False) if budget.left() > 0 else []
    note(f"{'mirrored ' if mirrored else ''}planes={len(segs)} coarse hypotheses={len(hyps)} hole-pair hypotheses="
         f"{len(holes)}")
    # every coarse peak gets the fine (hole-weighted) search; its score ranks them for the
    # expensive edge/hole fit (the coarse raster cannot see small holes)
    searched = []
    edge = math.radians(max(0.0, cfg.fine_search_deg - 0.25))
    for h in list(hyps[: cfg.coarse_candidates_fine]) + holes:
        if budget.left() <= 0:
            break
        Tf, sc = fine_search(pts, h.T, model, cfg, with_score=True, walks=0)
        # the two skin heights of one in-plane pose are one thickness apart: keep both
        if any(_close(Tf, o[0], ang=1.5, dist=min(1.5, 0.4 * part.t)) for o in searched):
            continue
        searched.append((Tf, sc, _turn(Tf, h.T) >= edge))
    searched.sort(key=lambda r: -r[1])
    # the best few that stopped on the edge of their angle window walk on (``fine_search_walks``): a coarse peak
    # several degrees off, on a datum whose outline barely fixes the rotation, would otherwise stay there
    walked = []
    n_walk = 0
    for Tf, sc, at_edge in searched:
        if at_edge and n_walk < cfg.fine_walk_candidates and cfg.fine_search_walks > 0 and budget.left() > 0:
            n_walk += 1
            Tw, sw = fine_search(pts, Tf, model, cfg, with_score=True, walks=cfg.fine_search_walks - 1)
            if sw > sc:
                Tf, sc = Tw, sw
        if any(_close(Tf, o[0], ang=1.5, dist=min(1.5, 0.4 * part.t)) for o in walked):
            continue
        walked.append((Tf, sc))
    searched = sorted(walked, key=lambda r: -r[1])
    # the refinement slots go to distinct poses: candidates that the fine search left a degree or two and a
    # millimetre or two apart converge onto one pose, and six of them could crowd out every other one (a strip
    # datum's face-down image outscoring its true pose on the raster); one candidate per cluster is refined, in
    # score order, until ``pose_hypotheses`` distinct poses are found
    reps: list[np.ndarray] = []
    for Tf, _ in searched:
        if not any(_same_cluster(Tf, o, model, part.t) for o in reps):
            reps.append(Tf)
    fits: list[InPlaneFit] = []
    for Tf in reps:
        if budget.left() <= 0 or len(fits) >= cfg.pose_hypotheses:
            break
        fit = refine_datum(pts, Tf, model, cfg, spacing)
        if any(_close(fit.T, o.T, dist=min(1.0, 0.4 * part.t)) for o in fits):
            continue
        fits.append(fit)
    fits.sort(key=lambda f: -f.score)
    out = [_Hyp(fit=f, mirrored=mirrored) for f in fits]
    trigger = _free_datum_trigger(model, fits, cfg)
    if trigger is not None and budget.left() > 0:
        extra = _free_datum_poses(pts, vox, part, model, cfg, budget, spacing, fits, trigger)
        note(f"{'mirrored ' if mirrored else ''}free datum ({trigger}): {len(extra)} more pose(s) from plane seeds "
             "and symmetric images")
        out = sorted(out + [_Hyp(fit=f, mirrored=mirrored, free_extra=True) for f in extra], key=lambda h: -h.fit.score)
    return out


def _free_datum_trigger(model: DatumModel, fits: list[InPlaneFit], cfg: InspectConfig) -> str | None:
    """Why the in-plane pose must come from the flanges for every hypothesis, else None.

    NO_IN_PLANE_CONSTRAINT: the best refined fit leaves all three in-plane directions undetermined (a
    datum with no usable cut edge and no hole). SYMMETRIC_DATUM: the datum's material maps onto itself
    under an in-plane rotation (:meth:`DatumModel.rotational_symmetries`) and the best fit leaves some
    in-plane direction undetermined. A datum whose own edges fix the pose never triggers it."""
    if cfg.free_datum_rank_k <= 0 or not fits:
        return None
    k = int(fits[0].undetermined.shape[0])
    if k >= 3:
        return "NO_IN_PLANE_CONSTRAINT"
    if k and model.rotational_symmetries()[0]:
        return "SYMMETRIC_DATUM"
    return None


def _free_datum_poses(pts, vox, part: PartView, model: DatumModel, cfg: InspectConfig, budget: "_Budget",
                      spacing: float, fits: list[InPlaneFit], trigger: str) -> list[InPlaneFit]:
    """More refined datum poses when the datum does not fix its own in-plane pose (``_free_datum_trigger``).

    - NO_IN_PLANE_CONSTRAINT: the best raster peak of every candidate plane that no global peak represents
      (``coarse_hypotheses(plane_seeds=True)``, at most ``free_datum_seeds``). Such a datum's outline
      correlates with the formed part's large planes as well as with its own plane, so the global peaks
      can all land on those planes and the datum's own plane never gets a fine search.
    - both triggers: the datum's symmetric images of the best ``free_datum_rank_k`` fits (and seeds). The
      images tie on everything the datum shows; only the flanges tell them apart.
    Each pose gets the same fine search and refinement as any coarse peak; repeats of a known pose are dropped.
    """
    new: list[InPlaneFit] = []

    def add(T: np.ndarray) -> None:
        if budget.left() <= 0:
            return
        Tf = fine_search(pts, T, model, cfg, walks=0)
        f = refine_datum(pts, Tf, model, cfg, spacing)
        if not any(_close(f.T, o.T, dist=min(1.0, 0.4 * part.t)) for o in fits + new):
            new.append(f)

    if trigger == "NO_IN_PLANE_CONSTRAINT" and cfg.free_datum_seeds > 0:
        seeds, _ = coarse_hypotheses(vox, part, model, cfg, mirrored=False, budget_left=budget.left, plane_seeds=True)
        for hp in seeds[:cfg.free_datum_seeds]:
            add(hp.T)
    base = sorted(fits + new, key=lambda f: -f.score)[:cfg.free_datum_rank_k]
    for f in base:
        for Ti in model.symmetric_images(f.T):
            add(Ti)
    return new


def _free_datum_rank(hyps: list[_Hyp], pts, vox, part: PartView, model: DatumModel, cfg: InspectConfig,
                     budget: "_Budget", note: Callable[[str], None], spacing: float) -> list[_Hyp]:
    """Per-hypothesis flange-ends ranking for a free datum (``_free_datum_trigger``; bounded by ``free_datum_rank_k``).

    The best ``free_datum_rank_k`` articulated hypotheses (by total) whose fit leaves an in-plane direction
    undetermined get the flange-ends fit (:func:`_refine_free_directions`, the step the chosen pose gets),
    run on that hypothesis' own articulated fit. A refined pose replaces its hypothesis only when its total
    is higher; otherwise the hypothesis stays as it was. The list is returned sorted by total."""
    out = list(hyps)
    top = sorted((h for h in out if h.art is not None and h.fit.undetermined.size), key=lambda h: -h.total)
    for h in top[:cfg.free_datum_rank_k]:
        if budget.left() <= 0:
            break
        r = _refine_free_directions(h, pts, vox, part, model, cfg, budget, lambda m: None, spacing)
        if r is not None and r is not h and r.total > h.total:
            note(f"free datum: flange ends move a hypothesis, total {h.total:.3f} -> {r.total:.3f}")
            out[out.index(h)] = r
    return sorted(out, key=lambda h: -h.total)


def _same_cluster(a: np.ndarray, b: np.ndarray, model: DatumModel, t: float, ang_deg: float = 3.0,
                  in_plane_mm: float = 2.0) -> bool:
    """Two candidate poses the refinement would bring together: within ``ang_deg``, their frames within
    ``in_plane_mm`` in the datum plane and on the same skin height (``0.4 t``) at the datum's reference point."""
    if math.degrees(_turn(a, b)) > ang_deg:
        return False
    q = np.asarray(model.p0, dtype=np.float64)
    d = apply(a @ np.linalg.inv(b), q[None])[0] - q
    dn = float(d @ model.n0)
    return abs(dn) < 0.4 * t and float(np.linalg.norm(d - dn * model.n0)) < in_plane_mm


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle (rad) between two poses."""
    r = a[:3, :3] @ b[:3, :3].T
    return math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))


def _close(a: np.ndarray, b: np.ndarray, ang: float = 1.0, dist: float = 1.0) -> bool:
    r = a[:3, :3] @ b[:3, :3].T
    c = (float(np.trace(r)) - 1.0) / 2.0
    if math.degrees(math.acos(max(-1.0, min(1.0, c)))) > ang:
        return False
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) < dist


def _articulate(h: _Hyp, pts, vox, part: PartView, model: DatumModel, cfg: InspectConfig, budget: _Budget,
                reversed_search: bool = False) -> None:
    T = h.fit.T
    full_reg = apply(T, pts)
    vox_reg = apply(T, vox.points)
    vox_nrm = vox.normals @ T[:3, :3].T
    n0 = model.n0
    d0 = float(n0 @ model.p0)
    skins = (0, 1) if (h.fit.two_skins or h.fit.thickness_mm is not None) else \
        ((0,) if 0 in h.fit.skin_heights else (1,))
    # every single-skin offset uses the sheet's thickness measured on the datum when both skins were seen
    t_sheet = float(h.fit.thickness_mm) if h.fit.thickness_mm is not None else part.t
    art = fit_articulated(full_reg, vox, vox_reg, vox_nrm, part, n0, d0, t_sheet, skins, cfg, budget.left,
                          reversed_search=reversed_search)
    evidence = []
    for j in range(part.n):
        bf = art.bends[j]
        child = art.panels[part.bends[j].child]
        parent = art.panels[part.bends[j].parent]
        if not bf.found or child.state == "UNKNOWN":
            evidence.append((None, bf.reason or child.reason or "OCCLUDED"))
            continue
        if bf.flat:
            evidence.append((abs(bf.theta_deg or 0.0), None))
            continue
        skin = part.bends[j].oml_skin
        try:
            pp, pc = parent.skin_plane(skin), child.skin_plane(skin)
            if bf.frame is not None:
                # signed by the side the child lies on: negative when it was formed the wrong way
                sweep = signed_sweep_deg(pp.n, pc.n, bf.frame.a, cfg.reversed_wrap_margin_deg)
            else:
                sweep = math.degrees(math.acos(max(-1.0, min(1.0, float(pp.n @ pc.n)))))
        except (AssertionError, TypeError):
            sweep = abs(bf.theta_deg or 0.0)
        evidence.append((sweep, None))
    h.art = art
    h.decodes = decode_states(part, evidence, cfg)
    found = sum(1 for b in art.bends if b.found)
    h.explained, h.explained_tight = _model_explained(part, art, h.decodes, vox_reg, cfg)
    # the tight share separates the two skin heights of a datum seen on one side only: a wrong
    # height leaves its bend cylinders, cut walls and hole walls one thickness off the model
    h.total = float(h.fit.score + 0.5 * h.explained + cfg.explained_tight_weight * h.explained_tight
                    + 0.1 * found / max(part.n, 1))


def _refine_free_directions(best: _Hyp, pts, vox, part: PartView, model: DatumModel, cfg: InspectConfig,
                            budget: _Budget, note: Callable[[str], None], spacing: float) -> _Hyp | None:
    """The chosen pose with its free in-plane directions fixed by the cut ends of fitted flanges, else None.

    Rows come from :func:`~domains.production.engine.inspect.registration.child_edge_rows`
    on the chosen hypothesis' articulated fit (its model points block an edge
    whose band holds other material), the solution from
    :func:`~domains.production.engine.inspect.registration.resolve_free_directions`.
    The refined pose is articulated again and observed once more (the second
    solution, a few hundredths at most, is applied when above 0.02 mm). It is
    refused, and the free direction stays withheld, when the refinement changes
    a decoded bend state, the articulated model explains less of the scan
    (``explained`` drops by more than 0.02), the budget runs out or the second
    observation no longer resolves the direction.
    """
    from domains.production.engine.inspect.registration import child_edge_rows, resolve_free_directions

    base = best.fit
    h = best
    for it in range(2):
        if h.art is None or budget.left() <= 0:
            return None
        reg = apply(h.fit.T, pts)
        owner = h.art.owner[vox.inverse]
        blockers = _articulated_model(part, h.art, h.decodes or [])
        blockers = blockers[np.isfinite(blockers).all(axis=1)]
        rows = child_edge_rows(reg, owner, part, model, h.art.panels, cfg, spacing, blockers=blockers)
        fit, info = resolve_free_directions(base, h.fit.T, rows, model, cfg)
        if fit is None:
            note(f"free in-plane direction kept: {info.get('reason')}; flange ends offered "
                 f"{info.get('edges_offered')} rejected {info.get('rejected')}")
            return None
        shift = float(np.linalg.norm(info["shift"]))
        note(f"free in-plane direction fixed by flange ends (pass {it + 1}): shift "
             f"{[round(v, 4) for v in info['shift']]} edges {info['edges_used']} rejected {info['rejected']} "
             f"information {[round(v, 1) for v in info['information']]} sigma {info['sigma_obs']:.3f} "
             f"scatter {info['edge_scatter']:.3f}")
        if it == 1 and shift <= 0.02:
            return h
        nh = _Hyp(fit=fit, mirrored=best.mirrored)
        _articulate(nh, pts, vox, part, model, cfg, budget)
        if nh.art is None or budget.left() <= 0:
            return None
        if [d.state for d in (nh.decodes or [])] != [d.state for d in (best.decodes or [])]:
            note("free-direction refinement changed a decoded bend state: refused")
            return None
        if nh.explained < best.explained - 0.02:
            note(f"free-direction refinement explains less of the scan ({nh.explained:.3f} < {best.explained:.3f}): "
                 "refused")
            return None
        h = nh
    return h


def _thickness(fit: InPlaneFit, part: PartView, cfg: InspectConfig, warnings: list[str]) -> Thickness:
    """The thickness every single-skin offset uses (the datum's measured one, else the compiled one) and its U.

    Measured: ``u = hypot(u_thickness_fraction t, sigma_t)``. Compiled: the stock
    tolerance ``u_thickness_unmeasured_fraction t`` joins the level term; when the
    datum showed only its skin 1 the frame itself was placed ``t`` below skin 0,
    so its height carries that tolerance too (``Thickness.frame_mm``).
    """
    t_cad = float(part.t)
    if fit.thickness_mm is not None:
        t = float(fit.thickness_mm)
        if abs(t - t_cad) > cfg.thickness_warn_fraction * t_cad:
            warnings.append(f"THICKNESS_DIFFERS_FROM_CAD:measured={t:.3f}mm:compiled={t_cad:.3f}mm")
        return Thickness(t_mm=t, u_mm=math.hypot(cfg.u_thickness_fraction * t, float(fit.thickness_sigma_mm or 0.0)),
                         measured=True)
    stock = cfg.u_thickness_unmeasured_fraction * t_cad
    frame = stock if (1 in fit.skin_heights and 0 not in fit.skin_heights) else 0.0
    return Thickness(t_mm=t_cad, u_mm=math.hypot(cfg.u_thickness_fraction * t_cad, stock), measured=False,
                     frame_mm=frame)


def _ambiguity(hyps: list[_Hyp], part: PartView, cfg: InspectConfig,
               evidence: Callable[[_Hyp], tuple[int, int]] | None = None, note: Callable[[str], None] | None = None
               ) -> tuple[bool, list[list[str]], float, _Hyp]:
    """(ambiguous, states of the competing hypotheses, margin, chosen hypothesis).

    Rivals whose relative motion matches a ROTATION candidate of some compiled
    stage's symmetry test (twins) and whose total score or explained shares are
    within ``ambiguity_margin`` of the best explain the scan equally well (a
    twin maps the datum onto itself, so its datum score differs only by noise).
    Among them the planned order is the soft prior: an
    order-consistent decode is preferred (a symmetric U-channel with one flange
    formed is S1, not the out-of-order "other flange" image of it). A *near twin*
    (the motion matches a rotation E1 found within ``near_twin_rms_mm`` RMS, above
    its 0.3 mm symmetry threshold) whose total is within the margin joins this
    choice too: such a part's datum barely tells the two poses apart, so the
    order decides which half of a half-made part was formed. Before the order,
    the holes decide (``evidence``, :func:`_hole_evidence`): a rival whose
    confirmed-minus-contradicted holes beat the best total's by
    ``hole_evidence_min_diff`` is chosen, because a twin image that puts holes
    on solid sheet (a wall's hole pattern the symmetry does not preserve) is
    wrong however close its totals are. The pose is
    ambiguous only when E1's symmetry test of the *chosen decode's* stage matches
    the motion to a rival and the two disagree on a bend state or on the identity
    of a formed bend; a near twin never makes it ambiguous.
    """
    scored = [h for h in hyps if h.art is not None]
    if not scored:
        return False, [], 1.0, hyps[0]
    best = scored[0]
    if len(scored) < 2:
        return False, [], 1.0, best
    margin = float(best.total - scored[1].total)
    pool = [best]
    for h in scored[1:]:
        # a near twin (a rotation E1 finds within ``near_twin_rms_mm`` but above its symmetry threshold)
        # joins the order-prior choice only; it can never make the pose ambiguous (below)
        rel = h.fit.T @ np.linalg.inv(best.fit.T)
        if _matching_symmetry(rel, part, cfg):
            # a certified twin maps the datum onto itself, so its datum-score difference is noise;
            # it is a rival when the totals or the explained shares of the articulated model agree
            same = abs(best.explained - h.explained) + abs(best.explained_tight - h.explained_tight)
            if best.total - h.total <= cfg.ambiguity_margin or same <= cfg.ambiguity_margin:
                pool.append(h)
        elif _matching_symmetry(rel, part, cfg, rms_max=cfg.near_twin_rms_mm):
            if best.total - h.total <= cfg.ambiguity_margin:
                pool.append(h)
        # anything else is not a symmetric twin: simply a worse pose
    if len(pool) < 2:
        return False, [], margin, best

    def states_of(h: _Hyp) -> list[str]:
        return [d.state for d in (h.decodes or [])]

    holes_chosen = None
    if evidence is not None:
        ev = {id(h): evidence(h) for h in pool}
        score = {k: c - x for k, (c, x) in ev.items()}
        top = max(pool, key=lambda h: (score[id(h)], h.total))
        if note is not None:
            note("twin hole evidence (confirmed/contradicted): " + " ".join(
                f"{h.total:.3f}:{ev[id(h)][0]}/{ev[id(h)][1]}" for h in pool))
        if top is not best and score[id(top)] - score[id(best)] >= cfg.hole_evidence_min_diff:
            holes_chosen = top
    consistent = [h for h in pool if "UNKNOWN" not in states_of(h) and _order_consistent(part, states_of(h))]
    if holes_chosen is not None:
        chosen = holes_chosen
    else:
        chosen = best if (any(h is best for h in consistent) or not consistent) else consistent[0]
    chosen_states = states_of(chosen)
    formed = [b.uid for b in part.bends if chosen_states[b.index] in ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED")]
    decoded = part.stage_for_formed(formed)
    out_states = [chosen_states]
    ambiguous = False
    chosen.twins, chosen.twin_perms = [], []
    for h in pool:
        if h is chosen:
            continue
        matches = _matching_symmetry(h.fit.T @ np.linalg.inv(chosen.fit.T), part, cfg)
        if decoded is not None:
            matches = [(sid, c) for sid, c in matches if sid == decoded.stage_model_id]
        if not matches:
            continue  # the decoded stage itself is not symmetric under this motion
        st = states_of(h)
        changes_identity = any(any(c.bend_permutation.get(u, u) != u for u in formed) for _, c in matches)
        if st != chosen_states or changes_identity:
            ambiguous = True
            out_states.append(st)
            chosen.twins.append(h)
            chosen.twin_perms.append([dict(c.bend_permutation) for _, c in matches])
    return ambiguous, out_states, margin, chosen


def _mirror_refusal(mirror_total: float, proper_total: float, mirror_holes: int, proper_holes: int, *,
                    symmetric: bool, margin: float) -> str | None:
    """The MIRRORED_PART_SUSPECTED reason, or None (the scan is not called mirrored).

    A chiral shape: the mirrored fit explaining clearly more (``margin``) without more filled datum holes is the
    evidence. A shape that is its own mirror image (``symmetric``, after its mirrored fit was converted to a
    proper pose): outline and flanges cannot tell the two apart, only asymmetric features can, so the claim
    needs the datum holes: strictly fewer filled holes under the mirrored fit than under the best proper pose."""
    if symmetric:
        return f"MIRROR_HOLES_FIT_BETTER:{mirror_holes}<{proper_holes}" if mirror_holes < proper_holes else None
    if mirror_total > proper_total + margin and mirror_holes <= proper_holes:
        return f"MIRROR_FITS_BETTER:{mirror_total:.3f}>{proper_total:.3f}"
    return None


def _decoded_stage_id(part: PartView, h: "_Hyp") -> str | None:
    """The exact compiled stage a hypothesis' decode names (None when its states name no exact stage)."""
    if not h.decodes:
        return None
    stage_id, exact = choose_stage_model(part, [d.state for d in h.decodes])
    return stage_id if exact else None


def _mirror_self_maps(part: PartView, stage_id: str | None) -> list[np.ndarray]:
    """The certified MIRROR self-maps (rms within E1's threshold) of compiled stage ``stage_id``, deduplicated
    ([] when ``stage_id`` is None: no exact stage, so no certified symmetry to rely on).

    E1's symmetry contract: such a candidate means a mirrored part is indistinguishable from the CAD at that
    stage, so E3 must never claim MIRRORED_PART_SUSPECTED for a scan of it."""
    out: list[np.ndarray] = []
    for stage in part.compiled.stages:
        if stage_id is None or stage.stage_model_id != stage_id:
            continue
        sym = stage.symmetry
        for cand in sym.candidates:
            if cand.kind != "MIRROR" or cand.rms_mm > sym.threshold_rms_mm + 1e-12:
                continue
            m = np.asarray(cand.transform, dtype=np.float64)
            if not any(np.allclose(m, o, atol=1e-3) for o in out):
                out.append(m)
    return out


def _matching_symmetry(rel: np.ndarray, part: PartView, cfg: InspectConfig, rms_max: float | None = None) -> list:
    """Every ROTATION symmetry candidate of a compiled stage that ``rel`` matches.

    Candidates count within E1's threshold (a certified symmetry), or, with
    ``rms_max``, within the larger of the two (a near symmetry: the part maps onto
    itself to within ``rms_max`` mm RMS).
    """
    final = part.cloud(f"S{part.n}").points
    probe = final[:: max(1, len(final) // 2000)]
    out = []
    for stage in part.compiled.stages:
        sym = stage.symmetry
        limit = sym.threshold_rms_mm if rms_max is None else max(sym.threshold_rms_mm, float(rms_max))
        for cand in sym.candidates:
            if cand.kind != "ROTATION" or cand.rms_mm > limit + 1e-12:
                continue
            m = np.asarray(cand.transform, dtype=np.float64)
            r = rel[:3, :3] @ m[:3, :3].T
            c = (float(np.trace(r)) - 1.0) / 2.0
            if math.degrees(math.acos(max(-1.0, min(1.0, c)))) > cfg.symmetry_match_deg:
                continue
            d = np.linalg.norm(apply(rel, probe) - apply(m, probe), axis=1)
            if float(np.median(d)) <= cfg.symmetry_match_mm:
                out.append((stage.stage_model_id, cand))
    return out


def _hole_evidence(h: _Hyp, pts: np.ndarray, vox: VoxelCloud, part: PartView, cfg: InspectConfig,
                   spacing: float) -> tuple[int, int]:
    """``(confirmed, contradicted)`` holes of hypothesis ``h``'s fitted panels (``hole_evidence_*`` config).

    Every hole loop of a panel with a fitted pose is carried to the registered
    frame by that pose. Scan points in the panel's material slab are projected on
    its plane: the core (the loop shrunk by ``max(hole_evidence_margin_mm, 2
    spacings)``) against the 3 mm annulus outside the loop. A core holding at most
    ``hole_evidence_open_fraction`` of the points the annulus density predicts
    confirms the hole; ``hole_evidence_filled_fraction`` or more contradicts it.
    Core points owned by another fitted panel (a tab still flat in its window)
    do not count. Holes in an unobserved area (annulus density below
    ``hole_evidence_min_density`` per mm2) or too small to judge (fewer than
    ``hole_evidence_min_expected`` predicted core points) are skipped, and so
    are holes another fitted panel fills (a tab still flat in its window).
    """
    from matplotlib.path import Path as MplPath

    from domains.production.engine.inspect.articulated import _in_core, inside_grown
    from domains.production.engine.inspect.geom import orthonormal_frame

    if h.art is None:
        return 0, 0
    reg = apply(h.fit.T, pts)
    owner = h.art.owner[vox.inverse]
    margin = max(cfg.hole_evidence_margin_mm, 2.0 * spacing)
    ring = 3.0
    confirmed = contradicted = 0
    for pf in h.art.panels:
        if pf.A is None or pf.n0 is None or pf.d0 is None or pf.state == "UNKNOWN" or pf.model_plane:
            continue
        holes = part.panels[pf.index].model.holes_mm
        if not holes:
            continue
        t = float(pf.t or part.t)
        hs = reg @ pf.n0 - float(pf.d0)
        slab = np.nonzero((hs < 0.6) & (hs > -t - 0.6))[0]
        if len(slab) < 20:
            continue
        e1, e2 = orthonormal_frame(pf.n0)
        sp = reg[slab]
        for hole in holes:
            hw = apply(pf.A, np.asarray(hole, dtype=np.float64))
            if len(hw) < 3:
                continue
            lo, hi = hw.min(axis=0) - ring - 1.0, hw.max(axis=0) + ring + 1.0
            box = np.nonzero(np.all(sp > lo, axis=1) & np.all(sp < hi, axis=1))[0]
            if len(box) < 10:
                continue
            q = np.stack([sp[box] @ e1, sp[box] @ e2], axis=1)
            poly = np.stack([hw @ e1, hw @ e2], axis=1)
            closed = np.vstack([poly, poly[:1]])
            area = 0.5 * abs(float(np.sum(closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1])))
            perim = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
            core_area = area - perim * margin
            if core_area <= 0.0:
                continue
            inside = MplPath(poly).contains_points(q)
            ann = inside_grown(poly, q, ring) & ~inside
            dens = float(ann.sum()) / (perim * ring + math.pi * ring * ring)
            if dens < cfg.hole_evidence_min_density:
                continue
            expected = dens * core_area
            if expected < cfg.hole_evidence_min_expected:
                continue
            core = _in_core(poly, q, margin)
            own = owner[slab[box[core]]]
            mine = (own < 0) | (own == pf.index)
            if int(np.sum(~mine)) >= cfg.hole_evidence_filled_fraction * expected:
                continue  # another fitted panel lies in it (a tab still flat in its window): no evidence either way
            n_core = int(np.sum(mine))
            if n_core <= cfg.hole_evidence_open_fraction * expected:
                confirmed += 1
            elif n_core >= cfg.hole_evidence_filled_fraction * expected:
                contradicted += 1
    return confirmed, contradicted


def _articulated_model(part: PartView, art: ArticulatedFit, decodes: list[BendDecode]) -> np.ndarray:
    """The articulated model in the registered frame (every 2nd final-stage sample; NaN rows: not placed).

    The final-stage sample cloud is mapped to the decoded formed set (E1's
    ``stage_points_from_compiled``), then moved panel by panel onto the fitted
    articulated poses (a bend zone with its parent).
    """
    from domains.production.engine.compile.stages import stage_points_from_compiled

    formed = [b.uid for b, d in zip(part.bends, decodes) if d.state in ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED")]
    final = part.cloud(f"S{part.n}")
    step = 2
    pts, nrm = final.points[::step], final.normals[::step]
    kind, index = final.segment_kind[::step], final.segment_index[::step]
    stage_pts, _ = stage_points_from_compiled(part.compiled, pts, nrm, kind, index, formed)
    stage_tr = part.formed_set_transforms(formed)
    owner = np.where(kind == 1, np.array([part.bends[int(i)].parent if kind_i == 1 else 0
                                          for i, kind_i in zip(index, kind)]), index)
    model = np.full_like(stage_pts, np.nan)
    for pidx, pf in enumerate(art.panels):
        if pf.A is None or pf.state == "UNKNOWN":
            continue
        sel = owner == pidx
        if not sel.any():
            continue
        C = pf.A @ np.linalg.inv(stage_tr[pidx])
        model[sel] = apply(C, stage_pts[sel])
    return model


def _model_explained(part: PartView, art: ArticulatedFit, decodes: list[BendDecode], vox_reg: np.ndarray,
                     cfg: InspectConfig) -> tuple[float, float]:
    """Shares of voxel points within ``explained_tol_mm`` and within a tight tolerance of the model surface.

    The model is the final-stage sample cloud mapped to the decoded formed set
    (E1's ``stage_points_from_compiled``), then moved panel by panel onto the
    fitted articulated poses; panels the fit could not place contribute nothing.
    """
    model = _articulated_model(part, art, decodes)
    ok = np.isfinite(model).all(axis=1)
    if ok.sum() < 10:
        return 0.0, 0.0
    tree = cKDTree(model[ok], balanced_tree=False, compact_nodes=False)
    d, _ = tree.query(vox_reg, k=1, distance_upper_bound=cfg.explained_tol_mm, workers=1)
    tight = min(cfg.explained_tight_max_mm, max(cfg.explained_tight_min_mm, cfg.explained_tight_fraction * part.t))
    return float(np.mean(np.isfinite(d))), float(np.mean(d <= tight))


def _registration_block(best: _Hyp, n_cand: int, margin: float, ambiguous: bool, degenerate: bool,
                        cfg: InspectConfig, unit_factor: float, identity_ambiguous: bool = False) -> RegistrationV1:
    rms = best.fit.edge_rms if math.isfinite(best.fit.edge_rms) else None
    return RegistrationV1(method="datum_plane_outline_edges_holes", raw_to_frame=_mat(best.fit.T), unit_factor=unit_factor,
                          datum_fitness=float(min(1.0, max(0.0, best.fit.datum_coverage))),
                          rmse_mm=None if rms is None else float(rms), pose_candidates_considered=int(n_cand),
                          pose_margin=float(max(0.0, margin)), pose_ambiguous=bool(ambiguous),
                          identity_ambiguous=bool(ambiguous and identity_ambiguous),
                          in_plane_degenerate=bool(degenerate and not ambiguous), abstained=False)


def _stage_detail(part: PartView, states: list[str], stage_id: str | None, exact: bool, ambiguous: bool) -> str:
    seq = " ".join(f"{b.label}:{s[0:3]}" for b, s in zip(part.bends, states))
    if ambiguous:
        return f"pose ambiguous (symmetric part); {seq}"
    if stage_id is None:
        return f"no compiled stage model for the decoded formed set; {seq}"
    return f"decoded against {stage_id}{'' if exact else ' (display only: some bends undetermined)'}; {seq}"


def _position_trust(part: PartView, fit: InPlaneFit, model: DatumModel, states: list[str],
                    geoms: dict[int, BendMeasure], lines, transforms, pose_degenerate: bool, cfg: InspectConfig
                    ) -> dict[int, tuple[bool, float | None]]:
    """Per measurable formed bend under a degenerate in-plane pose: (position trusted, frame sigma in mm).

    The bend-line position is measured along side A's outward direction at the
    nominal line. It is trusted when that direction does not depend on an
    undetermined pose direction (:meth:`InPlaneFit.position_determined`) and the
    determined directions' sigma there is at most ``degenerate_max_sigma_mm``.
    A well-determined pose returns ``{}`` (every position uses the pose's
    scalar sigma, as before).
    """
    out: dict[int, tuple[bool, float | None]] = {}
    if not pose_degenerate:
        return out
    for j, bend in enumerate(part.bends):
        bm = geoms.get(j)
        line = lines.get(bend.uid)
        if states[j] not in _FORMED_LIKE or bm is None or not bm.ok or line is None:
            continue
        q0, q1 = np.asarray(line[0], dtype=np.float64), np.asarray(line[1], dtype=np.float64)
        dn = unit(q1 - q0)
        out_a = transforms[bend.side_panel["A"]][:3, :3] @ bend.side_out["A"]
        oa = out_a - float(out_a @ dn) * dn
        if float(np.linalg.norm(oa)) < 1e-9:
            out[j] = (False, None)
            continue
        oa = unit(oa)
        mid = 0.5 * (q0 + q1)
        if not fit.position_determined(model, mid, oa):
            out[j] = (False, None)
            continue
        sig = fit.position_sigma_mm(model, mid, oa)
        out[j] = (bool(sig <= cfg.degenerate_max_sigma_mm), sig)
    return out


def _outcomes(part: PartView, best: _Hyp, states, decodes, geoms, lines, transforms, stage_id, exact, degenerate,
              ambiguous, idx: PointIndex, cfg: InspectConfig, budget: _Budget, thick: Thickness, ctx: _Ctx,
              pos_trust: dict[int, tuple[bool, float | None]] | None = None) -> list[BendOutcome]:
    out: list[BendOutcome] = []
    art = best.art
    n = part.n
    formed_set = {part.bends[j].uid for j, s in enumerate(states) if s in _FORMED_LIKE}
    k = len(formed_set)
    order_ok = stage_id == f"S{k}" and exact
    bg_stage = not ambiguous and "UNKNOWN" not in states and _order_consistent(part, states)
    pose_sigma = _pose_sigma(best.fit)
    fit = best.fit
    pose_bias = None if ctx.model is None else (lambda p, d: fit.position_bias_mm(ctx.model, p, d))
    datum_normal = None if ctx.model is None else ctx.model.n0
    erosion = float(fit.erosion_mm)
    for j in range(n):
        bend = part.bends[j]
        d = decodes[j] if decodes else None
        st = states[j]
        o = BendOutcome(state=st, prior=None if d is None else d.prior)
        o.nominal_line = lines.get(bend.uid)
        if st == "UNKNOWN":
            reason = "AMBIGUOUS_POSE" if ambiguous else ((d.reasons[0] if d and d.reasons else None) or "OCCLUDED")
            o.state_reasons = (reason,)
        if d is not None and not ambiguous:
            # a negative sweep is a child formed the wrong way: its included angle (above 180) is no evidence value
            o.measured_included_deg = None if d.measured_sweep_deg is None or d.measured_sweep_deg < 0.0 \
                else 180.0 - float(d.measured_sweep_deg)
            o.formed_fraction = d.formed_fraction
        bf = art.bends[j]
        o.diagnostics = {k2: v for k2, v in {
            "hough_theta_deg": bf.theta_deg, "hough_score": bf.score, "hinge_axial_coverage": bf.axial_coverage,
        }.items() if v is not None}
        # backgauge of the decoded mid-sequence stage S_k, on bend k. ENGINE_IO §4.3 wants it on every
        # order-consistent decode; when the formed set is order-consistent but not the prefix (a swap
        # inside one machine move) the S_k nominal does not describe the part: NOT_APPLICABLE_AT_STAGE.
        if bg_stage and 1 <= k <= n - 1 and bend.seq == k:
            cid = f"{bend.uid}/BACKGAUGE_LENGTH/S{k}"
            if cid in part.nominals:
                o.backgauge_cid = cid
                o.backgauge_na = not order_ok
        if st in _FORMED_LIKE and not ambiguous:
            # a flange bounded by a bend that is not yet formed has no length at this stage (contract amendment
            # ENGINE_IO-A1): NOT_APPLICABLE_AT_STAGE, whatever this bend's own geometry shows. An UNKNOWN neighbour
            # is not the stage: that side stays UNKNOWN(ADJACENT_BEND_NOT_MEASURED)
            for side in ("A", "B"):
                nom_f = part.nominals.get(f"{bend.uid}/FLANGE_LENGTH/{side}")
                if nom_f is None or nom_f.boundary_kind != "ADJACENT_BEND" or nom_f.recipe_requirement == "NOT_INSPECTED":
                    continue
                jb = part.bend_index.get(nom_f.adjacent_bend_uid or "")
                if jb is not None and states[jb] == "NOT_YET_FORMED":
                    o.stage_na.add(nom_f.characteristic_id)
            bm = geoms.get(j)
            if bm is None or not bm.ok:
                reason = "TIMEOUT" if budget.left() <= 0 else ((bm.reason if bm else None) or "OCCLUDED")
                for kind, side in (("FLANGE_LENGTH", "A"), ("FLANGE_LENGTH", "B"), ("INCLUDED_ANGLE", None),
                                   ("BEND_LINE_POSITION", None)):
                    o.forced_unknown[f"{bend.uid}/{kind}" + (f"/{side}" if side else "")] = reason
                if degenerate:  # the contract: a degenerate in-plane pose always reads REGISTRATION_UNTRUSTED
                    o.forced_unknown[f"{bend.uid}/BEND_LINE_POSITION"] = "REGISTRATION_UNTRUSTED"
                if o.backgauge_cid and not o.backgauge_na:
                    o.forced_unknown[o.backgauge_cid] = reason
                out.append(o)
                continue
            o.support_parent, o.support_child, o.coverage = bm.support_parent, bm.support_child, bm.coverage
            ang = angle_measurement(bm, cfg)
            o.measurements[f"{bend.uid}/INCLUDED_ANGLE"] = ang
            o.measured_included_deg = ang.value
            o.formed_fraction = (180.0 - float(ang.value)) / math.degrees(bend.sweep)
            if bm.reversed:
                # formed the wrong way: the angle (180 + |sweep|) FAILs; the CAD's OML skin is this fold's inner one,
                # so the lengths that start or end at its sharp are not this bend's characteristics. Its bend line
                # is measured by mirror symmetry (the fold's own outer sharp reflected across the parent's
                # mid-plane against the nominal) and published at that outer sharp
                o.notes = o.notes + ("BEND_FORMED_WRONG_WAY",)
                o.measured_included_deg = None
                for side in ("A", "B"):
                    o.forced_unknown[f"{bend.uid}/FLANGE_LENGTH/{side}"] = "UNSUPPORTED_FEATURE"
                if o.backgauge_cid and not o.backgauge_na:
                    o.forced_unknown[o.backgauge_cid] = "UNSUPPORTED_FEATURE"
                cid_p = f"{bend.uid}/BEND_LINE_POSITION"
                trusted, sigma_here = (pos_trust or {}).get(j, (True, None))
                m_pos = None
                if not (degenerate or not trusted):
                    out_a = transforms[bend.side_panel["A"]][:3, :3] @ bend.side_out["A"]
                    m_pos = reversed_position_measurement(bm, lines[bend.uid], out_a,
                                                          pose_sigma if sigma_here is None else sigma_here, cfg,
                                                          pose_bias, thick=thick, datum_normal=datum_normal)
                if m_pos is None or bm.rev_line_point is None:
                    o.forced_unknown[cid_p] = "REGISTRATION_UNTRUSTED" if (degenerate or not trusted) \
                        else "UNSUPPORTED_FEATURE"
                else:
                    o.measurements[cid_p] = m_pos
                    ok_r = bm.valid & np.isfinite(bm.rev_sharp).all(axis=1)
                    rel = bm.rev_sharp[ok_r] - bm.rev_line_point
                    shift_r = np.median(rel - np.outer(rel @ bm.line_dir, bm.line_dir), axis=0)
                    lo, hi = bm.visible
                    o.measured_line = (bm.rev_line_point + bm.line_dir * lo + shift_r,
                                       bm.rev_line_point + bm.line_dir * hi + shift_r)
                o.diagnostics["visible_length_mm"] = float(bm.visible[1] - bm.visible[0])
                o.diagnostics["stations"] = float(len(bm.z))
                out.append(o)
                continue
            # the measured lines are the station-fitted ones: the global sharp (and the apex derived from
            # it) moved onto the station median
            shift = np.zeros(3)
            vs = bm.sharp[bm.valid]
            if len(vs):
                rel = vs - bm.line_point
                shift = np.median(rel - np.outer(rel @ bm.line_dir, bm.line_dir), axis=0)
            # bend-line position against the decoded stage model's nominal OML sharp
            cid_p = f"{bend.uid}/BEND_LINE_POSITION"
            trusted, sigma_here = (pos_trust or {}).get(j, (True, None))
            if degenerate or not trusted:
                o.forced_unknown[cid_p] = "REGISTRATION_UNTRUSTED"
            else:
                side_a_panel = bend.side_panel["A"]
                out_a = transforms[side_a_panel][:3, :3] @ bend.side_out["A"]
                o.measurements[cid_p] = position_measurement(bm, lines[bend.uid], out_a,
                                                             pose_sigma if sigma_here is None else sigma_here, cfg,
                                                             pose_bias, thick=thick, datum_normal=datum_normal)
                lo, hi = bm.visible
                o.measured_line = (bm.line_point + bm.line_dir * lo + shift, bm.line_point + bm.line_dir * hi + shift)
            for side in ("A", "B"):
                cid = f"{bend.uid}/FLANGE_LENGTH/{side}"
                nom = part.nominals[cid]
                if nom.recipe_requirement == "NOT_INSPECTED" or nom.nominal is None or cid in o.stage_na:
                    continue
                if nom.boundary_kind != "ADJACENT_BEND" and ctx.free_edge_blocked is not None:
                    o.forced_unknown[cid] = ctx.free_edge_blocked
                    continue
                m = measure_flange(idx, part, art, geoms, j, side, nom.boundary_kind, nom.adjacent_bend_uid,
                                   ctx.spacing, cfg, erosion, creases=ctx.crease_sharps.get(bend.side_panel[side], ()),
                                   thick=thick)
                if nom.boundary_kind == "ADJACENT_BEND":
                    jb = part.bend_index.get(nom.adjacent_bend_uid or "")
                    if jb is not None and states[jb] not in _FORMED_LIKE:
                        m = Measurement.unknown("ADJACENT_BEND_NOT_MEASURED", "OML_SHARP_TO_OML_SHARP")
                o.measurements[cid] = m
                if m.value is not None:
                    w = bm.w[bend.side_panel[side]]
                    mid = 0.5 * (bm.line_point + bm.line_dir * bm.visible[0] + bm.line_point + bm.line_dir * bm.visible[1])
                    o.side_anchor[side] = mid + shift + w * (0.5 * float(m.value))
            if o.backgauge_cid is not None and not o.backgauge_na:
                nom = part.nominals[o.backgauge_cid]
                if nom.nominal is not None and nom.side is not None:
                    if nom.boundary_kind != "ADJACENT_BEND" and ctx.free_edge_blocked is not None:
                        o.forced_unknown[o.backgauge_cid] = ctx.free_edge_blocked
                    else:
                        unformed = {i for i, s in enumerate(states) if s == "NOT_YET_FORMED"}
                        island, crossed = part.flat_island(bend.side_panel[nom.side], j, unformed)
                        ks = _k_sigma(part, stage_id, j, nom.side, crossed, float(nom.nominal), ctx.note, k)
                        o.measurements[o.backgauge_cid] = measure_backgauge(
                            idx, part, art, geoms, j, nom.side, nom.boundary_kind, nom.adjacent_bend_uid, island,
                            crossed, ks, ctx.spacing, cfg, erosion, thick=thick)
            if bm.apex is not None:
                a0, a1 = bm.apex[0] + shift, bm.apex[1] + shift
                o.diagnostics.update({"outer_apex_start_x": float(a0[0]), "outer_apex_start_y": float(a0[1]),
                                      "outer_apex_start_z": float(a0[2]), "outer_apex_end_x": float(a1[0]),
                                      "outer_apex_end_y": float(a1[1]), "outer_apex_end_z": float(a1[2])})
            o.diagnostics["visible_length_mm"] = float(bm.visible[1] - bm.visible[0])
            o.diagnostics["stations"] = float(len(bm.z))
            if bm.one_skin:
                o.notes = o.notes + ("ONE_SKIN_OML_FROM_INNER_SKIN_PLUS_T",)
                o.diagnostics["thickness_used_mm"] = float(thick.t_mm)
                o.diagnostics["inner_skin_share_parent"] = float(bm.inner_parent)
                o.diagnostics["inner_skin_share_child"] = float(bm.inner_child)
        out.append(o)
    _bare_edges(out, part, cfg, ctx.note)
    if fit.undetermined.size and ctx.model is not None:
        _pose_borne_lengths(out, part, geoms, lines, fit, ctx.model, ctx.note)
    return out


def _gb(nom, value: float, u: float) -> str | None:
    """The contract's guard-band decision of ``value`` with U ``u`` (rounded as the row factory rounds), else None."""
    from contracts.production.characteristics import guard_band_decision

    if nom is None or nom.nominal is None or nom.tolerance is None or value is None or u is None:
        return None
    dev = round6(round6(float(value)) - round6(float(nom.nominal)))
    return guard_band_decision(dev, max(round6(float(u)), 1e-6), float(nom.tolerance.plus), float(nom.tolerance.minus))


def review_u(nom, value: float) -> float:
    """The smallest U (plus 1e-3) that puts ``value`` in the guard band (REVIEW): never PASS, never FAIL."""
    dev = round6(round6(float(value)) - round6(float(nom.nominal)))
    p, m = float(nom.tolerance.plus), float(nom.tolerance.minus)
    not_fail = max(dev - p, -m - dev, 0.0)
    not_pass = max(min(p - dev, dev + m), 0.0)
    return max(not_fail, not_pass) + 1e-3


def withhold_u(nom, value: float, u: float, delta: float) -> float:
    """U of a withheld reading: the observed alternative spread added, and at least :func:`review_u`."""
    return max(float(u) + float(delta), review_u(nom, value))


def _branch_withhold(outcomes: list[BendOutcome], part: PartView, best: _Hyp, states, decodes, geoms, lines,
                     transforms, stage_id, exact, degenerate, ambiguous, idx: PointIndex, cfg: InspectConfig,
                     budget: _Budget, thick: Thickness, ctx: "_Ctx", pos_trust) -> None:
    """e3-1.13 ``branch_withhold``: a characteristic that a near-tie choice decides.

    Every bend measured with a near-tie skin / window choice (``BendMeasure.alt``) is substituted by its
    alternative reading and the characteristics are measured again. A decided characteristic (PASS or FAIL)
    whose alternative reading is not the same decision is withheld: REVIEW, with the alternative's difference
    added to its U. A characteristic both readings decide the same keeps its reading. Nothing becomes PASS.
    """
    alts = [j for j, g in sorted(geoms.items()) if g is not None and g.ok and g.alt is not None]
    if not alts:
        return
    withheld: list[str] = []
    for j in alts:
        if budget.left() <= 0:
            break
        g2 = dict(geoms)
        g2[j] = geoms[j].alt
        ctx.note(f"branch near-tie {part.bends[j].label}: {','.join(sorted(set(geoms[j].ties)))}"
                 + ("" if geoms[j].alt.ok else f" (alternative not measured: {geoms[j].alt.reason})"))
        alt_out = _outcomes(part, best, states, decodes, g2, lines, transforms, stage_id, exact, degenerate,
                            ambiguous, idx, cfg, budget, thick, ctx, pos_trust)
        for o, oa in zip(outcomes, alt_out):
            for cid, m in o.measurements.items():
                if m.value is None or m.u is None or cid in o.forced_unknown:
                    continue
                nom = part.nominals.get(cid)
                d0 = _gb(nom, m.value, m.u)
                if d0 not in ("PASS", "FAIL"):
                    continue
                ma = oa.measurements.get(cid)
                if ma is None or ma.value is None or ma.u is None or cid in oa.forced_unknown:
                    continue  # the alternative reading cannot measure it: no evidence against this one
                if _gb(nom, ma.value, ma.u) == d0:
                    continue
                m.u = withhold_u(nom, m.value, m.u, abs(float(ma.value) - float(m.value)))
                m.notes = tuple(m.notes) + ("BRANCH_NEAR_TIE_WITHHELD",)
                withheld.append(cid.split("/", 1)[-1] if "/" in cid else cid)
    if withheld:
        ctx.note("branch near-tie withheld: " + " ".join(withheld))


def _station_free_decision(m: Measurement, nom) -> bool:
    """True when a station-nominal comparison decides the same whichever station nominal each station had.

    With the stations' place along the bend undetermined, the median deviation can lie anywhere between
    ``raw_median - max(profile)`` and ``raw_median - min(profile)``; the reading stands only when the guard-band
    decision is the same over that whole interval: identical at both ends, and either PASS at both ends (the PASS
    band is one interval) or both ends on the same side of the nominal (FAIL short .. FAIL long passes through PASS)."""
    from contracts.production.characteristics import guard_band_decision

    if m.profile_range is None or m.raw_median is None or nom is None or nom.tolerance is None or m.u is None:
        return False
    lo, hi = m.profile_range
    ends = [m.raw_median - hi, m.raw_median - lo]
    decs = [guard_band_decision(d, float(m.u), float(nom.tolerance.plus), float(nom.tolerance.minus)) for d in ends]
    if decs[0] != decs[1]:
        return False
    return decs[0] == "PASS" or (ends[0] > 0) == (ends[1] > 0)


def _pose_borne_lengths(outcomes: list[BendOutcome], part: PartView, geoms: dict[int, BendMeasure], lines,
                        fit: InPlaneFit, model: DatumModel, note: Callable[[str], None] | None = None) -> None:
    """Flange lengths that depend on an undetermined in-plane pose direction become UNKNOWN(REGISTRATION_UNTRUSTED).

    Lengths are scan-intrinsic only while two things do not come from the pose:
    - the skin identity of a sharp built from the *inner* skin moved by t (``BendMeasure.one_skin``): the skin is
      called inner by its height against the registered panel plane, so when that panel's normal lies along an
      undetermined pose direction (:meth:`InPlaneFit.position_determined`) a pose error of about t flips the skins
      and moves the sharp by about t. Every length that starts or ends at such a sharp is untrusted;
    - the station of a length compared with a station-varying nominal profile (``..._VS_STATION_NOMINAL``): the
      station's place along the bend line comes from the pose, so it is untrusted when the bend line's own
      direction is undetermined.
    """
    skin_borne: dict[int, bool] = {}
    for j, bm in geoms.items():
        line = lines.get(part.bends[j].uid)
        if bm is None or not bm.ok or not bm.one_skin or line is None:
            continue
        mid = 0.5 * (np.asarray(line[0], dtype=np.float64) + np.asarray(line[1], dtype=np.float64))
        sides = [(bm.inner_parent, bm.plane_parent), (bm.inner_child, bm.plane_child)]
        skin_borne[j] = any(share > 0 and pl is not None and not fit.position_determined(model, mid, unit(pl.n))
                            for share, pl in sides)
    hits = []
    for j, (bend, o) in enumerate(zip(part.bends, outcomes)):
        line = lines.get(bend.uid)
        for side in ("A", "B"):
            cid = f"{bend.uid}/FLANGE_LENGTH/{side}"
            m = o.measurements.get(cid)
            if m is None or m.value is None or cid in o.forced_unknown or line is None:
                continue
            nom = part.nominals.get(cid)
            jb = part.bend_index.get((nom.adjacent_bend_uid if nom is not None else None) or "")
            why = None
            if skin_borne.get(j) or (jb is not None and skin_borne.get(jb)):
                why = "skin"
            elif "VS_STATION_NOMINAL" in (m.method or ""):
                q0, q1 = np.asarray(line[0], dtype=np.float64), np.asarray(line[1], dtype=np.float64)
                if not fit.position_determined(model, 0.5 * (q0 + q1), unit(q1 - q0)) \
                        and not _station_free_decision(m, nom):
                    why = "station"
            if why is not None:
                o.forced_unknown[cid] = "REGISTRATION_UNTRUSTED"
                o.side_anchor.pop(side, None)
                hits.append(f"{bend.label}/{side}:{why}")
    if hits and note is not None:
        note("pose-borne flange lengths withheld: " + " ".join(hits))


def _spec(part: PartView, cid: str) -> tuple | None:
    nom = part.nominals.get(cid)
    if nom is None or nom.nominal is None or nom.tolerance is None:
        return None
    return (round6(float(nom.nominal)), round6(float(nom.tolerance.plus)), round6(float(nom.tolerance.minus)),
            nom.recipe_requirement)


def _matched_spec_ok(part: PartView, uid: str, kind: str, perms: list[list[dict]]) -> bool:
    """The matched characteristic (the twin's label of the same physical bend, ``perm[uid]``) has the same
    nominal and tolerance under every twin's every matching symmetry. Flange sides compare as a set (the
    symmetry may exchange a bend's sides)."""
    def specs(u: str) -> tuple:
        if kind == "FLANGE_LENGTH":
            return tuple(sorted((str(_spec(part, f"{u}/FLANGE_LENGTH/{s}")) for s in ("A", "B"))))
        return (str(_spec(part, f"{u}/{kind}")),)
    own = specs(uid)
    return all(specs(perm.get(uid, uid)) == own for twin in perms for perm in twin)


def _reading(o: BendOutcome, cid: str) -> tuple:
    if cid in o.stage_na:
        return ("NA",)
    if cid in o.forced_unknown:
        return ("UNKNOWN",)
    m = o.measurements.get(cid)
    if m is None or m.value is None or m.u is None:
        return ("UNKNOWN",)
    return ("M", float(m.value), max(float(m.u), 1e-6))


def _agree(part: PartView, cid: str, readings: list[tuple]) -> bool:
    """Every twin reads the characteristic, the values agree within the published (first) reading's U and the
    guard-band decision is the same under each twin's own U (so a PASS holds whichever twin is the part)."""
    from contracts.production.characteristics import guard_band_decision

    if any(r[0] != "M" for r in readings):
        return False
    nom = part.nominals.get(cid)
    if nom is None or nom.nominal is None or nom.tolerance is None:
        return False
    v0, u0 = readings[0][1], readings[0][2]
    decs = {guard_band_decision(r[1] - float(nom.nominal), r[2], float(nom.tolerance.plus), float(nom.tolerance.minus))
            for r in readings}
    return len(decs) == 1 and all(abs(r[1] - v0) <= u0 for r in readings)


def _twin_agreement(part: PartView, recipe, chosen: _Hyp, pts, vox, cfg: InspectConfig, budget: _Budget,
                    thick: Thickness, ctx: _Ctx, ambiguous_outcomes: list[BendOutcome],
                    note: Callable[[str], None]) -> tuple[list[BendOutcome], tuple] | None:
    """Readings identical under every tied certified symmetric twin (contract amendment ENGINE_IO-A2).

    Each twin (the chosen pose first) is decoded and measured as an unambiguous pose would be, without the
    crease search, the wrong-way search or the free-direction refinement. Per bend label: the bend is published
    only when every twin gives it the same state; a characteristic of a formed bend only when (a) the matched
    characteristic (the twin's label of the same physical bend, through E1's certified bend permutation) has
    the same nominal and tolerance, and (b) every twin reads it, within the published U, with the same
    guard-band decision. Everything else is UNKNOWN(AMBIGUOUS_POSE). Bend-line position, backgauge and the
    heatmap stay withheld: they depend on which bend is which. None when the budget runs out.
    """
    hyps = [chosen] + list(chosen.twins)
    per = []
    for h in hyps:
        if h.art is None or budget.left() <= 0:
            return None
        decodes = h.decodes or []
        states = [d.state for d in decodes]
        stage_id, exact = choose_stage_model(part, states)
        formed = [part.bends[j].uid for j, st in enumerate(states) if st in _FORMED_LIKE]
        lines = part.stage_lines(stage_id, formed)
        transforms = part.stage_transforms(stage_id, formed)
        idx = PointIndex(apply(h.fit.T, pts), vox)
        geoms = {j: measure_bend_geometry(idx, part, h.art, j, cfg) for j, st in enumerate(states)
                 if st in _FORMED_LIKE and budget.left() > 0}
        if budget.left() <= 0:
            return None
        pos_trust = {j: (False, None) for j in range(part.n)}  # positions are withheld anyway
        ctx.free_edge_blocked, ctx.crease_sharps = None, {}
        outs = _outcomes(part, h, states, decodes, geoms, lines, transforms, stage_id, exact, True, False, idx, cfg,
                         budget, thick, ctx, pos_trust)
        gen = _general_measurements(part, recipe, states, geoms, h.art, cfg, budget)
        per.append((states, outs, gen))
    merged: list[BendOutcome] = []
    published = 0
    for j, bend in enumerate(part.bends):
        sts = {p[0][j] for p in per}
        o = per[0][1][j]
        if len(sts) != 1 or o.state == "UNKNOWN":
            merged.append(ambiguous_outcomes[j])
            continue
        o.backgauge_cid, o.backgauge_na = None, False  # the stage model (and its backgauge) depends on the identity
        if o.state == "NOT_YET_FORMED":
            merged.append(o)
            continue
        o.measured_line = None
        o.forced_unknown[f"{bend.uid}/BEND_LINE_POSITION"] = "AMBIGUOUS_POSE"
        for kind, side in (("FLANGE_LENGTH", "A"), ("FLANGE_LENGTH", "B"), ("INCLUDED_ANGLE", None)):
            cid = f"{bend.uid}/{kind}" + (f"/{side}" if side else "")
            rs = [_reading(p[1][j], cid) for p in per]
            if all(r == ("NA",) for r in rs):
                continue
            if _matched_spec_ok(part, bend.uid, kind, chosen.twin_perms) and _agree(part, cid, rs):
                published += 1
                continue
            o.forced_unknown[cid] = "AMBIGUOUS_POSE"
            o.stage_na.discard(cid)
            if kind == "FLANGE_LENGTH":
                o.side_anchor.pop(side, None)
        if all(cid in o.forced_unknown for cid in (f"{bend.uid}/INCLUDED_ANGLE",)):
            o.measured_included_deg = None
        o.notes = o.notes + ("IDENTITY_AMBIGUOUS_TWIN_AGREEMENT",)
        merged.append(o)
    meas0, anchors0, na0 = per[0][2]
    meas = dict(meas0)
    for gm in recipe.general_measurements:
        cid = f"gm/{gm.measurement_id}"
        if all(cid in p[2][2] for p in per):
            continue
        rs = []
        for p in per:
            m = p[2][0].get(cid)
            rs.append(("M", float(m.value), max(float(m.u), 1e-6)) if m is not None and m.value is not None
                      and m.u is not None and cid not in p[2][2] else ("UNKNOWN",))
        if _agree(part, cid, rs):
            published += 1
            continue
        meas[cid] = Measurement.unknown("AMBIGUOUS_POSE", "GENERAL_TWIN_AGREEMENT")
    na = {cid for cid in na0 if all(cid in p[2][2] for p in per)}
    for cid in na0 - na:
        meas[cid] = Measurement.unknown("AMBIGUOUS_POSE", "GENERAL_TWIN_AGREEMENT")
    note(f"twin agreement over {len(hyps)} certified twins: {published} reading(s) identical under every twin "
         "published, the rest UNKNOWN(AMBIGUOUS_POSE)")
    return merged, (meas, anchors0, na)


def _bare_edges(outcomes: list[BendOutcome], part: PartView, cfg: InspectConfig,
                note: Callable[[str], None] | None = None) -> None:
    """A FREE_EDGE far edge seen without its cut wall can prove a side too long, never too short.

    A short-side FAIL on such an edge becomes UNKNOWN(EDGE_NOT_SEEN)
    (:func:`~domains.production.engine.inspect.metrology.bare_edge_untrusted`).
    """
    entries = [(o, cid, m) for o in outcomes for cid, m in o.measurements.items()
               if m.value is not None and m.wall_stations > 0]
    if note is not None and entries:
        note("far-edge cut walls seen: " + " ".join(f"{cid.split('/', 1)[1]}@{part.bends[part.bend_index[cid.split('/')[0]]].label}"
                                                    f"={m.walls_seen}/{m.wall_stations}" for _, cid, m in entries))
    for o, cid, m in entries:
        nom = part.nominals.get(cid)
        if nom is None or nom.tolerance is None:
            continue
        bad = bare_edge_untrusted(m, nom.nominal, nom.tolerance.plus, nom.tolerance.minus, cfg)
        if bad is not None:
            o.measurements[cid] = bad


def _general_measurements(part: PartView, recipe, states: list[str], geoms: dict[int, BendMeasure],
                          art: ArticulatedFit, cfg: InspectConfig, budget: _Budget) -> tuple[dict, dict, set]:
    """Operator-tagged straight-line characteristics (``gm/<id>``).

    ``bend_pair``: each CAD anchor belongs to the nearer of its two bends; its
    offset from that bend's nominal OML sharp (in the bend's own frame) is
    carried onto the measured OML sharp, so the measured distance is taken
    between the same physical loci as the nominal. A pair with a not-yet-formed
    bend is NOT_APPLICABLE_AT_STAGE; an unknown or unmeasured bend makes the
    measurement UNKNOWN. ``points`` anchors (on panel surfaces) have no defined
    scan correspondence yet and stay UNKNOWN(UNSUPPORTED_FEATURE).
    """
    meas: dict[str, Measurement] = {}
    anchors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    na: set[str] = set()
    for gm in recipe.general_measurements:
        cid = f"gm/{gm.measurement_id}"
        if gm.kind != "bend_pair":
            meas[cid] = Measurement.unknown("UNSUPPORTED_FEATURE", "GENERAL_POINTS")
            continue
        js = [part.bend_index[u] for u in gm.bend_uids]
        if any(states[j] == "NOT_YET_FORMED" for j in js):
            na.add(cid)
            continue
        if any(states[j] == "UNKNOWN" for j in js):
            meas[cid] = Measurement.unknown("BEND_STATE_UNKNOWN", "GENERAL_BEND_PAIR")
            continue
        bms = [geoms.get(j) for j in js]
        if any(bm is not None and bm.ok and bm.reversed for bm in bms):
            # a bend formed the wrong way: its OML sharp sits on the wrong skin, not on the anchor's locus
            meas[cid] = Measurement.unknown("UNSUPPORTED_FEATURE", "GENERAL_BEND_PAIR", "bend formed the wrong way")
            continue
        if any(bm is None or not bm.ok for bm in bms):
            missing = [bm for bm in bms if bm is None or not bm.ok]
            reason = "TIMEOUT" if any(bm is None for bm in missing) and budget.left() <= 0 else \
                next((bm.reason for bm in missing if bm is not None and bm.reason), "OCCLUDED")
            meas[cid] = Measurement.unknown(reason, "GENERAL_BEND_PAIR")
            continue
        pts_cad = [np.asarray(gm.point_start_mm, dtype=np.float64), np.asarray(gm.point_end_mm, dtype=np.float64)]
        owner = []
        for p in pts_cad:
            d = []
            for j in js:
                a, e = part.bends[j].locus("oml_virtual_sharp")
                dvec = unit(e - a)
                rel = p - a
                d.append(float(np.linalg.norm(rel - (rel @ dvec) * dvec)))
            owner.append(int(np.argmin(d)))
        if owner[0] == owner[1]:
            meas[cid] = Measurement.unknown("AMBIGUOUS_SEGMENTATION", "GENERAL_BEND_PAIR")
            continue
        out = []
        sig = []
        for p, oi in zip(pts_cad, owner):
            j, bm = js[oi], bms[oi]
            bend = part.bends[j]
            parent = art.panels[bend.parent]
            a, e = bend.locus("oml_virtual_sharp")
            dn = unit(e - a)
            q = a + ((p - a) @ dn) * dn
            off = p - q
            # nominal bend frame at the final pose: parent in-plane outward and parent OML normal
            w_nom = bend.p_out
            n_nom = unit(np.cross(dn, w_nom))
            c1, c2 = float(off @ w_nom), float(off @ n_nom)
            A = parent.A if parent.A is not None else np.eye(4)
            p_reg = apply(A, p[None])[0]
            lp, ld = bm.line_point, bm.line_dir
            qm = lp + ((p_reg - lp) @ ld) * ld
            w_m = bm.w_parent
            n_reg = A[:3, :3] @ n_nom
            n_m = n_reg - float(n_reg @ ld) * ld - float(n_reg @ w_m) * w_m
            n_m = unit(n_m) if float(np.linalg.norm(n_m)) > 1e-9 else unit(np.cross(ld, w_m))
            out.append(qm + c1 * w_m + c2 * n_m)
            sig.append(math.hypot(sharp_unit_along(bm, True, cfg), sharp_unit_along(bm, False, cfg)))
        dist = float(np.linalg.norm(out[1] - out[0]))
        comps = {"sharp_mm": math.hypot(*sig)}
        meas[cid] = Measurement(value=dist, u=u_budget("length", comps, cfg), method="GENERAL_BEND_PAIR_OML_TRANSPORT",
                                components=comps)
        anchors[cid] = (out[0], out[1])
    return meas, anchors, na


def _order_consistent(part: PartView, states: list[str]) -> bool:
    """The contract's order rule (``derive_stage_decision``): no unformed bend ranks below a formed one."""
    formed = [part.bends[j].rank for j, s in enumerate(states) if s in ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED")]
    unformed = [part.bends[j].rank for j, s in enumerate(states) if s == "NOT_YET_FORMED"]
    if "UNEXPECTED_FORMED" in states:
        return False
    return not (formed and unformed and min(unformed) < max(formed))


def _pose_sigma(fit: InPlaneFit) -> float:
    """Standard in-plane position uncertainty of the datum frame (mm) at the outline scale."""
    cov = fit.translation_cov
    if not np.all(np.isfinite(cov)):
        return 1.0
    ev = np.linalg.eigvalsh(cov)
    s_t = math.sqrt(max(0.0, float(ev.max())))
    s_r = fit.rotation_sigma_rad * fit.length_scale * 2.0
    return float(math.hypot(s_t, s_r))


def _k_sigma(part: PartView, stage_id: str, j: int, side: str, crossed: list[int], nominal: float,
             note: Callable[[str], None] | None = None, k: int | None = None) -> float:
    """``sigma_K |dN/dK|`` of the backgauge nominal of bend ``j`` at ``stage_id``, from hashed fields only.

    A flat strip of a not-yet-formed bend is ``sweep (Ri + K t)`` wide, so a
    station that crosses it on its way to the far boundary gains ``sweep t /
    |cos phi|`` per unit of K (``phi`` between the station and the strip's
    width direction). The stations are laid along bend ``j``'s line of the
    compiled stage model, toward the side's stage-pose outward direction; a
    strip counts when at least half of them cross it short of the nominal far
    boundary (the median the nominal is). ``sigma_K`` is the compiled
    ``k_factor.uncertainty``. E1's ``diagnostics.backgauge_nominals`` carry a
    numerical derivative too, but ``diagnostics`` is outside
    ``compiled_part_sha256``: it is compared here (a disagreement is logged) and
    never used, so an edited diagnostics block cannot change a decision.
    """
    stage = part.stages.get(stage_id)
    bend = part.bends[j]
    dndk = 0.0
    if stage is not None and bend.uid in stage.bend_lines and crossed:
        line = stage.bend_lines[bend.uid]
        q0, q1 = np.asarray(line.start_mm, dtype=np.float64), np.asarray(line.end_mm, dtype=np.float64)
        dj = unit(q1 - q0)
        rot = part.stage_transform(stage_id, bend.side_panel[side])[:3, :3]
        w = rot @ bend.side_out[side]
        w = unit(w - float(w @ dj) * dj)
        stations = [q0 + (q1 - q0) * f for f in np.linspace(0.05, 0.95, 11)]
        for c in crossed:
            bc = part.bends[c]
            lc = stage.bend_lines.get(bc.uid)
            if lc is None:
                continue
            c0, c1 = np.asarray(lc.start_mm, dtype=np.float64), np.asarray(lc.end_mm, dtype=np.float64)
            dc = c1 - c0
            if float(np.linalg.norm(dc)) < 1e-9:
                continue
            a_mat = np.stack([w, -dc], axis=1)
            hits = 0
            for o in stations:
                (lam, mu), *_ = np.linalg.lstsq(a_mat, c0 - o, rcond=None)
                if 0.0 <= mu <= 1.0 and 0.0 < lam < nominal + 1.0:
                    hits += 1
            if hits * 2 < len(stations):
                continue
            du = unit(dc)
            width_dir = w - float(w @ du) * du
            cos_phi = float(np.linalg.norm(width_dir))
            dndk += bc.sweep * bc.t / max(cos_phi, 0.2)
    value = float(dndk * part.k_sigma)
    if note is not None and k is not None:
        diag = (part.compiled.diagnostics or {}).get("backgauge_nominals", {}) or {}
        rec = diag.get(f"S{k}") if isinstance(diag, dict) else None
        ref = rec.get("k_sigma_mm") if isinstance(rec, dict) else None
        if isinstance(ref, (int, float)) and abs(float(ref) - value) > max(0.01, 0.1 * value):
            note(f"backgauge S{k}: k_sigma {value:.4f} mm from hashed fields; E1 diagnostics say {float(ref):.4f} "
                 f"(not used)")
    return value


def _heatmap(reg_pts: np.ndarray, part: PartView, stage_id: str, cfg: InspectConfig) -> tuple[np.ndarray, float]:
    cloud = part.cloud(stage_id)
    d, i = cloud.tree.query(reg_pts, k=1, distance_upper_bound=cfg.heatmap_max_distance_mm, workers=1)
    ok = np.isfinite(d)
    dev = np.full(len(reg_pts), np.nan, dtype=np.float64)
    ii = i[ok]
    diff = reg_pts[ok] - cloud.points[ii]
    signed = np.einsum("ij,ij->i", diff, cloud.normals[ii])
    lateral = np.linalg.norm(diff - signed[:, None] * cloud.normals[ii], axis=1)
    good = lateral <= cfg.heatmap_max_lateral_mm
    vals = np.where(good, signed, np.nan)
    dev[np.nonzero(ok)[0]] = vals
    # surface coverage: share of stage samples with a scan point within 1 mm
    tree = cKDTree(reg_pts[:: max(1, len(reg_pts) // 400_000)], balanced_tree=False, compact_nodes=False)
    probe = cloud.points[:: max(1, len(cloud.points) // 100_000)]
    dd, _ = tree.query(probe, k=1, distance_upper_bound=1.0, workers=1)
    return dev, float(np.mean(np.isfinite(dd)))


def _segment_labels(part: PartView, art: ArticulatedFit, vox: VoxelCloud, reg_pts: np.ndarray, geoms,
                    creases: list["_Crease"], ambiguous: bool) -> np.ndarray:
    """Per-point labels (ENGINE_IO §4.1): ``1 + panel``, ``1001 + bend``, 65535 unplanned crease, 0 otherwise.

    Under an ambiguous pose only the datum (which every twin maps onto itself)
    is labelled: a flange label would be the confident identity the contract
    forbids on a symmetric part.
    """
    lab_vox = np.zeros(len(vox.points), dtype=np.uint16)
    owner = art.owner
    if ambiguous:
        lab_vox[owner == part.datum] = np.uint16(part.datum + 1)
        return lab_vox[vox.inverse]
    lab_vox[owner >= 0] = (owner[owner >= 0] + 1).astype(np.uint16)
    for c in creases:
        free = lab_vox[c.voxels] == 0
        lab_vox[c.voxels[free]] = np.uint16(65535)
    labels = lab_vox[vox.inverse]
    for j, bm in geoms.items():
        if bm is None or not bm.ok or bm.line_point is None:
            continue
        bend = part.bends[j]
        rel = reg_pts - bm.line_point
        z = rel @ bm.line_dir
        perp = np.linalg.norm(rel - np.outer(z, bm.line_dir), axis=1)
        reach = bend.ro * (1.0 / math.cos(math.radians(min(bm.sweep_deg or 90.0, 170.0)) / 2.0)) + 0.5
        sel = (z > bm.visible[0] - 1) & (z < bm.visible[1] + 1) & (perp < reach) & ((labels == 0) | (labels == 65535))
        labels[sel] = 1001 + j
    return labels


@dataclass(eq=False)
class _Crease:
    model: UnplannedCreaseV1
    #: index of the fitted panel the crease hinges on
    panel: int
    #: unit direction of the crease line
    direction: np.ndarray
    #: OML virtual sharp (point, unit direction) of the crease, when both skins of the creased
    #: segment were seen; and its standard uncertainty along the panel (mm)
    sharp: tuple[np.ndarray, np.ndarray] | None = None
    sharp_u: float = 0.0
    #: voxel indices of the creased segment (segment label 65535)
    voxels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _crease_sharp(seg: np.ndarray, panel, t: float, contact: np.ndarray, cfg: InspectConfig, u_t: float = 0.0
                  ) -> tuple[tuple[np.ndarray, np.ndarray], float] | None:
    """OML virtual sharp of a crease: the convex skin's planes of the panel and the creased segment.

    Skin 0 continues across a crease (as across every bend); a segment bending
    toward ``-n0`` has skin 0 outside. With one skin of the segment seen, that
    skin is the one whose crease line lies where the panel's material (``contact``)
    ends: within 30 % of the two candidate lines' separation of it, and nearer it
    than the other by at least 25 % of that separation; otherwise None (the
    split flange is then UNKNOWN, never measured to the wrong line).
    """
    if len(seg) < 60:
        return None
    pl = robust_plane(seg, iterations=2)
    n = pl.n if float(pl.n @ panel.n0) > 0 else -pl.n
    dd = seg @ n - float(np.median(seg @ n))
    top, bottom = skin_peaks(dd, t)
    tol = min(cfg.grow_plane_tol_mm, 0.4 * t)
    down = float(np.mean(seg @ panel.n0) - (panel.d0 - 0.5 * t)) < 0.0
    oml_skin = 0 if down else 1

    def skin_plane(level: float):
        pts = seg[np.abs(dd - level) < tol]
        if len(pts) < 30:
            return None
        sp = robust_plane(pts, iterations=3)
        sn = sp.n if float(sp.n @ n) > 0 else -sp.n
        return sn, float(sn @ pts.mean(axis=0)), sp.rms

    moved = 0.0
    if bottom is not None:
        fit = skin_plane(top if oml_skin == 0 else bottom)
        if fit is None:
            return None
        sn, sd, rms = fit
    else:
        fit = skin_plane(top)
        if fit is None or len(contact) < 5:
            return None
        sn, sd, rms = fit
        lines = [_plane_line(sn, sd, panel.n0, panel.d0 - k * t) for k in (0, 1)]
        if lines[0] is None or lines[1] is None:
            return None
        e = np.cross(panel.n0, lines[0][1])
        e = e / max(float(np.linalg.norm(e)), 1e-12)
        if float((seg.mean(axis=0) - lines[0][0]) @ e) < 0:
            e = -e
        x_end = float(np.max((contact - lines[0][0]) @ e))
        x = [0.0, float((lines[1][0] - lines[0][0]) @ e)]
        miss = [abs(xk - x_end) for xk in x]
        # the panel's material must end at one of the two lines, clearly nearer it than the other
        sep = abs(x[1] - x[0])
        if abs(miss[0] - miss[1]) < 0.25 * sep or min(miss) > 0.3 * sep:
            return None
        seen = int(np.argmin(miss))
        if seen != oml_skin:  # the other skin of the segment, one thickness along its normal
            sd = sd - t if seen == 0 else sd + t
            moved = u_t
    line = _plane_line(sn, sd, panel.n0, panel.d0 - oml_skin * t)
    if line is None:
        return None
    s_ang = math.sqrt(max(0.0, 1.0 - float(sn @ panel.n0) ** 2))
    return line, math.hypot(max(rms, cfg.u_plane_rms_floor_mm), moved) / max(s_ang, 0.05)


def _creases(part: PartView, art: ArticulatedFit, vox: VoxelCloud, vox_reg: np.ndarray, vox_nrm: np.ndarray,
             reg_pts: np.ndarray, cfg: InspectConfig, thick: Thickness | None = None) -> list[_Crease]:
    """Planar unexplained regions hinged on a fitted panel at an angle (an extra bend or a crease)."""
    owned = art.owner >= 0
    region_any = np.zeros(len(vox_reg), dtype=bool)
    for p in art.panels:
        if len(p.region):
            region_any[p.region] = True
    planar = vox.spread < cfg.planar_spread(part.t, vox.slab)
    cand = planar & ~owned & ~region_any
    # exclude the bend zones of every bend (cylinders and strips)
    for bf, bend in zip(art.bends, part.bends):
        if bf.frame is None:
            continue
        rel = vox_reg - bf.frame.o
        z = rel @ bf.frame.a
        r = np.linalg.norm(rel - np.outer(z, bf.frame.a), axis=1)
        cand &= ~((z > bf.frame.z0 - 3) & (z < bf.frame.z1 + 3) & (r < bf.frame.band + 2.0))
    if cand.sum() < 20:
        return []
    from scipy.sparse.csgraph import connected_components

    idx = np.nonzero(cand)[0]
    sub = vox.graph[idx][:, idx]
    _, lab = connected_components(sub, directed=False)
    owned_idx = np.nonzero(owned)[0]
    owned_tree = cKDTree(vox_reg[owned_idx]) if len(owned_idx) else None
    out = []
    area_px = vox.voxel * vox.voxel
    for comp in _crease_groups(idx, lab, vox_reg, area_px, part.t, cfg):
        pts = vox_reg[comp]
        pl = robust_plane(pts, iterations=2)
        # thin strips (cut walls) are not creases
        c = pts - pts.mean(axis=0)
        ev = np.sort(np.linalg.eigvalsh(c.T @ c / len(c)))
        if math.sqrt(max(ev[1], 0.0)) * 3.46 < max(4.0, 2.5 * part.t):
            continue
        # the fitted panel it hinges on: the panel owning most voxels next to it (the crease's own
        # curved zone between them is not owned by anyone)
        near = owned_tree.query_ball_point(pts, r=cfg.crease_hinge_search_mm) if owned_tree is not None else []
        hits = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in near])) if len(near) else np.zeros(0, int)
        if len(hits) == 0:
            continue
        owners = art.owner[owned_idx[hits]]
        pidx = int(np.bincount(owners).argmax())
        panel = art.panels[pidx]
        if panel.n0 is None:
            continue
        ang = math.degrees(math.acos(min(1.0, abs(float(pl.n @ panel.n0)))))
        if ang < cfg.crease_min_angle_deg:
            continue
        # crease line: the contact points projected on the intersection line
        line = _plane_line(pl.n, pl.d, panel.n0, panel.d0 - 0.5 * (panel.t or part.t))
        if line is None:
            continue
        contact = vox_reg[owned_idx[hits][owners == pidx]]
        if len(contact) < 2:
            contact = pts
        seg = reg_pts[np.isin(vox.inverse, comp)]
        # a crease takes the material out of the panel's slab; an unowned strip of the panel itself
        # (a blurred edge zone, both skins and the wall in a few mm) fits a tilted plane but stays inside
        t_p = float(panel.t or part.t)
        hgt = seg @ panel.n0 - panel.d0
        departure = float(np.quantile(np.maximum(hgt, -t_p - hgt), 0.9)) if len(seg) else 0.0
        if departure < max(cfg.crease_min_departure_mm, 0.5 * t_p):
            continue
        sharp = _crease_sharp(seg, panel, t_p, contact, cfg, 0.0 if thick is None else thick.u_mm)
        # the crease is drawn on its OML virtual sharp (like a bend), else on the mid-plane line
        p0, dvec = sharp[0] if sharp is not None else line
        s = (contact - p0) @ dvec
        a, b = p0 + dvec * float(s.min()), p0 + dvec * float(s.max())
        if float(np.linalg.norm(b - a)) < 1.0:
            continue
        model = UnplannedCreaseV1(crease_id=f"C{len(out) + 1}", reason="EXTRA_BEND_SUSPECTED",
                                  polyline_mm=(tuple(round6(float(x)) for x in a), tuple(round6(float(x)) for x in b)),
                                  included_deg=float(min(180.0, max(1e-3, 180.0 - ang))),
                                  note=f"planar region of {len(comp) * area_px:.0f} mm2 at {ang:.1f} deg to "
                                       f"{part.panels[pidx].panel_id}"
                                       + ("" if sharp is not None else "; OML sharp unresolved"))
        out.append(_Crease(model=model, panel=pidx, direction=dvec, sharp=None if sharp is None else sharp[0],
                           sharp_u=0.0 if sharp is None else sharp[1], voxels=np.asarray(comp, dtype=np.int64)))
        if len(out) >= 16:
            break
    return out


def _crease_groups(idx: np.ndarray, lab: np.ndarray, vox_reg: np.ndarray, area_px: float, t: float,
                   cfg: InspectConfig) -> list[np.ndarray]:
    """Candidate crease regions: connected components, the two skins of one segment merged.

    A thick sheet's skins are often separate components (the cut wall between them
    is sparse); merged, the region's plane is the segment's mid-plane, so its crease
    line and OML sharp are not offset by ``t / sin(angle)``.
    """
    comps = []
    for lb in np.unique(lab):
        comp = idx[lab == lb]
        if len(comp) * area_px < 0.5 * cfg.crease_min_area_mm2 or len(comp) < 6:
            continue
        pts = vox_reg[comp]
        pl = robust_plane(pts, iterations=2)
        c = pts - pts.mean(axis=0)
        extent = 3.46 * math.sqrt(max(float(np.linalg.eigvalsh(c.T @ c / len(c)).max()), 0.0))
        comps.append((comp, pl, pts.mean(axis=0), extent))
    parent = list(range(len(comps)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            _, pi, ci, ei = comps[i]
            _, pj, cj, ej = comps[j]
            if abs(float(pi.n @ pj.n)) < math.cos(math.radians(3.0)):
                continue
            gap = abs(float(pi.n @ (cj - ci)))
            lateral = float(np.linalg.norm((cj - ci) - pi.n * float(pi.n @ (cj - ci))))
            if 0.5 * t <= gap <= 1.6 * t and lateral <= 0.5 * max(ei, ej):
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(comps)):
        groups.setdefault(root(i), []).append(i)
    out = []
    for members in groups.values():
        biggest = max(len(comps[i][0]) for i in members)
        if biggest * area_px < cfg.crease_min_area_mm2:
            continue
        out.append(np.concatenate([comps[i][0] for i in members]))
    return out


def _plane_line(n1, d1, n2, d2):
    d = np.cross(n1, n2)
    s = float(np.linalg.norm(d))
    if s < 1e-6:
        return None
    d = d / s
    m = np.array([n1, n2, d])
    return np.linalg.solve(m, np.array([d1, d2, 0.0])), d
