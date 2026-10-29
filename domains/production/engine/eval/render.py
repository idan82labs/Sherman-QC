"""G4 overlay renderer (plan §5 G4, §10 amendment 8): matplotlib/Agg, CPU only.

For one inspected scan it renders, in the engine frame (``cad_model_mm``):

- **Global views**: ISO plus three orthographic views (along the part's
  principal axes, so the views are informative whatever the CAD placement) of
  the registered scan, points coloured by segment and shaded by local normals,
  every planned bend's measured line drawn and labelled ``<seq> · <label> ·
  <uid6>``; a DIMENSIONS view with the flange-length arrows (along the flange
  direction fitted to the scan points at mid-span, nominal when too few
  points); a HEATMAP view of the signed deviations (legend ±band, same
  colours). With ``show_values=False`` (every reviewer packet) the DIMENSIONS
  view names the sides only and prints no length.
- **Per-bend close-ups** at an exact pixel scale (default 10 px/mm, never
  below 8) with a 10 mm scale bar, one image per bend (long bends are tiled):

  * SCAN panel -- camera looking at the outside of the bend along its outer
    bisector. The measured line is drawn **solid on the outer apex** (on the
    material; the apex is computed at the *measured* included angle,
    :func:`.engine_output.measured_outer_apex`; without a measured angle the
    solid line is the OML sharp and the header says so), the measured OML
    virtual sharp **dashed**, the nominal stage-model line **grey dashed**;
    points coloured by segment.
  * RECIPE CAD panel -- the decoded stage model's display mesh from the same
    camera and window, labelled from the recipe (never from the engine), so a
    reviewer can compare labels at the same spot (TAG_OK).
  * SECTION panel -- the profile at mid-span (looking along the bend axis):
    scan points, the CAD section, the measured apex and OML points, and the
    flange-length arrows from the measured OML sharp to the fitted far edge
    (direction fitted to the section's flange points; nominal if too few).

Reviewer images carry no metrics, decisions or nominal values (amendment 8:
reviewers "see no metrics"): packets render the global views with
``show_values=False``.

Canaries (:class:`Corruption`) are applied at render time by :mod:`.g4`: a
LINE_SHIFT moves the subject's measured lines (solid and dashed, and the
section markers) across the bend in the view plane, by more than the bend's
LANDS tolerance; a LABEL_SWAP exchanges the labels of the subject and its
partner bend **everywhere in the SCAN panel** (so the image never shows two
identical labels), and only the RECIPE CAD panel, which always stays
truthful, reveals the mismatch.

Point labels come from the engine's ``segment_labels.u16`` when present;
otherwise from the nearest point of the decoded stage model's sample cloud
(stated on the image as the label source).
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contracts.production.compiled_part import CompiledPartV1, StageModelV1
from contracts.production.part_recipe import PartRecipeV1

from .engine_output import (
    SEGMENT_BEND_BASE,
    SEGMENT_PANEL_BASE,
    SEGMENT_UNPLANNED,
    decoded_stage,
    load_compiled,
    load_engine_output,
    load_recipe,
    measured_outer_apex,
    row_characteristic,
    stage_locus,
    stage_transform,
)
from .plyio import read_ply, sha256_file
from .thresholds import G4_THRESHOLDS

__all__ = [
    "RENDER_VERSION",
    "SideOverlay",
    "BendOverlay",
    "RenderScene",
    "Corruption",
    "CloseupPlan",
    "scene_from_engine_output",
    "plan_closeups",
    "render_global_views",
    "render_closeup",
    "display_label",
    "fit_flange_direction",
]

RENDER_VERSION = "e4a-render-1.1.0"
_DPI = 100
_MAX_TILE_MM = 150.0
_LINE = "#111111"
_NOMINAL = "#808080"
_UNPLANNED = "#d0309a"
_UNASSIGNED = (0.80, 0.80, 0.80)
_PANEL_COLOURS = [
    (0.31, 0.47, 0.67), (0.35, 0.63, 0.31), (0.47, 0.37, 0.62), (0.18, 0.60, 0.62), (0.55, 0.55, 0.30),
    (0.40, 0.40, 0.75), (0.25, 0.50, 0.45), (0.60, 0.42, 0.55), (0.30, 0.58, 0.80), (0.50, 0.65, 0.40),
]
_BEND_COLOURS = [(0.93, 0.56, 0.17), (0.86, 0.38, 0.20), (0.95, 0.72, 0.25), (0.80, 0.45, 0.10)]


def display_label(seq: int, label: str, uid: str) -> str:
    """The on-image identity of a bend: ``<seq> · <label> · <uid6>``."""
    return f"{seq} · {label} · {uid[3:9]}"


@dataclass(frozen=True)
class SideOverlay:
    side: str
    outward: np.ndarray  # unit, stage pose
    actual_mm: float | None
    nominal_mm: float | None


@dataclass(frozen=True)
class BendOverlay:
    bend_uid: str
    label: str
    sequence_index: int
    state: str
    state_reasons: tuple[str, ...]
    measured_oml: np.ndarray | None
    measured_apex: np.ndarray | None
    nominal_line: np.ndarray
    nominal_apex: np.ndarray | None
    axis_dir: np.ndarray
    outward: np.ndarray  # view direction from the material to the camera (outer bisector or OML skin normal)
    inner_radius_mm: float
    thickness_mm: float
    sides: tuple[SideOverlay, ...]
    #: BEND_LINE_POSITION reason when it is not MEASURED (e.g. OCCLUDED, REGISTRATION_UNTRUSTED).
    position_reason: str | None = None
    #: How the solid line was placed (:data:`.engine_output.APEX_SOURCES`); None without a line.
    apex_source: str | None = None

    @property
    def text(self) -> str:
        return display_label(self.sequence_index, self.label, self.bend_uid)

    @property
    def lands_tolerance_mm(self) -> float:
        """The G4 LANDS tolerance of this bend: max(2 mm, Ri + t)."""
        return max(float(G4_THRESHOLDS["lands_rule"]["offset_floor_mm"]), self.inner_radius_mm + self.thickness_mm)


@dataclass
class RenderScene:
    title: str
    points: np.ndarray | None
    labels: np.ndarray | None
    label_source: str
    deviations: np.ndarray | None
    heatmap_band_mm: float
    stage_model_id: str | None
    stage_vertices: np.ndarray | None
    stage_faces: np.ndarray | None
    bends: list[BendOverlay]
    #: (seq, label, uid, nominal line at the stage) from the recipe/compiled part.
    cad_labels: list[tuple[int, str, str, np.ndarray]]
    validity_state: str
    summary_state: str
    spacing_mm: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)
    #: E1's ambiguous_bend_groups of the decoded stage model (amendment 8: the only
    #: justification of AMBIGUOUS). Sealed-key data; never drawn.
    ambiguous_groups: tuple[tuple[str, ...], ...] = ()
    pose_ambiguous: bool = False

    def bend(self, uid: str) -> BendOverlay:
        return next(b for b in self.bends if b.bend_uid == uid)

    def in_ambiguous_group(self, uid: str) -> bool:
        return any(uid in group for group in self.ambiguous_groups)


@dataclass(frozen=True)
class Corruption:
    """A canary corruption of one close-up (applied at render time).

    LINE_SHIFT moves the subject's measured lines by ``shift_mm`` across the bend.
    LABEL_SWAP draws ``label_map[uid]`` instead of the true label of every bend it
    names (the subject and its partner), in the whole SCAN panel and the header."""

    kind: str  # LINE_SHIFT | LABEL_SWAP
    shift_mm: float = 0.0
    label_map: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CloseupPlan:
    bend_uid: str
    tile_index: int
    tile_count: int
    #: along-axis range (mm, in the bend's line parameter) covered by this tile
    s0: float
    s1: float
    px_per_mm: float


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rot(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return m[:3, :3] @ np.asarray(v, dtype=np.float64)


def _bend_overlay(compiled: CompiledPartV1, stage: StageModelV1, row: Any | None, bend: Any) -> BendOverlay:
    uid = bend.bend_uid
    t_parent = stage_transform(stage, bend.parent_panel_id)
    formed_at_stage = uid in set(stage.formed_bend_uids)
    nominal_line = np.asarray([stage.bend_lines[uid].start_mm, stage.bend_lines[uid].end_mm], dtype=np.float64)
    axis_dir = _unit(nominal_line[1] - nominal_line[0])
    nominal_apex = None
    if formed_at_stage:
        apex = stage_locus(compiled, stage, uid, "outer_apex")
        cyl = stage_locus(compiled, stage, uid, "cylinder_axis")
        nominal_apex = apex
        w = apex.mean(axis=0) - cyl.mean(axis=0)
        w = _unit(w - axis_dir * float(w @ axis_dir))
    else:
        panel = next(p for p in compiled.panels if p.panel_id == bend.parent_panel_id)
        n0 = np.asarray(panel.skin0_plane.normal, dtype=np.float64)
        n = n0 if bend.oml_skin == 0 else (np.asarray(panel.skin1_plane.normal) if panel.skin1_plane else -n0)
        w = _unit(_rot(t_parent, n))
        w = _unit(w - axis_dir * float(w @ axis_dir))
    measured_oml = measured_apex = None
    apex_source = position_reason = None
    state, reasons = "UNKNOWN", ()
    sides = []
    if row is not None:
        state, reasons = row.state, tuple(row.state_reasons)
        position = row_characteristic(row, "BEND_LINE_POSITION")
        if position is not None and position.state != "MEASURED":
            position_reason = position.reason
        if row.display.measured_polyline_mm is not None:
            measured_oml = np.asarray([row.display.measured_polyline_mm[0], row.display.measured_polyline_mm[-1]],
                                      dtype=np.float64)
            measured_apex, apex_source = measured_outer_apex(compiled, stage, row, fallback="OML")
        for side in bend.sides:
            ch = row_characteristic(row, "FLANGE_LENGTH", side.side)
            t_side = stage_transform(stage, side.panel_id)
            sides.append(SideOverlay(side.side, _unit(_rot(t_side, side.outward_dir)),
                                     ch.actual if ch is not None and ch.state == "MEASURED" else None,
                                     None if ch is None else ch.expected))
    return BendOverlay(
        bend_uid=uid, label=bend.label, sequence_index=bend.sequence_index, state=state, state_reasons=reasons,
        measured_oml=measured_oml, measured_apex=measured_apex, nominal_line=nominal_line, nominal_apex=nominal_apex,
        axis_dir=axis_dir, outward=w, inner_radius_mm=bend.inner_radius_mm, thickness_mm=bend.thickness_mm,
        sides=tuple(sides), position_reason=position_reason, apex_source=apex_source)


def _nearest_labels(points: np.ndarray, cloud: Mapping[str, np.ndarray]) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(cloud["points"], dtype=np.float64))
    dist, idx = tree.query(points, k=1)
    kind = np.asarray(cloud["segment_kind"])[idx]
    seg = np.asarray(cloud["segment_index"])[idx].astype(np.int64)
    out = np.where(kind == 1, SEGMENT_BEND_BASE + seg, SEGMENT_PANEL_BASE + seg).astype(np.uint16)
    out[dist > 3.0] = 0
    return out


def scene_from_engine_output(engine_dir: str | os.PathLike, compiled_json: str | os.PathLike,
                             recipe_json: str | os.PathLike, *, title: str = "") -> RenderScene:
    """Build a :class:`RenderScene` from an E3 output directory (ENGINE_IO §4.1)."""
    compiled = load_compiled(compiled_json)
    recipe: PartRecipeV1 = load_recipe(recipe_json)
    eo = load_engine_output(engine_dir, compiled=compiled, recipe=recipe)
    if eo.result is None:
        raise ValueError(f"{engine_dir}: no valid result ({eo.status}: {eo.error})")
    result = eo.result
    stage = decoded_stage(result, compiled)
    rows = {r.bend_uid: r for r in result.bends}
    bends = [_bend_overlay(compiled, stage, rows.get(b.bend_uid), b) for b in compiled.bends]
    points = eo.registered_points()
    labels = deviations = None
    source = "NONE"
    spacing = 0.5
    if points is not None and len(points):
        labels = eo.segment_labels(len(points))
        source = "ENGINE"
        if labels is None:
            cloud_path = Path(compiled_json).parent / stage.sample_cloud.rel_path
            with np.load(cloud_path, allow_pickle=False) as z:
                labels = _nearest_labels(points, {k: z[k] for k in z.files})
            source = "NEAREST_STAGE_MODEL"
        deviations = eo.deviations(len(points))
        from scipy.spatial import cKDTree

        sub = points[np.unique(np.linspace(0, len(points) - 1, min(len(points), 20000)).astype(np.int64))]
        d, _ = cKDTree(points).query(sub, k=2)
        spacing = float(np.median(d[:, 1])) or 0.5
    mesh = read_ply(Path(compiled_json).parent / stage.display_mesh.rel_path)
    cad_labels = [(b.sequence_index, b.label, b.bend_uid,
                   np.asarray([stage.bend_lines[b.bend_uid].start_mm, stage.bend_lines[b.bend_uid].end_mm]))
                  for b in compiled.bends]
    # AMBIGUOUS is justified only by E1's symmetry test (amendment 8): the orbits of the decoded
    # stage model. When the engine decoded no stage (an ambiguous pose), the stage model whose formed
    # set equals the bends it did not report NOT_YET_FORMED; only if none matches, any stage model.
    if result.stage_decision.stage_model_id is not None:
        ref = stage
    else:
        maybe_formed = {r.bend_uid for r in result.bends if r.state != "NOT_YET_FORMED"}
        ref = next((st for st in compiled.stages if set(st.formed_bend_uids) == maybe_formed), None)
    if ref is not None:
        groups = tuple(tuple(g) for g in ref.symmetry.ambiguous_bend_groups)
        ambiguity_stage = ref.stage_model_id
    else:
        groups = tuple(dict.fromkeys(tuple(g) for st in compiled.stages for g in st.symmetry.ambiguous_bend_groups))
        ambiguity_stage = "ANY_STAGE"
    return RenderScene(
        title=title or f"{result.sku} {result.recipe.recipe_id}", points=points, labels=labels, label_source=source,
        deviations=deviations, heatmap_band_mm=float(result.viewer_assets.heatmap_band_mm),
        stage_model_id=result.stage_decision.stage_model_id, stage_vertices=mesh.vertices, stage_faces=mesh.faces,
        bends=bends, cad_labels=cad_labels, validity_state=result.inspection_validity.state,
        summary_state=result.summary.state, spacing_mm=spacing,
        meta={"engine_version": result.engine.engine_version, "stage_used": stage.stage_model_id,
              "ambiguity_stage": ambiguity_stage},
        ambiguous_groups=groups,
        pose_ambiguous=bool(result.inspection_validity.registration.pose_ambiguous),
    )


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _figure(width_px: int, height_px: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_px / _DPI, height_px / _DPI), dpi=_DPI, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _axes(fig, x0: int, y0: int, w: int, h: int, W: int, H: int):
    """Axes at an exact pixel rectangle (origin top-left)."""
    ax = fig.add_axes((x0 / W, 1.0 - (y0 + h) / H, w / W, h / H))
    ax.set_axis_off()
    return ax


def _segment_rgb(labels: np.ndarray | None, n: int) -> np.ndarray:
    rgb = np.tile(np.asarray(_UNASSIGNED), (n, 1))
    if labels is None:
        return rgb
    lab = labels.astype(np.int64)
    panel = (lab >= SEGMENT_PANEL_BASE) & (lab < SEGMENT_BEND_BASE)
    bend = (lab >= SEGMENT_BEND_BASE) & (lab < SEGMENT_UNPLANNED)
    pc = np.asarray(_PANEL_COLOURS)
    bc = np.asarray(_BEND_COLOURS)
    rgb[panel] = pc[(lab[panel] - SEGMENT_PANEL_BASE) % len(pc)]
    rgb[bend] = bc[(lab[bend] - SEGMENT_BEND_BASE) % len(bc)]
    rgb[lab == SEGMENT_UNPLANNED] = (0.82, 0.19, 0.60)
    return rgb


def _normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Unit normals from the covariance of the k nearest neighbours (sign-free)."""
    from scipy.spatial import cKDTree

    if len(points) < k + 1:
        return np.tile([0.0, 0.0, 1.0], (len(points), 1))
    _, idx = cKDTree(points).query(points, k=k)
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]


def _shade(rgb: np.ndarray, normals: np.ndarray, view: np.ndarray) -> np.ndarray:
    lam = np.abs(normals @ view)
    return np.clip(rgb * (0.40 + 0.60 * lam[:, None]), 0, 1)


def _basis(forward: np.ndarray, right_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed camera basis (right, up, forward); the camera looks along ``forward``."""
    f = _unit(np.asarray(forward, dtype=np.float64))
    r = np.asarray(right_hint, dtype=np.float64) - f * float(np.dot(right_hint, f))
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, [0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.cross(f, [1.0, 0.0, 0.0])
    r = _unit(r)
    u = _unit(np.cross(f, r))
    return r, u, f


def _scatter(ax, xy: np.ndarray, depth: np.ndarray, rgb: np.ndarray, size_px: float) -> None:
    order = np.argsort(-depth)  # far first (painter's algorithm)
    s = (max(size_px, 0.6) * 72.0 / _DPI) ** 2
    ax.scatter(xy[order, 0], xy[order, 1], s=s, c=rgb[order], marker="s", linewidths=0, rasterized=True)


def _halo(width: float = 4.5):
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=width, foreground="white")]


def _draw_line(ax, a: np.ndarray, b: np.ndarray, *, style: str, colour: str, width: float) -> None:
    ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=style, color=colour, linewidth=width,
            path_effects=_halo(width + 2.5), solid_capstyle="butt", zorder=5)


def _text(ax, x: float, y: float, text: str, *, size: float = 9.0, weight: str = "normal", colour: str = "#111111",
          box: str = "white") -> None:
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=colour, ha="center", va="center", zorder=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": box, "ec": "#555555", "lw": 0.6, "alpha": 0.95})


def _strip(fig, x0: int, y0: int, w: int, h: int, W: int, H: int, text: str, *, size: float = 10.5,
           weight: str = "bold") -> None:
    ax = _axes(fig, x0, y0, w, h, W, H)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.text(2, h / 2, text, fontsize=size, fontweight=weight, va="center")


def _scale_strip(fig, x0: int, y0: int, w: int, h: int, W: int, H: int, xlim: tuple[float, float],
                 bar_mm: float, note: str = "") -> None:
    """A strip under a panel with the same x scale, carrying the scale bar."""
    ax = _axes(fig, x0, y0, w, h, W, H)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    a = xlim[0] + (xlim[1] - xlim[0]) * 0.01
    ax.plot([a, a + bar_mm], [0.45, 0.45], color="#000000", linewidth=4.0, solid_capstyle="butt")
    for x in (a, a + bar_mm):
        ax.plot([x, x], [0.2, 0.7], color="#000000", linewidth=1.3)
    ax.text(a + bar_mm + (xlim[1] - xlim[0]) * 0.01, 0.45, f"{bar_mm:g} mm{note}", fontsize=10, va="center")


def _arrow(ax, a: np.ndarray, b: np.ndarray, *, colour: str = "#1d4ed8", width: float = 1.8) -> None:
    ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]), zorder=7,
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": width, "shrinkA": 0, "shrinkB": 0,
                            "mutation_scale": 15})


def _save(fig, path: Path, meta: Mapping[str, str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.png")
    fig.savefig(tmp, dpi=_DPI, metadata={"Software": RENDER_VERSION, **meta})
    os.replace(tmp, path)
    return sha256_file(path)


class _Labeler:
    """Greedy label placement: try candidate anchors until a box overlaps no earlier box.
    ``texts`` records every label drawn (audit trail of what a reviewer saw)."""

    def __init__(self, ax, px_per_unit: float) -> None:
        self.ax = ax
        self.k = px_per_unit
        self.boxes: list[tuple[float, float, float, float]] = []
        self.texts: list[str] = []

    def place(self, candidates: Sequence[np.ndarray], text: str, **style: Any) -> None:
        w = (len(text) * 7.4 * style.get("size", 9.0) / 9.0 + 14) / self.k
        h = 22 / self.k
        chosen = None
        for c in candidates:
            box = (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in self.boxes):
                chosen = (c, box)
                break
        if chosen is None:
            c = candidates[0]
            chosen = (c, (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2))
        self.boxes.append(chosen[1])
        self.texts.append(text)
        _text(self.ax, float(chosen[0][0]), float(chosen[0][1]), text, **style)


def _line_label_candidates(a2: np.ndarray, b2: np.ndarray, offset: float) -> list[np.ndarray]:
    d = b2 - a2
    n = np.array([-d[1], d[0]])
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    out = []
    for t in (0.5, 0.3, 0.7, 0.15, 0.85):
        p = a2 + t * d
        out += [p + n * offset, p - n * offset, p + n * 2.4 * offset, p - n * 2.4 * offset]
    return out


class _ZBuffer:
    """Coarse screen-space depth buffer of the drawn points (min depth per cell),
    used to draw overlay segments hidden behind material faded and dotted."""

    def __init__(self, xy: np.ndarray, depth: np.ndarray, cell_mm: float, tol_mm: float) -> None:
        self.cell = max(cell_mm, 1e-6)
        self.tol = tol_mm
        self.lo = xy.min(axis=0) - 1.0
        ij = np.floor((xy - self.lo) / self.cell).astype(np.int64)
        self.shape = (int(ij[:, 0].max()) + 2, int(ij[:, 1].max()) + 2)
        self.z = np.full(self.shape, np.inf)
        np.minimum.at(self.z, (ij[:, 0], ij[:, 1]), depth)

    def hidden(self, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
        ij = np.floor((xy - self.lo) / self.cell).astype(np.int64)
        inside = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        out = np.zeros(len(xy), dtype=bool)
        k = np.nonzero(inside)[0]
        out[k] = depth[k] > self.z[ij[k, 0], ij[k, 1]] + self.tol
        return out


def _draw_segment_occluded(ax, a3: np.ndarray, b3: np.ndarray, proj, depth_of, zbuf: "_ZBuffer | None", *,
                           style: str, colour: str, width: float, arrow: bool = False, samples: int = 80) -> None:
    """Draw a 3D segment; parts hidden behind scan points are drawn faded and dotted."""
    t = np.linspace(0.0, 1.0, samples)
    pts = a3[None, :] + t[:, None] * (b3 - a3)[None, :]
    xy = proj(pts)
    hid = zbuf.hidden(xy, depth_of(pts)) if zbuf is not None else np.zeros(samples, dtype=bool)
    start = 0
    for i in range(1, samples + 1):
        if i == samples or hid[i] != hid[start]:
            seg = xy[start:min(i + 1, samples)]
            if len(seg) >= 2:
                if hid[start]:
                    ax.plot(seg[:, 0], seg[:, 1], linestyle=":", color=colour, linewidth=max(1.0, width * 0.6),
                            alpha=0.55, zorder=4)
                else:
                    ax.plot(seg[:, 0], seg[:, 1], linestyle=style, color=colour, linewidth=width,
                            path_effects=_halo(width + 2.5), solid_capstyle="butt", zorder=5)
            start = i
    if arrow:
        _arrow(ax, xy[-2], xy[-1], colour=colour, width=width)


# ---------------------------------------------------------------------------
# global views
# ---------------------------------------------------------------------------


def _principal_axes(points: np.ndarray) -> np.ndarray:
    c = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(c[np.unique(np.linspace(0, len(c) - 1, min(len(c), 50000)).astype(np.int64))],
                             full_matrices=False)
    return vt  # rows: largest -> smallest variance


def _subsample(n: int, cap: int) -> np.ndarray:
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).astype(np.int64))


def fit_flange_direction(points: np.ndarray, origin: np.ndarray, nominal_dir: np.ndarray, axis_dir: np.ndarray,
                         length: float, *, slab_mm: float = 2.0, cone_deg: float = 25.0, min_points: int = 20
                         ) -> tuple[np.ndarray, bool]:
    """Direction of a flange seen from its bend line, fitted to the scan.

    Uses the points within ``slab_mm`` of the section plane through ``origin``
    (normal ``axis_dir``), 3 mm to ``length`` + 5 mm from ``origin`` and within
    ``cone_deg`` of ``nominal_dir``; their principal direction in the section
    plane, oriented like ``nominal_dir``. Returns ``(direction, fitted)``; the
    nominal direction (``fitted=False``) when fewer than ``min_points`` qualify."""
    a = _unit(np.asarray(axis_dir, dtype=np.float64))
    nd = np.asarray(nominal_dir, dtype=np.float64)
    nd = _unit(nd - a * float(nd @ a))
    if points is None or not len(points):
        return nd, False
    rel = np.asarray(points, dtype=np.float64) - origin
    along = rel @ a
    inplane = rel - np.outer(along, a)
    dist = np.linalg.norm(inplane, axis=1)
    cosang = (inplane @ nd) / np.maximum(dist, 1e-9)
    m = (np.abs(along) <= slab_mm) & (dist > 3.0) & (dist < length + 5.0) & (cosang > math.cos(math.radians(cone_deg)))
    if int(m.sum()) < min_points:
        return nd, False
    sel = inplane[m]
    _, _, vt = np.linalg.svd(sel - sel.mean(axis=0), full_matrices=False)
    d = _unit(vt[0] - a * float(vt[0] @ a))
    return (d if float(d @ nd) >= 0 else -d), True


def render_global_views(scene: RenderScene, out_dir: str | os.PathLike, *, stem: str = "global",
                        size_px: tuple[int, int] = (1600, 1200), max_points: int = 400_000,
                        show_values: bool = True) -> list[dict[str, Any]]:
    """Render ISO / TOP / FRONT / SIDE, DIMENSIONS and HEATMAP; returns image records.

    ``show_values=False`` (reviewer packets, amendment 8 "no metrics") labels the
    DIMENSIONS arrows with the side names only."""
    out = Path(out_dir)
    W, H = size_px
    head = 64
    records: list[dict[str, Any]] = []
    if scene.points is None or not len(scene.points):
        fig = _figure(W, H)
        ax = _axes(fig, 0, 0, W, H, W, H)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _text(ax, 0.5, 0.5, f"{scene.title}\nno registered scan ({scene.validity_state})", size=16)
        sha = _save(fig, out / f"{stem}_NO_SCAN.png", {"Title": "NO_SCAN"})
        return [{"kind": "GLOBAL_NO_SCAN", "file": f"{stem}_NO_SCAN.png", "sha256": sha}]
    pts = scene.points
    sel = _subsample(len(pts), max_points)
    p = pts[sel]
    normals = _normals(p)
    base_rgb = _segment_rgb(None if scene.labels is None else scene.labels[sel], len(p))
    e0, e1, e2 = _principal_axes(p)
    # cameras on the outside of the bends (the measured lines lie on the outer apex)
    out_sum = np.sum([b.outward for b in scene.bends if b.measured_apex is not None] or [np.zeros(3)], axis=0)
    e0, e1, e2 = (e if float(e @ out_sum) >= 0 else -e for e in (e0, e1, e2))
    iso = _unit(-(e0 + e1 + e2))
    centre = np.median(p, axis=0)
    views = [("ISO", iso, e0), ("TOP", -e2, e0), ("FRONT", -e1, e0), ("SIDE", -e0, e1),
             ("DIMENSIONS", iso, e0), ("HEATMAP", iso, e0)]
    for name, fwd, right in views:
        r, u, f = _basis(fwd, right)
        q = p - centre
        xy = np.column_stack([q @ r, q @ u])
        depth = q @ f
        lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
        scale = 1.0 / max((hi - lo)[0] / (W * 0.84), (hi - lo)[1] / ((H - head) * 0.80), 1e-9)  # px per mm
        fig = _figure(W, H)
        ax = _axes(fig, 0, head, W, H - head, W, H)
        mid = (lo + hi) / 2
        ax.set_xlim(mid[0] - W / scale / 2, mid[0] + W / scale / 2)
        ax.set_ylim(mid[1] - (H - head) / scale / 2, mid[1] + (H - head) / scale / 2)
        if name == "HEATMAP":
            rgb = _heat_rgb(None if scene.deviations is None else scene.deviations[sel], scene.heatmap_band_mm, len(p))
        else:
            rgb = _shade(base_rgb, normals, f)
        _scatter(ax, xy, depth, rgb, max(1.0, scene.spacing_mm * scale * 1.4))

        def proj(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x) - centre
            return np.column_stack([x @ r, x @ u])

        def depth_of(x: np.ndarray) -> np.ndarray:
            return (np.atleast_2d(x) - centre) @ f

        zbuf = _ZBuffer(xy, depth, cell_mm=3.0 / scale + scene.spacing_mm,
                        tol_mm=max(2.0, 4.0 * scene.spacing_mm))
        labeler = _Labeler(ax, scale)
        arrows_fitted = arrows_nominal = 0
        if name != "HEATMAP":
            for b in scene.bends:
                if b.measured_apex is not None:
                    line3, style, colour, width = b.measured_apex, "-", _LINE, 2.4
                    tag = b.text
                else:
                    line3, style, colour, width = b.nominal_line, "--", _NOMINAL, 1.6
                    tag = f"{b.text} ({b.state})"
                _draw_segment_occluded(ax, line3[0], line3[1], proj, depth_of, zbuf, style=style, colour=colour,
                                       width=width)
                a2 = proj(line3)
                if name == "DIMENSIONS" and b.measured_oml is not None:
                    o = b.measured_oml.mean(axis=0)
                    axis = _unit(b.measured_oml[1] - b.measured_oml[0])
                    for s in b.sides:
                        if s.actual_mm is None:
                            continue
                        direction, fitted = fit_flange_direction(pts, o, s.outward, axis, s.actual_mm)
                        arrows_fitted += int(fitted)
                        arrows_nominal += int(not fitted)
                        tip = o + direction * s.actual_mm
                        _draw_segment_occluded(ax, o, tip, proj, depth_of, zbuf, style="-", colour="#1d4ed8",
                                               width=1.8, arrow=True)
                        o2, t2 = proj(o)[0], proj(tip)[0]
                        text = f"{b.label}{s.side} {s.actual_mm:.2f} mm" if show_values else f"{b.label}{s.side}"
                        labeler.place([o2 + 0.62 * (t2 - o2), o2 + 0.45 * (t2 - o2), o2 + 0.8 * (t2 - o2)],
                                      text, size=8.5, box="#eef4ff")
                labeler.place(_line_label_candidates(a2[0], a2[1], 16 / scale), tag, size=9, weight="bold")
        _strip(fig, 10, 4, W - 20, 30, W, H, f"{scene.title} — {name} view (stage model {scene.stage_model_id})",
               size=13)
        legend = {
            "HEATMAP": f"signed point-to-surface deviation vs stage model {scene.stage_model_id}, legend ±"
                       f"{scene.heatmap_band_mm:g} mm (red = + outward, blue = - inward, grey = none)",
            "DIMENSIONS": "blue arrows: measured flange length from the measured OML sharp along the fitted flange "
                          "to the measured far edge (dotted = hidden behind material)",
        }.get(name, "solid black = measured line on the outer apex; grey dashed = nominal line (no measured line); "
                    f"dotted = hidden behind material; points by segment (labels: {scene.label_source})")
        _strip(fig, 10, 34, W - 20, 26, W, H, legend, size=10, weight="normal")
        if name == "HEATMAP":
            _heat_legend(fig, W, H, scene.heatmap_band_mm)
        fname = f"{stem}_{name}.png"
        sha = _save(fig, out / fname, {"Title": name})
        rec = {"kind": f"GLOBAL_{name}", "file": fname, "sha256": sha, "px_per_mm": round(scale, 4),
               "labels_drawn": list(labeler.texts), "values_shown": bool(show_values)}
        if name == "DIMENSIONS":
            rec["arrows"] = {"fitted_direction": arrows_fitted, "nominal_direction": arrows_nominal}
        records.append(rec)
    return records


def _heat_rgb(dev: np.ndarray | None, band: float, n: int) -> np.ndarray:
    from matplotlib import colormaps

    if dev is None:
        return np.tile(np.asarray(_UNASSIGNED), (n, 1))
    cmap = colormaps["RdBu_r"]
    v = np.clip(np.nan_to_num(dev, nan=0.0) / max(band, 1e-9), -1, 1)
    rgb = cmap((v + 1) / 2)[:, :3]
    rgb[~np.isfinite(dev)] = _UNASSIGNED
    return rgb


def _heat_legend(fig, W: int, H: int, band: float) -> None:
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cax = fig.add_axes(((W - 100) / W, 0.15, 18 / W, 0.6))
    sm = ScalarMappable(norm=Normalize(-band, band), cmap=colormaps["RdBu_r"])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("signed deviation (mm)")


# ---------------------------------------------------------------------------
# close-ups
# ---------------------------------------------------------------------------


def plan_closeups(scene: RenderScene, *, px_per_mm: float | None = None,
                  max_tile_mm: float = _MAX_TILE_MM) -> list[CloseupPlan]:
    """One close-up per bend (tiled when longer than ``max_tile_mm``)."""
    s = float(px_per_mm or G4_THRESHOLDS["closeup_px_per_mm"])
    if s < G4_THRESHOLDS["closeup_min_px_per_mm"]:
        raise ValueError(f"close-ups need >= {G4_THRESHOLDS['closeup_min_px_per_mm']} px/mm")
    plans = []
    for b in scene.bends:
        line = b.measured_apex if b.measured_apex is not None else b.nominal_line
        length = float(np.linalg.norm(line[1] - line[0]))
        n = max(1, math.ceil(length / max_tile_mm))
        for i in range(n):
            plans.append(CloseupPlan(b.bend_uid, i, n, length * i / n, length * (i + 1) / n, s))
    return plans


def _local_points(scene: RenderScene, origin: np.ndarray, r: np.ndarray, u: np.ndarray, f: np.ndarray,
                  x0: float, x1: float, half_h: float, depth: float) -> np.ndarray:
    if scene.points is None:
        return np.zeros(0, dtype=np.int64)
    q = scene.points - origin
    x, y, z = q @ r, q @ u, q @ f
    m = (x >= x0) & (x <= x1) & (np.abs(y) <= half_h) & (np.abs(z) <= depth)
    return np.nonzero(m)[0]


def _mesh_in_window(scene: RenderScene, origin, r, u, f, x0, x1, half_h, depth):
    v, fc = scene.stage_vertices, scene.stage_faces
    if v is None or fc is None or not len(fc):
        return None
    q = v - origin
    X, Y, Z = q @ r, q @ u, q @ f
    cx, cy, cz = X[fc].mean(axis=1), Y[fc].mean(axis=1), Z[fc].mean(axis=1)
    keep = (cx >= x0 - 20) & (cx <= x1 + 20) & (np.abs(cy) <= half_h + 20) & (np.abs(cz) <= depth)
    return X, Y, Z, fc[keep]


def _draw_mesh(ax, X, Y, Z, tri, view_f: np.ndarray, vertices: np.ndarray) -> None:
    from matplotlib.collections import PolyCollection

    if tri is None or not len(tri):
        return
    a, b, c = vertices[tri[:, 0]], vertices[tri[:, 1]], vertices[tri[:, 2]]
    n = np.cross(b - a, c - a)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lam = np.abs(n @ view_f)
    order = np.argsort(-Z[tri].mean(axis=1))
    polys = np.stack([np.column_stack([X[t], Y[t]]) for t in tri[order]])
    base = np.asarray((0.72, 0.74, 0.78))
    colours = np.clip(base * (0.35 + 0.65 * lam[order, None]), 0, 1)
    ax.add_collection(PolyCollection(polys, facecolors=colours, edgecolors=colours, linewidths=0.2, zorder=2))


def _section_polyline(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray, normal: np.ndarray
                      ) -> list[np.ndarray]:
    """Segments where the plane (origin, normal) cuts the mesh."""
    d = (vertices - origin) @ normal
    segs = []
    for t in faces:
        dv = d[t]
        if dv.min() > 0 or dv.max() < 0:
            continue
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = dv[i], dv[j]
            if (a > 0) != (b > 0):
                w = a / (a - b)
                pts.append(vertices[t[i]] + w * (vertices[t[j]] - vertices[t[i]]))
        if len(pts) == 2:
            segs.append(np.asarray(pts))
    return segs


def _fit_direction(pts2: np.ndarray, start: np.ndarray, nominal_dir: np.ndarray, length: float) -> np.ndarray:
    """Direction of the flange in the section: PCA of points in a 25 deg sector
    around the nominal direction, between 3 mm and ``length`` + 5 mm from ``start``."""
    if len(pts2) == 0:
        return nominal_dir
    rel = pts2 - start
    dist = np.linalg.norm(rel, axis=1)
    cosang = (rel @ nominal_dir) / np.maximum(dist, 1e-9)
    m = (dist > 3.0) & (dist < length + 5.0) & (cosang > math.cos(math.radians(25.0)))
    if m.sum() < 20:
        return nominal_dir
    c = rel[m] - rel[m].mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    return d if float(d @ nominal_dir) >= 0 else -d


def render_closeup(scene: RenderScene, plan: CloseupPlan, path: str | os.PathLike, *,
                   corruption: Corruption | None = None) -> dict[str, Any]:
    """Render one close-up image (module docstring); returns its record.

    Layout: header; left column SCAN panel over RECIPE CAD panel (same camera,
    same window, same exact px/mm), each with a title strip above and a
    scale-bar strip below; right column the SECTION panel and the legend."""
    b = scene.bend(plan.bend_uid)
    s = plan.px_per_mm
    subject = b.measured_apex if b.measured_apex is not None else b.nominal_line
    a_dir = _unit(subject[1] - subject[0])
    r, u, f = _basis(-b.outward, a_dir)
    origin = subject[0]
    margin = 10.0
    x0, x1 = plan.s0 - margin, plan.s1 + margin
    half_h = max(20.0, 2.0 * (b.inner_radius_mm + b.thickness_mm) + 12.0)
    depth = half_h + 10.0
    shift = np.zeros(3)
    swap: Mapping[str, str] = {}
    if corruption is not None:
        if corruption.kind == "LINE_SHIFT":
            shift = u * float(corruption.shift_mm)
        elif corruption.kind == "LABEL_SWAP" and corruption.label_map and b.bend_uid in corruption.label_map:
            swap = dict(corruption.label_map)
        else:
            raise ValueError(f"unknown or incomplete corruption {corruption.kind!r}")

    def label_for(bend: BendOverlay) -> str:
        return swap.get(bend.bend_uid, bend.text)

    label_text = label_for(b)
    w_mm, h_mm = x1 - x0, 2 * half_h
    Wp, Hp = int(round(w_mm * s)), int(round(h_mm * s))
    pad, head, title, scale_h, sec_px, legend_h = 12, 64, 26, 34, 640, 120
    W = pad + Wp + 2 * pad + sec_px + pad
    left_h = head + 2 * (title + Hp + scale_h)
    right_h = head + title + sec_px + scale_h + legend_h
    H = max(left_h, right_h) + pad
    fig = _figure(W, H)
    xlim = (x0, x0 + Wp / s)
    ylim = (-half_h, -half_h + Hp / s)

    def to2(q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q) - origin
        return np.column_stack([q @ r, q @ u])

    # -- SCAN panel ---------------------------------------------------------------
    y = head
    _strip(fig, pad, y, Wp, title, W, H, f"SCAN (engine registration) — points by segment, labels: {scene.label_source}")
    y += title
    ax = _axes(fig, pad, y, Wp, Hp, W, H)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor("#fbfbfb")
    idx = _local_points(scene, origin, r, u, f, x0, x1, half_h, depth)
    if len(idx):
        lp = scene.points[idx]
        rgb = _shade(_segment_rgb(None if scene.labels is None else scene.labels[idx], len(idx)), _normals(lp), f)
        q = lp - origin
        _scatter(ax, np.column_stack([q @ r, q @ u]), q @ f, rgb, max(1.5, scene.spacing_mm * s * 1.6))
    else:
        _text(ax, (x0 + x1) / 2, 0, "no scan points in this window", size=12)
    labeler = _Labeler(ax, s)
    for other in scene.bends:
        if other.bend_uid == b.bend_uid or other.measured_apex is None:
            continue
        o2 = to2(other.measured_apex)
        if _segment_in_window(o2, x0, x1, half_h):
            _draw_line(ax, o2[0], o2[1], style="-", colour=_LINE, width=2.0)
            labeler.place([np.asarray(_clip_point(c, x0, x1, half_h)) for c in
                           _line_label_candidates(o2[0], o2[1], 6.0)], label_for(other), size=10)
    n2 = to2(b.nominal_line)
    _draw_line(ax, n2[0], n2[1], style="--", colour=_NOMINAL, width=1.6)
    if b.measured_apex is not None:
        a2 = to2(b.measured_apex + shift)
        o2 = to2(b.measured_oml + shift)
        _draw_line(ax, o2[0], o2[1], style="--", colour=_LINE, width=1.6)
        _draw_line(ax, a2[0], a2[1], style="-", colour=_LINE, width=2.8)
        anchor_line = a2
    else:
        anchor_line = n2
    labeler.place([np.asarray(_clip_point(c, x0, x1, half_h)) for c in
                   _line_label_candidates(anchor_line[0], anchor_line[1], 7.0)], label_text, size=11.5,
                  weight="bold", box="#fff7d6")
    y += Hp
    _scale_strip(fig, pad, y, Wp, scale_h, W, H, xlim, G4_THRESHOLDS["scale_bar_mm"],
                 note=f"  ({s:g} px/mm)")
    y += scale_h

    # -- RECIPE CAD panel -----------------------------------------------------------
    _strip(fig, pad, y, Wp, title, W, H,
           f"RECIPE CAD, stage model {scene.stage_model_id or 'final'} — same camera; labels from the recipe")
    y += title
    ax2 = _axes(fig, pad, y, Wp, Hp, W, H)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_facecolor("#f4f6fa")
    mesh = _mesh_in_window(scene, origin, r, u, f, x0, x1, half_h, depth)
    if mesh is not None:
        X, Y, Z, tri = mesh
        _draw_mesh(ax2, X, Y, Z, tri, f, scene.stage_vertices)
    cad_labeler = _Labeler(ax2, s)
    for seq, label, uid, line in scene.cad_labels:
        l2 = to2(line)
        if not _segment_in_window(l2, x0, x1, half_h):
            continue
        _draw_line(ax2, l2[0], l2[1], style="--", colour="#3b3b3b", width=1.4)
        cad_labeler.place([np.asarray(_clip_point(c, x0, x1, half_h)) for c in
                           _line_label_candidates(l2[0], l2[1], 7.0)], display_label(seq, label, uid), size=11.5,
                          weight="bold", box="#e8f0ff")
    y += Hp
    _scale_strip(fig, pad, y, Wp, scale_h, W, H, xlim, G4_THRESHOLDS["scale_bar_mm"],
                 note=f"  ({s:g} px/mm)")

    # -- SECTION panel -----------------------------------------------------------------
    sx = pad + Wp + 2 * pad
    y = head
    mid = subject[0] + a_dir * ((plan.s0 + plan.s1) / 2)
    lengths = [x.actual_mm or x.nominal_mm or 0.0 for x in b.sides] + [20.0]
    ext = max(lengths) + 12.0
    sec_scale = sec_px / (2 * ext)
    _strip(fig, sx, y, sec_px, title, W, H, "SECTION at mid-span (looking along the bend)")
    y += title
    sec = _axes(fig, sx, y, sec_px, sec_px, W, H)
    sec.set_facecolor("#fbfbfb")
    sec.set_xlim(-ext, ext)
    sec.set_ylim(-ext, ext)
    su, sw = u, -f  # across the bend; outward (apex up)

    def to_sec(q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q) - mid
        return np.column_stack([q @ su, q @ sw])

    sec_pts = np.zeros((0, 2))
    if scene.points is not None:
        q = scene.points - mid
        near = (np.abs(q @ a_dir) <= 1.5) & (np.linalg.norm(q, axis=1) <= ext * 1.5)
        if near.any():
            sec_pts = to_sec(scene.points[near])
            rgb = _segment_rgb(None if scene.labels is None else scene.labels[near], int(near.sum()))
            sec.scatter(sec_pts[:, 0], sec_pts[:, 1], s=4.0, c=rgb, marker="s", linewidths=0, rasterized=True,
                        zorder=4)
    if scene.stage_vertices is not None and scene.stage_faces is not None:
        for seg in _section_polyline(scene.stage_vertices, scene.stage_faces, mid, a_dir):
            s2 = to_sec(seg)
            sec.plot(s2[:, 0], s2[:, 1], color="#666666", linewidth=1.0, zorder=3)
    nom_mid = to_sec(b.nominal_line.mean(axis=0))[0]
    sec.plot([nom_mid[0]], [nom_mid[1]], marker="x", color=_NOMINAL, ms=9, mew=2.0, zorder=6)
    if b.measured_oml is not None and b.measured_apex is not None:
        d_oml = _unit(b.measured_oml[1] - b.measured_oml[0])
        oml_mid = b.measured_oml[0] + d_oml * float((mid - b.measured_oml[0]) @ d_oml)
        apex_mid = oml_mid + (b.measured_apex.mean(axis=0) - b.measured_oml.mean(axis=0))
        o_s, a_s = to_sec(oml_mid + shift)[0], to_sec(apex_mid + shift)[0]
        for side in b.sides:
            if side.actual_mm is None:
                continue
            nd = to_sec(mid + side.outward)[0] - to_sec(mid)[0]
            nd = nd / max(float(np.linalg.norm(nd)), 1e-9)
            d = _fit_direction(sec_pts, o_s, nd, side.actual_mm)
            tip = o_s + d * side.actual_mm
            _arrow(sec, o_s, tip)
            _text(sec, *(o_s + d * side.actual_mm * 0.55 + np.array([0.0, 14.0 / sec_scale])), f"{b.label}{side.side}",
                  size=9, box="#eef4ff")
        sec.plot([o_s[0]], [o_s[1]], marker="o", mfc="white", mec=_LINE, ms=10, mew=2.0, zorder=7)
        sec.plot([a_s[0]], [a_s[1]], marker="o", color=_LINE, ms=7, zorder=8)
    y += sec_px
    bar = 10.0 if sec_scale * 10.0 >= 25 else 50.0
    _scale_strip(fig, sx, y, sec_px, scale_h, W, H, (-ext, ext), bar, note=f"  ({sec_scale:.1f} px/mm)")
    y += scale_h
    lg = _axes(fig, sx, y, sec_px, legend_h, W, H)
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    lg.text(0.0, 0.98, "camera: outside of the bend, along its bisector\n"
                       "solid black line / dot: measured line on the outer apex (on the material)\n"
                       "dashed black line / open circle: measured OML virtual sharp\n"
                       "   (face-on, both lie on the bisector plane and coincide)\n"
                       "grey dashed line / x: nominal line of the stage model\n"
                       "blue arrows: measured flange length to the fitted far edge",
            fontsize=9.5, va="top")

    # -- header --------------------------------------------------------------------------
    tile = f" — tile {plan.tile_index + 1}/{plan.tile_count}" if plan.tile_count > 1 else ""
    _strip(fig, pad, 4, W - 2 * pad, 30, W, H, f"{scene.title} — close-up of {label_text}{tile}", size=13.5)
    reasons = f" ({', '.join(b.state_reasons)})" if b.state_reasons else ""
    if b.measured_apex is None:
        line_note = (f" No measured line: bend-line position not measured ({b.position_reason})."
                     if b.position_reason else " No measured line.")
    elif b.apex_source == "OML_FALLBACK_NO_ANGLE":
        line_note = " Solid line = measured OML sharp (no measured angle, so the apex is not computed)."
    else:
        line_note = ""
    _strip(fig, pad, 34, W - 2 * pad, 26, W, H, f"Engine state shown: {b.state}{reasons}.{line_note}",
           size=10, weight="normal")
    sha = _save(fig, Path(path), {"Title": "CLOSEUP"})
    return {"kind": "CLOSEUP", "file": Path(path).name, "sha256": sha, "bend_uid": b.bend_uid, "px_per_mm": s,
            "tile_index": plan.tile_index, "tile_count": plan.tile_count, "width_px": W, "height_px": H,
            "panel_px": [Wp, Hp], "window_mm": [round(w_mm, 3), round(h_mm, 3)],
            "measured_line_shown": b.measured_apex is not None, "state_shown": b.state,
            "label_shown": label_text, "scan_labels_drawn": list(labeler.texts),
            "cad_labels_drawn": list(cad_labeler.texts), "apex_source": b.apex_source}


def _segment_in_window(seg2: np.ndarray, x0: float, x1: float, half_h: float) -> bool:
    lo, hi = seg2.min(axis=0), seg2.max(axis=0)
    return bool(hi[0] >= x0 and lo[0] <= x1 and hi[1] >= -half_h and lo[1] <= half_h)


def _clip_point(p: np.ndarray, x0: float, x1: float, half_h: float) -> tuple[float, float]:
    return float(np.clip(p[0], x0 + 14.0, x1 - 14.0)), float(np.clip(p[1], -half_h + 3.0, half_h - 3.0))
