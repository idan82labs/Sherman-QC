"""Offscreen 3D snapshots of an inspection result (matplotlib Agg, no display).

Four views are rendered from the viewer assets named by
``result.viewer_assets`` (paths relative to ``base_dir``), all in the result
frame ``cad_model_mm``:

- ``iso``   - isometric, camera at (+X, -Y, +Z)
- ``front`` - orthographic from -Y (X right, Z up)
- ``top``   - orthographic from +Z (X right, Y up)
- ``right`` - orthographic from +X (Y right, Z up)

Each view draws the decoded-stage display mesh (or the final mesh) as shaded
triangles (always every triangle: a mesh above ``DEFAULT_MAX_FACES`` is
skipped with a warning rather than thinned into a striped surface), the registered scan points coloured by signed deviation (diverging
map clipped at ±``heatmap_band_mm``; grey where the engine published no value;
a neutral colour when no heatmap is published), the nominal bend lines (grey
dashed), the measured OML lines (solid, coloured by the bend's worst decision),
unplanned creases, and a label per bend with its sequence number, label and a
``bend_uid`` short code. A scale bar, an axis triad and (with a heatmap) a
signed colour bar complete the view.

The projection is done here (orthographic, painter's algorithm: far elements
first), not with mplot3d, so occlusion and aspect are exact and the output is
deterministic: identical inputs give identical PNG bytes (fixed rcParams,
bundled Heebo font, PNG metadata stripped). Image text is Latin only; Hebrew
captions are set by the PDF, which shapes RTL text correctly.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from contracts.production.inspection_result import ShopfloorInspectionResultV1

__all__ = [
    "VIEWS",
    "VIEW_TITLES",
    "BendDraw",
    "Scene",
    "load_scene",
    "render_view",
    "render_views",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MAX_FACES",
]

DEFAULT_MAX_POINTS = 60_000
#: Display meshes are drawn whole: dropping faces would show a striped,
#: see-through surface. A mesh above this many triangles is not drawn at all
#: (``SNAPSHOT_MESH_TOO_LARGE`` warning; the next mesh kind or the scan points
#: carry the view). Agg draws ~300k triangles in about a second per view.
DEFAULT_MAX_FACES = 300_000

VIEWS: tuple[str, ...] = ("iso", "front", "top", "right")
VIEW_TITLES: dict[str, str] = {
    "iso": "ISO (from +X -Y +Z)",
    "front": "FRONT (from -Y)",
    "top": "TOP (from +Z)",
    "right": "RIGHT (from +X)",
}
#: view -> (camera forward direction, up hint)
_CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "iso": ((-1.0, 1.0, -1.0), (0.0, 0.0, 1.0)),
    "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}

#: Report accent (Sherman navy, from the logo) for the overview style.
_ACCENT = "#01416F"
_DECISION_RANK = {"FAIL": 4, "REVIEW": 3, "UNKNOWN": 2, "PASS": 1, "NOT_APPLICABLE": 0}
_COLORS = {
    "PASS": "#16A34A",
    "FAIL": "#DC2626",
    "REVIEW": "#F59E0B",
    "UNKNOWN": "#6B7280",
    "NOT_APPLICABLE": "#94A3B8",
    "nominal": "#64748B",
    "inactive": "#94A3B8",
    "mesh": (0.80, 0.84, 0.89),
    "scan_plain": "#475569",
    "nan": "#C3C9D2",
    "crease": "#C026D3",
}


@dataclass(frozen=True)
class BendDraw:
    sequence_index: int
    label: str
    uid_short: str
    state: str
    decision: str
    nominal: "np.ndarray"
    measured: "np.ndarray | None"
    anchor: "np.ndarray"


@dataclass
class Scene:
    points: "np.ndarray | None" = None
    deviations: "np.ndarray | None" = None
    band_mm: float = 0.5
    mesh_vertices: "np.ndarray | None" = None
    mesh_faces: "np.ndarray | None" = None
    mesh_kind: str | None = None
    bends: list[BendDraw] = field(default_factory=list)
    creases: list[tuple[str, "np.ndarray"]] = field(default_factory=list)
    heatmap_reason: str | None = None
    stage_model_id: str | None = None
    warnings: list[str] = field(default_factory=list)


def _bend_decision(row: Any) -> str:
    worst = "NOT_APPLICABLE"
    for char in row.characteristics:
        if _DECISION_RANK.get(char.decision, 0) > _DECISION_RANK[worst]:
            worst = char.decision
    return worst


def _ascii_label(label: str) -> str:
    return label if label and all(32 <= ord(c) < 127 for c in label) else ""


def _even_subsample(n: int, cap: int) -> "np.ndarray":
    import numpy as np

    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def load_scene(
    result: "ShopfloorInspectionResultV1",
    base_dir: str | Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    max_faces: int = DEFAULT_MAX_FACES,
) -> Scene:
    """Gather the drawable data. Missing or unreadable assets are skipped with
    a warning on ``scene.warnings`` (snapshots are display only)."""
    import numpy as np

    from domains.production.inspection.ply_io import PlyFormatError, read_ply
    from domains.production.inspection.viewer_assets import read_deviations
    from domains.production.reporting.labels import bend_display_label

    base = Path(base_dir)
    va = result.viewer_assets
    scene = Scene(
        band_mm=float(va.heatmap_band_mm),
        heatmap_reason=va.heatmap_unavailable_reason,
        stage_model_id=result.stage_decision.stage_model_id,
    )
    if va.registered_scan is not None:
        try:
            pts = read_ply(base / va.registered_scan).vertices
            dev = None
            if va.deviations is not None:
                dev = read_deviations(base / va.deviations, len(pts))
            idx = _even_subsample(len(pts), max_points)
            scene.points = pts[idx]
            scene.deviations = None if dev is None else dev[idx].astype(np.float64)
        except (OSError, PlyFormatError, ValueError) as exc:
            scene.warnings.append(f"SNAPSHOT_SCAN_UNAVAILABLE: {exc}")
            scene.points = scene.deviations = None
            scene.heatmap_reason = scene.heatmap_reason or "SNAPSHOT_SCAN_UNAVAILABLE"
    for kind in ("stage_mesh", "final_mesh"):
        rel = getattr(va, kind)
        if rel is None:
            continue
        try:
            mesh = read_ply(base / rel, require_faces=True)
        except (OSError, PlyFormatError, ValueError) as exc:
            scene.warnings.append(f"SNAPSHOT_MESH_UNAVAILABLE: {rel}: {exc}")
            continue
        faces = mesh.faces
        assert faces is not None
        if len(faces) > max_faces:
            scene.warnings.append(
                f"SNAPSHOT_MESH_TOO_LARGE: {rel}: {len(faces)} triangles > {max_faces}; mesh not drawn in the views"
            )
            continue
        scene.mesh_vertices, scene.mesh_faces = mesh.vertices, faces
        scene.mesh_kind = "stage" if kind == "stage_mesh" else "final"
        break
    for row in result.bends:
        disp = row.display
        scene.bends.append(
            BendDraw(
                sequence_index=row.sequence_index,
                label=_ascii_label(bend_display_label(result, row)),
                uid_short=row.bend_uid[3:7],
                state=row.state,
                decision=_bend_decision(row),
                nominal=np.asarray(disp.nominal_polyline_mm, dtype=np.float64),
                measured=None if disp.measured_polyline_mm is None else np.asarray(disp.measured_polyline_mm, dtype=np.float64),
                anchor=np.asarray(disp.label_anchor_mm, dtype=np.float64),
            )
        )
    for crease in result.unplanned_creases:
        scene.creases.append((crease.crease_id, np.asarray(crease.polyline_mm, dtype=np.float64)))
    return scene


def _basis(view: str) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np

    f, up = (np.asarray(v, dtype=np.float64) for v in _CAMERAS[view])
    f = f / np.linalg.norm(f)
    right = np.cross(f, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, f)
    return right, up2, f


def _nice_length(span: float) -> float:
    target = max(span * 0.2, 1e-6)
    exp = 10 ** math.floor(math.log10(target))
    for m in (1, 2, 5, 10):
        if m * exp >= target * 0.7:
            return m * exp
    return 10 * exp  # pragma: no cover


def _scene_points(scene: Scene) -> "np.ndarray":
    import numpy as np

    parts = []
    if scene.points is not None and len(scene.points):
        parts.append(scene.points)
    if scene.mesh_vertices is not None and len(scene.mesh_vertices):
        parts.append(scene.mesh_vertices)
    for b in scene.bends:
        parts.extend([b.nominal, b.anchor[None, :]])
        if b.measured is not None:
            parts.append(b.measured)
    for _, poly in scene.creases:
        parts.append(poly)
    if not parts:
        return np.zeros((1, 3))
    return np.concatenate(parts, axis=0)


_RC_OVERRIDES = {
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "font.size": 7,
    "path.simplify": False,
    "text.antialiased": True,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "axes.unicode_minus": True,
    "svg.hashsalt": "sherman",
}


def render_view(
    scene: Scene, view: str, *, width_px: int = 1400, height_px: int = 1000, dpi: int = 200, style: str = "detail"
) -> bytes:
    """Render one view to PNG bytes (deterministic).

    ``style="overview"`` is the clean report-cover picture: no heatmap colours
    (scan points in light grey), no colour bar or view titles, every line and
    label in the report accent (navy; not-yet-formed bends grey), and labels
    "<shop-order number> <studio label>" without the uid code."""
    if style not in ("detail", "overview"):
        raise ValueError(f"unknown style {style!r}")
    overview = style == "overview"
    import matplotlib
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties

    from domains.production.reporting.typefaces import font_path

    if view not in _CAMERAS:
        raise ValueError(f"unknown view {view!r}")
    right, up, fwd = _basis(view)
    font = FontProperties(fname=str(font_path("regular")), size=6.5)
    font_b = FontProperties(fname=str(font_path("bold")), size=7.5)

    def proj(p: "np.ndarray") -> "np.ndarray":
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        return np.stack([p @ right, p @ up], axis=1)

    base_rc = {k: v for k, v in matplotlib.rcParamsDefault.items() if k not in ("backend", "backend_fallback", "interactive")}
    with matplotlib.rc_context(base_rc):
        matplotlib.rcParams.update(_RC_OVERRIDES)
        fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="white")
        FigureCanvasAgg(fig)
        ax_rect = (0.02, 0.04, 0.88, 0.92) if overview else (0.02, 0.06, 0.84, 0.82)
        ax = fig.add_axes(ax_rect)
        ax.set_axis_off()

        allp = _scene_points(scene)
        uv_all = proj(allp)
        lo, hi = uv_all.min(axis=0), uv_all.max(axis=0)
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
        center2d = (lo + hi) / 2
        pad = 0.12 * span

        # --- mesh (painter's algorithm, lambert-ish shading)
        has_points = scene.points is not None and len(scene.points) > 0
        if scene.mesh_vertices is not None and scene.mesh_faces is not None and len(scene.mesh_faces):
            v, f = scene.mesh_vertices, scene.mesh_faces
            tri = v[f]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            norms = np.linalg.norm(normals, axis=1)
            normals = normals / np.where(norms > 0, norms, 1.0)[:, None]
            shade = 0.45 + 0.55 * np.abs(normals @ fwd)
            depth = (tri.mean(axis=1) @ fwd)
            order = np.argsort(-depth, kind="stable")
            base = np.asarray(_COLORS["mesh"])
            colors = np.clip(base[None, :] * shade[order, None], 0, 1)
            alpha = 0.5 if has_points else 1.0
            rgba = np.concatenate([colors, np.full((len(order), 1), alpha)], axis=1)
            polys = proj(tri[order].reshape(-1, 3)).reshape(-1, 3, 2)
            ax.add_collection(PolyCollection(polys, facecolors=rgba, edgecolors=rgba, linewidths=0.2, zorder=1))

        # --- scan points coloured by signed deviation
        norm = Normalize(vmin=-scene.band_mm, vmax=scene.band_mm, clip=True)
        cmap = matplotlib.colormaps["RdBu_r"]
        heat = has_points and scene.deviations is not None and not overview
        if has_points:
            pts = scene.points
            assert pts is not None
            depth = pts @ fwd
            order = np.argsort(-depth, kind="stable")
            uv = proj(pts[order])
            if heat:
                dev = scene.deviations[order]  # type: ignore[index]
                cols = cmap(norm(np.nan_to_num(dev, nan=0.0)))
                cols[np.isnan(dev)] = to_rgba(_COLORS["nan"])
            else:
                cols = np.tile(to_rgba(_COLORS["scan_plain"], 0.55) if not overview else to_rgba("#9AA5B1", 0.35),
                               (len(order), 1))
            ax.scatter(uv[:, 0], uv[:, 1], c=cols, s=1.0, marker="o", linewidths=0, zorder=2)

        # --- bend lines
        nominal_segments, nominal_colors = [], []
        for b in scene.bends:
            nominal_segments.append(proj(b.nominal))
            nominal_colors.append(_COLORS["inactive"] if b.state == "NOT_YET_FORMED" else _COLORS["nominal"])
        if nominal_segments:
            ax.add_collection(LineCollection(nominal_segments, colors=nominal_colors, linewidths=1.0,
                                             linestyles=(0, (4, 3)), zorder=3))
        for b in scene.bends:
            if b.measured is not None:
                uv = proj(b.measured)
                line_colour = _ACCENT if overview else _COLORS.get(b.decision, _COLORS["UNKNOWN"])
                ax.plot(uv[:, 0], uv[:, 1], color=line_colour, linewidth=1.8,
                        solid_capstyle="round", zorder=4)
        for crease_id, poly in scene.creases:
            uv = proj(poly)
            ax.plot(uv[:, 0], uv[:, 1], color=_COLORS["crease"], linewidth=1.4, linestyle="-.", zorder=4)
            mid = uv[len(uv) // 2]
            ax.text(mid[0], mid[1], crease_id, fontproperties=font, color=_COLORS["crease"], zorder=6)

        # Limits that match the axes box aspect, so "equal" needs no adjustment.
        box_ratio = (ax_rect[2] * width_px) / (ax_rect[3] * height_px)
        x_lo, x_hi = lo[0] - pad, hi[0] + pad
        y_lo, y_hi = lo[1] - pad, hi[1] + pad
        xr, yr = x_hi - x_lo, y_hi - y_lo
        if xr / yr < box_ratio:
            grow = (yr * box_ratio - xr) / 2
            x_lo, x_hi = x_lo - grow, x_hi + grow
        else:
            grow = (xr / box_ratio - yr) / 2
            y_lo, y_hi = y_lo - grow, y_hi + grow
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")

        # --- labels with leader lines: pushed away from the scene centre, and
        # moved around their anchor until they overlap no label placed before.
        axw, axh = ax_rect[2] * width_px, ax_rect[3] * height_px
        px_per_pt = dpi / 72.0

        def to_px(u: float, v: float) -> "np.ndarray":
            return np.array([ax_rect[0] * width_px + (u - x_lo) / (x_hi - x_lo) * axw,
                             ax_rect[1] * height_px + (v - y_lo) / (y_hi - y_lo) * axh])

        placed: list[tuple[float, float, float, float]] = []
        for b in sorted(scene.bends, key=lambda bend: bend.sequence_index):
            a2 = proj(b.anchor)[0]
            d = a2 - center2d
            n = float(np.linalg.norm(d))
            d = d / n if n > 1e-9 else np.array([0.0, 1.0])
            if overview:
                text = f"{b.sequence_index}  {b.label}" if b.label else f"{b.sequence_index}"
            else:
                text = f"{b.sequence_index} {b.label} {b.uid_short}" if b.label else f"{b.sequence_index} {b.uid_short}"
            w_px = (len(text) * 0.62 * 7.5 + 8.0) * px_per_pt
            h_px = 7.5 * 1.9 * px_per_pt
            anchor_px = to_px(a2[0], a2[1])
            offset_pt = d * 22.0
            for dist in (22.0, 36.0, 52.0, 70.0, 92.0):
                found = False
                for ang in (0, 35, -35, 70, -70, 110, -110, 145, -145, 180):
                    c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                    vec = np.array([d[0] * c - d[1] * s_, d[0] * s_ + d[1] * c]) * dist
                    centre = anchor_px + vec * px_per_pt
                    box = (centre[0] - w_px / 2, centre[1] - h_px / 2, centre[0] + w_px / 2, centre[1] + h_px / 2)
                    inside = (box[0] >= 0 and box[2] <= width_px * (0.9 if overview else 0.86) and box[1] >= 0
                              and box[3] <= height_px * (0.96 if overview else 0.9))
                    clear = all(box[2] < q[0] or box[0] > q[2] or box[3] < q[1] or box[1] > q[3] for q in placed)
                    if inside and clear:
                        offset_pt, found = vec, True
                        placed.append(box)
                        break
                if found:
                    break
            edge = _COLORS["inactive"] if b.state == "NOT_YET_FORMED" else (
                _ACCENT if overview else _COLORS.get(b.decision, _COLORS["UNKNOWN"]))
            ax.annotate(
                text, xy=(a2[0], a2[1]), xytext=(float(offset_pt[0]), float(offset_pt[1])), textcoords="offset points",
                fontproperties=font_b, color="#0F172A" if b.state != "NOT_YET_FORMED" else "#64748B",
                ha="center", va="center", zorder=7,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": edge, "lw": 0.9, "alpha": 0.92},
                arrowprops={"arrowstyle": "-", "color": edge, "lw": 0.7, "shrinkA": 0, "shrinkB": 1},
            )

        # --- scale bar (bottom-left, data units)
        length = _nice_length(span)
        x0, y0 = x_lo + 0.03 * (x_hi - x_lo), y_lo + 0.04 * (y_hi - y_lo)
        ax.plot([x0, x0 + length], [y0, y0], color="#0F172A", linewidth=1.4, zorder=8, solid_capstyle="butt")
        for xx in (x0, x0 + length):
            ax.plot([xx, xx], [y0 - span * 0.008, y0 + span * 0.008], color="#0F172A", linewidth=1.0, zorder=8)
        ax.text(x0 + length / 2, y0 + span * 0.015, f"{length:g} mm", fontproperties=font, ha="center", va="bottom",
                color="#0F172A", zorder=8)

        # --- axis triad (figure inset)
        tri_ax = fig.add_axes((0.86, 0.06, 0.12, 0.17))
        tri_ax.set_axis_off()
        tri_ax.set_xlim(-1.3, 1.3)
        tri_ax.set_ylim(-1.3, 1.3)
        tri_ax.set_aspect("equal")
        for vec, name, col in (((1, 0, 0), "X", "#DC2626"), ((0, 1, 0), "Y", "#16A34A"), ((0, 0, 1), "Z", "#2563EB")):
            e = np.asarray(vec, dtype=np.float64)
            uvv = np.array([e @ right, e @ up])
            if np.linalg.norm(uvv) < 0.15:
                tri_ax.plot([0], [0], marker="o", color=col, markersize=3)
                tri_ax.text(0.12, 0.12, name, fontproperties=font, color=col)
                continue
            tri_ax.annotate("", xy=(uvv[0], uvv[1]), xytext=(0, 0),
                            arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.0, "mutation_scale": 6})
            tri_ax.text(uvv[0] * 1.18, uvv[1] * 1.18, name, fontproperties=font, color=col, ha="center", va="center")

        # --- colour bar / heatmap note
        if overview:
            pass
        elif heat:
            cax = fig.add_axes((0.89, 0.34, 0.025, 0.5))
            from matplotlib.colorbar import ColorbarBase

            cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical", extend="both")
            band = scene.band_mm
            ticks = [-band, -band / 2, 0.0, band / 2, band]
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:+.2f}" if t else "0" for t in ticks])
            for lab in cax.get_yticklabels():
                lab.set_fontproperties(font)
            cax.set_title("mm", fontproperties=font, pad=3)
            fig.text(0.875, 0.88, f"Dev. vs {scene.stage_model_id or '-'}\n+ = outward", fontproperties=font,
                     va="bottom", ha="left", color="#334155")
        else:
            lines = ["No heatmap"]
            for part in (scene.heatmap_reason or "").split("_"):
                if not part:
                    continue
                if len(lines) > 1 and len(lines[-1]) + len(part) < 18:
                    lines[-1] += "_" + part
                else:
                    lines.append(part if len(lines) == 1 else "_" + part)
            fig.text(0.86, 0.88, "\n".join(lines[:6]), fontproperties=font, va="top", ha="left", color="#6B7280")

        title = VIEW_TITLES[view]
        if overview:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor="white", metadata={"Software": None})
            return buf.getvalue()
        mesh_note = {"stage": f"stage model {scene.stage_model_id or '-'}", "final": "final CAD"}.get(scene.mesh_kind or "", "no mesh")
        fig.text(0.02, 0.955, title, fontproperties=font_b, va="center", ha="left", color="#0F172A")
        fig.text(0.02, 0.92, f"frame cad_model_mm | mesh: {mesh_note}", fontproperties=font, va="center", ha="left",
                 color="#475569")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", metadata={"Software": None})
    return buf.getvalue()


def render_views(scene: Scene, views: tuple[str, ...] = VIEWS, **kwargs: Any) -> dict[str, bytes]:
    return {view: render_view(scene, view, **kwargs) for view in views}
