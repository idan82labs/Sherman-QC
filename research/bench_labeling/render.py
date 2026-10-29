"""Matplotlib (Agg) renders: numbered patches, bend-line verification, close-ups."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402

PATCH_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#469990", "#9a6324", "#800000", "#808000", "#000075", "#ffe119",
    "#dcbeff", "#aaffc3", "#fabed4", "#ffd8b1", "#1f77b4", "#d62728", "#2ca02c",
    "#8c564b", "#17becf", "#bcbd22",
]
LINE_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#e377c2",
                "#17becf", "#8c564b", "#bcbd22", "#000000"]
UNSEG_RGB = np.array([0.80, 0.80, 0.80])
HALO = [patheffects.withStroke(linewidth=5, foreground="white")]
VIEW_ORDER = ("iso", "front", "right", "top", "iso_under")
REQUIRED_VIEWS = ("iso", "front", "right", "top")


def _rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _frame(toward, up_hint):
    toward = toward / np.linalg.norm(toward)
    up = up_hint - (up_hint @ toward) * toward
    if np.linalg.norm(up) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(toward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        up = alt - (alt @ toward) * toward
    up = up / np.linalg.norm(up)
    right = np.cross(up, toward)
    return {"right": right.tolist(), "up": up.tolist(), "toward": toward.tolist()}


def camera_frames(points):
    """Display-only PCA camera frames. Coordinates are never transformed."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points[:: max(1, len(points) // 50000)] - centre, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    e3 = np.cross(e1, e2)
    frames = {
        "top": _frame(e3, e2),
        "front": _frame(-e2, e3),
        "right": _frame(e1, e3),
        "iso": _frame(e1 - e2 + e3, e3),
        "iso_under": _frame(-e1 + e2 - e3, -e3),
    }
    return {"centre": centre.tolist(), "views": frames,
            "pca_axes": [e1.tolist(), e2.tolist(), e3.tolist()]}


class View:
    """Orthographic projection onto a pixel canvas (y up)."""

    def __init__(self, frame, centre, bounds_points, width, height, margin=0.06):
        self.R = np.array([frame["right"], frame["up"], frame["toward"]], dtype=float)
        self.centre = np.asarray(centre, dtype=float)
        self.width, self.height = width, height
        q = (np.asarray(bounds_points) - self.centre) @ self.R.T
        lo, hi = q[:, :2].min(axis=0), q[:, :2].max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        usable = np.array([width, height]) * (1.0 - 2 * margin)
        self.px_per_mm = float(min(usable / span))
        self.mid = (lo + hi) / 2.0

    def project(self, pts):
        q = (np.asarray(pts, dtype=float).reshape(-1, 3) - self.centre) @ self.R.T
        px = (q[:, 0] - self.mid[0]) * self.px_per_mm + self.width / 2.0
        py = (q[:, 1] - self.mid[1]) * self.px_per_mm + self.height / 2.0
        return px, py, q[:, 2]

    def visibility(self, pts, cell_px, tol_mm):
        px, py, depth = self.project(pts)
        nx = int(self.width / cell_px) + 2
        ny = int(self.height / cell_px) + 2
        ix = np.clip((px / cell_px).astype(int), 0, nx - 1)
        iy = np.clip((py / cell_px).astype(int), 0, ny - 1)
        flat = iy * nx + ix
        zbuf = np.full(nx * ny, -np.inf)
        np.maximum.at(zbuf, flat, depth)
        return depth >= zbuf[flat] - tol_mm, (zbuf, nx, ny, cell_px)

    @staticmethod
    def depth_visible(px, py, depth, zinfo, tol_mm):
        zbuf, nx, ny, cell_px = zinfo
        ix = np.clip((np.asarray(px) / cell_px).astype(int), 0, nx - 1)
        iy = np.clip((np.asarray(py) / cell_px).astype(int), 0, ny - 1)
        z = zbuf[iy * nx + ix]
        return (depth >= z - tol_mm) | ~np.isfinite(z)


def _figure(width, height):
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    return fig, ax


def splat_px(view, spacing_mm):
    return float(np.clip(2.2 * spacing_mm * view.px_per_mm, 1.0, 10.0))


def _splat(ax, view, pts, normals, rgb, spacing_mm, alpha=1.0, cull_tol_mm=None):
    """Depth-sorted point splats. With ``cull_tol_mm`` hidden points (e.g. the far
    skin of the sheet) are dropped so they cannot bleed through gaps."""
    px, py, depth = view.project(pts)
    d_px = splat_px(view, spacing_mm)
    keep = np.ones(len(pts), dtype=bool)
    if cull_tol_mm is not None:
        keep, _ = view.visibility(pts, max(2.0, 1.5 * d_px), cull_tol_mm)
    order = np.flatnonzero(keep)[np.argsort(depth[keep])]
    shade = np.ones(len(pts))
    if normals is not None:
        shade = 0.40 + 0.60 * np.abs(normals @ view.R[2])
    colors = np.clip(rgb * shade[:, None], 0, 1)
    ax.scatter(px[order], py[order], s=(d_px * 0.72) ** 2, c=colors[order], marker="o",
               linewidths=0, alpha=alpha, rasterized=True)
    return d_px


def _scale_bar(ax, view, length_mm=10.0, x=30, y=30):
    length_px = length_mm * view.px_per_mm
    ax.plot([x, x + length_px], [y, y], color="black", linewidth=4, solid_capstyle="butt")
    ax.text(x + length_px / 2, y + 10, f"{length_mm:g} mm", ha="center", va="bottom",
            fontsize=13, fontweight="bold", path_effects=HALO)


def _title(ax, width, height, text):
    ax.text(16, height - 16, text, ha="left", va="top", fontsize=15, fontweight="bold",
            path_effects=HALO, family="DejaVu Sans")


def patch_colors(n_patches, adjacency):
    """Unique colours when the palette suffices; otherwise a greedy colouring that
    keeps adjacent patches distinct and spreads reuse across the palette."""
    k = len(PATCH_PALETTE)
    if n_patches <= k:
        return {pid: PATCH_PALETTE[pid - 1] for pid in range(1, n_patches + 1)}
    colors = {}
    for pid in range(1, n_patches + 1):
        used = {colors[o] for o in adjacency.get(pid, ()) if o in colors}
        start = (pid - 1) % k
        choice = next(((start + j) % k for j in range(k) if (start + j) % k not in used), start)
        colors[pid] = choice
    return {pid: PATCH_PALETTE[i] for pid, i in colors.items()}


def render_patch_views(points, normals, labels, patches, colors, frames, out_dir, *,
                       spacing_mm, thickness_mm, title, width=1600, height=1200):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.tile(UNSEG_RGB, (len(points), 1))
    for patch in patches:
        rgb[labels == patch["id"]] = _rgb(colors[patch["id"]])
    tol = max(0.25, 0.5 * (thickness_mm or 2.0))
    written = {}
    rng = np.random.default_rng(0)
    for name in VIEW_ORDER:
        view = View(frames["views"][name], frames["centre"], points, width, height)
        fig, ax = _figure(width, height)
        d_px = _splat(ax, view, points, normals, rgb, spacing_mm, cull_tol_mm=tol)
        visible, _ = view.visibility(points, max(2.0, 1.5 * d_px), tol)
        px, py, _ = view.project(points)
        placed = []
        for patch in sorted(patches, key=lambda p: -p["n_points"]):
            sel = np.flatnonzero((labels == patch["id"]) & visible)
            if len(sel) < max(8, 0.03 * patch["n_points"]):
                continue
            cx, cy = np.median(px[sel]), np.median(py[sel])
            cand = sel if len(sel) <= 300 else rng.choice(sel, 300, replace=False)
            centre_d = np.hypot(px[cand] - cx, py[cand] - cy)
            if placed:
                arr = np.array(placed)
                clash = np.min(np.hypot(px[cand][:, None] - arr[None, :, 0],
                                        py[cand][:, None] - arr[None, :, 1]), axis=1)
                score = np.minimum(clash, 70.0) - 0.15 * centre_d
            else:
                score = -centre_d
            best = cand[int(np.argmax(score))]
            x, y = float(px[best]), float(py[best])
            placed.append((x, y))
            ax.plot([x], [y], marker="o", markersize=7, color=colors[patch["id"]],
                    markeredgecolor="black", markeredgewidth=1.2)
            ax.text(x + 6, y + 6, str(patch["id"]), fontsize=22, fontweight="bold",
                    color="black", ha="left", va="bottom", path_effects=HALO)
        _title(ax, width, height, f"{title} | {name} | numbered planar patches")
        _scale_bar(ax, view)
        path = out_dir / f"patches_{name}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written[name] = path
    written["contact_sheet"] = contact_sheet(
        [written[n] for n in VIEW_ORDER], out_dir / "patches_contact_sheet.png",
        legend=_patch_legend(patches, colors))
    return written


def _patch_legend(patches, colors):
    rows = []
    shown = patches if len(patches) <= 60 else patches[:59]
    for p in shown:
        twin = f" twin {p['twin']}" if p.get("twin") else ""
        rows.append((colors[p["id"]], f"{p['id']:>3}  {p['area_mm2']:>8.0f} mm2{twin}"))
    rest = patches[len(shown):]
    if rest:
        top = max(p["area_mm2"] for p in rest)
        rows.append(("#ffffff", f"+{len(rest)} more <= {top:.0f} mm2 (patches.json)"))
    return rows


def contact_sheet(paths, out_path, *, legend=None, cols=3, tile=(800, 600)):
    import matplotlib.image as mpimg

    n_tiles = len(paths) + (1 if legend else 0)
    rows = int(np.ceil(n_tiles / cols))
    fig = plt.figure(figsize=(cols * tile[0] / 100.0, rows * tile[1] / 100.0), dpi=100)
    for i, path in enumerate(paths):
        ax = fig.add_axes([(i % cols) / cols, 1 - (i // cols + 1) / rows, 1 / cols, 1 / rows])
        ax.imshow(mpimg.imread(str(path)))
        ax.set_axis_off()
    if legend:
        i = len(paths)
        ax = fig.add_axes([(i % cols) / cols, 1 - (i // cols + 1) / rows, 1 / cols, 1 / rows])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        per_col = 20
        for k, (color, text) in enumerate(legend):
            col, row = divmod(k, per_col)
            x = 0.02 + col * 0.33
            y = 0.95 - row * 0.047
            ax.add_patch(plt.Rectangle((x, y - 0.015), 0.03, 0.03, color=color))
            ax.text(x + 0.04, y, text, fontsize=8.5, va="center", family="monospace")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return Path(out_path)


def _draw_line(ax, view, start, end, color, label, zinfo, tol, lw=3.5):
    seg = np.linspace(np.asarray(start), np.asarray(end), 60)
    px, py, depth = view.project(seg)
    vis = View.depth_visible(px, py, depth, zinfo, tol) if zinfo is not None else np.ones(60, bool)
    for k in range(59):
        style = "-" if (vis[k] and vis[k + 1]) else "--"
        ax.plot(px[k:k + 2], py[k:k + 2], color=color, linewidth=lw if style == "-" else lw * 0.6,
                linestyle=style, solid_capstyle="round")
    ax.plot([px[0]], [py[0]], marker="o", markersize=9, color=color, markeredgecolor="white")
    ax.plot([px[-1]], [py[-1]], marker="s", markersize=9, color=color, markeredgecolor="white")
    dx, dy = px[-1] - px[0], py[-1] - py[0]
    norm = np.hypot(dx, dy) or 1.0
    ox, oy = -dy / norm * 22, dx / norm * 22
    ax.text(px[30] + ox, py[30] + oy, label, fontsize=20, fontweight="bold", color=color,
            ha="center", va="center", path_effects=HALO)


def render_verification(points, normals, bends, frames, out_dir, *, spacing_mm,
                        thickness_mm, title, locus, width=1600, height=1200):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.tile(np.array([0.62, 0.62, 0.62]), (len(points), 1))
    tol = max(0.25, 0.5 * (thickness_mm or 2.0))
    written = {}
    for name in VIEW_ORDER:
        view = View(frames["views"][name], frames["centre"], points, width, height)
        fig, ax = _figure(width, height)
        d_px = _splat(ax, view, points, normals, rgb, spacing_mm, cull_tol_mm=tol)
        _, zinfo = view.visibility(points, max(2.0, 1.5 * d_px), tol)
        for k, bend in enumerate(bends):
            line = bend["label_line"]
            _draw_line(ax, view, line["start"], line["end"], LINE_PALETTE[k % len(LINE_PALETTE)],
                       bend["bend_id"], zinfo, tol + 1.0)
        _title(ax, width, height, f"{title} | {name} | {len(bends)} bend lines ({locus}); "
                                  "solid = visible, dashed = behind surface")
        _scale_bar(ax, view)
        path = out_dir / f"verify_{name}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written[name] = path
    written["contact_sheet"] = contact_sheet([written[n] for n in VIEW_ORDER],
                                             out_dir / "verify_contact_sheet.png")
    return written


def render_closeup(points, normals, labels, bend, color, out_path, *, patch_colors_map,
                   spacing_mm, width=1600, height=800):
    sharp = bend["sharp"]
    label = bend["label_line"]
    start, end = np.asarray(sharp["start"]), np.asarray(sharp["end"])
    d = np.asarray(bend["direction"])
    bis = np.asarray(bend["bisector"])
    radius = bend["zone_radius_used_mm"]
    rel = points - start
    s = rel @ d
    w = rel - np.outer(s, d)
    wn = np.linalg.norm(w, axis=1)
    length = bend["length_mm"]
    reach = max(20.0, 2.5 * radius)
    local = (s >= -15.0) & (s <= length + 15.0) & (wn <= reach)
    rgb = np.tile(np.array([0.70, 0.70, 0.70]), (len(points), 1))
    for pid, col in patch_colors_map.items():
        rgb[labels == pid] = 0.55 * _rgb(col) + 0.45
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("white")
    half_w = width // 2
    # left: face-on (viewer on the convex side, looking along -bisector)
    toward = -bis
    face = {"right": d.tolist(), "up": np.cross(toward, d).tolist(), "toward": toward.tolist()}
    ax = fig.add_axes([0, 0, 0.5, 1])
    ax.set_xlim(0, half_w)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    sel = np.flatnonzero(local)
    if len(sel):
        view = View(face, start, points[sel], half_w, height, margin=0.08)
        d_px = _splat(ax, view, points[sel], normals[sel] if normals is not None else None,
                      rgb[sel], spacing_mm)
        _, zinfo = view.visibility(points[sel], max(2.0, d_px), 1.0)
        _draw_line(ax, view, label["start"], label["end"], color, bend["bend_id"], zinfo, 1.5)
        _scale_bar(ax, view)
        ax.text(16, height - 16, f"{bend['bend_id']} face-on  ({view.px_per_mm:.1f} px/mm)",
                ha="left", va="top", fontsize=14, fontweight="bold", path_effects=HALO)
    # right: section slab across the middle of the line, looking along the line
    mid = length / 2.0
    slab = (np.abs(s - mid) <= max(1.0, 1.5 * spacing_mm)) & (wn <= radius + 12.0)
    ax2 = fig.add_axes([0.5, 0, 0.5, 1])
    ax2.set_xlim(0, half_w)
    ax2.set_ylim(0, height)
    ax2.set_axis_off()
    sel = np.flatnonzero(slab)
    if len(sel):
        up = -bis
        sec = {"right": np.cross(up, d).tolist(), "up": up.tolist(), "toward": d.tolist()}
        mid_pt = start + mid * d
        bounds = np.concatenate([points[sel], mid_pt[None, :]])
        view = View(sec, mid_pt, bounds, half_w, height, margin=0.10)
        px, py, _ = view.project(points[sel])
        ax2.scatter(px, py, s=14, c=np.clip(rgb[sel], 0, 1), linewidths=0)
        for key, ls in (("plane_a", "--"), ("plane_b", "--")):
            u = np.asarray(bend["side_dirs"][key])
            tr = np.stack([mid_pt, mid_pt + (radius + 10.0) * u])
            tx, ty, _ = view.project(tr)
            ax2.plot(tx, ty, linestyle=ls, color="black", linewidth=1.2)
        sx, sy, _ = view.project(mid_pt)
        ax2.plot(sx, sy, marker="x", markersize=14, color="black", markeredgewidth=2.5)
        lab_mid = (np.asarray(label["start"]) + np.asarray(label["end"])) / 2.0
        lx, ly, _ = view.project(lab_mid)
        ax2.plot(lx, ly, marker="o", markersize=12, color=color, markeredgecolor="black")
        _scale_bar(ax2, view, length_mm=5.0 if view.px_per_mm > 40 else 10.0)
        ax2.text(16, height - 16,
                 f"{bend['bend_id']} section (x = plane sharp, dot = label {bend['locus']})  "
                 f"incl {bend['included_angle_deg']:.1f} deg  ({view.px_per_mm:.1f} px/mm)",
                 ha="left", va="top", fontsize=13, fontweight="bold", path_effects=HALO)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return Path(out_path)
