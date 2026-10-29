"""G3 visual truth-table material (plan §10 amendment 7), built with no engine output.

Amendment 7: "The per-bend visual truth table is built before any engine output
is seen. Two independent reviewers compare the registered raw scan with the
numbered recipe CAD and record FORMED, NOT_FORMED or NOT_DETERMINABLE. A third
reviewer resolves disagreements, then the table is frozen."

Nothing here runs or reads E3, E1-compile reruns or the legacy arm. Inputs: the
raw scan and the frozen DEV_AUTO_CONFIRMED compiled part (CAD only).

**Display registration** (:func:`display_register` then :func:`refine_formed_set`, E4b,
not an engine): the raw
scan (voxel-thinned) is fitted to every sequence-prefix stage model ``S_0..S_N``
of the compiled part from principal-axis hypotheses (the 24 rotations that
map the scan's principal axes onto signed model axes) with trimmed
point-to-plane ICP; the best inlier share at 1.5 mm wins, then single-bend
toggles refine the placement (the DEV recipes' sequence is the CAD candidate
order, so a real staged scan need not match a prefix). It only places the scan
next to the CAD; the fitted configuration is written to a reviewer-blind file in
the freeze builder's work area (``W/builder_SEALED/g3_reviewer_blind/``), never
in the reviewer-facing tree and never on an image. A best inlier share below 0.5
is flagged ``LOW_CONFIDENCE`` on the images, and reviewers then judge from the
unregistered views.

**Images** (matplotlib Agg; opaque case references; no file name, part number,
stage key, count or engine value is drawn):

- ``raw_views.png``: the raw scan alone in its own principal frame (ISO, TOP,
  FRONT, SIDE);
- ``cad_numbered_views.png``: the recipe CAD (final stage model, every bend
  formed) with every bend line drawn and labelled ``<seq> · <label> · <uid6>``;
- ``overlay_views.png``: the registered scan (grey) with the numbered CAD bend
  lines and the faint CAD surface, same cameras as the CAD views;
- ``bend_<seq>.png`` (staged and provisional-complete cases): the section across
  bend ``seq`` at mid-span, the scan locally re-fitted to the bend's parent
  panel, against two recipe references (E1's ``stage_points_from_compiled``)
  that differ only in this bend: ``FORMED reference`` and ``NOT FORMED
  reference``. Every other bend is placed as in the best display fit, so the
  parent flange lies where the scan has it even when an upstream bend is not
  formed yet; neither reference prejudges this bend.

Material is rendered for every case of the G2, G3, repeatability and BROAD lists
(sections for the staged ones: ``G3_STAGED``, ``BROAD_STAGED``,
``PROVISIONAL_COMPLETE``). ``truth_table_TEMPLATE.json`` lists one row per (case, bend) with empty
``reviewer_1``, ``reviewer_2``, ``tiebreak``, ``final`` fields
(schema :data:`.physical.TRUTH_TABLE_SCHEMA`); ``REVIEWER_INSTRUCTIONS.md`` states the
rules. DEV-CHECK cases go to a ``devcheck_SEALED`` directory.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import FREEZE_TOOLS_VERSION, compact_png, now_utc, read_scan, write_json

__all__ = [
    "MATERIAL_VERSION",
    "Registration",
    "voxel_thin",
    "estimate_normals",
    "icp_point_to_plane",
    "display_register",
    "refine_formed_set",
    "render_case_material",
    "truth_table_template",
    "REVIEWER_INSTRUCTIONS",
]

MATERIAL_VERSION = "e4b-g3-material-1.0.0"
_INLIER_MM = 1.5
_LOW_CONFIDENCE = 0.5
_BEND_COLOURS = ["#d9480f", "#1971c2", "#2f9e44", "#ae3ec9", "#e8590c", "#0c8599", "#5f3dc4", "#c2255c",
                 "#66a80f", "#1864ab", "#a61e4d", "#087f5b"]


def voxel_thin(points: np.ndarray, voxel_mm: float, max_points: int | None = None, seed: int = 0) -> np.ndarray:
    """One point per occupied voxel (the first), then a seeded subsample to ``max_points``."""
    pts = np.asarray(points, dtype=np.float64)
    if not len(pts):
        return pts
    keys = np.floor((pts - pts.min(axis=0)) / voxel_mm).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    out = pts[np.sort(first)]
    if max_points is not None and len(out) > max_points:
        rng = np.random.default_rng(seed)
        out = out[np.sort(rng.choice(len(out), size=max_points, replace=False))]
    return out


def estimate_normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Unit normals from the smallest principal direction of each point's k neighbours."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        return np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def _rot_from_vec(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * (kx @ kx)


def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray, t0: np.ndarray, *,
                       iterations: int = 30, max_dist: Sequence[float] = (20.0, 10.0, 5.0, 3.0, 2.0),
                       trim: float = 0.9, tree: Any = None) -> tuple[np.ndarray, dict[str, float]]:
    """Trimmed point-to-plane ICP of ``src`` onto ``dst``; returns ``(T, quality)``."""
    from scipy.spatial import cKDTree

    tree = tree if tree is not None else cKDTree(dst)
    t = np.asarray(t0, dtype=np.float64).copy()
    schedule = list(max_dist) + [max_dist[-1]] * max(0, iterations - len(max_dist))
    for it in range(iterations):
        p = src @ t[:3, :3].T + t[:3, 3]
        d, idx = tree.query(p, k=1, distance_upper_bound=schedule[min(it, len(schedule) - 1)])
        ok = np.isfinite(d)
        if ok.sum() < 6:
            break
        dd = d[ok]
        keep = np.zeros(len(d), dtype=bool)
        cut = np.quantile(dd, trim) if len(dd) > 10 else dd.max()
        keep[np.nonzero(ok)[0][dd <= cut]] = True
        ps, q, n = p[keep], dst[idx[keep]], dst_normals[idx[keep]]
        a = np.hstack([np.cross(ps, n), n])
        b = np.einsum("ij,ij->i", q - ps, n)
        try:
            x, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        step = np.eye(4)
        step[:3, :3] = _rot_from_vec(x[:3])
        step[:3, 3] = x[3:]
        t = step @ t
        if np.linalg.norm(x) < 1e-6:
            break
    p = src @ t[:3, :3].T + t[:3, 3]
    d, _ = tree.query(p, k=1)
    inl = d <= _INLIER_MM
    return t, {"inlier_share": float(inl.mean()) if len(d) else 0.0,
               "inlier_rms_mm": float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("inf")}


def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and proper (det +1) principal axes, largest variance first."""
    c = points.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((points - c).T))
    axes = vecs[:, ::-1].copy()
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    return c, axes


def _proper_signed_permutations() -> list[np.ndarray]:
    """The 24 rotations that map coordinate axes onto signed coordinate axes."""
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[j, i] = s
            if np.linalg.det(m) > 0.5:
                out.append(m)
    return out


@dataclass(frozen=True)
class Registration:
    raw_to_cad: np.ndarray
    stage_model_id: str
    inlier_share: float
    inlier_rms_mm: float
    hypotheses: int
    confidence: str


def display_register(scan_points: np.ndarray, stages: Sequence[tuple[str, np.ndarray, np.ndarray]], *,
                     coarse_points: int = 3000, fine_points: int = 15000, keep: int = 4, seed: int = 0
                     ) -> Registration:
    """Best display fit of the scan to any stage model (module docstring)."""
    from scipy.spatial import cKDTree

    scan = voxel_thin(scan_points, 1.5, fine_points, seed)
    coarse = scan[np.random.default_rng(seed).choice(len(scan), size=min(coarse_points, len(scan)), replace=False)]
    cs, es = _pca(scan)
    trials = []
    models = []
    for sid, pts, nrm in stages:
        sel = np.random.default_rng(seed + 1).choice(len(pts), size=min(len(pts), 40000), replace=False)
        mp, mn = np.asarray(pts, float)[sel], np.asarray(nrm, float)[sel]
        tree = cKDTree(mp)
        models.append((sid, mp, mn, tree))
        cm, em = _pca(mp)
        for perm in _proper_signed_permutations():
            r = em @ perm @ es.T
            t0 = np.eye(4)
            t0[:3, :3] = r
            t0[:3, 3] = cm - r @ cs
            t, q = icp_point_to_plane(coarse, mp, mn, t0, iterations=10, tree=tree)
            trials.append((q["inlier_share"], -q["inlier_rms_mm"], len(models) - 1, t))
    trials.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = None
    for share, _, mi, t in trials[:keep]:
        sid, mp, mn, tree = models[mi]
        tf, q = icp_point_to_plane(scan, mp, mn, t, iterations=30, max_dist=(5.0, 3.0, 2.0), tree=tree)
        cand = (q["inlier_share"], -q["inlier_rms_mm"], sid, tf, q)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return Registration(np.eye(4), "NONE", 0.0, float("inf"), len(trials), "FAILED")
    share, _, sid, tf, q = best
    return Registration(tf, sid, q["inlier_share"], q["inlier_rms_mm"], len(trials),
                        "OK" if share >= _LOW_CONFIDENCE else "LOW_CONFIDENCE")


def refine_formed_set(scan_points: np.ndarray, compiled: Any, final_cloud: Mapping[str, np.ndarray],
                      start: set[str], t0: np.ndarray, *, max_passes: int = 6, min_gain: float = 0.005,
                      sample: int = 30_000, scan_sample: int = 6_000, seed: int = 0
                      ) -> tuple[set[str], np.ndarray, dict[str, Any]]:
    """Greedy single-bend toggles from the best prefix fit (display placement only).

    The recipe's sequence order is the CAD candidate order, so a real staged scan's formed set
    need not be a prefix. Starting from the best prefix stage and its pose, every bend is toggled
    (formed <-> flat, E1's ``stage_points_from_compiled``) and the change kept when the scan's
    inlier share rises by more than ``min_gain``; passes repeat until nothing improves. The result
    only places the scan and the section references; it is never shown to reviewers."""
    from scipy.spatial import cKDTree

    from domains.production.engine.compile import stage_points_from_compiled

    idx = _thin_idx(len(final_cloud["points"]), sample)
    base = {k: np.asarray(final_cloud[k])[idx] for k in ("points", "normals", "segment_kind", "segment_index")}
    scan = voxel_thin(scan_points, 1.5, scan_sample, seed)

    def evaluate(formed: set[str], t_init: np.ndarray) -> tuple[float, float, np.ndarray]:
        pts, nrm = stage_points_from_compiled(compiled, base["points"], base["normals"], base["segment_kind"],
                                              base["segment_index"], sorted(formed))
        t, q = icp_point_to_plane(scan, pts, nrm, t_init, iterations=12, max_dist=(5.0, 3.0, 2.0),
                                  tree=cKDTree(pts))
        return q["inlier_share"], q["inlier_rms_mm"], t

    current = set(start)
    share, rms, pose = evaluate(current, t0)
    toggles = []
    for _ in range(max_passes):
        improved = False
        for b in compiled.bends:
            cand = current ^ {b.bend_uid}
            c_share, c_rms, c_pose = evaluate(cand, pose)
            if c_share > share + min_gain:
                toggles.append({"bend_uid": b.bend_uid, "to": "FORMED" if b.bend_uid in cand else "FLAT",
                                "inlier_share": round(c_share, 4)})
                current, share, rms, pose, improved = cand, c_share, c_rms, c_pose, True
        if not improved:
            break
    return current, pose, {"inlier_share": share, "inlier_rms_mm": rms, "toggles": toggles}


# --------------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------------


def _figure(w: int, h: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    FigureCanvasAgg(fig)
    return fig


def _save(fig, path: Path) -> str:
    """PNG with a 192-colour adaptive palette (:func:`.common.compact_png`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, format="png")
    return compact_png(path)


def _views(frame_axes: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    e0, e1, e2 = frame_axes[:, 0], frame_axes[:, 1], frame_axes[:, 2]
    iso = -(e0 + e1 + e2) / math.sqrt(3.0)
    return [("ISO", iso, e0), ("TOP", -e2, e0), ("FRONT", -e1, e0), ("SIDE", -e0, e1)]


def _basis(forward: np.ndarray, right_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = forward / np.linalg.norm(forward)
    r = right_hint - f * float(right_hint @ f)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.cross(f, [1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def _shade(normals: np.ndarray, f: np.ndarray, base: Sequence[float]) -> np.ndarray:
    lam = 0.35 + 0.65 * np.abs(normals @ f)
    return np.clip(np.asarray(base)[None, :] * lam[:, None], 0, 1)


def _panel(fig, rect, pts: np.ndarray, normals: np.ndarray | None, centre: np.ndarray, view, colour, *,
           extent: float, size: float = 2.2, lines: Sequence[tuple[str, np.ndarray, str]] = (),
           faint: tuple[np.ndarray, np.ndarray] | None = None, title: str = "") -> None:
    name, fwd, right = view
    r, u, f = _basis(fwd, right)
    ax = fig.add_axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent * rect[3] / rect[2] * fig.get_figheight() / fig.get_figwidth(),
                extent * rect[3] / rect[2] * fig.get_figheight() / fig.get_figwidth())
    if faint is not None:
        q = faint[0] - centre
        order = np.argsort(q @ f)[::-1]
        rgb = _shade(faint[1][order], f, (0.62, 0.74, 0.92))
        ax.scatter((q @ r)[order], (q @ u)[order], s=0.6, c=rgb, linewidths=0, alpha=0.45, rasterized=True)
    if len(pts):
        q = pts - centre
        order = np.argsort(q @ f)[::-1]
        n = normals[order] if normals is not None else np.tile(-f, (len(order), 1))
        ax.scatter((q @ r)[order], (q @ u)[order], s=size, c=_shade(n, f, colour), linewidths=0, rasterized=True)
    for text, seg, col in lines:
        a = np.asarray(seg, float) - centre
        x, y = a @ r, a @ u
        ax.plot(x, y, color=col, linewidth=2.4, solid_capstyle="round", zorder=5)
        ax.text(float(x.mean()), float(y.mean()), text, fontsize=8.5, color="white", zorder=6,
                ha="center", va="center", bbox={"boxstyle": "round,pad=0.2", "fc": col, "ec": "none", "alpha": 0.92})
    ax.set_title(f"{title}{name}", fontsize=10)


def _grid(path: Path, pts: np.ndarray, normals: np.ndarray | None, frame: tuple[np.ndarray, np.ndarray], colour,
          header: str, *, lines=(), faint=None) -> str:
    centre, axes = frame
    q = pts - centre if len(pts) else np.zeros((1, 3))
    extent = float(np.percentile(np.linalg.norm(q, axis=1), 99.5)) * 1.08 + 5.0
    fig = _figure(1600, 1260)
    fig.text(0.01, 0.985, header, fontsize=12, va="top", weight="bold")
    rects = [(0.01, 0.50, 0.485, 0.44), (0.505, 0.50, 0.485, 0.44), (0.01, 0.03, 0.485, 0.44),
             (0.505, 0.03, 0.485, 0.44)]
    for rect, view in zip(rects, _views(axes)):
        _panel(fig, rect, pts, normals, centre, view, colour, extent=extent, lines=lines, faint=faint)
    return _save(fig, path)


def _label(seq: int, label: str, uid: str) -> str:
    return f"{seq} · {label} · {uid[3:9]}"


def _section(path: Path, *, title: str, scan: np.ndarray, formed: np.ndarray, flat: np.ndarray, origin: np.ndarray,
             axis: np.ndarray, across: np.ndarray, out: np.ndarray, half: float, note: str) -> str:
    fig = _figure(1100, 1000)
    ax = fig.add_axes((0.07, 0.08, 0.9, 0.8))
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)

    def sec(p: np.ndarray, width: float) -> np.ndarray:
        q = p - origin
        m = np.abs(q @ axis) <= width
        return np.column_stack([q[m] @ across, q[m] @ out])

    f2, n2, s2 = sec(formed, 0.6), sec(flat, 0.6), sec(scan, 1.2)
    ax.scatter(n2[:, 0], n2[:, 1], s=14, c="#f08c00", marker="s", linewidths=0, label="NOT FORMED reference (CAD)",
               alpha=0.55)
    ax.scatter(f2[:, 0], f2[:, 1], s=14, c="#1c7ed6", marker="s", linewidths=0, label="FORMED reference (CAD)",
               alpha=0.55)
    ax.scatter(s2[:, 0], s2[:, 1], s=3, c="#212529", marker="o", linewidths=0, label="scan (registered)")
    ax.plot([0], [0], marker="x", color="#868e96", ms=10, mew=2)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, color="#e9ecef", linewidth=0.6)
    ax.set_xlabel("mm across the bend line")
    ax.set_ylabel("mm")
    import textwrap

    fig.text(0.01, 0.975, title, fontsize=12.5, weight="bold", va="top")
    fig.text(0.01, 0.945, "\n".join(textwrap.wrap(note, 150)), fontsize=9.5, va="top")
    return _save(fig, path)


def _thin_idx(n: int, cap: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, cap)).astype(np.int64)) if n else np.zeros(0, dtype=np.int64)


def _flange_window(compiled: Any, bend_uid: str) -> float:
    """Half-width of a bend section: the longer nominal flange + 10 mm, within [25, 90] mm."""
    lengths = [float(c.nominal) for c in compiled.characteristics
               if c.kind == "FLANGE_LENGTH" and c.bend_uid == bend_uid and c.nominal is not None]
    return float(min(90.0, max(25.0, (max(lengths) if lengths else 30.0) + 10.0)))


def render_case_material(case: Mapping[str, Any], scan_path: Path, compiled_dir: Path | None, out_dir: Path,
                         blind_dir: Path, *, sections: bool) -> dict[str, Any]:
    """All images of one case (module docstring); returns its manifest entry."""
    from ..engine_output import load_compiled

    ref = case["case_ref"]
    geo = read_scan(scan_path, with_faces=False, purpose="G3 truth material scan")
    raw = voxel_thin(geo.points, 1.0, 70_000)
    raw_n = estimate_normals(raw)
    images: dict[str, str] = {}
    entry: dict[str, Any] = {"case_ref": ref, "scan_sha256": case["scan_sha256"], "recipe_sku": case.get("recipe_sku"),
                             "tools_version": FREEZE_TOOLS_VERSION, "material_version": MATERIAL_VERSION,
                             "engine_output_used": False, "images": images, "bends": [], "sections": bool(sections)}
    images["raw_views.png"] = _grid(out_dir / "raw_views.png", raw, raw_n, _pca(raw), (0.55, 0.55, 0.58),
                                    f"review case {ref} — RAW SCAN as exported (own principal frame), no CAD")
    cp_json = None if compiled_dir is None else compiled_dir / "compiled_part.json"
    if cp_json is None or not cp_json.is_file():
        entry["cad"] = "NO_COMPILED_PART"
        entry["image_count"] = len(images)
        return entry
    compiled = load_compiled(cp_json)
    if compiled.status != "COMPILED":
        entry["cad"] = f"COMPILE_{compiled.status}"
        entry["image_count"] = len(images)
        return entry
    entry["cad"] = "COMPILED"
    stages = sorted((s for s in compiled.stages if s.kind == "SEQUENCE_PREFIX"), key=lambda s: s.stage_index)
    clouds = []
    for st in stages:
        with np.load(compiled_dir / st.sample_cloud.rel_path, allow_pickle=False) as z:
            clouds.append((st.stage_model_id, np.asarray(z["points"], float), np.asarray(z["normals"], float)))
    final = stages[-1]
    with np.load(compiled_dir / final.sample_cloud.rel_path, allow_pickle=False) as z:
        fz = {k: np.asarray(z[k]) for k in z.files}
    reg = display_register(geo.points, clouds)
    fitted = next((st for st in stages if st.stage_model_id == reg.stage_model_id), final)
    placed, pose, refined = (set(fitted.formed_bend_uids), reg.raw_to_cad, None)
    if reg.confidence != "FAILED":
        placed, pose, refined = refine_formed_set(geo.points, compiled, fz, set(fitted.formed_bend_uids),
                                                  reg.raw_to_cad)
    rot, off = pose[:3, :3], pose[:3, 3]
    reg_pts, reg_n = raw @ rot.T + off, raw_n @ rot.T
    keep = _thin_idx(len(fz["points"]), 60_000)
    fin_pts, fin_n = fz["points"][keep].astype(float), fz["normals"][keep].astype(float)
    cad_frame = _pca(fz["points"].astype(float))
    lines = []
    for i, b in enumerate(compiled.bends):
        ln = final.bend_lines[b.bend_uid]
        lines.append((_label(b.sequence_index, b.label, b.bend_uid), np.asarray([ln.start_mm, ln.end_mm]),
                      _BEND_COLOURS[i % len(_BEND_COLOURS)]))
        entry["bends"].append({"bend_uid": b.bend_uid, "sequence_index": b.sequence_index, "label": b.label})
    images["cad_numbered_views.png"] = _grid(out_dir / "cad_numbered_views.png", fin_pts, fin_n, cad_frame,
                                             (0.36, 0.52, 0.80), f"review case {ref} — RECIPE CAD, every bend formed; "
                                             "bend lines numbered <seq> · <label> · <uid6>", lines=lines)
    conf = "" if reg.confidence == "OK" else "  [REGISTRATION LOW CONFIDENCE: judge from the raw views]"
    images["overlay_views.png"] = _grid(out_dir / "overlay_views.png", reg_pts, reg_n, cad_frame, (0.35, 0.35, 0.38),
                                        f"review case {ref} — RAW SCAN registered to the recipe CAD (grey), CAD surface "
                                        f"faint blue, numbered CAD bend lines{conf}", lines=lines,
                                        faint=(fin_pts, fin_n))
    entry["registration"] = {"method": "E4B_DISPLAY_REGISTRATION (not an engine)", "confidence": reg.confidence}
    write_json(blind_dir / f"{ref}.registration.json",
               {"case_ref": ref, "stage_model_fitted": reg.stage_model_id, "inlier_share": reg.inlier_share,
                "inlier_rms_mm": reg.inlier_rms_mm, "hypotheses": reg.hypotheses, "confidence": reg.confidence,
                "prefix_raw_to_cad": reg.raw_to_cad.tolist(), "raw_to_cad": pose.tolist(),
                "placement_formed_set": sorted(placed), "refinement": refined,
                "note": "reviewer-blind: never shown to G3 reviewers (display placement only, not a truth)"})
    if sections:
        from scipy.spatial import cKDTree

        from domains.production.engine.compile import stage_points_from_compiled, stage_transforms_from_compiled

        scan_full = geo.points @ rot.T + off
        for b in compiled.bends:
            # Both references keep every other bend as in the best display fit, so the parent flange of
            # this bend sits where the scan has it; only this bend differs (formed / not formed).
            with_b, without_b = sorted(placed | {b.bend_uid}), sorted(placed - {b.bend_uid})
            t_parent = stage_transforms_from_compiled(compiled, set(with_b))[b.parent_panel_id]
            r_p, o_p = t_parent[:3, :3], t_parent[:3, 3]
            ln = final.bend_lines[b.bend_uid]
            a = r_p @ np.asarray(ln.start_mm, float) + o_p
            c = r_p @ np.asarray(ln.end_mm, float) + o_p
            axis = (c - a) / max(float(np.linalg.norm(c - a)), 1e-9)
            mid = 0.5 * (a + c)
            apex = r_p @ (0.5 * (np.asarray(b.loci.outer_apex.start_mm, float)
                                 + np.asarray(b.loci.outer_apex.end_mm, float))) + o_p
            cyl = r_p @ (0.5 * (np.asarray(b.loci.cylinder_axis.start_mm, float)
                                + np.asarray(b.loci.cylinder_axis.end_mm, float))) + o_p
            out = apex - cyl
            out = out - axis * float(out @ axis)
            out = out / max(float(np.linalg.norm(out)), 1e-9)
            across = np.cross(out, axis)
            half = _flange_window(compiled, b.bend_uid)
            formed_pts, _ = stage_points_from_compiled(compiled, fz["points"], fz["normals"], fz["segment_kind"],
                                                       fz["segment_index"], with_b)
            flat_pts, _ = stage_points_from_compiled(compiled, fz["points"], fz["normals"], fz["segment_kind"],
                                                     fz["segment_index"], without_b)
            local = scan_full[np.linalg.norm(scan_full - mid, axis=1) <= half * 1.6]
            note = "scan registered to the recipe CAD; parent flange re-fitted locally"
            pidx = next(i for i, p in enumerate(compiled.panels) if p.panel_id == b.parent_panel_id)
            pm = (fz["segment_kind"] != 1) & (fz["segment_index"] == pidx)
            ppts = formed_pts[pm]
            pnr = fz["normals"][pm].astype(float) @ r_p.T
            close = np.linalg.norm(ppts - mid, axis=1) <= half
            if close.sum() >= 50 and len(local) >= 50:
                t_loc, q = icp_point_to_plane(local, ppts[close], pnr[close], np.eye(4), iterations=20,
                                              max_dist=(8.0, 5.0, 3.0, 2.0), trim=0.5, tree=cKDTree(ppts[close]))
                if q["inlier_share"] >= 0.15:
                    local = local @ t_loc[:3, :3].T + t_loc[:3, 3]
                else:
                    note = "scan registered to the recipe CAD (local parent re-fit not used)"
            else:
                note = "scan registered to the recipe CAD (too few parent-flange points for a local re-fit)"
            title = f"review case {ref} — section across bend {_label(b.sequence_index, b.label, b.bend_uid)}"
            fname = f"bend_{b.sequence_index:02d}.png"
            images[fname] = _section(out_dir / fname, title=title, scan=local, formed=formed_pts, flat=flat_pts,
                                     origin=mid, axis=axis, across=across, out=out, half=half,
                                     note=note + conf + ". FORMED: the scan follows the blue profile; NOT_FORMED: it "
                                     "follows the orange one; NOT_DETERMINABLE: the scan does not show the bend region.")
    entry["image_count"] = len(images)
    return entry


def truth_table_template(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Empty two-reviewer + tie-break table, one row per (case, bend)."""
    from .physical import TRUTH_TABLE_SCHEMA

    rows = []
    for e in entries:
        for b in e.get("bends") or []:
            rows.append({"case_ref": e["case_ref"], "bend_uid": b["bend_uid"], "sequence_index": b["sequence_index"],
                         "label": b["label"], "reviewer_1": None, "reviewer_2": None, "tiebreak": None,
                         "final": None, "note": ""})
    return {"schema": TRUTH_TABLE_SCHEMA, "template_created_at": now_utc(), "frozen_at": None,
            "labels": ["FORMED", "NOT_FORMED", "NOT_DETERMINABLE"], "reviewers": [], "rows": rows}


REVIEWER_INSTRUCTIONS = """# G3 visual truth table: reviewer instructions (plan §10 amendment 7)

You build the per-bend truth table **before anyone has seen engine output**. You see only the
raw scan, the recipe CAD and these instructions. Do not open any other run log, result,
engine directory or `*SEALED*` path unless you are a DEV-CHECK reviewer told to review the
`devcheck_SEALED/` material (the display registration's fitted placements are not in this
tree: they are never shown to reviewers).

For every row of `truth_table_TEMPLATE.json` (one row per case and recipe bend):

- `FORMED` — the scan shows this bend bent (it follows the blue FORMED reference in
  `bend_<seq>.png`, or the global views clearly show the flange folded).
- `NOT_FORMED` — the scan shows the flange still flat where this bend will be (it follows
  the orange NOT FORMED reference, or the flange lies in the plane of its parent).
- `NOT_DETERMINABLE` — the scan does not show the bend region or its flange well enough to
  decide (occluded, cut off, too sparse). Never guess.

Images per case (opaque case reference only): `raw_views.png` (the scan alone),
`cad_numbered_views.png` (the recipe CAD with numbered bend lines `<seq> · <label> · <uid6>`),
`overlay_views.png` (the scan registered to the CAD; the registration is a display aid, not
an engine — if it says LOW CONFIDENCE or looks wrong, judge from the raw views) and, for
staged cases, `bend_<seq>.png` sections.

Two reviewers fill `reviewer_1` and `reviewer_2` independently (do not look at the other
reviewer's answers). A third reviewer fills `tiebreak` only where they disagree. `final` is
the agreed label, else the tie-break. Then set `frozen_at` (UTC ISO) and list the reviewers.

Freezing (whoever assembles the table, still before any engine output is compared with it):
save it as `truth_table_FROZEN.json` next to the template (DEV-CHECK: inside
`devcheck_SEALED/`) and record it in the freeze ledger with
`python -m domains.production.engine.eval.freeze.prefreeze record-truth-table --truth-table PATH`
(the DEV-CHECK table, at the final DEV-CHECK run or by R1 only: add `--include-sealed`). The
command refuses an unfrozen table, an unknown label, a `final` that is not the agreement or the
tie-break, and a table that misses or adds a template row; it records the A1 table's path and
sha256 and the DEV-CHECK table's sha256 only, in one FREEZE entry of
`docs/production/runlog/freeze/LEDGER.json`.

Order (amendment 7, enforced): until a frozen table is recorded, `prefreeze run-physical` refuses
E3 on every truth-relevant case (groups `G3_STAGED`, `BROAD_STAGED`, `PROVISIONAL_COMPLETE`), and
the scorer rejects a table frozen or recorded after the first E3 record of such a case.

Groups (`MANIFEST.json` `group`): `G3_STAGED` (plan §5 G3; every bend has a section image),
`BROAD_STAGED` (BROAD scans outside the G3 list whose stage hints say partial; sections too),
`PROVISIONAL_COMPLETE` (corpus stage keys disagree with "complete"; sections too) and
`STATE_CHECK` (complete scans: confirm every bend is formed from the global views; mark any
bend that is not; a completeness check that no score reads, filled in the same pass).
"""
