"""Owner-facing PNGs for the N1 one-time run (after scoring). For each of the 4 N1 scans and the N1 canary:
overview iso + a second view (gold lines green, matched engine lines dark green, FP engine lines red, missed gold
lines orange), and cross-section close-ups (matched lines; the missed joggle as well; for zero-bend rolled scans one
section across the roll). Engine = primary arm zs_lane_v1:zs (N1 runs); canary = the existing frozen-CLI DEV output
integration/dev_cli/zs/sc_fee685eee4db (no new run). Uses research/bench_labeling/render.py helpers (matplotlib).
Writes newscans_N1/visuals/<part_or_gid>/*.png and visuals/index.json.
usage (repo root, heavy -> through the slot): sherman_slot.sh .venv/bin/python -B <this> [scan_id ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v1 import scoring  # noqa: E402
import bb_common as bb  # noqa: E402
from research.bench_labeling.scanio import load_scan  # noqa: E402
from research.bench_labeling import render as RD  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
NB = P / "newscans_N1"
VIS = NB / "visuals"
GOLD_C, ENG_C, FP_C, FN_C = "#1a9850", "#00441b", "#d73027", "#fd8d3c"
W, H = 1400, 1000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cases():
    pre = json.loads((NB / "PREREG.json").read_text())
    out = []
    for sid, v in pre["gold_of_record"]["labels"].items():
        led = {json.loads(x)["scan_id"]: json.loads(x) for x in (P / "gold/GOLD_LEDGER.jsonl").read_text().splitlines()}
        out.append({"sid": sid, "name": v["part_id"], "part_id": v["part_id"], "role": "NEW",
                    "scan": led[sid]["path"], "gold": v["path"], "gid": led[sid]["gid"],
                    "pred": NB / "runs/v1/zs" / sid / "eval_lines.json", "engine": "zs_lane_v1:zs (N1 one-time run)"})
    can = [c for c in json.loads((P / "gold/canaries.json").read_text())["canaries"] if c["batch"] == "N1"][0]
    out.append({"sid": can["scan_id"], "name": can["gid"], "part_id": None, "role": "CANARY", "gid": can["gid"],
                "scan": can["original_path"], "gold": can["gold_path"],
                "pred": P / "integration/dev_cli/zs" / can["scan_id"] / "eval_lines.json",
                "engine": "zs_lane_v1:zs (existing frozen-CLI DEV output; canary is a DEV scan)"})
    return out


def decimate(pts, nrm, target=120000):
    """Voxel decimation to about ``target`` points (bisection on the voxel size)."""
    lo = pts.min(0)
    a, b = 0.05, 20.0
    idx = np.arange(len(pts))
    if len(pts) > target:
        for _ in range(18):
            vox = (a * b) ** 0.5
            key = np.floor((pts - lo) / vox).astype(np.int64)
            _, idx = np.unique(key, axis=0, return_index=True)
            if len(idx) > target:
                a = vox
            else:
                b = vox
            if abs(len(idx) - target) < 0.08 * target:
                break
    spacing = float(np.sqrt(np.ptp(pts[idx], 0)[np.argsort(np.ptp(pts[idx], 0))[1:]].prod() / max(len(idx), 1)) * 0.7)
    return pts[idx], (None if nrm is None else nrm[idx]), max(spacing, 0.2)


def roll_axis(nrm):
    m = nrm[:: max(1, len(nrm) // 200000)]
    w, v = np.linalg.eigh(m.T @ m / len(m))
    return v[:, 0], float(w[0] / w.sum())


def draw_seg(ax, view, a, b, color, lw, ls="-", label=None, marker="o", zorder=5):
    px, py, _ = view.project(np.array([a, b]))
    ax.plot(px, py, color=color, linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=zorder)
    ax.plot(px, py, linestyle="none", marker=marker, markersize=lw * 1.6, color=color, markeredgecolor="white",
            zorder=zorder)
    if label:
        mx, my = px.mean(), py.mean()
        dx, dy = px[1] - px[0], py[1] - py[0]
        n = np.hypot(dx, dy) or 1.0
        ax.text(mx - dy / n * 18, my + dx / n * 18, label, fontsize=12, fontweight="bold", color=color,
                ha="center", va="center", path_effects=RD.HALO, zorder=zorder + 1)


def overview(c, pts, nrm, vox, frame, view_name, gold, pred, rec, out, title):
    view = RD.View(frame, pts.mean(0), pts, W, H, margin=0.08)
    fig, ax = RD._figure(W, H)
    rgb = np.tile(np.array([0.72, 0.72, 0.72]), (len(pts), 1))
    RD._splat(ax, view, pts, nrm, rgb, vox, cull_tol_mm=1.5)
    matched_g = {m["gold_id"]: m["prediction_id"] for m in rec.get("matches") or []}
    matched_p = set(matched_g.values())
    for g in gold:
        if g["bend_id"] in matched_g:
            draw_seg(ax, view, g["line_start"], g["line_end"], GOLD_C, 7, label=g["bend_id"], zorder=4)
        else:
            draw_seg(ax, view, g["line_start"], g["line_end"], FN_C, 6, label=f"{g['bend_id']} missed", zorder=6)
    for i, r in enumerate(pred):
        pid = f"P{i + 1:03d}"
        if pid in matched_p:
            draw_seg(ax, view, r["line_start"], r["line_end"], ENG_C, 2.5, ls="--", marker="s", zorder=7)
        else:
            draw_seg(ax, view, r["line_start"], r["line_end"], FP_C, 4, marker="s", label=f"E{i + 1} false", zorder=8)
    handles = [Line2D([], [], color=GOLD_C, lw=6, label="gold line, found by engine"),
               Line2D([], [], color=ENG_C, lw=2.5, ls="--", marker="s", label="engine line (matched)"),
               Line2D([], [], color=FP_C, lw=4, marker="s", label="engine line, false (no gold)"),
               Line2D([], [], color=FN_C, lw=6, label="gold line, missed by engine")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.94), fontsize=12, frameon=True)
    ax.text(14, H - 18, title, fontsize=15, fontweight="bold", va="top")
    RD._scale_bar(ax, view)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def plane_basis(d):
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def section_line(c, P_full, g, pred_line, gincl, eincl, status, out, radius=16.0, slab=0.35):
    a, b = np.asarray(g["line_start"]), np.asarray(g["line_end"])
    d = (b - a) / np.linalg.norm(b - a)
    m = (a + b) / 2
    rel = P_full - m
    s = rel @ d
    sel = (np.abs(s) < slab) & (np.linalg.norm(rel - np.outer(s, d), axis=1) < radius)
    q = rel[sel] - np.outer(s[sel], d)
    u, v = plane_basis(d)
    xy = np.c_[q @ u, q @ v]
    cen = xy.mean(0)
    if np.linalg.norm(cen) > 1e-6:          # rotate so the bulk of material is below the apex
        ang = np.arctan2(-cen[0], -cen[1])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        xy = xy @ R.T
    else:
        R = np.eye(2)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#7f7f7f", linewidths=0, rasterized=True, label="scan section")
    ax.plot([0], [0], marker="o", markersize=14, color=GOLD_C if status != "missed" else FN_C,
            markeredgecolor="black", linestyle="none", label=f"gold apex ({g['bend_id']}, incl {gincl:.1f} deg)")
    txt = f"{c['name']}  {g['bend_id']}: gold vs engine apex, section across the fold"
    if pred_line is not None:
        pa, pb = np.asarray(pred_line["line_start"]), np.asarray(pred_line["line_end"])
        pd = pb - pa
        t = ((m - pa) @ d) / (pd @ d)
        x = pa + t * pd - m
        e2 = np.array([x @ u, x @ v]) @ R.T
        ax.plot([e2[0]], [e2[1]], marker="X", markersize=16, color=ENG_C, markeredgecolor="white", linestyle="none",
                label=f"engine apex (incl {eincl:.1f} deg)" if eincl is not None else "engine apex")
        txt += f"\nengine apex {np.linalg.norm(e2):.2f} mm from gold apex; angle diff {eincl - gincl:+.2f} deg" \
            if eincl is not None else ""
    else:
        txt += "\nengine: no line here (missed)"
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius * 0.95, radius * 0.45)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.grid(alpha=0.3)
    ax.set_title(txt, fontsize=14, loc="left")
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def section_roll(c, P_full, nrm_full, pred, out, slab=0.6):
    ax_dir, _ = roll_axis(nrm_full)
    cen0 = P_full.mean(0)
    s0 = (P_full - cen0) @ ax_dir
    lo, hi = np.percentile(s0, [2, 98])
    # station along the axis: of a few candidates, the one cutting the fewest points (avoids a plate lying in the
    # section plane, e.g. an internal bulkhead) -- generic, no part-specific choice
    cands = [lo + f * (hi - lo) for f in (0.5, 0.35, 0.65, 0.2, 0.8)]
    best = min(cands, key=lambda t: int(np.count_nonzero(np.abs(s0 - t) < slab)))
    cen = cen0 + best * ax_dir
    s = s0 - best
    sel = np.abs(s) < slab
    frac_station = (best - lo) / (hi - lo)
    q = P_full[sel] - cen
    u, v = plane_basis(ax_dir)
    xy = np.c_[q @ u, q @ v]
    # orient: chord horizontal (largest spread on x), opening up
    _, _, vt = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    xy = (xy - xy.mean(0)) @ vt.T
    if np.mean(xy[:, 1] - np.median(xy[:, 1])) > 0:
        xy[:, 1] *= -1
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="#7f7f7f", linewidths=0, rasterized=True, label="scan section (both skins)")
    hit = 0
    for r in pred:
        a, b = np.asarray(r["line_start"]), np.asarray(r["line_end"])
        da, db = (a - cen) @ ax_dir, (b - cen) @ ax_dir
        if da * db <= 0 and abs(db - da) > 1e-6:
            hit += 1
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    n = len(pred)
    ax.set_title(f"{c['name']}: section across the roll at {100 * frac_station:.0f} % of its length. Smooth constant-radius roll, no formed fold; "
                 f"gold 0 lines.\nEngine ({c['engine'].split(' ')[0]}): {n} line(s) on this scan"
                 f"{'' if n == 0 else f', {hit} crossing this section'}", fontsize=13, loc="left")
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def shrink(p, limit=600_000):
    """Palette-quantize a PNG when it exceeds ``limit`` bytes (keeps ~1400 px width)."""
    from PIL import Image
    if p.stat().st_size > limit:
        Image.open(p).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(p, optimize=True)


def run_case(c):
    out_dir = VIS / c["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = json.loads((REPO / c["gold"]).read_text())
    gl = [{"bend_id": b["bend_id"], "line_start": b["line_start"], "line_end": b["line_end"],
           "incl": float(b["notes"].split("incl ")[1].split(" deg")[0])} for b in gold["bends"]]
    payload = bb.load_eval_lines(c["pred"])
    pred = [r for r in payload["owned_bend_regions"] if bb.valid_line(r.get("line_start"), r.get("line_end"))]
    spec = {"case_id": c["sid"], "family_id": c["name"], "tier": "N1_VIS", "gold": {"kind": "ai_studio",
            "path": c["gold"]}, "label_complete": True}
    rec = scoring.score_scan(spec, [{"start": r["line_start"], "end": r["line_end"]} for r in pred], diagnose=False)
    scan = load_scan(c["scan"])
    P_full, N_full = scan.points, scan.normals
    pts, nrm, vox = decimate(P_full, N_full)
    frames = RD.camera_frames(pts)
    zero = rec["disposition"] == "ZERO_BEND_CONTROL"
    ax_dir, frac = roll_axis(N_full)
    rolled = frac < 0.03 or zero      # zero-bend scans here are rolled shells: show them end-on
    if zero:
        tp, fp, fn = 0, rec["false_positive_lines"], 0
    else:
        tp, fp, fn = rec["true_positive"], rec["false_positive"], rec["false_negative"]
    f1 = (1.0 if fp == 0 else 0.0) if zero else round(rec["f1"], 4)
    head = (f"{c['name']}{' (canary, DEV)' if c['role'] == 'CANARY' else ''} | gold {len(gl)} lines | engine "
            f"{len(pred)} lines | TP {tp} FP {fp} FN {fn} | F1 {f1:.3f}{' (zero-bend display)' if zero else ''}")
    pngs = []
    p1 = out_dir / f"{c['name']}_overview_iso.png"
    overview(c, pts, nrm, vox, frames["views"]["iso"], "iso", gl, pred, rec, p1, head + " | iso view")
    pngs.append(p1)
    if rolled:
        f2 = RD._frame(ax_dir, np.asarray(frames["views"]["iso"]["up"]))
        v2name = "end-on"
    else:
        f2, v2name = frames["views"]["top"], "top"
    p2 = out_dir / f"{c['name']}_view2_{'endon' if rolled else 'top'}.png"
    overview(c, pts, nrm, vox, f2, v2name, gl, pred, rec, p2, head + f" | {v2name} view")
    pngs.append(p2)
    angle = []
    mt = {m["gold_id"]: m for m in rec.get("matches") or []}
    for g in gl:
        if g["bend_id"] in mt:
            i = int(mt[g["bend_id"]]["prediction_id"][1:]) - 1
            e = pred[i].get("incl")
            angle.append({"gold_id": g["bend_id"], "engine_line": f"E{i + 1}", "gold_incl_deg": round(g["incl"], 2),
                          "engine_incl_deg": None if e is None else round(float(e), 2),
                          "diff_deg": None if e is None else round(float(e) - g["incl"], 2),
                          "offset_mm": round(mt[g["bend_id"]]["transverse_offset_mm"], 2)})
    if zero or not gl:
        p = out_dir / f"{c['name']}_section_roll.png"
        section_roll(c, P_full, N_full, pred, p)
        pngs.append(p)
    else:
        matched = [g for g in gl if g["bend_id"] in mt]
        # pick up to 3 matched lines spread over distinct included angles (generic: distinct fold types)
        picks, seen = [], []
        for g in sorted(matched, key=lambda g: g["incl"]):
            if all(abs(g["incl"] - s) > 15 for s in seen):
                picks.append(g)
                seen.append(g["incl"])
        if len(picks) < 2:
            picks = matched[:2]
        picks = picks[:3]
        missed = [g for g in gl if g["bend_id"] not in mt]
        if missed:
            picks.append(missed[-1])
        for g in picks:
            if g["bend_id"] in mt:
                i = int(mt[g["bend_id"]]["prediction_id"][1:]) - 1
                pl, ei, st = pred[i], pred[i].get("incl"), "matched"
            else:
                pl, ei, st = None, None, "missed"
            p = out_dir / f"{c['name']}_section_{g['bend_id']}_{st}.png"
            section_line(c, P_full, g, pl, g["incl"], None if ei is None else float(ei), st, p)
            pngs.append(p)
    for p in pngs:
        shrink(p)
    return {"part_id": c["part_id"], "gid": c["gid"], "scan_id": c["sid"], "role": c["role"], "engine": c["engine"],
            "gold_count": len(gl), "engine_count": len(pred), "tp": tp, "fp": fp, "fn": fn,
            "f1": None if zero else f1, "f1_display": f1, "zero_bend": zero,
            "missed_gold": [] if zero else rec["unmatched_gold_ids"],
            "false_engine": [f"E{int(x[1:])}" for x in ([] if zero else rec["unmatched_prediction_ids"])] if not zero
            else [f"E{i + 1}" for i in range(len(pred))],
            "angle_diffs": angle,
            "pngs": [str(p.relative_to(NB)) for p in pngs],
            "png_bytes": {str(p.relative_to(NB)): p.stat().st_size for p in pngs},
            "decimated_points": int(len(pts)), "voxel_mm": round(float(vox), 3)}


def main(argv):
    sel = set(argv)
    rows = []
    for c in cases():
        if sel and c["sid"] not in sel and c["name"] not in sel:
            continue
        r = run_case(c)
        (VIS / c["name"] / "_row.json").write_text(json.dumps(r, indent=1) + "\n")
        print(c["name"], r["tp"], r["fp"], r["fn"], r["f1_display"], r["pngs"], flush=True)
        rows.append(r)


if __name__ == "__main__":
    main(sys.argv[1:])
