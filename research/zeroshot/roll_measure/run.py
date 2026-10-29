"""Per-scan roll measurement job (heavy: run through /private/tmp/claude-501/sherman_slot.sh).

usage: .venv/bin/python -B -m research.zeroshot.roll_measure.run --path SCAN.ply --name NAME --out DIR
           [--voxel MM] [--detect-only]
Writes DIR/NAME.json (measurements) and DIR/NAME.pkl (plot data). No labels, gold or CAD are read.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import slicer as SL  # noqa: E402
from research.zeroshot.d1_two_skin import two_skin as TS  # noqa: E402
from research.zeroshot.roll_measure import measure as M, geom as G  # noqa: E402


def load(path: str, voxel: float | None):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.triangles) == 0:
        return SL.mesh_from_point_cloud(str(path))
    if voxel:
        m = m.simplify_vertex_clustering(voxel_size=voxel)
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()
    return SL.mesh_from_arrays(np.asarray(m.vertices), np.asarray(m.triangles))


def thickness(mesh: SL.Mesh) -> dict:
    dm = TS.mesh_from_arrays(mesh.V, mesh.F)
    te = TS.estimate_thickness(dm, n_samples=60000)
    return {"t_mm": te.t_mm, "confidence": te.confidence, "two_sided_fraction": te.two_sided_fraction}


def analyse(mesh: SL.Mesh, prm: M.RollParams, detect_only=False, log=print, t_info=None, overlay=True):
    t_info = t_info or thickness(mesh)
    t = t_info["t_mm"] or 1.5
    log(f" D1 thickness t={t:.3f} conf={t_info['confidence']:.2f}")
    regs = M.detect_regions(mesh, prm, t, log=log)
    out, plot = [], []
    for i, r in enumerate(regs):
        tr = r["outer"] or r["inner"]
        R, c, l0, l1 = M._track_summary(tr, r["levels"])
        det = {"id": i, "R_track_mm": float(R), "length_mm": float(l1 - l0), "n_stations": len(tr),
               "skins": ("both" if (r["outer"] and r["inner"]) else ("outer" if r["outer"] else "inner")),
               "sweep_track_deg": float(np.degrees(np.median([a["sweep"] for _, a in tr]))),
               "axis_candidate": np.asarray(r["u"]).tolist()}
        if detect_only:
            out.append({"detection": det})
            plot.append(_plot_data(r))
            continue
        r2 = M.refine_region(mesh, r, prm, t, log=log)
        if r2.get("unconverged"):
            log(f"  region {i} dropped: axis refinement did not converge (oblique candidate)")
            continue
        mres = M.measure_region(r2, prm, t_info["t_mm"])
        mres["feature_class"] = classify(mres)
        out.append({"detection": det, "measure": mres})
        pd = _plot_data(r2)
        if overlay:
            pd["overlay"] = overlay_data(mesh, mres, r2["step"])
        plot.append(pd)
    if not detect_only:
        out, plot, dropped = dedupe_measured(out, plot)
        if dropped:
            log(f" post-refinement duplicates dropped: {dropped}")
    for i, o in enumerate(out):
        o["detection"]["id"] = i
    return {"thickness_d1": t_info, "n_regions": len(out), "regions": out}, plot


def classify(m):
    """CAD-side rule (research/cad_roll_solver/SCHEMA.md): a curved region is a ROLL when Ri > 10 t; between the
    scan detector's floor (max(12 mm, 5 t)) and 10 t it is a large-radius BEND for the bend lane / E1."""
    t = m["thickness"]["t_used_mm"]
    ri = m["R_inner"]["mean"]
    if t is None or ri is None:
        return {"class": "ROLL_UNVERIFIED", "ri_over_t": None, "rule": "Ri > 10 t (t or Ri unknown)"}
    q = ri / t
    return {"class": "ROLL" if q > 10 else "LARGE_RADIUS_BEND", "ri_over_t": q, "rule": "Ri > 10 t"}


def _rad(m):
    return m["R_inner"]["mean"] if m["R_inner"]["mean"] is not None else m["R_outer"]["mean"]


def dedupe_measured(out, plot):
    """After refinement, regions found from different candidate axes that converged onto the same roll
    (same axis line, radius and axial span) are duplicates: keep the best fit (lowest station rms)."""
    order = sorted(range(len(out)), key=lambda i: (out[i]["measure"]["fit_rms_mm"]["median"], -out[i]["measure"]["n_stations"]))
    keep, dropped = [], []
    for i in order:
        m = out[i]["measure"]
        d = np.asarray(m["axis"]["direction"])
        p = np.asarray(m["axis"]["point_mid"])
        R = _rad(m)
        dup = False
        for j in keep:
            q = out[j]["measure"]
            dq = np.asarray(q["axis"]["direction"])
            pq = np.asarray(q["axis"]["point_mid"])
            if abs(d @ dq) < np.cos(np.radians(3.0)) or abs(_rad(q) - R) > 0.05 * R + 1.0:
                continue
            v = p - pq
            if np.linalg.norm(v - (v @ dq) * dq) > 0.05 * R + 2.0:
                continue
            ov = min(v @ dq + m["length_mm"] / 2, q["length_mm"] / 2) - max(v @ dq - m["length_mm"] / 2, -q["length_mm"] / 2)
            if ov > 0.3 * min(m["length_mm"], q["length_mm"]):
                dup = True
                break
        (dropped if dup else keep).append(i)
    keep.sort()
    return [out[i] for i in keep], [plot[i] for i in keep], len(dropped)


def overlay_data(mesh, m, step, n_max=200000):
    """Scan vertices of the rolled region in axis coordinates (s along the axis, phi about it, rho from it,
    skin by normal direction) and full mesh sections at the first / middle / last station."""
    d = np.asarray(m["axis"]["direction"])
    p0 = np.asarray(m["axis"]["point_mid"])
    f1, f2 = G.frame(d)
    ps = m["per_station"]
    lv = np.array([q["level"] for q in ps])
    X = mesh.V - p0
    s = X @ d
    # station levels are along the region's slicing axis; convert via the station centres' projection
    s_lo, s_hi = -m["length_mm"] / 2 - 2 * step, m["length_mm"] / 2 + 2 * step
    Rin = m["R_inner"]["mean"] if m["R_inner"]["mean"] is not None else m["R_outer"]["mean"]
    t = m["thickness"]["t_used_mm"] or 0.0
    rho = np.linalg.norm(X - np.outer(s, d), axis=1)
    keep = (s > s_lo) & (s < s_hi) & (np.abs(rho - Rin - t / 2) < 0.25 * Rin + 3 * t + 5)
    idx = np.nonzero(keep)[0]
    if len(idx) > n_max:
        idx = np.random.default_rng(0).choice(idx, n_max, replace=False)
    Xi = X[idx]
    phi = np.arctan2(Xi @ f2, Xi @ f1)
    radial = Xi - np.outer(Xi @ d, d)
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-9)[:, None]
    skin = np.where(np.sum(mesh.vn[idx] * radial, 1) < 0, -1, 1)   # -1: concave (inner), +1: convex (outer)
    pts = {"s": s[idx], "phi": phi, "rho": rho[idx], "skin": skin.astype(np.int8)}
    # sections at first / middle / last station perpendicular to the refined axis
    secs = []
    for frac in (0.03, 0.5, 0.97):
        lev = float(p0 @ d + s_lo + 2 * step + frac * (s_hi - s_lo - 4 * step))
        levels, per = _one_section(mesh, d, lev)
        chains = [ch["pts"] for ch in per]
        secs.append({"level": lev, "s": lev - float(p0 @ d), "chains": chains})
    return {"pts": pts, "sections": secs, "frame": {"d": d, "p0": p0, "f1": f1, "f2": f2}}


def _one_section(mesh, u, level):
    """Mesh section by the single plane u.x = level, as 2-D chains in the z3 frame of u."""
    h = mesh.V @ u
    hf = h[mesh.F]
    sel = (hf.min(1) <= level + 2.0) & (hf.max(1) >= level - 2.0)
    Fs = mesh.F[sel]
    if len(Fs) == 0:
        return np.zeros(0), []
    vid = np.unique(Fs)
    remap = -np.ones(len(mesh.V), np.int64)
    remap[vid] = np.arange(len(vid))
    sub = SL.mesh_from_arrays(mesh.V[vid], remap[Fs])
    # slice_family puts its first level at min + step/2: choose the step so that level lands exactly
    hmin = float((sub.V @ u).min())
    step = 2.0 * (level - hmin) if level > hmin else 1.0
    levels, per = SL.slice_family(sub, u, step)
    return levels, (per[0] if len(per) else [])


def _plot_data(r):
    S = M._stations(r, r["levels"])
    keep = []
    for s in S:
        d = {"level": s["level"]}
        for sk in ("outer", "inner"):
            if sk in s:
                a = s[sk]
                d[sk] = {"c": a["c"], "R": a["R"], "start": a["start"], "sweep": a["sweep"], "pts": a["cover_pts"][::2],
                         "ends": a["ends"]}
        keep.append(d)
    return {"u": np.asarray(r["u"]), "stations": keep, "step": r["step"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=None)
    ap.add_argument("--detect-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mesh = load(a.path, a.voxel)
    print(f"{a.name}: {len(mesh.V)} V {len(mesh.F)} F ({time.time() - t0:.1f}s)")
    prm = M.RollParams()
    res, plot = analyse(mesh, prm, a.detect_only)
    res.update({"name": a.name, "path": a.path, "voxel_mm": a.voxel, "detect_only": a.detect_only,
                "n_vertices": int(len(mesh.V)), "params": M.asdict(prm), "wall_s": round(time.time() - t0, 1),
                "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)})
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=1, default=_js))
    sub = mesh.V[np.random.default_rng(0).choice(len(mesh.V), min(len(mesh.V), 60000), replace=False)]
    (out / f"{a.name}.pkl").write_bytes(pickle.dumps({"regions": plot, "cloud": sub}))
    print(f"{a.name}: {res['n_regions']} regions, {res['wall_s']} s, rss {res['peak_rss_gib']} GiB")
    for r in res["regions"]:
        d = r["detection"]
        msg = f"  region {d['id']}: R~{d['R_track_mm']:.1f} sweep~{d['sweep_track_deg']:.1f} len {d['length_mm']:.0f} {d['skins']}"
        if "measure" in r:
            m = r["measure"]
            f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
            msg += (f" | R_in {f2(m['R_inner']['mean'])}±{f2(m['R_inner']['u_boot'])} R_out {f2(m['R_outer']['mean'])}"
                    f" sweep {m['sweep_deg']['mean']:.2f} rms {m['fit_rms_mm']['median']:.3f} cov {m['coverage']['arc_fraction']['mean']:.2f}")
        print(msg)


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
