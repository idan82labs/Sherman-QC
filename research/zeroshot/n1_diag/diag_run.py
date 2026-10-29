"""Diagnostic re-run of the frozen zs_lane_v1 stages on one scan, dumping the intermediate products.

Read-only reuse of the frozen modules (z3_slicing.slicer, z2_self_recipe.run, zs_lane_v1.core, the pinned GBMs).
Z3 detect() is replayed step by step (same calls, same order) so the per-slice corners, tracks, segments, merged
segments, refined raw candidates and NMS survivors can be saved; the replay is checked against slicer.detect.
Gold is NOT read here. Output: one pickle in --out (scratch).

usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.n1_diag.diag_run --scan PLY --out DIR
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from research.zeroshot.zs_lane_v1 import core
from research.zeroshot.zs_lane_v1.core import REPO


def z3_replay(mesh):
    from research.zeroshot.z3_slicing import slicer as SL
    prm = SL.Params()
    th = SL.estimate_thickness(mesh)
    t = float(np.clip(th["t_mm"], 0.5, 6.0))
    fams = SL.axis_families(mesh, prm)
    tree = cKDTree(mesh.V)
    fam_out, raw = [], []
    for fi, f in enumerate(fams):
        u = np.asarray(f["axis"], float)
        e1, e2 = SL.orthonormal_frame(u)
        levels, chains = SL.slice_family(mesh, u, prm.step_mm)
        per_level = [SL.slice_corners(ch, prm, t) for ch in chains]
        tracks = SL.track_corners(levels, per_level, prm)
        segs = SL.tracks_to_segments(tracks, levels, u, prm, t, fi)
        seg_lines = [SL.segment_line(s, prm) for s in segs]
        merged = SL.merge_segments(segs, prm, mesh, tree, t)
        raw.extend(merged)
        corners = []
        for k, cs in enumerate(per_level):
            for c in cs:
                corners.append({"k": k, "lev": float(levels[k]),
                                "apex3": SL._lift(c["apex_o"], levels[k], u, e1, e2).tolist(),
                                "incl": float(c["incl"]), "skin": c["skin"], "la": float(c["la"]), "lb": float(c["lb"]),
                                "both": bool(c.get("both_skins", False)), "joggle": bool(c.get("joggle", False)),
                                "bis3": (c["bis"][0] * e1 + c["bis"][1] * e2).tolist()})
        trk = [{"k": [int(k) for k, _ in tr],
                "P": [SL._lift(c["apex_o"], levels[k], u, e1, e2).tolist() for k, c in tr],
                "incl": [float(c["incl"]) for _, c in tr]} for tr in tracks]
        fam_out.append({"axis": u.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "levels": levels.tolist(),
                        "score": f.get("score"), "corners": corners, "tracks": trk,
                        "seg_lines": [None if l is None else {k: v for k, v in l.items()} for l in seg_lines],
                        "merged": [dict(m) for m in merged]})
    pre_refine = [json.loads(json.dumps(c)) for c in raw]
    for c in raw:
        SL.refine_span(tree, c, prm.step_mm, t)
        c.update(SL.strip_evidence(mesh, tree, c, t))
        c["quality"] = c["n_support"] * c["support_frac"] * (0.5 + c["frac_outer"] + c["frac_both"])
    raw_ref = [json.loads(json.dumps(c)) for c in raw]
    merged = SL.merge_candidates(raw)
    return {"t_mm": t, "thickness": th, "families": fam_out, "pre_refine": pre_refine, "raw_refined": raw_ref,
            "candidates": merged, "params": asdict(prm)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from research.zeroshot.common import gold_guard
    from research.zeroshot.z3_slicing.custody import check_path
    from research.zeroshot.z3_slicing import slicer as SL
    dec = gold_guard.require_runnable(a.scan, final_testb=False)
    path = str(check_path(a.scan))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mesh = SL.load_mesh(path)
    np.savez_compressed(out / "mesh.npz", V=mesh.V, F=mesh.F, vn=mesh.vn, fn=mesh.fn)
    rep = z3_replay(mesh)
    ref = SL.detect(mesh)
    same = (len(ref["candidates"]) == len(rep["candidates"]) and all(
        np.allclose(x["start"], y["start"]) and np.allclose(x["end"], y["end"])
        for x, y in zip(ref["candidates"], rep["candidates"])))
    print(f"z3 replay: {len(rep['candidates'])} cands, identical to detect(): {same}  t={rep['t_mm']:.3f}", flush=True)
    ref["_mesh"] = mesh
    # Z2 (frozen it08 config), with its diagnostics
    from research.zeroshot.z2_self_recipe.run import run_scan
    t1 = time.time()
    payload, diag, scan = run_scan(Path(path))
    z2_lines = [{"start": [float(v) for v in r["line_start"]], "end": [float(v) for v in r["line_end"]],
                 "incl": float(r.get("included_angle_deg") or 90.0), "flanges": r.get("flanges"),
                 "span_mm": r.get("span_mm"), "cover": r.get("cover"), "ro_mm": r.get("ro_mm")}
                for r in payload.get("owned_bend_regions") or []]
    print(f"z2: {len(z2_lines)} lines, {len(diag['flanges'])} flanges, {len(diag['edges'])} edges, "
          f"{time.time() - t1:.0f}s", flush=True)
    np.savez_compressed(out / "z2_scan.npz", P=scan.P, N=scan.N, label=scan.label,
                        flange_of_point=scan.flange_of_point)
    # pooling + both pinned selectors + snap of every pooled candidate (the lane's arithmetic)
    cands, X = core.pool(ref, z2_lines, True)
    cfg = json.loads((REPO / "research/zeroshot/zs_lane_v1/config_v1.json").read_text())
    probs = {}
    for v in ("zs", "z3"):
        mdl = pickle.loads((REPO / cfg["variants"][v]["model"]).read_bytes())["model"]
        if v == "z3":
            keep = [i for i, c in enumerate(cands) if c["has_z3"]]
            p = np.full(len(cands), np.nan)
            p[keep] = mdl.predict(X[keep][:, :21])
        else:
            p = mdl.predict(X)
        probs[v] = p
    core.snap(cands, core.snap_points(scan, ref))
    for i, c in enumerate(cands):
        c["p_zs"], c["p_z3"] = float(probs["zs"][i]), float(probs["z3"][i])
        c.pop("_src", None)
    rec = {"scan_id": dec["scan_id"], "scan_path": path, "z3": rep, "z3_detect_identical": same,
           "z2_lines": z2_lines, "z2_diag": diag, "pool": cands, "X": X, "wall_s": time.time() - t0,
           "rss_gib": core._rss_gib()}
    (out / "diag.pkl").write_bytes(pickle.dumps(rec))
    print(f"done {time.time() - t0:.0f}s rss={core._rss_gib():.2f}G", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
