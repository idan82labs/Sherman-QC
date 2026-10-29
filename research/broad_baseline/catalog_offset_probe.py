#!/usr/bin/env python3
"""Probe: are legacy Codex catalog lines a constant translation away from GOLD_SET_V2 gold?

For each GOLD_SET_V2_NEW scan whose prior (legacy catalog) missed lines are ALL of cause
MISS_PRIOR_OFF_APEX_GT_4MM, this
  1 pairs every gold line with its nearest parallel (<5 deg) prior line (Hungarian on offset),
  2 fits ONE translation t minimising sum |(I - a a^T)(t - d_i)|^2 over the pairs
    (d_i = perpendicular offset of the prior line from the gold line), reports |t| and the
    per-pair residual after removing t, and the rank of the constraint system,
  3 measures how far the prior lines (and the gold lines) sit from the scan surface
    (distance of 21 samples per line to the nearest raw vertex), and the catalog's own
    endpoint-snap distances,
  4 reads the scan geometry around each prior line: dihedral of the two sides (normals of vertices
    within 3 mm of the line, split by side) to tell "on another fold / free edge / flat".
Read-only. Labels are not changed. Writes one JSON report.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import score as sc  # noqa: E402

ROOT = bb.ROOT
GS2 = ROOT / "docs/production/runlog/gold_set_v2"
CATALOG = ROOT / "data/bend_corpus/annotations/bend_line_development_catalog_v2_20260830.json"


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def seg(line):
    return np.asarray(line["line_start"], float), np.asarray(line["line_end"], float)


def perp_offset(gold, prior):
    """Vector from gold line (infinite) to prior-line midpoint, perpendicular to the gold axis."""
    g0, g1 = seg(gold)
    a = unit(g1 - g0)
    p0, p1 = seg(prior)
    m = 0.5 * (p0 + p1)
    d = m - g0
    return d - a * (d @ a), a


def angle(l1, l2):
    a, b = unit(np.subtract(*seg(l1)[::-1])), unit(np.subtract(*seg(l2)[::-1]))
    return float(np.degrees(np.arccos(min(1.0, abs(a @ b)))))


def samples(line, n=21):
    p0, p1 = seg(line)
    t = np.linspace(0, 1, n)[:, None]
    return p0 + t * (p1 - p0)


def load_scan(path):
    import trimesh
    mesh = trimesh.load(path, process=False)
    v = np.asarray(mesh.vertices, float)
    vn = np.asarray(mesh.vertex_normals, float) if hasattr(mesh, "vertex_normals") and len(mesh.faces) else None
    return v, vn


def local_geometry(line, v, vn, tree, radius=3.0, side_band=(2.0, 8.0)):
    """Dihedral across the line: median normals of vertices on either side (perp distance band)."""
    p0, p1 = seg(line)
    a = unit(p1 - p0)
    L = np.linalg.norm(p1 - p0)
    mid = 0.5 * (p0 + p1)
    idx = tree.query_ball_point(mid, r=L / 2 + side_band[1])
    if not idx or vn is None:
        return None
    pts = v[idx] - p0
    t = pts @ a
    keep = (t > 0.1 * L) & (t < 0.9 * L)
    perp = pts - np.outer(t, a)
    dist = np.linalg.norm(perp, axis=1)
    band = keep & (dist > side_band[0]) & (dist < side_band[1])
    near = keep & (dist <= radius)
    if band.sum() < 20:
        return {"near_vertices": int(near.sum()), "band_vertices": int(band.sum())}
    P = perp[band]
    # split sides by principal direction of the perpendicular offsets
    u, s, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    side = (P @ vt[0]) > 0
    n = vn[np.asarray(idx)[band]]
    out = {"near_vertices": int(near.sum()), "band_vertices": int(band.sum())}
    if side.sum() >= 10 and (~side).sum() >= 10:
        n1, n2 = unit(np.median(n[side], 0)), unit(np.median(n[~side], 0))
        out["dihedral_deg"] = round(float(np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1)))), 1)
        out["side_counts"] = [int(side.sum()), int((~side).sum())]
    else:
        out["one_sided"] = True
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    g = json.loads((GS2 / "GOLD_SET_V2.json").read_text())
    cat = {c["scan_sha256"]: c for c in json.loads(CATALOG.read_text())["cases"]}
    rows = []
    for r in g["new_gold"]:
        prior = r.get("prior") or {}
        mc = prior.get("missing_line_causes") or {}
        if not mc or not all(v["cause"] == "MISS_PRIOR_OFF_APEX_GT_4MM" for v in mc.values()):
            continue
        gold = sc.load_gold({"kind": "ai_studio", "path": r["label_path"]}, frame_records=None)["lines"]
        pri = sc.load_gold(prior["gold_spec"], frame_records=None)["lines"]
        catalog_case = cat.get(r["asset_sha256"]) or {}
        snaps = [b.get("maximum_endpoint_snap_mm") for b in catalog_case.get("bends", [])]
        # pair each gold line with nearest parallel prior line (one-to-one on |perp offset|)
        cost = np.full((len(gold), len(pri)), 1e9)
        for i, gl in enumerate(gold):
            for j, pl in enumerate(pri):
                if angle(gl, pl) < 5.0:
                    cost[i, j] = np.linalg.norm(perp_offset(gl, pl)[0])
        gi, pj = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(gi, pj) if cost[i, j] < 1e8]
        A, b, pair_rows = [], [], []
        for i, j in pairs:
            d, a = perp_offset(gold[i], pri[j])
            M = np.eye(3) - np.outer(a, a)
            A.append(M)
            b.append(M @ d)
            pair_rows.append({"gold": gold[i]["bend_id"], "prior": pri[j]["bend_id"],
                              "angle_deg": round(angle(gold[i], pri[j]), 2),
                              "perp_offset_mm": round(float(np.linalg.norm(d)), 2),
                              "perp_offset_vec": [round(float(x), 2) for x in d],
                              "missed": gold[i]["bend_id"] in mc})
        fit = None
        if A:
            AA, bb_ = np.vstack(A), np.concatenate(b)
            t, *_ = np.linalg.lstsq(AA, bb_, rcond=1e-2)
            rank = int(np.linalg.matrix_rank(AA, tol=1e-2 * np.linalg.norm(AA, 2)))
            res = [round(float(np.linalg.norm(Ai @ t - bi)), 2) for Ai, bi in zip(A, b)]
            for pr, rr in zip(pair_rows, res):
                pr["residual_after_t_mm"] = rr
            fit = {"t": [round(float(x), 2) for x in t], "t_norm_mm": round(float(np.linalg.norm(t)), 2),
                   "constraint_rank": rank, "max_residual_mm": max(res),
                   "underdetermined": rank < 3 or len(pairs) < 3}
        # best single translation: candidates = every parallel pair's perpendicular offset and every
        # least-squares t of two non-parallel pairs; scored by the frozen matcher (TP after shift)
        thresholds, match_lines = sc._matcher()
        base_tp = match_lines(gold, pri, thresholds)["true_positive"]
        cands = []
        par = [(i, j) for i in range(len(gold)) for j in range(len(pri)) if angle(gold[i], pri[j]) < 5.0]
        for i, j in par:
            cands.append(-perp_offset(gold[i], pri[j])[0])
        for x in range(len(par)):
            for y in range(x + 1, len(par)):
                (i1, j1), (i2, j2) = par[x], par[y]
                if i1 == i2 or j1 == j2:
                    continue
                Ms, bs = [], []
                for i, j in ((i1, j1), (i2, j2)):
                    d, a = perp_offset(gold[i], pri[j])
                    M = np.eye(3) - np.outer(a, a)
                    Ms.append(M)
                    bs.append(M @ d)
                tt, *_ = np.linalg.lstsq(np.vstack(Ms), np.concatenate(bs), rcond=1e-2)
                cands.append(-tt)
        best = {"tp": base_tp, "shift": [0, 0, 0], "shift_norm_mm": 0.0}
        for c in cands:
            shifted = [dict(ln, line_start=list(np.asarray(ln["line_start"]) + c),
                            line_end=list(np.asarray(ln["line_end"]) + c)) for ln in pri]
            tp = match_lines(gold, shifted, thresholds)["true_positive"]
            if tp > best["tp"]:
                best = {"tp": tp, "shift": [round(float(x), 2) for x in c],
                        "shift_norm_mm": round(float(np.linalg.norm(c)), 2)}
        best_shift = {"unshifted_tp": base_tp, "best_tp": best["tp"], "gold_lines": len(gold),
                      "prior_lines": len(pri), "shift": best["shift"], "shift_norm_mm": best["shift_norm_mm"],
                      "candidates": len(cands)}
        scan_path = catalog_case.get("scan_path") or str(ROOT / "data" / r["asset"].split(":", 1)[1])
        if bb.sha256_file(Path(scan_path)) != r["asset_sha256"]:
            raise SystemExit(f"scan sha mismatch {r['scan_id']}")
        v, vn = load_scan(scan_path)
        tree = cKDTree(v)
        surf = {}
        for name, lines in (("prior", pri), ("gold", gold)):
            surf[name] = {}
            for ln in lines:
                dd, _ = tree.query(samples(ln))
                surf[name][ln["bend_id"]] = {"median_mm": round(float(np.median(dd)), 2),
                                             "max_mm": round(float(dd.max()), 2),
                                             "geometry": local_geometry(ln, v, vn, tree)}
        rows.append({"scan_id": r["scan_id"], "family": r["family"], "legacy_ref": prior.get("legacy_ref"),
                     "legacy_tier": prior.get("legacy_tier"), "gold_lines": len(gold), "prior_lines": len(pri),
                     "missed_gold": sorted(mc), "pairs": pair_rows, "translation_fit": fit,
                     "best_single_translation": best_shift,
                     "catalog_case": catalog_case.get("case_id"),
                     "catalog_frame": catalog_case.get("coordinate_frame"),
                     "catalog_max_endpoint_snap_mm": max([s for s in snaps if s is not None], default=None),
                     "surface_distance": surf})
    report = {"schema": "sherman.gold_set_v2.catalog_offset_probe.v1", "created_at": bb.utc_now(),
              "catalog": str(CATALOG.relative_to(ROOT)), "catalog_sha256": bb.sha256_file(CATALOG),
              "scans": len(rows), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for row in rows:
        f = row["translation_fit"] or {}
        bs = row["best_single_translation"]
        print("BEST", row["scan_id"], f"tp {bs['unshifted_tp']}->{bs['best_tp']} of g{bs['gold_lines']}/p{bs['prior_lines']}",
              "shift", bs["shift_norm_mm"])
        print(row["scan_id"], row["family"], f"g{row['gold_lines']}/p{row['prior_lines']}",
              "t=", f.get("t_norm_mm"), "rank", f.get("constraint_rank"), "res", f.get("max_residual_mm"),
              "offs", [p["perp_offset_mm"] for p in row["pairs"]],
              "prior_surf_med", [s["median_mm"] for s in row["surface_distance"]["prior"].values()],
              "dih", [ (s["geometry"] or {}).get("dihedral_deg") for s in row["surface_distance"]["prior"].values()],
              "snap", row["catalog_max_endpoint_snap_mm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
