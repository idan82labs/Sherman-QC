#!/usr/bin/env python3
"""Auditor 3, sc_08686dc1d5d4: dense perpendicular sections along every gold line (read-only).

For each gold line, stations every STEP mm from -EXT to L+EXT along the line. At each station the
slab |(p-c).d| < SLAB is projected into the section plane (a = in-plate perpendicular, w = plate
normal), and the shared corner fit (gold11_audit/scripts/render_gold_xsections.fit_section) is run
on points within 22 mm of the gold point. Rows: fold present?, bend angle, outer apex / arc centre
/ virtual sharps relative to the gold point, nearest-scan distance. Writes stations_<id>.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/scripts")
from a3_global import gold_lines, load_points  # noqa: E402
from render_gold_xsections import fit_section  # noqa: E402

STEP = 5.0
EXT = 30.0
SLAB = 0.75


def unit(v):
    return v / np.linalg.norm(v)


def main():
    from scipy.spatial import cKDTree

    P = load_points()
    F = np.load(HERE / "_plate_frame.npy")
    o, e1, e2, n = F
    tree = cKDTree(P)
    spacing = 0.52
    out = {}
    for bid, s0, s1 in gold_lines():
        d = unit(s1 - s0)
        L = float(np.linalg.norm(s1 - s0))
        w = unit(n - d * (n @ d))  # plate normal in the section plane
        a = np.cross(w, d)  # in-plate perpendicular
        rows = []
        for s in np.arange(-EXT, L + EXT + 1e-6, STEP):
            c = s0 + s * d
            idx = np.array(tree.query_ball_point(c, 45.0), int)
            row = {"s": float(s), "frac": float(s / L), "u": float((c - o) @ e1), "v": float((c - o) @ e2)}
            if len(idx) == 0:
                row["fit_ok"] = False
                row["why"] = "no_points"
                rows.append(row)
                continue
            Q = P[idx] - c
            m = np.abs(Q @ d) < SLAB
            q = np.c_[Q[m] @ a, Q[m] @ w]
            row["n_slab"] = int(m.sum())
            nd, _ = tree.query(c)
            row["d_nearest_scan"] = float(nd)
            qf = q[np.linalg.norm(q, axis=1) < 22]
            fit = fit_section(qf, spacing) if len(qf) >= 40 else {"ok": False, "why": "too_few"}
            row["fit_ok"] = bool(fit.get("ok"))
            row["why"] = fit.get("why")
            if fit.get("ok"):
                g = fit["g"]
                row["bend_angle_deg"] = float(fit["bend_angle_deg"])
                for k in ("apex", "Xo", "Xi", "Xm", "C"):
                    v = fit.get(k)
                    row[k] = None if v is None else [float(v[0]), float(v[1])]
                    row["d_" + k] = None if v is None else float(np.linalg.norm(v))
                row["R_outer"] = fit.get("R_outer")
                row["two_skins"] = fit.get("two_skins")
                row["g"] = [float(g[0]), float(g[1])]
                if fit.get("apex") is not None:
                    vv = -np.asarray(fit["apex"])
                    row["gold_minus_apex_along_bisector"] = float(vv @ g)
                    row["gold_minus_apex_across"] = float(vv @ np.array([-g[1], g[0]]))
                    ap3 = c + fit["apex"][0] * a + fit["apex"][1] * w
                    row["apex_xyz"] = ap3.tolist()
                    row["apex_uvh"] = [float((ap3 - o) @ e1), float((ap3 - o) @ e2), float((ap3 - o) @ n)]
                row["gold_uvh"] = [float((c - o) @ e1), float((c - o) @ e2), float((c - o) @ n)]
                row["fold_here"] = bool(55 <= fit["bend_angle_deg"] <= 125 and fit.get("Xo") is not None
                                        and np.linalg.norm(fit["Xo"]) < 15)
            else:
                row["fold_here"] = False
                row["gold_uvh"] = [float((c - o) @ e1), float((c - o) @ e2), float((c - o) @ n)]
            rows.append(row)
        out[bid] = {"L": L, "d": d.tolist(), "a": a.tolist(), "w": w.tolist(), "rows": rows}
        on = [r for r in rows if r.get("fold_here")]
        print(bid, f"L={L:.1f}", "fold stations", len(on), "of", len(rows),
              "s-range", (min(r['s'] for r in on), max(r['s'] for r in on)) if on else None)
        for r in rows:
            if r.get("fit_ok"):
                print(f"  s={r['s']:7.1f} u={r['u']:8.1f} ang={r['bend_angle_deg']:6.1f} d_apex={r.get('d_apex') or float('nan'):5.2f} "
                      f"along={r.get('gold_minus_apex_along_bisector', float('nan')):5.2f} across={r.get('gold_minus_apex_across', float('nan')):5.2f} "
                      f"dC={r.get('d_C') or float('nan'):5.2f} dXo={r.get('d_Xo') or float('nan'):5.2f} dXi={r.get('d_Xi') or float('nan'):5.2f} "
                      f"scan={r['d_nearest_scan']:.2f} fold={r['fold_here']}")
            else:
                print(f"  s={r['s']:7.1f} u={r['u']:8.1f} no fit ({r.get('why')}) scan={r.get('d_nearest_scan', float('nan')):.2f}")
    (HERE / "stations_all.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
