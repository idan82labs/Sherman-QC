"""Run the auditor-2 section analysis along every gold line; print + save JSON."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json

import numpy as np

from a2_common import OUT, get_scan, gold_lines, perp_basis, unit
from a2_section import analyse_section, cyl_select, frame2d

xyz, nrm, faces = get_scan()
G = gold_lines()
only = sys.argv[1:] or sorted(G)
res = {}
for bid in only:
    a, b = G[bid]
    L = float(np.linalg.norm(b - a))
    idx, t, d, _ = cyl_select(xyz, nrm, a, b, radius=22.0, ext=25.0)
    e1, e2 = perp_basis(d)
    step = 2.0 if L > 40 else 1.0
    stations = np.arange(-20.0, L + 20.0 + 1e-6, step)
    rows = []
    for s in stations:
        k = np.abs(t - s) < 0.6
        if k.sum() < 20:
            rows.append({"s": float(s), "ok": False, "why": "empty", "n_pts": int(k.sum())})
            continue
        q, m, an = frame2d(xyz, nrm, idx[k], a, d, e1, e2, s)
        r = analyse_section(q, m, an, rmin=3.5, rmax=13.0)
        r["s"] = float(s)
        rows.append(r)
    res[bid] = {"length": L, "rows": rows}
    print(f"== {bid} L={L:.1f}")
    print("   s    ok  ang   tA   tB  g-apx(bis,lat)  g->Xo g->Xi  near  apexN")
    for r in rows:
        if not r.get("ok"):
            print(f"{r['s']:6.1f}  --  {r.get('why')} n={r.get('n_pts')}")
            continue
        print(f"{r['s']:6.1f}  ok {r['bend_angle_deg']:5.1f} {r.get('thickA', float('nan')):4.2f} "
              f"{r.get('thickB', float('nan')):4.2f}  "
              f"{r.get('gold_minus_apex_bis', float('nan')):6.2f} {r.get('gold_minus_apex_lat', float('nan')):6.2f}  "
              f"{r['gold_to_Xo']:5.2f} {r.get('gold_to_Xi', float('nan')):5.2f} {r['gold_to_nearest_scan']:5.2f} "
              f"{r.get('apex_n', 0):4d}")
(OUT / "a2_sections.json").write_text(json.dumps(res, indent=1, default=float))
