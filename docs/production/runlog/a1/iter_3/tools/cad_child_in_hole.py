"""CAD-only: which bends' children lie inside a hole/window of their parent panel when flat (S0)?"""
import json, sys
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
C = Path(sys.argv[1])
for d in sorted(C.iterdir()):
    f = d / "compiled_part.json"
    if not f.exists():
        continue
    cp = json.loads(f.read_text())
    if cp.get("status") != "COMPILED":
        continue
    panels = {p["panel_id"]: p for p in cp["panels"]}
    s0 = cp["stages"][0]
    T = {k: np.asarray(v) for k, v in s0["panel_transforms"].items()}
    out = []
    for b in cp["bends"]:
        par, ch = panels[b["parent_panel_id"]], panels[b["child_panel_id"]]
        if not par["holes_mm"]:
            continue
        Tp, Tc = T[b["parent_panel_id"]], T[b["child_panel_id"]]
        # parent plane frame at S0
        ob = np.asarray(par["outer_boundary_mm"]) @ Tp[:3, :3].T + Tp[:3, 3]
        c0 = ob.mean(axis=0)
        u, s, vt = np.linalg.svd(ob - c0)
        e1, e2 = vt[0], vt[1]
        cc = (np.asarray(ch["outer_boundary_mm"]) @ Tc[:3, :3].T + Tc[:3, 3])
        q = np.stack([(cc - c0) @ e1, (cc - c0) @ e2], axis=1)
        for hi, h in enumerate(par["holes_mm"]):
            hh = np.asarray(h) @ Tp[:3, :3].T + Tp[:3, 3]
            poly = np.stack([(hh - c0) @ e1, (hh - c0) @ e2], axis=1)
            inside = MplPath(poly).contains_points(q, radius=1e-6)
            if inside.mean() > 0.3:
                out.append(f"{b['label']}({b['parent_panel_id']}->{b['child_panel_id']}) child S0 outline {inside.mean():.0%} inside parent hole {hi}")
    print(d.name, "|", "; ".join(out) if out else "-")
