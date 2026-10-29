"""CAD bend count under the 'one line per collinear bend' convention: V10 bends
(+ lenient extras) whose canonical lines are collinear (axis angle < 2 deg,
perpendicular offset < tol mm) and have the same nominal angle are merged."""
import json, sys
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[3] / "docs/production/runlog/broad_baseline"


def lines(R):
    out = []
    v10f = set()
    for i, a in enumerate(R["v10_bend_axes"]):
        v10f |= {int(x) for x in a["bend_id"].split("_F", 1)[1].split("_")}
        out.append((f"C{i+1}", np.array(a["line_start"]), np.array(a["line_end"]), a["angular_sweeps_deg"][0]))
    k = 0
    for g in R["lenient_coaxial_groups"]:
        if g["kind"] == "BEND_CANDIDATE" and not set(g["face_indices"]) & v10f:
            k += 1
            out.append((f"L{k}", np.array(g["canonical_line"][0]), np.array(g["canonical_line"][1]), max(g["sweep_by_radius_deg"].values())))
    return out


def merge(ls, tol):
    n = len(ls); parent = list(range(n))
    def f(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            _, s1, e1, a1 = ls[i]; _, s2, e2, a2 = ls[j]
            d1 = (e1 - s1) / np.linalg.norm(e1 - s1); d2 = (e2 - s2) / np.linalg.norm(e2 - s2)
            if abs(d1 @ d2) < np.cos(np.radians(2)) or abs(a1 - a2) > 3: continue
            off = max(np.linalg.norm((p - s1) - d1 * ((p - s1) @ d1)) for p in (s2, e2))
            if off < tol: parent[f(i)] = f(j)
    groups = {}
    for i in range(n): groups.setdefault(f(i), []).append(ls[i][0])
    return sorted(groups.values(), key=lambda g: g[0])


res = {}
for raw in sorted(O.glob("cad_crosscheck_raw/[0-9]*.json")):
    R = json.loads(raw.read_text()); ls = lines(R)
    g1 = merge(ls, 1.0)
    res[raw.stem] = {"v10_plus_lenient": len(ls), "collinear_merged_count": len(g1), "collinear_groups": [g for g in g1 if len(g) > 1]}
    print(raw.stem, len(ls), len(g1), [g for g in g1 if len(g) > 1])
(O / "cad_crosscheck_match/_collinear_merge.json").write_text(json.dumps(res, indent=1))
