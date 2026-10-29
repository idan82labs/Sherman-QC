"""Distances of each gold line (at every 2 mm station on its target fold) to the candidate bend loci:
outer apex (arc midpoint), outer virtual sharp Xo, mid-surface apex, inner virtual sharp Xi,
inner apex (inner arc midpoint), arc centre C (bend axis). Signed position along the bisector
(+ = concave/inside) is reported relative to the outer apex."""
import json
import math
import numpy as np
from a3_foldlib import FOLDS, fit_station, load_gold, unit, OUT, R, o, AX

G = load_gold()
TARGET = {"M01": "LIP_+b (M01 side)", "M02": "LIP_-b (M02 side)", "M03": "TAB_+b (window b 64..82)",
          "M04": "TAB_+b (window b 64..82)", "M05": "TAB_-b (window b -90..-73)", "M06": "TAB_-b (window b -90..-73)",
          "M07": "WALL a~113 (M07 side)", "M08": "WALL a~113 (M07 side)"}
res = {}
for gid, (s, e) in G.items():
    fd = FOLDS[TARGET[gid]]
    s2, e2 = (s - o) @ R.T, (e - o) @ R.T
    ax = AX[fd["axis"]]
    lo, hi = sorted([s2[ax], e2[ax]])
    rows = []
    for v in np.arange(math.ceil(lo / 2) * 2, hi, 2.0):
        r = fit_station(fd, float(v))
        if not r.get("ok"):
            continue
        t = (v - s2[ax]) / (e2[ax] - s2[ax])
        gp = s2 + t * (e2 - s2)
        P = np.array([gp[AX[fd["x"]]], gp[AX[fd["y"]]]])
        g = r["g"]
        half = math.radians(r["included_deg"]) / 2
        apex, Xo, Ro, th = r["apex"], r["Xo"], r["Ro"], r["t"]
        loci = {"outer_apex": apex, "outer_sharp_Xo": Xo, "mid_apex": apex + 0.5 * th * g,
                "inner_sharp_Xi": Xo + (th / math.sin(half)) * g, "inner_apex": apex + th * g,
                "arc_centre_C": Xo + (Ro / math.sin(half)) * g}
        row = {"v": float(v)}
        for k, L in loci.items():
            row["d_" + k] = float(np.linalg.norm(P - L))
            row["pos_" + k] = float((L - apex) @ g)
        row["gold_pos_along_bisector"] = float((P - apex) @ g)
        row["gold_across"] = float((P - apex) @ np.array([-g[1], g[0]]))
        row["Ro"], row["t"], row["bend_deg"] = Ro, th, r["bend_deg"]
        rows.append(row)
    if not rows:
        continue
    keys = [k for k in rows[0] if k.startswith("d_")] + ["gold_pos_along_bisector", "gold_across", "Ro", "t", "bend_deg"]
    med = {k: float(np.median([q[k] for q in rows])) for k in keys}
    pos = {k: float(np.median([q[k] for q in rows])) for k in rows[0] if k.startswith("pos_")}
    res[gid] = {"target": TARGET[gid], "n": len(rows), "median": med, "locus_pos_along_bisector_from_apex": pos}
    print(gid, TARGET[gid], "n", len(rows))
    print("   gold along-bisector %+.2f across %+.2f | Ro %.2f t %.2f bend %.1f" % (
        med["gold_pos_along_bisector"], med["gold_across"], med["Ro"], med["t"], med["bend_deg"]))
    print("   locus positions (from apex, + inside):", {k[4:]: round(v, 2) for k, v in pos.items()})
    print("   gold distance to loci:", {k[2:]: round(v, 2) for k, v in med.items() if k.startswith("d_")})
(OUT / "a3_loci.json").write_text(json.dumps(res, indent=1))
