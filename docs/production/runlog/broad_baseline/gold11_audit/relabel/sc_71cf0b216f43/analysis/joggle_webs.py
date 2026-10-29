"""Locate the joggle web skins (raw + ds) in the part frame and propose plane_from_points seeds."""
import json, sys
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parent
d = np.load(sys.argv[1]); Q = d["Q"]; NQ = d["NQ"].astype(float); R = d["R"]; o = d["o"]
PREP = Path("/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/prep/sc_71cf0b216f43")
ds = np.load(PREP / "points_ds.npz"); Pd = (ds["points"] - o) @ R.T; Nd = ds["normals"] @ R.T; Ld = ds["labels"]

def to_raw(q):
    return (o + np.asarray(q) @ R).tolist()

out = {}
for name, yc, sgn in (("S9", -77.7, +1), ("S10", 83.6, -1)):
    for skin, sk in (("top", +1), ("bottom", -1)):
        th = np.radians(45.0)
        nexp = sk * np.array([0.0, sgn * np.sin(th), np.cos(th)])
        for lab, PP, NN in (("raw", Q, NQ), ("ds", Pd, Nd)):
            m = (np.abs(PP[:, 1] - yc) < 3.0) & (PP[:, 2] > 18.0) & (PP[:, 2] < 23.5) & (np.abs(PP[:, 0]) < 140)
            cosang = NN[m] @ nexp
            good = cosang > np.cos(np.radians(25))
            pts = PP[m][good]
            nn = NN[m][good]
            if len(pts) == 0:
                print(name, skin, lab, "none"); continue
            nm = nn.mean(0); nm /= np.linalg.norm(nm)
            ang = np.degrees(np.arccos(np.clip(nm @ np.array([0, 0, 1.0]) * sk, -1, 1)))
            # coverage along x in 5 mm bins
            bins = np.arange(-140, 141, 5)
            occ = np.histogram(pts[:, 0], bins=bins)[0]
            print(f"{name} {skin:6s} {lab:3s} n={len(pts):6d} y_med={np.median(pts[:,1]):.2f} z_med={np.median(pts[:,2]):.2f} "
                  f"mean_normal={np.round(nm,3)} tilt={ang:.1f} x-bins occupied {np.mean(occ>0):.2f} min/bin {occ.min()}")
            if lab == "raw":
                seeds = []
                for x0 in (-100, -50, 0, 50, 100):
                    sel = np.abs(pts[:, 0] - x0) < 2.0
                    if sel.sum():
                        q = np.median(pts[sel], axis=0)
                        seeds.append({"x": x0, "q": np.round(q, 3).tolist(), "raw": np.round(to_raw(q), 3).tolist(), "n": int(sel.sum())})
                out[f"{name}_{skin}"] = {"y_med": float(np.median(pts[:, 1])), "z_med": float(np.median(pts[:, 2])),
                                         "normal_part": nm.tolist(), "seeds": seeds}
(A / "joggle_webs.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, [ (s["x"], s["raw"], s["n"]) for s in v["seeds"]])
