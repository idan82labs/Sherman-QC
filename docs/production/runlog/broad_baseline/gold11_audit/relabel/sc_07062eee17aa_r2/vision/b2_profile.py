"""B2 (r2) per-2 mm profile: surface along the bisector from the label line, + = line in free air. Read-only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from research.bench_labeling.scanio import load_scan
HERE = Path(__file__).resolve().parent
P = load_scan(json.loads((HERE.parent / "lines.json").read_text())["scan_path"]).points
for run in ("r1", "r2"):
    lj = HERE.parent / "lines.json" if run == "r2" else HERE.parent.parent / "sc_07062eee17aa" / "lines.json"
    LJ = json.loads(lj.read_text())
    for b in LJ["bends"][:2]:
        s0 = np.array(b["label_line"]["start"]); e0 = np.array(b["label_line"]["end"]); d = np.array(b["direction"]); bis = np.array(b["bisector"])
        L = np.linalg.norm(e0 - s0); lat = np.cross(d, bis)
        rel = P - s0; t = rel @ d; w = rel - np.outer(t, d); xb, xl = w @ bis, w @ lat
        keep = (np.abs(xl) <= 0.35) & (xb > -3) & (xb < 4)
        ts, hs = [], []
        for a in np.arange(0, L, 2.0):
            m = keep & (t >= a) & (t < a + 2)
            ts.append(a + 1); hs.append(xb[m].min() if m.any() else np.nan)
        hs = np.array(hs); ok = np.isfinite(hs)
        print(run, b["bend_id"], "stations", ok.sum(), "/", len(hs), "gap t:", [round(x) for x, o in zip(ts, ok) if not o][:3], "...",
              "surface offset median %.2f  |.| median %.2f  p90 %.2f  frac|.|<=0.5 %.2f" % (np.nanmedian(hs), np.nanmedian(np.abs(hs)), np.nanpercentile(np.abs(hs), 90), np.mean(np.abs(hs[ok]) <= 0.5)))
        if b["bend_id"] == "B2":
            plt.plot(ts, hs, ".-", ms=2, label=f"{run} B2")
plt.axhline(0, c="k"); plt.xlabel("t from line start (mm)"); plt.ylabel("outer skin along bisector from label line (mm); + = line in free air")
plt.legend(); plt.title("B2 -Y flange: label line vs outer skin, every 2 mm (gap = notch)"); plt.grid(alpha=.3)
plt.gcf().set_size_inches(12, 4); plt.tight_layout(); plt.savefig(HERE / "b2_profile_r1_vs_r2.png", dpi=100)
