"""Light probe on the prep's downsampled cloud: per-station cross-section of a lines.json bend.

Reports, per 1 mm station along the bend's sharp line, where the scanned surface is along the
corner bisector (the tool's apex rule, restricted to that station) so fold segments and
window/ear stations can be told apart. Read-only; no scan reload."""
import json, sys
sys.dont_write_bytecode = True
import numpy as np

lines_path, prep_dir, bend_id = sys.argv[1:4]
step = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
L = json.load(open(lines_path))
b = next(x for x in L["bends"] if x["bend_id"] == bend_id)
pts = np.load(prep_dir + "/points_ds.npz")["points"]
p0 = np.array(b["line_origin"]); d = np.array(b["direction"]); bis = np.array(b["bisector"])
ua = np.array(b["side_dirs"]["plane_a"]); ub = np.array(b["side_dirs"]["plane_b"])
rel = pts - p0
s = rel @ d
w = rel - np.outer(s, d)
h = w @ bis
lat = np.linalg.norm(w - np.outer(h, bis), axis=1)
lo, hi = b["t_range"]
print(bend_id, "t_range", [round(lo, 2), round(hi, 2)], "apex_offset", round(b["apex_offset_mm"], 3),
      "expected outer", round(b["expected_apex_offset_mm"]["outer"], 3))
rows = []
for c in np.arange(lo - 6, hi + 6, step):
    m = (np.abs(s - c) <= step / 2) & (lat <= 0.6) & (h >= -1.0) & (h <= 6.0)
    hv = np.sort(h[m])
    # surface points near the sharp along the bisector (ear / flat plate through the sharp) vs arc
    n_ear = int(((hv >= -1.0) & (hv <= 0.3)).sum())
    n_arc = int(((hv > 0.5) & (hv <= 1.6)).sum())
    first = float(hv[0]) if len(hv) else None
    rows.append((round(float(c), 1), len(hv), first if first is None else round(first, 2), n_ear, n_arc))
for r in rows:
    print("t=%7.1f n=%3d first_h=%s near_sharp(h<=0.3)=%3d arc(0.5<h<=1.6)=%3d" % r)
