"""Verify bench_labeling lines against the raw scan (read-only).

For every ok bend: march stations every `step` mm along the label line, from t0-ext to t1+ext.
At each station take raw points in a slab |s - t| <= half and within `reach` mm of the line, and record
  d_surf : distance from the label line to the nearest scan point in the slab (0 = line on the surface)
  arc    : number of points within `arc_tol` mm of the label line whose normal is within 35 deg of the
           outward bisector (-bis): the scanned outer bend arc at the apex.
Fold extent = longest run of arc-present stations (gaps <= gap_ok mm bridged).
Writes verify_<tag>.json next to the lines dir.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

cache, lines_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ext = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
STEP = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
ONLY = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else None
d = np.load(cache); Q = d["Q"]; NQ = d["NQ"].astype(float); R = d["R"]; o = d["o"]
P = o + Q @ R          # raw coordinates
NRM = NQ @ R
doc = json.loads(lines_path.read_text())
step, half, reach, arc_tol, gap_ok = STEP, 0.5, 6.0, 0.6, 6.0
res = {}
for b in doc["bends"]:
    if b.get("status") != "ok" or (ONLY and b["bend_id"] not in ONLY):
        continue
    inner = b.get("skin") == "inner"
    dvec = np.asarray(b["direction"]); bis = np.asarray(b["bisector"])
    a0 = np.asarray(b["label_line"]["start"]); a1 = np.asarray(b["label_line"]["end"])
    t0, t1 = b["t_range"]
    origin = a0 - t0 * dvec           # point on the label line at t = 0
    rel = P - origin
    s = rel @ dvec
    w = rel - np.outer(s, dvec)
    dist = np.linalg.norm(w, axis=1)
    near = (dist <= reach) & (s >= t0 - ext - 1) & (s <= t1 + ext + 1)
    sN, dN, nN = s[near], dist[near], NRM[near]
    order = np.argsort(sN); sN, dN, nN = sN[order], dN[order], nN[order]
    out_ok = (nN @ (bis if inner else -bis)) >= np.cos(np.radians(35.0))  # inner (concave) skin: outward normal along +bis
    st = np.arange(t0 - ext, t1 + ext + 1e-9, step)
    rows = []
    for c in st:
        i0, i1 = np.searchsorted(sN, [c - half, c + half])
        dd = dN[i0:i1]
        arc = int(((dd <= arc_tol) & out_ok[i0:i1]).sum())
        rows.append((float(c), float(dd.min()) if len(dd) else None, arc))
    present = np.array([r[2] >= 3 for r in rows])
    # runs with small gaps bridged
    runs, k = [], 0
    while k < len(st):
        if not present[k]:
            k += 1; continue
        j = k
        while j < len(st) and present[j]:
            j += 1
        runs.append([float(st[k]), float(st[j - 1])]); k = j
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap_ok:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    main = max(merged, key=lambda r: r[1] - r[0]) if merged else None
    inside = [(c, ds, a) for (c, ds, a) in rows if t0 <= c <= t1]
    dsv = np.array([r[1] for r in inside if r[1] is not None])
    on = np.array([(r[1] is not None and r[1] <= 0.5 and r[2] >= 3) for r in inside])
    fold_len = (main[1] - main[0] + step) if main else 0.0
    overlap = max(0.0, min(t1, main[1] + step / 2) - max(t0, main[0] - step / 2)) if main else 0.0
    res[b["bend_id"]] = {
        "note": b.get("note", "")[:80],
        "t_range": [round(t0, 2), round(t1, 2)], "length_mm": round(t1 - t0, 1),
        "apex_offset_mm": b.get("apex_offset_mm"),
        "fold_runs_t": [[round(a, 1), round(c, 1)] for a, c in merged],
        "fold_main_t": [round(main[0], 1), round(main[1], 1)] if main else None,
        "fold_main_len_mm": round(fold_len, 1),
        "line_on_apex_frac": round(float(on.mean()), 3) if len(on) else None,
        "fold_coverage_by_line": round(overlap / fold_len, 3) if fold_len else None,
        "line_inside_fold_frac": round(overlap / (t1 - t0), 3),
        "d_surf_median": round(float(np.median(dsv)), 3) if len(dsv) else None,
        "d_surf_p90": round(float(np.percentile(dsv, 90)), 3) if len(dsv) else None,
        "d_surf_max": round(float(dsv.max()), 3) if len(dsv) else None,
        "profile": [[round(c, 1), None if ds is None else round(ds, 2), a] for (c, ds, a) in rows],
    }
    r = res[b["bend_id"]]
    print(b["bend_id"], r["note"][:40], "| line t", r["t_range"], "len", r["length_mm"], "| fold runs", r["fold_runs_t"],
          "| on_apex", r["line_on_apex_frac"], "cov", r["fold_coverage_by_line"], "inside", r["line_inside_fold_frac"],
          "| d_surf med/p90/max", r["d_surf_median"], r["d_surf_p90"], r["d_surf_max"])
out_path.write_text(json.dumps(res, indent=1))
