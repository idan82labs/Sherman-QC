"""Auditor-1 (sc_71cf0b216f43): z-buffered overview renders (part frame) with human gold (green, labelled)
and auditor-1 apex lines (magenta = 8 wall folds, cyan = 4 joggle folds)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_common import OUT, get
from a1_zbuf import render, project

P, VN, F, G, R, c = get()
ap = json.load(open(OUT + "/a1_apex_lines.json"))["folds"]
jg = json.load(open(OUT + "/a1_joggles.json"))["joggles"]
apex = {fid: np.array(v["apex_line_part"]) for fid, v in ap.items()}
jog = {}
for name, v in jg.items():
    for fold, ln in v["fold_lines_topskin_virtual_sharp"].items():
        e = np.array(ln["part"]); e[:, 0] = np.clip(e[:, 0], -160.5, 145.8)
        jog[f"{name[:5]}_{fold}"] = e
views = [(-90, 89.5, "TOP view (+z): top flanges / lips / joggles"), (-90, -89.5, "BOTTOM view (-z): frame flange with window"),
         (-90, 25, "front long wall (-y) outside, from above"), (90, 25, "back long wall (+y) outside, from above"),
         (180, 20, "left short wall (-x) outside"), (0, 20, "right short wall (+x) outside"),
         (-60, 55, "iso from front-right-top"), (120, 55, "iso from back-left-top")]
fig, axes = plt.subplots(4, 2, figsize=(26, 30), dpi=72)
for ax, (az, el, lab) in zip(axes.ravel(), views):
    img, ext, B = render(P, VN, az, el, px=0.3, splat=3)
    ax.imshow(img, extent=ext, origin="upper", interpolation="nearest")
    for bid, s, e in G:
        xs, ys = project(np.array([s, e]), B)
        ax.plot(xs, ys, color="#00b050", lw=2.0)
        ax.text(xs.mean(), ys.mean(), bid, color="white", fontsize=9, bbox=dict(fc="#00b050", ec="none", pad=1))
    for fid, e in apex.items():
        xs, ys = project(e, B); ax.plot(xs, ys, color="magenta", lw=1.0, ls="--")
    for fid, e in jog.items():
        xs, ys = project(e, B); ax.plot(xs, ys, color="cyan", lw=1.0, ls="--")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_title(lab + f" (az={az}, el={el})", fontsize=12)
fig.suptitle("sc_71cf0b216f43 (10839000) auditor-1: green = human gold M01-M08, magenta dashed = measured outer-apex lines of the 8 wall folds, "
             "cyan dashed = 4 joggle folds (not in gold)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(OUT + "/a1_overview_zbuf.png")
print("ok")
