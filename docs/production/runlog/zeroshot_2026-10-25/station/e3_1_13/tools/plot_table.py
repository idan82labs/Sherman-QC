"""Verdict-stability table + 23553000 B decision grid per arm. usage: plot_table.py ARMS.json OUT.png"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
A = json.load(open(sys.argv[1]))
ARMS = [("base", "e3-1.12.0 (live)"), ("root", "1.13 root fix only"), ("net", "1.13 root + safety net (a, K=2)"), ("optc", "1.13 root + honest U (c, K=3)")]
VARS = ["native", "s95_9501", "s95_9502", "s95_9503", "s95_9504", "s95_9505", "thin040"]
COL = {"PASS": "#4c9a5f", "FAIL": "#c0392b", "REVIEW": "#e0a526", "UNKNOWN": "#9a9a9a", "NOT_APPLICABLE": "#dddddd", None: "#ffffff"}
plt.rcParams.update({"font.size": 9.5})
fig = plt.figure(figsize=(15, 13.5), dpi=110)
ax = fig.add_axes([0.01, 0.77, 0.98, 0.19]); ax.axis("off")
hdr = ["arm", "chars FAIL<->PASS", "chars FAIL->PASS", "scans FAIL<->PASS", "chars any change", "coverage loss\n(native, of decided)", "G1 false PASS\n(7 realizations)", "E3 wall (s)\nP50 / P95 / max"]
rows = []
for k, nm in ARMS:
    a = A["arms"][k]; al = a["all"]; g = a["g1"]; w = a.get("wall_s") or {}
    rows.append([nm, al.get("chars_fail_pass", 0), al.get("chars_fail_to_pass", 0), al.get("scans_fail_pass", 0), al.get("chars_any_change", 0),
                 f'{al.get("coverage_loss_native_pct", 0):.2f} %', g.get("native_false_pass", 0) + g.get("perturbed_false_pass", 0),
                 f'{w.get("p50", 0):.0f} / {w.get("p95", 0):.0f} / {w.get("max", 0):.0f}'])
t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center", colWidths=[0.22] + [0.111] * 7)
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 2.1)
for (r, c), cell in t.get_celld().items():
    if r == 0: cell.set_facecolor("#e8e8e8")
    if c == 0: cell.set_text_props(ha="left")
ax.set_title("Verdict stability over 88 scans x 7 realizations (native, 5 x 95 % subsample, 0.40 mm thinning)", loc="left", fontsize=11)
bm = A["b_modes"]
for pi, (sid, title) in enumerate((("pc_6eba02723d4b", "23553000 B full scan"), ("pc_186463746daa", "23553000 B scan 2"))):
    rows = bm.get(sid, {})
    names = [n for n in rows if any(rows[n]["base"][v] != rows[n]["base"]["native"] or rows[n]["root"][v] != rows[n]["base"]["native"] for v in VARS)]
    names = sorted(names)[:12]
    ax = fig.add_axes([0.07 + pi * 0.49, 0.05, 0.42, 0.56]); ax.set_xlim(0, len(ARMS) * 7 + 3); ax.set_ylim(0, len(names)); ax.invert_yaxis(); ax.axis("off")
    for ai, (k, nm) in enumerate(ARMS):
        x0 = ai * 7.75
        ax.text(x0 + 3.5, 1.075, nm.replace("1.13 ", "1.13\n").replace(" (", "\n("), ha="center", va="bottom", fontsize=7.5, transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes))
        for vi, v in enumerate(VARS):
            ax.text(x0 + vi + 0.5, -0.1, ["nat", "s1", "s2", "s3", "s4", "s5", "thin"][vi], ha="center", va="bottom", fontsize=6.5, rotation=90)
            for ri, n in enumerate(names):
                d = rows[n][k][v]
                ax.add_patch(Rectangle((x0 + vi, ri), 0.95, 0.9, color=COL.get(d, "#ffffff")))
    for ri, n in enumerate(names):
        ax.text(-0.3, ri + 0.45, n.replace("FLANGE_LENGTH", "flange").replace("INCLUDED_ANGLE", "angle").replace("BEND_LINE_POSITION", "line pos"), ha="right", va="center", fontsize=8)
    ax.set_title(title + ": characteristics that change under 1.12 or 1.13 root", loc="center", fontsize=10.5, pad=75)
hs = [Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("PASS", "FAIL", "REVIEW", "UNKNOWN")]
fig.legend(hs, ["PASS", "FAIL", "REVIEW", "UNKNOWN"], loc="lower center", ncol=4, frameon=False)
fig.savefig(sys.argv[2]); print("ok")
