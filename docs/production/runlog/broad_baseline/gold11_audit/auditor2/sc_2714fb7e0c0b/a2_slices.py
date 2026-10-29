"""Generic slab slices in the part frame with gold lines (red) and best_f1 lines (orange).

usage: a2_slices.py OUT.png NCOLS "axis=val,half=h,x=lo:hi,y=lo:hi,title" ...
axis in u/v/w; the two remaining axes (in u,v,w order) are plotted as x,y.
Gold lines crossing the slab are drawn as a star; lines lying in the slab are drawn as segments.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a2_common import load, gold_lines, pred_lines, OUT

AX = {"u": 0, "v": 1, "w": 2}

def parse(spec):
    d = {}
    for part in spec.split(","):
        k, v = part.split("=", 1)
        d[k.strip()] = v.strip()
    return d

def main():
    out = OUT / sys.argv[1]
    ncols = int(sys.argv[2])
    specs = [parse(s) for s in sys.argv[3:]]
    q, R, O, raw = load()
    G = gold_lines(R, O)
    P = pred_lines(R, O)
    nrows = (len(specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.0 * nrows), dpi=95, squeeze=False)
    for ax, sp in zip(axes.ravel(), specs):
        a = [k for k in ("u", "v", "w") if k in sp][0]
        ia = AX[a]
        val = float(sp[a]); half = float(sp.get("half", 0.5))
        others = [i for i in range(3) if i != ia]
        xr = [float(t) for t in sp["x"].split(":")]
        yr = [float(t) for t in sp["y"].split(":")]
        m = (np.abs(q[:, ia] - val) < half) & (q[:, others[0]] > xr[0]) & (q[:, others[0]] < xr[1]) \
            & (q[:, others[1]] > yr[0]) & (q[:, others[1]] < yr[1])
        ax.scatter(q[m, others[0]], q[m, others[1]], s=float(sp.get("s", 2)), c="k", linewidths=0)
        for lines, col, ms in ((G, "#d62728", 12), (P, "orange", 8)):
            for k, (s, e) in lines.items():
                dvec = e - s
                L = np.linalg.norm(dvec)
                if abs(dvec[ia]) > 0.3 * L:
                    t = (val - s[ia]) / dvec[ia]
                    if -0.02 <= t <= 1.02:
                        x = s + t * dvec
                        ax.plot(x[others[0]], x[others[1]], "*", color=col, ms=ms, mec="k", mew=0.4)
                        if col == "#d62728":
                            ax.annotate(k, (x[others[0]], x[others[1]]), color=col, fontsize=8, xytext=(4, 4), textcoords="offset points")
                else:
                    # line roughly within slab plane: draw if near slab
                    if min(abs(s[ia] - val), abs(e[ia] - val)) < max(2.0, 3 * half):
                        ax.plot([s[others[0]], e[others[0]]], [s[others[1]], e[others[1]]], color=col, lw=2 if col != "orange" else 1,
                                ls="-" if col != "orange" else "--")
                        if col == "#d62728":
                            mm = (s + e) / 2
                            ax.annotate(k, (mm[others[0]], mm[others[1]]), color=col, fontsize=8)
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        names = ["u", "v", "w"]
        ax.set_xlabel(names[others[0]]); ax.set_ylabel(names[others[1]])
        ax.set_title(sp.get("t", f"{a}={val} +-{half} (n={int(m.sum())})"), fontsize=9)
    for ax in axes.ravel()[len(specs):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", out)

if __name__ == "__main__":
    main()
