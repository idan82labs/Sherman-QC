"""Feature-map PNG of a chirality map (research).

usage: python -m research.chirality.plot_map COMPILED_DIR MAP.json OUT.png [--stage S12] [--hyp 0] [--title T]

Three orthographic views of the stage's CAD sample cloud (grey) with the distinguishing features of one mirror
hypothesis: orange = material only the proper part has (the mirror image is open there), blue = material only
the mirror image has (the proper part is open there). Each feature is numbered; the side table lists type, size,
panel and probe counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from research.chirality.chirality_map import load

    ap = argparse.ArgumentParser()
    ap.add_argument("compiled_dir")
    ap.add_argument("map")
    ap.add_argument("out")
    ap.add_argument("--stage")
    ap.add_argument("--hyp", type=int, default=0)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    cdir = Path(a.compiled_dir)
    compiled = json.loads((cdir / "compiled_part.json").read_text())
    doc, pr = load(Path(a.map))
    sid = a.stage or doc["stages"][-1]["stage_model_id"]
    ordinal = next(i for i, s in enumerate(doc["stages"]) if s["stage_model_id"] == sid)
    srec = doc["stages"][ordinal]
    stage = next(s for s in compiled["stages"] if s["stage_model_id"] == sid)
    z = np.load(cdir / stage["sample_cloud"]["rel_path"])
    P = z["points"].astype(float)
    P = P[:: max(1, len(P) // 60000)]
    pids = [p["panel_id"] for p in compiled["panels"]]
    hrec = srec["hypotheses"][a.hyp] if srec["hypotheses"] else None
    sel = (pr["stage"] == ordinal) & (pr["hyp"] == a.hyp) & (pr["role"] < 2)
    pos, own, role, feat = pr["pos"][sel].astype(float), pr["owner"][sel], pr["role"][sel], pr["feature"][sel]
    st = np.empty_like(pos)
    for i in np.unique(own):
        T = np.asarray(stage["panel_transforms"][pids[int(i)]])
        m = own == i
        st[m] = pos[m] @ T[:3, :3].T + T[:3, 3]
    feats = [f for f in (hrec["features"] if hrec else []) if f["verifiable"] or f["n_deviating"] >= 200][:12]
    keep_ids = {f["feature"] for f in feats}
    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9])
    views = [((0, 1), "X", "Y", gs[0, 0]), ((0, 2), "X", "Z", gs[0, 1]), ((1, 2), "Y", "Z", gs[1, 0])]
    orange, blue = "#d9480f", "#1c7ed6"
    for (i, j), xl, yl, g in views:
        ax = fig.add_subplot(g)
        ax.scatter(P[:, i], P[:, j], s=0.2, c="#adb5bd", alpha=0.35, linewidths=0)
        for r, col in ((0, orange), (1, blue)):
            m = (role == r) & np.isin(feat, list(keep_ids))
            ax.scatter(st[m, i], st[m, j], s=7, c=col, linewidths=0)
        for f in feats:
            m = feat == f["feature"]
            if m.any():
                c = st[m].mean(axis=0)
                ax.annotate(str(f["feature"] + 1), (c[i], c[j]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                            weight="bold", color="#212529")
        ax.set_aspect("equal")
        ax.set_xlabel(f"{xl} (mm)")
        ax.set_ylabel(f"{yl} (mm)")
        ax.grid(alpha=0.2)
    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    lines = [f"Stage {sid}: {srec['class']}",
             f"E1 symmetry test: {'certified symmetric' if srec['e1_symmetry']['is_symmetric'] else 'not symmetric'}"
             f" (best mirror RMS {srec['e1_symmetry']['best_mirror_rms_mm']:.3f} mm)" if srec['e1_symmetry']['best_mirror_rms_mm'] is not None else "",
             ""]
    if hrec:
        lines += [f"Mirror hypothesis H{a.hyp} of {len(srec['hypotheses'])}: best proper fit of the mirror image,",
                  f"RMS {hrec['rms_mm']:.3f} mm, {100 * hrec['deviating_fraction']:.2f} % of samples > 1 mm off", ""]
        for f in feats:
            size = "x".join(f"{v:g}" for v in (f["size_mm"] or []))
            lines.append(f"{f['feature'] + 1}. {f['type']} {size} mm on {f['panel_id']}"
                         f"  (open in {f['opening_in'] or '-'}; max dev {f['max_deviation_mm']} mm)")
            lines.append(f"    probes: proper {f['probes']['proper_material']}, mirror {f['probes']['mirror_material']},"
                         f" ring {f['probes']['ring']}{'' if f['verifiable'] else '  (not verifiable)'}")
        more = len(hrec["features"]) - len(feats)
        if more > 0:
            lines.append(f"(+{more} smaller features not drawn)")
    lines += ["", "orange: material only on the proper part", "blue: material only on its mirror image"]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.suptitle(a.title or f"Chirality map {sid}", fontsize=14)
    fig.tight_layout()
    fig.savefig(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
