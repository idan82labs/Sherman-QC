"""Overlay renders for visual checks: scan points (grey), gold lines (green = matched, orange = missed),
Z3 lines (blue = TP, red = FP), in three orthographic views along the scan's principal axes.
usage: python -m research.zeroshot.z3_slicing.overlay OOF.json OUT_DIR SID [SID ...]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.zeroshot.z3_slicing import custody, slicer, zscore  # noqa: E402


def render(sid: str, lines: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inv = custody.dev_inventory()
    path = custody.check_dev_scan(sid, inv)
    mesh = slicer.load_mesh(str(path))
    V = mesh.V
    rng = np.random.default_rng(0)
    P = V[rng.choice(len(V), min(40000, len(V)), replace=False)]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    rec = zscore.score_lines(sid, inv[sid], lines)
    gold = zscore.S.load_gold(zscore.case_spec(sid, inv[sid])["gold"], frame_records=None)["lines"]
    mg = {m["gold_id"] for m in rec.get("matches", [])}
    mp = {m["prediction_id"] for m in rec.get("matches", [])}
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j) in zip(axs, ((0, 1), (0, 2), (1, 2))):
        Q = (P - c) @ vt.T
        ax.scatter(Q[:, i], Q[:, j], s=0.2, c="0.75")
        for g in gold:
            e = (np.array([g["line_start"], g["line_end"]]) - c) @ vt.T
            ok = g["bend_id"] in mg
            ax.plot(e[:, i], e[:, j], color="green" if ok else "orange", lw=4 if not ok else 3, alpha=0.6)
        for k, ln in enumerate(lines):
            e = (np.array([ln["start"], ln["end"]]) - c) @ vt.T
            ok = f"Z{k + 1:03d}" in mp
            ax.plot(e[:, i], e[:, j], color="blue" if ok else "red", lw=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"PC{i + 1} vs PC{j + 1}")
    fig.suptitle(f"{sid} {inv[sid]['family']}  TP={rec.get('true_positive')} FP={rec.get('false_positive')} "
                 f"FN={rec.get('false_negative')}  (green gold matched, orange gold missed, blue Z3 TP, red Z3 FP)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    oof, outdir, sids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    O = json.loads(oof.read_text())
    for sid in sids:
        sel = [c for c in O["scored"][sid] if c.get("score", 0) >= O["thresholds"][sid]]
        render(sid, sel, outdir / f"{sid}.png")
        print("wrote", outdir / f"{sid}.png")


if __name__ == "__main__":
    main()
