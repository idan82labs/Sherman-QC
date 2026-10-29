"""DEV dump of 'Z2 visibility' features for every pooled candidate of the frozen v1 DEV dump (no labels read):
re-runs the frozen Z2 (it08) on the scan and, around each candidate's snapped line (radius 2t + 6 mm, stations
inside the span), measures the fraction of Z2 points that belong to a Z2 flange, the number of distinct flanges
there, and whether Z2's flange graph tested an edge between two of them (0 none, 1 rejected, 2 accepted).
usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.n1_diag.z2feat_dump OUTDIR [i/n] [--n1 DIAG_DIR]"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def feats_for(P, fop, edges, t, lines):
    tree = cKDTree(P)
    E = {}
    for e in edges:
        k = (min(e["a"], e["b"]), max(e["a"], e["b"]))
        E[k] = max(E.get(k, 0), 2 if e["ok"] else 1)
    out = []
    r = 2 * t + 6.0
    for a, b in lines:
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        u = (b - a) / max(L, 1e-9)
        idx = tree.query_ball_point(0.5 * (a + b), 0.5 * L + r)
        if not idx:
            out.append([0.0, 0, 0]); continue
        idx = np.asarray(idx)
        Q = P[idx] - a
        s = Q @ u
        dist = np.linalg.norm(Q - np.outer(s, u), axis=1)
        m = (s >= 0) & (s <= L) & (dist <= r)
        if m.sum() < 5:
            out.append([0.0, 0, 0]); continue
        f = fop[idx[m]]
        lab = f >= 0
        frac = float(lab.mean())
        fl, cn = np.unique(f[lab], return_counts=True)
        near = [int(x) for x, c in zip(fl, cn) if c >= 0.1 * max(lab.sum(), 1)]
        ed = 0
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                ed = max(ed, E.get((min(near[i], near[j]), max(near[i], near[j])), 0))
        out.append([frac, min(len(near), 4), ed])
    return out


def main(argv):
    outdir = Path(argv[0]); outdir.mkdir(parents=True, exist_ok=True)
    if "--n1" in argv:
        import pickle
        dd = Path(argv[argv.index("--n1") + 1])
        R = pickle.loads((dd / "diag.pkl").read_bytes())
        z = np.load(dd / "z2_scan.npz")
        lines = [(c["start_s"], c["end_s"]) for c in R["pool"]]
        F = feats_for(z["P"], z["flange_of_point"], R["z2_diag"]["edges"], R["z2_diag"]["t_mm"], lines)
        (outdir / f"{R['scan_id']}.json").write_text(json.dumps(F))
        return 0
    from research.zeroshot.n1_diag import dev_data as D
    from research.zeroshot.common import gold_guard
    from research.zeroshot.z2_self_recipe.run import run_scan
    shard = argv[1] if len(argv) > 1 else "0/1"
    i, n = (int(x) for x in shard.split("/"))
    sp = D.specs()
    for j, s in enumerate(sorted(sp)):
        if j % n != i or (outdir / f"{s}.json").is_file():
            continue
        dump = json.loads((D.PROG / "integration/dev_dump" / f"{s}.json").read_text())
        path = str(Path(sp[s]["scan_path"]).expanduser())
        gold_guard.require_runnable(path)
        try:
            payload, diag, scan = run_scan(Path(path))
            F = feats_for(scan.P, scan.flange_of_point, diag["edges"], scan.t,
                          [(c["start_s"], c["end_s"]) for c in dump["candidates"]])
        except Exception as e:  # noqa: BLE001
            F = None
            print(s, "Z2_ERROR", e, flush=True)
        (outdir / f"{s}.json").write_text(json.dumps(F))
        print(s, len(dump["candidates"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
