"""GBM attribution for the rejected N1 candidates: one-feature-at-a-time interventional effect against DEV positive
rows (feature j replaced by each DEV positive's value, mean probability), plus the DEV-positive percentile of each value.
usage: .venv/bin/python -B -m research.zeroshot.n1_diag.attrib DIAG.pkl OUT.json"""
from __future__ import annotations
import json, pickle, sys
import numpy as np
from research.zeroshot.n1_diag import dev_data as D
from research.zeroshot.n1_diag.common import REPO, pm, is_match
from research.zeroshot.z3_slicing.selector import FEATS_BASE
from research.zeroshot.z3_slicing.pooled import EXTRA

NAMES = FEATS_BASE + EXTRA


def dev_rows():
    sp, V = D.load_v1()
    Xs, Ys = [], []
    for s, v in V.items():
        if not len(v["C"]):
            continue
        G = D.gold_lines(sp[s])
        y = np.array([any(is_match(pm(g, np.array([c["start_s"], c["end_s"]]))) for _, g in G) for c in v["C"]], float)
        Xs.append(v["X"]); Ys.append(y)
    return np.vstack(Xs), np.concatenate(Ys)


def main(diag, out):
    R = pickle.loads(open(diag, "rb").read())
    cfg = json.loads((REPO / "research/zeroshot/zs_lane_v1/config_v1.json").read_text())
    mz = pickle.loads((REPO / cfg["variants"]["zs"]["model"]).read_bytes())["model"]
    m3 = pickle.loads((REPO / cfg["variants"]["z3"]["model"]).read_bytes())["model"]
    X, Y = dev_rows()
    pos = X[Y == 1]
    rng = np.random.default_rng(0)
    ref = pos[rng.choice(len(pos), min(400, len(pos)), replace=False)]
    res = {}
    for name, idx in (("B4_piece2", 25), ("B14", 26), ("B25", 16), ("B26", 14), ("B1_z3", 18), ("B4_piece1", 21)):
        x = R["X"][idx]
        out_r = {"cand": idx, "p_zs": float(mz.predict(x[None])[0]), "p_z3": float(m3.predict(x[None, :21])[0]), "feats": {}}
        for model, nf, key in ((mz, 26, "zs"), (m3, 21, "z3")):
            base = float(model.predict(x[None, :nf])[0])
            eff = []
            for j in range(nf):
                Xr = np.repeat(x[None, :nf], len(ref), 0)
                Xr[:, j] = ref[:, j]
                eff.append(float(model.predict(Xr).mean()) - base)
            out_r[f"effect_{key}"] = {NAMES[j]: round(eff[j], 3) for j in np.argsort(eff)[::-1][:6]}
        for j in range(26):
            col = pos[:, j]
            col = col[np.isfinite(col)]
            out_r["feats"][NAMES[j]] = {"v": None if not np.isfinite(x[j]) else round(float(x[j]), 3),
                                        "dev_pos_pct": None if not np.isfinite(x[j]) or not len(col) else round(float((col < x[j]).mean() * 100), 1)}
        # counterfactual: what if Z2 had agreed (in_z2=1, n_src=2)?
        xc = x.copy(); xc[NAMES.index("in_z2")] = 1; xc[NAMES.index("n_src")] = 2
        out_r["p_zs_if_in_z2"] = float(mz.predict(xc[None])[0])
        res[name] = out_r
        print(name, "p_zs=%.3f p_z3=%.3f p_zs_if_in_z2=%.3f" % (out_r["p_zs"], out_r["p_z3"], out_r["p_zs_if_in_z2"]))
        print("   zs top effects:", out_r["effect_zs"])
        print("   z3 top effects:", out_r["effect_z3"])
        print("   vals:", {k: (v["v"], v["dev_pos_pct"]) for k, v in out_r["feats"].items()})
    res["_dev"] = {"rows": int(len(X)), "positives": int(Y.sum()), "ref_n": int(len(ref)),
                   "pos_rate_in_z2_0_has_z3": float(Y[(X[:, 21] == 1) & (X[:, 22] == 0)].mean()),
                   "pos_rate_in_z2_1": float(Y[X[:, 22] == 1].mean())}
    print(res["_dev"])
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
