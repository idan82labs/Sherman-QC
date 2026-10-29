"""Out-of-fold permutation importance of the Z4 GBM (AUC drop on each held-out fold, averaged), plus LR
standardised coefficients. -> z4/<out>/feature_importance.json"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from research.zeroshot.z4_selector import data as D
from research.zeroshot.z4_selector.selector import load_table, make_model
from research.zeroshot.z4_selector.custody import OUT, dev_inventory


def main(out_name, *tags):
    tags = list(tags) or ["base"]
    inv = dev_inventory()
    folds = D.make_folds(inv)
    T = load_table(tags, inv)
    X, y, keys = T["X"], T["y"], T["keys"]
    fa = np.array([folds["scan_fold"][s] for s in T["sid"]])
    rng = np.random.default_rng(0)
    drops = np.zeros((folds["k"], len(keys)))
    for k in range(folds["k"]):
        tr = fa != k
        m = make_model("gbm").fit(X[tr], y[tr])
        Xt, yt = X[~tr], y[~tr]
        base = roc_auc_score(yt, m.predict(Xt))
        for j in range(len(keys)):
            Xp = Xt.copy()
            # permute within scan so per-scan structure is kept
            sids = T["sid"][~tr]
            for s in np.unique(sids):
                idx = np.flatnonzero(sids == s)
                Xp[idx, j] = Xp[rng.permutation(idx), j]
            drops[k, j] = base - roc_auc_score(yt, m.predict(Xp))
    lr = make_model("lr").fit(X, y)
    coef = lr.m.coef_[0][:len(keys)]
    imp = sorted(({"feature": kk, "oof_auc_drop": round(float(drops[:, j].mean()), 5),
                   "lr_std_coef_all_data": round(float(coef[j]), 3)} for j, kk in enumerate(keys)),
                 key=lambda r: -r["oof_auc_drop"])
    (OUT / out_name).mkdir(parents=True, exist_ok=True)
    (OUT / out_name / "feature_importance.json").write_text(json.dumps(imp, indent=1))
    for r in imp[:25]:
        print(r)


if __name__ == "__main__":
    main(*sys.argv[1:])
