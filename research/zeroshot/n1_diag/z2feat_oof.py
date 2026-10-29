"""DEV OOF test of the 'Z2 visibility' selector features (fix for the B14 class): same GBM, folds, labels (snapped
place) and fixed 0.5 threshold as zs_lane_v1 train.py; features = 26 v1 pooled features (+ 3 visibility features).
usage: .venv/bin/python -B -m research.zeroshot.n1_diag.z2feat_oof Z2FEAT_DIR OUT.json [N1_DIAG_DIR N1_FEAT_JSON]"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
from research.zeroshot.n1_diag import dev_data as D, proto as PR
from research.zeroshot.n1_diag.common import pm, is_match
from research.zeroshot.z3_slicing.selector import GBM


def main(fdir, out, n1dir=None, n1feat=None):
    sp, V = D.load_v1()
    ids = [s for s in sp if s in V]
    F = {s: json.loads((Path(fdir) / f"{s}.json").read_text()) for s in ids}
    Y = {}
    for s in ids:
        G = D.gold_lines(sp[s])
        Y[s] = np.array([any(is_match(pm(g, np.array([c["start_s"], c["end_s"]]))) for _, g in G) for c in V[s]["C"]], float)

    def mat(s, extra):
        X = V[s]["X"]
        if not extra:
            return X
        f = np.array(F[s], float).reshape(len(X), 3) if F[s] is not None and len(X) else np.full((len(X), 3), np.nan)
        return np.hstack([X, f]) if len(X) else np.zeros((0, 29))

    res, probs = {}, {}
    for name, extra in (("base26", False), ("vis29", True)):
        P = {}
        for f in sorted(set(V[s]["fold"] for s in ids)):
            tr = [s for s in ids if V[s]["fold"] != f and len(V[s]["C"])]
            te = [s for s in ids if V[s]["fold"] == f]
            mdl = GBM().fit(np.vstack([mat(s, extra) for s in tr]), np.concatenate([Y[s] for s in tr]))
            for s in te:
                P[s] = mdl.predict(mat(s, extra)) if len(V[s]["C"]) else np.zeros(0)
        probs[name] = P
        per = {s: [{**PR.L_(c["start_s"], c["end_s"]), "parts": [k]} for k, (c, q) in enumerate(zip(V[s]["C"], P[s])) if q >= 0.5] for s in ids}
        if name == "base26":
            r, brecs = PR.evaluate(sp, per)
            base_per = per
        else:
            r, _ = PR.evaluate(sp, per, brecs)
        res[name] = r
        print(name, r, flush=True)
        r2, _ = PR.evaluate(sp, {s: PR.ext(per[s], V[s], r=1.3) for s in ids}, brecs)
        res[name + "+EXT"] = r2
        print(name + "+EXT", r2, flush=True)
    # what the visibility model does to the B14 class on DEV
    cls = json.loads(Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/diag/dev_classes.json").read_text())
    rows = []
    for r in cls["fn_rows"]:
        if r["class"] == "SEL_Z3ONLY_REJECT":
            s = r["scan"]
            ks = r["z3only_rejected"]
            rows.append({"scan": s, "gold": r["gold"], "p_base": [round(float(probs["base26"][s][k]), 3) for k in ks],
                         "p_vis": [round(float(probs["vis29"][s][k]), 3) for k in ks], "vis": [F[s][k] for k in ks]})
    res["b14_class_rows"] = rows
    for x in rows:
        print(x)
    if n1dir:
        # post-hoc, consumed: final models fit on all DEV, applied to the N1 47085000 pooled candidates
        R = pickle.loads((Path(n1dir) / "diag.pkl").read_bytes())
        fn1 = np.array(json.loads(Path(n1feat).read_text()), float)
        tr = [s for s in ids if len(V[s]["C"])]
        m = GBM().fit(np.vstack([mat(s, True) for s in tr]), np.concatenate([Y[s] for s in tr]))
        p = m.predict(np.hstack([R["X"], fn1]))
        res["n1_posthoc_consumed"] = {str(k): {"p_vis": round(float(p[k]), 3), "p_zs": round(R["pool"][k]["p_zs"], 3),
                                               "vis": fn1[k].tolist()} for k in (14, 16, 18, 21, 25, 26)}
        res["n1_posthoc_consumed_changed"] = [k for k in range(len(p)) if (p[k] >= 0.5) != (R["pool"][k]["p_zs"] >= 0.5)]
        print("N1 post-hoc consumed:", res["n1_posthoc_consumed"], res["n1_posthoc_consumed_changed"])
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:])
