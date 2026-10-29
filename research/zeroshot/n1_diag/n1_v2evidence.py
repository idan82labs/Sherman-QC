"""Post-hoc (N1 consumed) v2 evidence for the 47085000 failure candidates: station sweep (v2 sweep_min gate) and the
collinear-pair plate-continuity evidence that drives v2's join. Diagnosis only, no score.
usage: sherman_slot.sh .venv/bin/python -B -m research.zeroshot.n1_diag.n1_v2evidence DIAG_DIR OUT.json"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np


def main(dd, out):
    from research.zeroshot.zs_lane_v2 import evidence as E
    dd = Path(dd)
    R = pickle.loads((dd / "diag.pkl").read_bytes())
    z = np.load(dd / "mesh.npz")
    ctx = E.MeshCtx(z["V"], z["vn"], z["F"], float(R["z3"]["t_mm"]))
    C = R["z3"]["candidates"]
    res = {"sweep": {}, "pairs": []}
    for k in (14, 16, 18, 21, 25, 26):
        prof = E.station_profile(ctx, C[k])
        res["sweep"][k] = dict(zip(E.SWEEP_FEATS, [float(x) for x in (E.sweep_features(prof)[f] for f in E.SWEEP_FEATS)]))
    for i, j, g0, g1, ov in E.collinear_pairs(C, ctx.t):
        if {i, j} & {18, 21, 25}:
            ev = E.pair_evidence(ctx, C[i], C[j], g0, g1)
            res["pairs"].append({"i": i, "j": j, "g0": g0, "g1": g1, **{k: v for k, v in ev.items() if isinstance(v, (int, float, bool, str))}})
    print(json.dumps(res, indent=1, default=str))
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(*sys.argv[1:])
