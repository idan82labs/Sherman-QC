"""zs_lane_v4 re-slice stability (runtime side; label-free).

Z3 slices the mesh with planes every ``step_mm`` along each axis family, starting half a step above the lowest
vertex. A real fold is a continuous feature: any slice phase or spacing finds it. A spurious track (a seam, a
crease of scanner noise, a corner of a cut-out) is often an accident of where the slices fell. v4 re-runs the frozen
Z3 (zs_lane_v3.z3_merge.detect, same mesh, same thickness, same families cap) once with a different slice step
(1.5 mm instead of 2.0 mm, so the levels fall elsewhere) and records, for every pooled candidate with Z3 features,
whether the re-slice reproduces it (zs_lane_v3.fixes.is_dup: 4 deg / 2.5 mm / 50 % on the raw lines) and how the
best reproduction's support compares.
"""
from __future__ import annotations

import numpy as np

RESLICE_STEP_MM = 1.5


def restability(mesh, t: float, cands: list[dict], max_families: int = 20) -> dict:
    from research.zeroshot.zs_lane_v2 import z3s
    from research.zeroshot.zs_lane_v3 import z3_merge
    from research.zeroshot.zs_lane_v3.fixes import pair_geom
    prm = z3s.Params(fallback_frac_only=True, max_families=int(max_families), step_mm=RESLICE_STEP_MM)
    res = z3_merge.detect(mesh, prm, thickness={"t_mm": float(t), "source": "given"})
    R = res.get("candidates") or []
    match, ratio, qrat = [], [], []
    for c in cands:
        if not c.get("has_z3"):
            match.append(None); ratio.append(None); qrat.append(None)
            continue
        best = None
        for q in R:
            ang, off, ov = pair_geom(c["start"], c["end"], q["start"], q["end"])
            if ang < 4.0 and off < 2.5 and ov > 0.5:
                Lq = float(np.linalg.norm(np.subtract(q["end"], q["start"])))
                if best is None or Lq > best[0]:
                    best = (Lq, q)
        if best is None:
            match.append(False); ratio.append(0.0); qrat.append(0.0)
        else:
            Lc = float(np.linalg.norm(np.subtract(c["end"], c["start"])))
            match.append(True)
            ratio.append(round(best[0] / max(Lc, 1e-9), 4))
            qrat.append(round(float(best[1].get("support_frac", 0.0)), 4))
    return {"step": RESLICE_STEP_MM, "n_rerun": len(R), "match": match, "len_ratio": ratio, "support_frac": qrat}
