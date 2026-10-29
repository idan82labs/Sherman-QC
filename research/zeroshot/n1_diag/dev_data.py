"""DEV-only data access for the N1 diagnosis prototypes: the frozen v1 DEV dump (integration/dev_dump), the v1 DEV OOF
probabilities (integration/oof_v1), the v1 fold models (OOF z3 probabilities), the v2 DEV dump + OOF probabilities /
lines. DEV gold of record via zs_lane_v1.scoring.case_specs('dev'). Never reads TEST-B or N1 gold."""
from __future__ import annotations
import json, math, pickle
from pathlib import Path
import numpy as np
from research.zeroshot.n1_diag.common import PROG, pm, is_match  # noqa: F401  (sets sys.path)
from research.zeroshot.zs_lane_v1 import scoring
from research.zeroshot.z3_slicing import slicer as SL

V2_DUMP = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/dump3")


def specs():
    return scoring.case_specs("dev")


def gold_lines(spec):
    G = scoring.S.load_gold(spec["gold"], frame_records=None, locus="apex")["lines"]
    return [(g["bend_id"], np.array([g["line_start"], g["line_end"]], float)) for g in G]


def gold_notes(spec):
    try:
        d = json.loads((PROG.parents[3] / spec["gold"]["path"]).read_text()) if not Path(spec["gold"]["path"]).is_absolute() \
            else json.loads(Path(spec["gold"]["path"]).read_text())
    except Exception:  # noqa: BLE001
        return {}
    return {b.get("bend_id"): (b.get("notes") or "") for b in d.get("bends", [])}


def load_v1():
    sp = specs()
    oof = pickle.loads((PROG / "integration/oof_v1/zs_snap_oof_probs.pkl").read_bytes())["P"]
    fm = pickle.loads((PROG / "integration/oof_v1/fold_models.pkl").read_bytes())
    folds = scoring.dev_folds()
    out = {}
    for s in sp:
        p = PROG / "integration/dev_dump" / f"{s}.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        C = r["candidates"]
        X = np.array([[np.nan if x is None else x for x in row] for row in r["X"]], float).reshape(len(C), -1) if C else np.zeros((0, 26))
        pz = np.asarray(oof.get(s, np.zeros(len(C))), float)
        p3 = np.full(len(C), np.nan)
        keep = [i for i, c in enumerate(C) if c["has_z3"]]
        if keep:
            p3[keep] = fm[f"z3|_s|{folds[s]}"].predict(X[keep][:, :21])
        out[s] = {"C": C, "X": X, "p_zs": pz, "p_z3": p3, "z2": r["z2_lines"], "t": (r["stages"].get("z3") or {}).get("t_mm"),
                  "fold": folds[s]}
    return sp, out


def z2_attach(C, z2):
    """Which Z2 line (index) the pooling attached to each Z3 candidate (pool_scan rule, raw geometry)."""
    att = {}
    z3i = [j for j, c in enumerate(C) if c["has_z3"]]
    for li, ln in enumerate(z2):
        best, bj = None, -1
        for j in z3i:
            ang, off, ov = SL._pair(ln, C[j])
            if ang < 4.0 and off < 2.5 and ov > 0.5:
                sc = ang / 4 + off / 2.5 - ov
                if best is None or sc < best:
                    best, bj = sc, j
        if bj >= 0:
            att.setdefault(bj, []).append(li)
    return att


def score(spec, lines):
    return scoring.score_scan(spec, [{"start": list(a), "end": list(b)} for a, b in lines], diagnose=False)


def micro(sp, per_scan):
    recs = [score(sp[s], per_scan.get(s, [])) for s in sp]
    return scoring.aggregate(recs, ci=False), recs


def paired_boot(recs_a, recs_b, n=10000, seed=20261025):
    fa, fb = scoring._fam(recs_a), scoring._fam(recs_b)
    fams = sorted(set(fa) | set(fb))
    A = np.array([fa.get(f, [0, 0, 0]) for f in fams], float)
    B = np.array([fb.get(f, [0, 0, 0]) for f in fams], float)
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, len(fams), len(fams))
        d[k] = scoring.prf(*B[idx].sum(0))[2] - scoring.prf(*A[idx].sum(0))[2]
    better = sum(1 for i in range(len(fams)) if scoring.prf(*B[i])[2] > scoring.prf(*A[i])[2] + 1e-9)
    worse = sum(1 for i in range(len(fams)) if scoring.prf(*B[i])[2] < scoring.prf(*A[i])[2] - 1e-9)
    return {"d_micro": round(float(scoring.prf(*B.sum(0))[2] - scoring.prf(*A.sum(0))[2]), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "families_better": better, "families_worse": worse}
