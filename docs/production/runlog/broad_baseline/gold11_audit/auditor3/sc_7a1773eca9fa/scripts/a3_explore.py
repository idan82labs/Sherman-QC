"""Auditor 3: explore normal clusters around each gold line (independent of earlier renders)."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
from a3_load import load  # noqa: E402

GOLD = "/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/3aa393ad34b7ace90d84985c.json"

v, n, f = load()
gold = json.load(open(GOLD))["bends"]


def axial_modes(normals, k=4, tol_deg=12.0, min_frac=0.03):
    """Greedy axial mode finding on unit normals (n and -n merged)."""
    rem = normals.copy()
    modes = []
    cos_tol = np.cos(np.radians(tol_deg))
    rng = np.random.default_rng(0)
    for _ in range(k):
        if len(rem) < 50:
            break
        cand = rem[rng.choice(len(rem), size=min(400, len(rem)), replace=False)]
        dots = np.abs(cand @ rem.T)
        score = (dots > cos_tol).sum(1)
        best = cand[int(np.argmax(score))]
        sel = np.abs(rem @ best) > cos_tol
        # refine: principal axis of selected normals (axial mean)
        T = (rem[sel].T @ rem[sel])
        w, U = np.linalg.eigh(T)
        m = U[:, -1]
        sel = np.abs(rem @ m) > cos_tol
        frac = sel.sum() / len(normals)
        if frac < min_frac:
            break
        pos = (rem[sel] @ m > 0).sum()
        neg = (rem[sel] @ m < 0).sum()
        modes.append((m, frac, int(pos), int(neg)))
        rem = rem[~sel]
    return modes


for b in gold:
    S = np.array(b["line_start"]); E = np.array(b["line_end"])
    L = np.linalg.norm(E - S); g = (E - S) / L
    rel = v - S
    t = rel @ g
    w = rel - np.outer(t, g)
    lat = np.linalg.norm(w, axis=1)
    sel = (t > -10) & (t < L + 10) & (lat < 40)
    print(f"\n== {b['bend_id']} L={L:.1f} pts={sel.sum()} dir={np.round(g,4)}")
    modes = axial_modes(n[sel], k=5)
    for m, frac, pos, neg in modes:
        ang_g = np.degrees(np.arccos(abs(m @ g)))
        print(f"  mode {np.round(m,4)} frac={frac:.3f} +{pos} -{neg}  angle_to_gold_dir={ang_g:.1f}")
    if len(modes) >= 2:
        m1, m2 = modes[0][0], modes[1][0]
        ax = np.cross(m1, m2); ax /= np.linalg.norm(ax)
        print("  axis from top-2 modes", np.round(ax, 4), "angle to gold",
              np.degrees(np.arccos(abs(ax @ g))).round(2), "dihedral normals", np.degrees(np.arccos(abs(m1 @ m2))).round(2))
