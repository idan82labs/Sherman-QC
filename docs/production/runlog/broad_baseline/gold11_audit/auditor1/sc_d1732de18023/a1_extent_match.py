"""Fold extent at 0.5 mm steps, fitted outer-apex line per bend, coverage of the gold span, and the
frozen matcher's pair metrics gold vs the fitted apex line (5 deg / 4 mm / 60 % span)."""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, "/Users/idantal/ShermanQC/FullProject")
from a1_common import OUT, gold_lines, load_all, perp_basis, unit  # noqa: E402
from a1_xsec import RAD, SLAB, fit_station  # noqa: E402
from scripts.evaluate_f1_against_manual_gold import Thresholds, is_match, pair_metrics  # noqa: E402

STEP = 0.5
EXT = 30.0


def stations_for(xyz, nrm, a, b):
    d = unit(b - a)
    L = float(np.linalg.norm(b - a))
    u, w = perp_basis(d)
    rel = xyz - a
    t = rel @ d
    r2 = rel - np.outer(t, d)
    cyl = (np.linalg.norm(r2, axis=1) < RAD + 5) & (t > -EXT - 2) & (t < L + EXT + 2)
    idx = np.where(cyl)[0]
    tt = t[idx]
    Q = np.stack([rel[idx] @ u, rel[idx] @ w], 1)
    M = np.stack([nrm[idx] @ u, nrm[idx] @ w], 1)
    Mn = np.linalg.norm(M, axis=1)
    rows = []
    for s0 in np.arange(-EXT, L + EXT + 1e-6, STEP):
        sel = (np.abs(tt - s0) < SLAB) & (Mn > 0.5)
        q = Q[sel]
        mm = M[sel] / Mn[sel][:, None]
        keep = np.linalg.norm(q, axis=1) < RAD
        r = fit_station(q[keep], mm[keep])
        ok = (r is not None and r["Ao"] is not None and 60 < r["bend_angle"] < 120
              and min(r["th"]) > 0.8 and max(r["th"]) < 2.2 and min(r["flange_len"]) > 3.0)
        row = {"t": float(s0), "fold": bool(ok)}
        if ok:
            Ao = r["Ao"]
            row["apex_xyz"] = (a + d * s0 + u * Ao[0] + w * Ao[1]).tolist()
            row["d_apex"] = float(np.linalg.norm(Ao))
            row["flange_len"] = r["flange_len"]
        rows.append(row)
    return d, L, rows


def main():
    xyz, nrm, _ = load_all()
    G = gold_lines()
    th = Thresholds()
    out = {}
    for bid, (a, b) in G.items():
        d, L, rows = stations_for(xyz, nrm, a, b)
        ts = np.array([r["t"] for r in rows if r["fold"]])
        # fold extent = outermost fold stations, bridging gaps <= 10 mm
        gaps = [(float(ts[i]), float(ts[i + 1])) for i in np.where(np.diff(ts) > STEP * 1.5)[0]]
        t0, t1 = float(ts.min()), float(ts.max())
        big = [g for g in gaps if g[1] - g[0] > 10]
        A = np.array([r["apex_xyz"] for r in rows if r["fold"]])
        cen = A.mean(0)
        _, _, vt = np.linalg.svd(A - cen, full_matrices=False)
        ad = vt[0] if vt[0] @ d > 0 else -vt[0]
        resid = np.linalg.norm((A - cen) - np.outer((A - cen) @ ad, ad), axis=1)
        # apex line clipped to the fold extent (positions measured along gold t)
        def at_t(tq):
            # point on apex line whose projection on the gold axis equals tq
            s = (tq - (cen - a) @ d) / (ad @ d)
            return cen + ad * s
        apex_line = np.array([at_t(t0), at_t(t1)])
        gold = np.array([a, b])
        pm = pair_metrics(gold, apex_line)
        cov = max(0.0, min(L, t1) - max(0.0, t0)) / (t1 - t0)
        over = max(0.0, -t0) * 0 + max(0.0, 0.0 - t0)  # fold before gold start (mm)
        out[bid] = {
            "gold_length_mm": L,
            "fold_extent_t_mm": [t0, t1], "fold_length_mm": t1 - t0,
            "fold_gaps_mm": gaps, "fold_gaps_over_10mm": big,
            "fold_missing_before_gold_start_mm": max(0.0, 0.0 - t0),
            "fold_missing_after_gold_end_mm": max(0.0, t1 - L),
            "gold_overhang_before_fold_mm": max(0.0, t0 - 0.0),
            "gold_overhang_after_fold_mm": max(0.0, L - t1),
            "coverage_of_fold": cov,
            "apex_line_fit_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
            "apex_line_fit_max_mm": float(resid.max()),
            "apex_line_xyz": apex_line.tolist(),
            "angle_gold_vs_apex_deg": float(np.degrees(np.arccos(min(1.0, abs(ad @ d))))),
            "d_apex_stats_in_gold_span": {
                k: float(f(np.array([r["d_apex"] for r in rows if r["fold"] and 0 <= r["t"] <= L])))
                for k, f in (("min", np.min), ("median", np.median), ("max", np.max))},
            "matcher_gold_vs_fitted_apex_line": {**{k: float(v) for k, v in pm.items()},
                                                 "is_match": bool(is_match(pm, th))},
        }
        print(bid, json.dumps({k: v for k, v in out[bid].items() if k not in ("apex_line_xyz",)}, default=float))
    (OUT / "a1_extent_match.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
