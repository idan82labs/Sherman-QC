"""Runner: per-fold apex extraction + gold measurement (see a3_foldlib.py)."""
from a3_foldlib import *  # noqa: F401,F403

out = {"folds": {}, "gold": {}}
fold_lines = {}
for name, fd in FOLDS.items():
    vals = np.arange(fd["rng"][0], fd["rng"][1] + 1e-9, STEP)
    st = []
    for v in vals:
        r = fit_station(fd, float(v))
        st.append((float(v), r))
    ok = [(v, r) for v, r in st if r["ok"] and 60 <= r["bend_deg"] <= 120]
    # contiguous extent(s) of ok stations
    okv = np.array([v for v, _ in ok])
    runs = []
    if len(okv):
        start = okv[0]
        prev = okv[0]
        for v in okv[1:]:
            if v - prev > STEP * 2.5:  # allow one missed station
                runs.append((start, prev))
                start = v
            prev = v
        runs.append((start, prev))
    A3 = np.array([to3(fd, v, r["apex"]) for v, r in ok]) if ok else np.zeros((0, 3))
    X3 = np.array([to3(fd, v, r["Xo"]) for v, r in ok]) if ok else np.zeros((0, 3))
    rec = {"axis": fd["axis"], "n_ok": len(ok), "n_stations": len(st), "runs_axis_mm": runs,
           "bend_deg_median": float(np.median([r["bend_deg"] for _, r in ok])) if ok else None,
           "Ro_median": float(np.median([r["Ro"] for _, r in ok if r["Ro"] is not None])) if ok else None,
           "t_median": float(np.median([r["t"] for _, r in ok])) if ok else None,
           "apex_minus_Xo_median": float(np.median([r["d_apex_Xo"] for _, r in ok])) if ok else None,
           "fail_reasons": {}}
    for v, r in st:
        if not r["ok"]:
            rec["fail_reasons"].setdefault(r.get("why"), []).append(v)
    if len(A3) >= 2:
        # apex line: fit through apex points, endpoints at the extreme stations of the longest run
        c, d, _ = fit_line(A3)
        d = d * np.sign(d @ VEC[fd["axis"]])
        t = (A3 - c) @ d
        lo, hi = t.min(), t.max()
        p0, p1 = c + lo * d, c + hi * d
        resid = np.linalg.norm((A3 - c) - np.outer(t, d), axis=1)
        rec.update(apex_line=[p0.tolist(), p1.tolist()], apex_line_len=float(hi - lo),
                   apex_line_resid_max=float(resid.max()), apex_line_resid_rms=float(np.sqrt((resid ** 2).mean())))
        fold_lines[name] = (p0, p1, d, c)
    out["folds"][name] = rec
    out["folds"][name]["stations"] = [
        {"v": v, "ok": bool(r["ok"]), "why": r.get("why"),
         **({"apex": r["apex"].tolist(), "Xo": r["Xo"].tolist(), "bend_deg": r["bend_deg"], "Ro": r["Ro"],
             "t": r["t"], "d_apex_Xo": r["d_apex_Xo"]} if r["ok"] else {})} for v, r in st]
    print(f"{name}: ok {len(ok)}/{len(st)} runs {runs} bend {rec['bend_deg_median']} Ro {rec['Ro_median']} t {rec['t_median']} "
          f"apexline len {rec.get('apex_line_len')} resid max {rec.get('apex_line_resid_max')}")

# --- gold vs folds
th = Thresholds()
GOLD_FOLD = {"M01": "LIP_+b (M01 side)", "M02": "LIP_-b (M02 side)", "M03": "TAB_+b (window b 64..82)",
             "M04": "TAB_+b (window b 64..82)", "M05": "TAB_-b (window b -90..-73)", "M06": "TAB_-b (window b -90..-73)",
             "M07": "WALL a~113 (M07 side)", "M08": "WALL a~113 (M07 side)"}
for gid, (s, e) in G.items():
    fname = GOLD_FOLD[gid]
    fd = FOLDS[fname]
    s2, e2 = (s - o) @ R.T, (e - o) @ R.T
    ax = AX[fd["axis"]]
    rec = {"assumed_target_fold": fname, "gold_axis_range": sorted([float(s2[ax]), float(e2[ax])])}
    # per-station offsets where fold ok and gold crosses
    offs = []
    for stn in out["folds"][fname]["stations"]:
        if not stn["ok"]:
            continue
        v = stn["v"]
        if not (min(s2[ax], e2[ax]) <= v <= max(s2[ax], e2[ax])):
            continue
        t = (v - s2[ax]) / (e2[ax] - s2[ax])
        gp = s2 + t * (e2 - s2)
        gp2 = np.array([gp[AX[fd["x"]]], gp[AX[fd["y"]]]])
        apex = np.array(stn["apex"])
        Xo = np.array(stn["Xo"])
        gvec = unit(apex - Xo) if np.linalg.norm(apex - Xo) > 1e-6 else unit(np.array(fd["fA"]) + np.array(fd["fB"]))
        dv = gp2 - apex
        offs.append({"v": v, "d_apex": float(np.linalg.norm(dv)), "along_bisector": float(dv @ gvec),
                     "across": float(dv @ np.array([-gvec[1], gvec[0]])), "d_Xo": float(np.linalg.norm(gp2 - Xo))})
    if offs:
        da = np.array([q["d_apex"] for q in offs])
        rec.update(n_stations=len(offs), d_apex_median=float(np.median(da)), d_apex_min=float(da.min()),
                   d_apex_max=float(da.max()),
                   along_bisector_median=float(np.median([q["along_bisector"] for q in offs])),
                   across_median=float(np.median([q["across"] for q in offs])),
                   d_Xo_median=float(np.median([q["d_Xo"] for q in offs])))
        # trend: d_apex at the first/last 20 % of crossings
        k = max(1, len(offs) // 5)
        rec["d_apex_first20"] = float(np.median(da[:k]))
        rec["d_apex_last20"] = float(np.median(da[-k:]))
    # coverage
    runs = out["folds"][fname]["runs_axis_mm"]
    if runs:
        f_lo = min(r[0] for r in runs) - STEP / 2
        f_hi = max(r[1] for r in runs) + STEP / 2
        g_lo, g_hi = rec["gold_axis_range"]
        inter = max(0.0, min(f_hi, g_hi) - max(f_lo, g_lo))
        rec.update(fold_axis_range=[f_lo, f_hi], coverage_of_fold=inter / (f_hi - f_lo),
                   gold_fraction_on_fold=inter / (g_hi - g_lo))
    if fname in fold_lines:
        p0, p1, d, c = fold_lines[fname]
        pm = pair_metrics(np.array([s, e]), np.array([p0, p1]))
        rec["frozen_matcher_vs_apex_line"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pm.items()}
        rec["frozen_matcher_match"] = bool(is_match(pm, th))
        # angle gold vs apex line
    out["gold"][gid] = rec
    print(gid, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k != "frozen_matcher_vs_apex_line"}),
          "| matcher:", rec.get("frozen_matcher_vs_apex_line"), rec.get("frozen_matcher_match"))

(OUT / "a3_folds.json").write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
