"""SPAN-type misses of the pinned v1 lane (NO_CAND:SPAN rows of results/DIAGNOSIS.json): is the nearest candidate too
short or too long, and would an alternative extent hypothesis match the gold line -- the attached Z2 line's raw
extent, the union with the Z2 extent (EXT), the union with collinear neighbours (join), a split piece?
Analysis only (no tuning). usage: .venv/bin/python -B docs/production/runlog/zs_v4/tools/span_analysis.py"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "docs/production/runlog/zs_v4/tools"))
from research.zeroshot.zs_lane_v4 import cache_io as CI, evalkit as EK, profiles as PR  # noqa: E402
import diagnose as D  # noqa: E402

def main():
    diag = json.loads((EK.V4RUN / "results/DIAGNOSIS.json").read_text())
    pool = {s["case_id"]: s for s in EK.load_pool()}
    rows = [r for r in diag["fn_rows"] if r["lane"] == "v1" and r["component"].startswith("NO_CAND:SPAN")]
    out = Counter(); det = []
    for r in rows:
        s = pool[r["case_id"]]; d = CI.load(r["case_id"])
        gold = {g["bend_id"]: g for g in EK.gold_lines(s)}
        g = gold[r["gold_id"]]
        rv = d["v4"]; C = rv["candidates"]
        best = None
        for i, c in enumerate(C):
            m = D.metrics(g, c["start_s"], c["end_s"])
            if m["axis_angle_deg"] <= 5 and m["transverse_offset_mm"] <= 4:
                sc = m["gold_span_overlap"] + m["prediction_span_overlap"]
                if best is None or sc > best[0]:
                    best = (sc, i, m)
        if best is None:
            out["no_near_in_v4"] += 1; continue
        _, i, m = best
        kind = "short" if m["gold_span_overlap"] < 0.6 and m["prediction_span_overlap"] >= 0.6 else (
            "long" if m["prediction_span_overlap"] < 0.6 and m["gold_span_overlap"] >= 0.6 else "both")
        # alternative hypotheses
        alts = {}
        prof = (rv.get("profiles") or {}).get(str(i))
        a = np.asarray(C[i]["start_s"], float); b = np.asarray(C[i]["end_s"], float)
        L = float(np.linalg.norm(b - a)); u = (b - a) / max(L, 1e-9)
        for z in rv.get("z2_lines") or []:
            if D.matches(g, z["start"], z["end"]):
                alts["z2_raw"] = True
        if prof and prof.get("z2_ext"):
            lo = min([0.0] + [e[0] for e in prof["z2_ext"]]); hi = max([prof["L"]] + [e[1] for e in prof["z2_ext"]])
            if D.matches(g, (a + lo * u).tolist(), (a + hi * u).tolist()):
                alts["ext_union"] = True
        # union with collinear neighbours (any candidates within 4 deg / 2.5 mm)
        pts = [a, b]
        for j, c in enumerate(C):
            if j == i: continue
            from research.zeroshot.zs_lane_v3.fixes import pair_geom
            ang, off, ov = pair_geom(C[i]["start_s"], C[i]["end_s"], c["start_s"], c["end_s"])
            if ang < 4 and off < 2.5:
                pts += [np.asarray(c["start_s"], float), np.asarray(c["end_s"], float)]
        ss = [float((p - a) @ u) for p in pts]
        if D.matches(g, (a + min(ss) * u).tolist(), (a + max(ss) * u).tolist()):
            alts["collinear_union"] = True
        # split piece: sub-interval of the candidate that matches (gold projected)
        g0 = float((np.asarray(g["line_start"]) - a) @ u); g1 = float((np.asarray(g["line_end"]) - a) @ u)
        lo, hi = max(0.0, min(g0, g1)), min(L, max(g0, g1))
        if hi > lo and D.matches(g, (a + lo * u).tolist(), (a + hi * u).tolist()):
            alts["clip_to_gold_within_cand"] = True
        key = kind + ":" + (",".join(sorted(alts)) or "none")
        out[key] += 1
        det.append({**r, "kind": kind, "alts": sorted(alts), "gov": round(m["gold_span_overlap"], 2),
                    "pov": round(m["prediction_span_overlap"], 2), "cand_src": C[i]["src"]})
    print(json.dumps(out.most_common(), indent=0))
    (EK.V4RUN / "results/SPAN_ANALYSIS.json").write_text(json.dumps({"counts": out, "rows": det}, indent=1))

main()
