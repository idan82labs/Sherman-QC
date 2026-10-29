"""DEV prevalence of the four N1-47085000 failure classes, on the frozen v1 DEV OOF selections (zs|fixed|snap).
usage: .venv/bin/python -B -m research.zeroshot.n1_diag.dev_classes OUT.json"""
from __future__ import annotations
import json, sys
from collections import Counter
import numpy as np
from research.zeroshot.n1_diag import dev_data as D
from research.zeroshot.n1_diag.common import pm, is_match


def seg(c):
    return np.array([c["start_s"], c["end_s"]], float)


def joggle_ids(G):
    out = set()
    for i, (gi, a) in enumerate(G):
        for j, (gj, b) in enumerate(G):
            if i >= j:
                continue
            m = pm(a, b)
            if m["axis_angle_deg"] < 5 and 0.3 <= m["transverse_offset_mm"] <= 10 and m["shorter_span_overlap"] > 0.5:
                out |= {gi, gj}
    return out


def main(out):
    sp, V = D.load_v1()
    rows, cnt = [], Counter()
    base_lines = {s: [seg(c) for c, q in zip(v["C"], v["p_zs"]) if q >= 0.5] for s, v in V.items()}
    agg, recs = D.micro(sp, base_lines)
    print("v1 OOF reproduced:", agg["micro_f1"], agg["tp"], agg["fp"], agg["fn"], "ctrl", agg["controls"])
    recmap = {r["case_id"]: r for r in recs}
    n_gold = n_jog = n_jog_fn = 0
    jog_rows = []
    for s, v in V.items():
        r = recmap[s]
        if r.get("disposition") != "SCORED_COMPLETE_LABEL":
            continue
        G = D.gold_lines(sp[s])
        notes = D.gold_notes(sp[s])
        n_gold += len(G)
        matched = {m["gold_id"] for m in r.get("matches") or []}
        jog = joggle_ids(G) | {g for g, nt in notes.items() if "joggle" in nt.lower()}
        n_jog += len(jog)
        att = D.z2_attach(v["C"], v["z2"])
        att_of = {li: j for j, L in att.items() for li in L}
        for gid, g in G:
            isj = gid in jog
            if isj:
                # what the pool holds for a joggle line
                best = max([(is_match(pm(g, seg(c))), v["p_zs"][k], v["p_z3"][k], c["src"]) for k, c in enumerate(v["C"])],
                           key=lambda x: (x[0], x[1]), default=None)
                jog_rows.append({"scan": s, "gold": gid, "fn": gid not in matched, "born_match": bool(best and best[0]),
                                 "p_zs": None if not best else round(float(best[1]), 3),
                                 "p_z3": None if not best or best[2] != best[2] else round(float(best[2]), 3),
                                 "src": None if not best else best[3]})
            if gid in matched:
                continue
            fl = {}
            cand_match = [k for k, c in enumerate(v["C"]) if is_match(pm(g, seg(c)))]
            fl["born"] = bool(cand_match)
            fl["z3only_rejected"] = [k for k in cand_match if v["C"][k]["src"] == ["z3"] and v["p_zs"][k] < 0.5]
            fl["other_rejected"] = [k for k in cand_match if v["C"][k]["src"] != ["z3"] and v["p_zs"][k] < 0.5]
            fl["z3only_rejected_p_z3"] = [round(float(v["p_z3"][k]), 3) for k in fl["z3only_rejected"]]
            z2m = [li for li, ln in enumerate(v["z2"]) if is_match(pm(g, np.array([ln["start"], ln["end"]])))]
            fl["z2_match"] = bool(z2m)
            fl["z2_absorbed_by_short_z3"] = any(li in att_of and not is_match(pm(g, seg(v["C"][att_of[li]]))) for li in z2m)
            col = [k for k, c in enumerate(v["C"]) if c["has_z3"] and (lambda m: m["axis_angle_deg"] < 5 and m["transverse_offset_mm"] < 4)(pm(g, seg(c)))]
            covs = []
            L = np.linalg.norm(g[1] - g[0]); u = (g[1] - g[0]) / L
            iv = sorted([sorted([float((p - g[0]) @ u) for p in seg(v["C"][k])]) for k in col])
            tot, cur = 0.0, None
            for lo, hi in iv:
                lo, hi = max(lo, 0), min(hi, L)
                if hi <= lo:
                    continue
                if cur is None or lo > cur[1]:
                    if cur: tot += cur[1] - cur[0]
                    cur = [lo, hi]
                else:
                    cur[1] = max(cur[1], hi)
            if cur: tot += cur[1] - cur[0]
            fl["z3_split"] = len(col) >= 2 and not fl["born"] and tot / L >= 0.6
            fl["joggle"] = isj
            if fl["z3only_rejected"]:
                cls = "SEL_Z3ONLY_REJECT"
            elif fl["other_rejected"]:
                cls = "SEL_OTHER_REJECT"
            elif fl["z2_absorbed_by_short_z3"]:
                cls = "POOL_EXTENT"
            elif fl["z3_split"]:
                cls = "Z3_SPLIT"
            elif fl["z2_match"]:
                cls = "Z2_MATCH_OTHER"
            else:
                cls = "NOT_IN_POOL"
            cnt[cls] += 1
            if isj:
                n_jog_fn += 1
            rows.append({"scan": s, "family": sp[s]["family_id"], "gold": gid, "len": round(float(L), 1), "class": cls,
                         **{k: v_ for k, v_ in fl.items()}})
    res = {"baseline": {k: agg[k] for k in ("micro_f1", "tp", "fp", "fn")}, "n_gold": n_gold,
           "fn_classes": dict(cnt), "joggle_gold_lines": n_jog, "joggle_fn": n_jog_fn, "joggle_rows": jog_rows, "fn_rows": rows}
    print(json.dumps({k: v for k, v in res.items() if k not in ("fn_rows", "joggle_rows")}, indent=1))
    for r in rows:
        print(r["scan"], r["family"], r["gold"], r["len"], r["class"], "z3only_p_z3=", r["z3only_rejected_p_z3"], "jog" if r["joggle"] else "")
    print("joggle lines:")
    for r in jog_rows:
        print("  ", r)
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
