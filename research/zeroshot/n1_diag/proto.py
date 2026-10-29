"""DEV-only prototypes of the fixes proposed by the N1 47085000 diagnosis, applied on the frozen DEV OOF outputs of
zs_lane_v1 (zs|fixed|snap) and zs_lane_v2 (base|final). No lane code is changed; every rule is generic.

  EXT    : Z2-extent adoption. A selected line whose pooled candidate absorbed a Z2 line (pool rule 4 deg / 2.5 mm /
           50 %) that is >= r x longer and contains >= 90 % of the line takes the union of the two extents along the
           line's own axis; output lines then duplicated by the extended line (4 deg / 2.5 mm / 50 %) are dropped.
  Z3R    : Z3-strong rescue. A Z3-only candidate (no Z2 agreement) rejected by the pooled GBM is kept when the Z3-alone
           GBM (OOF fold model) gives p >= tau and it duplicates no output line. tau nested (chosen on 4 folds).
  TWIN   : twin rescue. A rejected Z3-only candidate with p_z3 >= 0.5 is kept when >= 2 selected candidates of the same
           scan share its signature (bend +-3 deg, log length +-0.2, log flank min/max +-0.3).
  JOG    : joggle-pair rescue. Two rejected Z3 candidates that are a joggle-branch pair (gap_med 0, bend >= 20 deg and
           within 15 deg of each other, parallel <= 3 deg, offset 0.5t..4t+4, >= 80 % overlap, length ratio >= 0.8,
           support >= 6 slices) are both kept.
usage: .venv/bin/python -B -m research.zeroshot.n1_diag.proto OUT.json"""
from __future__ import annotations
import json, math, sys
import numpy as np
from research.zeroshot.n1_diag import dev_data as D
from research.zeroshot.n1_diag.common import PROG, pm, is_match
from research.zeroshot.z3_slicing import slicer as SL
from research.zeroshot.zs_lane_v1 import scoring


def L_(a, b):
    return {"start": np.asarray(a, float), "end": np.asarray(b, float)}


def dup(x, y):
    ang, off, ov = SL._pair(x, y)
    return ang < 4.0 and off < 2.5 and ov > 0.5


def base_v1(v):
    return [{**L_(c["start_s"], c["end_s"]), "parts": [i]} for i, (c, q) in enumerate(zip(v["C"], v["p_zs"])) if q >= 0.5]


def base_v2(v2l):
    return [{**L_(r["start"], r["end"]), "parts": list(r["parts"])} for r in v2l]


def ext(lines, v, r=1.3, min_in=0.9, dedupe=True):
    att = D.z2_attach(v["C"], v["z2"])
    out, extended = [], []
    for ln in lines:
        a, b = ln["start"], ln["end"]
        Ll = np.linalg.norm(b - a)
        u = (b - a) / max(Ll, 1e-9)
        lo, hi = 0.0, Ll
        for i in ln["parts"]:
            for li in att.get(i, []):
                z = v["z2"][li]
                za, zb = np.asarray(z["start"], float), np.asarray(z["end"], float)
                Lz = np.linalg.norm(zb - za)
                s = sorted([(za - a) @ u, (zb - a) @ u])
                inside = max(0.0, min(hi, s[1]) - max(lo, s[0])) / max(hi - lo, 1e-9)
                if Lz >= r * (hi - lo) and inside >= min_in:
                    lo, hi = min(lo, s[0]), max(hi, s[1])
        n = {**ln, "start": a + lo * u, "end": a + hi * u, "ext": (lo < -1e-9 or hi > Ll + 1e-9)}
        out.append(n)
    if dedupe:
        drop = set()
        for k in sorted([k for k, o in enumerate(out) if o["ext"]], key=lambda k: -np.linalg.norm(out[k]["end"] - out[k]["start"])):
            if k in drop:
                continue
            Lk = np.linalg.norm(out[k]["end"] - out[k]["start"])
            for m, o in enumerate(out):
                if m != k and m not in drop and dup(out[k], o) and np.linalg.norm(o["end"] - o["start"]) <= Lk:
                    drop.add(m)
        out = [o for m, o in enumerate(out) if m not in drop]
    return out


def add_cands(lines, v, ks):
    out = list(lines)
    for k in ks:
        c = v["C"][k]
        n = {**L_(c["start_s"], c["end_s"]), "parts": [k], "added": True}
        if not any(dup(n, o) for o in out):
            out.append(n)
    return out


def selected_parts(lines):
    return {i for ln in lines for i in ln["parts"]}


def z3r(lines, v, tau):
    sel = selected_parts(lines)
    ks = [k for k, c in enumerate(v["C"]) if c["src"] == ["z3"] and k not in sel and v["p_z3"][k] >= tau]
    return add_cands(lines, v, ks)


def twin(lines, v, pmin=0.5):
    sel = sorted(selected_parts(lines))
    X = v["X"]
    sig = lambda k: (X[k, 3], X[k, 0], X[k, 5], X[k, 6])  # bend, log_len, log_fmin, log_fmax
    ks = []
    for k, c in enumerate(v["C"]):
        if c["src"] != ["z3"] or k in sel or not (v["p_z3"][k] >= pmin):
            continue
        b, ll, f0, f1 = sig(k)
        n = 0
        for j in sel:
            if not v["C"][j]["has_z3"]:
                continue
            bj, lj, g0, g1 = sig(j)
            if abs(b - bj) < 3 and abs(ll - lj) < 0.2 and abs(f0 - g0) < 0.3 and abs(f1 - g1) < 0.3 \
                    and not dup(L_(c["start_s"], c["end_s"]), L_(v["C"][j]["start_s"], v["C"][j]["end_s"])):
                n += 1
        if n >= 2:
            ks.append(k)
    return add_cands(lines, v, ks)


def jog_pairs(v):
    X, C, t = v["X"], v["C"], float(v["t"] or 1.5)
    idx = [k for k, c in enumerate(C) if c["has_z3"] and X[k, 13] < 1e-3 and X[k, 3] >= 20 and math.exp(X[k, 1]) - 1 >= 6]
    out = []
    for a_ in range(len(idx)):
        for b_ in range(a_ + 1, len(idx)):
            i, j = idx[a_], idx[b_]
            if abs(X[i, 3] - X[j, 3]) > 15:
                continue
            m = pm(np.array([C[i]["start"], C[i]["end"]]), np.array([C[j]["start"], C[j]["end"]]))
            r = min(m["gold_length_mm"], m["prediction_length_mm"]) / max(m["gold_length_mm"], m["prediction_length_mm"])
            if m["axis_angle_deg"] <= 3 and 0.5 * t <= m["transverse_offset_mm"] <= 4 * t + 4 and m["shorter_span_overlap"] >= 0.8 and r >= 0.8:
                out.append((i, j))
    return out


def jog(lines, v):
    sel = selected_parts(lines)
    ks = []
    for i, j in jog_pairs(v):
        if i not in sel and j not in sel:
            ks += [i, j]
    return add_cands(lines, v, ks)


def evaluate(sp, per, base_recs=None):
    recs = [D.score(sp[s], [(l["start"], l["end"]) for l in per.get(s, [])]) for s in sp]
    agg = scoring.aggregate(recs, ci=False)
    r = {k: agg[k] for k in ("micro_f1", "family_macro_f1", "tp", "fp", "fn")}
    r["ctrl_fp"] = agg["controls"]["fp_lines"]
    if base_recs is not None:
        r["vs_base"] = D.paired_boot(base_recs, recs, n=4000)
    return r, recs


def main(out):
    sp, V = D.load_v1()
    inv = {x["scan_id"]: x for x in json.loads((PROG / "INVENTORY.json").read_text())["scans"]}
    rolled = {s for s in sp if "ROLLED" in str((inv.get(s, {}).get("cad") or {}).get("compile_reason") or "")}
    v2l = json.loads((PROG / "v2/oof/OOF_base__final_lines.json").read_text())
    folds = {s: V[s]["fold"] for s in V}
    res = {"rolled_dev_scans": sorted(rolled), "arms": {}}
    # joggle-pair statistics on DEV (all pairs, label = both members match a gold line)
    npair = npos = 0
    for s, v in V.items():
        G = D.gold_lines(sp[s])
        for i, j in jog_pairs(v):
            npair += 1
            mi = any(is_match(pm(g, np.array([v["C"][i]["start_s"], v["C"][i]["end_s"]]))) for _, g in G)
            mj = any(is_match(pm(g, np.array([v["C"][j]["start_s"], v["C"][j]["end_s"]]))) for _, g in G)
            npos += int(mi and mj)
    res["jog_pairs_dev"] = {"pairs": npair, "both_true": npos}
    print("DEV joggle-branch pairs:", npair, "both members true:", npos, flush=True)
    for bname, mk in (("v1", lambda s: base_v1(V[s])), ("v2", lambda s: base_v2(v2l.get(s, [])))):
        base = {s: mk(s) for s in V}
        b, brecs = evaluate(sp, base)
        res["arms"][f"{bname}|base"] = b
        print(bname, "base", b, flush=True)

        def run(name, fn):
            per = {s: fn(base[s], V[s]) for s in V}
            r, recs = evaluate(sp, per, brecs)
            nb = {s: len(per[s]) - len(base[s]) for s in V}
            r["lines_added_rolled_or_ctrl"] = int(sum(max(nb[s], 0) for s in V if s in rolled or
                                                   next((x for x in brecs if x["case_id"] == s), {}).get("disposition") == "ZERO_BEND_CONTROL"))
            res["arms"][f"{bname}|{name}"] = r
            print(bname, name, r, flush=True)
            return per

        for rr in (1.2, 1.3, 1.5):
            run(f"EXT r{rr}", lambda l, v, rr=rr: ext(l, v, r=rr))
        for tau in (0.7, 0.8, 0.9, 0.95):
            run(f"Z3R tau{tau}", lambda l, v, tau=tau: z3r(l, v, tau))
        # nested tau for Z3R: per held-out fold, tau maximising micro-F1 on the other 4 folds
        grid = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)
        chosen = {}
        for f in sorted(set(folds.values())):
            tr = [s for s in V if folds[s] != f]
            best = None
            for tau in grid:
                recs = [D.score(sp[s], [(l["start"], l["end"]) for l in z3r(base[s], V[s], tau)]) for s in tr]
                f1 = scoring.aggregate(recs, ci=False)["micro_f1"]
                if best is None or f1 > best[0] + 1e-9:
                    best = (f1, tau)
            chosen[f] = best[1]
        res["arms"][f"{bname}|Z3R nested_taus"] = chosen
        run("Z3R nested", lambda l, v: z3r(l, v, chosen[v["fold"]]))
        run("TWIN", twin)
        run("JOG", jog)
        run("EXT r1.3 + Z3R tau0.9", lambda l, v: z3r(ext(l, v, r=1.3), v, 0.9))
        run("EXT r1.3 + Z3R nested", lambda l, v: z3r(ext(l, v, r=1.3), v, chosen[v["fold"]]))
        run("EXT r1.3 + TWIN", lambda l, v: twin(ext(l, v, r=1.3), v))
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])


def sweep_of(s):
    d = json.loads((D.V2_DUMP / f"{s}.json").read_text())
    return [None if x is None else float(x[0]) for x in d["evidence"]["sweep"]]


def z3r_sweep(lines, v, tau, sw, smin):
    sel = selected_parts(lines)
    ks = [k for k, c in enumerate(v["C"]) if c["src"] == ["z3"] and k not in sel and v["p_z3"][k] >= tau
          and sw[k] is not None and sw[k] >= smin]
    return add_cands(lines, v, ks)


def main_sweep(out):
    sp, V = D.load_v1()
    SW = {s: sweep_of(s) for s in V}
    v2l = json.loads((PROG / "v2/oof/OOF_base__final_lines.json").read_text())
    res = {}
    for bname, mk in (("v1", lambda s: base_v1(V[s])), ("v2", lambda s: base_v2(v2l.get(s, [])))):
        base = {s: mk(s) for s in V}
        b, brecs = evaluate(sp, base)
        for tau in (0.8, 0.9, 0.95):
            for smin in (0.25, 0.5, 0.75):
                per = {s: z3r_sweep(base[s], V[s], tau, SW[s], smin) for s in V}
                r, _ = evaluate(sp, per, brecs)
                res[f"{bname}|Z3R tau{tau} sweep>={smin}"] = r
                print(bname, tau, smin, r, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "sweep":
    pass
