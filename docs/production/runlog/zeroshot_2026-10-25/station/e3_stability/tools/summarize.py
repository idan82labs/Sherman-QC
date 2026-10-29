"""Prevalence tables, feature association, G1 correctness, fix simulations. usage: summarize.py analysis.json manifest.json OUT.json"""
import json, sys, itertools, math
from collections import Counter, defaultdict
import numpy as np
A = json.load(open(sys.argv[1])); man = json.load(open(sys.argv[2]))
VARS = ["native"] + [f"s95_{s}" for s in (9501, 9502, 9503, 9504, 9505)] + ["thin040"]
by_key = {j["key"]: j for j in A["jobs"]}
DEC = ("PASS", "FAIL")
GB_EPS = 1e-9

def guard(dev, u, plus, minus):
    if dev + u <= plus + GB_EPS and dev - u >= -minus - GB_EPS: return "PASS"
    if dev - u > plus + GB_EPS or dev + u < -minus - GB_EPS: return "FAIL"
    return "REVIEW"

def realizations(c):
    """decision/value/U per realization (native first)."""
    out = [("native", c["native"][1], c["native"][2], c["u"])]
    for v in VARS[1:]:
        x = c["var"].get(v, "MISSING")
        if x == "MISSING": continue
        out.append((v, None, None, None) if x is None else (v, x[1], x[2], x[3]))
    return out

def fp_pair(decs):  # FAIL and PASS both present
    return "FAIL" in decs and "PASS" in decs

# ---------- per corpus prevalence (entries, so A1 and COH each count their own scans) ----------
corp = {}
rows_all = []
for corpus in ("G1", "A1", "COH"):
    ents = [e for e in man["entries"] if e["corpus"] == corpus]
    scans = []; nchar = nchg = nfp = nf2p = ndecided = 0; dirs = Counter(); reg_changed = 0; validity_changed = 0; summary_changed = 0
    for e in ents:
        j = by_key[e["key"]]
        if j["status"] != "OK":
            scans.append(dict(id=e["id"], status=j["status"], note="native run rc=%s (compile BLOCKED)" % (j.get("native_meta") or {}).get("rc"))); continue
        ch = [c for c in j["chars"]]
        chg = [c for c in ch if c["changes"]]
        f2p = [c for c in ch if any(k == "FAIL->PASS" for _, k in c["changes"])]
        fpp = [c for c in ch if fp_pair([r[1] for r in realizations(c)])]
        for c in chg:
            for k in set(k for _, k in c["changes"]): dirs[k] += 1
        nm = j["native_meta"]; vs = j["variants"]
        vchg = any(v["validity"] != nm["validity"] for v in vs.values())
        schg = any(v["summary"] != nm["summary"] for v in vs.values())
        maxpose = max(((v["pose_delta"] or [0, 0]) for v in vs.values()), key=lambda p: p[1], default=[0, 0])
        mirror_dec = [v["log"].get("mirror") is not None for v in vs.values()]
        mirror_flag = (j["native_log"].get("mirror") is not None) != any(mirror_dec) or not all(m == mirror_dec[0] for m in mirror_dec)
        refused = [v["validity"] for v in vs.values()]
        reg = vchg or maxpose[0] > 1.0 or maxpose[1] > 1.0
        reg_changed += reg; validity_changed += vchg; summary_changed += schg
        nchar += len(ch); nchg += len(chg); nfp += len(fpp); nf2p += len(f2p)
        ndecided += sum(1 for c in ch if c["native"][1] in DEC)
        scans.append(dict(id=e["id"], key=e["key"], validity=nm["validity"], summary=nm["summary"], chars=len(ch), changed=len(chg),
                          fail_pass=len(fpp), fail_to_pass=len(f2p), validity_changed=vchg, summary_changed=schg,
                          summaries=sorted(set([nm["summary"]] + [v["summary"] for v in vs.values()])),
                          max_pose_delta_deg_mm=maxpose, pose_margin_native=j["native_log"].get("pose_margin"),
                          mirror_check_ran_native=j["native_log"].get("mirror_check_ran"), mirror_check_differs=mirror_flag,
                          native_wall_s=nm.get("wall"), deterministic_vs_reference=j.get("deterministic_vs_ref"),
                          changed_chars=[dict(label=c["label"], kind=c["kind"], side=c["side"], native=c["native"][1],
                                              decisions=c["decisions"], dirs=sorted(set(k for _, k in c["changes"])),
                                              spread=c["spread"], u=c["u"], spread_over_u=c["spread_over_u"], bimodal=c.get("bimodal"),
                                              modes=c.get("modes")) for c in chg]))
    ok = [s for s in scans if "chars" in s]
    corp[corpus] = dict(n_entries=len(ents), n_scans_run=len(ok), scans=scans,
                        scans_any_change=sum(1 for s in ok if s["changed"]), scans_fail_pass=sum(1 for s in ok if s["fail_pass"]),
                        scans_fail_to_pass=sum(1 for s in ok if s["fail_to_pass"]),
                        chars=nchar, chars_decided_native=ndecided, chars_any_change=nchg, chars_fail_pass=nfp, chars_fail_to_pass=nf2p,
                        directions=dict(dirs), scans_validity_changed=validity_changed, scans_registration_changed=reg_changed,
                        scans_summary_changed=summary_changed)
# overall over unique jobs
uj = [j for j in A["jobs"] if j["status"] == "OK"]
allc = [(j, c) for j in uj for c in j["chars"]]
overall = dict(unique_scans=len(uj), scans_any_change=sum(1 for j in uj if any(c["changes"] for c in j["chars"])),
               scans_fail_pass=sum(1 for j in uj if any(fp_pair([r[1] for r in realizations(c)]) for c in j["chars"])),
               scans_fail_to_pass=sum(1 for j in uj if any(any(k == "FAIL->PASS" for _, k in c["changes"]) for c in j["chars"])),
               chars=len(allc), chars_any_change=sum(1 for _, c in allc if c["changes"]),
               chars_fail_pass=sum(1 for _, c in allc if fp_pair([r[1] for r in realizations(c)])),
               chars_fail_to_pass=sum(1 for _, c in allc if any(k == "FAIL->PASS" for _, k in c["changes"])),
               runs=sum(j["n_runs"] for j in A["jobs"]))
for k in ("scans_any_change", "scans_fail_pass", "scans_fail_to_pass"): overall[k + "_pct"] = round(100 * overall[k] / overall["unique_scans"], 2)
for k in ("chars_any_change", "chars_fail_pass", "chars_fail_to_pass"): overall[k + "_pct"] = round(100 * overall[k] / overall["chars"], 3)
phys = [j for j in uj if j["corpus"] != "G1"]; physc = [(j, c) for j in phys for c in j["chars"]]
overall["physical"] = dict(unique_scans=len(phys), scans_any_change=sum(1 for j in phys if any(c["changes"] for c in j["chars"])),
                           scans_fail_pass=sum(1 for j in phys if any(fp_pair([r[1] for r in realizations(c)]) for c in j["chars"])),
                           chars=len(physc), chars_any_change=sum(1 for _, c in physc if c["changes"]),
                           chars_fail_pass=sum(1 for _, c in physc if fp_pair([r[1] for r in realizations(c)])))

# ---------- G1 correctness on disagreements ----------
g1 = []
for j in uj:
    if j["corpus"] != "G1": continue
    for c in j["chars"]:
        if not c["changes"]: continue
        t = c.get("truth") or {}
        correct = None if t.get("in_tol") is None else ("PASS" if t["in_tol"] else "FAIL")
        g1.append(dict(case=j["id"], label=c["label"], kind=c["kind"], side=c["side"], truth=t.get("truth"), truth_in_tol=t.get("in_tol"),
                       correct_decision=correct, native=c["native"][1], native_value=c["native"][2],
                       variants={v: (x[1] if x else None) for v, x in c["var"].items()},
                       native_correct=(c["native"][1] == correct) if correct and c["native"][1] in DEC else None,
                       variant_wrong_decided=[v for v, x in c["var"].items() if x and x[1] in DEC and correct and x[1] != correct]))

# ---------- features ----------
def bend_info(j):
    info = defaultdict(dict)
    for c in j["chars"]:
        if c["kind"] == "FLANGE_LENGTH" and c["expected"] is not None: info[c["label"]][c["side"]] = c["expected"]
        if c["kind"] == "INCLUDED_ANGLE": info[c["label"]]["angle"] = c["expected"]
    return info
feat = []
for j in uj:
    bi = bend_info(j)
    for c in j["chars"]:
        b = bi.get(c["label"], {})
        fl = [b[s] for s in ("A", "B") if s in b]
        feat.append(dict(corpus=j["corpus"], key=j["key"], unstable=bool(c["changes"]), fp=fp_pair([r[1] for r in realizations(c)]),
                         min_flange=min(fl) if fl else None, angle=b.get("angle"), t=j["native_log"].get("t_used"),
                         pose_margin=j["native_log"].get("pose_margin"), kind=c["kind"], spread_over_u=c["spread_over_u"],
                         bimodal=bool(c.get("bimodal")), support=c["support"], wrong_way=j["native_log"].get("wrong_way"),
                         mirror_ran=j["native_log"].get("mirror_check_ran")))
def table(rows, keyf, buckets):
    out = []
    for name, f in buckets:
        rs = [r for r in rows if f(keyf(r))]
        out.append(dict(bucket=name, chars=len(rs), unstable=sum(r["unstable"] for r in rs), fail_pass=sum(r["fp"] for r in rs),
                        unstable_pct=round(100 * sum(r["unstable"] for r in rs) / max(len(rs), 1), 2)))
    return out
phys_feat = [r for r in feat if r["corpus"] != "G1"]
nn = lambda f: (lambda v: v is not None and f(v))
features = dict(
    min_flange_mm=table(phys_feat, lambda r: r["min_flange"], [("<12", nn(lambda v: v < 12)), ("12-20", nn(lambda v: 12 <= v < 20)), ("20-40", nn(lambda v: 20 <= v < 40)), (">=40", nn(lambda v: v >= 40)), ("none", lambda v: v is None)]),
    angle_deg=table(phys_feat, lambda r: r["angle"], [("<75", nn(lambda v: v < 75)), ("75-105", nn(lambda v: 75 <= v <= 105)), (">105", nn(lambda v: v > 105)), ("none", lambda v: v is None)]),
    thickness_mm=table(phys_feat, lambda r: r["t"], [("<1.2", nn(lambda v: v < 1.2)), ("1.2-2", nn(lambda v: 1.2 <= v < 2)), (">=2", nn(lambda v: v >= 2))]),
    pose_margin=table(phys_feat, lambda r: r["pose_margin"], [("<0.05", nn(lambda v: v < 0.05)), ("0.05-0.2", nn(lambda v: 0.05 <= v < 0.2)), (">=0.2", nn(lambda v: v >= 0.2))]),
    kind=table(phys_feat, lambda r: r["kind"], [(k, (lambda k: lambda v: v == k)(k)) for k in ("FLANGE_LENGTH", "INCLUDED_ANGLE", "BEND_LINE_POSITION", "BACKGAUGE_LENGTH")]),
    g1_min_flange_mm=table([r for r in feat if r["corpus"] == "G1"], lambda r: r["min_flange"], [("<12", nn(lambda v: v < 12)), ("12-20", nn(lambda v: 12 <= v < 20)), (">=20", nn(lambda v: v >= 20))]),
)
unst = [r for r in feat if r["unstable"]]
features["unstable_bimodal"] = sum(r["bimodal"] for r in unst); features["unstable_total"] = len(unst)
features["fail_pass_bimodal"] = sum(r["bimodal"] for r in feat if r["fp"]); features["fail_pass_total"] = sum(r["fp"] for r in feat)
features["fail_pass_min_flange"] = sorted(set(r["min_flange"] for r in feat if r["fp"]))

# ---------- fix simulations (leave-out: each realization in turn is the production input; its self-check uses K others) ----------
def sim(policy, K):
    """returns (#chars with FAIL/PASS disagreement across checked outputs, #decided withheld, #decided total, per-realization)"""
    fp_chars = 0; withheld = decided = 0; f2p_vs_native = 0; changed = 0
    for j, c in allc:
        R = realizations(c)
        outs = []
        for i, (v, d, x, u) in enumerate(R):
            others = [r for k, r in enumerate(R) if k != i]
            combos = list(itertools.combinations(others, K)) if K <= len(others) else [tuple(others)]
            for comb in combos:
                outs.append((i, policy(R[i], comb, c)))
        # coverage measured for the native realization averaged over combos
        nat = [o for i, o in outs if i == 0]
        if c["native"][1] in DEC:
            decided += 1; withheld += sum(1 for o in nat if o not in DEC) / len(nat)
        decs = set(o for _, o in outs)
        if "FAIL" in decs and "PASS" in decs: fp_chars += 1
        if len(decs) > 1: changed += 1
    return dict(K=K, chars_fail_pass_residual=fp_chars, chars_any_change_residual=changed, decided_native=decided,
                decided_withheld=round(withheld, 2), coverage_loss_pct=round(100 * withheld / max(decided, 1), 3))
def pol_none(me, comb, c): return me[1]
def pol_selfcheck(me, comb, c):
    d = me[1]
    if d not in DEC: return d
    return d if all(r[1] == d for r in comb) else "REVIEW"
def pol_honest_u(me, comb, c):
    d, x, u = me[1], me[2], me[3]
    if d not in DEC or x is None or u is None or c["tp"] is None: return d
    vals = [x] + [r[2] for r in comb if r[2] is not None]
    if any(r[2] is None for r in comb): return "REVIEW"  # a check run that cannot measure it: withhold
    rng = max(vals) - min(vals)
    return guard(x - c["expected"], u + rng, c["tp"], c["tm"])
def pol_honest_u_rss(me, comb, c):
    d, x, u = me[1], me[2], me[3]
    if d not in DEC or x is None or u is None or c["tp"] is None: return d
    if any(r[2] is None for r in comb): return "REVIEW"
    vals = np.array([x] + [r[2] for r in comb]); s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return guard(x - c["expected"], math.hypot(u, 2 * s), c["tp"], c["tm"])
fixes = {"none": [sim(pol_none, 1)], "a_selfcheck": [sim(pol_selfcheck, k) for k in (1, 2, 3, 5)],
         "c_honest_u_range": [sim(pol_honest_u, k) for k in (1, 2, 3, 5)], "c_honest_u_2sigma_rss": [sim(pol_honest_u_rss, k) for k in (2, 3, 5)]}
# short-panel gated self-check (only bends with a flange < 25 mm get the check; others keep the native decision)
def gated(th):
    def p(me, comb, c, _th=th):
        return pol_selfcheck(me, comb, c) if c.get("_short") else me[1]
    return p
for j in uj:
    bi = bend_info(j); short = any(min([b[s] for s in ("A", "B") if s in b] or [999]) < 25 for b in bi.values())
    for c in j["chars"]: c["_short"] = short
fixes["a_selfcheck_gated_scan_has_flange_lt25"] = [sim(gated(25), k) for k in (2, 3, 5)]
scans_with_short = sum(1 for j in uj if any(c.get("_short") for c in j["chars"]))
# run time: native wall per scan
walls = [j["native_meta"].get("wall") for j in uj if j["native_meta"].get("wall")]
cost = dict(native_wall_median_s=float(np.median(walls)), native_wall_p95_s=float(np.percentile(walls, 95)), native_wall_max_s=float(max(walls)),
            scans_with_flange_lt25=scans_with_short, scans=len(uj))
json.dump(dict(per_corpus=corp, overall=overall, g1_disagreements=g1, features=features, fixes=fixes, cost=cost), open(sys.argv[3], "w"), indent=1, default=float)
for k, v in corp.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "scans"})
print("overall", overall)
print("G1", json.dumps(g1, default=str)[:1500])
print(json.dumps(features, indent=0)[:3000])
for k, v in fixes.items(): print(k, v)
print(cost)
