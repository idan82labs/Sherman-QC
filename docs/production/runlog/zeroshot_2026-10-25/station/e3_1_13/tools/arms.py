"""e3-1.13 candidate arms vs e3-1.12.0 (survey) on the survey protocol.
usage: arms.py MANIFEST BASE_RUNS CAND_RUNS OUT.json
Arms (per realization, decisions per characteristic id):
  base   e3-1.12.0 survey record (result.json)
  root   candidate primary (resample_check.primary_raw; or the record itself when no check ran)
  net    root + self-check (policy 1) over the first 2 resamples (offline, mirrors engine._resample_policy)
  optc   root + honest U (policy 2) over 3 resamples = the candidate result.json (checked against an offline recompute)
"""
import json, sys, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
VARS = ["native"] + [f"s95_{s}" for s in (9501, 9502, 9503, 9504, 9505)] + ["thin040"]
DEC = ("PASS", "FAIL")
G1C = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/zsprod/g1c")
EPS = 1e-9
man = json.load(open(sys.argv[1])); BASE = Path(sys.argv[2]); CAND = Path(sys.argv[3])
truth = {}
for line in open(G1C / "rows/chars.jsonl"):
    c = json.loads(line); truth[(c["case_id"], c["characteristic_id"])] = c

def r6(x): return round(float(x) + 0.0, 6)
def guard(dev, u, p, m):
    if dev + u <= p + EPS and dev - u >= -m - EPS: return "PASS"
    if dev - u > p + EPS or dev + u < -m - EPS: return "FAIL"
    return "REVIEW"

def load_result(d):
    p = d / "engine" / "result.json"
    if not p.exists(): return None
    r = json.loads(p.read_text())
    out = {}
    for b in r.get("bends", []):
        for c in b.get("characteristics", []):
            tol = c.get("tolerance") or {}
            out[c["characteristic_id"]] = dict(label=b.get("label"), kind=c["kind"], side=c.get("side"), decision=c["decision"],
                actual=c.get("actual"), u=c.get("uncertainty"), expected=c.get("expected"), tp=tol.get("plus"), tm=tol.get("minus"),
                notes=list((c.get("evidence") or {}).get("notes") or []), reason=c.get("reason"))
    for g in r.get("general_measurements", []) or []:
        c = g["characteristic"]; tol = c.get("tolerance") or {}
        out[c["characteristic_id"]] = dict(label="GEN", kind=c["kind"], side=None, decision=c["decision"], actual=c.get("actual"),
            u=c.get("uncertainty"), expected=c.get("expected"), tp=tol.get("plus"), tm=tol.get("minus"),
            notes=list((c.get("evidence") or {}).get("notes") or []), reason=c.get("reason"))
    meta = dict(validity=(r.get("inspection_validity") or {}).get("state"), summary=(r.get("summary") or {}).get("state"),
                engine=(r.get("engine") or {}).get("engine_version"))
    return out, meta

def policy1(prim, res):
    """prim: cid->(dec, act, u); res: list of reading dicts (or None). returns cid->decision"""
    inc = any(r is None or r["validity"] != "VALID" for r in res)
    out = {}
    for cid, (d, a, u) in prim.items():
        if d not in DEC: out[cid] = d; continue
        others = [None if (r is None or r["validity"] != "VALID") else r["chars"].get(cid) for r in res]
        out[cid] = d if all(x is not None and x[0] == d for x in others) else "REVIEW"
    return out, inc

def policy2(prim, res, info):
    out = {}; inc = any(r is None or r["validity"] != "VALID" for r in res)
    for cid, (d, a, u) in prim.items():
        if d not in DEC: out[cid] = d; continue
        others = [None if (r is None or r["validity"] != "VALID") else r["chars"].get(cid) for r in res]
        vals = [x[1] for x in others if x is not None and x[1] is not None]
        i = info[cid]
        dev = r6(r6(a) - r6(i["expected"]))
        if len(vals) < len(others): out[cid] = "REVIEW"; continue
        allv = np.asarray([a] + vals, dtype=float); sig = float(allv.std(ddof=1))
        out[cid] = guard(dev, max(r6(math.hypot(u, 2 * sig)), 1e-6), i["tp"], i["tm"])
    return out, inc

arms = ("base", "root", "net", "optc")
jobs = []
check_optc = Counter()
for j in man["unique_jobs"]:
    key = j["key"]
    J = dict(key=key, corpus=j["corpus"], id=j["id"], real={})
    for v in VARS:
        bd, cd = BASE / key / v, CAND / key / v
        b = load_result(bd); c = load_result(cd)
        crec = json.loads((cd / "record.json").read_text()) if (cd / "record.json").exists() else {}
        brec = json.loads((bd / "record.json").read_text()) if (bd / "record.json").exists() else {}
        R = dict(base=None, root=None, net=None, optc=None, wall={}, rss={}, notes={}, info={}, meta={})
        if b: R["base"] = {k: x["decision"] for k, x in b[0].items()}; R["meta"]["base"] = b[1]; R["info"] = {k: x for k, x in b[0].items()}
        R["wall"]["base"] = brec.get("wall_s"); R["rss"]["base"] = brec.get("peak_rss_bytes")
        R["vals_base"] = {k: (x["actual"], x["u"]) for k, x in b[0].items()} if b else {}
        if c:
            cinfo = c[0]; R["meta"]["cand"] = c[1]
            rc = crec.get("resample_check")
            if rc is None and (cd / "engine" / "resample_check.json").exists():
                rc = json.loads((cd / "engine" / "resample_check.json").read_text())
            R["notes"] = {k: x["notes"] for k, x in cinfo.items() if x["notes"]}
            if rc and rc.get("primary_raw"):
                prim = {k: tuple(x) for k, x in rc["primary_raw"]["chars"].items()}
                res = rc.get("resamples") or []
                R["root"] = {k: x[0] for k, x in prim.items()}
                R["vals_root"] = {k: (x[1], x[2]) for k, x in prim.items()}
                R["net"], inc2 = policy1(prim, res[:2])
                if len(res) < 3:  # primary not VALID: no resample ran; the written result is every arm's result
                    R["optc"] = {k: x["decision"] for k, x in cinfo.items()}
                    check_optc["primary_not_valid"] += 1
                if len(res) >= 3:
                    oc, inc3 = policy2(prim, res[:3], cinfo)
                    R["optc"] = {k: x["decision"] for k, x in cinfo.items()}
                    check_optc["agree" if oc == R["optc"] else "disagree"] += 1
                    if oc != R["optc"]:
                        check_optc["diff_" + key + "_" + v] = str({k: (oc[k], R["optc"][k]) for k in oc if oc[k] != R["optc"].get(k)})[:300]
                R["wall"].update(root=rc.get("primary_s"), optc=crec.get("wall_s"),
                                 net=(rc.get("primary_s") or 0) + sum((x or {}).get("seconds", 0) for x in res[:2]))
                R["rss"].update(root=crec.get("peak_rss_bytes"), optc=crec.get("peak_rss_bytes"), net=crec.get("peak_rss_bytes"))
                R["incomplete"] = inc2
                R["resample_validity"] = [None if x is None else x["validity"] for x in res]
            else:
                R["root"] = {k: x["decision"] for k, x in cinfo.items()}
                R["vals_root"] = {k: (x["actual"], x["u"]) for k, x in cinfo.items()}
                R["wall"]["root"] = crec.get("wall_s"); R["rss"]["root"] = crec.get("peak_rss_bytes")
            if not R["info"]: R["info"] = cinfo
        R["rc"] = dict(base=brec.get("rc"), cand=crec.get("rc"))
        J["real"][v] = R
    jobs.append(J)

def arm_stats(arm, sel):
    """FAIL<->PASS / FAIL->PASS over the 7 realizations; coverage vs base native; per-scan detail."""
    st = Counter(); per_scan = []; chars_detail = []
    for J in sel:
        nat = J["real"]["native"]
        if nat["base"] is None or nat.get(arm) is None:
            continue
        fp = f2p = chg = 0
        for cid, bdec in nat["base"].items():
            decs = [J["real"][v].get(arm, {}).get(cid) if J["real"][v].get(arm) else None for v in VARS]
            nd = decs[0]
            st["chars"] += 1
            if "FAIL" in decs and "PASS" in decs: fp += 1; chars_detail.append((J["id"], nat["info"][cid]["label"], nat["info"][cid]["kind"], nat["info"][cid]["side"], decs))
            if nd == "FAIL" and "PASS" in decs[1:]: f2p += 1
            if len(set(d for d in decs if d is not None)) > 1: chg += 1
            if bdec in DEC:
                st["base_decided"] += 1
                if nd not in DEC: st["withheld_native"] += 1
                elif nd != bdec: st["flipped_native"] += 1
            elif nd in DEC: st["gained_native"] += 1
            # coverage over all realizations: base decided in realization v, arm not decided in v
            for v in VARS:
                bb = J["real"][v].get("base"); aa = J["real"][v].get(arm)
                if not bb or not aa: continue
                if bb.get(cid) in DEC:
                    st["base_decided_all"] += 1
                    if aa.get(cid) not in DEC: st["withheld_all"] += 1
        st["chars_fail_pass"] += fp; st["chars_fail_to_pass"] += f2p; st["chars_any_change"] += chg
        st["scans"] += 1; st["scans_fail_pass"] += fp > 0; st["scans_fail_to_pass"] += f2p > 0; st["scans_any_change"] += chg > 0
    st = dict(st)
    st["coverage_loss_native_pct"] = round(100 * st.get("withheld_native", 0) / max(st.get("base_decided", 1), 1), 3)
    st["coverage_loss_all_pct"] = round(100 * st.get("withheld_all", 0) / max(st.get("base_decided_all", 1), 1), 3)
    return st, chars_detail

def g1_stats(arm):
    st = Counter(); wrong = []
    for J in jobs:
        if J["corpus"] != "G1": continue
        for v in VARS:
            R = J["real"][v]; dd = R.get(arm)
            if not dd: continue
            for cid, d in dd.items():
                t = truth.get((J["id"], cid))
                if not t or t.get("truth_in_tolerance") is None or not t.get("scorable", True): continue
                corr = "PASS" if t["truth_in_tolerance"] else "FAIL"
                k = "native" if v == "native" else "perturbed"
                st[f"{k}_scored"] += 1
                if d in DEC:
                    st[f"{k}_decided"] += 1
                    if d == corr: st[f"{k}_correct"] += 1
                    else:
                        st[f"{k}_wrong"] += 1
                        if d == "PASS": st[f"{k}_false_pass"] += 1
                        else: st[f"{k}_false_fail"] += 1
                        wrong.append((J["id"], v, cid, d, corr))
                else: st[f"{k}_withheld"] += 1
    return dict(st), wrong

out = {"arms": {}, "optc_offline_check": dict(check_optc)}
corpora = {"G1": [J for J in jobs if J["corpus"] == "G1"], "physical": [J for J in jobs if J["corpus"] != "G1"], "all": jobs}
for arm in arms:
    A = {}
    for cname, sel in corpora.items():
        A[cname], det = arm_stats(arm, sel)
        if cname == "all": A["fail_pass_chars"] = det
    A["g1"], A["g1_wrong"] = g1_stats(arm)
    walls = [J["real"]["native"]["wall"].get(arm) for J in jobs if J["real"]["native"]["wall"].get(arm)]
    wall_all = [J["real"][v]["wall"].get(arm) for J in jobs for v in VARS if J["real"][v]["wall"].get(arm)]
    rss = [J["real"][v]["rss"].get(arm) for J in jobs for v in VARS if J["real"][v]["rss"].get(arm)]
    if wall_all:
        A["wall_s"] = dict(n=len(wall_all), p50=float(np.percentile(wall_all, 50)), p95=float(np.percentile(wall_all, 95)), max=float(max(wall_all)),
                           native_p50=float(np.percentile(walls, 50)) if walls else None, native_max=float(max(walls)) if walls else None)
    if rss: A["peak_rss_gib_max"] = round(max(rss) / 2**30, 3)
    out["arms"][arm] = A
# native verdict changes vs base, per scan, with notes
chg = []
for J in jobs:
    nat = J["real"]["native"]
    if not nat["base"]: continue
    for arm in ("root", "net", "optc"):
        if not nat.get(arm): continue
        for cid, bd in nat["base"].items():
            ad = nat[arm].get(cid)
            if ad != bd:
                i = nat["info"].get(cid, {})
                vb = nat.get("vals_base", {}).get(cid); vr = nat.get("vals_root", {}).get(cid)
                chg.append(dict(arm=arm, corpus=J["corpus"], id=J["id"], key=J["key"], label=i.get("label"), kind=i.get("kind"), side=i.get("side"),
                                base=bd, cand=ad, base_val=vb, root_val=vr, notes=nat["notes"].get(cid)))
out["native_changes"] = chg
# realization-level verdict changes root vs base (all 7), counts by direction
dirs = defaultdict(Counter)
for J in jobs:
    for v in VARS:
        R = J["real"][v]
        if not R["base"] or not R.get("root"): continue
        for cid, bd in R["base"].items():
            rd = R["root"].get(cid)
            if rd != bd: dirs[J["corpus"]][f"{bd}->{rd}"] += 1
out["root_vs_base_all_realizations"] = {k: dict(v) for k, v in dirs.items()}
out["jobs_summary"] = [dict(key=J["key"], corpus=J["corpus"], id=J["id"], summaries={v: dict(base=(J["real"][v]["meta"].get("base") or {}).get("summary"), cand=(J["real"][v]["meta"].get("cand") or {}).get("summary")) for v in VARS}, rc={v: J["real"][v]["rc"] for v in VARS}, incomplete={v: J["real"][v].get("incomplete") for v in VARS}) for J in jobs]
bm = {}
for J in jobs:
    if J["key"] not in ("75611b1d377fec21", "f50be239406b9dce"): continue
    nat = J["real"]["native"]; rows = {}
    for cid, i in nat["info"].items():
        if i["label"] not in ("B5", "B6", "B7"): continue
        nm = f'{i["label"]} {i["kind"]}' + (f' {i["side"]}' if i["side"] else "")
        rows[nm] = {arm: {v: (J["real"][v].get(arm) or {}).get(cid) for v in VARS} for arm in arms}
        rows[nm]["vals_base"] = {v: (J["real"][v].get("vals_base") or {}).get(cid) for v in VARS}
        rows[nm]["vals_root"] = {v: (J["real"][v].get("vals_root") or {}).get(cid) for v in VARS}
        rows[nm]["tol"] = [i.get("expected"), i.get("tp"), i.get("tm")]
    bm[J["id"]] = rows
out["b_modes"] = bm
json.dump(out, open(sys.argv[4], "w"), indent=1, default=str)
for arm in arms:
    A = out["arms"][arm]
    print(arm, {c: {k: A[c].get(k) for k in ("scans", "chars", "chars_fail_pass", "chars_fail_to_pass", "scans_fail_pass", "chars_any_change", "coverage_loss_native_pct", "coverage_loss_all_pct", "gained_native", "flipped_native")} for c in corpora})
    print("   g1", A["g1"], "wall", A.get("wall_s"), "rss", A.get("peak_rss_gib_max"))
print("optc check", dict(check_optc))
print("native changes", len(chg))
