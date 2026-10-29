"""e3-1.14 pose-wells candidate vs e3-1.12.0 (survey) and e3-1.13.0 (e3c sweep) on the 88-scan survey protocol.
usage: arms2.py MANIFEST BASE_RUNS E3C_RUNS CAND_RUNS OUT.json
Arms (per realization, decision per characteristic id):
  base      e3-1.12.0 survey result.json
  c113      e3-1.13.0 = root + net K=2 policy 1 (offline over the e3c sweep's primary_raw + resamples[0:2])
  c113root  e3-1.13.0 primary only (net off)
  pw        candidate as written (primary + K=2 net, policy 1, in engine)
  pwroot    candidate primary only (pose wells + alternative-pose withholding, net off)
Scorer only: G1 truth is read here, never by the engine.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VARS = ["native"] + [f"s95_{s}" for s in (9501, 9502, 9503, 9504, 9505)] + ["thin040"]
DEC = ("PASS", "FAIL")
G1C = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/zsprod/g1c")
man = json.load(open(sys.argv[1]))
BASE, E3C, CAND = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
truth = {}
for line in open(G1C / "rows/chars.jsonl"):
    c = json.loads(line)
    truth[(c["case_id"], c["characteristic_id"])] = c


def load_result(d):
    p = d / "engine" / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    out = {}
    for b in r.get("bends", []):
        for c in b.get("characteristics", []):
            out[c["characteristic_id"]] = dict(label=b.get("label"), kind=c["kind"], side=c.get("side"),
                                               decision=c["decision"], actual=c.get("actual"), u=c.get("uncertainty"),
                                               notes=list((c.get("evidence") or {}).get("notes") or []))
    for g in r.get("general_measurements", []) or []:
        c = g["characteristic"]
        out[c["characteristic_id"]] = dict(label="GEN", kind=c["kind"], side=None, decision=c["decision"],
                                           actual=c.get("actual"), u=c.get("uncertainty"),
                                           notes=list((c.get("evidence") or {}).get("notes") or []))
    return out, dict(validity=(r.get("inspection_validity") or {}).get("state"),
                     summary=(r.get("summary") or {}).get("state"))


def policy1(prim, res):
    out = {}
    for cid, (d, a, u) in prim.items():
        if d not in DEC:
            out[cid] = d
            continue
        others = [None if (r is None or r["validity"] != "VALID") else r["chars"].get(cid) for r in res]
        out[cid] = d if all(x is not None and x[0] == d for x in others) else "REVIEW"
    return out


def rc_of(d):
    rp = d / "record.json"
    rec = json.loads(rp.read_text()) if rp.exists() else {}
    rc = rec.get("resample_check")
    if rc is None and (d / "engine" / "resample_check.json").exists():
        rc = json.loads((d / "engine" / "resample_check.json").read_text())
    return rec, rc


def log_of(d):
    p = d / "engine" / "engine.log"
    return p.read_text().splitlines() if p.exists() else []


jobs = []
for j in man["unique_jobs"]:
    J = dict(key=j["key"], corpus=j["corpus"], id=j["id"], real={})
    for v in VARS:
        R = {"info": {}, "wall": {}, "rss": {}, "events": {}}
        b = load_result(BASE / j["key"] / v)
        if b:
            R["base"] = {k: x["decision"] for k, x in b[0].items()}
            R["info"] = b[0]
        # e3-1.13 (e3c sweep: K=3 policy 2 run; root = primary_raw, net = policy 1 over the first 2)
        rec, rc = rc_of(E3C / j["key"] / v)
        c = load_result(E3C / j["key"] / v)
        if rc and rc.get("primary_raw"):
            prim = {k: tuple(x) for k, x in rc["primary_raw"]["chars"].items()}
            R["c113root"] = {k: x[0] for k, x in prim.items()}
            R["c113"] = policy1(prim, (rc.get("resamples") or [])[:2])
            res = rc.get("resamples") or []
            R["wall"]["c113"] = (rc.get("primary_s") or 0) + sum((x or {}).get("seconds", 0) for x in res[:2])
            R["wall"]["c113root"] = rc.get("primary_s")
        elif c:
            R["c113root"] = R["c113"] = {k: x["decision"] for k, x in c[0].items()}
            R["wall"]["c113"] = R["wall"]["c113root"] = rec.get("wall_s")
        # candidate
        d = CAND / j["key"] / v
        rec, rc = rc_of(d)
        c = load_result(d)
        if c:
            R["pw"] = {k: x["decision"] for k, x in c[0].items()}
            R["pw_notes"] = {k: x["notes"] for k, x in c[0].items() if x["notes"]}
            R["pw_vals"] = {k: (x["actual"], x["u"]) for k, x in c[0].items()}
            R["wall"]["pw"] = rec.get("wall_s")
            R["rss"]["pw"] = rec.get("peak_rss_bytes")
            if rc and rc.get("primary_raw"):
                R["pwroot"] = {k: x[0] for k, x in rc["primary_raw"]["chars"].items()}
                R["wall"]["pwroot"] = rc.get("primary_s")
                R["pw_incomplete"] = rc.get("incomplete")
            else:
                R["pwroot"] = R["pw"]
                R["wall"]["pwroot"] = rec.get("wall_s")
            lg = log_of(d)
            R["events"] = {
                "wells_line": next((l for l in lg if "pose wells:" in l), None),
                "switched": any("pose well replaces" in l for l in lg),
                "alt_line": next((l for l in lg if "alternative poses:" in l), None),
                "rc_line": next((l for l in lg if "resample check:" in l), None),
            }
            m = [re.search(r"; ([\d.]+)s\)$", l or "") for l in lg if "pose wells:" in l]
            R["wells_s"] = float(m[0].group(1)) if m and m[0] else None
            R["validity"] = c[1]["validity"]
        J["real"][v] = R
    jobs.append(J)

ARMS = ("base", "c113root", "c113", "pwroot", "pw")


def arm_stats(arm, sel):
    st = Counter()
    detail = []
    for J in sel:
        nat = J["real"]["native"]
        if not nat.get("base") or not nat.get(arm):
            continue
        fp = f2p = chg = 0
        for cid, bdec in nat["base"].items():
            decs = [(J["real"][v].get(arm) or {}).get(cid) for v in VARS]
            nd = decs[0]
            st["chars"] += 1
            if "FAIL" in decs and "PASS" in decs:
                fp += 1
                i = nat["info"].get(cid, {})
                detail.append([J["id"], i.get("label"), i.get("kind"), i.get("side"), decs])
            if nd == "FAIL" and "PASS" in decs[1:]:
                f2p += 1
            if len({x for x in decs if x is not None}) > 1:
                chg += 1
            if bdec in DEC:
                st["base_decided"] += 1
                if nd not in DEC:
                    st["withheld_native"] += 1
                elif nd != bdec:
                    st["flipped_native"] += 1
            elif nd in DEC:
                st["gained_native"] += 1
            for v in VARS:
                bb, aa = J["real"][v].get("base"), J["real"][v].get(arm)
                if bb and aa and bb.get(cid) in DEC:
                    st["base_decided_all"] += 1
                    st["withheld_all"] += aa.get(cid) not in DEC
        st["chars_fail_pass"] += fp
        st["chars_fail_to_pass"] += f2p
        st["chars_any_change"] += chg
        st["scans"] += 1
        st["scans_fail_pass"] += fp > 0
        st["scans_fail_to_pass"] += f2p > 0
        st["scans_any_change"] += chg > 0
    st = dict(st)
    st["coverage_loss_native_pct"] = round(100 * st.get("withheld_native", 0) / max(st.get("base_decided", 1), 1), 3)
    st["coverage_loss_all_pct"] = round(100 * st.get("withheld_all", 0) / max(st.get("base_decided_all", 1), 1), 3)
    return st, detail


def g1_stats(arm):
    st = Counter()
    wrong = []
    for J in jobs:
        if J["corpus"] != "G1":
            continue
        for v in VARS:
            dd = J["real"][v].get(arm)
            if not dd:
                continue
            for cid, d in dd.items():
                t = truth.get((J["id"], cid))
                if not t or t.get("truth_in_tolerance") is None or not t.get("scorable", True):
                    continue
                corr = "PASS" if t["truth_in_tolerance"] else "FAIL"
                k = "native" if v == "native" else "perturbed"
                st[f"{k}_scored"] += 1
                if d in DEC:
                    st[f"{k}_decided"] += 1
                    if d == corr:
                        st[f"{k}_correct"] += 1
                    else:
                        st[f"{k}_false_pass" if d == "PASS" else f"{k}_false_fail"] += 1
                        wrong.append([J["id"], v, cid, d, corr])
    return dict(st), wrong


out = {"arms": {}}
corp = {"G1": [J for J in jobs if J["corpus"] == "G1"], "physical": [J for J in jobs if J["corpus"] != "G1"],
        "all": jobs}
for arm in ARMS:
    A = {}
    for cn, sel in corp.items():
        A[cn], det = arm_stats(arm, sel)
        if cn == "all":
            A["fail_pass_chars"] = det
    A["g1"], A["g1_wrong"] = g1_stats(arm)
    wa = [J["real"][v]["wall"].get(arm) for J in jobs for v in VARS if J["real"][v]["wall"].get(arm)]
    if wa:
        A["wall_s"] = dict(n=len(wa), p50=round(float(np.percentile(wa, 50)), 1),
                           p95=round(float(np.percentile(wa, 95)), 1), max=round(float(max(wa)), 1))
    out["arms"][arm] = A
# pose-well events
ev = Counter()
fired = []
for J in jobs:
    for v in VARS:
        R = J["real"][v]
        e = R.get("events") or {}
        if not e:
            continue
        k = "native" if v == "native" else "perturbed"
        ev[f"{k}_runs"] += 1
        w = e.get("wells_line") or ""
        ev[f"{k}_wells_ran"] += bool(w)
        m = re.search(r"(\d+) other well\(s\), (\d+) articulated", w)
        if m:
            ev[f"{k}_other_wells"] += int(m.group(1)) > 0
            ev[f"{k}_articulated_wells"] += int(m.group(2)) > 0
        if e.get("switched"):
            ev[f"{k}_switched"] += 1
            fired.append([J["corpus"], J["id"], v, "SWITCH", w[:300]])
        if e.get("alt_line"):
            ev[f"{k}_alt_check"] += 1
            wh = "withheld none" not in e["alt_line"]
            ev[f"{k}_alt_withheld"] += wh
            fired.append([J["corpus"], J["id"], v, "ALT", e["alt_line"][:400]])
        if R.get("pw_incomplete"):
            ev[f"{k}_net_incomplete"] += 1
ws = [J["real"][v].get("wells_s") for J in jobs for v in VARS if J["real"][v].get("wells_s") is not None]
pr = [J["real"][v]["wall"].get("pwroot") for J in jobs for v in VARS
      if J["real"][v].get("wells_s") is not None and J["real"][v]["wall"].get("pwroot")]
out["pose_well_events"] = dict(ev)
out["pose_well_fired"] = fired
if ws:
    share = [w / p for w, p in zip(ws, pr) if p]
    out["wells_cost"] = dict(n=len(ws), p50_s=round(float(np.median(ws)), 2), p95_s=round(float(np.percentile(ws, 95)), 2),
                             max_s=round(max(ws), 2), share_of_primary_p50=round(float(np.median(share)), 3),
                             share_of_primary_p95=round(float(np.percentile(share, 95)), 3))
# native changes of the candidate vs e3-1.13 (both arms)
chg = []
for J in jobs:
    nat = J["real"]["native"]
    for a, b in (("c113root", "pwroot"), ("c113", "pw")):
        if not nat.get(a) or not nat.get(b):
            continue
        for cid, d0 in nat[a].items():
            d1 = nat[b].get(cid)
            if d1 != d0:
                i = nat["info"].get(cid, {})
                chg.append(dict(arms=f"{a}->{b}", corpus=J["corpus"], id=J["id"], label=i.get("label"),
                                kind=i.get("kind"), side=i.get("side"), before=d0, after=d1,
                                val=(nat.get("pw_vals") or {}).get(cid), notes=(nat.get("pw_notes") or {}).get(cid)))
out["native_changes_vs_e3_1_13"] = chg
done = sum(1 for J in jobs for v in VARS if J["real"][v].get("pw"))
out["coverage"] = {"candidate_realizations": done, "jobs": len(jobs)}
json.dump(out, open(sys.argv[5], "w"), indent=1, default=str)
print("candidate realizations", done)
for arm in ARMS:
    A = out["arms"][arm]
    print(arm, {c: {k: A[c].get(k) for k in ("scans", "chars_fail_pass", "chars_fail_to_pass", "scans_fail_pass",
                                             "chars_any_change", "coverage_loss_native_pct", "coverage_loss_all_pct",
                                             "flipped_native")} for c in corp})
    print("   g1", A["g1"], "wall", A.get("wall_s"))
print("events", dict(ev))
print("wells cost", out.get("wells_cost"))
print("native changes vs 1.13", len(chg))
