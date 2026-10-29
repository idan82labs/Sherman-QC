"""analyze.py: prevalence + fix tables from runs/*/record.json.
Prefixes: M/N/W = diagnostic copy (E3M_FORCE: live decision, mirrored fit always measured); L* = live repo (e3-1.14.0);
F* = fix prototype. G1 'live' = the diagnostic run (identical decision; G1 runs are far from the budget, natives checked
characteristic-identical); A1 'live' = the L* run. Writes collected.json and summary.json."""
import json, glob, os, collections
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3mirror"
recs = {d.split("/")[-2]: json.load(open(d)) for d in glob.glob(f"{S}/runs/*/record.json")}
man = json.load(open(f"{S}/../e3stab/manifest.json")); E = {e["id"]: e for e in man["unique_jobs"]}
g1 = {}
for l in open(f"{S}/g1_chirality.txt"):
    i, j = l.split(" ", 1); g1[i] = json.loads(j)
g1f = {}
for l in open(f"{S}/g1_final_chirality.txt"):
    i, j = l.split(" ", 1); g1f[i] = json.loads(j)
a1 = json.load(open(f"{S}/a1_chirality.json"))
final_a1 = {}
for l in open(f"{S}/a1_chirality_all.jsonl"):
    if l.startswith("{"):
        d = json.loads(l); final_a1.setdefault(d["cdir"], {})[int(d["stage"][1:])] = d
final_chiral = {cd: v[max(v)]["distinguishable"] for cd, v in final_a1.items()}
DECIDED = {"ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE"}


def oc(r):
    if r is None: return None
    return dict(state=r.get("state"), summary=r.get("summary"), n_pass=r.get("n_pass") or 0, n_fail=r.get("n_fail") or 0,
                wall=r.get("wall_s"), rss=r.get("peak_rss_bytes"), reasons=r.get("reasons"))


def dist(case):
    if case.startswith("pc_"):
        return bool(a1[case]["distinguishable"] and final_chiral[E[case]["compiled_dir"]])
    # G1: the shown stage (the final one for the ORDER_SWAP cases without a defined stage) and the final part
    return bool((g1.get(case) or g1f[case])["distinguishable"] and g1f[case]["distinguishable"])


out = {"mirror": [], "swap": [], "native": []}
for jid, r in sorted(recs.items()):
    if jid[0] in "LF" or jid.endswith(".v1"):
        continue
    kind = jid[0]
    if kind == "M":
        case = jid.split("_", 1)[1]; corpus = "A1" if case.startswith("pc_") else "G1"
        live = recs.get("L" + jid) if corpus == "A1" else r
        fix = recs.get("F" + jid)
        if fix is None and live is not None and live.get("state") != "VALID":
            fix_o = dict(oc(live), note="not rerun: refused live (the prototype only adds refusals)")
        else:
            fix_o = oc(fix)
        out["mirror"].append(dict(id=jid, case=case, axis=jid[1], corpus=corpus, sku=E[case].get("sku"),
                                  scan=os.path.basename(E[case]["scan"]) if corpus == "A1" else None,
                                  live=oc(live), fix=fix_o, distinguishable=dist(case), e3m=r.get("e3m")))
    elif kind == "W":
        x, y = jid[2:].split("__as__"); corpus = "A1" if x.startswith("pc_") else "G1"
        live = recs.get("L" + jid) if corpus == "A1" else r
        fix = recs.get("F" + jid)
        fix_o = oc(fix) if fix is not None else (dict(oc(live), note="not rerun: refused live") if live else None)
        out["swap"].append(dict(id=jid, scan_case=x, declared_case=y, corpus=corpus, live=oc(live), fix=fix_o,
                                e3m=r.get("e3m")))
    elif kind == "N":
        case = jid[2:]; corpus = "A1" if case.startswith("pc_") else "G1"
        live = recs.get("L" + jid) if corpus == "A1" else r
        f = recs.get("F" + jid)
        extra = 0.0
        if f is not None:
            dg = {json.dumps(p.get("sub")): p for p in r.get("e3m") or []}
            for p in f.get("e3m") or []:
                q = dg.get(json.dumps(p.get("sub")), {})
                if p.get("ran") and not q.get("suspicious", False):
                    extra += float(p.get("mirror_s") or 0.0)
        out["native"].append(dict(id=jid, case=case, corpus=corpus, live=oc(live), fix=oc(f), diag=oc(r),
                                  chars_equal_fix_live=(f or {}).get("chars") == (live or {}).get("chars") if f and live else None,
                                  fix_added_s=round(extra, 2),
                                  fix_passes=[{k: p.get(k) for k in ("sub", "ran", "seed_only", "near_maps", "mirror_s", "why", "dt", "de", "hole_p", "hole_m")}
                                              for p in (f or {}).get("e3m") or []],
                                  e3m=r.get("e3m")))
json.dump(out, open(f"{S}/collected.json", "w"), indent=1, default=str)


def tally(rows, key):
    c = collections.Counter()
    for x in rows:
        o = x.get(key)
        if o is None: c["missing"] += 1; continue
        c["n"] += 1
        if o["state"] == "VALID":
            c["VALID"] += 1
            if o["n_pass"] > 0: c["VALID_with_PASS"] += 1; c["PASS_characteristics"] += o["n_pass"]
            if o["summary"] in DECIDED: c["decided_part_PASS"] += 1
        else:
            c[o["state"]] += 1
    return dict(sorted(c.items()))


summ = {}
for corpus in ("G1", "A1"):
    rows = [x for x in out["mirror"] if x["corpus"] == corpus]
    d = [x for x in rows if x["distinguishable"]]; s = [x for x in rows if not x["distinguishable"]]
    summ[f"mirror_{corpus}"] = {"distinguishable": {"live": tally(d, "live"), "fix": tally(d, "fix")},
                                "same_part_mirror": {"live": tally(s, "live"), "fix": tally(s, "fix")}}
    rows = [x for x in out["swap"] if x["corpus"] == corpus]
    summ[f"swap_{corpus}"] = {"live": tally(rows, "live"), "fix": tally(rows, "fix")}
    rows = [x for x in out["native"] if x["corpus"] == corpus]
    new_ref = [x["case"] for x in rows if x["fix"] and x["live"] and x["live"]["state"] == "VALID" and x["fix"]["state"] != "VALID"]
    lost = sum(max(0, x["live"]["n_pass"] - x["fix"]["n_pass"]) for x in rows if x["fix"] and x["live"])
    changed = [x["case"] for x in rows if x["chars_equal_fix_live"] is False]
    added = [x["fix_added_s"] for x in rows if x["fix"]]
    trig = [x["case"] for x in rows if any(p.get("ran") for p in x["fix_passes"]) and x["fix_added_s"] > 0]
    wl = [(x["fix"]["wall"], x["live"]["wall"]) for x in rows if x["fix"] and x["live"] and x["live"] is not x.get("diag")]
    summ[f"native_{corpus}"] = {"live": tally(rows, "live"), "fix": tally(rows, "fix"), "new_refusals": new_ref,
                                "pass_lost": lost, "characteristics_changed": changed,
                                "fix_added_seconds_per_scan": {"max": max(added or [0]), "mean": round(sum(added) / max(len(added), 1), 2),
                                                               "scans_paying": len(trig), "which": trig}}
    if corpus == "A1":
        live_w = [x["live"]["wall"] for x in rows if x["live"]]; fix_w = [x["fix"]["wall"] for x in rows if x["fix"]]
        summ[f"native_{corpus}"]["wall_live_max_s"] = max(live_w or [0]); summ[f"native_{corpus}"]["wall_fix_max_s"] = max(fix_w or [0])
        summ[f"native_{corpus}"]["wall_pairs"] = {x["case"]: [x["live"]["wall"] if x["live"] else None, x["fix"]["wall"] if x["fix"] else None] for x in rows}
json.dump(summ, open(f"{S}/summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
