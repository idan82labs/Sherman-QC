"""analyze.py -> ../results.json: synthetic suite (frozen pass conditions, FREEZE.json) and the seen real-data
regression, candidate e3-1.15.0-rc vs live e3-1.14.0 on identical inputs."""
import json, collections, statistics
from pathlib import Path
R = Path("/Users/idantal/ShermanQC/FullProject")
W = R / "docs/production/runlog/chirality/work"
M = R / "docs/production/runlog/zeroshot_2026-10-25/station/e3_mirror"
def rec(p):
    f = Path(p) / "record.json"
    return json.loads(f.read_text()) if f.exists() else None
def brief(r):
    if r is None: return None
    c = r.get("chirality") or {}
    return {"rc": r["rc"], "state": r.get("state"), "reasons": r.get("reasons"), "n_pass": r.get("n_pass"), "n_fail": r.get("n_fail"),
            "withheld": r.get("n_lr_withheld"), "decision": c.get("decision"), "reason": c.get("reason"), "stage": c.get("stage_model_id"),
            "class": c.get("class"), "pass_before": c.get("pass_before_check"), "wall": r["wall_s"], "rss": r.get("peak_rss_bytes"),
            "timed_out": r.get("timed_out"), "chars": [x[:6] for x in r.get("chars", [])]}
MIR = {"MX", "MY", "MZ", "OX"}; GEN = {"G", "T", "O"}
plan = json.load(open(W / "plan.json"))
rows = []
for it in plan:
    for v in it["variants"]:
        rows.append(dict(key=it["key"], corpus=it["corpus"], part=it["part"], role=it["role"], v=v, stage_class=it["stage_class"],
                         part_class=it["part_class"], cand=brief(rec(W / "runs" / it["key"] / v)),
                         live=brief(rec(W / "runs_live/syn" / it["key"] / v))))
done = [r for r in rows if r["cand"] and r["live"]]
def npass(x): return (x or {}).get("n_pass") or 0
chiral_m = [r for r in done if r["v"] in MIR and r["stage_class"] != "ACHIRAL"]
achiral_m = [r for r in done if r["v"] in MIR and r["stage_class"] == "ACHIRAL"]
gen = [r for r in done if r["v"] in GEN]
def lst(rs): return [f"{r['key']}/{r['v']}" for r in rs]
syn = {
    "runs_planned": len(rows), "runs_with_both_engines": len(done),
    "engine_failures": {"cand_rc_nonzero": lst([r for r in done if r["cand"]["rc"] != 0]), "live_rc_nonzero": lst([r for r in done if r["live"]["rc"] != 0]),
                        "timeouts": lst([r for r in done if r["cand"]["timed_out"] or r["live"]["timed_out"]])},
    "P1_chiral_mirrors_with_PASS": {"n": len(chiral_m), "cand": len([r for r in chiral_m if npass(r["cand"])]),
                                    "cand_pass_chars": sum(npass(r["cand"]) for r in chiral_m),
                                    "live": len([r for r in chiral_m if npass(r["live"])]),
                                    "live_pass_chars": sum(npass(r["live"]) for r in chiral_m),
                                    "cand_cases": [f"{r['key']}/{r['v']}:{npass(r['cand'])}:{r['cand']['decision']}" for r in chiral_m if npass(r["cand"])]},
    "P2a_genuine_refused_mirrored": {"n": len(gen), "cand": lst([r for r in gen if r["cand"]["state"] == "MIRRORED_PART_SUSPECTED"]),
                                     "live": lst([r for r in gen if r["live"]["state"] == "MIRRORED_PART_SUSPECTED"])},
    "P2b_achiral_mirrors_refused_mirrored": {"n": len(achiral_m), "cand": lst([r for r in achiral_m if r["cand"]["state"] == "MIRRORED_PART_SUSPECTED"]),
                                             "live": lst([r for r in achiral_m if r["live"]["state"] == "MIRRORED_PART_SUSPECTED"])},
    "achiral_mirrors": {"n": len(achiral_m), "cand_states": dict(collections.Counter(r["cand"]["state"] for r in achiral_m)),
                        "live_states": dict(collections.Counter(r["live"]["state"] for r in achiral_m)),
                        "chars_identical_cand_vs_live": sum(1 for r in achiral_m if r["cand"]["chars"] == r["live"]["chars"]),
                        "achiral_guard_fired": lst([r for r in achiral_m + gen if any("ACHIRAL_STAGE" in x for x in (r["cand"]["reasons"] or []))])},
    "P3_occluded_genuine": {"O_runs": sum(1 for r in done if r["v"] == "O"),
                            "O_valid_cand": sum(1 for r in done if r["v"] == "O" and r["cand"]["state"] == "VALID"),
                            "O_withheld": sum(1 for r in done if r["v"] == "O" and r["cand"]["state"] == "VALID" and r["cand"]["decision"] not in ("PROPER_CONFIRMED", "ACHIRAL")),
                            "O_confirmed_by_unoccluded_features": sum(1 for r in done if r["v"] == "O" and r["cand"]["state"] == "VALID" and r["cand"]["decision"] == "PROPER_CONFIRMED")},
}
cov = {}
for v in ("G", "T", "O"):
    rs = [r for r in done if r["v"] == v]
    cov[v] = {"n": len(rs),
              "live_valid": sum(1 for r in rs if r["live"]["state"] == "VALID"), "cand_valid": sum(1 for r in rs if r["cand"]["state"] == "VALID"),
              "live_pass_chars": sum(npass(r["live"]) for r in rs), "cand_pass_chars": sum(npass(r["cand"]) for r in rs),
              "cand_withheld_scans": lst([r for r in rs if r["cand"]["state"] == "VALID" and r["cand"]["decision"] not in ("PROPER_CONFIRMED", "ACHIRAL")]),
              "validity_changed": [f"{r['key']}/{v}: live {r['live']['state']} -> cand {r['cand']['state']}" for r in rs if r["live"]["state"] != r["cand"]["state"]],
              "not_valid_both": [f"{r['key']}/{v}: {r['live']['state']} {r['live']['reasons']} | cand {r['cand']['state']} {r['cand']['reasons']}"
                                 for r in rs if r["live"]["state"] != "VALID"],
              "non_pass_chars_identical": sum(1 for r in rs if [c for c in r["cand"]["chars"]] == [c for c in r["live"]["chars"]])}
syn["genuine_coverage"] = cov
syn["outcomes"] = {eng: {f"{v} {s} {d}": n for (v, s, d), n in sorted(collections.Counter(
    (r["v"], r[eng]["state"], r[eng]["decision"] if eng == "cand" else "") for r in done).items(), key=str)} for eng in ("cand", "live")}
syn["by_corpus_role_mirrors"] = {f"{k[0]} {k[1]}": dict(n=n) for k, n in collections.Counter((r["corpus"], r["role"]) for r in chiral_m).items()}
walls_c = [r["cand"]["wall"] for r in done]; walls_l = [r["live"]["wall"] for r in done]
syn["runtime"] = {"cand_p50": statistics.median(walls_c), "cand_p95": sorted(walls_c)[int(0.95 * len(walls_c)) - 1], "cand_max": max(walls_c),
                  "live_p50": statistics.median(walls_l), "live_p95": sorted(walls_l)[int(0.95 * len(walls_l)) - 1], "live_max": max(walls_l),
                  "rss_max_gib": max(max(r["cand"]["rss"] or 0, r["live"]["rss"] or 0) for r in done) / 2**30}
syn["rows"] = rows
# ------------------------------------------------------------------ real (seen)
groups = json.load(open(W / "real_groups.json"))
mres = json.load(open(M / "results.json"))
disting = {r["id"]: r.get("distinguishable") for r in mres["runs"]["mirror"]}
held_chiral = {r["case"]: r.get("final_part_chiral") for r in mres["heldout_confirmation"]["rows"]}
real = []
for name, g in groups.items():
    c, l = brief(rec(W / "runs_real" / name)), brief(rec(W / "runs_live/real" / name))
    if name.startswith("HM"): chir = held_chiral.get(name[4:])
    elif name.startswith("M"): chir = disting.get(name)
    else: chir = None
    real.append(dict(name=name, group=g, prior_distinguishable=chir, cand=c, live=l))
rd = [r for r in real if r["cand"] and r["live"]]
summ = {}
for g in sorted(set(groups.values())):
    rs = [r for r in rd if r["group"] == g]
    s = {"n": len(rs), "live_valid_with_pass": sum(1 for r in rs if r["live"]["state"] == "VALID" and npass(r["live"])),
         "cand_valid_with_pass": sum(1 for r in rs if r["cand"]["state"] == "VALID" and npass(r["cand"])),
         "live_pass_chars": sum(npass(r["live"]) for r in rs), "cand_pass_chars": sum(npass(r["cand"]) for r in rs),
         "cand_states": dict(collections.Counter(r["cand"]["state"] for r in rs)), "live_states": dict(collections.Counter(r["live"]["state"] for r in rs)),
         "cand_decisions": dict(collections.Counter(str(r["cand"]["decision"]) for r in rs)),
         "chars_identical": sum(1 for r in rs if r["cand"]["chars"] == r["live"]["chars"] and r["cand"]["state"] == r["live"]["state"]),
         "changed": [f"{r['name']}: live {r['live']['state']}/{npass(r['live'])}P -> cand {r['cand']['state']}/{npass(r['cand'])}P ({r['cand']['decision']}, {r['cand']['reason']})"
                     for r in rs if not (r["cand"]["chars"] == r["live"]["chars"] and r["cand"]["state"] == r["live"]["state"])]}
    if "mirror" in g:
        ch = [r for r in rs if r["prior_distinguishable"]]
        s["prior_chiral"] = {"n": len(ch), "live_valid_with_pass": sum(1 for r in ch if r["live"]["state"] == "VALID" and npass(r["live"])),
                             "cand_valid_with_pass": sum(1 for r in ch if r["cand"]["state"] == "VALID" and npass(r["cand"])),
                             "cand_with_pass_cases": [r["name"] for r in ch if npass(r["cand"])]}
        sp = [r for r in rs if r["prior_distinguishable"] is False]
        s["prior_same_part"] = {"n": len(sp), "cand_states": dict(collections.Counter(r["cand"]["state"] for r in sp)),
                                "live_states": dict(collections.Counter(r["live"]["state"] for r in sp))}
    summ[g] = s
out = {"schema": "chirality_results_v1", "candidate": "e3-1.15.0-rc (work/repo_c = live e3-1.14.0 + e3_1_15_0_rc_chirality.patch)",
       "live": "e3-1.14.0 (repo working tree)", "freeze": json.load(open(W.parent / "FREEZE.json"))["map_thresholds_sha256"],
       "synthetic": syn, "real_seen": {"note": "already-seen data: regression only, not generalization evidence", "summary": summ, "rows": real},
       "missing": {"synthetic": len(rows) - len(done), "real": len(real) - len(rd)}}
for name in ("g1_live", "g1_cand"):
    p = W / name / "scorecard.json"
    if p.exists():
        md = (W / name / "scorecard.md").read_text().splitlines()
        out[name + "_gates"] = [l for l in md if l.startswith("| G1.")]
(W.parent / "results.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in syn.items() if k not in ("rows", "outcomes", "genuine_coverage")}, indent=1, default=str)[:6000])
print(json.dumps(summ, indent=1)[:6000])
