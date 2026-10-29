"""Analyse the stability survey: per job, native vs 5 x s95 + thin040 decisions, value spread vs U, registration change,
G1 correctness of disagreements. usage: analyze.py MANIFEST OUT.json"""
import json, re, sys, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
R = Path(__file__).resolve().parents[1]
G1C = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/zsprod/g1c")
VARS = ["native"] + [f"s95_{s}" for s in (9501, 9502, 9503, 9504, 9505)] + ["thin040"]
man = json.load(open(sys.argv[1]))
truth = {}
for line in open(G1C / "rows/chars.jsonl"):
    c = json.loads(line); truth[(c["case_id"], c["characteristic_id"])] = c

def eng_log(d):
    p = d / "engine" / "engine.log"
    if not p.exists(): return {}
    t = p.read_text(); o = {}
    m = re.search(r"pose margin=([\d.]+) ambiguous=(\w+)", t); o["pose_margin"] = float(m.group(1)) if m else None
    o["ambiguous"] = m.group(2) == "True" if m else None
    m = re.search(r"mirrored best total=([\d.]+) holes_mismatch=(\d+) vs proper ([\d.]+)/(\d+)", t)
    o["mirror"] = [float(m.group(1)), float(m.group(3))] if m else None
    o["mirror_check_ran"] = "mirror_check" in t
    m = re.search(r"best datum score=([\d.]+) explained=([\d.]+) total=([\d.]+)", t); o["best_total"] = float(m.group(3)) if m else None
    m = re.search(r"thickness used ([\d.]+) mm", t); o["t_used"] = float(m.group(1)) if m else None
    o["wrong_way"] = "formed the wrong way" in t
    o["symmetric_rival"] = "symmetric rival chosen" in t
    o["mirror_converted"] = "mirror-symmetric CAD" in t
    o["free_datum"] = "free datum" in t
    m = re.search(r"points=(\d+) spacing=([\d.]+)", t); o["points"] = int(m.group(1)) if m else None; o["spacing"] = float(m.group(2)) if m else None
    return o

def load(d):
    p = d / "engine" / "result.json"
    rec = json.loads((d / "record.json").read_text()) if (d / "record.json").exists() else {}
    if not p.exists(): return None, rec
    r = json.loads(p.read_text()); out = {}
    for b in r.get("bends", []):
        for c in b.get("characteristics", []):
            tol = c.get("tolerance") or {}
            out[c["characteristic_id"]] = dict(label=b.get("label"), bend_state=b.get("state"), kind=c["kind"], side=c.get("side"),
                state=c["state"], decision=c["decision"], actual=c.get("actual"), expected=c.get("expected"), u=c.get("uncertainty"),
                tp=tol.get("plus"), tm=tol.get("minus"), reason=c.get("reason"), boundary=c.get("boundary_kind"),
                support=(c.get("evidence") or {}).get("support_points"), notes=(c.get("evidence") or {}).get("notes"),
                nst=(c.get("stations") or {}).get("count"))
    reg = (r.get("inspection_validity") or {}).get("registration") or {}
    meta = dict(validity=(r.get("inspection_validity") or {}).get("state"), reasons=(r.get("inspection_validity") or {}).get("reasons"),
                summary=(r.get("summary") or {}).get("state"), stage=(r.get("stage_decision") or {}).get("detail", "")[:40],
                T=reg.get("raw_to_frame"), pose_margin=reg.get("pose_margin"), pose_ambiguous=reg.get("pose_ambiguous"),
                wall=rec.get("wall_s"), rss=rec.get("peak_rss_bytes"), timed_out=rec.get("timed_out"), sha=rec.get("result_stripped_sha256"))
    return out, meta

def cls(a, b):
    if a == b: return None
    if a == "FAIL" and b == "PASS": return "FAIL->PASS"
    if a == "PASS" and b == "FAIL": return "PASS->FAIL"
    return f"{a}->{b}"

def pose_delta(T0, T1):
    if T0 is None or T1 is None: return None
    T0, T1 = np.array(T0), np.array(T1); rel = T1 @ np.linalg.inv(T0)
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(rel[:3, :3]) - 1) / 2))))
    return [round(ang, 4), round(float(np.linalg.norm(rel[:3, 3])), 4)]

REF = {e["key"]: e["ref_engine"] for e in man["entries"] if e.get("ref_engine")}
jobs = []
for j in man["unique_jobs"]:
    jd = R / "runs" / j["key"]
    runs = {v: load(jd / v) for v in VARS if (jd / v / "record.json").exists()}
    logs = {v: eng_log(jd / v) for v in runs}
    if "native" not in runs or runs["native"][0] is None:
        jobs.append(dict(key=j["key"], corpus=j["corpus"], id=j["id"], status="NO_NATIVE", n_runs=len(runs),
                         native_meta=runs.get("native", (None, None))[1])); continue
    nat, nmeta = runs["native"]
    ref = Path(REF[j["key"]]) / "result.json" if REF.get(j["key"]) else None
    det = None
    if ref and ref.exists():
        rr = json.loads(ref.read_text()); rn = json.loads((jd / "native/engine/result.json").read_text())
        strip = lambda o: {k: strip(v) for k, v in o.items() if k not in ("created_at", "duration_s", "peak_rss_bytes", "context", "scan_file_name")} if isinstance(o, dict) else ([strip(x) for x in o] if isinstance(o, list) else o)
        det = json.dumps(strip(rr.get("bends")), sort_keys=True) == json.dumps(strip(rn.get("bends")), sort_keys=True) and \
              rr.get("summary") == rn.get("summary")
    chars = []
    for cid, c in nat.items():
        row = dict(cid=cid, label=c["label"], kind=c["kind"], side=c["side"], native=[c["state"], c["decision"], c["actual"]], u=c["u"],
                   expected=c["expected"], tp=c["tp"], tm=c["tm"], support=c["support"], boundary=c["boundary"], nst=c["nst"], var={})
        vals = [c["actual"]] if c["actual"] is not None else []
        changes = []
        for v in VARS[1:]:
            if v not in runs: continue
            o = runs[v][0]
            if o is None: row["var"][v] = None; changes.append((v, "RUN_INVALID")); continue
            x = o.get(cid)
            if x is None: row["var"][v] = None; changes.append((v, f"{c['decision']}->ABSENT")); continue
            row["var"][v] = [x["state"], x["decision"], x["actual"], x["u"]]
            if x["actual"] is not None: vals.append(x["actual"])
            k = cls(c["decision"], x["decision"])
            if k: changes.append((v, k))
        row["changes"] = changes
        row["spread"] = (max(vals) - min(vals)) if len(vals) >= 2 else None
        row["spread_over_u"] = (row["spread"] / c["u"]) if row["spread"] is not None and c["u"] else None
        decs = [c["decision"]] + [row["var"][v][1] for v in VARS[1:] if row["var"].get(v)]
        row["decisions"] = dict(Counter(decs))
        # bimodality: values split into two clusters separated by > 3 * U (gap test on sorted values)
        if len(vals) >= 3 and c["u"]:
            sv = sorted(vals); gaps = np.diff(sv); gi = int(np.argmax(gaps))
            row["bimodal"] = bool(gaps[gi] > max(3 * c["u"], 0.05) and gi + 1 >= 1 and len(sv) - gi - 1 >= 1)
            row["modes"] = [float(np.median(sv[:gi + 1])), float(np.median(sv[gi + 1:]))] if row["bimodal"] else None
        if j["corpus"] == "G1":
            t = truth.get((j["id"], cid))
            if t:
                row["truth"] = dict(truth=t.get("truth"), in_tol=t.get("truth_in_tolerance"), scorable=t.get("scorable"),
                                    applicable=t.get("applicable"), must_be_unknown=t.get("must_be_unknown"))
        chars.append(row)
    variants = {}
    for v in VARS[1:]:
        if v not in runs: continue
        o, m = runs[v]
        variants[v] = dict(validity=m.get("validity"), summary=m.get("summary"), reasons=m.get("reasons"),
                           pose_delta=pose_delta(nmeta.get("T"), m.get("T")), pose_margin=m.get("pose_margin"),
                           pose_ambiguous=m.get("pose_ambiguous"), wall=m.get("wall"), timed_out=m.get("timed_out"), log=logs.get(v))
    jobs.append(dict(key=j["key"], corpus=j["corpus"], id=j["id"], status="OK", n_runs=len(runs), deterministic_vs_ref=det,
                     native_meta={k: v for k, v in nmeta.items() if k != "T"}, native_log=logs.get("native"), variants=variants, chars=chars))
json.dump({"jobs": jobs}, open(sys.argv[2], "w"), indent=1, default=float)
n = sum(1 for j in jobs if j["status"] == "OK")
print("jobs", len(jobs), "ok", n)
