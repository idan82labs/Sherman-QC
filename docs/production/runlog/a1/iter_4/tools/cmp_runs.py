"""Compare E3 result.json of two runs per case: validity, stage, per bend state / angle / flange decisions.
usage: cmp_runs.py DIR_A(pattern with {c}) DIR_B(pattern with {c}) case...  [--truth SCORECARD]"""
import json, sys
from pathlib import Path

args = sys.argv[1:]
truth = {}
if "--truth" in args:
    i = args.index("--truth"); sc = json.load(open(args[i + 1])); del args[i:i + 2]
    for r in sc["G3"]["rows"]:
        truth[(r["case_ref"], r["bend_uid"])] = r["truth"]
A, B = args[0], args[1]
cases = args[2:]


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    return json.load(open(p))


def summ(r):
    if r is None:
        return None, {}
    out = {}
    for b in r["bends"]:
        d = {"state": b["state"]}
        for ch in b.get("characteristics") or []:
            k = ch["kind"]
            side = ch.get("side") or ""
            key = k[:5] + (side[-1:] if side else "")
            dev = ch.get("deviation")
            d[key] = (None if dev is None else round(dev, 2), ch.get("decision"), None if ch.get("u") is None else round(ch["u"], 2))
        out[b["label"]] = (b.get("bend_uid"), d)
    v = r["inspection_validity"]["state"]
    st = (r.get("stage_decision") or {}).get("stage_model_id")
    return (v, st), out


for c in cases:
    ra = load(A.format(c=c)); rb = load(B.format(c=c))
    (ha, ba), (hb, bb) = summ(ra), summ(rb)
    print(f"=== {c}  A={ha}  B={hb}")
    labels = sorted(set(ba) | set(bb), key=lambda s: (len(s), s))
    for l in labels:
        ua, da = ba.get(l, (None, {}))
        ub, db = bb.get(l, (None, {}))
        uid = ua or ub
        t = truth.get((c, uid), "")
        def fmt(d):
            if not d: return "-"
            s = d["state"][:10]
            for k in sorted(k for k in d if k != "state"):
                dev, dec, u = d[k]
                s += f" {k}:{dev}/{(dec or '')[:4]}"
            return s
        flag = "  " if (da.get("state") == db.get("state")) else "**"
        print(f" {flag}{l:4s} truth={t[:10]:10s} A: {fmt(da):60s} | B: {fmt(db)}")
